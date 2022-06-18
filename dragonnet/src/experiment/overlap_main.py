from overlap_models import *
from src.process_result.overlap_ate import *
from generate_data import *
from original_dragonnet_code.experiment.ihdp_main import train_and_predict_dragons_trimmed
import original_dragonnet_code.semi_parametric_estimation.ate as ate_trimmed
import os
import time
import matplotlib.pyplot as plt
from original_dragonnet_code.semi_parametric_estimation.helpers import truncate_all_by_g
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN


def _split_output(yt_hat, t, y, y_scaler, x, index):
    yt_hat_t0_reshaped = yt_hat[:, 0].copy()
    yt_hat_t0_reshaped.shape = [yt_hat_t0_reshaped.shape[0], 1]
    yt_hat_t1_reshaped = yt_hat[:, 1].copy()
    yt_hat_t1_reshaped.shape = [yt_hat_t1_reshaped.shape[0], 1]

    q_t0 = y_scaler.inverse_transform(yt_hat_t0_reshaped)
    q_t1 = y_scaler.inverse_transform(yt_hat_t1_reshaped)
    g = yt_hat[:, 2].copy()

    if yt_hat.shape[1] == 4:
        eps = yt_hat[:, 3][0]
    else:
        eps = np.zeros_like(yt_hat[:, 2])

    y = y_scaler.inverse_transform(y.copy())

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': t, 'y': y, 'x': x, 'index': index, 'eps': eps}


# Same as original DragonNet model but without the propensity rescaling
def train_and_predict_dragons(t, y_unscaled, x, targeted_regularization=True, output_dir='',
                              knob_loss=dragonnet_loss_binarycross, ratio=1., dragon='', val_split=0.2, batch_size=64):
    verbose = 0
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)

    if dragon == 'tarnet':
        dragonnet = make_tarnet(x.shape[1], 0.01)

    elif dragon == 'dragonnet':
        dragonnet = make_dragonnet(x.shape[1], 0.01)

    metrics = [regression_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

    if targeted_regularization:
        loss = make_tarreg_loss(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss

    i = 0
    tf.random.set_random_seed(i)
    np.random.seed(i)
    train_index = list(range(t.shape[0]))

    x_train = x[train_index]
    y_train = y[train_index]
    t_train = t[train_index]

    yt_train = np.concatenate([y_train, t_train], 1)

    dragonnet.compile(
        optimizer=Adam(lr=1e-3),
        loss=loss, metrics=metrics)

    adam_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=1e-8, cooldown=0, min_lr=0)

    ]

    dragonnet.fit(x_train, yt_train, callbacks=adam_callbacks,
                  validation_split=val_split,
                  epochs=100,
                  batch_size=batch_size, verbose=verbose)

    sgd_callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=40, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0)
    ]

    sgd_lr = 1e-5
    momentum = 0.9
    dragonnet.compile(optimizer=SGD(lr=sgd_lr, momentum=momentum, nesterov=True), loss=loss,
                      metrics=metrics)
    dragonnet.fit(x_train, yt_train, callbacks=sgd_callbacks,
                  validation_split=val_split,
                  epochs=300,
                  batch_size=batch_size, verbose=verbose)

    yt_hat_train = dragonnet.predict(x_train)
    train_outputs = _split_output(yt_hat_train, t_train, y_train, y_scaler, x_train, train_index)
    K.clear_session()

    return train_outputs


def get_results(data_dir='synth_data', knob_loss=dragonnet_loss_binarycross, ratio=1.):
    models = ['dragonnet', 'trimmed', 'tarnet']
    sub_pop_sizes = [0.1, 0.25, 0.5]
    propensities = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.001]
    sample_sizes = [100, 200, 500, 1000, 2000]
    replications = 200

    start_time = time.time()
    subtimes = start_time
    for model in models:
        for sub_pop in sub_pop_sizes:
            for samples in sample_sizes:
                batch_size = 64 if samples < 500 else (128 if samples < 2000 else 512)

                for prop in propensities:

                    print(f"\tModel: {model}\t Pop: {sub_pop}\t Size: {samples}\t Propensity: {prop}")
                    data_path = os.path.join(data_dir, f"sub_pop_{sub_pop}/samples_{samples}/propens_{prop}")

                    ates = []
                    abs_errs = []

                    for i in range(replications):
                        x, t, y = read_data(os.path.join(data_path, f"{i}.csv"))
                        if model == 'trimmed':
                            data = train_and_predict_dragons_trimmed(t, y, x,
                                                             targeted_regularization=True,
                                                             knob_loss=knob_loss, ratio=ratio, dragon=model,
                                                             val_split=0.2,
                                                             batch_size=batch_size)
                        elif model == 'tarnet':
                            data = train_and_predict_dragons(t, y, x,
                                                             targeted_regularization=False,
                                                             knob_loss=knob_loss, ratio=ratio, dragon=model,
                                                             val_split=0.2,
                                                             batch_size=batch_size)
                        else:
                            data = train_and_predict_dragons(t, y, x,
                                                                     targeted_regularization=True,
                                                                     knob_loss=knob_loss, ratio=ratio, dragon=model,
                                                                     val_split=0.2,
                                                                     batch_size=batch_size)

                        q_t0, q_t1, g, t, y_dragon = data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), \
                                                         data['g'].reshape(-1, 1), data['t'].reshape(-1, 1), \
                                                         data['y'].reshape(-1, 1)
                        if model == 'trimmed':
                            psi_tmle, _, _, _, _, _ = ate_trimmed.psi_tmle_cont_outcome(q_t0, q_t1, g, t, y_dragon)
                        elif model == 'tarnet':
                            psi_tmle = psi_naive(q_t0, q_t1, g, t, y_dragon)
                        else:
                            psi_tmle, _, _, _, _, _ = psi_tmle_cont_outcome(q_t0, q_t1, g, t, y_dragon)
                        abs_err = abs(1.0 - psi_tmle)
                        ates.append(psi_tmle)
                        abs_errs.append(abs_err)

                    # print("--------- NOT STORING RESULTS ---------")
                    results_path = f"synth_result/{model}/sub_pop_{sub_pop}/samples_{samples}/propens_{prop}"
                    os.makedirs(results_path, exist_ok=True)
                    pd.DataFrame(np.array(ates)).to_csv(os.path.join(results_path, f"ates.csv"), header=None, index=None)
                    pd.DataFrame(np.array(abs_errs)).to_csv(os.path.join(results_path, f"abs_errs.csv"), header=None, index=None)

                    print(f"\t\tTime taken (s): {int(time.time() - subtimes)}")
                    print(f"\t\tTotal time elapsed (min): {int((time.time() - start_time)/60)}\n")
                    subtimes = time.time()


def get_ihdp_results(data_dir='ihdp_data', knob_loss=dragonnet_loss_binarycross, ratio=1., batch_size=64):

    models = ['dragonnet', 'trimmed', 'tarnet']
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    replications = 200

    print("Starting ihdp estimations!")
    start_time = time.time()
    subtimes = start_time
    for model in models:
        for q in qs:
            print(f"\tModel: {model}\t q: {q}")
            data_path = os.path.join(data_dir, f"q_{int(q*100)}")

            ates = []
            abs_errs = []

            for i in range(replications):
                y, t, mu0, mu1, x = read_ihdp_data(os.path.join(data_path, f"{i}.csv"))
                if model == 'trimmed':
                    data = train_and_predict_dragons_trimmed(t, y, x,
                                                     targeted_regularization=True,
                                                     knob_loss=knob_loss, ratio=ratio, dragon=model,
                                                     val_split=0.2,
                                                     batch_size=batch_size)
                elif model == 'tarnet':
                    data = train_and_predict_dragons(t, y, x,
                                                     targeted_regularization=False,
                                                     knob_loss=knob_loss, ratio=ratio, dragon=model,
                                                     val_split=0.2,
                                                     batch_size=batch_size)
                else:
                    data = train_and_predict_dragons(t, y, x,
                                                             targeted_regularization=True,
                                                             knob_loss=knob_loss, ratio=ratio, dragon=model,
                                                             val_split=0.2,
                                                             batch_size=batch_size)

                q_t0, q_t1, g, t, y_dragon = data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), \
                                                 data['g'].reshape(-1, 1), data['t'].reshape(-1, 1), \
                                                 data['y'].reshape(-1, 1)
                if model == 'trimmed':
                    psi_tmle, _, _, _, _, _ = ate_trimmed.psi_tmle_cont_outcome(q_t0, q_t1, g, t, y_dragon)
                elif model == 'tarnet':
                    psi_tmle = psi_naive(q_t0, q_t1, g, t, y_dragon)
                else:
                    psi_tmle, _, _, _, _, _ = psi_tmle_cont_outcome(q_t0, q_t1, g, t, y_dragon)

                truth = (mu1 - mu0).mean()
                abs_err = abs(truth - psi_tmle)
                ates.append(psi_tmle)
                abs_errs.append(abs_err)

            # print("--------- NOT STORING RESULTS ---------")
            results_path = f"ihdp_results/{model}/q_{int(q*100)}"
            os.makedirs(results_path, exist_ok=True)
            pd.DataFrame(np.array(ates)).to_csv(os.path.join(results_path, f"ates.csv"), header=None, index=None)
            pd.DataFrame(np.array(abs_errs)).to_csv(os.path.join(results_path, f"abs_errs.csv"), header=None, index=None)

            print(f"\t\tTime taken (s): {int(time.time() - subtimes)}")
            print(f"\t\tTotal time elapsed (min): {int((time.time() - start_time)/60)}\n")
            subtimes = time.time()


def get_treg_errors(data_dir='ihdp_data', knob_loss=dragonnet_loss_binarycross, ratio=1., batch_size=64):

    models = ['dragonnet', 'trimmed']
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    q = qs[-1]
    dataset = 0

    treg_trimmed = None
    treg_dragonnet = None
    for model in models:
        data_path = os.path.join(data_dir, f"q_{int(q*100)}")
        y, t, mu0, mu1, x = read_ihdp_data(os.path.join(data_path, f"{dataset}.csv"))
        if model == 'trimmed':
            data = train_and_predict_dragons_trimmed(t, y, x,
                                             targeted_regularization=True,
                                             knob_loss=knob_loss, ratio=ratio, dragon=model,
                                             val_split=0.2,
                                             batch_size=batch_size)
        else:
            data = train_and_predict_dragons(t, y, x,
                                                     targeted_regularization=True,
                                                     knob_loss=knob_loss, ratio=ratio, dragon=model,
                                                     val_split=0.2,
                                                     batch_size=batch_size)

        q_t0, q_t1, g, t, y_dragon = data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), \
                                         data['g'].reshape(-1, 1), data['t'].reshape(-1, 1), \
                                         data['y'].reshape(-1, 1)

        # calculate the values of the exact t-reg term under investigation (which uses propensities)
        if model == 'trimmed':
            q_t0, q_t1, g, t, y_dragon = truncate_all_by_g(q_t0, q_t1, g, t, y_dragon)
            treg_trimmed_0 = np.zeros_like(t) * (1.0 / g) - (1.0 - np.zeros_like(t)) / (1.0 - g)
            treg_trimmed_1 = np.ones_like(t) * (1.0 / g) - (1.0 - np.ones_like(t)) / (1.0 - g)
            treg_trimmed_0, treg_trimmed_1 = treg_trimmed_0.flatten(), treg_trimmed_1.flatten()
            treg_trimmed = np.concatenate((treg_trimmed_0, treg_trimmed_1))
        else:
            treg_dragonnet_0 = np.zeros_like(t) * (1.0 / g) - (1.0 - np.zeros_like(t)) / (1.0 - g)
            treg_dragonnet_1 = np.ones_like(t) * (1.0 / g) - (1.0 - np.ones_like(t)) / (1.0 - g)
            treg_dragonnet_0, treg_dragonnet_1 = treg_dragonnet_0.flatten(), treg_dragonnet_1.flatten()
            treg_dragonnet = np.concatenate((treg_dragonnet_0, treg_dragonnet_1))

    fig1, ax1 = plt.subplots()
    ax1.set_title(
        f"Boxplot of targeted regularization perturbances for sample {dataset} \nfor q={q}")
    ax1.boxplot([treg_dragonnet, treg_trimmed])
    plt.xticks([1, 2], ['DragonNet \nwithout trimming', 'DragonNet \nwith trimming'])
    plt.ylabel("Perturbance")
    plt.show()


def main():
    # get_ihdp_results()
    get_treg_errors()

    print("Finished running!")


if __name__ == '__main__':
    main()
