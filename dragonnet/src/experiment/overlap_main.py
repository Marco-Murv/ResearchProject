from overlap_models import *
from src.process_result.overlap_ate import *
from generate_data import *
from ihdp_main import train_and_predict_dragons_trimmed
import src.semi_parametric_estimation.ate as ate_trimmed
import os
import glob
import time
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.optimizers import rmsprop, SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN


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


def train_and_predict_dragons(t, y_unscaled, x, targeted_regularization=True, output_dir='',
                              knob_loss=dragonnet_loss_binarycross, ratio=1., dragon='', val_split=0.2, batch_size=64):
    verbose = 0
    # t = t.reshape(-1, 1)
    # y_unscaled = y_unscaled.reshape(-1, 1)
    y_scaler = StandardScaler().fit(y_unscaled)
    y = y_scaler.transform(y_unscaled)

    if dragon == 'tarnet':
        # print("I am here making tarnet")
        dragonnet = make_tarnet(x.shape[1], 0.01)

    elif dragon == 'dragonnet':
        # print("I am here making dragonnet")
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

    # models = ['dragonnet', 'trimmed', 'tarnet']
    models = ['trimmed', 'tarnet']
    sub_pop_sizes = [0.1, 0.25, 0.5]
    propensities = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.001]
    sample_sizes = [100, 200, 500, 1000, 2000] # removed 5000 for calculation time; can perhaps add in future separately
    replications = 200

    start_time = time.time()
    subtimes = start_time
    for model in models:
        for sub_pop in sub_pop_sizes:
            for samples in sample_sizes:
                batch_size = 64 if samples < 500 else (128 if samples < 2000 else 512)

                for prop in propensities:
                    if model == 'tarnet' and samples != 1000:  # Only need MAE results for tarnet
                        continue
                    if samples != 1000 and ((abs(prop - 0.3)) < 0.00001 or (abs(prop - 0.1)) < 0.00001):
                        continue

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
                        else:
                            psi_tmle, _, _, _, _, _ = psi_tmle_cont_outcome(q_t0, q_t1, g, t, y_dragon)
                        abs_err = abs(1.0 - psi_tmle)
                        ates.append(psi_tmle)
                        abs_errs.append(abs_err)

                    results_path = f"synth_result/{model}/sub_pop_{sub_pop}/samples_{samples}/propens_{prop}"
                    os.makedirs(results_path, exist_ok=True)
                    pd.DataFrame(np.array(ates)).to_csv(os.path.join(results_path, f"ates.csv"), header=None, index=None)
                    pd.DataFrame(np.array(abs_errs)).to_csv(os.path.join(results_path, f"abs_errs.csv"), header=None, index=None)

                    print(f"\t\tTime taken (s): {int(time.time() - subtimes)}")
                    print(f"\t\tTotal time elapsed (min): {int((time.time() - start_time)/60)}\n")
                    subtimes = time.time()


def run_mae(data_base_dir='', output_dir='~/result/ihdp/', knob_loss=dragonnet_loss_binarycross,
             ratio=1., dragon='dragonnet'):
    print("the dragon is {}".format(dragon))

    replications = 20
    data_size = 1000
    # sub_pop_sizes = [0.1, 0.25, 0.5, 0.8]
    sub_pop_sizes = [0.1, 0.25, 0.5]
    propensity = [0.5, 0.1, 0.05, 0.02, 0.01, 0.001]
    # propensity = [0.5, 0.1, 0.001]
    print("Replications: {}".format(replications))
    print("Sample sizes: {}".format(data_size))
    mae = []

    locs = list(range(len(propensity)))
    plt.xticks(locs, propensity)
    plt.xlabel("Probability of treatment for sub-population")
    plt.ylabel("Mean Abs. Error")
    start_time = time.time()
    for sub_pop_size in sub_pop_sizes:
        mae_sub = []
        mae_tuple = []
        for probability in propensity:
            for repl in range(replications):
                print("\n Sub_pop: {}, \t Probability: {}, \t Replication: {}".format(sub_pop_size, probability, repl))
                simulation_output_dir = os.path.join(output_dir, str(repl))
                os.makedirs(simulation_output_dir, exist_ok=True)

                x, t, y = generate_unif(data_size, sub_pop_size, probability)
                train_output = train_and_predict_dragons(t, y, x,
                                                               targeted_regularization=True,
                                                               output_dir=simulation_output_dir,
                                                               knob_loss=knob_loss, ratio=ratio, dragon=dragon,
                                                               val_split=0.2, batch_size=64)

                train_output_dir = os.path.join(simulation_output_dir, "targeted_regularization")
                os.makedirs(train_output_dir, exist_ok=True)
                for num, output in enumerate(train_output):
                    np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                        **output)
                elapsed_time = time.time() - start_time
                print("***************************** elapsed_time is: ", elapsed_time)
            mean_abs_err = get_mae(n_replication=replications)
            mae_tuple.append((probability, mean_abs_err))
            mae_sub.append(mean_abs_err)
        print("----------- Finished with sub_pop {} -----------".format(sub_pop_size))
        print(mae_tuple)
        mae.append(mae_sub)
        plt.plot(locs, mae_sub, label="{}% of pop.".format((int) (sub_pop_size*100)))
    plt.title("MAE when reaching overlap violations for \n sub-populations of different sizes.")
    plt.legend()
    plt.show()


def run_conv(data_base_dir='', output_dir='~/result/ihdp/', knob_loss=dragonnet_loss_binarycross,
             ratio=1., dragon='dragonnet'):
    print("the dragon is {}".format(dragon))

    replications = 50
    data_sizes = [100, 200, 500, 1000, 2000]
    propensity = [0.5, 0.05, 0.01]
    sub_pop_size = 0.5
    print("Replications: {}".format(replications))
    print("Sample sizes: {}".format(data_sizes))
    mae = []

    locs = list(range(len(data_sizes)))
    plt.xticks(locs, data_sizes)
    plt.xlabel("Probability of treatment for sub-population")
    plt.ylabel("Mean Abs. Error")
    start_time = time.time()
    for probability in propensity:
        mae_sub = []
        mae_tuple = []
        for data_size in data_sizes:
            for repl in range(replications):
                print("\n Data_size: {}, \t Probability: {}, \t Replication: {}".format(data_size, probability, repl))
                simulation_output_dir = os.path.join(output_dir, str(repl))
                os.makedirs(simulation_output_dir, exist_ok=True)

                x, t, y = generate_unif(data_size, sub_pop_size, probability)
                train_output = train_and_predict_dragons(t, y, x,
                                                               targeted_regularization=True,
                                                               output_dir=simulation_output_dir,
                                                               knob_loss=knob_loss, ratio=ratio, dragon=dragon,
                                                               val_split=0.2, batch_size=64)

                train_output_dir = os.path.join(simulation_output_dir, "targeted_regularization")
                os.makedirs(train_output_dir, exist_ok=True)
                for num, output in enumerate(train_output):
                    np.savez_compressed(os.path.join(train_output_dir, "{}_replication_train.npz".format(num)),
                                        **output)
                elapsed_time = time.time() - start_time
                print("***************************** elapsed_time is: ", elapsed_time)
            mean_abs_err = get_mae(n_replication=replications)
            mae_tuple.append((data_size, mean_abs_err))
            mae_sub.append(mean_abs_err)
        print("----------- Finished with probability {} -----------".format(probability))
        print(mae_tuple)
        mae.append(mae_sub)
        plt.plot(locs, mae_sub, label="{}% prob. of treatment".format((int) (probability*100)))
    plt.title("Convergence of estimator over increasing \n sample sizes.")
    plt.legend()
    plt.show()


def main():
    output_base_dir = 'results/overlap'
    knob = 'dragonnet'
    # output_dir = os.path.join(output_base_dir, knob)
    # run_conv(data_base_dir='', output_dir=output_dir, dragon='dragonnet')
    # run_mae(data_base_dir='', output_dir=output_dir, dragon='dragonnet')
    # get_results()

    knob_loss = dragonnet_loss_binarycross
    ratio = 1.
    for model in ['trimmed']:
        for sub_pop in [0.5]:
            for samples in [1000]:
                batch_size = 64 if samples < 500 else (128 if samples < 2000 else 512)

                for prop in [0.001]:
                    data_path = os.path.join('synth_data', f"sub_pop_{sub_pop}/samples_{samples}/propens_{prop}")

                    for i in range(200):
                        x, t, y = read_data(os.path.join(data_path, f"{i}.csv"))
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
                        if model == 'trimmed':
                            psi_tmle, _, _, _, _, _ = ate_trimmed.psi_tmle_cont_outcome(q_t0, q_t1, g, t, y_dragon)
                        else:
                            psi_tmle, _, _, _, _, _ = psi_tmle_cont_outcome(q_t0, q_t1, g, t, y_dragon)


    print("Finished running!")


if __name__ == '__main__':
    main()
