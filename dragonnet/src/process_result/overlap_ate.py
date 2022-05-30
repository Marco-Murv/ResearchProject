from numpy import load

from src.semi_parametric_estimation.overlap_ate_calculators import *


def load_data(knob='dragonnet', replication=1, model='targeted_regularization', train_test='train'):
    """
    loading train test experiment results
    """

    file_path = 'results/overlap/{}/'.format(knob)
    data = load(file_path + '{}/{}/0_replication_{}.npz'.format(replication, model, train_test))

    return data['q_t0'].reshape(-1, 1), data['q_t1'].reshape(-1, 1), data['g'].reshape(-1, 1), \
           data['t'].reshape(-1, 1), data['y'].reshape(-1, 1), data['index'].reshape(-1, 1), data['eps'].reshape(-1, 1)


def get_estimate(q_t0, q_t1, g, t, y_dragon, index, eps):
    """
    getting the back door adjustment & TMLE estimation
    """

    # psi_n = psi_naive(q_t0, q_t1, g, t, y_dragon)
    psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss = psi_tmle_cont_outcome(q_t0, q_t1, g, t, y_dragon)
    return 0, psi_tmle, initial_loss, final_loss, g_loss


def get_mae(train_test='train', n_replication=20):
    knob = 'dragonnet'
    model = 'targeted_regularization'
    tmle_errors = []
    for rep in range(n_replication):
        q_t0, q_t1, g, t, y_dragon, index, eps = load_data(knob, rep, model, train_test)
        truth = 1.0 #fixed value for the default outcome function of synthetic data!

        _, psi_tmle, initial_loss, final_loss, g_loss = get_estimate(q_t0, q_t1, g, t, y_dragon, index, eps)

        tmle_err = abs(truth - psi_tmle).mean()
        tmle_errors.append(tmle_err)

    return np.mean(tmle_errors)
