import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

def generate_unif(sample_size, sub_pop_size, probability):
    # using three covariates
    covariates = np.random.uniform(0, 1, (sample_size, 3))

    treatment = np.zeros(sample_size)
    indices = covariates[:, 2] < sub_pop_size
    treatment[indices] = np.random.binomial(1, probability, np.sum(indices))
    treatment[np.invert(indices)] = np.random.binomial(1, 0.5, np.sum(np.invert(indices)))

    outcomes = np.ones(sample_size) + treatment + covariates[:, 0] + 2 * covariates[:, 1] + 0.5 * covariates[:, 2] \
               + np.random.standard_normal(sample_size)

    return covariates, treatment, outcomes


def create_synthetic_datasets(data_dir="synth_data"):
    sub_pop_sizes = [0.1, 0.25, 0.5]
    propensities = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    replications = 200

    path = ''
    for sub_pop in sub_pop_sizes:
        for samples in sample_sizes:
            for prop in [0.0]:
                # Create the folder if it doesn't exist
                path = os.path.join(data_dir, f"sub_pop_{sub_pop}/samples_{samples}/propens_{prop}")
                os.makedirs(path, exist_ok=True)

                # make 200 replications of the data with these specific settings and store
                for i in range(replications):
                    cov, t, y = generate_unif(samples, sub_pop, prop)
                    t = t.reshape(-1, 1)
                    y = y.reshape(-1, 1)
                    comb = np.hstack((cov, t, y))
                    pd.DataFrame(comb).to_csv(os.path.join(path, f"{i}.csv"), header=None, index=None)

    return


def read_data(path):
    result = np.loadtxt(path, delimiter=',')
    cov = result[:, 0:3]
    t = result[:, 3].reshape(-1, 1)
    y = result[:, 4].reshape(-1, 1)
    return cov, t, y


def take_random_ihdp_samples(data_dir="ihdp_data"):
    data_train = np.load(os.path.join(data_dir, f"ihdp_npci_1-1000.train.npz"))
    data_test = np.load(os.path.join(data_dir, f"ihdp_npci_1-1000.test.npz"))

    samples = np.random.choice(1000, 200)
    path = os.path.join(data_dir, f"original_samples")
    os.makedirs(path, exist_ok=True)

    for i in range(200):
        index = samples[i]

        x_train, x_test = data_train['x'][:, :, index], data_test['x'][:, :, index]
        x = np.concatenate((x_train, x_test), axis=0)

        t_train, t_test = data_train['t'][:, index], data_test['t'][:, index]
        t = np.concatenate((t_train, t_test))

        y_train, y_test = data_train['yf'][:, index], data_test['yf'][:, index]
        y = np.concatenate((y_train, y_test))

        mu0_train, mu0_test = data_train['mu0'][:, index], data_test['mu0'][:, index]
        mu0 = np.concatenate((mu0_train, mu0_test))

        mu1_train, mu1_test = data_train['mu1'][:, index], data_test['mu1'][:, index]
        mu1 = np.concatenate((mu1_train, mu1_test))

        combined = np.hstack((mu1.reshape(-1, 1), x))
        combined = np.hstack((mu0.reshape(-1, 1), combined))
        combined = np.hstack((t.reshape(-1, 1), combined))
        combined = np.hstack((y.reshape(-1, 1), combined))

        pd.DataFrame(combined).to_csv(os.path.join(path, f"{i}.csv"), header=None, index=None)
    pass


def read_ihdp_data(path):
    result = np.loadtxt(path, delimiter=',')
    y = result[:, 0].reshape(-1, 1)
    t = result[:, 1].reshape(-1, 1)
    mu0 = result[:, 2].reshape(-1, 1)
    mu1 = result[:, 3].reshape(-1, 1)
    x = result[:, 4:]

    return y, t, mu0, mu1, x


def split_data(y, t, mu0, mu1, x, control=True):
    if control:
        indices = t < 0.5
    else:
        indices = t > 0.5

    return y[indices], t[indices], mu0[indices], mu1[indices], x[indices, :]


def generate_imbalanced_ihdp_data(data_dir="ihdp_data", num_samples=200, remove=347):
    samples_path = os.path.join(data_dir, f"original_samples")
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]

    for q in qs:
        print(f"\nq = {q}")
        results_path = os.path.join(data_dir, f"q_{int(q*100)}")
        os.makedirs(results_path, exist_ok=True)

        prop_avg = []
        prop_max = []
        prop_med = []

        for i in range(num_samples):
            removed = []

            y, t, mu0, mu1, x = read_ihdp_data(os.path.join(samples_path, f"{i}.csv"))
            y = y.flatten()
            t = t.flatten()
            mu0 = mu0.flatten()
            mu1 = mu1.flatten()

            y_c, t_c, mu0_c, mu1_c, x_c = split_data(y, t, mu0, mu1, x, control=True)
            y_t, t_t, mu0_t, mu1_t, x_t = split_data(y, t, mu0, mu1, x, control=False)

            clf = LogisticRegression(random_state=0).fit(x, t)
            propens = clf.predict_proba(x_c)[:, 1]

            while len(removed) < remove:
                if np.random.uniform() < q:
                    m = np.zeros(propens.size, dtype=bool)
                    m[removed] = True
                    a = np.ma.array(propens, mask=m)
                    removed.append(np.argmax(a))
                else:
                    removed.append(np.random.choice(np.setdiff1d(range(0, propens.size), removed)))

            mask = np.ones(propens.size, dtype=bool)
            mask[removed] = False
            y_c, t_c, mu0_c, mu1_c, x_c = y_c[mask], t_c[mask], mu0_c[mask], mu1_c[mask], x_c[mask, :]
            y, t, mu0, mu1, x = np.concatenate((y_t, y_c)), np.concatenate((t_t, t_c)),\
                                np.concatenate((mu0_t, mu0_c)), np.concatenate((mu1_t, mu1_c)), \
                                np.concatenate((x_t, x_c), axis=0)

            combined = np.hstack((mu1.reshape(-1, 1), x))
            combined = np.hstack((mu0.reshape(-1, 1), combined))
            combined = np.hstack((t.reshape(-1, 1), combined))
            combined = np.hstack((y.reshape(-1, 1), combined))

            prop_avg.append(np.mean(propens[mask]))
            prop_max.append(np.max(propens[mask]))
            prop_med.append(np.median(propens[mask]))

            pd.DataFrame(combined).to_csv(os.path.join(results_path, f"{i}.csv"), header=None, index=None)
        pd.DataFrame(np.array(prop_avg)).to_csv(os.path.join(results_path, f"propensities_avg.csv"), header=None, index=None)
        pd.DataFrame(np.array(prop_max)).to_csv(os.path.join(results_path, f"propensities_max.csv"), header=None, index=None)
        pd.DataFrame(np.array(prop_med)).to_csv(os.path.join(results_path, f"propensities_median.csv"), header=None, index=None)

    pass


def main():
    generate_imbalanced_ihdp_data()
    print("Finished running!")


if __name__ == '__main__':
    main()
