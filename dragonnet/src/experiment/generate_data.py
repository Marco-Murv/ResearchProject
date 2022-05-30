import numpy as np
import os
import pandas as pd

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
            for prop in propensities:
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
    # path = os.path.join("synth_data", f"sub_pop_{0.1}/samples_{100}/propens_{0.1}")
    # result = np.loadtxt(os.path.join(path, f"{0}.csv"), delimiter=',')

    result = np.loadtxt(path, delimiter=',')
    cov = result[:, 0:3]
    t = result[:, 3].reshape(-1, 1)
    y = result[:, 4].reshape(-1, 1)
    return cov, t, y


def main():
    # create_synthetic_datasets()
    read_data(None)
    print("Finished running!")


if __name__ == '__main__':
    main()
