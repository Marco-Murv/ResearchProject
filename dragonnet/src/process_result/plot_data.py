import matplotlib.pyplot as plt
import numpy as np

# models = ['dragonnet', 'trimmed', 'tarnet']
# sub_pop_sizes = [0.1, 0.25, 0.5]
# propensities = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.001]
# sample_sizes = [100, 200, 500, 1000, 2000]
#
# results_path = f"synth_result/{model}/sub_pop_{sub_pop}/samples_{samples}/propens_{prop}"
#
# locs = list(range(len(propensity)))
# plt.xticks(locs, propensity)
# plt.xlabel("Probability of treatment for sub-population")
# plt.ylabel("Mean Abs. Error")
#
# plt.plot(locs, mae_sub, label="{}% of pop.".format((int) (sub_pop_size*100)))
# plt.title("MAE when reaching overlap violations for \n sub-populations of different sizes.")
# plt.legend()
# plt.show()

def check_errs():
    title_names = {'dragonnet': 'DragonNet without trimming', 'trimmed': 'DragonNet with trimming', 'tarnet': 'TARnet'}

    plt.xlabel("Replication")
    plt.ylabel("Mean Abs. Error")
    sub_pop_sizes = [0.1, 0.25]
    for sub_pop in sub_pop_sizes:
        mae = []
        results_path = f"synth_result/{'dragonnet'}/sub_pop_{sub_pop}/samples_{1000}/propens_{0.5}/abs_errs.csv"
        errors = np.loadtxt(results_path, delimiter=',')  # list of 200 abs errors
        mae = errors

        plt.plot(list(range(len(mae))), mae, label="{}% of pop. affected".format((int)(sub_pop * 100)))

    # plt.xlim(max(propensities), min(propensities))
    plt.title(f"MAEs of {title_names['dragonnet']} .")
    plt.legend(loc='upper left')
    plt.show()

    return


def plot_maes():
    models = ['dragonnet', 'trimmed', 'tarnet']
    # models = ['trimmed', 'tarnet']
    # models = ['dragonnet']
    sub_pop_sizes = [0.1, 0.25, 0.5]
    # propensities = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.001]
    propensities = [0.05, 0.02, 0.01, 0.001]
    sample_size = 1000

    title_names = {'dragonnet': 'DragonNet without trimming', 'trimmed': 'DragonNet with trimming', 'tarnet': 'TARnet'}

    for model in models:
        plt.xlabel("Probability of treatment")
        plt.ylabel("Mean Abs. Error")

        for sub_pop in sub_pop_sizes:
            mae = []
            for prop in propensities:
                results_path = f"synth_result/{model}/sub_pop_{sub_pop}/samples_{sample_size}/propens_{prop}/abs_errs.csv"
                errors = np.loadtxt(results_path, delimiter=',')  # list of 200 abs errors
                mae.append(np.mean(errors))

            plt.plot(propensities, mae, label="{}% of pop. affected".format((int)(sub_pop * 100)))

        plt.xlim(max(propensities), min(propensities))
        plt.title(f"MAEs of {title_names[model]} for \n increasingly lower propensity scores.")
        plt.legend(loc='upper left')
        plt.show()
    return


def plot_conv():
    models = ['dragonnet', 'trimmed']
    # models = ['dragonnet']
    sub_pop_sizes = [0.1, 0.25, 0.5]
    # propensities = [0.5, 0.05, 0.02, 0.01, 0.001]
    # propensities = [0.05, 0.02, 0.01, 0.001]
    propensities = [0.5, 0.05, 0.01, 0.001]
    sample_sizes = [100, 200, 500, 1000, 2000]

    title_names = {'dragonnet': 'DragonNet without trimming', 'trimmed': 'DragonNet with trimming', 'tarnet': 'TARnet'}

    for model in models:
        plt.xlabel("Sample size")
        plt.ylabel("Mean Abs. Error")

        for prop in propensities:
            mae = []
            for samples in sample_sizes:
                results_path = f"synth_result/{model}/sub_pop_{sub_pop_sizes[0]}/samples_{samples}/propens_{prop}/abs_errs.csv"
                # results_path = f"synth_result/{model}/sub_pop_{sub_pop_sizes[1]}/samples_{samples}/propens_{prop}/ates.csv"
                errors = np.loadtxt(results_path, delimiter=',')  # list of 200 abs errors
                mae.append(np.mean(errors))

            plt.plot(sample_sizes, mae, label="{}% prob. of treatment".format((prop * 100)))

        plt.title(f"Convergence of ATE estimator for {title_names[model]} \n when {int(sub_pop_sizes[0]*100)}% of population is affected.")
        plt.legend(loc='upper right')
        plt.show()
    return


def plot_box():
    models = ['dragonnet', 'trimmed']
    sub_pop_sizes = [0.1, 0.25, 0.5]
    propensities = [0.5, 0.05, 0.02, 0.01, 0.001]
    # propensities = [0.05, 0.02, 0.01, 0.001]
    # propensities = [0.5, 0.01, 0.001]
    # sample_sizes = [100, 200, 500, 1000, 2000]
    # sample_sizes = [500, 1000, 2000]
    sample_sizes = [1000]

    title_names = {'dragonnet': 'DragonNet without trimming', 'trimmed': 'DragonNet with trimming', 'tarnet': 'TARnet'}

    for prop in propensities:
        for samples in sample_sizes:
            results_path_dnet = f"synth_result/{'dragonnet'}/sub_pop_{sub_pop_sizes[2]}/samples_{samples}/propens_{prop}/abs_errs.csv"
            # results_path_dnet = f"synth_result/{'dragonnet'}/sub_pop_{sub_pop_sizes[2]}/samples_{samples}/propens_{prop}/ates.csv"
            errors_dnet = np.loadtxt(results_path_dnet, delimiter=',')  # list of 200 abs errors

            results_path_trim = f"synth_result/{'trimmed'}/sub_pop_{sub_pop_sizes[2]}/samples_{samples}/propens_{prop}/abs_errs.csv"
            # results_path_trim = f"synth_result/{'trimmed'}/sub_pop_{sub_pop_sizes[2]}/samples_{samples}/propens_{prop}/ates.csv"
            errors_trim = np.loadtxt(results_path_trim, delimiter=',')  # list of 200 abs errors

            fig1, ax1 = plt.subplots()
            ax1.set_title(f"Boxplot of ATE estimators when \n{int(sub_pop_sizes[2] * 100)}% of population has {prop*100}% of treatment.")
            ax1.boxplot([errors_dnet, errors_trim])
            plt.show()
    return


def main():
    # plot_maes()
    # plot_conv()
    plot_box()
    print("Finished running!")


if __name__ == '__main__':
    main()