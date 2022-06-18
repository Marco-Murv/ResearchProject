import matplotlib.pyplot as plt
import numpy as np

def check_errs():
    title_names = {'dragonnet': 'DragonNet without trimming', 'trimmed': 'DragonNet with trimming', 'tarnet': 'TARnet'}

    plt.xlabel("Replication")
    plt.ylabel("Mean Abs. Error")
    sub_pop_sizes = [0.1, 0.25]
    for sub_pop in sub_pop_sizes:
        results_path = f"synth_result/{'dragonnet'}/sub_pop_{sub_pop}/samples_{1000}/propens_{0.5}/abs_errs.csv"
        errors = np.loadtxt(results_path, delimiter=',')  # list of 200 abs errors
        mae = errors

        plt.plot(list(range(len(mae))), mae, label="{}% of pop. affected".format((int)(sub_pop * 100)))

    plt.title(f"MAEs of {title_names['dragonnet']} .")
    plt.legend(loc='upper left')
    plt.show()

    return


def plot_maes():
    models = ['dragonnet', 'trimmed', 'tarnet']
    sub_pop_sizes = [0.1, 0.25, 0.5]
    propensities = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.001]
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
    models = ['dragonnet', 'trimmed', 'tarnet']
    # sub_pop_sizes = [0.1, 0.25, 0.5]c
    sub_pop_sizes = [0.5]
    propensities = [0.5, 0.05, 0.02, 0.01, 0.001]
    sample_sizes = [100, 200, 500, 1000, 2000]

    title_names = {'dragonnet': 'DragonNet without trimming', 'trimmed': 'DragonNet with trimming', 'tarnet': 'TARnet'}

    for model in models:
        plt.xlabel("Sample size")
        plt.ylabel("Mean Abs. Error")

        for prop in propensities:
            mae = []
            for samples in sample_sizes:
                results_path = f"synth_result/{model}/sub_pop_{sub_pop_sizes[0]}/samples_{samples}/propens_{prop}/abs_errs.csv"
                errors = np.loadtxt(results_path, delimiter=',')  # list of 200 abs errors
                mae.append(np.mean(errors))

            plt.plot(sample_sizes, mae, label="{}% prob. of treatment".format((prop * 100)))

        plt.title(f"Convergence of MAEs of {title_names[model]} \n when {int(sub_pop_sizes[0]*100)}% of population is affected.")
        plt.legend(loc='upper right')
        plt.show()
    return


def plot_box():
    sub_pop_sizes = [0.1, 0.25, 0.5]
    propensities = [0.5, 0.05, 0.02, 0.01, 0.001]
    sample_sizes = [1000]

    title_names = {'dragonnet': 'DragonNet without trimming', 'trimmed': 'DragonNet with trimming', 'tarnet': 'TARnet'}

    for prop in propensities:
        for samples in sample_sizes:
            results_path_dnet = f"synth_result/{'dragonnet'}/sub_pop_{sub_pop_sizes[1]}/samples_{samples}/propens_{prop}/abs_errs.csv"
            errors_dnet = np.loadtxt(results_path_dnet, delimiter=',')  # list of 200 abs errors

            results_path_trim = f"synth_result/{'trimmed'}/sub_pop_{sub_pop_sizes[1]}/samples_{samples}/propens_{prop}/abs_errs.csv"
            errors_trim = np.loadtxt(results_path_trim, delimiter=',')  # list of 200 abs errors

            results_path_tar = f"synth_result/{'tarnet'}/sub_pop_{sub_pop_sizes[1]}/samples_{samples}/propens_{prop}/abs_errs.csv"
            errors_tar = np.loadtxt(results_path_tar, delimiter=',')  # list of 200 abs errors

            fig1, ax1 = plt.subplots()
            ax1.set_title(f"Boxplot of ATE estimators when {int(sub_pop_sizes[1] * 100)}% of population \n has {prop*100}% of treatment and sample size {sample_sizes[0]}.")
            ax1.boxplot([errors_dnet, errors_trim, errors_tar])
            plt.xticks([1, 2, 3], ['DragonNet \nwithout trimming', 'DragonNet \nwith trimming', 'TARnet'])
            plt.ylabel("Mean Abs. Error")
            plt.show()
    return


def plot_mae_ihdp():
    models = ['dragonnet', 'trimmed', 'tarnet']
    title_names = {'dragonnet': 'DragonNet without trimming', 'trimmed': 'DragonNet with trimming', 'tarnet': 'TARnet'}
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]

    for model in models:
        plt.xlabel("q")
        plt.ylabel("Mean Abs. Error")

        mae = []
        for q in qs:
            results_path = f"ihdp_results/{model}/q_{int(q*100)}/abs_errs.csv"
            errors = np.loadtxt(results_path, delimiter=',')  # list of 200 abs errors
            mae.append(np.mean(errors))

        plt.plot(qs, mae, label=f"{title_names[model]}")

    plt.title(f"MAEs using IHDP data with increasing imbalance \nbetween treated and non-treated.")
    plt.legend(loc='upper left')
    plt.show()
    return


def plot_box_ihdp():
    models = ['dragonnet', 'trimmed', 'tarnet']
    title_names = {'dragonnet': 'DragonNet without trimming', 'trimmed': 'DragonNet with trimming', 'tarnet': 'TARnet'}
    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    qs = [0.75, 1.0]

    for q in qs:
        results_path_dnet = f"ihdp_results/dragonnet/q_{int(q*100)}/abs_errs.csv"
        errors_dnet = np.loadtxt(results_path_dnet, delimiter=',')  # list of 200 abs errors
        if q == 0.75:
            errors_dnet = errors_dnet[errors_dnet < 50]
        if q == 1.0:
            errors_dnet = errors_dnet[errors_dnet < 200]

        results_path_trim = f"ihdp_results/trimmed/q_{int(q*100)}/abs_errs.csv"
        errors_trim = np.loadtxt(results_path_trim, delimiter=',')  # list of 200 abs errors

        results_path_tar = f"ihdp_results/tarnet/q_{int(q*100)}/abs_errs.csv"
        errors_tar = np.loadtxt(results_path_tar, delimiter=',')  # list of 200 abs errors

        fig1, ax1 = plt.subplots()
        ax1.set_title(
            f"Boxplot of ATE estimators for q={q}.")
        ax1.boxplot([errors_dnet, errors_trim, errors_tar])
        plt.xticks([1, 2, 3], ['DragonNet \nwithout trimming', 'DragonNet \nwith trimming', 'TARnet'])
        plt.ylabel("Mean Abs. Error")
        plt.show()
    return


def main():
    # plot_maes()
    plot_conv()
    # plot_box()
    # plot_mae_ihdp()
    # plot_box_ihdp()
    print("Finished running!")


if __name__ == '__main__':
    main()