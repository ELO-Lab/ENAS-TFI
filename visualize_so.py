import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as p

plt.rc('font', family='Times New Roman')

colors_list = ['k', 'g', 'r', 'b', 'm', 'y']

benchmark_and_dataset_dict = {
    'SO-NAS101': ['NASBench101', 'CIFAR-10'],
    'SO-NAS201-1': ['NASBench201', 'CIFAR-10'],
    'SO-NAS201-2': ['NASBench201', 'CIFAR-100'],
    'SO-NAS201-3': ['NASBench201', 'ImageNet16-120']
}

def visualize_2D_search(objective_0_mean, objective_1_mean, objective_1_stdev,
                        tmp_objective_1_mean, tmp_objective_1_stdev,
                        label, color, axis_labels=('x', 'y'),
                        plot_original=False):
    if plot_original:
        plt.step(objective_0_mean, objective_1_mean, 'k--', label=label, linewidth=1, where='post')
        plt.scatter(objective_0_mean, objective_1_mean, s=15, facecolors='none', edgecolors='red', marker='^',
                    label='Best Architecture Found')

        plt.fill_between(objective_0_mean,
                         objective_1_mean - objective_1_stdev,
                         objective_1_mean + objective_1_stdev, alpha=0.2, fc='black', step='post')

    else:
        plt.step(objective_0_mean, objective_1_mean, label=label, c=color, where='post')
        plt.scatter(objective_0_mean, objective_1_mean, s=15, facecolors='none',  edgecolors='red', marker='^')

        plt.fill_between(objective_0_mean,
                         objective_1_mean - objective_1_stdev,
                         objective_1_mean + objective_1_stdev, alpha=0.2, fc=color, step='post')

    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])


def visualize_2D_evaluate(objective_0_mean, objective_1_mean, objective_1_stdev,
                          tmp_objective_1_mean, tmp_objective_1_stdev,
                          label, color, axis_labels=('x', 'y'),
                          plot_original=False):
    if plot_original:
        plt.step(objective_0_mean, tmp_objective_1_mean, 'k--', label=label, linewidth=1, where='post')
        plt.scatter(objective_0_mean, tmp_objective_1_mean, s=15, facecolors='none', edgecolors='blue', marker='^',
                    label='Best Architecture Found')

        plt.fill_between(objective_0_mean,
                         tmp_objective_1_mean - tmp_objective_1_stdev,
                         tmp_objective_1_mean + tmp_objective_1_stdev, alpha=0.2, fc='black', step='post')

    else:
        plt.step(objective_0_mean, tmp_objective_1_mean, label=label, c=color, where='post')
        plt.scatter(objective_0_mean, tmp_objective_1_mean, s=15, facecolors='none',  edgecolors='blue', marker='^')

        plt.fill_between(objective_0_mean,
                         tmp_objective_1_mean - tmp_objective_1_stdev,
                         tmp_objective_1_mean + tmp_objective_1_stdev, alpha=0.2, fc=color, step='post')

    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])


def visualize_2D_all(objective_0_mean, objective_1_mean, objective_1_stdev,
                     tmp_objective_1_mean, tmp_objective_1_stdev,
                     label, color, axis_labels=('x', 'y'),
                     plot_original=False):
    if plot_original:
        plt.step(objective_0_mean, objective_1_mean, 'k--', label=label, linewidth=1, where='post')
        plt.scatter(objective_0_mean, objective_1_mean, s=15, facecolors='none', edgecolors='red', marker='^', label='Best Architecture Found (validation)')

        plt.step(objective_0_mean, tmp_objective_1_mean, 'k--', linewidth=1, where='post')
        plt.scatter(objective_0_mean, tmp_objective_1_mean, s=15, facecolors='none', edgecolors='blue',
                    marker='^', label='Best Architecture Found (testing)')

    else:
        plt.step(objective_0_mean, objective_1_mean, label=label, c=color, where='post')
        plt.scatter(objective_0_mean, objective_1_mean, s=15, facecolors='none',  edgecolors='red', marker='^')

        plt.step(objective_0_mean, tmp_objective_1_mean, c=color, where='post')
        plt.scatter(objective_0_mean, tmp_objective_1_mean, s=15, edgecolors='blue', facecolors='none',
                    marker='^')

    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])

''' ---------------------------------- Visualize "nEvals" and "Hypervolume" ---------------------------------------- '''
def get_nGens_and_Accuracy():
    val_accuracy_mean_each_exp, test_accuracy_mean_each_exp = [], []
    val_accuracy_std_each_exp, test_accuracy_std_each_exp = [], []

    nGens_mean_each_exp = []

    for exp in experiments_list:
        val_accuracy_each_run = []
        test_accuracy_each_run = []
        nGens_each_run = []

        n_runs = len(['_' for f in os.listdir(exp) if '.' not in f])
        for i in range(n_runs):
            path_result = os.path.join(exp, f'{i}')
            nGens_history, best_arch_history = p.load(open(os.path.join(path_result, 'best_architecture_each_gen.p'), 'rb'))

            val_accuracy_history, test_accuracy_history = [], []
            for arch_list in best_arch_history:
                val_accuracy_history.append(arch_list[-1]['validation_accuracy'])
                test_accuracy_history.append(arch_list[-1]['testing_accuracy'])
            val_accuracy_each_run.append(val_accuracy_history)
            test_accuracy_each_run.append(test_accuracy_history)
            nGens_each_run.append(nGens_history)
        val_accuracy_each_run = np.array(val_accuracy_each_run)
        test_accuracy_each_run = np.array(test_accuracy_each_run)
        nGens_each_run = np.array(nGens_each_run)
        p.dump([nGens_each_run, val_accuracy_each_run, test_accuracy_each_run],
               open(os.path.join(PATH_RESULTS, 'nEvals_accuracy', exp.split('\\')[-1] + '.p'), 'wb'))

        val_accuracy_mean = np.mean(val_accuracy_each_run, axis=0)
        val_accuracy_stdev = np.std(val_accuracy_each_run, axis=0)

        test_accuracy_mean = np.mean(test_accuracy_each_run, axis=0)
        test_accuracy_stdev = np.std(test_accuracy_each_run, axis=0)

        nGens_mean = np.mean(nGens_each_run, axis=0)

        val_accuracy_mean_each_exp.append(val_accuracy_mean)
        val_accuracy_std_each_exp.append(val_accuracy_stdev)

        test_accuracy_mean_each_exp.append(test_accuracy_mean)
        test_accuracy_std_each_exp.append(test_accuracy_stdev)

        nGens_mean_each_exp.append(nGens_mean)

    return [nGens_mean_each_exp,
            val_accuracy_mean_each_exp, val_accuracy_std_each_exp,
            test_accuracy_mean_each_exp, test_accuracy_std_each_exp
            ]


def visualize_nGens_and_Accuracy():
    try:
        os.mkdir(os.path.join(PATH_RESULTS, 'nEvals_accuracy'))
    except:
        pass

    nGens_and_Accuracy = get_nGens_and_Accuracy()

    nGens_mean_each_exp = nGens_and_Accuracy[0]
    val_accuracy_mean_each_exp, val_accuracy_std_each_exp = nGens_and_Accuracy[1], nGens_and_Accuracy[2]
    test_accuracy_mean_each_exp, test_accuracy_std_each_exp = nGens_and_Accuracy[3], nGens_and_Accuracy[4]

    for require in ['search', 'evaluate', 'all']:
        axis_lbs = ['#Gens', 'Accuracy']

        for i, exp in enumerate(experiments_list):
            if '.' not in exp:
                if platform == "linux" or platform == "linux2":
                    hyperparameters = exp.split('/')[-1][:-20].split('_')
                else:
                    hyperparameters = exp.split('//')[-1][:-20].split('_')
            else:
                break
            algorithm_name = hyperparameters[1]
            variant_configuration = hyperparameters[3:]

            label = f'{algorithm_name}'
            plot_original = True
            if variant_configuration[0] == 'True':
                label += f' w/ Warmup (M = {variant_configuration[1]})'
                plot_original = False

            if require == 'search':
                visualize_2D_search(objective_0_mean=nGens_mean_each_exp[i],
                                    objective_1_mean=val_accuracy_mean_each_exp[i],
                                    objective_1_stdev=val_accuracy_std_each_exp[i],
                                    tmp_objective_1_mean=test_accuracy_mean_each_exp[i],
                                    tmp_objective_1_stdev=test_accuracy_std_each_exp[i],
                                    plot_original=plot_original,
                                    axis_labels=axis_lbs, label=label, color=colors_list[i])
            elif require == 'evaluate':
                visualize_2D_evaluate(objective_0_mean=nGens_mean_each_exp[i],
                                      objective_1_mean=val_accuracy_mean_each_exp[i],
                                      objective_1_stdev=val_accuracy_std_each_exp[i],
                                      tmp_objective_1_mean=test_accuracy_mean_each_exp[i],
                                      tmp_objective_1_stdev=test_accuracy_std_each_exp[i],
                                      plot_original=plot_original,
                                      axis_labels=axis_lbs, label=label, color=colors_list[i])
            else:
                visualize_2D_all(objective_0_mean=nGens_mean_each_exp[i],
                                 objective_1_mean=val_accuracy_mean_each_exp[i],
                                 objective_1_stdev=val_accuracy_std_each_exp[i],
                                 tmp_objective_1_mean=test_accuracy_mean_each_exp[i],
                                 tmp_objective_1_stdev=test_accuracy_std_each_exp[i],
                                 plot_original=plot_original,
                                 axis_labels=axis_lbs, label=label, color=colors_list[i])
        plt.grid()
        # if problem_name in ['SO-NAS201-1', 'SO-NAS201-2', 'SO-NAS201-3']:
        #     true_best_architecture = p.load(open(f'{path_data}/{benchmark}/[{dataset}]_best_arch.p', 'rb'))
        # else:
        #     true_best_architecture = p.load(open(f'{path_data}/{benchmark}/best_arch.p', 'rb'))
        x_limit = len(val_accuracy_mean_each_exp[-1])
        x_ = x_limit + 2
        # if require == 'search':
        #     if benchmark == 'NASBench201':
        #         true_best_architecture_val_acc = true_best_architecture['validation']['val_acc']
        #     else:
        #         true_best_architecture_val_acc = true_best_architecture['val_acc'][-1]
        #     plt.scatter(x_, true_best_architecture_val_acc, marker='*', c='red', s=20,
        #                 label='True Best Architecture')
        # elif require == 'evaluate':
        #     if benchmark == 'NASBench201':
        #         true_best_architecture_test_acc = true_best_architecture['testing']['test_acc']
        #     else:
        #         true_best_architecture_test_acc = true_best_architecture['test_acc'][-1]
        #     plt.scatter(x_, true_best_architecture_test_acc, marker='*', c='blue', s=20,
        #                 label='True Best Architecture')
        # else:
        #     if benchmark == 'NASBench201':
        #         true_best_architecture_val_acc = true_best_architecture['validation']['val_acc']
        #         true_best_architecture_test_acc = true_best_architecture['testing']['test_acc']
        #     else:
        #         true_best_architecture_val_acc = true_best_architecture['val_acc'][-1]
        #         true_best_architecture_test_acc = true_best_architecture['test_acc'][-1]
        #     plt.scatter(x_, true_best_architecture_val_acc, marker='*', c='red', s=20,
        #                 label='True Best Architecture (validation)')
        #     plt.scatter(x_, true_best_architecture_test_acc, marker='*', c='blue', s=20,
        #                 label='True Best Architecture (testing)')
        if LOG_Y:
            plt.xscale('log')
        if LOG_Y:
            plt.yscale('log')
        plt.legend(loc=4)
        if require == 'search' or require == 'evaluate':
            title = f'{problem_name} | {require}'
            figure_name = f'{PATH_RESULTS}/[{problem_name}_{dataset}]_result({require}).png'
        else:
            title = f'{problem_name}'
            figure_name = f'{PATH_RESULTS}/[{problem_name}_{dataset}]_result.png'
        plt.title(title)
        plt.savefig(figure_name, dpi=300)
        plt.clf()


''' ------------------------------------ Main ------------------------------------ '''
def main():
    visualize_nGens_and_Accuracy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_results', type=str)
    args = parser.parse_args()
    LOG_X = False
    LOG_Y = False
    PATH_RESULTS = args.path_results
    checked_lst = ['IGD', 'Hypervolume', 'nEvals', 'png']

    from sys import platform

    if platform == "linux" or platform == "linux2":
        path_data = '/'.join(os.path.abspath(__file__).split('/')[:-1]) + '/data'
    elif platform == "win32" or platform == "win64":
        path_data = '/'.join(os.path.abspath(__file__).split('\\')[:-1]) + '/data'
    else:
        raise ValueError()
    """ =========================================== """
    exp_0 = None
    problem_name = None
    dataset = None
    for experiment in os.listdir(PATH_RESULTS):
        if any(word in experiment for word in checked_lst):
            continue
        else:
            exp_0 = experiment
            exp_0 = exp_0.split('_')
            problem_name = exp_0[0]
            break

    benchmark, dataset = benchmark_and_dataset_dict[problem_name]
    if problem_name in ['SO-NAS201-1', 'SO-NAS201-2', 'SO-NAS201-3']:
        maxEvals = 1000
    else:
        maxEvals = 5000
    """ =========================================== """
    experiments_list = []
    for experiment in os.listdir(PATH_RESULTS):
        if any(word in experiment for word in checked_lst):
            continue
        else:
            experiments_list.append(os.path.join(PATH_RESULTS, experiment))
    """ =========================================== """
    main()
