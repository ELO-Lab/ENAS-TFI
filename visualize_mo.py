import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as p

from pymoo.indicators.hv import _HyperVolume
from scipy.interpolate import interp1d
from sys import platform

plt.rc('font', family='Times New Roman')

line_dict = {
    'NSGA-II': ['black', 'solid'],
    'NSGA-II w/ Warmup (M = 500)': ['green', 'dashed']
}

def visualize_2D(objective_0_mean, tmp_objective_1_mean, tmp_objective_1_stdev, label, line, axis_labels=('x', 'y')):
    color = line[0]
    style = line[1]
    plt.plot(objective_0_mean, tmp_objective_1_mean, c=color, ls=style, label=label, linewidth=2)
    plt.fill_between(objective_0_mean,
                     tmp_objective_1_mean - tmp_objective_1_stdev,
                     tmp_objective_1_mean + tmp_objective_1_stdev, alpha=0.15, fc=color)
    plt.ylabel(axis_labels[1])

''' ----------------------------------- Visualize "nEvals" and "IGD-value" ----------------------------------------- '''
def get_nEvals_and_IGD(pop_size):
    IGD_mean_each_exp = []
    IGD_stdev_each_exp = []

    nEvals_mean_each_exp = []

    for exp in experiments_list:
        if '.' in exp:
            continue
        IGD_each_run = []
        nEvals_each_run = []

        n_runs = len(['_' for f in os.listdir(exp) if '.' not in f])
        for i in range(n_runs):
            path_result = os.path.join(exp, f'{i}')
            tmp_data = np.array(p.load(open(os.path.join(path_result, '#Evals_and_IGD_each_gen.p'), 'rb')))
            idx_nEvals_history = np.array(tmp_data[0], dtype=int) - 1
            if idx_nEvals_history[-1] > maxEvals - 1:
                idx_nEvals_history[-1] = maxEvals - 1

            tmp_nEvals_history, tmp_IGD_history = p.load(
                open(os.path.join(path_result, '#Evals_and_IGD_evaluate.p'), 'rb'))

            nEvals_history, IGD_history = [tmp_nEvals_history[0]], [tmp_IGD_history[0]]

            for j in range(1, len(tmp_nEvals_history)):
                last_ = min(tmp_nEvals_history[j], maxEvals)
                for m in range(tmp_nEvals_history[j - 1] + 1, last_):
                    nEvals_history.append(m)
                    IGD_history.append(tmp_IGD_history[j - 1])

                if tmp_nEvals_history[j] <= maxEvals:
                    nEvals_history.append(tmp_nEvals_history[j])
                    IGD_history.append(tmp_IGD_history[j])

            for m in range(nEvals_history[-1] + 1, maxEvals + 1):
                nEvals_history.append(m)
                IGD_history.append(IGD_history[-1])

            nEvals_history = np.array(nEvals_history)[idx_nEvals_history]
            IGD_history = np.array(IGD_history)[idx_nEvals_history]
            new_nEvals_history = np.arange(pop_size, maxEvals + 1)
            f1 = interp1d(nEvals_history, IGD_history)
            new_IGD_history = f1(new_nEvals_history)

            IGD_each_run.append(np.array(new_IGD_history))

            nEvals_each_run.append(np.array(new_nEvals_history))

        IGD_each_run = np.array(IGD_each_run)
        nEvals_each_run = np.array(nEvals_each_run)

        p.dump([nEvals_each_run, IGD_each_run],
               open(os.path.join(PATH_RESULTS, 'nEvals_IGD', exp.split('\\')[-1] + '_IGD.p'), 'wb'))

        IGD_mean = np.mean(IGD_each_run, axis=0)
        IGD_stdev = np.std(IGD_each_run, axis=0)
        nEvals_mean = np.mean(nEvals_each_run, axis=0, dtype=int)

        IGD_mean_each_exp.append(IGD_mean)
        IGD_stdev_each_exp.append(IGD_stdev)

        nEvals_mean_each_exp.append(nEvals_mean)
    return [nEvals_mean_each_exp,
            IGD_mean_each_exp, IGD_stdev_each_exp]


def visualize_nEvals_and_IGD(logX=None, logY=None):
    try:
        os.mkdir(os.path.join(PATH_RESULTS, 'nEvals_IGD'))
    except:
        pass

    for exp in experiments_list:
        if '.' not in exp:
            if platform == "linux" or platform == "linux2":
                hyperparameters = exp.split('/')[-1][:-20].split('_')
            else:
                hyperparameters = exp.split('//')[-1][:-20].split('_')
        else:
            continue
    population_size = int(hyperparameters[2])

    nEvals_and_IGD = get_nEvals_and_IGD(population_size)

    nEvals_mean_each_exp = nEvals_and_IGD[0]
    IGD_mean_each_exp, IGD_stdev_each_exp = nEvals_and_IGD[1], nEvals_and_IGD[2]

    fig, ax = plt.subplots(1)
    axis_lbs = ['#Evals', 'IGD']

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
        if variant_configuration[0] == 'True':
            label += f' w/ Warmup (M = {int(variant_configuration[1])})'
        line = line_dict[label]
        visualize_2D(objective_0_mean=nEvals_mean_each_exp[i],
                     tmp_objective_1_mean=IGD_mean_each_exp[i], tmp_objective_1_stdev=IGD_stdev_each_exp[i],
                     axis_labels=axis_lbs, label=label, line=line)
    plt.grid(linestyle='--')
    if logX is None:
        logX = LOG_X
    if logY is None:
        logY = LOG_Y
    if logX:
        plt.xscale('log')
    if logY:
        plt.yscale('log')
    for label in (ax.get_xticklabels()):
        label.set_fontsize(8)
    plt.legend(bbox_to_anchor=(-0.15, -0.06, 1.3, .02), mode='expand', fontsize=8, ncol=2, frameon=False)

    title = f'{problem_name} | x-axis: #Evals'
    figure_name = f'{PATH_RESULTS}/IGD.png'
    plt.title(title, fontsize=12)
    plt.savefig(figure_name, dpi=300)
    plt.clf()

''' ---------------------------------- Visualize "nEvals" and "Hypervolume" ---------------------------------------- '''
def get_reference_point():
    max_f0, max_f1 = -np.inf, -np.inf
    for exp in experiments_list:
        n_runs = len(['_' for f in os.listdir(exp) if '.' not in f])
        for i in range(n_runs):
            f_reference_pt = os.path.join(exp, f'{i}', 'reference_point(evaluate).p')
            f0, f1 = p.load(open(f_reference_pt, 'rb'))
            max_f0 = max(max_f0, f0)
            max_f1 = max(max_f1, f1)

    reference_point = [max_f0 + 1e-5, max_f1 + 1e-5]
    return reference_point


def calculate_Hypervolume_value(hypervolume_calculator, non_dominated_front, reference_point):
    hypervolume = hypervolume_calculator.compute(non_dominated_front)
    return hypervolume / np.prod(reference_point)


def get_nEvals_and_Hypervolume(hypervolume_calculator, reference_point, pop_size):
    hypervolume_mean_each_exp = []
    hypervolume_stdev_each_exp = []

    nEvals_mean_each_exp = []

    for exp in experiments_list:
        hypervolume_each_run = []
        nEvals_each_run = []

        n_runs = len(['_' for f in os.listdir(exp) if '.' not in f])
        for i in range(n_runs):
            path_result = os.path.join(exp, f'{i}')
            tmp_data = np.array(p.load(open(os.path.join(path_result, '#Evals_and_IGD_each_gen.p'), 'rb')))
            idx_nEvals_history = np.array(tmp_data[0], dtype=int) - 1
            if idx_nEvals_history[-1] > maxEvals - 1:
                idx_nEvals_history[-1] = maxEvals - 1

            tmp_nEvals_history, tmp_EA_history = p.load(open(os.path.join(path_result, '#Evals_and_Elitist_Archive_evaluate.p'), 'rb'))

            tmp_Hypervolume_history = []
            for j in range(len(tmp_EA_history)):
                non_dominated_front_testing_C10 = np.unique(np.array(tmp_EA_history[j][2]), axis=0)
                hypervolume_value = calculate_Hypervolume_value(hypervolume_calculator,
                                                                non_dominated_front_testing_C10,
                                                                reference_point)
                tmp_Hypervolume_history.append(hypervolume_value)

            nEvals_history, Hypervolume_history = [tmp_nEvals_history[0]], [tmp_Hypervolume_history[0]]

            for j in range(1, len(tmp_nEvals_history)):
                last_ = min(tmp_nEvals_history[j], maxEvals)
                for m in range(tmp_nEvals_history[j - 1] + 1, last_):
                    nEvals_history.append(m)
                    Hypervolume_history.append(tmp_Hypervolume_history[j - 1])

                if tmp_nEvals_history[j] <= maxEvals:
                    nEvals_history.append(tmp_nEvals_history[j])
                    Hypervolume_history.append(tmp_Hypervolume_history[j])
            for m in range(nEvals_history[-1] + 1, maxEvals + 1):
                nEvals_history.append(m)
                Hypervolume_history.append(tmp_Hypervolume_history[-1])

            nEvals_history = np.array(nEvals_history)[idx_nEvals_history]
            Hypervolume_history = np.array(Hypervolume_history)[idx_nEvals_history]

            new_nEvals_history = np.arange(pop_size, maxEvals + 1)

            f1 = interp1d(nEvals_history, Hypervolume_history)
            new_Hypervolume_history = f1(new_nEvals_history)

            hypervolume_each_run.append(np.array(new_Hypervolume_history))
            nEvals_each_run.append(np.array(new_nEvals_history))

        hypervolume_each_run = np.array(hypervolume_each_run)
        nEvals_each_run = np.array(nEvals_each_run)

        p.dump([nEvals_each_run, hypervolume_each_run],
               open(os.path.join(PATH_RESULTS, 'nEvals_Hypervolume', exp.split('\\')[-1] + '_Hypervolume.p'), 'wb'))

        hypervolume_mean = np.mean(hypervolume_each_run, axis=0)
        hypervolume_stdev = np.std(hypervolume_each_run, axis=0)

        nEvals_mean = np.mean(nEvals_each_run, axis=0)

        hypervolume_mean_each_exp.append(hypervolume_mean)
        hypervolume_stdev_each_exp.append(hypervolume_stdev)

        nEvals_mean_each_exp.append(nEvals_mean)

    return [nEvals_mean_each_exp,
            hypervolume_mean_each_exp, hypervolume_stdev_each_exp]


def visualize_nEvals_and_Hypervolume(logX=None, logY=None):
    try:
        os.mkdir(os.path.join(PATH_RESULTS, 'nEvals_Hypervolume'))
    except:
        pass

    reference_point = get_reference_point()
    hypervolume_calculator = _HyperVolume(reference_point)

    for exp in experiments_list:
        if '.' not in exp:
            if platform == "linux" or platform == "linux2":
                hyperparameters = exp.split('/')[-1][:-20].split('_')
            else:
                hyperparameters = exp.split('//')[-1][:-20].split('_')
        else:
            continue
    population_size = int(hyperparameters[2])

    nEvals_and_Hypervolume = get_nEvals_and_Hypervolume(hypervolume_calculator, reference_point, population_size)
    nEvals_mean_each_exp = nEvals_and_Hypervolume[0]
    hypervolume_mean_each_exp, hypervolume_stdev_each_exp = nEvals_and_Hypervolume[1], nEvals_and_Hypervolume[2]

    fig, ax = plt.subplots(1)
    axis_lbs = ['#Evals', 'Hypervolume']

    for i, exp in enumerate(experiments_list):
        if '.' not in exp:
            if platform == "linux" or platform == "linux2":
                hyperparameters = exp.split('/')[-1][:-20].split('_')
            else:
                hyperparameters = exp.split('//')[-1][:-20].split('_')
        else:
            continue
        algorithm_name = hyperparameters[1]

        variant_configuration = hyperparameters[3:]
        label = f'{algorithm_name}'
        if variant_configuration[0] == 'True':
            label += f' w/ Warmup (M = {int(variant_configuration[1])})'
        line = line_dict[label]
        visualize_2D(objective_0_mean=nEvals_mean_each_exp[i],
                     tmp_objective_1_mean=hypervolume_mean_each_exp[i],
                     tmp_objective_1_stdev=hypervolume_stdev_each_exp[i],
                     axis_labels=axis_lbs, label=label, line=line)

    plt.grid(linestyle='--')
    if logX is None:
        logX = LOG_X
    if logY is None:
        logY = LOG_Y
    if logX:
        plt.xscale('log')
    if logY:
        plt.yscale('log')

    for label in (ax.get_xticklabels()):
        label.set_fontsize(8)
    plt.legend(bbox_to_anchor=(-0.15, -0.06, 1.3, .02), mode='expand', fontsize=8, ncol=2, frameon=False)

    title = f'{problem_name} | x-axis: #Evals'
    figure_name = f'{PATH_RESULTS}/Hypervolume.png'

    plt.title(title, fontsize=12)
    plt.savefig(figure_name, dpi=300)
    plt.clf()


''' ------------------------------------ Main ------------------------------------ '''


def main():
    visualize_nEvals_and_IGD()
    visualize_nEvals_and_Hypervolume()

if __name__ == '__main__':
    LOG_X = True
    LOG_Y = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_results', type=str)
    args = parser.parse_args()
    PATH_RESULTS = args.path_results
    checked_lst = ['IGD', 'Hypervolume', 'nEvals', 'png']
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

    if problem_name in ['MO-NAS201-1', 'MO-NAS201-2', 'MO-NAS201-3']:
        maxEvals = 3000
    else:
        maxEvals = 30000
    """ =========================================== """
    experiments_list = []
    for experiment in os.listdir(PATH_RESULTS):
        if any(word in experiment for word in checked_lst):
            continue
        else:
            experiments_list.append(os.path.join(PATH_RESULTS, experiment))
    """ =========================================== """
    main()