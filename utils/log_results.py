import numpy as np
import matplotlib.pyplot as plt
import pickle as p
from utils import calculate_IGD_value

def save_reference_point(reference_point, path_results, error='None'):
    p.dump(reference_point, open(f'{path_results}/reference_point({error}).p', 'wb'))


def save_Non_dominated_Front_and_Elitist_Archive(non_dominated_front, n_evals, elitist_archive, n_gens, path_results):
    """
    - This function is used to save the non-dominated front and Elitist Archive at the end of each generation.
    """
    p.dump([non_dominated_front, n_evals], open(f'{path_results}/non_dominated_front/gen_{n_gens}.p', 'wb'))
    p.dump(elitist_archive, open(f'{path_results}/elitist_archive/gen_{n_gens}.p', 'wb'))


def visualize_IGD_value_and_nEvals(nEvals_history, IGD_history, path_results, error='search'):
    """
    - This function is used to visualize 'IGD_values' and 'nEvals' at the end of the search.
    """
    plt.xscale('log')
    plt.xlabel('#Evals')
    plt.ylabel('IGD value')
    plt.grid()
    plt.plot(nEvals_history, IGD_history)
    plt.savefig(f'{path_results}/#Evals-IGD({error})')
    plt.clf()

def visualize_Elitist_Archive_and_Pareto_Front(elitist_archive, pareto_front, objective_0, path_results, error='testing'):
    non_dominated_front = np.array(elitist_archive)
    non_dominated_front = np.unique(non_dominated_front, axis=0)

    plt.scatter(pareto_front[:, 0], pareto_front[:, 1],
                facecolors='none', edgecolors='b', s=40, label=f'Pareto-optimal Front')
    plt.scatter(non_dominated_front[:, 0], non_dominated_front[:, 1],
                c='red', s=15, label=f'Non-dominated Front')

    plt.xlabel(objective_0 + '(normalize)')
    plt.ylabel('Error')

    plt.legend()
    plt.grid()
    plt.savefig(f'{path_results}/non_dominated_front({error})')
    plt.clf()

def visualize_Elitist_Archive(elitist_archive, objective_0, path_results):
    non_dominated_front = np.array(elitist_archive)
    non_dominated_front = np.unique(non_dominated_front, axis=0)

    plt.scatter(non_dominated_front[:, 0], non_dominated_front[:, 1],
                facecolors='none', edgecolors='b', s=40, label=f'Non-dominated Front')

    plt.xlabel(objective_0 + '(normalize)')
    plt.ylabel('Error')

    plt.legend()
    plt.grid()
    plt.savefig(f'{path_results}/non_dominated_front')
    plt.clf()

def do_each_gen(type_of_problem, **kwargs):
    algorithm = kwargs['algorithm']
    if type_of_problem == 'single-objective':
        pop = {
            'X': algorithm.pop.get('X'),
            'hashKey': algorithm.pop.get('hashKey'),
            'F': algorithm.pop.get('F')
        }
        algorithm.pop_history.append(pop)

        F = algorithm.pop.get('F')
        best_arch_F = np.max(F)
        algorithm.best_F_history.append(best_arch_F)

        idx_best_arch = F == best_arch_F
        best_arch_X_list = np.unique(algorithm.pop.get('X')[idx_best_arch], axis=0)
        best_arch_list = []
        for arch_X in best_arch_X_list:
            arch_info = {
                'X': arch_X,
                'testing_accuracy': algorithm.problem.get_accuracy(arch_X, final=True),
                'validation_accuracy': algorithm.problem.get_accuracy(arch_X)
            }
            best_arch_list.append(arch_info)
        algorithm.best_arch_history.append(best_arch_list)
        algorithm.nGens_history.append(algorithm.nGens + 1)
    elif type_of_problem == 'multi-objective':
        non_dominated_front = np.array(algorithm.E_Archive_search.F)
        non_dominated_front = np.unique(non_dominated_front, axis=0)

        # Update reference point (use for calculating the Hypervolume value)
        algorithm.reference_point_search[0] = max(algorithm.reference_point_search[0], max(non_dominated_front[:, 0]))
        algorithm.reference_point_search[1] = max(algorithm.reference_point_search[1], max(non_dominated_front[:, 1]))

        IGD_value_search = calculate_IGD_value(pareto_front=algorithm.problem.pareto_front_validation,
                                               non_dominated_front=non_dominated_front)

        algorithm.nEvals_history_each_gen.append(algorithm.nEvals)
        algorithm.IGD_history_each_gen.append(IGD_value_search)

    else:
        raise ValueError(f'Not supported {type_of_problem} problem')

def finalize(type_of_problem, **kwargs):
    algorithm = kwargs['algorithm']
    if type_of_problem == 'single-objective':
        plt.xlim([0, algorithm.nGens_history[-1] + 2])
        plt.plot(algorithm.nGens_history, algorithm.best_F_history, c='blue')
        plt.scatter(algorithm.nGens_history, algorithm.best_F_history, c='black', s=12, label='Best Architecture')
        plt.xlabel('#Gens')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.savefig(f'{algorithm.path_results}/best_architecture_each_gen')
        plt.clf()

        p.dump([algorithm.nGens_history, algorithm.best_arch_history], open(f'{algorithm.path_results}/best_architecture_each_gen.p', 'wb'))
        p.dump([algorithm.nGens_history, algorithm.pop_history], open(f'{algorithm.path_results}/population_each_gen.p', 'wb'))

    elif type_of_problem == 'multi-objective':
        p.dump([algorithm.nEvals_history, algorithm.IGD_history_search], open(f'{algorithm.path_results}/#Evals_and_IGD_search.p', 'wb'))
        p.dump([algorithm.nEvals_history, algorithm.IGD_history_evaluate], open(f'{algorithm.path_results}/#Evals_and_IGD_evaluate.p', 'wb'))
        p.dump([algorithm.nEvals_history, algorithm.E_Archive_history_search], open(f'{algorithm.path_results}/#Evals_and_Elitist_Archive_search.p', 'wb'))
        p.dump([algorithm.nEvals_history, algorithm.E_Archive_history_evaluate], open(f'{algorithm.path_results}/#Evals_and_Elitist_Archive_evaluate.p', 'wb'))
        p.dump([algorithm.nEvals_history_each_gen, algorithm.IGD_history_each_gen], open(f'{algorithm.path_results}/#Evals_and_IGD_each_gen.p', 'wb'))

        save_reference_point(reference_point=algorithm.reference_point_search, path_results=algorithm.path_results,
                             error='search')
        save_reference_point(reference_point=algorithm.reference_point_evaluate, path_results=algorithm.path_results,
                             error='evaluate')

        visualize_Elitist_Archive_and_Pareto_Front(elitist_archive=algorithm.E_Archive_search.F,
                                                   pareto_front=algorithm.problem.pareto_front_validation,
                                                   objective_0=algorithm.problem.objectives_lst[0],
                                                   path_results=algorithm.path_results, error='search')

        visualize_Elitist_Archive_and_Pareto_Front(elitist_archive=algorithm.E_Archive_history_evaluate[-1][-1],
                                                   pareto_front=algorithm.problem.pareto_front_testing,
                                                   objective_0=algorithm.problem.objectives_lst[0],
                                                   path_results=algorithm.path_results, error='evaluate')

        visualize_IGD_value_and_nEvals(IGD_history=algorithm.IGD_history_search, nEvals_history=algorithm.nEvals_history,
                                       path_results=algorithm.path_results, error='search')

        visualize_IGD_value_and_nEvals(IGD_history=algorithm.IGD_history_evaluate, nEvals_history=algorithm.nEvals_history,
                                       path_results=algorithm.path_results, error='evaluate')
    else:
        raise ValueError(f'Not supported {type_of_problem} problem')
