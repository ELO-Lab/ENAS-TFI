import os
import time
import argparse
from datetime import datetime

from factory import get_problem, get_algorithm
from operators.crossover import PointCrossover
from operators.mutation import BitStringMutation
from operators.sampling.random_sampling import RandomSampling
from operators.selection import TournamentSelection, RankAndCrowdingSurvival
from sys import platform

population_size_dict = {
    'SO-NAS101': 100, 'SO-NAS201-1': 40, 'SO-NAS201-2': 40, 'SO-NAS201-3': 40,
    'MO-NAS101': 100, 'MO-NAS201-1': 20, 'MO-NAS201-2': 20, 'MO-NAS201-3': 20,
}

def main(kwargs):
    if platform == "linux" or platform == "linux2":
        root_project = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    elif platform == "win32" or platform == "win64":
        root_project = '\\'.join(os.path.abspath(__file__).split('\\')[:-1])
    else:
        raise ValueError()

    if kwargs.path_results is None:
        try:
            os.makedirs(f'{root_project}/results/{kwargs.problem_name}')
        except FileExistsError:
            pass
        PATH_RESULTS = f'{root_project}/results/{kwargs.problem_name}'
    else:
        try:
            os.makedirs(f'{kwargs.path_results}/{kwargs.problem_name}')
        except FileExistsError:
            pass
        PATH_RESULTS = f'{kwargs.path_results}/{kwargs.problem_name}'

    path_data = root_project + '/data'
    problem = get_problem(problem_name=kwargs.problem_name, path_data=path_data)
    problem.set_up()
    ''' ==================================================================================================== '''
    pop_size = population_size_dict[kwargs.problem_name]

    n_runs = kwargs.n_runs
    init_seed = kwargs.seed

    warm_up = bool(kwargs.warm_up)
    nSamples_for_warm_up = kwargs.nSamples_for_warm_up
    zero_cost_method = 'synflow'

    sampling = RandomSampling()
    crossover = PointCrossover('2X')
    mutation = BitStringMutation()

    algorithm = get_algorithm(algorithm_name=kwargs.algorithm_name)
    if algorithm.name == 'GA':
        survival = TournamentSelection(k=4)
    elif algorithm.name == 'NSGA-II':
        survival = RankAndCrowdingSurvival()
    else:
        raise ValueError()
    algorithm.set_hyperparameters(pop_size=pop_size,
                                  sampling=sampling,
                                  crossover=crossover,
                                  mutation=mutation,
                                  survival=survival,
                                  warm_up=warm_up,
                                  nSamples_for_warm_up=nSamples_for_warm_up,
                                  zero_cost_method=zero_cost_method,
                                  path_data_zero_cost_method=path_data,
                                  debug=bool(kwargs.debug))
    time_now = datetime.now()

    dir_name = time_now.strftime(
        f'{kwargs.problem_name}_{algorithm.name}_{pop_size}_'
        f'{warm_up}_{nSamples_for_warm_up}_'
        f'd%d_m%m_H%H_M%M_S%S')

    root_path = PATH_RESULTS + '/' + dir_name
    os.mkdir(root_path)
    print(f'--> Create folder {root_path} - Done\n')

    random_seeds_list = [init_seed + run * 100 for run in range(n_runs)]
    executed_time_list = []

    with open(f'{root_path}/logging.txt', 'w') as f:
        f.write(f'******* PROBLEM *******\n')
        f.write(f'- Benchmark: {problem.name}\n')
        f.write(f'- Dataset: {problem.dataset}\n')
        f.write(f'- Maximum number of evaluations: {problem.maxEvals}\n')
        f.write(f'- List of objectives: {problem.objectives_lst}\n\n')

        f.write(f'******* ALGORITHM *******\n')
        f.write(f'- Algorithm: {algorithm.name}\n')
        f.write(f'- Population size: {algorithm.pop_size}\n')
        f.write(f'- Crossover method: {algorithm.crossover.method}\n')
        f.write(f'- Mutation method: Bit-string\n')
        f.write(f'- Selection method: {algorithm.survival.name}\n\n')
        if warm_up:
            f.write(f'******* WARM UP METHOD *******\n')
            f.write(f'- Number of samples for warm up: {algorithm.nSamples_for_warm_up}\n\n')

        f.write(f'******* ENVIRONMENT *******\n')
        f.write(f'- Number of running experiments: {n_runs}\n')
        f.write(f'- Random seed each run: {random_seeds_list}\n')
        f.write(f'- Path for saving results: {root_path}\n')
        f.write(f'- Debug: {algorithm.debug}\n\n')

    for run_i in range(n_runs):
        print(f'---- Run {run_i + 1}/{n_runs} ----')
        random_seed = random_seeds_list[run_i]

        path_results = root_path + '/' + f'{run_i}'

        os.mkdir(path_results)
        s = time.time()

        algorithm.reset()
        algorithm.path_results = path_results
        algorithm.solve(problem, random_seed)
        executed_time = time.time() - s
        executed_time_list.append(executed_time)
        print('This run take', executed_time_list[-1], 'seconds')
    print('==' * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' PROBLEM '''
    parser.add_argument('--problem_name', type=str, default='MO-NAS201-1', help='the problem name',
                        choices=['SO-NAS101', 'SO-NAS201-1', 'SO-NAS201-2', 'SO-NAS201-3', 'MO-NAS101', 'MO-NAS201-1',
                                 'MO-NAS201-2', 'MO-NAS201-3'])
    ''' EVOLUTIONARY ALGORITHM '''
    parser.add_argument('--algorithm_name', type=str, default='NSGA-II', help='the algorithm name', choices=['GA', 'NSGA-II'])

    ''' WARM-UP '''
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--nSamples_for_warm_up', type=int, default=0)

    ''' ENVIRONMENT '''
    parser.add_argument('--path_results', type=str, default=None, help='path for saving results')
    parser.add_argument('--n_runs', type=int, default=21, help='number of experiment runs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    args = parser.parse_args()

    main(args)
