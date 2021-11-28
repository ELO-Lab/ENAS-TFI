from problems import NASBench101, NASBench201
from algorithms import NSGAII, GeneticAlgorithm

problem_configuration = {
    'SO-NAS101': {
        'maxEvals': 5000,
        'dataset': 'CIFAR-10',
        'type_of_problem': 'single-objective'
    },
    'SO-NAS201-1': {
        'maxEvals': 1000,
        'dataset': 'CIFAR-10',
        'type_of_problem': 'single-objective'
    },
    'SO-NAS201-2': {
        'maxEvals': 1000,
        'dataset': 'CIFAR-100',
        'type_of_problem': 'single-objective'
    },
    'SO-NAS201-3': {
        'maxEvals': 1000,
        'dataset': 'ImageNet16-120',
        'type_of_problem': 'single-objective'
    },
    'MO-NAS101': {
        'maxEvals': 30000,
        'dataset': 'CIFAR-10',
        'type_of_problem': 'multi-objective'
    },
    'MO-NAS201-1': {
        'maxEvals': 3000,
        'dataset': 'CIFAR-10',
        'type_of_problem': 'multi-objective'
    },
    'MO-NAS201-2': {
        'maxEvals': 3000,
        'dataset': 'CIFAR-100',
        'type_of_problem': 'multi-objective'
    },
    'MO-NAS201-3': {
        'maxEvals': 3000,
        'dataset': 'ImageNet16-120',
        'type_of_problem': 'multi-objective'
    }
}

def get_problem(problem_name, **kwargs):
    config = problem_configuration[problem_name]
    if 'NAS101' in problem_name:
        return NASBench101(maxEvals=config['maxEvals'], dataset=config['dataset'], type_of_problem=config['type_of_problem'], **kwargs)
    elif 'NAS201' in problem_name:
        return NASBench201(maxEvals=config['maxEvals'], dataset=config['dataset'], type_of_problem=config['type_of_problem'], **kwargs)
    else:
        raise ValueError(f'Not supporting this problem - {problem_name}.')

def get_algorithm(algorithm_name, **kwargs):
    if algorithm_name == 'GA':
        return GeneticAlgorithm()
    elif algorithm_name == 'NSGA-II':
        return NSGAII()
    else:
        raise ValueError(f'Not supporting this algorithm - {algorithm_name}.')


if __name__ == '__main__':
    pass