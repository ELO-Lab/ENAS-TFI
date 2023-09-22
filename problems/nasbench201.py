import pickle as p
import numpy as np
from problems.NAS_problem import Problem


def get_key_in_data(arch):
    """
    Get the key which is used to represent the architecture in "self.data".
    """
    return ''.join(map(str, arch))


class NASBench201(Problem):
    def __init__(self, dataset, maxEvals, **kwargs):
        """
        # NAS-Benchmark-201 provides us with the information (e.g., the training loss, the testing accuracy,
        the validation accuracy, the number of FLOPs, etc) of all architectures in the search space. Therefore, if we
        want to evaluate any architectures in the search space, we just need to query its information in the data.\n
        -----------------------------------------------------------------

        - path_data -> the path contains NAS-Bench-201 data.
        - data -> NAS-Bench-201 data.
        - available_ops -> the available operators can choose in the search space.
        - maxLength -> the maximum length of compact architecture.
        """
        super().__init__(maxEvals, 'NASBench201', dataset, **kwargs)

        self.type_of_problem = kwargs['type_of_problem']
        if self.type_of_problem == 'single-objective':
            self.objectives_lst = ['val_acc']
        elif self.type_of_problem == 'multi-objective':
            self.objectives_lst = ['FLOPs', 'val_error']
        else:
            raise ValueError()

        ''' ------- Additional Hyper-parameters ------- '''
        self.available_ops = [0, 1, 2, 3, 4]
        self.maxLength = 6

        self.path_data = kwargs['path_data'] + '/NASBench201'
        self.data = None

        self.best_arch = None

    def _get_accuracy(self, arch, final=False):
        """
        - Get the accuracy of architecture. E.g., the testing accuracy, the validation accuracy.
        """
        key = get_key_in_data(arch)
        if final:
            acc = self.data['200'][key]['test_acc']
        else:
            acc = self.data['200'][key]['val_acc']
        return acc

    def _get_complexity_metric(self, arch):
        """
        - In NAS-Bench-201 problem, the efficiency metric is nFLOPs.
        - The returned nFLOPs is normalized.
        """
        key = get_key_in_data(arch)
        nFLOPs = round((self.data['200'][key]['FLOPs'] - self.min_max['FLOPs']['min']) /
                      (self.min_max['FLOPs']['max'] - self.min_max['FLOPs']['min']), 6)
        return nFLOPs

    def _set_up(self):
        available_datasets = ['CIFAR-10', 'CIFAR-100', 'ImageNet16-120']
        if self.dataset not in available_datasets:
            raise ValueError(f'Just only supported these datasets: CIFAR-10; CIFAR-100; ImageNet16-120.'
                             f'{self.dataset} dataset is not supported at this time.')

        f_data = open(f'{self.path_data}/[{self.dataset}]_data.p', 'rb')
        self.data = p.load(f_data)
        f_data.close()

        if self.type_of_problem == 'single-objective':
            # f_best_arch = open(f'{self.path_data}/[{self.dataset}]_best_arch.p', 'rb')
            self.best_arch = None
            # f_best_arch.close()
        elif self.type_of_problem == 'multi-objective':
            f_min_max = open(f'{self.path_data}/[{self.dataset}]_min_max.p', 'rb')
            self.min_max = p.load(f_min_max)
            f_min_max.close()

            f_pareto_front_testing = open(f'{self.path_data}/[{self.dataset}]_pareto_front(testing).p', 'rb')
            self.pareto_front_testing = p.load(f_pareto_front_testing)
            f_pareto_front_testing.close()

            f_pareto_front_validation = open(f'{self.path_data}/[{self.dataset}]_pareto_front(validation).p', 'rb')
            self.pareto_front_validation = p.load(f_pareto_front_validation)
            f_pareto_front_validation.close()

        print('--> Set Up - Done')

    def _get_a_compact_architecture(self):
        return np.random.choice(self.available_ops, self.maxLength)

    def _evaluate(self, arch):
        acc = self.get_accuracy(arch)
        if self.type_of_problem == 'single-objective':
            return acc
        elif self.type_of_problem == 'multi-objective':
            complex_metric = self.get_complexity_metric(arch)
            return [complex_metric, 1 - acc]

    def _isValid(self, arch):
        return True


if __name__ == '__main__':
    # dataset = 'CIFAR-10'
    # path_data = 'D:/Files/BENCHMARKS'
    # problem = NASBench201(dataset=dataset, maxEvals=100, property='single', path_data=path_data)
    # problem.set_up()
    # print(problem.name, problem.property, problem.objective_0, problem.objective_1)
    # print(problem.best_arch)
    # print(problem.pareto_front)
    # X = problem.sample_a_compact_architecture()
    # print(X)
    # F = problem.evaluate(X)
    # print(F)
    pass
