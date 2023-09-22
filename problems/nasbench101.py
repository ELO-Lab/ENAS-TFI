from problems.NAS_problem import Problem
import numpy as np
import pickle as p

from benchmark_api.nasbench import wrap_api as api

"""
    0: CONV 1x1
    1: CONV 3x3
    2: MAXPOOL 3x3
"""
class NASBench101(Problem):
    def __init__(self, maxEvals, dataset='CIFAR-10', **kwargs):
        """
        # NAS-Benchmark-101 provides us the information (e.g., the testing accuracy, the validation accuracy,
        the number of parameters) of all architectures in the search space. Therefore, if we want to evaluate any
        architectures in the search space, we just need to query its information in the data.\n
        -----------------------------------------------------------------

        - path_data -> the path contains NAS-Bench-101 data.
        - data -> NAS-Bench-101 data.
        - OPS -> the available operators can choose in the search space.
        - IDX_OPS -> the index of operators in compact architecture.
        - EDGES -> 0: doesn't have edge; 1: have edge.
        - IDX_EDGES -> the index of edges in compact architecture.
        - maxLength -> the maximum length of compact architecture.
        """

        if maxEvals is None:
            maxEvals = 30000
        super().__init__(maxEvals, 'NASBench101', dataset, **kwargs)

        self.type_of_problem = kwargs['type_of_problem']
        if self.type_of_problem == 'single-objective':
            self.objectives_lst = ['val_acc']
        elif self.type_of_problem == 'multi-objective':
            self.objectives_lst = ['nParams', 'val_error']
        else:
            raise ValueError()

        ''' ------- Additional Hyper-parameters ------- '''
        self.OPS = [2, 3, 4]
        self.IDX_OPS = [1, 3, 6, 10, 15]

        self.EDGES = [0, 1]
        self.IDX_EDGES = [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27]

        self.maxLength = 28
        self.api = api.NASBench_()

        self.path_data = kwargs['path_data'] + '/NASBench101'
        self.data = None
        self.best_arch = None

    def _get_accuracy(self, arch, final=False):
        """
        - Get the accuracy of architecture. E.g., the testing accuracy, the validation accuracy.
        """
        key = self.get_key_in_data(arch)
        if final:
            acc = self.data['108'][key]['test_acc']
        else:
            acc = self.data['108'][key]['val_acc']
        return acc

    def _get_complexity_metric(self, arch):
        """
        - In NAS-Bench-101 problem, the efficiency metric is nParams.
        - The returned nParams is normalized.
        """
        key = self.get_key_in_data(arch)
        nParams = round((self.data['108'][key]['n_params'] - self.min_max['n_params']['min']) /
                      (self.min_max['n_params']['max'] - self.min_max['n_params']['min']), 6)
        return nParams

    def _set_up(self):
        f_data = open(f'{self.path_data}/data.p', 'rb')
        self.data = p.load(f_data)
        f_data.close()

        if self.type_of_problem == 'single-objective':
            # f_best_arch = open(f'{self.path_data}/best_arch.p', 'rb')
            self.best_arch = None
            # f_best_arch.close()

        elif self.type_of_problem == 'multi-objective':
            f_min_max = open(f'{self.path_data}/min_max.p', 'rb')
            self.min_max = p.load(f_min_max)
            f_min_max.close()

            f_pareto_front_testing = open(f'{self.path_data}/pareto_front(testing).p', 'rb')
            self.pareto_front_testing = p.load(f_pareto_front_testing)
            f_pareto_front_testing.close()

            f_pareto_front_validation = open(f'{self.path_data}/pareto_front(validation).p', 'rb')
            self.pareto_front_validation = p.load(f_pareto_front_validation)
            f_pareto_front_validation.close()

        else:
            raise ValueError()

        print('--> Set Up - Done')

    def X2matrices(self, X):
        edges_matrix = np.zeros((7, 7), dtype=np.int8)
        for row in range(6):
            idx_list = None
            if row == 0:
                idx_list = [2, 4, 7, 11, 16, 22]
            elif row == 1:
                idx_list = [5, 8, 12, 17, 23]
            elif row == 2:
                idx_list = [9, 13, 18, 24]
            elif row == 3:
                idx_list = [14, 19, 25]
            elif row == 4:
                idx_list = [20, 26]
            elif row == 5:
                idx_list = [27]
            for i, edge in enumerate(idx_list):
                if X[edge] - 1 == 0:
                    edges_matrix[row][row + i + 1] = 1

        ops_matrix = ['input']
        for i in self.IDX_OPS:
            if X[i] == 2:
                ops_matrix.append('conv1x1-bn-relu')
            elif X[i] == 3:
                ops_matrix.append('conv3x3-bn-relu')
            else:
                ops_matrix.append('maxpool3x3')
        ops_matrix.append('output')

        return edges_matrix, ops_matrix

    def get_key_in_data(self, X):
        edges_matrix, ops_matrix = self.X2matrices(X)
        model_spec = api.ModelSpec(edges_matrix, ops_matrix)
        key = self.api.get_module_hash(model_spec)
        return key

    def _get_a_compact_architecture(self):
        arch = np.zeros(self.maxLength, dtype=np.int8)
        arch[self.IDX_OPS] = np.random.choice(self.OPS, len(self.IDX_OPS))
        arch[self.IDX_EDGES] = np.random.choice(self.EDGES, len(self.IDX_EDGES))
        arch[0] = 1
        arch[21] = 5
        return arch

    def _evaluate(self, arch):
        acc = self.get_accuracy(arch)
        if self.type_of_problem == 'single-objective':
            return acc
        elif self.type_of_problem == 'multi-objective':
            complex_metric = self.get_complexity_metric(arch)
            return [complex_metric, 1 - acc]

    def _isValid(self, X):
        edges_matrix, ops_matrix = self.X2matrices(X)
        model_spec = api.ModelSpec(edges_matrix, ops_matrix)
        return self.api.is_valid(model_spec)


if __name__ == '__main__':
    pass
