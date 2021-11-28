"""
Inspired: https://github.com/msu-coinlab/pymoo
"""
from algorithms import Algorithm

import pickle as p
from models import Population
from utils import get_hashKey, set_seed

from warm_up import get_population_X


class GeneticAlgorithm(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(name='GA', **kwargs)

    def _initialize(self):
        if not self.warm_up:
            P = self.sampling.do(self.problem)
            for i in range(self.pop_size):
                F = self.evaluate(P[i].X)
                P[i].set('F', F)
        else:
            if self.path_preprocessed_population is None:
                tmp_P = self.sampling.do(self.problem)
                P_X = get_population_X(tmp_P=tmp_P, pop_size=self.pop_size,
                                       problem=self.problem, zc_predictor=self.zero_cost_predictor)
                root_path = '/'.join(self.path_results.split('/')[:-1])
                path_preprocessed_population = root_path + '/population'
                p.dump(P_X, open(path_preprocessed_population + '/' + f'{self.problem.name}_{self.problem.dataset}_'
                                                                      f'population_{self.zero_cost_method}_'
                                                                      f'{self.nSamples_for_warm_up}_{self.seed}.p', 'wb'))
            else:
                P_X = p.load(
                    open(self.path_preprocessed_population + '/' + f'{self.problem.name}_{self.problem.dataset}_'
                                                                   f'population_{self.zero_cost_method}_'
                                                                   f'{self.nSamples_for_warm_up}_{self.seed}.p', 'rb')
                )
            set_seed(self.seed)
            P = Population(self.pop_size)
            for i, X in enumerate(P_X):
                hashKey = get_hashKey(X, problem_name=self.problem.name)
                F = self.evaluate(X)
                P[i].set('X', X)
                P[i].set('hashKey', hashKey)
                P[i].set('F', F)
        self.pop = P

if __name__ == '__main__':
    pass
