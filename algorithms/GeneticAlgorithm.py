"""
Inspired: https://github.com/msu-coinlab/pymoo
"""
from algorithms import Algorithm

import pickle as p
import os
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
            path_data = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/prepopulation'
            if self.problem.type_of_problem == 'single-objective':
                path_data += '/SONAS'
            elif self.problem.type_of_problem == 'multi-objective':
                path_data += '/MONAS'
            else:
                raise ValueError()
            try:
                P_X = p.load(
                    open(path_data + '/' + f'{self.problem.name}_{self.problem.dataset}_'
                                           f'population_{self.zero_cost_method}_'
                                           f'{self.nSamples_for_warm_up}_{self.seed}.p', 'rb')
                )
            except FileNotFoundError:
                tmp_P = self.sampling.do(self.problem)
                P_X = get_population_X(tmp_P=tmp_P, pop_size=self.pop_size,
                                       problem=self.problem, zc_predictor=self.zero_cost_predictor)
                p.dump(P_X, open(path_data + '/' + f'{self.problem.name}_{self.problem.dataset}_'
                                                   f'population_{self.zero_cost_method}_'
                                                   f'{self.nSamples_for_warm_up}_{self.seed}.p', 'wb'))
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
