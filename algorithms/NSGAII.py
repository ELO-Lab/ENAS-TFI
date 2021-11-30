"""
Inspired: https://github.com/msu-coinlab/pymoo
"""
import pickle as p
import os
from algorithms import Algorithm
from models.individual import Individual
from models.population import Population
from utils import get_hashKey, set_seed

from warm_up import get_population_X
from sys import platform


INF = 9999999
class NSGAII(Algorithm):
    def __init__(self, **kwargs):
        super().__init__(name='NSGA-II', **kwargs)
        self.individual = Individual(rank=INF, crowding=-1)

    def _reset(self):
        self.individual = Individual(rank=INF, crowding=-1)

    def _initialize(self):
        if not self.warm_up:
            P = self.sampling.do(self.problem)
            for i in range(self.pop_size):
                F = self.evaluate(P[i].X)
                P[i].set('F', F)
                self.E_Archive_search.update(P[i], algorithm=self)

        else:
            if platform == "linux" or platform == "linux2":
                path_data = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/prepopulation'
            elif platform == "win32" or platform == "win64":
                path_data = '\\'.join(os.path.abspath(__file__).split('\\')[:-2]) + '/prepopulation'
            else:
                raise ValueError()
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
                self.E_Archive_search.update(P[i], algorithm=self)
        self.pop = P

if __name__ == '__main__':
    pass
