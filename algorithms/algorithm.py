"""
Inspired: https://github.com/msu-coinlab/pymoo
"""
import numpy as np

from utils import (
    get_hashKey,
    set_seed,
    ElitistArchive,
    calculate_IGD_value,
    do_each_gen,
    finalize,
)
from models import Population

class Algorithm:
    def __init__(self, **kwargs):
        """
        List of general hyperparameters:\n
        - *name*: the algorithm name
        - *pop_size*: the population size
        - *sampling*: the sampling processor
        - *crossover*: the crossover processor
        - *mutation*: the mutation processor
        - *survival*: the survival processor
        - *pop*: the population
        - *problem*: the problem is being solved
        - *seed*: random seed
        - *nGens*: the number of generations was passed
        - *nEvals*: the number of evaluate function calls (or the number of trained architectures (in NAS problems))
        - *path_results*: the folder where the results will be saved on
        - *IGD_history*: list of IGD value each generation (for MONAS problems)
        - *nEvals_history*: list of the number of trained architectures each generation
        - *reference_point*: the reference point (for calculation Hypervolume indicator) (for MONAS problems)
        - *E_Archive*: the Elitist Archive (for MONAS problems)
        """
        # General hyperparameters
        self.name = kwargs['name']

        self.pop_size = None
        self.sampling = None
        self.crossover = None
        self.mutation = None
        self.survival = None

        self.pop = None
        self.problem = None

        self.seed = 0
        self.nGens = 0
        self.nEvals = 0

        self.path_results = None
        self.debug = False

        """ [Method] - Warmup """
        self.zero_cost_method = None
        self.zero_cost_predictor = None
        self.path_data_zero_cost_method = None

        self.warm_up = None
        self.nSamples_for_warm_up = None
        self.path_preprocessed_population = None

        # SONAS problems
        self.nGens_history = []
        self.best_F_history = []
        self.pop_history = []
        self.best_arch_history = []

        # MONAS problems
        self.IGD_history_each_gen = []
        self.nEvals_history_each_gen = []

        self.reference_point_search = [-np.inf, -np.inf]
        self.reference_point_evaluate = [-np.inf, -np.inf]

        self.E_Archive_search = None
        self.E_Archive_evaluate = None

        self.nEvals_history = []
        self.E_Archive_history_search = []
        self.E_Archive_history_evaluate = []
        self.IGD_history_search = []
        self.IGD_history_evaluate = []
    """ ---------------------------------- Evaluate ---------------------------------- """
    def evaluate(self, X):
        """
        - Call function *problem.evaluate* to evaluate the fitness values of solutions.
        """
        F = self.problem.evaluate(X)
        self.nEvals += 1
        return F

    """ ---------------------------------- Initialize ---------------------------------- """
    def initialize(self):
        self._initialize()

    """ ---------------------------------- Mating ---------------------------------- """
    def mating(self, P):
        self._mating(P)

    """ ----------------------------------- Next ----------------------------------- """
    def next(self, pop):
        self._next(pop)

    """ ---------------------------------- Setting Up ---------------------------------- """
    def set_hyperparameters(self,
                            pop_size=None,
                            sampling=None,
                            crossover=None,
                            mutation=None,
                            survival=None,
                            **kwargs):
        self.pop_size = pop_size
        self.sampling = sampling
        self.crossover = crossover
        self.mutation = mutation
        self.survival = survival

        self.debug = kwargs['debug']

        """ [Method] - Warmup """
        self.zero_cost_method = kwargs['zero_cost_method']
        self.zero_cost_predictor = None
        self.path_data_zero_cost_method = kwargs['path_data_zero_cost_method']

        self.warm_up = kwargs['warm_up']
        self.nSamples_for_warm_up = kwargs['nSamples_for_warm_up']

    def reset(self):
        self.pop = None

        self.seed = 0
        self.nGens = 0
        self.nEvals = 0

        self.path_results = None

        # SONAS problems
        self.nGens_history = []
        self.best_F_history = []
        self.pop_history = []
        self.best_arch_history = []

        # MONAS problems
        self.IGD_history_each_gen = []
        self.nEvals_history_each_gen = []

        self.reference_point_search = [-np.inf, -np.inf]
        self.reference_point_evaluate = [-np.inf, -np.inf]

        self.E_Archive_search = ElitistArchive()
        self.E_Archive_evaluate = ElitistArchive(log_each_change=False)

        self.nEvals_history = []
        self.E_Archive_history_search = []
        self.E_Archive_history_evaluate = []
        self.IGD_history_search = []
        self.IGD_history_evaluate = []

        self._reset()

    def set_up(self, problem, seed):
        self.problem = problem
        self.seed = seed
        set_seed(self.seed)

        self.sampling.nSamples = self.pop_size

        if self.warm_up:
            self.sampling.nSamples = self.nSamples_for_warm_up
            from zero_cost_methods import get_config_for_zero_cost_predictor, get_zero_cost_predictor
            config = get_config_for_zero_cost_predictor(problem=problem, seed=seed,
                                                        path_data=self.path_data_zero_cost_method)
            self.zero_cost_predictor = get_zero_cost_predictor(config=config, method_type=self.zero_cost_method)

    """ ---------------------------------- Solving ---------------------------------- """
    def solve(self, problem, seed):
        self.set_up(problem, seed)
        self._solve()

    """ -------------------------------- Do Each Gen -------------------------------- """
    def do_each_gen(self):
        """
        Operations which the algorithm perform at the end of each generation.
        """
        self._do_each_gen()
        do_each_gen(type_of_problem=self.problem.type_of_problem, algorithm=self)
        if self.debug:
            print(f'{self.nEvals}/{self.problem.maxEvals}')
            print(f'{self.IGD_history_each_gen[-1]}')
            print('='*40)

    """ -------------------------------- Do each having change in EA -------------------------------- """
    def log_elitist_archive(self):  # For solving MONAS problems
        non_dominated_front = np.array(self.E_Archive_search.F)
        non_dominated_front = np.unique(non_dominated_front, axis=0)

        self.nEvals_history.append(self.nEvals)

        IGD_value_search = calculate_IGD_value(pareto_front=self.problem.pareto_front_validation,
                                               non_dominated_front=non_dominated_front)

        self.IGD_history_search.append(IGD_value_search)
        self.E_Archive_history_search.append([self.E_Archive_search.X.copy(),
                                              self.E_Archive_search.hashKey.copy(),
                                              self.E_Archive_search.F.copy()])

        size = len(self.E_Archive_search.X)
        tmp_pop = Population(size)
        for i in range(size):
            X = self.E_Archive_search.X[i]
            hashKey = get_hashKey(X, self.problem.name)
            F = [self.problem.get_complexity_metric(X), 1 - self.problem.get_accuracy(X, final=True)]
            tmp_pop[i].set('X', X)
            tmp_pop[i].set('hashKey', hashKey)
            tmp_pop[i].set('F', F)
            self.E_Archive_evaluate.update(tmp_pop[i])

        non_dominated_front = np.array(self.E_Archive_evaluate.F)
        non_dominated_front = np.unique(non_dominated_front, axis=0)

        self.reference_point_evaluate[0] = max(self.reference_point_evaluate[0], max(non_dominated_front[:, 0]))
        self.reference_point_evaluate[1] = max(self.reference_point_evaluate[1], max(non_dominated_front[:, 1]))

        IGD_value_evaluate = calculate_IGD_value(pareto_front=self.problem.pareto_front_testing,
                                                 non_dominated_front=non_dominated_front)

        self.IGD_history_evaluate.append(IGD_value_evaluate)
        self.E_Archive_history_evaluate.append([self.E_Archive_evaluate.X.copy(),
                                                self.E_Archive_evaluate.hashKey.copy(),
                                                self.E_Archive_evaluate.F.copy()])
        self.E_Archive_evaluate = ElitistArchive(log_each_change=False)

    """ -------------------------------- Finalize -------------------------------- """
    def finalize(self):
        self._finalize()
        finalize(type_of_problem=self.problem.type_of_problem, algorithm=self)

    """ -------------------------------------------- Abstract Methods -----------------------------------------------"""
    def _solve(self):
        self.initialize()
        self.do_each_gen()
        while self.nEvals < self.problem.maxEvals:
            self.nGens += 1
            self.next(self.pop)
            self.do_each_gen()
        self.finalize()

    def _initialize(self):
        pass

    def _reset(self):
        pass

    def _mating(self, P):
        O = self.crossover.do(self.problem, P, algorithm=self)
        O = self.mutation.do(self.problem, P, O, algorithm=self)
        return O

    def _next(self, pop):
        offsprings = self._mating(pop)
        pool = pop.merge(offsprings)
        pop = self.survival.do(pool, self.pop_size)
        self.pop = pop

    def _do_each_gen(self):
        pass

    def _finalize(self):
        pass

if __name__ == '__main__':
    pass
