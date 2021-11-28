import numpy as np

from models.population import Population
from operators.crossover import Crossover
from utils import get_hashKey, check_valid


def crossover(parent_1, parent_2, typeC, **kwargs):
    problem_name = kwargs['problem_name']
    offspring_1, offspring_2 = parent_1.copy(), parent_2.copy()

    if typeC == '1X':  # 1-point crossover
        point = np.random.randint(1, len(parent_1))

        offspring_1[point:], offspring_2[point:] = offspring_2[point:], offspring_1[point:].copy()

    elif typeC == '2X':  # 2-point crossover
        if problem_name == 'NASBench301':
            points_list = np.random.choice(range(2, len(parent_1) - 1, 2), 2, replace=False)
        else:
            points_list = np.random.choice(range(1, len(parent_1) - 1), 2, replace=False)

        lower_pt = min(points_list)
        upper_pt = max(points_list)

        offspring_1[lower_pt: upper_pt], offspring_2[lower_pt: upper_pt] = \
            offspring_2[lower_pt: upper_pt], offspring_1[lower_pt: upper_pt].copy()

    elif typeC == 'UX':  # Uniform crossover
        pts = np.random.randint(0, 2, parent_1.shape, dtype=np.bool)
        offspring_1[pts], offspring_2[pts] = offspring_2[pts], offspring_1[pts].copy()

    return [offspring_1, offspring_2]


class PointCrossover(Crossover):
    def __init__(self, method=None):
        super().__init__(n_parents=2, prob=0.9)
        available_methods = ['1X', '2X', 'UX']
        if method not in available_methods:
            raise ValueError('Invalid crossover method: ' + method)
        else:
            self.method = method

    def _do(self, problem, P, **kwargs):
        problem_name = problem.name

        offspring_size = len(P)
        O = Population(offspring_size)
        O_hashKey = []

        n = 0
        nCrossovers, maxCrossovers = 0, offspring_size * 5

        while True:
            I = np.random.choice(offspring_size, size=(offspring_size // 2, self.n_parents), replace=False)
            P_ = P[I]
            for i in range(len(P_)):
                if np.random.random() < self.prob:
                    o_X = crossover(P_[i][0].X, P_[i][1].X, self.method, problem_name=problem_name)
                    for j, X in enumerate(o_X):
                        if problem.isValid(o_X[j]):
                            o_hashKey = get_hashKey(o_X[j], problem_name)
                            if kwargs['algorithm'].problem.type_of_problem == 'single-objective':
                                O_hashKey = []
                                kwargs['algorithm'].E_Archive_search.DS = []

                            if check_valid(o_hashKey,
                                           O=O_hashKey, DS=kwargs['algorithm'].E_Archive_search.DS) \
                                    or (nCrossovers - maxCrossovers > 0):
                                O_hashKey.append(o_hashKey)

                                if kwargs['algorithm'].problem.type_of_problem == 'single-objective':
                                    o_F = None
                                else:
                                    o_F = kwargs['algorithm'].evaluate(o_X[j])
                                O[n].set('X', o_X[j])
                                O[n].set('hashKey', o_hashKey)
                                O[n].set('F', o_F)

                                if kwargs['algorithm'].problem.type_of_problem == 'multi-objective':
                                    kwargs['algorithm'].E_Archive_search.update(O[n], algorithm=kwargs['algorithm'])

                                n += 1
                                if n - offspring_size == 0:
                                    return O
                else:
                    for o in P_[i]:
                        if kwargs['algorithm'].problem.type_of_problem == 'single-objective':
                            o_F = None
                        else:
                            o_F = o.F
                        O[n].set('X', o.X)
                        O[n].set('hashKey', o.hashKey)
                        O[n].set('F', o_F)
                        n += 1
                        if n - offspring_size == 0:
                            return O
            nCrossovers += 1


if __name__ == '__main__':
    pass
