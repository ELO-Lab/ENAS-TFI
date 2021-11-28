import numpy as np

from .mutation import Mutation
from models.population import Population
from utils import get_hashKey, check_valid


class BitStringMutation(Mutation):
    def __init__(self):
        super().__init__(prob=1)

    def _do(self, problem, P, O, **kwargs):
        problem_name = problem.name

        P_hashKey = P.get('hashKey')
        O_old_X = O.get('X')

        offspring_size = len(O)
        len_X = len(O_old_X[-1])

        nMutations, maxMutations = 0, offspring_size * 5

        self.prob = 1 / len_X

        O_new = Population(offspring_size)
        O_new_hashKey = []

        n = 0
        while True:
            for X_old in O_old_X:
                o_X = X_old.copy()

                for m, prob in enumerate(np.random.rand(len_X)):
                    if prob <= self.prob:
                        if problem_name == 'NASBench101':
                            if m == 0 or m == 21:
                                continue
                            elif m in problem.IDX_OPS:
                                available_ops = problem.OPS.copy()
                            else:
                                available_ops = problem.EDGES.copy()
                        else:
                            available_ops = problem.available_ops.copy()
                        available_ops.remove(o_X[m])
                        new_op = np.random.choice(available_ops)
                        o_X[m] = new_op

                if problem.isValid(o_X):
                    o_hashKey = get_hashKey(o_X, problem_name)
                    if kwargs['algorithm'].problem.type_of_problem == 'single-objective':
                        O_new_hashKey = []
                        P_hashKey = []
                        kwargs['algorithm'].E_Archive_search.DS = []

                    if check_valid(o_hashKey,
                                   O=O_new_hashKey, P=P_hashKey, DS=kwargs['algorithm'].E_Archive_search.DS) \
                            or (nMutations - maxMutations > 0):
                        O_new_hashKey.append(o_hashKey)

                        o_F = kwargs['algorithm'].evaluate(o_X)
                        O_new[n].set('X', o_X)
                        O_new[n].set('hashKey', o_hashKey)
                        O_new[n].set('F', o_F)

                        if kwargs['algorithm'].problem.type_of_problem == 'multi-objective':
                            kwargs['algorithm'].E_Archive_search.update(O_new[n], algorithm=kwargs['algorithm'])

                        n += 1
                        if n - offspring_size == 0:
                            return O_new
            nMutations += 1


if __name__ == '__main__':
    pass
