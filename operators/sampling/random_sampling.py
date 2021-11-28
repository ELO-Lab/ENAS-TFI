from models import Population
from utils import get_hashKey


class RandomSampling:
    def __init__(self, nSamples=0):
        self.nSamples = nSamples

    def do(self, problem, **kwargs):
        problem_name = problem.name
        P = Population(self.nSamples)
        n = 0

        P_hashKey = []
        while n < self.nSamples:
            X = problem.sample_a_compact_architecture()
            if problem.isValid(X):
                hashKey = get_hashKey(X, problem_name)
                if hashKey not in P_hashKey:
                    P[n].set('X', X)
                    P[n].set('hashKey', hashKey)
                    n += 1
        return P


if __name__ == '__main__':
    # import numpy as np
    # np.random.seed(1)
    # sampling = RandomSampling(10)
    #
    # from problems import NASBench201
    # dataset = 'CIFAR-10'
    # path_data = 'D:/Files/BENCHMARKS'
    # problem = NASBench201(dataset=dataset, maxEvals=100, problem_property='single', path_data=path_data)
    # problem.set_up()
    # pop = sampling.do(problem=problem)
    # print(pop.get('X'))
    pass
