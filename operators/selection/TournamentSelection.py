import numpy as np
from models.population import Population

class TournamentSelection:
    def __init__(self, k):
        self.k = k
        self.name = f'Tournament Selection k = {k}'

    def do(self, pool, pop_size):
        P = Population(pop_size)
        pool_size = len(pool)
        n = 0
        while True:
            I = np.random.choice(pool_size, size=(pool_size // self.k, self.k), replace=False)
            pool_ = pool[I]
            for i in range(len(pool_)):
                pool_F = pool_[i].get('F')
                idx_best = np.argmax(pool_F)
                P[n].set('X', pool_[i][idx_best].X)
                P[n].set('hashKey', pool_[i][idx_best].hashKey)
                P[n].set('F', pool_[i][idx_best].F)
                n += 1
                if n - pop_size == 0:
                    return P
