class Crossover:
    def __init__(self, n_parents, prob):
        self.n_parents = n_parents
        self.prob = prob

    def do(self, problem, pop, **kwargs):
        offsprings = self._do(problem, pop, **kwargs)
        return offsprings

    def _do(self, problem, pop, **kwargs):
        pass
