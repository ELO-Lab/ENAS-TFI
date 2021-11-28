class Mutation:
    def __init__(self, prob):
        self.prob = prob

    def do(self, problem, pop, offsprings, **kwargs):
        offsprings = self._do(problem, pop, offsprings, **kwargs)
        return offsprings

    def _do(self, problem, pop, offsprings, **kwargs):
        pass
