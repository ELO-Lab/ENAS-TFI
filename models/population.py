import numpy as np
from .individual import Individual


class Population(np.ndarray):
    def __new__(cls, n_individuals=0, individual=Individual(), *args, **kwargs):
        obj = super(Population, cls).__new__(cls, n_individuals, dtype=individual.__class__).view(cls)
        for i in range(n_individuals):
            obj[i] = individual.copy()
        obj.individual = individual
        return obj

    def merge(self, other):
        obj = np.concatenate([self, other]).view(Population)
        obj.individual = self.individual
        return obj

    def copy(self, order='C'):
        pop = Population(n_individuals=len(self), individual=self.individual)
        for i in range(len(self)):
            pop[i] = self[i]
        return pop

    def __deepcopy__(self, memo, *args, **kwargs):
        return self.copy()

    def new(self, *args):
        if len(args) == 1:
            return Population(n_individuals=args[0], individual=self.individual)
        else:
            n = len(args[1]) if len(args) > 0 else 0
            pop = Population(n_individuals=n, individual=self.individual)
            if len(args) > 0:
                pop.set(*args)
            return pop

    def collect(self, func, as_numpy_array=True):
        val = []
        for i in range(len(self)):
            val.append(func(self[i]))
        if as_numpy_array:
            val = np.array(val)
        return val

    def set(self, *args):
        for i in range(int(len(args) / 2)):
            key, values = args[i * 2], args[i * 2 + 1]
            if len(values) != len(self):
                raise Exception('Population Set Attribute Error: Number of values and population size do not match!')
            for j in range(len(values)):
                if key in self[j].__dict__:
                    self[j].__dict__[key] = values[j]
                else:
                    self[j].data[key] = values[j]

    def get(self, *args):
        val = {}
        for c in args:
            val[c] = []

        for i in range(len(self)):
            for c in args:
                if c in self[i].__dict__:
                    val[c].append(self[i].__dict__[c])
                elif c in self[i].data:
                    val[c].append(self[i].data[c])
        res = [val[c] for c in args]

        if len(args) == 1:
            return np.array(res[0])
        else:
            return tuple(res)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.individual = getattr(obj, 'individual', None)


if __name__ == '__main__':
    pass
