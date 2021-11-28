class Problem:
    def __init__(self, maxEvals, name, dataset, **kwargs):
        """
        # Hyper-parameters:\n
        - *maxEvals* -> the maximum number of evaluated architecture.
        - *name* -> the name of used benchmark (or problem's name). E.g., MacroNAS, NAS-Bench-101, NAS-Bench-201, NAS-Bench-301.
        - *dataset* -> the dataset is used to train and evaluate architectures.
        - *type_of_problem* -> 'single-objective' or 'multi-objective'
        """
        self.maxEvals = maxEvals
        self.name = name
        self.dataset = dataset
        self.type_of_problem = None

    def set_up(self):
        """
        - Set up necessary things.
        """
        self._set_up()

    def sample_a_compact_architecture(self):
        """
        Sample a compact architecture in the search space.
        """
        return self._get_a_compact_architecture()

    def get_accuracy(self, X, final=False):
        return self._get_accuracy(X, final)

    def get_complexity_metric(self, X):
        return self._get_complexity_metric(X)

    def evaluate(self, arch):
        """
        Calculate the objective value.
        """
        return self._evaluate(arch)

    def isValid(self, arch):
        """
        - Checking if the architecture is valid or not.\n
        - NAS-Bench-101 doesn't provide information of all architecture in the search space. Therefore, when doing experiments on this benchmark, we need to check if the architecture is valid or not.\n
        """
        return self._isValid(arch)

    def _set_up(self):
        pass

    def _get_a_compact_architecture(self):
        raise NotImplementedError

    def _get_accuracy(self, X, final=False):
        raise NotImplementedError

    def _get_complexity_metric(self, X):
        raise NotImplementedError

    def _evaluate(self, X):
        raise NotImplementedError

    def _isValid(self, arch):
        raise NotImplementedError


if __name__ == '__main__':
    pass
