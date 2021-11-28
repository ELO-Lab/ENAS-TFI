import copy


class Individual:
    def __init__(self, **kwargs) -> None:
        self.X = None
        self.F = None
        self.hashKey = None
        self.data = kwargs

    def set(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        elif key in self.data:
            self.data[key] = value

    def copy(self):
        ind = copy.copy(self)
        ind.data = self.data.copy()
        return ind

    def get(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        if key in self.data:
            return self.data[key]
        return None


if __name__ == '__main__':
    pass
