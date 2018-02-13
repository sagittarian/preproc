class Step(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return self.transform(data)

    def fit(self, data):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError



class Log(Step):
    pass


class Diff(Step):
    pass


class SelectWavelengths(Step):
    pass


class SubtractAvg(Step):
    pass
