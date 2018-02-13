class Preprocess(object):
    def __init__(self, steps=()):
        self.steps = list(steps)

    def __call__(self, data):
        return self.transform(data)

    def fit(self, data):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError
