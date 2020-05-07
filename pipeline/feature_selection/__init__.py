class FeatureSelection(object):

    def __init__(self, strategy, k=10):
        self._strategy = strategy
        self._k = k

    def execute(self, argument):
        if (argument['X'].shape[1] < self._k):
            return argument
        new_X = self._strategy.fit_transform(argument['X'], argument['y'])
        argument['X'] = new_X
        return argument

