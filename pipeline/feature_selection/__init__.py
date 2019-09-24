class FeatureSelection(object):

    def __init__(self, strategy):
        self._strategy = strategy

    def execute(self, argument):
        new_X = self._strategy.fit_transform(argument['X'], argument['y'])
        argument['X'] = new_X
        return argument

