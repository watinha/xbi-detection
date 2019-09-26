class ClassifierTunning():

    def __init__(self, grid, model, groups=None):
        self._grid = grid
        self._model = model
        self._groups = groups

    def execute(self, data):
        if (self._groups == None):
            self._grid.fit(data['X'], data['y'])
        else:
            attributes = [ attribute[0] for attribute in data['attributes'] ]
            groups = data['data'][:, attributes.index(self._groups)].tolist()
            self._grid.fit(data['X'], data['y'], groups=groups)
        self._model.set_params(**self._grid.best_params_)
        data['model'] = self._model
        return data

