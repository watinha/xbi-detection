class ClassifierTunning():

    def __init__(self, grid, model):
        self._grid = grid
        self._model = model

    def execute(self, data):
        self._grid.fit(data['X'], data['y'])
        self._model.set_params(**self._grid.best_params_)
        data['model'] = self._model
        return data

