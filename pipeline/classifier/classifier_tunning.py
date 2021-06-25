class ClassifierTunning():

    def __init__(self, grid, model, groups=None):
        self._grid = grid
        self._model = model
        self._groups = groups

    def execute(self, data):
        data['grid'] = self._grid
        data['model'] = self._model
        return data

