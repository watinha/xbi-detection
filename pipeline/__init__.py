class Pipeline():
    def __init__(self, actions):
        self._actions = actions

    def execute(self, arguments):
        for action in self._actions:
            arguments = action.execute(arguments)
        return arguments
