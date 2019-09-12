import arff

class ArffLoader():

    def __init__(self):
        pass

    def execute(self, data):
        return arff.load(data)
