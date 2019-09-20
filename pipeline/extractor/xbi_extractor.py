import np

class XBIExtractor():

    def __init__(self, features, class_label):
        self._features = features
        self._class_label = class_label

    def execute(self, arff_data):
        arff_data['features'] = self._features
        attributes = [ attribute[0]
                for attribute in arff_data['attributes'] ]
        Xt = []
        for feature in self._features:
            Xt.append(arff_data['data'][:,attributes.index(feature)])

        arff_data['X'] = np.array(Xt, dtype='float64').T
        arff_data['y'] = np.array(arff_data['data'][:,attributes.index(self._class_label)])

        return arff_data
