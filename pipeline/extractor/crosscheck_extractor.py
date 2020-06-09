import math,np

class CrossCheckExtractor():

    def __init__(self,class_attr='Result'):
        self._class_attr = class_attr

    def execute(self, arff_data):
        attributes = [ attribute[0] for attribute in arff_data['attributes'] ]
        dataset = arff_data['data']

        X = (arff_data['X'].T.tolist() if 'X' in arff_data else [])

        base_height = np.array(dataset[:,attributes.index('baseHeight')], dtype='float64')
        target_height = np.array(dataset[:,attributes.index('targetHeight')], dtype='float64')
        base_width = np.array(dataset[:,attributes.index('baseWidth')], dtype='float64')
        target_width = np.array(dataset[:,attributes.index('targetWidth')], dtype='float64')
        X.append(np.minimum(base_height * base_width, target_height * target_width))

        base_x = np.array(dataset[:,attributes.index('baseX')], dtype='float64')
        target_x = np.array(dataset[:,attributes.index('targetX')], dtype='float64')
        base_y = np.array(dataset[:,attributes.index('baseY')], dtype='float64')
        target_y = np.array(dataset[:,attributes.index('targetY')], dtype='float64')
        X.append(np.sqrt(
            np.power(np.abs(base_x - target_x), 2) +
            np.power(np.abs(base_y - target_y), 2)))

        X.append(np.abs((base_height * base_width) - (target_height * target_width)) /
                 np.maximum(
                     np.minimum(base_height * base_width, target_height * target_width),
                     np.ones(len(base_height))
                 ))

        X.append(dataset[:,attributes.index('chiSquared')])

        arff_data['X'] = np.array(X, dtype='float64').T
        prev_features = (arff_data['features'] if 'features' in arff_data else [])
        arff_data['features'] = prev_features + ['area', 'displacement', 'sdr', 'chisquared']
        arff_data['y'] = np.array(arff_data['data'][:,attributes.index(self._class_attr)], dtype='int16')
        return arff_data
