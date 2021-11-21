import np

class ImageMomentsExtractor():

    def execute(self, arff_data, attributes, X):
        base_centroid_x = np.array(arff_data['data'][:, attributes.index('base_centroid_x')], dtype='float64')
        base_centroid_y = np.array(arff_data['data'][:, attributes.index('base_centroid_y')], dtype='float64')
        base_orientation = np.array(arff_data['data'][:, attributes.index('base_orientation')], dtype='float64')

        target_centroid_x = np.array(arff_data['data'][:, attributes.index('target_centroid_x')], dtype='float64')
        target_centroid_y = np.array(arff_data['data'][:, attributes.index('target_centroid_y')], dtype='float64')
        target_orientation = np.array(arff_data['data'][:, attributes.index('target_orientation')], dtype='float64')

        X.append(np.abs(base_centroid_x - target_centroid_x))
        X.append(np.abs(base_centroid_y - target_centroid_y))
        X.append(np.abs(base_orientation - target_orientation))

        return X
