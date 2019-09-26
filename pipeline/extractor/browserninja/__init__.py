import np

class BrowserNinjaCompositeExtractor():

    def __init__(self, class_attr='Result', extractors = []):
        self._extractors = extractors
        self._class_attr = class_attr

    def execute(self, arff_data):
        arff_data['features'] = ['childsNumber', 'textLength', 'area',
                          'phash', 'chiSquared', 'imageDiff',
                          'width_comp', 'height_comp',
                          'left_visibility', 'right_visibility',
                          'left_comp', 'right_comp', 'y_comp']

        attributes = [ attribute[0]
                for attribute in arff_data['attributes'] ]

        X = (arff_data['X'].T.tolist() if 'X' in arff_data else [])

        for extractor in self._extractors:
            X = extractor.execute(arff_data, attributes, X)

        arff_data['X'] = np.array(X, dtype='float64').T
        arff_data['y'] = np.array(arff_data['data'][:, attributes.index(self._class_attr)])

        return arff_data


class ComplexityExtractor():

    def execute(self, arff_data, attributes, X):
        X.append(arff_data['data'][:, attributes.index('childsNumber')])
        X.append(arff_data['data'][:, attributes.index('textLength')])
        base_width = np.array(arff_data['data'][:, attributes.index('baseWidth')], dtype='float64')
        target_width = np.array(arff_data['data'][:, attributes.index('targetWidth')], dtype='float64')
        base_height = np.array(arff_data['data'][:, attributes.index('baseHeight')], dtype='float64')
        target_height = np.array(arff_data['data'][:, attributes.index('targetHeight')], dtype='float64')
        X.append(np.maximum(base_width * base_height, target_width * target_height))
        return X


class ImageComparisonExtractor():

    def execute(self, arff_data, attributes, X):
        X.append(arff_data['data'][:, attributes.index('phash')])
        X.append(arff_data['data'][:, attributes.index('chiSquared')])

        image_diff = np.array(arff_data['data'][:, attributes.index('imageDiff')], dtype='float64')
        base_width = np.array(arff_data['data'][:, attributes.index('baseWidth')], dtype='float64')
        target_width = np.array(arff_data['data'][:, attributes.index('targetWidth')], dtype='float64')
        base_height = np.array(arff_data['data'][:, attributes.index('baseHeight')], dtype='float64')
        target_height = np.array(arff_data['data'][:, attributes.index('targetHeight')], dtype='float64')
        min_area = np.minimum(base_width * base_height, target_width * target_height)
        X.append(image_diff / (np.maximum(min_area, 1) * 255))
        return X


class SizeViewportExtractor():

    def execute(self, arff_data, attributes, X):
        base_width = np.array(arff_data['data'][:, attributes.index('baseWidth')], dtype='float64')
        target_width = np.array(arff_data['data'][:, attributes.index('targetWidth')], dtype='float64')
        base_height = np.array(arff_data['data'][:, attributes.index('baseHeight')], dtype='float64')
        target_height = np.array(arff_data['data'][:, attributes.index('targetHeight')], dtype='float64')
        base_viewport = np.array(arff_data['data'][:, attributes.index('baseViewportWidth')], dtype='float64')
        target_viewport = np.array(arff_data['data'][:, attributes.index('targetViewportWidth')], dtype='float64')
        X.append(np.abs((base_width - target_width)/np.maximum(np.abs(base_viewport - target_viewport), 1)))
        X.append(np.abs((base_height - target_height)/np.maximum(np.maximum(base_height, target_height), 1)))
        return X


class VisibilityExtractor():

    def execute(self, arff_data, attributes, X):
        base_width = np.array(arff_data['data'][:, attributes.index('baseWidth')], dtype='float64')
        target_width = np.array(arff_data['data'][:, attributes.index('targetWidth')], dtype='float64')
        base_viewport = np.array(arff_data['data'][:, attributes.index('baseViewportWidth')], dtype='float64')
        target_viewport = np.array(arff_data['data'][:, attributes.index('targetViewportWidth')], dtype='float64')
        base_left = np.array(arff_data['data'][:, attributes.index('baseX')], dtype='float64')
        base_right = np.array(base_viewport - (base_left + base_width), dtype='float64')
        target_left = np.array(arff_data['data'][:, attributes.index('targetX')], dtype='float64')
        target_right = np.array(target_viewport - (target_left + target_width), dtype='float64')
        X.append((base_right - base_viewport) - (target_right - target_viewport))
        X.append((base_left - base_viewport) - (target_left - target_viewport))
        return X


class PositionViewportExtractor():

    def execute(self, arff_data, attributes, X):
        base_width = np.array(arff_data['data'][:, attributes.index('baseWidth')], dtype='float64')
        target_width = np.array(arff_data['data'][:, attributes.index('targetWidth')], dtype='float64')
        base_viewport = np.array(arff_data['data'][:, attributes.index('baseViewportWidth')], dtype='float64')
        target_viewport = np.array(arff_data['data'][:, attributes.index('targetViewportWidth')], dtype='float64')
        base_left = np.array(arff_data['data'][:, attributes.index('baseX')], dtype='float64')
        base_right = np.array(base_viewport - (base_left + base_width), dtype='float64')
        target_left = np.array(arff_data['data'][:, attributes.index('targetX')], dtype='float64')
        target_right = np.array(target_viewport - (target_left + target_width), dtype='float64')
        X.append(np.abs((base_left - target_left) / np.maximum(np.abs(base_viewport - target_viewport), 1)))
        X.append(np.abs((base_right - target_right) / np.maximum(np.abs(base_viewport - target_viewport), 1)))
        base_y = np.array(arff_data['data'][:, attributes.index('baseY')], dtype='float64')
        target_y = np.array(arff_data['data'][:, attributes.index('targetY')], dtype='float64')
        X.append(np.abs((base_y - target_y) / np.maximum(np.abs(base_viewport - target_viewport), 1)))
        return X
