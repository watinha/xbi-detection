import np, arff

from unittest import TestCase

from pipeline.extractor.browserninja.font_family_extractor import FontFamilyExtractor

class FontFamilyExtractorTest(TestCase):

    def generate_arff(self, data):
        arff_header = """@RELATION browserninja.website
@ATTRIBUTE childsNumber NUMERIC
@ATTRIBUTE textLength NUMERIC
@ATTRIBUTE imageDiff NUMERIC
@ATTRIBUTE baseFontFamily STRING
@ATTRIBUTE targetFontFamily STRING
@ATTRIBUTE Result {0,1}
@DATA
"""
        arff_data = arff.load(arff_header + data)
        arff_data['data'] = np.array(arff_data['data'])
        return arff_data

    def setUp(self):
        self.extractor = FontFamilyExtractor()

    def test_extractor_returns_consistent_X_features_y_vectors(self):
        arff = self.generate_arff("""1,2,3,Arial,Arial,1
4,5,6,Arial,Arial,1""")
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 0])
        features = ['abobrinha', 'pepino', 'mamao']
        arguments = {
                'X': X, 'y': y, 'features': features,
                'data': arff['data'], 'attributes': arff['attributes']}
        result = self.extractor.execute(arguments)
        self.assertEqual(X[:,0].tolist(), result['X'][:,0].tolist())
        self.assertEqual(X[:,1].tolist(), result['X'][:,1].tolist())
        self.assertEqual(X[:,2].tolist(), result['X'][:,2].tolist())
        self.assertEqual(features, result['features'])
        self.assertEqual(y.tolist(), result['y'].tolist())

    def test_execute_use_one_hot_encoding_for_font_data(self):
        arff = self.generate_arff("""1,2,3,Arial,Arial,1
4,5,6,Arial,Arial,1""")
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 0])
        features = ['abobrinha', 'pepino', 'mamao']
        arguments = {
                'X': X, 'y': y, 'features': features,
                'data': arff['data'], 'attributes': arff['attributes']}
        result = self.extractor.execute(arguments)
        self.assertEqual(5, result['X'].shape[1])
        self.assertEqual([1, 1], result['X'][:,3].tolist())
        self.assertEqual([1, 1], result['X'][:,4].tolist())

    def test_execute_use_one_hot_encoding_for_font_data_with_more_data(self):
        arff = self.generate_arff("""1,2,3,Arial,Bonito,1
7,8,9,Deustrech,Arial,1
4,5,6,Bonito,Deustrech,1""")
        X = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6]])
        y = np.array([1, 0, 0])
        features = ['abobrinha', 'pepino', 'mamao']
        arguments = {
                'X': X, 'y': y, 'features': features,
                'data': arff['data'], 'attributes': arff['attributes']}
        result = self.extractor.execute(arguments)
        self.assertEqual((3,9), result['X'].shape)
        self.assertEqual([1, 0, 0], result['X'][:,3].tolist())
        self.assertEqual([0, 0, 1], result['X'][:,4].tolist())
        self.assertEqual([0, 1, 0], result['X'][:,5].tolist())
        self.assertEqual([0, 1, 0], result['X'][:,6].tolist())
        self.assertEqual([1, 0, 0], result['X'][:,7].tolist())
        self.assertEqual([0, 0, 1], result['X'][:,8].tolist())

    def test_real_dataset_for_measuring_X_matrix_conformance(self):
        arff_data = self.generate_arff("""1,2,3,Arial,Bonito,1
7,8,9,Deustrech,Arial,1
10,11,12,Bonito,Arial,1
4,5,6,Bonito,Deustrech,1""")
        X = np.ones(np.array(arff_data['data']).shape)
        y = np.ones(X.shape[0])
        arguments = {
                'X': X, 'y': y, 'attributes': arff_data['attributes'],
                'data': np.array(arff_data['data']) }
        result = self.extractor.execute(arguments)
        self.assertEqual(y.shape[0], result['X'].shape[0])
