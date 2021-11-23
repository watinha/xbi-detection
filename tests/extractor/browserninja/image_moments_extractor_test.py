import arff,np

from unittest import TestCase

from pipeline.extractor.browserninja.image_moments_extractor import ImageMomentsExtractor

class BrowserNinjaExtractorTest(TestCase):

    def generate_arff(self, data, class_attr='Result'):
        arff_header = """@RELATION browserninja.website
@ATTRIBUTE base_centroid_x NUMERIC
@ATTRIBUTE base_centroid_y NUMERIC
@ATTRIBUTE base_orientation NUMERIC
@ATTRIBUTE target_centroid_x NUMERIC
@ATTRIBUTE target_centroid_y NUMERIC
@ATTRIBUTE target_orientation NUMERIC
@ATTRIBUTE %s {0,1}
@DATA
""" % (class_attr)
        self.extractor = ImageMomentsExtractor()
        arff_data = arff.load(arff_header + data)
        attributes = [ attribute[0] for attribute in arff_data['attributes'] ]
        arff_data['data'] = np.array(arff_data['data'])
        arff_data['features'] = []
        return (arff_data, attributes)

    def test_extracts_differences_between_centroid_and_orientation_attributes(self):
        X_expected = np.array([[8, 9, 11], [0.4, 0.4, 0.4]])
        (arff_data, attributes) = self.generate_arff("""1,2,3,9,11,14,1
0.1,0.2,0.3,0.5,0.6,0.7,0""")
        result = self.extractor.execute(arff_data, attributes, [])
        result = np.array(result).T.tolist()
        self.assertEqual(2, len(result))
        np.testing.assert_almost_equal(X_expected, result)
        self.assertEqual(['centroid_x', 'centroid_y', 'orientation'], arff_data['features'])


    def test_extracts_differences_between_centroid_and_orientation_attributes_with_X(self):
        X_expected = np.array([[1, 2, 3, 3, 3], [3, 4, 7, 6, 11]])
        X_initial = np.array([[1, 2], [3, 4]])
        (arff_data, attributes) = self.generate_arff("""1,2,3,4,5,6,1
10,11,12,3,5,1,0""")
        arff_data['features'] = ['S', 'J']
        result = self.extractor.execute(arff_data, attributes, X_initial.T.tolist())
        result = np.array(result).T.tolist()
        self.assertEqual(2, len(result))
        np.testing.assert_almost_equal(X_expected, result)
        self.assertEqual(['S', 'J', 'centroid_x', 'centroid_y', 'orientation'], arff_data['features'])

