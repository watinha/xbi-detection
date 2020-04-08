import np, arff

from unittest import TestCase

from pipeline.extractor.browserninja.relative_position_extractor import RelativePositionExtractor

class RelativePositionExtractorTest(TestCase):

    def generate_arff(self, data, class_attr='Result'):
        arff_header = """@RELATION browserninja.website
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE baseParentX NUMERIC
@ATTRIBUTE targetParentX NUMERIC
@ATTRIBUTE baseParentY NUMERIC
@ATTRIBUTE targetParentY NUMERIC
@ATTRIBUTE baseViewportWidth NUMERIC
@ATTRIBUTE targetViewportWidth NUMERIC
@ATTRIBUTE basePreviousSiblingLeft NUMERIC
@ATTRIBUTE targetPreviousSiblingLeft NUMERIC
@ATTRIBUTE basePreviousSiblingTop NUMERIC
@ATTRIBUTE targetPreviousSiblingTop NUMERIC
@ATTRIBUTE baseNextSiblingLeft NUMERIC
@ATTRIBUTE targetNextSiblingLeft NUMERIC
@ATTRIBUTE baseNextSiblingTop NUMERIC
@ATTRIBUTE targetNextSiblingTop NUMERIC
@ATTRIBUTE childsNumber NUMERIC
@ATTRIBUTE textLength NUMERIC
@ATTRIBUTE %s {0,1}
@DATA
""" % (class_attr)
        arff_data = arff.load(arff_header + data)
        attributes = [ attribute[0] for attribute in arff_data['attributes'] ]
        arff_data['data'] = np.array(arff_data['data'])
        arff_data['features'] = []
        return (arff_data, attributes)

    def setUp(self):
        self.extractor = RelativePositionExtractor()

    def test_extractor_implements_execute_method_with_correct_arguments (self):
        X = np.array([[1, 2, 3], [11, 12, 13]])
        arff_string = """1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,1
1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        result = self.extractor.execute(arff_data, attributes, X.T.tolist())
        result = np.array(result).T.tolist()
        self.assertEqual(2, len(result))
        self.assertEqual(X.tolist(), (np.array(result)[0:2,0:3]).tolist())

    def test_execute_extracts_relative_distance_to_parent_element(self):
        X = []
        arff_string = """1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        result = self.extractor.execute(arff_data, attributes, X)
        result = np.array(result).T.tolist()
        self.assertEqual(1, len(result))
        self.assertEqual(((1 - 9) - (2 - 10)) / 7 , result[0][0])
        self.assertEqual(((3 - 11) - (4 - 12)) / 5, result[0][1])

    def test_execute_extracts_relative_distance_to_parent_element_with_different_values (self):
        X = []
        arff_string = """13,20,60,6,10,11,13,10,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        result = self.extractor.execute(arff_data, attributes, X)
        result = np.array(result).T.tolist()
        self.assertEqual(1, len(result))
        self.assertEqual(abs((13 - 9) - (20 - 10)) / 10, result[0][0])
        self.assertEqual(abs((60 - 11) - (6 - 12)) / 10, result[0][1])

    def test_execute_extracts_relative_distance_to_parent_element_with_different_again (self):
        X = []
        arff_string = """13,20,60,6,10,7,13,50,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        result = self.extractor.execute(arff_data, attributes, X)
        result = np.array(result).T.tolist()
        self.assertEqual(1, len(result))
        self.assertEqual(abs((13 - 9) - (20 - 10)) / 13, result[0][0])
        self.assertEqual(abs((60 - 11) - (6 - 12)) / 7, result[0][1])

    def test_execute_extracts_relative_distance_to_previous_sibling_element(self):
        X = []
        arff_string = """1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        result = self.extractor.execute(arff_data, attributes, X)
        result = np.array(result).T.tolist()
        self.assertEqual(1, len(result))
        self.assertEqual(((1 - 15) - (2 - 16)) / 7 , result[0][2])
        self.assertEqual(((3 - 17) - (4 - 18)) / 5, result[0][3])

    def test_execute_extracts_relative_distance_to_previous_sibling_element_with_different_values (self):
        X = []
        arff_string = """13,20,60,6,10,11,13,10,9,10,11,12,13,14,99,16,45,300,19,20,21,22,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        result = self.extractor.execute(arff_data, attributes, X)
        result = np.array(result).T.tolist()
        self.assertEqual(1, len(result))
        self.assertEqual(abs((13 - 99) - (20 - 16)) / 10, result[0][2])
        self.assertEqual(abs((60 - 45) - (6 - 300)) / 10, result[0][3])

    def test_execute_extracts_relative_distance_to_previous_sibling_element_with_different_again (self):
        X = []
        arff_string = """13,20,60,6,10,7,13,50,9,10,11,12,13,14,99,16,45,300,19,20,21,22,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        result = self.extractor.execute(arff_data, attributes, X)
        result = np.array(result).T.tolist()
        self.assertEqual(1, len(result))
        self.assertEqual(abs((13 - 99) - (20 - 16)) / 13, result[0][2])
        self.assertEqual(abs((60 - 45) - (6 - 300)) / 7, result[0][3])

    def test_execute_extracts_relative_distance_to_next_sibling_element(self):
        X = []
        arff_string = """1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        result = self.extractor.execute(arff_data, attributes, X)
        result = np.array(result).T.tolist()
        self.assertEqual(1, len(result))
        self.assertEqual(((1 - 19) - (2 - 20)) / 7 , result[0][4])
        self.assertEqual(((3 - 21) - (4 - 22)) / 5, result[0][5])

    def test_execute_extracts_relative_distance_to_next_sibling_element_with_different_values (self):
        X = []
        arff_string = """13,20,60,6,10,11,13,10,9,10,11,12,13,14,99,16,45,300,233,180,30,17,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        result = self.extractor.execute(arff_data, attributes, X)
        result = np.array(result).T.tolist()
        self.assertEqual(1, len(result))
        self.assertEqual(abs((13 - 233) - (20 - 180)) / 10, result[0][4])
        self.assertEqual(abs((60 - 30) - (6 - 17)) / 10, result[0][5])

    def test_execute_extracts_relative_distance_to_next_sibling_element_with_different_again (self):
        X = []
        arff_string = """13,20,60,6,10,7,13,50,9,10,11,12,13,14,99,16,45,300,233,180,30,17,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        result = self.extractor.execute(arff_data, attributes, X)
        result = np.array(result).T.tolist()
        self.assertEqual(1, len(result))
        self.assertEqual(abs((13 - 233) - (20 - 180)) / 13, result[0][4])
        self.assertEqual(abs((60 - 30) - (6 - 17)) / 7, result[0][5])

    def test_execute_includes_the_features_names_in_arff_data (self):
        X = []
        arff_string = """13,20,60,6,10,7,13,50,9,10,11,12,13,14,99,16,45,300,233,180,30,17,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        result = self.extractor.execute(arff_data, attributes, X)
        result = np.array(result).T.tolist()
        self.assertEqual(1, len(result))
        self.assertEqual(['parent_x', 'parent_y', 'previous_sibling_x', 'previous_sibling_y',
                          'next_sibling_x', 'next_sibling_y'], arff_data['features'])

    def test_execute_includes_the_features_names_in_arff_data_with_other_extractors (self):
        X = []
        arff_string = """13,20,60,6,10,7,13,50,9,10,11,12,13,14,99,16,45,300,233,180,30,17,23,24,1"""
        (arff_data, attributes) = self.generate_arff(arff_string)
        arff_data['features'] = ['other_features', 'another']
        result = self.extractor.execute(arff_data, attributes, X)
        result = np.array(result).T.tolist()
        self.assertEqual(1, len(result))
        self.assertEqual(['other_features', 'another', 'parent_x', 'parent_y',
                          'previous_sibling_x', 'previous_sibling_y',
                          'next_sibling_x', 'next_sibling_y'], arff_data['features'])
