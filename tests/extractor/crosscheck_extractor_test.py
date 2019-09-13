import np,arff,math

from unittest import TestCase

from pipeline.extractor.crosscheck_extractor import CrossCheckExtractor

class CrossCheckExtractorTest(TestCase):

    def test_calculate_single_features (self):
        arff_data = arff.load("""@RELATION crosscheck
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE chiSquared NUMERIC
@ATTRIBUTE xbi {0,1}
@DATA
1,2,3,4,5,6,7,8,9,0
10,12,12,14,14,16,16,18,33,1""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = CrossCheckExtractor(class_attr='xbi')
        result = extractor.execute(arff_data)
        self.assertEqual(2, len(result['X']))
        self.assertEqual(['area', 'displacement', 'sdr', 'chisquared'], result['features'])
        self.assertEqual(4, len(result['X'][0]))
        self.assertEqual(35, result['X'][0][0])
        self.assertAlmostEqual(math.sqrt(2), result['X'][0][1], places=2)
        self.assertAlmostEqual((48-35)/35, result['X'][0][2], places=2)
        self.assertEqual(9, result['X'][0][3])
        self.assertEqual(4, len(result['X'][1]))
        self.assertEqual(4, len(result['X'][0]))
        self.assertEqual(224, result['X'][1][0])
        self.assertAlmostEqual(2*math.sqrt(2), result['X'][1][1], places=2)
        self.assertAlmostEqual((16*18-14*16) / (14*16), result['X'][1][2], places=2)
        self.assertEqual(33, result['X'][1][3])

        self.assertEqual(2, len(result['y']))
        self.assertEqual(0, result['y'][0])
        self.assertEqual(1, result['y'][1])
        self.assertEqual(arff_data['attributes'], result['attributes'])

    def test_calculate_features_with_different_values (self):
        arff_data = arff.load("""@RELATION crosscheck
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE chiSquared NUMERIC
@ATTRIBUTE Result {0,1}
@DATA
1,4,5,9,9,9,10,10,0.7,1
1,1,5,5,9,9,10,10,0.1,0
9,15,8,16,10,20,10,50,0.0,1""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = CrossCheckExtractor(class_attr='Result')
        result = extractor.execute(arff_data)
        self.assertEqual(3, len(result['X']))
        self.assertEqual(['area', 'displacement', 'sdr', 'chisquared'], result['features'])
        self.assertEqual(4, len(result['X'][0]))
        self.assertEqual(90, result['X'][0][0])
        self.assertEqual(5, result['X'][0][1])
        self.assertEqual(0, result['X'][0][2])
        self.assertAlmostEqual(0.7, result['X'][0][3], places=2)
        self.assertEqual(4, len(result['X'][1]))
        self.assertEqual(90, result['X'][1][0])
        self.assertEqual(0, result['X'][1][1])
        self.assertEqual(0, result['X'][1][2])
        self.assertAlmostEqual(0.1, result['X'][1][3], places=2)
        self.assertEqual(4, len(result['X'][2]))
        self.assertEqual(100, result['X'][2][0])
        self.assertEqual(10, result['X'][2][1])
        self.assertAlmostEqual(9, result['X'][2][2], places=2)
        self.assertEqual(0, result['X'][2][3])

        self.assertEqual(3, len(result['y']))
        self.assertEqual(1, result['y'][0])
        self.assertEqual(0, result['y'][1])
        self.assertEqual(1, result['y'][2])

    def test_deal_with_zero_area (self):
        arff_data = arff.load("""@RELATION crosscheck
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE chiSquared NUMERIC
@ATTRIBUTE xbi {0,1}
@DATA
1,2,3,4,0,6,7,8,9,0
10,12,12,14,0,16,16,18,33,1""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = CrossCheckExtractor(class_attr='xbi')
        result = extractor.execute(arff_data)
        self.assertEqual(2, len(result['X']))
        self.assertEqual(['area', 'displacement', 'sdr', 'chisquared'], result['features'])
        self.assertEqual(4, len(result['X'][0]))
        self.assertEqual(0, result['X'][0][0])
        self.assertAlmostEqual(math.sqrt(2), result['X'][0][1], places=2)
        self.assertAlmostEqual(48, result['X'][0][2], places=2)
        self.assertEqual(9, result['X'][0][3])
        self.assertEqual(4, len(result['X'][1]))
        self.assertEqual(4, len(result['X'][0]))
        self.assertEqual(0, result['X'][1][0])
        self.assertAlmostEqual(2*math.sqrt(2), result['X'][1][1], places=2)
        self.assertAlmostEqual(16*18, result['X'][1][2], places=2)
        self.assertEqual(33, result['X'][1][3])

        self.assertEqual(2, len(result['y']))
        self.assertEqual(0, result['y'][0])
        self.assertEqual(1, result['y'][1])
