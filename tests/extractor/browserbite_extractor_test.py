import np,arff,math

from unittest import TestCase

from pipeline.extractor.browserbite_extractor import BrowserbiteExtractor

class BrowserbiteExtractorTest(TestCase):

    def test_calculate_single_features (self):
        arff_data = arff.load("""@RELATION browserbite
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE base_bin1 NUMERIC
@ATTRIBUTE base_bin2 NUMERIC
@ATTRIBUTE base_bin3 NUMERIC
@ATTRIBUTE base_bin4 NUMERIC
@ATTRIBUTE base_bin5 NUMERIC
@ATTRIBUTE base_bin6 NUMERIC
@ATTRIBUTE base_bin7 NUMERIC
@ATTRIBUTE base_bin8 NUMERIC
@ATTRIBUTE base_bin9 NUMERIC
@ATTRIBUTE base_bin10 NUMERIC
@ATTRIBUTE target_bin1 NUMERIC
@ATTRIBUTE target_bin2 NUMERIC
@ATTRIBUTE target_bin3 NUMERIC
@ATTRIBUTE target_bin4 NUMERIC
@ATTRIBUTE target_bin5 NUMERIC
@ATTRIBUTE target_bin6 NUMERIC
@ATTRIBUTE target_bin7 NUMERIC
@ATTRIBUTE target_bin8 NUMERIC
@ATTRIBUTE target_bin9 NUMERIC
@ATTRIBUTE target_bin10 NUMERIC
@ATTRIBUTE targetPlatform STRING
@ATTRIBUTE ncc NUMERIC
@ATTRIBUTE missmatch NUMERIC
@ATTRIBUTE xbi {0,1}
@DATA
1,2,3,4,5,6,7,8,77,78,79,70,71,72,73,74,75,76,90,90,90,90,90,90,90,90,90,90,'iOS 12.1 - Safari -- iOS - iPhoneSE',0.22,0.9,0
10,12,12,14,14,16,16,33,77,78,79,70,71,72,73,74,75,76,90,90,90,90,90,90,90,90,90,90,'iOS 12.1 - Safari -- iOS - iPhone 8 Plus',1,0.75,1""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = BrowserbiteExtractor(class_attr='xbi')
        result = extractor.execute(arff_data)
        self.assertEqual(2, len(result['X']))
        self.assertEqual(['diff_height', 'diff_width', 'diff_x', 'diff_y',
                          'missmatch', 'correlation',
                          'base_bin1', 'base_bin2', 'base_bin3', 'base_bin4', 'base_bin5',
                          'base_bin6', 'base_bin7', 'base_bin8', 'base_bin9', 'base_bin10',
                          'platform_id'
                          ], result['features'])
        self.assertEqual(arff_data['attributes'], result['attributes'])
        np.testing.assert_almost_equal(
                [5, 7, 1, 3, 0.9, 0.22, 77,78,79,70,71,72,73,74,75,76,1], result['X'][0])
        np.testing.assert_almost_equal(
                [14, 16, 10, 12, 0.75, 1, 77,78,79,70,71,72,73,74,75,76,0], result['X'][1])
        np.testing.assert_almost_equal([0, 1], result['y'])

    def test_calculate_other_features (self):
        arff_data = arff.load("""@RELATION browserbite
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE base_bin1 NUMERIC
@ATTRIBUTE base_bin2 NUMERIC
@ATTRIBUTE base_bin3 NUMERIC
@ATTRIBUTE base_bin4 NUMERIC
@ATTRIBUTE base_bin5 NUMERIC
@ATTRIBUTE base_bin6 NUMERIC
@ATTRIBUTE base_bin7 NUMERIC
@ATTRIBUTE base_bin8 NUMERIC
@ATTRIBUTE base_bin9 NUMERIC
@ATTRIBUTE base_bin10 NUMERIC
@ATTRIBUTE target_bin1 NUMERIC
@ATTRIBUTE target_bin2 NUMERIC
@ATTRIBUTE target_bin3 NUMERIC
@ATTRIBUTE target_bin4 NUMERIC
@ATTRIBUTE target_bin5 NUMERIC
@ATTRIBUTE target_bin6 NUMERIC
@ATTRIBUTE target_bin7 NUMERIC
@ATTRIBUTE target_bin8 NUMERIC
@ATTRIBUTE target_bin9 NUMERIC
@ATTRIBUTE target_bin10 NUMERIC
@ATTRIBUTE targetPlatform STRING
@ATTRIBUTE ncc NUMERIC
@ATTRIBUTE missmatch NUMERIC
@ATTRIBUTE internal {0,1}
@DATA
1,2,3,4,5,6,7,8,77,78,79,70,71,72,73,74,75,76,90,90,90,90,90,90,90,90,90,90,'iOS 12.1 - Safari -- iOS - iPhone 8 Plus',0.42,0.25,0
3,6,2,5,1,4,6,9,70,71,72,71,72,80,90,74,75,76,90,9,90,90,90,90,90,90,90,90,'iOS 12.1 - Safari -- iOS - iPhone SE',0.77,0.5,0
10,12,12,14,14,16,16,33,77,78,79,70,71,72,73,74,75,76,90,90,90,90,90,90,90,90,90,90,'Android',0.8,0.75,1""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = BrowserbiteExtractor(class_attr='internal')
        result = extractor.execute(arff_data)
        self.assertEqual(3, len(result['X']))
        np.testing.assert_almost_equal(
                [5, 7, 1, 3, 0.25, 0.42, 77,78,79,70,71,72,73,74,75,76,1], result['X'][0])
        np.testing.assert_almost_equal(
                [1, 6, 3, 2, 0.5, 0.77, 70,71,72,71,72,80,90,74,75,76,2], result['X'][1])
        np.testing.assert_almost_equal(
                [14, 16, 10, 12, 0.75, 0.8,77,78,79,70,71,72,73,74,75,76,0], result['X'][2])
        np.testing.assert_almost_equal([0, 0, 1], result['y'])

    def test_calculate_other_features_and_previous_features (self):
        arff_data = arff.load("""@RELATION browserbite
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE base_bin1 NUMERIC
@ATTRIBUTE base_bin2 NUMERIC
@ATTRIBUTE base_bin3 NUMERIC
@ATTRIBUTE base_bin4 NUMERIC
@ATTRIBUTE base_bin5 NUMERIC
@ATTRIBUTE base_bin6 NUMERIC
@ATTRIBUTE base_bin7 NUMERIC
@ATTRIBUTE base_bin8 NUMERIC
@ATTRIBUTE base_bin9 NUMERIC
@ATTRIBUTE base_bin10 NUMERIC
@ATTRIBUTE target_bin1 NUMERIC
@ATTRIBUTE target_bin2 NUMERIC
@ATTRIBUTE target_bin3 NUMERIC
@ATTRIBUTE target_bin4 NUMERIC
@ATTRIBUTE target_bin5 NUMERIC
@ATTRIBUTE target_bin6 NUMERIC
@ATTRIBUTE target_bin7 NUMERIC
@ATTRIBUTE target_bin8 NUMERIC
@ATTRIBUTE target_bin9 NUMERIC
@ATTRIBUTE target_bin10 NUMERIC
@ATTRIBUTE targetPlatform STRING
@ATTRIBUTE ncc NUMERIC
@ATTRIBUTE missmatch NUMERIC
@ATTRIBUTE external {0,1}
@DATA
1,2,3,4,5,6,7,8,77,78,79,70,71,72,73,74,75,76,90,90,90,90,90,90,90,90,90,90,'iOS 12.1 - Safari -- iOS - iPhone 8 Plus',0.23,0.33,0
3,6,2,5,1,4,6,9,70,71,72,71,72,80,90,74,75,76,90,9,90,90,90,90,90,90,90,90,'iOS 12.1 - Safari -- iOS - iPhone SE',0.14,0.66,0
10,12,12,14,14,16,16,33,77,78,79,70,71,72,73,74,75,76,90,90,90,90,90,90,90,90,90,90,'Android',0.91,0.99,1""")
        arff_data['data'] = np.array(arff_data['data'])
        arff_data['X'] = np.array([[1, 2], [1, 2], [1, 2]])
        arff_data['features'] = ['bla', 'bla1']
        extractor = BrowserbiteExtractor(class_attr='external')
        result = extractor.execute(arff_data)
        self.assertEqual(3, len(result['X']))
        np.testing.assert_almost_equal(
                [1, 2, 5, 7, 1, 3, 0.33, 0.23, 77,78,79,70,71,72,73,74,75,76,1], result['X'][0])
        np.testing.assert_almost_equal(
                [1, 2, 1, 6, 3, 2, 0.66, 0.14, 70,71,72,71,72,80,90,74,75,76,2], result['X'][1])
        np.testing.assert_almost_equal(
                [1, 2, 14, 16, 10, 12, 0.99, 0.91, 77,78,79,70,71,72,73,74,75,76,0], result['X'][2])
        np.testing.assert_almost_equal([0, 0, 1], result['y'])
        self.assertEqual(['bla', 'bla1',
                          'diff_height', 'diff_width', 'diff_x', 'diff_y',
                          'missmatch', 'correlation',
                          'base_bin1', 'base_bin2', 'base_bin3', 'base_bin4', 'base_bin5',
                          'base_bin6', 'base_bin7', 'base_bin8', 'base_bin9', 'base_bin10',
                          'platform_id'
                          ], result['features'])
