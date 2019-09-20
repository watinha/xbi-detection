import arff,np

from unittest import TestCase

from pipeline.extractor.browserninja_extractor import BrowserNinjaExtractor

class BrowserNinjaExtractorTest(TestCase):

    def test_execute_extracts_features_from_arff(self):
        arff_data = arff.load("""@RELATION browserninja.website
@ATTRIBUTE childsNumber NUMERIC
@ATTRIBUTE textLength NUMERIC
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE imageDiff NUMERIC
@ATTRIBUTE chiSquared NUMERIC
@ATTRIBUTE baseViewportWidth NUMERIC
@ATTRIBUTE targetViewportWidth NUMERIC
@ATTRIBUTE phash NUMERIC
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = BrowserNinjaExtractor(class_attr='Result')
        result = extractor.execute(arff_data)
        self.assertEqual(arff_data['attributes'], result['attributes'])
        self.assertEqual(['childsNumber', 'textLength', 'area',
                          'phash', 'chiSquared', 'imageDiff',
                          'width_comp', 'height_comp',
                          'left_visibility', 'right_visibility',
                          'left_comp', 'right_comp', 'y_comp'], result['features'])

    def test_extracts_complexity_features(self):
        arff_data = arff.load("""@RELATION browserninja.website
@ATTRIBUTE childsNumber NUMERIC
@ATTRIBUTE textLength NUMERIC
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE imageDiff NUMERIC
@ATTRIBUTE chiSquared NUMERIC
@ATTRIBUTE baseViewportWidth NUMERIC
@ATTRIBUTE targetViewportWidth NUMERIC
@ATTRIBUTE phash NUMERIC
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = BrowserNinjaExtractor(class_attr='Result')
        result = extractor.execute(arff_data)
        self.assertEqual(13, result['X'][0][0])
        self.assertEqual(17, result['X'][0][1])
        self.assertEqual(48, result['X'][0][2])
        self.assertEqual(0, result['X'][1][0])
        self.assertEqual(0, result['X'][1][1])
        self.assertEqual(48, result['X'][1][2])

    def test_execute_extracts_image_comparison_features(self):
        arff_data = arff.load("""@RELATION browserninja.website
@ATTRIBUTE childsNumber NUMERIC
@ATTRIBUTE textLength NUMERIC
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE imageDiff NUMERIC
@ATTRIBUTE chiSquared NUMERIC
@ATTRIBUTE baseViewportWidth NUMERIC
@ATTRIBUTE targetViewportWidth NUMERIC
@ATTRIBUTE phash NUMERIC
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,1,2,3,4,5,3,7,10,1000,0.25,360,414,0.15,0""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = BrowserNinjaExtractor(class_attr='Result')
        result = extractor.execute(arff_data)
        self.assertEqual(0.3, result['X'][0][3])
        self.assertEqual(0.12, result['X'][0][4])
        self.assertEqual(100/(35 * 255), result['X'][0][5])
        self.assertEqual(0.15, result['X'][1][3])
        self.assertEqual(0.25, result['X'][1][4])
        self.assertEqual(1000/(30 * 255), result['X'][1][5])

    def test_execute_extracts_image_comparison_features_when_area_is_zero(self):
        arff_data = arff.load("""@RELATION browserninja.website
@ATTRIBUTE childsNumber NUMERIC
@ATTRIBUTE textLength NUMERIC
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE imageDiff NUMERIC
@ATTRIBUTE chiSquared NUMERIC
@ATTRIBUTE baseViewportWidth NUMERIC
@ATTRIBUTE targetViewportWidth NUMERIC
@ATTRIBUTE phash NUMERIC
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,0,8,100,0.12,360,414,0.3,0
0,0,1,2,3,4,5,3,7,10,1000,0.25,360,414,0.15,0""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = BrowserNinjaExtractor(class_attr='Result')
        result = extractor.execute(arff_data)
        self.assertEqual(0.3, result['X'][0][3])
        self.assertEqual(0.12, result['X'][0][4])
        self.assertEqual(100/255, result['X'][0][5])

    def test_execute_extracts_height_and_width_comparison(self):
        arff_data = arff.load("""@RELATION browserninja.website
@ATTRIBUTE childsNumber NUMERIC
@ATTRIBUTE textLength NUMERIC
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE imageDiff NUMERIC
@ATTRIBUTE chiSquared NUMERIC
@ATTRIBUTE baseViewportWidth NUMERIC
@ATTRIBUTE targetViewportWidth NUMERIC
@ATTRIBUTE phash NUMERIC
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,1,2,3,4,10,15,20,33,1000,0.25,360,380,0.15,0""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = BrowserNinjaExtractor(class_attr='Result')
        result = extractor.execute(arff_data)
        self.assertEqual(1/54, result['X'][0][6])
        self.assertEqual(1/6, result['X'][0][7])
        self.assertEqual(13/20, result['X'][1][6])
        self.assertEqual(1/3, result['X'][1][7])

    def test_execute_extracts_visibility_comparison(self):
        arff_data = arff.load("""@RELATION browserninja.website
@ATTRIBUTE childsNumber NUMERIC
@ATTRIBUTE textLength NUMERIC
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE imageDiff NUMERIC
@ATTRIBUTE chiSquared NUMERIC
@ATTRIBUTE baseViewportWidth NUMERIC
@ATTRIBUTE targetViewportWidth NUMERIC
@ATTRIBUTE phash NUMERIC
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,380,0.15,0""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = BrowserNinjaExtractor(class_attr='Result')
        result = extractor.execute(arff_data)
        # right = (354 - 360) - (406 - 414)
        self.assertEqual(2, result['X'][0][8])
        self.assertEqual(53, result['X'][0][9])
        # right = (240 - 360) - (197 - 380)
        self.assertEqual(63, result['X'][1][8])
        self.assertEqual(-30, result['X'][1][9])

    def test_execute_comparison_based_on_viewport (self):
        arff_data = arff.load("""@RELATION browserninja.website
@ATTRIBUTE childsNumber NUMERIC
@ATTRIBUTE textLength NUMERIC
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE imageDiff NUMERIC
@ATTRIBUTE chiSquared NUMERIC
@ATTRIBUTE baseViewportWidth NUMERIC
@ATTRIBUTE targetViewportWidth NUMERIC
@ATTRIBUTE phash NUMERIC
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,380,0.15,0""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = BrowserNinjaExtractor(class_attr='Result')
        result = extractor.execute(arff_data)
        self.assertEqual(1/54, result['X'][0][10])
        self.assertEqual(52/54, result['X'][0][11])
        self.assertEqual(1/54, result['X'][0][12])
        self.assertEqual(50/20, result['X'][1][10])
        self.assertEqual(43/20, result['X'][1][11])
        self.assertEqual(5/20, result['X'][1][12])

    def test_execute_also_insert_y_labels (self):
        arff_data = arff.load("""@RELATION browserninja.website
@ATTRIBUTE childsNumber NUMERIC
@ATTRIBUTE textLength NUMERIC
@ATTRIBUTE baseX NUMERIC
@ATTRIBUTE targetX NUMERIC
@ATTRIBUTE baseY NUMERIC
@ATTRIBUTE targetY NUMERIC
@ATTRIBUTE baseHeight NUMERIC
@ATTRIBUTE targetHeight NUMERIC
@ATTRIBUTE baseWidth NUMERIC
@ATTRIBUTE targetWidth NUMERIC
@ATTRIBUTE imageDiff NUMERIC
@ATTRIBUTE chiSquared NUMERIC
@ATTRIBUTE baseViewportWidth NUMERIC
@ATTRIBUTE targetViewportWidth NUMERIC
@ATTRIBUTE phash NUMERIC
@ATTRIBUTE xbi {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,380,0.15,1""")
        arff_data['data'] = np.array(arff_data['data'])
        extractor = BrowserNinjaExtractor(class_attr='xbi')
        result = extractor.execute(arff_data)
        self.assertEqual(2, len(result['y']))
        self.assertEqual('0', result['y'][0])
        self.assertEqual('1', result['y'][1])
