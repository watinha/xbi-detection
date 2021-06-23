import arff,np

from unittest import TestCase

from pipeline.extractor.browserninja import *

class BrowserNinjaExtractorTest(TestCase):

    def generate_arff(self, data, class_attr='Result', extractors=[]):
        arff_header = """@RELATION browserninja.website
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
@ATTRIBUTE %s {0,1}
@DATA
""" % (class_attr)
        self.extractor = BrowserNinjaCompositeExtractor(class_attr=class_attr, extractors=extractors)
        return arff_header + data


    def test_execute_extracts_features_from_arff(self):
        data = """13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0"""
        arff_data = arff.load(self.generate_arff(data))
        arff_data['data'] = np.array(arff_data['data'])
        result = self.extractor.execute(arff_data)
        self.assertEqual(arff_data['attributes'], result['attributes'])
        self.assertEqual([], result['features'])
        #self.assertEqual(['childsNumber', 'textLength', 'area',
        #                  'phash', 'chiSquared', 'imageDiff',
        #                  'width_comp', 'height_comp',
        #                  'left_visibility', 'right_visibility',
        #                  'left_comp', 'right_comp', 'y_comp'], result['features'])

    def test_extracts_complexity_features(self):
        arff_data = arff.load(self.generate_arff(
            """13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0""",
            extractors=[ ComplexityExtractor() ]))
        arff_data['data'] = np.array(arff_data['data'])
        result = self.extractor.execute(arff_data)
        self.assertEqual(13, result['X'][0][0])
        self.assertEqual(17, result['X'][0][1])
        self.assertEqual(48, result['X'][0][2])
        self.assertEqual(0, result['X'][1][0])
        self.assertEqual(0, result['X'][1][1])
        self.assertEqual(48, result['X'][1][2])
        self.assertEqual(['childsNumber', 'textLength', 'area'], result['features'])

    def test_execute_extracts_image_comparison_features(self):
        arff_data = arff.load(self.generate_arff("""13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,1,2,3,4,5,3,7,10,1000,0.25,360,414,0.15,0""",
            extractors=[ ComplexityExtractor(), ImageComparisonExtractor() ]))
        arff_data['data'] = np.array(arff_data['data'])
        result = self.extractor.execute(arff_data)
        self.assertEqual(0.3, result['X'][0][3])
        self.assertEqual(0.12, result['X'][0][4])
        self.assertEqual(100/(35 * 255), result['X'][0][5])
        self.assertEqual(0.15, result['X'][1][3])
        self.assertEqual(0.25, result['X'][1][4])
        self.assertEqual(1000/(30 * 255), result['X'][1][5])
        self.assertEqual(['childsNumber', 'textLength', 'area',
                          'phash', 'chiSquared', 'imageDiff'], result['features'])

    def test_execute_extracts_image_comparison_features_when_area_is_zero(self):
        arff_data = arff.load(self.generate_arff("""13,17,1,2,3,4,5,6,0,8,100,0.12,360,414,0.3,0
0,0,1,2,3,4,5,3,7,10,1000,0.25,360,414,0.15,0""",
            extractors=[ ComplexityExtractor(), ImageComparisonExtractor() ]))
        arff_data['data'] = np.array(arff_data['data'])
        result = self.extractor.execute(arff_data)
        self.assertEqual(0.3, result['X'][0][3])
        self.assertEqual(0.12, result['X'][0][4])
        self.assertEqual(100/255, result['X'][0][5])
        self.assertEqual(['childsNumber', 'textLength', 'area',
                          'phash', 'chiSquared', 'imageDiff'], result['features'])

    def test_execute_extracts_height_and_width_comparison(self):
        arff_data = arff.load(self.generate_arff("""13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,1,2,3,4,10,15,20,33,1000,0.25,360,380,0.15,0""",
            extractors=[ ComplexityExtractor(), ImageComparisonExtractor(), SizeViewportExtractor() ]))
        arff_data['data'] = np.array(arff_data['data'])
        result = self.extractor.execute(arff_data)
        self.assertEqual(1/54, result['X'][0][6])
        self.assertEqual(1/6, result['X'][0][7])
        self.assertEqual(13/20, result['X'][1][6])
        self.assertEqual(1/3, result['X'][1][7])
        self.assertEqual(['childsNumber', 'textLength', 'area',
                          'phash', 'chiSquared', 'imageDiff',
                          'width_comp', 'height_comp'], result['features'])

    def test_execute_extracts_height_and_width_comparison_when_zero_happens(self):
        arff_data = arff.load(self.generate_arff("""13,17,1,2,3,4,5,6,7,8,100,0.12,360,360,0.3,0
0,0,1,2,3,4,0,0,20,33,1000,0.25,360,380,0.15,0""",
            extractors=[ ComplexityExtractor(), ImageComparisonExtractor(), SizeViewportExtractor() ]))
        arff_data['data'] = np.array(arff_data['data'])
        result = self.extractor.execute(arff_data)
        self.assertEqual(1, result['X'][0][6])
        self.assertEqual(1/6, result['X'][0][7])
        self.assertEqual(13/20, result['X'][1][6])
        self.assertEqual(0, result['X'][1][7])
        self.assertEqual(['childsNumber', 'textLength', 'area',
                          'phash', 'chiSquared', 'imageDiff',
                          'width_comp', 'height_comp'], result['features'])

    def test_execute_extracts_visibility_comparison(self):
        arff_data = arff.load(self.generate_arff("""13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,380,0.15,0""",
            extractors=[ ComplexityExtractor(), ImageComparisonExtractor(), SizeViewportExtractor(), VisibilityExtractor() ]))
        arff_data['data'] = np.array(arff_data['data'])
        result = self.extractor.execute(arff_data)
        # right = (354 - 360) - (406 - 414)
        self.assertEqual(2, result['X'][0][8])
        self.assertEqual(53, result['X'][0][9])
        # right = (240 - 360) - (197 - 380)
        self.assertEqual(63, result['X'][1][8])
        self.assertEqual(-30, result['X'][1][9])
        self.assertEqual(['childsNumber', 'textLength', 'area',
                          'phash', 'chiSquared', 'imageDiff',
                          'width_comp', 'height_comp',
                          'left_visibility', 'right_visibility'], result['features'])

    def test_execute_extracts_visibility_comparison_and_concatenate_other_features(self):
        arff_data = arff.load(self.generate_arff("""13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,380,0.15,0""",
            extractors=[ ComplexityExtractor(), ImageComparisonExtractor(), SizeViewportExtractor(), VisibilityExtractor() ]))
        arff_data['data'] = np.array(arff_data['data'])
        arff_data['X'] = np.array([[1, 7], [7, 7]])
        arff_data['features'] = ['super', 'bash']
        result = self.extractor.execute(arff_data)
        # right = (354 - 360) - (406 - 414)
        self.assertEqual(1, result['X'][0][0])
        self.assertEqual(7, result['X'][0][1])
        self.assertEqual(2, result['X'][0][10])
        self.assertEqual(53, result['X'][0][11])
        # right = (240 - 360) - (197 - 380)
        self.assertEqual(7, result['X'][1][0])
        self.assertEqual(7, result['X'][1][1])
        self.assertEqual(63, result['X'][1][10])
        self.assertEqual(-30, result['X'][1][11])
        self.assertEqual(['super', 'bash',
                          'childsNumber', 'textLength', 'area',
                          'phash', 'chiSquared', 'imageDiff',
                          'width_comp', 'height_comp',
                          'left_visibility', 'right_visibility'], result['features'])

    def test_execute_comparison_based_on_viewport (self):
        arff_data = arff.load(self.generate_arff("""13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,380,0.15,0""",
            extractors=[ ComplexityExtractor(), ImageComparisonExtractor(), SizeViewportExtractor(),
              VisibilityExtractor(), PositionViewportExtractor() ]))
        arff_data['data'] = np.array(arff_data['data'])
        result = self.extractor.execute(arff_data)
        self.assertEqual(1/54, result['X'][0][10])
        self.assertEqual(52/54, result['X'][0][11])
        self.assertEqual(1/54, result['X'][0][12])
        self.assertEqual(50/20, result['X'][1][10])
        self.assertEqual(43/20, result['X'][1][11])
        self.assertEqual(5/20, result['X'][1][12])

    def test_execute_comparison_based_on_viewport_when_zero_happens (self):
        arff_data = arff.load(self.generate_arff("""13,17,1,2,3,4,5,6,7,8,100,0.12,360,360,0.3,0
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,360,0.15,0""",
            extractors=[ ComplexityExtractor(), ImageComparisonExtractor(), SizeViewportExtractor(),
              VisibilityExtractor(), PositionViewportExtractor() ]))
        arff_data['data'] = np.array(arff_data['data'])
        result = self.extractor.execute(arff_data)
        self.assertEqual(1, result['X'][0][10])
        self.assertEqual(2, result['X'][0][11])
        self.assertEqual(1, result['X'][0][12])
        self.assertEqual(50, result['X'][1][10])
        # 240 - 177
        self.assertEqual(63, result['X'][1][11])
        self.assertEqual(5, result['X'][1][12])
        self.assertEqual(['childsNumber', 'textLength', 'area',
                          'phash', 'chiSquared', 'imageDiff',
                          'width_comp', 'height_comp',
                          'left_visibility', 'right_visibility',
                          'left_comp', 'right_comp', 'y_comp'], result['features'])


    def test_execute_also_insert_y_labels (self):
        arff_data = arff.load(self.generate_arff("""13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,380,0.15,1""", class_attr='xbi'))
        arff_data['data'] = np.array(arff_data['data'])
        result = self.extractor.execute(arff_data)
        self.assertEqual(2, len(result['y']))
        self.assertEqual(0, result['y'][0])
        self.assertEqual(1, result['y'][1])

    def test_execute_extracts_image_comparison_features_if_X_is_not_empty(self):
        arff_data = arff.load(self.generate_arff("""13,17,1,2,3,4,5,6,7,8,100,0.12,360,414,0.3,0
0,0,1,2,3,4,5,3,7,10,1000,0.25,360,414,0.15,0""",
            extractors=[ ImageComparisonExtractor() ]))
        arff_data['data'] = np.array(arff_data['data'])
        arff_data['X'] = np.array([[1, 2, 3], [4,  5, 6]])
        result = self.extractor.execute(arff_data)
        self.assertEqual(0.3, result['X'][0][3])
        self.assertEqual(0.12, result['X'][0][4])
        self.assertEqual(100/(35 * 255), result['X'][0][5])
        self.assertEqual(0.15, result['X'][1][3])
        self.assertEqual(0.25, result['X'][1][4])
        self.assertEqual(1000/(30 * 255), result['X'][1][5])
        self.assertEqual(['phash', 'chiSquared', 'imageDiff'], result['features'])

    def test_execute_extracts_platform_ids_from_arff (self):
        arff_data = arff.load("""@RELATION browserninja.website
@ATTRIBUTE basePlatform STRING
@ATTRIBUTE targetPlatform STRING
@ATTRIBUTE Result {0,1}
@DATA
'iOS 12.1 - Safari -- iOS - iPhone 8','Android null - Chrome -- Android - MotoG4',0
'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhone 8 Plus',1
'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhoneSE',1
""")
        arff_data['data'] = np.array(arff_data['data'])
        self.extractor = BrowserNinjaCompositeExtractor(class_attr='Result', extractors=[PlatformExtractor()])
        result = self.extractor.execute(arff_data)
        self.assertEqual(0, result['X'][0][0])
        self.assertEqual(1, result['X'][1][0])
        self.assertEqual(2, result['X'][2][0])
        self.assertEqual(['platform_id'], result['features'])

    def test_execute_extracts_platform_ids_from_arff_with_non_null_parameter (self):
        arff_data = arff.load("""@RELATION browserninja.website
@ATTRIBUTE basePlatform STRING
@ATTRIBUTE targetPlatform STRING
@ATTRIBUTE Result {0,1}
@DATA
'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhoneSE',1
'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhone 8 Plus',1
'iOS 12.1 - Safari -- iOS - iPhone 8','Android null - Chrome -- Android - MotoG4',0
'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhone 8 Plus',1
""")
        arff_data['data'] = np.array(arff_data['data'])
        arff_data['X'] = np.array([ [0, 1], [1, 5], [3, 4], [9, 13] ])
        self.extractor = BrowserNinjaCompositeExtractor(class_attr='Result', extractors=[PlatformExtractor()])
        result = self.extractor.execute(arff_data)
        self.assertEqual(2, result['X'][0][2])
        self.assertEqual(1, result['X'][1][2])
        self.assertEqual(0, result['X'][2][2])
        self.assertEqual(1, result['X'][3][2])
        self.assertEqual(['platform_id'], result['features'])

    def test_execute_extracts_platform_ids_with_complexity_extractor (self):
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
@ATTRIBUTE basePlatform STRING
@ATTRIBUTE targetPlatform STRING
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,360,0.3,'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhoneSE',1
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,360,0.15,'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhone 8 Plus',1
""")
        arff_data['data'] = np.array(arff_data['data'])
        self.extractor = BrowserNinjaCompositeExtractor(class_attr='Result',
                extractors=[PlatformExtractor(), ComplexityExtractor()])
        result = self.extractor.execute(arff_data)
        self.assertEqual(1, result['X'][0][0])
        self.assertEqual(0, result['X'][1][0])
        self.assertEqual(['platform_id', 'childsNumber', 'textLength', 'area'], result['features'])

    def test_execute_extracts_platform_ids_with_image_extractor (self):
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
@ATTRIBUTE basePlatform STRING
@ATTRIBUTE targetPlatform STRING
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,360,0.3,'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhoneSE',1
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,360,0.15,'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhone 8 Plus',1
""")
        arff_data['data'] = np.array(arff_data['data'])
        self.extractor = BrowserNinjaCompositeExtractor(class_attr='Result',
                extractors=[ImageComparisonExtractor(), PlatformExtractor()])
        result = self.extractor.execute(arff_data)
        self.assertEqual(1, result['X'][0][3])
        self.assertEqual(0, result['X'][1][3])
        self.assertEqual(['phash', 'chiSquared', 'imageDiff', 'platform_id'], result['features'])

    def test_execute_extracts_platform_ids_with_size_viewport_extractor (self):
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
@ATTRIBUTE basePlatform STRING
@ATTRIBUTE targetPlatform STRING
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,360,0.3,'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhoneSE',1
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,360,0.15,'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhone 8 Plus',1
""")
        arff_data['data'] = np.array(arff_data['data'])
        self.extractor = BrowserNinjaCompositeExtractor(class_attr='Result',
                extractors=[PlatformExtractor(), SizeViewportExtractor()])
        result = self.extractor.execute(arff_data)
        self.assertEqual(1, result['X'][0][0])
        self.assertEqual(0, result['X'][1][0])
        self.assertEqual(['platform_id', 'width_comp', 'height_comp'], result['features'])

    def test_execute_extracts_platform_ids_with_visibility_extractor (self):
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
@ATTRIBUTE basePlatform STRING
@ATTRIBUTE targetPlatform STRING
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,360,0.3,'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhoneSE',1
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,360,0.15,'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhone 8 Plus',1
""")
        arff_data['data'] = np.array(arff_data['data'])
        self.extractor = BrowserNinjaCompositeExtractor(class_attr='Result',
                extractors=[PlatformExtractor(), VisibilityExtractor()])
        result = self.extractor.execute(arff_data)
        self.assertEqual(1, result['X'][0][0])
        self.assertEqual(0, result['X'][1][0])
        self.assertEqual(['platform_id', 'left_visibility', 'right_visibility'], result['features'])

    def test_execute_extracts_platform_ids_with_position_viewport (self):
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
@ATTRIBUTE basePlatform STRING
@ATTRIBUTE targetPlatform STRING
@ATTRIBUTE Result {0,1}
@DATA
13,17,1,2,3,4,5,6,7,8,100,0.12,360,360,0.3,'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhoneSE',1
0,0,100,150,20,15,10,15,20,33,1000,0.25,360,360,0.15,'iOS 12.1 - Safari -- iOS - iPhone 8','iOS 12.1 - Safari -- iOS - iPhone 8 Plus',1
""")
        arff_data['data'] = np.array(arff_data['data'])
        self.extractor = BrowserNinjaCompositeExtractor(class_attr='Result',
                extractors=[PlatformExtractor(), PositionViewportExtractor()])
        result = self.extractor.execute(arff_data)
        self.assertEqual(1, result['X'][0][0])
        self.assertEqual(0, result['X'][1][0])
        self.assertEqual(['platform_id', 'left_comp', 'right_comp', 'y_comp'], result['features'])
