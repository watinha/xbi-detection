import arff, np

from unittest import TestCase

from pipeline.extractor.xbi_extractor import XBIExtractor


class XBIExtractorTest(TestCase):

    def test_extracts_features_from_dataset (self):
        arff_example = """@RELATION iris
@ATTRIBUTE aaa  NUMERIC
@ATTRIBUTE width   NUMERIC
@ATTRIBUTE bbb  NUMERIC
@ATTRIBUTE height   NUMERIC
@ATTRIBUTE class        {Iris-setosa,Iris-versicolor,Iris-virginica}
@DATA
5.1,3.5,1.4,0.2,Iris-setosa"""
        data = arff.load(arff_example)
        data['data'] = np.array(data['data'])
        extractor = XBIExtractor(
                ['width', 'height'], 'class')
        result = extractor.execute(data)
        self.assertEqual(['width', 'height'],
                result['features'])
        self.assertEqual(1, len(result['X']))
        self.assertEqual(2, len(result['X'][0]))
        self.assertEqual('3.5', result['X'][0][0])
        self.assertEqual('0.2', result['X'][0][1])
        self.assertEqual(1, len(result['y']))
        self.assertEqual('Iris-setosa', result['y'][0])

    def test_extracts_features_from_different_data (self):
        arff_example = """@RELATION otherdata
@ATTRIBUTE supimpa  NUMERIC
@ATTRIBUTE nothing   NUMERIC
@ATTRIBUTE anotherr  NUMERIC
@ATTRIBUTE somethingelse   NUMERIC
@ATTRIBUTE anotherthing   NUMERIC
@ATTRIBUTE result        {xbi, none}
@DATA
1,2,3,4,5,none
6,7,8,9,10,none
11,12,13,14,15,none
16,17,18,19,20,none
21,22,23,24,25,none
26,27,28,29,30,xbi"""
        data = arff.load(arff_example)
        data['data'] = np.array(data['data'])
        extractor = XBIExtractor(
            ['supimpa', 'somethingelse', 'nothing'],
                'result')
        result = extractor.execute(data)
        self.assertEqual(
                ['supimpa', 'somethingelse', 'nothing'],
                result['features'])
        self.assertEqual(6, len(result['X']))
        self.assertEqual(3, len(result['X'][3]))
        self.assertEqual('21.0', result['X'][4][0])
        self.assertEqual('24.0', result['X'][4][1])
        self.assertEqual('22.0', result['X'][4][2])
        self.assertEqual(6, len(result['y']))
        self.assertEqual('none', result['y'][0])
        self.assertEqual('xbi', result['y'][5])
