from pipeline.loader.arff_loader import ArffLoader

from unittest import TestCase

class ArffLoaderTest(TestCase):

    def test_execute_loads_arff_with_three_rows (self):
        arff_example = """@RELATION iris
@ATTRIBUTE sepallength  NUMERIC
@ATTRIBUTE sepalwidth   NUMERIC
@ATTRIBUTE petallength  NUMERIC
@ATTRIBUTE petalwidth   NUMERIC
@ATTRIBUTE class        {Iris-setosa,Iris-versicolor,Iris-virginica}
@DATA
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa"""
        loader = ArffLoader()
        result = loader.execute(arff_example)
        self.assertEqual(3, len(result['data']))
        self.assertEqual('4.9', result['data'][1][0])
        self.assertEqual(5, len(result['attributes']))
        self.assertEqual('sepalwidth', result['attributes'][1][0])
        self.assertEqual('<U32', result['data'].dtype)

    def test_execute_loads_arff_with_two_rows (self):
        arff_example = """@RELATION iris
@ATTRIBUTE sepallength  NUMERIC
@ATTRIBUTE sepalwidth   NUMERIC
@ATTRIBUTE another      NUMERIC
@ATTRIBUTE class        {Iris-setosa,Iris-versicolor,Iris-virginica}
@DATA
5.1,3.5,1.4,Iris-setosa
4.7,3.2,1.3,Iris-setosa"""
        loader = ArffLoader()
        result = loader.execute(arff_example)
        self.assertEqual(2, len(result['data']))
        self.assertEqual('3.5', result['data'][0][1])
        self.assertEqual(4, len(result['attributes']))
        self.assertEqual('another', result['attributes'][2][0])
        self.assertEqual('<U32', result['data'].dtype)
