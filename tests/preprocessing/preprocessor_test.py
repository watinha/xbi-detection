import numpy as np

from unittest import TestCase
from unittest.mock import Mock

from pipeline.preprocessing import Preprocessor

class PreprocessorTest (TestCase):

    def test_execute_runs_method_and_returns_0 (self):
        X = np.array([[ 0, 0], [ 0.,  0.]])
        X_result = np.array([[ 0   ,  0],
			     [ 0   ,  0]])
        arguments = { 'X': X, 'y': 'result' }
        preprocessor = Preprocessor()
        result = preprocessor.execute(arguments)
        self.assertAlmostEqual(X_result[0][0], result['X'][0][0], 2)
        self.assertAlmostEqual(X_result[0][1], result['X'][0][1], 2)
        self.assertAlmostEqual(X_result[1][0], result['X'][1][0], 2)
        self.assertAlmostEqual(X_result[1][1], result['X'][1][1], 2)

    def test_execute_runs_method_and_returns_value (self):
        X = np.array([[ 1., -1.,  2.], [ 2.,  0.,  0.], [ 0.,  1., -1.]])
        X_result = np.array([[ 0   , -1.22,  1.34],
			     [ 1.22,  0   , -0.27],
			     [-1.22,  1.22, -1.07]])
        arguments = { 'X': X, 'y': 'result' }
        preprocessor = Preprocessor()
        result = preprocessor.execute(arguments)
        self.assertAlmostEqual(X_result[0][0], result['X'][0][0], 2)
        self.assertAlmostEqual(X_result[0][1], result['X'][0][1], 2)
        self.assertAlmostEqual(X_result[0][2], result['X'][0][2], 2)
        self.assertAlmostEqual(X_result[1][0], result['X'][1][0], 2)
        self.assertAlmostEqual(X_result[1][1], result['X'][1][1], 2)
        self.assertAlmostEqual(X_result[1][2], result['X'][1][2], 2)
        self.assertAlmostEqual(X_result[2][0], result['X'][2][0], 2)
        self.assertAlmostEqual(X_result[2][1], result['X'][2][1], 2)
        self.assertAlmostEqual(X_result[2][2], result['X'][2][2], 2)
