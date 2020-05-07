import numpy as np

from unittest import TestCase
from unittest.mock import Mock

from pipeline.feature_selection import FeatureSelection

class FeatureSelectionTest(TestCase):

    def test_execute_runs_fit_and_transform_in_x_data (self):
        stub_x = np.array([[1, 2, 3], [4, 5, 6]])
        stub_y = 'another thing'
        arguments = { 'X': stub_x, 'y': stub_y }
        sklearn_mock = Mock()
        sklearn_mock.fit_transform = Mock(return_value='transformed')

        feature_selection = FeatureSelection(sklearn_mock, k=2)
        result = feature_selection.execute(arguments)

        sklearn_mock.fit_transform.assert_called_with(stub_x, stub_y)
        self.assertEqual('transformed', result['X'])
        self.assertEqual(arguments['y'], result['y'])

    def test_execute_should_not_run_if_matrix_minor_than_k (self):
        stub_x = np.array([[0, 1], [1, 2]])
        stub_y = 'another thing'
        arguments = { 'X': stub_x, 'y': stub_y }
        sklearn_mock = Mock()
        sklearn_mock.fit_transform = Mock(return_value='transformed')

        feature_selection = FeatureSelection(sklearn_mock, k=5)
        result = feature_selection.execute(arguments)

        sklearn_mock.fit_transform.assert_not_called()
        self.assertEqual(arguments, result)
