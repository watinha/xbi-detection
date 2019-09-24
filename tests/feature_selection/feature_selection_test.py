from unittest import TestCase
from unittest.mock import Mock

from pipeline.feature_selection import FeatureSelection

class FeatureSelectionTest(TestCase):

    def test_execute_runs_fit_and_transform_in_x_data (self):
        stub_x = 'something'
        stub_y = 'another thing'
        arguments = { 'X': stub_x, 'y': stub_y }
        sklearn_mock = Mock()
        sklearn_mock.fit_transform = Mock(return_value='transformed')

        feature_selection = FeatureSelection(sklearn_mock)
        result = feature_selection.execute(arguments)

        sklearn_mock.fit_transform.assert_called_with('something', 'another thing')
        self.assertEqual('transformed', result['X'])
        self.assertEqual(arguments['y'], result['y'])
