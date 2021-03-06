import np

from unittest import TestCase
from unittest.mock import Mock

from pipeline.classifier.classifier_tunning import ClassifierTunning;

class ClassifierTunningTest(TestCase):

    def test_calling_tunning_methods (self):
        grid_mock = Mock()
        grid_mock.best_params_ = { 'params': True }
        data = {
            'X': np.array([[1, 2, 3], [4, 5, 6]]),
            'y': np.array(['none', 'xbi']),
        }
        model = Mock()
        action = ClassifierTunning(grid_mock, model)
        result = action.execute(data)
        grid_mock.fit.assert_called_with(data['X'], data['y'])
        model.set_params.assert_called_with(**grid_mock.best_params_)

        self.assertEqual(model, result['model'])

    def test_calling_tunning_methods_with_groups (self):
        groups = ['group1', 'group2']
        grid_mock = Mock()
        grid_mock.best_params_ = { 'params': True }
        data = {
            'X': np.array([[1, 2, 3], [4, 5, 6]]),
            'y': np.array(['none', 'xbi']),
            'attributes': [ ('URL', 'STRING'), ('id', 'INTEGER'), ('position', 'INTEGER') ],
            'data': np.array([['group1', 1, 2], ['group2', 3, 4]])
        }
        model = Mock()
        action = ClassifierTunning(grid_mock, model, groups='URL')
        result = action.execute(data)
        grid_mock.fit.assert_called_with(data['X'], data['y'], groups=groups)
        model.set_params.assert_called_with(**grid_mock.best_params_)

        self.assertEqual(model, result['model'])
