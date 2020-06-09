import np

from unittest import TestCase
from unittest.mock import Mock,patch

from pipeline.model_evaluation.groupkfold_cv import GroupKFoldCV

class GroupKFoldCVTest(TestCase):

    def test_cross_val_score_is_called_with_parameters (self):
        args = {
            'attributes': [('URL', 'STRING'), ('id', 'NUMERIC')],
            'data': np.array([['abobrinha', 2], ['pepino', 3]]),
            'X': np.array([[1, 2, 3], [4, 5, 6]]),
            'y': np.array([1, 2]),
            'model': {}
        }
        score_stub = {}
        folds_stub = {}
        cross_val_score_mock = Mock()
        cross_val_score_mock.return_value = score_stub
        action = GroupKFoldCV(folds_stub, 'URL', cross_val_score_mock)
        result = action.execute(args)
        [(model, X, y), keyed_args] = cross_val_score_mock.call_args
        self.assertEqual(args['model'], model)
        np.testing.assert_array_equal(args['X'], X)
        np.testing.assert_array_equal(args['y'], y)
        self.assertEqual(folds_stub, keyed_args['cv'])
        self.assertEqual(['abobrinha', 'pepino'], keyed_args['groups'])
        self.assertEqual(['f1', 'precision', 'recall', 'roc_auc'], keyed_args['scoring'])
        self.assertEqual(score_stub, result['score'])

    def test_cross_val_score_is_called_with_different_groups (self):
        args = {
            'attributes': [('URL', 'STRING'), ('id', 'NUMERIC')],
            'data': np.array([['abobrinha', 2], ['pepino', 3], ['negativo', 4]]),
            'X': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            'y': np.array([1, 2, 3]),
            'model': Mock()
        }
        score_stub = {}
        folds_stub = {}
        cross_val_score_mock = Mock()
        cross_val_score_mock.return_value = score_stub
        action = GroupKFoldCV(folds_stub, 'id', cross_val_score_mock)
        result = action.execute(args)
        [(model, X, y), keyed_args] = cross_val_score_mock.call_args
        self.assertEqual(args['model'], model)
        np.testing.assert_array_equal(args['X'], X)
        np.testing.assert_array_equal(args['y'], y)
        self.assertEqual(folds_stub, keyed_args['cv'])
        self.assertEqual(['2', '3', '4'], keyed_args['groups'])
        self.assertEqual(['f1', 'precision', 'recall', 'roc_auc'], keyed_args['scoring'])
        self.assertEqual(score_stub, result['score'])
