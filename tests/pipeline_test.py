from pipeline import Pipeline

from unittest import TestCase
from unittest.mock import Mock

class PipelineTest(TestCase):

    def test_empty_pipeline(self):
        initial_args = {}
        pipeline = Pipeline([])
        result = pipeline.execute(initial_args);
        self.assertEqual(0, len(pipeline._actions))
        self.assertEqual(initial_args, result)

    def test_single_action(self):
        initial_args = {}
        action_mock = Mock()
        action_mock.execute = Mock(return_value='somathin')
        pipeline = Pipeline([ action_mock ])
        result = pipeline.execute(initial_args)
        action_mock.execute.assert_called_once_with(initial_args)
        self.assertEqual('somathin', result)

    def test_two_actions(self):
        arguments = {}
        action1 = Mock()
        action1.execute = Mock(return_value='somathin')
        action2 = Mock()
        action2.execute = Mock(return_value='other thing')
        pipeline = Pipeline([ action1, action2 ])
        result = pipeline.execute(arguments)
        action1.execute.assert_called_once_with(arguments)
        action2.execute.assert_called_once_with('somathin')
        self.assertEqual('other thing', result)
