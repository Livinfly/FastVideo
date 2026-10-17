"""Failure notifications must survive later success and support older suites."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from typing import Any

from common import read_json, write_json
from monitor import show
from state import status_view, write_status


class StateTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.suite = Path(self.temp.name)
        self.group: dict[str, Any] = dict(gpus=2, model='fasth3_v1', directory='fasth3_v1/g2')
        self.directory = self.suite / self.group['directory']
        self.directory.mkdir(parents=True)
        self.state: dict[str, Any] = dict(status='running', pid=os.getpid(), total=2, completed=[], current=self.group)
        write_json(self.suite / 'plan.json', dict(groups=[self.group], modes=['dit'], warmup=1, repeats=1))

    def tearDown(self):
        self.temp.cleanup()

    def rank_failure(self) -> None:
        write_json(
            self.directory / 'error_rank1.json',
            dict(status='memory_insufficient',
                 phase='warmup',
                 traceback=
                 'Traceback (most recent call last):\n  native code\ntorch.OutOfMemoryError: CUDA out of memory\n'))

    def test_success_has_no_failure(self):
        write_status(self.suite, self.state)
        state = read_json(self.suite / 'status.json')
        self.assertFalse(state['has_failures'])
        self.assertEqual(state['failure_count'], 0)
        self.assertIsNone(state['last_error'])
        self.assertIn('updated_at_utc', state)

    def test_rank_failure_survives_later_success(self):
        self.rank_failure()
        self.state['completed'].append(dict(**self.group, status='memory_insufficient', exit_code=1))
        write_status(self.suite, self.state)
        self.state['current'] = None
        self.state['completed'].append(dict(gpus=1, directory='fasth3_v1/g1', status='complete', exit_code=0))
        self.state['status'] = 'complete_with_failures'
        write_status(self.suite, self.state)
        state = read_json(self.suite / 'status.json')
        self.assertTrue(state['has_failures'])
        self.assertEqual(state['failure_count'], 1)
        self.assertIn('CUDA out of memory', state['last_error']['summary'])
        self.assertEqual(state['last_error']['phase'], 'warmup')
        self.assertEqual(state['last_error']['log_file'], 'fasth3_v1/g2/inference.log')
        self.assertEqual(state['last_error']['error_files'], ['fasth3_v1/g2/error_rank1.json'])

    def test_killed_worker_has_fallback_and_exit_code(self):
        self.state['completed'].append(dict(**self.group, status='failed', exit_code=-9))
        write_status(self.suite, self.state)
        state = read_json(self.suite / 'status.json')
        self.assertTrue(state['has_failures'])
        self.assertEqual(state['completed'][0]['exit_code'], -9)
        self.assertEqual(state['last_error']['error_files'], [])
        self.assertIn('inference.log', state['last_error']['summary'])

    def test_cuda_advice_does_not_replace_error_summary(self):
        write_json(
            self.directory / 'error_rank0.json',
            dict(phase='load',
                 traceback='Traceback (most recent call last):\n'
                 'torch.AcceleratorError: CUDA error: invalid argument\n'
                 'For debugging consider passing CUDA_LAUNCH_BLOCKING=1\n'
                 'Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n'))
        view = status_view(self.suite, self.state)
        self.assertEqual(view['last_error']['summary'], 'torch.AcceleratorError: CUDA error: invalid argument')

    def test_coordinator_error_without_worker(self):
        self.state.update(status='interrupted_or_failed', current=None)
        write_status(self.suite, self.state, error=RuntimeError('GPU query failed'))
        state = read_json(self.suite / 'status.json')
        self.assertTrue(state['has_failures'])
        self.assertEqual(state['last_error']['summary'], 'RuntimeError: GPU query failed')
        self.assertEqual(state['last_error']['phase'], 'coordinator')

    def test_legacy_monitor_reports_failure_without_rewriting(self):
        self.rank_failure()
        write_json(self.suite / 'status.json', self.state)
        original = (self.suite / 'status.json').read_bytes()
        view = status_view(self.suite, self.state)
        self.assertEqual(view['failure_count'], 1)
        self.assertNotIn('has_failures', self.state)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertTrue(show(self.suite))
        self.assertIn('CUDA out of memory', output.getvalue())
        self.assertIn('error_rank1.json', output.getvalue())
        self.assertEqual((self.suite / 'status.json').read_bytes(), original)

    def test_legacy_coordinator_failure_has_a_notice(self):
        self.state.update(status='interrupted_or_failed', current=None)
        write_json(self.suite / 'status.json', self.state)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertFalse(show(self.suite))
        self.assertIn('Run ended with status interrupted_or_failed', output.getvalue())


if __name__ == '__main__':
    unittest.main()
