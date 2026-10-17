"""GPU admission checks without a CUDA context, plus predecessor and lock failures."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from typing import Any
from unittest.mock import Mock, patch

import gpu_wait


class GPUWaitTests(unittest.TestCase):

    def setUp(self):
        self.idle = [dict(index='0', uuid='GPU-test', memory_mib=0, utilization=0)]

    def test_selection_and_all_busy_signals(self):
        self.assertEqual(gpu_wait.select_devices(self.idle, '0'), ['GPU-test'])
        self.assertEqual(gpu_wait.select_devices(self.idle, 'GPU-t'), ['GPU-test'])
        for visible in ['', '0,0', '1']:
            with self.assertRaises(ValueError):
                gpu_wait.select_devices(self.idle, visible)
        self.assertFalse(gpu_wait.busy_reasons(self.idle, [], ['GPU-test']))
        self.assertTrue(gpu_wait.busy_reasons(self.idle, [['GPU-test', '12']], ['GPU-test']))
        for change in [dict(memory_mib=257), dict(utilization=1)]:
            self.assertTrue(gpu_wait.busy_reasons([{**self.idle[0], **change}], [], ['GPU-test']))

    def test_shared_lock_excludes_independently_imported_launcher(self):
        spec = importlib.util.spec_from_file_location('other_framework_gpu_wait', gpu_wait.__file__)
        assert spec is not None and spec.loader is not None
        other = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(other)
        self.assertEqual(other.LOCK_ROOT, gpu_wait.LOCK_ROOT)
        with tempfile.TemporaryDirectory() as folder, patch.object(gpu_wait,'LOCK_ROOT',Path(folder)), \
                patch.object(other,'LOCK_ROOT',Path(folder)), \
                patch.object(gpu_wait,'inventory',return_value=self.idle), \
                patch.object(gpu_wait,'query',return_value=[]), patch.object(gpu_wait.time,'sleep'):
            notices: list[Any] = []
            handles = gpu_wait.wait_for_idle(['GPU-test'], notices.append)
            self.assertEqual(notices, [[dict(reason='confirming GPUs remain idle')]])

            def abort(reasons):
                self.assertIn('holds its lock', reasons[0]['reason'])
                raise RuntimeError('blocked by lock')

            try:
                with self.assertRaisesRegex(RuntimeError, 'blocked by lock'):
                    other.wait_for_idle(['GPU-test'], abort)
            finally:
                for handle in handles:
                    handle.close()
            handles = gpu_wait.wait_for_idle(['GPU-test'], lambda reasons: None)
            for handle in handles:
                handle.close()

    def test_numeric_mask_preserves_uuid_selection_order(self):
        devices = [dict(index='0', uuid='GPU-a'), dict(index='3', uuid='GPU-b')]
        self.assertEqual(gpu_wait.numeric_visible_devices(devices, ['GPU-b', 'GPU-a']), '3,0')
        with self.assertRaises(KeyError):
            gpu_wait.numeric_visible_devices(devices, ['GPU-missing'])

    def test_device_identity_rejects_cuda_nvml_order_mismatch(self):
        cuda = SimpleNamespace(set_device=Mock(), get_device_properties=Mock(return_value=SimpleNamespace(uuid='abcd')))
        torch = SimpleNamespace(cuda=cuda)
        gpu_wait.check_device_identity(torch, 1, 'GPU-ABCD')
        cuda.set_device.assert_called_once_with(1)
        cuda.get_device_properties.assert_called_once_with(1)
        with self.assertRaisesRegex(RuntimeError, 'CUDA/NVML GPU order mismatch'):
            gpu_wait.check_device_identity(torch, 1, 'GPU-other')

    def test_waits_for_busy_gpu_and_checks_idle_twice(self):
        busy = [{**self.idle[0], 'utilization': 100}]
        with tempfile.TemporaryDirectory() as folder, patch.object(gpu_wait,'LOCK_ROOT',Path(folder)), \
                patch.object(gpu_wait,'inventory',side_effect=[busy,self.idle,self.idle]) as inv, \
                patch.object(gpu_wait,'query',return_value=[]), patch.object(gpu_wait.time,'sleep'):
            notices: list[Any] = []
            handles = gpu_wait.wait_for_idle(['GPU-test'], notices.append)
            for handle in handles:
                handle.close()
            self.assertEqual(inv.call_count, 3)
            self.assertEqual(notices[0][0]['utilization'], 100)

    def test_query_failure_does_not_launch_and_releases_locks(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(gpu_wait,'LOCK_ROOT',Path(folder)), \
                patch.object(gpu_wait,'inventory',side_effect=RuntimeError('nvidia-smi failed')):
            with self.assertRaisesRegex(RuntimeError, 'nvidia-smi failed'):
                gpu_wait.wait_for_idle(['GPU-test'], lambda reasons: None)
            with patch.object(gpu_wait,'inventory',return_value=self.idle), \
                    patch.object(gpu_wait,'query',return_value=[]), patch.object(gpu_wait.time,'sleep'):
                handles = gpu_wait.wait_for_idle(['GPU-test'], lambda reasons: None)
                for handle in handles:
                    handle.close()

    def test_predecessor_wait_and_failures(self):
        with patch.object(gpu_wait,'read_json',side_effect=[dict(status='running',pid=12),dict(status='complete')]), \
                patch.object(gpu_wait.os,'kill') as kill, patch.object(gpu_wait.time,'sleep'):
            notices: list[Any] = []
            gpu_wait.wait_for_suite(Path('previous/status.json'), notices.append)
            kill.assert_called_once_with(12, 0)
            self.assertEqual(len(notices), 1)
        with patch.object(gpu_wait, 'read_json', return_value=dict(status='interrupted_or_failed')), \
                self.assertRaisesRegex(RuntimeError, 'Preceding suite failed'):
            gpu_wait.wait_for_suite(Path('previous/status.json'), lambda r: None)
        with patch.object(gpu_wait,'read_json',return_value=dict(status='running',pid=12)), \
                patch.object(gpu_wait.os,'kill',side_effect=ProcessLookupError), self.assertRaises(ProcessLookupError):
            gpu_wait.wait_for_suite(Path('previous/status.json'), lambda r: None)


if __name__ == '__main__':
    unittest.main()
