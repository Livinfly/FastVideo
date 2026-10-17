"""CPU checks for model isolation, portable launch and timing boundaries (no GPU workers)."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, TextIO
import unittest
from unittest.mock import MagicMock, patch

from common import HERE, aggregate, read_json, schedule, write_json
from config import validate_profile
from report import collect, generate_report
import run as launcher
from timing import DiTEvents
import worker


def record(mode='dit', warmup=False, repeat=1, seconds=10) -> dict[str, Any]:
    return aggregate([
        dict(rank=i,
             response_e2e_s=seconds if i == 0 else None,
             distributed_e2e_s=seconds + i,
             dit_step_ms=steps if mode == 'dit' else None,
             peak_allocated_gib=1,
             peak_reserved_gib=2) for i, steps in enumerate([[1000, 3000], [2000, 1000]])
    ],
                     mode=mode,
                     tag='test',
                     warmup=warmup,
                     repeat=repeat)


class BenchmarkTests(unittest.TestCase):

    def test_preferred_dense_attention_rejects_fallback(self):
        from runtime import dense_attention_info

        class FlashAttentionImpl:
            pass

        class SDPAAttentionImpl:
            pass

        for name in ['blocks.0.attn2.attn', 'token_refiner.blocks.0.attn.distributed_attention']:
            module = SimpleNamespace(attn_impl=FlashAttentionImpl())
            transformer = SimpleNamespace(named_modules=lambda name=name, module=module: [(name, module)])
            self.assertEqual(dense_attention_info(transformer)[0]['implementation'], 'FlashAttentionImpl')
            module.attn_impl = SDPAAttentionImpl()
            with self.assertRaisesRegex(ValueError, 'did not select FlashAttention'):
                dense_attention_info(transformer)
        with self.assertRaisesRegex(ValueError, 'did not select FlashAttention'):
            dense_attention_info(SimpleNamespace(named_modules=lambda: []))

    def test_native_config_defaults_and_serialization(self):
        from dataclasses import asdict
        import json
        from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
        from runtime import generator_config, json_config
        for version in ['v1', 'v2']:
            profile = read_json(HERE / f'configs/fasth3_{version}.json')
            config = generator_config(profile, '/example/model', 4)
            self.assertEqual(config.engine.parallelism.sp_size, 4)
            self.assertTrue(config.engine.offload.dit_layerwise)
            self.assertFalse(config.engine.offload.pin_cpu_memory)
            self.assertFalse(config.engine.compile.enabled)
            self.assertFalse(config.engine.use_fsdp_inference)
        encoded = json.dumps(json_config(asdict(MiniMaxH3PipelineConfig())))
        self.assertNotIn(' at 0x', encoded)
        self.assertIn('bf16', encoded)

    def test_profiles_reject_wrong_grid_dense_and_stage_sync(self):
        for name, count in [('v1', 4), ('v2', 8)]:
            profile = read_json(HERE / f'configs/fasth3_{name}.json')
            validate_profile(profile)
            self.assertEqual(len(profile['contract']['dmd_denoising_steps']), count)
            self.assertEqual(profile['sampling']['num_inference_steps'], count + 1)
            for field, value in [('num_inference_steps', count), ('guidance_scale', 5)]:
                bad = copy.deepcopy(profile)
                bad['sampling'][field] = value
                with self.assertRaises(ValueError):
                    validate_profile(bad)
            bad = copy.deepcopy(profile)
            bad['environment']['FASTVIDEO_STAGE_LOGGING'] = '1'
            with self.assertRaises(ValueError):
                validate_profile(bad)
            bad = copy.deepcopy(profile)
            bad['pipeline']['experimental']['VSA_sparsity'] = 0
            with self.assertRaises(ValueError):
                validate_profile(bad)

    def test_timer_includes_existing_root_hooks_and_has_no_step_sync(self):
        import torch
        trace = []

        class Event:

            def __init__(self, **kwargs):
                pass

            def record(self):
                trace.append('event')

            def elapsed_time(self, other):
                return 1.

        class Model(torch.nn.Module):

            def forward(self, x):
                trace.append('forward')
                return x

        model = Model()
        model.register_forward_pre_hook(lambda *a: trace.append('fsdp_pre'))
        model.register_forward_hook(lambda *a: trace.append('fsdp_post'))
        fake = SimpleNamespace(cuda=SimpleNamespace(Event=Event, synchronize=lambda: trace.append('sync')))
        timer = DiTEvents(fake, 1)
        self.assertEqual(trace.count('sync'), 1)
        timer.attach(model)
        trace.clear()
        model(1)
        self.assertEqual(trace, ['event', 'fsdp_pre', 'forward', 'fsdp_post', 'event'])
        self.assertEqual(timer.elapsed_ms(), [1.])
        timer.detach()
        trace.clear()
        model(1)
        self.assertEqual(trace, ['fsdp_pre', 'forward', 'fsdp_post'])

    def test_model_speedups_are_isolated_and_history_is_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            suite = Path(folder)
            groups: list[dict[str, Any]] = [
                dict(model=model, gpus=n, directory=f'{model}/g{n}') for model in ['v1', 'v2'] for n in [2, 1]
            ]
            plan = dict(groups=groups,
                        modes=['dit', 'e2e'],
                        warmup=1,
                        repeats=1,
                        models={},
                        source_branch='exp-fixed-version',
                        source_commit='test',
                        inference_source_modified=False,
                        validate_video=False,
                        timing_schema_version=3)
            write_json(suite / 'plan.json', plan)
            write_json(suite / 'status.json',
                       dict(status='complete', total=4, completed=[dict(g, status="complete") for g in groups]))
            for group in groups:
                path = suite / group['directory']
                path.mkdir(parents=True)
                seconds = (10 if group['model'] == 'v1' else 90) / group['gpus']
                records = [
                    record(mode, warmup, repeat, seconds) for mode, warmup, repeat in schedule(plan['modes'], 1, 1)
                ]
                write_json(path / 'metrics.json', dict(status='complete', records=records))
            rows = collect(suite, plan)
            for row in rows:
                self.assertEqual(row['e2e_speedup'], row['gpus'])
                self.assertIsNone(row['response_e2e_s']['stdev'])
            generate_report(suite)
            self.assertIn('单次', (suite / 'report.md').read_text())
            self.assertEqual(len(read_json(suite / 'summary.json')['timing_comparisons']), 4)
            # An archived baseline alone must never turn on historical comparison.
            shutil.copy2(suite / 'summary.json', suite / 'historical_baseline.json')
            generate_report(suite)
            self.assertNotIn('## 历史对比', (suite / 'report.md').read_text())
            path = suite / 'v1/g1/metrics.json'
            data = read_json(path)
            data['records'].pop()
            write_json(path, data)
            self.assertNotIn('e2e_speedup',
                             next(r for r in collect(suite, plan) if r['model'] == 'v1' and r['mode'] == 'e2e'))

    def test_relocated_dry_run(self):
        with tempfile.TemporaryDirectory(prefix='fastvideo space ') as folder:
            checkout = Path(folder) / 'clone'
            shutil.copytree(HERE, checkout / 'benchmarks', ignore=shutil.ignore_patterns('__pycache__'))
            command = [
                sys.executable,
                str(checkout / 'benchmarks/run.py'), '--gpus', '8', '1', '--dry-run', '--model-path',
                'fasth3_v1=relative model1', '--model-path', 'fasth3_v2=relative model2'
            ]
            result = subprocess.run(command, cwd=folder, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), 4)
            self.assertIn('fasth3_v2/g8', result.stdout)
            self.assertFalse((checkout / 'outputs').exists())


class LauncherTests(unittest.TestCase):

    def exercise(self, failure=None, phase='load') -> None:
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            folder = Path(temporary)
            suite = folder / 'suite'
            argv = [
                'run.py', '--gpus', '2', '1', '--output',
                str(suite), '--model-path', 'fasth3_v1=' + str(folder), '--model-path', 'fasth3_v2=' + str(folder)
            ]
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch.object(launcher, 'model_manifest', return_value={}))
            stack.enter_context(
                patch.object(launcher,
                             'inventory',
                             return_value=[dict(index=str(i), uuid=f'GPU-{i}') for i in range(2)]))
            stack.enter_context(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1,0'}))
            stack.enter_context(
                patch.object(launcher.subprocess, 'check_output', side_effect=['exp-fixed-version\n', 'commit\n', '']))
            stack.enter_context(patch.object(launcher, 'wait_for_suite'))
            signals = stack.enter_context(patch.object(launcher.signal, 'signal'))
            locks: list[TextIO] = []
            launched = []

            def wait(selected, notify):
                self.assertEqual(selected, ['GPU-1', 'GPU-0'][:len(selected)])
                notify([dict(reason='busy')])
                self.assertEqual(read_json(suite / 'status.json')['status'], 'waiting')
                handle = (folder / f'lock{len(locks)}').open('w')
                locks.append(handle)
                return [handle]

            def popen(command, **kwargs):
                self.assertFalse(locks[-1].closed)
                directory = Path(command[command.index('--output') + 1])
                launched.append(directory)
                self.assertEqual(read_json(suite / 'status.json')['status'], 'running')
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '1,0' if directory.name == 'g2' else '1')
                self.assertEqual(kwargs['env']['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
                if failure == 'interrupt' and len(launched) == 1:

                    def interrupted_wait(**kwargs):
                        signals.call_args.args[1](launcher.signal.SIGTERM, None)

                    return SimpleNamespace(pid=123456789, wait=interrupted_wait)
                if failure and len(launched) == 1:
                    write_json(directory / 'error_rank0.json',
                               dict(status=failure, phase=phase, traceback='RuntimeError: test'))
                    return SimpleNamespace(pid=123456789, wait=lambda **kwargs: 1)
                write_json(directory / 'metrics.json', dict(status='complete', records=[record(warmup=True), record()]))
                return SimpleNamespace(pid=123456789, wait=lambda **kwargs: 0)

            stack.enter_context(patch.object(launcher, 'wait_for_idle', side_effect=wait))
            stack.enter_context(patch.object(launcher.subprocess, 'Popen', side_effect=popen))
            with redirect_stdout(io.StringIO()):
                if failure == 'interrupt':
                    with self.assertRaises(KeyboardInterrupt):
                        launcher.main()
                elif failure == 'memory_insufficient' and phase in {'load', 'warmup'}:
                    with self.assertRaises(SystemExit):
                        launcher.main()
                elif failure:
                    with self.assertRaisesRegex(RuntimeError, 'Group failed'):
                        launcher.main()
                else:
                    launcher.main()
            self.assertTrue(all(h.closed for h in locks))
            state = read_json(suite / 'status.json')
            self.assertEqual(state['has_failures'], bool(failure))
            stopped = failure and not (failure == 'memory_insufficient' and phase in {'load', 'warmup'})
            self.assertEqual(len(launched), 1 if stopped else 4)
            self.assertTrue((suite / 'scripts/configs/fasth3_v1.json').is_file())
            self.assertFalse((suite / 'historical_baseline.json').exists())

    def test_sigterm_updates_status_and_releases_locks(self):
        self.exercise('interrupt')

    def test_success(self):
        self.exercise()

    def test_worker_failure_stops_matrix(self):
        self.exercise('failed')

    def test_load_oom_continues(self):
        self.exercise('memory_insufficient')

    def test_formal_oom_stops(self):
        self.exercise('memory_insufficient', 'repeat')


class WorkerTests(unittest.TestCase):

    def test_native_save_boundary_and_no_dit_hooks_in_e2e(self):
        for modes in [['e2e'], ['dit', 'e2e']]:
            with self.subTest(modes=modes):
                self.exercise_modes(modes)

    def exercise_modes(self, modes) -> None:
        from contextlib import nullcontext
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            folder = Path(temporary)
            trace = []
            profile = read_json(HERE / 'configs/fasth3_v1.json')
            plan = dict(models=dict(test=dict(profile=profile, path='model')),
                        selected_gpu_uuids=['GPU-test'],
                        modes=modes,
                        warmup=1,
                        repeats=1,
                        validate_video=False)
            write_json(folder / 'plan.json', plan)
            torch, dist, timer = MagicMock(), MagicMock(), MagicMock()
            torch.distributed = dist
            torch.no_grad.side_effect = nullcontext
            torch.cuda.OutOfMemoryError = MemoryError
            torch.cuda.synchronize.side_effect = lambda: trace.append('sync')
            torch.cuda.max_memory_allocated.return_value = 1
            torch.cuda.max_memory_reserved.return_value = 2
            torch.cuda.get_device_properties.return_value = SimpleNamespace(name='test',
                                                                            uuid='GPU-test',
                                                                            major=9,
                                                                            minor=0,
                                                                            total_memory=2**30)
            torch.get_num_threads.return_value = 8
            torch.version.cuda = 'test'
            dist.barrier.side_effect = lambda: trace.append('barrier')
            dist.all_gather_object.side_effect = lambda dest, value: dest.__setitem__(0, value)
            timer.elapsed_ms.return_value = [1000] * 4
            timer.attach.side_effect = lambda module: trace.append('attach')
            timer.detach.side_effect = lambda: trace.append('detach')
            output = folder / 'output.mp4'
            output.touch()
            generator = MagicMock()

            def generate(request):
                trace.append('native_generate_save')
                return SimpleNamespace(size=(768, 1344, 124), audio=[1.], video_path=str(output))

            generator.generate.side_effect = generate
            stack.enter_context(
                patch.dict(
                    sys.modules, {
                        'torch': torch,
                        'torch.distributed': dist,
                        'numpy': MagicMock(),
                        'fastvideo': SimpleNamespace(VideoGenerator=SimpleNamespace(from_config=lambda cfg: generator))
                    }))
            stack.enter_context(patch.dict(os.environ, {'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '1'}))
            stack.enter_context(
                patch.object(
                    sys, 'argv',
                    ['worker.py', '--model', 'test', '--plan',
                     str(folder / 'plan.json'), '--output',
                     str(folder)]))
            stack.enter_context(patch.object(worker, 'configure_environment'))
            stack.enter_context(patch.object(worker, 'check_native_dependencies', return_value={}))
            stack.enter_context(patch.object(worker, 'dense_attention_info', return_value=[]))
            stack.enter_context(patch.object(worker, 'generator_config', return_value={}))
            stack.enter_context(patch.object(worker, 'asdict', return_value={}))
            stack.enter_context(
                patch.object(worker, 'check_pipeline', return_value=(MagicMock(), dict(expected_dit_forwards=4))))
            stack.enter_context(patch.object(worker, 'make_request'))
            events = stack.enter_context(patch.object(worker, 'DiTEvents', return_value=timer))
            stack.enter_context(patch.object(worker.importlib.metadata, 'version', return_value='test'))

            def clock():
                trace.append('clock')
                return float(len(trace))

            stack.enter_context(patch.object(worker.time, 'perf_counter', side_effect=clock))
            with redirect_stdout(io.StringIO()):
                worker.main()
            self.assertEqual(events.call_count, int('dit' in modes))
            self.assertEqual(timer.attach.call_count, 2 * int('dit' in modes))
            for i, value in enumerate(trace):
                if value == 'native_generate_save':
                    self.assertEqual(trace[i - 1:i + 4], ['clock', 'native_generate_save', 'clock', 'barrier', 'clock'])
            metrics = read_json(folder / 'metrics.json')
            self.assertEqual(metrics['status'], 'complete')
            for row in metrics['records']:
                self.assertEqual('dit_forward_sum_max_rank_s' in row, row['mode'] == 'dit')
                self.assertEqual(len(row['per_rank']), 1)


if __name__ == '__main__':
    unittest.main()
