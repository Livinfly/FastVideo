"""Portable named-model matrix with idle GPU waiting, background logs and state."""
from typing import Any
import argparse
from contextlib import ExitStack
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

from common import HERE, ROOT, SCHEMA, measurements_complete, read_json, write_json
from config import model_manifest, validate_profile
from gpu_wait import inventory, numeric_visible_devices, select_devices, wait_for_idle, wait_for_suite
from lifecycle import stop_worker
from state import write_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',
                        type=Path,
                        nargs='+',
                        default=[HERE / 'configs/fasth3_v1.json', HERE / 'configs/fasth3_v2.json'])
    parser.add_argument('--model-path',
                        action='append',
                        default=[],
                        metavar='NAME=PATH',
                        help='Repeat for every selected profile; complete local model snapshot')
    prompts = parser.add_mutually_exclusive_group()
    parser.add_argument('--input-frames',
                        action='append',
                        default=[],
                        metavar='NAME=PATH',
                        help='Original NPY/NPZ frames for a LongCat refinement profile')
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompt-file', type=Path)
    parser.add_argument('--gpus', type=int, nargs='+', default=[1])
    parser.add_argument('--modes', choices=['dit', 'e2e'], nargs='+', default=['dit'])
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--validate-video', action='store_true')
    parser.add_argument('--baseline', type=Path, help='Opt in: compare another suite summary.json')
    parser.add_argument('--wait-for', type=Path, help='Also wait for another suite status.json')
    args = parser.parse_args()
    if min(args.gpus) < 1 or args.warmup < 1 or args.repeats < 1:
        parser.error('Require positive GPU counts, >=1 warmup and >=1 measured request')
    if len(set(args.gpus)) != len(args.gpus) or len(set(args.modes)) != len(args.modes):
        parser.error('Duplicate GPU counts/modes are not allowed')
    paths = {}
    for entry in args.model_path:
        name, separator, path = entry.partition('=')
        if not separator or not path or name in paths:
            parser.error('--model-path must be unique NAME=PATH entries')
        paths[name] = str(Path(path).expanduser().resolve())
    inputs = {}
    for entry in args.input_frames:
        name, separator, path = entry.partition('=')
        if not separator or not path or name in inputs:
            parser.error('--input-frames must be unique NAME=PATH entries')
        inputs[name] = str(Path(path).expanduser().resolve())
    models = {}
    for path in args.config:
        profile = read_json(path)
        validate_profile(profile)
        name = profile['name']
        if name in models:
            parser.error(f'Duplicate profile name: {name}')
        if args.prompt is not None:
            profile['prompt'] = args.prompt
        if args.prompt_file:
            profile['prompt'] = args.prompt_file.read_text().strip()
        models[name] = dict(profile=profile, path=paths.get(name))
        if profile['adapter'] == 'longcat_refine':
            if name not in inputs:
                parser.error(f'Supply --input-frames {name}=PATH for LongCat refinement')
            models[name]['input_frames'] = inputs[name]
    if set(inputs) - {name for name, item in models.items() if item['profile']['adapter'] == 'longcat_refine'}:
        parser.error('--input-frames only accepts selected LongCat refinement profiles')
    if set(paths) != set(models):
        parser.error('Supply --model-path NAME=PATH for exactly these profiles: ' + ', '.join(models))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    suite = args.output.resolve() if args.output else ROOT / 'outputs/benchmarks' / stamp
    groups = [dict(model=name, gpus=n, directory=f'{name}/g{n}') for name in models for n in args.gpus]
    commands = [[
        sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f"--nproc_per_node={g['gpus']}",
        str(suite / 'scripts/worker.py'), '--plan',
        str(suite / 'plan.json'), '--model', g['model'], '--output',
        str(suite / g['directory'])
    ] for g in groups]
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return
    branch = subprocess.check_output(['git', '-C', str(ROOT), 'branch', '--show-current'], text=True).strip()
    if branch != 'exp-fixed-version':
        parser.error(f'Expected exp-fixed-version branch; found {branch!r}')
    manifests = {
        name: model_manifest(Path(item['path']), item['profile'], args.gpus, item.get('input_frames'))
        for name, item in models.items()
    }
    devices = select_devices(inventory(), os.environ.get('CUDA_VISIBLE_DEVICES'))
    if max(args.gpus) > len(devices):
        parser.error(f'Requested {max(args.gpus)} GPUs, only {len(devices)} selected')
    if args.baseline:
        read_json(args.baseline)
    if suite.exists():
        parser.error(f'Refusing to overwrite {suite}')
    if args.background:
        suite.parent.mkdir(parents=True, exist_ok=True)
        log_path = suite.parent / f'{suite.name}.launcher.log'
        forwarded = [x for x in sys.argv[1:] if x != '--background']
        if args.output is None:
            forwarded += ['--output', str(suite)]
        with log_path.open('x') as log:
            child = subprocess.Popen([sys.executable, str(HERE / 'run.py'), *forwarded],
                                     stdout=log,
                                     stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        write_json(suite.parent / f'{suite.name}.job.json', dict(pid=child.pid, suite=str(suite), log=str(log_path)))
        print(f'PID: {child.pid}\nSuite: {suite}\nLog: {log_path}')
        return

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    suite.mkdir(parents=True)
    shutil.copytree(HERE, suite / 'scripts', ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(
        ['git', '-C', str(ROOT), 'diff', 'HEAD', '--', 'fastvideo', 'fastvideo-kernel'], text=True)
    (suite / 'inference_source.diff').write_text(diff)
    write_json(suite / 'model_manifest.json', manifests)
    baseline = suite / 'historical_baseline.json' if args.baseline else None
    if baseline:
        shutil.copy2(args.baseline, baseline)
    plan = dict(timing_schema_version=SCHEMA,
                created_utc=stamp,
                source_commit=commit,
                source_branch=branch,
                inference_source_modified=bool(diff),
                models=models,
                model_manifest_sha256={
                    name: hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
                    for name, manifest in manifests.items()
                },
                groups=groups,
                gpu_counts=args.gpus,
                selected_gpu_uuids=devices,
                modes=args.modes,
                warmup=args.warmup,
                repeats=args.repeats,
                validate_video=args.validate_video,
                wait_for=str(args.wait_for.resolve()) if args.wait_for else None,
                pid=os.getpid())
    write_json(suite / 'plan.json', plan)
    status: dict[str, Any] = dict(status='waiting', pid=os.getpid(), total=len(groups), completed=[], current=None)
    from report import generate_report

    def waiting(reasons) -> None:
        status.update(status='waiting', waiting_reason=reasons)
        write_status(suite, status)
        print('Waiting: ' + str(reasons), flush=True)

    write_status(suite, status)
    try:
        generate_report(suite, baseline=baseline)
        wait_for_suite(args.wait_for, waiting)
        for group, command in zip(groups, commands, strict=False):
            directory = suite / group['directory']
            directory.mkdir(parents=True)
            status['current'] = group
            with ExitStack() as locks:
                for handle in wait_for_idle(devices[:group['gpus']], waiting):
                    locks.enter_context(handle)
                status.update(status='running', waiting_reason=[])
                write_status(suite, status)
                env = {
                    **os.environ, 'CUDA_VISIBLE_DEVICES': numeric_visible_devices(inventory(), devices[:group['gpus']]),
                    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                    'FASTVIDEO_BENCH_SOURCE_ROOT': str(ROOT),
                    'OMP_NUM_THREADS': '8',
                    'HF_HUB_OFFLINE': '1',
                    'TRANSFORMERS_OFFLINE': '1',
                    'TOKENIZERS_PARALLELISM': 'false',
                    'PYTHONUNBUFFERED': '1'
                }
                (directory / 'command.sh').write_text(shlex.join(command) + '\n')
                with (directory / 'inference.log').open('w') as log:
                    child = subprocess.Popen(command,
                                             cwd=ROOT,
                                             env=env,
                                             stdout=log,
                                             stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    try:
                        code = child.wait()
                    finally:
                        stop_worker(child)
            result = dict(group, exit_code=code, status='complete' if code == 0 else 'failed')
            metrics = directory / 'metrics.json'
            if code == 0 and (not metrics.exists() or not measurements_complete(read_json(metrics), plan)):
                result['status'] = 'failed'
                result['reason'] = 'Worker exited without all planned measurements'
            if code != 0:
                errors = [read_json(p) for p in directory.glob('error_rank*.json')]
                if errors and all(e['status'] == 'memory_insufficient' and e['phase'] in {'load', 'warmup'}
                                  for e in errors):
                    result['status'] = 'memory_insufficient'
            status['completed'].append(result)
            write_status(suite, status)
            generate_report(suite, baseline=baseline)
            if result['status'] == 'failed':
                raise RuntimeError(f'Group failed; inspect {directory}/inference.log')
        status.update(status='complete' if all(r['status'] == 'complete'
                                               for r in status['completed']) else 'complete_with_failures',
                      current=None)
    except BaseException as error:
        status['status'] = 'interrupted_or_failed'
        write_status(suite, status, error=error)
        raise
    finally:
        write_status(suite, status)
        generate_report(suite, baseline=baseline)
    print(f'Report: {suite / "report.md"}', flush=True)
    if status['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
