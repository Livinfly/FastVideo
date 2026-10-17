"""Inspect saved progress; --watch polls outside benchmark processes (stdlib only)."""
import argparse
import datetime
import os
from pathlib import Path
import re
import time

from common import read_json
from state import status_view


def show(suite) -> bool:
    path = suite / 'status.json'
    if not path.exists():
        print(f'Waiting for suite initialization: {suite}', flush=True)
        job_path = suite.parent / f'{suite.name}.job.json'
        if job_path.exists():
            job = read_json(job_path)
            try:
                os.kill(job['pid'], 0)
            except ProcessLookupError:
                print(f"Launcher exited before initialization. See {job['log']}", flush=True)
                return False
        return True
    state = status_view(suite, read_json(path))
    try:
        os.kill(state['pid'], 0)
        alive = True
    except ProcessLookupError:
        alive = False
    plan = read_json(suite / 'plan.json')
    print(
        f"\n{datetime.datetime.now().astimezone().isoformat(timespec='seconds')} "
        f"status={state['status']} PID={state['pid']} alive={alive} "
        f"groups={len(state['completed'])}/{state['total']}",
        flush=True)
    if state['has_failures']:
        error = state['last_error']
        print(f"FAILED groups={state['failure_count']}: {error['summary']}", flush=True)
        print('Error log:', suite / error['log_file'], flush=True)
        if error['error_files']:
            print('Rank errors:', ', '.join(error['error_files']), flush=True)
    print('model             GPUs  mode  warmup  measured', flush=True)
    for group in plan['groups']:
        metrics = suite / group['directory'] / 'metrics.json'
        records = read_json(metrics)['records'] if metrics.exists() else []
        for mode in plan['modes']:
            subset = [r for r in records if r['mode'] == mode]
            warmup = sum(r['warmup'] for r in subset)
            measured = len(subset) - warmup
            print(
                f"{group['model']:16}  {group['gpus']:4}  {mode:4}  {warmup}/{plan['warmup']}     {measured}/{plan['repeats']}",
                flush=True)
    if state.get('waiting_reason'):
        print('Waiting:', state['waiting_reason'], flush=True)
    if state['current']:
        directory = suite / state['current']['directory']
        progress = directory / 'progress.json'
        print('Current:',
              state['current']['directory'],
              read_json(progress) if progress.exists() else 'loading',
              flush=True)
        log_path = directory / 'inference.log'
        if log_path.exists():
            with log_path.open('rb') as log:
                log.seek(max(0, log_path.stat().st_size - 32768))
                tail = log.read().decode(errors='replace')
            steps = re.findall(r'(\d+)%\|[^\r\n]*?\b(\d+/\d+)\s*\[([^\]\r\n]*)', tail)
            if steps:
                print('Latest tqdm:', steps[-1], '(from log; may include loading bars)', flush=True)
    print(f"Report: {suite / 'report.md'}", flush=True)
    if not alive and state['status'] in {'running', 'waiting'}:
        print('Process is no longer alive; suite is incomplete. Inspect launcher/inference logs.', flush=True)
    return state['status'] in {'running', 'waiting'} and alive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    while True:
        running = show(args.suite.resolve())
        if not args.watch or not running:
            break
        time.sleep(args.interval)
