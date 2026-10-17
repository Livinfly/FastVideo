"""Human-readable failure state; only the coordinator writes status.json."""
from typing import Any
import datetime
from pathlib import Path
import re

from common import read_json, write_json


def failure_details(suite, group, fallback='Worker failed; inspect inference.log') -> dict[str, Any]:
    directory = Path(suite) / group['directory']
    paths = sorted(directory.glob('error_rank*.json'))
    details = dict(directory=group['directory'],
                   gpus=group.get('gpus'),
                   summary=fallback,
                   phase='unknown',
                   log_file=str(Path(group['directory']) / 'inference.log'),
                   error_files=[str(p.relative_to(suite)) for p in paths])
    for path in paths:
        try:
            error = read_json(path)
        except (OSError, ValueError):
            continue
        lines = error.get('traceback', '').strip().splitlines()
        if lines:
            # CUDA appends debugging advice after the exception; show the actual error.
            exceptions = [line for line in lines if re.match(r'^[A-Za-z_][\w.]*: ', line)]
            details.update(summary=(exceptions[-1] if exceptions else lines[-1])[:2000],
                           phase=error.get('phase', 'unknown'))
            break
    return details


def status_view(suite, state) -> dict[str, Any]:
    """Also enrich old suites for monitoring, without rewriting a live runner's file."""
    state = dict(state)
    failed = []
    for group in state.get('completed', []):
        if group['status'] != 'complete':
            group = dict(group)
            group.setdefault('error',
                             failure_details(suite, group, group.get('reason', 'Worker failed; inspect inference.log')))
            failed.append(group)
    state['failure_count'] = len(failed)
    state['last_error'] = state.get('launcher_error') or (failed[-1]['error'] if failed else None)
    current = state.get('current')
    if current and list((Path(suite) / current['directory']).glob('error_rank*.json')):
        # A rank can already have failed while torchrun is still cleaning up peers.
        state['last_error'] = failure_details(suite, current)
        if not any(g['directory'] == current['directory'] for g in failed):
            state['failure_count'] += 1
    if state['last_error'] is None and state.get('status') in {'complete_with_failures', 'interrupted_or_failed'}:
        state['last_error'] = dict(summary='Run ended with status ' + state['status'],
                                   phase='coordinator',
                                   log_file=f'../{Path(suite).name}.launcher.log',
                                   error_files=[])
    state['has_failures'] = bool(failed or state.get('launcher_error') or state['last_error'])
    return state


def write_status(suite, state, error=None) -> None:
    if error is not None:
        state['launcher_error'] = dict(summary=f'{type(error).__name__}: {error}',
                                       phase='coordinator',
                                       log_file=f'../{Path(suite).name}.launcher.log',
                                       error_files=[])
    # Persist group diagnostics when first observed; later successful groups do not hide them.
    for group in state.get('completed', []):
        if group['status'] != 'complete' and 'error' not in group:
            group['error'] = failure_details(suite, group, group.get('reason', 'Worker failed; inspect inference.log'))
    state.update(status_view(suite, state))
    state['updated_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(Path(suite) / 'status.json', state)
