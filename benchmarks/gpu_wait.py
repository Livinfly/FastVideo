"""Host-only idle checks and cooperative local locks; never poll during inference."""
from typing import Any, TextIO
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import time

from common import read_json

LOCK_ROOT = Path('/tmp') / f'video-benchmark-gpus-{os.getuid()}'


def query(kind, fields) -> list[list[str]]:
    output = subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
                                     text=True)
    return [[v.strip() for v in row] for row in csv.reader(output.splitlines()) if row]


def inventory() -> list[dict[str, Any]]:
    return [
        dict(index=index, uuid=uuid, memory_mib=int(memory), utilization=int(util))
        for index, uuid, memory, util in query('gpu', 'index,uuid,memory.used,utilization.gpu')
    ]


def select_devices(devices, visible) -> list[str]:
    """Resolve nvidia-smi indices / UUID prefixes to stable lock identities."""
    if visible is None:
        return [d['uuid'] for d in devices]
    selected = []
    for token in visible.split(','):
        token = token.strip()
        matches = [
            d['uuid'] for d in devices
            if d['index'] == token or (token.startswith('GPU-') and d['uuid'].startswith(token))
        ]
        if len(matches) != 1 or matches[0] in selected:
            raise ValueError(f'Invalid, ambiguous or repeated GPU selector: {token!r}')
        selected += matches
    return selected


def numeric_visible_devices(devices, selected) -> str:
    """Native FastVideo requires NVML indices; workers verify CUDA UUIDs before loading."""
    by_uuid = {d['uuid']: d['index'] for d in devices}
    return ','.join(by_uuid[uuid] for uuid in selected)


def check_device_identity(torch, local_rank, expected_uuid) -> None:
    """Fail before model load if CUDA and NVML enumerate the selected GPU differently."""
    torch.cuda.set_device(local_rank)
    actual = str(torch.cuda.get_device_properties(local_rank).uuid)
    if actual.removeprefix('GPU-').lower() != expected_uuid.removeprefix('GPU-').lower():
        raise RuntimeError(f'CUDA/NVML GPU order mismatch for local rank {local_rank}: '
                           f'expected {expected_uuid}, got {actual}. Refusing to use an unlocked GPU.')


def busy_reasons(devices, processes, selected) -> list[dict[str, Any]]:
    by_uuid = {d['uuid']: d for d in devices}
    reasons = []
    for uuid in selected:
        if uuid not in by_uuid:
            raise RuntimeError(f'GPU disappeared: {uuid}')
        device = by_uuid[uuid]
        pids = [pid for gpu, pid in processes if gpu == uuid]
        if pids or device['memory_mib'] > 256 or device['utilization'] > 0:
            reasons.append(
                dict(uuid=uuid, pids=pids, memory_mib=device['memory_mib'], utilization=device['utilization']))
    return reasons


def wait_for_idle(selected, on_wait, interval=30) -> list[TextIO]:
    """Return held lock files after two idle observations; caller holds through group exit."""
    # One namespace across framework checkouts on this host, for the same user.
    lock_dir = LOCK_ROOT
    lock_dir.mkdir(exist_ok=True, mode=0o700)
    while True:
        handles = []
        ready = False
        try:
            locked = []
            for uuid in sorted(selected):
                handle = (lock_dir / (uuid + '.lock')).open('a')
                handles.append(handle)
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    locked.append(dict(uuid=uuid, reason='another benchmark holds its lock'))
            if locked:
                on_wait(locked)
            else:
                # Two samples avoid launching in the short gap between other jobs.
                for sample in range(2):
                    reasons = busy_reasons(inventory(), query('compute-apps', 'gpu_uuid,pid'), selected)
                    if reasons:
                        on_wait(reasons)
                        break
                    if sample == 0:
                        on_wait([dict(reason='confirming GPUs remain idle')])
                        time.sleep(interval)
                else:
                    ready = True
                    return handles
        finally:
            if not ready:
                for handle in handles:
                    handle.close()
        time.sleep(interval)


def wait_for_suite(path, on_wait, interval=30) -> None:
    if path is None:
        return
    while True:
        previous = read_json(path)
        if previous['status'] in {'complete', 'complete_with_failures'}:
            return
        if previous['status'] not in {'running', 'waiting'}:
            raise RuntimeError('Preceding suite failed: ' + str(previous))
        os.kill(previous['pid'], 0)
        on_wait([dict(suite=str(path), reason='preceding suite still running')])
        time.sleep(interval)
