"""CPU-only helpers, executed outside the timed request."""
from typing import Any
from collections.abc import Iterator
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('FASTVIDEO_BENCH_SOURCE_ROOT', HERE.parent))
SCHEMA = 3
FIELDS = ('response_e2e_s', 'distributed_e2e_s', 'dit_forward_sum_max_rank_s', 'dit_forward_step_max_sum_s')


def read_json(path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path, value) -> None:
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.', suffix='.tmp',
                                     delete=False) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary = Path(stream.name)
    temporary.replace(path)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def schedule(modes, warmup, repeats) -> Iterator[tuple[str, bool, int]]:
    """Each mode gets full warmups, then AB/BA pairs to reduce ordering bias."""
    for phase, count in [('warmup', warmup), ('repeat', repeats)]:
        for index in range(count):
            for mode in (modes if index % 2 == 0 else list(reversed(modes))):
                yield mode, phase == 'warmup', index + 1


def aggregate(per_rank, *, mode, tag, warmup, repeat) -> dict[str, Any]:
    ranks = sorted(per_rank, key=lambda item: item['rank'])
    if not ranks or [r['rank'] for r in ranks] != list(range(len(ranks))):
        raise ValueError('Missing or duplicate ranks')
    record = dict(timing_schema_version=SCHEMA,
                  mode=mode,
                  tag=tag,
                  warmup=warmup,
                  repeat=repeat,
                  per_rank=ranks,
                  response_e2e_s=ranks[0]['response_e2e_s'],
                  distributed_e2e_s=max(r['distributed_e2e_s'] for r in ranks),
                  peak_allocated_gib=max(r['peak_allocated_gib'] for r in ranks),
                  peak_reserved_gib=max(r['peak_reserved_gib'] for r in ranks))
    for field in FIELDS[:2]:
        if not math.isfinite(record[field]) or record[field] <= 0:
            raise ValueError(f'Invalid {field}')
    if mode == 'dit':
        matrix = [r['dit_step_ms'] for r in ranks]
        if not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
            raise ValueError('Inconsistent DiT forward counts')
        if any(not math.isfinite(t) or t < 0 for row in matrix for t in row):
            raise ValueError('Invalid CUDA event time')
        record.update(dit_forward_sum_max_rank_s=max(map(sum, matrix)) / 1000,
                      dit_forward_step_max_sum_s=sum(map(max, zip(*matrix, strict=False))) / 1000)
    elif any(r['dit_step_ms'] is not None for r in ranks):
        raise ValueError('e2e mode must not contain DiT measurements')
    from sparse_attention import aggregate_sparse
    aggregate_sparse(record, ranks)
    return record


def measurements_complete(metrics, plan, mode=None) -> bool:
    """Require every planned warmup/repeat exactly once; shared by runner and report."""
    observed = sorted(
        (r['mode'], r['warmup'], r['repeat']) for r in metrics.get('records', []) if mode is None or r['mode'] == mode)
    expected = sorted(schedule([mode] if mode else plan['modes'], plan['warmup'], plan['repeats']))
    return metrics.get('status') == 'complete' and observed == expected


def summarize(records) -> dict[str, Any]:
    result = {}
    from sparse_attention import SPARSE_FIELDS, validate_record_scopes
    validate_record_scopes(records)
    for field in FIELDS + SPARSE_FIELDS:
        values = [r[field] for r in records if field in r]
        if values:
            result[field] = dict(values=values,
                                 mean=statistics.mean(values),
                                 median=statistics.median(values),
                                 stdev=statistics.stdev(values) if len(values) > 1 else None,
                                 min=min(values),
                                 max=max(values))
    return result


def review_measurements(rows) -> list[str]:
    """Flag measured-request variation without changing the requested experiment plan."""
    findings = []
    for row in rows:
        for field in FIELDS:
            stat = row.get(field)
            if not stat:
                continue
            values = stat['values']
            noisy = stat['stdev'] is not None and stat['stdev'] / stat['mean'] > .02
            trend = (len(values) >= 3 and all(a > b for a, b in zip(values, values[1:], strict=False))
                     and (values[0] - values[-1]) / values[0] > .02)
            if noisy or trend:
                findings.append(f"{row['model']}/g{row['gpus']}/{row['mode']} {field}: " + ('波动较大；' if noisy else '') +
                                ('建议增加预热；' if trend else ''))
    return findings
