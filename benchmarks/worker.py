"""One torchrun rank; E2E wraps native generation/save, events wrap complete DiT."""
import argparse
from dataclasses import asdict
import importlib.metadata
import os
from pathlib import Path
import random
import sys
import time
import traceback

from common import ROOT, SCHEMA, aggregate, read_json, schedule, sha256, write_json
from gpu_wait import check_device_identity
from runtime import (check_native_dependencies, check_pipeline, configure_environment, dense_attention_info,
                     generator_config, make_request, video_metadata)
from timing import DiTEvents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    profile = plan['models'][args.model]['profile']
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    configure_environment(profile, args.output, rank)
    sys.path.insert(0, str(ROOT))
    generator, timer, audit = None, None, None
    phase = 'load'
    records = []

    def dump(name, value) -> None:
        if rank == 0:
            write_json(args.output / name, value)

    try:
        import numpy as np
        import torch
        import torch.distributed as dist
        check_device_identity(torch, int(os.environ['LOCAL_RANK']), plan['selected_gpu_uuids'][rank])
        from fastvideo import VideoGenerator

        dependency_info = check_native_dependencies()
        dump('progress.json', dict(status='loading'))
        start_load = time.perf_counter()
        config = generator_config(profile, plan['models'][args.model]['path'], world)
        generator = VideoGenerator.from_config(config)
        pipe = generator.executor.worker.pipeline
        pipe.post_init()  # Official lazy stages and schedule must exist before attaching hooks.
        transformer, effective = check_pipeline(pipe, profile)
        effective.update(native_dependencies=dependency_info,
                         dense_attention_implementations=dense_attention_info(transformer))
        if profile['adapter'] == 'fastwan21_14b':
            from fastwan import VSAExecutionAudit
            audit = VSAExecutionAudit(transformer, profile)
        stage1_frames = None
        if profile['adapter'] == 'longcat_refine':
            from longcat import LongCatExecutionAudit, load_frames
            stage1_frames = load_frames(plan['models'][args.model]['input_frames'], profile['contract'])
            audit = LongCatExecutionAudit(profile)
            audit.handle = transformer.register_forward_pre_hook(audit.observe, with_kwargs=True)
        timer = DiTEvents(torch, effective['expected_dit_forwards']) if 'dit' in plan['modes'] else None
        torch.cuda.synchronize()
        dist.barrier()
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        devices = [None] * world
        dist.all_gather_object(
            devices,
            dict(rank=rank,
                 name=props.name,
                 uuid=str(getattr(props, 'uuid', 'unknown')),
                 capability=[props.major, props.minor],
                 total_memory_gib=props.total_memory / 2**30))
        effective.update(
            generator=asdict(config),
            profile=profile,
            devices=devices,
            python=sys.version,
            cuda=torch.version.cuda,
            world_size=world,
            timing_schema_version=SCHEMA,
            load_seconds_excluded=time.perf_counter() - start_load,
            cpu_threads=torch.get_num_threads(),
            packages={
                name: importlib.metadata.version(name)
                for name in
                ['torch', 'torchvision', 'torchaudio', 'transformers', 'diffusers', 'fastvideo-kernel', 'triton', 'av']
            },
            environment={
                key: value
                for key, value in os.environ.items() if key.startswith(('FASTVIDEO_', 'NCCL_'))
                or key in {'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS', 'PYTORCH_CUDA_ALLOC_CONF'}
            })
        dump('effective_config.json', effective)
        sampling = profile['sampling']
        for mode, warmup, repeat in schedule(plan['modes'], plan['warmup'], plan['repeats']):
            phase = 'warmup' if warmup else 'repeat'
            tag = f'{mode}_{phase}_{repeat}'
            output = args.output / f'{tag}.mp4'
            request = (make_request(profile, output, stage1_frames)
                       if profile['adapter'] == 'longcat_refine' else make_request(profile, output))
            if audit is not None:
                audit.rows.clear()
            if mode == 'dit':
                timer.attach(transformer)
            random.seed(sampling['seed'])
            np.random.seed(sampling['seed'])
            torch.manual_seed(sampling['seed'])
            torch.cuda.manual_seed_all(sampling['seed'])
            dump('progress.json', dict(status='running', tag=tag))
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            response_s = None
            with torch.no_grad():
                # E2E START: text/noise, DiT, schedulers, VAEs, video/audio encode + save.
                started = time.perf_counter()
                result = generator.generate(request)
                if rank == 0:
                    response_s = time.perf_counter() - started  # E2E END: native MP4 save returned.
                # No added pipeline-end/per-forward CUDA sync in either mode.
                dist.barrier()
                distributed_s = time.perf_counter() - started

            # Validation, event queries, statistics and writes are OUTSIDE E2E.
            if mode == 'dit':
                timer.detach()
            torch.cuda.synchronize()
            durations = timer.elapsed_ms() if mode == 'dit' else None
            local = dict(rank=rank,
                         response_e2e_s=response_s,
                         distributed_e2e_s=distributed_s,
                         dit_step_ms=durations,
                         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            if audit is not None:
                local['bsa_execution' if profile['adapter'] == 'longcat_refine' else 'vsa_execution'] = audit.collect()
            per_rank = [None] * world
            dist.all_gather_object(per_rank, local)
            record = aggregate(per_rank, mode=mode, tag=tag, warmup=warmup, repeat=repeat)
            if rank == 0:
                if tuple(result.size) != (sampling['height'], sampling['width'], sampling['num_frames']):
                    raise ValueError(f'Unexpected output size: {result.size}')
                if profile['adapter'] == 'minimax_h3' and (
                        result.audio is None or not torch.isfinite(torch.as_tensor(result.audio)).all().item()):
                    raise ValueError('Missing or non-finite generated audio')
                if not result.video_path or not Path(result.video_path).is_file():
                    raise ValueError('Missing native saved video')
                record.update(output_shape=[1, sampling['num_frames'], sampling['height'], sampling['width'], 3],
                              video_file=str(Path(result.video_path).relative_to(args.output)))
                if plan['validate_video']:
                    media = (video_metadata(result.video_path, require_audio=False)
                             if profile['adapter'] != 'minimax_h3' else video_metadata(result.video_path))
                    if any(media[k] != sampling[s]
                           for k, s in [('frames', 'num_frames'), ('width', 'width'), ('height',
                                                                                       'height'), ('fps', 'fps')]):
                        raise ValueError(f'Unexpected saved media: {media}')
                    record.update(video_metadata=media, video_sha256=sha256(result.video_path))
                records.append(record)
                dump('metrics.json', dict(status='running', timing_schema_version=SCHEMA, records=records))
                print(f'{tag}: E2E={response_s:.3f}s, distributed={distributed_s:.3f}s', flush=True)
            del result
        dump('metrics.json', dict(status='complete', timing_schema_version=SCHEMA, records=records))
        dump('progress.json', dict(status='complete'))
    except BaseException as error:
        # Only explicit loading/warmup CUDA OOM may let the coordinator continue.
        torch_module = sys.modules.get('torch')
        oom_type = getattr(getattr(torch_module, 'cuda', None), 'OutOfMemoryError', ())
        oom = isinstance(error, oom_type) and phase in {'load', 'warmup'}
        write_json(
            args.output / f'error_rank{rank}.json',
            dict(phase=phase, status='memory_insufficient' if oom else 'failed', traceback=traceback.format_exc()))
        raise
    finally:
        if audit is not None:
            audit.handle.remove()
        if timer is not None:
            timer.detach()
        if generator is not None:
            generator.shutdown()


if __name__ == '__main__':
    main()
