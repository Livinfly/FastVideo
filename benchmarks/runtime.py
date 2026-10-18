"""FastH3 adapter: native typed API and small checks outside timed requests."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


def json_config(value) -> Any:
    """Native config contains mapping callables; record stable names, not memory addresses."""

    def fallback(item):
        if callable(item):
            return item.__module__ + '.' + item.__qualname__
        return str(item)

    return json.loads(json.dumps(value, default=fallback))


def configure_environment(profile, output, rank):
    # Native subprocess tools must resolve from the selected virtualenv too.
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    # Avoid inherited experiment toggles silently changing this named workload.
    for key in list(os.environ):
        if key.startswith('FASTVIDEO_') and key != 'FASTVIDEO_BENCH_SOURCE_ROOT':
            del os.environ[key]
    os.environ.update(profile['environment'])
    os.environ['FASTVIDEO_STAGE_LOGGING'] = '0'
    for key, leaf in [('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
        os.environ[key] = str(output / 'cache' / f'rank{rank}' / leaf)


def check_native_dependencies():
    import flash_attn
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise ValueError('Native ffmpeg saver unavailable; run benchmarks/setup.sh')
    return dict(flash_attn_version=flash_attn.__version__, ffmpeg_executable=ffmpeg)


def dense_attention_info(transformer):
    routes = [
        dict(module=name, implementation=type(module.attn_impl).__name__)
        for name, module in transformer.named_modules() if hasattr(module, 'attn_impl')
    ]
    required = [
        row for row in routes
        if row['module'].startswith('token_refiner.') or '.attn2.' in row['module'] or '.cross_attn.' in row['module']
    ]
    if not required or any(row['implementation'] != 'FlashAttentionImpl' for row in required):
        raise ValueError(f'Native preferred DiT dense attention did not select FlashAttention: {required}')
    return routes


def generator_config(profile, model_path, world):
    from fastvideo.api import (CompileConfig, ComponentConfig, EngineConfig, GeneratorConfig, OffloadConfig,
                               ParallelismConfig, PipelineSelection)
    engine = dict(profile['engine'])
    for key, cls in [('offload', OffloadConfig), ('compile', CompileConfig)]:
        if key in engine:
            engine[key] = cls(**engine[key])
    pipeline = dict(profile['pipeline'])
    if 'dmd_denoising_steps' in profile['contract']:
        pipeline['experimental'] = dict(pipeline['experimental'],
                                        dmd_denoising_steps=profile['contract']['dmd_denoising_steps'])
    if profile['adapter'] == 'longcat_refine':
        pipeline['components'] = ComponentConfig(lora_path=str(
            Path(model_path) / 'lora/refinement/refinement_lora.safetensors'),
                                                 lora_nickname='refinement')
    return GeneratorConfig(model_path=str(model_path),
                           engine=EngineConfig(num_gpus=world,
                                               execution_backend='external_launcher',
                                               parallelism=ParallelismConfig(sp_size=world, tp_size=1),
                                               **engine),
                           pipeline=PipelineSelection(**pipeline))


def check_pipeline(pipe, profile):
    if profile['adapter'] == 'longcat_refine':
        from longcat import check_pipeline as check_longcat
        return check_longcat(pipe, profile)
    if profile['adapter'] == 'fastwan21_14b':
        from fastwan import check_pipeline as check_wan
        return check_wan(pipe, profile)
    from fastvideo.pipelines.lazy_module import is_lazy_module
    transformer = pipe.get_module('transformer')
    if is_lazy_module(transformer):
        raise ValueError('DiT hooks require a stable transformer instance; set engine.offload.lazy_module_load=false')
    ladder = list(pipe.fastvideo_args.pipeline_config.dmd_denoising_steps)
    if ladder != profile['contract']['dmd_denoising_steps']:
        raise ValueError(f'Unexpected effective DMD ladder: {ladder}')
    for name, key in [('scheduler', 'video_shift'), ('audio_scheduler', 'audio_shift')]:
        if float(pipe.get_module(name).shift) != profile['contract'][key]:
            raise ValueError(f'Unexpected effective {name} shift')
    routes = [type(block.attn.distributed_attention.attn_impl).__name__ for block in transformer.transformer_blocks]
    if not routes or set(routes) != {'MiniMaxH3VSAImpl'}:
        raise ValueError(f'Expected VSA-H3 on every main transformer block, got {routes}')
    for key in ['VSA_sparsity', 'VSA_tile_size']:
        if getattr(pipe.fastvideo_args, key) != profile['pipeline']['experimental'][key]:
            raise ValueError(f'Unexpected effective {key}')
    return transformer, dict(pipeline_class=type(pipe).__name__,
                             transformer_class=type(transformer).__name__,
                             attention_implementations=routes,
                             pipeline_config=json_config(asdict(pipe.fastvideo_args.pipeline_config)),
                             expected_dit_forwards=len(ladder),
                             resolved_execution={
                                 key: getattr(pipe.fastvideo_args, key)
                                 for key in [
                                     'dit_cpu_offload', 'dit_layerwise_offload', 'text_encoder_cpu_offload',
                                     'vae_cpu_offload', 'pin_cpu_memory', 'lazy_module_load', 'use_fsdp_inference',
                                     'sp_size', 'tp_size', 'enable_torch_compile', 'inference_torch_compile',
                                     'enable_torch_compile_vae', 'vae_parallel_decode', 'video_decode_backend'
                                 ]
                             },
                             forward_layout=[dict(forward_index=i, dmd_rung=rung) for i, rung in enumerate(ladder)])


def make_request(profile, path, stage1_frames=None):
    if profile['adapter'] == 'longcat_refine':
        from longcat import make_request as longcat_request
        return longcat_request(profile, path, stage1_frames)
    from fastvideo.api import GenerationRequest, OutputConfig, SamplingConfig
    return GenerationRequest(prompt=profile['prompt'],
                             negative_prompt=profile['negative_prompt'],
                             sampling=SamplingConfig(**profile['sampling']),
                             output=OutputConfig(output_path=str(path), save_video=True, return_frames=False))


def video_metadata(path, require_audio=True):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        video = dict(frames=sum(1 for _ in container.decode(video=0)),
                     fps=float(stream.average_rate),
                     width=stream.width,
                     height=stream.height,
                     codec=stream.codec_context.name)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            if not require_audio:
                return dict(video, audio_samples=0)
            raise ValueError('Saved output has no audio stream')
        samples = sum(frame.samples for frame in container.decode(audio=0))
        if samples <= 0:
            raise ValueError('Saved audio is empty')
        video.update(audio_samples=samples, audio_sample_rate=container.streams.audio[0].rate)
    return video
