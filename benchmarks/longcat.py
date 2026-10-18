"""LongCat refinement adapter; native BSA, scheduler, LoRA and in-memory input API."""
import copy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import zipfile
from typing import Any

from common import read_json, sha256

MODEL_ID = 'FastVideo/LongCat-Video-T2V-Diffusers'
BSA = dict(sparsity=0.9375, chunk_3d_shape_q=[4, 4, 4], chunk_3d_shape_k=[4, 4, 4])


def validate_profile(profile):
    if profile['model_id'] != MODEL_ID:
        raise ValueError('Expected the converted LongCat checkpoint')
    sampling = profile['sampling']
    for key in ['height', 'width', 'num_frames', 'fps', 'num_inference_steps', 'max_sequence_length']:
        if type(sampling[key]) is not int or sampling[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if sampling['guidance_scale'] != 1 or sampling['batch_cfg'] is not False:
        raise ValueError('Refinement uses one conditional branch, CFG1')
    if type(sampling['seed']) is not int or not 0 <= sampling['seed'] < 2**32:
        raise ValueError('Invalid seed')
    contract = profile['contract']
    if (contract['t_thresh'] != 0.5 or contract['spatial_refine_only'] is not False or contract['num_cond_frames'] != 0
            or contract['scheduler_shift'] != 12.0 or sampling['num_inference_steps'] != 50
            or contract['expected_dit_forwards'] != 5):
        raise ValueError('Require the reviewed 50-node / threshold0.5 refinement schedule')
    if sampling['num_frames'] != 2 * contract['input_shape'][0]:
        raise ValueError('Expected temporal doubling of the original input frames')
    expected = dict(workload_type='t2v',
                    experimental=dict(enable_bsa=True,
                                      bsa_sparsity=0.9375,
                                      bsa_chunk_q=[4, 4, 4],
                                      bsa_chunk_k=[4, 4, 4],
                                      refine_target_size=[sampling['height'], sampling['width']]))
    if profile['pipeline'] != expected:
        raise ValueError('Require native LongCat BSA with checkpoint sparsity and tile444')
    if {'num_gpus', 'execution_backend', 'parallelism'} & profile['engine'].keys():
        raise ValueError('GPU topology is controlled by --gpus')
    if profile['environment']:
        raise ValueError('LongCat uses native environment defaults, with stage logging disabled by the harness')


def input_sha256(path: str | Path) -> str:
    if Path(path).suffix != '.npz':
        return sha256(path)
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['frames.npy']:
            raise ValueError('NPZ must contain exactly frames.npy')
        with archive.open('frames.npy') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def load_frames(path: str | Path, contract: dict[str, Any]) -> Any:
    import numpy as np
    if input_sha256(path) != contract['input_frames_sha256']:
        raise ValueError('Original input NPY checksum differs from the selected workload')
    if Path(path).suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            frames = archive['frames']
    else:
        frames = np.load(path, allow_pickle=False)
    if frames.dtype != np.uint8 or list(frames.shape) != contract['input_shape']:
        raise ValueError('Expected original uint8 [T,H,W,3] frames')
    return frames


def model_manifest(path, profile, gpu_counts, input_frames):
    path = Path(path)
    for component in [
            'transformer/config.json', 'scheduler/scheduler_config.json', 'text_encoder/config.json', 'vae/config.json',
            'tokenizer/tokenizer_config.json', 'model_index.json', 'lora/refinement/refinement_lora.safetensors'
    ]:
        if not (path / component).is_file():
            raise ValueError(f'Missing LongCat model component: {component}')
    if read_json(path / 'model_index.json')['_class_name'] != 'LongCatPipeline':
        raise ValueError('Expected native LongCatPipeline')
    config = read_json(path / 'transformer/config.json')
    if (config['depth'], config['num_attention_heads']) != (48, 32):
        raise ValueError('Unexpected LongCat architecture')
    if config['bsa_params'] != BSA or read_json(path / 'scheduler/scheduler_config.json')['shift'] != 12:
        raise ValueError('Checkpoint BSA/scheduler contract differs')
    if any(n not in (1, 2, 4, 8) for n in gpu_counts):
        raise ValueError('Reviewed LongCat SP counts are 1/2/4/8')
    if input_frames is None:
        raise ValueError('Pass original frames with --input-frames NAME=PATH')
    frames = load_frames(input_frames, profile['contract'])
    configs, weights = {}, {}
    for item in path.rglob('*'):
        if not item.is_file() or '.cache' in item.parts:
            continue
        name = str(item.relative_to(path))
        if item.suffix == '.json' and not item.name.endswith('.index.json'):
            configs[name] = dict(sha256=sha256(item), config=read_json(item))
        elif item.suffix in {'.safetensors', '.bin', '.pt'}:
            weights[name] = item.stat().st_size
    return dict(configs=configs,
                weight_file_sizes=weights,
                input_frames=dict(path=str(input_frames),
                                  shape=list(frames.shape),
                                  npy_sha256=profile['contract']['input_frames_sha256']),
                note='Config hashes and weight sizes only; original input is hashed separately.')


def refinement_timesteps(scheduler: Any, profile: dict[str, Any]) -> list[float]:
    import torch
    scheduler = copy.deepcopy(scheduler)
    steps = profile['sampling']['num_inference_steps']
    scheduler.set_timesteps(steps, sigmas=torch.linspace(1.0, 0.001, steps), device='cpu')
    threshold = torch.tensor(profile['contract']['t_thresh'] * 1000, dtype=scheduler.timesteps.dtype)
    return torch.cat([threshold.unsqueeze(0), scheduler.timesteps[scheduler.timesteps < threshold]]).tolist()


def check_pipeline(pipe, profile):
    from fastvideo.pipelines.lazy_module import is_lazy_module
    from fastvideo.pipelines.stages.longcat_refine_init import refinement_canvas
    from runtime import json_config
    transformer = pipe.get_module('transformer')
    if is_lazy_module(transformer) or type(pipe).__name__ != 'LongCatPipeline':
        raise ValueError('Require a stable native LongCat transformer')
    config = pipe.fastvideo_args.pipeline_config
    if pipe.get_module('scheduler').config.shift != profile['contract']['scheduler_shift']:
        raise ValueError('Unexpected native LongCat scheduler shift')
    if not config.enable_bsa or config.use_distill or len(transformer.blocks) != 48:
        raise ValueError('Expected BSA refinement without distilled stage1 inference')
    routes = []
    for block in transformer.blocks:
        if not block.self_attn.enable_bsa or block.self_attn.bsa_params != BSA:
            raise ValueError('A LongCat layer did not select the expected BSA route')
        routes.append('flash_attn_bsa_3d')
    if pipe.cur_adapter_name != 'refinement' or pipe.lora_strength != 1:
        raise ValueError('Expected the native refinement LoRA merged at strength1')
    ladder = refinement_timesteps(pipe.get_module('scheduler'), profile)
    if len(ladder) != profile['contract']['expected_dit_forwards']:
        raise ValueError(f'Unexpected refinement forward count: {ladder}')
    shape = profile['contract']['input_shape']
    bucket = refinement_canvas(shape[1], shape[2], config.refine_target_size)
    if list(bucket) != [profile['sampling']['height'], profile['sampling']['width']]:
        raise ValueError(f'Native refinement bucket differs from requested output: {bucket}')
    return transformer, dict(pipeline_class=type(pipe).__name__,
                             transformer_class=type(transformer).__name__,
                             attention_implementations=routes,
                             pipeline_config=json_config(asdict(config)),
                             expected_dit_forwards=len(ladder),
                             refinement_timesteps=ladder,
                             bsa_params=BSA,
                             resolved_execution={
                                 key: getattr(pipe.fastvideo_args, key)
                                 for key in [
                                     'dit_cpu_offload', 'dit_layerwise_offload', 'text_encoder_cpu_offload',
                                     'vae_cpu_offload', 'pin_cpu_memory', 'lazy_module_load', 'use_fsdp_inference',
                                     'sp_size', 'tp_size', 'enable_torch_compile', 'inference_torch_compile',
                                     'enable_torch_compile_vae', 'vae_parallel_decode', 'video_decode_backend'
                                 ]
                             })


def make_request(profile, path, stage1_frames):
    from fastvideo.api import GenerationRequest, InputConfig, OutputConfig, SamplingConfig
    if stage1_frames is None:
        raise ValueError('Original stage1 frames must be loaded before timing')
    return GenerationRequest(
        prompt=profile['prompt'],
        negative_prompt=profile['negative_prompt'],
        inputs=InputConfig(stage1_video=stage1_frames),
        sampling=SamplingConfig(**profile['sampling']),
        extensions={key: profile['contract'][key]
                    for key in ['t_thresh', 'spatial_refine_only', 'num_cond_frames']},
        output=OutputConfig(output_path=str(path), save_video=True, return_frames=False))


class LongCatExecutionAudit:
    """Record executed shape/step indices without retaining tensors or syncing CUDA."""

    def __init__(self, profile):
        sampling = profile['sampling']
        self.grid = [math.ceil(sampling['num_frames'] / 16) * 4, sampling['height'] // 16, sampling['width'] // 16]
        self.forwards = profile['contract']['expected_dit_forwards']
        self.rows = []
        self.handle = None

    def observe(self, module, inputs, kwargs):
        from fastvideo.forward_context import get_forward_context
        hidden = kwargs['hidden_states']
        grid = [hidden.shape[2], hidden.shape[3] // 2, hidden.shape[4] // 2]
        if grid != self.grid or hidden.shape[0] != 1:
            raise ValueError(f'Unexpected executed LongCat input: {list(hidden.shape)}')
        self.rows.append(
            dict(forward_index=get_forward_context().current_timestep,
                 global_patch_grid=grid,
                 blocks=math.prod(n // 4 for n in grid),
                 sparsity=BSA['sparsity']))

    def collect(self):
        if [row['forward_index'] for row in self.rows] != list(range(self.forwards)):
            raise ValueError('Unexpected executed LongCat forward sequence')
        return list(self.rows)
