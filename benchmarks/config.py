"""Portable named workloads; model-specific validation stays beside their adapter."""
from typing import Any
import math
import re

from common import read_json, sha256


def validate_profile(profile) -> None:
    if set(profile) - {'execution_note'} != {
            'name', 'adapter', 'model_id', 'sampling', 'prompt', 'negative_prompt', 'pipeline', 'engine', 'environment',
            'contract'
    }:
        raise ValueError('Use the profile keys shown in configs/fasth3_v1.json')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', profile['name']):
        raise ValueError('Model name must be a safe directory name')
    if profile['adapter'] == 'fastwan21_14b':
        from fastwan import validate_profile as validate_wan
        validate_wan(profile)
        return
    if profile['adapter'] == 'longcat_refine':
        from longcat import validate_profile as validate_longcat
        validate_longcat(profile)
        return
    if profile['adapter'] != 'minimax_h3':
        raise ValueError('Only the minimax_h3 adapter is currently implemented')
    sampling, contract = profile['sampling'], profile['contract']
    for key in ['height', 'width', 'num_frames', 'fps', 'num_inference_steps']:
        if type(sampling[key]) is not int or sampling[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(sampling['seed']) is not int or not 0 <= sampling['seed'] < 2**32:
        raise ValueError('Invalid seed')
    if sampling['guidance_scale'] != 1 or sampling['batch_cfg'] is not False:
        raise ValueError('FastH3 distilled profiles require CFG=1 and batch_cfg=false')
    ladder = contract['dmd_denoising_steps']
    if (not ladder or any(type(x) is not int or not 0 < x < 1000 for x in ladder)
            or any(a <= b for a, b in zip(ladder, ladder[1:], strict=False))
            or sampling['num_inference_steps'] != len(ladder) + 1):
        raise ValueError('FastH3 steps count sigma nodes: expected len(DMD ladder) + 1')
    experimental = profile['pipeline']['experimental']
    if experimental['attention_backend'] != 'VIDEO_SPARSE_ATTN_H3':
        raise ValueError('FastH3 requires VSA-H3')
    if not math.isfinite(experimental['VSA_sparsity']) or not 0 < experimental['VSA_sparsity'] < 1:
        raise ValueError('Expected sparse VSA rate in (0, 1)')
    if experimental['VSA_tile_size'] != 64:
        raise ValueError('Portable FastH3 profiles require 64-token Triton tiles')
    if experimental.get('dmd_denoising_steps', ladder) != ladder:
        raise ValueError('Conflicting DMD ladder')
    if {'num_gpus', 'execution_backend', 'parallelism'} & profile['engine'].keys():
        raise ValueError('GPU topology is controlled by --gpus, not engine overrides')
    if any(not k.startswith('FASTVIDEO_') or not isinstance(v, str) for k, v in profile['environment'].items()):
        raise ValueError('Profile environment only accepts FASTVIDEO_* string values')
    if profile['environment'].get('FASTVIDEO_STAGE_LOGGING', '0') != '0':
        raise ValueError('Stage logging adds intermediate CUDA synchronization; keep it disabled')


def model_manifest(path, profile, gpu_counts, input_frames=None) -> dict[str, Any]:
    """Read small configs and file sizes, never hash full weight payloads at launch."""
    if profile['adapter'] == 'longcat_refine':
        from longcat import model_manifest as longcat_manifest
        return longcat_manifest(path, profile, gpu_counts, input_frames)
    if profile['adapter'] == 'fastwan21_14b':
        from fastwan import model_manifest as wan_manifest
        return wan_manifest(path, profile, gpu_counts)
    required = [
        'transformer/config.json', 'scheduler/scheduler_config.json', 'audio_scheduler/scheduler_config.json',
        'vae/config.json', 'audio_vae/config.json', 'text_encoder/config.json'
    ]
    for name in required:
        if not (path / name).is_file():
            raise ValueError(f'Missing model component: {path / name}')
    if not any((path / name).is_file() for name in ['model_index.json', 'modular_model_index.json']):
        raise ValueError('Pass a complete model snapshot, not its transformer subdirectory')
    contract = profile['contract']
    for component, key in [('scheduler', 'video_shift'), ('audio_scheduler', 'audio_shift')]:
        if read_json(path / component / 'scheduler_config.json')['shift'] != contract[key]:
            raise ValueError(f'{component} shift does not match selected profile')
    metadata = path / 'fastvideo_inference.json'
    if metadata.exists():
        data = read_json(metadata)
        if (data['dmd_denoising_steps'] != contract['dmd_denoising_steps']
                or data['num_inference_steps'] != profile['sampling']['num_inference_steps']):
            raise ValueError('Checkpoint schedule does not match selected profile')
    heads = read_json(path / 'transformer/config.json')['num_attention_heads']
    if any(heads % n for n in gpu_counts):
        raise ValueError(f'GPU counts must divide {heads} attention heads')
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
                note='Config hashes and weight sizes only; weight payloads are not hashed.')
