"""FastWan 2.1 adapter: native pipeline checks and VSA observation only."""
import json
import math
from pathlib import Path
import struct
from dataclasses import asdict
from typing import Any

from common import read_json, sha256

MODEL_ID = 'FastVideo/FastWan2.1-T2V-14B-Diffusers'
REVISION = 'ab7d6149c10bb41ba39226e2262cd2e9ee85b88e'


def geometry(sampling: dict[str, Any]) -> tuple[list[int], int, int]:
    grid = [(sampling['num_frames'] - 1) // 4 + 1, sampling['height'] // 16, sampling['width'] // 16]
    blocks = math.prod(math.ceil(n / 4) for n in grid)
    return grid, blocks, max(1, math.ceil((1 - 0.9) * blocks))


def validate_profile(profile):
    if profile['model_id'] != MODEL_ID:
        raise ValueError('Require the selected FastWan 2.1 14B release')
    s = profile['sampling']
    if set(s) != {'height', 'width', 'num_frames', 'fps', 'seed', 'guidance_scale', 'num_inference_steps', 'batch_cfg'}:
        raise ValueError('Unexpected Wan sampling keys')
    for key in ['height', 'width', 'num_frames', 'fps']:
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if s['height'] % 16 or s['width'] % 16 or s['num_frames'] % 4 != 1:
        raise ValueError('Wan dimensions must preserve VAE and patch shape')
    if type(s['seed']) is not int or not 0 <= s['seed'] < 2**32:
        raise ValueError('Invalid seed')
    if s['num_inference_steps'] != 3 or s['guidance_scale'] != 1 or s['batch_cfg'] is not False:
        raise ValueError('Require three DMD forwards / CFG1 / batch_cfg=false')
    if profile['contract'] != dict(dmd_denoising_steps=[1000, 757, 522], flow_shift=8.0):
        raise ValueError('Require native FastWan DMD ladder and shift8')
    if profile['pipeline'] != dict(workload_type='t2v',
                                   experimental=dict(attention_backend='VIDEO_SPARSE_ATTN', VSA_sparsity=0.9)):
        raise ValueError('Require native Wan VSA with sparsity0.9')
    if not isinstance(profile['prompt'], str) or not profile['prompt'].strip() or profile['negative_prompt'] != '':
        raise ValueError('Require a prompt and no unused negative prompt for CFG1')
    if {'num_gpus', 'execution_backend', 'parallelism'} & profile['engine'].keys():
        raise ValueError('GPU topology is controlled by --gpus')
    if any(not key.startswith('FASTVIDEO_') or not isinstance(value, str)
           for key, value in profile['environment'].items()):
        raise ValueError('Only FASTVIDEO_* string environment settings are accepted')
    if profile['environment'].get('FASTVIDEO_STAGE_LOGGING', '0') != '0':
        raise ValueError('Stage logging adds CUDA synchronizations')


def tensor_headers(path: Path) -> dict[str, Any]:
    tensors = {}
    for shard in sorted((path / 'transformer').glob('*.safetensors')):
        with shard.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if size > 64 * 1024**2:
                raise ValueError('Invalid safetensors header')
            header = json.loads(stream.read(size))
        for name, entry in header.items():
            if name == '__metadata__':
                continue
            if name in tensors or entry['data_offsets'][1] + size + 8 > shard.stat().st_size:
                raise ValueError(f'Duplicate tensor or truncated shard: {name}')
            tensors[name] = dict(entry, file=str(shard))
    return tensors


def model_manifest(path, profile, gpu_counts):
    path = Path(path)
    verified = read_json(path / 'download_verification.json')
    if (verified.get('model_id'), verified.get('revision'), verified.get('status')) != (MODEL_ID, REVISION, 'complete'):
        raise ValueError('Require a complete verified pinned FastWan snapshot')
    configs, weights = {}, {}
    for entry in verified['files']:
        item = path / entry['file']
        if not item.is_file() or item.stat().st_size != entry['size']:
            raise ValueError(f'Missing/truncated pinned model asset: {item}')
        if item.suffix == '.json':
            configs[entry['file']] = dict(sha256=sha256(item), config=read_json(item))
            if configs[entry['file']]['sha256'] != entry['sha256']:
                raise ValueError(f'Modified pinned config: {item}')
        elif item.suffix == '.safetensors':
            weights[entry['file']] = item.stat().st_size
    if read_json(path / 'model_index.json')['_class_name'] != 'WanDMDPipeline':
        raise ValueError('Expected WanDMDPipeline')
    config = read_json(path / 'transformer/config.json')
    if (config['num_layers'], config['num_attention_heads'], config['attention_head_dim']) != (40, 40, 128):
        raise ValueError('Unexpected 14B architecture')
    if any(n < 1 or 40 % n for n in gpu_counts):
        raise ValueError('GPU counts must divide 40 attention heads')
    tensors = tensor_headers(path)
    gates = {name: entry for name, entry in tensors.items() if '.to_gate_compress.' in name}
    expected = {f'blocks.{i}.to_gate_compress.{kind}' for i in range(40) for kind in ['weight', 'bias']}
    if set(gates) != expected:
        raise ValueError('Incomplete learned VSA gates')
    for name, entry in gates.items():
        if entry['dtype'] != 'BF16' or entry['shape'] != ([5120, 5120] if name.endswith('weight') else [5120]):
            raise ValueError(f'Unexpected VSA gate geometry: {name}')
    return dict(model_id=MODEL_ID,
                revision=REVISION,
                configs=configs,
                weight_file_sizes=weights,
                download_verification=verified,
                transformer_tensors=len(tensors),
                trained_gate_tensors=len(gates),
                note='All downloaded LFS weights were SHA256 verified during preparation; launch checks sizes/configs.')


def loaded_block_tensor(block: Any, name: str) -> Any:
    """Read native offload host storage without materializing or detaching its hook."""
    manager = getattr(block, '_hook_manager', None)
    hook = manager.forward_hooks.get('LayerwiseOffloadHook') if manager is not None else None
    if hook is not None and name in hook.state.cpu_named_parameters:
        return hook.state.cpu_named_parameters[name]
    return block.get_parameter(name)


def check_pipeline(pipe, profile):
    import torch
    from safetensors import safe_open
    from fastvideo.attention.backends.video_sparse_attn import VSA_TILE_SIZE
    from fastvideo.pipelines.lazy_module import is_lazy_module
    from runtime import json_config

    transformer = pipe.get_module('transformer')
    if is_lazy_module(transformer):
        raise ValueError('DiT hooks require a stable, eagerly loaded transformer')
    config = pipe.fastvideo_args.pipeline_config
    if type(pipe).__name__ != 'WanDMDPipeline' or list(config.dmd_denoising_steps) != [1000, 757, 522]:
        raise ValueError('Expected native Wan DMD pipeline / three fixed transitions')
    if config.flow_shift != 8.0 or pipe.get_module('scheduler').config.shift != 8.0:
        raise ValueError('Unexpected inference scheduler shift')
    stages = [stage for stage in pipe.stages if type(stage).__name__ == 'DmdDenoisingStage']
    if len(stages) != 1 or stages[0].scheduler.config.shift != 8.0 or len(stages[0].scheduler.timesteps) != 1000:
        raise ValueError('DMD must index a separate complete shift8 training-noise table')
    if pipe.fastvideo_args.VSA_sparsity != 0.9 or VSA_TILE_SIZE != (4, 4, 4):
        raise ValueError('Unexpected Wan VSA sparsity/tile shape')
    headers = tensor_headers(Path(pipe.fastvideo_args.model_path))
    routes, gates = [], []
    # Verify gate payloads in full, outside timing, one layer at a time.
    for index, block in enumerate(transformer.blocks):
        if type(block).__name__ != 'WanTransformerBlock_VSA':
            raise ValueError('Native model did not select VSA blocks')
        impl = block.attn1.attn_impl
        if type(impl).__name__ != 'VideoSparseAttentionImpl':
            raise ValueError(f'Unexpected Wan attention implementation: {type(impl).__name__}')
        routes.append(type(impl).__name__)
        for kind in ['weight', 'bias']:
            name = f'blocks.{index}.to_gate_compress.{kind}'
            parameter = loaded_block_tensor(block, f'to_gate_compress.{kind}')
            if hasattr(parameter, 'full_tensor'):
                parameter = parameter.full_tensor()
            with safe_open(headers[name]['file'], framework='pt', device='cpu') as source:
                expected = source.get_tensor(name)
                actual = parameter.detach().cpu().to(expected.dtype)
                if not torch.equal(actual, expected):
                    raise ValueError(f'Loaded VSA gate differs from checkpoint: {name}')
            gates.append(name)
    if len(routes) != 40 or len(gates) != 80:
        raise ValueError('Expected all 40 layers / 80 trained gate tensors')
    grid, blocks, topk = geometry(profile['sampling'])
    return transformer, dict(
        pipeline_class=type(pipe).__name__,
        transformer_class=type(transformer).__name__,
        attention_implementations=routes,
        pipeline_config=json_config(asdict(config)),
        expected_dit_forwards=3,
        video_grid=grid,
        video_blocks=blocks,
        topk=topk,
        vsa_tile_shape=list(VSA_TILE_SIZE),
        verified_gate_tensors=len(gates),
        resolved_execution={
            key: getattr(pipe.fastvideo_args, key)
            for key in [
                'dit_cpu_offload', 'dit_layerwise_offload', 'text_encoder_cpu_offload', 'vae_cpu_offload',
                'pin_cpu_memory', 'lazy_module_load', 'use_fsdp_inference', 'sp_size', 'tp_size',
                'enable_torch_compile', 'inference_torch_compile', 'enable_torch_compile_vae', 'vae_parallel_decode',
                'video_decode_backend'
            ]
        },
        forward_layout=[dict(forward_index=i, dmd_rung=rung) for i, rung in enumerate(config.dmd_denoising_steps)])


class VSAExecutionAudit:
    """Inspect native metadata without copying tensors, changing math or synchronizing CUDA."""

    def __init__(self, transformer, profile):
        self.grid, self.blocks, self.topk = geometry(profile['sampling'])
        self.rows = []
        self.handle = transformer.register_forward_pre_hook(self.observe)

    def observe(self, module, inputs):
        from fastvideo.forward_context import get_forward_context
        from fastvideo.attention.backends.video_sparse_attn import compute_topk
        context = get_forward_context()
        metadata = context.attn_metadata
        if metadata is None or metadata.variable_block_sizes.numel() != self.blocks:
            raise ValueError('Missing or incorrect runtime VSA metadata')
        topk = compute_topk(metadata.VSA_sparsity, metadata.variable_block_sizes.numel())
        if topk != self.topk:
            raise ValueError('Unexpected runtime VSA topk')
        self.rows.append(
            dict(forward_index=context.current_timestep,
                 video_blocks=self.blocks,
                 topk=topk,
                 sparsity=metadata.VSA_sparsity))

    def collect(self):
        if len(self.rows) != 3 or [r['forward_index'] for r in self.rows] != [0, 1, 2]:
            raise ValueError('Expected exactly three native VSA transformer calls')
        return list(self.rows)
