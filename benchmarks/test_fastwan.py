"""Regression coverage for Wan admission and runtime audit without model weights."""
import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from common import HERE, read_json
from config import validate_profile
from fastwan import VSAExecutionAudit, geometry, loaded_block_tensor, tensor_headers


class FastWanTests(unittest.TestCase):

    def test_gate_verification_reads_native_offload_host_payload(self):
        import torch
        block = torch.nn.Module()
        block.to_gate_compress = torch.nn.Linear(2, 2)
        expected = torch.ones(2, 2)
        self.assertIs(loaded_block_tensor(block, 'to_gate_compress.weight'), block.to_gate_compress.weight)
        state = SimpleNamespace(cpu_named_parameters={'to_gate_compress.weight': expected}, gpu_named_parameters={})
        block._hook_manager = SimpleNamespace(forward_hooks={'LayerwiseOffloadHook': SimpleNamespace(state=state)})
        block.to_gate_compress.weight.data = torch.empty(0, 0)
        self.assertIs(loaded_block_tensor(block, 'to_gate_compress.weight'), expected)
        self.assertEqual(block.to_gate_compress.weight.numel(), 0)
        self.assertEqual(state.gpu_named_parameters, {})

    def test_wan_uses_three_steps_and_its_own_metadata_contract(self):
        profile = read_json(HERE / 'configs/fastwan21_14b.json')
        validate_profile(profile)
        self.assertEqual(geometry(profile['sampling']), ([21, 45, 80], 1440, 144))
        for section, key, value in [('sampling', 'num_inference_steps', 4), ('sampling', 'height', 721),
                                    ('contract', 'flow_shift', 3), ('contract', 'dmd_denoising_steps', [999, 757,
                                                                                                        522])]:
            bad = copy.deepcopy(profile)
            bad[section][key] = value
            with self.assertRaises(ValueError):
                validate_profile(bad)
        bad = copy.deepcopy(profile)
        bad['pipeline']['experimental']['attention_backend'] = 'FLASH_ATTN'
        with self.assertRaises(ValueError):
            validate_profile(bad)

    def test_shard_header_rejects_truncation_and_duplicate_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'transformer').mkdir()
            header = json.dumps({'weight': dict(dtype='BF16', shape=[1], data_offsets=[0, 2])}).encode()
            payload = struct.pack('<Q', len(header)) + header
            shard = path / 'transformer/one.safetensors'
            shard.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, 'truncated'):
                tensor_headers(path)
            shard.write_bytes(payload + bytes(2))
            self.assertIn('weight', tensor_headers(path))
            (path / 'transformer/two.safetensors').write_bytes(payload + bytes(2))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                tensor_headers(path)

    def test_runtime_audit_rejects_dense_or_incomplete_execution(self):
        import torch
        from fastvideo.forward_context import set_forward_context

        transformer = torch.nn.Identity()
        audit = VSAExecutionAudit(transformer, read_json(HERE / 'configs/fastwan21_14b.json'))
        metadata = SimpleNamespace(variable_block_sizes=torch.empty(1440), VSA_sparsity=0.9)
        try:
            with set_forward_context(current_timestep=0,
                                     attn_metadata=None), self.assertRaisesRegex(ValueError, 'metadata'):
                transformer(torch.empty(1))
            for step in range(3):
                with set_forward_context(current_timestep=step, attn_metadata=metadata):
                    transformer(torch.empty(1))
            self.assertEqual(len(audit.collect()), 3)
            self.assertEqual({r['topk'] for r in audit.collect()}, {144})
            audit.rows.pop()
            with self.assertRaisesRegex(ValueError, 'three'):
                audit.collect()
        finally:
            audit.handle.remove()


if __name__ == '__main__':
    unittest.main()
