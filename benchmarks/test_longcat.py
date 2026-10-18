"""Check LongCat workload alignment against native scheduling and canvas selection."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common import HERE, read_json
from config import validate_profile
from longcat import refinement_timesteps
from runtime import generator_config, make_request


class LongCatTests(unittest.TestCase):

    def setUp(self):
        self.profile = read_json(HERE / 'configs/longcat_refine.json')

    def test_native_canvas_default_and_explicit_alignment(self):
        from fastvideo.pipelines.stages.longcat_refine_init import refinement_canvas
        self.assertEqual(refinement_canvas(640, 640), (960, 960))
        self.assertEqual(refinement_canvas(640, 640, [1024, 1024]), (1024, 1024))
        for bad in [[1000, 1024], [0, 1024], [1024], [True, 1024]]:
            with self.subTest(size=bad), self.assertRaises(ValueError):
                refinement_canvas(640, 640, bad)

    def test_real_profile_native_schedule_and_request(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        validate_profile(self.profile)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=12.0)
        original = scheduler.timesteps.clone()
        ladder = refinement_timesteps(scheduler, self.profile)
        self.assertEqual(len(ladder), 5)
        self.assertEqual(ladder[0], 500.0)
        self.assertAlmostEqual(ladder[1], 443.0223388671875)
        self.assertTrue(original.equal(scheduler.timesteps))
        config = generator_config(self.profile, '/example/LongCat-Video-FastVideo', 4)
        self.assertTrue(config.engine.offload.dit_layerwise)
        self.assertFalse(config.engine.offload.pin_cpu_memory)
        self.assertEqual(config.pipeline.components.lora_nickname, 'refinement')
        self.assertEqual(config.pipeline.experimental['refine_target_size'], [1024, 1024])
        self.assertNotIn('dmd_denoising_steps', config.pipeline.experimental)
        request = make_request(self.profile, Path('/tmp/video.mp4'), stage1_frames=[])
        self.assertEqual(request.extensions, dict(t_thresh=0.5, spatial_refine_only=False, num_cond_frames=0))
        self.assertEqual(request.sampling.num_frames, 186)

    def test_sparse_profile_cannot_silently_change_work(self):
        for section, key, value in [('sampling', 'guidance_scale', 2), ('contract', 'expected_dit_forwards', 50),
                                    ('pipeline', 'experimental', dict(enable_bsa=False))]:
            changed = copy.deepcopy(self.profile)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_profile(changed)

    def test_refinement_restores_vae_before_each_encode_and_honors_offload(self):
        import numpy as np
        import torch
        from fastvideo.pipelines.stages.longcat_refine_init import LongCatRefineInitStage

        test = self

        class VAE:

            def __init__(self) -> None:
                self.device = torch.device('meta')  # Unloaded: cannot encode until restored.
                self.config = SimpleNamespace()
                self.events: list[tuple[str, str]] = []

            def to(self, device):
                self.device = torch.device(device)
                self.events.append(('move', self.device.type))
                return self

            def parameters(self):
                yield SimpleNamespace(device=self.device, dtype=torch.float32)

            def encode(self, video):
                test.assertEqual(self.device, torch.device('cpu'))
                test.assertEqual(video.device, self.device)
                self.events.append(('encode', self.device.type))
                return SimpleNamespace(latents=torch.zeros(1, 16, 4, 8, 8))

        for offload in [True, False]:
            with self.subTest(offload=offload):
                vae = VAE()
                stage = LongCatRefineInitStage(vae)
                args = SimpleNamespace(sp_size=1,
                                       vae_cpu_offload=offload,
                                       pipeline_config=SimpleNamespace(refine_target_size=[64, 64]))
                with patch('fastvideo.pipelines.stages.longcat_refine_init.get_local_torch_device',
                           return_value=torch.device('cpu')):
                    for _ in range(2):
                        batch = SimpleNamespace(refine_from=None,
                                                stage1_video=[np.zeros((64, 64, 3), dtype=np.uint8)],
                                                spatial_refine_only=False,
                                                t_thresh=0.5,
                                                num_cond_frames=0,
                                                prompt_embeds=[torch.zeros(1)])
                        result = stage.forward(batch, args)
                        self.assertEqual(tuple(result.latents.shape), (1, 16, 4, 8, 8))
                    stage.forward(SimpleNamespace(refine_from=None, stage1_video=None), args)
                expected = [('move', 'cpu'), ('encode', 'cpu')]
                if offload:
                    expected.append(('move', 'cpu'))
                self.assertEqual(vae.events, expected * 2)


if __name__ == '__main__':
    unittest.main()
