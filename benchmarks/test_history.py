"""A matching requested profile does not prove matching resolved policies/assets."""

import copy
import unittest

from report import history_mismatches


class HistoryTests(unittest.TestCase):
    def test_runtime_policy_and_model_identity_must_match(self):
        plan = dict(
            timing_schema_version=3, source_commit="test", warmup=1,
            inference_source_modified=False,
            models={"test": dict(profile={"engine": {}, "seed": 1024})},
            model_manifest_sha256={"test": "verified-checkpoint-a"},
        )
        row = dict(model="test", effective=dict(
            packages={"torch": "test"}, cuda="test", pipeline_config={},
            attention_implementations=["VSA"], devices=[dict(name="test GPU")],
            resolved_execution=dict(offload=False, disable_conditioning_cache=False), cpu_threads=8,
        ))
        history, old = dict(plan=copy.deepcopy(plan)), copy.deepcopy(row)
        self.assertEqual(history_mismatches(plan, row, history, old), [])
        for key in ["offload", "disable_conditioning_cache"]:
            changed = copy.deepcopy(row)
            changed["effective"]["resolved_execution"][key] = True
            self.assertIn("resolved_execution", history_mismatches(plan, changed, history, old))
        changed = copy.deepcopy(plan)
        changed["model_manifest_sha256"]["test"] = "verified-checkpoint-b"
        self.assertIn("model manifest", history_mismatches(changed, row, history, old))
        del changed["model_manifest_sha256"]
        self.assertIn("model manifest", history_mismatches(changed, row, history, old))


if __name__ == "__main__":
    unittest.main()
