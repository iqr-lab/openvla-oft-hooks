import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.robot.openvla_hooks.hook_runner import emit_all, set_enabled_hooks, set_hook_config
from experiments.robot.openvla_hooks.io import HookRecordWriter
from experiments.robot.openvla_hooks.runtime import collect_hook_records
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction


class _FakeVisionBackbone:
    def get_num_images_in_input(self):
        return 2

    def get_num_patches(self):
        return 256


class OpenVLAHookTests(unittest.TestCase):
    def tearDown(self):
        set_enabled_hooks([])
        set_hook_config({})

    def test_unknown_hook_raises(self):
        set_enabled_hooks(["does_not_exist"])
        with self.assertRaisesRegex(ValueError, "Unknown OpenVLA hook"):
            emit_all({})

    def test_disabled_hooks_emit_no_records(self):
        set_enabled_hooks([])
        self.assertEqual(emit_all({}), [])

    def test_collect_hook_records_for_lightweight_payload(self):
        set_enabled_hooks(["token_spans", "action_chunks"])
        payload = {
            "observation_input": {},
            "token_spans": {"prefix": {"start": 0, "end": 3}},
            "prefix_embeddings": torch.zeros(1, 3, 4),
            "prefix_final_hidden_state": torch.zeros(1, 3, 4),
            "prefix_gradients": None,
            "raw_attention_weights": None,
            "value_vectors": None,
            "action_chunks": {"chunks": torch.zeros(1, 1, 8, 7), "noises": None},
        }

        records = collect_hook_records(payload=payload, metadata={"query_idx": 0})

        self.assertEqual([record["hook_name"] for record in records], ["action_chunks", "token_spans"])
        self.assertEqual(tuple(records[0]["data"]["chunks"].shape), (1, 1, 8, 7))
        self.assertEqual(records[0]["metadata"]["query_idx"], 0)

    def test_token_spans_reflect_causal_prediction_shift_for_default_libero_inputs(self):
        model = OpenVLAForActionPrediction.__new__(OpenVLAForActionPrediction)
        model.vision_backbone = _FakeVisionBackbone()

        spans = model._build_token_spans(
            NUM_PATCHES=513,
            NUM_PROMPT_TOKENS=12,
            action_prediction_start=525,
            action_prediction_end=581,
            prefix_end=526,
            use_proprio=True,
        )

        self.assertEqual(spans["bos"], {"start": 0, "end": 1})
        self.assertEqual(spans["image"]["full"], {"start": 1, "end": 257})
        self.assertEqual(spans["image"]["wrist"], {"start": 257, "end": 513})
        self.assertEqual(spans["proprio"], {"start": 513, "end": 514})
        self.assertEqual(spans["prompt"], {"start": 514, "end": 526})
        self.assertEqual(spans["prefix"], {"start": 0, "end": 526})
        self.assertEqual(spans["action_prediction_positions"], {"start": 525, "end": 581})
        self.assertEqual(spans["action_tokens"], {"start": 526, "end": 582})
        self.assertEqual(spans["stop"], {"start": 582, "end": 583})

    def test_hook_writer_saves_and_updates_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HookRecordWriter(tmpdir, None, {"hooks": {"enabled": ["token_spans"]}})
            path = writer.save_query(
                inputs={"state": np.zeros(2)},
                outputs={"actions": np.zeros((1, 7))},
                hook_records=[{"hook_name": "token_spans", "data": {}}],
                metadata={"success": None},
            )
            writer.update_query_metadata(path, {"success": True})

            saved = np.load(Path(path), allow_pickle=True).item()
            self.assertEqual(Path(path).name, "step_0.npy")
            self.assertTrue((Path(tmpdir) / "output" / "hook_manifest.json").exists())
            self.assertIn("inputs/state", saved)
            self.assertIn("outputs/actions", saved)
            self.assertIn("hook_records", saved)
            self.assertTrue(saved["outputs/metadata/success"])
            self.assertTrue(saved["hook_records"][0]["metadata"]["success"])


if __name__ == "__main__":
    unittest.main()
