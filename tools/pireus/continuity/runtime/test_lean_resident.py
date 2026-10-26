#!/usr/bin/env python3
"""Local evidence for the file-backed routed-expert peak. No GPU and no torch."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import lean_resident

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "validation/uninstrumented-complete-11958/launch.log"
META = ROOT / "validation/model-meta-profile-11887.json"
GIB = 1024**3
# Measured on the meta probe (jobs 11886/11887): per-rank parameter storage.
PARAMETER_STORAGE_BYTES = 83111894180
UINT8_BYTES = 62813896704
FP8_BYTES = 7851737088
LAYER2_EXPERT_BYTES = 6 * GIB


def checkpoint_experts(rank="1"):
    rows = {}
    for line in LOG.read_text().splitlines():
        if "OFFLINE_CHECKPOINT_TENSOR_BEGIN" not in line:
            continue
        row = json.loads(line)
        if row.get("stage") != "OFFLINE_CHECKPOINT_TENSOR_BEGIN" or row.get("rank") != rank:
            continue
        name = row["name"]
        parts = name.split(".")
        if "shared_experts" in parts:
            continue
        if ".experts.w13_weight" not in name and ".experts.w2_weight" not in name:
            continue
        if name in rows:
            raise AssertionError("duplicate checkpoint tensor " + name)
        if row["bytes"] % 2:
            raise AssertionError("expert tensor is not an even TP shard " + name)
        param = name[len("model."):] if name.startswith("model.") else name
        param = param.replace(".w13_weight.", ".w13_weight_").replace(".w2_weight.", ".w2_weight_")
        rows[name] = dict(name=param, bytes=row["bytes"] // 2, dtype=row["dtype"])
    return list(rows.values())


def measured_parameters():
    experts = checkpoint_experts()
    expert_bytes = sum(item["bytes"] for item in experts)
    if expert_bytes > PARAMETER_STORAGE_BYTES:
        raise AssertionError("expert bytes exceed measured parameter storage")
    return experts + [dict(name="llm.norm.weight", bytes=PARAMETER_STORAGE_BYTES - expert_bytes)]


class Estimate(unittest.TestCase):
    def test_names(self):
        self.assertTrue(lean_resident.is_routed_expert_parameter(
            "llm.layers.23.mlp.experts.w13_weight"))
        self.assertTrue(lean_resident.is_routed_expert_parameter(
            "mlp.experts.w13_weight_scale"))
        self.assertTrue(lean_resident.is_routed_expert_parameter("w2_weight_scale_2"))
        self.assertFalse(lean_resident.is_routed_expert_parameter(
            "llm.layers.23.mlp.shared_experts.w13_weight"))
        self.assertFalse(lean_resident.is_routed_expert_parameter(
            "mlp.experts.w13_input_scale"))
        self.assertFalse(lean_resident.is_routed_expert_parameter("llm.embed_tokens.weight"))

    def test_active_layer_not_the_sum(self):
        parameters = [
            dict(name="llm.layers.0.mlp.experts.w13_weight", bytes=100),
            dict(name="llm.layers.0.mlp.experts.w2_weight", bytes=40),
            dict(name="llm.layers.1.mlp.experts.w13_weight", bytes=10),
            dict(name="llm.layers.0.mlp.shared_experts.w13_weight", bytes=7),
            dict(name="llm.norm.weight", bytes=3),
        ]
        estimate = lean_resident.estimate_anonymous_peak(parameters)
        self.assertEqual(estimate["expert_bytes"], 150)
        self.assertEqual(estimate["max_active_layer_bytes"], 140)
        self.assertEqual(estimate["anonymous_before_bytes"], 160)
        self.assertEqual(estimate["anonymous_after_peak_bytes"], 150)
        self.assertEqual(estimate["saved_bytes"], 10)
        self.assertEqual(estimate["resident_non_expert_bytes"], 10)

    def test_archived_rank_peak(self):
        experts = checkpoint_experts()
        by_dtype = {}
        for item in experts:
            by_dtype[item["dtype"]] = by_dtype.get(item["dtype"], 0) + item["bytes"]
        self.assertEqual(by_dtype["torch.uint8"], UINT8_BYTES)
        self.assertEqual(by_dtype["torch.float8_e4m3fn"], FP8_BYTES)
        self.assertEqual(by_dtype["torch.bfloat16"], LAYER2_EXPERT_BYTES)
        meta = json.loads(META.read_text())["observations"][0]
        self.assertEqual(meta["parameter_storage_bytes"], PARAMETER_STORAGE_BYTES)
        self.assertEqual(meta["dtype_bytes"]["torch.uint8"], UINT8_BYTES)
        self.assertEqual(meta["dtype_bytes"]["torch.float8_e4m3fn"], FP8_BYTES)
        estimate = lean_resident.estimate_anonymous_peak(measured_parameters())
        self.assertGreaterEqual(estimate["saved_bytes"], 6 * 1000**3)
        self.assertEqual(estimate["saved_bytes"], estimate["expert_bytes"] - LAYER2_EXPERT_BYTES)
        self.assertEqual(estimate["max_active_layer_bytes"], LAYER2_EXPERT_BYTES)
        self.assertEqual(estimate["anonymous_before_bytes"], PARAMETER_STORAGE_BYTES)
        self.assertEqual(
            estimate["anonymous_after_peak_bytes"],
            PARAMETER_STORAGE_BYTES - estimate["expert_bytes"] + LAYER2_EXPERT_BYTES)
        self.assertGreater(estimate["saved_bytes"], 65 * GIB)
        print(json.dumps(dict(evidence="archived-rank1-checkpoint-and-meta-11887", **estimate),
                         sort_keys=True))


class Protocol(unittest.TestCase):
    def test_sparse_park_and_one_resident_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            resident = lean_resident.ByteResident(directory)
            self.assertEqual(resident.park_fresh("w13", 4096), 0)
            slot = resident.slots["w13"]
            self.assertEqual(slot["path"].stat().st_size, 4096)
            self.assertEqual(slot["path"].stat().st_blocks, 0)
            payload = bytes([index % 251 for index in range(4096)])
            resident.write("w13", payload)
            self.assertEqual(resident.anonymous, 0)
            self.assertEqual(resident.materialize("w13"), payload)
            self.assertEqual(resident.anonymous, 4096)
            resident.release("w13")
            self.assertEqual(resident.anonymous, 0)
            slot["file"].seek(0)
            self.assertEqual(slot["file"].read(), payload)
            self.assertEqual(resident.materialize("w13"), payload)
            resident.release("w13")
            resident.close()

    def test_sequential_peak_is_the_largest_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            resident = lean_resident.ByteResident(directory)
            sizes = {"a": 100, "b": 400, "c": 50}
            for name, size in sizes.items():
                resident.park_fresh(name, size)
                resident.write(name, bytes([len(name)]) * size)
            for name in sizes:
                resident.materialize(name)
                resident.release(name)
            self.assertEqual(resident.peak, 400)
            self.assertEqual(resident.anonymous, 0)
            resident.materialize("a")
            resident.materialize("b")
            self.assertEqual(resident.anonymous, 500)
            self.assertEqual(resident.peak, 500)
            resident.close()


class Wiring(unittest.TestCase):
    def test_offline_generate_is_opt_in_and_the_guard_stays(self):
        runtime = Path(__file__).resolve().parent
        launch = (runtime / "launch_pair.py").read_text()
        self.assertIn('"lean_resident.py"', launch)
        self.assertIn(
            "PIREUS_LEAN_RESIDENT=1 PIREUS_OFFLINE_MODE=generate",
            launch)
        generate = (runtime / "offline_generate.py").read_text()
        self.assertIn('os.environ.get("PIREUS_LEAN_RESIDENT") == "1"', generate)
        self.assertIn("lean_resident.install()", generate)
        profile = (runtime / "profile_model_memory.py").read_text()
        self.assertIn("estimate_anonymous_peak", profile)
        self.assertIn("raise SystemExit(74)", profile)
        self.assertNotIn("lean_resident.install()", profile)
        guard = (runtime / "memory_guard.py").read_text()
        self.assertIn("choices=[33, 36]", guard)
        self.assertIn("RESERVE_BYTES = 36 * 1024**3", guard)
        serve = (runtime / "serve_rank.sh").read_text()
        self.assertIn("--reserve-gib 33", serve)
        self.assertNotIn("PIREUS_LEAN_RESIDENT", serve)


if __name__ == "__main__":
    unittest.main()
