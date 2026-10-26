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


class Budget(unittest.TestCase):
    # Reported by the operator for pilot-20261024f. Not remeasured here.
    LOAD_HEADROOM_BYTES = 69_500_000_000
    GENERATE_HEADROOM_BYTES = 72_000_000_000
    MEASURED_PARKED_BYTES = 77108124672
    MEASURED_FORWARD_SECONDS = 9.09

    def test_default_budget_is_50e9_and_keeps_the_margin(self):
        self.assertEqual(lean_resident.DEFAULT_BUDGET_BYTES, 50_000_000_000)
        self.assertEqual(lean_resident.MARGIN_BYTES, 10 * GIB)
        self.assertEqual(lean_resident.budget_bytes(), 50_000_000_000)
        for headroom in (self.LOAD_HEADROOM_BYTES, self.GENERATE_HEADROOM_BYTES):
            self.assertTrue(lean_resident.headroom_keeps_margin(
                headroom, lean_resident.DEFAULT_BUDGET_BYTES))
        previous = os.environ.get("PIREUS_LEAN_RESIDENT_BUDGET_BYTES")
        os.environ["PIREUS_LEAN_RESIDENT_BUDGET_BYTES"] = "123"
        try:
            self.assertEqual(lean_resident.budget_bytes(), 123)
        finally:
            if previous is None:
                os.environ.pop("PIREUS_LEAN_RESIDENT_BUDGET_BYTES", None)
            else:
                os.environ["PIREUS_LEAN_RESIDENT_BUDGET_BYTES"] = previous
        with self.assertRaises(ValueError):
            lean_resident.headroom_keeps_margin(-1, 0)

    def test_archived_prefix_and_accounted_peak(self):
        experts = checkpoint_experts()
        plan = lean_resident.select_resident_plan(experts, 50_000_000_000)
        self.assertEqual(plan["resident_layer_count"], 21)
        self.assertEqual(plan["resident_layers"][0], "llm.layers.2")
        self.assertEqual(plan["resident_layers"][-1], "llm.layers.22")
        self.assertEqual(plan["mmap_layers"][0], "llm.layers.23")
        self.assertEqual(plan["mmap_layers"][-1], "llm.layers.41")
        self.assertEqual(plan["resident_bytes"], 42681237504)
        self.assertEqual(plan["parked_bytes"], 34426847232)
        self.assertEqual(plan["staging_slots"], 2)
        self.assertEqual(plan["staging_copies"], 2)
        self.assertEqual(plan["staging_capacity_bytes"], 1811939328)
        self.assertEqual(plan["staging_bytes"], 7247757312)
        self.assertEqual(plan["accounted_bytes"], 49928994816)
        self.assertLessEqual(plan["accounted_bytes"], plan["budget_bytes"])
        self.assertEqual(plan["resident_bytes"] + plan["parked_bytes"],
                         sum(item["bytes"] for item in experts))
        self.assertTrue(lean_resident.headroom_keeps_margin(
            self.LOAD_HEADROOM_BYTES, plan["accounted_bytes"]))
        ordered = []
        for item in experts:
            key = lean_resident.layer_key(item["name"])
            if key not in ordered:
                ordered.append(key)
        ordered.sort(key=lean_resident._layer_sort_key)
        previous = -1
        for budget in range(0, 80_000_000_000, 1_000_000_000):
            try:
                chosen = lean_resident.select_resident_plan(experts, budget)
            except ValueError:
                continue
            count = chosen["resident_layer_count"]
            self.assertGreaterEqual(count, previous)
            self.assertEqual(chosen["resident_layers"], ordered[:count])
            self.assertEqual(chosen["mmap_layers"], ordered[count:])
            self.assertLessEqual(chosen["staging_slots"], 2)
            self.assertEqual(chosen["accounted_bytes"],
                             chosen["resident_bytes"] + chosen["staging_bytes"])
            previous = count
        self.assertEqual(
            lean_resident.select_resident_plan(experts, 50_000_000_000), plan)
        ratio = plan["parked_bytes"] / sum(item["bytes"] for item in experts)
        seconds = self.MEASURED_FORWARD_SECONDS * ratio
        print(json.dumps(dict(
            evidence="archived-rank1-checkpoint-scaled-by-job-17675-effective-rate",
            resident_layer_count=plan["resident_layer_count"],
            resident_bytes=plan["resident_bytes"],
            parked_bytes=plan["parked_bytes"],
            staging_bytes=plan["staging_bytes"],
            accounted_bytes=plan["accounted_bytes"],
            estimated_forward_seconds=round(seconds, 3),
            measured_forward_seconds=self.MEASURED_FORWARD_SECONDS,
            measured_parked_bytes=self.MEASURED_PARKED_BYTES,
        ), sort_keys=True))

    def test_nothing_and_everything(self):
        parameters = [dict(name="llm.layers.2.mlp.experts.w13_weight", bytes=1000)]
        with self.assertRaises(ValueError):
            lean_resident.select_resident_plan(parameters, budget=10)
        held = lean_resident.select_resident_plan(parameters, budget=1000)
        self.assertEqual(held["resident_layer_count"], 1)
        self.assertEqual(held["parked_bytes"], 0)
        self.assertEqual(held["staging_bytes"], 0)
        self.assertEqual(held["accounted_bytes"], 1000)

    def test_small_budget_keeps_two_slot_staging_inside_the_cap(self):
        # Six layers of 10 bytes. All-resident is 60, over the budget.
        # One resident layer plus two slots of pinned and device slabs is 50.
        parameters = [dict(name="llm.layers.%d.mlp.experts.w13_weight" % index, bytes=10)
                      for index in range(6)]
        plan = lean_resident.select_resident_plan(parameters, budget=55, align=1)
        self.assertEqual(plan["resident_layer_count"], 1)
        self.assertEqual(plan["mmap_layer_count"], 5)
        self.assertEqual(plan["staging_slots"], 2)
        self.assertEqual(plan["staging_capacity_bytes"], 10)
        self.assertEqual(plan["staging_bytes"], 40)
        self.assertEqual(plan["accounted_bytes"], 50)
        self.assertLessEqual(plan["accounted_bytes"], 55)
        with self.assertRaises(ValueError):
            lean_resident.prefetch_trace(["a"], slots=3)


class Prefetch(unittest.TestCase):
    def test_copy_of_the_next_layer_starts_before_compute_ends(self):
        layers = ["L%d" % index for index in range(5)]
        events = lean_resident.prefetch_trace(layers)
        self.assertTrue({event[2] for event in events} <= {0, 1})

        def at(kind, layer):
            return next(index for index, event in enumerate(events)
                        if event[0] == kind and event[1] == layer)

        for index, layer in enumerate(layers):
            self.assertLess(at("copy_end", layer), at("compute_begin", layer))
            self.assertLess(at("compute_begin", layer), at("compute_end", layer))
            self.assertEqual(events[at("copy_begin", layer)][2], index % 2)
            if index + 1 < len(layers):
                self.assertLess(at("copy_begin", layers[index + 1]), at("compute_end", layer))
            if index >= 2:
                self.assertLess(at("compute_end", layers[index - 2]), at("copy_begin", layer))
        self.assertEqual(lean_resident.prefetch_trace([]), [])


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
        self.assertIn("commit_residency", generate)
        self.assertNotIn("os.posix_fadvise", (runtime / "lean_resident.py").read_text())
        self.assertIn("select_resident_plan", profile)
        pilot = (ROOT / "pilot.py").read_text()
        self.assertIn(
            'launch_cell_stage(target, "generate", "offline-generate",\n'
            '                                             target / "offline-bundle.json", 240)',
            pilot)
        self.assertIn("promotion_threshold_percent=5", pilot)
        self.assertIn("total_proposals", pilot)


if __name__ == "__main__":
    unittest.main()
