import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import os
import subprocess
import benchmark_pair
from benchmark_pair import deduplicate_candidates, observe_available, remote_root_for, stage, NODES
from cycle import HERE, encoded, digest
from pilot import settled_job_state, cells, paired_completion, launch_cell_stage, validate_checks, REQUIRED_CHECKS, summarize
from tokenized_cycle import pack_offline


class PilotControls(unittest.TestCase):
    def test_design_has_nine_independent_frozen_cells(self):
        design = cells()
        self.assertEqual(len(design), 9)
        self.assertEqual(len({c["id"] for c in design}), 9)
        self.assertEqual(sum(c["budget"] for c in design), 288)
        for round_id in range(3):
            round_cells = [c for c in design if c["round"] == round_id]
            self.assertEqual({c["condition"] for c in round_cells},
                             {"deterministic", "inkling-no-ontology", "inkling-ontology"})

    def test_dedup_requires_native_identity_and_identical_material(self):
        a = dict(id="000", layout=0, ptx="000.ptx", ptx_sha256="a" * 64)
        b = a | dict(id="001", ptx="001.ptx")
        c = a | dict(id="002", ptx="002.ptx")
        selected, aliases = deduplicate_candidates([a, b, c], {"000": 10, "001": 10, "002": 11})
        self.assertEqual([v["id"] for v in selected], ["000", "002"])
        self.assertEqual(aliases, {"000": "000", "001": "000", "002": "002"})
        with self.assertRaisesRegex(ValueError, "different material"):
            deduplicate_candidates([a, b | dict(ptx_sha256="b" * 64)], {"000": 10, "001": 10})

    def test_paired_completion_rejects_missing_duplicate_and_mixed_jobs(self):
        launch = dict(mode="offline-generate", command=["srun"], nodes=[dict(pod="a"), dict(pod="b")])
        completed = [dict(stage="OFFLINE_CYCLE_COMPLETE", rank=str(r), job="123") for r in range(2)]
        def log(items):
            return "\n".join(json.dumps(v) for v in [launch, *items])
        self.assertEqual(paired_completion(log(completed), "OFFLINE_CYCLE_COMPLETE")[0], "123")
        for bad in [completed[:1], completed + completed[:1],
                    [completed[0], completed[1] | dict(job="124")]]:
            with self.assertRaises(ValueError):
                paired_completion(log(bad), "OFFLINE_CYCLE_COMPLETE")

    def test_interrupted_stage_never_submits_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "offline-bundle.json"
            bundle.write_text("{}")
            folder = root / "pilot-stages" / "generate"
            folder.mkdir(parents=True)
            command = [sys.executable, str(HERE / "runtime/launch_pair.py"), "offline-generate",
                       "--minutes", "120", "--input-bundle", str(bundle)]
            (folder / "intent.json").write_bytes(encoded(dict(command=command,
                                                             bundle_sha256=digest(bundle.read_bytes()))))
            (folder / "launch.log").write_text("interrupted before paired completion\n")
            with patch("pilot.subprocess.run") as submit:
                with self.assertRaises(ValueError):
                    launch_cell_stage(root, "generate", "offline-generate", bundle, 120)
                submit.assert_not_called()

    def test_ci_acceptance_requires_current_head_and_latest_success(self):
        head = "a" * 40
        rows = [[name, "completed", "success", head, "2026-09-07T12:00:00Z", "url"]
                for name in REQUIRED_CHECKS]
        def check(candidate, remote_head=head):
            with patch("pilot.subprocess.check_output", side_effect=[
                    remote_head + "\n", "\n".join(json.dumps(v) for v in candidate)]):
                return validate_checks(dict(source_commit=head))
        self.assertEqual(check(rows)["source_commit"], head)
        for altered in [rows[:-1],
                        rows + [rows[0][:1] + ["in_progress", None, head,
                                               "2026-09-07T13:00:00Z", "url"]],
                        [rows[0][:1] + ["completed", "failure", *rows[0][3:]], *rows[1:]]]:
            with self.assertRaises(ValueError):
                check(altered)
        with self.assertRaisesRegex(ValueError, "current PR head"):
            check(rows, "b" * 40)

    def test_summary_refuses_completion_without_full_hardware_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = cells()
            (root / "pilot-manifest.json").write_text(json.dumps(
                dict(source_commit="a" * 40, cells=design)))
            value = dict(generated=32, validated=32, admitted=32, unique_plans=4,
                         hardware_benchmarked=0, gain_eligible=0)
            for cell in design:
                target = root / cell["id"]
                target.mkdir()
                (target / "pilot-cell-complete.json").write_text(json.dumps(value))
            with patch("pilot.verify", return_value={}), patch("pilot.report", return_value=value):
                with self.assertRaisesRegex(ValueError, "full evidence"):
                    summarize(root)

    def test_undeclared_offline_budgets_refuse_before_receipt_access(self):
        for budget in [0, 7, 9, 31, 33, 288]:
            with self.assertRaisesRegex(ValueError, "8- or 32-proposal"):
                pack_offline(Path("/never-accessed"), dict(transport="sglang-offline-token-ids", budget=budget))


class RunScopedRemoteRoot(unittest.TestCase):
    """pilot-20261029a reused pilot-20261024g's worker directory: identical cell
    manifests named the same cycle-* root, so probe/timing were observed instead
    of measured and the write-once stage refused a different gpu-manifest."""

    def make_cell(self, base, run, gpu_manifest):
        root = base / run / "round-0-deterministic"
        root.mkdir(parents=True)
        (root / "manifest.json").write_bytes(encoded(dict(schema=1, condition="deterministic")))
        (root / "gpu-manifest.json").write_text(gpu_manifest)
        return root

    def worker(self, scratch):
        (scratch / "runtime").mkdir(parents=True)
        def mapped(path):
            return str(scratch / Path(path).relative_to("/scratch/pireus"))
        def fake_remote(pod, *args, stdin=None):
            self.assertEqual(args[:2], ("python3", "-c"))
            program = args[2].replace("os.chown(root,1000,1000)", "None")
            return subprocess.check_output([sys.executable, "-c", program, mapped(args[3])], input=stdin,
                                           stderr=subprocess.DEVNULL)
        def fake_test(command, **kwargs):
            self.assertEqual(command[-3:-1], ["test", "-f"])
            return SimpleNamespace(returncode=0 if os.path.isfile(mapped(command[-1])) else 1)
        return mapped, fake_remote, fake_test

    def test_identical_manifests_in_two_runs_never_share_a_remote_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            a = self.make_cell(base, "pilot-a", "{}")
            b = self.make_cell(base, "pilot-b", "{}")
            self.assertEqual((a / "manifest.json").read_bytes(), (b / "manifest.json").read_bytes())
            first = remote_root_for(a)
            self.assertNotEqual(first, remote_root_for(b))
            self.assertTrue(first.startswith("/scratch/pireus/runs/cycle-"
                                             + digest((a / "manifest.json").read_bytes())[:20] + "-"))
            identity = (a / "remote-identity.json").read_bytes()
            journal = (a / "journal.jsonl").read_bytes()
            self.assertEqual(remote_root_for(a), first)
            self.assertEqual((a / "remote-identity.json").read_bytes(), identity)
            self.assertEqual((a / "journal.jsonl").read_bytes(), journal)
            # A run dir recreated at the same path is a different run: a fresh nonce.
            (base / "pilot-a").rename(base / "pilot-a.previous")
            reused = self.make_cell(base, "pilot-a", "{}")
            self.assertEqual(reused, a)
            self.assertNotEqual(remote_root_for(reused), first)
            (a / "manifest.json").write_bytes(encoded(dict(schema=1, condition="other")))
            with self.assertRaisesRegex(ValueError, "another cell manifest"):
                remote_root_for(a)

    def test_observations_and_staging_never_cross_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            mapped, fake_remote, fake_test = self.worker(base / "worker")
            a = self.make_cell(base, "pilot-a", '{"candidates":"a"}')
            b = self.make_cell(base, "pilot-b", '{"candidates":"b"}')
            pods = ["pod-0", "pod-1"]
            root_a, root_b = remote_root_for(a), remote_root_for(b)
            names = ["remote-identity.json", "gpu-manifest.json"]
            with patch("benchmark_pair.remote", side_effect=fake_remote):
                stage(pods, a, root_a, names)
                stage(pods, a, root_a, names)
                stage(pods, b, root_b, names)
                (a / "gpu-manifest.json").write_text('{"candidates":"changed"}')
                with self.assertRaises(subprocess.CalledProcessError):
                    stage(pods, a, root_a, names)
            for node in NODES:
                for prefix in ("observed-", "timings-"):
                    inventory = Path(mapped(root_a)) / (prefix + node) / "inventory.json"
                    inventory.parent.mkdir(parents=True)
                    inventory.write_text("{}")
            with patch("benchmark_pair.subprocess.run", side_effect=fake_test):
                for phase in ("probe", "timing"):
                    self.assertTrue(observe_available(pods, root_a, phase))
                    self.assertFalse(observe_available(pods, root_b, phase))


class SettledJobState(unittest.TestCase):
    def run_states(self, states, deadline=120):
        outputs = iter(states)
        now = [0.0]
        def fake_run(command, **kwargs):
            self.assertEqual(command[:3], ["scontrol", "show", "job"])
            state, code = next(outputs)
            return SimpleNamespace(returncode=0, stdout=f"JobId=7 JobState={state} ExitCode={code}\n")
        def fake_sleep(seconds):
            now[0] += seconds
        with patch("pilot.subprocess.run", side_effect=fake_run):
            return settled_job_state("7", deadline, 2, clock=lambda: now[0], sleep=fake_sleep)

    def test_completing_then_completed_is_accepted(self):
        raw, observed = self.run_states([("COMPLETING", "0:0"), ("COMPLETING", "0:0"), ("COMPLETED", "0:0")])
        self.assertIn("JobState=COMPLETED", raw)
        self.assertEqual([o["state"] for o in observed], ["COMPLETING", "COMPLETING", "COMPLETED"])

    def test_terminal_failure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "raw_state=FAILED raw_exit=1:0"):
            self.run_states([("COMPLETING", "1:0"), ("FAILED", "1:0")])

    def test_completed_with_nonzero_exit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "raw_exit=0:9"):
            self.run_states([("COMPLETED", "0:9")])

    def test_persistent_completing_times_out(self):
        with self.assertRaisesRegex(ValueError, "raw_state=COMPLETING"):
            self.run_states([("COMPLETING", "0:0")] * 100, deadline=10)


class FocusedSearchSpace(unittest.TestCase):
    def test_deterministic_cells_stay_unique_inside_focused_region(self):
        from cycle import (deterministic_plan, FOCUSED_LANE_STRIDES, FOCUSED_UNROLLS,
                           FOCUSED_LAYOUTS, FOCUSED_LOADS, proposal_template)
        base = proposal_template("c" * 64)
        for round_id in range(3):
            plans = [deterministic_plan(base, i, round_id) for i in range(32)]
            self.assertEqual(len({json.dumps(p, sort_keys=True) for p in plans}), 32)
            for p in plans:
                self.assertIn(p["lane_stride"], FOCUSED_LANE_STRIDES)
                self.assertIn(p["unroll"], FOCUSED_UNROLLS)
                self.assertIn(p["layout"], FOCUSED_LAYOUTS)
                self.assertIn(p["load"], FOCUSED_LOADS)
                self.assertTrue(0 <= p["lane_offset"] < 16)
                self.assertEqual({k: p[k] for k in base if k not in
                                  ("lane_stride", "lane_offset", "load", "layout", "unroll")},
                                 {k: base[k] for k in base if k not in
                                  ("lane_stride", "lane_offset", "load", "layout", "unroll")})

    def test_focused_region_is_inside_admitted_grammar(self):
        from cycle import FOCUSED_LANE_STRIDES, FOCUSED_UNROLLS
        self.assertTrue(all(s % 2 == 1 and 1 <= s <= 15 for s in FOCUSED_LANE_STRIDES))
        self.assertTrue(set(FOCUSED_UNROLLS) <= {1, 2, 4, 8, 16})

    def test_prompts_state_the_focused_region(self):
        from cycle import LOWERING_PROMPT, OPERATOR_PROMPT, FOCUSED_REGION_TEXT
        for prompt in (LOWERING_PROMPT, OPERATOR_PROMPT):
            self.assertIn(FOCUSED_REGION_TEXT, prompt)
            self.assertNotIn("unroll in [1,2,4,8,16]", prompt)

    def test_promotion_threshold_and_blocks_are_unchanged(self):
        text = (HERE / "pilot.py").read_text()
        self.assertIn("promotion_threshold_percent=5", text)
        self.assertIn("benchmark_blocks_per_node=30", text)
        bench = (HERE / "gpu_benchmark.py").read_text()
        self.assertIn('manifest["blocks"]==30', bench)

    def test_every_timed_replay_follows_an_untimed_warm_replay(self):
        bench = (HERE / "gpu_benchmark.py").read_text()
        self.assertIn("WARM_REPLAYS_PER_MEASUREMENT=1", bench)
        warm = bench.index("for _ in range(WARM_REPLAYS_PER_MEASUREMENT):graph.replay()")
        timed = bench.index("start.record();graph.replay();end.record();end.synchronize()")
        self.assertLess(warm, timed)
        self.assertNotIn("end.synchronize()", bench[warm:timed])


if __name__ == "__main__":
    unittest.main()
