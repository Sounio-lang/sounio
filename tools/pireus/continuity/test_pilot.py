import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from benchmark_pair import deduplicate_candidates
from cycle import HERE, encoded, digest
from pilot import cells, paired_completion, launch_cell_stage, validate_checks, REQUIRED_CHECKS, summarize
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


if __name__ == "__main__":
    unittest.main()
