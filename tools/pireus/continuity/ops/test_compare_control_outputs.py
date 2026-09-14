import json
from pathlib import Path
import tempfile
import unittest

from compare_control_outputs import compare, sha


def write(path, value):
    path.write_text(json.dumps(value))


class ComparisonControls(unittest.TestCase):
    def exercise(self, token_drift=False, profile_drift=False, failed=False, tamper=False):
        with tempfile.TemporaryDirectory() as d:
            ref, control = Path(d) / "reference", Path(d) / "control"
            ref.mkdir()
            control.mkdir()
            rm = dict(input_sha256="input", original_runtime_sha256="original",
                      runtime_files={"offline_generate.py": "instrumented", "guard.py": "guard"})
            write(ref / "manifest.json", rm)
            cm = dict(input_sha256="input", instrumentation=False, parent_diagnostic_job=11957,
                      parent_manifest_sha256=sha((ref / "manifest.json").read_bytes()),
                      runtime_sha256="original",
                      runtime_files={"offline_generate.py": "original", "guard.py": "guard"})
            write(control / "manifest.json", cm)
            for root, job in ((ref, 11957), (control, 11958)):
                for rank in (0, 1):
                    directory = root / f"rank-{rank}"
                    directory.mkdir()
                    for i in range(32):
                        changed = root == control and i == 0
                        row = dict(job=str(job), index=i, input_sha256="input",
                                   output_ids=[2 if changed and token_drift else 1], completion_tokens=1,
                                   finish_reason="stop", execution_profile={"guard": 34 if changed and profile_drift else 33})
                        write(directory / f"offline-{job}-{rank}-{i:03d}.json", row)
                write(root / "summary.json", dict(job=job,
                    diagnostic_batch_complete=not (root == control and failed),
                    issues=["failed"] if root == control and failed else [],
                    file_hashes={str(p.relative_to(root)): sha(p.read_bytes())
                                 for p in root.rglob("*") if p.is_file()}))
            if tamper:
                (control / "rank-0/offline-11958-0-000.json").write_text("{}")
            return compare(ref, control)

    def test_equal_outputs_do_not_promote_pilot(self):
        r = self.exercise()
        self.assertTrue(r["complete"])
        self.assertTrue(r["output_ids_equal"])
        self.assertTrue(r["execution_profiles_equal"])
        self.assertEqual(r["comparisons_per_rank"], {"0": 32, "1": 32})
        self.assertFalse(r["pilot_acceptance"])
        self.assertFalse(r["memory_root_cause_established"])

    def test_token_and_profile_changes_are_reported_separately(self):
        r = self.exercise(token_drift=True)
        self.assertFalse(r["output_ids_equal"])
        self.assertTrue(r["execution_profiles_equal"])
        r = self.exercise(profile_drift=True)
        self.assertTrue(r["output_ids_equal"])
        self.assertFalse(r["execution_profiles_equal"])

    def test_failed_control_is_not_compared_as_success(self):
        r = self.exercise(failed=True)
        self.assertFalse(r["complete"])
        self.assertIsNone(r["output_ids_equal"])

    def test_changed_receipt_refuses_comparison(self):
        with self.assertRaises(ValueError):
            self.exercise(tamper=True)


if __name__ == "__main__":
    unittest.main()
