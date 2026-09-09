import json
from pathlib import Path
import tempfile
import unittest
import shutil
from check_frozen_feedback import inputs, source_checks, HERE

CANONICAL = HERE / "validation/feedback-smoke-freeze-20260909/execution-freeze.json"
SPEC = json.loads(CANONICAL.read_bytes())


def checks():
    return [dict(name=name, head_sha=SPEC["source_commit"], id=i,
                 started_at="2026-09-09T00:00:00Z", status="completed", conclusion="success")
            for i, name in enumerate(SPEC["required_source_checks"])]


class ReadinessControls(unittest.TestCase):
    def test_all_exact_source_checks_required(self):
        self.assertEqual(len(source_checks(SPEC, checks())), 3)
        with self.assertRaisesRegex(ValueError, "missing"):
            source_checks(SPEC, checks()[:-1])

    def test_green_other_revision_refused(self):
        rows = checks()
        rows[0]["head_sha"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "missing"):
            source_checks(SPEC, rows)

    def test_newer_failure_or_running_check_beats_old_green(self):
        for status, conclusion in (("completed", "failure"), ("in_progress", None)):
            rows = checks()
            rows.append(dict(rows[0], id=999, started_at="2026-09-09T01:00:00Z",
                             status=status, conclusion=conclusion))
            with self.assertRaisesRegex(ValueError, "not green"):
                source_checks(SPEC, rows)

    def test_freeze_mutation_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "execution-freeze.json").write_bytes(CANONICAL.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "freeze identity"):
                inputs(root)

    def test_real_inputs_pass_and_changed_bundle_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "attempt"
            shutil.copytree(CANONICAL.parent, root)
            shutil.copytree(HERE / "runtime", root / "runtime",
                            ignore=shutil.ignore_patterns("__pycache__"))
            self.assertEqual(inputs(root)["source_commit"], SPEC["source_commit"])
            bundle = root / "with-feedback/offline-bundle.json"
            bundle.write_bytes(bundle.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "frozen artifact"):
                inputs(root)

    def test_runtime_mutation_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "execution-freeze.json").write_bytes(CANONICAL.read_bytes())
            (root / "runtime").mkdir()
            first = next(iter(SPEC["runtime_sha256"]))
            (root / "runtime" / first).write_bytes(b"changed runtime")
            with self.assertRaisesRegex(ValueError, "frozen artifact"):
                inputs(root)


if __name__ == "__main__":
    unittest.main()
