"""Transport recovery controls; native semantics and frozen pilot code are unchanged."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))
import resume_pilot as recovery
from cycle import digest, encoded

ROW = "11955|COMPLETED|0:0|gpuorangefs-multi-spark-8e54,gpuorangefs-multi-spark-3c59|2026-09-07T22:59:01|2026-09-07T22:59:25\n"

class AccountingRecovery(unittest.TestCase):
    def test_exact_success_and_step_rows(self):
        self.assertEqual(recovery.completed_accounting(ROW + ROW.replace("11955|", "11955.0|"), "11955")[1], "COMPLETED")

    def test_incomplete_failed_duplicate_foreign_or_bad_time_refuses(self):
        for raw in ["", ROW + ROW, ROW.replace("COMPLETED", "COMPLETING"),
                    ROW.replace("0:0", "1:0"), ROW.replace("11955|", "11956|"),
                    ROW.replace("spark-3c59", "spark-other"),
                    ROW.replace("22:59:25", "22:58:25"),
                    ROW.replace("2026-09-07T22:59:25", "Unknown")]:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    recovery.completed_accounting(raw, "11955")

    def stage(self, root):
        folder = root / "pilot-stages" / "encode"
        folder.mkdir(parents=True)
        bundle = root / "encode-bundle.json"
        bundle.write_text("{}")
        intent = dict(command=[sys.executable, str(recovery.HERE / "runtime/launch_pair.py"),
                               "tokenize", "--minutes", "15", "--input-bundle", str(bundle)],
                      bundle_sha256=digest(bundle.read_bytes()))
        (folder / "intent.json").write_bytes(encoded(intent))
        values = [dict(mode="tokenize", command=["srun"], nodes=[dict(pod="a"), dict(pod="b")])]
        values.extend(dict(stage="TOKENIZER_TRANSPORT_PASS", rank=str(r), job="11955") for r in range(2))
        (folder / "launch.log").write_text("\n".join(json.dumps(v) for v in values))
        return folder, bundle

    def test_adopts_finished_stage_without_submitting_and_journals_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, _ = self.stage(root)
            original = (folder / "intent.json").read_bytes(), (folder / "launch.log").read_bytes()
            receipt = dict(source="Slurm accounting via controller sacct", stdout=ROW,
                           exit_code=0, helper_sha256=digest(recovery.HELPER.read_bytes()))
            with patch.object(recovery, "verify"), patch.object(recovery, "query_accounting", return_value=receipt), patch.object(recovery.subprocess, "run") as submit:
                recovery.recover_stage(folder)
                submit.assert_not_called()
            self.assertEqual(original, ((folder / "intent.json").read_bytes(), (folder / "launch.log").read_bytes()))
            result = json.loads((folder / "completed.json").read_text())
            self.assertEqual(result["job"], "11955")
            entries = [json.loads(line) for line in (root / "journal.jsonl").read_text().splitlines()]
            self.assertEqual(len(entries), 2)

    def test_mutated_input_or_partial_pair_never_queries_or_submits(self):
        for mutation in ["bundle", "log"]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                folder, bundle = self.stage(root)
                if mutation == "bundle":
                    bundle.write_text('{"changed":true}')
                else:
                    (folder / "launch.log").write_text("partial\n")
                with patch.object(recovery, "verify"), patch.object(recovery, "query_accounting") as query:
                    with self.assertRaises(ValueError):
                        recovery.recover_stage(folder)
                    query.assert_not_called()
                self.assertFalse((folder / "completed.json").exists())

if __name__ == "__main__":
    unittest.main()
