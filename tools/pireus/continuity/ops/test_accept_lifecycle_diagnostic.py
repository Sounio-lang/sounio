#!/usr/bin/env python3
"""Offline custody controls; modified historical copies are synthetic fixtures."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from accept_lifecycle_diagnostic import accept
from freeze_lifecycle_diagnostic import create, GENERATED_SHA
from feedback_smoke import HERE

class CustodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.frozen = cls.root/"frozen"
        create(cls.frozen)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.case = Path(tempfile.mkdtemp(dir=self.root))
        self.worker = self.case/"worker"
        shutil.copytree(HERE/"validation/feedback-smoke-inference-20260909/without-feedback-11969/worker-receipts", self.worker)

    def synthetic_diagnostic_receipts(self):
        # Copies only: do not rewrite the immutable evidence or claim GPU execution.
        for rank in (0, 1):
            p = self.worker/f"rank-{rank}-complete.json"
            r = json.loads(p.read_bytes())
            r["helper_sha256"] = GENERATED_SHA
            p.write_text(json.dumps(r))

    def run_accept(self, name="accepted", job="11969"):
        return accept(self.frozen, "without-feedback", self.worker, self.case/name, job)

    def test_old_runtime_receipts_refused(self):
        with self.assertRaisesRegex(ValueError, "receipt identity"):
            self.run_accept()
        self.assertFalse((self.case/"accepted/token-custody.json").exists())

    def test_declared_runtime_and_no_hardware_promotion(self):
        self.synthetic_diagnostic_receipts()
        r = self.run_accept()
        self.assertTrue(r["complete_token_receipts_valid"])
        self.assertFalse(r["hardware_qualified"])
        self.assertFalse(r["pilot_acceptance"])
        with self.assertRaises(FileExistsError):
            self.run_accept()

    def test_wrong_job_and_rank_disagreement_refused(self):
        self.synthetic_diagnostic_receipts()
        with self.assertRaisesRegex(ValueError, "job/stage"):
            self.run_accept(job="wrong-job")
        p = self.worker/"rank-1-000.json"
        r = json.loads(p.read_bytes())
        r["output_ids"] = [123]
        p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError, "two-rank token response disagreement"):
            self.run_accept()
        self.assertFalse((self.case/"accepted/token-custody.json").exists())

if __name__ == "__main__":
    unittest.main()
