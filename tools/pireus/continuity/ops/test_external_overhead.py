#!/usr/bin/env python3
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import freeze_external_overhead as f
from check_frozen_feedback import source_checks

class FreezeTests(unittest.TestCase):
    def test_full_runtime_and_workload_scope(self):
        s=f.specification()
        self.assertEqual(len(s["runtime_sha256"]["baseline"]),101)
        self.assertEqual(len(s["runtime_sha256"]["observed"]),104)
        self.assertEqual(len(s["files_sha256"]),10)
        self.assertEqual(s["required_execution_profile"]["max_total_tokens"],6144)
        self.assertEqual(s["required_execution_profile"]["early_stop_gib"],33)
        self.assertFalse(s["operations"]["automatic_retry"])
    def test_runtime_mutation_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"frozen";f.create(p)
            target=p/"observed/runtime/memory_guard.py";target.chmod(0o644)
            target.write_bytes(target.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError,"artifact mismatch"):f.verify(p)
    def test_input_mutation_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"frozen";f.create(p)
            target=p/"without-feedback/offline-bundle.json";target.chmod(0o644)
            target.write_text("{}")
            with self.assertRaisesRegex(ValueError,"artifact mismatch"):f.verify(p)
    def test_old_source_ci_refused(self):
        s=f.specification()
        checks=[dict(name=n,head_sha=s["parent_source_commit"],id=i,status="completed",conclusion="success",started_at="2026-09-09")
                for i,n in enumerate(s["required_source_checks"])]
        with self.assertRaises(ValueError):source_checks(s,checks)
    def test_freeze_replacement_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"frozen";p.mkdir();(p/"execution-freeze.json").write_text("{}")
            with self.assertRaises(ValueError):f.verify(p)
if __name__=="__main__":unittest.main()
