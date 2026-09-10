#!/usr/bin/env python3
"""Corruption and traversal controls against the actual archived canary."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from verify_archive import verify_archive

REPO = Path(__file__).resolve().parents[4]
ARCHIVE = REPO / "tools/pireus/continuity/validation/real-inkling-cycle-20260907"

class ArchiveControls(unittest.TestCase):
    def test_real_archive(self):
        result = verify_archive(ARCHIVE, REPO)
        self.assertEqual(result["paired_responses_verified"], 8)
        self.assertEqual(result["originally_untracked_sources"], 1)
        self.assertFalse(result["source_snapshot_complete_at_original_commit"])
        self.assertFalse(result["runtime_rerun"])
        self.assertFalse(result["new_hardware_acceptance"])

    def test_corruption_refusals(self):
        def summary_change(root, change):
            path = root / "archive-summary.json"
            value = json.loads(path.read_bytes())
            change(value)
            path.write_text(json.dumps(value))
        cases = {
            "changed proposal": lambda root: (root / "000.proposal.json").write_text("{}"),
            "missing rank": lambda root: (root / "worker/rank-1-000.json").unlink(),
            "unlisted file": lambda root: (root / "extra.json").write_text("{}"),
            "parent traversal": lambda root: summary_change(root, lambda s: s["archive_files"][0].update(path="../outside")),
            "duplicate path": lambda root: summary_change(root, lambda s: s["archive_files"].append(s["archive_files"][0])),
            "false token count": lambda root: summary_change(root, lambda s: s.update(generated_tokens=1)),
            "false readiness": lambda root: summary_change(root, lambda s: s.update(claim_ready=True)),
            "wrong source": lambda root: summary_change(root, lambda s: s.update(source_commit="0"*40)),
            "symlink": lambda root: (root / "external").symlink_to(REPO / "AGENTS.md"),
        }
        for name, corrupt in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "archive"
                shutil.copytree(ARCHIVE, root)
                corrupt(root)
                with self.assertRaises((ValueError, FileNotFoundError, subprocess.CalledProcessError)):
                    verify_archive(root, REPO)


    def test_missing_or_forged_recovery_refuses(self):
        source = Path(__file__).with_name("source-recoveries.json")
        for mutation in ("missing", "wrong_hash"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                value = json.loads(source.read_bytes())
                if mutation == "missing":
                    value["sources"] = {}
                else:
                    value["sources"]["runtime/test_preflight_binding.py"]["sha256"] = "0"*64
                path = Path(directory) / "recovery.json"
                path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "unpreserved historical source"):
                    verify_archive(ARCHIVE, REPO, path)

if __name__ == "__main__":
    unittest.main()
