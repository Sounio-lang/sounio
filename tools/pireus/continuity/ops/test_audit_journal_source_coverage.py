import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from audit_journal_source_coverage import audit

class SourceCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.name = "tools/pireus/continuity/ops/adapter.py"
        self.file = self.repo / self.name
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(b"original\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
        self.source = self.git("rev-parse", "HEAD").strip()
        self.expected = {self.name: hashlib.sha256(b"original\n").hexdigest()}

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True)

    def test_committed_bytes_pass_without_promoting_ci(self):
        result = audit(self.repo, self.source, self.expected)
        self.assertTrue(result["orchestration_content_verified"])
        for key in ("source_ci_checked", "source_qualified", "inference_submitted", "protocol_rebound"):
            self.assertFalse(result[key])

    def test_worktree_repair_cannot_repair_old_commit(self):
        self.file.write_bytes(b"replacement\n")
        expected = {self.name: hashlib.sha256(self.file.read_bytes()).hexdigest()}
        result = audit(self.repo, self.source, expected)
        self.assertFalse(result["orchestration_content_verified"])
        self.assertTrue(result["files"][0]["present"])

    def test_missing_source_file_is_not_worktree_presence(self):
        name = "tools/pireus/continuity/ops/new_adapter.py"
        (self.repo / name).write_bytes(b"new")
        result = audit(self.repo, self.source, {name: hashlib.sha256(b"new").hexdigest()})
        self.assertFalse(result["orchestration_content_verified"])
        self.assertFalse(result["files"][0]["present"])

    def test_noncommit_and_unbounded_inventory_rejected(self):
        for source, expected in [
            ("HEAD", self.expected), ("0" * 40, self.expected),
            (self.source, {}), (self.source, {"../escape": "0" * 64}),
            (self.source, {self.name: "not-a-digest"})]:
            with self.subTest(source=source, expected=expected):
                with self.assertRaises(ValueError):
                    audit(self.repo, source, expected)

if __name__ == "__main__":
    unittest.main()
