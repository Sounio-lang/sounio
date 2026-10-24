"""Negative controls for the independent terminal packet audit."""
import copy
import json
import pathlib
import tarfile
import tempfile
import unittest
import gen2_cpu_receipt as receipt


class TerminalPacketAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        r = self.root
        (r / "tree").mkdir()
        (r / "tree/source.sio").write_bytes(b"fn main() -> i32 { 0 }")
        with tarfile.open(r / "source.tar", "w") as archive:
            archive.add(r / "tree/source.sio", arcname="source.sio")
        for name in ("madaros", "measure.py", "compile.log", "samples.jsonl", "madaros.gen2"):
            (r / name).write_bytes(name.encode())
        protocol = {"files_sha256": {n: receipt.sha(r / n) for n in ("source.tar", "madaros")},
                    "node": "test-node", "cpus": 4, "slurm_memory_mib": 36864, "source": "test-source"}
        self.write("protocol.json", protocol)
        self.expected = {"protocol_sha256": receipt.sha(r / "protocol.json"),
                         "runner_sha256": receipt.sha(r / "measure.py"),
                         "job": "11991", "boot_id": "test-boot"}
        self.entry = {**self.expected, "hostname": "test-node", "allocation": {
            "SLURM_JOB_ID": "11991", "SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_NODE": "36864"}}
        self.write("attempt-entered.json", self.entry)
        self.result = {"job": "11991", "compiler_rc": 0, "timed_out": False,
                       "boot_unchanged": True, "source_files_unchanged": True,
                       "compiler_unchanged": True, "artifact_exists": True,
                       "artifact_sha256": receipt.sha(r / "madaros.gen2"), "compile_complete": True,
                       "ci_qualified": False, "inkling_qualified": False, "causal_claim": False}
        self.write("result.json", self.result)

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data))

    def audit(self):
        return receipt.audit(self.root, self.expected)

    def test_success_does_not_promote_scheduler_banner_ci_or_inkling(self):
        result = self.audit()
        self.assertTrue(result["compile_complete"])
        for key in ("scheduler_terminal_verified", "banner_verified", "ci_qualified", "inkling_qualified", "causal_claim"):
            self.assertIs(result[key], False)

    def test_missing_terminal_is_not_completion(self):
        (self.root / "result.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.audit()

    def test_changed_source_rejected_despite_runner_true(self):
        (self.root / "tree/source.sio").write_text("changed")
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            self.audit()

    def test_added_source_rejected(self):
        (self.root / "tree/extra.sio").write_text("extra")
        with self.assertRaisesRegex(ValueError, "source inventory"):
            self.audit()

    def test_changed_artifact_rejected(self):
        (self.root / "madaros.gen2").write_text("changed")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            self.audit()

    def test_wrong_job_and_boot_rejected(self):
        for key in ("job", "boot_id"):
            changed = {**self.entry, key: "wrong"}
            self.write("attempt-entered.json", changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "attempt identity"):
                self.audit()

    def test_wrong_allocation_rejected(self):
        changed = copy.deepcopy(self.entry)
        changed["allocation"]["SLURM_MEM_PER_NODE"] = "16384"
        self.write("attempt-entered.json", changed)
        with self.assertRaisesRegex(ValueError, "memory allocation"):
            self.audit()

    def test_failed_and_timeout_results_preserved_without_completion(self):
        for rc, timed_out in ((7, False), (-15, True)):
            self.write("result.json", {**self.result, "compiler_rc": rc,
                                      "timed_out": timed_out, "compile_complete": False})
            audited = self.audit()
            self.assertEqual(audited["compiler_rc"], rc)
            self.assertIs(audited["compile_complete"], False)

    def test_empty_output_with_zero_exit_rejected(self):
        (self.root / "madaros.gen2").write_bytes(b"")
        self.write("result.json", {**self.result,
                   "artifact_sha256": receipt.sha(self.root / "madaros.gen2"),
                   "compile_complete": False})
        with self.assertRaisesRegex(ValueError, "exit zero"):
            self.audit()

    def test_promoted_claim_rejected(self):
        self.write("result.json", {**self.result, "ci_qualified": True})
        with self.assertRaisesRegex(ValueError, "unsupported promotion"):
            self.audit()

    def test_changed_runner_and_protocol_rejected(self):
        for name in ("measure.py", "protocol.json"):
            original = (self.root / name).read_bytes()
            (self.root / name).write_bytes(original + b" ")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "mismatch"):
                self.audit()
            (self.root / name).write_bytes(original)


if __name__ == "__main__":
    unittest.main()
