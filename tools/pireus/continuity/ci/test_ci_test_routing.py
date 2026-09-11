import json
import os
from pathlib import Path
import re
import subprocess
import unittest
from route_slow_tests import ROOT, MANIFEST, partition


class RequiredSlowWitnesses(unittest.TestCase):
    def test_partition_and_matrix_cover_same_tests(self):
        spec = json.loads(MANIFEST.read_text())
        slow = [row["path"] for row in spec["tests"]]
        ordinary = "tests/stdlib/crypto/test_sha256.sio"
        selected, routed = partition([ordinary, *slow])
        self.assertEqual(selected, [ordinary])
        self.assertEqual(set(routed), set(slow))
        for paths in [[ordinary, *slow[:-1]], [ordinary, *slow, slow[0]]]:
            with self.assertRaises(ValueError):
                partition(paths)
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        job = workflow.split("  pireus-slow-stdlib:\n")[1].split("  full-test-suite:\n")[0]
        matrix = re.findall(r"^          - (test_pireus_\w+\.sio)$", job, re.M)
        self.assertEqual(sorted(matrix), sorted(Path(p).name for p in slow))

    def test_composite_ci_refuses_failed_missing_or_skipped_slow_job(self):
        names = ["contracts", "native-selfhost-linux-x86_64",
                 "source-bootstrap-selfhost-linux-x86_64", "madaros-current-source-deref-f64",
                 "native-selfhost-macos-arm64", "full-test-suite", "pireus-slow-stdlib",
                 "madaros-witness-gate", "gate-wave-0", "sounio-lint", "lean-proofs", "website"]
        needs = {name: dict(result="success") for name in names}
        needs["impact"] = dict(result="success", outputs=dict(stdlib="true"))
        def check():
            return subprocess.run(["python3", str(ROOT / "scripts/ci/evaluate_ci_decision.py")],
                                  env=dict(os.environ, NEEDS_JSON=json.dumps(needs)),
                                  capture_output=True, text=True, timeout=10)
        self.assertEqual(check().returncode, 0)
        for state in ["failure", "cancelled", "skipped", "missing"]:
            needs["pireus-slow-stdlib"] = dict(result=state)
            result = check()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pireus-slow-stdlib", result.stderr)


if __name__ == "__main__":
    unittest.main()
