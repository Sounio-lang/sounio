"""Harness controls only: no scientific output or compiler is validated here."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROBES = (
    "rep_traj_bug", "rep_stagnation", "rep_adiabatic_bug",
    "gbs_oracle", "h2_ignition_uq_demo",
)


class ChemistrySelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for directory in ("scripts/ci", "bin", "benchmarks/chemistry/golden", "stdlib"):
            (self.root / directory).mkdir(parents=True)
        self.script = self.root / "scripts/ci/chemistry_probe_golden_gate.sh"
        shutil.copyfile(ROOT / "scripts/ci/chemistry_probe_golden_gate.sh", self.script)
        for probe in PROBES:
            (self.root / f"benchmarks/chemistry/golden/{probe}.lean_single.txt").write_text(probe + "\n")
        (self.root / "bin/souc-lean-single-x86_64").write_text("fixture, never executed\n")
        compiler = self.root / "bin/compiler-fixture"
        compiler.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
assert sys.argv[1] == "run"
assert os.environ["SOUNIO_SOUC_ENGINE"] == "lean_single"
probe = Path(sys.argv[2]).stem
with open(os.environ["CALLS"], "a") as f:
    f.write(probe + "\\n")
if probe == os.environ.get("FAIL_PROBE"):
    sys.exit(124)
print("different output" if probe == os.environ.get("DIFF_PROBE") else probe)
''')
        compiler.chmod(0o755)
        # Only stub the time wrapper; verify its unchanged production deadline.
        timeout = self.root / "bin/timeout"
        timeout.write_text('#!/bin/bash\n[[ "$1" == 1500 ]] || exit 99\nshift\nexec "$@"\n')
        timeout.chmod(0o755)
        self.calls = self.root / "calls"
        self.env = dict(os.environ, SOUC=str(compiler), CALLS=str(self.calls),
                        PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"])
        self.env.pop("REGEN", None)

    def run_gate(self, *args, **env):
        return subprocess.run(["bash", str(self.script), *args],
                              env=dict(self.env, **env), capture_output=True, text=True)

    def invoked(self):
        return self.calls.read_text().splitlines() if self.calls.exists() else []

    def test_default_runs_all_oracles_even_after_failure(self):
        result = self.run_gate(FAIL_PROBE=PROBES[2])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.invoked(), list(PROBES))
        self.assertIn("run exited 124", result.stderr)

    def test_each_selected_probe_runs_exactly_once(self):
        for probe in PROBES:
            with self.subTest(probe=probe):
                self.calls.unlink(missing_ok=True)
                result = self.run_gate(probe)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.invoked(), [probe])

    def test_unknown_empty_or_extra_selection_fails_before_execution(self):
        for args in (("unknown",), ("",), (PROBES[0], PROBES[1])):
            with self.subTest(args=args):
                self.assertEqual(self.run_gate(*args).returncode, 2)
                self.assertEqual(self.invoked(), [])

    def test_selected_failure_and_golden_mismatch_fail_closed(self):
        self.assertNotEqual(self.run_gate(PROBES[0], FAIL_PROBE=PROBES[0]).returncode, 0)
        self.assertNotEqual(self.run_gate(PROBES[0], DIFF_PROBE=PROBES[0]).returncode, 0)
        (self.root / f"benchmarks/chemistry/golden/{PROBES[0]}.lean_single.txt").unlink()
        self.assertNotEqual(self.run_gate(PROBES[0]).returncode, 0)

    def test_workflow_matrix_covers_every_oracle_and_retains_failures(self):
        workflow = (ROOT / ".github/workflows/chemistry-probe-golden.yml").read_text()
        block = re.search(r"(?m)^        probe:\n((?:          - [a-z0-9_]+\n)+)", workflow)
        self.assertIsNotNone(block)
        self.assertEqual(re.findall(r"- ([a-z0-9_]+)", block[1]), list(PROBES))
        self.assertIn("fail-fast: false", workflow)
        self.assertNotIn("continue-on-error:", workflow)
        self.assertNotRegex(workflow, r"(?m)^  (pull_request|push|schedule):")
        self.assertIn("  workflow_call:", workflow)
        self.assertIn("  workflow_dispatch:", workflow)
        self.assertIn('bash scripts/ci/chemistry_probe_golden_gate.sh "$PROBE"', workflow)
        self.assertIn('PROBE: ${{ matrix.probe }}', workflow)
        for probe in PROBES:
            self.assertTrue((ROOT / f"benchmarks/chemistry/golden/{probe}.lean_single.txt").is_file())


if __name__ == "__main__":
    unittest.main()
