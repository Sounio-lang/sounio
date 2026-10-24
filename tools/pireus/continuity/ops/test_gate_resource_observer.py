"""Behavioral controls for passive CI telemetry; no compiler/model execution."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / ".git").exists())
OBSERVER = ROOT / "scripts/lib/gate_process_observer.sh"


class ResourceObserverTests(unittest.TestCase):
    def invoke(self, command, setup="", interval="1"):
        with tempfile.TemporaryDirectory() as directory:
            log = pathlib.Path(directory) / "command.log"
            result = subprocess.run(
                ["bash", "-c",
                 'set -euo pipefail; source "$1"; ' + setup +
                 '; gate_observe_command control "$2" bash -c "$3"',
                 "test", str(OBSERVER), str(log), command],
                env={**__import__("os").environ, "SOUNIO_GATE_OBSERVE_SECONDS": interval},
                capture_output=True, timeout=10,
            )
            return result, log.read_bytes() if log.exists() else None

    def test_success_and_exact_command_log(self):
        result, log = self.invoke("printf 'out\n'; printf 'err\n' >&2", ":")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(log, b"out\nerr\n")
        self.assertIn(b"GATE_RESOURCE label=control phase=start", result.stdout)
        self.assertIn(b"GATE_RESOURCE label=control phase=end", result.stdout)

    def test_failure_is_preserved(self):
        result, log = self.invoke("printf failure; exit 7", ":")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(log, b"failure")
        self.assertIn(b"command_rc=7", result.stdout)

    def test_telemetry_failure_cannot_fail_command(self):
        result, log = self.invoke("printf survived", "gate_observe_resources() { return 73; }")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(log, b"survived")

    def test_periodic_sampling_and_no_retry(self):
        result, log = self.invoke("printf once; sleep 2", ":")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(log, b"once")
        self.assertIn(b"GATE_RESOURCE label=control phase=sample", result.stdout)

    def test_invalid_interval_does_not_start_command(self):
        result, log = self.invoke("printf forbidden", ":", interval="0")
        self.assertEqual(result.returncode, 64)
        self.assertIsNone(log)


if __name__ == "__main__":
    unittest.main()
