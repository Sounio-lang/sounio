"""Execute the real runner and native wrapper against small Sounio programs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]


class RunnerArguments(unittest.TestCase):
    def test_literal_argv_and_exit_status_through_both_layers(self):
        with tempfile.TemporaryDirectory(prefix="pireus-argv-") as directory:
            root = Path(directory)
            for name in ("scripts/dev/run_sio_test_suite.sh",
                         "scripts/ci/souc-native-wrapper.sh",
                         "scripts/lib/resolve_souc.sh"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            tests = root / "tests/run-pass"
            tests.mkdir(parents=True)
            (root / "bin").mkdir()
            compiler = ROOT / "bin/souc-lean-single-x86_64"
            self.assertTrue(compiler.is_file(), "required real bootstrap compiler")
            env = dict(os.environ, SOUNIO_TEST_SOUC_BIN=str(compiler),
                       SOUNIO_SOUC_BIN=str(compiler), SOUNIO_SOUC_RAW_MODE="legacy")
            fixture = tests / "argv.sio"
            fixture.write_text('''//@ run-pass
fn main() -> i32 with IO {
    if arg_count() != 5 { return 31 }
    if get_arg(0) != "with spaces" { return 32 }
    if get_arg(1) != "" { return 33 }
    if get_arg(2) != "*.sio" { return 34 }
    if get_arg(3) != "$(touch SHOULD_NOT_EXIST)" { return 35 }
    if get_arg(4) != "--flag" { return 36 }
    0
}
''')
            sidecar = fixture.with_suffix(".sio.args")
            sidecar.write_text("with spaces\n\n*.sio\n$(touch SHOULD_NOT_EXIST)\n--flag\n")
            command = ["bash", str(root / "scripts/dev/run_sio_test_suite.sh"),
                       "--filter-exact", "argv.sio", "--jobs", "1"]
            run = subprocess.run(command, env=env, cwd=root, capture_output=True,
                                 text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("Pass: 1", run.stdout)
            self.assertFalse((root / "SHOULD_NOT_EXIST").exists())
            sidecar.write_text("wrong\n\n*.sio\n$(touch SHOULD_NOT_EXIST)\n--flag\n")
            run = subprocess.run(command, env=env, cwd=root, capture_output=True,
                                 text=True, timeout=60)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("run exited 32", run.stdout)
            deadline = fixture.with_suffix(".sio.timeout")
            deadline.write_text("not-a-timeout\n")
            run = subprocess.run(command, env=env, cwd=root, capture_output=True,
                                 text=True, timeout=60)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("invalid timeout sidecar", run.stdout)
            deadline.unlink()
            sidecar.unlink()
            fixture.write_text("//@ run-pass\nfn main() -> i32 { 0 }\n")
            run = subprocess.run(command, env=env, cwd=root, capture_output=True,
                                 text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            run = subprocess.run(
                ["bash", str(root / "scripts/ci/souc-native-wrapper.sh"), "run"],
                env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 2)


if __name__ == "__main__":
    unittest.main()
