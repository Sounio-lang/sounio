#!/usr/bin/env python3
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from memory_guard import supervise

class MemoryGuardTests(unittest.TestCase):
    def quiet(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return supervise(*args, **kwargs)

    def test_cli_cannot_lower_guard_to_protected_floor(self):
        env = os.environ | {"SLURM_JOB_ID": "test", "SLURM_PROCID": "0"}
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("memory_guard.py")),
                                 "--reserve-gib", "32", "--", "/bin/true"],
                                env=env, capture_output=True)
        self.assertEqual(result.returncode, 2)

    def test_child_status(self):
        self.assertEqual(self.quiet([sys.executable, "-c", "raise SystemExit(7)"],
                                   read_memory=lambda: 100, reserve=50), 7)

    def test_refuses_before_spawn(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / "spawned"
            self.assertEqual(self.quiet([sys.executable, "-c",
                "from pathlib import Path;Path(__import__('sys').argv[1]).touch()", str(marker)],
                read_memory=lambda: 49, reserve=50), 75)
            self.assertFalse(marker.exists())

    def test_stops_only_owned_process_group(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"],
                                     start_new_session=True)
        try:
            with tempfile.TemporaryDirectory() as root:
                pidfile = Path(root) / "pid"
                code = "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(30)"
                def memory():
                    return 49 if pidfile.exists() else 100
                self.assertEqual(self.quiet([sys.executable, "-c", code, str(pidfile)],
                                           read_memory=memory, reserve=50), 75)
                self.assertIsNone(unrelated.poll())
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pidfile.read_text()), 0)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_observation_error_kills_child(self):
        with tempfile.TemporaryDirectory() as root:
            pidfile = Path(root) / "pid"
            code = "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(30)"
            def memory():
                if pidfile.exists():
                    raise RuntimeError("missing observation")
                return 100
            with self.assertRaisesRegex(RuntimeError, "missing observation"):
                self.quiet([sys.executable, "-c", code, str(pidfile)],
                           read_memory=memory, reserve=50)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pidfile.read_text()), 0)

if __name__ == "__main__":
    unittest.main()
