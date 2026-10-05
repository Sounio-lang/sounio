import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from compare_first_request import events, unique
from prepare_decode_probe import prepare, SOURCE_SHA256, ANCHOR, REPLACEMENT


class DiagnosticControls(unittest.TestCase):
    def test_missing_duplicate_and_mixed_rank_evidence_refuse(self):
        row = {"stage": "OFFLINE_FIRST_TOKEN", "job": "11956", "rank": "0", "index": 0}
        self.assertEqual(unique([row], row["stage"], 0, 0), row)
        for rows in ([], [row, row]):
            with self.assertRaises(ValueError):
                unique(rows, row["stage"], 0, 0)
        with self.assertRaises(ValueError):
            unique([row], row["stage"], 1, 0)
        with self.assertRaises(ValueError):
            events(json.dumps(row).encode(), 11939)

    def test_probe_changes_only_declared_observations(self):
        path = Path(__file__).resolve().parents[1] / "runtime/offline_generate.py"
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SOURCE_SHA256)
        out = prepare(raw)
        self.assertEqual(out.decode().replace(REPLACEMENT, ANCHOR).encode(), raw)
        self.assertEqual(out.count(b"runner.decode(next_ids, batch)"), 1)
        with self.assertRaises(ValueError):
            prepare(raw + b"\n")

    def test_existing_output_cannot_be_replaced(self):
        import subprocess
        import sys
        source = Path(__file__).resolve().parents[1] / "runtime/offline_generate.py"
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "runtime.py"
            out.write_bytes(b"existing evidence")
            cmd = [sys.executable, str(Path(__file__).with_name("prepare_decode_probe.py")),
                   "--source", str(source), "--output", str(out)]
            result = subprocess.run(cmd, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(out.read_bytes(), b"existing evidence")


if __name__ == "__main__":
    unittest.main()
