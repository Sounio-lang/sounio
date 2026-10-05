import json
from pathlib import Path
import shutil
import tempfile
import unittest
from build_control_feedback import SOURCE, build


class FeedbackControls(unittest.TestCase):
    def test_real_archive_preserves_native_decisions(self):
        packet = build()
        self.assertEqual(len(packet["materials"]), 18)
        self.assertEqual(sum(len(m["occurrences"]) for m in packet["materials"]), 64)
        for material in packet["materials"]:
            original = json.loads((SOURCE / (material["representative"] + ".gain.json")).read_text())
            self.assertEqual(material["native_gain"], original)
        self.assertFalse(packet["pilot_acceptance"])
        self.assertFalse(packet["automatic_blacklist"])

    def test_tampered_gain_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "archive"
            shutil.copytree(SOURCE, source)
            path = source / "000.gain.json"
            path.write_text(path.read_text().replace("NO_GAIN", "GAIN"))
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                build(source)

    def test_rehashed_audit_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "archive"
            shutil.copytree(SOURCE, source)
            path = source / "hardware-audit.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "audit identity mismatch"):
                build(source)


if __name__ == "__main__":
    unittest.main()
