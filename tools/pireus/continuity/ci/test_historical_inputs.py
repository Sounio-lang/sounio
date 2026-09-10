import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("inputs", Path(__file__).with_name("prepare_historical_inputs.py"))
INPUTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INPUTS)


class HistoricalInputs(unittest.TestCase):
    def test_corrupt_existing_input_is_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "input")
            path.write_bytes(b"corrupted")
            item = dict(path=str(path), url="https://invalid.example/never-requested",
                        bytes=4, sha256=hashlib.sha256(b"good").hexdigest())
            with patch.object(INPUTS, "INPUTS", [item]), patch.object(INPUTS.urllib.request, "urlopen") as fetch:
                with self.assertRaisesRegex(ValueError, "existing historical input differs"):
                    INPUTS.prepare()
                fetch.assert_not_called()
            self.assertEqual(path.read_bytes(), b"corrupted")

    def test_corrupt_download_is_never_installed(self):
        import io
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "input")
            item = dict(path=str(path), url="https://invalid.example/control",
                        bytes=4, sha256=hashlib.sha256(b"good").hexdigest())
            with patch.object(INPUTS, "INPUTS", [item]), patch.object(
                    INPUTS.urllib.request, "urlopen", return_value=io.BytesIO(b"evil")):
                with self.assertRaisesRegex(ValueError, "downloaded historical input differs"):
                    INPUTS.prepare()
            self.assertFalse(path.exists())

    def test_real_pinned_downloads_and_parent_custody(self):
        with tempfile.TemporaryDirectory() as directory:
            items = copy.deepcopy(INPUTS.INPUTS)
            for index, item in enumerate(items):
                item["path"] = str(Path(directory, str(index)))
            with patch.object(INPUTS, "INPUTS", items):
                report = INPUTS.prepare(verify_history=True)
                self.assertEqual(report["historical_parents"], 5)
                self.assertTrue(report["parent_git_history_verified"])
                self.assertFalse(report["current_hardware_acceptance"])


if __name__ == "__main__":
    unittest.main()
