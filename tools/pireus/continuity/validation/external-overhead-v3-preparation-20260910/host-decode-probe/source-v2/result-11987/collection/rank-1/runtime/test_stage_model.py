"""Transport regression controls: an early EOF is resumable, corruption is refused."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import stage_model

class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}

class SnapshotTransportTests(unittest.TestCase):
    def entry(self, payload):
        return {"rfilename": "weights.bin", "size": len(payload),
                "lfs": {"sha256": hashlib.sha256(payload).hexdigest()}}

    def test_early_eof_resumes_exact_remaining_range(self):
        payload = b"abcdefghi"
        calls = []
        def open_response(request, timeout):
            calls.append(request.get_header("Range"))
            if len(calls) == 1:
                return Response(payload[:3])
            self.assertEqual(request.get_header("Range"), "bytes=3-")
            return Response(payload[3:], 206, {"Content-Range": "bytes 3-8/9"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(stage_model.urllib.request, "urlopen", open_response), patch.object(stage_model.time, "sleep"):
                receipt = stage_model.fetch(root, self.entry(payload))
            self.assertEqual((root / "weights.bin").read_bytes(), payload)
            self.assertEqual(receipt["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(calls, [None, "bytes=3-"])
            self.assertFalse((root / "weights.bin.partial").exists())

    def test_corruption_is_retained_without_completion(self):
        payload = b"abcdefghi"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(stage_model.urllib.request, "urlopen", return_value=Response(b"XXXXXXXXX")):
                with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                    stage_model.fetch(root, self.entry(payload))
            self.assertFalse((root / "weights.bin").exists())
            self.assertEqual((root / "weights.bin.partial").read_bytes(), b"XXXXXXXXX")

    def test_existing_completed_corruption_is_not_overwritten(self):
        payload = b"abcdefghi"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.bin").write_bytes(b"XXXXXXXXX")
            with self.assertRaisesRegex(ValueError, "existing completed file failed integrity"):
                stage_model.fetch(root, self.entry(payload))
            self.assertEqual((root / "weights.bin").read_bytes(), b"XXXXXXXXX")

    def test_ignored_resume_range_is_refused(self):
        payload = b"abcdefghi"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.bin.partial").write_bytes(payload[:3])
            with patch.object(stage_model.urllib.request, "urlopen", return_value=Response(payload, 200)):
                with self.assertRaisesRegex(ValueError, "did not honor resume"):
                    stage_model.fetch(root, self.entry(payload))
            self.assertEqual((root / "weights.bin.partial").read_bytes(), payload[:3])

if __name__ == "__main__":
    unittest.main()
