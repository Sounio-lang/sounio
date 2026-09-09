import json
from pathlib import Path
import tempfile
import unittest
from feedback_smoke import FEEDBACK, HERE, requests, stage

CONTEXT = (HERE / "validation/control-union-hardware-20260908/context.json").read_bytes()


class FeedbackSmokeControls(unittest.TestCase):
    def test_arms_have_identical_base_requests(self):
        arms, projection = requests(CONTEXT)
        self.assertEqual(len(projection["rows"]), 18)
        for a, b in zip(arms["without-feedback"], arms["with-feedback"]):
            self.assertEqual({k:v for k,v in a.items() if k != "messages"},
                             {k:v for k,v in b.items() if k != "messages"})
            self.assertTrue(b["messages"][0]["content"].startswith(
                a["messages"][0]["content"] + "\nMeasured feedback"))
        self.assertEqual([b["seed"] for b in arms["with-feedback"]], list(range(8)))

    def test_wrong_context_refused(self):
        with self.assertRaisesRegex(ValueError, "context mismatch"):
            requests(CONTEXT.replace(b'"precision":64', b'"precision":32'))

    def test_authority_injection_refused_before_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.json"
            packet = json.loads(FEEDBACK.read_bytes())
            packet["expected_result"] = "GAIN"
            path.write_text(json.dumps(packet))
            root = Path(tmp) / "attempt"
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                stage(root, CONTEXT, path)
            self.assertFalse(root.exists())

    def test_missing_attachment_refused_before_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "attempt"
            with self.assertRaises(FileNotFoundError):
                stage(root, CONTEXT, Path(tmp) / "missing")
            self.assertFalse(root.exists())

    def test_staged_requests_not_execution_acceptance_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "attempt"
            manifest = stage(root, CONTEXT)
            self.assertFalse(manifest["token_budget_verified"])
            self.assertFalse(manifest["execution_profile_frozen"])
            self.assertEqual(len(list(root.glob("*/*.request.json"))), 16)
            with self.assertRaises(FileExistsError):
                stage(root, CONTEXT)


if __name__ == "__main__":
    unittest.main()
