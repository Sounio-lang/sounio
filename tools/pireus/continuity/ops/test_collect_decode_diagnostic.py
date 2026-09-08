import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collect_decode_diagnostic import collect


def raw(value):
    return (json.dumps(value) + "\n").encode()


class CollectionControls(unittest.TestCase):
    def exercise(self, state="COMPLETED", exitcode="0:0", partial=False, mismatch=False):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            attempt = root / "attempt"
            attempt.mkdir()
            manifest = dict(input_sha256="input", probe_sha256="probe", source_commit="source")
            m = raw(manifest)
            (attempt / "manifest.json").write_bytes(m)
            log = b"preserved diagnostic log\n"
            (attempt / "launch.log").write_bytes(log)
            (attempt / "launcher-exit.json").write_bytes(raw(dict(returncode=0,
                log_sha256=hashlib.sha256(log).hexdigest())))
            start = dict(job=11957, attempt=str(attempt), manifest_sha256=hashlib.sha256(m).hexdigest(),
                         workers=[dict(worker=dict(pod=f"worker-{r}", uid=f"uid-{r}")) for r in (0, 1)])
            startpath = root / "start.json"
            startpath.write_bytes(raw(start))
            workers = {}
            for rank in (0, 1):
                texts = {}
                results = []
                for i in range(31 if partial else 32):
                    response = dict(job="11957", input_sha256="input", index=i,
                                    output_ids=[2 if mismatch and rank == 1 and i == 0 else 1])
                    text = raw(response).decode()
                    texts[f"offline-11957-{rank}-{i:03d}.json"] = text
                    results.append(dict(index=i, response_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                        output_tokens=1))
                if not partial:
                    texts[f"offline-11957-{rank}-complete.json"] = raw(dict(job="11957",
                        input_sha256="input", rank=str(rank), helper_sha256="probe", results=results)).decode()
                workers[f"worker-{rank}"] = texts

            def call(cmd, **kwargs):
                if "get" in cmd:
                    pod = cmd[cmd.index("pod") + 1]
                    return raw({"metadata": {"uid": "uid-" + pod[-1]}})
                if "sacct" in cmd:
                    return (f"11957|{state}|{exitcode}|gpuorangefs-multi-spark-3c59,"
                            "gpuorangefs-multi-spark-8e54|2026-09-08T01:00:00|2026-09-08T02:00:00\n")
                pod = cmd[cmd.index("exec") + 1]
                return raw(workers[pod])

            with patch("collect_decode_diagnostic.subprocess.check_output", side_effect=call):
                with contextlib.redirect_stdout(io.StringIO()):
                    collect(attempt, startpath, root / "out")
            return json.loads((root / "out/summary.json").read_text())

    def test_complete_diagnostic_never_promotes_pilot(self):
        r = self.exercise()
        self.assertTrue(r["diagnostic_batch_complete"])
        self.assertEqual(r["paired_responses"], 32)
        self.assertFalse(r["pilot_acceptance"])
        self.assertFalse(r["performance_evidence"])

    def test_partial_and_mismatched_receipts_do_not_qualify(self):
        for kwargs in (dict(partial=True), dict(mismatch=True)):
            r = self.exercise(**kwargs)
            self.assertFalse(r["diagnostic_batch_complete"])
            self.assertTrue(r["issues"])

    def test_failed_or_active_accounting_does_not_qualify(self):
        for state, code in (("FAILED", "75:0"), ("RUNNING", "0:0")):
            r = self.exercise(state=state, exitcode=code)
            self.assertFalse(r["diagnostic_batch_complete"])
            self.assertTrue(r["issues"])


if __name__ == "__main__":
    unittest.main()
