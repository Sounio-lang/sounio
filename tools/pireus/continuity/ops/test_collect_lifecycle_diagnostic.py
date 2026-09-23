#!/usr/bin/env python3
"""Synthetic transport tests using immutable historical launch/accounting."""
import ast
import base64
import json
from pathlib import Path
import tempfile
import unittest
from collect_lifecycle_diagnostic import collect
from freeze_lifecycle_diagnostic import create
from feedback_smoke import HERE

class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.frozen = cls.root/"frozen"
        create(cls.frozen)
        cls.stage = HERE/"validation/feedback-smoke-inference-20260909/without-feedback-11969"
        cls.pods = {}
        for line in (cls.stage/"launch.log").read_text().splitlines():
            try: row=json.loads(line)
            except ValueError: continue
            if row.get("mode") == "offline-generate":
                cls.pods={p["pod"]:p["uid"] for p in row["nodes"]}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.case=Path(tempfile.mkdtemp(dir=self.root))
        self.running=False
        self.bad_uid=False

    def transport(self, args):
        if "sacct" in args:
            raw=(self.stage/"accounting.txt").read_bytes()
            return raw.replace(b"COMPLETED", b"RUNNING") if self.running else raw
        if "get" in args:
            name=args[args.index("pod")+1]
            return json.dumps({"metadata":{"uid":"wrong" if self.bad_uid else self.pods[name]}}).encode()
        code=args[-1]
        paths=ast.literal_eval(code.split("for name,path in ",1)[1].split(".items():",1)[0])
        result={}
        for name in paths:
            p=self.stage/"worker-receipts"/name
            if name.endswith("-0.jsonl"):
                # Preserve a deliberately truncated journal without accepting it.
                result[name]={"base64":base64.b64encode(b'{"partial":').decode()}
            elif p.exists():
                result[name]={"base64":base64.b64encode(p.read_bytes()).decode()}
            else: result[name]={"missing":True}
        return json.dumps(result).encode()

    def call(self):
        return collect(self.frozen,self.stage,self.case/"collected","11969",run=self.transport)

    def test_raw_partial_and_missing_preserved(self):
        r=self.call()
        self.assertFalse(r["diagnostic_complete"])
        self.assertFalse(r["hardware_qualified"])
        self.assertIn("lifecycle-11969-1.jsonl",r["missing"])
        self.assertEqual((self.case/"collected/worker-receipts/lifecycle-11969-0.jsonl").read_bytes(),b'{"partial":')
        with self.assertRaises(FileExistsError): self.call()

    def test_live_job_refused_before_output(self):
        self.running=True
        with self.assertRaisesRegex(ValueError,"terminal"): self.call()
        self.assertFalse((self.case/"collected").exists())

    def test_changed_worker_refused_without_receipt(self):
        self.bad_uid=True
        with self.assertRaisesRegex(ValueError,"UID"): self.call()
        self.assertFalse((self.case/"collected/collection.json").exists())

if __name__ == "__main__":
    unittest.main()
