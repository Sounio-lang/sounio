#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from publish_observer_target import publish,await_ack,exclusive_json,digest
from external_rank_supervisor import supervise,complete_rows
from build_external_observer_integration import build,generate,BASE,GUARD_SHA,ENTRY_SHA

class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.entry=self.root/"entry.py"
        self.entry.write_text("print('entry')\n")
        self.env=patch.dict(os.environ,{"SLURM_JOB_ID":"713","PIREUS_RANK":"0"})
        self.env.start();self.addCleanup(self.env.stop)
    def target(self):
        return publish(self.entry,digest(self.entry.read_bytes()),self.root/"target.json","a"*64)
    def ack(self,target,sha):
        return {k:target[k] for k in ("nonce","job","rank","pid","starttime_ticks")}|dict(
            schema="pireus-observer-attachment-ack-v1",handoff_sha256=sha,
            observer_pid=os.getpid()+1,first_sample_valid=True)
    def test_publish_and_ack_identity(self):
        target,sha=self.target()
        self.assertEqual(target["pid"],os.getpid())
        self.assertEqual(target["starttime_ticks"],int(Path("/proc/self/stat").read_text().rsplit(") ",1)[1].split()[19]))
        exclusive_json(self.root/"ack.json",self.ack(target,sha))
        self.assertTrue(await_ack(self.root/"ack.json",target,sha,0.1)["first_sample_valid"])
    def test_stale_ack_refused(self):
        target,sha=self.target()
        ack=self.ack(target,sha);ack["nonce"]="b"*64
        exclusive_json(self.root/"ack.json",ack)
        with self.assertRaisesRegex(ValueError,"acknowledgement"):
            await_ack(self.root/"ack.json",target,sha,0.1)
    def test_missing_ack_times_out(self):
        target,sha=self.target()
        with self.assertRaises(TimeoutError):
            await_ack(self.root/"missing.json",target,sha,0.01)
    def test_entry_change_and_duplicate_handoff_refused(self):
        with self.assertRaisesRegex(ValueError,"source identity"):
            publish(self.entry,"0"*64,self.root/"target.json","a"*64)
        self.target();raw=(self.root/"target.json").read_bytes()
        with self.assertRaises(FileExistsError):self.target()
        self.assertEqual((self.root/"target.json").read_bytes(),raw)
    def test_guardian75_preserved_without_target(self):
        rc=supervise([sys.executable,"-c","raise SystemExit(75)"],self.root/"run","uid","boot","x",1,2)
        self.assertEqual(rc,75)
        result=json.loads((self.root/"run/result.json").read_bytes())
        self.assertFalse(result["observer_attached"]);self.assertEqual(result["guardian_returncode"],75)
    def test_zero_exit_without_attachment_is_not_pass(self):
        rc=supervise([sys.executable,"-c","pass"],self.root/"run","uid","boot","x",1,2)
        self.assertEqual(rc,76)
        self.assertFalse(json.loads((self.root/"run/result.json").read_bytes())["integration_complete"])
    def test_partial_journal_line_not_accepted(self):
        p=self.root/"journal.jsonl";p.write_bytes(b'{"stage":"OBSERVER_START"}\n{"stage":')
        self.assertEqual(len(complete_rows(p)),1)
    def test_build_preserves_guardian_and_entry(self):
        out=self.root/"build";m=build(out)
        self.assertEqual(digest((out/"memory_guard.py").read_bytes()),GUARD_SHA)
        self.assertEqual(digest((out/"offline_generate.py").read_bytes()),ENTRY_SHA)
        self.assertIn("external_rank_supervisor.py",(out/"serve_rank.sh").read_text())
        self.assertEqual(subprocess.run(["bash","-n",str(out/"serve_rank.sh")]).returncode,0)
        with self.assertRaises(ValueError):generate((BASE/"runtime/serve_rank.sh").read_bytes()+b"\n")

if __name__=="__main__":unittest.main()
