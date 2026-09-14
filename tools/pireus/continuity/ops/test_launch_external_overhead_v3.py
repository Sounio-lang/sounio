#!/usr/bin/env python3
import os,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import launch_external_overhead_v3 as l
class LaunchTests(unittest.TestCase):
    def test_tmux_required_before_source_read(self):
        with patch.dict(os.environ,{},clear=True),patch.object(l,"verify") as verify:
            with self.assertRaisesRegex(ValueError,"tmux"):l.prerequisites(Path("/unused"),"baseline",Path("/unused-stage"))
            verify.assert_not_called()
    def test_existing_attempt_refused(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"TMUX":"test"}),patch.object(l,"verify") as verify:
            with self.assertRaisesRegex(ValueError,"already exists"):l.prerequisites(Path("/unused"),"baseline",Path(tmp))
            verify.assert_not_called()
    def test_observed_requires_baseline(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"TMUX":"test"}),patch.object(l,"verify",return_value={}),patch.object(l,"readiness") as ready:
            with self.assertRaisesRegex(ValueError,"baseline custody"):l.prerequisites(Path("/unused"),"observed",Path(tmp)/"new")
            ready.assert_not_called()
    def test_source_ci_refusal_propagates(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"TMUX":"test"}),patch.object(l,"verify",return_value={}),patch.object(l,"readiness",side_effect=ValueError("source CI pending")):
            with self.assertRaisesRegex(ValueError,"CI pending"):l.prerequisites(Path("/unused"),"baseline",Path(tmp)/"new")
    def workers(self):
        return [dict(node=n,uid="uid-"+str(i),boot_id="boot-"+str(i)) for i,n in enumerate(("spark-3c59","spark-8e54"))]
    def test_command_binds_each_rank_and_checks_runtime(self):
        a=l.command(self.workers(),"a"*64,{"memory_guard.py":"b"*64})
        self.assertIn("--exclusive",a);self.assertIn("--kill-on-bad-exit=1",a)
        self.assertIn("--mem=110G",a);self.assertIn("--time=55",a)
        s=a[-1]
        for rank in (0,1):
            self.assertIn("PIREUS_RANK="+str(rank),s)
            self.assertIn("PIREUS_EXTERNAL_WORKER_UID=uid-"+str(rank),s)
        self.assertIn("runtime changed before entry",s)
        self.assertIn("input changed before entry",s)
        self.assertLess(s.index("OVERHEAD_RUNTIME_VERIFIED"),s.index("exec /scratch/pireus/runtime/serve_rank.sh"))
        subprocess.run(["bash","-n"],input=s.encode(),check=True)
    def test_reversed_workers_refused(self):
        with self.assertRaises(ValueError):l.command(list(reversed(self.workers())),"a"*64,{})
if __name__=="__main__":unittest.main()
