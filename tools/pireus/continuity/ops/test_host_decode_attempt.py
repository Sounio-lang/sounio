#!/usr/bin/env python3
"""Local controls only; no scheduler or model calls."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import host_decode_attempt as a

class AttemptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.frozen=Path(cls.temp.name)/"frozen"
        a.create(cls.frozen)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_freeze_exact_runtime(self):
        s=a.verify(self.frozen)
        self.assertEqual(s["runtime_sha256"]["baseline"],a.protocol()["runtime_sha256"])
        self.assertEqual(s["orchestration_sha256"],a.orchestration_hashes())
    def test_no_tmux_refuses_before_network(self):
        with patch.dict(os.environ,{},clear=True),patch.object(a,"readiness") as ready:
            with self.assertRaisesRegex(ValueError,"tmux"):a.prerequisites(self.frozen,"baseline",Path("/unused"))
            ready.assert_not_called()
    def test_existing_attempt_refused(self):
        with patch.dict(os.environ,{"TMUX":"synthetic"}),patch.object(a,"readiness") as ready:
            with self.assertRaisesRegex(ValueError,"no retry"):a.prerequisites(self.frozen,"baseline",self.frozen)
            ready.assert_not_called()
    def test_observed_arm_refused(self):
        with patch.dict(os.environ,{"TMUX":"synthetic"}),patch.object(a,"readiness") as ready:
            with self.assertRaisesRegex(ValueError,"single diagnostic"):a.prerequisites(self.frozen,"observed",Path("/unused"))
            ready.assert_not_called()
    def test_ci_failure_refuses_before_stage(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"TMUX":"synthetic"}):
            stage=Path(tmp)/"attempt"
            with patch.object(a,"readiness",side_effect=ValueError("CI refused")),patch.object(a,"check_pair") as pair:
                with self.assertRaisesRegex(ValueError,"CI refused"):a.launch(self.frozen,"baseline",stage)
                pair.assert_not_called();self.assertFalse(stage.exists())
    def test_partial_failed_collection_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/"stage";stage.mkdir()
            spec=a.verify(self.frozen)
            workers=[dict(node=n,pod="worker-"+str(r),uid="uid-"+str(r),boot_id="boot-"+str(r)) for r,n in enumerate(["spark-3c59","spark-8e54"])]
            start=dict(arm="baseline",source_commit=spec["source_commit"],freeze_sha256=a.digest((self.frozen/"execution-freeze.json").read_bytes()),workers=workers)
            (stage/"start.json").write_text(json.dumps(start))
            (stage/"launch.log").write_text("synthetic failure")
            (stage/"exit-code").write_text("75")
            (stage/"readiness.json").write_text("{}")
            def run(argv):
                if "sacct" in argv:return b"123|pireus-inkling-offline-generate|FAILED|75:0|gpuorangefs-multi-spark-3c59,gpuorangefs-multi-spark-8e54|2026-09-10T03:00:00|2026-09-10T03:00:01\n"
                if "get" in argv:
                    w=next(w for w in workers if w["pod"] in argv)
                    return json.dumps({"metadata":{"uid":w["uid"]},"spec":{"nodeName":w["node"]}}).encode()
                paths=ast.literal_eval(argv[-1].split("paths=",1)[1].split(";result=",1)[0])
                return json.dumps({k:{"missing":True} for k in paths}).encode()
            r=a.collect(self.frozen,stage,root/"out","baseline","123",run)
            self.assertEqual(r["terminal_state"],"FAILED")
            self.assertTrue(r["missing"]);self.assertFalse(r["hardware_qualified"])

class ProbeTests(unittest.TestCase):
    def rows(self):
        result=[]
        for step in range(1,16):
            for stage in ["HOST_DECODE_BEGIN","HOST_DECODE_END"]:
                ns=len(result)*100
                result.append(dict(schema="pireus-host-decode-probe-v1",stage=stage,job="123",rank="0",pid=99,index=0,step=step,
                    monotonic_ns=ns,read_duration_ns=10,diagnostic_only=True,device_synchronized=False,
                    files={k:dict(raw="native units",error=None,monotonic_ns=ns,duration_ns=1) for k in ["host_meminfo","host_vmstat","process_status"]},
                    cuda_allocator={k:dict(value=100,error=None) for k in ["memory_allocated","memory_reserved"]}))
        return result
    def test_complete_window_is_not_model_qualification(self):
        r=a.inspect_probes(self.rows(),"123",0,99)
        self.assertTrue(r["complete_probe_window"]);self.assertFalse(r["loaded_model_qualified"])
    def test_failed_mid_call_preserves_valid_prefix(self):
        r=a.inspect_probes(self.rows()[:3],"123",0,99)
        self.assertTrue(r["partial_prefix"]);self.assertFalse(r["complete_probe_window"])
    def test_invalid_rows_refused(self):
        changes=[lambda r:r[0].update(job="wrong"),lambda r:r[0].update(pid=98),
                 lambda r:r[1].update(stage="HOST_DECODE_BEGIN"),
                 lambda r:r[1].update(monotonic_ns=0),
                 lambda r:r[0]["files"]["host_meminfo"].update(duration_ns=11)]
        for change in changes:
            rows=self.rows();change(rows)
            with self.subTest(change=change),self.assertRaises(ValueError):a.inspect_probes(rows,"123",0,99)
    def test_missing_metric_not_complete(self):
        rows=self.rows();rows[0]["files"]["host_meminfo"].update(raw=None,error="PermissionError")
        r=a.inspect_probes(rows,"123",0,99)
        self.assertFalse(r["complete_probe_window"]);self.assertEqual(len(r["missing_metrics"]),1)
if __name__=="__main__":unittest.main()
