#!/usr/bin/env python3
import copy,json
from pathlib import Path
import unittest
from evaluate_external_overhead import response_parity,lifecycle_span,evaluate
from freeze_external_overhead import specification

BASE=Path(__file__).resolve().parents[1]/"validation/lifecycle-diagnostic-inference-20260909/without-feedback-11971/collected/worker-receipts"
class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.spec=specification()
        self.left=[[ (BASE/f"rank-{r}-{i:03d}.json").read_bytes() for i in range(8)] for r in (0,1)]
        self.right=[[json.dumps(json.loads(b)|{"job":"99999"},sort_keys=True).encode() for b in rank] for rank in self.left]
    def test_only_job_may_differ(self):
        self.assertTrue(response_parity(self.left,self.right,"11971","99999"))
    def test_profile_change_refused(self):
        v=json.loads(self.right[0][0]);v["execution_profile"]["max_total_tokens"]=16384
        self.right[0][0]=self.right[1][0]=json.dumps(v).encode()
        with self.assertRaises(ValueError):response_parity(self.left,self.right,"11971","99999")
    def test_output_change_refused(self):
        v=json.loads(self.right[0][0]);v["output_ids"][0]+=1
        self.right[0][0]=self.right[1][0]=json.dumps(v).encode()
        with self.assertRaises(ValueError):response_parity(self.left,self.right,"11971","99999")
    def test_historical_lifecycle_complete(self):
        rows=[json.loads(s) for s in (BASE/"lifecycle-11971-0.jsonl").read_text().splitlines()]
        self.assertGreater(lifecycle_span(rows,"11971",0),0)
        rows=[r for r in rows if not (r["stage"]=="CLEANUP_AFTER" and r["index"]==3)]
        with self.assertRaises(ValueError):lifecycle_span(rows,"11971",0)
    def arms(self):
        # Entire timing/metric fixture below is synthetic; never hardware evidence.
        arms=[]
        stages=["EXTEND_ENTRY","DECODE_ENTRY","DECODE_EXIT","PROPOSAL_SAVED","CLEANUP_BEFORE","CLEANUP_AFTER","REFERENCES_RELEASED"]
        for job,responses in (("11971",self.left),("99999",self.right)):
            arm=dict(job=job,responses=responses,lifecycle=[],guardian=[],external=[])
            for rank in (0,1):
                common=dict(job=job,rank=str(rank),pid=100+rank,schema="pireus-lifecycle-observation-v1")
                rows=[common|dict(stage="OBSERVER_START",monotonic_ns=100_000_000)]
                for i in range(8):
                    for stage in stages:
                        rows.append(common|dict(stage=stage,index=i,monotonic_ns=100_000_000+len(rows)*10_000_000))
                rows.append(common|dict(stage="OBSERVER_END",monotonic_ns=700_000_000))
                arm["lifecycle"].append(rows)
                arm["guardian"].append(dict(stage="MEMORY_GUARD_CHILD_EXIT",job=job,rank=str(rank),returncode=0,minimum_bytes=36*1024**3))
                extcommon=dict(job=job,rank=str(rank),target_pid=100+rank,observer_pid=200+rank,binding_sha256="test")
                ext=[extcommon|dict(stage="OBSERVER_START",monotonic_ns=0)]
                for ns in range(50_000_000,750_000_001,50_000_000):
                    metrics={k:dict(error=None,value="max") for k in self.spec["acceptance"]["required_external_metrics"]}
                    for k in ("cgroup_memory.events","cgroup_memory.events.local"):
                        metrics[k]["value"]="oom 0\noom_kill 0\noom_group_kill 0"
                    metrics["host_meminfo"]["value"]={"MemAvailable":36*1024**3}
                    metrics["process_smaps_rollup"]["value"]={"Pss":1024}
                    ext.append(extcommon|dict(stage="SAMPLE",monotonic_ns=ns,identity_valid=True,metrics=metrics))
                ext.append(extcommon|dict(stage="TARGET_INVALIDATED",monotonic_ns=800_000_000,metrics=None))
                arm["external"].append(ext)
            arms.append(arm)
        return arms
    def test_screen_pass_is_not_custody_acceptance(self):
        r=evaluate(self.spec,*self.arms())
        self.assertTrue(r["numerical_semantic_screen_pass"])
        self.assertFalse(r["loaded_model_overhead_qualified"])
    def test_memory_threshold_refused(self):
        a,b=self.arms();b["guardian"][0]["minimum_bytes"]-=257*1024**2
        with self.assertRaisesRegex(ValueError,"memory overhead"):evaluate(self.spec,a,b)
    def test_external_missing_metric_refused(self):
        a,b=self.arms();b["external"][0][1]["metrics"]["cgroup_memory.high"]["error"]="PermissionError"
        with self.assertRaisesRegex(ValueError,"metric missing"):evaluate(self.spec,a,b)
    def test_missing_tail_refused(self):
        a,b=self.arms();b["external"][0].pop()
        with self.assertRaisesRegex(ValueError,"tail missing"):evaluate(self.spec,a,b)
    def test_oom_refused(self):
        a,b=self.arms();b["external"][0][2]["metrics"]["cgroup_memory.events"]["value"]="oom 1\noom_kill 0\noom_group_kill 0"
        with self.assertRaisesRegex(ValueError,"OOM"):evaluate(self.spec,a,b)
    def test_decode_threshold_refused(self):
        a,b=self.arms()
        for rows in b["lifecycle"]:
            for r in rows:r["monotonic_ns"]=int(r["monotonic_ns"]*1.11)
        with self.assertRaisesRegex(ValueError,"decode overhead"):evaluate(self.spec,a,b)
if __name__=="__main__":unittest.main()
