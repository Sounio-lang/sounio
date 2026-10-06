#!/usr/bin/env python3
"""Synthetic extensions of frozen CPU rows; never reinterpret old hardware as v3."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from cpu_memory_control_v3 import analyze, PROTOCOL
BASE=Path(__file__).resolve().parents[1]/"validation/external-container-cpu-v2-20260909/attempt/rank-0"
class V3Tests(unittest.TestCase):
    def setUp(self):
        self.rows=[json.loads(s) for s in (BASE/"journal.jsonl").read_text().splitlines()]
        self.phases=[json.loads(s) for s in (BASE/"cpu-phases.jsonl").read_text().splitlines()]
        for row in self.rows:
            row["schema"]=PROTOCOL["journal_schema"]
            row["observer_profile"]=PROTOCOL["observer_profile"]
        self.rows[0].update(scheduling=PROTOCOL["scheduling"],observer_resource_scope=PROTOCOL["observer_resource_scope"])
        self.samples=[r for r in self.rows if r["stage"]=="SAMPLE"]
        for i,s in enumerate(self.samples):
            s["observer_resources"]={
                "observer_pid":s["observer_pid"],"scope":PROTOCOL["observer_resource_scope"],
                "process_cpu_ns":1000+i*100,
                "status":{"value":{"VmRSS":20*1024**2,"VmHWM":21*1024**2},
                          "error":None,"format":"kB-fields","monotonic_ns":s["monotonic_ns"],"duration_ns":1}}
    def test_synthetic_good_has_narrow_claim(self):
        r=analyze(self.rows,self.phases)
        self.assertTrue(r["deadline_cadence_pass"])
        self.assertTrue(r["observer_resource_readability_pass"])
        self.assertTrue(r["custody_required_separately"])
        self.assertFalse(r["loaded_model_overhead_qualified"])
        self.assertFalse(r["observer_overhead_budget_qualified"])
    def test_old_profile_rejected(self):
        self.rows[0]["observer_profile"]="external-v2"
        with self.assertRaisesRegex(ValueError,"version"):analyze(self.rows,self.phases)
    def test_post_read_sleep_declaration_rejected(self):
        self.rows[0]["scheduling"]="post-read-sleep"
        with self.assertRaisesRegex(ValueError,"declaration"):analyze(self.rows,self.phases)
    def test_observer_is_not_target(self):
        self.samples[0]["observer_resources"]["observer_pid"]=self.samples[0]["target_pid"]
        with self.assertRaisesRegex(ValueError,"identity"):analyze(self.rows,self.phases)
    def test_missing_rss_and_wrong_units_rejected(self):
        for key,value in (("value",None),("error","PermissionError"),("format","raw")):
            saved=copy.deepcopy(self.samples[0]["observer_resources"]["status"])
            self.samples[0]["observer_resources"]["status"][key]=value
            with self.assertRaisesRegex(ValueError,"status"):analyze(self.rows,self.phases)
            self.samples[0]["observer_resources"]["status"]=saved
    def test_negative_cpu_and_counter_regression(self):
        for value in (-1,0):
            self.samples[1]["observer_resources"]["process_cpu_ns"]=value
            with self.assertRaisesRegex(ValueError,"CPU"):analyze(self.rows,self.phases)
    def test_high_water_cannot_decrease(self):
        self.samples[1]["observer_resources"]["status"]["value"]["VmHWM"]-=1
        with self.assertRaisesRegex(ValueError,"high-water"):analyze(self.rows,self.phases)
    def test_resource_timestamp_must_be_in_sample(self):
        self.samples[0]["observer_resources"]["status"]["monotonic_ns"]-=1
        with self.assertRaisesRegex(ValueError,"time outside"):analyze(self.rows,self.phases)
    def test_forged_gap_rejected(self):
        self.samples[1]["sample_gap_ns"]=1
        with self.assertRaisesRegex(ValueError,"reported gap"):analyze(self.rows,self.phases)
    def test_delayed_exit_detection_rejected(self):
        self.rows[-1]["monotonic_ns"]+=1_000_000_000
        with self.assertRaisesRegex(ValueError,"boundary"):analyze(self.rows,self.phases)
    def test_missing_oom_inherited_gate_still_rejects(self):
        self.samples[0]["metrics"]["cgroup_memory.events"]["value"]="low 0\n"
        with self.assertRaises(AssertionError):analyze(self.rows,self.phases)
    def test_optimized_python_fails_closed(self):
        r=subprocess.run([sys.executable,"-O","-c","from cpu_memory_control_v3 import analyze; analyze([],[])"],
                         cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn("optimized Python disables",r.stderr)
if __name__=="__main__":unittest.main()
