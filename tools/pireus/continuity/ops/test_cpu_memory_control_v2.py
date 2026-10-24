#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import unittest
from cpu_memory_control_v2 import analyze, PROTOCOL
from external_memory_observer import metric
import tempfile

BASE=Path(__file__).resolve().parents[1]/"validation/external-observer-integration-20260909/cpu-attempt/rank-0"
class OracleTests(unittest.TestCase):
    def setUp(self):
        # Synthetic oracle fixture derived from historical rows; added limits are
        # test inputs, never evidence that 11974 sampled or passed v2.
        self.rows=[json.loads(s) for s in (BASE/"journal.jsonl").read_text().splitlines()]
        self.phases=[json.loads(s) for s in (BASE/"cpu-phases.jsonl").read_text().splitlines()]
        for s in self.rows:
            if s["stage"]=="SAMPLE":
                for key in PROTOCOL["required_limits"]:
                    s["metrics"]["cgroup_"+key]={"value":"max","error":None}
    def sample(self):
        return next(s for s in self.rows if s["stage"]=="SAMPLE")
    def test_total_cache_offset_is_not_false_failure(self):
        self.assertTrue(analyze(self.rows,self.phases)["cpu_memory_oracle_pass"])
    def test_missing_limit_refused(self):
        self.sample()["metrics"]["cgroup_memory.high"]={"value":None,"error":"PermissionError"}
        with self.assertRaises(AssertionError):analyze(self.rows,self.phases)
    def test_changed_limit_refused(self):
        self.sample()["metrics"]["cgroup_memory.max"]["value"]=512*1024**2
        with self.assertRaises(AssertionError):analyze(self.rows,self.phases)
    def test_oom_refused(self):
        self.sample()["metrics"]["cgroup_memory.events"]["value"]+="oom_kill 1\n"
        with self.assertRaises(AssertionError):analyze(self.rows,self.phases)
    def test_gap_refused(self):
        samples=[s for s in self.rows if s["stage"]=="SAMPLE"]
        self.rows=[s for s in self.rows if s not in samples[2:8]]
        with self.assertRaises(AssertionError):analyze(self.rows,self.phases)
    def test_flat_anon_refused(self):
        for s in self.rows:
            if s["stage"]=="SAMPLE":
                lines=s["metrics"]["cgroup_memory.stat"]["value"].splitlines()
                s["metrics"]["cgroup_memory.stat"]["value"]="\n".join("anon 1" if x.startswith("anon ") else x for x in lines)
        with self.assertRaises(AssertionError):analyze(self.rows,self.phases)
    def test_changed_identity_refused(self):
        self.sample()["rank"]="1"
        with self.assertRaises(AssertionError):analyze(self.rows,self.phases)
    def test_limit_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"limit"
            for raw,value in [("max","max"),("536870912",536870912)]:
                p.write_text(raw);self.assertEqual(metric(p,"bytes-or-max")["value"],value)
            for raw in ["-1","bogus",""]:
                p.write_text(raw);r=metric(p,"bytes-or-max")
                self.assertIsNotNone(r["error"]);self.assertIsNone(r["value"])
if __name__=="__main__":unittest.main()
