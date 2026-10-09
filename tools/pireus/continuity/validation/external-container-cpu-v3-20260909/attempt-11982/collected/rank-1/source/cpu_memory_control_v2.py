#!/usr/bin/env python3
"""Predeclared v2 CPU allocation oracle; no inference or scheduler operations."""
import json
import statistics
from pathlib import Path

MIB = 1024**2
PROTOCOL = {
    "schema": "pireus-cpu-memory-control-v2",
    "allocation_bytes": 64*MIB,
    "phase_order": ["baseline", "allocated", "released", "exiting"],
    "settling_ns": 300_000_000,
    "minimum_samples_per_phase": 3,
    "maximum_sample_gap_ns": 500_000_000,
    "pss_delta_bounds_bytes": [48*MIB, 80*MIB],
    "anon_delta_bounds_bytes": [48*MIB, 80*MIB],
    "slurm_memory_per_node": "512M",
    "guardian_reserve_gib": 33,
    "protected_floor_gib": 32,
    "retries": 0,
    "total_memory_rise_required": False,
    "cache_and_reclaim_are_observations": True,
    "required_limits": ["memory.max", "memory.high", "memory.swap.max"],
    "no_oom_event_increase": True,
    "scope": "CPU 64MiB allocation in Apptainer on exclusive Spark pair; no model",
    "limit_scope": "actual target task cgroup; ancestor limits not inferred",
}

def fields(raw):
    pairs = [line.split() for line in raw.splitlines()]
    assert all(len(p)==2 for p in pairs), "malformed counter table"
    assert len({p[0] for p in pairs})==len(pairs), "duplicate counter"
    return {k:int(v) for k,v in pairs}

def analyze(rows, phases):
    assert [p["phase"] for p in phases]==PROTOCOL["phase_order"], "phase order"
    assert all(a["monotonic_ns"]<b["monotonic_ns"] for a,b in zip(phases,phases[1:])), "phase times"
    samples=[r for r in rows if r["stage"]=="SAMPLE"]
    assert samples and rows[-1]["stage"]=="TARGET_INVALIDATED" and rows[-1]["metrics"] is None
    assert samples[0]["monotonic_ns"]<phases[0]["monotonic_ns"], "entry preceded observer"
    assert all(s["identity_valid"] for s in samples)
    assert all(p["pid"]==samples[0]["target_pid"] for p in phases)
    identity=("job","rank","target_pid","observer_pid","binding_sha256")
    assert all(all(s[k]==samples[0][k] for k in identity) for s in rows), "identity drift"
    assert samples[0]["target_pid"]!=samples[0]["observer_pid"]
    assert all(0<b["monotonic_ns"]-a["monotonic_ns"]<=PROTOCOL["maximum_sample_gap_ns"] for a,b in zip(samples,samples[1:])), "sample gap"
    required=["process_smaps_rollup","cgroup_memory.current","cgroup_memory.stat","cgroup_memory.events","cgroup_memory.events.local"]+["cgroup_"+n for n in PROTOCOL["required_limits"]]
    assert all(s["metrics"][k]["error"] is None and s["metrics"][k]["value"] is not None for s in samples for k in required), "metric unavailable"
    limits={}
    for key in PROTOCOL["required_limits"]:
        values=[s["metrics"]["cgroup_"+key]["value"] for s in samples]
        assert all(v=="max" or type(v) is int and v>=0 for v in values), "invalid limit"
        assert all(v==values[0] for v in values), "limit changed"
        limits[key]=values[0]
    event_deltas={}
    for name in ("cgroup_memory.events","cgroup_memory.events.local"):
        counters=[fields(s["metrics"][name]["value"]) for s in samples]
        for key in ("oom","oom_kill","oom_group_kill"):
            assert all(key in c for c in counters), "missing OOM counter"
            assert all(c[key]==counters[0][key] for c in counters), "OOM event"
        event_deltas[name]={k:counters[-1][k]-v for k,v in counters[0].items()}
    values={}
    for a,b in zip(phases,phases[1:]):
        selected=[s for s in samples if a["monotonic_ns"]+PROTOCOL["settling_ns"]<s["monotonic_ns"] and s["monotonic_ns"]+s["duration_ns"]<b["monotonic_ns"]]
        assert len(selected)>=PROTOCOL["minimum_samples_per_phase"], "insufficient phase coverage"
        values[a["phase"]]={
            "pss":statistics.median(s["metrics"]["process_smaps_rollup"]["value"]["Pss"] for s in selected),
            "current":statistics.median(s["metrics"]["cgroup_memory.current"]["value"] for s in selected),
        }
        for key in ("anon","file","pgscan_direct","pgsteal_direct"):
            values[a["phase"]][key]=statistics.median(fields(s["metrics"]["cgroup_memory.stat"]["value"])[key] for s in selected)
    for key,bounds in (("pss",PROTOCOL["pss_delta_bounds_bytes"]),("anon",PROTOCOL["anon_delta_bounds_bytes"])):
        for reference in ("baseline","released"):
            delta=values["allocated"][key]-values[reference][key]
            assert bounds[0]<=delta<=bounds[1], key+" allocation/release delta"
    return {"cpu_memory_oracle_pass":True,"phase_medians":values,"task_limits":limits,
            "event_deltas":event_deltas,"inference_qualified":False,
            "handoff_custody_requires_separate_verification":True}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("journal",type=Path);p.add_argument("phases",type=Path)
    a=p.parse_args()
    print(json.dumps(analyze([json.loads(s) for s in a.journal.read_text().splitlines()],
                            [json.loads(s) for s in a.phases.read_text().splitlines()]),indent=2))
