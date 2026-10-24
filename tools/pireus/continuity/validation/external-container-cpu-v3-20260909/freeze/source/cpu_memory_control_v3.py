#!/usr/bin/env python3
"""Prospective v3 CPU oracle. Successful evaluation is not custody qualification."""
import json
from pathlib import Path
from cpu_memory_control_v2 import analyze as memory_analyze, PROTOCOL as MEMORY_PROTOCOL

PROTOCOL = {
    "schema": "pireus-cpu-memory-control-v3",
    "parent_memory_oracle": MEMORY_PROTOCOL,
    "observer_profile": "external-observer-deadline-v3",
    "journal_schema": "pireus-external-memory-observation-v2",
    "scheduling": "actual-start-deadline-no-catchup",
    "interval_seconds": 0.2,
    "maximum_sample_gap_ns": 500_000_000,
    "observer_required_fields": ["VmRSS", "VmHWM"],
    "observer_resource_scope": "observer-process-only",
    "observer_cpu_counter": "cumulative-process-cpu-nanoseconds",
    "observer_overhead_budget_qualified": False,
    "scope": "64MiB CPU/container allocation, deadline cadence and observer resource readability; no model",
    "guardian_reserve_gib": 33,
    "protected_floor_gib": 32,
    "retries": 0,
    "attempts": 1,
    "custody_required_separately": True,
}
def require(ok, message):
    if not ok:
        raise ValueError(message)
def integer(value):
    return type(value) is int and value >= 0
def analyze(rows, phases):
    # v2's frozen numerical oracle uses assertions; never allow python -O to
    # disable inherited acceptance conditions.
    require(__debug__, "optimized Python disables parent oracle checks")
    require(rows and rows[0]["stage"] == "OBSERVER_START", "start missing")
    require(all(r["schema"] == PROTOCOL["journal_schema"]
                and r["observer_profile"] == PROTOCOL["observer_profile"] for r in rows), "observer version mismatch")
    start=rows[0]
    require(start["scheduling"] == PROTOCOL["scheduling"]
            and start["interval_seconds"] == PROTOCOL["interval_seconds"]
            and start["observer_resource_scope"] == PROTOCOL["observer_resource_scope"], "scheduling/resource declaration mismatch")
    memory=memory_analyze(rows,phases)
    require(rows[-1]["monotonic_ns"] >= phases[-1]["monotonic_ns"], "target exit not covered")
    samples=[r for r in rows if r["stage"] == "SAMPLE"]
    times=[start["monotonic_ns"]]+[s["monotonic_ns"] for s in samples]+[rows[-1]["monotonic_ns"]]
    require(all(integer(t) for t in times), "invalid timestamps")
    gaps=[b-a for a,b in zip(times,times[1:])]
    require(all(0 <= g <= PROTOCOL["maximum_sample_gap_ns"] for g in gaps), "boundary sampling gap")
    require(all(g>0 for g in gaps[1:]), "nonpositive sampling gap")
    resources=[]
    for i,s in enumerate(samples):
        expected_gap=None if i==0 else s["monotonic_ns"]-samples[i-1]["monotonic_ns"]
        require(s["sample_gap_ns"] == expected_gap, "reported gap mismatch")
        require(integer(s["duration_ns"]), "invalid sample duration")
        require(s["monotonic_ns"]+s["duration_ns"] <= (samples[i+1]["monotonic_ns"] if i+1<len(samples) else rows[-1]["monotonic_ns"]), "sample overlaps successor")
        own=s["observer_resources"]
        require(own["observer_pid"] == s["observer_pid"] and own["observer_pid"] != s["target_pid"]
                and own["scope"] == PROTOCOL["observer_resource_scope"], "observer resource identity")
        status=own["status"]
        require(status["error"] is None and isinstance(status["value"],dict)
                and status["format"] == "kB-fields", "observer status unavailable")
        value=status["value"]
        require(all(k in value and integer(value[k]) for k in PROTOCOL["observer_required_fields"]), "observer RSS missing/invalid")
        require(0 < value["VmRSS"] <= value["VmHWM"], "observer RSS/high-water mismatch")
        require(integer(own["process_cpu_ns"]), "observer CPU counter invalid")
        require(integer(status["monotonic_ns"]) and integer(status["duration_ns"])
                and s["monotonic_ns"] <= status["monotonic_ns"]
                and status["monotonic_ns"]+status["duration_ns"] <= s["monotonic_ns"]+s["duration_ns"], "observer resource time outside sample")
        resources.append(own)
    cpu=[r["process_cpu_ns"] for r in resources]
    hwm=[r["status"]["value"]["VmHWM"] for r in resources]
    require(all(a<=b for a,b in zip(cpu,cpu[1:])) and cpu[-1]>cpu[0], "observer CPU counter did not advance monotonically")
    require(all(a<=b for a,b in zip(hwm,hwm[1:])), "observer high-water decreased")
    return dict(schema="pireus-cpu-memory-oracle-v3-result",
        cpu_memory_oracle=memory,deadline_cadence_pass=True,observer_resource_readability_pass=True,
        samples=len(samples),maximum_gap_including_boundaries_ns=max(gaps),
        observer_peak_rss_bytes=max(r["status"]["value"]["VmRSS"] for r in resources),
        observer_high_water_bytes=max(hwm),observer_cpu_delta_ns=cpu[-1]-cpu[0],
        observer_overhead_budget_qualified=False,custody_required_separately=True,
        inference_qualified=False,loaded_model_overhead_qualified=False)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("journal",type=Path);p.add_argument("phases",type=Path)
    a=p.parse_args()
    print(json.dumps(analyze([json.loads(s) for s in a.journal.read_text().splitlines()],
                             [json.loads(s) for s in a.phases.read_text().splitlines()]),indent=2))
