#!/usr/bin/env python3
"""Frozen v3 loaded screening checks; never submit or grant custody acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import evaluate_external_overhead as parent

SPEC_PATH = Path(__file__).resolve().parents[1] / "validation/external-overhead-v3-preparation-20260910/proposal.json"
SPEC_SHA = "011a65f2741c336588ac3bfae4609a3bbe469469d1ce2714378468be3f2a911e"
require = parent.require

def specification():
    raw = SPEC_PATH.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SPEC_SHA, "v3 specification changed")
    return json.loads(raw)

def integer(value):
    return type(value) is int and value >= 0

def observer_resources(rows):
    """Check the deadline-v3 fields without assuming a CPU allocation workload."""
    require(len(rows) >= 4 and rows[0]["stage"] == "OBSERVER_START"
            and rows[-1]["stage"] == "TARGET_INVALIDATED"
            and rows[-1]["metrics"] is None, "v3 observer boundaries")
    require(all(r["schema"] == "pireus-external-memory-observation-v2"
                and r["observer_profile"] == "external-observer-deadline-v3"
                for r in rows), "v3 observer version")
    start = rows[0]
    require(start["scheduling"] == "actual-start-deadline-no-catchup"
            and start["interval_seconds"] == 0.2
            and start["observer_resource_scope"] == "observer-process-only",
            "v3 scheduling/resource declaration")
    samples = rows[1:-1]
    require(all(s["stage"] == "SAMPLE" for s in samples), "unexpected observer event")
    identity = ("job", "rank", "target_pid", "observer_pid", "binding_sha256")
    require(all(all(r[k] == start[k] for k in identity) for r in rows), "v3 identity drift")
    require(integer(start["target_pid"]) and integer(start["observer_pid"])
            and start["target_pid"] > 0 and start["observer_pid"] > 0
            and start["target_pid"] != start["observer_pid"], "v3 process identity")
    times = [r["monotonic_ns"] for r in rows]
    require(all(integer(t) for t in times), "v3 timestamp type")
    gaps = [b-a for a,b in zip(times,times[1:])]
    require(0 <= gaps[0] <= 500_000_000
            and all(0 < g <= 500_000_000 for g in gaps[1:]), "v3 boundary sampling gap")
    resources = []
    for i,s in enumerate(samples):
        require(s["identity_valid"] is True, "v3 invalid sample identity")
        expected = None if i == 0 else s["monotonic_ns"] - samples[i-1]["monotonic_ns"]
        require(s["sample_gap_ns"] == expected, "v3 reported gap mismatch")
        successor = rows[i+2]["monotonic_ns"]
        require(integer(s["duration_ns"]) and s["monotonic_ns"] + s["duration_ns"] <= successor,
                "v3 sample overlaps successor")
        own = s["observer_resources"]
        require(own["observer_pid"] == s["observer_pid"] and own["scope"] == "observer-process-only",
                "v3 resource identity")
        status = own["status"]
        require(status["error"] is None and isinstance(status["value"], dict)
                and status["format"] == "kB-fields", "v3 observer status unavailable")
        value = status["value"]
        require(all(k in value and integer(value[k]) for k in ("VmRSS", "VmHWM"))
                and 0 < value["VmRSS"] <= value["VmHWM"], "v3 RSS/high-water invalid")
        require(integer(own["process_cpu_ns"]), "v3 CPU counter invalid")
        require(integer(status["monotonic_ns"]) and integer(status["duration_ns"])
                and s["monotonic_ns"] <= status["monotonic_ns"]
                and status["monotonic_ns"] + status["duration_ns"] <= s["monotonic_ns"] + s["duration_ns"],
                "v3 resource timestamp outside sample")
        resources.append(own)
    cpu = [r["process_cpu_ns"] for r in resources]
    hwm = [r["status"]["value"]["VmHWM"] for r in resources]
    require(all(a <= b for a,b in zip(cpu,cpu[1:])) and cpu[-1] > cpu[0], "v3 CPU counter progression")
    require(all(a <= b for a,b in zip(hwm,hwm[1:])), "v3 high-water decreased")
    return dict(samples=len(samples), maximum_gap_including_boundaries_ns=max(gaps),
                observer_peak_rss_bytes=max(r["status"]["value"]["VmRSS"] for r in resources),
                observer_high_water_bytes=max(hwm), observer_cpu_delta_ns=cpu[-1]-cpu[0],
                observer_resource_readability_pass=True, deadline_cadence_pass=True,
                observer_overhead_budget_qualified=False)

def evaluate(spec, baseline, observed):
    require(spec == specification(), "v3 specification mismatch")
    # Only the schema is adapted for the frozen parent numerical checks.
    # Bounds, response equality, lifecycle, OOM and coverage remain unchanged.
    inherited = copy.deepcopy(spec)
    inherited["schema"] = "pireus-external-overhead-freeze-v2"
    numerical = parent.evaluate(inherited, baseline, observed)
    resources = [observer_resources(rows) for rows in observed["external"]]
    require(len(resources) == 2, "two observer ranks required")
    return dict(schema="pireus-external-overhead-v3-screen-result",
                numerical_semantic_screen_pass=True, parent_screen=numerical,
                observer_resources=resources, specification_sha256=SPEC_SHA,
                custody_required_separately=True, loaded_model_overhead_qualified=False,
                pilot_acceptance=False)
