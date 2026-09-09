#!/usr/bin/env python3
"""Verify the immutable 11974 CPU packet; never submit or replay a job."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

QUALIFICATION_SHA = "3c4dcde1464766f37dd80c2d71c5f5445b62a341df5cad0e984e22d6e63d257e"

def verify(root):
    raw = (root / "qualification.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == QUALIFICATION_SHA, "qualification changed"
    q = json.loads(raw)
    for name, sha in q["files_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name
    assert q["job"] == 11974 and q["container_handoff_pass"]
    assert not q["original_memory_control_pass"] and not q["allocation_release_visible"]
    assert not q["inference_executed"] and not q["loaded_model_overhead_qualified"]
    for result in q["results"]:
        rank = root / ("rank-" + str(result["rank"]))
        rows = [json.loads(s) for s in (rank / "journal.jsonl").read_text().splitlines()]
        phases = [json.loads(s) for s in (rank / "cpu-phases.jsonl").read_text().splitlines()]
        assert rows[-1]["stage"] == "TARGET_INVALIDATED" and rows[-1]["metrics"] is None
        samples = [s for s in rows if s["stage"] == "SAMPLE"]
        assert len(samples) == result["samples"] == 35
        assert samples[0]["monotonic_ns"] < phases[0]["monotonic_ns"]
        values = {}
        for a, b in zip(phases, phases[1:]):
            selected = [s for s in samples if a["monotonic_ns"] + 300_000_000 < s["monotonic_ns"] < b["monotonic_ns"]]
            assert len(selected) >= 3
            assert all(v["error"] is None for s in selected for v in s["metrics"].values())
            value = {
                "Pss": statistics.median(s["metrics"]["process_smaps_rollup"]["value"]["Pss"] for s in selected),
                "cgroup_current": statistics.median(s["metrics"]["cgroup_memory.current"]["value"] for s in selected),
            }
            for key in ("anon", "file", "pgscan_direct", "pgsteal_direct"):
                value[key] = statistics.median(int(dict(line.split() for line in s["metrics"]["cgroup_memory.stat"]["value"].splitlines())[key]) for s in selected)
            values[a["phase"]] = value
        assert values == result["phase_values"], "phase medians"
        checks = {key: {
            "rise_pass": values["allocated"][key] - values["baseline"][key] >= threshold,
            "release_pass": values["allocated"][key] - values["released"][key] >= threshold
        } for key, threshold in (("Pss", 48 * 1024**2), ("cgroup_current", 32 * 1024**2))}
        assert checks == result["original_memory_checks"]
        assert checks["Pss"] == {"rise_pass": True, "release_pass": True}
        assert checks["cgroup_current"] == {"rise_pass": False, "release_pass": True}
    return {"packet_integrity": "PASS", "phase_reproduction": "PASS",
            "container_handoff": "PASS", "original_memory_control": "FAIL",
            "inference_executed": False}

if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "cpu-attempt"), indent=2))
