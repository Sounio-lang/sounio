#!/usr/bin/env python3
"""Compare preserved first-request observations without inferring a root cause."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    raw = Path(path).read_bytes()
    return raw, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                 "bytes": len(raw)}


def events(raw, job):
    result = []
    for line in raw.decode().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("stage"):
            if "job" in row and str(row["job"]) != str(job):
                raise ValueError("mixed job identity in log")
            result.append(row)
    return result


def unique(rows, stage, rank, index=None):
    found = [r for r in rows if r["stage"] == stage and
             str(r.get("rank")) == str(rank) and
             (index is None or r.get("index") == index)]
    if len(found) != 1:
        raise ValueError(f"missing or duplicate {stage} rank{rank} index{index}")
    return found[0]


def compare(old_log, new_log, old_bundle, new_bundle):
    sources = {}
    data = {}
    for name, path in [("canary_log", old_log), ("pilot_log", new_log),
                       ("canary_bundle", old_bundle), ("pilot_bundle", new_bundle)]:
        raw, sources[name] = read(path)
        data[name] = raw
    old = events(data["canary_log"], 11939)
    new = events(data["pilot_log"], 11956)
    bundles = [json.loads(data[n]) for n in ("canary_bundle", "pilot_bundle")]
    if [len(b["items"]) for b in bundles] != [8, 32]:
        raise ValueError("unexpected experimental batch sizes")
    if bundles[0]["revision"] != bundles[1]["revision"]:
        raise ValueError("model revisions differ")
    first = [b["items"][0] for b in bundles]
    if any(r["index"] != 0 or not r["input_ids"] for r in first):
        raise ValueError("missing first request")
    profile_rows = [[unique(rows, "OFFLINE_EXECUTION_PROFILE", rank)["execution_profile"]
                     for rank in (0, 1)] for rows in (old, new)]
    if any(pair[0] != pair[1] for pair in profile_rows):
        raise ValueError("rank execution profiles differ")
    profiles = [p[0] for p in profile_rows]
    differences = {k: [profiles[0].get(k), profiles[1].get(k)]
                   for k in set(profiles[0]) | set(profiles[1])
                   if profiles[0].get(k) != profiles[1].get(k)}
    stages = {}
    for stage in ("OFFLINE_REQUEST_BEGIN", "OFFLINE_EXTEND_BEGIN", "OFFLINE_EXTEND_END",
                  "OFFLINE_FIRST_TOKEN"):
        stages[stage] = {}
        for rank in (0, 1):
            a, b = [unique(rows, stage, rank, 0) for rows in (old, new)]
            metrics = {}
            for k in ("available_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes"):
                if k in a and k in b:
                    metrics[k] = {"canary": a[k], "pilot": b[k], "pilot_minus_canary": b[k] - a[k]}
            for k in ("Rss", "Pss_Anon", "Pss_File"):
                if k in a.get("process_memory", {}) and k in b.get("process_memory", {}):
                    x, y = a["process_memory"][k], b["process_memory"][k]
                    metrics["process_" + k + "_bytes"] = {
                        "canary": x, "pilot": y, "pilot_minus_canary": y - x}
            stages[stage][str(rank)] = metrics
    stop = unique(new, "MEMORY_GUARD_STOP", 0)
    completed = [r for r in new if r["stage"] == "OFFLINE_PROPOSAL_SAVED"]
    if completed:
        raise ValueError("pilot no longer matches zero-completed-proposal comparison")
    for rank in (0, 1):
        if unique(old, "MEMORY_GUARD_CHILD_EXIT", rank)["returncode"] != 0:
            raise ValueError("canary did not exit normally")
        unique(old, "OFFLINE_PROPOSAL_SAVED", rank, 0)
    return {
        "schema": "pireus-first-request-comparison-v1",
        "jobs": {"canary": 11939, "pilot": 11956},
        "sources": sources,
        "request": {"input_tokens": [len(r["input_ids"]) for r in first],
                    "input_ids_equal": first[0]["input_ids"] == first[1]["input_ids"],
                    "generation_parameter_differences": {
                        k: [first[0].get(k), first[1].get(k)]
                        for k in set(first[0]) | set(first[1])
                        if k != "input_ids" and first[0].get(k) != first[1].get(k)}},
        "profile_differences": differences,
        "stage_metrics": stages,
        "pilot_guard_stop": stop,
        "root_cause_established": False,
        "limitations": [
            "Distinct prompts and different wall-clock runs confound causal attribution.",
            "No per-decode-step process/CUDA or host accounting in either log.",
            "First token does not reveal total tokens produced before an interrupted receipt.",
            "A sampled MemAvailable stop does not establish an atomic floor guarantee.",
            "Host memory includes unrelated processes and reclaimable caches."
        ],
        "next_control": {
            "workload": "Replay frozen pilot request0 in a separately identified diagnostic attempt.",
            "observations": [
                "Before/after decode: rank, request index, step, monotonic timestamp.",
                "Host MemAvailable plus process RSS/PSS and CUDA allocated/reserved bytes.",
                "Trace file-backed LM-head tile lifetime only if decode boundaries localize the loss."
            ],
            "preserve": ["input_ids", "seed", "temperature", "stop_token_ids", "max_new_tokens",
                         "33GiB guard", "32GiB floor", "native parity and gain criteria"],
            "promotion": "Diagnostic success alone does not complete a32-request pilot cell."
        }
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("old-log", "new-log", "old-bundle", "new-bundle"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(compare(a.old_log, a.new_log, a.old_bundle, a.new_bundle), indent=2, sort_keys=True))
