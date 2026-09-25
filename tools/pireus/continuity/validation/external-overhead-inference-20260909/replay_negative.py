#!/usr/bin/env python3
"""Replay the negative 11976/11977 screen. Never submit or qualify partial data."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "ops"))
from external_overhead_custody import accounting, qualify
from evaluate_external_overhead import lifecycle_span
from cpu_memory_control_v2 import fields

PINS = {
    "baseline-11976": "510f98de0c30d183fa71caf21b6955ad86a59bbcdcd36de636a9c0f333da738f",
    "observed-11977": "6703d22ee93fa2e71068dedc44df4c5af3e7b91a5a987601960ec97a3d43b9a7",
}
def require(ok, message):
    if not ok:
        raise ValueError(message)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_bytes())
def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]
def packet(name):
    root = HERE / name
    require(sha(root / "collection.json") == PINS[name], "collection pin")
    c = read(root / "collection.json")
    for name, pin in c["files_sha256"].items():
        p = root / name
        require(p.resolve().is_relative_to(root.resolve()) and sha(p) == pin, "artifact hash: " + name)
    require(all(not (root / n).exists() for n in c["missing"]), "missing artifact appeared")
    return root, c
def analyze():
    base, bc = packet("baseline-11976")
    obs, oc = packet("observed-11977")
    spec_path = HERE.parent / "external-overhead-freeze-v2-20260909/execution-freeze.json"
    spec = read(spec_path)
    require(sha(spec_path) == bc["freeze_sha256"] == oc["freeze_sha256"], "freeze pin")
    require(bc["source_commit"] == oc["source_commit"] == spec["source_commit"], "source identity")
    require(bc["job"] == "11976" and oc["job"] == "11977", "job identity")
    ba = accounting((base / "accounting.txt").read_bytes(), "11976", True)
    oa = accounting((obs / "accounting.txt").read_bytes(), "11977")
    require(oa[2:4] == ["FAILED", "75:0"] and (obs / "exit-code").read_text().strip() == "75", "negative terminal")
    require(ba[6] <= oa[5], "arm order")
    expected_missing = {f"worker-receipts/rank-{r}-{n}.json" for r in (0,1) for n in ("complete","006","007")}
    require(set(oc["missing"]) == expected_missing and not bc["missing"], "missing inventory")
    starts = [read(p / "start.json") for p in (base,obs)]
    require(starts[0]["workers"] == starts[1]["workers"], "paired worker identity")
    for root,c,arm,start in zip((base,obs),(bc,oc),("baseline","observed"),starts):
        require(start["arm"] == arm and start["freeze_sha256"] == c["freeze_sha256"], "start identity")
        require(start["runtime_before_sha256"] == [spec["runtime_sha256"][arm]] * 2, "prelaunch runtime")
        for rank,w in enumerate(start["workers"]):
            for moment in ("before","after"):
                pod = read(root / f"worker-{rank}-{moment}.json")
                require(pod["metadata"]["uid"] == w["uid"] and pod["spec"]["nodeName"] == w["node"], "pod identity")
            require((root / f"rank-{rank}/boot-id.txt").read_text().strip() == w["boot_id"], "boot identity")
            life = rows(root / f"worker-receipts/lifecycle-{c['job']}-{rank}.jsonl")
            barrier = read(root / f"rank-{rank}/runtime-before.json")
            require(barrier["job"] == c["job"] and barrier["rank"] == str(rank)
                    and barrier["worker_uid"] == w["uid"] and barrier["boot_id"] == w["boot_id"]
                    and barrier["runtime_sha256"] == spec["runtime_sha256"][arm]
                    and barrier["input_sha256"] == start["input_sha256"]
                    and barrier["monotonic_ns"] < life[0]["monotonic_ns"], "runtime barrier")
            for name,pin in spec["runtime_sha256"][arm].items():
                require(sha(root / f"rank-{rank}/runtime" / name) == pin, "runtime bytes")
    logs = []
    for root in (base,obs):
        parsed=[]
        for line in (root / "launch.log").read_bytes().splitlines():
            try: item=json.loads(line)
            except ValueError: continue
            if isinstance(item,dict): parsed.append(item)
        logs.append(parsed)
    partial_tokens=0
    partial_differences=[]
    for i in range(6):
        values=[]
        for root,job in ((base,"11976"),(obs,"11977")):
            a=(root / f"worker-receipts/rank-0-{i:03d}.json").read_bytes()
            b=(root / f"worker-receipts/rank-1-{i:03d}.json").read_bytes()
            require(a == b, "partial rank byte parity")
            v=json.loads(a)
            require(v["job"] == job and v["index"] == i and v["sampling_authority_rank"] == 0
                    and v["completion_tokens"] == len(v["output_ids"]) > 0, "partial response identity")
            values.append({k:x for k,x in v.items() if k != "job"})
        partial_differences.append(dict(index=i, differing_fields=sorted(k for k in values[0].keys() | values[1].keys() if values[0].get(k)!=values[1].get(k))))
        partial_tokens += values[1]["completion_tokens"]
    ranks=[]
    for rank,w in enumerate(starts[1]["workers"]):
        life=rows(obs / f"worker-receipts/lifecycle-11977-{rank}.jsonl")
        p=obs / f"rank-{rank}/external"
        target,ack,binding,result=[read(p/n) for n in ("target.json","attached.json","binding.json","result.json")]
        require(target["job"] == "11977" and target["rank"] == str(rank)
                and target["boot_id"] == w["boot_id"] and target["pid"] == life[0]["pid"]
                and target["entry_sha256"] == spec["runtime_sha256"]["observed"]["offline_generate.py"], "target")
        require(target["nonce"] == ack["nonce"] == result["nonce"] and ack["first_sample_valid"] is True, "attachment")
        require(all(ack[k] == target[k] for k in ("job","rank","pid","starttime_ticks")), "ack target")
        require(ack["handoff_sha256"] == sha(p/"target.json") and ack["binding_sha256"] == sha(p/"binding.json"), "ack hashes")
        require(binding["expected"] == {k:target[k] for k in ("job","rank","pid","starttime_ticks","boot_id")} | {"worker_uid":w["uid"]}, "binding identity")
        require(binding["observer_helper_sha256"] == spec["runtime_sha256"]["observed"]["external_memory_observer.py"], "observer source")
        for name,pin in result["files_sha256"].items():
            require((p/name).resolve().is_relative_to(p.resolve()) and sha(p/name) == pin, "supervisor hash")
        journal=rows(p/"journal.jsonl")
        binding_pin=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
        require(all(r["binding_sha256"] == binding_pin and r["observer_pid"] == ack["observer_pid"]
                    and r["target_pid"] == target["pid"] and r["job"] == "11977" and r["rank"] == str(rank) for r in journal), "journal identity")
        samples=[r for r in journal if r["stage"] == "SAMPLE"]
        require(all(r["identity_valid"] for r in samples), "sample identity")
        times=[r["monotonic_ns"] for r in samples]
        if journal[-1]["stage"] == "TARGET_INVALIDATED": times.append(journal[-1]["monotonic_ns"])
        require(all(b>a for a,b in zip(times,times[1:])), "sample order")
        gap=max(b-a for a,b in zip(times,times[1:]))
        metric_errors=sum(any(s["metrics"][k]["error"] is not None or s["metrics"][k]["value"] is None
                              for k in spec["acceptance"]["required_external_metrics"]) for s in samples)
        oom={}
        for name in ("cgroup_memory.events","cgroup_memory.events.local"):
            tables=[fields(s["metrics"][name]["value"]) for s in samples]
            oom[name]={k:max(t[k] for t in tables)-min(t[k] for t in tables) for k in ("oom","oom_kill","oom_group_kill")}
        mins=[min(r["minimum_bytes"] for r in log if r.get("rank")==str(rank) and r.get("stage","").startswith("MEMORY_GUARD") and "minimum_bytes" in r) for log in logs]
        stops=[r for r in logs[1] if r.get("rank")==str(rank) and r.get("stage")=="MEMORY_GUARD_STOP"]
        try:
            lifecycle_span(life,"11977",rank)
        except ValueError as e: lifecycle_error=str(e)
        else: raise ValueError("unexpected complete lifecycle")
        ranks.append(dict(rank=rank, baseline_minimum_available_bytes=mins[0],
            observed_minimum_available_bytes=mins[1], observed_minimum_loss_bytes=mins[0]-mins[1],
            memory_loss_bound_exceeded=mins[0]-mins[1]>spec["acceptance"]["maximum_loss_minimum_host_available_bytes"],
            guardian_stops=stops, external_samples=len(samples), maximum_sample_gap_ns=gap,
            sample_gap_pass=gap<=spec["acceptance"]["observer_maximum_sample_gap_ns"],
            required_metric_error_samples=metric_errors, observed_oom_counter_ranges=oom,
            external_tail=journal[-1]["stage"], lifecycle_error=lifecycle_error,
            entry_preceded_by_external_sample=samples[0]["monotonic_ns"]<life[0]["monotonic_ns"],
            guardian_returncode=result["guardian_returncode"], observer_returncode=result["observer_returncode"],
            integration_complete=result["integration_complete"]))
    return dict(schema="pireus-external-overhead-negative-closure-v1",
        source_commit=spec["source_commit"], freeze_sha256=sha(spec_path), collections=PINS,
        evidence_hashes_and_recorded_identity_verified=True, successful_pair_custody=False,
        terminal_state="FAILED", exit_code="75:0", completed_requests_per_rank=6,
        saved_output_tokens_per_rank=partial_tokens, partial_within_job_rank_parity=True,
        partial_cross_arm_parity=all(not d['differing_fields'] for d in partial_differences),
        partial_cross_arm_differences=partial_differences,
        missing_artifacts=sorted(oc["missing"]), full_response_parity=None, full_decode_ratio=None,
        loaded_model_overhead_qualified=False, pilot_acceptance=False, retry_performed=False,
        conclusion="FAILED_SCREEN_INCOMPLETE_PAIR", ranks=ranks,
        limitations=["Partial parity covers indices 0..5 only.",
          "Minima compare full baseline to interrupted observed run; not causal overhead.",
          "No complete lifecycle or rank-0 target-exit coverage.",
          "No unsampled OOM or protected-floor claim follows from sampled counters.",
          "Single fixed-order pair cannot isolate observer cost from order/cache/host state."])

if __name__ == "__main__":
    print(json.dumps(analyze(),indent=2))
