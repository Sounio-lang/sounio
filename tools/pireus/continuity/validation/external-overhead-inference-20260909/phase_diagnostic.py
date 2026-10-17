#!/usr/bin/env python3
"""Post-hoc diagnostics on pinned archives; no inference or acceptance changes."""
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from replay_negative import analyze, HERE, read, rows, require
sys.path.insert(0,str(HERE.parent.parent/"ops"))
from cpu_memory_control_v2 import fields

def metrics(sample):
    m=sample["metrics"]
    require(all(v["error"] is None and v["value"] is not None for v in m.values()),"unreadable metric")
    stat=fields(m["cgroup_memory.stat"]["value"])
    return dict(host_available=m["host_meminfo"]["value"]["MemAvailable"],
                host_cached=m["host_meminfo"]["value"]["Cached"],
                host_anon=m["host_meminfo"]["value"]["AnonPages"],
                host_slab=m["host_meminfo"]["value"]["Slab"],
                process_pss=m["process_smaps_rollup"]["value"]["Pss"],
                process_pss_anon=m["process_smaps_rollup"]["value"].get("Pss_Anon"),
                process_pss_file=m["process_smaps_rollup"]["value"].get("Pss_File"),
                cgroup_current=m["cgroup_memory.current"]["value"],
                cgroup_anon=stat["anon"],cgroup_file=stat["file"],cgroup_kernel=stat["kernel"],
                cgroup_pgscan=stat["pgscan"],cgroup_pgsteal=stat["pgsteal"])

def main():
    closure=analyze()
    require(closure==read(HERE/"negative-result.json"),"closure replay changed")
    bundle=read(HERE/"diagnostic-input-bundle.json")
    bundle_sha=hashlib.sha256((HERE/"diagnostic-input-bundle.json").read_bytes()).hexdigest()
    for arm in ("baseline-11976","observed-11977"):
        require(read(HERE/arm/"start.json")["input_sha256"]==bundle_sha,"input bundle pin")
    ranks=[]
    for rank in (0,1):
        ext=rows(HERE/f"observed-11977/rank-{rank}/external/journal.jsonl")
        samples=[r for r in ext if r["stage"]=="SAMPLE"]
        life=rows(HERE/f"observed-11977/worker-receipts/lifecycle-11977-{rank}.jsonl")
        base=rows(HERE/f"baseline-11976/worker-receipts/lifecycle-11976-{rank}.jsonl")
        a,b=max(zip(samples,samples[1:]),key=lambda pair:pair[1]["monotonic_ns"]-pair[0]["monotonic_ns"])
        prior=[r for r in life if r["stage"]!="HOST_SAMPLE" and r["monotonic_ns"]<=a["monotonic_ns"]][-1]
        gap=dict(start_ns=a["monotonic_ns"],end_ns=b["monotonic_ns"],
            gap_ns=b["monotonic_ns"]-a["monotonic_ns"],sample_duration_ns=a["duration_ns"],
            between_sample_work_ns=b["monotonic_ns"]-a["monotonic_ns"]-a["duration_ns"],
            metric_read_durations_ns={k:v["duration_ns"] for k,v in a["metrics"].items()},
            preceding_hook=dict(stage=prior["stage"],index=prior["index"],monotonic_ns=prior["monotonic_ns"]),
            before_first_extend=b["monotonic_ns"]<next(r["monotonic_ns"] for r in life if r["stage"]=="EXTEND_ENTRY"))
        hooks=[]
        for stage in ("DECODE_ENTRY","REFERENCES_RELEASED"):
            for observed in [r for r in life if r["stage"]==stage]:
                baseline=next(r for r in base if r["stage"]==stage and r["index"]==observed["index"])
                require(all(r.get("host_error") is None and r.get("process_error") is None for r in (observed,baseline)),"hook read error")
                preceding=max((s for s in samples if s["monotonic_ns"]<=observed["monotonic_ns"]),key=lambda s:s["monotonic_ns"])
                hooks.append(dict(stage=stage,index=observed["index"],monotonic_ns=observed["monotonic_ns"],
                    baseline_host=baseline["host"],observed_host=observed["host"],
                    baseline_process=baseline["process"],observed_process=observed["process"],
                    baseline_cuda=baseline["cuda"],observed_cuda=observed["cuda"],
                    host_available_delta_observed_minus_baseline=observed["host"]["MemAvailable"]-baseline["host"]["MemAvailable"],
                    process_pss_delta_observed_minus_baseline=observed["process"]["Pss"]-baseline["process"]["Pss"],
                    cuda_allocated_equal=observed["cuda"]["allocated"]==baseline["cuda"]["allocated"],
                    preceding_external_sample_age_ns=observed["monotonic_ns"]-preceding["monotonic_ns"],
                    preceding_external_sample_duration_ns=preceding["duration_ns"],
                    preceding_external_metrics=metrics(preceding)))
        minimum=min(samples,key=lambda s:s["metrics"]["host_meminfo"]["value"]["MemAvailable"])
        first=next(h for h in hooks if h["stage"]=="DECODE_ENTRY" and h["index"]==0)
        start_metrics=first["preceding_external_metrics"]
        end_metrics=metrics(minimum)
        ranks.append(dict(rank=rank,largest_gap=gap,hooks=hooks,
            minimum_external_sample=dict(monotonic_ns=minimum["monotonic_ns"],metrics=end_metrics),
            first_decode_to_minimum_external_delta={k:end_metrics[k]-start_metrics[k] for k in end_metrics if end_metrics[k] is not None and start_metrics[k] is not None},
            memory_views_are_not_additive=True))
    left=read(HERE/"baseline-11976/worker-receipts/rank-0-003.json")["output_ids"]
    right=read(HERE/"observed-11977/worker-receipts/rank-0-003.json")["output_ids"]
    item=bundle["items"][3]
    require(item["index"]==3,"request identity")
    result=dict(schema="pireus-posthoc-phase-diagnostic-v1",
        negative_closure_sha256=hashlib.sha256((HERE/"negative-result.json").read_bytes()).hexdigest(),
        input_bundle_sha256=bundle_sha,post_hoc=True,causal_attribution=False,
        qualification_changed=False,new_inference_job=False,
        ranks=ranks,token_difference=dict(request_index=3,baseline_length=len(left),observed_length=len(right),
            differing_positions=[dict(offset=i,baseline=x,observed=y) for i,(x,y) in enumerate(zip(left,right)) if x!=y],
            frozen_sampling={k:v for k,v in item.items() if k not in ("input_ids","index")},
            decoded_semantics_verified=False),
        limitations=["Hook and preceding external metric reads are not atomic snapshots; age and duration are recorded.",
            "External sampling misses the guardian minimum; no interpolation to that value.",
            "PSS, CUDA allocator, host memory and cgroup views overlap and cannot be added.",
            "Read duration includes scheduler and kernel waiting; not measured observer CPU time.",
            "Same seed and temperature do not establish deterministic floating-point execution.",
            "No logits, observer-process RSS or ancestor-cgroup memory samples were captured.",
            "Fixed-order unequal-length arms cannot establish causal overhead."])
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
