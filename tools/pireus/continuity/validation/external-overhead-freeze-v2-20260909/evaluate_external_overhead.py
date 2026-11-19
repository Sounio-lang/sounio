#!/usr/bin/env python3
"""Numerical/semantic checks only; collection and runtime custody are separate gates."""
import json
from cpu_memory_control_v2 import fields

def require(ok,message):
    if not ok:raise ValueError(message)

def response_parity(left,right,left_job,right_job):
    require(left_job!=right_job,"distinct jobs required")
    decoded=[]
    for job,ranks in ((left_job,left),(right_job,right)):
        require(len(ranks)==2 and all(len(r)==8 for r in ranks),"complete paired outputs required")
        parsed=[]
        for index,(a,b) in enumerate(zip(*ranks)):
            require(a==b,"within-job rank bytes differ")
            value=json.loads(a)
            require(value["job"]==job and value["index"]==index,"response identity mismatch")
            require(value["completion_tokens"]==len(value["output_ids"])>0,"token count mismatch")
            require(value["sampling_authority_rank"]==0,"sampling authority mismatch")
            # No other field may be excluded, including revision/profile/input hash.
            parsed.append({k:v for k,v in value.items() if k!="job"})
        decoded.append(parsed)
    require(decoded[0]==decoded[1],"cross-job response content differs")
    return True

def lifecycle_span(rows,job,rank):
    require(rows and rows[0]["stage"]=="OBSERVER_START" and rows[-1]["stage"]=="OBSERVER_END","lifecycle incomplete")
    require(all(r["job"]==job and r["rank"]==str(rank) and r["schema"]=="pireus-lifecycle-observation-v1" for r in rows),"lifecycle identity")
    require(len({r["pid"] for r in rows})==1,"lifecycle PID drift")
    # Host sampler rows may interleave with hooks; use each hook's own timestamp.
    ordered=("EXTEND_ENTRY","DECODE_ENTRY","DECODE_EXIT","PROPOSAL_SAVED","CLEANUP_BEFORE","CLEANUP_AFTER","REFERENCES_RELEASED")
    by_stage={stage:[r for r in rows if r["stage"]==stage] for stage in ordered}
    require(all([r["index"] for r in v]==list(range(8)) for v in by_stage.values()),"lifecycle request coverage")
    hooks=[by_stage[stage][i] for i in range(8) for stage in ordered]
    require(all(a["monotonic_ns"]<=b["monotonic_ns"] for a,b in zip(hooks,hooks[1:])),"lifecycle order")
    span=by_stage["DECODE_EXIT"][-1]["monotonic_ns"]-by_stage["DECODE_ENTRY"][0]["monotonic_ns"]
    require(span>0,"invalid decode interval")
    return span

def external_coverage(rows,lifecycle,job,rank,acceptance):
    require(rows and rows[0]["stage"]=="OBSERVER_START" and rows[-1]["stage"]=="TARGET_INVALIDATED"
            and rows[-1]["metrics"] is None,"external tail missing")
    samples=[r for r in rows if r["stage"]=="SAMPLE"]
    require(bool(samples),"no external samples")
    first=samples[0]
    keys=("job","rank","target_pid","observer_pid","binding_sha256")
    require(all(all(r[k]==first[k] for k in keys) for r in rows),"external identity drift")
    require(first["job"]==job and first["rank"]==str(rank) and first["target_pid"]==lifecycle[0]["pid"]
            and first["observer_pid"]!=first["target_pid"],"external target identity")
    require(first["monotonic_ns"]<lifecycle[0]["monotonic_ns"] and rows[-1]["monotonic_ns"]>=lifecycle[-1]["monotonic_ns"],"external coverage boundary")
    times=[s["monotonic_ns"] for s in samples]+[rows[-1]["monotonic_ns"]]
    require(all(0<b-a<=acceptance["observer_maximum_sample_gap_ns"] for a,b in zip(times,times[1:])),"external sampling gap")
    for s in samples:
        require(s["identity_valid"],"external identity invalid")
        for key in acceptance["required_external_metrics"]:
            metric=s["metrics"][key]
            require(metric["error"] is None and metric["value"] is not None,"external metric missing: "+key)
        require("MemAvailable" in s["metrics"]["host_meminfo"]["value"]
                and "Pss" in s["metrics"]["process_smaps_rollup"]["value"],"required numeric fields missing")
    for name in ("cgroup_memory.events","cgroup_memory.events.local"):
        tables=[fields(s["metrics"][name]["value"]) for s in samples]
        for key in ("oom","oom_kill","oom_group_kill"):
            require(all(key in t and t[key]==tables[0][key] for t in tables),"OOM increment or missing counter")
    return {"samples":len(samples),"maximum_gap_ns":max(b-a for a,b in zip(times,times[1:]))}

def evaluate(spec,baseline,observed):
    """Each arm supplies raw response bytes, journals, and parsed guardian log rows.
    Caller MUST independently verify immutable collection/runtime hashes, scheduler,
    worker identity, acknowledgement and completion-receipt custody.
    """
    require(spec["schema"]=="pireus-external-overhead-freeze-v2","wrong protocol")
    require(spec["acceptance"]["cross_arm_response_equality_except_fields"]==["job"],"parity exclusions changed")
    a=spec["acceptance"]
    response_parity(baseline["responses"],observed["responses"],baseline["job"],observed["job"])
    result=[]
    for rank in (0,1):
        spans=[];minimums=[]
        for arm in (baseline,observed):
            spans.append(lifecycle_span(arm["lifecycle"][rank],arm["job"],rank))
            guards=[r for r in arm["guardian"] if r.get("stage")=="MEMORY_GUARD_CHILD_EXIT"
                    and r.get("job")==arm["job"] and r.get("rank")==str(rank)]
            require(len(guards)==1 and guards[0]["returncode"]==0,"guardian completion")
            minimum=guards[0]["minimum_bytes"]
            require(type(minimum) is int and minimum>=a["guardian_reserve_gib"]*1024**3,"guardian minimum")
            minimums.append(minimum)
        coverage=external_coverage(observed["external"][rank],observed["lifecycle"][rank],observed["job"],rank,a)
        loss=minimums[0]-minimums[1];ratio=spans[1]/spans[0]
        require(loss<=a["maximum_loss_minimum_host_available_bytes"],"memory overhead exceeds bound")
        require(ratio<=a["maximum_decode_elapsed_ratio"],"decode overhead exceeds bound")
        result.append(dict(rank=rank,memory_loss_bytes=loss,decode_ratio=ratio,external=coverage))
    return dict(numerical_semantic_screen_pass=True,ranks=result,
                custody_required_separately=True,loaded_model_overhead_qualified=False,
                pilot_acceptance=False)
