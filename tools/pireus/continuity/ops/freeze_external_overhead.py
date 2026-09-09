#!/usr/bin/env python3
"""Freeze a paired loaded-model screening; never allocate or run inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from check_frozen_feedback import source_checks

HERE=Path(__file__).resolve().parents[1]
SOURCE="054b7d672d3e5f6546c1e807b695f4ca78690ede"
PARENT=HERE/"validation/lifecycle-diagnostic-freeze-20260909"
PARENT_SHA="6510a1fc1e9fb35131efedc6424ccedcb0dba2ecac38b3a3c178ebc063ba50e2"
BUILD="tools/pireus/continuity/validation/external-container-cpu-v2-20260909/"
BUILD_SHA="6e72e268327e08d1fbb117296fc5e812c84b93e086d25e810f2cc787ac9e82a3"
def digest(raw):return hashlib.sha256(raw).hexdigest()
def encoded(x):return (json.dumps(x,sort_keys=True,separators=(",",":"))+"\n").encode()
def git_blob(source,path):
    return subprocess.check_output(["git","show",source+":"+path],cwd=HERE)

def specification():
    raw=(PARENT/"execution-freeze.json").read_bytes()
    if digest(raw)!=PARENT_SHA:raise ValueError("parent freeze changed")
    parent=json.loads(raw)
    raw=git_blob(SOURCE,BUILD+"manifest.json")
    if digest(raw)!=BUILD_SHA:raise ValueError("integration build changed")
    build=json.loads(raw)
    baseline=dict(parent["runtime_sha256"])
    observed=baseline|build["files_sha256"]
    inputs={n:sha for n,sha in parent["files_sha256"].items() if n.startswith("without-feedback/")}
    return dict(schema="pireus-external-overhead-freeze-v1",
        source_commit=SOURCE,parent_source_commit=parent["source_commit"],
        parent_freeze_sha256=PARENT_SHA,integration_build_sha256=BUILD_SHA,
        experiment="external-observer-loaded-overhead-screen-v1",
        arm_order=["baseline","observed"],input_arm="without-feedback",
        batch_size_per_arm=8,files_sha256=inputs,
        runtime_sha256={"baseline":baseline,"observed":observed},
        required_source_checks=parent["required_source_checks"],
        required_execution_profile=parent["required_execution_profile"],
        model_image_lock=parent["model_image_lock"],
        acceptance=dict(
            both_arms_complete_requests=8,rank_output_byte_parity=True,
            cross_arm_output_byte_parity=True,
            guardian_reserve_gib=33,protected_floor_gib=32,
            maximum_loss_minimum_host_available_bytes=256*1024**2,
            maximum_decode_elapsed_ratio=1.10,
            observer_maximum_sample_gap_ns=500_000_000,
            observer_required_coverage="before model entry through target exit on both ranks",
            required_external_metrics=["host_meminfo","process_smaps_rollup",
                "cgroup_memory.current","cgroup_memory.stat","cgroup_memory.events",
                "cgroup_memory.events.local","cgroup_memory.max","cgroup_memory.high","cgroup_memory.swap.max"],
            complete_lifecycle_journals_both_arms=True,
            no_new_oom_events=True,
            decode_interval="first DECODE_ENTRY monotonic_ns to last DECODE_EXIT monotonic_ns per rank, including inter-request gaps",
            memory_comparison="observed guardian minimum available bytes minus baseline, per rank",
            missing_or_incomplete_evidence="FAIL_OR_INCONCLUSIVE_NEVER_PASS"),
        screening_limits="engineering bounds declared prospectively; one fixed-order pair cannot establish general overhead or causality",
        operations=dict(exclusive_pair=True,fresh_preflight_before_each_arm=True,
            empty_pair_queue_required=True,remote_tmux_required=True,
            no_concurrent_other_pair_workload=True,maximum_attempts_per_arm=1,
            stop_after_any_failed_arm=True,automatic_retry=False,
            worker_uid_boot_continuity=True,no_guard_floor_cache_or_allocator_changes=True),
        inference_executed=False,loaded_model_overhead_qualified=False,pilot_acceptance=False)

def create(root):
    spec=specification();blobs={}
    for name,sha in spec["files_sha256"].items():
        raw=(PARENT/name).read_bytes()
        if digest(raw)!=sha:raise ValueError("input changed: "+name)
        blobs[name]=raw
    generated=json.loads(git_blob(SOURCE,BUILD+"manifest.json"))["files_sha256"]
    for arm,files in spec["runtime_sha256"].items():
        for name,sha in files.items():
            if arm=="observed" and name in generated:raw=git_blob(SOURCE,BUILD+name)
            elif name=="offline_generate.py":
                raw=git_blob(spec["parent_source_commit"],"tools/pireus/continuity/validation/lifecycle-diagnostic-20260909/"+name)
            else:raw=git_blob(spec["parent_source_commit"],"tools/pireus/continuity/runtime/"+name)
            if digest(raw)!=sha:raise ValueError("runtime changed: "+arm+"/"+name)
            blobs[arm+"/runtime/"+name]=raw
    root.mkdir(parents=True,exist_ok=False)
    for name,raw in blobs.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        p.chmod(0o555 if p.suffix==".sh" else 0o444)
    (root/"execution-freeze.json").write_bytes(encoded(spec))
    return verify(root)

def verify(root):
    spec=specification()
    if (root/"execution-freeze.json").read_bytes()!=encoded(spec):raise ValueError("freeze changed")
    tables=[(spec["files_sha256"],"")]+[(table,arm+"/runtime/") for arm,table in spec["runtime_sha256"].items()]
    for table,prefix in tables:
        for name,sha in table.items():
            p=root/(prefix+name)
            if not p.resolve().is_relative_to(root.resolve()) or digest(p.read_bytes())!=sha:
                raise ValueError("artifact mismatch: "+prefix+name)
    return spec

def readiness(root):
    spec=verify(root)
    raw=subprocess.check_output(["gh","api","--paginate","repos/Sounio-lang/sounio/commits/"+SOURCE+"/check-runs",
        "--jq",".check_runs[] | @json"],cwd=HERE,text=True,timeout=60)
    checks=source_checks(spec,[json.loads(s) for s in raw.splitlines() if s.strip()])
    return dict(schema="pireus-external-overhead-readiness-v1",source_commit=SOURCE,
        freeze_sha256=digest(encoded(spec)),helper_sha256=digest(Path(__file__).read_bytes()),
        source_checks=checks,source_and_inputs_ready=True,live_preflight_required=True,
        inference_executed=False,pilot_acceptance=False)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["create","verify","readiness"])
    p.add_argument("root",type=Path);p.add_argument("--output",type=Path);a=p.parse_args()
    if a.mode=="create":create(a.root)
    elif a.mode=="verify":verify(a.root)
    else:
        if a.output is None:p.error("--output required")
        r=readiness(a.root)
        with a.output.open("x") as out:out.write(json.dumps(r,indent=2)+"\n")
