#!/usr/bin/env python3
"""Freeze/verify a separately identified diagnostic; no allocation or inference."""
import argparse
import json
from pathlib import Path
import subprocess
from feedback_smoke import HERE, digest
from check_frozen_feedback import source_checks

SOURCE = "0cbf8267d42c4e8d2ea27ce4f18734259cb6b042"
PARENT_SHA = "c75f94a648e8dd52c599e5d1ce722888b7c33389e8201c2256e7316aacf5827b"
GENERATED_SHA = "8794694d22b4319a8e8df719eabdce9a89115fa2d1d11e630780bf3c9d867d5d"
PARENT = HERE/"validation/feedback-smoke-freeze-v2-20260909"
BUILD = HERE/"validation/lifecycle-diagnostic-20260909"

def specification():
    raw=(PARENT/"execution-freeze.json").read_bytes()
    if digest(raw)!=PARENT_SHA:
        raise ValueError("parent freeze identity mismatch")
    spec=json.loads(raw)
    generated=(BUILD/"offline_generate.py").read_bytes()
    if digest(generated)!=GENERATED_SHA:
        raise ValueError("diagnostic artifact identity mismatch")
    parent_source=spec["source_commit"]
    spec.update(schema="pireus-lifecycle-execution-freeze-v1",
        experiment="feedback-lifecycle-diagnostic-v1", source_commit=SOURCE,
        input_source_commit=parent_source, parent_freeze_sha256=PARENT_SHA,
        input_artifacts_unchanged=True, only_runtime_replacement="offline_generate.py",
        diagnostic=dict(host_interval_seconds=1, decode_observation_stride=16,
          journal="/scratch/pireus/receipts/lifecycle-JOB-RANK.jsonl",
          required_events=["OBSERVER_START","EXTEND_ENTRY","DECODE_ENTRY",
            "DECODE_SAMPLE","DECODE_EXIT","PROPOSAL_SAVED","CLEANUP_BEFORE",
            "CLEANUP_AFTER","REFERENCES_RELEASED","HOST_SAMPLE","OBSERVER_END"],
          missing_tail_is_missing=True, hardware_overhead_measured=False),
        runtime_snapshot_strategy="git source runtime; replace only offline_generate.py with pinned generated artifact",
        ci_acceptance=False, inference_completed=False, hardware_executed=False)
    spec["runtime_sha256"]["offline_generate.py"]=GENERATED_SHA
    return spec

def encoded(spec):
    return (json.dumps(spec,sort_keys=True,separators=(",",":"))+"\n").encode()

def verify(root):
    spec=specification()
    if (root/"execution-freeze.json").read_bytes()!=encoded(spec):
        raise ValueError("diagnostic freeze mismatch")
    for table,prefix in (("runtime_sha256","runtime/"),("files_sha256","")):
        for name,sha in spec[table].items():
            p=root/(prefix+name)
            if not p.resolve().is_relative_to(root.resolve()) or digest(p.read_bytes())!=sha:
                raise ValueError("artifact mismatch: "+prefix+name)
    return spec

def create(root):
    spec=specification()
    blobs={}
    for name,sha in spec["files_sha256"].items():
        raw=(PARENT/name).read_bytes()
        if digest(raw)!=sha: raise ValueError("input changed: "+name)
        blobs[name]=raw
    for name,sha in spec["runtime_sha256"].items():
        if name=="offline_generate.py":
            raw=(BUILD/name).read_bytes()
        else:
            raw=subprocess.check_output(["git","show",SOURCE+":tools/pireus/continuity/runtime/"+name],cwd=HERE)
        if digest(raw)!=sha: raise ValueError("runtime changed: "+name)
        blobs["runtime/"+name]=raw
    root.mkdir(exist_ok=False)
    for name,raw in blobs.items():
        p=root/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(raw)
        p.chmod(0o555 if p.suffix==".sh" else 0o444)
    (root/"execution-freeze.json").write_bytes(encoded(spec))
    return verify(root)

def readiness(root):
    spec=verify(root)
    raw=subprocess.check_output(["gh","api","--paginate",
        "repos/Sounio-lang/sounio/commits/"+SOURCE+"/check-runs",
        "--jq",".check_runs[] | @json"],cwd=HERE,text=True,timeout=60)
    checks=source_checks(spec,[json.loads(x) for x in raw.splitlines() if x.strip()])
    return dict(schema="pireus-lifecycle-readiness-v1",source_commit=SOURCE,
        freeze_sha256=digest(encoded(spec)),source_checks=checks,
        helper_sha256=digest(Path(__file__).read_bytes()),
        source_and_inputs_ready=True,live_host_preflight_required=True,
        hardware_executed=False,diagnostic_complete=False,pilot_acceptance=False)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["create","verify","readiness"])
    p.add_argument("root",type=Path)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.mode=="create": create(args.root)
    elif args.mode=="verify": verify(args.root)
    else:
        if args.output is None: p.error("readiness requires --output")
        result=readiness(args.root)
        with args.output.open("x") as out: out.write(json.dumps(result,indent=2)+"\n")
