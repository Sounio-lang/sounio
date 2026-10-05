#!/usr/bin/env python3
"""Materialize the exact v3 paired-screen packet; never submit inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import evaluate_external_overhead_v3 as evaluator
from check_frozen_feedback import source_checks
HERE = Path(__file__).resolve().parents[1]
PREPARATION = HERE / "validation/external-overhead-v3-preparation-20260910"
HELPERS = (
    "ops/freeze_external_overhead_v3.py", "ops/launch_external_overhead_v3.py",
    "ops/external_overhead_custody_v3.py", "ops/evaluate_external_overhead_v3.py",
    "ops/evaluate_external_overhead.py", "ops/cpu_memory_control_v2.py",
    "ops/check_frozen_feedback.py", "runtime/preflight.py", "tokenized_cycle.py",
)
def digest(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return (json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()
def require(ok,message):
    if not ok: raise ValueError(message)
def orchestration_hashes():
    return {name:digest((HERE/name).read_bytes()) for name in HELPERS}
def specification():
    original = evaluator.specification()
    receipt=HERE/"validation/external-container-cpu-v3-20260909/attempt-11982/qualification.json"
    raw=receipt.read_bytes()
    require(digest(raw)==original["cpu_prerequisite"]["qualification_sha256"],"CPU prerequisite changed")
    cpu=json.loads(raw)
    require(cpu["job"]=="11982" and cpu["cpu_container_control_qualified"] is True
            and cpu["custody_verified"] is True
            and cpu["source_commit"]==original["source_commit"],"CPU prerequisite not qualified")
    spec = copy.deepcopy(original)
    spec.update(schema="pireus-external-overhead-freeze-v3",
                state="FROZEN_NOT_EXECUTED", execution_ready=True,
                screening_specification=original,
                orchestration_sha256=orchestration_hashes())
    spec.pop("remaining_execution_gates")
    return spec
def inventory():
    pins=json.loads((PREPARATION/"manifest.json").read_bytes())
    raw=(PREPARATION/"runtime-git-inventory.json").read_bytes()
    require(digest(raw)==pins["runtime-git-inventory.json"],"runtime inventory changed")
    return json.loads(raw)
def create(root):
    spec=specification();blobs={}
    table=inventory()
    for arm,files in spec["runtime_sha256"].items():
        for name,pin in files.items():
            ref=table[arm+"/"+name]
            require(ref["sha256"]==pin,"inventory pin mismatch")
            raw=subprocess.check_output(["git","show",ref["commit"]+":"+ref["path"]],cwd=HERE)
            require(digest(raw)==pin,"runtime blob mismatch")
            blobs[arm+"/runtime/"+name]=raw
    parent=HERE/"validation/lifecycle-diagnostic-freeze-20260909"
    for name,pin in spec["files_sha256"].items():
        raw=(parent/name).read_bytes()
        require(digest(raw)==pin,"input mismatch")
        blobs[name]=raw
    root.mkdir(parents=True,exist_ok=False)
    for name,raw in blobs.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw);path.chmod(0o555 if path.suffix==".sh" else 0o444)
    (root/"execution-freeze.json").write_bytes(encoded(spec))
    return verify(root)
def verify(root):
    require(__debug__,"optimized Python disables inherited barriers")
    spec=specification()
    require((root/"execution-freeze.json").read_bytes()==encoded(spec),"freeze/helper identity changed")
    expected=dict(spec["files_sha256"])
    for arm,files in spec["runtime_sha256"].items():
        expected.update({arm+"/runtime/"+name:pin for name,pin in files.items()})
    actual={str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(actual-set(expected)-{"execution-freeze.json"}==set(),"unexpected frozen artifact")
    for name,pin in expected.items():
        path=root/name
        require(path.resolve().is_relative_to(root.resolve()) and digest(path.read_bytes())==pin,"frozen artifact mismatch: "+name)
    return spec
def readiness(root):
    spec=verify(root)
    raw=subprocess.check_output(["gh","api","--paginate","repos/Sounio-lang/sounio/commits/"+spec["source_commit"]+"/check-runs",
                                 "--jq",".check_runs[] | @json"],cwd=HERE,text=True,timeout=60)
    checks=source_checks(spec,[json.loads(s) for s in raw.splitlines() if s.strip()])
    return dict(source_commit=spec["source_commit"],freeze_sha256=digest((root/"execution-freeze.json").read_bytes()),
                orchestration_sha256=spec["orchestration_sha256"],source_checks=checks,
                source_and_inputs_ready=True,live_preflight_required=True,inference_executed=False)
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["create","verify","readiness"])
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    result=globals()[args.mode](args.root)
    print(json.dumps(result,indent=2))
