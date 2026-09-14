#!/usr/bin/env python3
"""Generate a separate launch profile; qualified runtime files are never edited."""
import argparse
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
SERVE_SHA="e8e44e63d2662f93c38f44018c23ec4a4aa389976458154c83420da6b6640b7c"
GUARD_SHA="7a5f24ef3863cc1b12c9a219ec25bf7b1a952a21624721ce0c62f58f777abbd8"
CONTAINER_SHA="4a1a5cc6825752c864b27ea6669f380215f35a45f66907ab1a5db743cabafdc1"
ENTRY_SHA="8794694d22b4319a8e8df719eabdce9a89115fa2d1d11e630780bf3c9d867d5d"
OBSERVER_SHA="f4f64cceb4fd29fe85ac2f236e4a4c723071c5b3189f80ed62cd5a7178eb99bd"
def digest(raw):return hashlib.sha256(raw).hexdigest()

def generate(raw):
    if digest(raw)!=SERVE_SHA:raise ValueError("serve-rank source identity mismatch")
    source=raw.decode()
    old="  entrypoint=(/scratch/pireus/runtime/offline_generate.py)"
    new="  entrypoint=(/scratch/pireus/runtime/publish_observer_target.py --entry /scratch/pireus/runtime/offline_generate.py --entry-sha "+ENTRY_SHA+" --)"
    anchor='exec python3 /scratch/pireus/runtime/memory_guard.py "${guard_args[@]}" --'
    replacement=('exec python3 /scratch/pireus/runtime/external_rank_supervisor.py '
       '--output "/scratch/pireus/receipts/external-${SLURM_JOB_ID}-${PIREUS_RANK}" '
       '--worker-uid "${PIREUS_EXTERNAL_WORKER_UID:?declared worker UID required}" '
       '--boot-id "${PIREUS_EXTERNAL_BOOT_ID:?declared boot ID required}" '
       '--entry-sha '+ENTRY_SHA+' -- python3 /scratch/pireus/runtime/memory_guard.py "${guard_args[@]}" --')
    if source.count(old)!=1 or source.count(anchor)!=1:
        raise ValueError("launch integration anchor mismatch")
    # This generated profile is offline-only; do not silently affect HTTP/other modes.
    mode_guard='[[ "${PIREUS_OFFLINE_MODE:-}" == "generate" ]] || { echo "external profile requires offline generate" >&2; exit 76; }\n'
    return source.replace("set -euo pipefail\n","set -euo pipefail\n"+mode_guard,1).replace(old,new,1).replace(anchor,replacement,1)

def build(output):
    raw=(BASE/"runtime/serve_rank.sh").read_bytes()
    entry=(BASE/"validation/lifecycle-diagnostic-20260909/offline_generate.py").read_bytes()
    guard=(BASE/"runtime/memory_guard.py").read_bytes()
    container=(BASE/"runtime/run_in_container.sh").read_bytes()
    observer=(BASE/"ops/external_memory_observer.py").read_bytes()
    for value,pin in ((entry,ENTRY_SHA),(guard,GUARD_SHA),(container,CONTAINER_SHA),(observer,OBSERVER_SHA)):
        if digest(value)!=pin:raise ValueError("integration dependency identity mismatch")
    files={"serve_rank.sh":generate(raw).encode(),"offline_generate.py":entry,
           "memory_guard.py":guard,"run_in_container.sh":container,"external_memory_observer.py":observer}
    for name in ("publish_observer_target.py","external_rank_supervisor.py"):
        files[name]=(BASE/"ops"/name).read_bytes()
    output.mkdir(exist_ok=False)
    for name,value in files.items():(output/name).write_bytes(value)
    manifest=dict(schema="pireus-external-observer-integration-build-v1",
        identity="feedback-lifecycle-external-observer-deadline-v3",
        observer_profile="external-observer-deadline-v3",observer_schema="pireus-external-memory-observation-v2",
        scheduling="actual-start-deadline-no-catchup",observer_resource_scope="observer-process-only",files_sha256={n:digest(b) for n,b in files.items()},
        base_serve_sha256=SERVE_SHA,entry_sha256=ENTRY_SHA,guard_sha256=GUARD_SHA,
        original_guard_unchanged=True,original_container_launcher_unchanged=True,
        lifecycle_inference_algorithm_unchanged=True,offline_only=True,
        startup_ack_seconds=30,external_interval_seconds=0.2,external_run_limit_seconds=3500,
        full_runtime_freeze_complete=False,source_ci_qualified=False,
        container_control_qualified=False,inference_executed=False,loaded_model_overhead_qualified=False)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    build(p.parse_args().output)
