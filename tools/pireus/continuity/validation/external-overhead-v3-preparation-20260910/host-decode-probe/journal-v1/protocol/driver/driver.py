#!/usr/bin/env python3
"""Pinned single-attempt journal diagnostic and terminal collection."""
import hashlib,json,os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
BASE=Path("/workspace/.wt/pireus-integration-20260906/tools/pireus/continuity")
ROOT=Path("/workspace/.cache/pireus-continuity/host-decode-journal-v1-run-20260910")
FROZEN=Path("/workspace/.cache/pireus-continuity/host-decode-journal-v1-freeze-20260910")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def save(name,value):
    with (ROOT/name).open("x") as out:json.dump(value,out,indent=2);out.write("\n")
def main():
    require(__debug__,"checked Python required")
    require(sys.argv[1:] in (["--check"],["--execute"]),"choose --check or --execute")
    contract=json.loads((ROOT/"contract.json").read_bytes())
    require(sha(Path(__file__))==contract["driver_sha256"],"driver changed")
    require(sha(Path(contract["slurm_conf"]))==contract["slurm_conf_sha256"],"Slurm configuration changed")
    require(sha(FROZEN/"execution-freeze.json")==contract["freeze_sha256"],"freeze changed")
    for name,pin in contract["orchestration_sha256"].items():
        require(sha(BASE/name)==pin,"orchestration changed: "+name)
    sys.path.insert(0,str(BASE/"ops"))
    import host_decode_journal_attempt as adapter
    attempt=adapter.context()
    spec=attempt.verify(FROZEN)
    require(spec["source_commit"]==contract["source_commit"],"source changed")
    if sys.argv[1:]==["--check"]:
        print(json.dumps(dict(packet_verified=True,source_commit=spec["source_commit"],
                             source_ci_checked=False,hardware_submitted=False)))
        return 0
    require(os.environ.get("TMUX"),"remote tmux required")
    require(not (ROOT/"attempt-entered.json").exists(),"attempt already entered; no retry")
    os.environ.update(SLURM_CONF=contract["slurm_conf"],PYTHONOPTIMIZE="0",PYTHONDONTWRITEBYTECODE="1")
    ready=attempt.readiness(FROZEN)
    save("attempt-entered.json",dict(readiness=ready,automatic_retry=False))
    rc=1
    try:
        rc=attempt.launch(FROZEN,"baseline",ROOT/"stage")
    except Exception as exc:
        save("launch-error.json",dict(error=type(exc).__name__,message=str(exc),automatic_retry=False))
    finally:
        save("launch-exit.json",dict(returncode=rc,automatic_retry=False))
    # Collect only a completed launcher invocation with one attributable job.
    # A launch refusal without exit-code is terminal; never infer a submission.
    if not (ROOT/"stage/exit-code").exists():
        return rc
    jobs=set()
    for line in (ROOT/"stage/launch.log").read_bytes().splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if isinstance(row,dict) and row.get("stage")=="OVERHEAD_RUNTIME_VERIFIED":
            jobs.add(row.get("job"))
    require(len(jobs)==1 and all(isinstance(job,str) and job.isdigit() for job in jobs),"unique job identity required")
    job=jobs.pop()
    try:
        adapter.collect(FROZEN,ROOT/"stage",ROOT/"collection",job)
        pin=sha(ROOT/"collection/collection.json")
        save("collection-pin.json",dict(sha256=pin,job=job))
        result=adapter.inspect(FROZEN,ROOT/"collection",pin)
        save("inspection.json",result)
    except Exception as exc:
        save("collection-or-inspection-error.json",dict(job=job,error=type(exc).__name__,message=str(exc),
             qualified=False,automatic_retry=False))
        return rc if rc else 1
    return rc
if __name__=="__main__":raise SystemExit(main())
