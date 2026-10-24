import json,sys,time,subprocess,traceback,hashlib
from pathlib import Path
ops=Path("/workspace/.wt/pireus-integration-20260906/tools/pireus/continuity/ops")
sys.path.insert(0,str(ops))
from freeze_external_overhead import verify,SOURCE
from launch_external_overhead import launch
root=Path("/workspace/.cache/pireus-continuity/external-overhead-driver-20260909-v2")
frozen=Path("/workspace/.cache/pireus-continuity/external-overhead-freeze-20260909-v2")
expected=json.loads((root/"driver-freeze.json").read_bytes())
def check():
 for n,sha in expected["helper_sha256"].items():
  assert hashlib.sha256((ops/n).read_bytes()).hexdigest()==sha,"orchestration helper changed"
 assert SOURCE==expected["source_commit"]
 verify(frozen)
def emit(stage,**data):
 print(json.dumps(dict(stage=stage,utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**data)),flush=True)
try:
 check();deadline=time.monotonic()+7200
 while True:
  raw=subprocess.check_output(["gh","api","repos/Sounio-lang/sounio/actions/runs/34402455439"],timeout=60)
  r=json.loads(raw);assert r["head_sha"]==SOURCE and r["id"]==34402455439
  (root/"ci-latest.json").write_bytes(raw)
  emit("CI_OBSERVED",status=r["status"],conclusion=r["conclusion"])
  if r["status"]=="completed":
   if r["conclusion"]!="success":raise ValueError("exact source CI did not pass")
   break
  if time.monotonic()>=deadline:raise ValueError("CI observation window ended; no inference submitted")
  time.sleep(45)
 check()
 emit("BASELINE_PREPARING")
 rc=launch(frozen,"baseline",root/"baseline-stage")
 emit("BASELINE_TERMINAL",returncode=rc)
 (root/"driver-result.json").write_text(json.dumps(dict(stage="BASELINE_TERMINAL",returncode=rc,observed_arm_started=False))+"\n")
except Exception as exc:
 emit("DRIVER_STOPPED",error=type(exc).__name__+": "+str(exc))
 traceback.print_exc()
 (root/"driver-result.json").write_text(json.dumps(dict(stage="DRIVER_STOPPED",error=str(exc),observed_arm_started=False))+"\n")
 raise
