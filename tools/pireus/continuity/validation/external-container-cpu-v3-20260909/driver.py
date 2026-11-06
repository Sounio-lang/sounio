import hashlib,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
sys.dont_write_bytecode=True
REPO=Path("/workspace/.wt/pireus-integration-20260906")
ROOT=Path("/workspace/.cache/pireus-continuity/external-container-cpu-v3-driver-20260909")
FROZEN=Path("/workspace/.cache/pireus-continuity/external-container-cpu-v3-freeze-20260909")
sys.path.insert(0,str(REPO/"tools/pireus/continuity/ops"))
import cpu_v3_attempt as a
contract=json.loads((ROOT/"driver-contract.json").read_bytes())
def event(stage,**extra):
 print(json.dumps(dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**extra)),flush=True)
def main():
 p=a.packet(FROZEN)
 began=time.monotonic()
 while time.monotonic()-began<6*3600:
  raw=subprocess.check_output(["gh","api","repos/Sounio-lang/sounio/actions/runs/34415603204"])
  ci=json.loads(raw)
  a.require(ci["head_sha"]==p["source_commit"],"CI source mismatch")
  if ci["status"]=="completed" and ci["conclusion"]!="success":
   event("STOP_SOURCE_CI",conclusion=ci["conclusion"]);return
  checks=json.loads(subprocess.check_output(["gh","api","repos/Sounio-lang/sounio/commits/"+p["source_commit"]+"/check-runs?per_page=100"]))["check_runs"]
  failed=[]
  for name in p["required_source_checks"]:
   rs=[r for r in checks if r["name"]==name and r["head_sha"]==p["source_commit"]]
   if rs:
    r=max(rs,key=lambda v:(v.get("started_at") or "",v["id"]))
    if r["status"]=="completed" and r["conclusion"]!="success":failed.append(name)
  if failed:event("STOP_SOURCE_CHECK",checks=failed);return
  if ci["status"]=="completed":
   try:a.readiness(FROZEN)
   except ValueError as exc:event("WAIT_SOURCE_CHECKS",reason=str(exc))
   else:break
  else:event("WAIT_SOURCE_CI",run=ci["id"],status=ci["status"])
  time.sleep(30)
 else:
  event("STOP_CI_WAIT_LIMIT");return
 a.require(a.orchestration_hashes()==contract["orchestration_sha256"],"orchestration changed while waiting")
 a.require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==contract["driver_sha256"],"driver changed")
 event("SOURCE_READY")
 rc=a.launch(FROZEN,ROOT/"attempt")
 event("ATTEMPT_TERMINAL",returncode=rc)
 rows=[]
 for line in (ROOT/"attempt/launch.log").read_bytes().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if isinstance(r,dict):rows.append(r)
 jobs={r["job"] for r in rows if r.get("stage")=="CPU_V3_RUNTIME_VERIFIED"}
 a.require(len(jobs)==1,"no unique allocated job receipt; manual accounting review required")
 job=jobs.pop()
 for _ in range(12):
  raw=a.kube("exec","slurm-pilot-controller-0","-c","slurmctld","--","sacct","-j",job,"--noheader","--parsable2","--format=JobID,JobName,State,ExitCode,NodeList,Start,End")
  try:a.accounting(raw,job)
  except ValueError:time.sleep(5)
  else:break
 else:raise ValueError("terminal accounting unavailable; no resubmission")
 a.collect(FROZEN,ROOT/"attempt",ROOT/"collected",job)
 pin=a.digest((ROOT/"collected/collection.json").read_bytes())
 event("COLLECTED",job=job,collection_sha256=pin)
 try:r=a.qualify(FROZEN,ROOT/"collected",pin)
 except (ValueError,AssertionError,KeyError) as exc:
  a.write(ROOT/"qualification-refusal.json",dict(job=job,collection_sha256=pin,error=type(exc).__name__+": "+str(exc),qualified=False))
  event("QUALIFICATION_REFUSED",job=job,error=str(exc))
 else:
  a.write(ROOT/"qualification.json",r);event("CPU_V3_QUALIFIED",job=job,loaded_model_overhead_qualified=False)
try:main()
except Exception as exc:
 event("DRIVER_STOP",error=type(exc).__name__+": "+str(exc),retry=False)
 traceback.print_exc()
finally:
 event("DRIVER_TERMINAL")
