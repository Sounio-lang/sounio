import hashlib,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path("/workspace/.cache/pireus-continuity/external-container-cpu-v3-driver-20260909")
SOURCE="6e470eab01d8ba3645765e82fc604d75975cdb7f"
def event(stage,**extra):
 print(json.dumps(dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**extra)),flush=True)
def main():
 contract=json.loads((ROOT/"driver-contract.json").read_bytes())
 if (ROOT/"attempt").exists():raise ValueError("existing hardware attempt; no retry")
 deadline=time.monotonic()+6*3600
 while time.monotonic()<deadline:
  run=json.loads(subprocess.check_output(["gh","api","repos/Sounio-lang/sounio/actions/runs/34415603204"]))
  if run["head_sha"]!=SOURCE or run["run_attempt"]!=2:raise ValueError("CI identity changed")
  if run["status"]=="completed":
   if run["conclusion"]!="success":
    event("STOP_SOURCE_CI_ATTEMPT_2",conclusion=run["conclusion"]);return
   break
  event("WAIT_SOURCE_CI_ATTEMPT_2",status=run["status"])
  time.sleep(30)
 else:raise ValueError("CI recovery observation limit; no hardware launch")
 if (ROOT/"attempt").exists():raise ValueError("existing hardware attempt; no retry")
 if hashlib.sha256((ROOT/"driver.py").read_bytes()).hexdigest()!=contract["driver_sha256"]:raise ValueError("original driver changed")
 event("RESUME_ORIGINAL_DRIVER",hardware_attempts_before=0)
 with (ROOT/"driver.log").open("ab") as log:
  rc=subprocess.call([sys.executable,"-u",str(ROOT/"driver.py")],stdout=log,stderr=subprocess.STDOUT)
 event("ORIGINAL_DRIVER_EXITED",exit_code=rc,qualification_not_inferred=True)
try:main()
except Exception as exc:
 event("RECOVERY_STOP",error=type(exc).__name__+": "+str(exc),hardware_retry=False)
finally:event("RECOVERY_TERMINAL")
