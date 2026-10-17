import hashlib,json,os,resource,signal,socket,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/"manifest.json").read_bytes())
for name,sha in manifest["files"].items():
 if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError("source bytes changed: "+name)
job=os.environ["SLURM_JOB_ID"]
start={"job":job,"hostname":socket.gethostname(),"boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),"manifest_sha256":hashlib.sha256((ROOT/"manifest.json").read_bytes()).hexdigest(),"scope":"raw compiler capacity timing; not PIREUS qualification","timeout_seconds":600}
(ROOT/"start.json").write_text(json.dumps(start,indent=2)+"\n")
print(json.dumps(start),flush=True)
began=time.monotonic()
with (ROOT/"compile.log").open("xb") as log:
 p=subprocess.Popen([str(ROOT/"madaros"),str(ROOT/"boundary_main.sio"),"-o",str(ROOT/"boundary.elf")],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
 timed_out=False
 try:rc=p.wait(timeout=600)
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(p.pid,signal.SIGTERM)
  try:rc=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
elapsed=time.monotonic()-began
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
result=dict(start,compiler_rc=rc,timed_out=timed_out,compile_seconds=elapsed,compiler_maxrss_kib=usage.ru_maxrss,compiler_user_seconds=usage.ru_utime,compiler_system_seconds=usage.ru_stime,artifact_exists=(ROOT/"boundary.elf").exists(),runtime_rc=None,qualified=False)
if rc==0 and result["artifact_exists"]:
 (ROOT/"boundary.elf").chmod(0o755)
 with (ROOT/"run.log").open("xb") as log:
  run=subprocess.run(["timeout","10",str(ROOT/"boundary.elf")],stdout=log,stderr=subprocess.STDOUT)
 result["runtime_rc"]=run.returncode
 result["artifact_sha256"]=hashlib.sha256((ROOT/"boundary.elf").read_bytes()).hexdigest()
result["raw_boundary_witness_pass"]=rc==0 and result["runtime_rc"]==7
(ROOT/"result.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
