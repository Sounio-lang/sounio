#!/usr/bin/env python3
"""Workspace coordinator for job-bound material observation and Sounio decisions."""
import argparse,fcntl,hashlib,io,json,os,subprocess,tarfile,shlex
from runtime.preflight import check_pair
from pathlib import Path
from cycle import atomic,encoded,digest,event,verify,proposal_template
HERE=Path(__file__).resolve().parent
NODES=["gpuorangefs-multi-spark-3c59","gpuorangefs-multi-spark-8e54"]

def kube(*args):
 return subprocess.check_output(["kubectl",*args])

def workers():
 if kube("-n","beagle","get","lease","pireus-spark-pair","-o","jsonpath={.spec.holderIdentity}").decode()!="slurm-owned":
  raise ValueError("Spark lease is not slurm-owned")
 pods=json.loads(kube("-n","slurm-pilot","get","pods","-o","json"))["items"]
 result=[]
 for node in ["spark-3c59","spark-8e54"]:
  matches=[p for p in pods if p["spec"].get("nodeName")==node and p["metadata"]["name"].startswith("slurm-pilot-worker-spark-") and p["status"]["phase"]=="Running"]
  if len(matches)!=1:raise ValueError("expected one worker per Spark")
  result.append(matches[0]["metadata"]["name"])
 return result

def remote(pod,*args,stdin=None):
 command=["kubectl","-n","slurm-pilot","exec",*(["-i"] if stdin is not None else []),pod,"-c","slurmd","--",*args]
 return subprocess.check_output(command,input=stdin)

def stage(pods,root,remote_root,names):
 data=io.BytesIO()
 with tarfile.open(fileobj=data,mode="w") as archive:
  for name in names:
   payload=(root/name).read_bytes();info=tarfile.TarInfo(name);info.size=len(payload)
   archive.addfile(info,io.BytesIO(payload))
 program="""import sys,tarfile,os
from pathlib import Path,PurePosixPath
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True);os.chown(root,1000,1000)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for member in archive:
  rel=PurePosixPath(member.name)
  assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
  p=root/member.name;p.parent.mkdir(parents=True,exist_ok=True)
  data=archive.extractfile(member).read()
  if p.exists():assert p.read_bytes()==data,'immutable staged artifact differs: '+str(p)
  else:p.write_bytes(data);p.chmod(0o444)
"""
 for pod in pods:
  remote(pod,"python3","-c",program,remote_root,stdin=data.getvalue())
  for name in ["gpu_probe.py","gpu_benchmark.py"]:
   remote(pod,"python3","-c","from pathlib import Path;import sys;p=Path(sys.argv[1]);p.write_bytes(sys.stdin.buffer.read())","/scratch/pireus/runtime/"+name,stdin=(HERE/name).read_bytes())

def run_gpu(root,remote_root,phase):
 log=root/(phase+".log")
 with log.open("ab") as output:
  env=os.environ.copy();env["TMPDIR"]="/tmp"
  argv=["srun","-p","gpu-orangefs","-w",",".join(NODES),"--nodes=2","--ntasks=2","--ntasks-per-node=1","-c8","--gres=gpu:gb10:1","--mem=12G","--exclusive","--propagate=NONE","--kill-on-bad-exit=1","--time=30","--job-name=pireus-"+phase,
        "/scratch/pireus/runtime/run_in_container.sh","python3","/scratch/pireus/runtime/"+("gpu_probe.py" if phase=="probe" else "gpu_benchmark.py"),"--run",remote_root]
  expected=json.loads((HERE/"runtime/runtime-lock.json").read_text())["image"]["derived_sif_sha256"]
  payload="set -euo pipefail\n"+"echo "+shlex.quote(expected+"  /scratch/pireus/images/inkling-spark.sif")+" | sha256sum -c -\nexec "+shlex.join(argv[-5:])
  argv=argv[:-5]+["bash","-c",payload]
  output.write(encoded({"command":argv}));output.flush()
  result=subprocess.run(argv,cwd="/tmp",env=env,stdout=output,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(phase+" job failed; raw log retained")

def observe_available(pods,remote_root,phase):
 prefix="observed-" if phase=="probe" else "timings-"
 for pod,node in zip(pods,NODES):
  command=["kubectl","-n","slurm-pilot","exec",pod,"-c","slurmd","--","test","-f",remote_root+"/"+prefix+node+"/inventory.json"]
  if subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:return False
 return True

def collect(pods,root,remote_root,phase):
 prefix="observed-" if phase=="probe" else "timings-"
 for pod,node in zip(pods,NODES):
  data=remote(pod,"tar","cf","-","-C",remote_root,prefix+node)
  with tarfile.open(fileobj=io.BytesIO(data)) as archive:
   for member in archive:
    if member.isdir():continue
    rel=Path(member.name)
    if not member.isfile() or rel.is_absolute() or ".." in rel.parts:raise ValueError("unsafe observed artifact")
    target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
    atomic(target,archive.extractfile(member).read());event(root,phase+"-observation",target)

def native(executable,*args):
 run=subprocess.run([str(executable.resolve()),*map(str,args)],capture_output=True,timeout=120)
 if run.returncode not in (0,1):raise RuntimeError("Sounio execution failed: "+run.stderr.decode(errors="replace"))
 value=json.loads(run.stdout)
 if value.get("authority")!="Sounio":raise ValueError("invalid native authority stream")
 return run.returncode,run.stdout,value

def main():
 ap=argparse.ArgumentParser()
 for flag in ["run","engine","fixture-engine","parity-engine","gain-engine"]:ap.add_argument("--"+flag,type=Path,required=True)
 a=ap.parse_args();root=a.run
 lock=(root/".lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not os.environ.get("TMUX"):raise SystemExit("run the workspace cycle inside remote tmux")
 manifest=verify(root)
 if digest(a.engine.read_bytes())!=manifest["engine_sha256"]:raise ValueError("admission engine drift")
 config=dict(fixture_engine_sha256=digest(a.fixture_engine.read_bytes()),
   parity_engine_sha256=digest(a.parity_engine.read_bytes()),gain_engine_sha256=digest(a.gain_engine.read_bytes()),
   method="paired-median-ppm-bootstrap4000-v1",blocks=30,vectors=16384,replays=32,
   thresholds=dict(median_ppm=50000,low_ci_ppm_exclusive=0),scope="resident-layout-kernel",
   layout_conversion_included=False,baseline="both batched direct and shuffle controls",
   interrupted_trial_policy="retain incomplete files; never reinterpret them as complete")
 atomic(root/"benchmark-contract.json",encoded(config));event(root,"benchmark-contract",root/"benchmark-contract.json")
 fixture=subprocess.run([str(a.fixture_engine.resolve())],capture_output=True,check=True,timeout=120).stdout
 atomic(root/"numeric-fixtures.jsonl",fixture)
 controls=[];candidates=[]
 control_root=root/"controls";control_root.mkdir(exist_ok=True)
 for name,load in [("direct",0),("shuffle",1)]:
  proposal=proposal_template(digest((root/"context.json").read_bytes()))|{"load":load}
  path=control_root/(name+".proposal.json");atomic(path,encoded(proposal))
  code,raw,receipt=native(a.engine,root/"context.json",path)
  if code or receipt["decision"]!="ADMIT":raise ValueError("baseline admission refused")
  atomic(control_root/(name+".receipt.json"),raw)
  result=subprocess.run([str(a.engine.resolve()),str((root/"context.json").resolve()),str(path.resolve()),"ptx"],capture_output=True,check=True,timeout=60)
  artifact=control_root/(name+".ptx");atomic(artifact,result.stdout)
  controls.append(dict(id=name,layout=0,ptx=artifact.relative_to(root).as_posix(),ptx_sha256=digest(result.stdout)))
 for i in range(manifest["budget"]):
  prefix=root/("%03d"%i);receipt_path=prefix.with_suffix(".receipt.json")
  if not receipt_path.exists():raise ValueError("admission incomplete")
  receipt=json.loads(receipt_path.read_text())
  if receipt["decision"]!="ADMIT":continue
  artifact=prefix.with_suffix(".ptx")
  if not artifact.exists():raise ValueError("materialization incomplete")
  proposal=json.loads(prefix.with_suffix(".proposal.json").read_text())
  candidates.append(dict(id="%03d"%i,layout=proposal["layout"],ptx=artifact.name,ptx_sha256=digest(artifact.read_bytes())))
 gpu_manifest=dict(schema=1,fixtures_sha256=digest(fixture),candidates=controls+candidates)
 atomic(root/"gpu-manifest.json",encoded(gpu_manifest));event(root,"gpu-manifest",root/"gpu-manifest.json")
 remote_root="/scratch/pireus/runs/cycle-"+digest((root/"manifest.json").read_bytes())[:20]
 print(json.dumps({"host_preflight":check_pair()}),flush=True)
 pods=workers()
 names=["gpu-manifest.json","numeric-fixtures.jsonl",*[item["ptx"] for item in controls+candidates]]
 stage(pods,root,remote_root,names)
 if not (root/"probe.complete.json").exists():
  if not observe_available(pods,remote_root,"probe"):run_gpu(root,remote_root,"probe")
  collect(pods,root,remote_root,"probe")
  atomic(root/"probe.complete.json",encoded(dict(remote_root=remote_root,pods=pods)))
  event(root,"probe",root/"probe.complete.json")
 eligible=[];parity={}
 for item in controls+candidates:
  accepted=True;receipts=[]
  for node in NODES:
   folder=root/("observed-"+node)
   code,raw,receipt=native(a.parity_engine,folder/"reference.hex",folder/(item["id"]+".hex"))
   target=folder/(item["id"]+".parity.json");atomic(target,raw);event(root,"parity",target)
   accepted=accepted and code==0 and receipt["decision"]=="PASS"
   receipts.append(receipt)
  parity[item["id"]]=receipts
  if item in controls and not accepted:raise ValueError("baseline material parity failed")
  if item in candidates and accepted:eligible.append(item)
 bm=config|dict(fixtures_sha256=digest(fixture),controls=controls,candidates=eligible)
 atomic(root/"benchmark-manifest.json",encoded(bm));event(root,"benchmark-manifest",root/"benchmark-manifest.json")
 stage(pods,root,remote_root,["benchmark-manifest.json"])
 if eligible and not (root/"timing.complete.json").exists():
  if not observe_available(pods,remote_root,"timing"):run_gpu(root,remote_root,"timing")
  collect(pods,root,remote_root,"timing")
  atomic(root/"timing.complete.json",encoded(dict(remote_root=remote_root,pods=pods)))
  event(root,"timing",root/"timing.complete.json")
 decisions=[]
 for candidate in eligible:
  paths=[root/("timings-"+node)/(candidate["id"]+"."+control["id"]+".tsv") for node in NODES for control in controls]
  code,raw,value=native(a.gain_engine,*paths)
  if code:raise ValueError("invalid timing observations")
  destination=root/(candidate["id"]+".gain.json");atomic(destination,raw);event(root,"gain",destination)
  decisions.append(dict(candidate=candidate["id"],receipt=value))
 result=dict(schema=1,semantic_authority="Sounio",material_candidates=len(candidates),
   material_pass=len(eligible),material_refused=len(candidates)-len(eligible),decisions=decisions,
   gain_eligible=sum(x["receipt"]["gain_gate"]=="PASS" for x in decisions),claim_ready=False)
 atomic(root/"benchmark-report.json",encoded(result));event(root,"benchmark-report",root/"benchmark-report.json")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
