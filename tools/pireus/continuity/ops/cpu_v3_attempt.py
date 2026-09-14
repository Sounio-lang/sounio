#!/usr/bin/env python3
"""Single frozen v3 CPU/container attempt and terminal custody. Never retry."""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1]
PACKET=HERE/"validation/external-container-cpu-v3-20260909"
spec=importlib.util.spec_from_file_location("cpu_v3_freeze",PACKET/"verify_freeze.py")
freeze=importlib.util.module_from_spec(spec);spec.loader.exec_module(freeze)
from check_frozen_feedback import source_checks
sys.path.insert(0,str(HERE/"runtime"))
from preflight import check_pair
NODES=["spark-3c59","spark-8e54"]
SLURM_NODES=["gpuorangefs-multi-"+n for n in NODES]
JOB_NAME="pireus-external-cpu-v3"
def require(ok,msg):
    if not ok:raise ValueError(msg)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(p.read_bytes())
def orchestration_hashes():
    names=("ops/cpu_v3_attempt.py","ops/check_frozen_feedback.py","runtime/preflight.py",
           "validation/external-container-cpu-v3-20260909/verify_freeze.py")
    return {n:digest((HERE/n).read_bytes()) for n in names}
def write(p,v):
    with p.open("x") as f:f.write(json.dumps(v,indent=2)+"\n")
def run(argv,**kw):return subprocess.check_output(argv,**kw)
def kube(*args):return run(["kubectl","-n","slurm-pilot",*args],text=True,timeout=60)
def packet(root):
    freeze.verify(root)
    return read(root/"protocol.json")
def readiness(root):
    p=packet(root)
    raw=run(["gh","api","--paginate","repos/Sounio-lang/sounio/commits/"+p["source_commit"]+"/check-runs",
             "--jq",".check_runs[] | @json"],text=True,timeout=60)
    selected=source_checks(p,[json.loads(line) for line in raw.splitlines() if line])
    return dict(source_commit=p["source_commit"],protocol_sha256=digest((root/"protocol.json").read_bytes()),
                source_checks=selected,source_ci_qualified=True)
def workers():
    pods=json.loads(kube("get","pods","-o","json"))["items"]
    result=[]
    for node in NODES:
        candidates=[p for p in pods if p["spec"].get("nodeName")==node
                    and p["metadata"]["name"].startswith("slurm-pilot-worker-spark-")
                    and p["status"]["phase"]=="Running" and not p["metadata"].get("deletionTimestamp")]
        require(len(candidates)==1,"unique worker required")
        p=candidates[0]
        boot=kube("exec",p["metadata"]["name"],"-c","slurmd","--","cat","/proc/sys/kernel/random/boot_id").strip()
        result.append(dict(node=node,pod=p["metadata"]["name"],uid=p["metadata"]["uid"],boot_id=boot))
    return result
def remote_root(protocol_pin):
    return "/scratch/pireus/diagnostics/cpu-v3-"+protocol_pin[:20]
def command(ws,p,pin):
    require([w["node"] for w in ws]==NODES,"worker order")
    source=remote_root(pin)+"/source"
    identity=[]
    for rank,w in enumerate(ws):
        identity.append("*"+w["node"].removeprefix("spark-")+") export PIREUS_RANK="+str(rank)+
            " PIREUS_EXTERNAL_WORKER_UID="+shlex.quote(w["uid"])+
            " PIREUS_EXTERNAL_BOOT_ID="+shlex.quote(w["boot_id"])+";;")
    barrier="""import hashlib,json,os,time
from pathlib import Path
expected=EXPECTED
source=Path(SOURCE)
actual={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in expected}
if actual!=expected:raise ValueError('runtime before entry mismatch')
if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=os.environ['PIREUS_EXTERNAL_BOOT_ID']:raise ValueError('boot changed')
r=dict(stage='CPU_V3_RUNTIME_VERIFIED',job=os.environ['SLURM_JOB_ID'],rank=os.environ['PIREUS_RANK'],runtime_sha256=actual,protocol_sha256=PIN,worker_uid=os.environ['PIREUS_EXTERNAL_WORKER_UID'],boot_id=os.environ['PIREUS_EXTERNAL_BOOT_ID'],monotonic_ns=time.monotonic_ns())
path=Path('/scratch/pireus/receipts')/('cpu-v3-runtime-'+r['job']+'-'+r['rank']+'.json')
with path.open('x') as out:json.dump(r,out)
print(json.dumps(r),flush=True)
""".replace("EXPECTED",repr(p["runtime_files_sha256"])).replace("SOURCE",repr(source)).replace("PIN",repr(pin))
    q=shlex.quote
    body="\n".join(["set -euo pipefail",'case "$(hostname)" in',*identity,"*) exit 2;;","esac",
        "unset PIREUS_MARLIN_OVERLAY",
        "python3 -c "+q(barrier),
        "echo "+q(p["sif_sha256"]+"  /scratch/pireus/images/inkling-spark.sif")+" | sha256sum -c -",
        "exec python3 "+q(source+"/external_rank_supervisor.py")+
        ' --output "/scratch/pireus/receipts/cpu-v3-${SLURM_JOB_ID}-${PIREUS_RANK}"'+
        ' --worker-uid "$PIREUS_EXTERNAL_WORKER_UID" --boot-id "$PIREUS_EXTERNAL_BOOT_ID"'+
        " --entry-sha "+q(p["runtime_files_sha256"]["cpu-target.py"])+
        " -- python3 "+q(source+"/memory_guard.py")+" --reserve-gib 33 -- bash "+q(source+"/run_in_container.sh")+
        " python3 "+q(source+"/publish_observer_target.py")+" --entry "+q(source+"/cpu-target.py")+
        " --entry-sha "+q(p["runtime_files_sha256"]["cpu-target.py"])+" --"])
    return ["srun","-p","gpu-orangefs","-w",",".join(SLURM_NODES),"--nodes=2","--ntasks=2","--ntasks-per-node=1",
            "-c2","--mem=512M","--exclusive","--propagate=NONE","--kill-on-bad-exit=1","--time=5","--job-name="+JOB_NAME,"bash","-c",body]
def launch(root,stage):
    require(bool(os.environ.get("TMUX")),"remote tmux required")
    require(not stage.exists(),"attempt already exists; no retry")
    p=packet(root);ready=readiness(root);preflight=check_pair()
    require(not run(["squeue","-h","-p","gpu-orangefs"],text=True).strip(),"pair queue occupied")
    ws=workers();pin=digest((root/"protocol.json").read_bytes())
    stage.mkdir(parents=True,exist_ok=False)
    write(stage/"readiness.json",ready);write(stage/"preflight.json",preflight)
    blobs={n:base64.b64encode((root/"source"/n).read_bytes()).decode() for n in p["runtime_files_sha256"]}
    for w in ws:
        require(read_live(w)["metadata"]["uid"]==w["uid"],"worker changed")
        transfer="""import base64,json,os,sys,hashlib
from pathlib import Path
d=json.load(sys.stdin);root=Path(d['root']);root.mkdir(parents=True,exist_ok=False)
(root/'source').mkdir()
for n,b in d['files'].items():
 p=root/'source'/n;p.write_bytes(base64.b64decode(b,validate=True));p.chmod(0o444)
receipt=Path('/scratch/pireus/receipts');receipt.mkdir(exist_ok=True);os.chown(receipt,1000,1000)
print(json.dumps({n:hashlib.sha256((root/'source'/n).read_bytes()).hexdigest() for n in d['files']}))
"""
        raw=run(["kubectl","-n","slurm-pilot","exec","-i",w["pod"],"-c","slurmd","--","python3","-c",transfer],
                input=json.dumps(dict(root=remote_root(pin),files=blobs)).encode())
        require(json.loads(raw)==p["runtime_files_sha256"],"transfer hash mismatch")
    argv=command(ws,p,pin)
    start=dict(schema="pireus-cpu-v3-start",source_commit=p["source_commit"],protocol_sha256=pin,workers=ws,
               runtime_sha256=p["runtime_files_sha256"],command=argv,launcher_sha256=digest(Path(__file__).read_bytes()),
               orchestration_sha256=orchestration_hashes())
    write(stage/"start.json",start)
    packet(root);readiness(root);check_pair()
    require(workers()==ws,"worker changed before allocation")
    require(not run(["squeue","-h","-p","gpu-orangefs"],text=True).strip(),"pair queue changed")
    env=os.environ.copy();env["TMPDIR"]="/tmp"
    with (stage/"launch.log").open("xb") as out:
        rc=subprocess.call(argv,cwd="/tmp",env=env,stdout=out,stderr=subprocess.STDOUT)
    (stage/"exit-code").write_text(str(rc)+"\n")
    return rc
def read_live(w):return json.loads(kube("get","pod",w["pod"],"-o","json"))
def accounting(raw,job,success=False):
    main=[s.split("|") for s in raw.splitlines() if s.split("|")[0]==job]
    require(len(main)==1 and len(main[0])>=7,"unique accounting required")
    r=main[0]
    require(r[1]==JOB_NAME and set(r[4].split(","))==set(SLURM_NODES),"job identity mismatch")
    require(r[2].split()[0] in {"COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY","PREEMPTED","BOOT_FAIL","DEADLINE"},"job not terminal")
    require(all(v not in ("","Unknown") for v in r[5:7]),"accounting timestamps absent")
    if success:require(r[2:4]==["COMPLETED","0:0"],"job failed")
    return r
def collect(root,stage,output,job):
    p=packet(root);start=read(stage/"start.json")
    require(job.isdigit(),"invalid job")
    require(start["protocol_sha256"]==digest((root/"protocol.json").read_bytes()),"start protocol")
    require((stage/"exit-code").exists(),"launcher not terminal")
    raw=kube("exec","slurm-pilot-controller-0","-c","slurmctld","--","sacct","-j",job,"--noheader","--parsable2","--format=JobID,JobName,State,ExitCode,NodeList,Start,End")
    terminal=accounting(raw,job)
    output.mkdir(parents=True,exist_ok=False)
    for n in ("start.json","readiness.json","preflight.json","launch.log","exit-code"):(output/n).write_bytes((stage/n).read_bytes())
    (output/"accounting.txt").write_text(raw)
    missing=[]
    for rank,w in enumerate(start["workers"]):
        for moment in ("before","after"):
            pod=read_live(w);require(pod["metadata"]["uid"]==w["uid"],"worker changed")
            write(output/f"worker-{rank}-{moment}.json",pod)
            if moment=="after":continue
            prefix=f"/scratch/pireus/receipts/cpu-v3-{job}-{rank}/"
            paths={n:prefix+n for n in ("target.json","attached.json","binding.json","journal.jsonl","observer.log","cpu-phases.jsonl","result.json")}
            paths["runtime-before.json"]=f"/scratch/pireus/receipts/cpu-v3-runtime-{job}-{rank}.json"
            paths["boot-id.txt"]="/proc/sys/kernel/random/boot_id"
            paths.update({"source/"+n:remote_root(start["protocol_sha256"])+"/source/"+n for n in p["runtime_files_sha256"]})
            code="import json,base64;from pathlib import Path\npaths="+repr(paths)+"\nout={}\nfor n,p in paths.items():\n try:out[n]=base64.b64encode(Path(p).read_bytes()).decode()\n except FileNotFoundError:out[n]=None\nprint(json.dumps(out))"
            data=json.loads(kube("exec",w["pod"],"-c","slurmd","--","python3","-c",code))
            require(set(data)==set(paths),"collection inventory")
            for n,value in data.items():
                dest=output/f"rank-{rank}"/n
                if value is None:missing.append(str(dest.relative_to(output)));continue
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(base64.b64decode(value,validate=True))
    for n in ("start.json","readiness.json","preflight.json","launch.log","exit-code"):
        require((output/n).read_bytes()==(stage/n).read_bytes(),"stage changed during collection")
    receipt=dict(schema="pireus-cpu-v3-collection",job=job,protocol_sha256=start["protocol_sha256"],terminal_state=terminal[2],
        missing=missing,qualified=False,files_sha256={str(f.relative_to(output)):digest(f.read_bytes()) for f in output.rglob("*") if f.is_file()})
    write(output/"collection.json",receipt)
    return receipt

def qualify(root,collected,pin):
    p=packet(root);raw=(collected/"collection.json").read_bytes()
    require(digest(raw)==pin,"collection pin mismatch")
    c=json.loads(raw)
    require(c["schema"]=="pireus-cpu-v3-collection" and not c["missing"],"incomplete collection")
    for name,sha in c["files_sha256"].items():
        path=collected/name
        require(path.resolve().is_relative_to(collected.resolve()) and digest(path.read_bytes())==sha,"collected file mismatch: "+name)
    require(c["protocol_sha256"]==digest((root/"protocol.json").read_bytes()),"collection protocol mismatch")
    job=c["job"];accounting((collected/"accounting.txt").read_text(),job,True)
    require((collected/"exit-code").read_text().strip()=="0","launcher failure")
    start=read(collected/"start.json")
    require(start["source_commit"]==p["source_commit"] and start["protocol_sha256"]==c["protocol_sha256"]
            and start["runtime_sha256"]==p["runtime_files_sha256"],"start identity")
    require(start["command"]==command(start["workers"],p,c["protocol_sha256"]),"launch command mismatch")
    require(start["launcher_sha256"]==digest(Path(__file__).read_bytes()),"launcher/qualifier source mismatch")
    require(start["orchestration_sha256"]==orchestration_hashes(),"orchestration helper drift")
    ready=read(collected/"readiness.json")
    require(ready["source_commit"]==p["source_commit"] and ready["protocol_sha256"]==c["protocol_sha256"],"readiness identity")
    source_checks(p,list(ready["source_checks"].values()))
    logs=[]
    for line in (collected/"launch.log").read_bytes().splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if isinstance(r,dict):logs.append(r)
    require(not any(r.get("stage")=="MEMORY_GUARD_STOP" for r in logs),"guardian stopped")
    results=[]
    for rank,w in enumerate(start["workers"]):
        d=collected/f"rank-{rank}"
        for moment in ("before","after"):
            pod=read(collected/f"worker-{rank}-{moment}.json")
            require(pod["metadata"]["uid"]==w["uid"] and pod["spec"]["nodeName"]==w["node"],"worker custody mismatch")
        require((d/"boot-id.txt").read_text().strip()==w["boot_id"],"boot custody mismatch")
        for name,sha in p["runtime_files_sha256"].items():
            require(digest((d/"source"/name).read_bytes())==sha,"runtime custody mismatch")
        barrier=read(d/"runtime-before.json")
        require(barrier["stage"]=="CPU_V3_RUNTIME_VERIFIED" and barrier["job"]==job and barrier["rank"]==str(rank)
                and barrier["runtime_sha256"]==p["runtime_files_sha256"] and barrier["protocol_sha256"]==c["protocol_sha256"]
                and barrier["worker_uid"]==w["uid"] and barrier["boot_id"]==w["boot_id"],"pre-entry barrier identity")
        require(sum(r==barrier for r in logs)==1,"barrier not uniquely in job output")
        phases=[json.loads(s) for s in (d/"cpu-phases.jsonl").read_text().splitlines()]
        target,ack,binding,result=[read(d/n) for n in ("target.json","attached.json","binding.json","result.json")]
        require(target["schema"]=="pireus-observer-target-handoff-v1" and ack["schema"]=="pireus-observer-attachment-ack-v1","handoff schema")
        require(target["job"]==job and target["rank"]==str(rank) and target["boot_id"]==w["boot_id"]
                and target["entry_sha256"]==p["runtime_files_sha256"]["cpu-target.py"]
                and target["pid"]==phases[0]["pid"],"CPU target identity")
        require(barrier["monotonic_ns"]<phases[0]["monotonic_ns"],"pre-entry barrier time")
        require(ack["nonce"]==target["nonce"]==result["nonce"] and ack["first_sample_valid"] is True,"ack nonce/validity")
        require(all(ack[k]==target[k] for k in ("job","rank","pid","starttime_ticks")),"ack target identity")
        require(ack["handoff_sha256"]==digest((d/"target.json").read_bytes())
                and ack["binding_sha256"]==digest((d/"binding.json").read_bytes()),"ack hashes")
        require(binding["expected"]=={k:target[k] for k in ("job","rank","pid","starttime_ticks","boot_id")}|{"worker_uid":w["uid"]}
                and binding["observer_helper_sha256"]==p["runtime_files_sha256"]["external_memory_observer.py"],"binding identity/source")
        require(binding["schema"]=="pireus-external-observer-binding-v1"
                and all(binding["observed"][k]==v for k,v in binding["expected"].items())
                and "job_"+job in Path(binding["observed"]["cgroup"]["membership"]).parts,"observed binding mismatch")
        require(result["job"]==job and result["rank"]==str(rank)
                and result["integration_complete"] is True and result["guardian_returncode"]==0
                and result["observer_returncode"]==3 and result["entry_sha256"]==target["entry_sha256"],"supervisor result")
        for name,sha in result["files_sha256"].items():
            path=d/name
            require(path.resolve().is_relative_to(d.resolve()) and digest(path.read_bytes())==sha,"supervisor file hash")
        journal=[json.loads(s) for s in (d/"journal.jsonl").read_text().splitlines()]
        binding_pin=digest(json.dumps(binding,sort_keys=True).encode())
        require(all(r["binding_sha256"]==binding_pin and r["observer_pid"]==ack["observer_pid"]
                    and r["job"]==job and r["rank"]==str(rank) and r["target_pid"]==target["pid"] for r in journal),"journal binding")
        guards=[r for r in logs if r.get("stage")=="MEMORY_GUARD_CHILD_EXIT" and r.get("job")==job and r.get("rank")==str(rank)]
        require(len(guards)==1 and guards[0]["returncode"]==0 and guards[0]["minimum_bytes"]>=33*1024**3,"guardian completion")
        oracle=sys.modules["cpu_memory_control_v3"].analyze(journal,phases)
        results.append(dict(rank=rank,oracle=oracle,guardian=guards[0]))
    return dict(schema="pireus-cpu-v3-qualification",job=job,source_commit=p["source_commit"],
        protocol_sha256=c["protocol_sha256"],collection_sha256=pin,custody_verified=True,
        cpu_container_control_qualified=True,results=results,observer_overhead_budget_qualified=False,
        inference_executed=False,loaded_model_overhead_qualified=False)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    for mode in ("launch","collect","qualify"):
        s=sub.add_parser(mode);s.add_argument("--frozen",type=Path,required=True)
        if mode in ("launch","collect"):s.add_argument("--stage",type=Path,required=True)
        if mode=="collect":s.add_argument("--job",required=True)
        if mode in ("collect","qualify"):s.add_argument("--output",type=Path,required=True)
        if mode=="qualify":
            s.add_argument("--collected",type=Path,required=True);s.add_argument("--collection-pin",required=True)
    a=parser.parse_args()
    if a.mode=="launch":raise SystemExit(launch(a.frozen,a.stage))
    if a.mode=="collect":collect(a.frozen,a.stage,a.output,a.job)
    if a.mode=="qualify":write(a.output,qualify(a.frozen,a.collected,a.collection_pin))
