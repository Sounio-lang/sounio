#!/usr/bin/env python3
"""One explicit overhead arm, remote tmux + exclusive Slurm; no automatic retry."""
import argparse
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from freeze_external_overhead_v3 import verify,readiness,digest,HERE,orchestration_hashes
from external_overhead_custody_v3 import arm_data
sys.path.insert(0,str(HERE/"runtime"))
from preflight import check_pair

SIF_SHA="3dbfccad3355b27d8a09bd4c0c5895d02a43e1b64203960905b810bbb6bccbe3"
MODEL="/scratch/pireus/models/Inkling-Small-NVFP4/b6a99534467840620d411e4cd4ad5819b2610d9c"
NODES=["gpuorangefs-multi-spark-3c59","gpuorangefs-multi-spark-8e54"]
def kube(*args):
    return subprocess.check_output(["kubectl",*args],text=True,timeout=60)

def prerequisites(frozen,arm,stage,baseline=None,baseline_pin=None):
    if not __debug__:raise ValueError("optimized Python disables barriers")
    require=os.environ.get("TMUX")
    if not require:raise ValueError("remote tmux required")
    if arm not in ("baseline","observed"):raise ValueError("unknown arm")
    if stage.exists():raise ValueError("attempt already exists; no replay")
    spec=verify(frozen)
    prior=None
    if arm=="observed":
        if baseline is None or baseline_pin is None:raise ValueError("baseline custody required")
        prior=arm_data(frozen,baseline,baseline_pin,"baseline")
        # The second arm may only follow a complete successful first arm.
        for rank in (0,1):
            rows=[r for r in prior["guardian"] if r.get("stage")=="MEMORY_GUARD_CHILD_EXIT" and r.get("job")==prior["job"] and r.get("rank")==str(rank)]
            if len(rows)!=1 or rows[0]["returncode"]!=0 or rows[0]["minimum_bytes"]<33*1024**3:
                raise ValueError("baseline guardian not qualified")
    ready=readiness(frozen)  # Fresh exact-source check; never reuse old-source CI.
    return spec,ready,prior

def command(workers,bundle_sha,runtime):
    identities=[]
    for rank,w in enumerate(workers):
        if w["node"]!=["spark-3c59","spark-8e54"][rank]:raise ValueError("worker order")
        identities.append("*"+w["node"].removeprefix("spark-")+") export PIREUS_RANK="+str(rank)+
            " PIREUS_EXTERNAL_WORKER_UID="+shlex.quote(w["uid"])+
            " PIREUS_EXTERNAL_BOOT_ID="+shlex.quote(w["boot_id"])+";;")
    check="""import hashlib,json,os,time
from pathlib import Path
expected=EXPECTED
actual={n:hashlib.sha256((Path('/scratch/pireus/runtime')/n).read_bytes()).hexdigest() for n in expected}
assert actual==expected,'runtime changed before entry'
assert hashlib.sha256(Path('/scratch/pireus/runtime/tokenizer-input-BUNDLE.json').read_bytes()).hexdigest()=='BUNDLE','input changed before entry'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==os.environ['PIREUS_EXTERNAL_BOOT_ID'],'boot changed'
receipt=dict(stage='OVERHEAD_RUNTIME_VERIFIED',job=os.environ['SLURM_JOB_ID'],rank=os.environ['PIREUS_RANK'],runtime_sha256=actual,input_sha256='BUNDLE',boot_id=os.environ['PIREUS_EXTERNAL_BOOT_ID'],worker_uid=os.environ['PIREUS_EXTERNAL_WORKER_UID'],monotonic_ns=time.monotonic_ns())
path=Path('/scratch/pireus/receipts')/('overhead-runtime-'+receipt['job']+'-'+receipt['rank']+'.json')
with path.open('x') as out:json.dump(receipt,out)
print(json.dumps(dict(stage='OVERHEAD_RUNTIME_VERIFIED',job=receipt['job'],rank=receipt['rank'],receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest())),flush=True)
""".replace("EXPECTED",repr(runtime)).replace("BUNDLE",bundle_sha)
    body="\n".join(["set -euo pipefail",'case "$(hostname)" in',*identities,"*) exit 2;;","esac",
        "python3 -c "+shlex.quote(check),
        "echo "+shlex.quote(SIF_SHA+"  /scratch/pireus/images/inkling-spark.sif")+" | sha256sum -c -",
        "python3 /scratch/pireus/runtime/qualify_model.py "+MODEL+" /scratch/pireus/runtime/inkling-files.json --receipt /scratch/pireus/receipts/inkling-model.json",
        "export PIREUS_OFFLINE_MODE=generate PIREUS_TOKEN_IDS=1 PIREUS_OFFLINE_INPUT=/scratch/pireus/runtime/tokenizer-input-"+bundle_sha+".json",
        "exec /scratch/pireus/runtime/serve_rank.sh"])
    return ["srun","-p","gpu-orangefs","-w",",".join(NODES),"--nodes=2","--ntasks=2","--ntasks-per-node=1",
        "-c8","--gres=gpu:gb10:1","--mem=110G","--exclusive","--propagate=NONE","--kill-on-bad-exit=1",
        "--time=55","--job-name=pireus-inkling-offline-generate","bash","-c",body]

def launch(frozen,arm,stage,baseline=None,baseline_pin=None):
    spec,ready,prior=prerequisites(frozen,arm,stage,baseline,baseline_pin)
    reports=check_pair()
    queue=subprocess.check_output(["squeue","-h","-p","gpu-orangefs"],text=True)
    if queue.strip():raise ValueError("pair queue occupied")
    pods=json.loads(kube("-n","slurm-pilot","get","pods","-o","json"))["items"]
    workers=[]
    for node in ("spark-3c59","spark-8e54"):
        matches=[p for p in pods if p["spec"].get("nodeName")==node and p["metadata"]["name"].startswith("slurm-pilot-worker-spark-")
                 and p["status"]["phase"]=="Running" and not p["metadata"].get("deletionTimestamp")]
        if len(matches)!=1:raise ValueError("unique worker required")
        p=matches[0]
        boot=kube("-n","slurm-pilot","exec",p["metadata"]["name"],"-c","slurmd","--","cat","/proc/sys/kernel/random/boot_id").strip()
        workers.append(dict(node=node,pod=p["metadata"]["name"],uid=p["metadata"]["uid"],boot_id=boot,ip=p["status"]["podIP"]))
    if prior is not None and prior["workers"]!=workers:raise ValueError("worker changed between arms")
    stage.mkdir(parents=True,exist_ok=False)
    ready_raw=(json.dumps(ready,indent=2)+"\n").encode()
    (stage/"readiness.json").write_bytes(ready_raw)
    (stage/"preflight.json").write_text(json.dumps(reports,indent=2)+"\n")
    (stage/"queue-before.txt").write_text(queue)
    runtime=spec["runtime_sha256"][arm]
    blobs={n:base64.b64encode((frozen/arm/"runtime"/n).read_bytes()).decode() for n in runtime}
    bundle=(frozen/"without-feedback/offline-bundle.json").read_bytes();bundle_sha=digest(bundle)
    remote_marker="/scratch/pireus/diagnostics/overhead-"+digest((frozen/"execution-freeze.json").read_bytes())[:20]+"-"+arm
    before=[]
    for w in workers:
        live=json.loads(kube("-n","slurm-pilot","get","pod",w["pod"],"-o","json"))
        if live["metadata"]["uid"]!=w["uid"]:raise ValueError("worker UID changed")
        code="""import sys,json,base64,hashlib,os
from pathlib import Path
d=json.load(sys.stdin);Path(d['marker']).mkdir(parents=True,exist_ok=False)
root=Path('/scratch/pireus/runtime')
for name,b64 in d['blobs'].items():
 p=root/name;p.write_bytes(base64.b64decode(b64,validate=True));p.chmod(0o755)
p=root/('tokenizer-input-'+d['bundle_sha']+'.json');raw=base64.b64decode(d['bundle'])
assert not p.exists() or p.read_bytes()==raw
p.write_bytes(raw);p.chmod(0o444)
for name in ('receipts','cache'):
 p=Path('/scratch/pireus')/name;p.mkdir(exist_ok=True);os.chown(p,1000,1000)
print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in d['blobs']}))
"""
        payload=dict(marker=remote_marker,blobs=blobs,bundle_sha=bundle_sha,bundle=base64.b64encode(bundle).decode())
        argv=["kubectl","-n","slurm-pilot","exec","-i",w["pod"],"-c","slurmd","--","python3","-c",code]
        result=subprocess.run(argv,input=json.dumps(payload).encode(),stdout=subprocess.PIPE,check=True)
        hashes=json.loads(result.stdout)
        if hashes!=runtime:raise ValueError("transferred runtime mismatch")
        before.append(hashes)
        # Existing, pinned, idempotent worker prerequisite setup.
        setup=kube("-n","slurm-pilot","exec",w["pod"],"-c","slurmd","--","python3","/scratch/pireus/runtime/worker_prerequisites.py")
        (stage/("prerequisite-"+w["node"]+".txt")).write_text(setup)
    argv=command(workers,bundle_sha,runtime)
    start=dict(schema="pireus-overhead-start-v1",arm=arm,source_commit=spec["source_commit"],
        freeze_sha256=digest((frozen/"execution-freeze.json").read_bytes()),workers=workers,
        runtime_before_sha256=before,input_sha256=bundle_sha,command=argv,
        baseline_collection_sha256=baseline_pin,launcher_sha256=digest(Path(__file__).read_bytes()),
        orchestration_sha256=orchestration_hashes())
    (stage/"start.json").write_text(json.dumps(start,indent=2)+"\n")
    # Revalidate custody/preflight immediately before submission; a refusal is terminal
    # for this attempt directory. There is no catch-and-retry path.
    verify(frozen);readiness(frozen);check_pair()
    if subprocess.check_output(["squeue","-h","-p","gpu-orangefs"],text=True).strip():raise ValueError("pair queue changed")
    env=os.environ.copy()
    env.update(TMPDIR="/tmp",MASTER_ADDR=workers[0]["ip"],MASTER_PORT="29571",
        NCCL_NET="IB",NCCL_IB_DISABLE="0",NCCL_DEBUG="INFO",NCCL_SOCKET_IFNAME="eth0")
    with (stage/"launch.log").open("xb") as out:
        rc=subprocess.call(argv,env=env,stdout=out,stderr=subprocess.STDOUT,cwd="/tmp")
    (stage/"exit-code").write_text(str(rc)+"\n")
    return rc

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--frozen",type=Path,required=True);p.add_argument("--stage",type=Path,required=True)
    p.add_argument("--arm",choices=["baseline","observed"],required=True)
    p.add_argument("--baseline",type=Path);p.add_argument("--baseline-pin")
    a=p.parse_args()
    raise SystemExit(launch(a.frozen,a.arm,a.stage,a.baseline,a.baseline_pin))
