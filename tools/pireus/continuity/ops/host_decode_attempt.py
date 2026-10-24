#!/usr/bin/env python3
"""One frozen host-decode diagnostic; preserve partial terminal evidence, no retry."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
from check_frozen_feedback import source_checks
from launch_external_overhead_v3 import command, kube, check_pair
from external_overhead_custody_v3 import accounting, KUBE
HERE=Path(__file__).resolve().parents[1]
PROTOCOL=HERE/"validation/external-overhead-v3-preparation-20260910/host-decode-probe/diagnostic-protocol.json"
PROTOCOL_SHA="e768bf86c456eaf9b4470fab4a1b81500bb4ef24be33be9754b68764af389227"
HELPERS=("ops/host_decode_attempt.py","ops/launch_external_overhead_v3.py",
         "ops/external_overhead_custody_v3.py","ops/check_frozen_feedback.py","runtime/preflight.py")
def digest(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()
def require(ok,message):
    if not ok:raise ValueError(message)
def orchestration_hashes():return {n:digest((HERE/n).read_bytes()) for n in HELPERS}
def protocol():
    raw=PROTOCOL.read_bytes()
    require(digest(raw)==PROTOCOL_SHA,"diagnostic protocol changed")
    return json.loads(raw)
def specification():
    p=protocol()
    return dict(schema="pireus-host-decode-execution-v1",source_commit=p["source_commit"],
                diagnostic_protocol_sha256=PROTOCOL_SHA,orchestration_sha256=orchestration_hashes(),
                runtime_sha256={"baseline":p["runtime_sha256"]},files_sha256=p["files_sha256"],
                required_source_checks=p["required_source_checks"],diagnostic_only=True)
def create(root):
    require(not root.exists(),"frozen root exists")
    p=protocol();spec=specification();blobs={}
    for name,pin in p["runtime_sha256"].items():
        ref=p["runtime_git_sources"][name]
        raw=subprocess.check_output(["git","show",ref["commit"]+":"+ref["path"]],cwd=HERE)
        require(digest(raw)==pin,"runtime source mismatch")
        blobs["baseline/runtime/"+name]=raw
    parent=HERE/"validation/lifecycle-diagnostic-freeze-20260909"
    for name,pin in p["files_sha256"].items():
        raw=(parent/name).read_bytes();require(digest(raw)==pin,"input mismatch")
        blobs[name]=raw
    root.mkdir(parents=True,exist_ok=False)
    for name,raw in blobs.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        path.chmod(0o555 if path.suffix==".sh" else 0o444)
    (root/"execution-freeze.json").write_bytes(encoded(spec))
    return verify(root)
def verify(root):
    require(__debug__,"checked Python required")
    spec=specification()
    require((root/"execution-freeze.json").read_bytes()==encoded(spec),"freeze/helper identity changed")
    expected=dict(spec["files_sha256"])
    expected.update({"baseline/runtime/"+n:h for n,h in spec["runtime_sha256"]["baseline"].items()})
    actual={str(f.relative_to(root)) for f in root.rglob("*") if f.is_file()}
    require(actual==set(expected)|{"execution-freeze.json"},"frozen inventory mismatch")
    for name,pin in expected.items():
        path=root/name
        require(path.resolve().is_relative_to(root.resolve()) and digest(path.read_bytes())==pin,"frozen artifact mismatch")
    return spec
def readiness(root):
    spec=verify(root)
    raw=subprocess.check_output(["gh","api","--paginate","repos/Sounio-lang/sounio/commits/"+spec["source_commit"]+"/check-runs",
                                 "--jq",".check_runs[] | @json"],text=True,timeout=60)
    checks=source_checks(spec,[json.loads(s) for s in raw.splitlines() if s.strip()])
    return dict(source_commit=spec["source_commit"],freeze_sha256=digest((root/"execution-freeze.json").read_bytes()),
                orchestration_sha256=spec["orchestration_sha256"],source_checks=checks,diagnostic_only=True)
def prerequisites(frozen,arm,stage,baseline=None,baseline_pin=None):
    require(__debug__ and os.environ.get("TMUX"),"remote tmux and checked Python required")
    require(arm=="baseline" and baseline is None and baseline_pin is None,"single diagnostic arm only")
    require(not stage.exists(),"attempt already exists; no retry")
    return verify(frozen),readiness(frozen),None

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
    start=dict(schema="pireus-host-decode-start-v1",arm=arm,source_commit=spec["source_commit"],
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

def collect(frozen,stage,output,arm,job,run=subprocess.check_output):
    require(arm=="baseline" and job.isdigit(),"invalid arm/job")
    spec=verify(frozen)
    start=json.loads((stage/"start.json").read_bytes())
    require(start["arm"]==arm and start["source_commit"]==spec["source_commit"]
            and start["freeze_sha256"]==digest((frozen/"execution-freeze.json").read_bytes()),"start identity mismatch")
    require(len(start["workers"])==2 and [w["node"] for w in start["workers"]]==["spark-3c59","spark-8e54"],"worker order")
    raw=run(KUBE+["exec","slurm-pilot-controller-0","-c","slurmctld","--","sacct","-a","-j",job,"--noheader","--parsable2","--format=JobID,JobName,State,ExitCode,NodeList,Start,End"])
    terminal=accounting(raw,job)
    output.mkdir(exist_ok=False)
    for name in ("start.json","launch.log","exit-code","readiness.json"):
        (output/name).write_bytes((stage/name).read_bytes())
    (output/"accounting.txt").write_bytes(raw)
    missing={}
    for rank,w in enumerate(start["workers"]):
        before=run(KUBE+["get","pod",w["pod"],"-o","json"])
        require(json.loads(before)["metadata"]["uid"]==w["uid"],"worker UID changed")
        (output/f"worker-{rank}-before.json").write_bytes(before)
        paths={f"rank-{rank}/runtime-before.json":f"/scratch/pireus/receipts/overhead-runtime-{job}-{rank}.json",
               f"worker-receipts/rank-{rank}-complete.json":f"/scratch/pireus/receipts/offline-{job}-{rank}-complete.json",
               f"worker-receipts/lifecycle-{job}-{rank}.jsonl":f"/scratch/pireus/receipts/lifecycle-{job}-{rank}.jsonl",
               f"rank-{rank}/boot-id.txt":"/proc/sys/kernel/random/boot_id"}
        paths.update({f"worker-receipts/rank-{rank}-{i:03d}.json":f"/scratch/pireus/receipts/offline-{job}-{rank}-{i:03d}.json" for i in range(8)})
        # Archive bytes, not only remotely asserted checksums.
        paths.update({f"rank-{rank}/runtime/{name}":"/scratch/pireus/runtime/"+name for name in spec["runtime_sha256"][arm]})
        if arm=="observed":
            paths.update({f"rank-{rank}/external/{name}":f"/scratch/pireus/receipts/external-{job}-{rank}/"+name for name in ("target.json","attached.json","binding.json","journal.jsonl","result.json","observer.log")})
        code="import json,base64;from pathlib import Path;paths="+repr(paths)+";result={}\nfor n,p in paths.items():\n try:result[n]={'base64':base64.b64encode(Path(p).read_bytes()).decode()}\n except FileNotFoundError:result[n]={'missing':True}\nprint(json.dumps(result))"
        data=json.loads(run(KUBE+["exec",w["pod"],"-c","slurmd","--","python3","-c",code]))
        require(set(data)==set(paths),"collection inventory mismatch")
        for name,item in data.items():
            if item=={"missing":True}:missing[name]=paths[name];continue
            p=output/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(base64.b64decode(item["base64"],validate=True))
        after=run(KUBE+["get","pod",w["pod"],"-o","json"])
        require(json.loads(after)["metadata"]["uid"]==w["uid"],"worker UID changed during collection")
        (output/f"worker-{rank}-after.json").write_bytes(after)
    for name in ("start.json","launch.log","exit-code","readiness.json"):
        require((stage/name).read_bytes()==(output/name).read_bytes(),"stage changed during collection")
    receipt=dict(schema="pireus-host-decode-custody-v1",arm=arm,job=job,
        source_commit=spec["source_commit"],freeze_sha256=digest((frozen/"execution-freeze.json").read_bytes()),
        terminal_state=terminal[2],missing=missing,hardware_qualified=False,
        files_sha256={str(p.relative_to(output)):digest(p.read_bytes()) for p in output.rglob("*") if p.is_file()})
    (output/"collection.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt

def inspect_probes(rows, job, rank, target_pid=None):
    selected=[r for r in rows if r.get("schema")=="pireus-host-decode-probe-v1" and r.get("rank")==str(rank)]
    require(len(selected)<=30,"too many probe records")
    expected=[(step,stage) for step in range(1,16) for stage in ("HOST_DECODE_BEGIN","HOST_DECODE_END")]
    require([(r.get("step"),r.get("stage")) for r in selected]==expected[:len(selected)],"probe prefix order")
    require(all(type(r["pid"]) is int and r["pid"]>0 for r in selected),"invalid probe PID")
    pids={r["pid"] for r in selected}
    require(len(pids)<=1 and (target_pid is None or not pids or pids=={target_pid}),"probe PID identity")
    missing=[];previous_end=None
    for i,r in enumerate(selected):
        require(r["job"]==job and r["index"]==0 and r["diagnostic_only"] is True
                and r["device_synchronized"] is False,"probe job/scope identity")
        begin=r["monotonic_ns"];duration=r["read_duration_ns"]
        require(type(begin) is int and type(duration) is int and begin>=0 and duration>=0,"probe timestamps")
        require(previous_end is None or begin>=previous_end,"overlapping probe calls")
        previous_end=begin+duration
        require(set(r["files"])=={"host_meminfo","host_vmstat","process_status"},"probe file inventory")
        for name,metric in r["files"].items():
            t=metric["monotonic_ns"];dt=metric["duration_ns"]
            require(type(t) is int and type(dt) is int and dt>=0 and begin<=t<=t+dt<=previous_end,"file time outside probe")
            require((metric["error"] is None and isinstance(metric["raw"],str))
                    or (isinstance(metric["error"],str) and metric["raw"] is None),"invalid file result")
            if metric["error"] is not None:missing.append(dict(record=i,metric=name,error=metric["error"]))
        require(set(r["cuda_allocator"])=={"memory_allocated","memory_reserved"},"allocator inventory")
        for name,metric in r["cuda_allocator"].items():
            require((metric["error"] is None and type(metric["value"]) is int and metric["value"]>=0)
                    or (isinstance(metric["error"],str) and metric["value"] is None),"invalid allocator result")
            if metric["error"] is not None:missing.append(dict(record=i,metric=name,error=metric["error"]))
    return dict(rank=rank,records=len(selected),complete_probe_window=len(selected)==30 and not missing,
                missing_metrics=missing,partial_prefix=len(selected)<30,loaded_model_qualified=False)

def inspect_collection(frozen,root,pin):
    spec=verify(frozen);raw=(root/"collection.json").read_bytes()
    require(digest(raw)==pin,"collection pin mismatch")
    c=json.loads(raw)
    require(c["schema"]=="pireus-host-decode-custody-v1" and c["arm"]=="baseline"
            and c["source_commit"]==spec["source_commit"]
            and c["freeze_sha256"]==digest((frozen/"execution-freeze.json").read_bytes()),"collection identity")
    for name,sha in c["files_sha256"].items():
        path=root/name
        require(path.resolve().is_relative_to(root.resolve()) and digest(path.read_bytes())==sha,"collected artifact mismatch")
    accounting((root/"accounting.txt").read_bytes(),c["job"])
    start=json.loads((root/"start.json").read_bytes())
    require(start["schema"]=="pireus-host-decode-start-v1"
            and start["command"]==command(start["workers"],start["input_sha256"],spec["runtime_sha256"]["baseline"]),
            "diagnostic launch command")
    require(start["orchestration_sha256"]==orchestration_hashes()
            and start["launcher_sha256"]==digest(Path(__file__).read_bytes()),"diagnostic helpers")
    require(start["source_commit"]==spec["source_commit"] and start["freeze_sha256"]==c["freeze_sha256"]
            and start["runtime_before_sha256"]==[spec["runtime_sha256"]["baseline"]]*2
            and start["input_sha256"]==digest((frozen/"without-feedback/offline-bundle.json").read_bytes()),"start source/input")
    ready=json.loads((root/"readiness.json").read_bytes())
    require(ready["source_commit"]==spec["source_commit"] and ready["freeze_sha256"]==c["freeze_sha256"],"readiness identity")
    source_checks(spec,list(ready["source_checks"].values()))
    require(len(start["workers"])==2 and [w["node"] for w in start["workers"]]==["spark-3c59","spark-8e54"],"worker order")
    parsed=[]
    for line in (root/"launch.log").read_bytes().splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if isinstance(r,dict):parsed.append(r)
    require(all(r.get("rank") in ("0","1") for r in parsed if r.get("schema")=="pireus-host-decode-probe-v1"),"unknown probe rank")
    results=[]
    for rank,w in enumerate(start["workers"]):
        for moment in ("before","after"):
            pod=json.loads((root/f"worker-{rank}-{moment}.json").read_bytes())
            require(pod["metadata"]["uid"]==w["uid"] and pod["spec"]["nodeName"]==w["node"],"worker custody")
        require((root/f"rank-{rank}/boot-id.txt").read_text().strip()==w["boot_id"],"boot custody")
        for name,sha in spec["runtime_sha256"]["baseline"].items():
            require(digest((root/f"rank-{rank}/runtime"/name).read_bytes())==sha,"runtime custody")
        lifecycle=root/f"worker-receipts/lifecycle-{c['job']}-{rank}.jsonl"
        target_pid=None
        if lifecycle.exists():
            first=json.loads(lifecycle.read_text().splitlines()[0])
            require(first["job"]==c["job"] and first["rank"]==str(rank),"lifecycle identity")
            target_pid=first["pid"]
        require(target_pid is not None or not any(r.get("schema")=="pireus-host-decode-probe-v1" and r.get("rank")==str(rank) for r in parsed),"probe target lacks lifecycle binding")
        results.append(inspect_probes(parsed,c["job"],rank,target_pid))
    return dict(schema="pireus-host-decode-diagnostic-inspection-v1",job=c["job"],
                collection_sha256=pin,terminal_state=c["terminal_state"],missing_artifacts=c["missing"],
                ranks=results,diagnostic_only=True,loaded_model_qualified=False,
                timing_eligible=False,pilot_acceptance=False)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["create","verify","launch","collect","inspect"])
    parser.add_argument("--frozen",type=Path,required=True)
    parser.add_argument("--stage",type=Path);parser.add_argument("--output",type=Path)
    parser.add_argument("--job");parser.add_argument("--collection-pin")
    args=parser.parse_args()
    if args.mode=="create":result=create(args.frozen)
    elif args.mode=="verify":result=verify(args.frozen)
    elif args.mode=="launch":
        if args.stage is None:parser.error("--stage required")
        raise SystemExit(launch(args.frozen,"baseline",args.stage))
    elif args.mode=="collect":
        if args.stage is None or args.output is None or args.job is None:parser.error("--stage --output --job required")
        result=collect(args.frozen,args.stage,args.output,"baseline",args.job)
    else:
        if args.output is None or args.collection_pin is None:parser.error("--output --collection-pin required")
        result=inspect_collection(args.frozen,args.output,args.collection_pin)
    print(json.dumps(result,indent=2))
