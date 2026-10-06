#!/usr/bin/env python3
"""Terminal receipt collection and immutable custody checks; never submit a job."""
import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from freeze_external_overhead_v3 import verify, digest, HERE, orchestration_hashes
from check_frozen_feedback import source_checks
from evaluate_external_overhead import require, lifecycle_span
from evaluate_external_overhead_v3 import evaluate
sys.path.insert(0,str(HERE))
from tokenized_cycle import accept_offline

NODES={"gpuorangefs-multi-spark-3c59","gpuorangefs-multi-spark-8e54"}
KUBE=["kubectl","-n","slurm-pilot"]

def accounting(raw,job,success=False):
    main=[line.split("|") for line in raw.decode().splitlines() if line.split("|")[0]==job]
    require(len(main)==1 and len(main[0])>=7,"unique accounting required")
    r=main[0]
    require(r[1]=="pireus-inkling-offline-generate" and set(r[4].split(","))==NODES,"job name/nodes mismatch")
    require(all(t not in ("","Unknown") for t in r[5:7]),"accounting timestamps missing")
    require(r[2].split()[0] in {"COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY","PREEMPTED","BOOT_FAIL","DEADLINE"},"job not terminal")
    if success:require(r[2:4]==["COMPLETED","0:0"],"job not successful")
    return r

def load_packet(root,pin):
    raw=(root/"collection.json").read_bytes()
    require(digest(raw)==pin,"collection hash mismatch")
    c=json.loads(raw)
    require(c["schema"]=="pireus-external-overhead-custody-v1","collection schema")
    for name,sha in c["files_sha256"].items():
        p=root/name
        require(p.resolve().is_relative_to(root.resolve()) and digest(p.read_bytes())==sha,"collected artifact mismatch: "+name)
    require(not c["missing"],"incomplete collection")
    return c

def collect(frozen,stage,output,arm,job,run=subprocess.check_output):
    require(arm in ("baseline","observed") and job.isdigit(),"invalid arm/job")
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
    receipt=dict(schema="pireus-external-overhead-custody-v1",arm=arm,job=job,
        source_commit=spec["source_commit"],freeze_sha256=digest((frozen/"execution-freeze.json").read_bytes()),
        terminal_state=terminal[2],missing=missing,hardware_qualified=False,
        files_sha256={str(p.relative_to(output)):digest(p.read_bytes()) for p in output.rglob("*") if p.is_file()})
    (output/"collection.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt

def external_identity(p,worker,job,rank,lifecycle,runtime):
    def read(n):return json.loads((p/n).read_bytes())
    t,a,b,r=[read(n) for n in ("target.json","attached.json","binding.json","result.json")]
    require(t["schema"]=="pireus-observer-target-handoff-v1" and a["schema"]=="pireus-observer-attachment-ack-v1","handoff schema")
    require(t["job"]==job and t["rank"]==str(rank) and t["boot_id"]==worker["boot_id"]
            and t["pid"]==lifecycle[0]["pid"] and t["entry_sha256"]==runtime["offline_generate.py"],"target identity")
    require(a["nonce"]==t["nonce"]==r["nonce"] and a["first_sample_valid"] is True,"ack identity")
    require(all(a[k]==t[k] for k in ("job","rank","pid","starttime_ticks")),"ack target mismatch")
    require(a["handoff_sha256"]==digest((p/"target.json").read_bytes()) and a["binding_sha256"]==digest((p/"binding.json").read_bytes()),"ack hashes")
    require(b["expected"]=={k:t[k] for k in ("job","rank","pid","starttime_ticks","boot_id")}|{"worker_uid":worker["uid"]},"binding expected identity")
    require(b["observer_helper_sha256"]==runtime["external_memory_observer.py"],"observer source mismatch")
    require(b["schema"]=="pireus-external-observer-binding-v1" and all(b["observed"][k]==v for k,v in b["expected"].items()),"observed binding mismatch")
    require("job_"+job in Path(b["observed"]["cgroup"]["membership"]).parts,"observed cgroup job mismatch")
    require(r["integration_complete"] is True and r["guardian_returncode"]==0 and r["observer_returncode"]==3
            and r["job"]==job and r["rank"]==str(rank),"supervisor completion")
    for name,sha in r["files_sha256"].items():
        path=p/name
        require(path.resolve().is_relative_to(p.resolve()) and digest(path.read_bytes())==sha,"supervisor file hash")
    rows=[json.loads(s) for s in (p/"journal.jsonl").read_text().splitlines()]
    binding_sha=digest(json.dumps(b,sort_keys=True).encode())
    require(all(x["binding_sha256"]==binding_sha and x["observer_pid"]==a["observer_pid"] for x in rows),"journal binding identity")
    return rows

def arm_data(frozen,root,pin,arm):
    spec=verify(frozen);c=load_packet(root,pin)
    require(c["arm"]==arm and c["source_commit"]==spec["source_commit"]
            and c["freeze_sha256"]==digest((frozen/"execution-freeze.json").read_bytes()),"collection freeze mismatch")
    job=c["job"];accounting((root/"accounting.txt").read_bytes(),job,success=True)
    require((root/"exit-code").read_text().strip()=="0","launcher failure")
    start=json.loads((root/"start.json").read_bytes())
    require(start["arm"]==arm and start["freeze_sha256"]==c["freeze_sha256"] and start["source_commit"]==c["source_commit"],"launch identity")
    require(len(start["workers"])==2,"worker count")
    from launch_external_overhead_v3 import command
    require(start["command"]==command(start["workers"],start["input_sha256"],spec["runtime_sha256"][arm]),"launch command mismatch")
    require(start["launcher_sha256"]==digest((HERE/"ops/launch_external_overhead_v3.py").read_bytes())
            and start["orchestration_sha256"]==orchestration_hashes(),"orchestration source mismatch")
    ready=json.loads((root/"readiness.json").read_bytes())
    require(ready["source_commit"]==spec["source_commit"] and ready["freeze_sha256"]==c["freeze_sha256"]
            and ready["orchestration_sha256"]==spec["orchestration_sha256"],"arm readiness identity")
    source_checks(spec,list(ready["source_checks"].values()))

    require(start["runtime_before_sha256"]==[spec["runtime_sha256"][arm]]*2,"prelaunch runtime hashes")
    require(start["input_sha256"]==digest((frozen/"without-feedback/offline-bundle.json").read_bytes()),"prelaunch input hash")
    data=dict(job=job,responses=[],lifecycle=[],guardian=[],external=[],workers=start["workers"])
    for line in (root/"launch.log").read_bytes().splitlines():
        try:r=json.loads(line)
        except ValueError:continue
        if isinstance(r,dict):data["guardian"].append(r)
    require(not any(r.get("stage")=="MEMORY_GUARD_STOP" for r in data["guardian"]),"guardian stop present")
    worker=root/"worker-receipts"
    for rank,w in enumerate(start["workers"]):
        require(w["node"]==["spark-3c59","spark-8e54"][rank],"worker node/rank mismatch")
        for moment in ("before","after"):
            p=json.loads((root/f"worker-{rank}-{moment}.json").read_bytes())
            require(p["metadata"]["uid"]==w["uid"] and p["spec"]["nodeName"]==w["node"],"worker custody")
        require((root/f"rank-{rank}/boot-id.txt").read_text().strip()==w["boot_id"],"boot changed")
        for name,sha in spec["runtime_sha256"][arm].items():
            require(digest((root/f"rank-{rank}/runtime"/name).read_bytes())==sha,"runtime snapshot mismatch")
        complete=json.loads((worker/f"rank-{rank}-complete.json").read_bytes())
        require(complete["job"]==job and complete["stage"]=="OFFLINE_CYCLE_COMPLETE"
                and complete["execution_profile"]["actual_swa_tokens"]==896,"completion identity")
        rows=[json.loads(s) for s in (worker/f"lifecycle-{job}-{rank}.jsonl").read_text().splitlines()]
        barrier=json.loads((root/f"rank-{rank}/runtime-before.json").read_bytes())
        require(barrier["stage"]=="OVERHEAD_RUNTIME_VERIFIED" and barrier["job"]==job and barrier["rank"]==str(rank),"runtime barrier identity")
        require(barrier["runtime_sha256"]==spec["runtime_sha256"][arm] and barrier["input_sha256"]==start["input_sha256"],"runtime barrier hashes")
        require(barrier["boot_id"]==w["boot_id"] and barrier["worker_uid"]==w["uid"] and barrier["monotonic_ns"]<rows[0]["monotonic_ns"],"runtime barrier worker/time")
        lifecycle_span(rows,job,rank);data["lifecycle"].append(rows)
        data["responses"].append([(worker/f"rank-{rank}-{i:03d}.json").read_bytes() for i in range(8)])
        if arm=="observed":data["external"].append(external_identity(root/f"rank-{rank}/external",w,job,rank,rows,spec["runtime_sha256"][arm]))
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);(p/"offline-bundle.json").write_bytes((frozen/"without-feedback/offline-bundle.json").read_bytes())
        accept_offline(p,{"transport":"sglang-offline-token-ids","budget":8},worker,runtime_root=frozen/arm/"runtime")
    return data

def qualify(frozen,baseline,baseline_pin,observed,observed_pin):
    spec=verify(frozen)
    left=arm_data(frozen,baseline,baseline_pin,"baseline")
    right=arm_data(frozen,observed,observed_pin,"observed")
    require(left["workers"]==right["workers"],"worker identity changed between arms")
    baseline_account=accounting((baseline/"accounting.txt").read_bytes(),left["job"],True)
    observed_account=accounting((observed/"accounting.txt").read_bytes(),right["job"],True)
    require(baseline_account[6]<=observed_account[5],"arm order or overlap")
    start=json.loads((observed/"start.json").read_bytes())
    require(start["baseline_collection_sha256"]==baseline_pin,"observed baseline binding")
    result=evaluate(spec["screening_specification"],left,right)
    result.update(custody_required_separately=False,custody_verified=True,
        loaded_model_overhead_qualified=True,qualification_scope=spec["experiment"],
        collections={"baseline":baseline_pin,"observed":observed_pin})
    return result

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    c=sub.add_parser("collect")
    for name in ("frozen","stage","output"):c.add_argument("--"+name,type=Path,required=True)
    c.add_argument("--arm",choices=["baseline","observed"],required=True);c.add_argument("--job",required=True)
    q=sub.add_parser("qualify")
    for name in ("frozen","baseline","observed","output"):q.add_argument("--"+name,type=Path,required=True)
    q.add_argument("--baseline-pin",required=True);q.add_argument("--observed-pin",required=True)
    a=p.parse_args()
    if a.mode=="collect":collect(a.frozen,a.stage,a.output,a.arm,a.job)
    else:
        result=qualify(a.frozen,a.baseline,a.baseline_pin,a.observed,a.observed_pin)
        with a.output.open("x") as out:out.write(json.dumps(result,indent=2)+"\n")
