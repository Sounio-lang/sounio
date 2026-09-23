"""Supplement terminal custody with per-rank journals; never submit a model job."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from host_decode_attempt import inspect_probes
from host_decode_journal import read_journal
from external_overhead_custody_v3 import KUBE, accounting

RUNTIME_SHA="d05c50461b739e9880c06cd488c06dde74d0012286cde27e1e91cabc9a1ad0fd"
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def pinned(root,name,pin):
    path=root/name
    require(path.resolve().is_relative_to(root.resolve()),"path escapes archive")
    raw=path.read_bytes()
    require(sha(raw)==pin,"artifact digest")
    return raw
def base_collection(root,pin):
    c=json.loads(pinned(root,"collection.json",pin))
    require(c["schema"]=="pireus-host-decode-custody-v1" and c["arm"]=="baseline","base schema")
    for name,digest in c["files_sha256"].items():pinned(root,name,digest)
    def read(name):return pinned(root,name,c["files_sha256"][name])
    terminal=accounting(read("accounting.txt"),c["job"])
    require(terminal[2]==c["terminal_state"],"terminal identity")
    start=json.loads(read("start.json"))
    require(start["source_commit"]==c["source_commit"] and start["freeze_sha256"]==c["freeze_sha256"],"base source")
    require([w["node"] for w in start["workers"]]==["spark-3c59","spark-8e54"],"worker order")
    for rank,w in enumerate(start["workers"]):
        require(start["runtime_before_sha256"][rank]["offline_generate.py"]==RUNTIME_SHA,"journal runtime required")
        require(sha(read(f"rank-{rank}/runtime/offline_generate.py"))==RUNTIME_SHA,"runtime bytes")
        require(read(f"rank-{rank}/boot-id.txt").decode().strip()==w["boot_id"],"base boot")
        for moment in ("before","after"):
            pod=json.loads(read(f"worker-{rank}-{moment}.json"))
            require(pod["metadata"]["uid"]==w["uid"] and pod["spec"]["nodeName"]==w["node"],"base worker")
    return c,start,read
def collect(base,pin,output,run=subprocess.check_output):
    c,start,_=base_collection(base,pin)
    output.mkdir(exist_ok=False)
    files={};missing={}
    for rank,w in enumerate(start["workers"]):
        for moment in ("before","after"):
            raw=run(KUBE+["get","pod",w["pod"],"-o","json"])
            pod=json.loads(raw)
            require(pod["metadata"]["uid"]==w["uid"] and pod["spec"]["nodeName"]==w["node"],"journal worker changed")
            name=f"worker-{rank}-{moment}.json";(output/name).write_bytes(raw);files[name]=sha(raw)
            if moment=="after":continue
            remote=f"/scratch/pireus/receipts/host-decode-{c['job']}-{rank}.jsonl"
            code="import pathlib,json,base64; p=pathlib.Path("+repr(remote)+"); print(json.dumps({'boot':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'data':base64.b64encode(p.read_bytes()).decode() if p.exists() else None}))"
            data=json.loads(run(KUBE+["exec",w["pod"],"-c","slurmd","--","python3","-c",code]))
            require(data["boot"]==w["boot_id"],"journal boot changed")
            boot_name=f"worker-{rank}-boot.txt";boot_raw=(data["boot"]+"\n").encode()
            (output/boot_name).write_bytes(boot_raw);files[boot_name]=sha(boot_raw)
            name=f"rank-{rank}.jsonl"
            if data["data"] is None:missing[name]=remote
            else:
                raw=base64.b64decode(data["data"],validate=True)
                (output/name).write_bytes(raw);files[name]=sha(raw)
    receipt=dict(schema="pireus-host-journal-custody-v1",job=c["job"],base_collection_sha256=pin,
                 files_sha256=files,missing=missing,source_qualified=False,hardware_qualified=False)
    (output/"collection.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt
def inspect(base,pin,root,journal_pin):
    c,start,read=base_collection(base,pin)
    j=json.loads(pinned(root,"collection.json",journal_pin))
    require(j["schema"]=="pireus-host-journal-custody-v1" and j["job"]==c["job"]
            and j["base_collection_sha256"]==pin,"journal collection identity")
    expected={f"worker-{rank}-{moment}.json" for rank in range(2) for moment in ("before","after")}
    expected.update(f"worker-{rank}-boot.txt" for rank in range(2))
    journal_names={f"rank-{rank}.jsonl" for rank in range(2)}
    require(set(j["files_sha256"])|set(j["missing"])==expected|journal_names
            and not (set(j["files_sha256"])&set(j["missing"]))
            and set(j["missing"])<=journal_names,"journal inventory")
    for name,digest in j["files_sha256"].items():pinned(root,name,digest)
    audit_path=Path(__file__).resolve().parents[1]/"validation/external-overhead-v3-preparation-20260910/host-decode-probe/runtime-binding-audit/audit.py"
    require(sha(audit_path.read_bytes())=="6d5ae6cbcac07929006d682ae20618fcfaa8b753846b01ae109c4cde4285f753","runtime auditor changed")
    spec=importlib.util.spec_from_file_location("host_journal_binding",audit_path)
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    binding=auditor.audit(base,pin)
    results=[]
    for rank,w in enumerate(start["workers"]):
        for moment in ("before","after"):
            name=f"worker-{rank}-{moment}.json"
            pod=json.loads(pinned(root,name,j["files_sha256"][name]))
            require(pod["metadata"]["uid"]==w["uid"] and pod["spec"]["nodeName"]==w["node"],"journal worker identity")
        boot_name=f"worker-{rank}-boot.txt"
        require(pinned(root,boot_name,j["files_sha256"][boot_name]).decode().strip()==w["boot_id"],"journal boot identity")
        name=f"rank-{rank}.jsonl"
        lifecycle_name=f"worker-receipts/lifecycle-{c['job']}-{rank}.jsonl"
        if name in j["missing"]:
            results.append(dict(rank=rank,records=0,complete_probe_window=False,missing_journal=True))
            continue
        require(lifecycle_name in c["files_sha256"],"journal lacks lifecycle")
        lifecycle=[json.loads(line) for line in read(lifecycle_name).splitlines()]
        require(bool(lifecycle),"empty lifecycle")
        pid=lifecycle[0]["pid"]
        require(type(pid) is int and pid>0 and all(row["pid"]==pid for row in lifecycle),"lifecycle PID")
        rows=read_journal(root/name,j["files_sha256"][name],c["job"],rank,pid)
        first=lifecycle[0]["monotonic_ns"];last=lifecycle[-1]["monotonic_ns"]
        require(all(first<=row["monotonic_ns"]<=row["monotonic_ns"]+row["read_duration_ns"]<=last for row in rows),"probe outside lifecycle")
        results.append(inspect_probes(rows,c["job"],rank,pid))
    return dict(schema="pireus-host-journal-inspection-v1",job=c["job"],
        base_collection_sha256=pin,journal_collection_sha256=journal_pin,ranks=results,runtime_binding=binding,
        complete_probe_windows=all(r["complete_probe_window"] for r in results),
        source_qualified=False,full_diagnostic_custody_qualified=False,
        loaded_model_qualified=False,timing_eligible=False,pilot_acceptance=False)
