#!/usr/bin/env python3
"""Audit archived response bytes and probe transport; no inference replay."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PIN="b03202231b1bb4f890a8a29c5c0ee202df2bba6408c307b69519fda5c38fe109"
def sha(raw):return hashlib.sha256(raw).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def analyze():
    root=ROOT/"collection"
    require(sha((root/"collection.json").read_bytes())==PIN,"collection pin")
    c=json.loads((root/"collection.json").read_bytes())
    require(c["job"]=="11987" and c["terminal_state"]=="COMPLETED" and not c["missing"],"terminal collection")
    for name,pin in c["files_sha256"].items():
        p=root/name
        require(p.resolve().is_relative_to(root.resolve()) and sha(p.read_bytes())==pin,"artifact: "+name)
    start=json.loads((root/"start.json").read_bytes())
    ranks=[]
    responses=[]
    for rank in range(2):
        complete=json.loads((root/f"worker-receipts/rank-{rank}-complete.json").read_bytes())
        require(complete["job"]=="11987" and complete["rank"]==str(rank) and complete["model_loaded"] is True,"completion identity")
        require(complete["stage"]=="OFFLINE_CYCLE_COMPLETE" and complete["input_sha256"]==start["input_sha256"],"completion scope")
        require(complete["helper_sha256"]==start["runtime_before_sha256"][rank]["offline_generate.py"],"runtime identity")
        require([x["index"] for x in complete["results"]]==list(range(8)),"result inventory")
        outputs=[]
        for item in complete["results"]:
            raw=(root/f"worker-receipts/rank-{rank}-{item['index']:03d}.json").read_bytes()
            require(sha(raw)==item["response_sha256"],"response digest")
            response=json.loads(raw)
            require(response["job"]=="11987" and response["index"]==item["index"] and response["input_sha256"]==start["input_sha256"],"response identity")
            require(response["completion_tokens"]==len(response["output_ids"])==item["output_tokens"],"token count")
            require(response["execution_profile"]==complete["execution_profile"],"profile mismatch")
            outputs.append(raw)
        responses.append(outputs)
        ranks.append(dict(rank=rank,responses=len(outputs),tokens=sum(x["output_tokens"] for x in complete["results"]),execution_profile=complete["execution_profile"]))
    require(responses[0]==responses[1],"rank response bytes differ")
    valid=[];malformed=[];guard_exits=[]
    for number,line in enumerate((root/"launch.log").read_bytes().splitlines(),1):
        try:row=json.loads(line)
        except ValueError as exc:
            if b"pireus-host-decode-probe-v1" in line:
                malformed.append(dict(line=number,bytes=len(line),sha256=sha(line),error=str(exc)))
            continue
        if not isinstance(row,dict):continue
        if row.get("schema")=="pireus-host-decode-probe-v1":valid.append(row)
        if row.get("stage")=="MEMORY_GUARD_CHILD_EXIT" and row.get("job")=="11987":guard_exits.append(row)
    require(sorted(x["rank"] for x in guard_exits)==["0","1"],"guardian completion inventory")
    require(all(x["returncode"]==0 and x["minimum_bytes"]>=33*1024**3 for x in guard_exits),"guardian failure")
    return dict(schema="pireus-host-decode-terminal-summary-v1",job=11987,collection_sha256=PIN,
        terminal_state=c["terminal_state"],ranks=ranks,response_bytes_equal=True,
        inference_completed=True,guardian_exits=guard_exits,
        valid_probe_records_by_rank={str(rank):sum(x.get("rank")==str(rank) for x in valid) for rank in range(2)},
        malformed_probe_lines=malformed,probe_window_qualified=False,
        full_diagnostic_custody_qualified=False,timing_eligible=False,pilot_acceptance=False,
        causal_memory_explanation_established=False)
if __name__=="__main__":print(json.dumps(analyze(),indent=2))
