import json,sys,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
arm,job=sys.argv[1:]
stage=root/arm/"collected"
digest=lambda raw:hashlib.sha256(raw).hexdigest()
custody=json.loads((stage/"collection.json").read_bytes())
assert custody["job"]==job and custody["accounting_state"]=="COMPLETED" and not custody["missing"]
for name,sha in custody["files_sha256"].items():
 assert digest((stage/name).read_bytes())==sha
assert (stage/"exit-code").read_text().strip()=="0"
account=[x.split("|") for x in (stage/"accounting.txt").read_text().splitlines()]
assert len([r for r in account if r[0]==job and r[2:4]==["COMPLETED","0:0"]])==1
token=json.loads((root/arm/"accepted/token-custody.json").read_bytes())
assert token["job"]==job and token["complete_token_receipts_valid"] is True
logs=[]
for line in (stage/"launch.log").read_bytes().splitlines():
 try:logs.append(json.loads(line))
 except ValueError:pass
guards=[r for r in logs if r.get("stage")=="MEMORY_GUARD_CHILD_EXIT"]
assert len(guards)==2 and {r["rank"] for r in guards}=={"0","1"}
assert all(r["job"]==job and r["returncode"]==0 and r["minimum_bytes"]>=33*1024**3 for r in guards)
summary=[]
required=["EXTEND_ENTRY","DECODE_ENTRY","DECODE_EXIT","PROPOSAL_SAVED","CLEANUP_BEFORE","CLEANUP_AFTER","REFERENCES_RELEASED"]
for rank in (0,1):
 rows=[json.loads(x) for x in (stage/f"worker-receipts/lifecycle-{job}-{rank}.jsonl").read_bytes().splitlines()]
 assert rows[0]["stage"]=="OBSERVER_START" and rows[-1]["stage"]=="OBSERVER_END"
 assert sum(r["stage"]=="OBSERVER_START" for r in rows)==sum(r["stage"]=="OBSERVER_END" for r in rows)==1
 assert len({r["pid"] for r in rows})==1
 assert all(r["job"]==job and r["rank"]==str(rank) and r["schema"]=="pireus-lifecycle-observation-v1" for r in rows)
 assert all(type(r["monotonic_ns"]) is int and r["observation_duration_ns"]>=0 for r in rows)
 for i in range(8):
  events=[r for r in rows if r["index"]==i and r["stage"] in required]
  assert [r["stage"] for r in events]==required
  assert [r["monotonic_ns"] for r in events]==sorted(r["monotonic_ns"] for r in events)
  assert any(r["stage"]=="DECODE_SAMPLE" and r["index"]==i for r in rows)
  raw=(stage/f"worker-receipts/rank-{rank}-{i:03d}.json").read_bytes()
  hits=[r for r in logs if r.get("stage")=="OFFLINE_PROPOSAL_SAVED" and r.get("rank")==str(rank) and r.get("index")==i]
  assert len(hits)==1 and hits[0]["job"]==job and hits[0]["response_sha256"]==digest(raw)
 assert all(sum(r["stage"]==s for r in rows)==8 for s in required)
 completion=json.loads((stage/f"worker-receipts/rank-{rank}-complete.json").read_bytes())
 assert len([r for r in logs if r.get("stage")=="OFFLINE_CYCLE_COMPLETE" and r.get("rank")==str(rank) and r.get("job")==job])==1
 summary.append(dict(rank=rank,rows=len(rows),output_tokens=sum(x["output_tokens"] for x in completion["results"]),
  max_host_sample_gap_ns=max(r.get("sample_gap_ns",0) for r in rows),
  hook_observation_duration_ns_sum=sum(r["observation_duration_ns"] for r in rows if r["stage"] in required or r["stage"]=="DECODE_SAMPLE"),
  post_release=[{k:r.get(k) for k in ("index","host","process","cuda")} for r in rows if r["stage"]=="REFERENCES_RELEASED"],
  observation_errors=[r for r in rows if r.get("host_error") or r.get("process_error") or r.get("owned_child_error")]))
result=dict(schema="pireus-lifecycle-arm-qualification-v1",job=job,arm=arm,completed_requests=8,
 inference_completed=True,journals_complete=True,guardian=guards,ranks=summary,
 paired_diagnostic_complete=False,pilot_acceptance=False,instrumentation_overhead_qualified=False,
 collection_sha256=digest((stage/"collection.json").read_bytes()),
 token_custody_sha256=digest((root/arm/"accepted/token-custody.json").read_bytes()),
 qualifier_sha256=digest(Path(__file__).read_bytes()))
with (root/arm/"qualification.json").open("x") as out:out.write(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items() if k!="ranks"},indent=2))
