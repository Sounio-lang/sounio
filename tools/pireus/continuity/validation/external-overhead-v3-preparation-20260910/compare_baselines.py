#!/usr/bin/env python3
"""Replay a post hoc comparison of two preserved baselines; no inference."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
SOURCES={
 "11976":("external-overhead-inference-20260909/baseline-11976","510f98de0c30d183fa71caf21b6955ad86a59bbcdcd36de636a9c0f333da738f"),
 "11983":("external-overhead-v3-preparation-20260910/baseline-11983-negative/baseline-collected","22de93eb702c53d3ae901887b04670dc08dcccfd531a8a4c8f2a34667a69a123"),
}
def require(ok,message):
 if not ok:raise ValueError(message)
def read(path):return json.loads(path.read_bytes())
def rows(path):
 result=[]
 for line in path.read_bytes().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if isinstance(r,dict):result.append(r)
 return result
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def analyze():
 packets={}
 for job,(name,pin) in SOURCES.items():
  root=BASE/name
  require(digest(root/"collection.json")==pin,"collection identity changed")
  collection=read(root/"collection.json")
  for name,sha in collection["files_sha256"].items():
   path=root/name
   require(path.resolve().is_relative_to(root.resolve()) and digest(path)==sha,"collected bytes changed")
  packets[job]={"start":read(root/"start.json"),"log":rows(root/"launch.log"),
                "lifecycle":[rows(root/f"worker-receipts/lifecycle-{job}-{r}.jsonl") for r in (0,1)]}
 a,b=packets["11976"],packets["11983"]
 identity={k:a["start"][k]==b["start"][k] for k in ("runtime_before_sha256","input_sha256","workers","command")}
 require(all(identity.values()),"comparison identity changed")
 result=[]
 for rank in (0,1):
  stages={}
  for stage in ("OFFLINE_MODEL_READY","OFFLINE_EXTEND_BEGIN","OFFLINE_EXTEND_END","OFFLINE_FIRST_TOKEN"):
   selected=[]
   for data in (a,b):
    match=[r for r in data["log"] if r.get("rank")==str(rank) and r.get("stage")==stage and r.get("index",0)==0]
    require(len(match)==1,"ambiguous aligned phase")
    selected.append(match[0])
   stages[stage]={"old_available_bytes":selected[0]["available_bytes"],"new_available_bytes":selected[1]["available_bytes"],
                  "delta_available_bytes":selected[1]["available_bytes"]-selected[0]["available_bytes"]}
  aligned=[]
  for stage in ("EXTEND_ENTRY","DECODE_ENTRY","DECODE_SAMPLE"):
   match=[]
   for data in (a,b):
    found=[r for r in data["lifecycle"][rank] if r["stage"]==stage and r.get("index")==0 and r.get("token_index") in (None,0)]
    require(len(found)==1,"ambiguous lifecycle phase")
    match.append(found[0])
   left,right=match
   require(all(r.get(k) is None for r in match for k in ("host_error","process_error","owned_child_error")),"missing phase metric")
   aligned.append({"stage":stage,"cuda_equal":left["cuda"]==right["cuda"],
                   "old_cuda":left["cuda"],"new_cuda":right["cuda"],
                   "host_delta_bytes":{k:right["host"][k]-left["host"][k] for k in left["host"]},
                   "process_delta_bytes":{k:right["process"][k]-left["process"][k] for k in left["process"]},
                   "observation_duration_ns":[left["observation_duration_ns"],right["observation_duration_ns"]]})
  result.append({"rank":rank,"phases":stages,"lifecycle":aligned})
 return {"schema":"pireus-baseline-11976-11983-posthoc-comparison","source_collections":SOURCES,
         "identity_equal":identity,"ranks":result,"completed_responses_per_rank":{"11976":8,"11983":0},
         "external_v3_observer_present":False,"overhead_estimate_available":False,
         "cause_established":False,"limits":["Host MemAvailable is not process PSS or CUDA allocator accounting.",
          "No token-16 sample or completed response exists for 11983.",
          "Environment, kernel and non-model memory are not fully captured by the archived phase metrics."]}
if __name__=="__main__":print(json.dumps(analyze(),indent=2))
