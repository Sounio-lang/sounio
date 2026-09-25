import json,subprocess,sys,hashlib,base64,shutil
from pathlib import Path
repo=Path("/workspace/.wt/pireus-integration-20260906")
attempt=Path("/workspace/.cache/pireus-continuity/feedback-smoke-inference-20260909-v1")
arm="without-feedback"; job="11969"; stage=attempt/arm/"generation"
assert (stage/"exit-code").read_text().strip()=="0"
accounting=subprocess.check_output(["kubectl","-n","slurm-pilot","exec","slurm-pilot-controller-0","-c","slurmctld","--","sacct","-j",job,"--noheader","--parsable2","--format=JobID,JobName,State,ExitCode,NodeList,Start,End"])
rows=[x.split("|") for x in accounting.decode().splitlines()]
main=[x for x in rows if x[0]==job]
assert len(main)==1 and main[0][2:4]==["COMPLETED","0:0"]
assert set(main[0][4].split(","))=={"gpuorangefs-multi-spark-3c59","gpuorangefs-multi-spark-8e54"}
assert main[0][5] not in ("Unknown","") and main[0][6] not in ("Unknown","")
(stage/"accounting.txt").write_bytes(accounting)
logs=[]
for line in (stage/"launch.log").read_text().splitlines():
 try: logs.append(json.loads(line))
 except ValueError: pass
launch=[r for r in logs if r.get("mode")=="offline-generate"]
assert len(launch)==1
pods=launch[0]["nodes"]
worker=stage/"worker-receipts"; worker.mkdir(exist_ok=False)
for rank,pod in enumerate(pods):
 live=json.loads(subprocess.check_output(["kubectl","-n","slurm-pilot","get","pod",pod["pod"],"-o","json"]))
 assert live["metadata"]["uid"]==pod["uid"]
 paths={f"rank-{rank}-complete.json":f"/scratch/pireus/receipts/offline-{job}-{rank}-complete.json"}
 paths.update({f"rank-{rank}-{i:03d}.json":f"/scratch/pireus/receipts/offline-{job}-{rank}-{i:03d}.json" for i in range(8)})
 code="import json,base64;from pathlib import Path;p="+repr(paths)+";print(json.dumps({k:base64.b64encode(Path(v).read_bytes()).decode() for k,v in p.items()}))"
 result=json.loads(subprocess.check_output(["kubectl","-n","slurm-pilot","exec",pod["pod"],"-c","slurmd","--","python3","-c",code]))
 for n,data in result.items(): (worker/n).write_bytes(base64.b64decode(data))
sys.path.insert(0,str(repo/"tools/pireus/continuity"))
from tokenized_cycle import accept_offline
from cycle import digest
f=json.loads((attempt/"execution-freeze.json").read_text())
for name,h in f["runtime_sha256"].items():
 assert digest((repo/"tools/pireus/continuity/runtime"/name).read_bytes())==h
accepted=stage/"accepted"; accepted.mkdir(exist_ok=False)
shutil.copyfile(attempt/arm/"offline-bundle.json",accepted/"offline-bundle.json")
accept_offline(accepted,dict(transport="sglang-offline-token-ids",budget=8),worker)
for rank in (0,1):
 r=json.loads((worker/f"rank-{rank}-complete.json").read_text())
 assert r["job"]==job and r["stage"]=="OFFLINE_CYCLE_COMPLETE"
 assert r["execution_profile"]["actual_swa_tokens"]==896
 assert [x["index"] for x in r["results"]]==list(range(8))
 assert len([x for x in logs if x.get("stage")=="OFFLINE_CYCLE_COMPLETE" and x.get("rank")==str(rank)])==1
 for i in range(8):
  raw=(worker/f"rank-{rank}-{i:03d}.json").read_bytes()
  hits=[x for x in logs if x.get("stage")=="OFFLINE_PROPOSAL_SAVED" and x.get("rank")==str(rank) and x.get("index")==i]
  assert len(hits)==1 and hits[0]["response_sha256"]==digest(raw)
guards=[x for x in logs if x.get("stage")=="MEMORY_GUARD_CHILD_EXIT"]
assert len(guards)==2 and all(x["returncode"]==0 and x["minimum_bytes"]>=33*1024**3 for x in guards)
summary=dict(schema="pireus-feedback-arm-qualification-v1",job=int(job),arm=arm,completed_requests=8,inference_completed=True,paired_response_bytes_equal=True,output_tokens=sum(x["output_tokens"] for x in r["results"]),guardian=guards,pilot_acceptance=False,files_sha256={str(p.relative_to(stage)):digest(p.read_bytes()) for p in stage.rglob("*") if p.is_file()})
(stage/"qualification.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k!="files_sha256"},indent=2))
