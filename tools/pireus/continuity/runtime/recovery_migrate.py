#!/usr/bin/env python3
"""Observe and CAS a pinned observer-only freeze migration, admitted by Sounio."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(data):return hashlib.sha256(data).hexdigest()
def fields(text):return dict(line.split("=",1) for line in text.splitlines() if "=" in line)
def tokens(text):return dict(x.split("=",1) for x in text.split() if "=" in x)
def kube(*a):return subprocess.check_output(["kubectl",*a],text=True,timeout=60)
def get(*a):return json.loads(kube(*a,"-o","json"))
def main():
 ap=argparse.ArgumentParser()
 for key in ("old-root","new-root","engine","evidence"):ap.add_argument("--"+key,type=Path,required=True)
 ap.add_argument("--holder",required=True);ap.add_argument("--apply",action="store_true");a=ap.parse_args()
 if not os.environ.get("TMUX"):raise ValueError("remote tmux required")
 lock=json.loads((HERE/"recovery-migration-lock.json").read_text())
 assert sha(a.engine.read_bytes())==lock["engine_sha256"]
 for name,value in lock["sources"].items():assert sha((HERE/name).read_bytes())==value
 old_path=a.old_root/"tools/cluster/spark_pair_arbiter.freeze.v1"
 new_path=a.new_root/"tools/cluster/spark_pair_arbiter.freeze.v1"
 assert sha(old_path.read_bytes())==lock["old_freeze"]
 assert sha(new_path.read_bytes())==lock["new_freeze"]
 old=fields(old_path.read_text());new=fields(new_path.read_text())
 assert {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}=={"material_backend_sha256"}
 cfg=fields((a.old_root/"tools/cluster/spark_pair_arbiter.policy.v1").read_text())
 for root in (a.old_root,a.new_root):
  output=subprocess.check_output([str(root/"scripts/dev/spark_pair_arbiter.sh"),"verify"],text=True,timeout=60)
  assert output.startswith("SPARK_PAIR_VERIFY_PASS")
 a.evidence.mkdir(parents=True,exist_ok=False)
 def save(name,value):
  data=value.encode() if isinstance(value,str) else (json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
  path=a.evidence/name
  with path.open("wb") as out:out.write(data);out.flush();os.fsync(out.fileno())
  fd=os.open(a.evidence,os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return path
 frame_raw=subprocess.check_output([str(a.old_root/"scripts/dev/spark_pair_arbiter_k8s_backend.sh"),
  "--policy",str(a.old_root/"tools/cluster/spark_pair_arbiter.policy.v1"),"--freeze",str(old_path),
  "facts","--holder",a.holder],text=True,timeout=190)
 frame=tokens(frame_raw);save("old-frame.txt",frame_raw)
 lease=get("-n","beagle","get","lease","pireus-spark-pair")
 journal=get("-n","beagle","get","configmap",cfg["bootstrap_journal"])
 nodes=[get("get","node",cfg["node_"+str(i)+"_k8s"]) for i in range(2)]
 nodeset=get("-n",cfg["nodeset_namespace"],"get","nodeset",cfg["nodeset_name"])
 pods=get("get","pods","-A")["items"]
 save("lease-before.json",lease);save("journal-before.json",journal);save("nodes-before.json",nodes)
 reports=[]
 for i in range(2):
  node=cfg["node_"+str(i)+"_k8s"]
  selected=[p for p in pods if p["spec"].get("nodeName")==node and p["metadata"]["namespace"]=="beagle"
    and p["metadata"]["name"].startswith("pireus-spark-host-fence-") and not p["metadata"].get("deletionTimestamp")]
  assert len(selected)==1
  raw=kube("-n","beagle","exec",selected[0]["metadata"]["name"],"-c","host-fence","--","bash","/fence/host-fence.sh","report")
  save(node+".report.txt",raw);reports.append(tokens(raw))
 ann=lease["metadata"]["annotations"];epoch=ann[cfg["epoch_annotation"]]
 expiry=datetime.datetime.fromisoformat(lease["spec"]["renewTime"].replace("Z","+00:00")).timestamp()+lease["spec"]["leaseDurationSeconds"]
 now=datetime.datetime.now(datetime.timezone.utc).timestamp()
 jd=journal["data"]
 journal_bound=jd.get("semanticsFreezeSha256")==lock["old_freeze"]
 # A crash after the first CAS is replayable only for this exact lease/revision.
 journal_partial=(jd.get("semanticsFreezeSha256")==lock["new_freeze"] and
   jd.get("recoveryObserverFromFreeze")==lock["old_freeze"] and
   jd.get("recoveryObserverLeaseUID")==lease["metadata"]["uid"] and
   jd.get("recoveryObserverOldLeaseRV")==lease["metadata"]["resourceVersion"])
 worker_pods=[p for p in pods if p["metadata"]["namespace"]==cfg["nodeset_namespace"] and
   any(o.get("uid")==cfg["nodeset_uid"] for o in p["metadata"].get("ownerReferences",[]))]
 selector=nodeset.get("status",{}).get("selector","")
 labels=dict(x.split("=",1) for x in selector.split(",") if "=" in x)
 assert labels,"unknown worker selector"
 worker_pods += [p for p in pods if p["metadata"]["namespace"]==cfg["nodeset_namespace"] and
   all(p["metadata"].get("labels",{}).get(k)==v for k,v in labels.items())]
 unwanted=[p for p in pods if any(p["metadata"].get("labels",{}).get(k)=="true" for k in
   ("pireus.sounio.dev/spark-pair-reservation","pireus.sounio.dev/spark-pair-workload"))]
 save("workload-inventory.json",dict(worker_uids=[p["metadata"]["uid"] for p in worker_pods],
  unwanted_uids=[p["metadata"]["uid"] for p in unwanted],nodeset_uid=nodeset["metadata"]["uid"],selector=selector))
 checks=[
  old["authority_sha256"]==new["authority_sha256"] and old["native_executable_sha256"]==new["native_executable_sha256"],
  ann[cfg["freeze_hash_annotation"]]==lock["old_freeze"] and ann[cfg["source_hash_annotation"]]==old["authority_sha256"],
  all(n["metadata"]["uid"]==cfg["node_"+str(i)+"_uid"] for i,n in enumerate(nodes)) and nodeset["metadata"]["uid"]==cfg["nodeset_uid"],
  ann[cfg["state_annotation"]]=="RECOVERY_REQUIRED" and lease["spec"]["holderIdentity"]==a.holder and expiry<now,
  all(cfg["slurmd_selector_key"] not in n["metadata"]["labels"] for n in nodes),
  not worker_pods,
  all(r["grant_mode"]=="FENCED" and r["grant_valid"]=="0" for r in reports),
  all(r["freeze_sha256"]==lock["old_freeze"] and r["source_sha256"]==old["authority_sha256"] and
    r["boot_id"]==ann[cfg["host_boot_"+str(i)+"_annotation"]] for i,r in enumerate(reports)),
  all(r["watchdog"]=="1" and 0<=int(r["reported_monotonic"])-int(r["watchdog_monotonic"])<=55 for r in reports),
  all(r["device_barrier"]=="1" and r["device_barrier_source_sha256"]==old["device_barrier_source_sha256"] for r in reports),
  all(r["protected"]=="1" for r in reports),
  all(all(r[k]=="1" for k in ("inventory","services","restarts","docker_claims","consumers","cgroups")) for r in reports),
  not unwanted,
  frame["epoch"]==epoch and frame["observed_epoch"]==epoch and all(r["grant_epoch"]==epoch and
    r["lease_uid"]==lease["metadata"]["uid"] and r["grant_owner"]==a.holder for r in reports),
  int(frame["slurm_mask"]),
  (journal_bound or journal_partial) and jd.get("sounioSourceSha256")==old["authority_sha256"],
 ]
 path=save("facts.v1"," ".join(map(str,[1]+[int(x) for x in checks]+[int(r["mem_available_mb"]) for r in reports]))+"\n")
 run=subprocess.run([str(a.engine.resolve()),str(path.resolve())],capture_output=True,text=True,timeout=60)
 save("decision.json",run.stdout);receipt=json.loads(run.stdout)
 save("custody.json",dict(lock=lock,facts_sha256=sha(path.read_bytes()),apply=a.apply))
 assert run.returncode==0 and receipt["authority"]=="Sounio" and receipt["decision"]=="ALLOW"
 assert receipt["action"]=="REBIND_RECOVERY_OBSERVER" and receipt["gpu_grant"] is False and receipt["resume"] is False
 if not a.apply:print("RECOVERY_OBSERVER_MIGRATION_REVIEWABLE_ALLOW");return
 # A second complete native admission immediately precedes the commit sequence.
 fresh=a.evidence/"recheck"
 subprocess.run([sys.executable,str(Path(__file__).resolve()),"--old-root",str(a.old_root.resolve()),
  "--new-root",str(a.new_root.resolve()),"--engine",str(a.engine.resolve()),"--holder",a.holder,
  "--evidence",str(fresh.resolve())],check=True,timeout=240)
 fresh_lease=json.loads((fresh/"lease-before.json").read_text())
 assert fresh_lease["metadata"]["resourceVersion"]==lease["metadata"]["resourceVersion"]
 fresh_journal=json.loads((fresh/"journal-before.json").read_text())
 assert fresh_journal["metadata"]["resourceVersion"]==journal["metadata"]["resourceVersion"]
 # Journal prepares first, lease CAS commits. Neither changes scheduler state.
 current=get("-n","beagle","get","lease","pireus-spark-pair")
 assert current["metadata"]["resourceVersion"]==lease["metadata"]["resourceVersion"]
 def esc(s):return s.replace("~","~0").replace("/","~1")
 updated=dict(jd,semanticsFreezeSha256=lock["new_freeze"],recoveryObserverFromFreeze=lock["old_freeze"],
   recoveryObserverToFreeze=lock["new_freeze"],recoveryObserverLeaseUID=lease["metadata"]["uid"],
   recoveryObserverOldLeaseRV=lease["metadata"]["resourceVersion"],recoveryObserverDecisionSHA=sha(run.stdout.encode()))
 jp=[dict(op="test",path="/metadata/uid",value=journal["metadata"]["uid"]),
   dict(op="test",path="/metadata/resourceVersion",value=journal["metadata"]["resourceVersion"]),
   dict(op="replace",path="/data",value=updated)]
 save("journal-patch.json",jp);save("phase.json",dict(phase="JOURNAL_INTENT"))
 save("journal-effect.txt",kube("-n","beagle","patch","configmap",cfg["bootstrap_journal"],"--type=json","-p",json.dumps(jp)))
 lp=[dict(op="test",path="/metadata/uid",value=lease["metadata"]["uid"]),
   dict(op="test",path="/metadata/resourceVersion",value=lease["metadata"]["resourceVersion"]),
   dict(op="test",path="/metadata/annotations/"+esc(cfg["state_annotation"]),value="RECOVERY_REQUIRED"),
   dict(op="test",path="/metadata/annotations/"+esc(cfg["freeze_hash_annotation"]),value=lock["old_freeze"]),
   dict(op="replace",path="/metadata/annotations/"+esc(cfg["freeze_hash_annotation"]),value=lock["new_freeze"])]
 save("lease-patch.json",lp);save("phase.json",dict(phase="LEASE_INTENT"))
 save("lease-effect.txt",kube("-n","beagle","patch","lease","pireus-spark-pair","--type=json","-p",json.dumps(lp)))
 post=get("-n","beagle","get","lease","pireus-spark-pair");save("lease-after.json",post)
 assert post["spec"]==lease["spec"]
 expected=dict(ann);expected[cfg["freeze_hash_annotation"]]=lock["new_freeze"]
 assert post["metadata"]["annotations"]==expected
 post_journal=get("-n","beagle","get","configmap",cfg["bootstrap_journal"]);save("journal-after.json",post_journal)
 assert post_journal["data"]==updated
 for i in range(2):
  node=cfg["node_"+str(i)+"_k8s"]
  fresh_pods=get("-n","beagle","get","pods")["items"]
  selected=[p for p in fresh_pods if p["spec"].get("nodeName")==node and
    p["metadata"]["name"].startswith("pireus-spark-host-fence-") and not p["metadata"].get("deletionTimestamp")]
  assert len(selected)==1
  raw=kube("-n","beagle","exec",selected[0]["metadata"]["name"],"-c","host-fence","--","bash","/fence/host-fence.sh","report")
  save(node+".post-report.txt",raw);r=tokens(raw)
  assert r["grant_mode"]=="FENCED" and r["grant_valid"]=="0"
  assert all(r[k]=="1" for k in ("watchdog","device_barrier","protected"))
 save("phase.json",dict(phase="OBSERVER_REBOUND",gpu_grant=False,scheduler_state="RECOVERY_REQUIRED"))
 print("RECOVERY_OBSERVER_REBOUND; canonical recovery required")
if __name__=="__main__":main()
