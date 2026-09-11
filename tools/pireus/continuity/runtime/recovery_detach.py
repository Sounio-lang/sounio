#!/usr/bin/env python3
"""Transport one bounded Sounio recovery decision into exact node label removal."""
import argparse,datetime,hashlib,json,os,subprocess,time,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
NODES=("spark-3c59","spark-8e54")
def sha(data):return hashlib.sha256(data).hexdigest()
def fields(raw):return dict(x.split("=",1) for x in raw.split() if "=" in x)
def kube(*args):return subprocess.check_output(["kubectl",*args],text=True,timeout=60)
def jsonk(*args):return json.loads(kube(*args,"-o","json"))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--engine",type=Path,required=True)
 ap.add_argument("--arbiter-root",type=Path,required=True);ap.add_argument("--evidence",type=Path,required=True)
 ap.add_argument("--holder",required=True);ap.add_argument("--apply",action="store_true");a=ap.parse_args()
 if not os.environ.get("TMUX"):raise ValueError("remote tmux required")
 lock=json.loads((HERE/"recovery-detach-lock.json").read_text())
 assert sha(a.engine.read_bytes())==lock["executable_sha256"],"native authority drift"
 for name,expected in lock["sources"].items():assert sha((HERE/name).read_bytes())==expected,"source drift"
 policy=a.arbiter_root/"tools/cluster/spark_pair_arbiter.policy.v1"
 freeze=a.arbiter_root/"tools/cluster/spark_pair_arbiter.freeze.v1"
 cfg=fields(policy.read_text());frozen=fields(freeze.read_text())
 assert sha(freeze.read_bytes())==lock["arbiter_freeze_sha256"],"unexpected predecessor freeze"
 verified=subprocess.check_output([str(a.arbiter_root/"scripts/dev/spark_pair_arbiter.sh"),"verify"],text=True,timeout=60)
 assert verified.startswith("SPARK_PAIR_VERIFY_PASS")
 a.evidence.mkdir(parents=True,exist_ok=False)
 def save(name,value):
  data=value.encode() if isinstance(value,str) else (json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
  path=a.evidence/name
  with path.open("wb") as out:out.write(data);out.flush();os.fsync(out.fileno())
  fd=os.open(a.evidence,os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return path
 base=[str(a.arbiter_root/"scripts/dev/spark_pair_arbiter_k8s_backend.sh"),"--policy",str(policy),"--freeze",str(freeze)]
 raw_frame=subprocess.check_output(base+["facts","--holder",a.holder],text=True,timeout=190)
 frame=fields(raw_frame);save("arbiter-frame.txt",raw_frame)
 lease=jsonk("-n","beagle","get","lease","pireus-spark-pair")
 save("lease.json",lease)
 nodes=[jsonk("get","node",n) for n in NODES]
 nodeset=jsonk("-n",cfg["nodeset_namespace"],"get","nodeset",cfg["nodeset_name"])
 save("nodes.json",nodes);save("nodeset.json",nodeset)
 key=cfg["slurmd_selector_key"]
 selectors=jsonk("get","daemonsets,deployments,statefulsets,nodesets","-A")
 selector_evidence=[]
 for item in selectors["items"]:
  spec=item["spec"];template=spec.get("template",{}).get("spec",{})
  selector_evidence.append(dict(kind=item["kind"],name=item["metadata"]["name"],
    namespace=item["metadata"].get("namespace"),uid=item["metadata"]["uid"],
    nodeSelector=template.get("nodeSelector",{}),affinity=template.get("affinity",{}),
    selector_key_in_spec=key in json.dumps(spec),
    source_spec_sha256=sha(json.dumps(spec,sort_keys=True).encode())))
 save("selector-inventory.json",selector_evidence)
 target_selector=nodeset["spec"]["template"]["spec"]["nodeSelector"].get(key)==cfg["slurmd_selector_value"]
 isolated=all(key not in json.dumps(item["spec"]) for item in selectors["items"] if item["metadata"]["uid"]!=cfg["nodeset_uid"])

 pods=jsonk("-n","beagle","get","pods")["items"];reports=[]
 for n in NODES:
  found=[p for p in pods if p["spec"].get("nodeName")==n and p["metadata"]["name"].startswith("pireus-spark-host-fence-") and not p["metadata"].get("deletionTimestamp")]
  assert len(found)==1
  raw=kube("-n","beagle","exec",found[0]["metadata"]["name"],"-c","host-fence","--","bash","/fence/host-fence.sh","report")
  save(n+".host-report.txt",raw);reports.append(fields(raw))
 ann=lease["metadata"]["annotations"];epoch=ann[cfg["epoch_annotation"]]
 expiry=datetime.datetime.fromisoformat(lease["spec"]["renewTime"].replace("Z","+00:00")).timestamp()+lease["spec"]["leaseDurationSeconds"]
 checks=[
  all(p["metadata"]["uid"]==cfg["node_"+str(i)+"_uid"] for i,p in enumerate(nodes)),
  nodeset["metadata"]["uid"]==cfg["nodeset_uid"],
  verified.startswith("SPARK_PAIR_VERIFY_PASS") and target_selector and isolated,
  ann[cfg["state_annotation"]]=="RECOVERY_REQUIRED" and frame["state"]=="RECOVERY_REQUIRED",
  lease["spec"]["holderIdentity"]==a.holder,
  expiry-time.time()>60,
  frame["epoch"]==epoch and frame["observed_epoch"]==epoch and all(r["grant_epoch"]==epoch and r["grant_owner"]==a.holder for r in reports),
  all(r["lease_uid"]==lease["metadata"]["uid"] for r in reports),
  all(r["grant_mode"]=="FENCED" and r["grant_valid"]=="0" for r in reports),
  all(r["source_sha256"]==frozen["authority_sha256"] and r["freeze_sha256"]==lock["arbiter_freeze_sha256"] for r in reports),
  all(r["watchdog"]=="1" and 0<=int(r["reported_monotonic"])-int(r["watchdog_monotonic"])<=55 for r in reports),
  all(r["device_barrier"]=="1" and r["device_barrier_source_sha256"]==frozen["device_barrier_source_sha256"] for r in reports),
  all(r["protected"]=="1" for r in reports),
  all(all(r[k]=="1" for k in ("inventory","services","restarts","docker_claims")) for r in reports),
  int(frame["slurm_mask"]),
  all(r["consumers"]=="1" for r in reports),
 ]
 values=[1]+[int(v) for v in checks]+[int(r["mem_available_mb"]) for r in reports]
 input_path=save("facts.v1"," ".join(map(str,values))+"\n")
 decision=subprocess.run([str(a.engine.resolve()),str(input_path.resolve())],capture_output=True,text=True,timeout=60)
 save("decision.json",decision.stdout);save("decision.stderr.txt",decision.stderr)
 receipt=json.loads(decision.stdout)
 assert receipt["authority"]=="Sounio" and receipt["action"]=="DETACH_FENCED_WORKERS"
 assert receipt["gpu_grant"] is False and receipt["resume"] is False and receipt["claim_ready"] is False
 assert receipt["minimum_free_memory_mb"]==32768
 save("custody.json",dict(facts_sha256=sha(input_path.read_bytes()),engine_sha256=lock["executable_sha256"],apply=a.apply,contract=lock))
 if decision.returncode or receipt["decision"]!="ALLOW":raise ValueError("Sounio refused recovery detachment")
 if not a.apply:print("RECOVERY_DETACH_REVIEWABLE_ALLOW");return
 # Re-observe through the same native gate immediately before each node effect.
 applied=[]
 try:
  for index in range(2):
   fresh=a.evidence/("recheck-"+str(index))
   subprocess.run([sys.executable,str(Path(__file__).resolve()),"--engine",str(a.engine.resolve()),
     "--arbiter-root",str(a.arbiter_root.resolve()),"--holder",a.holder,
     "--evidence",str(fresh.resolve())],check=True,timeout=240)
   refreshed=json.loads((fresh/"lease.json").read_text())
   assert refreshed["metadata"]["resourceVersion"]==lease["metadata"]["resourceVersion"],"lease changed during re-observation"
   node=json.loads((fresh/"nodes.json").read_text())[index]
   current=jsonk("-n","beagle","get","lease","pireus-spark-pair")
   assert current["metadata"]["resourceVersion"]==lease["metadata"]["resourceVersion"],"lease changed before effect"
   assert expiry-time.time()>15,"lease too near expiry"
   assert node["metadata"]["labels"].get(key)==cfg["slurmd_selector_value"]
   escaped=key.replace("~","~0").replace("/","~1")
   patch=[dict(op="test",path="/metadata/uid",value=node["metadata"]["uid"]),
     dict(op="test",path="/metadata/resourceVersion",value=node["metadata"]["resourceVersion"]),
     dict(op="test",path="/metadata/labels/"+escaped,value=cfg["slurmd_selector_value"]),
     dict(op="remove",path="/metadata/labels/"+escaped)]
   save(node["metadata"]["name"]+".patch.json",patch)
   save("effect-status.json",dict(status="APPLYING",applied=applied,pending=node["metadata"]["name"],pair_atomic=False))
   result=kube("patch","node",node["metadata"]["name"],"--type=json","-p",json.dumps(patch))
   save(node["metadata"]["name"]+".effect.txt",result)
   applied.append(node["metadata"]["name"])
  post_nodes=[jsonk("get","node",n) for n in NODES]
  post_lease=jsonk("-n","beagle","get","lease","pireus-spark-pair")
  save("post-nodes.json",post_nodes);save("post-lease.json",post_lease)
  assert all(key not in node["metadata"]["labels"] for node in post_nodes)
  assert post_lease["metadata"]["resourceVersion"]==lease["metadata"]["resourceVersion"]
  for n in NODES:
   fresh_pods=jsonk("-n","beagle","get","pods")["items"]
   found=[p for p in fresh_pods if p["spec"].get("nodeName")==n and p["metadata"]["name"].startswith("pireus-spark-host-fence-") and not p["metadata"].get("deletionTimestamp")]
   assert len(found)==1
   raw=kube("-n","beagle","exec",found[0]["metadata"]["name"],"-c","host-fence","--","bash","/fence/host-fence.sh","report")
   save(n+".post-host.txt",raw);r=fields(raw)
   assert r["grant_mode"]=="FENCED" and r["grant_valid"]=="0"
   assert all(r[k]=="1" for k in ("watchdog","device_barrier","protected"))
  save("effect-status.json",dict(status="DETACHED_FENCED",applied=applied,pair_atomic=False))
 except BaseException:
  save("effect-status.json",dict(status="PARTIAL_OR_UNVERIFIED" if applied else "NOT_APPLIED",applied=applied,pair_atomic=False))
  raise

 print("RECOVERY_DETACH_APPLIED; GPU remains fenced; resume only through canonical arbiter")
if __name__=="__main__":main()
