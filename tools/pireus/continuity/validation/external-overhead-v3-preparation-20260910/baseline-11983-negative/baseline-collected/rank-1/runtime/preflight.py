"""Read-only host grant preflight; the canonical Sounio arbiter owns recovery."""
import json,subprocess
PREFIX="pireus.sounio.dev/"
def validate_report(lease, node, fields):
 ann=lease["metadata"]["annotations"]
 if lease["spec"]["holderIdentity"]!="slurm-owned" or ann[PREFIX+"spark-pair-state"]!="SLURM_OWNED":
  raise ValueError("Spark lease is not SLURM_OWNED")
 expected={"node":node,"grant_mode":"SLURM",
  "grant_owner":ann[PREFIX+"host-fence-owner"],
  "grant_epoch":ann[PREFIX+"spark-pair-epoch"],
  "lease_uid":lease["metadata"]["uid"],
  "lease_resource_version":ann[PREFIX+"host-lease-resource-version"],
  "transaction_id":ann[PREFIX+"host-transaction"],
  "decision_receipt_sha256":ann[PREFIX+"host-decision-receipt"],
  "pair_digest":ann[PREFIX+"host-pair-digest"],
  "boot_id":ann[PREFIX+"host-boot-"+node.removeprefix("spark-")],
  "source_sha256":ann[PREFIX+"sounio-source-sha256"],
  "freeze_sha256":ann[PREFIX+"semantics-freeze-sha256"]}
 if ann[PREFIX+"host-fence-epoch"]!=expected["grant_epoch"]:
  raise ValueError("host epoch annotation mismatch")
 for key,value in expected.items():
  if not value or fields.get(key)!=value:raise ValueError("host binding mismatch: "+key)
 for key in ("grant_valid","watchdog","device_barrier","inventory","services","restarts","docker_claims","consumers","cgroups","memory","protected"):
  if fields.get(key)!="1":raise ValueError("host preflight refused "+key)
 if not 0<=int(fields["reported_monotonic"])-int(fields["watchdog_monotonic"])<=55 or int(fields["mem_available_mb"])<32768:
  raise ValueError("stale watchdog or insufficient host memory")

def check_pair():
 def kube(*a):return subprocess.check_output(["kubectl",*a],text=True,timeout=60)
 lease=json.loads(kube("-n","beagle","get","lease","pireus-spark-pair","-o","json"))
 pods=json.loads(kube("-n","beagle","get","pods","-o","json"))["items"]
 reports=[]
 for node in ("spark-3c59","spark-8e54"):
  matches=[p for p in pods if p["spec"].get("nodeName")==node and p["metadata"]["name"].startswith("pireus-spark-host-fence-") and not p["metadata"].get("deletionTimestamp")]
  if len(matches)!=1:raise ValueError("expected one host fence observer on "+node)
  raw=kube("-n","beagle","exec",matches[0]["metadata"]["name"],"-c","host-fence","--","/bin/bash","/fence/host-fence.sh","report")
  fields=dict(x.split("=",1) for x in raw.split() if "=" in x)
  validate_report(lease,node,fields)
  reports.append(raw.strip())
 fresh=json.loads(kube("-n","beagle","get","lease","pireus-spark-pair","-o","json"))
 if fresh["metadata"]["uid"]!=lease["metadata"]["uid"] or fresh["metadata"]["resourceVersion"]!=lease["metadata"]["resourceVersion"]:
  raise ValueError("Lease changed during host preflight")
 return reports
