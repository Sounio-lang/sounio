"""Read-only host grant preflight; the canonical Sounio arbiter owns recovery."""
import json,subprocess
def check_pair():
 def kube(*a):return subprocess.check_output(["kubectl",*a],text=True)
 if kube("-n","beagle","get","lease","pireus-spark-pair","-o","jsonpath={.spec.holderIdentity}")!="slurm-owned":
  raise ValueError("Spark lease is not slurm-owned")
 pods=json.loads(kube("-n","beagle","get","pods","-o","json"))["items"]
 reports=[]
 for node in ("spark-3c59","spark-8e54"):
  matches=[p for p in pods if p["spec"].get("nodeName")==node and p["metadata"]["name"].startswith("pireus-spark-host-fence-") and not p["metadata"].get("deletionTimestamp")]
  if len(matches)!=1:raise ValueError("expected one host fence observer on "+node)
  raw=kube("-n","beagle","exec",matches[0]["metadata"]["name"],"-c","host-fence","--","/bin/bash","/fence/host-fence.sh","report")
  fields=dict(x.split("=",1) for x in raw.split() if "=" in x)
  if fields.get("grant_owner")!="slurm-owned":raise ValueError("host owner mismatch: "+raw)
  for key in ("grant_valid","watchdog","device_barrier","inventory","services","restarts","docker_claims","consumers","cgroups","memory","protected"):
   if fields.get(key)!="1":raise ValueError("host preflight refused "+key+": "+raw)
  reports.append(raw.strip())
 return reports
