#!/usr/bin/env python3
"""Check YAML clip semantics against exact content address and bridge bytes."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def parsed(path):
 raw=subprocess.check_output(["kubectl","create","--dry-run=client","-f",str(path),"-o","json"],text=True)
 decoder=json.JSONDecoder(); docs=[]
 while raw.strip():
  raw=raw.lstrip();obj,end=decoder.raw_decode(raw);docs.append(obj);raw=raw[end:]
 return next(d for d in docs if d["kind"]=="ConfigMap")
def validate(path,policy,admission):
 cm=parsed(path);data=cm["data"]["host-fence.sh"].encode()
 raw=path.read_text().split("  host-fence.sh: |\n",1)[1].split("---\n",1)[0]
 extracted="".join(line[4:] if line.startswith("    ") else line for line in raw.splitlines(keepends=True)).encode()
 transported=extracted.rstrip(b"\n")+b"\n"
 digest=hashlib.sha256(data).hexdigest();name="pireus-spark-host-fence-"+digest[:12]
 assert extracted==data==transported,"YAML/extraction/transport byte mismatch"
 assert cm["metadata"]["name"]==name,"ConfigMap content address mismatch"
 assert "host_fence_configmap="+name+"\n" in policy
 assert admission.count(name)==3
 return digest
manifest=ROOT/"tools/cluster/spark_pair_host_fence.yaml"
policy=(ROOT/"tools/cluster/spark_pair_arbiter.policy.v1").read_text()
admission=(ROOT/"tools/cluster/spark_pair_arbiter_admission.yaml").read_text()
digest=validate(manifest,policy,admission)
with tempfile.TemporaryDirectory() as tmp:
 bad=Path(tmp)/"host.yaml"
 # Reproduce the exact raw-extraction/YAML clipping regression.
 bad.write_text(manifest.read_text().replace("esac\n---","esac\n\n---"))
 try:validate(bad,policy,admission)
 except AssertionError:pass
 else:raise AssertionError("trailing blank regression accepted")
 bad.write_text(manifest.read_text().replace("pireus-spark-host-fence-"+digest[:12],"pireus-spark-host-fence-000000000000"))
 try:validate(bad,policy,admission)
 except AssertionError:pass
 else:raise AssertionError("wrong content address accepted")
print("PIREUS_HOST_CONTENT_ADDRESS_PASS yaml_extraction_transport_equal=1 extra_blank_refused=1 wrong_address_refused=1 sha256="+digest)
