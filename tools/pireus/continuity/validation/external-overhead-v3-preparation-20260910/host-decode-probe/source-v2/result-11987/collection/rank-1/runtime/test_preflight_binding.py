#!/usr/bin/env python3
import copy,json
from pathlib import Path
from preflight import validate_report,PREFIX
v=Path(__file__).resolve().parent.parent/"validation/slurm-recovered-preflight"
lease=json.loads((v/"lease.json").read_text())
negative=0
for node in ("spark-3c59","spark-8e54"):
 fields=dict(x.split("=",1) for x in (v/(node+".txt")).read_text().split() if "=" in x)
 validate_report(lease,node,fields)
 for key,value in [("grant_owner","slurm-owned"),("grant_mode","K8S"),("grant_epoch","14"),
  ("lease_uid","wrong"),("transaction_id","0"*64),("pair_digest","none"),
  ("boot_id","wrong"),("freeze_sha256","0"*64),("grant_valid","0"),
  ("watchdog","0"),("device_barrier","0"),("mem_available_mb","32767"),
  ("watchdog_monotonic",str(int(fields["reported_monotonic"])-56))]:
  bad=dict(fields);bad[key]=value
  try:validate_report(lease,node,bad)
  except ValueError:negative+=1
  else:raise AssertionError(key+" mismatch accepted")
 for key,value in [("spark-pair-state","RECOVERY_REQUIRED"),("host-fence-epoch","14")]:
  bad=copy.deepcopy(lease);bad["metadata"]["annotations"][PREFIX+key]=value
  try:validate_report(bad,node,fields)
  except ValueError:negative+=1
  else:raise AssertionError(key+" mismatch accepted")
print("PIREUS_PREFLIGHT_BINDING_PASS positive=2 negative="+str(negative))
