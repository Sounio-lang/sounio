#!/usr/bin/env python3
"""Small workspace Linux control for observer RSS/CPU reader, not Spark qualification."""
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
OPS=Path(__file__).resolve().parents[2]/"ops"
sys.path.insert(0,str(OPS))
from external_memory_observer import observer_resources, OBSERVER_PROFILE

before=observer_resources()
allocation=bytearray(8*1024**2)
for offset in range(0,len(allocation),4096):allocation[offset]=1
after=observer_resources()
for r in (before,after):
    if r["status"]["error"] is not None:raise ValueError("observer status unavailable")
    if r["status"]["value"]["VmHWM"]<r["status"]["value"]["VmRSS"]:
        raise ValueError("invalid observer RSS/HWM")
delta=after["status"]["value"]["VmRSS"]-before["status"]["value"]["VmRSS"]
if not 4*1024**2<=delta<=16*1024**2:raise ValueError("8 MiB reader control outside 4..16 MiB bounds")
if after["process_cpu_ns"]<=before["process_cpu_ns"]:raise ValueError("CPU counter did not advance")
print(json.dumps(dict(schema="pireus-observer-resource-reader-control-v1",
    observer_profile=OBSERVER_PROFILE,observer_sha256=hashlib.sha256((OPS/"external_memory_observer.py").read_bytes()).hexdigest(),
    platform=sys.platform,allocation_bytes=len(allocation),rss_delta_bytes=delta,
    before=before,after=after,reader_control_pass=True,
    target_identity_integration_tested=False,spark_control_qualified=False,
    loaded_model_overhead_qualified=False),indent=2))
