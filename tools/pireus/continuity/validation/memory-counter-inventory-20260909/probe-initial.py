#!/usr/bin/env python3
"""Read-only counter inventory in the current worker namespace; no model import."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

def read(path):
    times=[]
    try:
        for _ in range(10):
            started=time.monotonic_ns()
            raw=Path(path).read_text()
            times.append(time.monotonic_ns()-started)
        return dict(available=True,raw=raw,unit="raw kernel text; units retained",
            reads=10,read_ns_median=statistics.median(times),read_ns_max=max(times))
    except (OSError,ValueError) as exc:
        return dict(available=False,raw=None,error=type(exc).__name__,detail=str(exc))

def command(args):
    started=time.monotonic_ns()
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=15)
        return dict(argv=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
            duration_ns=time.monotonic_ns()-started)
    except (OSError,subprocess.TimeoutExpired) as exc:
        return dict(argv=args,returncode=None,error=type(exc).__name__,detail=str(exc),
            duration_ns=time.monotonic_ns()-started)

def inventory():
    paths=["/proc/sys/kernel/random/boot_id","/proc/meminfo","/proc/vmstat",
        "/proc/pressure/memory","/proc/self/cgroup","/proc/self/smaps_rollup",
        "/proc/self/status","/proc/self/mountinfo",
        "/sys/fs/cgroup/cgroup.controllers","/sys/fs/cgroup/cgroup.type",
        "/sys/fs/cgroup/memory.current","/sys/fs/cgroup/memory.peak",
        "/sys/fs/cgroup/memory.stat","/sys/fs/cgroup/memory.events",
        "/sys/fs/cgroup/memory.pressure","/sys/fs/cgroup/memory.max",
        "/proc/driver/nvidia/version"]
    result=dict(schema="pireus-memory-counter-inventory-v1",
        wall_time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        monotonic_ns=time.monotonic_ns(),pid=os.getpid(),uid=os.getuid(),
        scope="worker namespace, idle read cost; no live model-process sampling",
        gpu_workload_launched=False,loaded_inference_overhead_qualified=False,
        files={p:read(p) for p in paths})
    result["commands"]=[command(["nvidia-smi","--query-gpu=index,name,uuid,memory.total,memory.used,memory.free","--format=csv,noheader"]),
        command(["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader"])]
    return result

if __name__=="__main__":
    print(json.dumps(inventory(),indent=2))
