#!/usr/bin/env python3
"""Advisory eviction of this allocation's read-only checkpoint file cache only."""
import json
import os
from pathlib import Path
assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_PROCID")
root = Path("/scratch/pireus/models/Inkling-Small-NVFP4/b6a99534467840620d411e4cd4ad5819b2610d9c")
files = sorted(root.glob("model-*-of-*.safetensors"))
if len(files) != 9:
    raise ValueError("unexpected checkpoint shard set")
before = Path("/proc/meminfo").read_text()
for p in files:
    with p.open("rb") as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
def available(text):
    return next(int(l.split()[1])*1024 for l in text.splitlines() if l.startswith("MemAvailable:"))
print(json.dumps(dict(stage="OWN_CHECKPOINT_FADVISE", job=os.environ["SLURM_JOB_ID"],
    rank=os.environ["PIREUS_RANK"], files=len(files), content_modified=False,
    advisory_only=True, before_available=available(before),
    after_available=available(Path("/proc/meminfo").read_text()))), flush=True)
