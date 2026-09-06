#!/usr/bin/env python3
"""Idempotent setup for the isolated Spark runtime; run as worker root."""
import json, resource
from pathlib import Path
def main():
 helper=Path("/scratch/pireus/runtime/apptainer-1.5.3/usr/libexec/apptainer/bin/squashfuse_ll")
 original=helper.with_name("squashfuse_ll.elf")
 if not original.exists():
  assert helper.read_bytes()[:4]==b"\x7fELF"
  helper.rename(original)
 wrapper='#!/bin/sh\nexec /lib/ld-linux-aarch64.so.1 --library-path /scratch/pireus/runtime/lib '+str(original)+' "$@"\n'
 if helper.exists():assert helper.read_text()==wrapper,"unrelated helper contents"
 else:helper.write_text(wrapper);helper.chmod(0o755)
 # Runtime-only change. Reapply after recreation of the worker pod.
 pids=[]
 for p in Path("/proc").iterdir():
  if p.name.isdigit():
   try:
    if (p/"comm").read_text().strip()=="slurmd":pids.append(int(p.name))
   except FileNotFoundError:pass
 assert len(pids)==1,pids
 pid=pids[0];before=resource.prlimit(pid,resource.RLIMIT_MEMLOCK)
 resource.prlimit(pid,resource.RLIMIT_MEMLOCK,(-1,-1))
 print(json.dumps(dict(slurmd_pid=pid,memlock_before=before,memlock_after=resource.prlimit(pid,resource.RLIMIT_MEMLOCK))))
if __name__=="__main__":main()
