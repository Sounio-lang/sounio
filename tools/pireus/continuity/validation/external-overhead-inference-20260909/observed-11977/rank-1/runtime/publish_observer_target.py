#!/usr/bin/env python3
"""Publish a complete target handoff, await observer acknowledgement, exec entry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

def digest(raw):return hashlib.sha256(raw).hexdigest()

def exclusive_json(path, value):
    """Publish complete bytes atomically; never overwrite an existing attempt."""
    path=Path(path);temporary=path.with_name(path.name+".writing")
    raw=(json.dumps(value,sort_keys=True)+"\n").encode()
    with temporary.open("xb") as out:
        out.write(raw);out.flush();os.fsync(out.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink()
    return digest(raw)

def publish(entry, expected_sha, handoff, nonce):
    if not re.fullmatch("[0-9a-f]{64}",nonce):
        raise ValueError("invalid attempt nonce")
    raw=Path(entry).read_bytes()
    if digest(raw)!=expected_sha:raise ValueError("entry source identity mismatch")
    job=os.environ.get("SLURM_JOB_ID","");rank=os.environ.get("PIREUS_RANK","")
    if not job.isdigit() or rank not in ("0","1"):raise ValueError("Slurm job/rank required")
    pid=os.getpid()
    ticks=int(Path("/proc/self/stat").read_text().rsplit(") ",1)[1].split()[19])
    value=dict(schema="pireus-observer-target-handoff-v1",job=job,rank=rank,pid=pid,
        starttime_ticks=ticks,boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        nonce=nonce,entry_sha256=expected_sha,entry=str(Path(entry).resolve()),
        monotonic_ns=time.monotonic_ns())
    return value,exclusive_json(handoff,value)

def await_ack(path, target, handoff_sha, seconds=30):
    if not 0<seconds<=60:raise ValueError("ack deadline outside bounds")
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        try:ack=json.loads(Path(path).read_bytes())
        except FileNotFoundError:
            time.sleep(0.05);continue
        if (ack.get("schema")!="pireus-observer-attachment-ack-v1"
            or any(ack.get(k)!=target[k] for k in ("nonce","job","rank","pid","starttime_ticks"))
            or ack.get("handoff_sha256")!=handoff_sha
            or ack.get("first_sample_valid") is not True
            or type(ack.get("observer_pid")) is not int):
            raise ValueError("observer acknowledgement identity mismatch")
        return ack
    raise TimeoutError("observer acknowledgement absent; entry was not executed")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entry",type=Path,required=True);p.add_argument("--entry-sha",required=True)
    p.add_argument("args",nargs=argparse.REMAINDER)
    a=p.parse_args()
    target,sha=publish(a.entry,a.entry_sha,os.environ["PIREUS_EXTERNAL_HANDOFF"],os.environ["PIREUS_EXTERNAL_NONCE"])
    await_ack(os.environ["PIREUS_EXTERNAL_ACK"],target,sha)
    if digest(a.entry.read_bytes())!=a.entry_sha:raise ValueError("entry changed before exec")
    args=a.args[1:] if a.args[:1]==["--"] else a.args
    os.execv(sys.executable,[sys.executable,str(a.entry),*args])

if __name__=="__main__":main()
