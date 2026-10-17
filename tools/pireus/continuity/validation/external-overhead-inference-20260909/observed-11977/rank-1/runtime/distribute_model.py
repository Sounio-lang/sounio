#!/usr/bin/env python3
"""Resume immutable snapshot transport to Spark scratch; qualify hashes separately."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

PODS = ["slurm-pilot-worker-spark-5kw2b", "slurm-pilot-worker-spark-6vkch"]
REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"
TARGET = "/scratch/pireus/models/Inkling-Small-NVFP4/" + REVISION
REMOTE_STATE = """
from pathlib import Path
import json,sys
p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
part=p.with_name(p.name+'.partial')
print(json.dumps({'complete':p.stat().st_size if p.is_file() else -1,
                  'partial':part.stat().st_size if part.is_file() else 0}))
"""
REMOTE_APPEND = """
from pathlib import Path
import sys,os
p=Path(sys.argv[1]);offset=int(sys.argv[2]);size=int(sys.argv[3])
part=p.with_name(p.name+'.partial')
if p.exists():raise SystemExit('completed destination already exists')
if (part.stat().st_size if part.exists() else 0)!=offset:raise SystemExit('resume offset drift')
with part.open('ab' if offset else 'wb') as out:
 while True:
  b=sys.stdin.buffer.read(8*1024*1024)
  if not b:break
  out.write(b);offset+=len(b)
  if offset>size:raise SystemExit('source exceeds frozen size')
 out.flush();os.fsync(out.fileno())
if offset!=size:raise SystemExit('partial transfer retained')
os.replace(part,p)
print('COPIED_BYTES',offset)
"""

def command(pod, code, *args, stdin=False):
    return ["kubectl", "-n", "slurm-pilot", "exec", *(["-i"] if stdin else []),
            pod, "-c", "slurmd", "--", "python3", "-c", code, *map(str, args)]

def distribute(pod, source, entries):
    pending = {x["rfilename"]: x for x in entries}
    while pending:
        progressed = False
        for name, entry in list(pending.items()):
            path = source / name
            if not path.is_file():
                continue
            if path.stat().st_size != entry["size"]:
                raise ValueError("completed source size drift")
            state = json.loads(subprocess.check_output(command(pod, REMOTE_STATE, TARGET + "/" + name), text=True))
            if state["complete"] >= 0:
                if state["complete"] != entry["size"]:
                    raise ValueError("existing completed destination size mismatch")
                print(f"ALREADY_COPIED {pod} {name}; cryptographic qualification pending", flush=True)
            else:
                offset = state["partial"]
                if offset > entry["size"]:
                    raise ValueError("partial destination exceeds frozen size")
                print(f"COPY {pod} {name} offset={offset}", flush=True)
                proc = subprocess.Popen(command(pod, REMOTE_APPEND, TARGET + "/" + name,
                                               offset, entry["size"], stdin=True), stdin=subprocess.PIPE)
                try:
                    with path.open("rb") as inp:
                        inp.seek(offset)
                        for block in iter(lambda: inp.read(8 * 1024**2), b""):
                            proc.stdin.write(block)
                    proc.stdin.close()
                    if proc.wait() != 0:
                        raise RuntimeError("transport failed; partial is resumable")
                except BaseException:
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.close()
                    proc.wait()
                    raise
            del pending[name]
            progressed = True
        if not progressed:
            log = Path("/tmp/pireus-inkling-model-stage.log")
            if log.exists() and "exit_code=" in log.read_text() and not (source / "snapshot-staging.json").exists():
                raise RuntimeError("source staging stopped without complete receipt")
            time.sleep(10)
    print(f"TRANSPORT_COMPLETE {pod}; cryptographic qualification pending", flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    args = p.parse_args()
    entries = json.loads(Path(__file__).with_name("inkling-files.json").read_text())
    for entry in entries:
        name = Path(entry["rfilename"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("unsafe manifest path")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(distribute, pod, args.source, entries) for pod in PODS]
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
