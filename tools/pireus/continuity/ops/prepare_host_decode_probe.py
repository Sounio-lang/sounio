#!/usr/bin/env python3
"""Prepare a separate first-15-decode host probe; never run or overwrite inference."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
SOURCE_SHA256="8794694d22b4319a8e8df719eabdce9a89115fa2d1d11e630780bf3c9d867d5d"
HELPER="\ndef host_decode_probe(stage, index, step):\n    started = time.monotonic_ns()\n    records = {}\n    for name, filename in ((\"host_meminfo\", \"/proc/meminfo\"),\n                           (\"host_vmstat\", \"/proc/vmstat\"),\n                           (\"process_status\", \"/proc/self/status\")):\n        begin = time.monotonic_ns()\n        try:\n            raw = Path(filename).read_text()\n            # Keep native units in the raw text; do not infer allocation totals.\n            records[name] = dict(raw=raw, error=None)\n        except (OSError, ValueError) as exc:\n            records[name] = dict(raw=None, error=type(exc).__name__)\n        records[name].update(monotonic_ns=begin, duration_ns=time.monotonic_ns()-begin)\n    counters = {}\n    for name in (\"memory_allocated\", \"memory_reserved\"):\n        try:\n            counters[name] = dict(value=getattr(torch.cuda, name)(), error=None)\n        except Exception as exc:\n            counters[name] = dict(value=None, error=type(exc).__name__)\n    row = dict(schema=\"pireus-host-decode-probe-v1\", stage=stage, job=os.environ[\"SLURM_JOB_ID\"],\n               rank=os.environ[\"PIREUS_RANK\"], pid=os.getpid(), index=index, step=step,\n               monotonic_ns=started, read_duration_ns=time.monotonic_ns()-started,\n               files=records, cuda_allocator=counters, diagnostic_only=True,\n               device_synchronized=False)\n    # read_duration_ns excludes serialization/output; full call cost is not claimed.\n    print(json.dumps(row, sort_keys=True), flush=True)\n"
ANCHOR="                with comm.change_state(enable=True):\n                    next_ids, logits = runner.decode(next_ids, batch)\n"
REPLACEMENT="                if item[\"index\"] == 0 and step < 15:\n                    host_decode_probe(\"HOST_DECODE_BEGIN\", item[\"index\"], step + 1)\n                with comm.change_state(enable=True):\n                    next_ids, logits = runner.decode(next_ids, batch)\n                if item[\"index\"] == 0 and step < 15:\n                    host_decode_probe(\"HOST_DECODE_END\", item[\"index\"], step + 1)\n"
INSERT="def release_loading_temporaries(model, label):"
def prepare(raw):
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256:raise ValueError("source identity mismatch")
    text=raw.decode()
    if text.count(ANCHOR)!=1 or text.count(INSERT)!=1:raise ValueError("nonunique probe anchor")
    result=text.replace(ANCHOR,REPLACEMENT).replace(INSERT,HELPER+"\n"+INSERT)
    ast.parse(result)
    if result.replace(REPLACEMENT,ANCHOR).replace(HELPER+"\n","")!=text:
        raise ValueError("probe changes more than declared blocks")
    return result.encode()
def write(source,output):
    result=prepare(source.read_bytes())
    with output.open("xb") as out:out.write(result)
    return dict(source_sha256=SOURCE_SHA256,output_sha256=hashlib.sha256(result).hexdigest(),
                scope="request 0 decode calls 1 through 15 only",maximum_probe_records_per_rank=30,
                no_new_device_synchronization=True,hardware_executed=False,
                runtime_qualified=False,timing_eligible=False)
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(write(args.source,args.output),indent=2))
