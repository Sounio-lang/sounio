#!/usr/bin/env python3
"""Build a separate source artifact; never edits the qualified runtime."""
import argparse
import hashlib
import json
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
PIN = "fa4fb62f1f8a5d57a40374ef4ad839718caa171b902fb115cc85247191ca8cc0"

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def generate(raw, observer):
    if digest(raw) != PIN:
        raise ValueError("base runtime identity mismatch")
    source = raw.decode()
    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("instrumentation anchor mismatch")
        source = source.replace(old, new, 1)
    setup = (
        "_lifecycle_namespace = {}\n"
        "exec(compile(" + repr(observer) + ", '<lifecycle-observer>', 'exec'), _lifecycle_namespace)\n"
        "lifecycle = _lifecycle_namespace['Observer'](\n"
        "    Path('/scratch/pireus/receipts') / ('lifecycle-' + os.environ['SLURM_JOB_ID'] + '-' + os.environ['PIREUS_RANK'] + '.jsonl'),\n"
        "    os.environ['SLURM_JOB_ID'], os.environ['PIREUS_RANK'], torch.cuda)\n"
        "lifecycle.start()\n"
        "import atexit\n"
        "atexit.register(lifecycle.close)\n\n")
    replace_once("def emit(stage, **fields):", setup+"def emit(stage, **fields):")
    replace_once('        inference_memory("OFFLINE_EXTEND_BEGIN", index=item["index"])',
                 '        lifecycle.mark("EXTEND_ENTRY", item["index"])\n'
                 '        inference_memory("OFFLINE_EXTEND_BEGIN", index=item["index"])')
    replace_once('        for step in range(item["max_new_tokens"]):',
                 '        lifecycle.mark("DECODE_ENTRY", item["index"], 0)\n'
                 '        for step in range(item["max_new_tokens"]):\n'
                 '            if step % 16 == 0:\n'
                 '                lifecycle.mark("DECODE_SAMPLE", item["index"], step)')
    replace_once('        response = dict(schema=1, transport="sglang-offline-token-ids", index=item["index"],',
                 '        lifecycle.mark("DECODE_EXIT", item["index"], len(output))\n'
                 '        response = dict(schema=1, transport="sglang-offline-token-ids", index=item["index"],')
    replace_once('        runner.cleanup(batch)\n        del batch, req, logits, next_ids',
                 '        lifecycle.mark("PROPOSAL_SAVED", item["index"], len(output))\n'
                 '        lifecycle.mark("CLEANUP_BEFORE", item["index"], len(output))\n'
                 '        runner.cleanup(batch)\n'
                 '        lifecycle.mark("CLEANUP_AFTER", item["index"], len(output))\n'
                 '        del batch, req, logits, next_ids\n'
                 '        lifecycle.mark("REFERENCES_RELEASED", item["index"], len(output))')
    compile(source, "<diagnostic>", "exec")
    return source

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    raw=(BASE/"runtime/offline_generate.py").read_bytes()
    observer=Path(__file__).with_name("lifecycle_observer.py").read_text()
    generated=generate(raw,observer)
    args.output.mkdir(exist_ok=False)
    (args.output/"offline_generate.py").write_text(generated)
    manifest=dict(schema="pireus-lifecycle-diagnostic-build-v1",
        base_runtime_sha256=PIN, observer_sha256=digest(observer.encode()),
        generated_runtime_sha256=digest(generated.encode()),
        builder_sha256=digest(Path(__file__).read_bytes()),
        identity="feedback-lifecycle-diagnostic-v1",
        runtime_replacement_required=True, host_interval_seconds=1,
        decode_observation_stride=16, guard_changed=False, hardware_executed=False,
        execution_profile_qualified=False, request_bytes_changed=False,
        missing_cgroup_device_accounting=True)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
