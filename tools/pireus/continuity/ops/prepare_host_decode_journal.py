#!/usr/bin/env python3
"""Build a separately identified journal transport; never deploy or execute it."""
import ast
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = "8c67c707750a051b2845d11ffc03e038ff488906f8a5fe26557c4150aa21ba2e"
ANCHOR = '    print(json.dumps(row, sort_keys=True), flush=True)\n'
REPLACEMENT = '''    global _host_decode_journal
    if _host_decode_journal is None:
        _host_decode_journal = HostDecodeJournal("/scratch/pireus/receipts",
            os.environ["SLURM_JOB_ID"], os.environ["PIREUS_RANK"], os.getpid())
    print(json.dumps(_host_decode_journal.write(row), sort_keys=True), flush=True)
'''
INSERT = "def host_decode_probe(stage, index, step):"

def prepare(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("source identity")
    text = raw.decode()
    if text.count(ANCHOR) != 1 or text.count(INSERT) != 1:
        raise ValueError("nonunique transport anchor")
    helper = Path(__file__).with_name("host_decode_journal.py").read_text()
    # Embed only writer and dependencies; offline reader remains in tooling.
    helper = helper.split("\ndef read_journal(", 1)[0]
    block = helper + "\n_host_decode_journal = None\n\n"
    result = text.replace(ANCHOR, REPLACEMENT).replace(INSERT, block + INSERT)
    ast.parse(result)
    if result.replace(block + INSERT, INSERT).replace(REPLACEMENT, ANCHOR) != text:
        raise ValueError("unexpected runtime mutation")
    return result.encode()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = prepare(args.source.read_bytes())
    with args.output.open("xb") as stream:
        stream.write(data)
    print(json.dumps(dict(source_sha256=SOURCE_SHA256,
        runtime_sha256=hashlib.sha256(data).hexdigest(), hardware_executed=False,
        timing_eligible=False, collector_integrated=False)))
