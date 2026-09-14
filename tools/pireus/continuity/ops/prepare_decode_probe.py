#!/usr/bin/env python3
"""Produce a separate, pinned diagnostic source; never overwrite the pilot runtime."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = "fa4fb62f1f8a5d57a40374ef4ad839718caa171b902fb115cc85247191ca8cc0"
ANCHOR = """                with comm.change_state(enable=True):
                    next_ids, logits = runner.decode(next_ids, batch)
"""
REPLACEMENT = """                if item["index"] == 0:
                    inference_memory("OFFLINE_DECODE_PROBE_BEGIN", index=item["index"],
                                     step=step + 1, monotonic_ns=time.monotonic_ns(),
                                     diagnostic_only=True)
                with comm.change_state(enable=True):
                    next_ids, logits = runner.decode(next_ids, batch)
                if item["index"] == 0:
                    inference_memory("OFFLINE_DECODE_PROBE_END", index=item["index"],
                                     step=step + 1, monotonic_ns=time.monotonic_ns(),
                                     diagnostic_only=True)
"""


def prepare(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("unqualified runtime source")
    text = raw.decode()
    if text.count(ANCHOR) != 1:
        raise ValueError("decode anchor is not unique")
    result = text.replace(ANCHOR, REPLACEMENT)
    ast.parse(result)
    if result.replace(REPLACEMENT, ANCHOR) != text:
        raise ValueError("diagnostic modifies more than its declared probes")
    return result.encode()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = prepare(a.source.read_bytes())
    # Exclusive creation also refuses same-source paths, symlinks and existing artifacts.
    with a.output.open("xb") as f:
        f.write(out)
    print(json.dumps({"schema": "pireus-decode-probe-build-v1",
                      "source_sha256": SOURCE_SHA256,
                      "output_sha256": hashlib.sha256(out).hexdigest(),
                      "output_path": str(a.output),
                      "diagnostic_only": True,
                      "hardware_executed": False,
                      "observer_effect": "Additional procfs reads and log writes may alter timing.",
                      "cuda_observation": "No new device synchronization; counters observed at host call boundaries."},
                     indent=2))
