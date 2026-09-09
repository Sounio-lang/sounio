#!/usr/bin/env python3
"""Replay numerical checks and verify immutable custody, without hardware."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PIN="bb52639973bede3905c7f991b79e57954adbd0ade53b2c4b359de78084d5841c"
def verify(root):
    raw=(root/"qualification.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PIN, "qualification changed"
    q=json.loads(raw)
    for name,sha in q["files_sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    m=json.loads((root/"manifest.json").read_bytes())
    assert hashlib.sha256((root/"protocol.json").read_bytes()).hexdigest()==m["protocol_sha256"]
    protocol=json.loads((root/"protocol.json").read_bytes())
    assert protocol["schema"]=="pireus-cpu-memory-control-v2"
    for name,sha in m["files_sha256"].items():
        assert hashlib.sha256((root/"source"/name).read_bytes()).hexdigest()==sha
        assert protocol["runtime_files_sha256"][name]==sha
    spec=importlib.util.spec_from_file_location("frozen_oracle",root/"source/cpu_memory_control_v2.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    assert all(protocol[k]==v for k,v in mod.PROTOCOL.items())
    assert q["job"]==11975 and q["cpu_v2_control_pass"] and q["container_handoff_pass"]
    assert not q["inference_executed"] and not q["loaded_model_overhead_qualified"]
    for r in q["results"]:
        p=root/("rank-"+str(r["rank"]))
        rows=[json.loads(s) for s in (p/"journal.jsonl").read_text().splitlines()]
        phases=[json.loads(s) for s in (p/"cpu-phases.jsonl").read_text().splitlines()]
        assert mod.analyze(rows,phases)==r["v2_memory_oracle"]
    return {"integrity":"PASS","frozen_oracle_reproduction":"PASS","job":11975,"inference_qualified":False}
if __name__=="__main__":
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/"attempt"),indent=2))
