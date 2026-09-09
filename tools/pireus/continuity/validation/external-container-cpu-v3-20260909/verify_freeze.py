#!/usr/bin/env python3
"""Verify the frozen v3 CPU packet. This does not submit or qualify a job."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
PIN="33ae4e7ee290e0a7b11daa4649575053b2d1c56e9cb18235d7fd96fa74ed419d"
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def verify(root):
    require(digest(root/"manifest.json")==PIN,"freeze manifest changed")
    manifest=json.loads((root/"manifest.json").read_bytes())
    require(digest(root/"protocol.json")==manifest["protocol_sha256"],"protocol changed")
    protocol=json.loads((root/"protocol.json").read_bytes())
    require(protocol["source_commit"]==manifest["source_commit"],"source identity")
    require(protocol["runtime_files_sha256"]==manifest["files_sha256"],"source inventory")
    actual={str(p.relative_to(root/"source")) for p in (root/"source").rglob("*") if p.is_file()}
    require(actual==set(manifest["files_sha256"]),"unexpected or missing frozen source")
    for name,pin in manifest["files_sha256"].items():
        p=root/"source"/name
        require(p.resolve().is_relative_to((root/"source").resolve()) and digest(p)==pin,"source changed: "+name)
    # The two frozen oracle modules are imported only after every file is pinned.
    for name in ("cpu_memory_control_v2","cpu_memory_control_v3"):
        spec=importlib.util.spec_from_file_location(name,root/"source"/(name+".py"))
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    require(all(protocol[k]==v for k,v in mod.PROTOCOL.items()),"protocol/oracle mismatch")
    return dict(freeze_integrity_pass=True,source_commit=manifest["source_commit"],
        protocol_sha256=manifest["protocol_sha256"],source_files=len(actual),
        source_ci_qualified=False,container_control_qualified=False,
        loaded_model_overhead_qualified=False,inference_executed=False)
if __name__=="__main__":
    print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/"freeze"),indent=2))
