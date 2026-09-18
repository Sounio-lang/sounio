"""Replay a comparison from two downloaded, digest-pinned GitHub artifacts."""
import hashlib, json, pathlib, sys, zipfile
EXPECTED = [
 "8f528464cbc8ae6c76a7da1dcb85c66473f0d5b72f1a38137182ffa1b239a709",
 "75f540968a94c7515ea398f7a1d48c4484035238af115ee59ab5db272b8e7bac",
]
def sha(data): return hashlib.sha256(data).hexdigest()
if len(sys.argv) != 3:
    raise SystemExit("usage: compare.py SUCCESS_ZIP TIMEOUT_ZIP")
paths = [pathlib.Path(p) for p in sys.argv[1:]]
for path, digest in zip(paths, EXPECTED):
    if sha(path.read_bytes()) != digest:
        raise SystemExit("artifact digest mismatch")
with zipfile.ZipFile(paths[0]) as success, zipfile.ZipFile(paths[1]) as timeout:
    members = {}
    for name in ["madaros", "fixed-point/check.log", "fixed-point/gen2.log"]:
        a, b = success.read(name), timeout.read(name)
        members[name] = {"success_sha256": sha(a), "timeout_sha256": sha(b),
                         "success_bytes": len(a), "timeout_bytes": len(b),
                         "identical": a == b}
    a, b = success.read("fixed-point/gen2.log"), timeout.read("fixed-point/gen2.log")
    prefix = a.startswith(b)
    result = {"schema": "pireus-ci-binary-comparison-v1",
              "success_run": 34439721975, "timeout_run": 34463834489,
              "artifact_sha256": EXPECTED, "members": members,
              "timeout_log_is_success_prefix": prefix,
              "success_only_suffix": a[len(b):].decode() if prefix else None,
              "runner_equivalence_established": False,
              "memory_causes_slowdown_established": False,
              "timeout_source_qualified": False}
print(json.dumps(result, indent=2))
