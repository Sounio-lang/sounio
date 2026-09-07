"""Verify historical script custody, explicitly without executing those scripts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
BASELINE = "04dae64c915b6449d9f7de8309dbf82e5f73a7a7"
MANIFEST = Path(__file__).with_name("gate-custody.json")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_blobs(paths):
    names = sorted(set(paths))
    request = "".join(BASELINE + ":" + p + "\n" for p in names).encode()
    raw = subprocess.check_output(["git", "cat-file", "--batch"], input=request, cwd=ROOT)
    result = {}
    offset = 0
    for name in names:
        end = raw.index(b"\n", offset)
        header = raw[offset:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("missing original Git blob: " + name)
        size = int(header[2])
        start = end + 1
        result[name] = raw[start:start + size]
        offset = start + size + 1
    if offset != len(raw):
        raise ValueError("unexpected Git batch output")
    return result


def verify(named, manifest=None):
    if manifest is None:
        manifest = json.loads(MANIFEST.read_text())
    if manifest["schema"] != 1 or manifest["baseline_commit"] != BASELINE:
        raise ValueError("unrecognized historical baseline")
    if manifest["runtime_replay"] is not False or manifest["current_hardware_acceptance"] is not False:
        raise ValueError("custody cannot claim runtime acceptance")
    gates = manifest["gates"]
    paths = [g["path"] for g in gates]
    if len(paths) != 80 or len(set(paths)) != 80 or manifest["gate_count"] != 80:
        raise ValueError("historical gate inventory changed")
    if len(named) != 80 or set(named) != set(paths):
        raise ValueError("workflow must name the exact 80 gates once")
    deps = manifest["snapshot_dependencies"]
    if not deps:
        raise ValueError("empty historical dependency inventory")
    if set(deps) != {p for g in gates for p in g["literal_snapshot_dependencies"]}:
        raise ValueError("historical dependency association differs")
    for path in [*paths, *deps]:
        p = Path(path)
        if p.is_absolute() or ".." in p.parts or "\n" in path or ":" in path:
            raise ValueError("invalid repository path")
    blobs = git_blobs([*paths, *deps])
    for gate in gates:
        if gate["scope"] not in {"frozen-authority-replay", "current-or-material-gate"}:
            raise ValueError("unrecognized execution scope")
        if gate["custody_verification_executes_gate"] is not False:
            raise ValueError("custody cannot promote a gate to behavioral execution")
        path = gate["path"]
        if sha(blobs[path]) != gate["sha256"] or sha((ROOT / path).read_bytes()) != gate["sha256"]:
            raise ValueError("historical gate script differs: " + path)
    drift = []
    for path, expected in deps.items():
        if sha(blobs[path]) != expected:
            raise ValueError("historical snapshot dependency differs: " + path)
        current = ROOT / path
        if not current.is_file() or sha(current.read_bytes()) != expected:
            drift.append(path)
    return dict(schema=1, baseline_commit=BASELINE, named_gates=len(paths),
                verified_original_snapshot_dependencies=len(deps),
                scopes=dict(Counter(g["scope"] for g in gates)),
                current_tree_dependency_drift=sorted(drift),
                dependency_inventory_complete=False,
                dependency_inventory_scope=manifest["dependency_scope"],
                runtime_replay=False, behavioral_gate_execution=False,
                current_hardware_acceptance=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gates", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.gates), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")
