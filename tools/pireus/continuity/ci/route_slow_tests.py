"""Partition the real harness inventory; every slow witness has a required CI job."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = Path(__file__).with_name("slow-tests.json")


def partition(paths):
    spec = json.loads(MANIFEST.read_text())
    slow = {row["path"] for row in spec["tests"]}
    if len(slow) != 11 or len(spec["tests"]) != 11:
        raise ValueError("slow witness inventory changed")
    for row in spec["tests"]:
        source = ROOT / row["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("slow witness source changed")
        if source.with_suffix(".sio.timeout").read_text().strip() != str(row["timeout_seconds"]):
            raise ValueError("slow witness deadline changed")
    normalized = []
    for path in paths:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        normalized.append(p.resolve().relative_to(ROOT).as_posix())
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate harness inventory")
    if not slow.issubset(set(normalized)):
        raise ValueError("slow witnesses missing from real harness inventory")
    ordinary = [p for p in normalized if p not in slow]
    if not ordinary:
        raise ValueError("empty ordinary harness inventory")
    assert set(ordinary).isdisjoint(slow)
    assert set(ordinary) | slow == set(normalized)
    return ordinary, sorted(slow)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ordinary, slow = partition(args.input.read_text().splitlines())
    args.output.write_text("\n".join(ordinary) + "\n")
    print(json.dumps(dict(ordinary=len(ordinary), dedicated=len(slow), dedicated_tests=slow,
                          dropped=0, skipped=0, dedicated_ci_required=True)))
