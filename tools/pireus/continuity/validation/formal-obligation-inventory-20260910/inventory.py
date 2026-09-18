"""Inventory historical receipt bytes; does not execute or certify their claims."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE = "342b3f4e36c78c48d70b3ca27b7ef69eb659001e"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PREFIX = "tools/pireus/"
PATTERNS = ("*.formal-parity.v13", "*.parity-open.v13", "*.parity-open.v14")

def inventory():
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", SOURCE, "--", "tools/pireus"],
        cwd=ROOT, text=True).splitlines()
    paths = sorted(p for p in tracked if str(Path(p).parent) == "tools/pireus"
                   and any(Path(p).match(pattern) for pattern in PATTERNS))
    if not paths:
        raise ValueError("missing receipt inventory")
    records = []
    for name in paths:
        frozen = subprocess.check_output(["git", "show", SOURCE + ":" + name], cwd=ROOT)
        current = (ROOT / name).read_bytes()
        if current != frozen:
            raise ValueError("receipt differs from source: " + name)
        fields = [line.split("=", 1) for line in frozen.decode().splitlines()
                  if "=" in line and not line.lstrip().startswith("#")]
        records.append(dict(path=name, sha256=hashlib.sha256(frozen).hexdigest(),
                            fields_in_source_order=fields))
    return dict(schema="pireus-formal-receipt-inventory-v1", source_commit=SOURCE,
                scope="historical receipt transcription and current byte comparison",
                formal_proofs_reexecuted=False, formal_closure_established=False,
                receipt_claims_independently_verified=False,
                records=records)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raw = (json.dumps(inventory(), indent=2, ensure_ascii=False) + "\n").encode()
    path = HERE / "receipt-index.json"
    if args.check:
        if path.read_bytes() != raw:
            raise ValueError("inventory changed")
        print("RECEIPT_INVENTORY_BYTES_MATCH; NO_FORMAL_REEXECUTION")
    else:
        with path.open("xb") as out:
            out.write(raw)
        print("RECEIPT_INVENTORY_WRITTEN; NO_FORMAL_REEXECUTION")

if __name__ == "__main__":
    main()
