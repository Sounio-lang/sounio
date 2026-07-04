#!/usr/bin/env bash
# Madaros v2 S3 readiness gate: HLIR core hygiene before SSA/hash contracts.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HLIR_IR="${ROOT_DIR}/self-hosted/hlir/ir.sio"
echo "[madaros-v2-s3] START"
echo "[madaros-v2-s3] hlir_ir=$HLIR_IR"

python3 - "$HLIR_IR" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
match = re.search(r"enum\s+HlirTypeKind\s*\{(?P<body>.*?)\n\}", text, re.S)
if not match:
    raise SystemExit("missing enum HlirTypeKind")

seen = {}
duplicates = []
for offset, raw_line in enumerate(match.group("body").splitlines(), start=text[:match.start("body")].count("\n") + 1):
    line = raw_line.split("//", 1)[0].strip().rstrip(",")
    if not line:
        continue
    name = line.split()[0]
    if not re.match(r"^HlirType[A-Za-z0-9_]+$", name):
        continue
    if name in seen:
        duplicates.append((name, seen[name], offset))
    else:
        seen[name] = offset

if duplicates:
    for name, first, second in duplicates:
        print(f"duplicate {name}: first_line={first} duplicate_line={second}", file=sys.stderr)
    raise SystemExit("duplicate HlirTypeKind variants")

required = {"HlirTypeContest", "HlirTypeRobust", "HlirTypeKnowledge", "HlirTypeValidated"}
missing = sorted(required - set(seen))
if missing:
    raise SystemExit(f"missing epistemic HLIR variants: {missing}")

print(f"[madaros-v2-s3] HlirTypeKind variants={len(seen)} duplicates=0")
PY

echo "[madaros-v2-s3] PASS: HLIR type enum unique; native HLIR roundtrip gate still pending"
