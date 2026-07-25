#!/usr/bin/env bash
# L4 — fixed-point of the β¹¹ lean_single seed on itself (genA == genB).
# Not the full Makefile JIT→gen1→gen2→gen3 tower; that uses bin/souc-linux-x86_64
# as stage0. This measures: does the shipped β¹¹ ELF recompile lean_single.sio
# to a bit-identical successor?
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/results/deep_four"
mkdir -p "$OUT"
SEED="$ROOT/bin/souc-lean-single-x86_64"
SRC="$ROOT/self-hosted/compiler/lean_single.sio"
A="$OUT/lean_beta11_genA.elf"
B="$OUT/lean_beta11_genB.elf"
LOG="$OUT/l4_fixedpoint.RUNLOG.txt"

{
  echo "seed=$SEED"
  echo "src=$SRC"
  ls -la "$SEED"
  echo "→ genA: seed compiles lean_single"
  scripts/dev/souc-build-lock.sh "$SEED" "$SRC" "$A"
  chmod +x "$A"
  echo "→ genB: genA compiles lean_single"
  scripts/dev/souc-build-lock.sh "$A" "$SRC" "$B"
  chmod +x "$B"
  HA=$(md5sum "$A" | awk '{print $1}')
  HB=$(md5sum "$B" | awk '{print $1}')
  HS=$(md5sum "$SEED" | awk '{print $1}')
  echo "md5_seed=$HS"
  echo "md5_genA=$HA"
  echo "md5_genB=$HB"
  if [[ "$HA" == "$HB" ]]; then
    echo "L4_FIXEDPOINT_GENA_GENB=PASS"
  else
    echo "L4_FIXEDPOINT_GENA_GENB=FAIL_HONEST"
  fi
  if [[ "$HS" == "$HA" ]]; then
    echo "L4_SEED_EQ_GENA=PASS"
  else
    echo "L4_SEED_EQ_GENA=FAIL_HONEST"
    echo "L4_NOTE=seed may lag genA if seed was built by older parent"
  fi
} 2>&1 | tee "$LOG"

python3 - <<'PY'
import json, re, subprocess
from pathlib import Path
from datetime import datetime, timezone
log = Path("results/deep_four/l4_fixedpoint.RUNLOG.txt").read_text()
sha = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
def g(k):
    m = re.search(rf"^{re.escape(k)}(.*)$", log, re.M)
    return m.group(1).strip() if m else None
rec = {
  "schema": "deep_four.l4_lean_single_fixedpoint.v1",
  "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  "git_sha": sha,
  "md5_seed": g("md5_seed="),
  "md5_genA": g("md5_genA="),
  "md5_genB": g("md5_genB="),
  "genA_eq_genB": g("L4_FIXEDPOINT_GENA_GENB="),
  "seed_eq_genA": g("L4_SEED_EQ_GENA="),
  "scope": "self-compile fixed point of bin/souc-lean-single-x86_64 on lean_single.sio; not full make build tower",
}
Path("results/deep_four/l4_fixedpoint.receipt.v1.json").write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
PY
