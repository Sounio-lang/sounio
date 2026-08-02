#!/usr/bin/env bash
# SCO Step IV — runtime gate: rebuilt Madaros + corpus + install/ablation markers.
#
# Env:
#   SCO_SKIP_REBUILD=1  — do not build; require pre-existing SCO_MADAROS ELF
#   SCO_MADAROS         — path to Madaros ELF (default artifacts/self-hosted/madaros-sco)
#   SCO_FORCE_REBUILD=1 — rebuild even if ELF is newer than lower.sio
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

MADAROS="${SCO_MADAROS:-$ROOT/artifacts/self-hosted/madaros-sco}"
LOWER="$ROOT/self-hosted/ir/lower.sio"
fail=0

echo "### Step IV markers ###"
for marker in SCO_ETAU_INSTALL_V1 SCO_ETAU_ELEVATE_SCIENTIFIC_EFFECTS SCO_ETAU_ABLATION_ENV; do
  if grep -q "$marker" "$LOWER"; then
    echo "PASS: $marker"
  else
    echo "FAIL: missing $marker"
    fail=1
  fi
done

need_rebuild=0
if [[ "${SCO_FORCE_REBUILD:-0}" == "1" ]]; then
  need_rebuild=1
elif [[ ! -x "$MADAROS" ]]; then
  need_rebuild=1
elif [[ "$LOWER" -nt "$MADAROS" ]]; then
  echo "NOTE: lower.sio newer than $MADAROS — rebuild required for runtime claim"
  need_rebuild=1
fi

if [[ "$need_rebuild" -eq 1 ]]; then
  if [[ "${SCO_SKIP_REBUILD:-0}" == "1" ]]; then
    echo "FAIL: rebuild required but SCO_SKIP_REBUILD=1"
    fail=1
  else
    echo "### Rebuild Madaros → $MADAROS ###"
    # build_modular_madaros.sh already serializes via souc-build-lock.sh.
    # Do NOT wrap again (non-recursive flock deadlocks on the same lockfile).
    if bash scripts/ci/build_modular_madaros.sh "$MADAROS"; then
      echo "PASS: rebuild"
    else
      echo "FAIL: rebuild"
      fail=1
    fi
  fi
else
  echo "PASS: existing ELF fresh enough: $MADAROS"
fi

if [[ -x "$MADAROS" ]]; then
  echo "### Corpus under rebuilt SOUC ###"
  # Prefer direct ELF as SOUC so we do not depend on bin/souc wrapper routing.
  if SOUC="$MADAROS" bash scripts/ci/sco_corpus_gate.sh; then
    echo "PASS: corpus under $MADAROS"
  else
    echo "FAIL: corpus under rebuilt compiler"
    fail=1
  fi

  # Ablation wiring smoke: compiler must still accept compile with ablation set.
  # (Does not yet assert strategy integers — needs IR dump probe.)
  echo "### Ablation env smoke (compile only) ###"
  OUT="$(mktemp -d)"
  trap 'rm -rf "$OUT"' EXIT
  if SOUNIO_SCO_ABLATION=1 "$MADAROS" check examples/sco_corpus/v2_algebra/v2_main.sio \
      >"$OUT/abl.check.out" 2>"$OUT/abl.check.err"; then
    echo "PASS: ablation env check"
  else
    echo "FAIL: ablation env check"
    tail -20 "$OUT/abl.check.err" || true
    fail=1
  fi
else
  echo "FAIL: no executable Madaros at $MADAROS"
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  echo "SCO_STEP4_RUNTIME_GATE_FAIL"
  exit 1
fi

echo "SCO_STEP4_RUNTIME_GATE_OK"
exit 0
