#!/usr/bin/env bash
# memory_certificate_gate.sh — Spine A work-novelty: emit + verify Gen 0 cert
#
# Proves the certificate pipeline runs on this machine:
#   1. emit Estimated cert for generation 0 against declared phase bounds
#   2. verify structure (no Unknown)
#   3. emit a second copy and assert fixed-point on phase bounds
#   4. optionally bind to committed madaros ELF sha256 when present
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "memory_certificate_gate"

require_tool python3 "emit/verify memory certificates"

EMIT="$ROOT/scripts/memory_cert/emit_certificate.py"
VERIFY="$ROOT/scripts/memory_cert/verify_certificate.py"
BOUNDS="$ROOT/artifacts/memory_certs/madaros_phase_bounds.v1.json"
[[ -f "$EMIT" && -f "$VERIFY" && -f "$BOUNDS" ]] || gate_fail "missing emit/verify/bounds"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/memory-cert-gate.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

ELF=""
for candidate in \
  "$ROOT/bin/madaros-linux-x86_64" \
  "$ROOT/artifacts/self-hosted/madaros" \
  "$ROOT/bin/madaros"
do
  if [[ -f "$candidate" ]]; then
    ELF="$candidate"
    break
  fi
done

emit_one() {
  local out="$1"
  local -a args=(python3 "$EMIT" --generation 0 --out "$out" --repo "$ROOT" --bounds "$BOUNDS" --kind Estimated)
  if [[ -n "$ELF" ]]; then
    args+=(--elf "$ELF")
  fi
  "${args[@]}"
}

emit_one "$TMP/gen0.json"
[[ -s "$TMP/gen0.json" ]] || gate_fail "empty certificate"

if [[ -n "$ELF" ]]; then
  python3 "$VERIFY" "$TMP/gen0.json" --elf "$ELF"
else
  python3 "$VERIFY" "$TMP/gen0.json"
fi

emit_one "$TMP/gen0b.json" >/dev/null
python3 "$VERIFY" "$TMP/gen0.json" --fixed-point-with "$TMP/gen0b.json"

python3 "$EMIT" --generation 0 --out "$TMP/unk.json" --repo "$ROOT" --bounds "$BOUNDS" --kind Unknown >/dev/null
set +e
python3 "$VERIFY" "$TMP/unk.json" >/dev/null 2>"$TMP/unk.err"
URC=$?
set -e
[[ "$URC" != "0" ]] || gate_fail "Unknown bound should fail verify"

SEED="$ROOT/artifacts/memory_certs/madaros_gen0.seed.json"
SEED_ARGS=(python3 "$EMIT" --generation 0 --out "$SEED" --repo "$ROOT" --bounds "$BOUNDS"
  --kind Estimated --notes "CI seed: Estimated phase budgets; not Lean-proved.")
if [[ -n "$ELF" ]]; then
  SEED_ARGS+=(--elf "$ELF")
fi
"${SEED_ARGS[@]}"

gate_pass "emit+verify+fixed-point+Unknown-refused"
echo "measured=yes"
echo "seed=$SEED"
echo "elf=${ELF:-none}"
exit 0
