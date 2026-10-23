#!/usr/bin/env bash
# bootstrap_memory_cert_gate.sh — Gen0 seed + Gen1 from-source cert + verify chain
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "bootstrap_memory_cert_gate"

EMIT_SH="$ROOT/scripts/bootstrap/emit_memory_certificate.sh"
VERIFY_SH="$ROOT/scripts/bootstrap/verify_memory_certificate.sh"
[[ -x "$EMIT_SH" && -x "$VERIFY_SH" ]] || {
  chmod +x "$EMIT_SH" "$VERIFY_SH" 2>/dev/null || true
}
[[ -f "$EMIT_SH" && -f "$VERIFY_SH" ]] || gate_fail "missing bootstrap emit/verify scripts"

PREBUILT="$ROOT/bin/madaros-linux-x86_64"
MADAROS="$ROOT/artifacts/self-hosted/madaros"

# Gen 0 pinned to committed prebuilt when available
if [[ -f "$PREBUILT" ]]; then
  bash "$EMIT_SH" 0 "$PREBUILT" "$ROOT/artifacts/memory_certs/madaros_gen0.json"
else
  bash "$EMIT_SH" 0
fi
SEED0="$ROOT/artifacts/memory_certs/madaros_gen0.json"
[[ -s "$SEED0" ]] || gate_fail "missing $SEED0"
if [[ -f "$PREBUILT" ]]; then
  bash "$VERIFY_SH" "$SEED0" "$PREBUILT"
else
  bash "$VERIFY_SH" "$SEED0"
fi

if [[ -x "$MADAROS" ]]; then
  bash "$EMIT_SH" 1 "$MADAROS" "$ROOT/artifacts/memory_certs/madaros_gen1.json"
  bash "$VERIFY_SH" "$ROOT/artifacts/memory_certs/madaros_gen1.json" "$MADAROS"
  # Bounds must match Gen0 (same declared budgets); ELF hashes must differ
  python3 - "$SEED0" "$ROOT/artifacts/memory_certs/madaros_gen1.json" <<'PY'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
pa = {p["id"]: p["bound_bytes"] for p in a["phases"]}
pb = {p["id"]: p["bound_bytes"] for p in b["phases"]}
assert pa == pb, (pa, pb)
ha, hb = a.get("elf_sha256") or "", b.get("elf_sha256") or ""
assert ha and hb and ha != hb, "Gen0/Gen1 must bind different ELF digests when from-source exists"
print("chain_ok: same bounds, distinct elf digests")
print(f"gen0_elf={ha[:16]}… gen1_elf={hb[:16]}…")
PY
  gate_pass "Gen0+Gen1 certificates chained"
else
  gate_pass "Gen0 certificate only (no from-source Madaros yet)"
fi
echo "measured=yes"
exit 0
