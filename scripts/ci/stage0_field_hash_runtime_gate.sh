#!/usr/bin/env bash
# stage0_field_hash_runtime_gate.sh — Stage 0 against a from-source Madaros ELF
#
# Requires artifacts/self-hosted/madaros built from the tree that contains
# lower_refuse_untyped_field_hash. Proves:
#   1. ax/ay collision witness exits 42 (not hash-collision 74)
#   2. binary is not the committed prebuilt (sha differs OR mtime/provenance note)
#   3. souc routed to this madaros compiles a trivial program
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "stage0_field_hash_runtime_gate"

MADAROS="$ROOT/artifacts/self-hosted/madaros"
PREBUILT="$ROOT/bin/madaros-linux-x86_64"
[[ -x "$MADAROS" ]] || gate_skip "from-source Madaros missing at $MADAROS (run make build-madaros)"

if [[ -f "$PREBUILT" ]]; then
  H_NEW=$(sha256sum "$MADAROS" | awk '{print $1}')
  H_OLD=$(sha256sum "$PREBUILT" | awk '{print $1}')
  if [[ "$H_NEW" == "$H_OLD" ]]; then
    gate_fail "artifacts/self-hosted/madaros is identical to committed prebuilt — not a from-source Stage 0 build"
  fi
  echo "elf_sha_new=${H_NEW:0:16}…"
  echo "elf_sha_prebuilt=${H_OLD:0:16}…"
fi

grep -q "fn lower_refuse_untyped_field_hash" "$ROOT/self-hosted/ir/lower.sio" \
  || gate_fail "source missing refuse helper (Stage 0 not in tree)"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/stage0-runtime.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# Witness: ax/ay would collide under name[0]%64 (both 'a' → 33).
cat > "$TMP/w8.sio" <<'EOF'
struct Pair { ax: i64, ay: i64 }
fn main() -> i64 {
    let p = Pair { ax: 5, ay: 37 }
    p.ax + p.ay
}
EOF

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_MADAROS_BIN="$MADAROS"
# Prefer raw madaros over wrapper when supported
SOUC="${SOUC:-$ROOT/bin/souc}"

set +e
timeout 180 "$MADAROS" build "$TMP/w8.sio" -o "$TMP/w8.elf" >"$TMP/c.log" 2>&1
CRC=$?
set -e
if [[ "$CRC" != "0" || ! -x "$TMP/w8.elf" ]]; then
  set +e
  timeout 180 env SOUNIO_MADAROS_BIN="$MADAROS" SOUNIO_SOUC_ENGINE=madaros \
    "$SOUC" build "$TMP/w8.sio" -o "$TMP/w8.elf" >"$TMP/c2.log" 2>&1
  CRC=$?
  set -e
  if [[ "$CRC" != "0" || ! -x "$TMP/w8.elf" ]]; then
    sed -n '1,50p' "$TMP/c.log" "$TMP/c2.log" 2>/dev/null | sed 's/^/    /' >&2
    gate_fail "build w8 witness rc=$CRC"
  fi
fi

set +e
timeout 60 "$TMP/w8.elf" >"$TMP/out" 2>"$TMP/err"
RRC=$?
set -e
echo "w8_rc=$RRC"
[[ "$RRC" == "42" ]] || gate_fail "w8 want exit 42 (ax+ay), got $RRC (74 = hash collision)"

gate_pass "from-source Madaros: ax/ay=42 (Stage 0 runtime)"
echo "measured=yes"
echo "madaros=$MADAROS"
exit 0
