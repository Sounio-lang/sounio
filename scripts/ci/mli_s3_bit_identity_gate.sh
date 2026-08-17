#!/usr/bin/env bash
# MLI S3 bit-identity gate (design §6.4, pin O5).
#
# Pipeline A: the resolved souc engine compiles the golden source
#             `fn add1(x: f64) -> f64 { x + 1.0 }` and the gate extracts
#             add1's bytes from the produced ELF at run time (first function
#             in the executable LOAD segment; boundary = the next
#             `push rbp; mov rbp,rsp` prologue). The golden oracle is thus
#             the SESSION emitter's actual output, never a frozen blob.
# Pipeline B: self-hosted/mli/s3_emit_runner.sio — hand-built add1 IR ->
#             ir_to_mli_scalar -> V-struct -> mli_legalize_x86 -> bytes.
# PASS iff the byte sequences are identical.
#
# Positive control (must fire on garbage): pipeline A's extraction is only
# trusted if it contains the movabs of IEEE 1.0 (bytes 72 184 .. 240 63) and
# ends in ret (195) — a wrong function boundary fails loudly here.
#
# Env: SOUC overrides the engine (default bin/souc). Instrument discipline:
# CI/Slurm callers should point SOUC at a session-built binary via
# MADAROS_RAW_BIN or a fresh bin/souc; the gate itself is engine-agnostic.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() {
    echo "MLI_S3_BIT_IDENTITY: FAIL — $1"
    exit 1
}

# ---------------------------------------------------------------- pipeline A
cat > "$TMP/golden.sio" <<'SIO'
fn add1(x: f64) -> f64 { x + 1.0 }

fn main() -> i64 with IO {
    let r = add1(41.5)
    if r == 42.5 {
        println("ok")
        return 0
    }
    println("bad")
    1
}
SIO

"$SOUC" compile "$TMP/golden.sio" -o "$TMP/golden.elf" > "$TMP/compile.log" 2>&1 \
    || fail "pipeline A compile failed (see $TMP/compile.log)"

# Executable LOAD segment: offset + filesize from readelf -lW (hex fields;
# converted with shell arithmetic — mawk has no strtonum).
read -r TOFF_HEX TSIZE_HEX < <(readelf -lW "$TMP/golden.elf" \
    | awk '$1=="LOAD" && / R E /{print $2, $5; exit}')
[ -n "${TOFF_HEX:-}" ] && [ -n "${TSIZE_HEX:-}" ] || fail "could not locate executable LOAD segment"
TOFF=$((TOFF_HEX))
TSIZE=$((TSIZE_HEX))

# add1 is the first function; its end is the byte before the SECOND
# `55 48 89 e5` prologue in the segment.
PINNED="$(od -An -v -tu1 -j "$TOFF" -N "$TSIZE" "$TMP/golden.elf" | tr -s ' \n' ' ' | awk '
{
    n = split($0, a, " ")
    end = -1
    for (i = 5; i <= n - 3; i++) {
        if (a[i] == 85 && a[i+1] == 72 && a[i+2] == 137 && a[i+3] == 229) { end = i - 1; break }
    }
    if (end < 1) { exit 1 }
    out = ""
    for (i = 1; i <= end; i++) out = out a[i] " "
    print out
}')" || fail "could not find the second prologue (function boundary)"

# Positive controls on the extraction itself.
echo "$PINNED" | grep -q "72 184 0 0 0 0 0 0 240 63" \
    || fail "extracted bytes lack the movabs(1.0) pattern — wrong boundary?"
[ "$(echo "$PINNED" | awk '{print $NF}')" = "195" ] \
    || fail "extracted bytes do not end in ret — wrong boundary?"

# ---------------------------------------------------------------- pipeline B
MLIB="$("$SOUC" run self-hosted/mli/s3_emit_runner.sio 2>/dev/null | grep '^S3BYTES ' | sed 's/^S3BYTES //')" \
    || fail "pipeline B emit runner produced no bytes"
[ -n "$MLIB" ] || fail "pipeline B emitted an empty byte list"

# ---------------------------------------------------------------- compare
A="$(echo "$PINNED" | tr -s ' ' ' ' | sed 's/^ //; s/ $//')"
B="$(echo "$MLIB" | tr -s ' ' ' ' | sed 's/^ //; s/ $//')"

NA=$(echo "$A" | wc -w)
NB=$(echo "$B" | wc -w)

if [ "$A" = "$B" ]; then
    echo "MLI_S3_BIT_IDENTITY: PASS — $NA bytes bit-identical (pipeline A = pinned emitter, pipeline B = IR->MLI->legalize_x86)"
    exit 0
fi

echo "MLI_S3_BIT_IDENTITY: FAIL — byte mismatch (A=$NA bytes, B=$NB bytes)"
echo "A: $A"
echo "B: $B"
# First differing position, for the mimicry fix loop.
awk -v a="$A" -v b="$B" 'BEGIN {
    na = split(a, xa, " "); nb = split(b, xb, " ")
    n = na < nb ? na : nb
    for (i = 1; i <= n; i++) {
        if (xa[i] != xb[i]) { printf "first diff at byte %d: A=%s B=%s\n", i-1, xa[i], xb[i]; exit }
    }
    printf "prefix identical; lengths differ (%d vs %d)\n", na, nb
}'
exit 1
