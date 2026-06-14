#!/usr/bin/env bash
# scripts/ci/canonical_compiler_gate.sh
#
# THE canonical lean_single fixed-point gate. The canonical lean_single
# bootstrap ELF must be the byte-identical self-hosting fixed point of:
#
#                   self-hosted/compiler/lean_single.sio
#
# NOTE (2026-06-14): bin/souc is now the DEFAULT-COMPILER WRAPPER that routes to
# Madaros — it is no longer the lean_single ELF. The lean_single ELF is preserved
# as bin/souc-lean-single-x86_64 and remains the bootstrap seed and the fixed
# point this gate validates. Override the checked binary with SOUNIO_CANONICAL_SOUC.
# Everything else (the Rust seed at artifacts/omega/souc-bin/, stray mc*.elf,
# scratch builds) is NON-canonical: a cold-bootstrap seed, a scratch build, or drift.
#
# Unlike scripts/ci/lean_single_fixed_point_gate.sh (which only WARNs when the
# shipped binary is out of sync — the gap that let bin/souc silently go stale,
# e.g. shipping a 2026-05-29 fixed point with NO stack-clash probe), this gate
# FAILS when the installed bin/souc is not the fixed point of the CURRENT source.
#
# Exit 0 = PASS  (bin/souc IS the canonical self-reproducing fixed point)
# Exit 1 = FAIL  (bin/souc drifted from lean_single.sio's fixed point)
# Exit 0 + SKIP on non-Linux/x86-64.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null||echo x)" != "Linux" || "$(uname -m 2>/dev/null||echo x)" != "x86_64" ]]; then
  echo "[canonical-compiler] SKIP: x86-64 Linux-only gate" >&2; exit 0
fi

# Default to the preserved lean_single ELF (bin/souc is now the Madaros wrapper).
SOUC_DEFAULT="$ROOT_DIR/bin/souc-lean-single-x86_64"
[[ -x "$SOUC_DEFAULT" ]] || SOUC_DEFAULT="$ROOT_DIR/bin/souc-linux-x86_64"
SOUC="${SOUNIO_CANONICAL_SOUC:-$SOUC_DEFAULT}"
SRC="self-hosted/compiler/lean_single.sio"
[[ -x "$SOUC" ]] || { echo "[canonical-compiler] FAIL: no executable lean_single ELF at $SOUC"; exit 1; }

WORK="$(mktemp -d /tmp/sounio-canonical.XXXXXX)"; trap 'rm -rf "$WORK"' EXIT
ulimit -s 1048576 2>/dev/null || true

# Self-reproduction: the canonical compiler, compiling its own source, must
# emit a byte-identical copy of itself. This is the fixed-point property AND
# proves bin/souc is exactly the compiler the current lean_single.sio defines.
if ! "$SOUC" "$SRC" "$WORK/repro" > "$WORK/repro.log" 2>&1; then
  echo "[canonical-compiler] FAIL: bin/souc could not compile $SRC"; tail -4 "$WORK/repro.log"; exit 1
fi
chmod +x "$WORK/repro"

BIN_MD5="$(md5sum "$SOUC"        | cut -d' ' -f1)"
REP_MD5="$(md5sum "$WORK/repro"  | cut -d' ' -f1)"
printf '[canonical-compiler] bin/souc md5     = %s\n' "$BIN_MD5"
printf '[canonical-compiler] self-compile md5 = %s\n' "$REP_MD5"

if [[ "$BIN_MD5" != "$REP_MD5" ]]; then
  echo "[canonical-compiler] FAIL: bin/souc is NOT the fixed point of $SRC."
  echo "[canonical-compiler]   bin/souc has drifted from the current source. To resync:"
  echo "[canonical-compiler]   bin/souc $SRC /tmp/s1 && /tmp/s1 $SRC /tmp/s2 && cp /tmp/s2 bin/souc"
  echo "[canonical-compiler]   (verify /tmp/s2 self-reproduces, then commit bin/souc)"
  exit 1
fi

echo "[canonical-compiler] PASS: bin/souc IS the canonical self-reproducing fixed point of $SRC (md5=$BIN_MD5)"
exit 0
