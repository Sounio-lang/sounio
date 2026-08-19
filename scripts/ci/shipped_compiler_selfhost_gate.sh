#!/usr/bin/env bash
# shipped_compiler_selfhost_gate.sh — refuse silent compiler/source drift.
#
# ENGINE: Madaros (the user-facing default per bin/souc), with lean_single as a
# secondary subject. Both compile paths are exercised; failure on either fails
# the gate. Per FLEET_CONSTRAINTS, this is the gate that catches "the compiler
# that is in git does not compile the tree that is in git" — the silent-defect
# class that bit #1689 (imports cap), the DCE census at PR #1951, and the e230
# v3 patch (committed against a binary that did not match its source).
#
# Subject — the shipped compiler binaries:
#   SHIPPED_MADAROS     = bin/madaros-linux-x86_64     (the default user compiler)
#   SHIPPED_LEAN_SINGLE = bin/souc-lean-single-x86_64  (the bootstrap seed)
# Both are tracked in git. Either can be overridden via env vars. Each must be
# a raw ELF, never a wrapper script (bin/souc and bin/madaros are wrappers and
# cannot be tested directly; they route through the raw ELFs above).
#
# Object — the canonical source the shipped compiler must reproduce:
#   CANONICAL_SRC = self-hosted/compiler/lean_single.sio
# This is the same source lean_single_fixed_point_gate.sh uses as its
# fixed-point probe (stage1 == stage2 == stage3). It is the smallest source
# in self-hosted/ that exercises the full Madaros multi-module pipeline.
#
# Witness contract:
#   G1 PASS: SHIPPED_MADAROS compiles CANONICAL_SRC into an x86-64 ELF,
#            ELF starts with \x7fELF (the four-byte ELF magic).
#   G2 PASS: SHIPPED_LEAN_SINGLE compiles CANONICAL_SRC into an x86-64 ELF,
#            ELF starts with \x7fELF.
#
# Why "compiles to an ELF" is the criterion (not md5 equality):
#   - The whole point of "shipped compiler compiles current source" is that a
#     fresh clone can rebuild the compiler from source. If the shipped binary
#     cannot compile the source, the seed chain is broken — and a fresh clone
#     has no recourse.
#   - The strict fixed-point (stage1 == stage2 == stage3) is owned by
#     lean_single_fixed_point_gate.sh. This gate's job is the FIRST link in
#     that chain: "can the shipped binary, AS IS, compile the current source?"
#     Fail-closed at this question.
#
# Positive control (RED):
#   On 2026-08-19, bin/madaros-linux-x86_64 (committed 3d1f143e7a at
#   2026-08-17) fails G1 with E007 typecheck errors in
#   compile_primary_a64 / compile_postfix_tail_a64 / compile_multiplicative_a64
#   — the shipped Madaros is stale w.r.t. self-hosted/compiler/lean_single.sio
#   on origin/main. This is the canonical "compiler does not compile tree"
#   that the gate exists to catch. The gate's first run on origin/main MUST
#   come up RED on G1, with the E007 errors named in the verdict.
#
# Wiring:
#   Hand-run as of 2026-08-19. Candidate site in .github/workflows/ci.yml is
#   the "Compiler lane status contract" step, immediately after
#   scripts/ci/compiler_lane_status_gate.sh. Wire ONLY after coordinating
#   with grok-cli4 and grok-cli5, who also have work in ci.yml.
#   Run: SOUNIO_STDLIB_PATH=$PWD/stdlib bash scripts/ci/shipped_compiler_selfhost_gate.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ── Platform guard ───────────────────────────────────────────────────────────
if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
    echo "[shipped-compiler-selfhost] SKIP: Linux-only gate" >&2
    exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in
    x86_64|amd64) ;;
    *)
        echo "[shipped-compiler-selfhost] SKIP: x86-64 Linux-only gate" >&2
        exit 0
        ;;
esac

# ── Subjects ─────────────────────────────────────────────────────────────────
SHIPPED_MADAROS="${SOUNIO_SHIPPED_MADAROS:-$ROOT_DIR/bin/madaros-linux-x86_64}"
SHIPPED_LEAN_SINGLE="${SOUNIO_SHIPPED_LEAN_SINGLE:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
CANONICAL_SRC="$ROOT_DIR/self-hosted/compiler/lean_single.sio"

# Self-test: subject files exist, are executable, are raw ELFs (not wrappers).
for label_subject in "Madaros:$SHIPPED_MADAROS" "LeanSingle:$SHIPPED_LEAN_SINGLE"; do
    label="${label_subject%%:*}"
    subj="${label_subject#*:}"
    if [[ ! -e "$subj" ]]; then
        echo "[shipped-compiler-selfhost] FAIL self-test: $label subject missing: $subj" >&2
        echo "[shipped-compiler-selfhost]      override with SOUNIO_SHIPPED_${label^^}=<raw ELF>" >&2
        exit 2
    fi
    if [[ ! -x "$subj" ]]; then
        echo "[shipped-compiler-selfhost] FAIL self-test: $label subject not executable: $subj" >&2
        exit 2
    fi
    if head -c2 "$subj" 2>/dev/null | grep -q '#!'; then
        echo "[shipped-compiler-selfhost] FAIL self-test: $label subject is a wrapper script, not a raw ELF: $subj" >&2
        echo "[shipped-compiler-selfhost]      a wrapper routes to whatever compiler it likes;" >&2
        echo "[shipped-compiler-selfhost]      self-testing it does not measure this binary." >&2
        exit 2
    fi
done

if [[ ! -s "$CANONICAL_SRC" ]]; then
    echo "[shipped-compiler-selfhost] FAIL self-test: canonical source missing or empty: $CANONICAL_SRC" >&2
    exit 2
fi

echo "[shipped-compiler-selfhost] madaros=$SHIPPED_MADAROS"
echo "[shipped-compiler-selfhost] lean_single=$SHIPPED_LEAN_SINGLE"
echo "[shipped-compiler-selfhost] canonical_src=$CANONICAL_SRC"

OUT_DIR="${SOUNIO_SHIPPED_COMPILER_SELFHOST_DIR:-$(mktemp -d /tmp/sounio-shipped-compiler-selfhost.XXXXXX)}"
mkdir -p "$OUT_DIR"

# Helper: returns 0 (true) if the ELF magic is present.
is_elf_magic() {
    head -c4 "$1" 2>/dev/null | od -An -c | grep -q '177   E   L   F'
}

# ── G1: Madaros compiles canonical source ────────────────────────────────────
G1_ELF="$OUT_DIR/g1_madaros.elf"
G1_LOG="$OUT_DIR/g1_madaros.log"

echo
echo "[shipped-compiler-selfhost] running G1: Madaros compiles lean_single.sio"
set +e
timeout 600 "$SHIPPED_MADAROS" -o "$G1_ELF" "$CANONICAL_SRC" >"$G1_LOG" 2>&1
G1_RC=$?
set -e

G1_PASS=0
if [[ $G1_RC -ne 0 ]]; then
    echo "[shipped-compiler-selfhost] G1 FAIL: Madaros exit=$G1_RC" >&2
    echo "[shipped-compiler-selfhost]      last 30 lines of compiler log:" >&2
    tail -n 30 "$G1_LOG" | sed 's/^/        /' >&2
elif [[ ! -f "$G1_ELF" ]]; then
    echo "[shipped-compiler-selfhost] G1 FAIL: Madaros exit=0 but no ELF at $G1_ELF" >&2
    echo "[shipped-compiler-selfhost]      this is the bare '-o' swallow: the binary wrote nothing" >&2
    echo "[shipped-compiler-selfhost]      and reported success. last 30 lines:" >&2
    tail -n 30 "$G1_LOG" | sed 's/^/        /' >&2
elif ! is_elf_magic "$G1_ELF"; then
    echo "[shipped-compiler-selfhost] G1 FAIL: ELF at $G1_ELF does not start with ELF magic" >&2
    echo "[shipped-compiler-selfhost]      first 16 bytes (od):" >&2
    head -c16 "$G1_ELF" | od -An -tx1 | sed 's/^/        /' >&2
else
    G1_SIZE=$(stat -c%s "$G1_ELF")
    echo "[shipped-compiler-selfhost] G1 PASS: madaros.elf=$G1_SIZE bytes"
    G1_PASS=1
fi

# ── G2: lean_single compiles canonical source ────────────────────────────────
G2_ELF="$OUT_DIR/g2_lean_single.elf"
G2_LOG="$OUT_DIR/g2_lean_single.log"

echo
echo "[shipped-compiler-selfhost] running G2: lean_single compiles lean_single.sio"
set +e
# lean_single CLI is positional: <src> <out>. NO -o flag — it parses -o as a
# source filename and emits "lex error line 1: Unexpected character".
timeout 600 "$SHIPPED_LEAN_SINGLE" "$CANONICAL_SRC" "$G2_ELF" >"$G2_LOG" 2>&1
G2_RC=$?
set -e

G2_PASS=0
if [[ $G2_RC -ne 0 ]]; then
    echo "[shipped-compiler-selfhost] G2 FAIL: lean_single exit=$G2_RC" >&2
    echo "[shipped-compiler-selfhost]      last 30 lines of compiler log:" >&2
    tail -n 30 "$G2_LOG" | sed 's/^/        /' >&2
elif [[ ! -f "$G2_ELF" ]]; then
    echo "[shipped-compiler-selfhost] G2 FAIL: lean_single exit=0 but no ELF at $G2_ELF" >&2
elif ! is_elf_magic "$G2_ELF"; then
    echo "[shipped-compiler-selfhost] G2 FAIL: ELF at $G2_ELF does not start with ELF magic" >&2
else
    G2_SIZE=$(stat -c%s "$G2_ELF")
    echo "[shipped-compiler-selfhost] G2 PASS: lean_single.elf=$G2_SIZE bytes"
    G2_PASS=1
fi

# ── Final verdict ────────────────────────────────────────────────────────────
echo
echo "[shipped-compiler-selfhost] verdict: G1=$G1_PASS G2=$G2_PASS"
echo "[shipped-compiler-selfhost] artifacts: $OUT_DIR"

if [[ $G1_PASS -eq 1 && $G2_PASS -eq 1 ]]; then
    echo "[shipped-compiler-selfhost] PASS: both shipped compilers compile canonical source"
    exit 0
fi

if [[ $G1_PASS -eq 0 ]]; then
    echo "[shipped-compiler-selfhost] FAIL: G1 Madaros cannot compile current source" >&2
    echo "[shipped-compiler-selfhost]       this is the silent-defect class: a fresh clone" >&2
    echo "[shipped-compiler-selfhost]       receives a compiler that does not compile the tree" >&2
    echo "[shipped-compiler-selfhost]       it carries. Refresh the binary via Slurm and re-run." >&2
fi
if [[ $G2_PASS -eq 0 ]]; then
    echo "[shipped-compiler-selfhost] FAIL: G2 lean_single cannot compile current source" >&2
    echo "[shipped-compiler-selfhost]       lean_single is the bootstrap seed; this means a fresh" >&2
    echo "[shipped-compiler-selfhost]       clone cannot rebuild the compiler at all." >&2
fi

exit 1
