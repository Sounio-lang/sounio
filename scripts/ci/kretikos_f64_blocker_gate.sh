#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_F64_BLOCKER_DIR:-$(mktemp -d /tmp/kretikos-f64-blocker.XXXXXX)}"
SOURCE="examples/kretikos/structural_vec_add_f64.sio"
STRUCTURAL_MANIFEST="examples/kretikos/structural_manifest.tsv"
PTX_OUT="$OUT_DIR/vec_add_f64.ptx"
PROFILE_OUT="$OUT_DIR/profile_source.out"
STRUCTURAL_OUT="$OUT_DIR/structural_manifest.out"
STRUCTURAL_ERR="$OUT_DIR/structural_manifest.err"

mkdir -p "$OUT_DIR"

./bin/souc check "$SOURCE" >"$OUT_DIR/souc_check.out" 2>"$OUT_DIR/souc_check.err"

./bin/kretikos profile-source "$SOURCE" >"$PROFILE_OUT"
grep -Fq "profile=vec_add_f64" "$PROFILE_OUT"
grep -Fq "runtime_backed=0" "$PROFILE_OUT"
grep -Fq "profile_not_runtime_backed" "$PROFILE_OUT"

./bin/kretikos emit-ptx vec_add_f64 -o "$PTX_OUT" >"$OUT_DIR/emit_ptx.out" 2>"$OUT_DIR/emit_ptx.err"
grep -Fq "ld.global.f64" "$PTX_OUT"
grep -Fq "add.f64" "$PTX_OUT"
grep -Fq "st.global.f64" "$PTX_OUT"

set +e
./bin/kretikos run-manifest "$STRUCTURAL_MANIFEST" --require-runtime >"$STRUCTURAL_OUT" 2>"$STRUCTURAL_ERR"
manifest_rc=$?
set -e

if [[ "$manifest_rc" -eq 0 ]]; then
  echo "kretikos_f64_blocker_gate: expected structural f64 manifest to fail under --require-runtime" >&2
  exit 1
fi

grep -Fq "kretikos: source corpus result - total=1 failed=1" "$STRUCTURAL_OUT"

SUMMARY_TSV="$(sed -n 's/^kretikos: source corpus summary - //p' "$STRUCTURAL_OUT" | tail -n 1)"
if [[ -z "$SUMMARY_TSV" || ! -f "$SUMMARY_TSV" ]]; then
  echo "kretikos_f64_blocker_gate: missing structural corpus summary" >&2
  exit 1
fi

grep -Fq "vec_add_f64" "$SUMMARY_TSV"
grep -Fq "structural_only" "$SUMMARY_TSV"
grep -Fq "False" "$SUMMARY_TSV"
grep -Fq "profile_not_runtime_backed" "$SUMMARY_TSV"

echo "kretikos_f64_blocker_gate: PASS out=$OUT_DIR summary=$SUMMARY_TSV"
