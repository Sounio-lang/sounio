#!/usr/bin/env bash
# Executable gate for the cube-sieve manifest skeleton.
#
# This proves only that the local producer smoke still compiles and emits the
# expected unpromotable K6/k=5 propagation-trail *format*. It is not a planar
# chi>=6 witness, does not certify any solver result, and does not check RUP
# validity. The hard-coded literals below are the DIMACS/Lean variable-convention
# smoke for the fixed cube 0->0, 1->1, 2->2, 3->3, 4->4 at k=5.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
SOUC="${SOUC:-$ROOT/bin/souc}"
# cube_sieve_skeleton.sio compiles and runs correctly with both engines.
# Madaros is now validated after the dynamic frame-size fix; lean_single remains
# the conservative default until Madaros self-compile fixed-point is reached.
SOUC_ENGINE="${SOUC_ENGINE:-lean_single}"
SRC="$ROOT/examples/erdos/cube_sieve_skeleton.sio"
ELF="$WORK/cube_sieve_skeleton.elf"
OUT="$WORK/cube_sieve_skeleton.out"

[[ -x "$SOUC" ]] || { echo "error: SOUC is not executable: $SOUC" >&2; exit 127; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
mkdir -p "$WORK"

echo "cube_sieve_skeleton_gate: workdir=$WORK"
if [[ "$SOUC_ENGINE" == "madaros" ]]; then
  SOUNIO_SOUC_ENGINE="$SOUC_ENGINE" "$SOUC" compile "$SRC" -o "$ELF" > "$WORK/compile.log"
else
  SOUNIO_SOUC_ENGINE="$SOUC_ENGINE" "$SOUC" "$SRC" "$ELF" > "$WORK/compile.log"
fi
chmod +x "$ELF"
"$ELF" > "$OUT"
python3 "$ROOT/examples/erdos/validate_cube_sieve_manifest.py" "$OUT" > "$WORK/validator.log"

sed 's/^  k=5$/  k=6/' "$OUT" > "$WORK/bad_k_header.out"
if python3 "$ROOT/examples/erdos/validate_cube_sieve_manifest.py" \
    "$WORK/bad_k_header.out" > "$WORK/bad_k_header.validator.log" 2>&1; then
  echo "error: validator accepted k=6 header with k=5 literals/trail" >&2
  exit 1
fi
rg -q 'bad precolour variable encoding|bad negated cube clause|unexpected fact clauses' \
  "$WORK/bad_k_header.validator.log"

sed 's/^  cube_assignment_count=5$/  cube_assignment_count=6/' "$OUT" \
  > "$WORK/bad_cube_assignment_count.out"
if python3 "$ROOT/examples/erdos/validate_cube_sieve_manifest.py" \
    "$WORK/bad_cube_assignment_count.out" > "$WORK/bad_cube_assignment_count.validator.log" 2>&1; then
  echo "error: validator accepted mismatched cube_assignment_count" >&2
  exit 1
fi
rg -q 'cube_assignment_count=6, found 5 rows' \
  "$WORK/bad_cube_assignment_count.validator.log"

sed 's/^    cube_assignment index=4 vertex=4 colour=4$/    cube_assignment index=4 vertex=4 colour=3/' \
  "$OUT" > "$WORK/bad_duplicate_cube_colour.out"
if python3 "$ROOT/examples/erdos/validate_cube_sieve_manifest.py" \
    "$WORK/bad_duplicate_cube_colour.out" > "$WORK/bad_duplicate_cube_colour.validator.log" 2>&1; then
  echo "error: validator accepted a duplicate cube colour" >&2
  exit 1
fi
rg -q 'duplicate cube assignment colour 3' \
  "$WORK/bad_duplicate_cube_colour.validator.log"

sed 's/rup_reason_clause=-25 -30 0/rup_reason_clause=-25 -29 0/' \
  "$OUT" > "$WORK/bad_reason.out"
if python3 "$ROOT/examples/erdos/validate_cube_sieve_manifest.py" \
    "$WORK/bad_reason.out" > "$WORK/bad_reason.validator.log" 2>&1; then
  echo "error: validator accepted a corrupted RUP reason clause" >&2
  exit 1
fi

sed 's/rup_reason_clause=-25 -30 0/rup_reason_clause=-30 -25 0/' \
  "$OUT" > "$WORK/bad_reason_order.out"
if python3 "$ROOT/examples/erdos/validate_cube_sieve_manifest.py" \
    "$WORK/bad_reason_order.out" > "$WORK/bad_reason_order.validator.log" 2>&1; then
  echo "error: validator accepted a swapped RUP reason clause" >&2
  exit 1
fi

cp "$OUT" "$WORK/bad_promotion.out"
printf 'promotable=1\n' >> "$WORK/bad_promotion.out"
if python3 "$ROOT/examples/erdos/validate_cube_sieve_manifest.py" \
    "$WORK/bad_promotion.out" > "$WORK/bad_promotion.validator.log" 2>&1; then
  echo "error: validator accepted an accidental promotable marker" >&2
  exit 1
fi

rg -q '^cube_sieve_skeleton v0$' "$OUT"
rg -q '^trust_boundary=search_untrusted__drat_lrat_lean_verified_required$' "$OUT"
rg -q '^section=k6_no5_smoke$' "$OUT"
rg -q '^section=g529_reference_probe$' "$OUT"
rg -q '^section=complete_graph_cube_propagation_smoke$' "$OUT"
rg -q '^  graph_family=complete_graph$' "$OUT"
rg -q '^  graph_id=0$' "$OUT"
rg -q '^  n=6$' "$OUT"
rg -q '^  k=5$' "$OUT"
rg -q '^  cube_assignment_count=5$' "$OUT"
rg -q '^    cube_assignment index=0 vertex=0 colour=0$' "$OUT"
rg -q '^    cube_assignment index=4 vertex=4 colour=4$' "$OUT"
rg -q '^  verified_claim=none$' "$OUT"
rg -q '^  geometry_claim=none$' "$OUT"
rg -q '^  proof_artifact_sha256=NONE$' "$OUT"
rg -q '^  rup_clause_negated_cube=-1 -7 -13 -19 -25 0$' "$OUT"
rg -q '^  propagation_passes=1$' "$OUT"
rg -q '^  termination_guard_tripped=0$' "$OUT"
rg -q '^  trail_len=5$' "$OUT"
rg -q '^  conflict=1$' "$OUT"
rg -q '^  conflict_vertex=5$' "$OUT"
rg -q '^  final_domains=1,2,4,8,16,0$' "$OUT"
rg -q '^promotion_gate=REJECT_NONE_PROOF_ARTIFACT$' "$OUT"
rg -q '^promotable=0$' "$OUT"
rg -q '^status=manifest_emitted_unpromotable$' "$OUT"

trail_steps="$(rg -c '^[[:space:]]+trail_step=' "$OUT")"
if [[ "$trail_steps" != "5" ]]; then
  echo "error: expected 5 propagation trail steps, got $trail_steps" >&2
  exit 1
fi
cube_assignments="$(rg -c '^[[:space:]]+cube_assignment index=' "$OUT")"
if [[ "$cube_assignments" != "5" ]]; then
  echo "error: expected 5 cube_assignment rows, got $cube_assignments" >&2
  exit 1
fi

if rg -q '^(promotion_gate=READY|promotable=1)$' "$OUT"; then
  echo "error: skeleton emitted a promotable marker" >&2
  exit 1
fi
if rg -q '^NEG_UNSUPPORTED$' "$OUT"; then
  echo "error: skeleton emitted a numeric-printing error marker" >&2
  exit 1
fi

sha256sum "$SRC" "$OUT"
cat "$WORK/validator.log"
echo "cube_sieve_skeleton_gate: PASS"
