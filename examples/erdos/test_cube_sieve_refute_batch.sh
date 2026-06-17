#!/usr/bin/env bash
# Gate for batch cube refutation artifacts through souc_sat.
#
# This proves the runner can emit per-cube CNF/DRAT/LRAT artifacts for a batch of
# cube subproblems. It still does not prove a global colouring obstruction,
# because no cube-cover certificate is checked here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
mkdir -p "$WORK/out"

cat > "$WORK/k6.edge" <<'EOF'
p edge 6 15
e 1 2
e 1 3
e 1 4
e 1 5
e 1 6
e 2 3
e 2 4
e 2 5
e 2 6
e 3 4
e 3 5
e 3 6
e 4 5
e 4 6
e 5 6
EOF

cat > "$WORK/k6.cubes" <<'EOF'
# one cube per line; assignments are zero-based vertex:colour
conflict: 0:0 1:1 2:2 3:3 4:4
small: 0:0
EOF

echo "cube_sieve_refute_batch_gate: workdir=$WORK"
python3 "$REFUTER" "$WORK/k6.edge" 5 "$WORK/k6.cubes" "$WORK/out" > "$WORK/refute.out"

rg -q '^cube_sieve_refute_batch v1$' "$WORK/refute.out"
rg -q '^output=dimacs_cube_refutation_batch_summary$' "$WORK/refute.out"
rg -q '^formula_kind=colourCNF$' "$WORK/refute.out"
rg -q '^n=6$' "$WORK/refute.out"
rg -q '^m=15$' "$WORK/refute.out"
rg -q '^k=5$' "$WORK/refute.out"
rg -q '^expected_vars=30$' "$WORK/refute.out"
rg -q '^base_clause_count=81$' "$WORK/refute.out"
rg -q '^souc_sha256=[0-9a-f]{64}$' "$WORK/refute.out"
rg -q '^souc_sat_source_sha256=[0-9a-f]{64}$' "$WORK/refute.out"
rg -q '^converter_sha256=[0-9a-f]{64}$' "$WORK/refute.out"
rg -q '^seed=0$' "$WORK/refute.out"
rg -q '^use_lrb=1$' "$WORK/refute.out"
rg -q '^sb_mode=0$' "$WORK/refute.out"
rg -q '^cube_count=2$' "$WORK/refute.out"
rg -q '^solver_unsat_count=2$' "$WORK/refute.out"
rg -q '^lrat_artifact_count=2$' "$WORK/refute.out"
rg -q '^failed_count=0$' "$WORK/refute.out"
rg -q '^subproblem_artifact=cnf_plus_cube_units_with_deletion_free_drat_and_lrat$' "$WORK/refute.out"
rg -q '^proof_checker=repo_local_rup_to_lrat_converter_only$' "$WORK/refute.out"
rg -q '^formal_proof_checker=none$' "$WORK/refute.out"
rg -q '^verified_claim=none$' "$WORK/refute.out"
rg -q '^global_unsat_claim=none$' "$WORK/refute.out"
rg -q '^geometry_claim=none$' "$WORK/refute.out"
rg -q '^cover_certificate_sha256=NONE$' "$WORK/refute.out"
rg -q '^promotion_gate=REJECT_NONE_CUBE_COVER_CERTIFICATE$' "$WORK/refute.out"
rg -q '^promotable=0$' "$WORK/refute.out"
rg -q '^cube index=0 id=conflict assignments=0:0,1:1,2:2,3:3,4:4 unit_lits=1,7,13,19,25 cube_assignment_count=5 cube=conflict/conflict\.cube cube_sha256=[0-9a-f]{64} cnf=conflict/conflict\.cnf cnf_sha256=[0-9a-f]{64} drat=conflict/conflict\.drat drat_sha256=[0-9a-f]{64} lrat=conflict/conflict\.lrat lrat_sha256=[0-9a-f]{64} cnf_vars=30 cnf_clauses=86 expected_cnf_clauses=86 drat_deletions=0 stdout_sha256=[0-9a-f]{64} converter_stderr_sha256=[0-9a-f]{64}$' "$WORK/refute.out"
rg -q '^cube index=1 id=small assignments=0:0 unit_lits=1 cube_assignment_count=1 cube=small/small\.cube cube_sha256=[0-9a-f]{64} cnf=small/small\.cnf cnf_sha256=[0-9a-f]{64} drat=small/small\.drat drat_sha256=[0-9a-f]{64} lrat=small/small\.lrat lrat_sha256=[0-9a-f]{64} cnf_vars=30 cnf_clauses=82 expected_cnf_clauses=82 drat_deletions=0 stdout_sha256=[0-9a-f]{64} converter_stderr_sha256=[0-9a-f]{64}$' "$WORK/refute.out"
rg -q '^status=subproblem_lrat_artifacts_emitted_unpromotable$' "$WORK/refute.out"

for cube_id in conflict small; do
  [[ -s "$WORK/out/$cube_id/$cube_id.cnf" ]]
  [[ -s "$WORK/out/$cube_id/$cube_id.drat" ]]
  [[ -s "$WORK/out/$cube_id/$cube_id.lrat" ]]
  rg -q 'empty=1' "$WORK/out/$cube_id/$cube_id.converter.stderr"
  if rg -q '^[[:space:]]*d[[:space:]]' "$WORK/out/$cube_id/$cube_id.drat"; then
    echo "error: deletion record found for cube $cube_id" >&2
    exit 1
  fi
done
rg -q '^1 0$' "$WORK/out/conflict/conflict.cnf"
rg -q '^25 0$' "$WORK/out/conflict/conflict.cnf"
rg -q '^1 0$' "$WORK/out/small/small.cnf"

cat > "$WORK/path3.edge" <<'EOF'
p edge 3 2
e 1 2
e 2 3
EOF
cat > "$WORK/path3.cubes" <<'EOF'
sat: 0:0
EOF
if python3 "$REFUTER" "$WORK/path3.edge" 3 "$WORK/path3.cubes" "$WORK/sat-out" \
    > "$WORK/sat-refute.out" 2>&1; then
  echo "error: refuter accepted a satisfiable cube subproblem" >&2
  exit 1
fi
rg -q 'souc_sat cube sat failed with exit 1' "$WORK/sat-refute.out"

echo "cube_sieve_refute_batch_gate: PASS"
