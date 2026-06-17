#!/usr/bin/env bash
# Gate for batch cube-propagation manifests.
#
# This keeps the search lane honest at the point where GPU work will fan out:
# many cubes in, per-cube replayable manifests plus a fail-closed summary out.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
BATCH="$ROOT/examples/erdos/cube_sieve_batch_manifest.py"
VALIDATOR="$ROOT/examples/erdos/validate_cube_sieve_manifest.py"

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
hard: 0:0
EOF

echo "cube_sieve_batch_manifest_gate: workdir=$WORK"
python3 "$BATCH" "$WORK/k6.edge" 5 "$WORK/k6.cubes" "$WORK/out" > "$WORK/batch.out"

rg -q '^cube_sieve_batch_manifest v1$' "$WORK/batch.out"
rg -q '^output=dimacs_cube_batch_summary$' "$WORK/batch.out"
rg -q '^cube_count=2$' "$WORK/batch.out"
rg -q '^conflict_count=1$' "$WORK/batch.out"
rg -q '^hard_count=1$' "$WORK/batch.out"
rg -q '^verified_claim=none$' "$WORK/batch.out"
rg -q '^geometry_claim=none$' "$WORK/batch.out"
rg -q '^proof_artifact_sha256=NONE$' "$WORK/batch.out"
rg -q '^promotion_gate=REJECT_NONE_PROOF_ARTIFACT$' "$WORK/batch.out"
rg -q '^promotable=0$' "$WORK/batch.out"
rg -q '^cube index=0 id=conflict cube=conflict.cube cube_sha256=[0-9a-f]{64} manifest=conflict.manifest manifest_sha256=[0-9a-f]{64} conflict=1 hard_cube=0 trail_len=5 conflict_vertex=5 final_domains=1,2,4,8,16,0$' "$WORK/batch.out"
rg -q '^cube index=1 id=hard cube=hard.cube cube_sha256=[0-9a-f]{64} manifest=hard.manifest manifest_sha256=[0-9a-f]{64} conflict=0 hard_cube=1 trail_len=5 conflict_vertex=-1 final_domains=1,30,30,30,30,30$' "$WORK/batch.out"
rg -q '^status=batch_manifest_emitted_unpromotable$' "$WORK/batch.out"

python3 "$VALIDATOR" "$WORK/out/conflict.manifest" > "$WORK/conflict.validator.log"
python3 "$VALIDATOR" "$WORK/out/hard.manifest" > "$WORK/hard.validator.log"

cat > "$WORK/duplicate-id.cubes" <<'EOF'
same: 0:0
same: 1:1
EOF

if python3 "$BATCH" "$WORK/k6.edge" 5 "$WORK/duplicate-id.cubes" "$WORK/dup-out" \
    > "$WORK/duplicate-id.out" 2>&1; then
  echo "error: batch producer accepted duplicate cube ids" >&2
  exit 1
fi
rg -q 'duplicate cube id same' "$WORK/duplicate-id.out"

cat > "$WORK/bad-token.cubes" <<'EOF'
bad: 0:0 nope
EOF

if python3 "$BATCH" "$WORK/k6.edge" 5 "$WORK/bad-token.cubes" "$WORK/bad-token-out" \
    > "$WORK/bad-token.out" 2>&1; then
  echo "error: batch producer accepted malformed cube token" >&2
  exit 1
fi
rg -q 'bad cube assignment token' "$WORK/bad-token.out"

cat "$WORK/conflict.validator.log"
cat "$WORK/hard.validator.log"
echo "cube_sieve_batch_manifest_gate: PASS"
