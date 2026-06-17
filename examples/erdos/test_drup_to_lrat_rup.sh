#!/usr/bin/env bash
# Focused smoke tests for the narrow RUP-addition DRUP -> LRAT converter.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONV="$ROOT/examples/erdos/drup_to_lrat_rup.py"
WORK="${WORK:-$(mktemp -d)}"

mkdir -p "$WORK"

cat > "$WORK/contradiction.cnf" <<'EOF'
p cnf 1 2
1 0
-1 0
EOF

cat > "$WORK/empty.drup" <<'EOF'
0
EOF

python3 -m py_compile "$CONV"
"$CONV" "$WORK/contradiction.cnf" "$WORK/empty.drup" "$WORK/empty.lrat" >"$WORK/ok.log" 2>&1
test -s "$WORK/empty.lrat"
# Local LRAT format consumed by `SounioSatReflect.parseLRAT`: new clause id,
# clause literals, 0, RUP hint clause ids, 0.
grep -q '^3 0 1 2 0$' "$WORK/empty.lrat"

cat > "$WORK/two-step.cnf" <<'EOF'
p cnf 2 3
1 2 0
-1 0
-2 0
EOF

cat > "$WORK/two-step.drup" <<'EOF'
2 0
0
EOF
"$CONV" "$WORK/two-step.cnf" "$WORK/two-step.drup" "$WORK/two-step.lrat" >"$WORK/two-step.log" 2>&1
grep -q '^4 2 0 1 2 0$' "$WORK/two-step.lrat"
grep -q '^5 0 2 3 4 0$' "$WORK/two-step.lrat"

cat > "$WORK/no-empty.drup" <<'EOF'
1 0
EOF
if "$CONV" "$WORK/contradiction.cnf" "$WORK/no-empty.drup" "$WORK/no-empty.lrat" >"$WORK/no-empty.log" 2>&1; then
  echo "expected proof without empty clause to fail" >&2
  exit 1
fi
grep -q "no empty-clause addition" "$WORK/no-empty.log"

cat > "$WORK/empty-not-final.drup" <<'EOF'
0
1 0
EOF
if "$CONV" "$WORK/contradiction.cnf" "$WORK/empty-not-final.drup" "$WORK/empty-not-final.lrat" >"$WORK/empty-not-final.log" 2>&1; then
  echo "expected proof with non-final empty clause to fail" >&2
  exit 1
fi
grep -q "proof line 2 appears after the final empty-clause addition" "$WORK/empty-not-final.log"

cat > "$WORK/delete.drup" <<'EOF'
d 1 0
0
EOF
# This converter intentionally supports only deletion-free RUP additions. A
# full DRUP/DRAT pipeline should use drat-trim -L or a deletion-aware converter.
if "$CONV" "$WORK/contradiction.cnf" "$WORK/delete.drup" "$WORK/delete.lrat" >"$WORK/delete.log" 2>&1; then
  echo "expected deletion proof to fail" >&2
  exit 1
fi
grep -q "deletion line" "$WORK/delete.log"

cat > "$WORK/out-of-range-positive.drup" <<'EOF'
2 0
0
EOF
if "$CONV" "$WORK/contradiction.cnf" "$WORK/out-of-range-positive.drup" "$WORK/out-of-range-positive.lrat" >"$WORK/range-positive.log" 2>&1; then
  echo "expected out-of-range proof to fail" >&2
  exit 1
fi
grep -q "proof literal outside declared range.*: 2" "$WORK/range-positive.log"

cat > "$WORK/out-of-range-negative.drup" <<'EOF'
-2 0
0
EOF
if "$CONV" "$WORK/contradiction.cnf" "$WORK/out-of-range-negative.drup" "$WORK/out-of-range-negative.lrat" >"$WORK/range-negative.log" 2>&1; then
  echo "expected negative out-of-range proof to fail" >&2
  exit 1
fi
grep -q "proof literal outside declared range.*: -2" "$WORK/range-negative.log"

echo "drup_to_lrat_rup smoke tests passed (workdir=$WORK)"
