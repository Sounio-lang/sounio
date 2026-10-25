#!/usr/bin/env bash
# Verify the unified Cayley-Dickson tower through level 6 (64 dimensions).
# Two independent implementations must agree: the Sounio public API and an
# exact-integer Python oracle. No tolerance or compiler-only self-assertion.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../lib/classify_compile.sh
source "$ROOT_DIR/scripts/lib/classify_compile.sh"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/cd-tower.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT

[[ -x "$SOUC" ]] || { echo "CD_TOWER_GATE_FAIL: compiler not executable: $SOUC" >&2; exit 1; }

python3 scripts/ci/cayley_dickson_tower_oracle.py >"$OUT/oracle.log"
grep -Fq 'CD_TOWER_ORACLE_OK levels=6 product_cases=34 basis_triples=283584 nonassociative=k1=0,k2=0,k3=168,k4=1848,k5=15960,k6=130200' "$OUT/oracle.log"
cat "$OUT/oracle.log"

"$SOUC" check tests/run-pass/cayley_dickson_tower.sio >"$OUT/check.log" 2>&1
"$SOUC" run tests/run-pass/cayley_dickson_tower.sio >"$OUT/run.log" 2>&1
grep -Fxq 'CAYLEY_DICKSON_TOWER_OK levels=6 max_dim=64' "$OUT/run.log"
cat "$OUT/run.log"

# The API is engine-independent: repeat the same identities on lean_single.
SOUNIO_SOUC_ENGINE=lean_single "$SOUC" check tests/run-pass/cayley_dickson_tower.sio >"$OUT/lean-check.log" 2>&1
SOUNIO_SOUC_ENGINE=lean_single "$SOUC" run tests/run-pass/cayley_dickson_tower.sio >"$OUT/lean-run.log" 2>&1
grep -Fxq 'CAYLEY_DICKSON_TOWER_OK levels=6 max_dim=64' "$OUT/lean-run.log"
echo 'CD_TOWER_LEAN_SINGLE_OK levels=6 max_dim=64'

# Invalid downward embedding must fail closed; a crash or a silent zero is not
# an acceptable substitute for the dimension diagnostic.
cat >"$OUT/invalid-embed.sio" <<'EOF'
use algebra::cayley_dickson::{cd_one, cd_embed}
fn main() -> i32 with IO, Mut, Div, Panic {
    let invalid = cd_embed(cd_one(3), 2)
    print_int(invalid.c[0 as usize] as i64)
    print("\n")
    return 0
}
EOF
rc=0
sounio_capture_rc rc "$OUT/invalid-embed.log" -- "$SOUC" run "$OUT/invalid-embed.sio"
# The current native runtime exits 1 on panic without printing its payload, so
# attribute the refusal by requiring successful compilation first and then an
# ordinary (non-crash, non-timeout) runtime failure from the invalid program.
grep -Fq 'Compilation successful!' "$OUT/invalid-embed.log"
sounio_expect_failed_exit "$rc" "$OUT/invalid-embed.log" || {
  cat "$OUT/invalid-embed.log" >&2
  echo "CD_TOWER_GATE_FAIL: invalid embedding was not refused cleanly: $(sounio_refusal_reason)" >&2
  exit 1
}
echo 'CD_TOWER_NEGATIVE_OK downward_embedding=refused'

echo 'CD_TOWER_GATE_OK levels=6 max_dim=64 oracle=exact-integer engines=madaros+lean_single'
