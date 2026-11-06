#!/usr/bin/env bash
# Verify the unified Cayley-Dickson tower through level 6 (64 dimensions).
#
# Three legs must agree, with no tolerance and no marker-only self-assertion:
#   1. the exact-integer Python oracle, whose two evaluators (XOR/sign kernel
#      and a recursive Dickson doubling with a scalar base case) share no code;
#   2. the Sounio public API under Madaros;
#   3. the Sounio public API under lean_single.
# Each Sounio leg emits the exact product coefficients of the shared oracle
# cases (levels 3/4/6, seeds 2x5) on BOTH public multiplication routes, and
# every emission must equal the oracle's expectation. Each leg also proves its
# engine identity from engine-specific compiler markers, so an inherited
# override cannot collapse both legs onto one engine.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../lib/classify_compile.sh
source "$ROOT_DIR/scripts/lib/classify_compile.sh"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/cd-tower.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT

fail() { echo "CD_TOWER_GATE_FAIL: $1" >&2; exit 1; }

# Engine identity must come from this gate alone. SOUNIO_SOUC_BIN execs a raw
# ELF and ignores the engine selector; MADAROS_RAW_BIN/SOUNIO_MADAROS_BIN
# retarget the Madaros leg; an inherited SOUNIO_SOUC_ENGINE would be pinned
# per-command below, but refusing it keeps the calling environment honest.
# This check runs before the gate binds its own launcher path, so it only ever
# sees inherited state.
for var in SOUC SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE MADAROS_RAW_BIN SOUNIO_MADAROS_BIN; do
  [[ -z "${!var:-}" ]] || fail "inherited override $var is set; engine selection belongs to this gate"
done

SOUC="$ROOT_DIR/bin/souc"
[[ -x "$SOUC" ]] || fail "compiler launcher not executable: $SOUC"

# 1. Oracle leg: the two independent evaluators must agree internally, and the
#    exact expectations they emit drive the cross-implementation comparison.
python3 scripts/ci/cayley_dickson_tower_oracle.py >"$OUT/oracle.log"
grep -Fq 'CD_TOWER_ORACLE_OK levels=6 product_cases=34 basis_triples=283584 nonassociative=k1=0,k2=0,k3=168,k4=1848,k5=15960,k6=130200' "$OUT/oracle.log" \
  || { cat "$OUT/oracle.log" >&2; fail "oracle self-check failed"; }
grep -E '^CD_TOWER_ORACLE_EXPECT ' "$OUT/oracle.log" >"$OUT/expected.txt" || true
[[ "$(wc -l <"$OUT/expected.txt")" == "3" ]] || fail "oracle emitted $(wc -l <"$OUT/expected.txt") expectation lines, want 3"
cat "$OUT/oracle.log"

# Gate-local emitter: prints the exact product coefficients for the shared
# oracle cases on both public multiplication routes, so this gate compares
# numbers with the oracle instead of trusting a success marker.
cat >"$OUT/emit_coeffs.sio" <<'EOF'
use algebra::cayley_dickson_tower::*

fn tower_vec(bits: i32, seed: i32) -> CDElement with Mut, Panic, Div {
    let n = 1 << bits
    var x = cd_zero(bits)
    var i: i32 = 0
    while i < n {
        let coeff = (i * 17 + seed * 11 + bits * 7) % 7 - 3
        x.c[i as usize] = coeff as f64
        i = i + 1
    }
    x
}

fn emit_coeffs(bits: i32, route: i32, product: CDElement) -> i32 with IO, Mut, Div, Panic {
    print("CD_TOWER_EMIT k=")
    print_int(bits)
    if route == 0 {
        print(" route=direct coeffs=")
    } else {
        print(" route=doubling coeffs=")
    }
    var i: i32 = 0
    while i < 64 {
        if i > 0 { print(",") }
        print_int(product.c[i as usize] as i64)
        i = i + 1
    }
    print("\n")
    return 0
}

fn main() -> i32 with IO, Mut, Div, Panic {
    emit_coeffs(3, 0, cd_mul(tower_vec(3, 2), tower_vec(3, 5)))
    emit_coeffs(3, 1, cd_mul_doubling(tower_vec(3, 2), tower_vec(3, 5)))
    emit_coeffs(4, 0, cd_mul(tower_vec(4, 2), tower_vec(4, 5)))
    emit_coeffs(4, 1, cd_mul_doubling(tower_vec(4, 2), tower_vec(4, 5)))
    emit_coeffs(6, 0, cd_mul(tower_vec(6, 2), tower_vec(6, 5)))
    emit_coeffs(6, 1, cd_mul_doubling(tower_vec(6, 2), tower_vec(6, 5)))
    return 0
}
EOF

compare_engine() { # <engine-label> <emit-log>
  local label="$1" log="$2" k route expected emitted
  for k in 3 4 6; do
    expected="$(grep -E "^CD_TOWER_ORACLE_EXPECT k=$k " "$OUT/expected.txt" | head -1)"
    [[ -n "$expected" ]] || fail "oracle expectation for k=$k missing"
    expected="${expected#* coeffs=}"
    for route in direct doubling; do
      emitted="$(grep -E "^CD_TOWER_EMIT k=$k route=$route " "$log" | head -1)"
      [[ -n "$emitted" ]] || fail "engine=$label k=$k route=$route: no emission"
      emitted="${emitted#* coeffs=}"
      [[ "$emitted" == "$expected" ]] || {
        echo "engine=$label k=$k route=$route emitted: $emitted" >&2
        echo "engine=$label k=$k oracle            : $expected" >&2
        fail "engine=$label k=$k route=$route: coefficients differ from the oracle"
      }
    done
  done
  echo "CD_TOWER_ENGINE_COEFFS_OK engine=$label levels=3,4,6 routes=direct+doubling"
}

run_engine_leg() { # <label> <engine-selector>
  local label="$1" engine="$2"
  local check_log="$OUT/$label.check.log" run_log="$OUT/$label.run.log" emit_log="$OUT/$label.emit.log"
  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" check tests/run-pass/cayley_dickson_tower.sio >"$check_log" 2>&1 \
    || { cat "$check_log" >&2; fail "engine=$label: check failed"; }
  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" run tests/run-pass/cayley_dickson_tower.sio >"$run_log" 2>&1 \
    || { cat "$run_log" >&2; fail "engine=$label: fixture run failed"; }
  grep -Fxq 'CAYLEY_DICKSON_TOWER_OK levels=6 max_dim=64' "$run_log" \
    || { cat "$run_log" >&2; fail "engine=$label: fixture marker missing"; }
  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" run "$OUT/emit_coeffs.sio" >"$emit_log" 2>&1 \
    || { cat "$emit_log" >&2; fail "engine=$label: emitter run failed"; }
  compare_engine "$label" "$emit_log"
}

# 2. Madaros leg. Identity proof: the check log must carry the Madaros-only
#    run_check_mode verdict and the run log the Madaros emitter banner, while
#    neither may carry lean_single's `compile: fns=` line -- otherwise this leg
#    silently ran the wrong engine (e.g. via the unbuilt-checkout fallback).
run_engine_leg madaros madaros
grep -Fq 'run_check_mode: verdict=0' "$OUT/madaros.check.log" \
  || { cat "$OUT/madaros.check.log" >&2; fail "madaros leg: Madaros check marker missing"; }
grep -Fq 'Compilation successful!' "$OUT/madaros.run.log" \
  || { cat "$OUT/madaros.run.log" >&2; fail "madaros leg: Madaros run marker missing"; }
if grep -Eq '^compile: fns=' "$OUT/madaros.check.log" "$OUT/madaros.run.log" "$OUT/madaros.emit.log"; then
  fail "madaros leg: lean_single compile marker present"
fi

# 3. lean_single leg. Identity proof is the mirror image: lean_single's
#    `compile: fns=` line must be present and no Madaros marker may appear.
run_engine_leg lean_single lean_single
grep -Eq '^compile: fns=' "$OUT/lean_single.check.log" \
  || { cat "$OUT/lean_single.check.log" >&2; fail "lean_single leg: lean compile marker missing"; }
if grep -Fq 'run_check_mode:' "$OUT/lean_single.check.log" \
  || grep -Fq 'Compilation successful!' "$OUT/lean_single.run.log" "$OUT/lean_single.emit.log"; then
  fail "lean_single leg: Madaros marker present"
fi

# 4. The invalid downward embedding must fail closed; a crash, a timeout, or a
#    silent zero is not an acceptable substitute for the dimension diagnostic.
#    The current native runtime exits 1 on panic without printing its payload,
#    so attribute the refusal by requiring successful compilation first and
#    then an ordinary (non-crash, non-timeout) runtime failure.
cat >"$OUT/invalid-embed.sio" <<'EOF'
use algebra::cayley_dickson_tower::*

fn main() -> i32 with IO, Mut, Div, Panic {
    let invalid = cd_embed(cd_one(3), 2)
    print_int(invalid.c[0 as usize] as i64)
    print("\n")
    return 0
}
EOF
rc=0
sounio_capture_rc rc "$OUT/invalid-embed.log" -- \
  env SOUNIO_SOUC_ENGINE=madaros "$SOUC" run "$OUT/invalid-embed.sio"
grep -Fq 'Compilation successful!' "$OUT/invalid-embed.log" \
  || { cat "$OUT/invalid-embed.log" >&2; fail "invalid-embedding probe did not compile"; }
sounio_expect_failed_exit "$rc" "$OUT/invalid-embed.log" || {
  cat "$OUT/invalid-embed.log" >&2
  fail "invalid embedding was not refused cleanly: $(sounio_refusal_reason)"
}
echo 'CD_TOWER_NEGATIVE_OK downward_embedding=refused'

echo 'CD_TOWER_GATE_OK levels=6 max_dim=64 oracle=exact-integer engines=madaros+lean_single'
