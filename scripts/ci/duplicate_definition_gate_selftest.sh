#!/usr/bin/env bash
# Prove the duplicate-definition gate fails when it should, and — just as
# important — that it does NOT fire on the two shapes that fooled the scanner
# four times before it settled: methods of the same name in different impl
# blocks, and one body formatted two ways.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
GATE="$ROOT_DIR/scripts/ci/duplicate_definition_gate.sh"

W="$(mktemp -d /tmp/dupdef_selftest.XXXXXX)"; trap 'rm -rf "$W"' EXIT
mkdir -p "$W/src"
printf 'identical=0\ndivergent=0\n' > "$W/frozen"

run() {  # run <dir> ; echoes rc and output
  SOUNIO_DUPDEF_ROOT="$1" SOUNIO_DUPDEF_MIN_FILES=1 \
  GATE_ARTIFACT="$W/out.json" bash "$GATE" 2>&1
}
check() {  # check <name> <expect pass|fail> <needle>
  local name="$1" expect="$2" needle="$3" out rc
  out="$(SOUNIO_DUPDEF_FROZEN=1 run "$W/src")"; rc=$?
  local got=pass; [ $rc -ne 0 ] && got=fail
  if [ "$got" != "$expect" ]; then
    echo "SELFTEST FAIL: $name expected $expect, got $got" >&2
    echo "$out" | sed 's/^/    /' >&2; return 1
  fi
  if [ -n "$needle" ] && ! grep -qF -- "$needle" <<<"$out"; then
    echo "SELFTEST FAIL: $name behaved correctly but never said '$needle'" >&2
    echo "$out" | sed 's/^/    /' >&2; return 1
  fi
  echo "  ok  $name ($expect)"
}

# The gate reads its baseline from a fixed path, so drive the cases by making
# the fixture tree exceed a zeroed baseline shipped alongside it.
cp scripts/ci/duplicate_definition.frozen "$W/real.frozen"
printf 'identical=0\ndivergent=0\n' > scripts/ci/duplicate_definition.frozen
restore() { cp "$W/real.frozen" scripts/ci/duplicate_definition.frozen; }
trap 'restore; rm -rf "$W"' EXIT

# GREEN — a clean tree has nothing to say.
cat > "$W/src/clean.sio" <<'F'
fn alpha(x: i64) -> i64 { x + 1 }
fn beta(x: i64) -> i64 { x + 2 }
F
check "clean tree passes" pass "identical=0" || exit 1

# GREEN — same method name in DIFFERENT impl blocks is not a duplicate.
# This is the shape that produced a false 104 on the first attempt.
cat > "$W/src/impls.sio" <<'F'
impl Alpha {
    fn get(self, i: i64) -> i64 {
        self.xs[i]
    }
}

impl Beta {
    fn get(self, i: i64) -> i64 {
        self.ys[i]
    }
}
F
check "same name in two impls is not a duplicate" pass "identical=0" || exit 1

# GREEN — one body, two formattings. Byte and whitespace comparison both
# called this a divergence; token comparison must not.
cat > "$W/src/format.sio" <<'F'
fn emit(nc: i64) { one(nc, 1); two(nc, 2) }
F
cat >> "$W/src/format.sio" <<'F'

fn emit_other(nc: i64) {
    one(nc, 1)
    two(nc, 2)
}
F
check "different names, same shape, no finding" pass "identical=0" || exit 1

# RED — a real identical duplicate at top level.
cat > "$W/src/dup_same.sio" <<'F'
fn twice(x: i64) -> i64 { x * 2 }

fn twice(x: i64) -> i64 {
    x * 2
}
F
check "identical duplicate is caught" fail "identical" || exit 1
rm "$W/src/dup_same.sio"

# RED — a brace inside a string literal must not hide the duplicate BELOW it.
# This is the case the gate missed in production: one unmatched brace in a
# pattern string kept the body-skip walking for 4012 lines, and a genuine
# duplicate past it was reported as absent. Without this control the scanner
# can go blind again and every count still reads green.
cat > "$W/src/dup_after_brace_literal.sio" <<'F'
fn parses(path: i64) -> bool {
    contains(path, "CorePair { left:")
}

fn twice_here(x: i64) -> i64 { x * 2 }

fn twice_here(x: i64) -> i64 {
    x * 2
}
F
check "a brace in a string literal does not hide a later duplicate" fail "identical" || exit 1
rm "$W/src/dup_after_brace_literal.sio"

# RED — and the divergent case is reported as DIVERGENT, not merely counted.
cat > "$W/src/dup_diff.sio" <<'F'
fn twice(x: i64) -> i64 { x * 2 }

fn twice(x: i64) -> i64 { x * 3 }
F
check "divergent duplicate is named DIVERGENT" fail "DIVERGENT" || exit 1
rm "$W/src/dup_diff.sio"

# RED — an empty scan is a broken instrument, not a clean tree.
check_empty() {
  local out rc
  out="$(SOUNIO_DUPDEF_ROOT="$W/nothing" SOUNIO_DUPDEF_MIN_FILES=1 GATE_ARTIFACT="$W/out.json" bash "$GATE" 2>&1)"; rc=$?
  [ $rc -ne 0 ] && grep -qF "CONTROL-FAIL" <<<"$out" \
    && echo "  ok  empty scan refuses (fail)" \
    || { echo "SELFTEST FAIL: empty scan did not refuse" >&2; echo "$out" | sed 's/^/    /' >&2; return 1; }
}
mkdir -p "$W/nothing"; check_empty || exit 1

echo "DUPLICATE_DEFINITION_SELFTEST_OK: 7 controls, 4 of them RED, each behaved as stated"
