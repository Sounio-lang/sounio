#!/usr/bin/env bash
# Prove the duplicate-definition gate fails when it should, and — just as
# important — that it does NOT fire on the two shapes that fooled the scanner
# four times before it settled: methods of the same name in different impl
# blocks, and one body formatted two ways.
#
# Since #2368 it also covers the cross-MODULE half: one exported name defined in
# two files. Those controls, and the proof that they discriminate against the
# scanner they replaced, are at the bottom.
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

# GREEN — a trait DECLARES a method with no body; the impl that provides it is
# not a second definition of the signature. Before this, the body-skip walk ran
# past the bodyless declaration and swallowed the impl block whole.
cat > "$W/src/trait_decl.sio" <<'F'
trait R {
    fn radd(self, o: Self) -> Self
}

impl R for i64 {
    fn radd(self, o: Self) -> Self { self + o }
}
F
check "a trait method signature is not a duplicate of its impl" pass "identical=0" || exit 1

# GREEN — two impls of ONE trait for TWO types are two scopes, not one.
cat > "$W/src/trait_two_types.sio" <<'F'
impl Mapping for Alpha {
    fn apply(self, n: i64) -> i64 { n + 1 }
}

impl Mapping for Beta {
    fn apply(self, n: i64) -> i64 { n * 2 }
}
F
check "one trait implemented for two types is not a duplicate" pass "identical=0" || exit 1

# GREEN — #[cfg]-guarded arms are alternatives; exactly one is ever compiled.
cat > "$W/src/cfg_arms.sio" <<'F'
#[cfg(target_arch = "x86_64")]
fn fast_hash(x: i64) -> i64 { x * 31 }

#[cfg(target_arch = "aarch64")]
fn fast_hash(x: i64) -> i64 { x * 33 }
F
check "cfg-guarded alternatives are not duplicates" pass "identical=0" || exit 1

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

# ---------------------------------------------------------------------------
# CROSS-MODULE controls (#2368).
#
# A control that behaves the same before and after the change it is supposed to
# guard is worth nothing, so this was checked rather than assumed. Verified
# 2026-09-01 by dropping `git show origin/main:...duplicate_definition_gate.sh`
# in as the gate (it must sit under scripts/ci/ — the gate derives ROOT_DIR from
# its own path, and running it from /tmp silently scans `/`). All TEN controls
# above passed unchanged; all SIX below failed, verbatim:
#
#   SELFTEST FAIL: divergent pub fn in two modules is XMOD-DIVERGENT expected fail, got pass
#   SELFTEST FAIL: identical pub fn in two modules is xmod-identical expected fail, got pass
#   SELFTEST FAIL: non-pub fn in two modules is not a cross-module duplicate behaved correctly but never said 'cross_module_divergent=0'
#   SELFTEST FAIL: cfg-guarded arms in two modules are not a cross-module duplicate behaved correctly but never said 'cross_module_divergent=0'
#   SELFTEST FAIL: pub methods of two types in two modules are not a cross-module duplicate behaved correctly but never said 'cross_module_divergent=0'
#   SELFTEST FAIL: a pub duplicate inside ONE file is not counted cross-module behaved correctly but never said 'cross_module_divergent=0'
#
# The RED ones discriminate on the VERDICT: the old scanner looked inside one
# file at a time and had nothing to say about two. The GREEN ones cannot
# discriminate that way -- an old scanner that reports nothing cross-module is
# trivially right about a shape that should report nothing -- so each pins the
# printed count instead, which the old scanner does not emit. They are stated
# as false-positive guards on the NEW code path, not as evidence of a fix.

# RED — the #2368 shape itself: one exported name, two modules, DIFFERENT
# bodies, and the importer picks by module so the other is unreachable.
cat > "$W/src/xmod_a.sio" <<'F'
pub fn compile_pipeline(p: i64) -> i64 {
    inline_pass(p)
    tco_pass(p)
}
F
cat > "$W/src/xmod_b.sio" <<'F'
pub fn compile_pipeline(p: i64) -> i64 {
    p
}
F
check "divergent pub fn in two modules is XMOD-DIVERGENT" fail "XMOD-DIVERGENT" || exit 1
rm "$W/src/xmod_a.sio" "$W/src/xmod_b.sio"

# RED — identical bodies in two modules are still one dead copy, and are
# reported separately so the two can be ratcheted apart.
cat > "$W/src/xmod_same_a.sio" <<'F'
pub fn shared_helper(x: i64) -> i64 { x * 2 }
F
cat > "$W/src/xmod_same_b.sio" <<'F'
pub fn shared_helper(x: i64) -> i64 {
    x * 2
}
F
check "identical pub fn in two modules is xmod-identical" fail "xmod-identical" || exit 1
rm "$W/src/xmod_same_a.sio" "$W/src/xmod_same_b.sio"

# GREEN — a NON-pub fn is not exported, and 959 of these exist in self-hosted/
# (bootstrap_v0.sio alone redefines most of parser/ by design). Gating them is
# not possible; see the gate header for the measured cuts.
cat > "$W/src/priv_a.sio" <<'F'
fn local_helper(x: i64) -> i64 { x + 1 }
F
cat > "$W/src/priv_b.sio" <<'F'
fn local_helper(x: i64) -> i64 { x + 999 }
F
check "non-pub fn in two modules is not a cross-module duplicate" pass "cross_module_divergent=0" || exit 1
rm "$W/src/priv_a.sio" "$W/src/priv_b.sio"

# GREEN — #[cfg]-guarded alternatives split ACROSS two files. Within one file
# this is already handled; the cross-module aggregation must inherit it and not
# reintroduce the false positive.
cat > "$W/src/cfg_x86.sio" <<'F'
#[cfg(target_arch = "x86_64")]
pub fn fast_hash(x: i64) -> i64 { x * 31 }
F
cat > "$W/src/cfg_arm.sio" <<'F'
#[cfg(target_arch = "aarch64")]
pub fn fast_hash(x: i64) -> i64 { x * 33 }
F
check "cfg-guarded arms in two modules are not a cross-module duplicate" pass "cross_module_divergent=0" || exit 1
rm "$W/src/cfg_x86.sio" "$W/src/cfg_arm.sio"

# GREEN — a method is not a module export. Two types in two files may each have
# `get`; keying those by name is the shape that produced a false 104.
cat > "$W/src/meth_a.sio" <<'F'
impl Alpha {
    pub fn get(self, i: i64) -> i64 { self.xs[i] }
}
F
cat > "$W/src/meth_b.sio" <<'F'
impl Beta {
    pub fn get(self, i: i64) -> i64 { self.ys[i] * 7 }
}
F
check "pub methods of two types in two modules are not a cross-module duplicate" pass "cross_module_divergent=0" || exit 1
rm "$W/src/meth_a.sio" "$W/src/meth_b.sio"

# GREEN — the same exported name defined TWICE IN ONE FILE is the within-file
# finding, not a cross-module one. Without the distinct-file requirement the
# two halves double-count and one ratchet moves the other.
cat > "$W/src/one_file_pub.sio" <<'F'
pub fn only_here(x: i64) -> i64 { x * 2 }

pub fn only_here(x: i64) -> i64 { x * 2 }
F
check "a pub duplicate inside ONE file is not counted cross-module" fail "cross_module_divergent=0" || exit 1
rm "$W/src/one_file_pub.sio"

# RED — an empty scan is a broken instrument, not a clean tree.
check_empty() {
  local out rc
  out="$(SOUNIO_DUPDEF_ROOT="$W/nothing" SOUNIO_DUPDEF_MIN_FILES=1 GATE_ARTIFACT="$W/out.json" bash "$GATE" 2>&1)"; rc=$?
  [ $rc -ne 0 ] && grep -qF "CONTROL-FAIL" <<<"$out" \
    && echo "  ok  empty scan refuses (fail)" \
    || { echo "SELFTEST FAIL: empty scan did not refuse" >&2; echo "$out" | sed 's/^/    /' >&2; return 1; }
}
mkdir -p "$W/nothing"; check_empty || exit 1

echo "DUPLICATE_DEFINITION_SELFTEST_OK: 16 controls, 7 of them RED, each behaved as stated"
