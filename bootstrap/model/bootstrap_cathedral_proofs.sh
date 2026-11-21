#!/bin/sh
# Verifies BootstrapCathedral kernel proofs, `#print axioms` purity
# (subset of {propext, Classical.choice, Quot.sound}, zero sorryAx),
# banned-token absence, and lakefile isolation.
# Pure POSIX sh + awk (zero Python).
#
# Usage:
#   bootstrap/model/bootstrap_cathedral_proofs.sh            # run --selftest + full Lean gate
#   bootstrap/model/bootstrap_cathedral_proofs.sh --selftest # fast negative/positive parser controls
set -eu

MODE="${1:-run}"
case "$MODE" in
  run|--selftest) ;;
  *)
    echo "usage: $0 [--selftest]" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FORMAL="$ROOT/formal"
CATHEDRAL="$FORMAL/Bootstrap/BootstrapCathedral.lean"
LAKEFILE="$FORMAL/lakefile.lean"

fail() {
  echo "bootstrap_cathedral_proofs: FAIL: $*" >&2
  exit 1
}

check_banned_tokens() {
  target_file="$1"
  [ -f "$target_file" ] || fail "missing Lean file: $target_file"
  awk '
    {
      line = $0
      gsub(/[^A-Za-z0-9_]/, " ", line)
      n = split(line, tok, /[[:space:]]+/)
      for (i = 1; i <= n; i++) {
        t = tok[i]
        if (t == "sorry" || t == "admit" || t == "axiom" || t == "constant" ||
            t == "native_decide" || t == "Mathlib" || t == "unsafe" ||
            t == "implemented_by" || t == "extern" || t == "ofReduceBool" ||
            t == "csimp" || t == "partial" || t == "opaque" || t == "noncomputable") {
          printf "%s:%d: banned token `%s`\n", FILENAME, NR, t > "/dev/stderr"
          bad = 1
        }
      }
    }
    END { if (bad) exit 1 }
  ' "$target_file"
}

check_lakefile_isolation() {
  [ -f "$LAKEFILE" ] || fail "missing $LAKEFILE"
  grep -q 'lean_lib BootstrapCathedral' "$LAKEFILE" ||
    fail "BootstrapCathedral library not registered in formal/lakefile.lean"
  awk '
    /^lean_lib Bootstrap[[:space:]]/ { in_pr = 1; next }
    /^lean_lib / || /^lean_exe / { in_pr = 0 }
    in_pr && /(BootstrapCathedral|SporosCheckArm64|SporosVm|SporosProvLen)/ {
      printf "formal/lakefile.lean:%d: heavy module leaked into PR Bootstrap: %s\n", NR, $0 > "/dev/stderr"
      bad = 1
    }
    END { if (bad) exit 1 }
  ' "$LAKEFILE"
}

verify_axiom_log() {
  want="$1"
  log_file="$2"
  [ -f "$log_file" ] || fail "missing axiom log: $log_file"
  awk -v want="$want" '
    BEGIN {
      allowed["propext"] = 1
      allowed["Classical.choice"] = 1
      allowed["Quot.sound"] = 1
      got = 0
      bad = 0
    }
    /sorryAx/ {
      printf "forbidden sorryAx found: %s\n", $0 > "/dev/stderr"
      bad = 1
    }
    /does not depend on any axioms/ {
      got++
      next
    }
    /depends on axioms: \[/ {
      got++
      s = $0
      sub(/^.*depends on axioms: \[/, "", s)
      sub(/\].*$/, "", s)
      n = split(s, ax, ",")
      for (i = 1; i <= n; i++) {
        a = ax[i]
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", a)
        if (a != "" && !(a in allowed)) {
          printf "forbidden non-standard axiom `%s` in line: %s\n", a, $0 > "/dev/stderr"
          bad = 1
        }
      }
      next
    }
    END {
      if (got != want + 0) {
        printf "expected %d #print axioms outputs, got %d\n", want, got > "/dev/stderr"
        bad = 1
      }
      if (bad) exit 1
    }
  ' "$log_file"
}

run_selftest() {
  w="$(mktemp -d)"
  trap 'rm -rf "$w"' EXIT

  # 1. Positive control: empty axioms + standard constructive/classical core
  cat > "$w/ok.log" <<'EOF'
'T1' does not depend on any axioms
'T2' depends on axioms: [propext, Classical.choice, Quot.sound]
EOF
  verify_axiom_log 2 "$w/ok.log" || fail "selftest positive control failed"

  # 2. Negative control: sorryAx must fail
  cat > "$w/sorry.log" <<'EOF'
'T1' depends on axioms: [propext, sorryAx]
EOF
  if verify_axiom_log 1 "$w/sorry.log" 2>/dev/null; then
    fail "selftest failed to reject sorryAx"
  fi

  # 3. Negative control: non-standard axiom must fail
  cat > "$w/custom.log" <<'EOF'
'T1' depends on axioms: [propext, customUntrustedAxiom]
EOF
  if verify_axiom_log 1 "$w/custom.log" 2>/dev/null; then
    fail "selftest failed to reject non-standard axiom"
  fi

  # 4. Negative control: truncated output (count mismatch) must fail
  cat > "$w/short.log" <<'EOF'
'T1' does not depend on any axioms
EOF
  if verify_axiom_log 2 "$w/short.log" 2>/dev/null; then
    fail "selftest failed to reject missing #print axioms line"
  fi

  # 5. Negative control: banned tokens in Lean source must fail
  for tok in sorry native_decide axiom ofReduceBool Mathlib; do
    printf 'theorem evil : True := by %s\n' "$tok" > "$w/bad_$tok.lean"
    if check_banned_tokens "$w/bad_$tok.lean" 2>/dev/null; then
      fail "selftest failed to reject banned token $tok"
    fi
  done

  # 6. Verify real repository files (if present) have zero banned tokens and clean lakefile isolation
  check_banned_tokens "$CATHEDRAL"
  check_lakefile_isolation

  rm -rf "$w"
  trap - EXIT
  echo "bootstrap_cathedral_proofs: selftest PASS (6 controls + source/lakefile hygiene)"
}

if [ "$MODE" = "--selftest" ]; then
  run_selftest
  exit 0
fi

# Full run always runs self-tests first
run_selftest

pick_lake() {
  if [ -n "${LAKE:-}" ] && [ -x "${LAKE}" ]; then
    printf '%s' "$LAKE"
    return 0
  fi
  if command -v lake >/dev/null 2>&1; then
    command -v lake
    return 0
  fi
  if [ -x "${HOME:-}/.elan/bin/lake" ]; then
    printf '%s' "$HOME/.elan/bin/lake"
    return 0
  fi
  fail "lake not found"
}

[ -f "$FORMAL/lean-toolchain" ] || fail "missing $FORMAL/lean-toolchain"
TOOLCHAIN="$(tr -d ' \t\r\n' < "$FORMAL/lean-toolchain")"
case "$TOOLCHAIN" in
  leanprover/lean4:v4.33.0) ;;
  *) fail "unexpected formal/lean-toolchain '$TOOLCHAIN' (expected leanprover/lean4:v4.33.0)" ;;
esac

LAKE_BIN="$(pick_lake)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "bootstrap_cathedral_proofs: building BootstrapCathedral ($TOOLCHAIN)..."
(cd "$FORMAL" && ELAN_TOOLCHAIN="$TOOLCHAIN" "$LAKE_BIN" build BootstrapCathedral)

cat > "$WORK/CheckAxioms.lean" <<'EOF'
import Bootstrap.BootstrapCathedral

#print axioms BootstrapCathedral.tri_root_seed_genesis
#print axioms BootstrapCathedral.tri_root_cross_isa_iff
#print axioms BootstrapCathedral.tri_root_cross_isa_output_agreement
#print axioms BootstrapCathedral.tri_root_cross_isa_refuses
#print axioms BootstrapCathedral.committed_binaries_identity_links
#print axioms BootstrapCathedral.arm64_admit_to_x86vm_cathedral
#print axioms BootstrapCathedral.cathedral_ground_witnesses
#print axioms BootstrapCathedral.dual_isa_stage1_cathedral_iff
#print axioms BootstrapCathedral.rootA_admit_to_x86vm_cathedral
#print axioms BootstrapCathedral.stage1_cathedral_ground_witnesses
EOF

WANT="$(grep -c '^#print axioms ' "$WORK/CheckAxioms.lean")"
(cd "$FORMAL" && ELAN_TOOLCHAIN="$TOOLCHAIN" "$LAKE_BIN" env lean "$WORK/CheckAxioms.lean") > "$WORK/axioms.log" 2>&1 || {
  cat "$WORK/axioms.log" >&2
  fail "lean CheckAxioms.lean failed"
}

cat "$WORK/axioms.log"
verify_axiom_log "$WANT" "$WORK/axioms.log"

echo "bootstrap_cathedral_proofs: PASS ($WANT crown theorems verified <= {propext, Classical.choice, Quot.sound}, 0 sorryAx)"
echo "BOOTSTRAP_CATHEDRAL_PROOFS_OK"
