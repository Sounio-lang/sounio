#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUNIO_LOOM_CONTINUITY_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_CONTINUITY_ENGINE:-lean_single}"
STDLIB="$ROOT_DIR/stdlib"
MODULE="$STDLIB/coordination/loom_continuity.sio"
WITNESS="$ROOT_DIR/tests/compiler/loom_continuity_typestate_check.sio"
PRIVACY="$ROOT_DIR/tests/compiler/loom_continuity_privacy"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-continuity.XXXXXX")"

cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'loom-continuity-typestate: FAIL: %s\n' "$*" >&2
  exit 1
}

compose_witness() {
  local module="$1" output="$2"
  compose_source "$module" "$WITNESS" "$output"
}

compose_source() {
  local module="$1" source="$2" output="$3"
  {
    cat "$module"
    awk '
      /^use coordination::loom_continuity::\{/ { skip=1; next }
      skip && /^\}/ { skip=0; next }
      !skip { print }
    ' "$source"
  } > "$output"
}

expect_rejection() {
  local label="$1" source="$2" code="$3" rc=0
  local log="$WORK/$label.log"
  set +e
  SOUNIO_STDLIB_PATH="$STDLIB" "$SOUC" check "$source" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must reject with rc=1, got rc=$rc"
  }
  local count
  count="$(rg -c "error\[$code" "$log" || true)"
  if [[ "$code" == E039 || "$code" == E175 ]]; then
    [[ "$count" -ge 1 ]] || {
      cat "$log" >&2
      fail "$label must emit at least one $code"
    }
    return
  fi
  [[ "$count" -eq 1 ]] || {
    cat "$log" >&2
    fail "$label must emit exactly one $code"
  }
}

expect_composed_rejection() {
  local label="$1" source="$2" code="$3"
  local program="$WORK/$label.sio"
  compose_source "$MODULE" "$source" "$program"
  expect_rejection "$label" "$program" "$code"
}

[[ -x "$SOUC" ]] || fail "compiler is missing: $SOUC"

combined="$WORK/loom_continuity_kernel.sio"
compose_witness "$MODULE" "$combined"
runtime_log="$WORK/runtime.log"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$combined" >"$runtime_log" 2>&1 || {
  cat "$runtime_log" >&2
  fail 'single-module Loom continuity witness did not run'
}
rg -Fxq \
  'loom-continuity-typestate: PASS host_seal=1 linear=1 initial=1 clean=1 pod=1 signed=1 independent_pod=1 independent_clean=1 pre_spawn=1 measurement=1 measurement_roles=distinct measurement_disagreement=refused observation_authority=1 observation_authority_roles=distinct full_digest_disagreement=refused journal_principal_collapse=refused journal_quorum=2-of-3 single_share=refused quorum_principal_collapse=refused predecessor=refused signed_predecessor=refused collapsed_principal=refused pre_spawn_collapsed=refused missing_observation=refused count=refused kind=refused authority=refused' \
  "$runtime_log" || {
    cat "$runtime_log" >&2
    fail 'Loom continuity witness omitted its exact receipt'
  }

expect_rejection private-constructor \
  "$PRIVACY/loom_continuity_private_struct_main.sio" E176
expect_composed_rejection wrong-state-promotion \
  "$PRIVACY/loom_continuity_wrong_state_main.sio" E009
expect_composed_rejection unsigned-proof-for-signed-promotion \
  "$PRIVACY/loom_continuity_unsigned_proof_for_signed_promotion_main.sio" E009
expect_rejection unsealed-host-admission \
  "$PRIVACY/loom_continuity_unsealed_admission_main.sio" E175
expect_rejection private-signed-proof \
  "$PRIVACY/loom_continuity_private_signed_struct_main.sio" E176
expect_composed_rejection decision-as-independent-observation \
  "$PRIVACY/loom_continuity_decision_as_observation_main.sio" E009
expect_composed_rejection decision-facts-as-measurement \
  "$PRIVACY/loom_continuity_decision_facts_as_measurement_main.sio" E009
expect_composed_rejection full-decision-facts-as-measurement \
  "$PRIVACY/loom_continuity_full_decision_as_measurement_main.sio" E009
expect_rejection private-disjoint-principals \
  "$PRIVACY/loom_continuity_private_disjoint_principals_main.sio" E176
expect_rejection private-measurement-agreement \
  "$PRIVACY/loom_continuity_private_measurement_agreement_main.sio" E176
expect_rejection private-full-digest-agreement \
  "$PRIVACY/loom_continuity_private_full_digest_agreement_main.sio" E176
expect_rejection private-journal-quorum \
  "$PRIVACY/loom_continuity_private_journal_quorum_main.sio" E176
expect_composed_rejection single-journal-authority-as-quorum \
  "$PRIVACY/loom_continuity_single_journal_authority_as_quorum_main.sio" E009
expect_composed_rejection linear-reuse \
  "$PRIVACY/loom_continuity_linear_reuse_main.sio" E039

cp -a "$STDLIB" "$WORK/visibility-stdlib"
visibility_module="$WORK/visibility-stdlib/coordination/loom_continuity.sio"
visibility_count="$(rg -c '^fn admit_runtime_continuity\($' "$visibility_module")"
[[ "$visibility_count" -eq 1 ]] || \
  fail "expected one private host admission before sabotage, got $visibility_count"
awk '
  BEGIN { changed=0 }
  !changed && $0 == "fn admit_runtime_continuity(" {
    print "pub fn admit_runtime_continuity("
    changed=1
    next
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$visibility_module" > "$WORK/visibility-mutated.sio" || \
  fail 'could not make host admission public for sabotage control'
mv "$WORK/visibility-mutated.sio" "$visibility_module"
visibility_log="$WORK/visibility-sabotage.log"
set +e
SOUNIO_STDLIB_PATH="$WORK/visibility-stdlib" "$SOUC" check \
  "$PRIVACY/loom_continuity_unsealed_admission_main.sio" \
  >"$visibility_log" 2>&1
visibility_rc=$?
set -e
if rg -q 'error\[E175' "$visibility_log"; then
  cat "$visibility_log" >&2
  fail 'making host admission public did not remove the E175 refusal'
fi
if [[ "$visibility_rc" -ne 0 ]] && ! rg -q 'error\[E039' "$visibility_log"; then
  cat "$visibility_log" >&2
  fail 'visibility sabotage failed for a reason other than the known modular linear baseline'
fi

mkdir -p "$WORK/predecessor-stdlib/coordination"
cp "$MODULE" "$WORK/predecessor-stdlib/coordination/loom_continuity.sio"
predecessor_module="$WORK/predecessor-stdlib/coordination/loom_continuity.sio"
mutation_count="$(rg -c '^    if observed\.predecessor_semantic_head_token == 0 \{ return None \}$' \
  "$predecessor_module")"
[[ "$mutation_count" -eq 1 ]] || \
  fail "expected one Pod predecessor guard before mutation, got $mutation_count"
awk '
  BEGIN { changed=0 }
  !changed && $0 == "    if observed.predecessor_semantic_head_token == 0 { return None }" {
    print "    if false { return None }"
    changed=1
    next
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$predecessor_module" > "$WORK/predecessor-mutated.sio" || \
  fail 'could not apply the targeted predecessor-guard mutation'
mv "$WORK/predecessor-mutated.sio" "$predecessor_module"

sabotage_program="$WORK/sabotage_kernel.sio"
compose_witness "$predecessor_module" "$sabotage_program"
sabotage_log="$WORK/sabotage.log"
set +e
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$sabotage_program" >"$sabotage_log" 2>&1
sabotage_rc=$?
set -e
if [[ "$sabotage_rc" -eq 0 ]]; then
  cat "$sabotage_log" >&2
  fail 'removing the Pod predecessor guard did not expose the negative witness'
fi

mkdir -p "$WORK/signed-stdlib/coordination"
cp "$MODULE" "$WORK/signed-stdlib/coordination/loom_continuity.sio"
signed_module="$WORK/signed-stdlib/coordination/loom_continuity.sio"
signed_mutation_count="$(
  rg -c '^    if \(observed\.authenticity_mode != 1 && observed\.authenticity_mode != 2\) \|\| observed\.predecessor_receipt_token <= 0 \{$' \
    "$signed_module"
)"
[[ "$signed_mutation_count" -eq 2 ]] || \
  fail "expected two signed predecessor guards before mutation, got $signed_mutation_count"
awk '
  BEGIN { seen=0; changed=0 }
  $0 == "    if (observed.authenticity_mode != 1 && observed.authenticity_mode != 2) || observed.predecessor_receipt_token <= 0 {" {
    seen++
    if (seen == 2) {
      print "    if observed.authenticity_mode != 1 && observed.authenticity_mode != 2 {"
      changed=1
      next
    }
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$signed_module" > "$WORK/signed-mutated.sio" || \
  fail 'could not apply the signed predecessor mutation'
mv "$WORK/signed-mutated.sio" "$signed_module"

signed_sabotage_program="$WORK/signed_sabotage_kernel.sio"
compose_witness "$signed_module" "$signed_sabotage_program"
signed_sabotage_log="$WORK/signed-sabotage.log"
set +e
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$signed_sabotage_program" \
  >"$signed_sabotage_log" 2>&1
signed_sabotage_rc=$?
set -e
if [[ "$signed_sabotage_rc" -eq 0 ]]; then
  cat "$signed_sabotage_log" >&2
  fail 'removing the signed predecessor guard did not expose the negative witness'
fi

mkdir -p "$WORK/principal-stdlib/coordination"
cp "$MODULE" "$WORK/principal-stdlib/coordination/loom_continuity.sio"
principal_module="$WORK/principal-stdlib/coordination/loom_continuity.sio"
principal_mutation_count="$(
  rg -c '^    if signer_authority_token == observer_authority_token \{ return false \}$' \
    "$principal_module"
)"
[[ "$principal_mutation_count" -eq 1 ]] || \
  fail "expected one principal-disjointness guard before mutation, got $principal_mutation_count"
awk '
  BEGIN { changed=0 }
  !changed && $0 == "    if signer_authority_token == observer_authority_token { return false }" {
    print "    if false { return false }"
    changed=1
    next
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$principal_module" > "$WORK/principal-mutated.sio" || \
  fail 'could not apply the principal-disjointness mutation'
mv "$WORK/principal-mutated.sio" "$principal_module"

principal_sabotage_program="$WORK/principal_sabotage_kernel.sio"
compose_witness "$principal_module" "$principal_sabotage_program"
principal_sabotage_log="$WORK/principal-sabotage.log"
set +e
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$principal_sabotage_program" \
  >"$principal_sabotage_log" 2>&1
principal_sabotage_rc=$?
set -e
if [[ "$principal_sabotage_rc" -eq 0 ]]; then
  cat "$principal_sabotage_log" >&2
  fail 'removing the principal-disjointness guard did not expose the collapsed-principal witness'
fi

mkdir -p "$WORK/measurement-stdlib/coordination"
cp "$MODULE" "$WORK/measurement-stdlib/coordination/loom_continuity.sio"
measurement_module="$WORK/measurement-stdlib/coordination/loom_continuity.sio"
awk '
  BEGIN { in_function=0; skip_body=0; changed=0 }
  $0 == "fn measurement_tokens_agree(" {
    in_function=1
    print
    next
  }
  in_function && !skip_body {
    print
    if ($0 == ") -> bool {") {
      print "    true"
      skip_body=1
      changed=changed + 1
    }
    next
  }
  skip_body {
    if ($0 == "}") {
      print
      in_function=0
      skip_body=0
    }
    next
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$measurement_module" > "$WORK/measurement-mutated.sio" || \
  fail 'could not apply the measurement-agreement mutation'
mv "$WORK/measurement-mutated.sio" "$measurement_module"

measurement_sabotage_program="$WORK/measurement_sabotage_kernel.sio"
compose_witness "$measurement_module" "$measurement_sabotage_program"
measurement_sabotage_log="$WORK/measurement-sabotage.log"
set +e
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$measurement_sabotage_program" \
  >"$measurement_sabotage_log" 2>&1
measurement_sabotage_rc=$?
set -e
if [[ "$measurement_sabotage_rc" -eq 0 ]]; then
  cat "$measurement_sabotage_log" >&2
  fail 'forcing measurement agreement did not expose the disagreement witness'
fi

mkdir -p "$WORK/full-digest-stdlib/coordination"
cp "$MODULE" "$WORK/full-digest-stdlib/coordination/loom_continuity.sio"
full_digest_module="$WORK/full-digest-stdlib/coordination/loom_continuity.sio"
awk '
  BEGIN { in_function=0; skip_body=0; changed=0 }
  $0 == "fn full_digest_vectors_agree(" {
    in_function=1
    print
    next
  }
  in_function && !skip_body {
    print
    if ($0 == ") -> bool {") {
      print "    true"
      skip_body=1
      changed=changed + 1
    }
    next
  }
  skip_body {
    if ($0 == "}") {
      print
      in_function=0
      skip_body=0
    }
    next
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$full_digest_module" > "$WORK/full-digest-mutated.sio" || \
  fail 'could not apply the full-digest-agreement mutation'
mv "$WORK/full-digest-mutated.sio" "$full_digest_module"

full_digest_sabotage_program="$WORK/full_digest_sabotage_kernel.sio"
compose_witness "$full_digest_module" "$full_digest_sabotage_program"
full_digest_sabotage_log="$WORK/full-digest-sabotage.log"
set +e
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" run "$full_digest_sabotage_program" \
  >"$full_digest_sabotage_log" 2>&1
full_digest_sabotage_rc=$?
set -e
if [[ "$full_digest_sabotage_rc" -eq 0 ]]; then
  cat "$full_digest_sabotage_log" >&2
  fail 'forcing full-digest agreement did not expose the aliased digest witness'
fi

echo "loom-continuity-typestate: PASS positive_engine=$ENGINE negative_engine=madaros host-seal=E175 private=E176 wrong-state=E009 signed-type-separation=E009 signed-proof-private=E176 role-collapse=E009 measurement-role-collapse=E009 full-digest-role-collapse=E009 journal-quorum-role-collapse=E009 disjoint-proof-private=E176 measurement-agreement-private=E176 full-digest-agreement-private=E176 journal-quorum-private=E176 linear-reuse=E039 sabotage-host-seal=1 sabotage-predecessor-guard=1 sabotage-signed-predecessor=1 sabotage-principal-disjointness=1 sabotage-measurement-agreement=1 sabotage-full-digest-agreement=1"
