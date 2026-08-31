#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
INGRESS_SOURCE="$ROOT_DIR/tools/loom/src/loom_exec_ingress.ml"
INGRESS_RUNTIME="$ROOT_DIR/tools/loom/product_exec_ingress_dark.runtime.v1"
CONTRACT="$ROOT_DIR/tools/loom/PRODUCT_EXEC_INGRESS_REQUIRED_MODE_DEFAULT_V1.md"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PRODUCT_DYNAMIC_USER_EXEC_ATTACHMENT_V1.md"

fail() {
  printf 'sounio-loom-product-exec-ingress-required-mode-default-selftest: FAIL: %s\n' \
    "$*" >&2
  exit 1
}

field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "${path#$ROOT_DIR/} field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

ocaml_toplevel_let() {
  local path="$1" name="$2" count prefix body
  count="$(grep -cE "^let ${name} " "$path" || true)"
  [[ "$count" == 1 ]] ||
    fail "toplevel let ${name} occurs ${count} times in ${path#$ROOT_DIR/}"
  prefix="let ${name} "
  body="$(awk -v prefix="$prefix" '
    index($0, prefix) == 1 { p = 1 }
    p && /^[^[:space:](]/ && index($0, prefix) != 1 { exit }
    p { print }
  ' "$path")"
  [[ -n "$body" ]] || fail "toplevel let ${name} extracted empty"
  printf '%s' "$body"
}

for path in "$INGRESS_SOURCE" "$INGRESS_RUNTIME" "$CONTRACT" "$GARDEN"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required parent is absent: ${path#$ROOT_DIR/}"
done

grep -Fq 'Make descriptor absence fail closed for product execution tools.' \
  "$GARDEN" || fail 'Garden gate 5 text drifted'

[[ "$(field "$INGRESS_RUNTIME" schema)" == loom-product-exec-ingress-dark-runtime-v1 ]] ||
  fail 'dark ExecIngress schema drifted'
[[ "$(field "$INGRESS_RUNTIME" required_mode_default)" == false ]] ||
  fail 'runtime required_mode_default flipped; this lane does not authorize gate 5'
[[ "$(field "$INGRESS_RUNTIME" descriptor_dark_attached)" == true ]] ||
  fail 'dark descriptor attachment drifted'
[[ "$(field "$INGRESS_RUNTIME" parity_open)" == false ]] || fail 'parity_open raised'
[[ "$(field "$INGRESS_RUNTIME" claim_ready)" == false ]] || fail 'claim_ready raised'

mode_slice="$(ocaml_toplevel_let "$INGRESS_SOURCE" required_mode)"
[[ "$mode_slice" == $'let required_mode () =\n'* ]] ||
  fail 'required_mode let shape drifted'
[[ "$mode_slice" == *'Sys.getenv_opt "SOUNIO_LOOM_EXEC_INGRESS_REQUIRED"'* ]] ||
  fail 'required_mode no longer reads SOUNIO_LOOM_EXEC_INGRESS_REQUIRED'
[[ "$mode_slice" == *'| None | Some "0" -> false'* ]] ||
  fail 'required_mode unset/0 default is no longer false'
[[ "$mode_slice" == *'| Some "1" -> true'* ]] ||
  fail 'required_mode no longer has an explicit fail-closed branch'

grep -Fq 'if required_mode () then failf "product-exec-ingress-descriptor-absent"' \
  "$INGRESS_SOURCE" ||
  fail 'descriptor-absent fail-closed path is no longer gated on required_mode'

SABOTAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-required-mode-default.XXXXXX")"
cleanup() { rm -rf "$SABOTAGE_DIR"; }
trap cleanup EXIT
sed 's/| None | Some "0" -> false/| None | Some "0" -> true/' \
  "$INGRESS_SOURCE" > "$SABOTAGE_DIR/required-mode-sabotage.ml"
mode_sabotage="$(ocaml_toplevel_let "$SABOTAGE_DIR/required-mode-sabotage.ml" required_mode)"
[[ "$mode_sabotage" != *'| None | Some "0" -> false'* ]] ||
  fail 'required_mode default sabotage did not flip unset/0 to true'
[[ "$mode_sabotage" == *'| None | Some "0" -> true'* ]] ||
  fail 'required_mode default sabotage did not produce the true default'

printf '%s\n' \
  'sounio-loom-product-exec-ingress-required-mode-default-selftest: PASS semantic_authority=Sounio producer=Bash role=RATCHET_ONLY action=9031 probe=named-let-required_mode source=tools/loom/src/loom_exec_ingress.ml env=SOUNIO_LOOM_EXEC_INGRESS_REQUIRED unset_or_zero=false one=true descriptor_absent_fail_closed_gated=true required_mode_default=false gate5_closed=false production_activation=false exec_attached=false parity_open=false claim_ready=false python_executed=false rust_executed=false causal_sabotage=PASS next=do-not-flip-required-mode-until-gate-4-and-activation-garden'
