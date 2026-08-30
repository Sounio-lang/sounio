#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/exec-intent-projection.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT
MANIFEST="$ROOT_DIR/tools/loom/exec_intent_envelope.freeze.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-exec-intent-projection-v1-20260830.txt"

fail() {
  printf 'sounio-loom-exec-intent-projection-selftest: FAIL: %s test_root=%s\n' \
    "$*" "$TEST_ROOT" >&2
  exit 1
}

manifest_value() {
  local key="$1" line count
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

manifest_sha256="$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
[[ "$manifest_sha256" == \
   8a95e587ccc81c16da17d56b9649d04bc9c3e764d66fc938c195d95568e7608e ]] ||
  fail 'frozen Sounio manifest hash drifted'

dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
RUNTIME="$TEST_ROOT/sounio-loom-exec-intent-envelope"
SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_intent_envelope_fixture.sh" \
  >/dev/null
[[ "$(sha256sum "$RUNTIME" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] ||
  fail 'source-fresh Sounio 9034 runtime hash drifted'

RAW_ONE=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
RAW_TWO=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
COMMAND="$(manifest_value command_sha256)"
WRONG_COMMAND="1${COMMAND:1}"

project() {
  local raw_event="$1"
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_RUNTIME="$RUNTIME" \
    "$LOOM" exec-intent-probe --root "$ROOT_DIR" --mode project \
      --raw-event "$raw_event" --command "$COMMAND"
}

positive_one="$(project "$RAW_ONE")"
positive_two="$(project "$RAW_TWO")"
for item in "$positive_one" "$positive_two"; do
  [[ "$item" == LOOM_EXEC_INTENT_PROJECTION\ mode=project* && \
     "$item" == *'semantic_authority=Sounio action=9034 operational_kernel=OCaml'* && \
     "$item" == *"manifest_sha256=$manifest_sha256"* && \
     "$item" == *"event_sha256=$(manifest_value semantic_event_sha256)"* && \
     "$item" == *"command_sha256=$COMMAND"* && \
     "$item" == *'raw_event_is_semantic_identity=false'* && \
     "$item" == *'ocaml_projection_attached=true'* ]] ||
    fail "positive projection diverged: $item"
done
[[ "$positive_one" == *"raw_event_sha256=$RAW_ONE"* && \
   "$positive_two" == *"raw_event_sha256=$RAW_TWO"* ]] ||
  fail 'raw provider evidence was not retained separately'
event_one="$(sed -n 's/.* event_sha256=\([^ ]*\).*/\1/p' <<<"$positive_one")"
event_two="$(sed -n 's/.* event_sha256=\([^ ]*\).*/\1/p' <<<"$positive_two")"
[[ "$event_one" == "$event_two" && "$event_one" != "$RAW_ONE" && \
   "$event_two" != "$RAW_TWO" ]] ||
  fail 'raw provider bytes still influence semantic identity'

control="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_RUNTIME="$RUNTIME" \
  "$LOOM" exec-intent-probe --root "$ROOT_DIR" --mode command-mismatch \
    --raw-event "$RAW_ONE" --command "$COMMAND")"
[[ "$control" == *'decision=SOUNIO_EXEC_INTENT_ENVELOPE DENY555'* && \
   "$control" == *'control_refused=true material_mutation=false'* ]] ||
  fail "Sounio command-mismatch control diverged: $control"

set +e
wrong_refusal="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_RUNTIME="$RUNTIME" \
  "$LOOM" exec-intent-probe --root "$ROOT_DIR" --mode project \
    --raw-event "$RAW_ONE" --command "$WRONG_COMMAND" 2>&1)"
wrong_code=$?
set -e
[[ $wrong_code -eq 1 && \
   "$wrong_refusal" == *'exec-intent-command-mismatch-denied:'* && \
   "$wrong_refusal" == *'DENY555'* ]] ||
  fail "wrong command did not fail through Sounio DENY555: $wrong_refusal"

cp "$MANIFEST" "$TEST_ROOT/tampered.freeze.v1"
printf 'tamper=true\n' >>"$TEST_ROOT/tampered.freeze.v1"
set +e
manifest_refusal="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_MANIFEST="$TEST_ROOT/tampered.freeze.v1" \
  SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_RUNTIME="$RUNTIME" \
  "$LOOM" exec-intent-probe --root "$ROOT_DIR" --mode project \
    --raw-event "$RAW_ONE" --command "$COMMAND" 2>&1)"
manifest_code=$?
set -e
[[ $manifest_code -eq 1 && \
   "$manifest_refusal" == *'exec-intent-manifest-hash-mismatch'* ]] ||
  fail 'tampered Sounio manifest did not fail closed before projection'

cp "$RUNTIME" "$TEST_ROOT/tampered-runtime"
printf X >>"$TEST_ROOT/tampered-runtime"
set +e
runtime_refusal="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_RUNTIME="$TEST_ROOT/tampered-runtime" \
  "$LOOM" exec-intent-probe --root "$ROOT_DIR" --mode project \
    --raw-event "$RAW_ONE" --command "$COMMAND" 2>&1)"
runtime_code=$?
set -e
[[ $runtime_code -eq 1 && \
   "$runtime_refusal" == *'exec-intent-runtime-hash-mismatch'* ]] ||
  fail 'tampered Sounio runtime did not fail closed before execution'

mkdir "$TEST_ROOT/empty-root"
set +e
missing_refusal="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_RUNTIME="$RUNTIME" \
  "$LOOM" exec-intent-probe --root "$TEST_ROOT/empty-root" --mode project \
    --raw-event "$RAW_ONE" --command "$COMMAND" 2>&1)"
missing_code=$?
set -e
[[ $missing_code -eq 1 && \
   "$missing_refusal" == *'exec-intent-file-missing:'* ]] ||
  fail 'missing frozen projection policy did not fail closed'

sed 's/if command_sha256 <> policy.command_sha256 then (/if false then (/' \
  "$ROOT_DIR/tools/loom/src/loom_exec_intent.ml" >"$TEST_ROOT/loom_exec_intent.ml"
grep -Fq 'if false then (' "$TEST_ROOT/loom_exec_intent.ml" ||
  fail 'causal mutation did not delete the OCaml command equality rule'
cat >"$TEST_ROOT/mutant_main.ml" <<'EOF'
let () =
  let projection =
    Loom_exec_intent.project ~root:Sys.argv.(1)
      ~raw_event_sha256:Sys.argv.(2) ~command_sha256:Sys.argv.(3)
  in
  Printf.printf "MUTANT_ADMITTED projected_command=%s raw_event=%s\n"
    projection.command_sha256 projection.raw_event_sha256
EOF
(
  cd "$TEST_ROOT"
  ocamlfind ocamlopt -package unix,cryptokit -linkpkg loom_exec_intent.ml \
    mutant_main.ml -o mutant-projector
)
mutant="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_RUNTIME="$RUNTIME" \
  "$TEST_ROOT/mutant-projector" "$ROOT_DIR" "$RAW_ONE" "$WRONG_COMMAND")"
[[ "$mutant" == MUTANT_ADMITTED* && "$mutant" == *"raw_event=$RAW_ONE"* ]] ||
  fail "rule-deletion mutant did not admit the unchanged wrong command: $mutant"

oracle_sentinel="$TEST_ROOT/prohibited-oracle-executed"
mkdir "$TEST_ROOT/oracles"
for name in python python3 rustc cargo; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$oracle_sentinel" \
    >"$TEST_ROOT/oracles/$name"
  chmod 0755 "$TEST_ROOT/oracles/$name"
done
PATH="$TEST_ROOT/oracles:$PATH" project "$RAW_ONE" >/dev/null
[[ ! -e "$oracle_sentinel" ]] || fail 'a prohibited Python or Rust oracle executed'
dependencies="$(ldd "$LOOM" 2>&1 || true; ldd "$RUNTIME" 2>&1 || true)"
printf '%s\n' "$dependencies" | grep -Eqi 'python|rust' &&
  fail 'a runtime has a prohibited Python or Rust dependency'

result="$(printf 'sounio-loom-exec-intent-projection-selftest: PASS semantic_authority=Sounio action=9034 stage=SEMANTICS_FROZEN operational_kernel=OCaml operational_role=OPERATIONAL_PROJECTION manifest_sha256=%s source_sha256=%s executable_sha256=%s operational_source_sha256=%s raw_event_separate=true raw_event_is_semantic_identity=false command_mismatch=DENY555 manifest_tamper=REFUSED runtime_tamper=REFUSED missing_policy=REFUSED causal_rule=command_sha256_equal causal_mutant=ADMITTED python_executed=false rust_executed=false ocaml_projection_attached=true provider_lifecycle_attached=false arbitrary_command_projection=false exec_attached=false production_activation=false parity_open=false claim_ready=false' \
  "$manifest_sha256" "$(manifest_value source_sha256)" \
  "$(manifest_value executable_sha256)" \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_exec_intent.ml" | cut -d ' ' -f 1)")"
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] ||
  fail 'exact projection evidence is absent or linked'
[[ "$(cat "$EVIDENCE")" == "$result" ]] ||
  fail 'exact projection evidence drifted'
printf '%s\n' "$result"
