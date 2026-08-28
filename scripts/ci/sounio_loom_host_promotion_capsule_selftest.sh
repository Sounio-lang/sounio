#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_host_promotion_capsule.sh"
PROMOTER="$ROOT_DIR/scripts/dev/promote_loom_host_capsule.sh"

fail() {
  printf 'sounio-loom-host-promotion-capsule-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

run_refusal() {
  local label="$1"
  shift
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ $status -eq 70 ]] || fail "$label exited $status instead of fail-closed 70: $output"
  printf '%s\n' "$output"
}

repack() {
  local source_parent="$1"
  local output="$2"
  tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
    --format=posix --pax-option=delete=atime,delete=ctime \
    -C "$source_parent" -cf "$output" capsule-v1
}

replace_record() {
  local path="$1" key="$2" value="$3"
  local temporary="$path.next"
  local line name seen=0
  : > "$temporary"
  while IFS= read -r line || [[ -n "$line" ]]; do
    name="${line%%=*}"
    if [[ "$name" == "$key" ]]; then
      [[ $seen -eq 0 ]] || fail "duplicate record while replacing: $key"
      printf '%s=%s\n' "$key" "$value" >> "$temporary"
      seen=1
    else
      printf '%s\n' "$line" >> "$temporary"
    fi
  done < "$path"
  [[ $seen -eq 1 ]] || fail "record field is absent while replacing: $key"
  mv "$temporary" "$path"
}

[[ -x "$BUILDER" && -x "$PROMOTER" ]] || fail 'capsule builder or promoter is unavailable'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-host-promotion-selftest.XXXXXX")"
cleanup() {
  find "$WORK" -type d -exec chmod u+rwx {} + 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

CAPSULE_ONE="$WORK/capsule-one.tar"
CAPSULE_TWO="$WORK/capsule-two.tar"
build_one="$(SOUNIO_LOOM_ALLOW_DIRTY_CAPSULE=1 "$BUILDER" --output "$CAPSULE_ONE")"
build_two="$(SOUNIO_LOOM_ALLOW_DIRTY_CAPSULE=1 "$BUILDER" --output "$CAPSULE_TWO")"
[[ "$build_one" == 'LOOM_HOST_PROMOTION_CAPSULE_BUILD PASS '* ]] || fail "first build failed: $build_one"
[[ "$build_two" == 'LOOM_HOST_PROMOTION_CAPSULE_BUILD PASS '* ]] || fail "second build failed: $build_two"
CAPSULE_SHA256="$(sha256_file "$CAPSULE_ONE")"
[[ "$CAPSULE_SHA256" == "$(sha256_file "$CAPSULE_TWO")" ]] || fail 'source-identical capsule builds were not deterministic'
cmp -s "$CAPSULE_ONE" "$CAPSULE_TWO" || fail 'source-identical capsule bytes differed'

verify_output="$($PROMOTER --archive "$CAPSULE_ONE" --expected-sha256 "$CAPSULE_SHA256" --mode verify)"
[[ "$verify_output" == 'LOOM_HOST_PROMOTION_CAPSULE_VERIFY PASS '* ]] || fail "clean verification failed: $verify_output"
[[ "$verify_output" == *'source_tree_state=DIRTY_UNPROMOTABLE'* ]] || fail 'development capsule hid its dirty-source boundary'

OUTER_TAMPER="$WORK/outer-tamper.tar"
cp "$CAPSULE_ONE" "$OUTER_TAMPER"
printf X | dd of="$OUTER_TAMPER" bs=1 seek=1024 conv=notrunc status=none
outer_refusal="$(run_refusal outer-hash "$PROMOTER" --archive "$OUTER_TAMPER" --expected-sha256 "$CAPSULE_SHA256" --mode verify)"
[[ "$outer_refusal" == *'capsule archive hash drifted'* ]] || fail 'outer-byte sabotage did not trigger the archive-hash rule'

PAYLOAD_TREE="$WORK/payload-tree"
mkdir "$PAYLOAD_TREE"
tar --same-permissions -xf "$CAPSULE_ONE" -C "$PAYLOAD_TREE"
BROKER="$(find "$PAYLOAD_TREE/capsule-v1/rootfs/usr/lib/sounio/loom/releases" -mindepth 2 -maxdepth 2 -name loom-kernel-principal-broker -type f)"
[[ -n "$BROKER" ]] || fail 'test capsule omitted the broker payload'
chmod 0755 "$BROKER"
printf X >> "$BROKER"
PAYLOAD_TAMPER="$WORK/payload-tamper.tar"
repack "$PAYLOAD_TREE" "$PAYLOAD_TAMPER"
PAYLOAD_TAMPER_SHA256="$(sha256_file "$PAYLOAD_TAMPER")"
payload_refusal="$(run_refusal inner-payload-hash "$PROMOTER" --archive "$PAYLOAD_TAMPER" --expected-sha256 "$PAYLOAD_TAMPER_SHA256" --mode verify)"
[[ "$payload_refusal" == *'payload content hash drifted'* ]] || fail 'inner-byte sabotage did not trigger the payload-content rule'

TRAVERSAL_SOURCE="$WORK/traversal-source"
mkdir "$TRAVERSAL_SOURCE"
printf attack > "$TRAVERSAL_SOURCE/escape"
TRAVERSAL="$WORK/traversal.tar"
tar --transform='s#^escape$#capsule-v1/../escape#' -C "$TRAVERSAL_SOURCE" -cf "$TRAVERSAL" escape
TRAVERSAL_SHA256="$(sha256_file "$TRAVERSAL")"
traversal_refusal="$(run_refusal path-traversal "$PROMOTER" --archive "$TRAVERSAL" --expected-sha256 "$TRAVERSAL_SHA256" --mode verify)"
[[ "$traversal_refusal" == *'traverses a directory'* ]] || fail 'path-traversal sabotage did not trigger the archive-path rule'
[[ ! -e "$WORK/escape" ]] || fail 'path-traversal sabotage escaped the verification root'

PYTHON_TREE="$WORK/python-tree"
mkdir "$PYTHON_TREE"
tar --same-permissions -xf "$CAPSULE_ONE" -C "$PYTHON_TREE"
PYTHON_GATE="$PYTHON_TREE/capsule-v1/meta/sounio_loom_kernel_principal_broker_host_gate.sh"
PYTHON_MARKER="$WORK/python-oracle-executed"
chmod 0644 "$PYTHON_GATE"
cat > "$PYTHON_GATE" <<EOF
#!/usr/bin/env python3
from pathlib import Path
Path("$PYTHON_MARKER").write_text("executed")
EOF
chmod 0555 "$PYTHON_GATE"
PYTHON_MANIFEST="$PYTHON_TREE/capsule-v1/meta/capsule.manifest.v1"
replace_record "$PYTHON_MANIFEST" host_gate_sha256 "$(sha256_file "$PYTHON_GATE")"
PYTHON_CAPSULE="$WORK/python-oracle.tar"
repack "$PYTHON_TREE" "$PYTHON_CAPSULE"
PYTHON_CAPSULE_SHA256="$(sha256_file "$PYTHON_CAPSULE")"
python_refusal="$(run_refusal python-oracle "$PROMOTER" --archive "$PYTHON_CAPSULE" --expected-sha256 "$PYTHON_CAPSULE_SHA256" --mode verify)"
[[ "$python_refusal" == *'host gate language is not the mechanical Bash installation boundary'* ]] ||
  fail 'Python oracle did not trigger the pre-execution language rule'
[[ ! -e "$PYTHON_MARKER" ]] || fail 'Python oracle executed before refusal'

SABOTAGED_PROMOTER="$WORK/promoter-without-language-rule.sh"
LANGUAGE_RULE='[[ "$(head -n 1 "$HOST_GATE")" == '\''#!/usr/bin/env bash'\'' ]] || fail '\''host gate language is not the mechanical Bash installation boundary'\'''
grep -Fqx "$LANGUAGE_RULE" "$PROMOTER" || fail 'language rule is absent or changed'
grep -Fvx "$LANGUAGE_RULE" "$PROMOTER" > "$SABOTAGED_PROMOTER"
chmod 0555 "$SABOTAGED_PROMOTER"
install -m 0555 "$SABOTAGED_PROMOTER" "$PYTHON_TREE/capsule-v1/meta/promote_loom_host_capsule.sh"
replace_record "$PYTHON_MANIFEST" promoter_sha256 "$(sha256_file "$SABOTAGED_PROMOTER")"
PYTHON_SABOTAGED_CAPSULE="$WORK/python-oracle-sabotaged-rule.tar"
repack "$PYTHON_TREE" "$PYTHON_SABOTAGED_CAPSULE"
PYTHON_SABOTAGED_SHA256="$(sha256_file "$PYTHON_SABOTAGED_CAPSULE")"
sabotaged_output="$($SABOTAGED_PROMOTER --archive "$PYTHON_SABOTAGED_CAPSULE" --expected-sha256 "$PYTHON_SABOTAGED_SHA256" --mode verify)"
[[ "$sabotaged_output" == 'LOOM_HOST_PROMOTION_CAPSULE_VERIFY PASS '* ]] ||
  fail "removing only the language rule did not admit the unchanged Python witness: $sabotaged_output"
[[ ! -e "$PYTHON_MARKER" ]] || fail 'sabotage control unexpectedly executed the Python oracle'

preflight_refusal="$(run_refusal dirty-preflight "$PROMOTER" --archive "$CAPSULE_ONE" --expected-sha256 "$CAPSULE_SHA256" --mode preflight)"
[[ "$preflight_refusal" == *'dirty-source capsule cannot reach host preflight or promotion'* ]] ||
  fail 'dirty source capsule did not trigger the host-promotion rule'

printf 'sounio-loom-host-promotion-capsule-selftest: PASS semantic_producer=Sounio semantic_role=SEMANTIC_AUTHORITY transport_role=MECHANICAL_PACKAGING archive_sha256=%s deterministic=PASS outer_tamper=refused inner_payload_tamper=refused path_traversal=refused python_oracle=refused_pre_execution python_oracle_executed=false causal_sabotage=ALLOW dirty_preflight=refused resident_action_9030_attached=true decision_transport_material=true parity_open=false claim_ready=false launch=closed material_broker=false material_capsule=false material_invocation=false material_grant=false material_execution=false barrier_release=false\n' \
  "$CAPSULE_SHA256"
