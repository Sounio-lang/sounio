#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_SHA="571f3bfce00324202abd98bf62f7c7ced4f0340e"
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${C2_V0_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-c2-v0.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
SOURCE="$ROOT_DIR/tools/eisa/eisa_enir_c2_rump.eisa"
PINNED_SOURCE="$TMP_DIR/eisa_enir_c2_rump.eisa"
COMPARE="$ROOT_DIR/scripts/dev/madaros_v2_c2_v0_compare.py"
BUILD_LOCK="$ROOT_DIR/scripts/dev/souc-build-lock.sh"
ENVIRONMENT="$(uname -srm)"

fail() {
  echo "C2_V0_FIRST_DIVERGENCE_GATE_FAIL: $*" >&2
  exit 1
}

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
[[ -f "$SOURCE" ]] || fail "missing C2 source: $SOURCE"
[[ -f "$COMPARE" ]] || fail "missing C2 comparator: $COMPARE"
[[ -x "$BUILD_LOCK" ]] || fail "missing executable build lock: $BUILD_LOCK"
[[ "$(git rev-parse "$BASE_SHA")" == "$BASE_SHA" ]] || fail "required canon base is unavailable"
git merge-base --is-ancestor "$BASE_SHA" HEAD || fail "required canon base is not an ancestor of HEAD"

path_allowed() {
  case "$1" in
    tools/eisa/eisa_enir_c2_rump.eisa|scripts/dev/madaros_v2_c2_v0_compare.py|scripts/dev/madaros_v2_c2_v0_gate.sh|.claude/llm_offload_log.md) return 0 ;;
    *) return 1 ;;
  esac
}

if path_allowed self-hosted/enir/driver.sio; then
  fail "write-set policy negative accepted a compiler path"
fi

while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  path_allowed "$path" || fail "committed BASE..HEAD path is outside the C2 write-set: $path"
done < <(git diff --name-only "$BASE_SHA"..HEAD)

while IFS= read -r entry; do
  path="${entry:3}"
  case "$path" in
    *" -> "*) path="${path##* -> }" ;;
  esac
  case "$path" in
    tools/eisa/eisa_enir_c2_rump.eisa|scripts/dev/madaros_v2_c2_v0_compare.py|scripts/dev/madaros_v2_c2_v0_gate.sh|.claude/llm_offload_log.md) ;;
    *) fail "worktree contains a change outside the C2 write-set: $path" ;;
  esac
done < <(git status --porcelain)

git diff --quiet "$BASE_SHA" -- \
  self-hosted stdlib docs scripts/ci .github \
  || fail "C2 witness changed a read-only compiler, stdlib, governance, or CI surface"

cp "$SOURCE" "$PINNED_SOURCE"
SOURCE_SHA="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
[[ "$(sha256sum "$PINNED_SOURCE" | cut -d' ' -f1)" == "$SOURCE_SHA" ]] \
  || fail "pinned C2 source does not match the declared fixture"
SEED_SHA="$(sha256sum "$SEED" | cut -d' ' -f1)"
BUILD_LOCK_SHA="$(sha256sum "$BUILD_LOCK" | cut -d' ' -f1)"
COMPARE_SHA="$(sha256sum "$COMPARE" | cut -d' ' -f1)"

"$BUILD_LOCK" "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "source-fresh ENIR driver build produced no ELF"
chmod +x "$DRIVER"

"$DRIVER" lower-v1 "$PINNED_SOURCE" >"$TMP_DIR/run-a.enir"
"$DRIVER" lower-v2 "$PINNED_SOURCE" >"$TMP_DIR/run-b.enir"
"$DRIVER" run "$TMP_DIR/run-a.enir" >"$TMP_DIR/run-a.trace"
"$DRIVER" run "$TMP_DIR/run-b.enir" >"$TMP_DIR/run-b.trace"
[[ "$(sha256sum "$SEED" | cut -d' ' -f1)" == "$SEED_SHA" ]] || fail "Stage0 seed changed during evidence generation"
[[ "$(sha256sum "$BUILD_LOCK" | cut -d' ' -f1)" == "$BUILD_LOCK_SHA" ]] || fail "build-lock script changed during evidence generation"
[[ "$(sha256sum "$COMPARE" | cut -d' ' -f1)" == "$COMPARE_SHA" ]] || fail "comparator changed during evidence generation"

DRIVER_SHA="$(sha256sum "$DRIVER" | cut -d' ' -f1)"
A_ARTIFACT_SHA="$(sha256sum "$TMP_DIR/run-a.enir" | cut -d' ' -f1)"
B_ARTIFACT_SHA="$(sha256sum "$TMP_DIR/run-b.enir" | cut -d' ' -f1)"
A_TRACE_SHA="$(sha256sum "$TMP_DIR/run-a.trace" | cut -d' ' -f1)"
B_TRACE_SHA="$(sha256sum "$TMP_DIR/run-b.trace" | cut -d' ' -f1)"

common_args=(
  --source-a "$PINNED_SOURCE" --source-b "$PINNED_SOURCE"
  --compiler-a "$DRIVER" --compiler-b "$DRIVER"
  --artifact-a "$TMP_DIR/run-a.enir" --artifact-b "$TMP_DIR/run-b.enir"
  --trace-a "$TMP_DIR/run-a.trace" --trace-b "$TMP_DIR/run-b.trace"
  --seed "$SEED" --build-lock "$BUILD_LOCK" --comparator "$COMPARE"
  --requested-a eisa_v1+dd64_expansion --requested-b eisa_v2+qd128_expansion
  --compiler-revision-a "$BASE_SHA" --compiler-revision-b "$BASE_SHA"
  --environment-a "$ENVIRONMENT" --environment-b "$ENVIRONMENT"
)
integrity_args=(
  --expect "source_a=$SOURCE_SHA" --expect "source_b=$SOURCE_SHA"
  --expect "compiler_a=$DRIVER_SHA" --expect "compiler_b=$DRIVER_SHA"
  --expect "artifact_a=$A_ARTIFACT_SHA" --expect "artifact_b=$B_ARTIFACT_SHA"
  --expect "trace_a=$A_TRACE_SHA" --expect "trace_b=$B_TRACE_SHA"
  --expect "seed=$SEED_SHA" --expect "build_lock=$BUILD_LOCK_SHA"
  --expect "comparator=$COMPARE_SHA"
)

python3 "$COMPARE" "${common_args[@]}" "${integrity_args[@]}" \
  --projection full-epistemic --receipt "$TMP_DIR/diverged.json"
python3 "$COMPARE" "${common_args[@]}" "${integrity_args[@]}" \
  --projection value-bits-only --receipt "$TMP_DIR/equivalent-value-only.json"

sed 's/let five5=5\.5/let five5=5.25/' "$PINNED_SOURCE" >"$TMP_DIR/source-other.eisa"
OTHER_SOURCE_SHA="$(sha256sum "$TMP_DIR/source-other.eisa" | cut -d' ' -f1)"
source_other_integrity_args=(
  --expect "source_a=$SOURCE_SHA" --expect "source_b=$OTHER_SOURCE_SHA"
  --expect "compiler_a=$DRIVER_SHA" --expect "compiler_b=$DRIVER_SHA"
  --expect "artifact_a=$A_ARTIFACT_SHA" --expect "artifact_b=$B_ARTIFACT_SHA"
  --expect "trace_a=$A_TRACE_SHA" --expect "trace_b=$B_TRACE_SHA"
  --expect "seed=$SEED_SHA" --expect "build_lock=$BUILD_LOCK_SHA"
  --expect "comparator=$COMPARE_SHA"
)
python3 "$COMPARE" "${common_args[@]}" \
  --source-b "$TMP_DIR/source-other.eisa" \
  "${source_other_integrity_args[@]}" \
  --receipt "$TMP_DIR/incomparable-source.json"

python3 "$COMPARE" "${common_args[@]}" "${integrity_args[@]}" \
  --environment-b "${ENVIRONMENT}-controlled-negative" \
  --receipt "$TMP_DIR/incomparable-environment.json"

sed '0,/|site=27|/s//|site=270|/' "$TMP_DIR/run-b.trace" >"$TMP_DIR/unaligned.trace"
UNALIGNED_TRACE_SHA="$(sha256sum "$TMP_DIR/unaligned.trace" | cut -d' ' -f1)"
unaligned_integrity_args=(
  --expect "source_a=$SOURCE_SHA" --expect "source_b=$SOURCE_SHA"
  --expect "compiler_a=$DRIVER_SHA" --expect "compiler_b=$DRIVER_SHA"
  --expect "artifact_a=$A_ARTIFACT_SHA" --expect "artifact_b=$B_ARTIFACT_SHA"
  --expect "trace_a=$A_TRACE_SHA" --expect "trace_b=$UNALIGNED_TRACE_SHA"
  --expect "seed=$SEED_SHA" --expect "build_lock=$BUILD_LOCK_SHA"
  --expect "comparator=$COMPARE_SHA"
)
python3 "$COMPARE" "${common_args[@]}" \
  --trace-b "$TMP_DIR/unaligned.trace" \
  "${unaligned_integrity_args[@]}" \
  --receipt "$TMP_DIR/unaligned.json"

python3 "$COMPARE" "${common_args[@]}" "${integrity_args[@]}" --run-b-status 1 \
  --receipt "$TMP_DIR/blocked.json"

expectation_must_fail() {
  local label="$1"
  shift
  set +e
  python3 "$COMPARE" "${common_args[@]}" "$@" --receipt "$TMP_DIR/expectation-$label.json"
  local rc=$?
  set -e
  [[ "$rc" -eq 2 ]] || fail "$label expectation negative returned rc=$rc; expected rc=2"
}

expectation_must_fail zero
expectation_must_fail partial --expect "source_a=$SOURCE_SHA"

grep -v '^enir-exec|.*|ordinal=2|' "$TMP_DIR/run-a.trace" \
  | sed 's/|observations=3|/|observations=2|/' >"$TMP_DIR/truncated-a.trace"
grep -v '^enir-exec|.*|ordinal=2|' "$TMP_DIR/run-b.trace" \
  | sed 's/|observations=3|/|observations=2|/' >"$TMP_DIR/truncated-b.trace"
TRUNCATED_A_SHA="$(sha256sum "$TMP_DIR/truncated-a.trace" | cut -d' ' -f1)"
TRUNCATED_B_SHA="$(sha256sum "$TMP_DIR/truncated-b.trace" | cut -d' ' -f1)"
truncated_integrity_args=(
  --expect "source_a=$SOURCE_SHA" --expect "source_b=$SOURCE_SHA"
  --expect "compiler_a=$DRIVER_SHA" --expect "compiler_b=$DRIVER_SHA"
  --expect "artifact_a=$A_ARTIFACT_SHA" --expect "artifact_b=$B_ARTIFACT_SHA"
  --expect "trace_a=$TRUNCATED_A_SHA" --expect "trace_b=$TRUNCATED_B_SHA"
  --expect "seed=$SEED_SHA" --expect "build_lock=$BUILD_LOCK_SHA"
  --expect "comparator=$COMPARE_SHA"
)
python3 "$COMPARE" "${common_args[@]}" \
  --trace-a "$TMP_DIR/truncated-a.trace" --trace-b "$TMP_DIR/truncated-b.trace" \
  "${truncated_integrity_args[@]}" --receipt "$TMP_DIR/paired-truncation.json"

tamper_must_fail() {
  local label="$1"
  shift
  set +e
  python3 "$COMPARE" "${common_args[@]}" "${integrity_args[@]}" "$@" \
    --receipt "$TMP_DIR/tamper-$label.json"
  local rc=$?
  set -e
  [[ "$rc" -eq 2 ]] || fail "$label tamper returned rc=$rc; expected integrity rc=2"
}

sed -E '0,/\|value_bits=[^|]*/s//|value_bits=1/' "$TMP_DIR/run-b.trace" >"$TMP_DIR/tamper-value.trace"
tamper_must_fail value --trace-b "$TMP_DIR/tamper-value.trace"

sed '0,/|site=27|/s//|site=270|/' "$TMP_DIR/run-b.trace" >"$TMP_DIR/tamper-site.trace"
tamper_must_fail site --trace-b "$TMP_DIR/tamper-site.trace"

sed 's/resource|64/resource|65/' "$TMP_DIR/run-b.enir" >"$TMP_DIR/tamper-artifact.enir"
tamper_must_fail artifact --artifact-b "$TMP_DIR/tamper-artifact.enir"

tamper_must_fail source --source-b "$TMP_DIR/source-other.eisa"

python3 - "$TMP_DIR" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def load(name):
    return json.loads((root / name).read_text(encoding="ascii"))

positive = load("diverged.json")
assert positive["comparison_status"] == "DIVERGED"
assert positive["alignment_status"] == "ALIGNED"
assert positive["integrity_status"]["status"] == "VERIFIED"
identity = positive["first_divergence"]["operation_identity"]
assert identity == {"ordinal": 1, "site": 27, "source_span": 25, "value_id": 22}
assert positive["first_divergence"]["differing_field"] == "correction1_bits"
assert positive["first_divergence"]["run_a"]["correction2_bits"] == "not_applicable"
assert positive["first_divergence"]["run_a"]["correction3_bits"] == "not_applicable"

equivalent = load("equivalent-value-only.json")
assert equivalent["comparison_status"] == "OBSERVED_EQUIVALENT"
assert "dd64 and qd128 expansion correction limbs" in equivalent["blind_spots"]
assert load("incomparable-source.json")["comparison_status"] == "INCOMPARABLE"
assert load("incomparable-environment.json")["comparison_status"] == "INCOMPARABLE"
assert load("unaligned.json")["comparison_status"] == "UNALIGNED"
assert load("blocked.json")["comparison_status"] == "BLOCKED"
paired_truncation = load("paired-truncation.json")
assert paired_truncation["comparison_status"] == "UNALIGNED"
assert paired_truncation["integrity_status"]["status"] == "VERIFIED"

for label in ("zero", "partial"):
    incomplete = load(f"expectation-{label}.json")
    assert incomplete["comparison_status"] == "BLOCKED"
    assert incomplete["integrity_status"]["status"] == "FAILED"
    assert incomplete["integrity_status"]["missing_expectations"]

for label in ("value", "site", "artifact", "source"):
    tampered = load(f"tamper-{label}.json")
    assert tampered["integrity_status"]["status"] == "FAILED"
    assert tampered["comparison_status"] == "BLOCKED"
PY

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$TMP_DIR"/*.json "$KEEP_DIR/"
  cp "$TMP_DIR"/run-a.enir "$TMP_DIR"/run-b.enir "$KEEP_DIR/"
  cp "$TMP_DIR"/run-a.trace "$TMP_DIR"/run-b.trace "$KEEP_DIR/"
fi

RECEIPT_SHA="$(sha256sum "$TMP_DIR/diverged.json" | cut -d' ' -f1)"
echo "C2_V0_FIRST_DIVERGENCE_GATE_PASS base=$BASE_SHA status=DIVERGED ordinal=1 site=27 source_span=25 field=correction1_bits value_only=OBSERVED_EQUIVALENT source=INCOMPARABLE environment=INCOMPARABLE identity=UNALIGNED paired_truncation=UNALIGNED failed_run=BLOCKED expectation_coverage=2/2 tampers=4/4 integrity=VERIFIED receipt_sha256=$RECEIPT_SHA"
