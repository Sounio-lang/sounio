#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E1_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi
SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
KEEP_DIR="${E1_RECEIPT_DIR:-}"
TMP_DIR="$(mktemp -d /tmp/madaros-e1-enir-shadow.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

DRIVER="$TMP_DIR/madaros-enir"
ARTIFACT_A="$TMP_DIR/a.enir"
ARTIFACT_B="$TMP_DIR/b.enir"
NATIVE_RECEIPT="$TMP_DIR/native.receipt"
RECEIPT="$TMP_DIR/receipt.json"

fail() {
  echo "E1_ENIR_SHADOW_GATE_FAIL: $*" >&2
  exit 1
}

current_branch_name() {
  git symbolic-ref --quiet --short HEAD 2>/dev/null || git rev-parse --abbrev-ref HEAD 2>/dev/null || true
}

e1_lane_enabled() {
  [[ "${E1_LANE:-0}" == "1" ]] && return 0
  local branch
  branch="$(current_branch_name)"
  case "$branch" in
    *enir*|*e1-shadow*) return 0 ;;
  esac
  return 1
}

protected_surface_drift_paths() {
  git diff --name-only "$BASE_REF" HEAD -- \
    self-hosted/compiler/main.sio \
    self-hosted/ir \
    self-hosted/native \
    self-hosted/wasm \
    self-hosted/gpu \
    stdlib/runtime
}

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"

# E1 is shadow-only: canonical compiler, IR, ABI, runtime, and codegen sources
# must be byte-identical to the selected canonical base for the E1 experiment.
# Outside the E1 lane, protected-surface drift is reported as an explicit skip:
# it is baseline drift for the E1 owner, not evidence that this PR broke E1.
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
DRIFT_PATHS="$(protected_surface_drift_paths)"
if [[ -n "$DRIFT_PATHS" ]]; then
  DRIFT_LIST="${DRIFT_PATHS//$'\n'/,}"
  if e1_lane_enabled; then
    fail "E1 changed a lowering/codegen/ABI/runtime surface: $DRIFT_LIST"
  fi
  echo "E1_ENIR_SHADOW_GATE_SKIP status=skip reason=protected_surface_drift_outside_e1_lane base_ref=$BASE_REF drift_paths=$DRIFT_LIST"
  exit 0
fi

scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >/dev/null
chmod +x "$DRIVER"

"$DRIVER" emit >"$ARTIFACT_A"
"$DRIVER" emit >"$ARTIFACT_B"
cmp -s "$ARTIFACT_A" "$ARTIFACT_B" || fail "repeated native emission changed bytes"
"$DRIVER" roundtrip "$ARTIFACT_A" >"$TMP_DIR/roundtrip.enir"
cmp -s "$ARTIFACT_A" "$TMP_DIR/roundtrip.enir" || fail "native roundtrip changed bytes"
"$DRIVER" verify "$ARTIFACT_A" >"$NATIVE_RECEIPT"

python3 scripts/dev/madaros_v2_e1_enir_shadow_verify.py \
  --artifact "$ARTIFACT_A" \
  --corpus tools/eisa/eisa_evm_run.sio \
  --native-receipt "$NATIVE_RECEIPT" \
  --receipt "$RECEIPT"

python3 - "$ARTIFACT_A" "$TMP_DIR" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="ascii")
out = Path(sys.argv[2])

def write(name: str, text: str) -> None:
    (out / name).write_text(text, encoding="ascii")

write("bad_unknown_tag.enir", source.replace("type|0|", "bogus|0|", 1))
write("bad_duplicate_op.enir", source.replace("op|1|0|", "op|0|0|", 1))
write("bad_type_ref.enir", source.replace("value|9|2|", "value|9|9|", 1))
write("bad_fp_class.enir", source.replace("value|0|0|1|4607182418800017408|3|", "value|0|0|1|4607182418800017408|0|", 1))
write("bad_branch_uncertainty.enir", source.replace("type|2|4|2|1|", "type|2|4|2|2|", 1))
write("bad_branch_target.enir", source.replace("op|11|10|-1|-1|-1|-1|14|", "op|11|10|-1|-1|-1|-1|99|", 1))
write("bad_duplicate_obs.enir", source.replace("obs|1|golden_add|", "obs|0|golden_add|", 1))
write("bad_footer_count.enir", source.replace("end|3|10|10|1|16|39|30", "end|3|10|10|1|16|38|30"))
write("bad_noncanonical_i64.enir", source.replace("type|0|4|", "type|00|4|", 1))
(out / "bad_final_newline.enir").write_bytes(source.encode("ascii")[:-1])
write("valid_numeric_tamper.enir", source.replace("4607182418800017408", "4607182418800017409", 1))
write("valid_i64_extremes.enir", source.replace(
    "value|1|2|1|4611686018427387904|3|0|0|0|0|0|0|-1|1",
    "value|1|2|1|4611686018427387904|3|-9223372036854775808|9223372036854775807|0|0|0|0|-1|1",
    1,
))
write("valid_manifest_tamper.enir", source.replace("obs|0|golden_mul|", "obs|0|golden_mul_alt|", 1))
PY

for bad in \
  bad_unknown_tag bad_duplicate_op bad_type_ref bad_fp_class bad_branch_uncertainty \
  bad_branch_target bad_duplicate_obs bad_footer_count bad_noncanonical_i64 bad_final_newline; do
  if "$DRIVER" verify "$TMP_DIR/$bad.enir" >"$TMP_DIR/$bad.native.log" 2>&1; then
    fail "native verifier accepted invalid mutation: $bad"
  fi
  if python3 scripts/dev/madaros_v2_e1_enir_shadow_verify.py \
      --artifact "$TMP_DIR/$bad.enir" --corpus tools/eisa/eisa_evm_run.sio \
      >"$TMP_DIR/$bad.independent.log" 2>&1; then
    fail "independent verifier accepted invalid mutation: $bad"
  fi
done

"$DRIVER" verify "$TMP_DIR/valid_numeric_tamper.enir" >"$TMP_DIR/tamper.native"
python3 scripts/dev/madaros_v2_e1_enir_shadow_verify.py \
  --artifact "$TMP_DIR/valid_numeric_tamper.enir" \
  --corpus tools/eisa/eisa_evm_run.sio \
  --native-receipt "$TMP_DIR/tamper.native" \
  --receipt "$TMP_DIR/tamper.receipt.json" >/dev/null
python3 - "$RECEIPT" "$TMP_DIR/tamper.receipt.json" <<'PY'
import json, sys
base = json.load(open(sys.argv[1], encoding="utf-8"))
tamper = json.load(open(sys.argv[2], encoding="utf-8"))
if base["canonical_sha256"] == tamper["canonical_sha256"] or base["canonical_l64"] == tamper["canonical_l64"]:
    raise SystemExit("valid numeric tamper did not change both hashes")
PY

"$DRIVER" verify "$TMP_DIR/valid_i64_extremes.enir" >"$TMP_DIR/extremes.native"
python3 scripts/dev/madaros_v2_e1_enir_shadow_verify.py \
  --artifact "$TMP_DIR/valid_i64_extremes.enir" \
  --corpus tools/eisa/eisa_evm_run.sio \
  --native-receipt "$TMP_DIR/extremes.native" >/dev/null

"$DRIVER" verify "$TMP_DIR/valid_manifest_tamper.enir" >/dev/null
if python3 scripts/dev/madaros_v2_e1_enir_shadow_verify.py \
    --artifact "$TMP_DIR/valid_manifest_tamper.enir" \
    --corpus tools/eisa/eisa_evm_run.sio >/dev/null 2>&1; then
  fail "source-derived corpus checker accepted manifest tamper"
fi

cp tools/eisa/eisa_evm_run.sio "$TMP_DIR/corpus_tamper.sio"
python3 - "$TMP_DIR/corpus_tamper.sio" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
s = s.replace("    if run_img(v2_mem_poison_img()) != 0 { fails = fails + 1 }\n", "", 1)
p.write_text(s, encoding="utf-8")
PY
if python3 scripts/dev/madaros_v2_e1_enir_shadow_verify.py \
    --artifact "$ARTIFACT_A" --corpus "$TMP_DIR/corpus_tamper.sio" >/dev/null 2>&1; then
  fail "checker accepted a corpus with only 29 program calls"
fi

if [[ -n "$KEEP_DIR" ]]; then
  mkdir -p "$KEEP_DIR"
  cp "$ARTIFACT_A" "$KEEP_DIR/enir-shadow.canonical"
  cp "$NATIVE_RECEIPT" "$KEEP_DIR/enir-shadow.native.receipt"
  cp "$RECEIPT" "$KEEP_DIR/enir-shadow.receipt.json"
fi

SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["canonical_sha256"])' "$RECEIPT")"
L64="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["canonical_l64"])' "$RECEIPT")"
echo "E1_ENIR_SHADOW_GATE_PASS sha256=$SHA256 l64=$L64 invalid_mutations=10 valid_hash_tamper=1 i64_extremes=pass manifest_tamper=1 programs=30 observations=39 codegen_diff=0"
