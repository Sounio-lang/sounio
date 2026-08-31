#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${E3E_BASE_REF:-}"
if [[ -z "$BASE_REF" ]]; then
  BASE_REF="$(git merge-base HEAD "${ENIR_GATE_BASE:-origin/main}")"
fi

SEED="${SOUC_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
TMP_DIR="$(mktemp -d /tmp/madaros-e3e-equal-event.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
DRIVER="$TMP_DIR/madaros-enir"

fail() { echo "E3E_EQUAL_VALUE_DISTINCT_EVENT_GATE_FAIL: $*" >&2; exit 1; }

. scripts/dev/madaros_v2_enir_gate_scope.sh

[[ -x "$SEED" ]] || fail "missing executable Stage0 seed: $SEED"
git rev-parse --verify "$BASE_REF" >/dev/null 2>&1 || fail "base ref not found: $BASE_REF"
E3E_PROTECTED=(
  self-hosted/compiler/main.sio self-hosted/ir self-hosted/native self-hosted/wasm \
  self-hosted/gpu stdlib/runtime stdlib/eisa stdlib/math/dd64.sio stdlib/math/qd128.sio \
  self-hosted/enir/qd.sio self-hosted/enir/mir_cfg.sio self-hosted/enir/mir_join.sio \
  tools/eisa/eisa_evm_run.sio \
)
madaros_v2_enir_gate_scope_or_skip "$BASE_REF" "E3E_EQUAL_VALUE_DISTINCT_EVENT_GATE" \
  "E3E changed production codegen/ABI/runtime, pinned qd semantics, Join-MIR, or METRON oracle" "${E3E_PROTECTED[@]}"
scripts/dev/souc-build-lock.sh "$SEED" self-hosted/enir/driver.sio "$DRIVER" >"$TMP_DIR/driver-build.log" 2>&1
[[ -s "$DRIVER" ]] || fail "source-fresh ENIR driver build produced no ELF"
chmod +x "$DRIVER"
run_driver() { timeout 30s "$DRIVER" "$@"; }

VERIFY_TOKEN='jmir-ok|'
RELATION_TOKEN='jmir-relation-ok|'

for path in then else; do
  source="tools/eisa/eisa_enir_v2_equal_${path}.eisa"
  [[ -f "$source" ]] || fail "missing source fixture: $source"
  run_driver lower-join-v2 "$source" >"$TMP_DIR/$path.enir"
  run_driver lower-join-mir "$TMP_DIR/$path.enir" >"$TMP_DIR/$path.jmir"
  run_driver lower-join-mir "$TMP_DIR/$path.enir" >"$TMP_DIR/$path.jmir.replay"
  cmp -s "$TMP_DIR/$path.jmir" "$TMP_DIR/$path.jmir.replay" || fail "$path Join MIR lowering is nondeterministic"
  run_driver verify-join-mir "$TMP_DIR/$path.jmir" >"$TMP_DIR/$path.verify"
  run_driver validate-join-mir "$TMP_DIR/$path.enir" "$TMP_DIR/$path.jmir" >"$TMP_DIR/$path.relation"
  grep -Fq "$VERIFY_TOKEN" "$TMP_DIR/$path.verify" || fail "$path verification omitted its success receipt"
  grep -Fq "$RELATION_TOKEN" "$TMP_DIR/$path.relation" || fail "$path relation omitted its success receipt"
  run_driver run-join-mir "$TMP_DIR/$path.jmir" >"$TMP_DIR/$path.execution"
  run_driver run-join-mir "$TMP_DIR/$path.jmir" >"$TMP_DIR/$path.execution.replay"
  cmp -s "$TMP_DIR/$path.execution" "$TMP_DIR/$path.execution.replay" || fail "$path execution receipt is nondeterministic"
done

python3 - "$TMP_DIR/then.execution" "$TMP_DIR/else.execution" <<'PY'
from pathlib import Path
import hashlib
import sys

def rows(path):
    parsed = {}
    raw = Path(path).read_bytes()
    for line in raw.decode("ascii").splitlines():
        parts = line.split("|")
        if not parts[0] or len(parts) < 2:
            raise SystemExit(f"malformed receipt line in {path}: {line!r}")
        fields = {}
        for item in parts[1:]:
            if item.count("=") != 1:
                raise SystemExit(f"malformed receipt field in {path}: {item!r}")
            key, value = item.split("=", 1)
            if not key or not value or key in fields:
                raise SystemExit(f"invalid receipt field in {path}: {item!r}")
            fields[key] = value
        parsed.setdefault(parts[0], []).append(fields)
    return parsed, hashlib.sha256(raw).hexdigest()

then, then_hash = rows(sys.argv[1])
otherwise, else_hash = rows(sys.argv[2])

def one(data, tag):
    values = data.get(tag, [])
    if len(values) != 1:
        raise SystemExit(f"expected one {tag} receipt, got {len(values)}")
    return values[0]

def require(row, tag, keys):
    missing = [key for key in keys if key not in row]
    if missing:
        raise SystemExit(f"{tag} receipt omitted keys: {','.join(missing)}")

observation_words = ("value_bits", "error0_bits", "error1_bits", "error2_bits", "error3_bits", "uncertainty_bits", "status")
then_observation = one(then, "jmir-exec")
else_observation = one(otherwise, "jmir-exec")
require(then_observation, "then observation", observation_words)
require(else_observation, "else observation", observation_words)
if tuple(then_observation[key] for key in observation_words) != tuple(else_observation[key] for key in observation_words):
    raise SystemExit("observable numeric states differ")

def branch_control(data):
    values = [row for row in data.get("jmir-control", []) if row.get("condition") != "-1"]
    if len(values) != 1:
        raise SystemExit(f"expected one conditional control receipt, got {len(values)}")
    require(values[0], "conditional control", ("condition", "edge"))
    return values[0]

then_control = branch_control(then)
else_control = branch_control(otherwise)
if then_control["edge"] == else_control["edge"]:
    raise SystemExit("control receipts erased the distinct realized edges")

then_scalar = one(then, "jmir-scalar-phi")
else_scalar = one(otherwise, "jmir-scalar-phi")
require(then_scalar, "then scalar phi", ("incoming_value",))
require(else_scalar, "else scalar phi", ("incoming_value",))
if then_scalar["incoming_value"] == else_scalar["incoming_value"]:
    raise SystemExit("scalar phi receipts erased the distinct incoming values")

then_memory = then.get("jmir-memory-phi", [])
else_memory = otherwise.get("jmir-memory-phi", [])
if len(then_memory) != 2 or len(else_memory) != 2:
    raise SystemExit("expected two memory phi receipts per execution")
for row in then_memory + else_memory:
    require(row, "memory phi", ("slot", "incoming_version"))
then_versions = {row["slot"]: row["incoming_version"] for row in then_memory}
else_versions = {row["slot"]: row["incoming_version"] for row in else_memory}
if len(then_versions) != 2 or len(else_versions) != 2:
    raise SystemExit("memory phi receipts contain duplicate slots")
if then_versions == else_versions:
    raise SystemExit("memory phi receipts erased the distinct incoming versions")
if then_hash == else_hash:
    raise SystemExit("distinct event receipts have identical hashes")

print(
    "E3E_EQUAL_VALUE_DISTINCT_EVENT_WITNESS_PASS "
    f"observable_bits={then_observation['value_bits']} "
    f"then_edge={then_control['edge']} else_edge={else_control['edge']} "
    f"then_scalar={then_scalar['incoming_value']} else_scalar={else_scalar['incoming_value']} "
    f"then_receipt_sha256={then_hash} else_receipt_sha256={else_hash}"
)
PY
