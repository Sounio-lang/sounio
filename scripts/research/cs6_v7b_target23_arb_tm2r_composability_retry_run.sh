#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="$ROOT/scripts/research/cs6_v7b_target23_arb_tm2r_composability_retry_worker.py"
DEFAULT_OUT="$ROOT/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_composability_v1"
OUT_DIR="${CS6_OUTPUT_DIR:-$DEFAULT_OUT}"
DEPS="${CS6_PYTHONPATH:-/tmp/sounio-cs6-arb-full-leaf-deps}"
TILE="${CS6_SOURCE_TILE:?set CS6_SOURCE_TILE to XLEL, XLEH, XHEL, or XHEH}"

case "$TILE" in
  XLEL|XLEH|XHEL|XHEH) ;;
  *) echo "invalid CS6_SOURCE_TILE: $TILE" >&2; exit 2 ;;
esac
[[ -d "$DEPS/flint" ]] || {
  echo "python-flint dependency directory is unavailable: $DEPS" >&2
  exit 2
}

mkdir -p "$OUT_DIR"
result="$OUT_DIR/support_${TILE}.json"
stderr="$OUT_DIR/support_${TILE}.stderr.txt"
tmp_result="$result.tmp.$$"
tmp_stderr="$stderr.tmp.$$"
trap 'rm -f "$tmp_result" "$tmp_stderr"' EXIT

set +e
CS6_SOURCE_TILE="$TILE" PYTHONPATH="$DEPS" PYTHONDONTWRITEBYTECODE=1 \
  python3 -B "$WORKER" > "$tmp_result" 2> "$tmp_stderr"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  mv "$tmp_stderr" "$OUT_DIR/support_${TILE}.retry-incomplete.stderr.txt"
  echo "composability retry worker failed for $TILE with rc=$rc" >&2
  exit "$rc"
fi

PYTHONPATH="$DEPS" PYTHONDONTWRITEBYTECODE=1 python3 -B - "$tmp_result" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="ascii") as handle:
    payload = json.load(handle)
if payload.get("schema") != "sounio.cs6.v7b-target23-arb-tm2r-composability-carrier.v1":
    raise SystemExit("wrong composability carrier schema")
if payload.get("execution_profile") != "EXTENDED_SPLIT_BUDGET_V1":
    raise SystemExit("wrong retry execution profile")
if payload.get("max_event_split_depth") != 12:
    raise SystemExit("wrong retry split-depth budget")
if payload.get("max_event_split_nodes_per_tile") != 255:
    raise SystemExit("wrong retry split-node budget")
if payload.get("selected_source_chain_certificate") is not True:
    raise SystemExit("retry worker did not certify its complete source tile")
if payload.get("point_fallback_used") is not False:
    raise SystemExit("point fallback is forbidden")
if payload.get("terminal_domain_cover_certified") is not True:
    raise SystemExit("terminal symbolic domains do not certify the source cover")
if not payload.get("carriers"):
    raise SystemExit("retry worker emitted no final carriers")
PY

mv "$tmp_result" "$result"
mv "$tmp_stderr" "$stderr"
trap - EXIT
echo "CS6_COMPOSABILITY_RETRY_TILE_COMPLETE=$TILE"
