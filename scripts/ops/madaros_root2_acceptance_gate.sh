#!/usr/bin/env bash
# Operator gate for the Madaros Root2/SRET/enum repair lane.
#
# This is intentionally not wired into CI while the blocker is open. It packages
# the current acceptance repros so the compiler owner can rerun the same
# environment-clean checks before marking the lane ready.

set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ALLOW_FAIL=0
OUT_DIR="${SOUNIO_MADAROS_ROOT2_GATE_DIR:-}"

usage() {
  cat <<'USAGE'
Usage: scripts/ops/madaros_root2_acceptance_gate.sh [--allow-fail] [--out-dir DIR] [--root DIR]

Runs the current Madaros Root2/SRET acceptance probes with SOUC/Madaros routing
environment variables unset:

  1. scripts/ci/madaros_operational_contract_gate.sh
  2. scripts/ci/native_v2_enum_match_gate.sh
  3. scripts/run_sio_test_suite.sh sret_forwarding --verbose

By default the script exits non-zero if any probe fails. Use --allow-fail for
read-only diagnostic snapshots while the blocker is still open. Use --root to
run the gate from a current checkout against an older compiler worktree without
copying this script into that worktree.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-fail)
      ALLOW_FAIL=1
      shift
      ;;
    --out-dir)
      if [[ $# -lt 2 ]]; then
        echo "[madaros-root2] --out-dir requires a directory" >&2
        exit 2
      fi
      OUT_DIR="$2"
      shift 2
      ;;
    --root)
      if [[ $# -lt 2 ]]; then
        echo "[madaros-root2] --root requires a repository directory" >&2
        exit 2
      fi
      ROOT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[madaros-root2] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ROOT_DIR="$(cd "$ROOT_DIR" && pwd)" || exit 1
cd "$ROOT_DIR" || exit 1

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(mktemp -d /tmp/sounio-madaros-root2-gate.XXXXXX)"
fi
mkdir -p "$OUT_DIR" || exit 1

printf '[madaros-root2] root=%s\n' "$ROOT_DIR"
printf '[madaros-root2] out=%s\n' "$OUT_DIR"
printf '[madaros-root2] allow_fail=%s\n' "$ALLOW_FAIL"

FAILURES=0

run_probe() {
  local name="$1"
  shift
  local log="$OUT_DIR/$name.log"

  printf '[madaros-root2] RUN %s\n' "$name"
  env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN "$@" >"$log" 2>&1
  local rc=$?

  if [[ "$rc" -eq 0 ]]; then
    printf '[madaros-root2] PASS %s\n' "$name"
  else
    printf '[madaros-root2] FAIL %s rc=%s log=%s\n' "$name" "$rc" "$log" >&2
    tail -n 80 "$log" >&2 || true
    FAILURES=$((FAILURES + 1))
  fi
}

run_probe madaros_operational_contract \
  bash scripts/ci/madaros_operational_contract_gate.sh

run_probe native_v2_enum_match \
  bash scripts/ci/native_v2_enum_match_gate.sh

run_probe sret_forwarding \
  bash scripts/run_sio_test_suite.sh sret_forwarding --verbose

printf '[madaros-root2] failures=%s\n' "$FAILURES"

if [[ "$FAILURES" -ne 0 && "$ALLOW_FAIL" -ne 1 ]]; then
  printf '[madaros-root2] FAIL: acceptance gate still red\n' >&2
  exit 1
fi

if [[ "$FAILURES" -ne 0 ]]; then
  printf '[madaros-root2] DIAGNOSTIC: failures allowed for open blocker\n'
else
  printf '[madaros-root2] PASS: Root2/SRET acceptance probes are green\n'
fi
