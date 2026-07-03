#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/dev/madaros_two_gate.sh <madaros-elf> [--gate-b <probe.sio>]

Gate A: the 6 imported-SMT tests under tests/stdlib/theorem/test_smt_*.sio.
Gate B: a dissertation-facing probe. Defaults to a generated println(f64)
        probe; pass --gate-b to use a specific .sio probe.

Prints: two_gate: A=<n>/6 B=<pass|fail>
Exits 0 only when A=6/6 and B=pass.

Env:
  SOUNIO_MADAROS_TWO_GATE_STDLIB_PATH  stdlib root (default: repo stdlib)
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

ELF="$1"
shift
GATE_B=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gate-b)
      if [[ $# -lt 2 ]]; then
        echo "error: --gate-b requires a probe path" >&2
        exit 2
      fi
      GATE_B="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MADAROS="$ROOT_DIR/bin/madaros"
GATE_STDLIB="${SOUNIO_MADAROS_TWO_GATE_STDLIB_PATH:-$ROOT_DIR/stdlib}"

if [[ ! -x "$ELF" ]]; then
  echo "error: Madaros ELF not executable: $ELF" >&2
  exit 2
fi
if [[ ! -x "$MADAROS" ]]; then
  echo "error: wrapper not executable: $MADAROS" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/madaros-two-gate.XXXXXX")"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [[ -z "$GATE_B" ]]; then
  GATE_B="$TMP_DIR/gate_b_println_f64.sio"
  cat > "$GATE_B" <<'EOF'
fn main() -> i64 {
    println(0.5)
    0
}
EOF
fi

mapfile -t SMT_TESTS < <(find "$ROOT_DIR/tests/stdlib/theorem" -maxdepth 1 -name 'test_smt_*.sio' -print | sort)
if [[ "${#SMT_TESTS[@]}" -ne 6 ]]; then
  echo "error: expected exactly 6 test_smt fixtures, found ${#SMT_TESTS[@]}" >&2
  exit 2
fi

run_probe() {
  local probe="$1"
  local log="$2"
  MADAROS_RAW_BIN="$ELF" SOUNIO_STDLIB_PATH="$GATE_STDLIB" "$MADAROS" run "$probe" >"$log" 2>&1
}

gate_a_pass=0
for test_path in "${SMT_TESTS[@]}"; do
  test_log="$TMP_DIR/$(basename "$test_path").log"
  run_probe "$test_path" "$test_log"
  probe_rc=$?
  if [[ "$probe_rc" -eq 0 ]]; then
    gate_a_pass=$((gate_a_pass + 1))
  else
    echo "gateA FAIL: $(basename "$test_path") (rc=$probe_rc)" >&2
  fi
done

echo "gateA=$gate_a_pass/6"

gate_b_log="$TMP_DIR/gate_b.log"
run_probe "$GATE_B" "$gate_b_log"
gate_b_rc=$?
if [[ "$gate_b_rc" -eq 0 ]]; then
  gate_b_status="pass"
else
  gate_b_status="fail"
  echo "gateB FAIL: $GATE_B (rc=$gate_b_rc)" >&2
fi

echo "gateB=$gate_b_status"
echo "two_gate: A=$gate_a_pass/6 B=$gate_b_status"

if [[ "$gate_a_pass" -eq 6 && "$gate_b_status" == "pass" ]]; then
  exit 0
fi
exit 1
