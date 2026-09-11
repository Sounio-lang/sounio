#!/usr/bin/env bash
# KL-4: public fields remain accessible; private fields reject external
# reads, stores, and struct-literal initialization with E259.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true

SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-}}"
[[ -x "$SOUC" ]] || { echo "FAIL: souc is not executable: $SOUC" >&2; exit 2; }
[[ -n "$RAW" && -x "$RAW" ]] || {
  echo "FAIL: MADAROS_RAW_BIN must name a current-source Madaros ELF" >&2
  exit 2
}

WORK="$(mktemp -d /tmp/madaros-kl4-field-visibility.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

cat >"$WORK/kl4_gate_lib.sio" <<'SIO'
pub struct Capsule {
    pub visible: i64,
    hidden: i64,
}

impl Capsule {
    pub fn new(value: i64) -> Capsule {
        Capsule { visible: value, hidden: value + 1 }
    }

    pub fn reveal(&self) -> i64 {
        self.hidden
    }
}
SIO

cat >"$WORK/read.sio" <<'SIO'
use kl4_gate_lib::{Capsule}
fn main() -> i64 {
    let c = Capsule::new(1)
    c.hidden
}
SIO

cat >"$WORK/store.sio" <<'SIO'
use kl4_gate_lib::{Capsule}
fn main() -> i64 {
    var c = Capsule::new(1)
    c.hidden = 9
    c.visible
}
SIO

cat >"$WORK/literal.sio" <<'SIO'
use kl4_gate_lib::{Capsule}
fn main() -> i64 {
    let c = Capsule { visible: 1, hidden: 2 }
    c.visible
}
SIO

expect_e259() {
  local label="$1"
  local source="$2"
  local log="$WORK/$label.log"
  local rc=0
  set +e
  MADAROS_RAW_BIN="$RAW" "$SOUC" check "$source" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 1 ]] || { cat "$log"; echo "FAIL: $label rc=$rc, expected 1" >&2; exit 1; }
  [[ "$(grep -Fc 'error[E259' "$log" || true)" -eq 1 ]] || {
    cat "$log"
    echo "FAIL: $label did not emit exactly one E259" >&2
    exit 1
  }
  grep -Fq 'struct field is private in its defining module' "$log" || {
    cat "$log"
    echo "FAIL: $label lacked canonical E259 message" >&2
    exit 1
  }
  echo "PASS $label=E259"
}

expect_e259 read "$WORK/read.sio"
expect_e259 store "$WORK/store.sio"
expect_e259 literal "$WORK/literal.sio"

RUN_LOG="$WORK/run.log"
MADAROS_RAW_BIN="$RAW" "$SOUC" run tests/run-pass/kl4_field_visibility_ok.sio >"$RUN_LOG" 2>&1
grep -Fxq 'KL4_FIELD_VISIBILITY_OK' "$RUN_LOG" || {
  cat "$RUN_LOG"
  echo "FAIL: positive witness lacked exact sentinel" >&2
  exit 1
}
echo "PASS public-read-and-private-method=KL4_FIELD_VISIBILITY_OK"

# Sabotage control: making the private field public must make all three
# negative probes check successfully, proving that E259 is visibility-driven.
python3 - "$WORK/kl4_gate_lib.sio" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
p.write_text(p.read_text().replace("    hidden: i64,", "    pub hidden: i64,"))
PY
for probe in read store literal; do
  log="$WORK/sabotage-$probe.log"
  MADAROS_RAW_BIN="$RAW" "$SOUC" check "$WORK/$probe.sio" >"$log" 2>&1 || {
    cat "$log"
    echo "FAIL: sabotage did not release $probe" >&2
    exit 1
  }
done
echo "PASS sabotage-private-to-pub=releases-read-store-literal"
echo "MADAROS_KL4_FIELD_VISIBILITY_GATE_OK"
