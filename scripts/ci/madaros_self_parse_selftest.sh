#!/usr/bin/env bash
# Proves that madaros_self_parse_gate.sh can FAIL.
#
# Modelled on scripts/ci/executable_payload_compare_selftest.sh: feed the
# verdict a known-good input and assert it accepts, then feed it known-bad ones
# and assert it REJECTS. A gate that cannot fail is decoration, and this whole
# family exists because a compiler that could not parse its own source tree sat
# behind 417 green gates.
#
# Arm 1 runs everywhere in under a second and needs no compiler, so it runs on
# every PR — including docs-only ones. That is what stops the gate rotting into
# a no-op the way an exit-code-only check would.
# Arms 2 and 3 need a Madaros and SKIP without one.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERDICT="$ROOT_DIR/scripts/lib/boundary_closure_verdict.sh"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-self-parse-selftest.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

[[ -x "$VERDICT" ]] || { echo "error: $VERDICT missing or not executable" >&2; exit 1; }

# ---------------------------------------------------------------- arm 1
# Synthetic reports. Numbers match what was measured on 2026-08-04:
# 120 nodes and 340 edges against capacity 256/512.
emit_report() {  # emit_report <file> <status> <saturated> <parse_failed> <nodes> <edges> [failed_path]
  local f="$1" st="$2" sat="$3" pf="$4" n="$5" e="$6" fp="${7:-}"
  {
    echo "Madaros v0.80.0 -- the Sounio self-hosted compiler"
    echo ""
    echo "SOUNIO_BOUNDARY_CLOSURE_V1"
    printf 'status\t%s\n' "$st"
    printf 'capacity\t256\n'
    printf 'saturated\t%s\n' "$sat"
    printf 'parse_failed\t%s\n' "$pf"
    [[ -n "$fp" ]] && printf 'failed_path\t%s\n' "$fp"
    local i
    for ((i = 0; i < n; i++)); do printf 'node\tself-hosted/x%s.sio\n' "$i"; done
    for ((i = 0; i < e; i++)); do printf 'edge\ta\tb\n'; done
  } >"$f"
}

must_accept() {
  if ! "$VERDICT" "$1" >/dev/null 2>&1; then
    echo "error: verdict REJECTED a good report ($2)" >&2; exit 1
  fi
}
must_reject() {
  if "$VERDICT" "$1" >/dev/null 2>&1; then
    echo "error: verdict ACCEPTED $2 — it is not measuring that" >&2; exit 1
  fi
}

emit_report "$WORK_DIR/good" complete false false 120 340
must_accept "$WORK_DIR/good" "the measured shape"

emit_report "$WORK_DIR/parsefail" incomplete false true 72 123 "self-hosted/resolve/scope.sio"
must_reject "$WORK_DIR/parsefail" "a closure that failed to parse"

emit_report "$WORK_DIR/nopath" incomplete false true 72 123
must_reject "$WORK_DIR/nopath" "parse_failed with no failed_path"

emit_report "$WORK_DIR/saturated" complete true false 256 512
must_reject "$WORK_DIR/saturated" "a SATURATED closure"

emit_report "$WORK_DIR/overnodes" complete false false 250 340
must_reject "$WORK_DIR/overnodes" "a closure over the 80% node headroom"

emit_report "$WORK_DIR/overedges" complete false false 120 500
must_reject "$WORK_DIR/overedges" "a closure over the 80% edge headroom"

emit_report "$WORK_DIR/unres" complete false false 120 340
printf 'unresolved\tcaller.sio\tmissing::thing\n' >>"$WORK_DIR/unres"
must_reject "$WORK_DIR/unres" "an unresolved import"

# The field-absence arm. A report that stops emitting a field reads as empty,
# and empty != "true" — so a naive check would pass on a report that measures
# nothing at all.
grep -v $'^saturated\t' "$WORK_DIR/good" >"$WORK_DIR/nosat"
must_reject "$WORK_DIR/nosat" "a report with no 'saturated' field"

: >"$WORK_DIR/empty"
must_reject "$WORK_DIR/empty" "an empty report"

printf 'total nonsense\nno header here\n' >"$WORK_DIR/nonsense"
must_reject "$WORK_DIR/nonsense" "output with no SOUNIO_BOUNDARY_CLOSURE_V1 header"

echo "  arm1 verdict function: accepts the good shape, rejects 9 bad ones"

# ---------------------------------------------------------------- arms 2 and 3
MADAROS="${SOUNIO_MADAROS_SELF_PARSE_BIN:-${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}}"
if [[ ! -x "$MADAROS" ]] || head -c2 "$MADAROS" 2>/dev/null | grep -q '#!'; then
  echo "  arms 2-3: SKIP (no raw Madaros ELF at $MADAROS)"
  echo "MADAROS_SELF_PARSE_SELFTEST_PASS"
  exit 0
fi

ulimit -s 524288 2>/dev/null || true

# The trees live in $WORK_DIR, never in the repo: arm 2 writes source the parser
# MUST refuse, and the gate's own tree sweep would trip over it.
mkdir -p "$WORK_DIR/bad/broken" "$WORK_DIR/pin/resolve"

cat >"$WORK_DIR/bad/broken/thing.sio" <<'BAD'
module broken::thing
fn ( ) -> { { {
BAD
cat >"$WORK_DIR/bad/main.sio" <<'BAD'
use broken::thing::*
fn main() -> i64 with IO { 0 }
BAD

if ! timeout 300 "$MADAROS" --science-boundary-closure "$WORK_DIR/bad/main.sio" \
     >"$WORK_DIR/bad.report" 2>&1; then
  echo "  note: closure mode exited non-zero on the bad tree (it normally exits 0 regardless)"
fi
must_reject "$WORK_DIR/bad.report" "a real closure over genuinely unparseable source"
echo "  arm2 end-to-end: compiler detects, report carries it, verdict refuses"

# Arm 3 — the #1624 pin. `module resolve::scope` is the exact shape that broke:
# parse_module_item tested == TokenKind::Ident alone, so the path stopped at the
# keyword segment. Red on any compiler carrying that defect, green after. It
# lives in a temp tree so it cannot be "fixed" by editing the repo.
cat >"$WORK_DIR/pin/resolve/scope.sio" <<'PIN'
module resolve::scope

pub struct Scope {
    start_index: i64,
    depth: i64,
}
PIN
cat >"$WORK_DIR/pin/main.sio" <<'PIN'
use resolve::scope::*
fn main() -> i64 with IO { 0 }
PIN

timeout 300 "$MADAROS" --science-boundary-closure "$WORK_DIR/pin/main.sio" \
  >"$WORK_DIR/pin.report" 2>&1 || true
if ! "$VERDICT" "$WORK_DIR/pin.report" >/dev/null 2>&1; then
  echo "error: this compiler cannot parse a module path with a keyword segment" >&2
  echo "       (\`module resolve::scope\` — the #1624 defect). Report:" >&2
  grep -vE '^(node|edge)\b' "$WORK_DIR/pin.report" | sed 's/^/       /' >&2
  exit 1
fi
echo "  arm3 #1624 pin: keyword segment in a module path still parses"

echo "MADAROS_SELF_PARSE_SELFTEST_PASS"
