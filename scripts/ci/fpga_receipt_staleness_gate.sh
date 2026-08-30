#!/usr/bin/env bash
# Guards against an artifacts/fpga/*.json receipt reporting a *_synth_status or
# *_sim_status of "pass" without either declaring itself stale or the RTL it
# claims to have processed actually being present.
#
# Measured 2026-07-27: hardware/** had never been versioned in this repository
# yet artifacts/fpga/fpga_seed_report.json reported synth_status: pass and
# merkle_lane_core_rtl_present: true for RTL sources
# (hardware/fpga/k_axi_merkle_lane.v and siblings) that do not exist and never
# existed in this checkout. That receipt now carries stale: true; this gate
# stops the next one from repeating the same false signal silently.
#
# Measured 2026-08-30 -- this gate had two independent holes, separated by a
# 2x2 control (key shape x presence of hardware/):
#
#   key                       hardware/ present (today)   hardware/ absent (07-27)
#   k_axi_synth_status: pass  ACCEPTED                    refused
#   synth_status: pass        ACCEPTED                    ACCEPTED
#
# 1. Self-disarm. The backing check was `[[ ! -d hardware ]]`, a proxy for "no
#    RTL here". On 2026-08-04, eight days after this gate landed, commits
#    6e2113459d and 40116b661d added hardware/fpga/u250_catastrophe_scan/ --
#    Vitis HLS C++ and shell, twelve files, zero .v/.sv. The directory existed,
#    the proxy went false, and the gate stopped refusing anything. The backing
#    check now counts tracked RTL sources by extension at any path, so an
#    unrelated directory cannot switch it off.
#
# 2. Bare keys were never covered. The match was `k.endswith("_synth_status")`,
#    which does not match a key that IS `synth_status` -- the top-level field in
#    fpga_seed_report.json, and the exact field the 2026-07-27 incident was
#    about. That hole was present from the day this gate was written. Keys are
#    now matched when they equal the suffix or end with it.
#
# The header also promised *_sim_status coverage the implementation never had;
# it does now.
#
# Run with --selftest (or SELFTEST=1) to exercise the negative controls alone;
# they run automatically before the real check.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# A receipt's "pass" is backed only if this checkout actually tracks RTL for it
# to describe. Sources are counted by extension, at any path -- keying on a
# directory name is what let an unrelated HLS lane disarm this gate.
rtl_source_count() {
  git ls-files -- '*.v' '*.sv' '*.vhd' '*.vhdl' 2>/dev/null | grep -c . || true
}

# Does this receipt claim a pass for something it did not necessarily do?
# Both synth and sim are load-bearing: a "sim_status: pass" for absent RTL is
# the same false signal as a synth one.
claims_unbacked_pass() {
  python3 -c '
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
except (OSError, json.JSONDecodeError):
    print("unreadable"); sys.exit(0)
if not isinstance(data, dict):
    print("false"); sys.exit(0)
if data.get("stale"):
    print("stale"); sys.exit(0)
def claims(k):
    # a key that IS "synth_status" matters as much as "k_axi_synth_status";
    # endswith("_synth_status") alone missed the receipt field this gate exists for
    return any(k == s or k.endswith("_" + s) for s in ("synth_status", "sim_status"))

hit = any(claims(k) and v == "pass" for k, v in data.items())
print("true" if hit else "false")
' "$1"
}

scan_reports() {
  local dir="$1" fail=0 report verdict
  shopt -s nullglob
  for report in "$dir"/*.json; do
    verdict="$(claims_unbacked_pass "$report")"
    case "$verdict" in
      unreadable|stale|false) continue ;;
    esac
    if [[ "$(rtl_source_count)" -eq 0 ]]; then
      echo "$report: reports a *_synth_status or *_sim_status of \"pass\" but is" >&2
      echo "  not marked stale:true, and this checkout tracks no RTL sources" >&2
      echo "  (*.v, *.sv, *.vhd, *.vhdl) for it to describe." >&2
      echo "  Either the RTL this report claims to have processed needs to be" >&2
      echo "  actually present, or the report needs stale:true + stale_reason." >&2
      fail=1
    fi
  done
  return "$fail"
}

selftest() {
  local tmp rc
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  # 1. a bare synth pass with no stale marker must be refused
  printf '{"synth_status": "pass"}' > "$tmp/synth.json"
  if scan_reports "$tmp" 2>/dev/null; then
    echo "selftest FAILED: an unbacked *_synth_status pass was accepted" >&2
    return 1
  fi
  rm -f "$tmp/synth.json"

  # 2. the same for sim -- the coverage this gate's header promised since 07-27
  printf '{"k_axi_sim_status": "pass"}' > "$tmp/sim.json"
  if scan_reports "$tmp" 2>/dev/null; then
    echo "selftest FAILED: an unbacked *_sim_status pass was accepted" >&2
    return 1
  fi
  rm -f "$tmp/sim.json"

  # 3. an honest receipt that declares itself stale must be allowed through
  printf '{"stale": true, "stale_reason": "generated elsewhere", "synth_status": "pass"}' > "$tmp/stale.json"
  if ! scan_reports "$tmp" 2>/dev/null; then
    echo "selftest FAILED: a receipt honestly marked stale was refused" >&2
    return 1
  fi
  rm -f "$tmp/stale.json"

  # 4. null control -- a receipt claiming no pass at all must be allowed, so
  #    the three refusals above are attributable to the pass claim itself
  printf '{"synth_status": "fail", "note": "nothing claimed"}' > "$tmp/nopass.json"
  if ! scan_reports "$tmp" 2>/dev/null; then
    echo "selftest FAILED: a receipt claiming no pass was refused" >&2
    return 1
  fi

  echo "fpga receipt staleness selftest passed (2 refusals + stale exemption + null control)."
}

selftest

if [[ "${1:-}" == "--selftest" || "${SELFTEST:-}" == "1" ]]; then
  exit 0
fi

if ! scan_reports artifacts/fpga; then
  exit 1
fi

echo "fpga receipt staleness check passed: no non-stale report claims synthesis of absent RTL."
