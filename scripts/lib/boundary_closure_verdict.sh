#!/usr/bin/env bash
# Verdict over a SOUNIO_BOUNDARY_CLOSURE_V1 report.
#
# `madaros --science-boundary-closure <src>` walks the `use` graph, parses every
# module in the closure and nothing else, and prints the report this script
# reads. Exit 0 iff the closure is complete AND has headroom.
#
# THE TRAP THIS EXISTS TO AVOID: the compiler exits 0 whether the closure
# completed or not. Measured on 2026-08-04 against the on-disk Madaros:
# `status incomplete`, `parse_failed true`, rc=0. A gate written against the
# exit code passes forever and looks exactly like a working gate. The verdict
# must come from the report body, which is why this is a separate, pure text
# function with no compiler dependency — so its own selftest can run on every
# PR, including docs-only ones, in under a second.
#
# usage: boundary_closure_verdict.sh <report-file>
set -uo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <report file>" >&2
  exit 64
fi

REPORT="$1"
reject() { echo "BOUNDARY_CLOSURE_REJECT: $*" >&2; exit 1; }

[[ -f "$REPORT" ]] || reject "no report at $REPORT"
[[ -s "$REPORT" ]] || reject "report is empty — the compiler produced nothing"

# The report is preceded by a three-line Madaros banner and a blank line.
grep -qxF 'SOUNIO_BOUNDARY_CLOSURE_V1' "$REPORT" \
  || reject "no SOUNIO_BOUNDARY_CLOSURE_V1 header — this is not a closure report, so nothing below was measured"

field() {  # field <name> -> value, empty if absent
  awk -F'\t' -v k="$1" '$1 == k { print $2; exit }' "$REPORT"
}
count() {  # count <name> -> number of lines whose first tab-field is <name>
  awk -F'\t' -v k="$1" '$1 == k { n++ } END { print n + 0 }' "$REPORT"
}

status=$(field status)
capacity=$(field capacity)
saturated=$(field saturated)
parse_failed=$(field parse_failed)
failed_path=$(field failed_path)

# Every field must be PRESENT. An absent field reads as empty, and an empty
# string compared against "true" is false — i.e. a report that stopped emitting
# `saturated` would silently satisfy a naive `!= true` test.
[[ -n "$status"       ]] || reject "no 'status' field — the report format changed and this verdict is no longer reading what it claims to"
[[ -n "$capacity"     ]] || reject "no 'capacity' field — cannot check headroom against an unknown bound"
[[ -n "$saturated"    ]] || reject "no 'saturated' field"
[[ -n "$parse_failed" ]] || reject "no 'parse_failed' field"

nodes=$(count node)
edges=$(count edge)
unresolved=$(count unresolved)

[[ "$parse_failed" == "false" ]] \
  || reject "the compiler could not parse a module in its own closure: ${failed_path:-<path not reported>}"

# failed_path is emitted only when parse_failed is true, so this can only be
# reached when the compiler claims failure without saying where. That regressed
# once; it does not get to regress silently again.
if [[ "$parse_failed" == "true" && -z "$failed_path" ]]; then
  reject "parse_failed with no failed_path — the report says the closure broke and refuses to say where"
fi

[[ "$saturated" == "false" ]] \
  || reject "closure SATURATED at capacity $capacity — the walk was truncated and 'status' below is about a partial graph"

[[ "$status" == "complete" ]] \
  || reject "status=$status (nodes=$nodes edges=$edges unresolved=$unresolved)"

[[ "$unresolved" -eq 0 ]] \
  || reject "$unresolved unresolved import(s) — a 'use' resolves to nothing"

# A closure of zero or one node is not a closure; it is a compiler that gave up
# immediately, or a report shape this function no longer understands.
[[ "$nodes" -ge 2 ]] \
  || reject "only $nodes node(s) — nothing meaningful was walked"

# Headroom. ModuleClosure is fixed-capacity (module_frontend.sio:2403 area:
# paths [String;256], edges [String;512]); on overflow the walker sets
# `saturated` and returns false. Measured 2026-08-04: 340 of 512 edges, 66%.
# Asserting at 80% turns the day it approaches the cap into a red PR instead of
# a mystery truncation.
node_cap="$capacity"
edge_cap=$(( capacity * 2 ))
node_limit=$(( node_cap * 80 / 100 ))
edge_limit=$(( edge_cap * 80 / 100 ))

[[ "$nodes" -le "$node_limit" ]] \
  || reject "closure at $nodes/$node_cap nodes (>80%) — raise ModuleClosure capacity before it truncates silently"
[[ "$edges" -le "$edge_limit" ]] \
  || reject "closure at $edges/$edge_cap edges (>80%) — raise ModuleClosure capacity before it truncates silently"

echo "BOUNDARY_CLOSURE_OK nodes=$nodes/$node_cap edges=$edges/$edge_cap unresolved=0"
