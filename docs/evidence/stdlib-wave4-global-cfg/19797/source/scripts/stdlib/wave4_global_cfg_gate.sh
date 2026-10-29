#!/usr/bin/env bash
# C++23 component against immutable existing captures; never invokes Sounio.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SLURM_JOB_ID:?Slurm required}" "${CXX:?qualified CXX required}" "${CXX_SHA256:?CXX pin required}"
: "${WIDE_CAPTURE:?complete immutable source capture required}" "${SCALAR_CAPTURE:?complete immutable source capture required}"
: "${WIDE_CAPTURE_SHA256:?}" "${SCALAR_CAPTURE_SHA256:?}"
OUT="${1:?fresh output directory required}"
mkdir "$OUT"
printf '%s  %s\n' "$CXX_SHA256" "$CXX" > "$OUT/CXX.sha256"
printf '%s  %s\n' "$WIDE_CAPTURE_SHA256" "$WIDE_CAPTURE" > "$OUT/CAPTURES.sha256"
printf '%s  %s\n' "$SCALAR_CAPTURE_SHA256" "$SCALAR_CAPTURE" >> "$OUT/CAPTURES.sha256"
sha256sum -c "$OUT/CXX.sha256" > "$OUT/CXX.before.verify"
sha256sum -c "$OUT/CAPTURES.sha256" > "$OUT/CAPTURES.before.verify"
sha256sum "$ROOT/tests/fixtures/wave4_global_cfg/reference.cpp" \
 "$ROOT/tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full_reference.cpp" > "$OUT/SOURCES.sha256"
cleanup() {
 local rc=$?
 trap - EXIT
 sha256sum -c "$OUT/SOURCES.sha256" > "$OUT/SOURCES.after.verify" || rc=1
 sha256sum -c "$OUT/CAPTURES.sha256" > "$OUT/CAPTURES.after.verify" || rc=1
 sha256sum -c "$OUT/CXX.sha256" > "$OUT/CXX.after.verify" || rc=1
 printf '%s\n' "$rc" > "$OUT/STATUS"
 exit "$rc"
}
trap cleanup EXIT
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror \
 "$ROOT/tests/fixtures/wave4_global_cfg/reference.cpp" -o "$OUT/reference" \
 > "$OUT/build.stdout" 2> "$OUT/build.stderr"
sha256sum "$OUT/reference" > "$OUT/AUDITOR.before.sha256"
timeout 120 "$OUT/reference" --self-controls "$OUT/controls" > "$OUT/controls.stdout" 2> "$OUT/controls.stderr"
grep -q '^GLOBAL_CFG_CONTROLS_OK count=21$' "$OUT/controls.stdout"
for kind in wide scalar; do
 input="$WIDE_CAPTURE"
 [[ "$kind" == wide ]] || input="$SCALAR_CAPTURE"
 set +e
 timeout 120 "$OUT/reference" --audit "$input" > "$OUT/$kind.audit.stdout" 2> "$OUT/$kind.audit.stderr"
 rc=$?
 set -e
 printf '%s\n' "$rc" > "$OUT/$kind.audit.status"
 [[ "$rc" == 1 ]]
done
grep -q '^GLOBAL_CFG_SUMMARY functions=19 executable=3 extern_empty=16 data=0 .* total_missing=6 coverage=complete$' "$OUT/wide.audit.stdout"
grep -q '^GLOBAL_CFG_FUNCTION fi=0 .* missing=3 ' "$OUT/wide.audit.stdout"
grep -q '^GLOBAL_CFG_FUNCTION fi=1 .* missing=3 ' "$OUT/wide.audit.stdout"
grep -q '^GLOBAL_CFG_SUMMARY functions=14 executable=2 extern_empty=12 data=0 .* total_missing=3 coverage=complete$' "$OUT/scalar.audit.stdout"
grep -q '^GLOBAL_CFG_FUNCTION fi=0 .* missing=3 ' "$OUT/scalar.audit.stdout"
sha256sum -c "$OUT/AUDITOR.before.sha256" > "$OUT/AUDITOR.after.verify"
printf '%s\n' 'GLOBAL_CFG_COMPONENT_BASELINE_OK; defective captures correctly rejected; no candidate qualification' > "$OUT/COMPONENT_ONLY"
