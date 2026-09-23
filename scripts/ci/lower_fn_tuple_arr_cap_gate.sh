#!/usr/bin/env bash
# LOWER_FN_TUPLE_ARR physical-extent / logical-cap coherence (#2570).
#
# self-hosted/ir/lower.sio's f64-array-tuple table has a logical admission
# guard (`LOWER_FN_TUPLE_ARR_COUNT < LOWER_FN_TUPLE_ARR_CAP`) and four
# physical storage arrays (HASH, MASK, NAME_BUF, NAME_LEN) that must be able
# to hold at least CAP entries. Sizing those four off LOWER_FN_TUPLE_ARR_CAP
# directly (`[i64; LOWER_FN_TUPLE_ARR_CAP]`) was tried and reverted: a global
# array sized by a non-literal expression silently under-allocates at this
# scale on the current bootstrap compiler -- measured as a SIGSEGV writing
# the last index of an isolated 384*4096-byte global sized the same way,
# while a small-scale version of the identical pattern ran clean. Array
# lengths in this file must stay literal integers.
#
# That leaves the four extents and the two constants as six independent
# literals a human has to keep in sync by hand. A Copilot review on this PR
# caught exactly the failure mode: bump LOWER_FN_TUPLE_ARR_CAP (which is
# what the "would need a bigger LOWER_FN_TUPLE_ARR_CAP" refusal message in
# the same file invites) without also bumping the four storage arrays, and
# the guard admits slots the physical arrays cannot hold -- an out-of-bounds
# write that `souc check` cannot see, because every literal involved is
# independently well-typed.
#
# This gate is the substitute for a compile-time check the language cannot
# express here: it parses the six literals back out of source and asserts
# HASH == MASK == NAME_LEN == CAP and NAME_BUF == STRIDE * CAP.
set -euo pipefail
cd "$(dirname "$0")/../.."
. "$(dirname "${BASH_SOURCE[0]}")/../lib/gate_assert.sh"
# Copilot follow-up (#2570): gate_write_artifact (called below at the
# positive control and both final writes) is defined in gate_artifact.sh,
# not gate_assert.sh. It happened to still be available here because
# gate_assert.sh itself sources gate_artifact.sh transitively -- but every
# other gate that calls gate_write_artifact sources it directly rather than
# relying on that, and for good reason: it is an implementation detail of
# gate_assert.sh, not a documented contract, and would silently break this
# gate under `set -euo pipefail` ("command not found") if a future refactor
# of gate_assert.sh ever dropped it. Source it directly, matching the
# convention every other artifact-writing gate already follows.
. "$(dirname "${BASH_SOURCE[0]}")/../lib/gate_artifact.sh"
gate_name "lower_fn_tuple_arr_cap"

LOWER=self-hosted/ir/lower.sio
ART=artifacts/gates/lower_fn_tuple_arr_cap.v1.json
mkdir -p "$(dirname "$ART")"

# check <file> -- fails (exit 1) unless the six literals agree. Run inside a
# subshell by callers that need a sabotaged copy to fail without killing the
# gate itself: gate_fail below calls `exit`, which only unwinds the subshell.
check() {
  local file="$1"
  require_file "$file" "check: no such file: $file"

  # sed capture groups, not grep -oE + a second digit-only pass: the type
  # tokens themselves ("i64", "i8") contain digits, so a second `[0-9]+`
  # sweep over the whole matched line picks up "64" from "i64" as a spurious
  # extra match ahead of the real extent. Anchoring the capture group to
  # exactly the text between "; " and "]" (or "= " and end of line) sidesteps
  # that instead of relying on which match happens to sort last.
  local cap stride hash_n mask_n namebuf_n namelen_n
  cap=$(sed -nE 's/^let LOWER_FN_TUPLE_ARR_CAP: i64 = ([0-9]+)$/\1/p' "$file")
  stride=$(sed -nE 's/^let LOWER_FN_TUPLE_ARR_NAME_STRIDE: i64 = ([0-9]+)$/\1/p' "$file")
  hash_n=$(sed -nE 's/^var LOWER_FN_TUPLE_ARR_HASH: \[i64; ([0-9]+)\].*$/\1/p' "$file")
  mask_n=$(sed -nE 's/^var LOWER_FN_TUPLE_ARR_MASK: \[i64; ([0-9]+)\].*$/\1/p' "$file")
  namebuf_n=$(sed -nE 's/^var LOWER_FN_TUPLE_ARR_NAME_BUF: \[i8; ([0-9]+)\].*$/\1/p' "$file")
  namelen_n=$(sed -nE 's/^var LOWER_FN_TUPLE_ARR_NAME_LEN: \[i64; ([0-9]+)\].*$/\1/p' "$file")

  require_nonempty "$cap" "LOWER_FN_TUPLE_ARR_CAP not found in $file"
  require_nonempty "$stride" "LOWER_FN_TUPLE_ARR_NAME_STRIDE not found in $file"
  require_nonempty "$hash_n" "LOWER_FN_TUPLE_ARR_HASH extent not found in $file"
  require_nonempty "$mask_n" "LOWER_FN_TUPLE_ARR_MASK extent not found in $file"
  require_nonempty "$namebuf_n" "LOWER_FN_TUPLE_ARR_NAME_BUF extent not found in $file"
  require_nonempty "$namelen_n" "LOWER_FN_TUPLE_ARR_NAME_LEN extent not found in $file"

  [[ "$hash_n" == "$cap" ]] || gate_fail "LOWER_FN_TUPLE_ARR_HASH extent ($hash_n) != LOWER_FN_TUPLE_ARR_CAP ($cap)"
  [[ "$mask_n" == "$cap" ]] || gate_fail "LOWER_FN_TUPLE_ARR_MASK extent ($mask_n) != LOWER_FN_TUPLE_ARR_CAP ($cap)"
  [[ "$namelen_n" == "$cap" ]] || gate_fail "LOWER_FN_TUPLE_ARR_NAME_LEN extent ($namelen_n) != LOWER_FN_TUPLE_ARR_CAP ($cap)"
  local expect_namebuf=$(( stride * cap ))
  [[ "$namebuf_n" == "$expect_namebuf" ]] || gate_fail "LOWER_FN_TUPLE_ARR_NAME_BUF extent ($namebuf_n) != NAME_STRIDE * CAP ($expect_namebuf)"

  printf '%s %s %s %s %s %s' "$cap" "$stride" "$hash_n" "$mask_n" "$namebuf_n" "$namelen_n"
}

# Positive control FIRST. A checker that has never failed has measured
# nothing: if the sabotaged copy passes, this gate inspects nothing and must
# not be allowed to report green on the real file.
#
# Copilot follow-up (#2570): this used to hard-code the extent to sabotage
# (4096 -> 4097). A future, correctly synchronized cap raise changes that
# literal in the real file -- at which point the hard-coded sed finds
# nothing to replace, the "sabotaged" copy comes out byte-identical to the
# real (now-valid) file, check() on it PASSES, and this control reports
# CONTROL_FAIL on a change that did nothing wrong. That would block the
# exact synchronized update this gate exists to allow. Derive the extent to
# sabotage from the source instead of a literal, so the control still fires
# no matter what the cap currently is.
current_hash_n=$(sed -nE 's/^var LOWER_FN_TUPLE_ARR_HASH: \[i64; ([0-9]+)\].*$/\1/p' "$LOWER")
require_nonempty "$current_hash_n" "LOWER_FN_TUPLE_ARR_HASH extent not found in $LOWER -- cannot derive a sabotage target"
sabotaged_hash_n=$(( current_hash_n + 1 ))
SAB=$(mktemp); trap 'rm -f "$SAB"' EXIT
sed "s/^var LOWER_FN_TUPLE_ARR_HASH: \[i64; ${current_hash_n}\]/var LOWER_FN_TUPLE_ARR_HASH: [i64; ${sabotaged_hash_n}]/" "$LOWER" | gate_write_artifact "$SAB"
if ( check "$SAB" ) >/dev/null 2>&1; then
  echo "CONTROL_FAIL: the sabotaged extent passed. This gate inspects nothing."
  printf '{"status":"fail","reason":"positive control did not fire","metrics":{"total":6,"passed":0,"failed":1,"not_run":0}}\n' | gate_write_artifact "$ART"
  exit 1
fi
echo "control: sabotaged LOWER_FN_TUPLE_ARR_HASH extent rejected, as required"

vals="$(check "$LOWER")"
read -r cap stride hash_n mask_n namebuf_n namelen_n <<<"$vals"
echo "LOWER_FN_TUPLE_ARR_CAP_OK: cap=$cap stride=$stride hash=$hash_n mask=$mask_n namebuf=$namebuf_n namelen=$namelen_n"
printf '{"status":"pass","metrics":{"cap":%s,"stride":%s,"hash":%s,"mask":%s,"namebuf":%s,"namelen":%s,"total":6,"passed":6,"failed":0,"not_run":0}}\n' \
  "$cap" "$stride" "$hash_n" "$mask_n" "$namebuf_n" "$namelen_n" | gate_write_artifact "$ART"
