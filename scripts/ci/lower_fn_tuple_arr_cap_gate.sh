#!/usr/bin/env bash
# LOWER_FN_TUPLE_ARR physical-extent / logical-cap coherence (#2570).
#
# self-hosted/ir/lower.sio's f64-array-tuple table has a logical admission
# guard (`LOWER_FN_TUPLE_ARR_COUNT < LOWER_FN_TUPLE_ARR_CAP`) and five
# physical storage arrays (HASH, MASK, SCALAR, NAME_BUF, NAME_LEN) that must
# be able to hold at least CAP entries. Sizing those five off
# LOWER_FN_TUPLE_ARR_CAP directly (`[i64; LOWER_FN_TUPLE_ARR_CAP]`) was tried
# and reverted: a global array sized by a non-literal expression silently
# under-allocates at this scale on the current bootstrap compiler -- measured
# as a SIGSEGV writing the last index of an isolated 384*4096-byte global
# sized the same way, while a small-scale version of the identical pattern
# ran clean. Array lengths in this file must stay literal integers.
#
# That leaves the five extents and the two constants as seven independent
# literals a human has to keep in sync by hand. A Copilot review on this PR
# caught exactly the failure mode: bump LOWER_FN_TUPLE_ARR_CAP (which is
# what the "would need a bigger LOWER_FN_TUPLE_ARR_CAP" refusal message in
# the same file invites) without also bumping the storage arrays, and the
# guard admits slots the physical arrays cannot hold -- an out-of-bounds
# write that `souc check` cannot see, because every literal involved is
# independently well-typed.
#
# Copilot follow-up (#2570): LOWER_FN_TUPLE_ARR_SCALAR (the second-order,
# TypeFn-chain tuple-return scalar-mask table added alongside MASK) is
# written and read at the exact same slot index as MASK, by the exact same
# lower_fn_tuple_arr_insert call -- an under-sized SCALAR array would be an
# out-of-bounds write the moment a slot MASK could legally reach is written,
# the identical failure mode this gate already exists to catch for the other
# four extents.
#
# Copilot follow-up (#2570): LOWER_FN_ARR_CHAIN_{HASH,NAME_BUF,NAME_LEN} (the
# separate array-chain-returning-function table added alongside
# LOWER_FN_TUPLE_ARR_*) reuse LOWER_FN_TUPLE_ARR_CAP and
# LOWER_FN_TUPLE_ARR_NAME_STRIDE for their own insertion guard and name
# storage, but are their OWN three physical arrays -- this gate checked only
# the five LOWER_FN_TUPLE_ARR_* ones, so a future synchronized cap raise
# could update every LOWER_FN_TUPLE_ARR_* extent, pass this gate, and still
# leave the three LOWER_FN_ARR_CHAIN_* arrays at their old (now too small)
# size, since nothing here was reading them at all. Checked the same way,
# against the same CAP and STRIDE.
#
# Merge of #2570 into main: main's table identifies rows by interned name id
# (LOWER_FN_TUPLE_ARR_NAME_ID) with an open-addressed index
# (LOWER_FN_TUPLE_ARR_SLOT), so the NAME_BUF/NAME_LEN/STRIDE extents this gate
# originally checked no longer exist; it now checks HASH, MASK, NAME_ID, SCALAR,
# CHAIN_HASH and CHAIN_NAME_ID against CAP, and the slot index against SLOT_SIZE.
#
# This gate is the substitute for a compile-time check the language cannot
# express here: it parses the ten literals back out of source and asserts
# HASH == MASK == SCALAR == NAME_LEN == CHAIN_HASH == CHAIN_NAME_LEN == CAP
# and NAME_BUF == CHAIN_NAME_BUF == STRIDE * CAP.
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

# check <file> -- fails (exit 1) unless the ten literals agree. Run inside a
# subshell by callers that need a sabotaged copy to fail without killing the
# gate itself: gate_fail below calls `exit`, which only unwinds the subshell.
check() {
  local file="$1"
  require_file "$file" "check: no such file: $file"

  # sed capture groups, not grep -oE + a second digit-only pass: the type
  # tokens themselves ("i64", "i8") contain digits, so a second `[0-9]+`
  # sweep over the whole matched line picks up "64" from "i64" as a spurious
  # extra match ahead of the real extent.
  local cap hash_n mask_n nameid_n scalar_n slot_size slot_mask slot_n
  local chain_hash_n chain_nameid_n
  cap=$(sed -nE 's/^let LOWER_FN_TUPLE_ARR_CAP: i64 = ([0-9]+)$/\1/p' "$file")
  hash_n=$(sed -nE 's/^var LOWER_FN_TUPLE_ARR_HASH: \[i64; ([0-9]+)\].*$/\1/p' "$file")
  mask_n=$(sed -nE 's/^var LOWER_FN_TUPLE_ARR_MASK: \[i64; ([0-9]+)\].*$/\1/p' "$file")
  nameid_n=$(sed -nE 's/^var LOWER_FN_TUPLE_ARR_NAME_ID: \[i64; ([0-9]+)\].*$/\1/p' "$file")
  scalar_n=$(sed -nE 's/^var LOWER_FN_TUPLE_ARR_SCALAR: \[i64; ([0-9]+)\].*$/\1/p' "$file")
  slot_size=$(sed -nE 's/^let LOWER_FN_TUPLE_ARR_SLOT_SIZE: i64 = ([0-9]+)$/\1/p' "$file")
  slot_mask=$(sed -nE 's/^let LOWER_FN_TUPLE_ARR_SLOT_MASK: i64 = ([0-9]+)$/\1/p' "$file")
  slot_n=$(sed -nE 's/^var LOWER_FN_TUPLE_ARR_SLOT: \[i64; ([0-9]+)\].*$/\1/p' "$file")
  chain_hash_n=$(sed -nE 's/^var LOWER_FN_ARR_CHAIN_HASH: \[i64; ([0-9]+)\].*$/\1/p' "$file")
  chain_nameid_n=$(sed -nE 's/^var LOWER_FN_ARR_CHAIN_NAME_ID: \[i64; ([0-9]+)\].*$/\1/p' "$file")

  require_nonempty "$cap" "LOWER_FN_TUPLE_ARR_CAP not found in $file"
  require_nonempty "$hash_n" "LOWER_FN_TUPLE_ARR_HASH extent not found in $file"
  require_nonempty "$mask_n" "LOWER_FN_TUPLE_ARR_MASK extent not found in $file"
  require_nonempty "$nameid_n" "LOWER_FN_TUPLE_ARR_NAME_ID extent not found in $file"
  require_nonempty "$scalar_n" "LOWER_FN_TUPLE_ARR_SCALAR extent not found in $file"
  require_nonempty "$slot_size" "LOWER_FN_TUPLE_ARR_SLOT_SIZE not found in $file"
  require_nonempty "$slot_mask" "LOWER_FN_TUPLE_ARR_SLOT_MASK not found in $file"
  require_nonempty "$slot_n" "LOWER_FN_TUPLE_ARR_SLOT extent not found in $file"
  require_nonempty "$chain_hash_n" "LOWER_FN_ARR_CHAIN_HASH extent not found in $file"
  require_nonempty "$chain_nameid_n" "LOWER_FN_ARR_CHAIN_NAME_ID extent not found in $file"

  [[ "$hash_n" == "$cap" ]] || gate_fail "LOWER_FN_TUPLE_ARR_HASH extent ($hash_n) != LOWER_FN_TUPLE_ARR_CAP ($cap)"
  [[ "$mask_n" == "$cap" ]] || gate_fail "LOWER_FN_TUPLE_ARR_MASK extent ($mask_n) != LOWER_FN_TUPLE_ARR_CAP ($cap)"
  [[ "$nameid_n" == "$cap" ]] || gate_fail "LOWER_FN_TUPLE_ARR_NAME_ID extent ($nameid_n) != LOWER_FN_TUPLE_ARR_CAP ($cap)"
  [[ "$scalar_n" == "$cap" ]] || gate_fail "LOWER_FN_TUPLE_ARR_SCALAR extent ($scalar_n) != LOWER_FN_TUPLE_ARR_CAP ($cap)"
  [[ "$chain_hash_n" == "$cap" ]] || gate_fail "LOWER_FN_ARR_CHAIN_HASH extent ($chain_hash_n) != LOWER_FN_TUPLE_ARR_CAP ($cap)"
  [[ "$chain_nameid_n" == "$cap" ]] || gate_fail "LOWER_FN_ARR_CHAIN_NAME_ID extent ($chain_nameid_n) != LOWER_FN_TUPLE_ARR_CAP ($cap)"
  # Open-addressed index: physical extent == SLOT_SIZE, SLOT_SIZE a power of
  # two with SLOT_MASK == SLOT_SIZE-1, and at least 2x CAP so the insert's
  # probe always finds an empty bucket (load factor <= 0.5).
  [[ "$slot_n" == "$slot_size" ]] || gate_fail "LOWER_FN_TUPLE_ARR_SLOT extent ($slot_n) != LOWER_FN_TUPLE_ARR_SLOT_SIZE ($slot_size)"
  [[ "$slot_mask" == "$(( slot_size - 1 ))" ]] || gate_fail "LOWER_FN_TUPLE_ARR_SLOT_MASK ($slot_mask) != SLOT_SIZE-1 ($(( slot_size - 1 )))"
  (( (slot_size & (slot_size - 1)) == 0 )) || gate_fail "LOWER_FN_TUPLE_ARR_SLOT_SIZE ($slot_size) is not a power of two"
  (( slot_size >= 2 * cap )) || gate_fail "LOWER_FN_TUPLE_ARR_SLOT_SIZE ($slot_size) < 2 * LOWER_FN_TUPLE_ARR_CAP ($(( 2 * cap )))"

  printf '%s %s %s %s %s %s %s %s %s %s' "$cap" "$hash_n" "$mask_n" "$nameid_n" "$scalar_n" "$slot_size" "$slot_mask" "$slot_n" "$chain_hash_n" "$chain_nameid_n"
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
  printf '{"status":"fail","reason":"positive control did not fire","metrics":{"total":10,"passed":0,"failed":1,"not_run":0}}\n' | gate_write_artifact "$ART"
  exit 1
fi
echo "control: sabotaged LOWER_FN_TUPLE_ARR_HASH extent rejected, as required"

vals="$(check "$LOWER")"
read -r cap hash_n mask_n nameid_n scalar_n slot_size slot_mask slot_n chain_hash_n chain_nameid_n <<<"$vals"
echo "LOWER_FN_TUPLE_ARR_CAP_OK: cap=$cap hash=$hash_n mask=$mask_n name_id=$nameid_n scalar=$scalar_n slot_size=$slot_size slot_mask=$slot_mask slot=$slot_n chain_hash=$chain_hash_n chain_name_id=$chain_nameid_n"
printf '{"status":"pass","metrics":{"cap":%s,"hash":%s,"mask":%s,"name_id":%s,"scalar":%s,"slot_size":%s,"slot_mask":%s,"slot":%s,"chain_hash":%s,"chain_name_id":%s,"total":10,"passed":10,"failed":0,"not_run":0}}\n' \
  "$cap" "$hash_n" "$mask_n" "$nameid_n" "$scalar_n" "$slot_size" "$slot_mask" "$slot_n" "$chain_hash_n" "$chain_nameid_n" | gate_write_artifact "$ART"
