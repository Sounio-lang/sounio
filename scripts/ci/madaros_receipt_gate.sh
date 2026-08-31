#!/usr/bin/env bash
# A binary that asserts its own identity must be telling the truth.
#
# artifacts/self-hosted/madaros.gate-receipt is TRACKED in git and claims a
# sha256, a source_commit and a gate result for whatever artifact it NAMES --
# today bin/madaros-linux-x86_64, not the path in this file's own name --
# which is NOT tracked (.gitignore:206). Measured 2026-08-04:
#
#     receipt sha256   5629c3a48b6c...    file sha256   6303ec70187b...
#     source_commit    96a303edd          1603 commits behind HEAD
#     created_utc      2026-07-11         file mtime    2026-07-30
#     readers          none               writers       none
#
# So the receipt described a different binary, and nothing noticed for weeks.
# Meanwhile ~82 files resolve that ELF as an oracle, scripts/install.sh:96-102
# PREFERS it over the committed bin/madaros-linux-x86_64, and the only version
# assertion anywhere is madaros_full_gate.sh grepping for the literal string
# "Madaros v0.80.0" — which the stale binary still prints.
#
# I used that file as an A/B control and published a false claim on the strength
# of it. A receipt nobody checks is not provenance; it is decoration that looks
# like provenance, which is worse than none.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "madaros_receipt_gate"

RECEIPT="$ROOT_DIR/artifacts/self-hosted/madaros.gate-receipt"

require_nonempty_file "$RECEIPT" "the tracked receipt is missing or empty"
require_text "madaros_full_gate_receipt_v1" "$RECEIPT"

field() {
  local k="$1" v
  v="$(awk -F= -v k="$k" '$1 == k { sub("^" k "=", ""); print; exit }' "$RECEIPT")"
  printf '%s' "$v"
}

claimed_sha="$(field sha256)"
claimed_commit="$(field source_commit)"
claimed_gate="$(field gate)"
claimed_result="$(field gate_result)"
claimed_artifact="$(field artifact)"

require_nonempty "$claimed_artifact" "receipt has no artifact= line — it does not say what it is a receipt FOR"
require_nonempty "$claimed_sha"    "receipt has no sha256= line"
require_nonempty "$claimed_commit" "receipt has no source_commit= line"
require_nonempty "$claimed_gate"   "receipt has no gate= line"
require_nonempty "$claimed_result" "receipt has no gate_result= line"

[[ "$claimed_result" == "pass" ]] \
  || gate_fail "receipt records gate_result=$claimed_result — a binary whose own gate did not pass must not be shipped"

# The gate it names must exist. `gate=` carries a bare filename.
gate_path="$ROOT_DIR/scripts/ci/$claimed_gate"
require_file "$gate_path" "receipt names gate '$claimed_gate' which does not exist at scripts/ci/"

# The commit it was built from should be an ancestor of HEAD. But CI checks out
# shallow (actions/checkout@v4 with no fetch-depth, ci.yml Contracts job), so an
# older commit is genuinely ABSENT from the clone. "I cannot see that commit" and
# "that commit is not an ancestor" are different facts and must not collapse into
# one verdict — collapsing an unknown into a failure is the same error as
# collapsing it into a pass, just in the safe-looking direction.
#
# The sha256 check below is the load-bearing one and works on any clone.
if ! git rev-parse --verify -q "${claimed_commit}^{commit}" >/dev/null 2>&1; then
  if [[ -f "$(git rev-parse --git-dir)/shallow" ]]; then
    echo "  receipt: source=$claimed_commit NOT VERIFIED — shallow clone, the commit is not in this checkout"
  else
    gate_fail "receipt source_commit=$claimed_commit is not a commit in this repository, and this is a full clone"
  fi
else
  if ! git merge-base --is-ancestor "$claimed_commit" HEAD 2>/dev/null; then
          # Not in this history. Two legitimate ways that happens, and neither is a
          # false claim:
          #
          #   squash merge      the branch collapsed into one new commit, so the id
          #                     the receipt names never lands
          #   written early     madaros_write_receipt.sh records HEAD when the
          #                     artifact is modified-but-uncommitted, and HEAD is by
          #                     definition the commit BEFORE the new binary. That
          #                     commit therefore carries the OLD blob.
          #
          # I first accepted only the squash shape, by requiring the named commit to
          # carry the same blob as HEAD. That failed main at e5e7ce8aff on a receipt
          # whose sha256 matched the shipped ELF byte for byte -- the second shape
          # above. Once a commit is outside this history its blob proves nothing
          # either way, so provenance is reported UNVERIFIED and the sha256 check
          # below is what stands. It always was the load-bearing one; making the
          # commit id load-bearing is what broke twice in one day.
          head_blob="$(git rev-parse -q --verify "HEAD:$claimed_artifact" 2>/dev/null || true)"
          claimed_blob="$(git rev-parse -q --verify "$claimed_commit:$claimed_artifact" 2>/dev/null || true)"
          if [[ -n "$claimed_blob" && "$claimed_blob" == "$head_blob" ]]; then
            echo "  receipt: source=$claimed_commit was rewritten out of this history (squash); it carried this exact artifact"
          else
            echo "  receipt: source=$claimed_commit is not in this history -- provenance UNVERIFIED"
            echo "  receipt: the sha256 comparison below is what stands"
          fi
          delivered="$(git log -1 --format=%H -- "$claimed_artifact" 2>/dev/null)"
          claimed_commit="${delivered:-$claimed_commit}"
        fi
  behind="$(git rev-list --count "${claimed_commit}..HEAD" 2>/dev/null || echo '?')"
  echo "  receipt: gate=$claimed_gate result=$claimed_result source=$claimed_commit (${behind} commits behind HEAD)"
fi

# Verify the artifact the receipt NAMES, whatever that is. A receipt that points
# at an absolute host path, or at a gitignored local build, is unverifiable on
# any other checkout — which is how this one stayed wrong for weeks. Requiring
# the subject to be a tracked, repo-relative path is what makes the claim
# portable and therefore checkable.
case "$claimed_artifact" in
  /*) gate_fail "receipt names an ABSOLUTE path ($claimed_artifact) — a receipt tied to one machine's filesystem cannot be verified anywhere else" ;;
esac

BINARY="$ROOT_DIR/$claimed_artifact"
if ! git ls-files --error-unmatch "$claimed_artifact" >/dev/null 2>&1; then
  gate_fail "receipt names '$claimed_artifact', which is not tracked in git. A receipt for a file that is not in the repository is a claim nobody can check — point it at the committed binary (bin/madaros-linux-x86_64)."
fi
require_nonempty_file "$BINARY" "receipt names '$claimed_artifact' but there is no such file"

actual_sha="$(sha256sum "$BINARY" | awk '{print $1}')"
require_nonempty "$actual_sha" "sha256sum produced nothing for $BINARY"

if [[ "$actual_sha" != "$claimed_sha" ]]; then
  echo "  receipt claims $claimed_sha"
  echo "  file is       $actual_sha"
  echo
  echo "  This ELF is not the one the receipt was written for. Whatever the receipt"
  echo "  names is resolved as an oracle by ~82 scripts, so a stale receipt lets every"
  echo "  one of them attest a build nobody measured. Re-run the gate against the ELF"
  echo "  that is actually there, then re-emit the receipt:"
  echo "    bash scripts/ci/madaros_full_gate.sh && bash scripts/ci/madaros_write_receipt.sh"
  gate_fail "$claimed_artifact does not match its own receipt"
fi

gate_pass "binary matches its receipt ($actual_sha)"
