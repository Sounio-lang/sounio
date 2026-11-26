#!/usr/bin/env bash
# gen2_receipt_gate.sh — a compiler tree without a fixed-point receipt is red.
#
# WHY THIS EXISTS
#
# The gen2 rung (Madaros compiling itself until gen2 == gen3) takes ~2 h 50 min
# and 56–64 GiB. GitHub-hosted runners lost it again and again (#3741: three
# runs with the runner gone, no failed step). It left the PR run in #3746. It
# runs on the DARWIN cluster as a Slurm job, and the merge rule became: "a
# compiler PR needs a Slurm fixed-point receipt at its head before merge".
#
# A rule that only lives in a document is enforced by whoever remembers it. This
# gate moves it into CI Decision.
#
# WHAT IT CHECKS
#
# The identity of the compiler source is the git tree hash of `self-hosted/` at
# HEAD. A merge of main that touches only docs or CI keeps that hash, so it
# keeps its receipt. Any change under `self-hosted/` gives a new hash, and that
# hash needs its own receipt:
#
#   docs/evidence/gen2-fixpoint/<tree>.receipt
#
# with these key=value lines (one per key, no duplicates):
#
#   tree          the same 40-hex tree hash as the file name
#   commit        the commit the Slurm job checked out (≥ 10 hex)
#   slurm_job     the job id (digits)
#   node          the Slurm node name
#   rung          fixpoint
#   gen1_sha256   ≥ 12 hex
#   gen2_sha256   ≥ 12 hex
#   gen3_sha256   ≥ 12 hex, equal to gen2_sha256
#   maxrss_kb     digits
#   wall_s        digits
#   date          YYYY-MM-DD
#   receipt_doc   where the human-readable receipt lives
#
# When the commit is present in the local clone, its `self-hosted/` tree must
# be the receipt's tree. A shallow clone may not have it; that is reported,
# not passed off as checked.
#
# WHAT IT DOES NOT CHECK
#
# It cannot re-run the job. It checks that the receipt exists, belongs to this
# exact compiler source, and records a fixed point. Whoever commits a receipt
# is vouching for the job. The DCO sign-off on that commit is the vouching.
#
# Usage: gen2_receipt_gate.sh [--selftest]
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "gen2_receipt_gate"

RECEIPT_DIR_REL="docs/evidence/gen2-fixpoint"
KEYS="tree commit slurm_job node rung gen1_sha256 gen2_sha256 gen3_sha256 maxrss_kb wall_s date receipt_doc"

# check_receipt <repo> <tree> -> prints one diagnostic line, rc 0 iff valid.
check_receipt() {
  local repo="$1" tree="$2"
  local f="$repo/$RECEIPT_DIR_REL/$tree.receipt"
  if [[ ! -f "$f" ]]; then
    echo "GEN2_RECEIPT_MISSING tree=$tree expected=$RECEIPT_DIR_REL/$tree.receipt"
    return 1
  fi
  local k n v
  declare -A val=()
  for k in $KEYS; do
    n=$(grep -c "^$k=" "$f" || true)
    if [[ "$n" -ne 1 ]]; then
      echo "GEN2_RECEIPT_BAD tree=$tree key=$k count=$n (need exactly 1)"
      return 1
    fi
    v=$(grep "^$k=" "$f" | head -1 | cut -d= -f2-)
    if [[ -z "$v" ]]; then
      echo "GEN2_RECEIPT_BAD tree=$tree key=$k empty"
      return 1
    fi
    val[$k]="$v"
  done
  [[ "${val[tree]}" == "$tree" ]] || { echo "GEN2_RECEIPT_BAD tree=$tree field_tree=${val[tree]}"; return 1; }
  [[ "${val[rung]}" == "fixpoint" ]] || { echo "GEN2_RECEIPT_BAD tree=$tree rung=${val[rung]}"; return 1; }
  [[ "${val[commit]}" =~ ^[0-9a-f]{10,40}$ ]] || { echo "GEN2_RECEIPT_BAD tree=$tree commit=${val[commit]}"; return 1; }
  [[ "${val[slurm_job]}" =~ ^[0-9]+$ ]] || { echo "GEN2_RECEIPT_BAD tree=$tree slurm_job=${val[slurm_job]}"; return 1; }
  [[ "${val[maxrss_kb]}" =~ ^[0-9]+$ ]] || { echo "GEN2_RECEIPT_BAD tree=$tree maxrss_kb=${val[maxrss_kb]}"; return 1; }
  [[ "${val[wall_s]}" =~ ^[0-9]+$ ]] || { echo "GEN2_RECEIPT_BAD tree=$tree wall_s=${val[wall_s]}"; return 1; }
  [[ "${val[date]}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "GEN2_RECEIPT_BAD tree=$tree date=${val[date]}"; return 1; }
  for k in gen1_sha256 gen2_sha256 gen3_sha256; do
    [[ "${val[$k]}" =~ ^[0-9a-f]{12,64}$ ]] || { echo "GEN2_RECEIPT_BAD tree=$tree $k=${val[$k]}"; return 1; }
  done
  if [[ "${val[gen2_sha256]}" != "${val[gen3_sha256]}" ]]; then
    echo "GEN2_RECEIPT_NOT_FIXPOINT tree=$tree gen2=${val[gen2_sha256]} gen3=${val[gen3_sha256]}"
    return 1
  fi
  local ctree=""
  if git -C "$repo" cat-file -e "${val[commit]}^{commit}" 2>/dev/null; then
    ctree=$(git -C "$repo" rev-parse "${val[commit]}:self-hosted" 2>/dev/null || true)
    if [[ "$ctree" != "$tree" ]]; then
      echo "GEN2_RECEIPT_WRONG_COMMIT tree=$tree commit=${val[commit]} commit_tree=${ctree:-none}"
      return 1
    fi
    echo "GEN2_RECEIPT_OK tree=$tree job=${val[slurm_job]} node=${val[node]} commit=${val[commit]} commit_tree=checked"
  else
    echo "GEN2_RECEIPT_OK tree=$tree job=${val[slurm_job]} node=${val[node]} commit=${val[commit]} commit_tree=not-in-clone"
  fi
  return 0
}

selftest() {
  local t; t=$(mktemp -d); trap 'rm -rf "$t"' RETURN
  git -C "$t" init -q
  mkdir -p "$t/self-hosted" "$t/$RECEIPT_DIR_REL"
  echo 'fn main() {}' >"$t/self-hosted/main.sio"
  git -C "$t" add -A
  git -C "$t" -c user.name=t -c user.email=t@t commit -q -m one
  local c tree; c=$(git -C "$t" rev-parse HEAD); tree=$(git -C "$t" rev-parse HEAD:self-hosted)
  require_nonempty "$tree" "selftest tree hash"
  write() { # write <file> [skip-key] [override k=v]...
    local f="$1" skip="$2"; shift 2
    {
      for kv in "tree=$tree" "commit=$c" "slurm_job=28684" "node=r740" "rung=fixpoint" \
                "gen1_sha256=82e085c09464" "gen2_sha256=a2f0f811c2d4" "gen3_sha256=a2f0f811c2d4" \
                "maxrss_kb=39038888" "wall_s=10397" "date=2026-11-26" "receipt_doc=x.md"; do
        local k="${kv%%=*}" o
        [[ "$k" == "$skip" ]] && continue
        for o in "$@"; do [[ "${o%%=*}" == "$k" ]] && kv="$o"; done
        echo "$kv"
      done
    } >"$f"
  }
  local f="$t/$RECEIPT_DIR_REL/$tree.receipt" out pass=0
  expect() { # expect <ok|fail> <pattern> <label>
    local want="$1" pat="$2" label="$3" rc
    out=$(check_receipt "$t" "$tree"); rc=$?
    if [[ "$want" == ok && $rc -ne 0 ]] || [[ "$want" == fail && $rc -eq 0 ]] || ! grep -q "$pat" <<<"$out"; then
      gate_fail "selftest control '$label': want=$want rc=$rc out=$out"
    fi
    pass=$((pass + 1))
  }
  rm -f "$f";                                  expect fail GEN2_RECEIPT_MISSING missing
  write "$f" "";                               expect ok   "commit_tree=checked" valid
  write "$f" "" "gen3_sha256=ffffffffffff";    expect fail GEN2_RECEIPT_NOT_FIXPOINT not-fixpoint
  write "$f" "" "rung=gen2";                   expect fail "rung=gen2" wrong-rung
  write "$f" "slurm_job";                      expect fail "key=slurm_job count=0" missing-key
  write "$f" "" "tree=0000000000000000000000000000000000000000"; expect fail field_tree tree-field
  write "$f" ""; echo "rung=fixpoint" >>"$f";  expect fail "key=rung count=2" duplicate-key
  # A receipt naming a commit whose compiler source is different.
  echo 'fn main() { 1 }' >"$t/self-hosted/main.sio"
  git -C "$t" -c user.name=t -c user.email=t@t commit -q -am two
  local c2; c2=$(git -C "$t" rev-parse HEAD)
  write "$f" "" "commit=$c2";                  expect fail GEN2_RECEIPT_WRONG_COMMIT wrong-commit
  write "$f" "" "commit=abcdefabcdef";         expect ok   "commit_tree=not-in-clone" shallow
  require_min_count "$pass" 9 "selftest controls"
  echo "GEN2_RECEIPT_GATE_SELFTEST_PASS controls=$pass"
}

if [[ "${1:-}" == "--selftest" ]]; then
  selftest
  exit $?
fi

cd "$ROOT_DIR" || gate_fail "cannot cd to $ROOT_DIR"
TREE=$(git rev-parse HEAD:self-hosted 2>/dev/null || true)
require_nonempty "$TREE" "tree hash of self-hosted/ at HEAD"
out=$(check_receipt "$ROOT_DIR" "$TREE"); rc=$?
echo "$out"
if [[ $rc -ne 0 ]]; then
  cat <<EOF
The compiler source at HEAD (self-hosted/ tree $TREE) has no valid gen2
fixed-point receipt. Run the Slurm gen2 gate on this commit (templates:
claude/research/jobs/gen2_gate_*.sbatch) and commit
$RECEIPT_DIR_REL/$TREE.receipt with the job's result.
EOF
  gate_fail "no gen2 fixed-point receipt for self-hosted tree $TREE"
fi
gate_pass "gen2 fixed-point receipt present for self-hosted tree $TREE"
