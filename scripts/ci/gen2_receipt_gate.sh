#!/usr/bin/env bash
# gen2_receipt_gate.sh — CI Decision is red for a compiler state that has no
# gen2 fixed-point receipt.
#
# WHY THIS EXISTS
#
# The gen2 rung (Madaros compiling itself until gen2 == gen3) takes ~2 h 50 min
# and ~40 GiB. GitHub-hosted runners lost it again and again (#3741: three runs
# with the runner gone, no failed step), so it left the PR run in #3746 and runs
# on the DARWIN cluster as a Slurm job. The rule "a compiler PR needs a fixed-point
# receipt at its head before merge" would otherwise live only in a document,
# enforced by whoever remembers it. This gate moves it into CI Decision.
#
# WHAT IDENTIFIES A COMPILER STATE
#
# A receipt is keyed by its inputs key, not by the commit: sha256 over
# "<path> <git object id>" for every path in scripts/ci/gen2_receipt_inputs.txt
# (read at the same commit), plus that file's "param" lines. The list is
# fail-closed: compiler source, stdlib, the bootstrap ELFs, the gate, the build
# script and their helpers. A merge of main that touches none of them keeps the
# key and its receipt. A change to any of them needs a new job.
#
#   docs/evidence/gen2-fixpoint/<inputs key>.receipt
#
# key=value lines, every key exactly once:
#
#   inputs_key    64 hex, same as the file name
#   commit        40 hex: the commit the job checked out
#   slurm_job     digits
#   node          Slurm node name
#   rung          fixpoint
#   gate_rc       0
#   ir_parity     pass
#   gen1_sha256   64 hex
#   gen2_sha256   64 hex
#   gen3_sha256   64 hex, equal to gen2_sha256
#   maxrss_kb     digits
#   wall_s        digits
#   date          YYYY-MM-DD
#   receipt_doc   where the full receipt lives (prediction, and whether it held)
#
# The commit must be available: when it is not in the clone, the gate fetches
# it from origin, and if that fails the gate fails closed. Its inputs key,
# recomputed at that commit, must equal the receipt's.
#
# WHAT THIS IS NOT
#
# This is a signed-off attestation, checked for consistency. It is not a
# re-execution. The gate verifies that the receipt is well formed, that it
# records a fixed point with rc 0 and IR parity, and that the commit it names
# has exactly this state's inputs. It cannot verify that the job ran. That
# statement is made by the DCO-signed commit that adds the receipt, and the
# Slurm accounting (sacct -j <slurm_job>) is the record anyone can check it
# against.
#
# Usage: gen2_receipt_gate.sh [--selftest]
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "gen2_receipt_gate"

RECEIPT_DIR_REL="docs/evidence/gen2-fixpoint"
INPUTS_REL="scripts/ci/gen2_receipt_inputs.txt"
KEYS="inputs_key commit slurm_job node rung gate_rc ir_parity gen1_sha256 gen2_sha256 gen3_sha256 maxrss_kb wall_s date receipt_doc"

# inputs_key <repo> <rev> -> 64-hex key on stdout, rc != 0 if the list is missing.
inputs_key() {
  local repo="$1" rev="$2" list line oid n=0
  list="$(git -C "$repo" show "$rev:$INPUTS_REL" 2>/dev/null)" || return 1
  {
    while IFS= read -r line; do
      line="${line%%#*}"
      line="$(printf '%s' "$line" | sed -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//')"
      [[ -z "$line" ]] && continue
      n=$((n + 1))
      if [[ "$line" == param\ * ]]; then
        printf '%s\n' "$line"
      else
        oid="$(git -C "$repo" rev-parse --verify -q "$rev:$line" 2>/dev/null || echo absent)"
        printf '%s %s\n' "$line" "$oid"
      fi
    done <<<"$list"
    [[ $n -gt 0 ]] || echo "EMPTY-INPUT-LIST"
  } | sha256sum | cut -c1-64
}

# closure_check <repo> <rev> -> rc 0 iff the inputs list covers what its scripts
# use. Two checks: (1) every scripts/*.sh path named on a non-comment line of a
# listed .sh input is itself listed or lies under a listed directory; (2) every
# program in the IR-parity gate's PROGRAMS array is listed as
# tests/run-pass/<p>.sio. Each missing input is printed as
# GEN2_INPUTS_UNCOVERED <path> (named by <file>).
closure_check() {
  local repo="$1" rev="$2" list entries=() line f ref ok bad=0 e
  list="$(git -C "$repo" show "$rev:$INPUTS_REL" 2>/dev/null)" || { echo "GEN2_INPUTS_UNCOVERED $INPUTS_REL (missing)"; return 1; }
  while IFS= read -r line; do
    line="${line%%#*}"; line="$(printf '%s' "$line" | sed -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//')"
    [[ -z "$line" || "$line" == param\ * ]] && continue
    entries+=("$line")
  done <<<"$list"
  covered() {
    for e in "${entries[@]}"; do
      [[ "$1" == "$e" || "$1" == "$e"/* ]] && return 0
    done
    return 1
  }
  for f in "${entries[@]}"; do
    [[ "$f" == *.sh ]] || continue
    while IFS= read -r ref; do
      [[ -z "$ref" ]] && continue
      if ! covered "$ref"; then echo "GEN2_INPUTS_UNCOVERED $ref (named by $f)"; bad=1; fi
    done < <(git -C "$repo" show "$rev:$f" 2>/dev/null | grep -vE '^[[:space:]]*#' | grep -oE 'scripts/[A-Za-z0-9_./-]+\.sh' | sort -u)
  done
  local parity="scripts/ci/madaros_gen1_gen2_ir_parity_gate.sh" progs p
  if covered "$parity"; then
    progs="$(git -C "$repo" show "$rev:$parity" 2>/dev/null | sed -n 's/^PROGRAMS=(\(.*\))$/\1/p' | head -1)"
    if [[ -z "$progs" ]]; then echo "GEN2_INPUTS_UNCOVERED PROGRAMS (not found in $parity)"; bad=1; fi
    for p in $progs; do
      covered "tests/run-pass/$p.sio" || { echo "GEN2_INPUTS_UNCOVERED tests/run-pass/$p.sio (IR-parity program)"; bad=1; }
    done
  fi
  return $bad
}

# ensure_commit <repo> <commit>: rc 0 iff the commit object is (now) present.
ensure_commit() {
  local repo="$1" c="$2"
  git -C "$repo" cat-file -e "$c^{commit}" 2>/dev/null && return 0
  git -C "$repo" remote get-url origin >/dev/null 2>&1 || return 1
  git -C "$repo" fetch -q --depth 1 --no-tags origin "$c" >/dev/null 2>&1 || return 1
  git -C "$repo" cat-file -e "$c^{commit}" 2>/dev/null
}

# check_receipt <repo> <key> -> one diagnostic line, rc 0 iff valid.
check_receipt() {
  local repo="$1" key="$2"
  local f="$repo/$RECEIPT_DIR_REL/$key.receipt"
  if [[ ! -f "$f" ]]; then
    echo "GEN2_RECEIPT_MISSING inputs_key=$key expected=$RECEIPT_DIR_REL/$key.receipt"
    return 1
  fi
  local k n v
  declare -A val=()
  for k in $KEYS; do
    n=$(grep -c "^$k=" "$f" || true)
    if [[ "$n" -ne 1 ]]; then
      echo "GEN2_RECEIPT_BAD key=$k count=$n (need exactly 1)"
      return 1
    fi
    v=$(grep "^$k=" "$f" | head -1 | cut -d= -f2-)
    if [[ -z "$v" ]]; then
      echo "GEN2_RECEIPT_BAD key=$k empty"
      return 1
    fi
    val[$k]="$v"
  done
  [[ "${val[inputs_key]}" == "$key" ]] || { echo "GEN2_RECEIPT_BAD field_inputs_key=${val[inputs_key]} file_key=$key"; return 1; }
  [[ "${val[rung]}" == "fixpoint" ]] || { echo "GEN2_RECEIPT_BAD rung=${val[rung]}"; return 1; }
  [[ "${val[gate_rc]}" == "0" ]] || { echo "GEN2_RECEIPT_BAD gate_rc=${val[gate_rc]}"; return 1; }
  [[ "${val[ir_parity]}" == "pass" ]] || { echo "GEN2_RECEIPT_BAD ir_parity=${val[ir_parity]}"; return 1; }
  [[ "${val[commit]}" =~ ^[0-9a-f]{40}$ ]] || { echo "GEN2_RECEIPT_BAD commit=${val[commit]} (need 40 hex)"; return 1; }
  [[ "${val[slurm_job]}" =~ ^[0-9]+$ ]] || { echo "GEN2_RECEIPT_BAD slurm_job=${val[slurm_job]}"; return 1; }
  [[ "${val[maxrss_kb]}" =~ ^[0-9]+$ ]] || { echo "GEN2_RECEIPT_BAD maxrss_kb=${val[maxrss_kb]}"; return 1; }
  [[ "${val[wall_s]}" =~ ^[0-9]+$ ]] || { echo "GEN2_RECEIPT_BAD wall_s=${val[wall_s]}"; return 1; }
  [[ "${val[date]}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "GEN2_RECEIPT_BAD date=${val[date]}"; return 1; }
  for k in gen1_sha256 gen2_sha256 gen3_sha256; do
    [[ "${val[$k]}" =~ ^[0-9a-f]{64}$ ]] || { echo "GEN2_RECEIPT_BAD $k=${val[$k]} (need 64 hex)"; return 1; }
  done
  if [[ "${val[gen2_sha256]}" != "${val[gen3_sha256]}" ]]; then
    echo "GEN2_RECEIPT_NOT_FIXPOINT gen2=${val[gen2_sha256]} gen3=${val[gen3_sha256]}"
    return 1
  fi
  if ! ensure_commit "$repo" "${val[commit]}"; then
    echo "GEN2_RECEIPT_COMMIT_UNAVAILABLE commit=${val[commit]} (not in the clone and not fetchable from origin)"
    return 1
  fi
  local ckey; ckey="$(inputs_key "$repo" "${val[commit]}" || true)"
  if [[ "$ckey" != "$key" ]]; then
    echo "GEN2_RECEIPT_WRONG_COMMIT commit=${val[commit]} commit_inputs_key=${ckey:-none} receipt_inputs_key=$key"
    return 1
  fi
  echo "GEN2_RECEIPT_OK inputs_key=$key job=${val[slurm_job]} node=${val[node]} commit=${val[commit]} commit_inputs=verified"
  return 0
}

selftest() {
  local t; t=$(mktemp -d); trap 'rm -rf "$t"' RETURN
  local G=(-c user.name=t -c user.email=t@t)
  git -C "$t" init -q
  mkdir -p "$t/self-hosted" "$t/stdlib" "$t/scripts/ci" "$t/docs" "$t/$RECEIPT_DIR_REL"
  printf 'self-hosted\nstdlib\nscripts/ci/madaros_fixed_point_gate.sh\n%s\nparam expect=fixpoint\n' "$INPUTS_REL" >"$t/$INPUTS_REL"
  echo 'fn main() {}' >"$t/self-hosted/main.sio"
  echo 'fn lib() {}' >"$t/stdlib/lib.sio"
  echo 'echo gate' >"$t/scripts/ci/madaros_fixed_point_gate.sh"
  echo 'doc' >"$t/docs/a.md"
  git -C "$t" add -A; git -C "$t" "${G[@]}" commit -q -m one
  local c1 key1; c1=$(git -C "$t" rev-parse HEAD); key1=$(inputs_key "$t" HEAD)
  require_nonempty "$key1" "selftest inputs key"
  local H2=a2f0f811c2d4a2f0f811c2d4a2f0f811c2d4a2f0f811c2d4a2f0f811c2d4abcd
  write() { # write <file> [skip-key] [override k=v]...
    local f="$1" skip="$2"; shift 2
    {
      for kv in "inputs_key=$key1" "commit=$c1" "slurm_job=28684" "node=r740" "rung=fixpoint" \
                "gate_rc=0" "ir_parity=pass" \
                "gen1_sha256=82e085c0946482e085c0946482e085c0946482e085c0946482e085c09464aaaa" \
                "gen2_sha256=$H2" "gen3_sha256=$H2" \
                "maxrss_kb=39038888" "wall_s=10397" "date=2026-11-26" "receipt_doc=x.md"; do
        local k="${kv%%=*}" o
        [[ "$k" == "$skip" ]] && continue
        for o in "$@"; do [[ "${o%%=*}" == "$k" ]] && kv="$o"; done
        echo "$kv"
      done
    } >"$f"
  }
  local f="$t/$RECEIPT_DIR_REL/$key1.receipt" out pass=0
  expect() { # expect <ok|fail> <pattern> <label> [key]
    local want="$1" pat="$2" label="$3" key="${4:-$key1}" rc
    out=$(check_receipt "$t" "$key"); rc=$?
    if [[ "$want" == ok && $rc -ne 0 ]] || [[ "$want" == fail && $rc -eq 0 ]] || ! grep -q -- "$pat" <<<"$out"; then
      gate_fail "selftest control '$label': want=$want rc=$rc out=$out"
    fi
    pass=$((pass + 1))
  }
  rm -f "$f";                                   expect fail GEN2_RECEIPT_MISSING missing
  write "$f" "";                                expect ok   "commit_inputs=verified" valid
  write "$f" "" "gen3_sha256=${H2%????}ffff";   expect fail GEN2_RECEIPT_NOT_FIXPOINT not-fixpoint
  write "$f" "" "gen2_sha256=a2f0f811c2d4" "gen3_sha256=a2f0f811c2d4"; expect fail "need 64 hex" truncated-digest
  write "$f" "" "commit=${c1:0:12}";            expect fail "need 40 hex" short-commit
  write "$f" "" "rung=gen2";                    expect fail "rung=gen2" wrong-rung
  write "$f" "" "gate_rc=1";                    expect fail "gate_rc=1" failed-gate
  write "$f" "" "ir_parity=fail";               expect fail "ir_parity=fail" failed-parity
  write "$f" "slurm_job";                       expect fail "key=slurm_job count=0" missing-key
  write "$f" ""; echo "rung=fixpoint" >>"$f";   expect fail "key=rung count=2" duplicate-key
  write "$f" "" "inputs_key=$(printf '0%.0s' {1..64})"; expect fail field_inputs_key key-field
  write "$f" "" "commit=$(printf 'a%.0s' {1..40})"; expect fail GEN2_RECEIPT_COMMIT_UNAVAILABLE unavailable-commit
  # A docs-only commit keeps the key: the receipt for c1 still covers HEAD.
  write "$f" ""
  echo 'doc 2' >"$t/docs/a.md"; git -C "$t" "${G[@]}" commit -q -am docs
  [[ "$(inputs_key "$t" HEAD)" == "$key1" ]] || gate_fail "selftest: a docs-only change moved the inputs key"
  expect ok "commit_inputs=verified" docs-only-keeps-key "$(inputs_key "$t" HEAD)"
  # A stdlib change and a gate-script change each move the key: no receipt.
  echo 'fn lib() { 1 }' >"$t/stdlib/lib.sio"; git -C "$t" "${G[@]}" commit -q -am stdlib
  local key2; key2=$(inputs_key "$t" HEAD)
  [[ "$key2" != "$key1" ]] || gate_fail "selftest: a stdlib change kept the inputs key"
  expect fail GEN2_RECEIPT_MISSING stdlib-change-needs-receipt "$key2"
  echo 'echo gate v2' >"$t/scripts/ci/madaros_fixed_point_gate.sh"; git -C "$t" "${G[@]}" commit -q -am gate
  local key3; key3=$(inputs_key "$t" HEAD)
  [[ "$key3" != "$key2" ]] || gate_fail "selftest: a gate-script change kept the inputs key"
  expect fail GEN2_RECEIPT_MISSING gate-change-needs-receipt "$key3"
  # A receipt filed under the new key but naming the old commit: wrong inputs.
  local f3="$t/$RECEIPT_DIR_REL/$key3.receipt"
  write "$f3" "" "inputs_key=$key3"
  expect fail GEN2_RECEIPT_WRONG_COMMIT receipt-names-stale-commit "$key3"
  # Closure: a listed script naming an unlisted helper, and an unlisted
  # IR-parity program, are each refused; listing them passes.
  printf 'bash scripts/ci/helper_x.sh\n# scripts/ci/only_in_comment.sh\n' >"$t/scripts/ci/madaros_fixed_point_gate.sh"
  git -C "$t" "${G[@]}" commit -q -am uses-helper
  out=$(closure_check "$t" HEAD) && gate_fail "selftest: an unlisted helper passed the closure check"
  grep -q "GEN2_INPUTS_UNCOVERED scripts/ci/helper_x.sh" <<<"$out" || gate_fail "selftest: closure did not name the helper: $out"
  grep -q only_in_comment <<<"$out" && gate_fail "selftest: a comment mention was counted"
  pass=$((pass + 1))
  printf 'scripts/ci/helper_x.sh\nscripts/ci/madaros_gen1_gen2_ir_parity_gate.sh\n' >>"$t/$INPUTS_REL"
  echo x >"$t/scripts/ci/helper_x.sh"; printf 'PROGRAMS=(p_one)\n' >"$t/scripts/ci/madaros_gen1_gen2_ir_parity_gate.sh"
  git -C "$t" add -A; git -C "$t" "${G[@]}" commit -q -m list-helper
  out=$(closure_check "$t" HEAD) && gate_fail "selftest: an unlisted IR-parity program passed"
  grep -q "tests/run-pass/p_one.sio" <<<"$out" || gate_fail "selftest: closure did not name the program: $out"
  pass=$((pass + 1))
  printf 'tests/run-pass/p_one.sio\n' >>"$t/$INPUTS_REL"; git -C "$t" "${G[@]}" commit -q -am list-prog
  closure_check "$t" HEAD >/dev/null || gate_fail "selftest: a complete list failed the closure check: $(closure_check "$t" HEAD)"
  pass=$((pass + 1))
  require_min_count "$pass" 19 "selftest controls"
  echo "GEN2_RECEIPT_GATE_SELFTEST_PASS controls=$pass"
}

if [[ "${1:-}" == "--selftest" ]]; then
  selftest
  exit $?
fi

cd "$ROOT_DIR" || gate_fail "cannot cd to $ROOT_DIR"
closure_out="$(closure_check "$ROOT_DIR" HEAD)"; closure_rc=$?
if [[ $closure_rc -ne 0 ]]; then
  echo "$closure_out"
  gate_fail "$INPUTS_REL does not cover inputs that the listed scripts use (above); add them, which gives a new inputs key"
fi
KEY="$(inputs_key "$ROOT_DIR" HEAD || true)"
require_nonempty "$KEY" "inputs key of HEAD ($INPUTS_REL missing at HEAD?)"
echo "gen2 inputs key at HEAD: $KEY"
out=$(check_receipt "$ROOT_DIR" "$KEY"); rc=$?
echo "$out"
if [[ $rc -ne 0 ]]; then
  cat <<EOF
The compiler state at HEAD (gen2 inputs key $KEY) has no valid gen2
fixed-point receipt. Run the Slurm gen2 gate on a commit with these inputs
(templates: claude/research/jobs/gen2_gate_*.sbatch, full sha256 digests), and
commit $RECEIPT_DIR_REL/$KEY.receipt with the job's result.
EOF
  gate_fail "no valid gen2 fixed-point receipt for inputs key $KEY"
fi
gate_pass "gen2 fixed-point receipt present and consistent for inputs key $KEY"
