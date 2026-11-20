#!/bin/sh
# Reproducible build, seed-a parity, differential, and TMR verification gate
# for P4-3 (`sporos_vm`: post-admission native x86-64 Sporos bytecode VM).
#
# Root A framing: Root A is MUSE's hand-written checker
# (`bootstrap/roots/sporos/sporos-stage1.hex`); `spr1_check` is an independent
# second check (not Root A or Root A's trust base).
#
# Usage:
#   bootstrap/model/sporos_vm_gate.sh [--skip-lean] [--cases N] [--seed S] [N_SEEDS]
# Optional environment variables:
#   CXX               C++23 compiler override
#   LAKE              lake binary override
#   ELAN_TOOLCHAIN    Lean toolchain override (defaults to leanprover/lean4:v4.33.0
#                     when installed in elan, else formal/lean-toolchain)
#   SPOROS_EX_DIR     pre-generated directory of example *.spb files
#   SPOROS_SKIP_LEAN  set to 1 to skip the Lean SporosVmCheckBytes check
#   FUZZ_SEED         seed offset multiplier for generated programs (default 0)
set -eu

N=1000
SKIP_LEAN="${SPOROS_SKIP_LEAN:-0}"
SEED="${FUZZ_SEED:-0}"
POS_N=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-lean)
      SKIP_LEAN=1
      shift
      ;;
    --cases)
      [ "$#" -ge 2 ] || { echo "sporos_vm_gate: --cases requires an argument" >&2; exit 2; }
      N="$2"
      shift 2
      ;;
    --cases=*)
      N="${1#--cases=}"
      shift
      ;;
    --seed)
      [ "$#" -ge 2 ] || { echo "sporos_vm_gate: --seed requires an argument" >&2; exit 2; }
      SEED="$2"
      shift 2
      ;;
    --seed=*)
      SEED="${1#--seed=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "sporos_vm_gate: unknown option: $1" >&2
      exit 2
      ;;
    *)
      if [ -z "$POS_N" ]; then
        POS_N="$1"
        N="$1"
      else
        echo "sporos_vm_gate: unexpected argument: $1" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

SEED_OFFSET=0
if [ "$SEED" -ne 0 ]; then
  SEED_OFFSET=$((SEED * N))
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

if [ -d "$HOME/.elan/bin" ]; then
  case ":$PATH:" in
    *":$HOME/.elan/bin:"*) ;;
    *) PATH="$HOME/.elan/bin:$PATH"; export PATH ;;
  esac
fi

# Default ELAN_TOOLCHAIN: prefer leanprover/lean4:v4.33.0 when installed in elan,
# otherwise fall back to formal/lean-toolchain.
if [ -z "${ELAN_TOOLCHAIN:-}" ]; then
  if command -v elan >/dev/null 2>&1 &&
     elan toolchain list 2>/dev/null | grep -Eq '^leanprover/lean4:v4\.33\.0([[:space:]]|$)'; then
    ELAN_TOOLCHAIN="leanprover/lean4:v4.33.0"
    export ELAN_TOOLCHAIN
  elif [ -d "$HOME/.elan/toolchains/leanprover--lean4---v4.33.0" ]; then
    ELAN_TOOLCHAIN="leanprover/lean4:v4.33.0"
    export ELAN_TOOLCHAIN
  fi
fi

pick_cxx() {
  cat <<'EOF' > "$W/probe.cpp"
#include <cstdint>
int main() { return 0; }
EOF
  if [ -n "${CXX:-}" ]; then
    for std in -std=c++23 -std=c++2b; do
      if $CXX $std -O2 -Wall -Wextra -Werror -o "$W/probe" "$W/probe.cpp" >/dev/null 2>&1; then
        printf '%s %s' "$CXX" "$std"
        return 0
      fi
    done
  fi
  for cand in \
    "g++ -std=c++23" \
    "c++ -std=c++23" \
    "/usr/bin/clang++ -std=c++2b" \
    "/Library/Developer/CommandLineTools/usr/bin/clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++2b" \
    "clang++ -std=c++2b"; do
    if $cand -O2 -Wall -Wextra -Werror -o "$W/probe" "$W/probe.cpp" >/dev/null 2>&1; then
      printf '%s' "$cand"
      return 0
    fi
  done
  echo "sporos_vm_gate: no working C++23 compiler found" >&2
  return 1
}

pick_lake() {
  if [ -n "${LAKE:-}" ] && [ -x "${LAKE}" ]; then
    printf '%s' "$LAKE"
    return 0
  fi
  if command -v lake >/dev/null 2>&1; then
    command -v lake
    return 0
  fi
  if [ -x "$HOME/.elan/bin/lake" ]; then
    printf '%s' "$HOME/.elan/bin/lake"
    return 0
  fi
  return 1
}

CXX_CMD="$(pick_cxx)"
VM_GEN_SRC="$ROOT/bootstrap/model/sporos_vm_gen.cpp"
VM_FUZZ_SRC="$ROOT/bootstrap/model/sporos_vm_fuzz.cpp"
VM_BIN="$ROOT/bootstrap/model/sporos_vm"
VM_HEX="$ROOT/bootstrap/model/sporos_vm.hex"
VM_LEAN="$ROOT/formal/Bootstrap/SporosVmBinData.lean"

# ---------------------------------------------------------------------------
# Phase 1: C++23 -Wall -Wextra -Werror build + 2-run determinism + parity
# ---------------------------------------------------------------------------
$CXX_CMD -O2 -Wall -Wextra -Werror -o "$W/vm_gen" "$VM_GEN_SRC"
$CXX_CMD -O2 -Wall -Wextra -Werror -o "$W/fuzz" "$VM_FUZZ_SRC"
mkdir -p "$W/g1" "$W/g2"
"$W/vm_gen" "$W/g1" >/dev/null
"$W/vm_gen" "$W/g2" >/dev/null

for f in sporos_vm sporos_vm.hex SporosVmBinData.lean; do
  if ! cmp -s "$W/g1/$f" "$W/g2/$f"; then
    echo "sporos_vm_gate: FAIL 2-run nondeterminism in $f" >&2
    exit 1
  fi
done

"$W/fuzz" 1 > "$W/f1.spb"
"$W/fuzz" 1 > "$W/f2.spb"
if ! cmp -s "$W/f1.spb" "$W/f2.spb"; then
  echo "sporos_vm_gate: FAIL 2-run nondeterminism in sporos_vm_fuzz" >&2
  exit 1
fi

cmp -s "$W/g1/sporos_vm" "$VM_BIN" || { echo "sporos_vm_gate: FAIL sporos_vm mismatch vs committed" >&2; exit 1; }
cmp -s "$W/g1/sporos_vm.hex" "$VM_HEX" || { echo "sporos_vm_gate: FAIL sporos_vm.hex mismatch vs committed" >&2; exit 1; }
cmp -s "$W/g1/SporosVmBinData.lean" "$VM_LEAN" || { echo "sporos_vm_gate: FAIL SporosVmBinData.lean mismatch vs committed" >&2; exit 1; }

vm_bytes="$(wc -c < "$VM_BIN" | LC_ALL=C awk '{print $1}')"
echo "sporos_vm_gate: phase1 determinism=ok committed_parity=ok fuzz_build=ok bytes=$vm_bytes"

LAKE_BIN="$(pick_lake 2>/dev/null || true)"
if [ "$SKIP_LEAN" != "1" ] && [ "${SPOROS_SKIP_LEAN:-0}" != "1" ] && [ -n "$LAKE_BIN" ] && [ -f "$ROOT/formal/Bootstrap/SporosVmCheckBytes.lean" ]; then
  FORMAL_DIR="$ROOT/formal"
  DEC_OLEAN="$FORMAL_DIR/.lake/build/lib/lean/Bootstrap/X86Dec.olean"
  VM_OLEAN="$FORMAL_DIR/.lake/build/lib/lean/Bootstrap/X86Vm.olean"
  BIN_OLEAN="$FORMAL_DIR/.lake/build/lib/lean/Bootstrap/SporosVmBinData.olean"
  ( cd "$FORMAL_DIR" && "$LAKE_BIN" build Bootstrap.X86Dec Bootstrap.X86Int Bootstrap.X86Env Bootstrap.PDist >/dev/null )
  mkdir -p "$FORMAL_DIR/.lake/build/lib/lean/Bootstrap"
  if [ ! -f "$VM_OLEAN" ] || [ "$FORMAL_DIR/Bootstrap/X86Vm.lean" -nt "$VM_OLEAN" ] || [ "$DEC_OLEAN" -nt "$VM_OLEAN" ]; then
    ( cd "$FORMAL_DIR" && "$LAKE_BIN" env lean -R . -o "$VM_OLEAN" Bootstrap/X86Vm.lean )
  fi
  if [ ! -f "$BIN_OLEAN" ] || [ "$VM_LEAN" -nt "$BIN_OLEAN" ] || [ "$VM_OLEAN" -nt "$BIN_OLEAN" ]; then
    ( cd "$FORMAL_DIR" && "$LAKE_BIN" env lean -R . -o "$BIN_OLEAN" Bootstrap/SporosVmBinData.lean )
  fi
  ( cd "$FORMAL_DIR" && "$LAKE_BIN" env lean -R . Bootstrap/SporosVmCheckBytes.lean )
fi

# ---------------------------------------------------------------------------
# Phase 2: Native Linux x86-64 execution (seed-a, differential, TMR controls)
# ---------------------------------------------------------------------------
if [ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]; then
  echo "sporos_vm_gate: non-Linux-x86_64 host ($(uname -s)-$(uname -m)); Phase 2 deferred to Linux x86-64"
  exit 0
fi

# 2a. seed-a self-reproduction and Format A assembly parity
A_DIR="$ROOT/bootstrap/roots/a"
cp "$A_DIR/seed-a" "$W/seed-a"
chmod +x "$W/seed-a"
"$W/seed-a" < "$A_DIR/seed-a.hex" > "$W/seed-a.self"
cmp -s "$W/seed-a.self" "$A_DIR/seed-a" || { echo "sporos_vm_gate: FAIL seed-a self-reproduction" >&2; exit 1; }
"$W/seed-a" < "$VM_HEX" > "$W/vm.from_hex"
cmp -s "$W/vm.from_hex" "$VM_BIN" || { echo "sporos_vm_gate: FAIL seed-a < sporos_vm.hex != sporos_vm" >&2; exit 1; }
chmod +x "$W/vm.from_hex"
echo "sporos_vm_gate: phase2a seed_a_self=ok seed_a_vm_parity=ok"

# 2b. Build reference C++23 interpreter and sporos_gen (fuzz already built in Phase 1)
$CXX_CMD -O2 -Wall -Wextra -o "$W/fast" "$ROOT/bootstrap/model/sporos_fast.cpp"
$CXX_CMD -O2 -Wall -Wextra -o "$W/gen" "$ROOT/bootstrap/model/sporos_gen.cpp"

# Prepare example programs (pre-supplied SPOROS_EX_DIR or built via lake)
EX_DIR="$W/ex"
mkdir -p "$EX_DIR"
if [ -n "${SPOROS_EX_DIR:-}" ] && [ -d "${SPOROS_EX_DIR}" ]; then
  for ep in "$SPOROS_EX_DIR"/*.spb; do
    [ -f "$ep" ] && cp "$ep" "$EX_DIR/"
  done
elif [ -n "$LAKE_BIN" ]; then
  ( cd "$ROOT/formal" && "$LAKE_BIN" build sporos-examples >/dev/null )
  "$ROOT/formal/.lake/build/bin/sporos-examples" "$EX_DIR" >/dev/null
fi

mk_det_in() {
  len="$1"
  out="$2"
  seed="$3"
  if [ "$len" -le 0 ]; then
    : > "$out"
  else
    LC_ALL=C awk -v n="$len" -v s="$seed" 'BEGIN {
      for (i = 0; i < n; i++) printf "%c", ((i * 73 + s * 19) % 255) + 1
    }' > "$out"
  fi
}

mk_dem_in() {
  len="$1"
  out="$2"
  seed="$3"
  LC_ALL=C awk -v n="$len" -v s="$seed" 'BEGIN {
    printf "%c", 0
    for (i = 0; i < n; i++) printf "%c", ((i * 73 + s * 19) % 255) + 1
  }' > "$out"
}

norm_err() {
  sed 's/^sporos_fast: [^:]*: /sporos: X: /; s/^sporos_vm: [^:]*: /sporos: X: /; s/^sporos: [^:]*: not/sporos: X: not/; s/^sporos: [^:]*: refused/sporos: X: refused/' "$1" > "$2"
}

cmp_pair() {
  prog="$1"
  inp="$2"
  bin="$3"
  rc1=0; rc2=0
  "$W/fast" "$prog" < "$inp" > "$W/o1" 2> "$W/e1_raw" || rc1=$?
  "$bin" "$prog" < "$inp" > "$W/o2" 2> "$W/e2_raw" || rc2=$?
  norm_err "$W/e1_raw" "$W/e1"
  norm_err "$W/e2_raw" "$W/e2"
  [ "$rc1" = "$rc2" ] && cmp -s "$W/o1" "$W/o2" && cmp -s "$W/e1" "$W/e2"
}

ex_runs=0
vec_runs=0
gen_runs=0
fuzz_runs=0
disagreements=0

# Examples x 4 deterministic inputs
for p in "$EX_DIR"/*.spb; do
  [ -f "$p" ] || continue
  k=0
  while [ "$k" -lt 4 ]; do
    mk_det_in $((k * 37)) "$W/in" "$k"
    ex_runs=$((ex_runs + 1))
    if ! cmp_pair "$p" "$W/in" "$W/vm.from_hex"; then
      disagreements=$((disagreements + 1))
      echo "DIFF example: $p len=$((k * 37))" >&2
    fi
    k=$((k + 1))
  done
done

# Accept/reject vectors (+ truncated and byte-flipped variants)
: > "$W/empty_in"
for v in "$ROOT"/bootstrap/roots/sporos/vectors/*.spb; do
  [ -f "$v" ] || continue
  vec_runs=$((vec_runs + 1))
  if ! cmp_pair "$v" "$W/empty_in" "$W/vm.from_hex"; then
    disagreements=$((disagreements + 1))
    echo "DIFF vector: $v" >&2
  fi
  sz="$(wc -c < "$v" | LC_ALL=C awk '{print $1}')"
  if [ "$sz" -gt 1 ]; then
    dd if="$v" of="$W/v.trunc.spb" bs=1 count=$((sz - 1)) 2>/dev/null
    vec_runs=$((vec_runs + 1))
    if ! cmp_pair "$W/v.trunc.spb" "$W/empty_in" "$W/vm.from_hex"; then
      disagreements=$((disagreements + 1))
      echo "DIFF vector trunc: $v" >&2
    fi
  fi
  if [ "$sz" -gt 20 ]; then
    cp "$v" "$W/v.flip.spb"
    printf '\377' | dd of="$W/v.flip.spb" bs=1 seek=20 conv=notrunc 2>/dev/null
    vec_runs=$((vec_runs + 1))
    if ! cmp_pair "$W/v.flip.spb" "$W/empty_in" "$W/vm.from_hex"; then
      disagreements=$((disagreements + 1))
      echo "DIFF vector flip: $v" >&2
    fi
  fi
done

# N sporos_gen programs
s=1
while [ "$s" -le "$N" ]; do
  prog_seed=$((SEED_OFFSET + s))
  "$W/gen" "$prog_seed" > "$W/p.spb"
  mk_det_in $((prog_seed % 9)) "$W/in" "$prog_seed"
  gen_runs=$((gen_runs + 1))
  if ! cmp_pair "$W/p.spb" "$W/in" "$W/vm.from_hex"; then
    disagreements=$((disagreements + 1))
    echo "DIFF gen seed=$prog_seed" >&2
  fi
  s=$((s + 1))
done

# N sporos_vm_fuzz high-coverage programs (90% clean execution + 10% demand-refusal path)
s=1
while [ "$s" -le "$N" ]; do
  prog_seed=$((SEED_OFFSET + s))
  "$W/fuzz" "$prog_seed" > "$W/pf.spb"
  if [ $((prog_seed % 10)) -eq 0 ]; then
    mk_dem_in $((prog_seed % 9)) "$W/in" "$prog_seed"
  else
    mk_det_in $((prog_seed % 9)) "$W/in" "$prog_seed"
  fi
  fuzz_runs=$((fuzz_runs + 1))
  if ! cmp_pair "$W/pf.spb" "$W/in" "$W/vm.from_hex"; then
    disagreements=$((disagreements + 1))
    echo "DIFF fuzz seed=$prog_seed" >&2
  fi
  s=$((s + 1))
done

# ---------------------------------------------------------------------------
# 2c. Negative controls (6 non-equivalent mutants) & TMR perturbation checks
# ---------------------------------------------------------------------------
build_variant() {
  tag="$1"
  src="$2"
  mkdir -p "$W/$tag"
  $CXX_CMD -O2 -Wall -Wextra -Werror -o "$W/${tag}_gen" "$src"
  "$W/${tag}_gen" "$W/$tag" >/dev/null
  chmod +x "$W/$tag/sporos_vm"
}

# Negative control M1: provenance mutant (drop +1 on q in L_prim_prologue)
sed 's/A\.i(e_aluI(AluOp::Add, R10, 1), "q = qa + pc + 1");/A.i(e_aluI(AluOp::Add, R10, 0), "q = qa + pc + 0");/' \
  "$VM_GEN_SRC" > "$W/vm_m1.cpp"
if cmp -s "$VM_GEN_SRC" "$W/vm_m1.cpp"; then
  echo "sporos_vm_gate: M1 mutation was a no-op" >&2; exit 1
fi
build_variant m1 "$W/vm_m1.cpp"

# Negative control M2: ALU mutant (P_add uses Sub instead of Add)
LC_ALL=C awk '
  /A\.label\("P_add"\);/ { in_add = 1 }
  in_add && /AluOp::Add/ { sub(/AluOp::Add/, "AluOp::Sub"); in_add = 0; changed = 1 }
  { print }
  END { if (!changed) exit 1 }
' "$VM_GEN_SRC" > "$W/vm_m2.cpp"
if cmp -s "$VM_GEN_SRC" "$W/vm_m2.cpp"; then
  echo "sporos_vm_gate: M2 mutation was a no-op" >&2; exit 1
fi
build_variant m2 "$W/vm_m2.cpp"

# Negative control M3: signed-vs-unsigned comparison mutant (P_lts uses Cc::B instead of Cc::L)
sed 's/emit_cmp_prim("P_lts", Cc::L);/emit_cmp_prim("P_lts", Cc::B);/' \
  "$VM_GEN_SRC" > "$W/vm_m3.cpp"
if cmp -s "$VM_GEN_SRC" "$W/vm_m3.cpp"; then
  echo "sporos_vm_gate: M3 mutation was a no-op" >&2; exit 1
fi
build_variant m3 "$W/vm_m3.cpp"

# Negative control M4: memory load provenance (pm) accumulation dropped in emit_ld_prim
LC_ALL=C awk '
  /auto emit_ld_prim = / { in_ld = 1 }
  in_ld && /A\.i\(e_alu\(AluOp::Add, R10, R11\)\);/ {
    sub(/A\.i\(e_alu\(AluOp::Add, R10, R11\)\);/, "// M4 mutant: dropped pm += memp[addr+i]")
    in_ld = 0
    changed = 1
  }
  { print }
  END { if (!changed) exit 1 }
' "$VM_GEN_SRC" > "$W/vm_m4.cpp"
if cmp -s "$VM_GEN_SRC" "$W/vm_m4.cpp"; then
  echo "sporos_vm_gate: M4 mutation was a no-op" >&2; exit 1
fi
build_variant m4 "$W/vm_m4.cpp"

# Negative control M5: P_demand threshold refusal check bypassed
LC_ALL=C awk '
  /A\.label\("P_demand"\);/ { in_dem = 1 }
  in_dem && /A\.jcc\(Cc::NE, "L_fail_demand", "", true\);/ {
    sub(/A\.jcc\(Cc::NE, "L_fail_demand", "", true\);/, "// M5 mutant: bypassed demand refusal check")
    in_dem = 0
    changed = 1
  }
  { print }
  END { if (!changed) exit 1 }
' "$VM_GEN_SRC" > "$W/vm_m5.cpp"
if cmp -s "$VM_GEN_SRC" "$W/vm_m5.cpp"; then
  echo "sporos_vm_gate: M5 mutation was a no-op" >&2; exit 1
fi
build_variant m5 "$W/vm_m5.cpp"

# TMR1: perturb replica 2 (XOR every qword of code2 with 0x55) -> masked (0 diffs)
LC_ALL=C awk '
  /auto mk_copy = \[&\]\(i32 off_dst\) \{/ {
    print "  auto mk_copy = [&](i32 off_dst, i32 xmask = 0) {"
    next
  }
  /A\.i\(e_st_idx\(MemW::W64, RAX, RDX, 1, 0, R8\)\);/ {
    print "    if (xmask != 0) A.i(e_aluI(AluOp::Xor, R8, xmask));"
    print
    next
  }
  /mk_copy\(OFF_CODE2\);/ {
    print "  mk_copy(OFF_CODE2, 0x55);"
    changed = 1
    next
  }
  { print }
  END { if (!changed) exit 1 }
' "$VM_GEN_SRC" > "$W/vm_tmr1.cpp"
if cmp -s "$VM_GEN_SRC" "$W/vm_tmr1.cpp"; then
  echo "sporos_vm_gate: TMR1 mutation was a no-op" >&2; exit 1
fi
build_variant tmr1 "$W/vm_tmr1.cpp"

# TMR3: perturb replica 2 (XOR 0x55) AND replica 3 (XOR 0x2a) -> 3-way split exits 3
sed 's/mk_copy(OFF_CODE3);/mk_copy(OFF_CODE3, 0x2a);/' "$W/vm_tmr1.cpp" > "$W/vm_tmr3.cpp"
if cmp -s "$W/vm_tmr1.cpp" "$W/vm_tmr3.cpp"; then
  echo "sporos_vm_gate: TMR3 mutation was a no-op" >&2; exit 1
fi
build_variant tmr3 "$W/vm_tmr3.cpp"

# Negative control M_novote: replica 2 perturbed AND L_vm_vote patched to dispatch replica 2 directly
sed 's/0x48, 0x39, 0xd8,  \/\/ cmp rax, rbx/0x48, 0x89, 0xd8,  \/\/ mov rax, rbx (patched voter)/; s/0x74, 0x0d,        \/\/ je  +0x0d/0xeb, 0x0d,        \/\/ jmp +0x0d (patched voter)/' \
  "$W/vm_tmr1.cpp" > "$W/vm_novote.cpp"
if cmp -s "$W/vm_tmr1.cpp" "$W/vm_novote.cpp"; then
  echo "sporos_vm_gate: novote mutation was a no-op" >&2; exit 1
fi
build_variant novote "$W/vm_novote.cpp"

neg_runs=50
tmr1_gen_runs=100
tmr1_fuzz_runs=100
tmr1_ex_runs=0
m1_caught=0
m2_caught=0
m3_caught=0
m4_caught=0
m5_caught=0
novote_caught=0
tmr1_diffs=0
tmr3_exit3=0

# TMR1 over examples
for p in "$EX_DIR"/*.spb; do
  [ -f "$p" ] || continue
  mk_det_in 37 "$W/in" 1
  tmr1_ex_runs=$((tmr1_ex_runs + 1))
  if ! cmp_pair "$p" "$W/in" "$W/tmr1/sporos_vm"; then
    tmr1_diffs=$((tmr1_diffs + 1))
  fi
done

# TMR1 over 100 sporos_gen seeds
s=1
while [ "$s" -le "$tmr1_gen_runs" ]; do
  prog_seed=$((SEED_OFFSET + s))
  "$W/gen" "$prog_seed" > "$W/p.spb"
  mk_det_in $((prog_seed % 9)) "$W/in" "$prog_seed"
  if ! cmp_pair "$W/p.spb" "$W/in" "$W/tmr1/sporos_vm"; then
    tmr1_diffs=$((tmr1_diffs + 1))
  fi
  s=$((s + 1))
done

# TMR1 (100 seeds), TMR3 (50 seeds), and 6-mutant negative-control sweep (50 seeds) over sporos_vm_fuzz
s=1
while [ "$s" -le "$tmr1_fuzz_runs" ]; do
  prog_seed=$((SEED_OFFSET + s))
  "$W/fuzz" "$prog_seed" > "$W/pf.spb"
  mk_det_in $((prog_seed % 9)) "$W/in" "$prog_seed"
  if ! cmp_pair "$W/pf.spb" "$W/in" "$W/tmr1/sporos_vm"; then
    tmr1_diffs=$((tmr1_diffs + 1))
  fi
  if [ "$s" -le "$neg_runs" ]; then
    mk_dem_in $((prog_seed % 9)) "$W/in_dem" "$prog_seed"
    if ! cmp_pair "$W/pf.spb" "$W/in_dem" "$W/tmr1/sporos_vm"; then
      tmr1_diffs=$((tmr1_diffs + 1))
    fi
    if ! cmp_pair "$W/pf.spb" "$W/in" "$W/m1/sporos_vm" || ! cmp_pair "$W/pf.spb" "$W/in_dem" "$W/m1/sporos_vm"; then
      m1_caught=$((m1_caught + 1))
    fi
    if ! cmp_pair "$W/pf.spb" "$W/in" "$W/m2/sporos_vm" || ! cmp_pair "$W/pf.spb" "$W/in_dem" "$W/m2/sporos_vm"; then
      m2_caught=$((m2_caught + 1))
    fi
    if ! cmp_pair "$W/pf.spb" "$W/in" "$W/m3/sporos_vm" || ! cmp_pair "$W/pf.spb" "$W/in_dem" "$W/m3/sporos_vm"; then
      m3_caught=$((m3_caught + 1))
    fi
    if ! cmp_pair "$W/pf.spb" "$W/in" "$W/m4/sporos_vm" || ! cmp_pair "$W/pf.spb" "$W/in_dem" "$W/m4/sporos_vm"; then
      m4_caught=$((m4_caught + 1))
    fi
    if ! cmp_pair "$W/pf.spb" "$W/in" "$W/m5/sporos_vm" || ! cmp_pair "$W/pf.spb" "$W/in_dem" "$W/m5/sporos_vm"; then
      m5_caught=$((m5_caught + 1))
    fi
    if ! cmp_pair "$W/pf.spb" "$W/in" "$W/novote/sporos_vm" || ! cmp_pair "$W/pf.spb" "$W/in_dem" "$W/novote/sporos_vm"; then
      novote_caught=$((novote_caught + 1))
    fi
    rc3=0
    "$W/tmr3/sporos_vm" "$W/pf.spb" < "$W/in" >/dev/null 2>&1 || rc3=$?
    if [ "$rc3" = 3 ]; then
      tmr3_exit3=$((tmr3_exit3 + 1))
    fi
  fi
  s=$((s + 1))
done

total_runs=$((ex_runs + vec_runs + gen_runs + fuzz_runs))
tmr1_total=$((tmr1_ex_runs + tmr1_gen_runs + tmr1_fuzz_runs))
echo "sporos_vm_gate: ex_runs=$ex_runs vec_runs=$vec_runs gen_runs=$gen_runs fuzz_runs=$fuzz_runs total_runs=$total_runs disagreements=$disagreements tmr1_runs=$tmr1_total tmr1_diffs=$tmr1_diffs tmr3_exit3=$tmr3_exit3/$neg_runs m1_caught=$m1_caught/$neg_runs m2_caught=$m2_caught/$neg_runs m3_caught=$m3_caught/$neg_runs m4_caught=$m4_caught/$neg_runs m5_caught=$m5_caught/$neg_runs novote_caught=$novote_caught/$neg_runs"
[ "$disagreements" = 0 ] &&
  [ "$tmr1_diffs" = 0 ] &&
  [ "$tmr3_exit3" = "$neg_runs" ] &&
  [ "$m1_caught" = "$neg_runs" ] &&
  [ "$m2_caught" = "$neg_runs" ] &&
  [ "$m3_caught" = "$neg_runs" ] &&
  [ "$m4_caught" = "$neg_runs" ] &&
  [ "$m5_caught" = "$neg_runs" ] &&
  [ "$novote_caught" = "$neg_runs" ]
