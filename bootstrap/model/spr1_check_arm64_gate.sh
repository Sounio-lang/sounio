#!/bin/sh
# Tri-Silicon verification gate for Pillar P4-5 (`spr1_check_arm64`).
#
# Usage:
#   bootstrap/model/spr1_check_arm64_gate.sh [--cases N] [--seed S] [--skip-lean]
#
# Modes (auto-selected by host OS/arch):
#   - All hosts: C++23 generator build (-std=c++23 -O2 -Wall -Wextra -Werror),
#     2-run determinism across spr1_check_arm64, .bin, .hex, and
#     SporosCheckArm64BinData.lean (when formal/ is present), plus Lean
#     SporosCheckArm64CheckBytes.lean verification (when Lean is available).
#   - Darwin arm64 (Apple Silicon): MAP_JIT seed-c assembly reproduction,
#     4 examples + fixed vectors + N (default 10,000) differential cases
#     against sporos_fast (and Lean SporosCheckArm64Diff on examples/vectors),
#     plus caught negative control on modified binary.
#   - Linux x86_64 (Xeon): seed-a assembly reproduction from spr1_check_arm64.hex
#     plus C++23 g++ build and 2-run determinism.
#   - Linux aarch64 (native ARM64, e.g. NVIDIA DGX Spark): native seed-c
#     assembly reproduction, 4 examples + fixed vectors + N (default 10,000)
#     native differential cases (0 disagreements), unmodified 4-channel ptrace
#     perturbation sweeps over [tripled, core_end) and [vote, core_end) (0 false
#     accepts), and caught negative controls (both-guards modified; single-guard
#     modified + ptrace perturbation; single-bit status + GPR perturbation).
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"

CASES=10000
SEED="${FUZZ_SEED:-1}"
SKIP_LEAN="${SPOROS_SKIP_LEAN:-0}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cases)
      shift
      [ "$#" -gt 0 ] || { echo "SPR1_ARM64_GATE_FAIL: --cases requires N" >&2; exit 2; }
      CASES="$1"
      ;;
    --seed)
      shift
      [ "$#" -gt 0 ] || { echo "SPR1_ARM64_GATE_FAIL: --seed requires S" >&2; exit 2; }
      SEED="$1"
      ;;
    --skip-lean)
      SKIP_LEAN=1
      ;;
    -h|--help)
      echo "usage: $0 [--cases N] [--seed S] [--skip-lean]"
      exit 0
      ;;
    *)
      echo "SPR1_ARM64_GATE_FAIL: unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/spr1_arm64_gate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# Resolve C++23 compiler (POSIX sh; no bash arrays).
CXX_BIN=""
CXX_EXTRA=""
if [ -n "${CXX:-}" ]; then
  CXX_BIN="$CXX"
elif [ "$HOST_OS" = "Darwin" ]; then
  if /usr/bin/clang++ --version >/dev/null 2>&1; then
    CXX_BIN="/usr/bin/clang++"
  elif [ -x "/Library/Developer/CommandLineTools/usr/bin/clang++" ]; then
    CXX_BIN="/Library/Developer/CommandLineTools/usr/bin/clang++"
    if [ -d "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" ]; then
      CXX_EXTRA="-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
    fi
  else
    CXX_BIN="clang++"
  fi
else
  if command -v g++ >/dev/null 2>&1; then
    CXX_BIN="g++"
  elif command -v c++ >/dev/null 2>&1; then
    CXX_BIN="c++"
  else
    CXX_BIN="clang++"
  fi
fi

STD_FLAG="-std=c++23"
# shellcheck disable=SC2086
if ! printf 'int main(){return 0;}\n' | "$CXX_BIN" $CXX_EXTRA -x c++ -std=c++23 -fsyntax-only - >/dev/null 2>&1; then
  STD_FLAG="-std=c++2b"
fi

compile_cxx() {
  out_bin="$1"
  src_file="$2"
  # shellcheck disable=SC2086
  "$CXX_BIN" $CXX_EXTRA "$STD_FLAG" -O2 -Wall -Wextra -Werror -o "$out_bin" "$src_file"
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | LC_ALL=C awk '{print $1}'
  else
    shasum -a 256 "$1" | LC_ALL=C awk '{print $1}'
  fi
}

echo "SPR1_ARM64_GATE_START host=${HOST_OS}/${HOST_ARCH} cxx=${CXX_BIN} std=${STD_FLAG} cases=${CASES} seed=${SEED}"

# ---------------------------------------------------------------------------
# Phase 1: C++23 generator build + 2-run determinism + byte-exact match
# ---------------------------------------------------------------------------
BIN_REF="$ROOT/bootstrap/model/spr1_check_arm64"
FMT_C="$ROOT/bootstrap/model/spr1_check_arm64.bin"
FMT_A="$ROOT/bootstrap/model/spr1_check_arm64.hex"
GEN_SRC="$ROOT/bootstrap/model/spr1_check_arm64_gen.cpp"
LEAN_DATA="$ROOT/formal/Bootstrap/SporosCheckArm64BinData.lean"

compile_cxx "$WORK/gen" "$GEN_SRC"
mkdir -p "$WORK/run1" "$WORK/run2"
"$WORK/gen" "$WORK/run1" "$WORK/run1" >/dev/null
"$WORK/gen" "$WORK/run2" "$WORK/run2" >/dev/null

for f in spr1_check_arm64 spr1_check_arm64.bin spr1_check_arm64.hex SporosCheckArm64BinData.lean; do
  cmp -s "$WORK/run1/$f" "$WORK/run2/$f" || {
    echo "SPR1_ARM64_GATE_FAIL: generator non-deterministic on $f" >&2
    exit 1
  }
done

cmp -s "$WORK/run1/spr1_check_arm64" "$BIN_REF" || {
  echo "SPR1_ARM64_GATE_FAIL: generated spr1_check_arm64 differs from committed binary" >&2
  exit 1
}
cmp -s "$WORK/run1/spr1_check_arm64.bin" "$FMT_C" || {
  echo "SPR1_ARM64_GATE_FAIL: generated spr1_check_arm64.bin differs from committed listing" >&2
  exit 1
}
cmp -s "$WORK/run1/spr1_check_arm64.hex" "$FMT_A" || {
  echo "SPR1_ARM64_GATE_FAIL: generated spr1_check_arm64.hex differs from committed listing" >&2
  exit 1
}
if [ -f "$LEAN_DATA" ]; then
  cmp -s "$WORK/run1/SporosCheckArm64BinData.lean" "$LEAN_DATA" || {
    echo "SPR1_ARM64_GATE_FAIL: generated SporosCheckArm64BinData.lean differs from committed file" >&2
    exit 1
  }
fi
echo "GEN_DETERMINISM_OK sha256=$(sha256_of "$BIN_REF") bytes=$(wc -c < "$BIN_REF" | LC_ALL=C awk '{print $1}')"

# Optional Lean CheckBytes check when formal/ and lake are present
LAKE_BIN=""
if [ "$SKIP_LEAN" -eq 0 ] && [ -f "$ROOT/formal/Bootstrap/SporosCheckArm64CheckBytes.lean" ]; then
  if command -v lake >/dev/null 2>&1; then
    LAKE_BIN="$(command -v lake)"
  elif [ -x "${HOME:-}/.elan/bin/lake" ]; then
    LAKE_BIN="$HOME/.elan/bin/lake"
  fi
fi

ensure_bootstrap_olean() {
  mod="$1"
  out="$ROOT/formal/.lake/build/lib/lean/Bootstrap/${mod}.olean"
  if [ ! -f "$out" ]; then
    mkdir -p "$ROOT/formal/.lake/build/lib/lean/Bootstrap"
    (cd "$ROOT/formal" && "$LAKE_BIN" env lean -R . "Bootstrap/${mod}.lean" -o ".lake/build/lib/lean/Bootstrap/${mod}.olean")
  fi
}

if [ -n "$LAKE_BIN" ]; then
  if [ ! -f "$ROOT/formal/.lake/build/lib/lean/Bootstrap/SporosCheckArm64BinData.olean" ]; then
    (cd "$ROOT/formal" && "$LAKE_BIN" build Bootstrap.PDist)
    ensure_bootstrap_olean AArch64Env
    ensure_bootstrap_olean AArch64Phys
    ensure_bootstrap_olean SporosCheckArm64BinData
  fi
  (cd "$ROOT/formal" && "$LAKE_BIN" env lean --run Bootstrap/SporosCheckArm64CheckBytes.lean)
  echo "LEAN_CHECK_BYTES_OK"
fi

# ---------------------------------------------------------------------------
# Phase 2: Architecture-specific seed assembly reproduction
# ---------------------------------------------------------------------------
if [ "$HOST_OS" = "Linux" ] && [ "$HOST_ARCH" = "x86_64" ]; then
  if [ -f "$ROOT/bootstrap/roots/a/seed-a" ]; then
    cp "$ROOT/bootstrap/roots/a/seed-a" "$WORK/seed-a"
    chmod 755 "$WORK/seed-a"
    if [ -f "$ROOT/bootstrap/roots/a/seed-a.hex" ]; then
      "$WORK/seed-a" < "$ROOT/bootstrap/roots/a/seed-a.hex" > "$WORK/seed-a.self"
      cmp -s "$WORK/seed-a.self" "$ROOT/bootstrap/roots/a/seed-a" || {
        echo "SPR1_ARM64_GATE_FAIL: seed-a self-reproduction failed" >&2
        exit 1
      }
    fi
    "$WORK/seed-a" < "$FMT_A" > "$WORK/from_seed_a"
    cmp -s "$WORK/from_seed_a" "$BIN_REF" || {
      echo "SPR1_ARM64_GATE_FAIL: seed-a < spr1_check_arm64.hex != spr1_check_arm64" >&2
      exit 1
    }
    echo "SEED_A_ASSEMBLY_OK sha256=$(sha256_of "$WORK/from_seed_a")"
  fi
  echo "SPR1_ARM64_GATE_X86_64_OK"
  exit 0
fi

if [ "$HOST_ARCH" != "arm64" ] && [ "$HOST_ARCH" != "aarch64" ]; then
  echo "SPR1_ARM64_GATE_NON_ARM64_GEN_ONLY host=${HOST_OS}/${HOST_ARCH}"
  exit 0
fi

if [ "$HOST_OS" = "Darwin" ]; then
  compile_cxx "$WORK/jit" "$ROOT/bootstrap/model/aarch64_jit_exec.cpp"
  if [ -f "$ROOT/bootstrap/roots/c/seed-c.bin" ]; then
    "$WORK/jit" "$ROOT/bootstrap/roots/c/seed-c" < "$ROOT/bootstrap/roots/c/seed-c.bin" > "$WORK/seed-c.self"
    cmp -s "$WORK/seed-c.self" "$ROOT/bootstrap/roots/c/seed-c" || {
      echo "SPR1_ARM64_GATE_FAIL: seed-c MAP_JIT self-reproduction failed" >&2
      exit 1
    }
  fi
  "$WORK/jit" "$ROOT/bootstrap/roots/c/seed-c" < "$FMT_C" > "$WORK/from_seed_c"
  cmp -s "$WORK/from_seed_c" "$BIN_REF" || {
    echo "SPR1_ARM64_GATE_FAIL: MAP_JIT seed-c < spr1_check_arm64.bin != spr1_check_arm64" >&2
    exit 1
  }
  echo "SEED_C_MAP_JIT_ASSEMBLY_OK sha256=$(sha256_of "$WORK/from_seed_c")"
elif [ "$HOST_OS" = "Linux" ]; then
  cp "$ROOT/bootstrap/roots/c/seed-c" "$WORK/seed-c"
  chmod 755 "$WORK/seed-c"
  if [ -f "$ROOT/bootstrap/roots/c/seed-c.bin" ]; then
    "$WORK/seed-c" < "$ROOT/bootstrap/roots/c/seed-c.bin" > "$WORK/seed-c.self"
    cmp -s "$WORK/seed-c.self" "$ROOT/bootstrap/roots/c/seed-c" || {
      echo "SPR1_ARM64_GATE_FAIL: native seed-c self-reproduction failed" >&2
      exit 1
    }
  fi
  "$WORK/seed-c" < "$FMT_C" > "$WORK/from_seed_c"
  cmp -s "$WORK/from_seed_c" "$BIN_REF" || {
    echo "SPR1_ARM64_GATE_FAIL: native seed-c < spr1_check_arm64.bin != spr1_check_arm64" >&2
    exit 1
  }
  echo "SEED_C_NATIVE_ASSEMBLY_OK sha256=$(sha256_of "$WORK/from_seed_c")"
else
  echo "SPR1_ARM64_GATE_FAIL: unsupported arm64 OS: $HOST_OS" >&2
  exit 1
fi

# Use the seed-c-assembled executable copy for all runtime tests.
BIN_RUN="$WORK/spr1_check_arm64"
cp "$WORK/from_seed_c" "$BIN_RUN"
chmod 755 "$BIN_RUN"

# ---------------------------------------------------------------------------
# Phase 3: Differential corpus (examples + fixed vectors + N generated cases)
# ---------------------------------------------------------------------------
compile_cxx "$WORK/fuzz" "$ROOT/bootstrap/model/spr1_check_arm64_fuzz.cpp"

# Build batch check-only oracle from sporos_fast.cpp using separate-file sed
# (extracts decode + checkProg verbatim without modifying sporos_fast.cpp).
sed '/^\/\/ ---* semantics ---/,$d' "$ROOT/bootstrap/model/sporos_fast.cpp" > "$WORK/oracle_head.cpp"
cat "$WORK/oracle_head.cpp" > "$WORK/oracle.cpp"
cat << 'EOF' >> "$WORK/oracle.cpp"
}  // namespace

#include <fstream>
#include <iostream>

static std::vector<std::uint8_t> readBinFile(const std::string& p) {
  std::FILE* fp = std::fopen(p.c_str(), "rb");
  if (!fp) { std::fprintf(stderr, "oracle: cannot open %s\n", p.c_str()); std::exit(2); }
  std::vector<std::uint8_t> v;
  std::uint8_t buf[1 << 16];
  std::size_t k;
  while ((k = std::fread(buf, 1, sizeof buf, fp)) > 0) v.insert(v.end(), buf, buf + k);
  std::fclose(fp);
  return v;
}

int main(int argc, char** argv) {
  if (argc != 3 || std::string(argv[1]) != "--list") {
    std::fputs("usage: oracle --list LISTFILE\n", stderr);
    return 2;
  }
  std::ifstream in(argv[2]);
  if (!in) { std::fprintf(stderr, "oracle: cannot open list %s\n", argv[2]); return 2; }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto bs = readBinFile(line);
    auto pr = decode(bs);
    int rc = (pr && checkProg(*pr)) ? 0 : 5;
    std::printf("%d %s\n", rc, line.c_str());
  }
  std::fflush(stdout);
  return 0;
}
EOF
compile_cxx "$WORK/oracle" "$WORK/oracle.cpp"

mkdir -p "$WORK/ex" "$WORK/vec" "$WORK/cases"
"$WORK/fuzz" examples "$WORK/ex"
"$WORK/fuzz" vectors "$WORK/vec"
"$WORK/fuzz" gen "$CASES" "$SEED" "$WORK/cases"

# Assemble deterministic list of examples + fixed vectors
: > "$WORK/fixed.list"
for f in "$WORK"/ex/*.spb "$WORK"/vec/*.spb; do
  [ -f "$f" ] && printf '%s\n' "$f" >> "$WORK/fixed.list"
done
for d in "$ROOT/bootstrap/roots/sporos/vectors" "$ROOT/examples/sporos" "$ROOT/tests/sporos/vectors"; do
  if [ -d "$d" ]; then
    for f in "$d"/*.spb; do
      [ -f "$f" ] && printf '%s\n' "$f" >> "$WORK/fixed.list"
    done
  fi
done

# Assemble full differential list (fixed + generated fuzz cases)
cat "$WORK/fixed.list" > "$WORK/all.list"
for f in "$WORK"/cases/case_*.spb; do
  [ -f "$f" ] && printf '%s\n' "$f" >> "$WORK/all.list"
done

"$WORK/fuzz" run "$BIN_RUN" "$WORK/all.list" > "$WORK/bin.out"
"$WORK/oracle" --list "$WORK/all.list" > "$WORK/oracle.out"

LC_ALL=C awk '
NR == FNR {
  n1++
  b_rc[n1] = $1
  b_path[n1] = $2
  next
}
{
  n2++
  o_rc = $1
  o_path = $2
  if (b_path[n2] != o_path) {
    printf("DIFF_PATH_MISMATCH line=%d bin=%s oracle=%s\n", n2, b_path[n2], o_path) > "/dev/stderr"
    bad++
  } else if (b_rc[n2] != o_rc) {
    printf("DIFF_MISMATCH path=%s bin=%s oracle=%s\n", o_path, b_rc[n2], o_rc) > "/dev/stderr"
    bad++
  } else {
    if (o_rc == "0") acc++
    else if (o_rc == "5") rej++
    else {
      printf("DIFF_UNEXPECTED_RC path=%s rc=%s\n", o_path, o_rc) > "/dev/stderr"
      bad++
    }
  }
}
END {
  if (n1 != n2 || n1 == 0) {
    printf("DIFF_COUNT_MISMATCH bin=%d oracle=%d\n", n1 + 0, n2 + 0) > "/dev/stderr"
    exit 1
  }
  printf("DIFF_OK total=%d accept=%d reject=%d disagreements=%d\n", n1, acc + 0, rej + 0, bad + 0)
  if (bad + 0 != 0) exit 1
}
' "$WORK/bin.out" "$WORK/oracle.out"

# Verify 4 canonical examples explicitly exit 0 and fixed accept_/reject_ vectors match
for ex_name in echo hello upper fib; do
  rc=0
  "$WORK/fuzz" exec1 "$BIN_RUN" "$WORK/ex/${ex_name}.spb" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "SPR1_ARM64_GATE_FAIL: example ${ex_name}.spb exited $rc (expected 0)" >&2
    exit 1
  fi
done
echo "EXAMPLES_OK count=4 (echo,hello,upper,fib)"

# Cross-check fixed list against Lean SporosCheckArm64Diff when available
if [ -n "$LAKE_BIN" ] && [ -f "$ROOT/formal/Bootstrap/SporosCheckArm64Diff.lean" ]; then
  if [ ! -f "$ROOT/formal/.lake/build/lib/lean/Bootstrap/SporosCheckArm64Diff.olean" ]; then
    (cd "$ROOT/formal" && "$LAKE_BIN" build Bootstrap.PDist Bootstrap.TMR Bootstrap.TMRExposure Sporos.Bytecode Sporos.Check Sporos.SafeStep Sporos.Run Sporos.Examples)
    ensure_bootstrap_olean AArch64Env
    ensure_bootstrap_olean AArch64Phys
    ensure_bootstrap_olean TMRVoterArm64
    ensure_bootstrap_olean SporosCheckArm64Flat
    ensure_bootstrap_olean SporosCheckArm64Diff
  fi
  "$WORK/fuzz" run "$BIN_RUN" "$WORK/fixed.list" > "$WORK/fixed.bin.out"
  printf 'import Bootstrap.SporosCheckArm64Diff\n#eval (do let rc ← SporosCheckArm64Diff.main ["--list", "%s"]; if rc != 0 then throw (IO.userError s!"rc={rc}") : IO Unit)\n' "$WORK/fixed.list" \
    | (cd "$ROOT/formal" && "$LAKE_BIN" env lean -R . --stdin) > "$WORK/fixed.lean.out"
  LC_ALL=C awk '
  NR == FNR {
    n1++
    b_rc[n1] = $1
    b_path[n1] = $2
    next
  }
  {
    n2++
    want = ($1 == "accept") ? "0" : "5"
    if (b_path[n2] != $2 || b_rc[n2] != want) {
      printf("LEAN_DIFF_MISMATCH path=%s bin=%s lean=%s\n", $2, b_rc[n2], $1) > "/dev/stderr"
      bad++
    }
  }
  END {
    if (n1 != n2 || n1 == 0 || bad + 0 != 0) exit 1
    printf("LEAN_DIFF_FIXED_OK total=%d disagreements=0\n", n1)
  }
  ' "$WORK/fixed.bin.out" "$WORK/fixed.lean.out"
fi

# ---------------------------------------------------------------------------
# Phase 4: Symbol audit + caught negative controls
# ---------------------------------------------------------------------------
"$WORK/fuzz" syms "$FMT_C" "$BIN_RUN" > "$WORK/syms.txt"
TRIPLED_ADDR="$(LC_ALL=C awk '$2 == "tripled" || $2 == "p1" { print $1; exit }' "$WORK/syms.txt")"
VOTE_ADDR="$(LC_ALL=C awk '$2 == "vote" { print $1; exit }' "$WORK/syms.txt")"
GUARD_ADDR="$(LC_ALL=C awk '$2 == "guard" { print $1; exit }' "$WORK/syms.txt")"
# BSS descriptor addresses (fixed in spr1_check_arm64_gen.cpp: kBss + {0, 0x1000, 0x2000})
DESC1_ADDR="0x600000"
DESC2_ADDR="0x601000"
DESC3_ADDR="0x602000"
[ -n "$TRIPLED_ADDR" ] && [ -n "$VOTE_ADDR" ] && [ -n "$GUARD_ADDR" ] || {
  echo "SPR1_ARM64_GATE_FAIL: missing tripled/vote/guard symbols" >&2
  exit 1
}

# Offsets relative to guard (0x4076f4):
#   +0x24 (0x407718): guard 1 b.ne not_acc  (81 01 00 54)
#   +0x40 (0x407734): guard 2 b.ne not_acc  (a1 00 00 54)
#   +0x48 (0x40773c): sub x0, x20, #150     (80 5a 02 d1)
#   +0x80 (0x407774): movz x0, #5           (a0 00 80 d2)
#   +0x84 (0x407778): movz x2, #5           (a2 00 80 d2)
G1_ADDR="$(printf '0x%x' $((GUARD_ADDR + 36)))"
G2_ADDR="$(printf '0x%x' $((GUARD_ADDR + 64)))"
SUB_ADDR="$(printf '0x%x' $((GUARD_ADDR + 72)))"
REJ0_ADDR="$(printf '0x%x' $((GUARD_ADDR + 128)))"
REJ2_ADDR="$(printf '0x%x' $((GUARD_ADDR + 132)))"

# Negative control 1 (both guards + status derivation disabled -> reject vector exits 0, caught!)
"$WORK/fuzz" patch "$BIN_RUN" "$WORK/neg_both_guards" \
  "$G1_ADDR" "81010054>1f2003d5" \
  "$G2_ADDR" "a1000054>1f2003d5" \
  "$SUB_ADDR" "805a02d1>000080d2"

neg_rc=0
"$WORK/fuzz" exec1 "$WORK/neg_both_guards" "$WORK/vec/reject_empty.spb" || neg_rc=$?
if [ "$neg_rc" -ne 0 ]; then
  echo "SPR1_ARM64_GATE_FAIL: neg_both_guards did not produce expected false-accept (got rc=$neg_rc)" >&2
  exit 1
fi
echo "NEG_CTRL_BOTH_GUARDS_CAUGHT_OK modified_rc=${neg_rc} expected_oracle_rc=5"

# Native Linux aarch64 perturbation sweeps + single-guard negative controls
if [ "$HOST_OS" = "Linux" ] && [ "$HOST_ARCH" = "aarch64" ]; then
  # 4a-1. Unmodified binary: full 4-channel single-perturbation sweep from
  # L_tripled covering [L_tripled, L_core_end) (P1 + P2 + P3 + voter/guards),
  # including all 3 BSS descriptors and mapped input buffers on the mem channel.
  "$WORK/fuzz" inject "$BIN_RUN" "$WORK/vec/reject_empty.spb" "$TRIPLED_ADDR" \
    --expect reject --classes gpr,flags,fetch,mem --timeout-ms 500 --buf-extra 8 \
    --desc desc1 "$DESC1_ADDR" 16 0 8 \
    --desc desc2 "$DESC2_ADDR" 16 0 8 \
    --desc desc3 "$DESC3_ADDR" 16 0 8 > "$WORK/inj_tripled.txt"
  cat "$WORK/inj_tripled.txt"
  grep -q '^SDC_accept_invalid=0$' "$WORK/inj_tripled.txt" || {
    echo "SPR1_ARM64_GATE_FAIL: unmodified tripled inject reported non-zero false accepts" >&2
    exit 1
  }
  if grep -q '^class=mem injections=0 ' "$WORK/inj_tripled.txt"; then
    echo "SPR1_ARM64_GATE_FAIL: unmodified tripled inject ran 0 mem injections" >&2
    exit 1
  fi
  echo "NATIVE_AARCH64_TRIPLED_PERTURB_OK SDC_accept_invalid=0"

  # 4a-2. Unmodified binary: full 4-channel single-perturbation sweep from vote
  # over reject_empty.spb must produce 0 false accepts (exit 0 from inject).
  "$WORK/fuzz" inject "$BIN_RUN" "$WORK/vec/reject_empty.spb" "$VOTE_ADDR" \
    --expect reject --classes gpr,flags,fetch,mem --timeout-ms 500 > "$WORK/inj_clean.txt"
  cat "$WORK/inj_clean.txt"
  grep -q '^SDC_accept_invalid=0$' "$WORK/inj_clean.txt" || {
    echo "SPR1_ARM64_GATE_FAIL: unmodified inject reported non-zero false accepts" >&2
    exit 1
  }
  echo "NATIVE_AARCH64_CLEAN_PERTURB_OK SDC_accept_invalid=0"

  # 4b. Negative control 2a: single guard disabled (Guard 1 -> NOP, status
  # derivation -> movz x0, #0, Guard 2 intact). Fault-free golden still rejects
  # (rc=5), while single-perturbation sweep catches false accept (rc=1, SDC>=1).
  "$WORK/fuzz" patch "$BIN_RUN" "$WORK/neg_one_guard" \
    "$G1_ADDR" "81010054>1f2003d5" \
    "$SUB_ADDR" "805a02d1>000080d2"
  g_rc=0
  "$WORK/fuzz" exec1 "$WORK/neg_one_guard" "$WORK/vec/reject_empty.spb" || g_rc=$?
  if [ "$g_rc" -ne 5 ]; then
    echo "SPR1_ARM64_GATE_FAIL: neg_one_guard golden exited $g_rc (expected 5)" >&2
    exit 1
  fi
  inj_rc=0
  "$WORK/fuzz" inject "$WORK/neg_one_guard" "$WORK/vec/reject_empty.spb" "$VOTE_ADDR" \
    --expect reject --classes gpr,flags --timeout-ms 500 > "$WORK/inj_one_guard.txt" || inj_rc=$?
  if [ "$inj_rc" -ne 1 ]; then
    cat "$WORK/inj_one_guard.txt" >&2
    echo "SPR1_ARM64_GATE_FAIL: neg_one_guard perturbation was NOT caught (rc=$inj_rc)" >&2
    exit 1
  fi
  echo "NEG_CTRL_ONE_GUARD_PERTURB_CAUGHT_OK $(grep '^SDC_accept_invalid=' "$WORK/inj_one_guard.txt")"

  # 4c. Negative control 2b: pure-GPR perturbation control (weakening dual-reg
  # exit(5) status to Hamming-weight-1 exit(1) -> single GPR bit-flip on x0
  # causes false accept, caught by inject --classes gpr).
  "$WORK/fuzz" patch "$BIN_RUN" "$WORK/neg_gpr_status" \
    "$REJ0_ADDR" "a00080d2>200080d2" \
    "$REJ2_ADDR" "a20080d2>220080d2"
  inj_gpr_rc=0
  "$WORK/fuzz" inject "$WORK/neg_gpr_status" "$WORK/vec/reject_empty.spb" "$VOTE_ADDR" \
    --expect reject --classes gpr --timeout-ms 500 > "$WORK/inj_gpr.txt" || inj_gpr_rc=$?
  if [ "$inj_gpr_rc" -ne 1 ]; then
    cat "$WORK/inj_gpr.txt" >&2
    echo "SPR1_ARM64_GATE_FAIL: neg_gpr_status GPR perturbation was NOT caught (rc=$inj_gpr_rc)" >&2
    exit 1
  fi
  echo "NEG_CTRL_GPR_PERTURB_CAUGHT_OK $(grep '^SDC_accept_invalid=' "$WORK/inj_gpr.txt")"
fi

echo "SPR1_ARM64_GATE_PASS host=${HOST_OS}/${HOST_ARCH}"
