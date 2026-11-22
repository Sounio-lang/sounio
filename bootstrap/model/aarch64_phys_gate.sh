#!/usr/bin/env bash
# Tri-Silicon verification gate for Pillar P4-4 (AArch64Env, AArch64Phys,
# TMRVoterArm64, AArch64PhysCheck).
#
# Usage:
#   bootstrap/model/aarch64_phys_gate.sh [diff_cases] [mc_trials] [seed] [--neg-diff|--neg-voter|--selftest]
#   bootstrap/model/aarch64_phys_gate.sh --emit-only <out_dir> [diff_cases] [mc_trials] [seed]
#   bootstrap/model/aarch64_phys_gate.sh --verify-dir <in_dir> [diff_cases]
#
# Behavior by host architecture:
#   - Darwin arm64 (Apple Silicon): MAP_JIT native differential (>= 100k cases,
#     53 forms) + Hamming-1 fetch boundary guard-page sweep + 44B voter single-
#     fault sweep + Mode A/B 4-channel Monte Carlo + Lean selftest.
#   - Linux aarch64 (e.g. NVIDIA DGX Spark ARMv9): RWX native differential +
#     raw unmodified fork() svc #0 exit(3) + ptrace(PTRACE_SINGLESTEP) voter
#     check + Hamming-1 fetch boundary sweep + single-fault sweep + Mode A/B
#     Monte Carlo + Lean selftest (or --emit-only if Lean is on a companion host).
#   - Non-ARM64 hosts (e.g. Linux x86_64 CI): runs Lean kernel/model selftest
#     cleanly and exits 0 without falsely failing non-ARM64 CI.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"

diff_cases=100000
mc_trials=20000
seed=1
mode="full"
io_dir=""
pos=0

while [[ $# -gt 0 ]]; do
  arg="$1"
  shift
  case "$arg" in
    --selftest)  mode="selftest" ;;
    --neg-diff)  mode="neg-diff" ;;
    --neg-voter) mode="neg-voter" ;;
    --emit-only)
      mode="emit-only"
      io_dir="${1:?--emit-only requires <out_dir>}"
      shift
      ;;
    --verify-dir)
      mode="verify-dir"
      io_dir="${1:?--verify-dir requires <in_dir>}"
      shift
      ;;
    --help|-h)
      echo "usage: $0 [diff_cases] [mc_trials] [seed] [--neg-diff|--neg-voter|--selftest|--emit-only <dir>|--verify-dir <dir>]"
      exit 0
      ;;
    *)
      if [[ "$arg" =~ ^[0-9]+$ ]]; then
        case "$pos" in
          0) diff_cases="$arg" ;;
          1) mc_trials="$arg" ;;
          2) seed="$arg" ;;
          *) echo "AARCH64_PHYS_GATE_FAIL: unexpected extra argument: $arg" >&2; exit 2 ;;
        esac
        pos=$((pos + 1))
      else
        echo "AARCH64_PHYS_GATE_FAIL: unknown argument: $arg" >&2
        exit 2
      fi
      ;;
  esac
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

build_fuzz() {
  local host_os="$1"
  local out_bin="$2"
  local cxx_cmd=()
  if [[ -n "${CXX:-}" ]]; then
    cxx_cmd=("$CXX")
  elif [[ "$host_os" == "Darwin" ]]; then
    if /usr/bin/clang++ --version >/dev/null 2>&1; then
      cxx_cmd=("/usr/bin/clang++")
    elif [[ -x "/Library/Developer/CommandLineTools/usr/bin/clang++" ]]; then
      cxx_cmd=("/Library/Developer/CommandLineTools/usr/bin/clang++")
      if [[ -d "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" ]]; then
        cxx_cmd+=("-isysroot" "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
      fi
    else
      cxx_cmd=("clang++")
    fi
  else
    if command -v c++ >/dev/null 2>&1; then
      cxx_cmd=("c++")
    elif command -v g++ >/dev/null 2>&1; then
      cxx_cmd=("g++")
    else
      cxx_cmd=("clang++")
    fi
  fi

  local std_flag="-std=c++23"
  if ! printf 'int main(){return 0;}\n' | "${cxx_cmd[@]}" -x c++ -std=c++23 -fsyntax-only - >/dev/null 2>&1; then
    std_flag="-std=c++2b"
  fi

  "${cxx_cmd[@]}" "$std_flag" -O2 -Wall -Wextra -Werror \
    -o "$out_bin" "$root/bootstrap/model/aarch64_phys_fuzz.cpp"
}

host_os="$(uname -s)"
host_arch="$(uname -m)"

# Remote silicon trace emission mode (does not require Lean on the target board).
if [[ "$mode" == "emit-only" ]]; then
  if [[ "$host_arch" != "arm64" && "$host_arch" != "aarch64" ]]; then
    echo "AARCH64_PHYS_GATE_FAIL: --emit-only requires arm64/aarch64 host, got ${host_os}/${host_arch}" >&2
    exit 2
  fi
  mkdir -p "$io_dir"
  fuzz="$work/aarch64_phys_fuzz"
  build_fuzz "$host_os" "$fuzz"
  "$fuzz" --diff "$diff_cases" "$seed" > "$io_dir/diff.txt"
  "$fuzz" --phys "$mc_trials" "$seed" > "$io_dir/phys.txt"
  "$fuzz" --diff 530 "$seed" --neg-diff > "$io_dir/neg_diff.txt"
  "$fuzz" --phys 1000 "$seed" --neg-voter > "$io_dir/neg_voter.txt"
  grep -v '^FETCH_ALT ' "$io_dir/phys.txt"
  echo "AARCH64_PHYS_EMIT_OK host=${host_os}/${host_arch} diff_cases=${diff_cases} mc_trials=${mc_trials} seed=${seed}"
  exit 0
fi

# Resolve lean / lake
if [[ -n "${LEAN_BIN:-}" ]]; then
  lean="$LEAN_BIN/lean"
  lake="$LEAN_BIN/lake"
elif command -v lean >/dev/null 2>&1; then
  lean="$(command -v lean)"
  lake="$(command -v lake || true)"
elif [[ -x "$HOME/.elan/bin/lean" ]]; then
  lean="$HOME/.elan/bin/lean"
  lake="$HOME/.elan/bin/lake"
else
  echo "AARCH64_PHYS_GATE_FAIL: lean not found" >&2
  exit 2
fi

# The Lean side is the `aarch64-phys-check` executable (`Bootstrap/AArch64PhysCheck.lean`);
# lake rebuilds it when the model changed. The build log is shown only on failure.
if [[ -z "${lake:-}" || ! -x "$lake" ]]; then
  echo "AARCH64_PHYS_GATE_FAIL: lake not found next to $lean" >&2
  exit 2
fi
if ! (cd "$root/formal" && "$lake" build aarch64-phys-check > "$work/lake_build.log" 2>&1); then
  cat "$work/lake_build.log" >&2
  exit 1
fi
checker="$root/formal/.lake/build/bin/aarch64-phys-check"

run_checker() {
  "$checker" "$@"
}

# Always run the pure-Lean model selftest first (all architectures).
run_checker selftest

if [[ "$mode" == "verify-dir" ]]; then
  run_checker diff "$diff_cases" "$io_dir/diff.txt"
  run_checker verify-phys "$io_dir/phys.txt"
  if [[ -f "$io_dir/neg_diff.txt" ]]; then
    if run_checker diff 530 "$io_dir/neg_diff.txt" >/dev/null 2>&1; then
      echo "AARCH64_PHYS_GATE_FAIL: neg_diff.txt was NOT caught" >&2
      exit 1
    fi
    echo "AARCH64_PHYS_NEG_DIFF_CAUGHT_OK"
  fi
  if [[ -f "$io_dir/neg_voter.txt" ]]; then
    if run_checker verify-phys "$io_dir/neg_voter.txt" >/dev/null 2>&1; then
      echo "AARCH64_PHYS_GATE_FAIL: neg_voter.txt was NOT caught" >&2
      exit 1
    fi
    echo "AARCH64_PHYS_NEG_VOTER_CAUGHT_OK"
  fi
  exit 0
fi

if [[ "$mode" == "selftest" ]]; then
  exit 0
fi

if [[ "$host_arch" != "arm64" && "$host_arch" != "aarch64" ]]; then
  echo "AARCH64_PHYS_GATE_NON_ARM64_SELFTEST_ONLY host=${host_os}/${host_arch}"
  exit 0
fi

if [[ "$host_os" != "Darwin" && "$host_os" != "Linux" ]]; then
  echo "AARCH64_PHYS_GATE_NON_ARM64_SELFTEST_ONLY host=${host_os}/${host_arch}"
  exit 0
fi

fuzz="$work/aarch64_phys_fuzz"
build_fuzz "$host_os" "$fuzz"

if [[ "$mode" == "neg-diff" ]]; then
  "$fuzz" --diff "$diff_cases" "$seed" --neg-diff > "$work/diff.txt"
  run_checker diff "$diff_cases" "$work/diff.txt"
  exit $?
fi

if [[ "$mode" == "neg-voter" ]]; then
  "$fuzz" --phys "$mc_trials" "$seed" --neg-voter > "$work/phys.txt"
  run_checker verify-phys "$work/phys.txt"
  exit $?
fi

# Full hardware verification: differential + 4-channel physical perturbation + TMR voter.
"$fuzz" --diff "$diff_cases" "$seed" > "$work/diff.txt"
run_checker diff "$diff_cases" "$work/diff.txt"

"$fuzz" --phys "$mc_trials" "$seed" > "$work/phys.txt"
grep -v '^FETCH_ALT ' "$work/phys.txt"
run_checker verify-phys "$work/phys.txt"
