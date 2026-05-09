#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_F64_RUNTIME_DIR:-$(mktemp -d /tmp/kretikos-f64-runtime.XXXXXX)}"
mkdir -p "$OUT_DIR"

CASES=(
  "vec_add_f64|examples/kretikos/real_vec_add_f64.sio|add.f64|3112|sounio_bare_vec_add_f64_sm80|runtime_vec_add_f64_pass"
  "vec_sub_f64|examples/kretikos/real_vec_sub_f64.sio|sub.f64|3112|sounio_bare_vec_sub_f64_sm80|runtime_vec_sub_f64_pass"
  "vec_mul_f64|examples/kretikos/real_vec_mul_f64.sio|mul.rn.f64|3112|sounio_bare_vec_mul_f64_sm80|runtime_vec_mul_f64_pass"
  "vec_div_f64|examples/kretikos/real_vec_div_f64.sio|div.rn.f64|5160|sounio_bare_vec_div_f64_sm80|runtime_vec_div_f64_pass"
  "fma_f64|examples/kretikos/real_fma_f64.sio|fma.rn.f64|3112|sounio_bare_vec_fma_f64_sm80|runtime_fma_f64_pass"
)

for row in "${CASES[@]}"; do
  IFS='|' read -r profile source opcode cubin_size kernel pass_reason <<<"$row"
  case_dir="$OUT_DIR/$profile"
  mkdir -p "$case_dir/source"

  ./bin/souc check "$source" >"$case_dir/souc_check.out" 2>"$case_dir/souc_check.err"

  ./bin/kretikos profile-source "$source" >"$case_dir/profile_source.out"
  grep -Fq "profile=$profile" "$case_dir/profile_source.out"
  grep -Fq "cubin_kind=$profile" "$case_dir/profile_source.out"
  grep -Fq "runtime_rung=$profile" "$case_dir/profile_source.out"
  grep -Fq "kernel=$kernel" "$case_dir/profile_source.out"
  grep -Fq "runtime_backed=1" "$case_dir/profile_source.out"

  ./bin/kretikos emit-ptx "$profile" -o "$case_dir/$profile.ptx" >"$case_dir/emit_ptx.out" 2>"$case_dir/emit_ptx.err"
  grep -Fq "ld.global.f64" "$case_dir/$profile.ptx"
  grep -Fq "$opcode" "$case_dir/$profile.ptx"
  grep -Fq "st.global.f64" "$case_dir/$profile.ptx"

  ./bin/kretikos emit-cubin "$profile" -o "$case_dir/$profile.cubin" >"$case_dir/emit_cubin.out" 2>"$case_dir/emit_cubin.err"
  # CUBIN structural checks (replaces python struct.unpack heredoc):
  #   - kretikos cubin-validate: ELF64 magic + e_machine == 190
  #   - file size matches expected
  #   - kernel symbol byte sequence present in cubin
  ./bin/kretikos cubin-validate "$case_dir/$profile.cubin" >/dev/null
  __actual_size=$(stat -c %s "$case_dir/$profile.cubin")
  [ "$__actual_size" = "$cubin_size" ] || {
    echo "unexpected $profile CUBIN size: $__actual_size (want $cubin_size)" >&2
    exit 1
  }
  grep -aFq "$kernel" "$case_dir/$profile.cubin" || {
    echo "$profile CUBIN kernel symbol missing" >&2
    exit 1
  }

  ./bin/kretikos run-source "$source" -o "$case_dir/source" --force --validate-runtime >"$case_dir/run_source.out" 2>"$case_dir/run_source.err"
  # Bundle + source-profile validation (replaces python json.load heredoc).
  ./bin/kretikos kaxi-validate-evidence "$case_dir/source/kretikos_source_profile.v1.json" \
    --expect "profile=$profile" \
    --expect "runtime_backed=true" \
    --expect "runtime_rung=$profile" \
    --expect "kernel=$kernel" >/dev/null
  ./bin/kretikos kaxi-validate-evidence "$case_dir/source/kretikos_bundle.v1.json" \
    --expect "runtime_validation.rung=$profile" \
    --expect "runtime_validation.kernel=$kernel" \
    --expect "runtime_validation.status=any:pass|not_run" >/dev/null
  __runtime_status=$(./bin/kretikos kaxi-validate-evidence "$case_dir/source/kretikos_bundle.v1.json" --print-or-empty runtime_validation.status)
  if [ "$__runtime_status" = "pass" ]; then
    ./bin/kretikos kaxi-validate-evidence "$case_dir/source/kretikos_bundle.v1.json" \
      --expect "runtime_validation.reason=$pass_reason" >/dev/null
  fi
done

echo "kretikos_f64_runtime_gate: PASS out=$OUT_DIR cases=${#CASES[@]}"
