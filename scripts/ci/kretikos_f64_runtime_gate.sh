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
  python3 - "$case_dir/$profile.cubin" "$profile" "$cubin_size" "$kernel" <<'PY'
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
profile = sys.argv[2]
expected_size = int(sys.argv[3])
kernel = sys.argv[4].encode()
data = path.read_bytes()
if len(data) != expected_size:
    raise SystemExit(f"unexpected {profile} CUBIN size: {len(data)}")
if data[:4] != b"\x7fELF":
    raise SystemExit(f"{profile} CUBIN is not ELF")
machine = struct.unpack_from("<H", data, 18)[0]
if machine != 190:
    raise SystemExit(f"{profile} CUBIN e_machine mismatch: {machine}")
if kernel not in data:
    raise SystemExit(f"{profile} CUBIN kernel symbol missing")
PY

  ./bin/kretikos run-source "$source" -o "$case_dir/source" --force --validate-runtime >"$case_dir/run_source.out" 2>"$case_dir/run_source.err"
  python3 - "$case_dir/source/kretikos_bundle.v1.json" "$case_dir/source/kretikos_source_profile.v1.json" "$profile" "$kernel" "$pass_reason" <<'PY'
import json
import sys

bundle = json.load(open(sys.argv[1], encoding="utf-8"))
source = json.load(open(sys.argv[2], encoding="utf-8"))
profile = sys.argv[3]
kernel = sys.argv[4]
pass_reason = sys.argv[5]
runtime = bundle.get("runtime_validation") or {}
if source.get("profile") != profile:
    raise SystemExit("source profile mismatch")
if source.get("runtime_backed") is not True:
    raise SystemExit(f"{profile} must be runtime-backed")
if source.get("runtime_rung") != profile:
    raise SystemExit("runtime rung mismatch")
if source.get("kernel") != kernel:
    raise SystemExit("kernel mismatch")
if runtime.get("rung") != profile:
    raise SystemExit("bundle runtime rung mismatch")
if runtime.get("kernel") != kernel:
    raise SystemExit("bundle runtime kernel mismatch")
status = runtime.get("status")
reason = runtime.get("reason")
if status not in {"pass", "not_run"}:
    raise SystemExit(f"unexpected runtime status: {status}/{reason}")
if status == "pass" and reason != pass_reason:
    raise SystemExit(f"unexpected runtime pass reason: {reason}")
PY
done

echo "kretikos_f64_runtime_gate: PASS out=$OUT_DIR cases=${#CASES[@]}"
