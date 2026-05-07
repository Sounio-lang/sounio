#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_F64_RUNTIME_DIR:-$(mktemp -d /tmp/kretikos-f64-runtime.XXXXXX)}"
SOURCE="examples/kretikos/real_vec_add_f64.sio"
PTX_OUT="$OUT_DIR/vec_add_f64.ptx"
CUBIN_OUT="$OUT_DIR/vec_add_f64.cubin"
PROFILE_OUT="$OUT_DIR/profile_source.out"
SOURCE_DIR="$OUT_DIR/source"

mkdir -p "$OUT_DIR" "$SOURCE_DIR"

./bin/souc check "$SOURCE" >"$OUT_DIR/souc_check.out" 2>"$OUT_DIR/souc_check.err"

./bin/kretikos profile-source "$SOURCE" >"$PROFILE_OUT"
grep -Fq "profile=vec_add_f64" "$PROFILE_OUT"
grep -Fq "cubin_kind=vec_add_f64" "$PROFILE_OUT"
grep -Fq "runtime_rung=vec_add_f64" "$PROFILE_OUT"
grep -Fq "kernel=sounio_bare_vec_add_f64_sm80" "$PROFILE_OUT"
grep -Fq "runtime_backed=1" "$PROFILE_OUT"

./bin/kretikos emit-ptx vec_add_f64 -o "$PTX_OUT" >"$OUT_DIR/emit_ptx.out" 2>"$OUT_DIR/emit_ptx.err"
grep -Fq "ld.global.f64" "$PTX_OUT"
grep -Fq "add.f64" "$PTX_OUT"
grep -Fq "st.global.f64" "$PTX_OUT"

./bin/kretikos emit-cubin vec_add_f64 -o "$CUBIN_OUT" >"$OUT_DIR/emit_cubin.out" 2>"$OUT_DIR/emit_cubin.err"
python3 - "$CUBIN_OUT" <<'PY'
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes()
if len(data) != 3112:
    raise SystemExit(f"unexpected vec_add_f64 CUBIN size: {len(data)}")
if data[:4] != b"\x7fELF":
    raise SystemExit("vec_add_f64 CUBIN is not ELF")
machine = struct.unpack_from("<H", data, 18)[0]
if machine != 190:
    raise SystemExit(f"vec_add_f64 CUBIN e_machine mismatch: {machine}")
if b"sounio_bare_vec_add_f64_sm80" not in data:
    raise SystemExit("vec_add_f64 CUBIN kernel symbol missing")
PY

./bin/kretikos run-source "$SOURCE" -o "$SOURCE_DIR" --force --validate-runtime >"$OUT_DIR/run_source.out" 2>"$OUT_DIR/run_source.err"
python3 - "$SOURCE_DIR/kretikos_bundle.v1.json" "$SOURCE_DIR/kretikos_source_profile.v1.json" <<'PY'
import json
import sys

bundle = json.load(open(sys.argv[1], encoding="utf-8"))
source = json.load(open(sys.argv[2], encoding="utf-8"))
runtime = bundle.get("runtime_validation") or {}
if source.get("profile") != "vec_add_f64":
    raise SystemExit("source profile mismatch")
if source.get("runtime_backed") is not True:
    raise SystemExit("vec_add_f64 must be runtime-backed")
if source.get("runtime_rung") != "vec_add_f64":
    raise SystemExit("runtime rung mismatch")
if source.get("kernel") != "sounio_bare_vec_add_f64_sm80":
    raise SystemExit("kernel mismatch")
if runtime.get("rung") != "vec_add_f64":
    raise SystemExit("bundle runtime rung mismatch")
if runtime.get("kernel") != "sounio_bare_vec_add_f64_sm80":
    raise SystemExit("bundle runtime kernel mismatch")
status = runtime.get("status")
reason = runtime.get("reason")
if status not in {"pass", "not_run"}:
    raise SystemExit(f"unexpected runtime status: {status}/{reason}")
if status == "pass" and reason != "runtime_vec_add_f64_pass":
    raise SystemExit(f"unexpected runtime pass reason: {reason}")
PY

echo "kretikos_f64_runtime_gate: PASS out=$OUT_DIR"
