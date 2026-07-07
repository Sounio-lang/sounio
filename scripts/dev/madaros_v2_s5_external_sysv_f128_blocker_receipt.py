#!/usr/bin/env python3
"""Emit the S5 external SysV f128 boundary receipt.

This receipt makes the f128 extern boundary explicit and promotes the narrow
scalar binary128 relocatable-object oracle.
It proves that the parser/checker/lowerer front half accepts an extern C
binary128 signature, that native-v2 executable emission fails closed when an
external relocation would be required, and that native-v2 ET_REL output can be
linked with a C _Float128 passthrough helper and executed successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s5.external_sysv_f128_blocker_receipt/0.4"
STAGE = "S5_30_EXTERNAL_SYSV_F128_AGGREGATE_SRET_BLOCKER_CLASSIFIED"
CASE_ID = "external_sysv_f128_abi_blocked_front_half_received"

PASSTHRU_SOURCE = """extern "C" {
  fn passthru_f128(x: f128) -> f128;
}
fn main() -> i64 { 0 }
"""

PASSTHRU_CALL_SOURCE = """extern "C" {
  fn passthru_f128(x: f128) -> f128;
}
fn main() -> i64 {
  let x: f128 = 1.25 as f128
  let y: f128 = passthru_f128(x)
  if y == x { 0 } else { 7 }
}
"""

PASSTHRU_HELPER_C = """_Float128 passthru_f128(_Float128 x) {
  return x;
}
"""

AGGREGATE_SRET_SOURCE = """struct BoxF128 { tag: i64, x: f128, tail: i64 }
extern "C" { fn make_box(x: f128, n: i64) -> BoxF128; }
fn main() -> i64 {
  let b = make_box(1.0 as f128, 40)
  b.tag + b.tail
}
"""

AGGREGATE_SRET_HELPER_C = """struct BoxF128 { long tag; _Float128 x; long tail; };
struct BoxF128 make_box(_Float128 x, long n) {
  struct BoxF128 b;
  b.tag = n;
  b.x = x;
  b.tail = n + 1;
  return b;
}
"""


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("external SysV f128 blocker receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def run_command(cmd: list[str], cwd: Path, timeout_s: int) -> tuple[int, str, str]:
    env = os.environ.copy()
    raw = cwd / "artifacts" / "self-hosted" / "madaros"
    if raw.exists():
        env["MADAROS_RAW_BIN"] = str(raw)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def run_host_command(cmd: list[str], cwd: Path, timeout_s: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def require_fragment(path: Path, fragment: str, label: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if fragment not in text:
        raise SystemExit(f"missing source fragment for {label}: {path}")
    return {
        "label": label,
        "path": str(path),
        "fragment_sha256": sha256_text(fragment),
        "file_sha256": sha256_text(text),
        "present": True,
    }


def collect_source_evidence(root: Path) -> list[dict[str, Any]]:
    return [
        require_fragment(
            root / "self-hosted" / "parser" / "ast.sio",
            "is_extern: bool",
            "fndef_has_explicit_is_extern_bit",
        ),
        require_fragment(
            root / "self-hosted" / "parser" / "items.sio",
            "is_kernel: false, is_extern: true",
            "extern_block_parser_sets_is_extern_true",
        ),
        require_fragment(
            root / "self-hosted" / "ir" / "lower.sio",
            "IR_STRATEGY_EXTERN",
            "lowerer_preseeds_extern_strategy",
        ),
        require_fragment(
            root / "self-hosted" / "ir" / "lower.sio",
            "ir_call_extern(dst, callee_name, args, argc)",
            "lowerer_emits_symbolic_extern_call",
        ),
        require_fragment(
            root / "self-hosted" / "ir" / "ir.sio",
            "IrCallExtern",
            "ir_has_symbolic_extern_call_opcode",
        ),
        require_fragment(
            root / "self-hosted" / "ir" / "ir.sio",
            "pub fn ir_call_extern(dst: i64, symbol: Name",
            "ir_call_extern_records_symbol_name",
        ),
        require_fragment(
            root / "self-hosted" / "native" / "lower_ir.sio",
            "lower_call_extern",
            "legacy_extern_lowerer_is_symbol_reloc_path",
        ),
        require_fragment(
            root / "self-hosted" / "native" / "reloc.sio",
            "struct ExternReloc",
            "legacy_extern_reloc_shape_exists",
        ),
        require_fragment(
            root / "self-hosted" / "native" / "machine_ir.sio",
            "MIR_OP_PSEUDO_CALL",
            "native_v2_internal_call_path_is_fn_id_shape",
        ),
        require_fragment(
            root / "self-hosted" / "native" / "machine_ir.sio",
            "MIR_OP_PSEUDO_CALL_EXTERN",
            "native_v2_has_explicit_external_call_pseudo_shape",
        ),
        require_fragment(
            root / "self-hosted" / "native" / "machine_ir.sio",
            "MIR_OP_CALL_EXTERN_F128",
            "native_v2_has_scalar_external_f128_call_shape",
        ),
        require_fragment(
            root / "self-hosted" / "native" / "codegen_x86_linux.sio",
            "external_sysv_requires_relocatable_link",
            "native_v2_executable_emission_fails_closed_when_external_relocs_exist",
        ),
        require_fragment(
            root / "self-hosted" / "compiler" / "main.sio",
            "--native-v2-emit-obj",
            "native_v2_exposes_relocatable_object_mode",
        ),
        require_fragment(
            root / "self-hosted" / "compiler" / "main.sio",
            "external_call_symbols",
            "machine_module_json_exports_external_call_symbols",
        ),
    ]


def emit_passthru_case(root: Path, compiler: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    source_path = out_dir / "extern_c_passthru_f128_decl_received.sio"
    log_path = out_dir / "extern_c_passthru_f128_decl_received.check.log"
    source_path.write_text(PASSTHRU_SOURCE, encoding="utf-8")
    rc, stdout, stderr = run_command([str(compiler), "check", str(source_path)], root, timeout_s)
    log = stdout + stderr
    log_path.write_text(log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"extern C passthru_f128 declaration must check, got rc={rc}\n{log[-4000:]}")
    for fragment in ["E072", "kernel function must return", "E008", "return value does not match"]:
        if fragment in log:
            raise SystemExit(f"extern C passthru_f128 declaration emitted unexpected diagnostic {fragment!r}")
    return {
        "case_id": "extern_c_passthru_f128_decl_received",
        "class": "positive_extern_declaration_check_blocker_boundary",
        "symbol": "passthru_f128",
        "signature": "(f128)->f128",
        "source_sha256": sha256_text(PASSTHRU_SOURCE),
        "check_rc": rc,
        "check_log_sha256": sha256_text(log),
        "extern_declaration_accepted": True,
        "kernel_e072_absent": True,
        "return_mismatch_e008_absent": True,
        "lowerer_strategy_expected": "IR_STRATEGY_EXTERN",
        "ir_opcode_expected_if_called": "IrCallExtern",
        "native_v2_execution_attempted": False,
        "native_v2_external_sysv_f128_promoted": False,
    }


def emit_passthru_call_boundary_case(root: Path, compiler: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    case_id = "extern_c_passthru_f128_call_reaches_machineir_boundary"
    source_path = out_dir / f"{case_id}.sio"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    log_path = out_dir / f"{case_id}.native_v2.log"
    source_path.write_text(PASSTHRU_CALL_SOURCE, encoding="utf-8")
    rc, stdout, stderr = run_command(
        [
            str(compiler),
            "--native-v2-compile",
            str(source_path),
            "-o",
            str(elf_path),
            "--machine-module-json",
            str(mm_path),
        ],
        root,
        timeout_s,
    )
    log = stdout + stderr
    log_path.write_text(log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"{case_id}: wrapper command must return rc=0 while reporting native-v2 failure; log={log_path}")
    if "native_v2_compile: FAIL to_file rc=12" not in log:
        raise SystemExit(f"{case_id}: expected native-v2 fail-closed rc=12; log={log_path}")
    if "Segmentation fault" in log or "SIGSEGV" in log or "legacy fallback" in log:
        raise SystemExit(f"{case_id}: crash or fallback detected; log={log_path}")
    if elf_path.exists() and elf_path.stat().st_size > 0:
        raise SystemExit(f"{case_id}: external SysV f128 blocker must not emit an ELF")
    if not mm_path.exists() or mm_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: expected MachineModule JSON for classified external SysV blocker")

    module = json.loads(mm_path.read_text(encoding="utf-8"))
    if module.get("schema") != "madaros.v2.s5.machine_module/0.1":
        raise SystemExit(f"{case_id}: bad MachineModule schema")
    if module.get("legacy_fallback") is not False:
        raise SystemExit(f"{case_id}: MachineModule used fallback")
    if module.get("supported") is not True:
        raise SystemExit(f"{case_id}: MachineModule should now legalize the narrow scalar external f128 call")
    if module.get("unsupported_detail") not in ("", None):
        raise SystemExit(
            f"{case_id}: expected no MachineModule unsupported_detail after narrow external f128 legalization, "
            f"got {module.get('unsupported_detail')!r}"
        )
    external_symbols: list[str] = []
    for fn in module.get("functions", []):
        symbols = fn.get("external_call_symbols", [])
        if isinstance(symbols, list):
            external_symbols.extend(str(sym) for sym in symbols)
    if external_symbols != ["passthru_f128"]:
        raise SystemExit(f"{case_id}: expected MachineModule external_call_symbols ['passthru_f128'], got {external_symbols!r}")
    return {
        "case_id": case_id,
        "class": "negative_native_v2_external_sysv_call_boundary",
        "symbol": "passthru_f128",
        "signature": "(f128)->f128",
        "source_sha256": sha256_text(PASSTHRU_CALL_SOURCE),
        "native_v2_compile_rc": rc,
        "native_v2_compile_log_sha256": sha256_text(log),
        "elf_emitted": False,
        "machine_module_json_emitted": True,
        "machine_module_json_sha256": sha256_text(stable_json(module)),
        "machine_module_supported": module.get("supported"),
        "machine_module_unsupported_detail": module.get("unsupported_detail"),
        "machine_module_external_call_symbols": external_symbols,
        "machine_module_external_call_symbol_count": len(external_symbols),
        "legacy_fallback": False,
        "segfault": False,
        "native_v2_machineir_external_call_symbol_classified": True,
        "native_v2_machine_module_external_call_symbol_exported": True,
        "native_v2_executable_external_reloc_fail_closed": True,
        "native_v2_external_sysv_f128_promoted": False,
    }


def emit_passthru_relocatable_oracle_case(root: Path, compiler: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    case_id = "extern_c_passthru_f128_native_v2_relocatable_oracle"
    source_path = out_dir / f"{case_id}.sio"
    helper_path = out_dir / f"{case_id}.c"
    obj_path = out_dir / f"{case_id}.o"
    helper_obj_path = out_dir / f"{case_id}.helper.o"
    exe_path = out_dir / f"{case_id}.linked"
    compile_log_path = out_dir / f"{case_id}.native_v2_emit_obj.log"
    helper_log_path = out_dir / f"{case_id}.helper_compile.log"
    link_log_path = out_dir / f"{case_id}.link.log"
    run_log_path = out_dir / f"{case_id}.run.log"
    readelf_reloc_path = out_dir / f"{case_id}.readelf_reloc.txt"
    readelf_sym_path = out_dir / f"{case_id}.readelf_sym.txt"
    source_path.write_text(PASSTHRU_CALL_SOURCE, encoding="utf-8")
    helper_path.write_text(PASSTHRU_HELPER_C, encoding="utf-8")

    cc = shutil.which("gcc") or shutil.which("cc")
    readelf = shutil.which("readelf")
    if cc is None:
        raise SystemExit(f"{case_id}: gcc/cc is required for the host SysV f128 oracle")
    if readelf is None:
        raise SystemExit(f"{case_id}: readelf is required for relocation assertions")

    rc, stdout, stderr = run_command(
        [str(compiler), "--native-v2-emit-obj", str(source_path), "-o", str(obj_path)],
        root,
        timeout_s,
    )
    compile_log = stdout + stderr
    compile_log_path.write_text(compile_log, encoding="utf-8")
    if rc != 0 or not obj_path.exists() or obj_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: native-v2 object emission failed rc={rc}; log={compile_log_path}")

    rc, stdout, stderr = run_host_command([readelf, "-r", str(obj_path)], root, timeout_s)
    reloc_text = stdout + stderr
    readelf_reloc_path.write_text(reloc_text, encoding="utf-8")
    if rc != 0 or "R_X86_64_PLT32" not in reloc_text or "passthru_f128" not in reloc_text:
        raise SystemExit(f"{case_id}: expected R_X86_64_PLT32 relocation to passthru_f128; see {readelf_reloc_path}")

    rc, stdout, stderr = run_host_command([readelf, "-s", str(obj_path)], root, timeout_s)
    sym_text = stdout + stderr
    readelf_sym_path.write_text(sym_text, encoding="utf-8")
    if rc != 0 or "UND" not in sym_text or "passthru_f128" not in sym_text or " main" not in sym_text:
        raise SystemExit(f"{case_id}: expected undefined passthru_f128 and exported main symbols; see {readelf_sym_path}")

    rc, stdout, stderr = run_host_command([cc, "-std=gnu11", "-c", str(helper_path), "-o", str(helper_obj_path)], root, timeout_s)
    helper_log = stdout + stderr
    helper_log_path.write_text(helper_log, encoding="utf-8")
    if rc != 0 or not helper_obj_path.exists() or helper_obj_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: helper _Float128 object compile failed rc={rc}; log={helper_log_path}")

    rc, stdout, stderr = run_host_command([cc, "-no-pie", str(obj_path), str(helper_obj_path), "-o", str(exe_path)], root, timeout_s)
    link_log = stdout + stderr
    link_log_path.write_text(link_log, encoding="utf-8")
    if rc != 0 or not exe_path.exists() or exe_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: host link failed rc={rc}; log={link_log_path}")

    rc, stdout, stderr = run_host_command([str(exe_path)], root, timeout_s)
    run_log = stdout + stderr
    run_log_path.write_text(run_log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"{case_id}: linked SysV f128 oracle returned rc={rc}; log={run_log_path}")

    return {
        "case_id": case_id,
        "class": "positive_native_v2_relocatable_sysv_f128_oracle",
        "symbol": "passthru_f128",
        "signature": "(f128)->f128",
        "source_sha256": sha256_text(PASSTHRU_CALL_SOURCE),
        "helper_c_sha256": sha256_text(PASSTHRU_HELPER_C),
        "native_v2_emit_obj_rc": 0,
        "object_emitted": True,
        "object_sha256": sha256_bytes(obj_path.read_bytes()),
        "object_size": obj_path.stat().st_size,
        "readelf_reloc_sha256": sha256_text(reloc_text),
        "readelf_sym_sha256": sha256_text(sym_text),
        "relocation_kind": "R_X86_64_PLT32",
        "undefined_external_symbol": "passthru_f128",
        "exported_entry_symbol": "main",
        "host_c_compiler": cc,
        "host_helper_compile_rc": 0,
        "host_link_rc": 0,
        "linked_executable_sha256": sha256_bytes(exe_path.read_bytes()),
        "linked_executable_exit_code": 0,
        "f128_external_sysv_scalar_passthru_oracle_promoted": True,
        "f128_external_sysv_argument_oracle_promoted": True,
        "f128_external_sysv_return_oracle_promoted": True,
        "native_v2_external_relocatable_object_promoted": True,
        "native_v2_external_relocation_promoted": True,
        "external_aggregate_sret_abi_covered": False,
    }


def emit_passthru_wrapper_link_case(root: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    case_id = "extern_c_passthru_f128_native_v2_wrapper_link_oracle"
    source_path = out_dir / f"{case_id}.sio"
    helper_path = out_dir / f"{case_id}.c"
    helper_obj_path = out_dir / f"{case_id}.helper.o"
    exe_path = out_dir / f"{case_id}.linked"
    helper_log_path = out_dir / f"{case_id}.helper_compile.log"
    link_log_path = out_dir / f"{case_id}.native_v2_link.log"
    run_log_path = out_dir / f"{case_id}.run.log"
    source_path.write_text(PASSTHRU_CALL_SOURCE, encoding="utf-8")
    helper_path.write_text(PASSTHRU_HELPER_C, encoding="utf-8")

    wrapper = root / "bin" / "madaros"
    cc = shutil.which("gcc") or shutil.which("cc")
    if not wrapper.exists():
        raise SystemExit(f"{case_id}: wrapper not found: {wrapper}")
    if cc is None:
        raise SystemExit(f"{case_id}: gcc/cc is required for the host SysV f128 wrapper-link oracle")

    rc, stdout, stderr = run_host_command([cc, "-std=gnu11", "-c", str(helper_path), "-o", str(helper_obj_path)], root, timeout_s)
    helper_log = stdout + stderr
    helper_log_path.write_text(helper_log, encoding="utf-8")
    if rc != 0 or not helper_obj_path.exists() or helper_obj_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: helper _Float128 object compile failed rc={rc}; log={helper_log_path}")

    rc, stdout, stderr = run_command(
        [
            str(wrapper),
            "native-v2-link",
            str(source_path),
            "-o",
            str(exe_path),
            "--link-object",
            str(helper_obj_path),
            "--cc",
            cc,
        ],
        root,
        timeout_s,
    )
    link_log = stdout + stderr
    link_log_path.write_text(link_log, encoding="utf-8")
    if rc != 0 or "native-v2-link: emitted path=" not in link_log or not exe_path.exists() or exe_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: madaros native-v2-link failed rc={rc}; log={link_log_path}")

    rc, stdout, stderr = run_host_command([str(exe_path)], root, timeout_s)
    run_log = stdout + stderr
    run_log_path.write_text(run_log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"{case_id}: wrapper-linked SysV f128 oracle returned rc={rc}; log={run_log_path}")

    return {
        "case_id": case_id,
        "class": "positive_native_v2_wrapper_link_sysv_f128_oracle",
        "symbol": "passthru_f128",
        "signature": "(f128)->f128",
        "source_sha256": sha256_text(PASSTHRU_CALL_SOURCE),
        "helper_c_sha256": sha256_text(PASSTHRU_HELPER_C),
        "wrapper": str(wrapper),
        "wrapper_mode": "native-v2-link",
        "host_c_compiler": cc,
        "host_helper_compile_rc": 0,
        "native_v2_link_rc": 0,
        "native_v2_link_log_sha256": sha256_text(link_log),
        "linked_executable_sha256": sha256_bytes(exe_path.read_bytes()),
        "linked_executable_exit_code": 0,
        "f128_external_sysv_scalar_passthru_wrapper_link_promoted": True,
        "f128_external_sysv_argument_wrapper_link_promoted": True,
        "f128_external_sysv_return_wrapper_link_promoted": True,
        "native_v2_external_link_launcher_promoted": True,
        "external_aggregate_sret_abi_covered": False,
    }


def emit_aggregate_sret_blocker_case(root: Path, compiler: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    case_id = "extern_c_f128_aggregate_sret_remains_fail_closed"
    source_path = out_dir / f"{case_id}.sio"
    helper_path = out_dir / f"{case_id}.c"
    helper_obj_path = out_dir / f"{case_id}.helper.o"
    elf_path = out_dir / f"{case_id}.native_v2"
    linked_path = out_dir / f"{case_id}.linked"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    check_log_path = out_dir / f"{case_id}.check.log"
    compile_log_path = out_dir / f"{case_id}.native_v2_compile.log"
    helper_log_path = out_dir / f"{case_id}.helper_compile.log"
    link_log_path = out_dir / f"{case_id}.native_v2_link.log"
    source_path.write_text(AGGREGATE_SRET_SOURCE, encoding="utf-8")
    helper_path.write_text(AGGREGATE_SRET_HELPER_C, encoding="utf-8")

    rc, stdout, stderr = run_command([str(compiler), "check", str(source_path)], root, timeout_s)
    check_log = stdout + stderr
    check_log_path.write_text(check_log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"{case_id}: aggregate extern SRET front-half must typecheck, got rc={rc}; log={check_log_path}")

    rc, stdout, stderr = run_command(
        [
            str(compiler),
            "--native-v2-compile",
            str(source_path),
            "-o",
            str(elf_path),
            "--machine-module-json",
            str(mm_path),
        ],
        root,
        timeout_s,
    )
    compile_log = stdout + stderr
    compile_log_path.write_text(compile_log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"{case_id}: wrapper command must return rc=0 while reporting fail-closed native-v2 status")
    if "native_v2_compile: FAIL to_file rc=12" not in compile_log:
        raise SystemExit(f"{case_id}: expected fail-closed native-v2 rc=12; log={compile_log_path}")
    if "Segmentation fault" in compile_log or "SIGSEGV" in compile_log or "legacy fallback" in compile_log:
        raise SystemExit(f"{case_id}: crash or fallback detected; log={compile_log_path}")
    if elf_path.exists() and elf_path.stat().st_size > 0:
        raise SystemExit(f"{case_id}: aggregate extern SRET blocker must not emit an executable")
    if not mm_path.exists() or mm_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: expected MachineModule JSON for aggregate extern SRET blocker")

    module = json.loads(mm_path.read_text(encoding="utf-8"))
    if module.get("supported") is not False or module.get("unsupported_detail") != "external_sysv_abi_pending":
        raise SystemExit(f"{case_id}: expected unsupported external_sysv_abi_pending MachineModule")
    external_symbols: list[str] = []
    for fn in module.get("functions", []):
        symbols = fn.get("external_call_symbols", [])
        if isinstance(symbols, list):
            external_symbols.extend(str(sym) for sym in symbols)
    if external_symbols != ["make_box"]:
        raise SystemExit(f"{case_id}: expected MachineModule external_call_symbols ['make_box'], got {external_symbols!r}")

    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        raise SystemExit(f"{case_id}: gcc/cc is required for aggregate SRET linker blocker assertion")
    rc, stdout, stderr = run_host_command([cc, "-std=gnu11", "-c", str(helper_path), "-o", str(helper_obj_path)], root, timeout_s)
    helper_log = stdout + stderr
    helper_log_path.write_text(helper_log, encoding="utf-8")
    if rc != 0 or not helper_obj_path.exists() or helper_obj_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: helper aggregate SRET object compile failed rc={rc}; log={helper_log_path}")
    rc, stdout, stderr = run_command(
        [
            str(root / "bin" / "madaros"),
            "native-v2-link",
            str(source_path),
            "-o",
            str(linked_path),
            "--link-object",
            str(helper_obj_path),
            "--cc",
            cc,
        ],
        root,
        timeout_s,
    )
    link_log = stdout + stderr
    link_log_path.write_text(link_log, encoding="utf-8")
    if rc == 0 or linked_path.exists():
        raise SystemExit(f"{case_id}: aggregate extern SRET wrapper-link must remain blocked until ABI is promoted")
    if "external_sysv_abi_pending" not in link_log or "compiler produced no object" not in link_log:
        raise SystemExit(f"{case_id}: wrapper-link blocker reason changed; log={link_log_path}")

    return {
        "case_id": case_id,
        "class": "negative_external_aggregate_sret_f128_blocker_boundary",
        "symbol": "make_box",
        "signature": "(f128,i64)->BoxF128",
        "source_sha256": sha256_text(AGGREGATE_SRET_SOURCE),
        "helper_c_sha256": sha256_text(AGGREGATE_SRET_HELPER_C),
        "front_half_check_rc": 0,
        "native_v2_compile_rc": 0,
        "native_v2_compile_log_sha256": sha256_text(compile_log),
        "elf_emitted": False,
        "machine_module_json_emitted": True,
        "machine_module_supported": False,
        "machine_module_unsupported_detail": "external_sysv_abi_pending",
        "machine_module_external_call_symbols": external_symbols,
        "machine_module_external_call_symbol_count": len(external_symbols),
        "native_v2_link_rc": rc,
        "native_v2_link_log_sha256": sha256_text(link_log),
        "legacy_fallback": False,
        "segfault": False,
        "external_aggregate_sret_front_half_typechecks": True,
        "external_aggregate_sret_machineir_symbol_classified": True,
        "external_aggregate_sret_wrapper_link_fail_closed": True,
        "external_aggregate_sret_abi_promoted": False,
        "f128_external_sysv_abi_promoted": False,
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "madaros_v2_s5_external_sysv_f128_blocker.receipt.json"

    source_evidence = collect_source_evidence(root)
    passthru_case = emit_passthru_case(root, compiler, out_dir, int(args.timeout))
    passthru_call_case = emit_passthru_call_boundary_case(root, compiler, out_dir, int(args.timeout))
    relocatable_oracle_case = emit_passthru_relocatable_oracle_case(root, compiler, out_dir, int(args.timeout))
    wrapper_link_case = emit_passthru_wrapper_link_case(root, out_dir, int(args.timeout))
    aggregate_sret_blocker_case = emit_aggregate_sret_blocker_case(root, compiler, out_dir, int(args.timeout))
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE,
        "target": "x86_64-linux",
        "case_id": CASE_ID,
        "case_count": 5,
        "positive_case_count": 3,
        "negative_boundary_case_count": 2,
        "cases": [passthru_case, passthru_call_case, relocatable_oracle_case, wrapper_link_case, aggregate_sret_blocker_case],
        "source_evidence": source_evidence,
        "extern_decl_f128_typecheck_promoted": True,
        "parser_has_explicit_is_extern_bit": True,
        "ir_extern_strategy_promoted": True,
        "ir_call_extern_symbol_receipt_promoted": True,
        "native_v2_machineir_external_call_symbol_classified": True,
        "native_v2_machine_module_external_call_symbol_exported": True,
        "native_v2_machineir_external_call_symbol_promoted": True,
        "native_v2_external_relocation_promoted": True,
        "native_v2_external_relocatable_object_promoted": True,
        "native_v2_external_link_launcher_promoted": True,
        "f128_external_sysv_scalar_passthru_oracle_promoted": True,
        "f128_external_sysv_scalar_passthru_wrapper_link_promoted": True,
        "f128_external_sysv_abi_promoted": False,
        "f128_external_sysv_runtime_promoted": True,
        "f128_external_sysv_argument_oracle_promoted": True,
        "f128_external_sysv_return_oracle_promoted": True,
        "f128_external_sysv_argument_wrapper_link_promoted": True,
        "f128_external_sysv_return_wrapper_link_promoted": True,
        "external_aggregate_sret_front_half_typechecks": True,
        "external_aggregate_sret_machineir_symbol_classified": True,
        "external_aggregate_sret_wrapper_link_fail_closed": True,
        "external_aggregate_sret_abi_promoted": False,
        "f128_internal_opaque_direct_call_abi_promoted_elsewhere": True,
        "f128_internal_opaque_return_abi_promoted_elsewhere": True,
        "f128_sysv_classes_recorded_as_metadata_only": True,
        "blocked": True,
        "blocked_reason": "narrow_scalar_f128_wrapper_link_oracle_promoted_but_external_aggregate_sret_f128_remains_fail_closed_at_external_sysv_abi_pending",
        "roundtrip_contract": [
            "extern_C_f128_declaration_typechecks_without_kernel_diagnostics",
            "lowerer_has_IR_STRATEGY_EXTERN_and_IrCallExtern_symbol_path",
            "native_v2_executable_mode_fails_closed_when_external_relocatable_link_is_required",
            "MachineModule_JSON_exports_external_call_symbol_name_for_relocation_followup",
            "native_v2_emit_obj_exports_an_ET_REL_object_with_R_X86_64_PLT32_to_passthru_f128",
            "linked_C__Float128_passthrough_oracle_exits_zero",
            "madaros_wrapper_native_v2_link_emits_a_linked_executable_for_the_scalar_passthrough_oracle",
            "extern_C_f128_aggregate_sret_front_half_typechecks_but_native_v2_fails_closed_without_crash_or_elf",
            "madaros_wrapper_native_v2_link_refuses_aggregate_sret_until_external_sysv_abi_is_promoted",
        ],
        "missing_full_obligations": [
            "self-hosted native-v2 direct executable mode still requires an internal/external linker path for unresolved externs",
            "general external SysV ABI classification beyond exact scalar f128 passthrough",
            "external aggregate/SRET ABI oracle coverage",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        f"madaros-v2-s5-external-sysv-f128-blocker: cases={receipt['case_count']} "
        f"blocked={receipt['blocked']} sha={receipt['receipt_sha256'][:12]} receipt={receipt_path}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit_p = sub.add_parser("emit")
    emit_p.add_argument("--out-dir", required=True)
    emit_p.add_argument("--compiler", default=str(repo_root_from_script() / "bin" / "madaros"))
    emit_p.add_argument("--root", default=str(repo_root_from_script()))
    emit_p.add_argument("--timeout", type=int, default=120)
    emit_p.set_defaults(func=emit)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
