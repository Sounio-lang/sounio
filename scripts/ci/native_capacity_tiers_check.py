#!/usr/bin/env python3
"""Static checker behind scripts/ci/native_capacity_tiers_gate.sh.

Pins the native x86-64 backend capacity tiers to their named accessors and proves
that every overflow path is fail-closed with a distinct return code. See the gate
script for the full rationale.
"""
import pathlib
import re
import sys

LB = chr(123)
RB = chr(125)

SOURCES = [
    ("encode", "self-hosted/native/encode.sio"),
    ("frame", "self-hosted/native/frame.sio"),
    ("cgx", "self-hosted/native/codegen_x86_linux.sio"),
    ("cg", "self-hosted/native/codegen.sio"),
    ("ir", "self-hosted/ir/ir.sio"),
    ("elf", "self-hosted/native/elf.sio"),
    ("lower_ir", "self-hosted/native/lower_ir.sio"),
    ("driver", "self-hosted/compiler/native_compile_driver.sio"),
]


def fail(reason):
    print("NATIVE_CAPACITY_TIERS_FAIL reason=%s" % reason, file=sys.stderr)
    raise SystemExit(1)


def only(pattern, subject, reason, flags=re.MULTILINE):
    matches = list(re.finditer(pattern, subject, flags))
    if len(matches) != 1:
        fail("%s_count_%d" % (reason, len(matches)))
    return matches[0]


def accessor(name, visibility):
    # e.g. r"^pub fn nc_code_capacity_bytes\(\) -> i64 \{ ([0-9]+) \}$"
    return "^%sfn %s\\(\\) -> i64 \\%s ([0-9]+) \\%s$" % (visibility, name, LB, RB)


def tier(label, decl_match, acc_match, expected):
    vals = set(int(g) for g in decl_match.groups())
    vals.add(int(acc_match.group(1)))
    if vals != set([expected]):
        fail("%s_expected_%d_got_%s" % (label, expected, sorted(vals)))


def main():
    root = pathlib.Path(sys.argv[1])
    exp_code = int(sys.argv[2])
    exp_reloc = int(sys.argv[3])
    exp_elf = int(sys.argv[4])
    exp_legacy_elf = int(sys.argv[5])
    base_addr = int(sys.argv[6])
    exp_label = int(sys.argv[7])
    exp_rodata = int(sys.argv[8])
    exp_return_jumps = int(sys.argv[9])
    exp_strtab = int(sys.argv[10])
    exp_drv_label = int(sys.argv[11])

    loaded = []
    for _key, rel in SOURCES:
        path = root / rel
        if not path.is_file():
            fail("source_missing_%s" % rel.replace("/", "_"))
        loaded.append(path.read_text(encoding="utf-8"))
    encode, frame, cgx, cg, irsrc, elfsrc, lowerir, driver = loaded

    # Tier 1: NC_BIG_CODE. Declarations stay literals; bound checks use the
    # accessor. Pinned together so a half-applied bump cannot ship.
    tier("code_tier",
         only(r"^pub var NC_BIG_CODE: \[i8; ([0-9]+)\] = \[0; ([0-9]+)\]$",
              encode, "nc_big_code_decl"),
         only(accessor("nc_code_capacity_bytes", "pub "), encode,
              "nc_code_capacity_accessor"),
         exp_code)

    # Tier 2: the four NC_FLAT_RELOC_* arrays must agree with each other AND
    # with nc_flat_reloc_capacity().
    reloc_decls = re.findall(
        r"^pub var NC_FLAT_RELOC_(?:OFFSETS|KIND_CODES|IS_FUNCTIONS|TARGET_INDICES): "
        r"\[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$", frame, re.MULTILINE)
    if len(reloc_decls) != 4:
        fail("nc_flat_reloc_decl_count_%d" % len(reloc_decls))
    reloc_acc = only(accessor("nc_flat_reloc_capacity", "pub "), frame,
                     "nc_flat_reloc_accessor")
    vals = set(int(v) for pair in reloc_decls for v in pair)
    vals.add(int(reloc_acc.group(1)))
    if vals != set([exp_reloc]):
        fail("reloc_tier_expected_%d_got_%s" % (exp_reloc, sorted(vals)))

    # Tier 3/4: the native_v2 ELF image buffer and the legacy one.
    tier("elf_tier",
         only(r"^var NC_BIG_ELF: \[i8; ([0-9]+)\] = \[0; ([0-9]+)\]$",
              cgx, "nc_big_elf_decl"),
         only(accessor("nc_elf_capacity_bytes", ""), cgx,
              "nc_elf_capacity_accessor"),
         exp_elf)
    tier("legacy_elf_tier",
         only(r"^var NATIVE_ELF_BUF: \[i8; ([0-9]+)\] = \[0; ([0-9]+)\]$",
              cgx, "native_elf_buf_decl"),
         only(accessor("native_elf_buf_capacity_bytes", ""), cgx,
              "native_elf_buf_accessor"),
         exp_legacy_elf)

    # Tier 5: the three NC_V2_LABEL_* arrays, the per-function label/jump-patch
    # tier. Both of its bounds used to be silent (no else, no sentinel), which is
    # what made a function with more than ~128 `if`s miscompile at rc=0.
    label_decls = re.findall(
        r"^pub var NC_V2_LABEL_(?:OFFSETS|PATCH_OFFSETS|PATCH_IDS): "
        r"\[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$", frame, re.MULTILINE)
    if len(label_decls) != 3:
        fail("nc_v2_label_decl_count_%d" % len(label_decls))
    label_acc = only(accessor("nc_v2_label_capacity", "pub "), frame,
                     "nc_v2_label_accessor")
    vals = set(int(v) for pair in label_decls for v in pair)
    vals.add(int(label_acc.group(1)))
    if vals != set([exp_label]):
        fail("label_tier_expected_%d_got_%s" % (exp_label, sorted(vals)))

    # The tier is only PROVABLY non-overflowing while it is at least as large as
    # IR_MAX_INSTRS: every label and every patch originates from an IR instruction
    # read by the one emit loop, and that loop is bounded by IR_MAX_INSTRS. Pinned
    # as >=, not ==, so raising IR_MAX_INSTRS is caught here instead of silently
    # invalidating the argument.
    ir_max = only(r"^pub let IR_MAX_INSTRS: i64 = ([0-9]+)\s*$", irsrc,
                  "ir_max_instrs_decl")
    if exp_label < int(ir_max.group(1)):
        fail("label_tier_%d_below_ir_max_instrs_%s" % (exp_label, ir_max.group(1)))

    # Past 65536 the full per-function reset stops being free and needs a dirty
    # cursor. Force that decision here rather than letting it be discovered.
    if exp_label > 65536 and "NC_V2_LABEL_DIRTY_LEN" not in frame:
        fail("label_tier_%d_needs_dirty_cursor" % exp_label)

    # -- 2. no surviving duplicate literal bound checks --------------------
    # A comparison against a tier literal anywhere outside a declaration means a
    # call site was missed, and raising the tier is a silent no-op there.
    pairs = [("encode", encode, exp_code),
             ("codegen_x86_linux", cgx, exp_code),
             ("codegen", cg, exp_code),
             ("frame", frame, exp_reloc)]
    for name, subject, lit in pairs:
        bound = re.compile(r"[<>]=?\s*%d\b" % lit)
        for line in subject.splitlines():
            if bound.search(line) and "var " not in line:
                fail("stray_literal_bound_%s_%d" % (name, lit))
    # The RETIRED tier literals must not reappear as a bound in any of the four
    # backend sources: a missed call site is exactly how a tier raise becomes a
    # silent partial no-op (the defect this gate exists for). 2097152 has no
    # legitimate use left here. 65536 does (the unrelated narrow result buffer
    # `[i8; 65536]` in native_compile_result_ok_narrow), so for that literal only
    # relocation-table lines are flagged.
    old_code = re.compile(r"[<>]=?\s*2097152\b")
    old_reloc = re.compile(r"[<>]=?\s*65536\b")
    for name, subject, _lit in pairs:
        for line in subject.splitlines():
            if old_code.search(line):
                fail("stale_literal_bound_%s_2097152" % name)
            if old_reloc.search(line) and "flat_reloc" in line.lower():
                fail("stale_literal_bound_%s_65536" % name)

    # -- 3/4. relocation overflow is fail-closed with a DISTINCT rc --------
    if not re.search(r"pub reloc_overflow: bool,", frame):
        fail("reloc_overflow_field_missing")
    if not re.search(r"out\.reloc_overflow = false", frame):
        fail("reloc_overflow_not_initialised")

    body = only(r"^fn nc_add_flat_reloc\(.*?\n\%s$" % RB, cgx,
                "nc_add_flat_reloc", re.MULTILINE | re.DOTALL).group(0)
    if "nc_flat_reloc_capacity()" not in body:
        fail("nc_add_flat_reloc_not_using_accessor")
    sentinel = r"\%s\s*else\s*\%s(?:.|\n)*?\(\*nc\)\.reloc_overflow = true" % (RB, LB)
    if not re.search(sentinel, body):
        fail("nc_add_flat_reloc_missing_sentinel")
    only(r"^\s*while i < \(\*nc\)\.flat_reloc_count && i < nc_flat_reloc_capacity\(\) \%s$" % LB,
         cgx, "apply_relocations_bound")

    # rc=19 (code buffer) and rc=20 (relocations) must both exist, distinctly.
    # rc19 allows a diagnostic body between the brace and the return (matching
    # the rc22/label_overflow precedent below) -- 2026-09-02: the branch now
    # reports code_overflow_bytes_attempted so a capacity bump is sized against
    # a measured number instead of a guess. rc20 stays bare; no diagnostic body
    # has been added there yet.
    rc19 = r"if nc\.code_overflow \%s(?:.|\n)*?return 19\s*\n\s*\%s" % (LB, RB)
    # rc20 loosened the same way and for the same reason (2026-09-02):
    # reloc_overflow_count_attempted is now reported before returning.
    rc20 = r"if nc\.reloc_overflow \%s(?:.|\n)*?return 20\s*\n\s*\%s" % (LB, RB)
    if not re.search(rc19, cgx):
        fail("rc19_code_overflow_check_missing")
    if not re.search(rc20, cgx):
        fail("rc20_reloc_overflow_check_missing")

    # -- label overflow is fail-closed with its own distinct rc=22 ---------
    if not re.search(r"pub label_overflow: bool,", frame):
        fail("label_overflow_field_missing")
    if not re.search(r"out\.label_overflow = false", frame):
        fail("label_overflow_not_initialised")
    for fn_name in ("nc_add_label_patch", "nc_define_label"):
        fn_body = only(r"^fn %s\(.*?\n\%s$" % (fn_name, RB), cgx,
                       fn_name, re.MULTILINE | re.DOTALL).group(0)
        if "nc_v2_label_capacity()" not in fn_body:
            fail("%s_not_using_accessor" % fn_name)
        label_sentinel = (r"\%s\s*else\s*\%s(?:.|\n)*?nc_note_label_overflow\(nc\)"
                          % (RB, LB))
        if not re.search(label_sentinel, fn_body):
            fail("%s_missing_sentinel" % fn_name)
    # The reset must clear the WHOLE tier, not a per-function extent: a partial
    # reset leaves the previous function's real code offsets live above the bound,
    # which is the same wrong-target defect in a different shape.
    only(r"^\s*while i < nc_v2_label_capacity\(\) \%s$" % LB, cgx,
         "label_reset_full_bound")
    # patch_lid must be bounded before it indexes NC_V2_LABEL_OFFSETS. Unchecked,
    # it read past the end of the array into the adjacent globals -- code offsets,
    # always >= 0, so they passed the `target >= 0` guard and patched a plausible
    # but WRONG branch target.
    only(r"^\s*if patch_lid >= 0 && patch_lid < nc_v2_label_capacity\(\) \%s$" % LB,
         frame, "label_patch_lid_bounded")
    rc22 = r"if nc\.label_overflow \%s(?:.|\n)*?return 22\s*\n\s*\%s" % (LB, RB)
    if not re.search(rc22, cgx):
        fail("rc22_label_overflow_check_missing")

    # -- 5. the legacy NATIVE_ELF_BUF writer is no longer unguarded --------
    putu8 = only(r"^fn nc_elf_put_u8\(val: i64\) with Mut, Panic \%s(?:.|\n)*?^\%s$"
                 % (LB, RB), cgx, "nc_elf_put_u8", re.MULTILINE).group(0)
    if "native_elf_buf_capacity_bytes()" not in putu8:
        fail("nc_elf_put_u8_unbounded")
    if "NATIVE_ELF_OVERFLOW = 1" not in putu8:
        fail("nc_elf_put_u8_missing_sentinel")

    legacy = only(r"^fn compile_native_finalize_and_write_ref\((?:.|\n)*?^\%s$" % RB,
                  cgx, "compile_native_finalize_and_write_ref", re.MULTILINE).group(0)
    needles = [(r"\(\*nc\)\.code_overflow", "legacy_ignores_code_overflow"),
               (r"\(\*nc\)\.reloc_overflow", "legacy_ignores_reloc_overflow"),
               (r"NATIVE_ELF_OVERFLOW != 0", "legacy_ignores_elf_overflow"),
               (r"return 19", "legacy_missing_rc19"),
               (r"return 20", "legacy_missing_rc20"),
               (r"return 21", "legacy_missing_rc21"),
               (r"\(\*nc\)\.label_overflow", "legacy_ignores_label_overflow"),
               (r"return 22", "legacy_missing_rc22")]
    for needle, reason in needles:
        if not re.search(needle, legacy):
            fail(reason)

    # -- 5b. NC_BIG_RODATA (flat rodata: string/constant data) -- 2026-09-03,
    # #2393. Same shape as the code/reloc/label tiers: fixed array + accessor
    # + true attempted-count field + a distinct fail-closed rc (23). Unlike
    # those three, the append sites here (native_v2_rodata_append_updated_tail,
    # nc_rodata_add_ir_instr_name, nc_rodata_add_name_value,
    # native_v2_rodata_append_byte_into) used to bound-check the write WITHOUT
    # ever setting an overflow flag or refusing -- silent truncation, not a
    # capacity refusal, first measured on self-hosted/compiler/main.sio
    # (963,478 bytes needed against a 256 KiB cap). This tier proves the fix
    # holds together, not that runtime data stays under the new cap.
    tier("rodata_tier",
         only(r"^pub var NC_BIG_RODATA: \[i8; ([0-9]+)\] = \[0; ([0-9]+)\]$",
              frame, "nc_big_rodata_decl"),
         only(accessor("nc_flat_rodata_capacity", "pub "), frame,
              "nc_flat_rodata_capacity_accessor"),
         exp_rodata)
    if not re.search(r"pub rodata_overflow_bytes_attempted: i64,", frame):
        fail("rodata_overflow_bytes_attempted_field_missing")
    if not re.search(r"out\.rodata_overflow_bytes_attempted = 0", frame):
        fail("rodata_overflow_bytes_attempted_not_initialised")
    rc23 = r"if nc\.rodata_overflow_bytes_attempted > nc_flat_rodata_capacity\(\) \%s(?:.|\n)*?return 23\s*\n\s*\%s" % (LB, RB)
    if not re.search(rc23, cgx):
        fail("rc23_rodata_overflow_check_missing")

    # -- 6. the 0x400000 ELF LOAD BASE ADDRESS ---------------------------
    # This is the trap: 4194304 is BOTH the retired 4 MiB ELF capacity and the
    # ELF load address. The load base now has one source of truth,
    # nc_elf_base_addr(), which returns the handle capacity NC_HANDLE_CAPACITY
    # (0x400000 unless SOUNIO_NV2_HANDLE_LOG2 raises both; #2966). The literal
    # may appear only in that capacity's declaration and configuration, never
    # as a bound, and every ELF writer call takes the accessor.
    occ = []
    for line in cgx.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        if re.search(r"\b%d\b" % base_addr, stripped):
            occ.append(stripped)
    allowed = re.compile(r"^(?:var NC_HANDLE_CAPACITY: i64 = %d|NC_HANDLE_CAPACITY = %d|"
                         r"if str_eq\(v, \"22\"\) \{ cap = %d \})$"
                         % (base_addr, base_addr, base_addr))
    for line in occ:
        if re.search(r"[<>]=?\s*%d\b" % base_addr, line):
            fail("elf_base_addr_used_as_bound")
        if not allowed.search(line):
            fail("elf_base_addr_unexpected_form")
    if len(occ) != 3:
        fail("elf_base_addr_occurrence_count_%d" % len(occ))
    if not re.search(r"^fn nc_elf_base_addr\(\) -> i64 with Mut \{ NC_HANDLE_CAPACITY \}$", cgx, re.MULTILINE):
        fail("nc_elf_base_addr_accessor_missing")
    base_calls = len(re.findall(r"nc_elf_base_addr\(\)", cgx)) - 1
    if base_calls != 7:
        fail("nc_elf_base_addr_call_count_%d" % base_calls)

    check_1646(frame, cgx, cg, irsrc, elfsrc, lowerir, driver,
               exp_return_jumps, exp_strtab, exp_drv_label)

    print("NATIVE_CAPACITY_TIERS_CHECK "
          "code=%d reloc=%d elf=%d legacy_elf=%d label=%d rodata=%d base_addr_literals=%d "
          "return_jumps=%d strtab=%d drv_label=%d "
          "fail_closed=rc19/rc20/rc21/rc22/rc23/rc28/drv_rc6 coherent=pass"
          % (exp_code, exp_reloc, exp_elf, exp_legacy_elf, exp_label, exp_rodata, len(occ),
             exp_return_jumps, exp_strtab, exp_drv_label))


def fn_body(name, subject, reason, visibility=""):
    return only(r"^%sfn %s\((?:.|\n)*?^\%s$" % (visibility, name, RB), subject,
                reason, re.MULTILINE).group(0)


def check_1646(frame, cgx, cg, irsrc, elfsrc, lowerir, driver,
               exp_return_jumps, exp_strtab, exp_drv_label):
    ir_max = int(only(r"^pub let IR_MAX_INSTRS: i64 = ([0-9]+)\s*$", irsrc,
                      "ir_max_instrs_decl_1646").group(1))
    ir_max_vregs = int(only(r"^pub let IR_MAX_VREGS: i64 = ([0-9]+)\s*$", irsrc,
                            "ir_max_vregs_decl").group(1))

    # -- 10a. return-jump tier -------------------------------------------
    tier("return_jump_tier",
         only(r"^pub var NC_RETURN_JUMPS: \[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$",
              frame, "nc_return_jumps_decl"),
         only(accessor("nc_return_jump_capacity", "pub "), frame,
              "nc_return_jump_capacity_accessor"),
         exp_return_jumps)
    if exp_return_jumps < ir_max:
        fail("return_jump_tier_%d_below_ir_max_instrs_%d" % (exp_return_jumps, ir_max))
    for name, subject in (("frame", frame), ("codegen_x86_linux", cgx), ("codegen", cg)):
        if re.search(r"return_jump_count\s*<\s*16\b", subject) or \
           re.search(r"rj_i\s*<\s*16\b", subject):
            fail("stray_return_jump_literal_bound_%s" % name)
    rec = fn_body("record_return_jump_mut", frame, "record_return_jump_mut", "pub ")
    if "nc_return_jump_capacity()" not in rec:
        fail("record_return_jump_mut_not_using_accessor")
    if not re.search(r"\%s\s*else\s*\%s\s*\(\*nc\)\.label_overflow = true" % (RB, LB), rec):
        fail("record_return_jump_mut_missing_sentinel")
    mir_arm = (r"if c\.return_jump_count < nc_return_jump_capacity\(\) \%s(?:.|\n)*?"
               r"\%s\s*else\s*\%s(?:.|\n)*?c\.label_overflow = true" % (LB, RB, LB))
    for name, subject in (("codegen_x86_linux", cgx), ("codegen", cg)):
        if len(re.findall(mir_arm, subject)) != 1:
            fail("mir_ret_arm_missing_sentinel_%s" % name)

    # -- 10b. float-typing tier ------------------------------------------
    fl = only(r"^pub var NC_IS_FLOAT_REG: \[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$",
              frame, "nc_is_float_reg_decl")
    limit = int(only(r"^pub let NC_VREG_LIMIT: i64 = ([0-9]+)$", frame,
                     "nc_vreg_limit_decl").group(1))
    vals = set(int(g) for g in fl.groups())
    vals.add(limit)
    vals.add(ir_max_vregs)
    if len(vals) != 1:
        fail("float_typing_tier_not_ir_max_vregs_%s" % sorted(vals))
    for name, subject in (("codegen_x86_linux", cgx), ("codegen", cg), ("lower_ir", lowerir),
                          ("frame", frame)):
        lines = subject.splitlines()
        for i, line in enumerate(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if "NC_MAX_VREGS" in line and ("NC_IS_FLOAT_REG[" in line or
                                           ("NC_IS_FLOAT_REG[" in nxt and not line.strip().startswith("while"))):
                fail("float_typing_bounded_by_nc_max_vregs_%s_line_%d" % (name, i + 1))
            if line.strip().startswith("while") and "NC_MAX_VREGS" in line and "NC_IS_FLOAT_REG[" in nxt:
                fail("float_typing_partial_reset_%s_line_%d" % (name, i + 1))

    # -- 10c. string-table tier (elf.sio) --------------------------------
    tier("strtab_tier",
         only(r"^pub struct StringTable \%s\n    pub bytes: \[i8; ([0-9]+)\],$" % LB, elfsrc,
              "string_table_bytes_decl"),
         only(accessor("elf_strtab_capacity_bytes", "pub "), elfsrc,
              "elf_strtab_capacity_accessor"),
         exp_strtab)
    for fn_name in ("string_table_add", "data_section_align", "data_section_add_f64"):
        body = fn_body(fn_name, elfsrc, fn_name, "pub ")
        if "elf_strtab_capacity_bytes()" not in body:
            fail("%s_not_using_accessor" % fn_name)
        if "ELF_STRTAB_OVERFLOW = 1" not in body:
            fail("%s_missing_sentinel" % fn_name)
        if re.search(r"<\s*8192\b", body):
            fail("%s_stray_literal_bound" % fn_name)
    f64 = fn_body("data_section_add_f64", elfsrc, "data_section_add_f64_body", "pub ")
    guard = f64.find("if t.len + 8 > elf_strtab_capacity_bytes()")
    bump = f64.find("t.len = t.len + 8")
    if guard < 0 or bump < 0 or bump < guard:
        fail("data_section_add_f64_len_advanced_before_guard")
    rc28 = r"if ELF_STRTAB_OVERFLOW != 0 \%s(?:.|\n)*?return 28\s*\n\s*\%s" % (LB, RB)
    if len(re.findall(rc28, cgx)) != 2:
        fail("rc28_strtab_overflow_check_count_%d" % len(re.findall(rc28, cgx)))
    fin = fn_body("finalize_compiled_elf", cgx, "finalize_compiled_elf", "pub ")
    if "ELF_STRTAB_OVERFLOW != 0" not in fin or "nc.label_overflow" not in fin:
        fail("finalize_compiled_elf_ignores_overflow")
    if "ELF_STRTAB_OVERFLOW = 0" not in fn_body("compiler_new", frame, "compiler_new_1646", "pub "):
        fail("strtab_overflow_not_reset_per_compile")

    # -- 10d. driver label tier ------------------------------------------
    drv_decls = re.findall(
        r"^var DRV_LABEL_(?:OFFSETS|PATCH_OFFSETS|PATCH_IDS): "
        r"\[i64; ([0-9]+)\] = \[0; ([0-9]+)\]$", driver, re.MULTILINE)
    if len(drv_decls) != 3:
        fail("drv_label_decl_count_%d" % len(drv_decls))
    drv_acc = only(accessor("drv_label_capacity", ""), driver, "drv_label_capacity_accessor")
    vals = set(int(v) for pair in drv_decls for v in pair)
    vals.add(int(drv_acc.group(1)))
    if vals != set([exp_drv_label]):
        fail("drv_label_tier_expected_%d_got_%s" % (exp_drv_label, sorted(vals)))
    for fn_name in ("drv_define_label", "drv_add_label_patch", "drv_patch_labels"):
        body = fn_body(fn_name, driver, fn_name)
        if "drv_label_capacity()" not in body:
            fail("%s_not_using_accessor" % fn_name)
        if not re.search(r"\%s\s*else\s*\%s\s*DRV_LABEL_OVERFLOW = 1" % (RB, LB), body):
            fail("%s_missing_sentinel" % fn_name)
        if re.search(r"<\s*1024\b", body):
            fail("%s_stray_literal_bound" % fn_name)
    if "DRV_LABEL_OFFSETS[lid as usize] >= 0" not in fn_body("drv_patch_labels", driver, "drv_patch_labels_target"):
        fail("drv_patch_labels_patches_undefined_target")
    if not re.search(r"if DRV_LABEL_OVERFLOW != 0 \%s(?:.|\n)*?return 6\s*\n\s*\%s" % (LB, RB), driver):
        fail("drv_rc6_label_overflow_check_missing")
    if "DRV_LABEL_OVERFLOW = 0" not in fn_body("drv_reset_codegen", driver, "drv_reset_codegen"):
        fail("drv_label_overflow_not_reset")


if __name__ == "__main__":
    main()
