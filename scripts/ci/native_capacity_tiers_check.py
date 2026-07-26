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

    loaded = []
    for _key, rel in SOURCES:
        path = root / rel
        if not path.is_file():
            fail("source_missing_%s" % rel.replace("/", "_"))
        loaded.append(path.read_text(encoding="utf-8"))
    encode, frame, cgx, cg = loaded

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
    rc19 = r"if nc\.code_overflow \%s\s*\n\s*return 19\s*\n\s*\%s" % (LB, RB)
    rc20 = r"if nc\.reloc_overflow \%s\s*\n\s*return 20\s*\n\s*\%s" % (LB, RB)
    if not re.search(rc19, cgx):
        fail("rc19_code_overflow_check_missing")
    if not re.search(rc20, cgx):
        fail("rc20_reloc_overflow_check_missing")

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
               (r"return 21", "legacy_missing_rc21")]
    for needle, reason in needles:
        if not re.search(needle, legacy):
            fail(reason)

    # -- 6. the 0x400000 ELF LOAD BASE ADDRESS must survive as a literal ---
    # This is the trap: 4194304 is BOTH the retired 4 MiB ELF capacity and the
    # ELF load address. Every remaining occurrence must be a plain argument.
    occ = []
    for line in cgx.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        if re.search(r"\b%d\b" % base_addr, stripped):
            occ.append(stripped)
    if len(occ) != 6:
        fail("elf_base_addr_occurrence_count_%d" % len(occ))
    arg_form = re.compile(r"(?:,\s*%d\)|,\s*%d,|let base_addr: i64 = %d)"
                          % (base_addr, base_addr, base_addr))
    for line in occ:
        if re.search(r"[<>]=?\s*%d\b" % base_addr, line):
            fail("elf_base_addr_used_as_bound")
        if not arg_form.search(line):
            fail("elf_base_addr_unexpected_form")

    print("NATIVE_CAPACITY_TIERS_CHECK "
          "code=%d reloc=%d elf=%d legacy_elf=%d base_addr_literals=%d "
          "fail_closed=rc19/rc20/rc21 coherent=pass"
          % (exp_code, exp_reloc, exp_elf, exp_legacy_elf, len(occ)))


if __name__ == "__main__":
    main()
