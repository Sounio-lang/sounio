#!/usr/bin/env python3
"""Static checker behind scripts/ci/dynlink_tables_coherence_gate.sh.

Two hand-written tables in self-hosted/native/codegen_x86_linux.sio encode the
dynlink symbol->library mapping:

    native_v2_dynlink_lib_id_for_name   symbol name -> lib id
    native_v2_dynlink_soname_for_lib_id lib id      -> DT_NEEDED soname

Nothing in the language ties them together, and they fail ASYMMETRICALLY. A lib
id present in the first table but missing from the second does not error: the
soname lookup falls off the end of its chain and returns its final literal
("libkl14b_probe.so"). The ELF then carries a real DT_NEEDED naming the WRONG
library, ld.so loads it, and the symbol resolves to whatever that library
happens to export. It assembles, it links, it runs.

So the two id sets must be EQUAL, not merely overlapping. Equal is also not
enough on its own: the soname chain must END in a genuine row (not the bare
fallback) so that a future extra id cannot ride the fallback silently. Both
facts are checked here, and the extraction is floored so that a regex which
stops matching cannot report agreement over zero rows.
"""
import pathlib
import re
import sys

LB = chr(123)
RB = chr(125)


def fail(reason):
    print("DYNLINK_TABLES_FAIL reason=%s" % reason, file=sys.stderr)
    raise SystemExit(1)


def body(source, fn_name):
    # From `fn <name>(` to the next top-level closing brace at column 0.
    pattern = r"^fn %s\(.*?\n\%s$" % (re.escape(fn_name), RB)
    m = re.search(pattern, source, re.MULTILINE | re.DOTALL)
    if not m:
        fail("function_body_missing_%s" % fn_name)
    return m.group(0)


def main():
    root = pathlib.Path(sys.argv[1])
    rel = "self-hosted/native/codegen_x86_linux.sio"
    path = root / rel
    if not path.is_file():
        fail("source_missing_%s" % rel.replace("/", "_"))
    src = path.read_text(encoding="utf-8")

    name_body = body(src, "native_v2_dynlink_lib_id_for_name")
    son_body = body(src, "native_v2_dynlink_soname_for_lib_id")

    # -- symbol -> id: the ids reachable from real name comparisons ----------
    # Only `if native_v2_name_buf_eq(...) { return N }` rows carry a mapping.
    sym_rows = re.findall(
        r"if native_v2_name_buf_eq\([^)]*\)\s*\%s\s*return ([0-9]+)\s*\%s"
        % (LB, RB), name_body)
    if len(sym_rows) == 0:
        fail("symbol_table_extraction_empty")
    sym_ids = sorted(set(int(v) for v in sym_rows))

    # -- id -> soname: the ids the soname chain actually answers ------------
    son_ids = sorted(set(int(v) for v in
                         re.findall(r"if lib_id == ([0-9]+)\s*\%s" % LB, son_body)))
    if len(son_ids) == 0:
        fail("soname_table_extraction_empty")

    # -- the soname chain must end in a row, not the bare fallback ----------
    # A trailing `return "..."` that is not guarded by `if lib_id == N` means
    # the last row is missing and unknown ids ride the fallback.
    tail = son_body.rstrip()
    if not tail.endswith(RB):
        fail("soname_body_unterminated")
    inner = tail[: tail.rfind(RB)]
    last_stmt = inner.rstrip().splitlines()[-1].strip()
    # Sounio returns the trailing expression implicitly, so the fallback is
    # written as a bare string literal, not `return \"...\"`.
    if not re.match(r"(?:return\s+)?\"[^\"]+\"$", last_stmt):
        fail("soname_fallback_not_final_literal")
    if "lib_id ==" in last_stmt:
        fail("soname_last_row_is_guarded_not_fallback")

    # -- the two id sets must be EQUAL --------------------------------------
    missing_in_soname = [i for i in sym_ids if i not in son_ids]
    if missing_in_soname:
        # This is the silent-wrong-library case, named explicitly.
        fail("soname_missing_for_ids_%s"
             % "_".join(str(i) for i in missing_in_soname))
    orphan_soname = [i for i in son_ids if i not in sym_ids]
    if orphan_soname:
        # A soname row nobody can select is dead weight and usually the other
        # half of a renumbering that only half-landed.
        fail("soname_orphan_ids_%s" % "_".join(str(i) for i in orphan_soname))

    # -- one symbol name must not map to two ids ----------------------------
    # (Two DIFFERENT names sharing an id is normal: ZSTD_* all live in libzstd.)
    seen = {}
    for nm, lid in re.findall(
            r"make_name\(\"([^\"]+)\"\)\s*\n\s*if native_v2_name_buf_eq\([^)]*\)\s*\%s\s*return ([0-9]+)"
            % LB, name_body):
        if nm in seen and seen[nm] != lid:
            fail("symbol_name_double_mapped_%s" % nm)
        seen[nm] = lid
    if len(seen) == 0:
        fail("symbol_name_pairs_empty")

    print("DYNLINK_TABLES_CHECK symbols=%d distinct_names=%d lib_ids=%s "
          "soname_rows=%d coherent=pass"
          % (len(sym_rows), len(seen),
             ",".join(str(i) for i in sym_ids), len(son_ids)))


if __name__ == "__main__":
    main()
