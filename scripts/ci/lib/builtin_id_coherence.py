"""Check that every ffi_* builtin resolves to ONE id in every table that names it."""
import collections
import os
import re
import sys

# ffi_malloc and ffi_free reuse the general-purpose heap emitters rather than
# carrying an emitter of their own, so the consumer row names heap_alloc /
# heap_free. Every other name emits under its own name.
EMITTER_ALIAS = {"ffi_malloc": "heap_alloc", "ffi_free": "heap_free"}
PRODUCER = re.compile(r"name_is_(ffi_[a-z_0-9]+?)\(.*?\)\s*\{\s*return (\d+)")
CONSUMER = re.compile(
    r"builtin_id == (\d+) \{ "
    r"(?:native_v2_persist_builtin_emit_into\(nc, )?"
    r"emit_builtin_([a-z_0-9]+?)(?:_into)?\("
)
# The ffi_* range. Below this are main's own builtins, which this gate does not
# police -- it exists for the parallel range added alongside them.
FFI_ID_FLOOR = 39


# native_v2_builtin_id_for_name_ref is a THIRD producer table in the same file.
# Measured 2026-08-23: it holds 16 names, carries no ffi_* row, and has ZERO
# callers anywhere in self-hosted/ -- the only occurrence of its name in the
# tree is its own definition. It is therefore excluded from the agreement check
# above, which would otherwise report every ffi_* as diverging from a table
# nothing consults.
#
# That exclusion is only safe while it stays dead, so the gate checks that too.
# If someone wires it up without adding the ffi_* rows, every FFI call resolved
# through it silently stops being a builtin and lowers as a call to a function
# that does not exist.
#
# Not checked here, deliberately: self-hosted/native/codegen.sio defines its own
# native_v2_builtin_id_for_name and native_v2_emit_builtin_by_id_into carrying
# ids 1-19 only. That is a legacy copy imported by test harnesses and standalone
# drivers; self-hosted/compiler/main.sio imports native::codegen_x86_linux and
# NOT native::codegen, so the two never coexist in the shipping compiler.
DEAD_TABLE = "native_v2_builtin_id_for_name_ref"


def check_dead_table_stays_dead(root: str) -> int:
    """The excluded table must have no callers, or the exclusion is unsound."""
    import subprocess

    try:
        out = subprocess.run(
            ["grep", "-rn", DEAD_TABLE, os.path.join(root, "self-hosted"), "--include=*.sio"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception as exc:
        print(f"  could not scan for callers of {DEAD_TABLE}: {exc}")
        return 1
    lines = [l for l in out.splitlines() if l.strip()]
    definitions = [l for l in lines if f"fn {DEAD_TABLE}(" in l]
    callers = [l for l in lines if l not in definitions]
    if callers:
        print(f"  {DEAD_TABLE} has acquired {len(callers)} caller(s) and still carries no ffi_* row:")
        for line in callers:
            print(f"    {line.strip()[:120]}")
        print("  Either add the ffi_* rows to it or remove the caller. While it was dead the")
        print("  omission was harmless; with a caller, those calls stop being builtins.")
        return 1
    print(f"  {DEAD_TABLE}: still 0 callers, exclusion holds")
    return 0


def main(path: str) -> int:
    producers = collections.defaultdict(list)
    consumers = collections.defaultdict(list)
    for line in open(path):
        m = PRODUCER.search(line)
        if m:
            producers[m.group(1)].append(int(m.group(2)))
        m = CONSUMER.search(line)
        if m:
            consumers[m.group(2)].append(int(m.group(1)))

    if not producers:
        print("no ffi_* producer rows found at all -- the pattern no longer matches")
        return 1

    bad = 0
    for name in sorted(producers):
        ids = sorted(set(producers[name]))
        count = len(producers[name])
        emitted = sorted(
            i for i in consumers.get(EMITTER_ALIAS.get(name, name), []) if i >= FFI_ID_FLOOR
        )
        ok = len(ids) == 1 and ids == emitted and count == 2
        if not ok:
            bad += 1
        print(f"  {name:14s} producers={count}x{ids} emitter={emitted} {'ok' if ok else 'DIVERGES'}")

    every = [i for v in consumers.values() for i in v]
    dupes = sorted({i for i in set(every) if every.count(i) > 1})
    if dupes:
        # A duplicate id is not a cosmetic problem: the dispatch returns on
        # first match, so the later row is dead and its builtin silently
        # becomes the earlier one.
        bad += 1
        print(f"  DUPLICATE ids in the emitter, first match wins: {dupes}")

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    bad += check_dead_table_stays_dead(root)

    print(f"  problems={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
