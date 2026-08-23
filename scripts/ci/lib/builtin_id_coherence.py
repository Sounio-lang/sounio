"""Check that every ffi_* builtin resolves to ONE id in every table that names it."""
import collections
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

    print(f"  problems={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
