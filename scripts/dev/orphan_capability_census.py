#!/usr/bin/env python3
"""Find capability that exists and has no consumer.

The pattern this looks for was measured three times in one day and never
once on purpose:

  - the ZD surgical family: eight types, a Lean proof with no `sorry`, and a
    compiler that checks only that you wrote `with ZD`;
  - the provenance classifiers: `provenance_is_trusted`,
    `provenance_kind_needs_verification` -- a full trust taxonomy, reached by
    a live call chain, deciding nothing because `TypeEntry` drops the field;
  - the unit and refinement resolvers: present on one type-lowering spine and
    absent from the one annotations actually traverse.

A function is reported when nothing outside its own file names it. That is a
floor, not a verdict: a stdlib entry point with no in-tree caller may be a
deliberate public API, and this script cannot tell that from abandonment.
It says where to look, not what is wrong.

Usage:  python3 scripts/dev/orphan_capability_census.py [--min-lines N] [--json OUT]
"""
import os, re, sys, json, collections

ROOTS = ['stdlib', 'self-hosted']
CONSUMERS = ['stdlib', 'self-hosted', 'tests', 'examples', 'benchmarks']
SKIP_DIRS = {'.git', 'archive', 'bootstrap', 'node_modules', 'target', '.claude'}

DEF = re.compile(r'^\s*(pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', re.M)

def sio_files(roots):
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.endswith('.sio'):
                    yield os.path.join(dirpath, fn)

def main():
    min_lines = 3
    out_json = None
    a = sys.argv[1:]
    for i, x in enumerate(a):
        if x == '--min-lines' and i + 1 < len(a): min_lines = int(a[i+1])
        if x == '--json' and i + 1 < len(a): out_json = a[i+1]

    # 1. every definition, with the file that owns it
    owner = {}          # name -> set of files defining it
    is_pub = {}
    body_len = {}
    for p in sio_files(ROOTS):
        try: src = open(p, errors='ignore').read()
        except OSError: continue
        lines = src.split('\n')
        for m in DEF.finditer(src):
            name = m.group(2)
            owner.setdefault(name, set()).add(p)
            is_pub[name] = is_pub.get(name, False) or bool(m.group(1))
            start = src[:m.start()].count('\n')
            depth, n = 0, 0
            for l in lines[start:start+400]:
                depth += l.count('{') - l.count('}')
                n += 1
                if n > 1 and depth <= 0: break
            body_len[name] = max(body_len.get(name, 0), n)

    # 2. every mention, with the file it appears in
    mention = collections.defaultdict(set)
    mention_count = collections.Counter()
    word = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
    for p in sio_files(CONSUMERS):
        try: src = open(p, errors='ignore').read()
        except OSError: continue
        src = re.sub(r'//.*', '', src)
        for w in word.findall(src):
            if w in owner:
                mention[w].add(p)
                mention_count[w] += 1

    orphans = []
    for name, files in owner.items():
        if len(files) > 1:      # homonyms: cannot attribute, skip
            continue
        f = next(iter(files))
        # Mentions ANYWHERE, its own file included. The first version counted
        # only mentions outside the defining file and flagged 57% of the tree --
        # every private helper called by its own neighbours. That is
        # encapsulation, not abandonment. What is wanted is a name nothing
        # utters: no caller inside, no caller outside, no test, no example.
        if mention_count[name] > 1:
            continue
        if body_len.get(name, 0) < min_lines:
            continue
        orphans.append({'name': name, 'file': f,
                        'pub': is_pub.get(name, False),
                        'lines': body_len.get(name, 0)})

    orphans.sort(key=lambda o: (-o['lines'], o['file'], o['name']))
    by_mod = collections.Counter(o['file'] for o in orphans)

    print(f"defined functions scanned : {len(owner)}")
    print(f"with no consumer outside their own file : {len(orphans)}")
    print(f"  of those, pub : {sum(1 for o in orphans if o['pub'])}")
    print()
    print("modules holding the most, with the largest orphan in each:")
    for mod, n in by_mod.most_common(25):
        big = max((o for o in orphans if o['file'] == mod), key=lambda o: o['lines'])
        print(f"  {n:4d}  {mod:52s}  largest: {big['name']} ({big['lines']} lines)")

    if out_json:
        os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
        with open(out_json, 'w') as fh:
            json.dump({'total_defined': len(owner), 'orphans': orphans}, fh, indent=1)
        print(f"\nfull list: {out_json}")

if __name__ == '__main__':
    main()
