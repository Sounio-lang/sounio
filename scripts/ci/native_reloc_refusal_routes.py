#!/usr/bin/env python3
"""Derive every route from apply_relocations_into to a file write, and fail if
one of them reaches the write without passing native_reloc_refuse_if_invalid.

Called by scripts/ci/native_reloc_unknown_kind_gate.sh. Nothing here is a fixed
list of writers: the routes are read out of self-hosted/ every run.

Model (top-level `fn` bodies, `fn NAME(` to the first `}` in column 0):

  relocating call  a call to apply_relocations_into, or to a function that
                   applies relocations to the NativeCompiler its CALLER passed
                   in (it calls a relocating function with its own `nc`
                   parameter and does not write). Found by a fixpoint, so a new
                   wrapper is picked up without editing this file.
  write primitive  write_file( / io_write_*( -- a file write.
  writer           a function whose body contains a write primitive, or calls a
                   writer (fixpoint).
  guard            `let V = native_reloc_refuse_if_invalid(...)` followed, on
                   the next code line, by `if V != 0 { return V }`.

For every function F with a relocating call, take the lines after its first
relocating call. Each write primitive there, and each call there to a writer W,
must be covered: a guard in F between the relocating call and that line, or
(for a call to W) W itself guarded before its first write primitive or writer
call. A function that writes nothing after relocating and relocates its own
local NativeCompiler drops the image (a self-test); one that relocates the
caller's `nc` hands the image to its callers, which are checked in turn.

Exit 0 when every route is covered, 1 otherwise. Prints one line per route.
"""
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
SRC_DIR = os.path.join(ROOT, "self-hosted")
HELPER = "native_reloc_refuse_if_invalid"
APPLY = "apply_relocations_into"

FN_START = re.compile(r"^(?:pub )?fn ([A-Za-z_][A-Za-z0-9_]*)\s*\(")
WRITE_PRIM = re.compile(r"\b(write_file|io_write_[A-Za-z0-9_]+)\s*\(")
CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
GUARD_LET = re.compile(r"^\s*let\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*" + HELPER + r"\s*\(")


def strip_comment(line):
    # Sounio line comments; string literals in this backend never contain "//".
    i = line.find("//")
    return line if i < 0 else line[:i]


def load_functions():
    fns = []  # (file, name, start_line, [code lines])
    for dirpath, _dirs, files in os.walk(SRC_DIR):
        for f in sorted(files):
            if not f.endswith(".sio"):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, ROOT)
            with open(path, encoding="utf-8", errors="replace") as fh:
                lines = fh.read().split("\n")
            cur = None
            for no, raw in enumerate(lines, 1):
                if cur is None:
                    m = FN_START.match(raw)
                    if m:
                        cur = (rel, m.group(1), no, [])
                    else:
                        continue
                cur[3].append((no, strip_comment(raw)))
                if raw.startswith("}"):
                    fns.append(cur)
                    cur = None
    return fns


def calls_in(code):
    return [m.group(1) for m in CALL.finditer(code)]


def main():
    fns = load_functions()
    by_name = {}
    for fn in fns:
        by_name.setdefault(fn[1], []).append(fn)

    def body(fn):
        return fn[3][1:]  # skip the signature line

    # callee names per function, computed once
    callees = {}
    callers_of = {}
    for fn in fns:
        names = set(n for _l, c in body(fn) for n in calls_in(c)) - {fn[1]}
        callees[id(fn)] = names
        for n in names:
            callers_of.setdefault(n, []).append(fn)

    # writers (fixpoint over the reverse call graph): a write primitive, or a
    # call to a writer
    writers = set()
    for fn in fns:
        if any(WRITE_PRIM.search(c) for _n, c in body(fn)):
            writers.add(fn[1])
    pending = list(writers)
    while pending:
        w = pending.pop()
        for g in callers_of.get(w, []):
            if g[1] not in writers:
                writers.add(g[1])
                pending.append(g[1])

    def first_guard(lines, start_idx):
        """index of the first well-formed guard at or after start_idx, else None"""
        for i in range(start_idx, len(lines)):
            m = GUARD_LET.match(lines[i][1])
            if not m:
                continue
            var = m.group(1)
            for j in range(i + 1, len(lines)):
                nxt = lines[j][1].strip()
                if not nxt:
                    continue
                pat = r"^if\s+%s\s*!=\s*0\s*\{\s*return\s+%s\s*\}$" % (re.escape(var), re.escape(var))
                if re.match(pat, nxt):
                    return i
                break
        return None

    def writer_self_guarded(name):
        """every definition of `name` guards before its first write primitive / writer call"""
        for fn in by_name.get(name, []):
            lines = body(fn)
            first_write = None
            for i, (_no, c) in enumerate(lines):
                if WRITE_PRIM.search(c) or any(n in writers and n != name for n in calls_in(c)):
                    first_write = i
                    break
            if first_write is None:
                continue
            g = first_guard(lines, 0)
            if g is None or g >= first_write:
                return False
        return True

    relocating = {APPLY}
    failures = []
    report = []
    seen = set()
    work = []
    for fn in callers_of.get(APPLY, []):
        if fn[1] != APPLY:
            work.append(fn)

    while work:
        fn = work.pop(0)
        key = (fn[0], fn[1], fn[2])
        lines = body(fn)
        reloc_idx = None
        reloc_via = None
        for i, (_no, c) in enumerate(lines):
            hit = [n for n in calls_in(c) if n in relocating and n != fn[1]]
            if hit:
                reloc_idx, reloc_via = i, hit[0]
                break
        if reloc_idx is None or key in seen:
            continue
        seen.add(key)
        where = "%s:%d %s" % (fn[0], lines[reloc_idx][0], fn[1])
        reloc_line = lines[reloc_idx][1]
        guard_idx = first_guard(lines, reloc_idx + 1)
        reached = []
        for i in range(reloc_idx + 1, len(lines)):
            no, c = lines[i]
            if WRITE_PRIM.search(c):
                reached.append((i, no, WRITE_PRIM.search(c).group(1), True))
            for n in calls_in(c):
                if n in writers and n != fn[1]:
                    reached.append((i, no, n, False))
        if reached:
            for i, no, what, prim in reached:
                if guard_idx is not None and guard_idx < i:
                    report.append("PASS  route %s -[%s]-> %s (line %d): guarded in %s at line %d"
                                  % (fn[1], reloc_via, what, no, fn[1], lines[guard_idx][0]))
                elif not prim and writer_self_guarded(what):
                    report.append("PASS  route %s -[%s]-> %s (line %d): guarded inside %s"
                                  % (fn[1], reloc_via, what, no, what))
                else:
                    failures.append("FAIL  route %s -[%s]-> %s (%s:%d): reaches a write without %s"
                                    % (fn[1], reloc_via, what, fn[0], no, HELPER))
            continue
        # No write after relocating. Does the image escape to the caller?
        own_nc = re.search(r"\b%s\s*\(\s*nc\s*," % re.escape(reloc_via), reloc_line) is not None
        sig = fn[3][0][1]
        if own_nc and re.search(r"\bnc\s*:\s*&!", sig):
            relocating.add(fn[1])
            callers = callers_of.get(fn[1], [])
            if callers:
                report.append("INFO  %s relocates the caller's nc; following %d caller(s): %s"
                              % (where, len(callers), ", ".join(g[1] for g in callers)))
                work.extend(callers)
            else:
                report.append("INFO  %s relocates the caller's nc; no caller in self-hosted/ (no write reached)"
                              % where)
        else:
            report.append("INFO  %s relocates a local NativeCompiler and writes nothing after it (no write reached)"
                          % where)

    for line in report:
        print(line)
    for line in failures:
        print(line)
    if not any(l.startswith("PASS  route") for l in report):
        print("FAIL  no route from %s to a write was found -- the derivation is broken" % APPLY)
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
