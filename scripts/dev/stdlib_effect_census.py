#!/usr/bin/env python3
"""Machine-generated census of effect names and whether anything CONSUMES them.

The type census asks whether the compiler KNOWS a name. For effects that is the
wrong question — every name in the closed list is known by construction. The
question that decides whether `with E` means anything is whether some stage
INTERROGATES it: the checker consulting it to accept or refuse, and the backend
carrying it past lowering.

Per effect it reports:
  sites      uses of `with ... E` in versioned .sio outside archive/ and bootstrap/
  check      references to the name in self-hosted/check/  (a possible consumer)
  decides    references inside a conditional in check/     (a consumer that BRANCHES)
  ir         references in self-hosted/ir/
  native     references in self-hosted/native/

An effect with sites and no `decides` is written by programmers and consulted by
nothing.

ON THE `decides` HEURISTIC, because the number is only as good as its method.
It counts a reference whose LINE also carries a conditional token (if / while /
match / == / != / && / ||). That is crude: it will miss a branch written across
two lines, and it will count a name that merely shares a line with an unrelated
comparison. Treat it as a signal, not as a semantic measurement.

What survives the crudeness is the SHAPE. `Panic` has 48,560 `with` sites and
3,150 references inside self-hosted/check/, and not one of those 3,150 lines
carries a conditional. A heuristic this loose does not produce a zero by accident
at that scale — if anything branched on Panic, some line would have been caught.

The first run also found an inversion worth keeping in view: Mut, Panic and Div
account for ~147,000 `with` sites and share FOUR decision points, while ZD has 157
sites and 25. Decision density runs opposite to use.
"""
import json, os, re, subprocess, sys, collections

ROOT = subprocess.run(["git","rev-parse","--show-toplevel"],capture_output=True,text=True).stdout.strip()
os.chdir(ROOT)

def git_files(*pats):
    out = subprocess.run(["git","ls-files",*pats],capture_output=True,text=True).stdout
    return [l for l in out.split("\n") if l]

def read(p):
    try:
        with open(p, encoding="utf-8", errors="replace") as f: return f.read()
    except OSError: return ""

def closed_list():
    """The compiler's own closed list, read from the gate artifact if present,
    else re-derived from the effect-id table in check/."""
    for cand in ("/tmp/effect_name_closed_list.json",
                 "artifacts/gates/effect_name_closed_list.json"):
        if os.path.exists(cand):
            try:
                d = json.load(open(cand))
                n = d.get("closed_names")
                if n: return sorted(n)
            except Exception: pass
    src = "".join(read(f) for f in git_files("self-hosted/check/*.sio"))
    return sorted(set(re.findall(r'eff_name_is\(\s*"([A-Za-z][A-Za-z0-9_]*)"', src)))

def main():
    jsonout = sys.argv[sys.argv.index("--json")+1] if "--json" in sys.argv else None
    names = closed_list()
    if not names:
        print("EFFECT_CENSUS FAIL: closed list came back empty; the derivation is broken",
              file=sys.stderr)
        return 1

    corpus = [(f, read(f)) for f in git_files("*.sio")
              if not f.startswith(("archive/","bootstrap/"))]
    check_src  = {f: read(f) for f in git_files("self-hosted/check/*.sio")}
    ir_src     = "".join(read(f) for f in git_files("self-hosted/ir/*.sio"))
    native_src = "".join(read(f) for f in git_files("self-hosted/native/*.sio"))

    rows = []
    for e in names:
        w = re.compile(r'\bwith\b[^\n{]*\b%s\b' % re.escape(e))
        sites = sum(len(w.findall(src)) for _, src in corpus)
        chk = 0; decides = 0
        for f, src in check_src.items():
            for m in re.finditer(r'\b%s\b' % re.escape(e), src):
                chk += 1
                line_start = src.rfind("\n", 0, m.start()) + 1
                line = src[line_start: src.find("\n", m.start())]
                if re.search(r'\b(if|while|match|&&|\|\||==|!=)\b|[=!]=', line):
                    decides += 1
        rows.append({"effect": e, "sites": sites, "check": chk, "decides": decides,
                     "ir": len(re.findall(r'\b%s\b' % re.escape(e), ir_src)),
                     "native": len(re.findall(r'\b%s\b' % re.escape(e), native_src))})

    if jsonout: json.dump(rows, open(jsonout,"w"), indent=1)

    print("EFFECT_CENSUS effects=%d" % len(rows))
    print()
    print("%-26s %7s %7s %8s %5s %7s" % ("EFFECT","SITES","check","decides","ir","native"))
    for r in sorted(rows, key=lambda r: -r["sites"]):
        print("%-26s %7d %7d %8d %5d %7d" % (
            r["effect"], r["sites"], r["check"], r["decides"], r["ir"], r["native"]))
    silent = [r["effect"] for r in rows if r["sites"] > 0 and r["decides"] == 0]
    print()
    print("written but never branched on (%d): %s" % (len(silent), " ".join(silent)))
    unused = [r["effect"] for r in rows if r["sites"] == 0]
    print("in the closed list, never written (%d): %s" % (len(unused), " ".join(unused)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
