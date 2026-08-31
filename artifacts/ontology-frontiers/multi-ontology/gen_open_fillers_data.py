#!/usr/bin/env python3
"""gen_open_fillers_data.py — emit the `_open` packed variants that
open_fillers_elplus_driver.sio consumes.

WHY THIS FILE EXISTS
--------------------
The driver's own header names this generator:

    // Data: pato_open_packed.txt / cl_open_packed.txt (gen_open_fillers_data.py

and until now no such file was in the tree.  `pato_open_packed.txt` was
committed without it, so that artifact could not be re-derived, and
`cl_open_packed.txt` was simply absent -- which made
`ontology_multi_ontology_gate.sh` SKIP the driver and exit 1.

WHAT `_open` MEANS, DERIVED RATHER THAN ASSUMED
-----------------------------------------------
There was no specification to read, so the transformation was recovered by
differencing the one committed pair, pato_packed.txt -> pato_open_packed.txt:

    header       H 1887 -> 1888,  NSUB 2227 -> 2228,  NR/NEX/NDJ/NRS/NCH equal
    s rows       all 2227 present with BOTH ids +1, plus exactly one new: `s 1 0`
    x rows       37/37 present with the two CLASS ids +1, role id unchanged
    d rows       67/67 present with both ids +1
    h, k rows    5/5 and 2/2 byte-identical -- role-only, never shifted

So `_open` prepends one class at index 0 -- the open filler -- shifts every
existing class id up by one, and states that the former top (now id 1) is
subsumed by it.  Roles are untouched.

The remaining four header fields (atomic_edges, role_edges_atom, no_rc,
no_rs) are MIRROR values: the expected results of the EL+ closure and its
two ablations.  They cannot be shifted, only recomputed, so this generator
runs the same `run_mirror` that produced the originals rather than
transcribing numbers.

HOW TO KNOW IT IS RIGHT
-----------------------
    python3 gen_open_fillers_data.py --verify pato

regenerates pato_open_packed.txt from pato_packed.txt and requires the
result to be BYTE-IDENTICAL to the committed file.  A transformation
recovered by differencing one example is a guess until it reproduces that
example from the other side; this is that check, and it is the reason to
trust the cl output, which has no committed counterpart to compare against.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, "../real-data/scale")
sys.path.insert(0, "../real-data")

from gen_multi_data import run_mirror, emit_packed  # noqa: E402


def load_packed(path):
    """Parse a packed file into the tbox tuple run_mirror expects."""
    with open(path) as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    hdr = [int(x) for x in lines[0].split()]
    if len(hdr) != 13:
        sys.exit(f"{path}: header has {len(hdr)} ints, expected 13")
    h, nr, nsub, nex, ndj, nrs, nch = hdr[:7]
    sub2, exsub2, disj2, rsub2, rcomp2 = [], [], [], [], []
    for l in lines[1:]:
        p = l.split()
        if   p[0] == "s": sub2.append((int(p[1]), int(p[2])))
        elif p[0] == "x": exsub2.append((int(p[1]), int(p[2]), int(p[3])))
        elif p[0] == "d": disj2.append((int(p[1]), int(p[2])))
        elif p[0] == "h": rsub2.append((int(p[1]), int(p[2])))
        elif p[0] == "k": rcomp2.append((int(p[1]), int(p[2]), int(p[3])))
        else: sys.exit(f"{path}: unknown row kind {p[0]!r}")
    # The header is self-validating; honour that here too rather than
    # trusting the row counts we just read.
    for got, want, what in ((len(sub2), nsub, "sub"), (len(exsub2), nex, "exsub"),
                            (len(disj2), ndj, "disj"), (len(rsub2), nrs, "rsub"),
                            (len(rcomp2), nch, "rcomp")):
        if got != want:
            sys.exit(f"{path}: {what} rows {got} disagree with header {want}")
    return h, nr, sub2, exsub2, disj2, rsub2, rcomp2


def open_transform(tbox):
    """Prepend the open filler at id 0 and shift every class id up by one."""
    h, nr, sub2, exsub2, disj2, rsub2, rcomp2 = tbox
    return (
        h + 1,
        nr,
        [(1, 0)] + [(c + 1, p + 1) for c, p in sub2],
        [(c + 1, r, fl + 1) for c, r, fl in exsub2],
        [(a + 1, b + 1) for a, b in disj2],
        list(rsub2),      # role-only, never shifted
        list(rcomp2),
    )


def build(name, out_path):
    tbox = open_transform(load_packed(f"{name}_packed.txt"))
    st = run_mirror(f"{name}_open", tbox)
    h, nr, sub2, exsub2, disj2, rsub2, rcomp2 = tbox
    st.update(sub2=sub2, exsub2=exsub2, disj2=disj2, rsub2=rsub2, rcomp2=rcomp2)
    emit_packed(out_path, st)


def main():
    args = sys.argv[1:]
    if args[:1] == ["--verify"]:
        name = args[1] if len(args) > 1 else "pato"
        ref = f"{name}_open_packed.txt"
        if not os.path.exists(ref):
            sys.exit(f"--verify needs a committed {ref} to compare against")
        tmp = f"{ref}.regen"
        build(name, tmp)
        a, b = open(ref, "rb").read(), open(tmp, "rb").read()
        os.remove(tmp)
        if a != b:
            sys.exit(f"VERIFY FAILED: regenerated {name} differs from {ref}")
        print(f"VERIFY OK: {ref} reproduced byte-for-byte from {name}_packed.txt")
        return
    for name in (args or ["pato", "cl"]):
        build(name, f"{name}_open_packed.txt")


if __name__ == "__main__":
    main()
