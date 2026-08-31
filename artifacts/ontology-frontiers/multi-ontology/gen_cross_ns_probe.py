#!/usr/bin/env python3
"""gen_cross_ns_probe.py — round 16a: measure what namespace-only drops.

For each OBO target, run extract_obo under policy=ns_only and
policy=open_fillers, mirror both TBoxes, and report the delta in H,
exsub, role_edges, atomic_edges, conf.  This is the multi-namespace
measurement the round-13/15 honesty notes flagged but did not quantify
on CL/UBERON/PATO/ChEBI.

Does NOT rewrite the self-validating packed drivers (those stay
ns_only receipts).  Science product is the delta table in
CROSS_NS_RESULTS.md (emitted here as CROSS_NS_PROBE.out + optional md).

Targets default to what this worktree has under downloads/:
  pato (small control), cl (the round-13 poster child for foreign fillers),
  and optionally chebi / uberon via --only.

Run from this directory:
  python3 gen_cross_ns_probe.py
  python3 gen_cross_ns_probe.py --only cl
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, "../real-data/scale")
sys.path.insert(0, "../real-data")
sys.path.insert(0, ".")

from gen_multi_data import extract_obo, run_mirror  # noqa: E402
from gen_chebi_data import run_mirror_auto, SPARSE_H_THRESHOLD  # noqa: E402

RO = "downloads/ro.owl"

# (name, ns, owl relative path)
DEFAULT_TARGETS = [
    ("pato", "PATO", "downloads/pato.owl"),
    ("cl", "CL", "downloads/cl.owl"),
    ("uberon", "UBERON", "downloads/uberon.owl"),
    ("chebi", "CHEBI", "downloads/chebi.owl"),
]


def ensure_download(rel: str) -> str:
    """Prefer local path; fall back to shared /workspace/sounio downloads."""
    if os.path.isfile(rel) or os.path.islink(rel):
        return rel
    base = os.path.basename(rel)
    alt = f"/workspace/sounio/artifacts/ontology-frontiers/multi-ontology/downloads/{base}"
    if os.path.isfile(alt):
        os.makedirs(os.path.dirname(rel) or ".", exist_ok=True)
        if not os.path.exists(rel):
            os.symlink(alt, rel)
            print(f"[probe] symlinked {rel} -> {alt}")
        return rel
    return rel


def one_policy(name, ns, owl, policy):
    t0 = time.time()
    stats, tbox, *_rest = extract_obo(name, ns, owl, RO, policy=policy)
    # sparse auto for large H
    m = run_mirror_auto(f"{name}/{policy}", tbox) if stats["h"] >= SPARSE_H_THRESHOLD \
        else run_mirror(f"{name}/{policy}", tbox)
    # run_mirror returns slightly different key sets; normalise
    out = {
        "policy": policy,
        "h": m.get("h", stats["h"]),
        "nr": m.get("nr", stats["nr"]),
        "nsub": m.get("nsub", tbox[2] and len(tbox[2])),
        "nex": m.get("nex", len(tbox[3])),
        "ndj": m.get("ndj", len(tbox[4])),
        "nrs": m.get("nrs", len(tbox[5])),
        "nch": m.get("nch", len(tbox[6])),
        "atomic_edges": m["atomic_edges"],
        "role_edges": m["role_edges_atom"],
        "conf": m["conf"],
        "rounds": m["rounds"],
        "no_rc": m.get("no_rc", m.get("role_edges_atom_no_rc", -1)),
        "no_rs": m.get("no_rs", m.get("role_edges_atom_no_rs", -1)),
        "n_primary": stats["n_primary"],
        "n_foreign": stats["n_foreign"],
        "foreign_parent": stats["foreign_parent"],
        "foreign_filler": stats["foreign_filler"],
        "foreign_disj": stats["foreign_disj"],
        "super_side": stats["super_side"],
        "equiv_restr": stats["equiv_restr"],
        "secs": time.time() - t0,
    }
    # run_mirror key names for ablations
    if out["no_rc"] < 0 and "role_edges_atom" in m:
        # gen_multi_data.run_mirror returns no_rc / no_rs
        out["no_rc"] = m.get("no_rc", -1)
        out["no_rs"] = m.get("no_rs", -1)
    return out


def fmt_row(name, a, b):
    def d(k):
        return b[k] - a[k]

    amp_a = a["role_edges"] / a["nex"] if a["nex"] else 0.0
    amp_b = b["role_edges"] / b["nex"] if b["nex"] else 0.0
    lines = [
        f"### {name}",
        "",
        f"| metric | ns_only | open_fillers | Δ |",
        f"|---|---:|---:|---:|",
        f"| H | {a['h']} | {b['h']} | {d('h')} |",
        f"| foreign_interned | {a['n_foreign']} | {b['n_foreign']} | {d('n_foreign')} |",
        f"| sub | {a['nsub']} | {b['nsub']} | {d('nsub')} |",
        f"| exsub | {a['nex']} | {b['nex']} | {d('nex')} |",
        f"| disj | {a['ndj']} | {b['ndj']} | {d('ndj')} |",
        f"| NR | {a['nr']} | {b['nr']} | {d('nr')} |",
        f"| atomic_edges | {a['atomic_edges']} | {b['atomic_edges']} | {d('atomic_edges')} |",
        f"| role_edges | {a['role_edges']} | {b['role_edges']} | {d('role_edges')} |",
        f"| conf | {a['conf']} | {b['conf']} | {d('conf')} |",
        f"| amp (edges/exsub) | {amp_a:.1f}× | {amp_b:.1f}× | {amp_b - amp_a:+.1f} |",
        f"| foreign_filler dropped by ns_only | {a['foreign_filler']} | (recovered in open) | — |",
        f"| foreign_parent dropped by ns_only | {a['foreign_parent']} | (recovered in open) | — |",
        f"| super_side / equiv_restr (probed) | {a['super_side']}/{a['equiv_restr']} | {b['super_side']}/{b['equiv_restr']} | — |",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="single target name (pato|cl|uberon|chebi)")
    ap.add_argument("--skip-chebi", action="store_true",
                    help="skip ChEBI (slow / large)")
    args = ap.parse_args()

    ensure_download(RO)
    rows_md = [
        "# Round 16a — multi-namespace (open_fillers) vs ns_only",
        "",
        "Policy `open_fillers`: primary-namespace classes plus any",
        "parent/filler/disj partner of a primary subject, closed under",
        "superclasses.  Mirrors are the same bitmask/sparse engines as",
        "rounds 13–15.  Packed drivers remain ns_only receipts.",
        "",
    ]
    summary = []

    for name, ns, owl in DEFAULT_TARGETS:
        if args.only and name != args.only:
            continue
        if args.skip_chebi and name == "chebi":
            print(f"[probe] skip {name}")
            continue
        ensure_download(owl)
        if not os.path.isfile(owl) and not os.path.islink(owl):
            print(f"[probe] SKIP {name}: missing {owl}")
            rows_md.append(f"### {name}\n\n*skipped: missing {owl}*\n")
            continue
        print(f"\n======== {name} ========")
        a = one_policy(name, ns, owl, "ns_only")
        b = one_policy(name, ns, owl, "open_fillers")
        block = fmt_row(name, a, b)
        print(block)
        rows_md.append(block)
        summary.append((name, a, b))

    rows_md.append("## Takeaway\n")
    if summary:
        for name, a, b in summary:
            d_ex = b["nex"] - a["nex"]
            d_re = b["role_edges"] - a["role_edges"]
            rows_md.append(
                f"- **{name}**: recovering {a['foreign_filler']} foreign fillers "
                f"adds {d_ex} exsub axioms and {d_re} role edges "
                f"(H {a['h']}→{b['h']})."
            )
        rows_md.append("")
        rows_md.append(
            "If Δ role_edges ≫ 0, namespace-only understates the OWL TBox; the "
            "round-13/15 numbers remain exact for the *extracted* TBox."
        )
    text = "\n".join(rows_md) + "\n"
    out_md = "CROSS_NS_RESULTS.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[probe] wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
