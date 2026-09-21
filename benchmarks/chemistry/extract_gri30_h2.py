#!/usr/bin/env python3
"""Extract the complete H/O sub-mechanism (or, with --full, the entire
mechanism) from Cantera's gri30.yaml (GRI-Mech 3.0).

Default: keeps every reaction whose reactant/product species all belong to
the hydrogen set {H2,H,O,O2,OH,H2O,HO2,H2O2,N2,AR}. --full: all species and
all reactions. Emits JSON with NASA7 coefficients and the reactions
(Arrhenius, three-body with efficiencies, falloff/Troe, duplicates).
"""
import json, re, sys

FULL = "--full" in sys.argv
HSET = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "N2", "AR"]

import os
HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if a != "--full"]
YAML = args[0] if args else os.environ.get("GRI30_YAML", "/tmp/gri30.yaml")
OUT = os.path.join(HERE, "gri30_full_mechanism.json" if FULL else "gri30_h2_mechanism.json")
text = open(YAML).read()

# ---------- species NASA7 ----------
species = {}
sp_section = text.split("\nspecies:\n")[1].split("\nreactions:")[0]
# blocks start with "- name: X" (prepend \n so the very first block is also split)
blocks = re.split(r"\n- name: ", "\n" + sp_section.lstrip())
for b in blocks[1:]:
    name = b.split("\n", 1)[0].strip()
    if not FULL and name not in HSET:
        continue
    ranges = re.search(r"temperature-ranges: \[([\d., ]+)\]", b).group(1)
    ranges = [float(x) for x in ranges.split(",")]
    data = re.findall(r"- \[([^\]]+)\]", b)
    coeffs = [[float(x) for x in row.replace("\n", " ").split(",")] for row in data[:2]]
    species[name] = {"ranges": ranges, "coeffs": coeffs}

# ---------- reactions ----------
rx_section = text.split("\nreactions:\n")[1]
rx_blocks = re.split(r"\n- equation: ", rx_section)
# rx_blocks[0] holds the FIRST reaction (no leading newline before it);
# strip its leading "- equation: " prefix instead of dropping it.
if rx_blocks[0].lstrip().startswith("- equation: "):
    rx_blocks[0] = rx_blocks[0].lstrip()[len("- equation: "):]
    rx_blocks = rx_blocks
else:
    rx_blocks = rx_blocks[1:]
reactions = []
for b in rx_blocks:
    eq = b.split("\n", 1)[0].strip()
    eq = re.sub(r"\s+#.*$", "", eq)
    rtype = "arrhenius"
    if re.search(r"^\s*type: three-body", b, re.M):
        rtype = "three-body"
    elif re.search(r"^\s*type: falloff", b, re.M):
        rtype = "falloff"
    duplicate = bool(re.search(r"^\s*duplicate: true", b, re.M))

    def rc(label):
        m = re.search(label + r": \{A: ([^,]+), b: ([^,]+), Ea: ([^}]+)\}", b)
        return [float(m.group(1)), float(m.group(2)), float(m.group(3))] if m else None

    fwd = rc("rate-constant") if rtype != "falloff" else rc("high-P-rate-constant")
    low = rc("low-P-rate-constant") if rtype == "falloff" else None
    troe = None
    mt = re.search(r"Troe: \{A: ([^,]+), T3: ([^,]+), T1: ([^,]+), T2: ([^}]+)\}", b)
    if mt:
        troe = [float(mt.group(1)), float(mt.group(2)), float(mt.group(3)), float(mt.group(4))]
    eff = {}
    me = re.search(r"efficiencies: \{([^}]+)\}", b, re.S)
    if me:
        for kv in me.group(1).replace("\n", " ").split(","):
            k, v = kv.split(":")
            k = k.strip()
            if k in HSET:
                eff[k] = float(v)

    # parse equation: "A + B + M <=> C + D" (reversible) or "=>" (irreversible)
    eq_clean = eq.replace("(+M)", "").replace("+ M", "").strip()
    if "<=>" in eq_clean:
        lhs, rhs = eq_clean.split("<=>")
        reversible = True
    else:
        lhs, rhs = eq_clean.split("=>")
        reversible = False
    def parse_side(side):
        out = {}
        for term in side.strip().split(" + "):
            term = term.strip()
            m = re.match(r"^(\d+(?:\.\d+)?)\s+(.+)$", term)
            if m:
                out[m.group(2).strip()] = out.get(m.group(2).strip(), 0) + float(m.group(1))
            else:
                out[term] = out.get(term, 0) + 1.0
        return out
    react = parse_side(lhs)
    prod = parse_side(rhs)
    allsp = set(react) | set(prod)
    if not FULL and not allsp <= set(HSET):
        continue
    reactions.append({
        "eq": eq, "type": rtype, "fwd": fwd, "low": low, "troe": troe,
        "eff": eff, "react": react, "prod": prod, "duplicate": duplicate, "reversible": reversible,
    })

print("species kept:", sorted(species.keys()))
print("reactions kept:", len(reactions))
for r in reactions:
    print(f"  [{r['type']:>10}] {r['eq']}" + ("  DUP" if r["duplicate"] else ""))

json.dump({"species": species, "reactions": reactions}, open(OUT, "w"), indent=1)
print("wrote", OUT)
