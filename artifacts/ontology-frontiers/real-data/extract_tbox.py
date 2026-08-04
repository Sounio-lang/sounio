#!/usr/bin/env python3
"""extract_tbox.py — syntactic TBox extraction from the OAEI 2016 Anatomy
track OWL files (mouse.owl, human.owl).

This is a SYNTACTIC extraction, not a DL reasoning step: we parse the
RDF/XML with xml.etree.ElementTree (python3 stdlib only) and collect, per
ontology:

  * every declared named class  <owl:Class rdf:about="...">  with its
    first <rdfs:label> (labels are lowercased/whitespace-collapsed copies
    of the literal text);
  * every <rdfs:subClassOf rdf:resource="..."/> edge between two DECLARED
    named classes of the same ontology;
  * every anonymous <rdfs:subClassOf> carrying an inline
    <owl:Restriction> with exactly <owl:onProperty rdf:resource="P"/> +
    <owl:someValuesFrom rdf:resource="F"/> (existential restriction
    C ⊑ ∃P.F, the EL+ role axioms — round 9; previously these were
    SKIPPED).  Restrictions of any other shape (nested/anonymous
    fillers, allValuesFrom, cardinality, ...) are still skipped and
    counted;
  * every <owl:disjointWith rdf:resource="..."/> pair between two declared
    named classes of the same ontology (symmetrised on output);
  * every declared <owl:ObjectProperty rdf:about="..."> with its first
    <rdfs:label> (the role table; onProperty references to undeclared
    properties are appended to it);
  * every <rdfs:subPropertyOf rdf:resource="..."/> edge between two known
    properties (role hierarchy, EL+ roleSub);
  * every <owl:propertyChainAxiom rdf:parseType="Collection"> with exactly
    two <owl:ObjectProperty rdf:resource="..."/> members, attached to a
    declared property T, read as the chain r1 ∘ r2 ⊑ T (EL+ roleComp).
    Other chain shapes (nodeID lists, >2 members) are skipped/counted.

Outputs (relative to --out, default this directory):

  classes.tsv   id<TAB>ontology<TAB>iri<TAB>label
                (ids are per-ontology: mouse 0..E-1, human 0..H-1)
  roles.tsv     id<TAB>ontology<TAB>iri<TAB>label
                (ids are per-ontology, in declaration order)
  tbox.txt      header comment lines (# ...) then one axiom per line:
                  sub      <mouse|human> <child_id> <parent_id>
                  disj     <mouse|human> <a_id> <b_id>   (a_id < b_id)
                  exsub    <mouse|human> <child_id> <role_id> <filler_id>
                  roleSub  <mouse|human> <sub_role_id> <super_role_id>
                  roleComp <mouse|human> <r1> <r2> <r3>  (r1 ∘ r2 ⊑ r3)

No size cap is applied here (the OAEI anatomy ontologies are small:
~2.7k + ~3.3k classes).  The cap for the Sounio driver is applied later,
in gen_sounio_data.py, and is stated in REAL_RESULTS.md.

Data profile note (round 9): the Anatomy track uses exactly ONE role per
ontology (part_of), 1,637/1,662 existential restrictions (mouse/human),
and ZERO subPropertyOf / propertyChainAxiom axioms.  The roleSub/roleComp
extraction paths are therefore exercised only on the SNOMED-style
synthetic instance in scale/gen_elplus_data.py.

Round 11 (GO/RO role-rich mode, --go/--ro/--go-root): extracts an
ancestor-closed SLICE of GO (go-plus.owl) rooted at a chosen GO term,
plus the RO (ro.owl) role axioms over the roles the slice actually uses:

  * GO named classes, subClassOf edges, disjointWith (incl.
    AllDisjointClasses members), and existential restrictions
    C ⊑ ∃r.F with named filler F (any namespace; nested/anonymous
    fillers, allValuesFrom, intersections are skipped/counted).
    Superclass-side restrictions (∃r.F ⊑ C) were probed for and do NOT
    occur in go-plus (0 hits), so the profile is: multiple roles, role
    hierarchy + composition, no conjunctions.
  * RO owl:TransitiveProperty (element or rdf:type) is emitted as the
    chain r ∘ r ⊑ r; rdfs:subPropertyOf as roleSub; 2-member
    owl:propertyChainAxiom Collections (members may be any element with
    rdf:about — RO uses rdf:Description) as roleComp r1 ∘ r2 ⊑ T.
    Only axioms whose roles are all used by the kept slice restrictions
    are emitted.
  * Slice: descendants of --go-root (subClassOf cone), then ancestor
    closure over subClassOf + restriction fillers + disjoint partners.
    Aborts with a diagnostic if the slice exceeds --go-cap classes.
  * Role cap: roles are ranked by restriction usage; the least-used are
    dropped (with their restrictions) until NR <= --go-max-roles and the
    interned universe (H+1)*(NR+1) fits --go-max-u (the driver's dense
    feasibility bound).  Drops are reported, never silent.

Outputs with --go-out-prefix (default "go"): go_elplus_tbox.txt,
go_roles.tsv, go_classes.tsv in --out.
"""

import argparse
import sys
import xml.etree.ElementTree as ET

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OWL = "{http://www.w3.org/2002/07/owl#}"


def extract(path):
    """Return (classes, roles, sub, disj, exsub, rsub, rcomp, stats):
    classes: list of (iri, label) in document order (id = position)
    roles:   list of (iri, label) in declaration order (id = position);
             properties referenced by restrictions but never declared are
             appended (label "")
    sub:     list of (child_id, parent_id)
    disj:    list of (a_id, b_id) with a_id < b_id, deduplicated/symmetric
    exsub:   list of (child_id, role_id, filler_id)  (C ⊑ ∃role.filler)
    rsub:    list of (sub_role_id, super_role_id)
    rcomp:   list of (r1, r2, r3)  meaning r1 ∘ r2 ⊑ r3
    """
    tree = ET.parse(path)
    root = tree.getroot()

    id_of = {}
    classes = []
    for el in root.iter(OWL + "Class"):
        iri = el.get(RDF + "about")
        if iri is None:  # anonymous class expression — skip
            continue
        if iri in id_of:  # duplicate declaration block — merge
            continue
        label = ""
        for ch in el:
            if ch.tag == RDFS + "label" and ch.text:
                label = " ".join(ch.text.split())
                break
        id_of[iri] = len(classes)
        classes.append((iri, label))

    # ── Role table: declared ObjectProperties in document order ────────
    rid_of = {}
    roles = []

    def role_id(iri):
        """Intern a property iri (declared or referenced)."""
        if iri not in rid_of:
            rid_of[iri] = len(roles)
            roles.append((iri, ""))
        return rid_of[iri]

    rsub = set()
    rcomp = set()
    n_skipped_chain = 0
    n_skipped_rsub_ext = 0
    for el in root.iter(OWL + "ObjectProperty"):
        iri = el.get(RDF + "about")
        if iri is None:
            continue
        label = ""
        for ch in el:
            if ch.tag == RDFS + "label" and ch.text:
                label = " ".join(ch.text.split())
                break
        rid = role_id(iri)
        if label and not roles[rid][1]:
            roles[rid] = (iri, label)
        for ch in el:
            if ch.tag == RDFS + "subPropertyOf":
                res = ch.get(RDF + "resource")
                if res is not None:
                    sup = role_id(res)
                    if sup != rid:
                        rsub.add((rid, sup))
                else:
                    n_skipped_rsub_ext += 1
            elif ch.tag == OWL + "propertyChainAxiom":
                members = []
                if ch.get(RDF + "parseType") == "Collection":
                    for m in ch:
                        res = m.get(RDF + "resource")
                        if res is None:
                            members = None
                            break
                        members.append(res)
                if members is not None and len(members) == 2:
                    rcomp.add((role_id(members[0]), role_id(members[1]),
                               rid))
                else:
                    n_skipped_chain += 1

    sub = set()
    disj = set()
    exsub = set()
    n_skipped_anon = 0
    n_skipped_external = 0
    n_skipped_restr_shape = 0
    for el in root.iter(OWL + "Class"):
        iri = el.get(RDF + "about")
        if iri is None or iri not in id_of:
            continue
        cid = id_of[iri]
        for ch in el:
            res = ch.get(RDF + "resource")
            if ch.tag == RDFS + "subClassOf":
                if res is not None:
                    if res in id_of:
                        if id_of[res] != cid:
                            sub.add((cid, id_of[res]))
                    else:
                        n_skipped_external += 1
                    continue
                # anonymous subClassOf: look for an inline existential
                # restriction  C ⊑ ∃P.F  (round 9: no longer dropped)
                handled = False
                for sub_el in ch:
                    if sub_el.tag != OWL + "Restriction":
                        continue
                    prop = None
                    filler = None
                    n_fields = 0
                    for rf in sub_el:
                        n_fields += 1
                        if rf.tag == OWL + "onProperty":
                            prop = rf.get(RDF + "resource")
                        elif rf.tag == OWL + "someValuesFrom":
                            filler = rf.get(RDF + "resource")
                    if (n_fields == 2 and prop is not None
                            and filler is not None and filler in id_of):
                        exsub.add((cid, role_id(prop), id_of[filler]))
                        handled = True
                if not handled:
                    n_skipped_anon += 1
            elif ch.tag == OWL + "disjointWith":
                if res is not None and res in id_of and id_of[res] != cid:
                    a, b = sorted((cid, id_of[res]))
                    disj.add((a, b))
                elif res is not None and res not in id_of:
                    n_skipped_external += 1

    sub = sorted(sub)
    disj = sorted(disj)
    exsub = sorted(exsub)
    rsub = sorted(rsub)
    rcomp = sorted(rcomp)
    stats = {
        "skipped_anonymous_subclassof": n_skipped_anon,
        "skipped_external_refs": n_skipped_external,
        "skipped_chain_axioms": n_skipped_chain,
    }
    return classes, roles, sub, disj, exsub, rsub, rcomp, stats


def write_ontology(f, ont, roles, sub, disj, exsub, rsub, rcomp):
    for c, p in sub:
        f.write(f"sub {ont} {c} {p}\n")
    for a, b in disj:
        f.write(f"disj {ont} {a} {b}\n")
    for c, r, fl in exsub:
        f.write(f"exsub {ont} {c} {r} {fl}\n")
    for r, s in rsub:
        f.write(f"roleSub {ont} {r} {s}\n")
    for r1, r2, r3 in rcomp:
        f.write(f"roleComp {ont} {r1} {r2} {r3}\n")


# ── Round 11: GO/RO role-rich slice extraction ───────────────────────────

OBO = "http://purl.obolibrary.org/obo/"


def _restriction_fields(restr_el):
    """Return (prop_iri, filler_iri) for a simple someValuesFrom
    restriction with exactly onProperty + someValuesFrom, both named;
    (None, None) otherwise."""
    prop = None
    filler = None
    n_fields = 0
    for rf in restr_el:
        n_fields += 1
        if rf.tag == OWL + "onProperty":
            prop = rf.get(RDF + "resource")
        elif rf.tag == OWL + "someValuesFrom":
            filler = rf.get(RDF + "resource")
    if n_fields == 2 and prop is not None and filler is not None:
        return prop, filler
    return None, None


def parse_go(path):
    """Parse go-plus.owl.  Returns (order, labels, sub, exsub, disj,
    role_labels, stats):
      order:  class IRIs in document order
      labels: iri -> first rdfs:label
      sub:    (child_iri, parent_iri) named-class edges
      exsub:  (class_iri, prop_iri, filler_iri)  C ⊑ ∃r.F, F named
      disj:   set of (a_iri, b_iri) with a < b (string order)
      role_labels: prop iri -> label (declared ObjectProperties)
    """
    tree = ET.parse(path)
    root = tree.getroot()

    order = []
    labels = {}
    sub = []
    exsub = []
    disj = set()
    stats = {"skipped_restr_shape": 0, "skipped_anon_subclassof": 0,
             "skipped_dup_class": 0}
    for el in root.iter(OWL + "Class"):
        iri = el.get(RDF + "about")
        if iri is None:
            continue
        if iri in labels:
            stats["skipped_dup_class"] += 1
            continue
        label = ""
        for ch in el:
            if ch.tag == RDFS + "label" and ch.text:
                label = " ".join(ch.text.split())
                break
        labels[iri] = label
        order.append(iri)
    declared = set(order)
    for el in root.iter(OWL + "Class"):
        iri = el.get(RDF + "about")
        if iri is None:
            continue
        for ch in el:
            res = ch.get(RDF + "resource")
            if ch.tag == RDFS + "subClassOf":
                if res is not None:
                    if res != iri:
                        sub.append((iri, res))
                else:
                    handled = False
                    for se in ch:
                        if se.tag != OWL + "Restriction":
                            continue
                        prop, filler = _restriction_fields(se)
                        if prop is not None and filler in declared:
                            exsub.append((iri, prop, filler))
                            handled = True
                        else:
                            stats["skipped_restr_shape"] += 1
                    if not handled:
                        stats["skipped_anon_subclassof"] += 1
            elif ch.tag == OWL + "disjointWith":
                if res is not None and res != iri:
                    disj.add(tuple(sorted((iri, res))))
    for el in root.iter(OWL + "AllDisjointClasses"):
        for ch in el:
            if ch.tag == OWL + "members" and \
                    ch.get(RDF + "parseType") == "Collection":
                members = [m.get(RDF + "about") or m.get(RDF + "resource")
                           for m in ch]
                members = [m for m in members if m]
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        if members[i] != members[j]:
                            disj.add(tuple(sorted((members[i],
                                                   members[j]))))
    role_labels = {}
    for tag in (OWL + "ObjectProperty", OWL + "TransitiveProperty"):
        for el in root.iter(tag):
            iri = el.get(RDF + "about")
            if iri is None or iri in role_labels:
                continue
            label = ""
            for ch in el:
                if ch.tag == RDFS + "label" and ch.text:
                    label = " ".join(ch.text.split())
                    break
            role_labels[iri] = label
    # dedup preserving order
    sub = sorted(set(sub))
    exsub = sorted(set(exsub))
    return order, labels, sub, exsub, sorted(disj), role_labels, stats


def parse_ro(path):
    """Parse ro.owl role axioms.  Returns (rsub, chains, role_labels,
    stats):  rsub: (sub_iri, super_iri); chains: (r1_iri, r2_iri, t_iri)
    meaning r1 ∘ r2 ⊑ t (transitive properties included as r ∘ r ⊑ r)."""
    tree = ET.parse(path)
    root = tree.getroot()
    rsub = set()
    chains = set()
    role_labels = {}
    stats = {"skipped_chain_shape": 0, "skipped_rsub_anon": 0}
    for tag in (OWL + "ObjectProperty", OWL + "TransitiveProperty"):
        for el in root.iter(tag):
            iri = el.get(RDF + "about")
            if iri is None:
                continue
            transitive = (tag == OWL + "TransitiveProperty")
            label = ""
            for ch in el:
                if ch.tag == RDFS + "label" and ch.text:
                    label = " ".join(ch.text.split())
                    break
            if iri not in role_labels or (label and not role_labels[iri]):
                role_labels[iri] = label
            for ch in el:
                res = ch.get(RDF + "resource")
                if ch.tag == RDF + "type" and res is not None and \
                        res.endswith("#TransitiveProperty"):
                    transitive = True
                elif ch.tag == RDFS + "subPropertyOf":
                    if res is not None:
                        if res != iri:
                            rsub.add((iri, res))
                    else:
                        stats["skipped_rsub_anon"] += 1
                elif ch.tag == OWL + "propertyChainAxiom":
                    members = None
                    if ch.get(RDF + "parseType") == "Collection":
                        members = []
                        for m in ch:
                            ref = m.get(RDF + "about") or \
                                m.get(RDF + "resource")
                            if ref is None:
                                members = None
                                break
                            members.append(ref)
                    if members is not None and len(members) == 2:
                        chains.add((members[0], members[1], iri))
                    else:
                        stats["skipped_chain_shape"] += 1
            if transitive:
                chains.add((iri, iri, iri))
    return sorted(rsub), sorted(chains), role_labels, stats


def slice_go(order, labels, sub, exsub, disj, root_iri, cap,
             go_only=True):
    """Ancestor-closed slice under root_iri.  Returns (slice_iris, sub2,
    exsub2, disj2) with IRIs (not yet renumbered).  Aborts with a
    diagnostic if the cap is exceeded.

    go_only (default): only GO-namespace classes (/GO_) are pulled into
    the slice — external fillers (CHEBI/CL/UBERON/...) and non-GO
    parents/partners are left out, which keeps the ancestor closure
    bounded (measured: 262 vs 2263 classes for GO:0006915).  Restrictions
    whose filler falls outside are dropped later, at renumbering."""
    root_full = root_iri
    if not root_full.startswith("http"):
        root_full = OBO + root_full.replace(":", "_")
    children = {}
    for c, p in sub:
        children.setdefault(p, []).append(c)
    desc = set()
    stack = [root_full]
    while stack:
        c = stack.pop()
        if c in desc:
            continue
        desc.add(c)
        stack.extend(children.get(c, ()))
    if root_full not in desc or len(desc) <= 1:
        sys.exit(f"GO SLICE FAILED: root {root_iri} has no descendants "
                 f"(unknown id?)")
    print(f"go slice: {len(desc)} descendants of {root_iri} (incl. root)")
    if len(desc) > cap:
        sys.exit(f"GO SLICE FAILED: {len(desc)} descendants > cap {cap}; "
                 f"choose a smaller root")

    def admissible(iri):
        return not go_only or "/GO_" in iri

    inslice = set(desc)
    changed = True
    while changed:
        changed = False
        for c, p in sub:
            if c in inslice and p not in inslice and admissible(p):
                inslice.add(p)
                changed = True
        for c, _r, f in exsub:
            if c in inslice and f not in inslice and admissible(f):
                inslice.add(f)
                changed = True
        for a, b in disj:
            if a in inslice and b not in inslice and admissible(b):
                inslice.add(b)
                changed = True
            elif b in inslice and a not in inslice and admissible(a):
                inslice.add(a)
                changed = True
    print(f"go slice: {len(inslice)} classes after ancestor closure"
          f"{' (GO-only)' if go_only else ''}")
    if len(inslice) > cap:
        sys.exit(f"GO SLICE FAILED: {len(inslice)} classes after ancestor "
                 f"closure > cap {cap}; choose a smaller root")

    sub2 = [(c, p) for c, p in sub if c in inslice and p in inslice]
    exsub2 = [(c, r, f) for c, r, f in exsub
              if c in inslice and f in inslice]
    disj2 = [(a, b) for a, b in disj if a in inslice and b in inslice]
    slice_iris = [iri for iri in order if iri in inslice]
    return slice_iris, sub2, exsub2, disj2


def main_go(args):
    print(f"parsing {args.go} ...")
    order, labels, sub, exsub, disj, go_role_labels, gstats = \
        parse_go(args.go)
    print(f"go: {len(order)} declared classes, {len(sub)} sub, "
          f"{len(exsub)} exsub, {len(disj)} disj "
          f"(skipped restr_shape={gstats['skipped_restr_shape']}, "
          f"anon_subclassof={gstats['skipped_anon_subclassof']})")

    slice_iris, sub2, exsub2, disj2 = slice_go(
        order, labels, sub, exsub, disj, args.go_root, args.go_cap)
    id_of = {iri: i for i, iri in enumerate(slice_iris)}
    H = len(slice_iris)

    sub3 = sorted({(id_of[c], id_of[p]) for c, p in sub2})
    disj3 = sorted({(id_of[a], id_of[b]) for a, b in disj2})

    print(f"parsing {args.ro} ...")
    rsub, chains, ro_role_labels, rstats = parse_ro(args.ro)
    print(f"ro: {len(rsub)} subPropertyOf, {len(chains)} chains (incl. "
          f"transitive), skipped chain_shape="
          f"{rstats['skipped_chain_shape']}")

    # role ranking by restriction usage; drop least-used until caps fit.
    # Then RO-CLOSE the kept set: superproperties of kept roles and
    # composition targets of chains whose members are kept are ADDED
    # (roles that only ever receive derived edges — this is what makes
    # the roleSub/roleComp rules fire on real data), caps permitting.
    use = {}
    for _c, r, _f in exsub2:
        use[r] = use.get(r, 0) + 1

    def fits(n):
        return n <= args.go_max_roles and (H + 1) * (n + 1) <= args.go_max_u

    keep = sorted(use, key=lambda r: (-use[r], r))
    dropped = []
    while keep and not fits(len(keep)):
        dropped.append(keep.pop())
    added_closure = []
    changed = True
    while changed:
        changed = False
        for r, s in rsub:
            if r in keep and s not in keep and fits(len(keep) + 1):
                keep.append(s)
                added_closure.append(s)
                changed = True
        for r1, r2, t in chains:
            if (r1 in keep and r2 in keep and t not in keep
                    and fits(len(keep) + 1)):
                keep.append(t)
                added_closure.append(t)
                changed = True
    if dropped:
        print(f"go slice: dropped {len(dropped)} least-used roles to fit "
              f"caps: {[(r.split('/')[-1], use[r]) for r in dropped]}")
    if added_closure:
        print(f"go slice: RO-closure added roles (derived-edge targets):"
              f" {[r.split('/')[-1] for r in added_closure]}")
    rid_of = {r: i for i, r in enumerate(sorted(keep))}
    NR = len(rid_of)
    if NR == 0:
        sys.exit("GO SLICE FAILED: slice uses no roles; choose a "
                 "role-rich root")

    exsub3 = sorted({(id_of[c], rid_of[r], id_of[f])
                     for c, r, f in exsub2 if r in rid_of})
    rsub3 = sorted({(rid_of[r], rid_of[s]) for r, s in rsub
                    if r in rid_of and s in rid_of})
    chains3 = sorted({(rid_of[r1], rid_of[r2], rid_of[t])
                      for r1, r2, t in chains
                      if r1 in rid_of and r2 in rid_of and t in rid_of})
    print(f"go slice: H={H} NR={NR} sub={len(sub3)} exsub={len(exsub3)} "
          f"disj={len(disj3)} roleSub={len(rsub3)} roleComp={len(chains3)}"
          f" U={(H + 1) * (NR + 1)}")
    print(f"roles kept: {[(rid_of[r], r.split('/')[-1], use.get(r, 0))
                          for r in sorted(rid_of)]}")
    if len(rsub3) == 0 and len(chains3) == 0:
        print("WARNING: no RO role axioms apply to the kept roles; "
              "the slice is not role-rich")

    pfx = args.go_out_prefix
    with open(f"{args.out}/{pfx}_classes.tsv", "w") as f:
        f.write("id\tiri\tlabel\n")
        for i, iri in enumerate(slice_iris):
            f.write(f"{i}\t{iri}\t{labels.get(iri, '')}\n")
    with open(f"{args.out}/{pfx}_roles.tsv", "w") as f:
        f.write("id\tiri\tlabel\n")
        for r, i in sorted(rid_of.items(), key=lambda kv: kv[1]):
            f.write(f"{i}\t{r}\t"
                    f"{ro_role_labels.get(r) or go_role_labels.get(r, '')}"
                    f"\n")
    with open(f"{args.out}/{pfx}_elplus_tbox.txt", "w") as f:
        f.write(f"# classes_go {H}\n")
        f.write(f"# roles_go {NR}\n")
        f.write(f"# sub_go {len(sub3)}\n")
        f.write(f"# exsub_go {len(exsub3)}\n")
        f.write(f"# disj_go {len(disj3)}\n")
        f.write(f"# rolesub_go {len(rsub3)}\n")
        f.write(f"# rolecomp_go {len(chains3)}\n")
        f.write(f"# root {args.go_root}\n")
        f.write(f"# dropped_roles {len(dropped)}\n")
        for c, p in sub3:
            f.write(f"sub go {c} {p}\n")
        for c, r, fl in exsub3:
            f.write(f"exsub go {c} {r} {fl}\n")
        for a, b in disj3:
            f.write(f"disj go {a} {b}\n")
        for r, s in rsub3:
            f.write(f"roleSub go {r} {s}\n")
        for r1, r2, t in chains3:
            f.write(f"roleComp go {r1} {r2} {t}\n")
    print(f"wrote {pfx}_elplus_tbox.txt, {pfx}_roles.tsv, "
          f"{pfx}_classes.tsv to {args.out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="downloads/mouse.owl")
    ap.add_argument("--human", default="downloads/human.owl")
    ap.add_argument("--out", default=".")
    ap.add_argument("--go", default=None,
                    help="go-plus.owl path; enables the GO/RO role-rich "
                         "slice mode (round 11)")
    ap.add_argument("--ro", default="downloads/ro.owl")
    ap.add_argument("--go-root", default=None,
                    help="root GO id for the slice, e.g. GO:0006915")
    ap.add_argument("--go-cap", type=int, default=192,
                    help="maximum classes in the ancestor-closed slice")
    ap.add_argument("--go-max-roles", type=int, default=16)
    ap.add_argument("--go-max-u", type=int, default=2048,
                    help="maximum interned universe (H+1)*(NR+1)")
    ap.add_argument("--go-out-prefix", default="go")
    args = ap.parse_args(argv)

    if args.go is not None:
        if args.go_root is None:
            ap.error("--go requires --go-root (e.g. GO:0006915)")
        return main_go(args)

    (m_classes, m_roles, m_sub, m_disj, m_exsub, m_rsub, m_rcomp,
     m_stats) = extract(args.mouse)
    (h_classes, h_roles, h_sub, h_disj, h_exsub, h_rsub, h_rcomp,
     h_stats) = extract(args.human)

    with open(f"{args.out}/classes.tsv", "w") as f:
        f.write("id\tontology\tiri\tlabel\n")
        for i, (iri, label) in enumerate(m_classes):
            f.write(f"{i}\tmouse\t{iri}\t{label}\n")
        for i, (iri, label) in enumerate(h_classes):
            f.write(f"{i}\thuman\t{iri}\t{label}\n")

    with open(f"{args.out}/roles.tsv", "w") as f:
        f.write("id\tontology\tiri\tlabel\n")
        for i, (iri, label) in enumerate(m_roles):
            f.write(f"{i}\tmouse\t{iri}\t{label}\n")
        for i, (iri, label) in enumerate(h_roles):
            f.write(f"{i}\thuman\t{iri}\t{label}\n")

    with open(f"{args.out}/tbox.txt", "w") as f:
        f.write(f"# classes_mouse {len(m_classes)}\n")
        f.write(f"# classes_human {len(h_classes)}\n")
        f.write(f"# roles_mouse {len(m_roles)}\n")
        f.write(f"# roles_human {len(h_roles)}\n")
        f.write(f"# sub_mouse {len(m_sub)}\n")
        f.write(f"# sub_human {len(h_sub)}\n")
        f.write(f"# disj_mouse {len(m_disj)}\n")
        f.write(f"# disj_human {len(h_disj)}\n")
        f.write(f"# exsub_mouse {len(m_exsub)}\n")
        f.write(f"# exsub_human {len(h_exsub)}\n")
        f.write(f"# rolesub_mouse {len(m_rsub)}\n")
        f.write(f"# rolesub_human {len(h_rsub)}\n")
        f.write(f"# rolecomp_mouse {len(m_rcomp)}\n")
        f.write(f"# rolecomp_human {len(h_rcomp)}\n")
        f.write(f"# skipped_anonymous_subclassof_mouse "
                f"{m_stats['skipped_anonymous_subclassof']}\n")
        f.write(f"# skipped_anonymous_subclassof_human "
                f"{h_stats['skipped_anonymous_subclassof']}\n")
        f.write(f"# skipped_external_refs_mouse "
                f"{m_stats['skipped_external_refs']}\n")
        f.write(f"# skipped_external_refs_human "
                f"{h_stats['skipped_external_refs']}\n")
        f.write(f"# skipped_chain_axioms_mouse "
                f"{m_stats['skipped_chain_axioms']}\n")
        f.write(f"# skipped_chain_axioms_human "
                f"{h_stats['skipped_chain_axioms']}\n")
        write_ontology(f, "mouse", m_roles, m_sub, m_disj, m_exsub,
                       m_rsub, m_rcomp)
        write_ontology(f, "human", h_roles, h_sub, h_disj, h_exsub,
                       h_rsub, h_rcomp)

    for ont, classes, roles, sub, disj, exsub, rsub, rcomp, stats in (
            ("mouse", m_classes, m_roles, m_sub, m_disj, m_exsub, m_rsub,
             m_rcomp, m_stats),
            ("human", h_classes, h_roles, h_sub, h_disj, h_exsub, h_rsub,
             h_rcomp, h_stats)):
        print(f"{ont}: {len(classes)} classes, {len(roles)} roles, "
              f"{len(sub)} sub, {len(disj)} disj, {len(exsub)} exsub, "
              f"{len(rsub)} roleSub, {len(rcomp)} roleComp "
              f"(skipped anon={stats['skipped_anonymous_subclassof']}, "
              f"ext={stats['skipped_external_refs']}, "
              f"chains={stats['skipped_chain_axioms']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
