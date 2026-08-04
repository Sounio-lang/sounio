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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="downloads/mouse.owl")
    ap.add_argument("--human", default="downloads/human.owl")
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)

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
