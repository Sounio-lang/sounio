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
    named classes of the same ontology (anonymous restrictions, i.e.
    subClassOf nodes carrying rdf:nodeID, are skipped);
  * every <owl:disjointWith rdf:resource="..."/> pair between two declared
    named classes of the same ontology (symmetrised on output).

Outputs (relative to --out, default this directory):

  classes.tsv   id<TAB>ontology<TAB>iri<TAB>label
                (ids are per-ontology: mouse 0..E-1, human 0..H-1)
  tbox.txt      header comment lines (# ...) then one axiom per line:
                  sub  <mouse|human> <child_id> <parent_id>
                  disj <mouse|human> <a_id> <b_id>   (a_id < b_id, both dirs)

No size cap is applied here (the OAEI anatomy ontologies are small:
~2.7k + ~3.3k classes).  The cap for the Sounio driver is applied later,
in gen_sounio_data.py, and is stated in REAL_RESULTS.md.
"""

import argparse
import sys
import xml.etree.ElementTree as ET

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OWL = "{http://www.w3.org/2002/07/owl#}"


def extract(path):
    """Return (classes, sub, disj):
    classes: list of (iri, label) in document order (id = position)
    sub:     list of (child_id, parent_id)
    disj:    list of (a_id, b_id) with a_id < b_id, deduplicated/symmetric
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

    sub = set()
    disj = set()
    n_skipped_anon = 0
    n_skipped_external = 0
    for el in root.iter(OWL + "Class"):
        iri = el.get(RDF + "about")
        if iri is None or iri not in id_of:
            continue
        cid = id_of[iri]
        for ch in el:
            res = ch.get(RDF + "resource")
            if ch.tag == RDFS + "subClassOf":
                if res is None:
                    n_skipped_anon += 1  # owl:Restriction etc.
                elif res in id_of:
                    if id_of[res] != cid:
                        sub.add((cid, id_of[res]))
                else:
                    n_skipped_external += 1
            elif ch.tag == OWL + "disjointWith":
                if res is not None and res in id_of and id_of[res] != cid:
                    a, b = sorted((cid, id_of[res]))
                    disj.add((a, b))
                elif res is not None and res not in id_of:
                    n_skipped_external += 1

    sub = sorted(sub)
    disj = sorted(disj)
    stats = {
        "skipped_anonymous_subclassof": n_skipped_anon,
        "skipped_external_refs": n_skipped_external,
    }
    return classes, sub, disj, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse", default="downloads/mouse.owl")
    ap.add_argument("--human", default="downloads/human.owl")
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)

    m_classes, m_sub, m_disj, m_stats = extract(args.mouse)
    h_classes, h_sub, h_disj, h_stats = extract(args.human)

    with open(f"{args.out}/classes.tsv", "w") as f:
        f.write("id\tontology\tiri\tlabel\n")
        for i, (iri, label) in enumerate(m_classes):
            f.write(f"{i}\tmouse\t{iri}\t{label}\n")
        for i, (iri, label) in enumerate(h_classes):
            f.write(f"{i}\thuman\t{iri}\t{label}\n")

    with open(f"{args.out}/tbox.txt", "w") as f:
        f.write(f"# classes_mouse {len(m_classes)}\n")
        f.write(f"# classes_human {len(h_classes)}\n")
        f.write(f"# sub_mouse {len(m_sub)}\n")
        f.write(f"# sub_human {len(h_sub)}\n")
        f.write(f"# disj_mouse {len(m_disj)}\n")
        f.write(f"# disj_human {len(h_disj)}\n")
        f.write(f"# skipped_anonymous_subclassof_mouse "
                f"{m_stats['skipped_anonymous_subclassof']}\n")
        f.write(f"# skipped_anonymous_subclassof_human "
                f"{h_stats['skipped_anonymous_subclassof']}\n")
        f.write(f"# skipped_external_refs_mouse "
                f"{m_stats['skipped_external_refs']}\n")
        f.write(f"# skipped_external_refs_human "
                f"{h_stats['skipped_external_refs']}\n")
        for ont, axioms in (("mouse", m_sub), ("human", h_sub)):
            for c, p in axioms:
                f.write(f"sub {ont} {c} {p}\n")
        for ont, pairs in (("mouse", m_disj), ("human", h_disj)):
            for a, b in pairs:
                f.write(f"disj {ont} {a} {b}\n")

    print(f"mouse: {len(m_classes)} classes, {len(m_sub)} sub, "
          f"{len(m_disj)} disj "
          f"(skipped anon={m_stats['skipped_anonymous_subclassof']}, "
          f"ext={m_stats['skipped_external_refs']})")
    print(f"human: {len(h_classes)} classes, {len(h_sub)} sub, "
          f"{len(h_disj)} disj "
          f"(skipped anon={h_stats['skipped_anonymous_subclassof']}, "
          f"ext={h_stats['skipped_external_refs']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
