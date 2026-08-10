#!/usr/bin/env python3
"""lexical_match.py — simple lexical label matcher for the OAEI anatomy
pair (mouse.owl vs human.owl), python3 stdlib only.

Similarity formula (documented, deliberately simple):

  normalize(s)  = lowercase, every non-[a-z0-9] char -> space, collapse
                  whitespace, split into tokens
  STOPWORDS     = {of, the, and, or, a, an, to, in, for, by}
  tokens(s)     = normalize(s) minus STOPWORDS (if that empties the set,
                  the full normalized token set is used)
  conf(l1, l2)  = 1.0                       if normalize(l1) == normalize(l2)
                = |T1 ∩ T2| / |T1 ∪ T2|     otherwise (Jaccard on token sets)

Candidate generation uses a token inverted index over the human labels, so
only pairs sharing at least one non-stopword token are scored.  Per mouse
class we keep the --topk (default 3) best candidates with
conf >= --threshold (default 0.3); ties are broken by human id.

Outputs:
  mappings.tsv   mid<TAB>mouse_id<TAB>human_id<TAB>conf  (conf 4 decimals)
and prints candidate-level and top-1 precision/recall against the OAEI
reference alignment (downloads/reference.rdf, relation '=' only).
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET

STOPWORDS = {"of", "the", "and", "or", "a", "an", "to", "in", "for", "by"}
ALIGN_NS = "{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"


def normalize_tokens(label):
    toks = re.sub(r"[^a-z0-9]+", " ", label.lower()).split()
    content = [t for t in toks if t not in STOPWORDS]
    return toks, (content if content else toks)


def load_classes(path):
    mouse, human = [], []
    iri_of = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            cid, ont, iri, label = line.rstrip("\n").split("\t")
            rec = (int(cid), iri, label)
            iri_of[iri] = (ont, int(cid))
            (mouse if ont == "mouse" else human).append(rec)
    mouse.sort()
    human.sort()
    return mouse, human, iri_of


def load_reference(path, iri_of):
    """Return set of (mouse_id, human_id) with relation '='."""
    ref = set()
    if not path:
        return ref
    root = ET.parse(path).getroot()
    for cell in root.iter(ALIGN_NS + "Cell"):
        e1 = cell.find(ALIGN_NS + "entity1")
        e2 = cell.find(ALIGN_NS + "entity2")
        rel = cell.find(ALIGN_NS + "relation")
        if e1 is None or e2 is None or rel is None:
            continue
        if (rel.text or "").strip() != "=":
            continue
        r1 = e1.get(RDF + "resource")
        r2 = e2.get(RDF + "resource")
        if r1 in iri_of and r2 in iri_of:
            o1, i1 = iri_of[r1]
            o2, i2 = iri_of[r2]
            if o1 == "mouse" and o2 == "human":
                ref.add((i1, i2))
            elif o1 == "human" and o2 == "mouse":
                ref.add((i2, i1))
    return ref


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="classes.tsv")
    ap.add_argument("--reference", default="downloads/reference.rdf")
    ap.add_argument("--out", default="mappings.tsv")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args(argv)

    mouse, human, iri_of = load_classes(args.classes)

    # Token inverted index over human labels.
    h_tokens = []
    index = {}
    for hid, iri, label in human:
        _, toks = normalize_tokens(label)
        ts = set(toks)
        h_tokens.append(ts)
        for t in ts:
            index.setdefault(t, []).append(hid)

    candidates = []  # (mouse_id, human_id, conf)
    top1 = {}        # mouse_id -> (human_id, conf)
    for mid, iri, label in mouse:
        norm = " ".join(normalize_tokens(label)[0])
        _, toks = normalize_tokens(label)
        ts = set(toks)
        pool = set()
        for t in ts:
            pool.update(index.get(t, ()))
        scored = []
        for hid in pool:
            hnorm = " ".join(normalize_tokens(human[hid][2])[0])
            if norm == hnorm:
                conf = 1.0
            else:
                inter = len(ts & h_tokens[hid])
                union = len(ts | h_tokens[hid])
                conf = inter / union if union else 0.0
            if conf >= args.threshold:
                scored.append((hid, conf))
        scored.sort(key=lambda x: (-x[1], x[0]))
        for hid, conf in scored[: args.topk]:
            candidates.append((mid, hid, conf))
        if scored:
            top1[mid] = scored[0]

    candidates.sort(key=lambda x: (x[0], -x[2], x[1]))
    with open(args.out, "w") as f:
        f.write("mid\tmouse_id\thuman_id\tconf\n")
        for k, (mid, hid, conf) in enumerate(candidates):
            f.write(f"{k}\t{mid}\t{hid}\t{conf:.4f}\n")

    # Evaluation against the OAEI reference alignment.
    ref = load_reference(args.reference, iri_of) if args.reference else set()
    cand_set = {(m, h) for m, h, _ in candidates}
    top1_set = {(m, h) for m, (h, _) in top1.items()}

    def pr(found):
        if not ref:
            return (0, 0.0, 0.0)
        tp = len(found & ref)
        p = tp / len(found) if found else 0.0
        r = tp / len(ref)
        return (tp, p, r)

    tp_c, p_c, r_c = pr(cand_set)
    tp_1, p_1, r_1 = pr(top1_set)

    print(f"mouse classes: {len(mouse)}, human classes: {len(human)}")
    print(f"candidate mappings (conf>={args.threshold}, top-{args.topk}): "
          f"{len(candidates)}")
    print(f"mouse classes with >=1 candidate: {len(top1)}")
    print(f"reference mappings (=): {len(ref)}")
    print(f"candidates:  TP={tp_c} P={p_c:.4f} R={r_c:.4f}")
    print(f"top-1 only:  TP={tp_1} P={p_1:.4f} R={r_1:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
