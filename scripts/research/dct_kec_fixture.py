#!/usr/bin/env python3
"""KEC-alpha fixture builder (I/O plumbing ONLY — all math lives in Sounio).

Reads the Kruse et al. DCT df_wide CSV (post-exclusion, one wave), the canonical wordlist,
GloVe-6B-300d (primary semantic space) and NRC-VAD (control valence space). Builds two shared
k-NN word graphs (embedding + VAD), packs each subject's proximal/distal choice into a bitfield,
and emits a self-contained Sounio program dct_kec_wave{N}.sio that computes the KEC-alpha metric C
and the cross-subject statistics natively.

Convention (matches Kruse preprocessing vol1.Rmd): "this"=proximal, "that"=distal. We store bit=1
for "that"(distal), bit=0 for "this"(proximal). C uses (s_i-s_j)^2 so the sign/bit polarity is free.

Usage: python3 scripts/research/dct_kec_fixture.py --wave 1
"""
import argparse, csv, hashlib, json, math, sys, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DCT = ROOT / "data/external/dct_kruse"
EMB = ROOT / "data/external/embeddings"
OUT_SIO = ROOT / "examples/semantic_orc"
GLOVE_ZIP = EMB / "glove.6B.zip"
GLOVE_MEMBER = "glove.6B.300d.txt"
FASTTEXT_ZIP = EMB / "fasttext-wiki-news-300d-1M.vec.zip"
FASTTEXT_MEMBER = "wiki-news-300d-1M.vec"
NRC_VAD = EMB / "NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt"
KNN_K = 10
FIELD_STRIDE = None  # words of 64 bits, set after N known

# array caps in the emitted Sounio (must exceed real sizes)
MAXW = 512        # words
MAXE = 6144       # edges per graph
MAXS = 1024       # subjects
MAXFIELD = MAXS * 8  # subj * stride(<=8)


def load_wordlist():
    words = []
    with open(DCT / "wordlist.csv", newline="") as fh:
        r = csv.reader(fh)
        next(r)  # header "","word"
        for row in r:
            words.append(row[1].strip().lower())
    return words


def load_subjects(wave, words):
    """Return field bits (S,N) in {0,1}, phq_sum (S,), phq_class (S,), and missing report."""
    path = DCT / f"df_wide{wave}.csv"
    fields, phq_sum, phq_class = [], [], []
    n_missing_cells = 0
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames
        # map each wordlist word to its df column (exact match, lowercased compare)
        col_for = {}
        lc = {c.lower(): c for c in cols}
        for w in words:
            col_for[w] = lc.get(w)  # may be None
        missing_word_cols = [w for w in words if col_for[w] is None]
        for row in rd:
            f = np.zeros(len(words), dtype=np.int64)
            for wi, w in enumerate(words):
                c = col_for[w]
                v = (row.get(c) or "").strip().lower() if c else ""
                if v == "that":
                    f[wi] = 1
                elif v == "this":
                    f[wi] = 0
                else:
                    n_missing_cells += 1  # NA/blank -> default proximal(0), counted
            fields.append(f)
            phq_sum.append(float(row["PHQ9_sum"]))
            phq_class.append(int(float(row["PHQ9_class"])))
    return (np.array(fields), np.array(phq_sum), np.array(phq_class, dtype=np.int64),
            missing_word_cols, n_missing_cells)


def load_embedding(vocab, kind):
    """kind='glove' (glove.6B.300d.txt) or 'fasttext' (wiki-news-300d-1M.vec, has a count header)."""
    vecs = {}
    want = set(vocab)
    zp, member = (GLOVE_ZIP, GLOVE_MEMBER) if kind == "glove" else (FASTTEXT_ZIP, FASTTEXT_MEMBER)
    with zipfile.ZipFile(zp) as z, z.open(member) as fh:
        first = True
        for raw in io_lines(fh):
            if first and kind == "fasttext":
                first = False  # skip "N D" header line
                continue
            first = False
            sp = raw.rstrip(" ").split(" ")
            w = sp[0]
            if w in want:
                vecs[w] = np.asarray(sp[1:], dtype=np.float64)
                if len(vecs) == len(want):
                    break
    return vecs


def io_lines(fh):
    for b in fh:
        yield b.decode("utf-8", "ignore").rstrip("\n")


def load_vad(vocab):
    want = set(vocab)
    vad = {}
    with open(NRC_VAD) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4 and p[0] in want:
                vad[p[0]] = (float(p[1]), float(p[2]), float(p[3]))
    return vad


def knn_edges(words, vecmap, k, metric):
    """Symmetric k-NN edge list over the indices of `words` that have a vector.
    metric='cosine' -> weight=cosine sim (clipped >0); 'vad' -> weight=1/(1+euclid)."""
    idx = [i for i, w in enumerate(words) if w in vecmap]
    if len(idx) < k + 1:
        return [], idx
    M = np.stack([vecmap[words[i]] for i in idx])  # (m,d)
    if metric == "cosine":
        norm = np.linalg.norm(M, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        Mn = M / norm
        S = Mn @ Mn.T  # cosine similarity
        np.fill_diagonal(S, -np.inf)
        edge = {}
        for a in range(len(idx)):
            nn = np.argpartition(-S[a], k)[:k]
            for b in nn:
                w = float(S[a, b])
                if w <= 0:
                    continue
                i, j = idx[a], idx[b]
                key = (min(i, j), max(i, j))
                edge[key] = max(edge.get(key, 0.0), w)
        return sorted((i, j, w) for (i, j), w in edge.items()), idx
    else:  # vad euclidean
        # pairwise euclidean
        D = np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        edge = {}
        for a in range(len(idx)):
            nn = np.argpartition(D[a], k)[:k]
            for b in nn:
                d = float(D[a, b])
                w = 1.0 / (1.0 + d)
                i, j = idx[a], idx[b]
                key = (min(i, j), max(i, j))
                edge[key] = max(edge.get(key, 0.0), w)
        return sorted((i, j, w) for (i, j), w in edge.items()), idx


# ---------- Sounio emission helpers ----------
def fmt_f(v):
    t = f"{v:.10g}"
    if "e" not in t and "E" not in t and "." not in t and "inf" not in t and "nan" not in t:
        t += ".0"
    return t


def _signed64(v):
    v = int(v) & ((1 << 64) - 1)
    return v - (1 << 64) if v >= (1 << 63) else v


def emit_i64_loader(name, vals, cap):
    n = len(vals)
    lit = ", ".join(str(_signed64(v)) for v in vals) if n else "0"
    body = (f"fn {name}(out: &![i64; {cap}]) with Mut {{\n")
    if n:
        body += f"    let src: [i64; {n}] = [{lit}]\n"
        body += f"    var i: i64 = 0\n    while i < {n} {{ out[i] = src[i]; i = i + 1 }}\n"
    body += "}\n"
    return body


def emit_f64_loader(name, vals, cap):
    n = len(vals)
    lit = ", ".join(fmt_f(v) for v in vals) if n else "0.0"
    body = (f"fn {name}(out: &![f64; {cap}]) with Mut {{\n")
    if n:
        body += f"    let src: [f64; {n}] = [{lit}]\n"
        body += f"    var i: i64 = 0\n    while i < {n} {{ out[i] = src[i]; i = i + 1 }}\n"
    body += "}\n"
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, required=True, choices=[1, 2])
    ap.add_argument("--embedding", choices=["glove", "fasttext"], default="glove")
    args = ap.parse_args()
    wave = args.wave
    emb_kind = args.embedding

    words = load_wordlist()
    N = len(words)
    stride = (N + 63) // 64
    print(f"[wordlist] N={N} words, field stride={stride} i64 words/subject")

    fields, phq_sum, phq_class, missing_cols, missing_cells = load_subjects(wave, words)
    S = fields.shape[0]
    print(f"[wave{wave}] S={S} subjects | PHQ9_sum range {phq_sum.min():.0f}-{phq_sum.max():.0f} "
          f"mean {phq_sum.mean():.2f} | class1(>=10)={int(phq_class.sum())}")
    if missing_cols:
        print(f"[warn] {len(missing_cols)} wordlist words not found as df columns: {missing_cols[:8]}...")
    if missing_cells:
        print(f"[warn] {missing_cells} missing choice cells defaulted to proximal(0)")

    glove = load_embedding(words, emb_kind)
    vad = load_vad(words)
    print(f"[{emb_kind}] coverage {len(glove)}/{N} = {100*len(glove)/N:.1f}%")
    print(f"[vad]   coverage {len(vad)}/{N} = {100*len(vad)/N:.1f}%")

    emb_edges, emb_idx = knn_edges(words, glove, KNN_K, "cosine")
    vad_edges, vad_idx = knn_edges(words, vad, KNN_K, "vad")
    print(f"[graph] embedding edges={len(emb_edges)} (nodes covered {len(emb_idx)}) | "
          f"vad edges={len(vad_edges)} (nodes {len(vad_idx)})")

    # valence per word (NRC-VAD valence); NaN sentinel for missing -> Sounio skips via mask not needed,
    # we emit valence and a coverage count; valence_bias in Sounio uses only covered words via val!=999.
    valence = np.full(N, 999.0)
    for i, w in enumerate(words):
        if w in vad:
            valence[i] = vad[w][0]

    # pack fields to bits: bit=field[w], word = w>>6, pos = w&63 (pure python ints; bit 63 ok)
    packed = [0] * (S * stride)
    for s in range(S):
        for w in range(N):
            if fields[s, w]:
                packed[s * stride + (w >> 6)] |= (1 << (w & 63))

    ei = [e[0] for e in emb_edges]; ej = [e[1] for e in emb_edges]; ew = [e[2] for e in emb_edges]
    vi = [e[0] for e in vad_edges]; vj = [e[1] for e in vad_edges]; vw = [e[2] for e in vad_edges]

    # ---- write fixture data module ----
    data_path = OUT_SIO / f"dct_kec_wave{wave}_data.sio"
    with open(data_path, "w") as f:
        f.write(f"//! GENERATED by scripts/research/dct_kec_fixture.py — wave {wave}. DO NOT EDIT.\n")
        f.write(f"//! N_words={N} stride={stride} S={S} emb_edges={len(ei)} vad_edges={len(vi)}\n\n")
        f.write(f"fn fx_n_words() -> i64 {{ {N} }}\n")
        f.write(f"fn fx_stride() -> i64 {{ {stride} }}\n")
        f.write(f"fn fx_n_subj() -> i64 {{ {S} }}\n")
        f.write(f"fn fx_n_edges() -> i64 {{ {len(ei)} }}\n")
        f.write(f"fn fx_n_edges_vad() -> i64 {{ {len(vi)} }}\n\n")
        f.write(emit_i64_loader("load_ei", ei, MAXE))
        f.write(emit_i64_loader("load_ej", ej, MAXE))
        f.write(emit_f64_loader("load_ew", ew, MAXE))
        f.write(emit_i64_loader("load_vi", vi, MAXE))
        f.write(emit_i64_loader("load_vj", vj, MAXE))
        f.write(emit_f64_loader("load_vw", vw, MAXE))
        f.write(emit_f64_loader("load_valence", list(valence), MAXW))
        f.write(emit_i64_loader("load_field", list(packed), MAXFIELD))
        f.write(emit_f64_loader("load_phq", list(phq_sum), MAXS))
        f.write(emit_i64_loader("load_phqclass", list(phq_class), MAXS))

    # ---- manifest ----
    man = {
        "generator": "scripts/research/dct_kec_fixture.py",
        "wave": wave, "n_words": N, "stride": stride, "n_subjects": int(S),
        "embedding": emb_kind,
        "emb_edges": len(ei), "vad_edges": len(vi),
        "emb_coverage": len(glove), "vad_coverage": len(vad),
        "knn_k": KNN_K, "missing_word_cols": missing_cols, "missing_cells": int(missing_cells),
        "phq_sum_mean": float(phq_sum.mean()), "phq_class1": int(phq_class.sum()),
        "encoding": "bit=1 distal(that), bit=0 proximal(this); C uses (s_i-s_j)^2 so polarity free",
        "df_wide_sha256": hashlib.sha256((DCT / f"df_wide{wave}.csv").read_bytes()).hexdigest(),
    }
    man_path = OUT_SIO / f"dct_kec_wave{wave}_manifest.json"
    man_path.write_text(json.dumps(man, indent=2))
    print(f"[emit] {data_path.name} ({data_path.stat().st_size//1024} KB) + {man_path.name}")


if __name__ == "__main__":
    main()
