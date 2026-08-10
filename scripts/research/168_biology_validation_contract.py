#!/usr/bin/env python3
"""
168 biology preprint — biological validation contract.

Companion to:
  docs/papers/main/168-biology-preprint.typ

Re-derives every quantitative claim of the preprint from first principles
and audits the two biological mappings against real reference data:

  C1  343 = 133 + 42 + 168 partition of ordered CYP triples (brute force).
  C2  Octonion basis-associator census: 168 nonzero, norm in {0, 2}.
  C3  Fano plane facts: 7 lines, each point pair on exactly one line,
      each point on 3 lines, any two lines intersect (hence only triples,
      not pairs, can discriminate — supports the DDI test design).
  C4  CYP450 locus audit: cytogenetic bands of the seven FDA isoforms
      against the NCBI Gene reference values; the CYP2C trio (2C8/2C9/2C19)
      shares 10q23.33 and maps to a Fano line in the preprint's bijection.
  C5  Gauge analysis of the bijection: 1008 bijections place the CYP2C
      cluster on a Fano line, forming exactly 6 equivalence classes under
      the 168-element automorphism group PSL(2,7); exactly ONE line (the
      CYP2C line) is common to all six classes, so the other six "lines"
      of preprint Table 2 are representative-dependent.  Adding the
      "big three" constraint (CYP2D6–CYP3A4–CYP2C9 collinear) leaves
      336 bijections in 2 equivalence classes.
  C6  Genetic-code Hamming/hydrophobicity table: pair counts
      (186, 465, 620, 465, 186, 31), strictly monotonic class means
      (2.03, 3.02, 3.53, 3.78, 3.92, 4.00 under the documented
      stop-codon convention Delta-H measured with stop assigned 0),
      Pearson r = 0.199; stop-excluded variant also strictly monotonic
      with r = 0.218; Spearman rho = 0.208.
  C7  Permutation test for the r = 0.199 correlation (hydrophobicity
      values permuted among amino-acid categories, seed 168, 10 000
      permutations): p = 0.0147 < 0.05.
  C8  Encoding audit (LIMITATION clause): the purine/pyrimidine x
      weak/strong H-bond encoding ranks 3rd of the 6 encodings with
      A = (0,0), and plain nucleotide Hamming distance yields a HIGHER
      correlation (r = 0.261).  The Z_2^6 embedding is therefore NOT
      optimal for hydrophobicity; the preprint reports this.
  C9  Mutation robustness: 372 single-bit mutations between nonzero
      codons; 98 synonymous (26.3%), 208 class-preserving (55.9%)
      against a 30.9% chance baseline; binomial p < 1e-10.
  C10 Fano-line class coherence in PG(5,2) (NEGATIVE claim): 651 lines,
      64 same-class (9.8%) vs random baseline 11.5%; binomial
      p = 0.10 (non-significant) — confirms that codon organisation
      follows the metric, not the subplane structure.

Pass criteria: all verification clauses (C1..C7, C9) must reproduce the
published values exactly (within stated rounding); audit clauses (C5, C8,
C10) must reproduce the documented facts.  Prints

  BIO168_VALIDATION_VERDICT C_GREEN

iff every clause passes.

Pure Python stdlib, deterministic (fixed seeds), no network access.
"""

import itertools
import math
import random
from collections import Counter

RESULTS = []


def clause(name, ok, detail=''):
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")


# ============================================================================
# Shared constants
# ============================================================================

FANO_LINES = [(1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7),
              (5, 6, 1), (6, 7, 2), (7, 1, 3)]
FANO_LINE_SETS = [frozenset(l) for l in FANO_LINES]

# Preprint Table 1: basis index -> (isoform, chromosome band stated in paper)
PAPER_MAPPING = {
    1: ('CYP1A2', '15q24'),
    2: ('CYP2C9', '10q23'),
    3: ('CYP2C8', '10q23'),
    4: ('CYP2B6', '19q13'),
    5: ('CYP2C19', '10q23'),
    6: ('CYP2D6', '22q13'),
    7: ('CYP3A4', '7q22'),
}

# Reference cytogenetic bands (NCBI Gene, GRCh38 annotations).
REFERENCE_LOCI = {
    'CYP1A2': '15q24.1',
    'CYP2B6': '19q13.2',
    'CYP2C8': '10q23.33',
    'CYP2C9': '10q23.33',
    'CYP2C19': '10q23.33',
    'CYP2D6': '22q13.2',
    'CYP3A4': '7q22.1',
}

BASES = ['A', 'G', 'U', 'C']
BIOCHEM_ENC = {'A': (0, 0), 'G': (0, 1), 'U': (1, 0), 'C': (1, 1)}

CODON_TABLE = {}
for _c, _a in [
    ("UUU", "F"), ("UUC", "F"), ("UUA", "L"), ("UUG", "L"),
    ("CUU", "L"), ("CUC", "L"), ("CUA", "L"), ("CUG", "L"),
    ("AUU", "I"), ("AUC", "I"), ("AUA", "I"), ("AUG", "M"),
    ("GUU", "V"), ("GUC", "V"), ("GUA", "V"), ("GUG", "V"),
    ("UCU", "S"), ("UCC", "S"), ("UCA", "S"), ("UCG", "S"),
    ("CCU", "P"), ("CCC", "P"), ("CCA", "P"), ("CCG", "P"),
    ("ACU", "T"), ("ACC", "T"), ("ACA", "T"), ("ACG", "T"),
    ("GCU", "A"), ("GCC", "A"), ("GCA", "A"), ("GCG", "A"),
    ("UAU", "Y"), ("UAC", "Y"), ("UAA", "*"), ("UAG", "*"),
    ("CAU", "H"), ("CAC", "H"), ("CAA", "Q"), ("CAG", "Q"),
    ("AAU", "N"), ("AAC", "N"), ("AAA", "K"), ("AAG", "K"),
    ("GAU", "D"), ("GAC", "D"), ("GAA", "E"), ("GAG", "E"),
    ("UGU", "C"), ("UGC", "C"), ("UGA", "*"), ("UGG", "W"),
    ("CGU", "R"), ("CGC", "R"), ("CGA", "R"), ("CGG", "R"),
    ("AGU", "S"), ("AGC", "S"), ("AGA", "R"), ("AGG", "R"),
    ("GGU", "G"), ("GGC", "G"), ("GGA", "G"), ("GGG", "G"),
]:
    CODON_TABLE[_c] = _a

KD = {'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
      'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
      'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9,
      'R': -4.5}

CHEM_CLASS = {}
for _a in "AVLIMFWPG":
    CHEM_CLASS[_a] = 'nonpolar'
for _a in "STCYNQ":
    CHEM_CLASS[_a] = 'polar'
for _a in "KRH":
    CHEM_CLASS[_a] = 'positive'
for _a in "DE":
    CHEM_CLASS[_a] = 'negative'
CHEM_CLASS['*'] = 'stop'

CODONS = [''.join(p) for p in itertools.product(BASES, repeat=3)]
ZERO6 = (0, 0, 0, 0, 0, 0)


def codon_vec(codon, enc=BIOCHEM_ENC):
    return tuple(b for base in codon for b in enc[base])


def hamming(v1, v2):
    return sum(a != b for a, b in zip(v1, v2))


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for m in range(i, j + 1):
                r[order[m]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def binom_cdf_le(n, k, p):
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def binom_sf_ge(n, k, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


# ============================================================================
# C1: 343 = 133 + 42 + 168
# ============================================================================

def c1_partition():
    triv = assoc = nonassoc = 0
    for a, b, c in itertools.product(range(1, 8), repeat=3):
        if a == b or b == c or a == c:
            triv += 1
        elif frozenset((a, b, c)) in FANO_LINE_SETS:
            assoc += 1
        else:
            nonassoc += 1
    ok = (triv, assoc, nonassoc) == (133, 42, 168)
    clause('C1', ok, f'343 partition: trivial={triv} fano={assoc} nonassoc={nonassoc}')


# ============================================================================
# C2: octonion associator census
# ============================================================================

def c2_octonion():
    mult = {}
    for (i, j, k) in FANO_LINES:
        mult[(i, j)] = (1, k)
        mult[(j, k)] = (1, i)
        mult[(k, i)] = (1, j)
        mult[(j, i)] = (-1, k)
        mult[(k, j)] = (-1, i)
        mult[(i, k)] = (-1, j)
    for i in range(1, 8):
        mult[(i, i)] = (-1, 0)  # e_i^2 = -1 ; index 0 = real unit

    def omult(x, y):
        sx, ix = x
        sy, iy = y
        if ix == 0:
            return (sx * sy, iy)
        if iy == 0:
            return (sx * sy, ix)
        s, k = mult[(ix, iy)]
        return (sx * sy * s, k)

    nz = 0
    norms = Counter()
    for a, b, c in itertools.product(range(1, 8), repeat=3):
        lhs = omult(omult((1, a), (1, b)), (1, c))
        rhs = omult((1, a), omult((1, b), (1, c)))
        if lhs != rhs:
            nz += 1
            # associator = lhs - rhs; for distinct non-collinear indices the
            # two products are opposite, giving coefficient +-2 (norm 2).
            norms[abs(lhs[0] - rhs[0])] += 1
    ok = nz == 168 and set(norms) == {2}
    clause('C2', ok, f'octonion nonzero basis associators={nz}, norm set={sorted(norms)}')


# ============================================================================
# C3: Fano plane structural facts
# ============================================================================

def c3_fano_facts():
    pts = set(range(1, 8))
    pair_line = {}
    unique = True
    for p, q in itertools.combinations(pts, 2):
        containing = [l for l in FANO_LINE_SETS if p in l and q in l]
        if len(containing) != 1:
            unique = False
        pair_line[(p, q)] = containing[0] if containing else None
    point_degrees = Counter(p for l in FANO_LINES for p in l)
    intersecting = all(len(l1 & l2) == 1
                       for l1, l2 in itertools.combinations(FANO_LINE_SETS, 2))
    ok = (len(FANO_LINES) == 7 and unique
          and all(d == 3 for d in point_degrees.values()) and intersecting)
    clause('C3', ok,
           '7 lines; each pair on exactly 1 line (pairs cannot discriminate); '
           'each point on 3 lines; any two lines intersect')


# ============================================================================
# C4: CYP450 locus audit
# ============================================================================

def c4_locus_audit():
    mismatches = []
    for basis, (iso, paper_band) in PAPER_MAPPING.items():
        ref = REFERENCE_LOCI[iso]
        if not ref.startswith(paper_band):
            mismatches.append((iso, paper_band, ref))
    c2c = {b for b, (iso, _) in PAPER_MAPPING.items()
           if iso in ('CYP2C8', 'CYP2C9', 'CYP2C19')}
    c2c_is_line = frozenset(c2c) in FANO_LINE_SETS
    same_locus = len({REFERENCE_LOCI[PAPER_MAPPING[b][0]] for b in c2c}) == 1
    ok = not mismatches and c2c_is_line and same_locus
    clause('C4', ok,
           f'locus mismatches={mismatches}; CYP2C trio all at '
           f'{REFERENCE_LOCI["CYP2C9"]} and on Fano line {sorted(c2c)}: '
           f'{c2c_is_line and same_locus}')


# ============================================================================
# C5: gauge analysis of the bijection
# ============================================================================

def fano_automorphisms():
    pts = list(range(1, 8))
    autos = []
    for perm in itertools.permutations(pts):
        m = dict(zip(pts, perm))
        if all(frozenset(m[i] for i in l) in FANO_LINE_SETS for l in FANO_LINES):
            autos.append(m)
    return autos


def c5_gauge():
    isoforms = [PAPER_MAPPING[b][0] for b in range(1, 8)]
    c2c_set = {'CYP2C8', 'CYP2C9', 'CYP2C19'}
    big3_set = {'CYP2D6', 'CYP3A4', 'CYP2C9'}

    def constraint_count(extra_sets):
        valid = []
        for perm in itertools.permutations(range(1, 8)):
            m = dict(zip(isoforms, perm))
            if frozenset(m[i] for i in c2c_set) not in FANO_LINE_SETS:
                continue
            if any(frozenset(m[i] for i in s) not in FANO_LINE_SETS
                   for s in extra_sets):
                continue
            valid.append(m)
        return valid

    def equivalence_classes(valid, autos):
        seen = set()
        classes = []
        for m in valid:
            key = tuple(sorted(m.items()))
            if key in seen:
                continue
            orbit = {tuple(sorted({i: a[p] for i, p in m.items()}.items()))
                     for a in autos}
            seen |= orbit
            classes.append((m, orbit))
        return classes

    autos = fano_automorphisms()
    n_autos = len(autos)

    valid1 = constraint_count([])
    classes1 = equivalence_classes(valid1, autos)

    # lines (as isoform triples) common to every class representative's orbit
    def line_isoforms(m):
        pts = {p: iso for iso, p in m.items()}
        return {tuple(sorted(pts[i] for i in l)) for l in FANO_LINES}

    common = None
    for m, orbit in classes1:
        # check the line assignment for every member of the orbit
        orbit_lines = [line_isoforms(dict(k)) for k in orbit]
        stable = set.intersection(*map(set, orbit_lines)) if orbit_lines else set()
        common = stable if common is None else (common & stable)
    # also across classes
    common_across = None
    for m, orbit in classes1:
        ls = line_isoforms(m)
        common_across = ls if common_across is None else (common_across & ls)

    valid2 = constraint_count([big3_set])
    classes2 = equivalence_classes(valid2, autos)

    # lines shared by both surviving classes under the big-three constraint
    shared2 = None
    for m, orbit in classes2:
        ls = line_isoforms(m)
        shared2 = ls if shared2 is None else (shared2 & ls)
    expect_shared2 = {
        tuple(sorted(c2c_set)),
        tuple(sorted(big3_set)),
        tuple(sorted({'CYP1A2', 'CYP2B6', 'CYP2C9'})),
    }

    # paper's own mapping must satisfy both constraints
    paper_m = {iso: b for b, (iso, _) in PAPER_MAPPING.items()}
    paper_ok = (frozenset(paper_m[i] for i in c2c_set) in FANO_LINE_SETS
                and frozenset(paper_m[i] for i in big3_set) in FANO_LINE_SETS)

    ok = (n_autos == 168 and len(valid1) == 1008 and len(classes1) == 6
          and common_across == {tuple(sorted(c2c_set))}
          and len(valid2) == 336 and len(classes2) == 2 and paper_ok
          and shared2 == expect_shared2)
    clause('C5', ok,
           f'|Aut|={n_autos}; C2C-line bijections={len(valid1)} in '
           f'{len(classes1)} classes; gauge-invariant lines={sorted(common_across)}; '
           f'+big3 constraint: {len(valid2)} bijections in {len(classes2)} classes '
           f'sharing {len(shared2)} lines; '
           f'paper mapping satisfies both: {paper_ok}')


# ============================================================================
# C6: Hamming/hydrophobicity table
# ============================================================================

def hamming_pairs(enc=BIOCHEM_ENC):
    vecs = {c: codon_vec(c, enc) for c in CODONS}
    nz = [c for c in CODONS if vecs[c] != ZERO6]
    by_d = {d: [] for d in range(1, 7)}
    for c1, c2 in itertools.combinations(nz, 2):
        by_d[hamming(vecs[c1], vecs[c2])].append((c1, c2))
    return nz, by_d


def c6_hamming_table():
    nz, by_d = hamming_pairs()
    counts = {d: len(v) for d, v in by_d.items()}
    expect_n = {1: 186, 2: 465, 3: 620, 4: 465, 5: 186, 6: 31}

    # Documented convention (reproduces the preprint's table):
    # Delta-H for pairs involving a stop codon uses H(stop) = 0.
    kd0 = dict(KD)
    kd0['*'] = 0.0
    means0 = {d: sum(abs(kd0[CODON_TABLE[a]] - kd0[CODON_TABLE[b]])
                     for a, b in pairs) / len(pairs)
              for d, pairs in by_d.items()}
    expect_means = {1: 2.03, 2: 3.02, 3: 3.53, 4: 3.78, 5: 3.92, 6: 4.00}
    means_rounded = {d: round(means0[d], 2) for d in means0}

    mono0 = all(means0[d] < means0[d + 1] for d in range(1, 6))

    xs, ys = [], []
    for d in range(1, 7):
        for a, b in by_d[d]:
            xs.append(d)
            ys.append(abs(kd0[CODON_TABLE[a]] - kd0[CODON_TABLE[b]]))
    r0 = pearson(xs, ys)
    rho0 = spearman(xs, ys)

    # Stop-excluded variant (robustness check).
    means_ex, xs_e, ys_e = {}, [], []
    for d in range(1, 7):
        vals = [abs(KD[CODON_TABLE[a]] - KD[CODON_TABLE[b]])
                for a, b in by_d[d]
                if CODON_TABLE[a] != '*' and CODON_TABLE[b] != '*']
        means_ex[d] = sum(vals) / len(vals)
        xs_e.extend([d] * len(vals))
        ys_e.extend(vals)
    mono_ex = all(means_ex[d] < means_ex[d + 1] for d in range(1, 6))
    r_ex = pearson(xs_e, ys_e)

    ok = (counts == expect_n and means_rounded == expect_means and mono0
          and abs(r0 - 0.199) < 0.001 and abs(rho0 - 0.208) < 0.001
          and mono_ex and abs(r_ex - 0.218) < 0.001)
    clause('C6', ok,
           f'N={dict(sorted(counts.items()))}; means(stop=0)='
           f'{[round(means0[d], 3) for d in range(1, 7)]} mono={mono0}; '
           f'r={r0:.4f} rho={rho0:.4f}; stop-excluded means='
           f'{[round(means_ex[d], 3) for d in range(1, 7)]} mono={mono_ex} '
           f'r={r_ex:.4f}')
    return by_d, kd0, r0


# ============================================================================
# C7: permutation test for the Hamming/hydrophobicity correlation
# ============================================================================

def c7_permutation(by_d, kd0, r_obs, n_perm=10000, seed=168):
    rng = random.Random(seed)
    pairs = [(d, CODON_TABLE[a], CODON_TABLE[b])
             for d in range(1, 7) for a, b in by_d[d]]
    xs = [p[0] for p in pairs]
    categories = list(kd0.keys())
    values = [kd0[a] for a in categories]
    count_ge = 0
    for _ in range(n_perm):
        pv = values[:]
        rng.shuffle(pv)
        pmap = dict(zip(categories, pv))
        ys = [abs(pmap[a] - pmap[b]) for _, a, b in pairs]
        if pearson(xs, ys) >= r_obs:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)
    ok = p < 0.05
    clause('C7', ok,
           f'permutation test (n={n_perm}, seed={seed}): p={p:.4f} '
           f'(expected ~0.0147); r_obs={r_obs:.4f}')


# ============================================================================
# C8: encoding audit (limitation clause)
# ============================================================================

def c8_encoding_audit():
    def pearson_for_enc(enc):
        _, by_d = hamming_pairs(enc)
        kd0 = dict(KD)
        kd0['*'] = 0.0
        xs, ys = [], []
        for d in range(1, 7):
            for a, b in by_d[d]:
                xs.append(d)
                ys.append(abs(kd0[CODON_TABLE[a]] - kd0[CODON_TABLE[b]]))
        return pearson(xs, ys)

    r_biochem = pearson_for_enc(BIOCHEM_ENC)

    # all encodings with A = (0,0) (the zero-codon convention): 6 total
    res6 = []
    for perm in itertools.permutations([(0, 1), (1, 0), (1, 1)]):
        enc = {'A': (0, 0), 'G': perm[0], 'U': perm[1], 'C': perm[2]}
        res6.append((pearson_for_enc(enc), enc))
    res6.sort(key=lambda t: -t[0])
    rank6 = next(i for i, (_, e) in enumerate(res6, 1) if e == BIOCHEM_ENC)

    # plain nucleotide Hamming distance on codon strings
    nz = [c for c in CODONS if c != 'AAA']
    kd0 = dict(KD)
    kd0['*'] = 0.0
    xs, ys = [], []
    for c1, c2 in itertools.combinations(nz, 2):
        xs.append(sum(a != b for a, b in zip(c1, c2)))
        ys.append(abs(kd0[CODON_TABLE[c1]] - kd0[CODON_TABLE[c2]]))
    r_nuc = pearson(xs, ys)

    ok = (rank6 == 3 and abs(r_biochem - 0.199) < 0.001
          and abs(r_nuc - 0.261) < 0.001 and r_nuc > r_biochem)
    clause('C8', ok,
           f'biochemical encoding ranks {rank6}/6 (A=(0,0) fixed); '
           f'r_biochem={r_biochem:.4f} < r_nucleotide={r_nuc:.4f} '
           f'-> embedding is NOT hydrophobicity-optimal (reported as limitation)')


# ============================================================================
# C9: mutation robustness
# ============================================================================

def c9_mutation():
    vecs = {c: codon_vec(c) for c in CODONS}
    v2c = {v: c for c, v in vecs.items()}
    syn = cls = tot = 0
    for c in CODONS:
        v = vecs[c]
        if v == ZERO6:
            continue
        for i in range(6):
            v2 = list(v)
            v2[i] ^= 1
            v2 = tuple(v2)
            if v2 == ZERO6:
                continue  # mutations onto AAA excluded (preprint convention)
            c2 = v2c[v2]
            tot += 1
            a1, a2 = CODON_TABLE[c], CODON_TABLE[c2]
            if a1 == a2:
                syn += 1
            if CHEM_CLASS[a1] == CHEM_CLASS[a2]:
                cls += 1
    nz = [c for c in CODONS if vecs[c] != ZERO6]
    class_counts = Counter(CHEM_CLASS[CODON_TABLE[c]] for c in nz)
    p_base = sum(n * (n - 1) for n in class_counts.values()) / (63 * 62)
    p_binom = binom_sf_ge(tot, cls, p_base)
    ok = (tot == 372 and syn == 98 and cls == 208
          and abs(p_base - 0.309) < 0.001 and p_binom < 1e-10)
    clause('C9', ok,
           f'372 single-bit mutations: synonymous={syn} (26.3%), '
           f'class-preserving={cls} (55.9%) vs baseline {p_base * 100:.1f}%; '
           f'binomial p={p_binom:.2e}')


# ============================================================================
# C10: Fano-line class coherence in PG(5,2) (negative claim)
# ============================================================================

def c10_pg52_lines():
    vecs = {c: codon_vec(c) for c in CODONS if codon_vec(c) != ZERO6}
    v2c = {v: c for c, v in vecs.items()}
    lines = set()
    for v1, v2 in itertools.combinations(vecs.values(), 2):
        v3 = tuple(a ^ b for a, b in zip(v1, v2))
        if v3 != ZERO6 and v3 in v2c:
            lines.add(tuple(sorted((v1, v2, v3))))
    same_class = 0
    for line in lines:
        cs = [v2c[v] for v in line]
        cl = [CHEM_CLASS[CODON_TABLE[c]] for c in cs]
        if cl[0] == cl[1] == cl[2]:
            same_class += 1
    class_counts = Counter(CHEM_CLASS[CODON_TABLE[c]] for c in vecs)
    p_base = sum(n * (n - 1) * (n - 2) for n in class_counts.values()) / (63 * 62 * 61)
    p_binom = binom_cdf_le(len(lines), same_class, p_base)
    ok = (len(lines) == 651 and same_class == 64
          and abs(p_base - 0.115) < 0.001 and p_binom > 0.05)
    clause('C10', ok,
           f'{len(lines)} Fano lines; same-class={same_class} '
           f'({same_class / len(lines) * 100:.1f}%) vs baseline '
           f'{p_base * 100:.1f}%; binomial p={p_binom:.3f} '
           f'(non-significant -> confirms metric-not-subplane claim)')


# ============================================================================

def main():
    print('=' * 72)
    print('  168 BIOLOGY PREPRINT — BIOLOGICAL VALIDATION CONTRACT')
    print('=' * 72)
    c1_partition()
    c2_octonion()
    c3_fano_facts()
    c4_locus_audit()
    c5_gauge()
    by_d, kd0, r0 = c6_hamming_table()
    c7_permutation(by_d, kd0, r0)
    c8_encoding_audit()
    c9_mutation()
    c10_pg52_lines()

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print()
    print(f'{n_pass} clauses passed, {n_fail} failed')
    if n_fail == 0:
        print('BIO168_VALIDATION_VERDICT C_GREEN')
    else:
        print('BIO168_VALIDATION_VERDICT C_RED')
        for name, ok, detail in RESULTS:
            if not ok:
                print(f'  FAILED: {name} {detail}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
