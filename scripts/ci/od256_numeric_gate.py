#!/usr/bin/env python3
"""od256 GPU numeric gate — drive the emitted kernels on real inputs and compare
the read-back limbs to the mpmath oracle.

Layout: one thread per test case (tid t reads/writes at t*stride).
  two_sum / two_prod : stride 4  — in[0]=a in[1]=b ; out[2],out[3]
  add / mul          : stride 24 — in[0..7]=a in[8..15]=b ; out[16..23]

Flow:
  --gen DIR    : write <kernel>.in.f64 (raw little-endian doubles, the input mem
                 buffer) + <kernel>.truth.json (expected out limbs + mpmath true
                 value + mem_words) for the worker.
  worker       : kaxi_ptx_runner <kernel>.ptx --value-type f64 \
                   --mem-words <W> --threads <T> --init-file <kernel>.in.f64 \
                   --dump-file <kernel>.out.f64
  --check DIR  : read <kernel>.out.f64, compare out limbs to expected
                 (bit-exact — od256 uses Dekker split + IEEE ops, no FMA, so GPU
                 == the CPU reference) and reconstruct vs mpmath -> effective
                 bits. PASS if all cases bit-exact and >= --min-bits.
  --local      : gen, synthesize outputs with the reference (== the GPU kernel),
                 then check — validates the fixture/compare plumbing without GPU.

od256 f64 kernels are correctly-rounded IEEE-754 with no FMA, so GPU output is
bit-identical to scripts/ci/od256_renorm_gpu_ref.py (already proven vs the PTX
simulator). This gate confirms that on real silicon.
"""
import argparse, json, os, struct, random, sys
from mpmath import mp, mpf, fabs, log

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import od256_renorm_gpu_ref as ref

KERNELS = {
    "od256_two_sum":  {"stride": 4,  "scalar": True,  "op": ref.two_sum},
    "od256_two_prod": {"stride": 4,  "scalar": True,  "op": ref._two_prod},
    "od256_add":      {"stride": 24, "scalar": False, "op": ref.od_add_gpu},
    "od256_mul":      {"stride": 24, "scalar": False, "op": ref.od_mul_gpu},
}

def _rand_scalar(rng): return rng.uniform(-1e6, 1e6)
def _rand_od256(rng):
    v = mpf(rng.getrandbits(420)) / mpf(2)**rng.randint(-30, 30) * (1 if rng.random() < .5 else -1)
    return ref._split_mpf(v)

# --- adversarial inputs: the surface uniform-random never touches -------------
# Focus on the VALID domain (no overflow/nan): denormals (the FTZ cross-arch
# footgun), catastrophic cancellation (exercises the error limbs + full renorm),
# sub-ULP, signed zeros, powers of two, wide dynamic range. Stay inside Dekker's
# safe range (|x0| < ~2^900) so two_prod's split doesn't overflow.
DEN_MIN = 5e-324          # smallest positive subnormal
DEN_NORM = 2.0 ** -1022   # smallest normal

def _adv_scalar_pairs(rng, n):
    fixed = [
        (0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0),
        (1.0, -1.0), (1.0, 1.0), (-1.0, -1.0),
        (2.0**53, 1.0), (2.0**53, -1.0), (2.0**53, 0.5),      # rounding boundary
        (1.0, 2.0**-53), (1.0, 2.0**-54), (1.0, -2.0**-54),   # sub-ULP
        (DEN_MIN, DEN_MIN), (DEN_MIN, -DEN_MIN), (DEN_NORM, DEN_MIN),  # denormal add
        (1.0, DEN_MIN), (2.0**-200, DEN_MIN),                 # normal*denormal -> denormal (FTZ probe)
        (DEN_NORM, DEN_NORM), (2.0**-540, 2.0**-540),         # product underflows to denormal
        (3.0, 7.0), (0.1, 0.2), (2.0**500, 2.0**-500),        # wide range
    ]
    out = list(fixed)
    while len(out) < n:
        # mix: near-cancellation and denormal-scale random draws
        e = rng.randint(-1060, 500)
        a = rng.uniform(-1, 1) * 2.0**e
        if rng.random() < 0.5:                                # near-cancellation partner
            b = -a * (1.0 + rng.choice([0.0, 2.0**-52, 2.0**-40]))
        else:
            b = rng.uniform(-1, 1) * 2.0**rng.randint(-1060, 500)
        out.append((a, b))
    return out[:n]

def _adv_od256_pairs(rng, n, op_add):
    """n adversarial (a8,b8) pairs, all in the valid (finite, non-overflow) domain.
    op_add=True: cancellation-heavy (add). op_add=False: mul — pick a target PRODUCT
    exponent and split it across operands so a*b stays finite, spanning denormal
    (FTZ probe) → normal → near-overflow product regimes."""
    def mant(): return (mpf(rng.getrandbits(420)) / mpf(2)**420) * 2 - 1  # in (-1,1)
    def val(lo, hi): return mant() * mpf(2) ** rng.randint(lo, hi)
    out = []
    z = ref._split_mpf(mpf(0))
    out += [(z, ref._split_mpf(mpf(1))), (ref._split_mpf(mpf(1)), z), (z, z)]
    while len(out) < n:
        r = rng.random()
        if op_add:
            if r < 0.45:                                      # near cancellation: a + b ~ tiny
                v = val(-40, 40); a = ref._split_mpf(v)
                b = ref._split_mpf(-v + v * mpf(2) ** rng.randint(-410, -360))
            elif r < 0.7:                                     # denormal-scale operands
                a = ref._split_mpf(val(-1040, -990)); b = ref._split_mpf(val(-1040, -990))
            elif r < 0.85:                                    # huge (Dekker-safe) operands
                a = ref._split_mpf(val(760, 890)); b = ref._split_mpf(val(760, 890))
            else:                                             # wide range between operands
                a = ref._split_mpf(val(200, 300)); b = ref._split_mpf(val(-300, -200))
        else:                                                 # mul: bound the product exponent
            # target product regime: denormal / normal / near-overflow
            T = rng.choice([rng.randint(-1060, -1010),        # denormal product (FTZ probe)
                            rng.randint(-40, 40),             # normal
                            rng.randint(860, 900)])           # near-overflow (< 2^1023)
            sa = rng.randint(max(-900, T - 890), min(890, T + 890))
            sb = T - sa                                       # a*b ~ 2^(sa+sb) = 2^T
            a = ref._split_mpf(mant() * mpf(2) ** sa)
            b = ref._split_mpf(mant() * mpf(2) ** sb)
        out.append((a, b))
    return out[:n]

def gen(dir_, ncases, seed, adversarial=False):
    mp.prec = 800
    os.makedirs(dir_, exist_ok=True)
    rng = random.Random(seed)
    manifest = []
    for name, cfg in KERNELS.items():
        stride, scalar, op = cfg["stride"], cfg["scalar"], cfg["op"]
        if adversarial:
            pairs = _adv_scalar_pairs(rng, ncases) if scalar \
                else _adv_od256_pairs(rng, ncases, op_add=(name == "od256_add"))
        else:
            pairs = None
        mem = [0.0] * (ncases * stride)
        cases = []
        for t in range(ncases):
            base = t * stride
            if scalar:
                a, b = pairs[t] if adversarial else (_rand_scalar(rng), _rand_scalar(rng))
                mem[base + 0] = a; mem[base + 1] = b
                hi, lo = op(a, b)
                exp = [hi, lo]
                true = (mpf(a) + mpf(b)) if name == "od256_two_sum" else (mpf(a) * mpf(b))
                # multiplicative EFTs (Dekker) lose exactness once the product OR any
                # of its error limbs underflows into the denormal range — outside the
                # algorithm's guarantee, so don't gate bit-exactness there. A 2-limb
                # result needs |value| >= 2^-1022 * 2^53 for the low limb to stay normal.
                in_domain = _mul_in_domain(name, "od256_two_prod", true, 2)
                cases.append({"exp": exp, "exp_slots": [2, 3], "true": str(true), "dom": in_domain})
            else:
                a, b = pairs[t] if adversarial else (_rand_od256(rng), _rand_od256(rng))
                for i in range(8): mem[base + i] = a[i]; mem[base + 8 + i] = b[i]
                exp = op(a, b)
                ta = sum(mpf(x) for x in a); tb = sum(mpf(x) for x in b)
                true = (ta + tb) if name == "od256_add" else (ta * tb)
                # 8-limb result: all limbs stay normal only if |value| >= 2^-1022 * 2^(53*7).
                in_domain = _mul_in_domain(name, "od256_mul", true, 8)
                cases.append({"exp": exp, "exp_slots": list(range(16, 24)), "true": str(true), "dom": in_domain})
        with open(os.path.join(dir_, name + ".in.f64"), "wb") as f:
            f.write(struct.pack("<" + "d" * len(mem), *mem))
        with open(os.path.join(dir_, name + ".truth.json"), "w") as f:
            json.dump({"kernel": name, "stride": stride, "ncases": ncases, "mem_words": len(mem),
                       "mode": "adversarial" if adversarial else "random", "cases": cases}, f)
        # manifest row: kernel, ptx-basename, mem_words, threads(=ncases)
        manifest.append(f"{name}\t{name}.ptx\t{len(mem)}\t{ncases}")
        print(f"  gen {name}: {ncases} {'ADVERSARIAL' if adversarial else ''} cases, mem_words={len(mem)}")
    with open(os.path.join(dir_, "manifest.tsv"), "w") as f:
        f.write("\n".join(manifest) + "\n")

def _read_out(path, mem_words):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack("<" + "d" * mem_words, data[:8 * mem_words]))

F64_MIN_NORMAL = 2.2250738585072014e-308

def _has_denormal(limbs):
    return any(x != 0.0 and abs(x) < F64_MIN_NORMAL for x in limbs)

def _mul_in_domain(name, mul_name, true, nlimbs):
    """A multiplicative-EFT result is in-domain (Dekker exactness holds) only if the
    whole L-limb expansion stays in the normal range: |value| >= 2^-1022 * 2^(53*(L-1)).
    Additive kernels are always in-domain (TwoSum is exact for all finite inputs)."""
    if name != mul_name or true == 0:
        return True
    return abs(true) >= mpf(2) ** (-1022 + 53 * (nlimbs - 1))

def _limbs_equal(got, exp):
    import math
    for g, e in zip(got, exp):
        if g == e: continue
        if math.isnan(g) and math.isnan(e): continue   # both-NaN = same outcome
        return False
    return True

def _eff_bits(approx, true):
    mp.prec = 800
    t = mpf(true)
    if t == 0: return 999.0
    r = fabs((mpf(approx) - t) / t)
    return 999.0 if r == 0 else float(-log(r, 2))

def check(dir_, min_bits):
    mp.prec = 800
    all_ok = True
    for name, cfg in KERNELS.items():
        truth = json.load(open(os.path.join(dir_, name + ".truth.json")))
        adv = truth.get("mode") == "adversarial"
        out = _read_out(os.path.join(dir_, name + ".out.f64"), truth["mem_words"])
        stride = truth["stride"]
        bad_gated = 0; n_gated = 0; bad_underflow = 0; n_underflow = 0
        worst = 999.0; worst_case = -1
        for t, c in enumerate(truth["cases"]):
            base = t * stride
            got = [out[base + s] for s in c["exp_slots"]]
            eq = _limbs_equal(got, c["exp"])
            if c.get("dom", True):
                n_gated += 1
                if not eq:
                    bad_gated += 1
                    if bad_gated <= 3:
                        print(f"      GATED MISMATCH case {t}: got={got} exp={c['exp']}")
            else:                                    # underflow regime: characterized, not gated
                n_underflow += 1
                if not eq: bad_underflow += 1
            recon = sum(mpf(x) for x in got)
            b = _eff_bits(recon, c["true"])
            if b < worst: worst, worst_case = b, t
        # gate on in-domain cases only. In adversarial mode effective bits are
        # informational (cancellation legitimately loses them).
        ok = bad_gated == 0 if adv else (bad_gated == 0 and worst >= min_bits)
        all_ok = all_ok and ok
        tag = "ADVERSARIAL " if adv else ""
        bits = "n/a" if adv else f"{worst:.1f} bits (case {worst_case})"
        extra = ""
        if n_underflow:
            extra = f"; underflow regime {n_underflow} cases ({bad_underflow} differ from FTZ-free CPU ref, out-of-domain)"
        print(f"  {tag}{name}: {n_gated-bad_gated}/{n_gated} in-domain bit-exact vs ref, "
              f"worst {bits} -> {'PASS' if ok else 'FAIL'}{extra}")
    return all_ok

def local(dir_, ncases, seed, min_bits, adversarial=False):
    """Synthesize outputs with the reference (== the GPU kernel) and check."""
    gen(dir_, ncases, seed, adversarial=adversarial)
    for name, cfg in KERNELS.items():
        truth = json.load(open(os.path.join(dir_, name + ".truth.json")))
        mem = [0.0] * truth["mem_words"]; stride = truth["stride"]
        # replay inputs from the .in.f64 and place expected outputs (ref == GPU)
        inp = _read_out(os.path.join(dir_, name + ".in.f64"), truth["mem_words"])
        for t, c in enumerate(truth["cases"]):
            base = t * stride
            for k, s in enumerate(c["exp_slots"]): mem[base + s] = c["exp"][k]
        with open(os.path.join(dir_, name + ".out.f64"), "wb") as f:
            f.write(struct.pack("<" + "d" * len(mem), *mem))
    return check(dir_, min_bits)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", metavar="DIR")
    ap.add_argument("--check", metavar="DIR")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--dir", default="/tmp/od256_numeric")
    ap.add_argument("--cases", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--min-bits", type=float, default=400.0)
    ap.add_argument("--adversarial", action="store_true",
                    help="edge-case inputs (denormals/cancellation/dynamic range); pass = bit-exact vs ref")
    a = ap.parse_args()
    if a.gen:
        gen(a.gen, a.cases, a.seed, adversarial=a.adversarial); sys.exit(0)
    if a.check:
        sys.exit(0 if check(a.check, a.min_bits) else 1)
    # default / --local: self-test
    print(f"od256 numeric gate — LOCAL self-test ({a.cases} cases/kernel, "
          f"{'adversarial' if a.adversarial else 'random'})")
    ok = local(a.dir, a.cases, a.seed, a.min_bits, adversarial=a.adversarial)
    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
