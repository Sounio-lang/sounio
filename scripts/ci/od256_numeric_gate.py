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

def gen(dir_, ncases, seed):
    mp.prec = 800
    os.makedirs(dir_, exist_ok=True)
    rng = random.Random(seed)
    manifest = []
    for name, cfg in KERNELS.items():
        stride, scalar, op = cfg["stride"], cfg["scalar"], cfg["op"]
        mem = [0.0] * (ncases * stride)
        cases = []
        for t in range(ncases):
            base = t * stride
            if scalar:
                a = _rand_scalar(rng); b = _rand_scalar(rng)
                mem[base + 0] = a; mem[base + 1] = b
                hi, lo = op(a, b)
                exp = [hi, lo]
                true = (mpf(a) + mpf(b)) if name == "od256_two_sum" else (mpf(a) * mpf(b))
                cases.append({"exp": exp, "exp_slots": [2, 3], "true": str(true)})
            else:
                a = _rand_od256(rng); b = _rand_od256(rng)
                for i in range(8): mem[base + i] = a[i]; mem[base + 8 + i] = b[i]
                exp = op(a, b)
                ta = sum(mpf(x) for x in a); tb = sum(mpf(x) for x in b)
                true = (ta + tb) if name == "od256_add" else (ta * tb)
                cases.append({"exp": exp, "exp_slots": list(range(16, 24)), "true": str(true)})
        with open(os.path.join(dir_, name + ".in.f64"), "wb") as f:
            f.write(struct.pack("<" + "d" * len(mem), *mem))
        with open(os.path.join(dir_, name + ".truth.json"), "w") as f:
            json.dump({"kernel": name, "stride": stride, "ncases": ncases,
                       "mem_words": len(mem), "cases": cases}, f)
        # manifest row: kernel, ptx-basename, mem_words, threads(=ncases)
        manifest.append(f"{name}\t{name}.ptx\t{len(mem)}\t{ncases}")
        print(f"  gen {name}: {ncases} cases, mem_words={len(mem)}")
    with open(os.path.join(dir_, "manifest.tsv"), "w") as f:
        f.write("\n".join(manifest) + "\n")

def _read_out(path, mem_words):
    with open(path, "rb") as f:
        data = f.read()
    return list(struct.unpack("<" + "d" * mem_words, data[:8 * mem_words]))

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
        out = _read_out(os.path.join(dir_, name + ".out.f64"), truth["mem_words"])
        stride = truth["stride"]; bad_exact = 0; worst = 999.0
        for t, c in enumerate(truth["cases"]):
            base = t * stride
            got = [out[base + s] for s in c["exp_slots"]]
            if got != c["exp"]:
                bad_exact += 1
            recon = sum(mpf(x) for x in got)
            worst = min(worst, _eff_bits(recon, c["true"]))
        ok = bad_exact == 0 and worst >= min_bits
        all_ok = all_ok and ok
        print(f"  {name}: {truth['ncases']-bad_exact}/{truth['ncases']} bit-exact vs ref, "
              f"worst {worst:.1f} bits -> {'PASS' if ok else 'FAIL'}")
    return all_ok

def local(dir_, ncases, seed, min_bits):
    """Synthesize outputs with the reference (== the GPU kernel) and check."""
    gen(dir_, ncases, seed)
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
    a = ap.parse_args()
    if a.gen:
        gen(a.gen, a.cases, a.seed); sys.exit(0)
    if a.check:
        sys.exit(0 if check(a.check, a.min_bits) else 1)
    # default / --local: self-test
    print(f"od256 numeric gate — LOCAL self-test ({a.cases} cases/kernel)")
    ok = local(a.dir, a.cases, a.seed, a.min_bits)
    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
