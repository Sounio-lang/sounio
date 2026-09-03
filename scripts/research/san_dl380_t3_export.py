#!/usr/bin/env python3
"""Export the exact I6 scan artifacts for the on-target T3 acceptance run.

Generates, on the control VM (or any node with torch), the byte-exact
inputs and expected outputs of the SAN-ImageNet FPGA scan contract (clause
I6 of scripts/research/san_imagenet_fpga_dl380.py):

  artifacts/san_dl380_t3/meta.json        LUTs, q_deltas, expected outputs
  artifacts/san_dl380_t3/val_<family>.u16 quantized val cohorts (uint16 LE)
  artifacts/san_dl380_t3/stress.u16       the exact 1.2M-sample stress cohort

The acceptance script (san_dl380_t3_acceptance.py, pure stdlib) recomputes
the scan on the target node and must reproduce every expected value
bit-exactly: that equality IS theorem T3 (deployment soundness via
platform-independent integer semantics).
"""
import array
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "artifacts", "san_dl380_t3")

spec = importlib.util.spec_from_file_location(
    "san_imagenet_fpga_dl380", os.path.join(HERE, "san_imagenet_fpga_dl380.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def dump_u16(path, arr):
    a = np.asarray(arr, dtype=np.uint16)
    array.array("H", a.reshape(-1).tolist()).tofile(open(path, "wb"))
    return list(a.shape)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    meta = {"seed": m.SEED, "q15": m.Q15,
            "q_delta": dict(m.Q_DELTA),
            "lut": {f: [int(v) for v in m.EVAL_LUT[f]] for f in ("resnet", "vit")},
            "datasets": {}}

    # ---- val cohorts (replicates the I6 per-family path exactly) ----------
    for family in ("resnet", "vit"):
        net, _, _ = m.train_san(family)
        conf_mat = m.collect_conf_matrix(net)
        q15 = m.quantize_conf(conf_mat)
        model = m.U250SanScanModel(m.EVAL_LUT[family], m.Q_DELTA[family])
        k_idx, k_hist, k_cat, k_flops = model.scan(q15)
        r_idx, r_hist, r_cat, r_flops = m.host_reference_scan(
            q15, m.EVAL_LUT[family], m.Q_DELTA[family])
        assert (np.array_equal(k_idx, r_idx) and np.array_equal(k_hist, r_hist)
                and k_cat == r_cat and k_flops == r_flops), f"I6 broken on {family}"
        shape = dump_u16(os.path.join(OUT, f"val_{family}.u16"), q15)
        meta["datasets"][f"val_{family}"] = {
            "file": f"val_{family}.u16", "shape": shape,
            "family": family,
            "expected": {"hist": [int(v) for v in k_hist],
                         "catastrophes": k_cat, "flops_macs": k_flops}}
        print(f"  export[val_{family}]: shape={shape} cat={k_cat} "
              f"flops={k_flops / 1e9:.3f} GMAC")

    # ---- 1.2M stress cohort (replicates the I6 stress path exactly) -------
    rng_s = np.random.default_rng(m.SEED + 100)
    hist_r = np.asarray(meta["datasets"]["val_resnet"]["expected"]["hist"],
                        dtype=np.int64)
    measured = hist_r / hist_r.sum()
    depth_stress = rng_s.choice(len(measured), size=m.N_IMAGENET, p=measured)
    n_stages = len(measured) - 1
    q = rng_s.integers(0, m.Q_DELTA["resnet"],
                       size=(m.N_IMAGENET, n_stages)).astype(np.int64)
    hit_mask = ((np.arange(n_stages)[None, :] == depth_stress[:, None])
                & (depth_stress[:, None] < n_stages))
    q[hit_mask] = rng_s.integers(m.Q_DELTA["resnet"], m.Q15,
                                 size=int(hit_mask.sum()))
    lut = m.EVAL_LUT["resnet"]
    k_idx, k_hist, k_cat, k_flops = m.U250SanScanModel(
        lut, m.Q_DELTA["resnet"]).scan(q)
    r_idx, r_hist, r_cat, r_flops = m.host_reference_scan(
        q, lut, m.Q_DELTA["resnet"])
    assert (np.array_equal(k_idx, r_idx) and np.array_equal(k_hist, r_hist)
            and k_cat == r_cat and k_flops == r_flops), "I6 stress broken"
    shape = dump_u16(os.path.join(OUT, "stress.u16"), q)
    meta["datasets"]["stress_1p2M"] = {
        "file": "stress.u16", "shape": shape, "family": "resnet",
        "expected": {"hist": [int(v) for v in k_hist],
                     "catastrophes": k_cat, "flops_macs": k_flops}}
    meta["accumulator_bound"] = m.N_IMAGENET * int(m.EVAL_LUT["vit"][-1])
    print(f"  export[stress_1p2M]: shape={shape} cat={k_cat} "
          f"flops={k_flops / 1e9:.3f} GMAC")

    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    # flat key=value rendering for the C++ host (no JSON parser needed)
    with open(os.path.join(OUT, "expected.txt"), "w") as f:
        f.write(f"q15 {meta['q15']}\n")
        for fam in ("resnet", "vit"):
            f.write(f"q_delta_{fam} {meta['q_delta'][fam]}\n")
            f.write(f"lut_{fam} {' '.join(str(v) for v in meta['lut'][fam])}\n")
        f.write(f"accumulator_bound {meta['accumulator_bound']}\n")
        for name, ds in meta["datasets"].items():
            e = ds["expected"]
            f.write(f"{name}_family {ds['family']}\n")
            f.write(f"{name}_file {ds['file']}\n")
            f.write(f"{name}_shape {' '.join(str(v) for v in ds['shape'])}\n")
            f.write(f"{name}_hist {' '.join(str(v) for v in e['hist'])}\n")
            f.write(f"{name}_catastrophes {e['catastrophes']}\n")
            f.write(f"{name}_flops_macs {e['flops_macs']}\n")
    print(f"T3_EXPORT_OK dir={OUT} elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
