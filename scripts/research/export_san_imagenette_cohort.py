#!/usr/bin/env python3
"""Regenerate val_imagenette.u16 + expected.txt from saved confidences.

Usage after train_san_imagenette.py has produced
  datasets/imagenette2-160/san_val_confidences.npy
"""
import os
import json
import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "imagenette2-160")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "san_imagenette_real")
N_CONF = 4
MAX_POINTS = 8
Q_DELTA_RATIO = 0.55

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    confs = np.load(os.path.join(DATA_ROOT, "san_val_confidences.npy"))
    print(f"confidences shape={confs.shape}")

    q = np.clip(np.floor(confs * 32767.0), 0, 32767).astype(np.uint16)
    q = q[:, :N_CONF]

    cohort_path = os.path.join(OUT_DIR, "val_imagenette.u16")
    with open(cohort_path, "wb") as f:
        f.write(q.tobytes())

    lut = np.array([
        29583360,
        176877568,
        471269376,
        1060618240,
        2238156800,
        0, 0, 0
    ], dtype=np.uint64)

    q_delta = int(Q_DELTA_RATIO * 32767)
    exits = np.argmax((q >= q_delta).astype(np.int32), axis=1)
    exits = np.where(np.any(q >= q_delta, axis=1), exits, N_CONF)
    hist = np.bincount(exits, minlength=MAX_POINTS)
    cat = int(np.sum(exits == N_CONF))
    flops = int(np.sum(lut[exits]))

    expected = {
        "val_imagenette_shape": f"{q.shape[0]} {N_CONF}",
        "val_imagenette_family": "resnet",
        "val_imagenette_file": "val_imagenette.u16",
        "lut_resnet": " ".join(str(int(x)) for x in lut[:N_CONF + 1]),
        "q_delta_resnet": str(q_delta),
        "val_imagenette_hist": " ".join(str(int(x)) for x in hist[:N_CONF + 1]),
        "val_imagenette_catastrophes": str(cat),
        "val_imagenette_flops_macs": str(flops),
    }
    with open(os.path.join(OUT_DIR, "expected.txt"), "w") as f:
        for k, v in expected.items():
            f.write(f"{k} {v}\n")

    meta = {
        "dataset": "imagenette2-160",
        "n_samples": int(q.shape[0]),
        "n_conf": int(N_CONF),
        "q_delta": q_delta,
        "source": "real images, ResNet-18 pretrained on ImageNet-1k, SAN heads trained on ImageNette",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {cohort_path} ({q.shape[0]} samples)")
    print(f"hist={hist[:N_CONF+1].tolist()} cat={cat} flops={flops}")

if __name__ == "__main__":
    main()
