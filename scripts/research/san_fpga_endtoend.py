#!/usr/bin/env python3
"""san_fpga_endtoend.py — SAN-ImageNette FPGA end-to-end loop.

Loads (or trains) a SAN-ResNet-18 on ImageNette2-160, runs the validation cohort
through the PyTorch trunk, quantizes exit confidences to Q0.15, packs them into
the same 512-bit beat format used by host_san_scan.cpp, and streams the packed
cohort to host_san_scan_e2e.cpp on the DL380/U250 over stdin.

Modes:
  real (default):  spawn host_san_scan_e2e <xclbin> and parse card output.
  --mock-host:     skip XRT; run the integer scan in Python and validate that
                   the Python packing matches an independent reference.

The script is self-contained: it can import the model factory from
train_san_imagenette.py, but it also carries a local copy of the SAN-ResNet-18
class so it survives if that file is under active edit elsewhere.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_ROOT = os.path.join(REPO, "datasets", "imagenette2-160")
OUT_DIR = os.path.join(REPO, "artifacts", "san_imagenette_real")
CHECKPOINT = os.path.join(OUT_DIR, "model.pt")
META_PATH = os.path.join(OUT_DIR, "meta.json")

BATCH_SIZE = 128
EPOCHS = 3
LR = 1e-4
TRAIN_SUBSET = 4000
N_CONF = 4
MAX_POINTS = 8
LANES = 4
Q15 = 32767


def make_san_resnet18(num_classes=10, n_conf=N_CONF):
    """Same SAN-ResNet-18 early-exit model as train_san_imagenette.py."""
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in base.parameters():
        p.requires_grad = False
    for p in base.layer4.parameters():
        p.requires_grad = True

    class SAN(nn.Module):
        def __init__(self, base, num_classes, n_conf):
            super().__init__()
            self.conv1 = base.conv1
            self.bn1 = base.bn1
            self.relu = base.relu
            self.maxpool = base.maxpool
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.avgpool = base.avgpool
            self.head_dims = [64, 128, 256, 512, 512]
            self.heads = nn.ModuleList([nn.Linear(d, num_classes) for d in self.head_dims])

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            outs = []
            for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
                x = layer(x)
                h = self.avgpool(x).view(x.size(0), -1)
                outs.append(self.heads[len(outs)](h))
            h = self.avgpool(x).view(x.size(0), -1)
            outs.append(self.heads[-1](h))
            return outs

    return SAN(base, num_classes, n_conf)


def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_full = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=transform)
    if TRAIN_SUBSET and TRAIN_SUBSET < len(train_full):
        import random
        indices = random.Random(42).sample(range(len(train_full)), TRAIN_SUBSET)
        train_ds = Subset(train_full, indices)
    else:
        train_ds = train_full
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, len(train_full.classes)


def train_model(model, loader, device):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x, y in tqdm(loader, desc=f"epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            outs = model(x)
            loss = sum(F.cross_entropy(o, y) for o in outs)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}: loss={total_loss/len(loader):.3f}")


def extract_confidences(model, loader, device):
    model.eval()
    all_conf = []
    all_labels = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="forward"):
            x = x.to(device)
            outs = model(x)
            confs = torch.stack([F.softmax(o, dim=1).max(dim=1)[0] for o in outs], dim=1)
            all_conf.append(confs.cpu().numpy())
            all_labels.append(y.numpy())
    t1 = time.perf_counter()
    conf = np.concatenate(all_conf, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return conf, labels, t1 - t0


def quantize_pack(conf, q_delta_f):
    """Quantize confidences to Q0.15 and pack into 512-bit beats.

    Returns (q uint16[n_samples, n_conf], beats uint64[n_words*8]).
    """
    q = np.clip(np.floor(conf * Q15), 0, Q15).astype(np.uint16)
    q = q[:, :N_CONF]
    n_samples, n_conf = q.shape
    n_words = (n_samples + LANES - 1) // LANES
    beats = np.zeros(n_words * 8, dtype=np.uint64)

    for s in range(n_samples):
        lane = s % LANES
        beat = s // LANES
        for k in range(n_conf):
            bit_lo = beat * 512 + lane * 128 + k * 15
            wi = bit_lo // 64
            off = bit_lo % 64
            v = int(q[s, k]) & 0x7FFF
            beats[wi] &= ~(np.uint64(0x7FFF) << np.uint64(off))
            beats[wi] |= np.uint64(v) << np.uint64(off)
            if off > 49:
                rem = int(off) - 49
                beats[wi + 1] &= ~(np.uint64(0x7FFF) >> np.uint64(15 - rem))
                beats[wi + 1] |= np.uint64(v) >> np.uint64(15 - rem)

    return q, beats, n_words


def host_reference_scan(q, lut, q_delta):
    """Independent Python golden for the scan semantics."""
    n_conf = q.shape[1]
    n_points = n_conf + 1
    idx = np.argmax((q >= q_delta).astype(np.int32), axis=1)
    settled = np.any(q >= q_delta, axis=1)
    idx = np.where(settled, idx, n_conf)
    hist = np.bincount(idx, minlength=MAX_POINTS)
    cat = int(np.sum(~settled))
    flops = int(np.sum(lut[idx]))
    return idx, hist, cat, flops


def build_header(n_samples, n_conf, q_delta, lut):
    """Little-endian binary header for host_san_scan_e2e."""
    pieces = [struct.pack("<III", n_samples, n_conf, q_delta)]
    pieces.append(struct.pack("<I", 0))  # reserved
    for v in lut:
        pieces.append(struct.pack("<Q", int(v)))
    return b"".join(pieces)


def run_host_e2e(xclbin, header, beats_bytes):
    """Spawn host_san_scan_e2e and return parsed stdout."""
    exe = os.path.join(REPO, "hardware", "fpga", "u250_catastrophe_scan", "host_san_scan_e2e")
    if not os.path.isfile(exe):
        raise FileNotFoundError(f"host_san_scan_e2e not found at {exe}")
    proc = subprocess.Popen(
        [exe, xclbin],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate(header + beats_bytes, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"host_san_scan_e2e failed (code {proc.returncode}):\n{stderr.decode()}")
    return stdout.decode(), stderr.decode()


def parse_e2e_output(text):
    result = {}
    for line in text.strip().splitlines():
        if line.startswith("E2E_RESULT "):
            for token in line.split()[1:]:
                if "=" in token:
                    k, v = token.split("=", 1)
                    result[k] = v
        elif line.startswith("E2E_HIST "):
            parts = line.split()
            result.setdefault("hist", []).append(int(parts[2]))
        elif line.startswith("HOST_SAN_SCAN_E2E_PASS"):
            result["pass"] = True
    return result


def run_mock_host(q, lut, q_delta):
    """Mock host: compute scan in Python and return synthetic timing output."""
    t0 = time.perf_counter()
    idx, hist, cat, flops = host_reference_scan(q, lut, q_delta)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1e3
    msamples = q.shape[0] / total_ms / 1e3 if total_ms > 0 else 0
    n_points = q.shape[1] + 1
    lines = [
        f"E2E_RESULT n={q.shape[0]} n_conf={q.shape[1]} q_delta={q_delta} "
        f"catastrophes={cat} flops_macs={flops} setup_ms=0.000 dma_h2d_ms=0.000 "
        f"kernel_ms={total_ms:.3f} dma_d2h_ms=0.000 total_ms={total_ms:.3f} "
        f"Msamples/s={msamples:.2f}"
    ]
    for b in range(n_points):
        lines.append(f"E2E_HIST {b} {int(hist[b])}")
    lines.append("HOST_SAN_SCAN_E2E_PASS")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="SAN-FPGA end-to-end loop")
    parser.add_argument("--xclbin", default="krnl_san_scan.hw.xclbin",
                        help="XRT bitstream path (real mode)")
    parser.add_argument("--mock-host", action="store_true",
                        help="Skip XRT; validate Python scan/packing only")
    parser.add_argument("--q-delta", type=float, default=0.55,
                        help="Float confidence threshold (default 0.55)")
    parser.add_argument("--train", action="store_true",
                        help="Force retrain even if checkpoint exists")
    parser.add_argument("--device", default="auto",
                        help="torch device (auto/cpu/cuda)")
    args = parser.parse_args()

    if not os.path.isdir(DATA_ROOT):
        print(f"ERROR: ImageNette2-160 not found at {DATA_ROOT}", file=sys.stderr)
        print("Download and extract it from https://github.com/fastai/imagenette",
              file=sys.stderr)
        return 2

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )
    print(f"device={device}")

    os.makedirs(OUT_DIR, exist_ok=True)
    train_loader, val_loader, num_classes = get_data_loaders()

    model = make_san_resnet18(num_classes=num_classes, n_conf=N_CONF).to(device)

    if args.train or not os.path.isfile(CHECKPOINT):
        print("training SAN-ResNet-18 on ImageNette2-160...")
        train_model(model, train_loader, device)
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"saved checkpoint {CHECKPOINT}")
    else:
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        print(f"loaded checkpoint {CHECKPOINT}")

    conf, labels, forward_s = extract_confidences(model, val_loader, device)
    print(f"confidences shape={conf.shape} acc={np.mean(conf.argmax(axis=1) == labels):.3f}")

    q_delta = int(args.q_delta * Q15)

    # ResNet-18 prefix MAC LUT for 160x160 (same convention as train_san_imagenette.py)
    lut = np.array([
        29583360,
        176877568,
        471269376,
        1060618240,
        2238156800,
        0, 0, 0
    ], dtype=np.uint64)

    t_pack_0 = time.perf_counter()
    q, beats, n_words = quantize_pack(conf, q_delta)
    t_pack_1 = time.perf_counter()
    print(f"packed n_samples={q.shape[0]} n_words={n_words} bytes={beats.nbytes}")

    # Bit-exact validation: Python reference scan vs the Q array we packed.
    py_idx, py_hist, py_cat, py_flops = host_reference_scan(q, lut, q_delta)

    if args.mock_host:
        stdout = run_mock_host(q, lut, q_delta)
        stderr = ""
    else:
        xclbin_path = args.xclbin if os.path.isabs(args.xclbin) else os.path.join(REPO, args.xclbin)
        header = build_header(q.shape[0], q.shape[1], q_delta, lut)
        stdout, stderr = run_host_e2e(xclbin_path, header, beats.tobytes())

    result = parse_e2e_output(stdout)
    print("--- host output ---")
    print(stdout)
    if stderr:
        print("--- host stderr ---")
        print(stderr)

    # Cross-check Python golden vs host.
    ok = True
    host_hist = result.get("hist", [])
    for b in range(N_CONF + 1):
        if b >= len(host_hist) or int(host_hist[b]) != int(py_hist[b]):
            print(f"MISMATCH hist[{b}]: host={host_hist[b] if b < len(host_hist) else 'MISSING'} "
                  f"py={int(py_hist[b])}")
            ok = False
    if int(result.get("catastrophes", -1)) != py_cat:
        print(f"MISMATCH catastrophes: host={result.get('catastrophes')} py={py_cat}")
        ok = False
    if int(result.get("flops_macs", -1)) != py_flops:
        print(f"MISMATCH flops_macs: host={result.get('flops_macs')} py={py_flops}")
        ok = False

    meta = {
        "dataset": "imagenette2-160",
        "n_samples": int(q.shape[0]),
        "n_conf": int(N_CONF),
        "q_delta": q_delta,
        "forward_ms": round(forward_s * 1e3, 3),
        "pack_ms": round((t_pack_1 - t_pack_0) * 1e3, 3),
        "host_total_ms": float(result.get("total_ms", 0)),
        "host_kernel_ms": float(result.get("kernel_ms", 0)),
        "host_dma_h2d_ms": float(result.get("dma_h2d_ms", 0)),
        "host_dma_d2h_ms": float(result.get("dma_d2h_ms", 0)),
        "host_msamples": float(result.get("Msamples/s", 0)),
        "mock_host": args.mock_host,
        "bit_exact": ok,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    verdict = "SAN_FPGA_ENDTOEND_PASS" if ok else "SAN_FPGA_ENDTOEND_FAIL"
    print(f"{verdict} bit_exact={ok} forward_ms={meta['forward_ms']} "
          f"pack_ms={meta['pack_ms']} host_total_ms={meta['host_total_ms']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
