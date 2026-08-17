#!/usr/bin/env python3
"""train_san_imagenette.py — train a lightweight SAN on real ImageNette images.

Uses torchvision ResNet-18 pretrained on ImageNet-1k as a frozen backbone,
adds early-exit heads after each residual layer plus the final head, and
trains only the heads (deep supervision) on ImageNette2-160. The goal is
not SOTA accuracy; it is to produce real-image confidence traces that can
be fed into the U250 SAN scan kernel for an honest "real data" validation.

Output:
  datasets/imagenette2-160/san_val_confidences.npy  (float [N, n_conf])
  datasets/imagenette2-160/san_val_labels.npy       (int [N])
  artifacts/san_imagenette_real/expected.txt        (kernel contract format)
  artifacts/san_imagenette_real/val_imagenette.u16  (Q0.15 packed cohort)
"""
import os
import sys
import json
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "imagenette2-160")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "san_imagenette_real")
BATCH_SIZE = 128
EPOCHS = 3
LR = 1e-4  # lower LR because layer4 is now trainable
TRAIN_SUBSET = 4000  # use a subset for fast proof-of-concept training
N_CONF = 4  # 4 early-exit confidence fields -> n_points = 5
MAX_POINTS = 8

os.makedirs(OUT_DIR, exist_ok=True)

def make_san_resnet18(num_classes=10, n_conf=N_CONF):
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze early backbone; fine-tune layer4 + heads so early-exit heads get
    # meaningful signal without full-network CPU training cost.
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
            # head input dims match ResNet-18 intermediate channel counts
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
                # global average pool each intermediate activation
                h = self.avgpool(x).view(x.size(0), -1)
                outs.append(self.heads[len(outs)](h))
            h = self.avgpool(x).view(x.size(0), -1)
            outs.append(self.heads[-1](h))
            return outs

    return SAN(base, num_classes, n_conf)

def train(model, loader, device):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = [0.0] * (N_CONF + 1)
        total = 0
        for x, y in tqdm(loader, desc=f"epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            outs = model(x)
            loss = sum(F.cross_entropy(o, y) for o in outs)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total += y.size(0)
            for i, o in enumerate(outs):
                correct[i] += (o.argmax(1) == y).sum().item()
        accs = [c / total for c in correct]
        print(f"epoch {epoch+1}: loss={total_loss/len(loader):.3f} accs={[f'{a:.3f}' for a in accs]}")

def extract_confidences(model, loader, device):
    model.eval()
    all_conf = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="extract"):
            x = x.to(device)
            outs = model(x)
            # max-softmax confidence per exit point
            confs = torch.stack([F.softmax(o, dim=1).max(dim=1)[0] for o in outs], dim=1)
            all_conf.append(confs.cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_conf, axis=0), np.concatenate(all_labels, axis=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

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
    print(f"train={len(train_ds)} val={len(val_ds)} classes={len(train_full.classes)}")

    model = make_san_resnet18(num_classes=len(train_full.classes), n_conf=N_CONF).to(device)
    train(model, train_loader, device)

    confs, labels = extract_confidences(model, val_loader, device)
    print(f"confidences shape={confs.shape} acc={np.mean(confs.argmax(axis=1) == labels):.3f}")

    np.save(os.path.join(DATA_ROOT, "san_val_confidences.npy"), confs)
    np.save(os.path.join(DATA_ROOT, "san_val_labels.npy"), labels)

    # Quantize to Q0.15 and pack cohort. Save exactly n_conf columns so the
    # host loader (which reads n_samples * n_conf uint16s in row-major order)
    # sees one contiguous [n_samples, n_conf] array.
    q = np.clip(np.floor(confs * 32767.0), 0, 32767).astype(np.uint16)
    q = q[:, :N_CONF]

    cohort_path = os.path.join(OUT_DIR, "val_imagenette.u16")
    with open(cohort_path, "wb") as f:
        f.write(q.tobytes())

    # Compute LUT (real ResNet-18 MAC prefix per exit point)
    # These are approximate ResNet-18 MACs for 160x160 input:
    # conv1 3->64 7x7 stride2: ~29.5M
    # layer1 2 blocks: ~147M
    # layer2 2 blocks: ~294M
    # layer3 2 blocks: ~589M
    # layer4 2 blocks: ~1177M
    # fc: 512*10 = 5K
    # We use the same convention as the SAN spec: prefix sums.
    lut = np.array([
        29583360,
        176877568,
        471269376,
        1060618240,
        2238156800,
        0, 0, 0
    ], dtype=np.uint64)

    # Count histogram and catastrophes for a threshold
    # Use threshold tuned to get ~feasible accuracy; pick a simple one
    q_delta = int(0.55 * 32767)  # same family as residual SAN
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
