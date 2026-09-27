#!/usr/bin/env python3
"""Ord 2″ trained non-sedenion target — LSTM on the adding problem (multi-path).

Implements the corrected protocol (probe-corrected-protocol.md + PROBE-RESULT-lstm-adding.md):

  1. Train LSTMCell on the adding problem (long dependency; vanishing gradient operates).
  2. Multi-path state Jacobians ∂[h;c]_{t+1}/∂[h;c]_t along many input sequences.
  3. h→h block alignment curve align(k) vs baseline √(k/H).
  4. Mandatory controls:
       (i)  shape of align(k) — annihilation would peak at small k then drop to bulk;
            flat-high low-rank is NOT annihilation.
       (ii) untrained-init control — if init ≥ trained, signature is architectural.
       (iii) orientation-scramble null (same spectrum, killed alignment).
  5. Discovery/confirmation split with m† frozen on discovery half.

Verdict vocabulary (honest):
  ORD2_TRAINED_NO_SIGNATURE     — controls reject subspace death (expected classical)
  ORD2_TRAINED_SUBSPACE_DEATH   — rare positive; would need human review before claiming
  ORD2_TRAINED_SKIP             — torch/numpy missing
  ORD2_TRAINED_BROKEN           — train/probe failed

Does NOT claim clinical content or D3. Sedenion stacks remain the declared
architectural positive control in rupture_ord2_alignment_contract.py.

Requires: numpy + torch (use repo .venv). Pure-CPU defaults sized for CI.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

# Prefer project venv packages when launched as system python3
_VENV_SITE = os.path.join(
    os.path.dirname(__file__), "..", "..", ".venv", "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages",
)
if os.path.isdir(_VENV_SITE) and _VENV_SITE not in sys.path:
    sys.path.insert(0, os.path.abspath(_VENV_SITE))

try:
    import numpy as np
except Exception:
    print("ORD2_TRAINED_VERDICT ORD2_TRAINED_SKIP")
    print("ORD2_TRAINED_NOTE numpy_missing")
    print("ORD2_TRAINED_CONTRACT_OK")  # skip is not a hard fail for optional deps
    raise SystemExit(0)

try:
    import torch
    import torch.nn as nn
except Exception:
    print("ORD2_TRAINED_VERDICT ORD2_TRAINED_SKIP")
    print("ORD2_TRAINED_NOTE torch_missing")
    print("ORD2_TRAINED_CONTRACT_OK")
    raise SystemExit(0)


# ---- defaults (CI-friendly; override via env) ----
H = int(os.environ.get("ORD2_LSTM_H", "20"))
T = int(os.environ.get("ORD2_LSTM_T", "24"))
STEPS = int(os.environ.get("ORD2_LSTM_STEPS", "600"))
N_SEQ = int(os.environ.get("ORD2_LSTM_NSEQ", "40"))
BATCH = int(os.environ.get("ORD2_LSTM_BATCH", "64"))
SEED = int(os.environ.get("ORD2_LSTM_SEED", "20260725"))
DEV = os.environ.get("ORD2_LSTM_DEVICE", "cpu")


def gen_adding(n: int, T: int, rng: np.random.Generator):
    vals = rng.random((n, T)).astype("float32")
    mark = np.zeros((n, T), dtype="float32")
    for i in range(n):
        mark[i, rng.choice(T, 2, replace=False)] = 1.0
    X = np.stack([vals, mark], -1)
    y = (vals * mark).sum(1).astype("float32")
    return X, y


class Net(nn.Module):
    def __init__(self, H: int):
        super().__init__()
        self.H = H
        self.cell = nn.LSTMCell(2, H)
        self.out = nn.Linear(H, 1)

    def forward(self, X):
        B, T, _ = X.shape
        h = X.new_zeros(B, self.H)
        c = X.new_zeros(B, self.H)
        for t in range(T):
            h, c = self.cell(X[:, t], (h, c))
        return self.out(h).squeeze(-1)


def train(H: int, T: int, steps: int, dev: str, seed: int) -> tuple[Net, float]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = Net(H).to(dev)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    lossf = nn.MSELoss()
    last_mse = float("nan")
    for it in range(steps):
        X, y = gen_adding(BATCH, T, rng)
        X_t = torch.tensor(X, device=dev)
        y_t = torch.tensor(y, device=dev)
        opt.zero_grad()
        L = lossf(net(X_t), y_t)
        L.backward()
        opt.step()
        if it % max(1, steps // 4) == 0 or it == steps - 1:
            Xv, yv = gen_adding(256, T, rng)
            with torch.no_grad():
                last_mse = lossf(
                    net(torch.tensor(Xv, device=dev)), torch.tensor(yv, device=dev)
                ).item()
            chance = float(np.var(yv))
            print(f"TRAIN step={it} test_mse={last_mse:.5f} chance_var={chance:.4f}")
    return net, last_mse


def state_jacobians(net: Net, X_seq: torch.Tensor) -> list[np.ndarray]:
    """Per-step Jacobians of the joint state [h;c]. X_seq: (T,2)."""
    H = net.H
    h = X_seq.new_zeros(H)
    c = X_seq.new_zeros(H)
    Js = []
    for t in range(X_seq.shape[0]):
        st = torch.cat([h, c]).detach().requires_grad_(True)

        def step(s, t=t):
            nh, nc = net.cell(
                X_seq[t].unsqueeze(0),
                (s[:H].unsqueeze(0), s[H:].unsqueeze(0)),
            )
            return torch.cat([nh.squeeze(0), nc.squeeze(0)])

        J = torch.autograd.functional.jacobian(step, st, vectorize=True)
        Js.append(J.detach().cpu().numpy())
        with torch.no_grad():
            o = step(st)
            h, c = o[:H].detach(), o[H:].detach()
    return Js


def bottom_V(A: np.ndarray, k: int) -> np.ndarray:
    return np.linalg.svd(A, compute_uv=True)[2][-k:]


def align_curve(mats: list[np.ndarray], ks: list[int]) -> tuple[np.ndarray, np.ndarray]:
    dim = mats[0].shape[0]
    base = np.sqrt(np.asarray(ks, float) / dim)
    out = []
    for k in ks:
        cs = []
        for l in range(len(mats) - 1):
            V1, V2 = bottom_V(mats[l], k), bottom_V(mats[l + 1], k)
            cos = np.linalg.svd(V1 @ V2.T, compute_uv=False).mean()
            cs.append(float(cos))
        out.append(float(np.mean(cs)))
    return np.asarray(out), base


def orientation_scramble(mats: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    n = mats[0].shape[0]
    Os = [np.linalg.qr(rng.standard_normal((n, n)))[0] for _ in range(len(mats) + 1)]
    return [Os[t + 1] @ mats[t] @ Os[t].T for t in range(len(mats))]


def shoulder_k(align: np.ndarray, base: np.ndarray) -> int:
    ex = align - base
    drop = ex[:-1] - ex[1:]
    return int(np.argmax(drop)) + 1  # 1-indexed k


def probe_net(
    net: Net,
    *,
    n_seq: int,
    T: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    """Multi-path h→h alignment analysis with discovery/confirmation split."""
    H = net.H
    ks = list(range(1, max(2, H // 2 + 1)))  # k=1 .. H/2
    disc = n_seq // 2
    align_true = []
    align_scr = []
    shoulders = []
    for s in range(n_seq):
        X, _ = gen_adding(1, T, np.random.default_rng(10000 + s + seed))
        Xt = torch.tensor(X[0], device=next(net.parameters()).device)
        J = state_jacobians(net, Xt)
        hh = [j[:H, :H] for j in J]
        a, b = align_curve(hh, ks)
        align_true.append(a)
        shoulders.append(shoulder_k(a, b))
        scr = orientation_scramble(hh, np.random.default_rng(20000 + s + seed))
        asr, _ = align_curve(scr, ks)
        align_scr.append(asr)
    align_true = np.asarray(align_true)
    align_scr = np.asarray(align_scr)
    shoulders = np.asarray(shoulders)
    mdag = int(np.median(shoulders[:disc]))
    mdag = max(1, min(mdag, len(ks)))
    # confirmation half at frozen m†
    ci = slice(disc, n_seq)
    tr = align_true[ci, mdag - 1]
    nul = align_scr[ci, mdag - 1]
    d = float((tr.mean() - nul.mean()) / (0.5 * (tr.std() + nul.std()) + 1e-9))
    mean_curve = align_true.mean(0)
    base = np.sqrt(np.asarray(ks, float) / H)
    # shape: annihilation expects small-k peak then drop; flat-high = low rank / architecture
    k_half = max(1, len(ks) // 2)
    high_at_large_k = float(mean_curve[-1]) > float(base[-1] + 0.25)
    drops_to_base = float(mean_curve[-1]) < float(base[-1] + 0.10)
    print(f"PROBE label={label} H={H} n_seq={n_seq} T={T}")
    print(f"  mdag={mdag} mdag_over_H={mdag / H:.3f} shoulder_std={shoulders[:disc].std():.2f}")
    print(
        f"  confirm_align={tr.mean():.4f}±{tr.std():.4f} "
        f"scramble={nul.mean():.4f}±{nul.std():.4f} cohen_d={d:+.3f}"
    )
    print(
        f"  align_k_curve_mean={np.round(mean_curve, 3).tolist()}"
    )
    print(f"  baseline_sqrt={np.round(base, 3).tolist()}")
    print(
        f"  shape_high_at_large_k={high_at_large_k} drops_to_base={drops_to_base}"
    )
    return dict(
        label=label,
        mdag=mdag,
        cohen_d=d,
        mean_curve=mean_curve,
        base=base,
        high_at_large_k=high_at_large_k,
        drops_to_base=drops_to_base,
        confirm_align=float(tr.mean()),
        scramble_align=float(nul.mean()),
    )


def main() -> int:
    if os.environ.get("SOUNIO_SKIP_ORD2_TRAINED", "") in ("1", "true", "yes"):
        print("ORD2_TRAINED_VERDICT ORD2_TRAINED_SKIP")
        print("ORD2_TRAINED_NOTE skipped_by_env")
        print("ORD2_TRAINED_CONTRACT_OK")
        return 0

    print(f"ORD2_TRAINED_CONFIG H={H} T={T} STEPS={STEPS} N_SEQ={N_SEQ} DEV={DEV} SEED={SEED}")
    try:
        # --- train ---
        print("TRAIN_BEGIN")
        net, mse = train(H, T, STEPS, DEV, SEED)
        print(f"TRAIN_END test_mse={mse:.6f}")
        if not (mse < 0.08):  # must beat chance (~0.17) substantially
            print(f"TRAIN_WARN mse={mse:.4f} may be undertrained")

        # --- trained multi-path probe ---
        trained = probe_net(net, n_seq=N_SEQ, T=T, seed=SEED, label="trained")

        # --- untrained-init control ---
        torch.manual_seed(SEED + 99)
        net0 = Net(H).to(DEV)
        init = probe_net(net0, n_seq=max(16, N_SEQ // 2), T=T, seed=SEED + 1, label="init")

        # --- decision rules (PROBE-RESULT-lstm-adding) ---
        # Positive subspace death would need ALL of:
        #   - cohen_d large vs scramble
        #   - mdag < H/2 with drop toward baseline at large k
        #   - trained alignment >> init (not architectural)
        d = trained["cohen_d"]
        mdag = trained["mdag"]
        init_higher = init["confirm_align"] >= trained["confirm_align"] - 0.02
        shape_not_annihilation = trained["high_at_large_k"] and not trained["drops_to_base"]
        scramble_sep = d > 0.8
        small_shoulder = mdag < H / 2

        print(
            f"CONTROL_INIT trained_align={trained['confirm_align']:.4f} "
            f"init_align={init['confirm_align']:.4f} init_ge_trained={init_higher}"
        )
        print(
            f"CONTROL_SHAPE high_large_k={trained['high_at_large_k']} "
            f"drops_to_base={trained['drops_to_base']} "
            f"shape_not_annihilation={shape_not_annihilation}"
        )
        print(
            f"CONTROL_SCRAMBLE cohen_d={d:+.3f} scramble_sep={scramble_sep} "
            f"mdag={mdag} small_shoulder={small_shoulder}"
        )

        # Discovery claim only if all positive criteria hold
        if scramble_sep and small_shoulder and not shape_not_annihilation and not init_higher:
            print("ORD2_TRAINED_VERDICT ORD2_TRAINED_SUBSPACE_DEATH")
            print(
                "ORD2_TRAINED_NOTE unexpected_positive_requires_human_review; "
                "do_not_claim_without_replication"
            )
        else:
            print("ORD2_TRAINED_VERDICT ORD2_TRAINED_NO_SIGNATURE")
            reasons = []
            if init_higher:
                reasons.append("init_control_fails_architectural")
            if shape_not_annihilation:
                reasons.append("align_k_flat_high_low_rank_not_annihilation")
            if not scramble_sep:
                reasons.append("scramble_effect_small")
            if not small_shoulder:
                reasons.append("mdag_not_small")
            print("ORD2_TRAINED_NOTE " + ",".join(reasons) if reasons else "ORD2_TRAINED_NOTE controls_reject")
            print(
                "ORD2_TRAINED_NOTE matches_PROBE_RESULT_lstm_adding_classical_magnitude_not_annihilation"
            )

        print("ORD2_TRAINED_NOTE nonsed_target=LSTM_adding; D3_forbidden; no_clinical_claim")
        print("ORD2_TRAINED_CONTRACT_OK")
        return 0
    except Exception as e:
        print(f"ORD2_TRAINED_VERDICT ORD2_TRAINED_BROKEN")
        print(f"ORD2_TRAINED_NOTE error={type(e).__name__}:{e}")
        print("ORD2_TRAINED_CONTRACT_FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
