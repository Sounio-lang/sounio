#!/usr/bin/env python3
"""Ord 2″ trained non-sedenion target — diagonal S4-style SSM on the adding problem.

Companion to rupture_ord2_trained_lstm_probe.py. Same corrected protocol:

  multi-path state Jacobians ∂h_{t+1}/∂h_t, align(k) curve, orientation-scramble
  null, untrained-init control, discovery/confirmation split.

Model (honest scope):
  Real *diagonal* linear SSM (S4-style structured recurrence), not full complex
  HiPPO-S4 / Mamba. Sufficient as a second non-sedenion structured-SSM family
  where 4/8/4 has no algebraic reason to appear.

    h_{t+1} = diag(λ) h_t + B x_t
    y       = C h_T
    λ       = −softplus(a)   (stable diagonal)

Verdict vocabulary mirrors the LSTM probe:
  ORD2_S4_NO_SIGNATURE | ORD2_S4_SUBSPACE_DEATH | ORD2_S4_SKIP | ORD2_S4_BROKEN

Requires numpy+torch (.venv). Soft-skip without deps.
"""
from __future__ import annotations

import os
import sys
from typing import Any

_VENV_SITE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    ".venv",
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
)
if os.path.isdir(_VENV_SITE) and _VENV_SITE not in sys.path:
    sys.path.insert(0, os.path.abspath(_VENV_SITE))

try:
    import numpy as np
except Exception:
    print("ORD2_S4_VERDICT ORD2_S4_SKIP")
    print("ORD2_S4_NOTE numpy_missing")
    print("ORD2_S4_CONTRACT_OK")
    raise SystemExit(0)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    print("ORD2_S4_VERDICT ORD2_S4_SKIP")
    print("ORD2_S4_NOTE torch_missing")
    print("ORD2_S4_CONTRACT_OK")
    raise SystemExit(0)


H = int(os.environ.get("ORD2_S4_H", os.environ.get("ORD2_LSTM_H", "20")))
T = int(os.environ.get("ORD2_S4_T", os.environ.get("ORD2_LSTM_T", "24")))
STEPS = int(os.environ.get("ORD2_S4_STEPS", os.environ.get("ORD2_LSTM_STEPS", "600")))
N_SEQ = int(os.environ.get("ORD2_S4_NSEQ", os.environ.get("ORD2_LSTM_NSEQ", "40")))
BATCH = int(os.environ.get("ORD2_S4_BATCH", "64"))
SEED = int(os.environ.get("ORD2_S4_SEED", "20260725"))
DEV = os.environ.get("ORD2_S4_DEVICE", "cpu")


def gen_adding(n: int, T: int, rng: np.random.Generator):
    vals = rng.random((n, T)).astype("float32")
    mark = np.zeros((n, T), dtype="float32")
    for i in range(n):
        mark[i, rng.choice(T, 2, replace=False)] = 1.0
    X = np.stack([vals, mark], -1)
    y = (vals * mark).sum(1).astype("float32")
    return X, y


class DiagonalS4(nn.Module):
    """Minimal S4-style diagonal real SSM (not full complex HiPPO-S4)."""

    def __init__(self, H: int, d_in: int = 2):
        super().__init__()
        self.H = H
        # log-time-scale parameters → λ = −softplus(a) ∈ (−∞, 0)
        self.a = nn.Parameter(torch.randn(H) * 0.1)
        self.B = nn.Parameter(torch.randn(H, d_in) * 0.1)
        self.C = nn.Parameter(torch.randn(1, H) * 0.1)
        self.D = nn.Parameter(torch.zeros(1, d_in))

    def lambdas(self) -> torch.Tensor:
        return -F.softplus(self.a)

    def step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # h: (B,H) or (H,); x: (B,d) or (d,)
        lam = self.lambdas()
        return lam * h + x @ self.B.T

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B,T,2)
        Bsz, T, _ = X.shape
        h = X.new_zeros(Bsz, self.H)
        for t in range(T):
            h = self.step(h, X[:, t])
        y = (h @ self.C.T).squeeze(-1) + (X[:, -1] * self.D).sum(-1)
        return y


def train(H: int, T: int, steps: int, dev: str, seed: int) -> tuple[DiagonalS4, float]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = DiagonalS4(H).to(dev)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    lossf = nn.MSELoss()
    last_mse = float("nan")
    for it in range(steps):
        X, y = gen_adding(BATCH, T, rng)
        Xt = torch.tensor(X, device=dev)
        yt = torch.tensor(y, device=dev)
        opt.zero_grad()
        L = lossf(net(Xt), yt)
        L.backward()
        opt.step()
        if it % max(1, steps // 4) == 0 or it == steps - 1:
            Xv, yv = gen_adding(256, T, rng)
            with torch.no_grad():
                last_mse = lossf(
                    net(torch.tensor(Xv, device=dev)), torch.tensor(yv, device=dev)
                ).item()
            print(
                f"TRAIN step={it} test_mse={last_mse:.5f} "
                f"chance_var={float(np.var(yv)):.4f}"
            )
    return net, last_mse


def state_jacobians(net: DiagonalS4, X_seq: torch.Tensor) -> list[np.ndarray]:
    """Per-step ∂h_{t+1}/∂h_t along the realized input sequence. X_seq: (T,2)."""
    H = net.H
    h = X_seq.new_zeros(H)
    Js = []
    for t in range(X_seq.shape[0]):
        ht = h.detach().requires_grad_(True)

        def step(hh, t=t):
            return net.step(hh, X_seq[t])

        J = torch.autograd.functional.jacobian(step, ht, vectorize=True)
        Js.append(J.detach().cpu().numpy())
        with torch.no_grad():
            h = step(ht).detach()
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
    return int(np.argmax(drop)) + 1


def probe_net(
    net: DiagonalS4,
    *,
    n_seq: int,
    T: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    H = net.H
    ks = list(range(1, max(2, H // 2 + 1)))
    disc = n_seq // 2
    align_true, align_scr, shoulders = [], [], []
    for s in range(n_seq):
        X, _ = gen_adding(1, T, np.random.default_rng(30000 + s + seed))
        Xt = torch.tensor(X[0], device=next(net.parameters()).device)
        J = state_jacobians(net, Xt)
        a, b = align_curve(J, ks)
        align_true.append(a)
        shoulders.append(shoulder_k(a, b))
        scr = orientation_scramble(J, np.random.default_rng(40000 + s + seed))
        asr, _ = align_curve(scr, ks)
        align_scr.append(asr)
    align_true = np.asarray(align_true)
    align_scr = np.asarray(align_scr)
    shoulders = np.asarray(shoulders)
    mdag = int(np.median(shoulders[:disc]))
    mdag = max(1, min(mdag, len(ks)))
    ci = slice(disc, n_seq)
    tr = align_true[ci, mdag - 1]
    nul = align_scr[ci, mdag - 1]
    d = float((tr.mean() - nul.mean()) / (0.5 * (tr.std() + nul.std()) + 1e-9))
    mean_curve = align_true.mean(0)
    base = np.sqrt(np.asarray(ks, float) / H)
    high_at_large_k = float(mean_curve[-1]) > float(base[-1] + 0.25)
    drops_to_base = float(mean_curve[-1]) < float(base[-1] + 0.10)
    print(f"PROBE label={label} H={H} n_seq={n_seq} T={T}")
    print(
        f"  mdag={mdag} mdag_over_H={mdag / H:.3f} "
        f"shoulder_std={shoulders[:disc].std():.2f}"
    )
    print(
        f"  confirm_align={tr.mean():.4f}±{tr.std():.4f} "
        f"scramble={nul.mean():.4f}±{nul.std():.4f} cohen_d={d:+.3f}"
    )
    print(f"  align_k_curve_mean={np.round(mean_curve, 3).tolist()}")
    print(f"  baseline_sqrt={np.round(base, 3).tolist()}")
    print(
        f"  shape_high_at_large_k={high_at_large_k} drops_to_base={drops_to_base}"
    )
    return dict(
        mdag=mdag,
        cohen_d=d,
        confirm_align=float(tr.mean()),
        scramble_align=float(nul.mean()),
        high_at_large_k=high_at_large_k,
        drops_to_base=drops_to_base,
        mean_curve=mean_curve,
    )


def main() -> int:
    if os.environ.get("SOUNIO_SKIP_ORD2_S4", os.environ.get("SOUNIO_SKIP_ORD2_TRAINED", "")) in (
        "1",
        "true",
        "yes",
    ):
        print("ORD2_S4_VERDICT ORD2_S4_SKIP")
        print("ORD2_S4_NOTE skipped_by_env")
        print("ORD2_S4_CONTRACT_OK")
        return 0

    print(
        f"ORD2_S4_CONFIG H={H} T={T} STEPS={STEPS} N_SEQ={N_SEQ} "
        f"DEV={DEV} SEED={SEED} model=DiagonalS4"
    )
    try:
        print("TRAIN_BEGIN")
        net, mse = train(H, T, STEPS, DEV, SEED)
        print(f"TRAIN_END test_mse={mse:.6f}")
        if mse >= 0.12:
            print(f"TRAIN_WARN mse={mse:.4f} may_be_undertrained")

        trained = probe_net(net, n_seq=N_SEQ, T=T, seed=SEED, label="trained_s4")
        torch.manual_seed(SEED + 99)
        net0 = DiagonalS4(H).to(DEV)
        init = probe_net(
            net0, n_seq=max(16, N_SEQ // 2), T=T, seed=SEED + 1, label="init_s4"
        )

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

        # Diagonal S4 has A=diag(λ) so ∂h'/∂h is diagonal-dominated — expect
        # strong architectural alignment even without training.
        if scramble_sep and small_shoulder and not shape_not_annihilation and not init_higher:
            print("ORD2_S4_VERDICT ORD2_S4_SUBSPACE_DEATH")
            print(
                "ORD2_S4_NOTE unexpected_positive_requires_human_review; "
                "do_not_claim_without_replication"
            )
        else:
            print("ORD2_S4_VERDICT ORD2_S4_NO_SIGNATURE")
            reasons = []
            if init_higher:
                reasons.append("init_control_fails_architectural")
            if shape_not_annihilation:
                reasons.append("align_k_flat_high_or_diagonal_structure")
            if not scramble_sep:
                reasons.append("scramble_effect_small")
            if not small_shoulder:
                reasons.append("mdag_not_small")
            print(
                "ORD2_S4_NOTE "
                + (
                    ",".join(reasons)
                    if reasons
                    else "controls_reject"
                )
            )
            print(
                "ORD2_S4_NOTE diagonal_S4_is_structured_SSM_family_not_full_complex_HiPPO_S4"
            )

        print(
            "ORD2_S4_NOTE nonsed_target=DiagonalS4_adding; D3_forbidden; no_clinical_claim"
        )
        print("ORD2_S4_CONTRACT_OK")
        return 0
    except Exception as e:
        print("ORD2_S4_VERDICT ORD2_S4_BROKEN")
        print(f"ORD2_S4_NOTE error={type(e).__name__}:{e}")
        print("ORD2_S4_CONTRACT_FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
