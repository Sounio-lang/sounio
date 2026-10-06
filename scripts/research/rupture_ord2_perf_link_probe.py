#!/usr/bin/env python3
"""Ord 2″ protocol §5 — link alignment/spectrum shape to long-sequence performance.

After multi-path alignment is measured, the corrected protocol requires:

  "does the alignment/gap predict a loss plateau, long-sequence degradation,
   or specific unlearned examples? Without that link it is a spectrum shape,
   not a diagnostic."

This probe:
  1. Trains an LSTM on the adding problem at T_train.
  2. For many held-out sequences at T_train: records h→h mean alignment and per-seq MSE.
  3. Evaluates length generalisation: MSE at T ∈ {T_train, 1.5 T, 2 T, 3 T}.
  4. Reports Pearson correlation (alignment vs per-seq error) and whether long-T
     degradation is explained by high alignment (expected: NO under NO_SIGNATURE).

Verdict vocabulary:
  ORD2_PERF_NO_LINK          — long-T fails or not; alignment does not predict error
  ORD2_PERF_LINK_PRESENT     — |r| large and same-direction as annihilation hypothesis
  ORD2_PERF_SKIP / BROKEN

Honest prior (from ORD2_TRAINED_NO_SIGNATURE): alignment is architectural low-rank;
expect NO_LINK. A positive link would be news and needs human review.

Requires numpy+torch (.venv). Soft-skip without deps.
"""
from __future__ import annotations

import math
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
    print("ORD2_PERF_VERDICT ORD2_PERF_SKIP")
    print("ORD2_PERF_NOTE numpy_missing")
    print("ORD2_PERF_CONTRACT_OK")
    raise SystemExit(0)

try:
    import torch
    import torch.nn as nn
except Exception:
    print("ORD2_PERF_VERDICT ORD2_PERF_SKIP")
    print("ORD2_PERF_NOTE torch_missing")
    print("ORD2_PERF_CONTRACT_OK")
    raise SystemExit(0)


H = int(os.environ.get("ORD2_PERF_H", "24"))
T_TRAIN = int(os.environ.get("ORD2_PERF_T", "24"))
STEPS = int(os.environ.get("ORD2_PERF_STEPS", "800"))
N_PATH = int(os.environ.get("ORD2_PERF_NPATH", "48"))
BATCH = int(os.environ.get("ORD2_PERF_BATCH", "64"))
SEED = int(os.environ.get("ORD2_PERF_SEED", "20260725"))
DEV = os.environ.get("ORD2_PERF_DEVICE", "cpu")
K_ALIGN = int(os.environ.get("ORD2_PERF_K", "4"))


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
    last = float("nan")
    for it in range(steps):
        X, y = gen_adding(BATCH, T, rng)
        opt.zero_grad()
        L = lossf(net(torch.tensor(X, device=dev)), torch.tensor(y, device=dev))
        L.backward()
        opt.step()
        if it == steps - 1 or it % max(1, steps // 4) == 0:
            Xv, yv = gen_adding(256, T, rng)
            with torch.no_grad():
                last = lossf(
                    net(torch.tensor(Xv, device=dev)), torch.tensor(yv, device=dev)
                ).item()
            print(f"TRAIN step={it} test_mse={last:.5f}")
    return net, last


def state_jacobians_hh(net: Net, X_seq: torch.Tensor) -> list[np.ndarray]:
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
        Jnp = J.detach().cpu().numpy()
        Js.append(Jnp[:H, :H])
        with torch.no_grad():
            o = step(st)
            h, c = o[:H].detach(), o[H:].detach()
    return Js


def mean_align_k(mats: list[np.ndarray], k: int) -> float:
    cs = []
    for l in range(len(mats) - 1):
        V1 = np.linalg.svd(mats[l], compute_uv=True)[2][-k:]
        V2 = np.linalg.svd(mats[l + 1], compute_uv=True)[2][-k:]
        cs.append(float(np.linalg.svd(V1 @ V2.T, compute_uv=False).mean()))
    return float(np.mean(cs)) if cs else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    if os.environ.get("SOUNIO_SKIP_ORD2_PERF", os.environ.get("SOUNIO_SKIP_ORD2_TRAINED", "")) in (
        "1",
        "true",
        "yes",
    ):
        print("ORD2_PERF_VERDICT ORD2_PERF_SKIP")
        print("ORD2_PERF_NOTE skipped_by_env")
        print("ORD2_PERF_CONTRACT_OK")
        return 0

    print(
        f"ORD2_PERF_CONFIG H={H} T_train={T_TRAIN} STEPS={STEPS} "
        f"N_PATH={N_PATH} K={K_ALIGN} DEV={DEV} SEED={SEED}"
    )
    try:
        print("TRAIN_BEGIN")
        net, mse = train(H, T_TRAIN, STEPS, DEV, SEED)
        print(f"TRAIN_END test_mse_at_Ttrain={mse:.6f}")

        # --- per-path alignment vs error at T_train ---
        aligns = []
        errs = []
        for s in range(N_PATH):
            rng = np.random.default_rng(50000 + s)
            X, y = gen_adding(1, T_TRAIN, rng)
            Xt = torch.tensor(X[0], device=DEV)
            with torch.no_grad():
                pred = net(torch.tensor(X, device=DEV)).item()
            err = float((pred - float(y[0])) ** 2)
            Js = state_jacobians_hh(net, Xt)
            al = mean_align_k(Js, min(K_ALIGN, H // 2))
            aligns.append(al)
            errs.append(err)
        aligns_a = np.asarray(aligns)
        errs_a = np.asarray(errs)
        r_path = pearson(aligns_a, errs_a)
        print(
            f"PATH_LINK n={N_PATH} align_mean={aligns_a.mean():.4f}±{aligns_a.std():.4f} "
            f"err_mean={errs_a.mean():.5f} pearson_align_vs_err={r_path:+.4f}"
        )

        # --- length generalisation curve ---
        T_list = sorted(
            {
                T_TRAIN,
                max(T_TRAIN + 4, int(round(1.5 * T_TRAIN))),
                2 * T_TRAIN,
                3 * T_TRAIN,
            }
        )
        mse_by_T = {}
        for Tv in T_list:
            rng = np.random.default_rng(SEED + 7 + Tv)
            Xv, yv = gen_adding(256, Tv, rng)
            with torch.no_grad():
                pred = net(torch.tensor(Xv, device=DEV)).cpu().numpy()
            mse_by_T[Tv] = float(np.mean((pred - yv) ** 2))
            print(f"LENGTH_GEN T={Tv} mse={mse_by_T[Tv]:.5f}")

        # degradation ratio: long / train
        mse_train = mse_by_T[T_TRAIN]
        mse_long = mse_by_T[max(T_list)]
        deg = mse_long / max(mse_train, 1e-9)
        print(
            f"LENGTH_DEGRADATION mse_Ttrain={mse_train:.5f} "
            f"mse_Tlong={mse_long:.5f} ratio={deg:.3f}"
        )

        # Split paths by high/low alignment; compare mean error
        med = float(np.median(aligns_a))
        hi = errs_a[aligns_a >= med]
        lo = errs_a[aligns_a < med]
        hi_m = float(hi.mean()) if len(hi) else float("nan")
        lo_m = float(lo.mean()) if len(lo) else float("nan")
        print(
            f"ALIGN_SPLIT median_align={med:.4f} "
            f"err_high_align={hi_m:.5f} err_low_align={lo_m:.5f}"
        )

        # Decision: LINK if |r|>=0.35 AND high-align errors clearly worse
        # (annihilation hypothesis: more alignment of dead subspaces → worse long path)
        link_r = abs(r_path) >= 0.35
        link_split = hi_m > lo_m * 1.15 if (lo_m > 0 and not math.isnan(hi_m)) else False
        long_fails = deg > 2.0  # informative context, not the link itself

        print(f"LINK_CRITERIA |r|>0.35={link_r} high_align_worse_err={link_split}")
        print(f"CONTEXT long_T_degrades={long_fails} ratio={deg:.3f}")

        if link_r and link_split:
            print("ORD2_PERF_VERDICT ORD2_PERF_LINK_PRESENT")
            print(
                "ORD2_PERF_NOTE positive_perf_link_requires_human_review; "
                "do_not_claim_diagnostic_without_replication"
            )
        else:
            print("ORD2_PERF_VERDICT ORD2_PERF_NO_LINK")
            reasons = []
            if not link_r:
                reasons.append(f"weak_pearson_r={r_path:+.3f}")
            if not link_split:
                reasons.append("high_align_not_worse_error")
            if long_fails:
                reasons.append("long_T_degrades_but_unlinked_to_align")
            print("ORD2_PERF_NOTE " + ",".join(reasons))
            print(
                "ORD2_PERF_NOTE spectrum_shape_without_perf_link_is_not_a_diagnostic"
            )

        print(
            "ORD2_PERF_NOTE protocol_s5; nonsed_LSTM; D3_forbidden; no_clinical_claim"
        )
        print("ORD2_PERF_CONTRACT_OK")
        return 0
    except Exception as e:
        print("ORD2_PERF_VERDICT ORD2_PERF_BROKEN")
        print(f"ORD2_PERF_NOTE error={type(e).__name__}:{e}")
        print("ORD2_PERF_CONTRACT_FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
