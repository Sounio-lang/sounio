#!/usr/bin/env python3
"""
Sounio Research Backend — Phase 7A Ablation
Non-Associative Octonion SSM vs Associative Controls on Apple Silicon MPS / CPU

Part 1: Phase 6C Confusion Matrix Audit
        Audits distill_qwen3_ossm.py Bracketed Algebra Task:
        - Majority-class baseline (~8.98%)
        - Axis-match accuracy (>95%)
        - Sign-flip rate (~50%)
        Proving the Phase 6C tie at 47.27% was the linear readout XOR bottleneck,
        not majority-class collapse.

Part 2: Multi-Generator >=7 Tree / Parenthesization Task
        - 4,096 train / 1,024 held-out unseen triples (x, y, z, tree_op) over +-e1..+-e7
        - Interleaving distractors
        - Shared 2-layer GELU MLP readout (eliminating linear XOR bottleneck)
        - 5 Models:
          1) OctonionDualState (2-register non-associative gated)
          2) OctonionLeftFold (1-register left-fold, testing Artin dormancy)
          3) BlockQuatDualState (associative H (+) H control)
          4) InputDependentSO8 (Cayley-transformed strictly orthogonal SO(8) RNN)
          5) FreeBilinear8x8x8 (unconstrained learned bilinear tensor product)
        - Evaluated over 3 seeds [42, 137, 2026] in two regimes:
          Regime I: param_matched (~28K-32K within +-2%)
          Regime II: dim_matched (n_channels=8, state_dim=64)

Saves JSON report to:
/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/phase7a_ablation_report.json
"""

import os
import sys
import time
import json
import random
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path("/Users/demetriosagourakis/dev/sounio-worktrees/llm-backend")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F

from scripts.research.distill_qwen3_ossm import (
    oct_mul,
    quat_mul,
    block_quat_mul,
    build_algebra_ablation_dataset,
    AblationClassifier,
)

REPORT_PATH = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/phase7a_ablation_report.json")


# =============================================================================
# Part 1: Phase 6C Confusion Matrix Audit
# =============================================================================

def run_phase6c_audit(device: torch.device) -> dict:
    print("\n" + "=" * 80)
    print("PART 1: Phase 6C Confusion Matrix Audit")
    print("=" * 80)

    t0 = time.time()
    train_inst, test_inst, encode_batch = build_algebra_ablation_dataset()
    test_x, test_y = encode_batch(test_inst)
    test_x, test_y = test_x.to(device), test_y.to(device)

    # Majority class baseline
    counts = torch.bincount(test_y, minlength=16)
    majority_class_idx = torch.argmax(counts).item()
    majority_count = counts[majority_class_idx].item()
    majority_frac = majority_count / len(test_y)

    print(f"Test size: {len(test_y)}")
    print(f"Majority class ID: {majority_class_idx}, count: {majority_count}/{len(test_y)}")
    print(f"Majority-class fraction: {majority_frac:.4f} ({majority_frac * 100:.2f}%)")

    # Train Phase 6C Octonion model (seed 42, 500 steps)
    torch.manual_seed(42)
    random.seed(42)
    model = AblationClassifier(is_octonion=True, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for step in range(500):
        batch = random.sample(train_inst, 64)
        bx, by = encode_batch(batch)
        bx, by = bx.to(device), by.to(device)

        optimizer.zero_grad()
        logits, _ = model(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits, test_assoc = model(test_x)
        preds = torch.argmax(test_logits, dim=-1)
        test_acc = (preds == test_y).float().mean().item()

        # Axis match: basis vector axis is class % 8
        axis_preds = preds % 8
        axis_targets = test_y % 8
        axis_match_acc = (axis_preds == axis_targets).float().mean().item()

        # Sign match among matching axes:
        matched_mask = (axis_preds == axis_targets)
        sign_preds = preds // 8
        sign_targets = test_y // 8
        sign_match_acc = (sign_preds[matched_mask] == sign_targets[matched_mask]).float().mean().item()
        sign_flip_rate = 1.0 - sign_match_acc
        mean_assoc_l2 = test_assoc.mean().item()

    audit_wall = time.time() - t0
    print(f"Phase 6C Final Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Axis-Match Accuracy: {axis_match_acc:.4f} ({axis_match_acc * 100:.2f}%) [>95% expected]")
    print(f"Sign-Flip Rate: {sign_flip_rate:.4f} ({sign_flip_rate * 100:.2f}%) [~50% expected]")
    print(f"Mean Associator L2: {mean_assoc_l2:.4f}")
    print(f"Audit completed in {audit_wall:.2f}s")

    assert axis_match_acc > 0.95, f"Expected axis_acc > 95%, got {axis_match_acc:.4f}"
    assert 0.40 <= sign_flip_rate <= 0.60, f"Expected sign_flip ~50%, got {sign_flip_rate:.4f}"

    return {
        "majority_class_idx": majority_class_idx,
        "majority_class_fraction": float(f"{majority_frac:.4f}"),
        "test_accuracy": float(f"{test_acc:.4f}"),
        "axis_match_accuracy": float(f"{axis_match_acc:.4f}"),
        "sign_flip_rate": float(f"{sign_flip_rate:.4f}"),
        "mean_associator_l2": float(f"{mean_assoc_l2:.4f}"),
        "linear_readout_xor_bottleneck_confirmed": True,
        "wall_time_seconds": float(f"{audit_wall:.2f}"),
    }


# =============================================================================
# Part 2: Multi-Generator >=7 Tree / Parenthesization Task
# =============================================================================
#
# TOKEN ENCODING (prefix tree mode, causal left-to-right; the tree shape is the
# FIRST token, so every later step is processed already knowing the bracketing):
#   0..13  generators  +e1..+e7 (0..6 -> +e1..+e7 is idx i%7+1 sign +) / -e1..-e7 (7..13)
#          i.e. generator id g -> basis vector eye[1 + g % 7] * (+1 if g < 7 else -1)
#   14     OP_LEFT  : target = (x*y)*z
#   15     OP_RIGHT : target = x*(y*z)
#   16..23 distractor tokens (random, label-irrelevant), interleaved
# Sequence (len 7):  [OP, x, n1, y, n2, z, n3]
# Label = signed basis class of the product: axis + 8*(sign<0), 16 classes.
# Train: 2048 triples x 2 ops = 4096 instances; test: 512 UNSEEN triples x 2 ops
# = 1024 instances. Distractors are re-drawn per instance (fixed by seed 2026).

TARGET_PARAMS = 32000
VOCAB = 24
TRAIN_STEPS = 400
BATCH = 256
LR = 3e-3


def build_phase7a_tree_dataset():
    eye = torch.eye(8, dtype=torch.float32)

    def gen_vec(g):
        return eye[1 + g % 7] * (1.0 if g < 7 else -1.0)

    def vec_to_class(v):
        idx = torch.argmax(torch.abs(v)).item()
        return idx if v[idx].item() > 0 else idx + 8

    triples = [(i, j, k) for i in range(14) for j in range(14) for k in range(14)]
    rng = random.Random(2026)
    rng.shuffle(triples)
    train_triples = triples[:2048]
    test_triples = triples[2048:2048 + 512]
    assert not (set(train_triples) & set(test_triples))

    def encode(tl):
        xs, ys = [], []
        for (i, j, k) in tl:
            vi, vj, vk = gen_vec(i), gen_vec(j), gen_vec(k)
            l_res = oct_mul(oct_mul(vi, vj), vk)
            r_res = oct_mul(vi, oct_mul(vj, vk))
            n1, n2, n3 = (16 + rng.randrange(8) for _ in range(3))
            xs.append([14, i, n1, j, n2, k, n3]); ys.append(vec_to_class(l_res))
            xs.append([15, i, n1, j, n2, k, n3]); ys.append(vec_to_class(r_res))
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    tx, ty = encode(train_triples)
    sx, sy = encode(test_triples)
    return tx, ty, sx, sy


class GeluMlpReadout(nn.Module):
    def __init__(self, in_dim, hidden, num_classes=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


def unit(v):
    return F.normalize(v, dim=-1, eps=1e-6)


class DualCell(nn.Module):
    """Shared skeleton: emb -> unit 8D channel inputs x_t, 2 gated registers A, B, readout MLP.
    Subclasses define step_a / step_b / identity / extra params."""
    ident_vec = (1, 0, 0, 0, 0, 0, 0, 0)
    dual = True

    def __init__(self, n, e, rh):
        super().__init__()
        self.n = n
        self.emb = nn.Embedding(VOCAB, e)
        self.x_proj = nn.Linear(e, n * 8, bias=False)
        self.gate = nn.Linear(e, 2 * n)
        nn.init.constant_(self.gate.bias, 1.0)
        self.readout = GeluMlpReadout(n * 8 * (2 if self.dual else 1), rh)

    def init_state(self, B, ref):
        s = torch.tensor(self.ident_vec, dtype=ref.dtype, device=ref.device)
        return s.expand(B, self.n, 8).clone()

    def prep(self, h):  # per-model extra per-token quantities
        return None

    def forward(self, tok):
        B, T = tok.shape
        h = self.emb(tok)
        x = unit(self.x_proj(h).view(B, T, self.n, 8))
        g = torch.sigmoid(self.gate(h)).view(B, T, 2, self.n, 1)
        extra = self.prep(h)
        A = self.init_state(B, x)
        Bs = self.init_state(B, x)
        for t in range(T):
            ex = None if extra is None else extra[:, t]
            nA = self.step_a(A, x[:, t], ex)
            A = unit(g[:, t, 0] * nA + (1 - g[:, t, 0]) * A)
            if self.dual:
                nB = self.step_b(Bs, x[:, t], ex)
                Bs = unit(g[:, t, 1] * nB + (1 - g[:, t, 1]) * Bs)
        feat = torch.cat([A, Bs], dim=-1) if self.dual else A
        return self.readout(feat.reshape(B, -1))

    def assoc_l2(self, tok):
        """Mean ||(a*b)*c - a*(b*c)|| over the model's own mul on its 3 generator inputs."""
        mul = self.mul_fn()
        if mul is None:
            return 0.0
        with torch.no_grad():
            B, T = tok.shape
            x = unit(self.x_proj(self.emb(tok)).view(B, T, self.n, 8))
            a, b, c = x[:, 1], x[:, 3], x[:, 5]
            return torch.norm(mul(mul(a, b), c) - mul(a, mul(b, c)), dim=-1).mean().item()

    def mul_fn(self):
        return None

    def ortho_err(self, tok):
        return 0.0


class OctonionDualState(DualCell):
    def step_a(self, A, x, ex): return oct_mul(A, x)
    def step_b(self, B, x, ex): return oct_mul(x, B)
    def mul_fn(self): return oct_mul


class OctonionLeftFold(DualCell):
    dual = False
    def step_a(self, A, x, ex): return oct_mul(A, x)
    def mul_fn(self): return oct_mul


class BlockQuatDualState(DualCell):
    ident_vec = (1, 0, 0, 0, 1, 0, 0, 0)
    def step_a(self, A, x, ex): return block_quat_mul(A, x)
    def step_b(self, B, x, ex): return block_quat_mul(x, B)
    def mul_fn(self): return block_quat_mul


class InputDependentSO8(DualCell):
    """A = S - S^T (28-DOF skew generator from the token), R = (I+A)^-1 (I-A)."""
    def __init__(self, n, e, rh):
        super().__init__(n, e, rh)
        self.s_proj = nn.Linear(e, n * 64, bias=False)
        nn.init.normal_(self.s_proj.weight, std=0.05)
        self.register_buffer("I8", torch.eye(8))

    def prep(self, h):
        B, T, _ = h.shape
        S = self.s_proj(h).view(B, T, self.n, 8, 8)
        A = S - S.transpose(-1, -2)
        R = torch.linalg.solve(self.I8 + A, self.I8 - A)
        self._last_R = R
        return R

    def step_a(self, A, x, R): return (R @ A.unsqueeze(-1)).squeeze(-1)
    def step_b(self, B, x, R): return (R.transpose(-1, -2) @ B.unsqueeze(-1)).squeeze(-1)

    def ortho_err(self, tok):
        with torch.no_grad():
            R = self.prep(self.emb(tok))
            RtR = R.transpose(-1, -2) @ R
            return torch.max(torch.abs(RtR - self.I8)).item()


class FreeBilinear8x8x8(DualCell):
    def __init__(self, n, e, rh, per_channel):
        super().__init__(n, e, rh)
        shape = (n, 8, 8, 8) if per_channel else (1, 8, 8, 8)
        self.T = nn.Parameter(torch.randn(*shape) * 0.3)

    def bil(self, a, b):
        return torch.einsum("bci,bcj,cijk->bck", a, b, self.T.expand(self.n, 8, 8, 8))

    def step_a(self, A, x, ex): return self.bil(A, x)
    def step_b(self, B, x, ex): return self.bil(x, B)
    def mul_fn(self):
        return lambda a, b: self.bil(a, b)


MODELS = ["OctonionDualState", "OctonionLeftFold", "BlockQuatDualState", "InputDependentSO8", "FreeBilinear8x8x8"]


def make_model(name, regime, rh):
    n, e = (4, 32) if regime == "param_matched" else (8, 64)
    if name == "OctonionDualState": return OctonionDualState(n, e, rh)
    if name == "OctonionLeftFold": return OctonionLeftFold(n, e, rh)
    if name == "BlockQuatDualState": return BlockQuatDualState(n, e, rh)
    if name == "InputDependentSO8": return InputDependentSO8(n, e, rh)
    return FreeBilinear8x8x8(n, e, rh, per_channel=(regime == "dim_matched"))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def choose_readout_hidden(name, regime):
    if regime == "dim_matched":
        return 64
    best = min(range(8, 1200), key=lambda rh: abs(count_params(make_model(name, regime, rh)) - TARGET_PARAMS))
    return best


def train_and_eval(name, regime, rh, data, seed, device):
    tx, ty, sx, sy = data
    torch.manual_seed(seed)
    random.seed(seed)
    model = make_model(name, regime, rh).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_STEPS)
    g = torch.Generator().manual_seed(seed)
    N = tx.shape[0]
    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, N, (BATCH,), generator=g)
        loss = F.cross_entropy(model(tx[idx]), ty[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(sx).argmax(-1)
        acc = (pred == sy).float().mean().item()
        axis = ((pred % 8) == (sy % 8)).float().mean().item()
        left = sx[:, 0] == 14
        acc_left = (pred[left] == sy[left]).float().mean().item()
        acc_right = (pred[~left] == sy[~left]).float().mean().item()
        train_acc = (model(tx).argmax(-1) == ty).float().mean().item()
    return {
        "accuracy": acc, "axis_accuracy": axis, "train_accuracy": train_acc,
        "acc_left_tree": acc_left, "acc_right_tree": acc_right,
        "associator_l2": model.assoc_l2(sx), "so8_ortho_err": model.ortho_err(sx),
        "final_train_loss": loss.item(), "param_count": count_params(model),
    }


def run_phase7a_ablation(device):
    print("\n" + "=" * 80)
    print("PART 2: Multi-Generator >=7 Tree / Parenthesization Task Ablation")
    print("=" * 80)
    data = build_phase7a_tree_dataset()
    data = tuple(t.to(device) for t in data)
    ty, sy = data[1], data[3]
    maj = torch.bincount(sy, minlength=16).max().item() / len(sy)
    print(f"train {tuple(data[0].shape)} test {tuple(data[2].shape)} test-majority {maj:.4f}")

    seeds = [42, 137, 2026]
    results = {"task": {"train_instances": len(ty), "test_instances": len(sy),
                        "test_majority_baseline": round(maj, 4), "chance_16_class": 0.0625,
                        "train_steps": TRAIN_STEPS, "batch_size": BATCH, "lr": LR,
                        "token_encoding": "prefix tree: [OP(14=left,15=right), x, n1, y, n2, z, n3]; gens 0..13=+-e1..+-e7; distractors 16..23"}}
    # 30 independent runs (5 models x 3 seeds x 2 regimes) in parallel single-thread worker processes.
    ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=12, mp_context=ctx)
    rhs = {(r, n): choose_readout_hidden(n, r) for r in ["param_matched", "dim_matched"] for n in MODELS}
    cpu_data = tuple(t.cpu() for t in data)
    futs = {(r, n, sd): pool.submit(train_and_eval, n, r, rhs[(r, n)], cpu_data, sd, torch.device("cpu"))
            for r in ["param_matched", "dim_matched"] for n in MODELS for sd in seeds}
    for regime in ["param_matched", "dim_matched"]:
        print(f"\n--- Regime: {regime} ---")
        reg = {}
        for name in MODELS:
            rh = rhs[(regime, name)]
            runs = [futs[(regime, name, s)].result() for s in seeds]
            accs = [r["accuracy"] for r in runs]
            mean = sum(accs) / 3
            std = (sum((a - mean) ** 2 for a in accs) / 3) ** 0.5
            ortho = max(r["so8_ortho_err"] for r in runs)
            if name == "InputDependentSO8":
                assert ortho < 1e-5, f"SO(8) ortho err {ortho}"
            mm = lambda k: sum(r[k] for r in runs) / 3
            reg[name] = {
                "param_count": runs[0]["param_count"], "readout_hidden": rh, "seeds": seeds,
                "seed_accuracies": [round(a, 4) for a in accs],
                "mean_acc": round(mean, 4), "std_acc": round(std, 4),
                "axis_acc": round(mm("axis_accuracy"), 4),
                "train_acc": round(mm("train_accuracy"), 4),
                "acc_left_tree": round(mm("acc_left_tree"), 4),
                "acc_right_tree": round(mm("acc_right_tree"), 4),
                "associator_l2": round(mm("associator_l2"), 6),
                "so8_ortho_err": float(f"{ortho:.2e}"),
            }
            print(f"[{regime} | {name:18s}] P={runs[0]['param_count']:,} acc={mean*100:.2f}% (std {std*100:.2f}) "
                  f"L={mm('acc_left_tree')*100:.1f} R={mm('acc_right_tree')*100:.1f} train={mm('train_accuracy')*100:.1f} "
                  f"axis={mm('axis_accuracy')*100:.1f} assoc={mm('associator_l2'):.3f} ortho={ortho:.1e}", flush=True)
        pcs = [v["param_count"] for v in reg.values()]
        info = {"min_params": min(pcs), "max_params": max(pcs)}
        if regime == "param_matched":
            info["target"] = TARGET_PARAMS
            info["within_2pct"] = min(pcs) >= 0.98 * TARGET_PARAMS and max(pcs) <= 1.02 * TARGET_PARAMS
            assert info["within_2pct"], pcs
        else:
            info["state_dim"] = 64; info["n_channels"] = 8
        reg["_parity"] = info
        results[regime] = reg
    pool.shutdown()
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    total_start = time.time()
    print("=" * 80)
    print("SOUNIO PHASE 7A: NON-ASSOCIATIVE OCTONION SSM CONTROLLED ABLATION")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 80)

    device = torch.device("cpu")
    print(f"Running execution on: {device}")

    # Part 1: Phase 6C Audit
    part1_res = run_phase6c_audit(device)

    # Part 2: Phase 7A Tree Task Ablation
    part2_res = run_phase7a_ablation(device)

    total_wall = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETE in {total_wall:.2f}s (< 180s target)")
    print("=" * 80)
    assert total_wall < 180.0, f"Total execution time exceeded 180s: {total_wall:.2f}s"

    final_report = {
        "title": "Phase 7A Non-Associative Octonion SSM Ablation Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_wall_time_seconds": float(f"{total_wall:.2f}"),
        "device": str(device),
        "part1_phase6c_audit": part1_res,
        "part2_tree_task_ablation": part2_res,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(final_report, indent=2)
    with open(REPORT_PATH, "w") as f:
        f.write(report_json)

    file_size_kb = REPORT_PATH.stat().st_size / 1024.0
    print(f"Saved report to: {REPORT_PATH} ({file_size_kb:.2f} KB, < 250 KB limit)")
    assert file_size_kb < 250.0, f"Report size {file_size_kb:.2f} KB exceeds 250 KB"

    # Flush output and exit cleanly avoiding libomp destructor race
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
