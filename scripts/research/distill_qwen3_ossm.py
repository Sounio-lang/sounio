#!/usr/bin/env python3
"""
Sounio LLM Research Backend — Fase 6C
Qwen3 -> O-SSM Non-Associative Subspace Grafting + Controlled Ablation on Apple M5 Max MPS

Performs:
1. Startup Self-Test:
   - Vectorized Cayley-Dickson octonion product oct_mul(a, b) matching Sounio's exact sign table
   - Norm-multiplicativity verification to < 1e-5
   - Sounio selective step & associator_l2 verification to < 1e-5
2. Experiment A:
   - Controlled Parameter-Matched Non-Associative vs Associative Ablation
   - Held-out Bracketed Order-Sensitive Algebra Task (2048 train / 512 test)
   - OctonionOssmClassifier vs AssociativeControlClassifier (strictly identical parameter count: 27,408)
3. Experiment B:
   - Real Qwen/Qwen3-0.6B Subspace Grafting (--ossm-every 7 => layers [6, 13, 20, 27])
   - Unseen held-out evaluation vs training calibration split
   - Stage 1: Layerwise teacher-output matching (MSE + cosine loss) using hooked teacher states
   - Stage 2: End-to-end 28-layer hybrid forward distillation (temperature-1.0 KL divergence)
   - Save BF16 safetensors (ossm_every7.safetensors) and comprehensive report (ablation_report.json)
"""

import os
import sys
import time
import json
import random
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# 1. Exact Cayley-Dickson Octonion & Associative Quaternion Algebra
# =============================================================================

def oct_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Cayley-Dickson octonion product matching Sounio stdlib/llm/qwen3_ossm.sio
    qwen3_ossm_oct_mul_f32 sign table verbatim.
    Input tensors a, b have shape (..., 8).
    """
    a0, a1, a2, a3, a4, a5, a6, a7 = a.unbind(dim=-1)
    b0, b1, b2, b3, b4, b5, b6, b7 = b.unbind(dim=-1)

    c0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
    c1 = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6
    c2 = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5
    c3 = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4
    c4 = a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3
    c5 = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2
    c6 = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1
    c7 = a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0

    return torch.stack([c0, c1, c2, c3, c4, c5, c6, c7], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Standard associative Hamilton quaternion product for 4-vectors.
    Input tensors a, b have shape (..., 4).
    """
    a0, a1, a2, a3 = a.unbind(dim=-1)
    b0, b1, b2, b3 = b.unbind(dim=-1)

    c0 = a0*b0 - a1*b1 - a2*b2 - a3*b3
    c1 = a0*b1 + a1*b0 + a2*b3 - a3*b2
    c2 = a0*b2 - a1*b3 + a2*b0 + a3*b1
    c3 = a0*b3 + a1*b2 - a2*b1 + a3*b0

    return torch.stack([c0, c1, c2, c3], dim=-1)


def block_quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Associative control algebra on 8-vectors: direct sum of two associative quaternion algebras (4+4).
    Identical tensor shape [..., 8], zero extra/fewer parameters, strictly associative.
    """
    q1 = quat_mul(a[..., :4], b[..., :4])
    q2 = quat_mul(a[..., 4:], b[..., 4:])
    return torch.cat([q1, q2], dim=-1)


def calc_associator_l2(b: torch.Tensor, u: torch.Tensor, dt: torch.Tensor, mul_fn=oct_mul) -> torch.Tensor:
    """
    Computes ||(B * U) * DT - B * (U * DT)||_2 matching Sounio qwen3_ossm_associator_l2.
    Inputs b, u, dt have shape (..., n_oct, 8).
    Returns L2 norm over (n_oct, 8) with shape (...).
    """
    bu = mul_fn(b, u)
    left = mul_fn(bu, dt)
    udt = mul_fn(u, dt)
    right = mul_fn(b, udt)
    diff = left - right
    return torch.sqrt(torch.sum(diff ** 2, dim=(-2, -1)))


def sounio_ossm_step(H_prev: torch.Tensor, u: torch.Tensor, alpha: torch.Tensor,
                    b: torch.Tensor, dt: torch.Tensor, c: torch.Tensor,
                    mul_fn=oct_mul) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single selective step matching Sounio stdlib/llm/qwen3_ossm.sio qwen3_ossm_step_cpu:
      decay = sigmoid(alpha[..., 0:1])  # shared across 8 components of each octonion channel!
      inj = (B * U) * dt                # strictly left-parenthesized!
      H_new = decay * H_prev + inj
      Y = C * H_new
    """
    decay = torch.sigmoid(alpha[..., 0:1])
    bu = mul_fn(b, u)
    inj = mul_fn(bu, dt)
    H_new = decay * H_prev + inj
    Y = mul_fn(c, H_new)
    return H_new, Y


def ossm_scan(u: torch.Tensor, alpha: torch.Tensor, b: torch.Tensor, dt: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Vectorized O-SSM recurrence scan across sequence length T.
    All inputs have shape (B, T, n_oct, 8).
    Returns Y of shape (B, T, n_oct * 8).
    """
    B, T, n_oct, _ = u.shape
    bu = oct_mul(b, u)
    inj = oct_mul(bu, dt)
    decay_all = torch.sigmoid(alpha[..., 0:1])

    H = torch.zeros(B, n_oct, 8, dtype=u.dtype, device=u.device)
    ys = []
    for t in range(T):
        H = decay_all[:, t] * H + inj[:, t]
        y_t = oct_mul(c[:, t], H)
        ys.append(y_t)

    return torch.stack(ys, dim=1).reshape(B, T, n_oct * 8)


def run_startup_selftest(device: torch.device):
    """
    Validates mathematical consistency against Sounio's specification:
    1. Norm multiplicativity of oct_mul: ||a * b|| == ||a|| * ||b|| to < 1e-5
    2. Exact non-associativity of octonion basis triples: ||(e1*e2)*e4 - e1*(e2*e4)||_2 == 2.0
    3. Exact associativity of block_quat_mul: associator_l2 < 1e-5
    4. Sounio selective step numerical stability
    """
    print("[startup-selftest] Verifying Cayley-Dickson algebra and associator against Sounio stdlib...")
    torch.manual_seed(1337)
    a = torch.randn(64, 8, dtype=torch.float32, device=device)
    b = torch.randn(64, 8, dtype=torch.float32, device=device)
    c = torch.randn(64, 8, dtype=torch.float32, device=device)

    # 1. Norm multiplicativity
    ab = oct_mul(a, b)
    norm_ab = torch.norm(ab, dim=-1)
    expected_norm = torch.norm(a, dim=-1) * torch.norm(b, dim=-1)
    norm_err = torch.max(torch.abs(norm_ab - expected_norm)).item()
    assert norm_err < 1e-5, f"Norm multiplicativity failed: max error {norm_err} >= 1e-5"

    # 2. Canonical non-associative basis triple e1, e2, e4
    eye = torch.eye(8, dtype=torch.float32, device=device)
    e1, e2, e4 = eye[1], eye[2], eye[4]
    left = oct_mul(oct_mul(e1, e2), e4)
    right = oct_mul(e1, oct_mul(e2, e4))
    assoc_diff = torch.norm(left - right).item()
    assert abs(assoc_diff - 2.0) < 1e-5, f"Canonical non-associative triple error: {assoc_diff} != 2.0"

    # 3. Associative control associator
    assoc_quat = calc_associator_l2(a.unsqueeze(1), b.unsqueeze(1), c.unsqueeze(1), mul_fn=block_quat_mul)
    max_quat_assoc = assoc_quat.max().item()
    assert max_quat_assoc < 1e-5, f"Associative control associator failed: {max_quat_assoc} >= 1e-5"

    # 4. Selective step
    H0 = torch.zeros(16, 8, dtype=torch.float32, device=device)
    u_step = torch.randn(16, 8, dtype=torch.float32, device=device)
    alpha_step = torch.randn(16, 8, dtype=torch.float32, device=device)
    b_step = torch.randn(16, 8, dtype=torch.float32, device=device)
    dt_step = torch.randn(16, 8, dtype=torch.float32, device=device)
    c_step = torch.randn(16, 8, dtype=torch.float32, device=device)
    H1, Y1 = sounio_ossm_step(H0, u_step, alpha_step, b_step, dt_step, c_step, mul_fn=oct_mul)
    assert not torch.isnan(H1).any() and not torch.isnan(Y1).any()

    print(f"[startup-selftest] PASS! Norm error: {norm_err:.2e}, e1-e2-e4 assoc: {assoc_diff:.4f}, block-quat assoc: {max_quat_assoc:.2e}")


# =============================================================================
# 2. Experiment A: Controlled Parameter-Matched Non-Associative vs Associative Ablation
# =============================================================================

class AblationClassifier(nn.Module):
    """
    Parameter-matched classifier for the Bracketed Order-Sensitive Algebra Task.
    Can be configured with real Cayley-Dickson octonion multiplication (OctonionOssmClassifier)
    or associative block-quaternion multiplication (AssociativeControlClassifier).
    STRICTLY IDENTICAL tensor shapes, layer dimensions, and parameter count.
    """
    def __init__(self, is_octonion: bool, hidden_dim: int = 64, vocab_size: int = 28, num_classes: int = 16):
        super().__init__()
        self.is_octonion = is_octonion
        self.hidden_dim = hidden_dim
        self.n_oct = hidden_dim // 8
        self.mul_fn = oct_mul if is_octonion else block_quat_mul

        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.u_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.alpha_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape
        h = self.emb(x)

        u = self.u_proj(h).view(B, T, self.n_oct, 8)
        b = self.b_proj(h).view(B, T, self.n_oct, 8)
        c = self.c_proj(h).view(B, T, self.n_oct, 8)
        dt = self.dt_proj(h).view(B, T, self.n_oct, 8)
        alpha = self.alpha_proj(h).view(B, T, self.n_oct, 8)

        decay_all = torch.sigmoid(alpha[..., 0:1])
        bu = self.mul_fn(b, u)
        inj = self.mul_fn(bu, dt)

        H = torch.zeros(B, self.n_oct, 8, dtype=h.dtype, device=h.device)
        h_seq = []
        for t in range(T):
            H = decay_all[:, t] * H + inj[:, t]
            y_t = self.mul_fn(c[:, t], H).reshape(B, self.hidden_dim)
            h_seq.append(h[:, t] + self.out_proj(y_t))

        h_final = h_seq[-1]
        logits = self.classifier(h_final)

        # Associator norm under the model's actual recurrence multiplication
        assoc_l2 = calc_associator_l2(b, u, dt, mul_fn=self.mul_fn)
        return logits, assoc_l2


def build_algebra_ablation_dataset():
    """
    Constructs the Bracketed Order-Sensitive Algebra Task dataset.
    Basis states: {+-e0 .. +-e7} (16 states).
    Tokens:
      0..15: basis states
      16: op_mode 0 (left-bracketed (x * y) * z)
      17: op_mode 1 (right-bracketed x * (y * z))
      18..25: noisy distractors
    """
    eye = torch.eye(8, dtype=torch.float32)
    basis_vecs = torch.cat([eye, -eye], dim=0)

    def get_class(vec: torch.Tensor) -> int:
        idx = torch.argmax(torch.abs(vec)).item()
        sign = torch.sign(vec[idx]).item()
        return idx if sign > 0 else idx + 8

    instances = []
    for i in range(16):
        for j in range(16):
            for k in range(16):
                vi, vj, vk = basis_vecs[i], basis_vecs[j], basis_vecs[k]
                left = oct_mul(oct_mul(vi, vj), vk)
                right = oct_mul(vi, oct_mul(vj, vk))
                if not torch.allclose(left, right):
                    instances.append((i, j, k, 0, get_class(left)))
                    instances.append((i, j, k, 1, get_class(right)))

    # Deterministic train/test split
    rng = random.Random(42)
    rng.shuffle(instances)
    train_inst = instances[:2048]
    test_inst = instances[2048:2048 + 512]

    def encode_batch(insts):
        xs, ys = [], []
        for (i, j, k, op, target) in insts:
            d1 = 18 + (i % 8)
            d2 = 18 + (j % 8)
            d3 = 18 + (k % 8)
            op_tok = 16 if op == 0 else 17
            xs.append([i, d1, j, d2, k, d3, op_tok])
            ys.append(target)
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    return train_inst, test_inst, encode_batch


def run_experiment_a(device: torch.device) -> dict:
    """
    Executes Experiment A: Controlled Parameter-Matched Non-Associative vs Associative Ablation.
    """
    print("\n" + "="*80)
    print("EXPERIMENT A: Controlled Parameter-Matched Non-Associative vs Associative Ablation")
    print("="*80)

    train_inst, test_inst, encode_batch = build_algebra_ablation_dataset()
    test_x, test_y = encode_batch(test_inst)
    test_x, test_y = test_x.to(device), test_y.to(device)

    results = {}
    for is_oct, model_name, class_name in [
        (True, "Octonion O-SSM", "OctonionOssmClassifier"),
        (False, "Associative Control (Block-Quat)", "AssociativeControlClassifier")
    ]:
        print(f"\n--- Training {model_name} on {device} ---")
        torch.manual_seed(42)
        model = AblationClassifier(is_octonion=is_oct, hidden_dim=64).to(device)
        param_count = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        t_start = time.time()
        for step in range(500):
            batch = random.sample(train_inst, 64)
            bx, by = encode_batch(batch)
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            logits, assoc_l2 = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print(f"[{model_name}] Step {step+1}/500 | Train Loss: {loss.item():.4f} | Batch Assoc L2: {assoc_l2.mean().item():.6f}")

        train_wall_sec = time.time() - t_start

        # Evaluation on held-out test split
        model.eval()
        with torch.no_grad():
            test_logits, test_assoc = model(test_x)
            test_loss = criterion(test_logits, test_y).item()
            preds = torch.argmax(test_logits, dim=-1)
            test_acc = (preds == test_y).float().mean().item()
            mean_assoc = test_assoc.mean().item()

        print(f"[{model_name}] Final Results:")
        print(f"  - Parameters: {param_count:,}")
        print(f"  - Final Train Loss: {loss.item():.4f}")
        print(f"  - Held-out Test Loss: {test_loss:.4f}")
        print(f"  - Held-out Test Accuracy: {test_acc*100:.2f}%")
        print(f"  - Mean Associator L2: {mean_assoc:.6f}")
        print(f"  - Training Time: {train_wall_sec:.2f}s")

        results[class_name] = {
            "model_name": model_name,
            "model_class": class_name,
            "parameter_count": param_count,
            "final_train_loss": float(f"{loss.item():.4f}"),
            "held_out_test_loss": float(f"{test_loss:.4f}"),
            "held_out_test_accuracy": float(f"{test_acc:.4f}"),
            "mean_associator_l2": float(f"{mean_assoc:.6f}"),
            "training_time_seconds": float(f"{train_wall_sec:.2f}"),
        }

    p_oct = results["OctonionOssmClassifier"]["parameter_count"]
    p_quat = results["AssociativeControlClassifier"]["parameter_count"]
    assert p_oct == p_quat, f"Parameter count mismatch: {p_oct} != {p_quat}"

    return {
        "task_name": "Bracketed Order-Sensitive Algebra Task",
        "train_examples": len(train_inst),
        "test_examples": len(test_inst),
        "octonion_model": results["OctonionOssmClassifier"],
        "associative_control_model": results["AssociativeControlClassifier"],
        "parameter_count_match": True,
    }


# =============================================================================
# 3. Experiment B: Real Qwen/Qwen3-0.6B Subspace Grafting
# =============================================================================

class OssmGraftBlock(nn.Module):
    """
    Non-associative O-SSM graft block replacing self-attention at layers [6, 13, 20, 27].
    6 BF16 weight matrices without bias (1024x1024 each = 12 MiB per layer, 48 MiB total).
    """
    def __init__(self, hidden_size: int = 1024, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_oct = hidden_size // 8  # 128 channels

        self.u_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.b_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.dt_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.alpha_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

    def scan(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs O-SSM recurrence over sequence tokens:
          decay = sigmoid(alpha[..., 0:1])
          inj = (B * U) * dt
          H_new = decay * H_prev + inj
          Y = C * H_new
        Returns:
          Y_scan: [B, T, hidden_size]
          assoc_l2: [B, T]
        """
        B, T, _ = h.shape
        u = self.u_proj(h).view(B, T, self.n_oct, 8)
        alpha = self.alpha_proj(h).view(B, T, self.n_oct, 8)
        b = self.b_proj(h).view(B, T, self.n_oct, 8)
        dt = self.dt_proj(h).view(B, T, self.n_oct, 8)
        c = self.c_proj(h).view(B, T, self.n_oct, 8)

        # Associator L2 for monitoring non-associativity
        assoc_l2 = calc_associator_l2(b, u, dt, mul_fn=oct_mul)

        y_stacked = ossm_scan(u, alpha, b, dt, c)
        return y_stacked, assoc_l2

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_scan, assoc_l2 = self.scan(h)
        y_ossm = self.out_proj(y_scan)
        return y_ossm, assoc_l2


def build_calibration_dataset(tokenizer) -> list[str]:
    """
    Constructs 56 diverse training calibration sequences.
    Strictly excludes 'France', 'Paris', 'Japan', 'Tokyo', '2 + 2', 'Two plus two'.
    """
    texts = [
        # Science & Astronomy
        "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a giant molecular cloud.",
        "Photosynthesis is a biological process used by green plants to convert solar energy into chemical sugars.",
        "Mitochondria generate most of the chemical energy needed to power the cell's biochemical reactions.",
        "Gravitational waves are disturbances in spacetime curvature caused by accelerated massive celestial objects.",
        "The speed of light in a vacuum is exactly 299,792,458 meters per second by international agreement.",
        "Plate tectonics describes the large-scale motion of seven large plates and the movements of a larger number of smaller plates of Earth's lithosphere.",
        "DNA is composed of two polynucleotide chains that coil around each other to form a double helix.",
        "Quantum mechanics is a fundamental theory in physics that describes the behavior of nature at the scale of atoms and subatomic particles.",

        # History & Philosophy
        "The Roman Republic was the era of classical Roman civilization beginning with the overthrow of the Roman Kingdom in 509 BC.",
        "The Library of Alexandria was one of the largest and most significant libraries of the ancient world.",
        "Epistemology investigates the origin, nature, methods, and limits of human knowledge and belief systems.",
        "The Renaissance was a fervent period of European cultural, artistic, political and economic rebirth following the Middle Ages.",
        "The Silk Road was a network of Eurasian trade routes active from the Han dynasty until the Ottoman Empire.",
        "Aristotle wrote treatises on physics, metaphysics, poetry, theater, music, rhetoric, politics, and biology.",
        "The Industrial Revolution marked a major turning point in human history, influencing almost every aspect of daily life.",
        "Hammurabi was the sixth king of the First Babylonian dynasty who promulgated an extensive code of law.",

        # Computing & Engineering
        "A compiler translates computer code written in one programming language into another language, usually machine code.",
        "Operating systems coordinate CPU allocation, virtual memory mapping, storage access, and peripheral device control.",
        "A relational database organizes data into tables consisting of rows and columns with unique keys.",
        "Dynamic programming solves complex problems by breaking them down into simpler overlapping subproblems.",
        "The Fast Fourier Transform computes the discrete Fourier transform of a sequence in O(N log N) time complexity.",
        "Pointers in systems programming hold memory addresses of variables rather than the variable values themselves.",
        "Distributed consensus protocols allow multiple networked servers to agree on state transitions despite network partitions.",
        "Cache coherence protocols ensure that changes in the values of shared operands are propagated throughout the system promptly.",

        # Geography & Nature
        "The Amazon River in South America is the largest river by discharge volume of water in the world.",
        "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.",
        "The Sahara is the largest hot desert in the world and the third largest desert overall.",
        "Coral reefs are diverse underwater ecosystems held together by calcium carbonate structures secreted by corals.",
        "Lake Baikal in southern Siberia is the world's deepest and oldest freshwater lake by volume.",
        "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions, extending from the Arctic to the Antarctic.",
        "Glaciers are persistent bodies of dense ice that are constantly moving under their own gravity.",
        "The Mariana Trench is the deepest oceanic trench on Earth, situated in the western Pacific Ocean.",

        # Multi-turn ChatML conversations
        "<|im_start|>system\nYou are a helpful and knowledgeable science assistant.<|im_end|>\n<|im_start|>user\nWhat is the fundamental law of universal gravitation?<|im_end|>\n<|im_start|>assistant\nNewton's law states that every particle attracts every other particle with a force proportional to the product of their masses and inversely proportional to the square of the distance between their centers.<|im_end|>",
        "<|im_start|>system\nYou are a coding tutor.<|im_end|>\n<|im_start|>user\nExplain binary search in one clear sentence.<|im_end|>\n<|im_start|>assistant\nBinary search repeatedly divides a sorted interval in half to locate a target element in logarithmic time.<|im_end|>",
        "<|im_start|>system\nYou are an astronomy expert.<|im_end|>\n<|im_start|>user\nHow do stars produce energy?<|im_end|>\n<|im_start|>assistant\nStars generate energy in their dense cores through nuclear fusion, predominantly fusing hydrogen into helium under immense pressure and temperature.<|im_end|>",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat causes ocean tides on Earth?<|im_end|>\n<|im_start|>assistant\nOcean tides are primarily caused by the gravitational pull exerted by the Moon and the Sun upon Earth's rotating waters.<|im_end|>",
        "<|im_start|>system\nYou are an architecture expert.<|im_end|>\n<|im_start|>user\nWhat is the Roman arch and why was it significant?<|im_end|>\n<|im_start|>assistant\nThe Roman arch allowed builders to span wide openings by distributing compressive weight down through curved masonry piers.<|im_end|>",
        "<|im_start|>system\nYou are a mathematics instructor.<|im_end|>\n<|im_start|>user\nWhat is a prime number?<|im_end|>\n<|im_start|>assistant\nA prime number is an integer greater than one that has no positive divisors other than one and itself.<|im_end|>",
        "<|im_start|>system\nYou are a biology tutor.<|im_end|>\n<|im_start|>user\nWhat is an enzyme?<|im_end|>\n<|im_start|>assistant\nAn enzyme is a biological protein catalyst that accelerates biochemical reactions by lowering their activation energy.<|im_end|>",
        "<|im_start|>system\nYou are a linguistics assistant.<|im_end|>\n<|im_start|>user\nWhat is syntax in natural language?<|im_end|>\n<|im_start|>assistant\nSyntax is the set of grammatical rules and principles that govern how words combine into coherent phrases and clauses.<|im_end|>",

        # Additional diverse prose
        "The theory of general relativity asserts that gravity is not a Newtonian force but rather the curvature of spacetime.",
        "Superconductivity is a set of physical properties observed in certain materials where electrical resistance vanishes completely.",
        "In thermodynamics, entropy is a scientific concept as well as a measurable physical property that is most commonly associated with a state of disorder.",
        "The greenhouse effect is a process that occurs when energy from a planet's host star goes through its atmosphere and warms the surface.",
        "The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs and 900 islands.",
        "Geological strata provide a chronological record of Earth's planetary history through sedimentation and fossil deposits.",
        "The water cycle describes the continuous movement of water on, above and below the surface of the Earth.",
        "The ozone layer absorbs most of the Sun's ultraviolet radiation, preventing harmful wavelengths from damaging cellular DNA.",
        "The human nervous system consists of the central nervous system, including the brain and spinal cord, and the peripheral nervous system.",
        "Sedimentary rocks are formed by the accumulation or deposition of mineral or organic particles at Earth's surface.",
        "Metamorphic rocks arise from the transformation of existing rock to new types in a process called metamorphism.",
        "Igneous rock is formed through the cooling and solidification of magma or volcanic lava.",
        "Plate tectonics explains seismic activity, volcanic eruptions, and continental drift across geological epochs.",
        "An algorithm is a finite sequence of rigorous instructions, typically used to solve a class of specific problems or perform computation.",
        "Cryptography enables secure communication in the presence of adversarial third parties through mathematical encryption protocols.",
        "Microprocessors contain millions or billions of transistors integrated on a single semiconductor microchip."
    ]
    return texts


def run_experiment_b(device: torch.device, model_path: str, output_dir: Path) -> dict:
    """
    Executes Experiment B: Real Qwen/Qwen3-0.6B Subspace Grafting (--ossm-every 7 => layers [6, 13, 20, 27]).
    """
    print("\n" + "="*80)
    print("EXPERIMENT B: Real Qwen/Qwen3-0.6B Subspace Grafting (--ossm-every 7)")
    print("="*80)

    graft_layer_indices = [6, 13, 20, 27]
    print(f"Loading local base model from {model_path} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    calib_texts = build_calibration_dataset(tokenizer)
    print(f"Built {len(calib_texts)} training calibration sequences (strictly avoiding held-out topics).")

    # Tokenize training sequences
    calib_batches = []
    for txt in calib_texts:
        enc = tokenizer(txt, return_tensors="pt", max_length=96, truncation=True)
        calib_batches.append(enc.input_ids.to(device))

    # Initialize 4 OssmGraftBlocks with mathematically calibrated gain matching Qwen3 RMSNorm
    # Depth-scaled input gain ensures non-divergent recurrence across all 28 layers
    layer_scales = {6: 0.03, 13: 0.02, 20: 0.015, 27: 0.01}

    graft_blocks = nn.ModuleDict()
    for l_idx in graft_layer_indices:
        block = OssmGraftBlock(hidden_size=1024, dtype=torch.bfloat16).to(device)
        teacher_layer = model.model.layers[l_idx]
        s = layer_scales[l_idx]
        with torch.no_grad():
            block.u_proj.weight.data.copy_(teacher_layer.self_attn.q_proj.weight.data[:1024] * s)
            block.b_proj.weight.data.copy_(teacher_layer.self_attn.k_proj.weight.data * s)
            block.c_proj.weight.data.copy_(teacher_layer.self_attn.v_proj.weight.data * s)
            block.dt_proj.weight.data.copy_(teacher_layer.self_attn.v_proj.weight.data * s)
            block.out_proj.weight.data.copy_(teacher_layer.self_attn.o_proj.weight.data[:, :1024])
            torch.nn.init.normal_(block.alpha_proj.weight, mean=0.0, std=0.01)
        graft_blocks[str(l_idx)] = block

    total_graft_params = sum(p.numel() for p in graft_blocks.parameters())
    graft_bytes = total_graft_params * 2  # BF16
    print(f"Created 4 OssmGraftBlocks: {total_graft_params:,} parameters ({graft_bytes / (1024*1024):.2f} MiB).")

    # -------------------------------------------------------------------------
    # Stage 1: Layerwise Teacher-Output Matching via Hooked Intermediate States
    # -------------------------------------------------------------------------
    print("\n--- STAGE 1: Layerwise Teacher-Output Matching (~300 steps) ---")
    stage1_start = time.time()

    collected_states = {l: [] for l in graft_layer_indices}
    collected_teacher_logits = []

    def make_ln_hook(l):
        def hook(module, args, output):
            curr_h[l] = output.detach()
        return hook

    def make_attn_hook(l):
        def hook(module, args, output):
            attn_val = output[0].detach() if isinstance(output, tuple) else output.detach()
            curr_attn[l] = attn_val
        return hook

    print("Collecting teacher intermediate states (h, attn_out) across training sequences...")
    for inp_ids in calib_batches:
        curr_h = {}
        curr_attn = {}
        hooks = []
        for l_idx in graft_layer_indices:
            layer = model.model.layers[l_idx]
            hooks.append(layer.input_layernorm.register_forward_hook(make_ln_hook(l_idx)))
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(l_idx)))

        with torch.no_grad():
            out = model(inp_ids)
            collected_teacher_logits.append(out.logits.detach())

        for h in hooks:
            h.remove()

        for l_idx in graft_layer_indices:
            collected_states[l_idx].append((curr_h[l_idx], curr_attn[l_idx]))

    print(f"Collected states for {len(calib_batches)} sequences across layers {graft_layer_indices}.")

    stage1_metrics = {}
    for l_idx in graft_layer_indices:
        print(f"Optimizing GraftBlock for Layer {l_idx} on MPS (scale {layer_scales[l_idx]})...")
        block = graft_blocks[str(l_idx)]
        block.train()
        optimizer = torch.optim.AdamW(block.parameters(), lr=1.5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)

        layer_data = collected_states[l_idx]
        for step in range(300):
            # Sample random sequence
            h_sample, tgt_sample = random.choice(layer_data)

            optimizer.zero_grad()
            y_ossm, assoc_l2 = block(h_sample)

            mse = F.mse_loss(y_ossm, tgt_sample)
            cos_sim = F.cosine_similarity(y_ossm, tgt_sample, dim=-1).mean()
            cos_loss = 1.0 - cos_sim

            # Non-degeneracy regularizer: ensure assoc_l2 > 0.01
            assoc_mean = assoc_l2.mean()
            assoc_reg = 0.05 * torch.relu(0.02 - assoc_mean)

            loss = mse + cos_loss + assoc_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % 100 == 0:
                print(f"  [Layer {l_idx}] Step {step+1}/300 | Loss: {loss.item():.4f} (MSE: {mse.item():.4f}, Cos: {cos_sim.item():.4f}) | Assoc L2: {assoc_mean.item():.4f}")

        stage1_metrics[f"layer_{l_idx}"] = {
            "final_loss": float(f"{loss.item():.4f}"),
            "final_mse": float(f"{mse.item():.4f}"),
            "final_cosine_similarity": float(f"{cos_sim.item():.4f}"),
            "mean_associator_l2": float(f"{assoc_mean.item():.4f}")
        }

    stage1_wall = time.time() - stage1_start
    print(f"Stage 1 finished in {stage1_wall:.2f}s.")

    # -------------------------------------------------------------------------
    # Stage 2: End-to-End 28-Layer Hybrid Forward Distillation
    # -------------------------------------------------------------------------
    print("\n--- STAGE 2: End-to-End 28-Layer Hybrid Forward Distillation (~120 steps) ---")
    stage2_start = time.time()

    orig_forwards = {}
    last_assoc_readings = {}

    def make_grafted_layer_forward(orig_layer, graft_blk, l_num):
        def grafted_forward(hidden_states, *args, **kwargs):
            residual = hidden_states
            h = orig_layer.input_layernorm(hidden_states)
            y_ossm, assoc_l2 = graft_blk(h)
            last_assoc_readings[l_num] = assoc_l2.mean().item()
            hidden_states = residual + y_ossm

            residual = hidden_states
            hidden_states = orig_layer.post_attention_layernorm(hidden_states)
            hidden_states = orig_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states
        return grafted_forward

    for l_idx in graft_layer_indices:
        layer = model.model.layers[l_idx]
        orig_forwards[l_idx] = layer.forward
        layer.forward = make_grafted_layer_forward(layer, graft_blocks[str(l_idx)], l_idx)

    optimizer_e2e = torch.optim.AdamW(graft_blocks.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler_e2e = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_e2e, T_max=120, eta_min=1e-5)

    for step in range(120):
        idx = random.randint(0, len(calib_batches) - 1)
        inp_ids = calib_batches[idx]
        t_logits = collected_teacher_logits[idx].float()

        optimizer_e2e.zero_grad()
        h_out = model(inp_ids)
        h_logits = h_out.logits.float()

        p = F.softmax(t_logits, dim=-1)
        log_q = F.log_softmax(h_logits, dim=-1)
        kl = F.kl_div(log_q, p, reduction="batchmean")

        kl.backward()
        torch.nn.utils.clip_grad_norm_(graft_blocks.parameters(), 1.0)
        optimizer_e2e.step()
        scheduler_e2e.step()

        if (step + 1) % 30 == 0:
            print(f"[Stage 2] Step {step+1}/120 | Temperature-1.0 KL: {kl.item():.4f}")

    stage2_wall = time.time() - stage2_start
    print(f"Stage 2 finished in {stage2_wall:.2f}s.")

    # -------------------------------------------------------------------------
    # Evaluation on Held-Out Split
    # -------------------------------------------------------------------------
    print("\n--- HELD-OUT EVALUATION ---")
    held_out_prompts = [
        "The capital of France is",
        "The capital of Japan is",
        "Two plus two equals"
    ]

    eval_results = []
    graft_blocks.eval()

    for prompt in held_out_prompts:
        inp_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # 1. Un-patch to get teacher outputs
        for l_idx in graft_layer_indices:
            model.model.layers[l_idx].forward = orig_forwards[l_idx]

        with torch.no_grad():
            t_out = model(inp_ids)
            t_logits = t_out.logits.float()

        # 16-token greedy teacher generation
        t_seq = inp_ids.clone()
        for _ in range(16):
            with torch.no_grad():
                out = model(t_seq)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                t_seq = torch.cat([t_seq, next_tok], dim=-1)
        teacher_16tok = tokenizer.decode(t_seq[0])

        # 2. Re-patch to get hybrid outputs
        for l_idx in graft_layer_indices:
            model.model.layers[l_idx].forward = make_grafted_layer_forward(
                model.model.layers[l_idx], graft_blocks[str(l_idx)], l_idx
            )

        with torch.no_grad():
            h_out = model(inp_ids)
            h_logits = h_out.logits.float()

        # 16-token greedy hybrid generation
        h_seq = inp_ids.clone()
        for _ in range(16):
            with torch.no_grad():
                out = model(h_seq)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                h_seq = torch.cat([h_seq, next_tok], dim=-1)
        hybrid_16tok = tokenizer.decode(h_seq[0])

        # Next-token metrics at prompt boundary
        p_final = F.softmax(t_logits[:, -1, :], dim=-1)
        log_q_final = F.log_softmax(h_logits[:, -1, :], dim=-1)
        kl_final = F.kl_div(log_q_final, p_final, reduction="batchmean").item()

        t_top1_id = torch.argmax(t_logits[:, -1, :], dim=-1).item()
        h_top1_id = torch.argmax(h_logits[:, -1, :], dim=-1).item()
        t_top1_str = tokenizer.decode([t_top1_id])
        h_top1_str = tokenizer.decode([h_top1_id])
        agreed = (t_top1_id == h_top1_id)

        mean_prompt_assoc = sum(last_assoc_readings.values()) / len(last_assoc_readings)

        print(f"\nPrompt: {repr(prompt)}")
        print(f"  Teacher Top-1: {repr(t_top1_str)} (ID {t_top1_id})")
        print(f"  Hybrid  Top-1: {repr(h_top1_str)} (ID {h_top1_id}) | Agreement: {agreed}")
        print(f"  KL Divergence: {kl_final:.4f}")
        print(f"  Mean Associator L2: {mean_prompt_assoc:.4f}")
        print(f"  Teacher 16-tok: {repr(teacher_16tok)}")
        print(f"  Hybrid  16-tok: {repr(hybrid_16tok)}")

        eval_results.append({
            "prompt": prompt,
            "teacher_top1_token": t_top1_str,
            "teacher_top1_id": t_top1_id,
            "hybrid_top1_token": h_top1_str,
            "hybrid_top1_id": h_top1_id,
            "top1_agreement": agreed,
            "kl_divergence": float(f"{kl_final:.4f}"),
            "mean_associator_l2": float(f"{mean_prompt_assoc:.4f}"),
            "layer_associator_l2": {f"layer_{k}": float(f"{v:.4f}") for k, v in last_assoc_readings.items()},
            "teacher_completion_16tok": teacher_16tok,
            "hybrid_completion_16tok": hybrid_16tok
        })

    # -------------------------------------------------------------------------
    # Save BF16 safetensors file
    # -------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = output_dir / "ossm_every7.safetensors"
    print(f"\nSaving BF16 safetensors sidecar to {safetensors_path}...")

    tensors_to_save = {}
    for l_idx in graft_layer_indices:
        block = graft_blocks[str(l_idx)]
        tensors_to_save[f"ossm.layers.{l_idx}.u_proj.weight"] = block.u_proj.weight.data.to(torch.bfloat16).cpu()
        tensors_to_save[f"ossm.layers.{l_idx}.b_proj.weight"] = block.b_proj.weight.data.to(torch.bfloat16).cpu()
        tensors_to_save[f"ossm.layers.{l_idx}.c_proj.weight"] = block.c_proj.weight.data.to(torch.bfloat16).cpu()
        tensors_to_save[f"ossm.layers.{l_idx}.dt_proj.weight"] = block.dt_proj.weight.data.to(torch.bfloat16).cpu()
        tensors_to_save[f"ossm.layers.{l_idx}.alpha_proj.weight"] = block.alpha_proj.weight.data.to(torch.bfloat16).cpu()
        tensors_to_save[f"ossm.layers.{l_idx}.out_proj.weight"] = block.out_proj.weight.data.to(torch.bfloat16).cpu()

    save_file(tensors_to_save, str(safetensors_path))
    file_size_bytes = safetensors_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Safetensors saved successfully: {file_size_mb:.2f} MiB ({file_size_bytes:,} bytes).")
    assert file_size_mb < 55.0, f"Safetensors file size exceeded 55 MiB: {file_size_mb:.2f} MiB"

    return {
        "base_model": "Qwen/Qwen3-0.6B",
        "grafted_layers": graft_layer_indices,
        "ossm_every": 7,
        "total_graft_parameters": total_graft_params,
        "safetensors_file": str(safetensors_path),
        "safetensors_file_size_mb": float(f"{file_size_mb:.2f}"),
        "safetensors_file_size_bytes": file_size_bytes,
        "stage1_matching_metrics": stage1_metrics,
        "stage1_wall_time_seconds": float(f"{stage1_wall:.2f}"),
        "stage2_final_kl": float(f"{kl.item():.4f}"),
        "stage2_wall_time_seconds": float(f"{stage2_wall:.2f}"),
        "held_out_evaluation": eval_results
    }


# =============================================================================
# 4. Main Entrypoint & Report Generation
# =============================================================================

def main():
    print("="*80)
    print("Sounio Research — Fase 6C Backend Execution")
    print("Platform: Apple M5 Max | Device: MPS")
    print("="*80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Startup Self-Test
    run_startup_selftest(device)

    # Step 2: Experiment A (Controlled Ablation)
    exp_a_results = run_experiment_a(device)

    # Step 3: Experiment B (Qwen3 Subspace Grafting)
    model_path = "/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B"
    output_dir = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar")
    exp_b_results = run_experiment_b(device, model_path, output_dir)

    # Write ablation_report.json
    report_path = output_dir / "ablation_report.json"
    full_report = {
        "fase": "Fase 6C",
        "title": "Qwen3 -> O-SSM Non-Associative Subspace Grafting + Associative vs Non-Associative Ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "hardware": "Apple M5 Max MPS",
        },
        "experiment_a_controlled_ablation": exp_a_results,
        "experiment_b_subspace_grafting": exp_b_results,
    }

    print(f"\nWriting comprehensive report to {report_path}...")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"Report written ({report_path.stat().st_size:,} bytes).")
    print("\n[SUCCESS] Fase 6C Scientific Execution Complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        os._exit(1)
    finally:
        os._exit(0)
