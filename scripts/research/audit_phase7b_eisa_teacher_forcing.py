#!/usr/bin/env python3
"""
Sounio Research Backend — Phase 7B
Block-32 EISA Quantization & Exact Argmax Certification Audit
Evaluated on Qwen/Qwen3-0.6B across >= 2,048 Teacher-Forced Token Positions

Multi-Domain Corpus:
1. Factual Prose (Ancient Greek Classical Golden Age & Peloponnesian War)
2. Mathematics & Abstract Algebra (Cayley-Dickson, Hurwitz, Artin Alternativity)
3. Systems & Code (Cayley-Dickson and O-SSM State Space Implementation)
4. Clinical Pharmacology (DOACs, Apixaban PK/PD, Metabolism, Drug Interactions)
5. Formal Logic & Relational Semantics (Kripke Frames, Epistemic Axioms S4/S5)

Audit Targets:
1. Bit-exact block-32 Q4_EISA and Q8_EISA quantization matching stdlib/llm/qwen3_quant_eisa.sio:
   - Scale: f16(max_abs / 127) for Q8, f16(max_abs / 7) for Q4
   - Upward-rounded f16 rho matching compute_rho_bits_upward
   - Exact Cauchy-Schwarz error bound: err = sum_b [ rho * ||x_b||_2 + 1e-5 * (||w_b||_2 + rho) * ||x_b||_2 ]
2. q8_cone audit: BF16 backbone + Q8_EISA lm_head + Global Certifier & Cone Refiner
   - Over all >= 2,048 tokens:
     n_tokens, stage1_certified_count, stage1_certified_frac, stage1_false_certificates (MUST BE 0),
     95% Clopper-Pearson upper bound, cone size statistics (mean, median, p99),
     bf16_lm_head_bytes_touched_frac, bf16_lm_head_bandwidth_saved_pct, stage2_cone_refined_flips (MUST BE 0)
3. q4_cone audit: BF16 backbone + Q4_EISA lm_head + Global Certifier & Cone Refiner
   - Identical fields over all >= 2,048 tokens (stage1_false_certificates == 0, stage2_cone_refined_flips == 0)
4. full_28l_q8_eisa audit: All 28 layers + lm_head quantized to Q8_EISA
   - 28-layer empirical hidden drift ||dh||_2
   - Max logit drift ||z_q8_28l - z_bf16||_inf
   - Raw uncalibrated lm_head-only false certificates vs Lipschitz/spectral calibrated certificates
   - Total BF16 memory traffic saved vs 100%-BF16-verifying self-speculative decoding (Quasar)

Saves report to:
/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/phase7b_eisa_audit.json (< 250 KB)
"""

import os
import sys
import time
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_DIR = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B")
REPORT_PATH = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/phase7b_eisa_audit.json")


# =============================================================================
# 1. Deterministic Multi-Domain Corpus (>= 2,048 Tokens)
# =============================================================================

DOMAINS = {
    "factual_prose": (
        "The history of Athens during the Classical period of Ancient Greece spans roughly from the "
        "expulsion of the Peisistratid tyranny in 510 BC to the death of Alexander the Great in 323 BC. "
        "Following the Cleisthenic reforms, Athens established an early direct democracy in which all male "
        "citizens possessed equal political rights, freedom of speech, and the opportunity to participate "
        "directly in the political arena. Under the leadership of Pericles, Athens entered a golden age "
        "characterized by monumental architectural achievements, including the construction of the Parthenon "
        "on the Acropolis of Athens, the Erechtheion, and the Propylaea. Pericles promoted the arts and literature, "
        "fostering an environment in which tragedy, comedy, philosophy, and historical inquiry flourished. "
        "Prominent playwrights such as Aeschylus, Sophocles, and Euripides transformed theatrical expression, "
        "while historians Herodotus and Thucydides pioneered systematic historical documentation and analysis. "
        "In philosophy, Socrates engaged Athenian citizens in dialectical inquiry, laying foundational principles "
        "for Western philosophical thought, which were subsequently developed and institutionalized by Plato in "
        "his Academy and Aristotle in the Lyceum. The Peloponnesian War between the Delian League, dominated by "
        "Athens, and the Peloponnesian League, led by Sparta, reshaped the Greek geopolitical landscape. After "
        "enduring a devastating plague that killed a significant portion of its population including Pericles himself, "
        "and suffering a catastrophic defeat in the Sicilian Expedition, Athens ultimately surrendered to Sparta "
        "in 404 BC. Although the Spartan victory dismantled the Athenian empire and briefly established the "
        "oligarchic regime known as the Thirty Tyrants, Athenian democracy was restored the following year in 403 BC. "
        "Throughout the fourth century BC, Athens regained substantial naval strength and economic prosperity, "
        "continuing to serve as a major cultural, philosophical, and rhetorical center until the rise of Macedon "
        "under Philip II and Alexander the Great altered Mediterranean hegemony forever."
    ),
    "math_algebra": (
        "In abstract algebra, a Cayley-Dickson construction defines a sequence of algebras over a field, "
        "where each algebra has twice the dimension of the previous one. Starting from the real numbers R "
        "of dimension 1, the construction yields the complex numbers C of dimension 2, the quaternions H "
        "of dimension 4, and the octonions O of dimension 8. Each doubling step introduces an imaginary generator "
        "and modifies the algebraic properties of the system. While the real numbers form a complete ordered field "
        "and the complex numbers form an algebraically closed field that is commutative and associative, the "
        "quaternions surrender commutativity while preserving associativity, forming a non-commutative division ring. "
        "Proceeding from the quaternions to the octonions preserves the normed division property according to "
        "Hurwitz theorem, but surrenders associativity. Specifically, the octonions satisfy the weaker property of "
        "alternativity, which asserts that the associator of any two elements vanishes: [x, x, y] = 0 and [y, x, x] = 0 "
        "for all elements x and y. Consequently, by Artins theorem, any subalgebra generated by any two elements "
        "of an alternative algebra is associative. Non-associativity in the octonions is governed by the Fano plane, "
        "a finite projective plane with seven points and seven lines, encoding the multiplication table of the seven "
        "imaginary units e1 through e7. For any three imaginary units that do not lie on a common line of the Fano "
        "plane, such as e1, e2, and e4, their associator does not vanish: (e1 * e2) * e4 - e1 * (e2 * e4) = 2 * e7. "
        "When extending the construction beyond dimension 8 to the sedenions of dimension 16, the normed division "
        "algebra property is lost, introducing non-trivial zero divisors where two non-zero sedenions can multiply to zero."
    ),
    "code_systems": (
        "def cayley_dickson_multiply(a: list[float], b: list[float]) -> list[float]:\n"
        "    assert len(a) == len(b), 'Operand vectors must have identical dimensions'\n"
        "    n = len(a)\n"
        "    if n == 1:\n"
        "        return [a[0] * b[0]]\n"
        "    half = n // 2\n"
        "    a_L, a_R = a[:half], a[half:]\n"
        "    b_L, b_R = b[:half], b[half:]\n"
        "    b_R_conj = [b_R[0]] + [-x for x in b_R[1:]]\n"
        "    a_R_conj = [a_R[0]] + [-x for x in a_R[1:]]\n"
        "    term1 = [x - y for x, y in zip(cayley_dickson_multiply(a_L, b_L), cayley_dickson_multiply(b_R_conj, a_R))]\n"
        "    b_L_conj = [b_L[0]] + [-x for x in b_L[1:]]\n"
        "    term2 = [x + y for x, y in zip(cayley_dickson_multiply(b_R, a_L), cayley_dickson_multiply(a_R, b_L_conj))]\n"
        "    return term1 + term2\n\n"
        "class OctonionStateSpace:\n"
        "    def __init__(self, channels: int, hidden_dim: int):\n"
        "        self.channels = channels\n"
        "        self.hidden_dim = hidden_dim\n"
        "        self.state_left = [[0.0] * 8 for _ in range(channels)]\n"
        "        self.state_right = [[0.0] * 8 for _ in range(channels)]\n\n"
        "    def step(self, u_t: list[list[float]], decay: float, gate: float) -> list[list[float]]:\n"
        "        output = []\n"
        "        for ch in range(self.channels):\n"
        "            hl = self.state_left[ch]\n"
        "            hr = self.state_right[ch]\n"
        "            u = u_t[ch]\n"
        "            hl_hr = cayley_dickson_multiply(hl, hr)\n"
        "            left_assoc = cayley_dickson_multiply(hl_hr, u)\n"
        "            hr_u = cayley_dickson_multiply(hr, u)\n"
        "            right_assoc = cayley_dickson_multiply(hl, hr_u)\n"
        "            new_hl = [decay * h + gate * l + (1.0 - gate) * r for h, l, r in zip(hl, left_assoc, right_assoc)]\n"
        "            new_hr = [decay * h + ui for h, ui in zip(hr, u)]\n"
        "            self.state_left[ch] = new_hl\n"
        "            self.state_right[ch] = new_hr\n"
        "            output.append(new_hl)\n"
        "        return output\n"
    ),
    "clinical_pharmacology": (
        "Pharmacokinetics and pharmacodynamics of direct oral anticoagulants (DOACs) represent a crucial "
        "aspect of modern cardiovascular pharmacology. Apixaban is a potent, reversible, direct, and highly "
        "selective inhibitor of both free and clot-bound factor Xa, as well as prothrombinase activity, without "
        "requiring antithrombin III for antithrombotic activity. By inhibiting factor Xa, apixaban decreases "
        "thrombin generation and development of blood clots. The absolute bioavailability of apixaban is approximately "
        "50% for doses up to 10 mg, with peak plasma concentrations occurring 3 to 4 hours following oral administration. "
        "Food does not significantly affect the bioavailability of apixaban. Plasma protein binding in humans is "
        "approximately 87%. Apixaban is eliminated through multiple pathways including metabolism, biliary excretion, "
        "and direct renal excretion. Cytochrome P450 3A4/5 accounts for the majority of metabolic clearance, with minor "
        "contributions from CYP1A2, CYP2C8, CYP2C9, CYP2C19, and CYP2J2. Approximately 27% of total clearance occurs via "
        "renal excretion of unchanged drug. The apparent half-life in healthy subjects is approximately 12 hours following "
        "repeat oral administration. Co-administration of strong dual inhibitors of CYP3A4 and P-glycoprotein, such as "
        "ketoconazole, itraconazole, ritonavir, or clarithromycin, leads to significant increases in apixaban exposure, "
        "necessitating dosage reduction or avoidance. Conversely, strong inducers of both CYP3A4 and P-glycoprotein, such as "
        "rifampin, carbamazepine, phenytoin, or St. Johns Wort, decrease apixaban plasma concentrations and efficacy, "
        "increasing stroke and systemic embolism risk."
    ),
    "formal_logic": (
        "In classical propositional logic, a deduction system is sound if every provable formula is valid, and "
        "complete if every semantic consequence is provable. Consider a formal system defined over a set of propositional "
        "variables P and standard connectives: negation, conjunction, disjunction, implication, and equivalence. "
        "A model M assigns truth values in {0, 1} to each variable. The semantic entailment relation entails phi "
        "asserts that for all models M, if every premise in Gamma evaluates to 1, then phi evaluates to 1. In modal "
        "logic, the relational semantics of Kripke frames provide a framework for reasoning about necessity and "
        "possibility. A frame is a pair (W, R) where W is a non-empty set of possible worlds and R is a binary accessibility "
        "relation on W. The modal operator Box represents necessity, while Diamond represents possibility, defined dually "
        "as the negation of Box negation. The basic modal logic K is determined by the class of all Kripke frames. "
        "Imposing algebraic and topological constraints on the accessibility relation R gives rise to classic axiomatic "
        "extensions. When R is reflexive, the axiom T holds: Box phi implies phi. When R is transitive, the axiom 4 holds: "
        "Box phi implies Box Box phi. When R is symmetric, the axiom B holds: phi implies Box Diamond phi. The system S4 "
        "corresponds to reflexive and transitive frames, representing epistemic logic where knowledge implies truth and "
        "positive introspection holds. The system S5 corresponds to equivalence relations, where knowledge satisfies "
        "both positive and negative introspection."
    ),
}


# =============================================================================
# 2. Block-32 EISA Quantization & Upward-Rounded Bound Functions
# =============================================================================

def compute_rho_upward_tensor(exact_l2: torch.Tensor) -> torch.Tensor:
    """
    Computes bit-exact upward-rounded float16 rho matching compute_rho_bits_upward in Sounio.
    Guarantees rho >= exact_l2 for all entries.
    """
    r16 = exact_l2.to(torch.float16)
    rf = r16.float()
    mask = rf < exact_l2
    if mask.any():
        u16 = r16[mask].view(torch.int16)
        # In IEEE 754 positive floats, incrementing bit representation increases value
        u16_inc = torch.where((u16 >= 0) & (u16 < 0x7C00), u16 + 1, u16)
        rf[mask] = u16_inc.view(torch.float16).float()
    return rf


def quantize_w_eisa_q8(w_float: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bit-exact block-32 Q8_EISA quantization matching stdlib/llm/qwen3_quant_eisa.sio:
      scale = f16(max_abs / 127.0)
      q = clamp(round(w / scale), -127, 127)
      w_dequant = scale * q
      exact_l2 = sqrt(sum((w - w_dequant)^2))
      rho = compute_rho_upward(exact_l2)
    Returns:
      w_dequant: (V, K)
      rho: (V, K // 32)
      norm_w: (V, K // 32)
    """
    V, K = w_float.shape
    w_blocks = w_float.view(V, K // 32, 32)
    max_abs = torch.amax(torch.abs(w_blocks), dim=-1, keepdim=True)
    s = (max_abs / 127.0).to(torch.float16).float()
    s = torch.where(s == 0.0, torch.ones_like(s), s)
    q = torch.clamp(torch.round(w_blocks / s), -127, 127)
    w_dequant_blocks = s * q
    diff = w_blocks - w_dequant_blocks
    exact_l2 = torch.sqrt(torch.sum(diff ** 2, dim=-1))

    rho = compute_rho_upward_tensor(exact_l2)
    norm_w = torch.sqrt(torch.sum(w_dequant_blocks ** 2, dim=-1))
    w_dequant = w_dequant_blocks.view(V, K)
    return w_dequant, rho, norm_w


def quantize_w_eisa_q4(w_float: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Bit-exact block-32 Q4_EISA quantization matching stdlib/llm/qwen3_quant_eisa.sio:
      scale = f16(max_abs / 7.0)
      q = clamp(round(w / scale), -8, 7)
      w_dequant = scale * q
      exact_l2 = sqrt(sum((w - w_dequant)^2))
      rho = compute_rho_upward(exact_l2)
    """
    V, K = w_float.shape
    w_blocks = w_float.view(V, K // 32, 32)
    max_abs = torch.amax(torch.abs(w_blocks), dim=-1, keepdim=True)
    s = (max_abs / 7.0).to(torch.float16).float()
    s = torch.where(s == 0.0, torch.ones_like(s), s)
    q = torch.clamp(torch.round(w_blocks / s), -8, 7)
    w_dequant_blocks = s * q
    diff = w_blocks - w_dequant_blocks
    exact_l2 = torch.sqrt(torch.sum(diff ** 2, dim=-1))

    rho = compute_rho_upward_tensor(exact_l2)
    norm_w = torch.sqrt(torch.sum(w_dequant_blocks ** 2, dim=-1))
    w_dequant = w_dequant_blocks.view(V, K)
    return w_dequant, rho, norm_w


# =============================================================================
# 3. Global Worst-Case Certifier & Cone Refiner Engine
# =============================================================================

def audit_cone_decisions(
    w_dequant: torch.Tensor,
    rho: torch.Tensor,
    norm_w: torch.Tensor,
    w_true_bf16: torch.Tensor,
    h_tokens_bf16: torch.Tensor,
    quant_label: str,
    device: torch.device,
) -> dict:
    """
    Runs Global Worst-Case Certifier and Cone Refiner on all teacher-forced tokens.
    Strictly asserts:
      stage1_false_certificates == 0
      stage2_cone_refined_flips == 0
    """
    V, K = w_dequant.shape
    N = h_tokens_bf16.shape[0]

    h_float = h_tokens_bf16.float().to(device)
    w_dequant_dev = w_dequant.to(device)
    w_true_dev = w_true_bf16.float().to(device)
    rho_dev = rho.to(device)
    norm_w_dev = norm_w.to(device)

    # Precompute norm_x per block of 32: (N, K // 32)
    h_blocks = h_float.view(N, K // 32, 32)
    norm_x = torch.sqrt(torch.sum(h_blocks ** 2, dim=-1)) # (N, K // 32)

    stage1_certified_count = 0
    stage1_false_certificates = 0
    stage2_cone_refined_flips = 0

    cone_sizes = []
    margins = []

    print(f"\n[{quant_label}] Running certification over {N} tokens on {device}...")
    t0 = time.time()

    CHUNK_SIZE = 128
    for start_idx in range(0, N, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, N)
        sub_h = h_float[start_idx:end_idx] # (B, K)
        sub_nx = norm_x[start_idx:end_idx] # (B, K // 32)

        # MVM: (V, K) @ (K, B) -> (V, B)
        val_batch = w_dequant_dev @ sub_h.t() # (V, B)
        true_batch = w_true_dev @ sub_h.t() # (V, B)

        # Error bound: sum_b [ rho * nx + 1e-5 * (norm_w + rho) * nx ]
        # (V, K // 32) @ (K // 32, B) -> (V, B)
        err_part1 = rho_dev @ sub_nx.t()
        err_part2 = 1e-5 * ((norm_w_dev + rho_dev) @ sub_nx.t())
        err_batch = err_part1 + err_part2 # (V, B)

        B = end_idx - start_idx
        for b in range(B):
            token_idx = start_idx + b
            val = val_batch[:, b]
            err = err_batch[:, b]
            z_true = true_batch[:, b]

            i_star = torch.argmax(val).item()
            true_star = torch.argmax(z_true).item()
            low_star = val[i_star] - err[i_star]

            upper = val + err
            # max other upper
            # Set upper[i_star] = -inf to find max over j != i_star
            upper_other = upper.clone()
            upper_other[i_star] = float("-inf")
            max_other = torch.max(upper_other).item()
            global_margin = (low_star - max_other).item()
            margins.append(global_margin)

            is_certified = global_margin > 0.0

            if is_certified:
                stage1_certified_count += 1
                cone_sizes.append(1)
                # Verify zero false certificate
                if i_star != true_star:
                    slack = (z_true[true_star] - z_true[i_star]).item()
                    print(
                        f"\n[CRITICAL ERROR] Stage 1 FALSE CERTIFICATE at token {token_idx}!"
                    )
                    print(
                        f"  i_star={i_star}, true_star={true_star}, margin={global_margin:.6f}, slack={slack:.6f}"
                    )
                    stage1_false_certificates += 1
                    raise AssertionError(
                        f"STAGE 1 FALSE CERTIFICATE DETECTED: token={token_idx}, margin={global_margin:.6f}, err={err[i_star].item():.6f}"
                    )
            else:
                # Stage 2 Cone Refiner
                cone_mask = upper >= low_star
                cone_size = cone_mask.sum().item()
                cone_sizes.append(cone_size)

                # Refine among candidates
                cone_cand_indices = torch.where(cone_mask)[0]
                z_cone = z_true[cone_cand_indices]
                refined_local_best = torch.argmax(z_cone).item()
                refined_top1_id = cone_cand_indices[refined_local_best].item()

                if refined_top1_id != true_star:
                    slack = (z_true[true_star] - z_true[refined_top1_id]).item()
                    print(
                        f"\n[CRITICAL ERROR] Stage 2 CONE REFINED FLIP at token {token_idx}!"
                    )
                    print(
                        f"  refined={refined_top1_id}, true_star={true_star}, cone_size={cone_size}, slack={slack:.6f}"
                    )
                    stage2_cone_refined_flips += 1
                    raise AssertionError(
                        f"STAGE 2 CONE REFINED FLIP DETECTED: token={token_idx}, cone_size={cone_size}"
                    )

    elapsed = time.time() - t0
    certified_frac = stage1_certified_count / N

    # Exact 95% one-sided Clopper-Pearson upper bound: 1 - 0.05^(1/N_cert)
    if stage1_certified_count > 0:
        clopper_pearson_95 = 1.0 - (0.05 ** (1.0 / stage1_certified_count))
    else:
        clopper_pearson_95 = 1.0

    cone_sizes_sorted = sorted(cone_sizes)
    mean_cone_size = sum(cone_sizes) / N
    median_cone_size = cone_sizes_sorted[N // 2]
    p99_idx = min(int(N * 0.99), N - 1)
    p99_cone_size = cone_sizes_sorted[p99_idx]

    bf16_bytes_touched_frac = mean_cone_size / V
    bandwidth_saved_pct = (1.0 - bf16_bytes_touched_frac) * 100.0

    print(f"[{quant_label}] Results over {N} tokens ({elapsed:.2f}s):")
    print(f"  - Certified Count: {stage1_certified_count}/{N} ({certified_frac * 100:.2f}%)")
    print(f"  - Stage 1 False Certificates: {stage1_false_certificates} (MUST BE 0)")
    print(f"  - 95% One-Sided Clopper-Pearson Upper Bound: {clopper_pearson_95:.6f} ({clopper_pearson_95*100:.4f}%)")
    print(f"  - Mean Cone Size: {mean_cone_size:.2f} rows / {V} vocab")
    print(f"  - Median Cone Size: {median_cone_size} rows")
    print(f"  - P99 Cone Size: {p99_cone_size} rows")
    print(f"  - BF16 LM Head Bandwidth Saved: {bandwidth_saved_pct:.2f}%")
    print(f"  - Stage 2 Cone Refined Flips: {stage2_cone_refined_flips} (MUST BE 0)")

    assert stage1_false_certificates == 0, f"False certificates > 0: {stage1_false_certificates}"
    assert stage2_cone_refined_flips == 0, f"Cone refined flips > 0: {stage2_cone_refined_flips}"

    return {
        "n_tokens": N,
        "stage1_certified_count": stage1_certified_count,
        "stage1_certified_frac": float(f"{certified_frac:.4f}"),
        "stage1_false_certificates": stage1_false_certificates,
        "clopper_pearson_95_upper_bound": float(f"{clopper_pearson_95:.6e}"),
        "mean_cone_size": float(f"{mean_cone_size:.2f}"),
        "median_cone_size": int(median_cone_size),
        "p99_cone_size": int(p99_cone_size),
        "bf16_lm_head_bytes_touched_frac": float(f"{bf16_bytes_touched_frac:.6f}"),
        "bf16_lm_head_bandwidth_saved_pct": round(bandwidth_saved_pct, 4),
        "stage2_cone_refined_flips": stage2_cone_refined_flips,
        "wall_time_seconds": float(f"{elapsed:.2f}"),
    }


# =============================================================================
# 4. Full 28-Layer Q8_EISA Backbone & Head Drift Audit
# =============================================================================

def run_full_28l_q8_audit(
    model_bf16: nn.Module,
    all_input_ids: torch.Tensor,
    w_dequant_q8_head: torch.Tensor,
    rho_q8_head: torch.Tensor,
    norm_w_q8_head: torch.Tensor,
    device: torch.device,
    mean_cone_size: float = 0.0,
) -> dict:
    """
    Quantizes all 28 linear layers + lm_head to Q8_EISA.
    Measures:
      empirical 28-layer hidden drift ||dh||_2,
      max logit drift ||z_q8_28l - z_bf16||_inf,
      raw uncalibrated false certificates vs calibrated e_total = e_head + ||w_i||_2 * eps_h_hat,
      total BF16 memory traffic saved vs 100%-BF16-verifying self-speculative decoding (Quasar).
    """
    print("\n" + "=" * 80)
    print("AUDIT: Full 28-Layer Q8_EISA Backbone Drift & Calibration")
    print("=" * 80)

    t0 = time.time()
    # Evaluate across first 512 tokens of the teacher-forcing corpus
    eval_ids = all_input_ids[:, :512].to(device)
    N_eval = eval_ids.shape[1]
    print(f"Evaluating 28-layer drift across {N_eval} teacher-forced tokens on {device}...")

    # 1. Unquantized BF16 backbone forward pass
    with torch.no_grad():
        out_bf16 = model_bf16(eval_ids, output_hidden_states=True)
        h_bf16_28 = out_bf16.hidden_states[-1][0].float() # (N_eval, 1024)

    # 2. In-place block-32 Q8 quantization of all 28 transformer layers
    t_q = time.time()
    w_row_norms_head = torch.norm(model_bf16.lm_head.weight.data.float(), p=2, dim=-1).to(device)

    for l_idx, layer in enumerate(model_bf16.model.layers):
        for name, lin in layer.named_modules():
            if isinstance(lin, nn.Linear):
                w = lin.weight.data.float()
                w_b = w.view(w.shape[0], w.shape[1] // 32, 32)
                m = torch.amax(torch.abs(w_b), dim=-1, keepdim=True)
                s = (m / 127.0).to(torch.float16).float()
                s = torch.where(s == 0.0, torch.ones_like(s), s)
                q = torch.clamp(torch.round(w_b / s), -127, 127)
                lin.weight.data = (s * q).view(w.shape).to(torch.bfloat16)

    print(f"All 28 layers quantized to block-32 Q8_EISA in {time.time() - t_q:.2f}s")

    # 3. 28-layer Q8 forward pass
    with torch.no_grad():
        out_q8 = model_bf16(eval_ids, output_hidden_states=True)
        h_q8_28 = out_q8.hidden_states[-1][0].float() # (N_eval, 1024)

    # 4. Measure hidden state drift ||dh||_2
    dh = h_q8_28 - h_bf16_28 # (N_eval, 1024)
    dh_norms = torch.sqrt(torch.sum(dh ** 2, dim=-1)) # (N_eval,)
    mean_dh = dh_norms.mean().item()
    median_dh = dh_norms.median().item()
    p99_dh = torch.quantile(dh_norms, 0.99).item()
    max_dh = dh_norms.max().item()
    eps_h_hat = max_dh # conservative envelope across evaluated positions

    print(f"28-Layer Hidden Drift ||dh||_2:")
    print(f"  - Mean: {mean_dh:.4f}, Median: {median_dh:.4f}, P99: {p99_dh:.4f}, Max: {max_dh:.4f}")

    # 5. Measure logit drift and test uncalibrated vs calibrated certifiers
    V = w_dequant_q8_head.shape[0]
    w_head_q8_dev = w_dequant_q8_head.to(device)
    w_head_bf16_dev = model_bf16.lm_head.weight.data.float().to(device)
    rho_dev = rho_q8_head.to(device)
    norm_w_dev = norm_w_q8_head.to(device)

    # Compute unquantized BF16 logits vs 28L-Q8 logits
    z_bf16 = (w_head_bf16_dev @ h_bf16_28.t()).t() # (N_eval, V)
    z_q8_28l = (w_head_q8_dev @ h_q8_28.t()).t() # (N_eval, V)

    logit_drift = torch.abs(z_q8_28l - z_bf16)
    max_logit_drift = logit_drift.max().item()
    mean_logit_drift = logit_drift.mean().item()
    print(f"Logit Drift ||z_q8_28l - z_bf16||_inf: Max={max_logit_drift:.4f}, Mean={mean_logit_drift:.4f}")

    # Compute head quantization error bound
    h_blocks_q8 = h_q8_28.view(N_eval, 1024 // 32, 32)
    norm_x_q8 = torch.sqrt(torch.sum(h_blocks_q8 ** 2, dim=-1)) # (N_eval, 32)
    e_head = (rho_dev @ norm_x_q8.t() + 1e-5 * ((norm_w_dev + rho_dev) @ norm_x_q8.t())).t() # (N_eval, V)

    # Raw uncalibrated certifier (only e_head, ignoring backbone drift)
    raw_uncalibrated_certified = 0
    raw_uncalibrated_false_certs = 0

    # Calibrated certifier: e_total = e_head + ||w_i||_2 * eps_h_hat
    calibrated_certified = 0
    calibrated_false_certs = 0

    lip_additive = (w_row_norms_head * eps_h_hat).unsqueeze(0) # (1, V)
    e_calibrated = e_head + lip_additive # (N_eval, V)

    for t in range(N_eval):
        true_star = torch.argmax(z_bf16[t]).item()

        # 1. Raw uncalibrated
        val_t = z_q8_28l[t]
        err_raw = e_head[t]
        i_star_raw = torch.argmax(val_t).item()
        low_raw = val_t[i_star_raw] - err_raw[i_star_raw]
        up_raw = val_t + err_raw
        up_raw_other = up_raw.clone()
        up_raw_other[i_star_raw] = float("-inf")
        margin_raw = (low_raw - torch.max(up_raw_other)).item()
        if margin_raw > 0:
            raw_uncalibrated_certified += 1
            if i_star_raw != true_star:
                raw_uncalibrated_false_certs += 1

        # 2. Calibrated
        err_cal = e_calibrated[t]
        i_star_cal = torch.argmax(val_t).item()
        low_cal = val_t[i_star_cal] - err_cal[i_star_cal]
        up_cal = val_t + err_cal
        up_cal_other = up_cal.clone()
        up_cal_other[i_star_cal] = float("-inf")
        margin_cal = (low_cal - torch.max(up_cal_other)).item()
        if margin_cal > 0:
            calibrated_certified += 1
            if i_star_cal != true_star:
                calibrated_false_certs += 1

    print(f"Raw Uncalibrated Head-Only Certifier:")
    print(f"  - Certified: {raw_uncalibrated_certified}/{N_eval} ({raw_uncalibrated_certified/N_eval*100:.2f}%)")
    print(f"  - False Certificates: {raw_uncalibrated_false_certs}")

    print(f"Lipschitz/Spectral Calibrated Certifier (e_total = e_head + ||w_i||_2 * eps_h_hat):")
    print(f"  - Certified: {calibrated_certified}/{N_eval} ({calibrated_certified/N_eval*100:.2f}%)")
    print(f"  - False Certificates: {calibrated_false_certs} (MUST BE 0)")

    assert calibrated_false_certs == 0, f"Calibrated false certificates > 0: {calibrated_false_certs}"

    # Quasar memory traffic comparison:
    # Quasar runs 100% BF16 verification: 28 layers * 2 bytes + 151936 * 1024 * 2 bytes
    # EISA runs Q8 (1 byte) backbone + certified head (0 BF16 rows touched when certified)
    # Measured parameter byte sizes (BF16, before quantization is applied in place above).
    # NOTE: lm_head may be tied to the embedding; we count the lm_head matrix once as a head read.
    layer_params = sum(p.numel() for p in model_bf16.model.layers.parameters())
    head_params = model_bf16.lm_head.weight.numel()
    V_head = model_bf16.lm_head.weight.shape[0]
    bytes_28l_bf16 = layer_params * 2
    bytes_lm_head_bf16 = head_params * 2
    quasar_bytes = bytes_28l_bf16 + bytes_lm_head_bf16
    # EISA: layers bf16->q8 (2x compression); head only touches the cone rows in BF16.
    eisa_layers_bytes = bytes_28l_bf16 / 2.0
    eisa_head_bytes = bytes_lm_head_bf16 * (mean_cone_size / V_head)
    eisa_bytes = eisa_layers_bytes + eisa_head_bytes
    total_bf16_saved_pct = (1.0 - eisa_bytes / quasar_bytes) * 100.0
    # Conservative variant: also charge a full Q8 scan of the head (1 byte/elem) for stage 1.
    eisa_bytes_with_q8_scan = eisa_bytes + head_params * 1.0
    total_saved_pct_with_q8_scan = (1.0 - eisa_bytes_with_q8_scan / quasar_bytes) * 100.0

    return {
        "n_tokens_evaluated": N_eval,
        "hidden_drift_dh": {
            "mean": float(f"{mean_dh:.4f}"),
            "median": float(f"{median_dh:.4f}"),
            "p99": float(f"{p99_dh:.4f}"),
            "max": float(f"{max_dh:.4f}"),
            "eps_h_hat": float(f"{eps_h_hat:.4f}"),
        },
        "logit_drift": {
            "max": float(f"{max_logit_drift:.4f}"),
            "mean": float(f"{mean_logit_drift:.4f}"),
        },
        "raw_uncalibrated_head_only": {
            "certified_count": raw_uncalibrated_certified,
            "certified_frac": float(f"{raw_uncalibrated_certified / N_eval:.4f}"),
            "false_certificates": raw_uncalibrated_false_certs,
        },
        "lipschitz_calibrated": {
            "certified_count": calibrated_certified,
            "certified_frac": float(f"{calibrated_certified / N_eval:.4f}"),
            "false_certificates": calibrated_false_certs,
        },
        "quasar_comparison": {
            "quasar_bf16_verification_overhead_pct": 100.0,
            "bytes_28l_bf16": int(bytes_28l_bf16),
            "bytes_lm_head_bf16": int(bytes_lm_head_bf16),
            "mean_cone_size_used": float(mean_cone_size),
            "eisa_total_memory_bandwidth_saved_pct": round(total_bf16_saved_pct, 4),
            "eisa_total_saved_pct_incl_q8_head_scan": round(total_saved_pct_with_q8_scan, 4),
        },
        "wall_time_seconds": float(f"{time.time() - t0:.2f}"),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    total_start = time.time()
    print("=" * 80)
    print("SOUNIO PHASE 7B: EISA BLOCK-32 QUANTIZATION & ARMAX CERTIFICATION AUDIT")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Selected computation device: {device}")

    # 1. Load tokenizer and build deterministic corpus
    print("\n[1/4] Loading Qwen3-0.6B tokenizer and constructing multi-domain corpus...")
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR), trust_remote_code=True)

    tokenized_domains = {}
    all_input_ids_list = []
    for domain_name, text in DOMAINS.items():
        ids = tokenizer(text, return_tensors="pt").input_ids
        tokenized_domains[domain_name] = {
            "char_len": len(text),
            "n_tokens": ids.shape[1],
        }
        all_input_ids_list.append(ids)
        print(f"  - {domain_name:24s}: {ids.shape[1]:4d} tokens ({len(text):4d} chars)")

    all_input_ids = torch.cat(all_input_ids_list, dim=1)
    total_corpus_tokens = all_input_ids.shape[1]
    print(f"Total multi-domain corpus: {total_corpus_tokens} tokens (>= 2,048 required!)")
    assert total_corpus_tokens >= 2048, f"Corpus tokens {total_corpus_tokens} < 2048"

    # 2. Load model and compute unquantized teacher-forced representations
    print("\n[2/4] Loading Qwen3-0.6B base model on MPS/CPU...")
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(CHECKPOINT_DIR),
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    print(f"Model loaded in {time.time() - t_load:.2f}s")

    model.eval()
    print("Computing unquantized teacher-forced hidden states...")
    with torch.no_grad():
        out = model(all_input_ids.to(device), output_hidden_states=True)
        # Final layer hidden states
        h_all_bf16 = out.hidden_states[-1][0] # (total_corpus_tokens, 1024)

    # 3. Quantize lm_head to Q8_EISA and Q4_EISA
    print("\n[3/4] Quantizing lm_head to bit-exact block-32 Q8_EISA and Q4_EISA...")
    w_head_float = model.lm_head.weight.data.float().cpu()
    w_head_bf16 = model.lm_head.weight.data.cpu()

    t_q8 = time.time()
    w_dequant_q8, rho_q8, norm_w_q8 = quantize_w_eisa_q8(w_head_float)
    print(f"Q8_EISA lm_head quantized in {time.time() - t_q8:.2f}s")

    t_q4 = time.time()
    w_dequant_q4, rho_q4, norm_w_q4 = quantize_w_eisa_q4(w_head_float)
    print(f"Q4_EISA lm_head quantized in {time.time() - t_q4:.2f}s")

    # Run q8_cone audit
    q8_audit = audit_cone_decisions(
        w_dequant=w_dequant_q8,
        rho=rho_q8,
        norm_w=norm_w_q8,
        w_true_bf16=w_head_bf16,
        h_tokens_bf16=h_all_bf16,
        quant_label="q8_cone",
        device=device,
    )

    # Run q4_cone audit
    q4_audit = audit_cone_decisions(
        w_dequant=w_dequant_q4,
        rho=rho_q4,
        norm_w=norm_w_q4,
        w_true_bf16=w_head_bf16,
        h_tokens_bf16=h_all_bf16,
        quant_label="q4_cone",
        device=device,
    )

    # 4. Full 28-Layer Q8_EISA Backbone Drift & Calibration Audit
    print("\n[4/4] Running Full 28-Layer Q8_EISA Backbone & Head Drift Audit...")
    full_28l_audit = run_full_28l_q8_audit(
        model_bf16=model,
        all_input_ids=all_input_ids,
        w_dequant_q8_head=w_dequant_q8,
        rho_q8_head=rho_q8,
        norm_w_q8_head=norm_w_q8,
        device=device,
        mean_cone_size=q8_audit["mean_cone_size"],
    )

    total_wall = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"PHASE 7B EISA AUDIT COMPLETE in {total_wall:.2f}s")
    print("=" * 80)

    final_report = {
        "title": "Phase 7B EISA Block-32 Quantization & Verification Audit Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_wall_time_seconds": float(f"{total_wall:.2f}"),
        "device": str(device),
        "corpus": {
            "total_tokens": total_corpus_tokens,
            "domains": tokenized_domains,
        },
        "q8_cone": q8_audit,
        "q4_cone": q4_audit,
        "full_28l_q8_eisa": full_28l_audit,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(final_report, indent=2)
    with open(REPORT_PATH, "w") as f:
        f.write(report_json)

    file_size_kb = REPORT_PATH.stat().st_size / 1024.0
    print(f"Saved audit report to: {REPORT_PATH} ({file_size_kb:.2f} KB, < 250 KB limit)")
    assert file_size_kb < 250.0, f"Report size {file_size_kb:.2f} KB exceeds 250 KB limit"

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
