#!/usr/bin/env python3
"""
Phase 7B Step 1 — Oracle Ceiling on the 512-token full_28l_q8_eisa Audit Set.

Tests the exact upper limit of ANY sound full-28L Q8 argmax certifier using the
true per-row logit drift d_j = |z_j(Q8) - z_j(BF16)| on the 512 teacher-forced
tokens from audit_phase7b_eisa_teacher_forcing.py.

Stop rule: if oracle ceiling < 20%, stop and report without writing Step 2 code.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_phase7b_eisa_teacher_forcing import (  # noqa: E402
    CHECKPOINT_DIR,
    DOMAINS,
    quantize_w_eisa_q8,
)

REPORT_PATH = Path(
    "/Users/demetriosagourakis/sounio-research/checkpoints/"
    "Qwen3-0.6B-ossm-sidecar/phase7b_step1_oracle_ceiling.json"
)
STEP1B_REPORT_PATH = Path(
    "/Users/demetriosagourakis/sounio-research/checkpoints/"
    "Qwen3-0.6B-ossm-sidecar/phase7b_step1b_bound_shapes.json"
)
STOP_RULE_THRESHOLD = 0.20
STOP_RULE_1B_THRESHOLD = 0.10


def summarize_dist(x: torch.Tensor) -> dict:
    x_cpu = x.detach().float().cpu()
    qs = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    qv = torch.quantile(x_cpu, qs).tolist()
    return {
        "min": round(float(x_cpu.min().item()), 6),
        "p05": round(float(qv[0]), 6),
        "p10": round(float(qv[1]), 6),
        "p25": round(float(qv[2]), 6),
        "p50": round(float(qv[3]), 6),
        "mean": round(float(x_cpu.mean().item()), 6),
        "std": round(float(x_cpu.std(unbiased=True).item()), 6),
        "p75": round(float(qv[4]), 6),
        "p90": round(float(qv[5]), 6),
        "p95": round(float(qv[6]), 6),
        "p99": round(float(qv[7]), 6),
        "max": round(float(x_cpu.max().item()), 6),
        "frac_gt_0_5": round(float((x_cpu > 0.5).float().mean().item()), 6),
        "frac_gt_1_0": round(float((x_cpu > 1.0).float().mean().item()), 6),
        "frac_gt_2_0": round(float((x_cpu > 2.0).float().mean().item()), 6),
    }


def eval_bound_shape(
    name: str,
    z_run: torch.Tensor,
    top: torch.Tensor,
    top1_match: torch.Tensor,
    true_drift: torch.Tensor,
    bound_mat: torch.Tensor,
    compute_top_ratio: bool = True,
) -> dict:
    n_eval, vocab_size = z_run.shape
    n_pairs = n_eval * vocab_size
    idx = torch.arange(n_eval, device=z_run.device)

    bound_dev = bound_mat.to(z_run.device)
    lhs = z_run[idx, top] - bound_dev[idx, top]
    upper = (z_run + bound_dev).clone()
    upper[idx, top] = float("-inf")
    rhs = torch.max(upper, dim=-1).values
    cert_mask = (lhs - rhs) > 0.0
    del upper

    cert_count = int(cert_mask.sum().item())
    cert_frac = cert_count / n_eval
    false_certs = int((cert_mask & (~top1_match)).sum().item())

    viol_mask = true_drift > bound_dev
    pair_violations = int(viol_mask.sum().item())
    pair_violation_rate = pair_violations / n_pairs
    tokens_any_viol = int(viol_mask.any(dim=-1).sum().item())
    del viol_mask

    top_true_drift = true_drift[idx, top]
    top_bound = bound_dev[idx, top]
    top_viol = int((top_true_drift > top_bound).sum().item())

    res = {
        "name": name,
        "n_tokens": n_eval,
        "n_pairs": n_pairs,
        "certified_count": cert_count,
        "certified_frac": round(cert_frac, 6),
        "certified_pct": round(cert_frac * 100.0, 2),
        "false_certificates": false_certs,
        "pair_violations": pair_violations,
        "pair_violation_rate": pair_violation_rate,
        "pair_violation_pct": pair_violation_rate * 100.0,
        "tokens_with_any_row_violation": tokens_any_viol,
        "top_row_violations": top_viol,
        "top_row_violation_rate": round(top_viol / n_eval, 6),
    }
    if compute_top_ratio:
        nonzero_mask = top_true_drift > 0.0
        ratio = top_bound[nonzero_mask] / top_true_drift[nonzero_mask]
        res["top_row_n_zero_true_drift"] = int((~nonzero_mask).sum().item())
        res["top_row_bound_over_true_drift"] = summarize_dist(ratio)
    del bound_dev
    return res


def main() -> None:
    t0 = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 80)
    print("PHASE 7B STEP 1 — ORACLE CEILING (FULL 28L Q8_EISA, 512 TOKENS)")
    print(f"Device: {device}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR), trust_remote_code=True)
    all_input_ids = torch.cat(
        [tokenizer(text, return_tensors="pt").input_ids for text in DOMAINS.values()],
        dim=1,
    )
    eval_ids = all_input_ids[:, :512].to(device)
    n_eval = eval_ids.shape[1]

    model = AutoModelForCausalLM.from_pretrained(
        str(CHECKPOINT_DIR),
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    with torch.no_grad():
        out_bf16 = model(eval_ids, output_hidden_states=True)
        h_bf16_28 = out_bf16.hidden_states[-1][0].float()

    w_head_float = model.lm_head.weight.data.float().cpu()
    w_head_q8, rho_q8, norm_w_q8 = quantize_w_eisa_q8(w_head_float)
    w_head_bf16_dev = model.lm_head.weight.data.float().to(device)
    w_head_q8_dev = w_head_q8.to(device)

    for layer in model.model.layers:
        for _, lin in layer.named_modules():
            if isinstance(lin, nn.Linear):
                w = lin.weight.data.float()
                w_b = w.view(w.shape[0], w.shape[1] // 32, 32)
                m = torch.amax(torch.abs(w_b), dim=-1, keepdim=True)
                s = (m / 127.0).to(torch.float16).float()
                s = torch.where(s == 0.0, torch.ones_like(s), s)
                q = torch.clamp(torch.round(w_b / s), -127, 127)
                lin.weight.data = (s * q).view(w.shape).to(torch.bfloat16)

    with torch.no_grad():
        out_q8 = model(eval_ids, output_hidden_states=True)
        h_q8_28 = out_q8.hidden_states[-1][0].float()

    dh_norms = torch.sqrt(torch.sum((h_q8_28 - h_bf16_28) ** 2, dim=-1))
    z_bf16 = (w_head_bf16_dev @ h_bf16_28.t()).t()  # (512, V)
    z_q8 = (w_head_q8_dev @ h_q8_28.t()).t()        # (512, V)
    d = torch.abs(z_q8 - z_bf16)                    # (512, V)

    max_logit_drift = float(d.max().item())
    mean_logit_drift = float(d.mean().item())
    max_dh = float(dh_norms.max().item())
    mean_dh = float(dh_norms.mean().item())

    idx = torch.arange(n_eval, device=device)
    top = torch.argmax(z_q8, dim=-1)
    true_top = torch.argmax(z_bf16, dim=-1)
    top1_match = (top == true_top)

    # Primary oracle certificate: z_top(Q8) - d_top > max_{j != top} (z_j(Q8) + d_j)
    upper_q8 = (z_q8 + d).clone()
    upper_q8[idx, top] = float("-inf")
    rhs_q8 = torch.max(upper_q8, dim=-1).values
    lhs_q8 = z_q8[idx, top] - d[idx, top]
    oracle_gap = lhs_q8 - rhs_q8
    oracle_pass = oracle_gap > 0.0
    oracle_false_certs = int((oracle_pass & (~top1_match)).sum().item())
    assert oracle_false_certs == 0, f"Soundness violation: false_certs={oracle_false_certs}"

    # BF16-centered oracle reference: z_true(BF16) - d_true > max_{j != true} (z_j(BF16) + d_j)
    upper_bf16 = (z_bf16 + d).clone()
    upper_bf16[idx, true_top] = float("-inf")
    rhs_bf16 = torch.max(upper_bf16, dim=-1).values
    lhs_bf16 = z_bf16[idx, true_top] - d[idx, true_top]
    bf16_centered_pass = (lhs_bf16 - rhs_bf16) > 0.0

    # Margins & (top-1 margin) / (max drift) distributions
    top2_q8 = torch.topk(z_q8, k=2, dim=-1).values
    margin_q8 = top2_q8[:, 0] - top2_q8[:, 1]
    top2_bf16 = torch.topk(z_bf16, k=2, dim=-1).values
    margin_bf16 = top2_bf16[:, 0] - top2_bf16[:, 1]

    token_max_drift = torch.max(d, dim=-1).values
    d_top = d[idx, top]
    ratio_q8_tokmax = margin_q8 / token_max_drift.clamp_min(1e-30)
    ratio_bf16_tokmax = margin_bf16 / token_max_drift.clamp_min(1e-30)
    ratio_q8_globalmax = margin_q8 / max(max_logit_drift, 1e-30)

    # Uniform token-max-drift certificate: z_top - d_max > max_{j != top}(z_j + d_max)
    scalar_tokmax_pass = (margin_q8 - 2.0 * token_max_drift) > 0.0

    pass_count = int(oracle_pass.sum().item())
    pass_frac = pass_count / n_eval
    match_count = int(top1_match.sum().item())
    match_frac = match_count / n_eval
    bf16_centered_count = int(bf16_centered_pass.sum().item())
    scalar_tokmax_count = int(scalar_tokmax_pass.sum().item())
    stop_triggered = pass_frac < STOP_RULE_THRESHOLD

    dist_ratio_q8_tokmax = summarize_dist(ratio_q8_tokmax)
    dist_ratio_bf16_tokmax = summarize_dist(ratio_bf16_tokmax)
    dist_ratio_q8_globalmax = summarize_dist(ratio_q8_globalmax)
    dist_margin_q8 = summarize_dist(margin_q8)
    dist_margin_bf16 = summarize_dist(margin_bf16)
    dist_tokmax_drift = summarize_dist(token_max_drift)
    dist_d_top = summarize_dist(d_top)
    dist_oracle_gap = summarize_dist(oracle_gap)

    wall_s = time.time() - t0

    print(f"Parity check vs Phase 7B: dh mean={mean_dh:.4f} max={max_dh:.4f} | "
          f"logit drift mean={mean_logit_drift:.4f} max={max_logit_drift:.4f}")
    print(f"Top-1 greedy agreement (z_q8 vs z_bf16): {match_count}/{n_eval} ({match_frac * 100:.2f}%)")
    print(f"ORACLE CEILING (z_top(Q8) - d_top > max_{{j!=top}}(z_j(Q8) + d_j)): "
          f"{pass_count}/{n_eval} ({pass_frac * 100:.2f}%), false_certs={oracle_false_certs}")
    print(f"BF16-centered oracle pass (reference):  "
          f"{bf16_centered_count}/{n_eval} ({bf16_centered_count / n_eval * 100:.2f}%)")
    print(f"Scalar token-max-drift pass (> 2.0x):   "
          f"{scalar_tokmax_count}/{n_eval} ({scalar_tokmax_count / n_eval * 100:.2f}%)")
    print("\nDistribution of (top-1 margin_q8) / (token max drift):")
    print(f"  min={dist_ratio_q8_tokmax['min']:.4f}, p05={dist_ratio_q8_tokmax['p05']:.4f}, "
          f"p10={dist_ratio_q8_tokmax['p10']:.4f}, p25={dist_ratio_q8_tokmax['p25']:.4f}, "
          f"p50={dist_ratio_q8_tokmax['p50']:.4f}, mean={dist_ratio_q8_tokmax['mean']:.4f}, "
          f"p75={dist_ratio_q8_tokmax['p75']:.4f}, p90={dist_ratio_q8_tokmax['p90']:.4f}, "
          f"p95={dist_ratio_q8_tokmax['p95']:.4f}, p99={dist_ratio_q8_tokmax['p99']:.4f}, "
          f"max={dist_ratio_q8_tokmax['max']:.4f}")
    print(f"  frac > 0.5: {dist_ratio_q8_tokmax['frac_gt_0_5']*100:.2f}%, "
          f"frac > 1.0: {dist_ratio_q8_tokmax['frac_gt_1_0']*100:.2f}%, "
          f"frac > 2.0: {dist_ratio_q8_tokmax['frac_gt_2_0']*100:.2f}%")
    print("\nDistribution of (top-1 margin_q8) / (global max drift = "
          f"{max_logit_drift:.4f}):")
    print(f"  min={dist_ratio_q8_globalmax['min']:.4f}, p10={dist_ratio_q8_globalmax['p10']:.4f}, "
          f"p25={dist_ratio_q8_globalmax['p25']:.4f}, p50={dist_ratio_q8_globalmax['p50']:.4f}, "
          f"mean={dist_ratio_q8_globalmax['mean']:.4f}, p75={dist_ratio_q8_globalmax['p75']:.4f}, "
          f"p90={dist_ratio_q8_globalmax['p90']:.4f}, p95={dist_ratio_q8_globalmax['p95']:.4f}, "
          f"max={dist_ratio_q8_globalmax['max']:.4f}")
    print(f"  frac > 1.0: {dist_ratio_q8_globalmax['frac_gt_1_0']*100:.2f}%, "
          f"frac > 2.0: {dist_ratio_q8_globalmax['frac_gt_2_0']*100:.2f}%")
    print(f"\nSTOP_RULE_THRESHOLD = {STOP_RULE_THRESHOLD:.2f}")
    print(f"STOP_RULE_TRIGGERED = {stop_triggered} "
          f"({'STOP (< 20%)' if stop_triggered else 'PROCEED (>= 20%)'})")
    print(f"Wall time: {wall_s:.2f}s")

    report = {
        "title": "Phase 7B Step 1 — Oracle Ceiling on Full 28L Q8_EISA",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "n_tokens": n_eval,
        "parity_check": {
            "dh_mean": round(mean_dh, 4),
            "dh_max": round(max_dh, 4),
            "logit_drift_mean": round(mean_logit_drift, 4),
            "logit_drift_max": round(max_logit_drift, 4),
        },
        "oracle_ceiling": {
            "passed_count": pass_count,
            "passed_frac": round(pass_frac, 6),
            "passed_pct": round(pass_frac * 100.0, 2),
            "false_certificates": oracle_false_certs,
            "top1_greedy_match_count": match_count,
            "top1_greedy_match_frac": round(match_frac, 6),
            "bf16_centered_passed_count": bf16_centered_count,
            "bf16_centered_passed_frac": round(bf16_centered_count / n_eval, 6),
            "scalar_token_max_drift_passed_count": scalar_tokmax_count,
            "scalar_token_max_drift_passed_frac": round(scalar_tokmax_count / n_eval, 6),
        },
        "distributions": {
            "margin_q8_over_token_max_drift": dist_ratio_q8_tokmax,
            "margin_bf16_over_token_max_drift": dist_ratio_bf16_tokmax,
            "margin_q8_over_global_max_drift": dist_ratio_q8_globalmax,
            "margin_q8": dist_margin_q8,
            "margin_bf16": dist_margin_bf16,
            "token_max_drift": dist_tokmax_drift,
            "d_top": dist_d_top,
            "oracle_gap": dist_oracle_gap,
        },
        "stop_rule": {
            "threshold_frac": STOP_RULE_THRESHOLD,
            "triggered": stop_triggered,
        },
        "wall_time_seconds": round(wall_s, 2),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved Step 1 report to: {REPORT_PATH}")

    # =========================================================================
    # STEP 1b — ORACLE CEILINGS BY BOUND SHAPE
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 7B STEP 1b — ORACLE CEILINGS BY BOUND SHAPE (SAME 512 TOKENS)")
    print("=" * 80)

    t1b = time.time()
    dim_d = h_bf16_28.shape[-1]
    sqrt_d = math.sqrt(dim_d)
    dh = h_q8_28 - h_bf16_28  # (512, 1024)
    w_norms = torch.norm(w_head_bf16_dev, p=2, dim=-1)  # (151936,)

    # Q8 lm_head Cauchy-Schwarz error e_head (for cross-check alongside bare bounds)
    h_blocks_q8 = h_q8_28.view(n_eval, dim_d // 32, 32)
    norm_x_q8 = torch.sqrt(torch.sum(h_blocks_q8 ** 2, dim=-1))
    rho_dev = rho_q8.to(device)
    norm_w_dev = norm_w_q8.to(device)
    e_head = (rho_dev @ norm_x_q8.t() + 1e-5 * ((norm_w_dev + rho_dev) @ norm_x_q8.t())).t()

    # Pure backbone reference: W @ h_q8 vs W @ h_bf16 (isolates dh without Q8 head drift)
    z_q8_bb = (w_head_bf16_dev @ h_q8_28.t()).t()
    d_bb = torch.abs(z_q8_bb - z_bf16)
    top_bb = torch.argmax(z_q8_bb, dim=-1)
    top1_match_bb = top_bb == true_top

    # (a) norm: d_j = ||w_j||_2 * ||dh||_2
    bound_a = torch.outer(dh_norms, w_norms)  # (512, 151936)
    res_a = eval_bound_shape("a_norm", z_q8, top, top1_match, d, bound_a, compute_top_ratio=True)
    res_a_plus_head = eval_bound_shape(
        "a_norm_plus_e_head", z_q8, top, top1_match, d, bound_a + e_head, compute_top_ratio=True
    )
    res_a_bb = eval_bound_shape(
        "a_norm_pure_backbone", z_q8_bb, top_bb, top1_match_bb, d_bb, bound_a, compute_top_ratio=True
    )

    # (b) box: d_j = sum_i |w_ji| * |dh_i| (exact IEEE float32 on CPU to avoid MPS rounding drift)
    bound_b_cpu = torch.abs(dh.cpu()) @ torch.abs(w_head_float).t()
    bound_b = bound_b_cpu.to(device)
    del bound_b_cpu
    res_b = eval_bound_shape("b_box", z_q8, top, top1_match, d, bound_b, compute_top_ratio=True)
    res_b_plus_head = eval_bound_shape(
        "b_box_plus_e_head", z_q8, top, top1_match, d, bound_b + e_head, compute_top_ratio=True
    )
    res_b_bb = eval_bound_shape(
        "b_box_pure_backbone", z_q8_bb, top_bb, top1_match_bb, d_bb, bound_b, compute_top_ratio=True
    )
    del bound_b, z_q8_bb, d_bb

    # (c) probabilistic: d_j = t * ||w_j||_2 * ||dh||_2 / sqrt(d), t in {2, 3, 4, 5, 6}
    res_c_list = []
    res_c_plus_head_list = []
    for t_mult in [2, 3, 4, 5, 6]:
        bound_c = bound_a * (float(t_mult) / sqrt_d)
        rc = eval_bound_shape(
            f"c_prob_t{t_mult}", z_q8, top, top1_match, d, bound_c, compute_top_ratio=True
        )
        rc_h = eval_bound_shape(
            f"c_prob_t{t_mult}_plus_e_head",
            z_q8,
            top,
            top1_match,
            d,
            bound_c + e_head,
            compute_top_ratio=False,
        )
        rc["t"] = t_mult
        rc_h["t"] = t_mult
        res_c_list.append(rc)
        res_c_plus_head_list.append(rc_h)
        del bound_c

    del bound_a, e_head

    det_closed = (res_a["certified_frac"] < STOP_RULE_1B_THRESHOLD) and (
        res_b["certified_frac"] < STOP_RULE_1B_THRESHOLD
    )
    wall_1b = time.time() - t1b

    print("Deterministic bounds on 512 tokens (77,791,232 (token, row) pairs):")
    for lbl, r, r_h, r_bb in [
        ("(a) norm  ||w_j||_2 * ||dh||_2", res_a, res_a_plus_head, res_a_bb),
        ("(b) box   sum_i |w_ji| * |dh_i|", res_b, res_b_plus_head, res_b_bb),
    ]:
        rat = r["top_row_bound_over_true_drift"]
        rat_bb = r_bb["top_row_bound_over_true_drift"]
        print(
            f"  {lbl}:\n"
            f"    certified={r['certified_count']}/{n_eval} ({r['certified_pct']:.2f}%), "
            f"false_certs={r['false_certificates']}, "
            f"pair_violations(|z_q8-z_bf16|)={r['pair_violations']}/{r['n_pairs']} "
            f"({r['pair_violation_pct']:.6f}%)\n"
            f"    +e_head: certified={r_h['certified_count']}/{n_eval} ({r_h['certified_pct']:.2f}%), "
            f"pair_violations={r_h['pair_violations']}/{r_h['n_pairs']}\n"
            f"    pure backbone (|W*dh|): certified={r_bb['certified_count']}/{n_eval} ({r_bb['certified_pct']:.2f}%), "
            f"pair_violations={r_bb['pair_violations']}/{r_bb['n_pairs']}\n"
            f"    top-row bound/true_drift: median={rat['p50']:.4f}, p95={rat['p95']:.4f} "
            f"(pure_bb median={rat_bb['p50']:.4f}, p95={rat_bb['p95']:.4f})"
        )

    print("\nProbabilistic bounds (c) d_j = t * ||w_j||_2 * ||dh||_2 / sqrt(1024):")
    print("  t | cert / 512 (  %  ) | pair_viols / 77,791,232 (  rate  ) | top_viol/512 | false_certs | +e_head cert (%) | +e_head pair_viol")
    for rc, rc_h in zip(res_c_list, res_c_plus_head_list):
        print(
            f"  {rc['t']} | "
            f"{rc['certified_count']:4d}/512 ({rc['certified_pct']:5.2f}%) | "
            f"{rc['pair_violations']:10d} ({rc['pair_violation_pct']:7.4f}%) | "
            f"{rc['top_row_violations']:4d}/512 ({rc['top_row_violation_rate']*100:5.2f}%) | "
            f"{rc['false_certificates']:11d} | "
            f"{rc_h['certified_count']:4d}/512 ({rc_h['certified_pct']:5.2f}%) | "
            f"{rc_h['pair_violations']:10d} ({rc_h['pair_violation_pct']:7.4f}%)"
        )

    print(f"\nSTOP_RULE_1B_THRESHOLD = {STOP_RULE_1B_THRESHOLD:.2f}")
    print(
        f"STOP_RULE_1B_TRIGGERED = {det_closed} "
        f"({'STOP — deterministic route closed (< 10%)' if det_closed else 'PROCEED (>= 10%)'})"
    )
    print(f"Step 1b wall time: {wall_1b:.2f}s")

    step1b_report = {
        "title": "Phase 7B Step 1b — Oracle Ceilings by Bound Shape",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "n_tokens": n_eval,
        "vocab_size": int(w_head_float.shape[0]),
        "hidden_dim": int(dim_d),
        "n_pairs": int(n_eval * w_head_float.shape[0]),
        "deterministic": {
            "a_norm": res_a,
            "a_norm_plus_e_head": res_a_plus_head,
            "a_norm_pure_backbone": res_a_bb,
            "b_box": res_b,
            "b_box_plus_e_head": res_b_plus_head,
            "b_box_pure_backbone": res_b_bb,
        },
        "probabilistic": {
            "bare": res_c_list,
            "plus_e_head": res_c_plus_head_list,
        },
        "stop_rule_1b": {
            "threshold_frac": STOP_RULE_1B_THRESHOLD,
            "triggered": det_closed,
        },
        "wall_time_seconds": round(wall_1b, 2),
    }

    with open(STEP1B_REPORT_PATH, "w") as f:
        json.dump(step1b_report, f, indent=2)
    print(f"Saved Step 1b report to: {STEP1B_REPORT_PATH}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
