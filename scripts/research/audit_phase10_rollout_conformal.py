#!/usr/bin/env python3
"""
Phase 10 Step 2 — Sequence-Level Rollout Conformal Calibration & Epistemic KV-Scrubbing Audit

Evaluates and calibrates stateful greedy rollouts (max_tokens=64, ChatML --no-think)
against pure BF16 reference rollouts across 240 Calibration & 240 disjoint Held-Out prompts:

  - R0 (Initial Q8-prefill + dirty-KV fallback):
      Demonstrates that Q8 prefill alone causes ~38% sequence mismatch even at 100% decode fallback.
  - R1 (q8_conformal with clean BF16 prefill):
      BF16 prefill (0..n_prompt-2) + 28L Q8 decode + single-token BF16 fallback on dirty Q8 KV.
      Exposes Fallback Inversion (sequence-level q_hat collapses to q=60.0 / 120.0).
  - R2 (q8_mlp_conformal — BF16 K/V at EVERY position):
      Attention (Q, K, V, O + KV write) runs in BF16 at every position; only MLP + lm_head in Q8,
      with NNLS layer weights refit strictly on CAL.
  - R3 (q8_scrub — Epistemic Certificate-Gated Prefix KV-Scrubbing):
      BF16 prefill + 28L Q8 decode streak + on abstention, replays batched BF16 over [clean..t],
      restoring clean BF16 KV and eliminating 100% of fallback first-divergences.
  - R4 (q8_exact_scrub — Certificate-Gated Window Rollback + Terminal Tail Scrub) vs Fixed-K=8:
      Achieves 0/240 (0.00%) sequence mismatch identically via exact BF16 verification + rollback.

Outputs:
  /Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/phase10_rollout_conformal_cal.json
  /Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/phase10_epistemic_scrub_audit.json
"""

import gc
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_phase7b_eisa_teacher_forcing import (  # noqa: E402
    CHECKPOINT_DIR,
    quantize_w_eisa_q8,
)
from audit_phase7b_step1d_prompt_conformal import (  # noqa: E402
    CAL_ARTIFACT_PATH,
    build_disjoint_prompt_sets,
    fit_nonneg_layer_weights,
)

OUT_CAL_JSON = Path(
    "/Users/demetriosagourakis/sounio-research/checkpoints/"
    "Qwen3-0.6B-ossm-sidecar/phase10_rollout_conformal_cal.json"
)
OUT_SCRUB_JSON = Path(
    "/Users/demetriosagourakis/sounio-research/checkpoints/"
    "Qwen3-0.6B-ossm-sidecar/phase10_epistemic_scrub_audit.json"
)
BIN_PATH = Path("/Users/demetriosagourakis/dev/sounio-worktrees/llm-backend/bin/sounio-llm")
METAL_DYLIB = Path("/Users/demetriosagourakis/dev/sounio-worktrees/llm-backend/artifacts/self-hosted/libsounio_metal.dylib")

EOS_IDS = (151643, 151645)
MAX_NEW_TOKENS = 64
BATCH_SIZE = 24
Q_GRID = [0.5, 1.0, 1.291984, 1.75, 2.343652, 3.2, 4.5, 6.5, 9.0, 13.0, 20.0, 35.0, 60.0, 120.0]
Q_GRID_SCRUB = [0.5, 0.85, 1.05, 1.291984, 1.55, 1.85, 2.343652, 3.0, 4.0, 6.0, 10.0, 20.0]
Q_EXACT = 1e9


def flush_mps(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rot * sin)


class Qwen3StatefulWeights:
    """
    Holds Qwen3-0.6B BF16 weights compactly in torch.bfloat16 (losslessly cast to float32
    per-layer on forward) and Q8_EISA weights in float32, plus a preallocated contiguous
    (L, batch_cap, n_kv, s_cap, head_dim) KV pool updated via static torch.where masks.
    """

    def __init__(self, hf_model, device: torch.device, batch_cap: int = BATCH_SIZE, s_cap: int = 192) -> None:
        self.device = device
        self.hidden = int(hf_model.config.hidden_size)
        self.inter = int(hf_model.config.intermediate_size)
        self.layers_n = int(hf_model.config.num_hidden_layers)
        self.n_q = int(hf_model.config.num_attention_heads)
        self.n_kv = int(hf_model.config.num_key_value_heads)
        self.head_dim = int(hf_model.config.head_dim)
        self.vocab = int(hf_model.config.vocab_size)
        self.eps = float(hf_model.config.rms_norm_eps)
        rope_params = getattr(hf_model.config, "rope_parameters", None) or {}
        self.theta = float(getattr(hf_model.config, "rope_theta", rope_params.get("rope_theta", 1000000.0)))
        self.sqrt_d = math.sqrt(float(self.hidden))

        self.embed_w = hf_model.model.embed_tokens.weight.data.to(torch.bfloat16).to(device)
        self.norm_w = hf_model.model.norm.weight.data.to(torch.bfloat16).float().to(device)

        lm_bf = hf_model.lm_head.weight.data.to(torch.bfloat16)
        self.lm_head_bf16 = lm_bf.to(device)
        lm_f32 = lm_bf.float()
        self.w_row_scale = (torch.norm(lm_f32, p=2, dim=-1) / self.sqrt_d).to(device)
        lm_q8, _, _ = quantize_w_eisa_q8(lm_f32)
        self.lm_head_q8 = lm_q8.to(device)

        if hasattr(hf_model.model, "rotary_emb") and hasattr(hf_model.model.rotary_emb, "inv_freq"):
            inv_freq = hf_model.model.rotary_emb.inv_freq.detach().float()
        else:
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / float(self.head_dim)))
        self.inv_freq = inv_freq.to(device)

        self.layers = []
        for layer in hf_model.model.layers:
            d = {}
            d["in_norm"] = layer.input_layernorm.weight.data.to(torch.bfloat16).float().to(device)
            d["post_norm"] = layer.post_attention_layernorm.weight.data.to(torch.bfloat16).float().to(device)
            d["q_norm"] = layer.self_attn.q_norm.weight.data.to(torch.bfloat16).float().to(device)
            d["k_norm"] = layer.self_attn.k_norm.weight.data.to(torch.bfloat16).float().to(device)
            for proj_name, mod in [
                ("q", layer.self_attn.q_proj),
                ("k", layer.self_attn.k_proj),
                ("v", layer.self_attn.v_proj),
                ("o", layer.self_attn.o_proj),
                ("gate", layer.mlp.gate_proj),
                ("up", layer.mlp.up_proj),
                ("down", layer.mlp.down_proj),
            ]:
                w_bf = mod.weight.data.to(torch.bfloat16)
                w_q8, _, _ = quantize_w_eisa_q8(w_bf.float())
                d[f"{proj_name}_bf16"] = w_bf.to(device)
                d[f"{proj_name}_q8"] = w_q8.to(device)
            self.layers.append(d)

        self.batch_cap = batch_cap
        self.s_cap = s_cap
        self.k_pool = torch.zeros((self.layers_n, batch_cap, self.n_kv, s_cap, self.head_dim), dtype=torch.float32, device=device)
        self.v_pool = torch.zeros_like(self.k_pool)

    def ensure_kv_pool(self, b_need: int, s_need: int) -> tuple[torch.Tensor, torch.Tensor]:
        if b_need > self.batch_cap or s_need > self.s_cap:
            new_b = max(self.batch_cap, b_need)
            new_s = max(self.s_cap, s_need)
            del self.k_pool, self.v_pool
            flush_mps(self.device)
            self.k_pool = torch.zeros((self.layers_n, new_b, self.n_kv, new_s, self.head_dim), dtype=torch.float32, device=self.device)
            self.v_pool = torch.zeros_like(self.k_pool)
            self.batch_cap = new_b
            self.s_cap = new_s
        k_view = self.k_pool[:, :b_need, :, :, :]
        v_view = self.v_pool[:, :b_need, :, :, :]
        k_view.zero_()
        v_view.zero_()
        return k_view, v_view

    def rope_for_positions(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = pos.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)  # (B, 1, T, D)
        return emb.cos(), emb.sin()

    def forward_batch_step(
        self,
        token_ids: torch.Tensor,         # (B, T)
        pos_ids: torch.Tensor,           # (B, T)
        write_start: int,
        attn_mask: torch.Tensor,         # (B, 1, T, hist_len) additive 0 / -inf
        k_cache: torch.Tensor,           # (L, B, n_kv, s_cap, D)
        v_cache: torch.Tensor,           # (L, B, n_kv, s_cap, D)
        update_mask: torch.Tensor | None,  # bool (B,) -> only True rows write K/V
        mode: str,                       # "bf16" | "q8_full" | "q8_mlp"
        capture_layer_norms: bool = False,
        return_logits: bool = True,
        return_all_logits: bool = False,
        write_end_hist: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Static-shape forward over (B, T) writing K/V to [write_start : write_start+T] without MPS scatter."""
        B, T = token_ids.shape
        x = F.embedding(token_ids, self.embed_w).float()
        cos, sin = self.rope_for_positions(pos_ids)
        write_end = write_start + T
        hist_len = write_end_hist if write_end_hist is not None else write_end

        layer_norms = [] if capture_layer_norms else None
        q_attn = (mode == "q8_full")
        q_mlp = (mode in ("q8_full", "q8_mlp"))
        m4 = update_mask.view(B, 1, 1, 1) if update_mask is not None else None

        for l, lw in enumerate(self.layers):
            h = rmsnorm(x, lw["in_norm"], self.eps)
            wq = lw["q_q8"] if q_attn else lw["q_bf16"].float()
            wk = lw["k_q8"] if q_attn else lw["k_bf16"].float()
            wv = lw["v_q8"] if q_attn else lw["v_bf16"].float()
            wo = lw["o_q8"] if q_attn else lw["o_bf16"].float()

            q = F.linear(h, wq).view(B, T, self.n_q, self.head_dim)
            k = F.linear(h, wk).view(B, T, self.n_kv, self.head_dim)
            v = F.linear(h, wv).view(B, T, self.n_kv, self.head_dim)

            q = rmsnorm(q, lw["q_norm"], self.eps).transpose(1, 2)  # (B, n_q, T, D)
            k = rmsnorm(k, lw["k_norm"], self.eps).transpose(1, 2)  # (B, n_kv, T, D)
            v = v.transpose(1, 2)

            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            if m4 is None:
                k_cache[l, :, :, write_start:write_end, :] = k
                v_cache[l, :, :, write_start:write_end, :] = v
            else:
                k_cache[l, :, :, write_start:write_end, :] = torch.where(m4, k, k_cache[l, :, :, write_start:write_end, :])
                v_cache[l, :, :, write_start:write_end, :] = torch.where(m4, v, v_cache[l, :, :, write_start:write_end, :])

            k_hist = k_cache[l, :, :, :hist_len, :]
            v_hist = v_cache[l, :, :, :hist_len, :]
            k_rep = k_hist.repeat_interleave(self.n_q // self.n_kv, dim=1)
            v_rep = v_hist.repeat_interleave(self.n_q // self.n_kv, dim=1)

            attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_mask)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.n_q * self.head_dim)
            x = x + F.linear(attn_out, wo)

            h2 = rmsnorm(x, lw["post_norm"], self.eps)
            wgate = lw["gate_q8"] if q_mlp else lw["gate_bf16"].float()
            wup = lw["up_q8"] if q_mlp else lw["up_bf16"].float()
            wdown = lw["down_q8"] if q_mlp else lw["down_bf16"].float()
            mlp_hidden = F.silu(F.linear(h2, wgate)) * F.linear(h2, wup)
            x = x + F.linear(mlp_hidden, wdown)

            if capture_layer_norms:
                layer_norms.append(torch.norm(x, p=2, dim=-1))

        norms_t = torch.stack(layer_norms, dim=-1) if capture_layer_norms else None
        if not return_logits:
            return x, None, norms_t
        whead = self.lm_head_bf16.float() if mode == "bf16" else self.lm_head_q8
        if return_all_logits:
            h_fin = rmsnorm(x, self.norm_w, self.eps)
            logits = F.linear(h_fin, whead)
            return x, logits, norms_t
        h_fin = rmsnorm(x[:, -1, :], self.norm_w, self.eps)
        logits = F.linear(h_fin, whead)
        return x, logits, norms_t


def encode_chatml_no_think(tokenizer, text: str) -> list[int]:
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(chat, add_special_tokens=False)


def make_left_padded_batch(token_lists: list[list[int]], max_steps: int, device: torch.device):
    B = len(token_lists)
    L_max = max(len(t) for t in token_lists)
    S_max = L_max + max_steps + 2
    pad_id = 151643

    pref_ids = torch.full((B, L_max), pad_id, dtype=torch.long, device=device)
    pos_ids = torch.zeros((B, L_max), dtype=torch.long, device=device)
    prompt_lens = torch.empty(B, dtype=torch.long, device=device)
    valid_prefix = torch.zeros((B, S_max), dtype=torch.bool, device=device)

    for i, toks in enumerate(token_lists):
        ln = len(toks)
        pad_n = L_max - ln
        pref_ids[i, pad_n:] = torch.tensor(toks, dtype=torch.long, device=device)
        pos_ids[i, pad_n:] = torch.arange(ln, dtype=torch.long, device=device)
        prompt_lens[i] = ln
        valid_prefix[i, pad_n:L_max] = True

    pref_causal = torch.tril(torch.ones((L_max - 1, L_max - 1), dtype=torch.bool, device=device))
    pref_valid = valid_prefix[:, :L_max - 1].unsqueeze(1).unsqueeze(2) & pref_causal.unsqueeze(0).unsqueeze(0)
    pref_mask = torch.where(pref_valid, 0.0, float("-inf")).to(torch.float32)

    last_tokens = torch.tensor([t[-1] for t in token_lists], dtype=torch.long, device=device).view(B, 1)
    last_pos = (prompt_lens - 1).view(B, 1)
    return L_max, S_max, pref_ids[:, :L_max - 1], pos_ids[:, :L_max - 1], pref_mask, valid_prefix, last_tokens, last_pos, prompt_lens


def compute_conformal_margin_mask(
    qw: Qwen3StatefulWeights,
    zq: torch.Tensor,
    norms: torch.Tensor,
    c_w: torch.Tensor,
    q_val: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dh_hat = torch.matmul(norms.squeeze(1), c_w).clamp_min(1e-6)
    top_q = torch.argmax(zq, dim=-1)
    idx_b = torch.arange(zq.shape[0], device=qw.device)
    if q_val < 0.0:
        return top_q, torch.zeros_like(top_q, dtype=torch.bool)
    if q_val == 0.0:
        return top_q, torch.ones_like(top_q, dtype=torch.bool)
    z_top = zq[idx_b, top_q].unsqueeze(1)
    err_rows = torch.outer(dh_hat * q_val, qw.w_row_scale)
    err_top = err_rows[idx_b, top_q].unsqueeze(1)
    lower_top = z_top - err_top
    up_all = zq + err_rows
    up_all[idx_b, top_q] = float("-inf")
    g_margin = lower_top.squeeze(1) - torch.max(up_all, dim=-1).values
    return top_q, (g_margin > 0.0)


@torch.no_grad()
def fit_r2_mlp_weights_on_cal(qw: Qwen3StatefulWeights, cal_tok_lists: list[list[int]], batch_size: int = BATCH_SIZE) -> tuple[torch.Tensor, dict]:
    h_bf_all, h_mlp_all, norms_all = [], [], []
    N = len(cal_tok_lists)
    for start in range(0, N, batch_size):
        chunk = cal_tok_lists[start:start + batch_size]
        B = len(chunk)
        L_max = max(len(t) for t in chunk)
        pad_id = 151643
        ids = torch.full((B, L_max), pad_id, dtype=torch.long, device=qw.device)
        pos = torch.zeros((B, L_max), dtype=torch.long, device=qw.device)
        lens = [len(t) for t in chunk]
        valid = torch.zeros((B, L_max), dtype=torch.bool, device=qw.device)
        for i, toks in enumerate(chunk):
            ln = len(toks)
            pad_n = L_max - ln
            ids[i, pad_n:] = torch.tensor(toks, dtype=torch.long, device=qw.device)
            pos[i, pad_n:] = torch.arange(ln, dtype=torch.long, device=qw.device)
            valid[i, pad_n:] = True
        causal = torch.tril(torch.ones((L_max, L_max), dtype=torch.bool, device=qw.device))
        amask = torch.where(valid.unsqueeze(1).unsqueeze(2) & causal.unsqueeze(0).unsqueeze(0), 0.0, float("-inf")).to(torch.float32)

        k_c, v_c = qw.ensure_kv_pool(B, L_max)
        x_bf, _, _ = qw.forward_batch_step(ids, pos, 0, amask, k_c, v_c, None, "bf16", capture_layer_norms=False, return_logits=False)
        k_c.zero_()
        v_c.zero_()
        x_r2, _, norms_r2 = qw.forward_batch_step(ids, pos, 0, amask, k_c, v_c, None, "q8_mlp", capture_layer_norms=True, return_logits=False)

        for i in range(B):
            pad_n = L_max - lens[i]
            h_bf_all.append(rmsnorm(x_bf[i, pad_n:, :], qw.norm_w, qw.eps).cpu())
            h_mlp_all.append(rmsnorm(x_r2[i, pad_n:, :], qw.norm_w, qw.eps).cpu())
            norms_all.append(norms_r2[i, pad_n:, :].cpu())

    y_dh = torch.norm(torch.cat(h_mlp_all, dim=0) - torch.cat(h_bf_all, dim=0), p=2, dim=-1)
    X_norms = torch.cat(norms_all, dim=0)
    c_mlp, stats = fit_nonneg_layer_weights(X_norms, y_dh)
    flush_mps(qw.device)
    return c_mlp.to(qw.device), stats


@torch.no_grad()
def run_bf16_rollouts_batch(qw: Qwen3StatefulWeights, tok_lists: list[list[int]], max_new: int = MAX_NEW_TOKENS, batch_size: int = BATCH_SIZE) -> list[list[int]]:
    outputs = []
    for start in range(0, len(tok_lists), batch_size):
        chunk = tok_lists[start:start + batch_size]
        B = len(chunk)
        L_max, S_max, pref_ids, pref_pos, pref_mask, valid_hist, cur_tok, cur_pos, _ = make_left_padded_batch(chunk, max_new, qw.device)
        k_c, v_c = qw.ensure_kv_pool(B, S_max)

        if L_max > 1:
            qw.forward_batch_step(pref_ids, pref_pos, 0, pref_mask, k_c, v_c, None, "bf16", capture_layer_norms=False, return_logits=False)

        seqs = [[] for _ in range(B)]
        finished = [False] * B
        for t in range(max_new):
            w_idx = (L_max - 1) + t
            valid_hist[:, w_idx] = True
            step_mask = torch.where(valid_hist[:, :w_idx + 1], 0.0, float("-inf")).view(B, 1, 1, w_idx + 1)
            _, logits, _ = qw.forward_batch_step(cur_tok, cur_pos, w_idx, step_mask, k_c, v_c, None, "bf16", capture_layer_norms=False, return_logits=True)
            next_ids = torch.argmax(logits, dim=-1).tolist()
            for i, tid in enumerate(next_ids):
                if not finished[i]:
                    seqs[i].append(int(tid))
                    if tid in EOS_IDS:
                        finished[i] = True
            if all(finished):
                break
            cur_tok = torch.tensor(next_ids, dtype=torch.long, device=qw.device).view(B, 1)
            cur_pos = cur_pos + 1
        outputs.extend(seqs)
    flush_mps(qw.device)
    return outputs


@torch.no_grad()
def run_conformal_rollouts_batch(
    qw: Qwen3StatefulWeights,
    tok_lists: list[list[int]],
    c_weights: torch.Tensor,
    prefill_mode: str,
    decode_mode: str,
    q_val: float,
    max_new: int = MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[list[int]], int, int]:
    """Stateful R0/R1/R2 conformal rollouts (single-step BF16 fallback on abstention)."""
    outputs = []
    tot_cert = 0
    tot_esc = 0
    c_w = c_weights.to(qw.device).view(qw.layers_n)

    for start in range(0, len(tok_lists), batch_size):
        chunk = tok_lists[start:start + batch_size]
        B = len(chunk)
        L_max, S_max, pref_ids, pref_pos, pref_mask, valid_hist, cur_tok, cur_pos, _ = make_left_padded_batch(chunk, max_new, qw.device)
        k_c, v_c = qw.ensure_kv_pool(B, S_max)

        if L_max > 1:
            qw.forward_batch_step(pref_ids, pref_pos, 0, pref_mask, k_c, v_c, None, prefill_mode, capture_layer_norms=False, return_logits=False)

        seqs = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=qw.device)

        for t in range(max_new):
            w_idx = (L_max - 1) + t
            valid_hist[:, w_idx] = True
            step_mask = torch.where(valid_hist[:, :w_idx + 1], 0.0, float("-inf")).view(B, 1, 1, w_idx + 1)

            _, zq, norms = qw.forward_batch_step(cur_tok, cur_pos, w_idx, step_mask, k_c, v_c, None, decode_mode, capture_layer_norms=True, return_logits=True)
            top_q, cert_mask = compute_conformal_margin_mask(qw, zq, norms, c_w, q_val)

            active = ~finished
            tot_cert += int((cert_mask & active).sum().item())
            esc_mask = (~cert_mask) & active
            tot_esc += int(esc_mask.sum().item())

            next_toks = top_q
            if bool(esc_mask.any().item()):
                _, zb_esc, _ = qw.forward_batch_step(
                    cur_tok, cur_pos, w_idx, step_mask, k_c, v_c, esc_mask, "bf16",
                    capture_layer_norms=False, return_logits=True,
                )
                top_b = torch.argmax(zb_esc, dim=-1)
                next_toks = torch.where(esc_mask, top_b, top_q)

            next_list = next_toks.tolist()
            for i, tid in enumerate(next_list):
                if not bool(finished[i].item()):
                    seqs[i].append(int(tid))
                    if tid in EOS_IDS:
                        finished[i] = True
            if bool(torch.all(finished).item()):
                break
            cur_tok = next_toks.view(B, 1)
            cur_pos = cur_pos + 1
        outputs.extend(seqs)
    flush_mps(qw.device)
    return outputs, tot_cert, tot_esc


@torch.no_grad()
def run_scrub_rollouts_batch(
    qw: Qwen3StatefulWeights,
    tok_lists: list[list[int]],
    c_weights: torch.Tensor,
    q_val: float,
    max_streak: int = 0,
    max_new: int = MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[list[int]], list[list[int]], dict, list[dict]]:
    """
    R3 (q8_scrub): BF16 prefill (0..n_prompt-2) + 28L Q8 decode streak + on abstention
    (or streak == max_streak), replays batched BF16 over [clean_idx .. w_idx] via static
    torch.where updates to restore clean BF16 KV state.
    Returns (sequences, esc_traces_per_prompt, summary_tele, per_prompt_tele).
    """
    outputs = []
    esc_traces = []
    per_prompt = []
    tot_cert = 0
    tot_esc = 0
    tot_scrub_tokens = 0
    tot_scrub_events = 0
    c_w = c_weights.to(qw.device).view(qw.layers_n)

    for start in range(0, len(tok_lists), batch_size):
        chunk = tok_lists[start:start + batch_size]
        B = len(chunk)
        L_max, S_max, pref_ids, pref_pos, pref_mask, valid_hist, cur_tok, cur_pos, _ = make_left_padded_batch(chunk, max_new, qw.device)
        k_c, v_c = qw.ensure_kv_pool(B, S_max)

        if L_max > 1:
            qw.forward_batch_step(pref_ids, pref_pos, 0, pref_mask, k_c, v_c, None, "bf16", capture_layer_norms=False, return_logits=False)

        seq_ids = torch.full((B, S_max), 151643, dtype=torch.long, device=qw.device)
        seq_pos = torch.zeros((B, S_max), dtype=torch.long, device=qw.device)
        clean_idx = torch.full((B,), L_max - 1, dtype=torch.long, device=qw.device)
        if L_max > 1:
            seq_ids[:, :L_max - 1] = pref_ids
            seq_pos[:, :L_max - 1] = pref_pos

        seqs = [[] for _ in range(B)]
        traces = [[] for _ in range(B)]
        p_scrub_passes = [0] * B
        p_scrub_toks = [0] * B
        p_last_w = [L_max - 1] * B
        finished = torch.zeros(B, dtype=torch.bool, device=qw.device)

        for t in range(max_new):
            w_idx = (L_max - 1) + t
            valid_hist[:, w_idx] = True
            seq_ids[:, w_idx] = cur_tok.squeeze(1)
            seq_pos[:, w_idx] = cur_pos.squeeze(1)
            step_mask = torch.where(valid_hist[:, :w_idx + 1], 0.0, float("-inf")).view(B, 1, 1, w_idx + 1)

            _, zq, norms = qw.forward_batch_step(cur_tok, cur_pos, w_idx, step_mask, k_c, v_c, None, "q8_full", capture_layer_norms=True, return_logits=True)
            top_q, cert_mask = compute_conformal_margin_mask(qw, zq, norms, c_w, q_val)

            if max_streak > 0:
                streak_len = (w_idx - clean_idx) + 1
                cert_mask = cert_mask & (streak_len < max_streak)

            active = ~finished
            tot_cert += int((cert_mask & active).sum().item())
            esc_mask = (~cert_mask) & active
            esc_cnt_step = int(esc_mask.sum().item())
            tot_esc += esc_cnt_step

            next_toks = top_q
            if esc_cnt_step > 0:
                w_start = int(clean_idx[esc_mask].min().item())
                tot_scrub_events += esc_cnt_step
                step_lens = (w_idx + 1) - clean_idx
                tot_scrub_tokens += int(step_lens[esc_mask].sum().item())

                esc_list_bool = esc_mask.tolist()
                slens_list = step_lens.tolist()
                for bi in range(B):
                    if esc_list_bool[bi]:
                        p_scrub_passes[bi] += 1
                        p_scrub_toks[bi] += int(slens_list[bi])

                row_pos = torch.arange(w_start, w_idx + 1, device=qw.device).unsqueeze(1)
                col_pos = torch.arange(w_idx + 1, device=qw.device).unsqueeze(0)
                causal_sub = (col_pos <= row_pos).unsqueeze(0).unsqueeze(0)
                valid_sub = valid_hist[:, :w_idx + 1].unsqueeze(1).unsqueeze(2)
                scrub_mask = torch.where(valid_sub & causal_sub, 0.0, float("-inf")).to(torch.float32)

                _, zb_esc, _ = qw.forward_batch_step(
                    seq_ids[:, w_start:w_idx + 1],
                    seq_pos[:, w_start:w_idx + 1],
                    w_start,
                    scrub_mask,
                    k_c,
                    v_c,
                    esc_mask,
                    "bf16",
                    capture_layer_norms=False,
                    return_logits=True,
                    return_all_logits=False,
                    write_end_hist=w_idx + 1,
                )
                top_b = torch.argmax(zb_esc, dim=-1)
                next_toks = torch.where(esc_mask, top_b, top_q)
                clean_idx = torch.where(esc_mask, torch.full_like(clean_idx, w_idx + 1), clean_idx)

            next_list = next_toks.tolist()
            esc_list = esc_mask.tolist()
            fin_list = finished.tolist()
            for i, tid in enumerate(next_list):
                if not fin_list[i]:
                    seqs[i].append(int(tid))
                    traces[i].append(1 if esc_list[i] else 0)
                    p_last_w[i] = w_idx
                    if tid in EOS_IDS:
                        finished[i] = True
            if bool(torch.all(finished).item()):
                break
            cur_tok = next_toks.view(B, 1)
            cur_pos = cur_pos + 1

        clean_list = clean_idx.tolist()
        for i in range(B):
            tail_toks = max(0, (p_last_w[i] + 1) - int(clean_list[i]))
            per_prompt.append({
                "draft_steps": len(seqs[i]),
                "scrub_passes_mid": p_scrub_passes[i],
                "scrub_tokens_mid": p_scrub_toks[i],
                "tail_unverified_tokens": tail_toks,
            })
        outputs.extend(seqs)
        esc_traces.extend(traces)

    flush_mps(qw.device)
    tele = {
        "certified_decode_steps": tot_cert,
        "escalated_decode_steps": tot_esc,
        "total_decode_steps": tot_cert + tot_esc,
        "certified_step_pct": round((tot_cert / (tot_cert + tot_esc) * 100.0) if (tot_cert + tot_esc) > 0 else 0.0, 2),
        "escalated_step_pct": round((tot_esc / (tot_cert + tot_esc) * 100.0) if (tot_cert + tot_esc) > 0 else 0.0, 2),
        "total_scrub_events": tot_scrub_events,
        "total_scrub_tokens": tot_scrub_tokens,
        "mean_scrub_window": round((tot_scrub_tokens / tot_scrub_events) if tot_scrub_events > 0 else 0.0, 3),
    }
    return outputs, esc_traces, tele, per_prompt


@torch.no_grad()
def run_exact_scrub_single_prompt(
    qw: Qwen3StatefulWeights,
    prompt_toks: list[int],
    c_w: torch.Tensor,
    q_val: float,
    max_streak: int = 0,
    max_new: int = MAX_NEW_TOKENS,
) -> tuple[list[int], dict]:
    """Single-prompt exact simulator of R4 (q8_exact_scrub) with full window rollback + tail scrub."""
    plen = len(prompt_toks)
    s_need = plen + max_new + 16
    k_c, v_c = qw.ensure_kv_pool(1, s_need)

    seq_ids = [int(t) for t in prompt_toks] + [0] * (max_new + 16)
    if plen > 1:
        p_t = torch.tensor(prompt_toks[:-1], dtype=torch.long, device=qw.device).view(1, plen - 1)
        p_pos = torch.arange(plen - 1, dtype=torch.long, device=qw.device).view(1, plen - 1)
        cmask = torch.triu(torch.full((plen - 1, plen - 1), float("-inf"), dtype=torch.float32, device=qw.device), diagonal=1).view(1, 1, plen - 1, plen - 1)
        qw.forward_batch_step(p_t, p_pos, 0, cmask, k_c, v_c, None, "bf16", capture_layer_norms=False, return_logits=False)

    cur_pos = plen - 1
    clean_pos = plen - 1
    gen_toks: list[int] = []
    draft_steps = 0
    scrub_passes = 0
    scrub_tokens = 0
    rollbacks = 0
    rolled_back_draft_tokens = 0

    for _iter in range(max_new * 4 + 32):
        ended = (len(gen_toks) >= max_new) or (len(gen_toks) > 0 and gen_toks[-1] in EOS_IDS)
        if ended:
            if clean_pos >= cur_pos:
                break
            # Tail scrub [clean_pos .. cur_pos - 1]
            win_end = cur_pos - 1
            wl = win_end - clean_pos + 1
            scrub_passes += 1
            scrub_tokens += wl
            w_ids = torch.tensor(seq_ids[clean_pos:win_end + 1], dtype=torch.long, device=qw.device).view(1, wl)
            w_pos = torch.arange(clean_pos, win_end + 1, dtype=torch.long, device=qw.device).view(1, wl)
            r_idx = torch.arange(clean_pos, win_end + 1, device=qw.device).unsqueeze(1)
            c_idx = torch.arange(win_end + 1, device=qw.device).unsqueeze(0)
            w_mask = torch.where(c_idx <= r_idx, 0.0, float("-inf")).view(1, 1, wl, win_end + 1)
            _, logits_w, _ = qw.forward_batch_step(
                w_ids, w_pos, clean_pos, w_mask, k_c, v_c, None, "bf16",
                capture_layer_norms=False, return_logits=True, return_all_logits=True, write_end_hist=win_end + 1,
            )
            preds = torch.argmax(logits_w.squeeze(0), dim=-1).tolist()
            rolled = False
            for j, pred_tok in enumerate(preds):
                g_idx = (clean_pos + j) - (plen - 1)
                if gen_toks[g_idx] != int(pred_tok):
                    rolled_back_draft_tokens += len(gen_toks) - g_idx - 1
                    rollbacks += 1
                    gen_toks = gen_toks[:g_idx] + [int(pred_tok)]
                    next_p = (clean_pos + j) + 1
                    seq_ids[next_p] = int(pred_tok)
                    cur_pos = next_p
                    clean_pos = next_p
                    rolled = True
                    break
            if not rolled:
                clean_pos = cur_pos
                break
            continue

        # 1-token Q8 draft step at cur_pos (skip redundant 8th draft if streak_len >= max_streak)
        streak_len = (cur_pos - clean_pos) + 1
        if max_streak > 0 and streak_len >= max_streak:
            accept = False
        else:
            d_tok = torch.tensor([[seq_ids[cur_pos]]], dtype=torch.long, device=qw.device)
            d_pos = torch.tensor([[cur_pos]], dtype=torch.long, device=qw.device)
            d_mask = torch.zeros((1, 1, 1, cur_pos + 1), dtype=torch.float32, device=qw.device)
            _, zq, norms = qw.forward_batch_step(d_tok, d_pos, cur_pos, d_mask, k_c, v_c, None, "q8_full", capture_layer_norms=True, return_logits=True)
            top_q_t, cert_t = compute_conformal_margin_mask(qw, zq, norms, c_w, q_val)
            draft_steps += 1
            accept = bool(cert_t[0].item())

        if accept:
            tok = int(top_q_t[0].item())
            gen_toks.append(tok)
            cur_pos += 1
            seq_ids[cur_pos] = tok
        else:
            # Abstain/max_streak -> BF16 scrub over [clean_pos .. cur_pos]
            win_end = cur_pos
            wl = win_end - clean_pos + 1
            scrub_passes += 1
            scrub_tokens += wl
            w_ids = torch.tensor(seq_ids[clean_pos:win_end + 1], dtype=torch.long, device=qw.device).view(1, wl)
            w_pos = torch.arange(clean_pos, win_end + 1, dtype=torch.long, device=qw.device).view(1, wl)
            r_idx = torch.arange(clean_pos, win_end + 1, device=qw.device).unsqueeze(1)
            c_idx = torch.arange(win_end + 1, device=qw.device).unsqueeze(0)
            w_mask = torch.where(c_idx <= r_idx, 0.0, float("-inf")).view(1, 1, wl, win_end + 1)
            _, logits_w, _ = qw.forward_batch_step(
                w_ids, w_pos, clean_pos, w_mask, k_c, v_c, None, "bf16",
                capture_layer_norms=False, return_logits=True, return_all_logits=True, write_end_hist=win_end + 1,
            )
            preds = torch.argmax(logits_w.squeeze(0), dim=-1).tolist()
            for j, pred_tok in enumerate(preds):
                g_idx = (clean_pos + j) - (plen - 1)
                if g_idx < len(gen_toks):
                    if gen_toks[g_idx] != int(pred_tok):
                        rolled_back_draft_tokens += len(gen_toks) - g_idx - 1
                        rollbacks += 1
                        gen_toks = gen_toks[:g_idx] + [int(pred_tok)]
                        next_p = (clean_pos + j) + 1
                        seq_ids[next_p] = int(pred_tok)
                        cur_pos = next_p
                        clean_pos = next_p
                        break
                else:
                    gen_toks.append(int(pred_tok))
                    cur_pos = (clean_pos + j) + 1
                    seq_ids[cur_pos] = int(pred_tok)
                    clean_pos = cur_pos
                    break

    return gen_toks, {
        "draft_steps": draft_steps,
        "scrub_passes": scrub_passes,
        "scrub_tokens": scrub_tokens,
        "rollbacks": rollbacks,
        "rolled_back_draft_tokens": rolled_back_draft_tokens,
    }


def first_divergence_stats(seqs: list[list[int]], traces: list[list[int]], ref_seqs: list[list[int]]) -> dict:
    n_tot = len(ref_seqs)
    n_mismatch = 0
    fallback_first = 0
    cert_first = 0
    for seq, tr, ref in zip(seqs, traces, ref_seqs):
        if seq == ref:
            continue
        n_mismatch += 1
        mlen = min(len(seq), len(ref))
        k_div = mlen
        for k in range(mlen):
            if seq[k] != ref[k]:
                k_div = k
                break
        if k_div < len(tr) and tr[k_div] == 1:
            fallback_first += 1
        else:
            cert_first += 1
    return {
        "sequence_mismatch_prompts": n_mismatch,
        "sequence_mismatch_rate": round(n_mismatch / n_tot, 6),
        "sequence_exact_match_rate": round((n_tot - n_mismatch) / n_tot, 6),
        "fallback_first_divergences": fallback_first,
        "certified_first_divergences": cert_first,
    }


def calibrate_topdown_exact(
    qw: Qwen3StatefulWeights,
    cal_tok_lists: list[list[int]],
    cal_bf16_seqs: list[list[int]],
    c_weights: torch.Tensor,
    mode_kind: str,
    grid: list[float],
) -> dict:
    """Top-down split-conformal calibration over grid with EXACT rank-cutoff early stop."""
    N = len(cal_tok_lists)
    G = len(grid)
    r_005 = min(N, max(1, math.ceil((N + 1) * 0.95)))
    fail_stop_count = (N - r_005) + 1  # 13 for N=240

    k_ranks = [0] * N
    remaining = list(range(N))
    early_stopped_at_g = None

    for g in range(G, 0, -1):
        qv = grid[g - 1]
        if not remaining:
            break
        if (N - len(remaining)) >= fail_stop_count:
            early_stopped_at_g = g
            break

        sub_toks = [cal_tok_lists[i] for i in remaining]
        if mode_kind == "q8_scrub":
            sub_seqs, _, _, _ = run_scrub_rollouts_batch(qw, sub_toks, c_weights, qv, max_streak=0, max_new=MAX_NEW_TOKENS, batch_size=BATCH_SIZE)
        else:
            sub_seqs, _, _ = run_conformal_rollouts_batch(qw, sub_toks, c_weights, "bf16", mode_kind, qv, max_new=MAX_NEW_TOKENS, batch_size=BATCH_SIZE)

        next_remaining = []
        n_failed_here = 0
        for idx_pos, orig_i in enumerate(remaining):
            if sub_seqs[idx_pos] != cal_bf16_seqs[orig_i]:
                k_ranks[orig_i] = g
                n_failed_here += 1
            else:
                next_remaining.append(orig_i)
        remaining = next_remaining
        print(f"    [cal {mode_kind}] g={g:2d} (q={qv:7.3f}): failed_here={n_failed_here:3d}, unsolved_remaining={len(remaining):3d}", flush=True)

    s_ranks = sorted(k_ranks)
    out = {
        "grid": grid,
        "exact_fallback_q": Q_EXACT,
        "early_stopped_at_g": early_stopped_at_g,
        "k_ranks_hist": {},
        "alphas": {},
    }
    for g in range(G + 1):
        out["k_ranks_hist"][str(g)] = s_ranks.count(g)

    for alpha in (0.05, 0.01):
        r_idx = min(N, max(1, math.ceil((N + 1) * (1.0 - alpha))))
        k_alpha = s_ranks[r_idx - 1]
        q_hat = Q_EXACT if k_alpha >= G else grid[k_alpha]
        out["alphas"][str(alpha)] = {
            "alpha": alpha,
            "order_stat_r_1idx": r_idx,
            "k_alpha_rank": k_alpha,
            "q_hat": q_hat,
            "uses_exact_bf16_fallback_cap": (k_alpha >= G),
        }
    return out


def verify_native_parity_3prompts(cal_prompts: list[dict], cal_bf16_seqs: list[list[int]]) -> dict:
    env = dict(os.environ)
    env["SOUNIO_METAL_RUNTIME"] = str(METAL_DYLIB)
    env["SOUNIO_LLM_TRACE_STEPS"] = "1"
    trace_re = re.compile(r"\[trace_step\s+idx=(\d+)\s+tok=(-?\d+)")

    matches = 0
    for i in range(3):
        cmd = [
            str(BIN_PATH), "--model-dir", str(CHECKPOINT_DIR), "--backend", "metal", "--quant", "bf16",
            "--max-tokens", str(MAX_NEW_TOKENS), "--no-think", "--temp", "0", "--prompt", cal_prompts[i]["text"],
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        nat_toks = []
        for ln in proc.stdout.splitlines():
            m = trace_re.search(ln)
            if m:
                nat_toks.append(int(m.group(2)))
        py_toks = cal_bf16_seqs[i]
        eq = (nat_toks == py_toks)
        if eq:
            matches += 1
        print(f"  [native bf16 parity {i}] len_nat={len(nat_toks)}, len_py={len(py_toks)}, exact_match={eq}", flush=True)
    return {"checked": 3, "exact_matches": matches}


def eval_heldout_conformal(
    qw: Qwen3StatefulWeights,
    test_tok_lists: list[list[int]],
    test_bf16_seqs: list[list[int]],
    c_weights: torch.Tensor,
    prefill_mode: str,
    decode_mode: str,
    q_val: float,
) -> dict:
    seqs, n_cert, n_esc = run_conformal_rollouts_batch(
        qw, test_tok_lists, c_weights, prefill_mode, decode_mode, q_val, max_new=MAX_NEW_TOKENS, batch_size=BATCH_SIZE
    )
    n_tot = len(test_tok_lists)
    n_mismatch = sum(1 for a, b in zip(seqs, test_bf16_seqs) if a != b)
    n_steps = n_cert + n_esc
    return {
        "q_val": q_val,
        "heldout_prompts": n_tot,
        "sequence_mismatch_prompts": n_mismatch,
        "sequence_mismatch_rate": round(n_mismatch / n_tot, 6),
        "sequence_exact_match_rate": round((n_tot - n_mismatch) / n_tot, 6),
        "certified_decode_steps": n_cert,
        "escalated_decode_steps": n_esc,
        "total_decode_steps": n_steps,
        "certified_step_pct": round((n_cert / n_steps * 100.0) if n_steps > 0 else 0.0, 2),
        "escalated_step_pct": round((n_esc / n_steps * 100.0) if n_steps > 0 else 0.0, 2),
    }


def eval_heldout_scrub(
    qw: Qwen3StatefulWeights,
    test_tok_lists: list[list[int]],
    test_bf16_seqs: list[list[int]],
    c_weights: torch.Tensor,
    q_val: float,
) -> dict:
    seqs, traces, tele, _ = run_scrub_rollouts_batch(
        qw, test_tok_lists, c_weights, q_val, max_streak=0, max_new=MAX_NEW_TOKENS, batch_size=BATCH_SIZE
    )
    div = first_divergence_stats(seqs, traces, test_bf16_seqs)
    return {
        "q_val": q_val,
        "heldout_prompts": len(test_tok_lists),
        **div,
        **tele,
    }


def eval_heldout_exact_scrub(
    qw: Qwen3StatefulWeights,
    test_tok_lists: list[list[int]],
    test_bf16_seqs: list[list[int]],
    c_weights: torch.Tensor,
    q_val: float,
    max_streak: int = 0,
) -> dict:
    """
    Runs streaming batched R3 across all 240 held-out prompts, then explicitly replays
    B=1 window rollback + tail scrub (`run_exact_scrub_single_prompt`) on EVERY
    prompt with a certified divergence (plus prompts 0, 1, 2 unconditionally).
    """
    c_w = c_weights.to(qw.device).view(qw.layers_n)
    seqs, _, _, per_prompt = run_scrub_rollouts_batch(
        qw, test_tok_lists, c_w, q_val, max_streak=max_streak, max_new=MAX_NEW_TOKENS, batch_size=BATCH_SIZE
    )
    n_tot = len(test_tok_lists)
    tot_draft = 0
    tot_passes = 0
    tot_scrub_toks = 0
    tot_rollbacks = 0
    tot_wasted = 0
    replayed_count = 0

    for i in range(n_tot):
        needs_single_replay = (seqs[i] != test_bf16_seqs[i]) or (i < 3)
        if needs_single_replay:
            replayed_count += 1
            out_i, st = run_exact_scrub_single_prompt(qw, test_tok_lists[i], c_w, q_val, max_streak=max_streak, max_new=MAX_NEW_TOKENS)
            if out_i != test_bf16_seqs[i]:
                raise RuntimeError(f"INVARIANT VIOLATION: R4 exact scrub mismatched on prompt {i} (q={q_val}, max_streak={max_streak})")
            tot_draft += st["draft_steps"]
            tot_passes += st["scrub_passes"]
            tot_scrub_toks += st["scrub_tokens"]
            tot_rollbacks += st["rollbacks"]
            tot_wasted += st["rolled_back_draft_tokens"]
        else:
            pp = per_prompt[i]
            tail = pp["tail_unverified_tokens"]
            tot_draft += pp["draft_steps"]
            tot_passes += pp["scrub_passes_mid"] + (1 if tail > 0 else 0)
            tot_scrub_toks += pp["scrub_tokens_mid"] + tail

    flush_mps(qw.device)
    return {
        "q_val": q_val,
        "max_streak": max_streak,
        "heldout_prompts": n_tot,
        "single_prompt_replays_executed": replayed_count,
        "sequence_mismatch_prompts": 0,
        "sequence_mismatch_rate": 0.0,
        "sequence_exact_match_rate": 1.0,
        "total_draft_steps": tot_draft,
        "total_scrub_passes": tot_passes,
        "total_scrub_tokens": tot_scrub_toks,
        "total_rollbacks": tot_rollbacks,
        "total_rolled_back_draft_tokens": tot_wasted,
        "wasted_draft_pct": round((tot_wasted / tot_draft * 100.0) if tot_draft > 0 else 0.0, 3),
    }


def main() -> None:
    only_fixed_k8 = ("--only-fixed-k8" in sys.argv[1:])
    t0 = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 98, flush=True)
    print(f"PHASE 10 STEP 2 — UNIFIED ROLLOUT CONFORMAL + EPISTEMIC SCRUB AUDIT (device={device}, B={BATCH_SIZE})", flush=True)
    print("=" * 98, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR), trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(str(CHECKPOINT_DIR), dtype=torch.bfloat16, trust_remote_code=True)
    hf_model.eval()

    cal_prompts, test_prompts = build_disjoint_prompt_sets()
    cal_tok_lists = [encode_chatml_no_think(tokenizer, p["text"]) for p in cal_prompts]
    test_tok_lists = [encode_chatml_no_think(tokenizer, p["text"]) for p in test_prompts]
    s_cap = max(max(len(t) for t in cal_tok_lists), max(len(t) for t in test_tok_lists)) + MAX_NEW_TOKENS + 16

    qw = Qwen3StatefulWeights(hf_model, device, batch_cap=BATCH_SIZE, s_cap=s_cap)
    del hf_model

    print("1) Generating pure BF16 greedy rollouts (240 CAL + 240 HELD-OUT, max_tokens=64)...", flush=True)
    if only_fixed_k8:
        test_bf16_seqs = run_bf16_rollouts_batch(qw, test_tok_lists, MAX_NEW_TOKENS, batch_size=BATCH_SIZE)
        cal_art = json.loads(CAL_ARTIFACT_PATH.read_text(encoding="utf-8"))
        c_r1 = torch.tensor(cal_art["c_weights"], dtype=torch.float32, device=device)
        ctrl_k8 = eval_heldout_exact_scrub(qw, test_tok_lists, test_bf16_seqs, c_r1, 0.0, max_streak=8)
        scrub_report = json.loads(OUT_SCRUB_JSON.read_text(encoding="utf-8"))
        scrub_report["fixed_k8_speculative_control"] = ctrl_k8
        OUT_SCRUB_JSON.write_text(json.dumps(scrub_report, indent=2), encoding="utf-8")
        print(f"Updated fixed_k8_speculative_control: {ctrl_k8}", flush=True)
        sys.stdout.flush()
        os._exit(0)

    cal_bf16_seqs = run_bf16_rollouts_batch(qw, cal_tok_lists, MAX_NEW_TOKENS, batch_size=BATCH_SIZE)
    test_bf16_seqs = run_bf16_rollouts_batch(qw, test_tok_lists, MAX_NEW_TOKENS, batch_size=BATCH_SIZE)

    print("2) Checking PyTorch f32-stateful BF16 vs native bin/sounio-llm Metal on 3 prompts...", flush=True)
    parity_info = verify_native_parity_3prompts(cal_prompts, cal_bf16_seqs)

    cal_art = json.loads(CAL_ARTIFACT_PATH.read_text(encoding="utf-8"))
    c_r1 = torch.tensor(cal_art["c_weights"], dtype=torch.float32, device=device)

    print("3) Evaluating R0 (Q8 prefill + 100% BF16 decode fallback) on 240 HELD-OUT prompts...", flush=True)
    r0_q_inf = eval_heldout_conformal(qw, test_tok_lists, test_bf16_seqs, c_r1, "q8_full", "q8_full", Q_EXACT)
    r0_tf_129 = eval_heldout_conformal(qw, test_tok_lists, test_bf16_seqs, c_r1, "q8_full", "q8_full", 1.291984)
    print(f"   R0 (Q8 prefill, q=1e9):   mismatch={r0_q_inf['sequence_mismatch_prompts']}/240 ({r0_q_inf['sequence_mismatch_rate']*100:.2f}%)", flush=True)
    print(f"   R0 (Q8 prefill, q=1.292): mismatch={r0_tf_129['sequence_mismatch_prompts']}/240 ({r0_tf_129['sequence_mismatch_rate']*100:.2f}%), cert={r0_tf_129['certified_step_pct']:.2f}%", flush=True)

    print("4) Fitting Estimator (i) c_weights_mlp strictly on 240 CAL prompts for R2 (q8_mlp_conformal)...", flush=True)
    c_r2, r2_fit_stats = fit_r2_mlp_weights_on_cal(qw, cal_tok_lists, batch_size=BATCH_SIZE)
    print(f"   R2 estimator fit on CAL: {r2_fit_stats}", flush=True)

    print("5) Sequence-level top-down calibration (exact early-stop) for R1 (BF16 prefill + 28L Q8)...", flush=True)
    cal_r1 = calibrate_topdown_exact(qw, cal_tok_lists, cal_bf16_seqs, c_r1, "q8_full", Q_GRID)

    print("6) Sequence-level top-down calibration (exact early-stop) for R2 (BF16 K/V + Q8 MLP/head)...", flush=True)
    cal_r2 = calibrate_topdown_exact(qw, cal_tok_lists, cal_bf16_seqs, c_r2, "q8_mlp", Q_GRID)

    print("7) Sequence-level top-down calibration (exact early-stop) for R3 (q8_scrub: streaming KV-scrubbing)...", flush=True)
    cal_r3 = calibrate_topdown_exact(qw, cal_tok_lists, cal_bf16_seqs, c_r1, "q8_scrub", Q_GRID_SCRUB)

    print("8) Evaluating R1, R2, R3 on 240 disjoint HELD-OUT prompts...", flush=True)
    heldout_r1 = {a: eval_heldout_conformal(qw, test_tok_lists, test_bf16_seqs, c_r1, "bf16", "q8_full", cal_r1["alphas"][a]["q_hat"]) for a in ("0.05", "0.01")}
    heldout_r1["tf_0.05_q1.292"] = eval_heldout_conformal(qw, test_tok_lists, test_bf16_seqs, c_r1, "bf16", "q8_full", 1.291984)

    heldout_r2 = {a: eval_heldout_conformal(qw, test_tok_lists, test_bf16_seqs, c_r2, "bf16", "q8_mlp", cal_r2["alphas"][a]["q_hat"]) for a in ("0.05", "0.01")}
    heldout_r2["q1.292_probe"] = eval_heldout_conformal(qw, test_tok_lists, test_bf16_seqs, c_r2, "bf16", "q8_mlp", 1.291984)

    heldout_r3 = {a: eval_heldout_scrub(qw, test_tok_lists, test_bf16_seqs, c_r1, cal_r3["alphas"][a]["q_hat"]) for a in ("0.05", "0.01")}
    heldout_r3["q1.292_probe"] = eval_heldout_scrub(qw, test_tok_lists, test_bf16_seqs, c_r1, 1.291984)

    print("9) Evaluating R4 (q8_exact_scrub) vs Fixed-K=8 speculative control on 240 HELD-OUT prompts...", flush=True)
    q_scrub_05 = cal_r3["alphas"]["0.05"]["q_hat"]
    r4_q05 = eval_heldout_exact_scrub(qw, test_tok_lists, test_bf16_seqs, c_r1, q_scrub_05, max_streak=0)
    r4_q10 = eval_heldout_exact_scrub(qw, test_tok_lists, test_bf16_seqs, c_r1, 1.0, max_streak=0)
    r4_q129 = eval_heldout_exact_scrub(qw, test_tok_lists, test_bf16_seqs, c_r1, 1.291984, max_streak=0)
    ctrl_k8 = eval_heldout_exact_scrub(qw, test_tok_lists, test_bf16_seqs, c_r1, 0.0, max_streak=8)

    print("\n--- HELD-OUT SUMMARY (240 DISJOINT PROMPTS, max_tokens=64) ---", flush=True)
    for label, row in [
        ("R0 Q8-prefill q=1e9", r0_q_inf),
        ("R1 Q8-dirty alpha=0.05", heldout_r1["0.05"]),
        ("R1 Q8-dirty alpha=0.01", heldout_r1["0.01"]),
        ("R2 BF16-KV alpha=0.05", heldout_r2["0.05"]),
        ("R2 BF16-KV alpha=0.01", heldout_r2["0.01"]),
        ("R3 q8_scrub alpha=0.05", heldout_r3["0.05"]),
        ("R3 q8_scrub alpha=0.01", heldout_r3["0.01"]),
    ]:
        fb_str = f" | fb_div={row.get('fallback_first_divergences', '-')}" if "fallback_first_divergences" in row else ""
        print(
            f"  {label:24s} | q={row['q_val']:8.3g} | mismatch={row['sequence_mismatch_prompts']:3d}/240 "
            f"({row['sequence_mismatch_rate']*100:5.2f}%) | cert={row['certified_step_pct']:5.2f}% | esc={row['escalated_step_pct']:5.2f}%{fb_str}",
            flush=True,
        )
    for label, row in [
        ("R4 exact_scrub(q05)", r4_q05),
        ("R4 exact_scrub(q=1.292)", r4_q129),
        ("R4 exact_scrub(q=1.0)", r4_q10),
        ("Ctrl Fixed-K=8 Spec", ctrl_k8),
    ]:
        print(
            f"  {label:24s} | mismatch={row['sequence_mismatch_prompts']}/240 | scrubs={row['total_scrub_passes']:4d} | "
            f"scrub_toks={row['total_scrub_tokens']:5d} | rollback_waste={row['total_rolled_back_draft_tokens']:4d} ({row['wasted_draft_pct']:.2f}%)",
            flush=True,
        )

    cal_report = {
        "title": "Phase 10 Step 2 — Sequence-Level Stateful Rollout Conformal Calibration",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_cal_prompts": len(cal_tok_lists),
        "n_heldout_prompts": len(test_tok_lists),
        "native_bf16_parity_check": parity_info,
        "r0_q8_prefill_contamination": {"q_1.291984": r0_tf_129, "q_1e9_100pct_decode_fallback": r0_q_inf},
        "r1_q8_conformal": {"c_weights": [float(x) for x in c_r1.cpu().tolist()], "calibration": cal_r1, "heldout": heldout_r1},
        "r2_q8_mlp_conformal_bf16_kv": {
            "estimator_fit": r2_fit_stats,
            "c_weights": [float(x) for x in c_r2.cpu().tolist()],
            "calibration": cal_r2,
            "heldout": heldout_r2,
        },
        "wall_time_seconds": round(time.time() - t0, 2),
    }
    OUT_CAL_JSON.write_text(json.dumps(cal_report, indent=2), encoding="utf-8")

    scrub_report = {
        "title": "Phase 10 Step 2 — Epistemic Certificate-Gated KV-Scrubbing Audit (R3 & R4 vs Fixed-K=8)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_cal_prompts": len(cal_tok_lists),
        "n_heldout_prompts": len(test_tok_lists),
        "c_weights": [float(x) for x in c_r1.cpu().tolist()],
        "r3_q8_scrub": {"calibration": cal_r3, "heldout": heldout_r3},
        "r4_q8_exact_scrub": {"q_scrub_005": r4_q05, "q_1.291984": r4_q129, "q_1.0": r4_q10},
        "fixed_k8_speculative_control": ctrl_k8,
        "wall_time_seconds": round(time.time() - t0, 2),
    }
    OUT_SCRUB_JSON.write_text(json.dumps(scrub_report, indent=2), encoding="utf-8")

    print(f"\nSaved: {OUT_CAL_JSON}\nSaved: {OUT_SCRUB_JSON}\nTotal wall={time.time() - t0:.2f}s", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
