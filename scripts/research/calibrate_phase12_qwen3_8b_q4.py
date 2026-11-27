#!/usr/bin/env python3
"""
Phase 12 — Jacobian-Scaled Q4 Conformal Oracle, Rollout Calibration & Held-Out Audit

Subcommands:
  --stage nnls-8b    Compare Phase 11 linear proxy vs Jacobian-scaled proxy (inv_rms * (c_add + C*norms))
                     on Qwen3-8B (240 CAL prompts, 2-pass) and enforce kill-switch rule.
  --stage rollout    Pure native bin/sounio-llm sweep on Qwen3-8B 240 CAL prompts (NO PyTorch),
                     measuring fused BF16 + parity vs Phase 11 BF16 refs.
  --stage audit      Pure native bin/sounio-llm audit on Qwen3-8B 240 HELD-OUT prompts (NO PyTorch).
"""

import argparse
import gc
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_phase7b_step1d_prompt_conformal import build_disjoint_prompt_sets  # noqa: E402

REPO_ROOT = Path("/Users/demetriosagourakis/dev/sounio-worktrees/llm-backend")
BIN_PATH = REPO_ROOT / "bin/sounio-llm"
METAL_DYLIB = REPO_ROOT / "artifacts/self-hosted/libsounio_metal.dylib"
MODEL_8B = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-8B")

WEIGHTS_8B_P12_TXT = Path("/tmp/qwen3_8b_q4_p12_weights.txt")
WEIGHTS_8B_P12_META = Path("/tmp/qwen3_8b_q4_p12_nnls_meta.json")
P11_BF16_REFS = Path("/tmp/phase11_qwen3_8b_bf16_refs.json")
P12_BF16_REFS = Path("/tmp/phase12_qwen3_8b_bf16_refs.json")
OUT_CAL_JSON = Path("/tmp/phase12_qwen3_8b_q4_cal.json")
OUT_AUDIT_JSON = Path(
    "/Users/demetriosagourakis/sounio-research/checkpoints/"
    "Qwen3-0.6B-ossm-sidecar/phase12_qwen3_8b_q4_audit.json"
)
CAL_PROMPTS_TXT = Path("/tmp/phase12_cal240_prompts.txt")
CAL20_PROMPTS_TXT = Path("/tmp/phase12_cal20_prompts.txt")
HELDOUT_PROMPTS_TXT = Path("/tmp/phase12_heldout240_prompts.txt")

BENCH_TOKENS_RE = re.compile(r"^\[bench_tokens idx=(\d+)\s+n=(\d+)\s+ids=([0-9,-]*)\]$")
BENCH_PROMPT_RE = re.compile(
    r"^\[bench_prompt idx=(\d+)\s+prompt_tokens=(\d+)\s+gen_tokens=(\d+)\s+"
    r"prefill_ms=([-+0-9.eE]+)\s+decode_ms=([-+0-9.eE]+)\s+decode_tok_per_sec=([-+0-9.eE]+)\s+"
    r"elapsed_ms=([-+0-9.eE]+)\s+tok_per_sec=([-+0-9.eE]+)\s+certified=(\d+)\s+escalated=(\d+)\]$"
)
TRACE_STEP_RE = re.compile(
    r"^\[trace_step prompt=(\d+)\s+idx=(\d+)\s+tok=(-?\d+)\s+"
    r"q8_top1=(-?\d+)\s+cert=(\d+)\s+esc=(\d+)\s+margin=([-+0-9.eE]+)\]$"
)

Q_GRID_SCRUB = [0.60, 0.85, 1.15, 1.85, 3.00, 5.00]
Q_GRID_TAIL = [9.0, 18.0, 35.0]
R4_GRID = [0.55, 0.85, 1.35, 2.00]


def write_prompts_file(prompts: list[dict], path: Path) -> None:
    lines = [p["text"].replace("\n", " ").strip() for p in prompts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fit_stats(pred, y_dh) -> dict:
    import torch

    rmse = float(torch.sqrt(torch.mean((pred - y_dh) ** 2)).item())
    std_pred = float(pred.std().item())
    std_y = float(y_dh.std().item())
    if std_pred > 1e-12 and std_y > 1e-12:
        corr = float(torch.corrcoef(torch.stack([pred, y_dh]))[0, 1].item())
    else:
        corr = 0.0
    if not math.isfinite(corr):
        corr = 0.0
    ratio = y_dh / torch.clamp(pred, min=1e-6)
    return {
        "cal_rmse": round(rmse, 6),
        "cal_pearson_r": round(corr, 6),
        "pred_min": round(float(pred.min().item()), 6),
        "pred_mean": round(float(pred.mean().item()), 6),
        "rel_ratio_p50": round(float(torch.quantile(ratio, 0.50).item()), 6),
        "rel_ratio_p90": round(float(torch.quantile(ratio, 0.90).item()), 6),
        "rel_ratio_p95": round(float(torch.quantile(ratio, 0.95).item()), 6),
        "rel_ratio_p99": round(float(torch.quantile(ratio, 0.99).item()), 6),
        "rel_ratio_max": round(float(ratio.max().item()), 6),
    }


def run_nnls_stage() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import torch.nn as nn
    from scipy.optimize import nnls
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def sync_flush(dev: torch.device) -> None:
        gc.collect()
        if dev.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

    def quantize_q4_inplace(model: nn.Module) -> None:
        for layer in model.model.layers:
            for _, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    w = mod.weight.data.float()
                    out_d, in_d = w.shape
                    pad = (-in_d) % 32
                    w_pad = torch.nn.functional.pad(w, (0, pad)) if pad else w
                    w_b = w_pad.view(out_d, w_pad.shape[1] // 32, 32)
                    m = torch.amax(torch.abs(w_b), dim=-1, keepdim=True)
                    s = (m / 7.0).to(torch.float16).float()
                    s_safe = torch.where(s == 0.0, torch.ones_like(s), s)
                    q = torch.where(
                        s == 0.0,
                        torch.zeros_like(w_b),
                        torch.clamp(torch.round(w_b / s_safe), -8.0, 7.0),
                    )
                    w_rec = (s * q).view(out_d, w_pad.shape[1])[:, :in_d]
                    mod.weight.data = w_rec.to(torch.bfloat16)

    def collect_pass(quant_q4: bool, dev: torch.device, tok_ids: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_8B), dtype=torch.bfloat16, trust_remote_code=True
        ).to(dev)
        model.eval()
        if quant_q4:
            quantize_q4_inplace(model)
            sync_flush(dev)

        layer_norms: list[torch.Tensor] = []
        hooks = []
        if quant_q4:
            for layer in model.model.layers:
                def _make_hook():
                    def _hk(_mod, _inp, out):
                        hs = out[0] if isinstance(out, tuple) else out
                        layer_norms.append(torch.norm(hs[0].float(), p=2, dim=-1).cpu())
                    return _hk
                hooks.append(layer.register_forward_hook(_make_hook()))

        hs_all: list[torch.Tensor] = []
        norms_all: list[torch.Tensor] = []
        with torch.no_grad():
            for ids in tok_ids:
                layer_norms.clear()
                out = model(ids.to(dev), output_hidden_states=True)
                s0 = min(4, int(out.hidden_states[-1].shape[1]) - 1)
                hs_all.append(out.hidden_states[-1][0, s0:].float().cpu())
                if quant_q4:
                    norms_all.append(torch.stack(layer_norms, dim=-1)[s0:])
            sync_flush(dev)

        for hk in hooks:
            hk.remove()
        del model
        sync_flush(dev)
        norms_cat = torch.cat(norms_all, dim=0) if quant_q4 else None
        return torch.cat(hs_all, dim=0), norms_cat

    t0 = time.perf_counter()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_8B), trust_remote_code=True)
    cal_prompts, _ = build_disjoint_prompt_sets()
    tok_ids = []
    for p in cal_prompts:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": p["text"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tok_ids.append(tokenizer(chat, add_special_tokens=False, return_tensors="pt").input_ids)

    n_tokens_raw = sum(int(t.shape[1]) for t in tok_ids)
    print(f"[nnls-p12:8B] Pass 1 BF16 on {device} (240 CAL prompts, {n_tokens_raw} tokens)...", flush=True)
    h_bf16, _ = collect_pass(False, device, tok_ids)
    print(f"[nnls-p12:8B] Pass 2 Q4_EISA on {device}...", flush=True)
    h_q4, norms_q4 = collect_pass(True, device, tok_ids)
    assert norms_q4 is not None

    n_tokens = int(norms_q4.shape[0])
    n_layers = int(norms_q4.shape[1])
    y_dh = torch.norm(h_q4 - h_bf16, p=2, dim=-1)
    inv_rms_35 = torch.rsqrt((norms_q4[:, -1] ** 2) / 4096.0 + 1.0e-6)

    # 1) Linear NNLS (Phase 11 baseline)
    c_lin_np, _ = nnls(norms_q4.double().numpy(), y_dh.double().numpy())
    c_lin = torch.from_numpy(c_lin_np).float()
    pred_lin = norms_q4.float() @ c_lin
    stats_lin = _fit_stats(pred_lin, y_dh)

    # 2) Jacobian-scaled NNLS: inv_rms_35 * (c_add + sum_l c_l ||x_l||_2)
    x_jac = inv_rms_35[:, None] * torch.cat([torch.ones(n_tokens, 1), norms_q4], dim=1)
    w_jac_np, _ = nnls(x_jac.double().numpy(), y_dh.double().numpy())
    w_jac = torch.from_numpy(w_jac_np).float()
    c_add = float(w_jac[0].item())
    c_jac = w_jac[1:]
    pred_jac = x_jac.float() @ w_jac
    stats_jac = _fit_stats(pred_jac, y_dh)

    # Kill-switch rule: adopt jac iff r >= +0.20 AND lowers cal_rmse vs lin
    adopt_jac = (stats_jac["cal_pearson_r"] >= 0.20) and (stats_jac["cal_rmse"] < stats_lin["cal_rmse"])

    if adopt_jac:
        out_weights = [float(x) for x in c_jac.tolist()]
        out_c_add = c_add
        out_use_inv_rms = 1.0
        assert float(pred_jac.min().item()) > 0.0, f"Predicted jac dh_hat non-positive: {pred_jac.min().item()}"
    else:
        out_weights = [float(x) for x in c_lin.tolist()]
        out_c_add = 0.0
        out_use_inv_rms = 0.0
        assert float(pred_lin.min().item()) > 0.0, f"Predicted lin dh_hat non-positive: {pred_lin.min().item()}"

    assert sum(1 for v in out_weights if v > 1e-8) >= 1 or out_c_add > 1e-8, "All weights vanished!"

    lines = [f"{v:.8f}" for v in out_weights] + [f"{out_c_add:.8f}", f"{out_use_inv_rms:.1f}"]
    WEIGHTS_8B_P12_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "model": "Qwen3-8B",
        "n_layers": n_layers,
        "n_cal_prompts": len(cal_prompts),
        "n_cal_tokens": n_tokens,
        "dh_mean": round(float(y_dh.mean().item()), 6),
        "dh_p95": round(float(torch.quantile(y_dh, 0.95).item()), 6),
        "dh_max": round(float(y_dh.max().item()), 6),
        "inv_rms_mean": round(float(inv_rms_35.mean().item()), 6),
        "inv_rms_min": round(float(inv_rms_35.min().item()), 6),
        "inv_rms_max": round(float(inv_rms_35.max().item()), 6),
        "lin": {
            **stats_lin,
            "weights_nonzero": sum(1 for v in c_lin.tolist() if v > 1e-8),
            "weights": [float(v) for v in c_lin.tolist()],
        },
        "jac": {
            **stats_jac,
            "c_add": c_add,
            "weights_nonzero": sum(1 for v in c_jac.tolist() if v > 1e-8),
            "weights": [float(v) for v in c_jac.tolist()],
        },
        "adopt_jac": adopt_jac,
        "kill_switch_rule": "adopt_jac iff jac.cal_pearson_r >= 0.20 and jac.cal_rmse < lin.cal_rmse",
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    WEIGHTS_8B_P12_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"[nnls-p12:8B] LIN r={stats_lin['cal_pearson_r']:.4f} rmse={stats_lin['cal_rmse']:.4f} p95={stats_lin['rel_ratio_p95']:.4f} | "
        f"JAC r={stats_jac['cal_pearson_r']:.4f} rmse={stats_jac['cal_rmse']:.4f} p95={stats_jac['rel_ratio_p95']:.4f} c_add={c_add:.6f} | "
        f"adopt_jac={adopt_jac} elapsed={meta['elapsed_s']}s -> {WEIGHTS_8B_P12_TXT}",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def build_native_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["SOUNIO_METAL_RUNTIME"] = str(METAL_DYLIB)
    for k in (
        "SOUNIO_CONFORMAL_Q",
        "SOUNIO_CONFORMAL_WEIGHTS",
        "SOUNIO_SCRUB_MAX_STREAK",
        "SOUNIO_LLM_TRACE_STEPS",
    ):
        env.pop(k, None)
    if extra_env:
        env.update(extra_env)
    return env


def run_native_bench(
    prompts_file: Path,
    expected_n: int,
    extra_args: list[str],
    extra_env: dict[str, str] | None = None,
    max_tokens: int = 16,
    max_seq: int = 512,
) -> dict:
    cmd = [
        str(BIN_PATH),
        "--model-dir", str(MODEL_8B),
        "--backend", "metal",
        "--max-seq", str(max_seq),
        "--max-tokens", str(max_tokens),
        "--no-think",
        "--temp", "0",
        "--bench-prompts", str(prompts_file),
    ] + extra_args
    env = build_native_env(extra_env)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"bin/sounio-llm failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-2000:]}"
        )

    toks_by_idx: dict[int, list[int]] = {}
    stats_by_idx: dict[int, dict] = {}
    traces_by_idx: dict[int, list[dict]] = {}

    for raw_ln in proc.stdout.splitlines():
        ln = raw_ln.strip()
        if not ln:
            continue
        mt = BENCH_TOKENS_RE.match(ln)
        if mt:
            idx = int(mt.group(1))
            n_tok = int(mt.group(2))
            ids_str = mt.group(3)
            ids = [int(x) for x in ids_str.split(",") if x != ""] if ids_str else []
            if len(ids) != n_tok:
                raise RuntimeError(f"bench_tokens count mismatch at idx={idx}: {n_tok} vs {len(ids)}")
            toks_by_idx[idx] = ids
            continue
        mp = BENCH_PROMPT_RE.match(ln)
        if mp:
            idx = int(mp.group(1))
            stats_by_idx[idx] = {
                "idx": idx,
                "prompt_tokens": int(mp.group(2)),
                "gen_tokens": int(mp.group(3)),
                "prefill_ms": float(mp.group(4)),
                "decode_ms": float(mp.group(5)),
                "decode_tok_per_sec": float(mp.group(6)),
                "elapsed_ms": float(mp.group(7)),
                "tok_per_sec": float(mp.group(8)),
                "certified": int(mp.group(9)),
                "escalated": int(mp.group(10)),
            }
            continue
        ms = TRACE_STEP_RE.match(ln)
        if ms:
            p_idx = int(ms.group(1))
            traces_by_idx.setdefault(p_idx, []).append({
                "idx": int(ms.group(2)),
                "tok": int(ms.group(3)),
                "q8_top1": int(ms.group(4)),
                "cert": int(ms.group(5)),
                "esc": int(ms.group(6)),
                "margin": float(ms.group(7)),
            })

    if len(toks_by_idx) != expected_n or len(stats_by_idx) != expected_n:
        raise RuntimeError(
            f"Incomplete bench output: expected={expected_n} "
            f"tokens={len(toks_by_idx)} stats={len(stats_by_idx)}"
        )

    rows = []
    for i in range(expected_n):
        r = dict(stats_by_idx[i])
        r["token_ids"] = toks_by_idx[i]
        rows.append(r)

    tot_gen = sum(r["gen_tokens"] for r in rows)
    tot_cert = sum(r["certified"] for r in rows)
    tot_esc = sum(r["escalated"] for r in rows)
    tot_dec_s = sum(r["decode_ms"] for r in rows) / 1000.0
    tot_el_s = sum(r["elapsed_ms"] for r in rows) / 1000.0
    return {
        "wall_s": round(wall_s, 3),
        "total_gen_tokens": tot_gen,
        "total_certified": tot_cert,
        "total_escalated": tot_esc,
        "certified_pct": round(100.0 * tot_cert / max(1, tot_gen), 2),
        "mean_decode_tok_s": round(statistics.mean(r["decode_tok_per_sec"] for r in rows), 2),
        "agg_decode_tok_s": round(tot_gen / tot_dec_s, 2) if tot_dec_s > 0 else 0.0,
        "agg_e2e_tok_s": round(tot_gen / tot_el_s, 2) if tot_el_s > 0 else 0.0,
        "rows": rows,
        "traces_by_idx": traces_by_idx,
    }


def compare_vs_bf16(bf_rows: list[dict], run: dict) -> dict:
    n = len(bf_rows)
    exact_matches = 0
    mismatches = 0
    cert_first_div = 0
    fallback_first_div = 0
    traces = run["traces_by_idx"]
    for i in range(n):
        bf_ids = bf_rows[i]["token_ids"]
        q_ids = run["rows"][i]["token_ids"]
        if bf_ids == q_ids:
            exact_matches += 1
            continue
        mismatches += 1
        div_idx = None
        for k in range(min(len(bf_ids), len(q_ids))):
            if bf_ids[k] != q_ids[k]:
                div_idx = k
                break
        if div_idx is not None and i in traces and div_idx < len(traces[i]):
            if traces[i][div_idx]["cert"] == 1:
                cert_first_div += 1
            else:
                fallback_first_div += 1
    out = {
        "n_prompts": n,
        "exact_matches": exact_matches,
        "mismatches": mismatches,
        "mismatch_rate": round(mismatches / max(1, n), 6),
        "exact_rate": round(exact_matches / max(1, n), 6),
        "total_gen_tokens": run["total_gen_tokens"],
        "certified_pct": run["certified_pct"],
        "mean_decode_tok_s": run["mean_decode_tok_s"],
        "agg_decode_tok_s": run["agg_decode_tok_s"],
        "agg_e2e_tok_s": run["agg_e2e_tok_s"],
        "wall_s": run["wall_s"],
    }
    if traces:
        out["certified_first_divergences"] = cert_first_div
        out["fallback_first_divergences"] = fallback_first_div
    return out


def run_rollout_stage() -> None:
    if not WEIGHTS_8B_P12_TXT.exists():
        raise FileNotFoundError(f"Missing Phase 12 NNLS weights: {WEIGHTS_8B_P12_TXT}")
    cal_prompts, test_prompts = build_disjoint_prompt_sets()
    write_prompts_file(cal_prompts, CAL_PROMPTS_TXT)
    write_prompts_file(test_prompts, HELDOUT_PROMPTS_TXT)

    if P12_BF16_REFS.exists():
        print(f"[rollout-p12] Loading Phase 12 BF16 references from {P12_BF16_REFS}...", flush=True)
        cached = json.loads(P12_BF16_REFS.read_text(encoding="utf-8"))
        cal_bf = cached["cal_bf"]
        test_bf = cached["test_bf"]
    else:
        print("[rollout-p12] 1/4 Native Fused BF16 reference on 240 CAL prompts...", flush=True)
        cal_bf = run_native_bench(CAL_PROMPTS_TXT, 240, ["--quant", "bf16"])
        print(f"[rollout-p12] CAL BF16 agg_decode={cal_bf['agg_decode_tok_s']} tok/s", flush=True)

        print("[rollout-p12] 2/4 Native Fused BF16 reference on 240 HELD-OUT prompts...", flush=True)
        test_bf = run_native_bench(HELDOUT_PROMPTS_TXT, 240, ["--quant", "bf16"])
        print(f"[rollout-p12] HELD-OUT BF16 agg_decode={test_bf['agg_decode_tok_s']} tok/s", flush=True)

        P12_BF16_REFS.write_text(json.dumps({"cal_bf": cal_bf, "test_bf": test_bf}), encoding="utf-8")

    p11_parity = None
    if P11_BF16_REFS.exists():
        p11_cached = json.loads(P11_BF16_REFS.read_text(encoding="utf-8"))
        cal_match = sum(
            1 for a, b in zip(cal_bf["rows"], p11_cached["cal_bf"]["rows"])
            if a["token_ids"] == b["token_ids"]
        )
        test_match = sum(
            1 for a, b in zip(test_bf["rows"], p11_cached["test_bf"]["rows"])
            if a["token_ids"] == b["token_ids"]
        )
        p11_parity = {"cal_exact_vs_p11": cal_match, "heldout_exact_vs_p11": test_match}
        print(
            f"[rollout-p12] Fused BF16 parity vs Phase 11 unfused BF16: "
            f"CAL={cal_match}/240 HELD-OUT={test_match}/240",
            flush=True,
        )

    env_w = {"SOUNIO_CONFORMAL_WEIGHTS": str(WEIGHTS_8B_P12_TXT), "SOUNIO_LLM_TRACE_STEPS": "1"}
    sweep = []
    print("[rollout-p12] 3/4 Sweeping Q_GRID on 240 CAL (q4_scrub)...", flush=True)
    for q in Q_GRID_SCRUB:
        r = run_native_bench(
            CAL_PROMPTS_TXT, 240, ["--quant", "q4_scrub", "--conformal-q", f"{q}"], extra_env=env_w
        )
        m = compare_vs_bf16(cal_bf["rows"], r)
        m["q"] = q
        m["p_val_bound"] = round((m["mismatches"] + 1) / 241.0, 6)
        sweep.append(m)
        print(
            f"  q={q:<6} mismatches={m['mismatches']:>3}/240 p={(m['mismatches']+1)/241:.4f} "
            f"cert={m['certified_pct']:>5.1f}% dec={m['agg_decode_tok_s']} tok/s "
            f"fb_div={m.get('fallback_first_divergences', 0)}",
            flush=True,
        )
        if m["mismatches"] <= 1:
            break

    for q in Q_GRID_TAIL:
        if sweep[-1]["mismatches"] <= 1:
            break
        r = run_native_bench(
            CAL_PROMPTS_TXT, 240, ["--quant", "q4_scrub", "--conformal-q", f"{q}"], extra_env=env_w
        )
        m = compare_vs_bf16(cal_bf["rows"], r)
        m["q"] = q
        m["p_val_bound"] = round((m["mismatches"] + 1) / 241.0, 6)
        sweep.append(m)
        print(
            f"  q={q:<6} mismatches={m['mismatches']:>3}/240 p={(m['mismatches']+1)/241:.4f} "
            f"cert={m['certified_pct']:>5.1f}% dec={m['agg_decode_tok_s']} tok/s "
            f"fb_div={m.get('fallback_first_divergences', 0)}",
            flush=True,
        )

    def select_q_hat(alpha: float) -> float:
        for idx in range(len(sweep)):
            if all((sweep[j]["mismatches"] + 1) / 241.0 <= alpha for j in range(idx, len(sweep))):
                return float(sweep[idx]["q"])
        return float(sweep[-1]["q"])

    q_hat_005 = select_q_hat(0.05)
    q_hat_001 = select_q_hat(0.01)

    write_prompts_file(cal_prompts[:20], CAL20_PROMPTS_TXT)
    print("[rollout-p12] 4/4 Sweeping R4_GRID on 20 CAL (q4_exact_scrub)...", flush=True)
    r4_sweep = []
    env_r4 = {"SOUNIO_CONFORMAL_WEIGHTS": str(WEIGHTS_8B_P12_TXT)}
    for q in R4_GRID:
        r = run_native_bench(
            CAL20_PROMPTS_TXT, 20, ["--quant", "q4_exact_scrub", "--conformal-q", f"{q}"], extra_env=env_r4
        )
        m = compare_vs_bf16(cal_bf["rows"][:20], r)
        m["q"] = q
        assert m["exact_matches"] == 20, f"q4_exact_scrub CAL mismatch at q={q}: {m['mismatches']}"
        r4_sweep.append(m)
        print(
            f"  R4 q={q:<5} exact=20/20 cert={m['certified_pct']:>5.1f}% "
            f"mean_dec={m['mean_decode_tok_s']} agg_dec={m['agg_decode_tok_s']} tok/s",
            flush=True,
        )

    best_r4 = max(r4_sweep, key=lambda x: x["mean_decode_tok_s"])
    q_opt_r4 = float(best_r4["q"])

    payload = {
        "model": "Qwen3-8B",
        "max_tokens": 16,
        "max_seq": 512,
        "bf16_parity_vs_p11": p11_parity,
        "cal_bf16": {
            "agg_decode_tok_s": cal_bf["agg_decode_tok_s"],
            "agg_e2e_tok_s": cal_bf["agg_e2e_tok_s"],
            "token_ids": [r["token_ids"] for r in cal_bf["rows"]],
        },
        "heldout_bf16": {
            "agg_decode_tok_s": test_bf["agg_decode_tok_s"],
            "agg_e2e_tok_s": test_bf["agg_e2e_tok_s"],
            "token_ids": [r["token_ids"] for r in test_bf["rows"]],
        },
        "q_scrub_sweep": sweep,
        "r4_sweep": r4_sweep,
        "q_hat_005": q_hat_005,
        "q_hat_001": q_hat_001,
        "q_opt_r4": q_opt_r4,
    }
    OUT_CAL_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[rollout-p12] DONE q_hat_005={q_hat_005} q_hat_001={q_hat_001} q_opt_r4={q_opt_r4} -> {OUT_CAL_JSON}",
        flush=True,
    )


def clopper_pearson_bounds(k: int, n: int) -> dict:
    from scipy.stats import beta

    upper_1sided = 1.0 if k >= n else round(float(beta.ppf(0.95, k + 1, n - k)), 6)
    low_2 = 0.0 if k <= 0 else round(float(beta.ppf(0.025, k, n - k + 1)), 6)
    high_2 = 1.0 if k >= n else round(float(beta.ppf(0.975, k + 1, n - k)), 6)
    return {
        "cp95_upper_1sided": upper_1sided,
        "cp95_two_sided": [low_2, high_2],
    }


def run_audit_stage() -> None:
    cal_prompts, test_prompts = build_disjoint_prompt_sets()
    write_prompts_file(test_prompts, HELDOUT_PROMPTS_TXT)

    if OUT_CAL_JSON.exists():
        cal = json.loads(OUT_CAL_JSON.read_text(encoding="utf-8"))
        q_hat_005 = float(cal["q_hat_005"])
        q_hat_001 = float(cal["q_hat_001"])
        q_opt_r4 = float(cal["q_opt_r4"])
    else:
        q_hat_005 = 1.15
        q_hat_001 = 3.0
        q_opt_r4 = 0.55

    print("[audit-p12] Running fresh unloaded fused BF16 reference on 240 HELD-OUT prompts...", flush=True)
    test_bf = run_native_bench(HELDOUT_PROMPTS_TXT, 240, ["--quant", "bf16"])
    held_bf = {
        "agg_decode_tok_s": test_bf["agg_decode_tok_s"],
        "agg_e2e_tok_s": test_bf["agg_e2e_tok_s"],
        "token_ids": [r["token_ids"] for r in test_bf["rows"]],
    }
    print(f"[audit-p12] Unloaded fused HELD-OUT BF16 agg_decode={held_bf['agg_decode_tok_s']} tok/s", flush=True)

    p11_parity = {}
    if P12_BF16_REFS.exists():
        p12_cached = json.loads(P12_BF16_REFS.read_text(encoding="utf-8"))
        p12_match = sum(
            1 for a, b in zip(test_bf["rows"], p12_cached["test_bf"]["rows"])
            if a["token_ids"] == b["token_ids"]
        )
        assert p12_match == 240, f"Fresh fused BF16 mismatch vs cached P12_BF16_REFS: {p12_match}/240"
        p11_parity["heldout_exact_vs_p12_cached"] = p12_match
    if P11_BF16_REFS.exists():
        p11_cached = json.loads(P11_BF16_REFS.read_text(encoding="utf-8"))
        test_match = sum(
            1 for a, b in zip(test_bf["rows"], p11_cached["test_bf"]["rows"])
            if a["token_ids"] == b["token_ids"]
        )
        assert test_match == 240, f"Fresh fused BF16 mismatch vs P11_BF16_REFS: {test_match}/240"
        p11_parity["heldout_exact_vs_p11"] = test_match
    print(f"[audit-p12] BF16 token parity checks: {p11_parity}", flush=True)

    held_bf_ids = held_bf["token_ids"]
    bf_rows_stub = [{"token_ids": ids} for ids in held_bf_ids]

    modes = [
        ("r3_q4_scrub_005", ["--quant", "q4_scrub", "--conformal-alpha", "0.05"], {"SOUNIO_LLM_TRACE_STEPS": "1"}, q_hat_005),
        ("r3_q4_scrub_001", ["--quant", "q4_scrub", "--conformal-alpha", "0.01"], {"SOUNIO_LLM_TRACE_STEPS": "1"}, q_hat_001),
        ("r3_q4_scrub_fast_noguarantee", ["--quant", "q4_scrub", "--fast-no-guarantee"], {"SOUNIO_LLM_TRACE_STEPS": "1"}, q_opt_r4),
        ("r4_q4_exact_scrub_001", ["--quant", "q4_exact_scrub"], {}, q_hat_001),
        ("r4_q4_exact_scrub_005", ["--quant", "q4_exact_scrub", "--conformal-alpha", "0.05"], {}, q_hat_005),
        ("r4_q4_exact_scrub_fast_noguarantee", ["--quant", "q4_exact_scrub", "--fast-no-guarantee"], {}, q_opt_r4),
        ("r1_q4_conformal_probe_005", ["--quant", "q4_conformal", "--conformal-alpha", "0.05"], {"SOUNIO_LLM_TRACE_STEPS": "1"}, q_hat_005),
        ("ctrl_q4_fixed_k8_spec", ["--quant", "q4_exact_scrub", "--conformal-q", "0.0"], {"SOUNIO_SCRUB_MAX_STREAK": "8"}, 0.0),
    ]

    results = {}
    print("[audit-p12] Evaluating rebuilt bin/sounio-llm (ZERO env weights) on 240 HELD-OUT...", flush=True)
    for name, args, env_x, q_eff in modes:
        run = run_native_bench(HELDOUT_PROMPTS_TXT, 240, args, extra_env=env_x)
        m = compare_vs_bf16(bf_rows_stub, run)
        cp = clopper_pearson_bounds(m["mismatches"], m["n_prompts"])
        m["q_effective"] = q_eff
        m["cp95_upper_1sided"] = cp["cp95_upper_1sided"]
        m["cp95_two_sided"] = cp["cp95_two_sided"]
        m["speedup_vs_bf16_decode"] = round(m["agg_decode_tok_s"] / max(1e-9, held_bf["agg_decode_tok_s"]), 3)
        results[name] = m
        print(
            f"  {name:<34} q={q_eff:<4} exact={m['exact_matches']}/240 mism={m['mismatches']}/240 "
            f"cp95_ub={m['cp95_upper_1sided']:.4f} cert={m['certified_pct']:>5.1f}% "
            f"agg_dec={m['agg_decode_tok_s']} tok/s ({m['speedup_vs_bf16_decode']:.3f}x BF16) "
            f"fb_div={m.get('fallback_first_divergences', '-')}",
            flush=True,
        )

    assert results["r4_q4_exact_scrub_001"]["exact_matches"] == 240, "r4_q4_exact_scrub_001 failed 240/240!"
    assert results["r4_q4_exact_scrub_005"]["exact_matches"] == 240, "r4_q4_exact_scrub_005 failed 240/240!"
    assert results["r4_q4_exact_scrub_fast_noguarantee"]["exact_matches"] == 240, "r4_q4_exact_scrub_fast_noguarantee failed 240/240!"
    assert results["ctrl_q4_fixed_k8_spec"]["exact_matches"] == 240, "ctrl_q4_fixed_k8_spec failed 240/240!"
    assert results["r3_q4_scrub_005"]["fallback_first_divergences"] == 0, "r3 alpha=0.05 had fallback first-divergence!"
    assert results["r3_q4_scrub_001"]["fallback_first_divergences"] == 0, "r3 alpha=0.01 had fallback first-divergence!"

    out_doc = {
        "model": "Qwen3-8B",
        "split": "240 disjoint held-out prompts (4 domains x 60)",
        "max_tokens": 16,
        "max_seq": 512,
        "grid_selection_note": "fixed_grid_monotone_valid_but_conservative",
        "q_grid_scrub": Q_GRID_SCRUB,
        "default_baked_q": {
            "quant_mode_16_conformal_alpha001": 3.0,
            "quant_mode_17_conformal_alpha005": 1.15,
            "quant_mode_18_scrub_alpha001": 3.0,
            "quant_mode_19_scrub_alpha005": 1.15,
            "quant_mode_20_exact_scrub_alpha001": 3.0,
            "quant_mode_21_exact_scrub_alpha005": 1.15,
            "fast_no_guarantee_flag_q": 0.55,
        },
        "bf16_parity_vs_p11": p11_parity,
        "heldout_bf16": held_bf,
        "q_hat_005": q_hat_005,
        "q_hat_001": q_hat_001,
        "q_opt_r4": q_opt_r4,
        "audit_results": results,
    }
    OUT_AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT_JSON.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    print(f"[audit-p12] PASS -> {OUT_AUDIT_JSON}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["nnls-8b", "rollout", "audit"])
    args = parser.parse_args()

    if args.stage == "nnls-8b":
        run_nnls_stage()
    elif args.stage == "rollout":
        run_rollout_stage()
    elif args.stage == "audit":
        run_audit_stage()


if __name__ == "__main__":
    main()
