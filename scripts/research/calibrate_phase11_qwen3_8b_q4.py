#!/usr/bin/env python3
"""
Phase 11 Step 3A/3B — Q4 EISA NNLS Calibration, Rollout Conformal Sweep & Held-Out Audit

Subcommands:
  --stage nnls-0.6b  Fit 28L Q4 NNLS weights on Qwen3-0.6B (240 CAL prompts, 2-pass).
  --stage nnls-8b    Fit 36L Q4 NNLS weights on Qwen3-8B (240 CAL prompts, 2-pass).
  --stage rollout    Pure native bin/sounio-llm sweep on Qwen3-8B 240 CAL prompts (NO PyTorch).
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

from audit_phase7b_step1d_prompt_conformal import (  # noqa: E402
    build_disjoint_prompt_sets,
    fit_nonneg_layer_weights,
)

REPO_ROOT = Path("/Users/demetriosagourakis/dev/sounio-worktrees/llm-backend")
BIN_PATH = REPO_ROOT / "bin/sounio-llm"
METAL_DYLIB = REPO_ROOT / "artifacts/self-hosted/libsounio_metal.dylib"
MODEL_06B = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B")
MODEL_8B = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-8B")

WEIGHTS_06B_TXT = Path("/tmp/qwen3_06b_q4_nnls_weights.txt")
WEIGHTS_06B_META = Path("/tmp/qwen3_06b_q4_nnls_meta.json")
WEIGHTS_8B_TXT = Path("/tmp/qwen3_8b_q4_nnls_weights.txt")
WEIGHTS_8B_META = Path("/tmp/qwen3_8b_q4_nnls_meta.json")
OUT_CAL_JSON = Path("/tmp/phase11_qwen3_8b_q4_cal.json")
OUT_AUDIT_JSON = Path(
    "/Users/demetriosagourakis/sounio-research/checkpoints/"
    "Qwen3-0.6B-ossm-sidecar/phase11_qwen3_8b_q4_audit.json"
)
CAL_PROMPTS_TXT = Path("/tmp/phase11_cal240_prompts.txt")
CAL20_PROMPTS_TXT = Path("/tmp/phase11_cal20_prompts.txt")
HELDOUT_PROMPTS_TXT = Path("/tmp/phase11_heldout240_prompts.txt")

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

Q_GRID_SCRUB = [1.15, 1.85, 3.00, 5.00]
Q_GRID_TAIL = [9.0, 18.0, 35.0]
R4_GRID = [0.85, 1.35, 2.00]


def write_prompts_file(prompts: list[dict], path: Path) -> None:
    lines = [p["text"].replace("\n", " ").strip() for p in prompts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_nnls_stage(model_dir: Path, weights_out: Path, meta_out: Path, label: str) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import torch.nn as nn
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
            str(model_dir), dtype=torch.bfloat16, trust_remote_code=True
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

    from scipy.optimize import nnls

    t0 = time.perf_counter()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
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
    print(f"[nnls:{label}] Pass 1 BF16 on {device} (240 CAL prompts, {n_tokens_raw} tokens)...", flush=True)
    h_bf16, _ = collect_pass(False, device, tok_ids)
    print(f"[nnls:{label}] Pass 2 Q4_EISA on {device}...", flush=True)
    h_q4, norms_q4 = collect_pass(True, device, tok_ids)
    assert norms_q4 is not None

    n_tokens = int(norms_q4.shape[0])
    y_dh = torch.norm(h_q4 - h_bf16, p=2, dim=-1)
    c_np, _ = nnls(norms_q4.double().numpy(), y_dh.double().numpy())
    c_weights = torch.from_numpy(c_np).float()
    pred = norms_q4.float() @ c_weights
    rmse = float(torch.sqrt(torch.mean((pred - y_dh) ** 2)).item())
    std_pred = float(pred.std().item())
    std_y = float(y_dh.std().item())
    if std_pred > 1e-12 and std_y > 1e-12:
        corr = float(torch.corrcoef(torch.stack([pred, y_dh]))[0, 1].item())
    else:
        corr = 0.0
    if not math.isfinite(corr):
        corr = 0.0
    fit_stats = {"cal_rmse": round(rmse, 6), "cal_pearson_r": round(corr, 6)}

    vals = [float(x) for x in c_weights.tolist()]
    assert float(pred.min().item()) > 0.0, f"Predicted dh_hat non-positive: min={pred.min().item()}"
    assert sum(1 for v in vals if v > 1e-8) >= 1, "All NNLS weights vanished!"
    weights_out.write_text("\n".join(f"{v:.8f}" for v in vals) + "\n", encoding="utf-8")

    meta = {
        "model": label,
        "model_dir": str(model_dir),
        "n_layers": len(vals),
        "n_cal_prompts": len(cal_prompts),
        "n_cal_tokens": n_tokens,
        "dh_mean": round(float(y_dh.mean().item()), 6),
        "dh_p95": round(float(torch.quantile(y_dh, 0.95).item()), 6),
        "dh_max": round(float(y_dh.max().item()), 6),
        "weights_nonzero": sum(1 for v in vals if v > 1e-8),
        "weights": vals,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        **fit_stats,
    }
    meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"[nnls:{label}] DONE n_layers={len(vals)} r={fit_stats['cal_pearson_r']:.4f} "
        f"rmse={fit_stats['cal_rmse']:.4f} nonzero={meta['weights_nonzero']} "
        f"elapsed={meta['elapsed_s']}s -> {weights_out}",
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
    if not WEIGHTS_8B_TXT.exists():
        raise FileNotFoundError(f"Missing 36L NNLS weights: {WEIGHTS_8B_TXT}")
    cal_prompts, test_prompts = build_disjoint_prompt_sets()
    write_prompts_file(cal_prompts, CAL_PROMPTS_TXT)
    write_prompts_file(test_prompts, HELDOUT_PROMPTS_TXT)

    refs_cache = Path("/tmp/phase11_qwen3_8b_bf16_refs.json")
    if refs_cache.exists():
        print(f"[rollout] Loading cached BF16 references from {refs_cache}...", flush=True)
        cached = json.loads(refs_cache.read_text(encoding="utf-8"))
        cal_bf = cached["cal_bf"]
        test_bf = cached["test_bf"]
    else:
        print("[rollout] 1/4 Native BF16 reference on 240 CAL prompts...", flush=True)
        cal_bf = run_native_bench(CAL_PROMPTS_TXT, 240, ["--quant", "bf16"])
        print(f"[rollout] CAL BF16 agg_decode={cal_bf['agg_decode_tok_s']} tok/s", flush=True)

        print("[rollout] 2/4 Native BF16 reference on 240 HELD-OUT prompts...", flush=True)
        test_bf = run_native_bench(HELDOUT_PROMPTS_TXT, 240, ["--quant", "bf16"])
        print(f"[rollout] HELD-OUT BF16 agg_decode={test_bf['agg_decode_tok_s']} tok/s", flush=True)

        refs_cache.write_text(json.dumps({"cal_bf": cal_bf, "test_bf": test_bf}), encoding="utf-8")

    env_w = {"SOUNIO_CONFORMAL_WEIGHTS": str(WEIGHTS_8B_TXT), "SOUNIO_LLM_TRACE_STEPS": "1"}
    sweep = []
    print("[rollout] 3/4 Sweeping Q_GRID on 240 CAL (q4_scrub)...", flush=True)
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
    print("[rollout] 4/4 Sweeping R4_GRID on 20 CAL (q4_exact_scrub)...", flush=True)
    r4_sweep = []
    env_r4 = {"SOUNIO_CONFORMAL_WEIGHTS": str(WEIGHTS_8B_TXT)}
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
        f"[rollout] DONE q_hat_005={q_hat_005} q_hat_001={q_hat_001} q_opt_r4={q_opt_r4} -> {OUT_CAL_JSON}",
        flush=True,
    )


def run_audit_stage() -> None:
    if not OUT_CAL_JSON.exists():
        raise FileNotFoundError(f"Run --stage rollout first: missing {OUT_CAL_JSON}")
    cal = json.loads(OUT_CAL_JSON.read_text(encoding="utf-8"))
    held_bf_ids = cal["heldout_bf16"]["token_ids"]
    bf_rows_stub = [{"token_ids": ids} for ids in held_bf_ids]
    q_hat_005 = cal["q_hat_005"]

    modes = [
        ("r1_q4_conformal_probe", ["--quant", "q4_conformal", "--conformal-mode", "flip_exact", "--conformal-q", f"{q_hat_005}"], {"SOUNIO_LLM_TRACE_STEPS": "1"}),
        ("r3_q4_scrub_005", ["--quant", "q4_scrub", "--conformal-alpha", "0.05"], {"SOUNIO_LLM_TRACE_STEPS": "1"}),
        ("r3_q4_scrub_001", ["--quant", "q4_scrub", "--conformal-alpha", "0.01"], {"SOUNIO_LLM_TRACE_STEPS": "1"}),
        ("r4_q4_exact_scrub", ["--quant", "q4_exact_scrub"], {}),
        ("ctrl_q4_fixed_k8_spec", ["--quant", "q4_exact_scrub", "--conformal-q", "0.0"], {"SOUNIO_SCRUB_MAX_STREAK": "8"}),
    ]

    results = {}
    print("[audit] Evaluating rebuilt bin/sounio-llm (baked weights, ZERO env weights) on 240 HELD-OUT...", flush=True)
    for name, args, env_x in modes:
        run = run_native_bench(HELDOUT_PROMPTS_TXT, 240, args, extra_env=env_x)
        m = compare_vs_bf16(bf_rows_stub, run)
        results[name] = m
        print(
            f"  {name:<24} exact={m['exact_matches']}/240 mism={m['mismatches']}/240 "
            f"cert={m['certified_pct']:>5.1f}% agg_dec={m['agg_decode_tok_s']} tok/s "
            f"fb_div={m.get('fallback_first_divergences', '-')}",
            flush=True,
        )

    assert results["r4_q4_exact_scrub"]["exact_matches"] == 240, "r4_q4_exact_scrub failed 240/240!"
    assert results["ctrl_q4_fixed_k8_spec"]["exact_matches"] == 240, "ctrl_q4_fixed_k8_spec failed 240/240!"
    assert results["r3_q4_scrub_005"]["fallback_first_divergences"] == 0, "r3 alpha=0.05 had fallback first-divergence!"
    assert results["r3_q4_scrub_001"]["fallback_first_divergences"] == 0, "r3 alpha=0.01 had fallback first-divergence!"

    out_doc = {
        "model": "Qwen3-8B",
        "split": "240 disjoint held-out prompts (4 domains x 60)",
        "max_tokens": 16,
        "max_seq": 512,
        "heldout_bf16": cal["heldout_bf16"],
        "q_hat_005": cal["q_hat_005"],
        "q_hat_001": cal["q_hat_001"],
        "q_opt_r4": cal["q_opt_r4"],
        "audit_results": results,
    }
    OUT_AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT_JSON.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    print(f"[audit] PASS -> {OUT_AUDIT_JSON}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["nnls-0.6b", "nnls-8b", "rollout", "audit"])
    args = parser.parse_args()

    if args.stage == "nnls-0.6b":
        run_nnls_stage(MODEL_06B, WEIGHTS_06B_TXT, WEIGHTS_06B_META, "Qwen3-0.6B")
    elif args.stage == "nnls-8b":
        run_nnls_stage(MODEL_8B, WEIGHTS_8B_TXT, WEIGHTS_8B_META, "Qwen3-8B")
    elif args.stage == "rollout":
        run_rollout_stage()
    elif args.stage == "audit":
        run_audit_stage()


if __name__ == "__main__":
    main()
