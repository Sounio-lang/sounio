#!/usr/bin/env python3
"""
Length-Aware Q4 Conformal Recalibration & Held-Out Audit (L = 128, conditional L = 256).

Uses ONLY the native Sounio engine (bin/sounio-llm, Metal, Qwen3-8B, q4_scrub) with:
- Pure fused BF16 reference (Step 5c) on 240 CAL and 240 HELD-OUT disjoint prompts
- Fixed grid identical to Phase 12: Q_GRID_SCRUB = [0.60, 0.85, 1.15, 1.85, 3.00, 5.00]
  (extending upward onto Q_GRID_TAIL_UPWARD only if no point satisfies alpha)
- Per-run macOS free-memory guard (< 4.0 GB => discard & redo)
- Prompt-level first-divergence token index and certified-vs-fallback attribution
- 40-prompt HELD-OUT fused BF16 + q4_scrub speed baseline gated by swapouts_delta < 1,000
- Incremental checkpointing to disk after BF16 refs, 40-prompt baseline, and every CAL/audit step
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from scipy.stats import beta

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_phase7b_step1d_prompt_conformal import build_disjoint_prompt_sets  # noqa: E402

BIN_PATH = Path("/Users/demetriosagourakis/dev/sounio-worktrees/llm-backend/bin/sounio-llm")
METAL_DYLIB = Path(
    "/Users/demetriosagourakis/dev/sounio-worktrees/llm-backend/artifacts/self-hosted/libsounio_metal.dylib"
)
MODEL_8B = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-8B")
MODEL_DIR = MODEL_8B
MODEL_TAG = "qwen3_8b"
MODEL_LABEL = "Qwen3-8B"
SIDE_DIR = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar")

P12_L16_BF16_REFS = Path("/tmp/phase12_qwen3_8b_bf16_refs.json")
CAL_PROMPTS_TXT = Path("/tmp/phase12_len_cal240_prompts.txt")
HELDOUT_PROMPTS_TXT = Path("/tmp/phase12_len_heldout240_prompts.txt")
HELDOUT40_PROMPTS_TXT = Path("/tmp/phase12_len_heldout40_prompts.txt")
WARMUP2_PROMPTS_TXT = Path("/tmp/phase12_len_warmup2_prompts.txt")
WARMUP1_PROMPTS_TXT = Path("/tmp/phase12_len_warmup1_prompts.txt")

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
Q_GRID_TAIL_UPWARD = [9.0, 18.0, 35.0, 60.0, 120.0]

MIN_FREE_GB_FLOOR = 1.5
WAIT_FREE_GB_RESUME = 6.0
SWAPOUTS_TIMING_MAX_DELTA = 1000
POLL_INTERVAL_S = 5.0

_HW_MEMSIZE: int | None = None


def bf16_refs_path(length: int) -> Path:
    return SIDE_DIR / f"phase12_{MODEL_TAG}_bf16_refs_L{length}.json"


def out_len_path(length: int) -> Path:
    return SIDE_DIR / f"phase12_{MODEL_TAG}_q4_length_L{length}.json"


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2) + "\n"
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_prompts_file(prompts: list[dict], path: Path) -> None:
    lines = [p["text"].replace("\n", " ").strip() for p in prompts]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_hw_memsize() -> int:
    global _HW_MEMSIZE
    if _HW_MEMSIZE is None:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        _HW_MEMSIZE = int(out)
    return _HW_MEMSIZE


def sample_macos_memory() -> dict:
    hw_mem = get_hw_memsize()
    level_pct = int(subprocess.check_output(["sysctl", "-n", "kern.memorystatus_level"], text=True).strip())
    vm_out = subprocess.check_output(["vm_stat"], text=True)

    page_size = 16384
    m_ps = re.search(r"page size of (\d+) bytes", vm_out)
    if m_ps:
        page_size = int(m_ps.group(1))

    stats: dict[str, int] = {}
    for ln in vm_out.splitlines():
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        v_clean = v.strip().rstrip(".")
        if v_clean.isdigit():
            stats[k.strip()] = int(v_clean)

    wired_pages = stats.get("Pages wired down", 0)
    comp_pages = stats.get("Pages occupied by compressor", 0)
    swapouts = stats.get("Swapouts", 0)

    free_bytes = (level_pct / 100.0) * hw_mem
    return {
        "level_pct": level_pct,
        "free_gb": round(free_bytes / 1e9, 3),
        "free_gib": round(free_bytes / (1024 ** 3), 3),
        "wired_gb": round((wired_pages * page_size) / 1e9, 3),
        "compressor_gb": round((comp_pages * page_size) / 1e9, 3),
        "swapouts": swapouts,
    }


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


def clopper_pearson_upper_1sided(k: int, n: int) -> float:
    return 1.0 if k >= n else round(float(beta.ppf(0.95, k + 1, n - k)), 6)


def clopper_pearson_2sided(k: int, n: int) -> list[float]:
    lo = 0.0 if k <= 0 else round(float(beta.ppf(0.025, k, n - k + 1)), 6)
    hi = 1.0 if k >= n else round(float(beta.ppf(0.975, k + 1, n - k)), 6)
    return [lo, hi]


def parse_bench_stdout(stdout: str, expected_n: int, wall_s: float) -> dict:
    toks_by_idx: dict[int, list[int]] = {}
    stats_by_idx: dict[int, dict] = {}
    traces_by_idx: dict[int, list[dict]] = {}

    for raw_ln in stdout.splitlines():
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


def compute_model_weight_footprint() -> dict:
    st_files = sorted(MODEL_DIR.glob("*.safetensors"))
    bf16_disk_bytes = sum(p.stat().st_size for p in st_files)
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    h = int(cfg["hidden_size"])
    inter = int(cfg["intermediate_size"])
    n_layers = int(cfg["num_hidden_layers"])
    nq = int(cfg["num_attention_heads"])
    nkv = int(cfg["num_key_value_heads"])
    hd = int(cfg["head_dim"])
    v = int(cfg["vocab_size"])

    def _mat_q4_bytes(rows: int, cols: int) -> int:
        nb = (cols + 31) // 32
        return ((rows * nb * 20 + 63) // 64) * 64

    per_layer = (
        _mat_q4_bytes(nq * hd, h)
        + _mat_q4_bytes(nkv * hd, h)
        + _mat_q4_bytes(nkv * hd, h)
        + _mat_q4_bytes(h, nq * hd)
        + _mat_q4_bytes(inter, h)
        + _mat_q4_bytes(inter, h)
        + _mat_q4_bytes(h, inter)
    )
    q4_arena_bytes = _mat_q4_bytes(v, h) + n_layers * per_layer + 16384
    total_bytes = bf16_disk_bytes + q4_arena_bytes
    return {
        "bf16_safetensors_bytes": bf16_disk_bytes,
        "bf16_safetensors_gb": round(bf16_disk_bytes / 1e9, 3),
        "bf16_safetensors_gib": round(bf16_disk_bytes / (1024 ** 3), 3),
        "q4_arena_bytes": q4_arena_bytes,
        "q4_arena_gb": round(q4_arena_bytes / 1e9, 3),
        "q4_arena_gib": round(q4_arena_bytes / (1024 ** 3), 3),
        "total_resident_weight_bytes": total_bytes,
        "total_resident_weight_gb": round(total_bytes / 1e9, 3),
        "overhead_pct_over_bf16": round((q4_arena_bytes / max(1, bf16_disk_bytes)) * 100.0, 2),
    }


def run_unmeasured_warmup(label: str, quant_args: list[str], prompts_file: Path, max_tokens: int = 16) -> None:
    cmd = [
        str(BIN_PATH),
        "--model-dir", str(MODEL_DIR),
        "--backend", "metal",
        "--max-seq", "512",
        "--max-tokens", str(max_tokens),
        "--no-think",
        "--temp", "0",
        "--bench-prompts", str(prompts_file),
    ] + quant_args
    env = build_native_env(None)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Warmup {label} failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout={proc.stdout[-1000:]}\nstderr={proc.stderr[-1000:]}"
        )
    time.sleep(1.0)
    print(f"[warmup] {label} done wall_s={wall_s:.2f}", flush=True)


def run_native_with_mem_guard(
    label: str,
    prompts_file: Path,
    expected_n: int,
    extra_args: list[str],
    extra_env: dict[str, str] | None,
    max_tokens: int,
    max_seq: int = 512,
    require_timing_clean: bool = False,
    max_attempts: int = 3,
    allow_unclean_timing_fallback: bool = False,
) -> dict:
    cmd = [
        str(BIN_PATH),
        "--model-dir", str(MODEL_DIR),
        "--backend", "metal",
        "--max-seq", str(max_seq),
        "--max-tokens", str(max_tokens),
        "--no-think",
        "--temp", "0",
        "--bench-prompts", str(prompts_file),
    ] + extra_args
    env = build_native_env(extra_env)
    best_unclean_parsed: dict | None = None

    for attempt in range(1, max_attempts + 1):
        start_mem = sample_macos_memory()
        wait_waits = 0
        while start_mem["free_gb"] < MIN_FREE_GB_FLOOR and wait_waits < 12:
            print(
                f"[mem-wait] {label} attempt={attempt} start_free_gb={start_mem['free_gb']:.3f} < {MIN_FREE_GB_FLOOR:.1f}; sleeping 10s...",
                flush=True,
            )
            time.sleep(10.0)
            start_mem = sample_macos_memory()
            wait_waits += 1
        if start_mem["free_gb"] < MIN_FREE_GB_FLOOR:
            raise RuntimeError(f"{label}: free memory stuck below {MIN_FREE_GB_FLOOR:.1f} GB before start ({start_mem})")

        print(
            f"[mem-start] {label} attempt={attempt} start_free_gb={start_mem['free_gb']:.3f} "
            f"(level={start_mem['level_pct']}%, free_gib={start_mem['free_gib']:.3f}, wired_gb={start_mem['wired_gb']:.3f})",
            flush=True,
        )

        samples: list[dict] = []
        low_mem_event: dict | None = None
        stop_event = threading.Event()
        lock = threading.Lock()

        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def _monitor() -> None:
            nonlocal low_mem_event
            while not stop_event.wait(POLL_INTERVAL_S):
                try:
                    m = sample_macos_memory()
                except Exception:
                    continue
                with lock:
                    samples.append(m)
                    if m["free_gb"] < MIN_FREE_GB_FLOOR and low_mem_event is None:
                        low_mem_event = m
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        break

        mon_thread = threading.Thread(target=_monitor, daemon=True)
        mon_thread.start()
        stdout, stderr = proc.communicate()
        stop_event.set()
        mon_thread.join(timeout=5.0)
        wall_s = time.perf_counter() - t0

        end_mem = sample_macos_memory()
        if end_mem["free_gb"] < MIN_FREE_GB_FLOOR and low_mem_event is None:
            low_mem_event = end_mem

        if low_mem_event is not None:
            print(
                f"[mem-discard] {label} attempt={attempt} DISCARDED: min_free_gb={low_mem_event['free_gb']:.3f} < {MIN_FREE_GB_FLOOR:.3f} GB "
                f"(level={low_mem_event['level_pct']}%). Waiting for >={WAIT_FREE_GB_RESUME:.1f} GB free and redoing...",
                flush=True,
            )
            for _ in range(30):
                time.sleep(10.0)
                chk = sample_macos_memory()
                if chk["free_gb"] >= WAIT_FREE_GB_RESUME:
                    break
            continue

        if proc.returncode != 0:
            raise RuntimeError(
                f"bin/sounio-llm failed ({proc.returncode}) on {label}: {' '.join(cmd)}\n"
                f"stdout={stdout[-2000:]}\nstderr={stderr[-2000:]}"
            )

        all_samples = [start_mem] + samples + [end_mem]
        min_free_gb = round(min(s["free_gb"] for s in all_samples), 3)
        min_level_pct = min(s["level_pct"] for s in all_samples)
        max_wired_gb = round(max(s["wired_gb"] for s in all_samples), 3)
        max_compressor_gb = round(max(s["compressor_gb"] for s in all_samples), 3)
        swapouts_delta = max(0, end_mem["swapouts"] - start_mem["swapouts"])
        timing_valid = swapouts_delta < SWAPOUTS_TIMING_MAX_DELTA

        print(
            f"[mem-ok] {label} start_free_gb={start_mem['free_gb']:.3f} "
            f"min_free_gb={min_free_gb:.3f} end_free_gb={end_mem['free_gb']:.3f} "
            f"max_wired_gb={max_wired_gb:.3f} swapouts_delta={swapouts_delta} timing_valid={timing_valid}",
            flush=True,
        )

        parsed = parse_bench_stdout(stdout, expected_n, wall_s)
        parsed["mem"] = {
            "attempt": attempt,
            "start_free_gb": start_mem["free_gb"],
            "min_free_gb": min_free_gb,
            "end_free_gb": end_mem["free_gb"],
            "start_level_pct": start_mem["level_pct"],
            "min_level_pct": min_level_pct,
            "end_level_pct": end_mem["level_pct"],
            "start_wired_gb": start_mem["wired_gb"],
            "end_wired_gb": end_mem["wired_gb"],
            "max_wired_gb": max_wired_gb,
            "max_compressor_gb": max_compressor_gb,
            "swapouts_delta": swapouts_delta,
            "timing_valid": timing_valid,
            "poll_count": len(samples),
        }

        if require_timing_clean and not timing_valid:
            if (
                best_unclean_parsed is None
                or parsed["mem"]["swapouts_delta"] < best_unclean_parsed["mem"]["swapouts_delta"]
            ):
                best_unclean_parsed = parsed
            if attempt >= max_attempts:
                if allow_unclean_timing_fallback and best_unclean_parsed is not None:
                    print(
                        f"[timing-unclean] {label}: all {max_attempts} attempts had "
                        f"swapouts_delta >= {SWAPOUTS_TIMING_MAX_DELTA} "
                        f"(best={best_unclean_parsed['mem']['swapouts_delta']}); recording with timing_valid=False",
                        flush=True,
                    )
                    return best_unclean_parsed
                raise RuntimeError(
                    f"{label}: exceeded {max_attempts} attempts with swapouts_delta={swapouts_delta} >= {SWAPOUTS_TIMING_MAX_DELTA}"
                )
            print(
                f"[timing-retry] {label} attempt={attempt} swapouts_delta={swapouts_delta} >= {SWAPOUTS_TIMING_MAX_DELTA}; retrying after 5s...",
                flush=True,
            )
            time.sleep(5.0)
            continue

        return parsed

    if allow_unclean_timing_fallback and best_unclean_parsed is not None:
        return best_unclean_parsed
    raise RuntimeError(f"{label}: exceeded {max_attempts} attempts due to free_gb < {MIN_FREE_GB_FLOOR:.1f} GB")


def compare_vs_bf16_detailed(
    bf_rows: list[dict],
    prompts_meta: list[dict],
    run: dict,
    bf_speed_baseline: dict | None,
) -> dict:
    n = len(bf_rows)
    exact_matches = 0
    mismatches = 0
    cert_first_div = 0
    fallback_first_div = 0
    unknown_first_div = 0
    mismatch_positions = []
    traces = run["traces_by_idx"]

    for i in range(n):
        bf_ids = bf_rows[i]["token_ids"]
        q_ids = run["rows"][i]["token_ids"]
        if bf_ids == q_ids:
            exact_matches += 1
            continue

        mismatches += 1
        div_idx = min(len(bf_ids), len(q_ids))
        for k in range(min(len(bf_ids), len(q_ids))):
            if bf_ids[k] != q_ids[k]:
                div_idx = k
                break

        first_cert = None
        first_esc = None
        first_margin = None
        tr_list = traces.get(i, [])
        if div_idx < len(tr_list):
            tinfo = tr_list[div_idx]
            first_cert = int(tinfo["cert"])
            first_esc = int(tinfo["esc"])
            first_margin = float(tinfo["margin"])
            if first_cert == 1:
                cert_first_div += 1
            else:
                fallback_first_div += 1
        else:
            unknown_first_div += 1

        mismatch_positions.append({
            "prompt_idx": i,
            "prompt_id": prompts_meta[i]["id"],
            "domain": prompts_meta[i]["domain"],
            "first_div_token_idx": div_idx,
            "bf16_len": len(bf_ids),
            "q4_len": len(q_ids),
            "bf16_tok": bf_ids[div_idx] if div_idx < len(bf_ids) else None,
            "q4_tok": q_ids[div_idx] if div_idx < len(q_ids) else None,
            "first_div_certified": first_cert,
            "first_div_escalated": first_esc,
            "first_div_margin": first_margin,
        })

    cp_1sided = clopper_pearson_upper_1sided(mismatches, n)
    cp_2sided = clopper_pearson_2sided(mismatches, n)
    timing_valid = bool(run["mem"].get("timing_valid", False))
    base_valid = bool(bf_speed_baseline and bf_speed_baseline.get("timing_valid", False))
    if timing_valid and base_valid:
        sp_dec = round(run["agg_decode_tok_s"] / max(1e-9, bf_speed_baseline["agg_decode_tok_s"]), 3)
        sp_e2e = round(run["agg_e2e_tok_s"] / max(1e-9, bf_speed_baseline["agg_e2e_tok_s"]), 3)
    else:
        sp_dec = None
        sp_e2e = None

    return {
        "n_prompts": n,
        "exact_matches": exact_matches,
        "mismatches": mismatches,
        "mismatch_rate": round(mismatches / max(1, n), 6),
        "cal_exceedance": round((mismatches + 1) / (n + 1), 6),
        "cp95_upper_1sided": cp_1sided,
        "cp95_two_sided": cp_2sided,
        "total_gen_tokens": run["total_gen_tokens"],
        "certified_pct": run["certified_pct"],
        "timing_valid": timing_valid,
        "mean_decode_tok_s": run["mean_decode_tok_s"],
        "agg_decode_tok_s": run["agg_decode_tok_s"],
        "agg_e2e_tok_s": run["agg_e2e_tok_s"],
        "speedup_vs_bf16_decode": sp_dec,
        "speedup_vs_bf16_e2e": sp_e2e,
        "speedup_vs_bf16_heldout40_decode": sp_dec,
        "speedup_vs_bf16_heldout40_e2e": sp_e2e,
        "wall_s": run["wall_s"],
        "certified_first_divergences": cert_first_div,
        "fallback_first_divergences": fallback_first_div,
        "unknown_first_divergences": unknown_first_div,
        "mem": run["mem"],
        "mismatch_positions": mismatch_positions,
    }


def trim_bf16_summary(bf_run: dict) -> dict:
    return {
        "agg_decode_tok_s": bf_run["agg_decode_tok_s"],
        "agg_e2e_tok_s": bf_run["agg_e2e_tok_s"],
        "mean_decode_tok_s": bf_run["mean_decode_tok_s"],
        "total_gen_tokens": bf_run["total_gen_tokens"],
        "wall_s": bf_run["wall_s"],
        "mem": bf_run["mem"],
        "rows": bf_run["rows"],
    }


def bf16_brief_meta(bf_doc_entry: dict) -> dict:
    d = {
        k: bf_doc_entry[k]
        for k in ("agg_decode_tok_s", "agg_e2e_tok_s", "mean_decode_tok_s", "total_gen_tokens", "wall_s", "mem")
    }
    d["timing_valid"] = bf_doc_entry.get("mem", {}).get("swapouts_delta", SWAPOUTS_TIMING_MAX_DELTA) < SWAPOUTS_TIMING_MAX_DELTA
    return d


def load_or_run_bf16_refs(length: int, cal_prompts: list[dict], test_prompts: list[dict]) -> dict:
    rpath = bf16_refs_path(length)
    refs_doc: dict = {
        "model": MODEL_LABEL,
        "quant": "bf16",
        "max_tokens": length,
        "max_seq": 512,
    }
    if rpath.exists():
        loaded = json.loads(rpath.read_text(encoding="utf-8"))
        refs_doc.update(loaded)

    cal_valid = (
        isinstance(refs_doc.get("cal_bf"), dict)
        and len(refs_doc["cal_bf"].get("rows", [])) == 240
        and refs_doc["cal_bf"].get("mem", {}).get("min_free_gb", 0.0) >= MIN_FREE_GB_FLOOR
    )
    if cal_valid:
        cal_sw = refs_doc["cal_bf"]["mem"].get("swapouts_delta", -1)
        print(
            f"[bf16-L{length}] Reusing saved CAL BF16 token reference from {rpath} "
            f"(agg_dec={refs_doc['cal_bf']['agg_decode_tok_s']} tok/s, min_free_gb={refs_doc['cal_bf']['mem']['min_free_gb']}, "
            f"swapouts_delta={cal_sw}, timing_valid={cal_sw < SWAPOUTS_TIMING_MAX_DELTA})",
            flush=True,
        )
    else:
        print(f"[bf16-L{length}] 1/2 Running fused BF16 on 240 CAL prompts (max_tokens={length})...", flush=True)
        cal_raw = run_native_with_mem_guard(
            f"bf16_cal_L{length}", CAL_PROMPTS_TXT, 240, ["--quant", "bf16"], extra_env=None, max_tokens=length
        )
        refs_doc["cal_bf"] = trim_bf16_summary(cal_raw)
        atomic_write_json(rpath, refs_doc)
        print(
            f"[bf16-L{length}] Saved CAL BF16 -> {rpath} "
            f"(agg_dec={refs_doc['cal_bf']['agg_decode_tok_s']} tok/s, agg_e2e={refs_doc['cal_bf']['agg_e2e_tok_s']} tok/s)",
            flush=True,
        )

    test_valid = (
        isinstance(refs_doc.get("test_bf"), dict)
        and len(refs_doc["test_bf"].get("rows", [])) == 240
        and refs_doc["test_bf"].get("mem", {}).get("min_free_gb", 0.0) >= MIN_FREE_GB_FLOOR
    )
    if test_valid:
        test_sw = refs_doc["test_bf"]["mem"].get("swapouts_delta", -1)
        print(
            f"[bf16-L{length}] Reusing saved HELD-OUT BF16 token reference from {rpath} "
            f"(agg_dec={refs_doc['test_bf']['agg_decode_tok_s']} tok/s, min_free_gb={refs_doc['test_bf']['mem']['min_free_gb']}, "
            f"swapouts_delta={test_sw}, timing_valid={test_sw < SWAPOUTS_TIMING_MAX_DELTA})",
            flush=True,
        )
    else:
        print(f"[bf16-L{length}] 2/2 Running fused BF16 on 240 HELD-OUT prompts (max_tokens={length})...", flush=True)
        test_raw = run_native_with_mem_guard(
            f"bf16_heldout_L{length}", HELDOUT_PROMPTS_TXT, 240, ["--quant", "bf16"], extra_env=None, max_tokens=length
        )
        refs_doc["test_bf"] = trim_bf16_summary(test_raw)
        atomic_write_json(rpath, refs_doc)
        print(
            f"[bf16-L{length}] Saved CAL + HELD-OUT BF16 -> {rpath} "
            f"(agg_dec={refs_doc['test_bf']['agg_decode_tok_s']} tok/s, agg_e2e={refs_doc['test_bf']['agg_e2e_tok_s']} tok/s)",
            flush=True,
        )

    if MODEL_TAG == "qwen3_8b" and P12_L16_BF16_REFS.exists():
        p16 = json.loads(P12_L16_BF16_REFS.read_text(encoding="utf-8"))
        cal_pfx = sum(
            1
            for a, b in zip(refs_doc["cal_bf"]["rows"], p16["cal_bf"]["rows"])
            if a["token_ids"][:16] == b["token_ids"][:16]
        )
        test_pfx = sum(
            1
            for a, b in zip(refs_doc["test_bf"]["rows"], p16["test_bf"]["rows"])
            if a["token_ids"][:16] == b["token_ids"][:16]
        )
        refs_doc["l16_prefix_parity"] = {"cal": cal_pfx, "heldout": test_pfx}
        atomic_write_json(rpath, refs_doc)
        print(f"[bf16-L{length}] 16-token prefix parity vs Phase 12 L=16 BF16: CAL={cal_pfx}/240 HELD-OUT={test_pfx}/240", flush=True)
        assert cal_pfx == 240 and test_pfx == 240, f"BF16 16-token prefix mismatch: CAL={cal_pfx}/240 HELD-OUT={test_pfx}/240"

    return refs_doc


def load_or_run_bf16_heldout40_baseline(
    length: int,
    test_prompts: list[dict],
    test_bf_rows: list[dict],
    state_doc: dict,
) -> dict:
    cached = state_doc.get("bf16_heldout40_baseline")
    if (
        isinstance(cached, dict)
        and cached.get("n_prompts") == 40
        and cached.get("timing_valid") is True
        and cached.get("mem", {}).get("min_free_gb", 0.0) >= MIN_FREE_GB_FLOOR
        and cached.get("mem", {}).get("swapouts_delta", SWAPOUTS_TIMING_MAX_DELTA) < SWAPOUTS_TIMING_MAX_DELTA
        and cached.get("token_parity_vs_ref240") == 40
    ):
        print(
            f"[bf16-40-L{length}] Reusing clean 40-prompt HELD-OUT BF16 baseline: "
            f"agg_dec={cached['agg_decode_tok_s']} tok/s agg_e2e={cached['agg_e2e_tok_s']} tok/s "
            f"swapouts_delta={cached['mem']['swapouts_delta']}",
            flush=True,
        )
        return cached

    print(f"[bf16-40-L{length}] Timing fused BF16 on 40 HELD-OUT prompts (swapouts_delta < {SWAPOUTS_TIMING_MAX_DELTA} required)...", flush=True)
    run_unmeasured_warmup(f"bf16_L{length}", ["--quant", "bf16"], WARMUP2_PROMPTS_TXT, max_tokens=16)
    raw40 = run_native_with_mem_guard(
        f"bf16_heldout40_L{length}",
        HELDOUT40_PROMPTS_TXT,
        40,
        ["--quant", "bf16"],
        extra_env=None,
        max_tokens=length,
        require_timing_clean=True,
        max_attempts=4,
    )
    parity40 = sum(1 for a, b in zip(raw40["rows"], test_bf_rows[:40]) if a["token_ids"] == b["token_ids"])
    assert parity40 == 40, f"bf16_heldout40 token parity failed: {parity40}/40 vs 240-prompt ref"
    dom_counts: dict[str, int] = {}
    for p in test_prompts[:40]:
        dom_counts[p["domain"]] = dom_counts.get(p["domain"], 0) + 1
    out = {
        "n_prompts": 40,
        "domain_counts": dom_counts,
        "token_parity_vs_ref240": parity40,
        "agg_decode_tok_s": raw40["agg_decode_tok_s"],
        "agg_e2e_tok_s": raw40["agg_e2e_tok_s"],
        "mean_decode_tok_s": raw40["mean_decode_tok_s"],
        "total_gen_tokens": raw40["total_gen_tokens"],
        "wall_s": raw40["wall_s"],
        "timing_valid": raw40["mem"]["timing_valid"],
        "mem": raw40["mem"],
    }
    print(
        f"[bf16-40-L{length}] Clean HELD-OUT 40 BF16 baseline: agg_dec={out['agg_decode_tok_s']} tok/s "
        f"agg_e2e={out['agg_e2e_tok_s']} tok/s swapouts_delta={out['mem']['swapouts_delta']} "
        f"min_free_gb={out['mem']['min_free_gb']} parity={parity40}/40",
        flush=True,
    )
    return out


def select_q_smallest(sweep: list[dict], alpha: float) -> float | None:
    for entry in sweep:
        if entry["cal_exceedance"] <= alpha:
            return float(entry["q"])
    return None


def select_q_monotone(sweep: list[dict], alpha: float) -> float | None:
    for idx in range(len(sweep)):
        if all(sweep[j]["cal_exceedance"] <= alpha for j in range(idx, len(sweep))):
            return float(sweep[idx]["q"])
    return None


def run_length_pipeline(length: int) -> None:
    cal_prompts, test_prompts = build_disjoint_prompt_sets()
    write_prompts_file(cal_prompts, CAL_PROMPTS_TXT)
    write_prompts_file(test_prompts, HELDOUT_PROMPTS_TXT)
    write_prompts_file(test_prompts[:40], HELDOUT40_PROMPTS_TXT)
    write_prompts_file(test_prompts[:2], WARMUP2_PROMPTS_TXT)
    write_prompts_file(test_prompts[:1], WARMUP1_PROMPTS_TXT)

    refs_doc = load_or_run_bf16_refs(length, cal_prompts, test_prompts)
    cal_bf = refs_doc["cal_bf"]
    test_bf = refs_doc["test_bf"]

    out_path = out_len_path(length)
    state_doc: dict = {}
    if out_path.exists():
        state_doc = json.loads(out_path.read_text(encoding="utf-8"))

    cal_bf_meta = bf16_brief_meta(cal_bf)
    test_bf_meta = bf16_brief_meta(test_bf)
    weight_footprint = compute_model_weight_footprint()

    bf16_40_baseline = load_or_run_bf16_heldout40_baseline(length, test_prompts, test_bf["rows"], state_doc)

    # Restore previously evaluated valid CAL points if resuming
    existing_by_q: dict[float, dict] = {}
    for entry in state_doc.get("cal_sweep", []):
        if entry.get("mem", {}).get("min_free_gb", 0.0) >= MIN_FREE_GB_FLOOR:
            existing_by_q[float(entry["q"])] = entry

    sweep: list[dict] = []
    env_trace = {"SOUNIO_LLM_TRACE_STEPS": "1"}

    print(f"[cal-L{length}] Fixed grid (same as Phase 12): Q_GRID_SCRUB = {Q_GRID_SCRUB}", flush=True)
    print(f"[mem-footprint] {json.dumps(weight_footprint)}", flush=True)

    def _save_state(extra: dict | None = None) -> None:
        observed_q4_wired = [e["mem"].get("max_wired_gb", 0.0) for e in sweep if "mem" in e]
        if extra and isinstance(extra.get("heldout_audit"), dict):
            for v in extra["heldout_audit"].values():
                if isinstance(v, dict) and "mem" in v:
                    observed_q4_wired.append(v["mem"].get("max_wired_gb", 0.0))
        if extra and isinstance(extra.get("heldout40_timing"), dict):
            for v in extra["heldout40_timing"].values():
                if isinstance(v, dict) and "mem" in v:
                    observed_q4_wired.append(v["mem"].get("max_wired_gb", 0.0))
        mem_summary = {
            **weight_footprint,
            "free_memory_floor_gb": MIN_FREE_GB_FLOOR,
            "swapouts_delta_timing_threshold": SWAPOUTS_TIMING_MAX_DELTA,
            "bf16_heldout40_max_wired_gb": bf16_40_baseline["mem"].get("max_wired_gb"),
            "q4_scrub_max_wired_gb": max(observed_q4_wired) if observed_q4_wired else None,
        }
        doc = {
            "model": MODEL_LABEL,
            "quant": "q4_scrub",
            "length_L": length,
            "max_seq": 512,
            "q_grid_scrub": Q_GRID_SCRUB,
            "q_grid_tail_upward": Q_GRID_TAIL_UPWARD,
            "bf16_refs_file": str(bf16_refs_path(length)),
            "l16_prefix_parity": refs_doc.get("l16_prefix_parity"),
            "memory_footprint_summary": mem_summary,
            "cal_bf16": cal_bf_meta,
            "heldout_bf16": test_bf_meta,
            "bf16_heldout40_baseline": bf16_40_baseline,
            "cal_sweep": sweep,
        }
        if extra:
            doc.update(extra)
        atomic_write_json(out_path, doc)

    _save_state()

    for q in Q_GRID_SCRUB:
        if q in existing_by_q:
            m = existing_by_q[q]
            print(
                f"  [resume] CAL q={q:<5} mism={m['mismatches']:>3}/240 exc={m['cal_exceedance']:.4f} "
                f"cp95_ub={m['cp95_upper_1sided']:.4f} cert={m['certified_pct']:>5.1f}% "
                f"dec={m['agg_decode_tok_s']} e2e={m['agg_e2e_tok_s']} swap={m['mem'].get('swapouts_delta')} "
                f"timing_valid={m.get('timing_valid')} cert_div={m['certified_first_divergences']} fb_div={m['fallback_first_divergences']}",
                flush=True,
            )
        else:
            r = run_native_with_mem_guard(
                f"cal_L{length}_q{q}",
                CAL_PROMPTS_TXT,
                240,
                ["--quant", "q4_scrub", "--conformal-q", f"{q}"],
                extra_env=env_trace,
                max_tokens=length,
            )
            m = compare_vs_bf16_detailed(cal_bf["rows"], cal_prompts, r, bf16_40_baseline)
            m["q"] = q
            m["grid_source"] = "fixed_grid"
            print(
                f"  CAL q={q:<5} mism={m['mismatches']:>3}/240 exc={m['cal_exceedance']:.4f} "
                f"cp95_ub={m['cp95_upper_1sided']:.4f} cert={m['certified_pct']:>5.1f}% "
                f"dec={m['agg_decode_tok_s']} tok/s e2e={m['agg_e2e_tok_s']} tok/s "
                f"swap={m['mem']['swapouts_delta']} timing_valid={m['timing_valid']} "
                f"cert_div={m['certified_first_divergences']} fb_div={m['fallback_first_divergences']} "
                f"first_div_indices={[p['first_div_token_idx'] for p in m['mismatch_positions']]}",
                flush=True,
            )
        sweep.append(m)
        _save_state()
        if m["cal_exceedance"] <= 0.01:
            break

    q05_fixed = select_q_smallest(sweep, 0.05)
    q01_fixed = select_q_smallest(sweep, 0.01)
    extended_upward = False
    extension_reason = None

    if q05_fixed is None or q01_fixed is None:
        missing = []
        if q05_fixed is None:
            missing.append("alpha=0.05 (need cal_exceedance<=0.05, <=11/240)")
        if q01_fixed is None:
            missing.append("alpha=0.01 (need cal_exceedance<=0.01, <=1/240)")
        extension_reason = (
            f"Fixed grid {Q_GRID_SCRUB} had no point satisfying {', '.join(missing)}; "
            f"extending upward onto {Q_GRID_TAIL_UPWARD}."
        )
        extended_upward = True
        print(f"[cal-L{length}] {extension_reason}", flush=True)

        for q in Q_GRID_TAIL_UPWARD:
            if q in existing_by_q:
                m = existing_by_q[q]
                print(
                    f"  [resume] CAL(ext) q={q:<5} mism={m['mismatches']:>3}/240 exc={m['cal_exceedance']:.4f} "
                    f"cp95_ub={m['cp95_upper_1sided']:.4f} cert={m['certified_pct']:>5.1f}% "
                    f"dec={m['agg_decode_tok_s']} e2e={m['agg_e2e_tok_s']} swap={m['mem'].get('swapouts_delta')} timing_valid={m.get('timing_valid')}",
                    flush=True,
                )
            else:
                r = run_native_with_mem_guard(
                    f"cal_L{length}_ext_q{q}",
                    CAL_PROMPTS_TXT,
                    240,
                    ["--quant", "q4_scrub", "--conformal-q", f"{q}"],
                    extra_env=env_trace,
                    max_tokens=length,
                )
                m = compare_vs_bf16_detailed(cal_bf["rows"], cal_prompts, r, bf16_40_baseline)
                m["q"] = q
                m["grid_source"] = "upward_extension"
                print(
                    f"  CAL(ext) q={q:<5} mism={m['mismatches']:>3}/240 exc={m['cal_exceedance']:.4f} "
                    f"cp95_ub={m['cp95_upper_1sided']:.4f} cert={m['certified_pct']:>5.1f}% "
                    f"dec={m['agg_decode_tok_s']} tok/s e2e={m['agg_e2e_tok_s']} tok/s "
                    f"swap={m['mem']['swapouts_delta']} timing_valid={m['timing_valid']} "
                    f"cert_div={m['certified_first_divergences']} fb_div={m['fallback_first_divergences']} "
                    f"first_div_indices={[p['first_div_token_idx'] for p in m['mismatch_positions']]}",
                    flush=True,
                )
            sweep.append(m)
            _save_state({"extended_upward": extended_upward, "extension_reason": extension_reason})
            if m["cal_exceedance"] <= 0.01:
                break

    q_hat_005 = select_q_smallest(sweep, 0.05)
    q_hat_001 = select_q_smallest(sweep, 0.01)
    q_hat_005_mono = select_q_monotone(sweep, 0.05)
    q_hat_001_mono = select_q_monotone(sweep, 0.01)

    print(
        f"[cal-L{length}] Selected q_hat(0.05, L={length})={q_hat_005} (monotone={q_hat_005_mono}), "
        f"q_hat(0.01, L={length})={q_hat_001} (monotone={q_hat_001_mono})",
        flush=True,
    )
    if q_hat_005 is None or q_hat_001 is None:
        raise RuntimeError(
            f"Unable to satisfy alpha after upward extension at L={length}: "
            f"q_hat_005={q_hat_005}, q_hat_001={q_hat_001}"
        )

    existing_audit = state_doc.get("heldout_audit", {})
    audit_results: dict[str, dict] = {}
    audit_by_q: dict[float, dict] = {}

    for alpha_label, q_sel in [("alpha_0.05", q_hat_005), ("alpha_0.01", q_hat_001)]:
        prev = existing_audit.get(alpha_label)
        if (
            isinstance(prev, dict)
            and float(prev.get("q", -1.0)) == q_sel
            and prev.get("mem", {}).get("min_free_gb", 0.0) >= MIN_FREE_GB_FLOOR
            and "token_ids_first40" in prev
        ):
            m = prev
            print(
                f"  [resume] HELD-OUT {alpha_label} q={q_sel} mism={m['mismatches']}/240 "
                f"cp95_ub={m['cp95_upper_1sided']:.6f} cert={m['certified_pct']:.2f}% swap={m['mem'].get('swapouts_delta')}",
                flush=True,
            )
        elif q_sel in audit_by_q:
            m = dict(audit_by_q[q_sel])
            m["alpha_target"] = 0.05 if alpha_label == "alpha_0.05" else 0.01
            m["reused_identical_q"] = True
        else:
            run_unmeasured_warmup(f"heldout240_{alpha_label}_q{q_sel}", ["--quant", "q4_scrub", "--conformal-q", f"{q_sel}"], WARMUP1_PROMPTS_TXT, max_tokens=16)
            r = run_native_with_mem_guard(
                f"heldout_L{length}_{alpha_label}_q{q_sel}",
                HELDOUT_PROMPTS_TXT,
                240,
                ["--quant", "q4_scrub", "--conformal-q", f"{q_sel}"],
                extra_env=env_trace,
                max_tokens=length,
            )
            m = compare_vs_bf16_detailed(test_bf["rows"], test_prompts, r, bf16_40_baseline)
            m["q"] = q_sel
            m["alpha_target"] = 0.05 if alpha_label == "alpha_0.05" else 0.01
            m["reused_identical_q"] = False
            m["token_ids_first40"] = [row["token_ids"] for row in r["rows"][:40]]
            audit_by_q[q_sel] = m

        audit_results[alpha_label] = m
        _save_state({
            "extended_upward": extended_upward,
            "extension_reason": extension_reason,
            "q_hat_005": q_hat_005,
            "q_hat_001": q_hat_001,
            "q_hat_005_monotone": q_hat_005_mono,
            "q_hat_001_monotone": q_hat_001_mono,
            "heldout_audit": audit_results,
        })
        print(
            f"[audit-L{length}] {alpha_label} q={q_sel}: mism={m['mismatches']}/240 "
            f"cp95_ub={m['cp95_upper_1sided']:.6f} cert={m['certified_pct']:.2f}% "
            f"cert_div={m['certified_first_divergences']} fb_div={m['fallback_first_divergences']} "
            f"dec={m['agg_decode_tok_s']} tok/s e2e={m['agg_e2e_tok_s']} tok/s "
            f"swap={m['mem']['swapouts_delta']} timing_valid={m['timing_valid']} "
            f"sp_dec={m['speedup_vs_bf16_decode']} sp_e2e={m['speedup_vs_bf16_e2e']} "
            f"first_div_indices={[p['first_div_token_idx'] for p in m['mismatch_positions']]}",
            flush=True,
        )

    existing_h40 = state_doc.get("heldout40_timing", {})
    heldout40_timing: dict[str, dict] = {}
    h40_by_q: dict[float, dict] = {}
    for alpha_label, q_sel in [("alpha_0.05", q_hat_005), ("alpha_0.01", q_hat_001)]:
        prev40 = existing_h40.get(alpha_label)
        if (
            isinstance(prev40, dict)
            and float(prev40.get("q", -1.0)) == q_sel
            and prev40.get("mem", {}).get("min_free_gb", 0.0) >= MIN_FREE_GB_FLOOR
        ):
            m40 = prev40
            print(
                f"  [resume] HELD-OUT40 {alpha_label} q={q_sel} dec={m40['agg_decode_tok_s']} "
                f"e2e={m40['agg_e2e_tok_s']} swap={m40['mem']['swapouts_delta']} timing_valid={m40.get('timing_valid')}",
                flush=True,
            )
        elif q_sel in h40_by_q:
            m40 = dict(h40_by_q[q_sel])
            m40["alpha_target"] = 0.05 if alpha_label == "alpha_0.05" else 0.01
            m40["reused_identical_q"] = True
        else:
            run_unmeasured_warmup(f"heldout40_{alpha_label}_q{q_sel}", ["--quant", "q4_scrub", "--conformal-q", f"{q_sel}"], WARMUP1_PROMPTS_TXT, max_tokens=16)
            r40 = run_native_with_mem_guard(
                f"heldout40_L{length}_{alpha_label}_q{q_sel}",
                HELDOUT40_PROMPTS_TXT,
                40,
                ["--quant", "q4_scrub", "--conformal-q", f"{q_sel}"],
                extra_env=env_trace,
                max_tokens=length,
                require_timing_clean=True,
                max_attempts=2,
                allow_unclean_timing_fallback=True,
            )
            ref240_first40 = audit_results[alpha_label]["token_ids_first40"]
            match_240 = sum(1 for a, b in zip(r40["rows"], ref240_first40) if a["token_ids"] == b)
            assert match_240 == 40, f"heldout40 {alpha_label} mismatch vs 240-prompt audit[:40]: {match_240}/40"
            m40 = compare_vs_bf16_detailed(test_bf["rows"][:40], test_prompts[:40], r40, bf16_40_baseline)
            m40["q"] = q_sel
            m40["alpha_target"] = 0.05 if alpha_label == "alpha_0.05" else 0.01
            m40["reused_identical_q"] = False
            m40["parity_vs_audit240_first40"] = match_240
            h40_by_q[q_sel] = m40

        heldout40_timing[alpha_label] = m40
        print(
            f"[timing40-L{length}] {alpha_label} q={q_sel}: dec={m40['agg_decode_tok_s']} tok/s "
            f"({m40['speedup_vs_bf16_decode']}x BF16_40) e2e={m40['agg_e2e_tok_s']} tok/s "
            f"({m40['speedup_vs_bf16_e2e']}x BF16_40) cert={m40['certified_pct']:.2f}% "
            f"swap={m40['mem']['swapouts_delta']} min_free_gb={m40['mem']['min_free_gb']}",
            flush=True,
        )

    a05 = audit_results["alpha_0.05"]
    a01 = audit_results["alpha_0.01"]
    pass_005 = a05["cp95_upper_1sided"] <= 0.05
    pass_001 = a01["cp95_upper_1sided"] <= 0.01

    final_extra = {
        "extended_upward": extended_upward,
        "extension_reason": extension_reason,
        "q_hat_005": q_hat_005,
        "q_hat_001": q_hat_001,
        "q_hat_005_monotone": q_hat_005_mono,
        "q_hat_001_monotone": q_hat_001_mono,
        "heldout_audit": audit_results,
        "heldout40_timing": heldout40_timing,
        "step4_gate": {
            "mode": "per_alpha",
            "rule": "cp95_upper_1sided <= alpha",
            "alpha_0.05": {
                "pass": bool(pass_005),
                "cp95_upper_1sided": a05["cp95_upper_1sided"],
                "observed": f"{a05['mismatches']}/{a05['n_prompts']}",
                "status": "passed" if pass_005 else f"failed, {a05['mismatches']}/{a05['n_prompts']} observed",
            },
            "alpha_0.01": {
                "pass": bool(pass_001),
                "cp95_upper_1sided": a01["cp95_upper_1sided"],
                "observed": f"{a01['mismatches']}/{a01['n_prompts']}",
                "status": "passed" if pass_001 else f"not demonstrated, {a01['mismatches']}/{a01['n_prompts']} observed",
            },
        },
    }
    _save_state(final_extra)
    print(
        f"[done-L{length}] alpha_0.05={pass_005} ({a05['mismatches']}/{a05['n_prompts']}, cp95_ub={a05['cp95_upper_1sided']:.6f}) "
        f"alpha_0.01={pass_001} ({final_extra['step4_gate']['alpha_0.01']['status']}, cp95_ub={a01['cp95_upper_1sided']:.6f}) "
        f"-> {out_path}",
        flush=True,
    )


def main() -> None:
    global MODEL_DIR, MODEL_TAG, MODEL_LABEL, MIN_FREE_GB_FLOOR
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--model-dir", type=Path, default=MODEL_8B)
    parser.add_argument("--tag", type=str, default="qwen3_8b")
    parser.add_argument("--free-floor-gb", type=float, default=MIN_FREE_GB_FLOOR)
    parser.add_argument("--check-mem", action="store_true")
    args = parser.parse_args()

    MODEL_DIR = args.model_dir.resolve()
    MODEL_TAG = args.tag
    MODEL_LABEL = MODEL_DIR.name
    MIN_FREE_GB_FLOOR = float(args.free_floor_gb)

    if args.check_mem:
        mem = sample_macos_memory()
        cal_prompts, test_prompts = build_disjoint_prompt_sets()
        write_prompts_file(cal_prompts, CAL_PROMPTS_TXT)
        write_prompts_file(test_prompts, HELDOUT_PROMPTS_TXT)
        print(f"[check-mem] {json.dumps(mem)} cal={len(cal_prompts)} heldout={len(test_prompts)}", flush=True)
        return

    run_length_pipeline(args.length)


if __name__ == "__main__":
    main()
