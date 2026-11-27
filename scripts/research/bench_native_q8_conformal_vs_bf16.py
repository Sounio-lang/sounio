#!/usr/bin/env python3
"""
Phase 10 v2 Native Benchmark (ZERO PyTorch in the engine or harness):
Pure native bin/sounio-llm comparison on Apple Silicon Metal NoCopy across:
  - Matched batched-prefill bf16 baseline & q8_cone
  - R1 q8_conformal (alpha in {0.05, 0.01})
  - R2 q8_mlp_conformal (alpha in {0.05, 0.01})
  - R3 q8_scrub (alpha in {0.05, 0.01})
  - R4 q8_exact_scrub (q=0.85 and q=1.291984)
  - Ctrl Fixed-K=8 speculative control (q8_exact_scrub, q=0.0, SOUNIO_SCRUB_MAX_STREAK=8)

Part A: Persistent single-init --bench-prompts across 3 interleaved passes (20 prompts).
Part B: Native SOUNIO_LLM_TRACE_STEPS=1 step-trace first-divergence audit (R1 probe, R3 0.05/0.01).
Part C: Secondary subprocess per-prompt CLI comparison (1 interleaved pass, 5 key modes).
"""

import json
import mmap
import os
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

BIN_PATH = Path("/Users/demetriosagourakis/dev/sounio-worktrees/llm-backend/bin/sounio-llm")
MODEL_DIR = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B")
METAL_DYLIB = Path("/Users/demetriosagourakis/dev/sounio-worktrees/llm-backend/artifacts/self-hosted/libsounio_metal.dylib")
OUT_JSON = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B-ossm-sidecar/phase10_native_conformal_v2_benchmark.json")
PROMPTS_FILE = Path("/tmp/phase10_heldout20_prompts.txt")
WARMUP_FILE = Path("/tmp/phase10_warmup2_prompts.txt")

# 20 disjoint held-out prompts (zero overlap with Step 1d 240-prompt calibration split)
HELDOUT_20_PROMPTS = [
    # History (5)
    ("heldout_history_00", "history", "Historical institutional summary of Solonian Seisachtheia decrees (594 BC, Athens): examine how cancellation of Attic agricultural debt bondage and timocratic classes. Explain the fiscal and political consequences."),
    ("heldout_history_03", "history", "Historical institutional summary of Carolingian Silver Denier standard (794 AD, Frankfurt): examine how standardized pound-shilling-penny mint weight across Frankish counties. Explain the fiscal and political consequences."),
    ("heldout_history_05", "history", "Historical institutional summary of Magna Carta baronial clauses (1215 AD, Runnymede): examine how statutory restraint upon arbitrary royal scutage and wardship tallage. Explain the fiscal and political consequences."),
    ("heldout_history_08", "history", "Historical institutional summary of Dutch Vereenigde Oostindische share (1602 AD, Amsterdam): examine how permanent joint-stock equity capital tradable on secondary exchange. Explain the fiscal and political consequences."),
    ("heldout_history_10", "history", "Historical institutional summary of Code Napoleon civil property (1804 AD, Paris): examine how unified metric contract enforcement and partible freehold inheritance. Explain the fiscal and political consequences."),
    # Math (5)
    ("heldout_math_00", "math", "Theorem statement (Stokes exterior derivative theorem in oriented manifolds with boundary): verify why integral of a differential form over a boundary equals its exterior derivative interior integral. Connect this directly to vector calculus with explicit hypotheses."),
    ("heldout_math_01", "math", "Theorem statement (Euler-Lagrange stationarity condition in smooth action functionals): verify why vanishing first variation yields the second-order differential equations along extrema. Connect this directly to variational paths with explicit hypotheses."),
    ("heldout_math_04", "math", "Theorem statement (Riesz representation duality theorem in Hilbert spaces): verify why every continuous linear functional corresponds isometrically to inner product with a unique vector. Connect this directly to adjoint operators with explicit hypotheses."),
    ("heldout_math_08", "math", "Theorem statement (Courant-Fischer minimax characterization in Hermitian linear operators): verify why ordered real eigenvalues arise as minimax Rayleigh quotients over nested subspaces. Connect this directly to spectral gaps with explicit hypotheses."),
    ("heldout_math_13", "math", "Theorem statement (Lagrange multiplier KKT stationarity in constrained smooth optimization): verify why linear independence qualification ensures dual multipliers certify constrained local optima. Connect this directly to primal-dual gap with explicit hypotheses."),
    # Systems (5)
    ("heldout_systems_00", "systems", "Systems design review for `dijkstra_minheap` (non-negative single-source shortest paths): the critical path relies on binary min-heap decrease-key relaxation. Analyze memory-ordering hazards and why this preserves O(E log V) frontier."),
    ("heldout_systems_01", "systems", "Systems design review for `robin_hood_hash` (open-addressed flat hash table): the critical path relies on probe distance variance balancing. Analyze memory-ordering hazards and why this preserves high load-factor lookup."),
    ("heldout_systems_02", "systems", "Systems design review for `raft_append_log` (consensus replicated state machine): the critical path relies on term matching and prevLogIndex check. Analyze memory-ordering hazards and why this preserves quorum commit advancement."),
    ("heldout_systems_03", "systems", "Systems design review for `mvcc_tuple_vis` (snapshot isolation tuple scanner): the critical path relies on xmin and xmax transaction visibility. Analyze memory-ordering hazards and why this preserves lock-free reader scans."),
    ("heldout_systems_07", "systems", "Systems design review for `metal_nocopy_buf` (Apple Silicon unified memory buffer): the critical path relies on zero-copy page-aligned MTLBuffer. Analyze memory-ordering hazards and why this preserves shared CPU-GPU visibility."),
    # Clinical (5)
    ("heldout_clinical_00", "clinical", "Clinical pharmacology summary for Vancomycin (D-Ala-D-Ala peptidoglycan cell-wall cross-link inhibitor): disposition proceeds via predominantly passive glomerular filtration matching creatinine clearance. Outline therapeutic drug monitoring around AUC24/MIC 400-600 and major interactions."),
    ("heldout_clinical_01", "clinical", "Clinical pharmacology summary for Tacrolimus (FKBP12-calcineurin NFAT transcription inhibitor): disposition proceeds via intestinal P-gp efflux and CYP3A4/5 hepatic clearance. Outline therapeutic drug monitoring around whole-blood 12h trough ng/mL and major interactions."),
    ("heldout_clinical_06", "clinical", "Clinical pharmacology summary for Meropenem (penicillin-binding protein PBP2/PBP3 carbapenem beta-lactam): disposition proceeds via renal dehydropeptidase-I stability and tubular secretion. Outline therapeutic drug monitoring around percent fT above MIC and major interactions."),
    ("heldout_clinical_10", "clinical", "Clinical pharmacology summary for Voriconazole (fungal lanosterol 14-alpha-demethylase CYP51 triazole): disposition proceeds via saturable non-linear Michaelis-Menten CYP2C19 clearance. Outline therapeutic drug monitoring around 2-5.5 mg/L plasma trough and major interactions."),
    ("heldout_clinical_12", "clinical", "Clinical pharmacology summary for Isoniazid (mycobacterial InhA mycolic acid synthase hydrazide): disposition proceeds via polymorphic NAT2 hepatic N-acetylation slow vs fast phenotypes. Outline therapeutic drug monitoring around transaminase and B6 neuropathy and major interactions."),
]

PERSISTENT_MODES = [
    ("bf16", ["--quant", "bf16"], {}),
    ("q8_cone", ["--quant", "q8_cone"], {}),
    ("r1_q8_conformal_005", ["--quant", "q8_conformal", "--conformal-mode", "flip_exact", "--conformal-alpha", "0.05"], {}),
    ("r1_q8_conformal_001", ["--quant", "q8_conformal", "--conformal-mode", "flip_exact", "--conformal-alpha", "0.01"], {}),
    ("r2_q8_mlp_conformal_005", ["--quant", "q8_mlp_conformal", "--conformal-alpha", "0.05"], {}),
    ("r2_q8_mlp_conformal_001", ["--quant", "q8_mlp_conformal", "--conformal-alpha", "0.01"], {}),
    ("r3_q8_scrub_005", ["--quant", "q8_scrub", "--conformal-alpha", "0.05"], {}),
    ("r3_q8_scrub_001", ["--quant", "q8_scrub", "--conformal-alpha", "0.01"], {}),
    ("r4_q8_exact_scrub_q085", ["--quant", "q8_exact_scrub"], {}),
    ("r4_q8_exact_scrub_q1292", ["--quant", "q8_exact_scrub", "--conformal-q", "1.291984"], {}),
    ("ctrl_q8_fixed_k8_spec", ["--quant", "q8_exact_scrub", "--conformal-q", "0.0"], {"SOUNIO_SCRUB_MAX_STREAK": "8"}),
]

TRACE_AUDIT_MODES = [
    ("r1_probe_q1292", ["--quant", "q8_conformal", "--conformal-mode", "flip_exact", "--conformal-q", "1.291984"], {}),
    ("r3_q8_scrub_005", ["--quant", "q8_scrub", "--conformal-alpha", "0.05"], {}),
    ("r3_q8_scrub_001", ["--quant", "q8_scrub", "--conformal-alpha", "0.01"], {}),
]

SUBPROC_MODES = [
    ("bf16", ["--quant", "bf16"], {}),
    ("r3_q8_scrub_005", ["--quant", "q8_scrub", "--conformal-alpha", "0.05"], {}),
    ("r4_q8_exact_scrub_q085", ["--quant", "q8_exact_scrub"], {}),
    ("r4_q8_exact_scrub_q1292", ["--quant", "q8_exact_scrub", "--conformal-q", "1.291984"], {}),
    ("ctrl_q8_fixed_k8_spec", ["--quant", "q8_exact_scrub", "--conformal-q", "0.0"], {"SOUNIO_SCRUB_MAX_STREAK": "8"}),
]

STAT_RE = re.compile(
    r"\[sounio-llm:\s*prompt_tokens=(\d+)\s+gen_tokens=(\d+)\s+top1_margin=([-+0-9.eE]+)\s+"
    r"last_assoc_l2=([-+0-9.eE]+)\s+prefill_ms=([-+0-9.eE]+)\s+decode_ms=([-+0-9.eE]+)\s+"
    r"decode_tok_per_sec=([-+0-9.eE]+)\s+elapsed_ms=([-+0-9.eE]+)\s+tok_per_sec=([-+0-9.eE]+)\s+"
    r"certified=(\d+)\s+escalated=(\d+)\s+last_combined_err=([-+0-9.eE]+)\]"
)

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


def build_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["SOUNIO_METAL_RUNTIME"] = str(METAL_DYLIB)
    for k in ("SOUNIO_CONFORMAL_Q", "SOUNIO_SCRUB_MAX_STREAK", "SOUNIO_LLM_TRACE_STEPS"):
        env.pop(k, None)
    if extra_env:
        env.update(extra_env)
    return env


def start_warmer(model_file: Path, stop_ev: threading.Event) -> threading.Thread:
    ready_ev = threading.Event()

    def _run():
        with open(model_file, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            _ = bytes(mm[::16384])
            ready_ev.set()
            while not stop_ev.is_set():
                stop_ev.wait(1.5)
                if not stop_ev.is_set():
                    _ = bytes(mm[::65536])
            mm.close()

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    ready_ev.wait(timeout=30.0)
    return th


def write_prompt_files() -> None:
    lines = [ptxt.replace("\n", " ").strip() for _, _, ptxt in HELDOUT_20_PROMPTS]
    PROMPTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    WARMUP_FILE.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")


def run_persistent_bench(
    prompts_file: Path,
    expected_n: int,
    extra_args: list[str],
    extra_env: dict[str, str] | None = None,
    max_tokens: int = 64,
) -> dict:
    cmd = [
        str(BIN_PATH),
        "--model-dir", str(MODEL_DIR),
        "--backend", "metal",
        "--max-tokens", str(max_tokens),
        "--no-think",
        "--temp", "0",
        "--bench-prompts", str(prompts_file),
    ] + extra_args
    env = build_env(extra_env)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"bin/sounio-llm --bench-prompts exited {proc.returncode}: cmd={cmd}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
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
                raise RuntimeError(f"bench_tokens count mismatch at idx={idx}: n={n_tok} parsed={len(ids)}")
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

    expected_keys = set(range(expected_n))
    if set(toks_by_idx.keys()) != expected_keys:
        raise RuntimeError(f"Missing bench_tokens indices: got={sorted(toks_by_idx.keys())} expected={expected_n}")
    if set(stats_by_idx.keys()) != expected_keys:
        raise RuntimeError(f"Missing bench_prompt indices: got={sorted(stats_by_idx.keys())} expected={expected_n}")

    rows = []
    for idx in range(expected_n):
        row = dict(stats_by_idx[idx])
        row["prompt_id"] = HELDOUT_20_PROMPTS[idx][0]
        row["domain"] = HELDOUT_20_PROMPTS[idx][1]
        row["token_ids"] = toks_by_idx[idx]
        if row["gen_tokens"] != len(row["token_ids"]):
            raise RuntimeError(f"gen_tokens mismatch at idx={idx}: {row['gen_tokens']} vs {len(row['token_ids'])}")
        rows.append(row)

    tot_gen = sum(r["gen_tokens"] for r in rows)
    tot_decode_s = sum(r["decode_ms"] for r in rows) / 1000.0
    tot_elapsed_s = sum(r["elapsed_ms"] for r in rows) / 1000.0
    return {
        "wall_s": wall_s,
        "total_gen_tokens": tot_gen,
        "agg_decode_tok_per_sec": (tot_gen / tot_decode_s) if tot_decode_s > 0 else 0.0,
        "agg_e2e_tok_per_sec": (tot_gen / tot_elapsed_s) if tot_elapsed_s > 0 else 0.0,
        "agg_wall_tok_per_sec": (tot_gen / wall_s) if wall_s > 0 else 0.0,
        "rows": rows,
        "traces_by_idx": traces_by_idx,
    }


def run_single(
    prompt: str,
    extra_args: list[str],
    extra_env: dict[str, str] | None = None,
    max_tokens: int = 64,
) -> dict:
    cmd = [
        str(BIN_PATH),
        "--model-dir", str(MODEL_DIR),
        "--backend", "metal",
        "--max-tokens", str(max_tokens),
        "--no-think",
        "--temp", "0",
        "--prompt", prompt,
    ] + extra_args
    env = build_env(extra_env)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"bin/sounio-llm exited {proc.returncode}: cmd={cmd}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    lines = proc.stdout.splitlines()
    stat_line = ""
    text_lines = []
    for ln in lines:
        if ln.startswith("[sounio-llm:"):
            stat_line = ln
        else:
            text_lines.append(ln)
    m = STAT_RE.search(stat_line)
    if not m:
        raise RuntimeError(f"Failed to parse stats line from output:\n{proc.stdout}")
    gen_text = "\n".join(text_lines).strip()
    if not gen_text:
        raise RuntimeError(f"Empty generation text for cmd={cmd}")
    return {
        "prompt_tokens": int(m.group(1)),
        "gen_tokens": int(m.group(2)),
        "top1_margin": float(m.group(3)),
        "prefill_ms": float(m.group(5)),
        "decode_ms": float(m.group(6)),
        "decode_tok_per_sec": float(m.group(7)),
        "elapsed_ms": float(m.group(8)),
        "tok_per_sec": float(m.group(9)),
        "certified": int(m.group(10)),
        "escalated": int(m.group(11)),
        "last_combined_err": float(m.group(12)),
        "wall_ms": wall_s * 1000.0,
        "wall_tok_per_sec": int(m.group(2)) / wall_s if wall_s > 0 else 0.0,
        "text": gen_text,
    }


def main() -> None:
    write_prompt_files()
    n_prompts = len(HELDOUT_20_PROMPTS)

    stop_ev = threading.Event()
    warmer = start_warmer(MODEL_DIR / "model.safetensors", stop_ev)
    try:
        print("=" * 116, flush=True)
        print("PHASE 10 v2 NATIVE BENCHMARK — bin/sounio-llm (Metal NoCopy, ZERO PyTorch)", flush=True)
        print(f"Model: {MODEL_DIR} | Prompts: {n_prompts} disjoint held-out | max_tokens=64 | --no-think --temp 0", flush=True)
        print("=" * 116, flush=True)

        # Persistent warmup (2 prompts x 8 tokens)
        print("[Warmup] Running bf16 and r4_q8_exact_scrub_q085 on 2 prompts (max_tokens=8)...", flush=True)
        _ = run_persistent_bench(WARMUP_FILE, 2, ["--quant", "bf16"], {}, max_tokens=8)
        _ = run_persistent_bench(WARMUP_FILE, 2, ["--quant", "q8_exact_scrub"], {}, max_tokens=8)

        # ------------------------------------------------------------------
        # Part A: Primary Persistent --bench-prompts (3 interleaved passes)
        # ------------------------------------------------------------------
        print("\n[Part A] Persistent --bench-prompts across 3 interleaved passes (11 modes)...", flush=True)
        pass_runs: dict[str, list[dict]] = {name: [] for name, _, _ in PERSISTENT_MODES}
        for pass_idx in range(3):
            rot = (pass_idx * 4) % len(PERSISTENT_MODES)
            ordered = PERSISTENT_MODES[rot:] + PERSISTENT_MODES[:rot]
            print(f"\n--- Pass {pass_idx + 1}/3 (rot={rot}) ---", flush=True)
            for mname, margs, menv in ordered:
                res = run_persistent_bench(PROMPTS_FILE, n_prompts, margs, menv, max_tokens=64)
                pass_runs[mname].append(res)
                print(
                    f"  [p{pass_idx + 1}] {mname:24s} | agg_dec={res['agg_decode_tok_per_sec']:6.2f} tok/s | "
                    f"agg_e2e={res['agg_e2e_tok_per_sec']:6.2f} tok/s | wall={res['agg_wall_tok_per_sec']:6.2f} tok/s",
                    flush=True,
                )

        # Verify determinism across passes per mode
        for mname, _, _ in PERSISTENT_MODES:
            p0, p1, p2 = pass_runs[mname]
            for i in range(n_prompts):
                t0 = p0["rows"][i]["token_ids"]
                t1 = p1["rows"][i]["token_ids"]
                t2 = p2["rows"][i]["token_ids"]
                if not (t0 == t1 == t2):
                    raise RuntimeError(f"Non-deterministic token_ids across passes for mode={mname} idx={i}")

        bf16_token_ids = [r["token_ids"] for r in pass_runs["bf16"][0]["rows"]]

        persistent_summary: dict[str, dict] = {}
        bf_med_dec = statistics.median(r["agg_decode_tok_per_sec"] for r in pass_runs["bf16"])
        bf_med_e2e = statistics.median(r["agg_e2e_tok_per_sec"] for r in pass_runs["bf16"])
        bf_med_wall = statistics.median(r["agg_wall_tok_per_sec"] for r in pass_runs["bf16"])

        bf_pool_rows = [row for r in pass_runs["bf16"] for row in r["rows"]]
        bf_pool_gen = sum(row["gen_tokens"] for row in bf_pool_rows)
        bf_pool_dec = bf_pool_gen / (sum(row["decode_ms"] for row in bf_pool_rows) / 1000.0)
        bf_pool_e2e = bf_pool_gen / (sum(row["elapsed_ms"] for row in bf_pool_rows) / 1000.0)

        print("\n" + "=" * 116, flush=True)
        print("PART A SUMMARY (--bench-prompts, 3 passes, 20 held-out prompts)", flush=True)
        print("Mode                     | Med Agg Dec | Spd Dec | Med Agg E2E | Spd E2E | Med Wall | Cert/Gen (%)       | Esc/Gen (%)     | Exact BF16", flush=True)
        print("-" * 116, flush=True)

        for mname, _, _ in PERSISTENT_MODES:
            plist = pass_runs[mname]
            med_agg_dec = statistics.median(r["agg_decode_tok_per_sec"] for r in plist)
            med_agg_e2e = statistics.median(r["agg_e2e_tok_per_sec"] for r in plist)
            med_agg_wall = statistics.median(r["agg_wall_tok_per_sec"] for r in plist)

            all_rows = [row for r in plist for row in r["rows"]]
            pool_gen = sum(row["gen_tokens"] for row in all_rows)
            pool_dec_s = sum(row["decode_ms"] for row in all_rows) / 1000.0
            pool_e2e_s = sum(row["elapsed_ms"] for row in all_rows) / 1000.0
            pool_agg_dec = pool_gen / pool_dec_s if pool_dec_s > 0 else 0.0
            pool_agg_e2e = pool_gen / pool_e2e_s if pool_e2e_s > 0 else 0.0

            per_prompt_med_dec = [
                statistics.median(plist[k]["rows"][i]["decode_tok_per_sec"] for k in range(3))
                for i in range(n_prompts)
            ]
            per_prompt_med_e2e = [
                statistics.median(plist[k]["rows"][i]["tok_per_sec"] for k in range(3))
                for i in range(n_prompts)
            ]

            r0 = plist[0]["rows"]
            matches = sum(1 for i in range(n_prompts) if r0[i]["token_ids"] == bf16_token_ids[i])
            gen0 = sum(row["gen_tokens"] for row in r0)
            cert0 = sum(row["certified"] for row in r0)
            esc0 = sum(row["escalated"] for row in r0)
            cert_pct = (cert0 / gen0 * 100.0) if gen0 > 0 else 0.0
            esc_pct = (esc0 / gen0 * 100.0) if gen0 > 0 else 0.0

            spd_dec = (med_agg_dec / bf_med_dec) if bf_med_dec > 0 else 1.0
            spd_pool_dec = (pool_agg_dec / bf_pool_dec) if bf_pool_dec > 0 else 1.0
            spd_e2e = (med_agg_e2e / bf_med_e2e) if bf_med_e2e > 0 else 1.0
            spd_wall = (med_agg_wall / bf_med_wall) if bf_med_wall > 0 else 1.0

            persistent_summary[mname] = {
                "prompts": n_prompts,
                "passes": 3,
                "total_gen_tokens_per_pass": gen0,
                "pooled_gen_tokens": pool_gen,
                "pass_agg_decode_tok_per_sec": [round(r["agg_decode_tok_per_sec"], 4) for r in plist],
                "pass_agg_e2e_tok_per_sec": [round(r["agg_e2e_tok_per_sec"], 4) for r in plist],
                "pass_agg_wall_tok_per_sec": [round(r["agg_wall_tok_per_sec"], 4) for r in plist],
                "median_agg_decode_tok_per_sec": round(med_agg_dec, 4),
                "pooled_agg_decode_tok_per_sec": round(pool_agg_dec, 4),
                "prompt_median_mean_decode_tok_per_sec": round(statistics.mean(per_prompt_med_dec), 4),
                "prompt_median_median_decode_tok_per_sec": round(statistics.median(per_prompt_med_dec), 4),
                "speedup_decode_vs_bf16": round(spd_dec, 4),
                "speedup_pooled_decode_vs_bf16": round(spd_pool_dec, 4),
                "median_agg_e2e_tok_per_sec": round(med_agg_e2e, 4),
                "pooled_agg_e2e_tok_per_sec": round(pool_agg_e2e, 4),
                "prompt_median_mean_e2e_tok_per_sec": round(statistics.mean(per_prompt_med_e2e), 4),
                "prompt_median_median_e2e_tok_per_sec": round(statistics.median(per_prompt_med_e2e), 4),
                "speedup_e2e_vs_bf16": round(spd_e2e, 4),
                "median_agg_wall_tok_per_sec": round(med_agg_wall, 4),
                "speedup_wall_vs_bf16": round(spd_wall, 4),
                "certified_tokens": cert0,
                "escalated_events": esc0,
                "certified_pct_of_gen": round(cert_pct, 2),
                "escalated_per_gen_pct": round(esc_pct, 2),
                "exact_token_matches_bf16": matches,
                "exact_token_match_pct": round(matches / n_prompts * 100.0, 2),
            }

            print(
                f"{mname:24s} | {med_agg_dec:11.2f} | {spd_dec:6.3f}x | {med_agg_e2e:11.2f} | {spd_e2e:6.3f}x | "
                f"{med_agg_wall:8.2f} | {cert0:4d}/{gen0:4d} ({cert_pct:5.2f}%) | {esc0:4d}/{gen0:4d} ({esc_pct:5.2f}%) | "
                f"{matches:2d}/20 ({matches / n_prompts * 100.0:5.1f}%)",
                flush=True,
            )

        # ------------------------------------------------------------------
        # Part B: Native Step-Trace First-Divergence Audit
        # ------------------------------------------------------------------
        print("\n[Part B] Native Step-Trace First-Divergence Audit (SOUNIO_LLM_TRACE_STEPS=1)...", flush=True)
        trace_audit_summary: dict[str, dict] = {}
        for mname, margs, menv in TRACE_AUDIT_MODES:
            env_tr = dict(menv)
            env_tr["SOUNIO_LLM_TRACE_STEPS"] = "1"
            res = run_persistent_bench(PROMPTS_FILE, n_prompts, margs, env_tr, max_tokens=64)
            prompt_mismatches = 0
            cert_first_div = 0
            fb_first_div = 0
            details = []
            for idx in range(n_prompts):
                bf_toks = bf16_token_ids[idx]
                tr = res["traces_by_idx"].get(idx, [])
                tr.sort(key=lambda x: x["idx"])
                if len(tr) != len(res["rows"][idx]["token_ids"]):
                    raise RuntimeError(
                        f"Trace length mismatch for {mname} prompt={idx}: trace={len(tr)} vs gen={len(res['rows'][idx]['token_ids'])}"
                    )
                for k, st in enumerate(tr):
                    if st["idx"] != k or st["tok"] != res["rows"][idx]["token_ids"][k]:
                        raise RuntimeError(f"Trace step alignment mismatch for {mname} prompt={idx} k={k}")
                min_len = min(len(tr), len(bf_toks))
                div_pos = -1
                for k in range(min_len):
                    if tr[k]["tok"] != bf_toks[k]:
                        div_pos = k
                        break
                if div_pos >= 0 or len(tr) != len(bf_toks):
                    prompt_mismatches += 1
                    if div_pos >= 0:
                        st = tr[div_pos]
                        if st["cert"] == 1:
                            cert_first_div += 1
                            kind = "certified"
                        else:
                            fb_first_div += 1
                            kind = "fallback"
                        details.append({
                            "prompt_idx": idx,
                            "prompt_id": HELDOUT_20_PROMPTS[idx][0],
                            "first_div_pos": div_pos,
                            "kind": kind,
                            "tok": st["tok"],
                            "bf16_tok": bf_toks[div_pos],
                            "q8_top1": st["q8_top1"],
                            "margin": st["margin"],
                        })
                    else:
                        kind = "length_mismatch"
                        details.append({
                            "prompt_idx": idx,
                            "prompt_id": HELDOUT_20_PROMPTS[idx][0],
                            "first_div_pos": min_len,
                            "kind": kind,
                            "len_mode": len(tr),
                            "len_bf16": len(bf_toks),
                        })

            trace_audit_summary[mname] = {
                "prompts": n_prompts,
                "prompt_mismatches": prompt_mismatches,
                "certified_first_divergences": cert_first_div,
                "fallback_first_divergences": fb_first_div,
                "details": details,
            }
            print(
                f"  {mname:24s} | mismatches={prompt_mismatches:2d}/20 | "
                f"certified_first_div={cert_first_div} | fallback_first_div={fb_first_div}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # Part C: Secondary Subprocess CLI Benchmark (1 pass, interleaved)
        # ------------------------------------------------------------------
        print("\n[Part C] Secondary Subprocess per-prompt CLI Benchmark (1 interleaved pass, 5 modes)...", flush=True)
        subproc_runs: dict[str, list[dict]] = {name: [] for name, _, _ in SUBPROC_MODES}
        for idx, (pid, dom, ptxt) in enumerate(HELDOUT_20_PROMPTS):
            rot = idx % len(SUBPROC_MODES)
            ordered = SUBPROC_MODES[rot:] + SUBPROC_MODES[:rot]
            row_map = {}
            for mname, margs, menv in ordered:
                r = run_single(ptxt, margs, menv, max_tokens=64)
                r["prompt_id"] = pid
                r["domain"] = dom
                row_map[mname] = r
            bf_txt = row_map["bf16"]["text"]
            for mname, _, _ in SUBPROC_MODES:
                row_map[mname]["exact_match_bf16"] = (row_map[mname]["text"] == bf_txt)
                subproc_runs[mname].append(row_map[mname])

        subproc_summary: dict[str, dict] = {}
        sub_bf_dec = 0.0
        sub_bf_e2e = 0.0
        sub_bf_wall = 0.0

        print("-" * 100, flush=True)
        print("PART C SUMMARY (subprocess CLI, 1 pass, 20 prompts)", flush=True)
        print("Mode                     | Agg Dec tok/s | Spd Dec | Agg E2E tok/s | Spd E2E | Agg Wall | Exact BF16", flush=True)
        print("-" * 100, flush=True)
        for mname, _, _ in SUBPROC_MODES:
            rows = subproc_runs[mname]
            tot_gen = sum(r["gen_tokens"] for r in rows)
            tot_dec_s = sum(r["decode_ms"] for r in rows) / 1000.0
            tot_e2e_s = sum(r["elapsed_ms"] for r in rows) / 1000.0
            tot_wall_s = sum(r["wall_ms"] for r in rows) / 1000.0
            agg_dec = (tot_gen / tot_dec_s) if tot_dec_s > 0 else 0.0
            agg_e2e = (tot_gen / tot_e2e_s) if tot_e2e_s > 0 else 0.0
            agg_wall = (tot_gen / tot_wall_s) if tot_wall_s > 0 else 0.0
            matches = sum(1 for r in rows if r["exact_match_bf16"])

            if mname == "bf16":
                sub_bf_dec = agg_dec
                sub_bf_e2e = agg_e2e
                sub_bf_wall = agg_wall

            spd_dec = (agg_dec / sub_bf_dec) if sub_bf_dec > 0 else 1.0
            spd_e2e = (agg_e2e / sub_bf_e2e) if sub_bf_e2e > 0 else 1.0
            spd_wall = (agg_wall / sub_bf_wall) if sub_bf_wall > 0 else 1.0

            subproc_summary[mname] = {
                "prompts": len(rows),
                "total_gen_tokens": tot_gen,
                "agg_decode_tok_per_sec": round(agg_dec, 4),
                "speedup_decode_vs_bf16": round(spd_dec, 4),
                "agg_e2e_tok_per_sec": round(agg_e2e, 4),
                "speedup_e2e_vs_bf16": round(spd_e2e, 4),
                "agg_wall_tok_per_sec": round(agg_wall, 4),
                "speedup_wall_vs_bf16": round(spd_wall, 4),
                "exact_output_matches_bf16": matches,
                "exact_output_match_pct": round(matches / len(rows) * 100.0, 2),
            }
            print(
                f"{mname:24s} | {agg_dec:13.2f} | {spd_dec:6.3f}x | {agg_e2e:13.2f} | {spd_e2e:6.3f}x | "
                f"{agg_wall:8.2f} | {matches:2d}/20 ({matches / len(rows) * 100.0:5.1f}%)",
                flush=True,
            )

        # Hard correctness assertions
        for req_exact in ("r4_q8_exact_scrub_q085", "r4_q8_exact_scrub_q1292", "ctrl_q8_fixed_k8_spec"):
            pm = persistent_summary[req_exact]["exact_token_matches_bf16"]
            sm = subproc_summary[req_exact]["exact_output_matches_bf16"]
            if pm != n_prompts or sm != n_prompts:
                raise RuntimeError(f"Exact-match assertion failed for {req_exact}: persistent={pm}/{n_prompts}, subproc={sm}/{n_prompts}")

        for req_r3 in ("r3_q8_scrub_005", "r3_q8_scrub_001"):
            fb_div = trace_audit_summary[req_r3]["fallback_first_divergences"]
            if fb_div != 0:
                raise RuntimeError(f"Fallback first-divergence assertion failed for {req_r3}: got {fb_div}")

        out = {
            "benchmark": "phase10_native_q8_conformal_v2_vs_bf16",
            "binary": str(BIN_PATH),
            "model_dir": str(MODEL_DIR),
            "prompts": n_prompts,
            "max_tokens": 64,
            "persistent_passes": 3,
            "persistent_summary": persistent_summary,
            "trace_audit": trace_audit_summary,
            "subprocess_summary": subproc_summary,
            "persistent_runs": {
                mname: [
                    {
                        "pass": k,
                        "wall_s": round(r["wall_s"], 4),
                        "agg_decode_tok_per_sec": round(r["agg_decode_tok_per_sec"], 4),
                        "agg_e2e_tok_per_sec": round(r["agg_e2e_tok_per_sec"], 4),
                        "agg_wall_tok_per_sec": round(r["agg_wall_tok_per_sec"], 4),
                        "rows": [
                            {kk: vv for kk, vv in row.items() if kk != "token_ids"}
                            for row in r["rows"]
                        ],
                    }
                    for k, r in enumerate(pass_runs[mname])
                ]
                for mname, _, _ in PERSISTENT_MODES
            },
            "subprocess_runs": subproc_runs,
        }
        OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nALL ASSERTIONS PASSED. Saved: {OUT_JSON}", flush=True)
    finally:
        stop_ev.set()
        warmer.join(timeout=2.0)


if __name__ == "__main__":
    main()
