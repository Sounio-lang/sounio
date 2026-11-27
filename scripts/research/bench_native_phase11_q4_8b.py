#!/usr/bin/env python3
"""
Phase 11 Native Benchmark (ZERO PyTorch in engine or harness):
Pure native bin/sounio-llm benchmark on Apple Silicon Metal NoCopy for Qwen3-8B:
  - Matched batched-prefill bf16 baseline & q4_cone
  - R1 q4_conformal_probe (--conformal-mode flip_exact --conformal-q <q_hat_005>)
  - R3 q4_scrub (alpha in {0.05, 0.01})
  - R4 q4_exact_scrub (baked-in q_opt_r4)
  - Ctrl Fixed-K=8 speculative control (q4_exact_scrub, q=0.0, SOUNIO_SCRUB_MAX_STREAK=8)
Plus:
  - Native SOUNIO_LLM_TRACE_STEPS=1 step-trace first-divergence audit (R1, R3 0.05/0.01)
  - 1-pass Qwen3-0.6B pruned-head ablation (bf16 vs q8_exact_scrub vs q4_exact_scrub)
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path("/Users/demetriosagourakis/dev/sounio-worktrees/llm-backend")
BIN_PATH = REPO_ROOT / "bin/sounio-llm"
METAL_DYLIB = REPO_ROOT / "artifacts/self-hosted/libsounio_metal.dylib"
MODEL_8B = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-8B")
MODEL_06B = Path("/Users/demetriosagourakis/sounio-research/checkpoints/Qwen3-0.6B")
CAL_JSON = Path("/tmp/phase11_qwen3_8b_q4_cal.json")
OUT_JSON = Path(
    "/Users/demetriosagourakis/sounio-research/checkpoints/"
    "Qwen3-0.6B-ossm-sidecar/phase11_native_q4_8b_benchmark.json"
)
PROMPTS_FILE = Path("/tmp/phase11_heldout20_prompts.txt")
WARMUP_FILE = Path("/tmp/phase11_warmup2_prompts.txt")

HELDOUT_20_PROMPTS = [
    ("heldout_history_00", "history", "Historical institutional summary of Solonian Seisachtheia decrees (594 BC, Athens): examine how cancellation of Attic agricultural debt bondage and timocratic classes. Explain the fiscal and political consequences."),
    ("heldout_history_03", "history", "Historical institutional summary of Carolingian Silver Denier standard (794 AD, Frankfurt): examine how standardized pound-shilling-penny mint weight across Frankish counties. Explain the fiscal and political consequences."),
    ("heldout_history_05", "history", "Historical institutional summary of Magna Carta baronial clauses (1215 AD, Runnymede): examine how statutory restraint upon arbitrary royal scutage and wardship tallage. Explain the fiscal and political consequences."),
    ("heldout_history_08", "history", "Historical institutional summary of Dutch Vereenigde Oostindische share (1602 AD, Amsterdam): examine how permanent joint-stock equity capital tradable on secondary exchange. Explain the fiscal and political consequences."),
    ("heldout_history_10", "history", "Historical institutional summary of Code Napoleon civil property (1804 AD, Paris): examine how unified metric contract enforcement and partible freehold inheritance. Explain the fiscal and political consequences."),
    ("heldout_math_00", "math", "Theorem statement (Stokes exterior derivative theorem in oriented manifolds with boundary): verify why integral of a differential form over a boundary equals its exterior derivative interior integral. Connect this directly to vector calculus with explicit hypotheses."),
    ("heldout_math_01", "math", "Theorem statement (Euler-Lagrange stationarity condition in smooth action functionals): verify why vanishing first variation yields the second-order differential equations along extrema. Connect this directly to variational paths with explicit hypotheses."),
    ("heldout_math_04", "math", "Theorem statement (Riesz representation duality theorem in Hilbert spaces): verify why every continuous linear functional corresponds isometrically to inner product with a unique vector. Connect this directly to adjoint operators with explicit hypotheses."),
    ("heldout_math_08", "math", "Theorem statement (Courant-Fischer minimax characterization in Hermitian linear operators): verify why ordered real eigenvalues arise as minimax Rayleigh quotients over nested subspaces. Connect this directly to spectral gaps with explicit hypotheses."),
    ("heldout_math_13", "math", "Theorem statement (Lagrange multiplier KKT stationarity in constrained smooth optimization): verify why linear independence qualification ensures dual multipliers certify constrained local optima. Connect this directly to primal-dual gap with explicit hypotheses."),
    ("heldout_systems_00", "systems", "Systems design review for `dijkstra_minheap` (non-negative single-source shortest paths): the critical path relies on binary min-heap decrease-key relaxation. Analyze memory-ordering hazards and why this preserves O(E log V) frontier."),
    ("heldout_systems_01", "systems", "Systems design review for `robin_hood_hash` (open-addressed flat hash table): the critical path relies on probe distance variance balancing. Analyze memory-ordering hazards and why this preserves high load-factor lookup."),
    ("heldout_systems_02", "systems", "Systems design review for `raft_append_log` (consensus replicated state machine): the critical path relies on term matching and prevLogIndex check. Analyze memory-ordering hazards and why this preserves quorum commit advancement."),
    ("heldout_systems_03", "systems", "Systems design review for `mvcc_tuple_vis` (snapshot isolation tuple scanner): the critical path relies on xmin and xmax transaction visibility. Analyze memory-ordering hazards and why this preserves lock-free reader scans."),
    ("heldout_systems_07", "systems", "Systems design review for `metal_nocopy_buf` (Apple Silicon unified memory buffer): the critical path relies on zero-copy page-aligned MTLBuffer. Analyze memory-ordering hazards and why this preserves shared CPU-GPU visibility."),
    ("heldout_clinical_00", "clinical", "Clinical pharmacology summary for Vancomycin (D-Ala-D-Ala peptidoglycan cell-wall cross-link inhibitor): disposition proceeds via predominantly passive glomerular filtration matching creatinine clearance. Outline therapeutic drug monitoring around AUC24/MIC 400-600 and major interactions."),
    ("heldout_clinical_01", "clinical", "Clinical pharmacology summary for Tacrolimus (FKBP12-calcineurin NFAT transcription inhibitor): disposition proceeds via intestinal P-gp efflux and CYP3A4/5 hepatic clearance. Outline therapeutic drug monitoring around whole-blood 12h trough ng/mL and major interactions."),
    ("heldout_clinical_06", "clinical", "Clinical pharmacology summary for Meropenem (penicillin-binding protein PBP2/PBP3 carbapenem beta-lactam): disposition proceeds via renal dehydropeptidase-I stability and tubular secretion. Outline therapeutic drug monitoring around percent fT above MIC and major interactions."),
    ("heldout_clinical_10", "clinical", "Clinical pharmacology summary for Voriconazole (fungal lanosterol 14-alpha-demethylase CYP51 triazole): disposition proceeds via saturable non-linear Michaelis-Menten CYP2C19 clearance. Outline therapeutic drug monitoring around 2-5.5 mg/L plasma trough and major interactions."),
    ("heldout_clinical_12", "clinical", "Clinical pharmacology summary for Isoniazid (mycobacterial InhA mycolic acid synthase hydrazide): disposition proceeds via polymorphic NAT2 hepatic N-acetylation slow vs fast phenotypes. Outline therapeutic drug monitoring around transaminase and B6 neuropathy and major interactions."),
]

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


def run_persistent_bench(
    model_dir: Path,
    prompts_file: Path,
    expected_n: int,
    extra_args: list[str],
    extra_env: dict[str, str] | None = None,
    max_tokens: int = 64,
    max_seq: int = 512,
) -> dict:
    cmd = [
        str(BIN_PATH),
        "--model-dir", str(model_dir),
        "--backend", "metal",
        "--max-seq", str(max_seq),
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
            f"bin/sounio-llm --bench-prompts exited {proc.returncode}: {' '.join(cmd)}\n"
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
                raise RuntimeError(f"bench_tokens mismatch at idx={idx}: {n_tok} vs {len(ids)}")
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
    if set(toks_by_idx.keys()) != expected_keys or set(stats_by_idx.keys()) != expected_keys:
        raise RuntimeError(f"Incomplete bench output: expected={expected_n}")

    rows = []
    for idx in range(expected_n):
        row = dict(stats_by_idx[idx])
        row["prompt_id"] = HELDOUT_20_PROMPTS[idx][0]
        row["domain"] = HELDOUT_20_PROMPTS[idx][1]
        row["token_ids"] = toks_by_idx[idx]
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


def summarize_mode(runs: list[dict], bf_runs: list[dict]) -> dict:
    all_rows = [row for r in runs for row in r["rows"]]
    agg_dec_list = [r["agg_decode_tok_per_sec"] for r in runs]
    agg_e2e_list = [r["agg_e2e_tok_per_sec"] for r in runs]
    wall_list = [r["wall_s"] for r in runs]

    exact_matches = 0
    for r, bfr in zip(runs, bf_runs):
        for row, bfrow in zip(r["rows"], bfr["rows"]):
            if row["token_ids"] == bfrow["token_ids"]:
                exact_matches += 1

    pass0_matches = sum(
        1 for row, bfrow in zip(runs[0]["rows"], bf_runs[0]["rows"]) if row["token_ids"] == bfrow["token_ids"]
    )
    tot_gen = sum(r["gen_tokens"] for r in all_rows)
    tot_cert = sum(r["certified"] for r in all_rows)
    tot_esc = sum(r["escalated"] for r in all_rows)

    return {
        "passes": len(runs),
        "n_prompts": len(HELDOUT_20_PROMPTS),
        "pass0_exact_matches": pass0_matches,
        "total_exact_matches": exact_matches,
        "total_prompt_runs": len(all_rows),
        "mean_agg_decode_tok_s": round(statistics.mean(agg_dec_list), 2),
        "std_agg_decode_tok_s": round(statistics.pstdev(agg_dec_list), 2),
        "mean_prompt_decode_tok_s": round(statistics.mean(r["decode_tok_per_sec"] for r in all_rows), 2),
        "median_prompt_decode_tok_s": round(statistics.median(r["decode_tok_per_sec"] for r in all_rows), 2),
        "mean_agg_e2e_tok_s": round(statistics.mean(agg_e2e_list), 2),
        "mean_wall_s": round(statistics.mean(wall_list), 3),
        "certified_pct": round(100.0 * tot_cert / max(1, tot_gen), 2),
        "escalated_pct": round(100.0 * tot_esc / max(1, tot_gen), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-r1", type=float, default=None)
    args = parser.parse_args()

    q_r1 = args.q_r1
    if q_r1 is None and CAL_JSON.exists():
        cal = json.loads(CAL_JSON.read_text(encoding="utf-8"))
        q_r1 = float(cal["q_hat_005"])
    if q_r1 is None:
        q_r1 = 2.35

    lines = [ptxt.replace("\n", " ").strip() for _, _, ptxt in HELDOUT_20_PROMPTS]
    PROMPTS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    WARMUP_FILE.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    print("[bench] Warmup Qwen3-8B Metal (2 prompts, bf16 & q4_exact_scrub)...", flush=True)
    run_persistent_bench(MODEL_8B, WARMUP_FILE, 2, ["--quant", "bf16"], max_tokens=16)
    run_persistent_bench(MODEL_8B, WARMUP_FILE, 2, ["--quant", "q4_exact_scrub"], max_tokens=16)

    modes_8b = [
        ("bf16", ["--quant", "bf16"], {}),
        ("q4_cone", ["--quant", "q4_cone"], {}),
        ("r1_q4_conformal_probe", ["--quant", "q4_conformal", "--conformal-mode", "flip_exact", "--conformal-q", f"{q_r1}"], {}),
        ("r3_q4_scrub_005", ["--quant", "q4_scrub", "--conformal-alpha", "0.05"], {}),
        ("r3_q4_scrub_001", ["--quant", "q4_scrub", "--conformal-alpha", "0.01"], {}),
        ("r4_q4_exact_scrub", ["--quant", "q4_exact_scrub"], {}),
        ("ctrl_q4_fixed_k8_spec", ["--quant", "q4_exact_scrub", "--conformal-q", "0.0"], {"SOUNIO_SCRUB_MAX_STREAK": "8"}),
    ]

    runs_by_mode: dict[str, list[dict]] = {name: [] for name, _, _ in modes_8b}
    for p_idx in range(3):
        order = modes_8b if p_idx % 2 == 0 else list(reversed(modes_8b))
        for name, m_args, m_env in order:
            print(f"[bench] 8B pass {p_idx + 1}/3 mode={name}...", flush=True)
            runs_by_mode[name].append(run_persistent_bench(MODEL_8B, PROMPTS_FILE, 20, m_args, m_env, max_tokens=32))

    bf_runs = runs_by_mode["bf16"]
    bf_base_dec = statistics.mean(r["agg_decode_tok_per_sec"] for r in bf_runs)
    summary_8b = {}
    for name, _, _ in modes_8b:
        s = summarize_mode(runs_by_mode[name], bf_runs)
        s["speedup_vs_bf16_decode"] = round(s["mean_agg_decode_tok_s"] / max(1e-9, bf_base_dec), 3)
        summary_8b[name] = s
        print(
            f"  {name:<24} exact0={s['pass0_exact_matches']}/20 "
            f"dec={s['mean_agg_decode_tok_s']:>6.2f} tok/s ({s['speedup_vs_bf16_decode']:.3f}x BF16) "
            f"cert={s['certified_pct']:>5.1f}%",
            flush=True,
        )

    assert summary_8b["r4_q4_exact_scrub"]["total_exact_matches"] == 60, "r4_q4_exact_scrub failed 20/20 x3!"
    assert summary_8b["ctrl_q4_fixed_k8_spec"]["total_exact_matches"] == 60, "ctrl_q4_fixed_k8_spec failed 20/20 x3!"

    print("[bench] Trace audit (SOUNIO_LLM_TRACE_STEPS=1) on R1 probe and R3 (0.05/0.01)...", flush=True)
    trace_modes = [
        ("r1_q4_conformal_probe", ["--quant", "q4_conformal", "--conformal-mode", "flip_exact", "--conformal-q", f"{q_r1}"]),
        ("r3_q4_scrub_005", ["--quant", "q4_scrub", "--conformal-alpha", "0.05"]),
        ("r3_q4_scrub_001", ["--quant", "q4_scrub", "--conformal-alpha", "0.01"]),
    ]
    trace_audit = {}
    for name, m_args in trace_modes:
        tr = run_persistent_bench(
            MODEL_8B, PROMPTS_FILE, 20, m_args, extra_env={"SOUNIO_LLM_TRACE_STEPS": "1"}, max_tokens=32
        )
        mism = 0
        cert_fd = 0
        fb_fd = 0
        for i in range(20):
            bf_ids = bf_runs[0]["rows"][i]["token_ids"]
            q_ids = tr["rows"][i]["token_ids"]
            if bf_ids == q_ids:
                continue
            mism += 1
            for k in range(min(len(bf_ids), len(q_ids))):
                if bf_ids[k] != q_ids[k]:
                    if tr["traces_by_idx"][i][k]["cert"] == 1:
                        cert_fd += 1
                    else:
                        fb_fd += 1
                    break
        trace_audit[name] = {
            "exact_matches": 20 - mism,
            "mismatches": mism,
            "certified_first_divergences": cert_fd,
            "fallback_first_divergences": fb_fd,
        }
        print(f"  trace {name:<24} mism={mism}/20 cert_div={cert_fd} fb_div={fb_fd}", flush=True)

    assert trace_audit["r3_q4_scrub_005"]["fallback_first_divergences"] == 0
    assert trace_audit["r3_q4_scrub_001"]["fallback_first_divergences"] == 0

    print("[bench] 1-pass Qwen3-0.6B pruned-head ablation (max_tokens=64)...", flush=True)
    ablation_06b = {}
    bf06 = run_persistent_bench(MODEL_06B, PROMPTS_FILE, 20, ["--quant", "bf16"], max_tokens=64)
    for m_name, m_args in [
        ("bf16", ["--quant", "bf16"]),
        ("r4_q8_exact_scrub", ["--quant", "q8_exact_scrub"]),
        ("r4_q4_exact_scrub", ["--quant", "q4_exact_scrub"]),
    ]:
        r06 = bf06 if m_name == "bf16" else run_persistent_bench(MODEL_06B, PROMPTS_FILE, 20, m_args, max_tokens=64)
        ablation_06b[m_name] = summarize_mode([r06], [bf06])
        ablation_06b[m_name]["speedup_vs_bf16_decode"] = round(
            ablation_06b[m_name]["mean_agg_decode_tok_s"] / max(1e-9, bf06["agg_decode_tok_per_sec"]), 3
        )
        print(
            f"  0.6B {m_name:<20} exact={ablation_06b[m_name]['pass0_exact_matches']}/20 "
            f"dec={ablation_06b[m_name]['mean_agg_decode_tok_s']} tok/s "
            f"({ablation_06b[m_name]['speedup_vs_bf16_decode']:.3f}x BF16)",
            flush=True,
        )

    doc = {
        "model_8b": str(MODEL_8B),
        "max_tokens_8b": 32,
        "max_tokens_06b": 64,
        "q_r1_probe": q_r1,
        "summary_8b": summary_8b,
        "trace_audit_8b": trace_audit,
        "ablation_06b": ablation_06b,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"[bench] PASS -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
