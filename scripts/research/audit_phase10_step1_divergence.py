#!/usr/bin/env python3
"""
Step 1 Diagnostic Audit (Zero PyTorch):
Runs the native bin/sounio-llm binary with SOUNIO_LLM_TRACE_STEPS=1 on the 20
benchmark prompts across:
  - bf16 (reference)
  - q8_conformal_flip_005 (q=1.291984)
  - q8_conformal_flip_001 (q=2.343652)
  - q8_conformal_overtake_005 (q=8.754311)
  - q8_conformal_force_fallback (SOUNIO_CONFORMAL_Q=10000.0 -> 100% decode fallback)

For each prompt and mode, identifies the EXACT first divergent token index k*
vs pure BF16, and classifies whether k* was a CERTIFIED token (cert=1, esc=0)
or a FALLBACK token (cert=0, esc=1).
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bench_native_q8_conformal_vs_bf16 import (  # noqa: E402
    BIN_PATH,
    HELDOUT_20_PROMPTS,
    METAL_DYLIB,
    MODEL_DIR,
)

OUT_JSON = Path(
    "/Users/demetriosagourakis/sounio-research/checkpoints/"
    "Qwen3-0.6B-ossm-sidecar/phase10_step1_divergence_audit.json"
)

TRACE_RE = re.compile(
    r"\[trace_step\s+idx=(\d+)\s+tok=(-?\d+)\s+q8_top1=(-?\d+)\s+"
    r"cert=(\d+)\s+esc=(\d+)\s+margin=([-0-9.]+)\]"
)

MODES = [
    ("bf16", ["--quant", "bf16"], {}),
    ("q8_conformal_flip_005", ["--quant", "q8_conformal", "--conformal-mode", "flip_exact", "--conformal-alpha", "0.05"], {}),
    ("q8_conformal_flip_001", ["--quant", "q8_conformal", "--conformal-mode", "flip_exact", "--conformal-alpha", "0.01"], {}),
    ("q8_conformal_overtake_005", ["--quant", "q8_conformal", "--conformal-mode", "overtake", "--conformal-alpha", "0.05"], {}),
    ("q8_conformal_force_fallback", ["--quant", "q8_conformal", "--conformal-mode", "flip_exact", "--conformal-alpha", "0.01"], {"SOUNIO_CONFORMAL_Q": "10000.0"}),
]


def run_traced(prompt: str, args: list[str], extra_env: dict[str, str], max_tokens: int = 64) -> dict:
    env = dict(os.environ)
    env["SOUNIO_METAL_RUNTIME"] = str(METAL_DYLIB)
    env["SOUNIO_LLM_TRACE_STEPS"] = "1"
    env.pop("SOUNIO_CONFORMAL_Q", None)
    env.update(extra_env)

    cmd = [
        str(BIN_PATH),
        "--model-dir", str(MODEL_DIR),
        "--backend", "metal",
        "--max-tokens", str(max_tokens),
        "--no-think",
        "--temp", "0",
        "--prompt", prompt,
    ] + args
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"bin/sounio-llm failed ({proc.returncode}): {proc.stdout}\n{proc.stderr}")

    steps = []
    text_lines = []
    for ln in proc.stdout.splitlines():
        m = TRACE_RE.search(ln)
        if m:
            steps.append({
                "idx": int(m.group(1)),
                "tok": int(m.group(2)),
                "q8_top1": int(m.group(3)),
                "cert": int(m.group(4)),
                "esc": int(m.group(5)),
                "margin": float(m.group(6)),
            })
        elif not ln.startswith("[sounio-llm:"):
            text_lines.append(ln)
    if not steps:
        raise RuntimeError(f"No [trace_step ...] lines emitted:\n{proc.stdout}")
    return {"steps": steps, "text": "\n".join(text_lines).strip()}


def main() -> None:
    print("=" * 104)
    print("PHASE 10 STEP 1 DIVERGENCE AUDIT — Certified vs Fallback First-Divergence Attribution")
    print("=" * 104)

    all_runs = {mname: {} for mname, _, _ in MODES}
    for idx, (pid, dom, ptxt) in enumerate(HELDOUT_20_PROMPTS):
        for mname, margs, menv in MODES:
            all_runs[mname][pid] = run_traced(ptxt, margs, menv, max_tokens=64)
        print(f"Traced [{idx+1:02d}/20] {pid}")

    mode_summaries = {}
    for mname, _, _ in MODES:
        if mname == "bf16":
            continue
        mismatched = 0
        div_certified = 0
        div_fallback = 0
        div_at_k0 = 0
        prior_cert_before_fb_div = 0
        details = []

        print(f"\n--- {mname} ---")
        for pid, dom, _ in HELDOUT_20_PROMPTS:
            bf_steps = all_runs["bf16"][pid]["steps"]
            md_steps = all_runs[mname][pid]["steps"]
            n_cmp = min(len(bf_steps), len(md_steps))
            k_star = None
            for k in range(n_cmp):
                if bf_steps[k]["tok"] != md_steps[k]["tok"]:
                    k_star = k
                    break
            if k_star is None and len(bf_steps) != len(md_steps):
                k_star = n_cmp

            if k_star is None:
                details.append({"prompt_id": pid, "domain": dom, "match": True})
            else:
                mismatched += 1
                st = md_steps[k_star] if k_star < len(md_steps) else md_steps[-1]
                bft = bf_steps[k_star]["tok"] if k_star < len(bf_steps) else -1
                if st["cert"] == 1 and st["esc"] == 0:
                    kind = "CERTIFIED"
                    div_certified += 1
                else:
                    kind = "FALLBACK"
                    div_fallback += 1
                if k_star == 0:
                    div_at_k0 += 1
                n_prev_cert = sum(1 for x in md_steps[:k_star] if x["cert"] == 1)
                n_prev_esc = sum(1 for x in md_steps[:k_star] if x["esc"] == 1)
                if kind == "FALLBACK" and n_prev_cert > 0:
                    prior_cert_before_fb_div += 1
                rec = {
                    "prompt_id": pid,
                    "domain": dom,
                    "match": False,
                    "first_div_tok_idx": k_star,
                    "kind": kind,
                    "bf16_tok": bft,
                    "mode_tok": st["tok"],
                    "q8_top1": st["q8_top1"],
                    "cert": st["cert"],
                    "esc": st["esc"],
                    "margin": st["margin"],
                    "prior_certified_decode_steps": n_prev_cert,
                    "prior_fallback_decode_steps": n_prev_esc,
                }
                details.append(rec)
                print(
                    f"  {pid:20s}: k*={k_star:2d} ({kind:9s}) bf16={bft:6d} mode={st['tok']:6d} "
                    f"q8={st['q8_top1']:6d} margin={st['margin']:7.3f} "
                    f"(prior_dec: cert={n_prev_cert:2d}, esc={n_prev_esc:2d})"
                )

        mode_summaries[mname] = {
            "total_prompts": len(HELDOUT_20_PROMPTS),
            "exact_match_prompts": len(HELDOUT_20_PROMPTS) - mismatched,
            "mismatched_prompts": mismatched,
            "first_div_certified_count": div_certified,
            "first_div_fallback_count": div_fallback,
            "first_div_at_k0_prefill_output": div_at_k0,
            "fallback_div_with_prior_certified_steps": prior_cert_before_fb_div,
            "prompts": details,
        }
        print(
            f"SUMMARY {mname}: mismatched={mismatched}/20 | "
            f"first_div=CERTIFIED:{div_certified}, FALLBACK:{div_fallback}, at_k0:{div_at_k0}"
        )

    out = {
        "audit": "phase10_step1_divergence",
        "model_dir": str(MODEL_DIR),
        "calibration_reference_method": (
            "scripts/research/audit_phase7b_step1d_prompt_conformal.py (lines 312-342, 527-542): "
            "teacher-forced forward pass on the SAME clean prefix, with separate homogeneous KV caches "
            "(pure BF16 K/V for zb vs pure Q8 K/V for zq)."
        ),
        "mode_summaries": mode_summaries,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
