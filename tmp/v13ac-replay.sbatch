#!/bin/bash
#SBATCH --job-name=v13ac-replay
#SBATCH --partition=gpu-orangefs
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

BASE_ROOT="/orangefs/training/sounio-ai/repair-v13-octo-ossm-hybrid-20260524T180926-1364094"
RUN_ROOT="$BASE_ROOT/run-v13ac-anchor-balanced-from-v13y-20260525T065500"
REPLAY_ROOT="$RUN_ROOT/replay-20260525Tnext"
REPO="$RUN_ROOT/repo"
ADAPTER="$RUN_ROOT/adapter"
KEY="v13ac_replay"

export PYTHONPATH="/orangefs/training/sounio/python-wheels/sounio-lora-repair-v02-py312:${PYTHONPATH:-}"
export HF_HOME="/orangefs/training/sounio/hf-cache"
export TRANSFORMERS_CACHE="/orangefs/training/sounio/hf-cache/hub"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$REPLAY_ROOT/slices"
cd "$REPO"

python3 scripts/dev/build_sounio_ai_science_eval_slices.py \
  --prompts "$BASE_ROOT/repo/datasets/sounio-ai-eval/prompts.v2.jsonl" \
  --output-dir "$REPLAY_ROOT/slices" \
  --prefix prompts.v2

run_slice() {
  local name="$1"
  local prompts="$REPLAY_ROOT/slices/prompts.v2.$name.jsonl"
  if [ ! -s "$prompts" ]; then
    echo "skip empty slice $name"
    return 0
  fi
  python3 scripts/dev/batch_generate_sounio.py \
    --prompts "$prompts" \
    --output-dir "$REPLAY_ROOT/completions/$name" \
    --model "$KEY=$ADAPTER" \
    --base-model Qwen/Qwen2.5-Coder-1.5B \
    --samples-per-prompt 5 \
    --batch-size 8 \
    --max-new-tokens 512 \
    --temperature 0.7 \
    --top-p 0.95 \
    --deterministic-first-sample \
    --dtype float16 \
    --device-map auto \
    --prompt-style source_only
  python3 scripts/dev/sounio_ai_eval.py \
    --prompts "$prompts" \
    --output-dir "$REPLAY_ROOT/eval/$name" \
    --completions-dir "$REPLAY_ROOT/completions/$name" \
    --model "$KEY=$ADAPTER" \
    --samples-per-prompt 5 \
    --run \
    --souc bin/souc
}

run_slice algebra_octonion
run_slice scientific
run_slice ossm

python3 scripts/dev/batch_generate_sounio.py \
  --prompts "$BASE_ROOT/repo/datasets/sounio-ai-eval/prompts.v2.jsonl" \
  --output-dir "$REPLAY_ROOT/completions/full-v2" \
  --model "$KEY=$ADAPTER" \
  --base-model Qwen/Qwen2.5-Coder-1.5B \
  --samples-per-prompt 5 \
  --batch-size 8 \
  --max-new-tokens 512 \
  --temperature 0.7 \
  --top-p 0.95 \
  --deterministic-first-sample \
  --dtype float16 \
  --device-map auto \
  --prompt-style source_only

python3 scripts/dev/sounio_ai_eval.py \
  --prompts "$BASE_ROOT/repo/datasets/sounio-ai-eval/prompts.v2.jsonl" \
  --output-dir "$REPLAY_ROOT/eval/full-v2" \
  --completions-dir "$REPLAY_ROOT/completions/full-v2" \
  --model "$KEY=$ADAPTER" \
  --samples-per-prompt 5 \
  --run \
  --souc bin/souc

python3 - <<'PY'
from pathlib import Path
import json
import statistics

run = Path("/orangefs/training/sounio-ai/repair-v13-octo-ossm-hybrid-20260524T180926-1364094/run-v13ac-anchor-balanced-from-v13y-20260525T065500")
replay = run / "replay-20260525Tnext"
key = "v13ac_replay"

def latest_summary(name):
    paths = sorted((replay / "eval" / name / "results").glob("*.summary.json"))
    if not paths:
        raise SystemExit(f"missing summary for {name}")
    data = json.loads(paths[-1].read_text())
    return paths[-1], data[key]

def length_stats(name):
    root = replay / "completions" / name / key
    lengths = [len(p.read_text(errors="ignore")) for p in root.glob("*.sio")]
    return {
        "count": len(lengths),
        "avg": sum(lengths) / len(lengths) if lengths else 0.0,
        "median": statistics.median(lengths) if lengths else 0.0,
        "le_180": sum(x <= 180 for x in lengths),
        "le_140": sum(x <= 140 for x in lengths),
        "ge_260": sum(x >= 260 for x in lengths),
        "ge_500": sum(x >= 500 for x in lengths),
    }

alg_path, alg = latest_summary("algebra_octonion")
sci_path, sci = latest_summary("scientific")
ossm_path, ossm = latest_summary("ossm")
full_path, full = latest_summary("full-v2")

decision = {
    "status": "pass",
    "algebra_summary": str(alg_path.relative_to(replay)),
    "scientific_summary": str(sci_path.relative_to(replay)),
    "ossm_summary": str(ossm_path.relative_to(replay)),
    "full_v2_summary": str(full_path.relative_to(replay)),
    "algebra_compile_at_5": alg["compile_at_5"],
    "algebra_stdlib_check_pass_at_5": alg["categories"].get("stdlib_science", {}).get("check_pass_at_5", 0),
    "algebra_stdlib_run_pass": alg["categories"].get("stdlib_science", {}).get("run_pass", 0),
    "scientific_compile_at_5": sci["compile_at_5"],
    "scientific_runtime_pass_rate": sci["runtime_pass_rate"],
    "scientific_output_match_rate": sci["output_match_rate"],
    "scientific_lengths": length_stats("scientific"),
    "ossm_compile_at_5": ossm["compile_at_5"],
    "full_compile_at_1": full["compile_at_1"],
    "full_compile_at_3": full["compile_at_3"],
    "full_compile_at_5": full["compile_at_5"],
    "full_runtime_pass_rate": full["runtime_pass_rate"],
    "full_syntax_purity_rate": full["syntax_purity_rate"],
    "full_output_match_rate": full["output_match_rate"],
    "full_lengths": length_stats("full-v2"),
}
checks = [
    decision["algebra_stdlib_check_pass_at_5"] >= 3,
    decision["algebra_stdlib_run_pass"] >= 3,
    decision["scientific_compile_at_5"] >= 0.80,
    decision["scientific_lengths"]["avg"] >= 220,
    decision["ossm_compile_at_5"] >= 1.0,
    decision["full_compile_at_5"] >= 0.95,
    decision["full_runtime_pass_rate"] >= 0.70,
    decision["full_syntax_purity_rate"] >= 0.80,
]
if not all(checks):
    decision["status"] = "fail"
(replay / "replay-decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print(json.dumps(decision, indent=2, sort_keys=True))
if decision["status"] != "pass":
    raise SystemExit(42)
PY
