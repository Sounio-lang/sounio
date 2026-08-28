#!/usr/bin/env bash
set -euo pipefail

# Submit from the repo root or set SOUNIO_DIR explicitly. This wrapper stages a
# compiler-in-the-loop v13 rung on Slurm; it intentionally keeps Mandelbrot-d2
# out of the train set so the octonion/O-SSM effect is isolated.

SOUNIO_DIR="${SOUNIO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUNG_NAME="${RUNG_NAME:-repair_v13_octo_ossm_hybrid}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/orangefs/training/sounio-ai/${RUNG_NAME}-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-1.5B}"
ADAPTER_INIT="${ADAPTER_INIT:-repair_v04_direct}"
LORA_DATASET_MODE="${LORA_DATASET_MODE:-structure_hybrid}"
TARGET_RECORDS="${TARGET_RECORDS:-640}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.7}"
GEN_TOP_P="${GEN_TOP_P:-0.95}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-1024}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-2e-4}"
EVAL_FULL_ONLY="${EVAL_FULL_ONLY:-0}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-orangefs}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-plruntime}"
SBATCH_QOS="${SBATCH_QOS:-gpuorangefs}"
JOB_GPUS="${JOB_GPUS:-1}"
JOB_MEM="${JOB_MEM:-64G}"
JOB_TIME="${JOB_TIME:-12:00:00}"
JOB_CPUS="${JOB_CPUS:-4}"
BOOTSTRAP_DEPS="${BOOTSTRAP_DEPS:-0}"
PYTHONPATH_EXTRA="${PYTHONPATH_EXTRA:-}"
HF_HOME_DIR="${HF_HOME_DIR:-/orangefs/training/sounio/hf-cache}"

mkdir -p "${ARTIFACT_ROOT}"

if [[ "${JOB_GPUS}" != "1" ]]; then
  echo "error: JOB_GPUS=${JOB_GPUS} requested, but this wrapper launches a single-process Trainer without DDP/accelerate." >&2
  echo "Set JOB_GPUS=1, or use a launcher that explicitly consumes multiple GPUs." >&2
  exit 2
fi

SLURM_JOB_NAME="${RUNG_NAME//_/-}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:0:48}"

sbatch <<SLURM
#!/usr/bin/env bash
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH -p ${SBATCH_PARTITION}
#SBATCH -A ${SBATCH_ACCOUNT}
#SBATCH --qos=${SBATCH_QOS}
#SBATCH --gres=gpu:${JOB_GPUS}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=${JOB_CPUS}
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
#SBATCH --output=${ARTIFACT_ROOT}/slurm-%j.out
#SBATCH --error=${ARTIFACT_ROOT}/slurm-%j.err

set -euo pipefail
cd "${SOUNIO_DIR}"

if [[ -n "${PYTHONPATH_EXTRA}" ]]; then
  export PYTHONPATH="${PYTHONPATH_EXTRA}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export HF_HOME="${HF_HOME_DIR}"
export TRANSFORMERS_CACHE="${HF_HOME_DIR}/hub"

if [[ "${BOOTSTRAP_DEPS}" != "0" ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y python3-pip python3.12-venv python3-venv
  VENV_DIR="${ARTIFACT_ROOT}/venv"
  rm -rf "\${VENV_DIR}"
  python3 -m venv "\${VENV_DIR}"
  # shellcheck disable=SC1091
  source "\${VENV_DIR}/bin/activate"
  python3 -m pip install --upgrade pip
  python3 -m pip install torch transformers peft datasets accelerate
fi

python3 scripts/dev/build_sounio_ai_structure_hybrid_dataset.py
python3 scripts/dev/lora_finetune.py \\
  --model "${BASE_MODEL}" \\
  --adapter-init "${ADAPTER_INIT}" \\
  --output "${ARTIFACT_ROOT}/adapter" \\
  --dataset-mode "${LORA_DATASET_MODE}" \\
  --prompt-format source_only \\
  --target-records "${TARGET_RECORDS}" \\
  --batch-size "${TRAIN_BATCH_SIZE}" \\
  --max-length "${TRAIN_MAX_LENGTH}" \\
  --epochs "${TRAIN_EPOCHS}" \\
  --learning-rate "${TRAIN_LEARNING_RATE}" \\
  --heldout-prompts datasets/sounio-ai-eval/prompts.v2.jsonl \\
  --train-manifest "${ARTIFACT_ROOT}/train-manifest.json"

python3 scripts/dev/batch_generate_sounio.py \\
  --prompts datasets/sounio-ai-eval/prompts.v2.jsonl \\
  --output-dir "${ARTIFACT_ROOT}/completions/full-v2" \\
  --model "v13=${ARTIFACT_ROOT}/adapter" \\
  --base-model "${BASE_MODEL}" \\
  --samples-per-prompt "${SAMPLES_PER_PROMPT}" \\
  --temperature "${GEN_TEMPERATURE}" \\
  --top-p "${GEN_TOP_P}" \\
  --deterministic-first-sample \\
  --prompt-style source_only

python3 scripts/dev/sounio_ai_eval.py \\
  --prompts datasets/sounio-ai-eval/prompts.v2.jsonl \\
  --completions-dir "${ARTIFACT_ROOT}/completions/full-v2" \\
  --output-dir "${ARTIFACT_ROOT}/eval/full-v2" \\
  --model "v13=${ARTIFACT_ROOT}/adapter" \\
  --samples-per-prompt "${SAMPLES_PER_PROMPT}" \\
  --run

if [[ "${EVAL_FULL_ONLY}" != "0" ]]; then
  echo "artifact_root=${ARTIFACT_ROOT}"
  echo "eval_full_only=1"
  exit 0
fi

python3 scripts/dev/build_sounio_ai_science_eval_slices.py \\
  --prompts datasets/sounio-ai-eval/prompts.v2.jsonl \\
  --output-dir "${ARTIFACT_ROOT}/slices" \\
  --prefix prompts.v2

for slice in scientific ossm algebra_octonion knowledge_octonion; do
  python3 scripts/dev/batch_generate_sounio.py \\
    --prompts "${ARTIFACT_ROOT}/slices/prompts.v2.\${slice}.jsonl" \\
    --output-dir "${ARTIFACT_ROOT}/completions/\${slice}" \\
    --model "v13=${ARTIFACT_ROOT}/adapter" \\
    --base-model "${BASE_MODEL}" \\
    --samples-per-prompt "${SAMPLES_PER_PROMPT}" \\
    --temperature "${GEN_TEMPERATURE}" \\
    --top-p "${GEN_TOP_P}" \\
    --deterministic-first-sample \\
    --prompt-style source_only
  python3 scripts/dev/sounio_ai_eval.py \\
    --prompts "${ARTIFACT_ROOT}/slices/prompts.v2.\${slice}.jsonl" \\
    --completions-dir "${ARTIFACT_ROOT}/completions/\${slice}" \\
    --output-dir "${ARTIFACT_ROOT}/eval/\${slice}" \\
    --model "v13=${ARTIFACT_ROOT}/adapter" \\
    --samples-per-prompt "${SAMPLES_PER_PROMPT}" \\
    --run
done

python3 scripts/dev/build_sounio_ai_science_spine_prompts.py \\
  --output "${ARTIFACT_ROOT}/slices/prompts.science_spine_reference.jsonl"

python3 scripts/dev/sounio_ai_eval.py \\
  --prompts "${ARTIFACT_ROOT}/slices/prompts.science_spine_reference.jsonl" \\
  --output-dir "${ARTIFACT_ROOT}/eval/science-spine-reference-smoke" \\
  --model smoke=reference \\
  --use-reference-completions \\
  --run

echo "artifact_root=${ARTIFACT_ROOT}"
SLURM
