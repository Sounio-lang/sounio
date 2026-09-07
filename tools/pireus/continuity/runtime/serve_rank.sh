#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${MASTER_ADDR:?master pod address required}"
: "${MASTER_PORT:?rendezvous port required}"
: "${SLURM_PROCID:?Slurm rank required}"
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export TRITON_CACHE_DIR=/scratch/pireus/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/pireus/cache/inductor
export HF_HOME=/scratch/pireus/cache/huggingface
export XDG_CACHE_HOME=/scratch/pireus/cache
export TIKTOKEN_CACHE_DIR=/scratch/pireus/cache/tiktoken
echo "PIREUS_SERVING_JOB=$SLURM_JOB_ID rank=$PIREUS_RANK host=$(hostname)"
MODEL=/scratch/pireus/models/Inkling-Small-NVFP4/b6a99534467840620d411e4cd4ad5819b2610d9c
python3 -c 'import json,hashlib;from pathlib import Path;p=Path("/scratch/pireus/receipts/inkling-model.json");r=json.loads(p.read_text());assert r["revision"]=="b6a99534467840620d411e4cd4ad5819b2610d9c";assert r["manifest_sha256"]==hashlib.sha256(Path("/scratch/pireus/runtime/inkling-files.json").read_bytes()).hexdigest()'
PIREUS_MARLIN_OVERLAY=0 /scratch/pireus/runtime/run_in_container.sh python3 /scratch/pireus/runtime/install_marlin_overlay.py
export PIREUS_MARLIN_OVERLAY=1
/scratch/pireus/runtime/run_in_container.sh python3 /scratch/pireus/runtime/inspect_runtime.py
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072
if [[ "${PIREUS_COLD_CHECKPOINT:-0}" == "1" ]]; then
  python3 /scratch/pireus/runtime/evict_checkpoint_cache.py
fi
entrypoint=(-m sglang.launch_server)
guard_args=()
profile_args=()
if [[ "${PIREUS_META_PROBE:-0}" == "1" ]]; then
  entrypoint=(/scratch/pireus/runtime/profile_model_memory.py)
  if [[ "${PIREUS_META_SKIP_TOKENIZER:-0}" == "1" ]]; then
    profile_args=(--skip-tokenizer-init)
  fi
fi
if [[ "${PIREUS_CUDA_PROBE:-0}" == "1" ]]; then
  entrypoint=(/scratch/pireus/runtime/profile_cuda_memory.py)
fi
if [[ "${PIREUS_OFFLINE_MODE:-}" == "generate" ]]; then
  entrypoint=(/scratch/pireus/runtime/offline_generate.py)
fi
if [[ "${PIREUS_TOKEN_IDS:-0}" == "1" ]]; then
  profile_args=(--skip-tokenizer-init --disable-cuda-graph --chunked-prefill-size 128
                --max-mamba-cache-size 8 --disable-overlap-schedule --disable-custom-all-reduce
                --model-loader-extra-config '{"enable_multithread_load":false}'
                --weight-loader-drop-cache-after-load)
  guard_args=(--reserve-gib 33)
fi
exec python3 /scratch/pireus/runtime/memory_guard.py "${guard_args[@]}" -- /scratch/pireus/runtime/run_in_container.sh python3 "${entrypoint[@]}" \
  --model-path "$MODEL" --trust-remote-code --tp 2 --nnodes 2 \
  --node-rank "${PIREUS_RANK:?explicit node rank required}" --dist-init-addr "$MASTER_ADDR:$MASTER_PORT" \
  --quantization modelopt_fp4 --attention-backend triton --page-size 128 \
  --fp4-gemm-backend marlin --moe-runner-backend marlin \
  --mamba-radix-cache-strategy extra_buffer --mem-fraction-static 0.85 \
  --swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1 \
  --disable-prefill-cuda-graph --reasoning-parser inkling --tool-call-parser inkling \
  --context-length 16384 --max-total-tokens 16384 --max-running-requests 1 --host 0.0.0.0 --port 30000 "${profile_args[@]}"
