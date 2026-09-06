#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
: "${MASTER_ADDR:?master pod address required}"
: "${MASTER_PORT:?rendezvous port required}"
: "${SLURM_PROCID:?Slurm rank required}"
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=eth0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
MODEL=/scratch/pireus/models/Inkling-Small-NVFP4/b6a99534467840620d411e4cd4ad5819b2610d9c
python3 -c 'import json,hashlib;from pathlib import Path;p=Path("/scratch/pireus/receipts/inkling-model.json");r=json.loads(p.read_text());assert r["revision"]=="b6a99534467840620d411e4cd4ad5819b2610d9c";assert r["manifest_sha256"]==hashlib.sha256(Path("/scratch/pireus/runtime/inkling-files.json").read_bytes()).hexdigest()'
exec /scratch/pireus/runtime/run_in_container.sh python3 -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code --tp 2 --nnodes 2 \
  --node-rank "${PIREUS_RANK:?explicit node rank required}" --dist-init-addr "$MASTER_ADDR:$MASTER_PORT" \
  --quantization modelopt_fp4 --attention-backend triton --page-size 128 \
  --fp4-gemm-backend marlin --moe-runner-backend marlin \
  --mamba-radix-cache-strategy extra_buffer --mem-fraction-static 0.85 \
  --swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1 \
  --disable-prefill-cuda-graph --reasoning-parser inkling --tool-call-parser inkling \
  --context-length 16384 --max-running-requests 1 --host 0.0.0.0 --port 30000
