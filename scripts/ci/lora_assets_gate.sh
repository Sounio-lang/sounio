#!/usr/bin/env bash
# Validate LoRA/dataset assets without downloading models or requiring a GPU.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 "${ROOT_DIR}/docs/training/finetune/validate_lora_assets.py"
