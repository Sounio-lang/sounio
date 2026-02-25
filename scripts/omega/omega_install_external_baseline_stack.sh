#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VENV_PATH="${OMEGA_BASELINE_VENV_PATH:-artifacts/omega/.venv_external_baselines}"
REPORT_PATH="${OMEGA_BASELINE_STACK_REPORT:-artifacts/omega/external_baseline_stack_install.v1.json}"
PYTHON_BIN="${OMEGA_BASELINE_HOST_PYTHON:-python3}"
INSTALL_TORCH="${OMEGA_INSTALL_TORCH:-1}"
INSTALL_TRITON="${OMEGA_INSTALL_TRITON:-1}"

mkdir -p "$(dirname "$VENV_PATH")" "$(dirname "$REPORT_PATH")"

"$PYTHON_BIN" -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null
"$VENV_PATH/bin/python" -m pip install --upgrade "numpy>=1.26" >/dev/null

torch_status="skipped"
triton_status="skipped"
torch_reason=""
triton_reason=""

if [ "$INSTALL_TORCH" = "1" ]; then
  if "$VENV_PATH/bin/python" -m pip install "torch>=2.5.0" --index-url https://download.pytorch.org/whl/cpu >/tmp/omega_install_torch.log 2>&1; then
    torch_status="installed"
  else
    torch_status="failed"
    torch_reason="$(tail -n 20 /tmp/omega_install_torch.log | tr '\n' ' ' | sed 's/"/'\''/g')"
  fi
fi

if [ "$INSTALL_TRITON" = "1" ]; then
  if "$VENV_PATH/bin/python" -m pip install "triton>=3.0.0" >/tmp/omega_install_triton.log 2>&1; then
    triton_status="installed"
  else
    triton_status="failed"
    triton_reason="$(tail -n 20 /tmp/omega_install_triton.log | tr '\n' ' ' | sed 's/"/'\''/g')"
  fi
fi

cat >"$REPORT_PATH" <<JSON
{
  "schema": "sounio.omega.external-baseline-stack-install.v1",
  "venv_path": "$VENV_PATH",
  "python": "$VENV_PATH/bin/python",
  "torch": {
    "status": "$torch_status",
    "reason": "$torch_reason"
  },
  "triton": {
    "status": "$triton_status",
    "reason": "$triton_reason"
  }
}
JSON

echo "omega_install_external_baseline_stack: venv=$VENV_PATH torch=$torch_status triton=$triton_status report=$REPORT_PATH"
