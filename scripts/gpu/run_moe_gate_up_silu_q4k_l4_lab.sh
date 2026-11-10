#!/usr/bin/env bash
# Build and run the MoE gate+up SiLU Q4_K lab on the L4 Slurm node.
set -euo pipefail

if [[ "${MOE_GUS_L4_INNER:-0}" != "1" ]]; then
  ROOT="${SOUNIO_ROOT:-/workspace/sounio}"
  SRC="$ROOT/stdlib/hetero/kernels/moe_gate_up_silu_q4k_f32.cu"
  SCRIPT="$ROOT/scripts/gpu/run_moe_gate_up_silu_q4k_l4_lab.sh"
  OUT_DIR="$ROOT/stdlib/hetero/kernels"
  NODE="${SOUNIO_REMOTE_NODE:-gpuorangefs-r770-proxmox}"
  PARTITION="${SOUNIO_REMOTE_PARTITION:-all}"
  CPUS="${SOUNIO_REMOTE_CPUS:-8}"
  GRES="${SOUNIO_REMOTE_GRES:-gpu:1}"
  MEM="${SOUNIO_REMOTE_MEM:-8G}"
  TIMELIMIT="${SOUNIO_REMOTE_TIME:-00:40:00}"
  CA=/tmp/ca-certificates.crt
  MF=/tmp/Miniforge3-Linux-x86_64.sh
  LOG=/tmp/moe_gus_q4k_l4_srun.log

  if [[ ! -f "$SRC" ]]; then
    echo "missing $SRC" >&2
    exit 1
  fi
  if [[ -f /etc/ssl/certs/ca-certificates.crt ]]; then
    cp -f /etc/ssl/certs/ca-certificates.crt "$CA"
  fi
  if [[ ! -f "$MF" ]]; then
    echo "downloading Miniforge payload"
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$MF"
  fi

  REMOTE_SCRIPT=$(cat <<'INNER'
set -euo pipefail
W=/tmp/moe-gus-q4k-payload-$$
mkdir -p "$W"
cd "$W"
tar xzf -
MOE_GUS_L4_INNER=1 MOE_GUS_PAYLOAD="$W" bash "$W/run_moe_gate_up_silu_q4k_l4_lab.sh"
rc=$?
rm -rf "$W"
exit "$rc"
INNER
)

  set +e
  tar czf - -C "$(dirname "$SRC")" "$(basename "$SRC")" \
      -C "$(dirname "$SCRIPT")" "$(basename "$SCRIPT")" \
      -C /tmp "$(basename "$CA")" "$(basename "$MF")" \
    | srun --partition="$PARTITION" --nodelist="$NODE" --ntasks=1 \
        --cpus-per-task="$CPUS" --gres="$GRES" --mem="$MEM" --time="$TIMELIMIT" \
        --job-name="moe-gus-q4k-$$" /usr/bin/bash -c "$REMOTE_SCRIPT" 2>&1 \
    | tee "$LOG"
  SRUN_RC=${PIPESTATUS[1]}
  set -e

  awk '/^===BEGIN_LAB_OUT===$/ { copy=1; next } /^===END_LAB_OUT===$/ { copy=0 } copy' "$LOG" \
    > "$OUT_DIR/moe_gate_up_silu_q4k_f32_lab_out.txt"
  awk '/^===BEGIN_PTX_B64===$/ { copy=1; next } /^===END_PTX_B64===$/ { copy=0 } copy' "$LOG" \
    | grep -E '^[A-Za-z0-9+/=]+$' | base64 -d > "$OUT_DIR/moe_gate_up_silu_q4k_f32.ptx"

  if [[ ! -s "$OUT_DIR/moe_gate_up_silu_q4k_f32.ptx" ]]; then
    echo "missing PTX receipt in L4 output" >&2
    exit 1
  fi
  if ! grep -q '^LAB_EXIT:' "$OUT_DIR/moe_gate_up_silu_q4k_f32_lab_out.txt"; then
    echo "missing lab receipt in L4 output" >&2
    exit 1
  fi
  echo "wrote $OUT_DIR/moe_gate_up_silu_q4k_f32.ptx"
  echo "wrote $OUT_DIR/moe_gate_up_silu_q4k_f32_lab_out.txt"
  exit "$SRUN_RC"
fi

PAYLOAD="${MOE_GUS_PAYLOAD:-.}"
SRC="$PAYLOAD/moe_gate_up_silu_q4k_f32.cu"
WORK="${MOE_GUS_WORK:-/tmp/moe_gus_q4k_f32_lab}"
HOME_DIR="${MOE_GUS_HOME:-/tmp/moe_gus_home}"
CUDAENV="${CUDAENV_PREFIX:-/tmp/cudaenv126}"
export HOME="$HOME_DIR"
export XDG_CACHE_HOME="$HOME_DIR/.cache"
export CONDA_PKGS_DIRS=/tmp/conda_pkgs
export CONDA_ENVS_PATH=/tmp/conda_envs
mkdir -p "$HOME" "$XDG_CACHE_HOME" /tmp/conda_pkgs /tmp/conda_envs "$WORK"

if [[ ! -f "$SRC" ]]; then
  echo "missing source $SRC" >&2
  exit 1
fi

cp -f "$PAYLOAD/ca-certificates.crt" /tmp/ca-certificates.crt
export SSL_CERT_FILE=/tmp/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/tmp/ca-certificates.crt
export CURL_CA_BUNDLE=/tmp/ca-certificates.crt

if [[ ! -x "$CUDAENV/bin/nvcc" ]]; then
  echo "bootstrapping conda $CUDAENV"
  if [[ ! -x /tmp/mf126/bin/conda ]]; then
    bash "$PAYLOAD/Miniforge3-Linux-x86_64.sh" -b -p /tmp/mf126
  fi
  /tmp/mf126/bin/conda config --set ssl_verify /tmp/ca-certificates.crt
  /tmp/mf126/bin/conda create -y -p "$CUDAENV" -c conda-forge \
    gxx_linux-64 cuda-nvcc=12.6 cuda-cudart-dev=12.6 cuda-cccl=12.6 cuda-version=12.6
fi

export PATH="$CUDAENV/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDAENV/lib:${LD_LIBRARY_PATH:-}"
CXX="$(ls "$CUDAENV"/bin/*g++ 2>/dev/null | head -1 || true)"
CC="$(ls "$CUDAENV"/bin/*gcc 2>/dev/null | grep -v g++ | head -1 || true)"
if [[ -z "$CXX" ]]; then
  echo "missing conda g++" >&2
  exit 1
fi
if [[ -z "$CC" ]]; then
  CC="$CXX"
fi
ln -sfn "$CC" "$CUDAENV/bin/gcc"
ln -sfn "$CXX" "$CUDAENV/bin/g++"
ln -sfn "$CXX" "$CUDAENV/bin/c++"

echo "HOST=$(hostname)"
echo "CCBIN=$CXX"
nvidia-smi -L || true
"$CUDAENV/bin/nvcc" --version | tail -2

cp -f "$SRC" "$WORK/moe_gate_up_silu_q4k_f32.cu"
"$CUDAENV/bin/nvcc" -ccbin "$CXX" -O3 -ptx -arch=sm_89 \
  -o "$WORK/moe_gate_up_silu_q4k_f32.ptx" "$WORK/moe_gate_up_silu_q4k_f32.cu"
"$CUDAENV/bin/nvcc" -ccbin "$CXX" -O3 -arch=sm_89 -DMOE_GUS_LAB_HOST \
  -o "$WORK/moe_gate_up_silu_q4k_f32_lab" "$WORK/moe_gate_up_silu_q4k_f32.cu"

echo "=== LAB RUN ==="
set +e
"$WORK/moe_gate_up_silu_q4k_f32_lab" | tee "$WORK/lab_out.txt"
LAB_RC=${PIPESTATUS[0]}
set -e
echo "=== LAB DONE rc=$LAB_RC ==="

echo "===BEGIN_PTX_B64==="
base64 -w0 "$WORK/moe_gate_up_silu_q4k_f32.ptx"
echo
echo "===END_PTX_B64==="
echo "===BEGIN_LAB_OUT==="
cat "$WORK/lab_out.txt"
echo "===END_LAB_OUT==="
exit "$LAB_RC"
