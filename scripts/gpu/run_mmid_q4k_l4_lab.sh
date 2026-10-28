#!/usr/bin/env bash
# Self-contained L4 lab for mmid_q4k_dp4a_f32. Expects MMID_PAYLOAD with:
#   mmid_q4k_dp4a_f32.cu, this script, ca-certificates.crt,
#   Miniforge3-Linux-x86_64.sh
# Uses CUDA 12.6 nvcc so PTX .version is JIT-compatible with cluster drivers.
set -euo pipefail

PAYLOAD="${MMID_PAYLOAD:-.}"
SRC="$PAYLOAD/mmid_q4k_dp4a_f32.cu"
WORK="${MMID_WORK:-/tmp/mmid_f32_lab}"
HOME_DIR="${MMID_HOME:-/tmp/mmid_home}"
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

if [[ -f "$PAYLOAD/ca-certificates.crt" ]]; then
  cp -f "$PAYLOAD/ca-certificates.crt" /tmp/ca-certificates.crt
elif [[ -f /etc/ssl/certs/ca-certificates.crt ]]; then
  cp -f /etc/ssl/certs/ca-certificates.crt /tmp/ca-certificates.crt
fi
export SSL_CERT_FILE=/tmp/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/tmp/ca-certificates.crt
export CURL_CA_BUNDLE=/tmp/ca-certificates.crt

NVCC=""
if [[ -x "$CUDAENV/bin/nvcc" ]]; then
  NVCC="$CUDAENV/bin/nvcc"
elif command -v nvcc >/dev/null 2>&1; then
  NVCC="$(command -v nvcc)"
fi

if [[ -z "$NVCC" ]]; then
  echo "bootstrapping conda $CUDAENV"
  MF_SH=""
  if [[ -f "$PAYLOAD/Miniforge3-Linux-x86_64.sh" ]]; then
    MF_SH="$PAYLOAD/Miniforge3-Linux-x86_64.sh"
  else
    echo "missing Miniforge installer in payload" >&2
    exit 1
  fi
  if [[ ! -x /tmp/mf126/bin/conda ]]; then
    bash "$MF_SH" -b -p /tmp/mf126
  fi
  /tmp/mf126/bin/conda config --set ssl_verify /tmp/ca-certificates.crt
  /tmp/mf126/bin/conda create -y -p "$CUDAENV" -c conda-forge \
    gxx_linux-64 cuda-nvcc=12.6 cuda-cudart-dev=12.6 cuda-cccl=12.6 cuda-version=12.6
  NVCC="$CUDAENV/bin/nvcc"
fi

export PATH="$CUDAENV/bin:$(dirname "$NVCC"):${PATH:-}"
export LD_LIBRARY_PATH="$CUDAENV/lib:${LD_LIBRARY_PATH:-}"
CXX="$(ls "$CUDAENV"/bin/*g++ 2>/dev/null | head -1 || true)"
CC="$(ls "$CUDAENV"/bin/*gcc 2>/dev/null | grep -v g++ | head -1 || true)"
if [[ -z "${CXX:-}" ]]; then echo "missing conda g++" >&2; exit 1; fi
if [[ -z "${CC:-}" ]]; then CC="$CXX"; fi
export CXX CC
ln -sfn "$CC" "$CUDAENV/bin/gcc"
ln -sfn "$CXX" "$CUDAENV/bin/g++"
ln -sfn "$CXX" "$CUDAENV/bin/c++"

echo "HOST=$(hostname)"
echo "CCBIN=$CXX"
nvidia-smi -L || true
"$NVCC" --version | tail -2

cp -f "$SRC" "$WORK/mmid_q4k_dp4a_f32.cu"
"$NVCC" -ccbin "$CXX" -O3 -ptx -arch=sm_89 -o "$WORK/mmid_q4k_dp4a_f32.ptx" "$WORK/mmid_q4k_dp4a_f32.cu"
"$NVCC" -ccbin "$CXX" -O3 -arch=sm_89 -DMMID_LAB_HOST -o "$WORK/mmid_q4k_dp4a_f32_lab" "$WORK/mmid_q4k_dp4a_f32.cu"

echo "=== LAB RUN ==="
set +e
"$WORK/mmid_q4k_dp4a_f32_lab" | tee "$WORK/lab_out.txt"
LAB_RC=${PIPESTATUS[0]}
set -e
echo "=== LAB DONE rc=$LAB_RC ==="

echo "===BEGIN_PTX_B64==="
base64 -w0 "$WORK/mmid_q4k_dp4a_f32.ptx"
echo
echo "===END_PTX_B64==="
echo "===BEGIN_LAB_OUT==="
cat "$WORK/lab_out.txt"
echo "===END_LAB_OUT==="
exit "$LAB_RC"
