#!/usr/bin/env bash
set -euo pipefail
PAYLOAD="${MMID_PAYLOAD:-.}"
SRC="$PAYLOAD/mmid_q4k_dp4a_f32.cu"
WORK=/tmp/mmid_f32_lab
export HOME=/tmp/mmid_home
export XDG_CACHE_HOME=/tmp/mmid_home/.cache
export CONDA_PKGS_DIRS=/tmp/conda_pkgs
export CONDA_ENVS_PATH=/tmp/conda_envs
mkdir -p "$HOME" "$XDG_CACHE_HOME" /tmp/conda_pkgs /tmp/conda_envs "$WORK"
cp -f "$PAYLOAD/ca-certificates.crt" /tmp/ca-certificates.crt
export SSL_CERT_FILE=/tmp/ca-certificates.crt REQUESTS_CA_BUNDLE=/tmp/ca-certificates.crt CURL_CA_BUNDLE=/tmp/ca-certificates.crt
if [[ ! -x /tmp/cudaenv126/bin/nvcc ]]; then
  bash "$PAYLOAD/Miniforge3-Linux-x86_64.sh" -b -p /tmp/mf126
  /tmp/mf126/bin/conda config --set ssl_verify /tmp/ca-certificates.crt
  /tmp/mf126/bin/conda create -y -p /tmp/cudaenv126 -c conda-forge \
    gxx_linux-64 cuda-nvcc=12.6 cuda-cudart-dev=12.6 cuda-cccl=12.6 cuda-version=12.6
fi
export PATH=/tmp/cudaenv126/bin:$PATH
export LD_LIBRARY_PATH=/tmp/cudaenv126/lib:${LD_LIBRARY_PATH:-}
CXX=$(ls /tmp/cudaenv126/bin/*g++ | head -1)
CC=$(ls /tmp/cudaenv126/bin/*gcc | grep -v g++ | head -1)
ln -sfn "$CC" /tmp/cudaenv126/bin/gcc
ln -sfn "$CXX" /tmp/cudaenv126/bin/g++
nvcc --version | tail -2
nvcc -ccbin "$CXX" -O3 -ptx -arch=sm_89 -o "$WORK/mmid_q4k_dp4a_f32.ptx" "$SRC"
nvcc -ccbin "$CXX" -O3 -arch=sm_89 -DMMID_LAB_HOST -o "$WORK/mmid_q4k_dp4a_f32_lab" "$SRC"
head -15 "$WORK/mmid_q4k_dp4a_f32.ptx"
echo "=== LAB RUN ==="
"$WORK/mmid_q4k_dp4a_f32_lab" | tee "$WORK/lab_out.txt"
echo "===BEGIN_PTX_B64==="
base64 -w0 "$WORK/mmid_q4k_dp4a_f32.ptx"; echo
echo "===END_PTX_B64==="
echo "===BEGIN_LAB_OUT==="
cat "$WORK/lab_out.txt"
echo "===END_LAB_OUT==="
