#!/usr/bin/env bash
# build-native.sh — Compila o bridge Rust/PyO3 para sounio-py.
#
# Requer: rustup, maturin (pip install maturin)
# Produz: sounio/_sounio_native.so (cpython-3xx-linux-gnu.so)
#
# Após build, `import sounio` usará automaticamente o backend nativo.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[sounio-py] Verificando dependências..."

if ! command -v cargo &>/dev/null; then
    echo "[sounio-py] Rust não encontrado. Instalando via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
fi

if ! command -v maturin &>/dev/null; then
    echo "[sounio-py] Instalando maturin..."
    pip install maturin --quiet
fi

echo "[sounio-py] Compilando extensão Rust (release)..."
maturin develop --release

echo "[sounio-py] Verificando backend..."
python3 -c "
import sys; sys.path.insert(0, 'src')
import sounio
print(f'  backend : {sounio.__backend__}')
print(f'  version : {sounio.__version__}')
from sounio import measure, confidence_gate
k = measure(100.0, 5.0, 0.99, 'mg', 'HPLC_test')
print(f'  K       : {k}')
print(f'  cv%     : {k.cv_percent:.2f}%')
confidence_gate(k, 0.90)
print('  gate    : passed')
"

echo "[sounio-py] Build nativo concluído."
