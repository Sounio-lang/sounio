#!/usr/bin/env bash
# Sounio vs Julia vs Python — Benchmark Suite
# Usage: bash benchmarks/comparison/run_all.sh [--sounio-only|--python-only|--julia-only]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOUC="$ROOT_DIR/target/release/souc"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

header() { echo -e "\n${BOLD}${BLUE}═══ $1 ═══${NC}"; }
lang_header() { echo -e "  ${GREEN}[$1]${NC}"; }
skip_msg() { echo -e "  ${RED}[SKIP]${NC} $1 not found"; }

# Check tools
has_julia() { command -v julia &>/dev/null; }
has_python() { command -v python3 &>/dev/null; }
has_souc() { [ -f "$SOUC" ]; }

FILTER="${1:-all}"

# Build Sounio if needed
if [[ "$FILTER" == "all" || "$FILTER" == "--sounio-only" ]]; then
    if ! has_souc; then
        echo "Building souc (release)..."
        (cd "$ROOT_DIR" && cargo build -p souc --release --bin souc 2>/dev/null)
    fi
fi

run_sounio() {
    local sio_file="$1"
    if ! has_souc; then skip_msg "souc"; return; fi
    lang_header "Sounio"
    /usr/bin/time -f "  Wall: %e s  RSS: %M KB" "$SOUC" run "$sio_file" 2>&1
}

run_python() {
    local py_file="$1"
    if ! has_python; then skip_msg "python3"; return; fi
    lang_header "Python"
    /usr/bin/time -f "  Wall: %e s  RSS: %M KB" python3 "$py_file" 2>&1
}

run_julia() {
    local jl_file="$1"
    if ! has_julia; then skip_msg "julia"; return; fi
    lang_header "Julia"
    /usr/bin/time -f "  Wall: %e s  RSS: %M KB" julia "$jl_file" 2>&1
}

run_bench() {
    local name="$1"
    local dir="$SCRIPT_DIR/$2"

    header "$name"

    case "$FILTER" in
        all)
            run_sounio "$dir/sounio_bench.sio"
            run_python "$dir/python_bench.py"
            run_julia "$dir/julia_bench.jl"
            ;;
        --sounio-only) run_sounio "$dir/sounio_bench.sio" ;;
        --python-only) run_python "$dir/python_bench.py" ;;
        --julia-only)  run_julia "$dir/julia_bench.jl" ;;
    esac
}

echo -e "${BOLD}Sounio Benchmark Suite${NC}"
echo "Date: $(date -Iseconds)"
echo "Filter: $FILTER"

run_bench "ODE Solver (RK4)" "ode"
run_bench "Linear Algebra (QR Decomposition)" "linalg"
run_bench "Uncertainty Propagation (PK/Sobol)" "uncertainty"

echo ""
header "Done"
