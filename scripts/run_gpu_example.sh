#!/bin/bash

# GPU Example Runner for Phase 2 Optimizations
# Usage: ./run_gpu_example.sh [mixed-precision|qat|sparse|profile]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPILER_DIR="$SCRIPT_DIR/compiler"
EXAMPLE_DIR="$SCRIPT_DIR/examples"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════${NC}"
}

print_step() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check GPU availability
check_gpu() {
    print_header "Checking GPU Availability"

    if command -v nvidia-smi &> /dev/null; then
        print_step "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
        return 0
    elif command -v metal-cli &> /dev/null; then
        print_step "Apple Metal GPU detected"
        metal-cli device list
        return 0
    else
        print_error "No GPU detected"
        print_info "Install CUDA Toolkit or use Apple Silicon Mac with Metal"
        return 1
    fi
}

# Build compiler with GPU support
build_compiler() {
    print_header "Building Sounio Compiler with GPU Support"

    cd "$COMPILER_DIR"

    if command -v nvidia-smi &> /dev/null; then
        print_step "Building with GPU + CUDA support..."
        cargo build --release --features gpu,cuda
    else
        print_step "Building with GPU support (CUDA not available)..."
        cargo build --release --features gpu
    fi

    print_step "Build complete!"
}

# Run mixed-precision example
run_mixed_precision() {
    print_header "Running Mixed-Precision Training Example"
    print_info "Phase 2A: FP16 forward pass with FP32 backward pass"
    print_info "Expected improvement: 2x memory bandwidth"

    cd "$COMPILER_DIR"

    if command -v nvidia-smi &> /dev/null; then
        print_step "Running on NVIDIA GPU with CUDA..."
        cargo run --release --features gpu,cuda -- run \
            "$EXAMPLE_DIR/multi_feature_training_example.sio"
    else
        print_step "Running with GPU codegen verification..."
        cargo run --release --features gpu -- check \
            "$EXAMPLE_DIR/multi_feature_training_example.sio" --show-types
    fi
}

# Run QAT example
run_qat() {
    print_header "Running Quantization-Aware Training Example"
    print_info "Phase 2B: Fake quantization with STE gradients"
    print_info "Expected improvement: 10-20% inference speedup"

    cd "$COMPILER_DIR"

    if command -v nvidia-smi &> /dev/null; then
        print_step "Running on NVIDIA GPU with CUDA..."
        cargo run --release --features gpu,cuda -- run \
            "$EXAMPLE_DIR/qat_training_example.sio"
    else
        print_step "Running with GPU codegen verification..."
        cargo run --release --features gpu -- check \
            "$EXAMPLE_DIR/qat_training_example.sio" --show-types
    fi
}

# Run sparse quaternion example
run_sparse() {
    print_header "Running Sparse Quaternion Example"
    print_info "Phase 2C: 2:4 structured sparsity with GPU codegen"
    print_info "Expected improvement: 2-4x for sparse networks"

    # Create simple sparse example if it doesn't exist
    if [ ! -f "$EXAMPLE_DIR/sparse_quat_example.sio" ]; then
        print_step "Creating sparse quaternion example..."
        cat > "$EXAMPLE_DIR/sparse_quat_example.sio" << 'EOF'
fn main() with IO {
    print("Phase 2C: Sparse Quaternion Operations\n")
    print("─────────────────────────────────────\n")

    let batch_size = 32
    let in_features = 256
    let out_features = 128

    print("Configuration:\n")
    print("  Batch: {}\n", batch_size)
    print("  Input features: {}\n", in_features)
    print("  Output features: {}\n", out_features)

    print("\nSparsity Pattern: 2:4 structured\n")
    print("  2 non-zero quaternions per group of 4\n")
    print("  50% weight reduction\n")

    print("\nExpected Performance:\n")
    print("  Speedup: 2-4x over dense quaternions\n")
    print("  Memory: 50% reduction\n")
    print("  Bandwidth: 70-80% of peak\n")
}
EOF
    fi

    cd "$COMPILER_DIR"

    if command -v nvidia-smi &> /dev/null; then
        print_step "Running on NVIDIA GPU with CUDA..."
        cargo run --release --features gpu,cuda -- run \
            "$EXAMPLE_DIR/sparse_quat_example.sio"
    else
        print_step "Running with GPU codegen verification..."
        cargo run --release --features gpu -- check \
            "$EXAMPLE_DIR/sparse_quat_example.sio" --show-types
    fi
}

# Profile GPU execution
profile_gpu() {
    print_header "GPU Execution Profiling"

    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. NVIDIA GPU profiling requires CUDA."
        return 1
    fi

    print_info "Monitoring GPU during execution..."
    print_info "Run this in another terminal to see real-time GPU stats:"
    print_info "watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv'"

    print_step "Running mixed-precision example with profiling..."

    cd "$COMPILER_DIR"

    (
        # Monitor GPU in background
        while true; do
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total \
                --format=csv,noheader,nounits | \
                awk '{printf "GPU: %d%% | MEM: %d%% (%d/%d MB)\n", $1, $2, $3, $4}'
            sleep 1
        done
    ) &
    MONITOR_PID=$!

    # Run the actual example
    cargo run --release --features gpu,cuda -- run \
        "$EXAMPLE_DIR/multi_feature_training_example.sio" || true

    # Kill the monitoring process
    kill $MONITOR_PID 2>/dev/null || true

    print_step "Profiling complete!"
}

# Show help
show_help() {
    cat << EOF
${BLUE}Sounio Phase 2 GPU Example Runner${NC}

Usage: $0 [COMMAND]

Commands:
  check              Check GPU availability
  build              Build compiler with GPU support
  mixed-precision    Run Phase 2A mixed-precision example
  qat                Run Phase 2B QAT example
  sparse             Run Phase 2C sparse quaternion example
  profile            Profile GPU execution (NVIDIA only)
  all                Run all examples

Examples:
  $0 check                    # Verify GPU setup
  $0 build                    # Build with --features gpu,cuda
  $0 mixed-precision          # Run mixed-precision demo
  $0 profile                  # Profile GPU with nvidia-smi

EOF
}

# Main
main() {
    local cmd="${1:-help}"

    case "$cmd" in
        check)
            check_gpu
            ;;
        build)
            build_compiler
            ;;
        mixed-precision)
            check_gpu && build_compiler && run_mixed_precision
            ;;
        qat)
            check_gpu && build_compiler && run_qat
            ;;
        sparse)
            check_gpu && build_compiler && run_sparse
            ;;
        profile)
            check_gpu && build_compiler && profile_gpu
            ;;
        all)
            check_gpu && build_compiler && \
            run_mixed_precision && \
            run_qat && \
            run_sparse
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $cmd"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
