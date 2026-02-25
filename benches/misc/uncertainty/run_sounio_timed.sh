#!/bin/bash
# Timed Sounio uncertainty benchmarks
# Each benchmark is a separate .sio file for accurate per-benchmark timing

set -euo pipefail

SOUC="${SOUC:-/tmp/sounio-main/compiler/target/release/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-/tmp/sounio-main/stdlib}"
DIR="$(cd "$(dirname "$0")" && pwd)"
WARMUP=3
RUNS=10

run_bench() {
    local name="$1"
    local file="$2"
    local times=()

    # Warmup
    for ((w=0; w<WARMUP; w++)); do
        "$SOUC" run "$file" >/dev/null 2>&1
    done

    # Timed runs
    for ((r=0; r<RUNS; r++)); do
        local start end elapsed_ms
        start=$(date +%s%N)
        "$SOUC" run "$file" >/dev/null 2>&1
        end=$(date +%s%N)
        elapsed_ms=$(echo "scale=2; ($end - $start) / 1000000" | bc)
        times+=("$elapsed_ms")
    done

    # Sort and get median
    IFS=$'\n' sorted=($(sort -n <<<"${times[*]}")); unset IFS
    local mid=$((RUNS / 2))
    local median="${sorted[$mid]}"

    # Compute mean
    local sum=0
    for t in "${times[@]}"; do
        sum=$(echo "$sum + $t" | bc)
    done
    local mean=$(echo "scale=2; $sum / $RUNS" | bc)
    local min="${sorted[0]}"
    local max="${sorted[$((RUNS-1))]}"

    printf "%-35s Median: %10s ms  Mean: %10s ms  Min: %10s ms  Max: %10s ms\n" \
        "$name" "$median" "$mean" "$min" "$max"
}

echo "============================================================"
echo "Uncertainty Propagation Benchmarks - Sounio (Knowledge<T>)"
echo "Compiler: $SOUC"
echo "Warmup: $WARMUP, Runs: $RUNS"
echo "============================================================"
echo ""

# Generate individual benchmark files
for bench in gum_chain_10k gum_chain_100k monte_carlo_10k monte_carlo_100k cockcroft_gault_10k transcendental_10k; do
    cat > "$DIR/_bench_${bench}.sio" << 'SIOEOF'
SIOEOF
done

# GUM Chain 10K
cat > "$DIR/_bench_gum_chain_10k.sio" << 'EOF'
fn gum_chain(n_ops: i32) -> f64 with IO {
    let x = Knowledge { value: 10.0, epsilon: 0.01 }
    let y = Knowledge { value: 5.0, epsilon: 0.01 }
    var result = x
    var i = 0
    while i < n_ops {
        let m = i % 4
        if m == 0 { result = result + y }
        else if m == 1 { result = result - y }
        else if m == 2 { result = result * Knowledge { value: 1.01, epsilon: 0.001 } }
        else { result = result / Knowledge { value: 1.01, epsilon: 0.001 } }
        i = i + 1
    }
    return 0.0
}
fn main() with IO { let _ = gum_chain(10000) }
EOF

# GUM Chain 100K
cat > "$DIR/_bench_gum_chain_100k.sio" << 'EOF'
fn gum_chain(n_ops: i32) -> f64 with IO {
    let x = Knowledge { value: 10.0, epsilon: 0.01 }
    let y = Knowledge { value: 5.0, epsilon: 0.01 }
    var result = x
    var i = 0
    while i < n_ops {
        let m = i % 4
        if m == 0 { result = result + y }
        else if m == 1 { result = result - y }
        else if m == 2 { result = result * Knowledge { value: 1.01, epsilon: 0.001 } }
        else { result = result / Knowledge { value: 1.01, epsilon: 0.001 } }
        i = i + 1
    }
    return 0.0
}
fn main() with IO { let _ = gum_chain(100000) }
EOF

# Monte Carlo 10K
cat > "$DIR/_bench_monte_carlo_10k.sio" << 'EOF'
fn monte_carlo(n: i32) -> f64 with IO {
    let ret1 = Knowledge { value: 0.08, epsilon: 0.02 }
    let ret2 = Knowledge { value: 0.12, epsilon: 0.03 }
    let ret3 = Knowledge { value: 0.06, epsilon: 0.01 }
    let w1 = Knowledge { value: 0.4, epsilon: 0.0 }
    let w2 = Knowledge { value: 0.35, epsilon: 0.0 }
    let w3 = Knowledge { value: 0.25, epsilon: 0.0 }
    var i = 0
    while i < n {
        var v = Knowledge { value: 1000000.0, epsilon: 0.0 }
        v = v * (Knowledge { value: 1.0, epsilon: 0.0 } + ret1 * w1)
        v = v * (Knowledge { value: 1.0, epsilon: 0.0 } + ret2 * w2)
        v = v * (Knowledge { value: 1.0, epsilon: 0.0 } + ret3 * w3)
        i = i + 1
    }
    return 0.0
}
fn main() with IO { let _ = monte_carlo(10000) }
EOF

# Monte Carlo 100K
cat > "$DIR/_bench_monte_carlo_100k.sio" << 'EOF'
fn monte_carlo(n: i32) -> f64 with IO {
    let ret1 = Knowledge { value: 0.08, epsilon: 0.02 }
    let ret2 = Knowledge { value: 0.12, epsilon: 0.03 }
    let ret3 = Knowledge { value: 0.06, epsilon: 0.01 }
    let w1 = Knowledge { value: 0.4, epsilon: 0.0 }
    let w2 = Knowledge { value: 0.35, epsilon: 0.0 }
    let w3 = Knowledge { value: 0.25, epsilon: 0.0 }
    var i = 0
    while i < n {
        var v = Knowledge { value: 1000000.0, epsilon: 0.0 }
        v = v * (Knowledge { value: 1.0, epsilon: 0.0 } + ret1 * w1)
        v = v * (Knowledge { value: 1.0, epsilon: 0.0 } + ret2 * w2)
        v = v * (Knowledge { value: 1.0, epsilon: 0.0 } + ret3 * w3)
        i = i + 1
    }
    return 0.0
}
fn main() with IO { let _ = monte_carlo(100000) }
EOF

# Cockcroft-Gault 10K
cat > "$DIR/_bench_cockcroft_gault_10k.sio" << 'EOF'
fn cockcroft_gault(n: i32) -> f64 with IO {
    var total = Knowledge { value: 0.0, epsilon: 0.0 }
    var i = 0
    while i < n {
        let age_val = 65.0 + ((i % 30) as f64)
        let weight_val = 70.0 + ((i % 40) as f64)
        let scr_val = 1.2 + ((i % 10) as f64) * 0.1
        let age = Knowledge { value: age_val, epsilon: 0.007 }
        let weight = Knowledge { value: weight_val, epsilon: 0.014 }
        let scr = Knowledge { value: scr_val, epsilon: 0.04 }
        let numerator = (Knowledge { value: 140.0, epsilon: 0.0 } - age) * weight
        let denominator = Knowledge { value: 72.0, epsilon: 0.0 } * scr
        let crcl = numerator / denominator
        total = total + crcl
        i = i + 1
    }
    return 0.0
}
fn main() with IO { let _ = cockcroft_gault(10000) }
EOF

# Transcendental 10K
cat > "$DIR/_bench_transcendental_10k.sio" << 'EOF'
fn transcendental(n: i32) -> f64 with IO {
    var x = Knowledge { value: 1.0, epsilon: 0.01 }
    var i = 0
    while i < n {
        let x2 = x * x
        let x3 = x2 * x
        let y1 = x - x3 / Knowledge { value: 6.0, epsilon: 0.0 }
        let y1sq = y1 * y1
        let y2 = Knowledge { value: 1.0, epsilon: 0.0 } - y1sq / Knowledge { value: 2.0, epsilon: 0.0 }
        x = y2 * Knowledge { value: 0.99, epsilon: 0.001 }
        i = i + 1
    }
    return 0.0
}
fn main() with IO { let _ = transcendental(10000) }
EOF

run_bench "GUM Chain (10K ops)" "$DIR/_bench_gum_chain_10k.sio"
run_bench "GUM Chain (100K ops)" "$DIR/_bench_gum_chain_100k.sio"
run_bench "Monte Carlo (10K samples)" "$DIR/_bench_monte_carlo_10k.sio"
run_bench "Monte Carlo (100K samples)" "$DIR/_bench_monte_carlo_100k.sio"
run_bench "Cockcroft-Gault (10K patients)" "$DIR/_bench_cockcroft_gault_10k.sio"
run_bench "Transcendental (10K ops)" "$DIR/_bench_transcendental_10k.sio"

echo ""
echo "============================================================"
echo "All benchmarks complete"
echo "============================================================"

# Cleanup
rm -f "$DIR"/_bench_*.sio
