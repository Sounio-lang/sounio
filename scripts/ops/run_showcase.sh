#!/bin/bash
# AlphaGeoZero Showcase Runner
# First geometry theorem prover with honest confidence intervals
#
# Usage:
#   ./scripts/run_showcase.sh          # Quick showcase (5 problems)
#   ./scripts/run_showcase.sh full     # Full IMO-AG-30 (30 problems)
#   ./scripts/run_showcase.sh train    # With self-play training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPILER_DIR="$PROJECT_ROOT/compiler"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     AlphaGeoZero - Epistemic Geometry Theorem Prover             ║"
echo "║     First prover with honest confidence intervals                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Build in release mode
echo "🔨 Building compiler (release)..."
cd "$COMPILER_DIR"
cargo build --release 2>&1 | tail -3

# Run the appropriate showcase
MODE="${1:-quick}"

case "$MODE" in
    quick)
        echo ""
        echo "🚀 Running quick showcase (5 IMO problems)..."
        cargo test --release geometry::showcase::tests::test_showcase_config_default -- --nocapture 2>&1 | tail -20
        echo ""
        echo "✅ Quick showcase complete!"
        ;;
    full)
        echo ""
        echo "🚀 Running full IMO-AG-30 benchmark..."
        echo "   This may take 30-60 minutes..."
        cargo test --release geometry::imo_benchmark -- --nocapture 2>&1 | tail -50
        echo ""
        echo "✅ Full benchmark complete!"
        ;;
    train)
        echo ""
        echo "🚀 Running showcase with self-play training..."
        echo "   This may take several hours..."
        cargo test --release geometry::self_play -- --nocapture 2>&1 | tail -50
        echo ""
        echo "✅ Training showcase complete!"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: ./run_showcase.sh [quick|full|train]"
        exit 1
        ;;
esac

echo ""
echo "📊 Key Innovations Demonstrated:"
echo "   • Epistemic MCTS: PUCT + variance bonus explores uncertainty"
echo "   • Beta posteriors: Solve rate as Beta(solved+1, failed+1)"
echo "   • Variance-priority: Learns hardest problems first"
echo "   • Honest metrics: All results include confidence intervals"
echo ""
echo "🔗 Learn more: https://github.com/demetrios-lang/alphageo-zero-d"
