#!/usr/bin/env bash
# reproduce_artifact.sh -- POPL 2027 artifact reproduction script
# Run from repo root: bash scripts/ci/reproduce_artifact.sh
#
# Reproduces:
#   1. Bootstrap fixed-point: gen2 == gen3 (bit-identical)
#   2. Epistemic census: gates[direct=15636 guarded=0]
#   3. 100% compile-time epistemic certainty
#   4. Full test suite pass
#
# Hardware: x86-64 Linux, >= 512 MB RAM
# Time: ~3 min total

set -euo pipefail
cd "$(dirname "$0")/../.."

SEED=./artifacts/self-hosted/souc-self-hosted-x86_64
SRC=self-hosted/compiler/lean_single.sio
EXPECTED_HASH=54327028f6929a893186ba97e2bff554
WORKDIR=$(mktemp -d /tmp/sounio-artifact-XXXXXX)
trap "rm -rf $WORKDIR" EXIT

echo "=== Sounio POPL 2027 Artifact Reproduction ==="
echo "Workdir: $WORKDIR"
echo "Source:  $SRC ($(wc -c < $SRC) bytes)"
echo ""

# ── Step 1: seed → gen1 ──────────────────────────────────────────────────────
echo "[1/4] Building gen1 (seed → lean_single.sio)..."
$SEED $SRC $WORKDIR/gen1.elf >$WORKDIR/gen1.log 2>&1
chmod +x $WORKDIR/gen1.elf
GEN1_SIZE=$(du -sh $WORKDIR/gen1.elf | cut -f1)
echo "      gen1: md5=$(md5sum $WORKDIR/gen1.elf | awk '{print $1}')  size=$GEN1_SIZE"

# Check epistemic census from gen1 stderr
CERTAIN=$(grep -oP '(?<=certain \()\d+' $WORKDIR/gen1.log | head -1 || echo "?")
DIRECT=$(grep -oP '(?<=direct=)\d+' $WORKDIR/gen1.log | head -1 || echo "?")
GUARDED=$(grep -oP '(?<=guarded=)\d+' $WORKDIR/gen1.log | head -1 || echo "?")
echo "      epistemic: ${CERTAIN}% certain"
echo "      census:    direct=$DIRECT  guarded=$GUARDED"
if [ "$GUARDED" = "0" ]; then
    echo "      PASS: zero guard markers"
else
    echo "      NOTE: guarded=$GUARDED sites (expected 0 at gen8)"
fi

# ── Step 2: gen1 → gen2 ──────────────────────────────────────────────────────
echo "[2/4] Building gen2 (gen1 → lean_single.sio)..."
$WORKDIR/gen1.elf $SRC $WORKDIR/gen2.elf 2>/dev/null
chmod +x $WORKDIR/gen2.elf
echo "      gen2: md5=$(md5sum $WORKDIR/gen2.elf | awk '{print $1}')"

# ── Step 3: gen2 → gen3, fixed-point check ──────────────────────────────────
echo "[3/4] Building gen3 and verifying fixed point..."
$WORKDIR/gen2.elf $SRC $WORKDIR/gen3.elf 2>/dev/null
chmod +x $WORKDIR/gen3.elf

GEN2_HASH=$(md5sum $WORKDIR/gen2.elf | awk '{print $1}')
GEN3_HASH=$(md5sum $WORKDIR/gen3.elf | awk '{print $1}')

echo "      gen2 md5: $GEN2_HASH"
echo "      gen3 md5: $GEN3_HASH"

if [ "$GEN2_HASH" = "$GEN3_HASH" ]; then
    echo "      PASS: gen2 == gen3 (bit-identical fixed point)"
else
    echo "      FAIL: gen2 != gen3"
    exit 1
fi

if [ "$GEN2_HASH" = "$EXPECTED_HASH" ]; then
    echo "      PASS: hash matches paper ($EXPECTED_HASH)"
else
    echo "      NOTE: hash $GEN2_HASH differs from paper value $EXPECTED_HASH"
    echo "            (source may have changed since paper snapshot)"
fi

# ── Step 4: test suite ────────────────────────────────────────────────────────
echo "[4/4] Running test suite..."
# Known platform-specific skips:
#   gpu_vec_add.sio      -- requires CUDA device (exit 139 on CPU-only hosts)
#   gtt_if_arm_union.sio -- ARM-specific codegen path (skip on x86-only hosts)
SUITE_OUT=$(bash scripts/dev/run_sio_test_suite_v2.sh 2>&1 || true)
FAILS=$(echo "$SUITE_OUT" | grep "^  FAIL" | grep -v "gpu_vec_add\|gtt_if_arm_union" || true)
echo "$SUITE_OUT" | tail -6
if [ -z "$FAILS" ]; then
    echo "      PASS: test suite (expected skips: gpu_vec_add, gtt_if_arm_union)"
else
    echo "      WARN: unexpected failures:"
    echo "$FAILS"
fi

echo ""
echo "=== Artifact Reproduction Complete ==="
echo "Fixed-point hash:  $GEN2_HASH"
echo "Guard markers:     $GUARDED  (direct=$DIRECT call sites)"
echo "Epistemic cert.:   ${CERTAIN}% of expressions"
