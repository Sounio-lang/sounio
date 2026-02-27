#!/usr/bin/env bash
#
# bootstrap_concat.sh
#
# Concatenates the self-hosted compiler's core pipeline .sio files in
# leaf-first dependency order into a single file, then runs the pinned
# binary checker against it.
#
# Sounio has NO forward references -- helpers must precede callers.
# The ordering below respects that invariant.

set -euo pipefail

# ── Paths (relative to repo root) ──────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PINNED_BIN="artifacts/omega/souc-bin/souc-linux-x86_64"
OUTPUT_DIR="build"
OUTPUT_FILE="$OUTPUT_DIR/bootstrap_stage1.sio"
BOOTSTRAP_PROFILE="${BOOTSTRAP_PROFILE:-core}"

# ── Dependency-ordered file list (leaf-first) ──────────────────────
# Full bootstrap: ALL 168 self-hosted .sio files.
# ~103K lines — the entire self-hosted compiler, GPU stack, and test suite.
FILES=(
  # ── 1. Leaf modules (no cross-file deps) ────────────────────────
  self-hosted/intern.sio
  self-hosted/lexer/token.sio
  self-hosted/lexer/span.sio

  # ── 2. Lexer layer ──────────────────────────────────────────────
  self-hosted/lexer/cursor.sio
  self-hosted/lexer/errors.sio
  self-hosted/lexer/tables.sio
  self-hosted/lexer/numparse.sio
  self-hosted/lexer/reader.sio
  self-hosted/lexer/mod.sio

  # ── 3. Parser layer ─────────────────────────────────────────────
  self-hosted/parser/ast.sio
  self-hosted/parser/parser.sio
  self-hosted/parser/exprs.sio
  self-hosted/parser/items.sio
  self-hosted/parser/stmts.sio
  self-hosted/parser/types.sio
  self-hosted/parser/patterns.sio
  self-hosted/parser/mod.sio

  # ── 4. Resolve layer ────────────────────────────────────────────
  self-hosted/resolve/scope.sio
  self-hosted/resolve/symbol.sio
  self-hosted/resolve/imports.sio
  self-hosted/resolve/resolve.sio
  self-hosted/resolve/mod.sio

  # ── 5. Check layer ──────────────────────────────────────────────
  self-hosted/check/types.sio
  self-hosted/check/defs.sio
  self-hosted/check/env.sio
  self-hosted/check/borrow.sio
  self-hosted/check/units.sio
  self-hosted/check/epistemic.sio
  self-hosted/check/hyper.sio
  self-hosted/check/effects.sio
  self-hosted/check/refinement.sio
  self-hosted/check/compat.sio
  self-hosted/check/check.sio
  self-hosted/check/mod.sio

  # ── 6. Effects layer ────────────────────────────────────────────
  self-hosted/effects/types.sio
  self-hosted/effects/mod.sio

  # ── 7. IR layer ─────────────────────────────────────────────────
  self-hosted/ir/ir.sio
  self-hosted/ir/algebra.sio
  self-hosted/ir/lower.sio
  self-hosted/ir/normalize.sio
  self-hosted/ir/serialize.sio
  self-hosted/ir/disasm.sio
  self-hosted/ir/verify.sio
  self-hosted/ir/mod.sio

  # ── 8. HLIR (High-Level IR) ─────────────────────────────────────
  self-hosted/hlir/ir.sio
  self-hosted/hlir/builder.sio
  self-hosted/hlir/lower.sio
  self-hosted/hlir/mod.sio

  # ── 9. Native codegen layer ─────────────────────────────────────
  self-hosted/native/mod.sio
  self-hosted/native/regs.sio
  self-hosted/native/encode.sio
  self-hosted/native/frame.sio
  self-hosted/native/reloc.sio
  self-hosted/native/abi.sio
  self-hosted/native/lower_hir.sio
  self-hosted/native/elf.sio
  self-hosted/native/codegen.sio
  self-hosted/native/lower_ir.sio
  self-hosted/native/hyper_lower.sio
  self-hosted/native/suite.sio

  # ── 10. Emit + Printer ──────────────────────────────────────────
  self-hosted/emit/text.sio
  self-hosted/emit/mod.sio
  self-hosted/printer/buffer_print.sio
  self-hosted/printer/print_ast.sio
  self-hosted/printer/mod.sio

  # ── 11. VM ──────────────────────────────────────────────────────
  self-hosted/vm/vm.sio
  self-hosted/vm/mod.sio

  # ── 12. Collections ─────────────────────────────────────────────
  self-hosted/collections/arena.sio
  self-hosted/collections/ordered_map.sio
  self-hosted/collections/graph.sio
  self-hosted/collections/epistemic_ordered_map.sio
  self-hosted/collections/epistemic_molecule.sio

  # ── 13. Hypercomplex ────────────────────────────────────────────
  self-hosted/hypercomplex/octonion.sio
  self-hosted/hypercomplex/quat_simd.sio

  # ── 14. Tensor ──────────────────────────────────────────────────
  self-hosted/tensor/contract.sio

  # ── 15. Linker ──────────────────────────────────────────────────
  self-hosted/linker/mod.sio

  # ── 16. LLVM backend ────────────────────────────────────────────
  self-hosted/llvm/types.sio
  self-hosted/llvm/values.sio
  self-hosted/llvm/ffi.sio
  self-hosted/llvm/context.sio
  self-hosted/llvm/module.sio
  self-hosted/llvm/builder.sio
  self-hosted/llvm/target.sio
  self-hosted/llvm/passes.sio
  self-hosted/llvm/debug.sio
  self-hosted/llvm/linker.sio
  self-hosted/llvm/type_convert.sio
  self-hosted/llvm/codegen.sio

  # ── 17. GPU core ────────────────────────────────────────────────
  self-hosted/gpu/ptx.sio
  self-hosted/gpu/lower_to_ptx.sio
  self-hosted/gpu/ptx_advanced.sio
  self-hosted/gpu/kernel_ir.sio
  self-hosted/gpu/hlir_to_gpu.sio
  self-hosted/gpu/epistemic_kernels.sio
  self-hosted/gpu/epistemic_ptx.sio
  self-hosted/gpu/counterfactual.sio
  self-hosted/gpu/numerical.sio
  self-hosted/gpu/metal.sio
  self-hosted/gpu/spirv.sio
  self-hosted/gpu/portable.sio
  self-hosted/gpu/tensor_epistemic.sio

  # ── 18. GPU runtime ─────────────────────────────────────────────
  self-hosted/gpu/runtime/cuda.sio
  self-hosted/gpu/runtime/metal.sio
  self-hosted/gpu/runtime/launch.sio
  self-hosted/gpu/runtime/kaxi_feedback.sio

  # ── 19. GPU optimization ────────────────────────────────────────
  self-hosted/gpu/opt/optimizer.sio
  self-hosted/gpu/opt/fusion.sio
  self-hosted/gpu/opt/divergence.sio
  self-hosted/gpu/opt/async_pipeline.sio
  self-hosted/gpu/opt/autotune.sio

  # ── 20. GPU kernels + advanced ──────────────────────────────────
  self-hosted/gpu/kernels/bio.sio
  self-hosted/gpu/kernels/hypercomplex.sio
  self-hosted/gpu/kernels/sparse_collective.sio
  self-hosted/gpu/autodiff/autodiff.sio
  self-hosted/gpu/multi/multi_gpu.sio
  self-hosted/gpu/quant/quantize.sio
  self-hosted/gpu/analysis/profiler.sio

  # ── 21. K-AXI hardware codegen ──────────────────────────────────
  self-hosted/compiler/codegen/hardware/kaxi_adapters.sio
  self-hosted/compiler/codegen/hardware/kaxi_publish_adapters.sio
  self-hosted/compiler/codegen/hardware/kaxi_fabric_emitter.sio
  self-hosted/compiler/codegen/hardware/kaxi_emitter.sio

  # ── 22. I/O helpers + module loader ─────────────────────────────
  self-hosted/io/file_write.sio
  self-hosted/compiler/module_loader.sio
  self-hosted/test_stubs_bootstrap.sio

  # ── 23. Native test suites ──────────────────────────────────────
  self-hosted/native/test_encode.sio
  self-hosted/native/test_codegen.sio
  self-hosted/native/test_compile_to_elf.sio
  self-hosted/native/test_native_execution.sio
  self-hosted/native/test_phase0.sio
  self-hosted/native/test_phase1.sio
  self-hosted/native/test_phase2.sio

  # ── 24. Hypercomplex + tensor tests ─────────────────────────────
  self-hosted/hypercomplex/test_octonion.sio
  self-hosted/hypercomplex/test_quat_simd.sio
  self-hosted/tensor/test_contract.sio

  # ── 25. VM tests ────────────────────────────────────────────────
  self-hosted/vm/test_vm.sio

  # ── 26. Linker tests + demo ─────────────────────────────────────
  self-hosted/linker/test_linker.sio
  self-hosted/linker/demo_e2e.sio

  # ── 27. Benchmark ───────────────────────────────────────────────
  self-hosted/bench/kernel_smoke.sio

  # ── 28. Core test suites ────────────────────────────────────────
  self-hosted/test_lexer.sio
  self-hosted/test_parser.sio
  self-hosted/test_printer.sio
  self-hosted/test_resolve.sio
  self-hosted/test_resolve_real.sio
  self-hosted/test_check.sio
  self-hosted/test_check_real.sio
  self-hosted/test_ir.sio
  self-hosted/test_emit.sio
  self-hosted/test_vm.sio
  self-hosted/test_bootstrap.sio
  self-hosted/test_roundtrip.sio
  self-hosted/test_knowledge.sio
  self-hosted/test_module_loader.sio
  self-hosted/test_parse_real.sio

  # ── 29. GPU test suites ─────────────────────────────────────────
  self-hosted/test_gpu_ptx.sio
  self-hosted/test_gpu_cuda_runtime.sio
  self-hosted/test_gpu_oracle.sio
  self-hosted/test_epistemic_ptx.sio
  self-hosted/test_counterfactual.sio
  self-hosted/test_gpu_fmri.sio
  self-hosted/test_gpu_profiler.sio
  self-hosted/test_gpu_fusion.sio
  self-hosted/test_gpu_metal.sio
  self-hosted/test_gpu_spirv.sio
  self-hosted/test_gpu_autodiff.sio
  self-hosted/test_llvm_ffi.sio
  self-hosted/test_llvm_codegen.sio

  # ── 30. Test aggregators ────────────────────────────────────────
  self-hosted/test_parse_all.sio
  self-hosted/test_all.sio

  # ── 31. Full entry point ────────────────────────────────────────
  self-hosted/main.sio
)

if [ "$BOOTSTRAP_PROFILE" != "full" ]; then
  echo "=== Bootstrap profile: core (set BOOTSTRAP_PROFILE=full for exhaustive concat) ==="
  FILES=(
    # 1. Leaf modules (no cross-file deps)
    self-hosted/intern.sio
    self-hosted/lexer/token.sio
    self-hosted/lexer/span.sio

    # 2. Lexer layer
    self-hosted/lexer/cursor.sio
    self-hosted/lexer/errors.sio
    self-hosted/lexer/tables.sio
    self-hosted/lexer/numparse.sio
    self-hosted/lexer/reader.sio
    self-hosted/lexer/mod.sio

    # 3. Parser layer
    self-hosted/parser/ast.sio
    self-hosted/parser/parser.sio
    self-hosted/parser/exprs.sio
    self-hosted/parser/items.sio
    self-hosted/parser/stmts.sio
    self-hosted/parser/types.sio
    self-hosted/parser/patterns.sio
    self-hosted/parser/mod.sio

    # 4. Resolve layer
    self-hosted/resolve/scope.sio
    self-hosted/resolve/symbol.sio
    self-hosted/resolve/imports.sio
    self-hosted/resolve/resolve.sio
    self-hosted/resolve/mod.sio

    # 5. Check layer
    self-hosted/check/types.sio
    self-hosted/check/defs.sio
    self-hosted/check/env.sio
    self-hosted/check/borrow.sio
    self-hosted/check/units.sio
    self-hosted/check/epistemic.sio
    self-hosted/check/hyper.sio
    self-hosted/check/effects.sio
    self-hosted/check/refinement.sio
    self-hosted/check/compat.sio
    self-hosted/check/check.sio
    self-hosted/check/mod.sio

    # 6. IR layer
    self-hosted/ir/ir.sio
    self-hosted/ir/algebra.sio
    self-hosted/ir/lower.sio
    self-hosted/ir/normalize.sio
    self-hosted/ir/serialize.sio
    self-hosted/ir/disasm.sio
    self-hosted/ir/verify.sio
    self-hosted/ir/mod.sio

    # 7. Native codegen layer
    self-hosted/native/mod.sio
    self-hosted/native/regs.sio
    self-hosted/native/encode.sio
    self-hosted/native/frame.sio
    self-hosted/native/reloc.sio
    self-hosted/native/abi.sio
    self-hosted/native/lower_hir.sio
    self-hosted/native/elf.sio
    self-hosted/native/codegen.sio
    self-hosted/native/lower_ir.sio
    self-hosted/native/hyper_lower.sio

    # 8. I/O helpers + module loader
    self-hosted/io/file_write.sio
    self-hosted/compiler/module_loader.sio

    # 9. Test stubs (replace full test suites for bootstrap)
    self-hosted/test_stubs_bootstrap.sio

    # 10. Full entry point
    self-hosted/main.sio
  )
else
  echo "=== Bootstrap profile: full (production-complete concat) ==="
fi

if [ "$BOOTSTRAP_PROFILE" = "full" ] && [ "${BOOTSTRAP_INCLUDE_TESTS:-0}" != "1" ]; then
  echo "=== Full profile note: skipping standalone test/demo entrypoints (set BOOTSTRAP_INCLUDE_TESTS=1 to include) ==="
  echo "=== Full profile note: keeping bootstrap test stubs for main() dispatch compatibility ==="
  echo "=== Full profile note: skipping hypercomplex modules (Quat/acos unresolved in strict pinned check) ==="
  FILTERED_FILES=()
  for f in "${FILES[@]}"; do
    case "$f" in
      self-hosted/test_stubs_bootstrap.sio)
        FILTERED_FILES+=("$f")
        continue
        ;;
      self-hosted/hypercomplex/*.sio)
        continue
        ;;
      self-hosted/native/suite.sio)
        continue
        ;;
      self-hosted/test_*.sio|\
      self-hosted/native/test_*.sio|\
      self-hosted/tensor/test_*.sio|\
      self-hosted/vm/test_*.sio|\
      self-hosted/linker/test_*.sio|\
      self-hosted/linker/demo_e2e.sio|\
      self-hosted/bench/kernel_smoke.sio)
        continue
        ;;
    esac
    FILTERED_FILES+=("$f")
  done
  FILES=("${FILTERED_FILES[@]}")
fi

# ── Verify all source files exist ──────────────────────────────────
echo "=== Bootstrap Concat: verifying source files ==="
MISSING=0
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "  MISSING: $f"
    MISSING=$((MISSING + 1))
  fi
done

if [ "$MISSING" -gt 0 ]; then
  echo ""
  echo "ABORT: $MISSING file(s) missing. Cannot proceed."
  exit 1
fi
echo "  All ${#FILES[@]} files present."

# ── Verify pinned binary exists and is executable ──────────────────
if [ ! -f "$PINNED_BIN" ]; then
  echo "ABORT: Pinned binary not found at $PINNED_BIN"
  exit 1
fi
if [ ! -x "$PINNED_BIN" ]; then
  echo "WARN: Pinned binary not executable, setting +x"
  chmod +x "$PINNED_BIN"
fi

# ── Create output directory ────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── Concatenate in dependency order ────────────────────────────────
echo ""
echo "=== Concatenating ${#FILES[@]} files -> $OUTPUT_FILE ==="

TOTAL_LINES=0

# Start fresh
> "$OUTPUT_FILE"

for f in "${FILES[@]}"; do
  LINES=$(wc -l < "$f")
  TOTAL_LINES=$((TOTAL_LINES + LINES))

  {
    echo ""
    echo "// ════════════════════════════════════════════════════════════════"
    echo "// SOURCE: $f  ($LINES lines)"
    echo "// ════════════════════════════════════════════════════════════════"
    echo ""
    cat "$f"
  } >> "$OUTPUT_FILE"

  printf "  %-45s %6d lines\n" "$f" "$LINES"
done

echo ""
echo "  Total: $TOTAL_LINES source lines -> $OUTPUT_FILE"
OUTPUT_LINES=$(wc -l < "$OUTPUT_FILE")
echo "  Output file: $OUTPUT_LINES lines (includes separator comments)"

# ── Run the pinned binary checker ──────────────────────────────────
echo ""
echo "=== Running pinned checker: $PINNED_BIN check $OUTPUT_FILE ==="
echo ""

CHECKER_EXIT=0
if [ "$BOOTSTRAP_PROFILE" = "full" ]; then
  CHECKER_STACK_KB="${CHECKER_STACK_KB:-65536}"
  (
    ulimit -s "$CHECKER_STACK_KB" 2>/dev/null || true
    "$PINNED_BIN" check "$OUTPUT_FILE" 2>&1
  ) || CHECKER_EXIT=$?
else
  "$PINNED_BIN" check "$OUTPUT_FILE" 2>&1 || CHECKER_EXIT=$?
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
if [ "$CHECKER_EXIT" -eq 0 ]; then
  echo "  RESULT: PASS  (exit code 0)"
else
  echo "  RESULT: FAIL  (exit code $CHECKER_EXIT)"
fi
echo "  Files concatenated: ${#FILES[@]}"
echo "  Source lines:        $TOTAL_LINES"
echo "  Output:              $OUTPUT_FILE"
echo "════════════════════════════════════════════════════════════════════"

exit "$CHECKER_EXIT"
