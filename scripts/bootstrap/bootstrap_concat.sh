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
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PINNED_BIN="${PINNED_BIN:-bin/souc-linux-x86_64}"
if [ ! -x "$PINNED_BIN" ]; then
  JIT_FALLBACK="bin/souc"
  if [ -x "$JIT_FALLBACK" ]; then
    echo "bootstrap_concat: pinned binary not found at $PINNED_BIN, falling back to $JIT_FALLBACK"
    PINNED_BIN="$JIT_FALLBACK"
  else
    echo "bootstrap_concat: no usable souc binary found (checked $PINNED_BIN and $JIT_FALLBACK)" >&2
    exit 1
  fi
fi
OUTPUT_DIR="build"
OUTPUT_FILE="$OUTPUT_DIR/bootstrap_stage1.sio"
ARTIFACT_DIR="artifacts/omega"
BOOTSTRAP_PROFILE="${BOOTSTRAP_PROFILE:-core}"
BOOTSTRAP_INCLUDE_TESTS="${BOOTSTRAP_INCLUDE_TESTS:-0}"
BOOTSTRAP_FULL_INCLUDE_EXPERIMENTAL="${BOOTSTRAP_FULL_INCLUDE_EXPERIMENTAL:-0}"
BOOTSTRAP_FULL_INCLUDE_HYPERCOMPLEX="${BOOTSTRAP_FULL_INCLUDE_HYPERCOMPLEX:-0}"

# ── Dependency-ordered file list (leaf-first) ──────────────────────
# Full bootstrap: ALL self-hosted .sio files.
# ~110K+ lines — the entire self-hosted compiler, GPU stack, and test suite.
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
  self-hosted/parser/recovery.sio

  # ── 4. Resolve layer ────────────────────────────────────────────
  self-hosted/resolve/scope.sio
  self-hosted/resolve/symbol.sio
  self-hosted/resolve/imports.sio
  self-hosted/resolve/resolve.sio
  self-hosted/resolve/mod.sio
  self-hosted/resolve/scopes.sio
  self-hosted/resolve/modules.sio
  self-hosted/resolve/packages.sio

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
  self-hosted/check/ontology_side_table_cache.sio
  self-hosted/check/noise_sets.sio
  self-hosted/check/check.sio
  self-hosted/check/mod.sio
  self-hosted/check/patterns.sio
  self-hosted/check/infer.sio
  self-hosted/check/const_eval.sio
  self-hosted/check/traits.sio
  self-hosted/check/macros.sio
  self-hosted/check/lifetimes.sio
  self-hosted/check/borrows.sio
  self-hosted/check/monomorph.sio
  self-hosted/check/pat_decision.sio
  self-hosted/check/type_erasure.sio
  self-hosted/check/gadts.sio
  self-hosted/check/dependent.sio
  self-hosted/check/ownership.sio
  self-hosted/check/lint.sio
  self-hosted/check/layout.sio

  # ── 6. Effects layer ────────────────────────────────────────────
  self-hosted/effects/types.sio
  self-hosted/effects/mod.sio
  self-hosted/effects/checker.sio
  self-hosted/effects/handlers.sio

  # ── 6b. Diagnostics engine ────────────────────────────────────
  self-hosted/diagnostics/mod.sio

  # ── 6c. Analysis layer ──────────────────────────────────────────
  self-hosted/analysis/abstract_interp.sio

  # ── 7. IR layer ─────────────────────────────────────────────────
  self-hosted/ir/ir.sio
  self-hosted/ir/algebra.sio
  self-hosted/ir/lower.sio
  self-hosted/ir/normalize.sio
  self-hosted/ir/serialize.sio
  self-hosted/ir/disasm.sio
  self-hosted/ir/verify.sio
  self-hosted/ir/optimize.sio
  self-hosted/ir/ssa.sio
  self-hosted/ir/closure.sio
  self-hosted/ir/dep_graph.sio
  self-hosted/ir/cfg.sio
  self-hosted/ir/inline.sio
  self-hosted/ir/async_lower.sio
  self-hosted/ir/dce.sio
  self-hosted/ir/tailcall.sio
  self-hosted/ir/const_prop.sio
  self-hosted/ir/opt_strategy.sio
  self-hosted/ir/profile.sio
  self-hosted/ir/egraph.sio
  self-hosted/ir/opt_cleanup.sio
  self-hosted/ir/loop_opt.sio
  self-hosted/ir/auto_vectorize.sio
  self-hosted/ir/memory_analysis.sio
  self-hosted/ir/incremental.sio
  self-hosted/ir/locality.sio
  self-hosted/ir/mod.sio

  # ── 8. HLIR (High-Level IR) ─────────────────────────────────────
  self-hosted/hlir/ir.sio
  self-hosted/hlir/builder.sio
  self-hosted/hlir/opt_strategy.sio
  self-hosted/hlir/lower.sio
  self-hosted/hlir/mod.sio

  # ── 9. Native codegen layer ─────────────────────────────────────
  self-hosted/native/mod.sio
  self-hosted/native/regs.sio
  self-hosted/native/encode.sio
  self-hosted/native/runtime_context.sio
  self-hosted/native/target_policy.sio
  self-hosted/native/frame.sio
  self-hosted/native/reloc.sio
  self-hosted/native/abi.sio
  self-hosted/native/lower_hir.sio
  self-hosted/native/elf.sio
  self-hosted/native/machine_ir.sio
  self-hosted/native/gc.sio
  self-hosted/native/stack_maps.sio
  self-hosted/native/peephole.sio
  self-hosted/native/codegen.sio
  self-hosted/native/lower_ir.sio
  self-hosted/native/riscv.sio
  self-hosted/native/aarch64.sio
  self-hosted/native/dwarf.sio
  self-hosted/native/regalloc.sio
  self-hosted/native/codegen_plan.sio
  self-hosted/native/abi_lower.sio
  self-hosted/native/debug_info.sio
  self-hosted/native/macho.sio
  self-hosted/native/ffi.sio
  self-hosted/native/avx512.sio
  self-hosted/native/pe_coff.sio
  self-hosted/native/signal_handler.sio
  self-hosted/native/profiling.sio
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
  self-hosted/vm/gc.sio
  self-hosted/vm/jit_cache.sio
  self-hosted/vm/green_threads.sio
  self-hosted/vm/mod.sio

  # ── 12. Collections ─────────────────────────────────────────────
  self-hosted/collections/arena.sio
  self-hosted/collections/ordered_map.sio
  self-hosted/collections/graph.sio
  self-hosted/collections/epistemic_ordered_map.sio
  self-hosted/collections/epistemic_molecule.sio
  self-hosted/collections/epistemic_arena.sio
  self-hosted/collections/epistemic_reaction.sio

  # ── 13. Hypercomplex ────────────────────────────────────────────
  self-hosted/hypercomplex/octonion.sio
  self-hosted/hypercomplex/quat_simd.sio

  # ── 14. Tensor ──────────────────────────────────────────────────
  self-hosted/tensor/contract.sio
  self-hosted/tensor/ops.sio

  # ── 15. Linker ──────────────────────────────────────────────────
  self-hosted/linker/mod.sio
  self-hosted/linker/elf_writer.sio

  # ── 15b. Tools ─────────────────────────────────────────────────
  self-hosted/tools/formatter.sio

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
  self-hosted/gpu/opt/autotune.sio
  self-hosted/gpu/hlir_to_gpu.sio
  self-hosted/gpu/epistemic_kernels.sio
  self-hosted/gpu/epistemic_ptx.sio
  self-hosted/gpu/counterfactual.sio
  self-hosted/gpu/numerical.sio
  self-hosted/gpu/metal.sio
  self-hosted/gpu/spirv.sio
  self-hosted/gpu/spirv_text.sio
  self-hosted/gpu/spirv_lower.sio
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

  # ── 22. WebAssembly backend ─────────────────────────────────────
  self-hosted/wasm/emitter.sio
  self-hosted/wasm/encode.sio
  self-hosted/wasm/lower.sio

  # ── 22b. LSP protocol + completions ────────────────────────────
  self-hosted/lsp/protocol.sio
  self-hosted/lsp/completions.sio
  self-hosted/lsp/hover.sio
  self-hosted/lsp/goto_def.sio

  # ── 23. I/O helpers + module loader ─────────────────────────────
  self-hosted/interop/contract.sio
  self-hosted/io/env.sio
  self-hosted/io/trace.sio
  self-hosted/io/file_write.sio
  stdlib/compiler/ontology/model.sio
  stdlib/compiler/ontology/cache.sio
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
  self-hosted/test_epistemic_wmma.sio
  self-hosted/test_epistemic_tiled_gemm.sio
  self-hosted/test_epistemic_spirv.sio
  self-hosted/test_epistemic_hlir_gpu.sio
  self-hosted/test_epistemic_autodiff.sio
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
    self-hosted/parser/recovery.sio

    # 4. Resolve layer
    self-hosted/resolve/scope.sio
    self-hosted/resolve/symbol.sio
    self-hosted/resolve/imports.sio
    self-hosted/resolve/resolve.sio
    self-hosted/resolve/mod.sio
    self-hosted/resolve/scopes.sio
    self-hosted/resolve/modules.sio
    self-hosted/resolve/packages.sio

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
    self-hosted/check/ontology_side_table_cache.sio
    self-hosted/check/noise_sets.sio
    self-hosted/check/check.sio
    self-hosted/check/mod.sio
    self-hosted/check/patterns.sio
    self-hosted/check/infer.sio
    self-hosted/check/const_eval.sio
    self-hosted/check/traits.sio
    self-hosted/check/macros.sio
    self-hosted/check/lifetimes.sio
    self-hosted/check/borrows.sio
    self-hosted/check/monomorph.sio
    self-hosted/check/pat_decision.sio
    self-hosted/check/type_erasure.sio
    self-hosted/check/gadts.sio
    self-hosted/check/dependent.sio
    self-hosted/check/ownership.sio
    self-hosted/check/lint.sio
    self-hosted/check/layout.sio

    # 5b. Diagnostics engine
    self-hosted/diagnostics/mod.sio

    # 5c. Analysis layer
    self-hosted/analysis/abstract_interp.sio

    # 5d. Effects checker
    self-hosted/effects/checker.sio
    self-hosted/effects/handlers.sio

    # 6. IR layer
    self-hosted/ir/ir.sio
    self-hosted/ir/algebra.sio
    self-hosted/ir/lower.sio
    self-hosted/ir/normalize.sio
    self-hosted/ir/serialize.sio
    self-hosted/ir/disasm.sio
    self-hosted/ir/verify.sio
    self-hosted/ir/optimize.sio
    self-hosted/ir/ssa.sio
    self-hosted/ir/closure.sio
    self-hosted/ir/dep_graph.sio
    self-hosted/ir/cfg.sio
    self-hosted/ir/inline.sio
    self-hosted/ir/async_lower.sio
    self-hosted/ir/dce.sio
    self-hosted/ir/tailcall.sio
    self-hosted/ir/const_prop.sio
    self-hosted/ir/egraph.sio
    self-hosted/ir/loop_opt.sio
    self-hosted/ir/auto_vectorize.sio
    self-hosted/ir/memory_analysis.sio
    self-hosted/ir/incremental.sio
    self-hosted/ir/locality.sio
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
    self-hosted/native/machine_ir.sio
    self-hosted/native/runtime_context.sio
    self-hosted/native/target_policy.sio
    self-hosted/native/gc.sio
    self-hosted/native/stack_maps.sio
    self-hosted/native/peephole.sio
    self-hosted/native/codegen.sio
    self-hosted/native/lower_ir.sio
    self-hosted/native/riscv.sio
    self-hosted/native/aarch64.sio
    self-hosted/native/dwarf.sio
    self-hosted/native/regalloc.sio
    self-hosted/native/codegen_plan.sio
    self-hosted/native/abi_lower.sio
    self-hosted/native/debug_info.sio
    self-hosted/native/macho.sio
    self-hosted/native/ffi.sio
    self-hosted/native/avx512.sio
    self-hosted/native/pe_coff.sio
    self-hosted/native/signal_handler.sio
    self-hosted/native/profiling.sio
    self-hosted/native/hyper_lower.sio

    # 8. WebAssembly backend
    self-hosted/wasm/emitter.sio
    self-hosted/wasm/encode.sio
    self-hosted/wasm/lower.sio

    # 8b. Tensor + Linker + Tools expansion
    self-hosted/tensor/ops.sio
    self-hosted/linker/elf_writer.sio
    self-hosted/tools/formatter.sio

    # 8c. LSP protocol + completions
    self-hosted/lsp/protocol.sio
    self-hosted/lsp/completions.sio
    self-hosted/lsp/hover.sio
    self-hosted/lsp/goto_def.sio

    # 8d. VM layer
    self-hosted/vm/vm.sio
    self-hosted/vm/gc.sio
    self-hosted/vm/jit_cache.sio
    self-hosted/vm/green_threads.sio
    self-hosted/vm/mod.sio

    # 9. I/O helpers + module loader
    self-hosted/io/env.sio
    self-hosted/io/trace.sio
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

if [ "$BOOTSTRAP_PROFILE" = "full" ]; then
  if [ "$BOOTSTRAP_INCLUDE_TESTS" = "1" ]; then
    echo "=== Full profile: including tests/demos (stubs removed to avoid collisions) ==="
  else
    echo "=== Full profile note: skipping standalone test/demo entrypoints (set BOOTSTRAP_INCLUDE_TESTS=1 to include) ==="
    echo "=== Full profile note: keeping bootstrap test stubs for main() dispatch compatibility ==="
  fi
  if [ "$BOOTSTRAP_FULL_INCLUDE_EXPERIMENTAL" != "1" ]; then
    echo "=== Full profile note: skipping experimental GPU bundles (set BOOTSTRAP_FULL_INCLUDE_EXPERIMENTAL=1) ==="
  fi
  if [ "$BOOTSTRAP_FULL_INCLUDE_HYPERCOMPLEX" != "1" ]; then
    echo "=== Full profile note: skipping hypercomplex bundles (set BOOTSTRAP_FULL_INCLUDE_HYPERCOMPLEX=1) ==="
  fi

  FILTERED_FILES=()
  for f in "${FILES[@]}"; do
    # If real tests are included, drop bootstrap stubs to avoid duplicate defs.
    if [ "$BOOTSTRAP_INCLUDE_TESTS" = "1" ]; then
      case "$f" in
        self-hosted/test_stubs_bootstrap.sio)
          continue
          ;;
      esac
    else
      # For production-focused full profile, keep stubs and skip standalone tests/demos.
      case "$f" in
        self-hosted/test_stubs_bootstrap.sio)
          FILTERED_FILES+=("$f")
          continue
          ;;
        self-hosted/native/suite.sio|\
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
    fi

    if [ "$BOOTSTRAP_FULL_INCLUDE_EXPERIMENTAL" != "1" ]; then
      case "$f" in
        self-hosted/gpu/opt/optimizer.sio|\
        self-hosted/gpu/opt/divergence.sio|\
        self-hosted/gpu/opt/async_pipeline.sio|\
        self-hosted/gpu/kernels/*.sio|\
        self-hosted/gpu/autodiff/*.sio|\
        self-hosted/gpu/multi/*.sio|\
        self-hosted/gpu/quant/*.sio|\
        self-hosted/gpu/analysis/*.sio)
          continue
          ;;
      esac
    fi

    if [ "$BOOTSTRAP_FULL_INCLUDE_HYPERCOMPLEX" != "1" ]; then
      case "$f" in
        self-hosted/hypercomplex/*.sio)
          continue
          ;;
      esac
    fi

    FILTERED_FILES+=("$f")
  done
  FILES=("${FILTERED_FILES[@]}")
fi

# ── De-duplicate file list while preserving order ──────────────────
declare -A _SEEN_FILES=()
UNIQUE_FILES=()
DUP_FILE_COUNT=0
for f in "${FILES[@]}"; do
  if [ -n "${_SEEN_FILES[$f]+x}" ]; then
    DUP_FILE_COUNT=$((DUP_FILE_COUNT + 1))
    continue
  fi
  _SEEN_FILES[$f]=1
  UNIQUE_FILES+=("$f")
done
if [ "$DUP_FILE_COUNT" -gt 0 ]; then
  echo "=== Bootstrap note: dropped $DUP_FILE_COUNT duplicate file list entries ==="
fi
FILES=("${UNIQUE_FILES[@]}")

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

# ── Create output/artifact directories ─────────────────────────────
mkdir -p "$OUTPUT_DIR"
mkdir -p "$ARTIFACT_DIR"

# ── Concatenate in dependency order ────────────────────────────────
echo ""
echo "=== Concatenating ${#FILES[@]} files -> $OUTPUT_FILE ==="

TOTAL_LINES=0
STRIPPED_DECL_LINES=0

# Start fresh
> "$OUTPUT_FILE"

for f in "${FILES[@]}"; do
  LINES=$(wc -l < "$f")
  TOTAL_LINES=$((TOTAL_LINES + LINES))
  DECL_LINES=$(grep -Ec '^[[:space:]]*(module[[:space:]]+[A-Za-z0-9_:]+[[:space:]]*;?|use[[:space:]].*)[[:space:]]*$' "$f" || true)
  STRIPPED_DECL_LINES=$((STRIPPED_DECL_LINES + DECL_LINES))

  {
    echo ""
    echo "// ════════════════════════════════════════════════════════════════"
    echo "// SOURCE: $f  ($LINES lines)"
    echo "// ════════════════════════════════════════════════════════════════"
    echo ""
    awk '
      /^[[:space:]]*module[[:space:]]+[A-Za-z0-9_:]+[[:space:]]*;?[[:space:]]*$/ { next }
      in_use {
        if ($0 ~ /^[[:space:]]*}[[:space:]]*$/) {
          in_use = 0
        }
        next
      }
      /^[[:space:]]*use[[:space:]].*\{[[:space:]]*$/ {
        in_use = 1
        next
      }
      /^[[:space:]]*use[[:space:]].*$/ { next }
      { print }
    ' "$f"
  } >> "$OUTPUT_FILE"

  if [ "$DECL_LINES" -gt 0 ]; then
    printf "  %-45s %6d lines  (stripped %d module/use decls)\n" "$f" "$LINES" "$DECL_LINES"
  else
    printf "  %-45s %6d lines\n" "$f" "$LINES"
  fi
done

echo ""
echo "  Total: $TOTAL_LINES source lines -> $OUTPUT_FILE"
OUTPUT_LINES=$(wc -l < "$OUTPUT_FILE")
echo "  Output file: $OUTPUT_LINES lines (includes separator comments)"
echo "  Stripped module/use declarations: $STRIPPED_DECL_LINES"

# ── Run the pinned binary compile ──────────────────────────────────
#
# IMPORTANT: the souc ELF (lean_single.sio's compiled main) takes its
# source path as positional argv[1] — there is NO `check` subcommand,
# despite the historical naming.  Invoking `souc check <file>` would
# have souc try to open a source file literally named "check" (which
# doesn't exist → 0 bytes read), then write the empty-source ELF to
# <file> — clobbering the bundle.  Until 2026-05-18 this script
# silently produced a "no main" error from compiling the prelude alone
# and reported it as the bundle's typecheck status.
#
# Fix: invoke positionally — `souc <bundle> <elf-out>` — so the
# checker actually exercises the concatenated bundle.  The bundle has
# no `fn main`, so a successful compile-stage will still exit non-zero
# with "error: no main".  We treat that single error as PASS at the
# typecheck level (it means the bundle typechecked clean and only
# failed at the entry-point lookup).  Any OTHER error count, or a
# SIGSEGV (exit 139), is a real bundle-compile failure.
echo ""
BUNDLE_ELF_OUT="$ARTIFACT_DIR/bootstrap_stage1.elf"
if "$PINNED_BIN" --help 2>&1 | grep -q "Commands:"; then
  PINNED_CMD=( "$PINNED_BIN" build "$OUTPUT_FILE" -o "$BUNDLE_ELF_OUT" )
else
  PINNED_CMD=( "$PINNED_BIN" "$OUTPUT_FILE" "$BUNDLE_ELF_OUT" )
fi
echo "=== Running pinned compile: ${PINNED_CMD[*]} ==="
echo ""

CHECKER_EXIT=0
FULL_CHECK_LOG="$ARTIFACT_DIR/bootstrap_full_check.log"
set +e
CHECKER_STACK_KB="${CHECKER_STACK_KB:-unlimited}"
if [ "$BOOTSTRAP_PROFILE" = "full" ]; then
  (
    ulimit -s "$CHECKER_STACK_KB" 2>/dev/null || true
    "${PINNED_CMD[@]}" 2>&1
  ) | tee "$FULL_CHECK_LOG"
  CHECKER_EXIT="${PIPESTATUS[0]}"
else
  (
    ulimit -s "$CHECKER_STACK_KB" 2>/dev/null || true
    "${PINNED_CMD[@]}" 2>&1
  )
  CHECKER_EXIT="$?"
fi
set -e

if [ "$BOOTSTRAP_PROFILE" = "full" ] && [ "$CHECKER_EXIT" -ne 0 ]; then
  DUP_COUNT="$(grep -c "Duplicate definition" "$FULL_CHECK_LOG" || true)"
  UNDEF_COUNT="$(grep -c "Undefined variable" "$FULL_CHECK_LOG" || true)"
  TYPE_MISMATCH_COUNT="$(grep -c "Type mismatch" "$FULL_CHECK_LOG" || true)"
  RESOLUTION_COUNT="$(grep -c "Resolution errors" "$FULL_CHECK_LOG" || true)"
  STACK_OVERFLOW_COUNT="$(grep -c "stack overflow" "$FULL_CHECK_LOG" || true)"

  echo ""
  echo "=== BOOTSTRAP_FULL_ERROR_SUMMARY ==="
  echo "  duplicate_definitions: $DUP_COUNT"
  echo "  undefined_variables:   $UNDEF_COUNT"
  echo "  type_mismatches:       $TYPE_MISMATCH_COUNT"
  echo "  resolution_blocks:     $RESOLUTION_COUNT"
  echo "  stack_overflows:       $STACK_OVERFLOW_COUNT"
  echo "  full_check_log:        $FULL_CHECK_LOG"
  echo "===================================="

  SUMMARY_FILE="$ARTIFACT_DIR/bootstrap_full_error_summary.v1.json"
  cat > "$SUMMARY_FILE" <<EOF
{
  "schema": "sounio.bootstrap.full-error-summary.v1",
  "generated_at_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "profile": "full",
  "exit_code": $CHECKER_EXIT,
  "counts": {
    "duplicate_definitions": $DUP_COUNT,
    "undefined_variables": $UNDEF_COUNT,
    "type_mismatches": $TYPE_MISMATCH_COUNT,
    "resolution_blocks": $RESOLUTION_COUNT,
    "stack_overflows": $STACK_OVERFLOW_COUNT
  },
  "log_path": "$FULL_CHECK_LOG",
  "output_file": "$OUTPUT_FILE"
}
EOF
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
