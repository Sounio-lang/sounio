1. [BLOCKER] Aliased `weight.view(torch.int32).view(num_experts, *packed.shape)` assumes `gptq_marlin_repack` output shape is compatible with the reinterpreted original storage layout, but no evidence this holds for arbitrary `size_n`/`size_k`.
   <location: WEIGHT_BODY lines 14-16>
   <why it matters> Mismatch produces overlapping or out-of-bounds writes into the caller's uint8 buffer, silently corrupting the quantized weights while the later `torch.equal(bits(...))` check still passes on the corrupted data.
   <minimal fix> Replace the view alias with an explicit allocation + copy back only after verifying `packed.shape` against the expected per-expert int32 footprint; keep the original `torch.stack` path unless the exact shape contract is proven.

2. [BLOCKER] `_permute_scales` performs `scales.view(num_experts, *packed.shape)` on `float8_e4m3fn` data whose per-expert numel is asserted equal but whose logical dimensions are not, then writes via `output[i].copy_`.
   <location: SCALE_BODY lines 15-18>
   <why it matters> NVFP4 scale tensors carry explicit `(experts, N, K//16)` layout; a pure numel-preserving reshape produces incorrect scale indexing for Marlin, breaking dequantization even if byte counts match.
   <minimal fix> Compute the expected Marlin scale shape explicitly from `size_n`, `size_n`, `group_size` and reject the in-place path unless it matches exactly.

3. [MAJOR] Loop over experts issues `copy_` into an aliased tensor with no intervening `torch.cuda.synchronize()` or stream-order guarantee before the next `process_weights_after_loading` step.
   <location: both replacement bodies, after the `output[i].copy_` loop>
   <why it matters> The subsequent non-atomic GEMM test and pointer-equality assertions can observe partially written buffers on the same device, producing intermittent bitwise mismatches only visible under concurrent CUDA activity.
   <minimal fix> Add `torch.cuda.synchronize()` immediately after the expert loop inside each helper before returning the aliased tensor.

4. [MAJOR] `assert packed.numel() * packed.element_size() == weight[i].numel()` only checks total bytes, not that the int32 reinterpretation preserves the exact row-major order Marlin expects after `T.contiguous()`.
   <location: WEIGHT_BODY line 11>
   <why it matters> A layout transposition or padding difference inside `gptq_marlin_repack` will make the aliased storage contain transposed or misaligned blocks, yet the byte-count and final `torch.equal` checks will still pass.
   <minimal fix> Add an explicit shape/dtype contract check against the known Marlin int32 weight layout before enabling the in-place alias.

5. [MINOR] `output = None` sentinel plus late `assert output is not None` leaves a window where an empty expert list would return an undefined object, even though `num_experts >= 1` in the test matrix.
   <location: both replacement bodies, the `for i in range(num_experts)` loop>
   <why it matters> Defensive code should not rely on test coverage for an invariant that the production path could violate.
   <minimal fix> Initialize `output` from the first expert's packed tensor and start the loop at 1, or pre-allocate with `torch.empty`.
