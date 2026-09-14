# Marlin repack review disposition — 2026-09-07

Input packet SHA256: c85f3034a5fae978ee41df8e7529140cec294bb055d93060caa37eed46a59e4d.

The default math-review attempt returned ZAI/local errors and a Grok4.6 timeout.
The DeepSeek fallback also returned an API error. These are failed reviews.
The xai-fast math route answered “NO MATHEMATICAL CONTENT TO REVIEW”; this is
neither a code approval nor mathematical verification. A separate xai-fast
code review produced the findings retained in marlin-inplace-review.md.

Disposition by the implementing agent, based on the pinned source and independent
stock/candidate GPU controls (job11888):

1. Shape/alias bounds: the code checks the exact uint8 input shape and expert
   byte count, and PyTorch view validates the total element count. A contiguous
   view of (num_experts, *packed.shape) divides the existing buffer into disjoint
   expert regions; no unchecked pointer arithmetic is used. The candidate writes
   each kernel-produced packed tensor into its corresponding region. Stock and
   candidate inputs are independent clones; byte comparisons are against the
   independently transformed stock result, so shared corruption cannot explain
   the pass. The claim of silent out-of-bounds writes from this view is declined.
2. Scale layout: the destination view uses the actual output shape of the same
   stock permutation/process functions, exactly as stock torch.stack does. It
   does not expose the old scale layout after returning. Every consumed
   transformed parameter is compared bytewise to stock. The claim that reshaping
   alone loses the transformation is declined: copy_ writes the transformed
   values, rather than merely returning a reshape of the old values.
3. Synchronization: operations are issued on the same current CUDA stream during
   offline process_weights_after_loading. Stream ordering already orders the
   producer kernel, copy, and downstream consumer; no foreign stream is
   introduced by this patch. A device-wide synchronize is not needed to make
   this sequence correct. Online/hot reload, LoRA, and alternate-stream mutation
   are outside this pinned runtime profile.
4. Transposition: the original view(int32).T.contiguous() and exact original
   Marlin repack invocation remain in place; copy_ preserves the returned
   logical tensor values. Exact transformed parameter comparisons and real
   non-atomic GPU kernel controls pass. A second speculative transpose would
   alter the representation and is not introduced.
5. Empty experts: both helpers already end with assert output is not None.
   An empty expert set raises instead of returning an undefined object.
   The fixed checkpoint has 256 experts; this is not a general dynamic-expert API.

Qualification covered 8/256/128, 4/4096/1024, and the actual TP2 expert geometry
256/4096/1024 on both GB10 nodes. Per node, 692736 output components were
compared exactly with a test-only non-atomic reduction; all consumed transformed
parameter bytes match stock, the consumed-scale negative changes the result,
and in-place storage pointers are preserved. Peak extra allocated storage for
the 256-expert case is 12584960 bytes. Production atomic reduction repeat
variability is preserved in the receipts and is not claimed bitwise deterministic.

Actual read-only source mounts passed on both ranks in job11889. These are
layer and mounting controls, not full-model loading or serving acceptance.
The original image, checkpoint, quantization, and production kernels are unchanged.
The local source overlay is explicit and must not be represented as an
unmodified upstream runtime.
