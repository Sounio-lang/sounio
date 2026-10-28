# Integrated compiler frontier: worker capabilities and BigFrame array access

This evidence uses the immutable integrated compiler at /scratch/sounio-stdlib-wave3-20261027/root-integrated01/results/madaros, SHA25611332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4. Root built it in 19344 with Worker opt-in only in the sealed snapshot. This is not public WorkerFn acceptance or a production compiler deployment.

The original BigFrame witness remains SHA2566f95285b3f9c239382a7f44616644b7cdeeca1740daa51e177edf838c3b48a23. All library comparisons use root-integrated01/source/stdlib. Neither bf_rext_by nor the witness was rewritten.

## Findings

The old observable boolean probe now prints exactly LHS_VISITED and AND_FALSE in both loop-region modes. The grouped100k rolling-extrema component passes both modes, including its original assertions and BEFORE/AFTER_GROUPED_ROLLING_MAX markers.19406 runs the identical four ELFs compiled by19403; executable guards pass. The former eager-RHS failure is closed for these controls.

The reversible full-body diagnostic19407 checks and compiles in both modes. Mode0 exits181 after marker0266, reporting arena exhaustion on a544-byte allocation while entering the VALUE_COUNTS section. Mode1 exits139 after marker1773.19409 adds30 fine markers: it reaches original line 1781, the first bf_median_scaled_by call, and faults before line 1782. Removing all 108 marker statements reconstructs the exact original witness. This locates the current failure much later than the old rolling-max call775.

19412 extracts original lines1775–1782 and the exact approx_eq helper into a five-row component. It checks/compiles0 and SIGSEGVs in both modes. That excludes dependence on the large main function or on opt-in loop regions for this particular failure.

19426 compares9 ABI producers in both modes:6/7 integer words, mixed integer/f64 arguments, five-argument wrappers calling seven-argument helpers, and scalar/BigFrame returns. Every argument contributes to the result, with seventh sentinel707. All 18 controls pass native execution and the independent C++23 IEEE754-bit oracle. These controls do not establish every possible ABI signature, but they exclude a generic7-argument explanation for this witness.

The same19426 compares the scaled median source against an isolated stdlib clone. The clone changes exactly one expression inside bf_group_quantile_scaled: read_f64(res as *mut f64, k) to res[k]. Original scaled code SIGSEGVs in both modes; the index clone passes in both modes with3.75, exact bits4615626668101337088. Original stdlib hashes remain unchanged. The clone is diagnostic and has not been adopted as a repair.

19429 reduces the representation contrast further. Typed array access at lengths4 and1024 passes in both modes. Casting the array value to *mut f64 SIGSEGVs at both lengths and through a by-value helper, six failures. Casting a mutable array reference, or forwarding an array reference through a helper, returns normally but reads bits1 rather than bits4615626668101337088, six wrong-output refusals. Thus adding a reference is not a validated repair. The results are consistent with reading a handle/slot without resolving its payload; that representation explanation remains an inference until emitter/IR or fault-register attribution. The language contract for array-to-pointer casts still needs an explicit decision.

## Worker boundary

19397 COMPLETED0/8s MaxRSS411768 KiB qualifies 14 semantic refusal fixtures in both check and compile,28 phases. Captures, raw functions, forged capabilities, extraction outside the physical bridge, module-name spoofs, transitive IO/heap/global/aggregate/indirect calls and ABI mismatch are all refused without ELF output. Unsupported intrinsic/parser/E137 failures cannot qualify these controls. Source/compiler guards pass.

19389 passes thread checking after the new fixture's main declares the required Alloc and Div effects, then refuses native emission with final-IR identity10/capability201/closure0.19396 repeats that result and independently records F64 and I64 pool check0/compile-timeout124 at the original600-second deadlines. The job is FAILED1/20m07s, MaxRSS1007144 KiB. No positive pool native result or end-to-end normal pipeline is claimed. Core owns identity/remap and proof classification; no proof bypass was introduced.

The earlier file/TCP serial components and numerical normal-kernel oracles remain separate evidence. This round did not execute the full parallel CSV/TCP-to-Arrow gate after its positive preflight failed.

## Preserved failed attempts and resource evidence

-19386 fails staging because the root compiler snapshot excludes runtime/. No case executes. The separate immutable runtime archive matches root/services SHA256092cf6bd6cf1bfe54857b308b532a777b1edbffd8f7b592f8e637546583fd29f.
-19388 refuses the new thread fixture with E035 for omitted Alloc/Div. Only those effect declarations are added; its body and assertions remain unchanged.
-19399 fails0:53 before script execution, with no stdout; the new log directory was root-owned755. Owner metadata and the correction to the Slurm user are retained.
-19403's component native127 results occur before Sounio execution because /usr/bin/time is absent on DL380p. GNU time is copied to a private tools directory, not installed globally. The same job's full-program compilation exceeds8 GiB: two cgroup OOM kills, MaxRSS8386632 KiB, Slurm OUT_OF_MEMORY0:125. Its collection-status file0 does not override that state.
-19405 is canceled while pending on an impossible afterok dependency; it never runs.19406 reuses the four valid component ELFs instead of recompiling them.
-19407 uses 32 GiB; mode 0 compilation measures12053048 KiB. The earlier8 GiB reduction incorrectly estimated compilation from native RSS. The measured compiler peak is retained; no source or deadline was relaxed.
-19412 reuses an overly broad SCOPE string from its parent script. Its EXTRACTION.json, source hash and actual8 GiB resource request define the small-component experiment; it is not a full-body pass.
-19415 builds the original C++23 oracle on DL380.19417 fails before cases because DL380p does not mount its OrangeFS path.19421 repeats that storage-path refusal before compilation.19422 successfully builds the same oracle using a copied, qualified GCC 13 private sysroot and local scratch in 2s. No package installation, system tool overwrite or mount change occurs.
-19423 refuses reuse of the failed attempt's output directory; no case executes. Subsequent results use fresh directories.
-19425's initial F64 observations fail lowering at lower.sio9712 because directly printing f64_to_bits(...) has unresolved scalar kind. The final new fixtures bind the same expression to an explicit i64 local before printing. Inputs, computations, sentinels and oracle expectations are unchanged. The initial failure logs are retained.

## Reproduction and custody

The portable collector is tests/fixtures/wave4_bigframe_array/diagnostic.sh. It requires Slurm and a fresh output, compiles the C++23 oracle, requires three actual wrong-output refusals, seals source/compiler/stdlib/SDK/executables, and records 38 cases in 152 phase rows. Its success marker means collection completed; actual behavior failures remain in PHASES.tsv.

Worker results are under /scratch/sounio-stdlib-wave3-20261027/services/normal-integrated01. BigFrame marker results are under services/bigframe-integrated01. ABI results are under services/bigframe-abi07/results05, and minimal arrays under services/bigframe-array08/results. BigFrame/ABI/array results produced on DL380p are relayed to the same paths on DL380 for durable export. The private GCC 13 toolchain and all failed attempts are retained.

This evidence does not close the whole stdlib corpus, the default arena lifetime frontier, generic HeapVec, strict BIDI conformance, public WorkerFn, or the parallel scientific pipeline. No push or main merge was performed.

The portable collector was executed as19432; see portable-19432.md for all 152 phase counts, resource measurements and guard results.
