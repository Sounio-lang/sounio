# Raw03, first imported refusal, and live descriptor capture

Root approved source HEAD `e6926020dd215096b24859cf81530f255985ccbe`, archived directly from Git into `/scratch/sounio-stdlib-wave3-20261027/root-integrated03`. No Storage type foundation is present. Every other self-hosted source byte matches raw02's original source; only worker_proof differs. The exact capacity overlay is `eaa61d90e8d2d1897f545636c06496f8b0791fbbf206dca5bd4fa4151926ac60`. Worker main opt-in is added only to the build snapshot and retained as WORKER_OPT_IN.patch.

| Job | Result and scope |
|---|---|
|19587|COMPLETED0/4m12: current Worker source/checker/IR/native component corpus,18 envelope mutations, ordinary CFG/metadata controls and C++23 machine oracle; unmodified compiler-source comparison and before/after guards clean|
|19588|COMPLETED0/23m02: canonical modular build with legacy seed9cf,512MiB stack,32GiB allocation; EXPLICIT_ERRORS empty; source/seed/capacity/compiler guards clean|
|19589|FAILED1/10s: real SDK04 loader gate built runtime/reference0, directcheck0, directcompile1; no target ELF; identity0/capability0/closure141 fn1 at12. Gate stopped, later positives/negatives/normal cases not executed|
|19610|COMPLETED0 diagnostic: same original refusal with DUMP_ALL_CALLS; collision has16 instructions and one branch/cleanup/label/return envelope|
|19620|COMPLETED0 experiment recorder: original and distinct copy with only an added extern exit declaration both check0/compile1/141. This does not prove the declaration altered the lowered descriptor; no ELF was emitted|
|19630|COMPLETED0/9s: C++23 ptrace capture of actual descriptors and all16 collision instructions, preserving original141|

The qualified raw is `e0279ff62e97e861415cd8b105b854ae645aa61865b33b1e2b91184d2c28bf54`, at `root-integrated03/results/madaros`. Its legacy producer is `9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a`. SDK04 seal is `13f80e8c2afc2de5027ea2c070dd708c8dca8855c22e54cdb1b6bbcd90b10aaa`. Build, preflight and loader recipes and selected original logs/manifests are retained here; complete trees/ELFs remain in the scratch roots pending export19636.

The live descriptor of fn23 __rgn_exit_scalar has instr_count0, reg_count0, label_count0, param_count0, strategy0, returns_float0, sret0, bss0, first_ref0, f128/ref_struct masks0 and profiler-1. Enter fn16 has the same empty placeholder schema. The CALL still has argc2 and compiler-owned flag64067. Its fixed native emitter67 reads RDI/RSI; descriptor param_count0 is placeholder metadata, not a zero-argument machine ABI.

The actual collision envelope is exactly: instruction0 enter16 flag64066 producesr1; instruction12 BranchFalse r1 targets label1; instruction13 calls exit23 with flag64067, argc2 and cached operandsr1/r2; instruction14 is label1; instruction15 returnsr2. There is one envelope, not multiple epilogues. The previous guard's requirement for descriptor param_count2 therefore conflicts with the compiler's real factory0.

The C++ diagnostic locates the unique machine function using its error constants plus region flags, disassembles its101 callsites, and observes ir_region_get and the guard call. It validates the selected module's35 functions, named records, observed17824-byte record spacing, region generation and unchanged instruction snapshots. Function-table entries in this legacy-produced compiler are indirect record pointers; the diagnostic checks that representation rather than interpreting the chunk as inline records. It uses temporary breakpoints in its own child process; neither the compiler ELF file nor the IR is changed. The raw/source hashes and baseline141 are verified afterward.

This capture contains the16 complete IrInstr records and descriptors needed for this refusal. It is not a general arena/pool serializer: call arguments in these instructions are at most two and cached source operands are shown, while global pool columns are not exported. It must not be reused as a full WideCounts arena capture. Earlier diagnostic attempts were rejected before acceptance for formatting/link-generation errors, nonunique machine fingerprints or an incorrect inline-entry assumption; their paths remain under worker-raw03-live-schema01..06. No failed attempt is an accepted schema receipt.

The existing Model2 native components remain historical evidence of the hand emitter's ABI; they do not certify frontend descriptor2. The separately reviewed ABI0 patch and its components must qualify canonical factory0 and reject descriptor2 explicitly. A new integrated CLI build is required after that patch. Public Worker admission, pools, normal pipeline and HeapVec/HeapDeque safety remain open.
