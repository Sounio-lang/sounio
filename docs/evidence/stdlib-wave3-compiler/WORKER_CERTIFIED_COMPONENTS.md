# Worker capability compiler components

This component preserves the closed public native boundary introduced by
`18ab3f749fe33123ca4524563985e7354a77f409`. Production `compiler/main.sio` does
not call `worker_lower_backend_opt_in()`. The implementation and the component
receipts below are prerequisites for a future complete loader/native gate;
they do not qualify the public pool APIs, imported normal kernels, or HeapVec.

## Implementation

The checker records constructor requests by caller FnSigID, source span,
resolved target FnSigID, ABI and compilation epoch. Opaque parameter/return
signatures activate tracking even in a graph with no constructor. A graph
with any checker error cannot issue a lowering token. Initial lowering binds
a definition using its physical source owner, name and body span; explicit
module remaps preserve that defining identity. A cache or sidecar has no
constructor authority by itself. Raw extraction additionally requires the
loader's trusted physical `stdlib/ffi/worker.sio` identity.

Final IR is checked independently. Constructor operands must resolve to one
fresh named function, and opaque words can cross only matching fresh source
contracts. MAY-kind joins refuse capability/plain/FnRef and scalar/reference
mixtures. A CFG worklist additionally requires FnRef, opaque, stack-reference
and region-token definitions on every incoming path at each critical use.
It accepts valid repeated Copy definitions and loop back-edges. Callback CFGs
must have unique bounded labels, existing branch targets and no reachable
fallthrough beyond the function. Phi capability operands remain closed.

The callback closure admits scalar word operations and confined scalar stack
references, with resolved direct callees and an audited builtin mapping. It
rejects heap/aggregate/global/indirect/extern state, wide references, profiling
state and unmodeled operations. The legacy native LoadImm(-4242) println
sentinel is refused in certified IR because it emits IO. This is an explicit
remaining restriction on the full i64 domain, not a source integer whitelist.

The native writer makes a private deep IR copy, clones the complete certified
closure, retargets its direct edges and removes only the proven compiler region
operations. The original serial functions retain their regions and automatic
store barriers. Only exact certified clones omit IrStorePtr's automatic
barrier. Deep sealed IR snapshots are compared again at emission. Pooled
arguments are copied with `ir_arena_store_from`, including valid pool base zero;
plain `ir_arena_load` does not reconstruct those lists.

Protected raw extractors contain an immutable address allowlist separated by
ABI. Only actual emitted entry clones enter it; helper/original addresses,
invalid words and the other ABI trap before the FFI bridge. An empty list
refuses all words. This protects the typed API contract under code integrity;
it is not a sandbox against code using raw operations to rewrite process code.

## Evidence and reproducible gate

All builds and execution ran in Slurm on dl380. The gate requires a Slurm
allocation, a fresh output directory, the legacy standalone seed and a modern
raw checker. It copies the compiler, stdlib and fixtures, injects private test
metadata only in that disposable copy, keeps source/seed/ELF hashes and refuses
explicit bootstrap diagnostics even when the seed exits zero:

```
bash tests/fixtures/wave4_worker_capability/components/gate.sh \
  FRESH_OUTPUT_DIR LEGACY_SEED MODERN_RAW
```

The canonical runtime thunk is copied verbatim in `runtime_thunk.inc` from
`services/runtime/stdlib_platform.cpp`; its slice SHA256 is
`21212af3ad4b23e29183636b0804f59c1011322e7749eea4051b3c7edd4325de`.
The runtime source at extraction had SHA256
`092cf6bd6cf1bfe54857b308b532a777b1edbffd8f7b592f8e637546583fd29f`.
No runtime thunk or production callee-save convention was changed.

* **19149:** real parser/specializer/checker/lower identity and module remap
  component passed. The first AST import is deliberate for the legacy seed's
  Name layout handling. ELF SHA256
  `df244880bea923afee80f9bfacc45c99321f313d83095917f72eb9d796813d5f`.
* **19259 / 19273:** six actual single-file programs passed checker, lowering,
  identity, opaque flow, closure proof and cloning: i64, f64, transitive helper,
  confined stack references, recursion and opaque forward/return roundtrip.
  A graph without constructors retains its opaque signature contract; a raw
  function is refused even when lowering is forced after its checker error;
  a separate unrelated type error prevents constructor token issuance.
  19273 ELF SHA256
  `f13cff87d8e55ef167e80d6da388ea563645f64f073b8b4e0b6aeb37a534bab5`.
* **19276 paired before / 19272 after:** the same six CFG mutations compile
  without explicit seed errors against both sources. The previous proof
  refused only the pre-existing mixed-kind mutation (1/6) and accepted five
  skipped-definition defects. The corrected proof refuses 6/6. Both-branch
  FnRef/capability merges and a reference through a loop remain accepted.
  19272 also preserves all previous fifteen final IR mutations. ELF SHA256
  `a5a0ec4fa95956a29ca6626cc14e0b35be1753d0c5b084b245ccd2a270caa17d`.
* **19193 / 19277:** actual emitted machine code passed the C++23 canonical
  thunk oracle with eleven positive checks, eleven invalid word/ABI SIGILL
  controls, three deliberately wrong expected-result/signal controls and four
  empty-list refusals. Callee-save canaries and stack alignment are checked.
  Exact f64 words include +0, -0, qNaN payload and 0.5. All writable ELF data
  and BSS segments are PROT_NONE during callback/extractor execution. Sentinel,
  unsupported pointer width and unknown unary op fail before ELF output.
* **19277:** the portable gate passes six actual sources, the 15+6 IR controls,
  three valid CFG cases, modern native typecheck across 36 modules, legacy
  native generation and the independent C++23 machine-code oracle.
* **19284:** final portable gate additionally refuses three scalar CFG mutants:
  missing label, duplicate label and reachable function fallthrough. COMPLETED
  0 in 4m10; all source/seed guards and explicit bootstrap error counts are
  zero. Modern 8-module IR and 36-module native checks both return verdict 0.
  Final worker-output ELF SHA256 is
  `2774b2de43e10b8b14d255a1aef8523a19234fc8b50e3b1b7cbbec7208076852`,
  identical to the earlier qualified machine-code artifact. Receipts, complete
  compressed compile logs, disassembly and the exact five source hashes plus
  template hashes are included in `worker-certified-components/`.

Durable receipts are under
`/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/` in
`worker-identity-ast-first`, `worker-boundary-v2`, `worker-boundary-v3`,
`worker-ir-cfg-v5`, `worker-cfg-before`, `worker-native-abi-v7`,
`worker-portable-gate-v2` and `worker-portable-gate-v3`. The final source tree
contains no capacity WIP. Original source and compiler hash guards are checked
after execution. The failed portable attempt 19275 is preserved separately:
its packaging omitted the stdlib target of the `self-hosted/crypto` symlink;
it failed unreadable-import E224 before a driver ELF and was never accepted.

## Boundaries still open

The six actual sources exercise a single-file parser/checker/lower graph.
The adversarial IR and native code components deliberately inject source
metadata only in the snapshot; they are not imported/BFS admission receipts.
The native oracle currently exercises small scalar entries. Actual native
stack-reference helper barrier elision, canonical normal closure clones,
mixed concurrent/serial callers, physical worker-module admission through the
full loader, cache invalidation/reproof, and the services pool/thread matrix
still require the integrated candidate. The canonical runtime thunk already
preserves callee-save registers; direct C ABI calls to internal Sounio code
are not the native callback contract.

Earlier full 48-module modern identity-driver attempts remain unresolved:
19085 timed out without a verdict; 19126 was cancelled at the root's request
after 19m36 without a verdict. They are not pass evidence. No production
Worker issuance is enabled by these component results.
