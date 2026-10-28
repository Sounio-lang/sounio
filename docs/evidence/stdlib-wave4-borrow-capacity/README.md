# Borrow capacity: persistent storage candidate

Status: persistent borrow storage, real checker and BFS compiler gates qualify. The full original BigFrame native gate remains open; whole-stdlib acceptance is not claimed. No main merge or push.

## Causal reproduction

The unchanged original BigFrame fixture is SHA256 `6f95285b3f9c239382a7f44616644b7cdeeca1740daa51e177edf838c3b48a23`. Slurm19090 tests plain scalar bindings without any reference expression: proto02 accepts128 and refuses129/160/512 with E038 in both loop-region modes. Its original BigFrame reports89E037 and4354E038. The old21b3 compiler accepts these scalar cases but also accepts the genuine returned-loan mutation refusal; reverting that safety work is not a correction.

## Representation and invariants

BorrowEnv is a small header of immutable roots. A radix16 trie stores entry slots, stable binding-ID mappings, consumed bits, name buckets, owner/carrier indexes and deduplication bits. Updates allocate a fresh leaf and path; forks retain old roots. Binding lookup validates the mapped slot's actual stable ID, including after shadowing, scope pop and temporary compaction. Names use a hash bucket and exact comparison.

Every published BorrowEntry and Name is explicitly copied by constructor; getters also return explicit copies. The implementation does not rely on aggregate assignment copying an object. Entry cells have an8192-byte payload budget and64-byte tail canaries. The canary is checked at publication and read. It is a runtime bound guard, not a language sizeof or an address-of-field layout oracle.

Scope and branch frames are linked immutable records. Branch frames retain before/then consumed roots and phase; they have no16-level fallback. The checker preserves linear AND and affine OR merge rules. All direct BorrowEnv entry/branch-array consumers in check.sio were migrated. Public publication rejects replacement or duplicate active stable IDs and invalid Name lengths before copying. Only the private compactor may relocate an existing ID to a lower slot. Captures and call-argument snapshots clone the root header explicitly.

Loan lists carry validated lengths. Stable-ID indexes replace the nested entry scan; owner/carrier indexes and dedup radix maps avoid a scan of every old edge on each insertion. The former4096 traversal bound is removed. The shared arena holds only allocation/error metadata. Its resource error is sticky across forks and backtracking; E397 refuses compilation with a storage reason instead of reporting a fabricated user borrow conflict. The legacy loan_overflow semantic poison for a dead owner remains distinct from failure_kind/arena.error, so genuine escapes keep E091.

Pages remain allocated while any fork may reference them. This patch deliberately has no arena release API at an unproven ownership boundary; lifetime is the compiler process. Per-function reclamation remains a later ownership task.

## Preserved component receipts

-19109: initial component native assertion failure; preserved.
-19113: two-entry diagnostic probe only; not integrated acceptance.
-19137: functional512/4097/24 native assertions pass; the invalid address-difference layout probe fails the C++ guard. This probe did not establish contiguous field-reference addresses.
-19187: canary replacement, library/unit check and native0, C++23 marker guard0; all mutable fields, caller/getter mutation, forks, shadow/slot reuse,512 entries,4097 loans and24 branch snapshots pass. RSS658944KiB. The separate standalone checker-source check timed out124 after300seconds across33 modules; this is not a checker pass.
-19192: the same storage assertions and injected invalid index, byte-accounting overflow, corrupted cell canary, corrupted radix depth and corrupt loan length pass in modes0/1. Synthetic OOM-state propagation passes, explicitly not allocator fault injection.
-19194: the legacy9cf checker build emits E200 for `as *const i64` in the canary reader and is rejected even though the seed writes anELF. Candidate now uses the existing `*mut i64` pointer spelling for that read.
-19190: canceled without compiler verdict at parent request after preserving source/seed guards, last logs, sstat and job state. Its shared preimage emitted E200 in19194; it is not qualified.
-19202: corrected checker component accepts0/1/126/127/128/129/160/512 scalar bindings, refuses genuine illegal loans with E038, accepts balanced24 and refuses asymmetry24 with E040. The full raw build19202 was canceled without a raw verdict after preserving guards and resource/job state: its failure-kind/persist-edge function preimages are byte-identical to the diagnosed19214 regression.
-19211: exact public-setter hardening passes storage/native/C++ marker guards in modes0/1. Duplicate IDs, replacement of stable IDs and Name.len385 fail before publication; old roots and cells remain intact, while the arena error remains sticky. A private compactor relocation2→1 succeeds and preserves the older fork.
-19214: actual checker component preserves all51 statuses, but C++ refuses three changed escape diagnostics (E397 instead of E091). This failure is preserved and corrected by separating semantic poison from storage errors.
-19225: COMPLETED0 in1m36, batch MaxRSS2057224KiB,2CPU/4GiB. Exact final source passes storage512/4097/24+hardening and resource/semantic controls in modes0/1. Checker matrix passes51 (21positive/30negative), all original diagnostic codes, and7 wrong-output refusals;129/160/512 pass, balanced24 passes and asymmetry24 produces E040. Source guards and required outputs pass. Versioned receipt subset is in19225/.
-19228: build0 after22m11; exact raw826d9c18efbc8dc878633c355a8a1e76985aa40fa7faa47490e3544ecddb9143. Both modes pass all51 raw checks/compilations and21 positive native cases, all wrong-output controls, plus capacity and deep-branch gates. Original BigFrame check/compile pass in both modes, both baseline and immutable75d library overlay. Native fails181 (arena full) in mode0 and139 (SIGSEGV) in mode1. Job correctly FAILED1 after29m18; every final source/seed/compiler guard passes.
-19246: derived prefixes100/160/200/240/320/512 compile and run0 in mode1; these are diagnostics, not full-witness acceptance.
-19255/19258: full-body marker derivatives reproduce139 and locate it in the original line775 call to bf_rolling_max_by. Removing only the inserted println statements restores the exact original SHA.
-19264: standalone gsf100k rolling-max reduction also SIGSEGV139 in both modes. Observable lhs()/rhs() witness prints RHS_VISITED even when lhs is false in both modes, establishing eager && in raw826d. No library workaround or test alteration was applied; root owns upstream ADR014 integration.

The C++23 reference for the storage component checks required output markers. It is an artifact guard, not an independent mathematical or layout model.

## Integration custody

Worktree base12f277c8acbf8890fd58aa8a196910a38510a284. The full build copies sealed proto03/source without reapplying its capacity.patch. Its check.sio is a clean three-way merge of the12f base, this lane's migration and proto03's f861ef4811/ffeab2 SourceProfile changes. Revision02 merged checker SHA256 fd9b97d61fa3660dafea578c726671b248a4b0b3003fd3f392ca8c4abc54a094; borrow SHA256 dc9a538524cb22a98a55219261be58e568c0fdb54a91aa8e338ef0986c37ae9d.

Revision03 final borrow SHA256 d5559a728fd43d40c8b3ac8e36fc9f02648ed39816466a014c282636ac441220; merged checker remains fd9b97d61fa3660dafea578c726671b248a4b0b3003fd3f392ca8c4abc54a094. Revision03 checks errors after linear-scope traversal and preserves the reason of standalone source-list failures. Its semantic/storage distinction preserves all51 original diagnostics. The source/hash manifests distinguish it from earlier components. Build uses exact seed9cf, loop regions0 and no cache. Both loop-region modes and all existing returned-loan witnesses remain required.

Reproducible gates live in tests/fixtures/wave4_borrow_capacity: gate_storage.sh, gate_checker_component.sh (single-file checker), and gate_capacity.sh (requires rebuilt BFS compiler). Orchestration and results are in /workspace/.tmp/borrow-paged-* and /scratch/sounio-stdlib-wave3-20261027/borrow-paged-integration03. Earlier failures remain in integration01/02 and services/borrow-paged-* directories. Durable export19266 COMPLETED0 after1m32 (MaxRSS2131304KiB). Archive /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/borrow-profiler-export01/borrow-profiler.tar.gz; SHA2566ee48112220c17f9008b00264e7c58eddf00afd1eee32422c06f8d1f98902edb. It includes all before/failure/cancel receipts, frozen sources, seeds/raws, marker/standalone diagnostics, profiler both modes, DSO and commit patches. gzip and archive SHA verification pass. This custody success does not turn19228 or BigFrame native into a pass.
