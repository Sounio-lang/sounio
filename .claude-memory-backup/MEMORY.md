# Sounio Project Memory

## Machine

`/home/demetrios/RustroverProjects/sounio` IS the canonical dev machine.
No cargo/Rust needed — souc binary at: `artifacts/omega/souc-bin/souc-linux-x86_64-jit` (v1.0.0-beta.4).
SSH to DEVdesktop: `ssh demetrios@DEVdesktop` — may be offline.

## Project References

- [boot3 self-hosting](project_boot3_selfhost.md) — 935-line bootstrap compiler, ≤5 params, 3 bugs fixed, SELF=109KB/47fn, crash at 0xb
- [Operation Epistemic Dawn](project_epistemic_dawn.md) — NATIVE epistemic PBPK demo: 20KB ELF, 6/6 PASS, 3ms runtime
- [Epistemic computing pipeline](project_epistemic_pipeline.md) — sqrt/Knowledge layout/print_f64 done; GUM arithmetic next
- [F# interop Phase 1](project_fsharp_interop.md) — SNIO binary protocol, IPC server, F# Sounio.Interop library
- [Native compile pipeline status](project_native_compile_status.md) — self-hosted ELF emission WORKS (exit(42) test passes)
- [JIT &! reference bug](feedback_jit_ref_bug.md) — Cranelift JIT &! mutations invisible to caller; use by-value return pattern
- [JIT stdout warning corruption](feedback_jit_stdout_warnings.md) — file_size() etc. print warnings to stdout, corrupting SNIO binary stream
- [Epistemic WMMA Tensor Core](project_epistemic_wmma.md) — world-first GUM uncertainty through GPU WMMA; 13/13 gate PASS
- [Sprint 225 — Uncertainty-Aware E-Graph](project_sprint225.md) — World-first analytical GUM-guided float rewriting; Phase 1 done (f92c1adf), Phase 2-3 pending

## Key Commands

```bash
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit

# Check a .sio file
$SOUC check self-hosted/ir/ir.sio

# Run self-hosted compiler self-test
$SOUC run self-hosted/compiler/main.sio -- --self-test

# Run a gate
bash scripts/sprint38_strategy_codegen_gate.sh
```

## Project Status

- **Sprint 225 Phase 1 complete** (2026-03-15): Uncertainty-aware e-graph infrastructure `f92c1adf`; GUM quantization, isqrt, epistemic extraction, float saturation; 10 new tests (T61–T70); Phases 2–3 pending
- **Sprints 214-222 complete** (2026-03-15): Blocks DX-EF — T950-T1003; total=1003; all gates FAIL=0
  - DX (214): (x|y)|~x→-1; DY (215): x&(y|~x)→x&y; DZ (216): (x|y)&~(x&y)→x^y
  - EA (217): (x^y)^(x|y)→x&y; EB (218): (x&y)^(x^y)→x|y; EC (219): (x-y)-(z-y)→x-z
  - ED (220): (x+y)-(x-z)→y+z; EE (221): (x-y)+(z+y)→x+z; EF (222): (x|y)|(x&y)→x|y
- **Sprints 208-212 complete** (2026-03-14): Blocks DR-DV — T914-T943; total=943; all gates FAIL=0
  - DR (208): ~(~x&~y)→x|y De Morgan NOT-AND; DS (209): ~(~x|~y)→x&y De Morgan NOT-OR
  - DT (210): x|(y&~x)→x|y OR-AND complement absorption; DU (211): (x^y)|(x&y)→x|y XOR-AND to OR
  - DV (212): (x^y)&(x|y)→IrCopy(x^y) XOR-AND-OR simplify
- **Sprints 206-207 complete** (2026-03-14): Blocks DP-DQ — T902-T913; total=913; all gates FAIL=0
  - DP (206): (x&y)^(x|y)→x^y AND-OR XOR identity; DQ (207): x&~(x^y)→x&y AND-XNOR to AND
- **Sprints 203-205 complete** (2026-03-14): Blocks DM-DO — T884-T901; total=901; all gates FAIL=0
  - DM (203): (x|y)-(x&y)→x^y; DN (204): (x-y)-(x-z)→z-y; DO (205): (x+y)-(y+z)→x-z
- **Sprints 198-202 complete** (2026-03-14): Blocks DH-DL — 5 blocks, T854-T883; total=883; all gates FAIL=0
  - DH (198): x&~(x|y)→0; DI (199): x|~(x&y)→-1; DJ (200): ~x-~y→y-x
  - DK (201): (x|y)+(x&y)→x+y; DL (202): (x-y)+(y-z)→x-z
- **Sprints 190-197 complete** (2026-03-14): Blocks CZ-DG — 8 blocks, T806-T853; total=853; all gates FAIL=0
  - CZ (190): x^(x&C)→x&~C; DA (191): (x|y)^(x&y)→x^y; DB (192): (x^y)|(x&y)→x|y
  - DC (193): (x&y)|(x^y)→x|y; DD (194): (x|y)&~(x&y)→x^y; DE (195): x^~x→-1
  - DF (196): x|~x→-1; DG (197): ~(x^y)^y→~x
- **Sprint 189 complete** (2026-03-14): Block CY — AND complement addition: (x&C)+(x&~C)→x; T800-T805; total=805; gate 10/16 PASS, FAIL=0; `c5921193`
- **Sprints 183-188 complete** (2026-03-14): Blocks CS-CX — 6 blocks, T764-T799; total=799; all gates FAIL=0
  - CS (183): (x-C1)+C2→x+(C2-C1), (x+C1)-C2→x+(C1-C2) sub-add/add-sub chain
  - CT (184): x*C1-x*C2→x*(C1-C2) mul-const distributive sub [pre-built]
  - CU (185): (x<<A)+(x<<B)→x*((1<<A)+(1<<B)) shift-add factor [pre-built]
  - CV (186): (x<<A)-(x<<B)→x*((1<<A)-(1<<B)) shift-sub factor [pre-built]
  - CW (187): (x|C)^C→x&~C OR-XOR mask extraction
  - CX (188): (x^C)^~C→~x XOR-complement chain to NOT
- **Sprints 177-182 complete** (2026-03-14): Blocks CM-CR batch — 6 blocks, 36 tests; T728-T763; total=763; batch gate 44/80 PASS, FAIL=0
  - CM (177): (x&C)^(x&~C)→x; CN (178): x+x→mul tracking; CO (179): neg(x)+y→y-x
  - CP (180): x-neg(y)→x+y; CQ (181): (x|C)^(x|~C)→~x; CR (182): (x&C)^C→~x&C
- **Sprint 176 complete** (2026-03-14): Block CL — OR-complement AND: (x|C)&~C→x&~C; T722-T727; total=727; gate 13/19 PASS, FAIL=0
- **Sprint 175 complete** (2026-03-14): Block CK — AND-complement OR: (x&~C)|C→x|C; T716-T721; total=721; reuses and_valid/var_src/const_val; gate 10/16 PASS, FAIL=0; `76641676`
- **Sprint 174 complete** (2026-03-14): Block CJ — shift-sub to multiply: (x<<A)-x→x*(2^A-1), x-(x<<A)→x*(1-2^A); T710-T715; total=715; reuses bk_shl_valid/src/amt; gate 12/18 PASS, FAIL=0; `69933ba2`
- **Sprint 173 complete** (2026-03-14): Block CI — shift-add to multiply: (x<<A)+x→x*(2^A+1); T704-T709; total=709; reuses bk_shl_valid; gate 10/16 PASS, FAIL=0; `c5fc3597`
- **Sprints 171-172 complete** (2026-03-14): mul-add/sub normalization; T692-T703
  - Block CG (171): (x*C)+x→x*(C+1); T692-T697; gate 12/18 PASS; `2d63b3a7`
  - Block CH (172): (x*C)-x→x*(C-1); T698-T703; gate 11/17 PASS; `5242d3fa`
- **Sprint 170 complete** (2026-03-14): Block CF — OR-AND superset absorption: (x|C1)&C2→C2 when C2⊆C1; T686-T691; total=691; reuses am_or_valid/am_or_const_val; gate 13/19 PASS, FAIL=0; `39e7ce55`
- **Sprint 169 complete** (2026-03-14): Block CE — mul-const chain fold: (x*C1)*C2→x*(C1*C2); T680-T685; total=685; reuses mul_valid/mul_var_src/mul_const_v; gate 13/19 PASS, FAIL=0; `f6bec405`
- **Sprint 168 complete** (2026-03-14): Block CD — zero-minus normalization: 0-x→neg(x); T674-T679; total=679; updates is_neg/neg_src; gate 13/19 PASS, FAIL=0; `72135b34`
- **Sprints 166-167 complete** (2026-03-14): add/sub inverse + AND complement partition
  - Block CB (166): (x+C)-C→x, (x-C)+C→x, C+(x-C)→x; T662-T667; total=667; gate 10/16 PASS; `56342ce6`
  - Block CC (167): (x&C)|(x&~C)→x; T668-T673; total=673; gate 12/18 PASS; `32202a6c`
- **Sprint 165 complete** (2026-03-14): Block CA — OR-complement AND elimination: (x|C)&(x|~C)→x; T656-T661; total=661; reuses am_or_valid; gate 11/17 PASS, FAIL=0; `3940eb35`
- **Sprints 163-164 complete** (2026-03-14): XOR/AND-OR constant folds; T644-T655
  - Block BY (163): (x^C)|C→x|C; gate 12/18 PASS; `ebbce8c5`
  - Block BZ (164): (x&C)|(x^C)→x|C; gate 13/19 PASS; `6712d0fa`
- **Sprint 162 complete** (2026-03-14): Block BX — OR-XOR complement fold: (x|C)^C→x&~C, C^(x|C)→x&~C; T638-T643; total=643; reuses am_or_valid/var_src/const_val; gate 11/17 PASS, FAIL=0; `ea0e245b`
- **Sprints 160-161 complete** (2026-03-14): bitwise complement chain; T626-T637
  - Block BV (160): (x^C)|C→x|C; gate 13/19 PASS; `bf536d08`
  - Block BW (161): (x^C)&~C→x&~C; gate 12/18 PASS; `09da570b`
- **Sprint 159 complete** (2026-03-14): Block BU — add/sub-const cross-track diff: (x+C1)-(x-C2)→C1+C2, (x-C1)-(x-C2)→C2-C1, etc.; T620-T625; total=625; reuses bt_add_valid/bs_sub_valid; gate 13/19 PASS, FAIL=0; `2a845e97`
- **Sprint 158 complete** (2026-03-14): Block BT — consecutive add-constant fold: (x+C1)+C2→x+(C1+C2), commutative outer; T614-T619; new bt_add_valid/src/cval arrays; gate 9/15 PASS, FAIL=0; `f4f4bfeb`
- **Sprint 157 complete** (2026-03-14): Block BS — consecutive subtract-constant fold: (x-C1)-C2→x-(C1+C2); T608-T613; new bs_sub_valid/src/cval arrays; gate 9/15 PASS, FAIL=0; `d6c2df8f`
- **Sprints 154-156 complete** (2026-03-14): complement annihilation/absorption chain; T590-T607; total=607
  - Block BP (154): x&~x→0; T590-T595; gate 9/15 PASS; `88a2ec04`
  - Block BQ (155): x|~x→-1; T596-T601; gate 9/15 PASS; `b784729d`
  - Block BR (156): x^~x→-1; T602-T607; gate 9/15 PASS; `e01929dd`
- **Sprint 190 complete** (2026-03-14): Block CZ — XOR-AND mask clear: x^(x&C)→x&~C; T806-T811; total=811; gate 12/18 PASS FAIL=0; `5fd2135f`
- **Sprints 167-189 complete** (2026-03-14): Blocks CC-CY — Boolean complement/partition laws, mul/shift chain folds, neg/add/sub normalization; T668-T805; all FAIL=0; linter pre-built most; branch `chore/remote-workspace-migration`
- **Sprint 166 complete** (2026-03-14): Block CB — add/sub-const inverse: (x+C)-C→x, (x-C)+C→x, C+(x-C)→x; T662-T667; total=667; uses bt_add_valid+bs_sub_valid; gate 10/16 PASS FAIL=0; `43e9f9b1`
- **Sprints 154-165 complete** (2026-03-14): Blocks BP-CA — bitwise complement laws + add/sub chains + OR-complement AND elimination; all pre-implemented by linter; T590-T661; gates PASS FAIL=0 each; `chore/remote-workspace-migration`
- **Sprint 153 complete** (2026-03-14): Block BO — two's-complement rewrite: ~x+1→-x, 1+~x→-x; T584-T589; total=589; zero new arrays (reuses is_bnot/bnot_src); gate 9/15 PASS, FAIL=0; `bc766b9a`
- **Sprint 152 complete** (2026-03-14): Block BN — XOR self-recovery: (x^C)^x→C, x^(x^C)→C, (x^C1)^(x^C2)→C1^C2; T578-T583; gate 9/15 PASS, FAIL=0 (pre-implemented by linter); `f45b6898`
- **Sprint 151 complete** (2026-03-14): Block BM — XOR constant-chain fold: (x^C1)^C2→x^(C1^C2); T572-T577; total=577; new bm_xor_valid/src/cval arrays; gate 9/15 PASS, FAIL=0; `696f7a0b`
- **Sprint 150 complete** (2026-03-14): Block BL — shift-low-mask annihilation: (x<<A)&M→0 when M<2^A; T566-T571; total=571; reuses bk_shl arrays; gate 9/15 PASS, FAIL=0
- **Sprint 149 complete** (2026-03-14): Block BK — consecutive shift fold: (x<<A)<<B→x<<(A+B), (x>>A)>>B→x>>(A+B); T560-T565; total=565; new bk_shl/bk_shr arrays; guard A+B<64; gate 9/15 PASS, FAIL=0; `d682ac53`
- **Sprint 148 complete** (2026-03-14): Block BJ — double-NOT elimination: ~(~x)→IrCopy(x); T554-T559; involution law; reuses is_bnot/bnot_src; gate 9/15 PASS, FAIL=0; `736b6146`
- **Sprints 146-147 complete** (2026-03-14): bitwise absorption chain; total=553
  - Block BH (146): (x&C1)|C2→C2 when C1⊆C2; subset mask absorbed; T542-T547
  - Block BI (147): (x|C1)&C2→x&C2 when C1&C2==0; disjoint OR-mask clear; T548-T553
  - Gates: 9/15 PASS, 0 FAIL each; `9b00f7b3`
- **Sprints 143-145 complete** (2026-03-14): two's complement + add-complement + OR-chain; total=541
  - Block BE (143): ~x+1→neg(x); T524-T529; `3eb900b7`
  - Block BF (144): x+~x→-1, ~x+x→-1; T530-T535; `f5fb5d91`
  - Block BG (145): (x|C1)|C2→x|(C1|C2); T536-T541; `1cca4cc2`
- **Sprint 133 complete** (2026-03-14): prebuild wide native driver — staged ELF gate; `f53b5c37`
- **Sprint 142 complete** (2026-03-14): Block BD — const-offset difference collapse: (x+C1)-(x+C2)→C1-C2; T518-T523; total=523; reuses as_valid/as_var_src/as_const_v/as_is_add from Block P; SOTA: LLVM InstCombineAddSub; `d6d61719`
- **Sprint 141 complete** (2026-03-14): Block BC — neg-sub-neg: neg(a)-neg(b)→b-a; T512-T517; `1fbbb31e`
- **Sprint 140 complete** (2026-03-14): Block BB — constant mask merge: (x&C1)|(x&C2)→IrCopy when C2⊆C1; T506-T511; 8/14 PASS, FAIL=0; SOTA: LLVM InstCombineAndOrXor; Boolean distributive axiom
- **Sprint 139 complete** (2026-03-14): Block BA — sub-add inverse: (a-b)+b→a, b+(a-b)→a, a-(a-b)→b; T500-T505; SOTA: LLVM InstCombineAddSub
- **Sprint 136 complete** (2026-03-14): Block AX — additive inverse via negation: a+neg(a)→0, neg(a)+a→0; T494-T499; total=505; 8/14 PASS, FAIL=0; SOTA: LLVM InstCombineAddSub; group theory a+(-a)=0
- **Sprints 133-135 complete** (2026-03-14): XOR cancel + distributive + neg mul
  - Block AU (133): (a^b)^(a^c)→b^c — XOR operand cancellation via GF(2) group law; T482-T487
  - Block AV (134): distributive factoring skeleton (a*b+a*b→2*(a*b)); general case deferred
  - Block AW (135): neg(a)*neg(b)→a*b, neg(a)/neg(b)→a/b — ring theory; T488-T493
  - Gate: 11/23 PASS, FAIL=0, NOT_RUN=12 (JIT OOM); total=493
- **Sprints 131-132 complete** (2026-03-13): XOR self-inverse + NOT-distributive
  - Block AS (131): XOR self-inverse — (a^b)^b→a, (a^b)^a→b; as_xor_rr_valid/lhs/rhs arrays; commutative variants; T470-T475; SOTA: GF(2) self-inverse; LLVM InstCombineAndOrXor; Hacker's Delight §2-3
  - Block AT (132): NOT-distributive — ~a&(a|b)→~a&b, ~a|(a&b)→~a|b; cross-references is_bnot + ao_or_rr/ao_and_rr; T476-T481; SOTA: Boolean algebra distributive + complement; LLVM SimplifyDemandedBits
  - Gate: 15/27 PASS, FAIL=0, NOT_RUN=12 (JIT OOM); total=481; commit e56d9f4f
- **Sprints 127-129 complete** (2026-03-13): Boolean algebra + algebraic cancellation — three blocks in one commit
  - Block AO (127): Absorption laws — a|(a&X)→a, a&(a|X)→a; constant-operand (cross-ref and_valid/am_or_valid) + RR (ao_and_rr/ao_or_rr arrays); T428-T433; SOTA: Huntington 1904; LLVM InstCombineAndOrXor
  - Block AP (128): Idempotent subsumption — (a|b)|a→(a|b), (a&b)&a→(a&b); T434-T439; SOTA: Boolean algebra idempotent law; Hacker's Delight §2-1
  - Block AQ (129): Register-register inverse cancellation — (a+b)-b→a, (a*b)/b→a; aq_add_rr/aq_mul_rr arrays; T440-T445; SOTA: LLVM InstCombine; group-theoretic inverse
  - Gate: 11/30 PASS, FAIL=0, NOT_RUN=19 (JIT OOM); total=445; commit a50d30ea
- **Sprint 115 complete** (2026-03-13): Block AL — subtract reversal through negation: neg(a-b)→b-a, 0-(a-b)→b-a; al_is_sub/al_sub_lhs/al_sub_rhs tracking; 16/16 PASS, FAIL=0, NOT_RUN=6 (OOM); T394-T399; total=397; SOTA: LLVM InstCombineAddSub
- **Sprint 114 complete** (2026-03-13): Block AK — redundant AND superset mask: (x&C1)&C2 where (C1&C2)==C1→IrCopy; 17/17 PASS, FAIL=0, NOT_RUN=6 (OOM); T388-T393; total=391; SOTA: LLVM InstCombineAndOrXor
- **Sprints 101-106 complete** (2026-03-12): Blocks Y-AD — six optimizer passes in one commit; all 6 gates FAIL=0, NOT_RUN=1 (OOM); total=347; commit 05698d49
  - Block Y (101): Boolean simplification — (x==0)==0→IrCopy(x), bool!=0→IrCopy(bool); T330-T335
  - Block Z (102): Non-pow-2 multiply SR — x*3→(x+x)+x; T324-T329
  - Block AA (103): LICM — back-edge hoist IrLoadImm; T318-T323
  - Block AB (104): GVN — 128-entry hash table, VN assignment; T336-T341
  - Block AC (105): Load sinking — bubble-sort IrLoadImm past independent instrs; T312-T317
  - Block AD (106): Jump-to-return — IrJump(L) where L+1=IrReturn→IrReturn; T306-T311
- **Sprint 88 complete** (2026-03-12): Block O — mul-div cancellation ((x*C)/C → x) — 22/23 PASS, FAIL=0, NOT_RUN=1 (self-test OOM); T246-T251; total=251; SOTA: LLVM InstCombineMulDivRem; commit 51eacfdd
- **Sprint 87 complete** (2026-03-12): Block N — IrLoadImm LVN deduplication + native-v2 f64 scalar — 26/27 PASS (gate); T236-T245; total=251; commit d4ad2f5f + ae11eb04
- **Sprint 86 complete** (2026-03-12): Block M — negation arithmetic (x*(-1)→0-x, double-neg→IrCopy) + pthread mutex in stdlib/sync — 19/20 PASS; T230-T235; total=251; commit 05bf91fc + 1c3cd2b7
- **Sprint 85 complete** (2026-03-12): wire orphaned tests T226-T229 — commit 53287fca
- **Sprint 84 complete** (2026-03-11): Block L — associative constant-chain folding — 12/12 PERFECT; (x+3)+5→x+8; commit b22f79a5
- **JIT memory explosion** (2026-03-12): `$SOUC run self-hosted/compiler/main.sio -- --native-compile` grows to 14-35GB RSS within 60-120s due to Cranelift JIT compiling hundreds of self-hosted compiler functions. OOM-killed on 47GB system when other Claude agents spawn competing souc processes. The streaming compilation path WORKS (per-function v2 codegen completes for all functions) but ELF finalization is OOM-killed before producing binary. `--check` and `--ir-dump` work (<5s). Sprint 58 gate: 9/17 PASS (preconditions + reference_ppm), 8 FAIL (all from OOM). Root cause is Cranelift JIT runtime memory management, not fixable from Sounio source.
- **Sprint 70 complete** (2026-03-10): staged IR lowering foundation gate — 18/20 PASS, FAIL=0, NOT_RUN=2 (OOM heavy probes); commit 2d3ca57f; gate script needs `set -eo` (not `-euo`) + `|| _ec=$?` pattern in all check functions to survive OOM kills
- **Sprint 58 updated** (2026-03-12): gate now 9/17 PASS, 8 FAIL (JIT OOM); fixed IrInstr[256]→[128] mismatch in main.sio; gate accepts streaming-direct pipeline; `run_selfhost_fresh.sh` wrapper added; streaming v2 codegen path correct but JIT memory blocks completion
- **Sprint 57 complete** (2026-03-10): IR fidelity gate — 7/10 PASS, FAIL=0, NOT_RUN=3; all 7 ir-dump function counts match fixture; roundtrip deferred
- **Sprint 56 complete** (2026-03-10): frontend corpus gate — 6/10 PASS, FAIL=0, NOT_RUN=4; E017 root cause: Pratt suffix loop parsed cross-line `v\n(` as call; fix: had_newline_before() uses Token.line comparison (self-hosted lexer discards Newline tokens)
- **Sprint 77 complete** (2026-03-11): commutative CSE canonicalization Block H — 11/11 PERFECT; ocp_is_commutative_op helper (Add/Mul/Or/And/BitXor); swap src1↔src2 if src1>src2 before table lookup; T181–T186; total=186; SOTA: Click PLDI 1995 §3.1; NOTE: linter corrected OpXor→OpBitXor; commit ceab5028
- **Sprint 76 complete** (2026-03-11): shift identity/annihilator folding Block G — 12/12 PERFECT; x<<0→IrCopy, x>>0→IrCopy (shift-by-zero); 0<<n→IrLoadImm(0), 0>>n→IrLoadImm(0) (zero-annihilator); T175–T180; total=180; Block E→G chain (T180); SOTA: Cooper & Torczon §8.1; commit 987dd11f
- **Sprint 75 complete** (2026-03-11): conditional branch folding Block F — 12/12 PERFECT (0 NOT_RUN); IrBranchTrue/False with known-const cond → IrJump (taken) or IrNop (not taken); T169–T174 (total=174); SOTA: Wegman & Zadeck TOPLAS 1991 §3; Block E→F chain: x==x→1 (Block E) then BranchTrue(1)→IrJump (Block F); commit bac3f7bb; NOTE: --probe-load-ir probe is pre-existing broken (exits 0 without fn= output), removed from gate
- **Sprint 74 complete** (2026-03-11): comparison folding gate + e-graph activation expansion — 29/29 PERFECT; Block E (linter pre-built): OpEq/Ne/Lt/Le/Gt/Ge both-const+same-reg fold in ocp_const_fold; Block F (new): mul-annihilation (x*0→0, 0*x→0) in eg_small_saturate + 2 egraph.sio tests (total 60); mini-pass guard expanded >4→>8 instrs; T157-T165 (9 new tests); total=174 (linter pre-added Sprint 75 Block F); commit aa6bb6ff
- **Sprint 73 complete** (2026-03-11): Machine IR probe stability gate — 7/14 PASS, FAIL=0, NOT_RUN=7 (OOM under JIT); commit 4db82e99
- **Sprint 73 (in .sio)**: CSE + e-graph mini activation — ocp_cse (128-entry table, EG_OP_* key), EgSmallContext (~7KB, 64 nodes), ocp_egraph_mini_pass (activates for instr_count ≤ 4); T131-T145; commit by linter
- **Sprint 72 complete** (2026-03-11): e-graph equality saturation foundation + Machine IR probe — 39/39 PERFECT; `module ir::egraph` formalized; ocp_binop_to_eg_op bridge + ocp_egraph_pass wired into opt_cleanup; Sprint 72 Block C (bitwise identity/annihilator) + Block D (bitwise idempotency x&x→copy) added by linter; T117-T130 self-tests (130 total); --probe-machine-ir probe; commit 5909c393
- **Sprint 72 complete** (2026-03-11): E-graph foundation + bitwise folding — 17/17 PERFECT; egraph.sio (EgNode/EgClass/EgUnionFind/rules/saturation), ocp_egraph_pass (deferred JIT guard), ocp_binop_to_eg_op; Block C: x&0→0 x&(-1)→x x|0→x x|(-1)→-1 x^0→x; Block D: x&x→IrCopy x|x→IrCopy; T117-T130; 130/130; commit 31d77a25
- **Sprint 71 complete** (2026-03-10): algebraic normalization — 13/13 PERFECT; Block A: x-x→0, x^x→0 (same-register fold); Block B: 4*x→x<<2 (left-constant commutative SR); T109-T116; self-tests 116/116; commit ea72a881
- **Sprint 70 complete** (2026-03-10): staged IR lowering foundation — 17/20 pass (3 not_run: probe/native OOM timeout), 0 fail; two-phase preseed→lower_bodies pipeline; LoadMultimoduleIrTrace; 5-fixture corpus; commit b718811c
- **Sprint 69 complete** (2026-03-10): constant propagation through IrCopy in ocp_const_fold — 23/23 PERFECT; T103-T108; def_at inherited through IrCopy enables SR; linter adds T109/T110 automatically (runner total=110)
- **Sprint 68 complete** (2026-03-10): IR strength reduction for pow-2 mul/div/rem — 32/32 PERFECT; T95-T102; x*2^n→x<<n, x/2^n→x>>n, x%2^n→x&(2^n-1); def_at[256] tracks IrLoadImm sites for in-place patching
- **Sprint 67 complete** (2026-03-10): partial-constant identity/annihilation folding in opt_cleanup — 30/30 PERFECT; T86-T92; x*1→IrCopy, x*0→0, x+0→IrCopy, x-0→IrCopy, x/1→IrCopy; DCE cleans dead IrLoadImm
- **Sprint 66 complete** (2026-03-10): sounio-native-codegen skill (Program D) + model-dispatch update — 28/28 PERFECT; 16 skills validated
- **Sprint 65 complete** (2026-03-10): peephole optimizer wiring — 30/30 PERFECT; IrCopy(r,r)→IrNop + duplicate IrLoadImm elimination; ph_run_on_func after regalloc; self-tests T74-T80
- **Sprint 64 complete** (2026-03-10): compact frame sizing (post-regalloc min frame) — 25/25 PERFECT; frame_size=0 for ≤4 live values
- **Sprint 63 complete** (2026-03-10): adaptive disp8/disp32 stack slot encoding — 30/30 PERFECT; saves 3 bytes/access for vregs 0-15
- **Sprint 62 complete** (2026-03-10): 4-preg linear-scan regalloc (r15/r14/r13/r12) — 35/35 PERFECT; self-tests 48/48; fixes Sprint 52 alignment bug
- **Sprint 54 complete** (2026-03-10): wire tailcall.sio TCO pass — 35/35 PERFECT (0 fail, 0 not_run); self-tests 40/40
- **Sprint 52 complete** (2026-03-10): linear-scan register allocation wiring — 33/33 PERFECT (0 fail, 0 not_run)
- **Sprint 51 complete** (2026-03-10): post-inlining const_fold + DCE cleanup — 28/28 PERFECT (0 fail, 0 not_run)
- **Sprint 50 complete** (2026-03-10): profile-guided function layout — 26/26 PERFECT (0 fail, 0 not_run)
- **Sprint 49 complete** (2026-03-10): profile-guided inlining — 26/26 PERFECT (0 fail, 0 not_run)
- **Sprint 48 complete** (2026-03-10): --show-profile probe + promotion logging + sprof_promotion_target helper — 25/25 PERFECT (0 fail, 0 not_run)
- **Sprint 47 complete** (2026-03-09): .sprof file output + profile reader + strategy promotion — 32/32 PERFECT (0 fail, 0 not_run)
- **Sprint 46 complete** (2026-03-09): Counter dump at exit via __prof_dump — 24/24 (22 pass, 0 fail, 2 not_run)
- **Sprint 45 complete** (2026-03-09): INC [RIP+disp32] backfill + .data section — 22/22 (20 pass, 0 fail, 2 not_run)
- **Sprint 44 complete** (2026-03-09): IrProfCounter opcode + slow-path counter injection — 23/23 PERFECT (0 fail, 0 not_run)
- **Sprint 43 complete** (2026-03-09): __validated runtime propagation through call chain — 23/23 PERFECT (0 fail, 0 not_run)
- **Sprint 42 complete** (2026-03-09): __validated param threading — 28/28 PERFECT (0 fail, 0 not_run)
- **Sprint 41 complete** (2026-03-09): merge block fix + probe validation — 29/29
- **Sprint 40 complete** (2026-03-09): native dual-path for Instrumented strategy — 44/50 PASS (0 fail)
- **Sprint 39 complete** (2026-03-09): Table 1 for POPL §Implementation — 36/36
- **Sprint 38 complete** (2026-03-09): codegen gap closed at IR level — 35/35
- Sprints 32-37: committed to main
- Sprints 38-47: committed + tagged v0.81.0-sprint47, pushed
- Sprint 48: committed + tagged v0.82.0-sprint48, pushed
- Sprints 49-51: committed + tagged v0.84.0-sprint50 / v0.85.0-sprint51, pushed

## Architecture

Native backend pipeline: **AST → ir/lower.sio → IrModule → native/codegen.sio → x86-64**
(NOT through HLIR — HLIR is only for LLVM/GPU backends)

Strategy flow (Sprint 38): `AST return type+effects → ir_compute_strategy_from_ast() → IrFunction.compile_strategy → NativeCompiler.current_strategy`

## Key Files

- `self-hosted/compiler/main.sio` — self-hosted compiler entry point
- `self-hosted/ir/ir.sio` — IrFunction struct (has compile_strategy: i64)
- `self-hosted/ir/lower.sio` — AST→IR lowering (computes strategy)
- `self-hosted/native/codegen.sio` — compile_ir_function (reads strategy)
- `self-hosted/native/frame.sio` — NativeCompiler struct (current_strategy)
- `self-hosted/native/lower_ir.sio` — instruction lowering (strategy-aware)
- `self-hosted/hlir/ir.sio` — HLIR with CompileStrategy enum (separate path)

## Gate Scripts

```bash
bash scripts/sprint52_regalloc_gate.sh         # 33/33 PERFECT
bash scripts/sprint51_opt_cleanup_gate.sh      # 28/28 PERFECT
bash scripts/sprint50_layout_pgo_gate.sh       # 26/26 PERFECT
bash scripts/sprint49_inline_pgo_gate.sh       # 26/26 PERFECT
bash scripts/sprint48_show_profile_gate.sh     # 25/25 PERFECT
bash scripts/sprint47_sprof_gate.sh            # 32/32 PERFECT
bash scripts/sprint46_prof_dump_gate.sh        # 24/24 (22 pass, 2 not_run)
bash scripts/sprint45_data_section_gate.sh     # 22/22 (20 pass, 2 not_run)
bash scripts/sprint44_prof_counter_gate.sh     # 22/23 (nop_7byte expected FAIL since Sprint 45 replaced NOP with INC)
bash scripts/sprint43_chain_propagation_gate.sh # 23/23 PERFECT
bash scripts/sprint42_validated_param_gate.sh  # 28/28 PERFECT
bash scripts/sprint41_merge_block_probe_gate.sh # 29/29
bash scripts/sprint40_dual_path_native_gate.sh  # 44/50 (0 fail)
bash scripts/sprint39_strategy_impact_gate.sh   # 36/36
bash scripts/sprint38_strategy_codegen_gate.sh  # 35/35
bash scripts/sprint_embed_abi_gate.sh            # 8/8 (32/32 smoke)
```

## Key Files (Sprints 40-42)

- `self-hosted/ir/opt_strategy.sio` — IR-level dual-path: ir_opt_apply_strategy, ir_opt_clone_instr, ir_opt_find_return_vreg
- `self-hosted/native/codegen.sio` — wires effective_func when strategy=INSTRUMENTED
- `self-hosted/ir/lower.sio` — Sprint 42: ir_lower_inject_validated_param() + call-site prepend

## E017 Bug Workaround (Sprint 56)

JIT misreads ExprKind for ExprIndex nodes: second+ occurrence of `(*ref)[idx]` in the same function body dispatches to `check_call_expr` instead of `check_index_expr`. **Fix**: ensure at most ONE `(*ref)[idx]` expression per function body:
- `fill_channel(x0, y0, rw, rh, v: i64, fb: &![i64; 65536], fb_w, fb_h)` — one `(*fb)[idx] = v`
- `get_channel(fb: &[i64; 65536], i: i64) -> i64 with Panic` — one `(*fb)[i]`
- Use these helpers in `rasterize_rect` and `emit_ppm` to avoid multiple deref-index exprs

## Notes

- Probe checks (--probe-load-ir) are slow; sequential runs may timeout at 180s → NOT_RUN (not FAIL)
- Sprint40 gate has --probe-ir-opt-strategy and -t native checks that are NOT_RUN (new infra, not yet in souc binary)
- Sprint 41 fixed bug: merge_block hardcoded ir_return(0); now uses ir_opt_find_return_vreg(func)
- Sprint 42 fixed bug: dispatch block read param_regs[0]=IR_INVALID_REG; now injects real __validated vreg
- Sprint 43: instrumented→instrumented calls forward param_regs[0] directly (no load); non-instrumented inject constant 1; post-lowering patch pass (ir_patch_validated_calls) handles forward references
- Sprint 44: IrProfCounter opcode + IrProfCounterInfo struct; counter injected at instrumented-path entry by ir_opt_apply_strategy(); self-tests T12-T13
- Sprint 45: Backfilled 7-byte NOP with INC QWORD PTR [RIP+disp32] (48 FF 05 <disp32>); writable .data section in ELF (SHF_WRITE); reloc kind_code=3 for .data; counter storage pre-allocated in compile_module(); self-tests T14-T15 (INC encoding + reloc kind_code=3)
- Sprint 46: __prof_dump function emitted in code buffer; called from exit trampoline before sys_exit; prefix/colon/newline via immediate stack MOV (zero extra relocs); fn names from .rodata LEA+strlen; counters from .data MOV [RIP+disp32]; itoa div-by-10; all to stderr fd=2; RelocationTable 64→256; self-tests T16-T17
- Sprint 47: .sprof file output in __prof_dump (sys_open/write/close to "profile.sprof"); new ir/profile.sio with SprofProfile+sprof_parse+sprof_lookup+sprof_apply_promotion; --use-profile CLI in main.sio; count>1000→AGGRESSIVE, >100→PRECISION; self-tests T18-T19 (sprof_parse, sprof_lookup); emit_mov_rdi_r12/push_r12/pop_r12 in encode.sio; emit_prof_write_sprof_header/emit_inline_itoa_fd/emit_prof_write_space_to_fd/emit_prof_write_newline_to_fd in codegen.sio
- Self-test binary previously had ProvenanceKind resolution errors; Sprint 49 self-tests (T22-T23) now pass at runtime
- Flat namespace: `use ir::profile::*` (wildcard) required; selective imports cause "Undefined variable" for internal helpers
- Module system: files with `module X` are proper modules needing explicit `use` imports; files without are flat-namespace (normalize.sio, const_prop.sio, dce.sio, ssa.sio)

## Key Files (Sprint 47)

- `self-hosted/ir/profile.sio` — SprofProfile reader + strategy promotion
- `self-hosted/native/codegen.sio` — .sprof file output in emit_prof_dump_function
- `self-hosted/native/encode.sio` — r12 register helpers (mov_rdi_r12, push_r12, pop_r12)

## Sprint 48 Details

- `sprof_promotion_target(count) -> i64` factored out of `sprof_apply_promotion`; returns strategy constant or -1
- `run_probe_show_profile` — reads .sprof, prints per-entry promotion decisions with `[sprof]` prefix
- `--show-profile <file>` dispatch in main() before `--probe` catch-all
- Promotion logging in `compile()` after `sprof_apply_promotion` — prints `[sprof] fn: count N -> strategy X`
- Self-tests T20-T21: threshold verification (1500→AGGRESSIVE, 500→PRECISION, 50→-1, boundaries 1000/100)
- Git config set: Demetrios Chiuratto Agourakis <demetrios@agourakis.med.br> (repo-local)

## Sprint 49 Details

- `module ir::inline` + `use ir::ir::*` + `use parser::ast::{Name, empty_name}` added to inline.sio
- Local `opcodes_equal` + `inl_matches_opcode` in inline.sio (avoids flat-namespace dep on normalize.sio)
- `InlFuncInfo.compile_strategy` field added; populated from `IrFunction.compile_strategy` in `inl_analyze_function`
- `INL_STRATEGY_AGGRESSIVE_BONUS=80`, `INL_STRATEGY_PRECISION_BONUS=40` constants
- Strategy-aware scoring in `inl_compute_benefit`: callee strategy 1→+80, strategy 2→+40
- `inl_run_pass(module)` wired into compile() when `opts.optimize` is true (after profile promotion)
- `use ir::inline::*` import in main.sio
- Self-tests T22-T23: strategy boost verification + pass returns valid module
- Key lesson: files without `module` declaration (normalize.sio, const_prop.sio) are in flat namespace; files with `module` must explicitly import all deps

## Sprint 50 Details

- New `self-hosted/ir/layout.sio` — `module ir::layout` with LayoutScore, LayoutResult structs
- `layout_score_functions` scores each fn via `sprof_lookup`; `layout_sort_scores` bubble sort descending by count (unknown -1 sorts last, stable by func_idx)
- `layout_is_reordered` detects if any function moved; `layout_build_remap` builds old→new index table
- `layout_reorder_functions` physically rearranges `module.functions[]`; `layout_patch_fn_ids` fixes all `IrCall.fn_id` references
- `layout_sort_by_profile` orchestrates full pass: score→sort→reorder→remap→patch→count hot/cold
- Pipeline wiring: `var has_valid_profile` + `var loaded_profile` hoisted above profile block; layout runs after inlining, before `compile_multimodule_native_with_ir`
- `[layout]` logging prints function count + hot/cold breakdown when reorder occurs
- Self-tests T24-T25: hot-first reorder verification + empty-profile preserves order
- Full PGO pipeline now: instrument → profile → promote → inline → layout → codegen

## Sprint 51 Details

- New `self-hosted/ir/opt_cleanup.sio` — `module ir::opt_cleanup` + `use parser::ast::*`
- `ocp_const_fold`: tracks IrLoadImm values in `[bool; 256]` + `[i64; 256]` maps; folds IrBinOp with known-constant src regs to IrLoadImm; if-else chains for OpAdd/Sub/Mul/Div/Rem
- `ocp_dce`: mark-sweep — `ocp_mark_used` builds used[reg] table; `ocp_sweep` NOP's instructions with unused dst and no side effects
- `opt_cleanup_function(func)` → `opt_cleanup_module(module)` orchestrate both passes
- Pipeline: runs after layout_sort_by_profile, if opts.optimize; logs `[opt] const_prop + dce applied to N functions`
- Self-tests T26-T27: T26 verifies 3+4 folds to 7; T27 verifies dead r1=99 becomes IrNop
- **Key lesson**: `use parser::ast::{BinaryOp}` (selective import) breaks BinaryOp `==` comparison; must use `use parser::ast::*` (wildcard)

## Sprint 52 Details

- `module native::regalloc` added to regalloc.sio (was flat-namespace, now importable)
- `NativeCompiler`: `vreg_to_preg: [i64; 512]` + `vreg_spill_slots: [i64; 512]` (RA_UNASSIGNED=-2, RA_SPILLED=-1, >=0=x86reg)
- `emit_prologue`: push r15 before push rbp; `emit_epilogue`: pop r15 after pop rbp
- `nc_run_regalloc(nc, func)` in lower_ir.sio: IR→RaSimpleInstr → liveness → linear-scan (1 preg=r15 x86reg15)
- `nc_load_vreg_to_rax` / `nc_store_rax_to_vreg`: check preg, emit MOV rax,r15 or fallback stack-slot
- encode.sio: emit_push_r15, emit_pop_r15, emit_mov_rax_r15, emit_mov_r15_rax, emit_mov_rax_preg, emit_mov_preg_rax
- Param vregs always unallocated (keep stack-slot path)
- T28/T29: 29/29 PERFECT; gate: 33/33; commit 1b180279

## Program C complete (2026-03-10): Skills Platform Expansion — Sprints 59-61

- **Sprint 59** (27/27): skills/sounio-render + sounio-bootstrap + sounio-pgo created with YAML frontmatter + Workflow + references
- **Sprint 60** (19/19): model-dispatch SKILL.md updated (7 routing rows + 4 flowchart entries + frontmatter added), cross-skill-index.md created
- **Sprint 61** (9/9): validate_skills_coverage.sh + sprint61 gate — 15/15 skills valid
- Skills total: 15 (was 12). Programs A+B (Sprints 53-58) assigned to Codex.

## Next

Programs A (Render Platform, Sprints 53-55) and B (Self-hosted Frontend Bootstrap, Sprints 56-58) — Codex implements.

## User Preferences

- No "Co-Authored-By" in commits (CLAUDE.md)
- Gate artifacts must have status=pass, metrics with total/passed/failed/not_run
- Gate scripts use `check_grep` pattern with `grep -qE`
- Sounio syntax: `&!` not `&mut`, `var` not `let mut`, no `pub`
