<!-- docs:meta
topic_id: repo.docs.audit.madaros-memory-footprint-2026-10-22.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-memory-footprint-2026-10-22.readme
-->

# Madaros memory footprint — measured 2026-10-22

**Status: measurement plus one localised mechanism.** Every number below is
re-run by [`measure.sh`](measure.sh) (C++23 tools [`maxrss.cpp`](maxrss.cpp) and
[`bsscensus.cpp`](bsscensus.cpp), per ADR-009); the exact census needs the saved
build log (`BUILD_LOG=`) and the self-build step needs the derived seed
(`SEED_ELF=`). Source tree: `main` at
`27afa340`. Machine: a 2-vCPU, 7 GiB container. Two things were **not**
reproduced here and are recorded as reports, not measurements: the gen3
self-build reaching ~15 GB and dying by OOM, and the claim that gen2 is
bit-identical to gen1 (founder report, 2026-10-22). CI records gen3 failing
differently, with `arena full` and rc 181 (§3).

---

## 1. Measured

### 1.1 Binary layout

| ELF | file bytes | zero bytes | gzip bytes | R+X `LOAD` (filesz) | RW zero-fill `LOAD` (memsz) |
|---|---:|---:|---:|---:|---:|
| Madaros built from source | 129,688,636 | 37,670,445 (29%) | 12,743,665 | 129,688,636 | **3,807,438,624** |
| `bin/madaros-linux-x86_64` (committed) | 125,051,232 | 36,195,810 | 12,396,159 | 125,051,232 | 4,417,323,016 |
| `bin/souc-lean-single-x86_64` (seed) | 11,931,702 | 4,321,986 | 566,195 | 11,931,702 | 696,170,840 |

Both Madaros ELFs are a single R+X segment holding code and constants plus one
zero-fill RW segment. The code segment compresses 10:1, so the 130 MB is mostly
repetition, not information.

### 1.2 Peak resident set (`ru_maxrss`), three runs each

| compiling `fn main() with IO { println("hi") }` | peak RSS | wall |
|---|---:|---:|
| Madaros built from source | 229,616–229,624 KiB | 0.2 s |
| committed Madaros ELF | 1,252,672–1,252,904 KiB (two machines) | 1.0 s |
| seed (lean_single) | 23,944 KiB | 0.08 s |

A static zero-fill reservation costs address space, not resident memory, until
a page is touched; the cgroup that kills a build counts resident memory. So the
3.8 GB reservation is not by itself the OOM. The committed ELF touches 1.25 GB to
compile one line; the source-built one touches 230 MB. The committed ELF was
built from older source, and which of its arrays it touches at start-up was not
traced. Every large array in the current source (≥ 1 Mi entries) has a zero
initialiser — `PT_KIND`'s `[TokenKind::Eof; 2097152]`
(`self-hosted/parser/parser.sio:15`) included, since `Eof` is discriminant 0;
the non-zero initialisers (`[-1; N]`, e.g. `check/check.sio:4894`,
`ir/lower.sio:208–222`) are small — so start-up initialisation does not
explain the source-built 230 MB either.

### 1.3 The self-build

The seed compiling `self-hosted/compiler/main.sio` into Madaros (`scripts/ci/build_modular_madaros.sh`,
130 modules in the import closure, ~298,800 lines by `wc -l`): **peak RSS
695,356–736,268 KiB** over two independent runs on 2 vCPU (one `ru_maxrss`, one
`VmHWM` sampled every 5 s), 12–17 min wall. The output ELF was byte-identical
across the two runs. `measure.sh` step 6 repeats this when given the derived seed.

## 2. Where the 3.8 GB reservation comes from

### 2.1 Module-level arrays explain one third

`bsscensus` lists every `var NAME: [T; N]` in the 130-module closure. The seed's
slot width was measured, not assumed: compiling `var A: [T; 1048576]` grows the
seed's BSS by **1 byte per element for `i8` and 8 bytes for every other type**
(`u8`, `bool`, `i16`, `i32`, `u32`, `f32`, `i64`, `f64`). Under that layout:

| | bytes |
|---|---:|
| 415 module-level arrays | 1,225,713,428 |
| seed baseline (1 MiB aggregate-return scratch and fixed slots) | 1,050,328 |
| **not explained by module-level arrays** | **2,580,674,868** |

Largest contributors: `NC_BIG_ELF` 160 MiB (`native/codegen_x86_linux.sio`),
`NC_BIG_CODE` 128 MiB (`native/encode.sio`), seventeen `IR_A_*` arrays of 4 Mi
entries × 8 bytes, 544 MiB together (`ir/ir.sio`, 580 MiB for the file), the
parser's `PT_*` token arrays of 2 Mi entries (`parser/parser.sio`, 128 MiB).

### 2.2 A mechanism that produces the rest — localised and reproduced

In the seed's x86 backend, an aggregate of **1 MiB or more** passed or returned
by value gets a **fresh BSS block at every site**, never reused:

- `local_bss_spill_bytes() = 1048576` (`self-hosted/compiler/lean_single.sio:1426`);
- a local whose slots reach it spills to `bss_alloc_aligned` (`:3485`);
- a by-value return that reaches it gets its own `sret` block,
  `emit_sret_destination_x86` (`:3075`), one per call site;
- the macOS arm64 path pools statement temporaries per function
  (`a64_stmt_temp_alloc`, `:1478`, keyed on `TARGET_OS == 2`); on every other
  target, x86 and `aarch64-linux` included, it forwards straight to
  `bss_alloc_aligned`.

Reproduced: a 2 MiB struct passed and returned by value, `b = bump(b)` repeated
*n* times, gives seed BSS of 9,438,968 / 13,633,288 / 22,021,928 / 38,799,208
bytes for *n* = 1 / 2 / 4 / 8 — exactly **4,194,320 bytes per call site**: two
blocks of the struct's 2,097,160 bytes (262,145 slots), one `sret` and one spill. Madaros has
structs far above the threshold (e.g. `is_proj: [bool; 1048576]` and
`param_idx: [i64; 1048576]` as fields, `self-hosted/check/check.sio:4489–4490`,
16 MiB under the seed's layout), and passes compiler state by value as a
convention.

**Not yet established:** that this mechanism accounts for the full 2.58 GB. The
attribution needs a seed that reports bytes per `bss_alloc_aligned` caller;
until then 2.2 is a demonstrated mechanism, not a measured share.

**The most likely carrier, from an independent investigation.** PR #3015
(merged 2026-10-22, issue #2966) found the same convention from the stack side:
the IR `Lowerer` struct is about 4.5 MB by value, and `self-hosted/ir/lower.sio`
has 618 methods that take `self` by value (recounted here: 618, of which 125 also
return `Lowerer` by value). At that size every such call site is above the
seed's 1 MiB threshold, so under §2.2 each one reserves its own blocks; a few
hundred sites would make up the 2.58 GB. That is a prediction, not a measurement,
and it has a direct test: rebuilding with `Lowerer` taken as `&!Lowerer` (the
refactor #3015 sizes at weeks of work) should shrink the BSS by roughly the
number of converted sites times the struct's size under the seed's layout.

## 3. The gen3 failure, reasoned from what is known

If gen2 is bit-identical to gen1, building gen3 with gen2 is **the same
computation** as building gen2 with gen1: same binary, same input, same result,
same memory. A gen3 build that needs memory gen2's did not means one of:

1. the source compiled is not the same (more modules, different flags);
2. memory use is not deterministic (ASLR, hash order, parallelism);
3. gen1 and gen2 differ in something the comparison did not look at.

Check those first; they are cheap and may change the whole diagnosis. Scale
gives the size of the anomaly: the closure is ~298,800 lines (12.7 MB). The seed
compiles it at a ~0.7 GB peak, ~2.5 KiB per line (§1.3). A 15 GB peak would be
~50 KiB per line — twenty times what the seed needs for the same source. A
ratio like that, between two compilers of the same source, points at phases
that stay alive after they are consumed rather than at the source's size. The measurement this record
lacks is the one that matters: peak RSS of a Madaros-built Madaros compiling
`main.sio` (the gen2 → gen3 step), which a 7 GiB container cannot host safely.

**What CI has shown, and what it has not.** PR #3015 records that GitHub Actions
starts jobs with a 16 MB soft stack limit, where the Linux default is 8 MB. On
8 MB a Madaros capacity test with 8191 globals dies with SIGSEGV (rc 139); on
16 MB it passes. That is a capacity result. It does **not** show the gen3
self-compile completing, and an earlier version of this paragraph said that it
did (corrected 2026-10-23, after review on #3018). The Madaros Gen2 Recovery Proof
([run 39834416192](https://github.com/Sounio-lang/sounio/actions/runs/39834416192),
job 120019580004, head `aeb5a0081`, 2026-10-22) raises the stack to 512 MiB for
the build and runs the gen3 step with `stack=1572864` (1.5 GiB). It records:

```text
[rung gen2] gen1 compiles self-hosted/compiler/main.sio   rc=0 merged_ir_functions=14867
[rung run]  gen2 identifies itself                        banner=madaros
[rung gen3] gen2 compiles self-hosted/compiler/main.sio   rc=181
MADAROS_FIXED_POINT_OK: reached rung 'run' as recorded; the next wall is 'gen3'
```

The job is green because `run` is the rung it is set to expect. The reason for
rc 181 is in the run's artifact (`fixed-point/gen3.log`):

```text
madaros: arena full (allocation of 2080 bytes)
  the generated program's runtime heap is one fixed ~2 GiB bump arena that is
  never reclaimed: every aggregate value created or copied by value (struct
  literal, by-value array/struct argument or return, ...) takes fresh arena
  space for the life of the process.
```

So **gen3 is unproven**. With a 1.5 GiB stack it fails on the heap, not on the
stack. #2966's "arena full" is therefore not explained by an 8 MB stack. The
message names the same by-value convention as §2.2, seen this time in gen2's
runtime arena instead of in the seed's BSS: each by-value copy of compiler state
takes arena space that is never returned. That connects the two, but it is still
a hypothesis; the direct test is the `&!Lowerer` rebuild described in §2.2. The
workflow also carries a comment stating that a local run of the rungs peaked at
63,570,060 KiB RSS, and it adds a 48 GiB swap file. That figure is recorded here,
not measured. A gen3 failure report still needs its `ulimit -s` and its memory
limit, because the stack decides the capacity tests.

Triage for the gen7 fixed point (the self-compilation map `gen_{n+1} = F(gen_n)`):

1. compile the same generation twice; if it is not identical **to itself**, it
   is non-determinism (hash iteration order, addresses as sort keys, embedded
   timestamps or paths, uninitialised padding);
2. if deterministic but changing between generations, look for **host**
   floating-point inside the compiler, such as constant folding: each
   generation folds constants with the code the previous one generated (D8 of
   the [interval proposal](../../proposals/CERTIFIED_INTERVAL_PRIMITIVE_2026-10-22.md));
3. if it oscillates with period 2, a real miscompile of the compiler — the most
   serious of the three. Wheeler's diverse double-compiling is the independent
   check.

## 4. Remedies — design, not done

1. **A level never points upward.** An arena can die only if nothing below
   holds pointers into it. Lowering builds the next level by copying and
   translating into its own arena; when MIR is built, the HIR arena dies whole.
   Two legitimate long-lived arenas: interned data (strings, symbols, types) and
   provenance — the latter made of **indices, not pointers**, so the level above
   can die and the map stays valid. ADR-010 reclaims regions at function return;
   it does not cover data that escapes from one compiler phase into the next,
   which is exactly the self-build's case. This complements it.
2. **Return memory to the system.** Freeing into an allocator's free list does
   not lower RSS, and RSS is what the cgroup counts. Each phase arena is its own
   `mmap`, released with `munmap` when the phase ends.
3. **Indices, not boxes.** Small `Box` nodes become `u32` indices into vectors:
   4 bytes instead of a malloc header plus an 8-byte pointer. Bonus that protects
   the fixed point: iteration order no longer depends on address values, so a
   change of allocator cannot change the output.
4. **Seed:** pool `sret` and spill blocks per function by size on every
   target, as the macOS arm64 path already pools statement temporaries (§2.2). This changes the
   seed, so it needs the `make build` fixed point re-verified.
5. **Madaros:** replace the fixed static arrays (§2.1) by regions that grow on
   demand, so capacity is not reserved for the largest program ever compiled.
6. **Instrument first:** the compiler reports each arena's size at the end of
   each phase, so the next measurement is per phase rather than per process.

## Claims Forbidden

- Do not cite the 15 GB gen3 figure or gen2 = gen1 identity as measured here.
- Do not say gen3 completes in CI, or that #2966 was a stack artefact: the CI
  receipt is rc 181, `arena full`, with a 1.5 GiB stack (§3).
- Do not say the 3.8 GB reservation causes the OOM; reservation is not residence
  (§1.2).
- Do not attribute the unexplained 2.58 GB to §2.2 as a measured share.
