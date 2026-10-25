<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-178-x86.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-178-x86.readme
-->

# Test-only x86 directed primitive witness against MPFR

This is a C++23 test/reference component for D-34, based on the published finite
rational proof branch c1eb0bbb918d39cde71cdbdc05a4795161228dba, whose ancestry
contains remote main7ce5eddeef78c76c7e744150ed9c169c53a3d3b0.
It implements no Sounio type, IR opcode or language backend and provides no
interval/hardware certification label. The existing finite Rat proofs are
unchanged and do not prove these floating-point instructions correct.

## Actual executed scope

Slurm17447/source05 completes STATUS0 on the allocated cpu-ops Xeon Gold6526Y
worker. The C++ program actually checks1000000 deterministic finite-input cases
for each of add, subtract, multiply and divide in both binary32 and binary64.
Every case executes both the scalar XMM and ZMM-lane implementation and compares
both directed endpoint bit patterns to MPFR. The checked loop reports:

```
ER_TOTAL indexed_cases=8000000 scalar_cases=8000000 zmm_lanes=8000000
```

The loop counts indexed successful comparisons, not distinct input pairs. Edge
rewrites and deterministic generation may revisit a pair; no uniqueness or
deduplication claim is made. It does not infer unseen cases from a
requested count. Edge pairs precede seeded SplitMix64 raw-bit input generation.
They include signed zeros, cancellation, adjacent values around one, halfway
rounding, smallest and largest subnormal values, smallest normal values, largest
finite values, directed overflow and division by signed zero with a nonzero
numerator. Random exponent-all-ones encodings are deterministically mapped into
the finite domain. Indeterminate0/0 pairs are replaced in the main finite
arithmetic corpus and tested separately as refusals. This is a deterministic
differential test, not an exhaustive proof or a statistical coverage estimate.

Finite input values can produce infinite directed outputs. NaN and infinity
INPUTS are outside this witness's arithmetic domain and refuse before emission;
D-34 set-based handling of unbounded interval operands is still required.

## Independent oracle and target instructions

The final oracle uses MPFR4.2.1 with precision24/53, the binary32/binary64 target
exponent ranges and the operation's ternary result passed to mpfr_subnormalize.
This follows the official [MPFR subnormal-emulation API and binary64 example](https://www.mpfr.org/mpfr-4.2.1/mpfr.html#Exception-Related-Functions).
The final conversion is exact because the emulated result is already a target
value. Oracle exponent limits and flags are restored after each scoped object.

The previous source02 wide-precision directed oracle also passes all cases in
Slurm17431. Source03/Slurm17435 switches to the explicit IEEE-emulation oracle;
all eight per-format/operation result digests match the previous run exactly.
Source04 adds an actual comparison counter and symbolic intrinsic rounding
constants in17440. Independent review identified the word unique as inaccurate:
this counter counts indexed visits, not distinct input pairs. Source05 corrects
that label and requires both named runtime library hashes explicitly; final
17447 passes with unchanged arithmetic. None of those historical
packets are overwritten or relabeled as the final source.

Scalar and vector leaves use GCC AVX512F intrinsic definitions with immediate
round-down/round-up plus suppress-all-exceptions. Their actual disassembly must
contain vadd/vsub/vmul/vdiv ss/sd/ps/pd with both rd-sae and ru-sae. It must contain
no YMM embedded-rounding form and no FMA instruction. The official
[Intel compiler intrinsic reference](https://www.intel.com/content/dam/develop/external/us/en/documents/cpp_compiler_classic.pdf)
and the actual GCC avx512fintrin.h implementation are reference material; the
receipt records the compiler binary, actual emitted instructions and runtime
libraries separately.

Each guarded scalar/vector call requires MXCSR exactly0x1F80 and preserves it.
The test driver resets the environment around the independent MPFR calculation;
the tested guarded kernel refuses an unexpected environment rather than repairing
it. Five altered environments cover FTZ, DAZ, rounding mode, exception masks and
sticky flags. Both scalar and ZMM refusals preserve every output cell and the
caller's bad environment. NaN/infinity and0/0 refusals are also checked for both
widths, including a bad last lane to test atomic output preservation.

These observations do not establish Sounio optimizer barriers, implicit-FMA
policy in the compiler, decimal literal conversion, interval decorations, empty
sets, unbounded operand set semantics, explicit interval FMA, or the RISC-V/RVV
and PTX links. They are groundwork for those separate implementation gates.

## Nonvacuous controls and resources

Four actual counterfactual executions are required to fail with exit8 and their
specific diagnostic: nearest rounding substituted for directed rounding,
rounding directions swapped, subnormal results flushed to zero, and environment
guard bypassed. The first three fail at a numerical endpoint disagreement;
the last fails at the required environment refusal. The ordinary gate finishes0.

The final million-case process uses1.53s wall,1.52s user time and4200KiB maximum RSS in
this one observation. This is not a comparative performance benchmark or a claim
about a future Sounio backend. The test streams comparisons and stores compact
counters/digests, not millions of duplicate case records. The FNV64 digest is a
reproducibility fingerprint, not a cryptographic identity or the acceptance test;
each endpoint is directly checked before it contributes to that fingerprint.

MPFR/GMP headers were extracted into the task scratch from Ubuntu packages
matching the package-cache SHA256 values below. No global package installation,
quota change, old job modification or historical artifact removal occurred.
The runtime libraries were already installed and are hashed in the actual gate.

| Isolated header package | Expected and observed SHA256 |
|---|---|
| libmpfr-dev4.2.1-1build1.1 |362fcdeb7dff545af83e65a3a972519185efd4a231c23e13d519cf97a5c8e3e4|
| libgmp-dev6.3.0+dfsg-2ubuntu6.1 |a9847b5ecfff791a46cb198f29a06d9d23a20afdfdd434812f2b44ccfb61d46a|

To reproduce on a qualified x86 Linux CPU, allocate one Slurm CPU and provide
identified MPFR/GMP headers to the portable gate:

```
ER_MPFR_INCLUDE=/absolute/extracted/usr/include \
ER_GMP_INCLUDE=/absolute/extracted/usr/include/x86_64-linux-gnu \
bash scripts/stdlib/interval_x86_er_gate.sh /absolute/fresh-output
```

Unsupported AVX512F CPUs refuse the executable with77; that cannot satisfy this
gate. A successful compilation or a skipped target never counts as the million
case execution. The final source/hash/library/assembly/counterfactual receipts
live at /scratch/sounio-stdlib-inference-scale-20261024/SIO178-x86/result05.
No hosted CI or cross-target result is implied by this allocated-worker run.

## Compact custody

`evex-mpfr-text.tar.gz` is313443 bytes with SHA256
`8f053a701841c7f62c3f0abff0882ee40c066d9ee97afb9ea92e6dad4e6cf10b`.
It retains the scoped run outputs, compiler/oracle/header/source identities,
actual disassembly, controls, resource observations, before/final source variants,
launchers and an independently checked MANIFEST.sha256. No compiler, test ELF,
shared library, development-package archive or million-row dataset is shipped.
The original worker packets remain intact. Independent source review checked
oracle configuration, signed-zero/subnormal comparison and refusal staging; it
also prompted the corrected indexed-count wording and explicit library census.
Source review and differential execution are separate evidence categories.
