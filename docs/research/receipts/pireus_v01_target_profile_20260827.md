<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-v01-target-profile-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-v01-target-profile-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus v0.1 Target And Material Profile Receipt

Date: `2026-08-27`

Stage: `SEMANTICS_FROZEN`

Language-Producer: `Sounio`

Language-Role: `SEMANTIC_AUTHORITY`

Semantic-Lane-ID: `pireus-v01-target-profile-20260827`

## Mandatory Order

The acceptance sequence was:

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
```

The Garden file was created before `target_profile.sio` and its executable were
run. The semantic contract and this receipt were written only after a passing
Sounio execution existed. `PARITY_OPEN` and `CLAIM_READY` remain closed.

## Frozen Hashes

| Artifact | SHA-256 |
| --- | --- |
| Garden seed | `9a87359affaa33b8c49848d86ff140a2a9168a4ecbf5d0cb77ac5419cf32b8c2` |
| Concept extension contract | `cfb477b5591fa89fe54aed0dd86dfb0e49a758f56892e707d8d88ca47ba49cb7` |
| Frozen Pireus v0 model | `ee4589ab4dad2a47a136629dcab6e93aa2f215cf114a8e2f7b3f24a89d39ed9d` |
| Pireus v0.1 target-profile model | `d41726a8a7eba62132e3763cf6a71938de746ec9d58ce8a20caa40709546d6a4` |
| Ontology query kernel | `e36f9d7bb4e16dd7c68a69dd51ae5f2db96d9bd8209bf61483c9b3ee88ac8cbb` |
| Sounio executable source | `caa72eb54c10123eacc15fcb78e146e81775afcdb52089a9168153195d63f067` |
| Frozen semantics document | `fc9e2c8c895f3a2a955236a04a2a71abb6a5fd3a70e7c3961ed4a82d093d08da` |
| Extracted Sounio output | `2d9c3fafe0bb7184d300ffbc325ad6932e7d9426f010bd3af49abc40189676c0` |
| Rebuilt ontology wrapper | `ac705d2a14710b7034d08bd742c159acd1129b4f3760d69a522ee05fb7933395` |

## Toolchain

Public compiler resolution during the lane selected:

```text
launcher: /workspace/.wt/codex-3/bin/souc
selected: /workspace/sounio/bin/souc
identity: Madaros v0.80.0 -- the Sounio self-hosted compiler
```

Ontology validation used the rebuilt current-source wrapper:

```text
wrapper: /tmp/pireus-v01-ontology-validation-souc
checker driver: /tmp/sounio-ontology-validation-build/check_driver_probe.elf
driver compiler: /tmp/sounio-ontology-validation-build/lean-gen1.elf
fallback compiler: /workspace/.wt/codex-3/bin/souc
check resolution: unanimous
check provenance: rebuilt_direct
```

The wrapper build and acceptance path invoked no Python, Rust, Node, Ruby,
`awk`, or `bc`. Shell commands transported inputs, launched Sounio, compared
bytes, and calculated hashes; they did not act as semantic oracles.

## Execution Hardware

The Sounio witness executed on:

```text
OS: Linux 7.0.2-5-pve x86_64
CPU: Intel Xeon Gold 6526Y
sockets: 2
cores per socket: 16
threads per core: 2
logical CPUs: 64
```

## Material Input Observation

The five live Slurm worker pods reported these processor identities:

```text
r740-proxmox  Intel Xeon Gold 6148 CPU @ 2.40GHz
dl380-proxmox Intel Xeon Gold 6262V CPU @ 1.90GHz
5860-proxmox  Intel Xeon W3-2423
t560-proxmox  Intel Xeon Gold 6526Y
r770-proxmox  Intel Xeon 6730P
```

The same read recorded topology and selected `/proc/cpuinfo` flags. These are
inputs with evidence role `ObservedKernel`, not Sounio expected results and not
instruction-execution witnesses.

Repository-local Apple routing names `aarch64-macos`; DGX Spark routing names
`sm_121`. They establish that Apple Silicon and DGX are named operational
targets. No fresh remote machine observation succeeded or was claimed in this
receipt.

## Commands

```bash
./sounio-whereami --quick

kubectl -n slurm-pilot exec <worker-pod> -- sh -lc \
  'grep -m1 "^model name" /proc/cpuinfo; \
   grep -m1 "^flags" /proc/cpuinfo; lscpu'

bash scripts/ci/build_ontology_validation_souc.sh \
  /tmp/pireus-v01-ontology-validation-souc

/tmp/pireus-v01-ontology-validation-souc check \
  examples/pireus_target_profile_query.sio

/tmp/pireus-v01-ontology-validation-souc run \
  examples/pireus_target_profile_query.sio

/tmp/pireus-v01-ontology-validation-souc run \
  examples/pireus_target_profile_query.sio
```

The two extracted streams beginning with `SOUNIO_AUTHORITY` were byte-identical
under `cmp` before their shared hash was recorded.

## Sounio-Produced Result

```text
SOUNIO_AUTHORITY schema=pireus-target-profile.v0.1 role=SEMANTIC_AUTHORITY
PIREUS_V01_STORE triples=164

PIREUS_TARGETS canonical=3
 darwin_declared=1
 apple_declared=1
 dgx_declared=1

PIREUS_OBSERVED darwin_xeon=5
 apple_silicon=0
 dgx_spark=0

PIREUS_FEATURE_COUNTS avx512f=5
 avx512_vnni=4
 avx512_vbmi2=3
 amx_tile=3

PIREUS_OPERANDS data=1 selector=1 mask=0 dest=write matches=1
 expected_present=1
 data2_false_positive=0

PIREUS_OPERAND_CONTROL data=2 selector=1 mask=0 dest=read_write matches=1
 expected_present=1

PIREUS_OPERAND_NEGATIVE data=1 selector=0 mask=0 dest=write matches=0

PIREUS_V01_SUMMARY failures=0
```

Line breaks around integer fields are emitted by the rebuilt/current-source
runtime's `print_int`. No formatter rewrote the hashed semantic stream.

## Established Capability

Pireus v0.1 can represent and query:

- three canonical targets across x86-64, AArch64, and CUDA target families;
- five live Darwin Xeon machines with separate material profiles;
- selected kernel-reported features without promoting them to executable
  instruction support;
- operand roles and destination access independently of payload source count.

The positive results are protected by negative controls: Apple Silicon and DGX
are canonical but unobserved; the one-data-source form excludes the two-data-
source control; and the zero-selector form does not exist.

## Evidence Boundary

This receipt does not establish Apple or DGX runtime access, vendor instruction
encodings, OS-enabled feature execution, latency, throughput, scheduling,
lowering correctness, or Cayley-Dickson speedup. It establishes only the frozen
Sounio ontology and the named material observations.

The existing Apple and DGX operational scripts contain Python receipt helpers.
They were not executed and are not accepted as Pireus or language-authority
evidence under the founder's no-Python contract. Their later migration belongs
to their owning operational lanes and the canonical Loom guardian, not to a
second Pireus enforcement implementation.

## Parity And Review State

Lean, Koka, C++, Haskell, and external LLM parity/review were not invoked.
External LLM offload reviews invoked: none; this internal ontology receipt adds
no mathematical, clinical, or external-facing claim.

`PARITY_OPEN` requires both:

1. registration of proposed Concept-ID `SOUNIO-PIREUS-MATERIAL-ONTOLOGY`;
2. acceptance of this frozen Sounio receipt by the active Loom owner.

No receipt from a parity language or LLM may replace the Sounio source,
semantics, output, or hashes recorded here.

## Semantic Outcome

Semantic-Outcome: Pireus now distinguishes canonical target, observed machine,
processor profile, reported feature, operand role, and destination access.

Concept-Status-Before: v0 synthetic material-capability vocabulary frozen;
target-profile distinctions absent.

Concept-Status-After: v0 unchanged; v0.1 target/profile extension frozen,
pending registry and Loom acceptance.

Distinctions-Added: canonical versus observed; target versus machine versus
profile; reported feature versus executable capability; selector versus payload
data source; destination write versus read-write.

Distinctions-Preserved: Sounio semantic authority; evidence-role boundaries;
material profile distinct from source semantics; missing evidence distinct from
negative evidence.

Distinctions-Erased: none.

Evidence-Run: rebuilt-current-source ontology wrapper check plus two identical
Sounio executions.

Fallback-Path: the wrapper's canonical Madaros fallback agreed with the rebuilt
checker; no semantic fallback produced expected results.

Legacy-Kept: frozen Pireus v0 model and query remain unchanged.

Conflicting-Lanes: none observed in the scoped write set.

Next-Semantic-Interface: Loom receipt acceptance, then a new Garden-first
Sounio artifact for the initial vendor instruction slice.

## Loom Admission (Append-Only)

Recorded UTC: `2026-08-27T04:57:24Z`

The active Loom owner independently revalidated every frozen v0.1 hash, a
rebuilt-direct check, and two byte-identical `SOUNIO_AUTHORITY` streams. The
frozen receipt was then submitted to the executable Loom language-authority
contract as this transition:

```text
frame=9020
stage=2 SOUNIO_EXECUTABLE
action=3 FREEZE_SEMANTICS
language=1 Sounio
role=1 SEMANTIC_AUTHORITY
policy_state=1 available
semantic_write=1
expected_result_write=1
parity_receipt_valid=0
review_promoted=0
exception_and_waiver_fields=0
guardian_fields=0
parent_semantics_sha256=absent
waiver_sha256=absent
```

The complete receipt bindings were:

| Field | SHA-256 |
| --- | --- |
| Sounio source | `caa72eb54c10123eacc15fcb78e146e81775afcdb52089a9168153195d63f067` |
| Frozen semantics | `fc9e2c8c895f3a2a955236a04a2a71abb6a5fd3a70e7c3961ed4a82d093d08da` |
| Rebuilt Sounio toolchain wrapper | `ac705d2a14710b7034d08bd742c159acd1129b4f3760d69a522ee05fb7933395` |
| Execution hardware record | `fd73771f3ac0ac200b6d93641b95744ea673c80a83f6d588adb8a19c5e1cf8f0` |
| Command record | `3dba88f1091bfbd8f32ef3742a71c3116a28362bc4313a8f2c7fd14f0296d9a8` |
| Sounio result | `2d9c3fafe0bb7184d300ffbc325ad6932e7d9426f010bd3af49abc40189676c0` |

The hardware record was the following exact UTF-8 text with one final `LF`:

```text
os=Linux 7.0.2-5-pve
architecture=x86_64
cpu_model=Intel Xeon Gold 6526Y
sockets=2
cores_per_socket=16
threads_per_core=2
logical_cpus=64
```

The command record was the following exact UTF-8 text with one final `LF`:

```text
/tmp/pireus-v01-ontology-validation-souc run examples/pireus_target_profile_query.sio
```

The admission ran through the native operational realization of the frozen
Sounio policy. It did not introduce a second semantic kernel:

```text
loom_semantics_sha256=16e283166d29d6b18ed690b000e2eb595a7d965e4357553a8380714486429fff
runtime_path=/tmp/pireus-loom-language-authority-runtime
runtime_sha256=208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
runtime_selftest=SOUNIO_LANGUAGE_AUTHORITY_SELFTEST PASS cases=33
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
```

The decision hash includes its final `LF`. It equals the canonical Loom freeze
decision hash because both requests reach the same stable decision line.

A deliberate pre-execution Python-oracle request reused the complete receipt
bindings but declared `language=7 Python` and `role=7 PROHIBITED`. Loom refused
it before any Python interpreter or requested effect could run:

```text
exit_code=110
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SOUNIO_EXECUTABLE
decision_sha256=6fb4b46368e5dae161164f82e73ef0803084ae7a5d5cd8ec39588a1b9b44281d
```

This admission accepts the v0.1 transition to `SEMANTICS_FROZEN`.
`PARITY_OPEN` remains closed until the Loom owner durably registers
`SOUNIO-PIREUS-MATERIAL-ONTOLOGY`; no parity language or review system has
acquired semantic authority.
