<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-intel-vpermpd-selector-semantics-20260827
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-intel-vpermpd-selector-semantics-20260827
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Intel VPERMPD Selector Semantics Receipt

Receipt-Schema: `sounio-semantic-authority-receipt.v1`

Date: `2026-08-27`

Concept-ID: `SOUNIO-PIREUS-INTEL-VPERMPD-SELECTOR-SEMANTICS`

Semantic-Lane-ID: `pireus-intel-vpermpd-selector-semantics-20260827`

Producer-Language: `Sounio`

Producer-Role: `SEMANTIC_AUTHORITY`

Stage: `SEMANTICS_FROZEN`

Parity-Open: `false`

Claim-Ready: `false`

## Result

Sounio decoded the exact pinned Intel SDM Volume 2C bytes, recovered the
EVEX.512 vector-control selector rule, and matched it against the frozen
bits=4 Pireus `XOR_PERMUTE` layout.

```text
form=EVEX_512_VECTOR_CONTROL
selected_lane(k)=control[k]&7
dimension=16
chunk_count=2
chunk_lanes=8
xor_selector_cells=256
xor_selector_matches=256
abstract_form_applications=32
one_source_form_sufficient=true
two_source_form_required=false
```

The immediate-control form was kept separate and is incomplete for this
layout. For `d in [0,3]`, lane bit 2 is preserved and the same two-bit
permutation is repeated in both 256-bit halves. For `d in [4,7]`, lane bit 2
is flipped and cross-half selection is refused. All 18 internal negative
witnesses pass. The result contains zero material observations, zero
lowerings, zero cost records, no open parity, and no claim-ready promotion.

## Causal Commits

| Phase | Commit | Meaning |
| --- | --- | --- |
| Garden admission | `19d2c05578` | admitted the vendor-semantic question without expected values |
| first Sounio executable | `0258185d1cfc40a1a71f766059654f3f1eb0e294` | emitted the selector result before this prose existed |
| semantics freeze | enclosing Git commit | binds the reviewed contract, semantics, receipt, and registry |

The executable commit immediately follows the Garden commit.

## Sounio Source

| Artifact | SHA-256 |
| --- | --- |
| Garden | `6eec7cf4cab6e716a5205aef8e9bc59a73e0f718c9300e986362b37b229e5a5c` |
| module | `4f7e007aa432564b873c941e239c353aeed3b11883844b73ec7e31ace4811b20` |
| authority executable | `7dd75a46c35a41caa3bc358bf226e615d350aab5f897c219d6e2265d3ece3d66` |
| dedicated Sounio test | `f975c72592086ab5f3dc46d1ec4878eabb3d83f4908d2a1327ccb01b26e4877f` |
| source hash manifest | `65bbbcd780f15360593bddf128a11b4a6a2c119acf541f5604d438be56ec2cf7` |
| concatenated source bundle | `d75ea83405394789cecdeab919ae0817b8f1890e020386cc61db042c31d31e7b` |
| concept contract | `333626823397894a8eaf2449175c2b2db9e3923b3463e0da58f45f181392c42a` |
| frozen semantics | `ba25ceb18685ed656ecaf1c577eb95a698ca214fe2a939fce5a8ffd6d106b243` |

The source hash manifest is SHA-256 over the three `sha256sum` records in the
listed module, executable, and test order. The concatenated bundle hash is
SHA-256 over the exact three file byte streams in that same order. The Loom
`source_sha256` field uses the manifest hash and the receipt preserves both
representations plus every individual file hash.

## Frozen Parents

```text
xor_material_source_sha256=eadd752fbda1f50f24bed1260c54936d710af10973653982f5687cd8a551a575
xor_material_semantics_sha256=b4791514032859acc0e8888c4d35760f549a6267e02b2cd5f30a96c0b9dee554
xor_material_receipt_sha256=cc157a7d6ba33b945bc9537be1856bf60481573067ade5f166101ca36a98c1df
```

The child live-ran the parent evaluator and required an exact frozen-semantics
match before authorizing the selector-form result.

## Vendor Inputs

### Intel SDM

```text
manual_version=092
volume=2C
landing_page=https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
pdf_url=https://cdrdv2-public.intel.com/922483/326018-092-sdm-vol-2c.pdf
pdf_path=/tmp/intel-sdm-vol-2c-326018-092.pdf
pdf_bytes=3298744
pdf_sha256=939c9543ff98eefb80f5c5a517bf6f08e864497ea8e032334849f3e39a7b3b07
```

### Intel XED

```text
release=v2026.08.23
commit=0bcb6237345c5066726dcc08b3d87928df3b5b26
xed_path=/tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt
xed_bytes=458470
xed_sha256=e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038
```

The PDF supplied selector behavior. XED supplied form and lineage evidence
only.

## PDF Projection Receipt

```text
header_pdf_1_6=true
eof_offset=3298737
startxref_offset=116
startxref_indirect_object=true
stream_markers=2034
flate_streams=2032
decoded_flate_streams=2031
oversized_flate_outputs=1
oversized_output_stream_offset=3292436
oversized_output_class=/XRef
oversized_output_content_candidate=false
flate_decode_failures=0
description_page=5-522
description_stream_offset=1678050
operation_page=5-524
operation_stream_offset=1684617
operation_selector_cells=8
```

The one non-inflated Flate output is explicitly classified as the xref stream,
not as a content candidate. The semantic anchors are unique under the bounded
normalized text projection.

## Toolchain, Hardware, And Command

```text
public_wrapper=bin/souc
public_wrapper_sha256=ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
selected_engine=lean_single
compiler=bin/souc-lean-single-x86_64
compiler_sha256=6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
toolchain_record_sha256=0dd7961c7b9b16f0fd218092c651e9181e91cb1e1e4631fd17f0a756452c1556
kernel=7.0.2-5-pve
architecture=x86_64
logical_cpus=64
cpu_model=INTEL(R) XEON(R) GOLD 6526Y
hardware_record_sha256=8765315349a0cad84314745fb7237010b8c3450076760f68a3801f749bd85b12
```

The exact authority command was:

```bash
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
  examples/pireus_intel_vpermpd_semantics.sio \
  /tmp/intel-sdm-vol-2c-326018-092.pdf \
  /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt
```

Command SHA-256: `7f217e2458834bf27d1b3c7d828cd2eff258037b9952342ba630b2c37e85ee00`.

## Authority Stream

```text
lines=69
bytes=1692
sha256=f8757e0da4770dab7e83414cb128ba1db203af10dca8d79c9515e182855e7220
error=0
failures=0
```

The dedicated Sounio test emitted
`PIREUS_INTEL_VPERMPD_EXECUTABLE_OK`; its one-line result SHA-256 is
`eaab31011a85a430c3a323cd4f7e5e598d59abaadac5d499386af4992ff44bf0`.

## Preliminary Loom Decisions

Before the final receipt-bearing authority run, Loom admitted the Sounio
execution frame:

```text
frame_sha256=8440488ceb8d7ee1d883d8e3bfc3718574f0278b03fba1ee1f7af4e69e5c9a97
decision_sha256=2d490815fb2e56b74303a46cda871f5e28eefa053f5d5dbc0e701cdf97fab266
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE
```

The test and executable commit received separate Sounio-language-authority
ALLOW decisions. Their frame hashes are respectively
`0edbe91b01a3af98376cadab5855a06b312477c5646b28845d1bcb269e538c35`
and `3f30c56a8012d92e1a94ec51f06f4ab0133401c5da445696e73cd71d5d1e5b9c`.

## Default Compiler Diagnostic

`./bin/souc check examples/pireus_intel_vpermpd_semantics.sio` selected
Madaros v0.80 and exited 1 while checking the 12 imported modules. The log
contains the advisory science-boundary verdict `UNKNOWN` and diagnostics
including `E008`, `E012`, and `E035`. Its SHA-256 is
`6858260da44ea5035cdef0c7aff0b2c8e8bdc9b59a3eb8b8c8cb1d10744c4579`.

This is a default-path compiler/checker blocker. It is not a second expected
result, a fallback authority result, or a failure of the explicit lean_single
authority execution. No Madaros source was changed in this lane.

## Review-Only Offload

xAI/Grok 4.5 completed the full math review and found no wrong identity or
compound downstream breakage. It checked the vector selector, XOR bit split,
256-cell and 32-group counts, one-source boundary, immediate 4/4 split, and
the separation between abstract applications and machine instructions.

The review produced three precision improvements that were incorporated:
the 16-element/two-chunk domain is now explicit; the lane map is identified as
a permutation; and the immediate form states the repeated two-bit permutation
and lane-bit-2 split explicitly.

The required independent fan-out was degraded. Z.AI produced a long
reasoning trace consistent with those checks but ended by token limit without
a final verdict; its focused retry returned an empty artifact. Qwen and
Mistral failed with OpenRouter HTTP 402, while Groq and DeepSeek rejected their
configured credentials. These failures are not counted as passes.

```text
outcome=PASS_SINGLE_PROVIDER_DEGRADED
xai=COMPLETE
zai=INCOMPLETE_THEN_EMPTY
qwen=ERROR_402
mistral=ERROR_402
groq=ERROR_INVALID_KEY
deepseek=ERROR_INVALID_CREDENTIAL
raw=/tmp/llm-offload-l3XIVy/
raw_zai_retry=/tmp/llm-offload-wtftv9/
raw_qwen=/tmp/llm-offload-Cofm75/
raw_mistral=/tmp/llm-offload-zxtwoC/
raw_groq=/tmp/llm-offload-xS9k5b/
raw_deepseek=/tmp/llm-offload-4Q9COY/
```

No external model created or confirmed selector values, counts, source hashes,
or the material-match verdict. Sounio remains the sole semantic authority.

## Freeze Decision

The complete Loom frame bound the exact source manifest, reviewed semantics,
toolchain record, hardware record, command, and authority result:

```text
frame_sha256=a5ddad4562fe3691a31f2ad84f556cada9131ad7bbe09e18bdcf793de71a58f6
decision_sha256=ad57cad1376867f5cec01323e50d39319c7fb8ba0458a6066086d186a045b8cb
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN
parity_open=false
claim_ready=false
```

## Prohibited Producer Negative

A would-be Python parity producer was classified and framed without launching
an interpreter. Even with complete nonzero source, semantics, parent,
toolchain, hardware, and command fields, Loom refused it before execution:

```text
command_record_sha256=d8c923939c7203bff91cbbcb4f7300771fe3968de9dd3c1076e6075735042972
frame_sha256=55e86e29706ff5daa1105845dd8ec8da32d37e37723d1dd5e51833adc06fc756
decision_sha256=3e2b1112dc7ce41d6c752c48daca33e6ee400b93df1e3fafa795a5709b4aa2a3
exit_code=110
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN
interpreter_launch_count=0
```

Rust was not used.

## Canonical Targets

| Target | Canonical | Result in this lane | Materially observed |
| --- | --- | --- | --- |
| Darwin Xeon | true | vendor selector form matched to abstract XOR layout | false |
| Apple Silicon | true | unresolved | false |
| DGX | true | unresolved | false |

Canonical declaration is not observation.

## Closed Claims

This receipt does not authorize:

- compiler emission, machine instruction count, or full-operation lowering;
- AVX-512 availability or execution on Darwin Xeon;
- any Apple Silicon or DGX result;
- latency, throughput, scheduling, register pressure, cost, or speedup;
- the earlier roughly 112-instruction estimate;
- immediate-form completeness or a need for a two-source form outside this
  exact one-source `XOR_PERMUTE` boundary;
- Walsh-Hadamard diagonalization or subquadratic twisted convolution;
- a Fano-plane theorem or explanation of the seven-negative-sign pattern;
- Lean 4, Koka, C++, or Haskell parity;
- `PARITY_OPEN` or `CLAIM_READY`.
