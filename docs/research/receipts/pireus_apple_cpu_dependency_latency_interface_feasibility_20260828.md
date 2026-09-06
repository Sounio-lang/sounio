<!-- docs:meta
topic_id: repo.docs.research.receipts.pireus-apple-cpu-dependency-latency-interface-feasibility-20260828
authority: historical
audience: researchers
last_validated: 2026-08-28
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.receipts.pireus-apple-cpu-dependency-latency-interface-feasibility-20260828
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pireus Apple CPU Interface-Feasibility Freeze Receipt

Date: `2026-08-28`

Concept-ID:
`SOUNIO-PIREUS-APPLE-CPU-DEPENDENCY-LATENCY-INTERFACE-FEASIBILITY`

Decision: `SEMANTICS_FROZEN`

## Authority

```text
producing_language=Sounio
producing_role=SEMANTIC_AUTHORITY
authority_engine=lean_single
external_llm_role=REVIEW_ONLY
python_role=PROHIBITED
rust_role=PROHIBITED
fallback_path=NONE
```

No external LLM produced or confirmed the result. No Python, Rust, Node, Ruby,
shell, `awk`, or `bc` process was used as a semantic oracle. The established
Node docs-governance generator performed only the repository's declarative
metadata synchronization; it did not produce any semantic value.

## Causal Custody

```text
garden_commit=30237723bc53bbee48a93893be4da5b5f2118053
garden_sha256=19482cbceb1bf7f3f7236446ebeff8b7d46c7b99249ba5910ece145fad641dd7
first_executable_commit=c924d0014c88af8873eeaa3ca5d2c11cf468a167
first_executable_source_sha256=0893a32298d30cd1978039fa5b69c637e446aa5da112812fcf776cd52fbc4767
matcher_absent_in_first_executable=true
matcher_bearing_source_sha256=d8c7e6f9410c36f6858fb2379efa010a5adbaa32c615d89edc3e764a0606a6be
semantics_sha256=6819916ac4240923a149dd95ee9dcbeaba8d3826b7452dd819e177ff62ce8c7f
```

The first executable commit preceded the matcher and semantics document. The
Garden preceded the first executable. The dedicated gate extracts that commit
and replays the matcher-free source with stdout SHA-256
`8d1fd281f079f0287e4ceddfea31a3c51594e8cfb4b0196e2c9fa1b68b236c06`
(`141` lines, `2499` bytes). This is deterministic Git-ancestry evidence, not
an external timestamp claim. No parity implementation or material probe
created the frozen result.

## Source Bundle

```text
source=stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_feasibility.sio
source_sha256=d8c7e6f9410c36f6858fb2379efa010a5adbaa32c615d89edc3e764a0606a6be
example=examples/pireus_apple_cpu_dependency_latency_interface_feasibility.sio
example_sha256=b7e0f89c3684025407094d5abcdfa5508f7de56e5b415d14882d54dab6b41873
test=tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_interface_feasibility.sio
test_sha256=dc9345b15444a53fa46c14dc68e090be694757e0fbd12ad6ed0b37944668435c
semantics=docs/research/pireus_apple_cpu_dependency_latency_interface_feasibility_semantics.md
semantics_sha256=6819916ac4240923a149dd95ee9dcbeaba8d3826b7452dd819e177ff62ce8c7f
```

## Ordered Direct Parents

```text
0 19482cbceb1bf7f3f7236446ebeff8b7d46c7b99249ba5910ece145fad641dd7 docs/internal/garden/seeds/2026-08-28-pireus-apple-cpu-dependency-latency-interface-feasibility.md
1 3a92553aae0c4a9606f4964e1613a31452b7fb4d197b128dbb73cc24b87b550e stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio
2 9bd767db814e47bfc087e07c0f9ff33b65faea5b885ae0f8ed3a6e646c015e6d docs/research/pireus_apple_cpu_dependency_latency_request_semantics.md
3 0ee12f3502efb26056bdbcf850360c0a5df727627a3c67499d363744f7c73272 docs/research/receipts/pireus_apple_cpu_dependency_latency_request_20260828.md
4 cf4455690426038cc7477b673bcf763e9755e8147f1ff55e882086826626482b docs/research/evidence/pireus_apple_cpu_dependency_latency_request_20260828.txt
ordered_raw_manifest_sha256=bb0c19a4f03dea06ed496b3a9f7d8f29b3122962a8d08f0cc03f848cb0b91607
```

The request source is not merely hash-pinned. The child executes it live and
requires its frozen matcher to accept.

## Toolchain And Authority Hardware

```text
toolchain_manifest_sha256=850c094e02d85fee153297ccf8babbe171e3ec47def68ac2976c3473092b36ac
authority_host_manifest_sha256=464f1a4530cb0829854ddbafc0786d12cc9fc98cef1afced51f40679ba27517c
host=sounio-workspace-control-0
architecture=x86_64
kernel=7.0.2-5-pve
os=Ubuntu 24.04.4 LTS
cpu=INTEL(R) XEON(R) GOLD 6526Y
logical_cpus=64
sockets=2
cores_per_socket=16
threads_per_core=2
```

This is authority-execution hardware, not the target of the future material
probe. The Apple target remains exact `Mac17,7 / Apple M5 Max` and was not
contacted in this freeze.

## Commands

Authority command SHA-256:

```text
b1cbc4eabcd823c20c612eac0dc023f3b7876445026db62ca268c49aa49070f1
```

Test command SHA-256:

```text
17e2dbca7b1b32c7ae8e3b75a88e4d629b410ad5f81bd641390ee900e46dd692
```

Pre-matcher replay command SHA-256:

```text
dcc2c910e7684746ce7de95cf76ae126a2d482234217a6a0235b39676a33fdb8
```

Source-check command SHA-256:

```text
005113f65da96ab5df3dbce1140fd6ad760014e24b0026436439a033629e6ec7
```

All three records use explicit `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc`.
They invoke the public wrapper, never a raw ELF. The dedicated freeze gate
replays the authority and test records. The source-check record is retained as
an advisory standalone diagnostic because imported-parent baseline diagnostics
make its nonzero exit unsuitable as a freeze verdict.

The authority command supplies the five direct child parents followed by the
parent request's exact 33 transitive inputs, for 38 arguments total. PDF and XED
are inherited inputs to live parent evaluation; they do not define Apple
interface feasibility.

## Sounio Result

```text
parent_files=5
parent_files_matched=5
request_parent_live=true
request_id=4
target=701201
machine=707301
engine=707302
engine_kind=703210
isa=703221
subject_kind=706281
subject_node=0
quantity=706203
unit=706221
scope=706241
statistic=706262
samples=1001
warmups=128
environment_mask=2047
families=6
material_candidates=0
stage=709500
verdict=709510
candidate_count=0
terminal_count=0
feasible_count=0
refusal_count=0
cycle_ineligible_count=0
ontology_triples=25
negatives=32/32
error=0
failures=0
frozen_match=true
```

Named values:

```text
stage=ASSESSMENT_REQUESTED
verdict=UNASSESSED
```

This is not a claim that an Apple cycle interface exists or does not exist.

## Digests And Output

```text
parent_manifest_digest=3519613688:585074947:3796294859:195503079:3757313902:1414521947:791258454:3618061202
ontology_digest=19739672:3332345842:2129057504:398490585:1926487274:3200654896:4045422321:2496634919
result_digest=175669911:1759239310:3072997883:3130912607:2026014757:2084589116:1191266368:4096324446
authority_stdout_sha256=488b92632a0fdaa985618a67d03f84b81f69f0d7b33e2af243360f84215e81f5
authority_stdout_lines=143
authority_stdout_bytes=2537
test_stdout_sha256=6d5f78969bcbf3667fb0f0020cc0b28b42ed4e815ab0a20a84124dea6d93a57b
test_stdout_lines=1
test_stdout_bytes=51
```

The dedicated gate requires three byte-identical authority replays on the same
authority host. They establish deterministic replay for the pinned inputs, not
independent reproduction across hosts.

## Boundaries

```text
remote_execution=false
environment_bound=false
harness_frozen=false
measurand_validated=false
execution_authorized=false
cost_present=false
parity_open=false
claim_ready=false
```

No C++ producer was executed. No SSH or tailnet connection was opened. The
target locator was not serialized. The six observation slots remain empty.
The inherited request metadata records the future transport locator as
`demetrios@sounio-language-macbook`; that routing string is outside the child
semantic binding and did not cause target contact during this freeze.

## Tamper And Prohibition Gates

The dedicated gate requires:

1. mutate each of the five raw direct parents in a private temporary copy and
   require authority rejection;
2. require the exact matcher to accept the untampered result;
3. require all 32 named negatives;
4. send a deliberate Python semantic-execution request to Loom and require
   `SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language` before
   any interpreter launch;
5. leave no waiver requested or supplied.

The in-module parent negative is only a mask-level mutation witness. Raw-file
tamper closure belongs to the dedicated gate above.

## Review

Authorized external review checked the classifier arithmetic, terminal
partition, manifest cardinality, ontology count, negative table, digest
packing, causal wording, and governance status. Review findings tightened the
Sounio source before freeze and later corrected the semantics token spelling,
made the raw manifest recipe explicit, and removed an impossible historical
validation date. Review did not produce or confirm the authority result and
cannot be promoted to semantic or material evidence.

## Next Allowed Stage

Once Loom accepts the source and semantics hashes in one freeze receipt, the
next stage is `PARITY_OPEN`. That stage may admit a hash-bound C++
`MATERIAL_PARITY` probe for the exact Mac. It does not authorize a cost
measurement, lowering claim, or `CLAIM_READY` promotion.
