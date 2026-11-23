# Pireus continuity execution

Canonical plan: docs/roadmap/PIREUS_CONTINUITY_PLAN.md. Current acceptance state:
status.json. This directory preserves source lineage, review output and
executable validation receipts separately from planned milestones.

The current-source Madaros repair for Seq<T> struct-field ownership is in
commit 193670aa6b on the parent integration branch. Three unchanged ontology
queries and six scoped Pireus gates pass; formal V13/V14 remain OPEN.

## External proposal admission

admission.sio is the Sounio semantic boundary; cycle.py transports data and
persists immutable dependencies, raw requests/responses, proposals and receipts.
Its commands cover prepare, generate, validate, materialize, benchmark, report
and resume. The integrated benchmark coordinator completed the live deterministic baseline:
eight distinct plans, exact pair parity,30 paired blocks per node, eight
NO_GAIN decisions and zero promotion-eligible plans. New operators and the
GRPO corpus remain pending.
The real Inkling eight-request canary also completed in11939–11942: eight
admitted/materialized outputs, four distinct plans, exact pair parity and eight
NO_GAIN decisions. This completes a real canary cycle; the production pilot,
new operators and GRPO corpus remain pending.

Compile admission.sio through bin/souc with the rebuilt engine on the R770.
Run test_admission.py and test_cycle.py against that actual executable.
The committed gate transcript records adversarial refusal and an eight-plan
deterministic custody regression. No test fixture is a real LLM response.

prepare accepts --context-engine RESEARCH_CONTEXT.elf or a supplied context,
provenance evidence and the admission executable SHA256. The Sounio producer
queries TripleStore/SPARQL with declared research-local primitive facts.
These are not observations of running hardware. The semantic contract is
docs/internal/concepts/pireus-external-proposal-admission.md.

```sh
python3 tools/pireus/continuity/cycle.py prepare --run RUN \
  --context CONTEXT.json --evidence QUERY_RECEIPT.txt \
  --engine-sha256 ADMISSION_SHA256 --condition deterministic --budget 8
python3 tools/pireus/continuity/cycle.py generate --run RUN
python3 tools/pireus/continuity/cycle.py validate --run RUN --engine ADMISSION.elf
python3 tools/pireus/continuity/cycle.py resume --run RUN
```

HTTP Inkling transport requires the internal endpoint and its actual model ID.
The separate offline transport uses an owned Slurm batch and paired token-ID
receipts, without an HTTP endpoint. Generation preserves the original response without
repairing malformed JSON. An interrupted request with no persisted response
is ambiguous and will not be silently issued twice. resume verifies custody
and reports remaining stages; it does not mutate the frozen research context.

The production pilot still requires the founder's fixed three conditions,
three rounds, 32 proposals per condition, 30 interleaved measurement blocks
per node, and the existing promotion criteria. The deterministic baseline
and the real Inkling eight-request canary have measured results; both conclude
NO_GAIN. The full production pilot is not complete.


## Owned offline Inkling batch

The current transport is sglang-offline-token-ids. It preserves the pinned
checkpoint/SIF, original mixed NVFP4 and BF16 weights, configured context16384,
one request at a time and at most4096 output tokens. Four hash-bound source
overlays reduce unused Marlin allocation and bound expert/vocabulary copies.
Original Marlin kernels are unchanged; the LM-head uses the explicitly qualified
4096-row projection shape described below. Actual stock/candidate GPU controls and
real checkpoint-tensor loader controls are in validation/.

Checkpoint CPU staging slabs are at most4MiB. The separate file-backed LM-head
projection stages4096 BF16 rows (32MiB) at a time. This matters on the Sparks:
direct CUDA reads from an mmap-backed packed tensor reduced host availability
by about1.1GB per rank, while the staged real-tensor controls preserved exact
TP slices with MB-scale overhead. Checkpoint files and the original SIF remain
unchanged. The rank script fixes its own NCCL2-channel/256KiB profile and skips
the resident tokenizer, HTTP processes and linearized shared-expert copy.

Full initialization, all ten checkpoint files, postprocessing and canonical
hybrid caches completed on both ranks in11918. The first proposal then hit
the33GiB early guard. The native host floor remains32GiB. Releasing unused
allocator pages in11919 was insufficient. These results do not constitute
completed generation or general HTTP-serving acceptance.

For the frozen eight-request canary, the offline cache capacity is 6144 tokens. Its SWA/full ratio is 0.15: the prior 0.1 ratio
produced only 512 SWA tokens and was refused by the runtime admission floor
(511 sliding-window tokens plus one 128-token page) in job11922.
Each encoded prompt is344 tokens plus an unchanged4096-token output allowance.
Oversized requests refuse before loading, and actual pool capacity is checked
again before each request. The model context setting remains16384, but this
batch profile does not accept arbitrary16384-token requests. Cache budget is
part of the hashed runtime lock and a changed profile requires a newly prepared
run; do not rewrite a previous manifest.

Use cycle.py prepare with --transport sglang-offline-token-ids, --budget8 and
the intended native context/admission dependencies. Then use tokenized_cycle.py
pack-encode, the owned launch_pair.py tokenize job, and accept-encode with both
rank receipts. pack-offline creates the frozen inference input bundle.
launch_pair.py offline-generate --input-bundle RUN/offline-bundle.json runs the
actual pinned SGLang worker/ModelRunner/sampler path. It must run inside remote
tmux. Check the Slurm job's final state before collecting both complete receipts
and all16 per-rank response files for accept-offline.

After inference teardown, pack-decode and another owned tokenize job produce
paired decoder receipts. finalize preserves the decoded proposal text exactly.
Continue with native validate, materialize, applicable pair benchmark and report.
No malformed model text is repaired, and failed or ambiguous attempts remain
evidence rather than being silently replayed. Current outcome: status.json.

## Material evidence and measurement contract

Sounio emits admitted PTX through materialize_ptx.sio and fixtures through
numeric_fixtures.sio. Python only loads PTX and records GPU output bits.
material_parity.sio decides exact non-NaN bits / NaN class agreement.
The 320 vectors cover all 256 basis pairs, 32 dense inputs and 32 edge inputs.
Job 11859 passed all eight plans on both Sparks (5120 exact bits each).
Job 11860 ran an intentionally poisoned sign mask outside admission: both
nodes refused with 42 mismatches. These finite tests are not a general FP proof.

benchmark_decision.sio consumes four sets of 30 paired blocks: direct and
shuffle controls on each node. It computes median gain in ppm and a seeded
4000-resample percentile bootstrap. All four medians must reach 50000 ppm
and all four lower 95% bounds must be positive. This is an exploratory
per-comparison interval, not familywise coverage. The measured scope is
resident-layout kernels; layout conversion is excluded. Each CUDA event
brackets 32 captured kernel launches over 16384 vectors. Partial trials are
retained and refused on retry; they are not silently overwritten.

Materialize with cycle.py materialize --run RUN --engine ADMISSION.elf.
Run cycle.py benchmark inside workspace tmux, after generation teardown,
with --run, --engine, --fixture-engine, --parity-engine and --gain-engine.

The observer/launcher checks both host grants in addition to the Kubernetes
lease. It refuses FENCED or unknown observations and leaves recovery to the
frozen Spark Pair Arbiter. Checkpoint hashes passed on both nodes in job 11864;
that historical load failed under the host fence. Canonical recovery at epoch15
and production database relocation subsequently passed. Full initialization
now passes, but first-request memory acceptance remains pending as described
above. The historical recovery also exposed a race between
worker recreation and proving the fenced cgroup set empty.


Offline completion receipts and every token response explicitly carry execution_profile:
configured context16384, requested/actual full cache6144, actual SWA capacity,
SWA/full ratio0.15, page128, TP2, concurrency1, output4096, host floor32GiB,
guard33GiB, and existing-pynccl collective backend. Admission refuses an absent
or incompatible profile, including an HTTP or general16K acceptance claim.

The eager offline path enables the already initialized PyNCCL communicator for
forward/decode and rank-zero token broadcast. Its final barrier uses the existing
CPU group. Diagnostic11927 reproduced a lazy torch.distributed NCCL communicator
failing against the worker's64MiB /dev/shm at the embedding all-reduce.
Diagnostic11928 passes exact collective controls and two synthetic embedding/norm
passes on both ranks using PyNCCL. Those controls load no checkpoint tensors and
stop before the first transformer layer; they are not real generation evidence.


The current offline profile places only the input embedding in a read-only local
file-backed CPU mapping. After full checkpoint loading, the adapter streams the
local BF16 GPU shard to an owned file in4MiB blocks, synchronizing and dropping
clean file pages per block to bound write transients. Source/file hashes must
match embedding-offload-lock.json before inference. CUDA receives only owned
gathered token rows; original TP masking/all-reduce and expert weights retain
their runtime paths and precision. The separate LM-head placement is described below.

Control11930 compares every201024 vocabulary row to the real fused GPU embedding
on both ranks, plus344/1/boundary inputs and deliberate private COW corruption.
All bytes match and823394304 CUDA bytes/rank are released. This is input-embedding
qualification, not full inference acceptance. Completion receipts include the
rank-specific embedding storage hashes and helper identity, and responses declare
embedding_placement=file-backed-cpu. The importer rejects mismatched storage or
precision claims.


Inductor compilation is limited to one worker and asserted in the offline
profile. The container home is bound to the owned local SSD cache because TVM-FFI
uses ~/.cache even when XDG_CACHE_HOME is set; its former session-backed path
failed with ENOSPC in11934 despite ample SSD space.

runtime/warmup_offline_kernels.py is a separately named compilation diagnostic:
it constructs zero/one synthetic parameters with shape-based storage aliases,
reads no checkpoint tensors, and discards all outputs. Job11935 completes all42
layers for a344-token prefill and one decode on both ranks. This does not qualify
real model generation. The subsequent real run still verifies every checkpoint
file, qualified embedding hashes, resource/profile bounds and paired responses.

The real warmed run11936 reached transformer layers2/3 before the33GiB guards
stopped both ranks above the32GiB floor. No first token was generated.

The offline LM-head now streams original BF16 weights from a local file into
4096-row GPU tiles. Job11938 qualifies the complete file-backed path against
the stock LogitsProcessor GPU projection on21 synthetic hidden-state vectors per
rank using the actual checkpoint shards: all compared bytes match, deliberate
private corruption is detected/reverted, and unsupported multirow inputs refuse.
823394304 CUDA bytes/rank are released. This is observed equivalence on the
controls, not a universal numerical proof. The8192-row alternative differs on
the same controls and is rejected. LM-head placement, tile rows, M1 scope and
rank-specific weight/helper hashes are carried by completion receipts and checked
by the importer. Stock TP logits gathering and sampling remain in their original
paths. The next real run must separately establish generation.

The first real complete cycle is archived in validation/real-inkling-cycle-20260907.
Generation11939 emits943 tokens across eight requests, both ranks complete and
exit0; paired decode11940 preserves exact model text. Native Sounio admission
accepts8 outputs with4 distinct normalized plans. Material job11941 passes all
16 candidate-node comparisons (5120 exact components each); timing11942 measures
30 interleaved blocks per node against direct and shuffle controls. All8
decisions are NO_GAIN. Layout conversion is outside this resident-layout-kernel
timing scope. No scientific novelty, promotion, general FP proof, HTTP serving,
general16K inference or full pilot acceptance follows from this result.

## Continuous custody verification

.github/workflows/pireus-continuity.yml runs current transport controls,
preflight binding controls and archive corruption controls on pull requests.
ci/verify_archive.py verifies204 archived artifacts, the exact decoded text,
paired token responses and105 source dependencies against Git history.
It is archive/source custody validation; it does not rerun inference, numerical
parity or timing and cannot grant fresh hardware acceptance.

This audit found one originally untracked dependency: runtime/test_preflight_binding.py
was accidentally matched by the broad test_*_bin* ignore pattern. Its existing
bytes exactly match the frozen manifest hash and were recovered in9680b520d2.
ci/source-recoveries.json binds that recovery explicitly; the original run
manifest/receipts remain unchanged, and source_snapshot_complete_at_original_commit
remains false in the custody report. Missing/forged recovery evidence refuses.
The original CI failures (Contracts and Full Test Suite) remain independent
integration obligations; this workflow does not suppress or reclassify them.

## Full pilot driver

pilot.py prepares nine immutable cells, rotates condition order across three
rounds, and fixes 32 proposals per condition and round (288 total). The offline
transport accepts exactly 8 or 32 requests. The 32-request receipt scope is
frozen-offline-pilot-batch; context remains 16384, physical cache 6144 tokens,
native floor 32 GiB and early stop 33 GiB. This adds a request-budget path, not
new runtime acceptance. The first 32-request hardware run remains required.

Pilot material measurements deduplicate only within a condition and round.
Native Sounio plan identity and identical PTX bytes are both required, raw model
outputs remain intact, and material-deduplication.json records every alias.
Separate rounds retain independent measurements. The canary's eight archived
proposals reduce to four material representatives under this exact rule.

The driver requires CI Decision, continuity transport/archive CI, and gate
custody CI green on the frozen PR head before launching a pilot job. It binds
five native engine hashes and all cycle source dependencies. A stage records
its launch intent before submission. Recovery may adopt an observed completed
paired job, but never silently submits an interrupted request again.

## Operator atlas (eisa_h.pireus_operator_atlas.v1)

`stdlib/hardware/pireus/operator_atlas.sio` classifies a kind=2 operator's
16x16 sign table under GL(4,F2) x input swap x sign gauge. Canonical forms come
from the unchanged V13 canonicalizer, reached through the new public accessor
`pireus_operator_orbit_canonical_view`; no frozen V13 function, counter or
digest changed, and the lean_single V13 replay still matches its frozen
transcript. The references are the 62 frozen V13 classes (30 baseline + 32
admitted, re-derived from `operator_orbit_canonicalization.first.v13`; the
first 30 hash to its baseline `class_digest`) plus the three V1 GL4 corpus
members (untwisted XOR, Cayley-Dickson-16, diagonal bicharacter). The V13
classes do not contain the sedenion, so without the corpus phase 0 would read
as N2.

Ladder: N0 = bit-identical to a reference table; N1 = different table, same
canonical class as a reference; N2 = none of the 65 reference classes. N2 is
bounded relative quotient separation only. `scientific_novelty`,
`historical_novelty` and `claim_ready` stay false; H needs founder promotion.

Receipt contract v2: kind=2 validate receipts carry `"schema":2` and replace
`"novelty":"NOT_CLASSIFIED"` with the atlas block (`novelty`, `novelty_scope`,
`atlas_class`, `atlas_origin`, `canonical_table`, `canonical_matrix/swap/gauge`,
`atlas_v13_semantics_sha256`, `atlas_class_digest`, `scientific_novelty`,
`historical_novelty`). kind=1 receipts and every PTX mode are byte-identical.
The frozen r740 receipt digests are unchanged: the tests map a schema-2
receipt back to v1 by restoring only that block and require the old digest.
Gate: `scripts/ci/pireus_operator_atlas_gate.sh`.

### b002-b006 classification (CPU, 2026-11-20)

All 80 kind=2 proposals in inbox batches b002-b006 were reclassified through
`admission.sio` on CPU (no GPU, no Slurm). Every one is ADMIT and N2; their 41
distinct phase codes fall into 16 canonical classes. Over the whole 1024-code
bilinear twist space (upper-triangular B; transpose and alternating parts are
gauge-equivalent) there are exactly 32 canonical classes: the sedenion (phase 0,
N0, class 63) and 31 N2 classes, with 84/42/28/21/7/1 codes each. The batches
cover 16 of those 31. Here N2 means only "outside the 65 reference classes" of
a deliberately small atlas, not a scientific or historical claim. The fourth
column below counts codes per class in that 1024-code space.

| batch | kind=2 | N0 | N1 | N2 |
|---|---|---|---|---|
| b002 | 16 | 0 | 0 | 16 |
| b003 | 16 | 0 | 0 | 16 |
| b004 | 16 | 0 | 0 | 16 |
| b005 | 16 | 0 | 0 | 16 |
| b006 | 16 | 0 | 0 | 16 |

| class | proposals | phase codes | upper-triangular orbit size | canonical table |
|---|---|---|---|---|
| K1 | 24 | 32931, 32939, 34029, 35947, 36007, 36071, 36075, 36077 | 21 | `c00000009660500084cc6688f08e14aa4080a4a434c2d686266ee00e700eb00e` |
| K2 | 19 | 32800, 33792, 33824, 33825 | 7 | `c00000003ccaf00af02e00ee2466d86602fefefed6b626761a52d692e6982698` |
| K3 | 11 | 33004, 33930, 34018, 35983, 36076 | 84 | `c000000096605000743c9678007ee45abf805ba4cbc22986299eeffe7ffebffe` |
| K4 | 5 | 1, 32, 33, 1056, 1057 | 7 | `c00000003ccaf00af02e00ee2466d866fdfe01fe29b6d976e55229921998d998` |
| K5 | 4 | 35, 1029, 1120, 1123 | 7 | `c00000009660500084cc6688f08e14aa26e6c2c252a4b0e0400886681668d668` |
| K6 | 3 | 36, 67 | 28 | `c000000096605000743c9678007ee45ad9e63dc2ada44fe04ff889981998d998` |
| K7 | 3 | 1160, 2081 | 42 | `c00000009660500084cc6688f08e14aa83bc6798f7fe15bae5522332b3327332` |
| K8 | 2 | 35840, 35849 | 7 | `c00000009660500084cc6688f08e14aa7cbc989808feeaba1a52dc324c328c32` |
| K9 | 2 | 2113, 2223 | 84 | `c000000096605000743c9678007ee45a4080a4a434c2d686d69e10fe80fe40fe` |
| K10 | 1 | 36079 | 28 | `c000000096605000743c9678007ee45ab0705454c4322676266ee00e700eb00e` |
| K11 | 1 | 36046 | 84 | `c000000096605000743c9678007ee45a2916cd325d54bf10bf087968e9682968` |
| K12 | 1 | 35015 | 21 | `c00000005aac9cccb8f05ab4007ee45a26e6c2c29e687c2c7c34ba54e6982698` |
| K13 | 1 | 32774 | 21 | `c00000009660500084cc6688f08e14aad9e63dc2ada44fe0bf087968e9682968` |
| K14 | 1 | 2190 | 21 | `c00000009660500084cc6688f08e14aabf805ba4cbc22986d96e1f0e8f0e4f0e` |
| K15 | 1 | 32768 | 1 | `c00000003ccafaaa125af01eccb22896e5da01fe3b32d97615a2d3c2e9682968` |
| K16 | 1 | 3183 | 21 | `c00000005aac966c66b89678d496289602fefefeb0d040108cc440041668d668` |

## Claude API proposer: claude-api-proposer.v1 (new declared condition)

`claude/claude_proposer.py` is a NEW proposer condition. It does not replace,
redefine or count toward the Inkling pilot of
`docs/roadmap/PIREUS_CONTINUITY_PLAN.md`; `cycle.py`, `pilot.py` and their
`inkling-*` conditions are unchanged, and the Inkling pilot stays open. It
follows the same design: three arms (`deterministic-search`,
`claude-no-ontology`, `claude-ontology`), 32 kind=2 proposals per arm per
round, 3 rounds with the arm order rotated each round, and the same contracts.

- Context: `research_context.sio` output (TripleStore/SPARQL over the target
  ontology). Only `claude-ontology` sees it, verbatim plus a fixed legend; the
  other Claude arm gets "withheld". Both carry the same context hash.
- Deterministic arm: the 1024 upper-triangular phase codes in a fixed order
  (sorted by sha256 of the code), without repeats.
- Feedback between rounds only, per arm: phase, decision, reason, N-level and
  a class tag, taken from that arm's own Sounio receipts.
- Each round is generated in full, then its `batch.json` (sha256 of every
  proposal) is written, and only then admitted. A proposal changed after its
  batch hash is refused.
- Authority: the reply text is persisted unrepaired as the proposal; the
  native admission executable (`admission.sio` + atlas) decides. The model is
  never a novelty oracle. `scientific_novelty` and `claim_ready` stay false.
- Model: default `claude-sonnet-5-5`, the Claude API id checked against
  Anthropic's models overview on 2026-11-20. Both Claude arms use the same
  frozen model. The condition id is `claude-api-proposer.v1:<model>`, so an
  Opus run (`--model claude-opus-5-5`) is a separate declared condition in
  its own run directory. A resumed run with any changed model, prompt,
  engine or budget is refused.
- Sampling: `claude-sonnet-5-5` rejects temperature/top_p/top_k with a 400
  error, so no temperature is sent unless `--temperature` is given. The
  Messages API has no seed; the round and proposal index are in the prompt.
  Thinking is the model default (adaptive) with `output_config.effort`
  (default `medium`) and `max_tokens` 8192, all recorded in the manifest.
- Records: `manifest.json` freezes the model, effort, max_tokens, prompt
  hashes, prices, the context and engine hashes, and the proposer source hash.
  Each call stores its exact `NNN.request.json` and raw `NNN.response.json`,
  and `ledger.jsonl` records their sha256, request-id, returned model, stop
  reason, token counts, cost in USD and the running total.
- Budget: hard cap, `--budget-usd`, default 20. Before each call the worst
  case is reserved (request bytes as tokens plus max_tokens output, at
  US$2/US$10 per MTok for Sonnet 5.5). The run stops with `BUDGET_STOP`
  (exit 3) before the cap could be exceeded. An interrupted call whose
  request was written but no response arrived is never re-issued
  automatically.
- Modes: `estimate` is a dry run (no transport, no key, no files),
  `--mode mock` (the default) is deterministic and offline, and `--mode real`
  needs `--confirm-spend` and `ANTHROPIC_API_KEY`.
- Key: `ANTHROPIC_API_KEY` is read from the environment only, in real mode
  only. It is passed to the SDK client and nowhere else. Every artifact is
  checked for it before it is written, and errors are redacted.
- SDK: `claude/requirements.txt` pins `anthropic==2.12.0`.
- v2, `claude-api-proposer.v2:<model>`, is the default (`--history proposals+classes`).
  The first real smoke (v1, 2026-11-21) showed almost no diversity: 16
  identical-prompt calls gave phase 4660 thirteen times. In v2, each call of a
  Claude arm lists that arm's earlier proposals in the same round (phase, plus
  the class tag or refusal reason from the Sounio admission of each) and asks
  for a distinct operator. Those admissions are kept as `NNN.preview.json`,
  byte-identical to the later receipts. The batch hash is still written before
  the official admission. This intra-round feedback is part of this condition
  only (the plan's pilot has feedback between rounds only). The history has
  the same form in both Claude arms; only the ontology facts differ. No
  hardware metric enters it.
- Dedup (v2): within one arm and round, a proposal equal to an earlier one
  (same bytes, or the same JSON object after sorted-key canonicalisation) is
  recorded as `DUPLICATE` in the ledger and counted as refused in the report
  (`admitted_after_dedup`). It is still hashed and admitted, and no
  replacement call is made, so calls per arm and round stay `per_arm`.
  The same rule applies to every arm.
- Temperature is still not sent (Sonnet 5.5 rejects it); `temperature: null`
  is in the manifest. `--history none` reproduces v1 request bytes exactly
  (checked against the 16 requests of the real v1 smoke).
- Workspace: a key not scoped to a workspace is refused by the API (400)
  unless each request carries `anthropic-workspace-id`. Pass it with
  `--workspace-id` or env `ANTHROPIC_WORKSPACE_ID` (the ID is not a secret and
  is not hard-coded). It goes in the SDK client's default headers, not in
  `request.json`, and is recorded as `anthropic_workspace_id` in the manifest
  and in each ledger call. When it is not given, nothing changes.
- v3, `claude-api-proposer.v3:<model>`, is opt-in (`--prompt open`; keeps the
  v2 history and dedup, refuses `--history none`). Why v1/v2 replies were a bare
  phase number (~149 output tokens, `thinking_tokens: 0` in the real smoke):
  the prompt asked for "exactly one JSON object and nothing else", gave a
  template with every key filled, described the operator in one sentence, and
  effort was `medium` with adaptive thinking, so the model copied the template,
  changed `phase` and did not think. v3 changes only the proposer side:
  - prompt (`OPEN_PROMPT`, hash in manifest): what the operator is (B over F2,
    bit 4r+c of `phase`), the equivalence the classifier uses, every key and
    range the kind=2 admission accepts with its meaning, admitted and refused
    examples with their reasons (checked against the real executable by
    `OpenPromptExamplesAgainstRealAdmission`), and a request for a hypothesis
    instead of bit patterns. The kind=2 grammar only identifies the operator by
    `phase`; lowering keys change how it runs, not its class. admission.sio is
    unchanged.
  - reasoning: effort `high` and max_tokens 16000 by default (both in the
    manifest); thinking stays adaptive and its tokens are recorded per call
    (`thinking_tokens` in the ledger).
  - reply envelope `{"hypothesis", "B", "proposal"}` (`ENVELOPE_POLICY`, hash in
    manifest): only `proposal`, as compact sorted-key JSON, goes to Sounio;
    the rest is kept in `NNN.rationale.json` with its sha256 and a
    `b_matches_phase` flag in an `ENVELOPE` ledger row, never read by the judge.
    A malformed reply goes to Sounio unchanged and is refused (no repair).
  - Both Claude arms get the same prompt form, model, effort and max_tokens; only
    the ontology facts differ.
  Old flags give old bytes: `--history none` (v1) and the default (v2) produce
  request bodies equal to 04dbe129 (frozen digests in the tests) and identical
  mock-run trees apart from `proposer_source_sha256`.
  v3 smoke estimate (per-arm 8, rounds 3, 48 calls): expected about US$2.1
  (4,000 output tokens/call assumed, thinking included), worst case US$8.1
  (every call at 16,000), under the US$20 cap. A full v3 run (192 calls) has a
  worst case above US$20; BUDGET_STOP ends it at the cap.
- v4, `claude-api-proposer.v4:<model>`, is opt-in (`--prompt open-v4`; keeps the
  v2 history and dedup, refuses `--history none`). It keeps the v3 envelope,
  reasoning defaults (effort `high`, max_tokens 16000) and the same form in
  both Claude arms. Changes, all on the proposer side (admission.sio
  unchanged):
  - prompt (`OPEN_V4_PROMPT`, hash in the manifest): still states the kind=2
    grammar and what each lowering key does, and requires every proposal to
    set at least one lowering key away from the unvaried defaults
    (lane_stride=1, lane_offset=0, load=1, layout=0, unroll=1, optional
    sign_mode/pack/vector_load omitted or 0). It does not include a fully
    filled JSON template, the 1024-code / 32-class census, or the phase
    examples (1, 64, 32768) that became anchors in the v3 smoke. Intra-round
    history still carries Sounio class tags, and also the lowering keys of
    earlier proposals of the same arm. The context sha256 is given as an
    immutable value, not as a copyable object.
  - local consistency: if the envelope contains `B` and `b_to_phase(B)` is
    not `proposal.phase`, the proposer records `B_PHASE_MISMATCH` in the
    ledger, writes `NNN.local-refusal.json` (authority
    `claude-api-proposer.v4`, not Sounio), does not call admission, and does
    not make a replacement API call. The proposal remains in the batch hash.
    The check is not applied when the envelope is missing or has no `B`.
  Old flags give old bytes: `--history none` (v1), the default (v2) and
  `--prompt open` (v3) keep their frozen request digests.
  v4 smoke estimate is produced by `estimate --prompt open-v4 --per-arm 8
  --rounds 3 --budget-usd 20`; worst case stays under the US$20 cap for this
  geometry (48 calls, max_tokens 16000).
- Tests: `claude/test_claude_proposer.py` (CI step "PIREUS Claude proposer
  condition"), with no network and no key.

Estimate for a full run (`estimate` with the defaults: 192 calls):

| | value |
|---|---|
| expected input | about 200k tokens (request bytes / 3.5, feedback grows by round) |
| expected output | 1,500 tokens/call, 288k tokens (thinking included) |
| expected cost | about US$3.3 |
| worst case | US$17.1 (every call at max_tokens 8192), under the US$20 cap |

An offline mock full run (2026-11-20, real admission + atlas, synthetic
usage) produced 288 receipts and 9 batch hashes: the mock's fenced replies
were refused with PROPOSAL_FORMAT and every admitted operator was N2. Mock
replies are not model behaviour, so this checks the plumbing only.

## R740 pilot

`r740/R740_PILOT.md` is the plan for `gpuorangefs-multi-r740-proxmox` (RTX A5000, sm_86, and Quadro RTX 8000, sm_75). It is a separate launcher: no Spark host fence, no `pireus-spark-pair` lease, no `check_pair`. The Spark files named in that plan are unchanged. The gain rule is per architecture, with the same 50000 ppm median and positive bootstrap bound, and its hash was frozen before any r740 timing file existed. The amd64 SIF digest is the one Infra Sounio declared; the launcher hashes the file before srun and refuses on a mismatch. That file has not been re-hashed in this tree.

Use pilot.py prepare with --run, --engine, --context-engine, --fixture-engine,
--parity-engine, --gain-engine and --evidence after committing the source. Run
pilot.py run --run in remote tmux. pilot.py report --run reports partial state
without claiming completion. The current pilot is not started while integration
validation remains open.
