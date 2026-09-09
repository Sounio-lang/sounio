# Control feedback and next experiment — 2026-09-08

Owner: codex-pireus / continuity-20260906.
State: attachment implemented, paired tokenization qualified, and execution profile frozen at v2; exact-source CI pending; inference not launched.

## Result carried forward

The hardware archive at validation/control-union-hardware-20260908 records
18 distinct native materials from64 occurrences in control jobs11957/11958.
Jobs11963/11964 established finite-fixture parity on both Sparks and18 native
NO_GAIN decisions. Preserve the four comparison confidence intervals per
material. NO_GAIN means the frozen promotion gate was not satisfied; it does
not mean zero effect, invalid arithmetic, or global uselessness of the plan.
Layout conversion remains excluded. These cohorts are not independent samples
of64 distinct plans.

ops/build_control_feedback.py verifies the pinned hardware-audit digest and all
256 artifact hashes before exporting proposal, admission, gain, PTX identity,
and original membership. The packet is a transport of Sounio receipts; it does
not calculate a replacement reward or infer scientific novelty.

Reproduce to a new path:
    python3 tools/pireus/continuity/ops/build_control_feedback.py --output /tmp/pireus-feedback-new.json

The output is exclusive-create. Corrupt artifacts and an altered audit are
refused. The published packet is validation/control-feedback-20260908/feedback.json.

## Separately declared follow-up: feedback-smoke-v1

Question: does explicit measured feedback change the diversity of admissible
lowering proposals relative to an otherwise identical Inkling prompt?

Primary observable: count of native plan/PTX identities absent from this
18-material feedback set, within each eight-request arm. This is material
diversity relative to the measured set, not scientific novelty or gain.
Secondary observables: admission/refusal causes, repeated materials, tokens,
memory minima, and subsequently native parity/gain on distinct admitted plans.

Two arms, eight requests each: unchanged ontology/context without feedback,
and the same ontology/context with the receipt packet. Freeze the complete
request bytes, eight paired seeds, arm order, tokenizer counts, runtime and
image identities before loading. Run sequentially with exclusive Slurm
ownership. This is a descriptive smoke, not a powered causal estimate.
Prior control outputs differed under the same seeds; matching seeds do not
guarantee matching samples.

Keep target701202, dimension16, precision64, evaluation order1, FMA0 and the
existing admission grammar. Feedback is an evidence attachment; it must not
be promoted to ontological truth or inserted as authoritative expected output.
Known NO_GAIN plans remain legal; record repeats instead of silently banning
or repairing them. An invalid attachment must fail before model loading.

Execution profile: uninstrumented TP2 offline, concurrency1, physical cache6144,
context16384, output ceiling4096, early-stop33GiB, protected floor32GiB.
Tokenize the actual feedback requests before freezing. Require each input plus
output ceiling to fit6144; if it does not, declare and review a new compact
projection before freeze. Never silently truncate receipts or reduce output.
Bind BatchSize8 and the feedback digest into the new experiment identity.

Acceptance prerequisites before launch:
- Implement attachment transport and prompt/context separation.
- Verify missing/altered receipt, wrong context, and injected-authority controls.
- Freeze both request sets and exact profile, then verify current source checks.
- Capture pair ownership and memory preflight, use a new immutable attempt root.

Stop on rank loss, guard, hash mismatch or missing completion; preserve the
attempt without automatic retry. No empirical success criterion changes after
observing outputs. Any performance claim still needs the existing native
parity and gain gate on both Sparks against both controls.

## Canonical boundaries

Original pilot remains1/9 cells and32/288 proposals. Failed11956 is retained;
neither these control receipts nor this feedback packet resume or complete it.
M5 new operators and M6 GRPO remain NOT_STARTED. This is follow-up work on the
M4 lowering loop. These18 materials are exposed training/feedback data, not
holdout. V13/V14 remain OPEN. No HTTP/general16K, memory root-cause, universal
floating-point parity, or gain claim follows from this export.

Next executable task: await the v2 source checks, then run the readiness gate
and archive its receipt before publishing the preparation closure. A launch
still requires live pair ownership and memory preflight. Do not modify the
old pilot manifests.

## Attachment implementation receipt

ops/feedback_smoke.py now stages both eight-request arms with identical base
prompts and paired seeds0..7. The feedback arm appends an explicitly versioned
compact projection of18 plan IDs and their five lowering fields, scoped native
ADMIT/NO_GAIN observations, and the full feedback digest. Raw receipts,
confidence intervals and occurrence mapping are omitted from the prompt and
preserved in the attached full packet. This is a declared projection before
tokenization, not an implicit truncation.

Eight local controls PASS across the exporter and request builder. Requests
are staged at validation/feedback-smoke-requests-20260908. That immutable
staging manifest records token budget and execution-profile freeze as false
and refuses existing attempt roots. The later tokenizer and freeze receipts
below establish these subsequent stages without rewriting the staging record.

Paired tokenizer11965 completed0:0 on2026-09-09UTC (the request identity retains
its original20260908 label). All16 items match across ranks. Input tokens are
344 without feedback and808 with feedback; totals with4096 output are4440 and4904.
validation/feedback-smoke-tokenizer-11965 preserves receipts, input, mapping,
launcher log and accounting. ops/qualify_feedback_tokens.py reconstructs the
requests, verifies source dependencies, log-bound receipt hashes, pair identity,
durable accounting and physical6144 budget. No model was loaded during this
tokenizer qualification.

## Execution freeze v2 and remaining acceptance

The current freeze is
validation/feedback-smoke-freeze-v2-20260909/execution-freeze.json,
SHA256 c75f94a648e8dd52c599e5d1ce722888b7c33389e8201c2256e7316aacf5827b,
bound to source91553cbad3f0ef402e9c17e2c2435ff54b4ea11d.
It binds101 runtime files and20 input artifacts. All16 token-request files
and all runtime files are byte-identical to v1; source-bound manifests changed.
The private immutable execution root is
/workspace/.cache/pireus-continuity/feedback-smoke-execution-20260909-v2.

The prior freeze in validation/feedback-smoke-freeze-20260909 remains
unqualified. Its sourcee17959ac1238d3e0ceb5fbd9f94bd117e8ee8446 saw two hosted
runner shutdowns during Madaros compilation. Preserve both failures; their
cause remains unresolved. The new source adds a process/log observer without
changing compiler arguments, return status, timeout, or gate assertions.

CI run34304172437 checks the v2 source. Until its required exact-source checks
pass, ops/check_frozen_feedback.py must refuse readiness. The required checks
are CI Decision, transport-and-archive, and Archived script custody (no runtime
replay). Seven local readiness controls cover source identity, latest-check
selection, artifact integrity, and refusal to reuse v1 acceptance for v2.

A passing readiness receipt establishes source and frozen-input readiness only.
Live host preflight and runtime receipts remain required for inference.
No feedback-smoke generation, hardware result, gain, or pilot acceptance is
established by tokenization, freezing, or CI.
