<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-275.readme
authority: repo_only
audience: users
last_validated: 2026-10-25
validated_by: Node metadata selftest
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-275.readme
-->

# SIO-275: stdlib E2E JSON metadata in Node

[Notion SIO-275](https://app.notion.com/p/4043f37f937181aaa594c180e560d7bb)
tracks this operational metadata change under the massive stdlib epic.
[SIO-185](https://app.notion.com/p/4043f37f937181bf9309ee06a61bd790)
remains owned by Claude for the separate timeout measurements; its scope and
state were not changed.

The embedded Python in emit_json is replaced by a Node helper. The runner's
selection, annotations, check/run invocations, timeout handling, status counters
and final exit policy are byte-identical outside that function. The helper does
no scientific calculation and no Python process is used to validate it.

## Preserved metadata contract

The schema remains sounio.stdlib.e2e.result.v1, with the same context, totals,
results, failures and ignored projections. Empty or non-eight-field rows remain
ignored. Detail base64 retains the previous non-strict standard-alphabet/padding
behavior, followed by UTF-8 replacement; the TSV file itself is decoded as strict
UTF-8. BOMs in detail text remain present.

Non-strict base64 edge semantics were reviewed against the
[CPython 3.12.3 implementation](https://github.com/python/cpython/blob/v3.12.3/Modules/binascii.c#L366-L484).
This is source inspection, not an executed Python comparison. The runner emits
canonical base64 and decimal counters/process statuses; all such signed64-bit
integers are retained as exact JSON number tokens. Pretty-printed JSON retains
ASCII escaping, two-space indentation and the final newline.

UTC timestamps keep the previous ISO format: no fraction for an exact second,
otherwise six fractional digits ending in Z. Node's millisecond-resolution
wall-clock timestamp is zero-padded to that format; it does not assert additional
clock precision.

## Validation and limits

Run:

~~~bash
node scripts/stdlib/test_emit_stdlib_e2e_result.mjs
~~~

Ten selftests cover empty/canonical and non-strict base64; invalid padding;
invalid UTF-8 replacement and BOM retention; detail escaping; all eight fields;
pass/fail/skip and ignored metadata; malformed field-count skipping; integer
tokens beyond binary64's exact range; UTC formatting; fresh CLI output and
invalid-UTF8 refusal; and the unchanged runner outside emit_json.

The committed historical artifact at
artifacts/stdlib/stdlib_e2e_result.v1.json is only a format fixture. Rebuilding TSV
from its533 rows and emitting with its original timestamp produces exactly the
same JSON text. Its old totals and timestamp are not a current E2E measurement.

The focused metadata selftest is recorded here. The coordinator will execute
the actual nonempty E2E on the final scientific integration, with fresh JSON and
its own source/compiler/job receipts. This standalone change does not claim that
the full stdlib suite is green, qualify the timeout policy or reuse the old
committed JSON as current output.

The selftest also runs from an extracted source packet without a .git directory:
it checks a pinned SHA256 of the unchanged runner prefix/suffix. Supply the
historical fixture through STDLIB_E2E_JSON_FIXTURE when that artifact is outside
the extracted source tree. Its SHA256 is checked before parsing, so an arbitrary
JSON document cannot substitute for the declared historical fixture.

## Integration measurement

The first allocated integration measurement17578 produced fresh valid JSON for
all627 selected test paths:315 pass,99 fail,213 skip, runner status1. This is
a failed measured suite, not a pass inferred from the historical fixture. Its
selected source packet omitted tracked Pireus data inputs, so it is not treated
as a complete repository-input baseline. A separate full tracked-tree replay
uses unchanged source3c7 and the original runner policies; its result remains
pending until reconciled. The integration epic retains both receipts separately.
