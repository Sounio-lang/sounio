<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.readme
-->

# Stdlib inference and scientific rendering: evidence ledger

This sprint extends the native Sounio computation-to-figure path from remote
`main`. C++23 is the independent numerical reference; Sounio is the production
implementation. Raw receipts retain failed and superseded attempts alongside
accepted focused runs. A successful focused run does not establish a full corpus
pass, merge acceptance, calibrated inference or a universal performance result.

## Current accepted checkpoint: final three-pair measurement

The current numerical acceptance is frozen source `6ffc1a5bd8eaebeef2d15ccced830bfbc832a648`, library NB2 SHA `7d3fd3f8603fdb23bce3e442795e26e869327f0547a90fab7ceda2edb25aafc2` and source-built compiler SHA `d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6`. Full gate17020 and nested pilot17026 passed, with prepared input catalog unchanged. The pilot has16/16 available datasets and16 complete traces; it is not completed inferential calibration.

Final jobs17030/17031/17032 and independent C++ analysis17033 all completed0. Each pair used the same one-physical-core allocation, a2GiB memory reservation,45-minute job limit and1260-second native timeout. AB/BA/AB order alternated. All six executions preserved scientific stdout,180 hits out of199 replicates, plus-one p=.905 and final RNG state. Preflight and the historical single B199 pair are excluded.

| Pair | Before seconds | After seconds | Before RSS KiB | After RSS KiB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 956.85 | 467.35 | 373568 | 356004 |
| 2 | 957.69 | 466.26 | 373568 | 356564 |
| 3 | 959.81 | 465.63 | 373056 | 356884 |

The C++ analyzer reports median elapsed957.69→466.26s and ratio of medians `2.05398275640200746357`. Maximum RSS373568→356884KiB passes the no-more-than5-percent regression target; the time target is also true. This is a source-bound whole-program measurement for n256/alpha0/case0/B199, not an isolated-kernel timing, causal superiority claim or population performance guarantee. The driver exposes final RNG/hits/B, not uniform/proposal totals. Historical B31 measurements were replayed independently in16995 without pooling.

Bounded qualification job17048 was submitted only after root reviewed both actual target booleans. It actually started at2026-10-24T20:30:11Z with its original4GiB reservation unchanged. Its hard limit is23h45; the ledger conservatively reserves900 seconds for the earlier pilot and85500 for the round. Pilot elapsed110 seconds is separate from that allocation accounting. Submission does not imply start, completion or calibration. The old campaign remains separate and unavailable/partial outcomes are retained.

The sections below retain earlier checkpoints and their original pending/failure descriptions as history; the current acceptance above supersedes their current-state wording. The [final performance bundle](performance-final03/README.txt), [C++ report](performance-final03/REPORT.tsv) and [qualification start bundle](qualification-round-start/SCOPE.txt) retain their own immutable source/receipt identities. Functional integration PR3075 and compiler PR3080 remain subject to their actual exact-head CI and merge queue states; successful native evidence is not a merge claim.

## Implemented and measured

- **Paired NB2 likelihood:** checked local mean/dispersion increments preserve
  signed differences lost by subtracting separately rounded objectives. Three
  original unavailable campaign cases have independently negative fixed-point
  likelihood ratios. The implementation retains their sign; it does not clamp
  negative likelihood ratios to zero or omit failed replicates.
- **Nested profile bootstrap:** explicit finite embedding validation, separate
  warm-start storage and ordinary fit convergence checks coordinate null and
  alternative fits. Exact binary64 design reconstruction is an operational
  witness, not a proof of real model nesting or of prediction associativity.
  Full B2 seed/data concordance and fixed-alpha/incomplete-output controls are
  distinct from B199 qualification.
- **Genomics normalization:** reusable result storage, checked log status and
  strided views add explicit capacity/validation semantics. Three paired
  20000-by-512 trials show no measured speed or RSS improvement: median elapsed
  35.50s before and 35.69s after, median RSS 80384KiB for both.
- **Streaming line reduction:** caller-owned state commits only after a chunk
  passes preflight, can drain/reuse output buffers, and preserves global point
  indices. Independent geometry checks cover chunk boundaries and gap runs.
  The selected 2000001-point alternating-gap experiment took 1.79s/452608KiB
  before and 1.39s/62464KiB after. Both versions succeeded; these are measurements
  of one workload, not a general speedup guarantee or an old arena failure.
- **Scientific overlays:** opt-in two-layer panels render actual fit diagnostics
  through SVG and CPU Canvas, with shared series colors, clipping, text bounds
  and all-input preflight. Native numeric coordinates precede pixel projection;
  C++ checks geometry and rejects perturbed bindings. No pixel parity or human
  acceptance claim follows from structural checks alone.

## Performance and qualification boundaries

The selected n256/alpha0 B31/maxfits64 paired pilot improved from 154.51s to 138.28s
before seeded null-replica initialization. A later paired pilot comparing the
original path to the seeded null-replica candidate measured 156.49s to 128.64s.
Each pair preserved its recorded scientific result and RNG state. These are
separate pilot pairs, not an isolated causal comparison between the two new
candidates. B199 three-pair medians and the 2x target remain pending.

The historical B199 campaign retains unavailable cases and incomplete cells.
It cannot be pooled with B31 pilots or the new nested/traced implementation.
`assumptions_assessed`, `global_optimality_assessed` and
`finite_sample_validity_assessed` remain false. Neither seeded search nor a
stable local contrast certifies a global likelihood optimum.

## Custody and replay

- `inference/` contains immutable diagnostics, C++ concordance, normalization
  receipts and paired pilot receipts. Its own manifests bind the retained files.
- `reducer/` binds the streaming reducer experiment and independent controls.
- `viz/` binds the scientific overlay render/geometry controls.

Source and compiler identities, exit statuses, resource receipts and Slurm IDs
inside each bundle identify the actual execution. Generated ELFs and full source
archives remain on cluster scratch; they are deliberately excluded from Git.
Later integration, CI, tracing, campaign-report and qualification receipts must
be recorded separately rather than rewriting these historical results.

## Current-main compiler refresh

`compiler-main29ce/` records source commit
`29ce2abb281e7842881f7570a47934f890de70d2`, Slurm16873 and the fresh native
compiler hash `8c630110db238a38ad345fd782585d525b3a8d438276795cf5b253373824d31a`.
The build completed in7:18.69 wall time, with735920KiB maximum RSS. Two positive
native borrowing/callback probes completed; seven negative borrowing/private-call
probes were rejected as expected. The negative rows retain actual exit1, rather
than a transformed success code. This compiler differs from the earlier
`ee0f341e...` controlled-pilot compiler. Its focused compiler probes do not
establish scientific integration acceptance; that requires separate gate receipts.

`benchmark-drivers/` preserves the original before/after B199 drivers, the two
paired B31 pilot drivers, and the original transformation recipe. The B199 after
driver is preserved input, not evidence that an after B199 benchmark was executed.

## Complete numerical correctness gate

`functional-correctness4f421/` records Slurm16870 on the immutable source commit
`4f4210232bd57e3058651967e158d09c916b5508`, using the controlled `ee0f341e...`
compiler. Its final STATUS is0: native fixtures, independent C++ concordance,
mutant rejection and final source/compiler hash checks passed. This repeats the
previous gate after correcting the missing quadmath link and annotation order.
The complete phase table is readable directly; compressed receipts retain stdout,
stderr, resource measurements, executable identities and hash verifications.
This gate covers that numerical branch, including normalization and legacy
rendering; it does not include later overlay, tracing or marker-size extensions,
and is not B199 qualification. The separate combined-current-main gate remains
its own acceptance boundary.

## Combined inference, normalization and rendering gate

`combined-current-main/` records Slurm16875: final STATUS0 on the integrated
source archived at commit prefix761c2f71ca and compiled by the freshly rebuilt
main29ce compiler8c630110.... The actual input manifest, phase table and compressed
receipts bind that snapshot. All recorded native fixtures, C++ concordances,
negative controls and final source/compiler hash checks passed, including the
streaming reducer, layered scientific diagnostics, canonical bitmap rendering,
count/ODE workflows and the new numerical/normalization contracts. This is the
combined functional gate; later tracing and marker-sizing code require their
own renewed acceptance. Neither this gate nor the earlier B31 performance pilots
establish calibration or the B1992x target.

## Record owner, validated instrumentation and readable campaign figure

`compiler-record-owner/` retains a discriminating compiler regression and its
fix. A nested-if binding lost the nominal record owner; inline field projection
could read an adjacent boolean as alpha, while a typed borrowed helper read the
correct unchanged record. Saving the incorrect inline alpha was a diagnostic,
**not a scientific fix**. The recursive owner join now requires compatible known
layouts from both arms. Original failing witnesses pass with compiler92d2108e;
Slurm16899 runs the original unannotated traced/untraced B2 programs and checks
both against independent C++. Slurm16913 passes the composed current-source f64
compiler gate, including the permanent owner witnesses and incompatible-type
rejection. Failed invocations and superseded diagnostics remain labelled.

The ten original work metrics count actual fit/IRLS/QR/profile/rank and full-PMF
API callsites. They are not a flop count or an inferred sampler cost. The B2
original passes equality off/on, resets, bounded failures, snapshots and actual
RNG checks. OpsTraced16903 and final-summary16910 validate the traced generator,
its work rows, unavailable bounded traces, old-receipt rejection and unchanged
legacy generated datasets. They do not manufacture a passing final CENTRAL
receipt or constitute a completed qualification campaign.

`campaign-report/` binds the frozen original seven READY first batches, missing
cell masks, the portable raw asset, native scientific coordinates, SVG and actual
Canvas PNG. Count classifications and elapsed-time medians are independently
checked before projection. The graphic states partial/unassessed status and
includes early termination in unavailable-class times. The opt-in sized scatter
API accepts square radii1..4; the report uses radius3. The shared conservative
edge guard fixes valid marker footprints previously discarded before integer
projection. The failing-before witness, exact pixel/geometry controls, unchanged
radius1 legacy replay, zero-output preflight failures and three report mutants
are retained. This is execution plus agent visual inspection, not human approval
or calibration.

## Final cache integration and compiler self-check scopes

Commit `82b4316a20` adds separate cached ranking owners and a shared private difference core. Full-PMF fallbacks preserve signed contrast/status semantics, all normalizer metadata and the two-call contract on cache hits. Counts 0 through 63 use exact alpha-bit keys; count 64 and above bypass the bounded cache. The legacy scalar and difference owners retain their lightweight storage. New native controls and independent C++ far-PMF concordance are mandatory in both the central gate and the qualification receipt contract.

The selected B31 pilot16968 measured 154.90s before and 76.45s after, preserving scientific rows and RNG state (ratio 2.02616088947024198825 from the C++ analyzer). This is one selected pilot, not a B199 three-pair result. B199 preflight16982 passed all199 refits and the C++ completion check (180 hits, p=.905); its7:45.42 resource receipt is outside the measured comparisons. Three real pairs16988/16989/16990 execute serially with alternating order, followed by C++ analysis16991; their results remain pending. The fresh compiler build16983 generated ELF SHA `d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6`, then its guard launcher failed by applying chmod before compiling a fixture. This failure remains preserved. Guard-only retry16992 passed two native and seven negative controls using the unchanged compiled ELF; no rebuild was needed. Final central gate16994 is bound to frozen source `257fb0d0a25abef4633bcbe7ac8163f1b2a9983d` and that guarded compiler. Earlier pending gate submissions16984/16987 were superseded before execution to align the status contract and prerequisite retry; neither produced a gate result. No calibration or final performance acceptance follows from submission.

`combined-owner-viz-precache/` preserves passing full gate16933 on source `8d4da7665c`, before the final cache change. `compiler-selfcheck-scopes/` preserves baseline16973, the discarded effect-copy attempt16976, isolated borrow bridge16979 and integrated passing self-check16981. The baseline reproduced the same five errors with old and owner-corrected compilers; the final narrow source repair declares already-used effects and moves the existing raw-pointer cast inside a private borrowed bridge. The checker criteria and expected gen2 rung remain unchanged.

## Conservative aggregate cancellation correction

Additional adversarial audit16996 reproduced an available rounded zero for Poisson rows with contrasts `[-1e-300, +1, -1]`, in both compiler identities and three aggregate APIs. Exact binary rational classification16997 confirms the nonzero negative mean. Its corrected report supersedes an inconsistent auxiliary boolean in16996; raw native outputs remain preserved. The shared finish now returns unavailable5 when nonzero rows collapse to a zero sum. Mixed-row exact cancellation is also conservatively unavailable without an exact-sum certificate; pointwise-zero rows remain available0. The summation formula and nonzero operation order are unchanged.

Focused job16998 passed all15 native fixtures,24 new aggregate outputs through six APIs, exact-rational rounding-cell checks and the prior likelihood/cache/Poisson/B2 references. The identical new fixture compiled against source1132 and failed natively with exit1, then passed with the guard. `aggregate-cancellation/` binds the original diagnostics, corrected rational classification, failing-before/passing-after controls and independent references. Code commit `9ba1db5ae7` and frozen source `956c6b26a5ebe18ea8910aad2b72b57952078598` contain the correction and mandatory gate labels.

Full gate17009 runs that frozen source with the guarded d74 compiler; earlier gate16994 remains a separate pre-guard checkpoint. Final benchmark packet `profile_B199_final03` uses both variants compiled with d74 and new candidate7d3fd3; build/preflight17010 is pending. Its three real comparisons require both successful preflight and the new central gate. Old packet final02 produced one historical pair16988: before15:58.02/RSS373376KiB and after7:46.95/RSS356564KiB, with byte-identical scientific stdout and RNG. Remaining old pair/analysis jobs were held after the defect was found; that single pair is not pooled with final03 or counted as three-pair acceptance.

The component merge monitor is held so PR3074 cannot publish its pre-guard mean implementation. Draft integration3075 contains the complete corrected path. Compiler PR3080 remains separately subject to exact-head CI; its cancelled Linux native job requires a real retry after the enclosing workflow permits it. No merge, calibration or final performance acceptance is implied by these checkpoints.

## Prepared catalog and launcher failure boundaries

Gate17009 completed all numerical/native/reference outcomes with0, but the subsequent genuine receipt was rejected with65: launchers had materialized two declared derived prebuilt inputs after the initial catalog. `input-catalog-preparation/` preserves the original receipt, actual failed readiness recheck, two-file catalog delta and operational control17016. That control verifies the gzip/SHA/ELF/stamp chain and a stable complete catalog after warm materialization. No rejected receipt was rebound to a newer source.

Commit `830e6ba1a0` prepares the declared derived payloads before the initial manifest and compares the complete final catalog, with both rows required by qualification. An additional observed launcher defect captured0 from `if ! command` instead of preserving a failed materializer status. Commit `6ffc1a5bd8eaebeef2d15ccced830bfbc832a648` preserves the actual status through the else branch. Focused17019 passed216 launcher cases using real C++ ELF controls: bad prebuilt/no override0 before to78 after, valid override behavior and strict identity checks preserved. `materializer-exit-status/` keeps the failing-before/passing-after evidence and the earlier missing-utility operational failure17018. These ELF controls test CLI status propagation; numerical acceptance comes from the separate native/C++ gates.

The partially executed17017 was superseded before acceptance to include that boundary repair. Current full gate17020 uses frozen6ffc in `combined_final_cache04`, with source preparation and complete-catalog checks. The numeric library7d3fd3 and compilerd74 are unchanged from passing focused16998 and preflight17010; final03 benchmark inputs remain frozen. `nested_qualification_final03` is prepared only, tied to17020, complete catalog equality and a future genuine root receipt. Pilot and real paired measurements remain pending.

Compiler `compiler-gen2-ci/` retains exact-head run40115178059/job120934877273: expectedrun/reachedrun, gen1 SHA equal to locald74, gen2 SHA9a847402bbc4afc14600c7af416de76c85cd009f40a37a54bcb421d0ca0bb10a executing its banner, and the subsequent gen3 arena-full exit181. The compiler PR3080 currently has41 completed checks without failures and is in the GitHub merge queue. This establishes the executed gen2 rung; gen3 and fixed-point equality remain open.

PR3074 subsequently completed its already-existing queue entry and merged as ebc5eccf38088d7f436d50aa04a95f45b0e0cb75. Stopping the component monitor had not removed that entry. The corrected aggregate guard remains in integration3075. Remote main69e7842e was incorporated; after resolving only the duplicate placement of the font gate, all tracked canonical inputs under bin/scripts/self-hosted/stdlib/tests/examples are byte-identical to6ffc. The current frozen scientific checkpoint remains explicit.

## Accepted functional catalog and nested pilot

Canonical gate17020 completed0:0 in21:07 on frozen6ffc and compilerd74. Preparation, original input hashes, complete final catalog equality and compiler checks all passed. `functional-final-6ffc/` preserves actualRESULTS SHA d058e284ae30666e1d51d52fd8d6bd579954c001523cbac61c1ffc69e96efbe2, initial/finalINPUTS SHA c15be33cd683af0e625888d5dc284d0170935bd5646e934d776866e260454fb9, full text receipts, genuine seven-field receipt and readiness0. The current catalog was independently recomputed and remained equal before and after the pilot.

Pilot17026 completed0:0 in1:50. The executed C++ collector reports16datasets,16available,0unavailable and16complete traces with no partial prefix. Observed native counters include1024fit attempts,7936count draws,45351profile evaluations and1101488tracked full-PMF rows. `nested-pilot-6ffc/` preserves raw cases, generated driver, actual summaries, source/receipt/ELF identities and monitor checks. The auxiliary monitor first used the wrong cwd for a relative manifest; that operational failure is separated from its passing retry. These are computational-coverage results; calibration/global-optimality assessments remain open.

Final B199 pairs17030/17031/17032 and analysis17033 are now submitted serially. A short metadata checkpoint17029 replaced an expired Slurm dependency on the already-completed pilot. Effective memory requests are uniformly2048MiB, while45-minute allocations,1260-second native timeouts, source/data/compiler and physical-core binding remain unchanged. This is an allocation budget, separate from measured native RSS and cgroup enforcement. Pair1 completed with scientific stdout/RNG byte-identical; final medians and target acceptance wait for all six actual rows. Old pending16989/16990/16991 were cancelled only after preserving their unexecuted superseded states. Historical-summary16995 waits until after the final analysis. No old pair is pooled with this series.

A later remote-main change conflicted with the same compiler closure-checker site and removed PR3080 from its queue for merge_conflict. Both branches incorporated mainbe761447 and retained the validated scoped-borrow bridge. Narrow compiler head613810f871 has every self-hosted source file byte-identical to validated92c. Integration headb7cc717864 retains identical compiler CLI, scripts/lib, science gates, self-hosted, stdlib and tests relative to6ffc; the additional upstream bin/sounio-coord utility is outside that numerical execution path. Canonical qualification remains pinned to the actual6ffc snapshot rather than rebinding a receipt to the active branch. Exact-head CI and merge acceptance continue independently.
