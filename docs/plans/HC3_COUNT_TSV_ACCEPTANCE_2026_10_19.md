<!-- docs:meta
topic_id: repo.docs.plans.hc3-count-tsv-acceptance-2026-10-19
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.hc3-count-tsv-acceptance-2026-10-19
-->

# HC3, incremental counts and scientific QC figure — native acceptance

> Current checkpoint: main3 with remote main ea36493 incorporated; Slurm15633 accepts105/105. Original receipts are retained below.

Date:2026-10-19. Worktree:`/workspace/sounio-sprint-stats-viz-main2-20261019`.
Base:remote main`15a1983fd8d9bb4730f2fd6ca72e50f5e0c88d0f`.
This closes the numerical/consumer acceptance left open by implementation
commit`36f3b26ccc87c7582410760ca2b43ba9a40c6837`. The broader stdlib sprint continues.

## Native result

Forge/Slurm **15625** (`hc3-counts-full-20261019-v1`) passed **105/105** scientific
entries, zero failures/skips/errors, exit0:0, elapsed3:25. All101 prior entries
remain included; four new entries cover HC3 reference/status, incremental counts
and a counts-to-QC scientific figure. The runner discovers210 physical files
because each selected entry has an identical basic_math alias;105 are selected.

Snapshot SHA256:`dd084415159e0cd3338b98fb569998c061cb2fa95d90972b3a62917c06263ea8`.
Runtime manifest SHA256:`4e79cd8d904e7158f950a9dc2e5b7f972a27001e75c49c882868f2c74df56485`.
Runtime receipt asserts uid/gid1000 and clean-v1. Committed Madaros ELF SHA256:
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
The shipped compiler was materialized and verified, not rebuilt from source.
The legacy harness banner says lean_single/absent; it is not the compiler identity
used by bin/souc. Per-entry native logs record the actual modular compiler path.

Slurm batch MaxRSS7104836K and TotalCPU06:36.800 are whole-job observations.
The shorter elapsed time relative to the earlier101-entry run is not a paired
performance comparison and does not establish a speedup. No new kernel resource
improvement is claimed here.

## Independent HC3 concordance

`linear_hc3_reference.cpp` uses quad two-pass modified Gram-Schmidt on the raw
design, independent of production's equilibrated/pivoted Householder QR. A portable
`linear_hc3_fixture.cpp` emitter includes it by relative path. Root freshly compiled
and executed both in C++23 with libquadmath; generated fixture matches the frozen
Sounio fixture byte-for-byte. The emitter was added as a post-snapshot analysis
sidecar; it does not change any Sounio input in the accepted snapshot.

All24 reference cases are exercised natively, including heteroscedastic data,
high leverage, unit changes, extreme scales, row/column permutations, dimensions,
rank failure, inactive poison, nonfinite inputs, covariance representability and
maximum256x8 design. The maximum case uses compact input loops and retains every
active covariance entry and all256 leverages.

Nonzero coefficient checks are relative1e-9. SE/diagonal HC3 checks are relative
1e-7 without a1+abs floor; off-diagonal tolerance uses SE_i*SE_j as the covariance
unit. The perfect-design case uses closed-form influences conditional on the
actual reported residuals, with point coefficients checked independently. It does
not overwrite computed residuals with zeros. Zero-response covariance/SE must be
exactly zero. Metamorphic checks supplement independent numeric references.

The accepted estimator is covariance-only HC3. No cluster covariance, t/p/CI,
coverage study or assessed scientific assumptions is claimed. Rounded leverage
and representability limits remain as documented in ROBUST_COVARIANCE.md.

## Incremental parser and QC concordance

`count_tsv_reference.cpp` uses whole-token/prefix std::from_chars, independent of
production's incremental decimal multiply/add. Its generated Sounio fixture is
byte-identical to the native input. Twenty-four input cases run with chunk sizes
1,2,3,7,256, plus dedicated row backpressure, CRLF splitting, EOF, sticky errors,
invalid lengths/byte values, column counts and matrix integrity controls.

The canonical i64 maximum is accepted;9223372036854775808 is rejected with status5
at absolute byte18 without consuming the offending byte. QC rejects negative or
overflowing rows without partially updating numerical fields. The256x8 adapter
checks capacity and inactive padding. These are bounded executable cases, not a
claim of exhaustive parser verification or a formal proof.

The module receives caller-supplied bytes. Filesystem input, feature/sample IDs,
headers and unbounded matrix storage remain outside this API. QC may stream more
than256 features while counters remain representable; the normalization adapter
may not exceed256 rows. Blockwise median-ratio is not claimed to be global.

## Native figure

`viz_count_matrix_qc.sio` executes bytes->parser->QC->Canvas/SVG using synthetic
four-feature/three-sample input, chunk3, split CRLF and no final newline. Plotted
values come only from parser/QC output: totals8,15,12 counts and zero fractions
2/4,1/4,2/4. Units, denominator, sample order, synthetic provenance and absence of
biological assessment/intervals are explicit. Casts above2^53 are visibly qualified.
The native fixture also checks scene structure and three specific Canvas pixels;
it does not capture or compare a complete Canvas image.

Root freshly compiled the independent C++23 geometry checker. It independently
parses the raw TSV and verifies six SVG circles with exact pixel coordinates,
order, radius and RGB, plus series metadata and scientific labels. The real native
SVG passed. Three mutated copies (shift, missing mark, radius) were each rejected
with exit2. This checker is not a general XML validator or biological-quality test.

The standalone SVG is the exact first <svg>...</svg> block from the native HTML
log plus newline. Initial line-based extraction also retained trailing HTML;
inspection caught it, extraction was corrected, and checker/controls were rerun.
The delivered SVG passes xmllint. Quick Look's raster thumbnails were cropped and
are excluded from the delivery; SVG is the authoritative figure.

SVG SHA256:`f07611722d2981aab4d942eb0296dff203417502dd917b5241d7fe13dc801702`.

## Discriminating and non-discriminating controls

| Slurm job | Change in isolated copy | Outcome |
|---|---|---|
|15626|HC3 denominator1-h replaced by1, giving HC0 weighting|compiles; selected HC3 reference fails;0/1,exit1:0,5s|
|15627|Overflow threshold digit>7 weakened to digit>8|1/1 remains passing; not discriminating;exit0:0,3s|
|15629|Original overflow branch returns0 instead of error sentinel-1|compiles; selected parser reference fails;0/1,exit1:0,4s|

The first overflow mutation does not bypass the caller's negative-return guard.
Its PASS is preserved as a non-discriminating mutation, not reported as successful
negative evidence. Direct confirmation of its intermediate value was not measured.
The clamp-to-zero mutation avoids that ambiguity and is detected by unchanged
fixtures. These are injected coverage controls, not historical library defects.

## Source identities

- HC3 core:`b788b4cb384e2d9f82098a63264fe913b74c28df4a57dc79f2e70bbba0cf7fb7`.
- Counts core:`ddb0e0b9216f6b5c6535d812cc85caa656ae9b69ffb69f1a0928d2b95fb1ca74`.
- HC3 reference:`301c0949a60cfcdbf093d62be8da145417d91447ab53daf0288fc7863d4bbf1c`.
- HC3 portable emitter:`71604f4e9f040e81a75934021830ab7990ebe83262e7d7e7bebbca3a8ff055da`.
- HC3 fixture:`a688b5fc5b47febaea9d3575373b481ada79a6f2ffb4a5219dc7df5177f2777a`.
- HC3 status:`ad0a181c7aee6c20cdecf0401babec33aa67cc62fc159b2335f4620ab95247fc`.
- Counts reference:`07a401dd6d9deb2513a5d86a0c2e7e7ae058fc0bb4f58567927c07ce313bca21`.
- Counts fixture:`f2ecc2243d6c6251beb5d19ca266f1fab7c4b871b578f854fbca7718c7362b1d`.
- Figure fixture:`c8cb9cbe90a2b52128c57372170aea7396a19b5a0b51e001140f8ae036493b30`.
- Geometry checker:`a0cbd1f518cd7610363ca4be94f86ec7098c7608016cbfbedcf7e02b291bf48d`.

## Integration and next scientific scope

Core implementations are unchanged from36f3b26ccc; this checkpoint closes their
bounded numerical/native-consumer acceptance. Post-acceptance documentation and
portable-emitter packaging are separate from the frozen runtime snapshot.
Full compiler CI, PR/merge and broad statistical calibration remain separate.
Next scientific candidates: cluster score aggregation with an explicit inference
contract, filesystem/metadata count input, and independent negative-binomial
dispersion. The broad math/genomics/physics/scientific-viz sprint remains active.

## Custody isolation and refreshed remote main

The sections above retain the original main2/15a198/job15625 provenance.
During final review, main advanced to ea36493e88efeaa90f6bacabfb47d192d8bf41da.
The original worktree also acquired an unexpected replacement parser oracle
(fe53051cdfc0ea6cbbff211b88059b5e11c805b972d1d26a2a3ad85ee6b182e8) and an
extra genomics_count_tsv_status.sio. Neither belonged to the accepted snapshot or
this swarm. The replacement cloned the production algorithm and could not support
the independence/emitter claims. Its origin was not attributed.

To preserve concurrent work, neither file was overwritten or deleted. A new
worktree /workspace/sounio-sprint-stats-viz-main3-20261019 was created from36f3b26
and merged ea36493, producing merge c108a1cc690667c53b1d130771ba8db79f0c753e.
The accepted references and fixtures were copied directly from the frozen runtime
packet, restoring canonical parser reference07a401 in the new worktree. All ten
listed artifact identities were independently checked again. The extra status
fixture was not transported, packaged or staged. The previous worktree remains
available with its concurrent WIP intact.

The main update changes two epistemic modules and five knowledge fixtures; the
focused105-entry corpus is rerun against the refreshed tree. Full compiler CI and
those additional knowledge fixtures are not implied by the focused result.

### Refreshed-main acceptance: Slurm15633

Run hc3-counts-main3-full-20261019-v1 accepts **105/105**, zero failures/skips/errors,
exit0:0, elapsed3:25. Snapshot7f915b4f9f8f3aa78dfffba591044ec3e336b95a6c18f9fd6219312f1a6dc417;
runtime manifest8c5497686b641a2387733f2d551e5e98dc96f39f9eef58a15685775e31a2288c.
Receipt asserts uid/gid1000 and clean-v1. Batch MaxRSS7201268K, TotalCPU06:37.291;
no performance comparison is inferred. The new native figure passes the same
independent checker and is byte-identical to the original SVG, SHA f07611722d2981aab4d942eb0296dff203417502dd917b5241d7fe13dc801702.
The original controls retain their separate earlier snapshot identities; their
core and fixture inputs are identical to those checked in main3.
