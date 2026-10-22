<!-- docs:meta
topic_id: repo.docs.plans.scientific-kernel-audit-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.scientific-kernel-audit-2026-10-18
-->

# Next scientific kernels — source audit, 2026-10-18

This is a source-derived audit, not native acceptance. A dedicated subagent
performed read-only exploration; root reviewed the k-mer source. No code is
changed in these domain modules by the KS checkpoint.

## 1. Genomics: checked streaming k-mers

Public `genomics/core.sio::dna_kmer_count` documents k<=4 but does not guard
k or seq_len before its window loops. k<=0 can count empty windows; k>4 silently
drops indexes>=256; seq_len>256 can read outside the input buffer.
The private duplicate in `genomics/embedding.sio::kmer_count` does not validate
each base before positional encoding and computes unchecked 4^k for a fixed256
histogram. Its in-place normalization divides frequencies a second time when
called again, since total remains unchanged. These are source findings.

Proposed shared checked stream:
- k domain1..4, explicit status before data mutation.
- Fixed256 integer histogram, scalar rolling base4 index and valid-run length.
- Chunk API with prefix validation; preserve valid windows across boundaries.
- Explicit ambiguity policy: any base outside0..3 breaks the run, contributes no
  window crossing that base, and increments an ambiguity counter.
- Nonmutating probability view; no repeated destructive normalization.
- Checked total/counter arithmetic for long streams; sticky failure status.
- Migrate the public core adapter and private embedding caller after the shared
  result/compatibility contract is reviewed. Existing legacy Kmer256 has no
  error field; invalid input handling must be stated and a checked API exposed.

Independent C++23 oracle should enumerate raw windows (not rolling state),
compare all integer bins for k1..4, every partition of short sequences,
ambiguities and empty/short prefixes, boundary poison, invalidsizes and
overflow policy. Runtime/memory gains need measurement; O(n) rolling structure
alone is not a measured speedup.

## 2. Physics: isothermal entropy and standard uncertainty

`physics/thermo.sio::isothermal_entropy` uses a short unscaled log series on
vf/vi and an uncertainty expression multiplied by the resulting entropy.
Its equal-volume branch forces a volume contribution to zero, even when the
two endpoint uncertainties are not zero. Other file routines sum relative
uncertainties linearly despite the GUM wording. These source formulas require
an explicit uncertainty-model review.

Proposed narrow replacement: checked entropy change with finite positive
volumes, amount/domain checks, nonnegative standard uncertainties, declared
independence/covariance assumptions, safe log differences and a close-volume
log1p path. Combine sensitivity contributions with a scaled norm. Preserve a
typed unavailable state instead of returning zero entropy/uncertainty on errors.
Independent C++23 log/log1p plus sensitivity oracle; equal and near-equal volumes,
extreme ratios, amountzero with uncertainamount and endpoint-exchange symmetry.

## 3. Mathematics: information theory contracts

`math/information.sio` private functions named Shannon/KL exact use the rational
surrogate 2(x-1)/(x+1) for logarithms. Positive-p/zero-q KL terms are skipped,
normalization is unchecked, and fixed8 storage is trusted. The BSC endpoint
branch returnszero where deterministic channel capacity should be retained;
AWGN substitutes a saturating rational expression for logarithm.

Proposed bounded checked probability APIs: normalized finite nonnegative
vectors, explicit log base/units, compensated sums and canonical checked logs;
KL support mismatch must have an explicit infinite/unavailable state. C++23
raw long-double logs and known identities provide the independent oracle.
These are private paths; public genomics takes priority.

## KS review findings addressed by the current checkpoint

- Evaluate ECDF gaps only after complete equal-value groups.
- Use exact rank cross-products before final floating division.
- One-normal KS requires externally fixed mean/SD, not sample estimates.
- Use theta CDF for small lambda and direct survival expansion for largerlambda.
- Size/nonfinite/reference checks must precede array access.
- Correct tied-data D does not establish continuous-null calibration for ties.
- Finite opposite-sign input needs overflow-safe normal standardization.

No field kernel above is certified as implemented by this audit. The master
sprint retains all three as implementation tracks after KS acceptance.
