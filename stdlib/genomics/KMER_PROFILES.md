# Checked k-mer adapters and profile distance

This contract supplements [the streaming state API](KMER.md).

## Sequence core

`genomics::core::dna_kmer_count_checked(&[i32;256], seq_len:i32, k:i32)`
returns the shared KmerState. It checks k1..4 and seq_len0..256 through the shared
constructor/chunk API, including ambiguity breaks. The empty prefix is valid
but does not define a probability distribution.

The existing `dna_kmer_count` preserves Kmer256 (i32 bins plus k). On valid
input it delegates to the checked result; counts from a256-base prefix fit i32.
On invalid input it returns k=0 with zero counts. That explicit compatibility
sentinel differs from valid empty data (which retains requested k). Prefer the
checked entrypoint to inspect why a request failed.

## Profile API

`genomics::embedding` now exposes KmerProfile with public counts[256]:f64,
n_kmers:i32,k:i32,total:f64,status:i64,normalized:bool.

- `kmer_count(&[i32;4096], seq_len:i32, k:i32)` validates0..4096 and1..4
  before copying any sequence data. It reuses one256-base scratch chunk and
  the shared rolling state; windows across chunk boundaries are retained.
- status0 is a computed histogram; statuses1..4 follow the stream's
  invalid-k/invalid-length/invalid-state/overflow codes. Failed outputs have
  empty bins and total0; status remains essential.
- total is the raw number of valid windows even after normalization.
- `kmer_normalize(&!profile)->bool` is idempotent. A normalized profile returns
  true without dividing again. Empty valid profiles return false unchanged.
  Invalid profiles fail before mutation; malformed metadata/mass sets status3.
- Validation checks k/n_bins, finite nonnegative bounded total, finite
  nonnegative active bins and compensated total mass within
  1e-12*(1+expectedmass). Normalized profiles require positive raw total and
  probability mass nearone. Unused suffix bins are ignored.
- Public fields permit weighted/fractional data consistent with these checks;
  validation does not certify integer observations or stream provenance.
  Produced count profiles contain exact small integer counts in f64.

## Jensen-Shannon distance

`kmer_distance_checked(&a,&b)->KmerDistance{status:i64,value:f64}` consumes
both raw counts and normalized profiles without mutating either. It returns
sqrt of the equally weighted Jensen-Shannon divergence **in bits**.

| status | Meaning |
|---|---|
| 0 | Numerical distance available |
| 1 | Failed or malformed profile |
| 2 | Incompatible k |
| 3 | Empty empirical distribution |
| 4 | Arithmetic failure |

Payload is zero when unavailable. Compatibility `kmer_distance` returns -1 on
failure and otherwise delegates to the checked value.

Per-bin contributions use a symmetric nonnegative representation. For close
probabilities,24 positive even-power terms avoid cancellation of opposing KL
terms; more separated probabilities use canonical checked logarithms. Zero
probabilities are handled explicitly. Contributions use compensated summation
and canonical sqrt. No arbitrary positive probability floor is applied.
Distances have the mathematical unit interval bound; accepted floating mass
and arithmetic may differ at rounding scale, so the implementation checks a
small numerical envelope rather than declaring a formal proof.

Reference definition:
[Lin (1991), Divergence measures based on Shannon entropy](https://doi.org/10.1109/18.61115).
The independent C++23 reference uses direct raw-KL long-double logarithms,
rather than the production symmetric identity, including near-equal profiles.
These empirical distances are not alignment, ancestry probabilities, a
sequencing-error model or calibrated uncertainty.

## Preserved legacy sections and scope

PWM/motif routines and private hashed embeddings remain in the source.
The private embedding caller gains guards for dimension1..64 and invalid/empty
count profiles before modulo/indexing. Its remaining projection, zero-norm cases
and uncertainty magnitudes remain legacy/unassessed. They are not certified by
the checked k-mer distance tests.

The integration review verifies preservation of the original PWM/motif/embedding
suffix except for declared status fields, guards and heuristic comments.
Native acceptance is established only by the final restored-source receipt.

## Storage and execution

Streaming uses a fixed256-bin i64 histogram and scalar state; no heap allocation.
The4096-base adapter additionally uses a256-base scratch chunk and a returned
f64 profile. This trades bounded adapter scratch and wider counters for a shared
streaming API. No lower-RSS/speedup claim follows without measurement.
The rolling algorithm avoids rescanning each k-length window, but k remains
explicitly bounded to1..4.
