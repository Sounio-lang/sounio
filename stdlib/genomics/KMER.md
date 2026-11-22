# Checked streaming DNA k-mer histogram

Implementation: `genomics::kmer`. This module counts overlapping windows of
canonical DNA bases A=0, C=1, G=2, T=3, for k from 1 through 4. It does not
perform alignment, sequencing-error correction, strand canonicalization,
reverse-complement merging, motif inference, or probabilistic base assignment.

## API

- `kmer_new(k: i64) -> KmerState`
- `kmer_push_base(s: &!KmerState, base: i32) -> bool`
- `kmer_push_chunk(s: &!KmerState, seq: &[i32;256], n: i64) -> bool`
- `kmer_probability(s: &KmerState, index: i64) -> KmerProbability`

The mutation and probability functions declare Mut, Div, Panic effects.
Only the first n entries of a chunk are read. Valid chunk lengths are 0..256.
A zero-length chunk leaves a valid state unchanged.

The state has i64 fields status, k, n_bins, total_windows, rolling_index,
valid_run, accepted_bases, ambiguous_bases, and one counts:[i64;256] histogram.
n_bins is 4^k. Bins use left-to-right positional base-4 encoding. Each accepted
base updates rolling_index = (rolling_index*4 + base) % n_bins; valid_run
saturates at k. A window is counted once valid_run reaches k.

Any base outside 0..3 is an ambiguity break: ambiguous_bases increments,
valid_run and rolling_index reset to zero, and no window is added. The next
window requires k new canonical bases. accepted_bases counts only canonical
bases. Chunk boundaries do not break a run; independent sequences must use
fresh states or an explicit ambiguity break between them. There is no special
ASCII interpretation: encode A/C/G/T before calling this module.

## Mutation status

| Status | Meaning |
|---|---|
| 0 | Valid state and last operation accepted |
| 1 | Constructor received k outside 1..4 |
| 2 | Chunk length outside 0..256 |
| 3 | Invalid state header or invalid selected target bin |
| 4 | i64 counter or selected bin would overflow |

All errors are sticky. A subsequent push returns false without changing an
already failed state. Construct a new state to start again. For invalid k,
the constructor retains k, sets n_bins=0, and initializes every other numeric
field and every count to zero, with status=1.

Invalid chunk length changes only status, before any array access or
histogram/counter mutation. A valid-length operation checks the state header
before accessing its sequence. For each base, all counters and the selected
bin are checked before any base mutation. Overflow leaves that base unconsumed,
with only status changed to 4. Earlier bases of the chunk remain accepted;
there is no chunk-wide rollback. The caller must retain its own replay position
if resumption from a partial chunk matters. The state cannot resume after a
sticky error.

The bounded header check verifies k, n_bins, valid_run, rolling_index, and
nonnegative/consistent scalar counters. rolling_index is also bounded by
4^valid_run before the first complete window. A target bin is checked to be
nonnegative and no greater than total_windows before an increment.
Public fields permit caller mutation: these checks do not certify the sum of
all 256 externally modified bins, nor reconstruct a stream history. A state
must originate from kmer_new and be mutated only through this API. Arbitrary
serialized or externally edited histograms require separate full validation.

## Probability status

KmerProbability contains status:i64 and value:f64. Read status before value.

| Status | Meaning |
|---|---|
| 0 | value is count[index]/total_windows |
| 1 | Failed state, invalid header, or invalid selected count |
| 2 | Index outside 0..n_bins-1 |
| 3 | No complete window observed |

The payload is zero when unavailable. An empty histogram is not a probability
distribution; no artificial uniform mass is supplied. Probability lookup does
not mutate counts or total_windows. Integer counts remain exact until a
checked overflow; conversion to f64 is rounded for counts beyond the exact
integer range of f64. The probabilities are empirical frequencies, without
pseudocounts, uncertainty estimates, or a posterior interpretation.

## Execution and memory contract

One fixed 256-bin i64 histogram, no heap allocation and no sequence-sized
scratch. Chunk processing is O(n+k), with bounded k<=4: the header is checked
once per chunk and each base update is O(1). Public single-base calls repeat
the bounded header check. Calls can stream more than 256 bases by supplying
multiple chunks, preserving windows across chunk boundaries.

Independent C++23 window enumeration and native Sounio fixture execution are
the intended validation route. This contract documents implementation behavior;
a source edit alone is not a claim of native execution, full-corpus acceptance,
or a measured speed/RSS improvement.
