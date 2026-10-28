# Isolated ICU 79.1 complete BIDI diagnostic

This tranche keeps the pristine ICU 79.1 locked acceptance oracle, the official
Unicode18 corpus, its expected values, and Sounio production BIDI source unchanged.
A private ICU build is a diagnostic candidate. Its concordance cannot close the
pristine strict gate or the integrated whole-stdlib gate. No upstream publication
is part of this change.

The driver replays all 861948 official cases (770241 expanded BidiTest modes and
91707 BidiCharacterTest rows). It uses the existing parser and the unchanged
wave3 locked reference. The new collector continues after a disagreement so the
receipt includes complete counts, with at most 128 detailed mismatch rows.
Base levels and complete visual maps are checked for every case; exact scalar
levels use the same mixed-direction ICU contract as the locked reference.
Unidirectional ICU optimized levels remain separately observed and counted.
No case whitelist, tolerance, changed expected level, or corpus subset is added.

## Source diagnosis and boundary

[UAX9 revision52](https://www.unicode.org/reports/tr9/tr9-52.html) specifies the
order X6, BD14/BD15, BD16 and N0. The candidate addresses four behaviors:

* Carry the override flag through isolate initiators and matching PDI (the
  previously isolated two-assignment candidate).
* Identify brackets from their current type after X6; an overridden ON is not
  eligible for N0 pairing.
* Preserve original NSMs until W1 can inherit a bracket's final N0 direction,
  including over ignored BN characters.
* Honor BD16's 63-entry stack. Overflow returns an empty pair list for the entire
  isolating run sequence, including pairs completed before the overflow.

The overflow repair first discovers affected sequences while ICU's original
explicit resolver runs. It then restores original properties and flags and
replays that same resolver, suppressing bracket pairing only for affected
sequences. Canonical synonyms U+232A/U+3009 share a stack value. The candidate
uses additional per-call memory and internal sequence state; this is not a
performance improvement claim or a public ICU ABI change.

The tested adapter calls default logical-to-visual ubidi_setPara with null
external embedding levels. External styled levels, inverse reordering modes,
allocator failure, and other ICU APIs are outside this receipt. They are not
promoted by the official corpus result.

## Gates and preserved failures

* Slurm 19235: collector before qualification, COMPLETED0, expected failures.
  Pristine:101 mixed scalar-level divergences,30 visual-map divergences.
  Earlier two-assignment candidate:93 levels,26 maps. Both execute all 861948.
* Slurm 19247: first build attempt, FAILED1 because make was absent (127).
  Configure and both complete before comparisons succeeded.
* Slurm 19253: three-change candidate, FAILED1, all 861948 executed with 73 levels
  and 10 maps remaining. Its SDK and receipts remain isolated.
* The final run and artifact identities are recorded in RESULTS.md.

The 12 independent SDK controls exercise depths 62, 63, 64, 65, 129, both ASCII and
canonical-synonym brackets, completed pairs before overflow, and an overflow in
an RLI that must preserve outer pairs. Five full-corpus observation controls
corrupt one level, base, visual-map entry, missing observation, or duplicate
observation in RAM. Canonical input and expected files remain byte-identical.

The acceptance gate must reject this candidate prefix with status 2 before
creating its output directory. This policy control remains mandatory even when
the candidate agrees with the full official corpus.

## Reproduction

Run scripts/stdlib/wave4_bidi_icu_complete_candidate.sh with one fresh absolute
output directory under Slurm. Required explicit inputs are ICU_PRISTINE_ROOT,
ICU_TWO_ASSIGN_PREFIX, BIDI_POLICY_RAW and BIDI_POLICY_PLATFORM.
ICU_CANDIDATE_BUILD_JOBS may not exceed allocated CPUs; default4.
BIDI_TOOLCHAIN_LD supplies the recovered toolchain libraries when needed.
The script pins source archive/ubidi.cpp/ubidiimp.h, verifies unique patch
anchors and reverse byte equality, hashes both before SDKs, source inputs,
compiler, Node, make, and all produced SDKs/executables. Each phase has a timeout
and /usr/bin/time -v receipt. Final hash and symlink guards cover immutable inputs.
The worker uses a copied make executable with SHA256
d78b8f1d099fbcfb6f2f49ab87223b9b68fb3956642f92d6ec6de812e8afa965;
all builds and tests execute inside Slurm.

The API views commit 961a8035a3 remains an independent qualified change.
Generic HeapVec and the lowerer alias/field/ref dispatch blocker are not claimed
complete by this diagnostic.
