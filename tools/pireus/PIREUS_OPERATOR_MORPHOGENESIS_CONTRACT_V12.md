# Pireus Operator Morphogenesis V12 Contract

Status: SEMANTICS_FROZEN

Concept-ID: SOUNIO-PIREUS-PROOF-CARRYING-OPERATOR-MORPHOGENESIS

Semantic-Lane-ID: pireus-proof-carrying-morphogenesis-v12-20260830

## Authority

The only semantic producer is
`stdlib/hardware/pireus/operator_morphogenesis.sio`. The first executable must
exist in Git after Garden commit `1ea2499e82` and before its first result.

The ordered transition is:

    GARDEN
    -> SOUNIO_EXECUTABLE
    -> SEMANTICS_FROZEN
    -> PARITY_OPEN
    -> CLAIM_READY

Lean, Koka, C++, Haskell, target processes, and external reviewers cannot write
V12 semantics or expected results. Python and Rust are forbidden oracles.

## Parent Interface

The executable calls the live frozen v4 cubic forge and requires both its
ordinary validity result and
`pireus_cubic_operator_forge_matches_frozen_semantics`. It imports exactly the
48 by 256 `population.sign_bits` cells. The current v4 source and semantic
identities are pinned by the Garden and by the later first-result receipt.

The v11 frontier and material receipt are lineage and custody parents only.
They supply no sign bit or expected V12 result.

## Canonical Types

The V12 phenotype is a 256-bit normalized sign table. Bit zero means `+1` and
bit one means `-1`. The output lane for pair `(i,j)` is always `i XOR j`.

The V12 phase is the 225-bit interior truth table relative to the pinned
`cd_sigma` table. The V12 genotype is the 225-bit mixed Boolean ANF with one
coefficient for each pair of nonempty bit subsets.

The acting involution swaps lane-index bits zero and one. Its sign-table
pullback is `(q.tau)(i,j)=tau(q(i),q(j))`. An archive row is an exact 256-bit
image, not a digest proxy.

## Canonical Ordering

The 225 diagonal witness cells are ordered by `(i XOR j, i, j)` over nonzero
`i` and `j`.

The initial archive visits cubic child IDs in ascending order. For each child,
the lexicographically smaller of the direct and action images is considered
first, followed by the larger image when distinct. Exact duplicates already in
the archive are omitted.

Each generated epoch uses the same within-orbit order. Epochs are appended in
ascending order. This orbit-insertion order is the canonical archive order used
by the diagonalizer and all digests; it is not a global re-sort that could
invalidate earlier archive-index certificates.

## Generation

For archive rows `F_0..F_(m-1)`, `m<=225`, the generated phase `P` assigns the
`k`th witness cell to the complement of `phase(F_k)` there and assigns zero to
every other interior cell. Unit axes are zero.

The executable must:

1. derive and inverse-check the complete mixed ANF;
2. materialize all 256 signs and the XOR microprogram;
3. verify one original separator per archive row;
4. use archive closure to derive one transported separator per archive row for
   the candidate's action image;
5. compare both orbit images directly against every archive row;
6. append the exact distinct orbit images only after all checks pass;
7. repeat for exactly 16 epochs.

The fixed storage capacity is 128 archive images. The construction starts with
at most 96 images and appends at most two images per epoch, so the structural
bound is `96 + 2*16 = 128`. The diagonal witness capacity remains 225. Any
overflow, closure failure, or incomplete generation is fail-closed.

## Diagnostics

Negative-sign, square-sign, ordered-pair sign-asymmetry, and associator counts
are execution diagnostics. Exact integer probes exercise the canonical
ascending reduction spine. None of these values is a matcher or an
algebraic-law claim in the first executable.

Every epoch binds its input archive digest, phenotype, phase, ANF, separator
records, orbit kind, diagnostics, probes, and output archive digest. The run
digest binds the ordered epoch digests and final archive.

## Result Boundary

The first executable may report `constructive_bounded_relative_novelty=true`
only when all 16 epochs are complete, every original and transported separator
passes, both orbit images are collision-free against the prior closed archive,
all ANF and structural checks pass, and all negative controls are refused.

It must keep false:

- candidate selection and ranking;
- full-space exhaustion;
- GL, gauge, isotopy, or algebra-isomorphism completion;
- algebraic, algorithmic, material, performance, scientific, historical,
  global, priority, patent, publication, and claim-ready novelty;
- target lowering, target cost, and target performance admission.

The generated 256-entry sign microprogram is a future lowering input. It is not
a material implementation.

## First-Executable Rule

The matcher-free source may match structural constants only: 4 bits, 16 lanes,
256 sign cells, 225 interior cells, 48 parent children, group order 2, 16
epochs, archive capacity 128, and certificate capacity 4096.

It may not contain expected initial/final archive counts, orbit distributions,
signs, phases, ANFs, witness bits, law spectra, probes, or digests. Those become
eligible for a frozen matcher only after the first authorized Sounio transcript
has been committed.

## Freeze Boundary

The committed matcher was added only after the matcher-free executable, first
authorized Sounio result, and first-result receipt existed in Git. The freeze
receipt pins both source histories, the exact first transcript, and the exact
matcher replay.

This contract stops at `SEMANTICS_FROZEN`. It opens no Lean, Koka, C++, Haskell,
Kubernetes target, FPGA, or other material execution. A later `PARITY_OPEN`
receipt may authorize those roles to compare against this hash, but cannot let
them rewrite the semantics or expected result.
