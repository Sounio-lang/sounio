/-
  FORMAL_PARITY base for one already-frozen Sounio V13 execution record.

  The input, winner, and output below were emitted by the matcher-free Sounio
  executable at commit 73704f7afed6780c3a317b739cbd35fe94dbe395.  The freeze
  separately records that producer source and the later matcher-only frozen
  source; both hashes are packaged below.  The external gate binds them to the
  first transcript.  Lean does not choose or revise the record.  This module
  only packages admitted=0 and divides the exact 65536-code scan into
  independently checkable blocks.

  No minimum or executed-parity claim is made here.  The downstream block
  certificates must prove that no view in any block is below the emitted
  canonical table, then prove that the blocks cover the frozen scan carrier.
-/
import SounioPireusStreamingMinimumCorrespondence

namespace SounioPireusExecutedStreamingProbe

set_option maxHeartbeats 0
set_option maxRecDepth 100000

open SounioPireusOperatorOrbitCanonicalization
open SounioPireusGaugeCoboundaryAction
open SounioPireusGaugeSectionCanonicalization
open SounioPireusLinearSwapGaugeDescent
open SounioPireusGL4ActionEnumeration
open SounioPireusSignTableBitVecLex
open SounioPireusConcreteQuotientAction
open SounioPireusStreamingMinimumCorrespondence

def matcherFreeSounioSourceSha256 : String :=
  "3136968a83bbba18d56c543895d6bbd9530ccf6c59db78ac6b6f2fa3bd26c9e4"

def frozenMatcherSounioSourceSha256 : String :=
  "7ada1b17bf91fdb3f4c48877d2485f71a65bb4159d88cb7e4b288c77bfe3cdae"

def frozenSounioFirstTranscriptSha256 : String :=
  "16af63f5e0f8aa7e5c899f4c395404b83fb402f6bbdb5f20dea2a3d10ad2e19f"

def frozenSounioExecutableCommit : String :=
  "73704f7afed6780c3a317b739cbd35fe94dbe395"

def admittedProbeIndex : Nat := 0
def admittedProbeMatrixCode : Nat := 58475
def admittedProbeSwap : Bool := false
def admittedProbeGaugeWord : Nat := 933

def admittedProbeRawBits : BitVec 256 :=
  0x00003e3c693c330f180f6200176654667f7c7c291c7f1ffe076f044864cb67d2

def admittedProbeRawTable : SignTable :=
  unpackTable admittedProbeRawBits

def admittedProbeCanonicalBits : BitVec 256 :=
  0x00000004617a56057d2e6a13294d57496b0e7cb017b259955561265e4bda64e4

def admittedProbeCanonical : NormalizedBits :=
  normalizedBitsOfTable (unpackTable admittedProbeCanonicalBits)

theorem admitted_probe_canonical_value_matches_frozen_bits :
    admittedProbeCanonical.val = admittedProbeCanonicalBits := by
  native_decide

def scanEntriesInCodeRange
    (start size : Nat) (bound : start + size ≤ matrixCodes) :
    List GL4ScanEntry :=
  (List.range size).attach.filterMap fun offset =>
    let code := start + offset.val
    if invertible : matrixInvertible code = true then
      some ⟨code, every_invertible_4x4_code_is_in_the_scan code
        (by
          have offsetLt : offset.val < size := by
            simpa using offset.property
          omega)
        invertible⟩
    else
      none

def codeBlockSize : Nat := 1024
def codeBlockCount : Nat := 64

def codeBlockStart (block : Fin codeBlockCount) : Nat :=
  block.val * codeBlockSize

theorem code_block_bound (block : Fin codeBlockCount) :
    codeBlockStart block + codeBlockSize ≤ matrixCodes := by
  have blockLt : block.val < 64 := by
    simpa [codeBlockCount] using block.isLt
  change block.val * 1024 + 1024 ≤ 65536
  omega

def codeBlockEntries (block : Fin codeBlockCount) : List GL4ScanEntry :=
  scanEntriesInCodeRange (codeBlockStart block) codeBlockSize
    (code_block_bound block)

def codeBlockViews (block : Fin codeBlockCount) :
    List FrozenScanActionView :=
  (codeBlockEntries block).flatMap fun entry =>
    [(entry, false), (entry, true)]

def candidateNotBelowFrozenCanonical
    (view : FrozenScanActionView) : Bool :=
  decide (admittedProbeCanonical.val ≤
    (frozenScanCandidate admittedProbeRawTable view).val)

def codeBlockDominatesFrozenCanonical
    (block : Fin codeBlockCount) : Bool :=
  (codeBlockViews block).all candidateNotBelowFrozenCanonical

def admittedProbeWinnerEntry : GL4ScanEntry :=
  ⟨admittedProbeMatrixCode,
    every_invertible_4x4_code_is_in_the_scan admittedProbeMatrixCode
      (by decide) (by native_decide)⟩

def admittedProbeWinnerView : FrozenScanActionView :=
  (admittedProbeWinnerEntry, admittedProbeSwap)

theorem admitted_probe_winner_matrix_code :
    admittedProbeWinnerEntry.val = admittedProbeMatrixCode := by
  rfl

theorem admitted_probe_winner_gauge_word :
    (directSectionWord
      (rawAct (actionOfFrozenScanView admittedProbeWinnerView)
        admittedProbeRawTable)).val = admittedProbeGaugeWord := by
  native_decide

theorem admitted_probe_winner_candidate_bits :
    (frozenScanCandidate admittedProbeRawTable
      admittedProbeWinnerView).val = admittedProbeCanonicalBits := by
  native_decide

theorem admitted_probe_winner_candidate :
    frozenScanCandidate admittedProbeRawTable admittedProbeWinnerView =
      admittedProbeCanonical := by
  apply Subtype.ext
  rw [admitted_probe_winner_candidate_bits,
    admitted_probe_canonical_value_matches_frozen_bits]

end SounioPireusExecutedStreamingProbe
