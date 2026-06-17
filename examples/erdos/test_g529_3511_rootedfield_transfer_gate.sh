#!/usr/bin/env bash
# Gate the current {3,5,11} QF-fragment transfer for the G529 obstruction.
#
# This checks the honest boundary:
#   * the theorem transfers the current LRAT obstruction through a QF3511TransferWf target,
#   * the target surface records endpoint and edge-distance support in {3,5,11},
#   * the named RootedField3511 interface no longer requires the four-root RootedField theorem,
#   * the explicit phi3511 evaluator proves the compressed qmul core, bridges the 16-mask qmul
#     through {3,5,11} support, and derives qmul/qadd/qsub/unit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEAN_DIR="$ROOT/formal/lean4"
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi

[[ -x "$LOCK" ]] || { echo "error: missing build lock helper: $LOCK" >&2; exit 1; }
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }

TRANSFER="$LEAN_DIR/SounioDeGreyChi5Transfer3511.lean"
ROOTED="$LEAN_DIR/SounioDeGreyChi5Rooted3511.lean"
EVAL="$LEAN_DIR/SounioDeGreyChi5Eval3511.lean"
VITRINE="$LEAN_DIR/DeGreyChi5Vitrine.lean"
PROOF_HOLE_RE='^\s*(sorry|admit)\b|:= by\s*(sorry|admit)\b|by\s+(sorry|admit)\b|#exit'

if rg -n "$PROOF_HOLE_RE" "$TRANSFER" "$ROOTED" "$EVAL" "$VITRINE"; then
  echo "error: QF3511 transfer Lean surface contains an incomplete proof marker" >&2
  exit 1
fi

rg -q 'SounioDeGreyChi5Rooted3511' "$TRANSFER"
rg -q 'SounioDeGreyChi5Eval3511' "$LEAN_DIR/lakefile.lean"
rg -q 'scoped to the current embedding' "$TRANSFER"
rg -q 'standalone RootedField3511 interface' "$LEAN_DIR/lakefile.lean"
rg -q 'named three-root' "$ROOTED"
rg -q 'adjacent evaluator file derives an' "$ROOTED"
rg -q 'bridges de Grey.*16-mask qmul through.*3,5,11.*support' "$EVAL"
rg -q 'phi3511_qmul' "$EVAL"
if rg -q 'not yet a standalone RootedField3511 interface|does not introduce a new three-root field interface' \
    "$LEAN_DIR/lakefile.lean" "$TRANSFER" "$ROOTED" "$EVAL" "$VITRINE"; then
  echo "error: stale RootedField3511 boundary wording remains" >&2
  exit 1
fi

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" build SounioDeGreyChi5Transfer3511 SounioDeGreyChi5Rooted3511 \
    SounioDeGreyChi5Eval3511 DeGreyChi5Vitrine
)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM

cat > "$WORK/check_g529_3511_transfer.lean" <<'EOF'
import DeGreyChi5Vitrine

open UnitDistanceChromatic

noncomputable section

#check (DeGrey529.Transfer3511.QF3511TransferWf)
#check (DeGrey529.Transfer3511.qf3511Wf)
#check (DeGrey529.Transfer3511.qf3511Transfer_chi_ge_5_current_embedding)
#check (DeGrey529.Transfer3511.rootedTransfer3511)
#check (DeGrey529.Transfer3511.rootedField_chi_ge_5_current_3511)
#check (DeGrey529.Transfer3511.qf3511TransferCurrentEmbeddingCertificate :
  DeGrey529.Transfer3511.QF3511TransferCurrentEmbeddingCertificate)
#check (DeGrey529.Rooted3511.RootedField3511)
#check (DeGrey529.Rooted3511.RootedField3511.toQF3511TransferWf)
#check (DeGrey529.Rooted3511.RootedField3511.r3511)
#check (DeGrey529.Rooted3511.RootedField3511.generator_law3511)
#check (DeGrey529.Rooted3511.RootedField3511.evalNum3511)
#check (DeGrey529.Rooted3511.RootedField3511.evalNum8)
#check (DeGrey529.Rooted3511.RootedField3511.compressNum3511)
#check (DeGrey529.Rooted3511.RootedField3511.qmulNum3511)
#check (DeGrey529.Rooted3511.RootedField3511.radIdx3511_xor)
#check (DeGrey529.Rooted3511.RootedField3511.radIdx3511_land)
#check (DeGrey529.Rooted3511.RootedField3511.qf3511Wf_coeff_zero_of_unsupported)
#check (DeGrey529.Rooted3511.RootedField3511.qf3511Wf_sqrt7_coeffs_zero)
#check (DeGrey529.Rooted3511.RootedField3511.evalNum8_qmulNum3511)
#check (DeGrey529.Rooted3511.RootedField3511.evalNum3511_eq_evalNum8_compress)
#check (DeGrey529.Rooted3511.RootedField3511.evalNum3511_qmul_num_bridge)
#check (DeGrey529.Rooted3511.RootedField3511.evalNum3511_qmul)
#check (DeGrey529.Rooted3511.RootedField3511.phi3511)
#check (DeGrey529.Rooted3511.RootedField3511.IntCastNonzero)
#check (DeGrey529.Rooted3511.RootedField3511.phi3511_qmul)
#check (DeGrey529.Rooted3511.RootedField3511.phi3511_qadd)
#check (DeGrey529.Rooted3511.RootedField3511.phi3511_qsub)
#check (DeGrey529.Rooted3511.RootedField3511.phi3511_unit)
#check (DeGrey529.Rooted3511.RootedField3511.toDerivedQF3511TransferWf)
#check (DeGrey529.Rooted3511.RootedField3511.derived_phi3511_chi_ge_5_current_embedding)
#check (DeGrey529.Rooted3511.RootedField3511.Phi3511DerivedTransferCertificate)
#check (DeGrey529.Rooted3511.RootedField3511.phi3511DerivedTransferCertificate)
#check (DeGrey529.Rooted3511.RootedField3511.Phi3511AddSubUnitCertificate)
#check (DeGrey529.Rooted3511.rootedField3511_chi_ge_5_current_embedding)
#check (DeGrey529.Rooted3511.ofRootedField)
#check (DeGrey529.Rooted3511.rootedField_via_3511_chi_ge_5_current_embedding)
#check (DeGrey529.Rooted3511.rootedField3511CurrentEmbeddingCertificate :
  DeGrey529.Rooted3511.RootedField3511CurrentEmbeddingCertificate)
#check (DeGrey529.Showcase.qf3511_transfer_current_embedding_chi_ge_5)
#check (DeGrey529.Showcase.qf3511TransferCurrentEmbeddingCertificate :
  DeGrey529.Transfer3511.QF3511TransferCurrentEmbeddingCertificate)
#check (DeGrey529.Showcase.rootedField3511_transfer_current_embedding_chi_ge_5)
#check (DeGrey529.Showcase.rootedField3511CurrentEmbeddingCertificate :
  DeGrey529.Rooted3511.RootedField3511CurrentEmbeddingCertificate)
#check (DeGrey529.Showcase.rootedField3511PhiAddSubUnitCertificate)
#check (DeGrey529.Showcase.rootedField3511DerivedPhiTransferWf)
#check (DeGrey529.Showcase.rootedField3511DerivedPhiTransferCertificate)
#check (DeGrey529.Showcase.rootedField3511_derived_phi_transfer_current_embedding_chi_ge_5)

example :
    ∀ T : DeGrey529.Transfer3511.QF3511TransferWf,
      ¬ Nonempty (PlaneColouring (T.F × T.F) T.unit 4) :=
  DeGrey529.Transfer3511.qf3511TransferCurrentEmbeddingCertificate.transfer_no_four_colouring

example :
    DeGrey529.Param.edgeEndpointsInPrimeSubplane
      DeGrey529.Transfer3511.primeSupport = true :=
  DeGrey529.Transfer3511.qf3511TransferCurrentEmbeddingCertificate.endpoint_support

example :
    DeGrey529.edges.toList.all (fun e =>
      DeGrey529.Support.edgeDistanceTermsSupportedByPrimes e
        DeGrey529.Transfer3511.primeSupport) = true :=
  DeGrey529.Transfer3511.qf3511TransferCurrentEmbeddingCertificate.edge_distance_support

example :
    DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane
      DeGrey529.Transfer3511.primeSupport :=
  DeGrey529.Transfer3511.qf3511TransferCurrentEmbeddingCertificate.full_current_lrat_support

example :
    ∀ ps : List Nat, DeGrey529.Support.properPrimeSubsupport3511 ps →
      ¬ DeGrey529.Param.currentG529LRATObstructionSupportCarriedByPrimeSubplane ps :=
  DeGrey529.Transfer3511.qf3511TransferCurrentEmbeddingCertificate.no_proper_current_lrat_support

example :
    ∀ T : DeGrey529.Transfer3511.QF3511TransferWf,
      ¬ Nonempty (PlaneColouring (T.F × T.F) T.unit 4) :=
  DeGrey529.Showcase.qf3511TransferCurrentEmbeddingCertificate.transfer_no_four_colouring

example :
    DeGrey529.Param.edgeEndpointsInPrimeSubplane
      DeGrey529.Transfer3511.primeSupport = true :=
  DeGrey529.Showcase.qf3511TransferCurrentEmbeddingCertificate.endpoint_support

example :
    DeGrey529.edges.toList.all (fun e =>
      DeGrey529.Support.edgeDistanceTermsSupportedByPrimes e
        DeGrey529.Transfer3511.primeSupport) = true :=
  DeGrey529.Showcase.qf3511TransferCurrentEmbeddingCertificate.edge_distance_support

example :
    DeGrey529.Transfer3511.QF3511TransferCurrentEmbeddingCertificate :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.qf3511_transfer_current_embedding

example :
    ∀ R : DeGrey529.Rooted3511.RootedField3511, ∀ i j : Nat,
      R.mul (R.r3511 i) (R.r3511 j) =
        R.mul (R.ofNatProd3511 (Nat.land i j)) (R.r3511 (Nat.xor i j)) :=
  DeGrey529.Rooted3511.rootedField3511CurrentEmbeddingCertificate.three_bit_generator_law

example :
    ∀ R : DeGrey529.Rooted3511.RootedField3511,
      ¬ Nonempty (PlaneColouring (R.F × R.F) R.unit 4) :=
  DeGrey529.Rooted3511.rootedField3511CurrentEmbeddingCertificate.three_root_transfer

example :
    ∀ R : SounioSqrt.RootedField,
      ¬ Nonempty (PlaneColouring
        ((DeGrey529.Rooted3511.ofRootedField R).F ×
          (DeGrey529.Rooted3511.ofRootedField R).F)
        (DeGrey529.Rooted3511.ofRootedField R).unit 4) :=
  DeGrey529.Rooted3511.rootedField3511CurrentEmbeddingCertificate.compatibility_from_four_root

example :
    DeGrey529.Rooted3511.RootedField3511CurrentEmbeddingCertificate :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField3511_transfer_current_embedding

example :
    ∀ R : DeGrey529.Rooted3511.RootedField3511, R.IntCastNonzero →
      R.Phi3511AddSubUnitCertificate :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField3511_eval_add_sub_unit

example :
    ∀ R : DeGrey529.Rooted3511.RootedField3511, ∀ hden : R.IntCastNonzero,
      R.Phi3511DerivedTransferCertificate hden :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField3511_derived_phi_transfer_certificate

example :
    ∀ R : DeGrey529.Rooted3511.RootedField3511, ∀ hden : R.IntCastNonzero,
      ¬ Nonempty (PlaneColouring
        (R.F × R.F)
        (DeGrey529.Showcase.rootedField3511DerivedPhiTransferWf R hden).unit 4) :=
  DeGrey529.Showcase.scopedG5293511MinimalityShowcaseCertificate
    |>.rootedField3511_derived_phi_transfer

example (R : DeGrey529.Rooted3511.RootedField3511) (hden : R.IntCastNonzero) :
    ∀ x y,
      R.evalNum8 (DeGrey529.Rooted3511.RootedField3511.qmulNum3511 x y) =
        R.mul (R.evalNum8 x) (R.evalNum8 y) :=
  (DeGrey529.Showcase.rootedField3511PhiAddSubUnitCertificate R hden).compressed_qmul_core

example (R : DeGrey529.Rooted3511.RootedField3511) (hden : R.IntCastNonzero) :
    ∀ x y, DeGrey529.Transfer3511.qf3511Wf x → DeGrey529.Transfer3511.qf3511Wf y →
      R.phi3511 (DeGrey529.qmul x y) = R.mul (R.phi3511 x) (R.phi3511 y) :=
  (DeGrey529.Showcase.rootedField3511PhiAddSubUnitCertificate R hden).qmul_law

example (R : DeGrey529.Rooted3511.RootedField3511) (hden : R.IntCastNonzero) :
    ∀ x y, DeGrey529.Transfer3511.qf3511Wf x → DeGrey529.Transfer3511.qf3511Wf y →
      R.phi3511 (DeGrey529.qadd x y) = R.add (R.phi3511 x) (R.phi3511 y) :=
  (DeGrey529.Showcase.rootedField3511PhiAddSubUnitCertificate R hden).qadd_law

example (R : DeGrey529.Rooted3511.RootedField3511) (hden : R.IntCastNonzero) :
    ∀ x y, DeGrey529.Transfer3511.qf3511Wf x → DeGrey529.Transfer3511.qf3511Wf y →
      R.phi3511 (DeGrey529.qsub x y) =
        R.add (R.phi3511 x) (R.neg (R.phi3511 y)) :=
  (DeGrey529.Showcase.rootedField3511PhiAddSubUnitCertificate R hden).qsub_law

example (R : DeGrey529.Rooted3511.RootedField3511) (hden : R.IntCastNonzero) :
    ∀ d, DeGrey529.Transfer3511.qf3511Wf d → DeGrey529.isOne d = true →
      R.phi3511 d = R.one :=
  (DeGrey529.Showcase.rootedField3511PhiAddSubUnitCertificate R hden).unit_law

end

#print axioms DeGrey529.Transfer3511.qf3511Transfer_chi_ge_5_current_embedding
#print axioms DeGrey529.Transfer3511.rootedField_chi_ge_5_current_3511
#print axioms DeGrey529.Transfer3511.qf3511TransferCurrentEmbeddingCertificate
#print axioms DeGrey529.Rooted3511.RootedField3511.generator_law3511
#print axioms DeGrey529.Rooted3511.RootedField3511.qf3511Wf_coeff_zero_of_unsupported
#print axioms DeGrey529.Rooted3511.RootedField3511.qf3511Wf_sqrt7_coeffs_zero
#print axioms DeGrey529.Rooted3511.RootedField3511.evalNum8_qmulNum3511
#print axioms DeGrey529.Rooted3511.RootedField3511.evalNum3511_qmul_num_bridge
#print axioms DeGrey529.Rooted3511.RootedField3511.evalNum3511_qmul
#print axioms DeGrey529.Rooted3511.RootedField3511.phi3511_qmul
#print axioms DeGrey529.Rooted3511.RootedField3511.phi3511_qadd
#print axioms DeGrey529.Rooted3511.RootedField3511.phi3511_qsub
#print axioms DeGrey529.Rooted3511.RootedField3511.phi3511_unit
#print axioms DeGrey529.Rooted3511.RootedField3511.toDerivedQF3511TransferWf
#print axioms DeGrey529.Rooted3511.RootedField3511.derived_phi3511_chi_ge_5_current_embedding
#print axioms DeGrey529.Rooted3511.RootedField3511.phi3511DerivedTransferCertificate
#print axioms DeGrey529.Rooted3511.rootedField3511_chi_ge_5_current_embedding
#print axioms DeGrey529.Rooted3511.rootedField_via_3511_chi_ge_5_current_embedding
#print axioms DeGrey529.Rooted3511.rootedField3511CurrentEmbeddingCertificate
#print axioms DeGrey529.Showcase.qf3511_transfer_current_embedding_chi_ge_5
#print axioms DeGrey529.Showcase.qf3511TransferCurrentEmbeddingCertificate
#print axioms DeGrey529.Showcase.rootedField3511_transfer_current_embedding_chi_ge_5
#print axioms DeGrey529.Showcase.rootedField3511CurrentEmbeddingCertificate
#print axioms DeGrey529.Showcase.rootedField3511PhiAddSubUnitCertificate
#print axioms DeGrey529.Showcase.rootedField3511DerivedPhiTransferWf
#print axioms DeGrey529.Showcase.rootedField3511DerivedPhiTransferCertificate
#print axioms DeGrey529.Showcase.rootedField3511_derived_phi_transfer_current_embedding_chi_ge_5
EOF

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$WORK/check_g529_3511_transfer.lean"
) 2>&1 | tee "$WORK/check.out"

if rg -n 'error:|sorryAx' "$WORK/check.out"; then
  echo "error: generated RootedField3511 verifier failed or introduced sorryAx" >&2
  exit 1
fi

rg -q 'Rooted3511.rootedField3511_chi_ge_5_current_embedding' "$WORK/check.out"
rg -q 'Rooted3511.rootedField3511CurrentEmbeddingCertificate' "$WORK/check.out"
rg -q 'Rooted3511.RootedField3511.qf3511Wf_coeff_zero_of_unsupported' "$WORK/check.out"
rg -q 'Rooted3511.RootedField3511.evalNum8_qmulNum3511' "$WORK/check.out"
rg -q 'Rooted3511.RootedField3511.evalNum3511_qmul' "$WORK/check.out"
rg -q 'Rooted3511.RootedField3511.phi3511_qmul' "$WORK/check.out"
rg -q 'Rooted3511.RootedField3511.phi3511_qadd' "$WORK/check.out"
rg -q 'Rooted3511.RootedField3511.phi3511_unit' "$WORK/check.out"
rg -q 'Rooted3511.RootedField3511.toDerivedQF3511TransferWf' "$WORK/check.out"
rg -q 'Rooted3511.RootedField3511.derived_phi3511_chi_ge_5_current_embedding' "$WORK/check.out"
rg -q 'Rooted3511.RootedField3511.phi3511DerivedTransferCertificate' "$WORK/check.out"
rg -q 'Showcase.rootedField3511_transfer_current_embedding_chi_ge_5' "$WORK/check.out"
rg -q 'Showcase.rootedField3511PhiAddSubUnitCertificate' "$WORK/check.out"
rg -q 'Showcase.rootedField3511DerivedPhiTransferWf' "$WORK/check.out"
rg -q 'Showcase.rootedField3511DerivedPhiTransferCertificate' "$WORK/check.out"

echo "g529_3511_rootedfield_transfer_gate: PASS"
