import SounioZDScalarBit
open Sounio.ZDSignedState Sounio.ZDScalarBit
#print axioms native_iso_iff_counts
#print axioms actualNativeIso
#print axioms native_iso_iff_scalar_bit
#print axioms native_scalar_at_most_two
#print axioms native_scalar_two_sharp
example : nativeEdges 1 1 (by decide) (by decide)=168 ∧
    nativeTriangles 1 1 (by decide) (by decide)=288 := by
  have h := actual_native_counts 1 1 (by decide) (by decide)
  have e : edges (labelCodes 1 1).1=21 := by decide +kernel
  have p : positives (labelCodes 1 1).1=18 := by decide +kernel
  omega
example : nativeEdges 1 8 (by decide) (by decide)=168 ∧
    nativeTriangles 1 8 (by decide) (by decide)=0 := by
  have h := actual_native_counts 1 8 (by decide) (by decide)
  have e : edges (labelCodes 1 8).1=21 := by decide +kernel
  have p : positives (labelCodes 1 8).1=0 := by decide +kernel
  omega
example : nativeEdges 1 9 (by decide) (by decide)=72 ∧
    nativeTriangles 1 9 (by decide) (by decide)=0 := by
  have h := actual_native_counts 1 9 (by decide) (by decide)
  have e : edges (labelCodes 1 9).1=9 := by decide +kernel
  have p : positives (labelCodes 1 9).1=0 := by decide +kernel
  omega
