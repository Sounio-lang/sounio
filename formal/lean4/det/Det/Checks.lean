import Det.Values
import Det.Counterexample

namespace Det

-- A seed is an operand in both schedules. This failed for the former tree
-- implementation, which dropped the seed on every nonempty input list.
example : reduceWithOrder .seq (· + ·) 3 [5] = 8 := by decide
example : reduceWithOrder .balancedTree (· + ·) 3 [5] = 8 := by simp [reduceWithOrder, reduceTree, pairPass]

-- Nonassociative arithmetic distinguishes the two fixed schedules on the
-- same four operands. These are bit-level toy operations, not IEEE arithmetic.
example : reduceWithOrder .seq (· - ·) 20 [5, 4, 2] = 9 := by decide
example : reduceWithOrder .balancedTree (· - ·) 20 [5, 4, 2] = 13 := by simp [reduceWithOrder, reduceTree, pairPass]
example : reduceWithOrder .seq (· + ·) 3 [] = 3 := by decide
example : reduceWithOrder .balancedTree (· + ·) 3 [] = 3 := by simp [reduceWithOrder, reduceTree]

-- A vendor-dependent local can legitimately be bound and then ignored.
-- Its value differs, but the Det result may not depend on it.
private def ignoredVendor : Term := .letIn 0 (.vendor 0 []) (.f32Lit 0x3F800000)
private theorem ignoredVendor_typed : HasType TyEnv.empty ignoredVendor Ty.f32Det :=
  .letIn (.vendor .nil) .f32Lit

example : eval backendA Env.empty (.vendor 0 []) ≠
    eval backendB Env.empty (.vendor 0 []) := by decide
example (B : Backend) (otherVendor : Nat → List Bits → Bits) :
    eval B Env.empty ignoredVendor = eval { B with vendor := otherVendor } Env.empty ignoredVendor :=
  no_forge B otherVendor ignoredVendor_typed

-- Negative witness is paired with positive typing of the tainted result.
example : HasType TyEnv.empty implicitFlowWitness Ty.f32 := by
  unfold implicitFlowWitness
  apply HasType.ite (qc := .plain) (qb := .det)
  · exact .prim rfl (.cons (.vendor .nil) (.cons (.sub .f32Lit (.mk .det_le_plain)) .nil))
  · exact .f32Lit
  · exact .f32Lit

#print axioms det_agree
#print axioms no_forge
#print axioms implicit_flow_needed
#print axioms implicit_flow_needed_for_ref
#print axioms implicit_flow_witness_rejected
#print axioms det_f32_bits
#print axioms no_forge_direct
#print axioms no_forge_replacement

end Det
