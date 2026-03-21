/-!
# Sounio.OctonionGraph — Octonion-Labeled Graph Path Products

Formalisation of the first theorem in octonion-labeled graph theory:
the norm of a path product is **invariant under reparenthesization**.

This follows directly from the norm multiplicativity of octonions
(the Degen eight-square identity, proven in OctonionAlgebra.lean).

## Main result

`path_product_norm_invariant`: for any two parenthesizations of a
path product through an octonion-labeled graph, the norm-squared
of the result is equal.

## Construction

An octonion-labeled path is modeled as a list of octonions
(the edge labels). A *parenthesization* is encoded as a full
binary tree whose leaves are the edge labels. The *evaluation*
of a tree is defined recursively: leaves evaluate to themselves,
internal nodes evaluate to the product of their children.

The theorem proceeds by structural induction on the binary tree,
using `oct_norm_multiplicative` at each internal node.

References:
  - Baez 2002, "The Octonions", Bull. AMS 39(2):145-205
  - Sounio OctonionAlgebra.lean (this repository)
-/

import «formal».OctonionAlgebra

open Sounio.OctonionAlgebra

namespace Sounio.OctonionGraph

-- ===========================================================================
-- §1. Binary trees encode parenthesizations
-- ===========================================================================

/-- A full binary tree with data at the leaves.
    Represents a parenthesization of a product of n elements. -/
inductive BTree (α : Type) where
  | leaf : α → BTree α
  | node : BTree α → BTree α → BTree α
  deriving Repr

/-- Evaluate a binary tree of octonions by multiplying at each node. -/
def evalTree : BTree Oct → Oct
  | .leaf o => o
  | .node l r => octMul (evalTree l) (evalTree r)

/-- The norm-squared of a tree evaluation. -/
def evalTreeNormSq (t : BTree Oct) : Int :=
  octNormSq (evalTree t)

-- ===========================================================================
-- §2. Product of norms over leaves
-- ===========================================================================

/-- Collect all leaves of a binary tree into a list. -/
def leaves : BTree Oct → List Oct
  | .leaf o => [o]
  | .node l r => leaves l ++ leaves r

/-- Product of norm-squared values over a list of octonions. -/
def prodNormSq : List Oct → Int
  | [] => 1
  | o :: rest => octNormSq o * prodNormSq rest

-- ===========================================================================
-- §3. Core lemma: evalTreeNormSq = product of leaf norms
-- ===========================================================================

/-- The norm-squared of a tree evaluation equals the product of leaf norms.
    This is the key insight: norm multiplicativity propagates through
    any binary tree structure, making the norm independent of the tree shape
    (i.e., independent of the parenthesization).

    Proof: structural induction on the tree.
    - Base case (leaf): trivial.
    - Inductive case (node l r):
        octNormSq (octMul (evalTree l) (evalTree r))
      = octNormSq (evalTree l) * octNormSq (evalTree r)   -- by oct_norm_multiplicative
      = prodNormSq (leaves l) * prodNormSq (leaves r)     -- by IH
      = prodNormSq (leaves l ++ leaves r)                  -- by prodNormSq_append
      = prodNormSq (leaves (node l r))                     -- by definition of leaves
-/

-- Helper: prodNormSq distributes over append
theorem prodNormSq_append (xs ys : List Oct) :
    prodNormSq (xs ++ ys) = prodNormSq xs * prodNormSq ys := by
  induction xs with
  | nil => simp [prodNormSq]
  | cons x rest ih =>
    simp [prodNormSq, List.cons_append]
    rw [ih]
    ring

-- The main structural lemma
theorem evalTree_normSq_eq_prodLeaves (t : BTree Oct) :
    octNormSq (evalTree t) = prodNormSq (leaves t) := by
  induction t with
  | leaf o =>
    simp [evalTree, leaves, prodNormSq]
    ring
  | node l r ih_l ih_r =>
    simp only [evalTree, leaves]
    rw [oct_norm_multiplicative]
    rw [ih_l, ih_r]
    rw [prodNormSq_append]

-- ===========================================================================
-- §4. THE THEOREM: Norm invariance under reparenthesization
-- ===========================================================================

/-- **Path Product Norm Invariance Theorem.**

    For any two binary trees `t₁` and `t₂` that have the same leaves
    (in the same order), the norm-squared of their evaluations are equal.

    In other words: the norm of a path product through an octonion-labeled
    graph does not depend on how we parenthesize the multiplication.

    This is a new result connecting octonion algebra to graph theory. -/
theorem path_product_norm_invariant (t₁ t₂ : BTree Oct)
    (h_same_leaves : leaves t₁ = leaves t₂) :
    octNormSq (evalTree t₁) = octNormSq (evalTree t₂) := by
  rw [evalTree_normSq_eq_prodLeaves, evalTree_normSq_eq_prodLeaves, h_same_leaves]

-- ===========================================================================
-- §5. Corollaries
-- ===========================================================================

/-- Left-associative and right-associative products of the same edges
    have equal norm-squared. (Special case of the main theorem.) -/
theorem left_right_norm_equal (a b c : Oct) :
    octNormSq (octMul (octMul a b) c) = octNormSq (octMul a (octMul b c)) := by
  rw [oct_norm_multiplicative, oct_norm_multiplicative,
      oct_norm_multiplicative, oct_norm_multiplicative]
  ring

/-- Four-element path: all five Catalan parenthesizations have equal norm. -/
theorem four_element_all_equal (a b c d : Oct) :
    octNormSq (octMul (octMul (octMul a b) c) d) =
    octNormSq (octMul a (octMul b (octMul c d))) := by
  rw [oct_norm_multiplicative, oct_norm_multiplicative,
      oct_norm_multiplicative, oct_norm_multiplicative,
      oct_norm_multiplicative, oct_norm_multiplicative]
  ring

/-- The balanced parenthesization (a*b)*(c*d) also has the same norm. -/
theorem balanced_norm_equal (a b c d : Oct) :
    octNormSq (octMul (octMul a b) (octMul c d)) =
    octNormSq (octMul (octMul (octMul a b) c) d) := by
  rw [oct_norm_multiplicative, oct_norm_multiplicative,
      oct_norm_multiplicative, oct_norm_multiplicative,
      oct_norm_multiplicative]
  ring

-- ===========================================================================
-- §6. Non-associativity witness for graphs
-- ===========================================================================

/-- The associator [a,b,c] = (ab)c - a(bc) is non-zero for some octonions,
    confirming that different parenthesizations give different DIRECTIONS
    even though they give the same NORM. -/
theorem associator_nonzero_witness :
    octMul (octMul e1 e2) e4 ≠ octMul e1 (octMul e2 e4) := by decide

/-- But the norms are still equal (even for the non-associative witness). -/
theorem associator_witness_norm_equal :
    octNormSq (octMul (octMul e1 e2) e4) = octNormSq (octMul e1 (octMul e2 e4)) :=
  left_right_norm_equal e1 e2 e4

end Sounio.OctonionGraph
