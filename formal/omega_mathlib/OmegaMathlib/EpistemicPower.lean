import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace OmegaMathlib

/-- Clamp utility used by the Epistemic Power score. -/
def clamp (eps : Real) (x : Real) : Real := max eps (min 1 x)

/-- Log-domain utility to avoid underflow/overflow in products. -/
def logEpistemicPower (weights metrics : List Real) (penalty : Real) (eps : Real := 1e-9) : Real :=
  (List.zipWith (fun w m => w * Real.log (clamp eps m)) weights metrics).sum - penalty

/-- Canonical Epistemic Power utility in real space. -/
def epistemicPower (weights metrics : List Real) (penalty : Real) (eps : Real := 1e-9) : Real :=
  Real.exp (logEpistemicPower weights metrics penalty eps)

lemma clamp_lower_bound (eps x : Real) : eps <= clamp eps x := by
  unfold clamp
  exact le_max_left eps (min 1 x)

lemma clamp_upper_bound (eps x : Real) : clamp eps x <= max eps 1 := by
  unfold clamp
  exact max_le_max le_rfl (min_le_right x 1)

/-- Increasing penalty decreases log-EpistemicPower (antitone in penalty). -/
theorem logEpistemicPower_penalty_antitone
    (weights metrics : List Real) (eps p1 p2 : Real)
    (h : p1 <= p2) :
    logEpistemicPower weights metrics p2 eps <=
      logEpistemicPower weights metrics p1 eps := by
  unfold logEpistemicPower
  linarith

/-- Increasing penalty decreases EpistemicPower. -/
theorem epistemicPower_penalty_antitone
    (weights metrics : List Real) (eps p1 p2 : Real)
    (h : p1 <= p2) :
    epistemicPower weights metrics p2 eps <=
      epistemicPower weights metrics p1 eps := by
  unfold epistemicPower
  exact Real.exp_le_exp.mpr (logEpistemicPower_penalty_antitone weights metrics eps p1 p2 h)

end OmegaMathlib
