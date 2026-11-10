import Std

/-!
# Exact rational sign of the RODAS4 fourth clock-stage coefficient

Scope: a closed rational consistency calculation for the displayed, published
rounded coefficient data. This is not a proof of the Sounio implementation,
binary64 error, convergence, stability, or fourth-order accuracy of the solver.

For the autonomous clock equation tau' = 1, the frozen stage relation has
d4_clock = gamma * (1 + c41*d1 + c42*d2 + c43*d3).
The rationals below encode the decimal coefficient strings exactly, rather
than interpreting them as machine Float values.

Primary coefficient custody:
WRF commit 06d4240ae989cc3e50af412bb472df3d9048783c,
chem/KPP/kpp/kpp-2.1/int/rosenbrock.f90, RODAS4 table.
Raw file SHA256:
44eb88bcbd91ab2f12f37ba71d314038d6a771f1a8750f5f8b8c2b317940fa2a.
https://github.com/wrf-model/WRF/blob/06d4240ae989cc3e50af412bb472df3d9048783c/chem/KPP/kpp/kpp-2.1/int/rosenbrock.f90#L1052-L1093

The exact rational expression is not asserted equal to the independently
rounded printed negative d4 decimal. We prove its sign and a coarse enclosing
interval, which discriminate the erroneous positive sign without that false
equality claim. The decide +kernel tactic checks the explicitly unfolded closed propositions
in the Lean kernel; there is no native_decide or floating-point oracle.
-/

namespace Sounio.Rodas4TimeConsistency

def gamma : Rat := 1 / 4
def d1 : Rat := 1 / 4
def d2 : Rat := -1043 / 10000
def d3 : Rat := 207 / 2000

def c41 : Rat := -1073529058151375 / 10000000000000000
def c42 : Rat := -9594562251023355 / 1000000000000000
def c43 : Rat := -2047028614809616 / 100000000000000

def d4_clock : Rat := gamma * (1 + c41 * d1 + c42 * d2 + c43 * d3)

/-- This pins the exact clock-stage expression used by the sign theorem. -/
theorem d4_clock_identity :
    d4_clock =
      (1 / 4 : Rat) *
        (1 + (-1073529058151375 / 10000000000000000 : Rat) * (1 / 4 : Rat)
           + (-9594562251023355 / 1000000000000000 : Rat) * (-1043 / 10000 : Rat)
           + (-2047028614809616 / 100000000000000 : Rat) * (207 / 2000 : Rat)) := by
  rfl

set_option maxRecDepth 10000 in
/-- The fourth clock coefficient computed from the published data is negative. -/
theorem d4_clock_negative : d4_clock < 0 := by
  unfold d4_clock gamma c41 c42 c43 d1 d2 d3
  decide +kernel

set_option maxRecDepth 10000 in
/-- A deliberately coarse exact interval; not a floating-point error bound. -/
theorem d4_clock_interval :
    (-4 / 100 : Rat) < d4_clock ∧ d4_clock < (-3 / 100 : Rat) := by
  unfold d4_clock gamma c41 c42 c43 d1 d2 d3
  decide +kernel

def printed_positive_d4 : Rat := 3620000000000023 / 100000000000000000

set_option maxRecDepth 10000 in
/-- The positive printed coefficient cannot satisfy this frozen clock relation. -/
theorem positive_d4_is_inconsistent : d4_clock ≠ printed_positive_d4 := by
  unfold d4_clock gamma c41 c42 c43 d1 d2 d3 printed_positive_d4
  decide +kernel

end Sounio.Rodas4TimeConsistency

#print axioms Sounio.Rodas4TimeConsistency.d4_clock_negative
#print axioms Sounio.Rodas4TimeConsistency.d4_clock_interval
#print axioms Sounio.Rodas4TimeConsistency.positive_d4_is_inconsistent
