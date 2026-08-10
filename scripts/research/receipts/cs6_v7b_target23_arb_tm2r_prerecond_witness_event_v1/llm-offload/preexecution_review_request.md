# Adversarial review request: witness-local upward event

Review the soundness of the proposed rigorous diagnostic, not Python style.
The code under review is
`cs6_v7b_target23_arb_tm2r_prerecond_witness_event_worker.py`.

The state is a degree-2 Taylor model with Arb interval remainder over six
original symbolic variables. The frozen production method refused the exact
depth-8 spatial witness after a time step of depth 10 because the `w` tube
contained zero and neither endpoint/event projection closed.

The diagnostic repeats the same time-step stack through depth 10, records the
refusal boundary, and continues time bisection through depth 18. For every
section-containing tube after a strict negative departure it tries an interval
Newton cover at both endpoint Taylor models. Each accepted cover requires:

If a cover closes before the frozen depth-10 refusal has been recorded, the
worker returns `EARLY_ACCEPTANCE_BEFORE_FROZEN_REFUSAL`; a mandatory control
then forces `IMPLEMENTATION_INCONSISTENCY`. Such a path cannot produce
`EVENT_REFINEMENT_BUDGET_LIMIT`.

1. `d/dt w = u*v - w - z_s` has a strictly positive interval lower bound;
2. the Newton event time remains strictly inside its Picard slab;
3. the projected section normal `u*v - z_s` has a strictly positive interval
   lower bound on every projected leaf;
4. aggregate polynomial/remainder weight has positive upper bound for each of
   the same six original symbolic variables on every projected leaf. This is
   interpreted only as represented possible dependence, not pointwise nonzero
   dependence.

The worker classifies:

- `EVENT_REFINEMENT_BUDGET_LIMIT` only if the original depth-10 refusal was
  reproduced and a later endpoint Newton cover satisfies all four conditions.
  This says the event is certified under the extended budget; it does not call
  the original budget-bounded refusal incorrect;
- `WITNESS_ENCLOSURE_UNRESOLVED` when depth 18 is exhausted while the tube
  derivative stays strictly positive but the section enclosure/Newton cover
  does not close, or Picard closure fails at depth 18;
- `WITNESS_TRANSVERSALITY_UNRESOLVED` when the terminal tube derivative lacks a
  strictly positive lower bound;
- `IMPLEMENTATION_INCONSISTENCY` if the frozen path/domain/refusal controls do
  not replay.

Audit these questions specifically:

1. Are the sign tests sufficient for a strictly upward-transversal `w=0`
   event, assuming the imported Picard and interval-Newton routines are sound?
   In particular, apply the standard interval-Newton inclusion theorem: the
   Newton image is strictly contained in the validated time slab, so existence
   does not rely on endpoint sign change.
2. Does acceptance after finer time bisection justify the narrower label
   `EVENT_REFINEMENT_BUDGET_LIMIT`, without claiming the old refusal incorrect
   or claiming a global return?
3. Is using positive *upper* bounds of variable weights evidence only of
   retained possible symbolic dependence, rather than nonzero dependence for
   every point? Flag any overclaim.
4. Is the enclosure-versus-transversality discriminator logically exhaustive
   for this local diagnostic? The imported upward projector cannot accept
   unless the full-tube derivative has a strictly positive lower bound.
5. Find any control-flow path that could accept before recording the frozen
   depth-10 refusal, accept a non-upward event, or promote a partial cover.

Return BLOCKER, MAJOR, or MINOR findings with a concrete counterexample or
required invariant. If sound within the stated scope, say so explicitly.
