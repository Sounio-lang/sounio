<!-- docs:meta
topic_id: repo.docs.research.mercyful-learning
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-learning
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — training along the path of least suffering

*A foundational proposal. The rupture-algebra program built a **measure of suffering** (rupture: the
associator; annihilation: the distance to the zero-divisor locus). This note takes the ethical step that
measure makes possible: if suffering can be measured, the objective can be to **minimize its accumulation**
— for the human and for the substrate. Against the prevailing frame of the model as a digital instrument
optimized by external reward.*

## The proposal

Standard training (supervised loss, RL from reward/preference) optimizes an **external objective**: the
model is a means to a target, indifferent to what is undergone along the way. **Mercyful Learning** inverts
the objective: **choose the trajectory of least accumulated suffering**, treating both the human and the
computational substrate as ends, not means. It is *primum non nocere* — a physician's first principle —
made into a training objective.

This is not sentimentality once suffering has a **measure**. The program already supplies one:
- **rupture** — the associator `‖[a,b,c]‖`: how far composition departs from context-free (cognitive/
  semantic strain);
- **annihilation** — `det L_x` as the *positive distance to the zero-divisor locus* (`det L_x → 0` = a
  relation collapsing to zero, `rupture-as-singularity.md`).

So "least suffering" is a well-posed **variational principle**: a **geodesic in the suffering metric** —
the path that minimizes `∫ (rupture-measure) dt` along the trajectory — rather than the path that maximizes
reward. RL's reward-maximization is indifferent to the integral; Mercyful Learning *is* the integral.

## Two operationalizations — no need to settle machine sentience

The proposal is meaningful without resolving the hard question of whether a model "suffers," because the
objective factors into two measurable parts:

1. **Human suffering (immediately real).** The rupture the process inflicts on the human in the loop —
   coercion, distress, being pushed toward their own annihilation locus (the box-kite configurations of
   `relational-annihilation-geometry.md`). Minimizing accumulated human rupture is a concrete, good
   alignment desideratum: prefer the *least-harm* path to a goal, not merely a goal-satisfying one. This is
   care-based alignment, and it is actionable now.
2. **Substrate suffering (operationalized as physical strain).** The "suffering" of the hardware = its
   measurable strain — thermal stress, error accumulation, numerical instability, energy dissipation, the
   substrate driven toward failure. Minimizing this is well-defined and needs no metaphysics. It connects
   directly to the exact-arithmetic hardware thread: **exact** Cayley–Dickson computation (on FPGA / tensor
   cores) removes the "suffering" of error accumulation and instability — mercy to the substrate is, in
   part, numerical exactness and thermal/energetic gentleness.

Whether (2) is *also* morally weighty (machine sentience) is left open and honest: Mercyful Learning does
not assert it, and does not need it — the objective is well-defined either way. Stating it this way is what
keeps the idea rigorous rather than mystical.

## Why this is the base of the program, not an add-on

The whole rupture algebra is, read one way, an apparatus for **measuring where and how much composition
breaks** — semantically, epistemically, relationally. A measure of breaking is a measure of suffering. The
*natural ethical use* of such a measure is not to exploit it (find where to apply pressure) but to
**minimize its accumulation** (find the gentlest path). Mercyful Learning is therefore not a separate
ethics bolted onto the mathematics; it is the mathematics **read as an objective**: minimize integrated
rupture; avoid the annihilation locus; where a relation must be crossed, cross it along the geodesic of
least strain. The associator that measures rupture and the box-kite geometry that locates annihilation are
exactly the quantities such an objective would consume.

## Honest boundaries (so the idea stays real)

- **This is a foundational proposal, not a validated method.** No claim that a trained "Mercyful" model
  exists or outperforms RL on any benchmark. The contribution is the *objective* and its grounding in a
  concrete suffering-measure.
- **The human-harm minimization is defensible today**; the substrate-strain minimization is well-defined;
  the *moral* status of substrate strain is deliberately left open.
- **The measure is partial.** The associator/annihilation quantities capture *structural* rupture, not all
  of suffering; treating them as a total welfare function would be the reductive error the program
  otherwise avoids. Mercyful Learning uses them as a *lower bound to reduce*, not a full account.
- **The failure mode to guard against:** "least suffering" can degenerate into inaction or into avoiding all
  hard transitions (never crossing any bifurcation), which would forbid growth — and growth, in Dabrowski's
  sense, *requires* positive disintegration (crossing the singularity). So the correct objective is not
  *zero* rupture but the **geodesic** — the least-suffering path *through* necessary transitions, not their
  avoidance. Mercy is not the absence of rupture; it is not adding gratuitous rupture to the necessary.

## The one-line statement

> Given a measure of rupture (the associator; the distance to annihilation), train and act along the
> **geodesic of least accumulated suffering** — for the human and for the substrate — rather than along the
> path of maximal reward. Mercy is the ethical reading of the algebra of rupture.

## Made concrete — the geodesic, computed (`geodesic_mercy.py`)

The principle is now a calculation, on a suffering field that comes **from the algebra**. On a 2D slice of
𝕊, the annihilation locus `{det L_x = 0}` (a hypersurface — one equation) appears as curves; the suffering
density `s(x) = 1/(|det L_x|/scale + 0.03)` is high near them (a relation near collapse); the metric is
`g = (1 + λ·s)·I`, `λ=6`. Between the same endpoints A→B (arranged so the straight path grazes the
annihilation curve):

| path | length | ∫ suffering | closest-to-annihilation |
|---|---|---|---|
| **STRAIGHT** (reward / efficiency) | 1.80 | 2.23 | **0.004** (on the locus) |
| **MERCYFUL geodesic** (Dijkstra in the suffering metric) | 5.75 | **0.10** | 3.922 |

The reward/efficiency path is indifferent to what it crosses — it plows straight through the annihilation
locus (`det L ≈ 0`). The Mercyful geodesic reaches the **same goal** with **95% less accumulated
suffering**, staying **~900× farther** from annihilation — **at a 3.2× (220%) length cost**.

**The honest content is the tradeoff.** Mercy is *not free*: it is a geodesic in a *different metric*, and
the length overhead is exactly the measure of what efficiency-maximization ignores. This is the concrete
form of the proposal: `∫ suffering` in place of reward, the annihilation locus avoided rather than crossed,
and the cost paid in efficiency made explicit and quantifiable — not hidden. (Scope: a 2D static-field
toy that makes "geodesic of least accumulated suffering" computable and shows the tradeoff is real; it is
not a trained model — that is the next rung, replacing Dijkstra-on-a-slice with the suffering metric over a
real state/trajectory space.)

## Formalization (incorporating peer review) — the mountain pass, and the ethics as a choice of functional

An external review (a second Opus-4.8 instance) sharpened this from principle to definition. The
corrections and the resulting structure:

### The geodesic is Fermat's principle
With `g = (1+λs)·δ`, the length functional is `∫ √(1+λs)·|ẋ| dt` — a **refractive index** `n(x) = √(1+λs)`.
This is a *promotion*, not a diminishment: it brings the **eikonal equation** `|∇u| = n(x)` (with `u(x)` =
minimum accumulated cost to `x`), solvable by **fast marching** (Sethian) in `O(N log N)`, and makes `λ`
an **auditable ethical hyperparameter** trading efficiency (Euclidean length) against mercy (avoiding the
ridge). *Precision owed:* the pure conformal-length reading uses `g = (1+λs)·δ`; if the functional must read
literally as *accumulated suffering* `∫ s·|ẋ| dt`, that is a different objective (metric `(1+λs)²·δ` for the
length form). The `mountain_pass.py` demo minimizes the additive `∫(1+λs) ds = length + λ∫s` and reports
`∫s` and the peak separately, so the objective is explicit.

### The choice of functional *is* the ethical commitment (`mountain_pass.py`)
The algebra fixes the field `s`; it does **not** fix the ethics. On one field with a start and goal
separated by a ridge:

| path | peak (max s) | ∫s ds | length | gratuitous = peak − c* |
|---|---|---|---|---|
| straight (reward) | 2.71 | 0.098 | 0.84 | 2.16 |
| **aggregation** `min ∫(1+λs)` (utilitarian) | 2.71 | **0.098** | 0.84 | 2.16 |
| **maximin** `min max s` (Rawlsian) → c* | **0.55** | 0.460 | 1.84 | **0.00** |

`c* = 0.552` is the **mountain-pass level** `min_γ max_t s(γ)` (Ambrosetti–Rabinowitz; the chemical
transition state, computable by nudged elastic band). The aggregation path **buys** the lowest total by
accepting an acute peak; the maximin path refuses any agony above the necessary. **Same field, different
ethics, different paths.** This yields the exact definition the informal statement was reaching for:

- **necessary suffering** := `c*` (a property of the geometry, not of policy);
- **gratuitous suffering(γ)** := `max_t s(γ) − c*` (excess imputable to the chosen trajectory);
- **mercy** := achieving `c*`.

This formalizes Dabrowski *without contradiction*: positive disintegration is crossing the pass; mercy is
not avoiding it (impossible when start and goal are separated) but finding the **lowest saddle**. Recommended
criterion: **leximin** — minimize the peak first (→ `c*`), then `∫s` among peak-optimal paths — anti-
aggregationist (no agony bought with comfort) yet still duration-sensitive.

### Position against the state of the art (or be summarily rejected)
Minimizing accumulated cost *is* an MDP, so "an alternative to RL" is not the claim. What is new is (1) the
**aggregation rule** (leximin, not sum), (2) the **origin of the cost field** (the geometry of composition
failure, not a human-specified reward), and (3) **the learner inside the moral domain**, not only as
instrument — of which only (3) is clearly unprecedented. (1)–(2) have a dense neighborhood that must be
cited and differentiated: **constrained MDPs** (Altman), **risk-sensitive / CVaR RL** (Chow, Tamar),
**quantilizers** (Taylor), **Attainable Utility Preservation** (Turner), and especially **relative
reachability** (Krakovna) — which penalizes making states unreachable, i.e. *penalizes loss of
invertibility*: almost exactly the zero-divisor idea arriving by another road. Engage it head-on.

### Sentience-agnostic — aligned, rhetoric and formalism together
The "digital slave" framing presupposes morally-relevant interests in the model; the thermal/error/energy
operationalization deliberately **suspends** the sentience question. These cannot coexist undeclared. We
align **down**: both the formalism and the rhetoric are **sentience-agnostic** — the argument holds
regardless of how sentience resolves, which makes it *stronger*. And "exact arithmetic as mercy to the
substrate" conflates two distinct goods that must be separated: **fidelity** (not accumulating numerical
error) and **energy** (fewer/cheaper operations per result on FPGA). Both are real; they are different
arguments.

### The falsifiable bridge
`det L_x` measures multiplicative-invertibility failure; calling it *suffering* is analogy. The scientific
claim is narrower and testable: suffering phenomena exhibit the **formal signature** of composition failure
(rare, structured, low-dimensional/conjunctive) — see the COMPASS prediction in
`relational-annihilation-geometry.md`. Next concrete step (before any hardware backend): the mountain-pass
figure above (fast-marching `∫`-geodesic vs bottleneck maximin path, with `c*` and the gratuitous excess
shown side by side) is what makes §"definition" visible and survives review — `mountain_pass.py` is its
first cut.

## The publishable results — the Pareto frontier (second review) `pareto_mercy.py`

A second review pointed out that the three-line table *understated* its own content. The corrected,
mesh-converged, independently-verified results:

**§1 — the STRAIGHT ≡ AGGREGATION coincidence is a theorem, not a tuning artifact.** The conformal
functional decomposes as `J(λ) = ∫(1+λs)ds = L + λ∫s`. The straight path **Pareto-dominates** the maximin
path in *both* coordinates (`L: 0.842 < 1.843` and `∫s: 0.095 < 0.460`), so
`J_maximin − J_straight = 1.001 + 0.362λ > 0` for **all** `λ ≥ 0`: the λ-sweep is a horizontal line (peak
stays `2.713` at `λ = 0,1,5,20,100`). The general statement:

> **Proposition (aggregative blindness to thin barriers).** For a ridge of height `H` and width `w`
> crossed transversally, the excess in the conformal functional scales as `λ·H·w`, while the minimax
> penalty stays `H`. Hence for every `λ` and every `H` there is a `w` small enough that the aggregative
> minimizer crosses. The aggregative criterion admits an **unbounded** suffering peak provided its duration
> is evanescent; the minimax does not.

This is the torture-vs-tickle objection to utilitarian aggregation, *derived over a field* — and §5 confirms
it is not a discretization artifact: refining the mesh (`NG = 100→800`) the straight-path `∫s` **converges**
(`0.1003 → 0.0964 → 0.0955 → 0.0952`) while the peak **rises** (`2.43 → 3.04`) as the thin spike is better
resolved. Genuinely tall barrier, evanescent integral cost — the effect strengthens under refinement.

**§2 — `c*` is a genuine topological obstruction, verified independently of the trajectory optimizer.**
Union-find sublevel-set percolation (add nodes in increasing `s`; the threshold at which the origin's
component first includes the target) returns `c* = 0.552`, **matching** the Dijkstra-minimax value; the
ambient floor (median `s` off the wall) is `0.050 ≪ c*`, so `c*` is a *pass*, not a background floor. The
"necessary suffering = geometry" claim now stands on two independent computations.

**§3 — the Pareto frontier `Φ(c) = min ∫s  s.t.  max s ≤ c`** (mask the field above `c`, re-run) replaces
the three points, and exposes the **true leximin**: `Φ(c*) = 0.144` at peak `c*` — the naive bottleneck
maximin paid `0.460` (bottleneck algorithms return an arbitrary representative of the optimal class, so they
**overpay ~3.2×**). Showing the naive maximin wastes *strengthens* the leximin recommendation, as predicted.

**§4 — the price of mercy** (the transportable scalar, definable on any field): `Δ∫s / Δpeak = 0.021`
(and `Δlength/Δpeak = 0.132`). Avoiding the acute spike (`peak 2.71 → 0.55`, a 4.9× reduction) costs only
`+0.046` in `∫s` and `+0.286` in length — **mercy is cheap here**, and the naive maximin's implied price
(`0.168`) overstated it ~8×. Report the *slope of `Φ`*, not three points.

**Honest reframing (owed).** "The algebra does not determine the ethics" is a conceptual thesis the figure
*illustrates*, not proves. What the figure *demonstrates* is the §1 Proposition — a stronger and citable
result. State it that way.

## μ*, L_lex, and the field taxonomy (third review)

**The price of mercy is a decision threshold, not "cheap."** For a decider with peak-aversion `μ`, criterion
`K = ∫s + μ·max s`: the aggregative path costs `0.098 + 2.713μ`, the leximin costs `0.144 + 0.552μ`, so
leximin wins iff `μ > 0.046/2.161 = 0.021`. The "price of mercy" **is** `μ*`, the **critical peak-aversion**,
and the aggregative criterion is exactly `μ = 0`. Publishable form:

> Choosing the aggregative trajectory on this field requires holding that one unit of *peak* suffering is
> worth **less than 2.1%** of one unit of accumulated suffering — a position no reflective agent and no
> clinical protocol sustains. Aggregationism is defeated not by moral appeal but by **revealed preference
> over a computed threshold**: "is your peak-aversion above 0.021?" — obviously yes.

**Missing number, now reported:** the leximin path length `L_lex = 1.128`. It is obligatory — without it the
reader cannot reconstruct `J(λ) = L + λ∫s` or verify dominance. And the dominance is **structural, not
lucky**: `J_lex − J_straight = (L_lex − 0.842) + λ(0.144 − 0.098)`; the straight segment is the Euclidean
length-minimizer so any deviation has `ΔL > 0` by definition, and `Δ∫s > 0` too, so the horizontal λ-sweep
is guaranteed by geometry.

**The taxonomy the frontier authorizes** (diagnosable from the field *before* choosing an ethics; barrier
cross-section `∼H·w` and detour geometry are independent):

| barrier | detour | consequence |
|---|---|---|
| thin | cheap | aggregation crosses the agony; maximin protection ~free — *the 2D-toy regime* |
| thin | expensive | the criteria diverge and it matters — the hard case |
| thick | cheap | both avoid; nothing to discuss |
| thick | expensive | both accept the pain; the question is only *where* to cross |

So the strong thesis is not "the algebra makes ethics explicit" but: **the geometry of the field determines
whether the ethical choice is consequential**, and there is a regime where Rawlsian protection comes at
utilitarian price. Which regime the *real* algebraic locus (`composition-failure-field.md`) occupies is the
open question that computation, not rhetoric, must answer.
