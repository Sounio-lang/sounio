# Recovering principal Cayley–Dickson components from two counts

Research note — Demetrios Chiuratto Agourakis / Sounio research.
Session date: 2026-09-09 (America/Sao_Paulo).
Status: theorem extraction and bounded prior-art assessment; not a submission or a claim of established priority.

## Result and exact object

Let A_0=R and A_{n+1}=A_n ⊕ A_n, with
\[
(a,b)(c,d)=(ac-\bar d b,\;da+b\bar c).
\]
Use the recursively ordered standard basis. For n≥3 and 0<W<2^n, let G_{n,W}
have vertices
\[
x_{a,s}=(e_a,s e_{a\oplus W}),\qquad
0<a<2^n,\quad a\ne W,\quad s\in\{-1,1\}.
\]
Distinct vertices are adjacent exactly when both ordered products vanish.
These are normalized representatives of real lines, not all nonzero scalar
multiples as separate vertices. There are 2^(n+1)-4 vertices.

**Theorem 1.** At fixed n≥3,
\[
G_{n,W}\cong G_{n,V}
\iff (E(G_{n,W}),\tau(G_{n,W}))=(E(G_{n,V}),\tau(G_{n,V})),
\]
where E counts unordered edges and τ counts unordered triangles.

This is the existing all-depth theorem
[SounioZDSignedState.native_iso_iff_counts](../../formal/lean4/SounioZDSignedState.lean).
The present work extracts its inverse algorithm and human-readable argument.
It is not a newly proved Lean theorem in this session.

## The signed reduction and its counting argument

Put d=n-3 and q_d=2^(d+2)-1. The normal-form catalogue has base states K,Z
at depth zero, and R,T(x),C(x) at each subsequent depth. R can terminate
at any positive remaining depth. The signed matrices are K=I_3-J_3, Z=0,
R=I-J at the current order. For an order-q matrix M with zero diagonal,
\[
T(M)=
\begin{pmatrix}
0&{\bf1}^{t}&-{\bf1}^{t}\\
{\bf1}&M&M+I\\
-{\bf1}&M+I&M
\end{pmatrix},
\qquad
C(M)=
\begin{pmatrix}
0&0&0\\
0&M&M\\
0&M&M
\end{pmatrix}.
\]
Entries belong to {−1,0,1}. Let m be the number of nonzero unordered
off-diagonal pairs and p the number of unordered triangles whose three
signs multiply to +1. Direct counting gives:

| State at depth d+1 | m | p |
|---|---:|---:|
| R | q_d(2q_d+1) | 0 |
| T(x) | 3q_d+4m(x) | 6m(x)+8p(x) |
| C(x) | 4m(x) | 8p(x) |

The base counts are K:(3,0), Z:(0,0). The table describes a step from d to
d+1: when reconstructing a state at depth d≥1, use q=q_{d−1}. Thus the
n=4 examples below have d=1 and predecessor q_0=3, not q_1=7.
To verify the coefficient 6 in T, a support edge produces four positive
triangles with a repeated old index. Of its four triangles involving the
hub, exactly two are positive: their sign is (−1)^(α+β) M(r,s), with
α,β∈{0,1}. Old positive triangles produce eight copies.
The hub and the two copies of a single old index form a negative triangle.

**Structural lemma (existing formal dependency).** G_{n,W} is isomorphic to
the signed double cover of the first-channel normal form, with each cover
vertex replaced by two independent twins. More explicitly, the cover has
vertices (r,s,c), with s=±1 and c∈{0,1}, and adjacency s t=M(r,u).
The first-channel code is defined simultaneously with its auxiliary channel:
at d=0 use (K,Z); at depth d+1 let H=2^(d+3), and use
\[
(X,Y)(W)=
\begin{cases}
(TX(W),CY(W)),&W<H,\\
(R,C^{d+1}Z),&W=H,\\
(TY(W-H),CX(W-H)),&W>H,
\end{cases}
\]
with right-hand channels taken at depth d.

The algebra-to-matrix reduction is supplied by the existing arbitrary-depth
proofs native_adjacency_iff, actualNormalizations and actualNativeIso.
Their induction splits indices at H, pairs a with a⊕W, and switches signs
before applying the displayed blocks. This note uses that structural lemma;
it does not replace its full formal proof with numerical evidence.

Every support edge lifts to two edges, then to eight under duplication.
Every positive signed triangle has two compatible sign lifts, then sixteen
under duplication; negative triangles have none. Consequently
\[
E=8m,\qquad \tau=16p.
\]

## Reconstruction proof and executable inverse

**Lemma 2.** At a fixed depth the map x↦(m(x),p(x)) is injective.

At depth zero this is immediate. At the next depth, q=q_d is odd.
The C branch has even m, whereas R and T have odd m. Thus equality of
counts distinguishes C from the other branches. R cannot collide with T:
p(Tx)=0 forces m(x)=p(x)=0, whence m(Tx)=3q, strictly less than
q(2q+1), since q≥3. Two T states with equal counts have equal predecessor
m from their first equation, then equal predecessor p from their second;
two C states likewise. Induction finishes the proof.

For a constructive inverse, first recognize (q(2q+1),0) as R.
Otherwise use parity and the pullbacks
\[
C:\ (m,p)\mapsto(m/4,p/8),\qquad
T:\ (m,p)\mapsto((m-3q)/4,\,[p-6(m-3q)/4]/8).
\]
Reject a negative or nonintegral pullback and reject anything except
(3,0) or (0,0) at the base. Recognition of R is unambiguous by the argument
above. For valid states the forward equations give exactly the displayed
numerators 4m(x) and 8p(x), so divisibility is automatic. Conversely, each
accepted inverse step re-encodes to its input pair. Together with the base
check, this proves soundness and completeness on the catalogue. There are at most d pullback steps; this counts integer
arithmetic operations, not constant-time operations on arbitrarily large integers.

Graph isomorphisms preserve E and τ. Conversely, equal counts yield equal
codes by Lemma 2, hence isomorphic covers by the structural lemma.
This proves Theorem 1.

[decoder.py](decoder.py) implements this inverse with integer arithmetic.
Its native entry point rejects the auxiliary C/Z channel.
**Precondition:** the graph belongs to the stated family. Matching a count
pair does not certify membership of an arbitrary graph. The output is a
canonical isomorphism class, not recovery of the original W.

Neither count can be omitted from this particular pair, already for n=4:

| W | Code | E | τ |
|---:|---|---:|---:|
| 1 | TK | 168 | 288 |
| 8 | R | 168 | 0 |
| 9 | TZ | 72 | 0 |

The first two graphs refute completeness of E alone; the last two refute
completeness of τ alone. At n=3 all allowed labels have code K, so n=4 is
the first level of the family where either omission can fail. This is
minimality only with respect to deleting coordinates of (E,τ), not a
lower bound against every possible scalar encoding.

## Uniform weighted compression and its sharp obstruction

Define ε(x)=1 if the terminal state is K or R, and ε(x)=0 if it is Z;
T and C preserve ε. Since the two counts recover the code, ε is an
invariant of the graph's isomorphism class, not extra vertex labelling.
Let a be a positive odd integer, b=2^j with j≥0,
and 2a≤3b. Write F(x)=a m(x)+b p(x).

**Theorem 3 (existing formal result).** At fixed depth, F is injective on
each origin class ε=0 and ε=1. Thus (2aE+bτ,ε) is complete for the native
graphs, and each scalar value occurs in at most two isomorphism classes.

Here is the arithmetic behind the result. Since p is even, the parity
of F distinguishes C from R/T. Under a common C branch, the weight pair
pulls back to (a,2b), after division by 4. Under a common T branch it pulls
back from (4a+6b,8b) by removing a common power of two:

| Original weights | Divisor | New weights |
|---|---:|---|
| (1,1) | 2 | (5,4) |
| (1,2) | 16 | (1,1) |
| (3,2) | 8 | (3,2) |
| b≥4 | 4 | (a+3b/2,2b) |

All new pairs satisfy the same admissibility conditions. The ε=0 class
contains no R state, so induction applies immediately.

For ε=1, the remaining comparison is R versus T(x).
Let F_d=q_d(q_d−1)/2 be the maximum m at depth d.
Core states satisfy m(x)≥3·4^d, and F_d<8·4^d.
If x=K or R, m(x)=F_d and T(x) has the reset edge count but positive p.
If x=T(y) with core origin, the recurrences give
3m(x)+4p(x)>3F_d; together with 2a≤3b this makes F(Tx)>F(R).
If x=C(y), at its parent level q≡7 (mod 8), and
F(R)≡a while F(T(Cy))≡5a (mod 8), which differ because a is odd.
This excludes the last collision and completes the induction.

The native statement follows from 2aE+bτ=16F.
The formal statements are origin_value_injective,
native_iso_iff_scalar_bit and native_scalar_at_most_two in
[SounioZDScalarBit](../../formal/lean4/SounioZDScalarBit.lean), with the
density and weight lemmas in SounioZDScalarCore.

The upper bound two is attained at d=5 (n=8; ambient algebra dimension 512):

| W | ε | E | τ | 3E+τ |
|---:|---:|---:|---:|---:|
| 208 | 1 | 61,032 | 823,680 | 1,006,776 |
| 201 | 0 | 36,456 | 897,408 | 1,006,776 |

The graphs are nonisomorphic because their edge counts differ.
The admissible weights (a,b)=(3,2) produce 6E+2τ, twice the displayed scalar, namely 2,013,552 in both cases.
The formal witness is native_scalar_two_sharp.
This shows that one extra bit is necessary for this scalar on the full
family. It does not assert that every admissible scalar has a collision.

## Bibliographic verdict and claim proposed for a short note

The graph family is established prior art. Under the same doubling convention,
\[
e_a(s e_{a\oplus W})=\pm e_W,
\]
so our normalized vertices identify directly with Zhilina's C(e_W).
The inverse takes the normalized first coordinate and the sign of the second.
In her Corollary 3.21, for n≥3 and principal χ=−1, the excluded second
coordinates are ±e_a and ±e_0. W≠0 excludes the first, and a≠W excludes
the second. The coordinate product is ±e_W, not zero. Both graph definitions
use vanishing of both products of the doubled elements. This is a mathematical
identification from the definitions, not a new machine-checked theorem.

The detailed comparison is in [BIBLIOGRAPHY.md](BIBLIOGRAPHY.md).
**Decision:** discard “a new family / a new recursive classification of
Cayley–Dickson zero divisors” as a novelty claim. The two-count statement
is an elementary reconstruction refinement once the signed recurrence is
available; its short proof alone is not evidence of a substantial new
classification theory.

Retain the narrower candidate: **uniform reconstruction from two motif
counts, strengthened by an admissible-weight theorem with scalar fibres of
size at most two and a sharp obstruction.** No matching theorem was located
in the inspected sources. This supports a scoped novelty candidate for a
short note, not an established priority claim or a promise of publication.

Proposed mathematical claim:
“Within the principal basis-pair components of fixed dimension, edge and
triangle counts determine the isomorphism class. For positive odd a and
dyadic b satisfying 2a≤3b, each value of 2aE+bτ has at most two such classes;
an origin bit completes the invariant, and the bound is sharp.”

The first theorem is dependent on the existing structural reduction;
the weighted theorem is the substantive arithmetic refinement.
No scalar residual-separation conjecture or finite search depth enters
either theorem.

## Validation and use

Run from this directory:
    python3 test_decoder.py
    python3 decoder.py 1 168 288

The tests compare the inverse with the complete small catalogue, reject
invalid count pairs, exercise large integers at depth 64, and independently
construct native graphs by sparse Cayley–Dickson multiplication, including
both displayed scalar witnesses. These are implementation checks; the
unbounded theorems are supplied by the proofs above and the existing Lean
dependencies. See CHECKS.md for the current execution and review results.
