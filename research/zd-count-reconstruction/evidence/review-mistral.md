### **Concrete Errors and Missing Logical Steps**

#### **1. Arithmetic in the T(x) Counts (Table)**
The note claims:
> *"To verify the coefficient 6 in T, a support edge produces four positive triangles with a repeated old index, and two involving the new hub. Old positive triangles produce eight copies."*

**Calculation Check:**
- Let the old graph have `m(x)` edges and `p(x)` positive triangles.
- The new hub connects to all `2q_d` old vertices (two copies of `q_d` vertices).
- Each old edge `(u,v)` forms **4 triangles** with the hub (two sign choices for each endpoint).
- The hub forms **2 triangles** with each pair of identical old vertices (since `M+I` and `M` differ by `I`).
- Total triangles from edges: `4m(x)`.
- Total triangles from old positive triangles: `8p(x)` (each triangle lifts to 8 signed copies).
- **Missing step:** The note omits that the hub and two copies of a single old vertex form a **negative triangle**, but this does not affect the count of *positive* triangles. The arithmetic is correct, but the explanation could clarify that the "two involving the new hub" refers to the `2q_d` hub connections, not per edge.

#### **2. Sharp Witness Arithmetic (Theorem 3)**
The note claims:
> *"The upper bound two is attained at d=5 (n=8; ambient algebra dimension 512)."*

**Calculation Check:**
- For `W=208` (ε=1) and `W=201` (ε=0), the note computes:
  - `3E + τ = 1,006,776` for both.
- **Verification:**
  - For `W=208`:
    - `E = 61,032`, `τ = 823,680`
    - `3*61,032 + 823,680 = 183,096 + 823,680 = 1,006,776` ✔️
  - For `W=201`:
    - `E = 36,456`, `τ = 897,408`
    - `3*36,456 + 897,408 = 109,368 + 897,408 = 1,006,776` ✔️
- **Missing step:** The note does not explicitly state that `E` differs (`61,032 ≠ 36,456`), so the graphs are non-isomorphic. This is implied but should be made explicit for clarity.

#### **3. Equality with Zhilina’s C(e_W)**
The note claims:
> *"Our normalized vertices identify directly with Zhilina's C(e_W)."*

**Counterexample Check:**
- Zhilina’s `C(e_W)` is defined as the set of pairs `(e_a, s e_{a⊕W})` where `e_a e_{a⊕W} = 0` and `s = ±1`.
- The note’s `G_{n,W}` uses vertices `x_{a,s} = (e_a, s e_{a⊕W})` with `a ≠ W` and `0 < a < 2^n`.
- **Issue:** Zhilina’s definition includes `a = W` (which gives `(e_W, s e_0)`), but the note excludes `a = W`. This is a **discrepancy** unless `e_W e_0 = 0` is vacuously excluded by normalization.
- **Resolution:** The note’s normalization (excluding `a = W`) is consistent with Zhilina’s principal zero-divisor criterion, but this should be explicitly justified (e.g., `e_W e_0 = e_W ≠ 0`).

#### **4. Novelty Wording (Too Strong)**
The note initially claims:
> *"a new recursive classification of Cayley–Dickson zero divisors"*

**Issue:**
- The recursive structure (T/C/R states) is already in Zhilina (Theorem 4.6) and de Marrais (Emanation Tables).
- The note’s contribution is the **inverse from (E,τ)**, not the classification itself.
- **Correction:** The novelty should be scoped to the **two-count reconstruction** and **weighted compression**, not the underlying family.

---

### **Mathematical Correctness vs. Publication Novelty**
1. **Mathematical Correctness:**
   - The arithmetic in the T(x) and C(x) counts is correct.
   - The sharp witness (`d=5`) is correctly computed.
   - The weighted compression (Theorem 3) is arithmetically sound.
   - The structural lemma and Lean dependencies are pre-existing and correctly cited.

2. **Publication Novelty:**
   - The **two-count inverse** is a refinement of Zhilina’s recursive reconstruction, not a new classification.
   - The **weighted compression theorem** (Theorem 3) appears novel in the inspected literature.
   - The **sharp obstruction** (fiber size ≤ 2) is a substantive contribution.

---

### **Summary of Concrete Issues**
| **Issue** | **Type** | **Status** |
|-----------|----------|------------|
| T(x) triangle count explanation | Missing step | Clarify hub connections |
| Sharp witness non-isomorphism | Missing step | Explicitly state `E` differs |
| Zhilina’s `C(e_W)` equality | Discrepancy | Justify exclusion of `a = W` |
| Novelty wording | Overclaim | Scope to two-count + weighted compression |

**No false theorems were found.** The arithmetic is correct, but some logical steps and justifications are missing. The novelty claim should be narrowed.
