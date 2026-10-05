**CLEAR**

Review only. No semantic expected values were generated. Checks used the given recurrence, the frozen dump as an already-authored artifact, and integer Walsh identities.

### Convention X recurrence
`cd_sign` is the standard Cayley–Dickson product \((a,b)(c,d)=(ac-\overline{d}b,\,da+b\overline{c})\) on basis indices:
- real unit: \(\sigma(0,\cdot)=\sigma(\cdot,0)=+1\)
- base square: `bits <= 1` and both nonzero \(\Rightarrow -1\) (this is what the gate mutates)
- lo–lo: \(\sigma(a_\mathrm{lo},b_\mathrm{lo})\)
- lo–hi: \(\sigma(b_\mathrm{lo},a_\mathrm{lo})\)
- hi–lo: \(-\sigma(a_\mathrm{lo},b_\mathrm{lo})\)
- hi–hi: \(-\sigma(0,a_\mathrm{lo})\) if \(b_\mathrm{lo}=0\), else \(\sigma(b_\mathrm{lo},a_\mathrm{lo})\)

Spot-checks against the quaternion/octonion table (e.g. \(e_1e_2=+e_3\), \(e_1e_6=-e_7\), \(e_5e_6=-e_3\), \(e_6e_7=-e_1\)) hold. The `b_lo==0` arm in the hi–lo branch is dead (global \(b=0\) already returned \(+1\)) but equal to the live answer. Sedenion doubling is the same recurrence; inverted freeze rows are consistent (e.g. \(\sigma(1,8)=+1\), \(\sigma(8,1)=-1\), \(\sigma(4,8)=+1\)).

Row \(d=0\) is independently fixed by Convention X: \(S_0(0)=1\), \(S_0(i)=-1\) for \(i\neq 0\), Walsh \((-14,2,\ldots,2)\). That matches the freeze without using it as a generator.

### Inverse Walsh / Parseval (integer)
Unnormalized characters \(\chi_k(i)=(-1)^{\mathrm{parity}(k\land i)}\) are the correct \(\mathbb{Z}_2^4\) Walsh characters. Forward \(W=HS\) and check \(H^\top W = nS\) are implemented as
- `energy == dimension * dimension` (\(\|W\|_2^2=n\|S\|_2^2=n^2\) for \(\pm 1\) rows)
- `reconstructed == dimension * cd_sign(...)` (no division)

Those two checks prove Walsh invertibility and \(\pm 1\)-valued rows. They do **not** prove Convention X by themselves (any \(\pm 1\) table would invert). Convention X is bound by Sounio + freeze + the base-case mutant, which is the stated authority split.

Parseval on the printed rows is self-consistent (including \(d=9\ldots 15\) in \(\{-6,2,10\}\)). The weighted checksum \(\sum(d+1)(k+1)W[d,k]=21232\) matches the freeze/receipt. Support \(256/256\) and `REFUSED_AT_DIMENSION_16` match the dump.

### Mutation controls
- Convention X: `return 0-1` → `return 1` must still emit a spectrum and must **differ** from the freeze. Correct `set -e` / `&&` behavior. Protects the \(i^2=-1\) base; doubling-branch edits are not mutated, but they are covered by freeze `cmp` whenever source and freeze are not edited together.
- Constant character: `if bit_parity(...) == 0` → `if true` must exit nonzero and print `PIREUS_WALSH_TWIST_CHANNEL_SPECTRUM_FAIL`. Constant \(\chi\equiv 1\) breaks Parseval (row \(d=0\) energy \(16\cdot 14^2\neq 256\)) and reconstruction. Strings are unique in the source; `sed` is brittle but currently accurate.

Receipt hashes, `SEMANTIC_AUTHORITY`, `WalshCharacterChannels`, `equivalence=EXACT`, sparsity refusal, and claim boundary are grepped. Gate does not independently re-hash mutants; it does not need to.

### Bounded claims
Forbidden promotions are not asserted. Code/receipt/audit agree on exactness only for the tested dimension-16 sign tensor, dense support \(\Rightarrow\) no sparsity promotion, no asymptotic density, hardware/material pending. The displayed identity
\[
r[d]=\frac1{16}\sum_k W[d,k]\sum_i \chi_k(i)\,a[i]\,b[i\oplus d]
\]
is the exact rewriting of \(\sum_i S_d(i)a[i]b[i\oplus d]\). No \(a,b\) product is executed; sign reconstruction is algebraically enough.

### Algebraic exactness vs floating-point runtime
Authority path is `i64` only: coefficients, energy, checksum, and \(nS\) reconstruction. No float, no `/16` at runtime. The prose \(1/16\) is the algebraic inverse; executability is the multiplied form, so truncation/rounding cannot enter this gate. \(n=16\) values are far inside `i64`.

**Verdict:** recurrence, integer inverse/Parseval, mutations, and claim boundaries are consistent. No BLOCKER/MAJOR.
