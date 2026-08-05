# Dual math-review disposition

Date: 2026-08-05

Providers: xAI Grok 4.3 and Z.AI GLM-5.2.

## Accepted findings

- Both providers accepted the unstable degree `+1` argument and the combination of CAPD exit evidence with Arb stable-support and determinant evidence when bound to the same leaf, map, section, and coordinates.
- Z.AI independently rederived the CAPD Taylor convention, the `delta0=xi/256` scale, the C2 mean-value derivative, both `1/2` diagonal factors, the mixed term, the anchored inequality, the face-width subtraction, and the target-chart construction. It found the local covering claim sound and kept recurrent graph and chaos out of scope.
- Grok's MINOR request that the square operation be explicit is already met by `Interval.square()`: it returns `[0,max(a^2,b^2)]` when the input crosses zero. The review request also called the squares dependency-aware and nonnegative.

## Documented disagreements with Grok

### Hessian factor claim

Grok labeled the mixed-term handling wrong because it expected the diagonal factor `2` to be propagated to the off-diagonal term. This contradicts both the executed receipt and the executed worker source.

The receipt states:

```
DIAGONAL_TAYLOR_TO_DERIVATIVE_FACTOR=2
OFFDIAGONAL_TAYLOR_TO_DERIVATIVE_FACTOR=1
```

The hash-bound worker implements:

```cpp
return first == second ? 2.0 * coefficient : coefficient;
```

Thus

```
h00 = 2*c.D2P00
h01 = 1*c.D2P01
```

is the worker's actual convention. In the quadratic Taylor form, `1/2 d^T H d`, the two symmetric mixed entries produce `h01*d0*d1`, while a diagonal produces `(1/2)*h00*d0^2`. Z.AI independently derived exactly this formula. The Grok objection is rejected as a convention misread.

### Anchored monotonicity claim

Grok requested an additional right-face monotonicity check. The derivative enclosure is not a left-face enclosure: it evaluates

```
g0 + H00*[-1/256,1/256] + H01*[-1/512,1/512]
```

over the entire closed leaf, including both faces. Therefore for each fixed eta,

```
U(1,eta)-U(-1,eta) >= 2*inf(dU/dxi).
```

Taking the minimum over eta gives

```
min U_right >= min U_left + 2L >= left.lower + 2L.
```

The separation from the left upper enclosure is consequently

```
(left.lower + 2L) - left.upper
= 2L - width(left),
```

which is the exact positive gap in the aggregate. Z.AI independently checked this arithmetic. The Grok objection is rejected because the whole-leaf enclosure already supplies the requested right-face coverage.

### QR paragraph

Grok called the QR falsifier irrelevant to the local covering claim. It is mathematically separate, as the receipt and contract state, but it remains in the result because it directly answers the user's primary `18x` falsifier before the alternative route. No covering inference depends on the QR result.

## Final disposition

Outcome: `PASS_WITH_DOCUMENTED_DISAGREEMENTS`.

No reviewer finding required a mathematical or code correction. The gate remains fail-closed, performs deterministic raw-evidence reanalysis, and rejects 12 negative mutations. The certified scope is exactly one local h-set covering relation. Recurrent graph, Fibonacci covering, global HPG, chaos, chaotic attractor, open-problem solution, novelty, and priority remain false.
