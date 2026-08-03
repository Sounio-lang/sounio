The user wants me to review the math in this Python script.
The script generates "dyadic target-23 Picard-step known-answer vectors".
I need to check:
- Domain in scope: "Imprecise probability / p-boxes: containment under monotone transforms, interval-extension arithmetic, four-corner enumeration for non-monotone ops."
- Interval arithmetic functions: `enclose`, `add`, `neg`, `sub`, `mul`, `div2`, `scale2`.
- Picard iteration and Lipschitz bounds for a system of ODEs.

Let's look at `enclose`:
```python
def enclose(value: Fraction) -> Interval:
    return floor_fraction(value * ONE), ceil_fraction(value * ONE)
```
Wait, if `value * ONE` is exactly an integer, then `floor_fraction` and `ceil_fraction` are identical. That's fine, it encloses the exact value.

Let's look at `add`:
```python
def add(left: Interval, right: Interval) -> Interval:
    return left[0] + right[0], left[1] + right[1]
```
Correct.

Let's look at `neg`:
```python
def neg(value: Interval) -> Interval:
    return -value[1], -value[0]
```
Correct.

Let's look at `sub`:
```python
def sub(left: Interval, right: Interval) -> Interval:
    return add(left, neg(right))
```
Correct.

Let's look at `mul`:
```python
def mul(left: Interval, right: Interval) -> Interval:
    corners = [Fraction(a * b, ONE) for a in left for b in right]
    return floor_fraction(min(corners)), ceil_fraction(max(corners))
```
Correct (four-corner method, though `a` and `b` are integers representing `a/ONE`. So `a*b/ONE` is correct).

Let's look at `div2`:
```python
def div2(value: Interval) -> Interval:
    return floor_fraction(Fraction(value[0], 2)), ceil_fraction(Fraction(value[1], 2))
```
Correct, divides the bounds by 2, assuming `value` is positive? Wait, if `value[0]` is negative, `value[0] / 2` is the lower bound, and `value[1] / 2` is the upper bound. This is exact interval division by 2.

Let's look at `scale2`:
```python
def scale2(value: Interval) -> Interval:
    return value[0] * 2, value[1] * 2
```
Correct.

Let's check `field` (the ODE RHS):
```python
def field(state: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    x, y, w, _ell = state
    yy = mul(y, y)
    xy = mul(x, y)
    wzs = add(w, zs)
    return (
        sub(scale2(yy), xy),
        sub(xy, div2(mul(y, wzs))),
        sub(sub(xy, w), zs),
        sub(sub(sub(x, y), div2(wzs)), (ONE, ONE)),
    )
```
This computes `dx/dt = 2*y^2 - x*y`, etc. This is just interval arithmetic evaluation of a polynomial/rational vector field.

Let's look at `lipschitz_rows`:
This computes the Lipschitz constant rows for the vector field `field`.
```python
def lipschitz_rows(box: tuple[Interval, ...], zs: Interval) -> tuple[int, ...]:
    x, y, w, _ell = box
    wzs = add(w, zs)
    row0 = absolute_upper(y) + absolute_upper(sub(scale2(scale2(y)), x))
    row1 = absolute_upper(y) + absolute_upper(sub(x, div2(wzs))) + ceil_fraction(Fraction(absolute_upper(y), 2))
    row2 = absolute_upper(y) + absolute_upper(x) + ONE
    row3 = 2 * ONE + ONE // 2
    return row0, row1, row2, row3
```
Let's trace `row0`:
`f0 = 2*y^2 - x*y`
Partial derivatives:
`df0/dx = -y`
`df0/dy = 4*y - x`
`df0/dw = 0`
`df0/dell = 0`
Row 0 norm (L1 norm of Jacobian row):
`|y| + |4*y - x|`
Wait, `absolute_upper(sub(scale2(scale2(y)), x))`.
`scale2(scale2(y))` is `y * 4`. So `sub(scale2(scale2(y)), x)` is `4*y - x`.
Then `absolute_upper` takes the max of the absolute values of the endpoints.
Is this an upper bound for `|4*y - x|`? Yes, because for any interval `I`, `|x| <= max(|I[0]|, |I[1]|)`.
So `row0` is `|y| + |4*y - x|` bound. This is correct.

Let's trace `row1`:
`f1 = x*y - 0.5 * y * (w + zs)`
Partial derivatives:
`df1/dx = y`
`df1/dy = x - 0.5 * (w + zs)`
Wait, the script has:
`row1 = absolute_upper(y) + absolute_upper(sub(x, div2(wzs))) + ceil_fraction(Fraction(absolute_upper(y), 2))`
Wait, `df1/dy = x - 0.5 * wzs`.
But the script adds `ceil_fraction(Fraction(absolute_upper(y), 2))`. Where does this come from?
Ah! `w` is a state variable!
So `f1 = x*y - 0.5 * y * (w + zs)`.
`df1/dx = y`
`df1/dy = x - 0.5 * (w + zs)`
`df1/dw = -0.5 * y`
`df1/dell = 0`
So the L1 norm of the gradient of `f1` is:
`|y| + |x - 0.5 * (w + zs)| + |-0.5 * y|`
Which is `|y| + |x - 0.5 * wzs| + 0.5 * |y|`.
The script computes:
`absolute_upper(y)`  <-- `|y|`
`absolute_upper(sub(x, div2(wzs)))` <-- `|x - 0.5 * wzs|`
`ceil_fraction(Fraction(absolute_upper(y), 2))` <-- `ceil(0.5 * |y|)`
This matches exactly! It is a correct upper bound for the L1 norm of the gradient.

Let's trace `row2`:
`f2 = x*y - w - zs`
Partial derivatives:
`df2/dx = y`
`df2/dy = x`
`df2/dw = -1`
`df2/dell = 0`
L1 norm: `|y| + |x| + 1`.
Script:
`absolute_upper(y) + absolute_upper(x) + ONE` <-- `1` is `ONE` since the fixed point representation is fractional where `1.0 = ONE`.
This is correct!

Let's trace `row3`:
`f3 = x - y - 0.5*(w + zs) - 1.0`
Partial derivatives:
`df3/dx = 1`
`df3/dy = -1`
`df3/dw = -0.5`
`df3/dell = 0`
L1 norm: `1 + 1 + 0.5 = 2.5`.
Script: `2 * ONE + ONE // 2` <-- `2.0 + 0.5 = 2.5`.
This is correct!

Wait, what about `ell`? The fourth state variable is `_ell`.
Is there a derivative of `ell`? The script assumes `ell` is in the state but `f0, f1, f2, f3` do not depend on `ell`, and their derivatives with respect to `ell` are 0. Thus `ell` does not appear in the Jacobian bounds. That's fine.

What about `div2(mul(y, wzs))`?
In `field`:
`div2(mul(y, wzs))`.
If `y` and `wzs` are intervals, `mul(y, wzs)` computes the exact product as an interval, then `div2` divides the endpoints by 2. This is correct interval arithmetic.

Let's check `status`:
```python
    if any(not -DOMAIN_LIMIT < endpoint < DOMAIN_LIMIT for interval in intervals for endpoint in interval):
        return -2
```
This correctly bounds endpoints.

```python
    candidate = image(initial, box, zs)
    if any(not (container[0] < value[0] and value[1] < container[1]) for container, value in zip(box, candidate, strict=True)):
        return -4
```
Strict containment check (Picard step maps into the interior of the box). Correct.

```python
    contraction = ceil_fraction(Fraction(max(lipschitz_rows(box, zs)), STEP_DENOMINATOR))
    if contraction >= ONE:
        return -5
```
Max Lipschitz constant divided by `STEP_DENOMINATOR` (which is 256).
Wait, `STEP_DENOMINATOR` is `1 / dt`.
`time = (0, ONE // STEP_DENOMINATOR)`
`dt = 1/256`.
The Picard operator is `P(x) = x0 + dt * f(x)`.
The Lipschitz constant of `P` is `dt * L`, where `L` is the Lipschitz constant of `f`.
`L` is `max(lipschitz_rows(box, zs))`.
`dt` is `1/256` in real numbers, so in fixed point it's `ONE / 256`.
Then `L * dt` is `max(lipschitz_rows) * (ONE / 256)`.
The code uses `ceil_fraction(Fraction(max(lipschitz_rows(box, zs)), STEP_DENOMINATOR))`.
Wait, `max(lipschitz_rows)` is in fixed point format. It's `L_fixed = L_real * ONE`.
`Fraction(L_fixed, STEP_DENOMINATOR)` is `(L_real * ONE) / 256`.
Then `ceil_fraction` takes the ceiling of this value in the fixed point representation.
If this value `>= ONE` (which is 1.0), then `L_real * dt >= 1.0`, which means no contraction.
This is correct!

Wait, look at `image`:
```python
def image(initial: tuple[Interval, ...], box: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    time = (0, ONE // STEP_DENOMINATOR)
    return tuple(add(component, mul(time, derivative)) for component, derivative in zip(initial, field(box, zs), strict=True))
```
`time` is `(0, ONE // STEP_DENOMINATOR)`.
Wait, `ONE` is `1 << 96`. `ONE // 256` is exact since `256` divides `2^96`. So `time` is exactly `(0, ONE // STEP_DENOMINATOR)`.
Then `mul(time, derivative)` correctly propagates the interval `time`. Since `time` is `[0, dt]`, `mul` with `derivative` gives the interval product.

Let's check `initial_state`:
```python
    decimal = Fraction
    u = -decimal("0.004") + Fraction(447, 2) * decimal("0.008") / 256
    s = -decimal("0.3") + Fraction(651, 2) * decimal("0.6") / 512
    x = decimal("15.186446520640786") + decimal("-0.67430316214199759") * u + decimal("-0.94170446778164518") * s
    y = decimal("10.908543194765466") + decimal("-0.73845463335624273") * u + decimal("0.33644122125579123") * s
```
No math error here, just some specific constants. `Fraction(447, 2) * decimal("0.008")` is evaluated correctly.

Let's check `fixed_box`:
```python
def fixed_box(initial: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    box = initial
    for _ in range(64):
        candidate = image(initial, box, zs)
        widened = tuple((min(old[0], new[0]), max(old[1], new[1])) for old, new in zip(box, candidate, strict=True))
        if widened == box:
            return tuple((lower - BOX_INFLATION_RAW, upper + BOX_INFLATION_RAW) for lower, upper in box)
        box = widened
    raise ValueError("Picard candidate construction did not stabilize")
```
This computes a fixpoint by iteratively taking the union of `box` and `image(initial, box, zs)`. It inflates the box by `BOX_INFLATION_RAW` at the end to ensure strict containment (since `status` checks `container[0] < value[0]`). This is a standard interval Picard iteration.

Are there any subtle math errors?
In `lipschitz_rows`:
```python
    row1 = absolute_upper(y) + absolute_upper(sub(x, div2(wzs))) + ceil_fraction(Fraction(absolute_upper(y), 2))
```
Wait. The derivative of `f1 = x*y - 0.5 * y * wzs` with respect to `y` is `x - 0.5 * wzs`.
The derivative with respect to `x` is `y`.
The derivative with respect to `w` is `-0.5 * y`.
The derivative with respect to `zs` is `-0.5 * y`.
Does `zs` have any variation? No, `zs` is a constant parameter. The Jacobian should only be taken with respect to state variables `x, y, w, ell`.
So `df1/dw = -0.5 * y`.
The absolute upper bound for this is `0.5 * absolute_upper(y)`.
The code has `ceil_fraction(Fraction(absolute_upper(y), 2))`. Since `absolute_upper(y)` is an integer (representing the upper bound in fixed point), `ceil` of it divided by 2 is a valid upper bound for `0.5 * absolute_upper(y)`.
This is completely correct.

Wait, is `scale2` correct?
`scale2(value: Interval) -> Interval: return value[0] * 2, value[1] * 2`
Yes.

Is `sub(scale2(scale2(y)), x)` correct for `4y - x`?
`scale2(scale2(y))` -> `y*2*2 = 4y`. Correct.

Is `absolute_upper(y) + absolute_upper(sub(scale2(scale2(y)), x))` correct for the row 0 norm?
`f0 = 2*y^2 - x*y`
Gradient:
`df0/dx = -y`
`df0/dy = 4y - x`
So L1 norm of gradient is `|-y| + |4y - x| = |y| + |4y - x|`.
The code has `absolute_upper(y) + absolute_upper(sub(scale2(scale2(y)), x))`. Correct.

Let's check `f2 = x*y - w - zs`
`df2/dx = y`
`df2/dy = x`
`df2/dw = -1`
So `|y| + |x| + |-1|`.
Code: `absolute_upper(y) + absolute_upper(x) + ONE`. Correct.

Let's check `f3 = x - y - 0.5*(w + zs) - 1.0`
`df3/dx = 1`
`df3/dy = -1`
`df3/dw = -0.5`
So `1 + 1 + 0.5 = 2.5`.
Code: `2 * ONE + ONE // 2`. Correct.

Are there any issues with `ONE // STEP_DENOMINATOR`?
`STEP_DENOMINATOR` is 256.
`1 << 96 // 256 = 1 << 88`, exactly.

What about `floor_fraction` and `ceil_fraction`?
```python
def floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator

def ceil_fraction(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)
```
`Fraction` always normalizes such that `denominator > 0` and `numerator` has the sign.
So `value.numerator // value.denominator` is exact integer division rounding towards negative infinity. This is exactly `floor`.
And `-((-value.numerator) // value.denominator)` is `- (ceil(num/denom))` which is exactly `ceil`.
This is correct.

What about `mul(left: Interval, right: Interval)`?
```python
def mul(left: Interval, right: Interval) -> Interval:
    corners = [Fraction(a * b, ONE) for a in left for b in right]
    return floor_fraction(min(corners)), ceil_fraction(max(corners))
```
`a` and `b` are integers representing `a / ONE` and `b / ONE`.
The product is `(a * b) / (ONE * ONE)`.
But the code computes `Fraction(a * b, ONE)`, which is `(a * b) / ONE`.
So it correctly scales the product back to the fixed point representation!
And it evaluates all 4 corners. This is correct.

What about `div2`?
```python
def div2(value: Interval) -> Interval:
    return floor_fraction(Fraction(value[0], 2)), ceil_fraction(Fraction(value[1], 2))
```
`value` is in fixed point. To divide by 2, we just divide the fixed point value by 2. Since `Fraction(value[0], 2)` uses the integer `value[0]`, it divides it by 2 exactly.
This is correct.

Is there any interval math error?
In `add`:
`add(left, right)` adds lower bounds and upper bounds. Correct.
In `neg`:
`neg(value)` negates upper bound and lower bound and swaps them. Correct.
In `sub`:
`add(left, neg(right))`. Correct.

Let's check `enclose`:
```python
def enclose(value: Fraction) -> Interval:
    return floor_fraction(value * ONE), ceil_fraction(value * ONE)
```
Converts `value` (a `Fraction` in real numbers) to fixed point by multiplying by `ONE` and taking floor/ceil. Correct.

Wait, look at `reversed_initial = ((initial[0][1], initial[0][0]), *initial[1:])`
This reverses the bounds of `initial[0]` so lower > upper.
Then `status` checks:
`if any(lower > upper for lower, upper in intervals): return -1`
This will trigger `return -1`. Correct.

`outside_initial = (((DOMAIN_LIMIT, DOMAIN_LIMIT)), *initial[1:])`
This sets the bounds of `initial[0]` to `DOMAIN_LIMIT`.
Then `status` checks:
`if any(not -DOMAIN_LIMIT < endpoint < DOMAIN_LIMIT for interval in intervals for endpoint in interval): return -2`
Since `endpoint` is `DOMAIN_LIMIT`, `not -DOMAIN_LIMIT < DOMAIN_LIMIT` is True, so it returns `-2`. Correct.

Let's check `uninflated`:
```python
    uninflated = tuple((lower + BOX_INFLATION_RAW, upper - BOX_INFLATION_RAW) for lower, upper in box)
```
`box` was inflated by `BOX_INFLATION_RAW`.
`uninflated` takes the original pre-inflated box.
Then `candidate = image(initial, uninflated, zs)` is computed.
Wait, `status` has:
```python
    candidate = image(initial, box, zs)
    if any(not (container[0] < value[0] and value[1] < container[1]) for container, value in zip(box, candidate, strict=True)):
        return -4
```
If `box` is `uninflated`, `candidate` might touch the boundary or fall outside.
If it falls outside or touches, `not (container[0] < value[0])` will be True, returning `-4`.
But is `uninflated` guaranteed to fail this?
Since `box` was the fixpoint of the Picard iteration *before* inflation, `candidate` (which is `image(initial, box, zs)`) is a subset of `box`.
Wait, the Picard fixpoint means `image(initial, box) \subseteq box`.
So for `uninflated`, `container[0] <= value[0]` and `value[1] <= container[1]`.
But does it satisfy strict inequality?
Probably not. The Picard iteration usually results in at least one bound being exactly equal or very close. If it's exactly equal, `container[0] < value[0]` is False, so it returns `-4`.
Even if it happens to be strictly contained, the inflated box is the one returned by `fixed_box`. The test just uses `uninflated` to check the refusal logic of the circuit. So mathematically it works as a test case.

Is there any mathematical claim or assertion that might be wrong?
The Lipschitz constant bounds for `field` are exactly the L1 norms of the gradients, which are upper bounds for the operator norm of the Jacobian.
The `lipschitz_rows` computes `max(|I|)` for each term in the gradient.
For `f0 = 2y^2 - xy`:
`df0/dx = -y`, `df0/dy = 4y - x`.
`row0 = max(|y|) + max(|4y - x|)`.
This is correct because `sup_{x \in Box} ||grad f0||_1 <= sup |y| + sup |4y - x|`.
Is this standard? Yes, bounding the gradient component-wise by taking the maximum absolute value of its interval extension gives a valid Lipschitz constant.
Since `f0, f1, f2, f3` are all polynomials, their exact interval extensions bound their ranges, and the interval extensions of their partial derivatives bound the maximum of their absolute values.

Wait, look at `absolute_upper(sub(scale2(scale2(y)), x))`.
`sub` computes the interval for `4y - x`.
`absolute_upper` computes the maximum of the absolute values of the endpoints of this interval.
Is it true that `max(|I[0]|, |I[1]|)` bounds `|x|` for all `x \in I`?
Yes, if `x \in [a, b]`, then `|x| \le \max(|a|, |b|)`.
So this correctly bounds the absolute value of the derivative.
The only caveat is that `absolute_upper` might be slightly looser than necessary if the interval contains zero, but it's still a valid upper bound.

Wait, look at `row1`:
`df1/dw = -0.5 * y`.
The script uses `ceil_fraction(Fraction(absolute_upper(y), 2))`.
This is a valid upper bound for `|-0.5 * y|`.
Because `|-0.5 * y| = 0.5 * |y| <= 0.5 * absolute_upper(y) <= ceil(0.5 * absolute_upper(y))`.
This is also correct.

Let's check `contraction`:
`contraction = ceil_fraction(Fraction(max(lipschitz_rows(box, zs)), STEP_DENOMINATOR))`
This divides the max Lipschitz constant `L` by `STEP_DENOMINATOR` (which is `1/dt`).
Wait, if `L` is the Lipschitz constant of the field `f`, the Lipschitz constant of the Picard operator `P(x) = x0 + dt * f(x)` is `dt * L`.
`dt = 1/256`. So `1/dt = 256`.
The code uses `STEP_DENOMINATOR = 256`.
So it divides `L` by 256.
Since `lipschitz_rows` returns `L` in fixed point (meaning it's already multiplied by `ONE`),
`Fraction(L_fixed, 256)` gives `(L_real * ONE) / 256`.
Then `ceil_fraction` takes the ceiling.
If this value is `>= ONE` (i.e. `>= 1.0`), then `L_real / 256 >= 1.0`, so the step is not a contraction.
This is mathematically completely sound.

What about `field`:
```python
    return (
        sub(scale2(yy), xy),
        sub(xy, div2(mul(y, wzs))),
        sub(sub(xy, w), zs),
        sub(sub(sub(x, y), div2(wzs)), (ONE, ONE)),
    )
```
This is the exact interval arithmetic evaluation of the vector field.

What about `CASES=4` and `ACCEPTED_CASES=1`?
We have 4 cases generated. Only the first one `leaf331_center_valid` uses `box`, which is inflated by `BOX_INFLATION_RAW` and therefore should pass all checks and return status `1`.
The other 3 cases use `uninflated`, `reversed_initial`, `outside_initial`, which are designed to return statuses `-4`, `-1`, `-2` respectively.
So `ACCEPTED_CASES=1` is correct.

Is there any issue with:
`if any(not (container[0] < value[0] and value[1] < container[1]) for container, value in zip(box, candidate, strict=True)):`
This strictly requires `candidate` to be in the interior of `box`.
Since `box` is inflated by `1 << 64` on all sides, `candidate` will be strictly inside `box`. So status `1` is correctly returned for the valid case.

Wait, let me double check the interval operations.
`div2(value: Interval) -> Interval: return floor_fraction(Fraction(value[0], 2)), ceil_fraction(Fraction(value[1], 2))`
Is it guaranteed that `floor_fraction(value[0], 2) <= value[0]/2`?
Yes.
Is it guaranteed that `ceil_fraction(value[1], 2) >= value[1]/2`?
Yes.
Is `Fraction(a, 2)` mathematically correct?
If `a` is the fixed point representation, then real value is `a / ONE`.
Real value divided by 2 is `a / (2 * ONE)`.
Wait!
If I do `Fraction(a, 2)`, I am getting `a / 2`.
But the fixed point representation of `a / 2` is exactly `a / 2` because it's a dyadic rational.
Let's trace:
Real value is `R = a / ONE`.
We want to represent `R / 2` in fixed point.
`R / 2 = a / (2 * ONE)`.
In fixed point (where `1.0` is `ONE`), the representation is `(a / (2 * ONE)) * ONE = a / 2`.
So computing `a / 2` in the fixed point domain is perfectly correct!
Since `a` is an integer, `a/2` can be represented exactly if we allow fractional bits, but since we are in integer fixed point, `a/2` might need to be rounded.
Taking `floor(a / 2)` for the lower bound and `ceil(a / 2)` for the upper bound correctly encloses `a / 2`.
This is perfectly correct.

Let's check `scale2(value)`:
Returns `value[0] * 2, value[1] * 2`.
`2 * R = 2 * (a / ONE) = 2a / ONE`.
Fixed point representation is `2a`.
Since `a` is an integer, `2a` is an integer.
This is exact and correct.

Let's check `mul(left, right)`:
`corners = [Fraction(a * b, ONE) for a in left for b in right]`
Left is `L = a / ONE`, Right is `R = b / ONE`.
Product is `L * R = a * b / ONE^2`.
Fixed point representation of product is `(a * b / ONE^2) * ONE = a * b / ONE`.
The code computes `Fraction(a * b, ONE)` and takes `min` and `max`.
Then `floor_fraction` and `ceil_fraction` convert to integers.
This is perfectly correct.

Let's check `add`:
`L + R = a / ONE + b / ONE = (a + b) / ONE`.
Fixed point representation is `a + b`.
Code does `left[0] + right[0]`. Correct.

Everything mathematically checks out. The author clearly knows how to do exact rational interval arithmetic, exact fixed-point conversions, and component-wise Lipschitz bounding.

Are there any subtle mathematical errors?
Could `lipschitz_rows` be incorrect?
`f1 = x*y - 0.5 * y * wzs`.
`df1/dy = x - 0.5 * wzs`.
Wait!
`df1/dy = x - 0.5 * wzs`.
In `row1`, the code uses `absolute_upper(sub(x, div2(wzs)))`.
`x` is an interval. `div2(wzs)` is an interval.
`sub(x, div2(wzs))` gives the interval for `x - 0.5 * wzs`.
`absolute_upper` gives the max absolute value of this interval.
This bounds `|x - 0.5 * wzs|`. Correct.

What about `df1/dw`?
`f1 = x*y - 0.5 * y * (w + zs)`.
`df1/dw = -0.5 * y`.
The code uses `ceil_fraction(Fraction(absolute_upper(y), 2))`.
This bounds `|-0.5 * y|`. Correct.

What about `df1/dx`?
`df1/dx = y`.
Code uses `absolute_upper(y)`. Correct.

What about `row0`?
`f0 = 2y^2 - xy`
`df0/dx = -y`. Code uses `absolute_upper(y)`. Correct.
`df0/dy = 4y - x`. Code uses `absolute_upper(sub(scale2(scale2(y)), x))`. Correct.

What about `row2`?
`f2 = xy - w - zs`.
`df2/dx = y`. Code: `absolute_upper(y)`. Correct.
`df2/dy = x`. Code: `absolute_upper(x)`. Correct.
`df2/dw = -1`. Code: `ONE`. Correct.

What about `row3`?
`f3 = x - y - 0.5*(w + zs) - 1.0`.
`df3/dx = 1`.
`df3/dy = -1`.
`df3/dw = -0.5`.
L1 norm: `1 + 1 + 0.5 = 2.5`.
Code: `2 * ONE + ONE // 2`. Correct.

The derivative w.r.t `zs` is ignored, which is correct because `zs` is a fixed input parameter to the step, not a state variable.

What about `initial_state`:
```python
    u = -decimal("0.004") + Fraction(447, 2) * decimal("0.008") / 256
```
`u` is just a parameter computation.

What about `STATUS` checking?
```python
    if any(lower > upper for lower, upper in intervals):
        return -1
```
Correct.

```python
    if any(not -DOMAIN_LIMIT < endpoint < DOMAIN_LIMIT for interval in intervals for endpoint in interval):
        return -2
```
Correct.

```python
    candidate = image(initial, box, zs)
    if any(not (container[0] < value[0] and value[1] < container[1]) for container, value in zip(box, candidate, strict=True)):
        return -4
```
Correct.

```python
    contraction = ceil_fraction(Fraction(max(lipschitz_rows(box, zs)), STEP_DENOMINATOR))
    if contraction >= ONE:
        return -5
```
Correct.

Wait, is `lipschitz_rows` correctly using the bounds of `box`?
Yes, `x, y, w, _ell = box`.
Then it computes `absolute_upper` on these intervals.
Since `box` contains the state space for the Picard step, evaluating the derivatives over `box` gives valid bounds for the derivatives over the entire state space.
This guarantees the global Lipschitz constant over the box is correctly bounded.

Is there any problem with the Picard iteration in `fixed_box`?
```python
        candidate = image(initial, box, zs)
        widened = tuple((min(old[0], new[0]), max(old[1], new[1])) for old, new in zip(box, candidate, strict=True))
```
This computes the union of the current `box` and `image(initial, box, zs)`.
If `candidate \subseteq box`, then `widened == box`.
The function returns the inflated box.
This is mathematically the standard interval Picard operator fixpoint iteration.

Could `image` have an error?
```python
def image(initial: tuple[Interval, ...], box: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    time = (0, ONE // STEP_DENOMINATOR)
    return tuple(add(component, mul(time, derivative)) for component, derivative in zip(initial, field(box, zs), strict=True))
```
`time` represents the interval `[0, dt]`.
`mul(time, derivative)` correctly computes the interval product.
Since `derivative` can be negative, `mul` evaluates all 4 corners.
Then `add(component, ...)` adds the initial condition.
This correctly implements `x_0 + [0, dt] * f(box)`.
Since `f(box)` bounds the range of `f(x)` for `x \in box`, `[0, dt] * f(box)` bounds the integral over `[0, dt]`.
Therefore `image` exactly bounds the set of states reachable after 1 time step.

Is there any issue with `BOX_INFLATION_RAW` being a constant `1 << 64`?
`ONE = 1 << 96`.
So `BOX_INFLATION_RAW` is `2^64 / 2^96 = 2^{-32}` in real units.
This is a tiny epsilon added to the bounds to ensure strict inequality for the `return -4` check without practically changing the box size.

Is there any issue with `contraction >= ONE`?
If the real Lipschitz constant is `L`, the fixed point bound is `L_fixed`.
`L_fixed / 256` is computed. If it's `>= ONE`, then `L / 256 >= 1`, meaning the step is not a contraction.
So returning `-5` is correct.

Is there any issue with `div2`?
```python
def div2(value: Interval) -> Interval:
    return floor_fraction(Fraction(value[0], 2)), ceil_fraction(Fraction(value[1], 2))
```
Wait! If `value` is an interval, say `[-1, 1]`.
`div2` will do `floor(-1/2) = -1`, `ceil(1/2) = 1`.
So `div2([-1, 1]) = [-1, 1]`.
Is `div2(I)` an interval extension of `x/2`?
For `I = [a, b]`, the image of `I` under `x/2` is `[a/2, b/2]`.
So taking `floor(a/2)` for the lower bound and `ceil(b/2)` for the upper bound is an interval enclosure of `[a/2, b/2]`.
Because `floor(a/2) <= a/2` and `ceil(b/2) >= b/2`.
This is correct for a division by 2. It is not the tightest possible fixed point enclosure (which would be round-to-nearest), but it is a valid upper and lower bound.
Since `floor` and `ceil` are used everywhere appropriately, this is completely sound.

Wait, let's look at `absolute_upper(sub(x, div2(wzs)))`
Could there be a dependency problem?
`x` and `wzs` are distinct state variables. No dependency issue.
`wzs = add(w, zs)`.
`div2(wzs)` divides the sum by 2.
`sub(x, div2(wzs))` computes `x - wzs/2`.
Since `x` and `wzs` are intervals, evaluating `x - wzs/2` directly using interval arithmetic is the exact range of `x - wzs/2` (since it's a linear function).
So the interval is exact (up to fixed point rounding).
Thus `absolute_upper` gives the exact maximum absolute value of this linear function over the box.
This is fully correct and optimal!

What about `sub(scale2(scale2(y)), x)`?
`scale2(scale2(y
