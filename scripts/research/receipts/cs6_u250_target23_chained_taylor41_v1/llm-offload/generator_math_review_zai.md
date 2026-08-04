The user wants me to review the math in the provided Python code.
The code implements a validated ODE solver for a specific 4D system using Taylor models with Picard iteration, logarithmic norm estimates, and bisection for event location.
The math involved:
1. Fractional fixed-point arithmetic.
2. Interval arithmetic (`enclose`, `add`, `sub`, `mul`, `half`, `total`).
3. Vector field ODE definition.
4. Picard iteration for box invariance (`picard_image`, `picard_box`).
5. Lipschitz constant / Logarithmic norm calculation for contraction.
6. Taylor series coefficients `coefficients`.
7. Remainder bound calculation `exp_upper_raw` and `advance`.
8. Event location via bisection `locate_event`.

Let's check each math operation.

1. `floor_q`, `ceil_q`, `enclose`
`floor_q(value)` is `value.numerator // value.denominator` which correctly floors.
`ceil_q(value)` is `-((-value.numerator) // value.denominator)` which correctly ceils.

2. `add`, `sub`, `total`
`add` adds intervals correctly.
`sub` subtracts correctly.
`mul` computes all 4 corners and uses min/max. Floor for lower, ceil for upper. Wait, `min(corners) // ONE` is floor(min), `ceil(max)` is `-((-max(corners)) // ONE)`. Correct.
`half(value)` uses floor and ceil. Correct.
`total` adds them up.

3. `vector_field`
```python
    x, y, w, _ell = state
    xy = mul(x, y)
    yy = mul(y, y)
    wzs = add(w, zs)
    return (
        sub((2 * yy[0], 2 * yy[1]), xy),
        sub(xy, half(mul(y, wzs))),
        sub(sub(xy, w), zs),
        sub(sub(sub(x, y), half(wzs)), (ONE, ONE)),
    )
```
Let's see the system:
x' = 2y^2 - xy
y' = xy - y(w+zs)/2 = xy - 1/2 * y * (w+zs)
w' = xy - w - zs
\ell' = x - y - 1/2(w+zs) - 1
This matches the formulas. Wait, 1/2(y(w+zs)) is `half(mul(y, wzs))`, correct. Wait, `(ONE, ONE)` is subtracted in `\ell'`, which is `1`. Correct.

4. `picard_image`
```python
    time = (0, step_raw)
    return tuple(add(component, mul(time, derivative)) for component, derivative in zip(initial, vector_field(box, zs), strict=True))
```
Time is `(0, step_raw)`. But time should be a single value, here it's treated as an interval. Since step_raw > 0, the interval is strictly [0, step_raw].
The picard operator is F(box)(t) = init + step_raw * f(box).
Wait! The variable time is `h`. The integral of f from 0 to h is bounded by h * f(box).
But if `time` is just a single scalar `h`, why represent it as an interval `(0, step_raw)` instead of `(step_raw, step_raw)`?
Because `time` interval `[0, h]` covers all intermediate times if we were constructing the tube! Since we want an invariant set for the whole interval `[0, h]`, evaluating `mul((0, step_raw), derivative)` gives the maximum span of `h * derivative` for `h \in [0, step_raw]`.
Wait, `mul((0, step_raw), derivative)` computes max(0, step_raw) * min/max(derivative), which is exactly the interval of `h * v` for `h \in [0, step_raw]` and `v \in derivative`.
Yes, that correctly bounds the integral from 0 to step_raw, assuming `derivative` doesn't change sign or whatever, but the interval multiplication `h \in [0, step_raw]` and `v \in [v_low, v_up]` correctly bounds `h * v`. The extreme values of `h * v` on `[0, h] \times [v_low, v_up]` are indeed among `0 \cdot v_low`, `0 \cdot v_up`, `h \cdot v_low`, `h \cdot v_up`. The `mul` function checks these 4 corners. This is correct.

5. `ordinary_lipschitz`
```python
    x, y, w, _ell = box
    return max(
        magnitude(y) + magnitude(sub((4 * y[0], 4 * y[1]), x)),
        magnitude(y) + magnitude(sub(x, half(add(w, zs)))) + ceil_q(Fraction(magnitude(y), 2)),
        magnitude(y) + magnitude(x) + ONE,
        5 * ONE // 2,
    )
```
Wait, `4*y` interval is `(4 * y[0], 4 * y[1])`.
Is `sub((4 * y[0], 4 * y[1]), x)` equivalent to `4y - x`? Yes.
Let's re-evaluate the Jacobian of the vector field to see if this is a Lipschitz constant.
Vector field:
f1(x,y,w,z) = 2y^2 - xy
f2(x,y,w,z) = xy - 0.5 * y * w - 0.5 * y * z
f3(x,y,w,z) = xy - w - z
f4(x,y,w,z) = x - y - 0.5 * w - 0.5 * z - 1

Jacobian J:
df1/dx = -y, df1/dy = 4y - x, df1/dw = 0, df1/dz = 0
df2/dx = y, df2/dy = x - 0.5 w - 0.5 z, df2/dw = -0.5 y, df2/dz = -0.5 y
df3/dx = y, df3/dy = x, df3/dw = -1, df3/dz = -1
df4/dx = 1, df4/dy = -1, df4/dw = -0.5, df4/dz = -0.5

L1 norm of row 1: |-y| + |4y - x| + 0 <= mag(y) + mag(4y - x). (Assuming mag(y) is positive, which `magnitude` gives).
Row 2: |y| + |x - 0.5(w+z)| + |-0.5 y| <= mag(y) + mag(x - 0.5(w+z)) + 0.5 * mag(y).
Here the code has `ceil_q(Fraction(magnitude(y), 2))`, which is ceil(0.5 * mag(y)).
Wait, |df2/dz| is 0.5 y, but `zs` is just a parameter. The Jacobian with respect to the state variables (x, y, w, \ell) is:
Variables are x1=x, x2=y, x3=w, x4=\ell.
The state derivatives are:
dx1/dt = 2x2^2 - x1*x2
dx2/dt = x1*x2 - 0.5 x2*(x3+zs)
dx3/dt = x1*x2 - x3 - zs
dx4/dt = x1 - x2 - 0.5(x3+zs) - 1

Jacobian J (df_i / dx_j):
J11 = -y
J12 = 4y - x
J13 = 0
J14 = 0

J21 = y
J22 = x - 0.5(w+zs)
J23 = -0.5y
J24 = 0

J31 = y
J32 = x
J33 = -1
J34 = 0

J41 = 1
J42 = -1
J43 = -0.5
J44 = 0

Row 1 L1 norm: |-y| + |4y-x| + 0 + 0. Code has `magnitude(y) + magnitude(sub((4 * y[0], 4 * y[1]), x))`. Correct.
Row 2 L1 norm: |y| + |x-0.5(w+zs)| + |-0.5y| + 0. Code has `magnitude(y) + magnitude(sub(x, half(add(w, zs)))) + ceil_q(Fraction(magnitude(y), 2))`. Correct.
Row 3 L1 norm: |y| + |x| + |-1| + 0 = mag(y) + mag(x) + 1. Code has `magnitude(y) + magnitude(x) + ONE`. Correct.
Row 4 L1 norm: |1| + |-1| + |-0.5| + |0| = 2.5. Code has `5 * ONE // 2`. Correct.
Wait, does `magnitude(y)` correctly bound |-y|? Yes, `magnitude((y_low, y_up))` returns `max(|y_low|, |y_up|)`, which bounds |-y| since |-y| = |y|.

Wait, the `ordinary_lipschitz` bounds the L1 norm of the Jacobian. The maximum eigenvalue of a matrix is bounded by the maximum L1 norm of its rows. So the Lipschitz constant is correctly bounded.

6. `logarithmic_norm`
The logarithmic norm (Lozinskii measure) with respect to the L1 norm is `mu_1(f) = max_i ( f_{ii} + sum_{j \neq i} |f_{ij}| )`.
Let's check the code for `logarithmic_norm`:
```python
    x, y, w, _ell = box
    return max(
        -y[0] + magnitude(sub((4 * y[0], 4 * y[1]), x)),
        sub(x, half(add(w, zs)))[1] + magnitude(y) + ceil_q(Fraction(magnitude(y), 2)),
        -ONE + magnitude(y) + magnitude(x),
        5 * ONE // 2,
    )
```
For row 1: `J11 + sum_{j!=1} |J1j|` = `-y + |4y - x|`.
Since `J11 = -y`, we want the maximum possible value of `-y`. Since `y` is an interval `(y_low, y_up)`, the maximum of `-y` is `-y_low`, which is `-y[0]`.
The code uses `-y[0] + magnitude(sub((4 * y[0], 4 * y[1]), x))`. Correct.

For row 2: `J22 + |J21| + |J23|` = `x - 0.5(w+zs) + |y| + |-0.5y|`.
We want the maximum possible value of `x - 0.5(w+zs)`. The upper bound of `x - 0.5(w+zs)` is `sub(x, half(add(w, zs)))[1]`.
The code has `sub(x, half(add(w, zs)))[1] + magnitude(y) + ceil_q(Fraction(magnitude(y), 2))`. Correct.

For row 3: `J33 + |J31| + |J32|` = `-1 + |y| + |x|`.
Code has `-ONE + magnitude(y) + magnitude(x)`. Correct.

For row 4: `J44 + |J41| + |J42| + |J43|` = `0 + 1 + 1 + 0.5 = 2.5`.
Code has `5 * ONE // 2`. Correct.

7. `picard_box`
```python
        image = picard_image(initial, box, zs, step_raw)
        widened = hull(box, image)
        if widened == box:
            candidate = tuple((lower - PICARD_INFLATION_RAW, upper + PICARD_INFLATION_RAW) for lower, upper in box)
            candidate_image = picard_image(initial, candidate, zs, step_raw)
            if not all(outer[0] < inner[0] and inner[1] < outer[1] for outer, inner in zip(candidate, candidate_image, strict=True)):
                raise ValueError("inflated Picard box is not a strict self-map")
            contraction = -((-ordinary_lipschitz(candidate, zs) * step_raw) // ONE)
            if contraction >= ONE:
                raise ValueError("Picard box is not a strict contraction")
            return candidate, iteration, contraction
```
Wait, `outer[0] < inner[0] and inner[1] < outer[1]` means `inner` is strictly contained in `outer`. `outer` is `candidate`, `inner` is `candidate_image`. This checks `candidate_image \subset candidate`. Correct.

`contraction = -((-ordinary_lipschitz(candidate, zs) * step_raw) // ONE)` is just ceil( Lipschitz * step_raw ).
Then it checks `contraction < ONE`. Correct.

8. `coefficients`
This computes Taylor coefficients `c_{n}`.
The recurrence for Taylor coefficients of `x(t) = \sum c_n t^n` given `x' = f(x)` is `(n+1) c_{n+1} = \text{coeff of } t^n \text{ in } f(x(t))`.
In the code:
```python
def coefficients(state: tuple[Interval, ...], zs: Interval, step_raw: int, order: int) -> list[list[Interval]]:
    coeff = [[state[axis] if degree == 0 else (0, 0) for degree in range(order + 1)] for axis in range(4)]
    for degree in range(order):
        xy = total([mul(coeff[0][j], coeff[1][degree - j]) for j in range(degree + 1)])
        yy = total([mul(coeff[1][j], coeff[1][degree - j]) for j in range(degree + 1)])
        yw = total([mul(coeff[1][j], coeff[2][degree - j]) for j in range(degree + 1)])
        coeff[0][degree + 1] = scaled_divide(sub((2 * yy[0], 2 * yy[1]), xy), step_raw, degree + 1)
        coeff[1][degree + 1] = scaled_divide(sub(xy, half(add(yw, mul(zs, coeff[1][degree])))), step_raw, degree + 1)
        coeff[2][degree + 1] = scaled_divide(sub(xy, add(coeff[2][degree], zs if degree == 0 else (0, 0))), step_raw, degree + 1)
        constant = add(half(zs), (ONE, ONE)) if degree == 0 else (0, 0)
        coeff[3][degree + 1] = scaled_divide(sub(sub(sub(coeff[0][degree], coeff[1][degree]), half(coeff[2][degree])), constant), step_raw, degree + 1)
    return coeff
```
Wait! The time variable `t` goes from `0` to `h`.
The function computes the Taylor coefficients in terms of `h`?
No, typically one changes variable to `s = t/h`, so `t = hs`, `s \in [0,1]`.
Then `x(s) = \sum c_n s^n`.
`dx/dt = (1/h) dx/ds`.
So `dx/ds = h * f(x(s))`.
Thus `(n+1) c_{n+1} = h * [t^n] f(x(s))`.
Let's see: `scaled_divide(expr, step_raw, degree+1)`
`scaled_divide` does `directed_divide(mul(expr, (step_raw, step_raw)), degree + 1)`.
This is equivalent to `expr * h / (degree + 1)`.
Wait, `expr` is an interval. `mul(expr, (step_raw, step_raw))` computes `expr * h`.
Then `directed_divide(val, degree + 1)` computes `val / (degree + 1)`.
This EXACTLY matches `(n+1) c_{n+1} = h * [t^n] f(x(s))`.
Let's verify the `[t^n] f(x(s))` calculation:
`xy = \sum_{j=0}^n c_{1,j} c_{2, n-j}` (using 0-indexed: coeff[0] is x, coeff[1] is y, coeff[2] is w, coeff[3] is \ell).
`yy = \sum_{j=0}^n c_{1,j} c_{1, n-j}`.
`yw = \sum_{j=0}^n c_{1,j} c_{2, n-j}`.
Then for x (axis 0):
`2y^2 - xy` -> `2 * yy - xy`. Code: `sub((2 * yy[0], 2 * yy[1]), xy)`. Correct.
For y (axis 1):
`xy - 1/2 * y * (w+zs)`.
`y * w` is `yw`.
`y * zs` is `zs * c_{1, n}`. Code has `mul(zs, coeff[1][degree])`.
Then `xy - 1/2 (yw + zs * y_n)`. Code: `sub(xy, half(add(yw, mul(zs, coeff[1][degree]))))`. Correct.
For w (axis 2):
`xy - w - zs`.
Coefficient of `w` is `c_{2, n}`.
Coefficient of `zs` is `zs` if `n=0`, else `0`.
So `xy - c_{2,n} - (n==0 ? zs : 0)`. Code: `sub(xy, add(coeff[2][degree], zs if degree == 0 else (0, 0)))`. Correct.
For \ell (axis 3):
`x - y - 1/2 w - 1/2 zs - 1`.
`c_{0,n} - c_{1,n} - 1/2 c_{2,n} - (n==0 ? 1/2 zs + 1 : 0)`.
Code: `constant = add(half(zs), (ONE, ONE)) if degree == 0 else (0, 0)`.
Then `sub(sub(sub(coeff[0][degree], coeff[1][degree]), half(coeff[2][degree])), constant)`. Correct.

What about `scaled_divide`?
```python
def scaled_divide(value: Interval, step_raw: int, divisor: int) -> Interval:
    return directed_divide(mul(value, (step_raw, step_raw)), divisor)
```
Wait! `mul(value, (step_raw, step_raw))` multiplies `value` by `h`.
But `value` is already an interval! Wait, `value` is computed as interval.
So `mul(value, (step_raw, step_raw))` correctly bounds `h * value`.
Then `directed_divide` divides by `divisor`.
This perfectly matches `c_{n+1} = h * c'_n / (n+1)`. Correct.

9. `exp_upper_raw`
We need to bound `exp( \mu h )`.
Since `s \in [0, 1]`, `t \in [0, h]`, so the exponent is `\mu h s` for some `s \in [0, 1]`. The maximum value is `\mu h`.
The code computes `exp_upper_raw(mu_h)` where `mu_h = max(mu, 0) * h`.
`mu_h` is an integer, so it's `\mu h \cdot ONE`.
`exp` series is `\sum \frac{x^k}{k!}`.
Code:
```python
def exp_upper_raw(argument: int) -> int:
    if argument <= 0:
        return ONE
    term = ONE
    result = ONE
    for degree in range(1, 33):
        term = -((-(term * argument)) // (ONE * degree))
        result += term
    next_term = -((-(term * argument)) // (ONE * 33))
    ratio = -((-argument) // 34)
    tail = -((-(next_term * ONE)) // (ONE - ratio))
    return result + tail
```
Here `argument` is `x` (in fixed point). `term` is `x^k / k!`.
`term = ceil( term * argument / (degree * ONE) )`.
Wait, `term * argument` is `x^{k+1} \cdot ONE`.
Dividing by `degree * ONE` gives `x^{k+1} / k!`. Wait!
Let's trace fixed point:
Initial `term = ONE` (`x^0 / 0!`).
Loop degree 1:
`term = ceil( ONE * argument / (1 * ONE) ) = argument`. This is `x^1 / 1!`. Correct.
Loop degree 2:
`term = ceil( argument * argument / (2 * ONE) )`. This is `x^2 / 2!`. Correct.
Loop up to 32:
`term` is `ceil( x^{32} / 32! )`.
Wait, the loop goes `for degree in range(1, 33)`, so `degree` takes values 1 to 32.
After the loop, `term` is `x^{32} / 32!`.
Then:
`next_term = ceil( term * argument / (33 * ONE) )`.
Wait, `term` is `x^{32} / 32!`.
`term * argument` is `x^{33} / 32! \cdot ONE`.
Divided by `33 * ONE` gives `x^{33} / 33!`. Correct.
Then we need to bound the tail `\sum_{k=34}^\infty x^k / k!`.
For `k \ge 34`, the ratio of consecutive terms is `x / k \le x / 34`.
Since `x \le ONE >> 8` (at most 1/256), `x / 34` is very small.
The code calculates `ratio = ceil(argument / 34)`.
Then `tail = ceil( next_term / (1 - ratio/ONE) ) = ceil( next_term * ONE / (ONE - ratio) )`.
Since `next_term` is `x^{33} / 33!`, and the ratio is `x/34`, the tail is `next_term * \frac{x/34}{1 - x/34} = next_term \frac{x}{34 - x}`.
Wait! The series is `T_{34} + T_{35} + ...`.
`T_{34} = T_{33} * (x/34)`.
The code does:
`next_term = T_{33}`.
Then `tail = next_term / (1 - x/34)`.
Let's check the sum of geometric series:
`T_{34} + T_{35} + ... = T_{33} (x/34) + T_{33} (x/34)^2 + ... = T_{33} \frac{x/34}{1 - x/34} = T_{33} \frac{x}{34 - x}`.
But the code calculates `tail = next_term / (1 - ratio/ONE) = next_term / (1 - x/34)`.
This is `next_term + next_term (x/34) + ...`
Wait! `next_term` is `T_{33}`.
The tail should be `T_{34} + T_{35} + \dots`.
If `tail = next_term / (1 - ratio/ONE)`, then `tail = T_{33} / (1 - x/34) = T_{33} + T_{34} + T_{35} + \dots`.
But `T_{33}` is ALREADY in `result`!
Wait, the loop computes `term` up to `degree = 32`. So `result` contains `T_0` to `T_32`.
So the remaining terms are `T_33 + T_34 + \dots`.
If `tail = next_term / (1 - x/34)`, this exactly bounds `T_{33} + T_{34} + \dots`.
Let's check this carefully.
The loop is:
`term = ONE`, `result = ONE`.
degree 1: `term = T_1`, `result = T_0 + T_1`.
degree 2: `term = T_2`, `result = T_0 + T_1 + T_2`.
...
degree 32: `term = T_32`, `result = T_0 + ... + T_32`.
Then `next_term = ceil(T_32 * x / (33 ONE)) = T_33`.
Then `tail = ceil( next_term / (1 - x/34) )`.
Since `next_term \ge T_33`, `tail \ge T_33 / (1 - x/34)`.
And `T_33 / (1 - x/34) \ge T_33 + T_34 + \dots`
So `tail` overestimates `T_33 + T_34 + \dots`.
Then `result + tail` bounds `T_0 + \dots + T_32 + T_33 + \dots = \exp(x)`.
This is absolutely correct! Beautiful!

10. `advance`
```python
    mu = logarithmic_norm(box, zs)
    mu_h = -((-(max(mu, 0) * step_raw)) // ONE)
    if mu_h >= ONE:
        raise ValueError("logarithmic-norm amplification denominator is nonpositive")
    amplification = exp_upper_raw(mu_h)
    propagated = -((-(radius * amplification)) // ONE)
    next_radius = propagated + local_radius
```
Wait, if `mu_h >= ONE`, it raises an error.
`mu_h` is `max(mu, 0) * step_raw / ONE`.
If `mu_h \ge ONE`, the error message says "logarithmic-norm amplification denominator is nonpositive".
But wait! In `exp_upper_raw`, the tail has `tail = -((-(next_term * ONE)) // (ONE - ratio))`.
If `ratio \ge ONE`, `ONE - ratio` could be `0` or negative, leading to division by zero or negative tail.
But `ratio = ceil(argument / 34)`. Since `argument \le ONE >> 8` (which is `ONE / 256`), `ratio` is extremely small, nowhere near `ONE`.
Wait, `mu_h \ge ONE` means `\mu h \ge 1`.
If `\mu h \ge 1`, then `amplification = \exp(\mu h) \ge \exp(1)`, which is fine. The tail ratio `argument / 34` is still \le 1/34.
Why does it say "logarithmic-norm amplification denominator is nonpositive"?
Is there a place where `1 - \mu h` is in the denominator?
Ah! If `mu < 0`, then `mu_h` is `0`, and `exp_upper_raw(0)` returns `1`.
If `mu < 0`, the contraction factor is `e^{\mu h} \le 1`.
Wait, if you use a first-order Taylor method or Euler method, the error bound is often `\frac{\delta}{h \mu} (e^{\mu h} - 1)`.
Wait, if `mu < 0`, the logarithmic norm bounds the distance between two solutions:
`|x(t) - y(t)| \le e^{\mu t} |x(0) - y(0)|`.
So if `mu < 0`, `e^{\mu h} < 1`, the distance shrinks!
The maximum of `e^{\mu h}` for `\mu < 0` is `1` (at `t=0`).
So `amplification = 1` is perfectly correct! It bounds `e^{\mu h}` for `\mu \le 0`.
Wait, what if `mu > 0`? Then `e^{\mu h} > 1`.
The code does `amplification = exp_upper_raw(max(mu, 0) * step_raw)`. This bounds `e^{\mu h}`.
Is `propagated` the propagated error from the initial box?
Yes, `radius` is the initial box radius.
So `propagated` bounds the difference between the true solution starting from the initial box, and the true solution starting from the center.
Wait, what about `local_radius`?
`local_radius` is the local truncation error of the Taylor method.
Let's see how `enclosure` and `local_radius` are computed:
```python
    center_coeff = coefficients(center_intervals, zs, step_raw, ORDER - 1)
    wide_coeff = coefficients(center_box, zs, step_raw, ORDER)
    next_center = []
    local_radius = 0
    for axis in range(4):
        enclosure = add(total(center_coeff[axis]), wide_coeff[axis][ORDER])
```
`center_coeff` computes Taylor coefficients up to degree `ORDER - 1`, evaluated AT the exact center.
`wide_coeff` computes Taylor coefficients up to degree `ORDER`, evaluated for the wide box.
So `wide_coeff[axis][ORDER]` is the `ORDER`-th coefficient, computed with interval arithmetic over the entire box.
Wait, the Taylor expansion at the center is `c_0 + c_1 + ... + c_{ORDER-1} + R_{ORDER}`.
The remainder `R_{ORDER}` is bounded by `c_{ORDER}` evaluated over the entire domain!
Since `c_{ORDER}` is the Taylor coefficient `x^{(ORDER)}(\xi) / ORDER!`, evaluated over the box, `wide_coeff[axis][ORDER]` correctly bounds the remainder.
Wait, `wide_coeff[axis][ORDER]` is computed by `coefficients(center_box, ...)`.
`center_box` is the Picard box!
So the remainder is bounded by the Taylor coefficient of degree `ORDER` evaluated on the box.
Wait! Is it valid to just take `c_{ORDER}` as the remainder?
Yes, for `s \in [0, 1]`, the remainder is `c_{ORDER}(\xi) 1^{ORDER} = c_{ORDER}(\xi)`.
The true solution is `\sum_{k=0}^{ORDER-1} c_k + c_{ORDER}(\xi)`.
So the enclosure is `\sum_{k=0}^{ORDER-1} c_k^{(center)} + c_{ORDER}^{(box)}`.
Since `center` is inside the box, `c_k^{(center)}` is just an interval of zero width.
But `c_{ORDER}^{(box)}` is an interval.
So `enclosure` is an interval containing the true solution!
Wait, if `enclosure` is the interval containing the true solution starting FROM THE CENTER, then its width is the local truncation error.
`local_radius = max( midpoint - low, high - midpoint )` correctly bounds the error of the center.
Then `next_radius = propagated + local_radius`.
Wait! `propagated` bounds the error from the initial box.
If `x(t)` is the true solution from the center, and `y(t)` is the true solution from the box, then `|x(t) - y(t)| \le e^{\mu h} |x(0) - y(0)| = e^{\mu h} radius`.
So the total error at `h` is `error(x(t) from center) + error(y(t) from box wrt x(t))`.
The first is `local_radius`. The second is `propagated`.
So `next_radius = propagated + local_radius` is perfectly correct! (triangle inequality).

Let's double check `exp_upper_raw` logic for `mu_h >= ONE`.
Why does `mu_h >= ONE` raise `ValueError("logarithmic-norm amplification denominator is nonpositive")`?
Where is there a denominator `1 - \mu h`?
Ah! If you use the exact integral of the Jacobian, or if you use some other bound like `\frac{e^{\mu h} - 1}{\mu h}, this has `\mu h` in the denominator.
But the code computes `propagated = ceil(radius * amplification / ONE)`. There is NO denominator `1 - \mu h`!
So why does it say "logarithmic-norm amplification denominator is nonpositive"?
Wait! If `mu_h \ge ONE`, `\mu h \ge 1`.
Does `exp_upper_raw` fail if `argument \ge ONE`?
If `argument \ge ONE`, then `ratio = ceil(argument / 34)` could be `>= ceil(ONE/34)`.
If `argument \ge 34 ONE`, `ratio \ge ONE`.
Then `ONE - ratio` becomes zero or negative, so `tail = ceil(...) // (ONE - ratio)` would divide by zero or a negative number!
But `argument = mu_h`. It stops if `mu_h >= ONE`, which is long before `34 ONE`.
So the denominator is strictly positive!
Wait, the error message might just be a copy-paste from a previous version of the code where `1 - \mu h` was in the denominator.
Or maybe `mu_h` is just bounded by `1` to ensure `mu_h < 1` so `h` is small enough?
Since it raises an error and refuses the step, it's an over-constraint, but it's not mathematically WRONG. It just tightly restricts the step size.
Wait, if `mu_h >= ONE`, it raises an error. This is a check. If the condition is not mathematically necessary for the formulas used, is it an OVERREACH?
Yes, mathematically `exp_upper_raw` would work perfectly fine for `mu_h = 2 ONE` or `3 ONE`. The only problem would be if `ratio \ge ONE`.
But `ratio = ceil(mu_h / 34)`. So as long as `mu_h < 34 ONE`, the denominator is positive.
So checking `mu_h >= ONE` is an overreach.
Let's confirm: `exp_upper_raw` computes `tail = ceil(...) / (ONE - ratio)`.
If `mu_h = 2 ONE`, `ratio = ceil(2 ONE / 34)`, which is `ONE / 17`. This is strictly less than `ONE`. The denominator is `16/17 ONE > 0`. So `tail` is perfectly fine!
The Taylor coefficients `term` will also be computed correctly.
So `mu_h >= ONE` is a sufficient but unnecessary condition for the math to hold.
Let's call it `[OVERREACH] mu_h >= ONE check`.

Let's check `locate_event`.
```python
    high_result = advance(center, radius, zs, high)
    if sign(high_result["center"][2], int(high_result["radius"])) != 1:
        raise ValueError("event initial upper bracket is not strictly positive")
```
`high` is `step_raw`.
The state is `[x, y, w, \ell]`. The event checks `sign(center[2])`, which is `w`.
Wait! `center[2]` is `w`.
The event is `w(t) == 0`? No, the sign function is:
```python
def sign(center: int, radius: int) -> int:
    if center + radius < 0:
        return -1
    if center - radius > 0:
        return 1
    return 0
```
`sign` returns 1 if the entire interval is strictly positive.
It returns -1 if the entire interval is strictly negative.
It returns 0 if it contains 0.
So the event checks if `w` crosses 0.
Wait, if `sign == 1`, `w` is strictly positive.
Is the event `w = 0`?
Let's check what `normal` is.
```python
    normal = sub(mul(event_box[0], event_box[1]), zs)
```
`mul(event_box[0], event_box[1])` is `xy`.
So `normal = xy - zs`.
But wait! `w' = xy - w - zs`.
At the event, `w` crosses 0. Does `w' == 0`?
No, `normal` is `xy - zs`. This is `w' + w`. At `w = 0`, `w' = xy - zs`.
So `normal` checks if `w'` is strictly positive at the event.
If `normal[0] > 0`, then `w' > 0`, which means `w` is strictly increasing when it crosses 0. This ensures transversality of the event!
Let's check the code:
```python
    if normal[0] <= 0:
        raise ValueError("event normal velocity is not strictly positive")
```
This verifies that `w` is strictly increasing at the event (i.e. crosses 0 transversally from negative to positive). This is completely correct!

Wait! `locate_event` uses `low, high = 0, step_raw`.
`low_result = {"center": center, "radius": radius}`
`high_result = advance(center, radius, zs, high)`
It checks if `sign(high_result["
