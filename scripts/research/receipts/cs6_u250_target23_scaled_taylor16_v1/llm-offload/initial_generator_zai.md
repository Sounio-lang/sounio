The user wants me to review the math in the provided Python artifact.
The artifact computes interval arithmetic over fixed-point rational numbers for a Picard iteration / Taylor series (ODE solver or polynomial system solver).

I need to verify the math:
1. `floor_q` and `ceil_q` operations on Fractions.
2. Interval addition (`total`, `picard.add` - assumed to be standard interval addition $[a, b] + [c, d] = [a+c, b-d]$ wait, standard is $[a+c, b+d]$).
3. Division of intervals `divn` and `step_div`.
4. Taylor series / polynomial coefficient calculation in `coefficients`:
   - The equations for `coeff[0][degree + 1]`, `coeff[1][degree + 1]`, `coeff[2][degree + 1]`, `coeff[3][degree + 1]`.
   - These look like ODE equations for a vector field.
   Let's trace `coeff[0][degree + 1]`:
   `xy = sum(coeff[0][j] * coeff[1][degree - j])` -> Cauchy product for $x*y$
   `yy = sum(coeff[1][j] * coeff[1][degree - j])` -> Cauchy product for $y*y$
   `step_div(picard.sub(picard.scale2(yy), xy), degree + 1)`
   Wait, $\frac{2 y y - x y}{degree + 1}$? Since this is a Taylor series: $x_{n+1} = \frac{1}{n+1} \dots$
   Let's check the interval math for `step_div`. It takes an interval, divides by `divisor`, after multiplying by `H` (`ONE >> 8`).
   Let's ignore the scaling for a moment, it's just a fixed-point implementation.
   Are the operations mathematically sound for interval arithmetic?
   Wait, `divn` computes:
   `floor_q(value[0] / divisor), ceil_q(value[1] / divisor)`
   This is only correct if `divisor > 0`. Since `divisor` is `degree + 1`, it is $\ge 1$. So `divisor > 0` is true.
   `ceil_q(value: Fraction)` is `-((-value.numerator) // value.denominator)`. This correctly implements the ceiling division for integers in Python.

5. `non_strict` interval generation:
   `tuple((a + (1 << 64), b - (1 << 64)) for a, b in accepted[4:8])`
   This expands the intervals by adding $2^{64}$ to the lower bound and subtracting $2^{64}$ from the upper bound. This is correct for strict vs non-strict interval logic.

6. Let's look closely at `coefficients(state, zs, order)`:
   The polynomial for $X(t) = \sum X_n t^n$.
   $X_{n+1} = \frac{1}{n+1} (2 Y^2 - X Y)$
   $Y_{n+1} = \frac{1}{n+1} (X Y - \frac{1}{2}(Y W + Z_s Y))$
   $W_{n+1} = \frac{1}{n+1} (X Y - (W_n + Z_s))$ ? Wait, `picard.add(coeff[2][degree], zs if degree == 0 else (0, 0))`.
   $V_{n+1} = \frac{1}{n+1} (X - Y - \frac{1}{2} W - V_{constant})$
   These are just the mathematical ODEs encoded in fixed-point. The math of the Cauchy products `total([picard.mul(coeff[i][j], coeff[k][degree - j]) for j in range(degree + 1)])` correctly computes the coefficient of $t^{degree}$ for the product of two series, since the terms in the product up to $t^{degree}$ are fully known.

7. Evaluate function:
   `center_coeff = coefficients(initial, zs, ORDER - 1)`
   `box_coeff = coefficients(box, zs, ORDER)`
   Here `ORDER = 16`.
   `center_coeff` has degrees `0` to `15`. So `polynomial = [total(center_coeff[axis]) for axis in range(4)]`.
   Wait, `center_coeff[axis]` is a list of intervals.
   `total` adds them all up.
   This gives the evaluation of the Taylor series at $t=1$.
   Wait, but if `center_coeff` is computed from `initial` (which are the center points of the box, wait... `initial` is an interval!).
   Wait, if `initial` is an interval, then `center_coeff` is an interval Taylor polynomial.
   Then `box_coeff` is computed from `box`.
   `polynomial` gives the interval sum of `center_coeff`.
   `remainder` is `box_coeff[axis][ORDER]`. This is the term for $t^{16}$ of the `box` expansion.
   `next_state = picard.add(polynomial[axis], remainder[axis])`
   Wait, is the remainder just a single term? In interval Picard iterations, the remainder is usually bounded by a geometric series or something, but here it's just taking the $t^{16}$ term as the remainder? That seems mathematically wrong if it's supposed to be a rigorous remainder bound.
   Let's verify: `box_coeff[axis][ORDER]` is the coefficient of $t^{16}$. If we evaluate at $t=1$, the sum of the Taylor series is $\sum_{n=0}^{\infty} X_n$. The remainder is $\sum_{n=16}^{\infty} X_n$. Taking only $X_{16}$ is NOT the remainder! The remainder would be $X_{16} + X_{17} + \dots$.
   Unless the interval `box_coeff[axis][ORDER]` represents an absorbing bound? But it's computed recursively from `box_coeff[axis][15]` using the interval ODE. It's just the next term. It doesn't bound the infinite sum unless $t < 1$ or some shrinking lemma is applied. If evaluated at $t=1$, $X_{16}$ doesn't bound the rest.
   Wait! "scaled Taylor-16 target-23 transcript". Is there a $t$ scaling?
   `H = (ONE >> 8, ONE >> 8)`.
   `step_div` multiplies by `H` before dividing by `divisor`.
   So $H \approx 1/256$.
   Wait, the Taylor series is evaluated at $t = H \times 1$? No, if $X_{n+1} = \frac{H}{n+1} f(X_n)$, then this is equivalent to evaluating the Taylor series of the ODE $X' = f(X)$ at $t = H$.
   Then the polynomial is $X(H) \approx \sum_{n=0}^{15} X_n H^n / n! \dots$ wait.
   If $X_{n+1} = \frac{H}{n+1} f(\dots)$, this means $X_{n}$ is $X^{(n)}(0) H^n / n!$.
   Then the sum of $X_0 + X_1 + \dots + X_{15}$ evaluates the Taylor series at $t=1$ of the scaled ODE, which means $t=H$ of the original ODE.
   At $t=H$, the remainder of an interval Taylor series is NOT just the $n=16$ term. It's the sum of terms from $16$ to $\infty$. Taking just the 16th term as the remainder is WRONG. The standard way is to compute an upper bound for the radius of convergence and bound the tail, or use an interval Picard iteration where the interval `box` is invariant and we take the full infinite sum, but here `box_coeff[axis][16]` is just a single term.
   Is it WRONG? Let's think. If we use Picard iteration, the remainder is usually bounded by $M \sum_{i=k}^\infty \frac{(H L)^i}{i!}$ or similar. Here, they literally just use the 16th term. That's mathematically incorrect as a rigorous remainder unless proven otherwise.
   Wait, could `box_coeff[axis][ORDER]` be the result of a Picard operator that computes the FULL contraction? No, it's just computed recursively: `coeff[0][degree + 1] = \dots`.
   So it's just the 16th degree Taylor coefficient. This is an OVERREACH to call it a rigorous remainder, or WRONG if it claims rigorousness.
   Wait! The prompt says "EXACT_RATIONAL_SCALED_TAYLOR_RECONSTRUCTION=true".
   Maybe it's not a rigorous step, just a mathematical reconstruction of a specific finite series?
   The summary string has `FULL_ORBIT_CERTIFICATE=false`. This implies it's NOT claiming to be a rigorous orbit certificate. It is a "scaled Taylor-16 target-23 transcript". It might just be verifying the arithmetic of a fixed-point circuit.
   But let's review the claim "remainder". If the circuit expects the 16th term to be the remainder, and the code computes the 16th term, the code mathematically matches the spec. The mathematical soundness of calling the 16th term a "remainder" is a domain-specific question, but usually, in Taylor models, you need to bound the tail. If the script claims it's an exact reconstruction, then it's just a finite sum.
   However, I will flag this as `[OVERREACH]` or `[TIGHTENABLE]`.

Let's check `evaluate` indices:
`transcript = [endpoint for degree in range(ORDER) for axis in range(4) for endpoint in center_coeff[axis][degree]]`
This takes `ORDER` (16) degrees $\times 4$ axes $\times 2$ endpoints = $16 \times 4 \times 2 = 128$ words.
`transcript += [endpoint for value in remainder + polynomial + next_state for endpoint in value]`
`remainder` has 4 axes, `polynomial` has 4 axes, `next_state` has 4 axes.
Total values = 12 axes $\times 2$ endpoints = 24 words.
Total transcript size = $128 + 24 = 152$ words.
The return statement adds `(1,)`, so the return type is `tuple(transcript) + (1,)`.
Wait, `return (0,) * 152 + (-1,)` in the error cases. Length is 153.
Let's verify `result[-1]` logic:
`results` = `[evaluate(case) for _, case in cases]`.
`evaluate` returns 153 words.
Outputs length is `len(outputs)` = $3 \times 153 = 459$ words.

Let's check `evaluate` error handling:
`if any(lower > upper for lower, upper in intervals):` -> returns `-1`. This is for invalid intervals (lower bound > upper bound).
`if picard.status(initial, box, zs) != 1:` -> returns `-4`.

Let's check `next_words[2 * axis + 1] - next_words[2 * axis]`:
`next_words` is extracted from `result[144:152]`.
In `evaluate`:
`transcript` has 128 words.
Then `remainder` (8 words) goes into transcript -> indices 128 to 135.
Then `polynomial` (8 words) goes into transcript -> indices 136 to 143.
Then `next_state` (8 words) goes into transcript -> indices 144 to 151.
So `result[144:152]` correctly extracts `next_state`!

Let's check `MAX_REMAINDER_ABS_RAW`:
`remainder_words = result[128:136]` -> correctly extracts `remainder`!

Let's check `box_coeff = coefficients(box, zs, ORDER)`
`box_coeff[axis][ORDER]` for axis in 0..3 gives 4 intervals.
`remainder = [box_coeff[axis][ORDER] for axis in range(4)]`
This is correct.

Wait, `center_coeff = coefficients(initial, zs, ORDER - 1)`
`polynomial = [total(center_coeff[axis]) for axis in range(4)]`
If `ORDER - 1 = 15`, then `center_coeff` computes degrees 0 to 15.
`total` sums them up.
`polynomial` correctly contains the sum of degrees 0 to 15.

Wait, `coeff[0][degree + 1] = step_div(picard.sub(picard.scale2(yy), xy), degree + 1)`
`step_div` takes `value` and `divisor`.
`step_div(value, divisor)` returns `divn(picard.mul(value, H), divisor)`.
Inside `divn`: `floor_q(Fraction(value[0], divisor)), ceil_q(Fraction(value[1], divisor))`.
This is mathematically correct for positive `divisor`.

Wait, `non_strict` interval generation:
`tuple((a + (1 << 64), b - (1 << 64)) for a, b in accepted[4:8])`
This expands the box intervals. But wait! If we do `a + 2^64` and `b - 2^64`, we are increasing the lower bound and decreasing the upper bound!
Wait. To make an interval NON-STRICT, or to make it STRICT?
The case is named `picard_precondition_refusal`.
Wait, if it's a precondition refusal, it might mean it fails `picard.status != 1`.
Wait, if we INCREASE the lower bound and DECREASE the upper bound, we are SHRINKING the interval.
Is `accepted[4:8]` strict?
If `accepted` has tight bounds, shrinking them means `a > b` or something?
Wait. If the interval was $[a, b]$ and we want a "non-strict" interval? No, if we shrink the interval, maybe we break some precondition that requires the box to be strictly larger than the initial state?
Let's consider the interval operations:
`accepted` comes from `positive_input()`.
Maybe `accepted` is an interval that BARELY satisfies the precondition. By shrinking it (`a + 2^64`, `b - 2^64`), it FAILS the precondition (hence `refusal`).
Ah! To make it a valid refusal, we alter the box so it's too tight. Yes, this makes sense.

Let's re-verify the Cauchy products:
`xy = total([picard.mul(coeff[0][j], coeff[1][degree - j]) for j in range(degree + 1)])`
This correctly computes the `degree`-th coefficient of $X(t) Y(t)$.

Let's re-verify `picard.sub`, `picard.add`, `picard.mul`.
These are standard interval operations. For `mul`, standard interval multiplication is $[x_1, x_2] \times [y_1, y_2] = [\min(x_1 y_1, x_1 y_2, x_2 y_1, x_2 y_2), \max(\dots)]$. The code uses `picard.mul` which is imported. We can assume it's correct standard interval math.

Let's check the ODE definitions:
$x' = 2 y^2 - x y$
$y' = x y - \frac{1}{2}(y w + z_s y)$
$w' = x y - (w + z_s)$  (Wait, for degree == 0, it is $xy - (w + z_s)$, but for degree > 0, it is $xy - w$. This means the ODE is $w' = xy - w - z_s \delta(t)$? Or maybe $z_s$ is a constant input only at $t=0$? This looks like a specific control system or ODE with initial conditions, possibly a hybrid system where $z_s$ is an impulse or just part of the equation. Wait! If `zs` is an interval, maybe it's $w' = xy - w - z_s$? But the code says `zs if degree == 0 else (0, 0)`. This means $z_s$ is only subtracted for the constant term $t^0$. So it's an impulse at $t=0$! Wait, in Taylor series, $t^0$ is the value at $t=0$. If $z_s$ is only at $t=0$, that means $z_s$ is an initial condition offset. For example, if $w(0)$ was set to something, but the equation is $w' = xy - w$. Wait, if $w' = xy - w$, the Taylor coefficient of $t^0$ of $w'$ is $w_1 = w'(0) = xy(0) - w(0)$. If $z_s$ is subtracted, it means $w'(0) = xy(0) - w(0) - z_s$. But why would it be zero for degree > 0? This implies $w' = xy - w - z_s \delta(t)$? No, the Taylor coefficient of $w'$ for degree $n$ is $(n+1)w_{n+1}$. The code sets `coeff[2][degree + 1] = step_div(xy - (w_n + z_s), degree + 1)`. For $n=0$, $w_1 = xy_0 - w_0 - z_s$. For $n>0$, $w_{n+1} = xy_n - w_n$. This means the equation is $w' = xy - w - z_s$ for $n=0$? No, if the equation were $w' = xy - w - z_s$, then $z_s$ would be present for all $n$. If $z_s$ is only present for $n=0$, it's equivalent to $w'(0) = \dots - z_s$, which means $z_s$ is a constant added to the initial condition of $w'$ but not the rest. This is exactly equivalent to shifting $w$ by some function? Actually, this might be a specific step in a Picard iteration where the integral of $z_s$ is added. Let's assume it's correct per the domain (PK/Piccard).

Let's check $v' = x - y - \frac{1}{2}w - V_{constant}$.
For `coeff[3]` (let's call it $V$):
`constant = picard.add(picard.div2(zs), (ONE, ONE)) if degree == 0 else (0, 0)`
So for $n=0$, $V_1 = x_0 - y_0 - \frac{1}{2} w_0 - (\frac{1}{2} z_s + 1)$.
For $n>0$, $V_{n+1} = x_n - y_n - \frac{1}{2} w_n$.
This is identical to the $W$ equation: an affine shift only applied to the $t^0$ term.

Is there any math error in `coeff[3][degree + 1]`?
`step_div(picard.sub(picard.sub(picard.sub(coeff[0][degree], coeff[1][degree]), picard.div2(coeff[2][degree])), constant), degree + 1)`
Let's expand this carefully:
`x = coeff[0][degree]`
`y = coeff[1][degree]`
`w = coeff[2][degree]`
`step_div(x - y - w/2 - constant, degree + 1)`
This matches the formula perfectly.

Wait, is there any error in `floor_q` and `ceil_q` for negative fractions?
`floor_q(value: Fraction) -> int: return value.numerator // value.denominator`
In Python, `//` on integers is the floor division. Since `value.denominator > 0`, this is always exactly the mathematical floor.
`ceil_q(value: Fraction) -> int: return -((-value.numerator) // value.denominator)`
This correctly computes the ceiling. Because `ceil(x) = -floor(-x)`.
Since `value.denominator > 0`, `-value.numerator // value.denominator` is exactly `floor(-value)`. Thus `-` of it is exactly `ceil(value)`.
This is correct.

Let's check `divn(value, divisor)`:
`return floor_q(Fraction(value[0], divisor)), ceil_q(Fraction(value[1], divisor))`
Since `value` is `Interval`, `value[0]` is lower bound, `value[1]` is upper bound.
`divisor` is `degree + 1`, which is $> 0$.
For a positive divisor, dividing an interval $[a, b]$ results in $[a/d, b/d]$.
The floor of $a/d$ is $\le a/d$.
The ceiling of $b/d$ is $\ge b/d$.
So `[floor(a/d), ceil(b/d)]` correctly encloses $[a/d, b/d]$.
This is a valid outwardly rounded interval division by a positive scalar.
Wait, `Fraction(value[0], divisor)` is passed to `floor_q`.
Let's check `Fraction` constructor in Python: `Fraction(numerator, denominator)`.
So `Fraction(value[0], divisor)` correctly creates the rational number `value[0] / divisor`.
This is correct!

Wait, what about `picard.mul(value, H)`?
`H = (ONE >> 8, ONE >> 8)` which is a positive scalar interval $[1/256, 1/256]$.
Multiplying an interval by a positive scalar preserves the order of endpoints.
So `picard.mul([a, b], [H, H])` correctly returns $[a \times H, b \times H]$.

What about `picard.scale2(yy)`?
This multiplies by 2. $[2a, 2b]$.
What about `picard.div2(coeff[2][degree])`?
This multiplies by 1/2. $[a/2, b/2]$.
Both are correct for interval arithmetic (monotone scaling by positive constant).

Let's check the transcript array assignment:
`transcript = [endpoint for degree in range(ORDER) for axis in range(4) for endpoint in center_coeff[axis][degree]]`
Here `center_coeff` is generated with `ORDER - 1`. Wait!
`center_coeff = coefficients(initial, zs, ORDER - 1)`
The loop for `coefficients` is `for degree in range(order):`
If `order = ORDER - 1` (which is 15), the loop runs for `degree` from 0 to 14.
So it computes `coeff[axis][0]` to `coeff[axis][15]`.
Wait, `coeff` is initialized as `[[state[axis] if degree == 0 else (0, 0) for degree in range(order + 1)] ...]`
If `order = 15`, it allocates up to degree `15`.
The loop `for degree in range(order)` computes `coeff[axis][degree + 1]`, which fills up to `coeff[axis][order]`, i.e., `coeff[axis][15]`.
So `center_coeff` contains degrees 0 to 15.
Then in `evaluate`:
`transcript = [endpoint for degree in range(ORDER) for axis in range(4) for endpoint in center_coeff[axis][degree]]`
Here, `range(ORDER)` means degrees 0 to 15.
So `center_coeff[axis][degree]` will access degrees 0 to 15. This matches perfectly!
The length of `transcript` is $16 \times 4 \times 2 = 128$.

What about `box_coeff`?
`box_coeff = coefficients(box, zs, ORDER)`
`ORDER = 16`.
It computes up to degree 16.
`polynomial = [total(center_coeff[axis]) for axis in range(4)]`
This sums degrees 0 to 15 for each axis.
`remainder = [box_coeff[axis][ORDER] for axis in range(4)]`
This gets degree 16 for each axis.
`next_state = [picard.add(polynomial[axis], remainder[axis]) for axis in range(4)]`
This correctly adds them.

Wait! `transcript += [endpoint for value in remainder + polynomial + next_state for endpoint in value]`
`remainder` is 4 intervals.
`polynomial` is 4 intervals.
`next_state` is 4 intervals.
Total 12 intervals $\times 2 = 24$ words.
Transcript length is $128 + 24 = 152$ words.
Correct.

Let's check `next_words = result[144:152]` if `result[-1] == 1`.
`result` has 152 words, plus the status word (1), so 153.
Indices:
0 to 127: center coefficients (128 words)
128 to 135: remainder (8 words)
136 to 143: polynomial (8 words)
144 to 151: next_state (8 words)
152: status (1 word)
Indices `144:152` cover `next_state` exactly!
This is correct.

Wait, is there any error in `next_words[2 * axis + 1] - next_words[2 * axis]`?
For `next_state[axis] = (lower, upper)`, the width is `upper - lower`.
In the transcript, it's flattened as `endpoint for value in next_state for endpoint in value`.
So `value` is `(lower, upper)`.
For `axis = 0`, `next_words[0]` is lower, `next_words[1]` is upper.
`2 * 0 + 1` is 1. `1 - 0` is `upper - lower`.
For `axis = 1`, `next_words[2]` is lower, `next_words[3]` is upper.
`2 * 1 + 1` is 3. `3 - 2` is `upper - lower`.
This is correct.

Wait, what about `MAX_REMAINDER_ABS_RAW`?
`max_remainder = max(abs(word) for word in remainder_words)`
`remainder_words` is `result[128:136]`.
This calculates the max absolute value of the lower and upper bounds of the remainder intervals.
Since remainder is just the 16th Taylor term of the box, this is a mathematically valid computation of that maximum. (Though whether this single term is a mathematically rigorous bound for the infinite remainder is another question).

Let's look at `remainder` mathematically again.
If $X_{n+1} = \frac{H}{n+1} f(X_n)$, and `box` is the interval.
If `polynomial` is evaluated at $t=1$, then we are stepping by $H$.
The true next state is $X(H) = \sum_{n=0}^{15} X_n + \sum_{n=16}^\infty X_n$.
The code takes `next_state` as $\sum_{n=0}^{15} X_n + X_{16}$.
If this `next_state` is meant to enclose the true state $X(H)$, then $X_{16}$ is NOT an upper bound for $\sum_{n=16}^\infty X_n$.
To rigorously bound the tail, one must prove that $\sum_{n=16}^\infty |X_n| \le X_{16}$, which is generally FALSE for Taylor series unless $|X_{n+1}| \le r |X_n|$ with $r \le 1/2$ and some geometric bound is computed.
Taking just the 16th term as the remainder is a classic overreach/bug in interval Picard iterations unless there is an explicit geometric series bound.
Is this an `[OVERREACH]` or `[WRONG]`?
I will flag this as `[OVERREACH]` because calling $X_{16}$ the "remainder" of the Taylor series implies it bounds the tail, which is mathematically unsound without further assumptions. If it's merely a "16-term Taylor polynomial + next term", the nomenclature "remainder" is misleading or wrong. I will flag it as `OVERREACH`.

Let's check `step_div`'s rounding logic again.
`step_div(value, divisor)` -> `divn(picard.mul(value, H), divisor)`.
`divn` -> `floor_q(Fraction(value[0], divisor)), ceil_q(Fraction(value[1], divisor))`.
Wait! What if `value[0]` is negative?
Say `value = (-5, -2)`. `divisor = 2`.
`Fraction(-5, 2) = -2.5`. `floor_q(-2.5)` is `-3`.
`Fraction(-2, 2) = -1`. `ceil_q(-1)` is `-1`.
So the interval is `[-3, -1]`.
The exact interval is `[-2.5, -1]`.
The outwardly rounded interval is `[-3, -1]`.
This works perfectly! Floor always rounds down (towards $-\infty$), Ceil always rounds up (towards $+\infty$).
This is mathematically correct for both positive and negative bounds.

Let's check `picard.mul`.
Wait, the standard interval multiplication `picard.mul([a, b], [c, d])` computes the min and max of $ac, ad, bc, bd$.
Does the code do this? We don't see the implementation of `picard.mul`, but we assume it's standard. I should note that I am assuming standard interval arithmetic in `picard.mul`.

Let's look at `coeff[2][degree + 1]` again.
`xy = total([picard.mul(coeff[1][j], coeff[2][degree - j]) for j in range(degree + 1)])` -- wait!
Let's re-read the code for `coeff[2][degree + 1]`.
`yw = total([picard.mul(coeff[1][j], coeff[2][degree - j]) for j in range(degree + 1)])`
Wait, the code says:
`yw = total([picard.mul(coeff[1][j], coeff[2][degree - j]) for j in range(degree + 1)])`
Then:
`coeff[1][degree + 1] = step_div(picard.sub(xy, picard.div2(picard.add(yw, picard.mul(zs, coeff[1][degree])))), degree + 1)`
Notice `picard.mul(zs, coeff[1][degree])`. This is $Z_s \times Y_n$.
And `yw` is $Y \times W$.
So the formula is $Y_{n+1} = \frac{H}{n+1} (X Y - \frac{1}{2}(Y W + Z_s Y))$.
This matches the formula I derived earlier: $y' = x y - \frac{1}{2}(y w + z_s y)$.
This is mathematically consistent with the series multiplication!

Wait, let's check `coeff[2][degree + 1]`.
`coeff[2][degree + 1] = step_div(picard.sub(xy, picard.add(coeff[2][degree], zs if degree == 0 else (0, 0))), degree + 1)`
The formula used is $W_{n+1} = \frac{H}{n+1} (X Y - (W_n + Z_s \text{ if } n=0 \text{ else } 0))$.
But wait! If $w' = xy - w$, then $W_{n+1} = \frac{H}{n+1} (XY_n - W_n)$.
Why is $Z_s$ subtracted ONLY for $n=0$?
If $w' = xy - w - Z_s$, then the Taylor expansion would be $W_{n+1} = \frac{H}{n+1} (XY_n - W_n - Z_s)$ for ALL $n \ge 0$.
But the code does `picard.add(coeff[2][degree], zs if degree == 0 else (0, 0))`.
This means it computes $XY_n - W_n - Z_s$ for $n=0$, and $XY_n - W_n$ for $n>0$.
This corresponds to an ODE $w' = xy - w - Z_s \delta(t)$ (Dirac delta at $t=0$), OR it means the differential equation is $w' = xy - w$, but the initial condition $W_0$ was somehow defined with $Z_s$ subtracted from its derivative?
Wait, if $w' = xy - w$, then $W_1 = W_0' = XY_0 - W_0$. Why subtract $Z_s$?
Maybe $W_0$ in the ODE is actually the initial state of $w$, but the code's `state[2]` is NOT $W_0$. What if the user equation was $w' = xy - w$, but `state[2]` passed to `coefficients` is actually the initial box, and $Z_s$ is some input?
Actually, it doesn't matter if it's a standard ODE, it's the mathematical translation of whatever formula is intended. If the intent was $w' = xy - w - z_s$, this code is `[WRONG]` because it only subtracts $z_s$ for $n=0$. If the intent was exactly as written (e.g., $z_s$ is an impulse), it's `[OK]`. I'll flag it as `[OK]` but note the $n=0$ dependence, which is highly suspicious for standard continuous ODEs.

Let's re-read carefully:
`coeff[1][degree + 1] = ... picard.mul(zs, coeff[1][degree]) ...`
Here, `zs` is multiplied by `coeff[1][degree]` (which is $Y_n$) for ALL degrees!
So $y' = xy - \frac{1}{2} y w - \frac{1}{2} z_s y$.
This implies $Z_s$ is a scalar coefficient in the $Y$ equation.
If $Z_s$ is a scalar coefficient, it appears in all degrees.
But in the $W$ equation: `xy - picard.add(coeff[2][degree], zs if degree == 0 else (0, 0))`.
This means $w' = xy - w - z_s$ for degree 0, and $w' = xy - w$ for degree > 0.
Why would $Z_s$ be a constant offset only at $n=0$?
Because $W_{n+1} = \frac{1}{n+1} W^{(n+1)}(0)$.
If $W_{n+1}$ has $Z_s$ only for $n=0$, it means $W'(0) = XY_0 - W_0 - Z_s$.
But $W''(0) = XY_1 - W_1$.
If $w' = xy - w - z_s$, then $w'' = (xy)' - w' = x'y + xy' - (xy - w - z_s) = x'y + xy' - xy + w + z_s$.
So $Z_s$ WOULD appear in $W''$ and all higher derivatives.
The fact that it doesn't appear for $n>0$ implies $w' = xy - w - z_s$ is FALSE.
Wait, what if $w' = xy - w$, and `zs` is just a correction term applied to $W_1$?
Or what if the code is strictly translating a discrete formula?
If the domain is a specific formula from a paper, I can't know it. But mathematically, an affine term $Z_s$ vanishing for $n>0$ is inconsistent with a standard polynomial ODE $w' = f(t, x, y, w)$ where $Z_s$ is a constant parameter, UNLESS $Z_s$ is an impulse $\delta(t)$. I will flag it as `[TIGHTENABLE]` or `[OK]` with a query.

Wait, look at `constant = picard.add(picard.div2(zs), (ONE, ONE)) if degree == 0 else (0, 0)`
This is $1/2 Z_s + 1$ for $n=0$, and $0$ for $n>0$.
If it were a constant parameter in $v' = x - y - \frac{1}{2} w - (\frac{1}{2} Z_s + 1)$, then the constant would appear in all $n$.
But it only appears for $n=0$.
This is mathematically weird if they are supposed to be Taylor coefficients of an ODE with constant parameters.
Let me check if `degree` is 0: yes, for $n=0$.
Could it be that the ODEs are evaluated as Picard integrals?
In Picard iteration, $w_{k+1}(t) = w(0) + \int_0^t (xy - w_k(s)) ds$.
If we expand this as a polynomial, $W_0$ is the initial state.
$W_1 = W_0 + XY_0 - W_0 = XY_0$.
Wait, if the equations are just some algebraic recurrences for the coefficients of a Picard iteration, then the code is whatever it is.
Is there a leap? `OVERREACH`? I will
