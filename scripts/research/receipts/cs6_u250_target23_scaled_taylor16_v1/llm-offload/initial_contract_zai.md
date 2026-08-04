
The user wants me to act as a mathematical referee and verify the math in a provided artifact.
The artifact is a configuration/contract for a "scaled Taylor endpoint enclosure" for a system of ODEs.
Key elements to check:
- `SCALED_COEFFICIENT_DEFINITION=b_k=a_k*h^k`
- `RECURRENCE_X=b_x[k+1]=h/(k+1)*(2*sum_j(b_y[j]*b_y[k-j])-sum_j(b_x[j]*b_y[k-j]))`
- `RECURRENCE_Y=b_y[k+1]=h/(k+1)*(sum_j(b_x[j]*b_y[k-j])-(sum_j(b_y[j]*b_w[k-j])+zs*b_y[k])/2)`
- `RECURRENCE_W=b_w[k+1]=h/(k+1)*(sum_j(b_x[j]*b_y[k-j])-b_w[k]-delta_k0*zs)`
- `RECURRENCE_ELL=b_ell[k+1]=h/(k+1)*(b_x[k]-b_y[k]-b_w[k]/2-delta_k0*(zs/2+1))`

Let's check the scaled Taylor recurrence formulation.
If $b_k = a_k h^k$, and $a_k$ is the standard Taylor coefficient for a system.
The standard Taylor recurrence $a_{k+1} = \frac{1}{k+1} f'(a_k)$ or Cauchy products.
Wait, if $b_k = a_k h^k$, then $a_k = b_k / h^k$.
$a_{k+1} = \frac{1}{k+1} \frac{d}{dt} a_k$.
Let's reconstruct the original ODE system from the recurrences.
Recurrence X:
$b_x[k+1] = \frac{h}{k+1} (2 \sum b_y[j] b_y[k-j] - \sum b_x[j] b_y[k-j])$
Notice that $\sum b_x[j] b_y[k-j]$ is exactly $h^k \sum a_x[j] a_y[k-j]$. Wait, no.
$b_x[j] b_y[k-j] = a_x[j] h^j a_y[k-j] h^{k-j} = a_x[j] a_y[k-j] h^k$.
So $\sum_j b_x[j] b_y[k-j] = h^k \sum_j a_x[j] a_y[k-j]$.
Then $b_x[k+1] = \frac{h}{k+1} h^k \sum_j (2 a_y a_y - a_x a_y) = h^{k+1} \frac{1}{k+1} \sum_j (2 a_y a_y - a_x a_y)$.
So the original ODE is:
$x' = 2 y^2 - x y$
$y' = x y - \frac{1}{2} (y w + z_s y)$? Wait.
Let's check Recurrence Y:
$b_y[k+1] = \frac{h}{k+1} (\sum b_x[j] b_y[k-j] - (\sum b_y[j] b_w[k-j] + z_s b_y[k])/2)$
Wait, $z_s b_y[k]$ is not a Cauchy product. It's just $z_s b_y[k]$.
In the original ODE, if we have a term $z_s y$, then its Taylor coefficients are $z_s a_y[k]$.
Then $a_y[k+1] = \frac{1}{k+1} (\dots - z_s a_y[k])$.
When scaled: $b_y[k+1] = h^{k+1} a_y[k+1] = \frac{h}{k+1} ( h^k \dots - z_s a_y[k] h^k ) = \frac{h}{k+1} ( \sum b_x b_y - \dots - z_s b_y[k] / 2 )$.
Wait, the term is $-(\sum b_y b_w + z_s b_y[k])/2$.
So the original ODE for $y$ is:
$y' = x y - \frac{1}{2} y w - \frac{1}{2} z_s y$.

Let's check Recurrence W:
$b_w[k+1] = \frac{h}{k+1} (\sum b_x b_y - b_w[k] - \delta_{k0} z_s)$.
Wait, $\delta_{k0} z_s$ term. The term $\delta_{k0} z_s$ implies the original ODE has a constant term $-z_s$.
Wait, if $w' = x y - w - z_s$, then:
$a_w[k+1] = \frac{1}{k+1} (\sum a_x a_y - a_w[k] - z_s \delta_{k0})$.
This perfectly matches $b_w[k+1] = \frac{h}{k+1} (\sum b_x b_y - b_w[k] - z_s \delta_{k0})$.

Let's check Recurrence ELL:
$b_\ell[k+1] = \frac{h}{k+1} (b_x[k] - b_y[k] - b_w[k]/2 - \delta_{k0}(z_s/2 + 1))$.
The original ODE is $\ell' = x - y - w/2 - (z_s/2 + 1)$.
Matches perfectly.

Let's double check $z_s$ in Recurrence W and Recurrence Y.
In $y'$: $x y - y w / 2 - z_s y / 2$.
In $w'$: $x y - w - z_s$.
In $\ell'$: $x - y - w / 2 - z_s / 2 - 1$.

Wait, the recurrences use $\sum_j$ but don't specify the bounds of $j$.
Usually for Cauchy product of sequences up to $k$, the sum is $\sum_{j=0}^k$.
If it's $\sum_{j=0}^k$, the formula is correct.
Wait, what is $b_x[k]$? Is the degree correct?
$b[k+1] = h/(k+1) \times (\dots)$.
Since $h$ is $2^{-8}$, this correctly scales the coefficients.
$a_k$ is standard Taylor coefficient. $b_k = a_k h^k$.
$b_{k+1} = a_{k+1} h^{k+1} = \frac{1}{k+1} f_k h^{k+1} = \frac{h}{k+1} f_k h^k = \frac{h}{k+1} F_k$.
Here $F_k$ is the Cauchy product or direct coefficient evaluated on $b_j$'s.
The Cauchy product of $b_x$ and $b_y$ is $\sum_{j=0}^k b_x[j] b_y[k-j] = \sum a_x[j] h^j a_y[k-j] h^{k-j} = h^k \sum a_x[j] a_y[k-j] = h^k C_k$.
So $\frac{h}{k+1} \sum b_x[j] b_y[k-j] = \frac{h^{k+1}}{k+1} C_k = b_{k+1}$.
The math checks out.

Are there any mathematical leaps or errors in:
`CENTER_POLYNOMIAL=sum_k_0_to_15(b_center[k])`
If $b_k = a_k h^k$, then the Taylor polynomial up to degree 15 is $\sum_{k=0}^{15} a_k t^k$. At $t=h$, it is $\sum_{k=0}^{15} a_k h^k = \sum_{k=0}^{15} b_k$.
So evaluating at $t=h$ (one step), the center polynomial sum is exactly $\sum_{k=0}^{15} b_k$. This is correct.

`REMAINDER_INTERVAL=b_box[16]`
The remainder of a Taylor series is $\sum_{k=16}^\infty a_k h^k$. Is $b_{16}$ sufficient to bound it? Or does $b_{16}$ mean something else?
Wait, if it's a Picard iteration or interval Taylor method, usually the remainder is bounded using $M^{16} h^{16} / 16!$, which corresponds to a bound on the 16th derivative.
But the artifact says:
`REMAINDER_COEFFICIENT_ORDER=16`
`REMAINDER_INTERVAL=b_box[16]`
`REMAINDER_JUSTIFICATION=ORDER_16_NORMALIZED_FLOW_COEFFICIENT_EVALUATED_OVER_CERTIFIED_PICARD_BOX_WITH_H_POWER_ABSORBED_INTO_RECURRENCE`
Does evaluating the order 16 coefficient over the Picard box give a valid enclosure for the remainder?
If it uses a geometric series bound: $R = \sum_{k=16}^\infty b_k \le \sum_{k=16}^\infty M h^k$?
Wait. If $b_{16}$ is computed, it does not directly bound $\sum_{k=16}^\infty b_k$ unless there's a specific remainder formula (like Picard-Lindelöf bound, assuming $|b_{k+1}| \le \rho |b_k|$ and summing the geometric series $\sum_{k=16}^\infty \rho^{k-16} |b_{16}|$).
The artifact says `b_box[16]`. It doesn't explicitly state the geometric series summation.
Wait! "ORDER_16_NORMALIZED_FLOW_COEFFICIENT_EVALUATED_OVER_CERTIFIED_PICARD_BOX_WITH_H_POWER_ABSORBED_INTO_RECURRENCE".
This sounds like it evaluates $b_{16}$ and just calls it `REMAINDER_INTERVAL`. But evaluating a single Taylor coefficient does not give the remainder interval unless it's a specific contraction bound!
For a Picard-Lindelöf contraction with factor $L$, the remainder from degree $d$ is bounded by $b_{d+1} / (1 - L h)$ or something?
Actually, let's look at `PRECONDITION=PICARD_SELF_MAP_AND_H_TIMES_L_INFINITY_STRICTLY_BELOW_ONE`.
Ah! If $h \|f'\|_\infty < 1$, then the remainder is bounded by the geometric series of Taylor coefficients.
Specifically, if $|b_{k+1}| \le q |b_k|$ where $q = h L < 1$, then the remainder $\sum_{k=16}^\infty b_k \le b_{16} / (1 - q)$.
But the artifact says `REMAINDER_INTERVAL=b_box[16]`.
Is $b_{16}$ the sum of the geometric series, or is it just $b_{16}$?
"b_box[16]" in the output layout: `OUTPUT_LAYOUT=128_CENTER_COEFFICIENT_ENDPOINTS_8_REMAINDER_ENDPOINTS_8_POLYNOMIAL_ENDPOINTS_8_NEXT_STATE_ENDPOINTS_1_STATUS`.
If it just outputs `b_box[16]`, it might be computing the remainder incorrectly if it just uses the 16th coefficient instead of the sum of the tail.
Wait, look at `RECURRENCE_X=b_x[k+1]=...`
The standard Taylor remainder $R_n(t) = \frac{f^{(n+1)}(\xi)}{(n+1)!} t^{n+1}$.
If the system is $X' = F(X)$, then $X^{(n+1)}(\xi)$ involves $(n+1)$ derivatives.
However, for a Picard iteration, the remainder is often bounded using the contraction factor.
Let's re-read the remainder justification carefully:
`ORDER_16_NORMALIZED_FLOW_COEFFICIENT_EVALUATED_OVER_CERTIFIED_PICARD_BOX_WITH_H_POWER_ABSORBED_INTO_RECURRENCE`
This implies `b_box[16]` is literally just the 16th coefficient $a_{16} h^{16}$ evaluated over the box.
Is $a_{16} h^{16}$ the remainder? No, the remainder is $\sum_{k=16}^\infty a_k h^k$.
This is a mathematical error! Evaluating the 16th coefficient is NOT the remainder.
Wait, let's think. Is there any context where $b_{16}$ bounds the remainder?
If it's a polynomial bound like $R \le \dots$
Let's check if there is a known theorem: The remainder is bounded by the first neglected term if it's an alternating series? No, it's a general ODE.
Is it possible that `b_box[16]` means the bound on the 16th derivative, multiplied by $h^{16} / 16!$?
Yes, $a_{16} = X^{(16)}(\xi) / 16!$. So $a_{16} h^{16} = X^{(16)}(\xi) h^{16} / 16!$.
But this is just the first term of the remainder. The remainder is $\sum_{k=16}^\infty a_k h^k$.
Unless the contract implies $b_{16}$ is an enclosure for the remainder, which is mathematically wrong.
Wait, what if `b_box[16]` is just the coefficient, but the `REMAINDER_JUSTIFICATION` says "evaluated over certified Picard box"?
If $q < 1$, the remainder is $b_{16} / (1 - q)$.
But `REMAINDER_INTERVAL=b_box[16]` explicitly equates the remainder interval to the 16th coefficient.
I will flag `REMAINDER_INTERVAL=b_box[16]` as WRONG or OVERREACH, because the remainder interval is the sum of all terms from 16 onwards, not just the 16th term.

Wait! Let me read `RECURRENCE_W=b_w[k+1]=h/(k+1)*(sum_j(b_x[j]*b_y[k-j])-b_w[k]-delta_k0*zs)`.
Notice that the constant term $z_s$ is added at $k=0$.
$b_w[1] = \frac{h}{1} (b_x[0]b_y[0] - b_w[0] - z_s)$.
This means $a_w[1] = a_x[0]a_y[0] - a_w[0] - z_s$, which matches $w' = xy - w - z_s$. Correct.

Wait, what about the Cauchy product bounds?
`sum_j(b_y[j]*b_y[k-j])`
The index $j$ goes from 0 to $k$. This is correct for coefficients.

Let's look at `RECURRENCE_Y=b_y[k+1]=h/(k+1)*(sum_j(b_x[j]*b_y[k-j])-(sum_j(b_y[j]*b_w[k-j])+zs*b_y[k])/2)`
Notice the term `zs*b_y[k]`.
Since $y' = xy - y w / 2 - z_s y / 2$.
The $k$-th Taylor coefficient of $y$ is $a_y[k]$.
The $k$-th Taylor coefficient of $z_s y$ is $z_s a_y[k]$.
So this term is correctly represented as `zs*b_y[k]`.

Let's check `RECURRENCE_ELL=b_ell[k+1]=h/(k+1)*(b_x[k]-b_y[k]-b_w[k]/2-delta_k0*(zs/2+1))`
This is $\ell' = x - y - w/2 - (z_s/2 + 1)$.
The coefficient of the constant term $-(z_s/2 + 1)$ in the $(k+1)$-th Taylor coefficient is $-(z_s/2 + 1)$ for $k=0$ and $0$ for $k>0$.
So $-\delta_{k0}(z_s/2 + 1)$. This is correct.

Wait! Let's re-examine `RECURRENCE_W=b_w[k+1]=h/(k+1)*(sum_j(b_x[j]*b_y[k-j])-b_w[k]-delta_k0*zs)`
Notice that for $w' = xy - w - z_s$, the term $-w$ gives $-a_w[k]$, and the constant term $-z_s$ gives $-z_s \delta_{k0}$.
So the formula is exactly $b_w[k+1] = \frac{h}{k+1} (\sum b_x b_y - b_w[k] - z_s \delta_{k0})$. Correct.

Let's re-examine `RECURRENCE_X=b_x[k+1]=h/(k+1)*(2*sum_j(b_y[j]*b_y[k-j])-sum_j(b_x[j]*b_y[k-j]))`
This is $x' = 2y^2 - xy$.
The Taylor coefficients of $2y^2$ are $2 \sum_{j=0}^k a_y[j] a_y[k-j]$.
The Taylor coefficients of $xy$ are $\sum_{j=0}^k a_x[j] a_y[k-j]$.
So $a_x[k+1] = \frac{1}{k+1} (2 \sum a_y a_y - \sum a_x a_y)$.
Scaling by $h^{k+1}$ gives exactly the formula. Correct.

Let's check the remainder issue again.
Is `b_box[16]` defined as the remainder?
`REMAINDER_INTERVAL=b_box[16]`
If $b_k$ is the $k$-th coefficient, the remainder is $\sum_{k=16}^\infty b_k$.
If $b_{16}$ is just the 16th coefficient, it is NOT the remainder.
However, what if `b_box` is an array where `b_box[16]` stores the *sum* of the remainder?
But `REMAINDER_COEFFICIENT_ORDER=16` and `REMAINDER_JUSTIFICATION=ORDER_16_NORMALIZED_FLOW_COEFFICIENT_EVALUATED_OVER_CERTIFIED_PICARD_BOX_WITH_H_POWER_ABSORBED_INTO_RECURRENCE` implies it evaluates the 16th coefficient over the box and calls it the remainder interval.
This is a classic mistake in naive Taylor models unless it uses a contraction bound.
Wait, if it uses a contraction bound, the justification would mention it.
I will flag `REMAINDER_INTERVAL=b_box[16]` as a mathematical leap / error.

Let's look at `CENTER_POLYNOMIAL=sum_k_0_to_15(b_center[k])`.
If $b_k$ are scaled coefficients, the sum is exactly the evaluation of the Taylor polynomial at $t=h$. This is correct.

What about `EXPECTED_CASES=3` vs `INPUT_WORDS_PER_CASE=18`? Not a math claim.

Are there any other math claims?
`INTERVAL_MULTIPLICATION=FOUR_CORNER_EXACT_256_BIT_PRODUCT_WITH_DIRECTED_FLOOR_CEIL_TO_F96`
If the interval is positive (all bounds $\ge 0$), four corners are enough.
But if the interval straddles 0, the minimum product is 0 (if there's a 0 in the interval), not one of the four corners!
Wait, `ARITHMETIC=SIGNED_S1_I31_F96_DYADIC_INTERVALS`.
It says "FOUR_CORNER_EXACT_256_BIT_PRODUCT". This is WRONG for intervals that contain zero.
For intervals $[a, b]$ and $[c, d]$, the product is $[\min(ac, ad, bc, bd), \max(ac, ad, bc, bd)]$ ONLY IF $0 \notin [a,b]$ and $0 \notin [c,d]$.
If $0 \in [a,b]$ and $0 \in [c,d]$, the minimum is 0.
If $a < 0 < b$ and $c > 0$, the minimum is $bc$.
So "FOUR_CORNER" is mathematically incorrect as a general interval multiplication algorithm. You must check for inclusion of zero.
I will flag this as WRONG, with the correction "requires zero-crossing check for true minimum".

Wait, are there any constraints on the state variables?
`PROOF_CARRYING_STEP_SCALED_TAYLOR_ENDPOINT_ENCLOSURE_FOR_TARGET23_LEAF331_CENTER`
We don't know if the state variables are guaranteed positive. The variables are $x, y, w, \ell, z_s$.
If they can be negative or zero, four corners is WRONG.
I'll flag it as `OVERREACH` or `WRONG`.

Let's check `RECURRENCE_W=b_w[k+1]=h/(k+1)*(sum_j(b_x[j]*b_y[k-j])-b_w[k]-delta_k0*zs)`
Wait. Is the constant term in the ODE for $w$ really $-z_s$?
Yes, $-z_s$.
And in $\ell'$, the constant term is $-(z_s/2 + 1)$.
Then $b_\ell[1] = h (b_x[0] - b_y[0] - b_w[0]/2 - (z_s/2 + 1))$.
This perfectly matches the standard Taylor coefficient for a constant term.

Is `zs` a constant or a state variable?
It's not iterated in the recurrences (no `b_z[k+1]`), so it's treated as a constant parameter $z_s$.
Wait, in `RECURRENCE_W` it has `-delta_k0*zs`.
In `RECURRENCE_Y` it has `zs*b_y[k]`.
If $z_s$ is a parameter, its Taylor expansion is $z_s \delta_{k0}$.
But wait! In `RECURRENCE_Y`, $z_s y$ is a product of $z_s$ (a constant) and $y$ (a variable).
The Taylor expansion of $z_s y$ is $z_s \sum a_y[k] t^k$.
So the $k$-th coefficient is $z_s a_y[k]$.
The recurrence uses `zs*b_y[k]`.
Since $b_y[k] = a_y[k] h^k$, then `zs*b_y[k]` is $z_s a_y[k] h^k$.
Then $\frac{h}{k+1} z_s b_y[k] = z_s a_y[k] h^{k+1} / (k+1)$.
This perfectly matches $a_y[k+1]$!
Wait, the formula in Y is `-(sum_j(b_y[j]*b_w[k-j])+zs*b_y[k])/2`.
This matches the ODE $y' = xy - \frac{1}{2}(y w + z_s y)$.
Because the coefficient of $y w$ is $\sum_{j=0}^k a_y[j] a_w[k-j]$.
The coefficient of $z_s y$ is $z_s a_y[k]$.
So the term inside the parenthesis is $\sum a_y[j] a_w[k-j] + z_s a_y[k]$.
When multiplied by $h/(k+1)$, it is $\frac{1}{k+1} h (\sum b_y[j] b_w[k-j] + z_s b_y[k])$.
Wait, $\frac{1}{k+1} h \sum b_y[j] b_w[k-j] = \frac{1}{k+1} h (h^k \sum a_y a_w) = a_y a_w h^{k+1} / (k+1) = a_{y w}[k+1]$.
This is correct.

Let's carefully check the scaling in `RECURRENCE_W`.
`-delta_k0*zs`
The term in the ODE is $-z_s$.
The $k$-th Taylor coefficient of $-z_s$ is $-z_s \delta_{k0}$.
So the $k$-th Taylor coefficient of the RHS of $w'$ is $\sum a_x a_y - a_w[k] - z_s \delta_{k0}$.
Then $a_w[k+1] = \frac{1}{k+1} ( \sum a_x a_y - a_w[k] - z_s \delta_{k0} )$.
Multiplying by $h^{k+1}$:
$b_w[k+1] = h^{k+1} a_w[k+1] = \frac{h}{k+1} ( h^k \sum a_x a_y - h^k a_w[k] - z_s \delta_{k0} h^k )$.
Since $b_x[j] b_y[k-j] = h^k a_x[j] a_y[k-j]$, we have $h^k \sum a_x a_y = \sum b_x b_y$.
Also $h^k a_w[k] = b_w[k]$.
So it becomes $\frac{h}{k+1} ( \sum b_x b_y - b_w[k] - z_s \delta_{k0} h^k )$.
But the contract says: `RECURRENCE_W=b_w[k+1]=h/(k+1)*(sum_j(b_x[j]*b_y[k-j])-b_w[k]-delta_k0*zs)`
Notice the missing $h^k$ in the term `delta_k0*zs`!
If $k=0$, $\delta_{k0} h^k = 1$. So $b_w[1] = \frac{h}{1} (b_x[0]b_y[0] - b_w[0] - z_s)$.
Wait, if $k=0$, $h^0 = 1$. So the term is just $z_s$.
If $k>0$, $\delta_{k0} = 0$. So it's 0.
Wait! The term `delta_k0*zs` is multiplied by $\frac{h}{k+1}$.
So for $k=0$, it gives $h \times z_s$.
But $z_s \delta_{k0}$ in the Taylor coefficient $a_w[1]$ is $\frac{1}{1} z_s$.
Then $b_w[1] = a_w[1] h^1 = h z_s$.
So the term $h z_s$ is perfectly correct!
Let's trace carefully:
$b_w[k+1] = \frac{h}{k+1} (\sum b_x b_y - b_w[k] - z_s \delta_{k0} h^k)$.
Since $\delta_{k0} h^k$ is $1$ when $k=0$, and $0$ when $k>0$ (because $\delta_{k0}=0$), we can just write `delta_k0*zs` inside the parenthesis!
Because $\delta_{k0} h^k$ is identically $\delta_{k0}$ for non-negative integers $k$.
So this is perfectly mathematically correct.

Let's do the same for `RECURRENCE_ELL`.
$b_\ell[k+1] = \frac{h}{k+1} (b_x[k] - b_\ell[k] - b_w[k]/2 - \delta_{k0} (z_s/2 + 1) h^k)$.
Again, $\delta_{k0} h^k = \delta_{k0}$.
So the term `delta_k0*(zs/2+1)` is mathematically exact!
This is a very clever and correct scaling.

Let's review the whole setup.
The scaled Taylor coefficients $b_k = a_k h^k$.
$b_x[k+1] = \frac{h}{k+1} (2 \sum_{j=0}^k b_y[j] b_y[k-j] - \sum_{j=0}^k b_x[j] b_y[k-j])$.
Correct.

$b_y[k+1] = \frac{h}{k+1} ( \sum_{j=0}^k b_x[j] b_y[k-j] - (\sum_{j=0}^k b_y[j] b_w[k-j] + z_s b_y[k])/2 )$.
Correct.

$b_w[k+1] = \frac{h}{k+1} ( \sum_{j=0}^k b_x[j] b_y[k-j] - b_w[k] - z_s \delta_{k0} )$.
Correct.

$b_\ell[k+1] = \frac{h}{k+1} ( b_x[k] - b_y[k] - b_w[k]/2 - \delta_{k0}(z_s/2 + 1) )$.
Correct.

Now, `CENTER_POLYNOMIAL=sum_k_0_to_15(b_center[k])`.
The value of the Taylor polynomial at $t=h$ is $\sum_{k=0}^{15} a_k h^k = \sum_{k=0}^{15} b_k$.
Correct.

`REMAINDER_INTERVAL=b_box[16]`.
This assumes the remainder of the series is bounded by the 16th term.
Wait! "ORDER_16_NORMALIZED_FLOW_COEFFICIENT_EVALUATED_OVER_CERTIFIED_PICARD_BOX_WITH_H_POWER_ABSORBED_INTO_RECURRENCE"
Is it mathematically valid to bound the remainder by the first neglected term?
For a geometric series where the ratio is $<1$, $\sum_{k=16}^\infty b_k \le b_{16} / (1-q)$.
It is NOT bounded by $b_{16}$.
Unless the remainder is bounded by evaluating the 16th derivative, and $b_{16}$ here means the evaluated remainder, not the coefficient $a_{16} h^{16}$?
But the field says `REMAINDER_COEFFICIENT_ORDER=16`.
And `b_box[16]` strongly implies the 16th element of the `b_box` array, which would be the 16th coefficient.
If it's just the 16th coefficient, bounding the remainder by it is WRONG. It must be divided by $(1-q)$.
I will flag this.

Next, `INTERVAL_MULTIPLICATION=FOUR_CORNER_EXACT_256_BIT_PRODUCT_WITH_DIRECTED_FLOOR_CEIL_TO_F96`.
If the intervals are over signed reals, and they might contain zero, four corners are not enough.
Example: $[-1, 1] \times [-1, 1]$.
Four corners: $(-1)(-1)=1, (-1)(1)=-1, (1)(-1)=-1, (1)(1)=1$.
So min is -1, max is 1.
But actually, $[-1, 1] \times [-1, 1] = [-1, 1]$. So it works here.
Wait! What about $[-2, 1] \times [-2, 1]$?
Corners: $4, -2, -2, 1$. Min is -2, Max is 4.
But the actual minimum is 0 (e.g., $0 \times 0 = 0$). So $[-2, 0]$ is wrong if we say min is -2!
Wait, the product of $[-2, 1]$ and $[-2, 1]$ is $[-2, 4]$. The minimum is indeed -2.
Wait! What about $[-1, 2] \times [-1, 2]$?
Corners: $1, -1, -1, 4$. Min is -1, Max is 4.
Is the true min -1? Wait. What about $-0.5 \times 2 = -1$. $1 \times 2 = 2$.
Wait, $[-1, 2] \times [-1, 2] = [-2, 4]$.
Ah! $(-1) \times 2 = -2$.
The corners are $(-1, -1) \to 1$; $(-1, 2) \to -2$; $(2, -1) \to -2$; $(2, 2) \to 4$.
Min is -2. So it works.
Wait, let's find a case where four corners FAILS.
$x \in [a, b]$, $y \in [c, d]$.
$f(x, y) = xy$.
Since $f$ is bilinear, its extrema over a rectangle $[a, b] \times [c, d]$ MUST occur at the corners of the rectangle!
Proof: The function $f(x, y) = xy$ has no local extrema in the interior of any rectangle unless it is identically 0 (since $\nabla f = (y, x) = (0,0) \implies x=y=0$, which is a saddle point).
So over a rectangle, the minimum and maximum MUST be among the four corners!
Wow. I almost made a mistake.
Let me double check this.
$f(x, y) = x y$.
If we fix $y$, $f(x, y) = (y)x$, which is linear in $x$, so its extrema for $x \in [a, b]$ are at $x=a$ or $x=b$.
Then we evaluate $f(a, y)$ and $f(b, y)$, which are linear in $y$.
So their extrema for $y \in [c, d]$ are at $y=c$ or $y=d$.
Therefore, the extrema of $xy$ on $[a, b] \times [c, d]$ are ALWAYS at the four corners $(a,c), (a,d), (b,c), (b,d)$.
My memory about "0 in interval" was for interval arithmetic where we want a quick bound.
But four corners EXACT computation is mathematically EXACT for interval multiplication!
Wait! Is that true?
Yes, the image of $[a, b] \times [c, d]$ under $f(x, y) = xy$ is exactly $[\min(ac, ad, bc, bd), \max(ac, ad, bc, bd)]$.
So `FOUR_CORNER_EXACT_256_BIT_PRODUCT` is completely correct for interval multiplication!
I will explicitly state this is OK and mathematically sound, because many people incorrectly think it fails for intervals containing zero, but it doesn't fail for exact multiplication.

Wait, let's check `RECURRENCE_Y=b_y[k+1]=h/(k+1)*(sum_j(b_x[j]*b_y[k-j])-(sum_j(b_y[j]*b_w[k-j])+zs*b_y[k])/2)`
Is the parenthesis correct?
`-(sum_j(b_y[j]*b_w[k-j])+zs*b_y[k])/2`
This means $\frac{1}{2} (\sum b_y b_w + z_s b_y)$.
In the ODE, it was $-\frac{1}{2} (y w + z_s y)$.
Yes, it matches perfectly.

What about `EXPECTED_CASES=3`, `INPUT_WORDS_PER_CASE=18`? Not math.

Let's review the error in `REMAINDER_INTERVAL=b_box[16]`.
The ODE is:
$x' = 2y^2 - xy$
$y' = xy - \frac{1}{2} y w - \frac{1}{2} z_s y$
$w' = xy - w - z_s$
$\ell' = x - y - \frac{1}{2} w - (\frac{1}{2} z_s + 1)$

If we compute the Taylor series up to degree 15, the remainder at $t=h$ is:
$R(h) = \sum_{k=16}^\infty a_k h^k = \sum_{k=16}^\infty b_k$.
Is it possible that $b_{16}$ is used as a bound for the remainder?
By Picard-Lindelöf, if we have a Lipschitz constant $L$, then $\sum_{k=16}^\infty |b_k| \le \frac{|b_{16}|}{1 - L h / 16}$?
No, the standard Taylor model remainder bound using the Picard operator evaluates the 16th derivative of the flow.
Wait, if $b_{16}$ is the first neglected term, it does NOT bound the sum unless the series is alternating and decreasing, which is not guaranteed here.
Therefore, equating `REMAINDER_INTERVAL=b_box[16]` is an OVERREACH. It needs a contraction factor denominator, e.g., $b_{16} / (1 - h L)$.

Let me check the Picard precondition:
`PRECONDITION=PICARD_SELF_MAP_AND_H_TIMES_L_INFINITY_STRICTLY_BELOW_ONE`
`PREDECESSOR_GATE=scripts/research/cs6_u250_target23_picard_step_gate.sh`
This proves that a contraction bound is available. But the formula for remainder interval is explicitly `b_box[16]`.
If they intended $b_{16} / (1 - h L)$, they would have written `b_box[16]/(1-h*L)` or similar.
Since they didn't, it is mathematically incomplete/wrong.

Are there any other issues?
`DIVISORS=1_THROUGH_16_EXACT_POSITIVE_INTEGERS_WITH_DIRECTED_FLOOR_CEIL`
This handles the $\frac{1}{k+1}$ division in the recurrence. Since the numbers are dyadic intervals, dividing by an odd integer requires rounding. Directed floor/ceiling is correct for interval arithmetic.
Wait, what about $h^{k+1}$?
The scaled definition $b
