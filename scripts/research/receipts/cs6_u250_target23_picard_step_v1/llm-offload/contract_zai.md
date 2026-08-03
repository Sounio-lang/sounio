The user wants me to review the mathematical content of the provided artifact.
Artifact is a configuration/schema for a hardware proof-carrying validated system for a Picard iteration step.
Key fields with math content:
- ARITHMETIC=SIGNED_S1_I31_F96_DYADIC_INTERVALS
  - TOTAL_BITS=128, FRACTION_BITS=96
  - MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111 (Check: 1 sign + 31 int + 96 frac = 128 bits. Max raw exclusive value is 2^127. Why 2^111?)
- TIME_INTERVAL=[0,2^-8]
- ZS_EXACT_DECIMAL=22.3274637391
- VECTOR_FIELD_X=2*y*y-x*y
- VECTOR_FIELD_Y=x*y-y*(w+zs)/2
- VECTOR_FIELD_W=x*y-w-zs
- VECTOR_FIELD_ELL=x-y-(w+zs)/2-1
- INTERVAL_MULTIPLICATION=FOUR_CORNER_MIN_MAX_WITH_FLOOR_CEIL_DIRECTED_ROUNDING
- DIVISION=EXACT_POSITIVE_INTEGER_TWO_WITH_FLOOR_CEIL_DIRECTED_ROUNDING
- PICARD_IMAGE=X0_PLUS_[0,2^-8]_TIMES_F_OF_B
- SELF_MAP_OBLIGATION=PICARD_IMAGE_IS_STRICTLY_INSIDE_B_ON_ALL_FOUR_COMPONENTS
- LIPSCHITZ_NORM=INFINITY_NORM_ROW_SUM_BOUND_OF_JACOBIAN_ON_B
- CONTRACTION_OBLIGATION=CEIL_Q96(L_INFINITY/256)_RAW_IS_STRICTLY_LESS_THAN_2^96
  - L_INFINITY / 256. The time interval is 2^-8, which is 1/256. Wait, the Picard operator is P(x) = x0 + h * f(x). The Lipschitz constant of P is h * L_f. So L_P = 2^-8 * L_INFINITY.
  - If L_P < 1, then it's a contraction. L_P is in Q96. So L_P < 1 means L_P_raw < 2^96.
  - Let's check the formula: CEIL_Q96(L_INFINITY/256)_RAW < 2^96. This translates to CEIL_Q96(L_P) < 1. This is correct for a strict contraction (<1).

Let's verify the bit allocation:
"S1_I31_F96" implies 1 sign bit, 31 integer bits, 96 fractional bits.
Total bits = 1 + 31 + 96 = 128. This matches TOTAL_BITS=128.
But signed Q-format usually uses two's complement, which is 1 sign bit and 127 magnitude bits. If interpreted as sign-magnitude or 2's complement, the max positive value is 2^127 - 1. Wait, for a signed Q31.96, the max absolute value is 2^127.
However, the schema says MAX_ABS_INPUT_RAW_EXCLUSIVE = 2^111. Why 2^111? Is it out of scope or just an arbitrary bound to avoid overflow during interval multiplications?
"EXACT_PRODUCT_BITS=256". Multiplying two 128-bit numbers yields 256 bits. If we use 256-bit exact product, overflow is not an issue for intermediate products, but scaling down (shifting right by 96) and rounding is.
Wait, why 2^111? 111 + 96 = 207 bits. Maybe it's just a constraint, not an error.

Look at VECTOR_FIELD_ELL:
VECTOR_FIELD_ELL = x - y - (w+zs)/2 - 1.
Is there an error in math? The notation is just defining a vector field.

Let's check PICARD_IMAGE: X0_PLUS_[0,2^-8]_TIMES_F_OF_B.
This is standard Picard iteration: x_{n+1} = x_0 + \int_{0}^{h} f(x_n(t)) dt, but since it's a rectangle B, it evaluates to x_0 + [0, h] \cdot f(B).

Check CONTRACTION_OBLIGATION:
CEIL_Q96(L_INFINITY/256)_RAW < 2^96
L_INFINITY is the Lipschitz constant of the vector field f in infinity norm.
Lipschitz constant of the Picard operator is h * L_INFINITY.
Here h = 2^-8 = 1/256.
So L_Picard = L_INFINITY / 256.
For strict contraction, L_Picard < 1.
In Q96 fixed point, the raw value of L_Picard is (L_Picard) * 2^96.
The condition is L_Picard_raw < 2^96.
The artifact states: CEIL_Q96(L_INFINITY/256)_RAW < 2^96.
Ceil_Q96 means ceiling to the nearest Q96 representation. Since directed rounding might round up, we need the worst-case bound, so Ceiling.
If Ceil_Q96(L_INFINITY / 256)_RAW < 2^96, this implies Ceil_Q96(L_Picard) < 1.
This correctly checks L_Picard < 1 in exact math up to the ceiling rounding.

Wait, what if L_INFINITY is a Q96 number?
If L_INFINITY is represented as an integer L_INF_RAW, then L_INFINITY = L_INF_RAW / 2^96.
Then L_INFINITY / 256 = L_INF_RAW / (2^96 * 256).
So in Q96, the raw value of L_INFINITY / 256 is L_INF_RAW / 256.
Ceil_Q96(L_INFINITY/256)_RAW is CEIL(L_INF_RAW / 256).
So CEIL(L_INF_RAW / 256) < 2^96.
This correctly computes the ceiling of the Q96 representation of L_Picard.

Is there any other math?
"INFINITY_NORM_ROW_SUM_BOUND_OF_JACOBIAN_ON_B"
Jacobian of the vector field:
f1 = 2y^2 - xy  => df1/dx = -y, df1/dy = 4y - x, df1/dw = 0, df1/dl = 0
f2 = xy - y(w+zs)/2 => df2/dx = y, df2/dy = x - (w+zs)/2, df2/dw = -y/2, df2/dl = 0
f3 = xy - w - zs => df3/dx = y, df3/dy = x, df3/dw = -1, df3/dl = 0
f4 = x - y - (w+zs)/2 - 1 => df4/dx = 1, df4/dy = -1, df4/dw = -1/2, df4/dl = 0
The infinity norm of the Jacobian is the maximum absolute row sum.
Row 1 sum: |y| + |4y-x|
Row 2 sum: |y| + |x - (w+zs)/2| + |y/2|
Row 3 sum: |y| + |x| + 1
Row 4 sum: 1 + 1 + 1/2 = 2.5
All these bound the partial derivatives. The formula INFINITY_NORM_ROW_SUM_BOUND_OF_JACOBIAN_ON_B is mathematically correct for bounding the Lipschitz constant of the vector field.

Wait, the prompt says "Check derivations symbolically; flag every leap."
Let's re-read carefully.
Is MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111 mathematically consistent?
128 bits = 1 sign + 31 int + 96 frac.
If it's signed Q31.96, the range of integer part is [-2^31, 2^31 - 1].
The max absolute value in raw is 2^127.
Wait. Why 2^111?
Could it be related to 256-bit product?
If two numbers are ~2^127, their product is ~2^254, which fits in 256 bits.
What if the schema restricts the inputs to 2^111? Then their product is 2^222, which fits in 256 bits with plenty of room for sums of products (e.g. in Jacobian evaluation).
Maybe it's just a safe threshold. Not a mathematical error.

Let's check the vector field again:
VECTOR_FIELD_X=2*y*y-x*y
VECTOR_FIELD_Y=x*y-y*(w+zs)/2
VECTOR_FIELD_W=x*y-w-zs
VECTOR_FIELD_ELL=x-y-(w+zs)/2-1
Is it missing something? zs is a constant (22.3274637391).
This is standard. No leaps.

LIPSCHITZ_NORM=INFINITY_NORM_ROW_SUM_BOUND_OF_JACOBIAN_ON_B
CEIL_Q96(L_INFINITY/256)_RAW_IS_STRICTLY_LESS_THAN_2^96
Is it exactly correct?
Let L_INF_Q96_RAW be the ceiling of the infinity norm of the Jacobian evaluated on B, in Q96.
Then the exact Lipschitz constant of the Picard step (which is h = 2^-8) is L_P = L_INF * 2^-8.
The exact raw value in Q96 of L_P is L_INF_Q96_RAW / 256.
Because L_P = L_INF * 2^-8 = (L_INF_Q96_RAW / 2^96) * 2^-8 = L_INF_Q96_RAW / (2^96 * 2^8).
So the Q96 raw value is L_INF_Q96_RAW / 256.
To ensure it's strictly less than 1, we do CEIL(L_INF_Q96_RAW / 256) < 2^96.
This is perfectly correct.

Wait, is there an error in the mapping of Q formats?
Q96 means 96 fractional bits.
Value = Raw / 2^96.
If Value < 1, then Raw < 2^96.
This is exactly what is written: `_RAW_IS_STRICTLY_LESS_THAN_2^96`.
This is correct.

Let's consider the time interval: [0, 2^-8].
Integral of f over time is bounded by h * f(B).
h = 2^-8 = 1/256.
The Picard image is X0 + [0, 2^-8] * F(B).
This is the standard interval Picard operator.

Wait, looking at the fields:
STATUS_REVERSED_INTERVAL=-1
STATUS_OUTSIDE_FROZEN_DOMAIN=-2
STATUS_SELF_MAP_FAILURE=-4
No math, just enums.

Are there any WRONG claims?
Let's check LIPSCHITZ_NORM.
Lipschitz constant of vector field is max of row sums of absolute values of Jacobian.
Picard operator is x0 + integral_0^h f(x(t)) dt.
Since it's an autonomous system (t doesn't explicitly appear in f), the Picard operator is x0 + h * f(x).
Wait! The schema says:
VECTOR_FIELD_X=2*y*y-x*y
There is no time dependence. So it is autonomous.
Picard operator is P(x) = x0 + \int_0^h f(x) dt. Since f is independent of t, P(x) = x0 + h f(x).
The derivative is dP/dx = h * df/dx.
So the Lipschitz constant of P is h * L_f.
L_P = 2^-8 * L_f.
Is there any leap here? No, this is exact.

Wait, the Picard operator is usually x0 + h * f(x). The schema says:
PICARD_IMAGE=X0_PLUS_[0,2^-8]_TIMES_F_OF_B
This is exactly X0 + h * F(B).
Everything is standard interval arithmetic.

Let's carefully check:
TOTAL_BITS=128
FRACTION_BITS=96
EXACT_PRODUCT_BITS=256
MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111
If two 128-bit numbers are multiplied, the max exact product is 256 bits. The schema specifies EXACT_PRODUCT_BITS=256. This is correct: 128 + 128 = 256.
What about MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111?
Is it true that if A, B are strictly less than 2^111, then their product is strictly less than 2^222?
Yes, which fits easily in 256 bits.
Why 2^111?
Maybe to leave headroom for 4-corner multiplication which might sum up to 4 products?
Wait, in interval arithmetic, to compute the upper bound of a sum of products, we might accumulate.
But EXACT_PRODUCT_BITS=256 just describes the single product of two 128-bit numbers. 128*128=256. This is true.
Is there any contradiction? No.

Wait! Look at DIVISION=EXACT_POSITIVE_INTEGER_TWO_WITH_FLOOR_CEIL_DIRECTED_ROUNDING
Division by 2 is exact for integers, except for the last bit.
For floor/ceil rounding, dividing by 2 means shifting right by 1 and conditionally adding 1 for the ceiling.
The schema says `DIVISION=EXACT_POSITIVE_INTEGER_TWO_WITH_FLOOR_CEIL_DIRECTED_ROUNDING`.
Is division by 2 "EXACT"? For odd integers, floor(x/2) or ceil(x/2) are not exactly x/2.
But the schema calls it "EXACT_POSITIVE_INTEGER_TWO_WITH_FLOOR_CEIL_DIRECTED_ROUNDING". This means the operation is division by the exact positive integer 2, with floor/ceil directed rounding.
This is just nomenclature. Not a mathematical error.

Is there any mathematical claim that is WRONG or OVERREACH?
Let's look at `ZS_EXACT_DECIMAL=22.3274637391`
Can 22.3274637391 be represented exactly in Q96? No.
But the artifact says ZS_EXACT_DECIMAL, not ZS_EXACT_DYADIC.
If ZS is used in calculations, it will be rounded to dyadic. The artifact doesn't specify the dyadic value of zs, but it is used in exact rational bounds elsewhere?
"INDEPENDENT_VERIFIER=PYTHON_FRACTIONS_WITHOUT_IMPORTING_GENERATOR_OR_HLS_SOURCE"
The verifier uses Python Fractions (exact rationals). It will use 22.3274637391 = 223274637391 / 10000000000.
This is an exact rational.
But the hardware uses FLOOR_CEIL_DIRECTED_ROUNDING for dyadic intervals. So the hardware will use a dyadic interval containing ZS.
Wait, "CANDIDATE_BOX_CONSTRUCTION=EXACT_RATIONAL_MONOTONE_HULL_ITERATION_TO_FIXED_POINT_THEN_SYMMETRIC_2^64_RAW_INFLATION"
This implies the box B is found using exact rational arithmetic (which can represent 22.3274637391 exactly), and then the hardware performs the checks using dyadic intervals (FOUR_CORNER_MIN_MAX_WITH_FLOOR_CEIL).
This is perfectly sound. Exact rational is a superset of dyadic, so rounding exact rationals to dyadic floors and ceilings safely encloses the exact rationals.

Wait, the prompt asks to flag EVERY leap.
"Check derivations symbolically; flag every leap."

Could `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111` be a leap?
Why 2^111? A 128-bit signed Q-format integer has max positive value 2^127 - 1.
The artifact doesn't explain why inputs are bounded by 2^111.
Maybe it's an arbitrary hardware constraint, not a math claim. "MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111" is just a bound.

Wait! Look at `LIPSCHITZ_NORM=INFINITY_NORM_ROW_SUM_BOUND_OF_JACOBIAN_ON_B`
Is L_infinity norm of the Jacobian on B always a valid Lipschitz constant for the vector field on B?
Yes, by the Mean Value Theorem. For any x1, x2 in B,
||f(x1) - f(x2)|| <= sup_{x in B} ||J(x)|| * ||x1 - x2||.
Since B is a box (convex), the line segment between x1 and x2 is in B. So the Mean Value Theorem applies. This is a standard, completely rigorous bound.

Wait! Let's check the Jacobian bounds again.
VECTOR_FIELD_Y=x*y-y*(w+zs)/2
d/dy of Y is x - (w+zs)/2. This is correct.
VECTOR_FIELD_W=x*y-w-zs
d/dw of W is -1. This is correct.

Is `CEIL_Q96(L_INFINITY/256)_RAW_IS_STRICTLY_LESS_THAN_2^96` correct?
L_INFINITY is the infinity norm of the Jacobian.
It is evaluated as a Q96 value.
Let L_inf be the maximum row sum in exact math.
In the hardware, it's computed with directed rounding. The schema doesn't explicitly say how the Jacobian is evaluated in hardware, but `INFINITY_NORM_ROW_SUM_BOUND_OF_JACOBIAN_ON_B` is just the concept.
If L_inf_raw is the Q96 raw value of L_inf.
The contraction factor is h * L_inf = (1/256) * L_inf.
The Q96 raw value of the contraction factor is L_inf_raw / 256.
Since L_inf_raw / 256 might not be an integer, we take CEIL(L_inf_raw / 256).
This must be < 2^96 to be strictly less than 1.0.
Is this correct?
Yes! If CEIL(L_inf_raw / 256) < 2^96, then L_inf_raw / 256 < 2^96, so L_inf_raw < 2^105.
L_inf_raw / 2^96 < 2^9.
Wait! L_inf_raw / 2^96 = L_inf.
So L_inf < 512.
This implies h * L_inf < 1.
Since 1 in Q96 is 2^96, `< 2^96` correctly checks `< 1`.

Is there any overreach?
How does the hardware compute the Jacobian row sums?
The Jacobian involves evaluating expressions like `y`, `x - (w+zs)/2`, etc.
If evaluated with intervals, we get an interval for each partial derivative.
The maximum absolute value of the interval is used for the row sum.
This is sound. The leap might be omitting the explicit rounding mode for the Jacobian evaluation.
But the schema says `INTERVAL_MULTIPLICATION=FOUR_CORNER_MIN_MAX_WITH_FLOOR_CEIL_DIRECTED_ROUNDING`, which applies to all operations.

Wait, what if the Picard image is computed using Euler's method?
`PICARD_IMAGE=X0_PLUS_[0,2^-8]_TIMES_F_OF_B`
This is exactly one step of the Picard iteration, evaluating the integral as `h * F(B)`.
Because the system is autonomous, `\int_0^h f(x) dt = h * f(x)`. This is mathematically exact for a fixed x, and for an interval B, `\int_0^h F(B) dt = h * F(B)`.
There is NO overreach here.

Let's look at `STATUS_ACCEPTED=1`, etc. No math.

Let's look at `EXACT_PRODUCT_BITS=256`.
Is 128 * 128 = 256?
Yes, in two's complement, signed multiplication of 128 bits yields a 255-bit or 256-bit result. To be safe, 256 bits is exact.
But wait!
What if we multiply two 128-bit numbers and then add them?
"INFINITY_NORM_ROW_SUM_BOUND_OF_JACOBIAN_ON_B"
The Jacobian row sum involves ADDITION of partial derivatives.
If we compute partial derivatives, we might have terms like x - (w+zs)/2.
The products inside the partial derivatives are just variable * variable, or variable * constant.
Do we need 256 + log2(N) bits for the sum?
The schema says `EXACT_PRODUCT_BITS=256`. It doesn't say `EXACT_SUM_BITS=256`.
In hardware, addition is usually done in the accumulator. If the accumulator overflows, it's a bug.
But this is an artifact schema, not the hardware source code itself. The artifact claims `EXACT_PRODUCT_BITS=256`. This is an accurate description of the multiplier width.

Let's rethink:
Are there ANY mathematical errors in the artifact?
"DIVISION=EXACT_POSITIVE_INTEGER_TWO_WITH_FLOOR_CEIL_DIRECTED_ROUNDING"
"EXACT" and "FLOOR_CEIL_DIRECTED_ROUNDING" are contradictory terms.
If it's exact, there's no rounding. If it uses floor/ceil directed rounding, it is NOT exact.
Ah!
In fixed-point arithmetic, dividing by 2 (shifting right by 1) loses the least significant bit for odd numbers.
If the operation is directed rounding (floor/ceil), then it is NOT exact.
The artifact calls it "EXACT_POSITIVE_INTEGER_TWO_WITH_FLOOR_CEIL_DIRECTED_ROUNDING".
This is a logical contradiction in terms.
Wait, "DIVISION=EXACT_POSITIVE_INTEGER_TWO_WITH_FLOOR_CEIL_DIRECTED_ROUNDING"
Could "EXACT_POSITIVE_INTEGER_TWO" mean that the divisor (2) is an exact positive integer?
Yes! The divisor is the exact positive integer 2. The division of intervals by 2 is performed with floor/ceil directed rounding.
"Division by the exact positive integer 2, with floor/ceil directed rounding"
It's a noun phrase. "DIVISION = [by an] EXACT_POSITIVE_INTEGER_TWO [performed] WITH_FLOOR_CEIL_DIRECTED_ROUNDING".
This is not a contradiction. It means the divisor is exactly 2.

Wait, is there any error in:
`VECTOR_FIELD_X=2*y*y-x*y`
This is exact.

Let's look at `TIME_INTERVAL=[0,2^-8]`.
This is a closed interval. Length is 2^-8.
If `x0` is the initial condition, Picard step is `x0 + [0, 2^-8] * F(x)`. This is correct.

Wait, let's look at the Lebesgue measure of TIME_INTERVAL.
[0, 2^-8] has length 2^-8.
Is there any issue with multiplying an interval [0, a] with [b, c]?
If F(B) is an interval [F_min, F_max], then [0, h] * F(B) = [min(0, h*F_max), max(0, h*F_max)].
This is standard interval arithmetic.

Let's check the Q96 bounds:
128 bits total, 96 fractional bits. This leaves 32 bits for the integer part.
Since it's SIGNED_S1_I31_F96, it has 1 sign bit, 31 integer bits.
The maximum representable value is 2^31 - 2^-96.
ZS_EXACT_DECIMAL=22.3274637391. This easily fits in 31 integer bits.
MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111.
In Q96, raw 2^111 corresponds to a value of 2^111 / 2^96 = 2^15 = 32768.
So the inputs are bounded to have absolute values strictly less than 32768.
Since 32768 fits well within the 31 integer bits (which can go up to 2^31 - 1), this is mathematically consistent and avoids overflow when computing products (e.g., x*y).
If inputs are up to 2^15, their products are up to 2^30. Then multiplied by 2 (in 2*y*y) gives up to 2^31.
This fits in 31 integer bits!
Wait! `2 * y * y`.
If y < 32768 (which is 2^15), then y*y < 2^30.
Then 2 * y * y < 2^31.
But the max integer value for signed Q31.96 is 2^31 - 2^-96.
So 2^31 is STRICTLY GREATER than the maximum representable positive value!
Wait, let's verify this.
If y is strictly less than 2^111 raw.
In Q96, 2^111 raw is 2^15 = 32768.
So the maximum raw value is 2^111 - 1.
Then y*y is at most (2^111 - 1)^2 = 2^222 - 2*2^111 + 1.
When we scale this back to Q96, we divide by 2^96.
The raw value of the product is (y_raw * y_raw) >> 96.
Max value is approx 2^222 / 2^96 = 2^126 raw.
Wait. If we keep the product in Q96, the raw value is y_raw * y_raw / 2^96.
If y_raw < 2^111, then y_raw * y_raw < 2^222.
Then the Q96 raw value is < 2^222 / 2^96 = 2^126.
But wait, we have `2 * y * y`. The multiplier 2 makes it 2 * 2^126 = 2^127 raw.
The maximum positive signed 128-bit integer is 2^127 - 1.
So 2 * y * y would overflow the 128-bit signed integer representation if we use the full bound 2^111!
Let's re-evaluate carefully.
MAX_ABS_INPUT_RAW_EXCLUSIVE = 2^111.
This means inputs can be up to 2^111 - 1.
In Q96, `2 * y * y` is computed.
Does the computation overflow 128 bits?
The exact product is `2 * y_raw * y_raw`.
The max exact product is `2 * (2^111 - 1)^2` ≈ `2^223`.
The schema specifies EXACT_PRODUCT_BITS=256. 2^223 easily fits in 256 bits.
So the EXACT product does NOT overflow.
But when we convert it back to Q96 (which requires shifting right by 96 and rounding), the maximum Q96 value is `(2^223) >> 96 = 2^127`.
A Q96 value of 2^127 requires 128 bits (including sign bit? No, 2^127 is strictly greater than 2^127 - 1).
Wait! The maximum value of a signed 128-bit integer is 2^127 - 1.
If the Q96 raw value can be up to 2^127, it will OVERFLOW the signed 128-bit Q96 representation!
Is this a mathematical flaw in the schema's bound?
Let's check the exact wording: "MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111".
Is it possible that y is small enough that 2*y*y doesn't reach 2^127?
The schema says "MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111". This is a global bound for ALL inputs.
If the input `y` is 2^111 - 1, then `2 * y * y` will evaluate to 2^223 - ...
Divided by 2^96, the raw value is slightly less than 2^127.
Wait, if it's strictly less than 2^127, and the max signed 128-bit integer is 2^127 - 1, then it CAN overflow!
Let's calculate:
Max y_raw = 2^111 - 1.
Max 2 * y_raw * y_raw = 2 * (2^111 - 1)^2 = 2^223 - 2^113 + 2.
Convert to Q96: divide by 2^96.
Max Q96 raw = 2^127 - 2^17 + 2^-95. (Ignoring rounding for a moment).
Since we use ceiling directed rounding for the maximum bound, the ceiling might add up to 1.
So the maximum Q96 raw value for `2*y*y` can be `2^127 - 2^17 + 1`.
This is strictly less than `2^127 - 1`.
So it DOES NOT overflow the signed 128-bit representation!
Let's double check.
2 * (2^111 - 1)^2 = 2 * (2^222 - 2^112 + 1) = 2^223 - 2^113 + 2.
Divided by 2^96: 2^127 - 2^17 + 2 / 2^96.
Since 2 / 2^96 is strictly less than 1, the value is 2^127 - 2^17 + fraction.
Ceiling of this is 2^127 - 2^17 + 1.
Wait! The max signed 128-bit integer is 2^127 - 1.
Is 2^127 - 2^17 + 1 less than or equal to 2^127 - 1?
2^127 - 131072 + 1 = 2^127 - 131071.
This is indeed less than 2^127 - 1.
So it does NOT overflow.
What if the input is exactly 2^111? The bound says EXCLUSIVE=2^111. So the input cannot be 2^111.
What if we consider `x*y`?
Max x_raw = 2^111 - 1. Max y_raw = 2^111 - 1.
Max x*y raw exact product = 2^222 - 2^112 + 1.
Convert to Q96: 2^126 - 2^16 + 1/2^96.
Ceiling is 2^126 - 2^16 + 1.
This is well below 2^127 - 1.
So the arithmetic does not overflow!

Wait, what about `w`? `VECTOR_FIELD_W = x*y - w - zs`.
x*y can be 2^126 - 2^16 + 1.
w can be negative, up to -(2^111 - 1) (which in raw is about -2^111).
zs is a constant. ZS_EXACT_DECIMAL=22.3274637391.
In Q96, 22.327... is about 22 * 2^96 ≈ 2^4 * 2^96 = 2^100.
So x*y - w - zs can be around 2^126 + 2^111 + 2^100.
This is still way below 2^127 - 1.
Let's find the absolute maximum of any expression.
Max of 2*y*y is about 2^127 - 2^17.
Max of x*y is about 2^126.
So everything fits! This means MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111 was deliberately and precisely chosen to ensure that the exact Q96 result of any product in the vector field does not overflow 128 bits (up to 2^127 - 1).
This is an extremely tight and mathematically sound bound.

Let's check the Lipschitz norm again.
Jacobian partial derivatives:
df1/dx = -y
df1/dy = 4y - x
df2/dx = y
df2/dy = x - (w+zs)/2
df3/dx = y
df3/dy = x
df4/dx = 1
df4/dy = -1
These are linear expressions.
Max absolute value of y is ~2^111 raw (2^15 value).
So max row sum for row 1 is |y| + |4y - x|.
Max y = 2^15. Max 4y = 2^17.
So 4y - x can be 2^17 + 2^15 = 5 * 2^15.
So row 1 sum can be up to 6 * 2^15 = 192 * 2^12 = 768.
This is small.
Then we multiply by h = 1/256 = 2^-8.
Max contraction factor raw = 768 * 2^96 / 256 = 3 * 2^96.
This is less than 2^127.
So the contraction factor evaluation also doesn't overflow!
Everything fits perfectly in 128 bits.
Is `INFINITY_NORM_ROW_SUM_BOUND_OF_JACOBIAN_ON_B` mathematically correct?
Yes, the sup of the infinity norm of the Jacobian over a convex set B is a valid Lipschitz constant for the function on B.

Wait, look at `DIVISION=EXACT_POSITIVE_INTEGER_TWO_WITH_FLOOR_CEIL_DIRECTED_ROUNDING`.
Is there a mathematical error in the fact that it divides by 2 using FLOOR_CEIL?
For an interval [a, b], we want to compute [a, b] / 2.
Since 2 > 0, [a/2, b/2] with floor for a/2 and ceil for b/2.
If a is negative, e.g., a = -3. floor(-3/2) = -2.
If a is positive, e.g., a = 3. floor(3/2) = 1.
This gives a valid lower bound.
Ceil(b/2) gives a valid upper bound.
Is division by 2 mathematically correct this way?
Yes, floor and ceiling preserve the inclusion property for positive divisors.
Since the divisor is exactly 2 (which is > 0), the operations are correct.
Wait, what if we have `y * (w + zs) / 2`?
This is evaluated as a product of intervals first, then divided by 2.
The product interval is [min, max].
Divided by 2 with floor/ceil preserves the interval bounds.
This is mathematically correct.

Let's look at `CANDIDATE_BOX_CONSTRUCTION=EXACT_RATIONAL_MONOTONE_HULL_ITERATION_TO_FIXED_POINT_THEN_SYMMETRIC_2^64_RAW_INFLATION`.
This is an algorithm to find the box B.
It finds a fixed point using exact rational arithmetic, then inflates it by 2^64 raw.
Since 2^64 raw is 2^64 / 2^96 = 2^-32 in value.
This just adds a small epsilon to the bounds.
Is this mathematically sound? Yes.

Are there any leaps in the Lean 4 or other math?
There is no Lean 4 here, just a hardware contract.

Wait! Look at `VECTOR_FIELD_ELL=x-y-(w+zs)/2-1`.
The name is `ELL` (L).
Is there any issue with the variables?
x, y, w, L. The system has 4 dimensions.
Let's check the partial derivatives again:
f1 = 2y^2 - xy
f2 = xy - y(w+zs)/2
f3 = xy - w - zs
f4 = x - y - (w+zs)/2 - 1
Is this vector field related to some known equations?
Doesn't matter. The math is internally consistent.

What about `CEIL_Q96(L_INFINITY/256)_RAW_IS_STRICTLY_LESS_THAN_2^96`?
Let L_INF be the Lipschitz constant in exact math.
The hardware computes an interval bound of the Jacobian.
It takes the maximum absolute values of the intervals.
It sums them up to get the row sums.
It takes the maximum row sum.
Let this be L_bound.
Because of directed rounding (floor/ceil), the hardware interval encloses the exact Jacobian interval.
So the max row sum it computes, say L_hw, is an upper bound on the exact L_INF.
L_hw_raw is the Q96 representation of L_hw.
Then it computes L_P = L_hw / 256.
Since 256 = 2^8, L_P_raw = L_hw_raw / 2^8.
Since we want to strictly check if L_P < 1, we check if L_P_raw < 2^96.
Because of the division by 2^8, L_P_raw might not be an integer, so we take the ceiling.
We check if CEIL(L_P_raw) < 2^96.
Is this correct?
If CEIL(L_P_raw) < 2^96, then L_P_raw < 2^96.
This implies L_hw_raw / 256 < 2^96.
This implies L_hw < 1.
But L_hw is the Lipschitz constant of the Picard operator.
For the Picard operator to be a contraction, we need L_hw < 1.
So this check is mathematically EXACT and CORRECT.

Is there any subtle bug?
Wait. If L_P_raw < 2^96, then L_P < 1.
What if the hardware computes L_hw_raw, and L_hw_raw is EXACTLY 256 * 2^96?
Then CEIL(L_hw_raw / 256) = 2^96.
This would fail the strict check `< 2^96`.
This is correct, because L_P = 1, which is not a strict contraction.

What if L_P_raw is very close to 2^96?
The check is correct.

Could there be an OVERREACH in `LIPSCHITZ_NORM=INFINITY_NORM_ROW_SUM_BOUND_OF_JACOBIAN_ON_B`?
Is the Infinity Norm Row Sum Bound of the Jacobian ALWAYS a valid Lipschitz constant?
Yes, for a continuously differentiable function on a convex set, the operator norm of the derivative bounds the Lipschitz constant.
Specifically, for any x, y in B:
||f(x) - f(y)||_inf = max_i |f_i(x) - f_i(y)|
By Mean Value Theorem, f_i(x) - f_i(y) = J_i(z) * (x - y) for some z on the segment between x and y.
Since B is a box, it is convex, so z is in B.
|f_i(x) - f_i(y)| <= ||J_i(z)||_1 * ||x - y||_inf.
Taking max over i, we get ||f(x) - f(y)||_inf <= max_i sup_{z in B} ||J_i(z)||_1 * ||x - y||_inf.
This is exactly the infinity norm of the
