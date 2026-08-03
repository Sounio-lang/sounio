The user wants me to review the provided artifact for mathematical, algebraic, or statistical claims.
The artifact is a configuration/contract schema (key-value pairs) related to a hardware/software validation setup.

I need to scan the text for any mathematical, statistical, or algebraic claims.

Key claims to check:
1. `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`
2. `ARITHMETIC=SIGNED_Q31_96_DYADIC_ENDPOINTS` -> Total bits = 128, fraction bits = 96, so integer bits = 128 - 96 - 1 (sign bit) = 31. This matches "SIGNED_Q31_96".
3. `ALLOWED_POSITIVE_DIVISORS=2,3,6,41` -> 6 = 2 * 3. So divisors are 2, 3, 2*3, 41.
4. `ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`
5. `ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION`
6. `TOTAL_BITS=128`, `FRACTION_BITS=96`, `PRODUCT_BITS=256`. 128 * 128 = 256. Wait, two 128-bit integers multiplied yield a 256-bit product. But wait, SIGNED Q31.96 means 128-bit total. If we multiply two 128-bit numbers, the product can be 256 bits. If we are dealing with fractions (denominator 2^96), a Q31.96 number squared has 62 integer bits and 192 fractional bits, fitting in 256 bits total. Yes, 256 bits is exactly the size needed to store the product of two 128-bit integers without overflow.
7. Are there any contradictions in the math? "SIGNED_Q31_96" implies 1 sign bit, 31 integer bits, 96 fractional bits. Max absolute value exclusive: 31 bits of integer means the integer part can go up to 2^31 - 1. Wait. If it's signed Q31.96, the format is usually 1 sign bit, 31 integer bits, 96 fractional bits. The max value would be (2^31 - 1) + (1 - 2^-96) $\approx 2^{31}$. Wait, `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`. Ah! "RAW" means the integer representation. The raw integer is 128-bit (SIGNED). A signed 128-bit integer's maximum absolute value is 2^127. But the contract says `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`. Let's see why 2^111.
If integer bits = 31, the maximum value of the represented number is less than 2^31.
In Q31.96, the scaling factor is 2^-96.
So the raw integer value is (value) * 2^96.
Max raw integer = 2^31 * 2^96 = 2^127.
Why does it say `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`?
If 2^111 is the max raw integer, then the represented value max is 2^111 / 2^96 = 2^15 = 32768. But the format is Q31.96, meaning it's supposed to support up to 2^31! Why restrict the raw input to 2^111? Ah! If we do multiplication, the product of two raw 2^127 values is 2^254, which fits in 256 bits. But if the `PRODUCT_BITS=256`, then 2^127 * 2^127 = 2^254. This leaves only 2 bits for sign, which is exactly enough for signed 128-bit multiplication (product of two signed 128-bit ints fits in 256-bit). Wait. If `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`, then product max is 2^222, which also fits in 256 bits. Maybe 111 is derived from something else? Let me check if 111 is mathematically inconsistent.
If `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`, wait, 2^111 raw means it only uses 112 bits (plus sign bit = 113 bits). But the type is 128-bit. Why 111? Wait, if it's SIGNED, maybe 128 bits. If it's strictly Q31.96, max raw is 2^127. Why restrict to 2^111? Maybe to prevent overflow in a subsequent operation not listed, or maybe 2^111 is arbitrary? Wait, 2^111 might be related to 2^111. Wait! "SIGNED_Q31_96" means 1 sign bit, 31 int bits, 96 frac bits. But wait, 1 + 31 + 96 = 128. The max raw value for a positive number is `(2^31 - 1) * 2^96 + (2^96 - 1)`, which is strictly less than `2^127`. If it's SIGNED, max abs raw value is `2^127` (for -2^31 exactly, it is `-2^127`). But the contract says `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`. If this is true, the domain is restricted, but is there a mathematical leap? It's a domain restriction, so it's an assumption, not necessarily an error.

Wait, are there any other mathematical facts to check?
`ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION`
If we are dividing dyadic rationals, say $a = A / 2^{96}$ and $b = B / 2^{96}$, then $a / b = (A / 2^{96}) / (B / 2^{96}) = A / B$.
So dividing two Q31.96 numbers is exactly dividing their raw integer representations $A$ and $B$.
The result should be scaled by $2^{96}$ to be in Q31.96 format.
To get the lower bound of $(A/B) \times 2^{96}$, we compute $\lfloor (A \times 2^{96}) / B \rfloor$.
But the contract says `ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION`.
If this means `FLOOR(A / B)`, that gives the integer part of the quotient, not the Q31.96 value! Wait.
If the contract literally means division of the raw integers gives the Q31.96 quotient, that's WRONG.
Let's double check.
Let $A$ and $B$ be raw integers. Value is $v_A = A / 2^{96}$, $v_B = B / 2^{96}$.
$v_A / v_B = (A / 2^{96}) / (B / 2^{96}) = A / B$.
We want the result as a Q31.96 raw integer, let's call it $C$, such that $v_C = C / 2^{96}$.
So $C / 2^{96} \le A / B \implies C \le (A \times 2^{96}) / B$.
Therefore, `ROUNDING_DIV_POSITIVE_LOWER` must be `FLOOR(A * 2^96 / B)`, NOT `FLOOR(A / B)`.
The contract says `FLOOR_EXACT_INTEGER_DIVISION`. This would compute `FLOOR(A / B)`, which is off by a factor of $2^{96}$!
Let me check if "EXACT_INTEGER_DIVISION" could mean $A$ is pre-scaled? But the contract just says `FLOOR_EXACT_INTEGER_DIVISION`. This is a strong candidate for WRONG.

Let's check `ALLOWED_POSITIVE_DIVISORS=2,3,6,41`.
Is there a mathematical property of 2, 3, 6, 41? 41 is a prime. 6 is 2*3.
Is it related to `2^96`? No.

Let's check `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`.
Why 111? 128 - 17 = 111. 96 + 15 = 111.
Why 15?
If you multiply two numbers, product is $2^{222}$. This easily fits in 256 bits.
Maybe 111 is just an arbitrary bound for a frozen domain to ensure no overflow in multiple operations? It says `FROZEN_NO_OVERFLOW_DOMAIN`.

Let's check the division issue again.
`ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION`
If I have Q31.96 numbers, they are integers divided by $2^{96}$.
Let $X = x \cdot 2^{96}$, $Y = y \cdot 2^{96}$.
$X / Y = (x \cdot 2^{96}) / (y \cdot 2^{96}) = x / y$.
This is just the ratio of the values.
But the hardware is supposed to output a Q31.96 number.
The output should be $Z = z \cdot 2^{96}$.
We want $Z_{lower} \le X/Y \implies z_{lower} \cdot 2^{96} \le x/y \implies z_{lower} \le x/y \cdot 2^{96} = (x \cdot 2^{96}) / y$.
So the division is $(x \cdot 2^{96}) / y$, NOT $x / y$.
If `ROUNDING_DIV_POSITIVE_LOWER` means `FLOOR_EXACT_INTEGER_DIVISION`, i.e., $\lfloor x / y \rfloor$, then the result is missing the $2^{96}$ scaling factor.
Wait, maybe it means `FLOOR( (x * 2^96) / y )` and it is abbreviated?
"EXACT_INTEGER_DIVISION" implies dividing one integer by another, i.e., $x/y$.

Wait, the prompt says: "Verify the math in the supplied artifact. Check derivations symbolically; flag every leap."

What about `SIGNED_Q31_96_DYADIC_ENDPOINTS`?
`TOTAL_BITS=128`
`FRACTION_BITS=96`
`PRODUCT_BITS=256`
If we multiply two Q31.96 numbers:
$X = x / 2^{96}$, $Y = y / 2^{96}$.
Product is $(x \cdot y) / 2^{192}$.
To represent this in Q31.96, we need to divide by $2^{96}$.
So the raw integer product is $x \cdot y$.
The exact Q31.96 value is $(x \cdot y) / 2^{96}$.
So `ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96` perfectly matches this!
$\lfloor (x \cdot y) / 2^{96} \rfloor$.
This confirms that for multiplication, the scaling by $2^{96}$ is explicitly stated.
For division: $X / Y = x / y$.
The Q31.96 raw value should be $X / Y \times 2^{96} = x / y \times 2^{96} = (x \cdot 2^{96}) / y$.
So the lower bound should be `FLOOR(X_TIMES_2_POW_96 / Y)`.
But the contract says `ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION`.
This directly omits the $\times 2^{96}$ scaling!
Therefore, `ROUNDING_DIV_POSITIVE_LOWER` is mathematically WRONG. It loses 96 bits of precision.
If they compute `FLOOR_EXACT_INTEGER_DIVISION`, it evaluates to $\lfloor x / y \rfloor$, which is an integer, not a Q31.96 fixed-point number representing the quotient. Wait, if it is interpreted as the raw integer $z$, then the represented value is $z / 2^{96}$, which is $(x/y) / 2^{96}$. This is an error of $2^{96}$!
Minimal correction: `ROUNDING_DIV_POSITIVE_LOWER=FLOOR_DIVIDEND_TIMES_2_POW_96_BY_DIVISOR`.

Let me double check the division.
If I have 1.0 / 2.0.
In Q31.96:
$X = 2^{96}$
$Y = 2 \cdot 2^{96}$
Exact quotient is 0.5.
In Q31.96, 0.5 is $2^{95}$.
Using the contract's formula: `FLOOR_EXACT_INTEGER_DIVISION` -> $\lfloor 2^{96} / (2 \cdot 2^{96}) \rfloor = \lfloor 1/2 \rfloor = 0$.
The represented value of raw 0 is 0.0. But the exact quotient is 0.5!
This is a massive error (completely truncates all fractional bits, or rather shifts everything right by 96 bits).
Wait! Is it possible the contract assumes division is applied to values already scaled? No, the input is `SIGNED_Q31_96_DYADIC_ENDPOINTS`, meaning inputs are scaled by $2^{96}$.
So the division step must multiply the dividend by $2^{96}$ before dividing.
This is clearly a mathematical error in the specification.

Let me review the `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`.
Is there an error here?
SIGNED Q31.96 means 1 sign bit, 31 integer bits, 96 fractional bits. Total = 128 bits.
The maximum raw value for a positive number is `(2^31 - 1) * 2^96 + (2^96 - 1) = 2^127 - 1`.
So the maximum absolute value of a Q31.96 raw integer is exactly `2^127`.
If the contract says `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`, it means the inputs are strictly bounded within a Q15.96 domain! (15 + 96 = 111).
Is this a contradiction?
It says `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`. It does not contradict the format itself; it just means the values are guaranteed not to exceed `2^111`. This restricts the domain.
But is there any claim that implies a contradiction? No, restricting the domain to avoid overflow is standard.
However, is `2^111` a typo for something?
If we divide two Q31.96 numbers, say $A$ and $B$, $A$ max is $2^{127}$, $B$ min is $1$.
The quotient $A/B \times 2^{96}$ can be up to $2^{127} / 1 \times 2^{96} = 2^{223}$.
This easily fits in 256 bits.
So `2^111` might just be a safe bound for a specific physics/ODE domain.
Since `MAX_ABS_INPUT_RAW_EXCLUSIVE` is explicitly an `ASSUMPTION` (via `ASSUMPTIONS=TWOS_COMPLEMENT_AP_INT_SEMANTICS_AND_FROZEN_NO_OVERFLOW_DOMAIN`), this is just an input constraint.

Wait, is there any other math claim?
`ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`
`ROUNDING_MUL_UPPER=CEIL_EXACT_PRODUCT_DIV_2_POW_96`
Correct for Q31.96 multiplication.

`ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION`
`ROUNDING_DIV_POSITIVE_UPPER=CEIL_EXACT_INTEGER_DIVISION`
As shown above, this is mathematically WRONG for fixed-point arithmetic.
Minimal correction: `FLOOR_DIVIDEND_TIMES_2_POW_96_BY_DIVISOR` and `CEIL_DIVIDEND_TIMES_2_POW_96_BY_DIVISOR`.

Wait, if I look closely at `ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION`, what if the dividend is already multiplied by $2^{96}$?
The contract says:
`ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION`
If this is an operation on `SIGNED_Q31_96_DYADIC_ENDPOINTS`, it takes two Q31.96 numbers. The raw integers are $A$ and $B$.
The operation should return a Q31.96 number $C$ such that $C \approx A/B$.
Since $A$ and $B$ represent $a$ and $b$, $C$ represents $c = a/b$.
$C \le a/b \implies C \le (A/2^{96}) / (B/2^{96}) = A/B$.
So $C \le A/B \implies c = C/2^{96} \le A / (B \cdot 2^{96})$.
But we want $c \approx a/b = A/B$.
So we need $C/2^{96} \approx A/B \implies C \approx (A/B) \cdot 2^{96} = (A \cdot 2^{96}) / B$.
So `FLOOR_EXACT_INTEGER_DIVISION` computes $\lfloor A/B \rfloor$.
If $C = \lfloor A/B \rfloor$, then $c = \lfloor A/B \rfloor / 2^{96}$.
But $a/b = A/B$.
So $c$ is $2^{96}$ times smaller than $a/b$!
The error is a factor of $2^{96}$.

Let's consider another interpretation: maybe "DIVISION" here means dividing the raw integer $A$ by a scalar, like the divisors in `ALLOWED_POSITIVE_DIVISORS=2,3,6,41`?
Ah!!!
`ALLOWED_POSITIVE_DIVISORS=2,3,6,41`
If the divisors are exact scalars (integers), not Q31.96 numbers, then dividing $A$ by $2$ (an integer) is indeed $A / 2$.
Wait, if the divisors are raw integers $D \in \{2, 3, 6, 41\}$, and we want to divide the Q31.96 number $A$ by the integer $D$:
The true value is $a / D$.
In Q31.96, the raw value $C$ should satisfy $C/2^{96} \approx a / D = (A/2^{96}) / D = A / (D \cdot 2^{96})$.
So $C \approx A / D$.
Then `FLOOR_EXACT_INTEGER_DIVISION` computes $\lfloor A/D \rfloor$.
This gives exactly $C = \lfloor A/D \rfloor$.
This is PERFECTLY CORRECT!
Wow.
Let's verify this.
If you divide a fixed-point number by a scalar integer, you just divide the raw integer by the scalar.
For example, 0.5 in Q1.1 is raw 1.
Divide by 2. Raw 1 / 2 = 0.
Lower bound is 0. Value is 0.0. (Exact is 0.25, raw 0).
Upper bound is 1. Value is 0.5. (Raw 1).
So dividing the raw integer by the scalar divisor and taking FLOOR/CEIL is EXACTLY correct for scalar division.
But the contract says `ROUNDING_DIV_POSITIVE_LOWER`. It doesn't say "scalar".
However, `ALLOWED_POSITIVE_DIVISORS=2,3,6,41` is a field in the contract!
Why would `ALLOWED_POSITIVE_DIVISORS` be specified if the division is meant to be a general fixed-point division?
If the division is general fixed-point by fixed-point, then divisors can be any value.
The existence of `ALLOWED_POSITIVE_DIVISORS=2,3,6,41` strongly implies that the division operation is DIVISION BY A POSITIVE INTEGER SCALAR from that allowed set.
Let me check the other interpretations.
If the division is by a Q31.96 fixed point number, e.g., 2.0, its raw value is $2 \cdot 2^{96}$.
If the allowed divisors are scalars 2, 3, 6, 41.
If you divide $A$ by $2$, it's just integer division.
If this is the case, then `ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION` is CORRECT.
Is it though? "DIVISION" could mean division by a Q31.96 divisor. If the divisors were Q31.96, they would have raw values $2 \cdot 2^{96}$, $3 \cdot 2^{96}$, etc.
But `ALLOWED_POSITIVE_DIVISORS=2,3,6,41` are given as small integers. It is standard in hardware to have a fixed set of scalar divisors to avoid a full divider. A full divider is very expensive, whereas dividing by constants (2, 3, 6, 41) can be done via multiply-add or bit shifts, or a small integer divider.
But let's look at `ROUNDING_DIV_POSITIVE_UPPER=CEIL_EXACT_INTEGER_DIVISION`.
If $C_{upper} = \lceil A/D \rceil$, then the represented value is $\lceil A/D \rceil / 2^{96}$.
Since $A/D = a/D$, the represented value is $\lceil a \cdot 2^{96} / D \rceil / 2^{96} \ge a/D$.
This is mathematically sound for interval containment of division by a scalar.
So if "DIVISION" means "division by an allowed scalar", it is CORRECT.
If it means "division by a Q31.96 number", it is WRONG.
Is there any ambiguity? "ROUNDING_DIV_POSITIVE_LOWER". It doesn't specify "scalar".
But the presence of `ALLOWED_POSITIVE_DIVISORS=2,3,6,41` implies that division is restricted to these constants.
Let's write down the ambiguity, or assume it means division by a Q31.96.
Wait, if `ALLOWED_POSITIVE_DIVISORS` means the raw Q31.96 divisor must be one of those values?
No, a raw Q31.96 divisor of 2 represents $2 / 2^{96} \approx 0$. If you divide by that, you are multiplying by $2^{95}$, which would easily overflow `2^111` unless the dividend is 0.
If `ALLOWED_POSITIVE_DIVISORS` refers to the *true value* of the divisor, i.e., 2.0, 3.0, 6.0, 41.0.
Then the raw divisor is $2 \cdot 2^{96}$.
If we divide $A$ by $2 \cdot 2^{96}$, we want the result $C$ such that $C/2^{96} \approx A / (2 \cdot 2^{96})$.
Then $C \approx A / 2$.
This exactly matches integer division $A / 2$.
So whether "2" means the scalar integer 2, or the Q31.96 value 2.0, the operation `FLOOR(A / D)` or `FLOOR(A / (D * 2^96) * 2^96) = FLOOR(A / D)` is mathematically CORRECT!
Oh! I see!
If we divide Q31.96 $X$ by Q31.96 $Y$, $X = A / 2^{96}$, $Y = D \cdot 2^{96} / 2^{96} = D$.
Then $X / Y = (A / 2^{96}) / D = A / (D \cdot 2^{96})$.
To get the Q31.96 raw result $C$, we multiply by $2^{96}$:
$C = (A / (D \cdot 2^{96})) \cdot 2^{96} = A / D$.
So if the divisor is the *exact integer* $D$ (represented in Q31.96 as $D \cdot 2^{96}$), then the raw quotient is exactly $A / D$.
And taking the floor and ceil gives exactly the required bounds for the Q31.96 result!
Therefore, `ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION` is CORRECT, given that `ALLOWED_POSITIVE_DIVISORS` means the Q31.96 divisors are exact integer constants!
Wait, is it?
Let's re-verify.
True dividend: $x = A \cdot 2^{-96}$
True divisor: $d = D \cdot 2^{-96}$  (since $D \ge 2$ is the raw value? No, if the raw value is $D$, then the true value is $D \cdot 2^{-96} \ll 1$.
If the true divisor is the integer $D$, then its Q31.96 raw value is $D \cdot 2^{96}$.
Then $x / d = (A \cdot 2^{-96}) / (D \cdot 2^{96} \cdot 2^{-96}) = A / (D \cdot 2^{96})$.
Result raw value: $C = (x/d) \cdot 2^{96} = A / D$.
Yes! This is algebraically perfectly sound.
If the divisors are exact integers (e.g., raw $2 \cdot 2^{96}$), the hardware divides the raw dividend $A$ by the integer $D$.
Then `FLOOR_EXACT_INTEGER_DIVISION` is exactly the right operation.
So this claim is `[OK]`.

Let's check the `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111` again.
Could it be `[TIGHTENABLE]` or `[OVERREACH]`?
The `ASSUMPTIONS` says `TWOS_COMPLEMENT_AP_INT_SEMANTICS_AND_FROZEN_NO_OVERFLOW_DOMAIN`.
`SIGNED_Q31_96_DYADIC_ENDPOINTS` -> max value is $2^{127}$.
If `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`, it leaves 16 bits of headroom.
Why 16 bits?
Maybe for additions?
If we add $2^{111}$ to $2^{111}$, we get $2^{112}$. Still fits.
Maybe for division by the smallest allowed divisor, which is 2?
$2^{111} / 2 = 2^{110}$. Fits.
What if we divide by the raw value $2 \cdot 2^{96}$?
If the divisor is a Q31.96 value whose true value is $2$, its raw value is $2 \cdot 2^{96}$.
Then the true quotient is $x / 2$.
In raw, $C = A / 2$. So it shrinks.
What if we divide by a small Q31.96 value?
If allowed divisors are $2, 3, 6, 41$, maybe these are the *true values*.
What if we multiply?
$X \times Y$. Max raw input is $2^{111}$.
Product is $2^{222}$.
This is well within `PRODUCT_BITS=256`. (Max for 256 bits is $2^{255}-1$).
So there is plenty of room for multiplication.
Why 111?
Maybe it's not a leap, just a tight domain bound. `[OK]`.

Wait, look at `POSITIVE_WITNESS=ALL_FROZEN_ADVERSARIAL_OPERATIONS_CONTAIN_EXACT_RATIONAL_IMAGES_AND_MATCH_HLS_CSIM_BIT_FOR_BIT`
`INTENT_PRESERVED=NUMERICAL_VALUE_ARITHMETIC_ERROR_AND_SCIENTIFIC_UNCERTAINTY_REMAIN_DISTINCT`
`CLAIMS_INTRODUCED=BOUNDED_DYADIC_OPERATION_CONTAINMENT_IF_ALL_GATES_PASS`
There are no obvious mathematical errors in the algebraic operations, given the division is by exact scalars.

Wait! Let's look at `ALLOWED_POSITIVE_DIVISORS=2,3,6,41`.
Is 6 necessary to list if 2 and 3 are listed? Not a math error, just redundancy.
Is 41 related to something? 41 is a prime.
Are there any mathematical properties required?
`SIGNED_Q31_96_DYADIC_ENDPOINTS`
`TOTAL_BITS=128`
`FRACTION_BITS=96`
`PRODUCT_BITS=256`
Wait. Q31.96 means 1 sign bit, 31 integer bits, 96 fractional bits.
Total bits = 1 + 31 + 96 = 128. Correct.
Product of two 128-bit numbers is 256-bit. Correct.
Max input exclusive is $2^{111}$. This restricts the raw integer to 112 bits (plus sign bit = 113 bits).
Are there any claims that contradict?

What about `ROUNDING_ADD=EXACT_WITHIN_FROZEN_DOMAIN`
`ROUNDING_SUB=EXACT_WITHIN_FROZEN_DOMAIN`
Addition and subtraction of dyadic rationals with the same denominator $2^{96}$ is always exact, provided no overflow.
Since the domain is frozen to prevent overflow (`FROZEN_NO_OVERFLOW_DOMAIN`), this is correct.

What about `ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`
Let's check this.
Two Q31.96 numbers: $X = x \cdot 2^{-96}$, $Y = y \cdot 2^{-96}$.
Product is $P = X \times Y = (x \cdot y) \cdot 2^{-192}$.
We want to represent $P$ in Q31.96, which means we want $C \cdot 2^{-96} \le P \implies C \le (x \cdot y) \cdot 2^{-96}$.
So $C_{lower} = \lfloor (x \cdot y) / 2^{96} \rfloor$.
The contract says: `ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`.
This exactly matches $\lfloor (x \cdot y) / 2^{96} \rfloor$. Correct.

What about `ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION`
Let's check this again carefully.
We want to divide $X$ by $Y$.
If $Y$ is a general Q31.96 number, we showed that $C_{lower} = \lfloor (x \cdot 2^{96}) / y \rfloor$.
But the contract says `FLOOR_EXACT_INTEGER_DIVISION`.
If the contract implies division by an arbitrary Q31.96 number, this is a massive `[WRONG]` error.
But what if `ALLOWED_POSITIVE_DIVISORS=2,3,6,41` means the divisor $Y$ is one of those exact integers?
If $Y = 2$, then its Q31.96 raw representation $y = 2 \cdot 2^{96}$.
Then $X / Y = x \cdot 2^{-96} / 2 = x \cdot 2^{-97}$.
The Q31.96 raw result $C$ should satisfy $C \cdot 2^{-96} \le x \cdot 2^{-97} \implies C \le x / 2$.
Then $C_{lower} = \lfloor x / 2 \rfloor$.
The contract says `FLOOR_EXACT_INTEGER_DIVISION`.
Does "DIVISION" mean dividing the raw dividend by the raw divisor?
If $x$ is the raw dividend, and $y$ is the raw divisor, then `FLOOR(x / y)`.
If $y = 2 \cdot 2^{96}$, then $\lfloor x / (2 \cdot 2^{96}) \rfloor$.
But we just showed $C_{lower}$ should be $\lfloor x / 2 \rfloor$.
If we compute $\lfloor x / y \rfloor$ where $y = 2 \cdot 2^{96}$, we get $\lfloor x / 2^{97} \rfloor$, which is WRONG!
Wait. If the `ALLOWED_POSITIVE_DIVISORS` refers to the Q31.96 value (e.g. 2.0), its raw value is $2 \cdot 2^{96}$.
If "INTEGER DIVISION" means `FLOOR( dividend_raw / divisor_raw )`, then `FLOOR( x / (2 * 2^96) )` is computed.
But the correct raw Q31.96 quotient is $x / 2$.
These differ by a factor of $2^{96}$!
Ah. What if `ALLOWED_POSITIVE_DIVISORS=2,3,6,41` refers to the *raw* values of the divisors?
If $y = 2$ (raw value is 2), then the true value of the divisor is $2 \cdot 2^{-96}$.
Then $X / Y = (x \cdot 2^{-96}) / (2 \cdot 2^{-96}) = x / 2$.
The Q31.96 raw result $C$ should be $C \cdot 2^{-96} \approx x/2 \implies C \approx (x/2) \cdot 2^{96} = x \cdot 2^{95}$.
But `FLOOR_EXACT_INTEGER_DIVISION` of $x$ by $2$ is $\lfloor x / 2 \rfloor$.
This is also wrong by a factor of $2^{96}$!

Wait, is there ANY interpretation where `FLOOR_EXACT_INTEGER_DIVISION` is the correct Q31.96 result?
Only if the dividend is in a different format, or if it's a division of the *values* and the output format is an integer (not Q31.96).
But `ARITHMETIC=SIGNED_Q31_96_DYADIC_ENDPOINTS`. The output is supposed to be a Q31.96 endpoint.
Let's re-read carefully: `ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION`.
If it's dividing $X$ by $Y$ to produce $Z$.
If $Z = \lfloor X_{raw} / Y_{raw} \rfloor$.
Then $Z$ represents $Z \cdot 2^{-96} = \lfloor x/y \rfloor \cdot 2^{-96}$.
But the true quotient $X/Y = (x \cdot 2^{-96}) / (y \cdot 2^{-96}) = x/y$.
So $Z \cdot 2^{-96}$ is missing a factor of $2^{96}$ compared to the true quotient!
Let me write this down clearly.
True dividend $= x \cdot 2^{-96}$.
True divisor $= y \cdot 2^{-96}$.
True quotient $= x / y$.
Represented quotient $= Z \cdot 2^{-96}$.
We want $Z \cdot 2^{-96} \le x / y \implies Z \le (x / y) \cdot 2^{96} = (x \cdot 2^{96}) / y$.
So $Z_{lower} = \lfloor (x \cdot 2^{96}) / y \rfloor$.
But the contract says `FLOOR_EXACT_INTEGER_DIVISION`, which in a hardware context typically means integer division of the two operands: `x / y`.
If the hardware computes `x / y`, then it outputs $Z = \lfloor x / y \rfloor$.
This is mathematically WRONG by a factor of $2^{96}$!
Is there any other interpretation?
What if `DIVISION` means division by a scalar integer $N \in \{2, 3, 6, 41\}$?
If true divisor $= N$.
Then represented divisor is $N \cdot 2^{96}$.
But what if the operation is $X / N$?
True quotient $= (x \cdot 2^{-96}) / N = x / (N \cdot 2^{96})$.
Represented quotient $= Z \cdot 2^{-96}$.
We want $Z \cdot 2^{-96} \le x / (N \cdot 2^{96}) \implies Z \le x / N$.
So $Z_{lower} = \lfloor x / N \rfloor$.
In this case, `FLOOR_EXACT_INTEGER_DIVISION` of $x$ by $N$ is EXACTLY CORRECT!
Let's check the wording: `ROUNDING_DIV_POSITIVE_LOWER=FLOOR_EXACT_INTEGER_DIVISION`.
Notice it doesn't say `FLOOR_DIVIDEND_BY_DIVISOR`. It just says `FLOOR_EXACT_INTEGER_DIVISION`.
If the divisor is one of the `ALLOWED_POSITIVE_DIVISORS` (e.g., 2, 3, 6, 41), these are small integers.
If the division is a generic Q31.96 by Q31.96 division, the divisors would be anything.
The presence of `ALLOWED_POSITIVE_DIVISORS=2,3,6,41` strongly suggests the hardware supports division ONLY by these constants.
Dividing a Q31.96 number by a scalar constant $N$ is exactly equivalent to dividing its raw representation by $N$.
So `FLOOR_EXACT_INTEGER_DIVISION` is perfectly correct!
Let's consider if the divisor is a Q31.96 number. Then `ALLOWED_POSITIVE_DIVISORS=2,3,6,41` would mean the *true values* of the divisors are 2.0, 3.0, 6.0, 41.0.
If true value is 2.0, raw divisor is $2 \cdot 2^{96}$.
Then hardware divides $x$ by $2 \cdot 2^{96}$.
But wait, if the hardware does `FLOOR_EXACT_INTEGER_DIVISION`, does it divide $x$ by the raw divisor $2 \cdot 2^{96}$?
If so, $Z = \lfloor x / (2 \cdot 2^{96}) \rfloor$.
But we established that if divisor is 2.0, we need
