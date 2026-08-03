The user wants me to act as a mathematical referee and review the provided artifact for mathematical content.
Artifact is a schema/contract document: `sounio.cs6.u250-validated-dyadic-contract.v1`.
Let's check the contents for mathematical claims, derivations, proofs, statistical methodology, etc.

Fields:
- ARITHMETIC=SIGNED_S1_I31_F96_DYADIC_ENDPOINTS
- Q_NOTATION_CONVENTION=ONE_SEPARATE_SIGN_BIT_PLUS_31_INTEGER_BITS_PLUS_96_FRACTION_BITS
- TOTAL_BITS=128
- FRACTION_BITS=96
- PRODUCT_BITS=256
- ROUNDING_ADD=EXACT_WITHIN_FROZEN_DOMAIN
- ROUNDING_SUB=EXACT_WITHIN_FROZEN_DOMAIN
- ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96
- ROUNDING_MUL_UPPER=CEIL_EXACT_PRODUCT_DIV_2_POW_96
- DIVISION_SEMANTICS=Q31_96_VALUE_DIVIDED_BY_EXACT_POSITIVE_DIMENSIONLESS_INTEGER_D
- DIVISION_IS_NOT=GENERAL_Q31_96_BY_Q31_96_DIVISION
- ROUNDING_DIV_POSITIVE_LOWER_RAW=FLOOR(A_RAW/D)
- ROUNDING_DIV_POSITIVE_UPPER_RAW=CEIL(A_RAW/D)
- MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111

Let's check the math claims:
1. 1 sign bit + 31 integer bits + 96 fraction bits = 128 bits.
   1 + 31 + 96 = 128. This matches TOTAL_BITS=128. (OK)

2. Q31_96 format: usually this means 1 sign bit, 31 integer bits, 96 fraction bits.
   But wait, 31 integer bits? A signed 128-bit integer with 1 sign bit has 127 magnitude bits. If 96 are fraction bits, the integer bits are 127 - 96 = 31 bits. This matches.

3. ROUNDING_MUL_LOWER/UPPER: Product of two Q31.96 numbers.
   If A and B are in Q31.96, A = A_raw / 2^96, B = B_raw / 2^96.
   A * B = (A_raw * B_raw) / 2^192.
   To get the result back in Q31.96 (which is A_raw * B_raw / 2^192 * 2^96 = A_raw * B_raw / 2^96).
   PRODUCT_BITS: A_raw and B_raw are 128-bit signed, so their product is 256 bits.
   ROUNDING_MUL_LOWER = FLOOR_EXACT_PRODUCT_DIV_2_POW_96. This is exactly floor(A_raw * B_raw / 2^96). (OK)
   ROUNDING_MUL_UPPER = CEIL_EXACT_PRODUCT_DIV_2_POW_96. This is ceil(A_raw * B_raw / 2^96). (OK)
   Wait, interval arithmetic for multiplication of signed numbers:
   If intervals can contain positive and negative numbers, simply taking floor and ceil of the exact product isn't enough to find the min and max of the interval bounds. You need four-corner enumeration (multiply all combinations of bounds).
   However, the contract says ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96, ROUNDING_MUL_UPPER=CEIL_EXACT_PRODUCT_DIV_2_POW_96.
   Wait! The domain is `TWOS_COMPLEMENT_AP_INT_SEMANTICS_AND_FROZEN_NO_OVERFLOW_DOMAIN`.
   But the artifact says: `ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`, `ROUNDING_MUL_UPPER=CEIL_EXACT_PRODUCT_DIV_2_POW_96`.
   If we are multiplying two dyadic intervals [a, b] and [c, d], the exact product endpoints depend on the signs of the bounds. You cannot just compute floor(a*c / 2^96) and ceil(b*d / 2^96) unless it is guaranteed that all numbers in the intervals are positive, or the operation is just a scalar multiplication (not interval).
   Wait, the artifact says: `FLOOR_EXACT_PRODUCT_DIV_2_POW_96`. It doesn't specify interval operand bounds, it specifies how to round a *single* exact product. In interval arithmetic, `ROUNDING_MUL_LOWER` means the rounding mode to get the lower bound of the product of two exact dyadic numbers, but to get the lower bound of two *intervals*, you must take the min of the 4 corners. If it defines `ROUNDING_MUL_LOWER` as "how to round an exact product downwards", it is technically correct for a single multiplication. But if it implies interval multiplication, it is missing the 4-corner step.
   Let's see: `TRANSFORMATION=EXACT_DYADIC_ENDPOINT_ARITHMETIC_WITH_OUTWARD_ROUNDING_AND_FAIL_CLOSED_CERTIFICATE_TRANSCRIPT`. This explicitly implies outward rounding for endpoints. So 4-corner enumeration is needed for interval arithmetic if intervals can cross zero.
   Wait, does the domain forbid crossing zero? `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`. This implies inputs can be negative (since it's MAX_ABS).
   If `a` in [-M, M] and `b` in [-M, M], their product is in [-M^2, M^2].
   Let's re-read: `ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`. This just defines the rounding mode (floor) for the lower bound of the exact product. The 4-corner part is the responsibility of the interval arithmetic implementation. So it's fine.

4. DIVISION_SEMANTICS=Q31_96_VALUE_DIVIDED_BY_EXACT_POSITIVE_DIMENSIONLESS_INTEGER_D
   DIVISION_IS_NOT=GENERAL_Q31_96_BY_Q31_96_DIVISION
   ROUNDING_DIV_POSITIVE_LOWER_RAW=FLOOR(A_RAW/D)
   ROUNDING_DIV_POSITIVE_UPPER_RAW=CEIL(A_RAW/D)
   Wait! If A_RAW can be negative, then A_RAW/D where D > 0.
   If A_RAW < 0, then A_RAW / D is negative.
   Floor of a negative number moves it further negative (which is the lower bound).
   Ceil of a negative number moves it closer to zero (which is the upper bound).
   So FLOOR(A_RAW/D) is the lower bound and CEIL(A_RAW/D) is the upper bound, REGARDLESS of the sign of A_RAW (since D is strictly positive).
   This is mathematically perfectly correct for outward rounding of a scalar divided by a positive integer!
   Let's verify: A_RAW = -5, D = 2. -5/2 = -2.5.
   FLOOR(-2.5) = -3. CEIL(-2.5) = -2.
   Interval is [-3, -2]. The exact value -2.5 is contained. Outward rounding! (OK)

5. MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111.
   Inputs are S1_I31_F96.
   A 128-bit signed integer has range [-2^127, 2^127 - 1].
   If MAX_ABS_INPUT_RAW_EXCLUSIVE = 2^111, then inputs are strictly within (-2^111, 2^111).
   Maximum magnitude of exact product of two inputs: (2^111 - 1)^2 < 2^222.
   PRODUCT_BITS = 256. The exact product of two 128-bit signed integers takes at most 256 bits.
   Specifically, signed product of 2 numbers < 2^111 takes 222 bits. Fits in 256 bits. (OK)

Wait, is there any error in the bit math?
1 + 31 + 96 = 128. (OK)
Max value of S1_I31_F96: integer part is 31 bits + 1 sign bit.
The max positive value is (2^31 - 1) + (1 - 2^-96) < 2^31.
If MAX_ABS_INPUT_RAW_EXCLUSIVE = 2^111.
The raw integer value of 2^31 is 2^31 * 2^96 = 2^127.
Wait! The maximum representable value in S1_I31_F96 is `(2^127 - 1) / 2^96`, which is roughly `2^31 - 2^-96`.
So the max *raw* value is `2^127 - 1`.
But the contract says `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`.
If `MAX_ABS_INPUT_RAW_EXCLUSIVE = 2^111`, then the max *value* is `2^111 / 2^96 = 2^15`.
Why is the max input value restricted to 2^15 (32768)? If the integer part allows 31 bits (which is up to 2^31), why restrict to 2^15?
Ah! "MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111" implies values are bounded to ~2^15.
Is there any problem with this mathematically? It's a domain restriction. 2^111 < 2^127.
So the bound is valid, it just leaves a lot of headroom. This is fine (TIGHTENABLE? maybe, but it's a design choice).

Wait, let's check `ALLOWED_POSITIVE_DIVISORS=2,3,6,41`.
If D=41, FLOOR(A_RAW/41) and CEIL(A_RAW/41). This is exact arithmetic.

Let's check the rounding statements again.
For ADD/SUB:
ROUNDING_ADD=EXACT_WITHIN_FROZEN_DOMAIN
ROUNDING_SUB=EXACT_WITHIN_FROZEN_DOMAIN
Since A and B are bounded by 2^111, their sum is bounded by 2^112. The maximum raw value for 128-bit is ~2^127. So A+B will never overflow the 128-bit representation (it only needs 112 bits). Thus it is EXACT. (OK)

What about MUL?
A and B bounded by 2^111.
Their exact product A_raw * B_raw is bounded by 2^222.
This is shifted right by 96 bits: `(A_raw * B_raw) / 2^96`.
The result raw value is bounded by `2^222 / 2^96 = 2^126`.
Since 2^126 is less than the max raw value 2^127 - 1, the result fits in the 128-bit signed format without overflow!
Therefore, the rounding operations (floor and ceil) will perfectly fit in the 128-bit format, and no secondary overflow can occur. (OK)

This is beautiful. The domain limits inputs to 2^111 so that the product is at most 2^222, and after shifting by 96, it is 2^126, which perfectly fits in the 127 bits of signed magnitude in a 128-bit two's complement integer!
Wait, two's complement signed 128-bit integer range is [-2^127, 2^127 - 1].
Is 2^126 within [-2^127, 2^127 - 1]?
Yes, 2^126 < 2^127 - 1 (for 2^127 > 2).
So the maximum positive result is 2^126, which fits perfectly!
What if the inputs are negative?
Minimum input is -2^111 + 1.
Product is (-2^111 + 1)^2 = 2^222 - 2^112 + 1.
This is positive.
If one is positive and one is negative, the product is negative.
Minimum exact product is `-(2^111 - 1) * (2^111 - 1) = -2^222 + 2^112 - 1`.
Shifted by 96: `-2^126 + ...`.
Is `-2^126` >= `-2^127`?
Yes, -2^126 > -2^127.
So it fits!
Wait! FLOOR_EXACT_PRODUCT_DIV_2_POW_96.
If the exact product is negative, say `-2^126`, Floor does nothing. But what if the exact product is `-2^222 + 1`?
Divided by 2^96: `-2^126 + 2^-96`.
Floor of this is `-2^126 - 1` (in terms of the integer raw value, floor of -2^126 + epsilon is -2^126 - 1 + 2^96... wait, no).
Let's use integer arithmetic: `floor(X / 2^96)` where X is a negative integer.
If `X = -2^222 + 1`, `X / 2^96 = -2^126 + 2^-96`.
Floor of this is `-2^126 - 1`.
Wait, is `-2^126 - 1` representable in signed 128-bit?
Minimum signed 128-bit integer is `-2^127`.
Is `-2^126 - 1 >= -2^127`?
Yes, `-2^126 - 1` is greater than `-2^127`.
So it fits!
What about the upper bound? `ceil(X / 2^96)` where X is negative.
`ceil(-2^222 + 1) / 2^96 = -2^126`.
Which is `>= -2^127`.
What if X is positive? `X = 2^222 - 2^112 + 1`.
`floor(X / 2^96) = 2^126 - 2^20`.
This is positive, and `2^126 - 2^20 <= 2^127 - 1`.
So it fits!
So the claim `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111` is perfectly chosen to ensure no overflow occurs during the 128-bit arithmetic, even with the addition of the +/-1 from the floor/ceil rounding.
Wait, `2^126 + 1` is the max possible value of ceil?
If X is positive, `ceil(X / 2^96) <= 2^126`. Max possible is `2^126`. This is `< 2^127 - 1`.
If X is negative, `floor(X / 2^96)` can be as low as `-2^126 - 1`. This is `> -2^127`.
So `2^111` is indeed a safe bound.
Wait, is it `2^111` or could it be `2^112`?
If MAX_ABS_INPUT_RAW_EXCLUSIVE = 2^112.
Max product = (2^112 - 1)^2 = 2^224 - 2^113 + 1.
Divided by 2^96 = 2^128 - 2^17 + ...
This exceeds 2^127 - 1!
So 2^111 is exactly the largest power of 2 such that `(2^x - 1)^2 / 2^96 < 2^127`.
Let's check x=111: 2^222 / 2^96 = 2^126. Fits.
Let's check x=112: 2^224 / 2^96 = 2^128. Overflows.
So `2^111` is a strict tight bound to prevent overflow.
The math here is perfectly sound.

Let's check `ALLOWED_POSITIVE_DIVISORS=2,3,6,41`.
Wait, `6` is in the divisors. `6 = 2 * 3`. But the raw value is divided by D.
Is there any issue with `DIVISION_SEMANTICS`?
`ROUNDING_DIV_POSITIVE_LOWER_RAW=FLOOR(A_RAW/D)`
`ROUNDING_DIV_POSITIVE_UPPER_RAW=CEIL(A_RAW/D)`
If A_RAW is exact, A_RAW/D is a real number.
Floor moves it down, Ceil moves it up. Since D > 0, this correctly computes the outward rounded interval of the division of a single number.
Wait, if we are dividing an *interval* [A_LOWER, A_UPPER] by D:
The lower bound of the interval division is `FLOOR(A_LOWER/D)`.
The upper bound is `CEIL(A_UPPER/D)`.
This matches exactly if we substitute `A_RAW` with the respective endpoints!
So `FLOOR(A_LOWER/D)` and `CEIL(A_UPPER/D)` is correct.

What about the KAT cases?
EXPECTED_KAT_CASES = 96.
EXPECTED_VALID_CASES = 80.
EXPECTED_REFUSAL_CASES = 16.
80 + 16 = 96.
Are there any other statuses?
STATUS_VALID=1
STATUS_REVERSED_INTERVAL=-1
STATUS_OUTSIDE_FROZEN_DOMAIN=-2
STATUS_UNSUPPORTED_DIVISOR=-3

Wait! What about `STATUS_REVERSED_INTERVAL`? It's a status code, which implies an error condition (lower bound > upper bound).
What about `STATUS_OUTSIDE_FROZEN_DOMAIN=-2`? This matches `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111` rejection.
What about `STATUS_UNSUPPORTED_DIVISOR=-3`? Matches `ALLOWED_POSITIVE_DIVISORS=2,3,6,41` rejection.

Wait, if the artifact has no derivations but only declarative constants, are there any logical leaps?
The artifact specifies the Q format: `S1_I31_F96`. 1 + 31 + 96 = 128 bits.
The product bits = 256.
The fraction bits = 96.
The bounds `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`.
Everything here is a set of parameter definitions and algorithmic specifications for exact dyadic interval arithmetic.
Are there any incorrect claims?
Let's check `ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`.
If the product is exact, say `P = A_raw * B_raw`.
The exact product in the Q31.96 format is `P / 2^96`.
Floor of this is `floor(P / 2^96)`.
Wait, in integer division, `P // 2^96` in Python is floor division. So `FLOOR_EXACT_PRODUCT_DIV_2_POW_96` is correct.
Is `CEIL_EXACT_PRODUCT_DIV_2_POW_96` correct? Yes, `ceil(P / 2^96)`.
Is this mathematically sound for outward rounding of a single product?
If A and B are single dyadic values, the exact product is `P / 2^192`.
To round it *outward* (meaning we want to bound it), we need an interval.
Wait, the format is Q31.96. The exact product of two Q31.96 values is Q62.192.
To convert back to Q31.96, we divide by 2^96 (i.e. shift right by 96).
The true value is exactly representable in Q31.96 if and only if `P % 2^96 == 0`.
If it's not exactly representable, it falls between `floor(P / 2^96)` and `ceil(P / 2^96)` in the Q31.96 raw representation.
So rounding outward to the nearest representable Q31.96 bounds means taking `floor(P / 2^96)` and `ceil(P / 2^96)`.
This is precisely what `ROUNDING_MUL_LOWER` and `ROUNDING_MUL_UPPER` do!
So it is mathematically perfectly sound for scalar outward rounding.

What about interval multiplication?
If we have interval X = [A1, A2] and Y = [B1, B2].
The product is [min(Ai*Bj), max(Ai*Bj)].
To compute the lower bound, we multiply the four combinations of endpoints, take the minimum exact product `P_min`, and then compute `floor(P_min / 2^96)`.
Wait! Does the artifact claim that interval multiplication is *only* `FLOOR_EXACT_PRODUCT_DIV_2_POW_96`?
The artifact defines:
`ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`
`ROUNDING_MUL_UPPER=CEIL_EXACT_PRODUCT_DIV_2_POW_96`
It does NOT explicitly state the 4-corner enumeration for intervals.
It just says `ROUNDING_MUL_LOWER=...`.
However, for intervals, if the operation is an interval operation, the 4-corner step is REQUIRED.
"INTERVAL-EXTENSION ARITHMETIC, FOUR-CORNER ENUMERATION FOR NON-MONOTONE OPS" is explicitly in the scope of my task.
If this artifact implies it is doing interval arithmetic ("NEGATIVE_WITNESS=ANY_ENDPOINT_UNDER_APPROXIMATION...") but does not explicitly mention the four-corner step for multiplication, is that a flaw?
The artifact defines `ROUNDING_MUL_LOWER` and `ROUNDING_MUL_UPPER` as the *rounding rules* to apply to the exact product. It doesn't define how to choose the exact product (e.g. 4 corners).
BUT, wait! Multiplication of two intervals is NOT a monotone operation if 0 is inside the interval.
If the domain is `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111`, then intervals could be `[-2^111, 2^111]`, which includes 0.
So multiplication is definitely non-monotone over the allowed domain.
If the artifact implies that the lower bound of the interval product is obtained by just `FLOOR_EXACT_PRODUCT_DIV_2_POW_96` of the *respective* endpoints, that would be WRONG.
Wait, the artifact states:
`TRANSFORMATION=EXACT_DYADIC_ENDPOINT_ARITHMETIC_WITH_OUTWARD_ROUNDING_AND_FAIL_CLOSED_CERTIFICATE_TRANSCRIPT`
It defines the *operations* as dyadic endpoint arithmetic.
"ENDPOINT ARITHMETIC" is a specific term. In naive endpoint arithmetic, one might just multiply the lower bounds to get the new lower bound, which is WRONG if signs are mixed.
Let's see: `ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`.
It doesn't say `FLOOR(A_LOWER * B_LOWER / 2^96)`. It just says `FLOOR_EXACT_PRODUCT_DIV_2_POW_96`.
This phrasing `FLOOR_EXACT_PRODUCT_DIV_2_POW_96` leaves the `EXACT_PRODUCT` ambiguous. It could mean the exact product of the chosen corner endpoints.
However, is it an OVERREACH or a potential issue? Yes, if "endpoint arithmetic" is taken literally without the 4-corner rule, it's a well-known fallacy in interval arithmetic.
Let me flag it: `OVERREACH` or `TIGHTENABLE`. If the artifact intends to define interval multiplication, the 4-corner rule for determining the `EXACT_PRODUCT` to be floored/ceiled must be explicitly specified. I will flag this as `[TIGHTENABLE]`.

Wait, let me look at the division.
`ROUNDING_DIV_POSITIVE_LOWER_RAW=FLOOR(A_RAW/D)`
`ROUNDING_DIV_POSITIVE_UPPER_RAW=CEIL(A_RAW/D)`
Here it explicitly uses `A_RAW`. This implies a scalar division or an interval where the lower endpoint is divided by D.
For interval division by a strictly positive scalar D:
The operation is monotone.
So `A_LOWER / D <= A <= A_UPPER / D`.
So the lower bound is `FLOOR(A_LOWER / D)` and the upper bound is `CEIL(A_UPPER / D)`.
The math here perfectly works because D is positive and scalar, so the operation is monotone!
Thus, `FLOOR(A_LOWER/D)` and `CEIL(A_UPPER/D)` is correct.

What about division by an interval?
`DIVISION_IS_NOT=GENERAL_Q31_96_BY_Q31_96_DIVISION`
So it only divides by positive scalar integers. Monotone operation, so endpoint arithmetic is sufficient.

But for multiplication, it is interval by interval!
Wait, `MUL` is Q31.96 * Q31.96.
Is it interval by interval?
If so, multiplication is non-monotone.
Does the artifact say it performs interval-by-interval multiplication?
Let's check the schema again.
`TYPES_CHANGED=none`
`INTENT_PRESERVED=NUMERICAL_VALUE_ARITHMETIC_ERROR_AND_SCIENTIFIC_UNCERTAINTY_REMAIN_DISTINCT`
`ASSUMPTIONS=TWOS_COMPLEMENT_AP_INT_SEMANTICS_AND_FROZEN_NO_OVERFLOW_DOMAIN`
`ARITHMETIC=SIGNED_S1_I31_F96_DYADIC_ENDPOINTS`
`ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`
It does not explicitly restrict MUL to strictly positive intervals.
If it allows MUL of mixed-sign intervals, then `EXACT_PRODUCT` must be the minimum/maximum over the 4 corners.
I will flag the omission of the 4-corner selection for `EXACT_PRODUCT` in `ROUNDING_MUL_LOWER`.

Let's double check if there are any other mathematical claims.
`EXPECTED_KAT_CASES=96`
`EXPECTED_VALID_CASES=80`
`EXPECTED_REFUSAL_CASES=16`
80 + 16 = 96.
Are there 4 statuses?
STATUS_VALID=1
STATUS_REVERSED_INTERVAL=-1
STATUS_OUTSIDE_FROZEN_DOMAIN=-2
STATUS_UNSUPPORTED_DIVISOR=-3
Is there any claim about the counts?
No, they are just expected counts for tests.

Let's check `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111` again.
I proved that `2^111` is mathematically the maximum power of 2 for the inputs such that their product, after dividing by `2^96`, fits within `[-2^127, 2^127 - 1]` even with the `+1` adjustment from the floor/ceil.
Let's re-verify:
Max positive raw input: `M = 2^111 - 1`.
Max exact product: `P = M^2 = 2^222 - 2^112 + 1`.
To convert to Q31.96 format, we divide by `2^96`: `P / 2^96`.
Ceil of this: `ceil((2^222 - 2^112 + 1) / 2^96) = ceil(2^126 - 2^16 + 2^-96) = 2^126 - 2^16 + 1`.
Is `2^126 - 2^16 + 1 <= 2^127 - 1`?
Yes, `2^126` is much less than `2^127`.
Wait... wait.
If input is `2^111`, product is `2^222`.
`2^222 / 2^96 = 2^126`.
`2^126` fits in 128-bit signed! (Max is `2^127 - 1`).
Why did I think `2^126` overflows?
Ah! `2^126` does NOT overflow a 128-bit signed integer.
Wait, if `2^126` fits, could we use `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^112`?
If `MAX = 2^112 - 1`.
`P = (2^112 - 1)^2 = 2^224 - 2^113 + 1`.
`P / 2^96 = 2^128 - 2^17 + 2^-96`.
Ceil is `2^128 - 2^17 + 1`.
This is GREATER than `2^127 - 1`. So it overflows.
What if `MAX_ABS_INPUT_RAW_EXCLUSIVE = 2^127`?
If `MAX = 2^127`, `P = 2^254`, `P/2^96 = 2^158`, overflows.
So `2^111` is indeed a safe bound.
But is `2^112` over? Yes, `2^128 - ...` overflows.
Wait, is there a tighter bound?
Let `M` be the bound. `M^2 / 2^96 < 2^127`.
`M^2 < 2^223`.
`M < 2^{111.5} = 2^{111} * \sqrt{2} \approx 1.414 * 2^{111}`.
So `2^111` is the largest power of 2 that works!
So the choice of `2^111` is mathematically optimal among powers of 2.
This is an extremely solid mathematical design.

Let's review the Q_NOTATION_CONVENTION.
`Q_NOTATION_CONVENTION=ONE_SEPARATE_SIGN_BIT_PLUS_31_INTEGER_BITS_PLUS_96_FRACTION_BITS`
Usually, Qm.n format means m integer bits (including sign or not depending on convention) and n fraction bits.
Here, 1 sign + 31 integer + 96 fraction = 128 bits.
So the maximum positive value is `(2^31 - 1).(111...1)_2` which is roughly `2^31`.
So the value is bounded by `2^31`.
The raw integer is `value * 2^96`.
So the max raw integer is `< 2^31 * 2^96 = 2^127`.
This matches the 128-bit signed integer limits `[-2^127, 2^127 - 1]`.

Let's check the ROUNDING_DIV_POSITIVE formulas again.
`ROUNDING_DIV_POSITIVE_LOWER_RAW=FLOOR(A_RAW/D)`
`ROUNDING_DIV_POSITIVE_UPPER_RAW=CEIL(A_RAW/D)`
Assuming D is a positive integer, and A_RAW is a signed integer.
As I proved before, FLOOR and CEIL correctly expand the interval regardless of the sign of A_RAW.
This is perfectly correct.

Are there any logical issues with `STATUS_REVERSED_INTERVAL=-1`?
If an interval is `[A, B]` with `A > B`, this is an error.
The code has a status for this. No math issue.

Let's look at `CLAIMS_INTRODUCED=BOUNDED_DYADIC_OPERATION_CONTAINMENT_IF_ALL_GATES_PASS`.
This implies the claim is that the exact dyadic interval arithmetic correctly bounds the real operations.
To be correct, multiplication must use 4 corners.
Let's flag `[TIGHTENABLE]` on the `ROUNDING_MUL_LOWER` definition.
Claim: `ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`
If applied to intervals without the 4-corner rule, it underapproximates the lower bound when bounds have mixed signs.
Minimal correction: Explicitly state that `EXACT_PRODUCT` refers to the minimum (for LOWER) or maximum (for UPPER) over the 4-corner endpoint combinations, `min(A_lower*B_lower, A_lower*B_upper, A_upper*B_lower, A_upper*B_upper)`.

Let's consider if `FLOOR_EXACT_PRODUCT_DIV_2_POW_96` is just scalar arithmetic.
If `A_RAW` and `B_RAW` are single endpoints, then it is perfectly correct. The artifact might only be defining the rounding rules for scalars, and the interval logic is elsewhere.
But in a "review math in artifact" setting, if an interval arithmetic claim is made (`BOUNDED_DYADIC_OPERATION_CONTAINMENT`), the artifact must be mathematically complete with respect to it. Pointing out the lack of 4-corner rule is a valid `[TIGHTENABLE]`.

Let's check the `DIVISION_SEMANTICS=Q31_96_VALUE_DIVIDED_BY_EXACT_POSITIVE_DIMENSIONLESS_INTEGER_D`.
`DIVISION_IS_NOT=GENERAL_Q31_96_BY_Q31_96_DIVISION`
This explicitly limits division to scalars, thus ensuring monotonicity.
Why doesn't it explicitly limit MUL to scalar or strictly positive intervals?
Because MUL is `ROUNDING_MUL_LOWER=...`. If it's interval MUL, it's non-monotone.
I'll definitely flag this.

Let's review everything.
1. `[OK] 1 + 31 + 96 = 128 bits`
2. `[OK] PRODUCT_BITS = 256` for 128-bit * 128-bit.
3. `[OK] MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111` prevents overflow of the 128-bit signed product (after shifting by 96 bits), since `2^111 * 2^111 / 2^96 = 2^126 < 2^127`.
4. `[OK] FLOOR(A_RAW/D)` and `CEIL(A_RAW/D)` correctly compute outward rounding for division by a strictly positive integer D.
5. `[TIGHTENABLE] ROUNDING_MUL_LOWER=...` - Missing explicit 4-corner enumeration for interval multiplication if intervals cross zero.

Wait, is there any issue with the bitwise definitions?
`ROUNDING_ADD=EXACT_WITHIN_FROZEN_DOMAIN`
Since max input is `2^111`, `A + B` has max value `2^112`.
`2^112 < 2^127`, so it fits perfectly in the 128-bit format. Thus it's exact.
This is mathematically correct.

What about `STATUS_OUTSIDE_FROZEN_DOMAIN`?
If input is `>= 2^111` or `<= -2^111`, it's rejected.
This correctly preserves the invariant needed for MUL and ADD.

What about `ROUNDING_DIV_POSITIVE_LOWER_RAW`?
`FLOOR(A_RAW/D)` is an exact integer division with floor.
In C/C++, if A_RAW is negative, `A_RAW / D` truncates towards zero.
So computing floor requires adjusting: `A_RAW >= 0 ? A_RAW/D : (A_RAW - D + 1)/D`.
But mathematically, `FLOOR` is unambiguous. So the mathematical claim is correct, even if HLS/Csim implementation might need care. I am a mathematical referee, so the math claim is OK.

Are there any other math claims?
`NEGATIVE_WITNESS=ANY_ENDPOINT_UNDER_APPROXIMATION_OVERFLOW_UNDETECTED_MUTATION_OR_PICARD_SELF_MAP_FAILURE`
This is just a string in a schema.

What about `CONCEPT_IDS=PROPOSED_SOUNIO_CS6_PROOF_CARRYING_VALIDATED_HARDWARE`?
Just a string.

`INTENT_PRESERVED=NUMERICAL_VALUE_ARITHMETIC_ERROR_AND_SCIENTIFIC_UNCERTAINTY_REMAIN_DISTINCT`
String.

So the math claims are:
1. Bit allocation sums to 128 bits.
2. Input bounds (2^111) guarantee no overflow in 128-bit signed addition and multiplication (after downshifting by 96).
3. Outward rounding definitions for floor/ceil.

Let's do a quick calculation on the MUL bound again.
Max input = `2^111 - 1`.
Max product of inputs = `(2^111 - 1)^2 = 2^222 - 2^112 + 1`.
When we multiply two Q31.96 numbers, the raw representation is `P = (A * B) / 2^96`.
`P_max = (2^222 - 2^112 + 1) / 2^96 = 2^126 - 2^16 + 2^-96`.
We take the CEIL of this: `ceil(P_max) = 2^126 - 2^16 + 1`.
Max signed 128-bit integer is `2^127 - 1`.
Since `2^126 - 2^16 + 1 < 2^127 - 1`, it fits!
Min input = `-2^111 + 1`.
Min product = `-2^111 + 1` * `2^111 - 1` = `-2^222 + 2^112 - 1`.
Divided by `2^96`: `-2^126 + 2^16 - 2^-96`.
Floor of this: `-2^126 + 2^16 - 1`.
Min signed 128-bit integer is `-2^127`.
Since `-2^126 + 2^16 - 1 > -2^127`, it fits!
So the choice of 2^111 is brilliant. It maximizes the domain without overflowing 128-bit signed integers after multiplication.

Let's structure the output:
```
[OK] Bit allocation
[OK] Product bits
[OK] Domain bounds prevent overflow
[OK] Division outward rounding
[TIGHTENABLE] Multiplication outward rounding (requires 4 corners if intervals can be negative)
```

Is it possible that `MAX_ABS_INPUT_RAW_EXCLUSIVE=2^111` is just `2^111`?
Let's check if `2^111` is `1 << 111`. Yes.

Let's double check if `FLOOR(A_RAW/D)` is the lower bound.
Yes, for `A_LOWER/D`, since `D>0`, the function is monotonically increasing.
So `inf (A/D) = A_LOWER/D`.
The dyadic lower bound is `floor(A_LOWER/D)`, which translates to `FLOOR(A_RAW/D)`.
This is perfectly correct.

Same for `CEIL(A_RAW/D)`.

What about `ROUNDING_MUL_LOWER=FLOOR_EXACT_PRODUCT_DIV_2_POW_96`?
If we have an interval `[A1, A2]` and `[B1, B2]`.
The exact product is `min(A1B1, A1B2, A2B1, A2B2) / 2^96`.
Then we take the floor.
The artifact just says `FLOOR_EXACT_PRODUCT_DIV_2_POW_96`.
If `EXACT_PRODUCT` refers to the scalar product, it's fine. But the claim `BOUNDED_DYADIC_OPERATION_CONTAINMENT` is an interval claim. So it must be the interval product.
Therefore, `[TIGHTENABLE]` is the perfect verdict.

Let's review the format requirement.
```
[OK | WRONG | OVERREACH | TIGHTENABLE] <claim>
  <one-line justification or counter-example>
  <if WRONG: minimal correction>
```
I should list each checked claim with this format.

Example:
```
[OK] Bit allocation S1_I31_F96 totals 128 bits
  1 + 31 + 96 = 128. Total magnitude bits (127) minus fraction bits (96) yields 31 integer bits.

[OK] PRODUCT_BITS = 256
  The product of two 128-bit integers requires at most 256 bits.

[OK] MAX_ABS_INPUT_RAW_EXCLUSIVE = 2^111 guarantees no overflow
  Max product of (2^111 - 1)^2 / 2^96 yields ~2^126, which fits in signed 
