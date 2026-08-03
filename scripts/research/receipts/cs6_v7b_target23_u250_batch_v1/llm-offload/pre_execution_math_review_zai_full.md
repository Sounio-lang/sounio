The user wants me to act as a mathematical referee and verify the math in the supplied artifact.
The artifact contains several files related to an FPGA/HLS implementation of a numerical integration (RK4) for a dynamical system.

Let's look for mathematical claims in the code and documents.

1.  **Q24.40 Fixed Point and Arithmetic:**
    -   `kernel.cpp`, `host.cpp`, `batch_generate.py`.
    -   `FRAC_BITS = 40`
    -   `ONE_Q = 1 << FRAC_BITS`
    -   `STEP_Q = ONE_Q >> 10` -> $2^{40} / 2^{10} = 2^{30}$.
    -   `ZS = 22.3274637391` (double)
    -   `ZS_Q = 24549305999887LL` (int64). Wait, $2^{40} = 1099511627776$.
        $ZS = 22.3274637391$.
        $22.3274637391 \times 1099511627776 \approx 24549305999886.81$.
        So `ZS_Q` is correctly rounded to the nearest integer.
    -   Is `ZS_Q` an `int64`? Yes.
    -   Is `FRAC_BITS` 40? Yes.

2.  **RK4 Integration:**
    -   `rk4` function in `kernel.cpp`:
        -   `k1 = field(state)`
        -   `k2 = field(add_scaled(state, k1, qdiv(step, 2)))`
        -   `k3 = field(add_scaled(state, k2, qdiv(step, 2)))`
        -   `k4 = field(add_scaled(state, k3, step))`
        -   `weighted = k1 + 2*k2 + 2*k3 + k4`
        -   `out = state + qdiv(qmul(weighted, step), 6)`
        This matches the standard RK4 formula: $y_{n+1} = y_n + \frac{1}{6} h (k_1 + 2k_2 + 2k_3 + k_4)$.
        Is the fixed-point scaling correct?
        `qmul(a, b)` computes `(a * b) >> FRAC_BITS`. This assumes `a` and `b` are both in Q24.40. So $a \cdot b / 2^{40}$.
        `step` is $2^{30}$, which represents $2^{-10}$ in Q24.40. This is correct.
        `qdiv(step, 2)` divides the integer `step` by 2. Since `step` is $2^{30}$, `step / 2` is $2^{29}$, which represents $2^{-11}$ in Q24.40. This is correct.
        `qdiv(qmul(weighted, step), 6)`:
        `qmul(weighted, step)` scales Q24.40 * Q24.40 -> Q24.40.
        Then division by 6 is exact integer division of the Q24.40 result by 6. Wait, standard integer division `/ 6` in C++ for `ap_int` truncates toward zero. The python script `batch_generate.py` explicitly says: `SMALL_INTEGER_DIVISION=TRUNCATE_TOWARD_ZERO` and implements `qdiv` using `abs(value) // divisor`.
        So it correctly matches the integer model.

3.  **Dynamical System (`field`):**
    -   `xy = qmul(x, y)`
    -   `half_w_zs = qdiv(w + ZS_Q, 2)`
    -   `out.x = qmul(y, y) * 2 - xy`
    -   `out.y = xy - qmul(y, half_w_zs)`
    -   `out.w = xy - w - ZS_Q`
    -   `out.ell = x - y - half_w_zs - ONE_Q`
    -   Wait, is `qmul(y, y) * 2` correct?
        `qmul(y, y)` returns a Q24.40 number. Multiplying it by 2 (an integer) in C++ or Python just adds 1 to the integer value, effectively adding $2 \times 2^{-40} = 2^{-39}$ to the real value. But it should be $2 \times \text{real value}$, which means adding $2 \times 2^{-40}$ to the raw integer value.
        If `qmul(y, y)` returns $Y \in \mathbb{Z}$ such that $Y \cdot 2^{-40} \approx y^2$.
        Then `Y * 2` will be $2 Y$, which represents $2 Y \cdot 2^{-40} = 2 y^2$. This is exactly correct.

4.  **Event Localization (`localize_event`):**
    -   Finds a $w=0$ crossing.
    -   The description says "NEGATIVE_TO_NONNEGATIVE".
    -   `if (middle_state.w < 0) low = middle; else high = middle;`
    -   So if `w < 0`, it moves the lower bound up. If `w >= 0`, it moves the upper bound down.
    -   This correctly finds the point where `w` transitions from `<0` to `>=0`.

5.  **Determinant Formula (host.cpp):**
    -   `const double q0_area = (UX * SY - SX * UY) * RU * RS;`
    -   Let's check `UX = -0.67430316214199759`, `UY = -0.73845463335624273`
    -   `SX = -0.94170446778164518`, `SY = 0.33644122125579123`
    -   `UX*SY - SX*UY` is the 2D cross product (z-component of 3D cross product), which gives the signed area of the parallelogram spanned by U and S.
    -   Multiplying by `RU * RS` (which are 0.004 and 0.3) gives the area of the bounding box (or rather, scales the parallelogram). This is indeed the area of the initial region spanned by $U \cdot r_u$ and $S \cdot r_s$.
    -   `determinant = std::exp(ell2) * normal0 / normal2 * q0_area;`
    -   `normal0 = x0 * y0 - ZS`
    -   `normal2 = x2 * y2 - ZS`
    -   Wait, is `ell2` exactly the log of the area scaling factor?
    -   This seems to be a specific formula for a Poincaré map determinant. I cannot verify the physical derivation of `det = exp(ell) * n0 / n1 * A0` without knowing the exact system. But mathematically, the code correctly evaluates the formula written.

6.  **Hex/Float Precision in Python Script:**
    -   `math.ldexp(value, -FRAC_BITS)`
    -   `float.fromhex(decimal[key])`
    -   `q0_area = float((UNSTABLE_X * STABLE_Y - STABLE_X * UNSTABLE_Y) * RADIUS_U * RADIUS_S)`
    -   `UNSTABLE_X` etc. are `Fraction`s.
    -   In `initial_xy`:
        -   `u = -RADIUS_U + (Fraction(ui) + Fraction(1, 2)) * 2 * RADIUS_U / (1 << ud)`
        -   `s = -RADIUS_S + (Fraction(si) + Fraction(1, 2)) * 2 * RADIUS_S / (1 << sd)`
        -   `x = ORIGIN_X + UNSTABLE_X * u + STABLE_X * s`
        -   `y = ORIGIN_Y + UNSTABLE_Y * u + STABLE_Y * s`
        -   `return quantize(x), quantize(y)`
    -   `quantize(value: Fraction) -> int`:
        -   `magnitude = abs(value.numerator) // value.denominator`
        -   `return -magnitude if value < 0 else magnitude`
    -   This is correctly truncating toward zero.
    -   Wait, `qdiv` truncates toward zero: `magnitude = abs(value) // divisor; return -magnitude if value < 0 else magnitude`.
    -   `qmul(a, b)` in python: `(a * b) >> FRAC_BITS`.
        Since Python's `>>` operates on arbitrary precision integers, and negative integers shift toward negative infinity (floor division):
        Wait! In Python, `-1 >> 1` is `-1`.
        Let's check `>>` in Python vs C++:
        In C++, `-1 >> 1` is `-1` for signed types (implementation-defined, usually arithmetic shift, so -1).
        What about `-2 >> 1`?
        In Python, `-2 >> 1` is `-1`. (Floor)
        In C++ (ap_int and standard int), arithmetic shift right rounds toward negative infinity (floor).
        Wait! The contract says `ARITHMETIC_SHIFT_40`.
        If `a * b` is negative, e.g., `-3`, `>> 1` is `-2` in Python and C++ (if arithmetic shift).
        BUT wait, `TRUNCATE_TOWARD_ZERO` was for *division* (i.e. `qdiv`).
        The contract says: `SIGNED_MULTIPLY=128_BIT_PRODUCT_THEN_ARITHMETIC_SHIFT_40`.
        An arithmetic shift right by 1 divides by 2 and rounds toward $-\infty$.
        So Python's `>>` is exactly arithmetic shift!
        Let's verify:
        Python: `(-3) >> 1` = `-2`.
        C++ `ap_int<128>`: `-3 >> 1` = `-2`.
        So `qmul` matches.
        Wait! Let's check `qdiv`:
        `def qdiv(value: int, divisor: int) -> int:`
        `    magnitude = abs(value) // divisor`
        `    return -magnitude if value < 0 else magnitude`
        This correctly implements `TRUNCATE_TOWARD_ZERO`.
        In `kernel.cpp`:
        `static q_t qdiv(q_t value, int divisor) { return value / divisor; }`
        In C++, integer division `/` truncates toward zero! This exactly matches the Python implementation.
        So `qdiv` matches.

7.  **Wait, let's double check `qmul` in Python vs C++:**
    In C++:
    `static q_t qmul(q_t a, q_t b) { wide_t product = wide_t(a) * wide_t(b); return q_t(product >> FRAC_BITS); }`
    This uses `ap_int<128>`, and right shift.
    In Python:
    `def qmul(a: int, b: int) -> int: return (a * b) >> FRAC_BITS`
    Both use arithmetic shift. They will match.

8.  **Checking `ap_int` shift behavior:**
    Xilinx `ap_int` `>>` operator performs arithmetic right shift (sign-extending) for signed types.
    Python `>>` also performs arithmetic right shift (floor division).
    So they match perfectly.

9.  **Determinant formula in Python script (`batch_generate.py`):**
    ```python
    x2, y2, ell2 = (qfloat(result[index]) for index in (4, 5, 6))
    determinant = math.exp(ell2) * (qfloat(x0) * qfloat(y0) - float(ZS_TEXT)) / (x2 * y2 - float(ZS_TEXT)) * q0_area
    ```
    Wait, the host.cpp does:
    ```cpp
    double determinant = std::exp(ell2) * normal0 / normal2 * q0_area;
    ```
    This matches.

10. **Is there any mathematical leap?**
    Let's check the cross product:
    ```cpp
    const double q0_area = (UX * SY - SX * UY) * RU * RS;
    ```
    In Python:
    ```python
    q0_area = float((UNSTABLE_X * STABLE_Y - STABLE_X * UNSTABLE_Y) * RADIUS_U * RADIUS_S)
    ```
    This is mathematically sound. It's the determinant of the Jacobian of the linear map $(u, s) \mapsto (x, y)$ at $t=0$, which gives the initial area scaling factor.

    Let's check `event_pass` logic:
    `flags |= 1` if events == 2
    `flags |= 2` if initial_normal > 0
    `flags |= 4` if final_normal > 0
    So `flags == 7` means events == 2, initial_normal > 0, final_normal > 0.
    `determinant < 0` means `normal0 / normal2 * q0_area < 0`.
    Since `initial_normal > 0` and `final_normal > 0`, `normal0 / normal2 > 0`.
    Wait! If `initial_normal > 0` and `final_normal > 0`, then `normal0 / normal2 > 0`.
    For `determinant < 0`, we would need `q0_area < 0`.
    Let's compute `q0_area`:
    `UX * SY - SX * UY`
    UX = -0.6743..., SY = 0.3364... => `UX * SY` = -0.2268...
    SX = -0.9417..., UY = -0.7384... => `SX * UY` = +0.6956...
    `UX * SY - SX * UY` = -0.2268... - 0.6956... = -0.9225...
    Yes, `q0_area < 0`.
    So if `flags == 7` (both normals > 0), then `normal0 / normal2 > 0`.
    Then `normal0 / normal2 * q0_area < 0`.
    And `exp(ell2) > 0` always.
    So if `flags == 7`, the determinant MUST be strictly negative!
    Let's verify this. The host code checks:
    `bool pass = mismatches == 0 && event_pass == n && negative == n && inside == n;`
    Wait, `event_pass` is incremented if `word[1] == 2 && word[7] == 7`.
    And `negative == n` counts if `determinant < 0`.
    Is it mathematically guaranteed that if `word[7] == 7`, `determinant < 0`?
    Yes! As I proved, `normal0 / normal2` is positive, `exp(ell2)` is positive, and `q0_area` is strictly negative, so the product must be strictly negative.
    Wait, what if `normal2 == 0`? Then it's a division by zero.
    But `final_normal > 0` is flag 4, so if flag 4 is set, `final_normal > 0`, so `normal2 > 0`. Thus `normal2 != 0`.

    Is it possible that `determinant == 0`? No, since all factors are non-zero and `normal2 > 0`.
    So `negative == n` is mathematically redundant if `event_pass == n` is checked! (Assuming `q0_area < 0` is a constant).
    Wait, `q0_area` is a constant `double` in the code:
    `static constexpr double q0_area = (UX * SY - SX * UY) * RU * RS;`
    `RU = 0.004`, `RS = 0.3`.
    `q0_area` = (-0.22686 - 0.69572) * 0.012 = -0.92258 * 0.012 = -0.01107.
    It is strictly negative.
    Since `q0_area < 0`, `exp(ell2) > 0`, and `normal0/normal2 > 0`, `determinant < 0` is strictly guaranteed by `flags == 7`.
    So checking `negative == n` in `pass` is logically redundant (though harmless).

11. **Is there any issue with `qdiv` truncation?**
    `def qdiv(value: int, divisor: int) -> int:`
    `    magnitude = abs(value) // divisor`
    `    return -magnitude if value < 0 else magnitude`
    In C++: `return value / divisor;`
    Does `int64_t value / divisor` in C++ truncate toward zero?
    Yes, since C++11, integer division truncates toward zero.
    BUT what if `divisor` is negative?
    `qdiv` is only called with positive divisors: `qdiv(value, 2)`, `qdiv(value, 6)`. So this is completely safe.

12. **What about `qdiv` with `ap_int`?**
    `static q_t qdiv(q_t value, int divisor) { return value / divisor; }`
    In HLS, `ap_int` division with `int` will also truncate toward zero.

13. **Let's check `state.w + ZS_Q`:**
    In `field`:
    `q_t half_w_zs = qdiv(state.w + ZS_Q, 2);`
    This implies the ODE uses `(w + ZS) / 2`.
    Wait! Is `ZS_Q` correctly computed?
    `double ZS = 22.3274637391;`
    `q_t ZS_Q = 24549305999887LL;`
    Let's multiply `22.3274637391` by $2^{40}$.
    $22.3274637391 \times 1099511627776 = 24549305999886.8146496$.
    Rounding to nearest integer gives `24549305999887`.
    This is correct.
    However, in the Python code:
    `ZS = 24549305999887`
    `ZS_TEXT = "22.3274637391"`
    It uses `ZS` for `qdiv(w + ZS, 2)` and `float(ZS_TEXT)` for `float(qfloat(x0) * qfloat(y0) - float(ZS_TEXT))`.
    This is slightly inconsistent: the dynamical system uses `ZS = 24549305999887 / 2^{40}`, but the determinant formula uses `float(ZS_TEXT) = 22.3274637391`.
    Is `24549305999887 / 2^{40}` exactly `22.3274637391`?
    `24549305999887 / 2^{40} = 22.3274637391000072...`
    So the dynamical system actually uses `22.3274637391000072...`, while the determinant formula uses `22.3274637391` (which is `22.3274637390999999...` in double precision).
    Let's check `float(ZS_TEXT)` in Python.
    Wait, `ZS_TEXT = "22.3274637391"`, which evaluates to `22.3274637390999999...` in double precision.
    `ZS_Q` evaluates to `22.3274637391000072...` in double precision.
    This is a minor discrepancy ($10^{-14}$), but the math in the generator explicitly does:
    `determinant = math.exp(ell2) * (qfloat(x0) * qfloat(y0) - float(ZS_TEXT)) / (x2 * y2 - float(ZS_TEXT)) * q0_area`
    while the hardware `host.cpp` does:
    `double determinant = std::exp(ell2) * normal0 / normal2 * q0_area;`
    where `normal0 = x0 * y0 - ZS;` and `ZS = 22.3274637391;`.
    So `host.cpp` also uses `22.3274637391` for the determinant.
    This is a slight inconsistency between the ODE dynamics and the determinant calculation, but both Python and C++ host share this same slight inconsistency. Thus, they will match.

14. **Let's check the area `q0_area` in Python:**
    `q0_area = float((UNSTABLE_X * STABLE_Y - STABLE_X * UNSTABLE_Y) * RADIUS_U * RADIUS_S)`
    In Python, `UNSTABLE_X` etc are `Fraction`s.
    `UNSTABLE_X = Fraction("-0.67430316214199759")`
    So `q0_area` will be evaluated as a `Fraction`, then converted to `float`.
    In `host.cpp`:
    `const double UX = -0.67430316214199759;`
    `q0_area` is evaluated in `double` precision.
    Is `(float(Fraction(x))) * ...` exactly the same as `double` evaluation?
    Not necessarily! `float(Fraction)` rounds the *exact* result to the nearest double.
    `double` evaluation rounds at each step!
    Let's see:
    `UX * SY` evaluated in `double` is one rounding.
    `SX * UY` evaluated in `double` is another rounding.
    `(UX * SY - SX * UY)` evaluated in `double` is a third rounding.
    In Python: `float((UNSTABLE_X * STABLE_Y - STABLE_X * UNSTABLE_Y) * RADIUS_U * RADIUS_S)`
    This computes the exact rational product, and then rounds ONCE to `float`.
    This means `q0_area` in Python might differ from `q0_area` in `host.cpp` by 1 ULP!
    Let's verify this!
    ```python
    a = float(Fraction("-0.67430316214199759") * Fraction("0.33644122125579123") - Fraction("-0.94170446778164518") * Fraction("-0.73845463335624273") * Fraction("0.004") * Fraction("0.3"))
    ```
    vs
    ```cpp
    double q0_area = (-0.67430316214199759 * 0.33644122125579123 - (-0.94170446778164518) * (-0.73845463335624273)) * 0.004 * 0.3;
    ```
    Yes, this is a classic floating-point mismatch!
    If `determinant` depends on `q0_area`, then `determinant` in Python might differ slightly from `determinant` in `host.cpp`.
    BUT wait, the `determinant` is only compared against `max_delta` and interval endpoints `C0HORECT2_LOWER` etc.
    It is NOT compared bitwise with the C++ host. The C++ host computes `delta = std::abs(determinant - decimal_det)` and prints it.
    However, the Python script also computes `delta` and stores it in `reference.tsv`.
    The `host.cpp` reads `DECIMAL_RESULTS` which contains `FINE_DETERMINANT` and the interval endpoints.
    If the Python script and the C++ host compute `determinant` slightly differently (by 1 ULP), the `delta` computed might differ.
    Is this a math claim error? The artifact states it is "bit-exact reference transcript". The *transcript* is the `expected.bin`, which only contains the 8 output words per leaf (steps, events, event1_time, event2_time, x2, y2, ell2, flags).
    The `determinant` is NOT part of the `expected.bin` or the bit-exact check!
    The bit-exact check in `host.cpp` is:
    `for (size_t i = 0; i < expected.size(); ++i) if (actual[i] != expected[i])`
    And `expected.size()` is `inputs.size() / 2 * WORDS_PER_LEAF`. So it only checks the raw fixed-point RK4 outputs.
    The `determinant` is evaluated in floating point in BOTH the Python script and `host.cpp` just for the `ABS_DELTA_VS_DECIMAL` metric and CAPD containment check.
    Since it's not part of the "bit-exact" hardware verification (the `x2, y2, ell2` are the bit-exact parts), a 1 ULP difference in floating-point determinant is just a minor metric inconsistency, not a failure of the math.

15. **Is `q0_area` strictly negative?**
    Yes, `-0.92258 * 0.012 = -0.01107`. It's strictly negative.

16. **Let's check the interval containment logic:**
    `bool in = h0l < determinant && determinant < h0u && r0l < determinant && determinant < r0u;`
    This correctly checks if `determinant` is strictly inside the bounds of both intervals `C0HORECT2` and `C0RECT2`.

17. **Let's check `initial_xy` in Python:**
    ```python
    u = -RADIUS_U + (Fraction(ui) + Fraction(1, 2)) * 2 * RADIUS_U / (1 << ud)
    ```
    This generates points centered in each grid cell.
    For a grid of size $2^{u_d}$, the width is $2 \cdot RADIUS_U$. The cell width is $\frac{2 \cdot RADIUS_U}{2^{u_d}}$.
    The center of cell `ui` (0-indexed) is $-RADIUS_U + (ui + 0.5) \cdot \text{width}$.
    This matches the formula.

18. **Is there any algebraic identity that's wrong?**
    `event_pass` condition:
    `event_pass == n && negative == n`
    I showed that `negative == n` is redundant if `flags == 7` and `q0_area < 0`.
    Wait! Is `negative == n` exactly checking `determinant < 0`?
    ```cpp
    if (determinant < 0) ++negative;
    ```
    Yes.
    So the logic `pass = mismatches == 0 && event_pass == n && negative == n && inside == n;` has a redundant term `negative == n`.
    This is not WRONG, but it is `TIGHTENABLE` (or redundant). But since `event_pass == n` implies `flags == 7` for all leaves, which implies `determinant < 0`, checking `negative == n` is indeed mathematically redundant.

19. **Wait, let's carefully review `flags == 7` implies `determinant < 0`.**
    `flags = 1 (if events==2) | 2 (if initial_normal>0) | 4 (if final_normal>0)`
    If `flags == 7`, then `events == 2`, `initial_normal > 0`, `final_normal > 0`.
    `determinant = std::exp(ell2) * normal0 / normal2 * q0_area;`
    `normal0 = x0 * y0 - ZS`
    `normal2 = x2 * y2 - ZS`
    Since `initial_normal > 0`, `normal0 > 0`.
    Since `final_normal > 0`, `normal2 > 0`.
    `std::exp(ell2)` is strictly positive.
    `q0_area` is strictly negative.
    Therefore, the product of three positive numbers and one strictly negative number is strictly negative.
    Thus, `determinant < 0`.
    This means `negative == n` is a redundant check in `pass`.

20. **Are there any errors in the `event_bisections` loop?**
    ```cpp
    q_t low = 0;
    q_t high = STEP_Q;
    State high_state = rk4(left, high);
    for (int iteration = 0; iteration < EVENT_BISECTIONS; ++iteration) {
        q_t middle = (low + high) >> 1;
        State middle_state = rk4(left, middle);
        if (middle_state.w < 0) {
            low = middle;
        } else {
            high = middle;
            high_state = middle_state;
        }
    }
    event_step = high;
    event_state = high_state;
    ```
    This loop does 24 iterations.
    `STEP_Q` is $2^{30}$.
    24 iterations of bisection on $[0, 2^{30}]$ reduces the interval size to $2^{30-24} = 2^6 = 64$.
    So the precision of the event localization is $64 / 2^{40} = 2^{-34}$.
    Is there an off-by-one? 24 iterations means 24 right shifts.
    $30 - 24 = 6$.
    This is correct.

21. **Wait, look at `middle_state.w < 0` condition.**
    If `w` is negative, it moves `low` up to `middle`.
    So `low` always stays at a state where `w < 0`.
    `high` always stays at a state where `w >= 0`.
    When the loop terminates, `high` is the smallest multiple of 64 (in Q24.40 integer) such that `w >= 0`.
    This perfectly matches the "NEGATIVE_TO_NONNEGATIVE" event orientation.

22. **Is the RK4 formula correct?**
    ```cpp
    State k1 = field(state);
    State k2 = field(add_scaled(state, k1, qdiv(step, 2)));
    State k3 = field(add_scaled(state, k2, qdiv(step, 2)));
    State k4 = field(add_scaled(state, k3, step));
    State weighted;
    weighted.x = k1.x + 2 * k2.x + 2 * k3.x + k4.x;
    ...
    out.x = state.x + qdiv(qmul(weighted.x, step), 6);
    ```
    The standard RK4 is:
    $y_1 = y_0 + \frac{1}{6} h (k_1 + 2k_2 + 2k_3 + k_4)$
    where
    $k_1 = f(y_0)$
    $k_2 = f(y_0 + \frac{h}{2} k_1)$
    $k_3 = f(y_0 + \frac{h}{2} k_2)$
    $k_4 = f(y_0 + h k_3)$
    This code evaluates `weighted.x` as $k_1 + 2k_2 + 2k_3 + k_4$.
    Then computes `qdiv(qmul(weighted.x, step), 6)`.
    `qmul(weighted.x, step)` computes $(k_1 + 2k_2 + 2k_3 + k_4) \times h$.
    Then `qdiv(..., 6)` computes $\frac{1}{6} \dots$
    This matches the RK4 formula exactly.
    BUT wait!
    `add_scaled(state, k1, qdiv(step, 2))`
    This computes `state + qmul(k1, step/2)`.
    `qmul(k1, step/2)` computes $k_1 \times (h/2)$.
    This perfectly matches $y_0 + \frac{h}{2} k_1$.
    This is completely correct.

23. **What about `ap_int` 128-bit product in C++ vs Python?**
    In `qmul`:
    C++: `wide_t product = wide_t(a) * wide_t(b); return q_t(product >> FRAC_BITS);`
    Python: `return (a * b) >> FRAC_BITS`
    Both use signed 128-bit arithmetic (Python uses unbounded, but since `a` and `b` are 64-bit, `a*b` is at most 128-bit).
    The sign is preserved in both. `>> FRAC_BITS` in C++ for `ap_int` is arithmetic shift (floor).
    Python's `>>` is also floor.
    They match perfectly.

24. **Wait, is there an issue with the fractional part of `ell` in `field`?**
    `out.ell = x - y - half_w_zs - ONE_Q;`
    Where `ONE_Q = q_t(1) << FRAC_BITS`.
    This evaluates to $x - y - \frac{w + ZS}{2} - 1$.
    This is exactly the ODE term.

25. **Let's check `determinant` formula in Python again:**
    ```python
    determinant = math.exp(ell2) * (qfloat(x0) * qfloat(y0) - float(ZS_TEXT)) / (x2 * y2 - float(ZS_TEXT)) * q0_area
    ```
    Wait! In `host.cpp`:
    ```cpp
    double x0 = qdouble(inputs[2 * leaf]);
    double y0 = qdouble(inputs[2 * leaf + 1]);
    double x2 = qdouble(word[4]);
    double y2 = qdouble(word[5]);
    double ell2 = qdouble(word[6]);
    double normal0 = x0 * y0 - ZS;
    double normal2 = x2 * y2 - ZS;
    double determinant = std::exp(ell2) * normal0 / normal2 * q0_area;
    ```
    In Python:
    `x2, y2, ell2 = (qfloat(result[index]) for index in (4, 5, 6))`
    `qfloat` does `math.ldexp(value, -FRAC_BITS)`.
    Then `determinant = math.exp(ell2) * (qfloat(x0) * qfloat(y0) - float(ZS_TEXT)) / (x2 * y2 - float(ZS_TEXT)) * q0_area`
    Notice that `qfloat(x0) * qfloat(y0)` in Python is float multiplication.
    `x0 * y0` in `host.cpp` is also float multiplication.
    So they match.

26. **Is there any mathematical OVERREACH?**
    The pass condition in `host.cpp`:
    `bool pass = mismatches == 0 && event_pass == n && negative == n && inside == n;`
    I already noted `negative == n` is redundant.
    Is `inside == n` guaranteed by the math?
    No, `inside` depends on the intervals `C0HORECT2_LOWER` etc, which come from an external decimal replay file. It is an empirical property, not a mathematical theorem.

27. **Is there any other claim?**
    Let's check `event_pass` logic in `host.cpp`:
    `if (word[1] == 2 && word[7] == 7) ++event_pass;`
    This means exactly 2 events were found, and flags 2 and 4 were set.
    Flag 2 is set if `initial_normal > 0`.
    Flag 4 is set if `final_normal > 0`.
    This matches.

28. **What about the initial state in `target23_batch`?**
    ```cpp
    State state = {initial_xy[2 * leaf], initial_xy[2 * leaf + 1], 0, 0};
    ```
    `w` is initialized to 0.
    `ell` is initialized to 0.
    This matches `state = (x, y, 0, 0)` in Python.

29. **Is `time` handled correctly?**
    ```cpp
    q_t time = 0;
    ...
    time += STEP_Q;
    ...
    event1_time = time + local_time;
    ```
    This perfectly tracks the simulated time in Q24.40.
    In Python:
    ```python
    time = 0
    ...
    time += STEP
    ...
    event1_time = time + local_time
    ```
    Matches.

30. **Check `qdiv(step, 2)`:**
    In C++: `return value / divisor;` where `value` is `q_t`, `divisor` is `int`.
    If `step` is $2^{30}$, `step / 2` is $2^{29}$.
    This correctly evaluates to exactly half the step.

31. **What about `qdiv(qmul(weighted.x, step), 6)`?**
    `qmul(weighted.x, step)`
