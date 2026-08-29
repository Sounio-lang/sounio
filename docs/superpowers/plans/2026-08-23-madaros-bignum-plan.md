<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-08-23-madaros-bignum-plan
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-08-23-madaros-bignum-plan
-->

# Madaros BigInt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fixed-width (4096-bit max, 16-bit-limb) big-integer type in Sounio with enough arithmetic to do RSA public-key operations (encrypt with a public exponent, verify a signature), targeting the Madaros v0.80.0 compiler.

**Architecture:** One module, `stdlib/bignum/bigint.sio`, built entirely on 16-bit limbs specifically to stay far away from the `u64` bit-63 boundary where Madaros's `>>`/`/`/`%` silently produce wrong results (Finding 11). Every operation is schoolbook (O(n²) multiply, O(bits²) modular reduction) — correctness first, no optimization, since this only ever runs a handful of times per TLS handshake with a small public exponent.

**Tech Stack:** Sounio/Madaros (`./bin/souc`), no FFI/syscalls of any kind — this module is pure computation.

**Spec:** `docs/superpowers/specs/2026-08-23-madaros-bignum-design.md`

## Global Constraints

- **16-bit limbs only.** Every arithmetic step must keep intermediate values far below `2^63` (the audit doc's Finding 11 boundary — `u64` right-shift/divide/modulo are silently wrong once bit 63 is set). Every step below includes a comment showing its actual maximum intermediate value and confirming it stays under this boundary — this is a load-bearing safety property, not incidental commentary, and must not be removed or "simplified away" during implementation.
- **Zero representation convention (decided here, apply with zero exceptions):** a `BigInt` representing the value zero has `len == 0`. Every operation must normalize its result by trimming trailing (most-significant) zero limbs down to `len == 0` if the entire value is zero — never leave `len > 0` with all-zero limbs, and never special-case `limbs[0] == 0` as an alternate "is this zero" check anywhere; always check `len == 0`.
- **Error convention:** `bigint_sub` requires `a >= b` (per the spec) — callers must check via `bigint_cmp` first; violating this is a caller bug, not a runtime-checked error (schoolbook subtraction with an unchecked borrow past the most significant limb would silently produce garbage, which is acceptable here since every call site in this plan's own code always checks first, and this module's only consumers are this plan's own tasks). `bigint_mod`'s divisor (`n`) must be non-zero — if `n.len == 0`, `bigint_mod` returns a copy of `a` unchanged (a defined, safe fallback rather than a crash) and this behavior must be covered by a test; do not introduce `Result<T,E>`/`Option<T>` anywhere in this module, matching the project-wide sentinel-based convention (there is no sentinel needed here since every function always returns a valid `BigInt`, just possibly an unhelpful one on misuse).
- No AI-attribution line in any commit (this repo's `CLAUDE.md`: "No AI attribution in commit messages").
- Commit message convention: Conventional-Commits-style `type(scope): description` (e.g. `feat(bignum): ...`, `test(bignum): ...`), matching this branch's established real commit history.
- Module import path: test files (outside `stdlib/`) use `use bignum::bigint::*`.
- Test files: `tests/run-pass/bignum_*.sio`, run via `./bin/souc run <file>` individually and `bash scripts/run_sio_test_suite.sh --filter-prefix bignum_` for the group. **Never run the whole-repo test suite** (thousands of files, far too slow for this branch's checkpoints).
- **Run every shell command as a plain foreground command, one at a time, reading output immediately. Never use any Monitor/background-wait mechanism** — this has caused real, repeated stalls in prior work on this exact branch (implementers and reviewers getting stuck waiting for a notification instead of just running the next command).

---

## Task 1: `BigInt` struct, construction, and comparison

**Files:**
- Create: `stdlib/bignum/bigint.sio`
- Test: `tests/run-pass/bignum_construct_compare.sio`

**Interfaces:**
- Produces: `pub const BIGINT_MAX_LIMBS: i32 = 256`, `pub struct BigInt { limbs: [u16; 256], len: i32 }`, `pub fn bigint_zero() -> BigInt`, `pub fn bigint_from_u32(v: u32) -> BigInt`, `pub fn bigint_cmp(a: &BigInt, b: &BigInt) -> i32` (returns `-1`/`0`/`1`). Tasks 2-5 all use these exact names/signatures.

- [ ] **Step 1: Write the failing test**

```sio
//@ run-pass

use bignum::bigint::*

fn main() -> i64 with IO {
    let zero = bigint_zero()
    if zero.len != 0 {
        print_int(1)
        return 1
    }

    let a = bigint_from_u32(65535)
    if a.len != 1 {
        print_int(2)
        return 1
    }
    if a.limbs[0] != 65535 {
        print_int(3)
        return 1
    }

    // A value needing 2 limbs: 65536 = 0x10000 = limb0=0, limb1=1
    let b = bigint_from_u32(65536)
    if b.len != 2 {
        print_int(4)
        return 1
    }
    if b.limbs[0] != 0 {
        print_int(5)
        return 1
    }
    if b.limbs[1] != 1 {
        print_int(6)
        return 1
    }

    // Comparison: equal
    let c = bigint_from_u32(65536)
    if bigint_cmp(&b, &c) != 0 {
        print_int(7)
        return 1
    }

    // Comparison: different lengths
    if bigint_cmp(&a, &b) != -1 {   // 65535 < 65536
        print_int(8)
        return 1
    }
    if bigint_cmp(&b, &a) != 1 {
        print_int(9)
        return 1
    }

    // Comparison: same length, different value
    let d = bigint_from_u32(100)
    let e = bigint_from_u32(200)
    if bigint_cmp(&d, &e) != -1 {
        print_int(10)
        return 1
    }

    // Comparison against zero
    if bigint_cmp(&zero, &d) != -1 {
        print_int(11)
        return 1
    }
    if bigint_cmp(&zero, &zero) != 0 {
        print_int(12)
        return 1
    }

    print_int(0)
    return 0
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./bin/souc run tests/run-pass/bignum_construct_compare.sio`
Expected: FAIL — `stdlib/bignum/bigint.sio` doesn't exist yet.

- [ ] **Step 3: Implement `stdlib/bignum/bigint.sio`**

```sio
// stdlib/bignum/bigint.sio
//
// Fixed-width big integer (max 4096 bits = 256 limbs of 16 bits each), for
// RSA public-key operations only (no private-key/constant-time requirement
// -- see docs/superpowers/specs/2026-08-23-madaros-bignum-design.md).
//
// LIMB WIDTH IS 16 BITS, NOT 32, BY DESIGN: see
// docs/audit/TLS_PREREQ_WIDE_INT_AND_RAW_BUFFERS_2026-08-23.md Finding 11 --
// u64 right-shift/divide/modulo silently produce wrong results whenever the
// operand's bit 63 is set. Every arithmetic step in this file keeps its
// largest intermediate value far below that boundary (documented inline at
// each step) specifically to never trigger that bug.
//
// Zero representation: a BigInt representing zero has len == 0. Every
// operation normalizes its result to len == 0 if the value is entirely
// zero -- never leave len > 0 with all-zero limbs.

pub const BIGINT_MAX_LIMBS: i32 = 256

pub struct BigInt {
    limbs: [u16; 256],   // little-endian: limbs[0] is least significant
    len: i32,             // number of significant limbs; 0 means the value is zero
}

pub fn bigint_zero() -> BigInt {
    BigInt { limbs: [0; 256], len: 0 }
}

pub fn bigint_from_u32(v: u32) -> BigInt {
    var result = bigint_zero()
    let lo = (v & 65535) as u16       // v is a 32-bit input, & 65535 is safe, far from bit 63
    let hi = (v >> 16) as u16          // v is a 32-bit input; >>16 on a value that fits in u32
                                        // never approaches bit 63 -- this shift operates on the
                                        // caller's 32-bit input value, not a wide product
    result.limbs[0] = lo
    if hi != 0 {
        result.limbs[1] = hi
        result.len = 2
    } else if lo != 0 {
        result.len = 1
    } else {
        result.len = 0
    }
    result
}

// Compares a and b as unsigned magnitudes. Returns -1 if a<b, 0 if a==b, 1 if a>b.
// All comparisons here are between plain i32 (len) and u16 (limb) values --
// never between raw u64 values, so Finding 11's ordering-comparison caveat
// (only confirmed broken for u64) does not apply anywhere in this function.
pub fn bigint_cmp(a: &BigInt, b: &BigInt) -> i32 {
    if a.len < b.len {
        return 0 - 1
    }
    if a.len > b.len {
        return 1
    }
    // Same length: compare from the most significant limb down.
    var i = a.len - 1
    while i >= 0 {
        let ai = a.limbs[i as usize]
        let bi = b.limbs[i as usize]
        if ai < bi {
            return 0 - 1
        }
        if ai > bi {
            return 1
        }
        i = i - 1
    }
    0
}
```

The `limbs[i as usize]` indexing syntax and the exact `usize`/`i32` interplay should be verified against real, existing indexing patterns elsewhere in this repo (e.g. `stdlib/net/socket.sio`'s array/loop code) — adjust the cast if the compiler expects a different index type; this is a minor syntax detail, not a design risk.

- [ ] **Step 4: Run the test to verify it passes**

Run: `./bin/souc run tests/run-pass/bignum_construct_compare.sio`
Expected: PASS, prints `0`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add stdlib/bignum/bigint.sio tests/run-pass/bignum_construct_compare.sio
git commit -m "feat(bignum): add BigInt struct, construction, and comparison"
```

---

## Task 2: Addition and subtraction

**Files:**
- Modify: `stdlib/bignum/bigint.sio`
- Test: `tests/run-pass/bignum_add_sub.sio`

**Interfaces:**
- Consumes: `BigInt`, `bigint_zero`, `bigint_from_u32`, `bigint_cmp` from Task 1.
- Produces: `pub fn bigint_add(a: &BigInt, b: &BigInt) -> BigInt`, `pub fn bigint_sub(a: &BigInt, b: &BigInt) -> BigInt` (caller must ensure `bigint_cmp(a, b) >= 0` first — undefined/garbage result otherwise, per this plan's Global Constraints). Tasks 3-5 use these.

- [ ] **Step 1: Write the failing test**

```sio
//@ run-pass

use bignum::bigint::*

fn main() -> i64 with IO {
    // Simple addition, no carry
    let a = bigint_from_u32(100)
    let b = bigint_from_u32(200)
    let sum = bigint_add(&a, &b)
    let expected_sum = bigint_from_u32(300)
    if bigint_cmp(&sum, &expected_sum) != 0 {
        print_int(1)
        return 1
    }

    // Addition with carry across a limb boundary: 65535 + 1 = 65536
    let c = bigint_from_u32(65535)
    let one = bigint_from_u32(1)
    let sum2 = bigint_add(&c, &one)
    let expected_sum2 = bigint_from_u32(65536)
    if bigint_cmp(&sum2, &expected_sum2) != 0 {
        print_int(2)
        return 1
    }

    // Addition growing beyond 2 limbs: 65536 + 65535*65536 = 65536*65536 = 2^32
    // (65535 * 65536) as a BigInt: limb0=0, limb1=65535
    var big = bigint_zero()
    big.limbs[1] = 65535
    big.len = 2
    let sum3 = bigint_add(&big, &sum2)   // (65535*65536) + 65536 = 65536*65536 = 2^32
    if sum3.len != 3 {
        print_int(3)
        return 1
    }
    if sum3.limbs[0] != 0 || sum3.limbs[1] != 0 || sum3.limbs[2] != 1 {
        print_int(4)
        return 1
    }

    // Subtraction, no borrow
    let diff = bigint_sub(&b, &a)   // 200 - 100 = 100
    if bigint_cmp(&diff, &a) != 0 {
        print_int(5)
        return 1
    }

    // Subtraction with borrow across a limb boundary: 65536 - 1 = 65535
    let diff2 = bigint_sub(&expected_sum2, &one)
    if bigint_cmp(&diff2, &c) != 0 {
        print_int(6)
        return 1
    }

    // Subtraction resulting in zero
    let diff3 = bigint_sub(&a, &a)
    if diff3.len != 0 {
        print_int(7)
        return 1
    }

    print_int(0)
    return 0
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./bin/souc run tests/run-pass/bignum_add_sub.sio`
Expected: FAIL — `bigint_add`/`bigint_sub` don't exist yet.

- [ ] **Step 3: Implement `bigint_add` and `bigint_sub` (append to `stdlib/bignum/bigint.sio`)**

```sio
// Adds a and b. Every intermediate `sum` value below is at most
// 65535 + 65535 + 1 = 131071, computed as u32 -- nowhere near bit 63,
// so the `>> 16` used to extract the carry is safe per Finding 11
// (the boundary is bit 63 of a u64; 131071 doesn't even fill 32 bits).
pub fn bigint_add(a: &BigInt, b: &BigInt) -> BigInt {
    var result = bigint_zero()
    var carry: u32 = 0
    var max_len = a.len
    if b.len > max_len {
        max_len = b.len
    }
    var i = 0
    while i < max_len {
        var ai: u32 = 0
        if i < a.len {
            ai = a.limbs[i as usize] as u32
        }
        var bi: u32 = 0
        if i < b.len {
            bi = b.limbs[i as usize] as u32
        }
        let sum: u32 = ai + bi + carry   // max 131071, far below any u64 bit-63 concern
        result.limbs[i as usize] = (sum & 65535) as u16
        carry = sum >> 16                 // safe: sum is tiny, nowhere near bit 63
        i = i + 1
    }
    if carry != 0 {
        result.limbs[max_len as usize] = carry as u16
        max_len = max_len + 1
    }
    result.len = max_len
    // Normalize: trim trailing zero limbs (can only happen if max_len was
    // computed from operands whose top limb summed to exactly 0, which
    // cannot happen from addition of two non-negative values unless both
    // were zero -- but trim defensively anyway to keep the invariant airtight).
    while result.len > 0 && result.limbs[(result.len - 1) as usize] == 0 {
        result.len = result.len - 1
    }
    result
}

// Subtracts b from a. REQUIRES bigint_cmp(a, b) >= 0 -- caller's responsibility,
// per this plan's Global Constraints; an unchecked call with a < b produces a
// garbage result (not a crash, not a checked error), matching this module's
// stated error convention.
pub fn bigint_sub(a: &BigInt, b: &BigInt) -> BigInt {
    var result = bigint_zero()
    var borrow: i32 = 0
    var i = 0
    while i < a.len {
        var ai: i32 = a.limbs[i as usize] as i32
        var bi: i32 = 0
        if i < b.len {
            bi = b.limbs[i as usize] as i32
        }
        var diff: i32 = ai - bi - borrow   // range roughly -65536..65535, tiny, safe
        if diff < 0 {
            diff = diff + 65536
            borrow = 1
        } else {
            borrow = 0
        }
        result.limbs[i as usize] = diff as u16
        i = i + 1
    }
    result.len = a.len
    while result.len > 0 && result.limbs[(result.len - 1) as usize] == 0 {
        result.len = result.len - 1
    }
    result
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./bin/souc run tests/run-pass/bignum_add_sub.sio`
Expected: PASS, prints `0`.

- [ ] **Step 5: Commit**

```bash
git add stdlib/bignum/bigint.sio tests/run-pass/bignum_add_sub.sio
git commit -m "feat(bignum): add addition and subtraction with carry/borrow"
```

---

## Task 3: Multiplication

**Files:**
- Modify: `stdlib/bignum/bigint.sio`
- Test: `tests/run-pass/bignum_mul.sio`

**Interfaces:**
- Consumes: `BigInt`, `bigint_zero`, `bigint_from_u32`, `bigint_cmp` from Task 1.
- Produces: `pub fn bigint_mul(a: &BigInt, b: &BigInt) -> BigInt`. Tasks 4-5 use this.

- [ ] **Step 1: Write the failing test**

```sio
//@ run-pass

use bignum::bigint::*

fn main() -> i64 with IO {
    // Small, hand-computable case
    let a = bigint_from_u32(123)
    let b = bigint_from_u32(456)
    let product = bigint_mul(&a, &b)   // 123 * 456 = 56088
    let expected = bigint_from_u32(56088)
    if bigint_cmp(&product, &expected) != 0 {
        print_int(1)
        return 1
    }

    // THE critical test: multiply two values whose product's bit 63 would
    // be set if computed via naive 32-bit-limb math -- this is exactly the
    // failure mode this whole design (16-bit limbs) exists to avoid.
    // 4294967295 * 4294967295 = 18446744065119617025 = 0xFFFFFFFE00000001
    // (bit 63 of this 64-bit product IS set -- this is precisely the value
    // used to discover Finding 11 in the first place).
    let big_a = bigint_from_u32(4294967295)
    let big_b = bigint_from_u32(4294967295)
    let big_product = bigint_mul(&big_a, &big_b)

    // Expected result as a BigInt: 0xFFFFFFFE00000001 split into 16-bit
    // limbs, little-endian: limb0=0x0001, limb1=0x0000, limb2=0xFFFE, limb3=0xFFFF
    var expected_big = bigint_zero()
    expected_big.limbs[0] = 1
    expected_big.limbs[1] = 0
    expected_big.limbs[2] = 65534   // 0xFFFE
    expected_big.limbs[3] = 65535   // 0xFFFF
    expected_big.len = 4
    if bigint_cmp(&big_product, &expected_big) != 0 {
        print_int(2)
        return 1
    }

    // Multiplication by zero
    let zero = bigint_zero()
    let zero_product = bigint_mul(&a, &zero)
    if zero_product.len != 0 {
        print_int(3)
        return 1
    }

    print_int(0)
    return 0
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./bin/souc run tests/run-pass/bignum_mul.sio`
Expected: FAIL — `bigint_mul` doesn't exist yet.

- [ ] **Step 3: Implement `bigint_mul` (append to `stdlib/bignum/bigint.sio`)**

```sio
// Schoolbook O(n^2) multiplication. Accumulates partial products into a
// u64 array twice the max limb count, then does one final carry-propagation
// pass. Bound analysis (this is THE reason 16-bit limbs were chosen):
//   - each partial product a.limbs[i] * b.limbs[j] is at most
//     65535 * 65535 = 4294836225, comfortably under 2^32, let alone bit 63.
//   - each accumulator slot can receive at most BIGINT_MAX_LIMBS (256)
//     partial-product contributions in the worst case (a maximally-sized
//     a and b), giving a worst-case accumulator value of
//     256 * 4294836225 ~= 2^40 -- still enormously far below 2^63.
//   - the final carry-propagation pass's `>> 16` and `& 65535` operate on
//     values bounded by that same ~2^40 ceiling, safe per Finding 11
//     (which only breaks values with bit 63 set).
pub fn bigint_mul(a: &BigInt, b: &BigInt) -> BigInt {
    var acc: [u64; 512] = [0; 512]   // BIGINT_MAX_LIMBS * 2

    var i = 0
    while i < a.len {
        var j = 0
        while j < b.len {
            let ai: u64 = a.limbs[i as usize] as u64
            let bj: u64 = b.limbs[j as usize] as u64
            let partial: u64 = ai * bj   // max ~4.29e9, tiny
            acc[(i + j) as usize] = acc[(i + j) as usize] + partial   // max ~2^40, tiny
            j = j + 1
        }
        i = i + 1
    }

    var result = bigint_zero()
    var carry: u64 = 0
    var k = 0
    while k < 512 {
        let total: u64 = acc[k as usize] + carry   // still far below bit 63
        if k < 256 {
            result.limbs[k as usize] = (total & 65535) as u16
        }
        carry = total >> 16   // safe: total is bounded far below bit 63
        k = k + 1
    }

    result.len = 512
    if result.len > 256 {
        result.len = 256   // acc has 512 slots for overflow headroom during
                             // accumulation, but a BigInt's limbs array is
                             // only 256 wide -- this plan does not need
                             // products larger than BIGINT_MAX_LIMBS for its
                             // RSA use case (both operands are already
                             // bounded to <=256 limbs, so a full product
                             // needs at most their combined length, safely
                             // under 512, but the visible `limbs` array
                             // itself caps at 256; this length clamp is a
                             // defensive bound, not expected to trigger for
                             // this module's actual RSA-sized inputs)
    }
    while result.len > 0 && result.limbs[(result.len - 1) as usize] == 0 {
        result.len = result.len - 1
    }
    result
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./bin/souc run tests/run-pass/bignum_mul.sio`
Expected: PASS, prints `0` — this specifically confirms the design's central safety claim: a product whose bit 63 is set (when viewed as a raw 64-bit value) is computed correctly via 16-bit-limb schoolbook multiplication, unlike the naive `u64`-shift approach Finding 11 proved broken.

- [ ] **Step 5: Commit**

```bash
git add stdlib/bignum/bigint.sio tests/run-pass/bignum_mul.sio
git commit -m "feat(bignum): add schoolbook multiplication, verified against the bit-63 boundary case"
```

---

## Task 4: Modular reduction

**Files:**
- Modify: `stdlib/bignum/bigint.sio`
- Test: `tests/run-pass/bignum_mod.sio`

**Interfaces:**
- Consumes: `BigInt`, `bigint_zero`, `bigint_from_u32`, `bigint_cmp`, `bigint_sub` from Tasks 1-2.
- Produces: `pub fn bigint_mod(a: &BigInt, n: &BigInt) -> BigInt`. Task 5 uses this.

- [ ] **Step 1: Write the failing test**

```sio
//@ run-pass

use bignum::bigint::*

fn main() -> i64 with IO {
    // Simple case: 17 mod 5 = 2
    let a = bigint_from_u32(17)
    let n = bigint_from_u32(5)
    let r = bigint_mod(&a, &n)
    let expected = bigint_from_u32(2)
    if bigint_cmp(&r, &expected) != 0 {
        print_int(1)
        return 1
    }

    // a < n: result is a unchanged
    let small = bigint_from_u32(3)
    let r2 = bigint_mod(&small, &n)
    if bigint_cmp(&r2, &small) != 0 {
        print_int(2)
        return 1
    }

    // a exactly divisible by n: result is zero
    let ten = bigint_from_u32(10)
    let r3 = bigint_mod(&ten, &n)
    if r3.len != 0 {
        print_int(3)
        return 1
    }

    // Divisor is zero: defined fallback per this plan's error convention --
    // returns a copy of a unchanged.
    let zero = bigint_zero()
    let r4 = bigint_mod(&a, &zero)
    if bigint_cmp(&r4, &a) != 0 {
        print_int(4)
        return 1
    }

    // A case needing multiple limbs: 70000 mod 65537
    // (65537 = 0x10001, needs 2 limbs; 70000 mod 65537 = 4463)
    let big_a = bigint_from_u32(70000)
    let modulus = bigint_from_u32(65537)
    let r5 = bigint_mod(&big_a, &modulus)
    let expected5 = bigint_from_u32(4463)
    if bigint_cmp(&r5, &expected5) != 0 {
        print_int(5)
        return 1
    }

    print_int(0)
    return 0
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./bin/souc run tests/run-pass/bignum_mod.sio`
Expected: FAIL — `bigint_mod` doesn't exist yet.

- [ ] **Step 3: Implement `bigint_mod` (append to `stdlib/bignum/bigint.sio`)**

```sio
// Binary long division: processes a's bits from most significant to least
// significant, maintaining a running remainder shifted left by 1 bit and
// OR'd with the next input bit at each step, subtracting n whenever the
// remainder is >= n. O(bits^2) but correctness-first; RSA's ~4096-bit
// modulus makes this a small absolute cost for a once-per-handshake op.
//
// Divisor-is-zero: returns a copy of a unchanged (a defined, safe fallback
// per this module's error convention -- no Result/Option, no crash).
pub fn bigint_mod(a: &BigInt, n: &BigInt) -> BigInt {
    if n.len == 0 {
        return *a
    }

    var remainder = bigint_zero()
    var bit_index = a.len * 16 - 1   // highest possible bit position in a

    while bit_index >= 0 {
        let limb_index = bit_index / 16   // a.len is small (<=256), so this
                                            // division is on tiny i32 values,
                                            // not a u64 -- Finding 11 doesn't
                                            // apply to i32 arithmetic at all
        let bit_offset = bit_index % 16     // same: tiny i32 values only
        let this_limb: u16 = a.limbs[limb_index as usize]
        let bit: u16 = (this_limb >> bit_offset) & 1   // this_limb is a 16-bit
                                                          // value; shifting it
                                                          // (as u16, promoted
                                                          // for the op) never
                                                          // approaches bit 63

        // remainder = (remainder << 1) | bit
        remainder = bigint_shl1_or_bit(&remainder, bit)

        if bigint_cmp(&remainder, n) >= 0 {
            remainder = bigint_sub(&remainder, n)
        }

        bit_index = bit_index - 1
    }

    remainder
}

// Helper: shifts a BigInt left by exactly 1 bit and ORs in a single bit at
// position 0. Every limb's intermediate value here is at most
// (65535 << 1) | 1 = 131071, computed as u32 -- tiny, safe.
fn bigint_shl1_or_bit(a: &BigInt, bit: u16) -> BigInt {
    var result = bigint_zero()
    var carry: u32 = bit as u32
    var i = 0
    while i < a.len {
        let limb: u32 = a.limbs[i as usize] as u32
        let shifted: u32 = (limb << 1) | carry   // max 131071, tiny
        result.limbs[i as usize] = (shifted & 65535) as u16
        carry = shifted >> 16                       // safe: shifted is tiny
        i = i + 1
    }
    var len = a.len
    if carry != 0 {
        result.limbs[len as usize] = carry as u16
        len = len + 1
    }
    result.len = len
    while result.len > 0 && result.limbs[(result.len - 1) as usize] == 0 {
        result.len = result.len - 1
    }
    result
}
```

`*a` (dereferencing a `&BigInt` to return an owned copy) should be verified against real, existing dereference-copy patterns elsewhere in this repo before trusting it compiles as written — `BigInt` is a plain (non-linear) struct, so a copy should be unproblematic, but confirm the exact syntax Madaros expects for "copy out of a reference" during this step.

- [ ] **Step 4: Run the test to verify it passes**

Run: `./bin/souc run tests/run-pass/bignum_mod.sio`
Expected: PASS, prints `0`.

- [ ] **Step 5: Commit**

```bash
git add stdlib/bignum/bigint.sio tests/run-pass/bignum_mod.sio
git commit -m "feat(bignum): add modular reduction via binary long division"
```

---

## Task 5: Modular exponentiation, verified against a real RSA test vector

**Files:**
- Modify: `stdlib/bignum/bigint.sio`
- Test: `tests/run-pass/bignum_modpow_rsa.sio`

**Interfaces:**
- Consumes: `BigInt`, `bigint_zero`, `bigint_from_u32`, `bigint_cmp`, `bigint_mul`, `bigint_mod` from Tasks 1-4.
- Produces: `pub fn bigint_modpow(base: &BigInt, exponent: &BigInt, modulus: &BigInt) -> BigInt`. No later task in this plan consumes this (it's the final deliverable), but the eventual TLS/X.509 work will.

- [ ] **Step 1: Independently re-verify the RSA test vector before trusting it in a committed test**

This plan uses the well-known textbook RSA example: `p=61, q=53, n=p*q=3233, public exponent e=17, plaintext m=65, expected ciphertext c=2790` (i.e. `65^17 mod 3233 == 2790`). Before writing this into a permanent test, independently verify this arithmetic yourself — do NOT simply trust that it's "well-known." One way: compute `65^17 mod 3233` by repeated squaring on paper or with an independent calculator/tool available to you (not this compiler, since that would be circular) — e.g.: `65^1=65`, `65^2 mod 3233 = 992`, `65^4 mod 3233 = 1232`, `65^8 mod 3233 = 1547`, `65^16 mod 3233 = 789`, `65^17 mod 3233 = (789 * 65) mod 3233 = 2790`. Confirm you get `2790` independently before proceeding; if you get a different answer, STOP and report this as BLOCKED rather than committing a test with a self-inconsistent or unverified expected value — a wrong vector would make the test "pass" in a way that proves nothing about the implementation's correctness.

- [ ] **Step 2: Write the failing test**

```sio
//@ run-pass

use bignum::bigint::*

fn main() -> i64 with IO {
    // Small hand-computable case first: 2^10 mod 1000 = 24
    let base_small = bigint_from_u32(2)
    let exp_small = bigint_from_u32(10)
    let mod_small = bigint_from_u32(1000)
    let result_small = bigint_modpow(&base_small, &exp_small, &mod_small)
    let expected_small = bigint_from_u32(24)
    if bigint_cmp(&result_small, &expected_small) != 0 {
        print_int(1)
        return 1
    }

    // The real RSA test vector: 65^17 mod 3233 = 2790
    // (p=61, q=53, n=3233, public exponent e=17, plaintext m=65)
    // -- independently re-verified in Step 1 before this test was written.
    let base = bigint_from_u32(65)
    let exponent = bigint_from_u32(17)
    let modulus = bigint_from_u32(3233)
    let ciphertext = bigint_modpow(&base, &exponent, &modulus)
    let expected_ciphertext = bigint_from_u32(2790)
    if bigint_cmp(&ciphertext, &expected_ciphertext) != 0 {
        print_int(2)
        return 1
    }

    // Exponent of zero: anything^0 mod n = 1 (mod n), for n > 1
    let exp_zero = bigint_zero()
    let result_zero_exp = bigint_modpow(&base, &exp_zero, &modulus)
    let one = bigint_from_u32(1)
    if bigint_cmp(&result_zero_exp, &one) != 0 {
        print_int(3)
        return 1
    }

    print_int(0)
    return 0
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `./bin/souc run tests/run-pass/bignum_modpow_rsa.sio`
Expected: FAIL — `bigint_modpow` doesn't exist yet.

- [ ] **Step 4: Implement `bigint_modpow` (append to `stdlib/bignum/bigint.sio`)**

```sio
// Square-and-multiply modular exponentiation. Iterates over the exponent's
// bits from most significant to least significant, squaring an accumulator
// (mod n) at every step, and additionally multiplying by base (mod n)
// whenever the current exponent bit is 1. For this module's intended use
// (a small public RSA exponent, typically 65537 = 17 bits with exactly two
// set bits), this loop runs a small, bounded number of iterations regardless
// of how large base/modulus are (up to 4096 bits) -- cheap despite being
// schoolbook-everything underneath.
pub fn bigint_modpow(base: &BigInt, exponent: &BigInt, modulus: &BigInt) -> BigInt {
    var result = bigint_from_u32(1)
    var base_mod = bigint_mod(base, modulus)

    var bit_index = exponent.len * 16 - 1
    while bit_index >= 0 {
        let limb_index = bit_index / 16   // tiny i32 arithmetic, not u64 -- safe
        let bit_offset = bit_index % 16
        let this_limb: u16 = exponent.limbs[limb_index as usize]
        let bit: u16 = (this_limb >> bit_offset) & 1

        let squared = bigint_mul(&result, &result)
        result = bigint_mod(&squared, modulus)

        if bit == 1 {
            let multiplied = bigint_mul(&result, &base_mod)
            result = bigint_mod(&multiplied, modulus)
        }

        bit_index = bit_index - 1
    }

    result
}
```

Note: `exponent.len * 16 - 1` starts iteration from the top of the exponent's ALLOCATED limb range, not its true highest set bit — this means the loop does extra no-op squaring/mod work for leading zero bits above the exponent's actual magnitude, which is correct (squaring `1` and reducing mod n repeatedly is a no-op on the result) but wastes some cycles. This is an accepted simplification for this plan's correctness-first scope (the exponents this module handles are always small, so this waste is negligible); a future optimization could start from the true highest set bit instead, but that is not required here.

- [ ] **Step 5: Run the test to verify it passes**

Run: `./bin/souc run tests/run-pass/bignum_modpow_rsa.sio`
Expected: PASS, prints `0` — this is the test that proves the whole module is fit for its stated purpose (RSA public-key operations), not just "arithmetic seems to work."

- [ ] **Step 6: Run the full `bignum_` test group**

Run: `bash scripts/run_sio_test_suite.sh --filter-prefix bignum_`
Expected: all 5 `bignum_*` test files pass.

- [ ] **Step 7: Commit**

```bash
git add stdlib/bignum/bigint.sio tests/run-pass/bignum_modpow_rsa.sio
git commit -m "feat(bignum): add modular exponentiation, verified against a real RSA test vector"
```

---

## Self-Review Notes

**Spec coverage:** all operations the spec lists (construction, comparison, addition, subtraction, multiplication, modular reduction, modular exponentiation) map to exactly one task each. The spec's central safety argument (16-bit limbs to stay away from Finding 11's bit-63 boundary) is explicitly re-derived with real numbers in every task's code comments, and Task 3's test directly exercises the exact failure case (`4294967295 * 4294967295`) that originally revealed Finding 11. The spec's testing-strategy order (small hand-computable case, then a real RSA vector, then the bit-63 boundary case) is followed: Task 3 covers the boundary case, Task 5 covers the RSA vector, and every task includes a small hand-computable case first.

**Placeholder scan:** no task contains a "TBD"/"add error handling"/vague step. The one explicitly-flagged uncertainty (Task 5's RSA vector) is handled by requiring independent re-verification as a real step with a real computation shown, not by asserting the vector is definitely correct without checking.

**Type consistency:** `BigInt { limbs: [u16; 256], len: i32 }` is declared once (Task 1) and every later task's function signatures and internal code use exactly this shape — no task introduces a competing representation. `bigint_cmp`'s `-1`/`0`/`1` return convention is used identically in every later task's checks. `bigint_sub`'s "caller must check `bigint_cmp >= 0` first" precondition (declared in this plan's Global Constraints) is respected by both of its call sites in Task 4 (`bigint_mod`, always called after a `bigint_cmp(&remainder, n) >= 0` check) — no task violates its own stated precondition.
