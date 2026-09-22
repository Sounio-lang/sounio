# Core Examples

> **Checked surface.** `stdlib/core` has no generic `Option<T>` or `Result<T>`
> and no `match` on them. Optional values are `IntOption` / `FloatOption`
> (`int_some`, `float_some`, `int_option_is_some`, `float_option_unwrap`) and
> fallible values are `IntResult` / `FloatResult` (`int_ok`, `float_ok`,
> `float_result_is_ok`, `float_result_unwrap`) in `stdlib/core/option.sio` and
> `stdlib/core/result.sio`. There is no `Knowledge` type here; uncertain values
> are `Epistemic` from `stdlib/epistemic/knowledge.sio` (`ep_measured`,
> `ep_div`, `ep_val`).

## 1. Prelude Utilities

```sio
use core::prelude::*;

pub fn main() {
    // Numeric utilities
    let a = abs_f64(-3.14)
    assert(a == 3.14)

    let b = min_f64(1.0, 2.0)
    assert(b == 1.0)

    let c = max_f64(1.0, 2.0)
    assert(c == 2.0)

    let d = clamp_f64(5.0, 0.0, 1.0)
    assert(d == 1.0)
}
```

## 2. Optional Values

```sio
use core::option::{float_some, float_none, float_option_is_some, float_option_unwrap}

pub fn main() with Panic {
    let opt = float_some(42.0)
    assert(float_option_is_some(&opt))
    assert(float_option_unwrap(&opt) == 42.0)

    let missing = float_none()
    assert(!float_option_is_some(&missing))
}
```

There is no `Option<Epistemic>`. Carry uncertainty with `Epistemic` directly
(`ep_measured(42.0, 0.1)`), and optionality with `FloatOption` separately.

## 3. Result Error Handling

```sio
use core::result::{float_ok, float_err, float_result_is_ok, float_result_unwrap}

pub fn main() with Panic {
    let res = float_ok(3.14)
    assert(float_result_is_ok(&res))
    assert(float_result_unwrap(&res) > 3.0)

    let bad = float_err(1)
    assert(!float_result_is_ok(&bad))
}
```

## 4. Epistemic Division

```sio
use epistemic::knowledge::{ep_measured, ep_div, ep_val}

pub fn main() with Div, Panic {
    // ep_measured(val, std_dev). ep_div propagates variance by the GUM delta
    // method and panics on a zero divisor (the Div, Panic effects).
    let x = ep_measured(10.0, 0.1)
    let y = ep_measured(2.0, 0.05)

    let result = ep_div(&x, &y)
    assert(ep_val(&result) == 5.0)
}
```

`Result<Knowledge<f64>, ()>` and `a / b` on epistemic values are not on the checked surface. Confidence degrades through `ep_div` itself: the result keeps the minimum of the inputs' confidence. Anchor: `tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio`.

## 5. Integer Utilities

```sio
use core::prelude::*;

pub fn main() {
    let x = abs_i64(-10)
    assert(x == 10)

    let y = clamp_i64(15, 0, 10)
    assert(y == 10)
}
```

---

**Links to tests:** [`tests/stdlib/core/`](../../tests/stdlib/core/)
