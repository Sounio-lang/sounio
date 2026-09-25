# Core Examples

> **Checked surface.** Sounio has a checked built-in generic `Option<T>` (with
> `Some` / `None`) and `Result<T, E>`, so general optionality and error handling use
> those. `stdlib/core` additionally provides concrete *numeric* helpers that
> mirror them: `IntResult` / `FloatResult` (`int_ok`, `float_ok`,
> `float_result_is_ok`, `float_result_unwrap`) are exported from
> `stdlib/core/result.sio`, while `IntOption` / `FloatOption`
> (`int_some`, `float_some`, `int_option_is_some`, `float_option_unwrap`) are
> *module-internal* — `stdlib/core/option.sio` does not export them, so they are
> shown for reference rather than as an importable `core::option` API. There is
> no `Knowledge` type here; uncertain values are `Epistemic` from
> `stdlib/epistemic/knowledge.sio` (`ep_measured`, `ep_div`, `ep_val`).

## 1. Comparison & Numeric Utilities

```sio
use cmp::lib::{min_f64, max_f64, clamp_f64}

pub fn main() {
    // Numeric utilities (public helpers exported from cmp::lib)
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
use epistemic::knowledge::{Epistemic, ep_measured, ep_val};

pub fn main() with Panic {
    // Optional uncertain value
    let opt: Option<Epistemic> = Some(ep_measured(42.0, 0.1));

    match opt {
        Some(k) => {
            assert(ep_val(&k) == 42.0);
        },
        None => {},
    }

    let missing: Option<f64> = None
    var is_none: bool = true
    match missing {
        Some(_) => {
            is_none = false
        },
        None => {},
    }
    assert(is_none)
}
```

For reference, `stdlib/core/option.sio` also defines module-internal concrete
numeric helpers `FloatOption` / `IntOption` (`float_some`, `float_none`,
`float_option_is_some`, `float_option_unwrap`) — these are not exported, so
they cannot be `use`d as a `core::option` API.

There is no dedicated `Option<Epistemic>` helper in `stdlib/core`; since `Option<T>` is generic, you may wrap an `Epistemic` directly (the stdlib returns `Option<&Epistemic>`, e.g. `most_uncertain` in `stdlib/epistemic/active.sio`). Carry uncertainty with `Epistemic` directly
(`ep_measured(42.0, 0.1)`), and optionality with the built-in generic
`Option<T>` (or, within `stdlib/core`, the module-internal `FloatOption`) separately.

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
use core::result::Result;
use epistemic::knowledge::{Epistemic, ep_measured, ep_val, ep_div};

pub fn divide_epistemic(a: Epistemic, b: Epistemic) -> Result<Epistemic, ()> with Div {
    if ep_val(&b) == 0.0 {
        return Err(());
    }
    Ok(ep_div(&a, &b))
}

pub fn main() with Div, Panic {
    let x = ep_measured(10.0, 0.1);
    let y = ep_measured(2.0, 0.05);

    let result = divide_epistemic(x, y);

    match result {
        Ok(k) => assert(ep_val(&k) == 5.0),
        Err(_) => {},
    }
}
```

`Result<Knowledge<f64>, ()>` and `a / b` on epistemic values are not on the checked surface. Confidence degrades through `ep_div` itself: the result's confidence is the minimum of the inputs' confidence scaled by `97 / 100` (it never increases under division). Anchor: `stdlib/epistemic/knowledge.sio` — `ep_div` (around lines 177-186) computes `confidence: ep_clamp_conf(ep_min_conf(a.confidence, b.confidence) * 97 / 100)`. Because `confidence` is an `i64`, this is truncating integer division: roughly 97% of the inputs' minimum confidence, rounded down (e.g. confidence 1 becomes 0), so confidence never increases, though the effective loss can exceed 3% for small confidences.

## 5. Integer Utilities

```sio
use cmp::lib::{min_i64, max_i64, clamp_i64}

pub fn main() {
    // Integer helpers (public helpers exported from cmp::lib)
    let x = min_i64(3, 7)
    assert(x == 3)

    let y = max_i64(3, 7)
    assert(y == 7)

    let z = clamp_i64(15, 0, 10)
    assert(z == 10)
}
```

---

**Links to tests:** [`tests/stdlib/core/`](../../tests/stdlib/core/)
