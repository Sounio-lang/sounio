# Core Examples

## 1. Prelude Utilities

```sio
use core::prelude::*;

pub fn main() {
    // Numeric utilities
    let a = abs_f64(-3.14);
    assert(a == 3.14);
    
    let b = min_f64(1.0, 2.0);
    assert(b == 1.0);
    
    let c = max_f64(1.0, 2.0);
    assert(c == 2.0);
    
    let d = clamp_f64(5.0, 0.0, 1.0);
    assert(d == 1.0);
}
```

## 2. Option with Knowledge (Epistemic)

```sio
use core::option::Option;
use epistemic::knowledge::Knowledge;

pub fn main() with Mut, Panic {
    // Optional uncertain value
    let opt: Option<Knowledge<f64>> = Option::Some(Knowledge::measured(42.0, 0.1, "sensor_A"));
    
    match opt {
        Option::Some(k) => {
            assert(k.value == 42.0);
        },
        Option::None => {},
    }
}
```

## 3. Result Error Handling

```sio
use core::result::Result;

pub fn main() with Panic {
    let res: Result<f64, i64> = Ok(3.14);
    
    match res {
        Result::Ok(v) => assert(v > 3.0),
        Result::Err(_) => {},
    }
}
```

## 4. Confidence Degradation Through Error Path

```sio
use core::result::Result;
use epistemic::knowledge::Knowledge;

pub fn divide_epistemic(a: Knowledge<f64>, b: Knowledge<f64>) -> Result<Knowledge<f64>, ()> with Div {
    if b.value == 0.0 {
        return Err(());
    }
    Ok(a / b)
}

pub fn main() with Div, Panic {
    let x = Knowledge::measured(10.0, 0.1, "source");
    let y = Knowledge::measured(2.0, 0.05, "source");
    
    let result = divide_epistemic(x, y);
    
    match result {
        Ok(k) => assert(k.value == 5.0),
        Err(_) => {},
    }
}
```

## 5. Integer Utilities

```sio
use core::prelude::*;

pub fn main() {
    let x = abs_i64(-10);
    assert(x == 10);
    
    let y = clamp_i64(15, 0, 10);
    assert(y == 10);
}
```

---

**Links to tests:** [`tests/stdlib/core/`](../../tests/stdlib/core/)
