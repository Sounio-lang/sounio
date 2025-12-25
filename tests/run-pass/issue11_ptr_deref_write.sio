//@ run-pass
// Issue #11: pointer write via indexing should update the underlying value.

fn main() -> i64 {
    var x: i64 = 0

    let ptr = (&x) as *mut i64

    ptr[0] = 42

    return x
}
