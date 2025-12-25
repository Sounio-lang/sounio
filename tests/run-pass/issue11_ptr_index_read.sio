//@ run-pass
// Issue #11: pointer indexing read should not crash.

fn main() -> i64 {
    var x: i64 = 99
    let ptr = (&x) as *const i64
    return ptr[0]
}
