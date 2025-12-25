//@ run-pass
// Issue #11: reference deref read should work.

fn read_ref(r: &i64) -> i64 {
    return *r
}

fn main() -> i64 {
    var x: i64 = 42
    return read_ref(&x)
}
