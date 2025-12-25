//@ run-pass
// Issue #11: pointer indexing writes should update memory correctly.

fn main() -> i64 {
    var arr: [i64; 4] = [0, 0, 0, 0]

    // Get pointer to first element
    let ptr = (&arr[0]) as *mut i64

    ptr[0] = 10
    ptr[1] = 20
    ptr[2] = 30
    ptr[3] = 40

    // Verify via array access
    return arr[0] + arr[1] + arr[2] + arr[3]
}
