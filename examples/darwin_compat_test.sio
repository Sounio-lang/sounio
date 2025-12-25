// Test Darwin Atlas compatibility features

// Test 1: 'use' keyword as alias for 'import' (both :: and . work)
use std.math;

// Test 2: ++ concatenation operator
fn test_concat() -> i32 {
    let a = [1, 2, 3]
    let b = [4, 5, 6]
    let c = a ++ b
    return 0
}

// Test 3: Slice ranges
fn test_slices() -> i32 {
    let seq = [1, 2, 3, 4, 5]
    let k = 2

    // Slice from start to k
    let first = seq[..k]

    // Slice from k to end
    let rest = seq[k..]

    // Full slice
    let all = seq[..]

    // Range slice
    let mid = seq[1..4]

    return 0
}

// Test 4: Method calls on arrays (closure syntax)
fn test_methods() -> i32 {
    let arr = [1, 2, 3, 4, 5]

    // map with closure
    let doubled = arr.map(|x| x * 2)

    // reverse
    let rev = arr.reverse()

    // count with predicate
    let evens = arr.count(|x| x % 2 == 0)

    return 0
}

fn main() -> i32 {
    let r1 = test_concat()
    let r2 = test_slices()
    let r3 = test_methods()
    return r1 + r2 + r3
}
