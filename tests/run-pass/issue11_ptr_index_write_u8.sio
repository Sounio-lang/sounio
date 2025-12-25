//@ run-pass
// Issue #11: u8 pointer indexing writes.

fn write_byte(ptr: *mut u8, idx: i64, value: u8) {
    ptr[idx] = value
}

fn main() -> i64 {
    var arr: [u8; 8] = [0, 0, 0, 0, 0, 0, 0, 0]

    let ptr = (&arr[0]) as *mut u8

    write_byte(ptr, 0, 1)
    write_byte(ptr, 1, 2)
    write_byte(ptr, 2, 3)
    write_byte(ptr, 3, 4)

    return arr[0] as i64 + arr[1] as i64 + arr[2] as i64 + arr[3] as i64
}
