//@ run-pass
// Issue #11: pointer indexing writes into a u8 buffer.

fn write_byte(ptr: *mut u8, idx: i64, value: u8) {
    ptr[idx] = value
}

fn write_u64_le(ptr: *mut u8, idx: i64, value: i64) {
    let base = idx * 8
    var v = value
    var j: i64 = 0
    while j < 8 {
        ptr[base + j] = (v % 256) as u8
        v = v / 256
        j = j + 1
    }
}

fn read_u64_le(ptr: *const u8, idx: i64) -> i64 {
    let base = idx * 8
    var result: i64 = 0
    var mult: i64 = 1
    var j: i64 = 0
    while j < 8 {
        result = result + (ptr[base + j] as i64) * mult
        mult = mult * 256
        j = j + 1
    }
    return result
}

fn main() -> i64 {
    var buffer: [u8; 64] = [0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0]

    let ptr = (&buffer[0]) as *mut u8
    write_byte(ptr, 0, 42)
    if buffer[0] != 42 { return -1 }

    write_u64_le(ptr, 1, 12345678)
    let read_back = read_u64_le(ptr as *const u8, 1)
    if read_back != 12345678 { return -2 }

    write_byte(ptr, 16, 1)
    write_byte(ptr, 17, 2)
    write_byte(ptr, 18, 3)
    if buffer[16] != 1 { return -3 }
    if buffer[17] != 2 { return -4 }
    if buffer[18] != 3 { return -5 }

    var i: i64 = 0
    while i < 8 {
        ptr[24 + i] = (i + 10) as u8
        i = i + 1
    }
    var sum: i64 = 0
    i = 0
    while i < 8 {
        sum = sum + buffer[24 + i] as i64
        i = i + 1
    }
    if sum != 108 { return -6 }

    // Number of passing sub-tests.
    return 4
}
