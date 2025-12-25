// Simple test for raw pointer FFI builtins

fn main() {
    print("Testing raw pointer FFI builtins\n")

    // Test null_ptr
    let ptr = null_ptr()
    print("Created null pointer\n")

    // Test is_null
    let result = is_null(ptr)
    print("is_null(ptr) = ")
    print(result)
    print("\n")

    // Test ptr_from_addr
    let addr: i64 = 1000
    let ptr2 = ptr_from_addr(addr)
    print("Created pointer from address 1000\n")

    // Test ptr_addr
    let retrieved = ptr_addr(ptr2)
    print("ptr_addr(ptr2) = ")
    print(retrieved)
    print("\n")

    // Test ptr_eq
    let eq_result = ptr_eq(ptr, ptr)
    print("ptr_eq(ptr, ptr) = ")
    print(eq_result)
    print("\n")

    print("All tests done!\n")
}
