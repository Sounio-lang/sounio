// Test raw pointer types and FFI support

// Test raw pointer type parsing
fn test_pointer_types() {
    // Create null pointers (use type inference)
    let const_ptr = null_ptr()
    let mut_ptr = null_mut()

    // Check null status
    let is_const_null = is_null(const_ptr)
    let is_mut_null = is_null(mut_ptr)

    print("const_ptr is null: ")
    print(is_const_null)
    print("\n")

    print("mut_ptr is null: ")
    print(is_mut_null)
    print("\n")

    // Test pointer comparison
    let ptrs_equal = ptr_eq(const_ptr, const_ptr)
    print("ptr_eq(const_ptr, const_ptr): ")
    print(ptrs_equal)
    print("\n")
}

// Test pointer creation from addresses
fn test_pointer_from_address() {
    // Create pointer from address (use decimal instead of hex)
    let addr: i64 = 4096
    let ptr = ptr_from_addr(addr)

    // Get address back
    let retrieved_addr = ptr_addr(ptr)
    print("Created pointer at address: ")
    print(retrieved_addr)
    print("\n")

    // Test pointer arithmetic
    let offset_ptr = ptr_offset(ptr, 4)
    let offset_addr = ptr_addr(offset_ptr)
    print("After offset by 4: ")
    print(offset_addr)
    print("\n")
}

// Test pointer casting
fn test_pointer_casting() {
    let mut_ptr = ptr_from_addr_mut(8192)

    // Cast mutable to const
    let const_ptr = as_const(mut_ptr)

    print("Cast *mut to *const - is_null: ")
    print(is_null(const_ptr))
    print("\n")
}

// Test extern block parsing (C FFI declarations)
extern "C" {
    fn malloc(size: i64) -> *mut i8;
    fn free(ptr: *mut i8);
    fn strlen(s: *const i8) -> i64;
    fn memcpy(dest: *mut i8, src: *const i8, n: i64) -> *mut i8;
}

fn main() {
    print("=== Raw Pointer FFI Tests ===\n\n")

    print("--- Test 1: Pointer Types ---\n")
    test_pointer_types()

    print("\n--- Test 2: Pointer from Address ---\n")
    test_pointer_from_address()

    print("\n--- Test 3: Pointer Casting ---\n")
    test_pointer_casting()

    print("\n=== All FFI tests completed ===\n")
}
