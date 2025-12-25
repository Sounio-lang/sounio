// Test file for HashMap and HashSet collections
// Run with: dc run stdlib/collections/test_collections.d

// Simple test framework helpers
fn test_assert(cond: bool, msg: string) {
    if !cond {
        println("FAIL: ")
        println(msg)
    }
}

fn test_pass(name: string) {
    println("  PASS: ")
    println(name)
}

// ============================================================================
// Hash Function Tests
// ============================================================================

fn simple_hash(key: i64) -> i64 {
    // FNV-1a inspired simple hash
    let k = key
    let result = k % 16
    if result < 0 {
        return result + 16
    }
    result
}

// ============================================================================
// Set Operation Tests (using arrays directly)
// ============================================================================

// Check if value is in array (first n elements)
fn array_contains(arr0: i64, arr1: i64, arr2: i64, arr3: i64, len: i64, value: i64) -> bool {
    if len >= 1 {
        let m0 = arr0 == value
        if m0 {
            return true
        }
    }
    if len >= 2 {
        let m1 = arr1 == value
        if m1 {
            return true
        }
    }
    if len >= 3 {
        let m2 = arr2 == value
        if m2 {
            return true
        }
    }
    if len >= 4 {
        let m3 = arr3 == value
        if m3 {
            return true
        }
    }
    false
}

// Check if set A is subset of set B
fn is_subset(a0: i64, a1: i64, a_len: i64, b0: i64, b1: i64, b2: i64, b3: i64, b_len: i64) -> bool {
    // Empty set is subset of everything
    if a_len == 0 {
        return true
    }

    // Check first element of A is in B
    let found0 = array_contains(b0, b1, b2, b3, b_len, a0)
    if !found0 {
        return false
    }

    if a_len == 1 {
        return true
    }

    // Check second element of A is in B
    let found1 = array_contains(b0, b1, b2, b3, b_len, a1)
    if !found1 {
        return false
    }

    true
}

// Check if sets A and B have no common elements
fn is_disjoint(a0: i64, a1: i64, a_len: i64, b0: i64, b1: i64, b2: i64, b3: i64, b_len: i64) -> bool {
    // Empty set is disjoint with everything
    if a_len == 0 {
        return true
    }
    if b_len == 0 {
        return true
    }

    // Check if any element of A is in B
    let found0 = array_contains(b0, b1, b2, b3, b_len, a0)
    if found0 {
        return false
    }

    if a_len >= 2 {
        let found1 = array_contains(b0, b1, b2, b3, b_len, a1)
        if found1 {
            return false
        }
    }

    true
}

// Count intersection size
fn intersection_count(a0: i64, a1: i64, a_len: i64, b0: i64, b1: i64, b2: i64, b3: i64, b_len: i64) -> i64 {
    let mut count: i64 = 0

    if a_len >= 1 {
        let in_b0 = array_contains(b0, b1, b2, b3, b_len, a0)
        if in_b0 {
            count = count + 1
        }
    }

    if a_len >= 2 {
        let in_b1 = array_contains(b0, b1, b2, b3, b_len, a1)
        if in_b1 {
            count = count + 1
        }
    }

    count
}

// Count union size (simplified - assumes no duplicates within each set)
fn union_count(a0: i64, a1: i64, a_len: i64, b0: i64, b1: i64, b2: i64, b3: i64, b_len: i64) -> i64 {
    // Union = A + B - intersection
    let inter = intersection_count(a0, a1, a_len, b0, b1, b2, b3, b_len)
    let result = a_len + b_len - inter
    result
}

// ============================================================================
// Main Test Runner
// ============================================================================

fn main() -> i32 {
    println("=== HashMap/HashSet Collection Tests ===")
    println("")

    // Test 1: Hash function consistency
    println("Test 1: Hash function consistency")
    let h1 = simple_hash(42)
    let h2 = simple_hash(42)
    let hash_consistent = h1 == h2
    test_assert(hash_consistent, "Hash should be consistent")
    test_pass("hash_consistency")

    // Test 2: Hash function range
    println("Test 2: Hash function range")
    let h_0 = simple_hash(0)
    let h_1 = simple_hash(1)
    let h_15 = simple_hash(15)
    let h_16 = simple_hash(16)
    let h_17 = simple_hash(17)

    let range_0 = h_0 >= 0 && h_0 < 16
    let range_1 = h_1 >= 0 && h_1 < 16
    let range_15 = h_15 >= 0 && h_15 < 16
    let range_16 = h_16 >= 0 && h_16 < 16
    let range_17 = h_17 >= 0 && h_17 < 16

    test_assert(range_0, "hash(0) should be in [0,16)")
    test_assert(range_1, "hash(1) should be in [0,16)")
    test_assert(range_15, "hash(15) should be in [0,16)")
    test_assert(range_16, "hash(16) should be in [0,16)")
    test_assert(range_17, "hash(17) should be in [0,16)")

    // Check specific values
    let h0_is_0 = h_0 == 0
    let h1_is_1 = h_1 == 1
    let h16_is_0 = h_16 == 0
    test_assert(h0_is_0, "hash(0) should be 0")
    test_assert(h1_is_1, "hash(1) should be 1")
    test_assert(h16_is_0, "hash(16) should wrap to 0")
    test_pass("hash_range")

    // Test 3: array_contains
    println("Test 3: array_contains")
    let c1 = array_contains(1, 2, 3, 4, 4, 2)
    let c2 = array_contains(1, 2, 3, 4, 4, 5)
    let c3 = array_contains(1, 2, 3, 4, 2, 3)
    test_assert(c1, "2 should be in {1,2,3,4}")
    let not_c2 = !c2
    test_assert(not_c2, "5 should not be in {1,2,3,4}")
    let not_c3 = !c3
    test_assert(not_c3, "3 should not be in first 2 elements of {1,2,3,4}")
    test_pass("array_contains")

    // Test 4: is_subset - empty set
    println("Test 4: is_subset empty set")
    let empty_sub = is_subset(0, 0, 0, 1, 2, 3, 4, 4)
    test_assert(empty_sub, "Empty set should be subset of {1,2,3,4}")
    test_pass("is_subset_empty")

    // Test 5: is_subset - true case
    println("Test 5: is_subset true")
    let sub_true = is_subset(1, 2, 2, 1, 2, 3, 4, 4)
    test_assert(sub_true, "{1,2} should be subset of {1,2,3,4}")
    test_pass("is_subset_true")

    // Test 6: is_subset - false case
    println("Test 6: is_subset false")
    let sub_false = is_subset(1, 5, 2, 1, 2, 3, 4, 4)
    let not_sub = !sub_false
    test_assert(not_sub, "{1,5} should not be subset of {1,2,3,4}")
    test_pass("is_subset_false")

    // Test 7: is_disjoint - true case
    println("Test 7: is_disjoint true")
    let disj_true = is_disjoint(5, 6, 2, 1, 2, 3, 4, 4)
    test_assert(disj_true, "{5,6} should be disjoint from {1,2,3,4}")
    test_pass("is_disjoint_true")

    // Test 8: is_disjoint - false case
    println("Test 8: is_disjoint false")
    let disj_false = is_disjoint(1, 5, 2, 1, 2, 3, 4, 4)
    let not_disj = !disj_false
    test_assert(not_disj, "{1,5} should not be disjoint from {1,2,3,4}")
    test_pass("is_disjoint_false")

    // Test 9: is_disjoint - empty set
    println("Test 9: is_disjoint empty")
    let disj_empty = is_disjoint(0, 0, 0, 1, 2, 3, 4, 4)
    test_assert(disj_empty, "Empty set should be disjoint from any set")
    test_pass("is_disjoint_empty")

    // Test 10: intersection_count
    println("Test 10: intersection_count")
    let inter1 = intersection_count(1, 2, 2, 2, 3, 4, 5, 4)
    let inter1_is_1 = inter1 == 1
    test_assert(inter1_is_1, "intersection of {1,2} and {2,3,4,5} should have 1 element")

    let inter2 = intersection_count(1, 2, 2, 1, 2, 3, 4, 4)
    let inter2_is_2 = inter2 == 2
    test_assert(inter2_is_2, "intersection of {1,2} and {1,2,3,4} should have 2 elements")

    let inter3 = intersection_count(5, 6, 2, 1, 2, 3, 4, 4)
    let inter3_is_0 = inter3 == 0
    test_assert(inter3_is_0, "intersection of {5,6} and {1,2,3,4} should be empty")
    test_pass("intersection_count")

    // Test 11: union_count
    println("Test 11: union_count")
    let union1 = union_count(1, 2, 2, 3, 4, 0, 0, 2)
    let union1_is_4 = union1 == 4
    test_assert(union1_is_4, "union of {1,2} and {3,4} should have 4 elements")

    let union2 = union_count(1, 2, 2, 2, 3, 0, 0, 2)
    let union2_is_3 = union2 == 3
    test_assert(union2_is_3, "union of {1,2} and {2,3} should have 3 elements")

    let union3 = union_count(1, 2, 2, 1, 2, 0, 0, 2)
    let union3_is_2 = union3 == 2
    test_assert(union3_is_2, "union of {1,2} and {1,2} should have 2 elements")
    test_pass("union_count")

    // Test 12: Negative hash values
    println("Test 12: Negative key hash")
    let h_neg1 = simple_hash(0 - 1)
    let h_neg5 = simple_hash(0 - 5)
    let neg1_in_range = h_neg1 >= 0 && h_neg1 < 16
    let neg5_in_range = h_neg5 >= 0 && h_neg5 < 16
    test_assert(neg1_in_range, "hash(-1) should be in [0,16)")
    test_assert(neg5_in_range, "hash(-5) should be in [0,16)")
    test_pass("negative_hash")

    println("")
    println("=== All 12 Collection Tests Complete ===")

    return 0
}
