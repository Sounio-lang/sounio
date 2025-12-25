//@ run-pass
// Algebraic effects test

fn pure_computation(x: i32) -> i32 {
    x * 2
}

fn effectful_computation() -> i32 with IO {
    print("computing...\n")
    42
}

fn main() with IO {
    let result = pure_computation(21)
    // Check result is 42
    if result == 42 {
        print("pure_computation: PASS\n")
    } else {
        print("pure_computation: FAIL\n")
    }

    let data = effectful_computation()
    if data == 42 {
        print("effectful_computation: PASS\n")
    } else {
        print("effectful_computation: FAIL\n")
    }
}
