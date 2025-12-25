//@ compile-fail
//@ error-pattern: unit mismatch

use units::{mg, mL, kg}

fn main() {
    let dose: mg = 500.0
    let volume: mL = 10.0

    // This should fail: can't add mg to mL
    let bad = dose + volume
}
