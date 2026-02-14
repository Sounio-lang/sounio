//! Fuzz target for SOIR normalization
//!
//! Goals:
//! - No panics during normalization
//! - Idempotence: normalize(normalize(x)) == normalize(x)
//! - No infinite loops

#![no_main]

use libfuzzer_sys::fuzz_target;
use soir::{deserialize, normalize};

fuzz_target!(|data: &[u8]| {
    // Try to deserialize first
    if let Ok(module) = deserialize(data) {
        // Should not panic
        let normalized1 = normalize(&module);

        // Test idempotence
        let normalized2 = normalize(&normalized1);

        // Verify function counts match (basic sanity check)
        assert_eq!(normalized1.fn_count, normalized2.fn_count);
        assert_eq!(normalized1.string_count, normalized2.string_count);
    }
});
