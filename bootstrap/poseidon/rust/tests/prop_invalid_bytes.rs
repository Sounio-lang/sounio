mod common;

use poseidon_wrapper::{Module, PoseidonError, VmConfig};
use proptest::prelude::*;

#[test]
fn truncated_valid_fixture_returns_structured_errors() {
    for fixture in common::fixtures_with_tag("proptest_seed") {
        let bytes = common::read_fixture_bytes(&fixture);
        for prefix_len in 0..bytes.len() {
            let prefix = &bytes[..prefix_len];
            match Module::from_bytes(prefix) {
                Ok(module) => {
                    if prefix_len != bytes.len() {
                        let _ = module.run(&VmConfig { max_steps: 8 });
                    }
                }
                Err(err) => {
                    assert!(
                        matches!(
                            err,
                            PoseidonError::InvalidArgument
                                | PoseidonError::InvalidBytecode
                                | PoseidonError::UnsupportedVersion
                        ),
                        "fixture {} prefix_len {} unexpected error {:?}",
                        fixture.id,
                        prefix_len,
                        err
                    );
                }
            }
        }
    }
}

proptest! {
    #[test]
    fn arbitrary_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..256)) {
        match Module::from_bytes(&bytes) {
            Ok(module) => {
                let _ = module.run(&VmConfig { max_steps: 32 });
            }
            Err(err) => {
                prop_assert!(matches!(
                    err,
                    PoseidonError::InvalidArgument
                        | PoseidonError::IoError
                        | PoseidonError::FileTooLarge
                        | PoseidonError::InvalidBytecode
                        | PoseidonError::UnsupportedVersion
                        | PoseidonError::OutOfMemory
                        | PoseidonError::ExecutionError
                        | PoseidonError::ExecutionTimeout
                        | PoseidonError::NoMainFunction
                ));
            }
        }
    }
}
