#![no_main]

use libfuzzer_sys::fuzz_target;
use poseidon_wrapper::VmConfig;

fuzz_target!(|data: &[u8]| {
    if let Ok(module) = poseidon_wrapper::Module::from_bytes(data) {
        let _ = module.run(&VmConfig { max_steps: 32 });
    }
});
