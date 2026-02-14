use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let poseidon_dir = PathBuf::from(&manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("bootstrap/poseidon");

    // Compile C VM sources
    cc::Build::new()
        .file(poseidon_dir.join("vm.c"))
        .file(poseidon_dir.join("loader.c"))
        .file(poseidon_dir.join("runtime.c"))
        .include(&poseidon_dir)
        .warnings(true)
        .extra_warnings(true)
        .flag_if_supported("-std=c99")
        .flag_if_supported("-Wno-unused-parameter")
        .compile("poseidon");

    // Tell cargo to rerun if C sources change
    println!("cargo:rerun-if-changed={}/vm.c", poseidon_dir.display());
    println!("cargo:rerun-if-changed={}/vm.h", poseidon_dir.display());
    println!("cargo:rerun-if-changed={}/loader.c", poseidon_dir.display());
    println!("cargo:rerun-if-changed={}/loader.h", poseidon_dir.display());
    println!("cargo:rerun-if-changed={}/runtime.c", poseidon_dir.display());
    println!("cargo:rerun-if-changed={}/runtime.h", poseidon_dir.display());
    println!("cargo:rerun-if-changed={}/opcodes.h", poseidon_dir.display());
    println!("cargo:rerun-if-changed={}/platform.h", poseidon_dir.display());
}
