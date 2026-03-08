fn main() {
    let c_sources = [
        "../poseidon.c",
        "../loader.c",
        "../vm.c",
        "../runtime.c",
    ];

    let mut build = cc::Build::new();
    build.include("..");
    build.warnings(true);
    build.extra_warnings(true);
    for source in c_sources {
        build.file(source);
        println!("cargo:rerun-if-changed={source}");
    }
    println!("cargo:rerun-if-changed=../poseidon.h");
    println!("cargo:rerun-if-changed=../loader.h");
    println!("cargo:rerun-if-changed=../vm.h");
    println!("cargo:rerun-if-changed=../runtime.h");
    build.compile("poseidon");
}
