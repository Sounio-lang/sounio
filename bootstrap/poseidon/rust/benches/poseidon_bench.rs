use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use poseidon_wrapper::{Module, VmConfig};

#[path = "../tests/common/mod.rs"]
mod common;

fn bench_load_execute(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_execute");
    for (name, bytes) in common::bench_fixtures() {
        group.bench_with_input(BenchmarkId::from_parameter(name), &bytes, |b, fixture| {
            b.iter(|| {
                let module = Module::from_bytes(black_box(fixture.as_slice())).expect("module");
                let result = module.run(&VmConfig::default()).expect("run");
                black_box(result.exit_code);
                black_box(result.steps_executed);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_load_execute);
criterion_main!(benches);
