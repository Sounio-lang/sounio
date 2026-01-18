# Run Sounio benchmarks for performance analysis

Run the Sounio compiler benchmarks to measure and compare performance.

## Arguments
- `--bench <name>` - Specific benchmark: layout_bench, ontology_bench, gpu_bench, compiler_bench, sir_gpu_bench, locality_bench
- `--all` - Run all benchmarks
- `--save <file>` - Save results to file for comparison
- `--compare <file>` - Compare against saved baseline

## Examples
- `/sounio-bench --bench layout_bench` - Run layout benchmark
- `/sounio-bench --bench ontology_bench` - Run ontology benchmark
- `/sounio-bench --all` - Run all benchmarks
- `/sounio-bench --bench compiler_bench --save baseline.json` - Save results

$ARGUMENTS

Execute from the `compiler/` directory:

1. Parse arguments to determine which benchmarks to run

2. Available benchmarks:
   - `layout_bench` - Memory layout optimization benchmarks
   - `ontology_bench` - Scientific ontology query benchmarks
   - `gpu_bench` - GPU codegen benchmarks (requires --features gpu)
   - `compiler_bench` - Overall compiler performance
   - `sir_gpu_bench` - SIR to GPU lowering benchmarks
   - `locality_bench` - Cache locality benchmarks

3. Construct and run the cargo bench command:
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo bench --bench <name>
   ```

4. For GPU benchmarks, add `--features gpu`

5. Report benchmark results with timing information
