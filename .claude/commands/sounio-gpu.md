# GPU kernel compilation and testing

Compile GPU kernels and manage GPU-accelerated Sounio code targeting CUDA (PTX) or Vulkan (SPIR-V).

## Arguments
- `--build <file>` - Build GPU kernels from file
- `--emit <ptx|spirv>` - Emit GPU intermediate representation
- `--target <cuda|vulkan|metal>` - Target GPU API (default: cuda)
- `--test` - Run GPU test suite
- `--profile` - Profile GPU kernel execution
- `--list-devices` - List available GPU devices
- `--device <id>` - Select specific GPU device

## Examples
- `/sounio-gpu --build examples/gpu.sio --emit ptx` - Compile to PTX
- `/sounio-gpu --build examples/matmul.sio --emit spirv` - Compile to SPIR-V
- `/sounio-gpu --test` - Run GPU tests
- `/sounio-gpu --list-devices` - Show available GPUs
- `/sounio-gpu --profile examples/neural.sio` - Profile execution

$ARGUMENTS

Execute from the `compiler/` directory with GPU features:

```bash
cd /home/demetrios/sounio-1/compiler && cargo run --features gpu -- gpu <subcommand>
```

For kernel compilation:
```bash
cd /home/demetrios/sounio-1/compiler && cargo run --features gpu -- build <file> --emit-gpu ptx
```

## GPU Kernel Syntax

```sio
// Vector addition kernel
kernel fn vector_add(a: &[f32], b: &[f32], c: &![f32]) {
    let i = gpu.thread_id.x
    if i < a.len() {
        c[i] = a[i] + b[i]
    }
}

// Matrix multiplication with shared memory
kernel fn matmul(
    a: &[[f32; N]; M],
    b: &[[f32; P]; N],
    c: &![[f32; P]; M]
) {
    shared var tile_a: [[f32; TILE]; TILE]
    shared var tile_b: [[f32; TILE]; TILE]

    let row = gpu.block_id.y * TILE + gpu.thread_id.y
    let col = gpu.block_id.x * TILE + gpu.thread_id.x
    // ... tiled multiplication logic
}

// Launch configuration
let result = gpu.launch(vector_add, grid: (n/256, 1, 1), block: (256, 1, 1))(a, b, c)
```

## GPU Built-ins

**Thread identification:**
- `gpu.thread_id` - Thread ID within block (x, y, z)
- `gpu.block_id` - Block ID within grid (x, y, z)
- `gpu.block_dim` - Block dimensions
- `gpu.grid_dim` - Grid dimensions

**Memory qualifiers:**
- `shared` - Shared memory within block
- `constant` - Constant memory (read-only, cached)

**Synchronization:**
- `gpu.sync_threads()` - Block-level barrier
- `gpu.atomic_add()`, `gpu.atomic_max()`, etc.

## Target Backends

| Target | Output | Requirements |
|--------|--------|--------------|
| CUDA | PTX | NVIDIA GPU, CUDA toolkit |
| Vulkan | SPIR-V | Vulkan-capable GPU |
| Metal | Metal IR | Apple GPU (macOS/iOS) |

## Running GPU Tests

```bash
cd /home/demetrios/sounio-1/compiler && cargo test --features gpu --test 'gpu_*'
```

Tests verify:
- Kernel compilation correctness
- PTX/SPIR-V generation
- Runtime execution (if GPU available)
- Memory management
