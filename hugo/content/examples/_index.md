---
title: "Examples Gallery"
description: "Real-world Sounio code examples demonstrating epistemic computing, GPU acceleration, and scientific applications"
weight: 4
---

# Examples Gallery

Explore real-world Sounio code demonstrating epistemic computing, GPU acceleration, and domain-specific applications. All examples are runnable with `souc run`.

## Scientific Computing

### Pharmaceutical Dosing with Uncertainty

Type-safe drug dosing calculation with GUM-compliant uncertainty propagation:

```sio
// Real FDA data: Metformin 500mg tablets (NDA 020357)
let tablet_mass = Knowledge::new(
    value: 500.0,         // mg
    std_uncertainty: 5.0,  // USP ±1% tolerance
    confidence: 0.95
)

let patient_weight = Knowledge::new(
    value: 72.5,          // kg
    std_uncertainty: 0.5,  // Clinical scale precision
    confidence: 0.95
)

// Clark's Rule: Pediatric_dose = (Weight/70) * Adult_dose
let child_weight = Knowledge::new(value: 25.0, std_uncertainty: 0.3, confidence: 0.95)
let child_dose = (child_weight / Knowledge::exact(70.0)) * tablet_mass

// Result: 178.57 mg ± 5.74 mg (95% CI)
print(child_dose.ci_95())
```

### Climate Ensemble Averaging

Multi-model temperature projection with IPCC AR6 uncertainty:

```sio
// CMIP6 models for SSP2-4.5 at 2081-2100
let cesm2 = Knowledge::new(value: 2.8, std_uncertainty: 0.3, confidence: 0.90)
let gfdl_esm4 = Knowledge::new(value: 2.4, std_uncertainty: 0.25, confidence: 0.90)
let ukesm1 = Knowledge::new(value: 3.5, std_uncertainty: 0.35, confidence: 0.90)
let miroc6 = Knowledge::new(value: 2.6, std_uncertainty: 0.28, confidence: 0.90)

// Bayesian Model Averaging
let models = [cesm2, gfdl_esm4, ukesm1, miroc6]
let ensemble = Knowledge::ensemble_mean(models)

// Result: 2.8°C ± 0.6°C (includes between-model variance)
print("Global warming projection: ", ensemble)
```

## GPU Computing

### Vector Addition Kernel

Basic GPU kernel demonstrating Sounio's syntax:

```sio
kernel fn vector_add(a: &[f32], b: &[f32], c: &![f32]) with GPU {
    let i = gpu.thread_id.x
    if i < a.len() {
        c[i] = a[i] + b[i]
    }
}

fn main() with IO, GPU {
    let n = 1_000_000
    let a = gpu.alloc([1.0; n])
    let b = gpu.alloc([2.0; n])
    var c = gpu.alloc([0.0; n])

    vector_add<<<n/256, 256>>>(a, b, &!c)

    // Result: c[i] = 3.0 for all i
}
```

### Octonion Multiplication on GPU

GPU-accelerated hypercomplex algebra:

```sio
kernel fn octonion_multiply(
    a: &[Octonion],
    b: &[Octonion],
    c: &![Octonion]
) with GPU {
    let i = gpu.thread_id.x
    if i < a.len() {
        // 120 FLOPs per multiplication (64 muls + 56 adds)
        c[i] = oct_mul(a[i], b[i])
    }
}

// Benchmark: 142.7 GFLOPS on RTX 4090
```

## Uncertainty Quantification

### GUM-Compliant Measurement

Following ISO/IEC Guide 98-3:2008:

```sio
// Type A uncertainty (statistical)
let measurements = [10.2, 10.1, 10.3, 10.0, 10.2]
let type_a = Knowledge::from_samples(measurements)
// u(x) = s/√n = 0.11 / √5 = 0.049

// Type B uncertainty (instrument specification)
let instrument_accuracy: f64 = 0.05  // ±0.05 from calibration certificate
let type_b = instrument_accuracy / sqrt(3.0)  // Rectangular distribution

// Combined uncertainty
let combined_u = sqrt(type_a.std_uncertainty.powi(2) + type_b.powi(2))

// Expanded uncertainty (k=2 for 95% coverage)
let U_95 = 2.0 * combined_u
```

### Provenance Tracking

Full audit trail for regulatory compliance:

```sio
let measurement = Knowledge::new(
    value: 500.0,
    std_uncertainty: 2.5,
    confidence: 0.95
).with_provenance(
    source: "Lab-A Calibrated Scale",
    timestamp: now(),
    operator: "J. Smith",
    certificate: "CAL-2024-0123"
)

// Provenance propagates through calculations
let result = measurement * 2.0
print(result.provenance().sources())  // ["Lab-A Calibrated Scale"]
print(result.provenance().merkle_hash())  // Tamper-evident hash
```

## Linear Types

### Safe Resource Management

RAII with compile-time enforcement:

```sio
linear struct FileHandle {
    fd: i32
}

fn open(path: string) -> FileHandle with IO {
    FileHandle { fd: sys_open(path) }
}

fn read(handle: &FileHandle) -> string with IO {
    sys_read(handle.fd)
}

fn close(handle: FileHandle) with IO {
    sys_close(handle.fd)
    // handle is consumed, cannot be used again
}

fn main() with IO {
    let file = open("data.txt")
    let content = read(&file)
    close(file)  // Must be called exactly once
    // close(file)  // Error: file already consumed
}
```

## Units of Measure

### Dimensional Analysis

Compile-time unit checking prevents errors:

```sio
// SI base units
let mass: kg = 1.5
let distance: m = 100.0
let time: s = 9.58

// Derived units computed automatically
let velocity: m/s = distance / time   // 10.44 m/s
let energy: J = 0.5 * mass * velocity * velocity  // Kinetic energy

// Unit conversions
let energy_kwh: kWh = energy.convert()

// Compile-time error: incompatible units
// let bad: kg = distance  // Error: cannot assign m to kg
```

## Running Examples

All examples can be run with:

```bash
# Check types without execution
souc check example.sio

# Compile and run (JIT)
souc run example.sio

# Compile to native binary
souc build example.sio -o example
./example

# GPU examples require --features gpu
souc run --features gpu gpu_example.sio
```

## More Examples

- **[GitHub Examples Directory](https://github.com/sounio-lang/sounio/tree/main/examples)** — 315+ examples
- **[Interactive Playground](https://sounio-lang.github.io/playground)** — Try examples in your browser
