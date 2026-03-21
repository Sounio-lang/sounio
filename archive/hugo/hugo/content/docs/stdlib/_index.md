---
title: "Standard Library"
description: "API reference for Sounio's 76 standard library modules"
weight: 3
---

# Standard Library Reference

Sounio's standard library provides 76 modules covering scientific computing, uncertainty quantification, I/O, collections, and GPU primitives.

## Core Modules

### `std::prelude`

Automatically imported types and functions.

```sio
// Always available without explicit import
let x: i32 = 42
let s: string = "hello"
let arr: [f64; 3] = [1.0, 2.0, 3.0]
```

### `std::result`

Error handling with `Result<T, E>` type.

```sio
use std::result::*

fn parse_number(s: string) -> Result<i32, ParseError> {
    // ...
}

let value = parse_number("42")?  // Propagates error
```

### `std::option`

Optional values with `Option<T>`.

```sio
use std::option::*

let maybe: Option<i32> = Some(42)
let nothing: Option<i32> = None

match maybe {
    Some(x) => print(x),
    None => print("nothing")
}
```

---

## Epistemic Computing

### `std::knowledge`

The `Knowledge<T>` type for uncertainty-aware values.

```sio
use std::knowledge::*

// Create from value and uncertainty (GUM Type B)
let mass: Knowledge<kg> = Knowledge::new(
    value: 500.0,
    std_uncertainty: 2.5,
    confidence: 0.95
)

// Create from statistical samples (GUM Type A)
let measurements = [10.2, 10.1, 10.3, 10.0, 10.2]
let mean: Knowledge<f64> = Knowledge::from_samples(measurements)
// Computes: value = mean, uncertainty = s/√n

// Exact values (zero uncertainty)
let constant: Knowledge<f64> = Knowledge::exact(3.14159)

// Arithmetic propagates uncertainty automatically
let dose = mass / volume  // GUM-compliant propagation
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `value()` | Get the central value |
| `std_uncertainty()` | Standard uncertainty (1σ) |
| `rel_uncertainty()` | Relative uncertainty (u/x) |
| `ci_95()` | 95% confidence interval |
| `ci_99()` | 99% confidence interval |
| `expanded_uncertainty(k)` | U = k × u |

### `std::provenance`

Audit trails for regulatory compliance.

```sio
use std::provenance::*

let measurement = Knowledge::new(value: 500.0, std_uncertainty: 2.5, confidence: 0.95)
    .with_provenance(
        source: "Lab-A Calibrated Scale",
        timestamp: now(),
        operator: "J. Smith",
        certificate: "CAL-2024-0123"
    )

// Provenance propagates through calculations
let result = measurement * 2.0
print(result.provenance().sources())      // ["Lab-A Calibrated Scale"]
print(result.provenance().merkle_hash())  // Tamper-evident hash
```

### `std::ensemble`

Multi-model ensemble methods.

```sio
use std::ensemble::*

let models = [model_a, model_b, model_c, model_d]

// Simple mean (includes between-model variance)
let mean = Knowledge::ensemble_mean(models)

// Weighted by inverse variance
let weighted = Knowledge::ensemble_weighted(models)

// Bayesian Model Averaging
let bma = Knowledge::bma(models, weights: [0.3, 0.3, 0.2, 0.2])
```

---

## Units of Measure

### `std::units::si`

SI base and derived units with compile-time checking.

```sio
use std::units::si::*

// Base units
let mass: kg = 1.5
let length: m = 100.0
let time: s = 9.58
let current: A = 2.5
let temperature: K = 300.0
let amount: mol = 0.5
let intensity: cd = 100.0

// Derived units (automatically computed)
let velocity: m/s = length / time      // 10.44 m/s
let force: N = mass * (velocity / time) // kg⋅m/s²
let energy: J = force * length          // N⋅m = kg⋅m²/s²
let power: W = energy / time            // J/s
let pressure: Pa = force / (length * length)  // N/m²
```

### `std::units::pharma`

Pharmaceutical and medical units.

```sio
use std::units::pharma::*

let dose: mg = 500.0
let volume: mL = 250.0
let concentration: mg/mL = dose / volume  // 2.0 mg/mL

let infusion_rate: mL/h = 100.0
let duration: h = volume / infusion_rate  // 2.5 h

// Dosing by body weight
let patient_weight: kg = 70.0
let dose_per_kg: mg/kg = 10.0
let total_dose: mg = dose_per_kg * patient_weight  // 700 mg
```

### `std::units::convert`

Unit conversions.

```sio
use std::units::convert::*

let temp_c: degC = 25.0
let temp_k: K = temp_c.to_kelvin()     // 298.15 K
let temp_f: degF = temp_c.to_fahrenheit() // 77.0 °F

let mass_kg: kg = 2.5
let mass_lb: lb = mass_kg.to_pounds()  // 5.51 lb

let volume_L: L = 1.0
let volume_gal: gal = volume_L.to_gallons()  // 0.264 gal
```

---

## Collections

### `std::vec`

Dynamic arrays.

```sio
use std::vec::*

var v: Vec<i32> = Vec::new()
v.push(1)
v.push(2)
v.push(3)

let first = v[0]        // 1
let len = v.len()       // 3
let last = v.pop()      // Some(3)

// Iteration
for x in v {
    print(x)
}
```

### `std::hashmap`

Hash-based key-value storage.

```sio
use std::hashmap::*

var map: HashMap<string, i32> = HashMap::new()
map.insert("alice", 30)
map.insert("bob", 25)

let age = map.get("alice")  // Some(30)
let missing = map.get("charlie")  // None

for (key, value) in map {
    print(key, ":", value)
}
```

### `std::slice`

Borrowed views into arrays.

```sio
use std::slice::*

let arr = [1, 2, 3, 4, 5]
let head = arr[..3]   // [1, 2, 3]
let tail = arr[2..]   // [3, 4, 5]
let mid = arr[1..4]   // [2, 3, 4]

// Darwin Atlas concatenation
let combined = head ++ tail  // [1, 2, 3, 3, 4, 5]
```

---

## Mathematics

### `std::math`

Core mathematical functions.

```sio
use std::math::*

let x = sqrt(2.0)      // 1.414...
let y = sin(PI / 4.0)  // 0.707...
let z = exp(-1.0)      // 0.368...
let w = ln(E)          // 1.0

let a = abs(-5)        // 5
let b = max(3, 7)      // 7
let c = min(3, 7)      // 3
let d = clamp(x, 0.0, 1.0)  // Constrain to range
```

### `std::linalg`

Linear algebra primitives.

```sio
use std::linalg::*

let a: Matrix<f64, 2, 3> = [[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]
let b: Matrix<f64, 3, 2> = [[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]]
let c = a @ b  // Matrix multiplication

let v: Vector<f64, 3> = [1.0, 2.0, 3.0]
let norm = v.norm()     // Euclidean norm
let unit = v.normalize() // Unit vector
let dot = v.dot(v)      // Dot product
```

### `std::complex`

Complex number arithmetic.

```sio
use std::complex::*

let z1: Complex<f64> = Complex::new(3.0, 4.0)  // 3 + 4i
let z2: Complex<f64> = Complex::polar(5.0, PI/4.0)  // r⋅e^(iθ)

let sum = z1 + z2
let product = z1 * z2
let conj = z1.conj()     // 3 - 4i
let mag = z1.abs()       // 5.0
let arg = z1.arg()       // atan2(4, 3)
```

### `std::quaternion`

Quaternion algebra for rotations.

```sio
use std::quaternion::*

let q1: Quaternion<f64> = Quaternion::new(1.0, 0.0, 0.0, 0.0)
let q2 = Quaternion::from_axis_angle([0.0, 0.0, 1.0], PI/2.0)

let q3 = q1 * q2         // Hamilton product
let qn = q3.normalize()  // Unit quaternion
let qi = q3.inverse()    // Inverse

// Rotate a vector
let v = [1.0, 0.0, 0.0]
let rotated = q2.rotate(v)  // [0.0, 1.0, 0.0]
```

### `std::octonion`

Octonion algebra for advanced applications.

```sio
use std::octonion::*

let o1: Octonion<f64> = Octonion::new(
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
)
let o2 = Octonion::unit(3)  // e₃ basis element

let product = oct_mul(o1, o2)  // Non-associative!
let norm = o1.norm()
let conj = o1.conj()

// Moufang identity verification
// z(x(zy)) = ((zx)z)y
```

---

## I/O

### `std::io`

File and console I/O.

```sio
use std::io::*

fn main() with IO {
    // Console
    print("Hello, world!")
    let input = readline()

    // File reading
    let content = read_file("data.txt")

    // File writing
    write_file("output.txt", content)

    // Append
    append_file("log.txt", "New entry\n")
}
```

### `std::fs`

Filesystem operations.

```sio
use std::fs::*

fn main() with IO {
    let exists = file_exists("data.txt")
    let size = file_size("data.txt")
    let files = list_dir("./data")

    create_dir("./output")
    remove_file("temp.txt")
    rename("old.txt", "new.txt")
}
```

### `std::csv`

CSV parsing and generation.

```sio
use std::csv::*

fn main() with IO {
    // Read CSV with headers
    let data = read_csv("measurements.csv", has_header: true)

    for row in data.rows() {
        let value: f64 = row.get("measurement").parse()
        print(value)
    }

    // Write CSV
    var writer = CsvWriter::new("output.csv")
    writer.write_header(["time", "value", "uncertainty"])
    writer.write_row([0.0, 10.5, 0.3])
    writer.write_row([1.0, 11.2, 0.4])
    writer.close()
}
```

### `std::json`

JSON parsing and serialization.

```sio
use std::json::*

fn main() with IO {
    let text = read_file("config.json")
    let config = Json::parse(text)

    let name = config["name"].as_string()
    let count = config["count"].as_i32()
    let items = config["items"].as_array()

    // Serialize
    let obj = Json::object()
    obj.set("result", 42)
    obj.set("success", true)
    let output = obj.to_string()
}
```

---

## GPU Computing

### `std::gpu`

GPU primitives and kernel support.

```sio
use std::gpu::*

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

    // Launch kernel with grid/block dimensions
    vector_add<<<n/256, 256>>>(a, b, &!c)

    gpu.sync()  // Wait for completion
    let result = gpu.copy_to_host(c)
}
```

### `std::gpu::tensor`

GPU tensor operations.

```sio
use std::gpu::tensor::*

fn main() with GPU {
    let a: Tensor<f32, [2, 3]> = Tensor::from([[1.0, 2.0, 3.0],
                                                [4.0, 5.0, 6.0]])
    let b: Tensor<f32, [3, 2]> = Tensor::from([[1.0, 2.0],
                                                [3.0, 4.0],
                                                [5.0, 6.0]])

    let c = a.matmul(b)  // [2, 2] tensor
    let d = a.transpose()
    let e = a.relu()     // Element-wise ReLU
}
```

---

## Random & Statistics

### `std::random`

Random number generation.

```sio
use std::random::*

var rng = Rng::seed(42)

let uniform = rng.uniform(0.0, 1.0)
let normal = rng.normal(mean: 0.0, std: 1.0)
let integer = rng.range(1, 100)
let choice = rng.choice(["a", "b", "c"])

// Shuffle in place
var arr = [1, 2, 3, 4, 5]
rng.shuffle(&!arr)
```

### `std::stats`

Statistical functions.

```sio
use std::stats::*

let data = [1.0, 2.0, 3.0, 4.0, 5.0]

let m = mean(data)       // 3.0
let s = std_dev(data)    // 1.58...
let v = variance(data)   // 2.5
let med = median(data)   // 3.0
let q1 = percentile(data, 25.0)  // 2.0
let q3 = percentile(data, 75.0)  // 4.0

// Correlation
let x = [1.0, 2.0, 3.0, 4.0]
let y = [2.0, 4.0, 5.0, 4.0]
let r = correlation(x, y)  // Pearson r
```

---

## Module Index

| Module | Description |
|--------|-------------|
| `std::prelude` | Auto-imported basics |
| `std::result` | Error handling |
| `std::option` | Optional values |
| `std::knowledge` | Uncertainty-aware values |
| `std::provenance` | Audit trails |
| `std::ensemble` | Multi-model methods |
| `std::units::si` | SI units |
| `std::units::pharma` | Medical units |
| `std::units::convert` | Unit conversions |
| `std::vec` | Dynamic arrays |
| `std::hashmap` | Key-value maps |
| `std::slice` | Array views |
| `std::math` | Math functions |
| `std::linalg` | Linear algebra |
| `std::complex` | Complex numbers |
| `std::quaternion` | Quaternions |
| `std::octonion` | Octonions |
| `std::io` | File/console I/O |
| `std::fs` | Filesystem |
| `std::csv` | CSV handling |
| `std::json` | JSON handling |
| `std::gpu` | GPU primitives |
| `std::gpu::tensor` | GPU tensors |
| `std::random` | RNG |
| `std::stats` | Statistics |

---

## See Also

- **[API Reference](/docs/api/)** — Compiler and runtime APIs
- **[Language Guide](/docs/language/)** — Complete syntax reference
- **[Examples Gallery](/examples/)** — Runnable code examples
