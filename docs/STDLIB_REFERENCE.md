# Sounio Standard Library Reference (February 2026)

Comprehensive analysis of all 76 stdlib modules with implementation status, LOC, and usage guidance.

---

## Overview

| Metric | Value |
|--------|-------|
| **Total .sio files** | 512 |
| **Total modules** | 76 |
| **Total LOC** | ~215,725 |
| **Production-ready modules** | 20+ |
| **Stub/minimal modules** | 12 |

---

## All Modules (49 Total)

| Module | Purpose | Status |
|--------|---------|--------|
| **async** | Async/await runtime, futures, channels, timers | Fully Specified |
| **autodiff** | Forward-mode AD via dual numbers, reverse-mode | Implemented |
| **bayes** | Bayesian inference (MCMC, VI, diagnostics) | Core Implemented |
| **causal** | Pearl's do-calculus with epistemic uncertainty | Implemented |
| **cmp** | Comparison traits (Eq, Ord, PartialEq, PartialOrd) | Implemented |
| **collections** | Vec, HashMap, HashSet, Deque | Fully Implemented |
| **connectivity** | Brain connectivity metrics (fMRI networks) | Minimal Stub (7 lines) |
| **core** | Option<T>, Result<T, E> | Fully Implemented |
| **csv** | CSV parsing and serialization | Minimal |
| **darwin** | Darwin Atlas kernels (sequence symmetries) | Implemented |
| **darwin_pbpk** | PBPK simulation engines (Rodgers-Rowland, TSit5) | Heavily Implemented |
| **data** | DataFrames, Series, I/O operations | Partially Implemented |
| **epistemic** | Epistemic core (Knowledge, Confidence, Provenance) | Heavily Implemented (30+ files) |
| **ffi** | Foreign Function Interface, C interop | Fully Specified |
| **fmri** | fMRI analysis tools (atlas, connectivity) | Minimal Stub (15 lines) |
| **fractal** | Fractal dimension, lacunarity, multifractal analysis | Implemented (140 lines) |
| **fusion** | Data fusion/sensor fusion framework | Minimal Stub (7 lines) |
| **geometry** | Neuro-symbolic geometry (AlphaGeometry-inspired) | Implemented (46 lines index) |
| **gpu** | GPU kernels (FFT, smoothing, statistics) | Implemented (74 lines) |
| **graph** | Graph algorithms (random, entropy, curvature) | Implemented |
| **http** | HTTP client (libcurl FFI) | Implemented |
| **interop** | Interoperability (MedLang compatibility) | Partially Implemented |
| **io** | File I/O, directories, process control | Fully Implemented |
| **iter** | Iterator trait and combinators | Fully Implemented |
| **json** | JSON parsing/serialization (RFC 7159) | Fully Implemented |
| **linalg** | Linear algebra (BLAS, LAPACK, fixed-size vectors) | Fully Specified |
| **medlang** | MedLang DSL for pharmacology | Partially Implemented |
| **mem** | Memory management, allocators | Stub |
| **ml** | Machine learning | Stub |
| **nn** | Neural networks (autograd 12.7K lines) | Heavily Implemented |
| **ode** | ODE solvers (Tsit5, RK4, BDF, Radau) | Fully Implemented |
| **ontology** | Biomedical ontologies (SNOMED, LOINC, GO, HPO) | Partially Implemented |
| **optimize** | Optimization algorithms | Partially Implemented (121 lines) |
| **pbpk** | PBPK models (general framework) | Stub |
| **prob** | Probabilistic programming (sample, observe, inference) | Implemented |
| **profile** | Performance profiling tools | Implemented (1.2K lines) |
| **quantum** | Quantum computing (VQE with epistemic bounds) | Implemented (1.3K lines) |
| **random** | RNG (PCG64, Xoshiro256++) and distributions | Implemented (98 lines) |
| **search** | Search algorithms | Stub |
| **signal** | Signal processing | Minimal (13 lines) |
| **src** | Nested source directory | Special |
| **stats** | Statistical functions | Minimal (13 lines) |
| **str** | String utilities (trim, split, iterators) | Implemented |
| **string** | String type and operations | Implemented |
| **sync** | Synchronization (Mutex, RWLock, Channel) | Implemented |
| **systems** | Systems programming | Stub |
| **test** | Test framework (assertions, mocking, property-based) | Fully Implemented |
| **time** | Time/duration (DateTime, Duration, formatting) | Fully Implemented |
| **trace** | Distributed tracing | Implemented (1.1K lines) |
| **types** | Type utilities (refinement types) | Implemented |
| **units** | Units of measure (QUDT integration) | Implemented |

---

## Tier 1: Production-Ready

### core/ (Option, Result)

```
Status: Fully Implemented (424 + 462 lines)
```

**Option<T> Methods (30+):**
- `is_some()`, `is_none()`, `unwrap()`, `unwrap_or()`, `unwrap_or_else()`
- `map()`, `map_or()`, `map_or_else()`, `filter()`
- `and()`, `and_then()`, `or()`, `or_else()`
- `flatten()`, `zip()`, `take()`, `replace()`
- `ok_or()`, `ok_or_else()`, `transpose()`

**Result<T, E> Methods (40+):**
- `is_ok()`, `is_err()`, `ok()`, `err()`
- `map()`, `map_err()`, `map_or()`, `map_or_else()`
- `and()`, `and_then()`, `or()`, `or_else()`
- `unwrap()`, `unwrap_err()`, `unwrap_or()`, `unwrap_or_else()`
- `expect()`, `expect_err()`, `transpose()`

**Traits:** Clone, Eq, Ord, Debug, Hash, IntoIterator

---

### collections/ (Vec, HashMap, HashSet, Deque)

```
Status: Fully Implemented
- Vec: 865 lines, growth strategy, iterator adapters
- HashMap: 755 lines, open addressing with probing
- HashSet: 548 lines, set operations
- Deque: 567 lines, circular buffer
```

**Vec<T>:**
- Construction: `new()`, `with_capacity()`, `from_slice()`
- Access: `get()`, `get_mut()`, `first()`, `last()`, `[]` indexing
- Mutation: `push()`, `pop()`, `insert()`, `remove()`, `clear()`, `truncate()`
- Iteration: `iter()`, `iter_mut()`, `into_iter()`
- Capacity: `len()`, `capacity()`, `is_empty()`, `reserve()`, `shrink_to_fit()`

**HashMap<K, V>:**
- `new()`, `with_capacity()`, `insert()`, `remove()`, `get()`, `get_mut()`
- `contains_key()`, `keys()`, `values()`, `iter()`, `len()`, `is_empty()`

**HashSet<T>:**
- `new()`, `insert()`, `remove()`, `contains()`
- Set ops: `union()`, `intersection()`, `difference()`, `symmetric_difference()`
- `is_subset()`, `is_superset()`, `is_disjoint()`

**Deque<T>:**
- `new()`, `push_front()`, `push_back()`, `pop_front()`, `pop_back()`
- `front()`, `back()`, `get()`, `len()`, `is_empty()`

---

### io/ (File I/O, Process)

```
Status: Fully Implemented (770+ lines)
```

**File Operations:**
- `read_file(path: str) -> Result<str, IoError>`
- `write_file(path: str, content: str) -> Result<(), IoError>`
- `append_file(path: str, content: str) -> Result<(), IoError>`
- `file_exists(path: str) -> bool`
- `remove_file(path: str) -> Result<(), IoError>`
- `copy_file(src: str, dst: str) -> Result<(), IoError>`
- `rename(old: str, new: str) -> Result<(), IoError>`

**Directory Operations:**
- `create_dir(path: str) -> Result<(), IoError>`
- `create_dir_all(path: str) -> Result<(), IoError>`
- `remove_dir(path: str) -> Result<(), IoError>`
- `read_dir(path: str) -> Result<Vec<DirEntry>, IoError>`

**Console I/O:**
- `print(s: str)`, `println(s: str)`, `eprint(s: str)`, `eprintln(s: str)`
- `read_line() -> Result<str, IoError>`

**Error Variants:**
`NotFound`, `PermissionDenied`, `AlreadyExists`, `InvalidInput`, `ReadError`, `WriteError`, `DirectoryError`, `IoError`, `Other`

---

### iter/ (Iterator Trait & Combinators)

```
Status: Fully Implemented (1702 lines)
```

**Core Trait:**
```sio
trait Iterator {
    type Item
    fn next(&!self) -> Option<Self::Item>
    fn size_hint(&self) -> (usize, Option<usize>)
}
```

**Combinators:**
- Filtering: `filter()`, `filter_map()`, `take()`, `take_while()`, `skip()`, `skip_while()`
- Mapping: `map()`, `flat_map()`, `flatten()`
- Folding: `fold()`, `reduce()`, `sum()`, `product()`
- Searching: `find()`, `find_map()`, `position()`, `any()`, `all()`
- Collecting: `collect()`, `count()`, `last()`, `nth()`
- Combining: `zip()`, `chain()`, `enumerate()`

---

### time/ (Duration, DateTime, Instant)

```
Status: Fully Implemented (1322 lines)
```

**Duration:**
- `from_secs()`, `from_millis()`, `from_micros()`, `from_nanos()`
- `as_secs()`, `as_millis()`, `as_micros()`, `as_nanos()`
- `checked_add()`, `checked_sub()`, `saturating_add()`, `saturating_sub()`

**DateTime:**
- Construction: `now()`, `from_timestamp()`, `parse()`
- Components: `year()`, `month()`, `day()`, `hour()`, `minute()`, `second()`
- Formatting: `format()`, `to_rfc3339()`, `to_iso8601()`
- Arithmetic: `add_days()`, `add_hours()`, `diff()`

**Instant:**
- `now()`, `elapsed()`, `duration_since()`

---

### json/ (JSON Parsing/Serialization)

```
Status: Fully Implemented (1165 lines)
```

**Types:**
```sio
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(str),
    Array(Vec<JsonValue>),
    Object(HashMap<str, JsonValue>),
}
```

**API:**
- `parse(s: str) -> Result<JsonValue, JsonError>`
- `stringify(v: JsonValue) -> str`
- `stringify_pretty(v: JsonValue, indent: usize) -> str`

**Error Variants:**
`UnexpectedChar`, `UnexpectedEof`, `InvalidEscape`, `InvalidNumber`, `NestingTooDeep`, `TrailingComma`

---

### test/ (Testing Framework)

```
Status: Fully Implemented
```

**Assertions:**
- `assert(cond: bool)`, `assert_eq(a, b)`, `assert_ne(a, b)`
- `assert_approx(a: f64, b: f64, epsilon: f64)`
- `assert_in_range(v, min, max)`, `assert_contains(haystack, needle)`
- `assert_len(collection, expected_len)`

**Property-Based Testing:**
- `prop_check(gen: Generator<T>, prop: fn(T) -> bool, iterations: usize)`

**Mocking:**
- `Mock<T>`, `Stub<T>`, `expect_call()`, `returns()`, `verify()`

---

## Tier 2: Domain-Specific (Production Quality)

### linalg/ (Linear Algebra)

```
Status: Fully Specified
```

**Fixed-Size Types:**
- Vectors: `Vec2`, `Vec3`, `Vec4`, `Vec14` (PBPK)
- Matrices: `Mat2`, `Mat3`, `Mat4`, `Mat14Diag`, `Mat14Tridiag`

**Dynamic Types:**
- `DenseVector`, `DenseMatrix` (BLAS-backed)

**BLAS Level 1-3:**
- Level 1: `daxpy`, `ddot`, `dnrm2`, `dscal`, `dcopy`
- Level 2: `dgemv`, `dtrmv`, `dsymv`
- Level 3: `dgemm`, `dtrsm`, `dsyrk`

**LAPACK:**
- Decompositions: `lu`, `cholesky`, `qr`, `svd`, `eig`
- Solutions: `solve`, `least_squares`
- Properties: `inv`, `det`, `rank`, `cond`

---

### ode/ (ODE Solvers)

```
Status: Fully Implemented
```

**Solvers:**
| Solver | Type | Use Case |
|--------|------|----------|
| `Tsit5` | Adaptive RK | General non-stiff ODEs |
| `RK4` | Fixed-step | Simple systems, learning |
| `DOPRI5` | Adaptive | Robust general-purpose |
| `BDF` | Implicit | Stiff systems |
| `Radau5` | L-stable | Very stiff, DAEs |

**API:**
```sio
fn solve<F>(
    f: F,
    y0: Vec<f64>,
    tspan: (f64, f64),
    solver: Solver,
    options: SolverOptions,
) -> OdeSolution
where F: Fn(f64, &[f64]) -> Vec<f64>
```

**Features:**
- Event detection and handling
- Dense output interpolation
- Adaptive step control
- Jacobian-free methods

---

### epistemic/ (Epistemic Core)

```
Status: Heavily Implemented (40 files, 31,962 lines)
```

**Core Types:**
```sio
struct Knowledge<T> {
    value: T,
    confidence: Confidence,
    provenance: Provenance,
    temporal: TemporalContext,
}

struct Confidence {
    epsilon: f64,           // Uncertainty bound [0, 1]
    distribution: Option<BetaDist>,
}

struct Provenance {
    source: ProvenanceSource,
    operations: Vec<Operation>,
    merkle_root: Hash,
}
```

**Submodules:**
| File | LOC | Purpose |
|------|-----|---------|
| `core.sio` | 500+ | Uncertainty and Confidence |
| `causal.sio` | 400+ | Pearl's do-calculus |
| `mcmc.sio` | 1,570 | Metropolis-Hastings |
| `ode.sio` | 1,437 | Epistemic ODE solving |
| `optimization.sio` | 1,786 | Gradient descent with confidence |
| `discovery.sio` | 2,261 | Structure learning |
| `linalg.sio` | 1,459 | Epistemic matrix ops |
| `timeseries.sio` | 1,617 | Time series with uncertainty |
| `gum.sio` | 800+ | Guide to Uncertainty in Measurement |

**Operations:**
- Confidence propagation through computations
- Provenance DAG tracking
- Temporal validity constraints
- Epistemic firewalls

---

### autodiff/ (Automatic Differentiation)

```
Status: Implemented
```

**Forward Mode (Dual Numbers):**
```sio
struct Dual {
    value: f64,
    derivative: f64,
}

impl Dual {
    fn constant(x: f64) -> Dual
    fn variable(x: f64) -> Dual
    fn sin(self) -> Dual
    fn cos(self) -> Dual
    fn exp(self) -> Dual
    fn log(self) -> Dual
    fn sqrt(self) -> Dual
    fn pow(self, n: f64) -> Dual
}
```

**Reverse Mode:**
- Tape-based recording
- Backward pass for gradients
- Memory-efficient checkpointing

---

### nn/ (Neural Networks)

```
Status: Heavily Implemented (13,693 lines)
```

**Autograd Engine:**
- Computational graph construction
- Automatic backward pass
- Gradient accumulation
- Memory management

**Layers:**
- `Linear`, `Conv2d`, `BatchNorm`, `Dropout`
- `ReLU`, `Sigmoid`, `Tanh`, `Softmax`
- `LSTM`, `GRU`, `Attention`

**Optimizers:**
- `SGD`, `Adam`, `AdamW`, `RMSprop`

---

### quantum/ (Quantum Computing)

```
Status: Implemented (1,392 lines)
```

**VQE (Variational Quantum Eigensolver):**
```sio
struct VQE {
    ansatz: Ansatz,
    hamiltonian: Hamiltonian,
    optimizer: Optimizer,
    shots: usize,
    epistemic_bounds: EpistemicBounds,
}
```

**Features:**
- Hardware noise modeling
- Shot noise uncertainty
- Expressibility bounds
- Example: H2 molecule Hamiltonian

---

### darwin/ (Darwin Atlas Kernels)

```
Status: Implemented (atlas.sio)
```

**Scope:**
- Sequence operators: shift, reverse, complement, reverse_complement, hamming_distance
- Exact symmetry: orbit_size, orbit_ratio, palindrome, rc-fixed, rotational_period
- Approx metric: dmin, dmin_normalized, nearest_transform
- Quaternion lift: dicyclic elements, double cover verification

---

### darwin_pbpk/ (PBPK Simulation)

```
Status: Heavily Implemented (6,036 lines)
```

**Models:**
- Rodgers-Rowland tissue partition
- TSit5 integration for stiff systems
- Unit-safe computations (mg, mL, h)

**Compartments:**
- Plasma, liver, kidney, gut, muscle, adipose, etc.
- Clearance models
- Absorption models (oral, IV)

---

## Tier 3: Partially Implemented

### async/

```
Status: Implemented (7,435 lines, 13 files)
```

**Submodules:**
- `future`: Future trait and combinators
- `task`: Task spawning and management
- `executor`: Async runtime
- `channel`: Async channels
- `select`: Multi-future selection
- `stream`: Async iterators

**Implementation:** Core traits defined, executor framework sketched

---

### medlang/

```
Status: Partially Implemented
- mod.sio: 39 lines
- parser.sio: 1,337 lines
```

**Features:**
- MedLang drug definition parsing
- Model type conversion
- Sounio AST generation with Knowledge tracking

**Gap:** Semantic execution partial

---

### ontology/

```
Status: Partially Implemented (55 lines mod)
```

**Namespaces:**
- RDF, RDFS, OWL, XSD
- SNOMED-CT, LOINC, GO, HPO

**Gap:** Reasoning engine partial

---

## Tier 4: Stubs/Minimal

| Module | Lines | Needs |
|--------|-------|-------|
| `mem` | Stub | Allocators, Arena, Pool |
| `ml` | Stub | Tree models, ensemble, XGBoost |
| `search` | Stub | A*, BFS, DFS, beam search |
| `signal` | 13 | FFT, filtering, windowing |
| `stats` | 13 | Distributions, hypothesis tests |
| `connectivity` | 7 | Graph metrics, community detection |
| `fusion` | 7 | Kalman, particle filters |
| `fmri` | 15 | GLM, ROI analysis |
| `systems` | Stub | Low-level utilities |
| `csv` | Minimal | Streaming, custom delimiters |

---

## Quality Assessment

| Category | Score | Notes |
|----------|-------|-------|
| **Core Types** | 10/10 | Option, Result, Vec, HashMap fully featured |
| **Scientific Math** | 9/10 | ODE, autodiff, epistemic tracking excellent |
| **Epistemic System** | 10/10 | Unmatched - Knowledge<T> is novel |
| **Domain (Pharma)** | 8/10 | PBPK production-ready; MedLang partial |
| **Systems** | 5/10 | Async specified but not implemented |
| **Coverage** | 6/10 | 290 files but many stubs |
| **Documentation** | 8/10 | Good docstrings; some lack examples |

---

## Usage Guide

### Immediately Usable

```sio
use core::{Option, Result}
use collections::{Vec, HashMap, HashSet, Deque}
use io::{read_file, write_file, println}
use iter::Iterator
use time::{Duration, DateTime}
use json::{parse, stringify}
use test::{assert, assert_eq}
use linalg::{Vec3, Mat3, dgemm}
use ode::{solve, Tsit5}
use random::{Rng, PCG64}
use epistemic::{Knowledge, Confidence}
use autodiff::{Dual, gradient}
use bayes::{mcmc, metropolis_hastings}
use causal::{CausalDAG, do_calculus}
use nn::{Linear, Adam}
use quantum::{VQE}
use darwin_pbpk::{simulate}
use profile::{Timer, profile}
use trace::{span, trace}
use http::{get, post}
```

### Works But Incomplete

```sio
use async::{spawn, await}    // Traits only
use gpu::{kernel}            // Stubs
use medlang::{parse_drug}    // Parser works
use ontology::{query}        // Types only
use optimize::{minimize}     // Skeleton
```

### Not Yet Implemented

```sio
use mem::{Arena}             // Stub
use ml::{RandomForest}       // Stub
use search::{astar}          // Stub
use signal::{fft}            // Minimal
use stats::{ttest}           // Minimal
```

---

## Compilation Notes

Most modules compile with `souc check`. Known issues:

1. **Math functions** (`sqrt`, `log`, `exp`) - Require linking to libm
2. **GPU kernels** - Require CUDA availability
3. **HTTP** - Requires libcurl
4. **Ontology reasoning** - Partially stubbed

---

*Generated: February 2026*
