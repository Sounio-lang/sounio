<!-- docs:meta
topic_id: website.docs.compiler.scientific-features
authority: dual
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.compiler.scientific-features
-->

# Funcionalidades Científicas - Arquitetura do Compilador Sounio

## Visão Geral

O compilador Sounio possui suporte integrado para diversas áreas de computação científica, incluindo **álgebra hypercomplexa**, **equações diferenciais**, **computação quântica**, **machine learning**, e **otimização numérica**.

```
┌─────────────────────────────────────────────────────────────────────┐
│              FUNCIONALIDADES CIENTÍFICAS DO SOUNIO                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │   Linear      │  │   Geometry     │  │   Optimizer    │    │
│  │   Algebra     │  │                │  │                │    │
│  │ • Quaternions │  │ • Alpha Geo    │  │ • ODE Solvers │    │
│  │ • Octonions   │  │ • Predicates  │  │ • PDE Solvers │    │
│  │ • Matrices    │  │ • Proof State │  │ • NLP         │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │   Quantum     │  │   Machine      │  │   Epistemic   │    │
│  │   Computing   │  │   Learning     │  │   Computing   │    │
│  │ • Circuits    │  │ • QNN         │  │ • Uncertainty │    │
│  │ • VQE/UCCSD  │  │ • GP         │  │ • MCMC       │    │
│  │ • PennyLane  │  │ • Neural Nets │  │ • PCE        │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 1. Álgebra Hypercomplexa

### 1.1 Quaternions

#### Localização
- **Lexer tokens**: [`crates/souc/src/lexer/tokens.rs:186`](../../crates/souc/src/lexer/tokens.rs:186)
- **HLIR types**: [`crates/souc/src/hlir/ir.rs:152`](../../crates/souc/src/hlir/ir.rs:152)
- **Runtime**: [`crates/souc/src/backend/native/quat_runtime.rs`](../../crates/souc/src/backend/native/quat_runtime.rs)

#### Tipos Quaterniónicos no Compilador

```rust
// crates/souc/src/hlir/ir.rs:152
// Linear algebra primitives (SIMD-friendly)
Vec2,    // 2x f32
Vec3,    // 3x f32 (padded to 4 for SIMD)
Vec4,    // 4x f32
Mat2,    // 2x2 f32 (4 floats)
Mat3,    // 3x3 f32 (9 floats)
Mat4,    // 4x4 f32 (16 floats)
Quat,    // 4x f32 (x, y, z, w)
```

#### Definição Matemática

```
Quaternion: q = a + bi + cj + dk

onde:
- a, b, c, d ∈ ℝ (componentes reais)
- i² = j² = k² = ijk = -1 (unidades imaginárias)
- ij = k, jk = i, ki = j (multiplicação)
- ji = -k, kj = -i, ik = -j (anti-comutativa)
```

#### Operações Quaterniónicas

| Operação | Descrição | Complexidade |
|----------|-----------|--------------|
| `q1 * q2` | Multiplicação | O(16) FLOPs |
| `q1 / q2` | Divisão | O(32) FLOPs |
| `conj(q)` | Conjugado | O(4) ops |
| `norm(q)` | Norma | O(4) FLOPs |
| `slerp(q1, q2, t)` | Interpolação esférica | O(48) FLOPs |

#### Exemplo de Uso

```sio
// Rotação 3D com quaternions
let q1: Quat = Quat::from_axis_angle(axis: vec3(0, 0, 1), angle: PI / 4);
let q2: Quat = Quat::from_axis_angle(axis: vec3(1, 0, 0), angle: PI / 6);

// Interpolação esférica
let q_interp = q1.slerp(q2, 0.5);

// Conversão para matriz de rotação
let rotation_matrix = q_interp.to_rotation_matrix();
```

### 1.2 Octonions

#### Localização
- **Documentação**: [`docs/compiler/OCTONION_ALGEBRA.md`](../../docs/compiler/OCTONION_ALGEBRA.md)
- **HLIR types**: [`crates/souc/src/hlir/ir.rs:160`](../../crates/souc/src/hlir/ir.rs:160)

#### Tipos Octoniónicos

```rust
// crates/souc/src/hlir/ir.rs:160
// Hypercomplex types (Cayley-Dickson sequence)
Octonion,   // 8x f32 (a, b, c, d, e, f, g, h) - 256 bits
Sedenion,   // 16x f32 (e₀=1, e₁, ..., e₁₅) - 512 bits, has zero divisors
```

#### Definição Matemática (Cayley-Dickson)

```
Octonions: O = Q ⊕ Q·l

onde:
- Q são quaternions
- l é uma nova unidade com l² = -1
- Multiplicação: (q₀ + q₁·l)(q₂ + q₃·l) = (q₀q₂ - q̄₃q₁) + (q₃q₀ + q₁q̄₂)·l
```

#### Tabela de Multiplicação

```
     1   i   j   k   l  il  jl  kl
1    1   i   j   k   l  il  jl  kl
i    i  -1   k  -j  il  -l  kl -jl
j    j  -k  -1   i  jl -kl  -l  il
k    k   j  -i  -1  kl  jl -il  -l
l    l -il -jl -kl  -1   i   j   k
il  il   l -kl  jl  -i  -1  -k   j
jl  jl  kl   l -il  -j   k  -1  -i
kl  kl -jl  il   l  -k   j   i  -1
```

#### Propriedades Matemáticas

| Propriedade | Descrição |
|-------------|-----------|
| **Norma Multiplicativa** | \|xy\| = \|x\| · \|y\| |
| **Lei Alternativa** | (xx)y = x(xy), y(xx) = (yx)x |
| **Flexibilidade** | (xy)x = x(yx) |
| **Identidade de Jacobi** | [x,[y,z]] + [y,[z,x]] + [z,[x,y]] = 0 |

#### Exemplo de Uso

```sio
// Criar octonions
let o1: Octonion = Octonion::new(1, 2, 3, 4, 5, 6, 7, 8);
let o2: Octonion = Octonion::new(8, 7, 6, 5, 4, 3, 2, 1);

// Multiplicação não-associativa
let o3 = o1 * o2;  // O(64) multiplicações, O(56) adições

// Norma e inverso
let norm = o3.norm();  // √(a² + b² + ... + h²)
let inverse = o3.inverse();  // conj(o) / |o|²
```

### 1.3 Quaternionic Neural Networks (QNN)

#### Localização
- **HLIR types**: [`crates/souc/src/hlir/ir.rs:163`](../../crates/souc/src/hlir/ir.rs:163)
- **GPU kernels**: [`crates/souc/src/codegen/gpu/qnn_kernels.rs`](../../crates/souc/src/codegen/gpu/qnn_kernels.rs)

#### Tipos QNN

```rust
// crates/souc/src/hlir/ir.rs:163
// Quaternionic Neural Network types
QuatLinear,   // Quaternionic linear layer (struct with weights)
QuatConv2d,   // Quaternionic 2D convolution (struct with kernel)
QuatRnnState, // Quaternionic RNN state
QuatGate,     // Quaternionic gate
```

#### Arquitetura QNN

```
┌─────────────────────────────────────────────────────────────┐
│           QUATERNIONIC NEURAL NETWORK                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: [h₀, h₁, ..., hₙ] ⊂ ℍⁿ (quaternion-valued)      │
│         ↓                                                   │
│  ┌─────────────────┐                                       │
│  │ QuatConv2d     │  Convolução quaternionica              │
│  │                 │  K kernels, H×W feature maps          │
│  └────────┬────────┘                                       │
│           ↓                                                │
│  ┌─────────────────┐                                       │
│  │ QuatLinear     │  Fully connected quaternion layer       │
│  │                 │  W ∈ ℍ^(m×n), b ∈ ℍ^m                 │
│  └────────┬────────┘                                       │
│           ↓                                                │
│  Output: [o₀, o₁, ..., oₘ] ⊂ ℍᵐ                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Vantagens dos QNNs

| Vantagem | Descrição |
|----------|-----------|
| **Eficiência de Parâmetros** | 4× menos parâmetros que redes reais equivalentes |
| **Symmetry Breaking** | Natural handling de 4-fold symmetries |
| **3D Rotations** | April 2021 - improved 3D rotation handling |
| **Hypercomplex Arithmetic** | Intrinsics SIMD eficientes |

## 2. Equações Diferenciais

### 2.1 ODE Solvers

#### Localização
- **Runtime**: [`crates/souc/src/backend/native/ode_runtime.rs`](../../crates/souc/src/backend/native/ode_runtime.rs)
- **Lexer keywords**: [`crates/souc/src/lexer/tokens.rs:155`](../../crates/souc/src/lexer/tokens.rs:155)

#### Tokens para DSL de ODEs

```rust
// crates/souc/src/lexer/tokens.rs:155
// Scientific DSL keywords
#[token("ode")] Ode,
#[token("pde")] Pde,
#[token("causal")] Causal,
#[token("nodes")] Nodes,
#[token("edges")] Edges,
#[token("equations")] Equations,
#[token("state")] State,
#[token("params")] Params,
```

#### Definição de ODEs em Sounio

```sio
// Exemplo: Modelo Lotka-Volterra
ode LotkaVolterra {
    // Estado do sistema
    state {
        x: f64,  // Presa
        y: f64   // Predador
    }

    // Parâmetros
    params {
        alpha: f64 = 1.1,   // Taxa de crescimento da presa
        beta: f64 = 0.4,     // Taxa de predação
        gamma: f64 = 0.4,    // Taxa de mortalidade do predador
        delta: f64 = 0.1     // Taxa de reprodução do predador
    }

    // Sistema de equações diferenciais
    equations {
        dx/dt = alpha * x - beta * x * y,
        dy/dt = delta * x * y - gamma * y
    }

    // Condições iniciais
    initial {
        x(0) = 10.0,
        y(0) = 10.0
    }

    // Intervalo de tempo
    tspan: (0.0, 100.0)
}
```

#### Métodos de Solver

| Método | Descrição | Ordem | Estabilidade |
|--------|-----------|-------|--------------|
| Euler | Forward Euler | 1 | Estável p/ passo pequeno |
| RK4 | Runge-Kutta 4ª ordem | 4 | Estável p/ passo moderado |
| RK45 | Runge-Kutta-Fehlberg | 5(4) | Adaptive step |
| BS5 | Bogacki-Shampine | 4(3) | Adaptive step |
| CVODE | BDF | Variável | Estável p/ stiff |

#### Exemplo de Uso

```sio
// Resolver ODE
let solution = LotkaVolterra.solve(
    method: RK45,
    dt: 0.01,
    tspan: (0.0, 100.0)
);

// Acessar solução
let x_at_t10 = solution.eval(t: 10.0, var: "x");
let trajectory = solution.trajectory(var: "y");

// Plotar resultados
LotkaVolterra.plot(solution);
```

### 2.2 PDE Solvers

```sio
// Exemplo: Equação do Calor 2D
pde HeatEquation {
    domain {
        x: (0.0, 1.0),   // Domínio espacial X
        y: (0.0, 1.0),   // Domínio espacial Y
        t: (0.0, 1.0)    // Domínio temporal
    }

    // Equação diferencial parcial
    equation {
        ∂u/∂t = alpha * (∂²u/∂x² + ∂²u/∂y²)
    }

    // Condições de contorno
    boundary {
        u(0, y, t) = 0.0,    // Dirichlet
        u(1, y, t) = 0.0,
        u(x, 0, t) = 0.0,
        u(x, 1, t) = 0.0
    }

    // Condição inicial
    initial {
        u(x, y, 0) = sin(π * x) * sin(π * y)
    }
}
```

## 3. Números Duais (Automatic Differentiation)

#### Localização
- **HLIR types**: [`crates/souc/src/hlir/ir.rs:172`](../../crates/souc/src/hlir/ir.rs:172)

#### Definição

```
Dual Number: ε² = 0, ε ≠ 0

Dual(a, b) = a + bε

onde:
- a = parte real (valor)
- b = parte dual (derivada)
```

#### Operações

| Operação | Resultado |
|----------|-----------|
| `Dual(a, b) + Dual(c, d)` | `Dual(a + c, b + d)` |
| `Dual(a, b) * Dual(c, d)` | `Dual(a*c, a*d + b*c)` |
| `f(Dual(a, b))` | `Dual(f(a), f'(a)*b)` |

#### Exemplo de Uso

```sio
// Forward-mode automatic differentiation
let x = Dual(3.0, 1.0);  // x = 3 + 1ε

let y = x * x + 2.0 * x;  // y = x² + 2x
// y = Dual(9 + 6, 6 + 2) = Dual(15, 8)
// dy/dx em x=3 = 8

// Derivada de funções complexas
fn sigmoid(x: Dual) -> Dual {
    let one = Dual(1.0, 0.0);
    one / (one + (-x).exp())
}

let result = sigmoid(x);  // result = Dual(value, derivative)
print("Derivative:", result.dual_part);  // Imprime: 0.0498...
```

## 4. Computação Quântica

### 4.1 Quantum Circuit Representation

#### Localização
- **Módulo**: [`crates/souc/src/quantum/`](../../crates/souc/src/quantum/)

#### Estrutura do Circuito

```rust
// crates/souc/src/quantum/circuit.rs
pub struct QuantumCircuit {
    /// Number of qubits
    n_qubits: usize,
    /// Quantum gates in the circuit
    gates: Vec<QuantumGate>,
    /// Classical registers for measurement
    classical_regs: Vec<ClassicalRegister>,
    /// Circuit parameters
    params: CircuitParams,
}

pub enum QuantumGate {
    H(usize),           // Hadamard on qubit
    X(usize),          // Pauli-X
    Y(usize),          // Pauli-Y
    Z(usize),          // Pauli-Z
    CX(usize, usize), // CNOT (control, target)
    CZ(usize, usize), // CZ (control, target)
    RX(f64, usize),    // Rotation around X
    RY(f64, usize),   // Rotation around Y
    RZ(f64, usize),   // Rotation around Z
    T(usize),          // T gate (π/8)
    S(usize),          // S gate (√Z)
    U3(f64, f64, f64, usize),  // Universal single-qubit
    SWAP(usize, usize),
    ISWAP(usize, usize),
    // ...
}
```

#### Portas Quânticas

| Porta | Matriz | Descrição |
|-------|---------|-----------|
| H | 1/√2 [[1,1],[1,-1]] | Superposição |
| X | [[0,1],[1,0]] | Bit flip |
| Y | [[0,-i],[i,0]] | Bit + phase flip |
| Z | [[1,0],[0,-1]] | Phase flip |
| T | [[1,0],[0, e^(iπ/4)]] | π/8 gate |
| CNOT | Matriz 4×4 | Controlled-NOT |
| SWAP | Matriz 4×4 | Troca de qubits |

#### Exemplo de Circuito

```sio
// Criar circuito quântico
let circuit = QuantumCircuit::new(n_qubits: 4);

// Adicionar portas
circuit.h(0);           // Hadamard no qubit 0
circuit.cnot(0, 1);     // CNOT do qubit 0 para 1
circuit.ry(PI/4, 2);     // Rotação Y no qubit 2
circuit.cz(1, 3);       // CZ do qubit 1 para 3

// Medição
circuit.measure(0, 0);
circuit.measure(1, 1);

// Executar
let result = circuit.execute(n_shots: 1024);
```

### 4.2 VQE (Variational Quantum Eigensolver)

#### Localização
- **VQE implementation**: [`crates/souc/src/quantum/vqe.rs`](../../crates/souc/src/quantum/vqe.rs)

```rust
pub struct VQE {
    /// Hamiltonian do sistema
    hamiltonian: Hamiltonian,
    /// Parâmetros variacionais
    params: Vec<f64>,
    /// Circuito variacional (Ansatz)
    ansatz: QuantumCircuit,
    /// Otimizador clássico
    optimizer: Box<dyn Optimizer>,
}
```

#### Exemplo de Uso

```sio
// Definir Hamiltonian (ex: H₂ molecule)
let h2_hamiltonian = Hamiltonian::from_molecule(
    geometry: "H 0 0 0; H 0 0 0.74",
    basis: "sto-3g",
    mapping: "jordan-wigner"
);

// Criar VQE
let vqe = VQE::new(
    hamiltonian: h2_hamiltonian,
    ansatz: "uccsd",  // Coupled Cluster singles and doubles
    optimizer: COBYLA::new()
);

// Otimizar
let result = vqe.optimize(n_iterations: 1000);
print("Ground state energy:", result.energy);
```

### 4.3 UCCSD Ansatz

```rust
// crates/souc/src/quantum/uccsd.rs
pub struct UCCSD {
    /// Número de spin-orbitais
    n_spin_orbitals: usize,
    /// Número de elétrons
    n_electrons: usize,
    /// Parâmetros t (amplitudes)
    t1_amplitudes: Vec<f64>,
    t2_amplitudes: Vec<f64>,
}
```

## 5. Machine Learning

### 5.1 Gaussian Processes

#### Localização
- **EPistemic GP**: [`crates/souc/src/epistemic/gaussian_process.rs`](../../crates/souc/src/epistemic/gaussian_process.rs)

```rust
pub struct GaussianProcess {
    /// Kernel function
    kernel: Box<dyn Kernel>,
    /// Training data (X, y)
    training_data: (Vec<Vec<f64>>, Vec<f64>),
    /// GP hyperparameters
    length_scale: f64,
    variance: f64,
    noise: f64,
}

pub trait Kernel {
    fn compute(&self, x1: &[f64], x2: &[f64]) -> f64;
    fn gradient(&self, x: &[f64]) -> Vec<f64>;
}

pub enum KernelType {
    RBF,           // Radial Basis Function
    Matern32,      // Matérn 3/2
    Matern52,      // Matérn 5/2
    Periodic,      // Periodic kernel
    Linear,        // Linear kernel
    Composite,     // Sum/product of kernels
}
```

#### Exemplo de Uso

```sio
// Criar GP com kernel RBF
let kernel = Kernel::rbf(length_scale: 1.0, variance: 1.0);
let gp = GaussianProcess::new(kernel);

// Treinar com dados observados
gp.fit(X: training_data_x, y: training_data_y);

// Predizer com incerteza epistêmica
let (mean, variance) = gp.predict(x: query_point);
let confidence = 1.96 * sqrt(variance);  // 95% CI

// Predição epistêmica
let prediction: Knowledge<f64> = Knowledge::new(
    value: mean,
    uncertainty: variance.sqrt(),
    confidence: 0.95,
    source: "gaussian_process_regression"
);
```

### 5.2 MCMC Samplers

#### Localização
- **MCMC**: [`crates/souc/src/epistemic/mcmc.rs`](../../crates/souc/src/epistemic/mcmc.rs)

```rust
pub struct MCMC {
    /// Sampler type
    sampler: MCMCSampler,
    /// Number of samples
    n_samples: usize,
    /// Burn-in period
    burn_in: usize,
    /// Thinning factor
    thinning: usize,
}

pub enum MCMCSampler {
    MetropolisHastings {
        proposal: ProposalDistribution,
    },
    HamiltonianMC {
        step_size: f64,
        n_leapfrog: usize,
    },
    NUTS {
        max_tree_depth: usize,
        delta: f64,
    },
}
```

## 6. Otimizadores Numéricos

### 6.1 ODE/Optimization Integration

#### Localização
- **Optimizer**: [`crates/souc/src/optimizer/`](../../crates/souc/src/optimizer/)

```rust
pub trait Optimizer {
    fn optimize<F>(&mut self, f: F) -> OptimizationResult
    where
        F: Fn(&[f64]) -> f64;

    fn optimize_with_grad<F, G>(&mut self, f: F, grad: G) -> OptimizationResult
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>;
}
```

### 6.2 NLP (Nonlinear Programming)

```sio
// Exemplo: Otimização irrestrita
fn rosenbrock(x: [f64; 2]) -> f64 {
    let a = 1.0;
    let b = 100.0;
    return (a - x[0])^2 + b * (x[1] - x[0]^2)^2;
}

let optimizer = LBFGS::new(memory_size: 10);
let result = optimizer.minimize(
    objective: rosenbrock,
    initial_guess: [0.0, 0.0],
    tolerance: 1e-6
);

// Resultado epistêmico
let optimal: Knowledge<[f64; 2]> = Knowledge::new(
    value: result.minimizer,
    uncertainty: result.hessian_estimate.inverse(),
    confidence: 0.95,
    source: "lbfgs_optimization"
);
```

## 7. Tipos de Unidades PK/PD

### Sistema de Unidades Farmacocinéticas

#### Localização
- **Units PK/PD**: [`crates/souc/src/units/pkpd.rs`](../../crates/souc/src/units/pkpd.rs)

#### Unidades PK/PD Suportadas

| Símbolo | Unidade | Descrição |
|---------|---------|-----------|
| `mg` | Miligrama | Massa |
| `mL` | Mililitro | Volume |
| `mg/mL` | mg/mL | Concentração |
| `L/h` | Litros/hora | Clearance |
| `mg/L` | mg/L | Concentração |
| `h⁻¹` | Por hora | Constante de taxa |
| `mg*h/L` | AUC | Área sob curva |

#### Exemplo PK/PD

```sio
// Modelo farmacocinético
let dose: Quantity<f64, Milligram> = Quantity::new(500.0);
let volume: Quantity<f64, Liter> = Quantity::new(10.0);
let clearance: Quantity<f64, LiterPerHour> = Quantity::new(5.0);

// Cálculo de concentração
let concentration = dose / volume;  // Quantity<f64, MilligramPerLiter>

// Meia-vida
let half_life = (volume * ln(2.0)) / clearance;  // Quantity<f64, Hour>
```

## 8. Polynomial Chaos Expansion (PCE)

#### Localização
- **PCE**: [`crates/souc/src/epistemic/pce.rs`](../../crates/souc/src/epistemic/pce.rs)

```rust
pub struct PCE {
    /// Basis polynomials
    basis: PolynomialBasis,
    /// Expansion coefficients
    coefficients: Vec<f64>,
    /// Multi-indices (for multivariate)
    multi_indices: Vec<MultiIndex>,
}

pub struct MultiIndex {
    pub indices: Vec<usize>,  // Which polynomial in each dimension
    pub degree: usize,        // Total degree = sum(indices)
}

pub enum PolynomialFamily {
    Hermite,      // Normal distribution
    Legendre,     // Uniform distribution
    Laguerre,     // Exponential distribution
    Jacobi,       // Beta distribution
    Charlier,     // Poisson distribution
    Krawtchouk,   // Binomial distribution
}
```

## 9. Métricas de Documentação

### Documentos Relacionados

| Documento | Descrição |
|-----------|-----------|
| [`docs/compiler/OCTONION_ALGEBRA.md`](../../docs/compiler/OCTONION_ALGEBRA.md) | Fundamentos matemáticos de octonions |
| [`docs/compiler/QUATERNION_NEURAL_NETWORKS.md`](../../docs/compiler/QUATERNION_NEURAL_NETWORKS.md) | Redes neurais quaterniónicas |
| [`docs/GLM_4.7_INTEGRATION.md`](../../docs/GLM_4.7_INTEGRATION.md) | Integração GLM |
| [`docs/compiler/PHASE2_OPTIMIZATIONS.md`](../../docs/compiler/PHASE2_OPTIMIZATIONS.md) | Otimizações científicas |

### Arquivos Principais

| Arquivo | Linhas | Descrição |
|---------|--------|-----------|
| `linear/mod.rs` | ~300 | Álgebra linear |
| `geometry/mod.rs` | ~400 | Geometria algébrica |
| `quantum/*.rs` | ~3,000 | Computação quântica |
| `optimizer/*.rs` | ~500 | Otimizadores |
| `epistemic/pce.rs` | ~300 | Polynomial Chaos |
| `epistemic/mcmc.rs` | ~400 | MCMC samplers |

## 10. Arquitetura de Integração

```
┌─────────────────────────────────────────────────────────────────────┐
│              PIPELINE DE COMPUTAÇÃO CIENTÍFICA                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Source Code (.sio)                                                 │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐                                                │
│  │     Parser      │  → Parse DSL (ODE, PDE, Quantum)               │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │ Type Checker   │  → Verificar unidades, tipos epistêmicos       │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │   HLIR Build    │  → Construir IR científico                    │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ├────────────────────────────────────────┐                │
│           │                 │                    │                │
│           ▼                 ▼                    ▼                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  ODE/PDE IR    │ │ Quantum IR     │ │ ML IR          │   │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘   │
│           │                   │                    │                │
│           ▼                   ▼                    ▼                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  ODE Solver    │ │ Quantum Sim    │ │ GP/NN Impl    │   │
│  │  Runtime       │ │ Runtime        │ │ Runtime       │   │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘   │
│           │                   │                    │                │
│           └───────────────────┴────────────────────┘                │
│                               │                                         │
│                               ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              BACKEND EXECUTION                                │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  CPU: Native (x86-64), Cranelift                             │   │
│  │  GPU: PTX (NVIDIA), SPIR-V (Vulkan), MSL (Metal)            │   │
│  │  Quantum: Simulator ou Hardware (Qiskit, Cirq)                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Referências

- [`crates/souc/src/linear/mod.rs`](../../crates/souc/src/linear/mod.rs)
- [`crates/souc/src/geometry/mod.rs`](../../crates/souc/src/geometry/mod.rs)
- [`crates/souc/src/quantum/`](../../crates/souc/src/quantum/)
- [`crates/souc/src/optimizer/`](../../crates/souc/src/optimizer/)
- [`crates/souc/src/epistemic/pce.rs`](../../crates/souc/src/epistemic/pce.rs)
- [`crates/souc/src/epistemic/mcmc.rs`](../../crates/souc/src/epistemic/mcmc.rs)
- [`crates/souc/src/epistemic/gaussian_process.rs`](../../crates/souc/src/epistemic/gaussian_process.rs)
- [`docs/compiler/OCTONION_ALGEBRA.md`](../../docs/compiler/OCTONION_ALGEBRA.md)
