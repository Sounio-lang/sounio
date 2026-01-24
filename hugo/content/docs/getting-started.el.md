---
title: "Getting Started with Sounio"
description: "Installation and first steps with the Sounio compiler"
layout: "docs"
---

Καλώς ήρθατε στο **Sounio**, μια γλώσσα προγραμματισμού συστημάτων για **epistemic computing** — όπου κάθε τιμή μπορεί να μεταφέρει την αβεβαιότητά της.

## Εγκατάσταση

### Από Δυαδικό Αρχείο (Συνιστώμενο)

Κατεβάστε την τελευταία έκδοση για την πλατφόρμα σας:

```bash
# Linux/macOS
curl -sSf https://souniolang.org/install.sh | sh

# Or download directly
wget https://github.com/sounio-lang/sounio/releases/latest/download/souc-linux-x64.tar.gz
tar xzf souc-linux-x64.tar.gz
sudo mv souc /usr/local/bin/
```

### Από Πηγή

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio/compiler
cargo build --release
sudo cp target/release/souc /usr/local/bin/
```

### Επαλήθευση Εγκατάστασης

```bash
souc --version
# souc 0.93.0
```

## Το Πρώτο σας Πρόγραμμα

Δημιουργήστε ένα αρχείο `hello.sio`:

```sounio
fn main() -> i32 {
    print("Hello, Sounio!")
    println()
    0
}
```

Μεταγλωττίστε και εκτελέστε:

```bash
souc run hello.sio
# Output: Hello, Sounio!
```

Ή απλώς ελέγξτε τύπους:

```bash
souc check hello.sio
```

## Βασικές Έννοιες

### 1. Epistemic Types

Η χαρακτηριστική δυνατότητα του Sounio είναι ο τύπος `Knowledge<T>` — τιμές που μεταφέρουν την αβεβαιότητά τους:

```sounio
import sounio::epistemic::*

fn main() -> i32 {
    // Value with uncertainty
    let measurement = Knowledge::new(
        value: 42.0,
        uncertainty: 0.5,
        confidence: 0.95
    )

    // Uncertainty propagates through operations
    let doubled = measurement.mul(Knowledge::exact(2.0))

    print(doubled.to_string())
    // Output: 84.0000 +/- 1.9600 (95% CI)

    0
}
```

### 2. Μεταβλητές

```sounio
let x = 5              // immutable
var y = 10             // mutable

y = y + 1              // OK: y is mutable
// x = 6               // Error: x is immutable
```

### 3. Αναφορές

Το Sounio χρησιμοποιεί `&!` για μεταβλητές αναφορές (όχι `&mut` όπως στη Rust):

```sounio
fn increment(x: &!i32) {
    *x = *x + 1
}

fn main() -> i32 {
    var value = 10
    increment(&!value)
    print(value)  // 11
    0
}
```

### 4. Φυσικές Μονάδες

Ασφαλής ανάλυση διαστάσεων με τύπους:

```sounio
let distance: f64<m> = 100.0 m
let time: f64<s> = 9.58 s
let speed = distance / time  // Type: f64<m/s>

// Compile error: can't add meters and seconds
// let invalid = distance + time
```

### 5. Επιδράσεις

Οι συναρτήσεις δηλώνουν τις παρενέργειές τους:

```sounio
fn read_file(path: &str) -> String with IO {
    // Can perform I/O
}

fn pure_function(x: i32) -> i32 {
    // No effects allowed
    x * 2
}
```

### 6. MedLang DSL

Ειδικός σύνταξη για φαρμακοκινητική:

```sounio
import sounio::medlang::*

model OneCompartment {
    param CL: Knowledge<f64> = Knowledge::new(
        value: 10.0,
        uncertainty: 3.0,
        confidence: 0.95
    )
    param V: Knowledge<f64> = Knowledge::new(
        value: 50.0,
        uncertainty: 12.5,
        confidence: 0.95
    )

    compartment Central { volume: V }
    flow Central -> Elimination: CL

    observe Cp = Central.concentration
}
```

## Δομή Έργου

Ένα τυπικό έργο Sounio:

```
my_project/
├── src/
│   ├── main.sio
│   └── lib.sio
├── tests/
│   └── test_main.sio
├── examples/
│   └── demo.sio
└── sounio.toml
```

## Αναφορά Εντολών

```bash
# Type-check a file
souc check file.sio

# Run a file (JIT compilation)
souc run file.sio

# Compile to executable
souc build file.sio -o output

# Show AST
souc check file.sio --show-ast

# Show types
souc check file.sio --show-types

# Watch mode (recompile on changes)
souc watch file.sio

# Get help
souc --help
```

## Παραδείγματα

Ο κατάλογος `examples/` περιέχει πολλά λειτουργικά παραδείγματα:

| Αρχείο | Περιγραφή |
|--------|-----------|
| `hello.sio` | Hello World |
| `fibonacci.sio` | Αναδρομικός και επαναληπτικός Fibonacci |
| `uncertainty.sio` | Διάδοση αβεβαιότητας Knowledge<T> |
| `pkpd.sio` | Μοντέλο PK δύο θαλάμων |
| `effects.sio` | Δείγμα αλγεβρικών επιδράσεων |
| `gpu.sio` | Παράδειγμα πυρήνα GPU |
| `ode_demo.sio` | Λύση ODE |
| `autodiff.sio` | Αυτόματη διαφοροποίηση |

Εκτελέστε οποιοδήποτε παράδειγμα:

```bash
cd examples
souc run hello.sio
souc run fibonacci.sio
souc run uncertainty.sio
```

## Επόμενα Βήματα

- [Language Reference](./LLM_PROGRAMMING_GUIDE.md) — Ολοκληρωμένος οδηγός σύνταξης
- [Standard Library](../stdlib/) — Περιήγηση στη stdlib
- [Examples](../examples/) — Λειτουργικά παραδείγματα κώδικα
- [CHANGELOG](../CHANGELOG.md) — Ιστορικό εκδόσεων

## Λήψη Βοήθειας

- **GitHub Issues**: [sounio-lang/sounio](https://github.com/sounio-lang/sounio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sounio-lang/sounio/discussions)
- **Website**: [souniolang.org](https://souniolang.org)

---

🏛️ **Sounio** — Compute at the Horizon of Certainty
