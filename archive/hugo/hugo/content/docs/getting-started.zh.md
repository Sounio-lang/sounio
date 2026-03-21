---
title: "Getting Started with Sounio"
description: "Installation and first steps with the Sounio compiler"
layout: "docs"
---

欢迎来到 **Sounio**，一种用于 epistemic computing 的系统编程语言——每个值都可以携带其不确定性。

## 安装

### 从二进制文件（推荐）

为您的平台下载最新版本：

```bash
# Linux/macOS
curl -sSf https://souniolang.org/install.sh | sh

# Or download directly
wget https://github.com/sounio-lang/sounio/releases/latest/download/souc-linux-x64.tar.gz
tar xzf souc-linux-x64.tar.gz
sudo mv souc /usr/local/bin/
```

### 从源代码

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio/compiler
cargo build --release
sudo cp target/release/souc /usr/local/bin/
```

### 验证安装

```bash
souc --version
# souc 0.93.0
```

## 您的第一个程序

创建一个文件 `hello.sio`：

```sounio
fn main() -> i32 {
    print("Hello, Sounio!")
    println()
    0
}
```

编译并运行：

```bash
souc run hello.sio
# Output: Hello, Sounio!
```

或者仅检查类型：

```bash
souc check hello.sio
```

## 关键概念

### 1. Epistemic 类型

Sounio 的标志性特性是 `Knowledge<T>` 类型——携带不确定性的值：

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

### 2. 变量

```sounio
let x = 5              // immutable
var y = 10             // mutable

y = y + 1              // OK: y is mutable
// x = 6               // Error: x is immutable
```

### 3. 引用

Sounio 使用 `&!` 表示可变引用（不像 Rust 中的 `&mut`）：

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

### 4. 物理单位

类型安全的维度分析：

```sounio
let distance: f64<m> = 100.0 m
let time: f64<s> = 9.58 s
let speed = distance / time  // Type: f64<m/s>

// Compile error: can't add meters and seconds
// let invalid = distance + time
```

### 5. 效果

函数声明其副作用：

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

用于药代动力学的领域特定语法：

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

## 项目结构

一个典型的 Sounio 项目：

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

## 命令参考

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

## 示例

`examples/` 目录包含许多可运行的示例：

| File | Description |
|------|-------------|
| `hello.sio` | Hello World |
| `fibonacci.sio` | Recursive and iterative Fibonacci |
| `uncertainty.sio` | Knowledge<T> uncertainty propagation |
| `pkpd.sio` | Two-compartment PK model |
| `effects.sio` | Algebraic effects demo |
| `gpu.sio` | GPU kernel example |
| `ode_demo.sio` | ODE solving |
| `autodiff.sio` | Automatic differentiation |

运行任何示例：

```bash
cd examples
souc run hello.sio
souc run fibonacci.sio
souc run uncertainty.sio
```

## 后续步骤

- [Language Reference](./LLM_PROGRAMMING_GUIDE.md) — 完整的语法指南
- [Standard Library](../stdlib/) — 浏览标准库
- [Examples](../examples/) — 可运行的代码示例
- [CHANGELOG](../CHANGELOG.md) — 版本历史

## 获取帮助

- **GitHub Issues**: [sounio-lang/sounio](https://github.com/sounio-lang/sounio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sounio-lang/sounio/discussions)
- **Website**: [souniolang.org](https://souniolang.org)

---

🏛️ **Sounio** — 在确定性的地平线上计算
