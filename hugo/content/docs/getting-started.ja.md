---
title: "Getting Started with Sounio"
description: "Installation and first steps with the Sounio compiler"
layout: "docs"
---

**Sounio**へようこそ。Sounioは、epistemic computingのためのシステムプログラミング言語で、すべての値がその不確実性を運ぶことができます。

## インストール

### バイナリから（推奨）

プラットフォーム用の最新リリースをダウンロードしてください：

```bash
# Linux/macOS
curl -sSf https://souniolang.org/install.sh | sh

# Or download directly
wget https://github.com/sounio-lang/sounio/releases/latest/download/souc-linux-x64.tar.gz
tar xzf souc-linux-x64.tar.gz
sudo mv souc /usr/local/bin/
```

### ソースから

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio/compiler
cargo build --release
sudo cp target/release/souc /usr/local/bin/
```

### インストールの確認

```bash
souc --version
# souc 0.93.0
```

## 最初のプログラム

ファイル`hello.sio`を作成してください：

```sounio
fn main() -> i32 {
    print("Hello, Sounio!")
    println()
    0
}
```

コンパイルして実行してください：

```bash
souc run hello.sio
# Output: Hello, Sounio!
```

または、型チェックのみを行う：

```bash
souc check hello.sio
```

## 主要な概念

### 1. Epistemic Types

Sounioの特徴的な機能は、`Knowledge<T>`型です — 不確実性を運ぶ値です：

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

### 2. 変数

```sounio
let x = 5              // immutable
var y = 10             // mutable

y = y + 1              // OK: y is mutable
// x = 6               // Error: x is immutable
```

### 3. 参照

Sounioでは、可変参照に`&!`を使用します（Rustの`&mut`とは異なります）：

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

### 4. 物理単位

型安全な次元解析：

```sounio
let distance: f64<m> = 100.0 m
let time: f64<s> = 9.58 s
let speed = distance / time  // Type: f64<m/s>

// Compile error: can't add meters and seconds
// let invalid = distance + time
```

### 5. Effects

関数は副作用を宣言します：

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

薬物動態学のためのドメイン特化構文：

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

## プロジェクト構造

典型的なSounioプロジェクト：

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

## コマンドリファレンス

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

## 例

`examples/`ディレクトリには、多くの動作する例が含まれています：

| File | Description |
|------|-------------|
| `hello.sio` | ハロー ワールド |
| `fibonacci.sio` | 再帰的および反復的なフィボナッチ数列 |
| `uncertainty.sio` | Knowledge<T>の不確実性伝播 |
| `pkpd.sio` | 2コンパートメントPKモデル |
| `effects.sio` | 代数的効果のデモ |
| `gpu.sio` | GPUカーネルの例 |
| `ode_demo.sio` | ODEの解法 |
| `autodiff.sio` | 自動微分 |

任意の例を実行してください：

```bash
cd examples
souc run hello.sio
souc run fibonacci.sio
souc run uncertainty.sio
```

## 次の一歩

- [Language Reference](./LLM_PROGRAMMING_GUIDE.md) — 完全な構文ガイド
- [Standard Library](../stdlib/) — 標準ライブラリの閲覧
- [Examples](../examples/) — 動作するコード例
- [CHANGELOG](../CHANGELOG.md) — バージョン履歴

## ヘルプの入手

- **GitHub Issues**: [sounio-lang/sounio](https://github.com/sounio-lang/sounio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sounio-lang/sounio/discussions)
- **Website**: [souniolang.org](https://souniolang.org)

---

🏛️ **Sounio** — 確実性の地平で計算する
