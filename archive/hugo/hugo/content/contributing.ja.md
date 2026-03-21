---
title: "Contributing to Sounio"
description: "How to contribute code, documentation, and ideas to the Sounio project"
layout: "contributing"
---

Sounio への貢献にご興味をお持ちいただき、ありがとうございます！このドキュメントでは、貢献するためのガイドラインと指示を提供します。

## 行動規範

敬意を持って行動してください。建設的な態度で臨んでください。忍耐強くお待ちください。私たちは重要なものを築いています。

## 始め方

### 前提条件

- **Rust 1.70+** — コンパイラは Rust で書かれています
- **Git** — バージョン管理
- **LLVM 15+** (オプション) — LLVM バックエンド用

### ソースからビルドする

```bash
# Clone the repository
git clone https://github.com/sounio-lang/sounio.git
cd sounio

# Build the compiler
cd compiler
cargo build --release

# Run tests
cargo test

# Run the compiler
./target/release/souc run examples/hello.sio
```

## 開発ワークフロー

### 1. Fork してクローンする

```bash
git clone https://github.com/YOUR_USERNAME/sounio.git
cd sounio
git remote add upstream https://github.com/sounio-lang/sounio.git
```

### 2. ブランチを作成する

```bash
git checkout -b feature/your-feature-name
```

ブランチの命名規則：
- `feature/` — 新機能
- `fix/` — バグ修正
- `docs/` — ドキュメント
- `refactor/` — コードのリファクタリング
- `test/` — テストの追加

### 3. 変更を加える

- 以下のコードスタイルガイドラインに従ってください
- 新しい機能のためのテストを追加してください
- 必要に応じてドキュメントを更新してください

### 4. 変更をテストする

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy
```

### 5. コミットする

コミットメッセージ形式に従ってください：

```
[component] Brief description

Components: lexer, parser, ast, check, types, effects, hir, hlir,
           codegen, cli, docs, stdlib, tests, epistemic
```

例：
```
[parser] Add support for Knowledge<T> generic syntax
[stdlib] Implement bootstrap_correlation in connectivity module
[docs] Update README with new examples
```

### 6. プッシュして PR を作成する

```bash
git push origin feature/your-feature-name
```

次に GitHub で Pull Request を作成してください。

## コードスタイルガイドライン

### Rust (コンパイラ)

- フォーマットには `rustfmt` を使用してください
- コミット前に `clippy` を実行してください
- ライブラリコードでは `unwrap()` を使用せず、`?` または適切なエラーハンドリングを使用してください
- エラータイプには `thiserror` を使用してください
- ソーススパン付きの診断には `miette` を使用してください
- すべてのパブリックアイテムにドキュメントコメントが必要です

### Sounio (標準ライブラリ)

```sio
// Use descriptive names
fn compute_bootstrap_confidence_interval(data: &[f64], n_boot: i32) -> ConfidenceInterval

// Document functions
/// Computes the modularity of a network using the Louvain algorithm.
///
/// # Arguments
/// * `weights` - Adjacency matrix (N x N)
/// * `resolution` - Resolution parameter (default: 1.0)
///
/// # Returns
/// Modularity value in range [-0.5, 1.0]
fn louvain_modularity(weights: &[[f64]], resolution: f64) -> f64

// Use Knowledge<T> for uncertain values
let result = Knowledge::new(
    value: computed_value,
    uncertainty: computed_uncertainty,
    source: "bootstrap"
)
```

## 何を貢献するか

### 高優先度

- [ ] Language Server Protocol (LSP) の実装
- [ ] LLVM バックエンドの最適化
- [ ] パッケージマネージャー (`siopkg`)
- [ ] インタラクティブ REPL
- [ ] より多くの標準ライブラリモジュール

### 中優先度

- [ ] ドキュメントの改善
- [ ] 例プログラム
- [ ] パフォーマンスベンチマーク
- [ ] エディタ統合

### 常に歓迎

- バグ修正
- テストカバレッジの改善
- ドキュメントの明確化
- タイポ修正

## 標準ライブラリの貢献

標準ライブラリ (`stdlib/`) には、ドメイン固有のモジュールが含まれます：

| Module | Description |
|--------|-------------|
| `epistemic/` | 不確実性のコアタイプ |
| `medlang/` | PK/PD モデリング DSL |
| `fmri/` | 神経画像処理パイプライン |
| `causal/` | 因果推論 |
| `connectivity/` | ネットワーク解析 |
| `gpu/` | GPU 加速 |
| `optimize/` | 最適化 |
| `signal/` | 信号処理 |
| `data/` | DataFrames |
| `mcmc/` | MCMC サンプリング |
| `random/` | RNG |
| `quantum/` | 量子コンピューティング |
| `linalg/` | 線形代数 |
| `ode/` | ODE ソルバー |
| `bayes/` | ベイズ推論 |

標準ライブラリに追加する場合：
1. モジュール内の既存のパターンに従ってください
2. 適切な場所で不確実性の伝播を含めてください
3. 包括的なドキュメントコメントを追加してください
4. テストを書いてください

## 質問は？

- バグや機能リクエストについてはイシューを開いてください
- 質問についてはディスカッションを使用してください
- 新しいイシューを作成する前に既存のイシューを確認してください

## ライセンス

貢献することで、あなたの貢献が MIT License の下でライセンスされることに同意するものとします。

---

*エピステミックコンピューティングの未来を築くお手伝いをいただき、ありがとうございます！* 🏛️
