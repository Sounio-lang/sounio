---
title: "Contributing to Sounio"
description: "How to contribute code, documentation, and ideas to the Sounio project"
layout: "contributing"
---

感谢您对 Sounio 的贡献兴趣！本文档提供了贡献的指南和说明。

## 行为准则

尊重他人。具有建设性。保持耐心。我们正在构建一些重要的东西。

## 入门指南

### 先决条件

- **Rust 1.70+** — 编译器是用 Rust 编写的
- **Git** — 版本控制
- **LLVM 15+** (可选) — 用于 LLVM 后端

### 从源代码构建

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

## 开发工作流程

### 1. Fork 和克隆

```bash
git clone https://github.com/YOUR_USERNAME/sounio.git
cd sounio
git remote add upstream https://github.com/sounio-lang/sounio.git
```

### 2. 创建分支

```bash
git checkout -b feature/your-feature-name
```

分支命名约定：
- `feature/` — 新功能
- `fix/` — 错误修复
- `docs/` — 文档
- `refactor/` — 代码重构
- `test/` — 测试添加

### 3. 进行更改

- 遵循下面的代码风格指南
- 为新功能添加测试
- 必要时更新文档

### 4. 测试您的更改

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

### 5. 提交

遵循提交消息格式：

```
[component] Brief description

Components: lexer, parser, ast, check, types, effects, hir, hlir,
           codegen, cli, docs, stdlib, tests, epistemic
```

示例：
```
[parser] Add support for Knowledge<T> generic syntax
[stdlib] Implement bootstrap_correlation in connectivity module
[docs] Update README with new examples
```

### 6. 推送并创建 PR

```bash
git push origin feature/your-feature-name
```

然后在 GitHub 上创建拉取请求。

## 代码风格指南

### Rust (编译器)

- 使用 `rustfmt` 进行格式化
- 在提交前运行 `clippy`
- 库代码中不要使用 `unwrap()` — 使用 `?` 或适当的错误处理
- 使用 `thiserror` 定义错误类型
- 使用 `miette` 处理带有源代码跨度的诊断信息
- 所有公共项都需要文档注释

### Sounio (标准库)

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

## 贡献内容

### 高优先级

- [ ] 语言服务器协议 (LSP) 实现
- [ ] LLVM 后端优化
- [ ] 包管理器 (`siopkg`)
- [ ] 交互式 REPL
- [ ] 更多标准库模块

### 中优先级

- [ ] 文档改进
- [ ] 示例程序
- [ ] 性能基准测试
- [ ] 编辑器集成

### 始终欢迎

- 错误修复
- 测试覆盖率改进
- 文档澄清
- 拼写错误修复

## 标准库贡献

标准库 (`stdlib/`) 包含特定领域的模块：

| Module | Description |
|--------|-------------|
| `epistemic/` | 核心不确定性类型 |
| `medlang/` | PK/PD 建模 DSL |
| `fmri/` | 神经成像管道 |
| `causal/` | 因果推理 |
| `connectivity/` | 网络分析 |
| `gpu/` | GPU 加速 |
| `optimize/` | 优化 |
| `signal/` | 信号处理 |
| `data/` | DataFrames |
| `mcmc/` | MCMC 采样 |
| `random/` | RNG |
| `quantum/` | 量子计算 |
| `linalg/` | 线性代数 |
| `ode/` | ODE 求解器 |
| `bayes/` | 贝叶斯推理 |

添加标准库内容时：
1. 遵循模块中现有的模式
2. 在适当位置包含不确定性传播
3. 添加全面的文档注释
4. 编写测试

## 问题？

- 对于错误或功能请求，打开一个 issue
- 对于问题，使用讨论区
- 在创建新 issue 前检查现有 issue

## 许可

通过贡献，您同意您的贡献将根据 MIT 许可进行许可。

---

*感谢您帮助构建认知计算的未来！* 🏛️
