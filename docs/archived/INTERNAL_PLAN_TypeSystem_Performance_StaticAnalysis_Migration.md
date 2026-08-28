<!-- docs:meta
topic_id: repo.docs.archived.internal-plan-typesystem-performance-staticanalysis-migration
authority: archived
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.archived.internal-plan-typesystem-performance-staticanalysis-migration
-->


<!-- docs:status-note:start -->
> Docs status: `archived`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio Development Plan
## Type System, Performance, Static Analysis & Migration Tools

---

## 1. TYPE SYSTEM ENHANCEMENTS 🎯

### Current State
The Sounio compiler (self-hosted in `self-hosted/`) already has:
- Basic type expressions (named types, references, arrays, tuples)
- Generic arguments (`Option<T>`)
- Function types
- Knowledge types (epistemic)
- Refinement types (`knowledge_info`, `refinement_info`)
- Effects system (`with IO, Mut, Panic`)

### Proposed Additions

#### 1.1 Higher-Kinded Types (HKTs)
**What**: Types that abstract over type constructors
```sounio
// Container is a type constructor: * -> *
trait Container<F<_>> {
    fn map<A, B>(self: F<A>, f: (A) -> B) -> F<B>
}

// Instances for List, Option, etc.
impl Container<List<_>> for List<T> { ... }
impl Container<Option<_>> for Option<T> { ... }
```

**Files to modify**:
- `self-hosted/parser/types.sio` - Add HKT syntax parsing
- `self-hosted/parser/ast.sio` - Add AST nodes for type constructors
- `self-hosted/check/` (type checker) - Add kind checking

**Complexity**: HIGH - Requires kind system (*, * -> *, etc.)

#### 1.2 Generalized Algebraic Data Types (GADTs)
**What**: Types where constructors can have different type parameters
```sounio
enum Expr<T> {
    IntLit(i64) where T = i64,
    BoolLit(bool) where T = bool,
    Add(Box<Expr<i64>>, Box<Expr<i64>>>) where T = i64,
}
```

**Files to modify**:
- `self-hosted/parser/items.sio` - Parse GADT constructors
- `self-hosted/check/` - Add type equality constraints

**Complexity**: HIGH - Requires constraint-based type checking

#### 1.3 Associated Types
**What**: Type members of traits
```sounio
trait Iterator {
    type Item
    fn next(self) -> Option<Self::Item>
}

impl Iterator for List<T> {
    type Item = T
    fn next(self) -> Option<T> { ... }
}
```

**Complexity**: MEDIUM

#### 1.4 Type Families / Type Functions
**What**: Compute types from types
```sounio
type family Element<C> {
    List<T> => T
    Map<K, V> => (K, V)
    Option<T> => T
}
```

**Complexity**: HIGH

#### 1.5 Row Polymorphism
**What**: Structural typing for records
```sounio
fn get_name<T>(r: {name: String, ...T}) -> String {
    r.name
}
// Works with any record containing 'name: String'
```

**Complexity**: MEDIUM

---

## 2. PERFORMANCE OPTIMIZATIONS ⚡

### 2.1 Compiler Speed

#### Current Bottlenecks (to investigate)
- Lexer: Single-threaded, per-character processing
- Parser: Recursive descent with backtracking
- Type checker: No incremental checking
- No caching between compilations

#### Proposed Optimizations

| Optimization | Impact | Effort | Files |
|--------------|--------|--------|-------|
| **Parallel lexing** | 2-4x | Medium | `self-hosted/lexer/` |
| **Memoization** | 1.5-2x | Low | `self-hosted/parser/` |
| **Incremental compilation** | 5-10x | High | New module: `self-hosted/incremental/` |
| **Module caching** | 3-5x | Medium | `self-hosted/compiler/module_loader.sio` |
| **Parallel type checking** | 2-3x | High | `self-hosted/check/` |

#### Quick Wins
```sounio
// 1. Add simple parser memoization
fn parse_expr_memo(self) -> Expr with Cache {
    let key = (self.pos, self.lookahead)
    match self.cache.get(key) {
        Some(result) => result,
        None => {
            let result = self.parse_expr()
            self.cache.insert(key, result)
            result
        }
    }
}
```

### 2.2 Memory Usage

#### Current Issues
- AST nodes allocated individually (no arenas)
- String duplication throughout
- No sharing of common types

#### Proposed Solutions

| Optimization | Impact | Effort |
|--------------|--------|--------|
| **Arena allocation** | -50% memory | Medium |
| **String interning** | -30% memory | Low |
| **Flyweight types** | -20% memory | Low |
| **Streaming parser** | -40% memory | High |

#### Implementation: String Interning
```sounio
// self-hosted/intern.sio (already exists!)
mod intern {
    type InternPool {
        strings: Map<Hash, StringId>
        data: Vec<StringData>
    }
    
    fn intern(pool: &mut InternPool, s: String) -> StringId {
        let hash = hash_string(s)
        match pool.strings.get(hash) {
            Some(id) => id,
            None => {
                let id = pool.data.len()
                pool.data.push(s)
                pool.strings.insert(hash, id)
                id
            }
        }
    }
}
```

### 2.3 Binary Size

#### Current: 243MB (debug), ~?? (release)

#### Proposed Optimizations

| Technique | Impact | Effort |
|-----------|--------|--------|
| **Strip symbols** | -50MB | Trivial |
| **LTO (Link Time Optimization)** | -30% | Low |
| **Dead code elimination** | -20% | Medium |
| **Split debug info** | -40MB | Low |
| **Compression** | -60% | Low |

#### Quick Implementation
```bash
# Add to build scripts
strip target/release/souc
# Or in Cargo.toml (if using Cargo)
[profile.release]
lto = true
strip = true
opt-level = 3
```

---

## 3. STATIC ANALYSIS TOOLS 🔍

### 3.1 Dead Code Detection

#### What to Detect
- Unused functions
- Unused variables
- Unused imports
- Unreachable code
- Unused types

#### Implementation
```sounio
// self-hosted/analyze/dead_code.sio

fn analyze_dead_code(module: Module) -> Vec<Diagnostic> {
    let mut used = Set<SymbolId>::new()
    let mut defined = Set<SymbolId>::new()
    
    // Collect all definitions
    for item in module.items {
        match item {
            Item::Fn(f) => {
                defined.insert(f.id)
                // Mark used symbols in body
                used.extend(collect_uses(f.body))
            }
            Item::Import(i) => defined.insert(i.id),
            _ => {}
        }
    }
    
    // Find dead code
    let dead = defined.difference(used)
    dead.map(|id| Diagnostic::warning(id, "Dead code: unused item"))
}
```

**Integration**: Add `sounio analyze --dead-code` command

### 3.2 Complexity Metrics

#### Metrics to Compute
- Cyclomatic complexity
- Cognitive complexity
- Lines of code
- Function length
- Nesting depth
- Parameter count

#### Implementation
```sounio
// self-hosted/analyze/complexity.sio

struct ComplexityMetrics {
    cyclomatic: i64,      // Decision points + 1
    cognitive: i64,       // Nested conditionals
    lines: i64,
    nesting_depth: i64,
}

fn compute_complexity(fn: Function) -> ComplexityMetrics {
    let mut cyclo = 1
    let mut cognitive = 0
    let mut max_depth = 0
    
    for stmt in fn.body {
        match stmt {
            Stmt::If(_) => {
                cyclo += 1
                cognitive += current_depth
            }
            Stmt::Match(arms) => {
                cyclo += arms.len()
                cognitive += current_depth * arms.len()
            }
            Stmt::While(_) | Stmt::For(_) => {
                cyclo += 1
                cognitive += current_depth
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            }
            _ => {}
        }
    }
    
    ComplexityMetrics {
        cyclomatic: cyclo,
        cognitive: cognitive,
        lines: fn.line_count(),
        nesting_depth: max_depth,
    }
}
```

**Command**: `sounio analyze --complexity`

### 3.3 Additional Analysis Passes

| Analysis | Description | Priority |
|----------|-------------|----------|
| **Ownership analysis** | Verify linear/affine types | HIGH |
| **Effect tracking** | Verify effect polymorphism | HIGH |
| **Pattern exhaustiveness** | Ensure match completeness | MEDIUM |
| **Unused mutability** | Find unnecessary `mut` | LOW |
| **Shadowing detection** | Warn on variable shadowing | LOW |
| **Panic analysis** | Find potentially panicking code | MEDIUM |
| **Performance hints** | Suggest optimizations | LOW |

### 3.4 IDE Integration

Add LSP methods for analysis:
```
textDocument/diagnostic (pull model)
workspace/diagnostic (full workspace)
```

---

## 4. MIGRATION TOOLS 🔄

### 4.1 Auto-Upgrade System

#### Version Tracking
```sounio
// migration/version.sio

enum LanguageVersion {
    V0_1_0,  // Initial
    V0_2_0,  // Added epistemic types
    V0_3_0,  // Changed effect syntax
    V1_0_0,  // Stable release
}

struct Migration {
    from: LanguageVersion,
    to: LanguageVersion,
    transforms: Vec<Transform>,
}
```

#### Transformation Rules
```sounio
// migration/transforms.sio

trait Transform {
    fn apply(self, file: &mut File) -> Result<ChangeSet, Error>
    fn description(self) -> String
}

// Example: Effect syntax change
struct EffectSyntaxTransform {
    old_pattern: Regex,
    new_template: String,
}

impl Transform for EffectSyntaxTransform {
    fn apply(self, file: &mut File) -> Result<ChangeSet, Error> {
        // Find: `with IO, Mut`
        // Replace: `with IO + Mut`
        file.replace_all(self.old_pattern, self.new_template)
    }
}
```

### 4.2 Migration Commands

```bash
# Check what needs migration
sounio migrate --check --from 0.1.0 --to 0.2.0

# Preview changes
sounio migrate --dry-run --from 0.1.0 --to 0.2.0

# Apply changes
sounio migrate --from 0.1.0 --to 0.2.0

# Auto-detect version and migrate to latest
sounio migrate
```

### 4.3 Breaking Change Categories

| Category | Handling Strategy |
|----------|-------------------|
| **Syntax changes** | Auto-rewrite with regex/AST transform |
| **Semantic changes** | Warning + manual migration guide |
| **Removed features** | Error with suggestion |
| **Library renames** | Auto-update imports |

### 4.4 Migration File Format

```yaml
# migrations/v0_2_0.yaml
version: 0.2.0
breaking_changes:
  - name: "effect-syntax"
    description: "Effects now use + instead of ,"
    severity: "error"
    auto_fix: true
    pattern: 'with\s+([A-Za-z]+),\s*([A-Za-z]+)'
    replacement: 'with $1 + $2'
    
  - name: "knowledge-keyword"
    description: "'knowledge' renamed to 'epistemic'"
    severity: "error"
    auto_fix: true
    pattern: '\bknowledge\b'
    replacement: 'epistemic'
    
  - name: "effect-polymorphism"
    description: "Effect polymorphism syntax changed"
    severity: "warning"
    auto_fix: false
    help_url: "https://sounio.org/migrate/effects"
```

---

## IMPLEMENTATION PRIORITY 📊

### Phase 1: Quick Wins (1-2 weeks)
1. ✅ String interning (memory)
2. ✅ Strip symbols (binary size)
3. ✅ Dead code detection (analysis)
4. ✅ Complexity metrics (analysis)

### Phase 2: Core Improvements (1-2 months)
1. Parser memoization (performance)
2. Module caching (performance)
3. Basic migration tool (migration)
4. Associated types (type system)

### Phase 3: Advanced Features (2-3 months)
1. Incremental compilation (performance)
2. GADTs (type system)
3. Higher-kinded types (type system)
4. Full migration system (migration)

---

## FILES TO EXPLORE 📁

```
self-hosted/
├── main.sio                    # Entry point
├── parser/
│   ├── types.sio               # Type parsing
│   ├── items.sio               # Item parsing (structs, enums, etc.)
│   ├── ast.sio                 # AST definitions
│   └── parser.sio              # Main parser
├── ir/
│   ├── ir.sio                  # Intermediate representation
│   ├── optimize.sio            # Optimizations
│   └── lower.sio               # Code lowering
├── diagnostics/
│   └── mod.sio                 # Diagnostic reporting
└── compiler/
    └── module_loader.sio       # Module loading

tools/lsp/
└── sounio-lsp.sh               # LSP server
```

---

## NEXT STEPS 🚀

Pick one to start with:

1. **Performance**: Add string interning to reduce memory
2. **Static Analysis**: Implement dead code detection
3. **Migration**: Create basic migration framework
4. **Type System**: Add associated types to traits

Which would you like to tackle first? I can help implement any of these!
