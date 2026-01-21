# WP-4.2: Linear & Affine Types (Resource Management)

## Sounio Syntax Rules (CRITICAL)

- Use `var` for mutable variables
- Use `&!T` for mutable refs
- Array indexing requires `with Panic`

## Reference Implementation

See: `compiler/src/linear/modality.rs`
See: `compiler/src/linear/check.rs`
See: `compiler/src/linear/usage.rs`

## Target Output

**File**: `stdlib/compiler/linear/modality.sio`
**Estimated LOC**: ~2,000

## Specification

Implement linear and affine type checking for resource safety.

### Type Modalities

```sio
// Linearity modalities
fn MODALITY_UNRESTRICTED() -> i32 { 0 }  // Used 0 or ∞ times
fn MODALITY_RELEVANT() -> i32 { 1 }      // Used ≥ 1 times
fn MODALITY_AFFINE() -> i32 { 2 }        // Used ≤ 1 times
fn MODALITY_LINEAR() -> i32 { 3 }        // Used exactly 1 time

struct LinearType {
    modality: i32,     // How many times can be used?
    base_type: i64,    // Type index
}

// Annotation in source:
// `let x: i32 = ...`              // Unrestricted
// `let x: !i32 = ...`             // Relevant (must be used)
// `let x: i32 ~ = ...`            // Affine (can be unused)
// `let x: i32 ~ ! = ...`          // Linear (exactly once)
```

### Usage Tracking

```sio
struct UsageContext {
    // Track which variables have been used
    used_vars: [i64; 64],   // Indices of used variables
    n_used: i64,
    partial_uses: [PartialUse; 64],  // Incomplete pattern matches
    n_partial: i64,
}

struct PartialUse {
    var_idx: i64,
    pattern: Pattern,    // Which constructors consumed?
    remaining: Pattern,  // Remaining to consume
}

// Track current usage of variable
fn mark_used(ctx: &!UsageContext, var_idx: i64) -> bool with Panic {
    // Check if variable already marked used
    var i: i64 = 0;
    while i < ctx.n_used {
        if ctx.used_vars[i as usize] == var_idx {
            return false;  // Already used!
        }
        i = i + 1;
    }
    // Mark as used
    if ctx.n_used < 64 {
        ctx.used_vars[ctx.n_used as usize] = var_idx;
        ctx.n_used = ctx.n_used + 1;
        return true;
    }
    false
}

// Check if variable is dropped without being used
fn check_linearity(ctx: UsageContext, modality: i32, var_idx: i64) -> bool {
    var used = false;
    var i: i64 = 0;
    while i < ctx.n_used {
        if ctx.used_vars[i as usize] == var_idx {
            used = true;
        }
        i = i + 1;
    }

    // Relevant: must be used (≥ 1)
    if modality == MODALITY_RELEVANT() {
        return used;
    }

    // Affine: optional (≤ 1) - always OK
    // Linear: exactly 1 - must be used exactly once
    if modality == MODALITY_LINEAR() {
        return used;  // TODO: check not used multiple times
    }

    true
}
```

### Resource Types (Session Types)

```sio
// Channels for linear communication
struct Channel {
    direction: i32,   // 0 = send, 1 = receive
    payload_type: i64,
}

// Session protocol (linear state machine)
struct SessionType {
    states: [State; 16],
    n_states: i64,
    initial: i64,
}

struct State {
    name: [i8; 32],
    transitions: [Transition; 8],
    n_trans: i64,
    is_terminal: bool,
}

struct Transition {
    action: i32,       // send/receive
    payload_type: i64,
    next_state: i64,
}

// Example: file handle (must be opened, used, then closed)
fn file_handle_session() -> SessionType {
    // States: Closed -> Open -> Closed
    // Transitions: Open (Closed→Open), Read (Open→Open), Close (Open→Closed)
}
```

### Pattern Matching with Linear Types

For pattern matching to consume linear resources:

```sio
// Match on linear type consumes it
fn check_linear_pattern_match(ctx: &!UsageContext, scrutinee_idx: i64, patterns: [Pattern; 8], n: i64) -> bool {
    // Each pattern must consume scrutinee
    // If any pattern doesn't use scrutinee → error
    var all_patterns_use = true;
    var i: i64 = 0;
    while i < n {
        // Check if pattern[i] uses scrutinee
        // For linear types, pattern must handle all constructors
        i = i + 1;
    }
    all_patterns_use
}

// Mutable borrow restrictions
fn check_borrow_safety(ctx: &!UsageContext, var_idx: i64, borrow_kind: i32) -> bool {
    // borrow_kind: 0 = shared (&), 1 = exclusive (&!)
    // exclusive borrow: cannot have other borrows
    // shared borrow: cannot have exclusive borrow
    true
}
```

### Drop & RAII

```sio
// Implicit drop at scope exit
fn drop_resource(resource: LinearType) -> bool with Panic {
    // For linear: must have been explicitly consumed
    // For affine: OK to drop
    // For unrestricted: always OK
    true
}

// RAII: automatic resource cleanup
struct ResourceGuard {
    resource_idx: i64,
    cleanup_fn: i64,  // Function to call on drop
}

fn scope_exit(guard: ResourceGuard) -> bool {
    // Call cleanup_fn before exiting scope
    true
}
```

### Type Checking Rules

1. **Linear**: Used exactly once → must explicitly consume or transfer
2. **Affine**: Used ≤ once → can be ignored (implicit drop)
3. **Relevant**: Used ≥ once → cannot be unused
4. **Unrestricted**: Used 0+ times → standard type

### Example: File Handle

```
linear fn process_file(path: string) {
    let f: FileHandle = open(path);  // Linear resource
    let data = read(f);              // Consumes 'f'
    close(f);                        // 'f' already consumed - ERROR
}

linear fn process_file_correct(path: string) {
    let f: FileHandle = open(path);
    let data = read(f);
    // 'f' consumed implicitly (not needed after read)
}

affine fn maybe_process_file(path: string) {
    let f: FileHandle = open(path);
    // Can be ignored (affine allows unused resources)
}
```

### Key Insight

Linear types prevent use-after-free and ensure resources are used exactly once. Affine types relax to allow optional use. This provides memory safety without garbage collection, crucial for systems programming.
