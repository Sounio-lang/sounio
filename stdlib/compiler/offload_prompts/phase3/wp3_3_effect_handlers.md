# WP-3.3: Effect Handlers & Capabilities

## Sounio Syntax Rules (CRITICAL)

- Use `var` for mutable variables
- Array indexing requires `with Panic`
- NO type suffixes

## Reference Implementation

See: `compiler/src/effects/handler.rs`
See: `compiler/src/capabilities.rs`

## Target Output

**File**: `stdlib/compiler/effects/handler.sio`
**Estimated LOC**: ~1,500

## Specification

Implement effect handlers for handling and masking effects, plus capability-based security.

### Effect Handlers

Effect handlers are control structures that intercept effects:

```sio
struct EffectHandler {
    handled_effects: i64,  // Which effects this handler covers
    handler_fn: i64,       // Index to handler function
    return_type_idx: i64,  // Return type
}

// Handle IO effects
fn handle_io(comp: Computation) -> Result {
    // comp may use IO
    // handler intercepts IO calls and provides implementations
}

// Example handler for test mode
fn mock_read_file(path: string) -> string {
    // Returns mock data instead of reading file
    "mock file content"
}
```

### Capability System

Capabilities are security tokens that grant access to effects:

```sio
struct IOCapability {
    level: i32,     // 0 = read-only, 1 = read-write
    paths: [string; 16],
    n_paths: i64,
}

struct MutCapability {
    // Determines which references can be mutated
    regions: [Region; 32],
    n_regions: i64,
}

// All capabilities together
struct Capabilities {
    io_cap: IOCapability,
    mut_cap: MutCapability,
    alloc_cap: bool,      // Can allocate memory?
    gpu_cap: bool,        // Can use GPU?
    // etc.
}
```

### Handler Implementation

```sio
// Type of effect handler
struct HandlerImpl {
    effect_kind: i32,
    handler_name: [i8; 64],

    // Handler operations
    handle_op: (*i64) -> i64,     // Function pointer (placeholder)
    resume: (*i64) -> i64,        // Resume computation
}

// Register a handler for an effect
fn register_handler(ctx: &!TypeContext, effect: i32, handler: HandlerImpl) -> bool with Panic {
    // Add handler to handler stack
    // Return success if registered
}

// Execute computation with handler
fn with_handler(handler: EffectHandler, comp: Computation) -> Result {
    // 1. Push handler onto stack
    // 2. Execute computation (which may trigger effects)
    // 3. Effects are intercepted by handler
    // 4. Pop handler from stack
    // 5. Return result
}
```

### Effect Masking

Effects can be hidden (masked) after handling:

```sio
// After handling IO, the result type no longer has IO effect
fn handle_io_in_computation(comp: Computation with IO) -> Result {
    // Computation with IO effect
    let result = with_handler(io_handler, comp);
    // Result has no IO effect (masked/handled)
    result
}

fn mask_effect(effect_set: i64, effect_to_mask: i32) -> i64 with Panic {
    // Remove effect from set
    let bit = 1 << (effect_to_mask as i64);
    effect_set & ~bit
}
```

### Continuation-Based Handlers

For resumable computations:

```sio
struct ContinuationHandler {
    state: i64,
    operations: [Operation; 16],
    n_ops: i64,
}

// Yield control to handler and resume later
fn yield_effect(op: Operation) -> Result {
    // Pause computation
    // Handler processes operation
    // Computation resumes with result
}

// Resume from suspended state
fn resume_from_yield(state: i64) -> i64 {
    // Continue execution with result from handler
}
```

### Capability Checking

```sio
// Check if capability allows operation
fn check_io_capability(cap: IOCapability, path: string, op: i32) -> bool {
    // op: 0 = read, 1 = write
    if op == 1 && cap.level < 1 {
        return false;  // No write permission
    }

    // Check if path is in allowed list
    var i: i64 = 0;
    while i < cap.n_paths {
        if paths_equal(cap.paths[i as usize], path) {
            return true;
        }
        i = i + 1;
    }
    false
}

fn paths_equal(p1: string, p2: string) -> bool {
    // String comparison
    true
}

// Restrict capabilities
fn restrict_io(cap: IOCapability, allowed_paths: [string; 8], n: i64) -> IOCapability {
    var new_cap = cap;
    new_cap.n_paths = n;
    var i: i64 = 0;
    while i < n {
        new_cap.paths[i as usize] = allowed_paths[i as usize];
        i = i + 1;
    }
    new_cap
}
```

### Effect Composition

```sio
// Combine multiple effect handlers
fn compose_handlers(h1: EffectHandler, h2: EffectHandler) -> EffectHandler {
    EffectHandler {
        handled_effects: h1.handled_effects | h2.handled_effects,
        // ... other fields
    }
}

// Stack handlers
fn stack_handlers(handlers: [EffectHandler; 8], n: i64) -> EffectHandler {
    var result = EffectHandler { handled_effects: 0, handler_fn: -1, return_type_idx: -1 };
    var i: i64 = 0;
    while i < n {
        result = compose_handlers(result, handlers[i as usize]);
        i = i + 1;
    }
    result
}
```

### Testing Example

```
// Unsafe computation that needs IO
fn read_and_process(path: string) -> i64 with IO { ... }

// Handler that mocks IO
fn test_read_and_process() -> i64 {
    // Create mock handler
    let mock_handler = EffectHandler { ... };

    // Run computation with mock
    let result = with_handler(mock_handler, read_and_process("test.txt"));

    // Result no longer has IO effect (masked by handler)
    result
}
```

### Key Insight

Effect handlers provide abstraction: computations declare effects they use, but handlers can intercept and implement those effects differently (real I/O, mocks, etc.). Capabilities provide security by restricting which effects are allowed.
