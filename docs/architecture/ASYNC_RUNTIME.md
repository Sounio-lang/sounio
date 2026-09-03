<!-- docs:meta
topic_id: repo.docs.architecture.async-runtime
authority: historical
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.async-runtime
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio Async Runtime

This document describes the async/await runtime infrastructure for Sounio programs, including task scheduling, combinators, and channels.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Task Scheduler](#task-scheduler)
- [Combinators](#combinators)
  - [Join](#join)
  - [Select](#select)
- [Channels](#channels)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Overview

Sounio's async runtime provides:

- **Task Scheduling**: Priority-based task execution with dependency tracking
- **Join Combinator**: Wait for ALL tasks to complete
- **Select Combinator**: Wait for ANY task to complete (race)
- **Channels**: Bounded and unbounded communication between tasks

The runtime is designed for scientific computing workloads where:
- Tasks may have varying computational costs
- Some tasks depend on results from others
- Epistemic data (uncertainty) must flow through async boundaries

---

## Core Concepts

### Tasks and Futures

A **task** is a unit of async work identified by a `TaskId`. Tasks wrap `SounioFuture` objects that eventually produce a `SounioValue`.

```rust
// TaskId - unique identifier for async tasks
pub struct TaskId(u64);

// Task states
pub enum TaskState {
    Ready,              // Can be polled
    Pending,            // Waiting for event
    Completed(Value),   // Finished with result
    Failed(String),     // Finished with error
    Cancelled,          // Task was cancelled
}
```

### SounioValue

Values that flow through the async system:

```rust
pub enum SounioValue {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Future(TaskId),     // Nested future
    Array(Vec<Value>),
    Tuple(Vec<Value>),
    Struct { name, fields },
    Variant { enum_name, variant_name, fields },
    None,
    Some(Box<Value>),
    Ok(Box<Value>),
    Err(Box<Value>),
}
```

### Effects

Async operations in Sounio require the `Async` effect:

```sio
fn fetch_data(url: string) -> string with Async, IO {
    let response = http_get(url).await
    response.body
}

fn main() with Async {
    let result = spawn { fetch_data("https://api.example.com") }
    println(result.await)
}
```

---

## Task Scheduler

The `TaskScheduler` manages task execution with:

- **Ready queue**: Tasks that can be polled immediately
- **Suspension tracking**: Tasks waiting for events
- **Dependency graphs**: Wake propagation when tasks complete

### Architecture

```
                    ┌─────────────────────┐
                    │   TaskScheduler     │
                    ├─────────────────────┤
                    │ tasks: HashMap      │
                    │ ready: VecDeque     │
                    │ awaiting: HashMap   │
                    │ join_waiters        │
                    │ select_waiters      │
                    │ channel_waiters     │
                    └─────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │  Task 1  │       │  Task 2  │       │  Task 3  │
    │  Ready   │       │ Suspended│       │  Ready   │
    └──────────┘       └──────────┘       └──────────┘
```

### Suspension Reasons

Tasks can be suspended for various reasons:

```rust
pub enum SuspensionReason {
    Await(TaskId),           // Waiting for single task
    JoinAll(Vec<TaskId>),    // Waiting for ALL tasks
    SelectAny(Vec<TaskId>),  // Waiting for ANY task
    ChannelRecv(u64),        // Waiting for channel data
    ChannelSend(u64),        // Waiting for channel space
    Timer(Instant),          // Waiting for timer
}
```

### Scheduling Algorithm

1. **Poll** next ready task from queue
2. If task completes, **wake dependents**
3. If task suspends, register in waiter maps
4. Repeat until no ready tasks (stalled)

```rust
let mut scheduler = TaskScheduler::new();

// Schedule tasks
let id1 = scheduler.schedule(future1);
let id2 = scheduler.schedule(future2);

// Run until stalled
let completed = scheduler.run_until_stalled();
for (id, value) in completed {
    println!("Task {} completed with {:?}", id.0, value);
}
```

### Statistics

```rust
let stats = scheduler.stats();
// SchedulerStats {
//     ready: 5,       // Tasks ready to poll
//     suspended: 2,   // Tasks waiting
//     completed: 10,  // Tasks finished
//     total: 17,      // Total tasks
//     polls: 42,      // Poll operations
//     wakes: 12,      // Wake operations
// }
```

---

## Combinators

### Join

`JoinFuture` waits for **ALL** tasks to complete, returning results as an array.

```sio
// Sounio syntax
let results = join(task1, task2, task3).await
// results: [result1, result2, result3]
```

**Rust API**:

```rust
// Create join from task IDs
let join = JoinFuture::new(vec![id1, id2, id3]);

// Or from handles
let join = join(vec![handle1, handle2, handle3]);

// Convenience functions
let join = join2(a, b);
let join = join3(a, b, c);

// Check completion
if join.is_completed() {
    let results = join.try_get(); // Some(Array([...]))
}
```

**Behavior**:
- Waits for ALL tasks to complete
- Returns results in order (index matches input order)
- If any task fails, the join fails

### Select

`SelectFuture` waits for **ANY** task to complete (first one wins).

```sio
// Sounio syntax
let result = select(task1, task2, timeout).await
match result.index {
    0 => // task1 won
    1 => // task2 won
    2 => // timeout won
}
```

**Rust API**:

```rust
// Create select from task IDs
let select = SelectFuture::new(vec![id1, id2, id3]);

// Or from handles
let select = select(vec![handle1, handle2, handle3]);

// Convenience functions
let select = select2(a, b);
let select = select3(a, b, c);

// Result structure
pub struct SelectResult {
    pub index: usize,      // Which task won
    pub task_id: TaskId,   // Winner's ID
    pub value: SounioValue, // Winner's result
}

if select.is_completed() {
    let result = select.try_get(); // Some(SelectResult { ... })
}
```

**Behavior**:
- Returns immediately when first task completes
- Other tasks continue running (not cancelled)
- Only the first completion is recorded

---

## Channels

Channels provide communication between tasks. Both bounded and unbounded variants are supported.

### Unbounded Channels

```rust
let (tx, rx) = channel::<i32>();

// Send never blocks
tx.send(42).unwrap();
tx.send(43).unwrap();

// Receive values
assert_eq!(rx.recv(), Some(42));
assert_eq!(rx.recv(), Some(43));
assert_eq!(rx.recv(), None); // Empty
```

### Bounded Channels

```rust
let (tx, rx) = bounded_channel::<i32>(2); // Capacity of 2

tx.send(1).unwrap();
tx.send(2).unwrap();

// Channel full!
assert!(matches!(tx.send(3), Err(ChannelError::Full)));

// Drain one
rx.recv();

// Now can send
tx.send(3).unwrap();
```

### Channel Operations

```rust
// Non-blocking receive
match rx.try_recv() {
    Ok(value) => println!("Got {}", value),
    Err(ChannelError::Empty) => println!("Nothing yet"),
    Err(ChannelError::Closed) => println!("Channel closed"),
}

// Check state
rx.is_empty()   // true if no pending values
rx.len()        // number of pending values
rx.is_closed()  // true if channel closed

// Close channel
tx.close();
// After close:
// - send() returns Err(Closed)
// - recv() returns pending values, then None
```

### Channel Errors

```rust
pub enum ChannelError {
    Closed,  // Channel was closed
    Full,    // Bounded channel is full
    Empty,   // No values available
}
```

### Cloning

Both `Sender` and `Receiver` can be cloned for multi-producer/multi-consumer patterns:

```rust
let (tx, rx) = channel();

// Multi-producer
let tx2 = tx.clone();
tx.send(1).unwrap();
tx2.send(2).unwrap();

// Multi-consumer (both see same queue)
let rx2 = rx.clone();
assert_eq!(rx.recv(), Some(1));
assert_eq!(rx2.recv(), Some(2));
```

---

## API Reference

### TaskScheduler

| Method | Description |
|--------|-------------|
| `new()` | Create new scheduler |
| `schedule(future)` | Schedule task, returns TaskId |
| `schedule_with_priority(future, priority)` | Schedule with priority |
| `next_ready()` | Get next ready TaskId |
| `mark_ready(id)` | Mark task as ready to poll |
| `suspend(id, reason)` | Suspend task with reason |
| `wake_dependents(id)` | Wake tasks waiting on this one |
| `poll_next()` | Poll next ready task |
| `run_until_stalled()` | Run until no progress |
| `stats()` | Get scheduler statistics |
| `get_future(id)` | Get task's future |
| `is_completed(id)` | Check if task completed |
| `remove(id)` | Remove completed task |

### JoinFuture

| Method | Description |
|--------|-------------|
| `new(tasks)` | Create from TaskId vec |
| `task_id()` | Get this join's TaskId |
| `tasks()` | Get joined task IDs |
| `record_result(idx, value)` | Record one result |
| `is_completed()` | Check if all complete |
| `try_get()` | Get result if ready |

### SelectFuture

| Method | Description |
|--------|-------------|
| `new(tasks)` | Create from TaskId vec |
| `task_id()` | Get this select's TaskId |
| `tasks()` | Get candidate task IDs |
| `record_completion(idx, id, value)` | Record winner |
| `is_completed()` | Check if any complete |
| `try_get()` | Get SelectResult if ready |

### Channels

| Function | Description |
|----------|-------------|
| `channel<T>()` | Create unbounded channel |
| `bounded_channel<T>(cap)` | Create bounded channel |

| Sender Method | Description |
|--------------|-------------|
| `send(value)` | Send value |
| `try_send(value)` | Non-blocking send |
| `close()` | Close sending end |
| `is_closed()` | Check if closed |
| `channel_id()` | Get channel ID |

| Receiver Method | Description |
|----------------|-------------|
| `recv()` | Receive value (blocking) |
| `try_recv()` | Non-blocking receive |
| `close()` | Close receiving end |
| `is_closed()` | Check if closed |
| `is_empty()` | Check if buffer empty |
| `len()` | Buffer length |
| `channel_id()` | Get channel ID |

---

## Examples

### Basic Async Pattern

```sio
fn compute_intensive(n: i64) -> i64 with Async {
    // Simulated work
    let result = 0
    for i in 0..n {
        result = result + i
    }
    result
}

fn main() with Async, IO {
    // Spawn parallel computations
    let t1 = spawn { compute_intensive(1000) }
    let t2 = spawn { compute_intensive(2000) }
    let t3 = spawn { compute_intensive(3000) }

    // Wait for all
    let results = join(t1, t2, t3).await
    println("Results: ", results)
}
```

### Timeout Pattern

```sio
fn fetch_with_timeout(url: string, timeout_ms: i64) -> Option<string> with Async, IO {
    let fetch_task = spawn { http_get(url) }
    let timeout_task = spawn { sleep(timeout_ms); None }

    let result = select(fetch_task, timeout_task).await
    match result.index {
        0 => Some(result.value)
        1 => None  // Timed out
    }
}
```

### Producer-Consumer

```sio
fn producer(tx: Sender<i64>, count: i64) with Async {
    for i in 0..count {
        tx.send(i)
    }
    tx.close()
}

fn consumer(rx: Receiver<i64>) -> i64 with Async {
    let sum = 0
    while let Some(value) = rx.recv() {
        sum = sum + value
    }
    sum
}

fn main() with Async, IO {
    let (tx, rx) = channel()

    let prod = spawn { producer(tx, 100) }
    let cons = spawn { consumer(rx) }

    let results = join(prod, cons).await
    println("Sum: ", results[1])
}
```

### Fan-Out/Fan-In

```sio
fn worker(id: i64, input: i64) -> i64 with Async {
    // Process input
    input * 2 + id
}

fn main() with Async, IO {
    let inputs = [10, 20, 30, 40, 50]
    var tasks = []

    // Fan-out: spawn workers
    for (i, input) in inputs.enumerate() {
        tasks.push(spawn { worker(i, input) })
    }

    // Fan-in: collect results
    let results = join(tasks).await
    println("Results: ", results)
}
```

---

## Implementation Notes

### Thread Safety

- `SounioFuture` uses `Arc<Mutex<FutureState>>` for thread-safe sharing
- Channels use `Arc<Mutex<ChannelShared<T>>>` for shared state
- All types implement `Send + Sync` where appropriate

### Performance Considerations

1. **Priority Scheduling**: Lower priority number = higher priority
2. **O(1) Wake**: Waiter maps enable fast wake propagation
3. **Lazy Initialization**: Global scheduler/runtime init on first use
4. **Cloning Futures**: Uses `Arc` so cloning is cheap

### Limitations

1. **No Work Stealing**: Single-threaded scheduler (enhancement planned)
2. **No Async I/O Integration**: Currently wraps sync operations
3. **No Cancellation**: Select doesn't cancel non-winning tasks

---

## Test Coverage

The async runtime has comprehensive test coverage:

```
test_sounio_future_ready          - Future that starts completed
test_sounio_future_complete       - Complete a pending future
test_runtime_spawn                - Spawn tasks on runtime
test_runtime_block_on             - Block until completion
test_async_context                - Async state machine
test_scheduler_*                  - TaskScheduler operations
test_join_*                       - Join combinator
test_select_*                     - Select combinator
test_channel_*                    - Channel operations
```

Run tests:
```bash
cargo test async_runtime
cargo test scheduler
cargo test join_future
cargo test select_future
cargo test channel
```

---

## Related Documentation

- [Effects System](LLM_PROGRAMMING_GUIDE.md#effects) - `with Async` effect
- [GPU Runtime](GPU_RUNTIME.md) - GPU async operations
- [Epistemic Types](api/EPISTEMIC_API.md) - Uncertainty through async

---

*Last updated: January 2026 (v1.0.0)*
