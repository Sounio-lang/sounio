# Async/Await

This chapter covers asynchronous programming in Sounio, including async functions, the await expression, concurrent execution patterns, and integration with the effects system.

## Overview

Sounio provides first-class support for asynchronous programming through:

- **Async functions**: Functions that can suspend and resume execution
- **Await expressions**: Syntax for waiting on async operations
- **Futures**: Values representing computations that complete in the future
- **Concurrency primitives**: spawn, join, select, and channels
- **Effect tracking**: The `Async` effect tracks async operations in the type system

## Async Functions

### Declaration

Declare async functions with the `async` keyword:

```sio
async fn fetch_data(url: string) -> Data with Async, IO {
    let response = http_get(url).await
    let parsed = parse_response(response)
    return parsed
}
```

Key points:
- Async functions must declare the `Async` effect
- They return a `Future<T>` implicitly
- Use `.await` to wait for async operations inside

### The Async Effect

The `Async` effect tracks that a function performs asynchronous operations:

```sio
// Async-only function
async fn compute() -> i32 with Async {
    sleep(Duration::seconds(1)).await
    return 42
}

// Async with I/O
async fn read_remote(url: string) -> string with Async, IO {
    http_get(url).await
}

// Pure async computation (no side effects beyond async)
async fn pure_delay(x: i32) -> i32 with Async {
    yield_now().await
    return x * 2
}
```

## Await Expressions

### Basic Await

Use `.await` to suspend execution until a future completes:

```sio
async fn main() with Async, IO {
    // Wait for the fetch to complete
    let data = fetch_data("https://api.example.com/data").await

    // Process the result
    process(data)
}
```

### Chained Awaits

Chain multiple async operations:

```sio
async fn pipeline() with Async, IO {
    let user = get_user(123).await
    let posts = get_posts(user.id).await
    let comments = get_comments(posts[0].id).await
    return comments
}
```

### Await in Expressions

Await can be used anywhere an expression is expected:

```sio
async fn example() with Async {
    // In variable binding
    let x = compute().await

    // In function arguments
    process(fetch().await, transform().await)

    // In conditionals
    if check().await {
        handle_success()
    }

    // In match expressions
    match get_result().await {
        Ok(v) => use_value(v),
        Err(e) => handle_error(e),
    }
}
```

## Async Blocks

Create anonymous async computations with async blocks:

```sio
// Create a future without immediately awaiting
let future = async {
    expensive_computation()
}

// The computation runs when awaited
let result = future.await
```

Async blocks are useful for:
- Deferred execution
- Passing futures to combinators
- Creating futures conditionally

```sio
async fn conditional_work(flag: bool) with Async {
    let work = if flag {
        async { heavy_computation() }
    } else {
        async { light_computation() }
    }

    let result = work.await
    return result
}
```

## Concurrent Execution

### Spawn

Spawn tasks to run concurrently in the background:

```sio
async fn parallel_work() with Async {
    // Spawn a background task
    let handle = spawn {
        long_running_task()
    }

    // Continue with other work
    do_other_work()

    // Wait for the spawned task
    let result = handle.await
    return result
}
```

### Join

Wait for multiple futures to complete concurrently:

```sio
async fn fetch_all() with Async, IO {
    // Run three fetches concurrently
    let (users, posts, comments) = join(
        fetch_users(),
        fetch_posts(),
        fetch_comments()
    ).await

    // All three are complete
    return combine(users, posts, comments)
}
```

Join variants:
- `join(a, b)` - Wait for two futures
- `join3(a, b, c)` - Wait for three futures
- `join4(a, b, c, d)` - Wait for four futures
- `join_all(vec)` - Wait for a collection of futures

### Select

Wait for the first future to complete:

```sio
async fn with_timeout() with Async {
    select {
        result = compute() => {
            println("Computation finished: {}", result)
        }
        _ = timeout(Duration::seconds(5)) => {
            println("Timed out!")
        }
    }
}
```

Select with multiple branches:

```sio
async fn event_loop() with Async, IO {
    loop {
        select {
            msg = receiver.recv() => {
                handle_message(msg)
            }
            _ = ticker.tick() => {
                do_periodic_work()
            }
            _ = shutdown.recv() => {
                cleanup()
                break
            }
        }
    }
}
```

### Race

Get the result of whichever future completes first:

```sio
async fn fastest_mirror() with Async, IO {
    // Return whichever responds first
    let data = race(
        fetch_from_mirror1(),
        fetch_from_mirror2(),
        fetch_from_mirror3()
    ).await

    return data
}
```

## Channels

Channels provide async communication between tasks:

### MPSC (Multi-Producer, Single-Consumer)

```sio
async fn producer_consumer() with Async {
    // Create a channel
    let (sender, receiver) = mpsc::channel::<Message>(100)

    // Spawn producer
    spawn {
        for item in items {
            sender.send(item).await
        }
    }

    // Consume messages
    while let Some(msg) = receiver.recv().await {
        process(msg)
    }
}
```

### Oneshot

Single-value channels for request-response patterns:

```sio
async fn request_response() with Async {
    let (tx, rx) = oneshot::channel::<Response>()

    // Send request
    send_request(Request { reply_to: tx })

    // Wait for response
    let response = rx.await
    return response
}
```

### Broadcast

Send to multiple receivers:

```sio
async fn broadcast_updates() with Async {
    let (tx, _) = broadcast::channel::<Update>(16)

    // Multiple subscribers
    let rx1 = tx.subscribe()
    let rx2 = tx.subscribe()

    // Send to all
    tx.send(Update { data: "new data" })
}
```

## Async Closures

Create async closures for higher-order async programming:

```sio
async fn process_items(items: Vec<i32>) with Async {
    // Async closure
    let process = async |id: i32| -> Result<Data, Error> {
        let data = fetch_data(id).await
        transform(data)
    }

    // Process all items concurrently
    let results = join_all(
        items.iter().map(|id| process(*id))
    ).await

    return results
}
```

## Timeouts and Cancellation

### Timeout

Limit the time for an async operation:

```sio
async fn with_timeout() with Async, IO {
    let result = timeout(Duration::seconds(30), async {
        slow_operation().await
    }).await

    match result {
        Ok(value) => println("Got result: {}", value),
        Err(TimeoutError) => println("Operation timed out"),
    }
}
```

### Cancellation Tokens

Cooperative cancellation for long-running tasks:

```sio
async fn cancellable_work() with Async {
    let token = CancellationToken::new()

    // Spawn work that checks for cancellation
    let handle = spawn {
        loop {
            if token.is_cancelled() {
                println("Cancelled!")
                break
            }
            do_work_chunk().await
        }
    }

    // Cancel after some condition
    if should_cancel() {
        token.cancel()
    }

    handle.await
}
```

## Runtime

### block_on

Run async code from synchronous contexts:

```sio
fn main() -> i32 {
    // Entry point - block on the async main
    let result = block_on(async {
        initialize().await
        run_application().await
    })

    return result
}
```

### Runtime Configuration

Configure the async runtime for your needs:

```sio
fn main() -> i32 {
    // Single-threaded runtime
    let runtime = Runtime::new()

    // Multi-threaded runtime
    let runtime = Runtime::new_multi_thread()

    // Custom configuration
    let runtime = RuntimeBuilder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("Failed to build runtime")

    runtime.block_on(async_main())
}
```

## The Future Trait

Understand the underlying abstraction:

```sio
/// Core trait for async computations
pub trait Future {
    type Output

    /// Poll the future to see if it's ready
    fn poll(&mut self, cx: &mut Context) -> Poll<Self::Output>
}

/// Poll result
pub enum Poll<T> {
    Ready(T),    // Computation complete
    Pending,     // Not ready, will wake when progress can be made
}
```

### Implementing Custom Futures

```sio
struct Delay {
    deadline: Instant,
    registered: bool,
}

impl Future for Delay {
    type Output = ()

    fn poll(&mut self, cx: &mut Context) -> Poll<()> {
        if Instant::now() >= self.deadline {
            return Poll::Ready(())
        }

        if !self.registered {
            // Register to be woken at deadline
            register_timer(self.deadline, cx.waker().clone())
            self.registered = true
        }

        Poll::Pending
    }
}
```

## Error Handling

### Try-Await Pattern

Combine error handling with async:

```sio
async fn fallible_async() -> Result<Data, Error> with Async, IO {
    // Each await can fail
    let user = get_user(id).await?
    let posts = get_posts(user.id).await?
    let enriched = enrich(posts).await?

    return Ok(enriched)
}
```

### Error Recovery

Handle errors in concurrent operations:

```sio
async fn with_retry() with Async, IO {
    var attempts = 0

    loop {
        match fetch_data().await {
            Ok(data) => return data,
            Err(e) => {
                attempts = attempts + 1
                if attempts >= 3 {
                    panic("Failed after 3 attempts")
                }
                sleep(Duration::seconds(1)).await
            }
        }
    }
}
```

## Async I/O

The `async::io` module provides async file and network operations:

```sio
import async::io::{File, AsyncRead, AsyncWrite}

async fn copy_file(src: string, dst: string) with Async, IO {
    let source = File::open(src).await?
    let dest = File::create(dst).await?

    let mut buffer = [0u8; 8192]
    loop {
        let n = source.read(&mut buffer).await?
        if n == 0 { break }
        dest.write_all(&buffer[..n]).await?
    }

    Ok(())
}
```

## Async Streams

Process sequences of async values:

```sio
import async::stream::{Stream, StreamExt}

async fn process_stream() with Async {
    let stream = create_stream()

    // Iterate over async values
    while let Some(item) = stream.next().await {
        process(item)
    }

    // Or use combinators
    let results = stream
        .filter(|x| x > 0)
        .map(|x| x * 2)
        .collect::<Vec<_>>()
        .await
}
```

## Best Practices

1. **Always declare the Async effect**: Makes async boundaries explicit

2. **Avoid blocking in async code**: Use `spawn_blocking` for CPU-intensive work

3. **Use structured concurrency**: Prefer `join` and `select` over manual task management

4. **Handle cancellation gracefully**: Check cancellation tokens in loops

5. **Bound channel capacity**: Unbounded channels can cause memory issues

6. **Use timeouts**: Prevent hanging on network or I/O operations

7. **Propagate errors**: Use `?` to propagate errors through async call chains

```sio
// Good: structured concurrency with proper error handling
async fn fetch_all_data() -> Result<AllData, Error> with Async, IO {
    let (users, posts) = timeout(
        Duration::seconds(30),
        join(fetch_users(), fetch_posts())
    ).await??

    Ok(AllData { users, posts })
}
```
