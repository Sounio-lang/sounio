/// Async Channels
///
/// Provides various channel types for async message passing:
/// - MPSC (multi-producer, single-consumer)
/// - Oneshot (single value, single-use)
/// - Broadcast (multi-producer, multi-consumer)
/// - Watch (single value with change notification)

module async::channel

import async::future::{Future, Poll, Context, Waker}

/// Error returned when a channel is closed
pub struct SendError<T> {
    /// The value that couldn't be sent
    pub value: T,
}

impl<T> SendError<T> {
    /// Creates a new send error
    pub fn new(value: T) -> SendError<T> {
        SendError { value }
    }

    /// Returns the value that couldn't be sent
    pub fn into_inner(self) -> T {
        self.value
    }
}

/// Error returned when receiving from a closed channel
pub struct RecvError {
    _priv: (),
}

impl RecvError {
    fn new() -> RecvError {
        RecvError { _priv: () }
    }
}

/// Error returned when trying to receive without blocking
pub enum TryRecvError {
    /// The channel is empty
    Empty,
    /// The channel is closed
    Disconnected,
}

/// Error returned when trying to send without blocking
pub enum TrySendError<T> {
    /// The channel is full
    Full(T),
    /// The channel is closed
    Disconnected(T),
}

// =============================================================================
// MPSC Channel
// =============================================================================

/// Creates an unbounded MPSC channel
///
/// # Example
/// ```
/// async fn example() {
///     let (tx, mut rx) = mpsc::unbounded()
///
///     tx.send(1).unwrap()
///     tx.send(2).unwrap()
///
///     assert_eq!(rx.recv().await, Some(1))
///     assert_eq!(rx.recv().await, Some(2))
/// }
/// ```
pub mod mpsc {
    use super::*

    /// Creates an unbounded channel
    pub fn unbounded<T>() -> (UnboundedSender<T>, UnboundedReceiver<T>) {
        let shared = SharedState::new();
        (
            UnboundedSender { shared: &shared },
            UnboundedReceiver { shared: &shared }
        )
    }

    /// Creates a bounded channel with the given capacity
    pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
        let shared = BoundedSharedState::new(capacity);
        (
            Sender { shared: &shared },
            Receiver { shared: &shared }
        )
    }

    /// Shared state for unbounded channel
    struct SharedState<T> {
        queue: Vec<T>,
        closed: bool,
        recv_waker: Option<Waker>,
    }

    impl<T> SharedState<T> {
        fn new() -> SharedState<T> {
            SharedState {
                queue: Vec::new(),
                closed: false,
                recv_waker: None,
            }
        }
    }

    /// Sender half of an unbounded channel
    pub struct UnboundedSender<T> {
        shared: &SharedState<T>,
    }

    impl<T> UnboundedSender<T> {
        /// Sends a value, returning error if channel is closed
        pub fn send(&self, value: T) -> Result<(), SendError<T>> {
            if self.shared.closed {
                return Err(SendError::new(value));
            }

            self.shared.queue.push(value);

            // Wake receiver if waiting
            if let Some(waker) = self.shared.recv_waker.take() {
                waker.wake();
            }

            Ok(())
        }

        /// Returns true if the channel is closed
        pub fn is_closed(&self) -> bool {
            self.shared.closed
        }

        /// Closes the channel
        pub fn close(&self) {
            self.shared.closed = true;
            if let Some(waker) = self.shared.recv_waker.take() {
                waker.wake();
            }
        }

        /// Returns the number of messages in the queue
        pub fn len(&self) -> usize {
            self.shared.queue.len()
        }

        /// Returns true if the queue is empty
        pub fn is_empty(&self) -> bool {
            self.shared.queue.is_empty()
        }
    }

    impl<T> Clone for UnboundedSender<T> {
        fn clone(&self) -> UnboundedSender<T> {
            UnboundedSender { shared: self.shared }
        }
    }

    impl<T> Drop for UnboundedSender<T> {
        fn drop(&mut self) {
            // In real impl, would track sender count
        }
    }

    /// Receiver half of an unbounded channel
    pub struct UnboundedReceiver<T> {
        shared: &SharedState<T>,
    }

    impl<T> UnboundedReceiver<T> {
        /// Receives a value, waiting if necessary
        pub async fn recv(&mut self) -> Option<T> with Async {
            RecvFuture { receiver: self }.await
        }

        /// Tries to receive without waiting
        pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
            if !self.shared.queue.is_empty() {
                Ok(self.shared.queue.remove(0))
            } else if self.shared.closed {
                Err(TryRecvError::Disconnected)
            } else {
                Err(TryRecvError::Empty)
            }
        }

        /// Closes the channel
        pub fn close(&mut self) {
            self.shared.closed = true;
        }

        /// Returns the number of messages in the queue
        pub fn len(&self) -> usize {
            self.shared.queue.len()
        }

        /// Returns true if the queue is empty
        pub fn is_empty(&self) -> bool {
            self.shared.queue.is_empty()
        }
    }

    struct RecvFuture<'a, T> {
        receiver: &'a mut UnboundedReceiver<T>,
    }

    impl<'a, T> Future for RecvFuture<'a, T> {
        type Output = Option<T>

        fn poll(&mut self, cx: &mut Context) -> Poll<Option<T>> {
            if !self.receiver.shared.queue.is_empty() {
                Poll::Ready(Some(self.receiver.shared.queue.remove(0)))
            } else if self.receiver.shared.closed {
                Poll::Ready(None)
            } else {
                self.receiver.shared.recv_waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }

    /// Shared state for bounded channel
    struct BoundedSharedState<T> {
        queue: Vec<T>,
        capacity: usize,
        closed: bool,
        recv_waker: Option<Waker>,
        send_wakers: Vec<Waker>,
    }

    impl<T> BoundedSharedState<T> {
        fn new(capacity: usize) -> BoundedSharedState<T> {
            BoundedSharedState {
                queue: Vec::with_capacity(capacity),
                capacity,
                closed: false,
                recv_waker: None,
                send_wakers: Vec::new(),
            }
        }
    }

    /// Sender half of a bounded channel
    pub struct Sender<T> {
        shared: &BoundedSharedState<T>,
    }

    impl<T> Sender<T> {
        /// Sends a value, waiting if the channel is full
        pub async fn send(&self, value: T) -> Result<(), SendError<T>> with Async {
            SendFuture { sender: self, value: Some(value) }.await
        }

        /// Tries to send without waiting
        pub fn try_send(&self, value: T) -> Result<(), TrySendError<T>> {
            if self.shared.closed {
                return Err(TrySendError::Disconnected(value));
            }

            if self.shared.queue.len() >= self.shared.capacity {
                return Err(TrySendError::Full(value));
            }

            self.shared.queue.push(value);

            if let Some(waker) = self.shared.recv_waker.take() {
                waker.wake();
            }

            Ok(())
        }

        /// Returns true if the channel is closed
        pub fn is_closed(&self) -> bool {
            self.shared.closed
        }

        /// Returns the channel capacity
        pub fn capacity(&self) -> usize {
            self.shared.capacity
        }
    }

    impl<T> Clone for Sender<T> {
        fn clone(&self) -> Sender<T> {
            Sender { shared: self.shared }
        }
    }

    struct SendFuture<'a, T> {
        sender: &'a Sender<T>,
        value: Option<T>,
    }

    impl<'a, T> Future for SendFuture<'a, T> {
        type Output = Result<(), SendError<T>>

        fn poll(&mut self, cx: &mut Context) -> Poll<Result<(), SendError<T>>> {
            if self.sender.shared.closed {
                return Poll::Ready(Err(SendError::new(self.value.take().unwrap())));
            }

            if self.sender.shared.queue.len() < self.sender.shared.capacity {
                self.sender.shared.queue.push(self.value.take().unwrap());

                if let Some(waker) = self.sender.shared.recv_waker.take() {
                    waker.wake();
                }

                Poll::Ready(Ok(()))
            } else {
                self.sender.shared.send_wakers.push(cx.waker().clone());
                Poll::Pending
            }
        }
    }

    /// Receiver half of a bounded channel
    pub struct Receiver<T> {
        shared: &BoundedSharedState<T>,
    }

    impl<T> Receiver<T> {
        /// Receives a value, waiting if necessary
        pub async fn recv(&mut self) -> Option<T> with Async {
            BoundedRecvFuture { receiver: self }.await
        }

        /// Tries to receive without waiting
        pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
            if !self.shared.queue.is_empty() {
                let value = self.shared.queue.remove(0);

                // Wake a waiting sender
                if let Some(waker) = self.shared.send_wakers.pop() {
                    waker.wake();
                }

                Ok(value)
            } else if self.shared.closed {
                Err(TryRecvError::Disconnected)
            } else {
                Err(TryRecvError::Empty)
            }
        }

        /// Closes the channel
        pub fn close(&mut self) {
            self.shared.closed = true;
            // Wake all waiting senders
            for waker in self.shared.send_wakers.drain(..) {
                waker.wake();
            }
        }
    }

    struct BoundedRecvFuture<'a, T> {
        receiver: &'a mut Receiver<T>,
    }

    impl<'a, T> Future for BoundedRecvFuture<'a, T> {
        type Output = Option<T>

        fn poll(&mut self, cx: &mut Context) -> Poll<Option<T>> {
            if !self.receiver.shared.queue.is_empty() {
                let value = self.receiver.shared.queue.remove(0);

                // Wake a waiting sender
                if let Some(waker) = self.receiver.shared.send_wakers.pop() {
                    waker.wake();
                }

                Poll::Ready(Some(value))
            } else if self.receiver.shared.closed {
                Poll::Ready(None)
            } else {
                self.receiver.shared.recv_waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

// =============================================================================
// Oneshot Channel
// =============================================================================

/// A oneshot channel for sending a single value
///
/// # Example
/// ```
/// async fn example() {
///     let (tx, rx) = oneshot::channel()
///
///     spawn async {
///         tx.send(42)
///     }
///
///     let value = rx.await.unwrap()
///     assert_eq!(value, 42)
/// }
/// ```
pub mod oneshot {
    use super::*

    /// Creates a oneshot channel
    pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
        let shared = SharedState::new();
        (
            Sender { shared: &shared },
            Receiver { shared: &shared }
        )
    }

    struct SharedState<T> {
        value: Option<T>,
        closed: bool,
        waker: Option<Waker>,
    }

    impl<T> SharedState<T> {
        fn new() -> SharedState<T> {
            SharedState {
                value: None,
                closed: false,
                waker: None,
            }
        }
    }

    /// Sender half of a oneshot channel
    pub struct Sender<T> {
        shared: &SharedState<T>,
    }

    impl<T> Sender<T> {
        /// Sends a value, consuming the sender
        pub fn send(self, value: T) -> Result<(), T> {
            if self.shared.closed {
                return Err(value);
            }

            self.shared.value = Some(value);

            if let Some(waker) = self.shared.waker.take() {
                waker.wake();
            }

            Ok(())
        }

        /// Returns true if the receiver has been dropped
        pub fn is_closed(&self) -> bool {
            self.shared.closed
        }
    }

    impl<T> Drop for Sender<T> {
        fn drop(&mut self) {
            self.shared.closed = true;
            if let Some(waker) = self.shared.waker.take() {
                waker.wake();
            }
        }
    }

    /// Receiver half of a oneshot channel
    pub struct Receiver<T> {
        shared: &SharedState<T>,
    }

    impl<T> Receiver<T> {
        /// Tries to receive without waiting
        pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
            if let Some(value) = self.shared.value.take() {
                Ok(value)
            } else if self.shared.closed {
                Err(TryRecvError::Disconnected)
            } else {
                Err(TryRecvError::Empty)
            }
        }

        /// Closes the receiver, signaling to the sender
        pub fn close(&mut self) {
            self.shared.closed = true;
        }
    }

    impl<T> Future for Receiver<T> {
        type Output = Result<T, RecvError>

        fn poll(&mut self, cx: &mut Context) -> Poll<Result<T, RecvError>> {
            if let Some(value) = self.shared.value.take() {
                Poll::Ready(Ok(value))
            } else if self.shared.closed {
                Poll::Ready(Err(RecvError::new()))
            } else {
                self.shared.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }

    impl<T> Drop for Receiver<T> {
        fn drop(&mut self) {
            self.shared.closed = true;
        }
    }
}

// =============================================================================
// Broadcast Channel
// =============================================================================

/// A broadcast channel for multiple receivers
///
/// All receivers get all messages sent after they subscribe.
///
/// # Example
/// ```
/// async fn example() {
///     let (tx, mut rx1) = broadcast::channel(16)
///     let mut rx2 = tx.subscribe()
///
///     tx.send(1).unwrap()
///
///     assert_eq!(rx1.recv().await.unwrap(), 1)
///     assert_eq!(rx2.recv().await.unwrap(), 1)
/// }
/// ```
pub mod broadcast {
    use super::*

    /// Creates a broadcast channel with the given capacity
    pub fn channel<T: Clone>(capacity: usize) -> (Sender<T>, Receiver<T>) {
        let shared = SharedState::new(capacity);
        (
            Sender { shared: &shared },
            Receiver { shared: &shared, pos: 0 }
        )
    }

    struct SharedState<T> {
        buffer: Vec<T>,
        capacity: usize,
        head: usize,  // Next write position
        closed: bool,
        wakers: Vec<Waker>,
    }

    impl<T> SharedState<T> {
        fn new(capacity: usize) -> SharedState<T> {
            SharedState {
                buffer: Vec::with_capacity(capacity),
                capacity,
                head: 0,
                closed: false,
                wakers: Vec::new(),
            }
        }
    }

    /// Sender half of a broadcast channel
    pub struct Sender<T: Clone> {
        shared: &SharedState<T>,
    }

    impl<T: Clone> Sender<T> {
        /// Sends a value to all receivers
        pub fn send(&self, value: T) -> Result<usize, SendError<T>> {
            if self.shared.closed {
                return Err(SendError::new(value));
            }

            // Add to buffer (circular)
            if self.shared.buffer.len() < self.shared.capacity {
                self.shared.buffer.push(value);
            } else {
                let pos = self.shared.head % self.shared.capacity;
                self.shared.buffer[pos] = value;
            }
            self.shared.head += 1;

            // Wake all receivers
            let receiver_count = self.shared.wakers.len();
            for waker in self.shared.wakers.drain(..) {
                waker.wake();
            }

            Ok(receiver_count)
        }

        /// Creates a new receiver subscribed to this sender
        pub fn subscribe(&self) -> Receiver<T> {
            Receiver {
                shared: self.shared,
                pos: self.shared.head,
            }
        }

        /// Returns the number of active receivers
        pub fn receiver_count(&self) -> usize {
            // In real impl, would track this
            0
        }
    }

    impl<T: Clone> Clone for Sender<T> {
        fn clone(&self) -> Sender<T> {
            Sender { shared: self.shared }
        }
    }

    /// Error when a receiver has lagged behind
    pub struct LagError {
        /// Number of skipped messages
        pub skipped: u64,
    }

    /// Receiver half of a broadcast channel
    pub struct Receiver<T: Clone> {
        shared: &SharedState<T>,
        pos: usize,  // Next read position
    }

    impl<T: Clone> Receiver<T> {
        /// Receives the next value
        pub async fn recv(&mut self) -> Result<T, RecvError> with Async {
            BroadcastRecvFuture { receiver: self }.await
        }

        /// Tries to receive without waiting
        pub fn try_recv(&mut self) -> Result<T, TryRecvError> {
            if self.pos < self.shared.head {
                let buffer_pos = self.pos % self.shared.capacity;
                if buffer_pos < self.shared.buffer.len() {
                    let value = self.shared.buffer[buffer_pos].clone();
                    self.pos += 1;
                    Ok(value)
                } else {
                    Err(TryRecvError::Empty)
                }
            } else if self.shared.closed {
                Err(TryRecvError::Disconnected)
            } else {
                Err(TryRecvError::Empty)
            }
        }
    }

    impl<T: Clone> Clone for Receiver<T> {
        fn clone(&self) -> Receiver<T> {
            Receiver {
                shared: self.shared,
                pos: self.pos,
            }
        }
    }

    struct BroadcastRecvFuture<'a, T: Clone> {
        receiver: &'a mut Receiver<T>,
    }

    impl<'a, T: Clone> Future for BroadcastRecvFuture<'a, T> {
        type Output = Result<T, RecvError>

        fn poll(&mut self, cx: &mut Context) -> Poll<Result<T, RecvError>> {
            if self.receiver.pos < self.receiver.shared.head {
                let buffer_pos = self.receiver.pos % self.receiver.shared.capacity;
                if buffer_pos < self.receiver.shared.buffer.len() {
                    let value = self.receiver.shared.buffer[buffer_pos].clone();
                    self.receiver.pos += 1;
                    return Poll::Ready(Ok(value));
                }
            }

            if self.receiver.shared.closed {
                return Poll::Ready(Err(RecvError::new()));
            }

            self.receiver.shared.wakers.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

// =============================================================================
// Watch Channel
// =============================================================================

/// A watch channel for observing value changes
///
/// The sender can update a value, and receivers are notified when it changes.
///
/// # Example
/// ```
/// async fn example() {
///     let (tx, mut rx) = watch::channel(0)
///
///     spawn async move {
///         for i in 1..=5 {
///             tx.send(i)
///             sleep(Duration::from_millis(100)).await
///         }
///     }
///
///     while rx.changed().await.is_ok() {
///         println("Value: {}", *rx.borrow())
///     }
/// }
/// ```
pub mod watch {
    use super::*

    /// Creates a watch channel with an initial value
    pub fn channel<T>(initial: T) -> (Sender<T>, Receiver<T>) {
        let shared = SharedState::new(initial);
        (
            Sender { shared: &shared },
            Receiver { shared: &shared, version: 0 }
        )
    }

    struct SharedState<T> {
        value: T,
        version: u64,
        closed: bool,
        wakers: Vec<Waker>,
    }

    impl<T> SharedState<T> {
        fn new(value: T) -> SharedState<T> {
            SharedState {
                value,
                version: 1,
                closed: false,
                wakers: Vec::new(),
            }
        }
    }

    /// Sender half of a watch channel
    pub struct Sender<T> {
        shared: &SharedState<T>,
    }

    impl<T> Sender<T> {
        /// Sends a new value
        pub fn send(&self, value: T) -> Result<(), SendError<T>> {
            if self.shared.closed {
                return Err(SendError::new(value));
            }

            self.shared.value = value;
            self.shared.version += 1;

            // Wake all receivers
            for waker in self.shared.wakers.drain(..) {
                waker.wake();
            }

            Ok(())
        }

        /// Sends a new value only if it's different (requires Eq)
        pub fn send_if_modified<F>(&self, modify: F) -> bool
        where
            F: FnOnce(&mut T) -> bool
        {
            if modify(&mut self.shared.value) {
                self.shared.version += 1;
                for waker in self.shared.wakers.drain(..) {
                    waker.wake();
                }
                true
            } else {
                false
            }
        }

        /// Returns a reference to the current value
        pub fn borrow(&self) -> &T {
            &self.shared.value
        }

        /// Returns true if there are any receivers
        pub fn is_closed(&self) -> bool {
            self.shared.closed
        }

        /// Creates a new receiver
        pub fn subscribe(&self) -> Receiver<T> {
            Receiver {
                shared: self.shared,
                version: self.shared.version,
            }
        }
    }

    impl<T> Drop for Sender<T> {
        fn drop(&mut self) {
            self.shared.closed = true;
            for waker in self.shared.wakers.drain(..) {
                waker.wake();
            }
        }
    }

    /// Receiver half of a watch channel
    pub struct Receiver<T> {
        shared: &SharedState<T>,
        version: u64,
    }

    impl<T> Receiver<T> {
        /// Waits for the value to change
        pub async fn changed(&mut self) -> Result<(), RecvError> with Async {
            ChangedFuture { receiver: self }.await
        }

        /// Returns a reference to the current value
        pub fn borrow(&self) -> &T {
            &self.shared.value
        }

        /// Returns a reference and marks as seen
        pub fn borrow_and_update(&mut self) -> &T {
            self.version = self.shared.version;
            &self.shared.value
        }

        /// Returns true if the value has changed since last seen
        pub fn has_changed(&self) -> bool {
            self.version != self.shared.version
        }
    }

    impl<T> Clone for Receiver<T> {
        fn clone(&self) -> Receiver<T> {
            Receiver {
                shared: self.shared,
                version: self.version,
            }
        }
    }

    struct ChangedFuture<'a, T> {
        receiver: &'a mut Receiver<T>,
    }

    impl<'a, T> Future for ChangedFuture<'a, T> {
        type Output = Result<(), RecvError>

        fn poll(&mut self, cx: &mut Context) -> Poll<Result<(), RecvError>> {
            if self.receiver.shared.closed {
                return Poll::Ready(Err(RecvError::new()));
            }

            if self.receiver.version != self.receiver.shared.version {
                self.receiver.version = self.receiver.shared.version;
                return Poll::Ready(Ok(()));
            }

            self.receiver.shared.wakers.push(cx.waker().clone());
            Poll::Pending
        }
    }
}
