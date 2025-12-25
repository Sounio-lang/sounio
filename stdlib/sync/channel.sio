//! Message passing with channels
//!
//! Channels provide a way to send data between threads.

use std::sync::{Mutex, Arc}
use std::collections::Deque
use std::fmt::{Debug, Formatter, FmtError}

/// Creates a new asynchronous channel, returning the sender/receiver halves.
///
/// All data sent on the sender will become available on the receiver,
/// and no send will block the calling thread. The receiver will block
/// until a message becomes available.
///
/// # Examples
///
/// ```d
/// let (tx, rx) = channel()
///
/// tx.send(42)
///
/// assert_eq(rx.recv().unwrap(), 42)
/// ```
pub fn channel<T>() -> (Sender<T>, Receiver<T>) with Alloc {
    let inner = Arc::new(ChannelInner {
        queue: Mutex::new(Deque::new()),
        closed: Mutex::new(false),
        sender_count: Mutex::new(1),
    })

    (Sender { inner: inner.clone() }, Receiver { inner })
}

/// Creates a new synchronous channel with a bounded capacity.
///
/// The sender will block when the buffer is full.
///
/// # Examples
///
/// ```d
/// let (tx, rx) = sync_channel(1)
///
/// tx.send(42)  // This doesn't block
/// // tx.send(43) would block because buffer is full
///
/// assert_eq(rx.recv().unwrap(), 42)
/// ```
pub fn sync_channel<T>(bound: int) -> (SyncSender<T>, Receiver<T>) with Alloc {
    let inner = Arc::new(SyncChannelInner {
        queue: Mutex::new(Deque::new()),
        closed: Mutex::new(false),
        sender_count: Mutex::new(1),
        bound,
    })

    (SyncSender { inner: inner.clone() }, Receiver { inner: inner.clone() })
}

struct ChannelInner<T> {
    queue: Mutex<Deque<T>>,
    closed: Mutex<bool>,
    sender_count: Mutex<int>,
}

struct SyncChannelInner<T> {
    queue: Mutex<Deque<T>>,
    closed: Mutex<bool>,
    sender_count: Mutex<int>,
    bound: int,
}

/// The sending half of a channel.
///
/// Messages can be sent through this channel with send.
pub struct Sender<T> {
    inner: Arc<ChannelInner<T>>,
}

impl<T> Sender<T> {
    /// Sends a value on this channel.
    ///
    /// This method will never block. If the receiver has been dropped,
    /// this method returns an error containing the unsent value.
    ///
    /// # Examples
    ///
    /// ```d
    /// let (tx, rx) = channel()
    /// tx.send(42).unwrap()
    /// ```
    pub fn send(self: &Sender<T>, value: T) -> Result<unit, SendError<T>> {
        let closed = *self.inner.closed.lock()
        if closed {
            return Result::Err(SendError(value))
        }

        let mut queue = self.inner.queue.lock()
        queue.push_back(value)
        Result::Ok(())
    }
}

impl<T> Clone for Sender<T> {
    fn clone(self: &Sender<T>) -> Sender<T> {
        let mut count = self.inner.sender_count.lock()
        *count = *count + 1
        Sender { inner: self.inner.clone() }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(self: &!Sender<T>) {
        let mut count = self.inner.sender_count.lock()
        *count = *count - 1
        if *count == 0 {
            // Last sender - mark channel as closed
            *self.inner.closed.lock() = true
        }
    }
}

/// The sending half of a synchronous channel.
///
/// Messages can be sent through this channel with send, which will block
/// if the buffer is full.
pub struct SyncSender<T> {
    inner: Arc<SyncChannelInner<T>>,
}

impl<T> SyncSender<T> {
    /// Sends a value on this channel, blocking if the buffer is full.
    ///
    /// # Examples
    ///
    /// ```d
    /// let (tx, rx) = sync_channel(1)
    /// tx.send(42).unwrap()
    /// ```
    pub fn send(self: &SyncSender<T>, value: T) -> Result<unit, SendError<T>> {
        loop {
            let closed = *self.inner.closed.lock()
            if closed {
                return Result::Err(SendError(value))
            }

            let mut queue = self.inner.queue.lock()
            if queue.len() < self.inner.bound {
                queue.push_back(value)
                return Result::Ok(())
            }

            // Drop lock and yield
            drop(queue)
            std::thread::yield_now()
        }
    }

    /// Attempts to send a value on this channel without blocking.
    ///
    /// This method differs from send by returning immediately if the
    /// channel's buffer is full or no receiver is waiting to acquire
    /// some data.
    pub fn try_send(self: &SyncSender<T>, value: T) -> Result<unit, TrySendError<T>> {
        let closed = *self.inner.closed.lock()
        if closed {
            return Result::Err(TrySendError::Disconnected(value))
        }

        let mut queue = self.inner.queue.lock()
        if queue.len() < self.inner.bound {
            queue.push_back(value)
            Result::Ok(())
        } else {
            Result::Err(TrySendError::Full(value))
        }
    }
}

impl<T> Clone for SyncSender<T> {
    fn clone(self: &SyncSender<T>) -> SyncSender<T> {
        let mut count = self.inner.sender_count.lock()
        *count = *count + 1
        SyncSender { inner: self.inner.clone() }
    }
}

impl<T> Drop for SyncSender<T> {
    fn drop(self: &!SyncSender<T>) {
        let mut count = self.inner.sender_count.lock()
        *count = *count - 1
        if *count == 0 {
            *self.inner.closed.lock() = true
        }
    }
}

/// The receiving half of a channel.
pub struct Receiver<T> {
    inner: Arc<ChannelInner<T>>,
}

impl<T> Receiver<T> {
    /// Attempts to wait for a value on this receiver, returning an error
    /// if the corresponding channel has hung up.
    ///
    /// This function will always block the current thread if there is no data
    /// available and it's possible for more data to be sent.
    pub fn recv(self: &Receiver<T>) -> Result<T, RecvError> {
        loop {
            let mut queue = self.inner.queue.lock()

            if let Option::Some(value) = queue.pop_front() {
                return Result::Ok(value)
            }

            let closed = *self.inner.closed.lock()
            if closed && queue.is_empty() {
                return Result::Err(RecvError)
            }

            // Drop lock and yield
            drop(queue)
            std::thread::yield_now()
        }
    }

    /// Attempts to return a pending value on this receiver without blocking.
    ///
    /// This method will never block the caller in order to wait for data to
    /// become available. Instead, this will always return immediately with
    /// a possible option of pending data on the channel.
    pub fn try_recv(self: &Receiver<T>) -> Result<T, TryRecvError> {
        let mut queue = self.inner.queue.lock()

        if let Option::Some(value) = queue.pop_front() {
            return Result::Ok(value)
        }

        let closed = *self.inner.closed.lock()
        if closed {
            Result::Err(TryRecvError::Disconnected)
        } else {
            Result::Err(TryRecvError::Empty)
        }
    }

    /// Returns an iterator that will block waiting for messages.
    ///
    /// The iterator will return None when the channel has been disconnected.
    pub fn iter(self: &Receiver<T>) -> Iter<T> {
        Iter { receiver: self }
    }
}

impl<T> IntoIterator for Receiver<T> {
    type Item = T
    type IntoIter = IntoIter<T>

    fn into_iter(self: Receiver<T>) -> IntoIter<T> {
        IntoIter { receiver: self }
    }
}

/// An iterator over messages received on a channel.
pub struct Iter<T> {
    receiver: &Receiver<T>,
}

impl<T> Iterator for Iter<T> {
    type Item = T

    fn next(self: &!Iter<T>) -> Option<T> {
        self.receiver.recv().ok()
    }
}

/// An owning iterator over messages received on a channel.
pub struct IntoIter<T> {
    receiver: Receiver<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T

    fn next(self: &!IntoIter<T>) -> Option<T> {
        self.receiver.recv().ok()
    }
}

/// An error returned from the send function on channels.
///
/// A send operation can only fail if the receiving end of a channel is
/// disconnected, implying that the data could never be received.
pub struct SendError<T>(pub T);

impl<T> Debug for SendError<T> {
    fn fmt(self: &SendError<T>, f: &!Formatter) -> Result<unit, FmtError> {
        f.write_str("sending on a closed channel")
    }
}

impl<T> SendError<T> {
    /// Returns the value that was attempted to be sent.
    pub fn into_inner(self: SendError<T>) -> T {
        self.0
    }
}

/// An error returned from try_send.
pub enum TrySendError<T> {
    /// The channel's buffer is full.
    Full(T),
    /// The receiver has been dropped.
    Disconnected(T),
}

impl<T> TrySendError<T> {
    /// Returns the value that was attempted to be sent.
    pub fn into_inner(self: TrySendError<T>) -> T {
        match self {
            TrySendError::Full(v) => v,
            TrySendError::Disconnected(v) => v,
        }
    }

    /// Returns true if this error is due to a full buffer.
    pub fn is_full(self: &TrySendError<T>) -> bool {
        match self {
            TrySendError::Full(_) => true,
            TrySendError::Disconnected(_) => false,
        }
    }

    /// Returns true if this error is due to a disconnected receiver.
    pub fn is_disconnected(self: &TrySendError<T>) -> bool {
        match self {
            TrySendError::Full(_) => false,
            TrySendError::Disconnected(_) => true,
        }
    }
}

impl<T> Debug for TrySendError<T> {
    fn fmt(self: &TrySendError<T>, f: &!Formatter) -> Result<unit, FmtError> {
        match self {
            TrySendError::Full(_) => f.write_str("channel is full"),
            TrySendError::Disconnected(_) => f.write_str("receiving on a closed channel"),
        }
    }
}

/// An error returned from the recv function on a receiver.
///
/// The recv operation can only fail if the sending half of a channel is
/// disconnected, implying that no further messages will ever be received.
pub struct RecvError;

impl Debug for RecvError {
    fn fmt(self: &RecvError, f: &!Formatter) -> Result<unit, FmtError> {
        f.write_str("receiving on a closed channel")
    }
}

/// An error returned from try_recv.
pub enum TryRecvError {
    /// The channel is currently empty.
    Empty,
    /// The channel has been disconnected.
    Disconnected,
}

impl TryRecvError {
    /// Returns true if this error is due to an empty buffer.
    pub fn is_empty(self: &TryRecvError) -> bool {
        match self {
            TryRecvError::Empty => true,
            TryRecvError::Disconnected => false,
        }
    }

    /// Returns true if this error is due to a disconnected sender.
    pub fn is_disconnected(self: &TryRecvError) -> bool {
        match self {
            TryRecvError::Empty => false,
            TryRecvError::Disconnected => true,
        }
    }
}

impl Debug for TryRecvError {
    fn fmt(self: &TryRecvError, f: &!Formatter) -> Result<unit, FmtError> {
        match self {
            TryRecvError::Empty => f.write_str("channel is empty"),
            TryRecvError::Disconnected => f.write_str("channel is disconnected"),
        }
    }
}

// Unit tests
#[test]
fn test_channel_basic() {
    let (tx, rx) = channel()
    tx.send(42).unwrap()
    assert_eq(rx.recv().unwrap(), 42)
}

#[test]
fn test_channel_multiple_sends() {
    let (tx, rx) = channel()
    tx.send(1).unwrap()
    tx.send(2).unwrap()
    tx.send(3).unwrap()

    assert_eq(rx.recv().unwrap(), 1)
    assert_eq(rx.recv().unwrap(), 2)
    assert_eq(rx.recv().unwrap(), 3)
}

#[test]
fn test_channel_try_recv() {
    let (tx, rx) = channel()

    // Empty channel
    assert(rx.try_recv().is_err())

    tx.send(42).unwrap()

    // Now should succeed
    assert_eq(rx.try_recv().unwrap(), 42)
}

#[test]
fn test_sync_channel_try_send() {
    let (tx, rx) = sync_channel(1)

    // Should succeed (buffer has room)
    assert(tx.try_send(1).is_ok())

    // Should fail (buffer full)
    assert(tx.try_send(2).is_err())

    // Receive to make room
    rx.recv().unwrap()

    // Should succeed again
    assert(tx.try_send(3).is_ok())
}
