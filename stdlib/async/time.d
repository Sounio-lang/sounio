/// Async Timers and Time Utilities
///
/// Provides asynchronous sleep, timeouts, and interval timers.

module async::time

import async::future::{Future, Poll, Context, Waker}

/// A duration of time
pub struct Duration {
    /// Total nanoseconds
    nanos: u64,
}

impl Duration {
    /// Creates a duration from seconds
    pub fn from_secs(secs: u64) -> Duration {
        Duration { nanos: secs * 1_000_000_000 }
    }

    /// Creates a duration from milliseconds
    pub fn from_millis(millis: u64) -> Duration {
        Duration { nanos: millis * 1_000_000 }
    }

    /// Creates a duration from microseconds
    pub fn from_micros(micros: u64) -> Duration {
        Duration { nanos: micros * 1_000 }
    }

    /// Creates a duration from nanoseconds
    pub fn from_nanos(nanos: u64) -> Duration {
        Duration { nanos }
    }

    /// Creates a zero duration
    pub fn zero() -> Duration {
        Duration { nanos: 0 }
    }

    /// Returns the total seconds (truncated)
    pub fn as_secs(&self) -> u64 {
        self.nanos / 1_000_000_000
    }

    /// Returns the total milliseconds (truncated)
    pub fn as_millis(&self) -> u64 {
        self.nanos / 1_000_000
    }

    /// Returns the total microseconds (truncated)
    pub fn as_micros(&self) -> u64 {
        self.nanos / 1_000
    }

    /// Returns the total nanoseconds
    pub fn as_nanos(&self) -> u64 {
        self.nanos
    }

    /// Returns the subsecond nanoseconds
    pub fn subsec_nanos(&self) -> u32 {
        (self.nanos % 1_000_000_000) as u32
    }

    /// Returns true if this duration is zero
    pub fn is_zero(&self) -> bool {
        self.nanos == 0
    }

    /// Checked addition
    pub fn checked_add(&self, other: Duration) -> Option<Duration> {
        self.nanos.checked_add(other.nanos).map(|n| Duration { nanos: n })
    }

    /// Checked subtraction
    pub fn checked_sub(&self, other: Duration) -> Option<Duration> {
        if self.nanos >= other.nanos {
            Some(Duration { nanos: self.nanos - other.nanos })
        } else {
            None
        }
    }

    /// Saturating addition
    pub fn saturating_add(&self, other: Duration) -> Duration {
        Duration { nanos: self.nanos.saturating_add(other.nanos) }
    }

    /// Saturating subtraction
    pub fn saturating_sub(&self, other: Duration) -> Duration {
        Duration { nanos: self.nanos.saturating_sub(other.nanos) }
    }

    /// Multiplies by a scalar
    pub fn mul(&self, rhs: u32) -> Duration {
        Duration { nanos: self.nanos * rhs as u64 }
    }

    /// Divides by a scalar
    pub fn div(&self, rhs: u32) -> Duration {
        Duration { nanos: self.nanos / rhs as u64 }
    }
}

impl Add for Duration {
    type Output = Duration

    fn add(self, other: Duration) -> Duration {
        Duration { nanos: self.nanos + other.nanos }
    }
}

impl Sub for Duration {
    type Output = Duration

    fn sub(self, other: Duration) -> Duration {
        Duration { nanos: self.nanos - other.nanos }
    }
}

impl Ord for Duration {
    fn cmp(&self, other: &Duration) -> Ordering {
        self.nanos.cmp(&other.nanos)
    }
}

impl Eq for Duration {
    fn eq(&self, other: &Duration) -> bool {
        self.nanos == other.nanos
    }
}

/// A point in time
pub struct Instant {
    /// Nanoseconds since some reference point
    nanos: u64,
}

impl Instant {
    /// Returns the current instant
    pub fn now() -> Instant {
        // Would get actual system time
        Instant { nanos: 0 }
    }

    /// Returns the duration since another instant
    pub fn duration_since(&self, earlier: Instant) -> Duration {
        Duration { nanos: self.nanos.saturating_sub(earlier.nanos) }
    }

    /// Returns the duration elapsed since this instant
    pub fn elapsed(&self) -> Duration {
        Instant::now().duration_since(*self)
    }

    /// Adds a duration to this instant
    pub fn checked_add(&self, duration: Duration) -> Option<Instant> {
        self.nanos.checked_add(duration.nanos).map(|n| Instant { nanos: n })
    }

    /// Subtracts a duration from this instant
    pub fn checked_sub(&self, duration: Duration) -> Option<Instant> {
        if self.nanos >= duration.nanos {
            Some(Instant { nanos: self.nanos - duration.nanos })
        } else {
            None
        }
    }
}

impl Add<Duration> for Instant {
    type Output = Instant

    fn add(self, duration: Duration) -> Instant {
        Instant { nanos: self.nanos + duration.nanos }
    }
}

impl Sub<Duration> for Instant {
    type Output = Instant

    fn sub(self, duration: Duration) -> Instant {
        Instant { nanos: self.nanos - duration.nanos }
    }
}

impl Sub<Instant> for Instant {
    type Output = Duration

    fn sub(self, other: Instant) -> Duration {
        self.duration_since(other)
    }
}

impl Ord for Instant {
    fn cmp(&self, other: &Instant) -> Ordering {
        self.nanos.cmp(&other.nanos)
    }
}

impl Eq for Instant {
    fn eq(&self, other: &Instant) -> bool {
        self.nanos == other.nanos
    }
}

/// A future that completes after a duration
///
/// # Example
/// ```
/// async fn example() {
///     // Sleep for 1 second
///     sleep(Duration::from_secs(1)).await
///
///     // Do something after the delay
/// }
/// ```
pub struct Sleep {
    deadline: Instant,
    waker_registered: bool,
}

impl Sleep {
    /// Creates a new sleep future
    fn new(duration: Duration) -> Sleep {
        Sleep {
            deadline: Instant::now() + duration,
            waker_registered: false,
        }
    }

    /// Returns the deadline
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    /// Returns true if the sleep has expired
    pub fn is_elapsed(&self) -> bool {
        Instant::now() >= self.deadline
    }

    /// Resets the sleep timer
    pub fn reset(&mut self, duration: Duration) {
        self.deadline = Instant::now() + duration;
        self.waker_registered = false;
    }
}

impl Future for Sleep {
    type Output = ()

    fn poll(&mut self, cx: &mut Context) -> Poll<()> {
        if Instant::now() >= self.deadline {
            Poll::Ready(())
        } else {
            // Register waker to be called when deadline is reached
            if !self.waker_registered {
                // In a real implementation, this would register with the timer driver
                self.waker_registered = true;
            }
            Poll::Pending
        }
    }
}

/// Sleeps for the given duration
///
/// # Example
/// ```
/// async fn example() {
///     sleep(Duration::from_millis(100)).await
/// }
/// ```
pub fn sleep(duration: Duration) -> Sleep {
    Sleep::new(duration)
}

/// Sleeps until the given instant
pub fn sleep_until(deadline: Instant) -> Sleep {
    let now = Instant::now();
    if deadline > now {
        Sleep::new(deadline.duration_since(now))
    } else {
        Sleep::new(Duration::zero())
    }
}

/// A timeout wrapper for futures
///
/// Wraps a future with a timeout. If the future doesn't complete
/// within the duration, the timeout future completes with an error.
pub struct Timeout<F: Future> {
    future: F,
    deadline: Instant,
    waker_registered: bool,
}

impl<F: Future> Timeout<F> {
    /// Creates a new timeout
    fn new(future: F, duration: Duration) -> Timeout<F> {
        Timeout {
            future,
            deadline: Instant::now() + duration,
            waker_registered: false,
        }
    }

    /// Returns the deadline
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    /// Returns a reference to the inner future
    pub fn get_ref(&self) -> &F {
        &self.future
    }

    /// Returns a mutable reference to the inner future
    pub fn get_mut(&mut self) -> &mut F {
        &mut self.future
    }

    /// Consumes the timeout, returning the inner future
    pub fn into_inner(self) -> F {
        self.future
    }
}

/// Error returned when a timeout expires
pub struct Elapsed {
    _priv: (),
}

impl Elapsed {
    fn new() -> Elapsed {
        Elapsed { _priv: () }
    }
}

impl<F: Future> Future for Timeout<F> {
    type Output = Result<F::Output, Elapsed>

    fn poll(&mut self, cx: &mut Context) -> Poll<Result<F::Output, Elapsed>> {
        // First, check if the inner future is ready
        match self.future.poll(cx) {
            Poll::Ready(value) => return Poll::Ready(Ok(value)),
            Poll::Pending => {}
        }

        // Check if timeout has elapsed
        if Instant::now() >= self.deadline {
            return Poll::Ready(Err(Elapsed::new()));
        }

        // Register for timeout notification
        if !self.waker_registered {
            self.waker_registered = true;
        }

        Poll::Pending
    }
}

/// Wraps a future with a timeout
///
/// # Example
/// ```
/// async fn example() {
///     match timeout(Duration::from_secs(5), do_work()).await {
///         Ok(result) => println("Completed: {}", result),
///         Err(_) => println("Timed out"),
///     }
/// }
/// ```
pub fn timeout<F: Future>(duration: Duration, future: F) -> Timeout<F> {
    Timeout::new(future, duration)
}

/// Wraps a future with a deadline
pub fn timeout_at<F: Future>(deadline: Instant, future: F) -> Timeout<F> {
    let now = Instant::now();
    let duration = if deadline > now {
        deadline.duration_since(now)
    } else {
        Duration::zero()
    };
    Timeout::new(future, duration)
}

/// An interval timer that yields at regular intervals
///
/// # Example
/// ```
/// async fn example() {
///     let mut interval = interval(Duration::from_secs(1))
///
///     loop {
///         interval.tick().await
///         println("Tick!")
///     }
/// }
/// ```
pub struct Interval {
    /// Period between ticks
    period: Duration,
    /// Next tick deadline
    next: Instant,
    /// Missed tick behavior
    missed_tick_behavior: MissedTickBehavior,
}

/// How to handle missed ticks
pub enum MissedTickBehavior {
    /// Fire immediately for each missed tick (burst)
    Burst,
    /// Skip missed ticks, continue from now
    Skip,
    /// Delay next tick to maintain period from now
    Delay,
}

impl Interval {
    /// Creates a new interval timer
    fn new(period: Duration) -> Interval {
        Interval {
            period,
            next: Instant::now() + period,
            missed_tick_behavior: MissedTickBehavior::Burst,
        }
    }

    /// Creates an interval starting immediately
    fn new_immediate(period: Duration) -> Interval {
        Interval {
            period,
            next: Instant::now(),
            missed_tick_behavior: MissedTickBehavior::Burst,
        }
    }

    /// Sets the missed tick behavior
    pub fn set_missed_tick_behavior(&mut self, behavior: MissedTickBehavior) {
        self.missed_tick_behavior = behavior;
    }

    /// Returns the period
    pub fn period(&self) -> Duration {
        self.period
    }

    /// Resets the interval
    pub fn reset(&mut self) {
        self.next = Instant::now() + self.period;
    }

    /// Waits for the next tick
    pub async fn tick(&mut self) -> Instant with Async {
        let now = Instant::now();

        if now >= self.next {
            // Already past the deadline
            let tick_time = self.next;

            match self.missed_tick_behavior {
                MissedTickBehavior::Burst => {
                    // Just advance to next period
                    self.next = self.next + self.period;
                }
                MissedTickBehavior::Skip => {
                    // Skip to next future deadline
                    let elapsed = now.duration_since(self.next);
                    let periods = elapsed.as_nanos() / self.period.as_nanos() + 1;
                    self.next = self.next + self.period.mul(periods as u32);
                }
                MissedTickBehavior::Delay => {
                    // Delay from now
                    self.next = now + self.period;
                }
            }

            tick_time
        } else {
            // Wait until deadline
            sleep_until(self.next).await;
            let tick_time = self.next;
            self.next = self.next + self.period;
            tick_time
        }
    }
}

/// Creates an interval timer
///
/// The first tick happens after the period elapses.
pub fn interval(period: Duration) -> Interval {
    Interval::new(period)
}

/// Creates an interval timer that ticks immediately
///
/// The first tick happens immediately, subsequent ticks happen
/// at regular intervals.
pub fn interval_at(start: Instant, period: Duration) -> Interval {
    let mut interval = Interval::new(period);
    interval.next = start;
    interval
}

/// A deadline future that completes at a specific instant
pub struct Deadline {
    instant: Instant,
}

impl Deadline {
    /// Creates a new deadline
    pub fn new(instant: Instant) -> Deadline {
        Deadline { instant }
    }

    /// Returns the deadline instant
    pub fn deadline(&self) -> Instant {
        self.instant
    }

    /// Returns true if the deadline has passed
    pub fn is_elapsed(&self) -> bool {
        Instant::now() >= self.instant
    }
}

impl Future for Deadline {
    type Output = ()

    fn poll(&mut self, cx: &mut Context) -> Poll<()> {
        if Instant::now() >= self.instant {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Creates a future that completes at the given instant
pub fn deadline(instant: Instant) -> Deadline {
    Deadline::new(instant)
}

/// A future that yields once then completes
///
/// Useful for yielding control back to the executor without
/// actually sleeping.
pub struct YieldNow {
    yielded: bool,
}

impl Future for YieldNow {
    type Output = ()

    fn poll(&mut self, cx: &mut Context) -> Poll<()> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Yields execution to the runtime
///
/// This allows other tasks to run. Useful in compute-heavy
/// loops to prevent starving other tasks.
pub fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}

/// A throttle limiter for rate limiting
pub struct Throttle {
    /// Minimum period between operations
    period: Duration,
    /// Last operation time
    last: Option<Instant>,
}

impl Throttle {
    /// Creates a new throttle with the given period
    pub fn new(period: Duration) -> Throttle {
        Throttle {
            period,
            last: None,
        }
    }

    /// Creates a throttle that allows N operations per second
    pub fn per_second(n: u32) -> Throttle {
        let period = Duration::from_nanos(1_000_000_000 / n as u64);
        Throttle::new(period)
    }

    /// Waits until an operation is allowed
    pub async fn wait(&mut self) with Async {
        if let Some(last) = self.last {
            let elapsed = last.elapsed();
            if elapsed < self.period {
                sleep(self.period - elapsed).await;
            }
        }
        self.last = Some(Instant::now());
    }

    /// Returns true if an operation would be allowed immediately
    pub fn is_ready(&self) -> bool {
        match self.last {
            None => true,
            Some(last) => last.elapsed() >= self.period,
        }
    }

    /// Resets the throttle
    pub fn reset(&mut self) {
        self.last = None;
    }
}

/// A debouncer that delays execution until input settles
pub struct Debounce {
    /// Delay duration
    delay: Duration,
    /// Last trigger time
    last_trigger: Option<Instant>,
}

impl Debounce {
    /// Creates a new debouncer with the given delay
    pub fn new(delay: Duration) -> Debounce {
        Debounce {
            delay,
            last_trigger: None,
        }
    }

    /// Triggers the debouncer
    pub fn trigger(&mut self) {
        self.last_trigger = Some(Instant::now());
    }

    /// Waits for the input to settle
    pub async fn wait(&mut self) with Async {
        loop {
            if let Some(last) = self.last_trigger {
                let elapsed = last.elapsed();
                if elapsed >= self.delay {
                    // Settled
                    self.last_trigger = None;
                    return;
                } else {
                    // Wait for remaining time
                    sleep(self.delay - elapsed).await;
                }
            } else {
                // No trigger yet, wait indefinitely
                // In practice, would use proper notification
                sleep(self.delay).await;
            }
        }
    }
}
