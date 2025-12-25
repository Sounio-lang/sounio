/// Async Combinators: Select and Join
///
/// Provides combinators for concurrent execution:
/// - select: Wait for one of multiple futures
/// - join: Wait for all futures concurrently
/// - race: Return first completion (discarding others)

module async::select

import async::future::{Future, Poll, Context, Waker, Fuse}

// =============================================================================
// Select - Wait for first future to complete
// =============================================================================

/// Result of a select operation
pub enum SelectResult<A, B> {
    First(A),
    Second(B),
}

/// Selects between two futures, returning when either completes
///
/// # Example
/// ```
/// async fn example() {
///     let result = select(
///         async { sleep(Duration::from_secs(1)).await; "slow" },
///         async { sleep(Duration::from_millis(100)).await; "fast" }
///     ).await
///
///     match result {
///         SelectResult::First(v) => println("First: {}", v),
///         SelectResult::Second(v) => println("Second: {}", v),
///     }
/// }
/// ```
pub fn select<A: Future, B: Future>(a: A, b: B) -> Select2<A, B> {
    Select2 { a: Some(a), b: Some(b) }
}

/// Future for selecting between two futures
pub struct Select2<A: Future, B: Future> {
    a: Option<A>,
    b: Option<B>,
}

impl<A: Future, B: Future> Future for Select2<A, B> {
    type Output = SelectResult<A::Output, B::Output>

    fn poll(&mut self, cx: &mut Context) -> Poll<Self::Output> {
        // Poll first future
        if let Some(ref mut a) = self.a {
            if let Poll::Ready(value) = a.poll(cx) {
                self.a = None;
                return Poll::Ready(SelectResult::First(value));
            }
        }

        // Poll second future
        if let Some(ref mut b) = self.b {
            if let Poll::Ready(value) = b.poll(cx) {
                self.b = None;
                return Poll::Ready(SelectResult::Second(value));
            }
        }

        Poll::Pending
    }
}

/// Result of a 3-way select
pub enum Select3Result<A, B, C> {
    First(A),
    Second(B),
    Third(C),
}

/// Selects between three futures
pub fn select3<A: Future, B: Future, C: Future>(a: A, b: B, c: C) -> Select3<A, B, C> {
    Select3 { a: Some(a), b: Some(b), c: Some(c) }
}

pub struct Select3<A: Future, B: Future, C: Future> {
    a: Option<A>,
    b: Option<B>,
    c: Option<C>,
}

impl<A: Future, B: Future, C: Future> Future for Select3<A, B, C> {
    type Output = Select3Result<A::Output, B::Output, C::Output>

    fn poll(&mut self, cx: &mut Context) -> Poll<Self::Output> {
        if let Some(ref mut a) = self.a {
            if let Poll::Ready(value) = a.poll(cx) {
                self.a = None;
                return Poll::Ready(Select3Result::First(value));
            }
        }

        if let Some(ref mut b) = self.b {
            if let Poll::Ready(value) = b.poll(cx) {
                self.b = None;
                return Poll::Ready(Select3Result::Second(value));
            }
        }

        if let Some(ref mut c) = self.c {
            if let Poll::Ready(value) = c.poll(cx) {
                self.c = None;
                return Poll::Ready(Select3Result::Third(value));
            }
        }

        Poll::Pending
    }
}

/// Result of a 4-way select
pub enum Select4Result<A, B, C, D> {
    First(A),
    Second(B),
    Third(C),
    Fourth(D),
}

/// Selects between four futures
pub fn select4<A: Future, B: Future, C: Future, D: Future>(
    a: A, b: B, c: C, d: D
) -> Select4<A, B, C, D> {
    Select4 { a: Some(a), b: Some(b), c: Some(c), d: Some(d) }
}

pub struct Select4<A: Future, B: Future, C: Future, D: Future> {
    a: Option<A>,
    b: Option<B>,
    c: Option<C>,
    d: Option<D>,
}

impl<A: Future, B: Future, C: Future, D: Future> Future for Select4<A, B, C, D> {
    type Output = Select4Result<A::Output, B::Output, C::Output, D::Output>

    fn poll(&mut self, cx: &mut Context) -> Poll<Self::Output> {
        if let Some(ref mut a) = self.a {
            if let Poll::Ready(value) = a.poll(cx) {
                self.a = None;
                return Poll::Ready(Select4Result::First(value));
            }
        }

        if let Some(ref mut b) = self.b {
            if let Poll::Ready(value) = b.poll(cx) {
                self.b = None;
                return Poll::Ready(Select4Result::Second(value));
            }
        }

        if let Some(ref mut c) = self.c {
            if let Poll::Ready(value) = c.poll(cx) {
                self.c = None;
                return Poll::Ready(Select4Result::Third(value));
            }
        }

        if let Some(ref mut d) = self.d {
            if let Poll::Ready(value) = d.poll(cx) {
                self.d = None;
                return Poll::Ready(Select4Result::Fourth(value));
            }
        }

        Poll::Pending
    }
}

/// Selects from a vector of futures, returning the first to complete
pub fn select_vec<F: Future>(futures: Vec<F>) -> SelectVec<F> {
    SelectVec { futures }
}

pub struct SelectVec<F: Future> {
    futures: Vec<F>,
}

impl<F: Future> Future for SelectVec<F> {
    type Output = (usize, F::Output)

    fn poll(&mut self, cx: &mut Context) -> Poll<(usize, F::Output)> {
        for (i, future) in self.futures.iter_mut().enumerate() {
            if let Poll::Ready(value) = future.poll(cx) {
                return Poll::Ready((i, value));
            }
        }
        Poll::Pending
    }
}

// =============================================================================
// Join - Wait for all futures to complete
// =============================================================================

/// Joins two futures, waiting for both to complete
///
/// # Example
/// ```
/// async fn example() {
///     let (a, b) = join(
///         async { do_work_a().await },
///         async { do_work_b().await }
///     ).await
/// }
/// ```
pub fn join<A: Future, B: Future>(a: A, b: B) -> Join2<A, B> {
    Join2 {
        a: MaybeDone::Pending(a),
        b: MaybeDone::Pending(b),
    }
}

enum MaybeDone<F: Future> {
    Pending(F),
    Done(F::Output),
    Taken,
}

impl<F: Future> MaybeDone<F> {
    fn poll(&mut self, cx: &mut Context) -> bool {
        match self {
            MaybeDone::Pending(f) => {
                if let Poll::Ready(value) = f.poll(cx) {
                    *self = MaybeDone::Done(value);
                    true
                } else {
                    false
                }
            }
            MaybeDone::Done(_) => true,
            MaybeDone::Taken => true,
        }
    }

    fn take(&mut self) -> F::Output {
        match std::mem::replace(self, MaybeDone::Taken) {
            MaybeDone::Done(value) => value,
            _ => panic("MaybeDone not ready"),
        }
    }
}

pub struct Join2<A: Future, B: Future> {
    a: MaybeDone<A>,
    b: MaybeDone<B>,
}

impl<A: Future, B: Future> Future for Join2<A, B> {
    type Output = (A::Output, B::Output)

    fn poll(&mut self, cx: &mut Context) -> Poll<(A::Output, B::Output)> {
        let a_done = self.a.poll(cx);
        let b_done = self.b.poll(cx);

        if a_done && b_done {
            Poll::Ready((self.a.take(), self.b.take()))
        } else {
            Poll::Pending
        }
    }
}

/// Joins three futures
pub fn join3<A: Future, B: Future, C: Future>(a: A, b: B, c: C) -> Join3<A, B, C> {
    Join3 {
        a: MaybeDone::Pending(a),
        b: MaybeDone::Pending(b),
        c: MaybeDone::Pending(c),
    }
}

pub struct Join3<A: Future, B: Future, C: Future> {
    a: MaybeDone<A>,
    b: MaybeDone<B>,
    c: MaybeDone<C>,
}

impl<A: Future, B: Future, C: Future> Future for Join3<A, B, C> {
    type Output = (A::Output, B::Output, C::Output)

    fn poll(&mut self, cx: &mut Context) -> Poll<Self::Output> {
        let a_done = self.a.poll(cx);
        let b_done = self.b.poll(cx);
        let c_done = self.c.poll(cx);

        if a_done && b_done && c_done {
            Poll::Ready((self.a.take(), self.b.take(), self.c.take()))
        } else {
            Poll::Pending
        }
    }
}

/// Joins four futures
pub fn join4<A: Future, B: Future, C: Future, D: Future>(
    a: A, b: B, c: C, d: D
) -> Join4<A, B, C, D> {
    Join4 {
        a: MaybeDone::Pending(a),
        b: MaybeDone::Pending(b),
        c: MaybeDone::Pending(c),
        d: MaybeDone::Pending(d),
    }
}

pub struct Join4<A: Future, B: Future, C: Future, D: Future> {
    a: MaybeDone<A>,
    b: MaybeDone<B>,
    c: MaybeDone<C>,
    d: MaybeDone<D>,
}

impl<A: Future, B: Future, C: Future, D: Future> Future for Join4<A, B, C, D> {
    type Output = (A::Output, B::Output, C::Output, D::Output)

    fn poll(&mut self, cx: &mut Context) -> Poll<Self::Output> {
        let a_done = self.a.poll(cx);
        let b_done = self.b.poll(cx);
        let c_done = self.c.poll(cx);
        let d_done = self.d.poll(cx);

        if a_done && b_done && c_done && d_done {
            Poll::Ready((self.a.take(), self.b.take(), self.c.take(), self.d.take()))
        } else {
            Poll::Pending
        }
    }
}

/// Joins a vector of futures
pub fn join_all<F: Future>(futures: Vec<F>) -> JoinAll<F> {
    let states = futures.into_iter()
        .map(|f| MaybeDone::Pending(f))
        .collect();
    JoinAll { states }
}

pub struct JoinAll<F: Future> {
    states: Vec<MaybeDone<F>>,
}

impl<F: Future> Future for JoinAll<F> {
    type Output = Vec<F::Output>

    fn poll(&mut self, cx: &mut Context) -> Poll<Vec<F::Output>> {
        let mut all_done = true;

        for state in &mut self.states {
            if !state.poll(cx) {
                all_done = false;
            }
        }

        if all_done {
            let results = self.states.iter_mut()
                .map(|s| s.take())
                .collect();
            Poll::Ready(results)
        } else {
            Poll::Pending
        }
    }
}

// =============================================================================
// Try Join - Join with early exit on error
// =============================================================================

/// Joins two futures, returning early if either fails
///
/// # Example
/// ```
/// async fn example() -> Result<(i32, i32), Error> {
///     try_join(
///         async { Ok(1) },
///         async { Ok(2) }
///     ).await
/// }
/// ```
pub fn try_join<A, B, E, FA, FB>(a: FA, b: FB) -> TryJoin2<FA, FB>
where
    FA: Future<Output = Result<A, E>>,
    FB: Future<Output = Result<B, E>>,
{
    TryJoin2 {
        a: MaybeDone::Pending(a),
        b: MaybeDone::Pending(b),
    }
}

pub struct TryJoin2<A: Future, B: Future> {
    a: MaybeDone<A>,
    b: MaybeDone<B>,
}

impl<A, B, E, FA, FB> Future for TryJoin2<FA, FB>
where
    FA: Future<Output = Result<A, E>>,
    FB: Future<Output = Result<B, E>>,
{
    type Output = Result<(A, B), E>

    fn poll(&mut self, cx: &mut Context) -> Poll<Result<(A, B), E>> {
        // Poll first, checking for error
        match &mut self.a {
            MaybeDone::Pending(f) => {
                if let Poll::Ready(result) = f.poll(cx) {
                    match result {
                        Ok(v) => self.a = MaybeDone::Done(Ok(v)),
                        Err(e) => return Poll::Ready(Err(e)),
                    }
                }
            }
            _ => {}
        }

        // Poll second, checking for error
        match &mut self.b {
            MaybeDone::Pending(f) => {
                if let Poll::Ready(result) = f.poll(cx) {
                    match result {
                        Ok(v) => self.b = MaybeDone::Done(Ok(v)),
                        Err(e) => return Poll::Ready(Err(e)),
                    }
                }
            }
            _ => {}
        }

        // Check if both are done
        match (&self.a, &self.b) {
            (MaybeDone::Done(_), MaybeDone::Done(_)) => {
                let a_val = match self.a.take() {
                    Ok(v) => v,
                    Err(e) => return Poll::Ready(Err(e)),
                };
                let b_val = match self.b.take() {
                    Ok(v) => v,
                    Err(e) => return Poll::Ready(Err(e)),
                };
                Poll::Ready(Ok((a_val, b_val)))
            }
            _ => Poll::Pending,
        }
    }
}

/// Tries to join all futures, returning early on first error
pub fn try_join_all<T, E, F: Future<Output = Result<T, E>>>(
    futures: Vec<F>
) -> TryJoinAll<F, T, E> {
    let states = futures.into_iter()
        .map(|f| MaybeDone::Pending(f))
        .collect();
    TryJoinAll { states, _marker: PhantomData }
}

pub struct TryJoinAll<F, T, E> {
    states: Vec<MaybeDone<F>>,
    _marker: PhantomData<(T, E)>,
}

impl<T, E, F: Future<Output = Result<T, E>>> Future for TryJoinAll<F, T, E> {
    type Output = Result<Vec<T>, E>

    fn poll(&mut self, cx: &mut Context) -> Poll<Result<Vec<T>, E>> {
        let mut all_done = true;

        for state in &mut self.states {
            match state {
                MaybeDone::Pending(f) => {
                    if let Poll::Ready(result) = f.poll(cx) {
                        match result {
                            Ok(v) => *state = MaybeDone::Done(Ok(v)),
                            Err(e) => return Poll::Ready(Err(e)),
                        }
                    } else {
                        all_done = false;
                    }
                }
                _ => {}
            }
        }

        if all_done {
            let results: Result<Vec<T>, E> = self.states.iter_mut()
                .map(|s| s.take())
                .collect();
            Poll::Ready(results)
        } else {
            Poll::Pending
        }
    }
}

// =============================================================================
// Race - First completion wins, others are dropped
// =============================================================================

/// Races two futures, returning when either completes
///
/// Unlike select, race returns the same type from both branches.
pub fn race<T, A: Future<Output = T>, B: Future<Output = T>>(a: A, b: B) -> Race2<A, B> {
    Race2 { a: Some(a), b: Some(b) }
}

pub struct Race2<A: Future, B: Future> {
    a: Option<A>,
    b: Option<B>,
}

impl<T, A: Future<Output = T>, B: Future<Output = T>> Future for Race2<A, B> {
    type Output = T

    fn poll(&mut self, cx: &mut Context) -> Poll<T> {
        if let Some(ref mut a) = self.a {
            if let Poll::Ready(value) = a.poll(cx) {
                self.a = None;
                self.b = None;  // Drop the loser
                return Poll::Ready(value);
            }
        }

        if let Some(ref mut b) = self.b {
            if let Poll::Ready(value) = b.poll(cx) {
                self.a = None;  // Drop the loser
                self.b = None;
                return Poll::Ready(value);
            }
        }

        Poll::Pending
    }
}

/// Races a vector of futures
pub fn race_all<F: Future>(futures: Vec<F>) -> RaceAll<F> {
    RaceAll { futures }
}

pub struct RaceAll<F: Future> {
    futures: Vec<F>,
}

impl<F: Future> Future for RaceAll<F> {
    type Output = F::Output

    fn poll(&mut self, cx: &mut Context) -> Poll<F::Output> {
        for future in &mut self.futures {
            if let Poll::Ready(value) = future.poll(cx) {
                return Poll::Ready(value);
            }
        }
        Poll::Pending
    }
}

// =============================================================================
// Either - Type-safe wrapper for two different future types
// =============================================================================

/// An either type for two futures with different outputs
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<A: Future, B: Future> Future for Either<A, B> {
    type Output = Either<A::Output, B::Output>

    fn poll(&mut self, cx: &mut Context) -> Poll<Self::Output> {
        match self {
            Either::Left(a) => a.poll(cx).map(Either::Left),
            Either::Right(b) => b.poll(cx).map(Either::Right),
        }
    }
}

// Marker type for generic bounds
struct PhantomData<T> {}
