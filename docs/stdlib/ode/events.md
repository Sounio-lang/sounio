# ODE Event Handling API Reference

Event handling allows ODE solvers to detect and respond to specific conditions during integration. Events are used for discontinuities, state-dependent transitions, and root finding.

## Overview

Event handling in ODE solvers involves:
1. **Event detection**: Finding when a condition becomes true
2. **Event location**: Precisely locating the time of the event
3. **Event handling**: Executing callbacks or modifying state

## Event Types

### Zero-Crossing Events

The most common event type detects when a continuous function crosses zero.

```sio
/// Event function signature
/// Returns a value that triggers event when crossing zero
type EventFn = fn(t: f64, y: &[f64]) -> f64
```

### Direction Specification

```sio
/// Event crossing direction
pub enum EventDirection {
    /// Trigger when g(t,y) crosses zero from negative to positive
    Rising,
    /// Trigger when g(t,y) crosses zero from positive to negative
    Falling,
    /// Trigger on any zero crossing
    Both,
}
```

### Event Action

```sio
/// What to do when event is detected
pub enum EventAction {
    /// Continue integration without change
    Continue,
    /// Terminate integration at event time
    Terminate,
    /// Modify state and continue
    Reinitialize,
}
```

## Event Definition

```sio
/// Complete event specification
pub struct Event {
    /// Event detection function
    pub condition: fn(f64, &[f64]) -> f64,
    /// Crossing direction
    pub direction: EventDirection,
    /// Callback executed when event triggers
    pub callback: fn(f64, &![f64]) -> EventAction,
    /// Enable/disable this event
    pub active: bool,
}
```

### `Event::new`

Create a new event.

```sio
pub fn new(
    condition: fn(f64, &[f64]) -> f64,
    direction: EventDirection,
    callback: fn(f64, &![f64]) -> EventAction
) -> Event
```

**Example:**
```sio
// Detect when y[0] = 0.5
fn threshold_condition(t: f64, y: &[f64]) -> f64 {
    return y[0] - 0.5
}

fn on_threshold(t: f64, y: &![f64]) -> EventAction {
    println("Threshold reached at t = ", t)
    return EventAction::Continue
}

let event = Event::new(
    threshold_condition,
    EventDirection::Falling,
    on_threshold
)
```

## Event Location

### Bisection Method

The solver uses bisection to locate events precisely:

```sio
/// Locate event time using bisection
fn locate_event(
    solver: &ODESolver,
    t_lo: f64,
    t_hi: f64,
    y_lo: &[f64],
    y_hi: &[f64],
    event: &Event,
    tol: f64
) -> f64 {
    var t_a = t_lo
    var t_b = t_hi
    var g_a = (event.condition)(t_a, y_lo)

    while t_b - t_a > tol {
        let t_mid = 0.5 * (t_a + t_b)
        let y_mid = interpolate(solver, t_mid)
        let g_mid = (event.condition)(t_mid, &y_mid)

        if g_a * g_mid < 0.0 {
            t_b = t_mid
        } else {
            t_a = t_mid
            g_a = g_mid
        }
    }

    return 0.5 * (t_a + t_b)
}
```

## Common Event Patterns

### Bouncing Ball

Classic physics problem with discontinuous velocity.

```sio
// State: y[0] = height, y[1] = velocity
fn ball_rhs(t: f64, y: &[f64], dydt: &![f64]) {
    let g = 9.81
    dydt[0] = y[1]        // dh/dt = v
    dydt[1] = -g          // dv/dt = -g
}

// Event: ball hits ground (height = 0)
fn ground_hit(t: f64, y: &[f64]) -> f64 {
    return y[0]  // Triggers when height crosses zero
}

// Bounce: reverse velocity with energy loss
fn bounce(t: f64, y: &![f64]) -> EventAction {
    let restitution = 0.8
    y[1] = -restitution * y[1]  // Reverse and reduce velocity
    return EventAction::Reinitialize
}

let bounce_event = Event::new(ground_hit, EventDirection::Falling, bounce)
```

### Dosing Events

Pharmacokinetics with repeated dosing.

```sio
/// Scheduled dose times and amounts
struct DosingSchedule {
    times: &[f64],
    doses: &[f64],
    current_idx: i64,
}

/// Event: time for next dose
fn next_dose_time(t: f64, y: &[f64], schedule: &DosingSchedule) -> f64 {
    if schedule.current_idx >= schedule.times.len() as i64 {
        return 1e10  // No more doses
    }
    return t - schedule.times[schedule.current_idx as usize]
}

/// Apply dose by adding to gut compartment
fn apply_dose(t: f64, y: &![f64], schedule: &!DosingSchedule) -> EventAction {
    let dose = schedule.doses[schedule.current_idx as usize]
    y[0] = y[0] + dose  // Add to gut compartment
    schedule.current_idx = schedule.current_idx + 1
    return EventAction::Reinitialize
}
```

### Threshold Detection

Detect when a value crosses a threshold.

```sio
// Detect when concentration exceeds toxic level
fn toxic_level(t: f64, y: &[f64]) -> f64 {
    let toxic_threshold = 100.0
    return y[1] - toxic_threshold  // y[1] is concentration
}

fn warn_toxic(t: f64, y: &![f64]) -> EventAction {
    println("WARNING: Toxic concentration at t = ", t)
    println("  Concentration = ", y[1])
    return EventAction::Continue
}

let toxic_event = Event::new(
    toxic_level,
    EventDirection::Rising,
    warn_toxic
)
```

### Terminal Events

Stop integration at a specific condition.

```sio
// Stop when equilibrium reached
fn equilibrium_check(t: f64, y: &[f64]) -> f64 {
    // Detect when derivative magnitude is small
    let tol = 1e-6
    // Assuming y[0] represents the main state variable
    // This is a simplified check
    return abs(y[1]) - tol  // y[1] is velocity/derivative
}

fn terminate_at_equilibrium(t: f64, y: &![f64]) -> EventAction {
    println("Equilibrium reached at t = ", t)
    return EventAction::Terminate
}
```

## Solver Integration

### Adaptive Solver with Events

```sio
pub struct ODESolverWithEvents {
    /// Base solver configuration
    pub config: ODEConfig,
    /// Registered events
    pub events: &[Event],
    /// Event tolerance for location
    pub event_tol: f64,
    /// Maximum events before termination
    pub max_events: i64,
}

/// Solve with event handling
pub fn solve_with_events(
    rhs: fn(f64, &[f64], &![f64]),
    y0: &[f64],
    t_span: (f64, f64),
    solver: &ODESolverWithEvents
) -> ODESolutionWithEvents
```

### Solution with Event History

```sio
pub struct ODESolutionWithEvents {
    /// Final solution
    pub y_final: &[f64],
    /// Final time
    pub t_final: f64,
    /// Whether solver succeeded
    pub success: bool,
    /// Event times detected
    pub event_times: &[f64],
    /// Which event triggered at each time
    pub event_indices: &[i64],
    /// Number of events detected
    pub n_events: i64,
    /// Termination reason
    pub termination: TerminationReason,
}

pub enum TerminationReason {
    ReachedTEnd,
    EventTermination,
    MaxEventsExceeded,
    MaxStepsExceeded,
    StepSizeTooSmall,
}
```

## Implementation Notes

### Event Detection Algorithm

During each step from `t_n` to `t_{n+1}`:

1. Evaluate event function at both endpoints
2. Check for sign change (zero crossing)
3. If crossing detected:
   - Use bisection/interpolation to locate event precisely
   - Step to event time
   - Execute callback
   - Handle reinitialize/terminate as specified
4. Continue or terminate based on callback return

### Dense Output

For accurate event location, solvers provide dense output (interpolation):

```sio
/// Get solution at arbitrary time within last step
fn interpolate(solver: &ODESolver, t: f64) -> &[f64]
```

Tsit5 provides natural dense output using the Butcher tableau coefficients.

### Multiple Events

When multiple events are active:

1. Check all events for crossings
2. Process the earliest event first
3. After handling, re-check for other events at the event time
4. Continue until no events at current time

### Chattering Prevention

When events occur rapidly (chattering):

```sio
pub struct EventOptions {
    /// Minimum time between events of same type
    pub cooldown: f64,
    /// Skip events that would chatter
    pub prevent_chattering: bool,
}
```

## Performance Considerations

1. **Event function cost**: Keep event functions lightweight
2. **Location tolerance**: Balance precision vs. extra function evaluations
3. **Dense output**: Use native interpolation rather than extra steps
4. **Event count**: Limit active events for performance

## Example: Complete Integration

```sio
use ode::tsit5::*
use ode::events::*

fn main() {
    // Bouncing ball with stopping condition
    let config = default_config()

    let events = [
        Event::new(ground_hit, EventDirection::Falling, bounce),
        Event::new(velocity_threshold, EventDirection::Both, check_stop)
    ]

    let solver = ODESolverWithEvents {
        config: config,
        events: &events,
        event_tol: 1e-10,
        max_events: 100,
    }

    let y0 = [10.0, 0.0]  // height=10m, velocity=0
    let sol = solve_with_events(ball_rhs, &y0, (0.0, 100.0), &solver)

    println("Final time: ", sol.t_final)
    println("Events detected: ", sol.n_events)
    for i in 0..sol.n_events {
        println("  Event ", sol.event_indices[i], " at t=", sol.event_times[i])
    }
}
```

## See Also

- [ODE Solvers](solvers.md)
- [Linear Algebra](../linalg/matrices.md)
