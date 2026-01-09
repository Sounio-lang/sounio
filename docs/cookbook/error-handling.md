# Error Handling Recipes

Practical recipes for robust error handling in Sounio.

## Result<T, E> Patterns

### Problem

You have a function that can fail and need to handle the error appropriately.

### Solution

```sio
use core::{Result, Ok, Err}

enum ParseError {
    InvalidFormat { message: string },
    OutOfRange { value: f64, min: f64, max: f64 },
    MissingField { field: string },
}

fn parse_concentration(s: string) -> Result<f64, ParseError> {
    let trimmed = s.trim()

    if trimmed.is_empty() {
        return Err(ParseError::MissingField {
            field: "concentration",
        })
    }

    match trimmed.parse::<f64>() {
        Ok(value) => {
            // Validate range
            if value < 0.0 {
                return Err(ParseError::OutOfRange {
                    value: value,
                    min: 0.0,
                    max: f64::MAX,
                })
            }
            return Ok(value)
        }
        Err(_) => {
            return Err(ParseError::InvalidFormat {
                message: "Expected numeric value, got: " ++ trimmed,
            })
        }
    }
}

// Using the function
fn process_data(input: string) with IO {
    match parse_concentration(input) {
        Ok(conc) => {
            println("Concentration: " ++ conc.to_string())
        }
        Err(ParseError::InvalidFormat { message }) => {
            eprintln("Format error: " ++ message)
        }
        Err(ParseError::OutOfRange { value, min, max }) => {
            eprintln("Value " ++ value.to_string() ++ " out of range ["
                    ++ min.to_string() ++ ", " ++ max.to_string() ++ "]")
        }
        Err(ParseError::MissingField { field }) => {
            eprintln("Missing required field: " ++ field)
        }
    }
}
```

### Chaining Results

```sio
fn process_patient_data(
    id_str: string,
    dose_str: string,
    weight_str: string
) -> Result<PatientDose, ParseError> {
    // Parse each field, short-circuiting on first error
    let id = match id_str.parse::<i64>() {
        Ok(v) => v,
        Err(_) => return Err(ParseError::InvalidFormat {
            message: "Invalid patient ID",
        })
    }

    let dose = match dose_str.parse::<f64>() {
        Ok(v) => v,
        Err(_) => return Err(ParseError::InvalidFormat {
            message: "Invalid dose",
        })
    }

    let weight = match weight_str.parse::<f64>() {
        Ok(v) => v,
        Err(_) => return Err(ParseError::InvalidFormat {
            message: "Invalid weight",
        })
    }

    // Validate
    if dose <= 0.0 {
        return Err(ParseError::OutOfRange {
            value: dose, min: 0.0, max: 10000.0
        })
    }

    return Ok(PatientDose {
        patient_id: id,
        dose: dose,
        weight: weight,
    })
}
```

---

## Effect-Based Error Handling

### Problem

You want to use effects to handle errors in a structured way.

### Solution

```sio
// Define an error effect
effect Fail {
    fn fail(message: string) -> !;
}

// Function that may fail
fn validate_dose(dose: f64) with Fail {
    if dose <= 0.0 {
        perform Fail.fail("Dose must be positive")
    }
    if dose > 10000.0 {
        perform Fail.fail("Dose exceeds maximum allowed value")
    }
}

fn calculate_concentration(dose: f64, volume: f64) -> f64 with Fail {
    validate_dose(dose)

    if volume <= 0.0 {
        perform Fail.fail("Volume must be positive")
    }

    return dose / volume
}

// Handler that converts to Option
fn try_calculate(dose: f64, volume: f64) -> Option<f64> {
    handle {
        let result = calculate_concentration(dose, volume)
        Some(result)
    } with {
        fail(msg) => None,
    }
}

// Handler that converts to Result
fn safe_calculate(dose: f64, volume: f64) -> Result<f64, string> {
    handle {
        let result = calculate_concentration(dose, volume)
        Ok(result)
    } with {
        fail(msg) => Err(msg),
    }
}

// Handler that logs and provides default
fn calculate_with_default(dose: f64, volume: f64, default: f64) -> f64 with IO {
    handle {
        calculate_concentration(dose, volume)
    } with {
        fail(msg) => {
            eprintln("Warning: " ++ msg ++ ", using default")
            default
        }
    }
}
```

### Discussion

Effect-based error handling provides:
- **Separation of concerns**: Error handling logic is separate from business logic
- **Composability**: Multiple effects can be combined
- **Flexibility**: Different handlers can provide different behaviors
- **Type safety**: The type system tracks which effects a function may perform

---

## Combining Errors from Multiple Sources

### Problem

You need to process multiple items and collect all errors rather than stopping at the first one.

### Solution

```sio
struct BatchResult<T, E> {
    successes: Vec<T>,
    failures: Vec<(i64, E)>,  // (index, error)
}

fn process_batch<T, E, F>(
    items: Vec<string>,
    processor: F
) -> BatchResult<T, E>
where F: fn(string) -> Result<T, E>
{
    var successes: Vec<T> = vec![]
    var failures: Vec<(i64, E)> = vec![]

    for i in 0..items.len() {
        match processor(items[i].clone()) {
            Ok(value) => successes.push(value),
            Err(error) => failures.push((i as i64, error)),
        }
    }

    return BatchResult {
        successes: successes,
        failures: failures,
    }
}

// Example usage
fn process_concentrations(data: Vec<string>) -> BatchResult<f64, ParseError> {
    return process_batch(data, parse_concentration)
}

// Report results
fn report_batch_result<T, E: Debug>(result: BatchResult<T, E>) with IO {
    println("Successfully processed: " ++ result.successes.len().to_string())

    if result.failures.len() > 0 {
        println("Errors (" ++ result.failures.len().to_string() ++ "):")
        for failure in result.failures {
            let idx = failure.0
            let err = failure.1
            println("  Row " ++ idx.to_string() ++ ": " ++ format_error(err))
        }
    }
}
```

### Accumulating Validation Errors

```sio
struct ValidationErrors {
    errors: Vec<string>,
}

impl ValidationErrors {
    fn new() -> ValidationErrors {
        ValidationErrors { errors: vec![] }
    }

    fn add(&!self, error: string) {
        self.errors.push(error)
    }

    fn has_errors(&self) -> bool {
        self.errors.len() > 0
    }

    fn into_result<T>(self, value: T) -> Result<T, Vec<string>> {
        if self.errors.len() > 0 {
            Err(self.errors)
        } else {
            Ok(value)
        }
    }
}

fn validate_patient(patient: PatientData) -> Result<PatientData, Vec<string>> {
    var errors = ValidationErrors::new()

    if patient.age < 0 || patient.age > 150 {
        errors.add("Age must be between 0 and 150")
    }

    if patient.weight <= 0.0 {
        errors.add("Weight must be positive")
    }

    if patient.creatinine < 0.0 {
        errors.add("Creatinine cannot be negative")
    }

    if patient.name.is_empty() {
        errors.add("Name is required")
    }

    return errors.into_result(patient)
}
```

---

## Confidence-Based Fallbacks

### Problem

You want to use a primary method but fall back to alternatives based on confidence levels.

### Solution

```sio
use epistemic::core::*

struct MethodResult {
    value: EpistemicValue,
    method: string,
}

fn estimate_parameter(data: &[f64]) -> MethodResult {
    // Try primary method (requires more data)
    if data.len() >= 10 {
        let result = primary_estimation(data)
        if result.conf >= 0.90 {
            return MethodResult {
                value: result,
                method: "primary (MLE)",
            }
        }
    }

    // Try secondary method (works with less data)
    if data.len() >= 3 {
        let result = secondary_estimation(data)
        if result.conf >= 0.70 {
            return MethodResult {
                value: result,
                method: "secondary (moments)",
            }
        }
    }

    // Fallback to prior
    return MethodResult {
        value: epistemic_std(10.0, 5.0, 0.50),
        method: "fallback (prior)",
    }
}

fn primary_estimation(data: &[f64]) -> EpistemicValue {
    // Maximum likelihood estimation
    let n = data.len() as f64
    var sum = 0.0
    var sum_sq = 0.0

    for x in data {
        sum = sum + *x
        sum_sq = sum_sq + *x * *x
    }

    let mean = sum / n
    let variance = (sum_sq - sum * sum / n) / (n - 1.0)
    let std_error = sqrt_f64(variance / n)

    // Confidence based on sample size
    let confidence = 1.0 - 1.0 / n

    return epistemic_std(mean, std_error, confidence)
}

fn secondary_estimation(data: &[f64]) -> EpistemicValue {
    // Method of moments (simpler, less efficient)
    let n = data.len() as f64
    var sum = 0.0

    for x in data {
        sum = sum + *x
    }

    let mean = sum / n
    // Use conservative uncertainty estimate
    let std_error = mean * 0.2  // Assume 20% uncertainty

    let confidence = 0.5 + 0.1 * n  // Increases with sample size

    return epistemic_std(mean, std_error, min_f64(confidence, 0.85))
}
```

### Tiered Confidence Decisions

```sio
enum Decision {
    Proceed { confidence: f64 },
    RequireReview { reason: string },
    Reject { reason: string },
}

fn make_decision(measurement: EpistemicValue, threshold: f64) -> Decision {
    let value = measurement.value
    let conf = measurement.conf
    let lower = get_interval_lo(measurement)
    let upper = get_interval_hi(measurement)

    // High confidence: auto-proceed
    if conf >= 0.95 && lower > threshold {
        return Decision::Proceed { confidence: conf }
    }

    // Medium confidence: require review
    if conf >= 0.75 {
        if value > threshold {
            return Decision::RequireReview {
                reason: "Value above threshold but confidence is " ++ conf.to_string(),
            }
        } else {
            return Decision::Reject {
                reason: "Value below threshold",
            }
        }
    }

    // Low confidence: cannot make decision
    return Decision::RequireReview {
        reason: "Insufficient confidence (" ++ conf.to_string() ++ ") for decision",
    }
}
```

---

## Panic Handling

### Problem

You need to handle panics gracefully in critical sections.

### Solution

```sio
// Declare the Panic effect
fn risky_operation(x: f64) -> f64 with Panic {
    if x < 0.0 {
        panic("Negative input not allowed")
    }
    return sqrt_f64(x)
}

// Wrap in handler to catch panics
fn safe_sqrt(x: f64) -> Option<f64> {
    handle {
        Some(risky_operation(x))
    } with {
        panic(msg) => None,
    }
}

// Alternative: convert panic to Result
fn checked_sqrt(x: f64) -> Result<f64, string> {
    handle {
        Ok(risky_operation(x))
    } with {
        panic(msg) => Err(msg),
    }
}
```

### Discussion

In Sounio, `Panic` is an effect that must be declared. This makes panic-prone code explicit in the type system. Handlers can convert panics to recoverable errors.

---

## Logging Errors

### Problem

You want to log errors for debugging while still handling them gracefully.

### Solution

```sio
use std::io::*

enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

fn log(level: LogLevel, message: string) with IO {
    let prefix = match level {
        LogLevel::Debug => "[DEBUG]",
        LogLevel::Info => "[INFO]",
        LogLevel::Warning => "[WARN]",
        LogLevel::Error => "[ERROR]",
    }
    eprintln(prefix ++ " " ++ message)
}

fn process_with_logging(data: Vec<string>) -> Vec<f64> with IO {
    var results: Vec<f64> = vec![]
    var error_count = 0

    log(LogLevel::Info, "Processing " ++ data.len().to_string() ++ " items")

    for i in 0..data.len() {
        log(LogLevel::Debug, "Processing item " ++ i.to_string())

        match parse_concentration(data[i].clone()) {
            Ok(value) => {
                results.push(value)
            }
            Err(err) => {
                error_count = error_count + 1
                log(LogLevel::Warning,
                    "Failed to parse item " ++ i.to_string() ++ ": " ++ format_error(err))
            }
        }
    }

    if error_count > 0 {
        log(LogLevel::Warning,
            "Completed with " ++ error_count.to_string() ++ " errors")
    } else {
        log(LogLevel::Info, "Completed successfully")
    }

    return results
}
```

---

## Retry with Backoff

### Problem

You want to retry a failing operation with exponential backoff.

### Solution

```sio
struct RetryConfig {
    max_attempts: i32,
    initial_delay_ms: i32,
    max_delay_ms: i32,
    backoff_factor: f64,
}

fn with_retry<T, E, F>(
    operation: F,
    config: RetryConfig
) -> Result<T, E> with IO
where F: fn() -> Result<T, E>
{
    var attempt = 0
    var delay = config.initial_delay_ms

    while attempt < config.max_attempts {
        attempt = attempt + 1

        match operation() {
            Ok(value) => {
                if attempt > 1 {
                    log(LogLevel::Info,
                        "Succeeded on attempt " ++ attempt.to_string())
                }
                return Ok(value)
            }
            Err(err) => {
                if attempt >= config.max_attempts {
                    log(LogLevel::Error,
                        "Failed after " ++ attempt.to_string() ++ " attempts")
                    return Err(err)
                }

                log(LogLevel::Warning,
                    "Attempt " ++ attempt.to_string() ++ " failed, retrying in "
                    ++ delay.to_string() ++ "ms")

                sleep_ms(delay)

                // Exponential backoff
                delay = min_i32(
                    (delay as f64 * config.backoff_factor) as i32,
                    config.max_delay_ms
                )
            }
        }
    }

    // Should not reach here
    panic("Retry logic error")
}

// Usage
fn fetch_with_retry(url: string) -> Result<string, string> with IO {
    let config = RetryConfig {
        max_attempts: 3,
        initial_delay_ms: 100,
        max_delay_ms: 5000,
        backoff_factor: 2.0,
    }

    with_retry(|| fetch_url(url.clone()), config)
}
```

---

## See Also

- [Uncertainty Recipes](uncertainty-recipes.md) for confidence-based decisions
- [Data Loading](data-loading.md) for validation patterns
- [Effects System](../language/effects.md) for effect-based error handling
