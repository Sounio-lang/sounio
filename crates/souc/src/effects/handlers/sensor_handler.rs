//! Sensor Effect Handler
//!
//! This module implements the `Sensor` effect handler for physical sensor and
//! measurement operations through algebraic effects. The Sensor effect enables
//! reading from various sensors with explicit uncertainty tracking.
//!
//! # Effect Operations
//!
//! | Operation        | Arguments                      | Return Type | Confidence |
//! |-----------------|--------------------------------|-------------|------------|
//! | `read`          | `(sensor_id: String)`          | `Float`     | 0.95       |
//! | `read_with_unit`| `(sensor_id: String)`          | `Measurement` | 0.95     |
//! | `calibrate`     | `(sensor_id: String, ref: Float)` | `Unit`   | 1.0        |
//! | `get_accuracy`  | `(sensor_id: String)`          | `Float`     | 1.0        |
//! | `get_precision` | `(sensor_id: String)`          | `Float`     | 1.0        |
//! | `get_range`     | `(sensor_id: String)`          | `(Float, Float)` | 1.0   |
//! | `is_available`  | `(sensor_id: String)`          | `Bool`      | 1.0        |
//! | `batch_read`    | `(sensor_ids: Array)`          | `Array`     | 0.95       |
//! | `read_averaged` | `(sensor_id: String, n: Int)`  | `Float`     | 0.98       |
//!
//! # Epistemic Impact
//!
//! Sensor readings inherently carry uncertainty due to:
//! - Measurement noise and precision limits
//! - Environmental factors affecting accuracy
//! - Calibration drift over time
//! - Quantization errors in analog-to-digital conversion
//!
//! The default confidence factor of 0.95 (5% degradation) reflects typical
//! sensor uncertainty. Averaged readings have higher confidence (0.98) due
//! to noise reduction through averaging.
//!
//! # Supported Sensor Types
//!
//! The handler recognizes common sensor type prefixes:
//! - `temp_*`: Temperature sensors (°C, °F, K)
//! - `pressure_*`: Pressure sensors (Pa, bar, psi)
//! - `humidity_*`: Humidity sensors (% RH)
//! - `accel_*`: Accelerometers (m/s², g)
//! - `gyro_*`: Gyroscopes (rad/s, °/s)
//! - `mag_*`: Magnetometers (T, G)
//! - `light_*`: Light sensors (lux, cd)
//! - `distance_*`: Distance sensors (m, cm, mm)
//! - `voltage_*`: Voltage sensors (V, mV)
//! - `current_*`: Current sensors (A, mA)
//!
//! # Example
//!
//! ```text
//! // Sounio code using Sensor effect:
//! fn read_temperature() -> Knowledge<f64> with Sensor {
//!     let temp = read("temp_ambient")
//!     temp  // Automatically carries 0.95 confidence
//! }
//!
//! fn precise_measurement() -> f64 with Sensor {
//!     read_averaged("pressure_main", 10)  // Average 10 readings
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::interp::Value;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;

/// Confidence factor for single sensor readings (5% degradation)
const SENSOR_READ_CONFIDENCE_FACTOR: f64 = 0.95;

/// Confidence factor for averaged readings (2% degradation)
const SENSOR_AVERAGED_CONFIDENCE_FACTOR: f64 = 0.98;

/// Key for storing sensor calibration data
const SENSOR_CALIBRATION_KEY: &str = "__sensor_calibration";

/// Key for storing simulated sensor values
const SENSOR_VALUES_KEY: &str = "__sensor_values";

/// Key for storing sensor availability
const SENSOR_AVAILABLE_KEY: &str = "__sensor_available";

/// Key for storing RNG seed for sensor noise
const SENSOR_RNG_KEY: &str = "__sensor_rng_seed";

/// Key for storing total readings count
const SENSOR_READINGS_COUNT_KEY: &str = "__sensor_readings_count";

/// Static operation specifications for SensorHandler
static SENSOR_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_sensor_operations() -> &'static [OperationSpec] {
    SENSOR_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("read", "Float")
                .with_params(vec!["String"])
                .with_confidence_factor(SENSOR_READ_CONFIDENCE_FACTOR),
            OperationSpec::new("read_with_unit", "Measurement")
                .with_params(vec!["String"])
                .with_confidence_factor(SENSOR_READ_CONFIDENCE_FACTOR),
            OperationSpec::new("calibrate", "Unit").with_params(vec!["String", "Float"]),
            OperationSpec::new("get_accuracy", "Float").with_params(vec!["String"]),
            OperationSpec::new("get_precision", "Float").with_params(vec!["String"]),
            OperationSpec::new("get_range", "Tuple").with_params(vec!["String"]),
            OperationSpec::new("is_available", "Bool").with_params(vec!["String"]),
            OperationSpec::new("batch_read", "Array")
                .with_params(vec!["Array"])
                .with_confidence_factor(SENSOR_READ_CONFIDENCE_FACTOR),
            OperationSpec::new("read_averaged", "Float")
                .with_params(vec!["String", "Int"])
                .with_confidence_factor(SENSOR_AVERAGED_CONFIDENCE_FACTOR),
            OperationSpec::new("set_simulated_value", "Unit").with_params(vec!["String", "Float"]),
            OperationSpec::new("get_reading_count", "Int").with_params(vec![]),
        ]
    })
}

/// Sensor metadata for simulation
#[derive(Debug, Clone)]
struct SensorMetadata {
    /// Base value for simulation
    base_value: f64,
    /// Noise amplitude (±)
    noise: f64,
    /// Unit of measurement
    unit: String,
    /// Minimum range
    min_range: f64,
    /// Maximum range
    max_range: f64,
    /// Accuracy (as fraction, e.g., 0.01 = 1%)
    accuracy: f64,
    /// Precision (smallest detectable change)
    precision: f64,
}

impl SensorMetadata {
    fn for_sensor_type(sensor_id: &str) -> Self {
        // Determine sensor type from prefix
        if sensor_id.starts_with("temp_") {
            Self {
                base_value: 22.0,
                noise: 0.5,
                unit: "°C".to_string(),
                min_range: -40.0,
                max_range: 125.0,
                accuracy: 0.005,
                precision: 0.1,
            }
        } else if sensor_id.starts_with("pressure_") {
            Self {
                base_value: 101325.0,
                noise: 100.0,
                unit: "Pa".to_string(),
                min_range: 30000.0,
                max_range: 110000.0,
                accuracy: 0.01,
                precision: 1.0,
            }
        } else if sensor_id.starts_with("humidity_") {
            Self {
                base_value: 50.0,
                noise: 2.0,
                unit: "%RH".to_string(),
                min_range: 0.0,
                max_range: 100.0,
                accuracy: 0.03,
                precision: 0.5,
            }
        } else if sensor_id.starts_with("accel_") {
            Self {
                base_value: 0.0,
                noise: 0.01,
                unit: "m/s²".to_string(),
                min_range: -16.0 * 9.81,
                max_range: 16.0 * 9.81,
                accuracy: 0.01,
                precision: 0.001,
            }
        } else if sensor_id.starts_with("gyro_") {
            Self {
                base_value: 0.0,
                noise: 0.001,
                unit: "rad/s".to_string(),
                min_range: -34.9,
                max_range: 34.9,
                accuracy: 0.01,
                precision: 0.0001,
            }
        } else if sensor_id.starts_with("light_") {
            Self {
                base_value: 500.0,
                noise: 10.0,
                unit: "lux".to_string(),
                min_range: 0.0,
                max_range: 100000.0,
                accuracy: 0.05,
                precision: 1.0,
            }
        } else if sensor_id.starts_with("distance_") {
            Self {
                base_value: 1.0,
                noise: 0.01,
                unit: "m".to_string(),
                min_range: 0.02,
                max_range: 4.0,
                accuracy: 0.02,
                precision: 0.001,
            }
        } else if sensor_id.starts_with("voltage_") {
            Self {
                base_value: 3.3,
                noise: 0.01,
                unit: "V".to_string(),
                min_range: 0.0,
                max_range: 5.0,
                accuracy: 0.005,
                precision: 0.001,
            }
        } else if sensor_id.starts_with("current_") {
            Self {
                base_value: 0.1,
                noise: 0.001,
                unit: "A".to_string(),
                min_range: 0.0,
                max_range: 10.0,
                accuracy: 0.01,
                precision: 0.0001,
            }
        } else {
            // Generic sensor
            Self {
                base_value: 0.0,
                noise: 0.1,
                unit: "units".to_string(),
                min_range: -1000.0,
                max_range: 1000.0,
                accuracy: 0.01,
                precision: 0.01,
            }
        }
    }
}

/// Handler for the Sensor effect
///
/// SensorHandler provides sensor and measurement operations through algebraic
/// effects. It simulates various sensor types with realistic noise and
/// uncertainty characteristics for testing and interpretation.
///
/// # State Tracking
///
/// The handler tracks:
/// - Calibration offsets per sensor
/// - Simulated sensor values (for testing)
/// - Sensor availability status
/// - Total reading count
///
/// # Real Implementation
///
/// In a real embedded/IoT context, this handler would delegate to actual
/// hardware interfaces (I2C, SPI, ADC). The current implementation provides
/// simulation with configurable noise for testing epistemic propagation.
#[derive(Debug)]
pub struct SensorHandler {
    /// Whether to add simulated noise to readings
    add_noise: bool,
}

impl Default for SensorHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl SensorHandler {
    /// Create a new SensorHandler with noise enabled
    pub fn new() -> Self {
        Self { add_noise: true }
    }

    /// Create a SensorHandler without noise (deterministic, for testing)
    pub fn without_noise() -> Self {
        Self { add_noise: false }
    }

    /// Get or initialize RNG seed
    fn get_rng_seed(state: &mut HandlerState) -> u64 {
        match state.named_state.get(SENSOR_RNG_KEY) {
            Some(Value::Int(seed)) => *seed as u64,
            _ => {
                let seed = 54321u64;
                state
                    .named_state
                    .insert(SENSOR_RNG_KEY.to_string(), Value::Int(seed as i64));
                seed
            }
        }
    }

    /// Advance RNG and return new seed
    fn next_rng(state: &mut HandlerState) -> u64 {
        let seed = Self::get_rng_seed(state);
        // LCG parameters
        let new_seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        state
            .named_state
            .insert(SENSOR_RNG_KEY.to_string(), Value::Int(new_seed as i64));
        new_seed
    }

    /// Generate uniform random in [-1, 1]
    fn random_noise(state: &mut HandlerState) -> f64 {
        let seed = Self::next_rng(state);
        let normalized = ((seed >> 16) & 0x7FFFFFFF) as f64 / 0x40000000u64 as f64;
        normalized - 1.0
    }

    /// Get calibration offset for a sensor
    fn get_calibration(state: &HandlerState, sensor_id: &str) -> f64 {
        let key = format!("{}_{}", SENSOR_CALIBRATION_KEY, sensor_id);
        match state.named_state.get(&key) {
            Some(Value::Float(offset)) => *offset,
            _ => 0.0,
        }
    }

    /// Set calibration offset for a sensor
    fn set_calibration(state: &mut HandlerState, sensor_id: &str, offset: f64) {
        let key = format!("{}_{}", SENSOR_CALIBRATION_KEY, sensor_id);
        state.named_state.insert(key, Value::Float(offset));
    }

    /// Get simulated value for a sensor (if set)
    fn get_simulated_value(state: &HandlerState, sensor_id: &str) -> Option<f64> {
        let key = format!("{}_{}", SENSOR_VALUES_KEY, sensor_id);
        match state.named_state.get(&key) {
            Some(Value::Float(v)) => Some(*v),
            _ => None,
        }
    }

    /// Check if sensor is available
    fn is_sensor_available(state: &HandlerState, sensor_id: &str) -> bool {
        let key = format!("{}_{}", SENSOR_AVAILABLE_KEY, sensor_id);
        match state.named_state.get(&key) {
            Some(Value::Bool(b)) => *b,
            _ => true, // Default to available
        }
    }

    /// Increment reading count
    fn increment_reading_count(state: &mut HandlerState) {
        let count = match state.named_state.get(SENSOR_READINGS_COUNT_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        state
            .named_state
            .insert(SENSOR_READINGS_COUNT_KEY.to_string(), Value::Int(count + 1));
    }

    /// Extract a string from a Value
    fn extract_string(value: &Value, op: &str, param: &str) -> Result<String, HandlerResult> {
        match value {
            Value::String(s) => Ok(s.clone()),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "Sensor",
                op,
                format!("expected String for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Extract a float from a Value
    fn extract_float(value: &Value, op: &str, param: &str) -> Result<f64, HandlerResult> {
        match value {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "Sensor",
                op,
                format!("expected Float for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Extract an integer from a Value
    fn extract_int(value: &Value, op: &str, param: &str) -> Result<i64, HandlerResult> {
        match value {
            Value::Int(i) => Ok(*i),
            _ => Err(HandlerResult::Abort(HandlerError::new(
                "Sensor",
                op,
                format!("expected Int for {}, got {:?}", param, value.type_name()),
            ))),
        }
    }

    /// Perform a sensor reading with noise
    fn do_reading(&self, sensor_id: &str, state: &mut HandlerState) -> Result<f64, HandlerResult> {
        if !Self::is_sensor_available(state, sensor_id) {
            return Err(HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "read",
                format!("sensor '{}' is not available", sensor_id),
            )));
        }

        let meta = SensorMetadata::for_sensor_type(sensor_id);

        // Use simulated value if set, otherwise use base value
        let base = Self::get_simulated_value(state, sensor_id).unwrap_or(meta.base_value);

        // Apply calibration offset
        let calibration = Self::get_calibration(state, sensor_id);

        // Add noise if enabled
        let noise = if self.add_noise {
            Self::random_noise(state) * meta.noise
        } else {
            0.0
        };

        // Clamp to sensor range
        let value = (base + calibration + noise).clamp(meta.min_range, meta.max_range);

        Self::increment_reading_count(state);

        Ok(value)
    }

    /// Handle read operation
    fn handle_read(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "read",
                "read requires a sensor_id argument",
            ));
        }

        let sensor_id = match Self::extract_string(&args[0], "read", "sensor_id") {
            Ok(s) => s,
            Err(result) => return result,
        };

        match self.do_reading(&sensor_id, state) {
            Ok(value) => HandlerResult::Resume(Value::Float(value)),
            Err(result) => result,
        }
    }

    /// Handle read_with_unit operation
    fn handle_read_with_unit(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "read_with_unit",
                "read_with_unit requires a sensor_id argument",
            ));
        }

        let sensor_id = match Self::extract_string(&args[0], "read_with_unit", "sensor_id") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let meta = SensorMetadata::for_sensor_type(&sensor_id);

        match self.do_reading(&sensor_id, state) {
            Ok(value) => {
                // Return as tuple (value, unit, uncertainty)
                let uncertainty = meta.noise * 2.0; // 2-sigma uncertainty
                HandlerResult::Resume(Value::Tuple(vec![
                    Value::Float(value),
                    Value::String(meta.unit),
                    Value::Float(uncertainty),
                ]))
            }
            Err(result) => result,
        }
    }

    /// Handle calibrate operation
    fn handle_calibrate(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "calibrate",
                "calibrate requires sensor_id and reference value arguments",
            ));
        }

        let sensor_id = match Self::extract_string(&args[0], "calibrate", "sensor_id") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let reference = match Self::extract_float(&args[1], "calibrate", "reference") {
            Ok(f) => f,
            Err(result) => return result,
        };

        if !Self::is_sensor_available(state, &sensor_id) {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "calibrate",
                format!("sensor '{}' is not available", sensor_id),
            ));
        }

        let meta = SensorMetadata::for_sensor_type(&sensor_id);
        let current = Self::get_simulated_value(state, &sensor_id).unwrap_or(meta.base_value);
        let offset = reference - current;

        Self::set_calibration(state, &sensor_id, offset);

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle get_accuracy operation
    fn handle_get_accuracy(&self, args: &[Value]) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "get_accuracy",
                "get_accuracy requires a sensor_id argument",
            ));
        }

        let sensor_id = match Self::extract_string(&args[0], "get_accuracy", "sensor_id") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let meta = SensorMetadata::for_sensor_type(&sensor_id);
        HandlerResult::Resume(Value::Float(meta.accuracy))
    }

    /// Handle get_precision operation
    fn handle_get_precision(&self, args: &[Value]) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "get_precision",
                "get_precision requires a sensor_id argument",
            ));
        }

        let sensor_id = match Self::extract_string(&args[0], "get_precision", "sensor_id") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let meta = SensorMetadata::for_sensor_type(&sensor_id);
        HandlerResult::Resume(Value::Float(meta.precision))
    }

    /// Handle get_range operation
    fn handle_get_range(&self, args: &[Value]) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "get_range",
                "get_range requires a sensor_id argument",
            ));
        }

        let sensor_id = match Self::extract_string(&args[0], "get_range", "sensor_id") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let meta = SensorMetadata::for_sensor_type(&sensor_id);
        HandlerResult::Resume(Value::Tuple(vec![
            Value::Float(meta.min_range),
            Value::Float(meta.max_range),
        ]))
    }

    /// Handle is_available operation
    fn handle_is_available(&self, args: &[Value], state: &HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "is_available",
                "is_available requires a sensor_id argument",
            ));
        }

        let sensor_id = match Self::extract_string(&args[0], "is_available", "sensor_id") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let available = Self::is_sensor_available(state, &sensor_id);
        HandlerResult::Resume(Value::Bool(available))
    }

    /// Handle batch_read operation
    fn handle_batch_read(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "batch_read",
                "batch_read requires an array of sensor_ids",
            ));
        }

        let sensor_ids_rc = match &args[0] {
            Value::Array(arr) => arr.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Sensor",
                    "batch_read",
                    format!(
                        "expected Array for sensor_ids, got {:?}",
                        args[0].type_name()
                    ),
                ));
            }
        };

        let sensor_ids = sensor_ids_rc.borrow();
        let mut results = Vec::with_capacity(sensor_ids.len());

        for (i, id_value) in sensor_ids.iter().enumerate() {
            let sensor_id = match id_value {
                Value::String(s) => s.clone(),
                _ => {
                    return HandlerResult::Abort(HandlerError::new(
                        "Sensor",
                        "batch_read",
                        format!(
                            "expected String at index {}, got {:?}",
                            i,
                            id_value.type_name()
                        ),
                    ));
                }
            };

            match self.do_reading(&sensor_id, state) {
                Ok(value) => results.push(Value::Float(value)),
                Err(result) => return result,
            }
        }

        HandlerResult::Resume(Value::Array(Rc::new(RefCell::new(results))))
    }

    /// Handle read_averaged operation
    fn handle_read_averaged(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "read_averaged",
                "read_averaged requires sensor_id and count arguments",
            ));
        }

        let sensor_id = match Self::extract_string(&args[0], "read_averaged", "sensor_id") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let count = match Self::extract_int(&args[1], "read_averaged", "count") {
            Ok(n) => n,
            Err(result) => return result,
        };

        if count <= 0 {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "read_averaged",
                format!("count must be positive, got {}", count),
            ));
        }

        let mut sum = 0.0;
        for _ in 0..count {
            match self.do_reading(&sensor_id, state) {
                Ok(value) => sum += value,
                Err(result) => return result,
            }
        }

        let average = sum / count as f64;
        HandlerResult::Resume(Value::Float(average))
    }

    /// Handle set_simulated_value operation
    fn handle_set_simulated_value(
        &self,
        args: &[Value],
        state: &mut HandlerState,
    ) -> HandlerResult {
        if args.len() < 2 {
            return HandlerResult::Abort(HandlerError::new(
                "Sensor",
                "set_simulated_value",
                "set_simulated_value requires sensor_id and value arguments",
            ));
        }

        let sensor_id = match Self::extract_string(&args[0], "set_simulated_value", "sensor_id") {
            Ok(s) => s,
            Err(result) => return result,
        };

        let value = match Self::extract_float(&args[1], "set_simulated_value", "value") {
            Ok(f) => f,
            Err(result) => return result,
        };

        let key = format!("{}_{}", SENSOR_VALUES_KEY, sensor_id);
        state.named_state.insert(key, Value::Float(value));

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle get_reading_count operation
    fn handle_get_reading_count(&self, state: &HandlerState) -> HandlerResult {
        let count = match state.named_state.get(SENSOR_READINGS_COUNT_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        HandlerResult::Resume(Value::Int(count))
    }

    /// Set sensor availability (for testing)
    pub fn set_sensor_available(state: &mut HandlerState, sensor_id: &str, available: bool) {
        let key = format!("{}_{}", SENSOR_AVAILABLE_KEY, sensor_id);
        state.named_state.insert(key, Value::Bool(available));
    }

    /// Get total reading count (for testing)
    pub fn reading_count(state: &HandlerState) -> i64 {
        match state.named_state.get(SENSOR_READINGS_COUNT_KEY) {
            Some(Value::Int(n)) => *n,
            _ => 0,
        }
    }
}

impl HandlerCapability for SensorHandler {
    fn effect_name(&self) -> &str {
        "Sensor"
    }

    fn handler_name(&self) -> &str {
        "SensorHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_sensor_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match op {
            "read" => self.handle_read(args, state),
            "read_with_unit" => self.handle_read_with_unit(args, state),
            "calibrate" => self.handle_calibrate(args, state),
            "get_accuracy" => self.handle_get_accuracy(args),
            "get_precision" => self.handle_get_precision(args),
            "get_range" => self.handle_get_range(args),
            "is_available" => self.handle_is_available(args, state),
            "batch_read" => self.handle_batch_read(args, state),
            "read_averaged" => self.handle_read_averaged(args, state),
            "set_simulated_value" => self.handle_set_simulated_value(args, state),
            "get_reading_count" => self.handle_get_reading_count(state),
            _ => HandlerResult::Abort(HandlerError::new(
                "Sensor",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        // Sensor readings are not repeatable (different noise each time)
        false
    }

    fn epistemic_impact(&self, operation: &str) -> EpistemicImpact {
        match operation {
            "read" | "read_with_unit" | "batch_read" => {
                EpistemicImpact::with_confidence(SENSOR_READ_CONFIDENCE_FACTOR)
            }
            "read_averaged" => EpistemicImpact::with_confidence(SENSOR_AVERAGED_CONFIDENCE_FACTOR),
            _ => EpistemicImpact::none(),
        }
    }

    fn operation_linearity(&self, operation: &str) -> crate::effects::linearity::Linearity {
        use crate::effects::linearity::Linearity;
        // Sensor operations read from external world - mostly one-shot
        // (repeated reads would see different noise)
        match operation {
            "read"
            | "read_with_unit"
            | "batch_read"
            | "read_averaged"
            | "calibrate"
            | "set_simulated_value" => Linearity::ExactlyOnce,
            // Metadata queries are safe for multi-shot
            "get_accuracy" | "get_precision" | "get_range" | "is_available"
            | "get_reading_count" => Linearity::MultiShot,
            _ => Linearity::ExactlyOnce,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_state() -> HandlerState {
        HandlerState::new()
    }

    fn cont() -> Continuation {
        Continuation::new()
    }

    #[test]
    fn test_handler_identity() {
        let handler = SensorHandler::new();
        assert_eq!(handler.effect_name(), "Sensor");
        assert_eq!(handler.handler_name(), "SensorHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = SensorHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"read"));
        assert!(names.contains(&"read_with_unit"));
        assert!(names.contains(&"calibrate"));
        assert!(names.contains(&"get_accuracy"));
        assert!(names.contains(&"get_precision"));
        assert!(names.contains(&"get_range"));
        assert!(names.contains(&"is_available"));
        assert!(names.contains(&"batch_read"));
        assert!(names.contains(&"read_averaged"));
    }

    #[test]
    fn test_no_multi_shot() {
        let handler = SensorHandler::new();
        assert!(!handler.supports_multi_shot());
    }

    #[test]
    fn test_epistemic_impact_read() {
        let handler = SensorHandler::new();
        let impact = handler.epistemic_impact("read");
        assert!((impact.confidence_factor - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_epistemic_impact_averaged() {
        let handler = SensorHandler::new();
        let impact = handler.epistemic_impact("read_averaged");
        assert!((impact.confidence_factor - 0.98).abs() < 0.001);
    }

    #[test]
    fn test_read_temperature() {
        let handler = SensorHandler::without_noise();
        let mut state = new_state();

        let result = handler.handle(
            "read",
            &[Value::String("temp_ambient".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                // Default temp is 22.0
                assert!((v - 22.0).abs() < 1.0);
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }

        assert_eq!(SensorHandler::reading_count(&state), 1);
    }

    #[test]
    fn test_read_with_noise() {
        let handler = SensorHandler::new();
        let mut state = new_state();

        // Read multiple times - should get different values
        let mut values = Vec::new();
        for _ in 0..5 {
            let result = handler.handle(
                "read",
                &[Value::String("temp_ambient".to_string())],
                cont(),
                &mut state,
            );
            if let HandlerResult::Resume(Value::Float(v)) = result {
                values.push(v);
            }
        }

        // Values should be in reasonable range
        for v in &values {
            assert!(*v > 20.0 && *v < 24.0);
        }
    }

    #[test]
    fn test_read_with_unit() {
        let handler = SensorHandler::without_noise();
        let mut state = new_state();

        let result = handler.handle(
            "read_with_unit",
            &[Value::String("temp_ambient".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Tuple(elems)) => {
                assert_eq!(elems.len(), 3);
                match &elems[1] {
                    Value::String(unit) => assert_eq!(unit, "°C"),
                    _ => panic!("expected String unit"),
                }
            }
            other => panic!("expected Resume(Tuple), got {:?}", other),
        }
    }

    #[test]
    fn test_calibrate() {
        let handler = SensorHandler::without_noise();
        let mut state = new_state();

        // Set simulated value
        handler.handle(
            "set_simulated_value",
            &[Value::String("temp_test".to_string()), Value::Float(20.0)],
            cont(),
            &mut state,
        );

        // Calibrate to reference of 25.0
        handler.handle(
            "calibrate",
            &[Value::String("temp_test".to_string()), Value::Float(25.0)],
            cont(),
            &mut state,
        );

        // Read should now return ~25.0
        let result = handler.handle(
            "read",
            &[Value::String("temp_test".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!((v - 25.0).abs() < 0.1);
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_get_accuracy() {
        let handler = SensorHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "get_accuracy",
            &[Value::String("temp_ambient".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(acc)) => {
                assert!(acc > 0.0 && acc < 0.1);
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_get_range() {
        let handler = SensorHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "get_range",
            &[Value::String("temp_ambient".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Tuple(elems)) => {
                assert_eq!(elems.len(), 2);
                match (&elems[0], &elems[1]) {
                    (Value::Float(min), Value::Float(max)) => {
                        assert!(*min < *max);
                    }
                    _ => panic!("expected Float tuple"),
                }
            }
            other => panic!("expected Resume(Tuple), got {:?}", other),
        }
    }

    #[test]
    fn test_is_available() {
        let handler = SensorHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "is_available",
            &[Value::String("temp_test".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Bool(true)) => {}
            other => panic!("expected Resume(Bool(true)), got {:?}", other),
        }

        // Make unavailable
        SensorHandler::set_sensor_available(&mut state, "temp_test", false);

        let result = handler.handle(
            "is_available",
            &[Value::String("temp_test".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Bool(false)) => {}
            other => panic!("expected Resume(Bool(false)), got {:?}", other),
        }
    }

    #[test]
    fn test_read_unavailable() {
        let handler = SensorHandler::new();
        let mut state = new_state();

        SensorHandler::set_sensor_available(&mut state, "temp_broken", false);

        let result = handler.handle(
            "read",
            &[Value::String("temp_broken".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("not available"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_batch_read() {
        let handler = SensorHandler::without_noise();
        let mut state = new_state();

        let sensors = Value::Array(Rc::new(RefCell::new(vec![
            Value::String("temp_1".to_string()),
            Value::String("temp_2".to_string()),
        ])));

        let result = handler.handle("batch_read", &[sensors], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Array(arr)) => {
                assert_eq!(arr.borrow().len(), 2);
            }
            other => panic!("expected Resume(Array), got {:?}", other),
        }

        assert_eq!(SensorHandler::reading_count(&state), 2);
    }

    #[test]
    fn test_read_averaged() {
        let handler = SensorHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "read_averaged",
            &[Value::String("temp_ambient".to_string()), Value::Int(5)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                // Average should be close to base value
                assert!(v > 20.0 && v < 24.0);
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }

        assert_eq!(SensorHandler::reading_count(&state), 5);
    }

    #[test]
    fn test_read_averaged_invalid_count() {
        let handler = SensorHandler::new();
        let mut state = new_state();

        let result = handler.handle(
            "read_averaged",
            &[Value::String("temp_ambient".to_string()), Value::Int(0)],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("positive"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_set_simulated_value() {
        let handler = SensorHandler::without_noise();
        let mut state = new_state();

        handler.handle(
            "set_simulated_value",
            &[Value::String("temp_custom".to_string()), Value::Float(42.0)],
            cont(),
            &mut state,
        );

        let result = handler.handle(
            "read",
            &[Value::String("temp_custom".to_string())],
            cont(),
            &mut state,
        );

        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!((v - 42.0).abs() < 0.1);
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_different_sensor_types() {
        let handler = SensorHandler::without_noise();
        let mut state = new_state();

        // Test pressure sensor
        let result = handler.handle(
            "read",
            &[Value::String("pressure_main".to_string())],
            cont(),
            &mut state,
        );
        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!(v > 100000.0 && v < 102000.0); // ~101325 Pa
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }

        // Test humidity sensor
        let result = handler.handle(
            "read",
            &[Value::String("humidity_room".to_string())],
            cont(),
            &mut state,
        );
        match result {
            HandlerResult::Resume(Value::Float(v)) => {
                assert!(v > 40.0 && v < 60.0); // ~50% RH
            }
            other => panic!("expected Resume(Float), got {:?}", other),
        }
    }

    #[test]
    fn test_unknown_operation() {
        let handler = SensorHandler::new();
        let mut state = new_state();

        let result = handler.handle("unknown_op", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("unknown operation"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_default_impl() {
        let handler = SensorHandler::default();
        assert_eq!(handler.effect_name(), "Sensor");
    }
}
