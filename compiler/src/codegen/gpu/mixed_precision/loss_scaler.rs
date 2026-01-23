//! Dynamic loss scaling for mixed-precision training
//!
//! Prevents gradient underflow in FP16 by scaling the loss before backward pass
//! and unscaling gradients afterward.

/// Result of a loss scale update
#[derive(Clone, Debug, PartialEq)]
pub enum ScaleUpdate {
    /// No change to scale
    NoChange,
    /// Scale increased
    ScaleUp { old: f32, new: f32 },
    /// Scale decreased due to overflow
    ScaleDown { old: f32, new: f32 },
}

/// Dynamic loss scaler state and operations
#[derive(Clone, Debug)]
pub struct LossScaler {
    /// Current loss scale factor
    scale: f32,
    /// Growth factor for scaling up
    growth_factor: f32,
    /// Backoff factor for scaling down on overflow (typically 0.5)
    backoff_factor: f32,
    /// Number of successful steps since last overflow
    successful_steps: u32,
    /// Steps needed before growth
    growth_interval: u32,
    /// Minimum allowed scale
    min_scale: f32,
    /// Maximum allowed scale
    max_scale: f32,
    /// Track overflow state per iteration
    overflow_detected: bool,
}

impl LossScaler {
    /// Create a new loss scaler with default settings
    pub fn new(initial_scale: f32, growth_interval: u32) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            successful_steps: 0,
            growth_interval,
            min_scale: 1.0,
            max_scale: (1u32 << 24) as f32, // 16M max
            overflow_detected: false,
        }
    }

    /// Create with default parameters (scale=32768, interval=2000)
    pub fn default_scaler() -> Self {
        Self::new(32768.0, 2000)
    }

    /// Get current scale factor
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get inverse scale factor (for unscaling gradients)
    pub fn inv_scale(&self) -> f32 {
        1.0 / self.scale
    }

    /// Check if overflow was detected in the last update
    pub fn had_overflow(&self) -> bool {
        self.overflow_detected
    }

    /// Update scale based on overflow detection
    ///
    /// Call this after each backward pass with the result of overflow checking.
    /// Returns the type of scale update that occurred.
    pub fn update(&mut self, has_overflow: bool) -> ScaleUpdate {
        self.overflow_detected = has_overflow;

        if has_overflow {
            // Overflow detected: scale down and reset counter
            let old = self.scale;
            let new_scale = (self.scale * self.backoff_factor).max(self.min_scale);
            self.scale = new_scale;
            self.successful_steps = 0;

            if (old - new_scale).abs() > f32::EPSILON {
                ScaleUpdate::ScaleDown { old, new: new_scale }
            } else {
                ScaleUpdate::NoChange
            }
        } else {
            // No overflow: increment counter, maybe scale up
            self.successful_steps += 1;

            if self.successful_steps >= self.growth_interval {
                self.successful_steps = 0;
                let old = self.scale;
                let new_scale = (self.scale * self.growth_factor).min(self.max_scale);
                self.scale = new_scale;

                if (old - new_scale).abs() > f32::EPSILON {
                    ScaleUpdate::ScaleUp { old, new: new_scale }
                } else {
                    ScaleUpdate::NoChange
                }
            } else {
                ScaleUpdate::NoChange
            }
        }
    }

    /// Reset to initial state
    pub fn reset(&mut self, initial_scale: f32) {
        self.scale = initial_scale;
        self.successful_steps = 0;
        self.overflow_detected = false;
    }
}

impl Default for LossScaler {
    fn default() -> Self {
        Self::default_scaler()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_scaler() {
        let scaler = LossScaler::new(32768.0, 2000);
        assert_eq!(scaler.scale(), 32768.0);
        assert!(!scaler.had_overflow());
    }

    #[test]
    fn test_overflow_backoff() {
        let mut scaler = LossScaler::new(32768.0, 2000);
        let result = scaler.update(true);
        assert!(matches!(result, ScaleUpdate::ScaleDown { .. }));
        assert_eq!(scaler.scale(), 16384.0); // 32768 * 0.5
    }

    #[test]
    fn test_scale_growth() {
        let mut scaler = LossScaler::new(32768.0, 2);
        scaler.update(false); // step 1
        assert_eq!(scaler.scale(), 32768.0);
        let result = scaler.update(false); // step 2 -> growth
        assert!(matches!(result, ScaleUpdate::ScaleUp { .. }));
        assert_eq!(scaler.scale(), 65536.0); // 32768 * 2
    }

    #[test]
    fn test_min_scale_clamp() {
        let mut scaler = LossScaler::new(2.0, 2000);
        scaler.update(true); // -> 1.0
        scaler.update(true); // -> stays at 1.0 (min)
        assert_eq!(scaler.scale(), 1.0);
    }
}
