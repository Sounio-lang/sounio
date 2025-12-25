//! Type-level dimensional analysis
//!
//! Every physical quantity has dimensions in the 7 SI base quantities.
//! Dimensions are encoded as compile-time constants.

use std::fmt;

/// Type-level dimension representation
///
/// Encodes the exponents of the 7 SI base quantities:
/// - M: Mass (kilogram)
/// - L: Length (meter)
/// - T: Time (second)
/// - I: Electric current (ampere)
/// - Θ: Thermodynamic temperature (kelvin)
/// - N: Amount of substance (mole)
/// - J: Luminous intensity (candela)
///
/// Derived dimensions are expressed as products of powers:
/// - Velocity = L T⁻¹
/// - Force = M L T⁻²
/// - Concentration = M L⁻³
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Dimension {
    /// Mass exponent [M]
    pub mass: i8,
    /// Length exponent [L]
    pub length: i8,
    /// Time exponent [T]
    pub time: i8,
    /// Electric current exponent [I]
    pub current: i8,
    /// Temperature exponent [Θ]
    pub temperature: i8,
    /// Amount of substance exponent [N]
    pub amount: i8,
    /// Luminous intensity exponent [J]
    pub luminosity: i8,
}

impl Dimension {
    // ==========================================================================
    // Base Dimensions
    // ==========================================================================

    /// Dimensionless (pure number)
    pub const DIMENSIONLESS: Self = Self::new(0, 0, 0, 0, 0, 0, 0);

    /// Mass [M] - kilogram
    pub const MASS: Self = Self::new(1, 0, 0, 0, 0, 0, 0);

    /// Length [L] - meter
    pub const LENGTH: Self = Self::new(0, 1, 0, 0, 0, 0, 0);

    /// Time [T] - second
    pub const TIME: Self = Self::new(0, 0, 1, 0, 0, 0, 0);

    /// Electric current [I] - ampere
    pub const CURRENT: Self = Self::new(0, 0, 0, 1, 0, 0, 0);

    /// Temperature [Θ] - kelvin
    pub const TEMPERATURE: Self = Self::new(0, 0, 0, 0, 1, 0, 0);

    /// Amount of substance [N] - mole
    pub const AMOUNT: Self = Self::new(0, 0, 0, 0, 0, 1, 0);

    /// Luminous intensity [J] - candela
    pub const LUMINOSITY: Self = Self::new(0, 0, 0, 0, 0, 0, 1);

    // ==========================================================================
    // Common Derived Dimensions
    // ==========================================================================

    /// Area [L²]
    pub const AREA: Self = Self::new(0, 2, 0, 0, 0, 0, 0);

    /// Volume [L³]
    pub const VOLUME: Self = Self::new(0, 3, 0, 0, 0, 0, 0);

    /// Velocity [L T⁻¹]
    pub const VELOCITY: Self = Self::new(0, 1, -1, 0, 0, 0, 0);

    /// Acceleration [L T⁻²]
    pub const ACCELERATION: Self = Self::new(0, 1, -2, 0, 0, 0, 0);

    /// Force [M L T⁻²] - newton
    pub const FORCE: Self = Self::new(1, 1, -2, 0, 0, 0, 0);

    /// Energy [M L² T⁻²] - joule
    pub const ENERGY: Self = Self::new(1, 2, -2, 0, 0, 0, 0);

    /// Power [M L² T⁻³] - watt
    pub const POWER: Self = Self::new(1, 2, -3, 0, 0, 0, 0);

    /// Pressure [M L⁻¹ T⁻²] - pascal
    pub const PRESSURE: Self = Self::new(1, -1, -2, 0, 0, 0, 0);

    /// Frequency [T⁻¹] - hertz
    pub const FREQUENCY: Self = Self::new(0, 0, -1, 0, 0, 0, 0);

    /// Electric charge [I T] - coulomb
    pub const CHARGE: Self = Self::new(0, 0, 1, 1, 0, 0, 0);

    /// Voltage [M L² T⁻³ I⁻¹] - volt
    pub const VOLTAGE: Self = Self::new(1, 2, -3, -1, 0, 0, 0);

    /// Resistance [M L² T⁻³ I⁻²] - ohm
    pub const RESISTANCE: Self = Self::new(1, 2, -3, -2, 0, 0, 0);

    // ==========================================================================
    // Pharmacokinetic Dimensions
    // ==========================================================================

    /// Mass concentration [M L⁻³] - kg/m³, mg/L
    pub const CONCENTRATION: Self = Self::new(1, -3, 0, 0, 0, 0, 0);

    /// Molar concentration [N L⁻³] - mol/L
    pub const MOLAR_CONCENTRATION: Self = Self::new(0, -3, 0, 0, 0, 1, 0);

    /// Rate constant [T⁻¹] - per hour, per second
    pub const RATE_CONSTANT: Self = Self::new(0, 0, -1, 0, 0, 0, 0);

    /// Clearance [L³ T⁻¹] - volume per time
    pub const CLEARANCE: Self = Self::new(0, 3, -1, 0, 0, 0, 0);

    /// Volume per mass [L³ M⁻¹] - for Vd/kg
    pub const SPECIFIC_VOLUME: Self = Self::new(-1, 3, 0, 0, 0, 0, 0);

    /// AUC [M L⁻³ T] - concentration × time
    pub const AUC: Self = Self::new(1, -3, 1, 0, 0, 0, 0);

    /// Dose rate [M T⁻¹] - mass per time
    pub const DOSE_RATE: Self = Self::new(1, 0, -1, 0, 0, 0, 0);

    /// Normalized clearance [L³ T⁻¹ M⁻¹] - clearance per kg
    pub const NORMALIZED_CLEARANCE: Self = Self::new(-1, 3, -1, 0, 0, 0, 0);

    // ==========================================================================
    // Constructor
    // ==========================================================================

    /// Create a new dimension with given exponents
    pub const fn new(
        mass: i8,
        length: i8,
        time: i8,
        current: i8,
        temperature: i8,
        amount: i8,
        luminosity: i8,
    ) -> Self {
        Self {
            mass,
            length,
            time,
            current,
            temperature,
            amount,
            luminosity,
        }
    }

    // ==========================================================================
    // Operations
    // ==========================================================================

    /// Multiply dimensions (add exponents)
    ///
    /// Used when multiplying quantities: [A] × [B] = [A × B]
    pub const fn mul(&self, other: &Dimension) -> Dimension {
        Dimension {
            mass: self.mass + other.mass,
            length: self.length + other.length,
            time: self.time + other.time,
            current: self.current + other.current,
            temperature: self.temperature + other.temperature,
            amount: self.amount + other.amount,
            luminosity: self.luminosity + other.luminosity,
        }
    }

    /// Divide dimensions (subtract exponents)
    ///
    /// Used when dividing quantities: [A] / [B] = [A / B]
    pub const fn div(&self, other: &Dimension) -> Dimension {
        Dimension {
            mass: self.mass - other.mass,
            length: self.length - other.length,
            time: self.time - other.time,
            current: self.current - other.current,
            temperature: self.temperature - other.temperature,
            amount: self.amount - other.amount,
            luminosity: self.luminosity - other.luminosity,
        }
    }

    /// Reciprocal (negate all exponents)
    ///
    /// [1/A] = [A]⁻¹
    pub const fn recip(&self) -> Dimension {
        Dimension {
            mass: -self.mass,
            length: -self.length,
            time: -self.time,
            current: -self.current,
            temperature: -self.temperature,
            amount: -self.amount,
            luminosity: -self.luminosity,
        }
    }

    /// Raise to integer power (multiply all exponents)
    ///
    /// [A]ⁿ
    pub const fn pow(&self, n: i8) -> Dimension {
        Dimension {
            mass: self.mass * n,
            length: self.length * n,
            time: self.time * n,
            current: self.current * n,
            temperature: self.temperature * n,
            amount: self.amount * n,
            luminosity: self.luminosity * n,
        }
    }

    /// Square root (divide exponents by 2)
    ///
    /// Returns None if any exponent is odd (cannot take sqrt)
    pub const fn sqrt(&self) -> Option<Dimension> {
        if self.mass % 2 != 0
            || self.length % 2 != 0
            || self.time % 2 != 0
            || self.current % 2 != 0
            || self.temperature % 2 != 0
            || self.amount % 2 != 0
            || self.luminosity % 2 != 0
        {
            return None;
        }
        Some(Dimension {
            mass: self.mass / 2,
            length: self.length / 2,
            time: self.time / 2,
            current: self.current / 2,
            temperature: self.temperature / 2,
            amount: self.amount / 2,
            luminosity: self.luminosity / 2,
        })
    }

    /// Cube root (divide exponents by 3)
    ///
    /// Returns None if any exponent is not divisible by 3
    pub const fn cbrt(&self) -> Option<Dimension> {
        if self.mass % 3 != 0
            || self.length % 3 != 0
            || self.time % 3 != 0
            || self.current % 3 != 0
            || self.temperature % 3 != 0
            || self.amount % 3 != 0
            || self.luminosity % 3 != 0
        {
            return None;
        }
        Some(Dimension {
            mass: self.mass / 3,
            length: self.length / 3,
            time: self.time / 3,
            current: self.current / 3,
            temperature: self.temperature / 3,
            amount: self.amount / 3,
            luminosity: self.luminosity / 3,
        })
    }

    // ==========================================================================
    // Predicates
    // ==========================================================================

    /// Check if dimensionless
    pub const fn is_dimensionless(&self) -> bool {
        self.mass == 0
            && self.length == 0
            && self.time == 0
            && self.current == 0
            && self.temperature == 0
            && self.amount == 0
            && self.luminosity == 0
    }

    /// Check if dimensions are equal
    pub const fn equals(&self, other: &Dimension) -> bool {
        self.mass == other.mass
            && self.length == other.length
            && self.time == other.time
            && self.current == other.current
            && self.temperature == other.temperature
            && self.amount == other.amount
            && self.luminosity == other.luminosity
    }

    /// Check if this is a pure mass dimension
    pub const fn is_mass(&self) -> bool {
        self.mass != 0
            && self.length == 0
            && self.time == 0
            && self.current == 0
            && self.temperature == 0
            && self.amount == 0
            && self.luminosity == 0
    }

    /// Check if this is a pure length dimension
    pub const fn is_length(&self) -> bool {
        self.mass == 0
            && self.length != 0
            && self.time == 0
            && self.current == 0
            && self.temperature == 0
            && self.amount == 0
            && self.luminosity == 0
    }

    /// Check if this is a pure time dimension
    pub const fn is_time(&self) -> bool {
        self.mass == 0
            && self.length == 0
            && self.time != 0
            && self.current == 0
            && self.temperature == 0
            && self.amount == 0
            && self.luminosity == 0
    }

    /// Check if this is a concentration dimension (M L⁻³)
    pub const fn is_concentration(&self) -> bool {
        self.equals(&Self::CONCENTRATION)
    }

    /// Check if this is a clearance dimension (L³ T⁻¹)
    pub const fn is_clearance(&self) -> bool {
        self.equals(&Self::CLEARANCE)
    }

    // ==========================================================================
    // Named Dimension Detection
    // ==========================================================================

    /// Get the name of this dimension if it matches a known type
    pub fn name(&self) -> Option<&'static str> {
        match *self {
            Self::DIMENSIONLESS => Some("dimensionless"),
            Self::MASS => Some("mass"),
            Self::LENGTH => Some("length"),
            Self::TIME => Some("time"),
            Self::CURRENT => Some("electric current"),
            Self::TEMPERATURE => Some("temperature"),
            Self::AMOUNT => Some("amount of substance"),
            Self::LUMINOSITY => Some("luminous intensity"),
            Self::AREA => Some("area"),
            Self::VOLUME => Some("volume"),
            Self::VELOCITY => Some("velocity"),
            Self::ACCELERATION => Some("acceleration"),
            Self::FORCE => Some("force"),
            Self::ENERGY => Some("energy"),
            Self::POWER => Some("power"),
            Self::PRESSURE => Some("pressure"),
            Self::FREQUENCY => Some("frequency"),
            Self::CONCENTRATION => Some("concentration"),
            Self::MOLAR_CONCENTRATION => Some("molar concentration"),
            Self::CLEARANCE => Some("clearance"),
            Self::AUC => Some("AUC"),
            _ => None,
        }
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "1");
        }

        let parts: Vec<String> = Vec::new();

        // Collect positive exponents (numerator)
        let mut num: Vec<String> = Vec::new();
        let mut den: Vec<String> = Vec::new();

        let add_dim = |parts: &mut Vec<String>, name: &str, exp: i8| {
            if exp == 1 {
                parts.push(name.to_string());
            } else {
                parts.push(format!("{}{}", name, superscript(exp)));
            }
        };

        if self.mass > 0 {
            add_dim(&mut num, "M", self.mass);
        } else if self.mass < 0 {
            add_dim(&mut den, "M", -self.mass);
        }

        if self.length > 0 {
            add_dim(&mut num, "L", self.length);
        } else if self.length < 0 {
            add_dim(&mut den, "L", -self.length);
        }

        if self.time > 0 {
            add_dim(&mut num, "T", self.time);
        } else if self.time < 0 {
            add_dim(&mut den, "T", -self.time);
        }

        if self.current > 0 {
            add_dim(&mut num, "I", self.current);
        } else if self.current < 0 {
            add_dim(&mut den, "I", -self.current);
        }

        if self.temperature > 0 {
            add_dim(&mut num, "Θ", self.temperature);
        } else if self.temperature < 0 {
            add_dim(&mut den, "Θ", -self.temperature);
        }

        if self.amount > 0 {
            add_dim(&mut num, "N", self.amount);
        } else if self.amount < 0 {
            add_dim(&mut den, "N", -self.amount);
        }

        if self.luminosity > 0 {
            add_dim(&mut num, "J", self.luminosity);
        } else if self.luminosity < 0 {
            add_dim(&mut den, "J", -self.luminosity);
        }

        let num_str = if num.is_empty() {
            "1".to_string()
        } else {
            num.join(" ")
        };

        if den.is_empty() {
            write!(f, "{}", num_str)
        } else {
            write!(f, "{} / {}", num_str, den.join(" "))
        }
    }
}

/// Convert integer to superscript string
fn superscript(n: i8) -> String {
    let digits: Vec<char> = n.abs().to_string().chars().collect();
    let mut result = String::new();

    for d in digits {
        result.push(match d {
            '0' => '⁰',
            '1' => '¹',
            '2' => '²',
            '3' => '³',
            '4' => '⁴',
            '5' => '⁵',
            '6' => '⁶',
            '7' => '⁷',
            '8' => '⁸',
            '9' => '⁹',
            _ => d,
        });
    }

    if n < 0 {
        format!("⁻{}", result)
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mul() {
        // Force = Mass × Acceleration = M × L T⁻² = M L T⁻²
        let force = Dimension::MASS.mul(&Dimension::ACCELERATION);
        assert!(force.equals(&Dimension::FORCE));
    }

    #[test]
    fn test_dimension_div() {
        // Velocity = Length / Time = L / T = L T⁻¹
        let velocity = Dimension::LENGTH.div(&Dimension::TIME);
        assert!(velocity.equals(&Dimension::VELOCITY));
    }

    #[test]
    fn test_concentration() {
        // Concentration = Mass / Volume = M / L³ = M L⁻³
        let conc = Dimension::MASS.div(&Dimension::VOLUME);
        assert!(conc.equals(&Dimension::CONCENTRATION));
    }

    #[test]
    fn test_clearance() {
        // Clearance = Volume / Time = L³ / T = L³ T⁻¹
        let cl = Dimension::VOLUME.div(&Dimension::TIME);
        assert!(cl.equals(&Dimension::CLEARANCE));
    }

    #[test]
    fn test_sqrt() {
        // sqrt(Area) = sqrt(L²) = L
        let length = Dimension::AREA.sqrt().unwrap();
        assert!(length.equals(&Dimension::LENGTH));

        // sqrt(L³) should fail (odd exponent)
        assert!(Dimension::VOLUME.sqrt().is_none());
    }

    #[test]
    fn test_power() {
        // L³ = L^3
        let volume = Dimension::LENGTH.pow(3);
        assert!(volume.equals(&Dimension::VOLUME));
    }

    #[test]
    fn test_recip() {
        // 1/T = T⁻¹ = Frequency
        let freq = Dimension::TIME.recip();
        assert!(freq.equals(&Dimension::FREQUENCY));
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Dimension::VELOCITY), "L / T");
        assert_eq!(format!("{}", Dimension::FORCE), "M L / T²");
        assert_eq!(format!("{}", Dimension::DIMENSIONLESS), "1");
    }

    #[test]
    fn test_named() {
        assert_eq!(Dimension::MASS.name(), Some("mass"));
        assert_eq!(Dimension::CONCENTRATION.name(), Some("concentration"));
        assert_eq!(Dimension::CLEARANCE.name(), Some("clearance"));
    }
}
