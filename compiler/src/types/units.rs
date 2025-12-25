//! Units of measure with compile-time dimensional analysis
//!
//! D supports units of measure with compile-time dimensional analysis,
//! preventing errors like adding meters to seconds or using wrong dosage units.

use std::collections::HashMap;

/// Unit expression
#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    /// Base units with exponents: {"m": 1, "s": -2} = m/s²
    pub dimensions: HashMap<String, i32>,
    /// Scale factor (for conversions)
    pub scale: f64,
}

impl Unit {
    /// Create a dimensionless unit (pure number)
    pub fn dimensionless() -> Self {
        Self {
            dimensions: HashMap::new(),
            scale: 1.0,
        }
    }

    /// Create a base unit (e.g., "m", "kg", "s")
    pub fn base(name: &str) -> Self {
        let mut dims = HashMap::new();
        dims.insert(name.to_string(), 1);
        Self {
            dimensions: dims,
            scale: 1.0,
        }
    }

    /// Create a scaled unit (e.g., kilometer = 1000 * meter)
    pub fn scaled(name: &str, scale: f64) -> Self {
        let mut dims = HashMap::new();
        dims.insert(name.to_string(), 1);
        Self {
            dimensions: dims,
            scale,
        }
    }

    /// Multiply two units
    pub fn multiply(&self, other: &Unit) -> Unit {
        let mut dims = self.dimensions.clone();
        for (unit, power) in &other.dimensions {
            *dims.entry(unit.clone()).or_insert(0) += power;
        }
        // Remove zero exponents
        dims.retain(|_, v| *v != 0);
        Unit {
            dimensions: dims,
            scale: self.scale * other.scale,
        }
    }

    /// Divide two units
    pub fn divide(&self, other: &Unit) -> Unit {
        let mut dims = self.dimensions.clone();
        for (unit, power) in &other.dimensions {
            *dims.entry(unit.clone()).or_insert(0) -= power;
        }
        // Remove zero exponents
        dims.retain(|_, v| *v != 0);
        Unit {
            dimensions: dims,
            scale: self.scale / other.scale,
        }
    }

    /// Raise unit to a power
    pub fn power(&self, n: i32) -> Unit {
        let dims: HashMap<_, _> = self
            .dimensions
            .iter()
            .map(|(k, v)| (k.clone(), v * n))
            .filter(|(_, v)| *v != 0)
            .collect();
        Unit {
            dimensions: dims,
            scale: self.scale.powi(n),
        }
    }

    /// Take square root of unit (exponents must be even)
    pub fn sqrt(&self) -> Option<Unit> {
        let mut dims = HashMap::new();
        for (unit, power) in &self.dimensions {
            if power % 2 != 0 {
                return None; // Cannot take sqrt
            }
            dims.insert(unit.clone(), power / 2);
        }
        Some(Unit {
            dimensions: dims,
            scale: self.scale.sqrt(),
        })
    }

    /// Check if two units are dimensionally compatible (same dimensions)
    pub fn is_compatible(&self, other: &Unit) -> bool {
        self.dimensions == other.dimensions
    }

    /// Check if this is a dimensionless unit
    pub fn is_dimensionless(&self) -> bool {
        self.dimensions.is_empty()
    }

    /// Get conversion factor to another compatible unit
    pub fn conversion_factor(&self, other: &Unit) -> Option<f64> {
        if self.is_compatible(other) {
            Some(self.scale / other.scale)
        } else {
            None
        }
    }

    /// Format unit as a string (e.g., "m/s²")
    pub fn format(&self) -> String {
        if self.dimensions.is_empty() {
            return "1".to_string();
        }

        let mut pos = Vec::new();
        let mut neg = Vec::new();

        for (unit, power) in &self.dimensions {
            if *power > 0 {
                pos.push((unit.clone(), *power));
            } else {
                neg.push((unit.clone(), -*power));
            }
        }

        pos.sort_by(|a, b| a.0.cmp(&b.0));
        neg.sort_by(|a, b| a.0.cmp(&b.0));

        let format_part = |parts: &[(String, i32)]| -> String {
            parts
                .iter()
                .map(|(u, p)| {
                    if *p == 1 {
                        u.clone()
                    } else {
                        format!("{}^{}", u, p)
                    }
                })
                .collect::<Vec<_>>()
                .join("*")
        };

        let pos_str = format_part(&pos);
        let neg_str = format_part(&neg);

        if neg.is_empty() {
            pos_str
        } else if pos.is_empty() {
            format!("1/{}", neg_str)
        } else {
            format!("{}/{}", pos_str, neg_str)
        }
    }
}

impl std::fmt::Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}

/// Pre-defined SI base units
pub mod si {
    use super::*;

    pub fn meter() -> Unit {
        Unit::base("m")
    }
    pub fn kilogram() -> Unit {
        Unit::base("kg")
    }
    pub fn second() -> Unit {
        Unit::base("s")
    }
    pub fn ampere() -> Unit {
        Unit::base("A")
    }
    pub fn kelvin() -> Unit {
        Unit::base("K")
    }
    pub fn mole() -> Unit {
        Unit::base("mol")
    }
    pub fn candela() -> Unit {
        Unit::base("cd")
    }

    // Derived units
    pub fn newton() -> Unit {
        kilogram().multiply(&meter()).divide(&second().power(2))
    }

    pub fn joule() -> Unit {
        newton().multiply(&meter())
    }

    pub fn watt() -> Unit {
        joule().divide(&second())
    }

    pub fn pascal() -> Unit {
        newton().divide(&meter().power(2))
    }

    pub fn hertz() -> Unit {
        Unit::dimensionless().divide(&second())
    }

    pub fn coulomb() -> Unit {
        ampere().multiply(&second())
    }

    pub fn volt() -> Unit {
        watt().divide(&ampere())
    }

    pub fn ohm() -> Unit {
        volt().divide(&ampere())
    }
}

/// Pre-defined medical/pharmaceutical units
pub mod medical {
    use super::*;

    // Mass units
    pub fn milligram() -> Unit {
        Unit {
            dimensions: [("kg".into(), 1)].into(),
            scale: 1e-6,
        }
    }

    pub fn microgram() -> Unit {
        Unit {
            dimensions: [("kg".into(), 1)].into(),
            scale: 1e-9,
        }
    }

    pub fn gram() -> Unit {
        Unit {
            dimensions: [("kg".into(), 1)].into(),
            scale: 1e-3,
        }
    }

    // Volume units
    pub fn milliliter() -> Unit {
        Unit {
            dimensions: [("m".into(), 3)].into(),
            scale: 1e-9,
        }
    }

    pub fn liter() -> Unit {
        Unit {
            dimensions: [("m".into(), 3)].into(),
            scale: 1e-3,
        }
    }

    pub fn microliter() -> Unit {
        Unit {
            dimensions: [("m".into(), 3)].into(),
            scale: 1e-12,
        }
    }

    // Time units
    pub fn minute() -> Unit {
        Unit {
            dimensions: [("s".into(), 1)].into(),
            scale: 60.0,
        }
    }

    pub fn hour() -> Unit {
        Unit {
            dimensions: [("s".into(), 1)].into(),
            scale: 3600.0,
        }
    }

    pub fn day() -> Unit {
        Unit {
            dimensions: [("s".into(), 1)].into(),
            scale: 86400.0,
        }
    }

    // Concentration units
    pub fn mg_per_ml() -> Unit {
        milligram().divide(&milliliter())
    }

    pub fn mg_per_l() -> Unit {
        milligram().divide(&liter())
    }

    pub fn mol_per_l() -> Unit {
        Unit::base("mol").divide(&liter())
    }

    // Flow rates
    pub fn ml_per_min() -> Unit {
        milliliter().divide(&minute())
    }

    pub fn ml_per_hour() -> Unit {
        milliliter().divide(&hour())
    }

    pub fn l_per_hour() -> Unit {
        liter().divide(&hour())
    }

    // Dosing units
    pub fn mg_per_kg() -> Unit {
        milligram().divide(&Unit::base("kg"))
    }

    pub fn mg_per_kg_per_day() -> Unit {
        mg_per_kg().divide(&day())
    }

    // Area (for BSA-based dosing)
    pub fn square_meter() -> Unit {
        Unit::base("m").power(2)
    }

    pub fn mg_per_m2() -> Unit {
        milligram().divide(&square_meter())
    }
}

/// Unit checker for type checking
#[derive(Debug)]
pub struct UnitChecker {
    /// Known unit aliases
    aliases: HashMap<String, Unit>,
}

impl UnitChecker {
    pub fn new() -> Self {
        let mut aliases = HashMap::new();

        // Register common aliases
        // Mass
        aliases.insert("kg".into(), si::kilogram());
        aliases.insert("g".into(), medical::gram());
        aliases.insert("mg".into(), medical::milligram());
        aliases.insert("ug".into(), medical::microgram());
        aliases.insert("mcg".into(), medical::microgram());

        // Volume
        aliases.insert("L".into(), medical::liter());
        aliases.insert("mL".into(), medical::milliliter());
        aliases.insert("uL".into(), medical::microliter());

        // Time
        aliases.insert("s".into(), si::second());
        aliases.insert("min".into(), medical::minute());
        aliases.insert("h".into(), medical::hour());
        aliases.insert("hr".into(), medical::hour());
        aliases.insert("hours".into(), medical::hour());
        aliases.insert("day".into(), medical::day());
        aliases.insert("d".into(), medical::day());

        // Length
        aliases.insert("m".into(), si::meter());
        aliases.insert("m2".into(), medical::square_meter());

        // Concentrations
        aliases.insert("mg/mL".into(), medical::mg_per_ml());
        aliases.insert("mg/L".into(), medical::mg_per_l());
        aliases.insert("mol/L".into(), medical::mol_per_l());
        aliases.insert("M".into(), medical::mol_per_l());

        // Flow rates
        aliases.insert("mL/min".into(), medical::ml_per_min());
        aliases.insert("mL/h".into(), medical::ml_per_hour());
        aliases.insert("L/h".into(), medical::l_per_hour());

        // Dosing
        aliases.insert("mg/kg".into(), medical::mg_per_kg());
        aliases.insert("mg/kg/day".into(), medical::mg_per_kg_per_day());
        aliases.insert("mg/m2".into(), medical::mg_per_m2());

        Self { aliases }
    }

    /// Register a new unit alias
    pub fn register(&mut self, name: &str, unit: Unit) {
        self.aliases.insert(name.to_string(), unit);
    }

    /// Look up a unit by name
    pub fn lookup(&self, name: &str) -> Option<&Unit> {
        self.aliases.get(name)
    }

    /// Parse a unit expression (simple parsing)
    pub fn parse(&self, expr: &str) -> Option<Unit> {
        // Try direct lookup first
        if let Some(unit) = self.lookup(expr) {
            return Some(unit.clone());
        }

        // Try parsing compound units like "mg/mL"
        if let Some(pos) = expr.find('/') {
            let num = &expr[..pos];
            let den = &expr[pos + 1..];
            let num_unit = self.parse(num)?;
            let den_unit = self.parse(den)?;
            return Some(num_unit.divide(&den_unit));
        }

        // Try parsing products like "kg*m"
        if let Some(pos) = expr.find('*') {
            let left = &expr[..pos];
            let right = &expr[pos + 1..];
            let left_unit = self.parse(left)?;
            let right_unit = self.parse(right)?;
            return Some(left_unit.multiply(&right_unit));
        }

        // Try parsing powers like "m^2"
        if let Some(pos) = expr.find('^') {
            let base = &expr[..pos];
            let exp: i32 = expr[pos + 1..].parse().ok()?;
            let base_unit = self.parse(base)?;
            return Some(base_unit.power(exp));
        }

        None
    }

    /// Check if two units are compatible for an operation
    pub fn check_compatible(&self, u1: &Unit, u2: &Unit) -> bool {
        u1.is_compatible(u2)
    }

    /// Get the result unit of a binary operation
    pub fn binary_result(&self, op: UnitOp, u1: &Unit, u2: &Unit) -> Option<Unit> {
        match op {
            UnitOp::Add | UnitOp::Sub => {
                if u1.is_compatible(u2) {
                    Some(u1.clone())
                } else {
                    None
                }
            }
            UnitOp::Mul => Some(u1.multiply(u2)),
            UnitOp::Div => Some(u1.divide(u2)),
        }
    }
}

impl Default for UnitChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Unit operations
#[derive(Debug, Clone, Copy)]
pub enum UnitOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Unit error
#[derive(Debug, Clone)]
pub enum UnitError {
    Incompatible { expected: Unit, found: Unit },
    UnknownUnit(String),
    InvalidOperation { op: String, units: String },
}

impl std::fmt::Display for UnitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnitError::Incompatible { expected, found } => {
                write!(
                    f,
                    "Unit mismatch: expected {}, found {}",
                    expected.format(),
                    found.format()
                )
            }
            UnitError::UnknownUnit(name) => write!(f, "Unknown unit: {}", name),
            UnitError::InvalidOperation { op, units } => {
                write!(f, "Invalid operation {} on units {}", op, units)
            }
        }
    }
}

impl std::error::Error for UnitError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_multiply() {
        let m = si::meter();
        let s = si::second();
        let velocity = m.divide(&s);

        assert_eq!(velocity.dimensions.get("m"), Some(&1));
        assert_eq!(velocity.dimensions.get("s"), Some(&-1));
    }

    #[test]
    fn test_unit_power() {
        let m = si::meter();
        let m2 = m.power(2);

        assert_eq!(m2.dimensions.get("m"), Some(&2));
    }

    #[test]
    fn test_unit_compatible() {
        let mg = medical::milligram();
        let g = medical::gram();

        // Both are mass units
        assert!(mg.is_compatible(&g));

        let ml = medical::milliliter();
        // Mass and volume are not compatible
        assert!(!mg.is_compatible(&ml));
    }

    #[test]
    fn test_unit_conversion() {
        let mg = medical::milligram();
        let g = medical::gram();

        let factor = mg.conversion_factor(&g).unwrap();
        // 1 mg = 0.001 g, so factor should be ~0.001
        assert!((factor - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_unit_checker() {
        let checker = UnitChecker::new();

        let mg = checker.lookup("mg").unwrap();
        let ml = checker.lookup("mL").unwrap();

        // mg/mL should parse correctly
        let conc = checker.parse("mg/mL").unwrap();
        assert!(conc.is_compatible(&medical::mg_per_ml()));
    }

    #[test]
    fn test_medical_units() {
        let dose = medical::mg_per_kg();
        let weight = si::kilogram();
        let total_dose = dose.multiply(&weight);

        // mg/kg * kg = mg
        assert!(total_dose.is_compatible(&medical::milligram()));
    }

    #[test]
    fn test_unit_format() {
        let velocity = si::meter().divide(&si::second());
        assert_eq!(velocity.format(), "m/s");

        let acceleration = velocity.divide(&si::second());
        assert_eq!(acceleration.format(), "m/s^2");
    }
}
