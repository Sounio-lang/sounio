/// Wave 1 Example: Uncertainty Propagation
///
/// Demonstrates working with measurements that have uncertainty bounds.
/// Shows how uncertainty propagates through calculations.
///
/// This example shows:
/// - Creating uncertain values
/// - Arithmetic with uncertainty
/// - Analyzing measurement error

fn main() {
    println("=== Uncertainty Propagation ===");

    // Measure a resistor
    let resistance = Uncertain { mean: 100.0, std: 2.5 };  // 100 ± 2.5 Ω
    let voltage = Uncertain { mean: 5.0, std: 0.1 };       // 5 ± 0.1 V

    println!("Resistance: {} ± {} Ω", resistance.mean, resistance.std);
    println!("Voltage: {} ± {} V", voltage.mean, voltage.std);

    // Ohm's law: I = V / R
    let current = divide_uncertain(voltage, resistance);

    println!("\nOhm's Law: I = V / R");
    println!("Current: {:.4} ± {:.4} A", current.mean, current.std);

    // Power dissipation: P = V * I = V² / R
    let power = multiply_uncertain(voltage, current);

    println!("\nPower dissipation: P = V × I");
    println!("Power: {:.3} ± {:.3} W", power.mean, power.std);

    // Relative uncertainty
    let voltage_rel_unc = voltage.std / voltage.mean;
    let resistance_rel_unc = resistance.std / resistance.mean;
    let current_rel_unc = current.std / current.mean;

    println!("\n=== Relative Uncertainty ===");
    println!("Voltage uncertainty: {:.2}%", voltage_rel_unc * 100.0);
    println!("Resistance uncertainty: {:.2}%", resistance_rel_unc * 100.0);
    println!("Current uncertainty: {:.2}%", current_rel_unc * 100.0);

    // Temperature measurement
    println!("\n=== Temperature Measurement ===");
    let temp_celsius = Uncertain { mean: 25.0, std: 0.5 };
    println!("Temperature: {} ± {} °C", temp_celsius.mean, temp_celsius.std);

    // Convert to Fahrenheit: F = 9/5 * C + 32
    let temp_fahrenheit = scale_and_shift_uncertain(temp_celsius, 9.0/5.0, 32.0);
    println!("Temperature: {:.1} ± {:.2} °F", temp_fahrenheit.mean, temp_fahrenheit.std);

    // Measurement reproducibility
    println!("\n=== Multiple Measurements ===");
    let measurements = [24.8, 25.1, 25.0, 24.9, 25.2];
    let mean_val = compute_mean(measurements);
    let std_val = compute_std(measurements, mean_val);

    println!("Measurements: {} values", len(measurements));
    println!("Mean: {:.2} °C", mean_val);
    println!("Std Dev: {:.4} °C", std_val);
    println!("95% CI: [{:.2}, {:.2}] °C",
             mean_val - 1.96*std_val,
             mean_val + 1.96*std_val);
}

struct Uncertain {
    mean: f64,
    std: f64,
}

fn multiply_uncertain(a: Uncertain, b: Uncertain) -> Uncertain {
    // P(A×B) ≈ (A×B) ± sqrt((B×σ_A)² + (A×σ_B)²)
    let mean_product = a.mean * b.mean;
    let std_product = sqrt(
        (b.mean * a.std)*(b.mean * a.std) +
        (a.mean * b.std)*(a.mean * b.std)
    );

    return Uncertain { mean: mean_product, std: std_product };
}

fn divide_uncertain(a: Uncertain, b: Uncertain) -> Uncertain {
    // P(A/B) ≈ (A/B) ± (A/B) × sqrt((σ_A/A)² + (σ_B/B)²)
    let mean_quotient = a.mean / b.mean;
    let rel_std = sqrt(
        (a.std/a.mean)*(a.std/a.mean) +
        (b.std/b.mean)*(b.std/b.mean)
    );
    let std_quotient = mean_quotient * rel_std;

    return Uncertain { mean: mean_quotient, std: std_quotient };
}

fn scale_and_shift_uncertain(u: Uncertain, scale: f64, shift: f64) -> Uncertain {
    // For linear transformation: Y = scale*X + shift
    // Mean transforms: E[Y] = scale*E[X] + shift
    // Std dev scales: σ_Y = |scale| * σ_X
    return Uncertain {
        mean: scale * u.mean + shift,
        std: scale * u.std  // Assuming scale > 0
    };
}

fn compute_mean(values: [f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..len(values) {
        sum = sum + values[i];
    }
    return sum / (len(values) as f64);
}

fn compute_std(values: [f64], mean: f64) -> f64 {
    let mut sum_sq_dev = 0.0;
    for i in 0..len(values) {
        let dev = values[i] - mean;
        sum_sq_dev = sum_sq_dev + dev * dev;
    }
    let variance = sum_sq_dev / (len(values) as f64);
    return sqrt(variance);
}

fn sqrt(x: f64) -> f64 {
    if x == 0.0 { return 0.0; }
    let mut guess = x;
    for _ in 0..20 {
        guess = (guess + x / guess) / 2.0;
    }
    return guess;
}
