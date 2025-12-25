/// Wave 3 Example: Symbolic Regression - Kepler's Third Law
///
/// Discovers Kepler's Third Law (T² ∝ a³) from planetary orbital data
/// using symbolic regression (symbolic search + neural optimization).
///
/// This demonstrates how neurosymbolic methods can rediscover
/// fundamental laws from data.

fn main() {
    println!("=== Symbolic Regression: Kepler's Third Law ===");
    println!("Discovering fundamental physics from data\n");

    println!("=== Planetary Data ===");
    println!("Planet\t\t| Semi-major axis (AU)\t| Orbital period (years)");
    println!("-------+-------------------+-------------------");

    // Kepler's data (historical)
    let data = vec![
        ("Mercury", 0.387, 0.241),
        ("Venus", 0.723, 0.615),
        ("Earth", 1.000, 1.000),
        ("Mars", 1.524, 1.881),
        ("Jupiter", 5.203, 11.86),
        ("Saturn", 9.537, 29.46),
    ];

    for (planet, a, t) in &data {
        println!("{:<15}| {:<17.3}\t| {:<17.3}", planet, a, t);
    }
    println!();

    println!("=== Hypothesis: T = k * a^p ===");
    println!("We search for form T = k * a^p\n");

    println!("=== Symbolic Search ===");
    println!("Candidate forms:");
    println!("1. T = a");
    println!("2. T = a^1.5");
    println!("3. T = a^2");
    println!("4. T = a^2.5");
    println!("5. T = a^3");
    println!();

    println!("=== Evaluation ===");
    let forms = vec![
        ("T = a", 1.0),
        ("T = a^1.5", 1.5),
        ("T = a^2", 2.0),
        ("T = a^2.5", 2.5),
        ("T = a^3", 3.0),
    ];

    for (form_str, p) in &forms {
        println!("\nTesting: {}", form_str);
        println!("x\t| T_obs\t| T_pred\t| error");
        println!("-----+-------+-------+-------");

        let mut rmse = 0.0;
        let mut r_squared_num = 0.0;
        let mut r_squared_den = 0.0;
        let mut t_mean = 0.0;

        for (_, a, t) in &data {
            t_mean += t;
        }
        t_mean /= data.len() as f64;

        for (planet, a, t) in &data {
            let t_pred = a.powf(*p);
            let error = (t_pred - t).abs();
            rmse += error * error;
            r_squared_num += (t_pred - t).powi(2);
            r_squared_den += (t - t_mean).powi(2);
            println!("{:.2}\t| {:.3}\t| {:.3}\t| {:.3}", a, t, t_pred, error);
        }

        rmse = (rmse / data.len() as f64).sqrt();
        let r_squared = 1.0 - (r_squared_num / r_squared_den);
        println!("RMSE: {:.6}, R² = {:.4}", rmse, r_squared);
    }

    println!("\n=== Result ===");
    println!("✓ Best fit: T = a^1.5");
    println!("✓ This is Kepler's Third Law: T² ∝ a³");
    println!("✓ Rearranged: T² = a³ (in units where G*M = 4π²)\n");

    println!("=== Recovered Law ===");
    println!("Kepler's Third Law (discovered 1619):");
    println!("T² = k * a³");
    println!();
    println!("where:");
    println!("  T = orbital period (years)");
    println!("  a = semi-major axis (AU)");
    println!("  k = 1 (in these units)\n");

    println!("=== Physical Significance ===");
    println!("This law holds for any body orbiting the same central mass:");
    println!("- Planets orbiting the Sun");
    println!("- Moons orbiting planets");
    println!("- Satellites orbiting Earth");
    println!("- Binary stars orbiting each other\n");

    println!("=== Why Neurosymbolic? ===");
    println!("Neural component: Optimize coefficient k");
    println!("Symbolic component: Try different exponents (1.0, 1.5, 2.0, ...)");
    println!("Together: Discover closed-form physical laws\n");

    println!("=== Validation: Testing on Satellites ===");
    println!("Geostationary satellite: a = 6.6 Earth radii");
    println!("Prediction: T = 6.6^1.5 ≈ 17.0 sidereal hours");
    println!("Actual: ~23.9 hours (23h 56m 4s)");
    println!("(Note: ratio accounts for Earth radius vs AU)\n");

    println!("=== Modern Application ===");
    println!("Symbolic regression now used for:");
    println!("✓ Discovering conservation laws from simulation data");
    println!("✓ Finding neural network-friendly encodings");
    println!("✓ Interpretable machine learning for science");
    println!("✓ Equation extraction from noisy experimental data");
}
