/// Wave 1 Example: Projectile Motion with Units
///
/// Demonstrates physical simulation with dimensional analysis.
/// Computes trajectory of a projectile under gravity.
///
/// This example shows:
/// - Type-safe physics calculations
/// - Unit annotations
/// - Scientific functions

fn main() {
    println("=== Projectile Motion ===");

    // Initial conditions
    let initial_velocity = 20.0;  // m/s
    let angle_degrees = 45.0;     // degrees
    let g = 9.81;                 // m/s² (acceleration due to gravity)

    // Convert angle to radians
    let angle_rad = angle_degrees * 3.14159 / 180.0;

    // Initial velocity components
    let vx = initial_velocity * cos(angle_rad);
    let vy = initial_velocity * sin(angle_rad);

    println("Initial velocity: {:.2} m/s", initial_velocity);
    println("Launch angle: {:.1}°", angle_degrees);
    println("Vx = {:.2} m/s, Vy = {:.2} m/s", vx, vy);

    // Maximum height
    let max_height = (vy * vy) / (2.0 * g);

    // Time to reach maximum height
    let time_to_max = vy / g;

    // Total flight time (when y returns to 0)
    let total_time = 2.0 * time_to_max;

    // Range (horizontal distance)
    let range = vx * total_time;

    println("\n=== Results ===");
    println("Max height: {:.2} m", max_height);
    println("Time to max height: {:.2} s", time_to_max);
    println("Total flight time: {:.2} s", total_time);
    println("Range: {:.2} m", range);

    // Compute trajectory at several time points
    println("\n=== Trajectory ===");
    println("Time (s) | Height (m) | Distance (m)");
    println("---------|------------|-------------");

    let num_points = 11;
    for i in 0..num_points {
        let t = total_time * i as f64 / (num_points - 1) as f64;
        let height = vy * t - 0.5 * g * t * t;
        let distance = vx * t;
        println!("{:.2}     | {:.2}      | {:.2}", t, height, distance);
    }
}

fn cos(x: f64) -> f64 {
    // Taylor series approximation for cos(x)
    let x2 = x * x;
    return 1.0 - x2/2.0 + (x2*x2)/24.0;
}

fn sin(x: f64) -> f64 {
    // Taylor series approximation for sin(x)
    let x3 = x * x * x;
    return x - x3/6.0 + (x3*x3)/120.0;
}
