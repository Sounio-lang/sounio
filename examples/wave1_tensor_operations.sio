/// Wave 1 Example: Tensor Operations
///
/// Demonstrates working with multi-dimensional arrays (tensors).
/// Covers vector operations and basic linear algebra.
///
/// This example shows:
/// - Creating tensors with zeros/ones
/// - Vector operations
/// - Dot products
/// - Matrix concepts

fn main() {
    println("=== Tensor Operations ===");

    // Create vectors
    let v1 = [1.0, 2.0, 3.0];
    let v2 = [4.0, 5.0, 6.0];

    println!("Vector 1: [{}, {}, {}]", v1[0], v1[1], v1[2]);
    println!("Vector 2: [{}, {}, {}]", v2[0], v2[1], v2[2]);

    // Compute dot product
    let dot_product = dot(v1, v2);
    println!("Dot product: {}", dot_product);

    // Vector magnitude
    let mag_v1 = magnitude(v1);
    let mag_v2 = magnitude(v2);

    println!("Magnitude of V1: {:.4}", mag_v1);
    println!("Magnitude of V2: {:.4}", mag_v2);

    // Cosine similarity
    if mag_v1 > 0.0 && mag_v2 > 0.0 {
        let cos_similarity = dot_product / (mag_v1 * mag_v2);
        println!("Cosine similarity: {:.4}", cos_similarity);
    }

    // Create zero and one vectors
    let zeros = zeros(5);
    let ones = ones(5);

    println!("\nZeros vector length: {}", len(zeros));
    println!("Ones vector length: {}", len(ones));

    // Matrix representation (as nested arrays)
    println!("\n=== Matrix Operations ===");

    let matrix = create_matrix();
    print_matrix(matrix);

    // Frobenius norm (matrix magnitude)
    let norm = matrix_norm(matrix);
    println!("Frobenius norm: {:.4}", norm);
}

fn dot(a: [f64], b: [f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..len(a) {
        if i < len(b) {
            sum = sum + a[i] * b[i];
        }
    }
    return sum;
}

fn magnitude(v: [f64]) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..len(v) {
        sum_sq = sum_sq + v[i] * v[i];
    }
    return sqrt(sum_sq);
}

fn sqrt(x: f64) -> f64 {
    // Newton's method for square root
    if x == 0.0 {
        return 0.0;
    }
    let mut guess = x;
    for _ in 0..10 {
        guess = (guess + x / guess) / 2.0;
    }
    return guess;
}

fn zeros(n: i64) -> [f64] {
    let arr: [f64] = [0.0; n];
    return arr;
}

fn ones(n: i64) -> [f64] {
    let arr: [f64] = [1.0; n];
    return arr;
}

fn create_matrix() -> [[f64]] {
    let row1 = [1.0, 2.0, 3.0];
    let row2 = [4.0, 5.0, 6.0];
    let row3 = [7.0, 8.0, 9.0];

    let matrix: [[f64]] = [row1, row2, row3];
    return matrix;
}

fn print_matrix(m: [[f64]]) {
    println!("Matrix:");
    for i in 0..len(m) {
        print!("  [");
        for j in 0..len(m[i]) {
            if j > 0 { print!(", "); }
            print!("{:.1}", m[i][j]);
        }
        println!("]");
    }
}

fn matrix_norm(m: [[f64]]) -> f64 {
    // Frobenius norm = sqrt(sum of all element squares)
    let mut sum_sq = 0.0;
    for i in 0..len(m) {
        for j in 0..len(m[i]) {
            sum_sq = sum_sq + m[i][j] * m[i][j];
        }
    }
    return sqrt(sum_sq);
}
