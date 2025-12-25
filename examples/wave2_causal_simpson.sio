/// Wave 2 Example: Simpson's Paradox in Causal Inference
///
/// Demonstrates how causal reasoning resolves Simpson's paradox.
/// Classic example: UC Berkeley admissions data
///
/// This example shows:
/// - Causal reasoning with confounders
/// - Simpson's paradox detection
/// - Proper causal adjustment

fn main() {
    println("=== Simpson's Paradox ===");
    println("UC Berkeley Admissions Case Study");

    // Simulate data
    // Male applicants: 60% admitted (overall)
    // Female applicants: 50% admitted (overall)
    // Appears males have higher admission rate

    println("\n=== Marginal (Unadjusted) Statistics ===");
    let male_admitted = 1198;
    let male_total = 1993;
    let male_rate = male_admitted as f64 / male_total as f64;

    let female_admitted = 557;
    let female_total = 1831;
    let female_rate = female_admitted as f64 / female_total as f64;

    println("Males admitted: {}/{} = {:.1}%", male_admitted, male_total, male_rate * 100.0);
    println("Females admitted: {}/{} = {:.1}%", female_admitted, female_total, female_rate * 100.0);
    println!("Difference: {:.1}%", (male_rate - female_rate) * 100.0);

    // But when stratified by department, the pattern reverses!
    println("\n=== Conditional (Stratified by Department) ===");

    // Engineering (high admission rate)
    println("\nEngineering Department:");
    let eng_male_adm = 512;
    let eng_male_total = 825;
    let eng_male_rate = eng_male_adm as f64 / eng_male_total as f64;

    let eng_female_adm = 89;
    let eng_female_total = 108;
    let eng_female_rate = eng_female_adm as f64 / eng_female_total as f64;

    println("  Males: {:.1}%", eng_male_rate * 100.0);
    println("  Females: {:.1}%", eng_female_rate * 100.0);

    // Liberal Arts (lower admission rate)
    println("\nLiberal Arts Department:");
    let la_male_adm = 353;
    let la_male_total = 1022;
    let la_male_rate = la_male_adm as f64 / la_male_total as f64;

    let la_female_adm = 207;
    let la_female_total = 1075;
    let la_female_rate = la_female_adm as f64 / la_female_total as f64;

    println("  Males: {:.1}%", la_male_rate * 100.0);
    println("  Females: {:.1}%", la_female_rate * 100.0);

    println!("\n=== Causal Analysis ===");
    println("The confounder: DEPARTMENT");
    println("- Females apply more to Liberal Arts (lower admission rate)");
    println("- Males apply more to Engineering (higher admission rate)");
    println("- When we stratify by department, females have higher rates!");

    println!("\n=== Conclusion ===");
    println("This is Simpson's Paradox: the direction of association reverses");
    println("when we properly account for the confounder (department).");
    println!("\nCausal principle: Always adjust for confounders identified");
    println("in your causal DAG, not just observed association.");
}
