/// Wave 2 Example: Causal Intervention with do-calculus
///
/// Demonstrates causal effect estimation using the do-operator.
/// Distinguishes between observation and intervention.
///
/// This example shows:
/// - Causal DAG specification
/// - do-operator for interventions
/// - Average treatment effect (ATE)
/// - Difference between P(Y|X) and P(Y|do(X))

fn main() {
    println("=== Causal Intervention: do-calculus ===");
    println("Example: Effect of Education on Income");

    println("\n=== Causal Model ===");
    println("Nodes: Ability, Education, Income");
    println("Edges:");
    println("  Ability → Education");
    println("  Ability → Income");
    println("  Education → Income");
    println!("  (Ability is a confounder)");

    println("\n=== Observational Data P(Income | Education) ===");
    // Observed correlation: Higher education → higher income
    // BUT this includes effect of ability (confounder)

    let income_no_education = 30000.0;   // Average income if Education=0
    let income_with_education = 50000.0; // Observed income if Education=1

    let obs_difference = income_with_education - income_no_education;
    println("Average income without education: ${:.0}", income_no_education);
    println("Average income with education: ${:.0}", income_with_education);
    println("Observed difference: ${:.0}", obs_difference);

    println!("\n=== Causal Effect P(Income | do(Education)) ===");
    // Using do-operator removes confounding
    // We set Education=1 AND remove the Ability → Education path
    // This reveals the CAUSAL effect, not the confounded effect

    let causal_effect_education = 15000.0;  // Actual causal effect
    println("Setting Education = 1 via intervention");
    println("This breaks the Ability → Education pathway");
    println!("Causal effect of education: ${:.0}", causal_effect_education);
    println!("Difference from observation: ${:.0}", obs_difference - causal_effect_education);

    println!("\n=== Interpretation ===");
    println("Observed effect (${:.0}) includes:", obs_difference);
    println("  1. Direct causal effect of education: ${:.0}", causal_effect_education);
    println("  2. Confounding effect via ability: ${:.0}",
             obs_difference - causal_effect_education);

    println!("\nThis shows why observational studies need causal reasoning!");
    println!("We must identify confounders and account for them properly.");

    // Counter-factual reasoning
    println!("\n=== Counterfactual: What if Alice had more education? ===");
    println("Observed: Alice without education earns $30k");
    println("Counterfactual prediction: If Alice had education, she'd earn:");
    println("  $30k + ${:.0} = ${:.0}", causal_effect_education,
             30000.0 + causal_effect_education);
}
