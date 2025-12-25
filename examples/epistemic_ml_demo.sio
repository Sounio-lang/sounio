//@ run-pass
/// Epistemic Machine Learning Demo
///
/// Demonstrates integration of causal inference and ML with epistemic uncertainty.

fn main() -> i32 {
    println("")
    println("=== Epistemic ML Demo ===")
    println("Causal inference with epistemic uncertainty tracking")
    println("")
    
    // Simple example: Treatment -> Outcome with confounder
    var dag = dag_new()
    dag = dag_add_node(dag, "Treatment", NodeType::Treatment)
    dag = dag_add_node(dag, "Outcome", NodeType::Outcome)
    dag = dag_add_node(dag, "Age", NodeType::Confounder)
    
    // Add edges with epistemic uncertainty
    dag = dag_add_edge(dag, "Age", "Treatment", beta_new(8.0, 2.0), 0.3, 0.05)
    dag = dag_add_edge(dag, "Age", "Outcome", beta_new(9.0, 1.0), 0.4, 0.03)
    dag = dag_add_edge(dag, "Treatment", "Outcome", beta_new(6.0, 4.0), 0.5, 0.10)
    
    println("Causal DAG Structure:")
    dag_print(dag)
    println("")
    
    // Estimate average treatment effect
    println("Estimating Average Treatment Effect...")
    let ate = average_treatment_effect(dag, "Treatment", "Outcome")
    epistemic_print(ate)
    println("")
    
    // Check what to adjust for
    let adjustment = backdoor_adjustment(dag, "Treatment", "Outcome")
    print("Variables to adjust for: ")
    println(len(adjustment))
    println("")
    
    println("Key Features Demonstrated:")
    println("1. Every causal edge has Beta posterior on existence")
    println("2. Effect sizes carry epistemic uncertainty")
    println("3. Automatic backdoor adjustment set identification")
    println("4. Full uncertainty propagation through do-calculus")
    println("")
    
    return 0
}
