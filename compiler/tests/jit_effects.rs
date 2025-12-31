//! Integration tests for effect handlers
//!
//! Tests the effect dispatch infrastructure in both interpreter and JIT.

/// Test effect dispatch in interpreter mode
#[test]
fn test_interpreter_effect_dispatch() {
    use sounio::interp::effect_dispatch::{EffectContext, EffectKind};
    use sounio::interp::value::Value;
    use std::collections::HashMap;

    // Create effect context
    let mut ctx = EffectContext::with_seed(42);

    // Test Prob.sample with a Normal distribution struct
    let mut fields = HashMap::new();
    fields.insert("mean".to_string(), Value::Float(5.0));
    fields.insert("std".to_string(), Value::Float(1.0));
    let dist = Value::Struct {
        name: "Normal".to_string(),
        fields,
    };

    // Sample should return a float around 5.0
    let result = ctx.sample(dist);
    assert!(result.is_ok(), "Sample should succeed");

    if let Ok(Value::Float(v)) = result {
        // Value should be reasonable for N(5, 1)
        assert!(
            v > 0.0 && v < 10.0,
            "Sampled value {} should be in reasonable range for N(5, 1)",
            v
        );
        println!("Sampled value from N(5, 1): {}", v);
    }

    // Test Causal.do intervention
    let var = Value::String("X".to_string());
    let val = Value::Float(42.0);
    let result = ctx.do_intervention(var, val);
    assert!(result.is_ok(), "Do intervention should succeed");

    // Check intervention was recorded
    assert_eq!(ctx.state.interventions.len(), 1);
    assert_eq!(ctx.state.interventions[0].0, "X");
    println!("Intervention recorded: X = 42.0");
}

/// Test multiple samples have different values (randomness works)
#[test]
fn test_sample_randomness() {
    use sounio::interp::effect_dispatch::EffectContext;
    use sounio::interp::value::Value;
    use std::collections::HashMap;

    let mut ctx = EffectContext::with_seed(12345);

    let mut samples = Vec::new();
    for _ in 0..10 {
        let mut fields = HashMap::new();
        fields.insert("mean".to_string(), Value::Float(0.0));
        fields.insert("std".to_string(), Value::Float(1.0));
        let dist = Value::Struct {
            name: "Normal".to_string(),
            fields,
        };

        if let Ok(Value::Float(v)) = ctx.sample(dist) {
            samples.push(v);
        }
    }

    assert_eq!(samples.len(), 10, "Should have 10 samples");

    // Check that not all samples are identical (would indicate broken RNG)
    let first = samples[0];
    let all_same = samples.iter().all(|&s| (s - first).abs() < 0.0001);
    assert!(!all_same, "Samples should not all be identical");

    println!("10 samples from N(0, 1): {:?}", samples);

    // Check samples are roughly centered around 0
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    assert!(
        mean.abs() < 2.0,
        "Mean of samples {} should be close to 0",
        mean
    );
    println!("Sample mean: {}", mean);
}

/// Test observe operation accumulates log probability
#[test]
fn test_observe_log_prob() {
    use sounio::interp::effect_dispatch::EffectContext;
    use sounio::interp::value::Value;
    use std::collections::HashMap;

    let mut ctx = EffectContext::with_seed(999);

    // Create Normal(0, 1) distribution
    let mut fields = HashMap::new();
    fields.insert("mean".to_string(), Value::Float(0.0));
    fields.insert("std".to_string(), Value::Float(1.0));
    let dist = Value::Struct {
        name: "Normal".to_string(),
        fields,
    };

    // Observe value 0 (should have high probability for N(0,1))
    let result = ctx.observe(dist, Value::Float(0.0));
    assert!(result.is_ok(), "Observe should succeed");
    println!("Observe result: {:?}", result);
}

/// Test handler stack with custom handler
#[test]
fn test_custom_handler() {
    use sounio::interp::effect_dispatch::{EffectContext, EffectHandler, EffectKind};
    use sounio::interp::value::Value;
    use std::collections::HashMap;

    let mut ctx = EffectContext::with_seed(42);
    let initial_handlers = ctx.state.observations.len();

    // Push a custom handler that always returns 42.0 for sample
    let custom_handler = EffectHandler::new(EffectKind::Prob, "constant_sampler")
        .with_case("sample", |_args, _state| Ok(Value::Float(42.0)));

    ctx.push_handler(custom_handler);

    // Now sample should return 42.0
    let mut fields = HashMap::new();
    fields.insert("mean".to_string(), Value::Float(0.0));
    fields.insert("std".to_string(), Value::Float(1.0));
    let dist = Value::Struct {
        name: "Normal".to_string(),
        fields,
    };

    let result = ctx.sample(dist);
    assert!(result.is_ok());

    if let Ok(Value::Float(v)) = result {
        assert!(
            (v - 42.0).abs() < 0.001,
            "Custom handler should return 42.0, got {}",
            v
        );
        println!("Custom handler returned: {}", v);
    }

    // Pop handler and verify default handler is used again
    ctx.pop_handler();

    let mut fields2 = HashMap::new();
    fields2.insert("mean".to_string(), Value::Float(0.0));
    fields2.insert("std".to_string(), Value::Float(1.0));
    let dist2 = Value::Struct {
        name: "Normal".to_string(),
        fields: fields2,
    };

    let result2 = ctx.sample(dist2);
    assert!(result2.is_ok());

    if let Ok(Value::Float(v)) = result2 {
        assert!(
            (v - 42.0).abs() > 0.001,
            "After popping custom handler, should use default (not 42.0)"
        );
        println!("Default handler returned: {}", v);
    }
}

/// Test dispatch by effect name string
#[test]
fn test_dispatch_by_name() {
    use sounio::interp::effect_dispatch::EffectContext;
    use sounio::interp::value::Value;
    use std::collections::HashMap;

    let mut ctx = EffectContext::with_seed(777);

    // Test Prob.sample via dispatch_by_name
    let mut fields = HashMap::new();
    fields.insert("mean".to_string(), Value::Float(10.0));
    fields.insert("std".to_string(), Value::Float(0.5));
    let dist = Value::Struct {
        name: "Normal".to_string(),
        fields,
    };

    let result = ctx.dispatch_by_name("Prob", "sample", vec![dist]);
    assert!(result.is_ok(), "dispatch_by_name should succeed");

    if let Ok(Value::Float(v)) = result {
        println!("dispatch_by_name(Prob, sample) returned: {}", v);
        assert!(v > 5.0 && v < 15.0, "Value should be around 10");
    }

    // Test unknown effect
    let result = ctx.dispatch_by_name("UnknownEffect", "op", vec![]);
    assert!(result.is_err(), "Unknown effect should fail");
}
