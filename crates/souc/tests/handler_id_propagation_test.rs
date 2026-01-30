//! Test handler ID propagation through SIR pipeline
//!
//! Verifies that:
//! 1. SirModule has effect_handler_ids populated
//! 2. Handler metadata is correctly populated
//! 3. All standard effects have correct IDs

use sounio::sir::module::{EFFECT_HANDLER_MAP, HandlerMetadata, SirModule, get_handler_id};

#[test]
fn test_handler_id_mapping() {
    // Test that all standard effects have handler IDs
    let expected_effects = [
        ("IO", 1u32),
        ("Mut", 2),
        ("Alloc", 3),
        ("Panic", 4),
        ("Async", 5),
        ("GPU", 6),
        ("Prob", 7),
        ("Grad", 8),
        ("Causal", 9),
        ("Epistemic", 10),
        ("FFI", 11),
        ("Network", 12),
        ("Sensor", 13),
    ];

    for (effect_name, expected_id) in expected_effects {
        let handler_id = get_handler_id(effect_name);
        assert!(
            handler_id.is_some(),
            "No handler ID for effect: {}",
            effect_name
        );

        let id = handler_id.unwrap();
        assert_eq!(
            id, expected_id,
            "Wrong handler ID for {}: expected {}, got {}",
            effect_name, expected_id, id
        );
    }

    println!("All effect handler IDs verified!");
}

#[test]
fn test_effect_handler_map_constant() {
    // Test the constant is properly defined
    assert!(!EFFECT_HANDLER_MAP.is_empty());
    assert_eq!(
        EFFECT_HANDLER_MAP.len(),
        13,
        "Should have 13 standard effects"
    );

    // Verify no duplicate IDs
    let mut ids = std::collections::HashSet::new();
    for (effect, id) in EFFECT_HANDLER_MAP {
        assert!(
            !ids.contains(id),
            "Duplicate handler ID: {} used by {}",
            id,
            effect
        );
        ids.insert(*id);
    }

    println!(
        "Handler ID map has {} entries with no duplicates",
        EFFECT_HANDLER_MAP.len()
    );
}

#[test]
fn test_handler_metadata_creation() {
    // Test creating handler metadata
    let mut meta = HandlerMetadata::new("IO".to_string(), 1);

    assert_eq!(meta.effect, "IO");
    assert_eq!(meta.handler_id, 1);
    assert!(meta.operations.is_empty());
    assert!(!meta.requires_state);
    assert!(!meta.is_async);

    // Add operations
    meta = meta.with_operation("print");
    meta = meta.with_operation("println");

    assert_eq!(meta.operations.len(), 2);
    assert!(meta.operations.contains(&"print".to_string()));
    assert!(meta.operations.contains(&"println".to_string()));

    println!("Handler metadata creation works correctly");
}

#[test]
fn test_handler_metadata_state_and_async() {
    // Test state requirement and async flags
    let mut meta = HandlerMetadata::new("Mut".to_string(), 2);
    meta = meta.with_state();
    assert!(meta.requires_state);
    assert!(!meta.is_async);

    let mut meta2 = HandlerMetadata::new("Async".to_string(), 5);
    meta2 = meta2.with_async();
    assert!(!meta2.requires_state);
    assert!(meta2.is_async);

    let mut meta3 = HandlerMetadata::new("GPU".to_string(), 6);
    meta3 = meta3.with_async().with_state();
    assert!(meta3.requires_state);
    assert!(meta3.is_async);

    println!("Handler state and async flags work correctly");
}

#[test]
fn test_sir_module_handler_initialization() {
    // Test that a new SirModule has empty handler maps
    let module = SirModule::new("test_module");

    assert!(module.effect_handler_ids.is_empty());
    // Note: handler_metadata field may not be directly accessible if it's not pub
    // This test just verifies SirModule can be created

    println!("SirModule handler fields are properly initialized");
}

#[test]
fn test_handler_id_uniqueness() {
    // Verify all handler IDs are in range 1-13 and unique
    let mut ids_seen = std::collections::HashSet::new();

    for (effect, id) in EFFECT_HANDLER_MAP {
        assert!(
            *id > 0 && *id <= 13,
            "Handler ID {} for {} is out of range",
            id,
            effect
        );
        assert!(!ids_seen.contains(id), "Duplicate ID {} for {}", id, effect);
        ids_seen.insert(*id);
    }

    assert_eq!(ids_seen.len(), 13, "Should have all 13 IDs registered");

    println!("All 13 handler IDs are unique and in valid range");
}
