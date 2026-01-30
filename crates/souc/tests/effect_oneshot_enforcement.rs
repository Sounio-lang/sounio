//! Tests for one-shot continuation enforcement
//!
//! Verifies that:
//! 1. One-shot continuations can be resumed exactly once
//! 2. Attempting to resume a one-shot continuation twice raises an error
//! 3. Multi-shot continuations can be resumed multiple times
//! 4. Linearity constraints are properly enforced

use sounio::effects::continuation::{CapturedContinuation, ContinuationError, ResumePoint};
use sounio::effects::linearity::Linearity;
use sounio::interp::value::Value;

#[test]
fn test_oneshot_continuation_single_resume() {
    // Create a one-shot continuation
    let resume_point = ResumePoint::interpreter_closure(|v| Ok(v), Some("test-one-shot"));
    let mut cont = CapturedContinuation::new_one_shot(resume_point);

    // Should be able to resume once
    assert!(cont.can_resume());
    let result = cont.resume(Value::Int(42));
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), Value::Int(42));

    // Should have been resumed
    assert!(cont.has_been_resumed());
    assert_eq!(cont.resume_count, 1);

    // Should not be able to resume again
    assert!(!cont.can_resume());
}

#[test]
fn test_oneshot_continuation_double_resume_fails() {
    // Create a one-shot continuation
    let resume_point = ResumePoint::interpreter_closure(|v| Ok(v), Some("test-double-resume"));
    let mut cont = CapturedContinuation::new_one_shot(resume_point);

    // First resume should succeed
    let result1 = cont.resume(Value::Int(1));
    assert!(result1.is_ok());

    // Second resume should fail with AlreadyResumed error
    let result2 = cont.resume(Value::Int(2));
    assert!(result2.is_err());

    match result2.unwrap_err() {
        ContinuationError::AlreadyResumed { id, label } => {
            assert_eq!(id, cont.id);
            assert_eq!(label, Some("test-double-resume".to_string()));
        }
        other => panic!("Expected AlreadyResumed error, got {:?}", other),
    }
}

#[test]
fn test_multishot_continuation_multiple_resumes() {
    // Create a multi-shot continuation
    let resume_point = ResumePoint::interpreter_multi_shot(|v| Ok(v), Some("test-multi-shot"));
    let mut cont = CapturedContinuation::new_multi_shot(resume_point);

    // Should be able to resume multiple times
    assert!(cont.can_resume());
    assert!(cont.is_multi_shot);

    let result1 = cont.resume(Value::Int(1));
    assert!(result1.is_ok());
    assert_eq!(cont.resume_count, 1);

    let result2 = cont.resume(Value::Int(2));
    assert!(result2.is_ok());
    assert_eq!(cont.resume_count, 2);

    let result3 = cont.resume(Value::Int(3));
    assert!(result3.is_ok());
    assert_eq!(cont.resume_count, 3);

    // Still resumable
    assert!(cont.can_resume());
}

#[test]
fn test_linearity_types() {
    // Test that linearity types have correct properties
    assert!(Linearity::ExactlyOnce.is_linear());
    assert!(Linearity::ExactlyOnce.requires_resumption());
    assert!(!Linearity::ExactlyOnce.is_multi_shot());

    assert!(Linearity::OnceOrNever.is_affine());
    assert!(!Linearity::OnceOrNever.requires_resumption());
    assert!(!Linearity::OnceOrNever.is_multi_shot());

    assert!(Linearity::MultiShot.is_multi_shot());
    assert!(!Linearity::MultiShot.requires_resumption());
    assert!(!Linearity::MultiShot.is_linear());
}

#[test]
fn test_linearity_compatibility() {
    // Multi-shot can be used where one-shot is expected
    assert!(Linearity::MultiShot.is_compatible_with(Linearity::ExactlyOnce));
    assert!(Linearity::MultiShot.is_compatible_with(Linearity::OnceOrNever));

    // Linear can be used where affine is expected
    assert!(Linearity::ExactlyOnce.is_compatible_with(Linearity::OnceOrNever));

    // But not the other way around
    assert!(!Linearity::OnceOrNever.is_compatible_with(Linearity::ExactlyOnce));
    assert!(!Linearity::ExactlyOnce.is_compatible_with(Linearity::MultiShot));
}

#[test]
fn test_continuation_linearity_enforcement() {
    // One-shot continuation enforces linearity
    let resume_point = ResumePoint::interpreter_closure(|v| Ok(v), Some("test-enforcement"));
    let mut cont = CapturedContinuation::new_one_shot(resume_point);

    // Verify it's marked as one-shot
    assert!(!cont.is_multi_shot);
    assert_eq!(cont.resume_count, 0);

    // Resume once - should succeed
    let result = cont.resume(Value::Int(100));
    assert!(result.is_ok());
    assert_eq!(cont.resume_count, 1);

    // Attempt second resume - should fail
    let result = cont.resume(Value::Int(200));
    assert!(result.is_err());

    // Verify resume count didn't increase on failure
    assert_eq!(cont.resume_count, 1);
}
