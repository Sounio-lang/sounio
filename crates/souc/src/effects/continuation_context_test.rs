//! Unit tests for continuation context tracking

#[cfg(test)]
mod tests {
    use crate::effects::{ContinuationContext, ContinuationId};

    #[test]
    fn test_empty_context() {
        let ctx = ContinuationContext::new();
        assert!(!ctx.has_active());
        assert_eq!(ctx.current(), None);
        assert_eq!(ctx.depth(), 0);
    }

    #[test]
    fn test_push_pop() {
        let mut ctx = ContinuationContext::new();

        let id1 = ContinuationId::new(1);
        ctx.push(id1);

        assert!(ctx.has_active());
        assert_eq!(ctx.current(), Some(id1));
        assert_eq!(ctx.depth(), 1);

        let popped = ctx.pop();
        assert_eq!(popped, Some(id1));
        assert!(!ctx.has_active());
    }

    #[test]
    fn test_nested_continuations() {
        let mut ctx = ContinuationContext::new();

        let id1 = ContinuationId::new(1);
        let id2 = ContinuationId::new(2);
        let id3 = ContinuationId::new(3);

        ctx.push(id1);
        ctx.push(id2);
        ctx.push(id3);

        assert_eq!(ctx.depth(), 3);
        assert_eq!(ctx.current(), Some(id3)); // innermost

        ctx.pop();
        assert_eq!(ctx.current(), Some(id2));

        ctx.pop();
        assert_eq!(ctx.current(), Some(id1));

        ctx.pop();
        assert_eq!(ctx.current(), None);
    }

    #[test]
    fn test_clear() {
        let mut ctx = ContinuationContext::new();

        ctx.push(ContinuationId::new(1));
        ctx.push(ContinuationId::new(2));

        assert_eq!(ctx.depth(), 2);

        ctx.clear();

        assert_eq!(ctx.depth(), 0);
        assert!(!ctx.has_active());
    }

    #[test]
    fn test_pop_empty() {
        let mut ctx = ContinuationContext::new();
        assert_eq!(ctx.pop(), None);
    }
}
