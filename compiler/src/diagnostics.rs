//! Diagnostic reporting with source locations
//!
//! This module provides rich error messages with source locations using miette.
//!
//! Day 8 enhancements:
//! - Unit of measure errors with suggestions
//! - Parser recovery errors
//! - Enhanced error context and related information

use crate::common::Span;
use miette::{Diagnostic, NamedSource, Severity, SourceSpan};
use std::sync::Arc;
use thiserror::Error;

/// Source file for error reporting
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub name: String,
    pub content: Arc<str>,
}

impl SourceFile {
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: Arc::from(content.into()),
        }
    }

    pub fn to_named_source(&self) -> NamedSource<String> {
        NamedSource::new(self.name.clone(), self.content.to_string())
    }
}

/// Convert our Span to miette's SourceSpan
impl From<Span> for SourceSpan {
    fn from(span: Span) -> Self {
        SourceSpan::new(span.start.into(), span.len())
    }
}

/// Compiler diagnostic
#[derive(Error, Debug, Diagnostic, Clone)]
pub enum CompileError {
    // === Parse Errors ===
    #[error("Unexpected token: expected {expected}, found {found}")]
    #[diagnostic(code(parse::unexpected_token))]
    UnexpectedToken {
        expected: String,
        found: String,
        #[label("unexpected token here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Unexpected end of file")]
    #[diagnostic(code(parse::unexpected_eof))]
    UnexpectedEof {
        #[label("expected more tokens")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Resolution Errors ===
    #[error("Undefined variable `{name}`")]
    #[diagnostic(
        code(resolve::undefined_var),
        help("did you mean to declare this variable with `let`?")
    )]
    UndefinedVariable {
        name: String,
        #[label("not found in this scope")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Undefined type `{name}`")]
    #[diagnostic(code(resolve::undefined_type))]
    UndefinedType {
        name: String,
        #[label("type not found")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Duplicate definition of `{name}`")]
    #[diagnostic(code(resolve::duplicate_def))]
    DuplicateDefinition {
        name: String,
        #[label("redefined here")]
        span: SourceSpan,
        #[label("first defined here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Type Errors ===
    #[error("Type mismatch: expected `{expected}`, found `{found}`")]
    #[diagnostic(code(typecheck::mismatch))]
    TypeMismatch {
        expected: String,
        found: String,
        #[label("expected `{expected}`")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
        #[help]
        help: Option<String>,
    },

    #[error("Cannot unify `{t1}` with `{t2}`")]
    #[diagnostic(code(typecheck::unification_failed))]
    UnificationFailed {
        t1: String,
        t2: String,
        #[label("type mismatch here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Missing type annotation")]
    #[diagnostic(
        code(typecheck::annotation_required),
        help("add a type annotation: `let x: Type = ...`")
    )]
    AnnotationRequired {
        #[label("cannot infer type")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Effect Errors ===
    #[error("Unhandled effect `{effect}`")]
    #[diagnostic(
        code(effect::unhandled),
        help(
            "either handle this effect with `with handler {{ ... }}` or add it to the function signature"
        )
    )]
    UnhandledEffect {
        effect: String,
        #[label("effect `{effect}` escapes here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Effect `{effect}` not declared in function signature")]
    #[diagnostic(code(effect::undeclared))]
    UndeclaredEffect {
        effect: String,
        #[label("this operation has effect `{effect}`")]
        span: SourceSpan,
        #[label("function does not declare this effect")]
        fn_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Cannot perform `{effect}` in pure context")]
    #[diagnostic(code(effect::pure_context))]
    EffectInPureContext {
        effect: String,
        #[label("effectful operation here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Ownership Errors ===
    #[error("Use of moved value `{name}`")]
    #[diagnostic(code(ownership::use_after_move))]
    UseAfterMove {
        name: String,
        #[label("value used here after move")]
        use_span: SourceSpan,
        #[label("value moved here")]
        move_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Cannot borrow `{name}` as mutable because it is already borrowed")]
    #[diagnostic(code(ownership::already_borrowed))]
    AlreadyBorrowed {
        name: String,
        #[label("cannot borrow as mutable")]
        span: SourceSpan,
        #[label("previous borrow here")]
        prev_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Cannot borrow `{name}` as mutable more than once")]
    #[diagnostic(code(ownership::double_mut_borrow))]
    DoubleMutBorrow {
        name: String,
        #[label("second mutable borrow here")]
        span: SourceSpan,
        #[label("first mutable borrow here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Linearity Errors ===
    #[error("Linear value `{name}` used more than once")]
    #[diagnostic(
        code(linear::multiple_use),
        help("linear values must be used exactly once")
    )]
    LinearMultipleUse {
        name: String,
        #[label("second use here")]
        second_span: SourceSpan,
        #[label("first use here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Linear value `{name}` not consumed")]
    #[diagnostic(
        code(linear::not_consumed),
        help("linear values must be explicitly consumed before going out of scope")
    )]
    LinearNotConsumed {
        name: String,
        #[label("linear value declared here")]
        decl_span: SourceSpan,
        #[label("goes out of scope here without being consumed")]
        scope_end: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Affine value `{name}` used more than once")]
    #[diagnostic(code(affine::multiple_use))]
    AffineMultipleUse {
        name: String,
        #[label("second use here")]
        second_span: SourceSpan,
        #[label("first use here")]
        first_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Unit Errors ===
    #[error("Unit mismatch: expected `{expected}`, found `{found}`")]
    #[diagnostic(code(unit::mismatch))]
    UnitMismatch {
        expected: String,
        found: String,
        #[label("expected `{expected}`")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
        #[help]
        help: Option<String>,
    },

    #[error("Cannot add values with different units: `{u1}` and `{u2}`")]
    #[diagnostic(
        code(unit::incompatible_add),
        help("arithmetic operations require operands with compatible units")
    )]
    IncompatibleUnits {
        u1: String,
        u2: String,
        #[label("has unit `{u1}`")]
        left_span: SourceSpan,
        #[label("has unit `{u2}`")]
        right_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Unknown unit `{unit}`")]
    #[diagnostic(code(unit::unknown))]
    UnknownUnit {
        unit: String,
        #[label("unknown unit")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
        #[help]
        suggestion: Option<String>,
    },

    #[error("Cannot convert from `{from}` to `{to}`")]
    #[diagnostic(code(unit::incompatible_conversion))]
    IncompatibleConversion {
        from: String,
        to: String,
        #[label("cannot convert to `{to}`")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
        #[help]
        help: Option<String>,
    },

    #[error("Division by zero in unit computation")]
    #[diagnostic(code(unit::division_by_zero))]
    UnitDivisionByZero {
        #[label("zero divisor here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Unit inference failed for `{name}`")]
    #[diagnostic(code(unit::inference_failed), help("add an explicit unit annotation"))]
    UnitInferenceFailed {
        name: String,
        #[label("could not infer unit")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    // === Parser Recovery Errors ===
    #[error("Syntax error: {message}")]
    #[diagnostic(code(parse::syntax_error))]
    SyntaxError {
        message: String,
        #[label("{message}")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
        #[help]
        suggestion: Option<String>,
    },

    #[error("Unclosed delimiter `{open}`")]
    #[diagnostic(code(parse::unclosed_delimiter))]
    UnclosedDelimiter {
        open: String,
        expected: String,
        #[label("unclosed `{open}`")]
        open_span: SourceSpan,
        #[label("expected `{expected}` here")]
        expected_span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Missing semicolon")]
    #[diagnostic(
        code(parse::missing_semicolon),
        help("add `;` at the end of the statement")
    )]
    MissingSemicolon {
        #[label("expected `;` here")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
    },

    #[error("Invalid token in expression")]
    #[diagnostic(code(parse::invalid_token))]
    InvalidToken {
        found: String,
        #[label("unexpected `{found}`")]
        span: SourceSpan,
        #[source_code]
        src: NamedSource<String>,
        #[help]
        expected_tokens: Option<String>,
    },

    // === Generic Errors ===
    #[error("{message}")]
    #[diagnostic(code(general::error))]
    General {
        message: String,
        #[label("{label}")]
        span: SourceSpan,
        label: String,
        #[source_code]
        src: NamedSource<String>,
        #[help]
        help: Option<String>,
    },
}

/// Diagnostic severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Info,
    Hint,
}

impl From<DiagnosticLevel> for Severity {
    fn from(level: DiagnosticLevel) -> Self {
        match level {
            DiagnosticLevel::Error => Severity::Error,
            DiagnosticLevel::Warning => Severity::Warning,
            DiagnosticLevel::Info => Severity::Advice,
            DiagnosticLevel::Hint => Severity::Advice,
        }
    }
}

/// A suggestion for fixing an error
#[derive(Debug, Clone)]
pub struct Suggestion {
    /// Human-readable description of the fix
    pub message: String,
    /// The span to replace
    pub span: Span,
    /// The replacement text
    pub replacement: String,
}

impl Suggestion {
    pub fn new(message: impl Into<String>, span: Span, replacement: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span,
            replacement: replacement.into(),
        }
    }

    /// Insert text at a position (zero-length span)
    pub fn insert(message: impl Into<String>, pos: usize, text: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: Span::new(pos, pos),
            replacement: text.into(),
        }
    }

    /// Delete a span
    pub fn delete(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            replacement: String::new(),
        }
    }
}

/// Related information for a diagnostic
#[derive(Debug, Clone)]
pub struct RelatedInfo {
    /// The location of the related information
    pub span: Span,
    /// A message describing the relation
    pub message: String,
}

impl RelatedInfo {
    pub fn new(span: Span, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
        }
    }
}

/// Error reporter that collects diagnostics
pub struct Reporter {
    source: SourceFile,
    errors: Vec<CompileError>,
    warnings: Vec<CompileError>,
}

impl Reporter {
    pub fn new(source: SourceFile) -> Self {
        Self {
            source,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn error(&mut self, error: CompileError) {
        self.errors.push(error);
    }

    pub fn warning(&mut self, warning: CompileError) {
        self.warnings.push(warning);
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Create NamedSource for this file
    pub fn named_source(&self) -> NamedSource<String> {
        self.source.to_named_source()
    }

    /// Get the source file
    pub fn source(&self) -> &SourceFile {
        &self.source
    }

    /// Print all diagnostics
    pub fn emit_all(&self) {
        for warning in &self.warnings {
            eprintln!("{:?}", miette::Report::new(warning.clone()));
        }
        for error in &self.errors {
            eprintln!("{:?}", miette::Report::new(error.clone()));
        }
    }

    /// Consume and return errors
    pub fn into_errors(self) -> Vec<CompileError> {
        self.errors
    }

    /// Get errors by reference
    pub fn errors(&self) -> &[CompileError] {
        &self.errors
    }
}
