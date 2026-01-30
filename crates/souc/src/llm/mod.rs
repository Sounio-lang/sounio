//! LLM Integration Infrastructure
//!
//! Provides abstraction over multiple LLM backends for ontology generation.
//! Inspired by LLMs4OL Challenge 2025: hybrid approaches with domain embeddings
//! achieve best results for ontology learning tasks.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    LLMClientRegistry                        │
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
//! │  │ OpenAI   │  │Anthropic │  │  Ollama  │  │  Custom  │    │
//! │  │  Client  │  │  Client  │  │  Client  │  │  Client  │    │
//! │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
//! │       └─────────────┴─────────────┴─────────────┘          │
//! │                         │                                   │
//! │              ┌──────────▼──────────┐                       │
//! │              │    LLMRequest       │                       │
//! │              │  - prompt           │                       │
//! │              │  - system           │                       │
//! │              │  - temperature      │                       │
//! │              │  - max_tokens       │                       │
//! │              └──────────┬──────────┘                       │
//! │                         │                                   │
//! │              ┌──────────▼──────────┐                       │
//! │              │    LLMResponse      │                       │
//! │              │  - content          │                       │
//! │              │  - confidence       │                       │
//! │              │  - tokens_used      │                       │
//! │              └─────────────────────┘                       │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::llm::{LLMClientRegistry, LLMRequest, OntologyTask};
//!
//! // Initialize from environment variables
//! let registry = LLMClientRegistry::from_env();
//!
//! // Build a prompt for term extraction
//! let request = registry.prompts()
//!     .build_prompt(&OntologyTask::TermExtraction, &params)?;
//!
//! // Query the LLM
//! let response = registry.default_client()?.query(&request)?;
//!
//! // Estimate confidence from response
//! let confidence = response.confidence_indicators.to_confidence();
//! ```

pub mod client;
pub mod confidence;
pub mod prompts;

pub use client::{AnthropicClient, LLMClient, LLMClientRegistry, OllamaClient, OpenAIClient};
pub use confidence::{ConfidenceIndicators, analyze_confidence, indicators_to_confidence};
pub use prompts::{OntologyTask, PromptExample, PromptTemplate, PromptTemplates};

use std::fmt;

/// Supported LLM providers
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LLMProvider {
    /// OpenAI API (GPT-4, etc.)
    OpenAI,
    /// Anthropic API (Claude)
    Anthropic,
    /// Local model via Ollama
    Ollama,
    /// Custom endpoint
    Custom(String),
}

impl fmt::Display for LLMProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMProvider::OpenAI => write!(f, "OpenAI"),
            LLMProvider::Anthropic => write!(f, "Anthropic"),
            LLMProvider::Ollama => write!(f, "Ollama"),
            LLMProvider::Custom(url) => write!(f, "Custom({})", url),
        }
    }
}

/// LLM request configuration
#[derive(Debug, Clone)]
pub struct LLMRequest {
    /// The prompt to send
    pub prompt: String,
    /// System message (if supported)
    pub system: Option<String>,
    /// Temperature (0.0 = deterministic, 1.0+ = creative)
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

impl Default for LLMRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            system: None,
            temperature: 0.3, // Low for ontology tasks - need precision
            max_tokens: 4096,
            stop_sequences: vec!["```".to_string()],
        }
    }
}

impl LLMRequest {
    /// Create a new request with just a prompt
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            ..Default::default()
        }
    }

    /// Set the system message
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Add stop sequences
    pub fn with_stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.stop_sequences = sequences;
        self
    }
}

/// LLM response with metadata
#[derive(Debug, Clone)]
pub struct LLMResponse {
    /// Generated text
    pub content: String,
    /// Model used
    pub model: String,
    /// Token usage
    pub tokens_used: usize,
    /// Finish reason
    pub finish_reason: FinishReason,
    /// Raw confidence indicators extracted from response
    pub confidence_indicators: ConfidenceIndicators,
}

impl LLMResponse {
    /// Get the estimated epistemic confidence for this response
    pub fn estimated_confidence(&self) -> f64 {
        indicators_to_confidence(&self.confidence_indicators)
    }

    /// Check if the response completed normally
    pub fn is_complete(&self) -> bool {
        matches!(self.finish_reason, FinishReason::Stop)
    }

    /// Check if the response was truncated due to length
    pub fn is_truncated(&self) -> bool {
        matches!(self.finish_reason, FinishReason::Length)
    }
}

/// Reason why the LLM stopped generating
#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    /// Normal completion (hit stop sequence or end of response)
    Stop,
    /// Hit max_tokens limit
    Length,
    /// Content was filtered by safety systems
    ContentFilter,
    /// An error occurred
    Error(String),
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::ContentFilter => write!(f, "content_filter"),
            FinishReason::Error(msg) => write!(f, "error: {}", msg),
        }
    }
}

/// LLM-related errors
#[derive(Debug, Clone)]
pub enum LLMError {
    /// Provider not configured
    NotConfigured(LLMProvider),
    /// No providers available
    NoProvidersAvailable,
    /// API error
    ApiError { status: u16, message: String },
    /// Rate limited
    RateLimited { retry_after_secs: u64 },
    /// Response parsing failed
    ParseError(String),
    /// Network error
    NetworkError(String),
    /// Invalid request
    InvalidRequest(String),
    /// Timeout
    Timeout { elapsed_secs: u64 },
}

impl fmt::Display for LLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMError::NotConfigured(p) => write!(f, "LLM provider {} not configured", p),
            LLMError::NoProvidersAvailable => write!(f, "No LLM providers available"),
            LLMError::ApiError { status, message } => {
                write!(f, "API error {}: {}", status, message)
            }
            LLMError::RateLimited { retry_after_secs } => {
                write!(f, "Rate limited, retry after {}s", retry_after_secs)
            }
            LLMError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            LLMError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            LLMError::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            LLMError::Timeout { elapsed_secs } => {
                write!(f, "Request timed out after {}s", elapsed_secs)
            }
        }
    }
}

impl std::error::Error for LLMError {}

/// Configuration for LLM integration
#[derive(Debug, Clone)]
pub struct LLMConfig {
    /// Default provider to use
    pub default_provider: Option<LLMProvider>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries on transient errors
    pub max_retries: u32,
    /// Whether to cache responses
    pub enable_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            default_provider: None,
            timeout_secs: 60,
            max_retries: 3,
            enable_cache: true,
            cache_ttl_secs: 3600, // 1 hour
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_request_builder() {
        let request = LLMRequest::new("Extract terms from this text")
            .with_system("You are an ontology engineer")
            .with_temperature(0.2)
            .with_max_tokens(2048);

        assert_eq!(request.prompt, "Extract terms from this text");
        assert_eq!(
            request.system,
            Some("You are an ontology engineer".to_string())
        );
        assert!((request.temperature - 0.2).abs() < f32::EPSILON);
        assert_eq!(request.max_tokens, 2048);
    }

    #[test]
    fn test_llm_provider_display() {
        assert_eq!(format!("{}", LLMProvider::OpenAI), "OpenAI");
        assert_eq!(format!("{}", LLMProvider::Anthropic), "Anthropic");
        assert_eq!(format!("{}", LLMProvider::Ollama), "Ollama");
        assert_eq!(
            format!("{}", LLMProvider::Custom("http://localhost:8080".into())),
            "Custom(http://localhost:8080)"
        );
    }

    #[test]
    fn test_finish_reason_display() {
        assert_eq!(format!("{}", FinishReason::Stop), "stop");
        assert_eq!(format!("{}", FinishReason::Length), "length");
        assert_eq!(
            format!("{}", FinishReason::Error("test".into())),
            "error: test"
        );
    }
}
