//! LLM Client Implementations
//!
//! Provides concrete implementations for different LLM providers.
//! Each client implements the `LLMClient` trait for uniform access.
//!
//! # Supported Providers
//!
//! - **OpenAI**: GPT-4, GPT-4o, GPT-4o-mini via official API
//! - **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus via official API
//! - **Ollama**: Local models (Llama 3.1, Mistral, CodeLlama) via local server
//!
//! # Configuration
//!
//! Clients are configured via environment variables:
//!
//! - `OPENAI_API_KEY`: OpenAI API key
//! - `OPENAI_MODEL`: Model to use (default: gpt-4-turbo)
//! - `OPENAI_BASE_URL`: Custom API endpoint (for Azure, etc.)
//!
//! - `ANTHROPIC_API_KEY`: Anthropic API key
//! - `ANTHROPIC_MODEL`: Model to use (default: claude-sonnet-4-20250514)
//!
//! - `OLLAMA_HOST`: Ollama server URL (default: http://127.0.0.1:11434)
//! - `OLLAMA_MODEL`: Model to use (default: llama3.1:8b)

use super::*;
use std::collections::HashMap;
use std::env;

/// Trait for LLM client implementations
pub trait LLMClient: Send + Sync {
    /// Send a request and get a response
    fn query(&self, request: &LLMRequest) -> Result<LLMResponse, LLMError>;

    /// Check if the client is available and configured
    fn is_available(&self) -> bool;

    /// Get the provider type
    fn provider(&self) -> LLMProvider;

    /// Get the current model being used
    fn current_model(&self) -> &str;

    /// Get supported models for this provider
    fn supported_models(&self) -> Vec<String>;
}

/// Client registry for multiple LLM providers
pub struct LLMClientRegistry {
    clients: HashMap<LLMProvider, Box<dyn LLMClient>>,
    default_provider: Option<LLMProvider>,
    prompts: PromptTemplates,
    config: LLMConfig,
}

impl LLMClientRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            default_provider: None,
            prompts: PromptTemplates::default(),
            config: LLMConfig::default(),
        }
    }

    /// Create a new registry with custom configuration
    pub fn with_config(config: LLMConfig) -> Self {
        Self {
            clients: HashMap::new(),
            default_provider: config.default_provider.clone(),
            prompts: PromptTemplates::default(),
            config,
        }
    }

    /// Initialize from environment variables
    ///
    /// Checks for API keys and configures available providers automatically.
    pub fn from_env() -> Self {
        let mut registry = Self::new();

        // Check for OpenAI
        if env::var("OPENAI_API_KEY").is_ok() {
            registry.register(Box::new(OpenAIClient::from_env()));
            if registry.default_provider.is_none() {
                registry.default_provider = Some(LLMProvider::OpenAI);
            }
        }

        // Check for Anthropic
        if env::var("ANTHROPIC_API_KEY").is_ok() {
            registry.register(Box::new(AnthropicClient::from_env()));
            if registry.default_provider.is_none() {
                registry.default_provider = Some(LLMProvider::Anthropic);
            }
        }

        // Check for Ollama (try to connect)
        if env::var("OLLAMA_HOST").is_ok() || Self::check_ollama_available() {
            let client = OllamaClient::from_env();
            if client.is_available() {
                registry.register(Box::new(client));
                if registry.default_provider.is_none() {
                    registry.default_provider = Some(LLMProvider::Ollama);
                }
            }
        }

        registry
    }

    /// Check if Ollama is running on default port
    fn check_ollama_available() -> bool {
        std::net::TcpStream::connect("127.0.0.1:11434").is_ok()
    }

    /// Register a client
    pub fn register(&mut self, client: Box<dyn LLMClient>) {
        let provider = client.provider();
        self.clients.insert(provider, client);
    }

    /// Get a specific client by provider
    pub fn get(&self, provider: &LLMProvider) -> Option<&dyn LLMClient> {
        self.clients.get(provider).map(|c| c.as_ref())
    }

    /// Get the default client
    pub fn default_client(&self) -> Result<&dyn LLMClient, LLMError> {
        self.default_provider
            .as_ref()
            .and_then(|p| self.get(p))
            .ok_or(LLMError::NoProvidersAvailable)
    }

    /// Check if any provider is available
    pub fn is_available(&self) -> bool {
        self.clients.values().any(|c| c.is_available())
    }

    /// Get all available providers
    pub fn available_providers(&self) -> Vec<LLMProvider> {
        self.clients
            .iter()
            .filter(|(_, c)| c.is_available())
            .map(|(p, _)| p.clone())
            .collect()
    }

    /// Get the prompt templates
    pub fn prompts(&self) -> &PromptTemplates {
        &self.prompts
    }

    /// Get mutable prompt templates for customization
    pub fn prompts_mut(&mut self) -> &mut PromptTemplates {
        &mut self.prompts
    }

    /// Get configuration
    pub fn config(&self) -> &LLMConfig {
        &self.config
    }

    /// Query with automatic fallback to other providers on failure
    pub fn query_with_fallback(&self, request: &LLMRequest) -> Result<LLMResponse, LLMError> {
        let mut last_error = LLMError::NoProvidersAvailable;

        // Try default provider first
        if let Some(provider) = &self.default_provider
            && let Some(client) = self.get(provider)
        {
            match client.query(request) {
                Ok(response) => return Ok(response),
                Err(e) => last_error = e,
            }
        }

        // Try other providers
        for (provider, client) in &self.clients {
            if Some(provider) == self.default_provider.as_ref() {
                continue; // Already tried
            }
            if !client.is_available() {
                continue;
            }
            match client.query(request) {
                Ok(response) => return Ok(response),
                Err(e) => last_error = e,
            }
        }

        Err(last_error)
    }
}

impl Default for LLMClientRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// OpenAI Client
// ============================================================================

/// OpenAI API client implementation
pub struct OpenAIClient {
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenAIClient {
    /// Create a new OpenAI client
    pub fn new(api_key: String, model: String, base_url: String) -> Self {
        Self {
            api_key,
            model,
            base_url,
        }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        Self {
            api_key: env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4-turbo".to_string()),
            base_url: env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
        }
    }

    /// Build the request body for OpenAI API
    fn build_request_body(&self, request: &LLMRequest) -> serde_json::Value {
        let mut messages = Vec::new();

        if let Some(system) = &request.system {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system
            }));
        }

        messages.push(serde_json::json!({
            "role": "user",
            "content": &request.prompt
        }));

        serde_json::json!({
            "model": &self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stop": request.stop_sequences
        })
    }
}

impl LLMClient for OpenAIClient {
    fn query(&self, request: &LLMRequest) -> Result<LLMResponse, LLMError> {
        if !self.is_available() {
            return Err(LLMError::NotConfigured(LLMProvider::OpenAI));
        }

        let body = self.build_request_body(request);
        let url = format!("{}/chat/completions", self.base_url);

        // Note: In a real implementation, this would use reqwest with the `llm` feature
        // For now, we provide a stub that can be enabled with the feature flag
        #[cfg(feature = "llm")]
        {
            let client = reqwest::blocking::Client::new();
            let response = client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .map_err(|e| LLMError::NetworkError(e.to_string()))?;

            if !response.status().is_success() {
                return Err(LLMError::ApiError {
                    status: response.status().as_u16(),
                    message: response.text().unwrap_or_default(),
                });
            }

            let json: serde_json::Value = response
                .json()
                .map_err(|e| LLMError::ParseError(e.to_string()))?;

            let content = json["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();

            let indicators = analyze_confidence(&content);

            Ok(LLMResponse {
                content,
                model: self.model.clone(),
                tokens_used: json["usage"]["total_tokens"].as_u64().unwrap_or(0) as usize,
                finish_reason: FinishReason::Stop,
                confidence_indicators: indicators,
            })
        }

        #[cfg(not(feature = "llm"))]
        {
            // Stub implementation when LLM feature is not enabled
            Err(LLMError::NotConfigured(LLMProvider::OpenAI))
        }
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    fn provider(&self) -> LLMProvider {
        LLMProvider::OpenAI
    }

    fn current_model(&self) -> &str {
        &self.model
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "gpt-4-turbo".to_string(),
            "gpt-4o".to_string(),
            "gpt-4o-mini".to_string(),
            "gpt-4".to_string(),
            "gpt-3.5-turbo".to_string(),
        ]
    }
}

// ============================================================================
// Anthropic Client
// ============================================================================

/// Anthropic API client implementation
pub struct AnthropicClient {
    api_key: String,
    model: String,
}

impl AnthropicClient {
    /// Create a new Anthropic client
    pub fn new(api_key: String, model: String) -> Self {
        Self { api_key, model }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        Self {
            api_key: env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            model: env::var("ANTHROPIC_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string()),
        }
    }

    /// Build the request body for Anthropic API
    fn build_request_body(&self, request: &LLMRequest) -> serde_json::Value {
        serde_json::json!({
            "model": &self.model,
            "max_tokens": request.max_tokens,
            "system": request.system.as_deref().unwrap_or(""),
            "messages": [
                {"role": "user", "content": &request.prompt}
            ]
        })
    }
}

impl LLMClient for AnthropicClient {
    fn query(&self, request: &LLMRequest) -> Result<LLMResponse, LLMError> {
        if !self.is_available() {
            return Err(LLMError::NotConfigured(LLMProvider::Anthropic));
        }

        let body = self.build_request_body(request);

        #[cfg(feature = "llm")]
        {
            let client = reqwest::blocking::Client::new();
            let response = client
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .map_err(|e| LLMError::NetworkError(e.to_string()))?;

            if !response.status().is_success() {
                return Err(LLMError::ApiError {
                    status: response.status().as_u16(),
                    message: response.text().unwrap_or_default(),
                });
            }

            let json: serde_json::Value = response
                .json()
                .map_err(|e| LLMError::ParseError(e.to_string()))?;

            let content = json["content"][0]["text"]
                .as_str()
                .unwrap_or("")
                .to_string();

            let indicators = analyze_confidence(&content);

            Ok(LLMResponse {
                content,
                model: self.model.clone(),
                tokens_used: json["usage"]["output_tokens"].as_u64().unwrap_or(0) as usize,
                finish_reason: FinishReason::Stop,
                confidence_indicators: indicators,
            })
        }

        #[cfg(not(feature = "llm"))]
        {
            Err(LLMError::NotConfigured(LLMProvider::Anthropic))
        }
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    fn provider(&self) -> LLMProvider {
        LLMProvider::Anthropic
    }

    fn current_model(&self) -> &str {
        &self.model
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "claude-sonnet-4-20250514".to_string(),
            "claude-3-5-sonnet-20241022".to_string(),
            "claude-3-opus-20240229".to_string(),
            "claude-3-sonnet-20240229".to_string(),
            "claude-3-haiku-20240307".to_string(),
        ]
    }
}

// ============================================================================
// Ollama Client (Local Models)
// ============================================================================

/// Ollama client for local models
pub struct OllamaClient {
    host: String,
    model: String,
}

impl OllamaClient {
    /// Create a new Ollama client
    pub fn new(host: String, model: String) -> Self {
        Self { host, model }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        Self {
            host: env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            model: env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.1:8b".to_string()),
        }
    }

    /// Build the request body for Ollama API
    fn build_request_body(&self, request: &LLMRequest) -> serde_json::Value {
        serde_json::json!({
            "model": &self.model,
            "prompt": &request.prompt,
            "system": request.system.as_deref().unwrap_or(""),
            "stream": false,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens
            }
        })
    }

    /// Extract host and port for connection check
    fn host_and_port(&self) -> Option<String> {
        let url = self.host.replace("http://", "").replace("https://", "");
        Some(url)
    }
}

impl LLMClient for OllamaClient {
    fn query(&self, request: &LLMRequest) -> Result<LLMResponse, LLMError> {
        if !self.is_available() {
            return Err(LLMError::NotConfigured(LLMProvider::Ollama));
        }

        let body = self.build_request_body(request);
        let url = format!("{}/api/generate", self.host);

        #[cfg(feature = "llm")]
        {
            let client = reqwest::blocking::Client::new();
            let response = client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .map_err(|e| LLMError::NetworkError(e.to_string()))?;

            if !response.status().is_success() {
                return Err(LLMError::ApiError {
                    status: response.status().as_u16(),
                    message: response.text().unwrap_or_default(),
                });
            }

            let json: serde_json::Value = response
                .json()
                .map_err(|e| LLMError::ParseError(e.to_string()))?;

            let content = json["response"].as_str().unwrap_or("").to_string();
            let indicators = analyze_confidence(&content);

            Ok(LLMResponse {
                content,
                model: self.model.clone(),
                tokens_used: 0, // Ollama doesn't always report this
                finish_reason: FinishReason::Stop,
                confidence_indicators: indicators,
            })
        }

        #[cfg(not(feature = "llm"))]
        {
            Err(LLMError::NotConfigured(LLMProvider::Ollama))
        }
    }

    fn is_available(&self) -> bool {
        if let Some(addr) = self.host_and_port() {
            std::net::TcpStream::connect(&addr).is_ok()
        } else {
            false
        }
    }

    fn provider(&self) -> LLMProvider {
        LLMProvider::Ollama
    }

    fn current_model(&self) -> &str {
        &self.model
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "llama3.1:8b".to_string(),
            "llama3.1:70b".to_string(),
            "llama3.2:3b".to_string(),
            "mistral:7b".to_string(),
            "codellama:13b".to_string(),
            "codellama:34b".to_string(),
            "deepseek-coder:6.7b".to_string(),
            "phi3:mini".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_client_creation() {
        let client = OpenAIClient::new(
            "test-key".to_string(),
            "gpt-4o".to_string(),
            "https://api.openai.com/v1".to_string(),
        );
        assert!(client.is_available());
        assert_eq!(client.current_model(), "gpt-4o");
        assert_eq!(client.provider(), LLMProvider::OpenAI);
    }

    #[test]
    fn test_anthropic_client_creation() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "claude-sonnet-4-20250514".to_string(),
        );
        assert!(client.is_available());
        assert_eq!(client.current_model(), "claude-sonnet-4-20250514");
        assert_eq!(client.provider(), LLMProvider::Anthropic);
    }

    #[test]
    fn test_ollama_client_creation() {
        let client = OllamaClient::new(
            "http://localhost:11434".to_string(),
            "llama3.1:8b".to_string(),
        );
        assert_eq!(client.current_model(), "llama3.1:8b");
        assert_eq!(client.provider(), LLMProvider::Ollama);
    }

    #[test]
    fn test_registry_empty() {
        let registry = LLMClientRegistry::new();
        assert!(!registry.is_available());
        assert!(registry.default_client().is_err());
    }

    #[test]
    fn test_registry_with_client() {
        let mut registry = LLMClientRegistry::new();

        // Create a client directly
        let client = OpenAIClient::new(
            "test-key".to_string(),
            "gpt-4".to_string(),
            "https://api.openai.com/v1".to_string(),
        );
        registry.register(Box::new(client));
        registry.default_provider = Some(LLMProvider::OpenAI);

        assert!(registry.is_available());
        assert!(registry.default_client().is_ok());
    }

    #[test]
    fn test_supported_models() {
        let openai = OpenAIClient::from_env();
        let models = openai.supported_models();
        assert!(models.contains(&"gpt-4-turbo".to_string()));
        assert!(models.contains(&"gpt-4o".to_string()));

        let anthropic = AnthropicClient::from_env();
        let models = anthropic.supported_models();
        assert!(models.contains(&"claude-sonnet-4-20250514".to_string()));

        let ollama = OllamaClient::from_env();
        let models = ollama.supported_models();
        assert!(models.contains(&"llama3.1:8b".to_string()));
    }
}
