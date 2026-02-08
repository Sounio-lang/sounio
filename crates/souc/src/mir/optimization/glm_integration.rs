//! GLM-4.7 Integration for ML-Guided Optimization
//!
//! This module integrates GLM-4.7 (a large language model) into the Sounio compiler
//! to make intelligent optimization decisions based on code analysis.
//!
//! Types are shared with the local heuristic optimizer via `optimization_types`.

use super::optimization_types::{
    BlockFeatures, CodeFeatures, OptimizationSuggestion, OptimizationType,
};
use crate::mir::MirModule;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GLM-4.7 API Configuration
#[derive(Debug, Clone)]
pub struct GLMConfig {
    /// API endpoint URL
    pub api_url: String,
    /// API key for authentication
    pub api_key: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for randomness (0.0-1.0)
    pub temperature: f32,
    /// Timeout for API calls
    pub timeout_secs: u64,
}

impl Default for GLMConfig {
    fn default() -> Self {
        Self {
            api_url: "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions".to_string(),
            api_key: std::env::var("GLM_API_KEY").unwrap_or_default(),
            max_tokens: 1000,
            temperature: 0.1,
            timeout_secs: 30,
        }
    }
}

/// GLM-4.7 Integration Manager
pub struct GLMManager {
    config: GLMConfig,
    #[cfg(feature = "glm")]
    client: reqwest::Client,
    cache: HashMap<String, OptimizationSuggestion>,
}

impl GLMManager {
    /// Create a new GLM manager
    pub fn new(config: GLMConfig) -> Self {
        #[cfg(feature = "glm")]
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            #[cfg(feature = "glm")]
            client,
            cache: HashMap::new(),
        }
    }

    /// Analyze code and get optimization suggestions
    #[cfg(feature = "glm")]
    pub async fn analyze_and_suggest(
        &mut self,
        module: &MirModule,
        function_name: &str,
    ) -> Result<Vec<OptimizationSuggestion>, String> {
        // Generate cache key
        let cache_key = self.generate_cache_key(module, function_name);

        // Check cache first
        if let Some(suggestions) = self.cache.get(&cache_key) {
            return Ok(vec![suggestions.clone()]);
        }

        // Extract code features using shared extraction
        let features = super::optimization_types::extract_features(module, function_name);

        // Query GLM-4.7
        let suggestions = self.query_glm(&features).await?;

        // Cache results
        if !suggestions.is_empty() {
            self.cache.insert(cache_key, suggestions[0].clone());
        }

        Ok(suggestions)
    }

    /// Query GLM-4.7 API for optimization suggestions
    #[cfg(feature = "glm")]
    async fn query_glm(
        &self,
        features: &CodeFeatures,
    ) -> Result<Vec<OptimizationSuggestion>, String> {
        let prompt = self.build_prompt(features);

        let request_body = serde_json::json!({
            "model": "glm-4.7",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert compiler optimization assistant. Analyze the given code features and suggest specific optimizations for a scientific computing language with epistemic types (Knowledge<T>)."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        });

        let response = self
            .client
            .post(&self.config.api_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Failed to query GLM API: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("GLM API error {}: {}", status, body));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse GLM response: {}", e))?;

        self.parse_response(response_json)
    }

    /// Build prompt for GLM-4.7
    fn build_prompt(&self, features: &CodeFeatures) -> String {
        format!(
            r#"Analyze this code and suggest optimizations:

Code Features:
- Functions: {}
- Total Blocks: {}
- Total Instructions: {}
- Average Block Size: {:.2}
- Max Block Size: {}
- Arithmetic Operations: {}
- Memory Operations: {}
- Call Operations: {}
- Branches: {}

Block-level details:
{}

Type Distribution:
{:#?}

Provide 1-3 specific optimization suggestions in JSON format:
{{
    "suggestions": [
        {{
            "optimization_type": "ConstantPropagation|DeadCodeElimination|FunctionInlining|LoopUnrolling|StrengthReduction|CommonSubexpressionElimination|LoopInvariantCodeMotion|UncertaintyAwareOptimization",
            "confidence": 0.0-1.0,
            "target": "function_name or block_label",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "reasoning": "brief explanation"
        }}
    ]
}}"#,
            features.function_count,
            features.total_blocks,
            features.total_instructions,
            features.avg_block_size,
            features.max_block_size,
            features.arithmetic_ops,
            features.memory_ops,
            features.call_count,
            features.branch_count,
            serde_json::to_string_pretty(&features.block_features).unwrap_or_default(),
            serde_json::to_string_pretty(&features.type_distribution).unwrap_or_default()
        )
    }

    /// Parse GLM response
    fn parse_response(
        &self,
        response: serde_json::Value,
    ) -> Result<Vec<OptimizationSuggestion>, String> {
        let content = response
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(|content| content.as_str())
            .ok_or("Invalid response format")?;

        // Try to parse JSON from response
        let json_start = content.find('{');
        let json_end = content.rfind('}');

        if let (Some(start), Some(end)) = (json_start, json_end) {
            let json_str = &content[start..=end];
            let parsed: serde_json::Value = serde_json::from_str(json_str)
                .map_err(|e| format!("Failed to parse suggestions JSON: {}", e))?;

            let suggestions_json = parsed
                .get("suggestions")
                .and_then(|s| s.as_array())
                .ok_or("No suggestions found")?;

            let mut suggestions = Vec::new();
            for suggestion_json in suggestions_json {
                let suggestion: OptimizationSuggestion =
                    serde_json::from_value(suggestion_json.clone())
                        .map_err(|e| format!("Failed to parse suggestion: {}", e))?;
                suggestions.push(suggestion);
            }

            Ok(suggestions)
        } else {
            Err("No JSON found in response".to_string())
        }
    }

    /// Generate cache key for function analysis
    fn generate_cache_key(&self, module: &MirModule, function_name: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        module.hash(&mut hasher);
        function_name.hash(&mut hasher);
        let hash = hasher.finish();

        format!("{}_{:x}", function_name, hash)
    }

    /// Clear cache (useful for testing)
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Mock method for when GLM feature is disabled
    #[cfg(not(feature = "glm"))]
    pub async fn analyze_and_suggest(
        &mut self,
        _module: &MirModule,
        _function_name: &str,
    ) -> Result<Vec<OptimizationSuggestion>, String> {
        // Return mock suggestions when GLM is not available
        let suggestion = OptimizationSuggestion {
            optimization_type: OptimizationType::ConstantPropagation,
            confidence: 0.8,
            target: "test_function".to_string(),
            parameters: HashMap::new(),
            reasoning: "Mock suggestion for testing".to_string(),
        };
        Ok(vec![suggestion])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glm_config_default() {
        let config = GLMConfig::default();
        // API key comes from GLM_API_KEY env var; empty when unset
        assert_eq!(
            config.api_key,
            std::env::var("GLM_API_KEY").unwrap_or_default()
        );
        assert_eq!(config.max_tokens, 1000);
        assert_eq!(config.temperature, 0.1);
    }

    #[test]
    fn test_optimization_suggestion_serialization() {
        let suggestion = OptimizationSuggestion {
            optimization_type: OptimizationType::ConstantPropagation,
            confidence: 0.95,
            target: "test_function".to_string(),
            parameters: HashMap::from([("threshold".to_string(), "0.8".to_string())]),
            reasoning: "High confidence constant propagation opportunity".to_string(),
        };

        let json = serde_json::to_string(&suggestion).unwrap();
        let parsed: OptimizationSuggestion = serde_json::from_str(&json).unwrap();

        assert_eq!(
            parsed.optimization_type,
            OptimizationType::ConstantPropagation
        );
        assert_eq!(parsed.confidence, 0.95);
    }
}
