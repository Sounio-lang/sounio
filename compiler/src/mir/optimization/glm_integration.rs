//! GLM-4.7 Integration for ML-Guided Optimization
//!
//! This module integrates GLM-4.7 (a large language model) into the Sounio compiler
//! to make intelligent optimization decisions based on code analysis.

use crate::mir::{MirModule, MirFunction, MirBlock};
use crate::mir::instructions::{MirInstruction, MirBinaryOp};
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
            api_url: "https://open.bigmodel.cn/api/paas/v4/chat/completions".to_string(),
            api_key: std::env::var("GLM_API_KEY")
                .unwrap_or_else(|_| "622f603bf3a04a6c91b967d33231df34.BiTCkvs9VxeAywva".to_string()),
            max_tokens: 1000,
            temperature: 0.1,
            timeout_secs: 30,
        }
    }
}

/// Code analysis features for ML model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFeatures {
    /// Function-level features
    pub function_count: usize,
    pub total_blocks: usize,
    pub total_instructions: usize,
    pub avg_block_size: f64,
    pub max_block_size: usize,
    pub loop_count: usize,
    pub branch_count: usize,
    pub call_count: usize,
    pub arithmetic_ops: usize,
    pub memory_ops: usize,
    
    /// Block-level features
    pub block_features: Vec<BlockFeatures>,
    
    /// Type information
    pub type_distribution: HashMap<String, usize>,
    pub epistemic_types: usize, // Knowledge<T> types
    pub uncertainty_ops: usize, // Operations involving uncertainty
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockFeatures {
    pub instruction_count: usize,
    pub arithmetic_ops: usize,
    pub memory_loads: usize,
    pub memory_stores: usize,
    pub branches: usize,
    pub phi_nodes: usize,
    pub loop_depth: usize,
    pub has_uncertainty: bool,
}

/// Optimization suggestion from GLM-4.7
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Type of optimization recommended
    pub optimization_type: OptimizationType,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Target (function name, block label, etc.)
    pub target: String,
    /// Parameters for the optimization
    pub parameters: HashMap<String, String>,
    /// Reasoning from the model
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationType {
    ConstantPropagation,
    DeadCodeElimination,
    FunctionInlining,
    LoopUnrolling,
    StrengthReduction,
    CommonSubexpressionElimination,
    LoopInvariantCodeMotion,
    AliasAnalysis,
    ScalarReplacementOfAggregates,
    // ML-specific optimizations
    PredictiveInlining,
    AdaptiveUnrolling,
    UncertaintyAwareOptimization,
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

        // Extract code features
        let features = self.extract_features(module, function_name);
        
        // Query GLM-4.7
        let suggestions = self.query_glm(&features).await?;
        
        // Cache results
        if !suggestions.is_empty() {
            self.cache.insert(cache_key, suggestions[0].clone());
        }

        Ok(suggestions)
    }

    /// Extract relevant features from MIR for ML analysis
    fn extract_features(&self, module: &MirModule, function_name: &str) -> CodeFeatures {
        let mut features = CodeFeatures {
            function_count: module.functions.len(),
            total_blocks: 0,
            total_instructions: 0,
            avg_block_size: 0.0,
            max_block_size: 0,
            loop_count: 0,
            branch_count: 0,
            call_count: 0,
            arithmetic_ops: 0,
            memory_ops: 0,
            block_features: Vec::new(),
            type_distribution: HashMap::new(),
            epistemic_types: 0,
            uncertainty_ops: 0,
        };

        // Find target function
        if let Some(func) = module.functions.iter().find(|f| f.name == function_name) {
            for block in &func.blocks {
                let block_features = self.analyze_block(block);
                features.block_features.push(block_features.clone());
                
                features.total_blocks += 1;
                features.total_instructions += block.instructions.len();
                features.max_block_size = features.max_block_size.max(block.instructions.len());
                
                // Count operations
                for inst in &block.instructions {
                    match inst {
                        MirInstruction::Binary { op, .. } => {
                            if op.is_arithmetic() {
                                features.arithmetic_ops += 1;
                            }
                        }
                        MirInstruction::Load { .. } | MirInstruction::Store { .. } => features.memory_ops += 1,
                        MirInstruction::Call { .. } => features.call_count += 1,
                        _ => {}
                    }
                }
            }
            
            features.avg_block_size = features.total_instructions as f64 / features.total_blocks.max(1) as f64;
        }

        features
    }

    /// Analyze a single block
    fn analyze_block(&self, block: &MirBlock) -> BlockFeatures {
        let mut features = BlockFeatures {
            instruction_count: block.instructions.len(),
            arithmetic_ops: 0,
            memory_loads: 0,
            memory_stores: 0,
            branches: 0,
            phi_nodes: 0,
            loop_depth: 0,
            has_uncertainty: false,
        };

        for inst in &block.instructions {
            match inst {
                MirInstruction::Binary { op, .. } => {
                    if op.is_arithmetic() {
                        features.arithmetic_ops += 1;
                    }
                }
                MirInstruction::Load { .. } => features.memory_loads += 1,
                MirInstruction::Store { .. } => features.memory_stores += 1,
                MirInstruction::Phi { .. } => features.phi_nodes += 1,
                _ => {}
            }
        }

        features
    }

    /// Query GLM-4.7 API for optimization suggestions
    #[cfg(feature = "glm")]
    async fn query_glm(
        &self,
        features: &CodeFeatures,
    ) -> Result<Vec<OptimizationSuggestion>, String> {
        let prompt = self.build_prompt(features);
        
        let request_body = serde_json::json!({
            "model": "glm-4",
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

        let response = self.client
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
    fn parse_response(&self, response: serde_json::Value) -> Result<Vec<OptimizationSuggestion>, String> {
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

            let suggestions_json = parsed.get("suggestions")
                .and_then(|s| s.as_array())
                .ok_or("No suggestions found")?;

            let mut suggestions = Vec::new();
            for suggestion_json in suggestions_json {
                let suggestion: OptimizationSuggestion = serde_json::from_value(suggestion_json.clone())
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
        assert!(!config.api_key.is_empty());
        assert_eq!(config.max_tokens, 1000);
        assert_eq!(config.temperature, 0.1);
    }

    #[test]
    fn test_optimization_suggestion_serialization() {
        let suggestion = OptimizationSuggestion {
            optimization_type: OptimizationType::ConstantPropagation,
            confidence: 0.95,
            target: "test_function".to_string(),
            parameters: HashMap::from([
                ("threshold".to_string(), "0.8".to_string())
            ]),
            reasoning: "High confidence constant propagation opportunity".to_string(),
        };

        let json = serde_json::to_string(&suggestion).unwrap();
        let parsed: OptimizationSuggestion = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.optimization_type, OptimizationType::ConstantPropagation);
        assert_eq!(parsed.confidence, 0.95);
    }
}