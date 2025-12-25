//! Confidence Estimation from LLM Responses
//!
//! Based on research in epistemic integrity of LLMs (arXiv:2411.06528), this module
//! detects linguistic markers that indicate the LLM's internal certainty level.
//!
//! # Key Insight
//!
//! LLMs exhibit "epistemic miscalibration" - their linguistic assertiveness often
//! doesn't match their internal certainty. By analyzing hedging phrases, certainty
//! markers, and uncertainty indicators, we can better estimate true confidence.
//!
//! # Confidence Indicators
//!
//! | Indicator Type | Example Phrases | Effect on Confidence |
//! |----------------|-----------------|---------------------|
//! | Hedging | "might", "possibly", "perhaps" | Slight decrease |
//! | Certainty | "definitely", "always", "must" | Increase (with caution) |
//! | Uncertainty | "I'm not sure", "unclear" | Strong decrease |
//! | Source refs | "according to", "based on" | Increase |
//!
//! # Usage
//!
//! ```rust,ignore
//! use sounio::llm::confidence::{analyze_confidence, indicators_to_confidence};
//!
//! let text = "A scaffold might be a material entity. This seems consistent with BFO.";
//! let indicators = analyze_confidence(text);
//! let confidence = indicators_to_confidence(&indicators);
//! assert!(confidence < 0.8); // Hedging detected
//! ```

use std::fmt;

/// Indicators used to estimate epistemic confidence from LLM text
///
/// These are extracted from the LLM's natural language response and
/// used to estimate how confident we should be in the answer.
#[derive(Debug, Clone, Default)]
pub struct ConfidenceIndicators {
    /// Count of hedging phrases detected ("might", "possibly", "could be")
    ///
    /// Hedging often indicates appropriate epistemic humility and suggests
    /// the LLM is uncertain about its answer.
    pub hedge_count: usize,

    /// Count of certainty phrases detected ("definitely", "certainly", "always")
    ///
    /// High certainty language may indicate overconfidence. We apply
    /// diminishing returns to avoid rewarding excessive assertiveness.
    pub certainty_count: usize,

    /// Count of explicit uncertainty markers ("I'm not sure", "unclear")
    ///
    /// These are strong negative signals indicating the LLM knows it doesn't know.
    pub uncertainty_markers: usize,

    /// Count of references to external sources ("according to", "based on")
    ///
    /// Source references indicate grounded reasoning and increase confidence.
    pub source_references: usize,

    /// Count of reasoning steps or logical connectives ("therefore", "because")
    ///
    /// Chain-of-thought reasoning typically produces more reliable outputs.
    pub reasoning_markers: usize,

    /// Whether the response contains structured output (JSON, etc.)
    ///
    /// Structured outputs are easier to validate and parse correctly.
    pub has_structured_output: bool,

    /// Sentiment: -1.0 (very negative) to 1.0 (very positive)
    ///
    /// Extracted from emotional language in the response.
    pub sentiment: f64,
}

impl ConfidenceIndicators {
    /// Create empty indicators
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert to estimated confidence value
    pub fn to_confidence(&self) -> f64 {
        indicators_to_confidence(self)
    }

    /// Check if indicators suggest high uncertainty
    pub fn is_uncertain(&self) -> bool {
        self.uncertainty_markers > 0 || self.hedge_count > 3
    }

    /// Check if indicators suggest overconfidence
    pub fn may_be_overconfident(&self) -> bool {
        self.certainty_count > 3 && self.source_references == 0
    }

    /// Total linguistic markers found
    pub fn total_markers(&self) -> usize {
        self.hedge_count
            + self.certainty_count
            + self.uncertainty_markers
            + self.source_references
            + self.reasoning_markers
    }
}

impl fmt::Display for ConfidenceIndicators {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConfidenceIndicators(hedges={}, certainty={}, uncertain={}, sources={}, reasoning={})",
            self.hedge_count,
            self.certainty_count,
            self.uncertainty_markers,
            self.source_references,
            self.reasoning_markers
        )
    }
}

// Linguistic patterns for confidence estimation
// These are carefully curated based on epistemic integrity research

/// Hedging phrases that indicate uncertainty
const HEDGE_PHRASES: &[&str] = &[
    "might",
    "may",
    "could",
    "possibly",
    "perhaps",
    "probably",
    "likely",
    "potentially",
    "presumably",
    "seemingly",
    "apparently",
    "seems",
    "appear",
    "suggests",
    "indicates",
    "implies",
    "i think",
    "i believe",
    "in my opinion",
    "it seems",
    "it appears",
    "arguably",
    "ostensibly",
    "roughly",
    "approximately",
    "around",
    "about",
    "more or less",
];

/// Certainty phrases that indicate high confidence
const CERTAINTY_PHRASES: &[&str] = &[
    "definitely",
    "certainly",
    "absolutely",
    "always",
    "never",
    "must",
    "clearly",
    "obviously",
    "undoubtedly",
    "unquestionably",
    "invariably",
    "without doubt",
    "for certain",
    "proven",
    "established",
    "well-known",
    "standard",
    "fundamental",
    "essential",
    "necessarily",
    "is a",
    "are",
    "will be",
    "cannot",
];

/// Strong uncertainty markers
const UNCERTAINTY_MARKERS: &[&str] = &[
    "not sure",
    "uncertain",
    "unknown",
    "unclear",
    "ambiguous",
    "don't know",
    "cannot determine",
    "hard to say",
    "difficult to",
    "impossible to tell",
    "no way to know",
    "beyond my knowledge",
    "outside my expertise",
    "i lack",
    "insufficient information",
    "need more context",
    "?",
];

/// Source reference patterns
const SOURCE_PATTERNS: &[&str] = &[
    "according to",
    "based on",
    "as defined in",
    "as stated in",
    "per",
    "following",
    "from the",
    "in reference to",
    "citing",
    "referenced in",
    "documented in",
    "specified in",
    "defined by",
    "described in",
];

/// Reasoning markers indicating chain-of-thought
const REASONING_MARKERS: &[&str] = &[
    "therefore",
    "because",
    "since",
    "thus",
    "hence",
    "consequently",
    "as a result",
    "this means",
    "it follows",
    "we can conclude",
    "this implies",
    "given that",
    "considering",
    "taking into account",
    "first",
    "second",
    "third",
    "finally",
    "step",
    "next",
];

/// Analyze LLM response text for confidence indicators
///
/// Scans the text for linguistic patterns that suggest the LLM's
/// internal certainty level about its response.
pub fn analyze_confidence(text: &str) -> ConfidenceIndicators {
    let text_lower = text.to_lowercase();

    let mut indicators = ConfidenceIndicators::new();

    // Count hedging phrases
    for phrase in HEDGE_PHRASES {
        indicators.hedge_count += count_phrase(&text_lower, phrase);
    }

    // Count certainty phrases
    for phrase in CERTAINTY_PHRASES {
        indicators.certainty_count += count_phrase(&text_lower, phrase);
    }

    // Count uncertainty markers
    for marker in UNCERTAINTY_MARKERS {
        indicators.uncertainty_markers += count_phrase(&text_lower, marker);
    }

    // Count source references
    for pattern in SOURCE_PATTERNS {
        indicators.source_references += count_phrase(&text_lower, pattern);
    }

    // Count reasoning markers
    for marker in REASONING_MARKERS {
        indicators.reasoning_markers += count_phrase(&text_lower, marker);
    }

    // Check for structured output
    indicators.has_structured_output = text.contains('{') && text.contains('}');

    // Simple sentiment analysis (positive vs negative words)
    indicators.sentiment = analyze_sentiment(&text_lower);

    indicators
}

/// Count occurrences of a phrase in text (word-boundary aware for short phrases)
fn count_phrase(text: &str, phrase: &str) -> usize {
    if phrase.len() <= 3 {
        // For short phrases, require word boundaries
        text.split_whitespace()
            .filter(|word| {
                let clean: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                clean == phrase
            })
            .count()
    } else {
        // For longer phrases, use simple substring matching
        text.matches(phrase).count()
    }
}

/// Simple sentiment analysis (-1.0 to 1.0)
fn analyze_sentiment(text: &str) -> f64 {
    const POSITIVE_WORDS: &[&str] = &[
        "good",
        "great",
        "excellent",
        "correct",
        "accurate",
        "valid",
        "consistent",
        "appropriate",
        "proper",
        "suitable",
        "well-formed",
        "complete",
        "comprehensive",
    ];

    const NEGATIVE_WORDS: &[&str] = &[
        "bad",
        "wrong",
        "incorrect",
        "invalid",
        "inconsistent",
        "error",
        "mistake",
        "problem",
        "issue",
        "fail",
        "broken",
        "incomplete",
        "missing",
    ];

    let positive_count: i32 = POSITIVE_WORDS
        .iter()
        .map(|w| text.matches(w).count() as i32)
        .sum();

    let negative_count: i32 = NEGATIVE_WORDS
        .iter()
        .map(|w| text.matches(w).count() as i32)
        .sum();

    let total = positive_count + negative_count;
    if total == 0 {
        0.0
    } else {
        (positive_count - negative_count) as f64 / total as f64
    }
}

/// Convert confidence indicators to an epistemic confidence value (0.0-1.0)
///
/// This formula is based on empirical calibration from epistemic integrity research:
///
/// - **Base confidence**: 0.60 (conservative baseline for LLM claims)
/// - **Hedges**: Reduce confidence (appropriate epistemic humility)
/// - **Certainty**: Increase with diminishing returns (to avoid rewarding overconfidence)
/// - **Uncertainty markers**: Strong negative signal
/// - **Source references**: Increase confidence (grounded reasoning)
/// - **Reasoning markers**: Slight increase (CoT improves accuracy)
/// - **Structured output**: Slight increase (easier to validate)
pub fn indicators_to_confidence(indicators: &ConfidenceIndicators) -> f64 {
    const BASE_CONFIDENCE: f64 = 0.60;

    // Each hedge reduces confidence by 3%, up to 30% total reduction
    let hedge_factor = 1.0 - (0.03 * indicators.hedge_count.min(10) as f64);

    // Certainty increases confidence, but with logarithmic diminishing returns
    // This prevents overconfident responses from getting unreasonably high scores
    let certainty_bonus = if indicators.certainty_count > 0 {
        0.05 * (indicators.certainty_count as f64).ln_1p()
    } else {
        0.0
    };
    let certainty_factor = 1.0 + certainty_bonus.min(0.15);

    // Uncertainty markers are very strong negative signals (15% each)
    let uncertainty_factor = 1.0 - (0.15 * indicators.uncertainty_markers.min(5) as f64);

    // Source references increase confidence (5% each, up to 25%)
    let source_factor = 1.0 + (0.05 * indicators.source_references.min(5) as f64);

    // Reasoning markers slightly increase confidence (2% each, up to 20%)
    let reasoning_factor = 1.0 + (0.02 * indicators.reasoning_markers.min(10) as f64);

    // Structured output bonus (5%)
    let structure_factor = if indicators.has_structured_output {
        1.05
    } else {
        1.0
    };

    // Combine all factors
    let confidence = BASE_CONFIDENCE
        * hedge_factor
        * certainty_factor
        * uncertainty_factor
        * source_factor
        * reasoning_factor
        * structure_factor;

    // Clamp to valid range
    // Never go below 0.1 (some information is always conveyed)
    // Never go above 0.95 (LLM outputs should never be fully trusted)
    confidence.clamp(0.10, 0.95)
}

/// Quick confidence check without full analysis
pub fn quick_confidence_check(text: &str) -> f64 {
    let indicators = analyze_confidence(text);
    indicators_to_confidence(&indicators)
}

/// Confidence level categories for display
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLevel {
    /// Very low confidence (< 0.3)
    VeryLow,
    /// Low confidence (0.3 - 0.5)
    Low,
    /// Moderate confidence (0.5 - 0.7)
    Moderate,
    /// High confidence (0.7 - 0.85)
    High,
    /// Very high confidence (> 0.85)
    VeryHigh,
}

impl ConfidenceLevel {
    /// Get confidence level from a numeric value
    pub fn from_value(confidence: f64) -> Self {
        if confidence < 0.3 {
            ConfidenceLevel::VeryLow
        } else if confidence < 0.5 {
            ConfidenceLevel::Low
        } else if confidence < 0.7 {
            ConfidenceLevel::Moderate
        } else if confidence < 0.85 {
            ConfidenceLevel::High
        } else {
            ConfidenceLevel::VeryHigh
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            ConfidenceLevel::VeryLow => "Very low confidence - manual verification required",
            ConfidenceLevel::Low => "Low confidence - review recommended",
            ConfidenceLevel::Moderate => "Moderate confidence - some uncertainty",
            ConfidenceLevel::High => "High confidence - generally reliable",
            ConfidenceLevel::VeryHigh => "Very high confidence - well-supported",
        }
    }
}

impl fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfidenceLevel::VeryLow => write!(f, "very low"),
            ConfidenceLevel::Low => write!(f, "low"),
            ConfidenceLevel::Moderate => write!(f, "moderate"),
            ConfidenceLevel::High => write!(f, "high"),
            ConfidenceLevel::VeryHigh => write!(f, "very high"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hedge_detection() {
        let text = "This might possibly be a scaffold, but I'm not sure about that.";
        let indicators = analyze_confidence(text);

        assert!(indicators.hedge_count >= 2); // "might", "possibly"
        assert!(indicators.uncertainty_markers >= 1); // "not sure"
    }

    #[test]
    fn test_certainty_detection() {
        let text = "A scaffold is definitely a material entity. This is a well-established fact.";
        let indicators = analyze_confidence(text);

        assert!(indicators.certainty_count >= 2); // "definitely", "well-established"
    }

    #[test]
    fn test_source_reference_detection() {
        let text = "According to BFO, and based on the OBO Foundry guidelines, this is correct.";
        let indicators = analyze_confidence(text);

        assert!(indicators.source_references >= 2); // "according to", "based on"
    }

    #[test]
    fn test_reasoning_marker_detection() {
        let text = "First, we identify the entity. Therefore, since it has mass, we can conclude it's material.";
        let indicators = analyze_confidence(text);

        assert!(indicators.reasoning_markers >= 3); // "first", "therefore", "since", "conclude"
    }

    #[test]
    fn test_structured_output_detection() {
        let text = r#"{"term": "scaffold", "type": "material entity"}"#;
        let indicators = analyze_confidence(text);

        assert!(indicators.has_structured_output);
    }

    #[test]
    fn test_high_confidence_calculation() {
        let indicators = ConfidenceIndicators {
            hedge_count: 0,
            certainty_count: 3,
            uncertainty_markers: 0,
            source_references: 2,
            reasoning_markers: 4,
            has_structured_output: true,
            sentiment: 0.5,
        };

        let confidence = indicators_to_confidence(&indicators);
        assert!(confidence > 0.75); // High confidence with positive signals
        assert!(confidence <= 0.95); // Never exceeds cap
    }

    #[test]
    fn test_low_confidence_calculation() {
        let indicators = ConfidenceIndicators {
            hedge_count: 5,
            certainty_count: 0,
            uncertainty_markers: 2,
            source_references: 0,
            reasoning_markers: 0,
            has_structured_output: false,
            sentiment: -0.3,
        };

        let confidence = indicators_to_confidence(&indicators);
        assert!(confidence < 0.5);
        assert!(confidence >= 0.10); // Never below floor
    }

    #[test]
    fn test_confidence_level_from_value() {
        assert_eq!(ConfidenceLevel::from_value(0.1), ConfidenceLevel::VeryLow);
        assert_eq!(ConfidenceLevel::from_value(0.4), ConfidenceLevel::Low);
        assert_eq!(ConfidenceLevel::from_value(0.6), ConfidenceLevel::Moderate);
        assert_eq!(ConfidenceLevel::from_value(0.75), ConfidenceLevel::High);
        assert_eq!(ConfidenceLevel::from_value(0.9), ConfidenceLevel::VeryHigh);
    }

    #[test]
    fn test_real_llm_response() {
        // Simulated LLM response for term typing
        let response = r#"
        Based on the BFO ontology, a scaffold is classified as a material entity (BFO:0000040).

        Here's my reasoning:
        1. First, a scaffold has spatial extent and physical mass
        2. Therefore, it persists through time while maintaining identity
        3. This means it's an independent continuant
        4. Since it can exist on its own, it's a material entity

        {"term": "scaffold", "bfo_category": "BFO:0000040", "confidence": 0.92}
        "#;

        let indicators = analyze_confidence(response);
        let confidence = indicators_to_confidence(&indicators);

        // Should have moderate-high confidence due to:
        // - Source references ("Based on")
        // - Reasoning markers ("First", "Therefore", "This means", "Since")
        // - Structured output (JSON)
        // - Low hedging
        assert!(confidence > 0.65); // Moderate-high confidence
        assert!(indicators.reasoning_markers >= 3);
        assert!(indicators.source_references >= 1);
        assert!(indicators.has_structured_output);
    }

    #[test]
    fn test_uncertain_response() {
        let response = r#"
        I'm not sure about this classification. The term might be a process or possibly
        a material entity. It's unclear from the context. I would need more information
        to make a determination.
        "#;

        let indicators = analyze_confidence(response);
        let confidence = indicators_to_confidence(&indicators);

        assert!(confidence < 0.5);
        assert!(indicators.is_uncertain());
        assert!(indicators.hedge_count >= 2);
        assert!(indicators.uncertainty_markers >= 2);
    }

    #[test]
    fn test_overconfident_response() {
        let response = r#"
        This is definitely, absolutely, certainly a material entity. There is no doubt
        whatsoever. It is obviously and unquestionably the correct classification.
        "#;

        let indicators = analyze_confidence(response);

        assert!(indicators.may_be_overconfident());
        assert!(indicators.certainty_count >= 4);
        assert_eq!(indicators.source_references, 0);

        // Even with high certainty language, confidence should be capped
        let confidence = indicators_to_confidence(&indicators);
        assert!(confidence <= 0.95);
    }

    #[test]
    fn test_sentiment_analysis() {
        let positive = "This is a good, correct, and valid classification.";
        let negative = "This is wrong, incorrect, and contains errors.";
        let neutral = "The term is classified as a continuant.";

        let pos_indicators = analyze_confidence(positive);
        let neg_indicators = analyze_confidence(negative);
        let neu_indicators = analyze_confidence(neutral);

        assert!(pos_indicators.sentiment > 0.0);
        assert!(neg_indicators.sentiment < 0.0);
        // Neutral text should have no positive or negative words
        assert!(
            neu_indicators.sentiment.abs() < 0.1,
            "Neutral sentiment should be near zero, got {}",
            neu_indicators.sentiment
        );
    }
}
