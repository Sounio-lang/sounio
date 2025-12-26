//! REPL - Read-Eval-Print Loop for Sounio
//!
//! Provides an interactive shell for evaluating Sounio expressions and statements.
//!
//! # Epistemic Features
//!
//! The REPL provides rich visualization of epistemic values:
//! - Knowledge<T> with confidence badges and provenance
//! - Uncertain values with mean ± std display
//! - Confidence bars for visual indication
//! - Provenance chain exploration
//!
//! # Commands
//!
//! - `:help` - Show help
//! - `:type <expr>` - Show type of expression
//! - `:provenance <var>` - Show provenance chain
//! - `:uncertainty <var>` - Show uncertainty details
//! - `:confidence <var>` - Show confidence level
//! - `:info <var>` - Show full epistemic info
//! - `:env` - Show current bindings
//! - `:funcs` - Show defined functions

use crate::epistemic::{EpistemicStatus, Revisability, Source};
use crate::hir;
use crate::interp::{Interpreter, Value};
use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{Config, Context, Editor, Helper, Result as RlResult};
use std::borrow::Cow;
use std::collections::HashMap;

// =============================================================================
// ANSI COLOR CODES
// =============================================================================

mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";

    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";

    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    pub const BRIGHT_YELLOW: &str = "\x1b[93m";
    pub const BRIGHT_BLUE: &str = "\x1b[94m";
    pub const BRIGHT_CYAN: &str = "\x1b[96m";
}

// =============================================================================
// CONFIDENCE BADGES
// =============================================================================

/// Confidence badge based on confidence level
fn confidence_badge(confidence: f64) -> &'static str {
    if confidence >= 0.95 {
        "🟢" // High confidence
    } else if confidence >= 0.80 {
        "🟡" // Medium-high confidence
    } else if confidence >= 0.60 {
        "🟠" // Medium confidence
    } else if confidence >= 0.30 {
        "🔴" // Low confidence
    } else {
        "⚫" // Very low confidence
    }
}

/// Confidence bar visualization
fn confidence_bar(confidence: f64, width: usize) -> String {
    let filled = ((confidence * width as f64).round() as usize).min(width);
    let empty = width.saturating_sub(filled);

    let color = if confidence >= 0.8 {
        colors::GREEN
    } else if confidence >= 0.6 {
        colors::YELLOW
    } else {
        colors::RED
    };

    format!(
        "{}{}{}{}{}",
        color,
        "█".repeat(filled),
        colors::DIM,
        "░".repeat(empty),
        colors::RESET
    )
}

// =============================================================================
// REPL HELPER (AUTOCOMPLETION)
// =============================================================================

/// Commands available in the REPL
const COMMANDS: &[&str] = &[
    ":help",
    ":quit",
    ":q",
    ":exit",
    ":clear",
    ":env",
    ":funcs",
    ":ast",
    ":hir",
    ":types",
    ":jit",
    ":type",
    ":load",
    ":provenance",
    ":uncertainty",
    ":confidence",
    ":info",
    ":save",
];

/// Keywords in Sounio
const KEYWORDS: &[&str] = &[
    "let", "var", "fn", "struct", "enum", "effect", "if", "else", "while", "for", "in", "return",
    "true", "false", "pub", "use", "mod", "match", "with", "linear", "kernel", "type", "import",
];

/// REPL helper for autocompletion
struct ReplHelper {
    file_completer: FilenameCompleter,
    bindings: Vec<String>,
    functions: Vec<String>,
}

impl ReplHelper {
    fn new() -> Self {
        Self {
            file_completer: FilenameCompleter::new(),
            bindings: Vec::new(),
            functions: Vec::new(),
        }
    }

    fn update_bindings(&mut self, bindings: &HashMap<String, Value>) {
        self.bindings = bindings.keys().cloned().collect();
    }

    fn update_functions(&mut self, functions: &HashMap<String, String>) {
        self.functions = functions.keys().cloned().collect();
    }
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        // Command completion
        if line.starts_with(':') {
            let prefix = &line[..pos];
            let completions: Vec<Pair> = COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(prefix))
                .map(|cmd| Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                })
                .collect();
            return Ok((0, completions));
        }

        // File completion for :load
        if line.starts_with(":load ") {
            return self.file_completer.complete(line, pos, ctx);
        }

        // Variable/function completion
        let word_start = line[..pos]
            .rfind(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|i| i + 1)
            .unwrap_or(0);
        let prefix = &line[word_start..pos];

        if prefix.is_empty() {
            return Ok((pos, Vec::new()));
        }

        let mut completions: Vec<Pair> = Vec::new();

        // Add keywords
        for kw in KEYWORDS {
            if kw.starts_with(prefix) {
                completions.push(Pair {
                    display: kw.to_string(),
                    replacement: kw.to_string(),
                });
            }
        }

        // Add bindings
        for binding in &self.bindings {
            if binding.starts_with(prefix) {
                completions.push(Pair {
                    display: binding.clone(),
                    replacement: binding.clone(),
                });
            }
        }

        // Add functions
        for func in &self.functions {
            if func.starts_with(prefix) {
                completions.push(Pair {
                    display: format!("{}()", func),
                    replacement: format!("{}(", func),
                });
            }
        }

        Ok((word_start, completions))
    }
}

impl Hinter for ReplHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<String> {
        if line.starts_with(':') && pos == line.len() {
            // Suggest command completion
            for cmd in COMMANDS {
                if cmd.starts_with(line) && cmd.len() > line.len() {
                    return Some(cmd[line.len()..].to_string());
                }
            }
        }
        None
    }
}

impl Highlighter for ReplHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Owned(format!(
            "{}{}{}",
            colors::BRIGHT_CYAN,
            prompt,
            colors::RESET
        ))
    }
}

impl Validator for ReplHelper {}

impl Helper for ReplHelper {}

// =============================================================================
// EPISTEMIC VALUE TRACKING
// =============================================================================

/// Epistemic metadata for a REPL value
#[derive(Debug, Clone)]
pub struct EpistemicValue {
    /// The actual value
    pub value: Value,
    /// Epistemic status (confidence, source, revisability)
    pub status: EpistemicStatus,
    /// Expression that produced this value
    pub expression: String,
    /// Line number when created
    pub line: usize,
}

impl EpistemicValue {
    /// Create from a value with default epistemic status
    pub fn from_value(value: Value, expression: String, line: usize) -> Self {
        // Determine epistemic status based on value type
        let status = match &value {
            Value::Uncertain { mean: _, std } => {
                // Uncertain values have lower confidence based on std
                let confidence = (1.0 - std.abs().min(1.0)).max(0.1);
                EpistemicStatus::empirical(confidence, Source::Derivation("computation".into()))
            }
            Value::Distribution(_) => {
                EpistemicStatus::empirical(0.7, Source::ModelPrediction {
                    model: "distribution".into(),
                    version: None,
                })
            }
            _ => EpistemicStatus::axiomatic(),
        };

        Self {
            value,
            status,
            expression,
            line,
        }
    }

    /// Create with explicit epistemic status
    pub fn with_status(value: Value, status: EpistemicStatus, expression: String, line: usize) -> Self {
        Self {
            value,
            status,
            expression,
            line,
        }
    }

    /// Format for display
    pub fn format_display(&self) -> String {
        let conf = self.status.confidence.value();
        let badge = confidence_badge(conf);

        match &self.value {
            Value::Uncertain { mean, std } => {
                format!(
                    "{} {}{:.4} ± {:.4}{} {}[{:.1}%]{}",
                    badge,
                    colors::BRIGHT_YELLOW,
                    mean,
                    std,
                    colors::RESET,
                    colors::DIM,
                    conf * 100.0,
                    colors::RESET
                )
            }
            _ => {
                if conf < 1.0 {
                    format!(
                        "{} {:?} {}[{:.1}%]{}",
                        badge,
                        self.value,
                        colors::DIM,
                        conf * 100.0,
                        colors::RESET
                    )
                } else {
                    format!("{} {:?}", badge, self.value)
                }
            }
        }
    }

    /// Format provenance chain
    pub fn format_provenance(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "{}Provenance for: {}{}\n",
            colors::BOLD, self.expression, colors::RESET
        ));
        output.push_str(&format!("  Line: {}\n", self.line));
        output.push_str(&format!("  Source: {}\n", format_source(&self.status.source)));
        output.push_str(&format!(
            "  Revisability: {}\n",
            format_revisability(&self.status.revisability)
        ));
        if !self.status.evidence.is_empty() {
            output.push_str("  Evidence:\n");
            for ev in &self.status.evidence {
                output.push_str(&format!("    - {:?}\n", ev));
            }
        }
        output
    }

    /// Format uncertainty details
    pub fn format_uncertainty(&self) -> String {
        let mut output = String::new();
        let conf = &self.status.confidence;

        output.push_str(&format!(
            "{}Uncertainty for: {}{}\n",
            colors::BOLD, self.expression, colors::RESET
        ));

        match &self.value {
            Value::Uncertain { mean, std } => {
                output.push_str(&format!("  Mean: {:.6}\n", mean));
                output.push_str(&format!("  Std Dev: {:.6}\n", std));
                output.push_str(&format!("  95% CI: [{:.6}, {:.6}]\n", mean - 1.96 * std, mean + 1.96 * std));
            }
            _ => {
                output.push_str("  Type: Deterministic\n");
            }
        }

        output.push_str(&format!("  Confidence: {:.1}%\n", conf.value() * 100.0));
        output.push_str(&format!("  Bar: [{}] {}\n", confidence_bar(conf.value(), 20), confidence_badge(conf.value())));

        if let Some(lower) = conf.lower_bound() {
            if let Some(upper) = conf.upper_bound() {
                output.push_str(&format!("  Confidence Interval: [{:.1}%, {:.1}%]\n", lower * 100.0, upper * 100.0));
            }
        }

        output
    }

    /// Format confidence details
    pub fn format_confidence(&self) -> String {
        let conf = &self.status.confidence;
        format!(
            "{} {:.1}% [{}]",
            confidence_badge(conf.value()),
            conf.value() * 100.0,
            confidence_bar(conf.value(), 20)
        )
    }
}

/// Format source for display
fn format_source(source: &Source) -> String {
    match source {
        Source::Axiom => format!("{}Axiom{} (by definition)", colors::GREEN, colors::RESET),
        Source::Measurement { instrument, protocol, timestamp } => {
            let mut s = format!("{}Measurement{}", colors::BLUE, colors::RESET);
            if let Some(inst) = instrument {
                s.push_str(&format!(" via {}", inst));
            }
            if let Some(proto) = protocol {
                s.push_str(&format!(" ({})", proto));
            }
            if let Some(ts) = timestamp {
                s.push_str(&format!(" @ {}", ts));
            }
            s
        }
        Source::Derivation(name) => {
            format!("{}Derivation{} from {}", colors::YELLOW, colors::RESET, name)
        }
        Source::External { uri, accessed } => {
            let mut s = format!("{}External{} {}", colors::MAGENTA, colors::RESET, uri);
            if let Some(acc) = accessed {
                s.push_str(&format!(" (accessed {})", acc));
            }
            s
        }
        Source::OntologyAssertion { ontology, term } => {
            format!(
                "{}Ontology{} {}:{}",
                colors::CYAN, colors::RESET, ontology, term
            )
        }
        Source::ModelPrediction { model, version } => {
            let mut s = format!("{}Model{} {}", colors::BRIGHT_BLUE, colors::RESET, model);
            if let Some(ver) = version {
                s.push_str(&format!(" v{}", ver));
            }
            s
        }
        Source::Transformation { original, via } => {
            format!(
                "{} → {}{}{}",
                format_source(original),
                colors::DIM,
                via,
                colors::RESET
            )
        }
        Source::HumanAssertion { asserter } => {
            let mut s = format!("{}Human Assertion{}", colors::WHITE, colors::RESET);
            if let Some(who) = asserter {
                s.push_str(&format!(" by {}", who));
            }
            s
        }
        Source::Unknown => format!("{}Unknown{}", colors::DIM, colors::RESET),
    }
}

/// Format revisability for display
fn format_revisability(rev: &Revisability) -> String {
    match rev {
        Revisability::NonRevisable => {
            format!("{}Non-revisable{} (axiomatic)", colors::GREEN, colors::RESET)
        }
        Revisability::Revisable { conditions } => {
            format!(
                "{}Revisable{} if: {}",
                colors::YELLOW,
                colors::RESET,
                conditions.join(", ")
            )
        }
        Revisability::MustRevise { reason } => {
            format!(
                "{}Must Revise{}: {}",
                colors::RED,
                colors::RESET,
                reason
            )
        }
    }
}

// =============================================================================
// REPL CONFIGURATION
// =============================================================================

/// REPL configuration
#[derive(Debug, Clone)]
pub struct ReplConfig {
    /// Show AST after parsing
    pub show_ast: bool,
    /// Show HIR after type checking
    pub show_hir: bool,
    /// Show types of expressions
    pub show_types: bool,
    /// Use JIT compilation instead of interpreter
    pub use_jit: bool,
    /// History file path
    pub history_file: Option<String>,
    /// Show epistemic info with values
    pub show_epistemic: bool,
    /// Enable colored output
    pub colored: bool,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            show_ast: false,
            show_hir: false,
            show_types: true,
            use_jit: false,
            history_file: Some(".sounio_history".to_string()),
            show_epistemic: true,
            colored: true,
        }
    }
}

// =============================================================================
// REPL STATE
// =============================================================================

/// The REPL state
pub struct Repl {
    config: ReplConfig,
    /// Accumulated function definitions
    functions: HashMap<String, String>,
    /// Accumulated type definitions
    types: HashMap<String, String>,
    /// Variable bindings from previous expressions (raw values)
    bindings: HashMap<String, Value>,
    /// Epistemic bindings with full metadata
    epistemic_bindings: HashMap<String, EpistemicValue>,
    /// Line counter
    line_count: usize,
}

impl Repl {
    pub fn new(config: ReplConfig) -> Self {
        Self {
            config,
            functions: HashMap::new(),
            types: HashMap::new(),
            bindings: HashMap::new(),
            epistemic_bindings: HashMap::new(),
            line_count: 0,
        }
    }

    /// Run the REPL
    pub fn run(&mut self) -> RlResult<()> {
        // Configure rustyline with completions
        let config = Config::builder()
            .history_ignore_space(true)
            .completion_type(rustyline::CompletionType::List)
            .build();

        let mut rl: Editor<ReplHelper, DefaultHistory> = Editor::with_config(config)?;
        rl.set_helper(Some(ReplHelper::new()));

        // Load history
        if let Some(ref hist_file) = self.config.history_file {
            let _ = rl.load_history(hist_file);
        }

        self.print_banner();

        loop {
            // Update helper with current bindings
            if let Some(helper) = rl.helper_mut() {
                helper.update_bindings(&self.bindings);
                helper.update_functions(&self.functions);
            }

            let prompt = format!("sio[{}]> ", self.line_count);

            match rl.readline(&prompt) {
                Ok(line) => {
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    let _ = rl.add_history_entry(line);

                    // Handle commands
                    if line.starts_with(':') {
                        if self.handle_command(line) {
                            break;
                        }
                        continue;
                    }

                    // Handle multi-line input for function/type definitions
                    let input = if line.starts_with("fn ")
                        || line.starts_with("struct ")
                        || line.starts_with("enum ")
                        || line.starts_with("effect ")
                    {
                        self.read_multiline(&mut rl, line)?
                    } else {
                        line.to_string()
                    };

                    // Evaluate the input
                    self.eval_input(&input);
                    self.line_count += 1;
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    self.print_goodbye();
                    break;
                }
                Err(err) => {
                    eprintln!("{}Error: {:?}{}", colors::RED, err, colors::RESET);
                    break;
                }
            }
        }

        // Save history
        if let Some(ref hist_file) = self.config.history_file {
            let _ = rl.save_history(hist_file);
        }

        Ok(())
    }

    fn print_banner(&self) {
        println!(
            "{}{}Sounio REPL{} v0.93.0",
            colors::BOLD,
            colors::BRIGHT_CYAN,
            colors::RESET
        );
        println!(
            "{}Epistemic computing at the horizon of certainty{}",
            colors::DIM,
            colors::RESET
        );
        println!();
        println!(
            "Type {}:help{} for help, {}:quit{} to exit",
            colors::BRIGHT_GREEN,
            colors::RESET,
            colors::BRIGHT_GREEN,
            colors::RESET
        );
        if self.config.show_epistemic {
            println!(
                "Epistemic mode: {}ON{} - values show confidence badges",
                colors::GREEN,
                colors::RESET
            );
        }
        println!();
    }

    fn print_goodbye(&self) {
        println!(
            "\n{}Goodbye!{} Session had {} evaluations.",
            colors::BRIGHT_CYAN,
            colors::RESET,
            self.line_count
        );
    }

    fn read_multiline(&self, rl: &mut Editor<ReplHelper, DefaultHistory>, first_line: &str) -> RlResult<String> {
        let mut lines = vec![first_line.to_string()];
        let mut brace_count =
            first_line.matches('{').count() as i32 - first_line.matches('}').count() as i32;

        while brace_count > 0 {
            match rl.readline("... ") {
                Ok(line) => {
                    brace_count += line.matches('{').count() as i32;
                    brace_count -= line.matches('}').count() as i32;
                    lines.push(line);
                }
                Err(_) => break,
            }
        }

        Ok(lines.join("\n"))
    }

    fn handle_command(&mut self, cmd: &str) -> bool {
        let parts: Vec<&str> = cmd.split_whitespace().collect();

        match parts.first().copied() {
            Some(":quit") | Some(":q") | Some(":exit") => {
                self.print_goodbye();
                return true;
            }
            Some(":help") | Some(":h") | Some(":?") => {
                self.print_help();
            }
            Some(":clear") => {
                self.functions.clear();
                self.types.clear();
                self.bindings.clear();
                self.epistemic_bindings.clear();
                println!(
                    "{}Cleared{} all definitions and bindings.",
                    colors::GREEN,
                    colors::RESET
                );
            }
            Some(":ast") => {
                self.config.show_ast = !self.config.show_ast;
                println!(
                    "Show AST: {}{}{}",
                    if self.config.show_ast { colors::GREEN } else { colors::RED },
                    self.config.show_ast,
                    colors::RESET
                );
            }
            Some(":hir") => {
                self.config.show_hir = !self.config.show_hir;
                println!(
                    "Show HIR: {}{}{}",
                    if self.config.show_hir { colors::GREEN } else { colors::RED },
                    self.config.show_hir,
                    colors::RESET
                );
            }
            Some(":types") => {
                self.config.show_types = !self.config.show_types;
                println!(
                    "Show types: {}{}{}",
                    if self.config.show_types { colors::GREEN } else { colors::RED },
                    self.config.show_types,
                    colors::RESET
                );
            }
            Some(":jit") => {
                self.config.use_jit = !self.config.use_jit;
                println!(
                    "Use JIT: {}{}{}",
                    if self.config.use_jit { colors::GREEN } else { colors::RED },
                    self.config.use_jit,
                    colors::RESET
                );
            }
            Some(":epistemic") => {
                self.config.show_epistemic = !self.config.show_epistemic;
                println!(
                    "Epistemic mode: {}{}{}",
                    if self.config.show_epistemic { colors::GREEN } else { colors::RED },
                    if self.config.show_epistemic { "ON" } else { "OFF" },
                    colors::RESET
                );
            }
            Some(":env") => {
                self.print_environment();
            }
            Some(":funcs") => {
                self.print_functions();
            }
            Some(":load") if parts.len() > 1 => {
                self.load_file(parts[1]);
            }
            Some(":save") if parts.len() > 1 => {
                self.save_session(parts[1]);
            }
            Some(":type") if parts.len() > 1 => {
                let expr = parts[1..].join(" ");
                self.show_type(&expr);
            }
            // Epistemic commands
            Some(":provenance") | Some(":prov") => {
                if parts.len() > 1 {
                    self.show_provenance(parts[1]);
                } else {
                    self.show_all_provenance();
                }
            }
            Some(":uncertainty") | Some(":unc") => {
                if parts.len() > 1 {
                    self.show_uncertainty(parts[1]);
                } else {
                    self.show_all_uncertainty();
                }
            }
            Some(":confidence") | Some(":conf") => {
                if parts.len() > 1 {
                    self.show_confidence(parts[1]);
                } else {
                    self.show_all_confidence();
                }
            }
            Some(":info") | Some(":i") => {
                if parts.len() > 1 {
                    self.show_info(parts[1]);
                } else {
                    println!(
                        "{}Usage:{} :info <variable>",
                        colors::YELLOW,
                        colors::RESET
                    );
                }
            }
            Some(cmd) => {
                println!(
                    "{}Unknown command:{} {}",
                    colors::RED,
                    colors::RESET,
                    cmd
                );
                println!(
                    "Type {}:help{} for available commands.",
                    colors::BRIGHT_GREEN,
                    colors::RESET
                );
            }
            None => {}
        }

        false
    }

    fn print_help(&self) {
        println!(
            "{}{}Sounio REPL Commands{}",
            colors::BOLD,
            colors::BRIGHT_CYAN,
            colors::RESET
        );
        println!();

        println!("{}General:{}", colors::BOLD, colors::RESET);
        println!("  {}:help{}, :h, :?       Show this help", colors::GREEN, colors::RESET);
        println!("  {}:quit{}, :q, :exit   Exit the REPL", colors::GREEN, colors::RESET);
        println!("  {}:clear{}             Clear all definitions", colors::GREEN, colors::RESET);
        println!("  {}:env{}               Show current bindings", colors::GREEN, colors::RESET);
        println!("  {}:funcs{}             Show defined functions", colors::GREEN, colors::RESET);
        println!("  {}:load{} <file>       Load a Sounio source file", colors::GREEN, colors::RESET);
        println!("  {}:save{} <file>       Save session to file", colors::GREEN, colors::RESET);
        println!();

        println!("{}Inspection:{}", colors::BOLD, colors::RESET);
        println!("  {}:type{} <expr>       Show type of expression", colors::BLUE, colors::RESET);
        println!("  {}:ast{}               Toggle AST display", colors::BLUE, colors::RESET);
        println!("  {}:hir{}               Toggle HIR display", colors::BLUE, colors::RESET);
        println!("  {}:types{}             Toggle type display", colors::BLUE, colors::RESET);
        println!("  {}:jit{}               Toggle JIT compilation", colors::BLUE, colors::RESET);
        println!();

        println!("{}Epistemic ({}new{}{}):{}", colors::BOLD, colors::BRIGHT_GREEN, colors::RESET, colors::BOLD, colors::RESET);
        println!("  {}:epistemic{}         Toggle epistemic display", colors::MAGENTA, colors::RESET);
        println!("  {}:provenance{} [var]  Show provenance chain", colors::MAGENTA, colors::RESET);
        println!("  {}:uncertainty{} [var] Show uncertainty details", colors::MAGENTA, colors::RESET);
        println!("  {}:confidence{} [var]  Show confidence levels", colors::MAGENTA, colors::RESET);
        println!("  {}:info{} <var>        Show full epistemic info", colors::MAGENTA, colors::RESET);
        println!();

        println!("{}Confidence Badges:{}", colors::BOLD, colors::RESET);
        println!("  🟢 High (≥95%)   🟡 Good (≥80%)   🟠 Medium (≥60%)");
        println!("  🔴 Low (≥30%)    ⚫ Very Low (<30%)");
        println!();

        println!("{}Examples:{}", colors::BOLD, colors::RESET);
        println!("  {}1 + 2 * 3{}               Evaluate expression", colors::DIM, colors::RESET);
        println!("  {}let x = 42{}              Bind a variable", colors::DIM, colors::RESET);
        println!("  {}let u = uncertain(5.0, 0.3){}  Create uncertain value", colors::DIM, colors::RESET);
        println!("  {}:confidence x{}           Show x's confidence", colors::DIM, colors::RESET);
    }

    fn print_environment(&self) {
        if self.bindings.is_empty() {
            println!("{}No bindings.{}", colors::DIM, colors::RESET);
        } else {
            println!(
                "{}Current bindings:{} ({} total)",
                colors::BOLD,
                colors::RESET,
                self.bindings.len()
            );
            for (name, _) in &self.bindings {
                // Use epistemic display if available
                if let Some(ev) = self.epistemic_bindings.get(name) {
                    println!("  {} = {}", name, ev.format_display());
                } else if let Some(value) = self.bindings.get(name) {
                    println!("  {} = {:?}", name, value);
                }
            }
        }
    }

    fn print_functions(&self) {
        if self.functions.is_empty() {
            println!("{}No functions defined.{}", colors::DIM, colors::RESET);
        } else {
            println!(
                "{}Defined functions:{} ({} total)",
                colors::BOLD,
                colors::RESET,
                self.functions.len()
            );
            for name in self.functions.keys() {
                println!("  {}{}(){}", colors::BRIGHT_BLUE, name, colors::RESET);
            }
        }
    }

    // =========================================================================
    // EPISTEMIC COMMANDS
    // =========================================================================

    fn show_provenance(&self, var: &str) {
        if let Some(ev) = self.epistemic_bindings.get(var) {
            print!("{}", ev.format_provenance());
        } else if self.bindings.contains_key(var) {
            println!(
                "{}{}:{} No epistemic metadata (value is axiomatic)",
                colors::YELLOW,
                var,
                colors::RESET
            );
        } else {
            println!(
                "{}Unknown variable:{} {}",
                colors::RED,
                colors::RESET,
                var
            );
        }
    }

    fn show_all_provenance(&self) {
        if self.epistemic_bindings.is_empty() {
            println!("{}No epistemic bindings.{}", colors::DIM, colors::RESET);
            return;
        }

        println!(
            "{}Provenance for all bindings:{}",
            colors::BOLD,
            colors::RESET
        );
        for (name, ev) in &self.epistemic_bindings {
            println!(
                "  {}{}{}: {}",
                colors::BRIGHT_CYAN,
                name,
                colors::RESET,
                format_source(&ev.status.source)
            );
        }
    }

    fn show_uncertainty(&self, var: &str) {
        if let Some(ev) = self.epistemic_bindings.get(var) {
            print!("{}", ev.format_uncertainty());
        } else if self.bindings.contains_key(var) {
            println!(
                "{}{}:{} Deterministic value (no uncertainty)",
                colors::GREEN,
                var,
                colors::RESET
            );
        } else {
            println!(
                "{}Unknown variable:{} {}",
                colors::RED,
                colors::RESET,
                var
            );
        }
    }

    fn show_all_uncertainty(&self) {
        if self.epistemic_bindings.is_empty() {
            println!("{}No epistemic bindings.{}", colors::DIM, colors::RESET);
            return;
        }

        println!(
            "{}Uncertainty for all bindings:{}",
            colors::BOLD,
            colors::RESET
        );
        for (name, ev) in &self.epistemic_bindings {
            let conf = ev.status.confidence.value();
            println!(
                "  {}{}{}: {} {:.1}%",
                colors::BRIGHT_CYAN,
                name,
                colors::RESET,
                confidence_badge(conf),
                conf * 100.0
            );
        }
    }

    fn show_confidence(&self, var: &str) {
        if let Some(ev) = self.epistemic_bindings.get(var) {
            println!("{}: {}", var, ev.format_confidence());
        } else if self.bindings.contains_key(var) {
            println!(
                "{}: {} 100.0% [{}]",
                var,
                confidence_badge(1.0),
                confidence_bar(1.0, 20)
            );
        } else {
            println!(
                "{}Unknown variable:{} {}",
                colors::RED,
                colors::RESET,
                var
            );
        }
    }

    fn show_all_confidence(&self) {
        if self.bindings.is_empty() {
            println!("{}No bindings.{}", colors::DIM, colors::RESET);
            return;
        }

        println!(
            "{}Confidence levels:{}",
            colors::BOLD,
            colors::RESET
        );
        for name in self.bindings.keys() {
            let (badge, conf) = if let Some(ev) = self.epistemic_bindings.get(name) {
                let c = ev.status.confidence.value();
                (confidence_badge(c), c)
            } else {
                (confidence_badge(1.0), 1.0)
            };
            println!(
                "  {}{}{}: {} [{:.1}%]",
                colors::BRIGHT_CYAN,
                name,
                colors::RESET,
                badge,
                conf * 100.0
            );
        }
    }

    fn show_info(&self, var: &str) {
        if let Some(ev) = self.epistemic_bindings.get(var) {
            println!(
                "{}Full epistemic info for: {}{}{}",
                colors::BOLD,
                colors::BRIGHT_CYAN,
                var,
                colors::RESET
            );
            println!();
            println!("  {}Value:{}        {}", colors::BOLD, colors::RESET, ev.format_display());
            println!("  {}Expression:{}   {}", colors::BOLD, colors::RESET, ev.expression);
            println!("  {}Line:{}         {}", colors::BOLD, colors::RESET, ev.line);
            println!("  {}Source:{}       {}", colors::BOLD, colors::RESET, format_source(&ev.status.source));
            println!("  {}Revisability:{} {}", colors::BOLD, colors::RESET, format_revisability(&ev.status.revisability));
            println!(
                "  {}Confidence:{}   {} ({:.1}%)",
                colors::BOLD,
                colors::RESET,
                confidence_bar(ev.status.confidence.value(), 20),
                ev.status.confidence.value() * 100.0
            );
            if !ev.status.evidence.is_empty() {
                println!("  {}Evidence:{}     {} items", colors::BOLD, colors::RESET, ev.status.evidence.len());
                for e in &ev.status.evidence {
                    println!("    - {:?}", e);
                }
            }
        } else if let Some(value) = self.bindings.get(var) {
            println!(
                "{}Info for: {}{}{}",
                colors::BOLD,
                colors::BRIGHT_CYAN,
                var,
                colors::RESET
            );
            println!();
            println!("  {}Value:{}      {:?}", colors::BOLD, colors::RESET, value);
            println!("  {}Source:{}     {}", colors::BOLD, colors::RESET, format_source(&Source::Axiom));
            println!("  {}Confidence:{} {} (100.0%)", colors::BOLD, colors::RESET, confidence_bar(1.0, 20));
        } else {
            println!(
                "{}Unknown variable:{} {}",
                colors::RED,
                colors::RESET,
                var
            );
        }
    }

    fn save_session(&self, path: &str) {
        let mut content = String::new();
        content.push_str("// Sounio REPL session\n\n");

        for def in self.types.values() {
            content.push_str(def);
            content.push_str("\n\n");
        }

        for def in self.functions.values() {
            content.push_str(def);
            content.push_str("\n\n");
        }

        match std::fs::write(path, content) {
            Ok(()) => println!(
                "{}Saved{} session to {}",
                colors::GREEN,
                colors::RESET,
                path
            ),
            Err(e) => eprintln!(
                "{}Failed to save:{} {}",
                colors::RED,
                colors::RESET,
                e
            ),
        }
    }

    fn load_file(&mut self, path: &str) {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                println!("Loading {}...", path);
                self.eval_input(&content);
            }
            Err(e) => {
                eprintln!("Failed to load {}: {}", path, e);
            }
        }
    }

    fn show_type(&mut self, expr: &str) {
        // Build full source with definitions
        let source = self.build_source(expr);

        // Parse using the library functions
        match crate::lexer::lex(&source) {
            Ok(tokens) => {
                match crate::parser::parse(&tokens, &source) {
                    Ok(ast) => {
                        // Type check
                        match crate::check::check(&ast) {
                            Ok(hir) => {
                                // Find the type of the last expression
                                if let Some(item) = hir.items.last()
                                    && let hir::HirItem::Function(f) = item
                                {
                                    println!("Type: {:?}", f.body.ty);
                                }
                            }
                            Err(e) => {
                                eprintln!("Type error: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Parse error: {:?}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Lex error: {:?}", e);
            }
        }
    }

    fn eval_input(&mut self, input: &str) {
        // Check if this is a definition
        if input.starts_with("fn ")
            && let Some(name) = self.extract_fn_name(input)
        {
            self.functions.insert(name.clone(), input.to_string());
            println!("Defined function: {}", name);
            return;
        }

        if input.starts_with("struct ")
            && let Some(name) = self.extract_type_name(input)
        {
            self.types.insert(name.clone(), input.to_string());
            println!("Defined struct: {}", name);
            return;
        }

        if input.starts_with("enum ")
            && let Some(name) = self.extract_type_name(input)
        {
            self.types.insert(name.clone(), input.to_string());
            println!("Defined enum: {}", name);
            return;
        }

        // Check if this is a let binding
        let is_let = input.starts_with("let ");

        // Wrap expression in a main function for evaluation
        let wrapped = if is_let {
            // let x = expr -> we need to capture the value
            format!("{}\n__repl_result__", input)
        } else {
            input.to_string()
        };

        let source = self.build_source(&wrapped);

        if self.config.show_ast {
            println!("--- Source ---\n{}\n", source);
        }

        // Parse using library functions
        let tokens = match crate::lexer::lex(&source) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Lex error: {:?}", e);
                return;
            }
        };

        let ast = match crate::parser::parse(&tokens, &source) {
            Ok(ast) => ast,
            Err(e) => {
                eprintln!("Parse error: {:?}", e);
                return;
            }
        };

        if self.config.show_ast {
            println!("--- AST ---\n{:#?}\n", ast);
        }

        // Type check
        let hir = match crate::check::check(&ast) {
            Ok(hir) => hir,
            Err(e) => {
                eprintln!("Type error: {}", e);
                return;
            }
        };

        if self.config.show_hir {
            println!("--- HIR ---\n{:#?}\n", hir);
        }

        // Execute
        if self.config.use_jit {
            self.eval_jit(&hir);
        } else {
            self.eval_interp(&hir, is_let, input);
        }
    }

    fn eval_interp(&mut self, hir: &hir::Hir, is_let: bool, input: &str) {
        let mut interp = Interpreter::new();

        // Pre-populate environment with existing bindings
        for (name, value) in &self.bindings {
            interp.env_mut().define(name.clone(), value.clone());
        }

        match interp.run(hir) {
            Ok(value) => {
                if is_let {
                    // Extract binding name and store value
                    if let Some(name) = self.extract_let_name(input) {
                        // Create epistemic value with tracking
                        let ev = EpistemicValue::from_value(
                            value.clone(),
                            input.to_string(),
                            self.line_count,
                        );

                        // Display with epistemic formatting
                        if self.config.show_epistemic {
                            println!(
                                "{}{}{} = {}",
                                colors::BRIGHT_CYAN,
                                name,
                                colors::RESET,
                                ev.format_display()
                            );
                        } else {
                            println!("{} = {:?}", name, value);
                        }

                        // Store in both maps
                        self.bindings.insert(name.clone(), value);
                        self.epistemic_bindings.insert(name, ev);
                    }
                } else if value != Value::Unit {
                    // Display result with epistemic info
                    if self.config.show_epistemic {
                        let ev = EpistemicValue::from_value(
                            value.clone(),
                            input.to_string(),
                            self.line_count,
                        );
                        println!(
                            "{}=>{} {}",
                            colors::BRIGHT_GREEN,
                            colors::RESET,
                            ev.format_display()
                        );
                    } else if self.config.show_types {
                        println!("=> {:?}", value);
                    } else {
                        println!("{:?}", value);
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "{}Runtime error:{} {}",
                    colors::RED,
                    colors::RESET,
                    e
                );
            }
        }
    }

    fn eval_jit(&self, hir: &hir::Hir) {
        #[cfg(feature = "jit")]
        {
            use crate::codegen::cranelift::CraneliftJit;
            use crate::hlir;

            // Lower to HLIR
            let hlir_module = hlir::lower(hir);

            // Compile and run
            let jit = CraneliftJit::new();
            match jit.compile_and_run(&hlir_module) {
                Ok(result) => {
                    println!("=> {}", result);
                }
                Err(e) => {
                    eprintln!("JIT error: {}", e);
                }
            }
        }

        #[cfg(not(feature = "jit"))]
        {
            let _ = hir;
            eprintln!("JIT not enabled. Compile with --features jit");
        }
    }

    fn build_source(&self, expr: &str) -> String {
        let mut source = String::new();

        // Add type definitions
        for def in self.types.values() {
            source.push_str(def);
            source.push('\n');
        }

        // Add function definitions
        for def in self.functions.values() {
            source.push_str(def);
            source.push('\n');
        }

        // Wrap expression in main function
        source.push_str(&format!("fn main() -> i64 {{\n    {}\n}}\n", expr));

        source
    }

    fn extract_fn_name(&self, input: &str) -> Option<String> {
        // fn name(...) -> ...
        let input = input.strip_prefix("fn ")?.trim_start();
        let end = input.find('(')?;
        Some(input[..end].trim().to_string())
    }

    fn extract_type_name(&self, input: &str) -> Option<String> {
        // struct/enum name { ... }
        let input = if input.starts_with("struct ") {
            input.strip_prefix("struct ")?
        } else if input.starts_with("enum ") {
            input.strip_prefix("enum ")?
        } else {
            return None;
        };
        let input = input.trim_start();
        let end = input.find(|c: char| c == '{' || c == '<' || c.is_whitespace())?;
        Some(input[..end].trim().to_string())
    }

    fn extract_let_name(&self, input: &str) -> Option<String> {
        // let name = ...
        let input = input.strip_prefix("let ")?.trim_start();
        let end = input.find(|c: char| c == '=' || c == ':' || c.is_whitespace())?;
        Some(input[..end].trim().to_string())
    }
}

/// Run the REPL with default configuration
pub fn run() -> RlResult<()> {
    let mut repl = Repl::new(ReplConfig::default());
    repl.run()
}

/// Run the REPL with custom configuration
pub fn run_with_config(config: ReplConfig) -> RlResult<()> {
    let mut repl = Repl::new(config);
    repl.run()
}
