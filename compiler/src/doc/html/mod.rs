//! HTML documentation generator
//!
//! Generates responsive, themed HTML documentation.

mod render;
mod syntax;
mod templates;

pub use render::{DocGenError, HtmlRenderer};
pub use syntax::SyntaxHighlighter;
pub use templates::Templates;

/// Default CSS for documentation
pub const DEFAULT_CSS: &str = include_str!("assets/main.css");

/// Light theme CSS
pub const LIGHT_THEME_CSS: &str = include_str!("assets/light.css");

/// Dark theme CSS
pub const DARK_THEME_CSS: &str = include_str!("assets/dark.css");

/// Syntax highlighting CSS
pub const HIGHLIGHT_CSS: &str = include_str!("assets/highlight.css");

/// Main JavaScript
pub const MAIN_JS: &str = include_str!("assets/main.js");

/// Search JavaScript
pub const SEARCH_JS: &str = include_str!("assets/search.js");
