//! Embedded Standard Library Modules
//!
//! This module provides access to the stdlib/compiler modules that are embedded
//! in the binary at build time. These modules are used by the self-hosted
//! Sounio compiler for bootstrapping.

// Include the auto-generated module with embedded source code
include!(concat!(env!("OUT_DIR"), "/embedded_stdlib.rs"));

/// Get the source code for an embedded module by name
///
/// # Arguments
/// * `name` - Module name in format "lexer::mod" or "parser::expr"
///
/// # Returns
/// The source code of the module, or None if not found
pub fn get_module(name: &str) -> Option<&'static str> {
    get_embedded_modules().get(name).copied()
}

/// Check if a module is embedded
pub fn has_module(name: &str) -> bool {
    get_embedded_modules().contains_key(name)
}

/// Get all embedded module names
pub fn list_modules() -> &'static [&'static str] {
    MODULE_NAMES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modules_embedded() {
        // Verify at least some modules are embedded
        assert!(MODULE_COUNT > 0, "No modules embedded");
        assert!(!MODULE_NAMES.is_empty(), "Module names list empty");
    }

    #[test]
    fn test_get_embedded_modules() {
        let modules = get_embedded_modules();
        assert!(!modules.is_empty(), "No modules in map");
    }

    #[test]
    fn test_list_modules() {
        let modules = list_modules();
        assert!(!modules.is_empty(), "Module list is empty");

        // Check that some expected modules exist
        let has_lexer = modules.iter().any(|m| m.contains("lexer"));
        let has_parser = modules.iter().any(|m| m.contains("parser"));

        assert!(
            has_lexer || has_parser,
            "Expected lexer or parser modules to be present"
        );
    }
}
