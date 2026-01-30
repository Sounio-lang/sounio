//! HTML renderer for documentation

use std::fs;
use std::path::{Path, PathBuf};

use super::syntax::SyntaxHighlighter;
use super::templates::Templates;
use crate::doc::model::*;

/// HTML documentation renderer
pub struct HtmlRenderer {
    /// Output directory
    output_dir: PathBuf,

    /// Templates
    templates: Templates,

    /// Syntax highlighter
    highlighter: SyntaxHighlighter,

    /// Crate documentation
    crate_doc: CrateDoc,

    /// Enable dark mode toggle
    dark_mode: bool,

    /// Include source code
    include_source: bool,
}

impl HtmlRenderer {
    /// Create a new HTML renderer
    pub fn new(crate_doc: CrateDoc, output_dir: PathBuf) -> Self {
        Self {
            output_dir,
            templates: Templates::new(&crate_doc.name),
            highlighter: SyntaxHighlighter::new(),
            crate_doc,
            dark_mode: true,
            include_source: true,
        }
    }

    /// Enable/disable dark mode
    pub fn with_dark_mode(mut self, enabled: bool) -> Self {
        self.dark_mode = enabled;
        self
    }

    /// Enable/disable source code inclusion
    pub fn with_source(mut self, enabled: bool) -> Self {
        self.include_source = enabled;
        self
    }

    /// Generate all documentation
    pub fn generate(&self) -> Result<(), DocGenError> {
        // Create output directory
        fs::create_dir_all(&self.output_dir)?;

        // Copy static assets
        self.copy_assets()?;

        // Generate index page
        self.generate_index()?;

        // Generate module pages
        self.generate_module(&self.crate_doc.root_module, &self.output_dir)?;

        // Generate search index
        self.generate_search_index()?;

        // Generate source code pages (if enabled)
        if self.include_source && !self.crate_doc.source_files.is_empty() {
            self.generate_source_pages()?;
        }

        Ok(())
    }

    /// Copy static assets (CSS, JS)
    fn copy_assets(&self) -> Result<(), DocGenError> {
        let assets_dir = self.output_dir.join("static");
        fs::create_dir_all(&assets_dir)?;

        // Write CSS
        fs::write(assets_dir.join("main.css"), super::DEFAULT_CSS)?;
        fs::write(assets_dir.join("light.css"), super::LIGHT_THEME_CSS)?;
        fs::write(assets_dir.join("dark.css"), super::DARK_THEME_CSS)?;
        fs::write(assets_dir.join("highlight.css"), super::HIGHLIGHT_CSS)?;

        // Write JS
        fs::write(assets_dir.join("main.js"), super::MAIN_JS)?;
        fs::write(assets_dir.join("search.js"), super::SEARCH_JS)?;

        Ok(())
    }

    /// Generate index page
    fn generate_index(&self) -> Result<(), DocGenError> {
        let content = self.templates.render_index(&self.crate_doc);
        let path = self.output_dir.join("index.html");
        fs::write(path, content)?;
        Ok(())
    }

    /// Generate module documentation
    fn generate_module(&self, module: &ModuleDoc, base_dir: &Path) -> Result<(), DocGenError> {
        let module_dir = if module.path == self.crate_doc.name {
            base_dir.to_path_buf()
        } else {
            base_dir.join(&module.name)
        };
        fs::create_dir_all(&module_dir)?;

        // Generate module index (skip for root if already generated)
        if module.path != self.crate_doc.name {
            let content = self.templates.render_module(module, &self.crate_doc.name);
            fs::write(module_dir.join("index.html"), content)?;
        }

        // Generate function pages
        for func in &module.functions {
            let content = self
                .templates
                .render_function(func, module, &self.crate_doc.name);
            fs::write(module_dir.join(format!("fn.{}.html", func.name)), content)?;
        }

        // Generate type pages
        for ty in &module.types {
            let content = self.templates.render_type(ty, module, &self.crate_doc.name);
            let prefix = ty.kind.as_str();
            fs::write(
                module_dir.join(format!("{}.{}.html", prefix, ty.name)),
                content,
            )?;
        }

        // Generate trait pages
        for trait_doc in &module.traits {
            let content = self
                .templates
                .render_trait(trait_doc, module, &self.crate_doc.name);
            fs::write(
                module_dir.join(format!("trait.{}.html", trait_doc.name)),
                content,
            )?;
        }

        // Generate constant pages
        for const_doc in &module.constants {
            let content = self
                .templates
                .render_constant(const_doc, module, &self.crate_doc.name);
            fs::write(
                module_dir.join(format!("const.{}.html", const_doc.name)),
                content,
            )?;
        }

        // Recursively generate submodules
        for submodule in &module.modules {
            self.generate_module(submodule, &module_dir)?;
        }

        Ok(())
    }

    /// Generate search index JSON
    fn generate_search_index(&self) -> Result<(), DocGenError> {
        let search_json = serde_json::to_string(&self.crate_doc.search_index)?;
        fs::write(
            self.output_dir.join("search-index.js"),
            format!("window.searchIndex = {};", search_json),
        )?;
        Ok(())
    }

    /// Generate source code pages
    fn generate_source_pages(&self) -> Result<(), DocGenError> {
        let src_dir = self.output_dir.join("src");
        fs::create_dir_all(&src_dir)?;

        for source_file in &self.crate_doc.source_files {
            let highlighted = self.highlighter.highlight(&source_file.content);
            let content = self
                .templates
                .render_source(&source_file.path, &highlighted);

            let relative_path = source_file
                .path
                .file_name()
                .map(PathBuf::from)
                .unwrap_or_else(|| source_file.path.clone());
            let out_path = src_dir.join(relative_path.with_extension("html"));

            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(out_path, content)?;
        }

        Ok(())
    }
}

/// Documentation generation error
#[derive(Debug)]
pub enum DocGenError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Template(String),
}

impl From<std::io::Error> for DocGenError {
    fn from(e: std::io::Error) -> Self {
        DocGenError::Io(e)
    }
}

impl From<serde_json::Error> for DocGenError {
    fn from(e: serde_json::Error) -> Self {
        DocGenError::Json(e)
    }
}

impl std::fmt::Display for DocGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocGenError::Io(e) => write!(f, "IO error: {}", e),
            DocGenError::Json(e) => write!(f, "JSON error: {}", e),
            DocGenError::Template(e) => write!(f, "Template error: {}", e),
        }
    }
}

impl std::error::Error for DocGenError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_creation() {
        let crate_doc = CrateDoc::new("test".to_string(), "0.1.0".to_string());
        let renderer = HtmlRenderer::new(crate_doc, PathBuf::from("/tmp/doc"));
        assert!(renderer.dark_mode);
        assert!(renderer.include_source);
    }
}
