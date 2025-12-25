//! HTML templates for documentation

use std::path::Path;

use crate::doc::model::*;

/// HTML templates
pub struct Templates {
    /// Crate name (for titles)
    crate_name: String,
}

impl Templates {
    /// Create new templates
    pub fn new(crate_name: &str) -> Self {
        Self {
            crate_name: crate_name.to_string(),
        }
    }

    /// Render the base HTML structure
    fn base(&self, title: &str, content: &str, breadcrumbs: &[(&str, &str)]) -> String {
        let breadcrumb_html: String = breadcrumbs
            .iter()
            .map(|(name, href)| format!(r#"<a href="{}">{}</a>"#, href, name))
            .collect::<Vec<_>>()
            .join(" :: ");

        format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title} - {crate_name} Documentation</title>
    <link rel="stylesheet" href="/static/main.css">
    <link rel="stylesheet" href="/static/highlight.css">
    <link rel="stylesheet" href="/static/light.css" id="theme-light">
    <link rel="stylesheet" href="/static/dark.css" id="theme-dark" disabled>
</head>
<body>
    <nav class="sidebar">
        <div class="sidebar-header">
            <h1><a href="/index.html">{crate_name}</a></h1>
        </div>
        <div class="search-container">
            <input type="search" id="search-input" placeholder="Search...">
            <div id="search-results" class="search-results"></div>
        </div>
        <div class="sidebar-nav" id="sidebar-nav">
        </div>
    </nav>

    <main class="content">
        <header class="top-bar">
            <nav class="breadcrumbs">{breadcrumbs}</nav>
            <div class="controls">
                <button id="theme-toggle" title="Toggle theme">Toggle Theme</button>
            </div>
        </header>

        <article class="main-content">
            {content}
        </article>
    </main>

    <script src="/static/main.js"></script>
    <script src="/static/search.js"></script>
    <script src="/search-index.js"></script>
</body>
</html>"#,
            title = html_escape(title),
            crate_name = html_escape(&self.crate_name),
            breadcrumbs = breadcrumb_html,
            content = content,
        )
    }

    /// Render index page
    pub fn render_index(&self, crate_doc: &CrateDoc) -> String {
        let mut content = String::new();

        // Crate header
        content.push_str(&format!(
            r#"
<div class="crate-header">
    <h1>Crate <code>{}</code></h1>
    <span class="version">Version {}</span>
</div>
"#,
            html_escape(&crate_doc.name),
            html_escape(&crate_doc.version)
        ));

        // Crate documentation
        if let Some(ref doc) = crate_doc.doc {
            content.push_str("<div class=\"crate-doc\">");
            content.push_str(&render_markdown(doc));
            content.push_str("</div>");
        }

        // Modules
        if !crate_doc.root_module.modules.is_empty() {
            content.push_str("<h2 id=\"modules\">Modules</h2>");
            content.push_str("<ul class=\"item-list\">");
            for module in &crate_doc.root_module.modules {
                content.push_str(&format!(
                    r#"
<li>
    <a href="{}/index.html" class="item-name mod">{}</a>
    <span class="item-desc">{}</span>
</li>"#,
                    html_escape(&module.name),
                    html_escape(&module.name),
                    html_escape(module.doc.as_deref().unwrap_or(""))
                ));
            }
            content.push_str("</ul>");
        }

        // Structs
        let structs: Vec<_> = crate_doc
            .root_module
            .types
            .iter()
            .filter(|t| t.kind == TypeKind::Struct)
            .collect();
        if !structs.is_empty() {
            content.push_str("<h2 id=\"structs\">Structs</h2>");
            content.push_str(&self.render_type_list(&structs, "struct"));
        }

        // Enums
        let enums: Vec<_> = crate_doc
            .root_module
            .types
            .iter()
            .filter(|t| t.kind == TypeKind::Enum)
            .collect();
        if !enums.is_empty() {
            content.push_str("<h2 id=\"enums\">Enums</h2>");
            content.push_str(&self.render_type_list(&enums, "enum"));
        }

        // Traits
        if !crate_doc.root_module.traits.is_empty() {
            content.push_str("<h2 id=\"traits\">Traits</h2>");
            content.push_str(&self.render_trait_list(&crate_doc.root_module.traits));
        }

        // Functions
        if !crate_doc.root_module.functions.is_empty() {
            content.push_str("<h2 id=\"functions\">Functions</h2>");
            content.push_str(&self.render_function_list(&crate_doc.root_module.functions));
        }

        // Constants
        if !crate_doc.root_module.constants.is_empty() {
            content.push_str("<h2 id=\"constants\">Constants</h2>");
            content.push_str(&self.render_constant_list(&crate_doc.root_module.constants));
        }

        self.base(&crate_doc.name, &content, &[])
    }

    /// Render module page
    pub fn render_module(&self, module: &ModuleDoc, crate_name: &str) -> String {
        let mut content = String::new();

        // Module header
        content.push_str(&format!(
            r#"
<div class="item-header">
    <h1>Module <code>{}</code></h1>
</div>
"#,
            html_escape(&module.path)
        ));

        // Module documentation
        if let Some(ref doc) = module.doc {
            content.push_str("<div class=\"module-doc\">");
            content.push_str(&render_markdown(doc));
            content.push_str("</div>");
        }

        // Submodules
        if !module.modules.is_empty() {
            content.push_str("<h2 id=\"modules\">Modules</h2>");
            content.push_str("<ul class=\"item-list\">");
            for submodule in &module.modules {
                content.push_str(&format!(
                    r#"
<li>
    <a href="{}/index.html" class="item-name mod">{}</a>
    <span class="item-desc">{}</span>
</li>"#,
                    html_escape(&submodule.name),
                    html_escape(&submodule.name),
                    html_escape(submodule.doc.as_deref().unwrap_or(""))
                ));
            }
            content.push_str("</ul>");
        }

        // Structs
        let structs: Vec<_> = module
            .types
            .iter()
            .filter(|t| t.kind == TypeKind::Struct)
            .collect();
        if !structs.is_empty() {
            content.push_str("<h2 id=\"structs\">Structs</h2>");
            content.push_str(&self.render_type_list(&structs, "struct"));
        }

        // Enums
        let enums: Vec<_> = module
            .types
            .iter()
            .filter(|t| t.kind == TypeKind::Enum)
            .collect();
        if !enums.is_empty() {
            content.push_str("<h2 id=\"enums\">Enums</h2>");
            content.push_str(&self.render_type_list(&enums, "enum"));
        }

        // Traits
        if !module.traits.is_empty() {
            content.push_str("<h2 id=\"traits\">Traits</h2>");
            content.push_str(&self.render_trait_list(&module.traits));
        }

        // Functions
        if !module.functions.is_empty() {
            content.push_str("<h2 id=\"functions\">Functions</h2>");
            content.push_str(&self.render_function_list(&module.functions));
        }

        // Type Aliases
        let type_aliases: Vec<_> = module
            .types
            .iter()
            .filter(|t| t.kind == TypeKind::TypeAlias)
            .collect();
        if !type_aliases.is_empty() {
            content.push_str("<h2 id=\"type-aliases\">Type Aliases</h2>");
            content.push_str(&self.render_type_list(&type_aliases, "type"));
        }

        // Constants
        if !module.constants.is_empty() {
            content.push_str("<h2 id=\"constants\">Constants</h2>");
            content.push_str(&self.render_constant_list(&module.constants));
        }

        // Re-exports
        if !module.reexports.is_empty() {
            content.push_str("<h2 id=\"reexports\">Re-exports</h2>");
            content.push_str("<ul class=\"item-list\">");
            for reexport in &module.reexports {
                content.push_str(&format!(
                    r#"
<li>
    <code>pub use {}</code>
</li>"#,
                    html_escape(&reexport.original_path)
                ));
            }
            content.push_str("</ul>");
        }

        let breadcrumbs = self.build_breadcrumbs(&module.path);
        let breadcrumbs_refs: Vec<(&str, &str)> = breadcrumbs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        self.base(&format!("mod {}", module.name), &content, &breadcrumbs_refs)
    }

    /// Render function page
    pub fn render_function(
        &self,
        func: &FunctionDoc,
        module: &ModuleDoc,
        crate_name: &str,
    ) -> String {
        let mut content = String::new();

        // Function header
        content.push_str(&format!(
            r#"
<div class="item-header">
    <h1>Function <code>{}</code></h1>
</div>
"#,
            html_escape(&func.name)
        ));

        // Signature
        content.push_str("<pre class=\"signature\">");
        content.push_str(&html_escape(&func.signature));
        content.push_str("</pre>");

        // Source link
        if func.source.line > 0 {
            content.push_str(&format!(
                r#"
<div class="source-link">
    <a href="/src/{}.html#L{}">[src]</a>
</div>
"#,
                func.source.file.display(),
                func.source.line
            ));
        }

        // Documentation
        if let Some(ref doc) = func.doc {
            content.push_str("<div class=\"docblock\">");
            content.push_str(&render_markdown(doc));
            content.push_str("</div>");
        }

        // Parameters
        if !func.params.is_empty() {
            content.push_str("<h2 id=\"parameters\">Parameters</h2>");
            content.push_str("<dl class=\"params\">");
            for param in &func.params {
                content.push_str(&format!(
                    r#"
<dt><code>{}</code>: <code>{}</code></dt>
<dd></dd>
"#,
                    html_escape(&param.name),
                    html_escape(&param.ty.display)
                ));
            }
            content.push_str("</dl>");
        }

        // Return type
        if func.return_type.display != "unit" {
            content.push_str(&format!(
                r#"
<h2 id=\"returns\">Returns</h2>
<p><code>{}</code></p>
"#,
                html_escape(&func.return_type.display)
            ));
        }

        // Effects
        if !func.effects.is_empty() {
            content.push_str("<h2 id=\"effects\">Effects</h2>");
            content.push_str("<ul>");
            for effect in &func.effects {
                content.push_str(&format!("<li><code>{}</code></li>", html_escape(effect)));
            }
            content.push_str("</ul>");
        }

        let breadcrumbs = self.build_breadcrumbs(&func.path);
        let breadcrumbs_refs: Vec<(&str, &str)> = breadcrumbs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        self.base(&format!("fn {}", func.name), &content, &breadcrumbs_refs)
    }

    /// Render type (struct/enum) page
    pub fn render_type(&self, ty: &TypeDoc, module: &ModuleDoc, crate_name: &str) -> String {
        let mut content = String::new();

        let kind_str = ty.kind.as_str();

        // Type header
        content.push_str(&format!(
            r#"
<div class="item-header">
    <h1>{} <code>{}</code></h1>
</div>
"#,
            kind_str,
            html_escape(&ty.name)
        ));

        // Type definition
        content.push_str("<pre class=\"signature\">");
        content.push_str(&self.render_type_definition(ty));
        content.push_str("</pre>");

        // Documentation
        if let Some(ref doc) = ty.doc {
            content.push_str("<div class=\"docblock\">");
            content.push_str(&render_markdown(doc));
            content.push_str("</div>");
        }

        // Fields (for structs)
        if !ty.fields.is_empty() {
            content.push_str("<h2 id=\"fields\">Fields</h2>");
            content.push_str("<div class=\"fields\">");
            for field in &ty.fields {
                content.push_str(&format!(
                    r#"
<div class="field">
    <code class="field-name">{}</code>: <code class="field-type">{}</code>
    {}
</div>"#,
                    html_escape(&field.name),
                    html_escape(&field.ty.display),
                    field
                        .doc
                        .as_ref()
                        .map(|d| format!("<p class=\"field-doc\">{}</p>", render_markdown(d)))
                        .unwrap_or_default()
                ));
            }
            content.push_str("</div>");
        }

        // Variants (for enums)
        if !ty.variants.is_empty() {
            content.push_str("<h2 id=\"variants\">Variants</h2>");
            content.push_str("<div class=\"variants\">");
            for variant in &ty.variants {
                content.push_str(&format!(
                    r#"
<div class="variant">
    <code class="variant-name">{}</code>
    {}
</div>"#,
                    html_escape(&variant.name),
                    variant
                        .doc
                        .as_ref()
                        .map(|d| format!("<p class=\"variant-doc\">{}</p>", render_markdown(d)))
                        .unwrap_or_default()
                ));
            }
            content.push_str("</div>");
        }

        // Methods
        if !ty.methods.is_empty() {
            content.push_str("<h2 id=\"implementations\">Implementations</h2>");
            content.push_str("<div class=\"impl-block\">");
            content.push_str(&format!("<h3>impl {}</h3>", html_escape(&ty.name)));
            for method in &ty.methods {
                content.push_str(&self.render_method_summary(method));
            }
            content.push_str("</div>");
        }

        // Trait implementations
        if !ty.trait_impls.is_empty() {
            content.push_str("<h2 id=\"trait-implementations\">Trait Implementations</h2>");
            for impl_doc in &ty.trait_impls {
                content.push_str(&format!(
                    r#"
<div class="impl-block">
    <h3>impl {} for {}</h3>
</div>"#,
                    html_escape(&impl_doc.trait_path),
                    html_escape(&ty.name)
                ));
            }
        }

        let breadcrumbs = self.build_breadcrumbs(&ty.path);
        let breadcrumbs_refs: Vec<(&str, &str)> = breadcrumbs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        self.base(
            &format!("{} {}", kind_str, ty.name),
            &content,
            &breadcrumbs_refs,
        )
    }

    /// Render trait page
    pub fn render_trait(
        &self,
        trait_doc: &TraitDoc,
        module: &ModuleDoc,
        crate_name: &str,
    ) -> String {
        let mut content = String::new();

        // Trait header
        content.push_str(&format!(
            r#"
<div class="item-header">
    <h1>Trait <code>{}</code></h1>
</div>
"#,
            html_escape(&trait_doc.name)
        ));

        // Trait definition
        content.push_str("<pre class=\"signature\">");
        content.push_str(&self.render_trait_definition(trait_doc));
        content.push_str("</pre>");

        // Documentation
        if let Some(ref doc) = trait_doc.doc {
            content.push_str("<div class=\"docblock\">");
            content.push_str(&render_markdown(doc));
            content.push_str("</div>");
        }

        // Associated Types
        if !trait_doc.assoc_types.is_empty() {
            content.push_str("<h2 id=\"associated-types\">Associated Types</h2>");
            for assoc_type in &trait_doc.assoc_types {
                content.push_str(&format!(
                    r#"
<div class="associated-type">
    <code>type {}</code>
    {}
</div>"#,
                    html_escape(&assoc_type.name),
                    assoc_type
                        .doc
                        .as_ref()
                        .map(|d| format!("<p>{}</p>", render_markdown(d)))
                        .unwrap_or_default()
                ));
            }
        }

        // Required Methods
        if !trait_doc.required_methods.is_empty() {
            content.push_str("<h2 id=\"required-methods\">Required Methods</h2>");
            for method in &trait_doc.required_methods {
                content.push_str(&self.render_method_detail(method));
            }
        }

        // Provided Methods
        if !trait_doc.provided_methods.is_empty() {
            content.push_str("<h2 id=\"provided-methods\">Provided Methods</h2>");
            for method in &trait_doc.provided_methods {
                content.push_str(&self.render_method_detail(method));
            }
        }

        // Implementors
        if !trait_doc.implementors.is_empty() {
            content.push_str("<h2 id=\"implementors\">Implementors</h2>");
            content.push_str("<ul class=\"implementors-list\">");
            for implementor in &trait_doc.implementors {
                content.push_str(&format!(
                    r#"
<li>{}</li>"#,
                    html_escape(implementor)
                ));
            }
            content.push_str("</ul>");
        }

        let breadcrumbs = self.build_breadcrumbs(&trait_doc.path);
        let breadcrumbs_refs: Vec<(&str, &str)> = breadcrumbs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        self.base(
            &format!("trait {}", trait_doc.name),
            &content,
            &breadcrumbs_refs,
        )
    }

    /// Render constant page
    pub fn render_constant(
        &self,
        const_doc: &ConstantDoc,
        module: &ModuleDoc,
        crate_name: &str,
    ) -> String {
        let mut content = String::new();

        content.push_str(&format!(
            r#"
<div class="item-header">
    <h1>Constant <code>{}</code></h1>
</div>
"#,
            html_escape(&const_doc.name)
        ));

        content.push_str(&format!(
            r#"
<pre class="signature">pub const {}: {} = {};</pre>
"#,
            html_escape(&const_doc.name),
            html_escape(&const_doc.ty.display),
            html_escape(const_doc.value.as_deref().unwrap_or("..."))
        ));

        if let Some(ref doc) = const_doc.doc {
            content.push_str("<div class=\"docblock\">");
            content.push_str(&render_markdown(doc));
            content.push_str("</div>");
        }

        let breadcrumbs = self.build_breadcrumbs(&const_doc.path);
        let breadcrumbs_refs: Vec<(&str, &str)> = breadcrumbs
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        self.base(
            &format!("const {}", const_doc.name),
            &content,
            &breadcrumbs_refs,
        )
    }

    /// Render source code page
    pub fn render_source(&self, path: &Path, highlighted: &str) -> String {
        let content = format!(
            r#"
<div class="source-header">
    <h1>Source: {}</h1>
</div>
<pre class="source-code"><code>{}</code></pre>
"#,
            html_escape(&path.display().to_string()),
            highlighted
        );

        self.base(&format!("Source: {}", path.display()), &content, &[])
    }

    // Helper methods

    fn render_type_list(&self, types: &[&TypeDoc], prefix: &str) -> String {
        let mut html = String::from("<ul class=\"item-list\">");
        for ty in types {
            html.push_str(&format!(
                r#"
<li>
    <a href="{}.{}.html" class="item-name {}">{}</a>
    <span class="item-desc">{}</span>
</li>"#,
                prefix,
                html_escape(&ty.name),
                prefix,
                html_escape(&ty.name),
                html_escape(ty.doc.as_deref().unwrap_or(""))
            ));
        }
        html.push_str("</ul>");
        html
    }

    fn render_trait_list(&self, traits: &[TraitDoc]) -> String {
        let mut html = String::from("<ul class=\"item-list\">");
        for t in traits {
            html.push_str(&format!(
                r#"
<li>
    <a href="trait.{}.html" class="item-name trait">{}</a>
    <span class="item-desc">{}</span>
</li>"#,
                html_escape(&t.name),
                html_escape(&t.name),
                html_escape(t.doc.as_deref().unwrap_or(""))
            ));
        }
        html.push_str("</ul>");
        html
    }

    fn render_function_list(&self, functions: &[FunctionDoc]) -> String {
        let mut html = String::from("<ul class=\"item-list\">");
        for f in functions {
            html.push_str(&format!(
                r#"
<li>
    <a href="fn.{}.html" class="item-name fn">{}</a>
    <span class="signature-brief">{}</span>
    <span class="item-desc">{}</span>
</li>"#,
                html_escape(&f.name),
                html_escape(&f.name),
                html_escape(&self.brief_signature(f)),
                html_escape(f.doc.as_deref().unwrap_or(""))
            ));
        }
        html.push_str("</ul>");
        html
    }

    fn render_constant_list(&self, constants: &[ConstantDoc]) -> String {
        let mut html = String::from("<ul class=\"item-list\">");
        for c in constants {
            html.push_str(&format!(
                r#"
<li>
    <a href="const.{}.html" class="item-name const">{}</a>: <code>{}</code>
    <span class="item-desc">{}</span>
</li>"#,
                html_escape(&c.name),
                html_escape(&c.name),
                html_escape(&c.ty.display),
                html_escape(c.doc.as_deref().unwrap_or(""))
            ));
        }
        html.push_str("</ul>");
        html
    }

    fn render_method_summary(&self, method: &FunctionDoc) -> String {
        format!(
            "<div class=\"method\">\n\
             <a href=\"#method.{}\" class=\"method-name\">{}</a>\n\
             <code class=\"signature\">{}</code>\n\
             </div>",
            html_escape(&method.name),
            html_escape(&method.name),
            html_escape(&method.signature)
        )
    }

    fn render_method_detail(&self, method: &FunctionDoc) -> String {
        let mut html = format!(
            r#"
<div class="method-detail" id="method.{}">
    <h4><code>{}</code></h4>
"#,
            html_escape(&method.name),
            html_escape(&method.signature)
        );

        if let Some(ref doc) = method.doc {
            html.push_str("<div class=\"docblock\">");
            html.push_str(&render_markdown(doc));
            html.push_str("</div>");
        }

        html.push_str("</div>");
        html
    }

    fn render_type_definition(&self, ty: &TypeDoc) -> String {
        match ty.kind {
            TypeKind::Struct => {
                let mut def = String::new();
                if ty.modifiers.linear {
                    def.push_str("linear ");
                }
                if ty.modifiers.affine {
                    def.push_str("affine ");
                }
                def.push_str("pub struct ");
                def.push_str(&ty.name);
                if !ty.type_params.is_empty() {
                    def.push('<');
                    def.push_str(
                        &ty.type_params
                            .iter()
                            .map(|tp| tp.name.clone())
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                    def.push('>');
                }
                def.push_str(" {\n");
                for field in &ty.fields {
                    def.push_str(&format!("    {}: {},\n", field.name, field.ty.display));
                }
                def.push('}');
                html_escape(&def)
            }
            TypeKind::Enum => {
                let mut def = String::new();
                if ty.modifiers.linear {
                    def.push_str("linear ");
                }
                if ty.modifiers.affine {
                    def.push_str("affine ");
                }
                def.push_str("pub enum ");
                def.push_str(&ty.name);
                if !ty.type_params.is_empty() {
                    def.push('<');
                    def.push_str(
                        &ty.type_params
                            .iter()
                            .map(|tp| tp.name.clone())
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                    def.push('>');
                }
                def.push_str(" {\n");
                for variant in &ty.variants {
                    def.push_str(&format!("    {},\n", variant.name));
                }
                def.push('}');
                html_escape(&def)
            }
            TypeKind::TypeAlias => html_escape(&format!("pub type {} = ...", ty.name)),
            TypeKind::Union => html_escape(&format!("pub union {} {{ ... }}", ty.name)),
        }
    }

    fn render_trait_definition(&self, t: &TraitDoc) -> String {
        let mut def = String::from("pub trait ");
        def.push_str(&t.name);
        if !t.type_params.is_empty() {
            def.push('<');
            def.push_str(
                &t.type_params
                    .iter()
                    .map(|tp| tp.name.clone())
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            def.push('>');
        }
        if !t.super_traits.is_empty() {
            def.push_str(": ");
            def.push_str(&t.super_traits.join(" + "));
        }
        def.push_str(" {\n");

        for at in &t.assoc_types {
            def.push_str(&format!("    type {};\n", at.name));
        }

        for method in &t.required_methods {
            let sig = method.signature.trim_start_matches("pub ");
            def.push_str(&format!("    {};\n", sig));
        }

        def.push('}');
        html_escape(&def)
    }

    fn brief_signature(&self, f: &FunctionDoc) -> String {
        let params: String = f
            .params
            .iter()
            .map(|p| {
                if p.is_self {
                    match p.self_kind {
                        Some(SelfKind::Ref) => "&self".to_string(),
                        Some(SelfKind::RefMut) => "&!self".to_string(),
                        _ => "self".to_string(),
                    }
                } else {
                    format!("{}: {}", p.name, p.ty.display)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        format!("({}) -> {}", params, f.return_type.display)
    }

    fn build_breadcrumbs(&self, path: &str) -> Vec<(String, String)> {
        let parts: Vec<&str> = path.split("::").collect();
        let mut breadcrumbs = Vec::new();
        let mut current_path = String::new();

        for (i, part) in parts.iter().enumerate() {
            if i == 0 {
                current_path = part.to_string();
                breadcrumbs.push((part.to_string(), "/index.html".to_string()));
            } else {
                current_path = format!("{}::{}", current_path, part);
                let href = if i == parts.len() - 1 {
                    "#".to_string()
                } else {
                    format!("/{}/index.html", parts[1..=i].join("/"))
                };
                breadcrumbs.push((part.to_string(), href));
            }
        }

        breadcrumbs
    }
}

/// HTML escape
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Render markdown to HTML (simplified)
fn render_markdown(content: &str) -> String {
    // Simple markdown rendering
    // In a real implementation, use pulldown-cmark or similar
    let mut html = String::new();
    let mut in_code_block = false;
    let mut code_lang = String::new();
    let mut code_content = String::new();

    for line in content.lines() {
        if line.starts_with("```") {
            if in_code_block {
                // End code block
                html.push_str(&format!(
                    "<pre><code class=\"language-{}\">{}</code></pre>\n",
                    code_lang,
                    html_escape(&code_content)
                ));
                code_content.clear();
                in_code_block = false;
            } else {
                // Start code block
                code_lang = line.trim_start_matches("```").trim().to_string();
                if code_lang.is_empty() {
                    code_lang = "d".to_string();
                }
                in_code_block = true;
            }
            continue;
        }

        if in_code_block {
            if !code_content.is_empty() {
                code_content.push('\n');
            }
            code_content.push_str(line);
            continue;
        }

        // Headers
        if line.starts_with("### ") {
            html.push_str(&format!("<h3>{}</h3>\n", html_escape(&line[4..])));
        } else if line.starts_with("## ") {
            html.push_str(&format!("<h2>{}</h2>\n", html_escape(&line[3..])));
        } else if line.starts_with("# ") {
            html.push_str(&format!("<h1>{}</h1>\n", html_escape(&line[2..])));
        } else if line.starts_with("- ") || line.starts_with("* ") {
            html.push_str(&format!("<li>{}</li>\n", html_escape(&line[2..])));
        } else if line.trim().is_empty() {
            html.push_str("<br>\n");
        } else {
            // Inline code
            let processed = process_inline_code(line);
            html.push_str(&format!("<p>{}</p>\n", processed));
        }
    }

    html
}

/// Process inline code in text
fn process_inline_code(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '`' {
            let mut code = String::new();
            while let Some(&next) = chars.peek() {
                if next == '`' {
                    chars.next();
                    break;
                }
                code.push(chars.next().unwrap());
            }
            result.push_str(&format!("<code>{}</code>", html_escape(&code)));
        } else if c == '<' {
            result.push_str("&lt;");
        } else if c == '>' {
            result.push_str("&gt;");
        } else if c == '&' {
            result.push_str("&amp;");
        } else {
            result.push(c);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<test>"), "&lt;test&gt;");
        assert_eq!(html_escape("a & b"), "a &amp; b");
    }

    #[test]
    fn test_render_markdown_header() {
        let result = render_markdown("# Title");
        assert!(result.contains("<h1>Title</h1>"));
    }

    #[test]
    fn test_render_markdown_code() {
        let result = render_markdown("```d\nlet x = 1\n```");
        assert!(result.contains("<code"));
        assert!(result.contains("let x = 1"));
    }
}
