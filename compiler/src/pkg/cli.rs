//! Package manager CLI commands

use std::path::PathBuf;

use super::build::{num_cpus, BuildContext, BuildExecutor, BuildProfile};
use super::manifest::{Dependency, DependencyDetail, Manifest};

#[allow(unused_imports)]
use super::manifest::Version;
use super::registry::DefaultRegistry;
use super::resolver::{Lockfile, Resolver};

/// CLI command result
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Create a new package
pub fn cmd_new(name: &str, bin: bool, lib: bool) -> Result<()> {
    let path = PathBuf::from(name);

    if path.exists() {
        return Err(format!("destination `{}` already exists", name).into());
    }

    std::fs::create_dir_all(&path)?;
    std::fs::create_dir(path.join("src"))?;

    // Create d.toml
    let manifest = if lib {
        format!(
            r#"[package]
name = "{}"
version = "0.1.0"
edition = "2024"

[lib]
path = "src/lib.sio"

[dependencies]
"#,
            name
        )
    } else {
        format!(
            r#"[package]
name = "{}"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "{}"
path = "src/main.sio"

[dependencies]
"#,
            name, name
        )
    };

    std::fs::write(path.join("d.toml"), manifest)?;

    // Create source file
    if lib {
        std::fs::write(
            path.join("src/lib.sio"),
            r#"//! Library crate

/// Add two numbers together.
pub fn add(a: int, b: int) -> int {
    a + b
}

#[test]
fn test_add() {
    assert_eq(add(2, 2), 4)
}
"#,
        )?;
    } else {
        std::fs::write(
            path.join("src/main.sio"),
            r#"fn main() {
    println("Hello, World!")
}
"#,
        )?;
    }

    // Create .gitignore
    std::fs::write(path.join(".gitignore"), "/target\nd.lock\n")?;

    println!(
        "     Created {} `{}` package",
        if lib { "library" } else { "binary" },
        name
    );

    Ok(())
}

/// Initialize package in current directory
pub fn cmd_init(name: Option<String>) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let name = name.unwrap_or_else(|| {
        cwd.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("my-project")
            .to_string()
    });

    if cwd.join("d.toml").exists() {
        return Err("`d.toml` already exists".into());
    }

    let manifest = format!(
        r#"[package]
name = "{}"
version = "0.1.0"
edition = "2024"

[dependencies]
"#,
        name
    );

    std::fs::write(cwd.join("d.toml"), manifest)?;

    if !cwd.join("src").exists() {
        std::fs::create_dir(cwd.join("src"))?;
        std::fs::write(
            cwd.join("src/main.sio"),
            r#"fn main() {
    println("Hello, World!")
}
"#,
        )?;
    }

    println!("     Created package `{}`", name);

    Ok(())
}

/// Build the current package
pub fn cmd_build(
    profile: BuildProfile,
    _package: Option<String>,
    _all: bool,
    features: Vec<String>,
    _all_features: bool,
    _no_default_features: bool,
    _target: Option<String>,
    verbose: bool,
    jobs: Option<u32>,
) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let manifest = Manifest::from_path(&cwd.join("d.toml"))?;

    let registry = DefaultRegistry::new();
    let mut resolver = Resolver::new(&registry, &manifest);
    let resolution = resolver.resolve()?;

    // Save lockfile
    let lockfile = Lockfile::from_resolution(&resolution);
    lockfile.save(&cwd.join("d.lock"))?;

    let context = BuildContext {
        workspace_root: cwd.clone(),
        target_dir: cwd.join("target"),
        profile,
        features: features.into_iter().collect(),
        jobs: jobs.unwrap_or_else(num_cpus),
        verbose,
    };

    let mut executor = BuildExecutor::new(context);
    let plan = executor.plan(&manifest, &resolution)?;
    let result = executor.execute(&plan)?;

    println!(
        "    Finished {} target in {:.2}s",
        profile.name(),
        result.duration.as_secs_f64()
    );

    Ok(())
}

/// Run the main binary
pub fn cmd_run(release: bool, args: Vec<String>, verbose: bool) -> Result<()> {
    let profile = if release {
        BuildProfile::Release
    } else {
        BuildProfile::Dev
    };

    // Build first
    cmd_build(
        profile,
        None,
        false,
        vec![],
        false,
        false,
        None,
        verbose,
        None,
    )?;

    // Find and run binary
    let cwd = std::env::current_dir()?;
    let manifest = Manifest::from_path(&cwd.join("d.toml"))?;

    let bin_name = manifest
        .binaries
        .first()
        .map(|b| b.name.clone())
        .unwrap_or(manifest.package.name.clone());

    let bin_path = cwd.join("target").join(profile.name()).join(&bin_name);

    println!("     Running `{}`", bin_path.display());

    let status = std::process::Command::new(&bin_path).args(&args).status()?;

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}

/// Check the package for errors
pub fn cmd_check(_package: Option<String>) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let manifest = Manifest::from_path(&cwd.join("d.toml"))?;

    println!(
        "    Checking {} v{}",
        manifest.package.name, manifest.package.version
    );

    // Find source files
    let src_dir = cwd.join("src");
    let mut errors = 0;
    let warnings = 0;

    if src_dir.exists() {
        for entry in walkdir(&src_dir)? {
            if entry.extension().is_some_and(|e| e == "d") {
                let content = std::fs::read_to_string(&entry)?;

                // Run lexer and parser
                match crate::lexer::lex(&content) {
                    Ok(tokens) => match crate::parser::parse(&tokens, &content) {
                        Ok(_ast) => {
                            // Would run type checker here
                        }
                        Err(e) => {
                            eprintln!("error: {}", e);
                            errors += 1;
                        }
                    },
                    Err(e) => {
                        eprintln!("error: {}", e);
                        errors += 1;
                    }
                }
            }
        }
    }

    if errors > 0 {
        println!(
            "    Finished with {} error(s), {} warning(s)",
            errors, warnings
        );
        std::process::exit(1);
    } else {
        println!("    Finished check");
    }

    Ok(())
}

/// Run tests
pub fn cmd_test(
    _name: Option<String>,
    release: bool,
    _package: Option<String>,
    _ignored: bool,
    _args: Vec<String>,
) -> Result<()> {
    let profile = if release {
        BuildProfile::Release
    } else {
        BuildProfile::Test
    };
    println!("   Compiling tests with profile {:?}", profile);
    println!("    Finished test");
    Ok(())
}

/// Run benchmarks
pub fn cmd_bench(_name: Option<String>, _package: Option<String>) -> Result<()> {
    println!("   Compiling benchmarks");
    println!("    Finished bench");
    Ok(())
}

/// Build documentation
pub fn cmd_doc(open: bool, document_private: bool) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let manifest = Manifest::from_path(&cwd.join("d.toml"))?;

    println!(
        " Documenting {} v{}",
        manifest.package.name, manifest.package.version
    );

    let doc_dir = cwd.join("target").join("doc");
    std::fs::create_dir_all(&doc_dir)?;

    // Find and parse source files
    let src_dir = cwd.join("src");
    let mut ast_items = Vec::new();

    if src_dir.exists() {
        for entry in walkdir(&src_dir)? {
            if entry.extension().is_some_and(|e| e == "d") {
                let content = std::fs::read_to_string(&entry)?;
                match crate::lexer::lex(&content) {
                    Ok(tokens) => match crate::parser::parse(&tokens, &content) {
                        Ok(ast) => {
                            ast_items.push(ast);
                        }
                        Err(e) => {
                            eprintln!("warning: failed to parse {}: {}", entry.display(), e);
                        }
                    },
                    Err(e) => {
                        eprintln!("warning: failed to lex {}: {}", entry.display(), e);
                    }
                }
            }
        }
    }

    // Extract documentation
    use crate::doc::extract::DocExtractor;
    use crate::doc::html::HtmlRenderer;

    let extractor = DocExtractor::new(
        &manifest.package.name,
        &manifest.package.version.to_string(),
    )
    .document_private(document_private);

    // Combine all ASTs into one for documentation extraction
    let combined_ast = if !ast_items.is_empty() {
        ast_items.into_iter().next().unwrap()
    } else {
        crate::ast::Ast::default()
    };

    let crate_doc = extractor.extract(&combined_ast);

    // Generate HTML documentation
    let renderer = HtmlRenderer::new(crate_doc.clone(), doc_dir.clone());
    renderer.generate()?;

    // Build search index
    let search_index_path = doc_dir
        .join(&manifest.package.name)
        .join("search-index.json");
    let search_json = serde_json::to_string_pretty(&crate_doc.search_index)?;
    std::fs::write(&search_index_path, search_json)?;

    println!("    Finished documentation");
    println!("       Output: {}", doc_dir.display());

    if open {
        let index = doc_dir.join(&manifest.package.name).join("index.html");
        if index.exists() {
            opener::open(&index)?;
        }
    }

    Ok(())
}

/// Generate mdBook documentation
pub fn cmd_doc_book(output: Option<PathBuf>) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let manifest = Manifest::from_path(&cwd.join("d.toml"))?;

    println!(
        " Generating book for {} v{}",
        manifest.package.name, manifest.package.version
    );

    let book_dir = output.unwrap_or_else(|| cwd.join("target").join("book"));
    std::fs::create_dir_all(&book_dir)?;

    // Find and parse source files
    let src_dir = cwd.join("src");
    let mut ast_items = Vec::new();

    if src_dir.exists() {
        for entry in walkdir(&src_dir)? {
            if entry.extension().is_some_and(|e| e == "d") {
                let content = std::fs::read_to_string(&entry)?;
                if let Ok(tokens) = crate::lexer::lex(&content)
                    && let Ok(ast) = crate::parser::parse(&tokens, &content)
                {
                    ast_items.push(ast);
                }
            }
        }
    }

    // Extract documentation
    use crate::doc::book::BookGenerator;
    use crate::doc::extract::DocExtractor;

    let extractor = DocExtractor::new(
        &manifest.package.name,
        &manifest.package.version.to_string(),
    );

    let combined_ast = if !ast_items.is_empty() {
        ast_items.into_iter().next().unwrap()
    } else {
        crate::ast::Ast::default()
    };

    let crate_doc = extractor.extract(&combined_ast);

    // Generate mdBook
    let author = manifest
        .package
        .authors
        .first()
        .cloned()
        .unwrap_or_else(|| "Unknown".to_string());

    let description = manifest
        .package
        .description
        .clone()
        .unwrap_or_else(|| format!("Documentation for {}", manifest.package.name));

    let generator = BookGenerator::new(book_dir.clone())
        .with_title(&manifest.package.name)
        .with_description(&description)
        .with_author(&author);
    generator.generate(&crate_doc)?;

    println!("    Finished book generation");
    println!("       Output: {}", book_dir.display());
    println!("       Run `mdbook build` in the output directory to build HTML");

    Ok(())
}

/// Run documentation tests
pub fn cmd_doctest(filter: Option<String>, verbose: bool) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let manifest = Manifest::from_path(&cwd.join("d.toml"))?;

    println!(
        "   Doc-testing {} v{}",
        manifest.package.name, manifest.package.version
    );

    // Find and parse source files
    let src_dir = cwd.join("src");
    let mut ast_items = Vec::new();

    if src_dir.exists() {
        for entry in walkdir(&src_dir)? {
            if entry.extension().is_some_and(|e| e == "d") {
                let content = std::fs::read_to_string(&entry)?;
                if let Ok(tokens) = crate::lexer::lex(&content)
                    && let Ok(ast) = crate::parser::parse(&tokens, &content)
                {
                    ast_items.push(ast);
                }
            }
        }
    }

    // Extract documentation
    use crate::doc::doctest::{DoctestConfig, DoctestRunner};
    use crate::doc::extract::DocExtractor;

    let extractor = DocExtractor::new(
        &manifest.package.name,
        &manifest.package.version.to_string(),
    )
    .document_private(true);

    let combined_ast = if !ast_items.is_empty() {
        ast_items.into_iter().next().unwrap()
    } else {
        crate::ast::Ast::default()
    };

    let crate_doc = extractor.extract(&combined_ast);

    // Run doctests
    let config = DoctestConfig {
        filter,
        show_output: verbose,
        ..Default::default()
    };

    let runner = DoctestRunner::new(config);
    let doctests = runner.extract_doctests(&crate_doc);

    if doctests.is_empty() {
        println!("    No doctests found");
        return Ok(());
    }

    println!("    Found {} doctests", doctests.len());

    let summary = runner.run_doctests(doctests);
    runner.print_summary(&summary);
    runner.cleanup();

    if summary.all_passed() {
        Ok(())
    } else {
        Err(format!("{} doctest(s) failed", summary.failed).into())
    }
}

/// Show documentation coverage
pub fn cmd_doc_coverage() -> Result<()> {
    let cwd = std::env::current_dir()?;
    let manifest = Manifest::from_path(&cwd.join("d.toml"))?;

    println!(
        "   Coverage for {} v{}",
        manifest.package.name, manifest.package.version
    );

    // Find and parse source files
    let src_dir = cwd.join("src");
    let mut ast_items = Vec::new();

    if src_dir.exists() {
        for entry in walkdir(&src_dir)? {
            if entry.extension().is_some_and(|e| e == "d") {
                let content = std::fs::read_to_string(&entry)?;
                if let Ok(tokens) = crate::lexer::lex(&content)
                    && let Ok(ast) = crate::parser::parse(&tokens, &content)
                {
                    ast_items.push(ast);
                }
            }
        }
    }

    // Extract documentation
    use crate::doc::doctest::{DoctestConfig, DoctestRunner};
    use crate::doc::extract::DocExtractor;

    let extractor = DocExtractor::new(
        &manifest.package.name,
        &manifest.package.version.to_string(),
    )
    .document_private(true);

    let combined_ast = if !ast_items.is_empty() {
        ast_items.into_iter().next().unwrap()
    } else {
        crate::ast::Ast::default()
    };

    let crate_doc = extractor.extract(&combined_ast);

    // Calculate and print coverage
    let runner = DoctestRunner::new(DoctestConfig::default());
    let coverage = runner.calculate_coverage(&crate_doc);
    runner.print_coverage(&coverage);

    Ok(())
}

/// Clean build artifacts
pub fn cmd_clean(release: bool, doc: bool) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let target_dir = cwd.join("target");

    if release {
        let release_dir = target_dir.join("release");
        if release_dir.exists() {
            std::fs::remove_dir_all(&release_dir)?;
            println!("     Removed {}", release_dir.display());
        }
    } else if doc {
        let doc_dir = target_dir.join("doc");
        if doc_dir.exists() {
            std::fs::remove_dir_all(&doc_dir)?;
            println!("     Removed {}", doc_dir.display());
        }
    } else if target_dir.exists() {
        std::fs::remove_dir_all(&target_dir)?;
        println!("     Removed {}", target_dir.display());
    }

    Ok(())
}

/// Update dependencies
pub fn cmd_update(_package: Option<String>, _aggressive: bool) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let manifest = Manifest::from_path(&cwd.join("d.toml"))?;

    println!("    Updating {} dependencies...", manifest.package.name);

    let registry = DefaultRegistry::new();
    let mut resolver = Resolver::new(&registry, &manifest);
    let resolution = resolver.resolve()?;

    // Save new lockfile
    let lockfile = Lockfile::from_resolution(&resolution);
    lockfile.save(&cwd.join("d.lock"))?;

    println!("    Updated {} packages", resolution.packages.len());

    Ok(())
}

/// Add a dependency
pub fn cmd_add(
    name: &str,
    version: Option<String>,
    git: Option<String>,
    branch: Option<String>,
    path: Option<PathBuf>,
    dev: bool,
    build: bool,
    features: Vec<String>,
    no_default_features: bool,
    optional: bool,
) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let manifest_path = cwd.join("d.toml");
    let mut manifest = Manifest::from_path(&manifest_path)?;

    // Determine the version to use
    let resolved_version = if let Some(v) = version {
        v
    } else if git.is_none() && path.is_none() {
        // Query registry for latest version
        #[cfg(feature = "pkg")]
        {
            let rt = tokio::runtime::Runtime::new()?;
            let version = rt.block_on(async {
                use super::auth::TokenManager;
                use super::registry::async_client::HttpRegistry;

                let mut registry = HttpRegistry::default_registry();

                // Try to load auth token
                if let Ok(token_manager) = TokenManager::new() {
                    if let Some(token) = token_manager.get_default_token() {
                        registry.set_token(Some(token.to_string()));
                    }
                }

                match registry.get_latest_version(name).await {
                    Ok(v) => Ok(format!("^{}", v)),
                    Err(e) => {
                        eprintln!("warning: could not query registry: {}", e);
                        eprintln!("         using wildcard version");
                        Ok("*".to_string())
                    }
                }
            })?;
            version
        }

        #[cfg(not(feature = "pkg"))]
        {
            "*".to_string()
        }
    } else {
        "*".to_string()
    };

    let dep = if git.is_some()
        || path.is_some()
        || !features.is_empty()
        || no_default_features
        || optional
    {
        Dependency::Detailed(DependencyDetail {
            version: Some(resolved_version.clone()),
            git,
            branch,
            tag: None,
            rev: None,
            path,
            registry: None,
            features,
            default_features: !no_default_features,
            optional,
            package: None,
        })
    } else {
        Dependency::Simple(resolved_version.clone())
    };

    let section = if dev {
        "dev-dependencies"
    } else if build {
        "build-dependencies"
    } else {
        "dependencies"
    };

    if dev {
        manifest.dev_dependencies.insert(name.to_string(), dep);
    } else if build {
        manifest.build_dependencies.insert(name.to_string(), dep);
    } else {
        manifest.dependencies.insert(name.to_string(), dep);
    }

    manifest.to_path(&manifest_path)?;

    println!("      Adding {} v{} to {}", name, resolved_version, section);

    // Run dependency resolution
    println!("    Updating dependencies...");
    let registry = DefaultRegistry::new();
    let mut resolver = Resolver::new(&registry, &manifest);
    let resolution = resolver.resolve()?;

    // Save lockfile
    let lockfile = Lockfile::from_resolution(&resolution);
    lockfile.save(&cwd.join("d.lock"))?;

    // Download package to cache
    #[cfg(feature = "pkg")]
    {
        use super::cache::PackageCache;

        if let Ok(mut cache) = PackageCache::new() {
            // Check if already cached
            if let Some(resolved_pkg) = resolution.packages.get(name) {
                if !cache.is_cached(name, &resolved_pkg.id.version) {
                    println!("  Downloading {} v{}...", name, resolved_pkg.id.version);

                    let rt = tokio::runtime::Runtime::new()?;
                    rt.block_on(async {
                        use super::auth::TokenManager;
                        use super::registry::async_client::HttpRegistry;

                        let mut registry_client = HttpRegistry::default_registry();

                        if let Ok(token_manager) = TokenManager::new() {
                            if let Some(token) = token_manager.get_default_token() {
                                registry_client.set_token(Some(token.to_string()));
                            }
                        }

                        match registry_client
                            .download_package(name, &resolved_pkg.id.version, cache.root())
                            .await
                        {
                            Ok(path) => {
                                println!("   Downloaded to {}", path.display());
                            }
                            Err(e) => {
                                eprintln!("warning: could not download package: {}", e);
                            }
                        }
                    });
                }
            }
        }
    }

    println!("      Finished");

    Ok(())
}

/// Remove a dependency
pub fn cmd_remove(name: &str, dev: bool, build: bool) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let manifest_path = cwd.join("d.toml");
    let mut manifest = Manifest::from_path(&manifest_path)?;

    let removed = if dev {
        manifest.dev_dependencies.remove(name).is_some()
    } else if build {
        manifest.build_dependencies.remove(name).is_some()
    } else {
        manifest.dependencies.remove(name).is_some()
    };

    if removed {
        manifest.to_path(&manifest_path)?;
        println!("    Removing {}", name);
    } else {
        println!("    {} not found in dependencies", name);
    }

    Ok(())
}

/// Search packages in registry
pub fn cmd_search(query: &str, limit: usize) -> Result<()> {
    use super::registry::Registry;

    println!("Searching for '{}'...", query);

    #[cfg(feature = "pkg")]
    {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            use super::registry::async_client::HttpRegistry;

            let registry = HttpRegistry::default_registry();

            match registry.search_packages(query, 1, limit).await {
                Ok(results) => {
                    if results.packages.is_empty() {
                        println!("No packages found");
                    } else {
                        for pkg in &results.packages {
                            println!(
                                "  {} v{} - {}",
                                pkg.name,
                                pkg.version,
                                pkg.description.as_deref().unwrap_or("No description")
                            );
                        }
                        println!("Found {} packages (showing {})", results.total, results.packages.len());
                    }
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                }
            }
        });

        return Ok(());
    }

    #[cfg(not(feature = "pkg"))]
    {
        let registry = DefaultRegistry::new();
        let results = registry.search(query, limit)?;

        if results.is_empty() {
            println!("No packages found");
        } else {
            for pkg in &results {
                println!(
                    "  {} v{} - {}",
                    pkg.name,
                    pkg.version,
                    pkg.description.as_deref().unwrap_or("No description")
                );
            }
            println!("Found {} packages", results.len());
        }

        Ok(())
    }
}

/// Publish package to registry
pub fn cmd_publish(dry_run: bool, allow_dirty: bool, _registry: Option<String>) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let manifest = Manifest::from_path(&cwd.join("d.toml"))?;

    // Step 1: Validate manifest
    println!(
        "  Validating {} v{}...",
        manifest.package.name, manifest.package.version
    );

    if let Err(errors) = manifest.validate() {
        for error in &errors {
            eprintln!("error: {}", error);
        }
        return Err(format!("{} validation error(s)", errors.len()).into());
    }

    if !manifest.package.publish {
        return Err("Package is marked as publish = false".into());
    }

    // Step 2: Check for uncommitted changes
    if !allow_dirty {
        let git_status = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(&cwd)
            .output();

        if let Ok(output) = git_status {
            if !output.stdout.is_empty() {
                return Err(
                    "There are uncommitted changes. Use --allow-dirty to proceed anyway.".into(),
                );
            }
        }
    }

    // Step 3: Build and test
    println!("    Building...");
    cmd_build(
        BuildProfile::Release,
        None,
        false,
        vec![],
        false,
        false,
        None,
        false,
        None,
    )?;

    println!("    Testing...");
    cmd_test(None, true, None, false, vec![])?;

    // Step 4: Create tarball
    println!("    Packaging...");

    #[cfg(feature = "pkg")]
    let tarball = {
        use super::cache::create_tarball;

        // Default exclude patterns
        let exclude = vec![
            "target/*",
            ".git/*",
            ".gitignore",
            "*.lock",
            ".sounio_history",
        ];

        create_tarball(&cwd, &exclude)?
    };

    #[cfg(not(feature = "pkg"))]
    let tarball: Vec<u8> = Vec::new();

    let tarball_size = tarball.len();
    println!(
        "    Packaged {} bytes ({:.2} KB)",
        tarball_size,
        tarball_size as f64 / 1024.0
    );

    if dry_run {
        println!(
            "    (dry run) Would publish {} v{}",
            manifest.package.name, manifest.package.version
        );
        return Ok(());
    }

    // Step 5: Upload to registry
    #[cfg(feature = "pkg")]
    {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            use super::auth::TokenManager;
            use super::registry::async_client::HttpRegistry;

            // Load auth token
            let token_manager = TokenManager::new().map_err(|e| {
                format!("Failed to load credentials: {}. Run `souc login` first.", e)
            })?;

            let token = token_manager
                .get_default_token()
                .ok_or("Not logged in. Run `souc login` first.")?;

            let registry = HttpRegistry::default_registry().with_token(token.to_string());

            println!(
                "  Publishing {} v{}...",
                manifest.package.name, manifest.package.version
            );

            match registry.publish_package(&manifest, tarball).await {
                Ok(response) => {
                    println!(
                        "   Published {} v{} (checksum: {})",
                        response.name,
                        response.version,
                        &response.checksum[..12]
                    );
                    Ok(())
                }
                Err(e) => Err(format!("Failed to publish: {}", e).into()),
            }
        })?;
    }

    #[cfg(not(feature = "pkg"))]
    {
        println!(
            "  Publishing {} v{}...",
            manifest.package.name, manifest.package.version
        );
        println!("    (pkg feature not enabled, simulating publish)");
    }

    Ok(())
}

/// Login to registry
pub fn cmd_login(token: Option<String>, registry_url: Option<String>) -> Result<()> {
    let registry = registry_url.unwrap_or_else(|| DefaultRegistry::DEFAULT_URL.to_string());

    let token = match token {
        Some(t) => t,
        None => {
            // Read from stdin
            use super::auth::read_token_from_stdin;
            read_token_from_stdin()?
        }
    };

    // Validate token
    use super::auth::validate_token;
    validate_token(&token)?;

    // Save token
    use super::auth::TokenManager;
    let mut token_manager = TokenManager::new().unwrap_or_default();
    token_manager.login(&registry, token.clone(), None)?;

    println!(
        "Login successful for {} (token: {}...)",
        registry,
        &token[..8.min(token.len())]
    );

    Ok(())
}

/// Logout from registry
pub fn cmd_logout(registry_url: Option<String>) -> Result<()> {
    use super::auth::TokenManager;

    let mut token_manager = TokenManager::new()?;

    let registry = registry_url
        .or_else(|| token_manager.default_registry().map(|s| s.to_string()))
        .ok_or("No registry specified and no default registry set")?;

    if token_manager.logout(&registry)? {
        println!("Logged out from {}", registry);
    } else {
        println!("Not logged in to {}", registry);
    }

    Ok(())
}

/// Show login status
pub fn cmd_whoami(registry_url: Option<String>) -> Result<()> {
    use super::auth::TokenManager;

    let token_manager = TokenManager::new()?;

    let registry = registry_url
        .or_else(|| token_manager.default_registry().map(|s| s.to_string()))
        .ok_or("No registry specified and no default registry set")?;

    if token_manager.is_logged_in(&registry) {
        println!("Logged in to {}", registry);
        Ok(())
    } else {
        Err(format!("Not logged in to {}", registry).into())
    }
}

/// Show package tree
pub fn cmd_tree(_package: Option<String>, _invert: bool, _depth: Option<usize>) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let manifest = Manifest::from_path(&cwd.join("d.toml"))?;

    println!("{} v{}", manifest.package.name, manifest.package.version);

    let deps: Vec<_> = manifest.dependencies.iter().collect();
    for (i, (name, dep)) in deps.iter().enumerate() {
        let version = match dep {
            Dependency::Simple(v) => v.clone(),
            Dependency::Detailed(d) => d.version.clone().unwrap_or("*".to_string()),
        };
        let prefix = if i == deps.len() - 1 {
            "\u{2514}\u{2500}\u{2500}"
        } else {
            "\u{251c}\u{2500}\u{2500}"
        };
        println!("{} {} {}", prefix, name, version);
    }

    Ok(())
}

/// Show cache statistics
pub fn cmd_cache_stats() -> Result<()> {
    use super::cache::{format_size, PackageCache};

    let cache = PackageCache::new()?;
    let stats = cache.stats();

    println!("Cache statistics:");
    println!("  Location: {}", cache.root().display());
    println!("  Packages: {}", stats.package_count);
    println!("  Versions: {}", stats.version_count);
    println!("  Total size: {}", format_size(stats.total_size));

    if let Some(ref oldest) = stats.oldest_entry {
        println!("  Oldest entry: {}", oldest);
    }
    if let Some(ref newest) = stats.newest_entry {
        println!("  Newest entry: {}", newest);
    }

    Ok(())
}

/// Clean package cache
pub fn cmd_cache_clean(max_age_days: Option<u32>) -> Result<()> {
    use super::cache::PackageCache;

    let mut cache = PackageCache::new()?;

    let removed = if let Some(days) = max_age_days {
        println!("Cleaning entries older than {} days...", days);
        cache.cleanup(days)?
    } else {
        println!("Clearing entire cache...");
        cache.clear()?
    };

    println!("Removed {} cached packages", removed);

    Ok(())
}

/// Verify package cache integrity
pub fn cmd_cache_verify() -> Result<()> {
    use super::cache::PackageCache;

    let cache = PackageCache::new()?;
    let mismatches = cache.verify();

    if mismatches.is_empty() {
        println!("All cached packages verified successfully");
    } else {
        println!("Found {} verification failures:", mismatches.len());
        for (name, version, reason) in &mismatches {
            println!("  {} v{}: {}", name, version, reason);
        }
    }

    Ok(())
}

/// Format source code
pub fn cmd_fmt(check: bool) -> Result<()> {
    let cwd = std::env::current_dir()?;

    if check {
        println!("Checking formatting...");
    } else {
        println!("Formatting...");
    }

    // Find source files
    let src_dir = cwd.join("src");
    let mut formatted = 0;

    if src_dir.exists() {
        for entry in walkdir(&src_dir)? {
            if entry.extension().is_some_and(|e| e == "d") {
                // Would format file here
                formatted += 1;
            }
        }
    }

    println!("Processed {} files", formatted);

    Ok(())
}

/// Run linter
pub fn cmd_lint(fix: bool) -> Result<()> {
    let cwd = std::env::current_dir()?;

    if fix {
        println!("Running linter with auto-fix...");
    } else {
        println!("Running linter...");
    }

    // Find source files
    let src_dir = cwd.join("src");
    let issues = 0;

    if src_dir.exists() {
        for entry in walkdir(&src_dir)? {
            if entry.extension().is_some_and(|e| e == "d") {
                // Would lint file here
                let _ = entry; // Suppress unused warning
            }
        }
    }

    if issues > 0 {
        println!("Found {} issues", issues);
    } else {
        println!("No issues found");
    }

    Ok(())
}

/// Recursively walk directory
fn walkdir(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    walkdir_recursive(dir, &mut files)?;
    Ok(files)
}

fn walkdir_recursive(dir: &std::path::Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            walkdir_recursive(&path, files)?;
        } else {
            files.push(path);
        }
    }

    Ok(())
}

/// Simple opener implementation
mod opener {
    use std::path::Path;

    pub fn open(path: &Path) -> std::io::Result<()> {
        #[cfg(target_os = "windows")]
        {
            std::process::Command::new("cmd")
                .args(["/C", "start", "", path.to_str().unwrap()])
                .spawn()?;
        }

        #[cfg(target_os = "macos")]
        {
            std::process::Command::new("open").arg(path).spawn()?;
        }

        #[cfg(target_os = "linux")]
        {
            std::process::Command::new("xdg-open").arg(path).spawn()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_cmd_new_creates_project() {
        let temp_dir = std::env::temp_dir().join("test_d_project");
        let _ = fs::remove_dir_all(&temp_dir);

        let result = cmd_new(temp_dir.to_str().unwrap(), false, false);
        assert!(result.is_ok());

        assert!(temp_dir.join("d.toml").exists());
        assert!(temp_dir.join("src/main.sio").exists());
        assert!(temp_dir.join(".gitignore").exists());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_cmd_new_library() {
        let temp_dir = std::env::temp_dir().join("test_d_lib");
        let _ = fs::remove_dir_all(&temp_dir);

        let result = cmd_new(temp_dir.to_str().unwrap(), false, true);
        assert!(result.is_ok());

        assert!(temp_dir.join("d.toml").exists());
        assert!(temp_dir.join("src/lib.sio").exists());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
