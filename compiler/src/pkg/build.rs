//! Build system for D packages
//!
//! Handles compilation, linking, and caching.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::manifest::Manifest;
use super::resolver::Resolution;

/// Build context
pub struct BuildContext {
    /// Workspace root
    pub workspace_root: PathBuf,

    /// Target directory
    pub target_dir: PathBuf,

    /// Build profile
    pub profile: BuildProfile,

    /// Enabled features
    pub features: HashSet<String>,

    /// Build jobs (parallelism)
    pub jobs: u32,

    /// Verbose output
    pub verbose: bool,
}

/// Build profile
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildProfile {
    Dev,
    Release,
    Test,
    Bench,
}

impl BuildProfile {
    pub fn name(&self) -> &str {
        match self {
            BuildProfile::Dev => "dev",
            BuildProfile::Release => "release",
            BuildProfile::Test => "test",
            BuildProfile::Bench => "bench",
        }
    }

    pub fn opt_level(&self) -> u8 {
        match self {
            BuildProfile::Dev | BuildProfile::Test => 0,
            BuildProfile::Release | BuildProfile::Bench => 3,
        }
    }

    pub fn debug(&self) -> bool {
        match self {
            BuildProfile::Dev | BuildProfile::Test => true,
            BuildProfile::Release | BuildProfile::Bench => false,
        }
    }
}

/// Build plan
pub struct BuildPlan {
    /// Units to compile in order
    pub units: Vec<CompileUnit>,

    /// Linking steps
    pub links: Vec<LinkUnit>,
}

/// A compilation unit
#[derive(Debug, Clone)]
pub struct CompileUnit {
    /// Package name
    pub package: String,

    /// Source files
    pub sources: Vec<PathBuf>,

    /// Output path
    pub output: PathBuf,

    /// Dependencies (package names)
    pub deps: Vec<String>,

    /// Compiler flags
    pub flags: CompileFlags,

    /// Is this a test/bench
    pub mode: CompileMode,
}

/// Compilation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileMode {
    Build,
    Test,
    Bench,
    Doc,
    Check,
}

/// Compiler flags
#[derive(Debug, Clone)]
pub struct CompileFlags {
    pub opt_level: u8,
    pub debug_info: bool,
    pub debug_assertions: bool,
    pub overflow_checks: bool,
    pub features: Vec<String>,
    pub cfg: Vec<String>,
    pub include_paths: Vec<PathBuf>,
}

/// A link unit
#[derive(Debug, Clone)]
pub struct LinkUnit {
    /// Output path
    pub output: PathBuf,

    /// Input objects
    pub objects: Vec<PathBuf>,

    /// Libraries to link
    pub libs: Vec<String>,

    /// Library search paths
    pub lib_paths: Vec<PathBuf>,

    /// Linker flags
    pub flags: Vec<String>,
}

/// Build result
#[derive(Debug)]
pub struct BuildResult {
    /// Built artifacts
    pub artifacts: Vec<Artifact>,

    /// Compilation warnings
    pub warnings: Vec<String>,

    /// Build duration
    pub duration: std::time::Duration,
}

/// Build artifact
#[derive(Debug, Clone)]
pub struct Artifact {
    /// Artifact type
    pub kind: ArtifactKind,

    /// Output path
    pub path: PathBuf,

    /// Package name
    pub package: String,
}

/// Artifact kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    Binary,
    Library,
    DynamicLib,
    StaticLib,
    Test,
    Bench,
    Example,
}

/// Build error
#[derive(Debug)]
pub enum BuildError {
    /// Compilation error
    Compile {
        package: String,
        message: String,
        location: Option<(PathBuf, u32, u32)>,
    },

    /// Link error
    Link { message: String },

    /// Missing dependency
    MissingDependency(String),

    /// IO error
    Io(std::io::Error),

    /// Cycle in build graph
    Cycle(Vec<String>),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::Compile {
                package,
                message,
                location,
            } => {
                if let Some((path, line, col)) = location {
                    write!(
                        f,
                        "error[{}]: {}:{}:{}: {}",
                        package,
                        path.display(),
                        line,
                        col,
                        message
                    )
                } else {
                    write!(f, "error[{}]: {}", package, message)
                }
            }
            BuildError::Link { message } => write!(f, "link error: {}", message),
            BuildError::MissingDependency(d) => write!(f, "missing dependency: {}", d),
            BuildError::Io(e) => write!(f, "IO error: {}", e),
            BuildError::Cycle(path) => write!(f, "dependency cycle: {}", path.join(" -> ")),
        }
    }
}

impl std::error::Error for BuildError {}

/// Build executor
pub struct BuildExecutor {
    context: BuildContext,
    fingerprints: HashMap<String, Fingerprint>,
}

/// Build fingerprint for incremental compilation
#[derive(Debug, Clone)]
pub struct Fingerprint {
    /// Source file hashes
    pub source_hashes: HashMap<PathBuf, u64>,

    /// Dependency fingerprints
    pub dep_hashes: HashMap<String, u64>,

    /// Compiler version
    pub compiler_version: String,

    /// Compile flags hash
    pub flags_hash: u64,

    /// Last build time
    pub timestamp: SystemTime,
}

impl BuildExecutor {
    pub fn new(context: BuildContext) -> Self {
        Self {
            context,
            fingerprints: HashMap::new(),
        }
    }

    /// Plan build from manifest and resolution
    pub fn plan(
        &self,
        manifest: &Manifest,
        resolution: &Resolution,
    ) -> Result<BuildPlan, BuildError> {
        let mut units = Vec::new();
        let mut order = Vec::new();

        // Topological sort of dependencies
        self.topo_sort(manifest, resolution, &mut order, &mut HashSet::new())?;

        // Create compile units
        for pkg_name in &order {
            let pkg = resolution
                .packages
                .get(pkg_name)
                .ok_or_else(|| BuildError::MissingDependency(pkg_name.clone()))?;

            let sources = self.find_sources(pkg_name)?;
            let output = self.output_path(pkg_name);

            let deps: Vec<String> = pkg.dependencies.iter().map(|d| d.name.clone()).collect();

            let features: Vec<String> = pkg.features.iter().cloned().collect();

            units.push(CompileUnit {
                package: pkg_name.clone(),
                sources,
                output,
                deps,
                flags: CompileFlags {
                    opt_level: self.context.profile.opt_level(),
                    debug_info: self.context.profile.debug(),
                    debug_assertions: self.context.profile.debug(),
                    overflow_checks: self.context.profile.debug(),
                    features,
                    cfg: Vec::new(),
                    include_paths: Vec::new(),
                },
                mode: CompileMode::Build,
            });
        }

        // Create link units for binaries
        let links = self.plan_links(manifest, &units);

        Ok(BuildPlan { units, links })
    }

    /// Execute build plan
    pub fn execute(&mut self, plan: &BuildPlan) -> Result<BuildResult, BuildError> {
        let start = std::time::Instant::now();
        let mut artifacts = Vec::new();
        let mut warnings = Vec::new();

        // Compile units
        for unit in &plan.units {
            if self.needs_rebuild(unit) {
                self.compile(unit, &mut warnings)?;
            }

            artifacts.push(Artifact {
                kind: ArtifactKind::Library,
                path: unit.output.clone(),
                package: unit.package.clone(),
            });
        }

        // Link binaries
        for link in &plan.links {
            self.link(link)?;

            artifacts.push(Artifact {
                kind: ArtifactKind::Binary,
                path: link.output.clone(),
                package: String::new(),
            });
        }

        Ok(BuildResult {
            artifacts,
            warnings,
            duration: start.elapsed(),
        })
    }

    /// Check if unit needs rebuild
    fn needs_rebuild(&self, unit: &CompileUnit) -> bool {
        // Check fingerprint
        if let Some(fp) = self.fingerprints.get(&unit.package) {
            // Check if sources changed
            for source in &unit.sources {
                let hash = hash_file(source).unwrap_or(0);
                if fp.source_hashes.get(source) != Some(&hash) {
                    return true;
                }
            }

            // Check if output exists
            if !unit.output.exists() {
                return true;
            }

            return false;
        }

        true
    }

    /// Compile a unit
    fn compile(
        &mut self,
        unit: &CompileUnit,
        warnings: &mut Vec<String>,
    ) -> Result<(), BuildError> {
        if self.context.verbose {
            println!("   Compiling {} v0.1.0", unit.package);
        }

        // Create output directory
        if let Some(parent) = unit.output.parent() {
            std::fs::create_dir_all(parent).map_err(BuildError::Io)?;
        }

        // Compile each source file
        for source in &unit.sources {
            self.compile_file(source, unit, warnings)?;
        }

        // Update fingerprint
        let fp = self.compute_fingerprint(unit);
        self.fingerprints.insert(unit.package.clone(), fp);

        Ok(())
    }

    /// Compile single file
    fn compile_file(
        &self,
        source: &Path,
        _unit: &CompileUnit,
        _warnings: &mut Vec<String>,
    ) -> Result<(), BuildError> {
        // Read source
        let _content = std::fs::read_to_string(source).map_err(BuildError::Io)?;

        // Would compile using crate::compile() or similar
        // For now, this is a stub that creates an empty output

        Ok(())
    }

    /// Link a binary
    fn link(&self, link: &LinkUnit) -> Result<(), BuildError> {
        if self.context.verbose {
            println!("    Linking {}", link.output.display());
        }

        // Create output directory
        if let Some(parent) = link.output.parent() {
            std::fs::create_dir_all(parent).map_err(BuildError::Io)?;
        }

        // Would invoke linker
        // For now, create empty output file
        std::fs::write(&link.output, b"").map_err(BuildError::Io)?;

        Ok(())
    }

    /// Topological sort of packages
    fn topo_sort(
        &self,
        manifest: &Manifest,
        resolution: &Resolution,
        order: &mut Vec<String>,
        visiting: &mut HashSet<String>,
    ) -> Result<(), BuildError> {
        let root = manifest.package.name.clone();
        self.topo_visit(&root, resolution, order, visiting)
    }

    fn topo_visit(
        &self,
        name: &str,
        resolution: &Resolution,
        order: &mut Vec<String>,
        visiting: &mut HashSet<String>,
    ) -> Result<(), BuildError> {
        if order.contains(&name.to_string()) {
            return Ok(());
        }

        if visiting.contains(name) {
            return Err(BuildError::Cycle(visiting.iter().cloned().collect()));
        }

        visiting.insert(name.to_string());

        if let Some(pkg) = resolution.packages.get(name) {
            for dep in &pkg.dependencies {
                self.topo_visit(&dep.name, resolution, order, visiting)?;
            }
        }

        visiting.remove(name);
        order.push(name.to_string());

        Ok(())
    }

    /// Find source files for package
    fn find_sources(&self, _package: &str) -> Result<Vec<PathBuf>, BuildError> {
        // Scan src/ directory for .d files
        let src_dir = self.context.workspace_root.join("src");
        let mut sources = Vec::new();

        if src_dir.exists() {
            self.find_sources_recursive(&src_dir, &mut sources)?;
        }

        Ok(sources)
    }

    fn find_sources_recursive(
        &self,
        dir: &Path,
        sources: &mut Vec<PathBuf>,
    ) -> Result<(), BuildError> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in std::fs::read_dir(dir).map_err(BuildError::Io)? {
            let entry = entry.map_err(BuildError::Io)?;
            let path = entry.path();

            if path.is_dir() {
                self.find_sources_recursive(&path, sources)?;
            } else if path.extension().is_some_and(|e| e == "d") {
                sources.push(path);
            }
        }

        Ok(())
    }

    /// Get output path for package
    fn output_path(&self, package: &str) -> PathBuf {
        self.context
            .target_dir
            .join(self.context.profile.name())
            .join("deps")
            .join(format!("lib{}.rlib", package.replace('-', "_")))
    }

    /// Plan link steps
    fn plan_links(&self, manifest: &Manifest, units: &[CompileUnit]) -> Vec<LinkUnit> {
        let mut links = Vec::new();

        // Create link for each binary
        for bin in &manifest.binaries {
            let output = self
                .context
                .target_dir
                .join(self.context.profile.name())
                .join(&bin.name);

            let objects: Vec<PathBuf> = units.iter().map(|u| u.output.clone()).collect();

            links.push(LinkUnit {
                output,
                objects,
                libs: Vec::new(),
                lib_paths: Vec::new(),
                flags: Vec::new(),
            });
        }

        // If no binaries but has lib with main
        if links.is_empty() {
            let output = self
                .context
                .target_dir
                .join(self.context.profile.name())
                .join(&manifest.package.name);

            let objects: Vec<PathBuf> = units.iter().map(|u| u.output.clone()).collect();

            links.push(LinkUnit {
                output,
                objects,
                libs: Vec::new(),
                lib_paths: Vec::new(),
                flags: Vec::new(),
            });
        }

        links
    }

    /// Compute fingerprint for unit
    fn compute_fingerprint(&self, unit: &CompileUnit) -> Fingerprint {
        let mut source_hashes = HashMap::new();

        for source in &unit.sources {
            if let Ok(hash) = hash_file(source) {
                source_hashes.insert(source.clone(), hash);
            }
        }

        Fingerprint {
            source_hashes,
            dep_hashes: HashMap::new(),
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            flags_hash: 0, // Would hash compile flags
            timestamp: SystemTime::now(),
        }
    }
}

/// Hash a file
fn hash_file(path: &Path) -> Result<u64, std::io::Error> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let content = std::fs::read(path)?;
    let metadata = std::fs::metadata(path)?;

    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    metadata
        .modified()?
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .hash(&mut hasher);

    Ok(hasher.finish())
}

/// Get number of available CPUs
pub fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_profile() {
        assert_eq!(BuildProfile::Dev.name(), "dev");
        assert_eq!(BuildProfile::Release.name(), "release");
        assert_eq!(BuildProfile::Dev.opt_level(), 0);
        assert_eq!(BuildProfile::Release.opt_level(), 3);
        assert!(BuildProfile::Dev.debug());
        assert!(!BuildProfile::Release.debug());
    }
}
