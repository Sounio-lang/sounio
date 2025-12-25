//! Target Specification and Cross-Compilation Support
//!
//! This module provides comprehensive support for cross-compilation including:
//!
//! - **Target Specification** (`spec`): Target triple parsing, architecture/OS/ABI configuration
//! - **Sysroot Management** (`sysroot`): Discovery, building, and caching of target sysroots
//! - **Conditional Compilation** (`cfg`): Platform-specific code inclusion via cfg predicates
//! - **Linker Integration** (`linker`): Platform-specific linker invocation and script generation
//! - **Target Registry** (`registry`): Built-in target definitions for common platforms
//!
//! # Example
//!
//! ```ignore
//! use sounio::target::{TargetSpec, TargetRegistry, CfgContext, Linker};
//!
//! // Get a target from the registry
//! let registry = TargetRegistry::with_builtins();
//! let target = registry.get("x86_64-unknown-linux-gnu")?;
//!
//! // Create a cfg context for the target
//! let cfg = CfgContext::from_target(&target);
//! assert!(cfg.is_set("linux"));
//! assert!(cfg.is_set("unix"));
//!
//! // Create a linker for the target
//! let linker = Linker::new(&target)?;
//! ```

pub mod cfg;
pub mod linker;
pub mod registry;
pub mod spec;
pub mod sysroot;

// Re-export main types for convenience
pub use cfg::{CfgContext, CfgDirective, CfgError, CfgPredicate, CfgResult};
pub use linker::{
    Linker, LinkerError, LinkerInput, LinkerInputKind, LinkerResult, LinkerScriptBuilder,
    LinkerSection, MemoryRegion, OutputType,
};
pub use registry::{TargetRegistry, global_registry, targets};
pub use spec::{
    ArchSpec, Architecture, CRuntime, CodeModel, CodegenSpec, DebugInfo, EnvSpec, Environment,
    FramePointer, LinkerFlavor, LinkerSpec, LtoMode, OperatingSystem, OptLevel, OsSpec,
    PanicStrategy, RelocationModel, StripMode, TargetFeatures, TargetOptions, TargetOverrides,
    TargetResult, TargetSpec, TargetSpecError, TargetTriple, TlsModel, Visibility, host_triple,
};
pub use sysroot::{
    Sysroot, SysrootBuilder, SysrootComponent, SysrootConfig, SysrootError, SysrootManager,
    SysrootMetadata, SysrootResult, SysrootSource,
};
