//! Watch Mode, Hot Reload & Build Hooks
//!
//! This module provides:
//! - File system watching with debouncing
//! - Continuous build/test mode
//! - Hot reload for development
//! - Build script support (build.d)
//! - Build hooks for lifecycle events
//! - Development server with live reload
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Watch Mode                           │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │
//! │  │  Watcher │──>│ Debouncer│──>│ Watch Controller │   │
//! │  └──────────┘   └──────────┘   └────────┬─────────┘   │
//! │                                          │             │
//! │  ┌──────────────────────────────────────┴───────────┐ │
//! │  │              Build Pipeline                       │ │
//! │  │  ┌─────────┐   ┌─────────┐   ┌─────────────────┐ │ │
//! │  │  │ Pre-Hook│──>│  Build  │──>│ Post-Hook       │ │ │
//! │  │  └─────────┘   └─────────┘   └────────┬────────┘ │ │
//! │  └───────────────────────────────────────┼──────────┘ │
//! │                                          │             │
//! │  ┌───────────────────────────────────────┴───────────┐ │
//! │  │              Hot Reload / Dev Server              │ │
//! │  │  ┌──────────────┐   ┌────────────────────────┐   │ │
//! │  │  │ Hot Reload   │   │ Development Server     │   │ │
//! │  │  │ Engine       │   │ (Static + Live Reload) │   │ │
//! │  │  └──────────────┘   └────────────────────────┘   │ │
//! │  └───────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example Usage
//!
//! ```ignore
//! use sounio::watch::{WatchMode, WatchModeConfig};
//!
//! let config = WatchModeConfig {
//!     paths: vec!["src".into()],
//!     clear_screen: true,
//!     run_tests: true,
//!     ..Default::default()
//! };
//!
//! let mut watch = WatchMode::new(config)?;
//! watch.run()?;
//! ```

mod buildscript;
mod devserver;
mod hooks;
mod hotreload;
mod mode;
mod watcher;

pub use buildscript::{
    BuildInstruction, BuildScriptConfig, BuildScriptError, BuildScriptOutput, BuildScriptRunner,
    LinkKind,
};
pub use devserver::{
    CorsConfig, DevServer, DevServerConfig, DevServerError, DevServerStats, ProxyRule,
};
pub use hooks::{
    HookAction, HookCondition, HookConfig, HookContext, HookError, HookManager, HookPoint,
    HookResult,
};
pub use hotreload::{
    FunctionUpdate, HotReloadConfig, HotReloadEngine, HotReloadError, HotReloadRuntime,
    ReloadMessage, Relocation,
};
pub use mode::{BuildResult, WatchMode, WatchModeConfig, WatchModeError, WatchState, WatchStats};
pub use watcher::{FsEvent, FsEventKind, WatchConfig, WatchError, Watcher};

/// Convenience function to start watch mode with default configuration
pub fn watch(paths: &[std::path::PathBuf]) -> Result<(), WatchError> {
    let watch_config = WatchConfig {
        paths: paths.to_vec(),
        ..Default::default()
    };

    let config = WatchModeConfig {
        watch: watch_config,
        ..Default::default()
    };

    let mut watch_mode = WatchMode::new(config)?;
    watch_mode
        .run()
        .map_err(|e| WatchError::Io(std::io::Error::other(e.to_string())))
}

/// Convenience function to start development server
pub fn serve(root: std::path::PathBuf, port: u16) -> Result<(), DevServerError> {
    let config = DevServerConfig {
        root,
        port,
        ..Default::default()
    };

    let mut server = DevServer::new(config);
    server.start()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_module_exports() {
        // Verify all types are accessible
        let _: WatchConfig = WatchConfig::default();
        let _: WatchModeConfig = WatchModeConfig::default();
        let _: DevServerConfig = DevServerConfig::default();
        let _: BuildScriptConfig = BuildScriptConfig::default();
    }

    #[test]
    fn test_hook_point_values() {
        assert_eq!(HookPoint::PreBuild.as_str(), "pre-build");
        assert_eq!(HookPoint::PostBuild.as_str(), "post-build");
        assert_eq!(HookPoint::WatchStart.as_str(), "watch-start");
    }
}
