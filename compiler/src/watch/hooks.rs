//! Build Hooks Manager
//!
//! Provides a flexible hook system for build lifecycle events:
//! - Pre/post build hooks
//! - Pre/post test hooks
//! - Watch mode hooks
//! - Custom hook points
//!
//! Hooks can run shell commands, D scripts, or built-in actions.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Hook-related errors
#[derive(Debug, Error)]
pub enum HookError {
    #[error("Hook '{name}' failed with exit code {code}")]
    HookFailed { name: String, code: i32 },

    #[error("Hook '{name}' timed out after {timeout:?}")]
    Timeout { name: String, timeout: Duration },

    #[error("Hook '{name}' not found")]
    NotFound { name: String },

    #[error("Invalid hook configuration: {message}")]
    InvalidConfig { message: String },

    #[error("IO error in hook '{name}': {source}")]
    Io {
        name: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Hook chain '{chain}' failed at hook '{hook}'")]
    ChainFailed { chain: String, hook: String },
}

/// Points in the build lifecycle where hooks can be triggered
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookPoint {
    /// Before compilation starts
    PreBuild,
    /// After successful compilation
    PostBuild,
    /// After compilation failure
    OnBuildError,
    /// Before tests run
    PreTest,
    /// After tests complete
    PostTest,
    /// After test failure
    OnTestError,
    /// When watch mode starts
    WatchStart,
    /// When watch mode stops
    WatchStop,
    /// When file changes detected
    OnFileChange,
    /// Before hot reload
    PreReload,
    /// After hot reload
    PostReload,
    /// Before formatting
    PreFormat,
    /// After formatting
    PostFormat,
    /// Before linting
    PreLint,
    /// After linting
    PostLint,
    /// On any error
    OnError,
    /// Custom named hook point
    Custom,
}

impl HookPoint {
    /// Get the string name for this hook point
    pub fn as_str(&self) -> &'static str {
        match self {
            HookPoint::PreBuild => "pre-build",
            HookPoint::PostBuild => "post-build",
            HookPoint::OnBuildError => "on-build-error",
            HookPoint::PreTest => "pre-test",
            HookPoint::PostTest => "post-test",
            HookPoint::OnTestError => "on-test-error",
            HookPoint::WatchStart => "watch-start",
            HookPoint::WatchStop => "watch-stop",
            HookPoint::OnFileChange => "on-file-change",
            HookPoint::PreReload => "pre-reload",
            HookPoint::PostReload => "post-reload",
            HookPoint::PreFormat => "pre-format",
            HookPoint::PostFormat => "post-format",
            HookPoint::PreLint => "pre-lint",
            HookPoint::PostLint => "post-lint",
            HookPoint::OnError => "on-error",
            HookPoint::Custom => "custom",
        }
    }

    /// Parse a hook point from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pre-build" => Some(HookPoint::PreBuild),
            "post-build" => Some(HookPoint::PostBuild),
            "on-build-error" => Some(HookPoint::OnBuildError),
            "pre-test" => Some(HookPoint::PreTest),
            "post-test" => Some(HookPoint::PostTest),
            "on-test-error" => Some(HookPoint::OnTestError),
            "watch-start" => Some(HookPoint::WatchStart),
            "watch-stop" => Some(HookPoint::WatchStop),
            "on-file-change" => Some(HookPoint::OnFileChange),
            "pre-reload" => Some(HookPoint::PreReload),
            "post-reload" => Some(HookPoint::PostReload),
            "pre-format" => Some(HookPoint::PreFormat),
            "post-format" => Some(HookPoint::PostFormat),
            "pre-lint" => Some(HookPoint::PreLint),
            "post-lint" => Some(HookPoint::PostLint),
            "on-error" => Some(HookPoint::OnError),
            _ => None,
        }
    }
}

/// Type of hook action
#[derive(Debug, Clone)]
pub enum HookAction {
    /// Run a shell command
    Command {
        /// The command to run
        command: String,
        /// Arguments to pass
        args: Vec<String>,
        /// Working directory (None = project root)
        cwd: Option<PathBuf>,
        /// Environment variables to set
        env: HashMap<String, String>,
    },
    /// Run a D script
    Script {
        /// Path to the script
        path: PathBuf,
        /// Arguments to pass
        args: Vec<String>,
    },
    /// Run a built-in action
    Builtin {
        /// Name of the built-in action
        name: String,
        /// Configuration for the action
        config: HashMap<String, String>,
    },
    /// Run multiple hooks in sequence
    Chain {
        /// Names of hooks to run
        hooks: Vec<String>,
    },
    /// Run multiple hooks in parallel
    Parallel {
        /// Names of hooks to run
        hooks: Vec<String>,
    },
}

/// Configuration for a single hook
#[derive(Debug, Clone)]
pub struct HookConfig {
    /// Unique name for this hook
    pub name: String,
    /// Hook point(s) where this hook triggers
    pub points: Vec<HookPoint>,
    /// The action to perform
    pub action: HookAction,
    /// Timeout for the hook
    pub timeout: Duration,
    /// Whether to continue on failure
    pub continue_on_failure: bool,
    /// Condition for running (environment variable check)
    pub condition: Option<HookCondition>,
    /// Priority (lower = runs first)
    pub priority: i32,
    /// Whether this hook is enabled
    pub enabled: bool,
}

impl Default for HookConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            points: Vec::new(),
            action: HookAction::Command {
                command: String::new(),
                args: Vec::new(),
                cwd: None,
                env: HashMap::new(),
            },
            timeout: Duration::from_secs(300),
            continue_on_failure: false,
            condition: None,
            priority: 0,
            enabled: true,
        }
    }
}

/// Condition for running a hook
#[derive(Debug, Clone)]
pub enum HookCondition {
    /// Run if environment variable is set
    EnvSet(String),
    /// Run if environment variable equals value
    EnvEquals(String, String),
    /// Run if file exists
    FileExists(PathBuf),
    /// Run if in CI environment
    InCi,
    /// Run if NOT in CI environment
    NotInCi,
    /// Logical AND of conditions
    And(Vec<HookCondition>),
    /// Logical OR of conditions
    Or(Vec<HookCondition>),
    /// Logical NOT of condition
    Not(Box<HookCondition>),
}

impl HookCondition {
    /// Evaluate the condition
    pub fn evaluate(&self) -> bool {
        match self {
            HookCondition::EnvSet(var) => std::env::var(var).is_ok(),
            HookCondition::EnvEquals(var, value) => {
                std::env::var(var).map(|v| v == *value).unwrap_or(false)
            }
            HookCondition::FileExists(path) => path.exists(),
            HookCondition::InCi => {
                std::env::var("CI").is_ok() || std::env::var("GITHUB_ACTIONS").is_ok()
            }
            HookCondition::NotInCi => !HookCondition::InCi.evaluate(),
            HookCondition::And(conditions) => conditions.iter().all(|c| c.evaluate()),
            HookCondition::Or(conditions) => conditions.iter().any(|c| c.evaluate()),
            HookCondition::Not(condition) => !condition.evaluate(),
        }
    }
}

/// Context passed to hooks when they run
#[derive(Debug, Clone)]
pub struct HookContext {
    /// Project root directory
    pub project_root: PathBuf,
    /// Current hook point
    pub hook_point: HookPoint,
    /// Files that changed (for OnFileChange)
    pub changed_files: Vec<PathBuf>,
    /// Build success/failure
    pub build_success: Option<bool>,
    /// Test success/failure
    pub test_success: Option<bool>,
    /// Error message (for error hooks)
    pub error_message: Option<String>,
    /// Additional variables
    pub variables: HashMap<String, String>,
}

impl HookContext {
    /// Create a new hook context
    pub fn new(project_root: PathBuf, hook_point: HookPoint) -> Self {
        Self {
            project_root,
            hook_point,
            changed_files: Vec::new(),
            build_success: None,
            test_success: None,
            error_message: None,
            variables: HashMap::new(),
        }
    }

    /// Get environment variables for this context
    pub fn to_env(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert(
            "DEMETRIOS_PROJECT_ROOT".to_string(),
            self.project_root.display().to_string(),
        );
        env.insert(
            "DEMETRIOS_HOOK_POINT".to_string(),
            self.hook_point.as_str().to_string(),
        );

        if !self.changed_files.is_empty() {
            env.insert(
                "DEMETRIOS_CHANGED_FILES".to_string(),
                self.changed_files
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(":"),
            );
        }

        if let Some(success) = self.build_success {
            env.insert("DEMETRIOS_BUILD_SUCCESS".to_string(), success.to_string());
        }

        if let Some(success) = self.test_success {
            env.insert("DEMETRIOS_TEST_SUCCESS".to_string(), success.to_string());
        }

        if let Some(ref msg) = self.error_message {
            env.insert("DEMETRIOS_ERROR_MESSAGE".to_string(), msg.clone());
        }

        for (key, value) in &self.variables {
            env.insert(format!("DEMETRIOS_{}", key.to_uppercase()), value.clone());
        }

        env
    }
}

/// Result of running a hook
#[derive(Debug)]
pub struct HookResult {
    /// Hook name
    pub name: String,
    /// Whether it succeeded
    pub success: bool,
    /// Exit code (if applicable)
    pub exit_code: Option<i32>,
    /// Duration of execution
    pub duration: Duration,
    /// Stdout output
    pub stdout: String,
    /// Stderr output
    pub stderr: String,
}

/// Manager for build hooks
pub struct HookManager {
    /// All registered hooks
    hooks: HashMap<String, HookConfig>,
    /// Hooks indexed by hook point
    hooks_by_point: HashMap<HookPoint, Vec<String>>,
    /// Custom hook points
    custom_points: HashMap<String, HookPoint>,
    /// Project root
    project_root: PathBuf,
    /// Whether to print hook output
    verbose: bool,
}

impl HookManager {
    /// Create a new hook manager
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            hooks: HashMap::new(),
            hooks_by_point: HashMap::new(),
            custom_points: HashMap::new(),
            project_root,
            verbose: false,
        }
    }

    /// Set verbose mode
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Register a hook
    pub fn register(&mut self, config: HookConfig) -> Result<(), HookError> {
        if config.name.is_empty() {
            return Err(HookError::InvalidConfig {
                message: "Hook name cannot be empty".to_string(),
            });
        }

        // Index by hook points
        for point in &config.points {
            self.hooks_by_point
                .entry(*point)
                .or_default()
                .push(config.name.clone());
        }

        self.hooks.insert(config.name.clone(), config);
        Ok(())
    }

    /// Unregister a hook
    pub fn unregister(&mut self, name: &str) -> Option<HookConfig> {
        if let Some(config) = self.hooks.remove(name) {
            // Remove from point index
            for point in &config.points {
                if let Some(hooks) = self.hooks_by_point.get_mut(point) {
                    hooks.retain(|n| n != name);
                }
            }
            Some(config)
        } else {
            None
        }
    }

    /// Register a custom hook point
    pub fn register_custom_point(&mut self, name: &str) {
        self.custom_points
            .insert(name.to_string(), HookPoint::Custom);
    }

    /// Get all hooks for a hook point
    pub fn get_hooks(&self, point: HookPoint) -> Vec<&HookConfig> {
        let mut hooks: Vec<_> = self
            .hooks_by_point
            .get(&point)
            .map(|names| {
                names
                    .iter()
                    .filter_map(|name| self.hooks.get(name))
                    .filter(|h| h.enabled)
                    .collect()
            })
            .unwrap_or_default();

        // Sort by priority
        hooks.sort_by_key(|h| h.priority);
        hooks
    }

    /// Run all hooks for a hook point
    pub fn run(&self, point: HookPoint, context: &HookContext) -> Vec<HookResult> {
        let hooks = self.get_hooks(point);
        let mut results = Vec::new();

        for hook in hooks {
            // Check condition
            if let Some(ref condition) = hook.condition
                && !condition.evaluate()
            {
                continue;
            }

            let result = self.run_hook(hook, context);
            let failed = !result.success;

            results.push(result);

            if failed && !hook.continue_on_failure {
                break;
            }
        }

        results
    }

    /// Run a single hook
    pub fn run_hook(&self, hook: &HookConfig, context: &HookContext) -> HookResult {
        let start = Instant::now();

        match &hook.action {
            HookAction::Command {
                command,
                args,
                cwd,
                env,
            } => self.run_command(
                &hook.name,
                command,
                args,
                cwd.as_ref(),
                env,
                context,
                hook.timeout,
            ),
            HookAction::Script { path, args } => {
                self.run_script(&hook.name, path, args, context, hook.timeout)
            }
            HookAction::Builtin { name, config } => {
                self.run_builtin(&hook.name, name, config, context)
            }
            HookAction::Chain { hooks: chain_hooks } => {
                self.run_chain(&hook.name, chain_hooks, context, start)
            }
            HookAction::Parallel {
                hooks: parallel_hooks,
            } => self.run_parallel(&hook.name, parallel_hooks, context, start),
        }
    }

    /// Run a command hook
    fn run_command(
        &self,
        name: &str,
        command: &str,
        args: &[String],
        cwd: Option<&PathBuf>,
        env: &HashMap<String, String>,
        context: &HookContext,
        timeout: Duration,
    ) -> HookResult {
        let start = Instant::now();

        let working_dir = cwd.cloned().unwrap_or_else(|| self.project_root.clone());

        // Build environment
        let mut full_env = context.to_env();
        full_env.extend(env.clone());

        let result = Command::new(command)
            .args(args)
            .current_dir(&working_dir)
            .envs(&full_env)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        match result {
            Ok(output) => {
                let duration = start.elapsed();
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                if self.verbose {
                    if !stdout.is_empty() {
                        println!("[{}] stdout:\n{}", name, stdout);
                    }
                    if !stderr.is_empty() {
                        eprintln!("[{}] stderr:\n{}", name, stderr);
                    }
                }

                HookResult {
                    name: name.to_string(),
                    success: output.status.success(),
                    exit_code: output.status.code(),
                    duration,
                    stdout,
                    stderr,
                }
            }
            Err(e) => HookResult {
                name: name.to_string(),
                success: false,
                exit_code: None,
                duration: start.elapsed(),
                stdout: String::new(),
                stderr: e.to_string(),
            },
        }
    }

    /// Run a D script hook
    fn run_script(
        &self,
        name: &str,
        path: &Path,
        args: &[String],
        context: &HookContext,
        timeout: Duration,
    ) -> HookResult {
        // For now, run scripts via the dc interpreter
        let mut full_args = vec![path.display().to_string()];
        full_args.extend(args.iter().cloned());

        self.run_command(
            name,
            "dc",
            &full_args,
            Some(&self.project_root),
            &HashMap::new(),
            context,
            timeout,
        )
    }

    /// Run a built-in hook action
    fn run_builtin(
        &self,
        name: &str,
        builtin_name: &str,
        config: &HashMap<String, String>,
        context: &HookContext,
    ) -> HookResult {
        let start = Instant::now();

        let (success, output) = match builtin_name {
            "echo" => {
                let message = config.get("message").cloned().unwrap_or_default();
                println!("{}", message);
                (true, message)
            }
            "env" => {
                // Print environment variables
                let vars: Vec<_> = context
                    .to_env()
                    .into_iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect();
                let output = vars.join("\n");
                println!("{}", output);
                (true, output)
            }
            "touch" => {
                // Touch a file
                if let Some(path) = config.get("path") {
                    let full_path = self.project_root.join(path);
                    match std::fs::OpenOptions::new()
                        .create(true)
                        .write(true)
                        .truncate(false)
                        .open(&full_path)
                    {
                        Ok(_) => (true, format!("Touched {}", full_path.display())),
                        Err(e) => (false, e.to_string()),
                    }
                } else {
                    (false, "Missing 'path' config".to_string())
                }
            }
            "rm" => {
                // Remove a file
                if let Some(path) = config.get("path") {
                    let full_path = self.project_root.join(path);
                    match std::fs::remove_file(&full_path) {
                        Ok(_) => (true, format!("Removed {}", full_path.display())),
                        Err(e) => (false, e.to_string()),
                    }
                } else {
                    (false, "Missing 'path' config".to_string())
                }
            }
            "mkdir" => {
                // Create directory
                if let Some(path) = config.get("path") {
                    let full_path = self.project_root.join(path);
                    match std::fs::create_dir_all(&full_path) {
                        Ok(_) => (true, format!("Created {}", full_path.display())),
                        Err(e) => (false, e.to_string()),
                    }
                } else {
                    (false, "Missing 'path' config".to_string())
                }
            }
            "copy" => {
                // Copy a file
                if let (Some(from), Some(to)) = (config.get("from"), config.get("to")) {
                    let from_path = self.project_root.join(from);
                    let to_path = self.project_root.join(to);
                    match std::fs::copy(&from_path, &to_path) {
                        Ok(_) => (
                            true,
                            format!("Copied {} to {}", from_path.display(), to_path.display()),
                        ),
                        Err(e) => (false, e.to_string()),
                    }
                } else {
                    (false, "Missing 'from' or 'to' config".to_string())
                }
            }
            "sleep" => {
                // Sleep for a duration
                if let Some(ms) = config.get("ms").and_then(|s| s.parse::<u64>().ok()) {
                    std::thread::sleep(Duration::from_millis(ms));
                    (true, format!("Slept for {}ms", ms))
                } else {
                    (false, "Missing or invalid 'ms' config".to_string())
                }
            }
            _ => (false, format!("Unknown builtin: {}", builtin_name)),
        };

        HookResult {
            name: name.to_string(),
            success,
            exit_code: if success { Some(0) } else { Some(1) },
            duration: start.elapsed(),
            stdout: output,
            stderr: String::new(),
        }
    }

    /// Run a chain of hooks sequentially
    fn run_chain(
        &self,
        name: &str,
        hook_names: &[String],
        context: &HookContext,
        start: Instant,
    ) -> HookResult {
        let mut all_stdout = Vec::new();
        let mut all_stderr = Vec::new();

        for hook_name in hook_names {
            if let Some(hook) = self.hooks.get(hook_name) {
                let result = self.run_hook(hook, context);
                all_stdout.push(format!("[{}]\n{}", hook_name, result.stdout));
                all_stderr.push(format!("[{}]\n{}", hook_name, result.stderr));

                if !result.success {
                    return HookResult {
                        name: name.to_string(),
                        success: false,
                        exit_code: result.exit_code,
                        duration: start.elapsed(),
                        stdout: all_stdout.join("\n"),
                        stderr: all_stderr.join("\n"),
                    };
                }
            }
        }

        HookResult {
            name: name.to_string(),
            success: true,
            exit_code: Some(0),
            duration: start.elapsed(),
            stdout: all_stdout.join("\n"),
            stderr: all_stderr.join("\n"),
        }
    }

    /// Run hooks in parallel
    fn run_parallel(
        &self,
        name: &str,
        hook_names: &[String],
        context: &HookContext,
        start: Instant,
    ) -> HookResult {
        use std::sync::mpsc;
        use std::thread;

        let (tx, rx) = mpsc::channel();

        for hook_name in hook_names {
            if let Some(hook) = self.hooks.get(hook_name).cloned() {
                let tx = tx.clone();
                let ctx = context.clone();
                let manager_root = self.project_root.clone();
                let verbose = self.verbose;

                thread::spawn(move || {
                    // Create a minimal manager for this thread
                    let mut thread_manager = HookManager::new(manager_root);
                    thread_manager.set_verbose(verbose);
                    let result = thread_manager.run_hook(&hook, &ctx);
                    let _ = tx.send(result);
                });
            }
        }

        drop(tx);

        let mut all_stdout = Vec::new();
        let mut all_stderr = Vec::new();
        let mut all_success = true;
        let mut last_exit_code = Some(0);

        for result in rx {
            all_stdout.push(format!("[{}]\n{}", result.name, result.stdout));
            all_stderr.push(format!("[{}]\n{}", result.name, result.stderr));
            if !result.success {
                all_success = false;
                last_exit_code = result.exit_code;
            }
        }

        HookResult {
            name: name.to_string(),
            success: all_success,
            exit_code: last_exit_code,
            duration: start.elapsed(),
            stdout: all_stdout.join("\n"),
            stderr: all_stderr.join("\n"),
        }
    }

    /// Load hooks from a d.toml configuration
    pub fn load_from_toml(&mut self, toml_path: &Path) -> Result<(), HookError> {
        let content = std::fs::read_to_string(toml_path).map_err(|e| HookError::Io {
            name: "config".to_string(),
            source: e,
        })?;

        // Simple TOML parsing for hooks section
        // Format:
        // [[hooks]]
        // name = "my-hook"
        // points = ["pre-build"]
        // command = "echo"
        // args = ["hello"]

        let mut current_hook: Option<HookConfig> = None;

        for line in content.lines() {
            let line = line.trim();

            if line == "[[hooks]]" {
                // Save previous hook if any
                if let Some(hook) = current_hook.take() {
                    self.register(hook)?;
                }
                current_hook = Some(HookConfig::default());
            } else if let Some(ref mut hook) = current_hook
                && let Some((key, value)) = line.split_once('=')
            {
                let key = key.trim();
                let value = value.trim().trim_matches('"');

                match key {
                    "name" => hook.name = value.to_string(),
                    "command" => {
                        if let HookAction::Command {
                            ref mut command, ..
                        } = hook.action
                        {
                            *command = value.to_string();
                        }
                    }
                    "timeout" => {
                        if let Ok(secs) = value.parse::<u64>() {
                            hook.timeout = Duration::from_secs(secs);
                        }
                    }
                    "priority" => {
                        if let Ok(p) = value.parse::<i32>() {
                            hook.priority = p;
                        }
                    }
                    "enabled" => {
                        hook.enabled = value == "true";
                    }
                    "continue_on_failure" => {
                        hook.continue_on_failure = value == "true";
                    }
                    "points" => {
                        // Parse array: ["pre-build", "post-build"]
                        let points_str = value.trim_matches(|c| c == '[' || c == ']');
                        for point_str in points_str.split(',') {
                            let point_str = point_str.trim().trim_matches('"');
                            if let Some(point) = HookPoint::from_str(point_str) {
                                hook.points.push(point);
                            }
                        }
                    }
                    "args" => {
                        // Parse array: ["arg1", "arg2"]
                        if let HookAction::Command { ref mut args, .. } = hook.action {
                            let args_str = value.trim_matches(|c| c == '[' || c == ']');
                            for arg in args_str.split(',') {
                                let arg = arg.trim().trim_matches('"');
                                if !arg.is_empty() {
                                    args.push(arg.to_string());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Don't forget the last hook
        if let Some(hook) = current_hook {
            self.register(hook)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hook_point_roundtrip() {
        let points = [
            HookPoint::PreBuild,
            HookPoint::PostBuild,
            HookPoint::PreTest,
            HookPoint::PostTest,
            HookPoint::WatchStart,
            HookPoint::OnError,
        ];

        for point in points {
            let s = point.as_str();
            let parsed = HookPoint::from_str(s);
            assert!(parsed.is_some());
        }
    }

    #[test]
    fn test_hook_condition_env() {
        // SAFETY: This test is single-threaded and we clean up after ourselves
        unsafe {
            std::env::set_var("TEST_HOOK_VAR", "test_value");
        }

        let cond = HookCondition::EnvSet("TEST_HOOK_VAR".to_string());
        assert!(cond.evaluate());

        let cond = HookCondition::EnvEquals("TEST_HOOK_VAR".to_string(), "test_value".to_string());
        assert!(cond.evaluate());

        let cond = HookCondition::EnvEquals("TEST_HOOK_VAR".to_string(), "wrong".to_string());
        assert!(!cond.evaluate());

        // SAFETY: Cleaning up the variable we set
        unsafe {
            std::env::remove_var("TEST_HOOK_VAR");
        }
    }

    #[test]
    fn test_hook_condition_logic() {
        let always_true = HookCondition::EnvSet("PATH".to_string());
        let always_false = HookCondition::EnvSet("NONEXISTENT_VAR_12345".to_string());

        let and_cond = HookCondition::And(vec![always_true.clone(), always_false.clone()]);
        assert!(!and_cond.evaluate());

        let or_cond = HookCondition::Or(vec![always_true.clone(), always_false.clone()]);
        assert!(or_cond.evaluate());

        let not_cond = HookCondition::Not(Box::new(always_false));
        assert!(not_cond.evaluate());
    }

    #[test]
    fn test_hook_context_env() {
        let mut context = HookContext::new(PathBuf::from("/test"), HookPoint::PreBuild);
        context.build_success = Some(true);
        context
            .variables
            .insert("custom".to_string(), "value".to_string());

        let env = context.to_env();
        assert_eq!(
            env.get("DEMETRIOS_PROJECT_ROOT"),
            Some(&"/test".to_string())
        );
        assert_eq!(
            env.get("DEMETRIOS_HOOK_POINT"),
            Some(&"pre-build".to_string())
        );
        assert_eq!(
            env.get("DEMETRIOS_BUILD_SUCCESS"),
            Some(&"true".to_string())
        );
        assert_eq!(env.get("DEMETRIOS_CUSTOM"), Some(&"value".to_string()));
    }

    #[test]
    fn test_hook_manager_register() {
        let mut manager = HookManager::new(PathBuf::from("/test"));

        let hook = HookConfig {
            name: "test-hook".to_string(),
            points: vec![HookPoint::PreBuild, HookPoint::PostBuild],
            action: HookAction::Builtin {
                name: "echo".to_string(),
                config: HashMap::from([("message".to_string(), "hello".to_string())]),
            },
            ..Default::default()
        };

        manager.register(hook).unwrap();

        assert_eq!(manager.get_hooks(HookPoint::PreBuild).len(), 1);
        assert_eq!(manager.get_hooks(HookPoint::PostBuild).len(), 1);
        assert_eq!(manager.get_hooks(HookPoint::PreTest).len(), 0);
    }

    #[test]
    fn test_builtin_echo() {
        let manager = HookManager::new(PathBuf::from("/test"));
        let context = HookContext::new(PathBuf::from("/test"), HookPoint::PreBuild);

        let result = manager.run_builtin(
            "test",
            "echo",
            &HashMap::from([("message".to_string(), "Hello, World!".to_string())]),
            &context,
        );

        assert!(result.success);
        assert_eq!(result.stdout, "Hello, World!");
    }

    #[test]
    fn test_hook_priority() {
        let mut manager = HookManager::new(PathBuf::from("/test"));

        let hook1 = HookConfig {
            name: "low-priority".to_string(),
            points: vec![HookPoint::PreBuild],
            priority: 10,
            action: HookAction::Builtin {
                name: "echo".to_string(),
                config: HashMap::new(),
            },
            ..Default::default()
        };

        let hook2 = HookConfig {
            name: "high-priority".to_string(),
            points: vec![HookPoint::PreBuild],
            priority: 1,
            action: HookAction::Builtin {
                name: "echo".to_string(),
                config: HashMap::new(),
            },
            ..Default::default()
        };

        manager.register(hook1).unwrap();
        manager.register(hook2).unwrap();

        let hooks = manager.get_hooks(HookPoint::PreBuild);
        assert_eq!(hooks.len(), 2);
        assert_eq!(hooks[0].name, "high-priority");
        assert_eq!(hooks[1].name, "low-priority");
    }
}
