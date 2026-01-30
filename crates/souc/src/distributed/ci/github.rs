//! GitHub Actions integration
//!
//! This module generates GitHub Actions workflow files for CI/CD.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// GitHub Actions workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Workflow name
    pub name: String,

    /// Trigger events
    pub on: WorkflowTrigger,

    /// Environment variables
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,

    /// Jobs
    pub jobs: HashMap<String, Job>,
}

/// Workflow trigger
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WorkflowTrigger {
    Simple(Vec<String>),
    Detailed(HashMap<String, TriggerConfig>),
}

/// Trigger configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TriggerConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branches: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub paths: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "paths-ignore")]
    pub paths_ignore: Option<Vec<String>>,
}

/// Job definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// Job name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Runs on
    #[serde(rename = "runs-on")]
    pub runs_on: RunsOn,

    /// Strategy (matrix)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<Strategy>,

    /// Environment variables
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,

    /// Steps
    pub steps: Vec<Step>,

    /// Needs (dependencies)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub needs: Option<Vec<String>>,

    /// Condition
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "if")]
    pub condition: Option<String>,

    /// Timeout
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "timeout-minutes")]
    pub timeout_minutes: Option<u32>,
}

/// Runs-on specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RunsOn {
    Single(String),
    Matrix(String), // e.g., "${{ matrix.os }}"
}

/// Build matrix strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    /// Matrix values
    pub matrix: HashMap<String, Vec<serde_json::Value>>,

    /// Fail fast
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "fail-fast")]
    pub fail_fast: Option<bool>,

    /// Max parallel
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "max-parallel")]
    pub max_parallel: Option<u32>,
}

/// Workflow step
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Step {
    /// Step name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Uses (action)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uses: Option<String>,

    /// Run (shell command)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,

    /// With (action inputs)
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub with: HashMap<String, String>,

    /// Environment variables
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,

    /// Condition
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "if")]
    pub condition: Option<String>,

    /// Working directory
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "working-directory")]
    pub working_directory: Option<String>,

    /// ID (for outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// Workflow generator
pub struct WorkflowGenerator {
    project_name: String,
    targets: Vec<String>,
    features: Vec<String>,
    os_matrix: Vec<String>,
}

impl WorkflowGenerator {
    /// Create new workflow generator
    pub fn new(project_name: &str) -> Self {
        WorkflowGenerator {
            project_name: project_name.to_string(),
            targets: vec!["x86_64-unknown-linux-gnu".into()],
            features: vec![],
            os_matrix: vec![
                "ubuntu-latest".into(),
                "macos-latest".into(),
                "windows-latest".into(),
            ],
        }
    }

    /// Set targets
    pub fn targets(mut self, targets: Vec<String>) -> Self {
        self.targets = targets;
        self
    }

    /// Set features
    pub fn features(mut self, features: Vec<String>) -> Self {
        self.features = features;
        self
    }

    /// Set OS matrix
    pub fn os_matrix(mut self, os_matrix: Vec<String>) -> Self {
        self.os_matrix = os_matrix;
        self
    }

    /// Generate CI workflow
    pub fn generate_ci(&self) -> Workflow {
        let mut jobs = HashMap::new();

        // Build job
        jobs.insert(
            "build".into(),
            Job {
                name: Some("Build".into()),
                runs_on: RunsOn::Matrix("${{ matrix.os }}".into()),
                strategy: Some(Strategy {
                    matrix: [
                        (
                            "os".into(),
                            self.os_matrix
                                .iter()
                                .map(|s| serde_json::Value::String(s.clone()))
                                .collect(),
                        ),
                        (
                            "target".into(),
                            self.targets
                                .iter()
                                .map(|t| serde_json::Value::String(t.clone()))
                                .collect(),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    fail_fast: Some(false),
                    max_parallel: None,
                }),
                env: HashMap::new(),
                steps: vec![
                    Step {
                        name: Some("Checkout".into()),
                        uses: Some("actions/checkout@v4".into()),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Install D compiler".into()),
                        uses: Some("d-lang/setup-d@v1".into()),
                        with: [("version".into(), "latest".into())].into_iter().collect(),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Cache".into()),
                        uses: Some("actions/cache@v4".into()),
                        with: [
                            ("path".into(), "~/.d/cache\ntarget".into()),
                            (
                                "key".into(),
                                "${{ runner.os }}-${{ matrix.target }}-${{ hashFiles('**/d.lock') }}"
                                    .into(),
                            ),
                            (
                                "restore-keys".into(),
                                "${{ runner.os }}-${{ matrix.target }}-".into(),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Build".into()),
                        run: Some("souc build --target ${{ matrix.target }} --release".into()),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Test".into()),
                        run: Some("souc test".into()),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Upload artifacts".into()),
                        uses: Some("actions/upload-artifact@v4".into()),
                        with: [
                            (
                                "name".into(),
                                format!("{}-${{{{ matrix.target }}}}", self.project_name),
                            ),
                            ("path".into(), "target/${{ matrix.target }}/release/*".into()),
                        ]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                ],
                needs: None,
                condition: None,
                timeout_minutes: Some(30),
            },
        );

        // Test job
        jobs.insert(
            "test".into(),
            Job {
                name: Some("Test".into()),
                runs_on: RunsOn::Single("ubuntu-latest".into()),
                strategy: None,
                env: HashMap::new(),
                steps: vec![
                    Step {
                        name: Some("Checkout".into()),
                        uses: Some("actions/checkout@v4".into()),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Install D compiler".into()),
                        uses: Some("d-lang/setup-d@v1".into()),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Run tests".into()),
                        run: Some("souc test --all".into()),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Run linter".into()),
                        run: Some("souc lint --deny warnings".into()),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Check formatting".into()),
                        run: Some("souc fmt --check".into()),
                        ..Default::default()
                    },
                ],
                needs: None,
                condition: None,
                timeout_minutes: Some(15),
            },
        );

        Workflow {
            name: format!("{} CI", self.project_name),
            on: WorkflowTrigger::Detailed(
                [
                    (
                        "push".into(),
                        TriggerConfig {
                            branches: Some(vec!["main".into(), "develop".into()]),
                            tags: None,
                            paths: None,
                            paths_ignore: Some(vec!["**.md".into(), "docs/**".into()]),
                        },
                    ),
                    (
                        "pull_request".into(),
                        TriggerConfig {
                            branches: Some(vec!["main".into()]),
                            tags: None,
                            paths: None,
                            paths_ignore: None,
                        },
                    ),
                ]
                .into_iter()
                .collect(),
            ),
            env: [("D_BACKTRACE".into(), "1".into())].into_iter().collect(),
            jobs,
        }
    }

    /// Generate release workflow
    pub fn generate_release(&self) -> Workflow {
        let mut jobs = HashMap::new();

        // Build release job
        jobs.insert(
            "build".into(),
            Job {
                name: Some("Build Release".into()),
                runs_on: RunsOn::Matrix("${{ matrix.os }}".into()),
                strategy: Some(Strategy {
                    matrix: [
                        (
                            "os".into(),
                            self.os_matrix
                                .iter()
                                .map(|s| serde_json::Value::String(s.clone()))
                                .collect(),
                        ),
                        (
                            "target".into(),
                            self.targets
                                .iter()
                                .map(|t| serde_json::Value::String(t.clone()))
                                .collect(),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    fail_fast: Some(false),
                    max_parallel: None,
                }),
                env: HashMap::new(),
                steps: vec![
                    Step {
                        name: Some("Checkout".into()),
                        uses: Some("actions/checkout@v4".into()),
                        with: [("fetch-depth".into(), "0".into())].into_iter().collect(),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Install D compiler".into()),
                        uses: Some("d-lang/setup-d@v1".into()),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Build release".into()),
                        run: Some("souc build --target ${{ matrix.target }} --release".into()),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Generate provenance".into()),
                        run: Some("souc ci provenance --output provenance-${{ matrix.target }}.json".into()),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Upload artifacts".into()),
                        uses: Some("actions/upload-artifact@v4".into()),
                        with: [
                            (
                                "name".into(),
                                format!("{}-${{{{ matrix.target }}}}", self.project_name),
                            ),
                            (
                                "path".into(),
                                "target/${{ matrix.target }}/release/*\nprovenance-${{ matrix.target }}.json".into(),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                ],
                needs: None,
                condition: None,
                timeout_minutes: Some(30),
            },
        );

        // Release job
        jobs.insert(
            "release".into(),
            Job {
                name: Some("Create Release".into()),
                runs_on: RunsOn::Single("ubuntu-latest".into()),
                strategy: None,
                env: HashMap::new(),
                steps: vec![
                    Step {
                        name: Some("Checkout".into()),
                        uses: Some("actions/checkout@v4".into()),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Download artifacts".into()),
                        uses: Some("actions/download-artifact@v4".into()),
                        with: [("path".into(), "artifacts".into())].into_iter().collect(),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Create release".into()),
                        uses: Some("softprops/action-gh-release@v1".into()),
                        with: [
                            ("files".into(), "artifacts/**/*".into()),
                            ("generate_release_notes".into(), "true".into()),
                        ]
                        .into_iter()
                        .collect(),
                        env: [("GITHUB_TOKEN".into(), "${{ secrets.GITHUB_TOKEN }}".into())]
                            .into_iter()
                            .collect(),
                        ..Default::default()
                    },
                    Step {
                        name: Some("Publish to registry".into()),
                        run: Some("souc publish".into()),
                        env: [("D_TOKEN".into(), "${{ secrets.D_TOKEN }}".into())]
                            .into_iter()
                            .collect(),
                        condition: Some(
                            "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')"
                                .into(),
                        ),
                        ..Default::default()
                    },
                ],
                needs: Some(vec!["build".into()]),
                condition: None,
                timeout_minutes: Some(15),
            },
        );

        Workflow {
            name: format!("{} Release", self.project_name),
            on: WorkflowTrigger::Detailed(
                [(
                    "push".into(),
                    TriggerConfig {
                        branches: None,
                        tags: Some(vec!["v*".into()]),
                        paths: None,
                        paths_ignore: None,
                    },
                )]
                .into_iter()
                .collect(),
            ),
            env: HashMap::new(),
            jobs,
        }
    }

    /// Generate to YAML string
    pub fn to_yaml(&self, workflow: &Workflow) -> String {
        serde_yaml::to_string(workflow).unwrap()
    }

    /// Write workflow to file
    pub fn write_workflow(
        &self,
        workflow: &Workflow,
        path: &std::path::Path,
    ) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_yaml(workflow))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_generator() {
        let generator = WorkflowGenerator::new("test-project");
        let workflow = generator.generate_ci();

        assert_eq!(workflow.name, "test-project CI");
        assert!(workflow.jobs.contains_key("build"));
        assert!(workflow.jobs.contains_key("test"));
    }

    #[test]
    fn test_release_workflow() {
        let generator = WorkflowGenerator::new("test-project");
        let workflow = generator.generate_release();

        assert_eq!(workflow.name, "test-project Release");
        assert!(workflow.jobs.contains_key("build"));
        assert!(workflow.jobs.contains_key("release"));
    }

    #[test]
    fn test_workflow_yaml() {
        let generator = WorkflowGenerator::new("test-project");
        let workflow = generator.generate_ci();
        let yaml = generator.to_yaml(&workflow);

        assert!(yaml.contains("name: test-project CI"));
        assert!(yaml.contains("runs-on:"));
        assert!(yaml.contains("steps:"));
    }
}
