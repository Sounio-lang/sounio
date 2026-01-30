//! GitLab CI integration
//!
//! This module generates GitLab CI/CD configuration files.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// GitLab CI pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    /// Stages
    pub stages: Vec<String>,

    /// Variables
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub variables: HashMap<String, String>,

    /// Default settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<DefaultSettings>,

    /// Cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheConfig>,

    /// Jobs (flattened in YAML)
    #[serde(flatten)]
    pub jobs: HashMap<String, GitLabJob>,
}

/// Default settings
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DefaultSettings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_script: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_script: Option<Vec<String>>,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub key: String,
    pub paths: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
}

/// GitLab job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitLabJob {
    /// Stage
    pub stage: String,

    /// Image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,

    /// Script
    pub script: Vec<String>,

    /// Before script
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_script: Option<Vec<String>>,

    /// After script
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_script: Option<Vec<String>>,

    /// Variables
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub variables: HashMap<String, String>,

    /// Artifacts
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<Artifacts>,

    /// Only (deprecated but still used)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,

    /// Rules
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rules: Option<Vec<Rule>>,

    /// Needs (dependencies)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub needs: Option<Vec<String>>,

    /// Parallel matrix
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel: Option<Parallel>,

    /// Tags
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,

    /// Allow failure
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_failure: Option<bool>,

    /// Timeout
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<String>,
}

/// Artifacts configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifacts {
    pub paths: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub expire_in: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reports: Option<Reports>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub when: Option<String>,
}

/// Reports
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Reports {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub junit: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub coverage_report: Option<CoverageReport>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub codequality: Option<String>,
}

/// Coverage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub coverage_format: String,
    pub path: String,
}

/// Rule
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Rule {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "if")]
    pub condition: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub when: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub changes: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub exists: Option<Vec<String>>,
}

/// Parallel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parallel {
    pub matrix: Vec<HashMap<String, Vec<String>>>,
}

/// Pipeline generator
pub struct PipelineGenerator {
    project_name: String,
    targets: Vec<String>,
    image: String,
}

impl PipelineGenerator {
    /// Create new pipeline generator
    pub fn new(project_name: &str) -> Self {
        PipelineGenerator {
            project_name: project_name.to_string(),
            targets: vec!["x86_64-unknown-linux-gnu".into()],
            image: "d-lang/d:latest".into(),
        }
    }

    /// Set targets
    pub fn targets(mut self, targets: Vec<String>) -> Self {
        self.targets = targets;
        self
    }

    /// Set base image
    pub fn image(mut self, image: &str) -> Self {
        self.image = image.to_string();
        self
    }

    /// Generate pipeline
    pub fn generate(&self) -> Pipeline {
        let mut jobs = HashMap::new();

        // Build job
        jobs.insert(
            "build".into(),
            GitLabJob {
                stage: "build".into(),
                image: Some(self.image.clone()),
                script: vec!["souc build --target $TARGET --release".into()],
                before_script: Some(vec!["souc --version".into()]),
                after_script: None,
                variables: HashMap::new(),
                artifacts: Some(Artifacts {
                    paths: vec!["target/$TARGET/release/*".into()],
                    expire_in: Some("1 week".into()),
                    reports: None,
                    when: Some("on_success".into()),
                }),
                only: None,
                rules: Some(vec![Rule {
                    condition: Some("$CI_COMMIT_BRANCH".into()),
                    when: Some("on_success".into()),
                    changes: None,
                    exists: None,
                }]),
                needs: None,
                parallel: Some(Parallel {
                    matrix: vec![
                        [("TARGET".into(), self.targets.clone())]
                            .into_iter()
                            .collect(),
                    ],
                }),
                tags: None,
                allow_failure: None,
                timeout: Some("30m".into()),
            },
        );

        // Test job
        jobs.insert(
            "test".into(),
            GitLabJob {
                stage: "test".into(),
                image: Some(self.image.clone()),
                script: vec!["souc test --all".into()],
                before_script: None,
                after_script: None,
                variables: HashMap::new(),
                artifacts: Some(Artifacts {
                    paths: vec!["target/test-results.xml".into()],
                    expire_in: Some("1 week".into()),
                    reports: Some(Reports {
                        junit: Some("target/test-results.xml".into()),
                        coverage_report: Some(CoverageReport {
                            coverage_format: "cobertura".into(),
                            path: "target/coverage.xml".into(),
                        }),
                        codequality: None,
                    }),
                    when: Some("always".into()),
                }),
                only: None,
                rules: None,
                needs: None,
                parallel: None,
                tags: None,
                allow_failure: None,
                timeout: Some("15m".into()),
            },
        );

        // Lint job
        jobs.insert(
            "lint".into(),
            GitLabJob {
                stage: "test".into(),
                image: Some(self.image.clone()),
                script: vec!["souc fmt --check".into(), "souc lint".into()],
                before_script: None,
                after_script: None,
                variables: HashMap::new(),
                artifacts: Some(Artifacts {
                    paths: vec!["lint-report.json".into()],
                    expire_in: Some("1 week".into()),
                    reports: Some(Reports {
                        codequality: Some("lint-report.json".into()),
                        ..Default::default()
                    }),
                    when: Some("always".into()),
                }),
                only: None,
                rules: None,
                needs: None,
                parallel: None,
                tags: None,
                allow_failure: Some(true),
                timeout: Some("10m".into()),
            },
        );

        // Deploy job
        jobs.insert(
            "deploy".into(),
            GitLabJob {
                stage: "deploy".into(),
                image: Some(self.image.clone()),
                script: vec!["souc publish".into()],
                before_script: None,
                after_script: None,
                variables: [("D_TOKEN".into(), "$D_REGISTRY_TOKEN".into())]
                    .into_iter()
                    .collect(),
                artifacts: None,
                only: None,
                rules: Some(vec![Rule {
                    condition: Some("$CI_COMMIT_TAG".into()),
                    when: Some("on_success".into()),
                    changes: None,
                    exists: None,
                }]),
                needs: Some(vec!["build".into(), "test".into()]),
                parallel: None,
                tags: None,
                allow_failure: None,
                timeout: Some("10m".into()),
            },
        );

        // Pages job (documentation)
        jobs.insert(
            "pages".into(),
            GitLabJob {
                stage: "deploy".into(),
                image: Some(self.image.clone()),
                script: vec![
                    "souc doc".into(),
                    "mkdir -p public".into(),
                    "cp -r target/doc/* public/".into(),
                ],
                before_script: None,
                after_script: None,
                variables: HashMap::new(),
                artifacts: Some(Artifacts {
                    paths: vec!["public".into()],
                    expire_in: None,
                    reports: None,
                    when: Some("on_success".into()),
                }),
                only: None,
                rules: Some(vec![Rule {
                    condition: Some("$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH".into()),
                    when: Some("on_success".into()),
                    changes: None,
                    exists: None,
                }]),
                needs: Some(vec!["test".into()]),
                parallel: None,
                tags: None,
                allow_failure: None,
                timeout: Some("10m".into()),
            },
        );

        Pipeline {
            stages: vec!["build".into(), "test".into(), "deploy".into()],
            variables: [
                ("D_BACKTRACE".into(), "1".into()),
                ("GIT_DEPTH".into(), "0".into()),
            ]
            .into_iter()
            .collect(),
            default: Some(DefaultSettings {
                image: Some(self.image.clone()),
                before_script: None,
                after_script: None,
            }),
            cache: Some(CacheConfig {
                key: "$CI_COMMIT_REF_SLUG".into(),
                paths: vec!["~/.d/cache".into(), "target".into()],
                policy: Some("pull-push".into()),
            }),
            jobs,
        }
    }

    /// Generate to YAML string
    pub fn to_yaml(&self, pipeline: &Pipeline) -> String {
        serde_yaml::to_string(pipeline).unwrap()
    }

    /// Write pipeline to file
    pub fn write_pipeline(
        &self,
        pipeline: &Pipeline,
        path: &std::path::Path,
    ) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_yaml(pipeline))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_generator() {
        let generator = PipelineGenerator::new("test-project");
        let pipeline = generator.generate();

        assert_eq!(pipeline.stages.len(), 3);
        assert!(pipeline.jobs.contains_key("build"));
        assert!(pipeline.jobs.contains_key("test"));
        assert!(pipeline.jobs.contains_key("lint"));
        assert!(pipeline.jobs.contains_key("deploy"));
    }

    #[test]
    fn test_pipeline_yaml() {
        let generator = PipelineGenerator::new("test-project");
        let pipeline = generator.generate();
        let yaml = generator.to_yaml(&pipeline);

        assert!(yaml.contains("stages:"));
        assert!(yaml.contains("build:"));
        assert!(yaml.contains("test:"));
    }

    #[test]
    fn test_custom_targets() {
        let generator = PipelineGenerator::new("test-project").targets(vec![
            "x86_64-unknown-linux-gnu".into(),
            "aarch64-unknown-linux-gnu".into(),
        ]);

        let pipeline = generator.generate();
        let build_job = pipeline.jobs.get("build").unwrap();

        if let Some(parallel) = &build_job.parallel {
            let targets = parallel.matrix[0].get("TARGET").unwrap();
            assert_eq!(targets.len(), 2);
        }
    }
}
