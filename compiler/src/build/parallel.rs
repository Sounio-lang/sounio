//! Parallel build execution with work stealing.
//!
//! This module provides parallel execution of compilation tasks with
//! dependency-aware scheduling and work stealing for load balancing.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::graph::{BuildGraph, UnitId};

/// Job server for resource management (limits concurrent jobs)
pub struct JobServer {
    /// Maximum number of concurrent jobs
    max_jobs: usize,

    /// Currently active jobs
    active: Arc<Mutex<usize>>,
}

impl JobServer {
    /// Create a new job server
    pub fn new(max_jobs: usize) -> Self {
        JobServer {
            max_jobs,
            active: Arc::new(Mutex::new(0)),
        }
    }

    /// Create a job server based on CPU count
    pub fn from_cpu_count() -> Self {
        let max_jobs = num_cpus::get();
        Self::new(max_jobs)
    }

    /// Acquire a job token (blocks if at capacity)
    pub fn acquire(&self) -> JobToken {
        loop {
            {
                let mut active = self.active.lock().unwrap();
                if *active < self.max_jobs {
                    *active += 1;
                    return JobToken {
                        server: self.active.clone(),
                    };
                }
            }
            // Wait a bit before retrying
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Try to acquire a job token (returns None if at capacity)
    pub fn try_acquire(&self) -> Option<JobToken> {
        let mut active = self.active.lock().unwrap();
        if *active < self.max_jobs {
            *active += 1;
            Some(JobToken {
                server: self.active.clone(),
            })
        } else {
            None
        }
    }

    /// Get current job count
    pub fn active_count(&self) -> usize {
        *self.active.lock().unwrap()
    }

    /// Get maximum jobs
    pub fn max_jobs(&self) -> usize {
        self.max_jobs
    }
}

/// Token representing a job slot (released on drop)
pub struct JobToken {
    server: Arc<Mutex<usize>>,
}

impl Drop for JobToken {
    fn drop(&mut self) {
        let mut active = self.server.lock().unwrap();
        *active -= 1;
    }
}

/// Build task to be executed
#[derive(Clone)]
pub struct BuildTask {
    /// Unit to compile
    pub unit_id: UnitId,

    /// Priority (higher = more important)
    pub priority: i32,

    /// Dependencies that must complete first
    pub dependencies: Vec<UnitId>,
}

impl BuildTask {
    /// Create a new build task
    pub fn new(unit_id: UnitId, priority: i32, dependencies: Vec<UnitId>) -> Self {
        BuildTask {
            unit_id,
            priority,
            dependencies,
        }
    }
}

/// Result of executing a build task
#[derive(Debug, Clone)]
pub struct BuildResult {
    /// Unit that was compiled
    pub unit_id: UnitId,

    /// Whether compilation succeeded
    pub success: bool,

    /// Compilation time
    pub duration: Duration,

    /// Error message if failed
    pub error: Option<String>,
}

/// Parallel build executor
pub struct ParallelExecutor {
    /// Job server for concurrency control
    job_server: Arc<JobServer>,

    /// Build scheduler
    scheduler: BuildScheduler,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    pub fn new(max_jobs: usize) -> Self {
        ParallelExecutor {
            job_server: Arc::new(JobServer::new(max_jobs)),
            scheduler: BuildScheduler::new(),
        }
    }

    /// Create executor based on CPU count
    pub fn from_cpu_count() -> Self {
        Self::new(num_cpus::get())
    }

    /// Execute build tasks in parallel
    pub fn execute<F>(
        &mut self,
        graph: &BuildGraph,
        execute_fn: F,
    ) -> Result<Vec<BuildResult>, ParallelError>
    where
        F: Fn(UnitId) -> Result<(), String> + Send + Sync + 'static,
    {
        // Get compilation order
        let order = graph
            .compilation_order()
            .map_err(|e| ParallelError::InvalidGraph(e.to_string()))?;

        if order.is_empty() {
            return Ok(Vec::new());
        }

        // Create tasks
        let tasks: Vec<BuildTask> = order
            .iter()
            .map(|&unit_id| {
                let unit = graph.get_unit(unit_id).unwrap();
                let priority = unit.dependents.len() as i32; // Prioritize units with many dependents
                BuildTask::new(unit_id, priority, unit.dependencies.clone())
            })
            .collect();

        // Schedule and execute
        self.scheduler.schedule(tasks);
        self.execute_scheduled(execute_fn)
    }

    /// Execute scheduled tasks
    fn execute_scheduled<F>(&self, execute_fn: F) -> Result<Vec<BuildResult>, ParallelError>
    where
        F: Fn(UnitId) -> Result<(), String> + Send + Sync + 'static,
    {
        let execute_fn = Arc::new(execute_fn);
        let results = Arc::new(Mutex::new(Vec::new()));
        let completed = Arc::new(Mutex::new(Vec::new()));

        loop {
            // Get next ready task
            let task = {
                let comp = completed.lock().unwrap();
                self.scheduler.next_ready(&comp)
            };

            match task {
                Some(task) => {
                    let token = self.job_server.acquire();
                    let execute_fn = execute_fn.clone();
                    let results = results.clone();
                    let completed = completed.clone();
                    let unit_id = task.unit_id;

                    // Spawn task
                    thread::spawn(move || {
                        let start = Instant::now();
                        let result = execute_fn(unit_id);
                        let duration = start.elapsed();

                        let build_result = BuildResult {
                            unit_id,
                            success: result.is_ok(),
                            duration,
                            error: result.err(),
                        };

                        results.lock().unwrap().push(build_result);
                        completed.lock().unwrap().push(unit_id);
                        drop(token);
                    });
                }
                None => {
                    // No more ready tasks
                    if self.scheduler.is_complete(&completed.lock().unwrap()) {
                        break;
                    }
                    // Wait for some task to complete
                    thread::sleep(Duration::from_millis(10));
                }
            }
        }

        // Wait for all tasks to complete
        while self.job_server.active_count() > 0 {
            thread::sleep(Duration::from_millis(10));
        }

        let final_results = results.lock().unwrap().clone();
        Ok(final_results)
    }

    /// Get job server
    pub fn job_server(&self) -> &JobServer {
        &self.job_server
    }
}

/// Build scheduler with work-stealing
pub struct BuildScheduler {
    /// Work queues (one per priority level)
    queues: Arc<Mutex<HashMap<i32, VecDeque<BuildTask>>>>,

    /// All tasks (for tracking)
    all_tasks: Arc<Mutex<HashMap<UnitId, BuildTask>>>,
}

impl BuildScheduler {
    /// Create a new scheduler
    pub fn new() -> Self {
        BuildScheduler {
            queues: Arc::new(Mutex::new(HashMap::new())),
            all_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Schedule tasks
    pub fn schedule(&mut self, tasks: Vec<BuildTask>) {
        let mut queues = self.queues.lock().unwrap();
        let mut all_tasks = self.all_tasks.lock().unwrap();

        for task in tasks {
            let priority = task.priority;
            let unit_id = task.unit_id;

            queues.entry(priority).or_default().push_back(task.clone());

            all_tasks.insert(unit_id, task);
        }
    }

    /// Get next ready task (dependencies satisfied)
    pub fn next_ready(&self, completed: &[UnitId]) -> Option<BuildTask> {
        let mut queues = self.queues.lock().unwrap();

        // Sort priorities (highest first)
        let mut priorities: Vec<_> = queues.keys().copied().collect();
        priorities.sort_by(|a, b| b.cmp(a));

        for priority in priorities {
            if let Some(queue) = queues.get_mut(&priority) {
                // Find first task with all dependencies satisfied
                if let Some(pos) = queue
                    .iter()
                    .position(|task| task.dependencies.iter().all(|dep| completed.contains(dep)))
                {
                    return queue.remove(pos);
                }
            }
        }

        None
    }

    /// Check if all tasks are complete
    pub fn is_complete(&self, completed: &[UnitId]) -> bool {
        let all_tasks = self.all_tasks.lock().unwrap();
        all_tasks.keys().all(|unit_id| completed.contains(unit_id))
    }

    /// Get number of pending tasks
    pub fn pending_count(&self) -> usize {
        self.queues.lock().unwrap().values().map(|q| q.len()).sum()
    }
}

impl Default for BuildScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel execution errors
#[derive(Debug, thiserror::Error)]
pub enum ParallelError {
    #[error("Invalid build graph: {0}")]
    InvalidGraph(String),

    #[error("Task execution failed: {0}")]
    ExecutionError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_server() {
        let server = JobServer::new(2);

        assert_eq!(server.max_jobs(), 2);
        assert_eq!(server.active_count(), 0);

        let token1 = server.acquire();
        assert_eq!(server.active_count(), 1);

        let token2 = server.acquire();
        assert_eq!(server.active_count(), 2);

        // Should be at capacity
        assert!(server.try_acquire().is_none());

        drop(token1);
        assert_eq!(server.active_count(), 1);

        // Should be able to acquire again
        let _token3 = server.try_acquire();
        assert_eq!(server.active_count(), 2);
    }

    #[test]
    fn test_build_scheduler() {
        let mut scheduler = BuildScheduler::new();

        let task1 = BuildTask::new(UnitId(1), 1, vec![]);
        let task2 = BuildTask::new(UnitId(2), 2, vec![UnitId(1)]);
        let task3 = BuildTask::new(UnitId(3), 1, vec![]);

        scheduler.schedule(vec![task1, task2, task3]);

        assert_eq!(scheduler.pending_count(), 3);

        // Task 1 and 3 should be ready (no dependencies)
        let ready1 = scheduler.next_ready(&[]).unwrap();
        assert!(ready1.unit_id == UnitId(1) || ready1.unit_id == UnitId(3));

        // Task 2 not ready yet (depends on 1)
        let ready2 = scheduler.next_ready(&[]);
        let ready2_valid = match &ready2 {
            None => true,
            Some(task) => task.unit_id == UnitId(3) || task.unit_id == UnitId(1),
        };
        assert!(ready2_valid);

        // After completing task 1, task 2 should be ready
        let ready3 = scheduler.next_ready(&[UnitId(1), UnitId(3)]);
        assert_eq!(ready3.unwrap().unit_id, UnitId(2));
    }

    #[test]
    fn test_priority_scheduling() {
        let mut scheduler = BuildScheduler::new();

        let low = BuildTask::new(UnitId(1), 1, vec![]);
        let high = BuildTask::new(UnitId(2), 10, vec![]);

        scheduler.schedule(vec![low, high]);

        // High priority should come first
        let next = scheduler.next_ready(&[]).unwrap();
        assert_eq!(next.unit_id, UnitId(2));
        assert_eq!(next.priority, 10);
    }

    #[test]
    fn test_parallel_executor_basic() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut executor = ParallelExecutor::new(2);

        let mut graph = BuildGraph::new();
        let unit1 = super::super::graph::CompilationUnit::new(
            std::path::PathBuf::from("test1.sio"),
            super::super::graph::ContentHash::from_bytes(b"test1"),
        );
        let unit2 = super::super::graph::CompilationUnit::new(
            std::path::PathBuf::from("test2.sio"),
            super::super::graph::ContentHash::from_bytes(b"test2"),
        );

        let id1 = unit1.id;
        let id2 = unit2.id;

        graph.add_unit(unit1);
        graph.add_unit(unit2);
        graph.mark_clean(id1);
        graph.mark_clean(id2);

        // Make them dirty
        graph.invalidate(id1);
        graph.invalidate(id2);

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let results = executor
            .execute(&graph, move |_unit_id| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert!(results.iter().all(|r| r.success));
    }
}
