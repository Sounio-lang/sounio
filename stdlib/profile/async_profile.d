//! Async Task Profiler for Demetrios
//!
//! This module provides profiling capabilities specifically designed for
//! async/await code, tracking task creation, scheduling, and execution.
//!
//! # Example
//! ```d
//! import profile::async_profile;
//!
//! fn main() with IO, Async {
//!     let profiler = AsyncProfiler::new();
//!     profiler.start();
//!
//!     // ... async code ...
//!
//!     profiler.stop();
//!     profiler.report().print();
//! }
//! ```

use super::*;

//-----------------------------------------------------------------------------
// Task Tracking
//-----------------------------------------------------------------------------

/// Unique identifier for an async task
pub type TaskId = u64;

/// Task state
pub enum TaskState {
    /// Task created but not yet scheduled
    Created,
    /// Task is ready to run
    Ready,
    /// Task is currently executing
    Running,
    /// Task is waiting on I/O or another task
    Waiting,
    /// Task completed successfully
    Completed,
    /// Task failed with error
    Failed,
    /// Task was cancelled
    Cancelled,
}

impl TaskState {
    pub fn as_str(&self) -> &str {
        match self {
            TaskState::Created => "created",
            TaskState::Ready => "ready",
            TaskState::Running => "running",
            TaskState::Waiting => "waiting",
            TaskState::Completed => "completed",
            TaskState::Failed => "failed",
            TaskState::Cancelled => "cancelled",
        }
    }
}

/// Information about an async task
pub struct TaskInfo {
    /// Unique task ID
    pub id: TaskId,
    /// Task name (if named)
    pub name: Option<string>,
    /// Parent task ID (if spawned from another task)
    pub parent_id: Option<TaskId>,
    /// Current state
    pub state: TaskState,
    /// Creation timestamp (ns since profiler start)
    pub created_at: u64,
    /// First poll timestamp
    pub first_poll_at: Option<u64>,
    /// Last poll timestamp
    pub last_poll_at: Option<u64>,
    /// Completion timestamp
    pub completed_at: Option<u64>,
    /// Number of times polled
    pub poll_count: u64,
    /// Total time spent polling (ns)
    pub busy_time_ns: u64,
    /// Time spent waiting between polls (ns)
    pub idle_time_ns: u64,
    /// Stack trace at task creation
    pub spawn_location: Vec<Frame>,
    /// What this task is waiting on
    pub waiting_on: Option<WaitTarget>,
}

impl TaskInfo {
    /// Create new task info
    pub fn new(id: TaskId, created_at: u64) -> Self {
        TaskInfo {
            id: id,
            name: None,
            parent_id: None,
            state: TaskState::Created,
            created_at: created_at,
            first_poll_at: None,
            last_poll_at: None,
            completed_at: None,
            poll_count: 0,
            busy_time_ns: 0,
            idle_time_ns: 0,
            spawn_location: Vec::new(),
            waiting_on: None,
        }
    }

    /// Get total lifetime of the task
    pub fn lifetime_ns(&self) -> u64 {
        match self.completed_at {
            Some(end) => end - self.created_at,
            None => 0,
        }
    }

    /// Get time until first poll
    pub fn time_to_first_poll_ns(&self) -> Option<u64> {
        self.first_poll_at.map(|t| t - self.created_at)
    }

    /// Calculate busy percentage
    pub fn busy_percent(&self) -> f64 {
        let total = self.busy_time_ns + self.idle_time_ns;
        if total > 0 {
            (self.busy_time_ns as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// What a task is waiting on
pub enum WaitTarget {
    /// Waiting on I/O
    Io(IoWait),
    /// Waiting on another task
    Task(TaskId),
    /// Waiting on a timer
    Timer { deadline_ns: u64 },
    /// Waiting on a channel
    Channel { channel_id: u64, operation: ChannelOp },
    /// Waiting on a lock
    Lock { lock_id: u64 },
    /// Waiting on a condition variable
    Condvar { condvar_id: u64 },
    /// Waiting on multiple things (select)
    Select(Vec<WaitTarget>),
    /// Unknown wait reason
    Unknown,
}

/// I/O wait details
pub struct IoWait {
    /// File descriptor or handle
    pub fd: i64,
    /// Type of I/O operation
    pub operation: IoOperation,
    /// Bytes requested (if applicable)
    pub bytes_requested: Option<u64>,
}

/// I/O operation type
pub enum IoOperation {
    Read,
    Write,
    Accept,
    Connect,
    Recv,
    Send,
    Poll,
}

/// Channel operation
pub enum ChannelOp {
    Send,
    Recv,
}

//-----------------------------------------------------------------------------
// Poll Events
//-----------------------------------------------------------------------------

/// A poll event on a task
pub struct PollEvent {
    /// Task that was polled
    pub task_id: TaskId,
    /// Timestamp of poll start
    pub start_ns: u64,
    /// Duration of poll
    pub duration_ns: u64,
    /// Result of poll
    pub result: PollResult,
    /// Thread that polled
    pub thread_id: u64,
}

/// Result of polling a task
pub enum PollResult {
    /// Task made progress and is ready
    Ready,
    /// Task is pending, needs to wait
    Pending,
    /// Poll panicked
    Panicked,
}

//-----------------------------------------------------------------------------
// Async Profiler
//-----------------------------------------------------------------------------

/// Async task profiler
pub struct AsyncProfiler {
    /// Active tasks
    tasks: HashMap<TaskId, TaskInfo>,
    /// Completed tasks (kept for analysis)
    completed_tasks: Vec<TaskInfo>,
    /// Poll events
    poll_events: Vec<PollEvent>,
    /// Next task ID
    next_id: TaskId,
    /// Start time
    start_time_ns: u64,
    /// Whether profiler is active
    active: bool,
    /// Current task on each thread
    current_task: HashMap<u64, TaskId>,
    /// Task spawn hierarchy
    spawn_tree: HashMap<TaskId, Vec<TaskId>>,
}

impl AsyncProfiler {
    /// Create new async profiler
    pub fn new() -> Self {
        AsyncProfiler {
            tasks: HashMap::new(),
            completed_tasks: Vec::new(),
            poll_events: Vec::new(),
            next_id: 1,
            start_time_ns: 0,
            active: false,
            current_task: HashMap::new(),
            spawn_tree: HashMap::new(),
        }
    }

    /// Start profiling
    pub fn start(&!self) with IO {
        self.active = true;
        self.start_time_ns = clock_gettime_ns();
        self.tasks.clear();
        self.completed_tasks.clear();
        self.poll_events.clear();
        self.next_id = 1;
        self.current_task.clear();
        self.spawn_tree.clear();

        // Install hooks into async runtime
        self.install_hooks();
    }

    /// Stop profiling
    pub fn stop(&!self) with IO {
        self.active = false;
        self.remove_hooks();

        // Move all remaining tasks to completed
        for (_, task) in self.tasks.drain() {
            self.completed_tasks.push(task);
        }
    }

    /// Record task creation
    pub fn on_task_created(&!self, name: Option<string>) -> TaskId with IO {
        if !self.active {
            return 0;
        }

        let id = self.next_id;
        self.next_id += 1;

        let now = clock_gettime_ns() - self.start_time_ns;
        let mut task = TaskInfo::new(id, now);
        task.name = name;

        // Get parent task (current task on this thread)
        let tid = get_thread_id();
        task.parent_id = self.current_task.get(&tid).copied();

        // Capture spawn location
        let mut depth = 0u32;
        walk_stack(|ip| {
            if depth < 16 {
                task.spawn_location.push(Frame::new(ip));
                depth += 1;
                true
            } else {
                false
            }
        });

        // Update spawn tree
        if let Some(parent) = task.parent_id {
            let children = self.spawn_tree.entry(parent).or_insert(Vec::new());
            children.push(id);
        }

        self.tasks.insert(id, task);
        id
    }

    /// Record task poll start
    pub fn on_poll_start(&!self, task_id: TaskId) with IO {
        if !self.active {
            return;
        }

        let now = clock_gettime_ns() - self.start_time_ns;
        let tid = get_thread_id();

        // Update current task for this thread
        self.current_task.insert(tid, task_id);

        if let Some(task) = self.tasks.get_mut(&task_id) {
            // Record first poll
            if task.first_poll_at.is_none() {
                task.first_poll_at = Some(now);
            }

            // Calculate idle time since last poll
            if let Some(last) = task.last_poll_at {
                task.idle_time_ns += now - last;
            }

            task.state = TaskState::Running;
            task.poll_count += 1;
        }
    }

    /// Record task poll end
    pub fn on_poll_end(&!self, task_id: TaskId, result: PollResult) with IO {
        if !self.active {
            return;
        }

        let now = clock_gettime_ns() - self.start_time_ns;
        let tid = get_thread_id();

        // Clear current task
        self.current_task.remove(&tid);

        if let Some(task) = self.tasks.get_mut(&task_id) {
            // Calculate poll duration
            if let Some(last) = task.last_poll_at {
                task.busy_time_ns += now - last;
            }

            task.last_poll_at = Some(now);

            match result {
                PollResult::Ready => {
                    // Task completed
                    task.state = TaskState::Completed;
                    task.completed_at = Some(now);

                    // Move to completed
                    if let Some(completed_task) = self.tasks.remove(&task_id) {
                        self.completed_tasks.push(completed_task);
                    }
                }
                PollResult::Pending => {
                    task.state = TaskState::Waiting;
                }
                PollResult::Panicked => {
                    task.state = TaskState::Failed;
                    task.completed_at = Some(now);

                    if let Some(completed_task) = self.tasks.remove(&task_id) {
                        self.completed_tasks.push(completed_task);
                    }
                }
            }
        }

        // Record poll event
        self.poll_events.push(PollEvent {
            task_id: task_id,
            start_ns: now, // This should track start time separately
            duration_ns: 0, // Calculate properly
            result: result,
            thread_id: tid,
        });
    }

    /// Record what a task is waiting on
    pub fn on_task_waiting(&!self, task_id: TaskId, target: WaitTarget) {
        if !self.active {
            return;
        }

        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.waiting_on = Some(target);
        }
    }

    /// Record task cancellation
    pub fn on_task_cancelled(&!self, task_id: TaskId) with IO {
        if !self.active {
            return;
        }

        let now = clock_gettime_ns() - self.start_time_ns;

        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.state = TaskState::Cancelled;
            task.completed_at = Some(now);
        }

        if let Some(completed_task) = self.tasks.remove(&task_id) {
            self.completed_tasks.push(completed_task);
        }
    }

    /// Get number of active tasks
    pub fn active_task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Get task info by ID
    pub fn get_task(&self, id: TaskId) -> Option<&TaskInfo> {
        self.tasks.get(&id).or_else(|| {
            self.completed_tasks.iter().find(|t| t.id == id)
        })
    }

    /// Get all active tasks
    pub fn active_tasks(&self) -> impl Iterator<Item = &TaskInfo> {
        self.tasks.values()
    }

    /// Get all completed tasks
    pub fn completed_tasks(&self) -> &[TaskInfo] {
        &self.completed_tasks
    }

    /// Generate async profile report
    pub fn report(&self) -> AsyncProfileReport {
        let all_tasks: Vec<_> = self.tasks.values()
            .chain(self.completed_tasks.iter())
            .collect();

        AsyncProfileReport::from_tasks(&all_tasks, &self.poll_events)
    }

    /// Generate task timeline for visualization
    pub fn timeline(&self) -> TaskTimeline {
        TaskTimeline::from_profiler(self)
    }

    fn install_hooks(&self) with IO {
        // Hook into async runtime
        // This would register callbacks for:
        // - Task spawning
        // - Poll start/end
        // - Wake events
    }

    fn remove_hooks(&self) with IO {
        // Remove async runtime hooks
    }
}

//-----------------------------------------------------------------------------
// Async Profile Report
//-----------------------------------------------------------------------------

/// Async profiling report
pub struct AsyncProfileReport {
    /// Total number of tasks
    pub total_tasks: u64,
    /// Completed tasks
    pub completed_tasks: u64,
    /// Failed tasks
    pub failed_tasks: u64,
    /// Cancelled tasks
    pub cancelled_tasks: u64,
    /// Total polls across all tasks
    pub total_polls: u64,
    /// Total busy time (ns)
    pub total_busy_time_ns: u64,
    /// Total idle time (ns)
    pub total_idle_time_ns: u64,
    /// Slowest tasks
    pub slowest_tasks: Vec<TaskSummary>,
    /// Most polled tasks
    pub most_polled_tasks: Vec<TaskSummary>,
    /// Tasks with longest idle time
    pub longest_waiting_tasks: Vec<TaskSummary>,
}

/// Summary of a task
pub struct TaskSummary {
    /// Task ID
    pub id: TaskId,
    /// Task name
    pub name: Option<string>,
    /// Total lifetime (ns)
    pub lifetime_ns: u64,
    /// Busy time (ns)
    pub busy_time_ns: u64,
    /// Idle time (ns)
    pub idle_time_ns: u64,
    /// Poll count
    pub poll_count: u64,
    /// Final state
    pub state: TaskState,
}

impl AsyncProfileReport {
    /// Create report from tasks and events
    pub fn from_tasks(tasks: &[&TaskInfo], events: &[PollEvent]) -> Self {
        let mut completed = 0u64;
        let mut failed = 0u64;
        let mut cancelled = 0u64;
        let mut total_polls = 0u64;
        let mut total_busy = 0u64;
        let mut total_idle = 0u64;

        let mut summaries: Vec<TaskSummary> = Vec::new();

        for task in tasks {
            match task.state {
                TaskState::Completed => completed += 1,
                TaskState::Failed => failed += 1,
                TaskState::Cancelled => cancelled += 1,
                _ => {}
            }

            total_polls += task.poll_count;
            total_busy += task.busy_time_ns;
            total_idle += task.idle_time_ns;

            summaries.push(TaskSummary {
                id: task.id,
                name: task.name.clone(),
                lifetime_ns: task.lifetime_ns(),
                busy_time_ns: task.busy_time_ns,
                idle_time_ns: task.idle_time_ns,
                poll_count: task.poll_count,
                state: task.state.clone(),
            });
        }

        // Sort for slowest tasks
        let mut slowest = summaries.clone();
        slowest.sort_by(|a, b| b.lifetime_ns.cmp(&a.lifetime_ns));
        slowest.truncate(10);

        // Sort for most polled
        let mut most_polled = summaries.clone();
        most_polled.sort_by(|a, b| b.poll_count.cmp(&a.poll_count));
        most_polled.truncate(10);

        // Sort for longest waiting
        let mut longest_waiting = summaries;
        longest_waiting.sort_by(|a, b| b.idle_time_ns.cmp(&a.idle_time_ns));
        longest_waiting.truncate(10);

        AsyncProfileReport {
            total_tasks: tasks.len() as u64,
            completed_tasks: completed,
            failed_tasks: failed,
            cancelled_tasks: cancelled,
            total_polls: total_polls,
            total_busy_time_ns: total_busy,
            total_idle_time_ns: total_idle,
            slowest_tasks: slowest,
            most_polled_tasks: most_polled,
            longest_waiting_tasks: longest_waiting,
        }
    }

    /// Print report to stdout
    pub fn print(&self) with IO {
        println!("=== Async Profile Report ===");
        println!("");
        println!("Task Summary:");
        println!("  Total tasks:      {}", self.total_tasks);
        println!("  Completed:        {}", self.completed_tasks);
        println!("  Failed:           {}", self.failed_tasks);
        println!("  Cancelled:        {}", self.cancelled_tasks);
        println!("");
        println!("Timing:");
        println!("  Total polls:      {}", self.total_polls);
        println!("  Total busy time:  {:.2}ms", self.total_busy_time_ns as f64 / 1_000_000.0);
        println!("  Total idle time:  {:.2}ms", self.total_idle_time_ns as f64 / 1_000_000.0);

        let busy_percent = if self.total_busy_time_ns + self.total_idle_time_ns > 0 {
            (self.total_busy_time_ns as f64 /
             (self.total_busy_time_ns + self.total_idle_time_ns) as f64) * 100.0
        } else {
            0.0
        };
        println!("  Busy percentage:  {:.1}%", busy_percent);
        println!("");

        println!("Slowest Tasks:");
        for (i, task) in self.slowest_tasks.iter().enumerate() {
            let name = task.name.clone().unwrap_or(format!("task-{}", task.id));
            println!("  {}. {} - {:.2}ms ({} polls)",
                i + 1, name,
                task.lifetime_ns as f64 / 1_000_000.0,
                task.poll_count);
        }
        println!("");

        println!("Most Polled Tasks:");
        for (i, task) in self.most_polled_tasks.iter().enumerate() {
            let name = task.name.clone().unwrap_or(format!("task-{}", task.id));
            println!("  {}. {} - {} polls ({:.2}ms total)",
                i + 1, name,
                task.poll_count,
                task.lifetime_ns as f64 / 1_000_000.0);
        }
        println!("");

        println!("Longest Waiting Tasks:");
        for (i, task) in self.longest_waiting_tasks.iter().enumerate() {
            let name = task.name.clone().unwrap_or(format!("task-{}", task.id));
            println!("  {}. {} - {:.2}ms idle ({:.1}% of lifetime)",
                i + 1, name,
                task.idle_time_ns as f64 / 1_000_000.0,
                100.0 - task.busy_time_ns as f64 / task.lifetime_ns.max(1) as f64 * 100.0);
        }
    }
}

//-----------------------------------------------------------------------------
// Task Timeline
//-----------------------------------------------------------------------------

/// Timeline visualization data
pub struct TaskTimeline {
    /// Timeline entries
    entries: Vec<TimelineEntry>,
    /// Total duration
    duration_ns: u64,
}

/// A timeline entry
pub struct TimelineEntry {
    /// Task ID
    pub task_id: TaskId,
    /// Task name
    pub name: Option<string>,
    /// Spans of activity
    pub spans: Vec<TimelineSpan>,
}

/// A span of activity on the timeline
pub struct TimelineSpan {
    /// Start time (ns)
    pub start_ns: u64,
    /// End time (ns)
    pub end_ns: u64,
    /// Type of activity
    pub activity: ActivityType,
}

/// Type of task activity
pub enum ActivityType {
    /// Task is running (being polled)
    Running,
    /// Task is waiting
    Waiting,
    /// Task is ready but not scheduled
    Ready,
}

impl TaskTimeline {
    /// Create timeline from profiler data
    pub fn from_profiler(profiler: &AsyncProfiler) -> Self {
        let mut entries = Vec::new();
        let mut max_time = 0u64;

        // Combine active and completed tasks
        let all_tasks: Vec<_> = profiler.tasks.values()
            .chain(profiler.completed_tasks.iter())
            .collect();

        for task in all_tasks {
            let mut spans = Vec::new();

            // Create spans from poll events
            // (Simplified - real implementation would track actual poll intervals)
            if let (Some(first), Some(last)) = (task.first_poll_at, task.last_poll_at) {
                spans.push(TimelineSpan {
                    start_ns: first,
                    end_ns: last,
                    activity: ActivityType::Running,
                });

                if last > max_time {
                    max_time = last;
                }
            }

            entries.push(TimelineEntry {
                task_id: task.id,
                name: task.name.clone(),
                spans: spans,
            });
        }

        TaskTimeline {
            entries: entries,
            duration_ns: max_time,
        }
    }

    /// Render timeline as ASCII art
    pub fn render_ascii(&self, width: usize) -> string with IO {
        let mut output = String::new();

        // Header
        output.push_str(&format!("Task Timeline ({}ms total)\n",
            self.duration_ns as f64 / 1_000_000.0));
        output.push_str(&"─".repeat(width));
        output.push_str("\n");

        let ns_per_char = if self.duration_ns > 0 {
            self.duration_ns / width as u64
        } else {
            1
        };

        for entry in &self.entries {
            let name = entry.name.clone()
                .unwrap_or(format!("task-{}", entry.task_id));
            let name_col = format!("{:12}", truncate(&name, 12));
            output.push_str(&name_col);
            output.push_str(" │");

            let mut timeline: Vec<char> = vec![' '; width - 14];

            for span in &entry.spans {
                let start = (span.start_ns / ns_per_char) as usize;
                let end = (span.end_ns / ns_per_char) as usize;

                let char = match span.activity {
                    ActivityType::Running => '█',
                    ActivityType::Waiting => '░',
                    ActivityType::Ready => '▒',
                };

                for i in start..end.min(timeline.len()) {
                    timeline[i] = char;
                }
            }

            output.push_str(&timeline.iter().collect::<String>());
            output.push_str("│\n");
        }

        output.push_str(&"─".repeat(width));
        output.push_str("\n");
        output.push_str("Legend: █ Running  ░ Waiting  ▒ Ready\n");

        output
    }

    /// Export timeline as Chrome trace format
    pub fn to_chrome_trace(&self) -> string {
        let mut events = Vec::new();

        for entry in &self.entries {
            let name = entry.name.clone()
                .unwrap_or(format!("task-{}", entry.task_id));

            for span in &entry.spans {
                // Begin event
                events.push(format!(
                    r#"{{"name":"{}","cat":"task","ph":"B","ts":{},"pid":1,"tid":{}}}"#,
                    name,
                    span.start_ns / 1000, // Convert to microseconds
                    entry.task_id
                ));

                // End event
                events.push(format!(
                    r#"{{"name":"{}","cat":"task","ph":"E","ts":{},"pid":1,"tid":{}}}"#,
                    name,
                    span.end_ns / 1000,
                    entry.task_id
                ));
            }
        }

        format!("{{\"traceEvents\":[{}]}}", events.join(","))
    }
}

//-----------------------------------------------------------------------------
// Utility
//-----------------------------------------------------------------------------

fn truncate(s: &string, max: usize) -> string {
    if s.len() <= max {
        s.clone()
    } else {
        format!("{}…", &s[..max-1])
    }
}
