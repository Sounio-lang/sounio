constexpr std::string_view kProcessWitnessCellRuntimeSha256 =
    "d8b6ac5f22da5296d57524ba1ccb36b5e06a02bd1ba4387b3c7ada414cfd3cb9";
constexpr std::string_view kProcessWitnessPayloadSha256 =
    "7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d";
constexpr std::string_view kProcessWitnessPayloadManifestSha256 =
    "624ccd7297778803eff8d9972a33d5e55fb022f9e7e37f444f0aee13c22fb4da";

struct ProcessWitnessObservation {
  bool semantic_consumed = false;
  bool same_pid = false;
  bool start_tick = false;
  bool pidfd_live_at_ready = false;
  bool executable_transition = false;
  bool credential_unchanged = false;
  bool cgroup_unchanged = false;
  bool namespace_unchanged = false;
  bool environment_empty = false;
  bool descendants_empty = false;
  bool ready = false;
  bool done = false;
  bool sounio_refusal = false;
  bool pidfd_extinct = false;
  bool process_extinct = false;
  bool cgroup_unpopulated = false;
  bool unit_inactive = false;
  bool state_extinct = false;
  bool generation_extinct = false;
  bool authority_extinct = false;
  bool distinct_uid = false;

  bool transition_complete() const {
    return semantic_consumed && same_pid && start_tick && pidfd_live_at_ready &&
           executable_transition && credential_unchanged && cgroup_unchanged &&
           namespace_unchanged && environment_empty && descendants_empty &&
           ready && distinct_uid;
  }

  bool extinction_complete() const {
    return pidfd_extinct && process_extinct && cgroup_unpopulated &&
           unit_inactive && state_extinct && generation_extinct &&
           authority_extinct;
  }

  bool success_complete() const {
    return transition_complete() && done && extinction_complete();
  }
};

Manifest load_process_witness_payload_manifest(const std::string& path) {
  Manifest manifest = parse_frozen_manifest(
      path, kProcessWitnessPayloadManifestSha256,
      "ProcessWitness Sounio payload");
  if (manifest.require("schema") !=
          "loom-process-witness-handshake-payload-freeze-v1" ||
      manifest.require("stage") != "SOUNIO_HANDSHAKE_PAYLOAD_FROZEN" ||
      manifest.require("producing_language") != "Sounio" ||
      manifest.require("language_role") != "SEMANTIC_PAYLOAD" ||
      manifest.require("semantic_authority") != "Sounio" ||
      manifest.require("action") != "9030" ||
      manifest.require("executable_sha256") != kProcessWitnessPayloadSha256 ||
      manifest.require("two_phase") != "true" ||
      manifest.require("ready_before_close") != "true" ||
      manifest.require("material_grant") != "true" ||
      manifest.require("material_execution") != "false" ||
      manifest.require("launch_open") != "false" ||
      manifest.require("claim_ready") != "false") {
    throw Error("ProcessWitness payload manifest state invalid");
  }
  return manifest;
}

bool process_witness_root_artifact(const std::string& path, bool executable) {
  struct stat info {};
  return lstat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode) &&
         info.st_uid == 0 && info.st_gid == 0 && info.st_nlink == 1 &&
         (info.st_mode & 0222) == 0 &&
         (!executable || (info.st_mode & 0111) != 0);
}

std::string process_witness_link(const std::string& path) {
  std::array<char, PATH_MAX> buffer{};
  const ssize_t count = readlink(path.c_str(), buffer.data(), buffer.size() - 1);
  if (count <= 0) throw Error("ProcessWitness kernel link unavailable");
  return std::string(buffer.data(), static_cast<std::size_t>(count));
}

std::string process_witness_namespaces(pid_t pid) {
  std::ostringstream output;
  for (const std::string_view name :
       {"cgroup", "ipc", "mnt", "net", "pid", "user", "uts"}) {
    output << name << '='
           << process_witness_link("/proc/" + std::to_string(pid) + "/ns/" +
                                   std::string(name))
           << '\n';
  }
  return output.str();
}

std::string process_witness_cgroup_path(std::string_view record) {
  std::istringstream input{std::string(record)};
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t marker = line.find("::");
    if (marker != std::string::npos && marker + 2 < line.size() &&
        line[marker + 2] == '/') {
      return line.substr(marker + 2);
    }
  }
  throw Error("ProcessWitness cgroup v2 path unavailable");
}

bool process_witness_cgroup_unpopulated(const std::string& relative) {
  const std::string root = "/sys/fs/cgroup" + relative;
  struct stat info {};
  if (lstat(root.c_str(), &info) != 0) return errno == ENOENT;
  try {
    const std::string events = read_file(root + "/cgroup.events");
    return events.find("populated 0") != std::string::npos;
  } catch (...) {
    return false;
  }
}

bool process_witness_original_extinct(pid_t pid, std::uint64_t start_tick) {
  try {
    return process_start_tick(pid) != start_tick;
  } catch (...) {
    return true;
  }
}

class HostProcessWitnessCell {
 public:
  HostProcessWitnessCell(std::string runtime, std::string payload,
                         std::string payload_manifest, std::string systemd_run,
                         std::string systemctl, std::string generation,
                         bool bypass)
      : runtime_(canonical_path(runtime)), payload_(canonical_path(payload)),
        payload_manifest_(canonical_path(payload_manifest)),
        systemd_run_(canonical_path(systemd_run)),
        systemctl_(canonical_path(systemctl)),
        generation_(std::move(generation)), bypass_(bypass),
        unit_("sounio-loom-process-witness-" + std::to_string(getpid()) + "-" +
              random_generation_sha256().substr(0, 16) + ".service") {
    if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0) {
      throw Error("host ProcessWitness requires root broker identity");
    }
    if (file_sha256(runtime_) != kProcessWitnessCellRuntimeSha256 ||
        file_sha256(payload_) != kProcessWitnessPayloadSha256 ||
        file_sha256(payload_manifest_) !=
            kProcessWitnessPayloadManifestSha256) {
      throw Error("host ProcessWitness artifact hash mismatch");
    }
    if (!process_witness_root_artifact(runtime_, true) ||
        !process_witness_root_artifact(payload_, true) ||
        !process_witness_root_artifact(payload_manifest_, false)) {
      throw Error("host ProcessWitness artifact metadata unsafe");
    }
    spawn();
  }

  HostProcessWitnessCell(const HostProcessWitnessCell&) = delete;
  HostProcessWitnessCell& operator=(const HostProcessWitnessCell&) = delete;
  ~HostProcessWitnessCell() { terminate_noexcept(); }

  bool authenticated_before_release() const {
    if (bypass_) return false;
    return authenticated_for_executable(runtime_);
  }

  bool distinct_uid() const {
    return uid_ != 0 && gid_ != 0 && uid_ != getuid() && gid_ != getgid();
  }

  ProcessWitnessObservation execute(bool semantic_consumed,
                                    std::string_view close_frame) {
    if (finished_) throw Error("ProcessWitness cell already consumed");
    if (!bypass_) {
      if (!authenticated_before_release()) {
        throw Error("ProcessWitness pre-exec identity drifted");
      }
      quorum_write_all(release_.get(), "ARM\n" + generation_ + "\n",
                       std::chrono::steady_clock::now() + kExecQuorumDeadline);
    }
    ProcessWitnessObservation observation = observe_ready(semantic_consumed);
    if (!observation.transition_complete() && !bypass_) {
      throw Error("ProcessWitness transition did not authorize close");
    }
    quorum_write_all(release_.get(), close_frame,
                     std::chrono::steady_clock::now() + kExecQuorumDeadline);
    release_ = UniqueFd();
    const std::string outcome = quorum_read_line(
        result_.get(), result_buffer_, 512,
        std::chrono::steady_clock::now() + kExecQuorumDeadline,
        "ProcessWitness outcome timeout");
    observation.done = outcome == manifest_.require("done_line");
    observation.sounio_refusal =
        outcome == manifest_.require("refusal_line");
    const int status = await_client_exit();
    if (observation.done && status != 0) {
      throw Error("ProcessWitness DONE status diverged");
    }
    if (observation.sounio_refusal && status != 70) {
      throw Error("ProcessWitness refusal status diverged");
    }
    finish_extinction(observation);
    finished_ = true;
    return observation;
  }

  ProcessWitnessObservation abort_after_ready(bool semantic_consumed) {
    if (finished_ || bypass_) throw Error("ProcessWitness abort mode invalid");
    if (!authenticated_before_release()) {
      throw Error("ProcessWitness abort pre-exec identity drifted");
    }
    quorum_write_all(release_.get(), "ARM\n" + generation_ + "\n",
                     std::chrono::steady_clock::now() + kExecQuorumDeadline);
    ProcessWitnessObservation observation = observe_ready(semantic_consumed);
    release_ = UniqueFd();
    const std::string outcome = quorum_read_line(
        result_.get(), result_buffer_, 512,
        std::chrono::steady_clock::now() + kExecQuorumDeadline,
        "ProcessWitness abort outcome timeout");
    observation.sounio_refusal =
        outcome == manifest_.require("refusal_line");
    if (await_client_exit() != 70 || !observation.sounio_refusal) {
      throw Error("ProcessWitness authority-death refusal diverged");
    }
    finish_extinction(observation);
    finished_ = true;
    return observation;
  }

  bool refuse_without_release() {
    if (finished_ || bypass_ || !authenticated_before_release()) {
      throw Error("ProcessWitness treatment posture invalid");
    }
    release_ = UniqueFd();
    const std::string line = quorum_read_line(
        result_.get(), result_buffer_, 512,
        std::chrono::steady_clock::now() + kExecQuorumDeadline,
        "ProcessWitness treatment timeout");
    const int status = await_client_exit();
    ProcessWitnessObservation observation;
    finish_extinction(observation);
    finished_ = true;
    return line.rfind("LOOM_PROCESS_WITNESS_CLOSED reason=", 0) == 0 &&
           status == 70 && observation.extinction_complete();
  }

  bool refuse_release(std::string_view release) {
    if (finished_ || bypass_ || !authenticated_before_release()) {
      throw Error("ProcessWitness refusal posture invalid");
    }
    quorum_write_all(release_.get(), release,
                     std::chrono::steady_clock::now() + kExecQuorumDeadline);
    const std::string line = quorum_read_line(
        result_.get(), result_buffer_, 512,
        std::chrono::steady_clock::now() + kExecQuorumDeadline,
        "ProcessWitness refusal timeout");
    release_ = UniqueFd();
    const int status = await_client_exit();
    ProcessWitnessObservation observation;
    finish_extinction(observation);
    finished_ = true;
    return line.rfind("LOOM_PROCESS_WITNESS_CLOSED reason=", 0) == 0 &&
           status == 70 && observation.extinction_complete();
  }

 private:
  std::string property(const std::string& name) const {
    return quorum_tool_value(systemctl_,
                             {"show", unit_, "--property=" + name, "--value"});
  }

  bool unit_inactive() const {
    const QuorumToolResult result = quorum_run_tool(
        systemctl_, {"show", unit_, "--property=ActiveState", "--value"},
        std::chrono::seconds(2));
    if (result.exit_code != 0) return true;
    const std::string state = trim(result.output);
    return state == "inactive" || state == "failed" || state == "dead";
  }

  bool authenticated_for_executable(const std::string& executable) const {
    try {
      return pid_ > 1 && pidfd_live(pidfd_.get()) &&
             process_start_tick(pid_) == start_tick_ &&
             process_executable(pid_) == executable && process_uid(pid_) == uid_ &&
             process_gid(pid_) == gid_ && distinct_uid() &&
             sha256(process_cgroup(pid_)) == cgroup_sha256_ &&
             process_cgroup(pid_).find(unit_) != std::string::npos &&
             property("DynamicUser") == "yes" &&
             property("NoNewPrivileges") == "yes" &&
             property("ProtectSystem") == "strict" &&
             property("ProtectProc") == "invisible" &&
             property("ProcSubset") == "pid" &&
             property("PrivateNetwork") == "yes" &&
             property("MemoryDenyWriteExecute") == "yes";
    } catch (...) {
      return false;
    }
  }

  void spawn() {
    int release_pipe[2];
    int result_pipe[2];
    if (pipe2(release_pipe, O_CLOEXEC) != 0 ||
        pipe2(result_pipe, O_CLOEXEC) != 0) {
      throw Error("cannot create ProcessWitness custody pipes");
    }
    UniqueFd release_read(release_pipe[0]);
    UniqueFd release_write(release_pipe[1]);
    UniqueFd result_read(result_pipe[0]);
    UniqueFd result_write(result_pipe[1]);
    UniqueFd null_error(open("/dev/null", O_WRONLY | O_CLOEXEC));
    if (null_error.get() < 0) throw Error("cannot open ProcessWitness stderr");
    client_pid_ = fork();
    if (client_pid_ < 0) throw Error("cannot fork ProcessWitness systemd client");
    if (client_pid_ == 0) {
      if (dup2(release_read.get(), STDIN_FILENO) < 0 ||
          dup2(result_write.get(), STDOUT_FILENO) < 0 ||
          dup2(null_error.get(), STDERR_FILENO) < 0) {
        _exit(126);
      }
      release_read = UniqueFd();
      release_write = UniqueFd();
      result_read = UniqueFd();
      result_write = UniqueFd();
      null_error = UniqueFd();
      std::vector<std::string> arguments = {
          "--quiet",
          "--unit=" + unit_,
          "--service-type=exec",
          "--collect",
          "--pipe",
          "--wait",
          "--property=DynamicUser=yes",
          "--property=UMask=0077",
          "--property=NoNewPrivileges=yes",
          "--property=PrivateTmp=yes",
          "--property=PrivateDevices=yes",
          "--property=PrivateNetwork=yes",
          "--property=ProtectSystem=strict",
          "--property=ProtectHome=yes",
          "--property=ProtectKernelTunables=yes",
          "--property=ProtectKernelModules=yes",
          "--property=ProtectKernelLogs=yes",
          "--property=ProtectControlGroups=yes",
          "--property=ProtectClock=yes",
          "--property=ProtectProc=invisible",
          "--property=ProcSubset=pid",
          "--property=RestrictNamespaces=yes",
          "--property=RestrictSUIDSGID=yes",
          "--property=LockPersonality=yes",
          "--property=MemoryDenyWriteExecute=yes",
          "--property=RestrictRealtime=yes",
          "--property=SystemCallArchitectures=native",
          "--property=RestrictAddressFamilies=AF_UNIX",
          "--property=KillMode=mixed",
          "--property=TimeoutStopSec=2s",
          "--property=ReadOnlyPaths=" + runtime_,
          "--property=ReadOnlyPaths=" + payload_,
          "--property=ReadOnlyPaths=" + payload_manifest_,
          "--setenv=SOUNIO_LOOM_PROCESS_WITNESS_INTERNAL=1",
          "--setenv=SOUNIO_LOOM_PROCESS_WITNESS_UNIT=" + unit_,
          "--setenv=SOUNIO_LOOM_PROCESS_WITNESS_GENERATION=" + generation_};
      if (bypass_) {
        arguments.push_back(
            "--setenv=SOUNIO_LOOM_PROCESS_WITNESS_SABOTAGE=exact-release-bypass");
      }
      arguments.push_back("--");
      arguments.push_back(runtime_);
      arguments.push_back(bypass_ ? "--internal-host-process-witness-bypass"
                                  : "--internal-host-process-witness");
      arguments.push_back("--payload");
      arguments.push_back(payload_);
      arguments.push_back("--payload-manifest");
      arguments.push_back(payload_manifest_);
      std::vector<char*> values;
      values.reserve(arguments.size() + 2);
      values.push_back(const_cast<char*>(systemd_run_.c_str()));
      for (const std::string& argument : arguments) {
        values.push_back(const_cast<char*>(argument.c_str()));
      }
      values.push_back(nullptr);
      execv(systemd_run_.c_str(), values.data());
      _exit(127);
    }
    release_read = UniqueFd();
    result_write = UniqueFd();
    release_ = std::move(release_write);
    result_ = std::move(result_read);
    await_service_identity();
  }

  void observe_client() {
    if (client_pid_ <= 0 || client_reaped_) return;
    int status = 0;
    const pid_t waited = waitpid(client_pid_, &status, WNOHANG);
    if (waited < 0 && errno != EINTR) {
      throw Error("ProcessWitness systemd client wait failed");
    }
    if (waited == client_pid_) {
      client_reaped_ = true;
      client_status_ = status;
    }
  }

  void await_service_identity() {
    const auto deadline = std::chrono::steady_clock::now() + kExecQuorumDeadline;
    for (;;) {
      observe_client();
      if (client_reaped_) throw Error("ProcessWitness unit exited before identity");
      try {
        const auto parsed = parse_u64(property("MainPID"));
        if (parsed && *parsed > 1 &&
            *parsed <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
          pid_ = static_cast<pid_t>(*parsed);
          start_tick_ = process_start_tick(pid_);
          const std::string executable = process_executable(pid_);
          if ((!bypass_ && executable != runtime_) ||
              (bypass_ && executable != runtime_ && executable != payload_)) {
            throw Error("ProcessWitness initial executable mismatch");
          }
          pre_exec_cell_ = executable == runtime_;
          uid_ = process_uid(pid_);
          gid_ = process_gid(pid_);
          const std::string cgroup = process_cgroup(pid_);
          cgroup_sha256_ = sha256(cgroup);
          cgroup_relative_ = process_witness_cgroup_path(cgroup);
          namespace_sha256_ = sha256(process_witness_namespaces(pid_));
          pidfd_ = UniqueFd(open_pidfd(pid_));
          if (!distinct_uid()) {
            throw Error("ProcessWitness principal is not distinct");
          }
          if (!bypass_ && !authenticated_before_release()) {
            throw Error("ProcessWitness identity admission failed");
          }
          return;
        }
      } catch (const Error&) {
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        throw Error("ProcessWitness identity admission timed out");
      }
      poll(nullptr, 0, 5);
    }
  }

  ProcessWitnessObservation observe_ready(bool semantic_consumed) {
    const std::string ready = quorum_read_line(
        result_.get(), result_buffer_, 512,
        std::chrono::steady_clock::now() + kExecQuorumDeadline,
        "ProcessWitness READY timeout");
    ProcessWitnessObservation observation;
    observation.semantic_consumed = semantic_consumed;
    observation.ready = ready == manifest_.require("ready_line");
    observation.same_pid = pid_ > 1;
    observation.start_tick = process_start_tick(pid_) == start_tick_;
    observation.pidfd_live_at_ready = pidfd_live(pidfd_.get());
    observation.executable_transition =
        process_executable(pid_) == payload_ && (pre_exec_cell_ || bypass_);
    observation.credential_unchanged =
        process_uid(pid_) == uid_ && process_gid(pid_) == gid_;
    observation.cgroup_unchanged =
        sha256(process_cgroup(pid_)) == cgroup_sha256_;
    observation.namespace_unchanged =
        sha256(process_witness_namespaces(pid_)) == namespace_sha256_;
    observation.environment_empty =
        read_file("/proc/" + std::to_string(pid_) + "/environ").empty();
    observation.descendants_empty =
        trim(read_file("/proc/" + std::to_string(pid_) + "/task/" +
                       std::to_string(pid_) + "/children"))
            .empty();
    observation.distinct_uid = distinct_uid();
    return observation;
  }

  int await_client_exit() {
    const auto deadline = std::chrono::steady_clock::now() + kExecQuorumDeadline;
    while (!client_reaped_) {
      observe_client();
      if (client_reaped_) break;
      if (std::chrono::steady_clock::now() >= deadline) {
        throw Error("ProcessWitness systemd client exit timed out");
      }
      poll(nullptr, 0, 5);
    }
    client_pid_ = -1;
    if (WIFEXITED(client_status_)) return WEXITSTATUS(client_status_);
    return WIFSIGNALED(client_status_) ? 128 + WTERMSIG(client_status_) : 255;
  }

  void finish_extinction(ProcessWitnessObservation& observation) {
    const auto deadline = std::chrono::steady_clock::now() + kExecQuorumDeadline;
    for (;;) {
      pollfd candidate{pidfd_.get(), POLLIN, 0};
      const int ready = poll(&candidate, 1, 0);
      observation.pidfd_extinct = ready == 1 && (candidate.revents & POLLIN) != 0;
      observation.process_extinct =
          process_witness_original_extinct(pid_, start_tick_);
      observation.cgroup_unpopulated =
          process_witness_cgroup_unpopulated(cgroup_relative_);
      observation.unit_inactive = unit_inactive();
      if (observation.pidfd_extinct && observation.process_extinct &&
          observation.cgroup_unpopulated && observation.unit_inactive) {
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        throw Error("ProcessWitness extinction observation timed out");
      }
      poll(nullptr, 0, 10);
    }
    observation.authority_extinct = release_.get() < 0;
    observation.state_extinct =
        observation.pidfd_extinct && observation.process_extinct;
    observation.generation_extinct =
        observation.process_extinct && observation.cgroup_unpopulated &&
        observation.unit_inactive;
    pid_ = -1;
  }

  void terminate_noexcept() {
    release_ = UniqueFd();
    result_ = UniqueFd();
    try {
      static_cast<void>(quorum_run_tool(systemctl_, {"stop", unit_},
                                        std::chrono::seconds(2)));
      static_cast<void>(quorum_run_tool(systemctl_, {"reset-failed", unit_},
                                        std::chrono::seconds(2)));
    } catch (...) {
    }
    if (client_pid_ > 0 && !client_reaped_) {
      kill(client_pid_, SIGKILL);
      while (waitpid(client_pid_, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
    client_pid_ = -1;
    pid_ = -1;
  }

  std::string runtime_;
  std::string payload_;
  std::string payload_manifest_;
  std::string systemd_run_;
  std::string systemctl_;
  std::string generation_;
  bool bypass_ = false;
  std::string unit_;
  Manifest manifest_ = load_process_witness_payload_manifest(payload_manifest_);
  std::string cgroup_sha256_;
  std::string cgroup_relative_;
  std::string namespace_sha256_;
  UniqueFd release_;
  UniqueFd result_;
  UniqueFd pidfd_;
  std::string result_buffer_;
  pid_t client_pid_ = -1;
  pid_t pid_ = -1;
  std::uint64_t start_tick_ = 0;
  uid_t uid_ = 0;
  gid_t gid_ = 0;
  int client_status_ = 0;
  bool client_reaped_ = false;
  bool pre_exec_cell_ = false;
  bool finished_ = false;
};

bool process_witness_close_allowed(const ProcessWitnessObservation& observation) {
  return observation.transition_complete();
}

int selftest_host_process_witness(
    const std::string& root, const std::string& controller_manifest_path,
    const std::string& controller_runtime_path,
    const std::string& fixture_manifest_path,
    const std::string& fixture_bundle_path,
    const std::string& resident_runtime_path,
    const std::string& process_witness_runtime_path,
    const std::string& process_witness_payload_path,
    const std::string& process_witness_manifest_path,
    const std::string& systemd_run_path, const std::string& systemctl_path) {
  if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0 ||
      trim(read_file("/proc/1/comm")) != "systemd") {
    throw Error("host ProcessWitness selftest requires root under PID 1 systemd");
  }
  const Manifest controller_manifest = parse_frozen_manifest(
      controller_manifest_path, kFrozenExecGrantControllerManifestSha256,
      "ProcessWitness ExecGrantController");
  if (controller_manifest.require("producing_language") != "OCaml" ||
      controller_manifest.require("language_role") != "EFFECT_PARITY" ||
      controller_manifest.require("semantic_authority") != "Sounio" ||
      controller_manifest.require("action") != "9030" ||
      controller_manifest.require("material_grant") != "false" ||
      controller_manifest.require("material_execution") != "false") {
    throw Error("ProcessWitness controller manifest state invalid");
  }
  if (file_sha256(controller_runtime_path) !=
      controller_manifest.require("runtime_sha256")) {
    throw Error("ProcessWitness controller runtime hash mismatch");
  }
  const Manifest fixture_manifest = parse_frozen_manifest(
      fixture_manifest_path, kFrozenExecQuorumFixtureManifestSha256,
      "ProcessWitness Sounio fixtures");
  const auto fixtures =
      load_quorum_fixtures(fixture_manifest, fixture_bundle_path);
  const Manifest payload_manifest =
      load_process_witness_payload_manifest(process_witness_manifest_path);
  if (file_sha256(process_witness_runtime_path) !=
          kProcessWitnessCellRuntimeSha256 ||
      file_sha256(process_witness_payload_path) !=
          payload_manifest.require("executable_sha256")) {
    throw Error("ProcessWitness runtime binding invalid");
  }

  bool treatment_closed = false;
  bool positive_done = false;
  bool bypass_done = false;
  bool wrong_generation_closed = false;
  bool wrong_close_refused = false;
  bool controller_death_closed = false;
  bool broker_death_closed = false;
  bool replay_closed = false;
  bool payload_substitution_closed = false;
  bool forged_ready_closed = false;
  bool extinction_omission_closed = false;
  ProcessWitnessObservation positive;

  {
    const std::string generation = random_generation_sha256();
    HostProcessWitnessCell principal(
        process_witness_runtime_path, process_witness_payload_path,
        process_witness_manifest_path, systemd_run_path, systemctl_path,
        generation, false);
    treatment_closed = principal.refuse_without_release();
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    const QuorumReceipt issue =
        controller.command("ISSUE", fixtures.at("issue"));
    const QuorumReceipt consume =
        controller.command("CONSUME", fixtures.at("consume"));
    const bool semantic_consumed =
        consume.require("quorum_ready") == "true" &&
        consume.require("state") == "OUTCOME_PENDING" &&
        issue.require("controller_generation_sha256") == generation &&
        consume.require("controller_generation_sha256") == generation &&
        controller.authenticated();
    HostProcessWitnessCell principal(
        process_witness_runtime_path, process_witness_payload_path,
        process_witness_manifest_path, systemd_run_path, systemctl_path,
        generation, false);
    positive = principal.execute(semantic_consumed, "CLOSE\n");
    positive_done = positive.success_complete();
    static_cast<void>(controller.command("CLOSE", fixtures.at("close")));
    const std::string replay =
        controller.raw_command("CONSUME", fixtures.at("consume"));
    replay_closed = replay.find("REFUSE") != std::string::npos;
  }

  {
    const std::string generation = random_generation_sha256();
    HostProcessWitnessCell principal(
        process_witness_runtime_path, process_witness_payload_path,
        process_witness_manifest_path, systemd_run_path, systemctl_path,
        generation, true);
    ProcessWitnessObservation bypass = principal.execute(true, "CLOSE\n");
    bypass_done = bypass.success_complete();
  }

  {
    const std::string generation = random_generation_sha256();
    const std::string wrong = random_generation_sha256();
    HostProcessWitnessCell principal(
        process_witness_runtime_path, process_witness_payload_path,
        process_witness_manifest_path, systemd_run_path, systemctl_path,
        generation, false);
    wrong_generation_closed =
        principal.refuse_release("ARM\n" + wrong + "\n");
  }

  {
    const std::string generation = random_generation_sha256();
    HostProcessWitnessCell principal(
        process_witness_runtime_path, process_witness_payload_path,
        process_witness_manifest_path, systemd_run_path, systemctl_path,
        generation, false);
    const ProcessWitnessObservation wrong = principal.execute(true, "CLOZE\n");
    wrong_close_refused =
        wrong.transition_complete() && wrong.sounio_refusal &&
        wrong.extinction_complete() && !wrong.done;
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    static_cast<void>(controller.command("ISSUE", fixtures.at("issue")));
    controller.kill_controller();
    HostProcessWitnessCell principal(
        process_witness_runtime_path, process_witness_payload_path,
        process_witness_manifest_path, systemd_run_path, systemctl_path,
        generation, false);
    controller_death_closed = principal.refuse_without_release();
  }

  {
    const std::string generation = random_generation_sha256();
    HostProcessWitnessCell principal(
        process_witness_runtime_path, process_witness_payload_path,
        process_witness_manifest_path, systemd_run_path, systemctl_path,
        generation, false);
    const ProcessWitnessObservation aborted = principal.abort_after_ready(true);
    broker_death_closed = aborted.transition_complete() &&
                          aborted.sounio_refusal &&
                          aborted.extinction_complete() && !aborted.done;
  }

  try {
    HostProcessWitnessCell substituted(
        process_witness_runtime_path, process_witness_manifest_path,
        process_witness_manifest_path, systemd_run_path, systemctl_path,
        random_generation_sha256(), false);
    static_cast<void>(substituted);
  } catch (const Error&) {
    payload_substitution_closed = true;
  }

  ProcessWitnessObservation forged;
  forged.ready = true;
  forged_ready_closed = !process_witness_close_allowed(forged);
  ProcessWitnessObservation omitted = positive;
  omitted.authority_extinct = false;
  extinction_omission_closed =
      positive.success_complete() && !omitted.success_complete();

  if (!treatment_closed || !positive_done || !bypass_done ||
      !wrong_generation_closed || !wrong_close_refused ||
      !controller_death_closed || !broker_death_closed || !replay_closed ||
      !payload_substitution_closed || !forged_ready_closed ||
      !extinction_omission_closed) {
    throw Error("host ProcessWitness causal matrix diverged");
  }

  std::cout
      << "LOOM_HOST_PROCESS_WITNESS_GATE PASS semantic_authority=Sounio"
      << " controller=OCaml controller_role=EFFECT_PARITY"
      << " material_layer=C++20+Linux+systemd"
      << " material_role=MATERIAL_PARITY transitory=true"
      << " dynamic_user=true principal_distinct_uid=true"
      << " treatment=closed positive=done causal_bypass=done"
      << " causal_sabotage=PASS wrong_generation=closed"
      << " payload_substitution=closed forged_ready=closed"
      << " wrong_close=Sounio_refusal controller_death=closed"
      << " broker_death_after_ready=closed replay=closed"
      << " same_pid=true start_tick=true pidfd_at_ready=live"
      << " executable_transition=cell-to-Sounio"
      << " credential_unchanged=true cgroup_unchanged=true"
      << " namespace_unchanged=true environment_empty=true"
      << " descendants_empty=true two_phase=true"
      << " state_extinct=true generation_extinct=true"
      << " authority_extinct=true pidfd_extinct=true"
      << " cgroup_unpopulated=true unit_inactive=true"
      << " extinction_omission=closed complete_effects=false"
      << " process_witness_core=true material_grant=true"
      << " material_execution=false launch_open=false recycle_open=false"
      << " exec_attached=false commit_attached=false ci_attached=false"
      << " parity_open=false claim_ready=false"
      << " payload_manifest_sha256=" << kProcessWitnessPayloadManifestSha256
      << " payload_sha256=" << kProcessWitnessPayloadSha256
      << " cell_sha256=" << kProcessWitnessCellRuntimeSha256
      << " systemd_run_sha256=" << file_sha256(systemd_run_path)
      << " systemctl_sha256=" << file_sha256(systemctl_path) << '\n';
  return 0;
}
