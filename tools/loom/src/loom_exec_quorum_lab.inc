constexpr std::string_view kFrozenExecGrantControllerManifestSha256 =
    "62c110afe5da72ee88dc231219f4b5cdb02a936105f8be71391ce25993aa5d95";
constexpr std::string_view kFrozenExecQuorumFixtureManifestSha256 =
    "10401ebe4d302647220433eadb0b1240ce2b3128801f421f2712ae757f5105b5";
constexpr std::string_view kIntegratedPrincipalCellRuntimeSha256 =
    "7c67b2b022f9b6c5130573a6cac652a87db3b166a67a0c29673090bc1936d116";
constexpr std::string_view kHostExecQuorumPrincipalCellRuntimeSha256 =
    "2bc6aa3429b1f0b2a886b84cc77f7867c079d7efde0bef352354d2734f170efe";
constexpr auto kExecQuorumDeadline = std::chrono::seconds(5);

struct QuorumReceipt {
  std::map<std::string, std::string> fields;
  std::string line;

  const std::string& require(const std::string& key) const {
    const auto found = fields.find(key);
    if (found == fields.end() || found->second.empty()) {
      throw Error("controller receipt omitted " + key);
    }
    return found->second;
  }
};

struct BarrierObservation {
  bool opened = false;
  std::string reason;
};

void quorum_wait(int descriptor, short events,
                 std::chrono::steady_clock::time_point deadline,
                 std::string_view reason) {
  for (;;) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) throw Error(std::string(reason));
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now);
    pollfd candidate{descriptor, events, 0};
    const int ready = poll(&candidate, 1, static_cast<int>(remaining.count()) + 1);
    if (ready > 0) return;
    if (ready == 0) throw Error(std::string(reason));
    if (errno != EINTR) throw Error("quorum descriptor poll failed");
  }
}

void quorum_write_all(int descriptor, std::string_view value,
                      std::chrono::steady_clock::time_point deadline) {
  std::size_t offset = 0;
  while (offset < value.size()) {
    quorum_wait(descriptor, POLLOUT, deadline, "quorum descriptor write timeout");
    const ssize_t count =
        write(descriptor, value.data() + offset, value.size() - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 &&
               (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    } else {
      throw Error("quorum descriptor write failed");
    }
  }
}

std::string quorum_read_line(int descriptor, std::string& buffered,
                             std::size_t limit,
                             std::chrono::steady_clock::time_point deadline,
                             std::string_view timeout_reason) {
  std::array<char, 4096> bytes{};
  for (;;) {
    const std::size_t newline = buffered.find('\n');
    if (newline != std::string::npos) {
      std::string line = buffered.substr(0, newline);
      buffered.erase(0, newline + 1);
      return line;
    }
    if (buffered.size() > limit) throw Error("quorum response exceeded limit");
    quorum_wait(descriptor, POLLIN | POLLHUP, deadline, timeout_reason);
    const ssize_t count = read(descriptor, bytes.data(), bytes.size());
    if (count > 0) {
      buffered.append(bytes.data(), static_cast<std::size_t>(count));
    } else if (count == 0) {
      throw Error("quorum response EOF");
    } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
      throw Error("quorum response read failed");
    }
  }
}

bool lowercase_digest(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return std::isdigit(character) || (character >= 'a' && character <= 'f');
         });
}

std::string process_cgroup(pid_t pid) {
  return read_file("/proc/" + std::to_string(pid) + "/cgroup");
}

std::uint64_t process_credential(pid_t pid, std::string_view label) {
  std::istringstream input(read_file("/proc/" + std::to_string(pid) + "/status"));
  std::string line;
  const std::string prefix(label);
  while (std::getline(input, line)) {
    if (line.rfind(prefix, 0) != 0) continue;
    std::istringstream fields(line.substr(prefix.size()));
    std::array<std::uint64_t, 4> values{};
    if (!(fields >> values[0] >> values[1] >> values[2] >> values[3]) ||
        values[0] != values[1] || values[0] != values[2] ||
        values[0] != values[3]) {
      throw Error("process credential vector is split");
    }
    return values[0];
  }
  throw Error("process credential field unavailable");
}

uid_t process_uid(pid_t pid) {
  const std::uint64_t value = process_credential(pid, "Uid:");
  if (value > static_cast<std::uint64_t>(std::numeric_limits<uid_t>::max())) {
    throw Error("process UID exceeds native range");
  }
  return static_cast<uid_t>(value);
}

int open_pidfd(pid_t pid) {
#if defined(SYS_pidfd_open)
  const int descriptor = static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
  if (descriptor < 0) throw Error("pidfd_open failed");
  return descriptor;
#else
  static_cast<void>(pid);
  throw Error("pidfd_open is unavailable");
#endif
}

bool pidfd_live(int descriptor) {
  pollfd candidate{descriptor, POLLIN, 0};
  const int ready = poll(&candidate, 1, 0);
  return ready == 0;
}

gid_t process_gid(pid_t pid) {
  const std::uint64_t value = process_credential(pid, "Gid:");
  if (value > static_cast<std::uint64_t>(std::numeric_limits<gid_t>::max())) {
    throw Error("process GID exceeds native range");
  }
  return static_cast<gid_t>(value);
}

struct QuorumToolResult {
  int exit_code = 255;
  std::string output;
};

QuorumToolResult quorum_run_tool(const std::string& executable,
                                 const std::vector<std::string>& arguments,
                                 std::chrono::milliseconds timeout) {
  int output_pipe[2];
  if (pipe2(output_pipe, O_CLOEXEC) != 0) {
    throw Error("cannot create tool observation pipe");
  }
  UniqueFd output_read(output_pipe[0]);
  UniqueFd output_write(output_pipe[1]);
  const pid_t child = fork();
  if (child < 0) throw Error("cannot fork host observation tool");
  if (child == 0) {
    if (dup2(output_write.get(), STDOUT_FILENO) < 0 ||
        dup2(output_write.get(), STDERR_FILENO) < 0) {
      _exit(126);
    }
    output_read = UniqueFd();
    output_write = UniqueFd();
    std::vector<char*> values;
    values.reserve(arguments.size() + 2);
    values.push_back(const_cast<char*>(executable.c_str()));
    for (const std::string& argument : arguments) {
      values.push_back(const_cast<char*>(argument.c_str()));
    }
    values.push_back(nullptr);
    execv(executable.c_str(), values.data());
    _exit(127);
  }
  output_write = UniqueFd();
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  QuorumToolResult result;
  std::array<char, 4096> buffer{};
  bool eof = false;
  while (!eof) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      kill(child, SIGKILL);
      while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {
      }
      throw Error("host observation tool timed out");
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now);
    pollfd candidate{output_read.get(), POLLIN | POLLHUP, 0};
    const int ready = poll(&candidate, 1, static_cast<int>(remaining.count()) + 1);
    if (ready < 0 && errno == EINTR) continue;
    if (ready <= 0) continue;
    const ssize_t count = read(output_read.get(), buffer.data(), buffer.size());
    if (count > 0) {
      if (result.output.size() + static_cast<std::size_t>(count) > 64 * 1024) {
        kill(child, SIGKILL);
        while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {
        }
        throw Error("host observation tool output exceeded limit");
      }
      result.output.append(buffer.data(), static_cast<std::size_t>(count));
    } else if (count == 0) {
      eof = true;
    } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
      throw Error("host observation tool read failed");
    }
  }
  int status = 0;
  pid_t waited = -1;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited != child) throw Error("host observation tool wait failed");
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.exit_code = 128 + WTERMSIG(status);
  }
  return result;
}

std::string quorum_tool_value(const std::string& executable,
                              const std::vector<std::string>& arguments) {
  const QuorumToolResult result =
      quorum_run_tool(executable, arguments, std::chrono::seconds(2));
  if (result.exit_code != 0) {
    throw Error("host observation tool refused: " + trim(result.output));
  }
  return trim(result.output);
}

void await_exec_identity(pid_t pid, const std::string& executable,
                         std::uint64_t& start_tick) {
  const auto deadline = std::chrono::steady_clock::now() + kExecQuorumDeadline;
  for (;;) {
    int status = 0;
    const pid_t waited = waitpid(pid, &status, WNOHANG);
    if (waited == pid) throw Error("quorum child exited before identity admission");
    try {
      start_tick = process_start_tick(pid);
      if (process_executable(pid) == executable) return;
    } catch (...) {
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw Error("quorum child identity admission timeout");
    }
    poll(nullptr, 0, 1);
  }
}

QuorumReceipt parse_quorum_receipt(const std::string& line) {
  std::istringstream input(line);
  std::string prefix;
  if (!(input >> prefix) || prefix != "LOOM_EXEC_GRANT_CONTROLLER_V1") {
    throw Error("controller receipt prefix invalid");
  }
  QuorumReceipt receipt;
  receipt.line = line;
  std::string token;
  while (input >> token) {
    const std::size_t equals = token.find('=');
    if (equals == std::string::npos || equals == 0 || equals + 1 >= token.size()) {
      throw Error("controller receipt token malformed");
    }
    if (!receipt.fields.emplace(token.substr(0, equals), token.substr(equals + 1)).second) {
      throw Error("controller receipt field duplicated");
    }
  }
  if (receipt.require("semantic_authority") != "Sounio" ||
      receipt.require("single_resident_controller") != "true" ||
      receipt.require("non_bearer_transport") != "pending" ||
      receipt.require("material_grant") != "false" ||
      receipt.require("material_execution") != "false" ||
      receipt.require("barrier_release") != "false" ||
      receipt.require("exec_attached") != "false" ||
      receipt.require("parity_open") != "false" ||
      receipt.require("claim_ready") != "false") {
    throw Error("controller receipt promoted beyond frozen evidence");
  }
  for (const std::string key : {"controller_generation_sha256",
                                "transaction_digest_sha256", "frame_sha256",
                                "resident_generation_sha256"}) {
    if (!lowercase_digest(receipt.require(key))) {
      throw Error("controller receipt digest invalid");
    }
  }
  return receipt;
}

void validate_quorum_receipt(const QuorumReceipt& receipt,
                             std::string_view operation,
                             const std::string& generation,
                             const std::string& frame) {
  if (receipt.require("operation") != operation ||
      receipt.require("controller_generation_sha256") != generation ||
      receipt.require("frame_sha256") != sha256(frame)) {
    throw Error("controller receipt transaction binding invalid");
  }
}

std::map<std::string, std::string> load_quorum_fixtures(
    const Manifest& fixture_manifest, const std::string& bundle_path) {
  const std::string contents = read_file(bundle_path);
  if (sha256(contents) != fixture_manifest.require("bundle_sha256")) {
    throw Error("Sounio fixture bundle hash mismatch");
  }
  std::map<std::string, std::string> fixtures;
  std::istringstream input(contents);
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("FIXTURE ", 0) != 0) continue;
    const std::size_t label_end = line.find(' ', 8);
    if (label_end == std::string::npos || label_end + 1 >= line.size()) {
      throw Error("Sounio fixture line malformed");
    }
    if (!fixtures.emplace(line.substr(8, label_end - 8), line.substr(label_end + 1))
             .second) {
      throw Error("Sounio fixture label duplicated");
    }
  }
  for (const std::string label : {"issue", "consume", "close", "current", "python"}) {
    if (!fixtures.contains(label)) throw Error("Sounio fixture is absent: " + label);
  }
  if (fixtures.size() != 5) throw Error("unexpected Sounio fixture count");
  return fixtures;
}

class ExecQuorumController {
 public:
  ExecQuorumController(std::string runtime, std::string root,
                       std::string resident_runtime, std::string generation)
      : runtime_(canonical_path(runtime)), root_(canonical_path(root)),
        resident_runtime_(canonical_path(resident_runtime)),
        generation_(std::move(generation)) {
    spawn();
  }

  ExecQuorumController(const ExecQuorumController&) = delete;
  ExecQuorumController& operator=(const ExecQuorumController&) = delete;
  ~ExecQuorumController() { terminate_noexcept(); }

  QuorumReceipt command(std::string_view operation, const std::string& frame) {
    const std::string line = raw_command(operation, frame);
    QuorumReceipt receipt = parse_quorum_receipt(line);
    validate_quorum_receipt(receipt, operation, generation_, frame);
    return receipt;
  }

  std::string raw_command(std::string_view operation, const std::string& frame) {
    authenticate();
    const auto deadline = std::chrono::steady_clock::now() + kExecQuorumDeadline;
    quorum_write_all(input_.get(), std::string(operation) + " " + frame + "\n",
                     deadline);
    return quorum_read_line(output_.get(), output_buffer_, kMaximumAuthorityOutput,
                            deadline, "controller response timeout");
  }

  bool authenticated() {
    try {
      authenticate();
      return true;
    } catch (...) {
      return false;
    }
  }

  pid_t pid() const { return pid_; }
  std::uint64_t start_tick() const { return start_tick_; }
  const std::string& generation() const { return generation_; }

  void kill_controller() {
    if (pid_ > 0) kill(pid_, SIGKILL);
  }

 private:
  void spawn() {
    int input_pipe[2];
    int output_pipe[2];
    if (pipe2(input_pipe, O_CLOEXEC) != 0 ||
        pipe2(output_pipe, O_CLOEXEC) != 0) {
      throw Error("cannot create controller pipes");
    }
    UniqueFd input_read(input_pipe[0]);
    UniqueFd input_write(input_pipe[1]);
    UniqueFd output_read(output_pipe[0]);
    UniqueFd output_write(output_pipe[1]);
    const pid_t parent = getpid();
    pid_ = fork();
    if (pid_ < 0) throw Error("cannot fork ExecGrantController");
    if (pid_ == 0) {
      input_write = UniqueFd();
      output_read = UniqueFd();
      if (dup2(input_read.get(), STDIN_FILENO) < 0 ||
          dup2(output_write.get(), STDOUT_FILENO) < 0 ||
          dup2(output_write.get(), STDERR_FILENO) < 0 ||
          prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) {
        _exit(126);
      }
      const std::string parent_text = std::to_string(parent);
      if (setenv("SOUNIO_LOOM_HOOK_TEST_MODE", "1", 1) != 0 ||
          setenv("SOUNIO_LOOM_EXEC_GRANT_CONTROLLER", "1", 1) != 0 ||
          setenv("SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_ROOT", root_.c_str(), 1) != 0 ||
          setenv("SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_PARENT_PID",
                 parent_text.c_str(), 1) != 0 ||
          setenv("SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_GENERATION",
                 generation_.c_str(), 1) != 0 ||
          setenv("SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_DEADLINE_MS", "15000", 1) != 0 ||
          setenv("SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_RUNTIME",
                 resident_runtime_.c_str(), 1) != 0) {
        _exit(126);
      }
      char* const arguments[] = {const_cast<char*>(runtime_.c_str()), nullptr};
      execv(runtime_.c_str(), arguments);
      _exit(127);
    }
    input_read = UniqueFd();
    output_write = UniqueFd();
    input_ = std::move(input_write);
    output_ = std::move(output_read);
    await_exec_identity(pid_, runtime_, start_tick_);
    pidfd_ = UniqueFd(open_pidfd(pid_));
  }

  void authenticate() {
    if (pid_ <= 0 || !pidfd_live(pidfd_.get()) ||
        process_start_tick(pid_) != start_tick_ ||
        process_executable(pid_) != runtime_) {
      throw Error("ExecGrantController identity drift");
    }
  }

  void terminate_noexcept() {
    if (pid_ <= 0) return;
    input_ = UniqueFd();
    int status = 0;
    pid_t waited = waitpid(pid_, &status, WNOHANG);
    if (waited == 0) {
      kill(pid_, SIGKILL);
      while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
      }
    }
    pid_ = -1;
  }

  std::string runtime_;
  std::string root_;
  std::string resident_runtime_;
  std::string generation_;
  UniqueFd input_;
  UniqueFd output_;
  UniqueFd pidfd_;
  std::string output_buffer_;
  pid_t pid_ = -1;
  std::uint64_t start_tick_ = 0;
};

class ExecQuorumPrincipalCell {
 public:
  ExecQuorumPrincipalCell(std::string runtime, std::string generation)
      : runtime_(canonical_path(runtime)), generation_(std::move(generation)) {
    spawn();
  }

  ExecQuorumPrincipalCell(const ExecQuorumPrincipalCell&) = delete;
  ExecQuorumPrincipalCell& operator=(const ExecQuorumPrincipalCell&) = delete;
  ~ExecQuorumPrincipalCell() { terminate_noexcept(); }

  bool authenticated() const {
    try {
      return pid_ > 0 && pidfd_live(pidfd_.get()) &&
             process_start_tick(pid_) == start_tick_ &&
             process_executable(pid_) == runtime_ &&
             sha256(process_cgroup(pid_)) == cgroup_sha256_;
    } catch (...) {
      return false;
    }
  }

  bool distinct_uid() const { return uid_ != getuid(); }
  pid_t pid() const { return pid_; }

  BarrierObservation close_without_write() { return finish(""); }
  BarrierObservation release_exact() { return finish(generation_ + "\n"); }
  BarrierObservation release_payload(const std::string& payload) {
    return finish(payload);
  }

 private:
  static void duplicate_for_exec(int source, int target) {
    const int duplicate = fcntl(source, F_DUPFD_CLOEXEC, 10);
    if (duplicate < 0 || dup2(duplicate, target) < 0) _exit(126);
    close(duplicate);
  }

  void spawn() {
    int release_pipe[2];
    int result_pipe[2];
    if (pipe2(release_pipe, O_CLOEXEC) != 0 ||
        pipe2(result_pipe, O_CLOEXEC) != 0) {
      throw Error("cannot create integrated PrincipalCell pipes");
    }
    UniqueFd release_read(release_pipe[0]);
    UniqueFd release_write(release_pipe[1]);
    UniqueFd result_read(result_pipe[0]);
    UniqueFd result_write(result_pipe[1]);
    const pid_t parent = getpid();
    pid_ = fork();
    if (pid_ < 0) throw Error("cannot fork integrated PrincipalCell");
    if (pid_ == 0) {
      duplicate_for_exec(release_read.get(), 3);
      duplicate_for_exec(result_write.get(), 4);
      if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(126);
      const std::string parent_text = std::to_string(parent);
      if (setenv("SOUNIO_LOOM_PRINCIPAL_CELL_INTERNAL", "1", 1) != 0 ||
          setenv("SOUNIO_LOOM_PRINCIPAL_CELL_PARENT_PID", parent_text.c_str(), 1) !=
              0 ||
          setenv("SOUNIO_LOOM_PRINCIPAL_CELL_GENERATION", generation_.c_str(), 1) !=
              0) {
        _exit(126);
      }
      char* const arguments[] = {const_cast<char*>(runtime_.c_str()),
                                 const_cast<char*>("--internal-principal-cell"),
                                 nullptr};
      execv(runtime_.c_str(), arguments);
      _exit(127);
    }
    release_read = UniqueFd();
    result_write = UniqueFd();
    release_ = std::move(release_write);
    result_ = std::move(result_read);
    await_exec_identity(pid_, runtime_, start_tick_);
    pidfd_ = UniqueFd(open_pidfd(pid_));
    uid_ = process_uid(pid_);
    cgroup_sha256_ = sha256(process_cgroup(pid_));
  }

  BarrierObservation finish(const std::string& payload) {
    if (finished_) throw Error("PrincipalCell barrier already consumed");
    if (!payload.empty()) {
      const auto deadline = std::chrono::steady_clock::now() + kExecQuorumDeadline;
      quorum_write_all(release_.get(), payload, deadline);
    }
    release_ = UniqueFd();
    const auto deadline = std::chrono::steady_clock::now() + kExecQuorumDeadline;
    const std::string line = quorum_read_line(result_.get(), result_buffer_, 256,
                                              deadline, "PrincipalCell result timeout");
    result_ = UniqueFd();
    int status = 0;
    while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
    }
    pid_ = -1;
    finished_ = true;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      throw Error("integrated PrincipalCell failed");
    }
    constexpr std::string_view opened = "BARRIER_OPENED reason=";
    constexpr std::string_view closed = "BARRIER_CLOSED reason=";
    if (line.rfind(opened, 0) == 0) return {true, line.substr(opened.size())};
    if (line.rfind(closed, 0) == 0) return {false, line.substr(closed.size())};
    throw Error("integrated PrincipalCell result malformed");
  }

  void terminate_noexcept() {
    if (pid_ <= 0) return;
    release_ = UniqueFd();
    result_ = UniqueFd();
    kill(pid_, SIGKILL);
    while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {
    }
    pid_ = -1;
  }

  std::string runtime_;
  std::string generation_;
  std::string cgroup_sha256_;
  UniqueFd release_;
  UniqueFd result_;
  UniqueFd pidfd_;
  std::string result_buffer_;
  pid_t pid_ = -1;
  std::uint64_t start_tick_ = 0;
  uid_t uid_ = 0;
  bool finished_ = false;
};

class HostExecQuorumPrincipalCell {
 public:
  HostExecQuorumPrincipalCell(std::string runtime, std::string systemd_run,
                              std::string systemctl, std::string generation)
      : runtime_(canonical_path(runtime)),
        systemd_run_(canonical_path(systemd_run)),
        systemctl_(canonical_path(systemctl)),
        generation_(std::move(generation)),
        unit_("sounio-loom-host-exec-quorum-" + std::to_string(getpid()) + "-" +
              random_generation_sha256().substr(0, 16) + ".service") {
    if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0) {
      throw Error("host ExecQuorum requires root broker identity");
    }
    if (file_sha256(runtime_) != kHostExecQuorumPrincipalCellRuntimeSha256) {
      throw Error("host PrincipalCell runtime hash mismatch");
    }
    struct stat runtime_info {};
    if (stat(runtime_.c_str(), &runtime_info) != 0 ||
        !S_ISREG(runtime_info.st_mode) || runtime_info.st_uid != 0 ||
        runtime_info.st_gid != 0 || (runtime_info.st_mode & 0222) != 0 ||
        (runtime_info.st_mode & 0111) == 0) {
      throw Error("host PrincipalCell runtime metadata is unsafe");
    }
    spawn();
  }

  HostExecQuorumPrincipalCell(const HostExecQuorumPrincipalCell&) = delete;
  HostExecQuorumPrincipalCell& operator=(const HostExecQuorumPrincipalCell&) =
      delete;
  ~HostExecQuorumPrincipalCell() { terminate_noexcept(); }

  bool authenticated() const {
    try {
      return pid_ > 1 && pidfd_live(pidfd_.get()) &&
             process_start_tick(pid_) == start_tick_ &&
             process_executable(pid_) == runtime_ && process_uid(pid_) == uid_ &&
             process_gid(pid_) == gid_ && uid_ != 0 && gid_ != 0 &&
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

  bool distinct_uid() const { return uid_ != getuid() && gid_ != getgid(); }
  pid_t pid() const { return pid_; }
  uid_t uid() const { return uid_; }
  gid_t gid() const { return gid_; }
  const std::string& unit() const { return unit_; }
  const std::string& systemd_run() const { return systemd_run_; }
  const std::string& systemctl() const { return systemctl_; }

  BarrierObservation close_without_write() { return finish(""); }
  BarrierObservation release_exact() { return finish(generation_ + "\n"); }
  BarrierObservation release_payload(const std::string& payload) {
    return finish(payload);
  }

 private:
  std::string property(const std::string& name) const {
    return quorum_tool_value(systemctl_,
                             {"show", unit_, "--property=" + name, "--value"});
  }

  void spawn() {
    int release_pipe[2];
    int result_pipe[2];
    if (pipe2(release_pipe, O_CLOEXEC) != 0 ||
        pipe2(result_pipe, O_CLOEXEC) != 0) {
      throw Error("cannot create host PrincipalCell descriptor barrier");
    }
    UniqueFd release_read(release_pipe[0]);
    UniqueFd release_write(release_pipe[1]);
    UniqueFd result_read(result_pipe[0]);
    UniqueFd result_write(result_pipe[1]);
    UniqueFd null_error(open("/dev/null", O_WRONLY | O_CLOEXEC));
    if (null_error.get() < 0) throw Error("cannot open host PrincipalCell stderr");
    client_pid_ = fork();
    if (client_pid_ < 0) throw Error("cannot fork systemd transient-unit client");
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
      const std::vector<std::string> arguments = {
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
          "--setenv=SOUNIO_LOOM_HOST_EXEC_QUORUM_INTERNAL=1",
          "--setenv=SOUNIO_LOOM_HOST_EXEC_QUORUM_UNIT=" + unit_,
          "--setenv=SOUNIO_LOOM_HOST_EXEC_QUORUM_GENERATION=" + generation_,
          "--",
          runtime_,
          "--internal-host-exec-quorum"};
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
      throw Error("systemd transient-unit client wait failed");
    }
    if (waited == client_pid_) {
      client_reaped_ = true;
      client_status_ = status;
      if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        throw Error("systemd transient-unit client refused");
      }
    }
  }

  void await_service_identity() {
    const auto deadline = std::chrono::steady_clock::now() + kExecQuorumDeadline;
    for (;;) {
      observe_client();
      try {
        const std::string main_pid = property("MainPID");
        const auto parsed = parse_u64(main_pid);
        if (parsed && *parsed > 1 &&
            *parsed <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
          pid_ = static_cast<pid_t>(*parsed);
          start_tick_ = process_start_tick(pid_);
          if (process_executable(pid_) != runtime_) {
            throw Error("host PrincipalCell executable mismatch");
          }
          uid_ = process_uid(pid_);
          gid_ = process_gid(pid_);
          cgroup_sha256_ = sha256(process_cgroup(pid_));
          pidfd_ = UniqueFd(open_pidfd(pid_));
          if (!authenticated() || !distinct_uid()) {
            throw Error("host PrincipalCell identity admission failed");
          }
          return;
        }
      } catch (const Error&) {
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        throw Error("host PrincipalCell identity admission timed out");
      }
      poll(nullptr, 0, 10);
    }
  }

  void await_client_exit() {
    const auto deadline = std::chrono::steady_clock::now() + kExecQuorumDeadline;
    while (!client_reaped_) {
      observe_client();
      if (client_reaped_) break;
      if (std::chrono::steady_clock::now() >= deadline) {
        throw Error("systemd transient-unit client exit timed out");
      }
      poll(nullptr, 0, 10);
    }
    if (!WIFEXITED(client_status_) || WEXITSTATUS(client_status_) != 0) {
      throw Error("systemd transient-unit client exit failed");
    }
    client_pid_ = -1;
  }

  BarrierObservation finish(const std::string& payload) {
    if (finished_) throw Error("host PrincipalCell barrier already consumed");
    if (!authenticated()) throw Error("host PrincipalCell identity drifted");
    const auto deadline = std::chrono::steady_clock::now() + kExecQuorumDeadline;
    quorum_write_all(release_.get(), "ARM\n" + payload, deadline);
    release_ = UniqueFd();
    const std::string line = quorum_read_line(
        result_.get(), result_buffer_, 256, deadline,
        "host PrincipalCell result timeout");
    result_ = UniqueFd();
    await_client_exit();
    pid_ = -1;
    finished_ = true;
    constexpr std::string_view opened = "BARRIER_OPENED reason=";
    constexpr std::string_view closed = "BARRIER_CLOSED reason=";
    if (line.rfind(opened, 0) == 0) return {true, line.substr(opened.size())};
    if (line.rfind(closed, 0) == 0) return {false, line.substr(closed.size())};
    throw Error("host PrincipalCell result malformed");
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
  std::string systemd_run_;
  std::string systemctl_;
  std::string generation_;
  std::string unit_;
  std::string cgroup_sha256_;
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
  bool finished_ = false;
};

struct ExecQuorumFacts {
  bool semantic_permit = false;
  bool linear_consumption = false;
  bool controller_authenticated = false;
  bool generation_match = false;
  bool principal_authenticated = false;
  bool principal_distinct_uid = false;

  bool complete() const {
    return semantic_permit && linear_consumption && controller_authenticated &&
           generation_match && principal_authenticated && principal_distinct_uid;
  }
};

template <typename PrincipalCell>
BarrierObservation apply_exec_quorum(const ExecQuorumFacts& facts,
                                     PrincipalCell& principal) {
  if (facts.complete()) return principal.release_exact();
  return principal.close_without_write();
}

void require_barrier(const BarrierObservation& observation, bool opened,
                     std::string_view reason, std::string_view label) {
  if (observation.opened != opened || observation.reason != reason) {
    throw Error("integrated barrier case diverged: " + std::string(label) +
                ":opened=" + (observation.opened ? "true" : "false") +
                ":reason=" + observation.reason);
  }
}

int selftest_exec_quorum(const std::string& root,
                         const std::string& controller_manifest_path,
                         const std::string& controller_runtime_path,
                         const std::string& fixture_manifest_path,
                         const std::string& fixture_bundle_path,
                         const std::string& resident_runtime_path,
                         const std::string& barrier_runtime_path) {
  if (getuid() == 0 || geteuid() == 0) {
    throw Error("integrated quorum selftest refuses root execution");
  }
  const Manifest controller_manifest = parse_frozen_manifest(
      controller_manifest_path, kFrozenExecGrantControllerManifestSha256,
      "ExecGrantController");
  if (controller_manifest.require("schema") !=
          "loom-exec-grant-controller-runtime-v1" ||
      controller_manifest.require("stage") != "EFFECT_PARITY_FROZEN" ||
      controller_manifest.require("producing_language") != "OCaml" ||
      controller_manifest.require("language_role") != "EFFECT_PARITY" ||
      controller_manifest.require("semantic_authority") != "Sounio" ||
      controller_manifest.require("action") != "9030" ||
      controller_manifest.require("non_bearer_transport") != "pending" ||
      controller_manifest.require("material_grant") != "false" ||
      controller_manifest.require("barrier_release") != "false") {
    throw Error("controller manifest state invalid");
  }
  if (file_sha256(controller_runtime_path) !=
      controller_manifest.require("runtime_sha256")) {
    throw Error("controller runtime hash mismatch");
  }
  if (file_sha256(barrier_runtime_path) !=
      kIntegratedPrincipalCellRuntimeSha256) {
    throw Error("integrated PrincipalCell runtime hash mismatch");
  }
  const Manifest fixture_manifest = parse_frozen_manifest(
      fixture_manifest_path, kFrozenExecQuorumFixtureManifestSha256,
      "Sounio ExecQuorum fixtures");
  if (fixture_manifest.require("stage") != "SEMANTICS_FROZEN" ||
      fixture_manifest.require("producing_language") != "Sounio" ||
      fixture_manifest.require("semantic_authority") != "Sounio" ||
      fixture_manifest.require("action") != "9030") {
    throw Error("Sounio fixture manifest state invalid");
  }
  const auto fixtures = load_quorum_fixtures(fixture_manifest, fixture_bundle_path);

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    const QuorumReceipt treatment = controller.command("ISSUE", fixtures.at("current"));
    ExecQuorumPrincipalCell principal(barrier_runtime_path, generation);
    ExecQuorumFacts facts;
    facts.controller_authenticated = controller.authenticated();
    facts.generation_match = treatment.require("controller_generation_sha256") == generation;
    facts.principal_authenticated = principal.authenticated();
    facts.principal_distinct_uid = principal.distinct_uid();
    require_barrier(apply_exec_quorum(facts, principal), false, "eof", "treatment");
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    const QuorumReceipt issue = controller.command("ISSUE", fixtures.at("issue"));
    const QuorumReceipt consume =
        controller.command("CONSUME", fixtures.at("consume"));
    ExecQuorumPrincipalCell principal(barrier_runtime_path, generation);
    ExecQuorumFacts facts;
    facts.semantic_permit = consume.require("quorum_ready") == "true";
    facts.linear_consumption = consume.require("state") == "OUTCOME_PENDING";
    facts.controller_authenticated = controller.authenticated();
    facts.generation_match =
        issue.require("controller_generation_sha256") == generation &&
        consume.require("controller_generation_sha256") == generation;
    facts.principal_authenticated = principal.authenticated();
    facts.principal_distinct_uid = principal.distinct_uid();
    if (!facts.semantic_permit || !facts.linear_consumption ||
        !facts.controller_authenticated || !facts.generation_match ||
        !facts.principal_authenticated || facts.principal_distinct_uid) {
      throw Error("positive local quorum facts diverged");
    }
    require_barrier(apply_exec_quorum(facts, principal), false, "eof",
                    "same-uid-positive-control");
    static_cast<void>(controller.command("CLOSE", fixtures.at("close")));
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumPrincipalCell principal(barrier_runtime_path, generation);
    require_barrier(principal.release_exact(), true, "exact-release",
                    "exact-write-sabotage");
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    static_cast<void>(controller.command("ISSUE", fixtures.at("issue")));
    const std::string replay = controller.raw_command("ISSUE", fixtures.at("issue"));
    if (replay.find("controller-protocol-state-mismatch") == std::string::npos) {
      throw Error("controller replay was not refused");
    }
    ExecQuorumPrincipalCell principal(barrier_runtime_path, generation);
    require_barrier(principal.close_without_write(), false, "eof", "replay");
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    static_cast<void>(controller.command("ISSUE", fixtures.at("issue")));
    ExecQuorumPrincipalCell principal(barrier_runtime_path, generation);
    controller.kill_controller();
    require_barrier(principal.close_without_write(), false, "eof",
                    "controller-death");
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    const QuorumReceipt issue = controller.command("ISSUE", fixtures.at("issue"));
    const auto resident_pid = parse_u64(issue.require("resident_pid"));
    if (!resident_pid || *resident_pid > static_cast<std::uint64_t>(INT_MAX) ||
        kill(static_cast<pid_t>(*resident_pid), SIGKILL) != 0) {
      throw Error("cannot sabotage controller resident");
    }
    const std::string refusal =
        controller.raw_command("CONSUME", fixtures.at("consume"));
    if (refusal.find("REFUSE") == std::string::npos) {
      throw Error("resident death did not refuse consumption");
    }
    ExecQuorumPrincipalCell principal(barrier_runtime_path, generation);
    require_barrier(principal.close_without_write(), false, "eof", "resident-death");
  }

  {
    const std::string controller_generation = random_generation_sha256();
    const std::string principal_generation = random_generation_sha256();
    ExecQuorumPrincipalCell principal(barrier_runtime_path, principal_generation);
    require_barrier(principal.release_payload(controller_generation + "\n"), false,
                    "generation-mismatch", "wrong-generation");
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    const QuorumReceipt python = controller.command("ISSUE", fixtures.at("python"));
    ExecQuorumPrincipalCell principal(barrier_runtime_path, generation);
    ExecQuorumFacts facts;
    facts.controller_authenticated = controller.authenticated();
    facts.generation_match = python.require("controller_generation_sha256") == generation;
    facts.principal_authenticated = principal.authenticated();
    facts.principal_distinct_uid = principal.distinct_uid();
    require_barrier(apply_exec_quorum(facts, principal), false, "eof",
                    "python-authority-laundering");
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumPrincipalCell principal(barrier_runtime_path, generation);
    ExecQuorumFacts fake;
    fake.semantic_permit = true;
    fake.linear_consumption = true;
    fake.generation_match = true;
    fake.principal_authenticated = principal.authenticated();
    fake.principal_distinct_uid = principal.distinct_uid();
    require_barrier(apply_exec_quorum(fake, principal), false, "eof",
                    "textual-receipt-laundering");
  }

  std::cout
      << "LOOM_HOST_EXEC_QUORUM_SELFTEST PASS semantic_authority=Sounio"
      << " controller=OCaml role=EFFECT_PARITY material_layer=C++20+Linux"
      << " material_role=MATERIAL_PARITY transitory=true"
      << " single_resident_controller=true inherited_descriptor=true"
      << " treatment=closed positive_semantics=ready positive_local=closed"
      << " positive_local_reason=same-uid-principal"
      << " exact_write_sabotage=open replay=closed controller_death=closed"
      << " resident_death=closed wrong_generation=closed python=closed"
      << " textual_receipt=closed principal_pidfd=bound"
      << " principal_start_tick=bound principal_executable=bound"
      << " principal_cgroup=bound principal_distinct_uid=false"
      << " non_bearer_transport=measured same_uid_peer_isolation=false"
      << " material_threshold_measured=true descriptor_barrier_causal=true"
      << " material_grant=false material_execution=false barrier_release=false"
      << " launch_open=false exec_attached=false parity_open=false"
      << " claim_ready=false\n";
  return 0;
}

int selftest_host_exec_quorum(
    const std::string& root, const std::string& controller_manifest_path,
    const std::string& controller_runtime_path,
    const std::string& fixture_manifest_path,
    const std::string& fixture_bundle_path,
    const std::string& resident_runtime_path,
    const std::string& local_barrier_runtime_path,
    const std::string& host_barrier_runtime_path,
    const std::string& systemd_run_path, const std::string& systemctl_path) {
  if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0 ||
      trim(read_file("/proc/1/comm")) != "systemd") {
    throw Error("host ExecQuorum selftest requires root under PID 1 systemd");
  }
  const Manifest controller_manifest = parse_frozen_manifest(
      controller_manifest_path, kFrozenExecGrantControllerManifestSha256,
      "ExecGrantController");
  if (controller_manifest.require("schema") !=
          "loom-exec-grant-controller-runtime-v1" ||
      controller_manifest.require("stage") != "EFFECT_PARITY_FROZEN" ||
      controller_manifest.require("producing_language") != "OCaml" ||
      controller_manifest.require("language_role") != "EFFECT_PARITY" ||
      controller_manifest.require("semantic_authority") != "Sounio" ||
      controller_manifest.require("action") != "9030" ||
      controller_manifest.require("non_bearer_transport") != "pending" ||
      controller_manifest.require("material_grant") != "false" ||
      controller_manifest.require("barrier_release") != "false") {
    throw Error("host controller manifest state invalid");
  }
  if (file_sha256(controller_runtime_path) !=
      controller_manifest.require("runtime_sha256")) {
    throw Error("host controller runtime hash mismatch");
  }
  if (file_sha256(local_barrier_runtime_path) !=
          kIntegratedPrincipalCellRuntimeSha256 ||
      file_sha256(host_barrier_runtime_path) !=
          kHostExecQuorumPrincipalCellRuntimeSha256) {
    throw Error("host ExecQuorum barrier runtime hash mismatch");
  }
  const Manifest fixture_manifest = parse_frozen_manifest(
      fixture_manifest_path, kFrozenExecQuorumFixtureManifestSha256,
      "Sounio ExecQuorum fixtures");
  if (fixture_manifest.require("stage") != "SEMANTICS_FROZEN" ||
      fixture_manifest.require("producing_language") != "Sounio" ||
      fixture_manifest.require("semantic_authority") != "Sounio" ||
      fixture_manifest.require("action") != "9030") {
    throw Error("host Sounio fixture manifest state invalid");
  }
  const auto fixtures =
      load_quorum_fixtures(fixture_manifest, fixture_bundle_path);

  std::size_t positive_open_sentinels = 0;
  std::size_t sabotage_open_sentinels = 0;
  uid_t positive_uid = 0;
  gid_t positive_gid = 0;

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    const QuorumReceipt treatment =
        controller.command("ISSUE", fixtures.at("current"));
    HostExecQuorumPrincipalCell principal(
        host_barrier_runtime_path, systemd_run_path, systemctl_path, generation);
    ExecQuorumFacts facts;
    facts.controller_authenticated = controller.authenticated();
    facts.generation_match =
        treatment.require("controller_generation_sha256") == generation;
    facts.principal_authenticated = principal.authenticated();
    facts.principal_distinct_uid = principal.distinct_uid();
    require_barrier(apply_exec_quorum(facts, principal), false, "eof",
                    "host-treatment");
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    const QuorumReceipt issue =
        controller.command("ISSUE", fixtures.at("issue"));
    const QuorumReceipt consume =
        controller.command("CONSUME", fixtures.at("consume"));
    HostExecQuorumPrincipalCell principal(
        host_barrier_runtime_path, systemd_run_path, systemctl_path, generation);
    ExecQuorumFacts facts;
    facts.semantic_permit = consume.require("quorum_ready") == "true";
    facts.linear_consumption = consume.require("state") == "OUTCOME_PENDING";
    facts.controller_authenticated = controller.authenticated();
    facts.generation_match =
        issue.require("controller_generation_sha256") == generation &&
        consume.require("controller_generation_sha256") == generation;
    facts.principal_authenticated = principal.authenticated();
    facts.principal_distinct_uid = principal.distinct_uid();
    if (!facts.complete()) throw Error("positive host quorum facts diverged");
    positive_uid = principal.uid();
    positive_gid = principal.gid();
    const BarrierObservation observation = apply_exec_quorum(facts, principal);
    require_barrier(observation, true, "exact-release", "host-positive");
    positive_open_sentinels += observation.opened ? 1 : 0;
    static_cast<void>(controller.command("CLOSE", fixtures.at("close")));
  }

  {
    const std::string generation = random_generation_sha256();
    HostExecQuorumPrincipalCell principal(
        host_barrier_runtime_path, systemd_run_path, systemctl_path, generation);
    const BarrierObservation observation = principal.release_exact();
    require_barrier(observation, true, "exact-release", "host-exact-write-sabotage");
    sabotage_open_sentinels += observation.opened ? 1 : 0;
    bool second_release_refused = false;
    try {
      static_cast<void>(principal.release_exact());
    } catch (const Error&) {
      second_release_refused = true;
    }
    if (!second_release_refused) {
      throw Error("host PrincipalCell admitted a second release");
    }
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    static_cast<void>(controller.command("ISSUE", fixtures.at("issue")));
    const std::string replay =
        controller.raw_command("ISSUE", fixtures.at("issue"));
    if (replay.find("controller-protocol-state-mismatch") == std::string::npos) {
      throw Error("host controller replay was not refused");
    }
    HostExecQuorumPrincipalCell principal(
        host_barrier_runtime_path, systemd_run_path, systemctl_path, generation);
    require_barrier(principal.close_without_write(), false, "eof", "host-replay");
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    static_cast<void>(controller.command("ISSUE", fixtures.at("issue")));
    controller.kill_controller();
    HostExecQuorumPrincipalCell principal(
        host_barrier_runtime_path, systemd_run_path, systemctl_path, generation);
    require_barrier(principal.close_without_write(), false, "eof",
                    "host-controller-death");
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    const QuorumReceipt issue =
        controller.command("ISSUE", fixtures.at("issue"));
    const auto resident_pid = parse_u64(issue.require("resident_pid"));
    if (!resident_pid ||
        *resident_pid > static_cast<std::uint64_t>(INT_MAX) ||
        kill(static_cast<pid_t>(*resident_pid), SIGKILL) != 0) {
      throw Error("cannot sabotage host controller resident");
    }
    const std::string refusal =
        controller.raw_command("CONSUME", fixtures.at("consume"));
    if (refusal.find("REFUSE") == std::string::npos) {
      throw Error("host resident death did not refuse consumption");
    }
    HostExecQuorumPrincipalCell principal(
        host_barrier_runtime_path, systemd_run_path, systemctl_path, generation);
    require_barrier(principal.close_without_write(), false, "eof",
                    "host-resident-death");
  }

  {
    const std::string controller_generation = random_generation_sha256();
    const std::string principal_generation = random_generation_sha256();
    HostExecQuorumPrincipalCell principal(host_barrier_runtime_path,
                                          systemd_run_path, systemctl_path,
                                          principal_generation);
    require_barrier(principal.release_payload(controller_generation + "\n"), false,
                    "generation-mismatch", "host-wrong-generation");
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, generation);
    const QuorumReceipt python =
        controller.command("ISSUE", fixtures.at("python"));
    HostExecQuorumPrincipalCell principal(
        host_barrier_runtime_path, systemd_run_path, systemctl_path, generation);
    ExecQuorumFacts facts;
    facts.controller_authenticated = controller.authenticated();
    facts.generation_match =
        python.require("controller_generation_sha256") == generation;
    facts.principal_authenticated = principal.authenticated();
    facts.principal_distinct_uid = principal.distinct_uid();
    require_barrier(apply_exec_quorum(facts, principal), false, "eof",
                    "host-python-authority-laundering");
  }

  {
    const std::string generation = random_generation_sha256();
    HostExecQuorumPrincipalCell principal(
        host_barrier_runtime_path, systemd_run_path, systemctl_path, generation);
    ExecQuorumFacts fake;
    fake.semantic_permit = true;
    fake.linear_consumption = true;
    fake.generation_match = true;
    fake.principal_authenticated = principal.authenticated();
    fake.principal_distinct_uid = principal.distinct_uid();
    require_barrier(apply_exec_quorum(fake, principal), false, "eof",
                    "host-textual-receipt-laundering");
  }

  {
    const std::string generation = random_generation_sha256();
    ExecQuorumPrincipalCell principal(local_barrier_runtime_path, generation);
    ExecQuorumFacts facts;
    facts.semantic_permit = true;
    facts.linear_consumption = true;
    facts.controller_authenticated = true;
    facts.generation_match = true;
    facts.principal_authenticated = principal.authenticated();
    facts.principal_distinct_uid = principal.distinct_uid();
    if (!facts.principal_authenticated || facts.principal_distinct_uid) {
      throw Error("same-UID host sabotage control posture diverged");
    }
    require_barrier(apply_exec_quorum(facts, principal), false, "eof",
                    "host-same-uid-control");
  }

  if (positive_open_sentinels != 1 || sabotage_open_sentinels != 1) {
    throw Error("host open sentinel cardinality diverged");
  }

  std::cout
      << "LOOM_HOST_EXEC_QUORUM_HOST_GATE PASS semantic_authority=Sounio"
      << " controller=OCaml controller_role=EFFECT_PARITY"
      << " material_layer=C++20+Linux+systemd"
      << " material_role=MATERIAL_PARITY transitory=true"
      << " systemd_transport_authority=false dynamic_user=true"
      << " inherited_descriptor=true arm_authority=false"
      << " single_resident_controller=true treatment=closed"
      << " positive_host=open positive_open_sentinels="
      << positive_open_sentinels
      << " exact_write_sabotage=open sabotage_open_sentinels="
      << sabotage_open_sentinels
      << " total_open_sentinels="
      << positive_open_sentinels + sabotage_open_sentinels
      << " second_release=refused replay=closed controller_death=closed"
      << " resident_death=closed wrong_generation=closed python=closed"
      << " textual_receipt=closed same_uid=closed"
      << " causal_rule=three-object-quorum causal_sabotage=PASS"
      << " principal_pidfd=bound principal_start_tick=bound"
      << " principal_executable=bound principal_cgroup=bound"
      << " principal_distinct_uid=true positive_uid=" << positive_uid
      << " positive_gid=" << positive_gid
      << " non_bearer_exec_quorum=true descriptor_barrier_causal=true"
      << " linear_grant_consumption=true material_grant=true"
      << " material_execution=false launch_open=false recycle_open=false"
      << " exec_attached=false commit_attached=false ci_attached=false"
      << " parity_open=false claim_ready=false"
      << " systemd_run_sha256=" << file_sha256(systemd_run_path)
      << " systemctl_sha256=" << file_sha256(systemctl_path) << "\n";
  return 0;
}
