constexpr std::string_view kFrozenCausalRunGrantManifestSha256 =
    "24db7036ae2b01141dcb43e325ace6f11e5e5fa1e94ba8a931d5ce9194f93a80";
constexpr std::string_view kFrozenCausalAttestGrantManifestSha256 =
    "60050ea0310dd3b48dfe83d0d9d16d11adaf017a36a5efcfe6647fbe8e201bbd";
constexpr std::string_view kFrozenCausalWorkflowManifestSha256 =
    "ef8dc0fadeb1cb33b1ba31d77551b8ca962bc97d1204998057b071b9b7921437";
constexpr auto kCausalMaterialDeadline = std::chrono::seconds(30);

struct CausalCellGrantFixtures {
  std::string command;
  std::string command_sha256;
  std::string intent_sha256;
  std::string event_sha256;
  std::string artifact_sha256;
  std::string command_mismatch_code;
  std::string parent_bindings;
  std::map<std::string, std::string> frame_prefixes;
  std::map<std::string, std::string> frames;
};

CausalCellGrantFixtures load_causal_cell_grant_fixtures(
    const std::string& manifest_path, const std::string& bundle_path,
    std::string_view expected_manifest_sha256, std::string_view expected_schema,
    std::string_view expected_header, std::size_t expected_lines,
    bool require_artifact) {
  const Manifest manifest = parse_frozen_manifest(
      manifest_path, expected_manifest_sha256, "causal material grant");
  if (manifest.require("schema") != expected_schema ||
      manifest.require("stage") != "SEMANTICS_FROZEN" ||
      manifest.require("producing_language") != "Sounio" ||
      manifest.require("language_role") != "SEMANTIC_FIXTURE_PRODUCER" ||
      manifest.require("semantic_authority") != "Sounio" ||
      manifest.require("launch_action") != "9030" ||
      manifest.require("workflow_action") != "9037" ||
      manifest.require("command_mismatch_result") != "DENY492" ||
      manifest.require("causal_sabotage") != "PASS" ||
      manifest.require("run_ticket_is_bearer") != "false" ||
      manifest.require("run_ticket_is_execution_authority") != "false" ||
      manifest.require("launch_authority") != "action-9030" ||
      manifest.require("material_grant") != "false" ||
      manifest.require("material_execution") != "false" ||
      manifest.require("host_launch_attached") != "false" ||
      manifest.require("production_activation") != "false") {
    throw Error("causal material grant manifest posture invalid");
  }
  const std::string contents = read_file(bundle_path);
  if (sha256(contents) != manifest.require("bundle_sha256")) {
    throw Error("causal material grant bundle hash mismatch");
  }
  std::vector<std::string> lines;
  std::istringstream input(contents);
  std::string line;
  while (std::getline(input, line)) lines.push_back(line);
  if (lines.size() != expected_lines || lines.front() != expected_header) {
    throw Error("causal material grant bundle shape invalid");
  }
  CausalCellGrantFixtures fixtures;
  fixtures.command_mismatch_code = "492";
  const auto one = [&](std::string_view prefix) {
    std::string found;
    for (const std::string& candidate : lines) {
      if (candidate.rfind(prefix, 0) == 0) {
        if (!found.empty()) throw Error("causal grant bundle field duplicated");
        found = candidate.substr(prefix.size());
      }
    }
    if (found.empty()) throw Error("causal grant bundle field absent");
    return found;
  };
  fixtures.command = one("COMMAND ");
  fixtures.command_sha256 = one("COMMAND_SHA256 ");
  fixtures.intent_sha256 = one("INTENT_SHA256 ");
  fixtures.event_sha256 = one("EVENT_SHA256 ");
  fixtures.artifact_sha256 =
      require_artifact ? one("ARTIFACT_SHA256 ") : kFrozenOperationArtifactSha256;
  if (fixtures.command != manifest.require("command") ||
      fixtures.command_sha256 != manifest.require("command_sha256") ||
      fixtures.intent_sha256 != manifest.require("intent_sha256") ||
      fixtures.event_sha256 != manifest.require("event_sha256") ||
      sha256(fixtures.command) != fixtures.command_sha256 ||
      !lowercase_digest(fixtures.intent_sha256) ||
      !lowercase_digest(fixtures.event_sha256) ||
      !lowercase_digest(fixtures.artifact_sha256)) {
    throw Error("causal material grant lineage invalid");
  }
  fixtures.parent_bindings = one("PARENT_BINDINGS ");
  std::string encoded_bindings;
  for (const std::string key : {"grant_identity", "command_environment",
                                "peer_vector", "transition_journal",
                                "source_semantics_toolchain",
                                "result_receipt"}) {
    encoded_bindings += product_hex_digest_u32_limbs(one("BINDING " + key + " "));
  }
  for (const std::string label : {"issue", "consume", "close",
                                  "command_mismatch"}) {
    const std::string prefix = one("FRAME " + label + " ");
    fixtures.frame_prefixes.emplace(label, prefix);
    fixtures.frames.emplace(label, prefix + " " + fixtures.parent_bindings +
                                        encoded_bindings);
  }
  return fixtures;
}

std::string causal_live_grant_frame(
    const CausalCellGrantFixtures& fixtures, const std::string& label,
    const std::map<std::string, std::string>& bindings) {
  std::string frame = fixtures.frame_prefixes.at(label) + " " +
                      fixtures.parent_bindings;
  for (const std::string key : {"grant_identity", "command_environment",
                                "peer_vector", "transition_journal",
                                "source_semantics_toolchain",
                                "result_receipt"}) {
    const auto found = bindings.find(key);
    if (found == bindings.end()) {
      throw Error("causal live grant binding absent: " + key);
    }
    frame += product_hex_digest_u32_limbs(found->second);
  }
  return frame;
}

struct CausalMaterialObservation {
  std::string mode;
  std::string record;
  std::string record_sha256;
  std::string handle_type;
  std::string handle;
  std::string handle_sha256;
  std::map<std::string, std::string> fields;
  uid_t uid = 0;
  gid_t gid = 0;
  pid_t pid = 0;
  std::uint64_t start_tick = 0;
  std::string unit;
  std::string principal_sha256;
  bool dynamic_user = false;
  bool identity_authenticated = false;
  bool close_bound = false;
  bool pidfd_extinct = false;
  bool process_extinct = false;
  bool cgroup_unpopulated = false;
  bool unit_inactive = false;

  bool success_complete() const {
    return dynamic_user && identity_authenticated && close_bound &&
           pidfd_extinct && process_extinct && cgroup_unpopulated &&
           unit_inactive;
  }
};

class HostCausalMaterialCell {
 public:
  HostCausalMaterialCell(std::string runtime, std::string systemd_run,
                         std::string systemctl, std::string mode,
                         std::map<std::string, std::string> open_files)
      : runtime_(canonical_path(runtime)),
        systemd_run_(canonical_path(systemd_run)),
        systemctl_(canonical_path(systemctl)), mode_(std::move(mode)),
        open_files_(std::move(open_files)),
        unit_("sounio-loom-causal-" + std::to_string(getpid()) + "-" +
              random_generation_sha256().substr(0, 16) + ".service") {
    if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0) {
      throw Error("causal material host cell requires root Guardian");
    }
    if (mode_ != "RUN_EXACT" && mode_ != "ATTEST") {
      throw Error("causal material host cell mode invalid");
    }
    if (!process_witness_root_artifact(runtime_, true)) {
      throw Error("causal material cell runtime metadata unsafe");
    }
    for (auto& [name, path] : open_files_) {
      if (name.empty()) {
        throw Error("causal material descriptor name invalid");
      }
      for (const unsigned char character : name) {
        if (!(character >= 'a' && character <= 'z') &&
            !(character >= '0' && character <= '9') && character != '_') {
          throw Error("causal material descriptor name invalid");
        }
      }
      path = canonical_path(path);
      UniqueFd reference(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
      struct stat info {};
      if (reference.get() < 0 || fstat(reference.get(), &info) != 0 ||
          !S_ISREG(info.st_mode) ||
          info.st_uid != 0 || info.st_gid != 0 || info.st_nlink != 1 ||
          (info.st_mode & 0022) != 0) {
        throw Error("causal material descriptor source metadata unsafe");
      }
      reference_files_.emplace(name, std::move(reference));
    }
    expected_descriptor_binding_sha256_ = reference_descriptor_binding();
    spawn();
    await_ready();
  }

  HostCausalMaterialCell(const HostCausalMaterialCell&) = delete;
  HostCausalMaterialCell& operator=(const HostCausalMaterialCell&) = delete;
  ~HostCausalMaterialCell() { cleanup_noexcept(); }

  const std::string& principal_sha256() const { return principal_sha256_; }
  const std::string& descriptor_binding_sha256() const {
    return descriptor_binding_sha256_;
  }
  const std::string& unit() const { return unit_; }
  pid_t pid() const { return pid_; }
  std::uint64_t start_tick() const { return start_tick_; }
  uid_t uid() const { return uid_; }
  gid_t gid() const { return gid_; }

  CausalMaterialObservation execute(const std::string& arm) {
    if (finished_ || !authenticated()) {
      throw Error("causal material cell identity drifted before ARM");
    }
    const auto deadline = std::chrono::steady_clock::now() + kCausalMaterialDeadline;
    quorum_write_all(input_.get(), arm + "\n", deadline);
    const std::string header = quorum_read_line(
        output_.get(), output_buffer_, 128 * 1024, deadline,
        "causal material result timeout");
    const auto fields = operation_key_value_line(
        header, "LOOM_CAUSAL_MATERIAL_CELL_RESULT_V1");
    if (operation_require(fields, "mode") != mode_ ||
        operation_require(fields, "handle_is_bearer") != "false" ||
        operation_require(fields, "handle_is_execution_authority") != "false") {
      throw Error("causal material result header invalid");
    }
    if (quorum_read_line(output_.get(), output_buffer_, 4096, deadline,
                         "causal material record begin timeout") !=
        "LOOM_CAUSAL_MATERIAL_RECORD_BEGIN") {
      throw Error("causal material record begin invalid");
    }
    std::string record;
    for (;;) {
      const std::string line = quorum_read_line(
          output_.get(), output_buffer_, 128 * 1024, deadline,
          "causal material record timeout");
      if (line == "LOOM_CAUSAL_MATERIAL_RECORD_END") break;
      record += line + "\n";
      if (record.size() > 64 * 1024) throw Error("causal material record too large");
    }
    if (sha256(record) != operation_require(fields, "record_sha256") ||
        std::to_string(record.size()) != operation_require(fields, "record_bytes")) {
      throw Error("causal material record transport mismatch");
    }
    CausalMaterialObservation observation;
    observation.mode = mode_;
    observation.record = record;
    observation.record_sha256 = operation_require(fields, "record_sha256");
    observation.handle_type = operation_require(fields, "handle_type");
    observation.handle = operation_require(fields, "handle");
    observation.handle_sha256 = operation_require(fields, "handle_sha256");
    if (sha256(observation.handle) != observation.handle_sha256 ||
        (mode_ == "RUN_EXACT" &&
         observation.handle_type != "loom-result-v3") ||
        (mode_ == "ATTEST" &&
         observation.handle_type != "loom-attestation-v1")) {
      throw Error("causal material typed handle binding invalid");
    }
    observation.fields = fields;
    observation.uid = uid_;
    observation.gid = gid_;
    observation.pid = pid_;
    observation.start_tick = start_tick_;
    observation.unit = unit_;
    observation.principal_sha256 = principal_sha256_;
    observation.dynamic_user = true;
    observation.identity_authenticated = true;
    return observation;
  }

  void close(CausalMaterialObservation& observation) {
    if (finished_ || observation.mode != mode_ || !authenticated()) {
      throw Error("causal material cell CLOSE precondition invalid");
    }
    const auto deadline = std::chrono::steady_clock::now() + kCausalMaterialDeadline;
    quorum_write_all(input_.get(),
                     "CLOSE " + mode_ + " " + observation.record_sha256 + "\n",
                     deadline);
    input_ = UniqueFd();
    const auto fields = operation_key_value_line(
        quorum_read_line(output_.get(), output_buffer_, 4096, deadline,
                         "causal material CLOSE timeout"),
        "LOOM_CAUSAL_MATERIAL_CELL_CLOSED_V1");
    if (operation_require(fields, "mode") != mode_ ||
        operation_require(fields, "record_sha256") != observation.record_sha256 ||
        operation_require(fields, "authority_extinction") != "armed") {
      throw Error("causal material CLOSE binding invalid");
    }
    observation.close_bound = true;
    const auto extinction_deadline =
        std::chrono::steady_clock::now() + kCausalMaterialDeadline;
    while (std::chrono::steady_clock::now() < extinction_deadline) {
      observation.pidfd_extinct = !pidfd_live(pidfd_.get());
      observation.process_extinct = kill(pid_, 0) != 0 && errno == ESRCH;
      observation.cgroup_unpopulated = cgroup_unpopulated();
      if (observation.pidfd_extinct && observation.process_extinct &&
          observation.cgroup_unpopulated) {
        break;
      }
      poll(nullptr, 0, 5);
    }
    if (!observation.pidfd_extinct || !observation.process_extinct ||
        !observation.cgroup_unpopulated) {
      throw Error("causal material affirmative extinction incomplete");
    }
    const QuorumToolResult stop = quorum_run_tool(
        systemctl_, {"stop", unit_}, std::chrono::seconds(2));
    if (stop.exit_code != 0) throw Error("causal material unit stop refused");
    const int status = await_client_exit();
    if (status != 0) throw Error("causal material systemd client refused");
    while (std::chrono::steady_clock::now() < extinction_deadline) {
      observation.unit_inactive = unit_inactive();
      if (observation.unit_inactive) break;
      poll(nullptr, 0, 5);
    }
    if (!observation.unit_inactive) {
      throw Error("causal material unit extinction incomplete");
    }
    quorum_run_tool(systemctl_, {"reset-failed", unit_}, std::chrono::seconds(2));
    finished_ = true;
  }

 private:
  std::string property(const std::string& name) const {
    return quorum_tool_value(systemctl_,
                             {"show", unit_, "--property=" + name, "--value"});
  }

  bool authenticated() const {
    try {
      return pid_ > 1 && pidfd_live(pidfd_.get()) &&
             process_start_tick(pid_) == start_tick_ &&
             process_uid(pid_) == uid_ && process_gid(pid_) == gid_ &&
             process_executable(pid_) == runtime_ && uid_ != 0 && gid_ != 0 &&
             process_cgroup(pid_).find(unit_) != std::string::npos &&
             property("DynamicUser") == "yes" &&
             property("NoNewPrivileges") == "yes" &&
             property("ProtectSystem") == "strict" &&
             property("PrivateNetwork") == "yes" &&
             property("MemoryDenyWriteExecute") == "yes";
    } catch (...) {
      return false;
    }
  }

  bool unit_inactive() const {
    const QuorumToolResult result = quorum_run_tool(
        systemctl_, {"show", unit_, "--property=ActiveState", "--value"},
        std::chrono::seconds(2));
    if (result.exit_code != 0) return false;
    const std::string state = trim(result.output);
    return state == "inactive" || state == "failed" || state == "dead";
  }

  bool cgroup_unpopulated() const {
    try {
      const std::string control_group = property("ControlGroup");
      if (control_group.empty() || control_group.front() != '/' ||
          control_group.find("..") != std::string::npos) {
        return false;
      }
      std::istringstream events(
          read_file("/sys/fs/cgroup" + control_group + "/cgroup.events"));
      std::string key;
      std::string value;
      while (events >> key >> value) {
        if (key == "populated") return value == "0";
      }
    } catch (...) {
    }
    return false;
  }

  std::string reference_file_sha256(int descriptor) const {
    struct stat info {};
    if (fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) ||
        info.st_size < 0 || info.st_size > 16 * 1024 * 1024) {
      throw Error("causal reference descriptor shape invalid");
    }
    std::string contents;
    contents.resize(static_cast<std::size_t>(info.st_size));
    std::size_t offset = 0;
    while (offset < contents.size()) {
      const ssize_t count =
          pread(descriptor, contents.data() + offset, contents.size() - offset,
                static_cast<off_t>(offset));
      if (count > 0) {
        offset += static_cast<std::size_t>(count);
      } else if (count == 0) {
        throw Error("causal reference descriptor truncated");
      } else if (errno != EINTR) {
        throw Error("causal reference descriptor read failed");
      }
    }
    return sha256(contents);
  }

  std::string reference_descriptor_binding() const {
    std::ostringstream canonical;
    canonical << "LOOM_CAUSAL_MATERIAL_DESCRIPTORS/2";
    for (const auto& [name, descriptor] : reference_files_) {
      struct stat before {};
      struct stat after {};
      if (fstat(descriptor.get(), &before) != 0) {
        throw Error("causal reference descriptor fstat failed");
      }
      const std::string content_sha256 =
          reference_file_sha256(descriptor.get());
      if (fstat(descriptor.get(), &after) != 0 ||
          before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
          before.st_size != after.st_size || before.st_mode != after.st_mode ||
          before.st_uid != after.st_uid || before.st_gid != after.st_gid ||
          before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
          before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
          before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
          before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
        throw Error("causal reference descriptor changed while binding");
      }
      canonical << "|name=" << name << "|dev=" << before.st_dev
                << "|ino=" << before.st_ino << "|size=" << before.st_size
                << "|mode=" << before.st_mode << "|uid=" << before.st_uid
                << "|gid=" << before.st_gid
                << "|content_sha256=" << content_sha256;
    }
    return sha256(canonical.str());
  }

  void spawn() {
    int input_pipe[2];
    int output_pipe[2];
    if (pipe2(input_pipe, O_CLOEXEC) != 0 ||
        pipe2(output_pipe, O_CLOEXEC) != 0) {
      throw Error("cannot create causal material custody pipes");
    }
    UniqueFd input_read(input_pipe[0]);
    UniqueFd input_write(input_pipe[1]);
    UniqueFd output_read(output_pipe[0]);
    UniqueFd output_write(output_pipe[1]);
    client_pid_ = fork();
    if (client_pid_ < 0) throw Error("cannot fork causal systemd client");
    if (client_pid_ == 0) {
      if (dup2(input_read.get(), STDIN_FILENO) < 0 ||
          dup2(output_write.get(), STDOUT_FILENO) < 0 ||
          dup2(output_write.get(), STDERR_FILENO) < 0) {
        _exit(126);
      }
      input_read = UniqueFd();
      input_write = UniqueFd();
      output_read = UniqueFd();
      output_write = UniqueFd();
      std::vector<std::string> arguments = {
          "--quiet", "--unit=" + unit_, "--service-type=exec",
          "--pipe", "--wait", "--property=DynamicUser=yes",
          "--property=RemainAfterExit=yes",
          "--property=UMask=0077", "--property=NoNewPrivileges=yes",
          "--property=PrivateTmp=yes", "--property=PrivateDevices=yes",
          "--property=PrivateNetwork=yes", "--property=ProtectSystem=strict",
          "--property=ProtectHome=yes", "--property=ProtectKernelTunables=yes",
          "--property=ProtectKernelModules=yes", "--property=ProtectKernelLogs=yes",
          "--property=ProtectControlGroups=yes", "--property=ProtectClock=yes",
          "--property=ProtectProc=invisible", "--property=ProcSubset=pid",
          "--property=RestrictNamespaces=yes", "--property=RestrictSUIDSGID=yes",
          "--property=LockPersonality=yes", "--property=MemoryDenyWriteExecute=yes",
          "--property=RestrictRealtime=yes",
          "--property=SystemCallArchitectures=native",
          "--property=RestrictAddressFamilies=AF_UNIX",
          "--property=CapabilityBoundingSet=", "--property=AmbientCapabilities=",
          "--property=KillMode=control-group", "--property=TimeoutStopSec=2s",
          "--property=ReadOnlyPaths=" + runtime_};
      for (const auto& [name, path] : open_files_) {
        arguments.push_back("--property=OpenFile=" + path + ":" + name +
                            ":read-only");
      }
      arguments.insert(arguments.end(), {"--", runtime_, "--mode", mode_});
      std::vector<char*> values;
      values.reserve(arguments.size() + 2);
      values.push_back(const_cast<char*>(systemd_run_.c_str()));
      for (std::string& argument : arguments) values.push_back(argument.data());
      values.push_back(nullptr);
      execv(systemd_run_.c_str(), values.data());
      _exit(127);
    }
    input_read = UniqueFd();
    output_write = UniqueFd();
    input_ = std::move(input_write);
    output_ = std::move(output_read);
  }

  void await_ready() {
    const auto deadline = std::chrono::steady_clock::now() + kCausalMaterialDeadline;
    while (std::chrono::steady_clock::now() < deadline) {
      try {
        const auto parsed = parse_u64(property("MainPID"));
        if (parsed && *parsed > 1 &&
            *parsed <= static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
          pid_ = static_cast<pid_t>(*parsed);
          start_tick_ = process_start_tick(pid_);
          uid_ = process_uid(pid_);
          gid_ = process_gid(pid_);
          pidfd_ = UniqueFd(open_pidfd(pid_));
          if (authenticated()) break;
        }
      } catch (...) {
      }
      poll(nullptr, 0, 5);
    }
    if (!authenticated()) throw Error("causal material cell identity admission failed");
    const auto fields = operation_key_value_line(
        quorum_read_line(output_.get(), output_buffer_, 4096, deadline,
                         "causal material READY timeout"),
        "LOOM_CAUSAL_MATERIAL_CELL_READY_V1");
    if (operation_require(fields, "mode") != mode_ ||
        operation_require(fields, "semantic_authority") != "Sounio" ||
        operation_require(fields, "action") != "9037" ||
        operation_require(fields, "pid") != std::to_string(pid_) ||
        operation_require(fields, "start_tick") !=
            std::to_string(start_tick_) ||
        operation_require(fields, "uid") != std::to_string(uid_) ||
        operation_require(fields, "gid") != std::to_string(gid_) ||
        operation_require(fields, "executable_sha256") !=
            file_sha256(runtime_) ||
        operation_require(fields, "cgroup_sha256") !=
            sha256(process_cgroup(pid_)) ||
        operation_require(fields, "descriptor_binding_schema") !=
            "LOOM_CAUSAL_MATERIAL_DESCRIPTORS/2" ||
        operation_require(fields, "descriptor_binding_sha256") !=
            expected_descriptor_binding_sha256_ ||
        operation_require(fields, "arbitrary_path") != "false") {
      throw Error("causal material READY binding invalid");
    }
    principal_sha256_ = operation_require(fields, "principal_sha256");
    descriptor_binding_sha256_ =
        operation_require(fields, "descriptor_binding_sha256");
    const std::string expected_principal = sha256(
        "LOOM_CAUSAL_MATERIAL_PRINCIPAL/1|pid=" + std::to_string(pid_) +
        "|start_tick=" + std::to_string(start_tick_) + "|uid=" +
        std::to_string(uid_) + "|gid=" + std::to_string(gid_) +
        "|cgroup_sha256=" + sha256(process_cgroup(pid_)) +
        "|executable_sha256=" + file_sha256(runtime_));
    if (!lowercase_digest(principal_sha256_) ||
        !lowercase_digest(descriptor_binding_sha256_) ||
        principal_sha256_ != expected_principal ||
        descriptor_binding_sha256_ != expected_descriptor_binding_sha256_) {
      throw Error("causal material principal digest malformed");
    }
  }

  int await_client_exit() {
    const auto deadline = std::chrono::steady_clock::now() + kCausalMaterialDeadline;
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
      const pid_t waited = waitpid(client_pid_, &status, WNOHANG);
      if (waited == client_pid_) {
        client_pid_ = -1;
        return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
      }
      if (waited < 0 && errno != EINTR) throw Error("causal systemd client wait failed");
      poll(nullptr, 0, 5);
    }
    throw Error("causal systemd client exit timeout");
  }

  void cleanup_noexcept() {
    if (!unit_.empty()) {
      quorum_run_tool(systemctl_, {"stop", unit_}, std::chrono::seconds(2));
    }
    if (client_pid_ > 0) {
      kill(client_pid_, SIGKILL);
      while (waitpid(client_pid_, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
  }

  std::string runtime_;
  std::string systemd_run_;
  std::string systemctl_;
  std::string mode_;
  std::map<std::string, std::string> open_files_;
  std::map<std::string, UniqueFd> reference_files_;
  std::string unit_;
  std::string principal_sha256_;
  std::string descriptor_binding_sha256_;
  std::string expected_descriptor_binding_sha256_;
  std::string output_buffer_;
  UniqueFd input_;
  UniqueFd output_;
  UniqueFd pidfd_;
  pid_t client_pid_ = -1;
  pid_t pid_ = -1;
  std::uint64_t start_tick_ = 0;
  uid_t uid_ = 0;
  gid_t gid_ = 0;
  bool finished_ = false;
};

void causal_write_atomic_record(const std::string& path,
                                const std::string& contents, mode_t mode) {
  const std::string temporary =
      path + ".tmp-" + std::to_string(getpid()) + "-" +
      random_generation_sha256().substr(0, 8);
  UniqueFd descriptor(open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL |
                                                 O_CLOEXEC | O_NOFOLLOW,
                           0600));
  if (descriptor.get() < 0) throw Error("cannot create causal material record");
  quorum_write_all(descriptor.get(), contents,
                   std::chrono::steady_clock::now() + std::chrono::seconds(5));
  if (fsync(descriptor.get()) != 0 || fchmod(descriptor.get(), mode) != 0 ||
      fsync(descriptor.get()) != 0 || file_sha256(temporary) != sha256(contents)) {
    unlink(temporary.c_str());
    throw Error("causal material record atomic publish failed");
  }
  if (link(temporary.c_str(), path.c_str()) != 0) {
    const int publish_error = errno;
    unlink(temporary.c_str());
    if (publish_error == EEXIST) {
      throw Error("causal material record already published");
    }
    throw Error("causal material record atomic publish failed");
  }
  if (unlink(temporary.c_str()) != 0) {
    throw Error("causal material record temporary unlink failed");
  }
  UniqueFd directory(open(std::filesystem::path(path).parent_path().c_str(),
                          O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (directory.get() < 0 || fsync(directory.get()) != 0) {
    throw Error("causal material record directory fsync failed");
  }
}

int selftest_causal_atomic_publish(const std::string& root) {
  const std::string store = canonical_path(root);
  const std::string sentinel_path = store + "/sentinel.record";
  const std::string sentinel = read_file(sentinel_path);
  const std::string sentinel_sha256 = sha256(sentinel);
  bool refused = false;
  try {
    causal_write_atomic_record(sentinel_path, "replacement-must-not-publish\n",
                               0400);
  } catch (const Error& error) {
    refused = std::string(error.what()) ==
              "causal material record already published";
  }
  if (!refused || read_file(sentinel_path) != sentinel ||
      sha256(read_file(sentinel_path)) != sentinel_sha256) {
    throw Error("causal EEXIST publication corrupted prior authority");
  }

  const std::string fresh_path = store + "/fresh.record";
  causal_write_atomic_record(fresh_path, "fresh-authority\n", 0400);
  struct stat fresh_info {};
  if (lstat(fresh_path.c_str(), &fresh_info) != 0 ||
      !S_ISREG(fresh_info.st_mode) ||
      (fresh_info.st_mode & 0777) != 0400 ||
      read_file(fresh_path) != "fresh-authority\n") {
    throw Error("causal fresh publication did not persist exactly");
  }

  std::cout << "LOOM_CAUSAL_ATOMIC_PUBLISH PASS"
            << " existing_publish=REFUSED prior_authority_preserved=true"
            << " fresh_publish=PASS atomic_commit=link-no-replace+fsync"
            << " sentinel_sha256=" << sentinel_sha256
            << " python_executed=false rust_executed=false\n";
  return 0;
}

class CausalWorkflowJournalClient {
 public:
  CausalWorkflowJournalClient(std::string runtime, std::string repo_root,
                              std::string state_root, std::string workflow_id)
      : runtime_(canonical_path(runtime)), repo_root_(canonical_path(repo_root)),
        state_root_(canonical_path(state_root)),
        workflow_id_(std::move(workflow_id)) {
    if (!process_witness_root_artifact(runtime_, true)) {
      throw Error("causal workflow journal runtime metadata unsafe");
    }
    if (workflow_id_.empty() || workflow_id_.size() > 256 ||
        !std::all_of(workflow_id_.begin(), workflow_id_.end(),
                     [](unsigned char character) {
                       return std::isalnum(character) || character == '.' ||
                              character == '_' || character == '-';
                     })) {
      throw Error("causal workflow journal identity invalid");
    }
  }

  std::map<std::string, std::string> command(
      const std::string& operation, const std::string& expected_schema,
      const std::vector<std::string>& arguments = {}) const {
    std::vector<std::string> values = {operation, repo_root_, state_root_,
                                       workflow_id_};
    values.insert(values.end(), arguments.begin(), arguments.end());
    const QuorumToolResult result =
        quorum_run_tool(runtime_, values, std::chrono::seconds(30));
    if (result.exit_code != 0) {
      throw Error("causal workflow journal refused " + operation + ": " +
                  trim(result.output));
    }
    const std::string line = trim(result.output);
    const auto fields = operation_key_value_line(line, expected_schema);
    if (operation_require(fields, "phase").empty() ||
        operation_require(fields, "head_sha256").size() != 64 ||
        operation_require(fields, "compile_count").empty() ||
        operation_require(fields, "ticket_count").empty() ||
        operation_require(fields, "launch_count").empty()) {
      throw Error("causal workflow journal receipt incomplete");
    }
    return fields;
  }

 private:
  std::string runtime_;
  std::string repo_root_;
  std::string state_root_;
  std::string workflow_id_;
};

std::string causal_publish_typed_handle(const std::string& store,
                                        const std::string& handle_type,
                                        const std::string& handle,
                                        const std::string& handle_sha256) {
  if (handle.empty() || sha256(handle) != handle_sha256 ||
      !lowercase_digest(handle_sha256)) {
    throw Error("causal typed handle identity invalid");
  }
  const std::string directory_path = store + "/handles";
  if (mkdir(directory_path.c_str(), 0700) != 0 && errno != EEXIST) {
    throw Error("cannot create causal typed handle store");
  }
  struct stat directory_info {};
  if (lstat(directory_path.c_str(), &directory_info) != 0 ||
      !S_ISDIR(directory_info.st_mode) || directory_info.st_uid != 0 ||
      directory_info.st_gid != 0 || (directory_info.st_mode & 0777) != 0700) {
    throw Error("causal typed handle store metadata unsafe");
  }
  const std::string record =
      "schema=loom-causal-typed-handle-v1\nhandle_type=" + handle_type +
      "\nhandle_sha256=" + handle_sha256 + "\nhandle=" + handle +
      "\nhandle_is_bearer=false\nhandle_is_execution_authority=false\n";
  const std::string path = directory_path + "/" + handle_sha256 + ".handle";
  causal_write_atomic_record(path, record, 0400);
  return path;
}

bool causal_wait_for_pod_loss_release(const std::string& store,
                                      const std::string& guardian_generation,
                                      const std::string& workflow_id) {
  const std::string ready_record =
      "loom-causal-pod-loss-ready-v1\n"
      "guardian_generation=" + guardian_generation + "\nworkflow_id=" +
      workflow_id + "\nphase=RUNNING\n";
  const std::string ready_sha256 = sha256(ready_record);
  causal_write_atomic_record(store + "/pod-loss-ready.record", ready_record,
                             0400);
  const std::string release_path = store + "/pod-loss-release.record";
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(180);
  struct stat release_info {};
  while (std::chrono::steady_clock::now() < deadline) {
    if (lstat(release_path.c_str(), &release_info) == 0) break;
    if (errno != ENOENT) throw Error("causal Pod-loss release stat failed");
    poll(nullptr, 0, 20);
  }
  if (lstat(release_path.c_str(), &release_info) != 0 ||
      !S_ISREG(release_info.st_mode) || release_info.st_uid != 0 ||
      release_info.st_gid != 0 || release_info.st_nlink != 1 ||
      (release_info.st_mode & 0777) != 0400) {
    throw Error("causal Pod-loss release metadata unsafe");
  }
  const auto fields = operation_record_fields(
      read_file(release_path), "loom-causal-pod-loss-release-v1");
  const auto safe_uid = [](const std::string& value) {
    return value.size() >= 16 && value.size() <= 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char character) {
             return std::isalnum(character) || character == '-';
           });
  };
  if (fields.size() != 6 ||
      operation_require(fields, "guardian_generation") !=
          guardian_generation ||
      operation_require(fields, "workflow_id") != workflow_id ||
      operation_require(fields, "ready_sha256") != ready_sha256 ||
      operation_require(fields, "pod_a_deleted") != "true" ||
      !safe_uid(operation_require(fields, "pod_a_uid")) ||
      !safe_uid(operation_require(fields, "pod_b_uid")) ||
      operation_require(fields, "pod_a_uid") ==
          operation_require(fields, "pod_b_uid")) {
    throw Error("causal Pod-loss release binding invalid");
  }
  return true;
}

int selftest_causal_workflow_material_host(
    const std::string& root, const std::string& controller_runtime_path,
    const std::string& resident_runtime_path, const std::string& product_root,
    const std::string& product_runtime,
    const std::string& operation_fixture_manifest_path,
    const std::string& operation_fixture_bundle_path,
    const std::string& operation_catalog_manifest_path,
    const std::string& operation_result_manifest_path,
    const std::string& run_grant_manifest_path,
    const std::string& run_grant_bundle_path,
    const std::string& attest_grant_manifest_path,
    const std::string& attest_grant_bundle_path,
    const std::string& causal_manifest_path,
    const std::string& causal_journal_runtime_path,
    const std::string& material_cell_path, const std::string& store_root,
    const std::string& pod_loss_control,
    const std::string& systemd_run_path, const std::string& systemctl_path) {
  if (getuid() != 0 || trim(read_file("/proc/1/comm")) != "systemd") {
    throw Error("causal material host selftest requires root PID1 systemd");
  }
  const Manifest causal_manifest = parse_frozen_manifest(
      causal_manifest_path, kFrozenCausalWorkflowManifestSha256,
      "causal workflow action 9037");
  if (causal_manifest.require("stage") != "SEMANTICS_FROZEN" ||
      causal_manifest.require("producing_language") != "Sounio" ||
      causal_manifest.require("action") != "9037" ||
      causal_manifest.require("launch_authority") != "action-9030") {
    throw Error("causal workflow semantic authority posture invalid");
  }
  const OperationExecCellFixtures compile_fixtures =
      load_operation_exec_cell_fixtures(operation_fixture_manifest_path,
                                        operation_fixture_bundle_path);
  const OperationExecResultPolicy compile_policy =
      load_operation_exec_result_policy(operation_catalog_manifest_path,
                                        operation_result_manifest_path);
  const CausalCellGrantFixtures run_fixtures = load_causal_cell_grant_fixtures(
      run_grant_manifest_path, run_grant_bundle_path,
      kFrozenCausalRunGrantManifestSha256,
      "loom-causal-workflow-run-grant-fixture-freeze-v1",
      "LOOM_CAUSAL_WORKFLOW_RUN_GRANT_FIXTURE_V1 semantic_authority=Sounio launch_action=9030 workflow_action=9037 fixtures=4",
      17, true);
  const CausalCellGrantFixtures attest_fixtures = load_causal_cell_grant_fixtures(
      attest_grant_manifest_path, attest_grant_bundle_path,
      kFrozenCausalAttestGrantManifestSha256,
      "loom-causal-workflow-attest-grant-fixture-freeze-v1",
      "LOOM_CAUSAL_WORKFLOW_ATTEST_GRANT_FIXTURE_V1 semantic_authority=Sounio launch_action=9030 workflow_action=9037 fixtures=4",
      16, false);
  if (run_fixtures.artifact_sha256 != kFrozenOperationArtifactSha256) {
    throw Error("causal run grant artifact disagreed with compile semantics");
  }
  const std::string store = canonical_path(store_root);
  struct stat store_info {};
  if (lstat(store.c_str(), &store_info) != 0 || !S_ISDIR(store_info.st_mode) ||
      store_info.st_uid != 0 || store_info.st_gid != 0 ||
      (store_info.st_mode & 0777) != 0700) {
    throw Error("causal material store metadata unsafe");
  }

  const std::string guardian_generation = random_generation_sha256();
  const std::string workflow_generation = random_generation_sha256();
  const std::string journal_id = sha256(
      "LOOM_CAUSAL_JOURNAL/1|guardian=" + guardian_generation +
      "|workflow=" + workflow_generation);
  const std::string store_id = sha256(
      "LOOM_CAUSAL_STORE/1|device=" + std::to_string(store_info.st_dev) +
      "|inode=" + std::to_string(store_info.st_ino) + "|boot_id=" +
      trim(read_file("/proc/sys/kernel/random/boot_id")));
  const std::string controller_generation = random_generation_sha256();
  const std::string workflow_id =
      "canonical-call-b-" + workflow_generation.substr(0, 24);
  CausalWorkflowJournalClient journal(causal_journal_runtime_path, product_root,
                                      store, workflow_id);
  static_cast<void>(journal.command(
      "material-open", "MATERIAL_OPENED",
      {workflow_generation, guardian_generation, journal_id, store_id,
       controller_generation,
       "899d05ffe60528a6b71871e24fa0d1bc105cd033b7ae2c5a0a6d2bb808cdcad9"}));
  const auto compile_armed = journal.command("material-arm-compile",
                                             "MATERIAL_COMPILE_ARMED");

  const std::string compile_generation = random_generation_sha256();
  HostSounioCheckObservation compile;
  std::string compile_close_receipt_sha256;
  std::string artifact_path;
  {
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, compile_generation,
                                    "60000");
    HostSounioCheckCell cell(product_root, product_runtime, systemd_run_path,
                             systemctl_path, compile_fixtures, compile_policy,
                             compile_generation);
    const std::string compile_grant_identity = sha256(
        "LOOM_CAUSAL_LIVE_GRANT/1|mode=COMPILE|guardian=" +
        guardian_generation + "|generation=" + compile_generation +
        "|principal=" + cell.principal_sha256() + "|descriptors=" +
        cell.descriptor_sha256());
    const std::string compile_command_environment = sha256(
        "LOOM_CAUSAL_COMMAND_ENVIRONMENT/1|command=" +
        compile_fixtures.command +
        "|network=absent|shell=absent|dynamic_user=true");
    const std::string compile_peer_vector = sha256(
        "LOOM_CAUSAL_PEER_VECTOR/1|unit=" + cell.unit() + "|pid=" +
        std::to_string(cell.pid()) + "|start_tick=" +
        std::to_string(cell.start_tick()) + "|uid=" +
        std::to_string(cell.uid()) + "|gid=" + std::to_string(cell.gid()) +
        "|principal=" + cell.principal_sha256() + "|descriptors=" +
        cell.descriptor_sha256());
    const std::string compile_source_semantics_toolchain = sha256(
        "LOOM_CAUSAL_SOURCE_SEMANTICS_TOOLCHAIN/1|source=" +
        file_sha256(product_root + "/" + compile_fixtures.source) +
        "|semantics_manifest=" +
        std::string(kFrozenCausalWorkflowManifestSha256) + "|compiler=" +
        compile_policy.compiler_sha256 + "|artifact=" +
        std::string(kFrozenOperationArtifactSha256));
    const auto compile_bindings = [&](const std::string& transition_journal,
                                      const std::string& result_receipt) {
      return std::map<std::string, std::string>{
          {"grant_identity", compile_grant_identity},
          {"command_environment", compile_command_environment},
          {"peer_vector", compile_peer_vector},
          {"transition_journal", transition_journal},
          {"source_semantics_toolchain", compile_source_semantics_toolchain},
          {"result_receipt", result_receipt}};
    };
    {
      ExecQuorumController sabotage_controller(
          controller_runtime_path, root, resident_runtime_path,
          random_generation_sha256(), "60000");
      const QuorumReceipt refusal = sabotage_controller.command(
          "ISSUE", operation_live_grant_frame(
                       compile_fixtures, "command_mismatch",
                       compile_bindings(
                           operation_require(compile_armed, "head_sha256"),
                           compile_policy.manifest_sha256)));
      if (refusal.require("code") !=
              compile_fixtures.command_mismatch_code ||
          refusal.require("quorum_ready") != "false" ||
          refusal.require("controller_terminal") != "true") {
        throw Error("causal COMPILE action-9030 sabotage control failed");
      }
    }
    const QuorumReceipt issue = controller.command(
        "ISSUE", operation_live_grant_frame(
                     compile_fixtures, "issue",
                     compile_bindings(
                         operation_require(compile_armed, "head_sha256"),
                         compile_policy.manifest_sha256)));
    const std::string compile_issue_receipt_sha256 = sha256(issue.line + "\n");
    const QuorumReceipt consume = controller.command(
        "CONSUME", operation_live_grant_frame(
                       compile_fixtures, "consume",
                       compile_bindings(compile_issue_receipt_sha256,
                                        compile_issue_receipt_sha256)));
    if (issue.require("code") != "0" || consume.require("code") != "0" ||
        consume.require("state") != "OUTCOME_PENDING") {
      throw Error("causal COMPILE action-9030 grant refused");
    }
    static_cast<void>(journal.command("material-start-compile",
                                      "MATERIAL_COMPILE_RUNNING"));
    compile = cell.execute(sha256(issue.line + "\n" + consume.line + "\n"));
    artifact_path = cell.seal_artifact(store, compile);
    const QuorumReceipt close = controller.command(
        "CLOSE", operation_live_grant_frame(
                     compile_fixtures, "close",
                     compile_bindings(compile.record_sha256,
                                      compile.record_sha256)));
    if (close.require("state") != "CLOSED") {
      throw Error("causal COMPILE action-9030 close refused");
    }
    compile_close_receipt_sha256 = sha256(close.line + "\n");
    cell.close(compile, compile_close_receipt_sha256);
  }
  if (!compile.success_complete()) throw Error("causal COMPILE extinction incomplete");
  const std::string compile_record_path = store + "/compile.record";
  causal_write_atomic_record(compile_record_path, compile.record, 0400);
  static_cast<void>(journal.command(
      "material-close-compile", "MATERIAL_COMPILED",
      {compile_close_receipt_sha256, compile.record_sha256,
       sha256(compile.handle)}));

  const std::string run_grant_generation = random_generation_sha256();
  std::string run_ticket;
  bool pod_loss_synchronized = false;

  CausalMaterialObservation run;
  {
    HostCausalMaterialCell cell(material_cell_path, systemd_run_path,
                                systemctl_path, "RUN_EXACT",
                                {{"artifact", artifact_path}});
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, run_grant_generation,
                                    "60000");
    const std::string run_grant_identity = sha256(
        "LOOM_CAUSAL_LIVE_GRANT/1|mode=RUN_EXACT|guardian=" +
        guardian_generation + "|generation=" + run_grant_generation +
        "|principal=" + cell.principal_sha256() + "|descriptors=" +
        cell.descriptor_binding_sha256());
    const std::string run_command_environment = sha256(
        "LOOM_CAUSAL_COMMAND_ENVIRONMENT/1|command=" + run_fixtures.command +
        "|network=absent|shell=absent|dynamic_user=true");
    const std::string run_peer_vector = sha256(
        "LOOM_CAUSAL_PEER_VECTOR/1|unit=" + cell.unit() + "|pid=" +
        std::to_string(cell.pid()) + "|start_tick=" +
        std::to_string(cell.start_tick()) + "|uid=" +
        std::to_string(cell.uid()) + "|gid=" + std::to_string(cell.gid()) +
        "|principal=" + cell.principal_sha256() + "|descriptors=" +
        cell.descriptor_binding_sha256());
    const std::string run_source_semantics_toolchain = sha256(
        "LOOM_CAUSAL_SOURCE_SEMANTICS_TOOLCHAIN/1|source=" +
        compile.source_sha256 + "|semantics_manifest=" +
        std::string(kFrozenCausalWorkflowManifestSha256) + "|artifact=" +
        compile.artifact_sha256);
    const std::string run_preissue_receipt = sha256(
        "LOOM_CAUSAL_RUN_PREISSUE/1|compile_record=" +
        compile.record_sha256 + "|grant_identity=" + run_grant_identity);
    const auto run_bindings = [&](const std::string& transition_journal,
                                  const std::string& result_receipt) {
      return std::map<std::string, std::string>{
          {"grant_identity", run_grant_identity},
          {"command_environment", run_command_environment},
          {"peer_vector", run_peer_vector},
          {"transition_journal", transition_journal},
          {"source_semantics_toolchain", run_source_semantics_toolchain},
          {"result_receipt", result_receipt}};
    };
    {
      ExecQuorumController sabotage_controller(
          controller_runtime_path, root, resident_runtime_path,
          random_generation_sha256(), "60000");
      const QuorumReceipt refusal = sabotage_controller.command(
          "ISSUE", causal_live_grant_frame(
                       run_fixtures, "command_mismatch",
                       run_bindings(compile.record_sha256,
                                    run_preissue_receipt)));
      if (refusal.require("code") != run_fixtures.command_mismatch_code ||
          refusal.require("quorum_ready") != "false" ||
          refusal.require("controller_terminal") != "true") {
        throw Error("causal RUN_EXACT action-9030 sabotage control failed");
      }
    }
    const QuorumReceipt issue =
        controller.command("ISSUE", causal_live_grant_frame(
                                        run_fixtures, "issue",
                                        run_bindings(compile.record_sha256,
                                                     run_preissue_receipt)));
    if (issue.require("code") != "0") {
      throw Error("causal RUN_EXACT action-9030 issue refused");
    }
    const std::string issue_receipt_sha256 = sha256(issue.line + "\n");
    const std::string ticket_record =
        "schema=loom-causal-run-ticket-v1\nartifact_record_sha256=" +
        compile.record_sha256 + "\nartifact_sha256=" + compile.artifact_sha256 +
        "\nrun_event_sha256=" + run_fixtures.event_sha256 +
        "\nrun_grant_generation=" + run_grant_generation +
        "\nrun_grant_identity_sha256=" + run_grant_identity +
        "\nrun_issue_receipt_sha256=" + issue_receipt_sha256 +
        "\nguardian_generation=" + guardian_generation +
        "\ncell_principal_sha256=" + cell.principal_sha256() +
        "\ndescriptor_binding_sha256=" + cell.descriptor_binding_sha256() +
        "\nstate=RUN_ARMED\nrun_ticket_is_bearer=false\n"
        "run_ticket_is_execution_authority=false\n";
    run_ticket = sha256(ticket_record);
    causal_write_atomic_record(store + "/run-ticket.record", ticket_record, 0400);
    static_cast<void>(journal.command(
        "material-arm-run", "MATERIAL_RUN_ARMED",
        {run_ticket, run_grant_identity, run_grant_generation}));
    const QuorumReceipt consume = controller.command(
        "CONSUME", causal_live_grant_frame(
                       run_fixtures, "consume",
                       run_bindings(run_ticket, issue_receipt_sha256)));
    if (consume.require("code") != "0" ||
        consume.require("state") != "OUTCOME_PENDING") {
      throw Error("causal RUN_EXACT action-9030 consume refused");
    }
    const std::string run_start_receipt = sha256(consume.line + "\n");
    const std::string run_pid_identity = sha256(
        "LOOM_CAUSAL_RUN_PID/1|principal=" + cell.principal_sha256() +
        "|unit=" + cell.unit() + "|pid=" + std::to_string(cell.pid()) +
        "|start_tick=" + std::to_string(cell.start_tick()));
    static_cast<void>(journal.command(
        "material-mark-running", "MATERIAL_RUNNING",
        {run_start_receipt, run_pid_identity}));
    if (pod_loss_control == "enabled") {
      pod_loss_synchronized = causal_wait_for_pod_loss_release(
          store, guardian_generation, workflow_id);
    } else if (!pod_loss_control.empty() && pod_loss_control != "disabled") {
      throw Error("causal Pod-loss control mode invalid");
    }
    run = cell.execute("ARM RUN_EXACT " + run_ticket + " " +
                       compile.artifact_sha256);
    if (operation_require(run.fields, "exit_code") != "0" ||
        operation_require(run.fields, "stdout_sha256") !=
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ||
        operation_require(run.fields, "stderr_sha256") !=
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") {
      throw Error("causal RUN_EXACT observation diverged from Sounio freeze");
    }
    causal_write_atomic_record(store + "/result.record", run.record, 0400);
    static_cast<void>(causal_publish_typed_handle(
        store, run.handle_type, run.handle, run.handle_sha256));
    static_cast<void>(journal.command(
        "material-record-result", "MATERIAL_RUN_MEASURED",
        {operation_require(run.fields, "exit_code"),
         operation_require(run.fields, "stdout_sha256"),
         operation_require(run.fields, "stderr_sha256"), run.record_sha256,
         run.handle_sha256}));
    cell.close(run);
    if (!run.success_complete()) {
      throw Error("causal RUN_EXACT extinction incomplete before grant close");
    }
    const QuorumReceipt close = controller.command(
        "CLOSE", causal_live_grant_frame(
                     run_fixtures, "close",
                     run_bindings(run_ticket, run.record_sha256)));
    if (close.require("state") != "CLOSED") {
      throw Error("causal RUN_EXACT action-9030 close refused");
    }
  }
  if (!run.success_complete()) throw Error("causal RUN_EXACT extinction incomplete");
  static_cast<void>(journal.command(
      "material-close-run", "MATERIAL_RUN_CLOSED",
      {run.pidfd_extinct ? "true" : "false",
       run.process_extinct ? "true" : "false",
       run.cgroup_unpopulated && run.unit_inactive ? "true" : "false", "true",
       "true"}));

  const std::string hardware_record =
      "schema=loom-causal-hardware-record-v1\nhostname=" +
      trim(read_file("/etc/hostname")) + "\nkernel=" +
      trim(read_file("/proc/sys/kernel/osrelease")) +
      "\nboot_id=" + trim(read_file("/proc/sys/kernel/random/boot_id")) + "\n";
  const std::string hardware_path = store + "/hardware.record";
  causal_write_atomic_record(hardware_path, hardware_record, 0400);

  CausalMaterialObservation attest;
  const std::string attest_generation = random_generation_sha256();
  {
    HostCausalMaterialCell cell(
        material_cell_path, systemd_run_path, systemctl_path, "ATTEST",
        {{"compile_record", compile_record_path},
         {"hardware_record", hardware_path},
         {"result_record", store + "/result.record"},
         {"semantics_manifest", causal_manifest_path}});
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, attest_generation,
                                    "60000");
    const std::string attest_grant_identity = sha256(
        "LOOM_CAUSAL_LIVE_GRANT/1|mode=ATTEST|guardian=" +
        guardian_generation + "|generation=" + attest_generation +
        "|principal=" + cell.principal_sha256() + "|descriptors=" +
        cell.descriptor_binding_sha256());
    const std::string attest_command_environment = sha256(
        "LOOM_CAUSAL_COMMAND_ENVIRONMENT/1|command=" +
        attest_fixtures.command +
        "|network=absent|shell=absent|dynamic_user=true");
    const std::string attest_peer_vector = sha256(
        "LOOM_CAUSAL_PEER_VECTOR/1|unit=" + cell.unit() + "|pid=" +
        std::to_string(cell.pid()) + "|start_tick=" +
        std::to_string(cell.start_tick()) + "|uid=" +
        std::to_string(cell.uid()) + "|gid=" + std::to_string(cell.gid()) +
        "|principal=" + cell.principal_sha256() + "|descriptors=" +
        cell.descriptor_binding_sha256());
    const std::string attest_source_semantics_toolchain = sha256(
        "LOOM_CAUSAL_SOURCE_SEMANTICS_TOOLCHAIN/1|source=" +
        compile.source_sha256 + "|semantics_manifest=" +
        std::string(kFrozenCausalWorkflowManifestSha256) + "|compiler=" +
        compile_policy.compiler_sha256 + "|artifact=" + compile.artifact_sha256);
    const auto attest_bindings = [&](const std::string& transition_journal,
                                     const std::string& result_receipt) {
      return std::map<std::string, std::string>{
          {"grant_identity", attest_grant_identity},
          {"command_environment", attest_command_environment},
          {"peer_vector", attest_peer_vector},
          {"transition_journal", transition_journal},
          {"source_semantics_toolchain", attest_source_semantics_toolchain},
          {"result_receipt", result_receipt}};
    };
    {
      ExecQuorumController sabotage_controller(
          controller_runtime_path, root, resident_runtime_path,
          random_generation_sha256(), "60000");
      const QuorumReceipt refusal = sabotage_controller.command(
          "ISSUE", causal_live_grant_frame(
                       attest_fixtures, "command_mismatch",
                       attest_bindings(run.record_sha256,
                                       run.record_sha256)));
      if (refusal.require("code") != attest_fixtures.command_mismatch_code ||
          refusal.require("quorum_ready") != "false" ||
          refusal.require("controller_terminal") != "true") {
        throw Error("causal ATTEST action-9030 sabotage control failed");
      }
    }
    const QuorumReceipt issue =
        controller.command("ISSUE", causal_live_grant_frame(
                                        attest_fixtures, "issue",
                                        attest_bindings(run.record_sha256,
                                                        run.record_sha256)));
    if (issue.require("code") != "0") {
      throw Error("causal ATTEST action-9030 issue refused");
    }
    const std::string attest_issue_receipt_sha256 = sha256(issue.line + "\n");
    static_cast<void>(journal.command("material-arm-attest",
                                      "MATERIAL_ATTEST_ARMED"));
    const QuorumReceipt consume = controller.command(
        "CONSUME", causal_live_grant_frame(
                       attest_fixtures, "consume",
                       attest_bindings(attest_issue_receipt_sha256,
                                       run.record_sha256)));
    if (consume.require("code") != "0" ||
        consume.require("state") != "OUTCOME_PENDING") {
      throw Error("causal ATTEST action-9030 consume refused");
    }
    static_cast<void>(journal.command("material-start-attest",
                                      "MATERIAL_ATTEST_RUNNING"));
    attest = cell.execute(
        "ARM ATTEST " + compile.record_sha256 + " " + run.record_sha256 +
        " " + std::string(kFrozenCausalWorkflowManifestSha256) + " " +
        sha256(hardware_record));
    causal_write_atomic_record(store + "/attestation.record", attest.record,
                               0400);
    static_cast<void>(causal_publish_typed_handle(
        store, attest.handle_type, attest.handle, attest.handle_sha256));
    cell.close(attest);
    if (!attest.success_complete()) {
      throw Error("causal ATTEST extinction incomplete before grant close");
    }
    const QuorumReceipt close = controller.command(
        "CLOSE", causal_live_grant_frame(
                     attest_fixtures, "close",
                     attest_bindings(attest_issue_receipt_sha256,
                                     attest.record_sha256)));
    if (close.require("state") != "CLOSED") {
      throw Error("causal ATTEST action-9030 close refused");
    }
  }
  if (!attest.success_complete()) throw Error("causal ATTEST extinction incomplete");
  static_cast<void>(journal.command(
      "material-close-attest", "MATERIAL_ATTESTED_CLOSED",
      {attest.record_sha256, attest.handle_sha256}));

  std::cout
      << "LOOM_CAUSAL_WORKFLOW_MATERIAL_HOST PASS semantic_authority=Sounio"
      << " workflow_action=9037 launch_action=9030 material_language=C++20"
      << " material_role=MATERIAL_PARITY hostguardian=true dynamic_user=true"
      << " compile_actual=true artifact_sha256=" << compile.artifact_sha256
      << " artifact_record_sha256=" << compile.record_sha256
      << " run_ticket_sha256=" << run_ticket
      << " run_grant_generation=" << run_grant_generation
      << " run_exact_actual=true result_record_sha256=" << run.record_sha256
      << " attest_actual=true attestation_record_sha256="
      << attest.record_sha256
      << " compile_uid=" << compile.uid << " run_uid=" << run.uid
      << " attest_uid=" << attest.uid
      << " distinct_fresh_cells="
      << ((compile.principal_sha256 != run.principal_sha256 &&
           run.principal_sha256 != attest.principal_sha256 &&
           compile.principal_sha256 != attest.principal_sha256 &&
           compile.unit != run.unit && run.unit != attest.unit &&
           compile.unit != attest.unit && compile.success_complete() &&
           run.success_complete() && attest.success_complete())
              ? "true"
              : "false")
      << " fresh_cell_identity=unit+pid+start_tick+principal"
      << " inherited_descriptors=true arbitrary_path=false"
      << " action9037_executed=true dynamic_9030_bindings=true"
      << " action9030_sabotage=PASS"
      << " compile_grant_sabotage=DENY492"
      << " run_grant_sabotage=DENY492 attest_grant_sabotage=DENY492"
      << " action9030_sabotage=PASS"
      << " compile_grant_sabotage=DENY492"
      << " run_grant_sabotage=DENY492 attest_grant_sabotage=DENY492"
      << " typed_handle_persistence=true"
      << " run_ticket_is_bearer=false run_ticket_is_execution_authority=false"
      << " compile_count=1 ticket_count=1 launch_count=1"
      << " controller_recovery=false pod_loss_measured=false"
      << " pod_loss_synchronized="
      << (pod_loss_synchronized ? "true" : "false")
      << " python_executed=false rust_executed=false production_activation=false"
      << " parity_open=false claim_ready=false\n";
  return 0;
}
