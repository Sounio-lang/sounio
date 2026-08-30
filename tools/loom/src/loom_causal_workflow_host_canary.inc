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
  const std::string parents = one("PARENT_BINDINGS ");
  std::string encoded_bindings;
  for (const std::string key : {"grant_identity", "command_environment",
                                "peer_vector", "transition_journal",
                                "source_semantics_toolchain",
                                "result_receipt"}) {
    encoded_bindings += product_hex_digest_u32_limbs(one("BINDING " + key + " "));
  }
  for (const std::string label : {"issue", "consume", "close",
                                  "command_mismatch"}) {
    fixtures.frames.emplace(label,
                            one("FRAME " + label + " ") + " " + parents +
                                encoded_bindings);
  }
  return fixtures;
}

struct CausalMaterialObservation {
  std::string mode;
  std::string record;
  std::string record_sha256;
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
  bool unit_inactive = false;

  bool success_complete() const {
    return dynamic_user && identity_authenticated && close_bound &&
           pidfd_extinct && process_extinct && unit_inactive;
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
      if (name.empty() || name.find(':') != std::string::npos) {
        throw Error("causal material descriptor name invalid");
      }
      path = canonical_path(path);
      struct stat info {};
      if (lstat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) ||
          info.st_uid != 0 || info.st_gid != 0 || info.st_nlink != 1 ||
          (info.st_mode & 0022) != 0) {
        throw Error("causal material descriptor source metadata unsafe");
      }
    }
    spawn();
    await_ready();
  }

  HostCausalMaterialCell(const HostCausalMaterialCell&) = delete;
  HostCausalMaterialCell& operator=(const HostCausalMaterialCell&) = delete;
  ~HostCausalMaterialCell() { cleanup_noexcept(); }

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
    observation.handle_sha256 = operation_require(fields, "handle_sha256");
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
    const int status = await_client_exit();
    if (status != 0) throw Error("causal material systemd client refused");
    const auto extinction_deadline =
        std::chrono::steady_clock::now() + kCausalMaterialDeadline;
    while (std::chrono::steady_clock::now() < extinction_deadline) {
      observation.pidfd_extinct = !pidfd_live(pidfd_.get());
      observation.process_extinct = kill(pid_, 0) != 0 && errno == ESRCH;
      observation.unit_inactive = unit_inactive();
      if (observation.pidfd_extinct && observation.process_extinct &&
          observation.unit_inactive) {
        break;
      }
      poll(nullptr, 0, 5);
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
    if (result.exit_code != 0) return true;
    const std::string state = trim(result.output);
    return state == "inactive" || state == "failed" || state == "dead";
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
          "--quiet", "--unit=" + unit_, "--service-type=exec", "--collect",
          "--pipe", "--wait", "--property=DynamicUser=yes",
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
          "--property=KillMode=mixed", "--property=TimeoutStopSec=2s",
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
        operation_require(fields, "uid") != std::to_string(uid_) ||
        operation_require(fields, "gid") != std::to_string(gid_) ||
        operation_require(fields, "arbitrary_path") != "false") {
      throw Error("causal material READY binding invalid");
    }
    principal_sha256_ = operation_require(fields, "principal_sha256");
    if (!lowercase_digest(principal_sha256_)) {
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
    if (client_pid_ > 0) {
      quorum_run_tool(systemctl_, {"stop", unit_}, std::chrono::seconds(2));
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
  std::string unit_;
  std::string principal_sha256_;
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
      fsync(descriptor.get()) != 0 || link(temporary.c_str(), path.c_str()) != 0 ||
      unlink(temporary.c_str()) != 0) {
    unlink(temporary.c_str());
    unlink(path.c_str());
    throw Error("causal material record atomic publish failed");
  }
  UniqueFd directory(open(std::filesystem::path(path).parent_path().c_str(),
                          O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (directory.get() < 0 || fsync(directory.get()) != 0) {
    throw Error("causal material record directory fsync failed");
  }
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
    const std::string& material_cell_path, const std::string& store_root,
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

  const std::string compile_generation = random_generation_sha256();
  HostSounioCheckObservation compile;
  std::string artifact_path;
  {
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, compile_generation,
                                    "60000");
    HostSounioCheckCell cell(product_root, product_runtime, systemd_run_path,
                             systemctl_path, compile_fixtures, compile_policy,
                             compile_generation);
    const QuorumReceipt issue =
        controller.command("ISSUE", compile_fixtures.frames.at("issue"));
    const QuorumReceipt consume =
        controller.command("CONSUME", compile_fixtures.frames.at("consume"));
    if (issue.require("code") != "0" || consume.require("code") != "0" ||
        consume.require("state") != "OUTCOME_PENDING") {
      throw Error("causal COMPILE action-9030 grant refused");
    }
    compile = cell.execute(sha256(issue.line + "\n" + consume.line + "\n"));
    artifact_path = cell.seal_artifact(store, compile);
    const QuorumReceipt close =
        controller.command("CLOSE", compile_fixtures.frames.at("close"));
    if (close.require("state") != "CLOSED") {
      throw Error("causal COMPILE action-9030 close refused");
    }
    cell.close(compile, sha256(close.line + "\n"));
  }
  if (!compile.success_complete()) throw Error("causal COMPILE extinction incomplete");
  const std::string compile_record_path = store + "/compile.record";
  causal_write_atomic_record(compile_record_path, compile.record, 0400);

  const std::string guardian_generation = random_generation_sha256();
  const std::string run_grant_generation = random_generation_sha256();
  const std::string ticket_record =
      "schema=loom-causal-run-ticket-v1\nartifact_record_sha256=" +
      compile.record_sha256 + "\nartifact_sha256=" + compile.artifact_sha256 +
      "\nrun_event_sha256=" + run_fixtures.event_sha256 +
      "\nrun_grant_generation=" + run_grant_generation +
      "\nguardian_generation=" + guardian_generation +
      "\nrun_ticket_is_bearer=false\nrun_ticket_is_execution_authority=false\n";
  const std::string run_ticket = sha256(ticket_record);
  causal_write_atomic_record(store + "/run-ticket.record", ticket_record, 0400);

  CausalMaterialObservation run;
  {
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, run_grant_generation,
                                    "60000");
    const QuorumReceipt issue =
        controller.command("ISSUE", run_fixtures.frames.at("issue"));
    const QuorumReceipt consume =
        controller.command("CONSUME", run_fixtures.frames.at("consume"));
    if (issue.require("code") != "0" || consume.require("code") != "0" ||
        consume.require("state") != "OUTCOME_PENDING") {
      throw Error("causal RUN_EXACT action-9030 grant refused");
    }
    HostCausalMaterialCell cell(material_cell_path, systemd_run_path,
                                systemctl_path, "RUN_EXACT",
                                {{"artifact", artifact_path}});
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
    const QuorumReceipt close =
        controller.command("CLOSE", run_fixtures.frames.at("close"));
    if (close.require("state") != "CLOSED") {
      throw Error("causal RUN_EXACT action-9030 close refused");
    }
    cell.close(run);
  }
  if (!run.success_complete()) throw Error("causal RUN_EXACT extinction incomplete");

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
    ExecQuorumController controller(controller_runtime_path, root,
                                    resident_runtime_path, attest_generation,
                                    "60000");
    const QuorumReceipt issue =
        controller.command("ISSUE", attest_fixtures.frames.at("issue"));
    const QuorumReceipt consume =
        controller.command("CONSUME", attest_fixtures.frames.at("consume"));
    if (issue.require("code") != "0" || consume.require("code") != "0" ||
        consume.require("state") != "OUTCOME_PENDING") {
      throw Error("causal ATTEST action-9030 grant refused");
    }
    HostCausalMaterialCell cell(
        material_cell_path, systemd_run_path, systemctl_path, "ATTEST",
        {{"compile_record", compile_record_path},
         {"hardware_record", hardware_path},
         {"result_record", store + "/result.record"},
         {"semantics_manifest", causal_manifest_path}});
    attest = cell.execute(
        "ARM ATTEST " + compile.record_sha256 + " " + run.record_sha256 +
        " " + std::string(kFrozenCausalWorkflowManifestSha256) + " " +
        sha256(hardware_record));
    causal_write_atomic_record(store + "/attestation.record", attest.record,
                               0400);
    const QuorumReceipt close =
        controller.command("CLOSE", attest_fixtures.frames.at("close"));
    if (close.require("state") != "CLOSED") {
      throw Error("causal ATTEST action-9030 close refused");
    }
    cell.close(attest);
  }
  if (!attest.success_complete()) throw Error("causal ATTEST extinction incomplete");

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
      << ((compile.uid != run.uid && run.uid != attest.uid &&
           compile.uid != attest.uid)
              ? "true"
              : "false")
      << " inherited_descriptors=true arbitrary_path=false"
      << " run_ticket_is_bearer=false run_ticket_is_execution_authority=false"
      << " compile_count=1 ticket_count=1 launch_count=1"
      << " controller_recovery=false pod_loss_measured=false"
      << " python_executed=false rust_executed=false production_activation=false"
      << " parity_open=false claim_ready=false\n";
  return 0;
}
