constexpr auto kProductExecIngressHostDeadline = std::chrono::seconds(60);

enum class ProductIngressReply {
  Exact,
  BindingMismatch,
  CloseWithoutReply,
  ResultExact,
  ResultBindingMismatch,
  ResultReceiptMismatch,
  ResultManifestMismatch,
};

enum class ProductIngressExchangeExpectation {
  Request,
  RefusalBeforeRequest,
};

struct ProductIngressHostRun {
  std::string unit;
  std::string runtime_directory;
  std::string output;
  std::string ingress_log;
  std::string hook_log;
  std::string provider_daemon_log;
  std::string provider_guardian_log;
  int hook_code = 255;
  pid_t main_pid = 0;
  uid_t main_uid = 0;
  gid_t main_gid = 0;
  std::uint64_t main_start_tick = 0;
  pid_t descriptor_peer_pid = 0;
  pid_t descriptor_peer_parent_pid = 0;
  uid_t descriptor_peer_uid = 0;
  gid_t descriptor_peer_gid = 0;
  std::uint64_t descriptor_peer_start_tick = 0;
  bool identity_authenticated = false;
  bool dynamic_user = false;
  std::string raw_event_sha256;
  std::string event_sha256;
  std::string command_sha256;
};

using ProductIngressBoundCallback =
    std::function<void(const ProductIngressHostRun&)>;

struct ProductIngressResult {
  std::string handle;
  std::string receipt_sha256;
  std::string receipt;
  std::string manifest_sha256;
};

using ProductIngressResultCallback =
    std::function<ProductIngressResult(const ProductIngressHostRun&)>;

std::string product_hex_encode(std::string_view value) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string output(value.size() * 2, '0');
  for (std::size_t index = 0; index < value.size(); ++index) {
    const auto byte = static_cast<unsigned char>(value[index]);
    output[index * 2] = hex[byte >> 4];
    output[index * 2 + 1] = hex[byte & 0x0f];
  }
  return output;
}

std::string product_json_string(std::string_view value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          static constexpr char hex[] = "0123456789abcdef";
          output << "\\u00" << hex[character >> 4] << hex[character & 0x0f];
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  output << '"';
  return output.str();
}

std::string product_compact_diagnostic(std::string value) {
  for (char& character : value) {
    if (character == '\n' || character == '\r' || character == '\t') {
      character = ' ';
    }
  }
  constexpr std::size_t limit = 4096;
  if (value.size() > limit) value.resize(limit);
  return value;
}

std::string product_hook_event(const std::string& root,
                               const std::string& session,
                               const std::string& command) {
  return "{\"hook_event_name\":\"PreToolUse\",\"session_id\":" +
         product_json_string(session) + ",\"cwd\":" +
         product_json_string(root) +
         ",\"tool_name\":\"Bash\",\"tool_input\":{\"command\":" +
         product_json_string(command) + "}}\n";
}

std::string product_read_output(int descriptor,
                                std::chrono::steady_clock::time_point deadline) {
  std::string output;
  std::array<char, 4096> bytes{};
  for (;;) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) throw Error("product hook output timeout");
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now);
    pollfd candidate{descriptor, POLLIN | POLLHUP, 0};
    const int ready =
        poll(&candidate, 1, static_cast<int>(remaining.count()) + 1);
    if (ready == 0) throw Error("product hook output timeout");
    if (ready < 0) {
      if (errno == EINTR) continue;
      throw Error("product hook output poll failed");
    }
    const ssize_t count = read(descriptor, bytes.data(), bytes.size());
    if (count > 0) {
      output.append(bytes.data(), static_cast<std::size_t>(count));
      if (output.size() > 1024 * 1024) {
        throw Error("product hook output exceeded limit");
      }
    } else if (count == 0) {
      return output;
    } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
      throw Error("product hook output read failed");
    }
  }
}

std::string product_process_environment(pid_t pid) {
  return read_file("/proc/" + std::to_string(pid) + "/environ");
}

pid_t product_process_parent(pid_t pid) {
  const std::string record =
      read_file("/proc/" + std::to_string(pid) + "/stat");
  const std::size_t closing = record.rfind(')');
  if (closing == std::string::npos || closing + 2 >= record.size()) {
    throw Error("product connector process stat malformed");
  }
  std::istringstream input(record.substr(closing + 2));
  char state = 0;
  std::int64_t parent = 0;
  if (!(input >> state >> parent) || parent <= 0 ||
      parent > std::numeric_limits<pid_t>::max()) {
    throw Error("product connector parent PID missing");
  }
  return static_cast<pid_t>(parent);
}

bool product_environment_has(const std::string& environment,
                             const std::string& binding) {
  std::size_t offset = 0;
  while (offset < environment.size()) {
    const std::size_t end = environment.find('\0', offset);
    const std::size_t length =
        (end == std::string::npos ? environment.size() : end) - offset;
    if (environment.compare(offset, length, binding) == 0) return true;
    if (end == std::string::npos) break;
    offset = end + 1;
  }
  return false;
}

bool product_same_user_namespace(pid_t pid) {
  struct stat broker {};
  struct stat lane {};
  return stat("/proc/self/ns/user", &broker) == 0 &&
         stat(("/proc/" + std::to_string(pid) + "/ns/user").c_str(), &lane) == 0 &&
         broker.st_dev == lane.st_dev && broker.st_ino == lane.st_ino;
}

class ProductExecIngressHostCase {
 public:
  ProductExecIngressHostCase(
      std::string product_root, std::string product_runtime,
      std::string language_runtime, std::string resident_runtime,
      std::string systemd_run, std::string systemctl, std::string label,
      std::string command, bool dynamic_user, bool descriptor_present,
      ProductIngressReply reply,
      ProductIngressExchangeExpectation exchange_expectation,
      ProductIngressBoundCallback on_bound = {},
      std::string result_event_sha256 = {},
      ProductIngressResultCallback on_result = {},
      std::string provider_fixture = {})
      : product_root_(canonical_path(product_root)),
        product_runtime_(canonical_path(product_runtime)),
        language_runtime_(canonical_path(language_runtime)),
        resident_runtime_(canonical_path(resident_runtime)),
        systemd_run_(canonical_path(systemd_run)),
        systemctl_(canonical_path(systemctl)),
        label_(std::move(label)),
        command_(std::move(command)),
        dynamic_user_(dynamic_user),
        descriptor_present_(descriptor_present),
        reply_(reply),
        exchange_expectation_(exchange_expectation),
        on_bound_(std::move(on_bound)),
        result_event_sha256_(std::move(result_event_sha256)),
        on_result_(std::move(on_result)),
        provider_fixture_(provider_fixture.empty()
                              ? ""
                              : canonical_path(provider_fixture)),
        nonce_(random_generation_sha256().substr(0, 16)),
        unit_("sounio-loom-product-ingress-" + std::to_string(getpid()) + "-" +
              nonce_ + ".service"),
        runtime_directory_("sounio-loom-product-ingress-" + nonce_),
        runtime_path_("/run/" + runtime_directory_),
        socket_path_("/run/sounio-loom-product-ingress-" + nonce_ + ".sock") {
    if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0) {
      throw Error("product ExecIngress host canary requires root");
    }
    verify_runtime(product_runtime_, "product hook runtime");
    verify_runtime(language_runtime_, "language authority runtime");
    verify_runtime(resident_runtime_, "resident action-9031 runtime");
    if (!provider_fixture_.empty()) {
      verify_runtime(provider_fixture_, "provider hook fixture");
    }
  }

  ProductExecIngressHostCase(const ProductExecIngressHostCase&) = delete;
  ProductExecIngressHostCase& operator=(const ProductExecIngressHostCase&) =
      delete;
  ~ProductExecIngressHostCase() { cleanup_noexcept(); }

  ProductIngressHostRun execute() {
    if (descriptor_present_) create_listener();
    spawn_systemd_client();
    close_input();
    if (descriptor_present_) {
      accept_guardian_connection();
      await_identity();
      if (exchange_expectation_ ==
          ProductIngressExchangeExpectation::RefusalBeforeRequest) {
        collect_client_with_diagnostics();
        ProductIngressHostRun result = observation();
        cleanup_noexcept();
        return result;
      }
      try {
        exchange_binding();
      } catch (const Error& ingress_error) {
        connection_ = UniqueFd();
        try {
          collect_client_with_diagnostics();
        } catch (const Error& collection_error) {
          throw Error(std::string(ingress_error.what()) +
                      " collection_error=" + collection_error.what());
        }
        throw Error(std::string(ingress_error.what()) +
                    " hook_code=" + std::to_string(hook_code_) +
                    " output=" + product_compact_diagnostic(output_text_) +
                    " ingress_log=" +
                    product_compact_diagnostic(optional_log("ingress.tsv")) +
                    " hook_log=" +
                    product_compact_diagnostic(optional_log("hook.tsv")));
      }
    }
    collect_client_with_diagnostics();
    ProductIngressHostRun result = observation();
    cleanup_noexcept();
    return result;
  }

 private:
  void verify_runtime(const std::string& path, std::string_view label) {
    struct stat info {};
    if (stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) ||
        info.st_uid != 0 || info.st_gid != 0 || (info.st_mode & 0222) != 0 ||
        (info.st_mode & 0111) == 0) {
      throw Error(std::string(label) + " metadata is unsafe");
    }
  }

  std::string property(const std::string& name) const {
    return quorum_tool_value(systemctl_,
                             {"show", unit_, "--property=" + name, "--value"});
  }

  void create_listener() {
    if (socket_path_.size() >= sizeof(sockaddr_un::sun_path)) {
      throw Error("product ExecIngress host socket path is too long");
    }
    unlink(socket_path_.c_str());
    listener_ = UniqueFd(socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (listener_.get() < 0) throw Error("cannot create product host listener");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, socket_path_.c_str(), socket_path_.size() + 1);
    if (bind(listener_.get(), reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0 ||
        chmod(socket_path_.c_str(), 0600) != 0 || listen(listener_.get(), 1) != 0) {
      throw Error("cannot bind product host listener");
    }
  }

  void spawn_systemd_client() {
    const std::string event = product_hook_event(
        product_root_, "product-dynamic-user-" + nonce_, command_);
    raw_event_sha256_ = sha256(event);
    int input_pipe[2];
    int output_pipe[2];
    if (pipe2(input_pipe, O_CLOEXEC) != 0 ||
        pipe2(output_pipe, O_CLOEXEC) != 0) {
      throw Error("cannot create product systemd client pipes");
    }
    UniqueFd input_read(input_pipe[0]);
    UniqueFd input_write(input_pipe[1]);
    UniqueFd output_read(output_pipe[0]);
    UniqueFd output_write(output_pipe[1]);
    client_pid_ = fork();
    if (client_pid_ < 0) throw Error("cannot fork product systemd client");
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
          "--quiet",
          "--unit=" + unit_,
          "--service-type=exec",
          "--collect",
          "--pipe",
          "--wait",
          "--property=UMask=0077",
          "--property=NoNewPrivileges=yes",
          "--property=PrivateTmp=yes",
          "--property=PrivateDevices=yes",
          "--property=PrivateNetwork=yes",
          "--property=PrivateUsers=no",
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
          "--property=CapabilityBoundingSet=",
          "--property=AmbientCapabilities=",
          "--property=KillMode=mixed",
          "--property=TimeoutStopSec=2s",
          "--property=RuntimeDirectory=" + runtime_directory_,
          "--property=RuntimeDirectoryMode=0700",
          "--property=RuntimeDirectoryPreserve=yes",
          "--property=WorkingDirectory=" + product_root_,
          "--property=ReadOnlyPaths=" + product_root_,
          "--setenv=SOUNIO_LOOM_HOOK_TEST_MODE=1",
          "--setenv=SOUNIO_LOOM_EXEC_INGRESS_REQUIRED=1",
          "--setenv=SOUNIO_LOOM_EXEC_INGRESS_PROBE_ONLY=1",
          "--setenv=GIT_CONFIG_COUNT=1",
          "--setenv=GIT_CONFIG_KEY_0=safe.directory",
          "--setenv=GIT_CONFIG_VALUE_0=" + product_root_,
          "--setenv=SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT=" + product_root_,
          "--setenv=SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME=" +
              language_runtime_,
          "--setenv=SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_RUNTIME=" +
              resident_runtime_,
          "--setenv=SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG=" + runtime_path_ +
              "/hook.tsv",
          "--setenv=SOUNIO_LOOM_EXEC_INGRESS_DARK_LOG=" + runtime_path_ +
              "/ingress.tsv",
          "--setenv=SOUNIO_LOOM_ACTIVATION_DARK_LOG=" + runtime_path_ +
              "/activation.tsv",
          "--setenv=SOUNIO_LOOM_RESIDENT_RECEIPT_LOG=" + runtime_path_ +
              "/resident.tsv"};
      if (!result_event_sha256_.empty()) {
        arguments.push_back("--setenv=SOUNIO_LOOM_EXEC_INTENT_PROJECTION=1");
      }
      if (!provider_fixture_.empty()) {
        arguments.push_back("--setenv=SOUNIO_LOOM_PROVIDER_CODEX=" +
                            provider_fixture_);
        arguments.push_back("--setenv=SOUNIO_LOOM_PROVIDER_HOOK_RUNTIME=" +
                            product_runtime_);
        arguments.push_back("--setenv=SOUNIO_LOOM_PRODUCT_ACTIVATION_ROOT=" +
                            product_root_);
        arguments.push_back("--setenv=SOUNIO_LOOM_COORD_AUTO=0");
      }
      if (dynamic_user_) {
        arguments.push_back("--property=DynamicUser=yes");
      } else {
        arguments.push_back("--property=User=root");
        arguments.push_back("--property=Group=root");
      }
      if (descriptor_present_) {
        arguments.push_back("--setenv=SOUNIO_LOOM_EXEC_INGRESS_FD=3");
        arguments.push_back("--property=OpenFile=" + socket_path_ +
                            ":loom-exec-ingress");
      }
      if (label_ != "current_material") {
        arguments.push_back("--setenv=SOUNIO_LOOM_ACTIVATION_DARK_LABEL=" +
                            label_);
      }
      arguments.push_back("--");
      arguments.push_back(product_runtime_);
      if (provider_fixture_.empty()) {
        arguments.push_back("agent-hook");
        arguments.push_back("--agent");
        arguments.push_back("codex");
      } else {
        arguments.push_back("provider-start");
        arguments.push_back("--wait");
        arguments.push_back("--provider");
        arguments.push_back("codex");
        arguments.push_back("--state-dir");
        arguments.push_back(runtime_path_ + "/provider-state");
        arguments.push_back("--agent");
        arguments.push_back("provider-host-canary");
        arguments.push_back("--lane");
        arguments.push_back("provider-" + nonce_);
        arguments.push_back("--session-id");
        arguments.push_back("44444444-4444-4444-8444-444444444444");
        arguments.push_back("--cwd");
        arguments.push_back(product_root_);
        arguments.push_back("--prompt");
        arguments.push_back(event);
        arguments.push_back("--isolate-context");
      }
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
    input_read = UniqueFd();
    output_write = UniqueFd();
    input_ = std::move(input_write);
    output_ = std::move(output_read);
    if (provider_fixture_.empty()) {
      const auto deadline =
          std::chrono::steady_clock::now() + kProductExecIngressHostDeadline;
      quorum_write_all(input_.get(), event, deadline);
    }
  }

  void accept_guardian_connection() {
    const auto deadline =
        std::chrono::steady_clock::now() + kProductExecIngressHostDeadline;
    quorum_wait(listener_.get(), POLLIN, deadline,
                "product HostGuardian connection timeout");
    connection_ = UniqueFd(accept4(listener_.get(), nullptr, nullptr, SOCK_CLOEXEC));
    if (connection_.get() < 0) {
      throw Error("cannot accept product HostGuardian connection");
    }
    ucred peer{};
    socklen_t size = sizeof(peer);
    if (getsockopt(connection_.get(), SOL_SOCKET, SO_PEERCRED, &peer, &size) !=
        0) {
      throw Error("product HostGuardian peer credentials unavailable");
    }
    descriptor_peer_pid_ = peer.pid;
    descriptor_peer_uid_ = peer.uid;
    descriptor_peer_gid_ = peer.gid;
    if (peer.pid <= 1 || peer.uid != 0 || peer.gid != 0) {
      throw Error("product OpenFile connector is not a root executor: pid=" +
                  std::to_string(peer.pid) + " uid=" +
                  std::to_string(peer.uid) + " gid=" +
                  std::to_string(peer.gid));
    }
    descriptor_peer_parent_pid_ = product_process_parent(peer.pid);
    descriptor_peer_start_tick_ = process_start_tick(peer.pid);
    descriptor_peer_cgroup_sha256_ = sha256(process_cgroup(peer.pid));
    if (descriptor_peer_parent_pid_ != 1 ||
        process_cgroup(peer.pid).find(unit_) == std::string::npos) {
      throw Error("product OpenFile root executor lacks PID1/unit ancestry: pid=" +
                  std::to_string(peer.pid) + " parent=" +
                  std::to_string(descriptor_peer_parent_pid_));
    }
  }

  void observe_client() {
    if (client_pid_ <= 0 || client_reaped_) return;
    int status = 0;
    const pid_t waited = waitpid(client_pid_, &status, WNOHANG);
    if (waited < 0 && errno != EINTR) {
      throw Error("product systemd client wait failed");
    }
    if (waited == client_pid_) {
      client_reaped_ = true;
      client_status_ = status;
    }
  }

  void await_identity() {
    const auto deadline =
        std::chrono::steady_clock::now() + kProductExecIngressHostDeadline;
    for (;;) {
      observe_client();
      try {
        const auto parsed = parse_u64(property("MainPID"));
        if (parsed && *parsed > 1 &&
            *parsed <= static_cast<std::uint64_t>(
                           std::numeric_limits<pid_t>::max())) {
          main_pid_ = static_cast<pid_t>(*parsed);
          main_start_tick_ = process_start_tick(main_pid_);
          main_uid_ = process_uid(main_pid_);
          main_gid_ = process_gid(main_pid_);
          cgroup_sha256_ = sha256(process_cgroup(main_pid_));
          pidfd_ = UniqueFd(open_pidfd(main_pid_));
          const std::string environment = product_process_environment(main_pid_);
          const bool identity =
              process_executable(main_pid_) == product_runtime_ &&
              pidfd_live(pidfd_.get()) &&
              process_start_tick(main_pid_) == main_start_tick_ &&
              process_cgroup(main_pid_).find(unit_) != std::string::npos &&
              property("NoNewPrivileges") == "yes" &&
              property("ProtectSystem") == "strict" &&
              property("PrivateNetwork") == "yes" &&
              property("PrivateUsers") == "no" &&
              product_same_user_namespace(main_pid_) &&
              product_environment_has(
                  environment, "SOUNIO_LOOM_EXEC_INGRESS_REQUIRED=1") &&
              product_environment_has(
                  environment, "SOUNIO_LOOM_EXEC_INGRESS_PROBE_ONLY=1") &&
              product_environment_has(environment, "GIT_CONFIG_COUNT=1") &&
              product_environment_has(environment,
                                      "GIT_CONFIG_KEY_0=safe.directory") &&
              product_environment_has(environment,
                                      "GIT_CONFIG_VALUE_0=" + product_root_) &&
              (result_event_sha256_.empty() ||
               product_environment_has(
                   environment, "SOUNIO_LOOM_EXEC_INTENT_PROJECTION=1")) &&
              (provider_fixture_.empty() ||
               (product_environment_has(
                    environment,
                    "SOUNIO_LOOM_PROVIDER_CODEX=" + provider_fixture_) &&
                product_environment_has(
                    environment,
                    "SOUNIO_LOOM_PROVIDER_HOOK_RUNTIME=" + product_runtime_) &&
                product_environment_has(
                    environment,
                    "SOUNIO_LOOM_PRODUCT_ACTIVATION_ROOT=" + product_root_) &&
                product_environment_has(environment,
                                        "SOUNIO_LOOM_COORD_AUTO=0"))) &&
              (!descriptor_present_ ||
               (product_environment_has(environment, "LISTEN_FDS=1") &&
                product_environment_has(
                    environment, "LISTEN_FDNAMES=loom-exec-ingress"))) &&
              !product_environment_has(environment, "SOUNIO_LOOM_TOKEN_FILE=");
          const bool principal =
              dynamic_user_
                  ? property("DynamicUser") == "yes" && main_uid_ != 0 &&
                        main_gid_ != 0
                  : property("DynamicUser") == "no" && main_uid_ == 0 &&
                        main_gid_ == 0;
          const bool connector_transition =
              descriptor_peer_pid_ == main_pid_ &&
              descriptor_peer_parent_pid_ == 1 &&
              descriptor_peer_start_tick_ == main_start_tick_ &&
              descriptor_peer_cgroup_sha256_ == cgroup_sha256_;
          identity_authenticated_ = identity && principal && connector_transition;
          if (!identity_authenticated_) {
            throw Error("product lane principal identity admission failed");
          }
          return;
        }
      } catch (const Error&) {
      }
      if (client_reaped_) {
        throw Error("product hook exited before identity admission");
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        throw Error("product lane principal identity admission timed out");
      }
      poll(nullptr, 0, 10);
    }
  }

  void exchange_binding() {
    const auto deadline =
        std::chrono::steady_clock::now() + kProductExecIngressHostDeadline;
    std::string buffered;
    const std::string request = quorum_read_line(
        connection_.get(), buffered, 4096, deadline,
        "product ExecIngress request timeout");
    const std::size_t first = request.find('\t');
    const std::size_t second =
        first == std::string::npos ? std::string::npos : request.find('\t', first + 1);
    if (first == std::string::npos || second == std::string::npos ||
        request.substr(0, first) != "LOOM_EXEC_INGRESS/1") {
      throw Error("product ExecIngress request malformed");
    }
    const std::string event_sha = request.substr(first + 1, second - first - 1);
    std::string command_sha = request.substr(second + 1);
    if (!lowercase_digest(event_sha) || !lowercase_digest(command_sha)) {
      throw Error("product ExecIngress request digest malformed");
    }
    event_sha256_ = event_sha;
    command_sha256_ = command_sha;
    if (on_bound_) on_bound_(observation());
    if (reply_ == ProductIngressReply::CloseWithoutReply) {
      connection_ = UniqueFd();
      return;
    }
    if (reply_ == ProductIngressReply::BindingMismatch) {
      command_sha.back() = command_sha.back() == '0' ? '1' : '0';
    }
    if (reply_ == ProductIngressReply::ResultExact ||
        reply_ == ProductIngressReply::ResultBindingMismatch ||
        reply_ == ProductIngressReply::ResultReceiptMismatch ||
        reply_ == ProductIngressReply::ResultManifestMismatch) {
      if (!on_result_) throw Error("product ExecResult callback absent");
      ProductIngressResult result = on_result_(observation());
      if (!lowercase_digest(result.receipt_sha256) ||
          !lowercase_digest(result.manifest_sha256) || result.handle.empty() ||
          result.handle.size() > 512 || result.receipt.size() > 1024) {
        throw Error("product ExecResult callback malformed");
      }
      if (reply_ == ProductIngressReply::ResultBindingMismatch) {
        command_sha.back() = command_sha.back() == '0' ? '1' : '0';
      } else if (reply_ == ProductIngressReply::ResultReceiptMismatch) {
        result.receipt_sha256.back() =
            result.receipt_sha256.back() == '0' ? '1' : '0';
      } else if (reply_ == ProductIngressReply::ResultManifestMismatch) {
        result.manifest_sha256.back() =
            result.manifest_sha256.back() == '0' ? '1' : '0';
      }
      quorum_write_all(
          connection_.get(),
          "LOOM_EXEC_RESULT/1\t" + event_sha + "\t" + command_sha + "\t" +
              result.handle + "\t" + result.receipt_sha256 + "\t" +
              product_hex_encode(result.receipt) + "\t" +
              result.manifest_sha256 + "\n",
          deadline);
      shutdown(connection_.get(), SHUT_WR);
      connection_ = UniqueFd();
      return;
    }
    quorum_write_all(connection_.get(),
                     "LOOM_EXEC_INGRESS_BOUND/1\t" + event_sha + "\t" +
                         command_sha + "\n",
                     deadline);
    shutdown(connection_.get(), SHUT_WR);
    connection_ = UniqueFd();
  }

  void close_input() { input_ = UniqueFd(); }

  void collect_client() {
    const auto deadline =
        std::chrono::steady_clock::now() + kProductExecIngressHostDeadline;
    output_text_ = product_read_output(output_.get(), deadline);
    output_ = UniqueFd();
    while (!client_reaped_) {
      observe_client();
      if (client_reaped_) break;
      if (std::chrono::steady_clock::now() >= deadline) {
        throw Error("product systemd client exit timeout");
      }
      poll(nullptr, 0, 10);
    }
    if (WIFEXITED(client_status_)) {
      hook_code_ = WEXITSTATUS(client_status_);
    } else if (WIFSIGNALED(client_status_)) {
      hook_code_ = 128 + WTERMSIG(client_status_);
    }
    client_pid_ = -1;
  }

  void collect_client_with_diagnostics() {
    try {
      collect_client();
    } catch (const Error& error) {
      throw Error(std::string(error.what()) +
                  " provider_daemon_log=" +
                  product_compact_diagnostic(
                      optional_provider_log("daemon.log")) +
                  " provider_guardian_log=" +
                  product_compact_diagnostic(
                      optional_provider_log("guardian.log")));
    }
  }

  std::string optional_log(const std::string& name) const {
    const std::string path = runtime_path_ + "/" + name;
    if (!std::filesystem::exists(path)) return "";
    return read_file(path);
  }

  std::string optional_provider_log(const std::string& name) const {
    if (provider_fixture_.empty()) return "";
    const std::string path =
        runtime_path_ + "/provider-state/sessions/" +
        "provider-host-canary--provider-" + nonce_ + "/" + name;
    if (!std::filesystem::exists(path)) return "";
    return read_file(path);
  }

  ProductIngressHostRun observation() const {
    ProductIngressHostRun result;
    result.unit = unit_;
    result.runtime_directory = runtime_directory_;
    result.output = output_text_;
    result.ingress_log = optional_log("ingress.tsv");
    result.hook_log = optional_log("hook.tsv");
    result.provider_daemon_log = optional_provider_log("daemon.log");
    result.provider_guardian_log = optional_provider_log("guardian.log");
    result.hook_code = hook_code_;
    result.main_pid = main_pid_;
    result.main_uid = main_uid_;
    result.main_gid = main_gid_;
    result.main_start_tick = main_start_tick_;
    result.descriptor_peer_pid = descriptor_peer_pid_;
    result.descriptor_peer_parent_pid = descriptor_peer_parent_pid_;
    result.descriptor_peer_uid = descriptor_peer_uid_;
    result.descriptor_peer_gid = descriptor_peer_gid_;
    result.descriptor_peer_start_tick = descriptor_peer_start_tick_;
    result.identity_authenticated = identity_authenticated_;
    result.dynamic_user = dynamic_user_;
    result.raw_event_sha256 = raw_event_sha256_;
    result.event_sha256 = event_sha256_;
    result.command_sha256 = command_sha256_;
    return result;
  }

  void cleanup_noexcept() {
    input_ = UniqueFd();
    output_ = UniqueFd();
    connection_ = UniqueFd();
    listener_ = UniqueFd();
    if (client_pid_ > 0 && !client_reaped_) {
      kill(client_pid_, SIGKILL);
      while (waitpid(client_pid_, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
    client_pid_ = -1;
    try {
      static_cast<void>(quorum_run_tool(systemctl_, {"stop", unit_},
                                        std::chrono::seconds(2)));
      static_cast<void>(quorum_run_tool(systemctl_, {"reset-failed", unit_},
                                        std::chrono::seconds(2)));
    } catch (...) {
    }
    unlink(socket_path_.c_str());
    std::error_code ignored;
    std::filesystem::remove_all(runtime_path_, ignored);
  }

  std::string product_root_;
  std::string product_runtime_;
  std::string language_runtime_;
  std::string resident_runtime_;
  std::string systemd_run_;
  std::string systemctl_;
  std::string label_;
  std::string command_;
  bool dynamic_user_ = true;
  bool descriptor_present_ = true;
  ProductIngressReply reply_ = ProductIngressReply::Exact;
  ProductIngressExchangeExpectation exchange_expectation_ =
      ProductIngressExchangeExpectation::Request;
  ProductIngressBoundCallback on_bound_;
  std::string result_event_sha256_;
  ProductIngressResultCallback on_result_;
  std::string provider_fixture_;
  std::string nonce_;
  std::string unit_;
  std::string runtime_directory_;
  std::string runtime_path_;
  std::string socket_path_;
  std::string cgroup_sha256_;
  std::string descriptor_peer_cgroup_sha256_;
  std::string output_text_;
  UniqueFd listener_;
  UniqueFd connection_;
  UniqueFd input_;
  UniqueFd output_;
  UniqueFd pidfd_;
  pid_t client_pid_ = -1;
  pid_t main_pid_ = 0;
  uid_t main_uid_ = 0;
  gid_t main_gid_ = 0;
  std::uint64_t main_start_tick_ = 0;
  pid_t descriptor_peer_pid_ = 0;
  pid_t descriptor_peer_parent_pid_ = 0;
  uid_t descriptor_peer_uid_ = 0;
  gid_t descriptor_peer_gid_ = 0;
  std::uint64_t descriptor_peer_start_tick_ = 0;
  int client_status_ = 0;
  int hook_code_ = 255;
  bool client_reaped_ = false;
  bool identity_authenticated_ = false;
  std::string raw_event_sha256_;
  std::string event_sha256_;
  std::string command_sha256_;
};

bool product_contains(const std::string& value, std::string_view marker) {
  return value.find(marker) != std::string::npos;
}

std::string product_case_diagnostic(const ProductIngressHostRun& run) {
  return " hook_code=" + std::to_string(run.hook_code) +
         " main_pid=" + std::to_string(run.main_pid) +
         " main_uid=" + std::to_string(run.main_uid) +
         " main_gid=" + std::to_string(run.main_gid) +
         " descriptor_peer_pid=" +
         std::to_string(run.descriptor_peer_pid) +
         " descriptor_peer_uid=" +
         std::to_string(run.descriptor_peer_uid) +
         " identity_authenticated=" +
         (run.identity_authenticated ? "true" : "false") +
         " output=" + product_compact_diagnostic(run.output) +
         " ingress_log=" + product_compact_diagnostic(run.ingress_log) +
         " hook_log=" + product_compact_diagnostic(run.hook_log) +
         " provider_daemon_log=" +
         product_compact_diagnostic(run.provider_daemon_log) +
         " provider_guardian_log=" +
         product_compact_diagnostic(run.provider_guardian_log);
}

void require_product_case(bool condition, std::string_view reason,
                          const ProductIngressHostRun& run) {
  if (!condition) {
    throw Error(std::string(reason) + product_case_diagnostic(run));
  }
}

int selftest_product_exec_ingress_host(
    const std::string& product_root, const std::string& product_runtime,
    const std::string& language_runtime, const std::string& resident_runtime,
    const std::string& systemd_run, const std::string& systemctl) {
  if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0 ||
      trim(read_file("/proc/1/comm")) != "systemd") {
    throw Error("product ExecIngress host selftest requires root under PID1 systemd");
  }
  const std::string canonical_root = canonical_path(product_root);
  const std::string sentinel_prefix = "/run/loom-product-never-executed-" +
                                      random_generation_sha256().substr(0, 16);
  auto run = [&](std::string label, std::string command, bool dynamic_user,
                 bool descriptor_present, ProductIngressReply reply,
                 ProductIngressExchangeExpectation exchange_expectation =
                     ProductIngressExchangeExpectation::Request) {
    ProductExecIngressHostCase test(
        canonical_root, product_runtime, language_runtime, resident_runtime,
        systemd_run, systemctl, std::move(label), std::move(command),
        dynamic_user, descriptor_present, reply, exchange_expectation);
    return test.execute();
  };

  const ProductIngressHostRun treatment =
      run("current_material", "/usr/bin/touch " + sentinel_prefix,
          true, true, ProductIngressReply::Exact);
  require_product_case(
      treatment.hook_code == 0 && treatment.identity_authenticated &&
          treatment.dynamic_user && treatment.main_uid != 0 &&
          treatment.main_gid != 0 &&
          treatment.descriptor_peer_pid == treatment.main_pid &&
          treatment.descriptor_peer_parent_pid == 1 &&
          treatment.descriptor_peer_start_tick == treatment.main_start_tick &&
          treatment.descriptor_peer_uid == 0 &&
          treatment.descriptor_peer_gid == 0 &&
          product_contains(treatment.ingress_log, "decision=DENY\t") &&
          product_contains(treatment.ingress_log,
                           "reason=descriptor-bound-action-9031\t") &&
          product_contains(treatment.ingress_log,
                           "descriptor_present=true\tdescriptor_bound=true\t") &&
          product_contains(treatment.ingress_log,
                           "process_pid=" +
                               std::to_string(treatment.main_pid) + "\t") &&
          product_contains(treatment.ingress_log,
                           "process_uid=" +
                               std::to_string(treatment.main_uid) + "\t") &&
          product_contains(treatment.ingress_log,
                           "process_euid=" +
                               std::to_string(treatment.main_uid) + "\t") &&
          product_contains(treatment.ingress_log,
                           "process_gid=" +
                               std::to_string(treatment.main_gid) + "\t") &&
          product_contains(treatment.ingress_log,
                           "process_egid=" +
                               std::to_string(treatment.main_gid) + "\t") &&
          product_contains(treatment.ingress_log,
                           "peer_distinct_uid=true\t") &&
          product_contains(treatment.ingress_log,
                           "decision_authority=Sounio\t") &&
          product_contains(treatment.ingress_log,
                           "producing_language=Sounio\t") &&
          product_contains(treatment.ingress_log,
                           "production_activation=false\t") &&
          !std::filesystem::exists(sentinel_prefix),
      "dynamic-user treatment did not reach the frozen Sounio denial",
      treatment);

  const ProductIngressHostRun authority_sabotage =
      run("seal", "/usr/bin/touch " + sentinel_prefix + "-allow", true, true,
          ProductIngressReply::Exact);
  require_product_case(
      authority_sabotage.hook_code == 2 &&
          authority_sabotage.identity_authenticated &&
          product_contains(authority_sabotage.ingress_log, "decision=ALLOW\t") &&
          product_contains(authority_sabotage.output,
                           "product-exec-ingress-dark-unexpected-allow") &&
          !std::filesystem::exists(sentinel_prefix + "-allow"),
      "Sounio authority sabotage did not causally refuse at ExecIngress",
      authority_sabotage);

  const ProductIngressHostRun binding_sabotage =
      run("current_material", "/usr/bin/touch " + sentinel_prefix + "-bind",
          true, true, ProductIngressReply::BindingMismatch);
  require_product_case(
      binding_sabotage.hook_code == 2 &&
          binding_sabotage.identity_authenticated &&
          product_contains(binding_sabotage.output,
                           "product-exec-ingress-response-binding-mismatch") &&
          !std::filesystem::exists(sentinel_prefix + "-bind"),
      "descriptor binding sabotage was not refused", binding_sabotage);

  const ProductIngressHostRun guardian_death =
      run("current_material", "/usr/bin/touch " + sentinel_prefix + "-death",
          true, true, ProductIngressReply::CloseWithoutReply);
  require_product_case(
      guardian_death.hook_code == 2 && guardian_death.identity_authenticated &&
          product_contains(guardian_death.output,
                           "product-exec-ingress-response-eof") &&
          !std::filesystem::exists(sentinel_prefix + "-death"),
      "HostGuardian death did not fail closed", guardian_death);

  const ProductIngressHostRun same_uid =
      run("current_material", "/usr/bin/touch " + sentinel_prefix + "-same",
          false, true, ProductIngressReply::Exact,
          ProductIngressExchangeExpectation::RefusalBeforeRequest);
  require_product_case(
      same_uid.hook_code == 2 && same_uid.identity_authenticated &&
          !same_uid.dynamic_user && same_uid.main_uid == 0 &&
          product_contains(same_uid.output,
                           "product-exec-ingress-peer-not-distinct") &&
          product_contains(same_uid.ingress_log, "decision=DENY\t") &&
          product_contains(same_uid.ingress_log,
                           "reason=peer-not-distinct\t") &&
          !std::filesystem::exists(sentinel_prefix + "-same"),
      "same-UID host control was not refused", same_uid);

  const ProductIngressHostRun missing =
      run("current_material", "/usr/bin/touch " + sentinel_prefix + "-missing",
          true, false, ProductIngressReply::Exact);
  require_product_case(
      missing.hook_code == 2 && !missing.identity_authenticated &&
          missing.main_pid == 0 && missing.descriptor_peer_pid == 0 &&
          product_contains(missing.output,
                           "product-exec-ingress-descriptor-absent") &&
          product_contains(missing.ingress_log, "reason=descriptor-absent\t") &&
          !std::filesystem::exists(sentinel_prefix + "-missing"),
      "descriptor absence did not fail closed", missing);

  const ProductIngressHostRun python =
      run("current_material",
          "/usr/bin/python3 --version > " + sentinel_prefix + "-python", true,
          true, ProductIngressReply::Exact);
  const ProductIngressHostRun rust =
      run("current_material",
          "/usr/bin/rustc --version > " + sentinel_prefix + "-rust", true,
          true, ProductIngressReply::Exact);
  require_product_case(
      python.hook_code == 0 && rust.hook_code == 0 &&
          python.identity_authenticated && rust.identity_authenticated &&
          product_contains(python.ingress_log, "decision_authority=Sounio\t") &&
          product_contains(rust.ingress_log, "decision_authority=Sounio\t") &&
          !std::filesystem::exists(sentinel_prefix + "-python") &&
          !std::filesystem::exists(sentinel_prefix + "-rust"),
      "prohibited oracle attempt crossed the dark product ingress", python);

  std::cout
      << "LOOM_PRODUCT_EXEC_INGRESS_DYNAMIC_USER_HOST_GATE PASS"
      << " semantic_authority=Sounio action=9031"
      << " operational_attachment=OCaml"
      << " material_layer=C++20+Linux+systemd"
      << " material_role=MATERIAL_PARITY transitory=true"
      << " hostguardian=root-owned-broker service_manager=PID1-systemd"
      << " descriptor_connector=systemd-root-executor"
      << " descriptor_connector_authenticated=true dynamic_user=true"
      << " lane_cell_canary_attached=true"
      << " lane_cell_uid=" << treatment.main_uid
      << " lane_cell_gid=" << treatment.main_gid
      << " lane_cell_pidfd=bound lane_cell_start_tick=bound"
      << " lane_cell_executable=bound lane_cell_cgroup=bound"
      << " lane_cell_user_namespace=host"
      << " inherited_descriptor=true descriptor_open=systemd-OpenFile"
      << " descriptor_fd=3 descriptor_peer_pid=lane-main"
      << " descriptor_peer_parent_pid=1 descriptor_peer_uid=0"
      << " descriptor_peer_gid=0 connector_start_tick=lane-main"
      << " connector_cgroup=lane-main"
      << " connector_credential_transition=root-to-DynamicUser"
      << " descriptor_is_bearer=false event_hash=bound command_hash=bound"
      << " treatment=Sounio-DENY+hook-continues"
      << " sounio_allow_sabotage=hook-refused"
      << " binding_sabotage=refused guardian_death=refused"
      << " same_uid=refused missing_descriptor=refused"
      << " python_oracle=refused-before-execution"
      << " rust_oracle=refused-before-execution"
      << " causal_sabotage=PASS command_executed=false"
      << " distinct_uid_product_broker_canary=true"
      << " fleet_lane_cell_attached=false exec_cell_attached=false"
      << " material_execution=false production_activation=false"
      << " launch_open=false recycle_open=false exec_attached=false"
      << " commit_attached=false ci_attached=false parity_open=false"
      << " claim_ready=false"
      << " product_runtime_sha256=" << file_sha256(product_runtime)
      << " language_runtime_sha256=" << file_sha256(language_runtime)
      << " resident_runtime_sha256=" << file_sha256(resident_runtime)
      << " systemd_run_sha256=" << file_sha256(systemd_run)
      << " systemctl_sha256=" << file_sha256(systemctl) << "\n";
  return 0;
}
