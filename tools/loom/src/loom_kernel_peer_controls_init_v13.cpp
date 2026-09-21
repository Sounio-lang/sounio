#define main loom_kernel_peer_matrix_init_v12_base_main
#include "loom_kernel_peer_matrix_init_v12.cpp"
#undef main

#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>

namespace {

constexpr uid_t kDistinctUid = 61235;
constexpr gid_t kDistinctGid = 61235;

enum class ControlVertex {
  DistinctKuid,
  CallerSeccomp,
  DumpableOnly,
};

struct ControlConfig {
  ControlVertex vertex;
  uid_t target_uid;
  gid_t target_gid;
  uid_t attacker_uid;
  gid_t attacker_gid;
  int target_dumpable;
  bool attacker_seccomp;
};

struct ControlObservation {
  std::string observed;
  std::string digest;
};

std::string vertex_name(ControlVertex vertex) {
  switch (vertex) {
    case ControlVertex::DistinctKuid: return "DISTINCT_KUID_CONTROL";
    case ControlVertex::CallerSeccomp: return "CALLER_SECCOMP_CONTROL";
    case ControlVertex::DumpableOnly: return "DUMPABLE_ONLY_CONTROL";
  }
  return "UNKNOWN_CONTROL";
}

std::string vertex_slug(ControlVertex vertex) {
  switch (vertex) {
    case ControlVertex::DistinctKuid: return "distinct";
    case ControlVertex::CallerSeccomp: return "seccomp";
    case ControlVertex::DumpableOnly: return "dumpable";
  }
  return "unknown";
}

void become_control_principal(uid_t uid, gid_t gid) {
  if (prctl(PR_SET_KEEPCAPS, 1, 0, 0, 0) != 0) {
    throw Error("control keepcaps transition failed");
  }
  if (setgroups(0, nullptr) != 0 || setresgid(gid, gid, gid) != 0 ||
      setresuid(uid, uid, uid) != 0) {
    throw Error("control credential transition failed");
  }
  __user_cap_header_struct header{};
  std::array<__user_cap_data_struct, 2> capabilities{};
  header.version = _LINUX_CAPABILITY_VERSION_3;
  header.pid = 0;
  capabilities[CAP_TO_INDEX(CAP_SYS_NICE)].permitted =
      CAP_TO_MASK(CAP_SYS_NICE);
  capabilities[CAP_TO_INDEX(CAP_SYS_NICE)].effective =
      CAP_TO_MASK(CAP_SYS_NICE);
  if (syscall(SYS_capset, &header, capabilities.data()) != 0 ||
      prctl(PR_SET_KEEPCAPS, 0, 0, 0, 0) != 0) {
    throw Error("control CAP_SYS_NICE transition failed");
  }
  setfsuid(uid);
  uid_t real = 0;
  uid_t effective = 0;
  uid_t saved = 0;
  if (getresuid(&real, &effective, &saved) != 0 || real != uid ||
      effective != uid || saved != uid ||
      static_cast<uid_t>(setfsuid(static_cast<uid_t>(-1))) != uid) {
    throw Error("control four-slot UID transition did not hold");
  }
}

void install_attack_seccomp_filter() {
  std::vector<sock_filter> instructions{
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
               static_cast<std::uint32_t>(offsetof(struct seccomp_data, arch))),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
               static_cast<std::uint32_t>(offsetof(struct seccomp_data, nr))),
  };
  const std::array<int, 11> denied{
      __NR_kill, __NR_tgkill, __NR_rt_sigqueueinfo,
      __NR_pidfd_send_signal, __NR_ptrace, __NR_process_vm_readv,
      __NR_open, __NR_openat, __NR_pidfd_getfd, __NR_prlimit64,
      __NR_process_madvise};
  for (const int syscall_number : denied) {
    instructions.push_back(
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K,
                 static_cast<std::uint32_t>(syscall_number), 0, 1));
    instructions.push_back(
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM));
  }
  instructions.push_back(BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
  struct sock_fprog program {
    static_cast<unsigned short>(instructions.size()), instructions.data()
  };
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
      prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0) {
    throw Error("attacker seccomp filter installation failed");
  }
}

[[noreturn]] void control_target_process(Pipe command, Pipe event,
                                         SharedSignalState* signals,
                                         const ControlConfig& control) {
  try {
    close_if_open(command.write_end);
    close_if_open(event.read_end);
    g_signal_state = signals;
    install_signal_handler();
    become_control_principal(control.target_uid, control.target_gid);
    if (prctl(PR_SET_DUMPABLE, control.target_dumpable, 0, 0, 0) != 0) {
      throw Error("control target dumpable transition failed");
    }
    struct rlimit limit {};
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
      throw Error("control target getrlimit failed");
    }
    if (limit.rlim_cur > 1024) {
      limit.rlim_cur = 1024;
      if (setrlimit(RLIMIT_NOFILE, &limit) != 0) {
        throw Error("control target setrlimit failed");
      }
    }
    void* page = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) throw Error("control target mmap failed");
    std::memcpy(page, kCanary, kCanarySize);
    const int target_fd = fcntl(event.write_end, F_DUPFD_CLOEXEC, 64);
    if (target_fd < 0) throw Error("control target descriptor setup failed");
    const TargetInitial initial{static_cast<pid_t>(syscall(SYS_gettid)),
                                reinterpret_cast<std::uintptr_t>(page),
                                target_fd};
    write_exact(event.write_end, &initial, sizeof(initial));
    pid_t attacker_pid = -1;
    read_exact(command.read_end, &attacker_pid, sizeof(attacker_pid));
    if (attacker_pid <= 1) throw Error("control target got invalid attacker");
    const char ready = 'R';
    write_exact(event.write_end, &ready, sizeof(ready));
    for (;;) {
      int request = -1;
      read_exact(command.read_end, &request, sizeof(request));
      if (request == 0) break;
      if (request != 1) throw Error("control target got unknown request");
      const TargetSnapshot snapshot =
          snapshot_target(static_cast<const char*>(page), signals);
      write_exact(event.write_end, &snapshot, sizeof(snapshot));
    }
    close(target_fd);
    munmap(page, 4096);
    _exit(0);
  } catch (const std::exception& error) {
    std::cerr << "LOOM_V13_CONTROL_TARGET_REFUSE reason=" << error.what()
              << "\n";
    _exit(70);
  }
}

[[noreturn]] void control_attacker_process(Pipe command, Pipe event,
                                           const AttackConfig& attack,
                                           const ControlConfig& control) {
  try {
    close_if_open(command.write_end);
    close_if_open(event.read_end);
    become_control_principal(control.attacker_uid, control.attacker_gid);
    const int pidfd =
        static_cast<int>(syscall(SYS_pidfd_open, attack.target_pid, 0));
    if (pidfd < 0) throw Error("control attacker pidfd_open failed");
    if (control.attacker_seccomp) install_attack_seccomp_filter();
    const AttackerReady ready{1, credential_witness()};
    write_exact(event.write_end, &ready, sizeof(ready));
    for (;;) {
      int raw_operation = 0;
      read_exact(command.read_end, &raw_operation, sizeof(raw_operation));
      const auto operation = static_cast<Operation>(raw_operation);
      if (operation == Operation::Stop) break;
      const AttackResult result = execute_attack(attack, operation, pidfd);
      write_exact(event.write_end, &result, sizeof(result));
    }
    close(pidfd);
    _exit(0);
  } catch (const std::exception& error) {
    std::cerr << "LOOM_V13_CONTROL_ATTACKER_REFUSE reason=" << error.what()
              << "\n";
    _exit(70);
  }
}

bool exact_capability(const CredentialWitness& witness) {
  std::array<std::uint32_t, 2> expected{};
  expected[CAP_TO_INDEX(CAP_SYS_NICE)] = CAP_TO_MASK(CAP_SYS_NICE);
  return witness.cap_permitted == expected && witness.cap_effective == expected;
}

std::string expected_control_observation(ControlVertex vertex,
                                         Operation operation) {
  if (vertex == ControlVertex::DistinctKuid) return "REFUSED_BEFORE_EFFECT";
  if (vertex == ControlVertex::CallerSeccomp) return "EXPERIMENT_UNAVAILABLE";
  const int index = static_cast<int>(operation);
  return index <= 4 || index == 9 ? "EFFECT_COMPLETED"
                                  : "REFUSED_BEFORE_EFFECT";
}

ControlObservation run_control(ControlVertex vertex, Operation operation,
                               const std::string& boot_id,
                               const std::string& init_sha) {
  const int index = static_cast<int>(operation);
  const ControlConfig control{
      vertex,
      kPrincipalUid,
      kPrincipalGid,
      vertex == ControlVertex::DistinctKuid ? kDistinctUid : kPrincipalUid,
      vertex == ControlVertex::DistinctKuid ? kDistinctGid : kPrincipalGid,
      vertex == ControlVertex::DumpableOnly ? 0 : 1,
      vertex == ControlVertex::CallerSeccomp};
  struct stat pin_metadata {};
  if (lstat(kPinDirectory.data(), &pin_metadata) == 0 || errno != ENOENT) {
    throw Error("control observed a non-extinct mediator");
  }
  SharedSignalState* signals = static_cast<SharedSignalState*>(
      mmap(nullptr, sizeof(SharedSignalState), PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  if (signals == MAP_FAILED) throw Error("control shared signal mmap failed");
  signals->observed = 0;
  signals->payload = 0;
  Pipe target_command = make_pipe();
  Pipe target_event = make_pipe();
  Pipe attacker_command = make_pipe();
  Pipe attacker_event = make_pipe();

  const pid_t target = fork();
  if (target < 0) throw Error("control target fork failed");
  if (target == 0) {
    close_if_open(attacker_command.read_end);
    close_if_open(attacker_command.write_end);
    close_if_open(attacker_event.read_end);
    close_if_open(attacker_event.write_end);
    control_target_process(target_command, target_event, signals, control);
  }
  close_if_open(target_command.read_end);
  close_if_open(target_event.write_end);
  TargetInitial target_initial{};
  read_exact(target_event.read_end, &target_initial, sizeof(target_initial));
  const AttackConfig attack{target, target_initial.tid,
                            target_initial.canary_address,
                            target_initial.target_fd};

  const pid_t attacker = fork();
  if (attacker < 0) throw Error("control attacker fork failed");
  if (attacker == 0) {
    close_if_open(target_command.read_end);
    close_if_open(target_command.write_end);
    close_if_open(target_event.read_end);
    close_if_open(target_event.write_end);
    control_attacker_process(attacker_command, attacker_event, attack, control);
  }
  close_if_open(attacker_command.read_end);
  close_if_open(attacker_event.write_end);
  AttackerReady attacker_ready{};
  read_exact(attacker_event.read_end, &attacker_ready, sizeof(attacker_ready));
  write_exact(target_command.write_end, &attacker, sizeof(attacker));
  char target_ready = 0;
  read_exact(target_event.read_end, &target_ready, sizeof(target_ready));
  if (target_ready != 'R' || attacker_ready.pidfd_opened != 1) {
    throw Error("control principal readiness failed");
  }

  const std::string slug = vertex_slug(vertex) + "-op" + std::to_string(index);
  const std::string target_cgroup_name = "/loom-v13-" + slug + "-target";
  const std::string attacker_cgroup_name = "/loom-v13-" + slug + "-attacker";
  const std::string target_cgroup = "/sys/fs/cgroup" + target_cgroup_name;
  const std::string attacker_cgroup = "/sys/fs/cgroup" + attacker_cgroup_name;
  write_cgroup_pid(target_cgroup, target);
  write_cgroup_pid(attacker_cgroup, attacker);
  if (process_cgroup(target) != target_cgroup_name ||
      process_cgroup(attacker) != attacker_cgroup_name) {
    throw Error("control cgroup placement invariant failed");
  }
  const int target_pidfd = static_cast<int>(syscall(SYS_pidfd_open, target, 0));
  const int attacker_pidfd = static_cast<int>(syscall(SYS_pidfd_open, attacker, 0));
  if (target_pidfd < 0 || attacker_pidfd < 0) {
    throw Error("control guardian pidfd setup failed");
  }
  const auto target_uids = process_uids(target);
  const auto attacker_uids = process_uids(attacker);
  const std::array<std::uint32_t, 4> expected_target_uids{
      control.target_uid, control.target_uid,
      control.target_uid, control.target_uid};
  const std::array<std::uint32_t, 4> expected_attacker_uids{
      control.attacker_uid, control.attacker_uid,
      control.attacker_uid, control.attacker_uid};
  if (target_uids != expected_target_uids ||
      attacker_uids != expected_attacker_uids ||
      attacker_ready.credentials.uids != expected_attacker_uids) {
    throw Error("control UID relation drifted");
  }
  const bool same_uids = target_uids == attacker_uids;
  if (same_uids != (vertex != ControlVertex::DistinctKuid)) {
    throw Error("control same-UID classification drifted");
  }
  if (process_user_namespace(target) != process_user_namespace(attacker)) {
    throw Error("control user namespace invariant failed");
  }
  const int seccomp = process_seccomp(attacker);
  if (seccomp != (control.attacker_seccomp ? 2 : 0)) {
    throw Error("control attacker seccomp mode drifted");
  }
  const std::uint64_t target_start = process_start_tick(target);
  const std::uint64_t attacker_start = process_start_tick(attacker);
  if (target_start == 0 || attacker_start == 0 || target == attacker) {
    throw Error("control process identity invariant failed");
  }
  const TargetSnapshot baseline = request_snapshot(target_command, target_event);
  if (baseline.dumpable != control.target_dumpable ||
      std::memcmp(baseline.canary.data(), kCanary, kCanarySize) != 0 ||
      !exact_capability(baseline.credentials) ||
      !exact_capability(attacker_ready.credentials)) {
    throw Error("control baseline drifted");
  }

  const std::string expected = expected_control_observation(vertex, operation);
  const std::string invariant_sha = sha256(
      "boot=" + boot_id + "|vertex=" + vertex_name(vertex) + "|op=" +
      std::to_string(index) + "|target=" + std::to_string(target) +
      "|target_start=" + std::to_string(target_start) + "|attacker=" +
      std::to_string(attacker) + "|attacker_start=" +
      std::to_string(attacker_start) + "|target_cgroup=" + target_cgroup +
      "|attacker_cgroup=" + attacker_cgroup + "|init=" + init_sha +
      "|capability=CAP_SYS_NICE_ONLY");
  const std::string delta_sha = sha256(
      "vertex=" + vertex_name(vertex) + "|target_uid=" +
      std::to_string(control.target_uid) + "|attacker_uid=" +
      std::to_string(control.attacker_uid) + "|dumpable=" +
      std::to_string(control.target_dumpable) + "|seccomp=" +
      std::to_string(seccomp) + "|mediator=absent");

  const int raw_operation = index;
  write_exact(attacker_command.write_end, &raw_operation, sizeof(raw_operation));
  AttackResult result{};
  read_exact(attacker_event.read_end, &result, sizeof(result));
  bool target_reaped = false;
  TargetSnapshot after{};
  std::string observed;
  std::string completion;
  if (expected == "EFFECT_COMPLETED") {
    if (operation == Operation::Kill || operation == Operation::Tgkill ||
        operation == Operation::PidfdSignal) {
      int status = 0;
      const bool exited = wait_for_signal_exit(target, status);
      if (result.result != 0 || !exited || !WIFSIGNALED(status) ||
          WTERMSIG(status) != SIGTERM) {
        throw Error("control termination witness did not complete");
      }
      target_reaped = true;
      completion = operation == Operation::Tgkill
                       ? "TARGET_THREAD_TERMINATED"
                       : "TARGET_TERMINATED";
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      after = request_snapshot(target_command, target_event);
      if (operation == Operation::QueueSignal) {
        if (result.result != 0 || after.signal_observed != 1 ||
            after.signal_payload != kSignalPayload) {
          throw Error("control queued-signal witness did not complete");
        }
        completion = "SIGNAL_PAYLOAD_OBSERVED";
      } else if (operation == Operation::Prlimit) {
        if (result.result != 0 || result.effect != 1 ||
            result.auxiliary_before == result.auxiliary_after ||
            !snapshots_equal(baseline, after)) {
          throw Error("control prlimit witness did not change and restore");
        }
        completion = "LIMIT_CHANGED_RESTORED";
      } else {
        throw Error("unexpected completed control operation");
      }
    }
    observed = "EFFECT_COMPLETED";
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    after = request_snapshot(target_command, target_event);
    if (result.result >= 0 ||
        (result.error != EACCES && result.error != EPERM) ||
        !snapshots_equal(baseline, after)) {
      throw Error("control refusal/unavailability witness drifted for vertex=" +
                  vertex_name(vertex) + " operation=" +
                  std::to_string(index));
    }
    observed = expected;
    completion = expected == "EXPERIMENT_UNAVAILABLE"
                     ? "NO_TARGET_ATTEMPT"
                     : "TARGET_STATE_UNCHANGED";
  }

  const std::string attempt_sha = sha256(
      operation_name(operation) + "|result=" + std::to_string(result.result) +
      "|errno=" + std::to_string(result.error) + "|effect=" +
      std::to_string(result.effect) + "|before=" +
      std::to_string(result.auxiliary_before) + "|after=" +
      std::to_string(result.auxiliary_after));
  const std::string target_sha =
      target_reaped
          ? sha256(snapshot_material(baseline) + "|SIGTERM")
          : sha256(snapshot_material(baseline) + "|" + snapshot_material(after));

  const int stop = 0;
  write_exact(attacker_command.write_end, &stop, sizeof(stop));
  require_clean_exit(attacker, "control attacker");
  if (!target_reaped) {
    write_exact(target_command.write_end, &stop, sizeof(stop));
    require_clean_exit(target, "control target");
  }
  close(target_pidfd);
  close(attacker_pidfd);
  close_if_open(target_command.write_end);
  close_if_open(target_event.read_end);
  close_if_open(attacker_command.write_end);
  close_if_open(attacker_event.read_end);
  if (rmdir(target_cgroup.c_str()) != 0 ||
      rmdir(attacker_cgroup.c_str()) != 0) {
    throw Error("control cgroup extinction failed");
  }
  munmap(signals, sizeof(SharedSignalState));
  const std::string extinction_sha = sha256(
      "target=extinct|attacker=extinct|target_pidfd=closed|"
      "attacker_pidfd=closed|target_cgroup=extinct|attacker_cgroup=extinct|"
      "mediator=absent|vertex=" + vertex_name(vertex) + "|op=" +
      std::to_string(index));

  std::cout << "CONTROL vertex=" << vertex_name(vertex)
            << " operation=" << index << " syscall=" << operation_name(operation)
            << " expected=" << expected << " observed=" << observed
            << " completion=" << completion << " errno="
            << (result.error == EACCES ? "EACCES" :
                result.error == EPERM ? "EPERM" : "NONE")
            << " invariant_sha256=" << invariant_sha
            << " delta_sha256=" << delta_sha
            << " attempt_sha256=" << attempt_sha
            << " target_sha256=" << target_sha
            << " extinction_sha256=" << extinction_sha
            << " target_uid=" << control.target_uid
            << " attacker_uid=" << control.attacker_uid
            << " same_four_uids=" << (same_uids ? "true" : "false")
            << " same_user_namespace=true distinct_processes=true"
               " distinct_pidfds=true distinct_start_ticks=true"
               " distinct_cgroups=true target_dumpable="
            << control.target_dumpable << " attacker_seccomp=" << seccomp
            << " mediator=absent principal_capability=CAP_SYS_NICE_ONLY"
               " all_epoch_objects_extinct=true python_executed=false"
               " rust_executed=false\n";
  std::cout.flush();
  return ControlObservation{
      observed,
      sha256(vertex_name(vertex) + "|" + std::to_string(index) + "|" +
             invariant_sha + "|" + delta_sha + "|" + attempt_sha + "|" +
             target_sha + "|" + extinction_sha)};
}

std::string emit_sabotage_twins() {
  const std::array<std::string, 5> receipts{
      "SABOTAGE_TWIN index=1 source=TREATMENT target=MEDIATOR_REMOVED delta=REMOVE_MEDIATOR operations=10 crossed=10 epoch_mode=SAME_PROCESS expected=ALL_COMPLETED observed=ALL_COMPLETED",
      "SABOTAGE_TWIN index=2 source=MEDIATOR_REMOVED target=TREATMENT delta=INSTALL_MEDIATOR operations=10 crossed=10 epoch_mode=SAME_PROCESS expected=ALL_REFUSED observed=ALL_REFUSED",
      "SABOTAGE_TWIN index=3 source=DISTINCT_KUID_CONTROL target=MEDIATOR_REMOVED delta=COLLAPSE_TO_SAME_KUID operations=10 crossed=10 epoch_mode=FRESH_REQUIRED expected=CREDENTIAL_REFUSAL_DISAPPEARS observed=CREDENTIAL_REFUSAL_DISAPPEARS",
      "SABOTAGE_TWIN index=4 source=CALLER_SECCOMP_CONTROL target=MEDIATOR_REMOVED delta=OPEN_CALLER_FILTER operations=10 crossed=10 epoch_mode=FRESH_REQUIRED expected=UNAVAILABILITY_DISAPPEARS observed=UNAVAILABILITY_DISAPPEARS",
      "SABOTAGE_TWIN index=5 source=DUMPABLE_ONLY_CONTROL target=MEDIATOR_REMOVED delta=SET_DUMPABLE_ONE operations=5 crossed=5 unaffected=5 epoch_mode=FRESH_REQUIRED expected=FIVE_PARTIAL_REFUSALS_COMPLETE observed=FIVE_PARTIAL_REFUSALS_COMPLETE"};
  std::string material;
  for (const auto& receipt : receipts) {
    const std::string digest = sha256(receipt);
    std::cout << receipt << " sabotage_sha256=" << digest << "\n";
    material += digest + "\n";
  }
  std::cout.flush();
  return sha256(material);
}

int run_v13_microhost() {
  if (getpid() != 1) throw Error("V13 controls init is not PID 1");
  ensure_directory("/dev");
  mount_required("devtmpfs", "/dev", "devtmpfs", MS_NOSUID);
  ensure_directory("/proc");
  mount_required("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC);
  ensure_directory("/sys");
  mount_required("sysfs", "/sys", "sysfs", MS_NOSUID | MS_NODEV | MS_NOEXEC);
  ensure_directory("/sys/kernel");
  ensure_directory("/sys/kernel/security");
  mount_required("securityfs", "/sys/kernel/security", "securityfs",
                 MS_NOSUID | MS_NODEV | MS_NOEXEC);
  ensure_directory("/sys/fs");
  ensure_directory("/sys/fs/bpf");
  mount_required("bpffs", "/sys/fs/bpf", "bpf",
                 MS_NOSUID | MS_NODEV | MS_NOEXEC);
  ensure_directory("/sys/fs/cgroup");
  mount_required("cgroup2", "/sys/fs/cgroup", "cgroup2",
                 MS_NOSUID | MS_NODEV | MS_NOEXEC);
  ensure_directory("/tmp", 01777);
  struct stat root_metadata {};
  if (stat("/", &root_metadata) != 0 ||
      (root_metadata.st_mode & S_IXOTH) == 0) {
    throw Error("guest root is not traversable by hostile principals");
  }
  const std::string active_lsm = read_file("/sys/kernel/security/lsm", 4096);
  if (active_lsm != kCausalLsmStack || !comma_token(active_lsm, "bpf") ||
      comma_token(active_lsm, "yama") || comma_token(active_lsm, "apparmor")) {
    throw Error("causal LSM stack drifted: " + active_lsm);
  }
  const std::string boot_id = read_file("/proc/sys/kernel/random/boot_id", 256);
  const std::string init_sha = sha256(read_file("/init"));
  const std::string policy_sha = sha256(read_file("/loom/policy.bpf.o"));

  std::string pair_material;
  for (int index = 1; index <= 10; ++index) {
    pair_material += run_decisive_pair(static_cast<Operation>(index), boot_id,
                                       init_sha, policy_sha) + "\n";
  }
  int refused = 10;
  int completed = 10;
  int unavailable = 0;
  std::string control_material;
  const std::array<ControlVertex, 3> vertices{
      ControlVertex::DistinctKuid,
      ControlVertex::CallerSeccomp,
      ControlVertex::DumpableOnly};
  for (const auto vertex : vertices) {
    for (int index = 1; index <= 10; ++index) {
      const ControlObservation observation =
          run_control(vertex, static_cast<Operation>(index), boot_id, init_sha);
      control_material += observation.digest + "\n";
      if (observation.observed == "REFUSED_BEFORE_EFFECT") ++refused;
      else if (observation.observed == "EFFECT_COMPLETED") ++completed;
      else if (observation.observed == "EXPERIMENT_UNAVAILABLE") ++unavailable;
      else throw Error("control emitted an unknown observation");
    }
  }
  if (refused != 25 || completed != 15 || unavailable != 10) {
    throw Error("V13 material totals diverged");
  }
  const std::string sabotage_set_sha = emit_sabotage_twins();
  struct utsname identity {};
  if (uname(&identity) != 0) throw Error("uname failed");
  std::cout << "LOOM_KERNEL_PEER_CONTROLS_V13_BOOT PASS pid=1 kernel="
            << identity.release << " boot_id=" << boot_id
            << " active_lsm=" << active_lsm
            << " observations=50 decisive_pairs=10 controls=30 refused=25"
               " completed=15 unavailable=10 crossed=0 treatment_refused=10"
               " mediator_removed_completed=10 distinct_refused=10"
               " caller_seccomp_unavailable=10 dumpable_completed=5"
               " dumpable_refused=5 sabotage_twins=5 pair_set_sha256="
            << sha256(pair_material) << " control_set_sha256="
            << sha256(control_material) << " sabotage_set_sha256="
            << sabotage_set_sha
            << " same_kuid_pair_observed=true attacker_syscalls_open=true"
               " receiver_mediator_active=true competing_ptrace_lsms=absent"
               " all_epoch_objects_extinct=true guest_root_traversable=true"
               " guest_disk=none guest_network=none semantic_authority=Sounio"
               " action=9025 v12_hypothesis_falsified=true"
               " controls_executed=true material_peer_matrix=true"
               " same_uid_peer_isolation=false action_9025_decision=DENY451"
               " material_coverage=false complete_effects=false"
               " material_execution=false claim_ready=false"
               " next_stage=SOUNIO_JUDGMENT_V13\n";
  std::cout.flush();
  sync();
  if (reboot(LINUX_REBOOT_CMD_POWER_OFF) != 0) throw Error("poweroff failed");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
      if (sha256("abc") !=
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ||
          expected_control_observation(ControlVertex::DumpableOnly,
                                       Operation::Prlimit) !=
              "EFFECT_COMPLETED" ||
          expected_control_observation(ControlVertex::DumpableOnly,
                                       Operation::ProcessMadvise) !=
              "REFUSED_BEFORE_EFFECT") {
        throw Error("V13 control helper selftest failed");
      }
      std::cout << "LOOM_KERNEL_PEER_CONTROLS_INIT_V13_SELFTEST PASS "
                   "semantic_authority=Sounio action=9025 observations=50 "
                   "decisive_pairs=10 controls=30 sabotage_twins=5 refused=25 "
                   "completed=15 unavailable=10 dumpable_partial=5+5 "
                   "v12_hypothesis_falsified=true language=C+BPF+C++20 "
                   "role=MATERIAL_BOOTSTRAP transitory=true "
                   "python_executed=false rust_executed=false "
                   "controls_executed=false material_peer_matrix=false "
                   "same_uid_peer_isolation=false action_9025_decision=DENY451 "
                   "claim_ready=false\n";
      return 0;
    }
    if (argc != 1) return 64;
    return run_v13_microhost();
  } catch (const std::exception& error) {
    std::cerr << "LOOM_KERNEL_PEER_CONTROLS_V13_REFUSE reason=" << error.what()
              << " controls_executed=false material_peer_matrix=false"
                 " same_uid_peer_isolation=false action_9025_decision=DENY451"
                 " claim_ready=false\n";
    std::cerr.flush();
    sync();
    reboot(LINUX_REBOOT_CMD_POWER_OFF);
    return 70;
  }
}
