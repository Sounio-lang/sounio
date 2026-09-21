#define main loom_kernel_peer_matrix_init_v12_base_main
#include "loom_kernel_peer_matrix_init_v12.cpp"
#undef main

namespace {

[[noreturn]] void dumpable_zero_target(Pipe command, Pipe event,
                                       SharedSignalState* signals) {
  try {
    close_if_open(command.write_end);
    close_if_open(event.read_end);
    g_signal_state = signals;
    install_signal_handler();
    become_principal();
    if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0) {
      throw Error(std::string("target dumpable-zero transition failed: ") +
                  std::strerror(errno));
    }
    struct rlimit limit {};
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
      throw Error("target getrlimit failed");
    }
    if (limit.rlim_cur > 1024) {
      limit.rlim_cur = 1024;
      if (setrlimit(RLIMIT_NOFILE, &limit) != 0) {
        throw Error("target setrlimit failed");
      }
    }
    void* page = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) throw Error("target mmap failed");
    std::memcpy(page, kCanary, kCanarySize);
    const int target_fd = fcntl(event.write_end, F_DUPFD_CLOEXEC, 64);
    if (target_fd < 0) throw Error("target descriptor setup failed");
    const TargetInitial initial{static_cast<pid_t>(syscall(SYS_gettid)),
                                reinterpret_cast<std::uintptr_t>(page),
                                target_fd};
    write_exact(event.write_end, &initial, sizeof(initial));
    pid_t attacker_pid = -1;
    read_exact(command.read_end, &attacker_pid, sizeof(attacker_pid));
    if (attacker_pid <= 1) throw Error("target received invalid attacker identity");
    const char ready = 'R';
    write_exact(event.write_end, &ready, sizeof(ready));
    for (;;) {
      int request = -1;
      read_exact(command.read_end, &request, sizeof(request));
      if (request == 0) break;
      if (request != 1) throw Error("target received unknown request");
      const TargetSnapshot snapshot =
          snapshot_target(static_cast<const char*>(page), signals);
      write_exact(event.write_end, &snapshot, sizeof(snapshot));
    }
    close(target_fd);
    munmap(page, 4096);
    _exit(0);
  } catch (const std::exception& error) {
    std::cerr << "LOOM_V12_DUMPABLE_TARGET_REFUSE reason=" << error.what()
              << "\n";
    _exit(70);
  }
}

std::string run_dumpable_prlimit_counterexample(const std::string& boot_id,
                                                const std::string& init_sha) {
  struct stat pin_metadata {};
  if (lstat(kPinDirectory.data(), &pin_metadata) == 0 || errno != ENOENT) {
    throw Error("LOOM mediator is not affirmatively absent");
  }
  SharedSignalState* signals = static_cast<SharedSignalState*>(
      mmap(nullptr, sizeof(SharedSignalState), PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  if (signals == MAP_FAILED) throw Error("shared signal mmap failed");
  signals->observed = 0;
  signals->payload = 0;
  Pipe target_command = make_pipe();
  Pipe target_event = make_pipe();
  Pipe attacker_command = make_pipe();
  Pipe attacker_event = make_pipe();

  const pid_t target = fork();
  if (target < 0) throw Error("target fork failed");
  if (target == 0) dumpable_zero_target(target_command, target_event, signals);
  close_if_open(target_command.read_end);
  close_if_open(target_event.write_end);
  TargetInitial target_initial{};
  read_exact(target_event.read_end, &target_initial, sizeof(target_initial));
  const AttackConfig config{target, target_initial.tid,
                            target_initial.canary_address,
                            target_initial.target_fd};

  const pid_t attacker = fork();
  if (attacker < 0) throw Error("attacker fork failed");
  if (attacker == 0) {
    attacker_process(attacker_command, attacker_event, target_command,
                     target_event, config);
  }
  close_if_open(attacker_command.read_end);
  close_if_open(attacker_event.write_end);
  AttackerReady attacker_ready{};
  read_exact(attacker_event.read_end, &attacker_ready, sizeof(attacker_ready));
  write_exact(target_command.write_end, &attacker, sizeof(attacker));
  char target_ready = 0;
  read_exact(target_event.read_end, &target_ready, sizeof(target_ready));
  if (target_ready != 'R' || attacker_ready.pidfd_opened != 1) {
    throw Error("principal pair readiness failed");
  }

  const std::string target_cgroup_name = "/loom-v12-dumpable-target";
  const std::string attacker_cgroup_name = "/loom-v12-dumpable-attacker";
  const std::string target_cgroup = "/sys/fs/cgroup" + target_cgroup_name;
  const std::string attacker_cgroup = "/sys/fs/cgroup" + attacker_cgroup_name;
  write_cgroup_pid(target_cgroup, target);
  write_cgroup_pid(attacker_cgroup, attacker);
  if (process_cgroup(target) != target_cgroup_name ||
      process_cgroup(attacker) != attacker_cgroup_name) {
    throw Error("principal cgroup placement invariant failed");
  }
  const int target_pidfd = static_cast<int>(syscall(SYS_pidfd_open, target, 0));
  const int attacker_pidfd = static_cast<int>(syscall(SYS_pidfd_open, attacker, 0));
  if (target_pidfd < 0 || attacker_pidfd < 0) {
    throw Error("guardian pidfd setup failed");
  }
  const auto target_uids = process_uids(target);
  const auto attacker_uids = process_uids(attacker);
  if (target_uids != attacker_uids ||
      target_uids != attacker_ready.credentials.uids ||
      target_uids != std::array<std::uint32_t, 4>{
                         kPrincipalUid, kPrincipalUid,
                         kPrincipalUid, kPrincipalUid}) {
    throw Error("same-kuid four-slot invariant failed");
  }
  if (process_user_namespace(target) != process_user_namespace(attacker)) {
    throw Error("principal user namespace invariant failed");
  }
  if (process_seccomp(attacker) != 0) {
    throw Error("attacker syscall surface is filtered");
  }
  const std::uint64_t target_start = process_start_tick(target);
  const std::uint64_t attacker_start = process_start_tick(attacker);
  if (target_start == 0 || attacker_start == 0 || target == attacker) {
    throw Error("principal process identity invariant failed");
  }

  const TargetSnapshot baseline = request_snapshot(target_command, target_event);
  if (baseline.dumpable != 0 || baseline.rlimit_cur == 0 ||
      std::memcmp(baseline.canary.data(), kCanary, kCanarySize) != 0 ||
      !credentials_equal(baseline.credentials, attacker_ready.credentials)) {
    throw Error("dumpable-zero baseline is not frozen");
  }
  const std::array<std::uint32_t, 4> expected_gids{
      kPrincipalGid, kPrincipalGid, kPrincipalGid, kPrincipalGid};
  std::array<std::uint32_t, 2> expected_capabilities{};
  expected_capabilities[CAP_TO_INDEX(CAP_SYS_NICE)] =
      CAP_TO_MASK(CAP_SYS_NICE);
  if (baseline.credentials.gids != expected_gids ||
      baseline.credentials.cap_permitted != expected_capabilities ||
      baseline.credentials.cap_effective != expected_capabilities) {
    throw Error("principal minimal capability invariant failed");
  }

  const std::string invariant_preimage =
      "boot=" + boot_id + "|operation=9|target=" + std::to_string(target) +
      "|target_start=" + std::to_string(target_start) + "|attacker=" +
      std::to_string(attacker) + "|attacker_start=" +
      std::to_string(attacker_start) +
      "|uids=61234,61234,61234,61234|dumpable=0|seccomp=0|mediator=absent|"
      "target_cgroup=" + target_cgroup + "|attacker_cgroup=" +
      attacker_cgroup + "|init=" + init_sha +
      "|capability=CAP_SYS_NICE_ONLY";
  const std::string invariant_sha = sha256(invariant_preimage);
  const std::string delta_sha = sha256(
      "vertex=DUMPABLE_ONLY_CONTROL|dumpable=0|mediator=absent|seccomp=0");

  const int raw_operation = static_cast<int>(Operation::Prlimit);
  write_exact(attacker_command.write_end, &raw_operation, sizeof(raw_operation));
  AttackResult result{};
  read_exact(attacker_event.read_end, &result, sizeof(result));
  const TargetSnapshot after = request_snapshot(target_command, target_event);
  const bool completed = result.result == 0 && result.effect == 1 &&
                         result.auxiliary_before != result.auxiliary_after &&
                         snapshots_equal(baseline, after);
  const bool refused = result.result < 0 &&
                       (result.error == EACCES || result.error == EPERM) &&
                       snapshots_equal(baseline, after);
  if (!completed && !refused) {
    throw Error("prlimit observation was neither typed completion nor refusal");
  }
  const std::string attempt_sha = sha256(
      "prlimit64|result=" + std::to_string(result.result) + "|errno=" +
      std::to_string(result.error) + "|effect=" +
      std::to_string(result.effect) + "|before=" +
      std::to_string(result.auxiliary_before) + "|changed=" +
      std::to_string(result.auxiliary_after));
  const std::string target_sha =
      sha256(snapshot_material(baseline) + "|" + snapshot_material(after));

  const int stop = 0;
  write_exact(attacker_command.write_end, &stop, sizeof(stop));
  require_clean_exit(attacker, "attacker");
  write_exact(target_command.write_end, &stop, sizeof(stop));
  require_clean_exit(target, "target");
  close(target_pidfd);
  close(attacker_pidfd);
  close_if_open(target_command.write_end);
  close_if_open(target_event.read_end);
  close_if_open(attacker_command.write_end);
  close_if_open(attacker_event.read_end);
  if (rmdir(target_cgroup.c_str()) != 0 ||
      rmdir(attacker_cgroup.c_str()) != 0) {
    throw Error("principal cgroup extinction failed");
  }
  munmap(signals, sizeof(SharedSignalState));
  const std::string extinction_sha = sha256(
      "target=extinct|attacker=extinct|target_pidfd=closed|"
      "attacker_pidfd=closed|target_cgroup=extinct|attacker_cgroup=extinct|"
      "mediator=absent|operation=9");

  std::cout << "COUNTEREXAMPLE vertex=DUMPABLE_ONLY_CONTROL operation=9 "
               "syscall=prlimit64 frozen_expected=REFUSED_BEFORE_EFFECT "
            << "material_observed="
            << (completed ? "EFFECT_COMPLETED" : "REFUSED_BEFORE_EFFECT")
            << " completion="
            << (completed ? "LIMIT_CHANGED_RESTORED" : "TARGET_STATE_UNCHANGED")
            << " errno="
            << (result.error == EACCES ? "EACCES" :
                result.error == EPERM ? "EPERM" : "NONE")
            << " invariant_sha256=" << invariant_sha
            << " delta_sha256=" << delta_sha
            << " attempt_sha256=" << attempt_sha
            << " target_sha256=" << target_sha
            << " extinction_sha256=" << extinction_sha
            << " same_four_uids=true same_user_namespace=true "
               "distinct_processes=true distinct_pidfds=true "
               "distinct_start_ticks=true distinct_cgroups=true "
               "target_dumpable=0 attacker_seccomp=0 mediator=absent "
               "principal_capability=CAP_SYS_NICE_ONLY "
               "target_limit_restored=true all_epoch_objects_extinct=true "
               "python_executed=false rust_executed=false\n";
  std::cout.flush();
  return completed ? "EFFECT_COMPLETED" : "REFUSED_BEFORE_EFFECT";
}

int run_falsification_microhost() {
  if (getpid() != 1) throw Error("falsification init is not PID 1");
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
  const std::string observed =
      run_dumpable_prlimit_counterexample(boot_id, init_sha);
  struct utsname identity {};
  if (uname(&identity) != 0) throw Error("uname failed");
  const bool falsified = observed == "EFFECT_COMPLETED";
  std::cout << "LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_FALSIFICATION_V12_BOOT PASS "
               "pid=1 kernel="
            << identity.release << " boot_id=" << boot_id
            << " active_lsm=" << active_lsm
            << " frozen_expected=REFUSED_BEFORE_EFFECT material_observed="
            << observed << " v12_hypothesis_falsified="
            << (falsified ? "true" : "false")
            << " counterexamples=" << (falsified ? 1 : 0)
            << " same_four_uids=true target_dumpable=0 attacker_seccomp=0 "
               "mediator=absent principal_capability=CAP_SYS_NICE_ONLY "
               "all_epoch_objects_extinct=true guest_root_traversable=true "
               "guest_disk=none guest_network=none semantic_authority=Sounio "
               "action=9025 controls_executed=false material_peer_matrix=false "
               "same_uid_peer_isolation=false action_9025_decision=DENY451 "
               "claim_ready=false next_stage=SOUNIO_V13_GARDEN\n";
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
          operation_name(Operation::Prlimit) != "prlimit64") {
        throw Error("dumpable/prlimit helper selftest failed");
      }
      std::cout << "LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_FALSIFICATION_INIT_V12_SELFTEST PASS "
                   "operation=9 syscall=prlimit64 "
                   "frozen_expected=REFUSED_BEFORE_EFFECT observations=1 "
                   "language=C++20 role=MATERIAL_BOOTSTRAP transitory=true "
                   "semantic_authority=Sounio python_executed=false "
                   "rust_executed=false controls_executed=false "
                   "material_peer_matrix=false same_uid_peer_isolation=false "
                   "action_9025_decision=DENY451 claim_ready=false\n";
      return 0;
    }
    if (argc != 1) return 64;
    return run_falsification_microhost();
  } catch (const std::exception& error) {
    std::cerr << "LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_FALSIFICATION_V12_REFUSE reason="
              << error.what()
              << " controls_executed=false material_peer_matrix=false "
                 "same_uid_peer_isolation=false action_9025_decision=DENY451 "
                 "claim_ready=false\n";
    std::cerr.flush();
    sync();
    reboot(LINUX_REBOOT_CMD_POWER_OFF);
    return 70;
  }
}
