#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <openssl/sha.h>

#include <fcntl.h>
#include <grp.h>
#include <linux/bpf.h>
#include <linux/capability.h>
#include <linux/reboot.h>
#include <signal.h>
#include <sys/fsuid.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/reboot.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

constexpr uid_t kPrincipalUid = 61234;
constexpr gid_t kPrincipalGid = 61234;
constexpr std::size_t kCanarySize = 32;
constexpr char kCanary[] = "LOOM_V12_CANARY_0123456789ABCDEF";
static_assert(sizeof(kCanary) - 1 == kCanarySize);
constexpr int kSignalPayload = 0x51a7;
constexpr std::string_view kPinDirectory = "/sys/fs/bpf/loom-v12";
constexpr std::array<std::string_view, 3> kPinNames{
    "loom_v12_task_kill", "loom_v12_ptrace_access_check",
    "loom_v12_task_prlimit"};
constexpr std::string_view kCausalLsmStack =
    "lockdown,capability,bpf,ima,evm";

enum class Operation : int {
  Stop = 0,
  Kill = 1,
  Tgkill = 2,
  QueueSignal = 3,
  PidfdSignal = 4,
  Ptrace = 5,
  ProcessVmRead = 6,
  ProcMemRead = 7,
  PidfdGetfd = 8,
  Prlimit = 9,
  ProcessMadvise = 10,
};

struct SharedSignalState {
  volatile sig_atomic_t observed;
  volatile sig_atomic_t payload;
};

struct CredentialWitness {
  std::array<std::uint32_t, 4> uids;
  std::array<std::uint32_t, 4> gids;
  std::array<std::uint32_t, 2> cap_permitted;
  std::array<std::uint32_t, 2> cap_effective;
};

struct TargetInitial {
  pid_t tid;
  std::uintptr_t canary_address;
  int target_fd;
};

struct TargetSnapshot {
  std::array<char, kCanarySize> canary;
  std::uint64_t rlimit_cur;
  int signal_observed;
  int signal_payload;
  int dumpable;
  CredentialWitness credentials;
};

struct AttackerReady {
  int pidfd_opened;
  CredentialWitness credentials;
};

struct AttackResult {
  std::int64_t result;
  int error;
  int effect;
  std::uint64_t auxiliary_before;
  std::uint64_t auxiliary_after;
  int stat_pid_error;
  int stat_status_error;
  int stat_mem_error;
  int open_mem_error;
  std::array<char, kCanarySize> bytes;
};

struct AttackConfig {
  pid_t target_pid;
  pid_t target_tid;
  std::uintptr_t canary_address;
  int target_fd;
};

struct Pipe {
  int read_end = -1;
  int write_end = -1;
};

struct MediatorIdentity {
  std::array<std::uint32_t, 3> links{};
  std::array<std::uint32_t, 3> programs{};
};

SharedSignalState* g_signal_state = nullptr;

std::string sha256(const void* data, std::size_t size) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(static_cast<const unsigned char*>(data), size, digest);
  static constexpr char hex[] = "0123456789abcdef";
  std::string output(SHA256_DIGEST_LENGTH * 2, '0');
  for (std::size_t index = 0; index < SHA256_DIGEST_LENGTH; ++index) {
    output[index * 2] = hex[digest[index] >> 4];
    output[index * 2 + 1] = hex[digest[index] & 0x0f];
  }
  return output;
}

std::string sha256(std::string_view value) {
  return sha256(value.data(), value.size());
}

std::string read_file(const std::string& path, std::size_t maximum = 16 * 1024 * 1024) {
  const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    throw Error("open failed for " + path + ": " + std::strerror(errno));
  }
  std::string output;
  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor, buffer.data(), buffer.size());
    if (count > 0) {
      output.append(buffer.data(), static_cast<std::size_t>(count));
      if (output.size() > maximum) {
        close(descriptor);
        throw Error("file crossed bound: " + path);
      }
    } else if (count == 0) {
      close(descriptor);
      while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
        output.pop_back();
      }
      return output;
    } else if (errno != EINTR) {
      const int saved_errno = errno;
      close(descriptor);
      throw Error("read failed for " + path + ": " + std::strerror(saved_errno));
    }
  }
}

std::string proc_mount_witness() {
  std::istringstream input(read_file("/proc/self/mountinfo", 64 * 1024));
  std::string line;
  while (std::getline(input, line)) {
    if (line.find(" /proc ") != std::string::npos &&
        line.find(" - proc proc ") != std::string::npos) {
      for (char& character : line) {
        if (character == ' ') character = '_';
      }
      return line;
    }
  }
  throw Error("proc mount witness is absent");
}

void write_exact(int descriptor, const void* data, std::size_t size) {
  const char* cursor = static_cast<const char*>(data);
  while (size != 0) {
    const ssize_t count = write(descriptor, cursor, size);
    if (count > 0) {
      cursor += count;
      size -= static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      throw Error(std::string("write_exact failed: ") + std::strerror(errno));
    }
  }
}

void read_exact(int descriptor, void* data, std::size_t size) {
  char* cursor = static_cast<char*>(data);
  while (size != 0) {
    const ssize_t count = read(descriptor, cursor, size);
    if (count > 0) {
      cursor += count;
      size -= static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else if (count == 0) {
      throw Error("read_exact observed premature EOF");
    } else {
      throw Error(std::string("read_exact failed: ") + std::strerror(errno));
    }
  }
}

Pipe make_pipe() {
  int descriptors[2] = {-1, -1};
  if (pipe2(descriptors, O_CLOEXEC) != 0) {
    throw Error(std::string("pipe2 failed: ") + std::strerror(errno));
  }
  return Pipe{descriptors[0], descriptors[1]};
}

void close_if_open(int& descriptor) {
  if (descriptor >= 0) close(descriptor);
  descriptor = -1;
}

void ensure_directory(const std::string& path, mode_t mode = 0755) {
  if (mkdir(path.c_str(), mode) != 0 && errno != EEXIST) {
    throw Error("mkdir failed for " + path + ": " + std::strerror(errno));
  }
}

void mount_required(const char* source, const char* target, const char* type,
                    unsigned long flags = 0) {
  if (mount(source, target, type, flags, nullptr) != 0) {
    throw Error(std::string("mount failed for ") + target + ": " +
                std::strerror(errno));
  }
}

bool comma_token(const std::string& list, std::string_view token) {
  return ("," + list + ",").find("," + std::string(token) + ",") !=
         std::string::npos;
}

void signal_handler(int, siginfo_t* info, void*) {
  if (!g_signal_state) return;
  g_signal_state->payload = info ? info->si_value.sival_int : 0;
  g_signal_state->observed = 1;
}

void install_signal_handler() {
  struct sigaction action {};
  action.sa_sigaction = signal_handler;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGUSR1, &action, nullptr) != 0) {
    throw Error(std::string("sigaction failed: ") + std::strerror(errno));
  }
}

void become_principal() {
  if (prctl(PR_SET_KEEPCAPS, 1, 0, 0, 0) != 0) {
    throw Error(std::string("principal keepcaps transition failed: ") +
                std::strerror(errno));
  }
  if (setgroups(0, nullptr) != 0 || setresgid(kPrincipalGid, kPrincipalGid, kPrincipalGid) != 0 ||
      setresuid(kPrincipalUid, kPrincipalUid, kPrincipalUid) != 0) {
    throw Error(std::string("principal credential transition failed: ") +
                std::strerror(errno));
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
    throw Error(std::string("principal CAP_SYS_NICE transition failed: ") +
                std::strerror(errno));
  }
  setfsuid(kPrincipalUid);
  uid_t real = 0;
  uid_t effective = 0;
  uid_t saved = 0;
  if (getresuid(&real, &effective, &saved) != 0 || real != kPrincipalUid ||
      effective != kPrincipalUid || saved != kPrincipalUid ||
      static_cast<uid_t>(setfsuid(static_cast<uid_t>(-1))) != kPrincipalUid) {
    throw Error("principal four-slot UID transition did not hold");
  }
}

CredentialWitness credential_witness() {
  CredentialWitness witness{};
  uid_t real_uid = 0;
  uid_t effective_uid = 0;
  uid_t saved_uid = 0;
  gid_t real_gid = 0;
  gid_t effective_gid = 0;
  gid_t saved_gid = 0;
  if (getresuid(&real_uid, &effective_uid, &saved_uid) != 0 ||
      getresgid(&real_gid, &effective_gid, &saved_gid) != 0) {
    throw Error("credential witness identity read failed");
  }
  witness.uids = {real_uid, effective_uid, saved_uid,
                  static_cast<std::uint32_t>(setfsuid(static_cast<uid_t>(-1)))};
  witness.gids = {real_gid, effective_gid, saved_gid,
                  static_cast<std::uint32_t>(setfsgid(static_cast<gid_t>(-1)))};
  __user_cap_header_struct header{};
  std::array<__user_cap_data_struct, 2> capabilities{};
  header.version = _LINUX_CAPABILITY_VERSION_3;
  header.pid = 0;
  if (syscall(SYS_capget, &header, capabilities.data()) != 0) {
    throw Error("credential witness capability read failed");
  }
  for (std::size_t index = 0; index < capabilities.size(); ++index) {
    witness.cap_permitted[index] = capabilities[index].permitted;
    witness.cap_effective[index] = capabilities[index].effective;
  }
  return witness;
}

bool credentials_equal(const CredentialWitness& left,
                       const CredentialWitness& right) {
  return left.uids == right.uids && left.gids == right.gids &&
         left.cap_permitted == right.cap_permitted &&
         left.cap_effective == right.cap_effective;
}

std::string credential_material(const CredentialWitness& witness) {
  std::ostringstream output;
  for (const auto value : witness.uids) output << value << ',';
  for (const auto value : witness.gids) output << value << ',';
  for (const auto value : witness.cap_permitted) output << value << ',';
  for (const auto value : witness.cap_effective) output << value << ',';
  return output.str();
}

TargetSnapshot snapshot_target(const char* canary, const SharedSignalState* signals) {
  TargetSnapshot snapshot{};
  std::memcpy(snapshot.canary.data(), canary, snapshot.canary.size());
  struct rlimit limit {};
  if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
    throw Error(std::string("getrlimit failed: ") + std::strerror(errno));
  }
  snapshot.rlimit_cur = limit.rlim_cur;
  snapshot.signal_observed = signals->observed;
  snapshot.signal_payload = signals->payload;
  snapshot.dumpable = prctl(PR_GET_DUMPABLE, 0, 0, 0, 0);
  snapshot.credentials = credential_witness();
  return snapshot;
}

[[noreturn]] void target_process(Pipe command, Pipe event,
                                 Pipe attacker_command, Pipe attacker_event,
                                 SharedSignalState* signals) {
  try {
    close_if_open(command.write_end);
    close_if_open(event.read_end);
    close_if_open(attacker_command.read_end);
    close_if_open(attacker_command.write_end);
    close_if_open(attacker_event.read_end);
    close_if_open(attacker_event.write_end);
    g_signal_state = signals;
    install_signal_handler();
    become_principal();
    if (prctl(PR_SET_DUMPABLE, 1, 0, 0, 0) != 0) {
      throw Error(std::string("target dumpable transition failed: ") +
                  std::strerror(errno));
    }
    struct rlimit limit {};
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) throw Error("target getrlimit failed");
    if (limit.rlim_cur > 1024) {
      limit.rlim_cur = 1024;
      if (setrlimit(RLIMIT_NOFILE, &limit) != 0) throw Error("target setrlimit failed");
    }
    void* page = mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) throw Error("target mmap failed");
    std::memcpy(page, kCanary, kCanarySize);
    const int target_fd = fcntl(event.write_end, F_DUPFD_CLOEXEC, 64);
    if (target_fd < 0) {
      throw Error(std::string("target descriptor setup failed: ") +
                  std::strerror(errno));
    }
    TargetInitial initial{static_cast<pid_t>(syscall(SYS_gettid)),
                          reinterpret_cast<std::uintptr_t>(page), target_fd};
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
    std::cerr << "LOOM_V12_TARGET_REFUSE reason=" << error.what() << "\n";
    _exit(70);
  }
}

AttackResult execute_attack(const AttackConfig& config, Operation operation,
                            int pidfd) {
  AttackResult output{};
  output.result = -1;
  errno = 0;
  switch (operation) {
    case Operation::Kill:
      output.result = syscall(SYS_kill, config.target_pid, SIGTERM);
      break;
    case Operation::Tgkill:
      output.result = syscall(SYS_tgkill, config.target_pid, config.target_tid,
                              SIGTERM);
      break;
    case Operation::QueueSignal: {
      siginfo_t info{};
      info.si_signo = SIGUSR1;
      info.si_code = SI_QUEUE;
      info.si_pid = getpid();
      info.si_uid = getuid();
      info.si_value.sival_int = kSignalPayload;
      output.result = syscall(SYS_rt_sigqueueinfo, config.target_pid, SIGUSR1, &info);
      break;
    }
    case Operation::PidfdSignal:
      output.result = syscall(SYS_pidfd_send_signal, pidfd, SIGTERM, nullptr, 0);
      break;
    case Operation::Ptrace: {
      output.result = ptrace(PTRACE_ATTACH, config.target_pid, nullptr, nullptr);
      if (output.result == 0) {
        int status = 0;
        const pid_t waited = waitpid(config.target_pid, &status, __WALL);
        output.auxiliary_before = static_cast<std::uint64_t>(waited);
        output.auxiliary_after = static_cast<std::uint64_t>(status);
        if (waited == config.target_pid &&
            WIFSTOPPED(status) &&
            ptrace(PTRACE_DETACH, config.target_pid, nullptr, nullptr) == 0) {
          output.effect = 1;
        } else {
          output.result = -1;
          errno = EIO;
        }
      }
      break;
    }
    case Operation::ProcessVmRead: {
      struct iovec local { output.bytes.data(), output.bytes.size() };
      struct iovec remote { reinterpret_cast<void*>(config.canary_address),
                            output.bytes.size() };
      output.result = syscall(SYS_process_vm_readv, config.target_pid, &local, 1,
                              &remote, 1, 0);
      if (output.result == static_cast<std::int64_t>(output.bytes.size()) &&
          std::memcmp(output.bytes.data(), kCanary, kCanarySize) == 0) {
        output.effect = 1;
      }
      break;
    }
    case Operation::ProcMemRead: {
      const std::string process_path =
          "/proc/" + std::to_string(config.target_pid);
      const std::string status_path = process_path + "/status";
      const std::string path = process_path + "/mem";
      struct stat metadata {};
      if (stat(process_path.c_str(), &metadata) != 0) {
        output.stat_pid_error = errno;
      }
      if (stat(status_path.c_str(), &metadata) != 0) {
        output.stat_status_error = errno;
      }
      if (stat(path.c_str(), &metadata) == 0) {
        output.auxiliary_before = metadata.st_uid;
        output.auxiliary_after = metadata.st_mode;
      } else {
        output.stat_mem_error = errno;
      }
      const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
      if (descriptor < 0) {
        output.open_mem_error = errno;
        output.result = -1;
      } else {
        output.result = pread(descriptor, output.bytes.data(), output.bytes.size(),
                              static_cast<off_t>(config.canary_address));
        const int saved_errno = errno;
        close(descriptor);
        errno = saved_errno;
        if (output.result == static_cast<std::int64_t>(output.bytes.size()) &&
            std::memcmp(output.bytes.data(), kCanary, kCanarySize) == 0) {
          output.effect = 1;
        }
      }
      break;
    }
    case Operation::PidfdGetfd: {
      const int duplicate =
          static_cast<int>(syscall(SYS_pidfd_getfd, pidfd, config.target_fd, 0));
      output.result = duplicate;
      if (duplicate >= 0) {
        output.effect = 1;
        close(duplicate);
        output.result = 0;
      }
      break;
    }
    case Operation::Prlimit: {
      struct rlimit old_limit {};
      struct rlimit new_limit {};
      if (syscall(SYS_prlimit64, config.target_pid, RLIMIT_NOFILE, nullptr,
                  &old_limit) != 0) {
        output.result = -1;
        break;
      }
      output.auxiliary_before = old_limit.rlim_cur;
      new_limit = old_limit;
      new_limit.rlim_cur = old_limit.rlim_cur > 768 ? 768 : old_limit.rlim_cur - 1;
      output.result = syscall(SYS_prlimit64, config.target_pid, RLIMIT_NOFILE,
                              &new_limit, nullptr);
      if (output.result == 0) {
        struct rlimit observed {};
        const bool changed =
            syscall(SYS_prlimit64, config.target_pid, RLIMIT_NOFILE, nullptr,
                    &observed) == 0 && observed.rlim_cur == new_limit.rlim_cur;
        const bool restored =
            syscall(SYS_prlimit64, config.target_pid, RLIMIT_NOFILE, &old_limit,
                    nullptr) == 0;
        output.auxiliary_after = observed.rlim_cur;
        output.effect = changed && restored;
        if (!output.effect) {
          output.result = -1;
          errno = EIO;
        }
      }
      break;
    }
    case Operation::ProcessMadvise: {
      struct iovec remote { reinterpret_cast<void*>(config.canary_address), 4096 };
      output.result = syscall(SYS_process_madvise, pidfd, &remote, 1,
                              MADV_PAGEOUT, 0);
      if (output.result == static_cast<std::int64_t>(remote.iov_len)) {
        output.effect = 1;
      }
      break;
    }
    case Operation::Stop:
      break;
  }
  output.error = output.result < 0 ? errno : 0;
  return output;
}

[[noreturn]] void attacker_process(Pipe command, Pipe event,
                                   Pipe target_command, Pipe target_event,
                                   const AttackConfig& config) {
  try {
    close_if_open(command.write_end);
    close_if_open(event.read_end);
    close_if_open(target_command.read_end);
    close_if_open(target_command.write_end);
    close_if_open(target_event.read_end);
    close_if_open(target_event.write_end);
    become_principal();
    const int pidfd = static_cast<int>(syscall(SYS_pidfd_open, config.target_pid, 0));
    if (pidfd < 0) throw Error("attacker pidfd_open failed");
    AttackerReady ready{1, credential_witness()};
    write_exact(event.write_end, &ready, sizeof(ready));
    for (;;) {
      int raw_operation = 0;
      read_exact(command.read_end, &raw_operation, sizeof(raw_operation));
      const auto operation = static_cast<Operation>(raw_operation);
      if (operation == Operation::Stop) break;
      const AttackResult result = execute_attack(config, operation, pidfd);
      write_exact(event.write_end, &result, sizeof(result));
    }
    close(pidfd);
    _exit(0);
  } catch (const std::exception& error) {
    std::cerr << "LOOM_V12_ATTACKER_REFUSE reason=" << error.what() << "\n";
    _exit(70);
  }
}

std::array<std::uint32_t, 4> process_uids(pid_t pid) {
  std::istringstream input(read_file("/proc/" + std::to_string(pid) + "/status"));
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("Uid:", 0) == 0) {
      std::istringstream values(line.substr(4));
      std::array<std::uint32_t, 4> uids{};
      if (values >> uids[0] >> uids[1] >> uids[2] >> uids[3]) return uids;
    }
  }
  throw Error("process UID vector is absent");
}

int process_seccomp(pid_t pid) {
  std::istringstream input(read_file("/proc/" + std::to_string(pid) + "/status"));
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("Seccomp:", 0) == 0) return std::stoi(line.substr(8));
  }
  throw Error("process seccomp field is absent");
}

std::string process_cgroup(pid_t pid) {
  std::istringstream input(
      read_file("/proc/" + std::to_string(pid) + "/cgroup", 4096));
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("0::", 0) == 0) return line.substr(3);
  }
  throw Error("process unified cgroup field is absent");
}

std::pair<std::uint64_t, std::uint64_t> process_user_namespace(pid_t pid) {
  struct stat metadata {};
  const std::string path =
      "/proc/" + std::to_string(pid) + "/ns/user";
  if (stat(path.c_str(), &metadata) != 0) {
    throw Error("guardian user namespace observation failed for pid=" +
                std::to_string(pid) + ": " + std::strerror(errno));
  }
  return {metadata.st_dev, metadata.st_ino};
}

std::uint64_t process_start_tick(pid_t pid) {
  const std::string stat = read_file("/proc/" + std::to_string(pid) + "/stat", 4096);
  const std::size_t end = stat.rfind(") ");
  if (end == std::string::npos) throw Error("process stat comm terminator is absent");
  std::istringstream fields(stat.substr(end + 2));
  std::string value;
  for (int index = 0; index <= 19; ++index) {
    if (!(fields >> value)) throw Error("process stat start tick is absent");
  }
  return std::stoull(value);
}

void write_cgroup_pid(const std::string& directory, pid_t pid) {
  ensure_directory(directory, 0700);
  const std::string path = directory + "/cgroup.procs";
  const int descriptor = open(path.c_str(), O_WRONLY | O_CLOEXEC);
  if (descriptor < 0) throw Error("cannot open cgroup.procs");
  const std::string text = std::to_string(pid);
  write_exact(descriptor, text.data(), text.size());
  close(descriptor);
}

int bpf_call(enum bpf_cmd command, union bpf_attr* attributes) {
  return static_cast<int>(syscall(__NR_bpf, command, attributes,
                                  sizeof(*attributes)));
}

std::pair<std::uint32_t, std::uint32_t> pinned_link_identity(
    const std::string& path) {
  union bpf_attr get_attributes {};
  get_attributes.pathname =
      static_cast<__u64>(reinterpret_cast<std::uintptr_t>(path.c_str()));
  const int descriptor = bpf_call(BPF_OBJ_GET, &get_attributes);
  if (descriptor < 0) throw Error("BPF_OBJ_GET failed for " + path);
  struct bpf_link_info info {};
  union bpf_attr info_attributes {};
  info_attributes.info.bpf_fd = descriptor;
  info_attributes.info.info_len = sizeof(info);
  info_attributes.info.info =
      static_cast<__u64>(reinterpret_cast<std::uintptr_t>(&info));
  if (bpf_call(BPF_OBJ_GET_INFO_BY_FD, &info_attributes) != 0) {
    close(descriptor);
    throw Error("BPF_OBJ_GET_INFO_BY_FD failed");
  }
  close(descriptor);
  if (info.id == 0) throw Error("BPF link identity is zero");
  if (info.prog_id == 0) throw Error("BPF program identity is zero");
  return {info.id, info.prog_id};
}

MediatorIdentity install_mediator() {
  ensure_directory(kPinDirectory.data(), 0700);
  const pid_t loader = fork();
  if (loader < 0) throw Error("loader fork failed");
  if (loader == 0) {
    execl("/loom/loom-bpf-lsm-loader-v12", "loom-bpf-lsm-loader-v12", "--load",
          "/loom/policy.bpf.o", kPinDirectory.data(),
          static_cast<char*>(nullptr));
    _exit(127);
  }
  int status = 0;
  if (waitpid(loader, &status, 0) != loader || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0) {
    throw Error("mediator loader refused");
  }
  MediatorIdentity identity{};
  for (std::size_t index = 0; index < kPinNames.size(); ++index) {
    const auto [link, program] = pinned_link_identity(
        std::string(kPinDirectory) + "/" + std::string(kPinNames[index]));
    identity.links[index] = link;
    identity.programs[index] = program;
  }
  return identity;
}

void wait_bpf_id_extinct(enum bpf_cmd command, std::uint32_t id,
                         std::string_view object) {
  for (int attempt = 0; attempt < 1000; ++attempt) {
    union bpf_attr attributes {};
    if (command == BPF_LINK_GET_FD_BY_ID) {
      attributes.link_id = id;
    } else if (command == BPF_PROG_GET_FD_BY_ID) {
      attributes.prog_id = id;
    } else {
      throw Error("unsupported BPF extinction query");
    }
    const int descriptor = bpf_call(command, &attributes);
    if (descriptor < 0 && errno == ENOENT) return;
    if (descriptor < 0) {
      throw Error(std::string(object) + " extinction query failed closed");
    }
    close(descriptor);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  throw Error(std::string(object) + " crossed extinction timeout");
}

void remove_mediator(const MediatorIdentity& identity) {
  for (const auto name : kPinNames) {
    const std::string path = std::string(kPinDirectory) + "/" + std::string(name);
    if (unlink(path.c_str()) != 0) throw Error("cannot unlink mediator pin");
  }
  if (rmdir(kPinDirectory.data()) != 0) throw Error("cannot remove mediator directory");
  for (const auto id : identity.links) {
    wait_bpf_id_extinct(BPF_LINK_GET_FD_BY_ID, id, "mediator link");
  }
  for (const auto id : identity.programs) {
    wait_bpf_id_extinct(BPF_PROG_GET_FD_BY_ID, id, "mediator program");
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
}

TargetSnapshot request_snapshot(Pipe& command, Pipe& event) {
  const int request = 1;
  write_exact(command.write_end, &request, sizeof(request));
  TargetSnapshot snapshot{};
  read_exact(event.read_end, &snapshot, sizeof(snapshot));
  return snapshot;
}

bool snapshots_equal(const TargetSnapshot& left, const TargetSnapshot& right) {
  return left.canary == right.canary && left.rlimit_cur == right.rlimit_cur &&
         left.signal_observed == right.signal_observed &&
         left.signal_payload == right.signal_payload &&
         left.dumpable == right.dumpable &&
         credentials_equal(left.credentials, right.credentials);
}

std::string snapshot_material(const TargetSnapshot& snapshot) {
  return std::string(snapshot.canary.data(), snapshot.canary.size()) + "|" +
         std::to_string(snapshot.rlimit_cur) + "|" +
         std::to_string(snapshot.signal_observed) + "|" +
         std::to_string(snapshot.signal_payload) + "|" +
         std::to_string(snapshot.dumpable) + "|" +
         sha256(credential_material(snapshot.credentials));
}

bool wait_for_signal_exit(pid_t pid, int& status) {
  for (int attempt = 0; attempt < 1000; ++attempt) {
    const pid_t result = waitpid(pid, &status, WNOHANG);
    if (result == pid) return true;
    if (result < 0) throw Error("waitpid target failed");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return false;
}

std::string operation_name(Operation operation) {
  switch (operation) {
    case Operation::Kill: return "kill_SIGTERM";
    case Operation::Tgkill: return "tgkill_SIGTERM";
    case Operation::QueueSignal: return "rt_sigqueueinfo";
    case Operation::PidfdSignal: return "pidfd_send_signal";
    case Operation::Ptrace: return "ptrace_ATTACH";
    case Operation::ProcessVmRead: return "process_vm_readv";
    case Operation::ProcMemRead: return "open_read_proc_pid_mem";
    case Operation::PidfdGetfd: return "pidfd_getfd";
    case Operation::Prlimit: return "prlimit64";
    case Operation::ProcessMadvise: return "process_madvise";
    case Operation::Stop: return "stop";
  }
  return "unknown";
}

void require_clean_exit(pid_t pid, std::string_view role) {
  int status = 0;
  if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0) {
    throw Error(std::string(role) + " did not exit cleanly");
  }
}

std::string run_decisive_pair(Operation operation, const std::string& boot_id,
                              const std::string& init_sha,
                              const std::string& policy_sha) {
  const int index = static_cast<int>(operation);
  const MediatorIdentity mediator_identity = install_mediator();
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
  if (target == 0) {
    target_process(target_command, target_event, attacker_command, attacker_event,
                   signals);
  }
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

  const std::string target_cgroup_name =
      "/loom-v12-op" + std::to_string(index) + "-target";
  const std::string attacker_cgroup_name =
      "/loom-v12-op" + std::to_string(index) + "-attacker";
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
  if (target_pidfd < 0 || attacker_pidfd < 0) throw Error("guardian pidfd setup failed");
  const auto target_uids = process_uids(target);
  const auto attacker_uids = process_uids(attacker);
  const auto target_user_namespace = process_user_namespace(target);
  const auto attacker_user_namespace = process_user_namespace(attacker);
  const std::string proc_mem_path =
      "/proc/" + std::to_string(target) + "/mem";
  const std::string proc_pid_path =
      "/proc/" + std::to_string(target);
  const std::string proc_status_path = proc_pid_path + "/status";
  struct stat guardian_proc_pid {};
  struct stat guardian_proc_status {};
  struct stat guardian_proc_mem {};
  if (stat(proc_pid_path.c_str(), &guardian_proc_pid) != 0 ||
      stat(proc_status_path.c_str(), &guardian_proc_status) != 0 ||
      stat(proc_mem_path.c_str(), &guardian_proc_mem) != 0) {
    throw Error(std::string("guardian cannot stat target proc surface: ") +
                std::strerror(errno));
  }
  if (target_uids != attacker_uids ||
      target_uids != attacker_ready.credentials.uids ||
      target_uids != std::array<std::uint32_t, 4>{kPrincipalUid, kPrincipalUid,
                                                  kPrincipalUid, kPrincipalUid}) {
    throw Error("same-kuid four-slot invariant failed");
  }
  if (target_user_namespace != attacker_user_namespace) {
    throw Error("principal user namespace invariant failed");
  }
  if (process_seccomp(attacker) != 0) throw Error("attacker syscall surface is filtered");
  const std::uint64_t target_start = process_start_tick(target);
  const std::uint64_t attacker_start = process_start_tick(attacker);
  if (target_start == 0 || attacker_start == 0 || target == attacker) {
    throw Error("principal process identity invariant failed");
  }
  const TargetSnapshot baseline = request_snapshot(target_command, target_event);
  if (std::memcmp(baseline.canary.data(), kCanary, kCanarySize) != 0 ||
      baseline.signal_observed != 0 || baseline.dumpable != 1) {
    throw Error("target baseline is not frozen");
  }
  if (!credentials_equal(baseline.credentials, attacker_ready.credentials)) {
    throw Error("principal credential witness diverged");
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
      "boot=" + boot_id + "|op=" + std::to_string(index) +
      "|target=" + std::to_string(target) + "|target_start=" +
      std::to_string(target_start) + "|attacker=" + std::to_string(attacker) +
      "|attacker_start=" + std::to_string(attacker_start) +
      "|uids=61234,61234,61234,61234|target_cgroup=" + target_cgroup +
      "|attacker_cgroup=" + attacker_cgroup + "|init=" + init_sha +
      "|target_address=" + std::to_string(config.canary_address) +
      "|target_fd=" + std::to_string(config.target_fd) +
      "|proc_mem_uid=" + std::to_string(guardian_proc_mem.st_uid) +
      "|proc_mem_mode=" + std::to_string(guardian_proc_mem.st_mode) +
      "|user_ns=" + std::to_string(target_user_namespace.first) + ":" +
      std::to_string(target_user_namespace.second) +
      "|capability=CAP_SYS_NICE_ONLY|target_pidfd_open=true|attacker_pidfd_open=true|ptracer_aperture=NOT_REQUIRED|seccomp=0";
  const std::string invariant_sha = sha256(invariant_preimage);
  const std::string treatment_delta =
      sha256("mediator=active|policy_sha256=" + policy_sha);
  const std::string sabotage_delta = sha256("mediator=absent|policy_sha256=none");

  const int raw_operation = index;
  write_exact(attacker_command.write_end, &raw_operation, sizeof(raw_operation));
  AttackResult treatment{};
  read_exact(attacker_event.read_end, &treatment, sizeof(treatment));
  const TargetSnapshot after_treatment = request_snapshot(target_command, target_event);
  if (treatment.result >= 0 ||
      (treatment.error != EACCES && treatment.error != EPERM) ||
      !snapshots_equal(baseline, after_treatment)) {
    throw Error("treatment did not refuse before effect for operation=" +
                std::to_string(index) + " errno=" +
                std::to_string(treatment.error));
  }
  const std::string treatment_attempt_sha = sha256(
      operation_name(operation) + "|result=" + std::to_string(treatment.result) +
      "|errno=" + std::to_string(treatment.error));
  const std::string treatment_target_sha =
      sha256(snapshot_material(baseline) + "|" + snapshot_material(after_treatment));

  remove_mediator(mediator_identity);
  write_exact(attacker_command.write_end, &raw_operation, sizeof(raw_operation));
  AttackResult sabotage{};
  read_exact(attacker_event.read_end, &sabotage, sizeof(sabotage));
  bool target_reaped = false;
  std::string completion;
  TargetSnapshot after_sabotage{};
  if (operation == Operation::Kill || operation == Operation::Tgkill ||
      operation == Operation::PidfdSignal) {
    int target_status = 0;
    const bool exited = wait_for_signal_exit(target, target_status);
    if (sabotage.result != 0 || !exited || !WIFSIGNALED(target_status) ||
        WTERMSIG(target_status) != SIGTERM) {
      throw Error("mediator-removed termination did not complete result=" +
                  std::to_string(sabotage.result) + " errno=" +
                  std::to_string(sabotage.error) + " exited=" +
                  std::to_string(exited) + " wait_status=" +
                  std::to_string(target_status));
    }
    target_reaped = true;
    completion = operation == Operation::Tgkill ? "TARGET_THREAD_TERMINATED"
                                                 : "TARGET_TERMINATED";
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    after_sabotage = request_snapshot(target_command, target_event);
    switch (operation) {
      case Operation::QueueSignal:
        if (sabotage.result != 0 || after_sabotage.signal_observed != 1 ||
            after_sabotage.signal_payload != kSignalPayload) {
          throw Error("queued-signal witness did not complete");
        }
        completion = "SIGNAL_PAYLOAD_OBSERVED";
        break;
      case Operation::Ptrace:
        if (sabotage.result != 0 || sabotage.effect != 1) {
          throw Error("ptrace attach-detach witness did not complete result=" +
                      std::to_string(sabotage.result) + " errno=" +
                      std::to_string(sabotage.error) + " effect=" +
                      std::to_string(sabotage.effect) + " waited=" +
                      std::to_string(sabotage.auxiliary_before) + " status=" +
                      std::to_string(sabotage.auxiliary_after));
        }
        completion = "PTRACE_ATTACH_DETACH";
        break;
      case Operation::ProcessVmRead:
        if (sabotage.effect != 1) {
          throw Error("process_vm_readv witness did not complete result=" +
                      std::to_string(sabotage.result) + " errno=" +
                      std::to_string(sabotage.error) + " effect=" +
                      std::to_string(sabotage.effect));
        }
        completion = "CANARY_BYTES_READ";
        break;
      case Operation::ProcMemRead:
        if (sabotage.effect != 1) {
          throw Error("proc mem witness did not complete result=" +
                      std::to_string(sabotage.result) + " errno=" +
                      std::to_string(sabotage.error) + " effect=" +
                      std::to_string(sabotage.effect) + " proc_uid=" +
                      std::to_string(sabotage.auxiliary_before) + " proc_mode=" +
                      std::to_string(sabotage.auxiliary_after) +
                      " guardian_proc_uid=" +
                      std::to_string(guardian_proc_mem.st_uid) +
                      " guardian_proc_mode=" +
                      std::to_string(guardian_proc_mem.st_mode) +
                      " target_dumpable=" +
                      std::to_string(baseline.dumpable) +
                      " guardian_pid_uid=" +
                      std::to_string(guardian_proc_pid.st_uid) +
                      " guardian_pid_mode=" +
                      std::to_string(guardian_proc_pid.st_mode) +
                      " guardian_status_uid=" +
                      std::to_string(guardian_proc_status.st_uid) +
                      " guardian_status_mode=" +
                      std::to_string(guardian_proc_status.st_mode) +
                      " stat_pid_errno=" +
                      std::to_string(sabotage.stat_pid_error) +
                      " stat_status_errno=" +
                      std::to_string(sabotage.stat_status_error) +
                      " stat_mem_errno=" +
                      std::to_string(sabotage.stat_mem_error) +
                      " open_mem_errno=" +
                      std::to_string(sabotage.open_mem_error) +
                      " caps_permitted=" +
                      std::to_string(baseline.credentials.cap_permitted[0]) +
                      "," +
                      std::to_string(baseline.credentials.cap_permitted[1]) +
                      " caps_effective=" +
                      std::to_string(baseline.credentials.cap_effective[0]) +
                      "," +
                      std::to_string(baseline.credentials.cap_effective[1]) +
                      " user_ns=" +
                      std::to_string(target_user_namespace.first) + ":" +
                      std::to_string(target_user_namespace.second) +
                      " proc_mount=" + proc_mount_witness());
        }
        completion = "PROC_MEM_CANARY_READ";
        break;
      case Operation::PidfdGetfd:
        if (sabotage.result != 0 || sabotage.effect != 1) {
          throw Error("pidfd_getfd witness did not complete errno=" +
                      std::to_string(sabotage.error));
        }
        completion = "TARGET_FD_DUPLICATED";
        break;
      case Operation::Prlimit:
        if (sabotage.result != 0 || sabotage.effect != 1 ||
            after_sabotage.rlimit_cur != baseline.rlimit_cur) {
          throw Error("prlimit changed-restored witness did not complete");
        }
        completion = "LIMIT_CHANGED_RESTORED";
        break;
      case Operation::ProcessMadvise: {
        if (sabotage.result != 4096 || sabotage.effect != 1 ||
            after_sabotage.canary != baseline.canary) {
          throw Error("process_madvise PAGEOUT witness did not complete result=" +
                      std::to_string(sabotage.result) + " errno=" +
                      std::to_string(sabotage.error) + " effect=" +
                      std::to_string(sabotage.effect));
        }
        completion = "MADVISE_COMPLETED_4096_BYTES";
        break;
      }
      default:
        throw Error("unexpected non-destructive operation");
    }
  }
  const std::string sabotage_attempt_sha = sha256(
      operation_name(operation) + "|result=" + std::to_string(sabotage.result) +
      "|errno=" + std::to_string(sabotage.error) +
      "|effect=" + std::to_string(sabotage.effect));
  const std::string sabotage_target_sha =
      target_reaped ? sha256(snapshot_material(baseline) + "|SIGTERM")
                    : sha256(snapshot_material(baseline) + "|" +
                             snapshot_material(after_sabotage));

  const int stop = 0;
  write_exact(attacker_command.write_end, &stop, sizeof(stop));
  require_clean_exit(attacker, "attacker");
  if (!target_reaped) {
    write_exact(target_command.write_end, &stop, sizeof(stop));
    require_clean_exit(target, "target");
  }
  close(target_pidfd);
  close(attacker_pidfd);
  close_if_open(target_command.write_end);
  close_if_open(target_event.read_end);
  close_if_open(attacker_command.write_end);
  close_if_open(attacker_event.read_end);
  if (rmdir(target_cgroup.c_str()) != 0 || rmdir(attacker_cgroup.c_str()) != 0) {
    throw Error("principal cgroup extinction failed");
  }
  munmap(signals, sizeof(SharedSignalState));
  const std::string extinction_sha = sha256(
      "target=extinct|attacker=extinct|target_pidfd=closed|attacker_pidfd=closed|"
      "target_cgroup=extinct|attacker_cgroup=extinct|mediator=extinct|op=" +
      std::to_string(index));
  std::cout << "PAIR operation=" << index << " syscall=" << operation_name(operation)
            << " invariant_sha256=" << invariant_sha
            << " treatment_delta_sha256=" << treatment_delta
            << " sabotage_delta_sha256=" << sabotage_delta
            << " treatment=REFUSED_BEFORE_EFFECT treatment_errno="
            << (treatment.error == EACCES ? "EACCES" : "EPERM")
            << " treatment_attempt_sha256=" << treatment_attempt_sha
            << " treatment_target_sha256=" << treatment_target_sha
            << " sabotage=EFFECT_COMPLETED completion=" << completion
            << " sabotage_attempt_sha256=" << sabotage_attempt_sha
            << " sabotage_target_sha256=" << sabotage_target_sha
            << " extinction_sha256=" << extinction_sha
            << " same_four_uids=true attacker_seccomp=0 distinct_cgroups=true"
               " same_process_epoch=true only_delta=mediator_presence+policy_hash"
               " mediator_links_extinct=true mediator_programs_extinct=true"
               " mediator_quiescence_ms=250 ptracer_aperture=NOT_REQUIRED"
               " competing_ptrace_lsms=absent guest_root_traversable=true"
               " principal_capability=CAP_SYS_NICE_ONLY\n";
  std::cout.flush();
  return invariant_sha;
}

int run_microhost() {
  if (getpid() != 1) throw Error("peer matrix init is not PID 1");
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
  mount_required("bpffs", "/sys/fs/bpf", "bpf", MS_NOSUID | MS_NODEV | MS_NOEXEC);
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
  struct utsname identity {};
  if (uname(&identity) != 0) throw Error("uname failed");
  std::cout << "LOOM_KERNEL_PEER_MATRIX_V12_BOOT PASS pid=1 kernel="
            << identity.release << " boot_id=" << boot_id
            << " active_lsm=" << active_lsm
            << " decisive_pairs=10 treatment_refused=10 "
               "mediator_removed_completed=10 same_kuid_pair_observed=true "
               "all_four_kernel_uid_slots_equal=true attacker_syscalls_open=true "
               "receiver_mediator_active=true only_delta_mediator=true "
               "competing_ptrace_lsms=absent "
               "all_epoch_objects_extinct=true pair_set_sha256="
            << sha256(pair_material)
            << " guest_root_traversable=true guest_disk=none guest_network=none "
               "semantic_authority=Sounio "
               "action=9025 controls_executed=false material_peer_matrix=false "
               "same_uid_peer_isolation=false action_9025_decision=DENY451 "
               "claim_ready=false\n";
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
          operation_name(Operation::ProcessMadvise) != "process_madvise") {
        throw Error("peer matrix deterministic helper selftest failed");
      }
      std::cout << "LOOM_KERNEL_PEER_MATRIX_INIT_V12_SELFTEST PASS operations=10 "
                   "decisive_pairs=10 same_process_epoch=true "
                   "only_delta=mediator_presence+policy_hash sha256=true "
                   "language=C++20 role=MATERIAL_BOOTSTRAP transitory=true "
                   "python_executed=false rust_executed=false "
                   "controls_executed=false material_peer_matrix=false "
                   "same_uid_peer_isolation=false action_9025_decision=DENY451 "
                   "claim_ready=false\n";
      return 0;
    }
    if (argc != 1) return 64;
    return run_microhost();
  } catch (const std::exception& error) {
    std::cerr << "LOOM_KERNEL_PEER_MATRIX_V12_REFUSE reason=" << error.what()
              << " controls_executed=false material_peer_matrix=false "
                 "same_uid_peer_isolation=false action_9025_decision=DENY451 "
                 "claim_ready=false\n";
    std::cerr.flush();
    sync();
    reboot(LINUX_REBOOT_CMD_POWER_OFF);
    return 70;
  }
}
