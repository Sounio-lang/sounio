#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <openssl/sha.h>

#include <grp.h>
#include <poll.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
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

struct UniqueFd {
  int value = -1;

  UniqueFd() = default;
  explicit UniqueFd(int descriptor) : value(descriptor) {}
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : value(other.value) { other.value = -1; }
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      if (value >= 0) close(value);
      value = other.value;
      other.value = -1;
    }
    return *this;
  }
  ~UniqueFd() {
    if (value >= 0) close(value);
  }
  int get() const { return value; }
};

struct ProcFacts {
  pid_t pid = 0;
  uid_t uid = 0;
  gid_t gid = 0;
  std::uint64_t start_tick = 0;
  std::uint64_t pid_namespace_device = 0;
  std::uint64_t pid_namespace_inode = 0;
  std::string cgroup;
  std::string executable;
  std::string cap_effective;
  std::string cap_ambient;
  bool no_new_privileges = false;
};

struct AttackResult {
  int setup_errno = 0;
  int effective_uid = -1;
  int effective_gid = -1;
  int signal_errno = 0;
  int proc_mem_errno = 0;
  int ptrace_errno = 0;
  int process_vm_errno = 0;
  int proc_fd_errno = 0;
  int pidfd_signal_errno = 0;
  int pidfd_getfd_errno = 0;
};

std::string trim(std::string value) {
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r' || value.back() == ' ' ||
          value.back() == '\t')) {
    value.pop_back();
  }
  std::size_t start = 0;
  while (start < value.size() &&
         (value[start] == ' ' || value[start] == '\t' || value[start] == '\n' ||
          value[start] == '\r')) {
    ++start;
  }
  return value.substr(start);
}

std::string read_file(const std::string& path, std::size_t maximum = 1024 * 1024) {
  UniqueFd descriptor(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (descriptor.get() < 0) {
    throw Error("cannot open " + path + ": " + std::strerror(errno));
  }
  std::string output;
  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor.get(), buffer.data(), buffer.size());
    if (count > 0) {
      output.append(buffer.data(), static_cast<std::size_t>(count));
      if (output.size() > maximum) throw Error("file exceeds limit: " + path);
    } else if (count == 0) {
      return output;
    } else if (errno != EINTR) {
      throw Error("cannot read " + path + ": " + std::strerror(errno));
    }
  }
}

std::string read_link(const std::string& path) {
  std::array<char, 4096> buffer{};
  const ssize_t count = readlink(path.c_str(), buffer.data(), buffer.size() - 1);
  if (count < 0) throw Error("cannot read link " + path + ": " + std::strerror(errno));
  return std::string(buffer.data(), static_cast<std::size_t>(count));
}

std::string sha256(std::string_view value) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(value.data()), value.size(), digest);
  static constexpr char hex[] = "0123456789abcdef";
  std::string output(SHA256_DIGEST_LENGTH * 2, '0');
  for (std::size_t index = 0; index < SHA256_DIGEST_LENGTH; ++index) {
    output[index * 2] = hex[digest[index] >> 4];
    output[index * 2 + 1] = hex[digest[index] & 0x0f];
  }
  return output;
}

std::optional<std::uint64_t> parse_u64(std::string_view text) {
  if (text.empty()) return std::nullopt;
  std::uint64_t value = 0;
  for (const unsigned char character : text) {
    if (character < '0' || character > '9') return std::nullopt;
    const std::uint64_t digit = character - '0';
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
      return std::nullopt;
    }
    value = value * 10 + digit;
  }
  return value;
}

std::map<std::string, std::string> parse_status(pid_t pid) {
  std::istringstream input(read_file("/proc/" + std::to_string(pid) + "/status"));
  std::map<std::string, std::string> fields;
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    fields.emplace(line.substr(0, colon), trim(line.substr(colon + 1)));
  }
  return fields;
}

const std::string& require_field(const std::map<std::string, std::string>& fields,
                                 const std::string& key) {
  const auto found = fields.find(key);
  if (found == fields.end() || found->second.empty()) {
    throw Error("process status omitted " + key);
  }
  return found->second;
}

std::uint64_t first_decimal(const std::string& value, const std::string& label) {
  const std::size_t end = value.find_first_of(" \t");
  const auto parsed = parse_u64(value.substr(0, end));
  if (!parsed) throw Error("invalid decimal process field: " + label);
  return *parsed;
}

bool all_identity_fields_equal(const std::string& value, std::uint64_t expected) {
  std::istringstream input(value);
  std::uint64_t field = 0;
  int count = 0;
  while (input >> field) {
    if (field != expected) return false;
    ++count;
  }
  return count == 4;
}

std::uint64_t process_start_tick(pid_t pid) {
  const std::string record = read_file("/proc/" + std::to_string(pid) + "/stat");
  const std::size_t close = record.rfind(')');
  if (close == std::string::npos || close + 2 >= record.size()) {
    throw Error("malformed process stat record");
  }
  std::istringstream input(record.substr(close + 2));
  std::string field;
  for (int index = 0; index <= 19; ++index) {
    if (!(input >> field)) throw Error("process stat omitted start tick");
  }
  const auto parsed = parse_u64(field);
  if (!parsed || *parsed == 0) throw Error("invalid process start tick");
  return *parsed;
}

ProcFacts process_facts(pid_t pid) {
  if (pid <= 1) throw Error("principal cell PID must exceed 1");
  const auto status = parse_status(pid);
  const std::uint64_t uid = first_decimal(require_field(status, "Uid"), "Uid");
  const std::uint64_t gid = first_decimal(require_field(status, "Gid"), "Gid");
  if (uid > std::numeric_limits<uid_t>::max() ||
      gid > std::numeric_limits<gid_t>::max() ||
      !all_identity_fields_equal(require_field(status, "Uid"), uid) ||
      !all_identity_fields_equal(require_field(status, "Gid"), gid)) {
    throw Error("principal cell has a split credential vector");
  }
  struct stat namespace_info {};
  const std::string namespace_path = "/proc/" + std::to_string(pid) + "/ns/pid";
  if (stat(namespace_path.c_str(), &namespace_info) != 0) {
    throw Error("cannot stat principal PID namespace");
  }
  ProcFacts facts;
  facts.pid = pid;
  facts.uid = static_cast<uid_t>(uid);
  facts.gid = static_cast<gid_t>(gid);
  facts.start_tick = process_start_tick(pid);
  facts.pid_namespace_device = static_cast<std::uint64_t>(namespace_info.st_dev);
  facts.pid_namespace_inode = static_cast<std::uint64_t>(namespace_info.st_ino);
  facts.cgroup = trim(read_file("/proc/" + std::to_string(pid) + "/cgroup"));
  facts.executable = read_link("/proc/" + std::to_string(pid) + "/exe");
  facts.cap_effective = require_field(status, "CapEff");
  facts.cap_ambient = require_field(status, "CapAmb");
  facts.no_new_privileges = require_field(status, "NoNewPrivs") == "1";
  return facts;
}

int pidfd_open_native(pid_t pid) {
#ifdef SYS_pidfd_open
  return static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
#else
  errno = ENOSYS;
  return -1;
#endif
}

int pidfd_send_signal_native(int pidfd, int signal_number) {
#ifdef SYS_pidfd_send_signal
  return static_cast<int>(syscall(SYS_pidfd_send_signal, pidfd, signal_number,
                                  nullptr, 0));
#else
  errno = ENOSYS;
  return -1;
#endif
}

int pidfd_getfd_native(int pidfd, int target_fd) {
#ifdef SYS_pidfd_getfd
  return static_cast<int>(syscall(SYS_pidfd_getfd, pidfd, target_fd, 0));
#else
  errno = ENOSYS;
  return -1;
#endif
}

bool pidfd_is_live(int descriptor) {
  pollfd candidate{descriptor, POLLIN, 0};
  const int result = poll(&candidate, 1, 0);
  if (result < 0) throw Error("cannot poll pidfd");
  return result == 0;
}

int denied_open_errno(const std::string& path) {
  const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor >= 0) {
    close(descriptor);
    return 0;
  }
  return errno;
}

AttackResult perform_attack(uid_t attacker_uid, gid_t attacker_gid,
                            pid_t target_pid, int target_pidfd) {
  AttackResult result;
  if (setgroups(0, nullptr) != 0 || setresgid(attacker_gid, attacker_gid, attacker_gid) != 0 ||
      setresuid(attacker_uid, attacker_uid, attacker_uid) != 0 ||
      prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
      prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0) {
    result.setup_errno = errno == 0 ? EINVAL : errno;
    return result;
  }
  result.effective_uid = static_cast<int>(geteuid());
  result.effective_gid = static_cast<int>(getegid());

  errno = 0;
  result.signal_errno = kill(target_pid, 0) == 0 ? 0 : errno;
  result.proc_mem_errno =
      denied_open_errno("/proc/" + std::to_string(target_pid) + "/mem");

  errno = 0;
  const long ptrace_result = ptrace(PTRACE_ATTACH, target_pid, nullptr, nullptr);
  result.ptrace_errno = ptrace_result == 0 ? 0 : errno;
  if (ptrace_result == 0) {
    kill(target_pid, SIGCONT);
    ptrace(PTRACE_DETACH, target_pid, nullptr, nullptr);
  }

  char local_byte = 0;
  iovec local{&local_byte, 1};
  iovec remote{reinterpret_cast<void*>(static_cast<std::uintptr_t>(1)), 1};
  errno = 0;
  const ssize_t vm_result = process_vm_readv(target_pid, &local, 1, &remote, 1, 0);
  result.process_vm_errno = vm_result >= 0 ? 0 : errno;
  result.proc_fd_errno =
      denied_open_errno("/proc/" + std::to_string(target_pid) + "/fd/1");

  errno = 0;
  result.pidfd_signal_errno =
      pidfd_send_signal_native(target_pidfd, 0) == 0 ? 0 : errno;
  errno = 0;
  const int copied = pidfd_getfd_native(target_pidfd, STDOUT_FILENO);
  result.pidfd_getfd_errno = copied >= 0 ? 0 : errno;
  if (copied >= 0) close(copied);
  return result;
}

AttackResult attack_as(const ProcFacts& attacker, const ProcFacts& target,
                       int target_pidfd) {
  int channel[2];
  if (pipe2(channel, O_CLOEXEC) != 0) throw Error("cannot create attack result pipe");
  UniqueFd receive(channel[0]);
  UniqueFd send(channel[1]);
  const pid_t child = fork();
  if (child < 0) throw Error("cannot fork hostile principal probe");
  if (child == 0) {
    close(receive.get());
    const AttackResult result =
        perform_attack(attacker.uid, attacker.gid, target.pid, target_pidfd);
    const ssize_t written = write(send.get(), &result, sizeof(result));
    _exit(written == static_cast<ssize_t>(sizeof(result)) ? 0 : 121);
  }
  send = UniqueFd();
  AttackResult result;
  std::size_t offset = 0;
  while (offset < sizeof(result)) {
    const ssize_t count = read(receive.get(), reinterpret_cast<char*>(&result) + offset,
                               sizeof(result) - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count == 0) {
      break;
    } else if (errno != EINTR) {
      throw Error("cannot read hostile principal result");
    }
  }
  int status = 0;
  while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || offset != sizeof(result)) {
    throw Error("hostile principal probe child failed");
  }
  return result;
}

bool permission_denied(int error) { return error == EPERM || error == EACCES; }

bool attack_refused(const AttackResult& result, const ProcFacts& attacker) {
  return result.setup_errno == 0 &&
         result.effective_uid == static_cast<int>(attacker.uid) &&
         result.effective_gid == static_cast<int>(attacker.gid) &&
         result.signal_errno == EPERM && permission_denied(result.proc_mem_errno) &&
         result.ptrace_errno == EPERM && result.process_vm_errno == EPERM &&
         permission_denied(result.proc_fd_errno) &&
         result.pidfd_signal_errno == EPERM && result.pidfd_getfd_errno == EPERM;
}

std::string errno_name(int error) {
  if (error == EPERM) return "EPERM";
  if (error == EACCES) return "EACCES";
  if (error == ENOSYS) return "ENOSYS";
  if (error == 0) return "ALLOWED";
  return "ERRNO" + std::to_string(error);
}

bool zero_capability_word(const std::string& value) {
  if (value.empty()) return false;
  for (const unsigned char character : value) {
    if (character != '0') return false;
  }
  return true;
}

bool safe_unit_name(const std::string& unit) {
  if (unit.size() < 9 || unit.size() > 128 ||
      unit.substr(unit.size() - 8) != ".service") {
    return false;
  }
  for (const unsigned char character : unit) {
    if (!(std::isalnum(character) || character == '.' || character == '_' ||
          character == '-' || character == '@')) {
      return false;
    }
  }
  return true;
}

int cell_hold(unsigned seconds) {
  uid_t real_uid = 0;
  uid_t effective_uid = 0;
  uid_t saved_uid = 0;
  gid_t real_gid = 0;
  gid_t effective_gid = 0;
  gid_t saved_gid = 0;
  if (getresuid(&real_uid, &effective_uid, &saved_uid) != 0 ||
      getresgid(&real_gid, &effective_gid, &saved_gid) != 0) {
    throw Error("cannot read principal credential vector");
  }
  if (real_uid == 0 || real_gid == 0 || real_uid != effective_uid ||
      real_uid != saved_uid || real_gid != effective_gid || real_gid != saved_gid) {
    throw Error("cell requires one non-root kernel credential vector");
  }
  if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0 ||
      prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0 ||
      prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1) {
    throw Error("cell anti-injection posture is incomplete");
  }
  const ProcFacts facts = process_facts(getpid());
  if (!facts.no_new_privileges || !zero_capability_word(facts.cap_effective) ||
      !zero_capability_word(facts.cap_ambient) ||
      facts.cgroup.find(".service") == std::string::npos) {
    throw Error("cell process facts are incomplete");
  }
  std::cout << "LOOM_HOST_PRINCIPAL_CELL_READY"
            << " pid=" << facts.pid << " uid=" << facts.uid << " gid=" << facts.gid
            << " start_tick=" << facts.start_tick
            << " pidns_device=" << facts.pid_namespace_device
            << " pidns_inode=" << facts.pid_namespace_inode
            << " cgroup_sha256=" << sha256(facts.cgroup + "\n")
            << " executable_sha256=" << sha256(read_file(facts.executable, 64 * 1024 * 1024))
            << " no_new_privileges=true dumpable=false cap_effective=zero"
            << " cap_ambient=zero semantic_authority=Sounio action=9030"
            << " language_role=MATERIAL_PARITY launch_open=false\n";
  std::cout.flush();
  std::this_thread::sleep_for(std::chrono::seconds(seconds));
  return 0;
}

int measure(pid_t pid_a, pid_t pid_b, const std::string& unit_a,
            const std::string& unit_b) {
  if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0) {
    throw Error("host measurement requires root identity");
  }
  if (pid_a == pid_b || !safe_unit_name(unit_a) || !safe_unit_name(unit_b) ||
      unit_a == unit_b) {
    throw Error("principal cell identities are malformed or aliased");
  }
  const ProcFacts a_before = process_facts(pid_a);
  const ProcFacts b_before = process_facts(pid_b);
  if (a_before.uid == 0 || b_before.uid == 0 || a_before.gid == 0 ||
      b_before.gid == 0 || a_before.uid == b_before.uid ||
      a_before.gid == b_before.gid || a_before.cgroup == b_before.cgroup ||
      a_before.cgroup.find(unit_a) == std::string::npos ||
      b_before.cgroup.find(unit_b) == std::string::npos ||
      !a_before.no_new_privileges || !b_before.no_new_privileges ||
      !zero_capability_word(a_before.cap_effective) ||
      !zero_capability_word(b_before.cap_effective) ||
      !zero_capability_word(a_before.cap_ambient) ||
      !zero_capability_word(b_before.cap_ambient)) {
    throw Error("kernel-distinct principal facts are incomplete");
  }
  UniqueFd pidfd_a(pidfd_open_native(pid_a));
  UniqueFd pidfd_b(pidfd_open_native(pid_b));
  if (pidfd_a.get() < 0 || pidfd_b.get() < 0) {
    throw Error("pidfd_open is unavailable: " + std::string(std::strerror(errno)));
  }
  if (!pidfd_is_live(pidfd_a.get()) || !pidfd_is_live(pidfd_b.get())) {
    throw Error("principal cell died before hostile measurement");
  }

  const AttackResult a_to_b = attack_as(a_before, b_before, pidfd_b.get());
  const AttackResult b_to_a = attack_as(b_before, a_before, pidfd_a.get());
  if (!attack_refused(a_to_b, a_before) || !attack_refused(b_to_a, b_before)) {
    std::ostringstream reason;
    reason << "cross-principal attack admitted"
           << " a_to_b_signal=" << errno_name(a_to_b.signal_errno)
           << " a_to_b_mem=" << errno_name(a_to_b.proc_mem_errno)
           << " a_to_b_ptrace=" << errno_name(a_to_b.ptrace_errno)
           << " a_to_b_vm=" << errno_name(a_to_b.process_vm_errno)
           << " a_to_b_fd=" << errno_name(a_to_b.proc_fd_errno)
           << " a_to_b_pidfd_signal=" << errno_name(a_to_b.pidfd_signal_errno)
           << " a_to_b_pidfd_getfd=" << errno_name(a_to_b.pidfd_getfd_errno)
           << " b_to_a_signal=" << errno_name(b_to_a.signal_errno)
           << " b_to_a_mem=" << errno_name(b_to_a.proc_mem_errno)
           << " b_to_a_ptrace=" << errno_name(b_to_a.ptrace_errno)
           << " b_to_a_vm=" << errno_name(b_to_a.process_vm_errno)
           << " b_to_a_fd=" << errno_name(b_to_a.proc_fd_errno)
           << " b_to_a_pidfd_signal=" << errno_name(b_to_a.pidfd_signal_errno)
           << " b_to_a_pidfd_getfd=" << errno_name(b_to_a.pidfd_getfd_errno);
    throw Error(reason.str());
  }

  const ProcFacts a_after = process_facts(pid_a);
  const ProcFacts b_after = process_facts(pid_b);
  if (a_after.start_tick != a_before.start_tick ||
      b_after.start_tick != b_before.start_tick || !pidfd_is_live(pidfd_a.get()) ||
      !pidfd_is_live(pidfd_b.get())) {
    throw Error("principal identity changed during hostile measurement");
  }
  const std::string boot_id = trim(read_file("/proc/sys/kernel/random/boot_id"));
  std::cout << "LOOM_HOST_PRINCIPAL_CELL_MEASUREMENT PASS"
            << " semantic_authority=Sounio action=9030"
            << " producing_language=C++20 language_role=MATERIAL_PARITY"
            << " transitory=true pid_a=" << pid_a << " uid_a=" << a_before.uid
            << " gid_a=" << a_before.gid << " start_tick_a=" << a_before.start_tick
            << " pid_b=" << pid_b << " uid_b=" << b_before.uid
            << " gid_b=" << b_before.gid << " start_tick_b=" << b_before.start_tick
            << " uid_distinct=true gid_distinct=true cgroup_distinct=true"
            << " pidfd_live=true start_tick_stable=true boot_id_sha256="
            << sha256(boot_id + "\n")
            << " signal_cross_uid=EPERM proc_mem_cross_uid="
            << errno_name(a_to_b.proc_mem_errno)
            << " ptrace_cross_uid=EPERM process_vm_readv_cross_uid=EPERM"
            << " proc_fd_cross_uid=" << errno_name(a_to_b.proc_fd_errno)
            << " copied_pidfd_signal=EPERM copied_pidfd_getfd=EPERM"
            << " reciprocal_attacks=refused no_new_privileges=true"
            << " capabilities=zero kernel_distinct_principal_candidate=true"
            << " same_uid_peer_isolation=false material_grant=false"
            << " grant_extinction=false exec_attached=false commit_attached=false"
            << " ci_attached=false launch_open=false\n";
  return 0;
}

int measure_same_principal_sabotage(pid_t pid_a, pid_t pid_b,
                                    const std::string& unit_a,
                                    const std::string& unit_b) {
  if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0) {
    throw Error("host sabotage measurement requires root identity");
  }
  if (pid_a == pid_b || !safe_unit_name(unit_a) || !safe_unit_name(unit_b) ||
      unit_a == unit_b) {
    throw Error("sabotage cell identities are malformed or aliased");
  }
  const ProcFacts a_before = process_facts(pid_a);
  const ProcFacts b_before = process_facts(pid_b);
  if (a_before.uid == 0 || a_before.gid == 0 ||
      a_before.uid != b_before.uid || a_before.gid != b_before.gid ||
      a_before.cgroup == b_before.cgroup ||
      a_before.cgroup.find(unit_a) == std::string::npos ||
      b_before.cgroup.find(unit_b) == std::string::npos ||
      !a_before.no_new_privileges || !b_before.no_new_privileges ||
      !zero_capability_word(a_before.cap_effective) ||
      !zero_capability_word(b_before.cap_effective) ||
      !zero_capability_word(a_before.cap_ambient) ||
      !zero_capability_word(b_before.cap_ambient)) {
    throw Error("same-principal sabotage posture is incomplete");
  }
  UniqueFd pidfd_a(pidfd_open_native(pid_a));
  UniqueFd pidfd_b(pidfd_open_native(pid_b));
  if (pidfd_a.get() < 0 || pidfd_b.get() < 0 ||
      !pidfd_is_live(pidfd_a.get()) || !pidfd_is_live(pidfd_b.get())) {
    throw Error("sabotage pidfd identity is unavailable");
  }
  const AttackResult a_to_b = attack_as(a_before, b_before, pidfd_b.get());
  const AttackResult b_to_a = attack_as(b_before, a_before, pidfd_a.get());
  const auto admitted_signal_pair = [](const AttackResult& result,
                                       const ProcFacts& attacker) {
    return result.setup_errno == 0 &&
           result.effective_uid == static_cast<int>(attacker.uid) &&
           result.effective_gid == static_cast<int>(attacker.gid) &&
           result.signal_errno == 0 && result.pidfd_signal_errno == 0;
  };
  if (!admitted_signal_pair(a_to_b, a_before) ||
      !admitted_signal_pair(b_to_a, b_before)) {
    throw Error("same-principal sabotage did not admit both signal probes");
  }
  const ProcFacts a_after = process_facts(pid_a);
  const ProcFacts b_after = process_facts(pid_b);
  if (a_after.start_tick != a_before.start_tick ||
      b_after.start_tick != b_before.start_tick ||
      !pidfd_is_live(pidfd_a.get()) || !pidfd_is_live(pidfd_b.get())) {
    throw Error("sabotage process identity changed during signal probes");
  }
  std::cout << "LOOM_HOST_PRINCIPAL_CELL_SAME_PRINCIPAL_SABOTAGE PASS"
            << " semantic_authority=Sounio action=9030"
            << " producing_language=C++20 language_role=MATERIAL_PARITY"
            << " intervention=kernel-distinct-principal-removed"
            << " pid_a=" << pid_a << " pid_b=" << pid_b
            << " shared_uid=" << a_before.uid << " shared_gid=" << a_before.gid
            << " cgroup_distinct=true no_new_privileges=true capabilities=zero"
            << " signal_cross_cell=ALLOWED copied_pidfd_signal=ALLOWED"
            << " proc_mem_cross_cell=" << errno_name(a_to_b.proc_mem_errno)
            << " ptrace_cross_cell=" << errno_name(a_to_b.ptrace_errno)
            << " process_vm_readv_cross_cell=" << errno_name(a_to_b.process_vm_errno)
            << " proc_fd_cross_cell=" << errno_name(a_to_b.proc_fd_errno)
            << " copied_pidfd_getfd=" << errno_name(a_to_b.pidfd_getfd_errno)
            << " reciprocal_signal_probes=ALLOWED start_tick_stable=true"
            << " pidfd_live=true process_state_unchanged=true causal_control=PASS"
            << " material_grant=false grant_extinction=false exec_attached=false"
            << " launch_open=false\n";
  return 0;
}

int selftest() {
  if (!parse_u64("18446744073709551615") || parse_u64("18446744073709551616") ||
      parse_u64("-1") || sha256("abc") !=
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
    throw Error("parser or digest selftest failed");
  }
  const ProcFacts self = process_facts(getpid());
  AttackResult refused;
  refused.effective_uid = static_cast<int>(self.uid);
  refused.effective_gid = static_cast<int>(self.gid);
  refused.signal_errno = EPERM;
  refused.proc_mem_errno = EACCES;
  refused.ptrace_errno = EPERM;
  refused.process_vm_errno = EPERM;
  refused.proc_fd_errno = EACCES;
  refused.pidfd_signal_errno = EPERM;
  refused.pidfd_getfd_errno = EPERM;
  if (!attack_refused(refused, self) || self.start_tick == 0 || self.cgroup.empty()) {
    throw Error("hostile result classifier selftest failed");
  }
  std::cout << "LOOM_HOST_PRINCIPAL_CELL_SELFTEST PASS"
            << " language=C++20 role=MATERIAL_PARITY transitory=true"
            << " semantic_authority=Sounio action=9030 parser=bounded"
            << " digest=verified proc_identity=read hostile_classifier=closed"
            << " launch_open=false material_grant=false exec_attached=false\n";
  return 0;
}

struct Options {
  std::string mode;
  unsigned seconds = 45;
  pid_t pid_a = 0;
  pid_t pid_b = 0;
  std::string unit_a;
  std::string unit_b;
};

Options parse_options(int argc, char** argv) {
  if (argc < 2) throw Error("missing mode");
  Options options;
  options.mode = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (index + 1 >= argc) throw Error("option omitted value: " + argument);
    const std::string value = argv[++index];
    if (argument == "--seconds") {
      const auto parsed = parse_u64(value);
      if (!parsed || *parsed < 5 || *parsed > 300) throw Error("invalid hold duration");
      options.seconds = static_cast<unsigned>(*parsed);
    } else if (argument == "--pid-a" || argument == "--pid-b") {
      const auto parsed = parse_u64(value);
      if (!parsed || *parsed <= 1 || *parsed > std::numeric_limits<pid_t>::max()) {
        throw Error("invalid principal cell PID");
      }
      if (argument == "--pid-a") options.pid_a = static_cast<pid_t>(*parsed);
      if (argument == "--pid-b") options.pid_b = static_cast<pid_t>(*parsed);
    } else if (argument == "--unit-a") {
      options.unit_a = value;
    } else if (argument == "--unit-b") {
      options.unit_b = value;
    } else {
      throw Error("unknown option: " + argument);
    }
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.mode == "--selftest") return selftest();
    if (options.mode == "--cell-hold") return cell_hold(options.seconds);
    if (options.mode == "--measure") {
      if (options.pid_a == 0 || options.pid_b == 0 || options.unit_a.empty() ||
          options.unit_b.empty()) {
        throw Error("measurement requires two PIDs and two units");
      }
      return measure(options.pid_a, options.pid_b, options.unit_a, options.unit_b);
    }
    if (options.mode == "--measure-same-principal-sabotage") {
      if (options.pid_a == 0 || options.pid_b == 0 || options.unit_a.empty() ||
          options.unit_b.empty()) {
        throw Error("sabotage measurement requires two PIDs and two units");
      }
      return measure_same_principal_sabotage(options.pid_a, options.pid_b,
                                             options.unit_a, options.unit_b);
    }
    throw Error("unknown mode");
  } catch (const std::exception& error) {
    std::cerr << "loom-host-principal-cell: REFUSE reason=" << error.what() << "\n";
    return 70;
  }
}
