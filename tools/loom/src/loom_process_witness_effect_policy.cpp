#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <openssl/sha.h>

#include <linux/audit.h>
#include <linux/bpf.h>
#include <linux/filter.h>
#include <linux/io_uring.h>
#include <linux/landlock.h>
#include <linux/memfd.h>
#include <linux/seccomp.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kPolicyManifestSha256 =
    "d66b13252479252d5922ee0091e51a5bdb6a5eca9a592bb21f5db9dde344fee9";
constexpr std::uint32_t kRefuse = SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA);

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct Manifest {
  std::map<std::string, std::string> fields;
  std::string digest;

  const std::string& require(const std::string& key) const {
    const auto found = fields.find(key);
    if (found == fields.end() || found->second.empty()) {
      throw Error("manifest omitted " + key);
    }
    return found->second;
  }
};

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

std::string read_regular_file(const std::string& path) {
  struct stat info {};
  if (lstat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_nlink != 1) {
    throw Error("policy manifest is not one regular file");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) throw Error("cannot open policy manifest");
  std::string contents((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
  if (input.bad() || contents.empty() || contents.size() > 128 * 1024) {
    throw Error("cannot read bounded policy manifest");
  }
  return contents;
}

Manifest load_policy_manifest(const std::string& path) {
  const std::string contents = read_regular_file(path);
  Manifest manifest;
  manifest.digest = sha256(contents);
  if (manifest.digest != kPolicyManifestSha256) {
    throw Error("frozen Sounio V2 policy manifest hash mismatch");
  }
  std::size_t offset = 0;
  while (offset < contents.size()) {
    const std::size_t newline = contents.find('\n', offset);
    const std::size_t end =
        newline == std::string::npos ? contents.size() : newline;
    const std::string line = contents.substr(offset, end - offset);
    if (!line.empty()) {
      const std::size_t equals = line.find('=');
      if (equals == std::string::npos || equals == 0) {
        throw Error("malformed policy manifest line");
      }
      const auto [_, inserted] =
          manifest.fields.emplace(line.substr(0, equals), line.substr(equals + 1));
      if (!inserted) throw Error("duplicate policy manifest field");
    }
    if (newline == std::string::npos) break;
    offset = newline + 1;
  }
  if (manifest.require("schema") !=
          "loom-process-witness-effect-policy-plan-v2-freeze-v1" ||
      manifest.require("stage") != "SEMANTICS_FROZEN" ||
      manifest.require("producing_language") != "Sounio" ||
      manifest.require("language_role") != "SEMANTIC_POLICY_PLAN" ||
      manifest.require("semantic_authority") != "Sounio" ||
      manifest.require("action") != "9025" ||
      manifest.require("bundle_sha256") !=
          "5d9f3528e8dd5238c388f5bfd00606eeb13ddfa927ab48bca296fc69b9e2d236" ||
      manifest.require("allowed_syscall_count") != "4" ||
      manifest.require("allowed_syscalls") != "0,1,60,322" ||
      manifest.require("read_constraint") != "fd0" ||
      manifest.require("write_constraint") != "fd1_or_fd2" ||
      manifest.require("execveat_constraint") != "fd3_and_AT_EMPTY_PATH" ||
      manifest.require("architecture") != "AUDIT_ARCH_X86_64" ||
      manifest.require("architecture_mismatch") != "KILL_PROCESS" ||
      manifest.require("default_action") != "ERRNO_EP1" ||
      manifest.require("allowlist_kind") != "positive" ||
      manifest.require("argument_constraints") != "required" ||
      manifest.require("blacklist_fallback") != "false" ||
      manifest.require("landlock_required") != "true" ||
      manifest.require("landlock_fallback") != "false" ||
      manifest.require("v1_sufficient_for_native") != "false" ||
      manifest.require("v2_required_for_native") != "true" ||
      manifest.require("material_coverage") != "false" ||
      manifest.require("complete_effects") != "false" ||
      manifest.require("material_execution") != "false" ||
      manifest.require("claim_ready") != "false") {
    throw Error("Sounio V2 policy contract drifted");
  }
  return manifest;
}

class FilterBuilder {
 public:
  std::vector<sock_filter> code;

  std::size_t statement(std::uint16_t operation, std::uint32_t value) {
    code.push_back(BPF_STMT(operation, value));
    return code.size() - 1;
  }

  std::size_t equal(std::uint32_t value) {
    code.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, value, 0, 0));
    return code.size() - 1;
  }

  void patch_true(std::size_t jump, std::size_t target) {
    patch(jump, target, true);
  }

  void patch_false(std::size_t jump, std::size_t target) {
    patch(jump, target, false);
  }

 private:
  void patch(std::size_t jump, std::size_t target, bool branch_true) {
    if (target <= jump || target - jump - 1 > 255) {
      throw Error("seccomp branch exceeds classic-BPF range");
    }
    const auto delta = static_cast<std::uint8_t>(target - jump - 1);
    if (branch_true) {
      code.at(jump).jt = delta;
    } else {
      code.at(jump).jf = delta;
    }
  }
};

void emit_allow_one_argument(FilterBuilder& builder, std::uint32_t syscall_number,
                             std::uint32_t argument_offset,
                             std::uint32_t expected_low) {
  const std::size_t syscall_jump = builder.equal(syscall_number);
  builder.statement(BPF_LD | BPF_W | BPF_ABS, argument_offset);
  const std::size_t low_jump = builder.equal(expected_low);
  builder.statement(BPF_LD | BPF_W | BPF_ABS, argument_offset + 4);
  const std::size_t high_jump = builder.equal(0);
  const std::size_t allow =
      builder.statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
  const std::size_t deny = builder.statement(BPF_RET | BPF_K, kRefuse);
  const std::size_t next = builder.code.size();
  builder.patch_false(syscall_jump, next);
  builder.patch_false(low_jump, deny);
  builder.patch_true(high_jump, allow);
  builder.patch_false(high_jump, deny);
}

void emit_allow_write(FilterBuilder& builder) {
  const std::size_t syscall_jump = builder.equal(SYS_write);
  builder.statement(BPF_LD | BPF_W | BPF_ABS,
                    offsetof(seccomp_data, args[0]));
  const std::size_t fd1 = builder.equal(STDOUT_FILENO);
  const std::size_t fd2 = builder.equal(STDERR_FILENO);
  const std::size_t high_load = builder.code.size();
  builder.statement(BPF_LD | BPF_W | BPF_ABS,
                    offsetof(seccomp_data, args[0]) + 4);
  const std::size_t high = builder.equal(0);
  const std::size_t allow =
      builder.statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
  const std::size_t deny = builder.statement(BPF_RET | BPF_K, kRefuse);
  const std::size_t next = builder.code.size();
  builder.patch_false(syscall_jump, next);
  builder.patch_true(fd1, high_load);
  builder.patch_true(fd2, high_load);
  builder.patch_false(fd2, deny);
  builder.patch_true(high, allow);
  builder.patch_false(high, deny);
}

void emit_allow_execveat(FilterBuilder& builder) {
  const std::size_t syscall_jump = builder.equal(SYS_execveat);
  builder.statement(BPF_LD | BPF_W | BPF_ABS,
                    offsetof(seccomp_data, args[0]));
  const std::size_t fd_low = builder.equal(3);
  builder.statement(BPF_LD | BPF_W | BPF_ABS,
                    offsetof(seccomp_data, args[0]) + 4);
  const std::size_t fd_high = builder.equal(0);
  builder.statement(BPF_LD | BPF_W | BPF_ABS,
                    offsetof(seccomp_data, args[4]));
  const std::size_t flag_low = builder.equal(AT_EMPTY_PATH);
  builder.statement(BPF_LD | BPF_W | BPF_ABS,
                    offsetof(seccomp_data, args[4]) + 4);
  const std::size_t flag_high = builder.equal(0);
  const std::size_t allow =
      builder.statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
  const std::size_t deny = builder.statement(BPF_RET | BPF_K, kRefuse);
  const std::size_t next = builder.code.size();
  builder.patch_false(syscall_jump, next);
  builder.patch_false(fd_low, deny);
  builder.patch_false(fd_high, deny);
  builder.patch_false(flag_low, deny);
  builder.patch_true(flag_high, allow);
  builder.patch_false(flag_high, deny);
}

std::uint32_t sabotage_syscall(int family) {
  switch (family) {
    case 1: return SYS_execveat;
#ifdef SYS_clone3
    case 2: return SYS_clone3;
#else
    case 2: return SYS_clone;
#endif
    case 3: return SYS_openat;
    case 4: return SYS_dup3;
    case 5: return SYS_mmap;
#ifdef SYS_io_uring_setup
    case 6: return SYS_io_uring_setup;
#else
    case 6: return 425;
#endif
    case 7: return SYS_socket;
    case 8: return SYS_socket;
#ifdef SYS_memfd_create
    case 9: return SYS_memfd_create;
#else
    case 9: return 319;
#endif
#ifdef SYS_bpf
    case 10: return SYS_bpf;
#else
    case 10: return 321;
#endif
    case 11: return SYS_openat;
    case 12: return SYS_getpid;
    default: throw Error("invalid sabotage family");
  }
}

void emit_sabotage_allow(FilterBuilder& builder, int family) {
  const std::uint32_t syscall_number = sabotage_syscall(family);
  const std::size_t syscall_jump = builder.equal(syscall_number);
  if (family == 7 || family == 8) {
    builder.statement(BPF_LD | BPF_W | BPF_ABS,
                      offsetof(seccomp_data, args[0]));
    const std::size_t domain =
        builder.equal(family == 7 ? AF_INET : AF_UNIX);
    const std::size_t allow =
        builder.statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
    const std::size_t next = builder.code.size();
    builder.patch_false(syscall_jump, next);
    builder.patch_true(domain, allow);
    builder.patch_false(domain, next);
  } else {
    const std::size_t allow =
        builder.statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
    const std::size_t next = builder.code.size();
    builder.patch_true(syscall_jump, allow);
    builder.patch_false(syscall_jump, next);
  }
}

std::vector<sock_filter> compile_filter(int sabotage_family) {
  if (sabotage_family < 0 || sabotage_family > 12) {
    throw Error("invalid sabotage selector");
  }
  FilterBuilder builder;
  builder.statement(BPF_LD | BPF_W | BPF_ABS,
                    offsetof(seccomp_data, arch));
  const std::size_t architecture = builder.equal(AUDIT_ARCH_X86_64);
  builder.statement(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS);
  const std::size_t architecture_ok = builder.code.size();
  builder.patch_true(architecture, architecture_ok);
  builder.statement(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr));

  if (sabotage_family != 0) emit_sabotage_allow(builder, sabotage_family);

  emit_allow_one_argument(builder, SYS_read,
                          offsetof(seccomp_data, args[0]), STDIN_FILENO);
  emit_allow_write(builder);
  const std::size_t exit_jump = builder.equal(SYS_exit);
  const std::size_t exit_allow =
      builder.statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
  const std::size_t after_exit = builder.code.size();
  builder.patch_true(exit_jump, exit_allow);
  builder.patch_false(exit_jump, after_exit);
  emit_allow_execveat(builder);
  builder.statement(BPF_RET | BPF_K, kRefuse);
  return builder.code;
}

std::string filter_digest(const std::vector<sock_filter>& filter) {
  return sha256(filter.data(), filter.size() * sizeof(sock_filter));
}

int install_landlock() {
#if defined(SYS_landlock_create_ruleset) && defined(SYS_landlock_restrict_self)
  const int abi = static_cast<int>(
      syscall(SYS_landlock_create_ruleset, nullptr, 0,
              LANDLOCK_CREATE_RULESET_VERSION));
  if (abi < 4) return -1;
  landlock_ruleset_attr attributes{};
  attributes.handled_access_fs =
      LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_WRITE_FILE |
      LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR |
      LANDLOCK_ACCESS_FS_REMOVE_DIR | LANDLOCK_ACCESS_FS_REMOVE_FILE |
      LANDLOCK_ACCESS_FS_MAKE_CHAR | LANDLOCK_ACCESS_FS_MAKE_DIR |
      LANDLOCK_ACCESS_FS_MAKE_REG | LANDLOCK_ACCESS_FS_MAKE_SOCK |
      LANDLOCK_ACCESS_FS_MAKE_FIFO | LANDLOCK_ACCESS_FS_MAKE_BLOCK |
      LANDLOCK_ACCESS_FS_MAKE_SYM | LANDLOCK_ACCESS_FS_REFER |
      LANDLOCK_ACCESS_FS_TRUNCATE;
  attributes.handled_access_net =
      LANDLOCK_ACCESS_NET_BIND_TCP | LANDLOCK_ACCESS_NET_CONNECT_TCP;
  const int ruleset = static_cast<int>(
      syscall(SYS_landlock_create_ruleset, &attributes, sizeof(attributes), 0));
  if (ruleset < 0) return -1;
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
      syscall(SYS_landlock_restrict_self, ruleset, 0) != 0) {
    close(ruleset);
    return -1;
  }
  close(ruleset);
  return abi;
#else
  return -1;
#endif
}

bool install_seccomp(const std::vector<sock_filter>& filter) {
  if (filter.empty() || filter.size() > std::numeric_limits<unsigned short>::max()) {
    return false;
  }
  sock_fprog program{
      static_cast<unsigned short>(filter.size()),
      const_cast<sock_filter*>(filter.data())};
  return prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0 &&
         prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0;
}

long probe_family(int family) {
  errno = 0;
  switch (family) {
    case 1: {
      char program[] = "probe";
      char* const arguments[] = {program, nullptr};
      char* const environment[] = {nullptr};
      return syscall(SYS_execveat, -1, "", arguments, environment, 0);
    }
    case 2:
#ifdef SYS_clone3
      return syscall(SYS_clone3, nullptr, 0);
#else
      return syscall(SYS_clone, 0, nullptr, nullptr, nullptr, nullptr);
#endif
    case 3:
      return syscall(SYS_openat, AT_FDCWD, "/tmp/loom-effect-policy-probe",
                     O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    case 4:
      return syscall(SYS_dup3, STDIN_FILENO, 9, O_CLOEXEC);
    case 5:
      return syscall(SYS_mmap, nullptr, 4096, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    case 6: {
      io_uring_params parameters{};
#ifdef SYS_io_uring_setup
      return syscall(SYS_io_uring_setup, 1, &parameters);
#else
      return syscall(425, 1, &parameters);
#endif
    }
    case 7: return syscall(SYS_socket, AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    case 8: return syscall(SYS_socket, AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    case 9:
#ifdef SYS_memfd_create
      return syscall(SYS_memfd_create, "loom-effect-probe", MFD_CLOEXEC);
#else
      return syscall(319, "loom-effect-probe", MFD_CLOEXEC);
#endif
    case 10: {
      union bpf_attr attributes {};
      attributes.map_type = BPF_MAP_TYPE_ARRAY;
      attributes.key_size = 4;
      attributes.value_size = 4;
      attributes.max_entries = 1;
#ifdef SYS_bpf
      return syscall(SYS_bpf, BPF_MAP_CREATE, &attributes, sizeof(attributes));
#else
      return syscall(321, BPF_MAP_CREATE, &attributes, sizeof(attributes));
#endif
    }
    case 11:
      return syscall(SYS_openat, AT_FDCWD, "/proc/self/mem",
                     O_RDONLY | O_CLOEXEC, 0);
    case 12: return syscall(SYS_getpid);
    default: return -2;
  }
}

bool wait_success(pid_t pid) {
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) return false;
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int treatment_child(int family, const std::vector<sock_filter>& filter,
                    bool require_landlock) {
  if ((require_landlock && install_landlock() < 4) ||
      !install_seccomp(filter)) {
    _exit(90);
  }
  const long result = probe_family(family);
  const int saved_errno = errno;
  syscall(SYS_exit, result == -1 && saved_errno == EPERM ? 0 : 91);
  __builtin_unreachable();
}

bool run_treatment(int family, const std::vector<sock_filter>& filter,
                   bool require_landlock) {
  const pid_t pid = fork();
  if (pid < 0) throw Error("cannot fork treatment cell");
  if (pid == 0) treatment_child(family, filter, require_landlock);
  return wait_success(pid);
}

bool run_allowed_io(const std::vector<sock_filter>& filter,
                    bool require_landlock) {
  const pid_t pid = fork();
  if (pid < 0) throw Error("cannot fork allowlist cell");
  if (pid == 0) {
    if ((require_landlock && install_landlock() < 4) ||
        !install_seccomp(filter)) {
      _exit(90);
    }
    char byte = 0;
    const long read_result = syscall(SYS_read, STDIN_FILENO, &byte, 0);
    const long write_result = syscall(SYS_write, STDOUT_FILENO, &byte, 0);
    syscall(SYS_exit, read_result == 0 && write_result == 0 ? 0 : 91);
    __builtin_unreachable();
  }
  return wait_success(pid);
}

int query_landlock_abi() {
#ifdef SYS_landlock_create_ruleset
  return static_cast<int>(syscall(SYS_landlock_create_ruleset, nullptr, 0,
                                  LANDLOCK_CREATE_RULESET_VERSION));
#else
  return -1;
#endif
}

int selftest(const std::string& policy_manifest_path) {
  const Manifest manifest = load_policy_manifest(policy_manifest_path);
  const std::vector<sock_filter> treatment_filter = compile_filter(0);
  const std::string treatment_digest = filter_digest(treatment_filter);
  errno = 0;
  const int landlock_abi = query_landlock_abi();
  const int landlock_errno = landlock_abi < 0 ? errno : 0;
  const bool landlock_local = landlock_abi >= 4;
  const bool allowed_io = run_allowed_io(treatment_filter, landlock_local);
  if (!allowed_io) {
    throw Error("kernel cannot realize frozen seccomp surface");
  }

  int treatments = 0;
  for (int family = 1; family <= 12; ++family) {
    if (!run_treatment(family, treatment_filter, landlock_local)) {
      throw Error("kernel treatment did not refuse family " +
                  std::to_string(family));
    }
    ++treatments;
  }

  std::string sabotage_material;
  int structural_sabotages = 0;
  for (int family = 1; family <= 12; ++family) {
    const auto filter = compile_filter(family);
    const std::string digest = filter_digest(filter);
    if (digest == treatment_digest) {
      throw Error("sabotage filter equals treatment filter");
    }
    sabotage_material += std::to_string(family) + ":" + digest +
                         ":landlock-family-" + std::to_string(family) + "\n";
    ++structural_sabotages;
  }
  const std::string sabotage_set_digest = sha256(sabotage_material);
  std::cout
      << "LOOM_PROCESS_WITNESS_EFFECT_POLICY_SELFTEST PASS"
      << " semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY"
      << " transitory=true action=9025 policy_v2_bound=true"
      << " policy_manifest_sha256=" << manifest.digest
      << " landlock_abi=" << landlock_abi
      << " landlock_errno=" << landlock_errno
      << " landlock_local=" << (landlock_local ? "available" : "unavailable")
      << " seccomp_default=EPERM architecture=AUDIT_ARCH_X86_64"
      << " allowed_syscalls=0+1+60+322 allowed_io=true"
      << " seccomp_treatments=" << treatments
      << " local_landlock_treatments=" << (landlock_local ? treatments : 0)
      << " structural_sabotages=" << structural_sabotages
      << " material_sabotages=0"
      << " filter_sha256=" << treatment_digest
      << " sabotage_set_sha256=" << sabotage_set_digest
      << " host_gate_required=true"
      << " material_coverage=false complete_effects=false"
      << " material_execution=false launch_open=false recycle_open=false"
      << " exec_attached=false commit_attached=false ci_attached=false"
      << " parity_open=false claim_ready=false\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string_view(argv[1]) == "--selftest" &&
        std::string_view(argv[2]) == "--policy-manifest") {
      return selftest(argv[3]);
    }
    std::cerr << "usage: loom-process-witness-effect-policy --selftest "
                 "--policy-manifest <frozen-v2-manifest>\n";
    return 64;
  } catch (const std::exception& error) {
    std::cerr << "LOOM_PROCESS_WITNESS_EFFECT_POLICY_CLOSED reason="
              << error.what() << '\n';
    return 70;
  }
}
