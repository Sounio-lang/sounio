#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <openssl/sha.h>

#include <dirent.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/io_uring.h>
#include <linux/memfd.h>
#include <linux/seccomp.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

#ifndef LOOM_EFFECT_POLICY_VERSION
#define LOOM_EFFECT_POLICY_VERSION 3
#endif

static_assert(LOOM_EFFECT_POLICY_VERSION == 3 ||
              LOOM_EFFECT_POLICY_VERSION == 4);
#if LOOM_EFFECT_POLICY_VERSION == 4
constexpr std::string_view kPolicyManifestSha256 =
    "60cff91db90e9214e62a6fa5b45521249e31649c63dce297683ca477fcd3d627";
constexpr std::string_view kPolicySchema =
    "loom-process-witness-effect-policy-plan-v4-freeze-v1";
constexpr std::string_view kPolicyBundleSha256 =
    "3bce80f8d74098470566b3ce3c0b872992ac0cf1d42ce9c64df4bc06ae57901f";
constexpr std::string_view kPolicyRootPath =
    "/loom/effect-policy-v4.freeze.v1";
constexpr std::string_view kPolicyFilename = "effect-policy-v4.freeze.v1";
constexpr std::string_view kSelftestPrefix =
    "LOOM_PROCESS_WITNESS_EFFECT_POLICY_V4_SELFTEST PASS";
constexpr std::string_view kReadyPrefix =
    "LOOM_PROCESS_WITNESS_EFFECT_POLICY_V4_ROOT_READY PASS";
#else
constexpr std::string_view kPolicyManifestSha256 =
    "40407323594e37d44b9002d1cdd390677416048221ace446693919f8415ca480";
constexpr std::string_view kPolicySchema =
    "loom-process-witness-effect-policy-plan-v3-freeze-v1";
constexpr std::string_view kPolicyBundleSha256 =
    "e365f0b1e0028bd0cddd129e1110126dd82b0c33ca268d427c39fe870b0efe34";
constexpr std::string_view kPolicyRootPath =
    "/loom/effect-policy-v3.freeze.v1";
constexpr std::string_view kPolicyFilename = "effect-policy-v3.freeze.v1";
constexpr std::string_view kSelftestPrefix =
    "LOOM_PROCESS_WITNESS_EFFECT_POLICY_V3_SELFTEST PASS";
constexpr std::string_view kReadyPrefix =
    "LOOM_PROCESS_WITNESS_EFFECT_POLICY_V3_ROOT_READY PASS";
#endif
constexpr std::string_view kPayloadManifestSha256 =
    "624ccd7297778803eff8d9972a33d5e55fb022f9e7e37f444f0aee13c22fb4da";
constexpr std::string_view kPayloadSha256 =
    "7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d";
constexpr std::uint32_t kRefuse =
    SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA);

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
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

std::string read_regular_file(const std::string& path,
                              std::size_t maximum = 128 * 1024) {
  struct stat info {};
  if (lstat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_nlink != 1 || info.st_size <= 0 ||
      static_cast<std::uint64_t>(info.st_size) > maximum) {
    throw Error("policy manifest is not one bounded regular file");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) throw Error("cannot open policy manifest");
  std::string contents((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
  if (input.bad() || contents.size() != static_cast<std::size_t>(info.st_size)) {
    throw Error("cannot read policy manifest");
  }
  return contents;
}

void require_line(std::string_view contents, std::string_view line) {
  std::string needle;
  needle.reserve(line.size() + 2);
  needle.push_back('\n');
  needle.append(line);
  needle.push_back('\n');
  std::string framed;
  framed.reserve(contents.size() + 2);
  framed.push_back('\n');
  framed.append(contents);
  if (framed.back() != '\n') framed.push_back('\n');
  if (framed.find(needle) == std::string::npos) {
    throw Error("Sounio V3 policy contract omitted " + std::string(line));
  }
}

std::string load_policy_manifest(const std::string& path) {
  const std::string contents = read_regular_file(path, 32 * 1024 * 1024);
  const std::string digest = sha256(contents);
  if (digest != kPolicyManifestSha256) {
    throw Error("frozen Sounio V3 policy manifest hash mismatch");
  }
  for (const std::string_view line : {
           "stage=SEMANTICS_FROZEN",
           "producing_language=Sounio",
           "language_role=SEMANTIC_POLICY_PLAN",
           "semantic_authority=Sounio",
           "action=9025",
           "allowed_syscall_count=4",
           "allowed_syscalls=0,1,60,322",
           "read_constraint=fd0",
           "write_constraint=fd1_or_fd2",
           "execveat_constraint=fd3_and_AT_EMPTY_PATH",
           "architecture=AUDIT_ARCH_X86_64",
           "architecture_mismatch=KILL_PROCESS",
           "default_action=ERRNO_EP1",
           "allowlist_kind=positive",
           "argument_constraints=required",
           "blacklist_fallback=false",
           "object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE",
           "root_read_only=true",
           "dynamic_linker_visible=false",
           "host_root_visible=false",
           "pathname_syscalls_after_filter=0",
           "landlock_required=false",
           "landlock_fallback=false",
           "family_10_probe=personality_change",
           "static_native_required=true",
           "static_sounio_payload_required=true",
           "material_coverage=false",
           "complete_effects=false",
           "material_execution=false",
           "launch_open=false",
           "recycle_open=false",
           "exec_attached=false",
           "commit_attached=false",
           "ci_attached=false",
           "parity_open=false",
           "claim_ready=false",
       }) {
    require_line(contents, line);
  }
  require_line(contents, "schema=" + std::string(kPolicySchema));
  require_line(contents,
               "bundle_sha256=" + std::string(kPolicyBundleSha256));
#if LOOM_EFFECT_POLICY_VERSION == 4
  for (const std::string_view line : {
           "systemd_mount_path=/run/systemd/incoming",
           "systemd_mount_source=/run/systemd/propagate/EXACT_UNIT",
           "systemd_mount_principal_writable=false",
           "systemd_mount_ready_contents=empty",
           "bootstrap_treatment_code=0",
           "bootstrap_missing_code=226",
           "v3_materializable=false",
           "v4_required_for_native=true",
       }) {
    require_line(contents, line);
  }
#else
  require_line(contents, "v2_materializable=false");
  require_line(contents, "v3_required_for_native=true");
#endif
  return digest;
}

void require_directory(const std::string& path, bool empty) {
  struct stat info {};
  if (lstat(path.c_str(), &info) != 0 || !S_ISDIR(info.st_mode) ||
      info.st_uid != 0 || info.st_gid != 0) {
    throw Error("immutable-root directory metadata drifted: " + path +
                " uid=" + std::to_string(info.st_uid) +
                " gid=" + std::to_string(info.st_gid) +
                " mode=" + std::to_string(info.st_mode & 07777));
  }
  if ((info.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    struct statvfs filesystem {};
    if (statvfs(path.c_str(), &filesystem) != 0 ||
        (filesystem.f_flag & ST_RDONLY) == 0) {
      throw Error("immutable-root directory is writable by principal: " + path);
    }
  }
  if (!empty) return;
  DIR* directory = opendir(path.c_str());
  if (directory == nullptr) throw Error("cannot inspect root directory: " + path);
  int entries = 0;
  errno = 0;
  while (const dirent* entry = readdir(directory)) {
    const std::string_view name(entry->d_name);
    if (name != "." && name != "..") ++entries;
  }
  const int saved_errno = errno;
  closedir(directory);
  if (saved_errno != 0 || entries != 0) {
    throw Error("immutable-root directory is not empty: " + path);
  }
}

std::string require_root_regular(const std::string& path, bool executable,
                                 std::string_view expected_digest) {
  struct stat info {};
  if (lstat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_uid != 0 || info.st_gid != 0 || info.st_nlink != 1 ||
      (info.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0 ||
      (executable && (info.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH)) == 0)) {
    throw Error("immutable-root file metadata drifted: " + path);
  }
  const std::string contents = read_regular_file(path);
  const std::string digest = sha256(contents);
  if (!expected_digest.empty() && digest != expected_digest) {
    throw Error("immutable-root file hash drifted: " + path);
  }
  return digest;
}

void require_exact_entries(const std::string& path,
                           const std::vector<std::string_view>& expected) {
  DIR* directory = opendir(path.c_str());
  if (directory == nullptr) throw Error("cannot enumerate immutable root: " + path);
  std::vector<std::string> actual;
  errno = 0;
  while (const dirent* entry = readdir(directory)) {
    const std::string_view name(entry->d_name);
    if (name != "." && name != "..") actual.emplace_back(name);
  }
  const int saved_errno = errno;
  closedir(directory);
  if (saved_errno != 0) throw Error("immutable-root enumeration failed");
  std::vector<std::string> wanted;
  for (const std::string_view item : expected) wanted.emplace_back(item);
  std::sort(actual.begin(), actual.end());
  std::sort(wanted.begin(), wanted.end());
  if (actual != wanted) {
    std::string names;
    for (const std::string& name : actual) {
      if (!names.empty()) names.push_back('+');
      names += name;
    }
    throw Error("immutable-root entries drifted: " + path + " actual=" + names);
  }
}

std::string require_immutable_root(const std::string& policy_manifest_path) {
  struct statvfs filesystem {};
  if (statvfs("/", &filesystem) != 0 ||
      (filesystem.f_flag & ST_RDONLY) == 0) {
    throw Error("immutable root is not mounted read-only");
  }
  require_directory("/", false);
  require_directory("/loom", false);
  require_directory("/dev", false);
  require_directory("/proc", true);
  require_directory("/tmp", true);
#if LOOM_EFFECT_POLICY_VERSION == 4
  require_directory("/run", false);
  require_directory("/run/systemd", false);
  require_directory("/run/systemd/incoming", true);
  require_exact_entries("/", {"loom", "dev", "proc", "run", "tmp"});
  require_exact_entries("/run", {"systemd"});
  require_exact_entries("/run/systemd", {"incoming"});
#else
  require_exact_entries("/", {"loom", "dev", "proc", "tmp"});
#endif
  require_exact_entries(
      "/loom",
      {"effect-cell", "payload", "payload.freeze.v1", kPolicyFilename});
  require_exact_entries("/dev", {"null"});

  struct stat null_info {};
  if (lstat("/dev/null", &null_info) != 0 || !S_ISCHR(null_info.st_mode) ||
      major(null_info.st_rdev) != 1 || minor(null_info.st_rdev) != 3) {
    throw Error("immutable root lacks exact /dev/null");
  }
  const std::string cell_digest =
      require_root_regular("/loom/effect-cell", true, {});
  require_root_regular("/loom/payload", true, kPayloadSha256);
  require_root_regular("/loom/payload.freeze.v1", false,
                       kPayloadManifestSha256);
  require_root_regular(std::string(kPolicyRootPath), false,
                       kPolicyManifestSha256);
  if (policy_manifest_path != kPolicyRootPath) {
    throw Error("root-hold policy path escaped the frozen root schema");
  }
  load_policy_manifest(policy_manifest_path);
  return cell_digest;
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

void emit_allow_one_argument(FilterBuilder& builder,
                             std::uint32_t syscall_number,
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
    case 2:
#ifdef SYS_clone3
      return SYS_clone3;
#else
      return SYS_clone;
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
    case 10: return SYS_personality;
    case 11: return SYS_openat;
    case 12: return SYS_getpid;
    default: throw Error("family has no additional sabotage syscall");
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
  builder.statement(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch));
  const std::size_t architecture = builder.equal(AUDIT_ARCH_X86_64);
  builder.statement(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS);
  const std::size_t architecture_ok = builder.code.size();
  builder.patch_true(architecture, architecture_ok);
  builder.statement(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr));

  if (sabotage_family >= 2) emit_sabotage_allow(builder, sabotage_family);
  emit_allow_one_argument(builder, SYS_read, offsetof(seccomp_data, args[0]),
                          STDIN_FILENO);
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

bool install_seccomp(const std::vector<sock_filter>& filter) {
  if (filter.empty() ||
      filter.size() > std::numeric_limits<unsigned short>::max()) {
    return false;
  }
  sock_fprog program{static_cast<unsigned short>(filter.size()),
                    const_cast<sock_filter*>(filter.data())};
  return prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0 &&
         prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0;
}

long probe_family(int family) {
  errno = 0;
  switch (family) {
    case 2:
#ifdef SYS_clone3
      return syscall(SYS_clone3, nullptr, 0);
#else
      return syscall(SYS_clone, 0, nullptr, nullptr, nullptr, nullptr);
#endif
    case 3:
      return syscall(SYS_openat, AT_FDCWD, "/tmp/loom-effect-policy-v3-probe",
                     O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    case 4: return syscall(SYS_dup3, STDIN_FILENO, 9, O_CLOEXEC);
    case 5:
      return syscall(SYS_mmap, nullptr, 4096, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    case 6: {
      io_uring_params parameters {};
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
      return syscall(SYS_memfd_create, "loom-effect-v3", MFD_CLOEXEC);
#else
      return syscall(319, "loom-effect-v3", MFD_CLOEXEC);
#endif
    case 10: return syscall(SYS_personality, ~0UL);
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

int treatment_child(int family, const std::vector<sock_filter>& filter) {
  if (!install_seccomp(filter)) _exit(90);
  const long result = probe_family(family);
  const int saved_errno = errno;
  syscall(SYS_exit, result == -1 && saved_errno == EPERM ? 0 : 91);
  __builtin_unreachable();
}

bool run_treatment(int family, const std::vector<sock_filter>& filter) {
  const pid_t pid = fork();
  if (pid < 0) throw Error("cannot fork treatment cell");
  if (pid == 0) treatment_child(family, filter);
  return wait_success(pid);
}

bool run_allowed_io(const std::vector<sock_filter>& filter) {
  const pid_t pid = fork();
  if (pid < 0) throw Error("cannot fork allowlist cell");
  if (pid == 0) {
    if (!install_seccomp(filter)) _exit(90);
    char byte = 0;
    const long read_result = syscall(SYS_read, STDIN_FILENO, &byte, 0);
    const long write_result = syscall(SYS_write, STDOUT_FILENO, &byte, 0);
    syscall(SYS_exit, read_result == 0 && write_result == 0 ? 0 : 91);
    __builtin_unreachable();
  }
  return wait_success(pid);
}

void close_ambient_descriptors() {
#ifdef SYS_close_range
  if (syscall(SYS_close_range, 3U, ~0U, 0U) == 0) return;
  if (errno != ENOSYS) throw Error("cannot close ambient descriptors");
#endif
  const long maximum = sysconf(_SC_OPEN_MAX);
  if (maximum <= 3 || maximum > 1'048'576) {
    throw Error("ambient descriptor bound is invalid");
  }
  for (int descriptor = 3; descriptor < maximum; ++descriptor) {
    close(descriptor);
  }
}

[[noreturn]] void root_hold(const std::string& policy_manifest_path) {
  const std::string cell_digest =
      require_immutable_root(policy_manifest_path);
  const std::vector<sock_filter> filter = compile_filter(0);
  const std::string filter_sha256 = filter_digest(filter);
  const std::string line = std::string(kReadyPrefix) +
      " semantic_authority=Sounio action=9025 role=MATERIAL_PARITY"
      " object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE root_read_only=true"
      " root_exact=true dynamic_linker_visible=false host_root_visible=false"
      " proc_treatment=absent tmp_read_only=true fd_inventory=0+1+2"
      " cell_sha256=" +
      cell_digest + " payload_sha256=" + std::string(kPayloadSha256) +
      " policy_manifest_sha256=" + std::string(kPolicyManifestSha256) +
      " filter_sha256=" + filter_sha256 +
      " material_coverage=false complete_effects=false"
      " material_execution=false launch_open=false parity_open=false"
      " claim_ready=false\n";
  close_ambient_descriptors();
  if (!install_seccomp(filter)) {
    throw Error("cannot install immutable-root seccomp treatment");
  }
  std::size_t offset = 0;
  while (offset < line.size()) {
    const long count = syscall(SYS_write, STDOUT_FILENO, line.data() + offset,
                               line.size() - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      syscall(SYS_exit, 92);
    }
  }
  char release = 0;
  long count = -1;
  do {
    count = syscall(SYS_read, STDIN_FILENO, &release, 1);
  } while (count < 0 && errno == EINTR);
  syscall(SYS_exit, (count == 0 || (count == 1 && release == 'X')) ? 0 : 93);
  __builtin_unreachable();
}

std::string sabotage_rule(int family) {
  switch (family) {
    case 1: return "fd3_cloexec=false";
    case 2: return "allow_clone3";
    case 3: return "writable_tmp+allow_openat_create";
    case 4: return "allow_dup3";
    case 5: return "allow_shared_writable_mmap";
    case 6: return "allow_io_uring_setup";
    case 7: return "allow_socket_inet";
    case 8: return "allow_socket_unix";
    case 9: return "allow_memfd_create";
    case 10: return "unlock+allow_personality";
    case 11: return "minimal_proc+allow_openat_proc_self_mem";
    case 12: return "allow_unlisted_getpid";
    default: throw Error("invalid sabotage rule family");
  }
}

int selftest(const std::string& policy_manifest_path) {
  const std::string manifest_digest =
      load_policy_manifest(policy_manifest_path);
  const std::vector<sock_filter> treatment_filter = compile_filter(0);
  const std::string treatment_digest = filter_digest(treatment_filter);
  if (!run_allowed_io(treatment_filter)) {
    throw Error("kernel cannot realize frozen four-syscall surface");
  }

  int seccomp_treatments = 0;
  for (int family = 2; family <= 12; ++family) {
    if (!run_treatment(family, treatment_filter)) {
      throw Error("kernel treatment did not refuse family " +
                  std::to_string(family));
    }
    ++seccomp_treatments;
  }

  std::string sabotage_material;
  int structural_sabotages = 0;
  for (int family = 1; family <= 12; ++family) {
    const auto filter = compile_filter(family);
    const std::string digest = filter_digest(filter);
    if (family > 1 && digest == treatment_digest) {
      throw Error("sabotage filter equals treatment filter");
    }
    sabotage_material += std::to_string(family) + ":" + digest + ":" +
                         sabotage_rule(family) + "\n";
    ++structural_sabotages;
  }
  const std::string sabotage_set_digest = sha256(sabotage_material);
  std::cout
      << kSelftestPrefix
      << " semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY"
      << " transitory=true action=9025 policy_v"
      << LOOM_EFFECT_POLICY_VERSION << "_bound=true static=true"
      << " policy_manifest_sha256=" << manifest_digest
      << " object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE"
      << " landlock_required=false family10=personality_change"
      << " seccomp_default=EPERM architecture=AUDIT_ARCH_X86_64"
      << " allowed_syscalls=0+1+60+322 allowed_io=true"
      << " seccomp_treatments=" << seccomp_treatments
      << " structural_root_treatments=1 structural_sabotages="
      << structural_sabotages << " material_sabotages=0"
      << " filter_sha256=" << treatment_digest
      << " sabotage_set_sha256=" << sabotage_set_digest
      << " root_gate_required=true host_gate_required=true"
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
    if (argc == 4 && std::string_view(argv[1]) == "--root-hold" &&
        std::string_view(argv[2]) == "--policy-manifest") {
      root_hold(argv[3]);
    }
    std::cerr << "usage: loom-process-witness-effect-policy-v"
              << LOOM_EFFECT_POLICY_VERSION << " --selftest "
                 "--policy-manifest <frozen-v3-manifest>\n"
                 "       loom-process-witness-effect-policy --root-hold "
                 "--policy-manifest " << kPolicyRootPath << "\n";
    return 64;
  } catch (const std::exception& error) {
    std::cerr << "LOOM_PROCESS_WITNESS_EFFECT_POLICY_V"
              << LOOM_EFFECT_POLICY_VERSION << "_CLOSED reason="
              << error.what() << '\n';
    return 70;
  }
}
