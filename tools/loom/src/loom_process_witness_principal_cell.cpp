#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <openssl/sha.h>

#include <sys/prctl.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

constexpr std::string_view kFrozenPayloadManifestSha256 =
    "624ccd7297778803eff8d9972a33d5e55fb022f9e7e37f444f0aee13c22fb4da";
constexpr std::string_view kFrozenPayloadSha256 =
    "7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d";
constexpr std::string_view kZeroDigest =
    "0000000000000000000000000000000000000000000000000000000000000000";
constexpr auto kDeadline = std::chrono::seconds(5);
constexpr std::size_t kMaximumArtifactBytes = 16 * 1024 * 1024;

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
  int release() {
    const int descriptor = value;
    value = -1;
    return descriptor;
  }
  void reset() {
    if (value >= 0) close(value);
    value = -1;
  }
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

struct ChildProcess {
  pid_t pid = -1;
  UniqueFd input;
  UniqueFd output;
};

struct PositiveResult {
  bool same_pid = false;
  bool start_tick = false;
  bool pidfd = false;
  bool pre_exec_cell = false;
  bool post_exec_sounio = false;
  bool ready = false;
  bool done = false;
};

std::optional<std::uint64_t> parse_u64(std::string_view text) {
  if (text.empty()) return std::nullopt;
  std::uint64_t value = 0;
  for (const unsigned char character : text) {
    if (!std::isdigit(character)) return std::nullopt;
    const std::uint64_t digit = character - '0';
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
      return std::nullopt;
    }
    value = value * 10 + digit;
  }
  return value;
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

std::string read_regular_fd(int descriptor, std::string_view label) {
  if (lseek(descriptor, 0, SEEK_SET) < 0) {
    throw Error("cannot rewind " + std::string(label));
  }
  std::string output;
  std::array<char, 8192> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor, buffer.data(), buffer.size());
    if (count > 0) {
      output.append(buffer.data(), static_cast<std::size_t>(count));
      if (output.size() > kMaximumArtifactBytes) {
        throw Error(std::string(label) + " exceeds size limit");
      }
    } else if (count == 0) {
      if (lseek(descriptor, 0, SEEK_SET) < 0) {
        throw Error("cannot reset " + std::string(label));
      }
      return output;
    } else if (errno != EINTR) {
      throw Error("cannot read " + std::string(label) + ": " +
                  std::strerror(errno));
    }
  }
}

std::string read_bounded_path(const std::string& path, std::size_t limit) {
  UniqueFd descriptor(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (descriptor.get() < 0) throw Error("cannot read process posture");
  std::string output;
  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor.get(), buffer.data(), buffer.size());
    if (count > 0) {
      if (output.size() + static_cast<std::size_t>(count) > limit) {
        throw Error("process posture exceeded bound");
      }
      output.append(buffer.data(), static_cast<std::size_t>(count));
    } else if (count == 0) {
      return output;
    } else if (errno != EINTR) {
      throw Error("process posture read failed");
    }
  }
}

UniqueFd open_regular(const std::string& path, std::string_view label) {
  UniqueFd descriptor(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (descriptor.get() < 0) {
    throw Error("cannot open " + std::string(label) + ": " + std::strerror(errno));
  }
  struct stat info {};
  if (fstat(descriptor.get(), &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_nlink != 1) {
    throw Error(std::string(label) + " is not one regular file");
  }
  return descriptor;
}

Manifest load_payload_manifest(const std::string& path) {
  UniqueFd descriptor = open_regular(path, "payload manifest");
  const std::string contents = read_regular_fd(descriptor.get(), "payload manifest");
  Manifest manifest;
  manifest.digest = sha256(contents);
  if (manifest.digest != kFrozenPayloadManifestSha256) {
    throw Error("frozen payload manifest hash mismatch");
  }
  std::istringstream input(contents);
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos || equals == 0) {
      throw Error("malformed payload manifest line");
    }
    const auto [_, inserted] =
        manifest.fields.emplace(line.substr(0, equals), line.substr(equals + 1));
    if (!inserted) throw Error("duplicate payload manifest field");
  }
  if (manifest.require("schema") !=
          "loom-process-witness-handshake-payload-freeze-v1" ||
      manifest.require("stage") != "SOUNIO_HANDSHAKE_PAYLOAD_FROZEN" ||
      manifest.require("producing_language") != "Sounio" ||
      manifest.require("language_role") != "SEMANTIC_PAYLOAD" ||
      manifest.require("semantic_authority") != "Sounio" ||
      manifest.require("action") != "9030" ||
      manifest.require("executable_sha256") != kFrozenPayloadSha256 ||
      manifest.require("two_phase") != "true" ||
      manifest.require("ready_before_close") != "true" ||
      manifest.require("material_grant") != "true" ||
      manifest.require("material_execution") != "false" ||
      manifest.require("launch_open") != "false" ||
      manifest.require("claim_ready") != "false") {
    throw Error("manifest is not the frozen Sounio ProcessWitness handshake");
  }
  return manifest;
}

UniqueFd open_payload(const std::string& path, std::string_view expected_hash) {
  UniqueFd descriptor = open_regular(path, "Sounio payload");
  struct stat info {};
  if (fstat(descriptor.get(), &info) != 0) {
    throw Error("cannot stat Sounio payload");
  }
  if ((info.st_mode & (S_IWGRP | S_IWOTH)) != 0 ||
      (info.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH)) == 0 ||
      (info.st_uid != 0 && info.st_uid != geteuid())) {
    throw Error("Sounio payload ownership or mode is unsafe");
  }
  const std::string contents = read_regular_fd(descriptor.get(), "Sounio payload");
  if (sha256(contents) != expected_hash) {
    throw Error("Sounio payload hash mismatch");
  }
  return descriptor;
}

std::string canonical_path(const std::string& path) {
  std::array<char, PATH_MAX> buffer{};
  if (realpath(path.c_str(), buffer.data()) == nullptr) {
    throw Error("cannot canonicalize path: " + path);
  }
  return buffer.data();
}

std::string required_environment(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    throw Error(std::string("missing host environment: ") + name);
  }
  return value;
}

bool safe_unit_name(std::string_view unit) {
  if (unit.size() < 9 || unit.size() > 160 ||
      unit.substr(unit.size() - 8) != ".service") {
    return false;
  }
  for (const unsigned char character : unit) {
    if (!std::isalnum(character) && character != '.' && character != '_' &&
        character != '-' && character != '@') {
      return false;
    }
  }
  return true;
}

bool valid_generation(std::string_view generation) {
  if (generation.size() != 64) return false;
  for (const unsigned char character : generation) {
    if (!std::isdigit(character) && (character < 'a' || character > 'f')) {
      return false;
    }
  }
  return true;
}

std::string status_field(const std::string& status, std::string_view name) {
  std::istringstream input(status);
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind(name, 0) == 0) return line.substr(name.size());
  }
  throw Error("process posture field is absent");
}

std::uint64_t parse_hex_u64(std::string_view value) {
  if (value.empty() || value.size() > 16) throw Error("capability field invalid");
  std::uint64_t result = 0;
  for (const unsigned char character : value) {
    std::uint64_t digit = 0;
    if (character >= '0' && character <= '9') {
      digit = character - '0';
    } else if (character >= 'a' && character <= 'f') {
      digit = character - 'a' + 10;
    } else if (character >= 'A' && character <= 'F') {
      digit = character - 'A' + 10;
    } else {
      throw Error("capability field invalid");
    }
    result = result * 16 + digit;
  }
  return result;
}

void require_host_artifact(const std::string& path, bool executable) {
  struct stat info {};
  if (lstat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_uid != 0 || info.st_gid != 0 || info.st_nlink != 1 ||
      (info.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0 ||
      (executable && (info.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH)) == 0)) {
    throw Error("host artifact metadata is unsafe");
  }
}

void require_host_posture(const std::string& unit) {
  if (getppid() != 1) throw Error("host cell parent is not PID 1");
  uid_t real_uid = 0;
  uid_t effective_uid = 0;
  uid_t saved_uid = 0;
  gid_t real_gid = 0;
  gid_t effective_gid = 0;
  gid_t saved_gid = 0;
  if (getresuid(&real_uid, &effective_uid, &saved_uid) != 0 ||
      getresgid(&real_gid, &effective_gid, &saved_gid) != 0 || real_uid == 0 ||
      real_gid == 0 || real_uid != effective_uid || real_uid != saved_uid ||
      real_gid != effective_gid || real_gid != saved_gid) {
    throw Error("host cell requires one non-root credential vector");
  }
  if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0 ||
      prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0 ||
      prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1) {
    throw Error("host cell anti-injection posture is incomplete");
  }
  struct stat input {};
  struct stat output {};
  if (fstat(STDIN_FILENO, &input) != 0 || fstat(STDOUT_FILENO, &output) != 0 ||
      !S_ISFIFO(input.st_mode) || !S_ISFIFO(output.st_mode) ||
      (input.st_dev == output.st_dev && input.st_ino == output.st_ino)) {
    throw Error("host cell requires two distinct anonymous pipes");
  }
  const std::string status = read_bounded_path("/proc/self/status", 256 * 1024);
  const std::string cgroup = read_bounded_path("/proc/self/cgroup", 256 * 1024);
  const auto no_new_privileges = parse_u64(status_field(status, "NoNewPrivs:\t"));
  if (!no_new_privileges || *no_new_privileges != 1 ||
      parse_hex_u64(status_field(status, "CapEff:\t")) != 0 ||
      parse_hex_u64(status_field(status, "CapAmb:\t")) != 0 ||
      cgroup.find(unit) == std::string::npos) {
    throw Error("host cell kernel posture is incomplete");
  }
}

std::string process_executable(pid_t pid) {
  std::array<char, PATH_MAX> buffer{};
  const std::string path = "/proc/" + std::to_string(pid) + "/exe";
  const ssize_t count = readlink(path.c_str(), buffer.data(), buffer.size() - 1);
  if (count <= 0) throw Error("process executable is unavailable");
  return std::string(buffer.data(), static_cast<std::size_t>(count));
}

std::uint64_t process_start_tick(pid_t pid) {
  UniqueFd descriptor =
      open_regular("/proc/" + std::to_string(pid) + "/stat", "process stat");
  const std::string record = read_regular_fd(descriptor.get(), "process stat");
  const std::size_t close = record.rfind(')');
  if (close == std::string::npos || close + 2 >= record.size()) {
    throw Error("process stat is malformed");
  }
  std::istringstream input(record.substr(close + 2));
  std::string field;
  for (int index = 0; index <= 19; ++index) {
    if (!(input >> field)) throw Error("process start tick is missing");
  }
  const auto value = parse_u64(field);
  if (!value || *value == 0) throw Error("process start tick is invalid");
  return *value;
}

UniqueFd open_pidfd(pid_t pid) {
#ifdef SYS_pidfd_open
  const int descriptor = static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
  if (descriptor < 0) throw Error("pidfd_open failed: " + std::string(std::strerror(errno)));
  return UniqueFd(descriptor);
#else
  (void)pid;
  throw Error("pidfd_open is unavailable");
#endif
}

bool pidfd_is_live(int descriptor) {
  pollfd candidate{descriptor, POLLIN, 0};
  const int status = poll(&candidate, 1, 0);
  return status == 0 && candidate.revents == 0;
}

void write_all(int descriptor, std::string_view value) {
  std::size_t offset = 0;
  while (offset < value.size()) {
    const ssize_t count =
        write(descriptor, value.data() + offset, value.size() - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      throw Error("pipe write failed: " + std::string(std::strerror(errno)));
    }
  }
}

int remaining_milliseconds(std::chrono::steady_clock::time_point deadline) {
  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) return 0;
  const auto remaining =
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
  return static_cast<int>(remaining.count()) + 1;
}

std::string read_line(int descriptor) {
  const auto deadline = std::chrono::steady_clock::now() + kDeadline;
  std::string output;
  while (output.size() <= 512) {
    pollfd candidate{descriptor, static_cast<short>(POLLIN | POLLHUP), 0};
    const int status = poll(&candidate, 1, remaining_milliseconds(deadline));
    if (status < 0 && errno == EINTR) continue;
    if (status <= 0) throw Error("timed out reading child line");
    char character = 0;
    const ssize_t count = read(descriptor, &character, 1);
    if (count == 1) {
      output.push_back(character);
      if (character == '\n') return output;
    } else if (count == 0) {
      throw Error("child output closed before newline");
    } else if (errno != EINTR) {
      throw Error("child output read failed");
    }
  }
  throw Error("child line exceeds bound");
}

void require_output_eof(int descriptor) {
  const auto deadline = std::chrono::steady_clock::now() + kDeadline;
  for (;;) {
    pollfd candidate{descriptor, static_cast<short>(POLLIN | POLLHUP), 0};
    const int status = poll(&candidate, 1, remaining_milliseconds(deadline));
    if (status < 0 && errno == EINTR) continue;
    if (status <= 0) throw Error("timed out waiting for child output EOF");
    char byte = 0;
    const ssize_t count = read(descriptor, &byte, 1);
    if (count == 0) return;
    if (count == 1) throw Error("child emitted unexpected trailing output");
    if (errno != EINTR) throw Error("child EOF read failed");
  }
}

int wait_child(pid_t pid) {
  const auto deadline = std::chrono::steady_clock::now() + kDeadline;
  for (;;) {
    int status = 0;
    const pid_t result = waitpid(pid, &status, WNOHANG);
    if (result == pid) {
      if (WIFEXITED(status)) return WEXITSTATUS(status);
      return 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    }
    if (result < 0 && errno != EINTR) throw Error("waitpid failed");
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
      }
      throw Error("child did not exit before deadline");
    }
    poll(nullptr, 0, 5);
  }
}

std::string random_generation() {
  std::array<unsigned char, 32> bytes{};
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count = getrandom(bytes.data() + offset, bytes.size() - offset, 0);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      throw Error("generation randomness failed");
    }
  }
  return sha256(std::string_view(reinterpret_cast<const char*>(bytes.data()),
                                 bytes.size()));
}

std::string release_frame(std::string_view generation) {
  return "ARM\n" + std::string(generation) + "\n";
}

bool read_release_frame(int descriptor, std::string_view expected) {
  const auto deadline = std::chrono::steady_clock::now() + kDeadline;
  std::string frame;
  frame.reserve(expected.size());
  while (frame.size() < expected.size()) {
    pollfd candidate{descriptor, static_cast<short>(POLLIN | POLLHUP), 0};
    const int status = poll(&candidate, 1, remaining_milliseconds(deadline));
    if (status < 0 && errno == EINTR) continue;
    if (status <= 0) return false;
    char byte = 0;
    const ssize_t count = read(descriptor, &byte, 1);
    if (count == 1) {
      frame.push_back(byte);
    } else if (count == 0) {
      return false;
    } else if (errno != EINTR) {
      return false;
    }
  }
  if (frame != expected) return false;
  pollfd candidate{descriptor, static_cast<short>(POLLIN | POLLHUP), 0};
  const int status = poll(&candidate, 1, 0);
  return status == 0 && candidate.revents == 0;
}

[[noreturn]] void exec_payload_fd(int descriptor) {
  char program[] = "loom-process-witness-sounio";
  char* const arguments[] = {program, nullptr};
  char* const environment[] = {nullptr};
#ifdef SYS_execveat
  syscall(SYS_execveat, descriptor, "", arguments, environment, AT_EMPTY_PATH);
  throw Error("execveat failed: " + std::string(std::strerror(errno)));
#else
  (void)descriptor;
  throw Error("execveat is unavailable");
#endif
}

UniqueFd isolate_payload_descriptor(UniqueFd descriptor) {
  if (descriptor.get() != 3) {
    if (dup3(descriptor.get(), 3, O_CLOEXEC) < 0) {
      throw Error("cannot isolate payload descriptor");
    }
    descriptor.reset();
    descriptor = UniqueFd(3);
  }
#ifdef SYS_close_range
  if (syscall(SYS_close_range, 4U, ~0U, 0U) != 0) {
    if (errno != ENOSYS) throw Error("cannot close ambient descriptors");
    const long maximum = sysconf(_SC_OPEN_MAX);
    for (int candidate = 4; candidate < maximum; ++candidate) close(candidate);
  }
#else
  const long maximum = sysconf(_SC_OPEN_MAX);
  for (int candidate = 4; candidate < maximum; ++candidate) close(candidate);
#endif
  return descriptor;
}

[[noreturn]] void refuse(std::string_view reason) {
  const std::string line = "LOOM_PROCESS_WITNESS_CLOSED reason=" +
                           std::string(reason) + "\n";
  write_all(STDOUT_FILENO, line);
  _exit(70);
}

[[noreturn]] void guarded_child(const std::string& payload_path,
                                std::string_view expected_payload_hash,
                                std::string_view generation) {
  try {
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() == 1) {
      refuse("parent-identity");
    }
    UniqueFd payload = open_payload(payload_path, expected_payload_hash);
    if (!read_release_frame(STDIN_FILENO, release_frame(generation))) {
      refuse("release-frame");
    }
    payload = isolate_payload_descriptor(std::move(payload));
    exec_payload_fd(payload.get());
  } catch (const std::exception&) {
    refuse("payload-binding");
  }
}

[[noreturn]] void bypass_child(const std::string& payload_path,
                               std::string_view expected_payload_hash) {
  try {
    UniqueFd payload = open_payload(payload_path, expected_payload_hash);
    payload = isolate_payload_descriptor(std::move(payload));
    exec_payload_fd(payload.get());
  } catch (const std::exception&) {
    _exit(71);
  }
}

[[noreturn]] void host_internal(const std::string& payload_path,
                                const std::string& manifest_path, bool bypass) {
  try {
    if (required_environment("SOUNIO_LOOM_PROCESS_WITNESS_INTERNAL") != "1") {
      refuse("internal-marker");
    }
    const std::string unit =
        required_environment("SOUNIO_LOOM_PROCESS_WITNESS_UNIT");
    const std::string generation =
        required_environment("SOUNIO_LOOM_PROCESS_WITNESS_GENERATION");
    if (!safe_unit_name(unit) || !valid_generation(generation)) {
      refuse("host-identity");
    }
    if (bypass &&
        required_environment("SOUNIO_LOOM_PROCESS_WITNESS_SABOTAGE") !=
            "exact-release-bypass") {
      refuse("sabotage-marker");
    }
    require_host_artifact(payload_path, true);
    require_host_artifact(manifest_path, false);
    const Manifest manifest = load_payload_manifest(manifest_path);
    require_host_posture(unit);
    UniqueFd payload =
        open_payload(payload_path, manifest.require("executable_sha256"));
    if (!bypass &&
        !read_release_frame(STDIN_FILENO, release_frame(generation))) {
      refuse("release-frame");
    }
    payload = isolate_payload_descriptor(std::move(payload));
    exec_payload_fd(payload.get());
  } catch (const std::exception&) {
    refuse("host-posture");
  }
}

ChildProcess spawn_child(const std::string& payload_path,
                         std::string_view expected_payload_hash,
                         std::string_view generation, bool bypass) {
  int input_pipe[2] = {-1, -1};
  int output_pipe[2] = {-1, -1};
  if (pipe2(input_pipe, O_CLOEXEC) != 0 || pipe2(output_pipe, O_CLOEXEC) != 0) {
    throw Error("cannot create child pipes");
  }
  UniqueFd child_input(input_pipe[0]);
  UniqueFd parent_input(input_pipe[1]);
  UniqueFd parent_output(output_pipe[0]);
  UniqueFd child_output(output_pipe[1]);
  const pid_t pid = fork();
  if (pid < 0) throw Error("fork failed");
  if (pid == 0) {
    parent_input.reset();
    parent_output.reset();
    if (dup2(child_input.get(), STDIN_FILENO) < 0 ||
        dup2(child_output.get(), STDOUT_FILENO) < 0) {
      _exit(72);
    }
    child_input.reset();
    child_output.reset();
    if (bypass) {
      bypass_child(payload_path, expected_payload_hash);
    }
    guarded_child(payload_path, expected_payload_hash, generation);
  }
  child_input.reset();
  child_output.reset();
  return ChildProcess{pid, std::move(parent_input), std::move(parent_output)};
}

PositiveResult run_positive(const std::string& cell_path,
                            const std::string& payload_path,
                            const Manifest& manifest, bool bypass) {
  const std::string generation = random_generation();
  ChildProcess child =
      spawn_child(payload_path, manifest.require("executable_sha256"), generation,
                  bypass);
  UniqueFd pidfd = open_pidfd(child.pid);
  const std::uint64_t before_tick = process_start_tick(child.pid);
  const std::string before_executable = process_executable(child.pid);
  if (!bypass) write_all(child.input.get(), release_frame(generation));
  const std::string ready = read_line(child.output.get());
  const std::uint64_t after_tick = process_start_tick(child.pid);
  const std::string after_executable = process_executable(child.pid);
  PositiveResult result;
  result.same_pid = child.pid > 1;
  result.start_tick = before_tick == after_tick;
  result.pidfd = pidfd_is_live(pidfd.get());
  result.pre_exec_cell = before_executable == cell_path;
  result.post_exec_sounio = after_executable == payload_path;
  result.ready = ready == manifest.require("ready_line") + "\n";
  write_all(child.input.get(), "CLOSE\n");
  child.input.reset();
  const std::string done = read_line(child.output.get());
  result.done = done == manifest.require("done_line") + "\n";
  require_output_eof(child.output.get());
  if (wait_child(child.pid) != 0) throw Error("positive child status was not zero");
  return result;
}

bool complete(const PositiveResult& result) {
  return result.same_pid && result.start_tick && result.pidfd &&
         result.pre_exec_cell && result.post_exec_sounio && result.ready &&
         result.done;
}

bool bypass_complete(const PositiveResult& result) {
  return result.same_pid && result.start_tick && result.pidfd &&
         result.post_exec_sounio && result.ready && result.done;
}

bool run_refusal(const std::string& payload_path, const Manifest& manifest,
                 std::string_view mode) {
  const std::string generation = random_generation();
  const std::string hash =
      mode == "payload-hash" ? std::string(kZeroDigest)
                             : manifest.require("executable_sha256");
  ChildProcess child = spawn_child(payload_path, hash, generation, false);
  if (mode == "treatment") {
    child.input.reset();
  } else if (mode == "wrong-generation") {
    std::string wrong(64, '0');
    if (wrong == generation) wrong.assign(64, '1');
    write_all(child.input.get(), release_frame(wrong));
  } else if (mode == "extra-release") {
    write_all(child.input.get(), release_frame(generation) + "X");
  } else if (mode == "payload-hash") {
    write_all(child.input.get(), release_frame(generation));
  } else {
    throw Error("unknown refusal mode");
  }
  const std::string output = read_line(child.output.get());
  child.input.reset();
  require_output_eof(child.output.get());
  return output.rfind("LOOM_PROCESS_WITNESS_CLOSED reason=", 0) == 0 &&
         wait_child(child.pid) == 70;
}

int selftest(const std::string& invoked_path, const std::string& payload_path,
             const std::string& manifest_path) {
  const Manifest manifest = load_payload_manifest(manifest_path);
  const std::string cell = canonical_path(invoked_path);
  const std::string payload = canonical_path(payload_path);
  const PositiveResult guarded = run_positive(cell, payload, manifest, false);
  const bool treatment = run_refusal(payload, manifest, "treatment");
  const bool wrong_generation =
      run_refusal(payload, manifest, "wrong-generation");
  const bool extra_release = run_refusal(payload, manifest, "extra-release");
  const bool payload_hash = run_refusal(payload, manifest, "payload-hash");
  const PositiveResult bypass = run_positive(cell, payload, manifest, true);
  if (!complete(guarded) || !treatment || !wrong_generation || !extra_release ||
      !payload_hash || !bypass_complete(bypass)) {
    throw Error("ProcessWitness selftest invariant failed");
  }
  std::cout
      << "LOOM_PROCESS_WITNESS_PRINCIPAL_CELL_SELFTEST PASS"
      << " semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY"
      << " transitory=true action=9030 local_execveat=true same_pid=true"
      << " start_tick=true pidfd=true pre_exec=cell post_exec=Sounio"
      << " treatment=closed positive=done wrong_generation=closed"
      << " extra_release=closed payload_hash_mismatch=closed"
      << " causal_bypass=done causal_sabotage=PASS two_phase=true"
      << " same_descriptor_hash_and_exec=true no_read_ahead=true empty_env=true"
      << " host_internal_mode=bounded dynamic_user_required=true"
      << " host_bypass_sabotage=bounded"
      << " payload_manifest_sha256=" << manifest.digest
      << " payload_sha256=" << manifest.require("executable_sha256")
      << " principal_distinct_uid=false material_grant=true"
      << " material_execution=false host_execveat=false launch_open=false"
      << " recycle_open=false commit_attached=false ci_attached=false"
      << " python_executed=false rust_executed=false\n";
  return 0;
}

void usage(const char* program) {
  std::cerr << "usage: " << program
            << " --selftest --payload PATH --payload-manifest PATH\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 6 &&
        std::string_view(argv[1]) == "--internal-host-process-witness" &&
        std::string_view(argv[2]) == "--payload" &&
        std::string_view(argv[4]) == "--payload-manifest") {
      host_internal(argv[3], argv[5], false);
    }
    if (argc == 6 &&
        std::string_view(argv[1]) ==
            "--internal-host-process-witness-bypass" &&
        std::string_view(argv[2]) == "--payload" &&
        std::string_view(argv[4]) == "--payload-manifest") {
      host_internal(argv[3], argv[5], true);
    }
    bool run_selftest = false;
    std::string payload;
    std::string manifest;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--selftest") {
        run_selftest = true;
      } else if (argument == "--payload" && index + 1 < argc) {
        payload = argv[++index];
      } else if (argument == "--payload-manifest" && index + 1 < argc) {
        manifest = argv[++index];
      } else {
        usage(argv[0]);
        return 64;
      }
    }
    if (!run_selftest || payload.empty() || manifest.empty()) {
      usage(argv[0]);
      return 64;
    }
    return selftest(argv[0], payload, manifest);
  } catch (const std::exception& error) {
    std::cerr << "loom-process-witness-principal-cell: FAIL reason="
              << error.what() << '\n';
    return 1;
  }
}
