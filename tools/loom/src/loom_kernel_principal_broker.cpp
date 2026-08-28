#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <openssl/sha.h>

#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::string_view kFrozenManifestSha256 =
    "7bb5bbf30106d269644b0f9e6d80ee09f43eecf0e4a840bc3f429cfb6eca7cb5";
constexpr std::string_view kFrozenCapsuleManifestSha256 =
    "76ac860306c8cc00517f81f3fe2a4a2742a1cd4b9c4b4bb34b144b25fbcdf26f";
constexpr std::string_view kFrozenInvocationCellManifestSha256 =
    "61918604bf177753c6141f6cd0f05d342a1869ab8fc08d187306a481de33d70e";
constexpr std::string_view kZeroDigest =
    "0000000000000000000000000000000000000000000000000000000000000000";
constexpr std::size_t kMaximumAuthorityOutput = 1024 * 1024;
constexpr std::size_t kMaximumInvocationFrame = 64 * 1024;
constexpr std::size_t kMaximumRequest = kMaximumInvocationFrame + 6;
constexpr std::size_t kMaximumResponse = 4096;
constexpr auto kAuthorityTimeout = std::chrono::seconds(5);

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

struct CommandResult {
  int exit_code = 255;
  std::string output;
};

struct ActivationFacts {
  bool root_identity = false;
  bool pid1_systemd = false;
  bool parent_is_pid1 = false;
  bool service_cgroup = false;
  bool listen_environment = false;
  bool inherited_root_socket = false;
  bool privilege_environment_absent = false;
  bool artifacts_root_owned = false;
  bool policy_only = true;
  bool frozen_policy_bound = false;

  bool supervised() const {
    return pid1_systemd && parent_is_pid1 && service_cgroup && listen_environment;
  }
  bool complete() const {
    return root_identity && supervised() && inherited_root_socket &&
           privilege_environment_absent && artifacts_root_owned && policy_only &&
           frozen_policy_bound;
  }
};

enum class LeaseState : std::uint64_t {
  Free = 0,
  Reserved = 1,
  Mapped = 2,
  Launched = 3,
  Draining = 4,
  Quarantined = 5,
};

struct LeaseRecord {
  std::uint64_t sequence = 0;
  std::uint64_t epoch = 0;
  std::string lease;
  std::uint64_t generation = 0;
  LeaseState state = LeaseState::Free;
  std::uint64_t uid_start = 0;
  std::uint64_t uid_count = 0;
  std::uint64_t gid_start = 0;
  std::uint64_t gid_count = 0;
  std::string previous_digest;
  std::string digest;
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

std::string read_file(const std::string& path) {
  UniqueFd descriptor(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (descriptor.get() < 0) {
    throw Error("cannot open " + path + ": " + std::strerror(errno));
  }
  struct stat info {};
  if (fstat(descriptor.get(), &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_nlink != 1) {
    throw Error("artifact is not one regular file: " + path);
  }
  std::string output;
  std::array<char, 8192> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor.get(), buffer.data(), buffer.size());
    if (count > 0) {
      output.append(buffer.data(), static_cast<std::size_t>(count));
      if (output.size() > 16 * 1024 * 1024) {
        throw Error("artifact exceeds size limit: " + path);
      }
    } else if (count == 0) {
      return output;
    } else if (errno != EINTR) {
      throw Error("cannot read " + path + ": " + std::strerror(errno));
    }
  }
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

std::string file_sha256(const std::string& path) { return sha256(read_file(path)); }

Manifest parse_frozen_manifest(const std::string& path,
                               std::string_view expected_digest,
                               std::string_view label) {
  const std::string contents = read_file(path);
  Manifest manifest;
  manifest.digest = sha256(contents);
  if (manifest.digest != expected_digest) {
    throw Error("frozen " + std::string(label) + " manifest hash mismatch");
  }
  std::istringstream input(contents);
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos || equals == 0) {
      throw Error("malformed manifest line");
    }
    const std::string key = line.substr(0, equals);
    const std::string value = line.substr(equals + 1);
    if (!manifest.fields.emplace(key, value).second) {
      throw Error("duplicate manifest field: " + key);
    }
  }
  return manifest;
}

Manifest load_lease_manifest(const std::string& path) {
  Manifest manifest =
      parse_frozen_manifest(path, kFrozenManifestSha256, "action 9027");
  if (manifest.require("schema") !=
          "loom-kernel-principal-lease-authority-freeze-v1" ||
      manifest.require("stage") != "SEMANTICS_FROZEN" ||
      manifest.require("producing_language") != "Sounio" ||
      manifest.require("language_role") != "SEMANTIC_AUTHORITY" ||
      manifest.require("action") != "9027" ||
      manifest.require("parent_action") != "9026" ||
      manifest.require("material_broker") != "false") {
    throw Error("manifest does not describe frozen Sounio action 9027");
  }
  return manifest;
}

Manifest load_capsule_manifest(const std::string& path) {
  Manifest manifest = parse_frozen_manifest(
      path, kFrozenCapsuleManifestSha256, "action 9028");
  if (manifest.require("schema") !=
          "loom-kernel-principal-capsule-authority-freeze-v1" ||
      manifest.require("stage") != "SEMANTICS_FROZEN" ||
      manifest.require("producing_language") != "Sounio" ||
      manifest.require("language_role") != "SEMANTIC_AUTHORITY" ||
      manifest.require("action") != "9028" ||
      manifest.require("parent_action") != "9027" ||
      manifest.require("grandparent_action") != "9026" ||
      manifest.require("material_capsule") != "false") {
    throw Error("manifest does not describe frozen Sounio action 9028");
  }
  return manifest;
}

Manifest load_invocation_cell_manifest(const std::string& path) {
  Manifest manifest = parse_frozen_manifest(
      path, kFrozenInvocationCellManifestSha256, "action 9029");
  if (manifest.require("schema") !=
          "loom-kernel-invocation-cell-authority-freeze-v1" ||
      manifest.require("stage") != "SEMANTICS_FROZEN" ||
      manifest.require("producing_language") != "Sounio" ||
      manifest.require("language_role") != "SEMANTIC_AUTHORITY" ||
      manifest.require("action") != "9029" ||
      manifest.require("material_invocation") != "false" ||
      manifest.require("same_uid_peer_isolation") != "false" ||
      manifest.require("parity_open") != "false" ||
      manifest.require("claim_ready") != "false") {
    throw Error("manifest does not describe frozen Sounio action 9029");
  }
  return manifest;
}

void verify_authority(const Manifest& manifest, const std::string& authority_path,
                      std::string_view label) {
  if (file_sha256(authority_path) != manifest.require("executable_sha256")) {
    throw Error("Sounio " + std::string(label) + " authority executable hash mismatch");
  }
}

bool root_owned_regular(const std::string& path) {
  struct stat info {};
  return lstat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode) &&
         info.st_uid == 0 && info.st_gid == 0 && info.st_nlink == 1 &&
         (info.st_mode & (S_IWGRP | S_IWOTH)) == 0;
}

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

bool environment_u64_equals(const char* name, std::uint64_t expected) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) return false;
  const auto parsed = parse_u64(raw);
  return parsed && *parsed == expected;
}

std::string parent_path(const std::string& path) {
  const std::size_t slash = path.rfind('/');
  if (slash == std::string::npos) return ".";
  if (slash == 0) return "/";
  return path.substr(0, slash);
}

bool inherited_socket_valid(const std::string& expected_path) {
  constexpr int descriptor = 3;
  int socket_type = 0;
  socklen_t type_size = sizeof(socket_type);
  if (getsockopt(descriptor, SOL_SOCKET, SO_TYPE, &socket_type, &type_size) != 0 ||
      socket_type != SOCK_STREAM) {
    return false;
  }
  sockaddr_un address{};
  socklen_t address_size = sizeof(address);
  if (getsockname(descriptor, reinterpret_cast<sockaddr*>(&address), &address_size) != 0 ||
      address.sun_family != AF_UNIX || address.sun_path[0] == '\0' ||
      expected_path != address.sun_path) {
    return false;
  }
  struct stat descriptor_info {};
  struct stat path_info {};
  struct stat directory_info {};
  if (fstat(descriptor, &descriptor_info) != 0 ||
      lstat(expected_path.c_str(), &path_info) != 0 ||
      stat(parent_path(expected_path).c_str(), &directory_info) != 0) {
    return false;
  }
  return S_ISSOCK(descriptor_info.st_mode) && S_ISSOCK(path_info.st_mode) &&
         descriptor_info.st_ino == path_info.st_ino && path_info.st_uid == 0 &&
         path_info.st_gid == 0 && (path_info.st_mode & 0777) == 0600 &&
         S_ISDIR(directory_info.st_mode) && directory_info.st_uid == 0 &&
         directory_info.st_gid == 0 &&
         (directory_info.st_mode & (S_IWGRP | S_IWOTH)) == 0;
}

ActivationFacts measure_activation(const std::string& manifest_path,
                                   const std::string& authority_path,
                                   const std::string& socket_path,
                                   const std::string& capsule_manifest_path = "",
                                   const std::string& capsule_authority_path = "",
                                   const std::string& invocation_manifest_path = "",
                                   const std::string& invocation_authority_path = "") {
  ActivationFacts facts;
  facts.root_identity = getuid() == 0 && geteuid() == 0 && getgid() == 0 &&
                        getegid() == 0;
  try {
    facts.pid1_systemd = trim(read_file("/proc/1/comm")) == "systemd";
  } catch (...) {
    facts.pid1_systemd = false;
  }
  facts.parent_is_pid1 = getppid() == 1;
  try {
    facts.service_cgroup =
        read_file("/proc/self/cgroup").find("sounio-loom-principal-broker.service") !=
        std::string::npos;
  } catch (...) {
    facts.service_cgroup = false;
  }
  facts.listen_environment = environment_u64_equals("LISTEN_PID", getpid()) &&
                             environment_u64_equals("LISTEN_FDS", 1);
  facts.inherited_root_socket = inherited_socket_valid(socket_path);
  facts.privilege_environment_absent =
      std::getenv("SUDO_UID") == nullptr && std::getenv("SUDO_USER") == nullptr &&
      std::getenv("DOAS_USER") == nullptr && std::getenv("PKEXEC_UID") == nullptr;
  facts.artifacts_root_owned = root_owned_regular(manifest_path) &&
                               root_owned_regular(authority_path) &&
                               (capsule_manifest_path.empty() ||
                                root_owned_regular(capsule_manifest_path)) &&
                               (capsule_authority_path.empty() ||
                                root_owned_regular(capsule_authority_path)) &&
                               (invocation_manifest_path.empty() ||
                                root_owned_regular(invocation_manifest_path)) &&
                               (invocation_authority_path.empty() ||
                                root_owned_regular(invocation_authority_path));
  facts.frozen_policy_bound = true;
  return facts;
}

CommandResult run_authority(const std::string& authority_path,
                            const std::string& frame,
                            std::string_view expected_prefix) {
  int input_pipe[2];
  int output_pipe[2];
  if (pipe2(input_pipe, O_CLOEXEC) != 0 || pipe2(output_pipe, O_CLOEXEC) != 0) {
    throw Error("cannot create authority pipes");
  }
  UniqueFd input_read(input_pipe[0]);
  UniqueFd input_write(input_pipe[1]);
  UniqueFd output_read(output_pipe[0]);
  UniqueFd output_write(output_pipe[1]);
  const pid_t child = fork();
  if (child < 0) throw Error("cannot fork Sounio authority");
  if (child == 0) {
    dup2(input_read.get(), STDIN_FILENO);
    dup2(output_write.get(), STDOUT_FILENO);
    dup2(output_write.get(), STDERR_FILENO);
    char* const arguments[] = {const_cast<char*>(authority_path.c_str()), nullptr};
    execv(authority_path.c_str(), arguments);
    _exit(127);
  }
  input_read = UniqueFd();
  output_write = UniqueFd();
  std::string input = frame + "\n";
  std::size_t written = 0;
  while (written < input.size()) {
    const ssize_t count = write(input_write.get(), input.data() + written,
                                input.size() - written);
    if (count > 0) {
      written += static_cast<std::size_t>(count);
    } else if (count < 0 && errno != EINTR) {
      kill(child, SIGKILL);
      waitpid(child, nullptr, 0);
      throw Error("cannot write Sounio frame");
    }
  }
  input_write = UniqueFd();
  const int flags = fcntl(output_read.get(), F_GETFL, 0);
  if (flags < 0 || fcntl(output_read.get(), F_SETFL, flags | O_NONBLOCK) != 0) {
    kill(child, SIGKILL);
    waitpid(child, nullptr, 0);
    throw Error("cannot configure authority output");
  }
  std::string output;
  std::array<char, 4096> buffer{};
  int status = 0;
  bool exited = false;
  bool output_open = true;
  const auto deadline = std::chrono::steady_clock::now() + kAuthorityTimeout;
  while (!exited || output_open) {
    if (!exited) {
      const pid_t waited = waitpid(child, &status, WNOHANG);
      if (waited == child) exited = true;
      if (waited < 0 && errno != EINTR) throw Error("authority waitpid failed");
    }
    while (output_open) {
      const ssize_t count = read(output_read.get(), buffer.data(), buffer.size());
      if (count > 0) {
        output.append(buffer.data(), static_cast<std::size_t>(count));
        if (output.size() > kMaximumAuthorityOutput) {
          kill(child, SIGKILL);
          waitpid(child, nullptr, 0);
          throw Error("authority output exceeded limit");
        }
      } else if (count == 0) {
        output_open = false;
      } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      } else if (errno != EINTR) {
        throw Error("cannot read authority output");
      }
    }
    if (exited && !output_open) break;
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(child, SIGKILL);
      while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
      }
      throw Error("Sounio authority timeout");
    }
    pollfd descriptor{output_read.get(), POLLIN | POLLHUP, 0};
    poll(&descriptor, 1, 25);
  }
  if (!WIFEXITED(status)) throw Error("Sounio authority terminated abnormally");
  const int exit_code = WEXITSTATUS(status);
  output = trim(output);
  if (output.find('\n') != std::string::npos ||
      output.rfind(expected_prefix, 0) != 0) {
    throw Error("malformed Sounio authority output");
  }
  return {exit_code, output};
}

struct InvocationDecision {
  bool allowed = false;
  std::uint64_t code = 0;
};

InvocationDecision parse_invocation_decision(const CommandResult& result) {
  constexpr std::string_view allow_prefix =
      "SOUNIO_KERNEL_INVOCATION_CELL_ALLOW code=0 ";
  constexpr std::string_view deny_prefix =
      "SOUNIO_KERNEL_INVOCATION_CELL_DENY code=";
  if (result.output.rfind(allow_prefix, 0) == 0) {
    if (result.exit_code != 0) {
      throw Error("Sounio InvocationCell ALLOW exited nonzero");
    }
    return {true, 0};
  }
  if (result.output.rfind(deny_prefix, 0) != 0) {
    throw Error("malformed Sounio InvocationCell decision");
  }
  const std::size_t code_start = deny_prefix.size();
  const std::size_t code_end = result.output.find(' ', code_start);
  if (code_end == std::string::npos) {
    throw Error("Sounio InvocationCell decision omitted reason");
  }
  const auto code = parse_u64(std::string_view(result.output).substr(
      code_start, code_end - code_start));
  if (!code || *code == 0 || result.exit_code != static_cast<int>(*code & 0xff)) {
    throw Error("Sounio InvocationCell decision exit mismatch");
  }
  return {false, *code};
}

std::string validate_invocation_frame(std::string frame) {
  frame = trim(std::move(frame));
  if (frame.empty() || frame.size() > kMaximumInvocationFrame ||
      frame.find('\n') != std::string::npos ||
      frame.find('\r') != std::string::npos) {
    throw Error("InvocationCell frame is empty, multiline, or oversized");
  }
  return frame;
}

std::string load_invocation_frame(const std::string& path) {
  return validate_invocation_frame(read_file(path));
}

struct InvocationAdmission {
  CommandResult result;
  InvocationDecision decision;
  std::string receipt;
};

InvocationAdmission evaluate_invocation_cell(const Manifest& manifest,
                                             const std::string& authority_path,
                                             const std::string& frame) {
  const CommandResult result = run_authority(
      authority_path, frame, "SOUNIO_KERNEL_INVOCATION_CELL_");
  const InvocationDecision decision = parse_invocation_decision(result);
  std::ostringstream receipt;
  receipt << "LOOM_KERNEL_INVOCATION_CELL_MATERIAL_ADMISSION"
          << " schema=loom-kernel-invocation-cell-material-admission-v1"
          << " producing_language=C++20"
          << " language_role=MATERIAL_PARITY"
          << " semantic_authority=Sounio"
          << " action=9029"
          << " manifest_sha256=" << manifest.digest
          << " authority_sha256=" << file_sha256(authority_path)
          << " frame_sha256=" << sha256(frame + "\n")
          << " decision=" << (decision.allowed ? "ALLOW" : "DENY")
          << " decision_code=" << decision.code
          << " decision_sha256=" << sha256(result.output + "\n")
          << " material_invocation=false"
          << " same_uid_peer_isolation=false"
          << " launch_open=false";
  return {result, decision, receipt.str()};
}

std::string current_frame(const ActivationFacts& facts) {
  const int host_owned = facts.root_identity && facts.artifacts_root_owned ? 1 : 0;
  const int supervised = facts.supervised() ? 1 : 0;
  const int root_socket = facts.inherited_root_socket ? 1 : 0;
  const int direct_denied = facts.privilege_environment_absent && facts.supervised() ? 1 : 0;
  const int policy_only = facts.policy_only ? 1 : 0;
  const int frozen = facts.frozen_policy_bound ? 1 : 0;
  const std::string one = "1 1 1 1 1 1 1 1";
  const std::string zero = "0 0 0 0 0 0 0 0";
  std::ostringstream frame;
  frame << "9027 3 1 " << host_owned << ' ' << supervised << ' ' << root_socket
        << ' ' << direct_denied << ' ' << policy_only << ' ' << frozen << ' '
        // Syntactically valid later fields cannot launder an absent host boundary.
        << "1 1 1 1 1 1 1 2 40 41 "
        << "1 0 3 1 1 1 0 0 0 "
        << "0 0 0 0 0 0 "
        << "0 0 0 0 0 0 "
        << "0 0 0 0 0 0 "
        << "0 0 0 0 0 0 0 "
        << "0 0 0 " << one;
  for (int index = 0; index < 10; ++index) frame << ' ' << zero;
  return frame.str();
}

std::vector<std::string> split_words(const std::string& line) {
  std::istringstream input(line);
  std::vector<std::string> words;
  std::string word;
  while (input >> word) words.push_back(word);
  return words;
}

bool valid_lease_name(const std::string& value) {
  if (value.empty() || value.size() > 64) return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return std::isalnum(character) || character == '.' || character == '_' ||
           character == '-';
  });
}

std::string record_body(const LeaseRecord& record) {
  std::ostringstream output;
  output << "v1 " << record.sequence << ' ' << record.epoch << ' ' << record.lease
         << ' ' << record.generation << ' '
         << static_cast<std::uint64_t>(record.state) << ' ' << record.uid_start
         << ' ' << record.uid_count << ' ' << record.gid_start << ' '
         << record.gid_count << ' ' << record.previous_digest;
  return output.str();
}

LeaseRecord parse_record(const std::string& line) {
  const auto words = split_words(line);
  if (words.size() != 12 || words[0] != "v1" || !valid_lease_name(words[3])) {
    throw Error("malformed lease-journal record");
  }
  const auto sequence = parse_u64(words[1]);
  const auto epoch = parse_u64(words[2]);
  const auto generation = parse_u64(words[4]);
  const auto state = parse_u64(words[5]);
  const auto uid_start = parse_u64(words[6]);
  const auto uid_count = parse_u64(words[7]);
  const auto gid_start = parse_u64(words[8]);
  const auto gid_count = parse_u64(words[9]);
  if (!sequence || !epoch || !generation || !state || !uid_start || !uid_count ||
      !gid_start || !gid_count || *sequence == 0 || *epoch == 0 ||
      *generation == 0 || *state > 5 || *uid_count == 0 || *gid_count == 0 ||
      *uid_start > std::numeric_limits<std::uint64_t>::max() - *uid_count ||
      *gid_start > std::numeric_limits<std::uint64_t>::max() - *gid_count ||
      words[10].size() != 64 || words[11].size() != 64) {
    throw Error("invalid lease-journal field");
  }
  LeaseRecord record{*sequence, *epoch, words[3], *generation,
                     static_cast<LeaseState>(*state), *uid_start, *uid_count,
                     *gid_start, *gid_count, words[10], words[11]};
  if (sha256(record_body(record) + "\n") != record.digest) {
    throw Error("lease-journal digest mismatch");
  }
  return record;
}

bool allowed_transition(const std::optional<LeaseRecord>& previous,
                        const LeaseRecord& current) {
  if (!previous) return current.state == LeaseState::Reserved;
  if (current.uid_start != previous->uid_start ||
      current.uid_count != previous->uid_count ||
      current.gid_start != previous->gid_start ||
      current.gid_count != previous->gid_count || current.epoch < previous->epoch) {
    return false;
  }
  if (current.generation > previous->generation) {
    return previous->state == LeaseState::Free &&
           current.state == LeaseState::Reserved;
  }
  if (current.generation != previous->generation) return false;
  const LeaseState from = previous->state;
  const LeaseState to = current.state;
  return (from == LeaseState::Reserved && to == LeaseState::Mapped) ||
         (from == LeaseState::Mapped && to == LeaseState::Launched) ||
         (from == LeaseState::Launched && to == LeaseState::Draining) ||
         (from == LeaseState::Draining && to == LeaseState::Quarantined) ||
         (from == LeaseState::Quarantined && to == LeaseState::Free) ||
         (from != LeaseState::Free && from != LeaseState::Quarantined &&
          to == LeaseState::Quarantined && current.epoch > previous->epoch);
}

bool ranges_overlap(std::uint64_t left_start, std::uint64_t left_count,
                    std::uint64_t right_start, std::uint64_t right_count) {
  return left_start < right_start + right_count &&
         right_start < left_start + left_count;
}

class Journal {
 public:
  Journal(const std::string& path, bool create, bool create_exclusive,
          bool require_root, bool writable) {
    int flags = (writable ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW;
    if (create) flags |= O_CREAT;
    if (create_exclusive) flags |= O_EXCL;
    descriptor_ = UniqueFd(open(path.c_str(), flags, 0600));
    if (descriptor_.get() < 0) {
      throw Error("cannot open lease journal: " + std::string(std::strerror(errno)));
    }
    struct stat info {};
    if (fstat(descriptor_.get(), &info) != 0 || !S_ISREG(info.st_mode) ||
        info.st_nlink != 1 || (info.st_mode & 0777) != 0600 ||
        (require_root && (info.st_uid != 0 || info.st_gid != 0))) {
      throw Error("unsafe lease journal ownership or mode");
    }
    struct stat directory_info {};
    const std::string directory_path = parent_path(path);
    if (stat(directory_path.c_str(), &directory_info) != 0 ||
        !S_ISDIR(directory_info.st_mode) ||
        (require_root &&
         (directory_info.st_uid != 0 || directory_info.st_gid != 0 ||
          (directory_info.st_mode & (S_IWGRP | S_IWOTH)) != 0))) {
      throw Error("unsafe lease journal directory");
    }
    if (create) {
      UniqueFd directory(
          open(directory_path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
      if (directory.get() < 0 || fsync(directory.get()) != 0) {
        throw Error("cannot fsync lease journal directory");
      }
    }
    if (flock(descriptor_.get(), writable ? LOCK_EX : LOCK_SH) != 0) {
      throw Error("cannot lock lease journal");
    }
    replay();
  }

  void append(LeaseRecord record) {
    record.sequence = records_.empty() ? 1 : records_.back().sequence + 1;
    record.previous_digest = records_.empty() ? std::string(kZeroDigest)
                                              : records_.back().digest;
    const auto previous = latest_.find(record.lease);
    const std::optional<LeaseRecord> prior =
        previous == latest_.end() ? std::nullopt
                                  : std::optional<LeaseRecord>(previous->second);
    if (!allowed_transition(prior, record)) {
      throw Error("lease-journal transition refused");
    }
    if (!range_is_disjoint(record)) {
      throw Error("lease-journal range collision");
    }
    record.digest = sha256(record_body(record) + "\n");
    const std::string line = record_body(record) + " " + record.digest + "\n";
    if (lseek(descriptor_.get(), 0, SEEK_END) < 0) {
      throw Error("cannot seek lease journal");
    }
    std::size_t offset = 0;
    while (offset < line.size()) {
      const ssize_t count =
          write(descriptor_.get(), line.data() + offset, line.size() - offset);
      if (count > 0) {
        offset += static_cast<std::size_t>(count);
      } else if (count < 0 && errno != EINTR) {
        throw Error("cannot append lease journal");
      }
    }
    if (fsync(descriptor_.get()) != 0) throw Error("cannot fsync lease journal");
    records_.push_back(record);
    latest_[record.lease] = record;
  }

  const std::vector<LeaseRecord>& records() const { return records_; }
  const std::unordered_map<std::string, LeaseRecord>& latest() const {
    return latest_;
  }
  std::string head_digest() const {
    return records_.empty() ? std::string(kZeroDigest) : records_.back().digest;
  }
  std::uint64_t maximum_epoch() const {
    std::uint64_t maximum = 0;
    for (const auto& record : records_) maximum = std::max(maximum, record.epoch);
    return maximum;
  }

  std::size_t quarantine_uncertain() {
    const std::uint64_t recovery_epoch = maximum_epoch() + 1;
    std::vector<LeaseRecord> uncertain;
    for (const auto& entry : latest_) {
      if (entry.second.state != LeaseState::Free &&
          entry.second.state != LeaseState::Quarantined) {
        LeaseRecord quarantine = entry.second;
        quarantine.epoch = recovery_epoch;
        quarantine.state = LeaseState::Quarantined;
        quarantine.digest.clear();
        uncertain.push_back(quarantine);
      }
    }
    std::sort(uncertain.begin(), uncertain.end(),
              [](const LeaseRecord& left, const LeaseRecord& right) {
                return left.lease < right.lease;
              });
    for (const auto& record : uncertain) append(record);
    return uncertain.size();
  }

 private:
  void replay() {
    if (lseek(descriptor_.get(), 0, SEEK_SET) < 0) {
      throw Error("cannot rewind lease journal");
    }
    std::string contents;
    std::array<char, 8192> buffer{};
    for (;;) {
      const ssize_t count = read(descriptor_.get(), buffer.data(), buffer.size());
      if (count > 0) {
        contents.append(buffer.data(), static_cast<std::size_t>(count));
        if (contents.size() > 64 * 1024 * 1024) {
          throw Error("lease journal exceeds size limit");
        }
      } else if (count == 0) {
        break;
      } else if (errno != EINTR) {
        throw Error("cannot read lease journal");
      }
    }
    if (!contents.empty() && contents.back() != '\n') {
      throw Error("truncated lease journal");
    }
    std::istringstream input(contents);
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty()) throw Error("empty lease-journal record");
      LeaseRecord record = parse_record(line);
      if (record.sequence != records_.size() + 1 ||
          record.previous_digest !=
              (records_.empty() ? std::string(kZeroDigest) : records_.back().digest)) {
        throw Error("lease-journal chain discontinuity");
      }
      const auto previous = latest_.find(record.lease);
      const std::optional<LeaseRecord> prior =
          previous == latest_.end() ? std::nullopt
                                    : std::optional<LeaseRecord>(previous->second);
      if (!allowed_transition(prior, record)) {
        throw Error("lease-journal transition invalid during replay");
      }
      if (!range_is_disjoint(record)) {
        throw Error("lease-journal range collision during replay");
      }
      records_.push_back(record);
      latest_[record.lease] = record;
    }
  }

  bool range_is_disjoint(const LeaseRecord& candidate) const {
    if (candidate.state == LeaseState::Free) return true;
    for (const auto& entry : latest_) {
      if (entry.first == candidate.lease || entry.second.state == LeaseState::Free) {
        continue;
      }
      if (ranges_overlap(candidate.uid_start, candidate.uid_count,
                         entry.second.uid_start, entry.second.uid_count) ||
          ranges_overlap(candidate.gid_start, candidate.gid_count,
                         entry.second.gid_start, entry.second.gid_count)) {
        return false;
      }
    }
    return true;
  }

  UniqueFd descriptor_;
  std::vector<LeaseRecord> records_;
  std::unordered_map<std::string, LeaseRecord> latest_;
};

LeaseRecord make_record(std::uint64_t epoch, std::string lease,
                        std::uint64_t generation, LeaseState state) {
  LeaseRecord record;
  record.epoch = epoch;
  record.lease = std::move(lease);
  record.generation = generation;
  record.state = state;
  record.uid_start = 100000;
  record.uid_count = 1;
  record.gid_start = 100000;
  record.gid_count = 1;
  return record;
}

int selftest_journal(const std::string& path) {
  if (getuid() == 0 || geteuid() == 0) {
    throw Error("selftest journal refuses root execution");
  }
  Journal journal(path, true, true, false, true);
  journal.append(make_record(1, "selftest-lane", 1, LeaseState::Reserved));
  journal.append(make_record(1, "selftest-lane", 1, LeaseState::Mapped));
  journal.append(make_record(1, "selftest-lane", 1, LeaseState::Launched));
  journal.append(make_record(1, "selftest-lane", 1, LeaseState::Draining));
  journal.append(make_record(1, "selftest-lane", 1, LeaseState::Quarantined));
  journal.append(make_record(1, "selftest-lane", 1, LeaseState::Free));
  std::cout << "LOOM_KERNEL_PRINCIPAL_BROKER_JOURNAL_SELFTEST PASS records="
            << journal.records().size() << " head_sha256=" << journal.head_digest()
            << " final_state=FREE fsync=per-record\n";
  return 0;
}

int verify_journal(const std::string& path) {
  Journal journal(path, false, false, false, false);
  std::cout << "LOOM_KERNEL_PRINCIPAL_BROKER_JOURNAL_VERIFY PASS records="
            << journal.records().size() << " head_sha256=" << journal.head_digest()
            << "\n";
  return 0;
}

int selftest_recovery(const std::string& path) {
  if (getuid() == 0 || geteuid() == 0) {
    throw Error("recovery selftest refuses root execution");
  }
  {
    Journal journal(path, true, true, false, true);
    journal.append(make_record(1, "crashed-lane", 1, LeaseState::Reserved));
    journal.append(make_record(1, "crashed-lane", 1, LeaseState::Mapped));
    journal.append(make_record(1, "crashed-lane", 1, LeaseState::Launched));
  }
  Journal recovered(path, false, false, false, true);
  const std::size_t quarantined = recovered.quarantine_uncertain();
  const auto latest = recovered.latest().find("crashed-lane");
  if (quarantined != 1 || latest == recovered.latest().end() ||
      latest->second.state != LeaseState::Quarantined ||
      recovered.maximum_epoch() != 2) {
    throw Error("crash recovery did not force quarantine");
  }
  std::cout << "LOOM_KERNEL_PRINCIPAL_BROKER_RECOVERY_SELFTEST PASS records="
            << recovered.records().size() << " quarantined=1 final_state=QUARANTINED"
            << " recovery_epoch=2 head_sha256=" << recovered.head_digest() << "\n";
  return 0;
}

int selftest_collision(const std::string& path) {
  if (getuid() == 0 || geteuid() == 0) {
    throw Error("collision selftest refuses root execution");
  }
  Journal journal(path, true, true, false, true);
  journal.append(make_record(1, "lane-a", 1, LeaseState::Reserved));
  bool refused = false;
  try {
    journal.append(make_record(1, "lane-b", 1, LeaseState::Reserved));
  } catch (const Error& error) {
    refused = std::string(error.what()) == "lease-journal range collision";
  }
  if (!refused || journal.records().size() != 1) {
    throw Error("overlapping active range was not refused atomically");
  }
  std::cout << "LOOM_KERNEL_PRINCIPAL_BROKER_COLLISION_SELFTEST PASS"
            << " first=RESERVED second=REFUSED records=1\n";
  return 0;
}

int diagnose(const std::string& manifest_path, const std::string& authority_path,
             const std::string& socket_path) {
  const Manifest manifest = load_lease_manifest(manifest_path);
  verify_authority(manifest, authority_path, "action 9027");
  ActivationFacts facts = measure_activation(manifest_path, authority_path, socket_path);
  const CommandResult decision = run_authority(
      authority_path, current_frame(facts), "SOUNIO_KERNEL_PRINCIPAL_LEASE_");
  std::ostringstream receipt;
  receipt << "LOOM_KERNEL_PRINCIPAL_BROKER_DIAGNOSTIC"
          << " schema=loom-kernel-principal-broker-bootstrap-v1"
          << " manifest_sha256=" << manifest.digest
          << " authority_sha256=" << file_sha256(authority_path)
          << " root_identity=" << (facts.root_identity ? 1 : 0)
          << " pid1_systemd=" << (facts.pid1_systemd ? 1 : 0)
          << " parent_is_pid1=" << (facts.parent_is_pid1 ? 1 : 0)
          << " service_cgroup=" << (facts.service_cgroup ? 1 : 0)
          << " listen_environment=" << (facts.listen_environment ? 1 : 0)
          << " inherited_root_socket=" << (facts.inherited_root_socket ? 1 : 0)
          << " privilege_environment_absent="
          << (facts.privilege_environment_absent ? 1 : 0)
          << " artifacts_root_owned=" << (facts.artifacts_root_owned ? 1 : 0)
          << " activation_complete=" << (facts.complete() ? 1 : 0)
          << " decision_exit=" << decision.exit_code
          << " decision_sha256=" << sha256(decision.output + "\n")
          << " material_broker=0";
  const std::string receipt_text = receipt.str();
  std::cout << receipt_text << "\n";
  std::cout << "LOOM_KERNEL_PRINCIPAL_BROKER_RECEIPT_SHA256 "
            << sha256(receipt_text + "\n") << "\n";
  std::cout << decision.output << "\n";
  return 0;
}

int diagnose_invocation_cell(const std::string& manifest_path,
                             const std::string& authority_path,
                             const std::string& frame_path) {
  const Manifest manifest = load_invocation_cell_manifest(manifest_path);
  verify_authority(manifest, authority_path, "action 9029");
  const std::string frame = load_invocation_frame(frame_path);
  const InvocationAdmission admission =
      evaluate_invocation_cell(manifest, authority_path, frame);
  std::cout << admission.receipt << "\n";
  std::cout << "LOOM_KERNEL_INVOCATION_CELL_MATERIAL_RECEIPT_SHA256 "
            << sha256(admission.receipt + "\n") << "\n";
  std::cout << admission.result.output << "\n";
  return 0;
}

void write_response(int descriptor, const std::string& response) {
  std::size_t offset = 0;
  while (offset < response.size()) {
    const ssize_t count = write(descriptor, response.data() + offset,
                                response.size() - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno != EINTR) {
      return;
    }
  }
}

std::string status_response(const Journal& journal, const Manifest& lease_manifest,
                            const Manifest& capsule_manifest,
                            const Manifest& invocation_manifest) {
  std::array<std::size_t, 6> counts{};
  for (const auto& entry : journal.latest()) {
    ++counts[static_cast<std::size_t>(entry.second.state)];
  }
  std::ostringstream output;
  output << "LOOM_KERNEL_PRINCIPAL_BROKER_STATUS state=READY"
         << " lease_manifest_sha256=" << lease_manifest.digest
         << " lease_authority_sha256="
         << lease_manifest.require("executable_sha256")
         << " capsule_manifest_sha256=" << capsule_manifest.digest
         << " capsule_authority_sha256="
         << capsule_manifest.require("executable_sha256")
         << " invocation_manifest_sha256=" << invocation_manifest.digest
         << " invocation_authority_sha256="
         << invocation_manifest.require("executable_sha256")
         << " journal_head_sha256=" << journal.head_digest()
         << " epoch=" << journal.maximum_epoch()
         << " free=" << counts[0] << " reserved=" << counts[1]
         << " mapped=" << counts[2] << " launched=" << counts[3]
         << " draining=" << counts[4] << " quarantined=" << counts[5]
         << " admission_open=true launch_open=false recycle_open=false\n";
  return output.str();
}

std::string bootstrap_response(const std::string& request, const Journal* journal,
                               const Manifest* lease_manifest,
                               const Manifest* capsule_manifest,
                               const Manifest* invocation_manifest,
                               const std::string* invocation_authority_path) {
  if (request == "STATUS") {
    if (journal == nullptr || lease_manifest == nullptr ||
        capsule_manifest == nullptr || invocation_manifest == nullptr ||
        invocation_authority_path == nullptr) {
      return "DENY status-context-absent\n";
    }
    return status_response(*journal, *lease_manifest, *capsule_manifest,
                           *invocation_manifest);
  }
  if (request.rfind("ADMIT ", 0) == 0) {
    if (invocation_manifest == nullptr || invocation_authority_path == nullptr) {
      return "DENY admission-context-absent\n";
    }
    const std::string frame = validate_invocation_frame(request.substr(6));
    const InvocationAdmission admission = evaluate_invocation_cell(
        *invocation_manifest, *invocation_authority_path, frame);
    return admission.receipt + "\n";
  }
  if (request == "ADMIT") {
    return "DENY malformed-admission\n";
  }
  if (request.rfind("LAUNCH", 0) == 0) {
    return "DENY bootstrap-launch-closed\n";
  }
  if (request.rfind("RECYCLE", 0) == 0) {
    return "DENY bootstrap-recycle-closed\n";
  }
  return "DENY unknown-request\n";
}

void require_root_client_socket(const std::string& socket_path) {
  if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0) {
    throw Error("live broker probe requires root identity");
  }
  struct stat socket_info {};
  struct stat directory_info {};
  if (lstat(socket_path.c_str(), &socket_info) != 0 ||
      stat(parent_path(socket_path).c_str(), &directory_info) != 0 ||
      !S_ISSOCK(socket_info.st_mode) || socket_info.st_uid != 0 ||
      socket_info.st_gid != 0 || (socket_info.st_mode & 0777) != 0600 ||
      !S_ISDIR(directory_info.st_mode) || directory_info.st_uid != 0 ||
      directory_info.st_gid != 0 ||
      (directory_info.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw Error("live broker socket ownership or mode is unsafe");
  }
}

std::string exchange_with_live_broker(const std::string& socket_path,
                                      const std::string& request) {
  require_root_client_socket(socket_path);
  UniqueFd descriptor(socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (descriptor.get() < 0) throw Error("cannot create live broker probe socket");
  sockaddr_un address{};
  if (socket_path.size() >= sizeof(address.sun_path)) {
    throw Error("live broker socket path is too long");
  }
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
  if (connect(descriptor.get(), reinterpret_cast<sockaddr*>(&address),
              sizeof(address)) != 0) {
    throw Error("cannot connect to live broker socket: " +
                std::string(std::strerror(errno)));
  }
  ucred peer{};
  socklen_t peer_size = sizeof(peer);
  if (getsockopt(descriptor.get(), SOL_SOCKET, SO_PEERCRED, &peer, &peer_size) !=
          0 ||
      peer.uid != 0 || peer.gid != 0 || peer.pid <= 0) {
    throw Error("live broker peer is not a root host endpoint");
  }
  write_response(descriptor.get(), request + "\n");
  if (shutdown(descriptor.get(), SHUT_WR) != 0) {
    throw Error("cannot finish live broker probe request");
  }
  std::string response;
  std::array<char, 1024> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor.get(), buffer.data(), buffer.size());
    if (count > 0) {
      response.append(buffer.data(), static_cast<std::size_t>(count));
      if (response.size() > kMaximumResponse) {
        throw Error("live broker response exceeds size limit");
      }
      if (response.find('\n') != std::string::npos) break;
    } else if (count == 0) {
      break;
    } else if (errno != EINTR) {
      throw Error("cannot read live broker response");
    }
  }
  if (response.empty() || response.back() != '\n' ||
      response.find('\n') != response.size() - 1) {
    throw Error("live broker response is malformed");
  }
  return response;
}

int probe_live_broker(const std::string& socket_path) {
  const std::string status = exchange_with_live_broker(socket_path, "STATUS");
  if (status.rfind("LOOM_KERNEL_PRINCIPAL_BROKER_STATUS state=READY ", 0) != 0) {
    throw Error("live broker did not return READY status");
  }
  if (exchange_with_live_broker(socket_path, "LAUNCH sabotage") !=
          "DENY bootstrap-launch-closed\n" ||
      exchange_with_live_broker(socket_path, "RECYCLE sabotage") !=
          "DENY bootstrap-recycle-closed\n") {
    throw Error("live broker opened a bootstrap operation");
  }
  const std::string admission =
      exchange_with_live_broker(socket_path, "ADMIT 9029 3");
  if (admission.rfind(
          "LOOM_KERNEL_INVOCATION_CELL_MATERIAL_ADMISSION ", 0) != 0 ||
      admission.find(" decision=DENY decision_code=424 ") == std::string::npos ||
      admission.find(" material_invocation=false ") == std::string::npos ||
      admission.find(" launch_open=false\n") == std::string::npos ||
      exchange_with_live_broker(socket_path, "EXEC sabotage") !=
          "DENY unknown-request\n") {
    throw Error("live broker InvocationCell admission probe failed");
  }
  std::cout << status.substr(0, status.size() - 1)
            << " live_probe=PASS admission=DENY424 launch=closed recycle=closed"
            << " unknown=denied\n";
  return 0;
}

int selftest_protocol() {
  if (getuid() == 0 || geteuid() == 0) {
    throw Error("protocol selftest refuses root execution");
  }
  if (bootstrap_response("LAUNCH lane", nullptr, nullptr, nullptr, nullptr,
                         nullptr) !=
          "DENY bootstrap-launch-closed\n" ||
      bootstrap_response("RECYCLE lane", nullptr, nullptr, nullptr, nullptr,
                         nullptr) !=
          "DENY bootstrap-recycle-closed\n" ||
      bootstrap_response("ADMIT 9029 3", nullptr, nullptr, nullptr, nullptr,
                         nullptr) != "DENY admission-context-absent\n" ||
      bootstrap_response("ADMIT", nullptr, nullptr, nullptr, nullptr,
                         nullptr) != "DENY malformed-admission\n" ||
      bootstrap_response("EXEC lane", nullptr, nullptr, nullptr, nullptr,
                         nullptr) !=
          "DENY unknown-request\n" ||
      bootstrap_response("STATUS", nullptr, nullptr, nullptr, nullptr,
                         nullptr) !=
          "DENY status-context-absent\n") {
    throw Error("bootstrap protocol opened a material operation");
  }
  std::cout << "LOOM_KERNEL_PRINCIPAL_BROKER_PROTOCOL_SELFTEST PASS"
            << " admission_without_context=denied malformed_admission=denied"
            << " launch=closed recycle=closed unknown=denied partial_status=denied\n";
  return 0;
}

std::optional<std::string> read_request_line(int descriptor) {
  std::string request;
  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor, buffer.data(), buffer.size());
    if (count > 0) {
      request.append(buffer.data(), static_cast<std::size_t>(count));
      if (request.size() > kMaximumRequest) return std::nullopt;
      const std::size_t newline = request.find('\n');
      if (newline != std::string::npos) {
        if (newline != request.size() - 1 ||
            request.find('\n', newline + 1) != std::string::npos) {
          return std::nullopt;
        }
        request.pop_back();
        if (!request.empty() && request.back() == '\r') request.pop_back();
        if (request.empty() || request.find('\r') != std::string::npos) {
          return std::nullopt;
        }
        return request;
      }
    } else if (count == 0) {
      return std::nullopt;
    } else if (errno != EINTR) {
      return std::nullopt;
    }
  }
}

int serve(const std::string& manifest_path, const std::string& authority_path,
          const std::string& capsule_manifest_path,
          const std::string& capsule_authority_path,
          const std::string& invocation_manifest_path,
          const std::string& invocation_authority_path,
          const std::string& journal_path, const std::string& socket_path) {
  const Manifest lease_manifest = load_lease_manifest(manifest_path);
  const Manifest capsule_manifest = load_capsule_manifest(capsule_manifest_path);
  const Manifest invocation_manifest =
      load_invocation_cell_manifest(invocation_manifest_path);
  verify_authority(lease_manifest, authority_path, "action 9027");
  verify_authority(capsule_manifest, capsule_authority_path, "action 9028");
  verify_authority(invocation_manifest, invocation_authority_path, "action 9029");
  const ActivationFacts facts =
      measure_activation(manifest_path, authority_path, socket_path,
                         capsule_manifest_path, capsule_authority_path,
                         invocation_manifest_path, invocation_authority_path);
  if (!facts.complete()) {
    throw Error("host service-manager activation boundary incomplete");
  }
  Journal journal(journal_path, true, false, true, true);
  const std::size_t quarantined = journal.quarantine_uncertain();
  std::cerr << "loom-kernel-principal-broker: READY quarantined=" << quarantined
            << " journal_head_sha256=" << journal.head_digest() << "\n";
  for (;;) {
    UniqueFd client(accept4(3, nullptr, nullptr, SOCK_CLOEXEC));
    if (client.get() < 0) {
      if (errno == EINTR) continue;
      throw Error("accept failed");
    }
    ucred peer{};
    socklen_t peer_size = sizeof(peer);
    if (getsockopt(client.get(), SOL_SOCKET, SO_PEERCRED, &peer, &peer_size) != 0 ||
        peer.uid != 0 || peer.gid != 0) {
      write_response(client.get(), "DENY peer-not-root\n");
      continue;
    }
    const std::optional<std::string> request = read_request_line(client.get());
    if (!request.has_value()) {
      write_response(client.get(), "DENY malformed-request\n");
      continue;
    }
    write_response(client.get(), bootstrap_response(
                                     *request, &journal, &lease_manifest,
                                     &capsule_manifest, &invocation_manifest,
                                     &invocation_authority_path));
  }
}

struct Options {
  std::string mode;
  std::string manifest;
  std::string authority;
  std::string capsule_manifest;
  std::string capsule_authority;
  std::string invocation_manifest;
  std::string invocation_authority;
  std::string frame;
  std::string journal;
  std::string socket_path = "/run/sounio/loom-principal-broker.sock";
};

Options parse_options(int argc, char** argv) {
  if (argc < 2) throw Error("missing mode");
  Options options;
  options.mode = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (index + 1 >= argc) throw Error("option omitted value: " + argument);
    const std::string value = argv[++index];
    if (argument == "--manifest") {
      options.manifest = value;
    } else if (argument == "--authority") {
      options.authority = value;
    } else if (argument == "--capsule-manifest") {
      options.capsule_manifest = value;
    } else if (argument == "--capsule-authority") {
      options.capsule_authority = value;
    } else if (argument == "--invocation-manifest") {
      options.invocation_manifest = value;
    } else if (argument == "--invocation-authority") {
      options.invocation_authority = value;
    } else if (argument == "--frame") {
      options.frame = value;
    } else if (argument == "--journal") {
      options.journal = value;
    } else if (argument == "--socket-path") {
      options.socket_path = value;
    } else {
      throw Error("unknown option: " + argument);
    }
  }
  return options;
}

void require_artifacts(const Options& options) {
  if (options.manifest.empty() || options.authority.empty()) {
    throw Error("manifest and authority are required");
  }
}

void require_serve_artifacts(const Options& options) {
  require_artifacts(options);
  if (options.capsule_manifest.empty() || options.capsule_authority.empty()) {
    throw Error("capsule manifest and authority are required");
  }
  if (options.invocation_manifest.empty() ||
      options.invocation_authority.empty()) {
    throw Error("InvocationCell manifest and authority are required");
  }
}

void require_invocation_artifacts(const Options& options) {
  if (options.invocation_manifest.empty() ||
      options.invocation_authority.empty() || options.frame.empty()) {
    throw Error("InvocationCell manifest, authority, and frame are required");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.mode == "--diagnose") {
      require_artifacts(options);
      return diagnose(options.manifest, options.authority, options.socket_path);
    }
    if (options.mode == "--diagnose-invocation-cell") {
      require_invocation_artifacts(options);
      return diagnose_invocation_cell(options.invocation_manifest,
                                      options.invocation_authority,
                                      options.frame);
    }
    if (options.mode == "--selftest-journal") {
      if (options.journal.empty()) throw Error("journal is required");
      return selftest_journal(options.journal);
    }
    if (options.mode == "--verify-journal") {
      if (options.journal.empty()) throw Error("journal is required");
      return verify_journal(options.journal);
    }
    if (options.mode == "--selftest-protocol") {
      return selftest_protocol();
    }
    if (options.mode == "--probe-live") {
      return probe_live_broker(options.socket_path);
    }
    if (options.mode == "--selftest-recovery") {
      if (options.journal.empty()) throw Error("journal is required");
      return selftest_recovery(options.journal);
    }
    if (options.mode == "--selftest-collision") {
      if (options.journal.empty()) throw Error("journal is required");
      return selftest_collision(options.journal);
    }
    if (options.mode == "--serve") {
      require_serve_artifacts(options);
      if (options.journal.empty()) throw Error("journal is required");
      return serve(options.manifest, options.authority, options.capsule_manifest,
                   options.capsule_authority, options.invocation_manifest,
                   options.invocation_authority, options.journal,
                   options.socket_path);
    }
    throw Error("unknown mode");
  } catch (const std::exception& error) {
    std::cerr << "loom-kernel-principal-broker: REFUSE reason=" << error.what()
              << "\n";
    return 70;
  }
}
