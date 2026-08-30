#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <openssl/sha.h>
#include <openssl/evp.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern char** environ;

namespace {

constexpr std::size_t kMaximumRecordBytes = 64 * 1024;
constexpr auto kExecutionTimeout = std::chrono::seconds(15);

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct Fd {
  int value = -1;
  Fd() = default;
  explicit Fd(int descriptor) : value(descriptor) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& other) noexcept : value(other.value) { other.value = -1; }
  Fd& operator=(Fd&& other) noexcept {
    if (this != &other) {
      if (value >= 0) close(value);
      value = other.value;
      other.value = -1;
    }
    return *this;
  }
  ~Fd() {
    if (value >= 0) close(value);
  }
  int get() const { return value; }
};

std::string sha256(std::string_view value) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char*>(value.data()), value.size(),
         digest);
  static constexpr char hex[] = "0123456789abcdef";
  std::string output(SHA256_DIGEST_LENGTH * 2, '0');
  for (std::size_t index = 0; index < SHA256_DIGEST_LENGTH; ++index) {
    output[index * 2] = hex[digest[index] >> 4];
    output[index * 2 + 1] = hex[digest[index] & 0x0f];
  }
  return output;
}

bool digest(const std::string& value) {
  if (value.size() != 64) return false;
  for (const unsigned char character : value) {
    if (!std::isdigit(character) &&
        !(character >= 'a' && character <= 'f')) {
      return false;
    }
  }
  return value != std::string(64, '0');
}

std::string read_fd(int descriptor, std::size_t limit = kMaximumRecordBytes) {
  if (lseek(descriptor, 0, SEEK_SET) < 0) {
    throw Error("descriptor is not seekable");
  }
  std::string output;
  std::array<char, 8192> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor, buffer.data(), buffer.size());
    if (count > 0) {
      output.append(buffer.data(), static_cast<std::size_t>(count));
      if (output.size() > limit) throw Error("descriptor content too large");
    } else if (count == 0) {
      return output;
    } else if (errno != EINTR) {
      throw Error(std::string("descriptor read failed: ") + std::strerror(errno));
    }
  }
}

std::string file_sha256(int descriptor) {
  if (lseek(descriptor, 0, SEEK_SET) < 0) {
    throw Error("descriptor is not seekable");
  }
  EVP_MD_CTX* context = EVP_MD_CTX_new();
  if (context == nullptr) throw Error("SHA256 context allocation failed");
  const auto free_context = [&]() { EVP_MD_CTX_free(context); };
  if (EVP_DigestInit_ex(context, EVP_sha256(), nullptr) != 1) {
    free_context();
    throw Error("SHA256 init failed");
  }
  std::array<unsigned char, 64 * 1024> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor, buffer.data(), buffer.size());
    if (count > 0) {
      if (EVP_DigestUpdate(context, buffer.data(),
                           static_cast<std::size_t>(count)) != 1) {
        free_context();
        throw Error("SHA256 update failed");
      }
    } else if (count == 0) {
      break;
    } else if (errno != EINTR) {
      const std::string reason =
          std::string("descriptor read failed: ") + std::strerror(errno);
      free_context();
      throw Error(reason);
    }
  }
  unsigned char result[SHA256_DIGEST_LENGTH];
  unsigned int result_size = 0;
  if (EVP_DigestFinal_ex(context, result, &result_size) != 1 ||
      result_size != SHA256_DIGEST_LENGTH) {
    free_context();
    throw Error("SHA256 final failed");
  }
  free_context();
  static constexpr char alphabet[] = "0123456789abcdef";
  std::string output(SHA256_DIGEST_LENGTH * 2, '0');
  for (std::size_t index = 0; index < SHA256_DIGEST_LENGTH; ++index) {
    output[index * 2] = alphabet[result[index] >> 4];
    output[index * 2 + 1] = alphabet[result[index] & 0x0f];
  }
  return output;
}

std::string trim(std::string value) {
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r' ||
          value.back() == ' ' || value.back() == '\t')) {
    value.pop_back();
  }
  return value;
}

std::string read_small_path(const std::string& path) {
  Fd descriptor(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (descriptor.get() < 0) {
    throw Error("cannot read process identity: " + path);
  }
  return read_fd(descriptor.get(), 128 * 1024);
}

std::uint64_t start_tick() {
  const std::string value = trim(read_small_path("/proc/self/stat"));
  const std::size_t close = value.rfind(')');
  if (close == std::string::npos) throw Error("process stat malformed");
  std::istringstream input(value.substr(close + 2));
  std::string field;
  for (int index = 0; index <= 19; ++index) {
    if (!(input >> field)) throw Error("process stat truncated");
  }
  std::size_t consumed = 0;
  const auto parsed = std::stoull(field, &consumed, 10);
  if (consumed != field.size() || parsed == 0) {
    throw Error("process start tick malformed");
  }
  return parsed;
}

std::string executable_sha256() {
  Fd descriptor(open("/proc/self/exe", O_RDONLY | O_CLOEXEC));
  if (descriptor.get() < 0) throw Error("cannot open material cell executable");
  return file_sha256(descriptor.get());
}

struct Principal {
  pid_t pid = 0;
  uid_t uid = 0;
  gid_t gid = 0;
  std::uint64_t start = 0;
  std::string cgroup_sha256;
  std::string executable_sha256;
  std::string canonical;
  std::string sha256;
};

Principal principal() {
  Principal value;
  value.pid = getpid();
  value.uid = getuid();
  value.gid = getgid();
  if (value.uid == 0 || value.gid == 0) {
    throw Error("root material cell principal refused");
  }
  value.start = start_tick();
  value.cgroup_sha256 = sha256(read_small_path("/proc/self/cgroup"));
  value.executable_sha256 = executable_sha256();
  std::ostringstream canonical;
  canonical << "LOOM_CAUSAL_MATERIAL_PRINCIPAL/1"
            << "|pid=" << value.pid << "|start_tick=" << value.start
            << "|uid=" << value.uid << "|gid=" << value.gid
            << "|cgroup_sha256=" << value.cgroup_sha256
            << "|executable_sha256=" << value.executable_sha256;
  value.canonical = canonical.str();
  value.sha256 = sha256(value.canonical);
  return value;
}

std::map<std::string, int> inherited_descriptors() {
  const char* count_text = std::getenv("LISTEN_FDS");
  const char* names_text = std::getenv("LISTEN_FDNAMES");
  const char* pid_text = std::getenv("LISTEN_PID");
  if (count_text == nullptr || names_text == nullptr || pid_text == nullptr) {
    throw Error("named descriptor environment absent");
  }
  std::size_t consumed = 0;
  const long long listen_pid = std::stoll(pid_text, &consumed, 10);
  if (consumed != std::strlen(pid_text) || listen_pid != getpid()) {
    throw Error("named descriptor pid binding invalid");
  }
  consumed = 0;
  const long count = std::stol(count_text, &consumed, 10);
  if (consumed != std::strlen(count_text) || count <= 0 || count > 8) {
    throw Error("named descriptor count invalid");
  }
  std::vector<std::string> names;
  std::istringstream input(names_text);
  std::string name;
  while (std::getline(input, name, ':')) names.push_back(name);
  if (names.size() != static_cast<std::size_t>(count)) {
    throw Error("named descriptor vector mismatch");
  }
  std::map<std::string, int> descriptors;
  for (long index = 0; index < count; ++index) {
    const int descriptor = 3 + static_cast<int>(index);
    struct stat info {};
    if (fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) ||
        info.st_nlink != 1) {
      throw Error("named descriptor is not one regular file");
    }
    const int flags = fcntl(descriptor, F_GETFL);
    if (flags < 0 || (flags & O_ACCMODE) != O_RDONLY) {
      throw Error("named descriptor is not read-only");
    }
    if (!descriptors.emplace(names[index], descriptor).second) {
      throw Error("named descriptor duplicated");
    }
  }
  return descriptors;
}

std::string descriptor_binding(const std::map<std::string, int>& descriptors) {
  std::ostringstream canonical;
  canonical << "LOOM_CAUSAL_MATERIAL_DESCRIPTORS/1";
  for (const auto& [name, descriptor] : descriptors) {
    struct stat info {};
    if (fstat(descriptor, &info) != 0) throw Error("descriptor fstat failed");
    canonical << '|' << name << '=' << info.st_dev << ':' << info.st_ino
              << ':' << info.st_size;
  }
  return sha256(canonical.str());
}

std::vector<std::string> words(const std::string& line) {
  std::istringstream input(line);
  std::vector<std::string> output;
  std::string word;
  while (input >> word) output.push_back(word);
  return output;
}

std::map<std::string, std::string> record_fields(const std::string& record) {
  if (record.empty() || record.back() != '\n') {
    throw Error("record missing final newline");
  }
  std::istringstream input(record);
  std::string line;
  std::map<std::string, std::string> fields;
  if (!std::getline(input, line) || line.empty()) {
    throw Error("record schema absent");
  }
  fields.emplace("schema", line);
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos || equals == 0 || equals + 1 >= line.size() ||
        !fields.emplace(line.substr(0, equals), line.substr(equals + 1)).second) {
      throw Error("record field malformed or duplicated");
    }
  }
  return fields;
}

const std::string& require(const std::map<std::string, std::string>& fields,
                           const std::string& key) {
  const auto found = fields.find(key);
  if (found == fields.end() || found->second.empty()) {
    throw Error("record omitted " + key);
  }
  return found->second;
}

struct Execution {
  int exit_code = 255;
  std::string stdout_text;
  std::string stderr_text;
};

std::string read_pipe(int descriptor, std::size_t limit) {
  std::string output;
  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor, buffer.data(), buffer.size());
    if (count > 0) {
      output.append(buffer.data(), static_cast<std::size_t>(count));
      if (output.size() > limit) throw Error("artifact output too large");
    } else if (count == 0) {
      return output;
    } else if (errno != EINTR) {
      throw Error("artifact output read failed");
    }
  }
}

Execution run_artifact(int artifact) {
  int stdout_pipe[2];
  int stderr_pipe[2];
  if (pipe2(stdout_pipe, O_CLOEXEC) != 0 ||
      pipe2(stderr_pipe, O_CLOEXEC) != 0) {
    throw Error("cannot create artifact capture pipes");
  }
  Fd stdout_read(stdout_pipe[0]);
  Fd stdout_write(stdout_pipe[1]);
  Fd stderr_read(stderr_pipe[0]);
  Fd stderr_write(stderr_pipe[1]);
  const pid_t pid = fork();
  if (pid < 0) throw Error("cannot fork artifact");
  if (pid == 0) {
    if (dup2(stdout_write.get(), STDOUT_FILENO) < 0 ||
        dup2(stderr_write.get(), STDERR_FILENO) < 0) {
      _exit(126);
    }
    stdout_read = Fd();
    stdout_write = Fd();
    stderr_read = Fd();
    stderr_write = Fd();
    char argument[] = "loom-artifact";
    char* arguments[] = {argument, nullptr};
    char lang[] = "LANG=C";
    char locale[] = "LC_ALL=C";
    char timezone[] = "TZ=UTC";
    char epoch[] = "SOURCE_DATE_EPOCH=0";
    char path[] = "PATH=/usr/bin:/bin";
    char home[] = "HOME=/nonexistent";
    char* environment[] = {lang, locale, timezone, epoch, path, home, nullptr};
    fexecve(artifact, arguments, environment);
    _exit(127);
  }
  stdout_write = Fd();
  stderr_write = Fd();
  const auto deadline = std::chrono::steady_clock::now() + kExecutionTimeout;
  int status = 0;
  for (;;) {
    const pid_t waited = waitpid(pid, &status, WNOHANG);
    if (waited == pid) break;
    if (waited < 0 && errno != EINTR) throw Error("artifact wait failed");
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
      }
      throw Error("artifact execution timed out");
    }
    poll(nullptr, 0, 5);
  }
  Execution result;
  result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status)
                                      : 128 + WTERMSIG(status);
  result.stdout_text = read_pipe(stdout_read.get(), 64 * 1024);
  result.stderr_text = read_pipe(stderr_read.get(), 64 * 1024);
  return result;
}

void close_protocol(const std::string& mode, const std::string& record_sha256) {
  std::string line;
  if (!std::getline(std::cin, line)) throw Error("close frame absent");
  const auto frame = words(line);
  if (frame.size() != 3 || frame[0] != "CLOSE" || frame[1] != mode ||
      frame[2] != record_sha256) {
    throw Error("close frame binding invalid");
  }
  std::cout << "LOOM_CAUSAL_MATERIAL_CELL_CLOSED_V1 mode=" << mode
            << " record_sha256=" << record_sha256
            << " authority_extinction=armed\n" << std::flush;
}

int run_cell(const Principal& identity,
             const std::map<std::string, int>& descriptors,
             const std::string& descriptor_sha256) {
  if (descriptors.size() != 1 || !descriptors.contains("artifact")) {
    throw Error("RUN_EXACT descriptor set invalid");
  }
  const int artifact = descriptors.at("artifact");
  struct stat info {};
  if (fstat(artifact, &info) != 0 || (info.st_mode & 0111) == 0) {
    throw Error("RUN_EXACT artifact is not executable");
  }
  std::cout << "LOOM_CAUSAL_MATERIAL_CELL_READY_V1 mode=RUN_EXACT"
            << " semantic_authority=Sounio action=9037 pid=" << identity.pid
            << " start_tick=" << identity.start << " uid=" << identity.uid
            << " gid=" << identity.gid
            << " principal_sha256=" << identity.sha256
            << " descriptor_binding_sha256=" << descriptor_sha256
            << " inherited_descriptors=true arbitrary_path=false\n" << std::flush;
  std::string line;
  if (!std::getline(std::cin, line)) throw Error("RUN_EXACT arm frame absent");
  const auto frame = words(line);
  if (frame.size() != 4 || frame[0] != "ARM" || frame[1] != "RUN_EXACT" ||
      !digest(frame[2]) || !digest(frame[3])) {
    throw Error("RUN_EXACT arm frame malformed");
  }
  const std::string ticket = frame[2];
  const std::string artifact_sha256 = file_sha256(artifact);
  if (artifact_sha256 != frame[3]) {
    throw Error("RUN_EXACT artifact handle mismatch");
  }
  const Execution result = run_artifact(artifact);
  const std::string stdout_sha256 = sha256(result.stdout_text);
  const std::string stderr_sha256 = sha256(result.stderr_text);
  std::ostringstream record;
  record << "loom-causal-run-result-v1\n"
         << "semantic_authority=Sounio\nsemantic_action=9037\n"
         << "mode=RUN_EXACT\nrun_ticket_sha256=" << ticket << '\n'
         << "artifact_sha256=" << artifact_sha256 << '\n'
         << "principal_sha256=" << identity.sha256 << '\n'
         << "descriptor_binding_sha256=" << descriptor_sha256 << '\n'
         << "exit_code=" << result.exit_code << '\n'
         << "stdout_sha256=" << stdout_sha256 << '\n'
         << "stderr_sha256=" << stderr_sha256 << '\n';
  const std::string record_text = record.str();
  const std::string record_sha256 = sha256(record_text);
  const std::string handle =
      "loom-result-v3:" + artifact_sha256 + ":" + record_sha256;
  std::cout << "LOOM_CAUSAL_MATERIAL_CELL_RESULT_V1 mode=RUN_EXACT"
            << " record_sha256=" << record_sha256
            << " handle_sha256=" << sha256(handle)
            << " artifact_sha256=" << artifact_sha256
            << " exit_code=" << result.exit_code
            << " stdout_sha256=" << stdout_sha256
            << " stderr_sha256=" << stderr_sha256
            << " record_bytes=" << record_text.size()
            << " handle_is_bearer=false handle_is_execution_authority=false\n"
            << "LOOM_CAUSAL_MATERIAL_RECORD_BEGIN\n" << record_text
            << "LOOM_CAUSAL_MATERIAL_RECORD_END\n"
            << std::flush;
  close_protocol("RUN_EXACT", record_sha256);
  return result.exit_code == 0 ? 0 : 70;
}

int attest_cell(const Principal& identity,
                const std::map<std::string, int>& descriptors,
                const std::string& descriptor_sha256) {
  const std::array<std::string, 4> expected = {
      "compile_record", "hardware_record", "result_record",
      "semantics_manifest"};
  if (descriptors.size() != expected.size()) {
    throw Error("ATTEST descriptor set invalid");
  }
  for (const auto& name : expected) {
    if (!descriptors.contains(name)) throw Error("ATTEST descriptor absent: " + name);
  }
  std::cout << "LOOM_CAUSAL_MATERIAL_CELL_READY_V1 mode=ATTEST"
            << " semantic_authority=Sounio action=9037 pid=" << identity.pid
            << " start_tick=" << identity.start << " uid=" << identity.uid
            << " gid=" << identity.gid
            << " principal_sha256=" << identity.sha256
            << " descriptor_binding_sha256=" << descriptor_sha256
            << " inherited_descriptors=true arbitrary_path=false\n" << std::flush;
  std::string line;
  if (!std::getline(std::cin, line)) throw Error("ATTEST arm frame absent");
  const auto frame = words(line);
  if (frame.size() != 6 || frame[0] != "ARM" || frame[1] != "ATTEST") {
    throw Error("ATTEST arm frame malformed");
  }
  for (std::size_t index = 2; index < frame.size(); ++index) {
    if (!digest(frame[index])) throw Error("ATTEST arm digest malformed");
  }
  const std::string compile_text = read_fd(descriptors.at("compile_record"));
  const std::string result_text = read_fd(descriptors.at("result_record"));
  const std::string semantics_text =
      read_fd(descriptors.at("semantics_manifest"), 1024 * 1024);
  const std::string hardware_text = read_fd(descriptors.at("hardware_record"));
  if (sha256(compile_text) != frame[2] || sha256(result_text) != frame[3] ||
      sha256(semantics_text) != frame[4] || sha256(hardware_text) != frame[5]) {
    throw Error("ATTEST descriptor digest mismatch");
  }
  const auto compile = record_fields(compile_text);
  const auto result = record_fields(result_text);
  if (require(compile, "artifact_sha256") !=
      require(result, "artifact_sha256")) {
    throw Error("ATTEST artifact lineage mismatch");
  }
  if (require(result, "semantic_action") != "9037" ||
      require(result, "mode") != "RUN_EXACT" ||
      require(result, "exit_code") != "0") {
    throw Error("ATTEST result posture invalid");
  }
  std::ostringstream record;
  record << "loom-causal-attestation-record-v1\n"
         << "semantic_authority=Sounio\nsemantic_action=9037\n"
         << "source_sha256=" << require(compile, "source_sha256") << '\n'
         << "frozen_semantics_sha256=" << frame[4] << '\n'
         << "artifact_sha256=" << require(compile, "artifact_sha256") << '\n'
         << "artifact_record_sha256=" << frame[2] << '\n'
         << "result_record_sha256=" << frame[3] << '\n'
         << "toolchain_sha256=" << require(compile, "compiler_sha256") << '\n'
         << "hardware_record_sha256=" << frame[5] << '\n'
         << "principal_sha256=" << identity.sha256 << '\n'
         << "descriptor_binding_sha256=" << descriptor_sha256 << '\n'
         << "producing_language=C++20\n"
         << "language_role=MATERIAL_PARITY\n"
         << "handle_is_bearer=false\n"
         << "handle_is_execution_authority=false\n";
  const std::string record_text = record.str();
  const std::string record_sha256 = sha256(record_text);
  const std::string handle = "loom-attestation-v1:" + frame[4] + ":" +
                             frame[3] + ":" + record_sha256;
  std::cout << "LOOM_CAUSAL_MATERIAL_CELL_RESULT_V1 mode=ATTEST"
            << " record_sha256=" << record_sha256
            << " handle_sha256=" << sha256(handle)
            << " artifact_sha256=" << require(compile, "artifact_sha256")
            << " result_record_sha256=" << frame[3]
            << " semantics_sha256=" << frame[4]
            << " hardware_record_sha256=" << frame[5]
            << " record_bytes=" << record_text.size()
            << " handle_is_bearer=false handle_is_execution_authority=false\n"
            << "LOOM_CAUSAL_MATERIAL_RECORD_BEGIN\n" << record_text
            << "LOOM_CAUSAL_MATERIAL_RECORD_END\n"
            << std::flush;
  close_protocol("ATTEST", record_sha256);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 || std::string(argv[1]) != "--mode") {
      throw Error("usage: loom-causal-workflow-material-cell --mode RUN_EXACT|ATTEST");
    }
    const std::string mode = argv[2];
    const Principal identity = principal();
    const auto descriptors = inherited_descriptors();
    const std::string binding = descriptor_binding(descriptors);
    if (mode == "RUN_EXACT") return run_cell(identity, descriptors, binding);
    if (mode == "ATTEST") return attest_cell(identity, descriptors, binding);
    throw Error("material cell mode invalid");
  } catch (const Error& error) {
    std::cerr << "LOOM_CAUSAL_MATERIAL_CELL_REFUSED reason=" << error.what()
              << " material_execution=false\n";
    return 70;
  } catch (const std::exception& error) {
    std::cerr << "LOOM_CAUSAL_MATERIAL_CELL_REFUSED reason=unexpected:"
              << error.what() << " material_execution=false\n";
    return 70;
  }
}
