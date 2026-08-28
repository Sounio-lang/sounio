#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sys/prctl.h>
#include <sys/random.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kGenerationBytes = 32;
constexpr std::size_t kReleaseBytes = kGenerationBytes * 2 + 1;
constexpr std::size_t kMaximumInputBytes = kReleaseBytes * 2;
constexpr int kDeadlineMs = 75;

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class UniqueFd {
 public:
  UniqueFd() = default;
  explicit UniqueFd(int value) : value_(value) {}
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : value_(other.release()) {}
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) reset(other.release());
    return *this;
  }
  ~UniqueFd() { reset(); }

  int get() const { return value_; }
  int release() {
    const int value = value_;
    value_ = -1;
    return value;
  }
  void reset(int value = -1) {
    if (value_ >= 0) close(value_);
    value_ = value;
  }

 private:
  int value_ = -1;
};

struct Observation {
  bool opened = false;
  std::string reason;
};

enum class CaseKind {
  Treatment,
  Sabotage,
  WrongGeneration,
  Truncated,
  Oversized,
  Duplicate,
  Timeout,
  DescriptorAbsent,
};

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
      throw Error("descriptor write failed");
    }
  }
}

std::string random_generation() {
  std::array<unsigned char, kGenerationBytes> bytes{};
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count =
        getrandom(bytes.data() + offset, bytes.size() - offset, 0);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      throw Error("kernel random generation failed");
    }
  }
  static constexpr char hexadecimal[] = "0123456789abcdef";
  std::string output(bytes.size() * 2, '0');
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    output[index * 2] = hexadecimal[bytes[index] >> 4];
    output[index * 2 + 1] = hexadecimal[bytes[index] & 0x0f];
  }
  return output;
}

Observation read_barrier(int descriptor, const std::string& generation) {
  if (descriptor < 0) return {false, "descriptor-absent"};
  std::string input;
  std::array<char, 128> buffer{};
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kDeadlineMs);
  for (;;) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) return {false, "timeout"};
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now);
    pollfd candidate{descriptor, POLLIN | POLLHUP, 0};
    const int result = poll(&candidate, 1, static_cast<int>(remaining.count()) + 1);
    if (result < 0 && errno == EINTR) continue;
    if (result < 0) return {false, "poll-error"};
    if (result == 0) return {false, "timeout"};
    const ssize_t count = read(descriptor, buffer.data(), buffer.size());
    if (count > 0) {
      input.append(buffer.data(), static_cast<std::size_t>(count));
      if (input.size() > kMaximumInputBytes) return {false, "oversized"};
      continue;
    }
    if (count == 0) break;
    if (errno != EINTR) return {false, "read-error"};
  }
  const std::string expected = generation + "\n";
  if (input.empty()) return {false, "eof"};
  if (input == expected) return {true, "exact-release"};
  if (input.size() < expected.size()) return {false, "truncated"};
  if (input == expected + expected) return {false, "duplicate"};
  if (input.size() > expected.size()) return {false, "oversized"};
  return {false, "generation-mismatch"};
}

[[noreturn]] void child_main(int release_descriptor, int result_descriptor,
                             const std::string& generation, pid_t parent) {
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(125);
  const Observation observation = read_barrier(release_descriptor, generation);
  const std::string record =
      std::string(observation.opened ? "BARRIER_OPENED" : "BARRIER_CLOSED") +
      " reason=" + observation.reason + "\n";
  try {
    write_all(result_descriptor, record);
  } catch (...) {
    _exit(126);
  }
  _exit(0);
}

std::string payload_for(CaseKind kind, const std::string& generation) {
  const std::string exact = generation + "\n";
  switch (kind) {
    case CaseKind::Sabotage:
      return exact;
    case CaseKind::WrongGeneration:
      return std::string(generation.size(), '0') + "\n";
    case CaseKind::Truncated:
      return generation.substr(0, generation.size() - 1);
    case CaseKind::Oversized:
      return exact + "x";
    case CaseKind::Duplicate:
      return exact + exact;
    default:
      return "";
  }
}

Observation run_case(CaseKind kind, const std::string& generation) {
  int release_pipe[2];
  int result_pipe[2];
  if (pipe2(release_pipe, O_CLOEXEC) != 0 || pipe2(result_pipe, O_CLOEXEC) != 0) {
    throw Error("cannot create descriptor barrier");
  }
  UniqueFd release_read(release_pipe[0]);
  UniqueFd release_write(release_pipe[1]);
  UniqueFd result_read(result_pipe[0]);
  UniqueFd result_write(result_pipe[1]);
  const pid_t parent = getpid();
  const pid_t child = fork();
  if (child < 0) throw Error("cannot fork barrier child");
  if (child == 0) {
    release_write.reset();
    result_read.reset();
    const int descriptor =
        kind == CaseKind::DescriptorAbsent ? -1 : release_read.get();
    if (kind == CaseKind::DescriptorAbsent) release_read.reset();
    child_main(descriptor, result_write.get(), generation, parent);
  }
  release_read.reset();
  result_write.reset();
  if (kind != CaseKind::Timeout && kind != CaseKind::DescriptorAbsent) {
    const std::string payload = payload_for(kind, generation);
    if (!payload.empty()) write_all(release_write.get(), payload);
    release_write.reset();
  } else if (kind == CaseKind::DescriptorAbsent) {
    release_write.reset();
  }

  std::string result;
  std::array<char, 128> buffer{};
  for (;;) {
    const ssize_t count = read(result_read.get(), buffer.data(), buffer.size());
    if (count > 0) {
      result.append(buffer.data(), static_cast<std::size_t>(count));
      if (result.size() > 256) throw Error("barrier result exceeded limit");
    } else if (count == 0) {
      break;
    } else if (errno != EINTR) {
      throw Error("cannot read barrier result");
    }
  }
  release_write.reset();
  int status = 0;
  while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw Error("barrier child failed");
  }
  if (result.empty() || result.back() != '\n' ||
      result.find('\n') != result.size() - 1) {
    throw Error("barrier child returned malformed result");
  }
  result.pop_back();
  constexpr std::string_view opened = "BARRIER_OPENED reason=";
  constexpr std::string_view closed = "BARRIER_CLOSED reason=";
  if (result.rfind(opened, 0) == 0) {
    return {true, result.substr(opened.size())};
  }
  if (result.rfind(closed, 0) == 0) {
    return {false, result.substr(closed.size())};
  }
  throw Error("barrier child returned unknown result");
}

void require_case(CaseKind kind, const std::string& generation, bool opened,
                  std::string_view reason) {
  const Observation observation = run_case(kind, generation);
  if (observation.opened != opened || observation.reason != reason) {
    throw Error("barrier case diverged");
  }
}

int selftest() {
  const std::string generation = random_generation();
  require_case(CaseKind::Treatment, generation, false, "eof");
  require_case(CaseKind::Sabotage, generation, true, "exact-release");
  require_case(CaseKind::WrongGeneration, generation, false,
               "generation-mismatch");
  require_case(CaseKind::Truncated, generation, false, "truncated");
  require_case(CaseKind::Oversized, generation, false, "oversized");
  require_case(CaseKind::Duplicate, generation, false, "duplicate");
  require_case(CaseKind::Timeout, generation, false, "timeout");
  require_case(CaseKind::DescriptorAbsent, generation, false,
               "descriptor-absent");
  std::cout
      << "LOOM_PRINCIPAL_CELL_BARRIER_SELFTEST PASS language=C++20"
      << " role=MATERIAL_PARITY transitory=true semantic_authority=Sounio"
      << " treatment=CLOSED sabotage=OPEN causal_rule=descriptor-write"
      << " eof=closed timeout=closed wrong_generation=closed"
      << " truncated=closed oversized=closed duplicate=closed"
      << " descriptor_absent=closed open_sentinels=1 command_surface=false"
      << " material_grant=false material_execution=false launch_open=false"
      << " exec_attached=false\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--selftest") return selftest();
    throw Error("only --selftest is available");
  } catch (const std::exception& error) {
    std::cerr << "loom-principal-cell-barrier: REFUSE reason=" << error.what()
              << "\n";
    return 70;
  }
}

