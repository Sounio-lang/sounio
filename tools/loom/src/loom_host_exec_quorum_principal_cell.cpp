#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

#define main loom_principal_cell_barrier_v1_embedded_main
#include "loom_principal_cell_barrier.cpp"
#undef main

namespace {

constexpr std::string_view kHostArmRecord = "ARM\n";
constexpr int kHostArmDeadlineMs = 5000;

std::string host_required_environment(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    throw Error(std::string("missing host PrincipalCell environment: ") + name);
  }
  return value;
}

bool host_safe_unit_name(std::string_view unit) {
  if (unit.size() < 9 || unit.size() > 160 ||
      unit.substr(unit.size() - 8) != ".service") {
    return false;
  }
  return std::all_of(unit.begin(), unit.end(), [](unsigned char character) {
    return std::isalnum(character) || character == '.' || character == '_' ||
           character == '-' || character == '@';
  });
}

bool host_valid_generation(std::string_view generation) {
  return generation.size() == kGenerationBytes * 2 &&
         std::all_of(generation.begin(), generation.end(),
                     [](unsigned char character) {
                       return std::isdigit(character) ||
                              (character >= static_cast<unsigned char>('a') &&
                               character <= static_cast<unsigned char>('f'));
                     });
}

std::string host_read_file(const std::string& path, std::size_t limit) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw Error("host PrincipalCell cannot read " + path);
  std::string output;
  std::array<char, 4096> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      if (output.size() + static_cast<std::size_t>(count) > limit) {
        throw Error("host PrincipalCell file exceeded limit");
      }
      output.append(buffer.data(), static_cast<std::size_t>(count));
    }
  }
  if (!input.eof()) throw Error("host PrincipalCell file read failed");
  return output;
}

std::string host_status_field(const std::string& status, std::string_view name) {
  std::istringstream input(status);
  std::string line;
  const std::string prefix(name);
  while (std::getline(input, line)) {
    if (line.rfind(prefix, 0) == 0) return line.substr(prefix.size());
  }
  throw Error("host PrincipalCell status field absent: " + prefix);
}

std::uint64_t host_parse_u64(std::string_view value, int base = 10) {
  if (value.empty()) throw Error("host PrincipalCell integer is empty");
  std::size_t consumed = 0;
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(std::string(value), &consumed, base);
  } catch (...) {
    throw Error("host PrincipalCell integer is invalid");
  }
  if (consumed != value.size()) throw Error("host PrincipalCell integer is invalid");
  return static_cast<std::uint64_t>(parsed);
}

void host_require_descriptor_posture() {
  struct stat input {};
  struct stat output {};
  if (fstat(STDIN_FILENO, &input) != 0 || fstat(STDOUT_FILENO, &output) != 0) {
    throw Error("host PrincipalCell descriptors are unavailable");
  }
  if (!S_ISFIFO(input.st_mode) || !S_ISFIFO(output.st_mode) ||
      (input.st_dev == output.st_dev && input.st_ino == output.st_ino)) {
    throw Error("host PrincipalCell requires two distinct anonymous pipes");
  }
}

void host_require_process_posture(const std::string& unit) {
  if (getppid() != 1) throw Error("host PrincipalCell parent is not PID 1");
  uid_t real_uid = 0;
  uid_t effective_uid = 0;
  uid_t saved_uid = 0;
  gid_t real_gid = 0;
  gid_t effective_gid = 0;
  gid_t saved_gid = 0;
  if (getresuid(&real_uid, &effective_uid, &saved_uid) != 0 ||
      getresgid(&real_gid, &effective_gid, &saved_gid) != 0) {
    throw Error("host PrincipalCell credential vector is unavailable");
  }
  if (real_uid == 0 || real_gid == 0 || real_uid != effective_uid ||
      real_uid != saved_uid || real_gid != effective_gid ||
      real_gid != saved_gid) {
    throw Error("host PrincipalCell requires one non-root credential vector");
  }
  if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0 ||
      prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0 ||
      prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1) {
    throw Error("host PrincipalCell anti-injection posture is incomplete");
  }
  const std::string status = host_read_file("/proc/self/status", 256 * 1024);
  const std::string cgroup = host_read_file("/proc/self/cgroup", 256 * 1024);
  if (host_parse_u64(host_status_field(status, "NoNewPrivs:\t")) != 1 ||
      host_parse_u64(host_status_field(status, "CapEff:\t"), 16) != 0 ||
      host_parse_u64(host_status_field(status, "CapAmb:\t"), 16) != 0 ||
      cgroup.find(unit) == std::string::npos) {
    throw Error("host PrincipalCell kernel posture is incomplete");
  }
  host_require_descriptor_posture();
}

void host_consume_arm(int descriptor) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kHostArmDeadlineMs);
  for (const char expected : kHostArmRecord) {
    for (;;) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) throw Error("host PrincipalCell arm timeout");
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - now);
      pollfd candidate{descriptor, POLLIN | POLLHUP, 0};
      const int ready = poll(&candidate, 1, static_cast<int>(remaining.count()) + 1);
      if (ready < 0 && errno == EINTR) continue;
      if (ready <= 0) throw Error("host PrincipalCell arm timeout");
      char observed = '\0';
      const ssize_t count = read(descriptor, &observed, 1);
      if (count == 1 && observed == expected) break;
      if (count == 0) throw Error("host PrincipalCell arm EOF");
      if (count < 0 && errno == EINTR) continue;
      throw Error("host PrincipalCell arm mismatch");
    }
  }
}

int host_internal_main(int argc, char** argv) {
  if (argc != 2 ||
      std::string_view(argv[1]) != "--internal-host-exec-quorum") {
    throw Error("host PrincipalCell has no public material mode");
  }
  if (host_required_environment("SOUNIO_LOOM_HOST_EXEC_QUORUM_INTERNAL") != "1") {
    throw Error("host PrincipalCell internal marker is invalid");
  }
  const std::string unit =
      host_required_environment("SOUNIO_LOOM_HOST_EXEC_QUORUM_UNIT");
  const std::string generation =
      host_required_environment("SOUNIO_LOOM_HOST_EXEC_QUORUM_GENERATION");
  if (!host_safe_unit_name(unit) || !host_valid_generation(generation)) {
    throw Error("host PrincipalCell identity environment is invalid");
  }
  host_require_process_posture(unit);
  host_consume_arm(STDIN_FILENO);
  child_main(STDIN_FILENO, STDOUT_FILENO, generation, 1);
}

int host_selftest() {
  const std::string generation(64, 'a');
  if (!host_safe_unit_name("sounio-loom-host-exec-quorum-test.service") ||
      host_safe_unit_name("../test.service") || !host_valid_generation(generation) ||
      host_valid_generation(std::string(64, 'A'))) {
    throw Error("host PrincipalCell bounded parser selftest failed");
  }
  int descriptors[2];
  if (pipe2(descriptors, O_CLOEXEC) != 0) {
    throw Error("host PrincipalCell arm selftest pipe failed");
  }
  UniqueFd input(descriptors[0]);
  UniqueFd output(descriptors[1]);
  write_all(output.get(), std::string(kHostArmRecord) + generation + "\n");
  output.reset();
  host_consume_arm(input.get());
  const Observation observation = read_barrier(input.get(), generation);
  if (!observation.opened || observation.reason != "exact-release") {
    throw Error("host PrincipalCell arm boundary consumed release bytes");
  }
  std::cout << "LOOM_HOST_EXEC_QUORUM_PRINCIPAL_CELL_SELFTEST PASS"
            << " semantic_authority=Sounio material_language=C++20"
            << " material_role=MATERIAL_PARITY transitory=true"
            << " frozen_barrier_reused=true arm_exact=true arm_authority=false"
            << " read_ahead=false dynamic_user_required=true inherited_descriptor=true"
            << " material_grant=false material_execution=false launch_open=false\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
      return host_selftest();
    }
    return host_internal_main(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "loom-host-exec-quorum-principal-cell: REFUSE reason="
              << error.what() << "\n";
    return 70;
  }
}
