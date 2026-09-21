#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>

#define main loom_principal_cell_barrier_v1_embedded_main
#include "loom_principal_cell_barrier.cpp"
#undef main

namespace {

constexpr int kIntegratedReleaseDescriptor = 3;
constexpr int kIntegratedResultDescriptor = 4;

std::string required_integrated_environment(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    throw Error(std::string("missing integrated environment: ") + name);
  }
  return value;
}

pid_t integrated_parent_pid() {
  const std::string value =
      required_integrated_environment("SOUNIO_LOOM_PRINCIPAL_CELL_PARENT_PID");
  std::size_t consumed = 0;
  long parsed = 0;
  try {
    parsed = std::stol(value, &consumed, 10);
  } catch (...) {
    throw Error("integrated parent PID is invalid");
  }
  if (consumed != value.size() || parsed <= 0 || parsed > INT_MAX) {
    throw Error("integrated parent PID is invalid");
  }
  return static_cast<pid_t>(parsed);
}

std::string integrated_generation() {
  const std::string value =
      required_integrated_environment("SOUNIO_LOOM_PRINCIPAL_CELL_GENERATION");
  if (value.size() != kGenerationBytes * 2 ||
      !std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isdigit(character) ||
               (character >= static_cast<unsigned char>('a') &&
                character <= static_cast<unsigned char>('f'));
      })) {
    throw Error("integrated generation is invalid");
  }
  return value;
}

int integrated_main(int argc, char** argv) {
  if (argc != 2 || std::string_view(argv[1]) != "--internal-principal-cell") {
    throw Error("integrated PrincipalCell has no public mode");
  }
  if (required_integrated_environment("SOUNIO_LOOM_PRINCIPAL_CELL_INTERNAL") !=
      "1") {
    throw Error("integrated PrincipalCell marker is invalid");
  }
  const pid_t parent = integrated_parent_pid();
  if (getppid() != parent) throw Error("integrated PrincipalCell parent mismatch");
  const std::string generation = integrated_generation();
  child_main(kIntegratedReleaseDescriptor, kIntegratedResultDescriptor,
             generation, parent);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return integrated_main(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "loom-principal-cell-barrier-integrated: REFUSE reason="
              << error.what() << "\n";
    return 70;
  }
}
