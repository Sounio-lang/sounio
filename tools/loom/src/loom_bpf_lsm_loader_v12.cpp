#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
struct bpf_object;
struct bpf_program;
struct bpf_link;

bpf_object* bpf_object__open_file(const char* path, const void* options);
long libbpf_get_error(const void* pointer);
int bpf_object__load(bpf_object* object);
bpf_program* bpf_object__next_program(const bpf_object* object,
                                      bpf_program* previous);
const char* bpf_program__name(const bpf_program* program);
bpf_link* bpf_program__attach_lsm(const bpf_program* program);
int bpf_link__pin(bpf_link* link, const char* path);
int bpf_link__destroy(bpf_link* link);
void bpf_object__close(bpf_object* object);
int libbpf_strerror(int error, char* buffer, std::size_t size);
}

namespace {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

constexpr std::array<std::string_view, 3> kExpectedPrograms{
    "loom_v12_task_kill", "loom_v12_ptrace_access_check",
    "loom_v12_task_prlimit"};

std::string libbpf_error(int error) {
  std::array<char, 256> buffer{};
  if (libbpf_strerror(error, buffer.data(), buffer.size()) == 0) {
    return std::string(buffer.data());
  }
  return "libbpf error " + std::to_string(error);
}

bool expected_program(std::string_view name) {
  for (const auto expected : kExpectedPrograms) {
    if (name == expected) return true;
  }
  return false;
}

bool safe_pin_directory(std::string_view path) {
  constexpr std::string_view prefix = "/sys/fs/bpf/loom-v12";
  return path == prefix && path.find("..") == std::string_view::npos;
}

void load_and_pin(const char* object_path, const char* pin_directory) {
  if (!safe_pin_directory(pin_directory)) throw Error("unsafe pin directory");
  struct stat object_metadata {};
  struct stat pin_metadata {};
  if (lstat(object_path, &object_metadata) != 0 ||
      !S_ISREG(object_metadata.st_mode)) {
    throw Error("BPF object is absent or not regular");
  }
  if (lstat(pin_directory, &pin_metadata) != 0 ||
      !S_ISDIR(pin_metadata.st_mode)) {
    throw Error("pin directory is absent or not a directory");
  }

  bpf_object* object = bpf_object__open_file(object_path, nullptr);
  const long open_error = libbpf_get_error(object);
  if (open_error != 0) {
    throw Error("open refused: " + libbpf_error(static_cast<int>(open_error)));
  }
  std::vector<bpf_link*> links;
  std::vector<std::string> pins;
  try {
    const int load_status = bpf_object__load(object);
    if (load_status != 0) {
      throw Error("load refused: " + libbpf_error(load_status));
    }
    bpf_program* program = nullptr;
    while ((program = bpf_object__next_program(object, program)) != nullptr) {
      const char* raw_name = bpf_program__name(program);
      if (!raw_name || !expected_program(raw_name)) {
        throw Error("unexpected BPF program in frozen object");
      }
      bpf_link* link = bpf_program__attach_lsm(program);
      const long attach_error = libbpf_get_error(link);
      if (attach_error != 0) {
        throw Error(std::string("attach refused for ") + raw_name + ": " +
                    libbpf_error(static_cast<int>(attach_error)));
      }
      links.push_back(link);
      pins.push_back(std::string(pin_directory) + "/" + raw_name);
      const int pin_status = bpf_link__pin(link, pins.back().c_str());
      if (pin_status != 0) {
        throw Error(std::string("pin refused for ") + raw_name + ": " +
                    libbpf_error(pin_status));
      }
    }
    if (links.size() != kExpectedPrograms.size()) {
      throw Error("frozen BPF program count is not three");
    }
    for (bpf_link*& link : links) {
      const int destroy_status = bpf_link__destroy(link);
      link = nullptr;
      if (destroy_status != 0) {
        throw Error("loader descriptor close failed: " +
                    libbpf_error(destroy_status));
      }
    }
    bpf_object__close(object);
    object = nullptr;
    std::cout << "LOOM_BPF_LSM_LOADER_V12 PASS programs=3 links_pinned=3 "
                 "loader_link_fds_closed=true semantic_authority=Sounio "
                 "material_role=C++20_TRANSITORY action=9025 "
                 "same_uid_peer_isolation=false claim_ready=false\n";
  } catch (...) {
    for (const auto& pin : pins) unlink(pin.c_str());
    for (bpf_link* link : links) {
      if (link) bpf_link__destroy(link);
    }
    if (object) bpf_object__close(object);
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
      if (!expected_program("loom_v12_task_kill") ||
          expected_program("loom_v12_unknown") ||
          !safe_pin_directory("/sys/fs/bpf/loom-v12") ||
          safe_pin_directory("/sys/fs/bpf/loom-v12/../escape")) {
        throw Error("loader bounded-input selftest failed");
      }
      std::cout << "LOOM_BPF_LSM_LOADER_V12_SELFTEST PASS expected_programs=3 "
                   "pin_root=/sys/fs/bpf/loom-v12 language=C++20 "
                   "role=MATERIAL_BOOTSTRAP transitory=true python_executed=false "
                   "rust_executed=false same_uid_peer_isolation=false "
                   "claim_ready=false\n";
      return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "--load") {
      load_and_pin(argv[2], argv[3]);
      return 0;
    }
    return 64;
  } catch (const std::exception& error) {
    std::cerr << "LOOM_BPF_LSM_LOADER_V12_REFUSE reason=" << error.what()
              << " same_uid_peer_isolation=false action_9025=DENY451 "
                 "claim_ready=false\n";
    return 70;
  }
}
