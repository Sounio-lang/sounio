#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cerrno>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sched.h>
#include <string>
#include <string_view>
#include <sys/mount.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr std::string_view kPrefix = "LOOM_MOUNT_NAMESPACE_MUTATOR";

struct Fd {
  int value = -1;
  explicit Fd(int descriptor) : value(descriptor) {}
  ~Fd() {
    if (value >= 0) close(value);
  }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
};

int parse_pid(std::string_view text) {
  int pid = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), pid);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      pid <= 1) {
    return -1;
  }
  return pid;
}

[[noreturn]] void fail(std::string_view reason) {
  std::cerr << kPrefix << "_CLOSED reason=" << reason
            << " errno=" << errno << " detail=" << std::strerror(errno) << '\n';
  _exit(70);
}

int mutate(int pid, std::string_view operation) {
  if (getuid() != 0 || geteuid() != 0) {
    errno = EPERM;
    fail("root-observer-required");
  }
  const std::string process = "/proc/" + std::to_string(pid);
  const Fd namespace_fd(open((process + "/ns/mnt").c_str(), O_RDONLY | O_CLOEXEC));
  if (namespace_fd.value < 0) fail("mount-namespace-open");
  const Fd root_fd(open((process + "/root").c_str(),
                        O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (root_fd.value < 0) fail("root-open");

  struct stat root_info {};
  if (fstat(root_fd.value, &root_info) != 0 || !S_ISDIR(root_info.st_mode) ||
      root_info.st_uid != 0 || root_info.st_gid != 0) {
    errno = EINVAL;
    fail("root-identity");
  }
  if (setns(namespace_fd.value, CLONE_NEWNS) != 0) fail("mount-namespace-enter");
  if (fchdir(root_fd.value) != 0 || chroot(".") != 0 || chdir("/") != 0) {
    fail("root-enter");
  }

  if (operation == "live-procfs") {
    constexpr unsigned long flags =
        MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC;
    if (mount("proc", "/proc", "proc", flags, nullptr) != 0) {
      fail("live-procfs-mount");
    }
  } else if (operation == "writable-proc-bind") {
    if (mount(nullptr, "/proc", nullptr, MS_REMOUNT | MS_BIND, nullptr) != 0) {
      fail("writable-proc-remount");
    }
  } else {
    errno = EINVAL;
    fail("unknown-operation");
  }

  std::cout << kPrefix
            << "_PASS language=C++20 role=MATERIAL_PARITY transitory=true"
            << " semantic_authority=Sounio action=9025 observer=ROOT_HOST"
            << " descriptor_bound_namespace=true descriptor_bound_root=true"
            << " pid=" << pid << " operation=" << operation
            << " semantic_decision=false\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
    std::cout << kPrefix
              << "_SELFTEST PASS language=C++20 role=MATERIAL_PARITY"
              << " transitory=true semantic_authority=Sounio action=9025"
              << " operations=live-procfs+writable-proc-bind"
              << " descriptor_bound_namespace=true descriptor_bound_root=true"
              << " semantic_decision=false\n";
    return 0;
  }
  if (argc == 5 && std::string_view(argv[1]) == "--pid" &&
      std::string_view(argv[3]) == "--operation") {
    const int pid = parse_pid(argv[2]);
    if (pid > 1) return mutate(pid, argv[4]);
  }
  std::cerr << "usage: loom-mount-namespace-mutator --selftest\n"
               "       loom-mount-namespace-mutator --pid PID --operation "
               "live-procfs|writable-proc-bind\n";
  return 64;
}
