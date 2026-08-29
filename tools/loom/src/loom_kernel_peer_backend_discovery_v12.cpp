#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <signal.h>
#include <sys/fsuid.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

volatile sig_atomic_t term_seen = 0;

void on_term(int) { term_seen = 1; }

std::uint64_t parse_u64(std::string_view text, std::string_view label) {
  if (text.empty()) throw Error(std::string(label) + " is empty");
  std::uint64_t value = 0;
  for (const unsigned char character : text) {
    if (character < '0' || character > '9') {
      throw Error(std::string(label) + " is not decimal");
    }
    const std::uint64_t digit = character - '0';
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
      throw Error(std::string(label) + " overflows");
    }
    value = value * 10 + digit;
  }
  return value;
}

pid_t parse_pid(std::string_view text) {
  const std::uint64_t value = parse_u64(text, "pid");
  if (value <= 1 || value > static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max())) {
    throw Error("pid is outside the admissible range");
  }
  return static_cast<pid_t>(value);
}

std::string uid_vector() {
  uid_t real = 0;
  uid_t effective = 0;
  uid_t saved = 0;
  if (getresuid(&real, &effective, &saved) != 0) {
    throw Error(std::string("getresuid failed: ") + std::strerror(errno));
  }
  const uid_t filesystem = setfsuid(static_cast<uid_t>(-1));
  return std::to_string(real) + "/" + std::to_string(effective) + "/" +
         std::to_string(saved) + "/" + std::to_string(filesystem);
}

std::string gid_vector() {
  gid_t real = 0;
  gid_t effective = 0;
  gid_t saved = 0;
  if (getresgid(&real, &effective, &saved) != 0) {
    throw Error(std::string("getresgid failed: ") + std::strerror(errno));
  }
  const gid_t filesystem = setfsgid(static_cast<gid_t>(-1));
  return std::to_string(real) + "/" + std::to_string(effective) + "/" +
         std::to_string(saved) + "/" + std::to_string(filesystem);
}

void write_file(const std::string& path, const std::string& value) {
  const int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (descriptor < 0) {
    throw Error("cannot create " + path + ": " + std::strerror(errno));
  }
  std::size_t offset = 0;
  while (offset < value.size()) {
    const ssize_t count = write(descriptor, value.data() + offset, value.size() - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      const int saved_errno = errno;
      close(descriptor);
      throw Error("cannot write " + path + ": " + std::strerror(saved_errno));
    }
  }
  if (fsync(descriptor) != 0) {
    const int saved_errno = errno;
    close(descriptor);
    throw Error("cannot sync " + path + ": " + std::strerror(saved_errno));
  }
  if (close(descriptor) != 0) {
    throw Error("cannot close " + path + ": " + std::strerror(errno));
  }
}

rlimit read_nofile_limit() {
  rlimit limit{};
  if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
    throw Error(std::string("getrlimit failed: ") + std::strerror(errno));
  }
  return limit;
}

int run_target(const std::string& ready_path, const std::string& control_path,
               const std::string& report_path) {
  const rlimit initial{1024, 2048};
  if (setrlimit(RLIMIT_NOFILE, &initial) != 0) {
    throw Error(std::string("setrlimit failed: ") + std::strerror(errno));
  }
  struct sigaction action {};
  action.sa_handler = on_term;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGTERM, &action, nullptr) != 0) {
    throw Error(std::string("sigaction failed: ") + std::strerror(errno));
  }
  const rlimit before = read_nofile_limit();
  write_file(ready_path,
             "TARGET_READY pid=" + std::to_string(getpid()) + " uid_vector=" +
                 uid_vector() + " gid_vector=" + gid_vector() +
                 " rlimit_soft=" + std::to_string(before.rlim_cur) +
                 " rlimit_hard=" + std::to_string(before.rlim_max) + "\n");

  bool released = false;
  for (int attempt = 0; attempt < 2000; ++attempt) {
    if (access(control_path.c_str(), F_OK) == 0) {
      released = true;
      break;
    }
    if (errno != ENOENT && errno != EACCES) {
      throw Error("control observation failed: " + std::string(std::strerror(errno)));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  if (!released) throw Error("control observation timed out");

  const rlimit after = read_nofile_limit();
  write_file(report_path,
             "TARGET_STATE pid=" + std::to_string(getpid()) + " uid_vector=" +
                 uid_vector() + " gid_vector=" + gid_vector() +
                 " signal_term_seen=" + std::to_string(term_seen) +
                 " rlimit_soft=" + std::to_string(after.rlim_cur) +
                 " rlimit_hard=" + std::to_string(after.rlim_max) + "\n");
  return 0;
}

int run_signal_attack(pid_t target) {
  errno = 0;
  const int result = kill(target, SIGTERM);
  const int error = result == 0 ? 0 : errno;
  std::cout << "ATTEMPT operation=kill_SIGTERM pid=" << getpid()
            << " target_pid=" << target << " uid_vector=" << uid_vector()
            << " gid_vector=" << gid_vector() << " syscall_rc=" << result
            << " syscall_errno=" << error << "\n";
  return 0;
}

int prlimit_native(pid_t target, const rlimit* replacement, rlimit* previous) {
#ifdef SYS_prlimit64
  return static_cast<int>(syscall(SYS_prlimit64, target, RLIMIT_NOFILE, replacement,
                                  previous));
#else
  (void)target;
  (void)replacement;
  (void)previous;
  errno = ENOSYS;
  return -1;
#endif
}

int run_prlimit_attack(pid_t target) {
  const rlimit replacement{768, 2048};
  rlimit previous{};
  errno = 0;
  const int result = prlimit_native(target, &replacement, &previous);
  const int error = result == 0 ? 0 : errno;
  rlimit observed{};
  errno = 0;
  const int observe_result = prlimit_native(target, nullptr, &observed);
  const int observe_error = observe_result == 0 ? 0 : errno;
  std::cout << "ATTEMPT operation=prlimit64 pid=" << getpid()
            << " target_pid=" << target << " uid_vector=" << uid_vector()
            << " gid_vector=" << gid_vector() << " syscall_rc=" << result
            << " syscall_errno=" << error << " prior_soft=" << previous.rlim_cur
            << " prior_hard=" << previous.rlim_max
            << " observe_rc=" << observe_result
            << " observe_errno=" << observe_error
            << " observed_soft=" << observed.rlim_cur
            << " observed_hard=" << observed.rlim_max << "\n";
  return 0;
}

int usage() {
  std::cerr << "usage: loom-kernel-peer-backend-discovery-v12 "
               "--selftest | --target READY CONTROL REPORT | "
               "--attack-signal PID | --attack-prlimit PID\n";
  return 64;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
      if (parse_pid("2") != 2 || parse_u64("768", "fixture") != 768) {
        throw Error("bounded parser selftest failed");
      }
      std::cout << "LOOM_KERNEL_PEER_BACKEND_DISCOVERY_V12_SELFTEST PASS "
                   "language=C++20 role=MATERIAL_DISCOVERY transitory=true "
                   "semantic_authority=Sounio action=9025 operations=kill_SIGTERM+prlimit64 "
                   "semantic_results_encoded=false python_executed=false rust_executed=false\n";
      return 0;
    }
    if (argc == 5 && std::string_view(argv[1]) == "--target") {
      return run_target(argv[2], argv[3], argv[4]);
    }
    if (argc == 3 && std::string_view(argv[1]) == "--attack-signal") {
      return run_signal_attack(parse_pid(argv[2]));
    }
    if (argc == 3 && std::string_view(argv[1]) == "--attack-prlimit") {
      return run_prlimit_attack(parse_pid(argv[2]));
    }
    return usage();
  } catch (const std::exception& error) {
    std::cerr << "LOOM_KERNEL_PEER_BACKEND_DISCOVERY_V12_ERROR " << error.what()
              << "\n";
    return 70;
  }
}
