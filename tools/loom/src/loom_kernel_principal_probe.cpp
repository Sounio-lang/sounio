#include <openssl/sha.h>

#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <optional>
#include <poll.h>
#include <pwd.h>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr long kCgroup2SuperMagic = 0x63677270;
constexpr std::size_t kMaxCommandOutput = 1024 * 1024;
constexpr auto kCommandTimeout = std::chrono::seconds(5);

struct CommandResult {
  int exit_code;
  std::string output;
};

struct SubidRange {
  std::uint64_t start;
  std::uint64_t count;
  std::string record;
};

std::string read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::ostringstream output;
  output << input.rdbuf();
  return output.str();
}

std::string sha256(const std::string& value) {
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

CommandResult run_command(const std::vector<std::string>& arguments) {
  int descriptors[2];
  if (pipe(descriptors) != 0) {
    throw std::runtime_error("pipe failed");
  }
  const pid_t child = fork();
  if (child < 0) {
    close(descriptors[0]);
    close(descriptors[1]);
    throw std::runtime_error("fork failed");
  }
  if (child == 0) {
    close(descriptors[0]);
    dup2(descriptors[1], STDOUT_FILENO);
    dup2(descriptors[1], STDERR_FILENO);
    if (descriptors[1] != STDOUT_FILENO && descriptors[1] != STDERR_FILENO) {
      close(descriptors[1]);
    }
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (const auto& argument : arguments) {
      argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);
    execv(argv[0], argv.data());
    _exit(127);
  }
  close(descriptors[1]);
  const int flags = fcntl(descriptors[0], F_GETFL, 0);
  if (flags < 0 || fcntl(descriptors[0], F_SETFL, flags | O_NONBLOCK) != 0) {
    close(descriptors[0]);
    kill(child, SIGKILL);
    waitpid(child, nullptr, 0);
    throw std::runtime_error("failed to make command pipe nonblocking");
  }
  std::string output;
  char buffer[4096];
  int status = 0;
  bool exited = false;
  bool pipe_open = true;
  const auto deadline = std::chrono::steady_clock::now() + kCommandTimeout;
  for (;;) {
    if (!exited) {
      const pid_t waited = waitpid(child, &status, WNOHANG);
      if (waited == child) {
        exited = true;
      } else if (waited < 0 && errno != EINTR) {
        close(descriptors[0]);
        kill(child, SIGKILL);
        waitpid(child, nullptr, 0);
        throw std::runtime_error("waitpid failed");
      }
    }
    while (pipe_open) {
      const ssize_t count = read(descriptors[0], buffer, sizeof(buffer));
      if (count > 0) {
        output.append(buffer, static_cast<std::size_t>(count));
        if (output.size() > kMaxCommandOutput) {
          close(descriptors[0]);
          kill(child, SIGKILL);
          waitpid(child, nullptr, 0);
          throw std::runtime_error("command output exceeded limit");
        }
      } else if (count == 0) {
        pipe_open = false;
        break;
      } else if (errno == EINTR) {
        continue;
      } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      } else {
        close(descriptors[0]);
        kill(child, SIGKILL);
        waitpid(child, nullptr, 0);
        throw std::runtime_error("read failed");
      }
    }
    if (exited && !pipe_open) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      close(descriptors[0]);
      kill(child, SIGKILL);
      while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
      }
      return {124, output + "command-timeout\n"};
    }
    pollfd poll_descriptor{descriptors[0], POLLIN | POLLHUP, 0};
    poll(&poll_descriptor, 1, 25);
  }
  close(descriptors[0]);
  while (!exited && waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) throw std::runtime_error("waitpid failed");
  }
  if (WIFEXITED(status)) {
    return {WEXITSTATUS(status), output};
  }
  if (WIFSIGNALED(status)) {
    return {128 + WTERMSIG(status), output};
  }
  return {255, output};
}

bool executable_setuid_root(const std::string& path) {
  struct stat info {};
  return stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode) &&
         info.st_uid == 0 && (info.st_mode & S_ISUID) != 0 &&
         access(path.c_str(), X_OK) == 0;
}

std::optional<SubidRange> parse_subid(const std::string& path,
                                      const std::string& user) {
  std::istringstream input(read_file(path));
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t first = line.find(':');
    const std::size_t second =
        first == std::string::npos ? std::string::npos : line.find(':', first + 1);
    if (first == std::string::npos || second == std::string::npos ||
        line.substr(0, first) != user) {
      continue;
    }
    try {
      const std::uint64_t start = std::stoull(line.substr(first + 1, second - first - 1));
      const std::uint64_t count = std::stoull(line.substr(second + 1));
      if (count > 0) {
        return SubidRange{start, count, line};
      }
    } catch (...) {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<std::uint64_t> field_value(const std::string& output,
                                         const std::string& key) {
  const std::string marker = key + "=";
  const std::size_t start = output.find(marker);
  if (start == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t value_start = start + marker.size();
  const std::size_t end = output.find_first_of(" \n", value_start);
  try {
    return std::stoull(output.substr(value_start, end - value_start));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<std::uint64_t> map_to_outer(const std::string& path,
                                          std::uint64_t inner) {
  std::istringstream input(read_file(path));
  std::uint64_t inside = 0;
  std::uint64_t outside = 0;
  std::uint64_t count = 0;
  while (input >> inside >> outside >> count) {
    if (inner >= inside && inner - inside < count) {
      return outside + (inner - inside);
    }
  }
  return std::nullopt;
}

std::string executable_path() {
  std::vector<char> buffer(4096);
  const ssize_t count = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (count < 0) {
    throw std::runtime_error("readlink /proc/self/exe failed");
  }
  return std::string(buffer.data(), static_cast<std::size_t>(count));
}

int inside_report(const std::string& mode) {
  if (mode == "subordinate") {
    if (setresgid(1, 1, 1) != 0 || setresuid(1, 1, 1) != 0) {
      std::cout << "inside_transition=refused errno=" << errno << "\n";
      return 73;
    }
  }
  const auto host_uid = map_to_outer("/proc/self/uid_map", geteuid());
  const auto host_gid = map_to_outer("/proc/self/gid_map", getegid());
  if (!host_uid || !host_gid) {
    std::cout << "inside_mapping=unresolved\n";
    return 74;
  }
  const std::string setgroups = read_file("/proc/self/setgroups");
  std::cout << "inside_mapping=resolved inner_uid=" << geteuid()
            << " inner_gid=" << getegid() << " host_uid=" << *host_uid
            << " host_gid=" << *host_gid
            << " setgroups_denied=" << (setgroups.find("deny") != std::string::npos ? 1 : 0)
            << " uid_map_sha256=" << sha256(read_file("/proc/self/uid_map"))
            << " gid_map_sha256=" << sha256(read_file("/proc/self/gid_map")) << "\n";
  return 0;
}

bool cgroup_v2_present() {
  struct statfs info {};
  return statfs("/sys/fs/cgroup", &info) == 0 && info.f_type == kCgroup2SuperMagic;
}

int run_probe(bool simulate_helper_exit_only) {
#if !defined(__linux__) || !defined(__x86_64__)
  constexpr int linux = 0;
  constexpr int x86_64 = 0;
#else
  constexpr int linux = 1;
  constexpr int x86_64 = 1;
#endif
  const uid_t outer_uid = getuid();
  const gid_t outer_gid = getgid();
  const passwd* account = getpwuid(outer_uid);
  if (account == nullptr) {
    throw std::runtime_error("getpwuid failed");
  }
  const std::string user = account->pw_name;
  const auto subuid = parse_subid("/etc/subuid", user);
  const auto subgid = parse_subid("/etc/subgid", user);
  const bool ranges_ready = subuid && subgid && subuid->count >= 2 && subgid->count >= 2;
  const bool ranges_disjoint = ranges_ready &&
      !(outer_uid >= subuid->start && outer_uid < subuid->start + subuid->count) &&
      !(outer_gid >= subgid->start && outer_gid < subgid->start + subgid->count);
  const bool helpers = executable_setuid_root("/usr/bin/newuidmap") &&
                       executable_setuid_root("/usr/bin/newgidmap");
  const std::string self = executable_path();

  const CommandResult current = run_command(
      {"/usr/bin/unshare", "--user", "--map-current-user", self,
       "--inside-report", "current"});
  const bool user_namespace = current.exit_code == 0 &&
                              current.output.find("inside_mapping=resolved") != std::string::npos;
  const auto current_host_uid = field_value(current.output, "host_uid");
  const bool current_distinct = current_host_uid && *current_host_uid != outer_uid;

  CommandResult subordinate;
  if (simulate_helper_exit_only) {
    subordinate = run_command({"/usr/bin/true"});
  } else {
    subordinate = run_command(
        {"/usr/bin/unshare", "--user", "--map-auto", "--map-root-user", self,
         "--inside-report", "subordinate"});
  }
  const auto subordinate_host_uid = field_value(subordinate.output, "host_uid");
  const auto subordinate_host_gid = field_value(subordinate.output, "host_gid");
  const auto setgroups_denied = field_value(subordinate.output, "setgroups_denied");
  const bool uid_map_exact = ranges_ready && subordinate_host_uid &&
                             *subordinate_host_uid == subuid->start;
  const bool gid_map_exact = ranges_ready && subordinate_host_gid &&
                             *subordinate_host_gid == subgid->start;
  const bool mapping_exact = subordinate.exit_code == 0 && uid_map_exact &&
                             gid_map_exact && setgroups_denied && *setgroups_denied == 1;

  const CommandResult pid_namespace = run_command(
      {"/usr/bin/unshare", "--user", "--map-current-user", "--pid", "--fork",
       "/usr/bin/true"});
  const CommandResult mount_namespace = run_command(
      {"/usr/bin/unshare", "--user", "--map-current-user", "--mount",
       "/usr/bin/true"});
  const bool cgroup_v2 = cgroup_v2_present();
  const bool privilege_regain = access("/usr/bin/sudo", X_OK) == 0 &&
      run_command({"/usr/bin/sudo", "-n", "/usr/bin/true"}).exit_code == 0;

  const std::uint64_t principal_uid = mapping_exact ? *subordinate_host_uid : outer_uid;
  const std::uint64_t sibling_uid = ranges_ready ? subuid->start + 1 : outer_uid;
  const std::uint64_t principal_gid = mapping_exact ? *subordinate_host_gid : outer_gid;
  const std::uint64_t sibling_gid = ranges_ready ? subgid->start + 1 : outer_gid;
  const int material_isolation = mapping_exact && !privilege_regain ? 1 : 0;

  std::ostringstream receipt;
  receipt << "LOOM_KERNEL_PRINCIPAL_MATERIAL schema=loom-kernel-principal-material-v1"
          << " outer_uid=" << outer_uid << " outer_gid=" << outer_gid
          << " subuid_start=" << (subuid ? subuid->start : 0)
          << " subuid_count=" << (subuid ? subuid->count : 0)
          << " subgid_start=" << (subgid ? subgid->start : 0)
          << " subgid_count=" << (subgid ? subgid->count : 0)
          << " subid_record_sha256="
          << sha256((subuid ? subuid->record : "-") + "\n" +
                    (subgid ? subgid->record : "-") + "\n")
          << " ranges_disjoint=" << (ranges_disjoint ? 1 : 0)
          << " helpers_setuid_root=" << (helpers ? 1 : 0)
          << " user_namespace=" << (user_namespace ? 1 : 0)
          << " current_map_exit=" << current.exit_code
          << " current_host_uid=" << (current_host_uid ? *current_host_uid : 0)
          << " current_principal_distinct=" << (current_distinct ? 1 : 0)
          << " subordinate_map_exit=" << subordinate.exit_code
          << " subordinate_detail_sha256=" << sha256(subordinate.output)
          << " uid_map_exact=" << (uid_map_exact ? 1 : 0)
          << " gid_map_exact=" << (gid_map_exact ? 1 : 0)
          << " setgroups_denied_first="
          << (setgroups_denied && *setgroups_denied == 1 ? 1 : 0)
          << " pid_namespace=" << (pid_namespace.exit_code == 0 ? 1 : 0)
          << " mount_namespace=" << (mount_namespace.exit_code == 0 ? 1 : 0)
          << " cgroup_v2=" << (cgroup_v2 ? 1 : 0)
          << " outer_privilege_regain=" << (privilege_regain ? 1 : 0)
          << " sabotage=" << (simulate_helper_exit_only ? "helper-exit-only" : "none")
          << " mapping_materialized=" << (mapping_exact ? 1 : 0)
          << " material_isolation=" << material_isolation;
  const std::string receipt_text = receipt.str();
  std::cout << receipt_text << "\n";
  std::cout << "LOOM_KERNEL_PRINCIPAL_RECEIPT_SHA256 " << sha256(receipt_text + "\n")
            << "\n";

  const std::string one = "1 1 1 1 1 1 1 1";
  std::ostringstream frame;
  frame << "9026 3 1 " << linux << ' ' << x86_64 << ' '
        << (user_namespace ? 1 : 0) << ' '
        << (pid_namespace.exit_code == 0 ? 1 : 0) << ' '
        << (mount_namespace.exit_code == 0 ? 1 : 0) << ' '
        << (cgroup_v2 ? 1 : 0) << ' '
        << (mapping_exact ? 1 : 0) << ' ' << (uid_map_exact ? 1 : 0) << ' '
        << (gid_map_exact ? 1 : 0) << ' '
        << (setgroups_denied && *setgroups_denied == 1 ? 1 : 0) << ' '
        << (ranges_disjoint ? 1 : 0) << ' ' << outer_uid << ' ' << principal_uid
        << ' ' << sibling_uid << ' ' << outer_gid << ' ' << principal_gid << ' '
        << sibling_gid << ' '
        // Peer and injection attacks are intentionally not promoted by this
        // substrate probe. Later material stages must measure each as a fact.
        << "0 0 0 0 0 0 0 0 "
        << "0 0 0 0 0 0 "
        // Existing custody is not enough to bypass missing principal proof.
        << "1 1 1 1 1 0 5 0";
  for (int index = 0; index < 12; ++index) {
    frame << ' ' << one;
  }
  std::cout << "SOUNIO_FRAME " << frame.str() << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--inside-report") {
      return inside_report(argv[2]);
    }
    bool simulate_helper_exit_only = false;
    if (argc == 2 && std::string(argv[1]) == "--simulate-helper-exit-only") {
      simulate_helper_exit_only = true;
    } else if (argc != 1) {
      std::cerr << "usage: loom-kernel-principal-probe [--simulate-helper-exit-only]\n";
      return 64;
    }
    return run_probe(simulate_helper_exit_only);
  } catch (const std::exception& error) {
    std::cerr << "loom-kernel-principal-probe: " << error.what() << "\n";
    return 70;
  }
}
