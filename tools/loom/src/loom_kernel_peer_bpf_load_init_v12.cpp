#include <fcntl.h>
#include <linux/bpf.h>
#include <linux/reboot.h>
#include <sys/mount.h>
#include <sys/reboot.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

constexpr std::string_view kPinDirectory = "/sys/fs/bpf/loom-v12";
constexpr std::array<std::string_view, 3> kPinNames{
    "loom_v12_task_kill", "loom_v12_ptrace_access_check",
    "loom_v12_task_prlimit"};

void ensure_directory(const char* path, mode_t mode = 0755) {
  if (mkdir(path, mode) != 0 && errno != EEXIST) {
    throw Error(std::string("mkdir failed for ") + path + ": " +
                std::strerror(errno));
  }
}

void mount_required(const char* source, const char* target, const char* type,
                    unsigned long flags = 0) {
  if (mount(source, target, type, flags, nullptr) != 0) {
    throw Error(std::string("mount failed for ") + target + ": " +
                std::strerror(errno));
  }
}

std::string read_file(const char* path, std::size_t maximum = 64 * 1024) {
  const int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    throw Error(std::string("open failed for ") + path + ": " +
                std::strerror(errno));
  }
  std::string result;
  std::array<char, 4096> buffer{};
  for (;;) {
    const ssize_t count = read(descriptor, buffer.data(), buffer.size());
    if (count > 0) {
      result.append(buffer.data(), static_cast<std::size_t>(count));
      if (result.size() > maximum) {
        close(descriptor);
        throw Error(std::string("file exceeds bound: ") + path);
      }
    } else if (count == 0) {
      close(descriptor);
      while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
      }
      return result;
    } else if (errno != EINTR) {
      const int saved_errno = errno;
      close(descriptor);
      throw Error(std::string("read failed for ") + path + ": " +
                  std::strerror(saved_errno));
    }
  }
}

bool comma_token(const std::string& list, std::string_view token) {
  const std::string framed = "," + list + ",";
  return framed.find("," + std::string(token) + ",") != std::string::npos;
}

bool regular_readable(const char* path) {
  struct stat metadata {};
  return stat(path, &metadata) == 0 && S_ISREG(metadata.st_mode) &&
         access(path, R_OK) == 0;
}

int bpf_call(enum bpf_cmd command, union bpf_attr* attributes) {
  return static_cast<int>(
      syscall(__NR_bpf, command, attributes, sizeof(*attributes)));
}

std::uint32_t pinned_link_id(const std::string& path) {
  union bpf_attr get_attributes {};
  get_attributes.pathname =
      static_cast<__u64>(reinterpret_cast<std::uintptr_t>(path.c_str()));
  const int descriptor = bpf_call(BPF_OBJ_GET, &get_attributes);
  if (descriptor < 0) {
    throw Error("BPF_OBJ_GET failed for " + path + ": " + std::strerror(errno));
  }
  struct bpf_link_info info {};
  union bpf_attr info_attributes {};
  info_attributes.info.bpf_fd = static_cast<__u32>(descriptor);
  info_attributes.info.info_len = sizeof(info);
  info_attributes.info.info =
      static_cast<__u64>(reinterpret_cast<std::uintptr_t>(&info));
  if (bpf_call(BPF_OBJ_GET_INFO_BY_FD, &info_attributes) != 0) {
    const int saved_errno = errno;
    close(descriptor);
    throw Error("BPF_OBJ_GET_INFO_BY_FD failed for " + path + ": " +
                std::strerror(saved_errno));
  }
  close(descriptor);
  if (info.id == 0) throw Error("pinned BPF link has zero identity");
  return info.id;
}

void require_link_extinct(std::uint32_t id) {
  union bpf_attr attributes {};
  attributes.link_id = id;
  const int descriptor = bpf_call(BPF_LINK_GET_FD_BY_ID, &attributes);
  if (descriptor >= 0) {
    close(descriptor);
    throw Error("unpinned BPF link remains addressable by id=" +
                std::to_string(id));
  }
  if (errno != ENOENT) {
    throw Error("BPF link extinction query failed closed for id=" +
                std::to_string(id) + ": " + std::strerror(errno));
  }
}

void run_loader() {
  const pid_t child = fork();
  if (child < 0) throw Error(std::string("fork failed: ") + std::strerror(errno));
  if (child == 0) {
    execl("/loom/loom-bpf-lsm-loader-v12", "loom-bpf-lsm-loader-v12", "--load",
          "/loom/policy.bpf.o", kPinDirectory.data(),
          static_cast<char*>(nullptr));
    std::cerr << "LOOM_BPF_LSM_LOADER_V12_EXEC_REFUSE reason="
              << std::strerror(errno) << "\n";
    _exit(127);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child) {
    throw Error(std::string("waitpid failed: ") + std::strerror(errno));
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw Error("BPF loader did not exit cleanly status=" + std::to_string(status));
  }
}

int run_microhost() {
  if (getpid() != 1) throw Error("BPF-load init is not PID 1");
  ensure_directory("/dev");
  mount_required("devtmpfs", "/dev", "devtmpfs", MS_NOSUID);
  ensure_directory("/proc");
  mount_required("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC);
  ensure_directory("/sys");
  mount_required("sysfs", "/sys", "sysfs", MS_NOSUID | MS_NODEV | MS_NOEXEC);
  ensure_directory("/sys/kernel");
  ensure_directory("/sys/kernel/security");
  mount_required("securityfs", "/sys/kernel/security", "securityfs",
                 MS_NOSUID | MS_NODEV | MS_NOEXEC);
  ensure_directory("/sys/fs");
  ensure_directory("/sys/fs/bpf");
  mount_required("bpffs", "/sys/fs/bpf", "bpf", MS_NOSUID | MS_NODEV | MS_NOEXEC);

  const std::string active_lsm = read_file("/sys/kernel/security/lsm");
  const std::string boot_id = read_file("/proc/sys/kernel/random/boot_id", 256);
  if (!comma_token(active_lsm, "bpf")) throw Error("BPF LSM is not active");
  if (!regular_readable("/sys/kernel/btf/vmlinux")) {
    throw Error("kernel BTF is absent");
  }
  ensure_directory(kPinDirectory.data(), 0700);
  run_loader();

  std::array<std::uint32_t, kPinNames.size()> link_ids{};
  for (std::size_t index = 0; index < kPinNames.size(); ++index) {
    link_ids[index] = pinned_link_id(std::string(kPinDirectory) + "/" +
                                     std::string(kPinNames[index]));
  }
  for (const auto name : kPinNames) {
    const std::string pin = std::string(kPinDirectory) + "/" + std::string(name);
    if (unlink(pin.c_str()) != 0) {
      throw Error("cannot unlink BPF pin " + pin + ": " + std::strerror(errno));
    }
  }
  if (rmdir(kPinDirectory.data()) != 0) {
    throw Error(std::string("cannot remove BPF pin directory: ") +
                std::strerror(errno));
  }
  for (const auto id : link_ids) require_link_extinct(id);

  struct utsname identity {};
  if (uname(&identity) != 0) {
    throw Error(std::string("uname failed: ") + std::strerror(errno));
  }
  std::cout << "LOOM_KERNEL_PEER_BPF_LOAD_V12_BOOT PASS pid=1 kernel="
            << identity.release << " boot_id=" << boot_id
            << " active_lsm=" << active_lsm
            << " bpf_lsm_active=true btf=true programs_loaded=3 links_pinned=3 "
               "loader_exited=true loader_link_fds_closed=true "
               "pin_survival=true pins_unlinked=3 link_extinction=true "
               "guest_disk=none guest_network=none init_language=C++20 "
               "material_role=TRANSITORY semantic_authority=Sounio action=9025 "
               "material_peer_matrix=false same_uid_peer_isolation=false "
               "action_9025_decision=DENY451 claim_ready=false\n";
  std::cout.flush();
  sync();
  if (reboot(LINUX_REBOOT_CMD_POWER_OFF) != 0) {
    throw Error(std::string("poweroff failed: ") + std::strerror(errno));
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
      if (kPinNames.size() != 3 ||
          kPinNames[0] != "loom_v12_task_kill" ||
          kPinDirectory != "/sys/fs/bpf/loom-v12") {
        throw Error("BPF-load init constants selftest failed");
      }
      std::cout << "LOOM_KERNEL_PEER_BPF_LOAD_INIT_V12_SELFTEST PASS "
                   "programs=3 link_identity=BPF_OBJ_GET_INFO_BY_FD "
                   "extinction=BPF_LINK_GET_FD_BY_ID language=C++20 "
                   "role=MATERIAL_BOOTSTRAP transitory=true python_executed=false "
                   "rust_executed=false same_uid_peer_isolation=false "
                   "claim_ready=false\n";
      return 0;
    }
    if (argc != 1) return 64;
    return run_microhost();
  } catch (const std::exception& error) {
    std::cerr << "LOOM_KERNEL_PEER_BPF_LOAD_V12_REFUSE reason=" << error.what()
              << " material_peer_matrix=false same_uid_peer_isolation=false "
                 "action_9025_decision=DENY451 claim_ready=false\n";
    std::cerr.flush();
    sync();
    reboot(LINUX_REBOOT_CMD_POWER_OFF);
    return 70;
  }
}
