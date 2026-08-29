#include <fcntl.h>
#include <linux/reboot.h>
#include <sys/mount.h>
#include <sys/reboot.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <array>
#include <cerrno>
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

void ensure_directory(const char* path, mode_t mode = 0755) {
  if (mkdir(path, mode) != 0 && errno != EEXIST) {
    throw Error(std::string("mkdir failed for ") + path + ": " + std::strerror(errno));
  }
}

void mount_required(const char* source, const char* target, const char* type,
                    unsigned long flags = 0) {
  if (mount(source, target, type, flags, nullptr) != 0) {
    throw Error(std::string("mount failed for ") + target + ": " + std::strerror(errno));
  }
}

std::string read_file(const char* path, std::size_t maximum = 64 * 1024) {
  const int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    throw Error(std::string("open failed for ") + path + ": " + std::strerror(errno));
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

int run_microhost() {
  if (getpid() != 1) throw Error("microhost init is not PID 1");
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
  if (!regular_readable("/sys/kernel/btf/vmlinux")) throw Error("kernel BTF is absent");
  struct utsname identity {};
  if (uname(&identity) != 0) {
    throw Error(std::string("uname failed: ") + std::strerror(errno));
  }

  std::cout << "LOOM_KERNEL_PEER_MICROHOST_V12_BOOT PASS pid=1 kernel="
            << identity.release << " boot_id=" << boot_id
            << " active_lsm=" << active_lsm
            << " bpf_lsm_active=true securityfs=true bpffs=true btf=true "
               "guest_disk=none guest_network=none init_language=C++20 "
               "init_role=MATERIAL_BOOTSTRAP semantic_authority=Sounio action=9025 "
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
      if (!comma_token("capability,bpf,landlock", "bpf") ||
          comma_token("capability,landlock", "bpf")) {
        throw Error("bounded LSM parser selftest failed");
      }
      std::cout << "LOOM_KERNEL_PEER_MICROHOST_INIT_V12_SELFTEST PASS "
                   "language=C++20 role=MATERIAL_BOOTSTRAP transitory=true "
                   "semantic_authority=Sounio action=9025 disk=none network=none "
                   "python_executed=false rust_executed=false material_peer_matrix=false "
                   "same_uid_peer_isolation=false claim_ready=false\n";
      return 0;
    }
    if (argc != 1) return 64;
    return run_microhost();
  } catch (const std::exception& error) {
    std::cerr << "LOOM_KERNEL_PEER_MICROHOST_V12_REFUSE reason=" << error.what()
              << " material_peer_matrix=false same_uid_peer_isolation=false "
                 "action_9025_decision=DENY451 claim_ready=false\n";
    std::cerr.flush();
    sync();
    reboot(LINUX_REBOOT_CMD_POWER_OFF);
    return 70;
  }
}
