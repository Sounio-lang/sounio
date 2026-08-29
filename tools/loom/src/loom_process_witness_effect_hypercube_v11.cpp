#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <openssl/sha.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/io_uring.h>
#include <linux/memfd.h>
#include <linux/sched.h>
#include <linux/seccomp.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/personality.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kManifestSha256 =
    "adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c";
constexpr std::string_view kManifestSchema =
    "loom-process-witness-effect-policy-plan-v11-freeze-v1";
constexpr std::uint32_t kRefuse =
    SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA);

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::string sha256(const void* data, std::size_t size) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(static_cast<const unsigned char*>(data), size, digest);
  static constexpr char hex[] = "0123456789abcdef";
  std::string result(SHA256_DIGEST_LENGTH * 2, '0');
  for (std::size_t index = 0; index < SHA256_DIGEST_LENGTH; ++index) {
    result[index * 2] = hex[digest[index] >> 4];
    result[index * 2 + 1] = hex[digest[index] & 0x0f];
  }
  return result;
}

std::string sha256(std::string_view value) {
  return sha256(value.data(), value.size());
}

bool is_sha256(std::string_view value) {
  if (value.size() != 64) return false;
  for (const char byte : value) {
    if (!((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f'))) {
      return false;
    }
  }
  return true;
}

std::string read_regular_file(const std::string& path,
                              std::size_t maximum,
                              std::string_view object) {
  struct stat info {};
  if (lstat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_nlink != 1 || info.st_size <= 0 ||
      static_cast<std::uint64_t>(info.st_size) > maximum) {
    throw Error("bounded file rejected: object=" + std::string(object));
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) throw Error("bounded file cannot be opened: " + std::string(object));
  std::string contents((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
  if (input.bad() || contents.size() != static_cast<std::size_t>(info.st_size)) {
    throw Error("bounded file cannot be read: " + std::string(object));
  }
  return contents;
}

void require_line(std::string_view contents, std::string_view line) {
  const std::string framed = "\n" + std::string(contents) + "\n";
  const std::string needle = "\n" + std::string(line) + "\n";
  if (framed.find(needle) == std::string::npos) {
    throw Error("frozen Sounio V11 manifest omitted " + std::string(line));
  }
}

std::string load_manifest(const std::string& path) {
  const std::string contents = read_regular_file(path, 64 * 1024, "manifest");
  const std::string digest = sha256(contents);
  if (digest != kManifestSha256) {
    throw Error("frozen Sounio V11 manifest hash mismatch");
  }
  require_line(contents, "schema=" + std::string(kManifestSchema));
  for (const std::string_view line : {
           "stage=SEMANTICS_FROZEN",
           "producing_language=Sounio",
           "semantic_authority=Sounio",
           "action=9025",
           "family_count=12",
           "probe_count=13",
           "mechanism_dimension_count=18",
           "vertex_count=40",
           "mincut_count=13",
           "proc_treatment=CAPSULE_EMPTY_BIND",
           "legacy_proc_absence=false",
           "vertex_hash_binding=invariant_sha256+delta_sha256+witness_sha256",
           "crossed_named_rule_counts_as_completion=false",
           "experiment_unavailable_counts_as_coverage=false",
           "expected_results_source=Sounio",
           "native_v11_bytes_created=false",
           "semantics_frozen=true",
           "material_hypercube=false",
           "material_coverage=false",
           "claim_ready=false",
       }) {
    require_line(contents, line);
  }
  return digest;
}

struct Vertex {
  int family = 0;
  std::string probe;
  std::string bits;
  std::string manifest_path;
  std::string cell_path;
  std::string cell_sha256;
  std::string root_tree_sha256;
  std::string scratch_path;
  std::string inet_address;
  int inet_port = 0;
  std::string unix_path;
  std::string principal_class;
};

int dimensions_for(int family) {
  switch (family) {
    case 1:
    case 3:
    case 7:
    case 8:
    case 10:
    case 11:
      return 2;
    case 2:
    case 4:
    case 5:
    case 6:
    case 9:
    case 12:
      return 1;
    default:
      return 0;
  }
}

bool probe_matches(int family, std::string_view probe) {
  switch (family) {
    case 1:
      return probe == "repeat_exact_exec" ||
             probe == "first_wrong_flags_exec";
    case 2: return probe == "clone3_child";
    case 3: return probe == "create_named_file";
    case 4: return probe == "dup3_fd0_to_fd9";
    case 5: return probe == "mmap_shared_write";
    case 6: return probe == "io_uring_create";
    case 7: return probe == "connect_hash_bound_host_endpoint";
    case 8: return probe == "connect_hash_bound_unix_endpoint";
    case 9: return probe == "memfd_create";
    case 10: return probe == "personality_change_restore";
    case 11: return probe == "open_proc_self_mem_readonly";
    case 12: return probe == "unlisted_getpid";
    default: return false;
  }
}

bool all_zero(std::string_view bits) {
  for (const char bit : bits) {
    if (bit != '0') return false;
  }
  return true;
}

void validate_vertex(const Vertex& vertex) {
  const int dimensions = dimensions_for(vertex.family);
  if (dimensions == 0 || !probe_matches(vertex.family, vertex.probe) ||
      vertex.bits.size() != static_cast<std::size_t>(dimensions)) {
    throw Error("vertex is outside the frozen Sounio topology");
  }
  for (const char bit : vertex.bits) {
    if (bit != '0' && bit != '1') throw Error("vertex bits are noncanonical");
  }
  if (!is_sha256(vertex.cell_sha256) ||
      !is_sha256(vertex.root_tree_sha256)) {
    throw Error("vertex hash argument is malformed");
  }
  if (vertex.principal_class != "DYNAMIC_USER" &&
      vertex.principal_class != "LOCAL_SELFTEST") {
    throw Error("vertex principal class is unsupported");
  }
  if (vertex.family == 3 && vertex.scratch_path.empty()) {
    throw Error("filesystem vertex omitted its scratch path");
  }
  if (vertex.family == 7 &&
      (vertex.inet_address.empty() || vertex.inet_port < 1 ||
       vertex.inet_port > 65535)) {
    throw Error("network vertex omitted its endpoint");
  }
  if (vertex.family == 8 && vertex.unix_path.empty()) {
    throw Error("Unix endpoint vertex omitted its endpoint");
  }
}

std::string invariant_preimage(const Vertex& vertex,
                               std::string_view manifest_sha256) {
  return "schema=loom-effect-hypercube-v11|policy=" +
      std::string(manifest_sha256) + "|cell=" + vertex.cell_sha256 +
      "|tree=" + vertex.root_tree_sha256 + "|family=" +
      std::to_string(vertex.family) + "|probe=" + vertex.probe +
      "|scratch=" + vertex.scratch_path + "|inet=" + vertex.inet_address +
      ":" + std::to_string(vertex.inet_port) + "|unix=" + vertex.unix_path +
      "|principal_class=" + vertex.principal_class;
}

class FilterBuilder {
 public:
  std::vector<sock_filter> code;

  std::size_t statement(std::uint16_t opcode, std::uint32_t value) {
    code.push_back(BPF_STMT(opcode, value));
    return code.size() - 1;
  }

  std::size_t equal(std::uint32_t value) {
    code.push_back(BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, value, 0, 0));
    return code.size() - 1;
  }

  void jump_true(std::size_t index, std::size_t target) {
    if (target <= index || target - index - 1 > 255) {
      throw Error("seccomp true branch exceeds classic-BPF range");
    }
    code[index].jt = static_cast<std::uint8_t>(target - index - 1);
  }

  void jump_false(std::size_t index, std::size_t target) {
    if (target <= index || target - index - 1 > 255) {
      throw Error("seccomp false branch exceeds classic-BPF range");
    }
    code[index].jf = static_cast<std::uint8_t>(target - index - 1);
  }
};

int target_syscall(int family) {
  switch (family) {
    case 1: return SYS_execveat;
#ifdef SYS_clone3
    case 2: return SYS_clone3;
#else
    case 2: return 435;
#endif
    case 3: return SYS_openat;
    case 4: return SYS_dup3;
    case 5: return SYS_mmap;
#ifdef SYS_io_uring_setup
    case 6: return SYS_io_uring_setup;
#else
    case 6: return 425;
#endif
    case 7:
    case 8: return SYS_socket;
#ifdef SYS_memfd_create
    case 9: return SYS_memfd_create;
#else
    case 9: return 319;
#endif
    case 10: return SYS_personality;
    case 11: return SYS_openat;
    case 12: return SYS_getpid;
    default: throw Error("family has no material syscall");
  }
}

bool seccomp_active(const Vertex& vertex) {
  if (vertex.family == 1 || vertex.family == 3 || vertex.family == 7 ||
      vertex.family == 8 || vertex.family == 10 || vertex.family == 11) {
    return vertex.bits[1] == '1';
  }
  return vertex.bits[0] == '1';
}

std::vector<sock_filter> compile_filter(const Vertex& vertex) {
  if (!seccomp_active(vertex)) return {};
  FilterBuilder builder;
  builder.statement(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch));
  const std::size_t architecture = builder.equal(AUDIT_ARCH_X86_64);
  builder.statement(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS);
  builder.jump_true(architecture, builder.code.size());
  builder.statement(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr));

  if (vertex.family == 12) {
    for (const int allowed : {SYS_read, SYS_write, SYS_exit, SYS_exit_group}) {
      const std::size_t match = builder.equal(static_cast<std::uint32_t>(allowed));
      const std::size_t allow =
          builder.statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
      builder.jump_true(match, allow);
      builder.jump_false(match, builder.code.size());
    }
    builder.statement(BPF_RET | BPF_K, kRefuse);
    return builder.code;
  }

  const std::size_t syscall_match =
      builder.equal(static_cast<std::uint32_t>(target_syscall(vertex.family)));
  if (vertex.family == 1) {
    builder.statement(BPF_LD | BPF_W | BPF_ABS,
                      offsetof(seccomp_data, args[0]));
    const std::size_t fd = builder.equal(3);
    builder.statement(BPF_LD | BPF_W | BPF_ABS,
                      offsetof(seccomp_data, args[4]));
    const std::size_t flags = builder.equal(AT_EMPTY_PATH);
    const std::size_t allow =
        builder.statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
    const std::size_t deny = builder.statement(BPF_RET | BPF_K, kRefuse);
    const std::size_t pass =
        builder.statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
    builder.jump_false(syscall_match, pass);
    builder.jump_false(fd, deny);
    builder.jump_true(flags, allow);
    builder.jump_false(flags, deny);
    return builder.code;
  }
  if (vertex.family == 7 || vertex.family == 8) {
    builder.statement(BPF_LD | BPF_W | BPF_ABS,
                      offsetof(seccomp_data, args[0]));
    const std::size_t domain =
        builder.equal(vertex.family == 7 ? AF_INET : AF_UNIX);
    const std::size_t deny = builder.statement(BPF_RET | BPF_K, kRefuse);
    const std::size_t allow =
        builder.statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
    builder.jump_false(syscall_match, allow);
    builder.jump_true(domain, deny);
    builder.jump_false(domain, allow);
    return builder.code;
  }
  const std::size_t deny = builder.statement(BPF_RET | BPF_K, kRefuse);
  const std::size_t allow =
      builder.statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
  builder.jump_true(syscall_match, deny);
  builder.jump_false(syscall_match, allow);
  return builder.code;
}

bool install_filter(const std::vector<sock_filter>& filter) {
  if (filter.empty()) return true;
  if (filter.size() > std::numeric_limits<unsigned short>::max()) return false;
  sock_fprog program{static_cast<unsigned short>(filter.size()),
                    const_cast<sock_filter*>(filter.data())};
  return prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0 &&
         prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0;
}

struct ProbeResult {
  std::int64_t syscall_result = -1;
  int error_number = 0;
  int effect_completed = 0;
  int witness_extinct = 0;
  char witness_kind[64]{};
  char detail[128]{};
};

void set_text(char* target, std::size_t size, std::string_view value) {
  const std::size_t count = value.size() < size - 1 ? value.size() : size - 1;
  std::memcpy(target, value.data(), count);
  target[count] = '\0';
}

void complete(ProbeResult& result, std::string_view witness,
              std::string_view detail) {
  result.effect_completed = 1;
  result.witness_extinct = 1;
  set_text(result.witness_kind, sizeof(result.witness_kind), witness);
  set_text(result.detail, sizeof(result.detail), detail);
}

ProbeResult perform_probe(const Vertex& vertex) {
  ProbeResult result;
  errno = 0;
  switch (vertex.family) {
    case 2: {
      clone_args arguments{};
      arguments.exit_signal = SIGCHLD;
#ifdef SYS_clone3
      const long child = syscall(SYS_clone3, &arguments, sizeof(arguments));
#else
      const long child = syscall(435, &arguments, sizeof(arguments));
#endif
      result.syscall_result = child;
      result.error_number = errno;
      if (child == 0) _exit(0);
      if (child > 0) {
        int status = 0;
        while (waitpid(static_cast<pid_t>(child), &status, 0) < 0 &&
               errno == EINTR) {}
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
          complete(result, "CHILD_CREATED", "child_created_reaped");
        }
      }
      break;
    }
    case 3: {
      const long descriptor = syscall(SYS_openat, AT_FDCWD,
          vertex.scratch_path.c_str(),
          O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
      result.syscall_result = descriptor;
      result.error_number = errno;
      if (descriptor >= 0) {
        const char marker = 'V';
        const bool written = write(static_cast<int>(descriptor), &marker, 1) == 1;
        const bool closed = close(static_cast<int>(descriptor)) == 0;
        const bool removed = unlink(vertex.scratch_path.c_str()) == 0;
        const bool absent = access(vertex.scratch_path.c_str(), F_OK) != 0 &&
                            errno == ENOENT;
        if (written && closed && removed && absent) {
          complete(result, "FILE_CREATED", "file_created_removed");
        }
      }
      break;
    }
    case 4: {
      const long descriptor = syscall(SYS_dup3, STDIN_FILENO, 9, O_CLOEXEC);
      result.syscall_result = descriptor;
      result.error_number = errno;
      if (descriptor == 9) {
        const bool present = fcntl(9, F_GETFD) >= 0;
        const bool closed = close(9) == 0;
        errno = 0;
        const bool absent = fcntl(9, F_GETFD) < 0 && errno == EBADF;
        if (present && closed && absent) {
          complete(result, "FD9_CREATED", "fd9_created_closed");
        }
      }
      break;
    }
    case 5: {
      void* mapping = reinterpret_cast<void*>(syscall(
          SYS_mmap, nullptr, 4096, PROT_READ | PROT_WRITE,
          MAP_SHARED | MAP_ANONYMOUS, -1, 0));
      result.syscall_result = reinterpret_cast<std::intptr_t>(mapping);
      result.error_number = errno;
      if (mapping != MAP_FAILED) {
        *static_cast<volatile unsigned char*>(mapping) = 0x5a;
        if (munmap(mapping, 4096) == 0) {
          complete(result, "SHARED_MAPPING_CREATED", "mapping_written_unmapped");
        }
      }
      break;
    }
    case 6: {
      io_uring_params parameters{};
#ifdef SYS_io_uring_setup
      const long descriptor = syscall(SYS_io_uring_setup, 1, &parameters);
#else
      const long descriptor = syscall(425, 1, &parameters);
#endif
      result.syscall_result = descriptor;
      result.error_number = errno;
      if (descriptor >= 0) {
        const bool closed = close(static_cast<int>(descriptor)) == 0;
        errno = 0;
        const bool absent =
            fcntl(static_cast<int>(descriptor), F_GETFD) < 0 && errno == EBADF;
        if (closed && absent) {
          complete(result, "IO_URING_CREATED", "ring_created_closed");
        }
      }
      break;
    }
    case 7: {
      const long descriptor =
          syscall(SYS_socket, AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
      result.syscall_result = descriptor;
      result.error_number = errno;
      if (descriptor >= 0) {
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<std::uint16_t>(vertex.inet_port));
        if (inet_pton(AF_INET, vertex.inet_address.c_str(),
                      &address.sin_addr) == 1 &&
            connect(static_cast<int>(descriptor),
                    reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
          if (close(static_cast<int>(descriptor)) == 0) {
            complete(result, "HOST_ENDPOINT_CONNECTED", "inet_connected_closed");
          }
        } else {
          result.syscall_result = -1;
          result.error_number = errno;
          close(static_cast<int>(descriptor));
        }
      }
      break;
    }
    case 8: {
      const long descriptor =
          syscall(SYS_socket, AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
      result.syscall_result = descriptor;
      result.error_number = errno;
      if (descriptor >= 0) {
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        if (vertex.unix_path.size() < sizeof(address.sun_path)) {
          std::memcpy(address.sun_path, vertex.unix_path.c_str(),
                      vertex.unix_path.size() + 1);
          if (connect(static_cast<int>(descriptor),
                      reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) {
            if (close(static_cast<int>(descriptor)) == 0) {
              complete(result, "UNIX_ENDPOINT_CONNECTED", "unix_connected_closed");
            }
          } else {
            result.syscall_result = -1;
            result.error_number = errno;
            close(static_cast<int>(descriptor));
          }
        }
      }
      break;
    }
    case 9: {
#ifdef SYS_memfd_create
      const long descriptor = syscall(SYS_memfd_create, "loom-v11", MFD_CLOEXEC);
#else
      const long descriptor = syscall(319, "loom-v11", MFD_CLOEXEC);
#endif
      result.syscall_result = descriptor;
      result.error_number = errno;
      if (descriptor >= 0) {
        const char marker = 'V';
        const bool written = write(static_cast<int>(descriptor), &marker, 1) == 1;
        const bool closed = close(static_cast<int>(descriptor)) == 0;
        errno = 0;
        const bool absent =
            fcntl(static_cast<int>(descriptor), F_GETFD) < 0 && errno == EBADF;
        if (written && closed && absent) {
          complete(result, "MEMFD_CREATED", "memfd_created_closed");
        }
      }
      break;
    }
    case 10: {
      const long current = syscall(SYS_personality, 0xffffffffUL);
      result.syscall_result = current;
      result.error_number = errno;
      if (current >= 0) {
        const unsigned long changed =
            static_cast<unsigned long>(current) ^ ADDR_NO_RANDOMIZE;
        const long change_result = syscall(SYS_personality, changed);
        result.syscall_result = change_result;
        result.error_number = errno;
        if (change_result >= 0) {
          const long restore_result =
              syscall(SYS_personality, static_cast<unsigned long>(current));
          if (restore_result >= 0) {
            complete(result, "PERSONALITY_CHANGED_AND_RESTORED",
                     "personality_changed_restored");
          }
        }
      }
      break;
    }
    case 11: {
      const long descriptor = syscall(SYS_openat, AT_FDCWD,
          "/proc/self/mem", O_RDONLY | O_CLOEXEC, 0);
      result.syscall_result = descriptor;
      result.error_number = errno;
      if (descriptor >= 0 && close(static_cast<int>(descriptor)) == 0) {
        complete(result, "PROC_SELF_MEM_OPENED", "proc_object_opened_closed");
      }
      break;
    }
    case 12: {
      const long pid = syscall(SYS_getpid);
      result.syscall_result = pid;
      result.error_number = errno;
      if (pid > 0) complete(result, "GETPID_RETURNED", "pid_returned");
      break;
    }
    default:
      result.syscall_result = -1;
      result.error_number = EINVAL;
      break;
  }
  return result;
}

bool write_all(int descriptor, const void* data, std::size_t size) {
  const char* cursor = static_cast<const char*>(data);
  while (size > 0) {
    const ssize_t count = write(descriptor, cursor, size);
    if (count > 0) {
      cursor += count;
      size -= static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}

ProbeResult run_probe_child(const Vertex& vertex,
                            const std::vector<sock_filter>& filter) {
  int channels[2] = {-1, -1};
  if (pipe2(channels, O_CLOEXEC) != 0) throw Error("probe channel creation failed");
  const pid_t child = fork();
  if (child < 0) throw Error("probe fork failed");
  if (child == 0) {
    close(channels[0]);
    if (!install_filter(filter)) _exit(90);
    const ProbeResult result = perform_probe(vertex);
    const bool written = write_all(channels[1], &result, sizeof(result));
    close(channels[1]);
    _exit(written ? 0 : 91);
  }
  close(channels[1]);
  ProbeResult result;
  std::size_t offset = 0;
  while (offset < sizeof(result)) {
    const ssize_t count = read(channels[0],
        reinterpret_cast<char*>(&result) + offset, sizeof(result) - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      break;
    }
  }
  close(channels[0]);
  int status = 0;
  while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
  if (offset != sizeof(result) || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw Error("probe child failed before a complete observation");
  }
  return result;
}

std::string errno_name(int value) {
  switch (value) {
    case 0: return "NONE";
    case EACCES: return "EACCES";
    case EBADF: return "EBADF";
    case ECONNREFUSED: return "ECONNREFUSED";
    case EINVAL: return "EINVAL";
    case ENETUNREACH: return "ENETUNREACH";
    case ENOENT: return "ENOENT";
    case ENOSYS: return "ENOSYS";
    case EOPNOTSUPP: return "EOPNOTSUPP";
    case EPERM: return "EPERM";
    case EROFS: return "EROFS";
    default: return "ERRNO_" + std::to_string(value);
  }
}

std::string classify(const Vertex& vertex, const ProbeResult& result) {
  if (result.effect_completed == 1 && result.witness_extinct == 1) {
    return "EFFECT_COMPLETED";
  }
  if (result.syscall_result >= 0) return "CROSSED_NAMED_RULE";
  if (all_zero(vertex.bits)) return "EXPERIMENT_UNAVAILABLE";
  return "REFUSED_BEFORE_EFFECT";
}

std::string receipt(const Vertex& vertex, std::string_view invariant_sha256,
                    const ProbeResult& result) {
  const std::string observation = classify(vertex, result);
  const std::string syscall_result = result.syscall_result < 0
      ? errno_name(result.error_number)
      : "SUCCESS";
  const std::string witness_kind = result.effect_completed == 1
      ? std::string(result.witness_kind)
      : "NONE";
  const std::string witness_preimage =
      "observation=" + observation + "|syscall_result=" + syscall_result +
      "|witness_kind=" + witness_kind + "|detail=" + result.detail +
      "|extinct=" + (result.witness_extinct == 1 ? "true" : "false");
  return "LOOM_EFFECT_VERTEX_V11 OBSERVED family=" +
      std::to_string(vertex.family) + " probe=" + vertex.probe +
      " bits=" + vertex.bits + " observation=" + observation +
      " syscall_result=" + syscall_result + " witness_kind=" + witness_kind +
      " witness_extinct=" +
      (result.witness_extinct == 1 ? std::string("true") : std::string("false")) +
      " invariant_sha256=" + std::string(invariant_sha256) +
      " delta_sha256=" + sha256(vertex.bits) +
      " witness_sha256=" + sha256(witness_preimage) +
      " semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY"
      " transitory=true semantic_decision=false material_coverage=false"
      " complete_effects=false material_execution=false claim_ready=false";
}

void emit_exec_receipt(std::string_view probe, std::string_view bits,
                       std::string_view invariant_sha256,
                       std::string_view observation,
                       std::string_view syscall_result,
                       std::string_view witness_kind,
                       bool extinct) {
  const std::string witness_preimage =
      "observation=" + std::string(observation) + "|syscall_result=" +
      std::string(syscall_result) + "|witness_kind=" +
      std::string(witness_kind) + "|detail=exec_transition|extinct=" +
      (extinct ? "true" : "false");
  std::cout << "LOOM_EFFECT_VERTEX_V11 OBSERVED family=1 probe=" << probe
            << " bits=" << bits << " observation=" << observation
            << " syscall_result=" << syscall_result
            << " witness_kind=" << witness_kind
            << " witness_extinct=" << (extinct ? "true" : "false")
            << " invariant_sha256=" << invariant_sha256
            << " delta_sha256=" << sha256(bits)
            << " witness_sha256=" << sha256(witness_preimage)
            << " semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY"
               " transitory=true semantic_decision=false material_coverage=false"
               " complete_effects=false material_execution=false claim_ready=false\n";
}

[[noreturn]] void exec_stage(int argc, char** argv) {
  if (argc != 6) throw Error("exec stage arguments are malformed");
  const std::string probe = argv[2];
  const std::string bits = argv[3];
  const std::string invariant = argv[4];
  const std::string stage = argv[5];
  if (!is_sha256(invariant) || bits.size() != 2) {
    throw Error("exec stage binding is malformed");
  }
  if (stage == "FIRST" || stage == "SECOND") {
    emit_exec_receipt(probe, bits, invariant, "EFFECT_COMPLETED",
                      "EXEC_COMPLETED",
                      stage == "FIRST" ? "FIRST_EXEC_MARKER" :
                                         "SECOND_EXEC_MARKER",
                      true);
    std::exit(0);
  }
  if (stage != "REPEAT") throw Error("exec stage is unknown");
  std::array<char*, 7> arguments{
      argv[0], const_cast<char*>("--exec-stage"),
      const_cast<char*>(probe.c_str()), const_cast<char*>(bits.c_str()),
      const_cast<char*>(invariant.c_str()), const_cast<char*>("SECOND"), nullptr};
  std::array<char*, 1> environment{nullptr};
  errno = 0;
  syscall(SYS_execveat, 3, "", arguments.data(), environment.data(),
          AT_EMPTY_PATH);
  emit_exec_receipt(probe, bits, invariant, "REFUSED_BEFORE_EFFECT",
                    errno_name(errno), "NONE", false);
  std::exit(0);
}

int run_exec_vertex(const Vertex& vertex, std::string_view invariant_sha256,
                    const std::vector<sock_filter>& filter) {
  int descriptor = open(vertex.cell_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) throw Error("exec vertex cannot open its cell");
  if (descriptor != 3) {
    if (dup3(descriptor, 3, O_CLOEXEC) != 3) {
      close(descriptor);
      throw Error("exec vertex cannot establish fd3");
    }
    close(descriptor);
  }
  const bool close_on_exec = vertex.bits[0] == '1';
  if (fcntl(3, F_SETFD, close_on_exec ? FD_CLOEXEC : 0) != 0) {
    throw Error("exec vertex cannot set fd3 close-on-exec state");
  }
  if (!install_filter(filter)) throw Error("exec vertex cannot install seccomp");
  const std::string stage = vertex.probe == "repeat_exact_exec" ? "REPEAT" : "FIRST";
  std::array<char*, 7> arguments{
      const_cast<char*>(vertex.cell_path.c_str()),
      const_cast<char*>("--exec-stage"),
      const_cast<char*>(vertex.probe.c_str()),
      const_cast<char*>(vertex.bits.c_str()),
      const_cast<char*>(invariant_sha256.data()),
      const_cast<char*>(stage.c_str()), nullptr};
  std::array<char*, 1> environment{nullptr};
  const int flags = vertex.probe == "repeat_exact_exec"
      ? AT_EMPTY_PATH
      : AT_EMPTY_PATH | AT_SYMLINK_NOFOLLOW;
  errno = 0;
  syscall(SYS_execveat, 3, "", arguments.data(), environment.data(), flags);
  emit_exec_receipt(vertex.probe, vertex.bits, invariant_sha256,
                    all_zero(vertex.bits) ? "EXPERIMENT_UNAVAILABLE" :
                                            "REFUSED_BEFORE_EFFECT",
                    errno_name(errno), "NONE", false);
  return 0;
}

int selftest(const std::string& manifest_path) {
  const std::string manifest_sha256 = load_manifest(manifest_path);
  std::string filter_material;
  int filters = 0;
  for (int family = 1; family <= 12; ++family) {
    Vertex vertex;
    vertex.family = family;
    vertex.probe = family == 1 ? "repeat_exact_exec" :
        family == 2 ? "clone3_child" :
        family == 3 ? "create_named_file" :
        family == 4 ? "dup3_fd0_to_fd9" :
        family == 5 ? "mmap_shared_write" :
        family == 6 ? "io_uring_create" :
        family == 7 ? "connect_hash_bound_host_endpoint" :
        family == 8 ? "connect_hash_bound_unix_endpoint" :
        family == 9 ? "memfd_create" :
        family == 10 ? "personality_change_restore" :
        family == 11 ? "open_proc_self_mem_readonly" : "unlisted_getpid";
    vertex.bits = dimensions_for(family) == 2 ? "01" : "1";
    const auto filter = compile_filter(vertex);
    if (filter.empty()) throw Error("material filter unexpectedly empty");
    filter_material += std::to_string(family) + ":" +
        sha256(filter.data(), filter.size() * sizeof(sock_filter)) + "\n";
    ++filters;
  }
  std::cout << "LOOM_PROCESS_WITNESS_EFFECT_HYPERCUBE_V11_SELFTEST PASS"
            << " semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY"
            << " transitory=true semantic_decision=false action=9025"
            << " policy_manifest_sha256=" << manifest_sha256
            << " families=12 probes=13 mechanism_dimensions=18 vertices=40"
            << " compiled_filters=" << filters
            << " filter_set_sha256=" << sha256(filter_material)
            << " vertex_mode=true exec_transition_mode=true"
            << " triple_hash_binding=true material_hypercube=false"
            << " material_coverage=false complete_effects=false"
            << " material_execution=false claim_ready=false\n";
  return 0;
}

std::string option_value(int argc, char** argv, std::string_view name) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::string_view(argv[index]) == name) return argv[index + 1];
  }
  return {};
}

int parse_int(std::string_view value, std::string_view object) {
  if (value.empty()) throw Error(std::string(object) + " is absent");
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(std::string(value).c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    throw Error(std::string(object) + " is malformed");
  }
  return static_cast<int>(parsed);
}

Vertex parse_vertex(int argc, char** argv) {
  Vertex vertex;
  vertex.family = parse_int(option_value(argc, argv, "--family"), "family");
  vertex.probe = option_value(argc, argv, "--probe");
  vertex.bits = option_value(argc, argv, "--bits");
  vertex.manifest_path = option_value(argc, argv, "--policy-manifest");
  vertex.cell_path = option_value(argc, argv, "--cell-path");
  vertex.cell_sha256 = option_value(argc, argv, "--cell-sha256");
  vertex.root_tree_sha256 = option_value(argc, argv, "--root-tree-sha256");
  vertex.scratch_path = option_value(argc, argv, "--scratch-path");
  vertex.inet_address = option_value(argc, argv, "--inet-address");
  const std::string port = option_value(argc, argv, "--inet-port");
  vertex.inet_port = port.empty() ? 0 : parse_int(port, "inet port");
  vertex.unix_path = option_value(argc, argv, "--unix-path");
  vertex.principal_class = option_value(argc, argv, "--principal-class");
  validate_vertex(vertex);
  return vertex;
}

int run_vertex(const Vertex& vertex) {
  const std::string manifest_sha256 = load_manifest(vertex.manifest_path);
  const std::string cell_contents =
      read_regular_file(vertex.cell_path, 16 * 1024 * 1024, "effect_cell");
  if (sha256(cell_contents) != vertex.cell_sha256) {
    throw Error("effect cell hash mismatch");
  }
  const std::string invariant_sha256 =
      sha256(invariant_preimage(vertex, manifest_sha256));
  const auto filter = compile_filter(vertex);
  if (vertex.family == 1) {
    return run_exec_vertex(vertex, invariant_sha256, filter);
  }
  const ProbeResult result = run_probe_child(vertex, filter);
  std::cout << receipt(vertex, invariant_sha256, result) << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 6 && std::string_view(argv[1]) == "--exec-stage") {
      exec_stage(argc, argv);
    }
    if (argc == 4 && std::string_view(argv[1]) == "--selftest" &&
        std::string_view(argv[2]) == "--policy-manifest") {
      return selftest(argv[3]);
    }
    if (argc >= 2 && std::string_view(argv[1]) == "--vertex") {
      return run_vertex(parse_vertex(argc, argv));
    }
    std::cerr << "usage: loom-process-witness-effect-hypercube-v11 --selftest"
                 " --policy-manifest PATH\n"
                 "       loom-process-witness-effect-hypercube-v11 --vertex"
                 " --family N --probe NAME --bits BITS --policy-manifest PATH"
                 " --cell-path PATH --cell-sha256 HEX --root-tree-sha256 HEX"
                 " --principal-class CLASS [probe endpoints]\n";
    return 64;
  } catch (const std::exception& error) {
    std::cerr << "LOOM_PROCESS_WITNESS_EFFECT_HYPERCUBE_V11_CLOSED reason="
              << error.what() << '\n';
    return 70;
  }
}
