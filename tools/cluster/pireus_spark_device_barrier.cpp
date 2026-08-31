#include <linux/bpf.h>
#include <linux/bpf_common.h>
#include <linux/magic.h>

#include <fcntl.h>
#include <glob.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTagSize = 8;
constexpr std::size_t kInitialQueryCapacity = 8;
constexpr std::size_t kMaxPrograms = 64;
constexpr unsigned kQueryAttempts = 4;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

int bpf_call(enum bpf_cmd command, union bpf_attr* attributes) {
  return static_cast<int>(
      syscall(SYS_bpf, command, attributes, sizeof(*attributes)));
}

void close_checked(int descriptor) {
  if (descriptor >= 0) close(descriptor);
}

bpf_insn instruction(std::uint8_t code, std::uint8_t dst,
                     std::uint8_t src, std::int16_t offset,
                     std::int32_t immediate) {
  bpf_insn value{};
  value.code = code;
  value.dst_reg = dst;
  value.src_reg = src;
  value.off = offset;
  value.imm = immediate;
  return value;
}

std::set<unsigned> parse_majors(const std::string& raw) {
  std::set<unsigned> values;
  std::stringstream stream(raw);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty() ||
        token.find_first_not_of("0123456789") != std::string::npos) {
      fail("device majors must be a non-empty comma-separated decimal set");
    }
    const unsigned long parsed = std::stoul(token);
    if (parsed == 0 || parsed > 0xffffffffUL) {
      fail("device major is outside the kernel u32 range");
    }
    if (!values.insert(static_cast<unsigned>(parsed)).second) {
      fail("device major set contains a duplicate");
    }
  }
  if (values.empty() || values.size() > 32) {
    fail("device major set must contain between 1 and 32 values");
  }
  return values;
}

std::string join_majors(const std::set<unsigned>& values) {
  std::ostringstream output;
  bool first = true;
  for (const unsigned value : values) {
    if (!first) output << ',';
    output << value;
    first = false;
  }
  return output.str();
}

std::vector<bpf_insn> deny_major_program(const std::set<unsigned>& majors) {
  std::vector<bpf_insn> program;
  program.reserve(majors.size() + 5);
  program.push_back(instruction(
      BPF_LDX | BPF_MEM | BPF_W, BPF_REG_2, BPF_REG_1,
      static_cast<std::int16_t>(offsetof(bpf_cgroup_dev_ctx, major)), 0));
  std::size_t index = 0;
  for (const unsigned value : majors) {
    const auto jump = static_cast<std::int16_t>(majors.size() + 1 - index);
    program.push_back(instruction(BPF_JMP | BPF_JEQ | BPF_K, BPF_REG_2, 0,
                                  jump, static_cast<std::int32_t>(value)));
    ++index;
  }
  program.push_back(
      instruction(BPF_ALU64 | BPF_MOV | BPF_K, BPF_REG_0, 0, 0, 1));
  program.push_back(instruction(BPF_JMP | BPF_EXIT, 0, 0, 0, 0));
  program.push_back(
      instruction(BPF_ALU64 | BPF_MOV | BPF_K, BPF_REG_0, 0, 0, 0));
  program.push_back(instruction(BPF_JMP | BPF_EXIT, 0, 0, 0, 0));
  return program;
}

int load_program(const std::vector<bpf_insn>& program) {
  std::array<char, 65536> log{};
  static constexpr char license[] = "MIT";
  union bpf_attr attributes{};
  attributes.prog_type = BPF_PROG_TYPE_CGROUP_DEVICE;
  attributes.expected_attach_type = BPF_CGROUP_DEVICE;
  attributes.insn_cnt = static_cast<__u32>(program.size());
  attributes.insns = reinterpret_cast<__u64>(program.data());
  attributes.license = reinterpret_cast<__u64>(license);
  attributes.log_buf = reinterpret_cast<__u64>(log.data());
  attributes.log_size = static_cast<__u32>(log.size());
  attributes.log_level = 1;
  std::strncpy(attributes.prog_name, "pireus_devbar",
               BPF_OBJ_NAME_LEN - 1);
  const int descriptor = bpf_call(BPF_PROG_LOAD, &attributes);
  if (descriptor < 0) {
    fail("BPF_PROG_LOAD failed: " + std::string(std::strerror(errno)) +
         ": " + std::string(log.data()));
  }
  return descriptor;
}

struct ProgramIdentity {
  __u32 id;
  __u32 type;
  std::array<std::uint8_t, kTagSize> tag;
  std::vector<std::uint8_t> translated;
};

ProgramIdentity program_identity(int descriptor) {
  bpf_prog_info info{};
  union bpf_attr attributes{};
  attributes.info.bpf_fd = static_cast<__u32>(descriptor);
  attributes.info.info_len = sizeof(info);
  attributes.info.info = reinterpret_cast<__u64>(&info);
  if (bpf_call(BPF_OBJ_GET_INFO_BY_FD, &attributes) < 0) {
    fail("BPF_OBJ_GET_INFO_BY_FD failed: " +
         std::string(std::strerror(errno)));
  }

  std::vector<std::uint8_t> translated(info.xlated_prog_len);
  info.xlated_prog_insns = reinterpret_cast<__u64>(translated.data());
  attributes = {};
  attributes.info.bpf_fd = static_cast<__u32>(descriptor);
  attributes.info.info_len = sizeof(info);
  attributes.info.info = reinterpret_cast<__u64>(&info);
  if (bpf_call(BPF_OBJ_GET_INFO_BY_FD, &attributes) < 0) {
    fail("BPF_OBJ_GET_INFO_BY_FD translated program failed: " +
         std::string(std::strerror(errno)));
  }
  if (info.xlated_prog_len != translated.size()) {
    fail("translated BPF program length raced");
  }

  ProgramIdentity identity{};
  identity.id = info.id;
  identity.type = info.type;
  std::copy_n(info.tag, kTagSize, identity.tag.begin());
  identity.translated = std::move(translated);
  return identity;
}

std::string bytes_hex(const std::uint8_t* bytes, std::size_t size) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string output;
  output.reserve(size * 2);
  for (std::size_t index = 0; index < size; ++index) {
    const std::uint8_t value = bytes[index];
    output.push_back(digits[value >> 4]);
    output.push_back(digits[value & 0x0f]);
  }
  return output;
}

std::string tag_hex(const ProgramIdentity& identity) {
  return bytes_hex(identity.tag.data(), identity.tag.size());
}

std::string translated_hex(const ProgramIdentity& identity) {
  return bytes_hex(identity.translated.data(), identity.translated.size());
}

bool same_program(const ProgramIdentity& left,
                  const ProgramIdentity& right) {
  return left.type == BPF_PROG_TYPE_CGROUP_DEVICE &&
         left.type == right.type &&
         left.tag == right.tag && left.translated == right.translated;
}

struct QueryResult {
  std::vector<__u32> ids;
  std::vector<__u32> attach_flags;
  __u64 revision;
};

QueryResult query_programs(int cgroup, __u32 query_flags = 0) {
  std::size_t capacity = kInitialQueryCapacity;
  for (unsigned attempt = 0; attempt < kQueryAttempts; ++attempt) {
    std::vector<__u32> ids(capacity);
    std::vector<__u32> flags(capacity);
    union bpf_attr attributes{};
    attributes.query.target_fd = static_cast<__u32>(cgroup);
    attributes.query.attach_type = BPF_CGROUP_DEVICE;
    attributes.query.query_flags = query_flags;
    attributes.query.prog_cnt = static_cast<__u32>(capacity);
    attributes.query.prog_ids = reinterpret_cast<__u64>(ids.data());
    if (query_flags == 0) {
      attributes.query.prog_attach_flags =
          reinterpret_cast<__u64>(flags.data());
    }
    if (bpf_call(BPF_PROG_QUERY, &attributes) == 0) {
      if (attributes.query.prog_cnt > capacity) {
        capacity = attributes.query.prog_cnt;
        continue;
      }
      ids.resize(attributes.query.prog_cnt);
      flags.resize(query_flags == 0 ? attributes.query.prog_cnt : 0);
      return QueryResult{std::move(ids), std::move(flags),
                         attributes.query.revision};
    }
    if (errno != ENOSPC) {
      fail("BPF_PROG_QUERY failed: " + std::string(std::strerror(errno)));
    }
    capacity = std::max(capacity * 2,
                        static_cast<std::size_t>(attributes.query.prog_cnt));
    if (capacity > kMaxPrograms) {
      fail("too many device programs are attached to guarded cgroup");
    }
  }
  fail("BPF_PROG_QUERY did not stabilize");
}

int program_fd_by_id(__u32 identifier) {
  union bpf_attr attributes{};
  attributes.prog_id = identifier;
  const int descriptor = bpf_call(BPF_PROG_GET_FD_BY_ID, &attributes);
  if (descriptor < 0) {
    fail("BPF_PROG_GET_FD_BY_ID failed: " +
         std::string(std::strerror(errno)));
  }
  return descriptor;
}

void attach_program(int cgroup, int program, __u64 expected_revision) {
  union bpf_attr attributes{};
  attributes.target_fd = static_cast<__u32>(cgroup);
  attributes.attach_bpf_fd = static_cast<__u32>(program);
  attributes.attach_type = BPF_CGROUP_DEVICE;
  attributes.attach_flags = BPF_F_ALLOW_MULTI;
  attributes.expected_revision = expected_revision;
  if (bpf_call(BPF_PROG_ATTACH, &attributes) < 0) {
    fail("BPF_PROG_ATTACH with expected revision failed: " +
         std::string(std::strerror(errno)));
  }
}

void detach_program(int cgroup, int program, __u64 expected_revision) {
  union bpf_attr attributes{};
  attributes.target_fd = static_cast<__u32>(cgroup);
  attributes.attach_bpf_fd = static_cast<__u32>(program);
  attributes.attach_type = BPF_CGROUP_DEVICE;
  attributes.expected_revision = expected_revision;
  if (bpf_call(BPF_PROG_DETACH, &attributes) < 0) {
    fail("BPF_PROG_DETACH with expected revision failed: " +
         std::string(std::strerror(errno)));
  }
}

int open_cgroup(const std::string& path, struct stat* metadata) {
  const int descriptor =
      open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    fail("cannot open guarded cgroup: " + std::string(std::strerror(errno)));
  }
  struct statfs filesystem{};
  if (fstat(descriptor, metadata) < 0 ||
      fstatfs(descriptor, &filesystem) < 0 ||
      filesystem.f_type != CGROUP2_SUPER_MAGIC) {
    close_checked(descriptor);
    fail("guarded path is not an exact cgroup v2 directory");
  }
  return descriptor;
}

std::string read_first_line(const std::string& path) {
  std::ifstream input(path);
  std::string value;
  if (!input || !std::getline(input, value) || value.empty()) {
    fail("cannot read required identity file: " + path);
  }
  return value;
}

std::string state_text(const std::string& boot_id, const struct stat& cgroup,
                       const ProgramIdentity& identity,
                       const std::set<unsigned>& majors) {
  std::ostringstream output;
  output << "schema=pireus-device-barrier-state-v1\n"
         << "boot_id=" << boot_id << '\n'
         << "cgroup_dev=" << static_cast<unsigned long long>(cgroup.st_dev)
         << '\n'
         << "cgroup_ino=" << static_cast<unsigned long long>(cgroup.st_ino)
         << '\n'
         << "program_id=" << identity.id << '\n'
         << "program_tag=" << tag_hex(identity) << '\n'
         << "majors=" << join_majors(majors) << '\n'
         << "translated=" << translated_hex(identity) << '\n';
  return output.str();
}

std::string parent_directory(const std::string& path) {
  const std::size_t separator = path.find_last_of('/');
  if (separator == std::string::npos) return ".";
  if (separator == 0) return "/";
  return path.substr(0, separator);
}

void write_all(int descriptor, const std::string& contents) {
  std::size_t offset = 0;
  while (offset < contents.size()) {
    const ssize_t written =
        write(descriptor, contents.data() + offset, contents.size() - offset);
    if (written < 0) fail("device barrier state write failed");
    offset += static_cast<std::size_t>(written);
  }
}

void sync_parent(const std::string& path) {
  const std::string parent = parent_directory(path);
  const int descriptor =
      open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0 || fsync(descriptor) < 0) {
    close_checked(descriptor);
    fail("device barrier state directory sync failed");
  }
  close_checked(descriptor);
}

void write_state(const std::string& path, const std::string& contents) {
  const std::string temporary =
      path + ".tmp." + std::to_string(static_cast<long long>(getpid()));
  const int descriptor =
      open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (descriptor < 0) fail("cannot create device barrier state");
  try {
    write_all(descriptor, contents);
    if (fsync(descriptor) < 0) fail("device barrier state sync failed");
    close_checked(descriptor);
    if (rename(temporary.c_str(), path.c_str()) < 0) {
      fail("device barrier state rename failed");
    }
    sync_parent(path);
  } catch (...) {
    close_checked(descriptor);
    unlink(temporary.c_str());
    throw;
  }
}

std::string read_state(const std::string& path) {
  std::ifstream input(path);
  if (!input) fail("device barrier state is missing");
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) fail("device barrier state read failed");
  return contents.str();
}

void remove_state(const std::string& path) {
  if (unlink(path.c_str()) < 0) {
    fail("device barrier state removal failed: " +
         std::string(std::strerror(errno)));
  }
  sync_parent(path);
}

std::set<unsigned> device_majors(const std::string& root) {
  const std::array<std::string, 4> patterns = {
      root + "/nvidia*", root + "/nvidia-caps/*", root + "/dri/*",
      root + "/dma_heap/*"};
  std::set<unsigned> values;
  for (const std::string& pattern : patterns) {
    glob_t matches{};
    const int status = glob(pattern.c_str(), GLOB_NOSORT, nullptr, &matches);
    if (status != 0 && status != GLOB_NOMATCH) {
      globfree(&matches);
      fail("device inventory glob failed");
    }
    for (std::size_t index = 0; index < matches.gl_pathc; ++index) {
      struct stat metadata{};
      if (lstat(matches.gl_pathv[index], &metadata) < 0) {
        globfree(&matches);
        fail("device lstat failed: " + std::string(std::strerror(errno)));
      }
      if (S_ISLNK(metadata.st_mode)) {
        globfree(&matches);
        fail("device inventory contains a symlink");
      }
      if (S_ISCHR(metadata.st_mode)) {
        values.insert(static_cast<unsigned>(major(metadata.st_rdev)));
      } else if (!S_ISDIR(metadata.st_mode)) {
        globfree(&matches);
        fail("device inventory contains a non-device entry");
      }
    }
    globfree(&matches);
  }
  return values;
}

void verify_devices(const std::string& root,
                    const std::set<unsigned>& expected) {
  const std::set<unsigned> observed = device_majors(root);
  if (observed != expected) {
    fail("GPU device major inventory drifted: expected=" +
         join_majors(expected) + " observed=" + join_majors(observed));
  }
}

bool simulated_allow(const std::set<unsigned>& denied, unsigned value) {
  return denied.find(value) == denied.end();
}

void selftest() {
  const std::set<unsigned> denied = parse_majors("195,226,247,498,501");
  const auto program = deny_major_program(denied);
  bool duplicate_refused = false;
  try {
    static_cast<void>(parse_majors("195,195"));
  } catch (const std::exception&) {
    duplicate_refused = true;
  }
  if (program.size() != denied.size() + 5 || simulated_allow(denied, 195) ||
      simulated_allow(denied, 498) || !simulated_allow(denied, 1) ||
      !duplicate_refused) {
    fail("instruction generator selftest failed");
  }
  std::cout << "PIREUS_DEVICE_BARRIER_SELFTEST_PASS majors="
            << join_majors(denied)
            << " default=ALLOW matched=DENY duplicates=REFUSE\n";
}

void usage(const char* executable) {
  std::cerr << "usage: " << executable
            << " selftest | verify-devices DEVICE_ROOT MAJORS | "
               "attach|detach|status-attached|status-detached "
               "CGROUP MAJORS STATE\n";
}

void verify_single_attached(const QueryResult& query) {
  if (query.ids.size() != 1 || query.attach_flags.size() != 1 ||
      query.attach_flags.front() != BPF_F_ALLOW_MULTI) {
    fail("guarded cgroup does not have one exact direct device barrier");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "selftest") {
      selftest();
      return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "verify-devices") {
      const std::set<unsigned> majors = parse_majors(argv[3]);
      verify_devices(argv[2], majors);
      std::cout << "PIREUS_DEVICE_BARRIER_INVENTORY_PASS root=" << argv[2]
                << " majors=" << join_majors(majors) << '\n';
      return 0;
    }
    if (argc != 5) {
      usage(argv[0]);
      return 64;
    }

    const std::string command = argv[1];
    const std::string cgroup_path = argv[2];
    const std::set<unsigned> majors = parse_majors(argv[3]);
    const std::string state_path = argv[4];
    const std::string boot_id =
        read_first_line("/proc/sys/kernel/random/boot_id");
    const int expected_fd = load_program(deny_major_program(majors));
    const ProgramIdentity expected = program_identity(expected_fd);
    struct stat cgroup_metadata{};
    const int cgroup = open_cgroup(cgroup_path, &cgroup_metadata);
    QueryResult query = query_programs(cgroup);
    int attached_fd = -1;

    if (query.ids.size() > 1) {
      close_checked(cgroup);
      close_checked(expected_fd);
      fail("guarded cgroup has more than one direct device program");
    }
    if (query.ids.size() == 1) {
      verify_single_attached(query);
      attached_fd = program_fd_by_id(query.ids.front());
      if (!same_program(program_identity(attached_fd), expected)) {
        close_checked(attached_fd);
        close_checked(cgroup);
        close_checked(expected_fd);
        fail("guarded cgroup has a foreign device program");
      }
    }

    const std::string expected_state =
        state_text(boot_id, cgroup_metadata,
                   attached_fd >= 0 ? program_identity(attached_fd) : expected,
                   majors);

    if (command == "attach") {
      if (attached_fd < 0) {
        attach_program(cgroup, expected_fd, query.revision);
        query = query_programs(cgroup);
        verify_single_attached(query);
        attached_fd = program_fd_by_id(query.ids.front());
        if (!same_program(program_identity(attached_fd), expected)) {
          fail("post-attach device barrier identity mismatch");
        }
      }
      write_state(state_path,
                  state_text(boot_id, cgroup_metadata,
                             program_identity(attached_fd), majors));
      std::cout << "PIREUS_DEVICE_BARRIER state=ATTACHED id="
                << program_identity(attached_fd).id
                << " tag=" << tag_hex(expected)
                << " majors=" << join_majors(majors)
                << " cgroup=" << cgroup_path << '\n';
    } else if (command == "detach") {
      if (attached_fd < 0) fail("expected device barrier is not attached");
      if (read_state(state_path) != expected_state) {
        fail("device barrier state binding mismatch");
      }
      detach_program(cgroup, attached_fd, query.revision);
      if (!query_programs(cgroup).ids.empty()) {
        fail("device barrier remained attached after detach");
      }
      remove_state(state_path);
      std::cout << "PIREUS_DEVICE_BARRIER state=DETACHED tag="
                << tag_hex(expected) << " majors=" << join_majors(majors)
                << " cgroup=" << cgroup_path << '\n';
    } else if (command == "status-attached") {
      if (attached_fd < 0) fail("expected device barrier is not attached");
      if (read_state(state_path) != expected_state) {
        fail("device barrier state binding mismatch");
      }
      std::cout << "PIREUS_DEVICE_BARRIER state=ATTACHED id="
                << program_identity(attached_fd).id
                << " tag=" << tag_hex(expected)
                << " majors=" << join_majors(majors)
                << " cgroup=" << cgroup_path << '\n';
    } else if (command == "status-detached") {
      if (attached_fd >= 0) fail("device barrier is unexpectedly attached");
      errno = 0;
      if (access(state_path.c_str(), F_OK) == 0 || errno != ENOENT) {
        fail("detached device barrier has stale or unreadable state");
      }
      std::cout << "PIREUS_DEVICE_BARRIER state=DETACHED tag="
                << tag_hex(expected) << " majors=" << join_majors(majors)
                << " cgroup=" << cgroup_path << '\n';
    } else {
      usage(argv[0]);
      close_checked(attached_fd);
      close_checked(cgroup);
      close_checked(expected_fd);
      return 64;
    }

    close_checked(attached_fd);
    close_checked(cgroup);
    close_checked(expected_fd);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "pireus-device-barrier: " << error.what() << '\n';
    return 42;
  }
}
