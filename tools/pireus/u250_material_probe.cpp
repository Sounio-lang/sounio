// C++ MATERIAL_PARITY probe for the frozen Sounio dual-U250 semantics.
// It emits material facts only. Sounio alone classifies fleet admission.

#include <array>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr char kParentSemanticsSha256[] =
    "9f0fe0bd01baadec0c60b370bf9dd616a6d2063f1f22b7cdf131f2bc9b6f5586";
constexpr char kFreezeSha256[] =
    "db90647e5ce23029699c2c75232ac8e84ccd9818ec597083f6ce56739843f64a";
constexpr char kXbmgmt[] = "/opt/xilinx/xrt/bin/xbmgmt";
constexpr char kXrtSmi[] = "/opt/xilinx/xrt/bin/xrt-smi";
constexpr std::uint64_t kRequiredDdrBytes = 68719476736ULL;

std::string trim(std::string value) {
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r' ||
          value.back() == ' ' || value.back() == '\t')) {
    value.pop_back();
  }
  std::size_t start = 0;
  while (start < value.size() &&
         (value[start] == ' ' || value[start] == '\t')) {
    ++start;
  }
  return value.substr(start);
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

bool safe_bdf(std::string_view value) {
  if (value.size() != 12) {
    return false;
  }
  for (char character : value) {
    const bool hex = (character >= '0' && character <= '9') ||
                     (character >= 'a' && character <= 'f') ||
                     (character >= 'A' && character <= 'F');
    if (!hex && character != ':' && character != '.') {
      return false;
    }
  }
  return value[4] == ':' && value[7] == ':' && value[10] == '.';
}

int run_tool(const char *tool, const std::vector<std::string> &arguments) {
  std::vector<char *> argv;
  argv.reserve(arguments.size() + 2);
  argv.push_back(const_cast<char *>(tool));
  for (const std::string &argument : arguments) {
    argv.push_back(const_cast<char *>(argument.c_str()));
  }
  argv.push_back(nullptr);

  const pid_t child = fork();
  if (child < 0) {
    return errno == 0 ? 126 : errno;
  }
  if (child == 0) {
    const int null_fd = open("/dev/null", O_WRONLY);
    if (null_fd >= 0) {
      (void)dup2(null_fd, STDOUT_FILENO);
      (void)dup2(null_fd, STDERR_FILENO);
      if (null_fd > STDERR_FILENO) {
        (void)close(null_fd);
      }
    }
    execv(tool, argv.data());
    _exit(127);
  }
  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) {
      return errno == 0 ? 125 : errno;
    }
  }
  if (!WIFEXITED(status)) {
    return 128;
  }
  return WEXITSTATUS(status);
}

std::string json_string_after(const std::string &json,
                              std::string_view key,
                              std::size_t start = 0) {
  const std::string marker = "\"" + std::string(key) + "\"";
  const std::size_t key_position = json.find(marker, start);
  if (key_position == std::string::npos) {
    return {};
  }
  const std::size_t colon = json.find(':', key_position + marker.size());
  if (colon == std::string::npos) {
    return {};
  }
  const std::size_t quote = json.find('"', colon + 1);
  if (quote == std::string::npos) {
    return {};
  }
  const std::size_t end = json.find('"', quote + 1);
  if (end == std::string::npos) {
    return {};
  }
  return json.substr(quote + 1, end - quote - 1);
}

std::size_t count_occurrences(const std::string &text,
                              std::string_view needle) {
  std::size_t count = 0;
  std::size_t position = 0;
  while ((position = text.find(needle, position)) != std::string::npos) {
    ++count;
    position += needle.size();
  }
  return count;
}

std::uint64_t parse_hex(std::string value) {
  value = trim(value);
  if (value.rfind("0x", 0) == 0) {
    value.erase(0, 2);
  }
  if (value.empty()) {
    return 0;
  }
  std::uint64_t result = 0;
  for (char character : value) {
    std::uint64_t digit = 0;
    if (character >= '0' && character <= '9') {
      digit = static_cast<std::uint64_t>(character - '0');
    } else if (character >= 'a' && character <= 'f') {
      digit = static_cast<std::uint64_t>(character - 'a' + 10);
    } else if (character >= 'A' && character <= 'F') {
      digit = static_cast<std::uint64_t>(character - 'A' + 10);
    } else {
      return 0;
    }
    result = result * 16 + digit;
  }
  return result;
}

std::size_t count_pci_functions(std::uint64_t vendor,
                                std::uint64_t device) {
  std::size_t count = 0;
  const std::filesystem::path root = "/sys/bus/pci/devices";
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(root, error)) {
    if (error) {
      return 0;
    }
    const std::uint64_t observed_vendor =
        parse_hex(read_file(entry.path() / "vendor"));
    const std::uint64_t observed_device =
        parse_hex(read_file(entry.path() / "device"));
    if (observed_vendor == vendor && observed_device == device) {
      ++count;
    }
  }
  return count;
}

void emit_bool(const char *key, bool value) {
  std::cout << key << '=' << (value ? "true" : "false") << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3 || !safe_bdf(argv[1]) || !safe_bdf(argv[2])) {
    std::cerr << "usage: u250_material_probe MGMT_BDF USER_BDF\n";
    return 64;
  }
  const std::string management_bdf = argv[1];
  const std::string user_bdf = argv[2];
  const std::filesystem::path management_path =
      std::filesystem::path("/sys/bus/pci/devices") / management_bdf;
  const std::filesystem::path user_path =
      std::filesystem::path("/sys/bus/pci/devices") / user_bdf;
  const std::filesystem::path temporary =
      std::filesystem::path("/tmp") /
      ("pireus-u250-material-" + std::to_string(getpid()));
  std::filesystem::create_directories(temporary);
  const std::filesystem::path management_json = temporary / "management.json";
  const std::filesystem::path user_json = temporary / "user.json";

  const int xbmgmt_rc = run_tool(kXbmgmt, {
      "examine", "-d", management_bdf, "-r", "platform",
      "-f", "JSON", "-o", management_json.string(),
  });
  const int xrt_smi_rc = run_tool(kXrtSmi, {
      "examine", "-d", user_bdf, "-r", "platform", "-r", "memory",
      "-f", "JSON", "-o", user_json.string(),
  });

  const std::string management = read_file(management_json);
  const std::string user = read_file(user_json);
  const std::string version = read_file("/opt/xilinx/xrt/version.json");
  std::error_code remove_error;
  std::filesystem::remove_all(temporary, remove_error);

  std::array<char, 256> hostname{};
  const bool hostname_valid =
      gethostname(hostname.data(), hostname.size() - 1) == 0;
  const std::string serial = json_string_after(management, "serial_num");
  const std::string board_type = json_string_after(management, "board_type");
  const std::string board_name = json_string_after(management, "board_name");
  const std::string management_vbnv = json_string_after(management, "vbnv");
  const std::string user_vbnv = json_string_after(user, "vbnv");
  const std::string logic_uuid = json_string_after(user, "logic_uuid");
  const std::string xrt_version = json_string_after(version, "BUILD_VERSION");
  const std::string xrt_branch = json_string_after(version, "BUILD_BRANCH");
  const std::string xrt_hash = json_string_after(version, "VERSION_HASH");

  const std::uint64_t management_vendor =
      parse_hex(read_file(management_path / "vendor"));
  const std::uint64_t management_device =
      parse_hex(read_file(management_path / "device"));
  const std::uint64_t user_vendor =
      parse_hex(read_file(user_path / "vendor"));
  const std::uint64_t user_device =
      parse_hex(read_file(user_path / "device"));
  const bool management_ready =
      parse_hex(read_file(management_path / "ready")) == 1;
  const bool user_ready = parse_hex(read_file(user_path / "ready")) == 1;
  const std::size_t management_pf_count = count_pci_functions(0x10ee, 0x5004);
  const std::size_t user_pf_count = count_pci_functions(0x10ee, 0x5005);

  std::size_t ddr_bank_count = 0;
  std::uint64_t ddr_bytes = 0;
  for (int bank = 0; bank < 4; ++bank) {
    const std::string tag = "\"tag\": \"bank" + std::to_string(bank) + "\"";
    const std::size_t position = user.find(tag);
    if (position == std::string::npos) {
      continue;
    }
    ++ddr_bank_count;
    const std::string range = json_string_after(user, "range_bytes", position);
    ddr_bytes += parse_hex(range);
  }

  const bool facts_valid = xbmgmt_rc == 0 && xrt_smi_rc == 0 &&
      hostname_valid && std::string(hostname.data()) == "dl380-proxmox" &&
      serial == "22000321B01F" && board_type == "u250" &&
      board_name == "ALVEO U250 PQ" &&
      management_vbnv == "xilinx_u250_gen3x16_xdma_shell_4_1" &&
      user_vbnv == management_vbnv && !logic_uuid.empty() &&
      xrt_version == "2.23.0" && xrt_branch == "2026.1" &&
      !xrt_hash.empty() && management_vendor == 0x10ee &&
      user_vendor == 0x10ee && management_device == 0x5004 &&
      user_device == 0x5005 && management_ready && user_ready &&
      management_pf_count == 1 && user_pf_count == 1 &&
      ddr_bank_count == 4 && ddr_bytes >= kRequiredDdrBytes &&
      count_occurrences(management, "\"serial_num\"") == 1;

  std::cout << "schema=pireus.u250.material-probe.v1\n";
  std::cout << "producer_language=C++\n";
  std::cout << "producer_role=MATERIAL_PARITY\n";
  std::cout << "semantic_authority_language=Sounio\n";
  std::cout << "parent_semantics_sha256=" << kParentSemanticsSha256 << '\n';
  std::cout << "freeze_sha256=" << kFreezeSha256 << '\n';
  std::cout << "slot=0\n";
  std::cout << "target_family=AMD_ALVEO_U250\n";
  std::cout << "host=" << (hostname_valid ? hostname.data() : "UNAVAILABLE") << '\n';
  std::cout << "management_bdf=" << management_bdf << '\n';
  std::cout << "user_bdf=" << user_bdf << '\n';
  std::cout << "management_pf_count=" << management_pf_count << '\n';
  std::cout << "user_pf_count=" << user_pf_count << '\n';
  std::cout << "management_pci_vendor=0x" << std::hex << management_vendor << '\n';
  std::cout << "management_pci_device=0x" << std::hex << management_device << '\n';
  std::cout << "user_pci_vendor=0x" << std::hex << user_vendor << '\n';
  std::cout << "user_pci_device=0x" << std::hex << user_device << '\n';
  std::cout << std::dec;
  std::cout << "serial=" << serial << '\n';
  emit_bool("physical_identity_present", !serial.empty());
  std::cout << "board_type=" << board_type << '\n';
  std::cout << "board_name=" << board_name << '\n';
  std::cout << "shell=" << user_vbnv << '\n';
  std::cout << "logic_uuid=" << logic_uuid << '\n';
  emit_bool("management_ready", management_ready);
  emit_bool("user_ready", user_ready);
  std::cout << "xrt_version=" << xrt_version << '\n';
  std::cout << "xrt_branch=" << xrt_branch << '\n';
  std::cout << "xrt_hash=" << xrt_hash << '\n';
  std::cout << "ddr_bank_count=" << ddr_bank_count << '\n';
  std::cout << "ddr_bytes=" << ddr_bytes << '\n';
  std::cout << "xbmgmt_exit=" << xbmgmt_rc << '\n';
  std::cout << "xrt_smi_exit=" << xrt_smi_rc << '\n';
  emit_bool("material_execution_observed", true);
  emit_bool("material_probe_valid", facts_valid);
  emit_bool("semantic_verdict_emitted", false);
  emit_bool("classification_requested", false);
  emit_bool("cost_present", false);
  emit_bool("speedup_present", false);
  emit_bool("kernel_correctness_present", false);
  emit_bool("parity_open", false);
  emit_bool("claim_ready", false);
  return facts_valid ? 0 : 1;
}
