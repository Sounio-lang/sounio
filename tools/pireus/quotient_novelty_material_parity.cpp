// C++ MATERIAL_PARITY for the frozen Sounio quotient-novelty forge v5.
// This executable observes hardware identity only. It cannot select an
// operator, write semantics or expected results, or emit cost/performance data.

#include <algorithm>
#include <cctype>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/utsname.h>
#include <unistd.h>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace {

constexpr const char* kSounioSourceSha =
    "791d85d4b336d854c6ed3b2e662e8f09b05f8a6f6d1dc4c03807c87150751667";
constexpr const char* kSounioSemanticsSha =
    "9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21";
constexpr const char* kFormalReceiptSha =
    "cff661497206523f273613e07fd8455ba7f036c62853b3b978c1bc29aa527593";
constexpr const char* kEffectReceiptSha =
    "09dce0400a1fdac8876bdd136c011bc06db6e2651b1da6d39b3b414ba7e7330e";

constexpr const char* kAppleReceiptSha =
    "c00a3d4e556688829efadbbf640ea858cfe9520dc04103fa745cf1a8101f7840";
constexpr const char* kDgxReceiptSha =
    "3c10882eff43d3b197428839996c7a04c009c8f537d0c1451bdf3e8a13e2f385";
constexpr const char* kU250ReceiptSha =
    "9889567b684fcc0213ed38a44041e8475c4c9a71722b7baa1c6c064e1f1d0d7a";

struct Platform {
  std::string host;
  std::string kernel;
  std::string release;
  std::string arch;
  std::string cpu_model;
  std::string machine_model;
};

struct CudaIdentity {
  bool driver_present = false;
  int device_count = 0;
  int driver_version = 0;
  int compute_major = 0;
  int compute_minor = 0;
  std::string device_name;
};

struct U250Identity {
  int management_pf_count = 0;
  int user_pf_count = 0;
  int paired_card_count = 0;
  std::vector<std::string> management_bdfs;
  std::vector<std::string> user_bdfs;
  std::vector<std::string> paired_slots;
};

std::string trim(std::string value) {
  const auto discarded = [](unsigned char c) {
    return c == '\0' || std::isspace(c) != 0;
  };
  const auto first = std::find_if_not(value.begin(), value.end(), discarded);
  if (first == value.end()) return "";
  const auto last = std::find_if_not(value.rbegin(), value.rend(), discarded).base();
  return std::string(first, last);
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return "";
  std::ostringstream out;
  out << input.rdbuf();
  return trim(out.str());
}

std::string first_cpu_field(const std::vector<std::string>& keys) {
  std::ifstream input("/proc/cpuinfo");
  std::string line;
  while (std::getline(input, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = trim(line.substr(0, colon));
    for (const auto& expected : keys) {
      if (key == expected) return trim(line.substr(colon + 1));
    }
  }
  return "";
}

#if defined(__APPLE__)
std::string sysctl_string(const char* name) {
  size_t size = 0;
  if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) return "";
  std::vector<char> value(size, '\0');
  if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) return "";
  return trim(std::string(value.data(), strnlen(value.data(), value.size())));
}
#endif

Platform observe_platform() {
  Platform result;
  char hostname[256] = {};
  if (gethostname(hostname, sizeof(hostname) - 1) == 0) result.host = hostname;

  struct utsname uts {};
  if (uname(&uts) == 0) {
    result.kernel = uts.sysname;
    result.release = uts.release;
    result.arch = uts.machine;
  }

#if defined(__APPLE__)
  result.cpu_model = sysctl_string("machdep.cpu.brand_string");
  result.machine_model = sysctl_string("hw.model");
#else
  result.cpu_model = first_cpu_field({"model name", "Model", "Hardware", "Processor"});
  result.machine_model = read_file("/proc/device-tree/model");
#endif
  return result;
}

CudaIdentity observe_cuda() {
  CudaIdentity result;
  void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) return result;

  using CuInit = int (*)(unsigned int);
  using CuDeviceGetCount = int (*)(int*);
  using CuDeviceGet = int (*)(int*, int);
  using CuDeviceGetName = int (*)(char*, int, int);
  using CuDeviceComputeCapability = int (*)(int*, int*, int);
  using CuDriverGetVersion = int (*)(int*);

  const auto cu_init = reinterpret_cast<CuInit>(dlsym(library, "cuInit"));
  const auto cu_device_get_count =
      reinterpret_cast<CuDeviceGetCount>(dlsym(library, "cuDeviceGetCount"));
  const auto cu_device_get = reinterpret_cast<CuDeviceGet>(dlsym(library, "cuDeviceGet"));
  const auto cu_device_get_name =
      reinterpret_cast<CuDeviceGetName>(dlsym(library, "cuDeviceGetName"));
  const auto cu_device_compute_capability = reinterpret_cast<CuDeviceComputeCapability>(
      dlsym(library, "cuDeviceComputeCapability"));
  const auto cu_driver_get_version =
      reinterpret_cast<CuDriverGetVersion>(dlsym(library, "cuDriverGetVersion"));

  if (cu_init == nullptr || cu_device_get_count == nullptr || cu_device_get == nullptr ||
      cu_device_get_name == nullptr || cu_device_compute_capability == nullptr ||
      cu_driver_get_version == nullptr || cu_init(0) != 0) {
    dlclose(library);
    return result;
  }

  result.driver_present = true;
  cu_driver_get_version(&result.driver_version);
  if (cu_device_get_count(&result.device_count) == 0 && result.device_count > 0) {
    int device = 0;
    char name[256] = {};
    if (cu_device_get(&device, 0) == 0) {
      cu_device_get_name(name, sizeof(name), device);
      cu_device_compute_capability(&result.compute_major, &result.compute_minor, device);
      result.device_name = name;
    }
  }
  dlclose(library);
  return result;
}

U250Identity observe_u250() {
  U250Identity result;
  DIR* directory = opendir("/sys/bus/pci/devices");
  if (directory == nullptr) return result;
  while (const auto* entry = readdir(directory)) {
    if (entry->d_name[0] == '.') continue;
    const std::string bdf = entry->d_name;
    const std::string root = "/sys/bus/pci/devices/" + bdf;
    const std::string vendor = lower(read_file(root + "/vendor"));
    const std::string device = lower(read_file(root + "/device"));
    if (vendor != "0x10ee") continue;
    if (device == "0x5004") {
      ++result.management_pf_count;
      result.management_bdfs.push_back(bdf);
    } else if (device == "0x5005") {
      ++result.user_pf_count;
      result.user_bdfs.push_back(bdf);
    }
  }
  closedir(directory);
  std::sort(result.management_bdfs.begin(), result.management_bdfs.end());
  std::sort(result.user_bdfs.begin(), result.user_bdfs.end());
  for (const auto& management_bdf : result.management_bdfs) {
    if (management_bdf.size() < 2 || management_bdf.substr(management_bdf.size() - 2) != ".0") {
      continue;
    }
    const std::string slot = management_bdf.substr(0, management_bdf.size() - 2);
    if (std::find(result.user_bdfs.begin(), result.user_bdfs.end(), slot + ".1") !=
        result.user_bdfs.end()) {
      ++result.paired_card_count;
      result.paired_slots.push_back(slot);
    }
  }
  return result;
}

std::string join(const std::vector<std::string>& values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << values[i];
  }
  return out.str();
}

bool is_hex_digest(const char* value) {
  if (std::strlen(value) != 64) return false;
  for (const char* p = value; *p != '\0'; ++p) {
    if (!std::isxdigit(static_cast<unsigned char>(*p))) return false;
  }
  return true;
}

std::string argument_value(int argc, char** argv, const std::string& prefix) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument.rfind(prefix, 0) == 0) return argument.substr(prefix.size());
  }
  return "";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string target = argument_value(argc, argv, "--target=");
  if (target != "xeon" && target != "apple" && target != "dgx" && target != "u250") {
    std::cerr << "usage: quotient_novelty_material_parity --target=xeon|apple|dgx|u250\n";
    return 64;
  }

  if (!is_hex_digest(kSounioSourceSha) || !is_hex_digest(kSounioSemanticsSha) ||
      !is_hex_digest(kFormalReceiptSha) || !is_hex_digest(kEffectReceiptSha) ||
      !is_hex_digest(kAppleReceiptSha) || !is_hex_digest(kDgxReceiptSha) ||
      !is_hex_digest(kU250ReceiptSha)) {
    std::cerr << "frozen receipt identity is malformed\n";
    return 65;
  }

  const Platform platform = observe_platform();
  const CudaIdentity cuda = observe_cuda();
  const U250Identity u250 = observe_u250();
  const int observed_u250_cards = std::min(u250.paired_card_count, 2);

  int target_id = 0;
  std::string target_name;
  std::string locator;
  std::string parent_material_receipt = "none";
  bool identity_observed = false;

  if (target == "xeon") {
    target_id = 701200;
    target_name = "DARWIN_XEON";
    locator = "kubernetes:sounio-workspace-control-0";
    identity_observed = platform.kernel == "Linux" && platform.arch == "x86_64" &&
                        lower(platform.cpu_model).find("xeon") != std::string::npos;
  } else if (target == "apple") {
    target_id = 701201;
    target_name = "APPLE_SILICON";
    locator = "ssh:demetriosagourakis@sounio-language-macbook";
    parent_material_receipt = kAppleReceiptSha;
    identity_observed = platform.kernel == "Darwin" && platform.arch == "arm64" &&
                        platform.machine_model == "Mac17,7" &&
                        platform.cpu_model == "Apple M5 Max";
  } else if (target == "dgx") {
    target_id = 701202;
    target_name = "DGX_SPARK";
    locator = "ssh-via-t560:demetrios@192.168.3.24";
    parent_material_receipt = kDgxReceiptSha;
    identity_observed = platform.kernel == "Linux" && platform.arch == "aarch64" &&
                        cuda.driver_present && cuda.device_count >= 1 &&
                        cuda.device_name == "NVIDIA GB10" && cuda.compute_major == 12 &&
                        cuda.compute_minor == 1;
  } else {
    target_id = 711001;
    target_name = "DUAL_AMD_ALVEO_U250";
    locator = "kubernetes-node:dl380-proxmox";
    parent_material_receipt = kU250ReceiptSha;
    identity_observed = platform.kernel == "Linux" && platform.arch == "x86_64" &&
                        platform.host == "dl380-proxmox" &&
                        u250.management_pf_count == 1 && u250.user_pf_count == 1 &&
                        u250.paired_card_count == 1;
  }

  std::cout << "schema=pireus-quotient-novelty-material-parity-v5\n";
  std::cout << "producing_language=C++\n";
  std::cout << "producing_role=MATERIAL_PARITY\n";
  std::cout << "authority_language=Sounio\n";
  std::cout << "sounio_source_sha256=" << kSounioSourceSha << '\n';
  std::cout << "sounio_semantics_sha256=" << kSounioSemanticsSha << '\n';
  std::cout << "formal_parity_receipt_sha256=" << kFormalReceiptSha << '\n';
  std::cout << "effect_parity_receipt_sha256=" << kEffectReceiptSha << '\n';
  std::cout << "parent_material_receipt_sha256=" << parent_material_receipt << '\n';
  std::cout << "target_id=" << target_id << '\n';
  std::cout << "target_name=" << target_name << '\n';
  std::cout << "target_locator=" << locator << '\n';
  std::cout << "hostname=" << platform.host << '\n';
  std::cout << "kernel=" << platform.kernel << '\n';
  std::cout << "kernel_release=" << platform.release << '\n';
  std::cout << "architecture=" << platform.arch << '\n';
  std::cout << "cpu_model=" << platform.cpu_model << '\n';
  std::cout << "machine_model=" << platform.machine_model << '\n';
  std::cout << "cuda_driver_present=" << (cuda.driver_present ? "true" : "false") << '\n';
  std::cout << "cuda_driver_version=" << cuda.driver_version << '\n';
  std::cout << "cuda_device_count=" << cuda.device_count << '\n';
  std::cout << "cuda_device_name=" << cuda.device_name << '\n';
  std::cout << "cuda_compute_capability=" << cuda.compute_major << '.' << cuda.compute_minor
            << '\n';
  std::cout << "u250_management_pf_count=" << u250.management_pf_count << '\n';
  std::cout << "u250_user_pf_count=" << u250.user_pf_count << '\n';
  std::cout << "u250_paired_card_count=" << u250.paired_card_count << '\n';
  std::cout << "u250_management_bdfs=" << join(u250.management_bdfs) << '\n';
  std::cout << "u250_user_bdfs=" << join(u250.user_bdfs) << '\n';
  std::cout << "u250_paired_slots=" << join(u250.paired_slots) << '\n';
  std::cout << "declared_u250_card_count=2\n";
  std::cout << "observed_u250_card_count=" << observed_u250_cards << '\n';
  std::cout << "unresolved_u250_card_count=" << 2 - observed_u250_cards << '\n';
  std::cout << "dgx_48_status=UNRESOLVED\n";
  std::cout << "target_identity_observed=" << (identity_observed ? "true" : "false") << '\n';
  std::cout << "hash_bound_replay_only_after_sounio_freeze=true\n";
  std::cout << "canonical_target_receipt_semantics_fixed=true\n";
  std::cout << "lowering_cost_and_performance_remain_separate=true\n";
  std::cout << "lowering_cost_present=false\n";
  std::cout << "performance_present=false\n";
  std::cout << "cross_target_ranking_present=false\n";
  std::cout << "semantic_write=false\n";
  std::cout << "expected_result_write=false\n";
  std::cout << "no_material_receipt_promoted_to_semantic_authority=true\n";
  std::cout << "material_receipt_promotable_to_semantic_authority=false\n";
  std::cout << "selected_child=-1\n";
  std::cout << "claim_ready=false\n";

  if (!identity_observed) {
    std::cerr << "target identity mismatch for " << target << '\n';
    return 2;
  }
  return 0;
}
