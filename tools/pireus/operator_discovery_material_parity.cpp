// C++ MATERIAL_PARITY for the frozen Sounio Pireus Operator Discovery Engine v10.
// This program reconstructs the finite model and observes one target identity.
// It cannot select a different candidate, write semantics, or promote novelty.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
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

constexpr int kDimension = 16;
constexpr int kTensorCells = 4096;
constexpr int kAtlasClasses = 3;
constexpr int kGroupOrder = 2;
constexpr int kSeparatorCapacity = 6;
constexpr int kGrammarCandidates = 7200;
constexpr int kSearchBudget = 64;

constexpr const char* kSounioSourceSha =
    "919b6104cbce1c5f8643f5df88b9071305d3fee854f785ac63a883bc45f16117";
constexpr const char* kSounioSemanticsSha =
    "2640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5";
constexpr const char* kFormalReceiptSha =
    "dddc85352de064baeee09da91917ecc3790ac5fd362ba29b4dc204d86addaa30";
constexpr const char* kEffectReceiptSha =
    "eb8778c8ab7bf1627ef915ef6412bbc3de1e81e0807df7459858a03ecfe4d537";

constexpr std::array<std::uint32_t, 8> kSeedWords = {
    0U, 0U, 1010580540U, 4042322160U,
    2863311530U, 2863311530U, 2526451350U, 1515870810U,
};

enum class SearchOutcome {
  kQuotientCollision,
  kN2RelativeNovelty,
  kSearchIncomplete,
};

const char* search_outcome_name(SearchOutcome outcome) {
  switch (outcome) {
    case SearchOutcome::kQuotientCollision:
      return "QUOTIENT_COLLISION";
    case SearchOutcome::kN2RelativeNovelty:
      return "N2_RELATIVE_NOVELTY";
    case SearchOutcome::kSearchIncomplete:
      return "SEARCH_INCOMPLETE";
  }
  return "INVALID";
}

struct SearchResult {
  SearchOutcome outcome = SearchOutcome::kSearchIncomplete;
  int comparisons_required = kSeparatorCapacity;
  int comparisons_completed = 0;
  int matched_class = -1;
  int matched_action = -1;
  std::vector<int> separators;
  bool complete = false;
};

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

#if !defined(__APPLE__)
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
#endif

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
    if (lower(read_file(root + "/vendor")) != "0x10ee") continue;
    const std::string device = lower(read_file(root + "/device"));
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
    if (management_bdf.size() < 2 ||
        management_bdf.substr(management_bdf.size() - 2) != ".0") continue;
    const std::string slot = management_bdf.substr(0, management_bdf.size() - 2);
    if (std::find(result.user_bdfs.begin(), result.user_bdfs.end(), slot + ".1") !=
        result.user_bdfs.end()) {
      ++result.paired_card_count;
      result.paired_slots.push_back(slot);
    }
  }
  return result;
}

std::string join(const std::vector<int>& values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ':';
    out << values[i];
  }
  return out.str();
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

int tensor_index(int output, int input0, int input1) {
  return (input0 * kDimension + input1) * kDimension + output;
}

int seed_bit(int input0, int input1) {
  const int cell = input0 * kDimension + input1;
  return static_cast<int>((kSeedWords[static_cast<size_t>(cell / 32)] >> (cell % 32)) & 1U);
}

std::int64_t signed_coefficient(int value) { return value == 0 ? 1 : -1; }

std::int64_t parent_coefficient(int output, int input0, int input1) {
  return output == (input0 ^ input1) ? signed_coefficient(seed_bit(input0, input1)) : 0;
}

bool parent_associator_failure(int input0, int input1, int input2) {
  const auto left = parent_coefficient((input0 ^ input1) ^ input2, input0 ^ input1,
                                       input2) *
                    parent_coefficient(input0 ^ input1, input0, input1);
  const auto right = parent_coefficient(input0 ^ (input1 ^ input2), input0,
                                        input1 ^ input2) *
                     parent_coefficient(input1 ^ input2, input1, input2);
  return left != right;
}

struct Model {
  std::array<int, 256> parent_pair_counts{};

  Model() {
    for (int input0 = 0; input0 < kDimension; ++input0) {
      for (int input1 = 0; input1 < kDimension; ++input1) {
        int failures = 0;
        for (int input2 = 0; input2 < kDimension; ++input2) {
          if (parent_associator_failure(input0, input1, input2)) ++failures;
        }
        parent_pair_counts[static_cast<size_t>(input0 * kDimension + input1)] = failures;
      }
    }
  }

  static int rotl4(int value) { return ((value << 1) & 15) | ((value >> 3) & 1); }
  static int permute(int action, int value) {
    if (action == 0) return value;
    if (value == 0) return 1;
    if (value == 1) return 0;
    return value;
  }

  std::int64_t atlas(int class_id, int output, int input0, int input1) const {
    if (class_id == 0) return parent_coefficient(output, input0, input1);
    if (class_id == 1) {
      return output == (rotl4(input0) ^ input1 ^ 2)
                 ? signed_coefficient(seed_bit(input0, input1))
                 : 0;
    }
    if (class_id == 2) {
      return output == (input0 ^ input1)
                 ? signed_coefficient(seed_bit(input0, input1)) *
                       (1 + parent_pair_counts[static_cast<size_t>(input0 * 16 + input1)])
                 : 0;
    }
    return 0;
  }

  std::int64_t representative(int class_id, int action, int output, int input0,
                              int input1) const {
    return atlas(class_id, permute(action, output), permute(action, input0),
                 permute(action, input1));
  }

  static std::int64_t candidate(int output, int input0, int input1) {
    const auto parent = parent_coefficient(output, input0, input1);
    return tensor_index(output, input0, input1) == 272 ? parent + 1 : parent;
  }

  template <typename Left, typename Right>
  static int first_witness(const Left& left, const Right& right) {
    for (int cell = 0; cell < kTensorCells; ++cell) {
      const int output = cell % kDimension;
      const int input1 = (cell / kDimension) % kDimension;
      const int input0 = cell / (kDimension * kDimension);
      if (left(output, input0, input1) != right(output, input0, input1)) return cell;
    }
    return -1;
  }

  template <typename Candidate>
  SearchResult discover(const Candidate& candidate_fn, int budget) const {
    constexpr std::array<std::array<int, 2>, 6> pairs = {
        std::array<int, 2>{0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}};
    SearchResult result;
    for (const auto& pair : pairs) {
      if (budget == 0) return result;
      --budget;
      ++result.comparisons_completed;
      const int witness = first_witness(
          candidate_fn, [&](int output, int input0, int input1) {
            return representative(pair[0], pair[1], output, input0, input1);
          });
      if (witness < 0) {
        result.outcome = SearchOutcome::kQuotientCollision;
        result.matched_class = pair[0];
        result.matched_action = pair[1];
        result.complete = true;
        return result;
      }
      result.separators.push_back(witness);
    }
    if (result.comparisons_completed == kSeparatorCapacity &&
        static_cast<int>(result.separators.size()) == kSeparatorCapacity) {
      result.outcome = SearchOutcome::kN2RelativeNovelty;
      result.complete = true;
    }
    return result;
  }
};

struct Summary {
  int seed_weight = 0;
  int parent_associator_failures = 0;
  int parent_commutator_failures = 0;
  int group_action_checks = 0;
  int group_failures = 0;
  SearchResult candidate_search;
  bool collision_control_exact = false;
  bool incomplete_control_exact = false;
  int commutator_failures = 0;
  int associator_failures = 0;
};

Summary reconstruct(const Model& model) {
  Summary summary;
  for (int input0 = 0; input0 < kDimension; ++input0) {
    for (int input1 = 0; input1 < kDimension; ++input1) {
      summary.seed_weight += seed_bit(input0, input1);
      if (parent_coefficient(input0 ^ input1, input0, input1) !=
          parent_coefficient(input0 ^ input1, input1, input0)) {
        ++summary.parent_commutator_failures;
      }
      for (int input2 = 0; input2 < kDimension; ++input2) {
        if (parent_associator_failure(input0, input1, input2)) {
          ++summary.parent_associator_failures;
        }
      }
    }
  }

  bool group_ok = true;
  for (int action = 0; action < kGroupOrder; ++action) {
    for (int value = 0; value < kDimension; ++value) {
      group_ok = group_ok && Model::permute(action, value) < kDimension &&
                 Model::permute(action, Model::permute(action, value)) == value;
    }
  }
  for (int class_id = 0; class_id < kAtlasClasses; ++class_id) {
    for (int left = 0; left < kGroupOrder; ++left) {
      for (int right = 0; right < kGroupOrder; ++right) {
        for (int cell = 0; cell < kTensorCells; ++cell) {
          ++summary.group_action_checks;
          const int output = cell % kDimension;
          const int input1 = (cell / kDimension) % kDimension;
          const int input0 = cell / (kDimension * kDimension);
          const auto composed = model.atlas(
              class_id, Model::permute(right, Model::permute(left, output)),
              Model::permute(right, Model::permute(left, input0)),
              Model::permute(right, Model::permute(left, input1)));
          const auto multiplied =
              model.representative(class_id, left ^ right, output, input0, input1);
          group_ok = group_ok && composed == multiplied;
        }
      }
    }
  }
  summary.group_failures = group_ok ? 0 : 1;

  summary.candidate_search = model.discover(Model::candidate, kSearchBudget);
  const auto control = model.discover(
      [](int output, int input0, int input1) {
        return parent_coefficient(Model::permute(1, output), Model::permute(1, input0),
                                  Model::permute(1, input1));
      },
      kSeparatorCapacity);
  summary.collision_control_exact =
      control.outcome == SearchOutcome::kQuotientCollision &&
      control.comparisons_completed == 2 && control.matched_class == 0 &&
      control.matched_action == 1 && control.separators == std::vector<int>{0} &&
      control.complete;
  const auto incomplete = model.discover(Model::candidate, 1);
  summary.incomplete_control_exact =
      incomplete.outcome == SearchOutcome::kSearchIncomplete &&
      incomplete.comparisons_completed == 1 && incomplete.matched_class == -1 &&
      incomplete.matched_action == -1 &&
      incomplete.separators == std::vector<int>{272} && !incomplete.complete;

  for (int cell = 0; cell < kTensorCells; ++cell) {
    const int output = cell % kDimension;
    const int input1 = (cell / kDimension) % kDimension;
    const int input0 = cell / (kDimension * kDimension);
    if (Model::candidate(output, input0, input1) !=
        Model::candidate(output, input1, input0)) {
      ++summary.commutator_failures;
    }
  }
  for (int cell = 0; cell < 65536; ++cell) {
    const int output = cell % 16;
    const int input2 = (cell / 16) % 16;
    const int input1 = (cell / 256) % 16;
    const int input0 = cell / 4096;
    std::int64_t left = 0;
    std::int64_t right = 0;
    for (int middle = 0; middle < 16; ++middle) {
      left += Model::candidate(middle, input0, input1) *
              Model::candidate(output, middle, input2);
      right += Model::candidate(middle, input1, input2) *
               Model::candidate(output, input0, middle);
    }
    if (left != right) ++summary.associator_failures;
  }
  return summary;
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
  if (target != "xeon" && target != "apple" && target != "dgx24" &&
      target != "dgx48" && target != "u250") {
    std::cerr << "usage: operator_discovery_material_parity "
                 "--target=xeon|apple|dgx24|dgx48|u250\n";
    return 64;
  }
  if (!is_hex_digest(kSounioSourceSha) || !is_hex_digest(kSounioSemanticsSha) ||
      !is_hex_digest(kFormalReceiptSha) || !is_hex_digest(kEffectReceiptSha)) {
    std::cerr << "frozen receipt identity is malformed\n";
    return 65;
  }

  const Model model;
  const Summary summary = reconstruct(model);
  const Platform platform = observe_platform();
  const CudaIdentity cuda = observe_cuda();
  const U250Identity u250 = observe_u250();
  const std::vector<int> expected_separators = {272, 0, 0, 257, 272, 0};
  const bool reconstruction_match =
      summary.seed_weight == 96 && summary.parent_associator_failures == 768 &&
      summary.parent_commutator_failures == 112 &&
      summary.group_action_checks == 49152 && summary.group_failures == 0 &&
      summary.candidate_search.outcome == SearchOutcome::kN2RelativeNovelty &&
      summary.candidate_search.comparisons_completed == 6 &&
      summary.candidate_search.separators == expected_separators &&
      summary.candidate_search.complete && summary.collision_control_exact &&
      summary.incomplete_control_exact && summary.commutator_failures == 112 &&
      summary.associator_failures == 824;

  bool target_identity_observed = false;
  std::string target_name;
  std::string target_locator;
  if (target == "xeon") {
    target_name = "XEON";
    target_locator = "kubernetes:sounio-workspace-control-0";
    target_identity_observed = platform.kernel == "Linux" && platform.arch == "x86_64" &&
                               lower(platform.cpu_model).find("xeon") != std::string::npos;
  } else if (target == "apple") {
    target_name = "APPLE_SILICON";
    target_locator = "ssh:demetriosagourakis@sounio-language-macbook";
    target_identity_observed = platform.kernel == "Darwin" && platform.arch == "arm64" &&
                               platform.cpu_model == "Apple M5 Max";
  } else if (target == "dgx24" || target == "dgx48") {
    target_name = target == "dgx24" ? "DGX_GB10_24" : "DGX_GB10_48";
    target_locator = target == "dgx24" ? "ssh:demetrios@192.168.3.24"
                                        : "ssh:demetrios@192.168.3.48";
    target_identity_observed = platform.kernel == "Linux" && platform.arch == "aarch64" &&
                               cuda.driver_present && cuda.device_count >= 1 &&
                               cuda.device_name == "NVIDIA GB10" &&
                               cuda.compute_major == 12 && cuda.compute_minor == 1;
  } else {
    target_name = "AMD_ALVEO_U250";
    target_locator = "kubernetes-node:dl380-proxmox";
    target_identity_observed = platform.kernel == "Linux" && platform.arch == "x86_64" &&
                               (platform.host == "dl380-proxmox" ||
                                platform.host == "dl380-dl380-proxmox") &&
                               u250.paired_card_count >= 1;
  }

  std::cout << "schema=pireus-operator-discovery-material-parity-v10\n";
  std::cout << "producing_language=C++\n";
  std::cout << "producing_role=MATERIAL_PARITY\n";
  std::cout << "authority_language=Sounio\n";
  std::cout << "sounio_source_sha256=" << kSounioSourceSha << '\n';
  std::cout << "sounio_semantics_sha256=" << kSounioSemanticsSha << '\n';
  std::cout << "formal_parity_receipt_sha256=" << kFormalReceiptSha << '\n';
  std::cout << "effect_parity_receipt_sha256=" << kEffectReceiptSha << '\n';
  std::cout << "target=" << target << '\n';
  std::cout << "target_name=" << target_name << '\n';
  std::cout << "target_locator=" << target_locator << '\n';
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
  std::cout << "cuda_compute_capability=" << cuda.compute_major << '.'
            << cuda.compute_minor << '\n';
  std::cout << "u250_management_pf_count=" << u250.management_pf_count << '\n';
  std::cout << "u250_user_pf_count=" << u250.user_pf_count << '\n';
  std::cout << "u250_paired_card_count=" << u250.paired_card_count << '\n';
  std::cout << "u250_management_bdfs=" << join(u250.management_bdfs) << '\n';
  std::cout << "u250_user_bdfs=" << join(u250.user_bdfs) << '\n';
  std::cout << "u250_paired_slots=" << join(u250.paired_slots) << '\n';
  std::cout << "u250_declared_card_count=2\n";
  std::cout << "u250_observed_card_count=" << std::min(u250.paired_card_count, 2) << '\n';
  std::cout << "u250_unresolved_card_count=" << 2 - std::min(u250.paired_card_count, 2) << '\n';
  std::cout << "grammar_candidates_declared=" << kGrammarCandidates << '\n';
  std::cout << "grammar_candidates_evaluated_by_cpp=1\n";
  std::cout << "grammar_enumeration_performed_by_cpp=false\n";
  std::cout << "search_budget_declared=" << kSearchBudget << '\n';
  std::cout << "search_comparisons_consumed="
            << summary.candidate_search.comparisons_completed << '\n';
  std::cout << "seed_weight=" << summary.seed_weight << '\n';
  std::cout << "parent_associator_failures=" << summary.parent_associator_failures << '\n';
  std::cout << "parent_commutator_failures=" << summary.parent_commutator_failures << '\n';
  std::cout << "group_action_checks=" << summary.group_action_checks << '\n';
  std::cout << "group_failures=" << summary.group_failures << '\n';
  std::cout << "candidate_id=0\n";
  std::cout << "mutation_tensor_index=272\n";
  std::cout << "mutation_delta=1\n";
  std::cout << "candidate_outcome="
            << search_outcome_name(summary.candidate_search.outcome) << '\n';
  std::cout << "separator_witnesses=" << join(summary.candidate_search.separators) << '\n';
  std::cout << "collision_control_exact="
            << (summary.collision_control_exact ? "true" : "false") << '\n';
  std::cout << "incomplete_control_exact="
            << (summary.incomplete_control_exact ? "true" : "false") << '\n';
  std::cout << "commutator_failures=" << summary.commutator_failures << '\n';
  std::cout << "associator_failures=" << summary.associator_failures << '\n';
  std::cout << "material_reconstruction_match="
            << (reconstruction_match ? "true" : "false") << '\n';
  std::cout << "target_identity_observed="
            << (target_identity_observed ? "true" : "false") << '\n';
  std::cout << "material_scope=HOST_CXX_FROZEN_VALUE_RECONSTRUCTION_PLUS_TARGET_IDENTITY\n";
  std::cout << "candidate_replayed_by_cpp=true\n";
  std::cout << "sounio_executable_replayed_by_cpp=false\n";
  std::cout << "cross_language_equivalence_proved_by_cpp=false\n";
  std::cout << "formal_effect_receipt_hashes_verified_by_cpp=false\n";
  std::cout << "fpga_operator_kernel_execution=false\n";
  std::cout << "semantic_write=false\n";
  std::cout << "expected_result_write=false\n";
  std::cout << "candidate_selected_by_cpp=false\n";
  std::cout << "n3_novelty=false\n";
  std::cout << "n4_novelty=false\n";
  std::cout << "algorithmic_novelty=false\n";
  std::cout << "material_novelty=false\n";
  std::cout << "historical_novelty=false\n";
  std::cout << "priority_claim=false\n";
  std::cout << "claim_ready=false\n";
  std::cout << "result=" << (reconstruction_match && target_identity_observed ? "PASS" : "FAIL")
            << '\n';
  return reconstruction_match && target_identity_observed ? 0 : 2;
}
