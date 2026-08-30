// C++ MATERIAL_PARITY for the frozen Sounio Pireus Operator Novelty Frontier v11.
// It reconstructs the exhaustive finite census without selecting an operator or
// promoting the relative novelty certificate beyond the frozen Sounio meaning.

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

constexpr int kLanes = 16;
constexpr int kTensorCells = 4096;
constexpr int kCandidates = 7200;
constexpr int kRepresentatives = 6;
constexpr std::array<int, kRepresentatives> kExpectedAtlasSupports = {
    0, 176, 512, 474, 96, 272};
constexpr std::array<std::uint32_t, 8> kSeedWords = {
    0U, 0U, 1010580540U, 4042322160U,
    2863311530U, 2863311530U, 2526451350U, 1515870810U};

constexpr const char* kSounioSourceSha =
    "9289cd504385e2f1f4eed095d82a963cf2e5e67124bf8d267d1bc6ccda7ac36b";
constexpr const char* kSounioSemanticsSha =
    "f1e339ec7bc290f412d42bba3fa1ba609fd89947408ea422ab96026cce5883dc";
constexpr const char* kParentSemanticsSha =
    "2640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5";
constexpr const char* kFormalReceiptSha =
    "b56b1f331879c2a8bbb70dc0adfc5ac61e21e922834c391ce4d815397a589d21";
constexpr const char* kEffectReceiptSha =
    "b18f91987a5b169bebb1a02d3b200f4ecae513c28f83f16dabaf3a96f2524d71";

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

struct Profile {
  int mismatch_count = 0;
  int first_cell = -1;
  int second_cell = -1;
  std::int64_t first_delta = 0;
  std::int64_t second_delta = 0;
};

struct Census {
  std::array<int, kRepresentatives> atlas_supports{};
  int codec_checks = 0;
  int codec_failures = 0;
  int unit_boundary_checks = 0;
  int one_sparse_candidates = 0;
  int atlas_collision_candidates = 0;
  int atlas_collision_edges = 0;
  int n2_relative_novelty = 0;
  int separators = 0;
  int separator_failures = 0;
  int separator_formula_checks = 0;
  int c2_permutation_checks = 0;
  int c2_character_checks = 0;
  int c2_involution_checks = 0;
  int c2_failures = 0;
  int action_base_support = 0;
  int transported_inside_grammar = 0;
  int transported_outside_grammar = 0;
  int transported_over_base_support = 0;
  int transported_outside_base_support = 0;
  int quotient_outside = 0;
  int quotient_fixed = 0;
  int quotient_pairs = 0;
  int quotient_singletons = 0;
  int quotient_classes = 0;
  int quotient_in_grammar_images = 0;
  int direct_quotient_checks = 0;
  int quotient_failures = 0;
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

template <typename T>
std::string join(const std::vector<T>& values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << values[i];
  }
  return out.str();
}

std::string join(const std::array<int, kRepresentatives>& values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ':';
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
  return (input0 * kLanes + input1) * kLanes + output;
}

int cell_output(int cell) { return cell % kLanes; }
int cell_input1(int cell) { return (cell / kLanes) % kLanes; }
int cell_input0(int cell) { return cell / (kLanes * kLanes); }

int permute_lane(int value) {
  if (value == 0) return 1;
  if (value == 1) return 0;
  return value;
}

int permute_cell(int cell) {
  return tensor_index(permute_lane(cell_output(cell)),
                      permute_lane(cell_input0(cell)),
                      permute_lane(cell_input1(cell)));
}

int sign_lane(int value) { return value >= 0 && value < kLanes ? 1 : 0; }

int result_character(int cell) {
  return sign_lane(cell_output(cell)) * sign_lane(cell_input0(cell)) *
         sign_lane(cell_input1(cell));
}

int basis_character(int cell) { return result_character(permute_cell(cell)); }

int seed_bit(int input0, int input1) {
  const int cell = input0 * kLanes + input1;
  return static_cast<int>((kSeedWords[static_cast<size_t>(cell / 32)] >>
                           (cell % 32)) & 1U);
}

std::int64_t signed_coefficient(int value) { return value == 0 ? 1 : -1; }

std::int64_t parent_coefficient(int cell) {
  const int output = cell_output(cell);
  const int input0 = cell_input0(cell);
  const int input1 = cell_input1(cell);
  return output == (input0 ^ input1) ? signed_coefficient(seed_bit(input0, input1)) : 0;
}

bool parent_associator_failure(int input0, int input1, int input2) {
  const auto left =
      parent_coefficient(tensor_index((input0 ^ input1) ^ input2, input0 ^ input1,
                                      input2)) *
      parent_coefficient(tensor_index(input0 ^ input1, input0, input1));
  const auto right =
      parent_coefficient(tensor_index(input0 ^ (input1 ^ input2), input0,
                                      input1 ^ input2)) *
      parent_coefficient(tensor_index(input1 ^ input2, input1, input2));
  return left != right;
}

int candidate_cell(int id) {
  const int coordinate = id / 2;
  const int output = coordinate % 16;
  const int pair = coordinate / 16;
  const int input1 = pair % 15 + 1;
  const int input0 = pair / 15 + 1;
  return tensor_index(output, input0, input1);
}

std::int64_t candidate_delta(int id) { return id % 2 == 0 ? 1 : -1; }

int encode_candidate(int cell, std::int64_t delta) {
  if (cell < 0 || cell >= kTensorCells || (delta != 1 && delta != -1)) return -1;
  const int input0 = cell_input0(cell);
  const int input1 = cell_input1(cell);
  if (input0 < 1 || input0 >= kLanes || input1 < 1 || input1 >= kLanes) return -1;
  const int pair = (input0 - 1) * 15 + (input1 - 1);
  const int coordinate = pair * 16 + cell_output(cell);
  return coordinate * 2 + (delta == 1 ? 0 : 1);
}

struct Model {
  std::array<int, 256> pair_counts{};
  std::array<Profile, kRepresentatives> profiles{};
  std::array<std::int64_t, kTensorCells> action_difference{};

  Model() {
    for (int input0 = 0; input0 < kLanes; ++input0) {
      for (int input1 = 0; input1 < kLanes; ++input1) {
        for (int input2 = 0; input2 < kLanes; ++input2) {
          if (parent_associator_failure(input0, input1, input2)) {
            ++pair_counts[static_cast<size_t>(input0 * kLanes + input1)];
          }
        }
      }
    }
    for (int representative = 0; representative < kRepresentatives; ++representative) {
      for (int cell = 0; cell < kTensorCells; ++cell) {
        const auto delta = atlas(representative, cell) - parent_coefficient(cell);
        if (delta == 0) continue;
        Profile& profile = profiles[static_cast<size_t>(representative)];
        ++profile.mismatch_count;
        if (profile.first_cell < 0) {
          profile.first_cell = cell;
          profile.first_delta = delta;
        } else if (profile.second_cell < 0) {
          profile.second_cell = cell;
          profile.second_delta = delta;
        }
      }
    }
    for (int cell = 0; cell < kTensorCells; ++cell) {
      action_difference[static_cast<size_t>(cell)] =
          result_character(cell) * parent_coefficient(permute_cell(cell)) -
          parent_coefficient(cell);
    }
  }

  static int rotl4(int value) { return ((value << 1) & 15) | ((value >> 3) & 1); }

  std::int64_t atlas_class(int class_id, int cell) const {
    const int output = cell_output(cell);
    const int input0 = cell_input0(cell);
    const int input1 = cell_input1(cell);
    if (class_id == 0) return parent_coefficient(cell);
    if (class_id == 1) {
      return output == (rotl4(input0) ^ input1 ^ 2)
                 ? signed_coefficient(seed_bit(input0, input1))
                 : 0;
    }
    return output == (input0 ^ input1)
               ? signed_coefficient(seed_bit(input0, input1)) *
                     (1 + pair_counts[static_cast<size_t>(input0 * kLanes + input1)])
               : 0;
  }

  std::int64_t atlas(int representative, int cell) const {
    const int class_id = representative / 2;
    const int action = representative % 2;
    return (action == 0 ? 1 : result_character(cell)) *
           atlas_class(class_id, action == 0 ? cell : permute_cell(cell));
  }

  static std::int64_t candidate_value(int id, int cell) {
    return parent_coefficient(cell) + (cell == candidate_cell(id) ? candidate_delta(id) : 0);
  }

  int sparse_separator(int representative, int mutation_cell,
                       std::int64_t mutation_delta) const {
    const Profile& profile = profiles[static_cast<size_t>(representative)];
    if (profile.first_cell < 0) return mutation_cell;
    if (profile.first_cell < mutation_cell) return profile.first_cell;
    if (profile.first_cell > mutation_cell) return mutation_cell;
    if (profile.first_delta != mutation_delta) return mutation_cell;
    return profile.second_cell;
  }
};

Census reconstruct(const Model& model) {
  Census census;
  for (int representative = 0; representative < kRepresentatives; ++representative) {
    census.atlas_supports[static_cast<size_t>(representative)] =
        model.profiles[static_cast<size_t>(representative)].mismatch_count;
  }

  std::array<int, kTensorCells> seen{};
  for (int cell = 0; cell < kTensorCells; ++cell) {
    const int target = permute_cell(cell);
    const int character = basis_character(cell);
    ++census.c2_permutation_checks;
    ++census.c2_character_checks;
    census.c2_involution_checks += 2;
    if (target < 0 || target >= kTensorCells) {
      ++census.c2_failures;
      continue;
    }
    ++seen[static_cast<size_t>(target)];
    if (permute_cell(target) != cell) ++census.c2_failures;
    if (character != 1 || character * basis_character(target) != 1) {
      ++census.c2_failures;
    }
    if (model.action_difference[static_cast<size_t>(cell)] != 0) {
      ++census.action_base_support;
    }
  }
  for (int cell = 0; cell < kTensorCells; ++cell) {
    ++census.c2_permutation_checks;
    if (seen[static_cast<size_t>(cell)] != 1) ++census.c2_failures;
  }

  for (int id = 0; id < kCandidates; ++id) {
    const int mutation_cell = candidate_cell(id);
    const std::int64_t mutation_delta = candidate_delta(id);
    census.codec_checks += 2;
    census.unit_boundary_checks += 2;
    if (encode_candidate(mutation_cell, mutation_delta) != id ||
        cell_input0(mutation_cell) == 0 || cell_input1(mutation_cell) == 0) {
      ++census.codec_failures;
    }

    int candidate_support = 0;
    for (int cell = 0; cell < kTensorCells; ++cell) {
      if (Model::candidate_value(id, cell) != parent_coefficient(cell)) {
        ++candidate_support;
      }
    }
    if (candidate_support == 1) ++census.one_sparse_candidates;

    bool collision = false;
    for (int representative = 0; representative < kRepresentatives; ++representative) {
      const Profile& profile = model.profiles[static_cast<size_t>(representative)];
      const bool sparse_collision = profile.mismatch_count == 1 &&
                                    profile.first_cell == mutation_cell &&
                                    profile.first_delta == mutation_delta;
      int direct_separator = -1;
      for (int cell = 0; cell < kTensorCells; ++cell) {
        if (Model::candidate_value(id, cell) != model.atlas(representative, cell)) {
          direct_separator = cell;
          break;
        }
      }
      if (sparse_collision) {
        collision = true;
        ++census.atlas_collision_edges;
        if (direct_separator >= 0) ++census.separator_failures;
      } else {
        ++census.separators;
        ++census.separator_formula_checks;
        const int sparse_separator =
            model.sparse_separator(representative, mutation_cell, mutation_delta);
        if (direct_separator < 0 || direct_separator != sparse_separator ||
            Model::candidate_value(id, direct_separator) ==
                model.atlas(representative, direct_separator)) {
          ++census.separator_failures;
        }
      }
    }
    if (collision) {
      ++census.atlas_collision_candidates;
    } else {
      ++census.n2_relative_novelty;
    }

    const int target = permute_cell(mutation_cell);
    if (encode_candidate(target, mutation_delta) < 0) {
      ++census.transported_outside_grammar;
    } else {
      ++census.transported_inside_grammar;
    }
    if (model.action_difference[static_cast<size_t>(target)] == 0) {
      ++census.transported_outside_base_support;
    } else {
      ++census.transported_over_base_support;
    }
    int quotient_support = 0;
    int unique_cell = -1;
    std::int64_t unique_value = 0;
    for (int cell = 0; cell < kTensorCells; ++cell) {
      const int source = permute_cell(cell);
      const auto q_candidate = result_character(cell) *
          (parent_coefficient(source) +
           (source == mutation_cell ? mutation_delta : 0));
      const auto difference = q_candidate - parent_coefficient(cell);
      if (difference != 0) {
        ++quotient_support;
        unique_cell = cell;
        unique_value = difference;
      }
    }
    ++census.direct_quotient_checks;
    const int mapped = quotient_support == 1
                           ? encode_candidate(unique_cell, unique_value)
                           : -1;
    if (mapped < 0) {
      ++census.quotient_outside;
    } else if (mapped == id) {
      ++census.quotient_fixed;
    } else if (mapped > id) {
      ++census.quotient_pairs;
    }
  }
  census.quotient_singletons = census.quotient_outside + census.quotient_fixed;
  census.quotient_classes = census.quotient_singletons + census.quotient_pairs;
  census.quotient_in_grammar_images = census.quotient_fixed + 2 * census.quotient_pairs;
  if (census.quotient_singletons + 2 * census.quotient_pairs != kCandidates) {
    ++census.quotient_failures;
  }
  return census;
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
    std::cerr << "usage: operator_novelty_frontier_material_parity "
                 "--target=xeon|apple|dgx24|dgx48|u250\n";
    return 64;
  }
  if (!is_hex_digest(kSounioSourceSha) || !is_hex_digest(kSounioSemanticsSha) ||
      !is_hex_digest(kParentSemanticsSha) || !is_hex_digest(kFormalReceiptSha) ||
      !is_hex_digest(kEffectReceiptSha)) {
    std::cerr << "frozen receipt identity is malformed\n";
    return 65;
  }

  const Model model;
  const Census census = reconstruct(model);
  const Platform platform = observe_platform();
  const CudaIdentity cuda = observe_cuda();
  const U250Identity u250 = observe_u250();
  const bool reconstruction_match =
      census.atlas_supports == kExpectedAtlasSupports &&
      census.codec_checks == 14400 && census.codec_failures == 0 &&
      census.unit_boundary_checks == 14400 &&
      census.one_sparse_candidates == 7200 &&
      census.atlas_collision_candidates == 0 && census.atlas_collision_edges == 0 &&
      census.n2_relative_novelty == 7200 && census.separators == 43200 &&
      census.separator_formula_checks == 43200 && census.separator_failures == 0 &&
      census.c2_permutation_checks == 8192 && census.c2_character_checks == 4096 &&
      census.c2_involution_checks == 8192 && census.c2_failures == 0 &&
      census.action_base_support == 176 &&
      census.transported_inside_grammar == 6272 &&
      census.transported_outside_grammar == 928 &&
      census.transported_over_base_support == 228 &&
      census.transported_outside_base_support == 6972 &&
      census.quotient_outside == 7200 &&
      census.quotient_fixed == 0 && census.quotient_pairs == 0 &&
      census.quotient_singletons == 7200 && census.quotient_classes == 7200 &&
      census.quotient_in_grammar_images == 0 &&
      census.direct_quotient_checks == 7200 && census.quotient_failures == 0;

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
                               lower(platform.cpu_model).find("apple") != std::string::npos;
  } else if (target == "dgx24" || target == "dgx48") {
    target_name = target == "dgx24" ? "DGX_GB10_24" : "DGX_GB10_48";
    target_locator = target == "dgx24" ? "cluster-node:spark-3c59"
                                        : "cluster-node:spark-8e54";
    const std::string expected_host = target == "dgx24" ? "spark-3c59" : "spark-8e54";
    target_identity_observed = platform.kernel == "Linux" && platform.arch == "aarch64" &&
                               platform.host == expected_host &&
                               cuda.driver_present && cuda.device_count >= 1 &&
                               cuda.device_name == "NVIDIA GB10" &&
                               cuda.compute_major == 12 && cuda.compute_minor == 1;
  } else {
    target_name = "AMD_ALVEO_U250_DUAL_CARD";
    target_locator = "kubernetes-node:dl380-proxmox";
    target_identity_observed = platform.kernel == "Linux" && platform.arch == "x86_64" &&
                               (platform.host == "dl380-proxmox" ||
                                platform.host == "dl380-dl380-proxmox") &&
                               u250.paired_card_count >= 1;
  }

  std::cout << "schema=pireus-operator-novelty-frontier-material-parity-v11\n";
  std::cout << "producing_language=C++\n";
  std::cout << "producing_role=MATERIAL_PARITY\n";
  std::cout << "authority_language=Sounio\n";
  std::cout << "sounio_source_sha256=" << kSounioSourceSha << '\n';
  std::cout << "sounio_semantics_sha256=" << kSounioSemanticsSha << '\n';
  std::cout << "parent_semantics_sha256=" << kParentSemanticsSha << '\n';
  std::cout << "formal_parity_receipt_sha256=" << kFormalReceiptSha << '\n';
  std::cout << "effect_parity_receipt_sha256=" << kEffectReceiptSha << '\n';
  std::cout << "target=" << target << '\n';
  std::cout << "target_name=" << target_name << '\n';
  std::cout << "target_locator=" << target_locator << '\n';
  std::cout << "legacy_lan_locator="
            << (target == "dgx24" ? "ssh:demetrios@192.168.3.24"
                : target == "dgx48" ? "ssh:demetrios@192.168.3.48" : "") << '\n';
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
  std::cout << "grammar_candidates_declared=7200\n";
  std::cout << "grammar_candidates_evaluated_by_cpp=7200\n";
  std::cout << "grammar_enumeration_performed_by_cpp=true\n";
  std::cout << "codec_checks=" << census.codec_checks << '\n';
  std::cout << "codec_failures=" << census.codec_failures << '\n';
  std::cout << "unit_boundary_checks=" << census.unit_boundary_checks << '\n';
  std::cout << "one_sparse_candidates=" << census.one_sparse_candidates << '\n';
  std::cout << "atlas_representative_supports=" << join(census.atlas_supports) << '\n';
  std::cout << "atlas_collision_candidates=" << census.atlas_collision_candidates << '\n';
  std::cout << "atlas_collision_edges=" << census.atlas_collision_edges << '\n';
  std::cout << "n2_relative_novelty=" << census.n2_relative_novelty << '\n';
  std::cout << "separators=" << census.separators << '\n';
  std::cout << "separator_formula_checks=" << census.separator_formula_checks << '\n';
  std::cout << "separator_failures=" << census.separator_failures << '\n';
  std::cout << "equivalence_group=C2_diag\n";
  std::cout << "c2_permutation_checks=" << census.c2_permutation_checks << '\n';
  std::cout << "c2_character_checks=" << census.c2_character_checks << '\n';
  std::cout << "c2_involution_checks=" << census.c2_involution_checks << '\n';
  std::cout << "c2_failures=" << census.c2_failures << '\n';
  std::cout << "action_base_support=" << census.action_base_support << '\n';
  std::cout << "transported_mutation_inside_grammar="
            << census.transported_inside_grammar << '\n';
  std::cout << "transported_mutation_outside_grammar="
            << census.transported_outside_grammar << '\n';
  std::cout << "transported_mutation_over_base_difference_support="
            << census.transported_over_base_support << '\n';
  std::cout << "transported_mutation_outside_base_difference_support="
            << census.transported_outside_base_support << '\n';
  std::cout << "quotient_outside=" << census.quotient_outside << '\n';
  std::cout << "quotient_fixed=" << census.quotient_fixed << '\n';
  std::cout << "quotient_pairs=" << census.quotient_pairs << '\n';
  std::cout << "quotient_singletons=" << census.quotient_singletons << '\n';
  std::cout << "quotient_classes=" << census.quotient_classes << '\n';
  std::cout << "quotient_in_grammar_images=" << census.quotient_in_grammar_images << '\n';
  std::cout << "direct_quotient_checks=" << census.direct_quotient_checks << '\n';
  std::cout << "quotient_failures=" << census.quotient_failures << '\n';
  std::cout << "material_reconstruction_match="
            << (reconstruction_match ? "true" : "false") << '\n';
  std::cout << "target_identity_observed="
            << (target_identity_observed ? "true" : "false") << '\n';
  std::cout << "material_scope=HOST_CXX_EXHAUSTIVE_FROZEN_VALUE_RECONSTRUCTION_PLUS_TARGET_IDENTITY\n";
  std::cout << "numeric_values_status=EXHAUSTIVE_CPP_RECONSTRUCTION_OF_FROZEN_SOUNIO_VALUES\n";
  std::cout << "analytic_proof_by_cpp=false\n";
  std::cout << "digest_parity_performed_by_cpp=false\n";
  std::cout << "sounio_executable_replayed_by_cpp=false\n";
  std::cout << "cross_language_equivalence_proved_by_cpp=false\n";
  std::cout << "formal_effect_receipt_hashes_verified_by_cpp=false\n";
  std::cout << "native_gpu_operator_kernel_execution=false\n";
  std::cout << "fpga_operator_kernel_execution=false\n";
  std::cout << "semantic_write=false\n";
  std::cout << "expected_result_write=false\n";
  std::cout << "candidate_selected_by_cpp=false\n";
  std::cout << "candidate_selected=false\n";
  std::cout << "n3_novelty=false\n";
  std::cout << "n4_novelty=false\n";
  std::cout << "algorithmic_novelty=false\n";
  std::cout << "material_novelty=false\n";
  std::cout << "scientific_novelty=false\n";
  std::cout << "historical_novelty=false\n";
  std::cout << "priority_claim=false\n";
  std::cout << "claim_ready=false\n";
  std::cout << "result=" << (reconstruction_match && target_identity_observed ? "PASS" : "FAIL")
            << '\n';
  return reconstruction_match && target_identity_observed ? 0 : 2;
}
