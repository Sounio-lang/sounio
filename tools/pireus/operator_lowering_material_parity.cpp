// C++ MATERIAL_PARITY for the frozen Sounio Operator-Lowering Forge v6.
// This executable observes the local Xeon identity and binds prior target
// identity receipts. It never executes a generated lowering or turns target
// presence into semantic, cost, performance, or novelty evidence.

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/utsname.h>
#include <unistd.h>

namespace {

constexpr const char* kSounioSourceSha =
    "178663aa64bc44938afbe88874268d8078ee1d56e312add965d1470bb3b42ae0";
constexpr const char* kSounioSemanticsSha =
    "bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1";
constexpr const char* kFormalReceiptSha =
    "31f229664a627134898d476d3e5374cd7458401420f49316e129ea951386d169";
constexpr const char* kEffectReceiptSha =
    "9deba7c7f66d9e75e82dbfce7b0ed65e94713f602d0cce6a8190218c5b32629f";
constexpr const char* kParentMaterialReceiptSha =
    "c9a09126ff8f0de58d4054a201f5bcfcf39d998d4087a02ea949ee578b4623b5";
constexpr const char* kParentXeonEvidenceSha =
    "35207450acd83578c2584a316def2b5db4090c620b009529a78526df2721af90";
constexpr const char* kParentAppleEvidenceSha =
    "6f832c8bdac679bf010b3e1dc133222d27a49b8173535170ae0652abba0f17ab";
constexpr const char* kParentDgxEvidenceSha =
    "c431deff96ca855062c61ba0b2922b368e3bf5da30c6709f838e2a4848ec0bf9";
constexpr const char* kParentU250EvidenceSha =
    "4e7f26d70e65ec7a449e48be7e3a7dbfb4886ea575cfa995e787dd5abfba5b3f";

struct Platform {
  std::string host;
  std::string kernel;
  std::string release;
  std::string arch;
  std::string cpu_model;
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

std::string first_cpu_model() {
  std::ifstream input("/proc/cpuinfo");
  std::string line;
  while (std::getline(input, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    if (trim(line.substr(0, colon)) == "model name") {
      return trim(line.substr(colon + 1));
    }
  }
  return "";
}

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
  result.cpu_model = first_cpu_model();
  return result;
}

bool is_hex_digest(const char* value) {
  if (std::strlen(value) != 64) return false;
  for (const char* p = value; *p != '\0'; ++p) {
    if (!std::isxdigit(static_cast<unsigned char>(*p))) return false;
  }
  return true;
}

bool all_receipt_hashes_canonical() {
  const char* hashes[] = {
      kSounioSourceSha,
      kSounioSemanticsSha,
      kFormalReceiptSha,
      kEffectReceiptSha,
      kParentMaterialReceiptSha,
      kParentXeonEvidenceSha,
      kParentAppleEvidenceSha,
      kParentDgxEvidenceSha,
      kParentU250EvidenceSha,
  };
  for (const char* hash : hashes) {
    if (!is_hex_digest(hash)) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || std::string(argv[1]) != "--target=xeon") {
    std::cerr << "usage: operator_lowering_material_parity --target=xeon\n";
    return 64;
  }
  if (!all_receipt_hashes_canonical()) {
    std::cerr << "frozen receipt identity is malformed\n";
    return 65;
  }
  constexpr int kCandidateCells = 1120;
  constexpr int kProgramClasses = 560;
  constexpr int kSerializationsPerProgram = 2;
  constexpr int kTargetEnvelopes = 4;
  constexpr int kTargetCandidateCells = 280;
  if (kCandidateCells != kProgramClasses * kSerializationsPerProgram ||
      kCandidateCells != kTargetEnvelopes * kTargetCandidateCells) {
    std::cerr << "frozen atlas accounting is inconsistent\n";
    return 66;
  }

  const Platform platform = observe_platform();
  const bool identity_observed =
      platform.kernel == "Linux" && platform.arch == "x86_64" &&
      lower(platform.cpu_model).find("xeon") != std::string::npos;

  std::cout << "schema=pireus-operator-lowering-material-identity-evidence-v6.1\n";
  std::cout << "producing_language=C++\n";
  std::cout << "producing_role=MATERIAL_PARITY\n";
  std::cout << "authority_language=Sounio\n";
  std::cout << "sounio_source_sha256=" << kSounioSourceSha << '\n';
  std::cout << "sounio_semantics_sha256=" << kSounioSemanticsSha << '\n';
  std::cout << "formal_parity_receipt_sha256=" << kFormalReceiptSha << '\n';
  std::cout << "effect_parity_receipt_sha256=" << kEffectReceiptSha << '\n';
  std::cout << "parent_material_receipt_sha256=" << kParentMaterialReceiptSha << '\n';
  std::cout << "parent_xeon_evidence_sha256=" << kParentXeonEvidenceSha << '\n';
  std::cout << "parent_apple_evidence_sha256=" << kParentAppleEvidenceSha << '\n';
  std::cout << "parent_dgx_evidence_sha256=" << kParentDgxEvidenceSha << '\n';
  std::cout << "parent_u250_evidence_sha256=" << kParentU250EvidenceSha << '\n';
  std::cout << "target_id=726101\n";
  std::cout << "target_name=XEON\n";
  std::cout << "target_locator=kubernetes:sounio-workspace-control-0\n";
  std::cout << "hostname=" << platform.host << '\n';
  std::cout << "kernel=" << platform.kernel << '\n';
  std::cout << "kernel_release=" << platform.release << '\n';
  std::cout << "architecture=" << platform.arch << '\n';
  std::cout << "cpu_model=" << platform.cpu_model << '\n';
  std::cout << "target_identity_observed=" << (identity_observed ? "true" : "false")
            << '\n';
  std::cout << "candidate_cells=" << kCandidateCells << '\n';
  std::cout << "program_classes=" << kProgramClasses << '\n';
  std::cout << "serializations_per_program=" << kSerializationsPerProgram << '\n';
  std::cout << "candidate_program_quotient_exact=true\n";
  std::cout << "target_envelopes=" << kTargetEnvelopes << '\n';
  std::cout << "target_candidate_cells=" << kTargetCandidateCells << '\n';
  std::cout << "target_population_partition_exact=true\n";
  std::cout << "parent_target_identity_classes_bound=4\n";
  std::cout << "declared_physical_endpoints=6\n";
  std::cout << "parent_observed_physical_endpoints=4\n";
  std::cout << "unresolved_physical_endpoints=2\n";
  std::cout << "unresolved_endpoint_01=DGX_SPARK_192.168.3.48\n";
  std::cout << "unresolved_endpoint_02=AMD_ALVEO_U250_SLOT_1\n";
  std::cout << "typed_residuals=1120\n";
  std::cout << "compiler_emission_unresolved=1120\n";
  std::cout << "material_execution_unresolved=1120\n";
  std::cout << "cost_performance_unresolved=1120\n";
  std::cout << "v6_lowering_obligations_discharged_by_parent_receipts=0\n";
  std::cout << "admitted_lowerings=0\n";
  std::cout << "material_observer_processes_launched=1\n";
  std::cout << "generated_lowering_processes_launched=0\n";
  std::cout << "remote_target_processes_launched=0\n";
  std::cout << "lowering_cost_present=false\n";
  std::cout << "performance_present=false\n";
  std::cout << "cross_target_ranking_present=false\n";
  std::cout << "semantic_write=false\n";
  std::cout << "expected_result_write=false\n";
  std::cout << "parent_material_receipts_promotable_to_semantic_authority=false\n";
  std::cout << "material_identity_accounting_recorded=true\n";
  std::cout << "material_parity_complete=false\n";
  std::cout << "material_parity_incomplete_reason=NO_GENERATED_LOWERING_EXECUTED\n";
  std::cout << "material_target_coverage_complete=false\n";
  std::cout << "material_lowering_coverage_complete=false\n";
  std::cout << "observer_check_01=receipt_hash_shapes status=CHECKED\n";
  std::cout << "observer_check_02=candidate_program_quotient status=CHECKED\n";
  std::cout << "observer_check_03=target_population_partition status=CHECKED\n";
  std::cout << "observer_check_04=xeon_identity status=CHECKED\n";
  std::cout << "checked_observer_predicates=4\n";
  std::cout << "selected_candidate=-1\n";
  std::cout << "claim_ready=false\n";

  if (!identity_observed) {
    std::cerr << "target identity mismatch for xeon\n";
    return 2;
  }
  return 0;
}
