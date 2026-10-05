// C++ MATERIAL_PARITY for the frozen Sounio Pireus Operator Morphogenesis v12.
// It consumes the hash-bound Sounio transcript and reconstructs its published
// finite objects. It neither defines expected semantic results nor promotes a
// material observation into a novelty claim.

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/utsname.h>
#include <unistd.h>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace {

constexpr int kEpochs = 16;
constexpr int kLanes = 16;
constexpr int kInteriorCells = 225;
constexpr int kSignCells = 256;
constexpr int kCertificates = 3552;

constexpr const char* kSounioSourceSha =
    "0a637f7f3ac84ac501be337f22dff37e16a05dbc4a51d2090441b9cba4c8d05c";
constexpr const char* kSounioSemanticsSha =
    "999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4";
constexpr const char* kParentSemanticsSha =
    "e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff";
constexpr const char* kFormalReceiptSha =
    "0eb932b96838383a800f3889a331d16a10886621f29cda9c19e4e1ef74e0077c";
constexpr const char* kEffectReceiptSha =
    "714a662a230f986b934ccf709d782883633deef8e96184057a2782af47e70a5e";
constexpr const char* kTranscriptSha =
    "148dc490e1f6aaaf672e85fd06411755b7521930f3de5998f4c98b32af25f816";

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

std::string read_file_raw(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return "";
  std::ostringstream out;
  out << input.rdbuf();
  return out.str();
}

std::string read_file_trimmed(const std::string& path) {
  return trim(read_file_raw(path));
}

class Sha256 {
 public:
  Sha256()
      : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
               0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

  void update(const std::string& input) {
    for (unsigned char byte : input) {
      block_[used_++] = byte;
      if (used_ == block_.size()) {
        transform();
        bit_length_ += 512;
        used_ = 0;
      }
    }
  }

  std::string finish() {
    const std::uint64_t total_bits = bit_length_ + used_ * 8;
    block_[used_++] = 0x80U;
    if (used_ > 56) {
      while (used_ < 64) block_[used_++] = 0;
      transform();
      used_ = 0;
    }
    while (used_ < 56) block_[used_++] = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
      block_[used_++] = static_cast<std::uint8_t>(total_bits >> shift);
    }
    transform();
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::uint32_t word : state_) out << std::setw(8) << word;
    return out.str();
  }

 private:
  static std::uint32_t rotate_right(std::uint32_t value, int count) {
    return (value >> count) | (value << (32 - count));
  }

  void transform() {
    static constexpr std::array<std::uint32_t, 64> k = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<std::uint32_t, 64> words{};
    for (int i = 0; i < 16; ++i) {
      words[i] = static_cast<std::uint32_t>(block_[i * 4]) << 24 |
                 static_cast<std::uint32_t>(block_[i * 4 + 1]) << 16 |
                 static_cast<std::uint32_t>(block_[i * 4 + 2]) << 8 |
                 static_cast<std::uint32_t>(block_[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotate_right(words[i - 15], 7) ^
                               rotate_right(words[i - 15], 18) ^
                               (words[i - 15] >> 3);
      const std::uint32_t s1 = rotate_right(words[i - 2], 17) ^
                               rotate_right(words[i - 2], 19) ^
                               (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                               rotate_right(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temporary1 = h + s1 + choice + k[i] + words[i];
      const std::uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                               rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> block_{};
  std::size_t used_ = 0;
  std::uint64_t bit_length_ = 0;
};

std::string sha256(const std::string& input) {
  Sha256 digest;
  digest.update(input);
  return digest.finish();
}

struct Record {
  std::string kind;
  std::map<std::string, std::string> fields;
};

void add_fields(const std::string& line, std::map<std::string, std::string>* fields) {
  std::istringstream input(line);
  std::string token;
  while (input >> token) {
    const auto separator = token.find('=');
    if (separator == std::string::npos) continue;
    (*fields)[token.substr(0, separator)] = token.substr(separator + 1);
  }
}

Record collect_record(const std::vector<std::string>& lines, std::size_t* index) {
  Record record;
  const std::string header = trim(lines[*index]);
  const auto space = header.find(' ');
  record.kind = space == std::string::npos ? header : header.substr(0, space);
  add_fields(header, &record.fields);
  ++*index;
  while (*index < lines.size() && !trim(lines[*index]).empty()) {
    const std::string line = trim(lines[*index]);
    if (!line.empty() && line.front() != ':') add_fields(line, &record.fields);
    ++*index;
  }
  return record;
}

bool parse_i64(const std::string& text, std::int64_t* value) {
  errno = 0;
  char* end = nullptr;
  const long long parsed = std::strtoll(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') return false;
  *value = parsed;
  return true;
}

bool field_i64(const Record& record, const std::string& key, std::int64_t* value) {
  const auto found = record.fields.find(key);
  return found != record.fields.end() && parse_i64(found->second, value);
}

struct Microprogram {
  int left = -1;
  int right = -1;
  int destination = -1;
  int ordinal = -1;
  int sign = -1;
};

struct EpochSummary {
  int archive_before = -1;
  int archive_after = -1;
  int orbit_kind = -1;
  int negatives = -1;
  int square_negatives = -1;
  int commutator_defects = -1;
  int associator_defects = -1;
};

struct Transcript {
  std::array<std::array<int, kInteriorCells>, kEpochs> phase{};
  std::array<std::array<int, kInteriorCells>, kEpochs> anf{};
  std::array<std::array<Microprogram, kSignCells>, kEpochs> microprogram{};
  std::array<std::array<std::int64_t, kLanes>, kEpochs> probes{};
  std::array<EpochSummary, kEpochs> epochs{};
  std::map<std::string, std::int64_t> archive;
  std::map<std::string, std::int64_t> proof;
  int genome_records = 0;
  int microprogram_records = 0;
  int probe_records = 0;
  int epoch_records = 0;
  int certificate_records = 0;
  int certificate_failures = 0;
  int parse_failures = 0;
  int negative_controls_passed = -1;
  int negative_controls_total = -1;
  int boundary_proof_complete = -1;
  int boundary_candidate_selected = -1;
  int boundary_claim_ready = -1;
  int summary_error = -1;
  int summary_structural_failures = -1;
  int summary_valid = -1;

  Transcript() {
    for (auto& row : phase) row.fill(-1);
    for (auto& row : anf) row.fill(-1);
    for (auto& row : probes) row.fill(LLONG_MIN);
  }
};

bool bounded(std::int64_t value, int upper) {
  return value >= 0 && value < upper;
}

void store_record_fields(const Record& record,
                         std::map<std::string, std::int64_t>* destination,
                         int* failures) {
  for (const auto& [key, text] : record.fields) {
    std::int64_t value = 0;
    if (parse_i64(text, &value)) (*destination)[key] = value;
  }
  if (destination->empty()) ++*failures;
}

Transcript parse_transcript(const std::string& raw) {
  Transcript result;
  std::istringstream input(raw);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input, line)) lines.push_back(line);
  std::size_t index = 0;
  while (index < lines.size()) {
    const std::string current = trim(lines[index]);
    if (current.rfind("PIREUS_POM_", 0) != 0) {
      ++index;
      continue;
    }
    const Record record = collect_record(lines, &index);
    std::int64_t epoch = -1;
    if (record.kind == "PIREUS_POM_ARCHIVE") {
      store_record_fields(record, &result.archive, &result.parse_failures);
    } else if (record.kind == "PIREUS_POM_PROOF") {
      store_record_fields(record, &result.proof, &result.parse_failures);
    } else if (record.kind == "PIREUS_POM_EPOCH") {
      if (!field_i64(record, "epoch", &epoch) || !bounded(epoch, kEpochs)) {
        ++result.parse_failures;
        continue;
      }
      auto& out = result.epochs[epoch];
      std::int64_t values[7]{};
      const std::array<const char*, 7> keys = {
          "archive_before", "archive_after", "orbit_kind", "negatives",
          "square_negatives", "commutator_defects", "associator_defects"};
      bool valid = true;
      for (std::size_t i = 0; i < keys.size(); ++i) {
        valid = field_i64(record, keys[i], &values[i]) && valid;
      }
      if (!valid || out.archive_before != -1) {
        ++result.parse_failures;
        continue;
      }
      out.archive_before = static_cast<int>(values[0]);
      out.archive_after = static_cast<int>(values[1]);
      out.orbit_kind = static_cast<int>(values[2]);
      out.negatives = static_cast<int>(values[3]);
      out.square_negatives = static_cast<int>(values[4]);
      out.commutator_defects = static_cast<int>(values[5]);
      out.associator_defects = static_cast<int>(values[6]);
      ++result.epoch_records;
    } else if (record.kind == "PIREUS_POM_GENOME") {
      std::int64_t cell = -1;
      std::int64_t phase = -1;
      std::int64_t anf = -1;
      if (!field_i64(record, "epoch", &epoch) ||
          !field_i64(record, "cell", &cell) ||
          !field_i64(record, "phase", &phase) ||
          !field_i64(record, "anf", &anf) || !bounded(epoch, kEpochs) ||
          !bounded(cell, kInteriorCells) || !bounded(phase, 2) || !bounded(anf, 2) ||
          result.phase[epoch][cell] != -1) {
        ++result.parse_failures;
        continue;
      }
      result.phase[epoch][cell] = static_cast<int>(phase);
      result.anf[epoch][cell] = static_cast<int>(anf);
      ++result.genome_records;
    } else if (record.kind == "PIREUS_POM_MICROPROGRAM") {
      std::int64_t entry = -1;
      std::int64_t values[5]{};
      const std::array<const char*, 5> keys = {
          "left", "right", "destination", "ordinal", "sign_bit"};
      bool valid = field_i64(record, "epoch", &epoch) &&
                   field_i64(record, "entry", &entry);
      for (std::size_t i = 0; i < keys.size(); ++i) {
        valid = field_i64(record, keys[i], &values[i]) && valid;
      }
      if (!valid || !bounded(epoch, kEpochs) || !bounded(entry, kSignCells) ||
          result.microprogram[epoch][entry].left != -1) {
        ++result.parse_failures;
        continue;
      }
      result.microprogram[epoch][entry] = {
          static_cast<int>(values[0]), static_cast<int>(values[1]),
          static_cast<int>(values[2]), static_cast<int>(values[3]),
          static_cast<int>(values[4])};
      ++result.microprogram_records;
    } else if (record.kind == "PIREUS_POM_PROBE") {
      std::int64_t lane = -1;
      std::int64_t value = 0;
      if (!field_i64(record, "epoch", &epoch) ||
          !field_i64(record, "lane", &lane) ||
          !field_i64(record, "value", &value) || !bounded(epoch, kEpochs) ||
          !bounded(lane, kLanes) || result.probes[epoch][lane] != LLONG_MIN) {
        ++result.parse_failures;
        continue;
      }
      result.probes[epoch][lane] = value;
      ++result.probe_records;
    } else if (record.kind == "PIREUS_POM_CERTIFICATE") {
      std::int64_t id = -1;
      std::int64_t kind = -1;
      std::int64_t archive_index = -1;
      std::int64_t partner_index = -1;
      std::int64_t cell = -1;
      std::int64_t generated = -1;
      std::int64_t archived = -1;
      const bool valid =
          field_i64(record, "id", &id) && field_i64(record, "kind", &kind) &&
          field_i64(record, "epoch", &epoch) &&
          field_i64(record, "archive_index", &archive_index) &&
          field_i64(record, "partner_index", &partner_index) &&
          field_i64(record, "cell", &cell) &&
          field_i64(record, "generated_bit", &generated) &&
          field_i64(record, "archived_bit", &archived);
      if (!valid || id != result.certificate_records || !bounded(kind, 2) ||
          !bounded(epoch, kEpochs) || !bounded(archive_index, 128) ||
          !bounded(partner_index, 128) || !bounded(cell, kSignCells) ||
          !bounded(generated, 2) || !bounded(archived, 2)) {
        ++result.parse_failures;
      } else {
        if (generated == archived) ++result.certificate_failures;
        ++result.certificate_records;
      }
    } else if (record.kind == "PIREUS_POM_NEGATIVES") {
      std::int64_t passed = -1;
      std::int64_t total = -1;
      if (!field_i64(record, "passed", &passed) ||
          !field_i64(record, "total", &total)) {
        ++result.parse_failures;
      } else {
        result.negative_controls_passed = static_cast<int>(passed);
        result.negative_controls_total = static_cast<int>(total);
      }
    } else if (record.kind == "PIREUS_POM_BOUNDARY") {
      std::int64_t proof_complete = -1;
      std::int64_t selected = -1;
      std::int64_t claim_ready = -1;
      if (!field_i64(record, "proof_carrying_complete", &proof_complete) ||
          !field_i64(record, "candidate_selected", &selected) ||
          !field_i64(record, "claim_ready", &claim_ready)) {
        ++result.parse_failures;
      } else {
        result.boundary_proof_complete = static_cast<int>(proof_complete);
        result.boundary_candidate_selected = static_cast<int>(selected);
        result.boundary_claim_ready = static_cast<int>(claim_ready);
      }
    } else if (record.kind == "PIREUS_POM_SUMMARY") {
      std::int64_t error = -1;
      std::int64_t failures = -1;
      std::int64_t valid = -1;
      if (!field_i64(record, "error", &error) ||
          !field_i64(record, "structural_failures", &failures) ||
          !field_i64(record, "valid", &valid)) {
        ++result.parse_failures;
      } else {
        result.summary_error = static_cast<int>(error);
        result.summary_structural_failures = static_cast<int>(failures);
        result.summary_valid = static_cast<int>(valid);
      }
    }
  }
  return result;
}

int phase_index(int left, int right) {
  return (left - 1) * 15 + (right - 1);
}

int cd_sigma(int left, int right, int bits) {
  if (left == 0 || right == 0) return 1;
  if (bits <= 1) return -1;
  const int half = 1 << (bits - 1);
  const int left_high = left >= half ? 1 : 0;
  const int right_high = right >= half ? 1 : 0;
  const int left_low = left & (half - 1);
  const int right_low = right & (half - 1);
  if (left_high == 0 && right_high == 0) {
    return cd_sigma(left_low, right_low, bits - 1);
  }
  if (left_high == 0 && right_high == 1) {
    return cd_sigma(right_low, left_low, bits - 1);
  }
  if (left_high == 1 && right_high == 0) {
    return right_low == 0 ? cd_sigma(left_low, 0, bits - 1)
                          : -cd_sigma(left_low, right_low, bits - 1);
  }
  return right_low == 0 ? -cd_sigma(0, left_low, bits - 1)
                        : cd_sigma(right_low, left_low, bits - 1);
}

struct Reconstruction {
  int anf_checks = 0;
  int anf_failures = 0;
  int cd_sign_checks = 0;
  int microprogram_entries = 0;
  int microprogram_field_checks = 0;
  int microprogram_failures = 0;
  int diagnostic_checks = 0;
  int diagnostic_failures = 0;
  int probe_checks = 0;
  int probe_failures = 0;
};

Reconstruction reconstruct(const Transcript& transcript) {
  Reconstruction out;
  for (int epoch = 0; epoch < kEpochs; ++epoch) {
    for (int left = 1; left < kLanes; ++left) {
      for (int right = 1; right < kLanes; ++right) {
        int reconstructed = 0;
        for (int a = left; a > 0; a = (a - 1) & left) {
          for (int b = right; b > 0; b = (b - 1) & right) {
            reconstructed ^= transcript.anf[epoch][phase_index(a, b)];
          }
        }
        ++out.anf_checks;
        if (reconstructed != transcript.phase[epoch][phase_index(left, right)]) {
          ++out.anf_failures;
        }
      }
    }

    std::array<int, kSignCells> candidate{};
    int negatives = 0;
    int square_negatives = 0;
    for (int left = 0; left < kLanes; ++left) {
      for (int right = 0; right < kLanes; ++right) {
        const int cell = left * kLanes + right;
        const int phase = left == 0 || right == 0
                              ? 0
                              : transcript.phase[epoch][phase_index(left, right)];
        candidate[cell] = (cd_sigma(left, right, 4) < 0 ? 1 : 0) ^ phase;
        ++out.cd_sign_checks;
        if (candidate[cell] != 0) ++negatives;
        if (left == right && candidate[cell] != 0) ++square_negatives;
      }
    }

    int commutator_defects = 0;
    for (int left = 0; left < kLanes; ++left) {
      for (int right = 0; right < kLanes; ++right) {
        if (candidate[left * kLanes + right] != candidate[right * kLanes + left]) {
          ++commutator_defects;
        }
      }
    }
    int associator_defects = 0;
    for (int i = 0; i < kLanes; ++i) {
      for (int j = 0; j < kLanes; ++j) {
        for (int k = 0; k < kLanes; ++k) {
          const int defect = candidate[i * kLanes + j] ^
                             candidate[(i ^ j) * kLanes + k] ^
                             candidate[j * kLanes + k] ^
                             candidate[i * kLanes + (j ^ k)];
          if (defect != 0) ++associator_defects;
        }
      }
    }
    const auto& summary = transcript.epochs[epoch];
    const std::array<std::pair<int, int>, 4> diagnostics = {{
        {negatives, summary.negatives},
        {square_negatives, summary.square_negatives},
        {commutator_defects, summary.commutator_defects},
        {associator_defects, summary.associator_defects}}};
    for (const auto& [actual, expected] : diagnostics) {
      ++out.diagnostic_checks;
      if (actual != expected) ++out.diagnostic_failures;
    }

    for (int entry = 0; entry < kSignCells; ++entry) {
      const int destination = entry / kLanes;
      const int ordinal = entry % kLanes;
      const int left = ordinal;
      const int right = left ^ destination;
      const int sign = candidate[left * kLanes + right];
      const auto& observed = transcript.microprogram[epoch][entry];
      const std::array<std::pair<int, int>, 5> fields = {{
          {left, observed.left}, {right, observed.right},
          {destination, observed.destination}, {ordinal, observed.ordinal},
          {sign, observed.sign}}};
      ++out.microprogram_entries;
      for (const auto& [actual, expected] : fields) {
        ++out.microprogram_field_checks;
        if (actual != expected) ++out.microprogram_failures;
      }
    }

    for (int destination = 0; destination < kLanes; ++destination) {
      std::int64_t sum = 0;
      for (int left = 0; left < kLanes; ++left) {
        const int right = left ^ destination;
        const int sign = candidate[left * kLanes + right] == 0 ? 1 : -1;
        sum += sign * (left + 1) * (kLanes + 1 - right);
      }
      ++out.probe_checks;
      if (sum != transcript.probes[epoch][destination]) ++out.probe_failures;
    }
  }
  return out;
}

struct Platform {
  std::string host;
  std::string host_file;
  std::string kubernetes_node;
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
  result.host_file = read_file_trimmed("/etc/hostname");
  if (const char* node = std::getenv("PIREUS_K8S_NODE_NAME")) {
    result.kubernetes_node = node;
  }
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
  result.machine_model = read_file_trimmed("/proc/device-tree/model");
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
  const auto cu_compute = reinterpret_cast<CuDeviceComputeCapability>(
      dlsym(library, "cuDeviceComputeCapability"));
  const auto cu_driver_version =
      reinterpret_cast<CuDriverGetVersion>(dlsym(library, "cuDriverGetVersion"));
  if (cu_init == nullptr || cu_device_get_count == nullptr ||
      cu_device_get == nullptr || cu_device_get_name == nullptr ||
      cu_compute == nullptr || cu_driver_version == nullptr || cu_init(0) != 0) {
    dlclose(library);
    return result;
  }
  result.driver_present = true;
  cu_driver_version(&result.driver_version);
  if (cu_device_get_count(&result.device_count) == 0 && result.device_count > 0) {
    int device = 0;
    char name[256] = {};
    if (cu_device_get(&device, 0) == 0) {
      cu_device_get_name(name, sizeof(name), device);
      cu_compute(&result.compute_major, &result.compute_minor, device);
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
    if (lower(read_file_trimmed(root + "/vendor")) != "0x10ee") continue;
    const std::string device = lower(read_file_trimmed(root + "/device"));
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
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << values[i];
  }
  return out.str();
}

bool map_equals(const std::map<std::string, std::int64_t>& values,
                const std::string& key, std::int64_t expected) {
  const auto found = values.find(key);
  return found != values.end() && found->second == expected;
}

bool target_name_valid(const std::string& target) {
  return target == "xeon" || target == "apple" || target == "dgx24" ||
         target == "dgx48" || target == "u250";
}

}  // namespace

int main(int argc, char** argv) {
  std::string target;
  std::string transcript_path;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument.rfind("--target=", 0) == 0) target = argument.substr(9);
    if (argument.rfind("--transcript=", 0) == 0) transcript_path = argument.substr(13);
  }
  if (!target_name_valid(target) || transcript_path.empty()) {
    std::cerr << "usage: operator_morphogenesis_material_parity "
                 "--target=xeon|apple|dgx24|dgx48|u250 --transcript=PATH\n";
    return 64;
  }

  const std::string raw = read_file_raw(transcript_path);
  if (raw.empty()) {
    std::cerr << "material parity: empty or unreadable Sounio transcript\n";
    return 66;
  }
  const std::string transcript_sha = sha256(raw);
  const Transcript transcript = parse_transcript(raw);
  const Reconstruction reconstruction = reconstruct(transcript);

  const bool transcript_sha_match = transcript_sha == kTranscriptSha;
  const bool records_complete =
      transcript.parse_failures == 0 && transcript.epoch_records == kEpochs &&
      transcript.genome_records == kEpochs * kInteriorCells &&
      transcript.microprogram_records == kEpochs * kSignCells &&
      transcript.probe_records == kEpochs * kLanes &&
      transcript.certificate_records == kCertificates;
  const bool counters_consistent =
      map_equals(transcript.archive, "initial", 96) &&
      map_equals(transcript.archive, "final", 128) &&
      map_equals(transcript.archive, "generated", kEpochs) &&
      map_equals(transcript.proof, "anf_coefficients", kEpochs * kInteriorCells) &&
      map_equals(transcript.proof, "anf_reconstruction_checks",
                 reconstruction.anf_checks) &&
      map_equals(transcript.proof, "anf_reconstruction_failures",
                 reconstruction.anf_failures) &&
      map_equals(transcript.proof, "certificates", transcript.certificate_records) &&
      map_equals(transcript.proof, "certificate_failures",
                 transcript.certificate_failures) &&
      map_equals(transcript.proof, "microprogram_checks",
                 reconstruction.microprogram_entries) &&
      map_equals(transcript.proof, "microprogram_failures", 0);
  const bool boundaries_preserved =
      transcript.negative_controls_passed == transcript.negative_controls_total &&
      transcript.negative_controls_total == 24 &&
      transcript.boundary_proof_complete == 1 &&
      transcript.boundary_candidate_selected == 0 &&
      transcript.boundary_claim_ready == 0 && transcript.summary_error == 0 &&
      transcript.summary_structural_failures == 0 && transcript.summary_valid == 1;
  const bool reconstruction_match =
      transcript_sha_match && records_complete && counters_consistent &&
      boundaries_preserved && reconstruction.anf_checks == 3600 &&
      reconstruction.cd_sign_checks == 4096 &&
      reconstruction.microprogram_entries == 4096 &&
      reconstruction.microprogram_field_checks == 20480 &&
      reconstruction.diagnostic_checks == 64 && reconstruction.probe_checks == 256 &&
      reconstruction.anf_failures == 0 &&
      reconstruction.microprogram_failures == 0 &&
      reconstruction.diagnostic_failures == 0 && reconstruction.probe_failures == 0 &&
      transcript.certificate_failures == 0;

  const Platform platform = observe_platform();
  const CudaIdentity cuda = observe_cuda();
  const U250Identity u250 = observe_u250();

  bool target_identity_observed = false;
  std::string canonical_name;
  std::string locator;
  std::string scheduler_route;
  if (target == "xeon") {
    canonical_name = "XEON";
    locator = "local:sounio-workspace-control-0";
    scheduler_route = "LOCAL";
    target_identity_observed =
        platform.kernel == "Linux" && platform.arch == "x86_64" &&
        lower(platform.cpu_model).find("xeon") != std::string::npos;
  } else if (target == "apple") {
    canonical_name = "APPLE_SILICON";
    locator =
        "tailnet-ssh:demetriosagourakis@"
        "sounio-language-macbook.tail21cbc4.ts.net";
    scheduler_route = "SSH_TAILNET";
    target_identity_observed =
        platform.kernel == "Darwin" && platform.arch == "arm64" &&
        (lower(platform.cpu_model).find("apple") != std::string::npos ||
         lower(platform.machine_model).rfind("mac", 0) == 0);
  } else if (target == "dgx24" || target == "dgx48") {
    const std::string expected_host = target == "dgx24" ? "spark-3c59" : "spark-8e54";
    canonical_name = target == "dgx24" ? "DGX_GB10_24" : "DGX_GB10_48";
    locator = "kubernetes-node:" + expected_host;
    scheduler_route = "KUBERNETES";
    const bool host_match = platform.host == expected_host ||
                            platform.host_file == expected_host ||
                            platform.kubernetes_node == expected_host;
    target_identity_observed =
        platform.kernel == "Linux" && platform.arch == "aarch64" && host_match &&
        cuda.driver_present && cuda.device_count >= 1 &&
        cuda.device_name == "NVIDIA GB10" && cuda.compute_major == 12 &&
        cuda.compute_minor == 1;
  } else {
    canonical_name = "AMD_ALVEO_U250_DECLARED_DUAL_CARD";
    locator = "kubernetes-node:dl380-proxmox";
    scheduler_route = "KUBERNETES";
    const bool host_match = platform.host == "dl380-proxmox" ||
                            platform.host_file == "dl380-proxmox" ||
                            platform.kubernetes_node == "dl380-proxmox" ||
                            platform.host == "dl380-dl380-proxmox" ||
                            platform.host_file == "dl380-dl380-proxmox";
    target_identity_observed = platform.kernel == "Linux" &&
                               platform.arch == "x86_64" && host_match &&
                               u250.paired_card_count >= 1;
  }

  const int installed_u250 = target == "u250" ? std::min(u250.paired_card_count, 2) : 0;
  const int pending_u250 = target == "u250" ? 2 - installed_u250 : 2;

  std::cout << "schema=pireus-operator-morphogenesis-material-parity-v12\n";
  std::cout << "producing_language=C++\n";
  std::cout << "producing_role=MATERIAL_PARITY\n";
  std::cout << "authority_language=Sounio\n";
  std::cout << "sounio_source_sha256=" << kSounioSourceSha << '\n';
  std::cout << "sounio_semantics_sha256=" << kSounioSemanticsSha << '\n';
  std::cout << "parent_semantics_sha256=" << kParentSemanticsSha << '\n';
  std::cout << "formal_parity_receipt_sha256=" << kFormalReceiptSha << '\n';
  std::cout << "effect_parity_receipt_sha256=" << kEffectReceiptSha << '\n';
  std::cout << "sounio_transcript_sha256=" << transcript_sha << '\n';
  std::cout << "sounio_transcript_sha256_match="
            << (transcript_sha_match ? "true" : "false") << '\n';
  std::cout << "target=" << target << '\n';
  std::cout << "target_name=" << canonical_name << '\n';
  std::cout << "target_locator=" << locator << '\n';
  std::cout << "scheduler_route=" << scheduler_route << '\n';
  std::cout << "slurm_route_used=false\n";
  std::cout << "legacy_lan_locator=\n";
  std::cout << "hostname=" << platform.host << '\n';
  std::cout << "host_file_identity=" << platform.host_file << '\n';
  std::cout << "kubernetes_node_identity=" << platform.kubernetes_node << '\n';
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
  std::cout << "u250_installed_card_count=" << installed_u250 << '\n';
  std::cout << "u250_pending_installation_count=" << pending_u250 << '\n';
  std::cout << "u250_second_card_state="
            << (target == "u250" ? "PENDING_INSTALLATION" : "NOT_OBSERVED_ON_THIS_TARGET")
            << '\n';
  std::cout << "u250_declared_dual_card_coverage_complete="
            << (target == "u250" && installed_u250 == 2 ? "true" : "false") << '\n';
  std::cout << "transcript_parse_failures=" << transcript.parse_failures << '\n';
  std::cout << "transcript_records_complete=" << (records_complete ? "true" : "false")
            << '\n';
  std::cout << "epoch_records_consumed=" << transcript.epoch_records << '\n';
  std::cout << "genome_records_consumed=" << transcript.genome_records << '\n';
  std::cout << "certificate_rows_consumed_from_sounio="
            << transcript.certificate_records << '\n';
  std::cout << "certificate_rows_reconstructed_by_cpp=false\n";
  std::cout << "certificate_inequality_failures="
            << transcript.certificate_failures << '\n';
  std::cout << "anf_reconstruction_checks=" << reconstruction.anf_checks << '\n';
  std::cout << "anf_reconstruction_failures=" << reconstruction.anf_failures << '\n';
  std::cout << "cd_sign_reconstruction_checks=" << reconstruction.cd_sign_checks << '\n';
  std::cout << "microprogram_entries_reconstructed="
            << reconstruction.microprogram_entries << '\n';
  std::cout << "microprogram_field_checks=" << reconstruction.microprogram_field_checks
            << '\n';
  std::cout << "microprogram_failures=" << reconstruction.microprogram_failures << '\n';
  std::cout << "diagnostic_checks=" << reconstruction.diagnostic_checks << '\n';
  std::cout << "diagnostic_failures=" << reconstruction.diagnostic_failures << '\n';
  std::cout << "probe_checks=" << reconstruction.probe_checks << '\n';
  std::cout << "probe_failures=" << reconstruction.probe_failures << '\n';
  std::cout << "frozen_counters_consistent="
            << (counters_consistent ? "true" : "false") << '\n';
  std::cout << "sounio_boundaries_preserved="
            << (boundaries_preserved ? "true" : "false") << '\n';
  std::cout << "material_reconstruction_match="
            << (reconstruction_match ? "true" : "false") << '\n';
  std::cout << "target_identity_observed="
            << (target_identity_observed ? "true" : "false") << '\n';
  std::cout << "material_scope=HOST_CXX_RECONSTRUCTION_OF_HASH_BOUND_SOUNIO_TRANSCRIPT_PLUS_TARGET_IDENTITY\n";
  std::cout << "numeric_values_status=CPP_RECONSTRUCTION_FROM_FROZEN_SOUNIO_TRANSCRIPT\n";
  std::cout << "analytic_proof_by_cpp=false\n";
  std::cout << "archive_reconstructed_by_cpp=false\n";
  std::cout << "sounio_executable_replayed_by_cpp=false\n";
  std::cout << "native_gpu_operator_kernel_execution=false\n";
  std::cout << "fpga_operator_kernel_execution=false\n";
  std::cout << "semantic_write=false\n";
  std::cout << "expected_result_write=false\n";
  std::cout << "candidate_selected=false\n";
  std::cout << "algebraic_novelty=false\n";
  std::cout << "algorithmic_novelty=false\n";
  std::cout << "material_novelty=false\n";
  std::cout << "scientific_novelty=false\n";
  std::cout << "historical_novelty=false\n";
  std::cout << "priority_claim=false\n";
  std::cout << "claim_ready=false\n";
  std::cout << "result="
            << (reconstruction_match && target_identity_observed ? "PASS" : "FAIL")
            << '\n';
  return reconstruction_match && target_identity_observed ? 0 : 2;
}
