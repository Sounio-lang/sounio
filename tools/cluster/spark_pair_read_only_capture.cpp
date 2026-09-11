#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view kZeroDigest =
    "0000000000000000000000000000000000000000000000000000000000000000";
constexpr std::size_t kRestorableHashes = 8;
constexpr std::size_t kObservationHashes = 12;
constexpr std::array<std::string_view, 22> kDigestDomains = {
    "restorable.systemd_system",
    "restorable.systemd_user",
    "restorable.docker_recreate",
    "restorable.nodeset_spec",
    "restorable.device_plugin_spec",
    "restorable.taints",
    "restorable.labels",
    "restorable.protected_paths_metadata",
    "observation.boot_identity",
    "observation.systemd_runtime",
    "observation.docker_runtime",
    "observation.k8s_identity",
    "observation.slurm_runtime",
    "observation.gpu_runtime",
    "observation.bpf_runtime",
    "observation.protected_paths_current",
    "observation.toolchain_hardware_commands",
    "observation.capture_transcript",
    "observation.managed_state_sentinel",
    "receipt.node_manifest",
    "receipt.node_restorable",
    "receipt.node_observation"};

constexpr std::array<std::uint32_t, 64> kSha256RoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

constexpr std::uint32_t rotate_right(std::uint32_t value,
                                     std::uint32_t count) {
  return (value >> count) | (value << (32U - count));
}

class Sha256 final {
 public:
  void update(std::string_view bytes) {
    for (const unsigned char byte : bytes) {
      buffer_[buffer_size_++] = byte;
      if (buffer_size_ == buffer_.size()) {
        transform();
        bit_length_ += 512U;
        buffer_size_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> finalize() {
    bit_length_ += static_cast<std::uint64_t>(buffer_size_) * 8U;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56U) {
      while (buffer_size_ < 64U) buffer_[buffer_size_++] = 0U;
      transform();
      buffer_size_ = 0;
    }
    while (buffer_size_ < 56U) buffer_[buffer_size_++] = 0U;
    for (std::size_t index = 0; index < 8U; ++index) {
      buffer_[63U - index] =
          static_cast<std::uint8_t>(bit_length_ >> (index * 8U));
    }
    transform();

    std::array<std::uint8_t, 32> digest{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      for (std::size_t byte = 0; byte < 4U; ++byte) {
        digest[word * 4U + byte] = static_cast<std::uint8_t>(
            state_[word] >> (24U - static_cast<std::uint32_t>(byte) * 8U));
      }
    }
    return digest;
  }

 private:
  void transform() {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16U; ++index) {
      const std::size_t offset = index * 4U;
      words[index] = (static_cast<std::uint32_t>(buffer_[offset]) << 24U) |
                     (static_cast<std::uint32_t>(buffer_[offset + 1U]) << 16U) |
                     (static_cast<std::uint32_t>(buffer_[offset + 2U]) << 8U) |
                     static_cast<std::uint32_t>(buffer_[offset + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const std::uint32_t s0 = rotate_right(words[index - 15U], 7U) ^
                               rotate_right(words[index - 15U], 18U) ^
                               (words[index - 15U] >> 3U);
      const std::uint32_t s1 = rotate_right(words[index - 2U], 17U) ^
                               rotate_right(words[index - 2U], 19U) ^
                               (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                                 rotate_right(e, 25U);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 =
          h + sum1 + choose + kSha256RoundConstants[index] + words[index];
      const std::uint32_t sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                                 rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
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

  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t bit_length_ = 0;
  std::array<std::uint32_t, 8> state_ = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
};

std::string hex_digest(const std::array<std::uint8_t, 32>& digest) {
  constexpr std::string_view alphabet = "0123456789abcdef";
  std::string output(64, '0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    output[index * 2U] = alphabet[digest[index] >> 4U];
    output[index * 2U + 1U] = alphabet[digest[index] & 0x0fU];
  }
  return output;
}

std::string sha256_hex(std::string_view bytes) {
  Sha256 hash;
  hash.update(bytes);
  return hex_digest(hash.finalize());
}

bool valid_digest_domain(std::string_view domain) {
  return std::find(kDigestDomains.begin(), kDigestDomains.end(), domain) !=
         kDigestDomains.end();
}

std::string domain_frame(std::string_view domain, std::string_view payload) {
  std::string frame;
  frame.reserve(128U + payload.size());
  frame.append("schema=sounio-spark-read-only-domain-frame-v1\n");
  frame.append("domain=");
  frame.append(domain);
  frame.push_back('\n');
  frame.append("payload_bytes=");
  frame.append(std::to_string(payload.size()));
  frame.append("\npayload_begin\n");
  frame.append(payload);
  return frame;
}

struct NodeInput {
  std::string_view node_id;
  std::array<std::string_view, kRestorableHashes> restorable;
  std::array<std::string_view, kObservationHashes> observation;
  std::string_view started_unix_ns;
  std::string_view finished_unix_ns;
  bool boot_stable;
  bool all_commands_rc_zero;
  bool no_unknown_fields;
};

bool is_lower_sha256(std::string_view value) {
  if (value.size() != 64) return false;
  for (const char code : value) {
    if (!((code >= '0' && code <= '9') || (code >= 'a' && code <= 'f'))) {
      return false;
    }
  }
  return true;
}

bool parse_bool(std::string_view value, bool& result) {
  if (value == "true") {
    result = true;
    return true;
  }
  if (value == "false") {
    result = false;
    return true;
  }
  return false;
}

bool parse_normalized_u64(std::string_view value, std::uint64_t& result) {
  if (value.empty() || (value.size() > 1 && value.front() == '0')) return false;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

void append_line(std::string& output, std::string_view key,
                 std::string_view value) {
  output.append(key);
  output.push_back('=');
  output.append(value);
  output.push_back('\n');
}

std::string build_domain_contract() {
  std::string output;
  output.reserve(2048);
  append_line(output, "schema", "sounio-spark-read-only-domain-contract-v1");
  append_line(output, "frame_schema", "sounio-spark-read-only-domain-frame-v1");
  append_line(output, "framing", "HEADER_UTF8_LF_THEN_EXACT_PAYLOAD");
  append_line(output, "header_order", "schema domain payload_bytes payload_begin");
  append_line(output, "payload_length", "NORMALIZED_UNSIGNED_DECIMAL_BYTE_COUNT");
  append_line(output, "payload_bytes", "EXACT_UNMODIFIED_QUERY_BYTES");
  append_line(output, "digest_algorithm", "SHA256_LOWERCASE_HEX");
  append_line(output, "domain_count", "22");
  for (std::size_t index = 0; index < kDigestDomains.size(); ++index) {
    std::string key = "domain.";
    key.push_back(static_cast<char>('0' + (index / 10U)));
    key.push_back(static_cast<char>('0' + (index % 10U)));
    append_line(output, key, kDigestDomains[index]);
  }
  return output;
}

std::string boolean_name(bool value) { return value ? "true" : "false"; }

std::string_view observed_mutation(const NodeInput& input) {
  const bool sentinels_equal = input.observation[10] == input.observation[11];
  return sentinels_equal && input.boot_stable && input.all_commands_rc_zero &&
                 input.no_unknown_fields
             ? "NONE"
             : "UNRESOLVED";
}

bool valid_node(const NodeInput& input) {
  if (input.node_id != "spark-3c59" && input.node_id != "spark-8e54") {
    return false;
  }
  for (const auto value : input.restorable) {
    if (!is_lower_sha256(value)) return false;
  }
  for (const auto value : input.observation) {
    if (!is_lower_sha256(value)) return false;
  }
  std::uint64_t started = 0;
  std::uint64_t finished = 0;
  return parse_normalized_u64(input.started_unix_ns, started) &&
         parse_normalized_u64(input.finished_unix_ns, finished) &&
         started < finished;
}

std::string build_node_manifest(const NodeInput& input) {
  const std::string_view mutation = observed_mutation(input);

  std::string output;
  output.reserve(4096);
  append_line(output, "schema", "sounio-spark-node-read-only-capture-v1");
  append_line(output, "node_id", input.node_id);
  append_line(output, "role", "OBSERVATION_ONLY");
  append_line(output, "capture_temporality", "CURRENT_POSTINSTALL_OBSERVATION");
  append_line(output, "producer_effect", "READ_ONLY_OBSERVATION");
  append_line(output, "scheduler_mutation", mutation);
  append_line(output, "host_configuration_mutation", mutation);
  append_line(output, "historical_preinstall_receipt", "NOT_PRESENT");
  append_line(output, "historical_preinstall_receipt_sha256", kZeroDigest);
  append_line(output, "protected_content_receipt", "NOT_OBSERVED");
  append_line(output, "restorable.systemd_system_sha256", input.restorable[0]);
  append_line(output, "restorable.systemd_user_sha256", input.restorable[1]);
  append_line(output, "restorable.docker_recreate_sha256", input.restorable[2]);
  append_line(output, "restorable.nodeset_spec_sha256", input.restorable[3]);
  append_line(output, "restorable.device_plugin_spec_sha256", input.restorable[4]);
  append_line(output, "restorable.taints_sha256", input.restorable[5]);
  append_line(output, "restorable.labels_sha256", input.restorable[6]);
  append_line(output, "restorable.protected_paths_metadata_sha256", input.restorable[7]);
  append_line(output, "observation.boot_identity_sha256", input.observation[0]);
  append_line(output, "observation.systemd_runtime_sha256", input.observation[1]);
  append_line(output, "observation.docker_runtime_sha256", input.observation[2]);
  append_line(output, "observation.k8s_identity_sha256", input.observation[3]);
  append_line(output, "observation.slurm_runtime_sha256", input.observation[4]);
  append_line(output, "observation.gpu_runtime_sha256", input.observation[5]);
  append_line(output, "observation.bpf_runtime_sha256", input.observation[6]);
  append_line(output, "observation.protected_paths_current_sha256", input.observation[7]);
  append_line(output, "observation.toolchain_hardware_commands_sha256", input.observation[8]);
  append_line(output, "observation.capture_transcript_sha256", input.observation[9]);
  append_line(output, "observation.managed_state_pre_sha256", input.observation[10]);
  append_line(output, "observation.managed_state_post_sha256", input.observation[11]);
  append_line(output, "observation.capture_started_unix_ns", input.started_unix_ns);
  append_line(output, "observation.capture_finished_unix_ns", input.finished_unix_ns);
  append_line(output, "observation.boot_stable", boolean_name(input.boot_stable));
  append_line(output, "observation.all_commands_rc_zero",
              boolean_name(input.all_commands_rc_zero));
  append_line(output, "observation.no_unknown_fields",
              boolean_name(input.no_unknown_fields));
  append_line(output, "restorable", "false");
  append_line(output, "snapshot_binding_receipt", "NOT_ISSUED");
  append_line(output, "state_transition", "false");
  return output;
}

std::string build_restorable_receipt(const NodeInput& input) {
  std::string output;
  output.reserve(2048);
  append_line(output, "schema", "sounio-spark-node-restorable-candidate-v1");
  append_line(output, "node_id", input.node_id);
  append_line(output, "role", "POSTINSTALL_RESTORABLE_CANDIDATE");
  append_line(output, "systemd_system_sha256", input.restorable[0]);
  append_line(output, "systemd_user_sha256", input.restorable[1]);
  append_line(output, "docker_recreate_sha256", input.restorable[2]);
  append_line(output, "nodeset_spec_sha256", input.restorable[3]);
  append_line(output, "device_plugin_spec_sha256", input.restorable[4]);
  append_line(output, "taints_sha256", input.restorable[5]);
  append_line(output, "labels_sha256", input.restorable[6]);
  append_line(output, "protected_paths_metadata_sha256", input.restorable[7]);
  append_line(output, "historical_preinstall_receipt", "NOT_PRESENT");
  append_line(output, "historical_preinstall_receipt_sha256", kZeroDigest);
  append_line(output, "protected_content_receipt", "NOT_OBSERVED");
  append_line(output, "restorable", "false");
  append_line(output, "snapshot_binding_receipt", "NOT_ISSUED");
  append_line(output, "state_transition", "false");
  return output;
}

std::string build_observation_receipt(const NodeInput& input) {
  const std::string_view mutation = observed_mutation(input);
  std::string output;
  output.reserve(3072);
  append_line(output, "schema", "sounio-spark-node-boot-scoped-observation-v1");
  append_line(output, "node_id", input.node_id);
  append_line(output, "role", "BOOT_SCOPED_OBSERVATION");
  append_line(output, "capture_temporality", "CURRENT_POSTINSTALL_OBSERVATION");
  append_line(output, "producer_effect", "READ_ONLY_OBSERVATION");
  append_line(output, "boot_identity_sha256", input.observation[0]);
  append_line(output, "systemd_runtime_sha256", input.observation[1]);
  append_line(output, "docker_runtime_sha256", input.observation[2]);
  append_line(output, "k8s_identity_sha256", input.observation[3]);
  append_line(output, "slurm_runtime_sha256", input.observation[4]);
  append_line(output, "gpu_runtime_sha256", input.observation[5]);
  append_line(output, "bpf_runtime_sha256", input.observation[6]);
  append_line(output, "protected_paths_current_sha256", input.observation[7]);
  append_line(output, "toolchain_hardware_commands_sha256", input.observation[8]);
  append_line(output, "capture_transcript_sha256", input.observation[9]);
  append_line(output, "managed_state_pre_sha256", input.observation[10]);
  append_line(output, "managed_state_post_sha256", input.observation[11]);
  append_line(output, "capture_started_unix_ns", input.started_unix_ns);
  append_line(output, "capture_finished_unix_ns", input.finished_unix_ns);
  append_line(output, "boot_stable", boolean_name(input.boot_stable));
  append_line(output, "all_commands_rc_zero",
              boolean_name(input.all_commands_rc_zero));
  append_line(output, "no_unknown_fields",
              boolean_name(input.no_unknown_fields));
  append_line(output, "scheduler_mutation", mutation);
  append_line(output, "host_configuration_mutation", mutation);
  append_line(output, "historical_preinstall_receipt", "NOT_PRESENT");
  append_line(output, "historical_preinstall_receipt_sha256", kZeroDigest);
  append_line(output, "restorable", "false");
  append_line(output, "snapshot_binding_receipt", "NOT_ISSUED");
  append_line(output, "state_transition", "false");
  return output;
}

std::string build_pair_manifest(
    const std::array<std::string_view, 6>& digests) {
  std::string output;
  output.reserve(2048);
  append_line(output, "schema", "sounio-spark-pair-read-only-capture-v1");
  append_line(output, "role", "OBSERVATION_ONLY");
  append_line(output, "node0_id", "spark-3c59");
  append_line(output, "node0_manifest_sha256", digests[0]);
  append_line(output, "node0_restorable_sha256", digests[1]);
  append_line(output, "node0_observation_sha256", digests[2]);
  append_line(output, "node1_id", "spark-8e54");
  append_line(output, "node1_manifest_sha256", digests[3]);
  append_line(output, "node1_restorable_sha256", digests[4]);
  append_line(output, "node1_observation_sha256", digests[5]);
  append_line(output, "ordered_pair", "true");
  append_line(output, "historical_preinstall_receipt", "NOT_PRESENT");
  append_line(output, "historical_preinstall_receipt_sha256", kZeroDigest);
  append_line(output, "restorable", "false");
  append_line(output, "snapshot_binding_receipt", "NOT_ISSUED");
  append_line(output, "state_transition", "false");
  return output;
}

NodeInput fixture_node(std::string_view node_id) {
  return NodeInput{
      node_id,
      {"1111111111111111111111111111111111111111111111111111111111111111",
       "2222222222222222222222222222222222222222222222222222222222222222",
       "3333333333333333333333333333333333333333333333333333333333333333",
       "4444444444444444444444444444444444444444444444444444444444444444",
       "5555555555555555555555555555555555555555555555555555555555555555",
       "6666666666666666666666666666666666666666666666666666666666666666",
       "7777777777777777777777777777777777777777777777777777777777777777",
       "8888888888888888888888888888888888888888888888888888888888888888"},
      {"9999999999999999999999999999999999999999999999999999999999999999",
       "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
       "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
       "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
       "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
       "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
       "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
       "1212121212121212121212121212121212121212121212121212121212121212",
       "1313131313131313131313131313131313131313131313131313131313131313",
       "1414141414141414141414141414141414141414141414141414141414141414",
       "1515151515151515151515151515151515151515151515151515151515151515",
       "1515151515151515151515151515151515151515151515151515151515151515"},
      "1788250000000000000", "1788250001000000000", true, true, true};
}

std::array<std::string_view, 6> fixture_pair_digests() {
  return {
      "1616161616161616161616161616161616161616161616161616161616161616",
      "1717171717171717171717171717171717171717171717171717171717171717",
      "1818181818181818181818181818181818181818181818181818181818181818",
      "1919191919191919191919191919191919191919191919191919191919191919",
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"};
}

bool selftest() {
  const auto node0 = build_node_manifest(fixture_node("spark-3c59"));
  const auto node1 = build_node_manifest(fixture_node("spark-8e54"));
  const auto restorable = build_restorable_receipt(fixture_node("spark-3c59"));
  const auto observation = build_observation_receipt(fixture_node("spark-3c59"));
  const auto pair = build_pair_manifest(fixture_pair_digests());
  const auto domain_contract = build_domain_contract();
  const auto pair_digests = fixture_pair_digests();
  const bool pair_digests_valid =
      std::all_of(pair_digests.begin(), pair_digests.end(), is_lower_sha256);
  return sha256_hex("") ==
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" &&
         sha256_hex("abc") ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" &&
         valid_node(fixture_node("spark-3c59")) &&
         valid_node(fixture_node("spark-8e54")) &&
         pair_digests_valid &&
         std::count(node0.begin(), node0.end(), '\n') == 38 &&
         std::count(node1.begin(), node1.end(), '\n') == 38 &&
         std::count(restorable.begin(), restorable.end(), '\n') == 17 &&
         std::count(observation.begin(), observation.end(), '\n') == 29 &&
         std::count(domain_contract.begin(), domain_contract.end(), '\n') == 30 &&
         std::count(pair.begin(), pair.end(), '\n') == 16 &&
         node0.find("node_id=spark-3c59\n") != std::string::npos &&
         node1.find("node_id=spark-8e54\n") != std::string::npos &&
         pair.find("ordered_pair=true\n") != std::string::npos &&
         node0.find("restorable=true") == std::string::npos &&
         pair.find("state_transition=true") == std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
    if (!selftest()) {
      std::cout << "PIREUS_SPARK_PAIR_READ_ONLY_CAPTURE_CPP_SELFTEST_FAIL\n";
      return 1;
    }
    std::cout << "PIREUS_SPARK_PAIR_READ_ONLY_CAPTURE_CPP_SELFTEST_PASS "
                 "role=MATERIAL_OBSERVER_NON_AUTHORITY\n";
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--sha256") {
    const std::string payload(std::istreambuf_iterator<char>(std::cin), {});
    std::cout << sha256_hex(payload) << '\n';
    return 0;
  }
  if (argc == 3 && std::string_view(argv[1]) == "--hash-domain") {
    const std::string_view domain = argv[2];
    if (!valid_digest_domain(domain)) return 64;
    const std::string payload(std::istreambuf_iterator<char>(std::cin), {});
    std::cout << sha256_hex(domain_frame(domain, payload)) << '\n';
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--fixture-node0") {
    std::cout << build_node_manifest(fixture_node("spark-3c59"));
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--fixture-node1") {
    std::cout << build_node_manifest(fixture_node("spark-8e54"));
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--fixture-pair") {
    std::cout << build_pair_manifest(fixture_pair_digests());
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--fixture-domain-contract") {
    std::cout << build_domain_contract();
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--fixture-node0-restorable") {
    std::cout << build_restorable_receipt(fixture_node("spark-3c59"));
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--fixture-node1-restorable") {
    std::cout << build_restorable_receipt(fixture_node("spark-8e54"));
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--fixture-node0-observation") {
    std::cout << build_observation_receipt(fixture_node("spark-3c59"));
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--fixture-node1-observation") {
    std::cout << build_observation_receipt(fixture_node("spark-8e54"));
    return 0;
  }
  if (argc == 11 && std::string_view(argv[1]) == "--restorable") {
    NodeInput input{};
    input.node_id = argv[2];
    for (std::size_t index = 0; index < kRestorableHashes; ++index) {
      input.restorable[index] = argv[3 + index];
    }
    input.observation.fill(kZeroDigest);
    input.started_unix_ns = "0";
    input.finished_unix_ns = "1";
    if (!valid_node(input)) return 64;
    std::cout << build_restorable_receipt(input);
    return 0;
  }
  if (argc == 20 && std::string_view(argv[1]) == "--observation") {
    NodeInput input{};
    input.node_id = argv[2];
    input.restorable.fill(kZeroDigest);
    for (std::size_t index = 0; index < kObservationHashes; ++index) {
      input.observation[index] = argv[3 + index];
    }
    input.started_unix_ns = argv[15];
    input.finished_unix_ns = argv[16];
    if (!parse_bool(argv[17], input.boot_stable) ||
        !parse_bool(argv[18], input.all_commands_rc_zero) ||
        !parse_bool(argv[19], input.no_unknown_fields) || !valid_node(input)) {
      return 64;
    }
    std::cout << build_observation_receipt(input);
    return 0;
  }
  if (argc == 28 && std::string_view(argv[1]) == "--node") {
    NodeInput input{};
    input.node_id = argv[2];
    for (std::size_t index = 0; index < kRestorableHashes; ++index) {
      input.restorable[index] = argv[3 + index];
    }
    for (std::size_t index = 0; index < kObservationHashes; ++index) {
      input.observation[index] = argv[11 + index];
    }
    input.started_unix_ns = argv[23];
    input.finished_unix_ns = argv[24];
    if (!parse_bool(argv[25], input.boot_stable) ||
        !parse_bool(argv[26], input.all_commands_rc_zero) ||
        !parse_bool(argv[27], input.no_unknown_fields) || !valid_node(input)) {
      return 64;
    }
    std::cout << build_node_manifest(input);
    return 0;
  }
  if (argc == 8 && std::string_view(argv[1]) == "--pair") {
    std::array<std::string_view, 6> digests{};
    for (std::size_t index = 0; index < digests.size(); ++index) {
      digests[index] = argv[2 + index];
      if (!is_lower_sha256(digests[index])) return 64;
    }
    std::cout << build_pair_manifest(digests);
    return 0;
  }
  return 64;
}
