constexpr std::string_view kFrozenProductExecCellFixtureManifestSha256 =
    "c06fc76a5fc24ec430b843a4f6e11f99c0e1fd034af61a304fdee11a0889498e";
constexpr std::string_view kFrozenExecResultHandleManifestSha256 =
    "7c1955299f2b308c331760764c289b6617910549b5dfc6bc4cf1969457306a55";
constexpr std::string_view kFrozenOperationGrantFixtureManifestSha256 =
    "4abfbc8a0fa9cdd1c2164f9f72ea4d408939b3d53497fa7fcaf008d71b1ea1e4";
constexpr std::string_view kFrozenOperationCatalogManifestSha256 =
    "25c58880cfe568a3b55b479df4515b63589c2e4dca703c97917e5cbdef0f1561";
constexpr std::string_view kFrozenOperationResultRecordManifestSha256 =
    "58d4a49c5b2462261ee53cd06f3ca8e29d363c1a38bf47274fa98a67b79cc569";
constexpr std::string_view kFrozenOperationArtifactSha256 =
    "eff2ac0ef28b34d6cc4f008cfb08a30ba18a0874c8654c06a3c62ec2f48a249c";

struct ProductExecCellFixtures {
  std::string command;
  std::string command_sha256;
  std::string intent_sha256;
  std::string command_mismatch_result;
  std::string command_mismatch_code;
  std::map<std::string, std::string> frames;
};

struct ProductExecResultPolicy {
  std::string event_sha256;
  std::string command_sha256;
  std::string receipt_sha256;
  std::string handle;
  std::string manifest_sha256;
};

struct OperationExecCellFixtures {
  std::string command;
  std::string command_sha256;
  std::string intent_sha256;
  std::string event_sha256;
  std::string source;
  std::string command_mismatch_result;
  std::string command_mismatch_code;
  std::map<std::string, std::string> frames;
};

struct OperationExecResultPolicy {
  std::string manifest_sha256;
  std::string record_schema;
  std::string operation;
  std::string catalog_sha256;
  std::string command_template_sha256;
  std::string result_schema_sha256;
  std::string sandbox_profile_sha256;
  std::string compiler_sha256;
};

ProductExecResultPolicy load_product_exec_result_policy(
    const std::string& manifest_path,
    const ProductExecCellFixtures& fixtures,
    const Manifest& payload_manifest) {
  const Manifest manifest = parse_frozen_manifest(
      manifest_path, kFrozenExecResultHandleManifestSha256,
      "product ExecResult handle");
  if (manifest.require("schema") != "loom-exec-result-handle-freeze-v1" ||
      manifest.require("stage") != "SEMANTICS_FROZEN" ||
      manifest.require("producing_language") != "Sounio" ||
      manifest.require("language_role") != "SEMANTIC_AUTHORITY" ||
      manifest.require("action") != "9033" ||
      manifest.require("fixture_manifest_sha256") !=
          kFrozenProductExecCellFixtureManifestSha256 ||
      manifest.require("result_receipt_sha256") !=
          payload_manifest.require("positive_stdout_sha256") ||
      manifest.require("handle_is_bearer") != "false" ||
      manifest.require("handle_is_execution_authority") != "false" ||
      manifest.require("expected_results_encoded_in_material_layer") != "false" ||
      manifest.require("material_execution") != "false" ||
      manifest.require("exec_cell_attached") != "false" ||
      manifest.require("provider_hook_switched") != "false" ||
      manifest.require("production_activation") != "false") {
    throw Error("product ExecResult manifest state invalid");
  }
  ProductExecResultPolicy policy;
  policy.event_sha256 = manifest.require("event_sha256");
  policy.command_sha256 = fixtures.command_sha256;
  policy.receipt_sha256 = manifest.require("result_receipt_sha256");
  policy.handle = manifest.require("canonical_handle");
  policy.manifest_sha256 = std::string(kFrozenExecResultHandleManifestSha256);
  const std::string expected_handle =
      "loom-result-v1:" + policy.event_sha256 + ":" +
      manifest.require("grant_generation") + ":" + policy.receipt_sha256;
  if (!lowercase_digest(policy.event_sha256) ||
      !lowercase_digest(policy.command_sha256) ||
      !lowercase_digest(policy.receipt_sha256) ||
      policy.handle != expected_handle) {
    throw Error("product ExecResult manifest binding invalid");
  }
  return policy;
}

std::string product_hex_digest_u32_limbs(std::string_view digest) {
  if (!lowercase_digest(digest)) {
    throw Error("product ExecCell binding digest malformed");
  }
  std::ostringstream output;
  for (std::size_t offset = 0; offset < digest.size(); offset += 8) {
    const std::string limb(digest.substr(offset, 8));
    std::size_t consumed = 0;
    const unsigned long value = std::stoul(limb, &consumed, 16);
    if (consumed != limb.size() ||
        value > std::numeric_limits<std::uint32_t>::max()) {
      throw Error("product ExecCell binding limb malformed");
    }
    output << ' ' << value;
  }
  return output.str();
}

ProductExecCellFixtures load_product_exec_cell_fixtures(
    const std::string& manifest_path, const std::string& bundle_path) {
  const Manifest manifest = parse_frozen_manifest(
      manifest_path, kFrozenProductExecCellFixtureManifestSha256,
      "product ExecCell fixtures");
  if (manifest.require("schema") !=
          "loom-product-exec-cell-fixture-freeze-v1" ||
      manifest.require("stage") != "SEMANTICS_FROZEN" ||
      manifest.require("producing_language") != "Sounio" ||
      manifest.require("semantic_authority") != "Sounio" ||
      manifest.require("action") != "9030" ||
      manifest.require("source_precedes_material") != "true" ||
      manifest.require("causal_sabotage") != "PASS" ||
      manifest.require("material_grant") != "false" ||
      manifest.require("material_execution") != "false" ||
      manifest.require("production_activation") != "false") {
    throw Error("product ExecCell fixture manifest state invalid");
  }

  const std::string contents = read_file(bundle_path);
  if (sha256(contents) != manifest.require("bundle_sha256")) {
    throw Error("product ExecCell fixture bundle hash mismatch");
  }

  std::vector<std::string> lines;
  std::istringstream input(contents);
  std::string line;
  while (std::getline(input, line)) lines.push_back(line);
  if (lines.size() != 14 ||
      lines.front() !=
          "LOOM_PRODUCT_EXEC_CELL_FIXTURE_V1 semantic_authority=Sounio "
          "action=9030 fixtures=4") {
    throw Error("product ExecCell fixture bundle shape invalid");
  }

  ProductExecCellFixtures fixtures;
  constexpr std::string_view intent_prefix = "INTENT command_sha256=";
  constexpr std::string_view intent_separator = " intent_sha256=";
  if (lines[1].rfind(intent_prefix, 0) != 0) {
    throw Error("product ExecCell intent line malformed");
  }
  const std::size_t separator = lines[1].find(intent_separator);
  if (separator == std::string::npos) {
    throw Error("product ExecCell intent binding absent");
  }
  fixtures.command_sha256 = lines[1].substr(
      intent_prefix.size(), separator - intent_prefix.size());
  fixtures.intent_sha256 =
      lines[1].substr(separator + intent_separator.size());
  constexpr std::string_view command_prefix = "COMMAND ";
  if (lines[2].rfind(command_prefix, 0) != 0) {
    throw Error("product ExecCell command line malformed");
  }
  fixtures.command = lines[2].substr(command_prefix.size());
  if (!lowercase_digest(fixtures.command_sha256) ||
      !lowercase_digest(fixtures.intent_sha256) ||
      fixtures.command_sha256 != sha256(fixtures.command) ||
      fixtures.command_sha256 != manifest.require("command_sha256") ||
      fixtures.intent_sha256 != manifest.require("intent_sha256") ||
      fixtures.command != manifest.require("command")) {
    throw Error("product ExecCell command/intent binding invalid");
  }
  fixtures.command_mismatch_result =
      manifest.require("command_mismatch_result");
  constexpr std::string_view deny_prefix = "DENY";
  if (fixtures.command_mismatch_result.rfind(deny_prefix, 0) != 0 ||
      fixtures.command_mismatch_result.size() == deny_prefix.size() ||
      !std::all_of(fixtures.command_mismatch_result.begin() +
                       static_cast<std::ptrdiff_t>(deny_prefix.size()),
                   fixtures.command_mismatch_result.end(),
                   [](unsigned char character) {
                     return std::isdigit(character);
                   })) {
    throw Error("product ExecCell frozen mismatch result malformed");
  }
  fixtures.command_mismatch_code =
      fixtures.command_mismatch_result.substr(deny_prefix.size());

  constexpr std::string_view parents_prefix = "PARENT_BINDINGS ";
  if (lines[3].rfind(parents_prefix, 0) != 0) {
    throw Error("product ExecCell parent bindings absent");
  }
  const std::string parents = lines[3].substr(parents_prefix.size());
  std::map<std::string, std::string> bindings;
  for (std::size_t index = 4; index < 10; ++index) {
    constexpr std::string_view prefix = "BINDING ";
    if (lines[index].rfind(prefix, 0) != 0) {
      throw Error("product ExecCell binding line malformed");
    }
    const std::size_t name_end = lines[index].find(' ', prefix.size());
    if (name_end == std::string::npos ||
        !bindings.emplace(
                     lines[index].substr(prefix.size(),
                                         name_end - prefix.size()),
                     lines[index].substr(name_end + 1))
             .second) {
      throw Error("product ExecCell binding label malformed");
    }
  }

  const std::array<std::pair<std::string_view, std::string_view>, 6>
      binding_fields{{
          {"grant_identity", "grant_identity_sha256"},
          {"command_environment", "command_environment_sha256"},
          {"peer_vector", "peer_vector_sha256"},
          {"transition_journal", "transition_journal_sha256"},
          {"source_semantics_toolchain",
           "source_semantics_toolchain_sha256"},
          {"result_receipt", "result_receipt_sha256"},
      }};
  std::string encoded_bindings;
  for (const auto& [label, field] : binding_fields) {
    const auto found = bindings.find(std::string(label));
    if (found == bindings.end() || found->second != manifest.require(std::string(field))) {
      throw Error("product ExecCell binding disagreed with frozen manifest");
    }
    encoded_bindings += product_hex_digest_u32_limbs(found->second);
  }
  if (bindings.size() != binding_fields.size()) {
    throw Error("unexpected product ExecCell binding count");
  }

  for (std::size_t index = 10; index < lines.size(); ++index) {
    constexpr std::string_view prefix = "FRAME ";
    if (lines[index].rfind(prefix, 0) != 0) {
      throw Error("product ExecCell frame line malformed");
    }
    const std::size_t label_end = lines[index].find(' ', prefix.size());
    if (label_end == std::string::npos ||
        !fixtures.frames
             .emplace(lines[index].substr(prefix.size(),
                                          label_end - prefix.size()),
                      lines[index].substr(label_end + 1) + " " + parents +
                          encoded_bindings)
             .second) {
      throw Error("product ExecCell frame label malformed");
    }
  }
  for (const std::string label : {"issue", "consume", "close",
                                  "command_mismatch"}) {
    if (!fixtures.frames.contains(label)) {
      throw Error("product ExecCell fixture absent: " + label);
    }
  }
  if (fixtures.frames.size() != 4) {
    throw Error("unexpected product ExecCell fixture count");
  }
  return fixtures;
}

OperationExecCellFixtures load_operation_exec_cell_fixtures(
    const std::string& manifest_path, const std::string& bundle_path) {
  const Manifest manifest = parse_frozen_manifest(
      manifest_path, kFrozenOperationGrantFixtureManifestSha256,
      "operation ExecCell fixtures");
  if (manifest.require("schema") !=
          "loom-exec-operation-grant-fixture-freeze-v1" ||
      manifest.require("stage") != "SEMANTICS_FROZEN" ||
      manifest.require("producing_language") != "Sounio" ||
      manifest.require("language_role") != "SEMANTIC_FIXTURE_PRODUCER" ||
      manifest.require("semantic_authority") != "Sounio" ||
      manifest.require("action") != "9030" ||
      manifest.require("catalog_action") != "9035" ||
      manifest.require("result_action") != "9036" ||
      manifest.require("causal_sabotage") != "PASS" ||
      manifest.require("arbitrary_shell") != "false" ||
      manifest.require("material_grant") != "false" ||
      manifest.require("material_execution") != "false" ||
      manifest.require("host_payload_selection_attached") != "false" ||
      manifest.require("provider_lifecycle_attached") != "false" ||
      manifest.require("production_activation") != "false") {
    throw Error("operation ExecCell fixture manifest state invalid");
  }
  const std::string contents = read_file(bundle_path);
  if (sha256(contents) != manifest.require("bundle_sha256")) {
    throw Error("operation ExecCell fixture bundle hash mismatch");
  }
  std::vector<std::string> lines;
  std::istringstream input(contents);
  std::string line;
  while (std::getline(input, line)) lines.push_back(line);
  if (lines.size() != 15 ||
      lines.front() !=
          "LOOM_EXEC_OPERATION_GRANT_FIXTURE_V1 semantic_authority=Sounio "
          "action=9030 catalog_action=9035 result_action=9036 fixtures=4") {
    throw Error("operation ExecCell fixture bundle shape invalid");
  }
  OperationExecCellFixtures fixtures;
  constexpr std::string_view intent_prefix = "INTENT command_sha256=";
  constexpr std::string_view intent_separator = " intent_sha256=";
  if (lines[1].rfind(intent_prefix, 0) != 0) {
    throw Error("operation ExecCell intent line malformed");
  }
  const std::size_t separator = lines[1].find(intent_separator);
  if (separator == std::string::npos) {
    throw Error("operation ExecCell intent binding absent");
  }
  fixtures.command_sha256 = lines[1].substr(
      intent_prefix.size(), separator - intent_prefix.size());
  fixtures.intent_sha256 = lines[1].substr(separator + intent_separator.size());
  constexpr std::string_view command_prefix = "COMMAND ";
  constexpr std::string_view event_prefix = "EVENT ";
  if (lines[2].rfind(command_prefix, 0) != 0 ||
      lines[3].rfind(event_prefix, 0) != 0) {
    throw Error("operation ExecCell command/event line malformed");
  }
  fixtures.command = lines[2].substr(command_prefix.size());
  fixtures.event_sha256 = lines[3].substr(event_prefix.size());
  constexpr std::string_view source_prefix =
      "loom-exec-cell-v2 sounio-check source=";
  if (fixtures.command.rfind(source_prefix, 0) != 0 ||
      fixtures.command.size() <= source_prefix.size()) {
    throw Error("operation ExecCell command shape invalid");
  }
  fixtures.source = fixtures.command.substr(source_prefix.size());
  if (!lowercase_digest(fixtures.command_sha256) ||
      !lowercase_digest(fixtures.intent_sha256) ||
      !lowercase_digest(fixtures.event_sha256) ||
      fixtures.command_sha256 != sha256(fixtures.command) ||
      fixtures.command_sha256 != manifest.require("command_sha256") ||
      fixtures.intent_sha256 != manifest.require("intent_sha256") ||
      fixtures.event_sha256 != manifest.require("event_sha256") ||
      fixtures.command != manifest.require("command")) {
    throw Error("operation ExecCell command/event binding invalid");
  }
  fixtures.command_mismatch_result = manifest.require("command_mismatch_result");
  constexpr std::string_view deny_prefix = "DENY";
  if (fixtures.command_mismatch_result.rfind(deny_prefix, 0) != 0 ||
      fixtures.command_mismatch_result.size() == deny_prefix.size() ||
      !std::all_of(fixtures.command_mismatch_result.begin() +
                       static_cast<std::ptrdiff_t>(deny_prefix.size()),
                   fixtures.command_mismatch_result.end(),
                   [](unsigned char character) { return std::isdigit(character); })) {
    throw Error("operation ExecCell frozen mismatch result malformed");
  }
  fixtures.command_mismatch_code =
      fixtures.command_mismatch_result.substr(deny_prefix.size());
  constexpr std::string_view parents_prefix = "PARENT_BINDINGS ";
  if (lines[4].rfind(parents_prefix, 0) != 0) {
    throw Error("operation ExecCell parent bindings absent");
  }
  const std::string parents = lines[4].substr(parents_prefix.size());
  std::map<std::string, std::string> bindings;
  for (std::size_t index = 5; index < 11; ++index) {
    constexpr std::string_view prefix = "BINDING ";
    if (lines[index].rfind(prefix, 0) != 0) {
      throw Error("operation ExecCell binding line malformed");
    }
    const std::size_t name_end = lines[index].find(' ', prefix.size());
    if (name_end == std::string::npos ||
        !bindings.emplace(lines[index].substr(prefix.size(),
                                              name_end - prefix.size()),
                          lines[index].substr(name_end + 1)).second) {
      throw Error("operation ExecCell binding label malformed");
    }
  }
  const std::array<std::pair<std::string_view, std::string_view>, 6>
      binding_fields{{
          {"grant_identity", "grant_identity_sha256"},
          {"command_environment", "command_environment_sha256"},
          {"peer_vector", "peer_vector_sha256"},
          {"transition_journal", "transition_journal_sha256"},
          {"source_semantics_toolchain", "source_semantics_toolchain_sha256"},
          {"result_receipt", "result_receipt_sha256"},
      }};
  std::string encoded_bindings;
  for (const auto& [label, field] : binding_fields) {
    const auto found = bindings.find(std::string(label));
    if (found == bindings.end() ||
        found->second != manifest.require(std::string(field))) {
      throw Error("operation ExecCell binding disagreed with frozen manifest");
    }
    encoded_bindings += product_hex_digest_u32_limbs(found->second);
  }
  if (bindings.size() != binding_fields.size()) {
    throw Error("unexpected operation ExecCell binding count");
  }
  for (std::size_t index = 11; index < lines.size(); ++index) {
    constexpr std::string_view prefix = "FRAME ";
    if (lines[index].rfind(prefix, 0) != 0) {
      throw Error("operation ExecCell frame line malformed");
    }
    const std::size_t label_end = lines[index].find(' ', prefix.size());
    if (label_end == std::string::npos ||
        !fixtures.frames
             .emplace(lines[index].substr(prefix.size(), label_end - prefix.size()),
                      lines[index].substr(label_end + 1) + " " + parents +
                          encoded_bindings)
             .second) {
      throw Error("operation ExecCell frame label malformed");
    }
  }
  for (const std::string label : {"issue", "consume", "close",
                                  "command_mismatch"}) {
    if (!fixtures.frames.contains(label)) {
      throw Error("operation ExecCell fixture absent: " + label);
    }
  }
  if (fixtures.frames.size() != 4) {
    throw Error("unexpected operation ExecCell fixture count");
  }
  return fixtures;
}

OperationExecResultPolicy load_operation_exec_result_policy(
    const std::string& catalog_manifest_path,
    const std::string& result_manifest_path) {
  const Manifest catalog = parse_frozen_manifest(
      catalog_manifest_path, kFrozenOperationCatalogManifestSha256,
      "operation catalog");
  const Manifest result = parse_frozen_manifest(
      result_manifest_path, kFrozenOperationResultRecordManifestSha256,
      "operation result record");
  if (catalog.require("schema") != "loom-exec-operation-catalog-freeze-v1" ||
      catalog.require("stage") != "SEMANTICS_FROZEN" ||
      catalog.require("producing_language") != "Sounio" ||
      catalog.require("action") != "9035" ||
      catalog.require("arbitrary_shell") != "false" ||
      catalog.require("host_payload_selection_attached") != "false" ||
      result.require("schema") != "loom-exec-result-record-freeze-v1" ||
      result.require("stage") != "SEMANTICS_FROZEN" ||
      result.require("producing_language") != "Sounio" ||
      result.require("action") != "9036" ||
      result.require("parent_9035_manifest_sha256") !=
          kFrozenOperationCatalogManifestSha256 ||
      result.require("operation") != "sounio-check" ||
      result.require("handle_is_bearer") != "false" ||
      result.require("handle_is_execution_authority") != "false" ||
      result.require("artifact_executed") != "false" ||
      result.require("dynamic_user_host_attached") != "false" ||
      result.require("provider_result_returned") != "false" ||
      result.require("production_activation") != "false") {
    throw Error("operation catalog/result manifest state invalid");
  }
  OperationExecResultPolicy policy;
  policy.manifest_sha256 =
      std::string(kFrozenOperationResultRecordManifestSha256);
  policy.record_schema = result.require("record_schema");
  policy.operation = result.require("operation");
  policy.catalog_sha256 = result.require("catalog_sha256");
  policy.command_template_sha256 =
      catalog.require("sounio_check_command_template_sha256");
  policy.result_schema_sha256 = catalog.require("sounio_check_result_schema_sha256");
  policy.sandbox_profile_sha256 =
      catalog.require("sounio_check_sandbox_profile_sha256");
  policy.compiler_sha256 = catalog.require("toolchain_compiler_sha256");
  for (const std::string* digest :
       {&policy.catalog_sha256, &policy.command_template_sha256,
        &policy.result_schema_sha256, &policy.sandbox_profile_sha256,
        &policy.compiler_sha256}) {
    if (!lowercase_digest(*digest)) {
      throw Error("operation result policy digest malformed");
    }
  }
  return policy;
}

std::map<std::string, std::string> operation_key_value_line(
    const std::string& line, std::string_view expected_schema) {
  std::istringstream input(line);
  std::string schema;
  if (!(input >> schema) || schema != expected_schema) {
    throw Error("operation ExecCell line schema mismatch expected=" +
                std::string(expected_schema) + " actual=" +
                product_compact_diagnostic(line));
  }
  std::map<std::string, std::string> fields;
  std::string token;
  while (input >> token) {
    const std::size_t equals = token.find('=');
    if (equals == std::string::npos || equals == 0 ||
        equals + 1 >= token.size() ||
        !fields.emplace(token.substr(0, equals), token.substr(equals + 1)).second) {
      throw Error("operation ExecCell line field malformed");
    }
  }
  return fields;
}

const std::string& operation_require(
    const std::map<std::string, std::string>& fields,
    const std::string& key) {
  const auto found = fields.find(key);
  if (found == fields.end() || found->second.empty()) {
    throw Error("operation ExecCell line omitted " + key);
  }
  return found->second;
}

std::map<std::string, std::string> operation_record_fields(
    const std::string& record, std::string_view expected_schema) {
  if (record.empty() || record.back() != '\n' || record.size() > 64 * 1024) {
    throw Error("operation ExecCell record framing invalid");
  }
  std::istringstream input(record);
  std::string schema;
  if (!std::getline(input, schema) || schema != expected_schema) {
    throw Error("operation ExecCell record schema mismatch");
  }
  std::map<std::string, std::string> fields;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos || equals == 0 ||
        equals + 1 >= line.size() ||
        !fields.emplace(line.substr(0, equals), line.substr(equals + 1)).second) {
      throw Error("operation ExecCell record field malformed");
    }
  }
  return fields;
}

struct HostSounioCheckObservation {
  std::string handle;
  std::string record;
  std::string record_sha256;
  std::string artifact_sha256;
  std::string source_sha256;
  uid_t uid = 0;
  gid_t gid = 0;
  bool ready = false;
  bool identity_authenticated = false;
  bool arm_sent = false;
  bool close_bound = false;
  bool material_execution = false;
  bool artifact_measured = false;
  bool pidfd_extinct = false;
  bool process_extinct = false;
  bool cgroup_unpopulated = false;
  bool unit_inactive = false;
  bool runtime_directory_extinct = false;

  bool success_complete() const {
    return ready && identity_authenticated && arm_sent && close_bound &&
           material_execution && artifact_measured && pidfd_extinct && process_extinct &&
           cgroup_unpopulated && unit_inactive && runtime_directory_extinct;
  }
};

class HostSounioCheckCell {
 public:
  HostSounioCheckCell(std::string root, std::string runtime,
                      std::string systemd_run, std::string systemctl,
                      OperationExecCellFixtures fixtures,
                      OperationExecResultPolicy result_policy,
                      std::string generation)
      : root_(canonical_path(root)), runtime_(canonical_path(runtime)),
        systemd_run_(canonical_path(systemd_run)),
        systemctl_(canonical_path(systemctl)), fixtures_(std::move(fixtures)),
        result_policy_(std::move(result_policy)),
        generation_(std::move(generation)),
        nonce_(random_generation_sha256().substr(0, 16)),
        unit_("sounio-loom-operation-cell-" + std::to_string(getpid()) + "-" +
              nonce_ + ".service"),
        runtime_directory_("sounio-loom-operation-cell-" + nonce_),
        runtime_path_("/run/" + runtime_directory_),
        runtime_storage_path_("/run/private/" + runtime_directory_) {
    if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0) {
      throw Error("host operation ExecCell requires root broker identity");
    }
    if (!lowercase_digest(generation_)) {
      throw Error("host operation ExecCell generation malformed");
    }
    if (!process_witness_root_artifact(runtime_, true)) {
      throw Error("host operation ExecCell runtime metadata unsafe");
    }
    if (!root_system_tool(systemd_run_)) {
      throw Error("host operation ExecCell systemd-run metadata unsafe");
    }
    if (!root_system_tool(systemctl_)) {
      throw Error("host operation ExecCell systemctl metadata unsafe");
    }
    source_path_ = canonical_path(root_ + "/" + fixtures_.source);
    source_sha256_ = file_sha256(source_path_);
    runtime_sha256_ = file_sha256(runtime_);
    spawn();
    await_identity_and_ready();
  }

  HostSounioCheckCell(const HostSounioCheckCell&) = delete;
  HostSounioCheckCell& operator=(const HostSounioCheckCell&) = delete;
  ~HostSounioCheckCell() { cleanup_noexcept(); }

  uid_t uid() const { return uid_; }
  gid_t gid() const { return gid_; }
  const std::string& principal_sha256() const { return principal_sha256_; }
  const std::string& descriptor_sha256() const { return descriptor_sha256_; }

  bool authenticated_before_arm() const {
    try {
      return pid_ > 1 && pidfd_live(pidfd_.get()) &&
             process_start_tick(pid_) == start_tick_ &&
             process_uid(pid_) == uid_ && process_gid(pid_) == gid_ &&
             process_executable(pid_) == runtime_ &&
             sha256(process_cgroup(pid_)) == cgroup_sha256_ &&
             process_cgroup(pid_).find(unit_) != std::string::npos &&
             property("DynamicUser") == "yes" &&
             property("NoNewPrivileges") == "yes" &&
             property("ProtectSystem") == "strict" &&
             property("ProtectProc") == "invisible" &&
             property("ProcSubset") == "pid" &&
             property("PrivateNetwork") == "yes" &&
             property("PrivateDevices") == "yes" &&
             property("MemoryDenyWriteExecute") == "yes" &&
             property("RuntimeDirectoryPreserve") == "yes" &&
             runtime_storage_authenticated() &&
             uid_ != 0 && gid_ != 0;
    } catch (...) {
      return false;
    }
  }

  HostSounioCheckObservation execute(const std::string& grant_receipt_sha256) {
    if (finished_) throw Error("operation ExecCell already consumed");
    if (!lowercase_digest(grant_receipt_sha256) ||
        !authenticated_before_arm()) {
      throw Error("operation ExecCell pre-ARM identity drifted");
    }
    const auto deadline =
        std::chrono::steady_clock::now() + kProductExecIngressHostDeadline;
    quorum_write_all(
        input_.get(),
        "ARM " + generation_ + " " + fixtures_.event_sha256 + " " +
            principal_sha256_ + " " + descriptor_sha256_ + " " +
            grant_receipt_sha256 + "\n",
        deadline);
    HostSounioCheckObservation observation;
    observation.uid = uid_;
    observation.gid = gid_;
    observation.ready = ready_;
    observation.identity_authenticated = true;
    observation.arm_sent = true;
    const std::string header = quorum_read_line(
        output_.get(), output_buffer_, 4096, deadline,
        "operation ExecCell result header timeout");
    const auto header_fields = operation_key_value_line(
        header, "LOOM_EXEC_OPERATION_CELL_RESULT_V1");
    observation.record.clear();
    for (std::size_t line = 0; line < 18; ++line) {
      observation.record += quorum_read_line(
          output_.get(), output_buffer_, 4096, deadline,
          "operation ExecCell result record timeout") + "\n";
    }
    observation.record_sha256 = sha256(observation.record);
    observation.handle = operation_require(header_fields, "handle");
    observation.artifact_sha256 =
        operation_require(header_fields, "artifact_sha256");
    observation.source_sha256 = operation_require(header_fields, "source_sha256");
    validate_result(header_fields, grant_receipt_sha256, observation);
    observation.material_execution = true;
    observation.artifact_measured = verify_material_artifacts(observation);
    return observation;
  }

  void close(HostSounioCheckObservation& observation,
             const std::string& close_receipt_sha256) {
    if (finished_) throw Error("operation ExecCell CLOSE already consumed");
    if (!observation.artifact_measured) {
      throw Error("operation ExecCell CLOSE material unmeasured");
    }
    if (!lowercase_digest(close_receipt_sha256)) {
      throw Error("operation ExecCell CLOSE receipt malformed");
    }
    if (!authenticated_before_arm()) {
      throw Error("operation ExecCell CLOSE identity drifted");
    }
    const auto deadline =
        std::chrono::steady_clock::now() + kProductExecIngressHostDeadline;
    quorum_write_all(
        input_.get(),
        "CLOSE " + generation_ + " " + fixtures_.event_sha256 + " " +
            observation.record_sha256 + " " + close_receipt_sha256 + "\n",
        deadline);
    input_ = UniqueFd();
    const std::string closed = quorum_read_line(
        output_.get(), output_buffer_, 4096, deadline,
        "operation ExecCell CLOSED timeout");
    const auto fields = operation_key_value_line(
        closed, "LOOM_EXEC_OPERATION_CELL_CLOSED_V1");
    if (operation_require(fields, "semantic_authority") != "Sounio" ||
        operation_require(fields, "action") != "9030" ||
        operation_require(fields, "generation_sha256") != generation_ ||
        operation_require(fields, "event_sha256") != fixtures_.event_sha256 ||
        operation_require(fields, "record_sha256") !=
            observation.record_sha256 ||
        operation_require(fields, "close_receipt_sha256") !=
            close_receipt_sha256 ||
        operation_require(fields, "material_execution") != "true" ||
        operation_require(fields, "authority_extinction") != "armed") {
      throw Error("operation ExecCell CLOSED binding invalid");
    }
    observation.close_bound = true;
    const int status = await_client_exit();
    if (status != 0) {
      throw Error("operation ExecCell systemd client status diverged: " +
                  std::to_string(status));
    }
    finish_extinction(observation);
    finished_ = true;
  }

 private:
  static bool root_system_tool(const std::string& path) {
    struct stat info {};
    return lstat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode) &&
           info.st_uid == 0 && info.st_gid == 0 && info.st_nlink == 1 &&
           (info.st_mode & 0022) == 0 && (info.st_mode & 0111) != 0;
  }

  static bool path_absent(const std::string& path) {
    struct stat info {};
    return lstat(path.c_str(), &info) != 0 && errno == ENOENT;
  }

  bool runtime_storage_authenticated() const {
    struct stat link {};
    struct stat host_private {};
    struct stat host_directory {};
    struct stat cell_directory {};
    std::array<char, PATH_MAX> target{};
    const ssize_t target_size =
        readlink(runtime_path_.c_str(), target.data(), target.size() - 1);
    const std::string expected_target = "private/" + runtime_directory_;
    return lstat(runtime_path_.c_str(), &link) == 0 &&
           S_ISLNK(link.st_mode) && link.st_uid == 0 && link.st_gid == 0 &&
           target_size == static_cast<ssize_t>(expected_target.size()) &&
           std::string(target.data(), static_cast<std::size_t>(target_size)) ==
               expected_target &&
           lstat("/run/private", &host_private) == 0 &&
           S_ISDIR(host_private.st_mode) && host_private.st_uid == 0 &&
           host_private.st_gid == 0 &&
           (host_private.st_mode & 0777) == 0700 &&
           lstat(runtime_storage_path_.c_str(), &host_directory) == 0 &&
           S_ISDIR(host_directory.st_mode) &&
           (host_directory.st_mode & 0777) == 0700 &&
           lstat(cell_storage_path().c_str(), &cell_directory) == 0 &&
           S_ISDIR(cell_directory.st_mode) && cell_directory.st_uid == uid_ &&
           cell_directory.st_gid == gid_ &&
           (cell_directory.st_mode & 0777) == 0700;
  }

  std::string cell_storage_path() const {
    return "/proc/" + std::to_string(pid_) + "/root" +
           runtime_storage_path_;
  }

  std::string property(const std::string& name) const {
    return quorum_tool_value(systemctl_,
                             {"show", unit_, "--property=" + name, "--value"});
  }

  bool unit_inactive() const {
    const QuorumToolResult result = quorum_run_tool(
        systemctl_, {"show", unit_, "--property=ActiveState", "--value"},
        std::chrono::seconds(2));
    if (result.exit_code == 0) {
      const std::string state = trim(result.output);
      return state == "inactive" || state == "failed" || state == "dead";
    }
    const QuorumToolResult absent = quorum_run_tool(
        systemctl_, {"list-units", "--all", "--plain", "--no-legend",
                     "--no-pager", unit_},
        std::chrono::seconds(2));
    return absent.exit_code == 0 && trim(absent.output).empty();
  }

  void spawn() {
    int input_pipe[2];
    int output_pipe[2];
    if (pipe2(input_pipe, O_CLOEXEC) != 0 ||
        pipe2(output_pipe, O_CLOEXEC) != 0) {
      throw Error("cannot create operation ExecCell custody pipes");
    }
    UniqueFd input_read(input_pipe[0]);
    UniqueFd input_write(input_pipe[1]);
    UniqueFd output_read(output_pipe[0]);
    UniqueFd output_write(output_pipe[1]);
    struct stat descriptor {};
    if (fstat(input_write.get(), &descriptor) != 0 ||
        !S_ISFIFO(descriptor.st_mode)) {
      throw Error("cannot measure operation ExecCell descriptor");
    }
    descriptor_device_ = static_cast<std::uint64_t>(descriptor.st_dev);
    descriptor_inode_ = static_cast<std::uint64_t>(descriptor.st_ino);
    client_pid_ = fork();
    if (client_pid_ < 0) {
      throw Error("cannot fork operation ExecCell systemd client");
    }
    if (client_pid_ == 0) {
      if (dup2(input_read.get(), STDIN_FILENO) < 0 ||
          dup2(output_write.get(), STDOUT_FILENO) < 0 ||
          dup2(output_write.get(), STDERR_FILENO) < 0) {
        _exit(126);
      }
      input_read = UniqueFd();
      input_write = UniqueFd();
      output_read = UniqueFd();
      output_write = UniqueFd();
      std::vector<std::string> arguments = {
          "--quiet",
          "--unit=" + unit_,
          "--service-type=exec",
          "--collect",
          "--pipe",
          "--wait",
          "--property=DynamicUser=yes",
          "--property=UMask=0077",
          "--property=NoNewPrivileges=yes",
          "--property=PrivateTmp=yes",
          "--property=PrivateDevices=yes",
          "--property=PrivateNetwork=yes",
          "--property=PrivateUsers=no",
          "--property=ProtectSystem=strict",
          "--property=ProtectHome=yes",
          "--property=ProtectKernelTunables=yes",
          "--property=ProtectKernelModules=yes",
          "--property=ProtectKernelLogs=yes",
          "--property=ProtectControlGroups=yes",
          "--property=ProtectClock=yes",
          "--property=ProtectProc=invisible",
          "--property=ProcSubset=pid",
          "--property=RestrictNamespaces=yes",
          "--property=RestrictSUIDSGID=yes",
          "--property=LockPersonality=yes",
          "--property=MemoryDenyWriteExecute=yes",
          "--property=RestrictRealtime=yes",
          "--property=SystemCallArchitectures=native",
          "--property=RestrictAddressFamilies=AF_UNIX",
          "--property=CapabilityBoundingSet=",
          "--property=AmbientCapabilities=",
          "--property=KillMode=mixed",
          "--property=TimeoutStopSec=2s",
          "--property=RuntimeDirectory=" + runtime_directory_,
          "--property=RuntimeDirectoryMode=0700",
          "--property=RuntimeDirectoryPreserve=yes",
          "--property=WorkingDirectory=" + root_,
          "--property=ReadOnlyPaths=" + root_,
          "--",
          runtime_,
          "_exec-operation-cell",
          "--root", root_,
          "--source", fixtures_.source,
          "--output-dir", runtime_path_,
          "--unit", unit_,
          "--mode", "host"};
      std::vector<char*> values;
      values.reserve(arguments.size() + 2);
      values.push_back(const_cast<char*>(systemd_run_.c_str()));
      for (const std::string& argument : arguments) {
        values.push_back(const_cast<char*>(argument.c_str()));
      }
      values.push_back(nullptr);
      execv(systemd_run_.c_str(), values.data());
      _exit(127);
    }
    input_read = UniqueFd();
    output_write = UniqueFd();
    input_ = std::move(input_write);
    output_ = std::move(output_read);
  }

  void observe_client() {
    if (client_pid_ <= 0 || client_reaped_) return;
    int status = 0;
    const pid_t waited = waitpid(client_pid_, &status, WNOHANG);
    if (waited < 0 && errno != EINTR) {
      throw Error("operation ExecCell systemd client wait failed");
    }
    if (waited == client_pid_) {
      client_reaped_ = true;
      client_status_ = status;
    }
  }

  void await_identity_and_ready() {
    const auto deadline =
        std::chrono::steady_clock::now() + kProductExecIngressHostDeadline;
    for (;;) {
      observe_client();
      if (client_reaped_) {
        const int status =
            WIFEXITED(client_status_)
                ? WEXITSTATUS(client_status_)
                : (WIFSIGNALED(client_status_)
                       ? 128 + WTERMSIG(client_status_)
                       : 255);
        std::string diagnostic;
        try {
          diagnostic = read_remaining_output(
              std::chrono::steady_clock::now() + std::chrono::seconds(1));
        } catch (const Error& error) {
          diagnostic = std::string("diagnostic-unavailable:") + error.what();
        }
        throw Error("operation ExecCell exited before identity admission status=" +
                    std::to_string(status) + " output=" +
                    product_compact_diagnostic(diagnostic));
      }
      try {
        const auto parsed = parse_u64(property("MainPID"));
        if (parsed && *parsed > 1 &&
            *parsed <= static_cast<std::uint64_t>(
                           std::numeric_limits<pid_t>::max())) {
          pid_ = static_cast<pid_t>(*parsed);
          start_tick_ = process_start_tick(pid_);
          uid_ = process_uid(pid_);
          gid_ = process_gid(pid_);
          const std::string cgroup = process_cgroup(pid_);
          cgroup_sha256_ = sha256(cgroup);
          cgroup_relative_ = process_witness_cgroup_path(cgroup);
          pidfd_ = UniqueFd(open_pidfd(pid_));
          if (!authenticated_before_arm()) {
            throw Error("operation ExecCell identity admission failed");
          }
          break;
        }
      } catch (const Error&) {
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        throw Error("operation ExecCell identity admission timed out");
      }
      poll(nullptr, 0, 5);
    }
    const std::string ready_line = quorum_read_line(
        output_.get(), output_buffer_, 4096, deadline,
        "operation ExecCell READY timeout");
    const auto ready = operation_key_value_line(
        ready_line, "LOOM_EXEC_OPERATION_CELL_READY_V1");
    const std::string principal_canonical =
        "LOOM_EXEC_CELL_PRINCIPAL/1|pid=" + std::to_string(pid_) +
        "|start_tick=" + std::to_string(start_tick_) + "|uid=" +
        std::to_string(uid_) + "|gid=" + std::to_string(gid_) +
        "|cgroup_sha256=" + cgroup_sha256_ + "|executable_sha256=" +
        runtime_sha256_ + "|unit=" + unit_;
    principal_sha256_ = sha256(principal_canonical);
    const std::string descriptor_canonical =
        "LOOM_EXEC_CELL_DESCRIPTOR/1|unit=" + unit_ +
        "|fd=0|device=" + std::to_string(descriptor_device_) +
        "|inode=" + std::to_string(descriptor_inode_);
    descriptor_sha256_ = sha256(descriptor_canonical);
    ready_ =
        operation_require(ready, "semantic_authority") == "Sounio" &&
        operation_require(ready, "grant_action") == "9030" &&
        operation_require(ready, "catalog_action") == "9035" &&
        operation_require(ready, "result_action") == "9036" &&
        operation_require(ready, "pid") == std::to_string(pid_) &&
        operation_require(ready, "start_tick") == std::to_string(start_tick_) &&
        operation_require(ready, "uid") == std::to_string(uid_) &&
        operation_require(ready, "gid") == std::to_string(gid_) &&
        operation_require(ready, "cgroup_sha256") == cgroup_sha256_ &&
        operation_require(ready, "executable_sha256") == runtime_sha256_ &&
        operation_require(ready, "unit") == unit_ &&
        operation_require(ready, "principal_sha256") == principal_sha256_ &&
        operation_require(ready, "descriptor_device") ==
            std::to_string(descriptor_device_) &&
        operation_require(ready, "descriptor_inode") ==
            std::to_string(descriptor_inode_) &&
        operation_require(ready, "descriptor_binding_sha256") ==
            descriptor_sha256_ &&
        operation_require(ready, "host_mode") == "true" &&
        operation_require(ready, "inherited_descriptor") == "true" &&
        operation_require(ready, "material_execution") == "false";
    if (!ready_ || !authenticated_before_arm()) {
      throw Error("operation ExecCell READY binding mismatch");
    }
  }

  std::string read_remaining_output(
      std::chrono::steady_clock::time_point deadline) {
    std::string output = std::move(output_buffer_);
    output_buffer_.clear();
    std::array<char, 4096> bytes{};
    for (;;) {
      if (output.size() > 64 * 1024) {
        throw Error("operation ExecCell result exceeded limit");
      }
      quorum_wait(output_.get(), POLLIN | POLLHUP, deadline,
                  "operation ExecCell result timeout");
      const ssize_t count = read(output_.get(), bytes.data(), bytes.size());
      if (count > 0) {
        output.append(bytes.data(), static_cast<std::size_t>(count));
      } else if (count == 0) {
        output_ = UniqueFd();
        return output;
      } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
        throw Error("operation ExecCell result read failed");
      }
    }
  }

  void validate_result(
      const std::map<std::string, std::string>& header,
      const std::string& grant_receipt_sha256,
      const HostSounioCheckObservation& observation) const {
    const auto record = operation_record_fields(
        observation.record, result_policy_.record_schema);
    const auto record_value = [&](const std::string& key) -> const std::string& {
      return operation_require(record, key);
    };
    const std::string expected_handle =
        "loom-result-v2:" + fixtures_.event_sha256 + ":" + generation_ + ":" +
        observation.record_sha256;
    const bool header_valid =
        operation_require(header, "semantic_authority") == "Sounio" &&
        operation_require(header, "grant_action") == "9030" &&
        operation_require(header, "catalog_action") == "9035" &&
        operation_require(header, "result_action") == "9036" &&
        operation_require(header, "operation") == result_policy_.operation &&
        operation_require(header, "generation_sha256") == generation_ &&
        operation_require(header, "event_sha256") == fixtures_.event_sha256 &&
        operation_require(header, "principal_sha256") == principal_sha256_ &&
        operation_require(header, "descriptor_binding_sha256") ==
            descriptor_sha256_ &&
        operation_require(header, "grant_receipt_sha256") ==
            grant_receipt_sha256 &&
        operation_require(header, "source_sha256") == source_sha256_ &&
        operation_require(header, "artifact_sha256") ==
            kFrozenOperationArtifactSha256 &&
        operation_require(header, "record_sha256") == observation.record_sha256 &&
        operation_require(header, "handle") == expected_handle &&
        operation_require(header, "grant_arm_received") == "true" &&
        operation_require(header, "principal_self_measured") == "true" &&
        operation_require(header, "inherited_descriptor") == "true" &&
        operation_require(header, "host_mode") == "true" &&
        operation_require(header, "material_execution") == "true" &&
        operation_require(header, "artifact_executed") == "false" &&
        operation_require(header, "handle_is_bearer") == "false" &&
        operation_require(header, "handle_is_execution_authority") == "false";
    const bool record_valid =
        record.size() == 17 &&
        record_value("operation") == result_policy_.operation &&
        record_value("event_sha256") == fixtures_.event_sha256 &&
        record_value("command_template_sha256") ==
            result_policy_.command_template_sha256 &&
        record_value("generation_sha256") == generation_ &&
        record_value("source_sha256") == source_sha256_ &&
        record_value("compiler_sha256") == result_policy_.compiler_sha256 &&
        lowercase_digest(record_value("argv_sha256")) &&
        record_value("artifact_sha256") == kFrozenOperationArtifactSha256 &&
        record_value("artifact_bytes") ==
            operation_require(header, "artifact_bytes") &&
        lowercase_digest(record_value("stdout_sha256")) &&
        lowercase_digest(record_value("stderr_sha256")) &&
        lowercase_digest(record_value("diagnostics_sha256")) &&
        record_value("sandbox_profile_sha256") ==
            result_policy_.sandbox_profile_sha256 &&
        record_value("principal_sha256") == principal_sha256_ &&
        record_value("descriptor_binding_sha256") == descriptor_sha256_ &&
        record_value("grant_receipt_sha256") == grant_receipt_sha256 &&
        record_value("exit_code") == "0";
    if (!header_valid || !record_valid || observation.handle != expected_handle) {
      throw Error("operation ExecCell result binding invalid");
    }
  }

  int await_client_exit() {
    const auto deadline =
        std::chrono::steady_clock::now() + kProductExecIngressHostDeadline;
    while (!client_reaped_) {
      observe_client();
      if (client_reaped_) break;
      if (std::chrono::steady_clock::now() >= deadline) {
        throw Error("operation ExecCell systemd client exit timed out");
      }
      poll(nullptr, 0, 5);
    }
    client_pid_ = -1;
    if (WIFEXITED(client_status_)) return WEXITSTATUS(client_status_);
    return WIFSIGNALED(client_status_) ? 128 + WTERMSIG(client_status_) : 255;
  }

  bool verify_material_artifacts(
      const HostSounioCheckObservation& observation) const {
    const std::string artifact =
        cell_storage_path() + "/loom-sounio-check-" +
        source_sha256_.substr(0, 16) + ".elf";
    const auto measured_regular = [&](const std::string& path,
                                      const std::string& expected_sha256,
                                      std::optional<std::uint64_t> expected_bytes,
                                      std::string_view label) {
      struct stat info {};
      if (lstat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode)) {
        throw Error("operation ExecCell " + std::string(label) +
                    " is absent or non-regular");
      }
      if (info.st_uid != uid_ || info.st_gid != gid_) {
        throw Error("operation ExecCell " + std::string(label) +
                    " custody mismatch uid=" + std::to_string(info.st_uid) +
                    " gid=" + std::to_string(info.st_gid));
      }
      if ((info.st_mode & 0777) != 0600) {
        throw Error("operation ExecCell " + std::string(label) +
                    " mode mismatch");
      }
      if (info.st_size < 0 ||
          (expected_bytes &&
           static_cast<std::uint64_t>(info.st_size) != *expected_bytes)) {
        throw Error("operation ExecCell " + std::string(label) +
                    " size mismatch");
      }
      if (file_sha256(path) != expected_sha256) {
        throw Error("operation ExecCell " + std::string(label) +
                    " hash mismatch");
      }
    };
    const auto record = operation_record_fields(
        observation.record, result_policy_.record_schema);
    const auto artifact_bytes =
        parse_u64(operation_require(record, "artifact_bytes"));
    if (!artifact_bytes) {
      throw Error("operation ExecCell artifact byte count malformed");
    }
    measured_regular(artifact, observation.artifact_sha256, *artifact_bytes,
                     "artifact");
    if (observation.artifact_sha256 != kFrozenOperationArtifactSha256) {
      throw Error("operation ExecCell artifact differs from frozen result");
    }
    const std::string stdout_path = artifact + ".stdout";
    const std::string stderr_path = artifact + ".stderr";
    measured_regular(stdout_path, operation_require(record, "stdout_sha256"),
                     std::nullopt, "stdout");
    measured_regular(stderr_path, operation_require(record, "stderr_sha256"),
                     std::nullopt, "stderr");
    std::size_t entries = 0;
    for (const auto& entry :
         std::filesystem::directory_iterator(cell_storage_path())) {
      if (!entry.is_regular_file()) {
        throw Error("operation ExecCell runtime directory contains non-file");
      }
      ++entries;
    }
    if (entries != 3) {
      throw Error("operation ExecCell runtime directory entry count mismatch=" +
                  std::to_string(entries));
    }
    return true;
  }

  void finish_extinction(HostSounioCheckObservation& observation) {
    std::error_code ignored;
    std::filesystem::remove_all(runtime_storage_path_, ignored);
    std::filesystem::remove_all(runtime_path_, ignored);
    const auto deadline =
        std::chrono::steady_clock::now() + kProductExecIngressHostDeadline;
    for (;;) {
      pollfd candidate{pidfd_.get(), POLLIN, 0};
      const int ready = poll(&candidate, 1, 0);
      observation.pidfd_extinct =
          ready == 1 && (candidate.revents & POLLIN) != 0;
      observation.process_extinct =
          process_witness_original_extinct(pid_, start_tick_);
      observation.cgroup_unpopulated =
          process_witness_cgroup_unpopulated(cgroup_relative_);
      observation.unit_inactive = unit_inactive();
      observation.runtime_directory_extinct =
          path_absent(runtime_path_) && path_absent(runtime_storage_path_);
      if (observation.pidfd_extinct && observation.process_extinct &&
          observation.cgroup_unpopulated && observation.unit_inactive &&
          observation.runtime_directory_extinct) {
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        throw Error("operation ExecCell extinction observation timed out");
      }
      poll(nullptr, 0, 10);
    }
    pid_ = -1;
  }

  void cleanup_noexcept() {
    input_ = UniqueFd();
    output_ = UniqueFd();
    try {
      static_cast<void>(quorum_run_tool(systemctl_, {"stop", unit_},
                                        std::chrono::seconds(2)));
      static_cast<void>(quorum_run_tool(systemctl_, {"reset-failed", unit_},
                                        std::chrono::seconds(2)));
    } catch (...) {
    }
    if (client_pid_ > 0 && !client_reaped_) {
      kill(client_pid_, SIGKILL);
      while (waitpid(client_pid_, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
    std::error_code ignored;
    std::filesystem::remove_all(runtime_storage_path_, ignored);
    std::filesystem::remove_all(runtime_path_, ignored);
    client_pid_ = -1;
    pid_ = -1;
  }

  std::string root_;
  std::string runtime_;
  std::string systemd_run_;
  std::string systemctl_;
  OperationExecCellFixtures fixtures_;
  OperationExecResultPolicy result_policy_;
  std::string generation_;
  std::string nonce_;
  std::string unit_;
  std::string runtime_directory_;
  std::string runtime_path_;
  std::string runtime_storage_path_;
  std::string source_path_;
  std::string source_sha256_;
  std::string runtime_sha256_;
  std::string cgroup_sha256_;
  std::string cgroup_relative_;
  std::string principal_sha256_;
  std::string descriptor_sha256_;
  std::string output_buffer_;
  UniqueFd input_;
  UniqueFd output_;
  UniqueFd pidfd_;
  pid_t client_pid_ = -1;
  pid_t pid_ = -1;
  std::uint64_t start_tick_ = 0;
  std::uint64_t descriptor_device_ = 0;
  std::uint64_t descriptor_inode_ = 0;
  uid_t uid_ = 0;
  gid_t gid_ = 0;
  int client_status_ = 0;
  bool client_reaped_ = false;
  bool ready_ = false;
  bool finished_ = false;
};

int selftest_product_exec_cell_host(
    const std::string& root, const std::string& controller_manifest_path,
    const std::string& controller_runtime_path,
    const std::string& resident_runtime_path,
    const std::string& process_witness_runtime_path,
    const std::string& process_witness_payload_path,
    const std::string& process_witness_manifest_path,
    const std::string& product_root, const std::string& product_runtime,
    const std::string& product_language_runtime,
    const std::string& product_resident_runtime,
    const std::string& product_fixture_manifest_path,
    const std::string& product_fixture_bundle_path,
    const std::string& product_exec_result_manifest_path,
    const std::string& product_provider_hook_fixture_path,
    const std::string& operation_fixture_manifest_path,
    const std::string& operation_fixture_bundle_path,
    const std::string& operation_catalog_manifest_path,
    const std::string& operation_result_manifest_path,
    const std::string& systemd_run_path,
    const std::string& systemctl_path) {
  if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0 ||
      trim(read_file("/proc/1/comm")) != "systemd") {
    throw Error("product ExecCell host selftest requires root under PID1 systemd");
  }
  const Manifest controller_manifest = parse_frozen_manifest(
      controller_manifest_path, kFrozenExecGrantControllerManifestSha256,
      "product ExecCell controller");
  if (controller_manifest.require("stage") != "EFFECT_PARITY_FROZEN" ||
      controller_manifest.require("producing_language") != "OCaml" ||
      controller_manifest.require("language_role") != "EFFECT_PARITY" ||
      controller_manifest.require("semantic_authority") != "Sounio" ||
      controller_manifest.require("action") != "9030" ||
      controller_manifest.require("material_execution") != "false") {
    throw Error("product ExecCell controller manifest state invalid");
  }
  if (file_sha256(controller_runtime_path) !=
      controller_manifest.require("runtime_sha256")) {
    throw Error("product ExecCell controller runtime hash mismatch");
  }
  const ProductExecCellFixtures fixtures = load_product_exec_cell_fixtures(
      product_fixture_manifest_path, product_fixture_bundle_path);
  const Manifest payload_manifest =
      load_process_witness_payload_manifest(process_witness_manifest_path);
  const ProductExecResultPolicy result_policy = load_product_exec_result_policy(
      product_exec_result_manifest_path, fixtures, payload_manifest);
  const OperationExecCellFixtures operation_fixtures =
      load_operation_exec_cell_fixtures(operation_fixture_manifest_path,
                                        operation_fixture_bundle_path);
  const OperationExecResultPolicy operation_result_policy =
      load_operation_exec_result_policy(operation_catalog_manifest_path,
                                        operation_result_manifest_path);
  if (file_sha256(process_witness_payload_path) !=
          payload_manifest.require("executable_sha256") ||
      payload_manifest.require("executable_sha256") !=
          parse_frozen_manifest(product_fixture_manifest_path,
                                kFrozenProductExecCellFixtureManifestSha256,
                                "product ExecCell payload binding")
              .require("payload_sha256")) {
    throw Error("product ExecCell payload binding invalid");
  }

  bool positive_callback = false;
  bool positive_done = false;
  bool distinct_live_principals = false;
  bool controller_extinct = false;
  uid_t lane_uid = 0;
  gid_t lane_gid = 0;
  uid_t exec_uid = 0;
  gid_t exec_gid = 0;
  std::string product_event_sha256;
  std::string product_command_sha256;
  std::string positive_receipt;
  const std::string positive_generation = random_generation_sha256();
  ExecQuorumController positive_controller(
      controller_runtime_path, root, resident_runtime_path,
      positive_generation, "60000");
  ProductExecIngressHostCase positive_case(
      product_root, product_runtime, product_language_runtime,
      product_resident_runtime, systemd_run_path, systemctl_path,
      "current_material", fixtures.command, true, true,
      ProductIngressReply::ResultExact,
      ProductIngressExchangeExpectation::Request,
      [&](const ProductIngressHostRun& lane) {
        positive_callback = true;
        lane_uid = lane.main_uid;
        lane_gid = lane.main_gid;
        product_event_sha256 = lane.event_sha256;
        product_command_sha256 = lane.command_sha256;
        if (!lane.identity_authenticated || !lane.dynamic_user ||
            lane.command_sha256 != fixtures.command_sha256 ||
            !lowercase_digest(lane.event_sha256)) {
          throw Error("product LaneCell command admission binding invalid");
        }
        const QuorumReceipt issue = positive_controller.command(
            "ISSUE", fixtures.frames.at("issue"));
        const QuorumReceipt consume = positive_controller.command(
            "CONSUME", fixtures.frames.at("consume"));
        const bool semantic_consumed =
            issue.require("code") == "0" &&
            consume.require("code") == "0" &&
            consume.require("quorum_ready") == "true" &&
            consume.require("state") == "OUTCOME_PENDING" &&
            issue.require("controller_generation_sha256") ==
                positive_generation &&
            consume.require("controller_generation_sha256") ==
                positive_generation &&
            positive_controller.authenticated();
        HostProcessWitnessCell principal(
            process_witness_runtime_path, process_witness_payload_path,
            process_witness_manifest_path, systemd_run_path, systemctl_path,
            positive_generation, false);
        exec_uid = principal.uid();
        exec_gid = principal.gid();
        distinct_live_principals = principal.distinct_uid() && lane_uid != 0 &&
                                   lane_gid != 0 && exec_uid != lane_uid &&
                                   exec_gid != lane_gid;
        const ProcessWitnessObservation observation =
            principal.execute(semantic_consumed, "CLOSE\n");
        positive_receipt = observation.receipt;
        positive_done = semantic_consumed && distinct_live_principals &&
                        observation.success_complete() &&
                        sha256(positive_receipt) == result_policy.receipt_sha256;
        static_cast<void>(positive_controller.command(
            "CLOSE", fixtures.frames.at("close")));
      },
      result_policy.event_sha256,
      [&](const ProductIngressHostRun&) {
        return ProductIngressResult{result_policy.handle,
                                    result_policy.receipt_sha256,
                                    positive_receipt,
                                    result_policy.manifest_sha256};
      });
  const ProductIngressHostRun positive_lane = positive_case.execute();
  const auto controller_deadline =
      std::chrono::steady_clock::now() + kExecQuorumDeadline;
  while (positive_controller.authenticated() &&
         std::chrono::steady_clock::now() < controller_deadline) {
    poll(nullptr, 0, 5);
  }
  controller_extinct = !positive_controller.authenticated();
  require_product_case(
      positive_callback && positive_done && distinct_live_principals &&
          controller_extinct && positive_lane.hook_code == 0 &&
          positive_lane.identity_authenticated &&
          positive_lane.raw_event_sha256 != result_policy.event_sha256 &&
          positive_lane.event_sha256 == result_policy.event_sha256 &&
          positive_lane.command_sha256 == fixtures.command_sha256 &&
          product_contains(positive_lane.ingress_log, "decision=DENY\t") &&
          product_contains(positive_lane.ingress_log,
                           "reason=descriptor-result-bound-actions-9030+9031+9033+9034\t") &&
          product_contains(positive_lane.ingress_log,
                           "exec_intent_projected=true\t") &&
          product_contains(positive_lane.ingress_log,
                           "exec_intent_action=9034\t") &&
          product_contains(positive_lane.ingress_log, "result_returned=true\t") &&
          product_contains(positive_lane.ingress_log,
                           "raw_event_sha256=" + positive_lane.raw_event_sha256 +
                               "\t") &&
          product_contains(positive_lane.ingress_log,
                           "event_sha256=" + result_policy.event_sha256 + "\t") &&
          product_contains(positive_lane.ingress_log, "exec_attached=true\t") &&
          product_contains(positive_lane.output, "exec-result-present") &&
          product_contains(positive_lane.output, result_policy.handle) &&
          !product_contains(positive_lane.output, "exec-capability"),
      "product ExecCell positive composition failed", positive_lane);

  bool provider_callback = false;
  bool provider_done = false;
  bool provider_distinct_live_principals = false;
  bool provider_controller_extinct = false;
  std::string provider_receipt;
  const std::string provider_generation = random_generation_sha256();
  ExecQuorumController provider_controller(
      controller_runtime_path, root, resident_runtime_path,
      provider_generation, "60000");
  ProductExecIngressHostCase provider_case(
      product_root, product_runtime, product_language_runtime,
      product_resident_runtime, systemd_run_path, systemctl_path,
      "current_material", fixtures.command, true, true,
      ProductIngressReply::ResultExact,
      ProductIngressExchangeExpectation::Request,
      [&](const ProductIngressHostRun& lane) {
        provider_callback = true;
        if (!lane.identity_authenticated || !lane.dynamic_user ||
            lane.command_sha256 != fixtures.command_sha256 ||
            !lowercase_digest(lane.event_sha256)) {
          throw Error("provider LaneCell command admission binding invalid");
        }
        const QuorumReceipt issue = provider_controller.command(
            "ISSUE", fixtures.frames.at("issue"));
        const QuorumReceipt consume = provider_controller.command(
            "CONSUME", fixtures.frames.at("consume"));
        const bool semantic_consumed =
            issue.require("code") == "0" &&
            consume.require("code") == "0" &&
            consume.require("quorum_ready") == "true" &&
            consume.require("state") == "OUTCOME_PENDING" &&
            issue.require("controller_generation_sha256") ==
                provider_generation &&
            consume.require("controller_generation_sha256") ==
                provider_generation &&
            provider_controller.authenticated();
        HostProcessWitnessCell principal(
            process_witness_runtime_path, process_witness_payload_path,
            process_witness_manifest_path, systemd_run_path, systemctl_path,
            provider_generation, false);
        provider_distinct_live_principals =
            principal.distinct_uid() && lane.main_uid != 0 &&
            lane.main_gid != 0 && principal.uid() != lane.main_uid &&
            principal.gid() != lane.main_gid;
        const ProcessWitnessObservation observation =
            principal.execute(semantic_consumed, "CLOSE\n");
        provider_receipt = observation.receipt;
        provider_done = semantic_consumed &&
                        provider_distinct_live_principals &&
                        observation.success_complete() &&
                        sha256(provider_receipt) == result_policy.receipt_sha256;
        static_cast<void>(provider_controller.command(
            "CLOSE", fixtures.frames.at("close")));
      },
      result_policy.event_sha256,
      [&](const ProductIngressHostRun&) {
        return ProductIngressResult{result_policy.handle,
                                    result_policy.receipt_sha256,
                                    provider_receipt,
                                    result_policy.manifest_sha256};
      },
      product_provider_hook_fixture_path);
  const ProductIngressHostRun provider_lane = provider_case.execute();
  const auto provider_controller_deadline =
      std::chrono::steady_clock::now() + kExecQuorumDeadline;
  while (provider_controller.authenticated() &&
         std::chrono::steady_clock::now() < provider_controller_deadline) {
    poll(nullptr, 0, 5);
  }
  provider_controller_extinct = !provider_controller.authenticated();
  require_product_case(
      provider_callback && provider_done &&
          provider_distinct_live_principals && provider_controller_extinct &&
          provider_lane.hook_code == 0 &&
          provider_lane.identity_authenticated &&
          provider_lane.raw_event_sha256 != result_policy.event_sha256 &&
          provider_lane.event_sha256 == result_policy.event_sha256 &&
          provider_lane.command_sha256 == fixtures.command_sha256 &&
          product_contains(provider_lane.ingress_log,
                           "exec_intent_action=9034\t") &&
          product_contains(provider_lane.ingress_log,
                           "result_returned=true\t") &&
          product_contains(provider_lane.ingress_log,
                           "exec_attached=true\t") &&
          product_contains(provider_lane.output,
                           "LOOM_PROVIDER_STARTED schema=loom-provider-abi-v1 provider=codex") &&
          product_contains(provider_lane.output, "exec-result-present") &&
          product_contains(provider_lane.output, result_policy.handle) &&
          !product_contains(provider_lane.output, "exec-capability"),
      "provider lifecycle ExecCell composition failed", provider_lane);

  bool sabotage_callback = false;
  bool sabotage_exec_cell_created = false;
  bool sabotage_payload_executed = false;
  bool sabotage_denied = false;
  std::string sabotage_command_sha256;
  const std::string sabotage_generation = random_generation_sha256();
  ExecQuorumController sabotage_controller(
      controller_runtime_path, root, resident_runtime_path,
      sabotage_generation, "60000");
  ProductExecIngressHostCase sabotage_case(
      product_root, product_runtime, product_language_runtime,
      product_resident_runtime, systemd_run_path, systemctl_path,
      "current_material", fixtures.command + "-sabotage", true, true,
      ProductIngressReply::Exact, ProductIngressExchangeExpectation::Request,
      [&](const ProductIngressHostRun& lane) {
        sabotage_callback = true;
        sabotage_command_sha256 = lane.command_sha256;
        const QuorumReceipt refusal = sabotage_controller.command(
            "ISSUE", fixtures.frames.at("command_mismatch"));
        const bool grant_open = refusal.require("code") == "0" ||
                                refusal.require("quorum_ready") == "true";
        if (grant_open) {
          sabotage_exec_cell_created = true;
          HostProcessWitnessCell principal(
              process_witness_runtime_path, process_witness_payload_path,
              process_witness_manifest_path, systemd_run_path,
              systemctl_path, sabotage_generation, false);
          sabotage_payload_executed =
              principal.execute(true, "CLOSE\n").success_complete();
        }
        sabotage_denied =
            refusal.require("code") == fixtures.command_mismatch_code &&
            refusal.require("quorum_ready") == "false" &&
            refusal.require("controller_terminal") == "true";
      });
  const ProductIngressHostRun sabotage_lane = sabotage_case.execute();
  require_product_case(
      sabotage_callback && sabotage_denied &&
          sabotage_command_sha256 != fixtures.command_sha256 &&
          !sabotage_exec_cell_created && !sabotage_payload_executed &&
          sabotage_lane.hook_code == 0 &&
          product_contains(sabotage_lane.ingress_log, "decision=DENY\t"),
      "product ExecCell command sabotage did not fail closed", sabotage_lane);

  auto result_transport_refused = [&](ProductIngressReply reply,
                                      std::string_view expected_reason) {
    ProductExecIngressHostCase candidate(
        product_root, product_runtime, product_language_runtime,
        product_resident_runtime, systemd_run_path, systemctl_path,
        "current_material", fixtures.command, true, true, reply,
        ProductIngressExchangeExpectation::Request, {},
        result_policy.event_sha256,
        [&](const ProductIngressHostRun&) {
          return ProductIngressResult{result_policy.handle,
                                      result_policy.receipt_sha256,
                                      positive_receipt,
                                      result_policy.manifest_sha256};
        });
    const ProductIngressHostRun lane = candidate.execute();
    return lane.hook_code == 2 &&
           product_contains(lane.output, expected_reason) &&
           !product_contains(lane.output, "exec-result-present") &&
           !product_contains(lane.output, "exec-capability");
  };
  const bool result_binding_closed = result_transport_refused(
      ProductIngressReply::ResultBindingMismatch,
      "product-exec-ingress-response-binding-mismatch");
  const bool result_receipt_closed = result_transport_refused(
      ProductIngressReply::ResultReceiptMismatch,
      "result-transport-receipt-not-frozen-fixture");
  const bool result_manifest_closed = result_transport_refused(
      ProductIngressReply::ResultManifestMismatch,
      "result-transport-manifest-not-frozen-fixture");
  if (!result_binding_closed || !result_receipt_closed ||
      !result_manifest_closed) {
    throw Error("product ExecResult transport sabotage did not fail closed");
  }

  ProductExecIngressHostCase intent_mismatch_case(
      product_root, product_runtime, product_language_runtime,
      product_resident_runtime, systemd_run_path, systemctl_path,
      "current_material", fixtures.command + "-intent-mismatch", true, true,
      ProductIngressReply::Exact,
      ProductIngressExchangeExpectation::RefusalBeforeRequest, {},
      result_policy.event_sha256);
  const ProductIngressHostRun intent_mismatch_lane = intent_mismatch_case.execute();
  const bool intent_mismatch_closed =
      intent_mismatch_lane.hook_code == 2 &&
      product_contains(intent_mismatch_lane.output, "DENY555") &&
      !product_contains(intent_mismatch_lane.output, "exec-result-present") &&
      !product_contains(intent_mismatch_lane.output, "exec-capability");
  if (!intent_mismatch_closed) {
    throw Error("Sounio ExecIntent command mismatch did not fail closed" +
                product_case_diagnostic(intent_mismatch_lane));
  }

  bool operation_provider_callback = false;
  bool operation_provider_done = false;
  bool operation_distinct_live_principals = false;
  bool operation_controller_extinct = false;
  uid_t operation_lane_uid = 0;
  gid_t operation_lane_gid = 0;
  uid_t operation_exec_uid = 0;
  gid_t operation_exec_gid = 0;
  ProductIngressResult operation_transport;
  HostSounioCheckObservation operation_observation;
  const std::string operation_generation = random_generation_sha256();
  ExecQuorumController operation_controller(
      controller_runtime_path, root, resident_runtime_path,
      operation_generation, "60000");
  ProductExecIngressHostCase operation_provider_case(
      product_root, product_runtime, product_language_runtime,
      product_resident_runtime, systemd_run_path, systemctl_path,
      "current_material", operation_fixtures.command, true, true,
      ProductIngressReply::RecordExact,
      ProductIngressExchangeExpectation::Request,
      [&](const ProductIngressHostRun& lane) {
        operation_provider_callback = true;
        operation_lane_uid = lane.main_uid;
        operation_lane_gid = lane.main_gid;
        if (!lane.identity_authenticated || !lane.dynamic_user ||
            lane.event_sha256 != operation_fixtures.event_sha256 ||
            lane.command_sha256 != operation_fixtures.command_sha256) {
          throw Error("operation provider LaneCell admission binding invalid");
        }
        HostSounioCheckCell cell(
            product_root, product_runtime, systemd_run_path, systemctl_path,
            operation_fixtures, operation_result_policy,
            operation_generation);
        operation_exec_uid = cell.uid();
        operation_exec_gid = cell.gid();
        operation_distinct_live_principals =
            operation_lane_uid != 0 && operation_lane_gid != 0 &&
            operation_exec_uid != 0 && operation_exec_gid != 0 &&
            operation_exec_uid != operation_lane_uid &&
            operation_exec_gid != operation_lane_gid;
        const QuorumReceipt issue = operation_controller.command(
            "ISSUE", operation_fixtures.frames.at("issue"));
        const QuorumReceipt consume = operation_controller.command(
            "CONSUME", operation_fixtures.frames.at("consume"));
        const bool semantic_consumed =
            issue.require("code") == "0" && consume.require("code") == "0" &&
            consume.require("quorum_ready") == "true" &&
            consume.require("state") == "OUTCOME_PENDING" &&
            issue.require("controller_generation_sha256") ==
                operation_generation &&
            consume.require("controller_generation_sha256") ==
                operation_generation &&
            operation_controller.authenticated();
        if (!semantic_consumed || !operation_distinct_live_principals) {
          throw Error("operation ExecGrant did not reach the ARM boundary");
        }
        const std::string grant_receipt_sha256 =
            sha256(issue.line + "\n" + consume.line + "\n");
        operation_observation = cell.execute(grant_receipt_sha256);
        const QuorumReceipt close = operation_controller.command(
            "CLOSE", operation_fixtures.frames.at("close"));
        if (close.require("code") != "0" ||
            close.require("state") != "CLOSED" ||
            close.require("controller_generation_sha256") !=
                operation_generation) {
          throw Error("operation ExecGrant CLOSE was not admitted");
        }
        cell.close(operation_observation, sha256(close.line + "\n"));
        operation_provider_done = operation_observation.success_complete() &&
                                  operation_observation.artifact_sha256 ==
                                      kFrozenOperationArtifactSha256;
        operation_transport =
            ProductIngressResult{operation_observation.handle,
                                 operation_observation.record_sha256,
                                 operation_observation.record,
                                 operation_result_policy.manifest_sha256};
      },
      operation_fixtures.event_sha256,
      [&](const ProductIngressHostRun&) { return operation_transport; },
      product_provider_hook_fixture_path);
  const ProductIngressHostRun operation_provider_lane =
      operation_provider_case.execute();
  const auto operation_controller_deadline =
      std::chrono::steady_clock::now() + kExecQuorumDeadline;
  while (operation_controller.authenticated() &&
         std::chrono::steady_clock::now() < operation_controller_deadline) {
    poll(nullptr, 0, 5);
  }
  operation_controller_extinct = !operation_controller.authenticated();
  require_product_case(
      operation_provider_callback && operation_provider_done &&
          operation_distinct_live_principals && operation_controller_extinct &&
          operation_provider_lane.hook_code == 0 &&
          operation_provider_lane.identity_authenticated &&
          operation_provider_lane.raw_event_sha256 !=
              operation_fixtures.event_sha256 &&
          operation_provider_lane.event_sha256 ==
              operation_fixtures.event_sha256 &&
          operation_provider_lane.command_sha256 ==
              operation_fixtures.command_sha256 &&
          product_contains(operation_provider_lane.ingress_log,
                           "reason=descriptor-result-bound-actions-9030+9031+9035+9036\t") &&
          product_contains(operation_provider_lane.ingress_log,
                           "exec_intent_action=9035\t") &&
          product_contains(operation_provider_lane.ingress_log,
                           "exec_projection_kind=operation-catalog\t") &&
          product_contains(operation_provider_lane.ingress_log,
                           "result_action=9036\t") &&
          product_contains(operation_provider_lane.ingress_log,
                           "result_kind=operation-record\t") &&
          product_contains(operation_provider_lane.output,
                           "LOOM_PROVIDER_STARTED schema=loom-provider-abi-v1 provider=codex") &&
          product_contains(operation_provider_lane.output,
                           "exec-result-record-present") &&
          product_contains(operation_provider_lane.output,
                           operation_observation.handle) &&
          !product_contains(operation_provider_lane.output, "exec-capability"),
      "provider operation ExecCell composition failed", operation_provider_lane);

  bool operation_sabotage_callback = false;
  bool operation_sabotage_cell_created = false;
  bool operation_sabotage_denied = false;
  const std::string operation_sabotage_generation = random_generation_sha256();
  ExecQuorumController operation_sabotage_controller(
      controller_runtime_path, root, resident_runtime_path,
      operation_sabotage_generation, "60000");
  const std::string operation_sabotage_command =
      "loom-exec-cell-v2 sounio-check source=tests/verify-ir/call_a.sio";
  ProductExecIngressHostCase operation_sabotage_case(
      product_root, product_runtime, product_language_runtime,
      product_resident_runtime, systemd_run_path, systemctl_path,
      "current_material", operation_sabotage_command, true, true,
      ProductIngressReply::Exact, ProductIngressExchangeExpectation::Request,
      [&](const ProductIngressHostRun& lane) {
        operation_sabotage_callback = true;
        if (lane.command_sha256 == operation_fixtures.command_sha256) {
          throw Error("operation command sabotage did not change the digest");
        }
        const QuorumReceipt refusal = operation_sabotage_controller.command(
            "ISSUE", operation_fixtures.frames.at("command_mismatch"));
        const bool grant_open = refusal.require("code") == "0" ||
                                refusal.require("quorum_ready") == "true";
        if (grant_open) {
          operation_sabotage_cell_created = true;
        }
        operation_sabotage_denied =
            refusal.require("code") == operation_fixtures.command_mismatch_code &&
            refusal.require("quorum_ready") == "false" &&
            refusal.require("controller_terminal") == "true";
      });
  const ProductIngressHostRun operation_sabotage_lane =
      operation_sabotage_case.execute();
  require_product_case(
      operation_sabotage_callback && operation_sabotage_denied &&
          !operation_sabotage_cell_created &&
          operation_sabotage_lane.hook_code == 0 &&
          product_contains(operation_sabotage_lane.ingress_log,
                           "exec_intent_action=9035\t") &&
          !product_contains(operation_sabotage_lane.output,
                            "exec-result-record-present"),
      "operation command sabotage created an ExecCell",
      operation_sabotage_lane);

  auto operation_transport_refused = [&](ProductIngressReply reply,
                                         std::string_view expected_reason) {
    ProductExecIngressHostCase candidate(
        product_root, product_runtime, product_language_runtime,
        product_resident_runtime, systemd_run_path, systemctl_path,
        "current_material", operation_fixtures.command, true, true, reply,
        ProductIngressExchangeExpectation::Request, {},
        operation_fixtures.event_sha256,
        [&](const ProductIngressHostRun&) { return operation_transport; });
    const ProductIngressHostRun lane = candidate.execute();
    return lane.hook_code == 2 && product_contains(lane.output, expected_reason) &&
           !product_contains(lane.output, "exec-result-record-present") &&
           !product_contains(lane.output, "exec-capability");
  };
  const bool operation_result_binding_closed = operation_transport_refused(
      ProductIngressReply::RecordBindingMismatch,
      "product-exec-ingress-response-binding-mismatch");
  const bool operation_result_digest_closed = operation_transport_refused(
      ProductIngressReply::RecordDigestMismatch,
      "exec-result-record-transport-record-hash-mismatch");
  const bool operation_result_manifest_closed = operation_transport_refused(
      ProductIngressReply::RecordManifestMismatch,
      "exec-result-record-transport-manifest-mismatch");
  if (!operation_result_binding_closed || !operation_result_digest_closed ||
      !operation_result_manifest_closed) {
    throw Error("operation ExecResult transport sabotage did not fail closed");
  }

  std::cout
      << "loom-product-exec-cell-host: PASS semantic_authority=Sounio"
      << " action=9030 lane_action=9031 result_action=9033 intent_action=9034"
      << " intent_sha256=" << fixtures.intent_sha256
      << " command_sha256=" << fixtures.command_sha256
      << " product_event_sha256=" << product_event_sha256
      << " raw_product_event_sha256=" << positive_lane.raw_event_sha256
      << " product_command_sha256=" << product_command_sha256
      << " lane_uid=" << lane_uid << " lane_gid=" << lane_gid
      << " exec_uid=" << exec_uid << " exec_gid=" << exec_gid
      << " simultaneous_distinct_dynamic_users=true"
      << " same_pid_exec_transition=true outcome=DONE"
      << " extinction_complete=true controller_extinct=true"
      << " command_mismatch=" << fixtures.command_mismatch_result
      << " causal_sabotage=PASS"
      << " sabotage_exec_cell_created=false"
      << " sabotage_payload_executed=false"
      << " result_handle=" << result_policy.handle
      << " result_receipt_sha256=" << result_policy.receipt_sha256
      << " result_manifest_sha256=" << result_policy.manifest_sha256
      << " result_returned=true result_presenter=read-only"
      << " result_binding_sabotage=closed"
      << " result_receipt_sabotage=closed"
      << " result_manifest_sabotage=closed"
      << " exact_fixture_hook_switched=true local_exec_capability_used=false"
      << " raw_event_separate=true event_projection=Sounio-9034"
      << " event_override=false intent_command_mismatch=DENY555"
      << " provider_hook_switched=true provider_lifecycle_attached=true"
      << " provider_fixture_language=OCaml"
      << " operation_catalog_action=9035 operation_result_action=9036"
      << " operation_command_sha256=" << operation_fixtures.command_sha256
      << " operation_event_sha256=" << operation_fixtures.event_sha256
      << " operation_record_sha256=" << operation_observation.record_sha256
      << " operation_result_handle=" << operation_observation.handle
      << " operation_artifact_sha256="
      << operation_observation.artifact_sha256
      << " operation_lane_uid=" << operation_lane_uid
      << " operation_lane_gid=" << operation_lane_gid
      << " operation_exec_uid=" << operation_exec_uid
      << " operation_exec_gid=" << operation_exec_gid
      << " operation_simultaneous_distinct_dynamic_users=true"
      << " operation_protocol=READY+ARM+CLOSE"
      << " operation_close_receipt_bound=true"
      << " operation_runtime_storage=systemd-private-symlink"
      << " operation_runtime_custody_pre_close=true"
      << " operation_runtime_released_after_close=true"
      << " operation_principal_self_measured=true"
      << " operation_descriptor_self_measured=true"
      << " operation_pidfd_extinct=true"
      << " operation_cgroup_unpopulated=true"
      << " operation_unit_inactive=true"
      << " operation_runtime_directory_extinct=true"
      << " operation_record_returned=true"
      << " operation_result_presenter=read-only"
      << " operation_command_mismatch="
      << operation_fixtures.command_mismatch_result
      << " operation_sabotage_cell_created=false"
      << " operation_result_binding_sabotage=closed"
      << " operation_result_digest_sabotage=closed"
      << " operation_result_manifest_sabotage=closed"
      << " python_executed=false rust_executed=false"
      << " exec_cell_attached=true material_execution=true"
      << " test_only=true production_activation=false launch_open=false"
      << " recycle_open=false commit_attached=false ci_attached=false"
      << " parity_open=false claim_ready=false"
      << " product_fixture_manifest_sha256="
      << file_sha256(product_fixture_manifest_path)
      << " product_fixture_bundle_sha256="
      << file_sha256(product_fixture_bundle_path)
      << " payload_sha256=" << file_sha256(process_witness_payload_path)
      << " controller_runtime_sha256="
      << file_sha256(controller_runtime_path)
      << " product_runtime_sha256=" << file_sha256(product_runtime)
      << " process_witness_runtime_sha256="
      << file_sha256(process_witness_runtime_path) << '\n';
  return 0;
}
