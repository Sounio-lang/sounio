constexpr std::string_view kFrozenProductExecCellFixtureManifestSha256 =
    "c06fc76a5fc24ec430b843a4f6e11f99c0e1fd034af61a304fdee11a0889498e";
constexpr std::string_view kFrozenExecResultHandleManifestSha256 =
    "7c1955299f2b308c331760764c289b6617910549b5dfc6bc4cf1969457306a55";

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
                           "reason=descriptor-result-bound-actions-9030+9031+9033\t") &&
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

  std::cout
      << "loom-product-exec-cell-host: PASS semantic_authority=Sounio"
      << " action=9030 lane_action=9031 result_action=9033"
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
      << " raw_event_separate=true event_override=test-only+probe-only"
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
