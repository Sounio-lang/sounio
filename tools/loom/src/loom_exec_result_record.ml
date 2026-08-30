open Unix

exception Error of string

let pinned_manifest_sha256 =
  "58d4a49c5b2462261ee53cd06f3ca8e29d363c1a38bf47274fa98a67b79cc569"

type policy = {
  manifest_sha256 : string;
  source_sha256 : string;
  executable_sha256 : string;
  runtime : string;
  wire_schema : int;
  positive_word0 : int;
  positive_word1 : int;
  artifact_binding_word0 : int;
  issue_decision : string;
  artifact_binding_decision : string;
  fields_schema : string;
  record_schema : string;
  operation : string;
  catalog_sha256 : string;
  catalog_result_schema_sha256 : string;
  record_schema_sha256 : string;
  canonical_fields : string;
  handle_recipe : string;
  record_hash_recipe : string;
}

type binding = {
  event_sha256 : string;
  generation_sha256 : string;
  principal_sha256 : string;
  descriptor_binding_sha256 : string;
  grant_receipt_sha256 : string;
}

type issued = {
  record : string;
  record_sha256 : string;
  handle : string;
  authority_output_sha256 : string;
  manifest_sha256 : string;
}

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let sha256 = Loom_exec_intent.sha256
let sha256_file = Loom_exec_intent.sha256_file
let parse_manifest = Loom_exec_intent.parse_manifest
let required = Loom_exec_intent.required
let exact = Loom_exec_intent.exact
let decimal = Loom_exec_intent.decimal
let digest = Loom_exec_intent.digest
let require_regular_file = Loom_exec_intent.require_regular_file

let configured_path ~name ~default =
  Loom_exec_intent.configured_path ~name ~default

let manifest_path root =
  configured_path ~name:"SOUNIO_LOOM_EXEC_RESULT_RECORD_MANIFEST"
    ~default:(Filename.concat root "tools/loom/exec_result_record.freeze.v1")

let verify_file root manifest path_key hash_key reason =
  let path = Filename.concat root (required manifest path_key) in
  ignore (require_regular_file path);
  if sha256_file path <> required manifest hash_key then failf "%s" reason

let choose_runtime root manifest =
  let repository_runtime =
    Filename.concat root
      "tools/loom/_build/default/src/sounio-loom-exec-result-record"
  in
  let installed_runtime =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-exec-result-record"
  in
  let default =
    if Sys.file_exists repository_runtime then repository_runtime
    else installed_runtime
  in
  let selected =
    configured_path ~name:"SOUNIO_LOOM_EXEC_RESULT_RECORD_RUNTIME" ~default
  in
  let stat = require_regular_file selected in
  if stat.st_perm land 0o111 = 0 then
    failf "exec-result-record-runtime-not-executable";
  if sha256_file selected <> required manifest "executable_sha256" then
    failf "exec-result-record-runtime-hash-mismatch";
  Unix.realpath selected

let load ~root =
  let root = Unix.realpath root in
  let path = manifest_path root in
  ignore (require_regular_file path);
  if sha256_file path <> pinned_manifest_sha256 then
    failf "exec-result-record-manifest-hash-mismatch";
  let manifest = parse_manifest path in
  exact manifest "schema" "loom-exec-result-record-freeze-v1";
  exact manifest "stage" "SEMANTICS_FROZEN";
  exact manifest "producing_language" "Sounio";
  exact manifest "language_role" "SEMANTIC_AUTHORITY";
  exact manifest "action" "9036";
  exact manifest "concept_id" "SOUNIO-LOOM-EXEC-RESULT-RECORD";
  exact manifest "handle_is_bearer" "false";
  exact manifest "handle_is_execution_authority" "false";
  exact manifest "artifact_executed" "false";
  exact manifest "expected_results_encoded_in_material_layer" "false";
  exact manifest "causal_sabotage" "PASS";
  exact manifest "load_bearing_rule"
    "artifact_sha256_equal_canonical_record_field";
  exact manifest "ocaml_record_projection_attached" "false";
  exact manifest "dynamic_user_host_attached" "false";
  exact manifest "provider_result_returned" "false";
  exact manifest "production_activation" "false";
  exact manifest "parity_open" "false";
  exact manifest "claim_ready" "false";
  List.iter
    (fun (path_key, hash_key, reason) ->
      verify_file root manifest path_key hash_key reason)
    [ ("garden_path", "garden_sha256", "exec-result-record-garden-hash-mismatch");
      ("contract_path", "contract_sha256", "exec-result-record-contract-hash-mismatch");
      ("source_path", "source_sha256", "exec-result-record-source-hash-mismatch");
      ("entrypoint_path", "entrypoint_sha256", "exec-result-record-entrypoint-hash-mismatch");
      ("build_script_path", "build_script_sha256", "exec-result-record-build-hash-mismatch");
      ("selftest_path", "selftest_sha256", "exec-result-record-selftest-hash-mismatch");
      ("evidence_path", "evidence_sha256", "exec-result-record-evidence-hash-mismatch");
      ("parent_9035_manifest_path", "parent_9035_manifest_sha256", "exec-result-record-parent-hash-mismatch");
      ("toolchain_wrapper_path", "toolchain_wrapper_sha256", "exec-result-record-wrapper-hash-mismatch");
      ("toolchain_compiler_path", "toolchain_compiler_sha256", "exec-result-record-compiler-hash-mismatch") ];
  { manifest_sha256 = pinned_manifest_sha256;
    source_sha256 = required manifest "source_sha256";
    executable_sha256 = required manifest "executable_sha256";
    runtime = choose_runtime root manifest;
    wire_schema = required manifest "wire_schema" |> decimal "wire-schema";
    positive_word0 = required manifest "positive_word0" |> decimal "positive-word0";
    positive_word1 = required manifest "positive_word1" |> decimal "positive-word1";
    artifact_binding_word0 =
      required manifest "artifact_binding_word0" |> decimal "artifact-binding-word0";
    issue_decision = required manifest "issue_decision";
    artifact_binding_decision = required manifest "artifact_binding_decision";
    fields_schema = required manifest "fields_schema";
    record_schema = required manifest "record_schema";
    operation = required manifest "operation";
    catalog_sha256 = required manifest "catalog_sha256" |> digest "catalog";
    catalog_result_schema_sha256 =
      required manifest "catalog_result_schema_sha256"
      |> digest "catalog-result-schema";
    record_schema_sha256 =
      required manifest "record_schema_sha256" |> digest "record-schema";
    canonical_fields = required manifest "canonical_fields";
    handle_recipe = required manifest "handle_recipe";
    record_hash_recipe = required manifest "record_hash_recipe" }

let authority_frame policy word0 =
  Printf.sprintf "%d %d %d\n" policy.wire_schema word0 policy.positive_word1

let expected_output policy =
  String.concat "\n"
    [ policy.issue_decision; policy.fields_schema; policy.record_schema;
      policy.operation; policy.catalog_sha256;
      policy.catalog_result_schema_sha256; policy.record_schema_sha256;
      policy.canonical_fields; policy.handle_recipe;
      policy.record_hash_recipe; "handle_is_bearer=false";
      "handle_is_execution_authority=false"; "artifact_executed=false" ]

let require_digest label value = ignore (digest label value)

let artifact_binding_control ~root =
  let policy = load ~root in
  let code, output =
    Loom_exec_intent.process_exchange policy.runtime
      (authority_frame policy policy.artifact_binding_word0)
  in
  if code <> 42 || output <> policy.artifact_binding_decision then
    failf "exec-result-record-artifact-binding-control-diverged:%d:%s"
      code output;
  (policy, output)

let issue ~root ~(material : Loom_exec_catalog.material_result)
    ~(binding : binding) =
  let policy = load ~root in
  let projection = material.plan.projection in
  List.iter (fun (label, value) -> require_digest label value)
    [ ("event", binding.event_sha256);
      ("generation", binding.generation_sha256);
      ("principal", binding.principal_sha256);
      ("descriptor-binding", binding.descriptor_binding_sha256);
      ("grant-receipt", binding.grant_receipt_sha256);
      ("artifact", material.artifact_sha256);
      ("stdout", material.stdout_sha256);
      ("stderr", material.stderr_sha256);
      ("diagnostics", material.diagnostics_sha256) ];
  if projection.operation <> policy.operation ||
     projection.catalog_sha256 <> policy.catalog_sha256 ||
     projection.result_schema_sha256 <> policy.catalog_result_schema_sha256 ||
     binding.event_sha256 <> projection.semantic_event_sha256 then
    failf "exec-result-record-catalog-binding-mismatch";
  let artifact = require_regular_file material.plan.output_path in
  if artifact.st_size <> material.artifact_bytes ||
     sha256_file material.plan.output_path <> material.artifact_sha256 then
    failf "exec-result-record-artifact-binding-mismatch";
  let code, output =
    Loom_exec_intent.process_exchange policy.runtime
      (authority_frame policy policy.positive_word0)
  in
  if code <> 0 || output <> expected_output policy then
    failf "exec-result-record-authority-diverged:%d:%s" code output;
  let source_sha256 = Option.get projection.source_sha256 in
  let record =
    String.concat "\n"
      [ policy.record_schema;
        "operation=" ^ policy.operation;
        "event_sha256=" ^ binding.event_sha256;
        "command_template_sha256=" ^ projection.command_template_sha256;
        "generation_sha256=" ^ binding.generation_sha256;
        "source_sha256=" ^ source_sha256;
        "compiler_sha256=" ^ material.plan.compiler_sha256;
        "argv_sha256=" ^ material.plan.argv_sha256;
        "artifact_sha256=" ^ material.artifact_sha256;
        "artifact_bytes=" ^ string_of_int material.artifact_bytes;
        "stdout_sha256=" ^ material.stdout_sha256;
        "stderr_sha256=" ^ material.stderr_sha256;
        "diagnostics_sha256=" ^ material.diagnostics_sha256;
        "sandbox_profile_sha256=" ^ projection.sandbox_profile_sha256;
        "principal_sha256=" ^ binding.principal_sha256;
        "descriptor_binding_sha256=" ^ binding.descriptor_binding_sha256;
        "grant_receipt_sha256=" ^ binding.grant_receipt_sha256;
        "exit_code=0" ] ^ "\n"
  in
  let record_sha256 = sha256 record in
  { record; record_sha256;
    handle = Printf.sprintf "loom-result-v2:%s:%s:%s"
      binding.event_sha256 binding.generation_sha256 record_sha256;
    authority_output_sha256 = sha256 output;
    manifest_sha256 = policy.manifest_sha256 }
