open Unix

exception Error of string

let pinned_manifest_sha256 =
  "25c58880cfe568a3b55b479df4515b63589c2e4dca703c97917e5cbdef0f1561"

type entry = {
  name : string;
  word0 : int;
  semantic_event_sha256 : string;
  command_template_sha256 : string;
  argument_schema_sha256 : string;
  result_schema_sha256 : string;
  sandbox_profile_sha256 : string;
}

type policy = {
  manifest_sha256 : string;
  source_sha256 : string;
  executable_sha256 : string;
  runtime : string;
  wire_schema : int;
  common_word1 : int;
  project_decision : string;
  fields_schema : string;
  catalog_schema : string;
  catalog_sha256 : string;
  calibration : entry;
  sounio_check : entry;
  unknown_operation_word0 : int;
  unknown_operation_decision : string;
  invalid_argument_word0 : int;
  invalid_argument_decision : string;
}

type projection = {
  operation : string;
  source_path : string option;
  source_sha256 : string option;
  semantic_event_sha256 : string;
  command_template_sha256 : string;
  argument_schema_sha256 : string;
  result_schema_sha256 : string;
  sandbox_profile_sha256 : string;
  catalog_sha256 : string;
  manifest_sha256 : string;
  authority_source_sha256 : string;
  authority_executable_sha256 : string;
  authority_output_sha256 : string;
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
  configured_path ~name:"SOUNIO_LOOM_EXEC_OPERATION_CATALOG_MANIFEST"
    ~default:(Filename.concat root "tools/loom/exec_operation_catalog.freeze.v1")

let verify_file root manifest path_key hash_key reason =
  let path = Filename.concat root (required manifest path_key) in
  ignore (require_regular_file path);
  if sha256_file path <> required manifest hash_key then failf "%s" reason

let choose_runtime root manifest =
  let repository_runtime =
    Filename.concat root
      "tools/loom/_build/default/src/sounio-loom-exec-operation-catalog"
  in
  let installed_runtime =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-exec-operation-catalog"
  in
  let default =
    if Sys.file_exists repository_runtime then repository_runtime
    else installed_runtime
  in
  let selected =
    configured_path ~name:"SOUNIO_LOOM_EXEC_OPERATION_CATALOG_RUNTIME" ~default
  in
  let stat = require_regular_file selected in
  if stat.st_perm land 0o111 = 0 then failf "exec-catalog-runtime-not-executable";
  if sha256_file selected <> required manifest "executable_sha256" then
    failf "exec-catalog-runtime-hash-mismatch";
  Unix.realpath selected

let load_entry manifest prefix word_key =
  { name = required manifest (prefix ^ "_name");
    word0 = required manifest word_key |> decimal word_key;
    semantic_event_sha256 =
      required manifest (prefix ^ "_semantic_event_sha256")
      |> digest (prefix ^ "-semantic-event");
    command_template_sha256 =
      required manifest (prefix ^ "_command_template_sha256")
      |> digest (prefix ^ "-command-template");
    argument_schema_sha256 =
      required manifest (prefix ^ "_argument_schema_sha256")
      |> digest (prefix ^ "-argument-schema");
    result_schema_sha256 =
      required manifest (prefix ^ "_result_schema_sha256")
      |> digest (prefix ^ "-result-schema");
    sandbox_profile_sha256 =
      required manifest (prefix ^ "_sandbox_profile_sha256")
      |> digest (prefix ^ "-sandbox-profile") }

let load ~root =
  let root = Unix.realpath root in
  let path = manifest_path root in
  ignore (require_regular_file path);
  if sha256_file path <> pinned_manifest_sha256 then
    failf "exec-catalog-manifest-hash-mismatch";
  let manifest = parse_manifest path in
  exact manifest "schema" "loom-exec-operation-catalog-freeze-v1";
  exact manifest "stage" "SEMANTICS_FROZEN";
  exact manifest "producing_language" "Sounio";
  exact manifest "language_role" "SEMANTIC_AUTHORITY";
  exact manifest "action" "9035";
  exact manifest "concept_id" "SOUNIO-LOOM-EXEC-OPERATION-CATALOG";
  exact manifest "catalog_entries" "calibration,sounio-check";
  exact manifest "unknown_operation" "DENY562";
  exact manifest "invalid_argument" "DENY563";
  exact manifest "write_effect" "DENY564";
  exact manifest "template_mismatch" "DENY567";
  exact manifest "causal_sabotage" "PASS";
  exact manifest "load_bearing_rule"
    "operation_specific_command_template_sha256_equal";
  exact manifest "arbitrary_shell" "false";
  exact manifest "expected_results_encoded_in_material_layer" "false";
  exact manifest "ocaml_catalog_projection_attached" "false";
  exact manifest "host_payload_selection_attached" "false";
  exact manifest "provider_lifecycle_attached" "false";
  exact manifest "general_exec_attached" "false";
  exact manifest "production_activation" "false";
  exact manifest "parity_open" "false";
  exact manifest "claim_ready" "false";
  List.iter
    (fun (path_key, hash_key, reason) ->
      verify_file root manifest path_key hash_key reason)
    [ ("garden_path", "garden_sha256", "exec-catalog-garden-hash-mismatch");
      ("contract_path", "contract_sha256", "exec-catalog-contract-hash-mismatch");
      ("source_path", "source_sha256", "exec-catalog-source-hash-mismatch");
      ("entrypoint_path", "entrypoint_sha256", "exec-catalog-entrypoint-hash-mismatch");
      ("build_script_path", "build_script_sha256", "exec-catalog-build-script-hash-mismatch");
      ("selftest_path", "selftest_sha256", "exec-catalog-selftest-hash-mismatch");
      ("evidence_path", "evidence_sha256", "exec-catalog-evidence-hash-mismatch");
      ("parent_9030_manifest_path", "parent_9030_manifest_sha256", "exec-catalog-parent-9030-hash-mismatch");
      ("parent_9031_manifest_path", "parent_9031_manifest_sha256", "exec-catalog-parent-9031-hash-mismatch");
      ("parent_9033_manifest_path", "parent_9033_manifest_sha256", "exec-catalog-parent-9033-hash-mismatch");
      ("parent_9034_manifest_path", "parent_9034_manifest_sha256", "exec-catalog-parent-9034-hash-mismatch");
      ("toolchain_wrapper_path", "toolchain_wrapper_sha256", "exec-catalog-toolchain-wrapper-hash-mismatch");
      ("toolchain_compiler_path", "toolchain_compiler_sha256", "exec-catalog-toolchain-compiler-hash-mismatch") ];
  { manifest_sha256 = pinned_manifest_sha256;
    source_sha256 = required manifest "source_sha256";
    executable_sha256 = required manifest "executable_sha256";
    runtime = choose_runtime root manifest;
    wire_schema = required manifest "wire_schema" |> decimal "wire-schema";
    common_word1 = required manifest "common_word1" |> decimal "common-word1";
    project_decision = required manifest "project_decision";
    fields_schema = required manifest "fields_schema";
    catalog_schema = required manifest "catalog_schema";
    catalog_sha256 = required manifest "catalog_sha256" |> digest "catalog";
    calibration = load_entry manifest "calibration" "calibration_word0";
    sounio_check = load_entry manifest "sounio_check" "sounio_check_word0";
    unknown_operation_word0 =
      required manifest "unknown_operation_word0" |> decimal "unknown-operation-word0";
    unknown_operation_decision = required manifest "unknown_operation_decision";
    invalid_argument_word0 =
      required manifest "invalid_argument_word0" |> decimal "invalid-argument-word0";
    invalid_argument_decision = required manifest "invalid_argument_decision" }

let authority_frame policy word0 =
  Printf.sprintf "%d %d %d\n" policy.wire_schema word0 policy.common_word1

let expected_output policy entry =
  String.concat "\n"
    [ policy.project_decision; policy.fields_schema; policy.catalog_schema;
      policy.catalog_sha256; entry.name; entry.semantic_event_sha256;
      entry.command_template_sha256; entry.argument_schema_sha256;
      entry.result_schema_sha256; entry.sandbox_profile_sha256 ]

let control policy word0 decision label =
  let code, output =
    Loom_exec_intent.process_exchange policy.runtime
      (authority_frame policy word0)
  in
  if code <> 42 || output <> decision then
    failf "exec-catalog-%s-control-diverged:%d:%s" label code output

let valid_relative_sio path =
  path <> "" && Filename.is_relative path && Filename.check_suffix path ".sio"
  && not (String.contains path '\000')
  && String.for_all
       (function
         | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' | '-' | '.' | '/' -> true
         | _ -> false)
       path
  && List.for_all (fun part -> part <> "" && part <> "." && part <> "..")
       (String.split_on_char '/' path)

let source_binding ~root policy source =
  if not (valid_relative_sio source) then (
    control policy policy.invalid_argument_word0 policy.invalid_argument_decision
      "invalid-argument";
    failf "exec-catalog-invalid-argument-denied:%s" policy.invalid_argument_decision);
  let root = Unix.realpath root in
  let path = Filename.concat root source in
  let stat = require_regular_file path in
  if stat.st_kind <> S_REG then failf "exec-catalog-source-not-regular";
  let resolved = Unix.realpath path in
  let prefix = root ^ Filename.dir_sep in
  if String.length resolved <= String.length prefix
     || String.sub resolved 0 (String.length prefix) <> prefix
  then failf "exec-catalog-source-outside-worktree";
  (source, sha256_file resolved)

let project ~root ~operation ~source =
  let policy = load ~root in
  let entry, source_path, source_sha256 =
    if operation = policy.calibration.name then
      (match source with
      | None -> (policy.calibration, None, None)
      | Some _ -> failf "exec-catalog-calibration-rejects-source")
    else if operation = policy.sounio_check.name then
      let source =
        match source with
        | Some value -> value
        | None ->
            control policy policy.invalid_argument_word0
              policy.invalid_argument_decision "invalid-argument";
            failf "exec-catalog-invalid-argument-denied:%s"
              policy.invalid_argument_decision
      in
      let path, digest = source_binding ~root policy source in
      (policy.sounio_check, Some path, Some digest)
    else (
      control policy policy.unknown_operation_word0
        policy.unknown_operation_decision "unknown-operation";
      failf "exec-catalog-unknown-operation-denied:%s"
        policy.unknown_operation_decision)
  in
  let code, output =
    Loom_exec_intent.process_exchange policy.runtime
      (authority_frame policy entry.word0)
  in
  if code <> 0 then failf "exec-catalog-authority-refused:%d:%s" code output;
  if output <> expected_output policy entry then
    failf "exec-catalog-authority-output-mismatch";
  { operation = entry.name; source_path; source_sha256;
    semantic_event_sha256 = entry.semantic_event_sha256;
    command_template_sha256 = entry.command_template_sha256;
    argument_schema_sha256 = entry.argument_schema_sha256;
    result_schema_sha256 = entry.result_schema_sha256;
    sandbox_profile_sha256 = entry.sandbox_profile_sha256;
    catalog_sha256 = policy.catalog_sha256;
    manifest_sha256 = policy.manifest_sha256;
    authority_source_sha256 = policy.source_sha256;
    authority_executable_sha256 = policy.executable_sha256;
    authority_output_sha256 = sha256 output }
