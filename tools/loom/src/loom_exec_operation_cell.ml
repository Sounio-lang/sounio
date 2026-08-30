open Unix

exception Error of string

let pinned_grant_manifest_sha256 =
  "4abfbc8a0fa9cdd1c2164f9f72ea4d408939b3d53497fa7fcaf008d71b1ea1e4"

type grant_policy = {
  manifest_sha256 : string;
  command : string;
  command_sha256 : string;
  event_sha256 : string;
  intent_sha256 : string;
  source : string;
}

type principal = {
  pid : int;
  start_tick : string;
  uid : int;
  gid : int;
  cgroup_sha256 : string;
  executable_sha256 : string;
  unit : string;
  canonical : string;
  sha256 : string;
}

type descriptor_binding = {
  device : int;
  inode : int;
  canonical : string;
  sha256 : string;
}

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let sha256 = Loom_exec_intent.sha256
let sha256_file = Loom_exec_intent.sha256_file
let parse_manifest = Loom_exec_intent.parse_manifest
let required = Loom_exec_intent.required
let exact = Loom_exec_intent.exact
let digest = Loom_exec_intent.digest
let require_regular_file = Loom_exec_intent.require_regular_file

let configured_path ~name ~default =
  Loom_exec_intent.configured_path ~name ~default

let streaming_sha256_file path =
  let stat = Unix.lstat path in
  if stat.st_kind <> S_REG then
    failf "exec-operation-cell-executable-not-regular";
  let channel = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () ->
      Cryptokit.hash_channel (Cryptokit.Hash.sha256 ()) channel
      |> Cryptokit.transform_string (Cryptokit.Hexa.encode ()))

let read_bounded_file path =
  let channel = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () ->
      let limit = 64 * 1024 in
      let bytes = Bytes.create 4096 in
      let buffer = Buffer.create 4096 in
      let rec read total =
        match input channel bytes 0 (Bytes.length bytes) with
        | 0 -> Buffer.contents buffer
        | count ->
            let total = total + count in
            if total > limit then failf "exec-operation-cell-proc-file-too-large";
            Buffer.add_subbytes buffer bytes 0 count;
            read total
      in
      read 0)

let verify_file root manifest path_key hash_key reason =
  let path = Filename.concat root (required manifest path_key) in
  ignore (require_regular_file path);
  if sha256_file path <> required manifest hash_key then failf "%s" reason

let load_grant_policy ~root =
  let root = Unix.realpath root in
  let path =
    configured_path ~name:"SOUNIO_LOOM_EXEC_OPERATION_GRANT_FIXTURE_MANIFEST"
      ~default:(Filename.concat root
        "tools/loom/exec_operation_grant_fixture.freeze.v1")
  in
  ignore (require_regular_file path);
  if sha256_file path <> pinned_grant_manifest_sha256 then
    failf "exec-operation-cell-grant-manifest-hash-mismatch";
  let manifest = parse_manifest path in
  exact manifest "schema" "loom-exec-operation-grant-fixture-freeze-v1";
  exact manifest "stage" "SEMANTICS_FROZEN";
  exact manifest "producing_language" "Sounio";
  exact manifest "language_role" "SEMANTIC_FIXTURE_PRODUCER";
  exact manifest "semantic_authority" "Sounio";
  exact manifest "action" "9030";
  exact manifest "catalog_action" "9035";
  exact manifest "result_action" "9036";
  exact manifest "command_mismatch_result" "DENY492";
  exact manifest "causal_sabotage" "PASS";
  exact manifest "arbitrary_shell" "false";
  exact manifest "expected_results_encoded_in_material_layer" "false";
  exact manifest "material_grant" "false";
  exact manifest "material_execution" "false";
  exact manifest "host_payload_selection_attached" "false";
  exact manifest "provider_lifecycle_attached" "false";
  exact manifest "production_activation" "false";
  exact manifest "parity_open" "false";
  exact manifest "claim_ready" "false";
  List.iter
    (fun (path_key, hash_key, reason) ->
      verify_file root manifest path_key hash_key reason)
    [ ("garden_path", "garden_sha256", "exec-operation-cell-garden-hash-mismatch");
      ("source_path", "source_sha256", "exec-operation-cell-fixture-source-hash-mismatch");
      ("authority_manifest_path", "authority_manifest_sha256", "exec-operation-cell-action-9030-hash-mismatch");
      ("catalog_manifest_path", "catalog_manifest_sha256", "exec-operation-cell-action-9035-hash-mismatch");
      ("result_manifest_path", "result_manifest_sha256", "exec-operation-cell-action-9036-hash-mismatch");
      ("build_script_path", "build_script_sha256", "exec-operation-cell-fixture-build-hash-mismatch");
      ("selftest_path", "selftest_sha256", "exec-operation-cell-fixture-selftest-hash-mismatch");
      ("freeze_selftest_path", "freeze_selftest_sha256", "exec-operation-cell-fixture-freeze-gate-hash-mismatch");
      ("evidence_path", "evidence_sha256", "exec-operation-cell-fixture-evidence-hash-mismatch");
      ("toolchain_wrapper_path", "toolchain_wrapper_sha256", "exec-operation-cell-wrapper-hash-mismatch");
      ("toolchain_compiler_path", "toolchain_compiler_sha256", "exec-operation-cell-compiler-hash-mismatch") ];
  let command = required manifest "command" in
  let command_sha256 = required manifest "command_sha256" |> digest "command" in
  if sha256 command <> command_sha256 then
    failf "exec-operation-cell-command-hash-mismatch";
  let source = "tests/verify-ir/call_b.sio" in
  if command <> "loom-exec-cell-v2 sounio-check source=" ^ source then
    failf "exec-operation-cell-command-shape-mismatch";
  { manifest_sha256 = pinned_grant_manifest_sha256; command; command_sha256;
    event_sha256 = required manifest "event_sha256" |> digest "event";
    intent_sha256 = required manifest "intent_sha256" |> digest "intent";
    source }

let valid_unit value =
  value <> "" && String.length value <= 240
  && Filename.check_suffix value ".service"
  && String.for_all
       (function
         | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' | '-' | '.' | '@' | ':' -> true
         | _ -> false)
       value

let process_start_tick () =
  let value = read_bounded_file "/proc/self/stat" |> String.trim in
  let close =
    try String.rindex value ')' with Not_found ->
      failf "exec-operation-cell-proc-stat-malformed"
  in
  let suffix =
    String.sub value (close + 1) (String.length value - close - 1)
    |> String.trim
  in
  let fields =
    String.split_on_char ' ' suffix |> List.filter (fun field -> field <> "")
  in
  if List.length fields <= 19 then failf "exec-operation-cell-proc-stat-short";
  let value = List.nth fields 19 in
  if value = "" || not (String.for_all (function '0' .. '9' -> true | _ -> false) value)
  then failf "exec-operation-cell-start-tick-malformed";
  value

let measure_principal ~unit ~cgroup =
  if not (valid_unit unit) then failf "exec-operation-cell-unit-invalid";
  let pid = Unix.getpid () in
  let uid = Unix.getuid () in
  let gid = Unix.getgid () in
  if uid = 0 || gid = 0 then failf "exec-operation-cell-root-principal-refused";
  let start_tick = process_start_tick () in
  let cgroup_sha256 = sha256 cgroup in
  let executable = Unix.realpath "/proc/self/exe" in
  let executable_sha256 = streaming_sha256_file executable in
  let canonical =
    Printf.sprintf
      "LOOM_EXEC_CELL_PRINCIPAL/1|pid=%d|start_tick=%s|uid=%d|gid=%d|cgroup_sha256=%s|executable_sha256=%s|unit=%s"
      pid start_tick uid gid cgroup_sha256 executable_sha256 unit
  in
  { pid; start_tick; uid; gid; cgroup_sha256; executable_sha256; unit;
    canonical; sha256 = sha256 canonical }

let measure_descriptor ~unit =
  let descriptor = Unix.fstat Unix.stdin in
  if descriptor.st_kind <> S_FIFO then
    failf "exec-operation-cell-inherited-descriptor-not-pipe";
  let canonical =
    Printf.sprintf
      "LOOM_EXEC_CELL_DESCRIPTOR/1|unit=%s|fd=0|device=%d|inode=%d"
      unit descriptor.st_dev descriptor.st_ino
  in
  { device = descriptor.st_dev; inode = descriptor.st_ino; canonical;
    sha256 = sha256 canonical }

let require_private_output_directory ~mode path =
  if Filename.is_relative path then failf "exec-operation-cell-output-dir-not-absolute";
  let lexical = path in
  let before = Unix.lstat lexical in
  let resolved = Unix.realpath lexical in
  if before.st_kind = S_DIR then (
    if resolved <> lexical then failf "exec-operation-cell-output-dir-not-canonical")
  else if mode = "host" && before.st_kind = S_LNK then (
    let name = Filename.basename lexical in
    let expected_target = Filename.concat "private" name in
    let expected_resolved = Filename.concat "/run/private" name in
    if Filename.dirname lexical <> "/run" || name = "." || name = ".." ||
       Unix.readlink lexical <> expected_target || resolved <> expected_resolved then
      failf "exec-operation-cell-runtime-symlink-mismatch")
  else failf "exec-operation-cell-output-dir-not-directory";
  let stat = Unix.stat resolved in
  if stat.st_uid <> Unix.getuid () || stat.st_gid <> Unix.getgid () then
    failf "exec-operation-cell-output-dir-custody-mismatch";
  if stat.st_perm land 0o777 <> 0o700 then
    failf "exec-operation-cell-output-dir-not-private";
  lexical

let valid_invocation_id value =
  String.length value = 32
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let verify_host_context ~mode ~unit ~output_dir ~cgroup =
  if mode = "test" then (
    if Sys.getenv_opt "SOUNIO_LOOM_EXEC_OPERATION_CELL_TEST_MODE" <> Some "1" then
      failf "exec-operation-cell-test-mode-disabled";
    false)
  else if mode = "host" then (
    if Unix.getppid () <> 1 then failf "exec-operation-cell-host-parent-not-pid1";
    if not (String.contains cgroup '/') ||
       not (String.equal (Filename.basename (String.trim cgroup)) unit) then
      failf "exec-operation-cell-systemd-unit-mismatch";
    let invocation =
      match Sys.getenv_opt "INVOCATION_ID" with
      | Some value when valid_invocation_id value -> value
      | _ -> failf "exec-operation-cell-systemd-invocation-missing"
    in
    ignore invocation;
    if Sys.getenv_opt "RUNTIME_DIRECTORY" <> Some output_dir then
      failf "exec-operation-cell-runtime-directory-mismatch";
    if String.length output_dir < 6 || String.sub output_dir 0 5 <> "/run/" then
      failf "exec-operation-cell-runtime-directory-outside-run";
    true)
  else failf "exec-operation-cell-mode-must-be-test-or-host"

let parse_arm line =
  match String.split_on_char ' ' line with
  | [ "ARM"; generation; event; principal; descriptor; grant_receipt ] ->
      List.iter
        (fun (label, value) -> ignore (digest label value))
        [ ("generation", generation); ("event", event);
          ("principal", principal); ("descriptor", descriptor);
          ("grant-receipt", grant_receipt) ];
      (generation, event, principal, descriptor, grant_receipt)
  | _ -> failf "exec-operation-cell-arm-frame-malformed"

let parse_close line =
  match String.split_on_char ' ' line with
  | [ "CLOSE"; generation; event; record; close_receipt ] ->
      List.iter
        (fun (label, value) -> ignore (digest ("close-" ^ label) value))
        [ ("generation", generation); ("event", event); ("record", record);
          ("receipt", close_receipt) ];
      (generation, event, record, close_receipt)
  | _ -> failf "exec-operation-cell-close-frame-malformed"

let run ~root ~source ~output_dir ~unit ~mode =
  Unix.set_close_on_exec Unix.stdin;
  let root = Unix.realpath root in
  let policy = load_grant_policy ~root in
  let projection =
    Loom_exec_catalog.project ~root ~operation:"sounio-check" ~source:(Some source)
  in
  if source <> policy.source then failf "exec-operation-cell-source-not-frozen";
  if projection.semantic_event_sha256 <> policy.event_sha256 then
    failf "exec-operation-cell-catalog-event-mismatch";
  let output_dir = require_private_output_directory ~mode output_dir in
  let cgroup = read_bounded_file "/proc/self/cgroup" in
  let host_mode = verify_host_context ~mode ~unit ~output_dir ~cgroup in
  let principal = measure_principal ~unit ~cgroup in
  let descriptor = measure_descriptor ~unit in
  Printf.printf
    "LOOM_EXEC_OPERATION_CELL_READY_V1 semantic_authority=Sounio grant_action=9030 catalog_action=9035 result_action=9036 pid=%d start_tick=%s uid=%d gid=%d cgroup_sha256=%s executable_sha256=%s unit=%s principal_sha256=%s descriptor_device=%d descriptor_inode=%d descriptor_binding_sha256=%s host_mode=%s inherited_descriptor=true material_execution=false\n%!"
    principal.pid principal.start_tick principal.uid principal.gid
    principal.cgroup_sha256 principal.executable_sha256 principal.unit
    principal.sha256 descriptor.device descriptor.inode descriptor.sha256
    (if host_mode then "true" else "false");
  let arm =
    try input_line Stdlib.stdin with End_of_file ->
      failf "exec-operation-cell-arm-eof"
  in
  let generation, event, armed_principal, armed_descriptor, grant_receipt =
    parse_arm arm
  in
  if event <> policy.event_sha256 then failf "exec-operation-cell-arm-event-mismatch";
  if armed_principal <> principal.sha256 then
    failf "exec-operation-cell-arm-principal-mismatch";
  if armed_descriptor <> descriptor.sha256 then
    failf "exec-operation-cell-arm-descriptor-mismatch";
  let output =
    Filename.concat output_dir
      (Loom_exec_catalog.expected_output_basename
         (Option.get projection.source_sha256))
  in
  let material =
    Loom_exec_catalog.execute_sounio_check ~root ~source ~output
  in
  let binding : Loom_exec_result_record.binding =
    { event_sha256 = event; generation_sha256 = generation;
      principal_sha256 = principal.sha256;
      descriptor_binding_sha256 = descriptor.sha256;
      grant_receipt_sha256 = grant_receipt }
  in
  let issued = Loom_exec_result_record.issue ~root ~material ~binding in
  Printf.printf
    "LOOM_EXEC_OPERATION_CELL_RESULT_V1 semantic_authority=Sounio grant_action=9030 catalog_action=9035 result_action=9036 operation=sounio-check generation_sha256=%s event_sha256=%s principal_sha256=%s descriptor_binding_sha256=%s grant_receipt_sha256=%s source_sha256=%s artifact_sha256=%s artifact_bytes=%d record_sha256=%s handle=%s grant_arm_received=true principal_self_measured=true inherited_descriptor=true host_mode=%s material_execution=true artifact_executed=false handle_is_bearer=false handle_is_execution_authority=false\n%s%!"
    generation event principal.sha256 descriptor.sha256 grant_receipt
    (Option.get material.plan.projection.source_sha256)
    material.artifact_sha256 material.artifact_bytes issued.record_sha256
    issued.handle (if host_mode then "true" else "false") issued.record;
  let close =
    try input_line Stdlib.stdin with End_of_file ->
      failf "exec-operation-cell-close-eof"
  in
  let close_generation, close_event, close_record, close_receipt =
    parse_close close
  in
  if close_generation <> generation then
    failf "exec-operation-cell-close-generation-mismatch";
  if close_event <> event then failf "exec-operation-cell-close-event-mismatch";
  if close_record <> issued.record_sha256 then
    failf "exec-operation-cell-close-record-mismatch";
  Printf.printf
    "LOOM_EXEC_OPERATION_CELL_CLOSED_V1 semantic_authority=Sounio action=9030 generation_sha256=%s event_sha256=%s record_sha256=%s close_receipt_sha256=%s material_execution=true authority_extinction=armed\n%!"
    generation event issued.record_sha256 close_receipt;
  0
