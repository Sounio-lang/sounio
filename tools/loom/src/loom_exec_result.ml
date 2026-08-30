open Unix

exception Error of string

let pinned_manifest_sha256 =
  "7c1955299f2b308c331760764c289b6617910549b5dfc6bc4cf1969457306a55"

let max_file_bytes = 8 * 1024 * 1024
let max_receipt_bytes = 1024 * 1024
let authority_timeout_seconds = 5.0

type policy = {
  root : string;
  manifest_sha256 : string;
  source_sha256 : string;
  parent_9030_sha256 : string;
  parent_9031_sha256 : string;
  fixture_manifest_sha256 : string;
  command_sha256 : string;
  executable_sha256 : string;
  runtime : string;
  wire_schema : int;
  publish_word0 : int;
  resolve_word0 : int;
  command_mismatch_word0 : int;
  common_word1 : int;
  common_word2 : int;
  publish_decision : string;
  resolve_decision : string;
  command_mismatch_decision : string;
  handle_fields_schema : string;
  event_sha256 : string;
  grant_generation : int;
  result_receipt_sha256 : string;
  canonical_handle : string;
}

type access_purpose = Result_read | Authority_promotion

type stored_result = {
  handle : string;
  path : string;
  record_sha256 : string;
  receipt_sha256 : string;
  receipt : string;
  manifest_sha256 : string;
  authority_output_sha256 : string;
}

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let hex_of_string value =
  Cryptokit.transform_string (Cryptokit.Hexa.encode ()) value

let valid_sha256 value =
  String.length value = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let require_regular_file ?(mode = None) path =
  let stat =
    try Unix.lstat path with Unix_error (ENOENT, _, _) -> failf "file-missing:%s" path
  in
  if stat.st_kind <> S_REG then failf "file-not-regular:%s" path;
  Option.iter
    (fun expected ->
      if stat.st_perm land 0o777 <> expected then
        failf "file-mode-invalid:%s:%03o" path (stat.st_perm land 0o777))
    mode;
  stat

let read_file ?(limit = max_file_bytes) path =
  let stat = require_regular_file path in
  if stat.st_size > limit then failf "file-too-large:%s" path;
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () -> really_input_string channel (in_channel_length channel))

let sha256_file path = sha256 (read_file path)

let parse_fields label text =
  if text = "" || text.[String.length text - 1] <> '\n' then
    failf "%s-missing-final-newline" label;
  let lines = String.split_on_char '\n' text in
  let lines =
    match List.rev lines with
    | "" :: rest -> List.rev rest
    | _ -> failf "%s-missing-final-newline" label
  in
  let table = Hashtbl.create (List.length lines) in
  List.iter
    (fun line ->
      match String.index_opt line '=' with
      | None -> failf "%s-malformed-field" label
      | Some index ->
          let key = String.sub line 0 index in
          let value =
            String.sub line (index + 1) (String.length line - index - 1)
          in
          if key = "" || value = "" then failf "%s-empty-field" label;
          if Hashtbl.mem table key then failf "%s-duplicate-field:%s" label key;
          Hashtbl.add table key value)
    lines;
  table

let parse_manifest path = parse_fields "exec-result-manifest" (read_file path)

let required label table key =
  match Hashtbl.find_opt table key with
  | Some value when value <> "" -> value
  | _ -> failf "missing-%s-field:%s" label key

let exact table key expected =
  let actual = required "exec-result-manifest" table key in
  if actual <> expected then
    failf "exec-result-manifest-field-invalid:%s:%s" key actual

let decimal label value =
  if value = "" || not (String.for_all (function '0' .. '9' -> true | _ -> false) value)
  then failf "%s-not-decimal" label;
  try int_of_string value with _ -> failf "%s-out-of-range" label

let digest label value =
  let value = String.lowercase_ascii value in
  if not (valid_sha256 value) then failf "%s-invalid-sha256" label;
  value

let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"

let configured_path ~name ~default =
  match Sys.getenv_opt name with
  | Some value when value <> "" && test_mode () -> value
  | Some value when value <> "" -> failf "%s-override-requires-test-mode" name
  | _ -> default

let manifest_path root =
  configured_path ~name:"SOUNIO_LOOM_EXEC_RESULT_HANDLE_MANIFEST"
    ~default:(Filename.concat root "tools/loom/exec_result_handle.freeze.v1")

let verify_file root manifest path_key hash_key reason =
  let path = Filename.concat root (required "exec-result-manifest" manifest path_key) in
  ignore (require_regular_file path);
  if sha256_file path <> required "exec-result-manifest" manifest hash_key then
    failf "%s" reason

let choose_runtime root manifest =
  let repository_runtime =
    Filename.concat root
      "tools/loom/_build/default/src/sounio-loom-exec-result-handle"
  in
  let installed_runtime =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-exec-result-handle"
  in
  let default =
    if Sys.file_exists repository_runtime then repository_runtime
    else installed_runtime
  in
  let selected =
    configured_path ~name:"SOUNIO_LOOM_EXEC_RESULT_HANDLE_RUNTIME" ~default
  in
  let stat = require_regular_file selected in
  if stat.st_perm land 0o111 = 0 then failf "exec-result-runtime-not-executable";
  if sha256_file selected <> required "exec-result-manifest" manifest "executable_sha256"
  then failf "exec-result-runtime-hash-mismatch";
  Unix.realpath selected

let load ~root =
  let root = Unix.realpath root in
  let path = manifest_path root in
  ignore (require_regular_file path);
  if sha256_file path <> pinned_manifest_sha256 then
    failf "exec-result-manifest-hash-mismatch";
  let manifest = parse_manifest path in
  exact manifest "schema" "loom-exec-result-handle-freeze-v1";
  exact manifest "stage" "SEMANTICS_FROZEN";
  exact manifest "producing_language" "Sounio";
  exact manifest "language_role" "SEMANTIC_AUTHORITY";
  exact manifest "action" "9033";
  exact manifest "concept_id" "SOUNIO-LOOM-EXEC-RESULT-HANDLE";
  exact manifest "causal_sabotage" "PASS";
  exact manifest "load_bearing_rule" "command_sha256_equal";
  exact manifest "prewrite_validation" "true";
  exact manifest "handle_is_bearer" "false";
  exact manifest "handle_is_execution_authority" "false";
  exact manifest "handle_is_semantic_proof" "false";
  exact manifest "expected_results_encoded_in_material_layer" "false";
  exact manifest "material_execution" "false";
  exact manifest "exec_cell_attached" "false";
  exact manifest "result_store_attached" "false";
  exact manifest "provider_hook_switched" "false";
  exact manifest "production_activation" "false";
  exact manifest "parity_open" "false";
  exact manifest "claim_ready" "false";
  List.iter
    (fun (path_key, hash_key, reason) ->
      verify_file root manifest path_key hash_key reason)
    [ ("garden_path", "garden_sha256", "exec-result-garden-hash-mismatch");
      ("contract_path", "contract_sha256", "exec-result-contract-hash-mismatch");
      ("source_path", "source_sha256", "exec-result-source-hash-mismatch");
      ("entrypoint_path", "entrypoint_sha256", "exec-result-entrypoint-hash-mismatch");
      ("build_script_path", "build_script_sha256", "exec-result-build-script-hash-mismatch");
      ("selftest_path", "selftest_sha256", "exec-result-selftest-hash-mismatch");
      ("evidence_path", "evidence_sha256", "exec-result-evidence-hash-mismatch");
      ("parent_9030_manifest_path", "parent_9030_manifest_sha256", "exec-result-parent-9030-hash-mismatch");
      ("parent_9031_manifest_path", "parent_9031_manifest_sha256", "exec-result-parent-9031-hash-mismatch");
      ("fixture_manifest_path", "fixture_manifest_sha256", "exec-result-fixture-hash-mismatch");
      ("toolchain_wrapper_path", "toolchain_wrapper_sha256", "exec-result-toolchain-wrapper-hash-mismatch");
      ("toolchain_compiler_path", "toolchain_compiler_sha256", "exec-result-toolchain-compiler-hash-mismatch") ];
  let event_sha256 =
    required "exec-result-manifest" manifest "event_sha256" |> digest "event"
  in
  let result_receipt_sha256 =
    required "exec-result-manifest" manifest "result_receipt_sha256"
    |> digest "result-receipt"
  in
  let grant_generation =
    required "exec-result-manifest" manifest "grant_generation"
    |> decimal "grant-generation"
  in
  if grant_generation <= 0 then failf "grant-generation-not-positive";
  let canonical_handle =
    required "exec-result-manifest" manifest "canonical_handle"
  in
  let recomposed =
    Printf.sprintf "loom-result-v1:%s:%d:%s" event_sha256 grant_generation
      result_receipt_sha256
  in
  if canonical_handle <> recomposed then failf "exec-result-handle-binding-invalid";
  let fixture_path =
    Filename.concat root
      (required "exec-result-manifest" manifest "fixture_manifest_path")
  in
  let fixture = parse_fields "exec-result-fixture" (read_file fixture_path) in
  let command_sha256 =
    required "exec-result-fixture" fixture "command_sha256"
    |> digest "command"
  in
  let runtime = choose_runtime root manifest in
  { root;
    manifest_sha256 = pinned_manifest_sha256;
    source_sha256 = required "exec-result-manifest" manifest "source_sha256";
    parent_9030_sha256 = required "exec-result-manifest" manifest "parent_9030_manifest_sha256";
    parent_9031_sha256 = required "exec-result-manifest" manifest "parent_9031_manifest_sha256";
    fixture_manifest_sha256 = required "exec-result-manifest" manifest "fixture_manifest_sha256";
    command_sha256;
    executable_sha256 = required "exec-result-manifest" manifest "executable_sha256";
    runtime;
    wire_schema = required "exec-result-manifest" manifest "wire_schema" |> decimal "wire-schema";
    publish_word0 = required "exec-result-manifest" manifest "publish_word0" |> decimal "publish-word0";
    resolve_word0 = required "exec-result-manifest" manifest "resolve_word0" |> decimal "resolve-word0";
    command_mismatch_word0 = required "exec-result-manifest" manifest "command_mismatch_word0" |> decimal "command-mismatch-word0";
    common_word1 = required "exec-result-manifest" manifest "common_word1" |> decimal "common-word1";
    common_word2 = required "exec-result-manifest" manifest "common_word2" |> decimal "common-word2";
    publish_decision = required "exec-result-manifest" manifest "publish_decision";
    resolve_decision = required "exec-result-manifest" manifest "resolve_decision";
    command_mismatch_decision = required "exec-result-manifest" manifest "command_mismatch_decision";
    handle_fields_schema = required "exec-result-manifest" manifest "handle_fields_schema";
    event_sha256;
    grant_generation;
    result_receipt_sha256;
    canonical_handle }

let write_all descriptor value =
  let bytes = Bytes.of_string value in
  let rec loop offset =
    if offset < Bytes.length bytes then
      match Unix.write descriptor bytes offset (Bytes.length bytes - offset) with
      | 0 -> failf "short-write"
      | count -> loop (offset + count)
      | exception Unix_error (EINTR, _, _) -> loop offset
  in
  loop 0

let process_exchange ~timeout_seconds executable input =
  let stdin_reader, stdin_writer = Unix.pipe () in
  let output_reader, output_writer = Unix.pipe () in
  Unix.set_close_on_exec stdin_writer;
  Unix.set_close_on_exec output_reader;
  let pid =
    try
      Unix.create_process executable [| executable |] stdin_reader output_writer
        output_writer
    with error ->
      Unix.close stdin_reader;
      Unix.close stdin_writer;
      Unix.close output_reader;
      Unix.close output_writer;
      raise error
  in
  Unix.close stdin_reader;
  Unix.close output_writer;
  let status = ref None in
  let close descriptor = try Unix.close descriptor with _ -> () in
  let reap () =
    match !status with
    | Some _ -> ()
    | None ->
        (try Unix.kill pid Sys.sigkill with _ -> ());
        (try
           let _, observed = Unix.waitpid [] pid in
           status := Some observed
         with _ -> ())
  in
  Fun.protect
    ~finally:(fun () -> close stdin_writer; close output_reader; reap ())
    (fun () ->
      write_all stdin_writer input;
      close stdin_writer;
      let output = Buffer.create 512 in
      let bytes = Bytes.create 4096 in
      let deadline = Unix.gettimeofday () +. timeout_seconds in
      let eof = ref false in
      while not (!eof && Option.is_some !status) do
        let remaining = deadline -. Unix.gettimeofday () in
        if remaining <= 0. then failf "exec-result-authority-timeout";
        (match Unix.waitpid [ WNOHANG ] pid with
        | 0, _ -> ()
        | _, observed -> status := Some observed);
        if not !eof then (
          let ready, _, _ =
            Unix.select [ output_reader ] [] [] (min remaining 0.05)
          in
          if ready <> [] then
            match Unix.read output_reader bytes 0 (Bytes.length bytes) with
            | 0 -> eof := true
            | count ->
                if Buffer.length output + count > 16384 then
                  failf "exec-result-authority-output-too-large";
                Buffer.add_subbytes output bytes 0 count
            | exception Unix_error (EINTR, _, _) -> ())
        else if Option.is_none !status then
          ignore (Unix.select [] [] [] (min remaining 0.01))
      done;
      let code =
        match Option.get !status with
        | WEXITED value -> value
        | WSIGNALED signal | WSTOPPED signal -> 128 + signal
      in
      let output = Buffer.contents output in
      let output =
        if output <> "" && output.[String.length output - 1] = '\n' then
          String.sub output 0 (String.length output - 1)
        else output
      in
      (code, output))

let expected_output (policy : policy) decision =
  String.concat "\n"
    [ decision; policy.handle_fields_schema; policy.event_sha256;
      string_of_int policy.grant_generation; policy.result_receipt_sha256 ]

let authority_frame (policy : policy) word0 =
  Printf.sprintf "%d %d %d %d\n" policy.wire_schema word0 policy.common_word1
    policy.common_word2

let invoke_allow (policy : policy) ~word0 ~decision =
  let code, output =
    process_exchange ~timeout_seconds:authority_timeout_seconds policy.runtime
      (authority_frame policy word0)
  in
  if code <> 0 then failf "exec-result-authority-refused:%d:%s" code output;
  if output <> expected_output policy decision then
    failf "exec-result-authority-output-mismatch";
  output

let command_mismatch_control ~root =
  let policy = load ~root in
  let code, output =
    process_exchange ~timeout_seconds:authority_timeout_seconds policy.runtime
      (authority_frame policy policy.command_mismatch_word0)
  in
  if code <> 42 || output <> policy.command_mismatch_decision then
    failf "exec-result-command-mismatch-control-diverged:%d:%s" code output;
  (policy, output)

let secure_directory path =
  let stat = Unix.lstat path in
  if stat.st_kind <> S_DIR then failf "result-store-not-directory:%s" path;
  if stat.st_uid <> Unix.geteuid () then failf "result-store-owner-mismatch:%s" path;
  if stat.st_perm land 0o777 <> 0o700 then
    failf "result-store-mode-invalid:%s:%03o" path (stat.st_perm land 0o777)

let ensure_directory path =
  try secure_directory path with
  | Unix_error (ENOENT, _, _) ->
      Unix.mkdir path 0o700;
      secure_directory path

let prepare_store (policy : policy) store_root =
  if Filename.is_relative store_root then failf "result-store-root-not-absolute";
  ensure_directory store_root;
  let store_root = Unix.realpath store_root in
  let handles = Filename.concat store_root "handles" in
  ensure_directory handles;
  let event = Filename.concat handles policy.event_sha256 in
  ensure_directory event;
  let generation = Filename.concat event (string_of_int policy.grant_generation) in
  ensure_directory generation;
  (store_root, generation)

let record_path (policy : policy) store_root =
  let _, directory = prepare_store policy store_root in
  Filename.concat directory (policy.result_receipt_sha256 ^ ".result")

let expected_publish_output (policy : policy) =
  expected_output policy policy.publish_decision

let render_record (policy : policy) receipt =
  let authority_output_sha256 = sha256 (expected_publish_output policy) in
  String.concat "\n"
    [ "schema=loom-exec-result-record-v1";
      "semantic_authority=Sounio";
      "semantic_action=9033";
      "semantic_manifest_sha256=" ^ policy.manifest_sha256;
      "semantic_source_sha256=" ^ policy.source_sha256;
      "parent_9030_manifest_sha256=" ^ policy.parent_9030_sha256;
      "parent_9031_manifest_sha256=" ^ policy.parent_9031_sha256;
      "fixture_manifest_sha256=" ^ policy.fixture_manifest_sha256;
      "command_sha256=" ^ policy.command_sha256;
      "event_sha256=" ^ policy.event_sha256;
      "grant_generation=" ^ string_of_int policy.grant_generation;
      "result_receipt_sha256=" ^ policy.result_receipt_sha256;
      "canonical_handle=" ^ policy.canonical_handle;
      "authority_publish_output_sha256=" ^ authority_output_sha256;
      "material_language=OCaml";
      "material_role=OPERATIONAL_RESULT_STORE";
      "receipt_encoding=hex";
      "receipt_hex=" ^ hex_of_string receipt;
      "handle_is_bearer=false";
      "handle_is_execution_authority=false";
      "handle_is_semantic_proof=false";
      "reader_read_only=true";
      "handle_lookup_only=true";
      "handle_can_issue=false";
      "handle_can_consume=false";
      "handle_can_execute=false";
      "handle_can_replay=false";
      "exec_attached=false";
      "provider_hook_switched=false";
      "production_activation=false";
      "" ]

let nibble = function
  | '0' .. '9' as value -> Char.code value - Char.code '0'
  | 'a' .. 'f' as value -> 10 + Char.code value - Char.code 'a'
  | _ -> failf "result-record-receipt-hex-invalid"

let string_of_hex value =
  if String.length value mod 2 <> 0 then failf "result-record-receipt-hex-invalid";
  let output = Bytes.create (String.length value / 2) in
  for index = 0 to Bytes.length output - 1 do
    let high = nibble value.[index * 2] in
    let low = nibble value.[(index * 2) + 1] in
    Bytes.set output index (Char.chr ((high lsl 4) lor low))
  done;
  Bytes.to_string output

let canonical_record_rule record_text canonical = record_text = canonical

let validate_record (policy : policy) path =
  let stat = require_regular_file ~mode:(Some 0o400) path in
  if stat.st_uid <> Unix.geteuid () then failf "result-record-owner-mismatch";
  if stat.st_nlink <> 1 then failf "result-record-link-count-invalid";
  let text = read_file ~limit:(max_receipt_bytes * 2 + 8192) path in
  let fields = parse_fields "exec-result-record" text in
  let field key = required "exec-result-record" fields key in
  let exact_record key expected =
    if field key <> expected then failf "result-record-field-invalid:%s" key
  in
  exact_record "schema" "loom-exec-result-record-v1";
  exact_record "semantic_authority" "Sounio";
  exact_record "semantic_action" "9033";
  exact_record "semantic_manifest_sha256" policy.manifest_sha256;
  exact_record "semantic_source_sha256" policy.source_sha256;
  exact_record "parent_9030_manifest_sha256" policy.parent_9030_sha256;
  exact_record "parent_9031_manifest_sha256" policy.parent_9031_sha256;
  exact_record "fixture_manifest_sha256" policy.fixture_manifest_sha256;
  exact_record "command_sha256" policy.command_sha256;
  exact_record "event_sha256" policy.event_sha256;
  exact_record "grant_generation" (string_of_int policy.grant_generation);
  exact_record "result_receipt_sha256" policy.result_receipt_sha256;
  exact_record "canonical_handle" policy.canonical_handle;
  exact_record "authority_publish_output_sha256"
    (sha256 (expected_publish_output policy));
  exact_record "material_language" "OCaml";
  exact_record "material_role" "OPERATIONAL_RESULT_STORE";
  exact_record "receipt_encoding" "hex";
  exact_record "handle_is_bearer" "false";
  exact_record "handle_is_execution_authority" "false";
  exact_record "handle_is_semantic_proof" "false";
  exact_record "reader_read_only" "true";
  exact_record "handle_lookup_only" "true";
  exact_record "handle_can_issue" "false";
  exact_record "handle_can_consume" "false";
  exact_record "handle_can_execute" "false";
  exact_record "handle_can_replay" "false";
  exact_record "exec_attached" "false";
  exact_record "provider_hook_switched" "false";
  exact_record "production_activation" "false";
  let receipt = field "receipt_hex" |> string_of_hex in
  if String.length receipt > max_receipt_bytes then failf "result-receipt-too-large";
  if sha256 receipt <> policy.result_receipt_sha256 then
    failf "result-record-receipt-hash-mismatch";
  let canonical = render_record policy receipt in
  if not (canonical_record_rule text canonical) then
    failf "result-record-canonical-form-mismatch";
  { handle = policy.canonical_handle;
    path;
    record_sha256 = sha256 text;
    receipt_sha256 = policy.result_receipt_sha256;
    receipt;
    manifest_sha256 = policy.manifest_sha256;
    authority_output_sha256 = sha256 (expected_publish_output policy) }

let fsync_directory path =
  let descriptor = Unix.openfile path [ O_RDONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () -> Unix.fsync descriptor)

let publish ~root ~store_root ~receipt_path =
  let policy = load ~root in
  let receipt_stat = require_regular_file receipt_path in
  if receipt_stat.st_size > max_receipt_bytes then failf "result-receipt-too-large";
  let receipt = read_file ~limit:max_receipt_bytes receipt_path in
  if sha256 receipt <> policy.result_receipt_sha256 then
    failf "result-receipt-hash-mismatch";
  let path = record_path policy store_root in
  (try
     ignore (Unix.lstat path);
     failf "result-record-already-exists"
   with Unix_error (ENOENT, _, _) -> ());
  let authority_output =
    invoke_allow policy ~word0:policy.publish_word0
      ~decision:policy.publish_decision
  in
  let record = render_record policy receipt in
  if sha256 authority_output <> sha256 (expected_publish_output policy) then
    failf "exec-result-publish-decision-digest-mismatch";
  let directory = Filename.dirname path in
  let temporary =
    Filename.concat directory
      (Printf.sprintf ".result-%d-%08x.tmp" (Unix.getpid ()) (Random.bits ()))
  in
  let descriptor =
    Unix.openfile temporary [ O_WRONLY; O_CREAT; O_EXCL ] 0o600
  in
  (try
     Unix.set_close_on_exec descriptor;
     write_all descriptor record;
     Unix.fsync descriptor;
     Unix.fchmod descriptor 0o400;
     Unix.fsync descriptor;
     Unix.close descriptor;
     Unix.link temporary path;
     Unix.unlink temporary;
     fsync_directory directory
   with error ->
     (try Unix.close descriptor with _ -> ());
     (try Unix.unlink temporary with _ -> ());
     raise error);
  validate_record policy path

let parse_handle handle =
  match String.split_on_char ':' handle with
  | [ "loom-result-v1"; event; generation; receipt ] ->
      let event = digest "handle-event" event in
      let generation = decimal "handle-generation" generation in
      let receipt = digest "handle-receipt" receipt in
      if generation <= 0 then failf "handle-generation-not-positive";
      (event, generation, receipt)
  | _ -> failf "result-handle-malformed"

let resolve ~root ~store_root ~handle ~purpose =
  if purpose = Authority_promotion then
    failf "result-handle-authority-promotion-refused";
  let policy = load ~root in
  let event, generation, receipt = parse_handle handle in
  if event <> policy.event_sha256 || generation <> policy.grant_generation
     || receipt <> policy.result_receipt_sha256
     || handle <> policy.canonical_handle
  then failf "result-handle-not-frozen-fixture";
  let path = record_path policy store_root in
  let result = validate_record policy path in
  ignore
    (invoke_allow policy ~word0:policy.resolve_word0
       ~decision:policy.resolve_decision);
  result

let manifest_sha256 result = result.manifest_sha256
let authority_output_sha256 result = result.authority_output_sha256
