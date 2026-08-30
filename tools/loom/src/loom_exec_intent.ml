open Unix

exception Error of string

let pinned_manifest_sha256 =
  "8a95e587ccc81c16da17d56b9649d04bc9c3e764d66fc938c195d95568e7608e"

let max_file_bytes = 8 * 1024 * 1024
let authority_timeout_seconds = 5.0

type policy = {
  manifest_sha256 : string;
  source_sha256 : string;
  executable_sha256 : string;
  runtime : string;
  wire_schema : int;
  project_word0 : int;
  command_mismatch_word0 : int;
  common_word1 : int;
  project_decision : string;
  command_mismatch_decision : string;
  fields_schema : string;
  envelope_schema : string;
  semantic_event_sha256 : string;
  command_sha256 : string;
}

type projection = {
  raw_event_sha256 : string;
  event_sha256 : string;
  command_sha256 : string;
  manifest_sha256 : string;
  source_sha256 : string;
  executable_sha256 : string;
  authority_output_sha256 : string;
}

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let valid_sha256 value =
  String.length value = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let require_regular_file path =
  let stat =
    try Unix.lstat path
    with Unix_error (ENOENT, _, _) -> failf "exec-intent-file-missing:%s" path
  in
  if stat.st_kind <> S_REG then failf "exec-intent-file-not-regular:%s" path;
  stat

let read_file path =
  let stat = require_regular_file path in
  if stat.st_size > max_file_bytes then failf "exec-intent-file-too-large:%s" path;
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () -> really_input_string channel (in_channel_length channel))

let sha256_file path = sha256 (read_file path)

let parse_manifest path =
  let text = read_file path in
  if text = "" || text.[String.length text - 1] <> '\n' then
    failf "exec-intent-manifest-missing-final-newline";
  let lines = String.split_on_char '\n' text in
  let lines =
    match List.rev lines with
    | "" :: rest -> List.rev rest
    | _ -> failf "exec-intent-manifest-missing-final-newline"
  in
  let table = Hashtbl.create (List.length lines) in
  List.iter
    (fun line ->
      match String.index_opt line '=' with
      | None -> failf "exec-intent-manifest-malformed-field"
      | Some index ->
          let key = String.sub line 0 index in
          let value =
            String.sub line (index + 1) (String.length line - index - 1)
          in
          if key = "" || value = "" then
            failf "exec-intent-manifest-empty-field";
          if Hashtbl.mem table key then
            failf "exec-intent-manifest-duplicate-field:%s" key;
          Hashtbl.add table key value)
    lines;
  table

let required manifest key =
  match Hashtbl.find_opt manifest key with
  | Some value when value <> "" -> value
  | _ -> failf "exec-intent-manifest-field-missing:%s" key

let exact manifest key expected =
  let actual = required manifest key in
  if actual <> expected then
    failf "exec-intent-manifest-field-invalid:%s:%s" key actual

let decimal label value =
  if value = ""
     || not (String.for_all (function '0' .. '9' -> true | _ -> false) value)
  then failf "exec-intent-%s-not-decimal" label;
  try int_of_string value with _ -> failf "exec-intent-%s-out-of-range" label

let digest label value =
  let value = String.lowercase_ascii value in
  if not (valid_sha256 value) then failf "exec-intent-%s-invalid-sha256" label;
  value

let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"

let configured_path ~name ~default =
  match Sys.getenv_opt name with
  | Some value when value <> "" && test_mode () -> value
  | Some value when value <> "" -> failf "%s-override-requires-test-mode" name
  | _ -> default

let manifest_path root =
  configured_path ~name:"SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_MANIFEST"
    ~default:(Filename.concat root "tools/loom/exec_intent_envelope.freeze.v1")

let verify_file root manifest path_key hash_key reason =
  let path = Filename.concat root (required manifest path_key) in
  ignore (require_regular_file path);
  if sha256_file path <> required manifest hash_key then failf "%s" reason

let choose_runtime root manifest =
  let repository_runtime =
    Filename.concat root
      "tools/loom/_build/default/src/sounio-loom-exec-intent-envelope"
  in
  let installed_runtime =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-exec-intent-envelope"
  in
  let default =
    if Sys.file_exists repository_runtime then repository_runtime
    else installed_runtime
  in
  let selected =
    configured_path ~name:"SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_RUNTIME" ~default
  in
  let stat = require_regular_file selected in
  if stat.st_perm land 0o111 = 0 then
    failf "exec-intent-runtime-not-executable";
  if sha256_file selected <> required manifest "executable_sha256" then
    failf "exec-intent-runtime-hash-mismatch";
  Unix.realpath selected

let load ~root =
  let root = Unix.realpath root in
  let path = manifest_path root in
  ignore (require_regular_file path);
  if sha256_file path <> pinned_manifest_sha256 then
    failf "exec-intent-manifest-hash-mismatch";
  let manifest = parse_manifest path in
  exact manifest "schema" "loom-exec-intent-envelope-freeze-v1";
  exact manifest "stage" "SEMANTICS_FROZEN";
  exact manifest "producing_language" "Sounio";
  exact manifest "language_role" "SEMANTIC_AUTHORITY";
  exact manifest "action" "9034";
  exact manifest "concept_id" "SOUNIO-LOOM-EXEC-INTENT-ENVELOPE";
  exact manifest "causal_sabotage" "PASS";
  exact manifest "load_bearing_rule" "command_sha256_equal";
  exact manifest "raw_event_is_semantic_identity" "false";
  exact manifest "raw_event_separate" "true";
  exact manifest "descriptor_transport" "unix-stream-inherited";
  exact manifest "descriptor_is_bearer" "false";
  exact manifest "pathname_is_authority" "false";
  exact manifest "expected_results_encoded_in_material_layer" "false";
  exact manifest "ocaml_projection_attached" "false";
  exact manifest "provider_lifecycle_attached" "false";
  exact manifest "arbitrary_command_projection" "false";
  exact manifest "exec_attached" "false";
  exact manifest "production_activation" "false";
  exact manifest "parity_open" "false";
  exact manifest "claim_ready" "false";
  List.iter
    (fun (path_key, hash_key, reason) ->
      verify_file root manifest path_key hash_key reason)
    [ ("garden_path", "garden_sha256", "exec-intent-garden-hash-mismatch");
      ("contract_path", "contract_sha256", "exec-intent-contract-hash-mismatch");
      ("source_path", "source_sha256", "exec-intent-source-hash-mismatch");
      ("entrypoint_path", "entrypoint_sha256", "exec-intent-entrypoint-hash-mismatch");
      ("build_script_path", "build_script_sha256", "exec-intent-build-script-hash-mismatch");
      ("selftest_path", "selftest_sha256", "exec-intent-selftest-hash-mismatch");
      ("evidence_path", "evidence_sha256", "exec-intent-evidence-hash-mismatch");
      ("parent_9031_manifest_path", "parent_9031_manifest_sha256", "exec-intent-parent-9031-hash-mismatch");
      ("parent_9033_manifest_path", "parent_9033_manifest_sha256", "exec-intent-parent-9033-hash-mismatch");
      ("toolchain_wrapper_path", "toolchain_wrapper_sha256", "exec-intent-toolchain-wrapper-hash-mismatch");
      ("toolchain_compiler_path", "toolchain_compiler_sha256", "exec-intent-toolchain-compiler-hash-mismatch") ];
  let semantic_event_sha256 =
    required manifest "semantic_event_sha256" |> digest "semantic-event"
  in
  let command_sha256 = required manifest "command_sha256" |> digest "command" in
  let runtime = choose_runtime root manifest in
  { manifest_sha256 = pinned_manifest_sha256;
    source_sha256 = required manifest "source_sha256";
    executable_sha256 = required manifest "executable_sha256";
    runtime;
    wire_schema = required manifest "wire_schema" |> decimal "wire-schema";
    project_word0 = required manifest "project_word0" |> decimal "project-word0";
    command_mismatch_word0 =
      required manifest "command_mismatch_word0" |> decimal "command-mismatch-word0";
    common_word1 = required manifest "common_word1" |> decimal "common-word1";
    project_decision = required manifest "project_decision";
    command_mismatch_decision = required manifest "command_mismatch_decision";
    fields_schema = required manifest "fields_schema";
    envelope_schema = required manifest "envelope_schema";
    semantic_event_sha256;
    command_sha256 }

let write_all descriptor value =
  let bytes = Bytes.of_string value in
  let rec loop offset =
    if offset < Bytes.length bytes then
      match Unix.write descriptor bytes offset (Bytes.length bytes - offset) with
      | 0 -> failf "exec-intent-short-write"
      | count -> loop (offset + count)
      | exception Unix_error (EINTR, _, _) -> loop offset
  in
  loop 0

let process_exchange executable input =
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
      let deadline = Unix.gettimeofday () +. authority_timeout_seconds in
      let eof = ref false in
      while not (!eof && Option.is_some !status) do
        let remaining = deadline -. Unix.gettimeofday () in
        if remaining <= 0. then failf "exec-intent-authority-timeout";
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
                  failf "exec-intent-authority-output-too-large";
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

let authority_frame policy word0 =
  Printf.sprintf "%d %d %d\n" policy.wire_schema word0 policy.common_word1

let expected_output policy =
  String.concat "\n"
    [ policy.project_decision; policy.fields_schema; policy.envelope_schema;
      policy.semantic_event_sha256; policy.command_sha256 ]

let project ~root ~raw_event_sha256 ~command_sha256 =
  let raw_event_sha256 = digest "raw-event" raw_event_sha256 in
  let command_sha256 = digest "observed-command" command_sha256 in
  let policy = load ~root in
  if command_sha256 <> policy.command_sha256 then (
    let code, output =
      process_exchange policy.runtime
        (authority_frame policy policy.command_mismatch_word0)
    in
    if code <> 42 || output <> policy.command_mismatch_decision then
      failf "exec-intent-command-mismatch-control-diverged:%d:%s" code output;
    failf "exec-intent-command-mismatch-denied:%s" output);
  let code, output =
    process_exchange policy.runtime (authority_frame policy policy.project_word0)
  in
  if code <> 0 then failf "exec-intent-authority-refused:%d:%s" code output;
  if output <> expected_output policy then
    failf "exec-intent-authority-output-mismatch";
  { raw_event_sha256;
    event_sha256 = policy.semantic_event_sha256;
    command_sha256 = policy.command_sha256;
    manifest_sha256 = policy.manifest_sha256;
    source_sha256 = policy.source_sha256;
    executable_sha256 = policy.executable_sha256;
    authority_output_sha256 = sha256 output }

let command_mismatch_control ~root =
  let policy = load ~root in
  let code, output =
    process_exchange policy.runtime
      (authority_frame policy policy.command_mismatch_word0)
  in
  if code <> 42 || output <> policy.command_mismatch_decision then
    failf "exec-intent-command-mismatch-control-diverged:%d:%s" code output;
  (policy, output)
