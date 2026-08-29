open Unix

exception Error of string

let pinned_manifest_sha256 =
  "0024178b8928f0c82d794d390244e83e5ce431054587fc7dd609c0f25c2e5b4f"

let pinned_sandbox_sha256 =
  "52231e1caf55bcbc667b269f49c63599a6f7db4767ae6a039580d0ff853db712"

let pinned_activation_manifest_sha256 =
  "f2da55138bcfe5a8a2c65ebd79c1e534f152b33af5c6cc3d1f2b4eb3b4af6e7e"

let pinned_activation_runtime_sha256 =
  "d7521e8fb60501dc8192ebbeade4a09649164c5b509a2dda8af5c465bf3de793"

let pinned_resident_v5_manifest_sha256 =
  "b3cf8c1e0524be35fc67b2b5a779bad9a9291195d65dc82dbc87595396fb5353"

let pinned_activation_projection_sha256 =
  "8a72e9bcd510a751b856cf29960b7389486defcc4d13d7614546023d3d355014"

let authority_timeout_seconds = 5.0
let max_file_bytes = 8 * 1024 * 1024

type native_event = int * int * int * string * int

type native_result =
  int * int * int * int64 * int * int * int * int * int * int

external native_supervise :
  string * string array * string array * string * string * int64 * int ->
  (native_event -> int) -> native_result
  = "sounio_loom_membrane_supervise"

type policy = {
  manifest_sha256 : string;
  source_sha256 : string;
  source_u32 : string;
  semantics_sha256 : string;
  semantics_u32 : string;
  runtime : string;
  runtime_sha256 : string;
}

type activation_dark_policy = {
  action_manifest_sha256 : string;
  semantics_sha256 : string;
  operational_manifest_sha256 : string;
  resident_manifest_sha256 : string;
  projection_sha256 : string;
  frame : string;
  frame_sha256 : string;
  label : string;
}

type outcome = {
  kind : int;
  exit_code : int;
  signal : int;
  elapsed_us : int64;
  event_count : int;
  decision_code : int;
  timed_out : bool;
  policy_error : bool;
  landlock_abi : int;
  sandbox_sha256 : string;
  sandbox_ready : bool;
  authority_generation_sha256 : string;
  authority_pid : int;
  authority_sequence : int;
  activation_dark_code : int;
  activation_dark_result_sha256 : string;
  activation_dark_projection_sha256 : string;
  activation_dark_capsule_state : string;
  closure_code : int;
  closure_result_sha256 : string;
}

type product_launch_observation = {
  launch_code : int;
  launch_result_sha256 : string;
  launch_projection_sha256 : string;
  launch_capsule_state : string;
  launch_authority_generation_sha256 : string;
  launch_authority_pid : int;
  launch_authority_sequence : int;
}

type activation_dark_evaluation = {
  activation_policy : activation_dark_policy;
  activation_decision : Loom_resident.decision;
  activation_capsule_state : string;
}

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let starts_with value prefix =
  String.length value >= String.length prefix
  && String.sub value 0 (String.length prefix) = prefix

let ends_with value suffix =
  String.length value >= String.length suffix
  && String.sub value (String.length value - String.length suffix)
       (String.length suffix) = suffix

let trim = String.trim

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let read_file ?(limit = max_file_bytes) path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let output = Buffer.create 4096 in
      let bytes = Bytes.create 16384 in
      let rec loop total =
        let count = input channel bytes 0 (Bytes.length bytes) in
        if count = 0 then Buffer.contents output
        else if total + count > limit then failf "file-too-large:%s" path
        else (Buffer.add_subbytes output bytes 0 count; loop (total + count))
      in
      loop 0)

let sha256_file path = sha256 (read_file path)

let valid_sha256 value =
  String.length value = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let write_all descriptor value =
  let rec loop offset =
    if offset < String.length value then
      match Unix.write_substring descriptor value offset (String.length value - offset) with
      | 0 -> failf "short-membrane-write"
      | count -> loop (offset + count)
      | exception Unix_error (EINTR, _, _) -> loop offset
  in
  loop 0

let parse_manifest path =
  let table = Hashtbl.create 64 in
  read_file path |> String.split_on_char '\n'
  |> List.iter (fun line ->
         match String.index_opt line '=' with
         | None when line = "" -> ()
         | None -> failf "malformed-subprocess-membrane-manifest"
         | Some index ->
             let key = String.sub line 0 index in
             if Hashtbl.mem table key then failf "duplicate-membrane-field:%s" key;
             Hashtbl.add table key
               (String.sub line (index + 1) (String.length line - index - 1)));
  table

let required table key =
  match Hashtbl.find_opt table key with
  | Some value when value <> "" -> value
  | _ -> failf "missing-subprocess-membrane-field:%s" key

let digest_u32_of_hex digest =
  if String.length digest <> 64 then failf "invalid-sha256:%s" digest;
  List.init 8 (fun index ->
      let chunk = String.sub digest (index * 8) 8 in
      try Int64.to_string (Int64.of_string ("0x" ^ chunk))
      with _ -> failf "invalid-sha256:%s" digest)
  |> String.concat " "

let digest_u32_field table key =
  let values = String.split_on_char ',' (required table key) in
  if List.length values <> 8 then failf "invalid-membrane-digest:%s" key;
  List.iter
    (fun raw ->
      try
        let value = Int64.of_string raw in
        if value < 0L || value > 4294967295L then raise Exit
      with _ -> failf "invalid-membrane-digest:%s" key)
    values;
  String.concat " " values

let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"

let test_override name =
  match Sys.getenv_opt name with
  | Some value when value <> "" && test_mode () -> Some value
  | Some value when value <> "" -> failf "%s-override-requires-test-mode" name
  | _ -> None

let runtime_path root manifest =
  let sibling =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-subprocess-membrane-runtime"
  in
  let local =
    Filename.concat root
      "tools/loom/.runtime/sounio-loom-subprocess-membrane-runtime"
  in
  let selected =
    match Sys.getenv_opt "SOUNIO_LOOM_SUBPROCESS_MEMBRANE_RUNTIME" with
    | Some path when path <> "" -> path
    | _ when Sys.file_exists sibling -> sibling
    | _ -> local
  in
  if not (Sys.file_exists selected) then
    failf "subprocess-membrane-runtime-missing:%s" selected;
  let expected = required manifest "executable_sha256" in
  if sha256_file selected <> expected then
    failf "subprocess-membrane-runtime-hash-mismatch";
  (Unix.realpath selected, expected)

let load_policy root =
  let path =
    match test_override "SOUNIO_LOOM_SUBPROCESS_MEMBRANE_MANIFEST" with
    | Some path -> path
    | None -> Filename.concat root "tools/loom/subprocess_membrane.freeze.v1"
  in
  if not (Sys.file_exists path) then failf "subprocess-membrane-policy-missing";
  let manifest_sha256 = sha256_file path in
  if manifest_sha256 <> pinned_manifest_sha256 then
    failf "subprocess-membrane-policy-hash-mismatch";
  let manifest = parse_manifest path in
  if required manifest "schema" <> "loom-subprocess-membrane-freeze-v1"
     || required manifest "stage" <> "SEMANTICS_FROZEN"
     || required manifest "producing_language" <> "Sounio"
     || required manifest "language_role" <> "SEMANTIC_AUTHORITY"
     || required manifest "action" <> "9023"
     || required manifest "native_coverage_attested" <> "false"
     || required manifest "exec_attached" <> "false"
     || required manifest "commit_attached" <> "false"
     || required manifest "ci_attached" <> "false"
     || required manifest "parity_open" <> "false"
     || required manifest "claim_ready" <> "false"
  then failf "subprocess-membrane-policy-state-invalid";
  let source_path = Filename.concat root (required manifest "source_path") in
  let entrypoint_path = Filename.concat root (required manifest "entrypoint_path") in
  let source_sha256 = required manifest "source_sha256" in
  let semantics_sha256 = required manifest "semantics_sha256" in
  if sha256_file source_path <> source_sha256 then
    failf "subprocess-membrane-source-hash-mismatch";
  if sha256_file entrypoint_path <> required manifest "entrypoint_sha256" then
    failf "subprocess-membrane-entrypoint-hash-mismatch";
  if sha256 (read_file source_path ^ read_file entrypoint_path) <> semantics_sha256 then
    failf "subprocess-membrane-semantics-hash-mismatch";
  let authority_parent =
    Filename.concat root (required manifest "parent_execution_authority_manifest")
  in
  let outcome_parent =
    Filename.concat root (required manifest "parent_execution_outcome_manifest")
  in
  if sha256_file authority_parent <>
       required manifest "parent_execution_authority_manifest_sha256"
     || sha256_file outcome_parent <>
          required manifest "parent_execution_outcome_manifest_sha256"
  then failf "subprocess-membrane-parent-authority-mismatch";
  let runtime, runtime_sha256 = runtime_path root manifest in
  { manifest_sha256;
    source_sha256;
    source_u32 = digest_u32_field manifest "source_sha256_u32";
    semantics_sha256;
    semantics_u32 = digest_u32_field manifest "semantics_sha256_u32";
    runtime;
    runtime_sha256 }

let find_substring value marker =
  let rec loop offset =
    if offset + String.length marker > String.length value then None
    else if String.sub value offset (String.length marker) = marker then
      Some offset
    else loop (offset + 1)
  in
  loop 0

let load_activation_dark_policy root =
  let action_path =
    match test_override "SOUNIO_LOOM_ACTIVATION_DARK_ACTION_MANIFEST" with
    | Some path -> path
    | None ->
        Filename.concat root
          "tools/loom/kernel_peer_activation_capsule_authority.freeze.v1"
  in
  let operational_path =
    match test_override "SOUNIO_LOOM_ACTIVATION_DARK_OPERATIONAL_MANIFEST" with
    | Some path -> path
    | None ->
        Filename.concat root "tools/loom/kernel_peer_activation_capsule.runtime.v1"
  in
  let resident_path =
    match test_override "SOUNIO_LOOM_ACTIVATION_DARK_RESIDENT_MANIFEST" with
    | Some path -> path
    | None -> Filename.concat root "tools/loom/resident_membrane.runtime.v5"
  in
  let projection_path =
    match test_override "SOUNIO_LOOM_ACTIVATION_DARK_PROJECTION" with
    | Some path -> path
    | None ->
        Filename.concat root
          "tools/loom/kernel_peer_activation_capsule.current.v1"
  in
  if sha256_file action_path <> pinned_activation_manifest_sha256 then
    failf "activation-dark-action-manifest-hash-mismatch";
  if sha256_file operational_path <> pinned_activation_runtime_sha256 then
    failf "activation-dark-operational-manifest-hash-mismatch";
  if sha256_file resident_path <> pinned_resident_v5_manifest_sha256 then
    failf "activation-dark-resident-manifest-hash-mismatch";
  if sha256_file projection_path <> pinned_activation_projection_sha256 then
    failf "activation-dark-projection-hash-mismatch";
  let action = parse_manifest action_path in
  let operational = parse_manifest operational_path in
  let resident = parse_manifest resident_path in
  if required action "schema"
       <> "loom-kernel-peer-activation-capsule-authority-freeze-v1"
     || required action "stage" <> "SEMANTICS_FROZEN"
     || required action "producing_language" <> "Sounio"
     || required action "language_role" <> "SEMANTIC_AUTHORITY"
     || required action "action" <> "9031"
     || required action "operational_realization" <> "false"
     || required action "production_activation" <> "false"
     || required action "capsule_is_bearer" <> "false"
  then failf "activation-dark-action-state-invalid";
  if required operational "schema"
       <> "loom-kernel-peer-activation-capsule-ocaml-runtime-v1"
     || required operational "stage" <> "OPERATIONAL_REALIZATION_FROZEN"
     || required operational "producing_language" <> "OCaml"
     || required operational "language_role" <> "OPERATIONAL_REALIZATION"
     || required operational "semantic_authority" <> "Sounio"
     || required operational "operational_realization" <> "true"
     || required operational "production_activation" <> "false"
     || required operational "parent_9031_manifest_sha256"
          <> pinned_activation_manifest_sha256
     || required operational "parent_resident_v5_manifest_sha256"
          <> pinned_resident_v5_manifest_sha256
  then failf "activation-dark-operational-state-invalid";
  if required resident "schema" <> "loom-resident-membrane-runtime-v5"
     || required resident "runtime_frozen" <> "true"
     || required resident "process_model" <> "single-resident-sounio-pid"
     || required resident "parent_9031_sha256"
          <> pinned_activation_manifest_sha256
     || required resident "route_9031" <> "6"
     || required resident "production_activation" <> "false"
  then failf "activation-dark-resident-state-invalid";
  let label =
    match test_override "SOUNIO_LOOM_ACTIVATION_DARK_LABEL" with
    | Some label -> label
    | None -> "current_material"
  in
  if label <> "current_material" && label <> "seal" then
    failf "activation-dark-label-invalid";
  let prefix = "CASE label=" ^ label ^ " EXPECT code=" in
  let matches =
    read_file projection_path |> String.split_on_char '\n'
    |> List.filter (fun line -> starts_with line prefix)
  in
  let line =
    match matches with
    | [ line ] -> line
    | [] -> failf "activation-dark-projection-label-missing"
    | _ -> failf "activation-dark-projection-label-duplicate"
  in
  let marker = " FRAME " in
  let offset =
    match find_substring line marker with
    | Some offset -> offset + String.length marker
    | None -> failf "activation-dark-projection-frame-missing"
  in
  let frame = String.sub line offset (String.length line - offset) in
  if frame = "" || not (starts_with frame "9031 3 1 ") then
    failf "activation-dark-projection-frame-invalid";
  { action_manifest_sha256 = pinned_activation_manifest_sha256;
    semantics_sha256 = required action "semantics_sha256";
    operational_manifest_sha256 = pinned_activation_runtime_sha256;
    resident_manifest_sha256 = pinned_resident_v5_manifest_sha256;
    projection_sha256 = pinned_activation_projection_sha256;
    frame; frame_sha256 = sha256 frame; label }

let normalize_absolute cwd value =
  let raw = if Filename.is_relative value then Filename.concat cwd value else value in
  let parts = String.split_on_char '/' raw in
  let reduced =
    List.fold_left
      (fun stack part ->
        match part, stack with
        | ("" | "."), _ -> stack
        | "..", _ :: tail -> tail
        | "..", [] -> []
        | _, _ -> part :: stack)
      [] parts
    |> List.rev
  in
  "/" ^ String.concat "/" reduced

let within root path =
  path = root || starts_with path (if root = "/" then "/" else root ^ "/")

let basename_lower path = Filename.basename path |> String.lowercase_ascii

let classified_language target =
  let name = basename_lower target in
  if starts_with name "python" || starts_with name "pypy" then 7
  else if List.mem name [ "cargo"; "rustc"; "rustup"; "rustfmt"; "clippy-driver" ]
  then 8
  else if List.mem name [ "souc"; "madaros"; "sounio" ] || starts_with name "souc-"
  then 1
  else if List.mem name [ "lean"; "lake"; "elan" ] then 2
  else if name = "koka" then 3
  else if List.mem name [ "cc"; "c++"; "gcc"; "g++"; "clang"; "clang++" ]
  then 4
  else if List.mem name [ "ghc"; "runghc"; "cabal"; "stack" ] then 5
  else if List.mem name
      [ "claude"; "codex"; "cursor"; "grok"; "kimi"; "minimax";
        "gemini"; "ollama"; "vllm"; "zai" ]
  then 6
  else if starts_with name "ocaml" || List.mem name [ "dune"; "opam" ]
          || starts_with name "sounio-loom"
  then 9
  else if List.mem name [ "sh"; "bash"; "zsh"; "dash"; "fish"; "ksh" ]
  then 11
  else if name = "git" then 12
  else if starts_with target "<" then 13
  else 10

let purpose_for_language = function
  | 2 | 3 | 4 | 5 -> 3
  | 6 -> 4
  | _ -> 1

let surface_for_argv argv =
  match Array.to_list argv with
  | executable :: "commit" :: _ when basename_lower executable = "git" -> 2
  | _ -> 1

let semantic_target path = ends_with path ".sio"

let expected_result_target path =
  let name = basename_lower path in
  starts_with name "expected" || starts_with name "golden"
  || ends_with name ".expected" || ends_with name ".golden"

let hardware_record () =
  let kernel =
    if Sys.file_exists "/proc/sys/kernel/osrelease" then
      trim (read_file ~limit:65536 "/proc/sys/kernel/osrelease")
    else "unavailable"
  in
  String.concat "\n"
    [ "os_type=" ^ Sys.os_type;
      "kernel=" ^ kernel;
      "word_size=" ^ string_of_int Sys.word_size;
      "hostname=" ^ Unix.gethostname () ] ^ "\n"

type process_result = { code : int; output : string }

let exit_code = function
  | WEXITED code -> code
  | WSIGNALED signal | WSTOPPED signal -> 128 + signal

let run_authority ~root ~runtime ~environment frame =
  let stdin_read, stdin_write = Unix.pipe () in
  let output_read, output_write = Unix.pipe () in
  Unix.set_close_on_exec stdin_write;
  Unix.set_close_on_exec output_read;
  let pid =
    match Unix.fork () with
    | 0 ->
        Unix.close stdin_write;
        Unix.close output_read;
        Unix.dup2 stdin_read Unix.stdin;
        Unix.dup2 output_write Unix.stdout;
        Unix.dup2 output_write Unix.stderr;
        if stdin_read <> Unix.stdin then Unix.close stdin_read;
        if output_write <> Unix.stdout && output_write <> Unix.stderr then
          Unix.close output_write;
        (try Unix.chdir root; Unix.execve runtime [| runtime |] environment
         with _ -> Unix._exit 127)
    | pid -> pid
  in
  Unix.close stdin_read;
  Unix.close output_write;
  let close_noerr descriptor = try Unix.close descriptor with _ -> () in
  let kill_noerr () =
    (try Unix.kill pid Sys.sigkill with _ -> ());
    (try ignore (Unix.waitpid [] pid) with _ -> ())
  in
  Fun.protect
    ~finally:(fun () -> close_noerr stdin_write; close_noerr output_read)
    (fun () ->
      (try write_all stdin_write frame with error -> kill_noerr (); raise error);
      Unix.close stdin_write;
      let deadline = Unix.gettimeofday () +. authority_timeout_seconds in
      let output = Buffer.create 512 in
      let bytes = Bytes.create 4096 in
      let rec drain () =
        let remaining = deadline -. Unix.gettimeofday () in
        if remaining <= 0.0 then (kill_noerr (); failf "subprocess-membrane-policy-timeout");
        let ready, _, _ = Unix.select [ output_read ] [] [] remaining in
        if ready = [] then (kill_noerr (); failf "subprocess-membrane-policy-timeout")
        else
          match Unix.read output_read bytes 0 (Bytes.length bytes) with
          | 0 -> ()
          | count ->
              if Buffer.length output + count > max_file_bytes then (
                kill_noerr (); failf "subprocess-membrane-policy-output-too-large");
              Buffer.add_subbytes output bytes 0 count;
              drain ()
          | exception Unix_error (EINTR, _, _) -> drain ()
      in
      drain ();
      let _, status = Unix.waitpid [] pid in
      { code = exit_code status; output = Buffer.contents output })

let decision_code output =
  let marker = " code=" in
  match String.index_opt output '=' with
  | None -> failf "subprocess-membrane-decision-code-missing"
  | Some _ ->
      let rec find offset =
        if offset + String.length marker > String.length output then
          failf "subprocess-membrane-decision-code-missing"
        else if String.sub output offset (String.length marker) = marker then offset
        else find (offset + 1)
      in
      let offset = find 0 + String.length marker in
      let ending =
        match String.index_from_opt output offset ' ' with
        | Some value -> value
        | None -> String.length output
      in
      try int_of_string (String.sub output offset (ending - offset))
      with _ -> failf "subprocess-membrane-decision-code-invalid"

let zero_digest = "0 0 0 0 0 0 0 0"

let decision_frame policy ~surface ~effect_kind ~language ~target ~scope
    ~deadline_hash ~command_hash ~hardware_hash ~active_count ~outcome_hash
    ~outcome_complete ~termination_complete =
  let purpose = purpose_for_language language in
  let write_effect = effect_kind = 4 || effect_kind = 5 in
  let target_normalized = normalize_absolute "/" target in
  let scope_bound =
    if write_effect && not (starts_with target "<") && within scope target_normalized
    then 1 else 0
  in
  let supported = if starts_with target "<" then 0 else 1 in
  let ancestry = if effect_kind = 1 then 0 else 1 in
  let semantic_write = if write_effect && semantic_target target then 1 else 0 in
  let expected_write = if write_effect && expected_result_target target then 1 else 0 in
  let target_hash = sha256 target |> digest_u32_of_hex in
  let claim_hash = if write_effect then sha256 scope |> digest_u32_of_hex else zero_digest in
  let toolchain_hash =
    if Sys.file_exists target && not (Sys.is_directory target) then sha256_file target
    else sha256 ("target:" ^ target)
  in
  let event_hash =
    sha256
      (String.concat "\n"
         [ string_of_int effect_kind; string_of_int language; target;
           string_of_int active_count ])
  in
  let actor_hash = sha256 (string_of_int language ^ ":" ^ target) in
  String.concat " "
    [ "9023"; "3"; string_of_int surface; string_of_int effect_kind;
      string_of_int language; string_of_int purpose; "1"; "1"; "1";
      string_of_int ancestry; "1"; "1"; string_of_int scope_bound; "1";
      (if active_count <= 1 then "1" else "0");
      string_of_int outcome_complete; "0"; string_of_int supported; "1";
      string_of_int termination_complete; string_of_int semantic_write;
      string_of_int expected_write; "0"; "0"; "0"; "0"; "0"; "0"; "0";
      policy.source_u32; policy.semantics_u32; policy.semantics_u32;
      digest_u32_of_hex toolchain_hash; digest_u32_of_hex hardware_hash;
      digest_u32_of_hex command_hash; digest_u32_of_hex event_hash;
      digest_u32_of_hex actor_hash; target_hash; claim_hash;
      digest_u32_of_hex deadline_hash;
      (if outcome_complete = 1 then digest_u32_of_hex outcome_hash else zero_digest);
      zero_digest ] ^ "\n"

let invoke_decision ~root ~policy ~environment frame =
  let result = run_authority ~root ~runtime:policy.runtime ~environment frame in
  let output = trim result.output in
  if not (starts_with output "SOUNIO_SUBPROCESS_MEMBRANE_")
     || not (ends_with output "stage=SEMANTICS_FROZEN")
  then failf "subprocess-membrane-policy-invalid-result:rc=%d:%s" result.code output;
  let code = decision_code output in
  if code = 0 && result.code <> 0 then
    failf "subprocess-membrane-policy-exit-mismatch:rc=%d" result.code;
  (code, output)

let utc_now () =
  let tm = Unix.gmtime (Unix.gettimeofday ()) in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02dZ"
    (tm.tm_year + 1900) (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min tm.tm_sec

let git_common_dir root =
  let marker = Filename.concat root ".git" in
  if Sys.is_directory marker then Unix.realpath marker
  else
    let line =
      match read_file ~limit:65536 marker |> String.split_on_char '\n' with
      | value :: _ -> trim value
      | [] -> ""
    in
    if not (starts_with line "gitdir: ") then failf "invalid-gitdir-marker";
    let raw = String.sub line 8 (String.length line - 8) in
    let git_dir = Unix.realpath (normalize_absolute root raw) in
    let common_marker = Filename.concat git_dir "commondir" in
    if Sys.file_exists common_marker then
      let common =
        match read_file ~limit:65536 common_marker |> String.split_on_char '\n' with
        | value :: _ -> trim value
        | [] -> ""
      in
      Unix.realpath (normalize_absolute git_dir common)
    else git_dir

let decision_log_path root =
  match test_override "SOUNIO_LOOM_SUBPROCESS_MEMBRANE_LOG" with
  | Some path -> path
  | None -> Filename.concat (git_common_dir root) "sounio-loom-subprocess-membrane.tsv"

let append_decision ~root ~policy ~sandbox_sha256 ~effect_kind ~target ~frame
    ~decision ~output =
  let path = decision_log_path root in
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      let line =
        String.concat "\t"
          [ "schema=loom-subprocess-membrane-decision-v1";
            "utc=" ^ utc_now ();
            "decision=" ^ (if decision = 0 then "ALLOW" else "DENY");
            "code=" ^ string_of_int decision;
            "effect_kind=" ^ string_of_int effect_kind;
            "target_sha256=" ^ sha256 target;
            "manifest_sha256=" ^ policy.manifest_sha256;
            "source_sha256=" ^ policy.source_sha256;
            "semantics_sha256=" ^ policy.semantics_sha256;
            "runtime_sha256=" ^ policy.runtime_sha256;
            "sandbox_sha256=" ^ sandbox_sha256;
            "frame_sha256=" ^ sha256 frame;
            "authority_result_sha256=" ^ sha256 output ] ^ "\n"
      in
      write_all descriptor line;
      Unix.fsync descriptor;
      Unix.lockf descriptor F_ULOCK 0)

let activation_dark_log_path root =
  match test_override "SOUNIO_LOOM_ACTIVATION_DARK_LOG" with
  | Some path -> path
  | None ->
      Filename.concat (git_common_dir root)
        "sounio-loom-product-activation-dark.tsv"

let append_activation_dark_decision ~root ~policy ~capsule_state
    (decision : Loom_resident.decision) =
  let path = activation_dark_log_path root in
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      let line =
        String.concat "\t"
          [ "schema=loom-product-activation-dark-decision-v1";
            "utc=" ^ utc_now ();
            "decision=" ^ (if decision.code = 0 then "ALLOW" else "DENY");
            "code=" ^ string_of_int decision.code;
            "authorizing=false";
            "production_activation=false";
            "producing_language=Sounio";
            "language_role=SEMANTIC_AUTHORITY";
            "operational_language=OCaml";
            "operational_role=OPERATIONAL_REALIZATION";
            "action_manifest_sha256=" ^ policy.action_manifest_sha256;
            "semantics_sha256=" ^ policy.semantics_sha256;
            "operational_manifest_sha256="
              ^ policy.operational_manifest_sha256;
            "resident_manifest_sha256=" ^ policy.resident_manifest_sha256;
            "projection_sha256=" ^ policy.projection_sha256;
            "projection_label=" ^ policy.label;
            "capsule_state_after=" ^ capsule_state;
            "frame_sha256=" ^ policy.frame_sha256;
            "authority_result_sha256=" ^ sha256 decision.output;
            "authority_generation_sha256=" ^ decision.generation_sha256;
            "authority_pid=" ^ string_of_int decision.resident_pid;
            "authority_sequence=" ^ string_of_int decision.sequence;
            "authority_latency_us=" ^ Int64.to_string decision.latency_us ] ^ "\n"
      in
      write_all descriptor line;
      Unix.fsync descriptor;
      Unix.lockf descriptor F_ULOCK 0)

let product_launch_dark_log_path audit_root =
  match test_override "SOUNIO_LOOM_PRODUCT_LAUNCH_DARK_LOG" with
  | Some path -> path
  | None -> Filename.concat audit_root "product-launch-dark.tsv"

let valid_launch_source = function
  | "start" | "provider-start" | "provider-open" | "recover" -> true
  | _ -> false

let append_product_launch_dark_decision ~audit_root ~policy ~operation
    ~launch_source ~agent ~lane ~session_id ~cwd ~command_sha256 ~capsule_state
    (decision : Loom_resident.decision) =
  let path = product_launch_dark_log_path audit_root in
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      let line =
        String.concat "\t"
          [ "schema=loom-product-launch-dark-decision-v1";
            "utc=" ^ utc_now ();
            "operation=" ^ operation;
            "launch_source=" ^ launch_source;
            "decision=" ^ (if decision.code = 0 then "ALLOW" else "DENY");
            "code=" ^ string_of_int decision.code;
            "authorizing=false";
            "production_activation=false";
            "live_material=false";
            "producing_language=Sounio";
            "language_role=SEMANTIC_AUTHORITY";
            "operational_language=OCaml";
            "operational_role=OPERATIONAL_ATTACHMENT";
            "agent_sha256=" ^ sha256 agent;
            "lane_sha256=" ^ sha256 lane;
            "session_id_sha256=" ^ sha256 session_id;
            "cwd_sha256=" ^ sha256 cwd;
            "command_sha256=" ^ command_sha256;
            "action_manifest_sha256=" ^ policy.action_manifest_sha256;
            "semantics_sha256=" ^ policy.semantics_sha256;
            "operational_manifest_sha256="
              ^ policy.operational_manifest_sha256;
            "resident_manifest_sha256=" ^ policy.resident_manifest_sha256;
            "projection_sha256=" ^ policy.projection_sha256;
            "projection_label=" ^ policy.label;
            "capsule_state_after=" ^ capsule_state;
            "frame_sha256=" ^ policy.frame_sha256;
            "authority_result_sha256=" ^ sha256 decision.output;
            "authority_generation_sha256=" ^ decision.generation_sha256;
            "authority_pid=" ^ string_of_int decision.resident_pid;
            "authority_sequence=" ^ string_of_int decision.sequence;
            "authority_latency_us=" ^ Int64.to_string decision.latency_us ]
        ^ "\n"
      in
      write_all descriptor line;
      Unix.fsync descriptor;
      Unix.lockf descriptor F_ULOCK 0)

let authority_environment () =
  let prohibited_prefixes =
    [ "LD_PRELOAD="; "LD_LIBRARY_PATH="; "LD_AUDIT=";
      "SOUNIO_LOOM_SUBPROCESS_MEMBRANE_"; "SOUNIO_LOOM_RESIDENT_";
      "SOUNIO_LOOM_HOOK_TEST_MODE=" ]
  in
  Unix.environment () |> Array.to_list
  |> List.filter (fun binding ->
         not (List.exists (starts_with binding) prohibited_prefixes))
  |> Array.of_list

let evaluate_product_activation_dark ~policy_root ~audit_root ~deadline_ms =
  if deadline_ms < 1 || deadline_ms > 120_000 then
    failf "product-activation-dark-deadline-out-of-range";
  let policy_root = Unix.realpath policy_root in
  let audit_root = Unix.realpath audit_root in
  let policy = load_activation_dark_policy policy_root in
  let environment = authority_environment () in
  let decision, capsule_state =
    Loom_peer_activation_capsule.with_cell ~root:policy_root ~audit_root
      ~environment ~deadline_ms
      (fun capsule ->
        if Loom_peer_activation_capsule.manifest_sha256 capsule
             <> policy.action_manifest_sha256
           || Loom_peer_activation_capsule.semantics_sha256 capsule
                <> policy.semantics_sha256
           || Loom_peer_activation_capsule.resident_v5_sha256 capsule
                <> policy.resident_manifest_sha256
        then failf "product-activation-dark-capsule-binding-mismatch";
        let decision =
          Loom_peer_activation_capsule.seal capsule policy.frame
        in
        let capsule_state =
          Loom_peer_activation_capsule.state capsule
          |> Loom_peer_activation_capsule.state_name
        in
        (decision, capsule_state))
  in
  { activation_policy = policy;
    activation_decision = decision;
    activation_capsule_state = capsule_state }

let observe_product_launch ~policy_root ~audit_root ~operation ~launch_source
    ~agent ~lane ~session_id ~cwd ~command_sha256 ~deadline_ms =
  if operation <> "start" && operation <> "recover" then
    failf "product-launch-dark-operation-invalid";
  if not (valid_launch_source launch_source) then
    failf "product-launch-dark-source-invalid";
  if agent = "" || lane = "" || session_id = "" then
    failf "product-launch-dark-identity-missing";
  if not (valid_sha256 command_sha256) then
    failf "product-launch-dark-command-digest-invalid";
  let cwd = Unix.realpath cwd in
  let evaluation =
    evaluate_product_activation_dark ~policy_root ~audit_root ~deadline_ms
  in
  let policy = evaluation.activation_policy in
  let decision = evaluation.activation_decision in
  let capsule_state = evaluation.activation_capsule_state in
  append_product_launch_dark_decision ~audit_root ~policy ~operation
    ~launch_source ~agent ~lane ~session_id ~cwd ~command_sha256 ~capsule_state
    decision;
  if decision.code = 0 then failf "product-launch-dark-unexpected-allow";
  { launch_code = decision.code;
    launch_result_sha256 = sha256 decision.output;
    launch_projection_sha256 = policy.projection_sha256;
    launch_capsule_state = capsule_state;
    launch_authority_generation_sha256 = decision.generation_sha256;
    launch_authority_pid = decision.resident_pid;
    launch_authority_sequence = decision.sequence }

let run_probe ~root ~cwd ~scope ~deadline_ms ~argv =
  if Array.length argv = 0 then failf "subprocess-membrane-command-missing";
  if deadline_ms < 1 || deadline_ms > 120_000 then
    failf "subprocess-membrane-deadline-out-of-range";
  let root = Unix.realpath root in
  let cwd = Unix.realpath cwd in
  let scope = Unix.realpath scope in
  if not (within root cwd) then failf "subprocess-membrane-cwd-outside-root";
  if not (within root scope) then failf "subprocess-membrane-scope-outside-root";
  let executable =
    let raw = argv.(0) in
    let resolved = if Filename.is_relative raw then normalize_absolute cwd raw else raw in
    Unix.realpath resolved
  in
  let sandbox =
    match test_override "SOUNIO_LOOM_SUBPROCESS_MEMBRANE_SANDBOX" with
    | Some path -> Unix.realpath path
    | None -> Unix.realpath "/usr/bin/bwrap"
  in
  let sandbox_sha256 = sha256_file sandbox in
  if sandbox_sha256 <> pinned_sandbox_sha256 then
    failf "subprocess-membrane-sandbox-hash-mismatch";
  let argv = Array.copy argv in
  argv.(0) <- executable;
  let policy = load_policy root in
  let activation_policy = load_activation_dark_policy root in
  let authority_environment = authority_environment () in
  let child_environment = authority_environment in
  let command_hash = sha256 (Array.to_list argv |> String.concat "\000") in
  let hardware_hash = sha256 (hardware_record ()) in
  let deadline_us = Int64.mul (Int64.of_int deadline_ms) 1000L in
  let started_us = Loom_resident.now_us () in
  if started_us <= 0L then failf "subprocess-membrane-monotonic-clock-failed";
  let absolute_deadline_us = Int64.add started_us deadline_us in
  let remaining_deadline_us () =
    let remaining = Int64.sub absolute_deadline_us (Loom_resident.now_us ()) in
    if remaining <= 0L then failf "subprocess-membrane-policy-timeout";
    remaining
  in
  let remaining_deadline_ms () =
    let remaining = remaining_deadline_us () in
    Int64.div (Int64.add remaining 999L) 1000L |> Int64.to_int
  in
  let deadline_hash = sha256 ("deadline_us=" ^ Int64.to_string deadline_us ^ "\n") in
  let surface = surface_for_argv argv in
  Loom_peer_activation_capsule.with_cell ~root
    ~environment:authority_environment ~deadline_ms:(remaining_deadline_ms ())
    (fun capsule ->
  if Loom_peer_activation_capsule.manifest_sha256 capsule
       <> activation_policy.action_manifest_sha256
     || Loom_peer_activation_capsule.semantics_sha256 capsule
          <> activation_policy.semantics_sha256
     || Loom_peer_activation_capsule.resident_v5_sha256 capsule
          <> activation_policy.resident_manifest_sha256
  then failf "activation-dark-capsule-binding-mismatch";
  let resident = Loom_peer_activation_capsule.resident capsule in
  let activation_decision =
    Loom_peer_activation_capsule.seal capsule activation_policy.frame
  in
  let activation_capsule_state =
    Loom_peer_activation_capsule.state capsule
    |> Loom_peer_activation_capsule.state_name
  in
  append_activation_dark_decision ~root ~policy:activation_policy
    ~capsule_state:activation_capsule_state activation_decision;
  if activation_decision.code = 0 then
    failf "activation-dark-unexpected-allow";
  let decide_closure () =
    let frame = Loom_effect_closure.current_material_frame root in
    let decision =
      Loom_resident.decide_closure resident
        ~deadline_ms:(remaining_deadline_ms ()) frame
    in
    if decision.code = 0 then
      failf "effect-closure-current-material-admitted";
    (decision.code, sha256 decision.output)
  in
  let decide effect_kind target active_count outcome_hash outcome_complete
      termination_complete =
    try
      let language =
        if effect_kind = 3 || effect_kind = 1 then classified_language target
        else classified_language executable
      in
      let frame =
        decision_frame policy ~surface ~effect_kind ~language ~target ~scope
          ~deadline_hash ~command_hash ~hardware_hash ~active_count ~outcome_hash
          ~outcome_complete ~termination_complete
      in
      let resident_decision =
        Loom_resident.decide resident ~deadline_ms:(remaining_deadline_ms ()) frame
      in
      let code, output = resident_decision.code, resident_decision.output in
      append_decision ~root ~policy ~sandbox_sha256 ~effect_kind ~target ~frame
        ~decision:code ~output;
      code
    with
    | Loom_resident.Error reason
    | Error reason
    | Sys_error reason ->
        let output = "policy-error:" ^ reason in
        let frame = sha256 output in
        (try
           append_decision ~root ~policy ~sandbox_sha256 ~effect_kind ~target
             ~frame ~decision:403 ~output
         with _ -> ());
        403
    | Unix_error (error, function_name, argument) ->
        let output =
          Printf.sprintf "policy-error:%s:%s(%s)" (Unix.error_message error)
            function_name argument
        in
        (try
           append_decision ~root ~policy ~sandbox_sha256 ~effect_kind ~target
             ~frame:(sha256 output) ~decision:403 ~output
         with _ -> ());
        403
  in
  let closure_code, closure_result_sha256 = decide_closure () in
  let root_code = decide 1 executable 1 (sha256 "root-pending") 0 0 in
  if root_code <> 0 then
    { kind = 5; exit_code = 0; signal = 0; elapsed_us = 0L;
      event_count = 1; decision_code = root_code; timed_out = false;
      policy_error = root_code = 403; landlock_abi = 0; sandbox_sha256;
      sandbox_ready = false;
      authority_generation_sha256 = Loom_resident.generation resident;
      authority_pid = Loom_resident.pid resident;
      authority_sequence = Loom_resident.sequence resident;
      activation_dark_code = activation_decision.code;
      activation_dark_result_sha256 = sha256 activation_decision.output;
      activation_dark_projection_sha256 = activation_policy.projection_sha256;
      activation_dark_capsule_state = activation_capsule_state;
      closure_code; closure_result_sha256 }
  else
    let callback (effect_kind, _pid, _syscall, target, active_count) =
      decide effect_kind target active_count (sha256 "event-pending") 0 0
    in
    let native_flags =
      match test_override
              "SOUNIO_LOOM_SUBPROCESS_MEMBRANE_TEST_DISABLE_FS_OBSERVER"
      with
      | None -> 0
      | Some "1" -> 1
      | Some _ -> failf "invalid-disable-fs-observer-test-control"
    in
    let kind, exit_code, signal, elapsed_us, event_count, decision_code,
        timed_out, policy_error, landlock_abi, sandbox_ready =
      let sandbox_argv =
        Array.of_list
          ([ sandbox; "--die-with-parent"; "--new-session"; "--unshare-user";
             "--unshare-pid"; "--unshare-net"; "--unshare-ipc";
             "--unshare-uts"; "--ro-bind"; "/"; "/"; "--dev"; "/dev";
             "--tmpfs"; "/tmp"; "--bind"; scope; scope; "--chdir"; cwd;
             "--cap-drop"; "ALL"; "--" ] @ Array.to_list argv)
      in
      native_supervise
        (sandbox, sandbox_argv, child_environment, cwd, executable,
         remaining_deadline_us (), native_flags)
        callback
    in
    let outcome_hash =
      sha256
        (Printf.sprintf "kind=%d\nexit=%d\nsignal=%d\nelapsed_us=%Ld\n"
           kind exit_code signal elapsed_us)
    in
    let final_outcome_complete =
      match test_override
              "SOUNIO_LOOM_SUBPROCESS_MEMBRANE_TEST_FINAL_OUTCOME_INCOMPLETE"
      with
      | None -> 1
      | Some "1" -> 0
      | Some _ -> failf "invalid-final-outcome-incomplete-test-control"
    in
    let final_code =
      if kind = 4 then
        decide 7 executable 1 outcome_hash final_outcome_complete 1
      else if kind = 1 || kind = 2 then
        decide 6 executable 1 outcome_hash final_outcome_complete 0
      else decision_code
    in
    let final_kind =
      if final_code <> 0 && (kind = 1 || kind = 2) then 5 else kind
    in
    { kind = final_kind; exit_code; signal; elapsed_us; event_count;
      decision_code = (if final_code <> 0 then final_code else decision_code);
      timed_out = timed_out = 1;
      policy_error = policy_error = 1 || final_code = 403;
      landlock_abi; sandbox_sha256; sandbox_ready = sandbox_ready = 1;
      authority_generation_sha256 = Loom_resident.generation resident;
      authority_pid = Loom_resident.pid resident;
      authority_sequence = Loom_resident.sequence resident;
      activation_dark_code = activation_decision.code;
      activation_dark_result_sha256 = sha256 activation_decision.output;
      activation_dark_projection_sha256 = activation_policy.projection_sha256;
      activation_dark_capsule_state = activation_capsule_state;
      closure_code; closure_result_sha256 })

let exit_status outcome =
  match outcome.kind with
  | 1 -> outcome.exit_code
  | 2 -> 128 + outcome.signal
  | 4 -> 124
  | 5 | 6 -> 126
  | _ -> 126
