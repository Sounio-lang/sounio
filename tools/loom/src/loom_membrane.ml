open Unix

exception Error of string

let pinned_manifest_sha256 =
  "0024178b8928f0c82d794d390244e83e5ce431054587fc7dd609c0f25c2e5b4f"

let authority_timeout_seconds = 5.0
let max_file_bytes = 8 * 1024 * 1024

type native_event = int * int * int * string * int

type native_result =
  int * int * int * int64 * int * int * int * int

external native_supervise :
  string * string array * string array * string * int64 * int ->
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

type outcome = {
  kind : int;
  exit_code : int;
  signal : int;
  elapsed_us : int64;
  event_count : int;
  decision_code : int;
  timed_out : bool;
  policy_error : bool;
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

let append_decision ~root ~policy ~effect_kind ~target ~frame ~decision ~output =
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
            "frame_sha256=" ^ sha256 frame;
            "authority_result_sha256=" ^ sha256 output ] ^ "\n"
      in
      write_all descriptor line;
      Unix.fsync descriptor;
      Unix.lockf descriptor F_ULOCK 0)

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
  let argv = Array.copy argv in
  argv.(0) <- executable;
  let policy = load_policy root in
  let environment = Unix.environment () in
  let command_hash = sha256 (Array.to_list argv |> String.concat "\000") in
  let hardware_hash = sha256 (hardware_record ()) in
  let deadline_us = Int64.mul (Int64.of_int deadline_ms) 1000L in
  let deadline_hash = sha256 ("deadline_us=" ^ Int64.to_string deadline_us ^ "\n") in
  let surface = surface_for_argv argv in
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
      let code, output = invoke_decision ~root ~policy ~environment frame in
      append_decision ~root ~policy ~effect_kind ~target ~frame ~decision:code
        ~output;
      code
    with
    | Error reason
    | Sys_error reason ->
        let output = "policy-error:" ^ reason in
        let frame = sha256 output in
        (try
           append_decision ~root ~policy ~effect_kind ~target ~frame ~decision:403
             ~output
         with _ -> ());
        403
    | Unix_error (error, function_name, argument) ->
        let output =
          Printf.sprintf "policy-error:%s:%s(%s)" (Unix.error_message error)
            function_name argument
        in
        (try
           append_decision ~root ~policy ~effect_kind ~target ~frame:(sha256 output)
             ~decision:403 ~output
         with _ -> ());
        403
  in
  let root_code = decide 1 executable 1 (sha256 "root-pending") 0 0 in
  if root_code <> 0 then
    { kind = 5; exit_code = 0; signal = 0; elapsed_us = 0L;
      event_count = 1; decision_code = root_code; timed_out = false;
      policy_error = root_code = 403 }
  else
    let callback (effect_kind, _pid, _syscall, target, active_count) =
      decide effect_kind target active_count (sha256 "event-pending") 0 0
    in
    let kind, exit_code, signal, elapsed_us, event_count, decision_code,
        timed_out, policy_error =
      native_supervise
        (executable, argv, environment, cwd, deadline_us, 0)
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
      policy_error = policy_error = 1 || final_code = 403 }

let exit_status outcome =
  match outcome.kind with
  | 1 -> outcome.exit_code
  | 2 -> 128 + outcome.signal
  | 4 -> 124
  | 5 | 6 -> 126
  | _ -> 126
