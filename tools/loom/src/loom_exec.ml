open Unix

exception Error of string
exception Dynamic_command of string
exception Authority_denied of int * string

let pinned_manifest_sha256 =
  "d07823382125e668eb0d7afe5d52092de3c58ec5bef9655fa2f1f56a9c84d8c0"

let pinned_outcome_manifest_sha256 =
  "f5e63a2fd6a946cea1a4cb57013ae0cfa1772c42c3cc52e42d300dfb7b45e16e"

let max_file_bytes = 8 * 1024 * 1024
let max_command_bytes = 64 * 1024
let max_arguments = 256
let authority_timeout_seconds = 5.0
let default_capability_ttl_seconds = 30
let max_kernel_control_bytes = 2 * 1024 * 1024
let kernel_connect_timeout_seconds = 3.0

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

let hex_encode value =
  Cryptokit.transform_string (Cryptokit.Hexa.encode ()) value

let hex_decode name value =
  if String.length value mod 2 <> 0 then failf "invalid-capability-hex:%s" name;
  try Cryptokit.transform_string (Cryptokit.Hexa.decode ()) value
  with _ -> failf "invalid-capability-hex:%s" name

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

let sha256_file path =
  let stat = Unix.lstat path in
  if stat.st_kind <> S_REG then failf "file-not-regular:%s" path;
  let channel = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () ->
      Cryptokit.hash_channel (Cryptokit.Hash.sha256 ()) channel
      |> Cryptokit.transform_string (Cryptokit.Hexa.encode ()))

let write_all descriptor value =
  let rec loop offset =
    if offset < String.length value then
      match Unix.write_substring descriptor value offset (String.length value - offset) with
      | 0 -> failf "short-capability-write"
      | count -> loop (offset + count)
      | exception Unix_error (EINTR, _, _) -> loop offset
  in
  loop 0

let read_control_line descriptor =
  let output = Buffer.create 512 in
  let byte = Bytes.create 1 in
  let rec read () =
    match Unix.read descriptor byte 0 1 with
    | 0 -> failf "execution-kernel-closed-before-response"
    | _ ->
        let character = Bytes.get byte 0 in
        if character = '\n' then Buffer.contents output
        else if Buffer.length output >= max_kernel_control_bytes then
          failf "execution-kernel-response-too-large"
        else (Buffer.add_char output character; read ())
    | exception Unix_error (EINTR, _, _) -> read ()
  in
  read ()

let required_environment name =
  match Sys.getenv_opt name with
  | Some value when value <> "" -> value
  | _ -> failf "execution-kernel-environment-missing:%s" name

let kernel_request operation arguments =
  let socket = required_environment "SOUNIO_LOOM_SOCKET" in
  let token_file = required_environment "SOUNIO_LOOM_TOKEN_FILE" in
  let token = trim (read_file ~limit:65536 token_file) in
  let request =
    String.concat "\t" ("LOOM/1" :: token :: operation :: arguments) ^ "\n"
  in
  if String.length request > max_kernel_control_bytes then
    failf "execution-kernel-request-too-large";
  let deadline = Unix.gettimeofday () +. kernel_connect_timeout_seconds in
  let rec connect () =
    let descriptor = Unix.socket PF_UNIX SOCK_STREAM 0 in
    Unix.set_close_on_exec descriptor;
    try
      Unix.connect descriptor (ADDR_UNIX socket);
      descriptor
    with
    | Unix_error ((ENOENT | ECONNREFUSED), _, _) when Unix.gettimeofday () < deadline ->
        Unix.close descriptor;
        Unix.sleepf 0.025;
        connect ()
    | error ->
        Unix.close descriptor;
        raise error
  in
  let descriptor = connect () in
  Fun.protect
    ~finally:(fun () -> try Unix.close descriptor with _ -> ())
    (fun () ->
      write_all descriptor request;
      match String.split_on_char '\t' (read_control_line descriptor) with
      | "ERR" :: reason :: _ -> failf "execution-kernel-refused:%s" reason
      | fields -> fields)

let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"

let getenv_test_only name =
  match Sys.getenv_opt name with
  | Some value when value <> "" && test_mode () -> Some value
  | Some value when value <> "" -> failf "%s-override-requires-test-mode" name
  | _ -> None

let safe_log value =
  String.map (function '\t' | '\n' | '\r' -> ' ' | character -> character) value

let language_name = function
  | 1 -> "Sounio"
  | 2 -> "Lean4"
  | 3 -> "Koka"
  | 4 -> "C++"
  | 5 -> "Haskell"
  | 6 -> "ExternalLLM"
  | 7 -> "Python"
  | 8 -> "Rust"
  | 9 -> "OCaml"
  | 10 -> "NativeTool"
  | 11 -> "Shell"
  | 12 -> "Git"
  | _ -> "Unclassified"

let language_role = function
  | 1 -> "SEMANTIC_AUTHORITY"
  | 2 -> "FORMAL_PARITY"
  | 3 -> "EFFECT_PARITY"
  | 4 -> "MATERIAL_PARITY"
  | 5 -> "OPTIONAL_DENOTATIONAL_BASELINE"
  | 6 -> "REVIEW_ONLY"
  | 7 | 8 -> "FORBIDDEN"
  | 9 -> "OPERATIONAL_REALIZATION"
  | 10 -> "NATIVE_MECHANICAL"
  | 11 | 12 -> "MECHANICAL_TRANSPORT"
  | _ -> "UNCLASSIFIED"

let utc_now () =
  let tm = Unix.gmtime (Unix.gettimeofday ()) in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02dZ"
    (tm.tm_year + 1900) (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min tm.tm_sec

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

let canonical_directory path =
  if not (Sys.file_exists path) then failf "execution-cwd-missing:%s" path;
  let physical = Unix.realpath path in
  if not (Sys.is_directory physical) then failf "execution-cwd-not-directory:%s" path;
  physical

let within root path =
  path = root
  || starts_with path (if root = "/" then "/" else root ^ "/")

let find_repo_root cwd =
  let rec loop candidate =
    let marker = Filename.concat candidate ".git" in
    if Sys.file_exists marker then Unix.realpath candidate
    else
      let parent = Filename.dirname candidate in
      if parent = candidate then failf "execution-outside-git-worktree:%s" cwd
      else loop parent
  in
  loop (canonical_directory cwd)

let first_line value =
  match String.split_on_char '\n' value with line :: _ -> trim line | [] -> ""

let git_common_dir root =
  let marker = Filename.concat root ".git" in
  if Sys.is_directory marker then Unix.realpath marker
  else
    let line = first_line (read_file ~limit:65536 marker) in
    if not (starts_with line "gitdir: ") then failf "invalid-gitdir-marker";
    let raw = String.sub line 8 (String.length line - 8) in
    let git_dir = Unix.realpath (normalize_absolute root raw) in
    let common_marker = Filename.concat git_dir "commondir" in
    if Sys.file_exists common_marker then
      let common = first_line (read_file ~limit:65536 common_marker) in
      Unix.realpath (normalize_absolute git_dir common)
    else git_dir

let require_secure_directory path =
  if not (Sys.file_exists path) then Unix.mkdir path 0o700;
  let info = Unix.lstat path in
  if info.st_kind <> S_DIR then failf "capability-state-not-directory:%s" path;
  if info.st_uid <> Unix.geteuid () then failf "capability-state-owner-mismatch:%s" path;
  if info.st_perm land 0o077 <> 0 then failf "capability-state-mode-insecure:%s" path;
  path

let capability_directory root =
  match getenv_test_only "SOUNIO_LOOM_EXECUTION_CAPABILITY_DIR" with
  | Some path -> require_secure_directory path
  | None ->
      require_secure_directory
        (Filename.concat (git_common_dir root) "sounio-loom-execution-capabilities")

let decision_log_path root =
  match getenv_test_only "SOUNIO_LOOM_EXECUTION_AUTHORITY_LOG" with
  | Some path -> path
  | None ->
      let directory =
        require_secure_directory
          (Filename.concat (git_common_dir root) "sounio-loom-execution-authority")
      in
      Filename.concat directory "capabilities.tsv"

let append_decision_log ~root ~phase ~decision ~reason ~token_sha256
    ~manifest_sha256 ~source_sha256 ~semantics_sha256 ~command_hex
    ~command_sha256 ~executable_hex ~executable_sha256 ~hardware_record_hex
    ~hardware_sha256 ~environment_sha256 ~language ~execution_class
    ~closure_attested ~frame_sha256 ~authority_result =
  let path = decision_log_path root in
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      let line =
        String.concat "\t"
          [ "schema=loom-execution-capability-decision-v1";
            "utc=" ^ utc_now ();
            "phase=" ^ phase;
            "decision=" ^ decision;
            "reason=" ^ safe_log reason;
            "token_sha256=" ^ token_sha256;
            "manifest_sha256=" ^ manifest_sha256;
            "sounio_source_sha256=" ^ source_sha256;
            "semantics_sha256=" ^ semantics_sha256;
            "producing_language=" ^ language_name language;
            "language_role=" ^ language_role language;
            "command_hex=" ^ command_hex;
            "command_sha256=" ^ command_sha256;
            "toolchain_path_hex=" ^ executable_hex;
            "executable_sha256=" ^ executable_sha256;
            "hardware_record_hex=" ^ hardware_record_hex;
            "hardware_sha256=" ^ hardware_sha256;
            "environment_sha256=" ^ environment_sha256;
            "language_id=" ^ string_of_int language;
            "execution_class=" ^ string_of_int execution_class;
            "closure_attested=" ^ string_of_int closure_attested;
            "frame_sha256=" ^ frame_sha256;
            "authority_result=" ^ safe_log authority_result;
            "execution_result=pending" ] ^ "\n"
      in
      write_all descriptor line;
      Unix.fsync descriptor;
      Unix.lockf descriptor F_ULOCK 0)

type process_result = { code : int; output : string }

let exit_code = function
  | WEXITED code -> code
  | WSIGNALED signal | WSTOPPED signal -> 128 + signal

let run_process ?(input = "") ?(environment = Unix.environment ()) ~cwd command arguments =
  let stdin_read, stdin_write = Unix.pipe () in
  let output_read, output_write = Unix.pipe () in
  Unix.set_close_on_exec stdin_write;
  Unix.set_close_on_exec output_read;
  let argv = Array.of_list (command :: arguments) in
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
        (try Unix.chdir cwd; Unix.execve command argv environment
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
      (try write_all stdin_write input with error -> kill_noerr (); raise error);
      Unix.close stdin_write;
      let deadline = Unix.gettimeofday () +. authority_timeout_seconds in
      let output = Buffer.create 4096 in
      let bytes = Bytes.create 16384 in
      let rec drain () =
        let remaining = deadline -. Unix.gettimeofday () in
        if remaining <= 0.0 then (kill_noerr (); failf "execution-authority-timeout");
        let ready, _, _ = Unix.select [ output_read ] [] [] remaining in
        if ready = [] then (kill_noerr (); failf "execution-authority-timeout")
        else
          match Unix.read output_read bytes 0 (Bytes.length bytes) with
          | 0 -> ()
          | count ->
              if Buffer.length output + count > max_file_bytes then (
                kill_noerr (); failf "execution-authority-output-too-large");
              Buffer.add_subbytes output bytes 0 count;
              drain ()
          | exception Unix_error (EINTR, _, _) -> drain ()
      in
      drain ();
      let _, status = Unix.waitpid [] pid in
      { code = exit_code status; output = Buffer.contents output })

let parse_manifest path =
  let table = Hashtbl.create 64 in
  read_file path |> String.split_on_char '\n'
  |> List.iter (fun line ->
         match String.index_opt line '=' with
         | None when line = "" -> ()
         | None -> failf "malformed-execution-authority-manifest"
         | Some index ->
             let key = String.sub line 0 index in
             if Hashtbl.mem table key then failf "duplicate-execution-authority-field:%s" key;
             Hashtbl.add table key
               (String.sub line (index + 1) (String.length line - index - 1)));
  table

let required table key =
  match Hashtbl.find_opt table key with
  | Some value when value <> "" -> value
  | _ -> failf "missing-execution-authority-field:%s" key

let digest_u32_of_hex digest =
  if String.length digest <> 64 then failf "invalid-sha256:%s" digest;
  List.init 8 (fun index ->
      let chunk = String.sub digest (index * 8) 8 in
      try Int64.to_string (Int64.of_string ("0x" ^ chunk))
      with _ -> failf "invalid-sha256:%s" digest)
  |> String.concat " "

let digest_u32_field table key =
  let parts = String.split_on_char ',' (required table key) in
  if List.length parts <> 8 then failf "invalid-execution-authority-digest:%s" key;
  List.iter
    (fun part ->
      try
        let value = Int64.of_string part in
        if value < 0L || value > 4294967295L then raise Exit
      with _ -> failf "invalid-execution-authority-digest:%s" key)
    parts;
  String.concat " " parts

type policy = {
  manifest_sha256 : string;
  runtime : string;
  source_sha256 : string;
  source_u32 : string;
  semantics_sha256 : string;
  semantics_u32 : string;
}

type outcome_policy = {
  outcome_manifest_sha256 : string;
  outcome_runtime : string;
  outcome_runtime_sha256 : string;
  outcome_source_sha256 : string;
  outcome_source_u32 : string;
  outcome_semantics_sha256 : string;
  outcome_semantics_u32 : string;
}

let execution_authority_runtime root manifest =
  let explicit = Sys.getenv_opt "SOUNIO_LOOM_EXECUTION_AUTHORITY_RUNTIME" in
  let sibling =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-execution-authority-runtime"
  in
  let local =
    Filename.concat root
      "tools/loom/.runtime/sounio-loom-execution-authority-runtime"
  in
  let selected =
    match explicit with
    | Some path when path <> "" -> path
    | _ when Sys.file_exists sibling -> sibling
    | _ -> local
  in
  if not (Sys.file_exists selected) then failf "execution-authority-runtime-missing:%s" selected;
  if sha256_file selected <> required manifest "executable_sha256" then
    failf "execution-authority-runtime-hash-mismatch";
  Unix.realpath selected

let load_policy root =
  let manifest_path =
    match getenv_test_only "SOUNIO_LOOM_EXECUTION_AUTHORITY_MANIFEST" with
    | Some path -> path
    | None -> Filename.concat root "tools/loom/execution_authority.freeze.v2"
  in
  if not (Sys.file_exists manifest_path) then failf "execution-authority-policy-missing";
  let manifest_sha256 = sha256_file manifest_path in
  if manifest_sha256 <> pinned_manifest_sha256 then
    failf "execution-authority-policy-hash-mismatch";
  let manifest = parse_manifest manifest_path in
  if required manifest "schema" <> "loom-execution-authority-freeze-v2"
     || required manifest "stage" <> "SEMANTICS_FROZEN"
     || required manifest "producing_language" <> "Sounio"
     || required manifest "language_role" <> "SEMANTIC_AUTHORITY"
     || required manifest "parity_open" <> "false"
     || required manifest "claim_ready" <> "false"
  then failf "execution-authority-policy-state-invalid";
  let source_path = Filename.concat root (required manifest "source_path") in
  let entrypoint_path = Filename.concat root (required manifest "entrypoint_path") in
  let source_sha256 = required manifest "source_sha256" in
  let semantics_sha256 = required manifest "semantics_sha256" in
  if sha256_file source_path <> source_sha256 then
    failf "execution-authority-source-hash-mismatch";
  if sha256_file entrypoint_path <> required manifest "entrypoint_sha256" then
    failf "execution-authority-entrypoint-hash-mismatch";
  if sha256 (read_file source_path ^ read_file entrypoint_path) <> semantics_sha256 then
    failf "execution-authority-semantics-hash-mismatch";
  { manifest_sha256;
    runtime = execution_authority_runtime root manifest;
    source_sha256;
    source_u32 = digest_u32_field manifest "source_sha256_u32";
    semantics_sha256;
    semantics_u32 = digest_u32_field manifest "semantics_sha256_u32" }

let execution_outcome_runtime root manifest =
  let explicit = Sys.getenv_opt "SOUNIO_LOOM_EXECUTION_OUTCOME_RUNTIME" in
  let sibling =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-execution-outcome-runtime"
  in
  let local =
    Filename.concat root
      "tools/loom/.runtime/sounio-loom-execution-outcome-runtime"
  in
  let selected =
    match explicit with
    | Some path when path <> "" -> path
    | _ when Sys.file_exists sibling -> sibling
    | _ -> local
  in
  if not (Sys.file_exists selected) then
    failf "execution-outcome-runtime-missing:%s" selected;
  let expected = required manifest "executable_sha256" in
  if sha256_file selected <> expected then
    failf "execution-outcome-runtime-hash-mismatch";
  (Unix.realpath selected, expected)

let load_outcome_policy root =
  let manifest_path =
    match getenv_test_only "SOUNIO_LOOM_EXECUTION_OUTCOME_MANIFEST" with
    | Some path -> path
    | None -> Filename.concat root "tools/loom/execution_outcome.freeze.v1"
  in
  if not (Sys.file_exists manifest_path) then
    failf "execution-outcome-policy-missing";
  let manifest_sha256 = sha256_file manifest_path in
  if manifest_sha256 <> pinned_outcome_manifest_sha256 then
    failf "execution-outcome-policy-hash-mismatch";
  let manifest = parse_manifest manifest_path in
  if required manifest "schema" <> "loom-execution-outcome-freeze-v1"
     || required manifest "stage" <> "SEMANTICS_FROZEN"
     || required manifest "producing_language" <> "Sounio"
     || required manifest "language_role" <> "SEMANTIC_AUTHORITY"
     || required manifest "action" <> "9022"
     || required manifest "parity_open" <> "false"
     || required manifest "claim_ready" <> "false"
  then failf "execution-outcome-policy-state-invalid";
  let source_path = Filename.concat root (required manifest "source_path") in
  let entrypoint_path = Filename.concat root (required manifest "entrypoint_path") in
  let parent_path =
    Filename.concat root (required manifest "parent_execution_authority_manifest")
  in
  let source_sha256 = required manifest "source_sha256" in
  let semantics_sha256 = required manifest "semantics_sha256" in
  if sha256_file source_path <> source_sha256 then
    failf "execution-outcome-source-hash-mismatch";
  if sha256_file entrypoint_path <> required manifest "entrypoint_sha256" then
    failf "execution-outcome-entrypoint-hash-mismatch";
  if sha256 (read_file source_path ^ read_file entrypoint_path) <> semantics_sha256 then
    failf "execution-outcome-semantics-hash-mismatch";
  if sha256_file parent_path <> required manifest "parent_execution_authority_manifest_sha256"
     || sha256_file parent_path <> pinned_manifest_sha256
  then failf "execution-outcome-parent-authority-mismatch";
  let runtime, runtime_sha256 = execution_outcome_runtime root manifest in
  { outcome_manifest_sha256 = manifest_sha256;
    outcome_runtime = runtime;
    outcome_runtime_sha256 = runtime_sha256;
    outcome_source_sha256 = source_sha256;
    outcome_source_u32 = digest_u32_field manifest "source_sha256_u32";
    outcome_semantics_sha256 = semantics_sha256;
    outcome_semantics_u32 = digest_u32_field manifest "semantics_sha256_u32" }

type quote_state = Plain | Single | Double

let shell_meta = function
  | '|' | '&' | ';' | '<' | '>' | '(' | ')' | '$' | '`'
  | '*' | '?' | '[' | ']' | '{' | '}' | '~' -> true
  | _ -> false

let lex_command command =
  if command = "" then raise (Dynamic_command "empty-command");
  if String.length command > max_command_bytes then
    raise (Dynamic_command "command-too-large");
  let words = ref [] in
  let word = Buffer.create 64 in
  let started = ref false in
  let push () =
    if !started then (
      words := Buffer.contents word :: !words;
      Buffer.clear word;
      started := false)
  in
  let length = String.length command in
  let rec loop index state =
    if index >= length then (
      if state <> Plain then raise (Dynamic_command "unterminated-quote");
      push ();
      let result = List.rev !words in
      if result = [] then raise (Dynamic_command "empty-command");
      if List.length result > max_arguments then raise (Dynamic_command "too-many-arguments");
      result)
    else
      let character = command.[index] in
      if character = '\000' || character = '\n' || character = '\r' then
        raise (Dynamic_command "control-character")
      else
        match state, character with
        | Plain, (' ' | '\t') -> push (); loop (index + 1) Plain
        | Plain, '\'' -> started := true; loop (index + 1) Single
        | Plain, '"' -> started := true; loop (index + 1) Double
        | Plain, '\\' ->
            if index + 1 >= length then raise (Dynamic_command "trailing-backslash");
            let escaped = command.[index + 1] in
            if escaped = '\n' || escaped = '\r' then
              raise (Dynamic_command "line-continuation");
            started := true;
            Buffer.add_char word escaped;
            loop (index + 2) Plain
        | Plain, value when shell_meta value ->
            raise (Dynamic_command "shell-metacharacter")
        | Plain, value ->
            started := true;
            Buffer.add_char word value;
            loop (index + 1) Plain
        | Single, '\'' -> loop (index + 1) Plain
        | Single, value ->
            Buffer.add_char word value;
            loop (index + 1) Single
        | Double, '"' -> loop (index + 1) Plain
        | Double, ('$' | '`') -> raise (Dynamic_command "double-quote-expansion")
        | Double, '\\' ->
            if index + 1 >= length then raise (Dynamic_command "trailing-backslash");
            let escaped = command.[index + 1] in
            if escaped = '\n' || escaped = '\r' then
              raise (Dynamic_command "line-continuation");
            Buffer.add_char word escaped;
            loop (index + 2) Double
        | Double, value ->
            Buffer.add_char word value;
            loop (index + 1) Double
  in
  loop 0 Plain

let path_entries () =
  Option.value ~default:"/usr/local/bin:/usr/bin:/bin" (Sys.getenv_opt "PATH")
  |> String.split_on_char ':'

let resolve_executable cwd command =
  let candidates =
    if String.contains command '/' then [ normalize_absolute cwd command ]
    else
      List.map
        (fun directory ->
          Filename.concat (if directory = "" then cwd else directory) command)
        (path_entries ())
  in
  let found =
    List.find_opt
      (fun path ->
        try
          let info = Unix.stat path in
          if info.st_kind = S_REG then (Unix.access path [ X_OK ]; true)
          else false
        with _ -> false)
      candidates
  in
  match found with
  | Some path -> Unix.realpath path
  | None -> failf "execution-command-not-found:%s" command

let basename_lower path = Filename.basename path |> String.lowercase_ascii

let one_of value values = List.mem value values

let classified_language executable =
  let name = basename_lower executable in
  if starts_with name "python" || starts_with name "pypy" then 7
  else if one_of name [ "cargo"; "rustc"; "rustup"; "rustfmt"; "clippy-driver" ] then 8
  else if one_of name [ "souc"; "madaros"; "sounio" ] || starts_with name "souc-" then 1
  else if one_of name [ "lean"; "lake"; "elan" ] then 2
  else if name = "koka" then 3
  else if one_of name [ "cc"; "c++"; "gcc"; "g++"; "clang"; "clang++" ] then 4
  else if one_of name [ "ghc"; "runghc"; "cabal"; "stack" ] then 5
  else if one_of name
      [ "claude"; "codex"; "cursor"; "grok"; "kimi"; "minimax";
        "gemini"; "ollama"; "vllm"; "zai" ]
  then 6
  else if starts_with name "ocaml" || one_of name [ "dune"; "opam" ]
          || starts_with name "sounio-loom"
  then 9
  else if one_of name [ "sh"; "bash"; "zsh"; "dash"; "fish"; "ksh" ] then 11
  else if name = "git" then 12
  else 10

let forwarding_wrapper name =
  one_of name
    [ "chrt"; "env"; "ionice"; "nice"; "nohup"; "setsid"; "stdbuf";
      "timeout"; "xargs" ]

let environment_assignment word =
  match String.index_opt word '=' with
  | Some index when index > 0 && word.[0] <> '-' -> true
  | _ -> false

let env_forwarded_command arguments =
  let rec loop = function
    | [] -> None
    | "--" :: command :: _ -> Some command
    | ("-u" | "--unset" | "-C" | "--chdir" | "-S" | "--split-string")
      :: _value :: tail -> loop tail
    | option :: tail when starts_with option "-" -> loop tail
    | assignment :: tail when environment_assignment assignment -> loop tail
    | command :: _ -> Some command
  in
  loop arguments

let read_prefix path limit =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let bytes = Bytes.create limit in
      let count = input channel bytes 0 limit in
      Bytes.sub_string bytes 0 count)

let shebang_language cwd executable =
  let prefix = read_prefix executable 4096 in
  if not (starts_with prefix "#!") then None
  else
    let line =
      match String.index_opt prefix '\n' with
      | Some index -> String.sub prefix 2 (index - 2)
      | None -> String.sub prefix 2 (String.length prefix - 2)
    in
    let words =
      String.split_on_char ' ' (trim line) |> List.filter (( <> ) "")
    in
    match words with
    | [] -> None
    | interpreter :: arguments ->
        let resolved = resolve_executable cwd interpreter in
        let language = classified_language resolved in
        if basename_lower resolved = "env" then
          Option.map
            (fun command -> resolve_executable cwd command |> classified_language)
            (env_forwarded_command arguments)
        else Some language

let file_starts_with_elf path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let bytes = Bytes.create 4 in
      input channel bytes 0 4 = 4
      && Bytes.get bytes 0 = '\127'
      && Bytes.get bytes 1 = 'E'
      && Bytes.get bytes 2 = 'L'
      && Bytes.get bytes 3 = 'F')

let audited_leaf executable =
  let canonical = Unix.realpath executable in
  let name = basename_lower canonical in
  let info = Unix.stat canonical in
  within "/usr/bin" canonical
  && one_of name [ "true"; "false"; "printf"; "pwd" ]
  && info.st_kind = S_REG
  && info.st_uid = 0
  && info.st_perm land 0o022 = 0
  && file_starts_with_elf canonical

let environment_fixed_names =
  [ "BASHOPTS"; "BASH_ENV"; "CDPATH"; "ENV"; "GCONV_PATH"; "HOME"; "IFS";
    "LANG"; "LC_ALL"; "LC_CTYPE"; "LOGNAME"; "LOCPATH"; "NLSPATH"; "PATH";
    "SHELL"; "SHELLOPTS"; "SOUC_BIN"; "SOUNIO_STDLIB_PATH"; "TERM"; "TMPDIR";
    "TZ"; "USER"; "ZDOTDIR" ]

let environment_prefixes =
  [ "CAML_"; "DYLD_"; "LC_"; "LD_"; "LOOM_"; "OCAML"; "SOUC_";
    "SOUNIO_"; "XDG_" ]

let dangerous_environment_names =
  [ "BASHOPTS"; "BASH_ENV"; "ENV"; "GCONV_PATH"; "LOCPATH"; "NLSPATH";
    "SHELLOPTS"; "ZDOTDIR" ]

let valid_environment_name name =
  let length = String.length name in
  let initial = function
    | 'A' .. 'Z' | 'a' .. 'z' | '_' -> true
    | _ -> false
  in
  let subsequent = function
    | 'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' | '_' -> true
    | _ -> false
  in
  length > 0 && initial name.[0]
  && (let rec loop index =
        index = length || (subsequent name.[index] && loop (index + 1))
      in
      loop 1)

let selected_environment_name name =
  valid_environment_name name
  && (List.mem name environment_fixed_names
      || List.exists (starts_with name) environment_prefixes)

let environment_bindings () =
  let table = Hashtbl.create 64 in
  Unix.environment ()
  |> Array.iter (fun item ->
         match String.index_opt item '=' with
         | None -> ()
         | Some index ->
             let name = String.sub item 0 index in
             if selected_environment_name name then (
               if Hashtbl.mem table name then
                 failf "duplicate-execution-environment:%s" name;
               Hashtbl.add table name
                 (String.sub item (index + 1) (String.length item - index - 1))));
  table

let environment_record_from table =
  let fixed = List.sort_uniq String.compare environment_fixed_names in
  let dynamic =
    Hashtbl.fold
      (fun name _ names -> if List.mem name fixed then names else name :: names)
      table []
    |> List.sort String.compare
  in
  let entry name =
    match Hashtbl.find_opt table name with
    | Some value -> name ^ "=hex:" ^ hex_encode value
    | None -> name ^ "=absent"
  in
  String.concat "\n"
    ("schema=loom-execution-environment-v1" :: List.map entry (fixed @ dynamic))
  ^ "\n"

let environment_record () = environment_bindings () |> environment_record_from

let environment_hash record = sha256 record

let environment_array_from table =
  Hashtbl.fold (fun name value values -> (name ^ "=" ^ value) :: values) table []
  |> List.sort String.compare |> Array.of_list

let ensure_safe_shell_bridge_environment table =
  let dangerous name =
    List.mem name dangerous_environment_names
    || starts_with name "LD_" || starts_with name "DYLD_"
  in
  Hashtbl.iter
    (fun name value ->
      if dangerous name && value <> "" then
        failf "unsafe-shell-bridge-environment:%s" name)
    table

let cpu_model () =
  if not (Sys.file_exists "/proc/cpuinfo") then "unavailable"
  else
    read_file "/proc/cpuinfo" |> String.split_on_char '\n'
    |> List.find_map (fun line ->
           match String.index_opt line ':' with
           | Some index ->
               let key = String.sub line 0 index |> trim in
               if key = "model name" || key = "Hardware" then
                 Some (String.sub line (index + 1) (String.length line - index - 1) |> trim)
               else None
           | None -> None)
    |> Option.value ~default:"unavailable"

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
      "hostname=" ^ Unix.gethostname ();
      "cpu_model=" ^ cpu_model () ] ^ "\n"

let hardware_hash record = sha256 record

type measurement = {
  command : string;
  command_sha256 : string;
  argv : string list;
  executable : string;
  executable_sha256 : string;
  language : int;
  purpose : int;
  surface : int;
  execution_class : int;
  closure_attested : int;
  classification_reason : string;
}

let measure_command cwd command =
  let command_sha256 = sha256 command in
  try
    let words = lex_command command in
    let raw_executable = List.hd words in
    let executable = resolve_executable cwd raw_executable in
    let direct_language = classified_language executable in
    let executable_name = basename_lower executable in
    let forwarded_language =
      if executable_name = "env" then
        env_forwarded_command (List.tl words)
        |> Option.map (fun target -> resolve_executable cwd target |> classified_language)
      else None
    in
    let script_language =
      if direct_language = 10 then shebang_language cwd executable else None
    in
    let language, execution_class, classification_reason =
      match forwarded_language with
      | Some (7 as language) | Some (8 as language) ->
          (language, 2, "forbidden-language-behind-env")
      | _ when forwarding_wrapper executable_name ->
          (11, 3, "exec-forwarding-wrapper-unattested:" ^ executable_name)
      | _ ->
          (match script_language with
          | Some (7 as language) | Some (8 as language) ->
              (language, 2, "forbidden-language-shebang")
          | Some 11 -> (11, 3, "shell-shebang-unattested")
          | Some language -> (language, 4, "interpreter-closure-unattested")
          | None ->
              let execution_class =
                if direct_language = 11 then 3
                else if direct_language = 7 then 2
                else 4
              in
              (direct_language, execution_class, "closure-unattested"))
    in
    let purpose = if language >= 2 && language <= 5 then 3 else if language = 6 then 4 else 1 in
    let surface = if language = 12 && List.nth_opt words 1 = Some "commit" then 2 else 1 in
    let closure_attested =
      if forwarding_wrapper executable_name || script_language <> None then 0
      else if audited_leaf executable then 1
      else 0
    in
    { command;
      command_sha256;
      argv = executable :: List.tl words;
      executable;
      executable_sha256 = sha256_file executable;
      language;
      purpose;
      surface;
      execution_class;
      closure_attested;
      classification_reason =
        if closure_attested = 1 then "audited-leaf" else classification_reason }
  with
  | Dynamic_command reason ->
      { command;
        command_sha256;
        argv = [];
        executable = "";
        executable_sha256 = sha256 ("dynamic:" ^ reason);
        language = 11;
        purpose = 1;
        surface = 1;
        execution_class = 3;
        closure_attested = 0;
        classification_reason = reason }
  | Error reason when starts_with reason "execution-command-not-found:" ->
      { command;
        command_sha256;
        argv = [];
        executable = "";
        executable_sha256 = sha256 reason;
        language = 13;
        purpose = 1;
        surface = 1;
        execution_class = 5;
        closure_attested = 0;
        classification_reason = reason }

let authority_frame policy measurement hardware_sha256 =
  let zero = "0 0 0 0 0 0 0 0" in
  let toolchain = digest_u32_of_hex measurement.executable_sha256 in
  let hardware = digest_u32_of_hex hardware_sha256 in
  let command = digest_u32_of_hex measurement.command_sha256 in
  String.concat " "
    [ "9021"; "3"; string_of_int measurement.surface;
      string_of_int measurement.execution_class;
      string_of_int measurement.language; string_of_int measurement.purpose;
      "1"; "1"; string_of_int measurement.closure_attested;
      "0"; "0"; "0"; "0"; "0"; "0"; "0"; "0"; "0"; "0";
      policy.source_u32; policy.semantics_u32; policy.semantics_u32;
      toolchain; hardware; command; zero; zero ] ^ "\n"

let invoke_authority root policy frame environment =
  let result = run_process ~input:frame ~environment ~cwd:root policy.runtime [] in
  let decision = trim result.output in
  if result.code <> 0
     || not (starts_with decision "SOUNIO_EXECUTION_AUTHORITY_ALLOW code=0 ")
     || not (ends_with decision "stage=SEMANTICS_FROZEN")
  then raise (Authority_denied (result.code, decision));
  decision

let execution_outcome_frame policy ~outcome_kind ~exit_code ~signal
    ~elapsed_us ~hardware_sha256 ~command_sha256 ~environment_sha256
    ~executable_sha256 ~grant_sha256 ~generation_sha256
    ~issue_decision_sha256 ~consume_decision_sha256 ~result_sha256 =
  String.concat " "
    [ "9022"; "3"; "1"; string_of_int outcome_kind;
      string_of_int exit_code; string_of_int signal; Int64.to_string elapsed_us;
      "1"; "1"; "1"; "1"; "1";
      policy.outcome_source_u32; policy.outcome_semantics_u32;
      policy.outcome_semantics_u32;
      digest_u32_of_hex policy.outcome_runtime_sha256;
      digest_u32_of_hex hardware_sha256;
      digest_u32_of_hex command_sha256;
      digest_u32_of_hex environment_sha256;
      digest_u32_of_hex executable_sha256;
      digest_u32_of_hex grant_sha256;
      digest_u32_of_hex generation_sha256;
      digest_u32_of_hex issue_decision_sha256;
      digest_u32_of_hex consume_decision_sha256;
      digest_u32_of_hex result_sha256 ] ^ "\n"

let invoke_outcome_authority root policy frame environment =
  let result =
    run_process ~input:frame ~environment ~cwd:root policy.outcome_runtime []
  in
  let decision = trim result.output in
  if result.code <> 0
     || not (starts_with decision "SOUNIO_EXECUTION_OUTCOME_ALLOW code=0 ")
     || not (ends_with decision "stage=SEMANTICS_FROZEN")
  then raise (Authority_denied (result.code, decision));
  decision

let random_token () =
  let descriptor = Unix.openfile "/dev/urandom" [ O_RDONLY ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      let bytes = Bytes.create 32 in
      let rec read offset =
        if offset < Bytes.length bytes then
          match Unix.read descriptor bytes offset (Bytes.length bytes - offset) with
          | 0 -> failf "random-source-short-read"
          | count -> read (offset + count)
          | exception Unix_error (EINTR, _, _) -> read offset
      in
      read 0;
      hex_encode (Bytes.to_string bytes))

let capability_ttl_seconds () =
  match getenv_test_only "SOUNIO_LOOM_EXECUTION_CAPABILITY_TTL_SECONDS" with
  | None -> default_capability_ttl_seconds
  | Some raw ->
      let value = try int_of_string raw with _ -> failf "invalid-capability-ttl" in
      if value < 1 || value > 120 then failf "invalid-capability-ttl";
      value

let current_time_us () = Int64.of_float (Unix.gettimeofday () *. 1_000_000.0)

let fsync_directory path =
  let descriptor = Unix.openfile path [ O_RDONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close descriptor)
    (fun () -> Unix.fsync descriptor)

let execution_outcome_directory root =
  let authority =
    require_secure_directory
      (Filename.concat (git_common_dir root) "sounio-loom-execution-authority")
  in
  require_secure_directory (Filename.concat authority "outcomes")

let write_durable_outcome_record directory label digest content =
  let final_path = Filename.concat directory (label ^ "-" ^ digest ^ ".receipt") in
  if Sys.file_exists final_path then (
    if read_file final_path <> content then
      failf "execution-outcome-record-collision:%s" final_path;
    final_path)
  else
    let temporary =
      Filename.concat directory
        (Printf.sprintf ".%s.%d.%s" label (Unix.getpid ()) (random_token ()))
    in
    let descriptor = Unix.openfile temporary [ O_WRONLY; O_CREAT; O_EXCL ] 0o600 in
    Fun.protect
      ~finally:(fun () ->
        (try Unix.close descriptor with _ -> ());
        if Sys.file_exists temporary then (try Unix.unlink temporary with _ -> ()))
      (fun () ->
        Unix.fchmod descriptor 0o600;
        write_all descriptor content;
        Unix.fsync descriptor;
        Unix.close descriptor;
        Unix.rename temporary final_path;
        fsync_directory directory);
    final_path

type observed_child_outcome = {
  observed_kind : int;
  observed_exit_code : int;
  observed_signal : int;
  observed_elapsed_us : int64;
}

let linux_signal_number signal =
  let known =
    [ Sys.sighup, 1; Sys.sigint, 2; Sys.sigquit, 3; Sys.sigill, 4;
      Sys.sigtrap, 5; Sys.sigabrt, 6; Sys.sigbus, 7; Sys.sigfpe, 8;
      Sys.sigkill, 9; Sys.sigusr1, 10; Sys.sigsegv, 11; Sys.sigusr2, 12;
      Sys.sigpipe, 13; Sys.sigalrm, 14; Sys.sigterm, 15; Sys.sigchld, 17;
      Sys.sigcont, 18; Sys.sigstop, 19; Sys.sigtstp, 20; Sys.sigttin, 21;
      Sys.sigttou, 22; Sys.sigurg, 23; Sys.sigxcpu, 24; Sys.sigxfsz, 25;
      Sys.sigvtalrm, 26; Sys.sigprof, 27; Sys.sigpoll, 29; Sys.sigsys, 31 ]
  in
  match List.find_opt (fun (portable, _) -> signal = portable) known with
  | Some (_, native) -> native
  | None when signal > 0 && signal <= 255 -> signal
  | None -> failf "unsupported-portable-signal:%d" signal

let supervise_child executable argv environment =
  let forced_signal =
    match getenv_test_only "SOUNIO_LOOM_EXECUTION_OUTCOME_CHILD_SIGNAL" with
    | None -> None
    | Some raw ->
        let signal =
          try int_of_string raw
          with _ -> failf "invalid-execution-outcome-test-signal"
        in
        if signal < 1 || signal > 64 then
          failf "invalid-execution-outcome-test-signal";
        Some signal
  in
  let started_us = current_time_us () in
  let pid =
    match Unix.fork () with
    | 0 ->
        (try Unix.execve executable (Array.of_list argv) environment
         with _ -> Unix._exit 127)
    | pid -> pid
  in
  let forwarded = [ Sys.sigint; Sys.sigterm; Sys.sighup; Sys.sigquit ] in
  let previous =
    List.map
      (fun signal ->
        let behavior =
          Sys.signal signal
            (Sys.Signal_handle
               (fun received -> try Unix.kill pid received with _ -> ()))
        in
        (signal, behavior))
      forwarded
  in
  Option.iter (fun signal -> Unix.kill pid signal) forced_signal;
  let restore () =
    List.iter (fun (signal, behavior) -> Sys.set_signal signal behavior) previous
  in
  Fun.protect ~finally:restore (fun () ->
      let rec wait () =
        match Unix.waitpid [] pid with
        | _, status -> status
        | exception Unix_error (EINTR, _, _) -> wait ()
      in
      let status = wait () in
      let elapsed = Int64.sub (current_time_us ()) started_us in
      let elapsed = if elapsed < 0L then 0L else elapsed in
      match status with
      | WEXITED code ->
          { observed_kind = 1; observed_exit_code = code; observed_signal = 0;
            observed_elapsed_us = elapsed }
      | WSIGNALED signal ->
          { observed_kind = 2; observed_exit_code = 0;
            observed_signal = linux_signal_number signal;
            observed_elapsed_us = elapsed }
      | WSTOPPED signal ->
          { observed_kind = 3; observed_exit_code = 0;
            observed_signal = linux_signal_number signal;
            observed_elapsed_us = elapsed })

type kernel_outcome_context = {
  kernel_instance : string;
  kernel_generation : string;
  kernel_handle : string;
  kernel_grant_sha256 : string;
}

let kernel_record_outcome context receipt receipt_sha256 =
  match
    kernel_request "EXEC_OUTCOME"
      [ context.kernel_instance; context.kernel_generation;
        context.kernel_handle; receipt_sha256; hex_encode receipt ]
  with
  | [ "OK"; "EXEC_OUTCOME_RECORDED"; actual_instance; actual_generation;
      actual_receipt_sha256 ]
    when actual_instance = context.kernel_instance
         && actual_generation = context.kernel_generation
         && actual_receipt_sha256 = receipt_sha256 -> ()
  | _ -> failf "execution-kernel-invalid-outcome-response"

let append_outcome_commit_log ~root ~context ~outcome ~result_sha256
    ~receipt_sha256 ~decision_sha256 =
  let path = decision_log_path root in
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      let line =
        String.concat "\t"
          [ "schema=loom-execution-outcome-commit-v1";
            "utc=" ^ utc_now ();
            "decision=ALLOW";
            "kernel_generation=" ^ context.kernel_generation;
            "grant_sha256=" ^ context.kernel_grant_sha256;
            "handle_sha256=" ^ sha256 context.kernel_handle;
            "outcome_kind=" ^ string_of_int outcome.observed_kind;
            "exit_code=" ^ string_of_int outcome.observed_exit_code;
            "signal=" ^ string_of_int outcome.observed_signal;
            "elapsed_us=" ^ Int64.to_string outcome.observed_elapsed_us;
            "result_sha256=" ^ result_sha256;
            "receipt_sha256=" ^ receipt_sha256;
            "outcome_decision_sha256=" ^ decision_sha256;
            "execution_result=recorded" ] ^ "\n"
      in
      write_all descriptor line;
      Unix.fsync descriptor;
      Unix.lockf descriptor F_ULOCK 0)

let outcome_test_pause () =
  match getenv_test_only "SOUNIO_LOOM_EXECUTION_OUTCOME_PAUSE_SECONDS" with
  | None -> ()
  | Some raw ->
      let seconds =
        try float_of_string raw
        with _ -> failf "invalid-execution-outcome-test-pause"
      in
      if seconds < 0.0 || seconds > 10.0 then
        failf "invalid-execution-outcome-test-pause";
      Unix.sleepf seconds

let outcome_test_replay context receipt receipt_sha256 =
  match getenv_test_only "SOUNIO_LOOM_EXECUTION_OUTCOME_REPLAY" with
  | None -> ()
  | Some "1" ->
      (try
         kernel_record_outcome context receipt receipt_sha256;
         failf "execution-outcome-replay-was-accepted"
       with
       | Error reason
         when starts_with reason
                "execution-kernel-refused:exec-outcome-missing-or-replayed" -> ())
  | Some _ -> failf "invalid-execution-outcome-test-replay"

let write_capability directory token body =
  let final_path = Filename.concat directory (token ^ ".cap") in
  let temporary =
    Filename.concat directory
      (Printf.sprintf ".%s.tmp.%d.%s" token (Unix.getpid ()) (random_token ()))
  in
  let descriptor = Unix.openfile temporary [ O_WRONLY; O_CREAT; O_EXCL ] 0o600 in
  Fun.protect
    ~finally:(fun () ->
      (try Unix.close descriptor with _ -> ());
      if Sys.file_exists temporary then (try Unix.unlink temporary with _ -> ()))
    (fun () ->
      Unix.fchmod descriptor 0o600;
      write_all descriptor body;
      Unix.fsync descriptor;
      Unix.close descriptor;
      Unix.rename temporary final_path;
      fsync_directory directory);
  final_path

let capability_body ~root ~cwd ~token ~policy ~measurement ~hardware_record
    ~hardware_sha256 ~environment_record ~environment_sha256 ~frame ~decision =
  let issued_us = current_time_us () in
  let expires_us =
    Int64.add issued_us
      (Int64.mul (Int64.of_int (capability_ttl_seconds ())) 1_000_000L)
  in
  let fixed =
    [ "schema=loom-execution-capability-v1";
      "token=" ^ token;
      "issued_us=" ^ Int64.to_string issued_us;
      "expires_us=" ^ Int64.to_string expires_us;
      "uid=" ^ string_of_int (Unix.geteuid ());
      "root_hex=" ^ hex_encode root;
      "cwd_hex=" ^ hex_encode cwd;
      "command_hex=" ^ hex_encode measurement.command;
      "command_sha256=" ^ measurement.command_sha256;
      "environment_record_hex=" ^ hex_encode environment_record;
      "environment_sha256=" ^ environment_sha256;
      "executable_hex=" ^ hex_encode measurement.executable;
      "executable_sha256=" ^ measurement.executable_sha256;
      "broker_sha256=" ^ sha256_file (Unix.realpath Sys.executable_name);
      "manifest_sha256=" ^ policy.manifest_sha256;
      "source_sha256=" ^ policy.source_sha256;
      "semantics_sha256=" ^ policy.semantics_sha256;
      "hardware_record_hex=" ^ hex_encode hardware_record;
      "hardware_sha256=" ^ hardware_sha256;
      "producing_language=" ^ language_name measurement.language;
      "language_role=" ^ language_role measurement.language;
      "language=" ^ string_of_int measurement.language;
      "purpose=" ^ string_of_int measurement.purpose;
      "surface=" ^ string_of_int measurement.surface;
      "execution_class=" ^ string_of_int measurement.execution_class;
      "closure_attested=" ^ string_of_int measurement.closure_attested;
      "argv_count=" ^ string_of_int (List.length measurement.argv) ]
  in
  let arguments =
    List.mapi (fun index argument -> Printf.sprintf "arg_%d_hex=%s" index (hex_encode argument))
      measurement.argv
  in
  let tail =
    [ "frame_hex=" ^ hex_encode frame;
      "decision_hex=" ^ hex_encode decision ]
  in
  let body = String.concat "\n" (fixed @ arguments @ tail) ^ "\n" in
  body ^ "record_sha256=" ^ sha256 body ^ "\n"

let execution_observation_body ~root ~cwd ~token ~preexec_policy
    ~outcome_policy ~measurement ~hardware_sha256 ~environment_sha256
    ~context ~issue_decision ~consume_decision ~outcome =
  String.concat "\n"
    [ "schema=loom-execution-observation-v1";
      "observed_utc=" ^ utc_now ();
      "broker_pid=" ^ string_of_int (Unix.getpid ());
      "root_hex=" ^ hex_encode root;
      "cwd_hex=" ^ hex_encode cwd;
      "token_sha256=" ^ sha256 token;
      "preexec_manifest_sha256=" ^ preexec_policy.manifest_sha256;
      "preexec_source_sha256=" ^ preexec_policy.source_sha256;
      "preexec_semantics_sha256=" ^ preexec_policy.semantics_sha256;
      "outcome_manifest_sha256=" ^ outcome_policy.outcome_manifest_sha256;
      "outcome_source_sha256=" ^ outcome_policy.outcome_source_sha256;
      "outcome_semantics_sha256=" ^ outcome_policy.outcome_semantics_sha256;
      "outcome_runtime_sha256=" ^ outcome_policy.outcome_runtime_sha256;
      "command_hex=" ^ hex_encode measurement.command;
      "command_sha256=" ^ measurement.command_sha256;
      "environment_sha256=" ^ environment_sha256;
      "executable_hex=" ^ hex_encode measurement.executable;
      "executable_sha256=" ^ measurement.executable_sha256;
      "hardware_sha256=" ^ hardware_sha256;
      "kernel_instance=" ^ context.kernel_instance;
      "kernel_generation=" ^ context.kernel_generation;
      "kernel_generation_sha256=" ^ sha256 context.kernel_generation;
      "handle_sha256=" ^ sha256 context.kernel_handle;
      "grant_sha256=" ^ context.kernel_grant_sha256;
      "issue_decision_sha256=" ^ sha256 issue_decision;
      "consume_decision_sha256=" ^ sha256 consume_decision;
      "outcome_kind=" ^ string_of_int outcome.observed_kind;
      "exit_code=" ^ string_of_int outcome.observed_exit_code;
      "signal=" ^ string_of_int outcome.observed_signal;
      "elapsed_us=" ^ Int64.to_string outcome.observed_elapsed_us ] ^ "\n"

let execution_outcome_receipt observation result_sha256 outcome_policy
    outcome_frame outcome_decision =
  let body =
    observation ^
    String.concat "\n"
      [ "result_sha256=" ^ result_sha256;
        "outcome_frame_sha256=" ^ sha256 outcome_frame;
        "outcome_authority_manifest_sha256=" ^
          outcome_policy.outcome_manifest_sha256;
        "outcome_authority_source_sha256=" ^ outcome_policy.outcome_source_sha256;
        "outcome_authority_semantics_sha256=" ^
          outcome_policy.outcome_semantics_sha256;
        "outcome_authority_decision_hex=" ^ hex_encode outcome_decision;
        "outcome_authority_decision_sha256=" ^ sha256 outcome_decision ] ^ "\n"
  in
  body ^ "record_sha256=" ^ sha256 body ^ "\n"

let shell_quote value =
  "'" ^ String.concat "'\"'\"'" (String.split_on_char '\'' value) ^ "'"

let broker_command token =
  shell_quote (Unix.realpath Sys.executable_name)
  ^ " exec-capability --test-file-capability-fixture --token " ^ token

let file_capability_fixture_requested requested =
  if not requested then false
  else if not (test_mode ()) then
    failf "file-capability-fixture-requires-test-mode"
  else
    match Sys.getenv_opt "SOUNIO_LOOM_EXECUTION_CAPABILITY_DIR" with
    | Some value when value <> "" -> true
    | _ -> failf "file-capability-fixture-directory-missing"

let kernel_issue_capability ~cwd ~payload =
  let instance = required_environment "SOUNIO_LOOM_INSTANCE_ID" in
  let payload_sha256 = sha256 payload in
  let ttl = capability_ttl_seconds () in
  match
    kernel_request "EXEC_ISSUE"
      [ instance; hex_encode cwd; string_of_int ttl; payload_sha256;
        hex_encode payload ]
  with
  | [ "OK"; "EXEC_ISSUED"; actual_instance; generation; handle;
      expires_us; actual_payload_sha256 ]
    when actual_instance = instance && actual_payload_sha256 = payload_sha256 ->
      ignore
        (try Int64.of_string expires_us
         with _ -> failf "execution-kernel-invalid-expiry");
      if String.length generation <> 64 || String.length handle <> 64 then
        failf "execution-kernel-invalid-grant-identity";
      (instance, generation, handle)
  | _ -> failf "execution-kernel-invalid-issue-response"

let broker_command_kernel instance generation handle =
  String.concat " "
    [ shell_quote (Unix.realpath Sys.executable_name); "exec-capability";
      "--instance"; instance; "--generation"; generation; "--handle"; handle ]

let authorize_and_issue ~file_capability_fixture ~root ~cwd ~command =
  let root = Unix.realpath root in
  let cwd = canonical_directory cwd in
  if not (within root cwd) then failf "execution-cwd-outside-worktree:%s" cwd;
  let token_sha256 = ref "-" in
  let command_hex = hex_encode command in
  let command_sha256 = ref (sha256 command) in
  let manifest_sha256 = ref "-" in
  let source_sha256 = ref "-" in
  let semantics_sha256 = ref "-" in
  let executable_hex = ref "-" in
  let executable_sha256 = ref "-" in
  let hardware_record_value = hardware_record () in
  let hardware_record_hex = hex_encode hardware_record_value in
  let hardware_sha256_value = hardware_hash hardware_record_value in
  let environment_sha256_value = ref "-" in
  let language = ref 13 in
  let execution_class = ref 5 in
  let closure_attested = ref 0 in
  let frame_sha256 = ref "-" in
  let authority_result = ref "unavailable" in
  let append ~phase ~decision ~reason =
    append_decision_log ~root ~phase ~decision ~reason
      ~token_sha256:!token_sha256 ~manifest_sha256:!manifest_sha256
      ~source_sha256:!source_sha256 ~semantics_sha256:!semantics_sha256
      ~command_hex ~command_sha256:!command_sha256
      ~executable_hex:!executable_hex ~executable_sha256:!executable_sha256
      ~hardware_record_hex ~hardware_sha256:hardware_sha256_value
      ~environment_sha256:!environment_sha256_value
      ~language:!language ~execution_class:!execution_class
      ~closure_attested:!closure_attested ~frame_sha256:!frame_sha256
      ~authority_result:!authority_result
  in
  try
    let environment_bindings_value = environment_bindings () in
    ensure_safe_shell_bridge_environment environment_bindings_value;
    let environment_record_value = environment_record_from environment_bindings_value in
    let execution_environment = environment_array_from environment_bindings_value in
    environment_sha256_value := environment_hash environment_record_value;
    let policy = load_policy root in
    manifest_sha256 := policy.manifest_sha256;
    source_sha256 := policy.source_sha256;
    semantics_sha256 := policy.semantics_sha256;
    let measurement = measure_command cwd command in
    command_sha256 := measurement.command_sha256;
    executable_hex := hex_encode measurement.executable;
    executable_sha256 := measurement.executable_sha256;
    language := measurement.language;
    execution_class := measurement.execution_class;
    closure_attested := measurement.closure_attested;
    let frame = authority_frame policy measurement hardware_sha256_value in
    frame_sha256 := sha256 frame;
    let decision =
      try invoke_authority root policy frame execution_environment with
      | Authority_denied (code, decision) ->
          authority_result := decision;
          failf "execution-authority-denied:rc=%d:%s" code decision
    in
    authority_result := decision;
    if measurement.argv = [] || measurement.executable = "" then
      failf "execution-authority-allowed-empty-measurement";
    let token = random_token () in
    token_sha256 := sha256 token;
    let body =
      capability_body ~root ~cwd ~token ~policy ~measurement
        ~hardware_record:hardware_record_value
        ~hardware_sha256:hardware_sha256_value
        ~environment_record:environment_record_value
        ~environment_sha256:!environment_sha256_value ~frame ~decision
    in
    let replacement =
      if file_capability_fixture_requested file_capability_fixture then (
        let directory = capability_directory root in
        ignore (write_capability directory token body);
        broker_command token)
      else
        let instance, generation, handle =
          kernel_issue_capability ~cwd ~payload:body
        in
        broker_command_kernel instance generation handle
    in
    append ~phase:"ISSUE" ~decision:"ALLOW"
      ~reason:measurement.classification_reason;
    replacement
  with
  | Error reason as error ->
      (try append ~phase:"ISSUE" ~decision:"DENY" ~reason with _ -> ());
      raise error
  | Sys_error reason as error ->
      (try append ~phase:"ISSUE" ~decision:"DENY" ~reason with _ -> ());
      raise error
  | Unix_error (unix_error, function_name, argument) as error ->
      let reason =
        Printf.sprintf "%s:%s(%s)"
          (Unix.error_message unix_error) function_name argument
      in
      (try append ~phase:"ISSUE" ~decision:"DENY" ~reason with _ -> ());
      raise error

let parse_capability content =
  let lines = String.split_on_char '\n' content in
  let meaningful = List.filter (( <> ) "") lines in
  let reversed = List.rev meaningful in
  let digest_line, body_lines =
    match reversed with
    | digest :: rest when starts_with digest "record_sha256=" -> (digest, List.rev rest)
    | _ -> failf "capability-record-digest-missing"
  in
  let body = String.concat "\n" body_lines ^ "\n" in
  let expected = String.sub digest_line 14 (String.length digest_line - 14) in
  if sha256 body <> expected then failf "capability-record-digest-mismatch";
  let table = Hashtbl.create 48 in
  List.iter
    (fun line ->
      match String.index_opt line '=' with
      | None -> failf "malformed-capability-record"
      | Some index ->
          let key = String.sub line 0 index in
          if Hashtbl.mem table key then failf "duplicate-capability-field:%s" key;
          Hashtbl.add table key
            (String.sub line (index + 1) (String.length line - index - 1)))
    body_lines;
  table

let capability_required table key =
  match Hashtbl.find_opt table key with
  | Some value -> value
  | None -> failf "missing-capability-field:%s" key

let capability_int table key =
  try int_of_string (capability_required table key)
  with _ -> failf "invalid-capability-integer:%s" key

let capability_int64 table key =
  try Int64.of_string (capability_required table key)
  with _ -> failf "invalid-capability-integer:%s" key

let validate_token token =
  if String.length token <> 64 then failf "invalid-capability-token";
  String.iter
    (function '0' .. '9' | 'a' .. 'f' -> () | _ -> failf "invalid-capability-token")
    token

let execute_capability_content ~root ~token ~content ~burn ~cleanup
    ~outcome_context =
  validate_token token;
  let root = Unix.realpath root in
  let command_hex = ref "" in
  let command_sha256 = ref "-" in
  let manifest_sha256 = ref "-" in
  let source_sha256 = ref "-" in
  let semantics_sha256 = ref "-" in
  let executable_hex = ref "-" in
  let executable_sha256 = ref "-" in
  let hardware_record_value = hardware_record () in
  let hardware_record_hex = hex_encode hardware_record_value in
  let hardware_sha256_value = hardware_hash hardware_record_value in
  let environment_record_value = ref "" in
  let environment_sha256_value = ref "-" in
  let language = ref 13 in
  let execution_class = ref 5 in
  let closure_attested = ref 0 in
  let frame_sha256 = ref "-" in
  let authority_result = ref "unavailable" in
  let token_sha256 = sha256 token in
  let append ~decision ~reason =
    append_decision_log ~root ~phase:"CONSUME" ~decision ~reason
      ~token_sha256 ~manifest_sha256:!manifest_sha256
      ~source_sha256:!source_sha256 ~semantics_sha256:!semantics_sha256
      ~command_hex:!command_hex ~command_sha256:!command_sha256
      ~executable_hex:!executable_hex ~executable_sha256:!executable_sha256
      ~hardware_record_hex ~hardware_sha256:hardware_sha256_value
      ~environment_sha256:!environment_sha256_value
      ~language:!language ~execution_class:!execution_class
      ~closure_attested:!closure_attested ~frame_sha256:!frame_sha256
      ~authority_result:!authority_result
  in
  try
    let environment_bindings_value = environment_bindings () in
    ensure_safe_shell_bridge_environment environment_bindings_value;
    environment_record_value := environment_record_from environment_bindings_value;
    let execution_environment = environment_array_from environment_bindings_value in
    environment_sha256_value := environment_hash !environment_record_value;
    let table = parse_capability content in
    if capability_required table "schema" <> "loom-execution-capability-v1" then
      failf "capability-schema-mismatch";
    if capability_required table "token" <> token then failf "capability-token-mismatch";
    if capability_int table "uid" <> Unix.geteuid () then failf "capability-uid-mismatch";
    if current_time_us () > capability_int64 table "expires_us" then
      failf "capability-expired";
    let recorded_root = hex_decode "root" (capability_required table "root_hex") in
    let recorded_cwd = hex_decode "cwd" (capability_required table "cwd_hex") in
    if recorded_root <> root then failf "capability-root-mismatch";
    if Unix.realpath (Unix.getcwd ()) <> recorded_cwd then failf "capability-cwd-mismatch";
    let recorded_environment =
      hex_decode "environment_record"
        (capability_required table "environment_record_hex")
    in
    if recorded_environment <> !environment_record_value
       || capability_required table "environment_sha256"
          <> !environment_sha256_value
       || environment_hash recorded_environment <> !environment_sha256_value
    then
      failf "capability-environment-mismatch";
    let recorded_hardware =
      hex_decode "hardware_record" (capability_required table "hardware_record_hex")
    in
    if recorded_hardware <> hardware_record_value
       || capability_required table "hardware_sha256" <> hardware_sha256_value
       || sha256 recorded_hardware <> hardware_sha256_value
    then failf "capability-hardware-mismatch";
    command_hex := capability_required table "command_hex";
    let command = hex_decode "command" !command_hex in
    command_sha256 := sha256 command;
    if !command_sha256 <> capability_required table "command_sha256" then
      failf "capability-command-hash-mismatch";
    let policy = load_policy root in
    manifest_sha256 := policy.manifest_sha256;
    source_sha256 := policy.source_sha256;
    semantics_sha256 := policy.semantics_sha256;
    if policy.manifest_sha256 <> capability_required table "manifest_sha256"
       || policy.source_sha256 <> capability_required table "source_sha256"
       || policy.semantics_sha256 <> capability_required table "semantics_sha256"
    then failf "capability-authority-chain-mismatch";
    let broker_sha256 = sha256_file (Unix.realpath Sys.executable_name) in
    if broker_sha256 <> capability_required table "broker_sha256" then
      failf "capability-broker-hash-mismatch";
    let measurement = measure_command recorded_cwd command in
    executable_hex := hex_encode measurement.executable;
    executable_sha256 := measurement.executable_sha256;
    language := measurement.language;
    execution_class := measurement.execution_class;
    closure_attested := measurement.closure_attested;
    if measurement.executable <> hex_decode "executable" (capability_required table "executable_hex")
       || measurement.executable_sha256 <> capability_required table "executable_sha256"
       || measurement.language <> capability_int table "language"
       || measurement.purpose <> capability_int table "purpose"
       || measurement.surface <> capability_int table "surface"
       || measurement.execution_class <> capability_int table "execution_class"
       || measurement.closure_attested <> capability_int table "closure_attested"
       || language_name measurement.language
          <> capability_required table "producing_language"
       || language_role measurement.language <> capability_required table "language_role"
    then failf "capability-measurement-drift";
    let count = capability_int table "argv_count" in
    if count <> List.length measurement.argv then failf "capability-argv-count-mismatch";
    List.iteri
      (fun index argument ->
        let recorded =
          hex_decode (Printf.sprintf "arg_%d" index)
            (capability_required table (Printf.sprintf "arg_%d_hex" index))
        in
        if recorded <> argument then failf "capability-argv-mismatch:%d" index)
      measurement.argv;
    let frame = authority_frame policy measurement hardware_sha256_value in
    frame_sha256 := sha256 frame;
    if frame <> hex_decode "frame" (capability_required table "frame_hex") then
      failf "capability-frame-drift";
    let decision =
      try invoke_authority root policy frame execution_environment with
      | Authority_denied (code, decision) ->
          authority_result := decision;
          failf "execution-authority-denied:rc=%d:%s" code decision
    in
    authority_result := decision;
    if decision <> hex_decode "decision" (capability_required table "decision_hex") then
      failf "capability-decision-drift";
    (match outcome_context with
    | None ->
        burn ();
        append ~decision:"ALLOW" ~reason:"single-use-test-capability";
        (try Unix.execve measurement.executable (Array.of_list measurement.argv)
               execution_environment
         with Unix_error (error, function_name, argument) ->
           let reason =
             Printf.sprintf "execution-failed:%s:%s(%s)"
               (Unix.error_message error) function_name argument
           in
           append ~decision:"DENY" ~reason;
           failf "%s" reason)
    | Some context ->
        if context.kernel_grant_sha256 <> sha256 content then
          failf "execution-outcome-grant-digest-mismatch";
        let outcome_policy = load_outcome_policy root in
        let outcome_directory = execution_outcome_directory root in
        let issue_decision =
          hex_decode "decision" (capability_required table "decision_hex")
        in
        burn ();
        append ~decision:"ALLOW" ~reason:"single-use-kernel-capability";
        let outcome =
          supervise_child measurement.executable measurement.argv
            execution_environment
        in
        let observation =
          execution_observation_body ~root ~cwd:recorded_cwd ~token
            ~preexec_policy:policy ~outcome_policy ~measurement
            ~hardware_sha256:hardware_sha256_value
            ~environment_sha256:!environment_sha256_value ~context
            ~issue_decision ~consume_decision:decision ~outcome
        in
        let result_sha256 = sha256 observation in
        let observation_content =
          observation ^ "result_sha256=" ^ result_sha256 ^ "\n"
        in
        ignore
          (write_durable_outcome_record outcome_directory "observation"
             result_sha256 observation_content);
        let outcome_frame =
          execution_outcome_frame outcome_policy
            ~outcome_kind:outcome.observed_kind
            ~exit_code:outcome.observed_exit_code
            ~signal:outcome.observed_signal
            ~elapsed_us:outcome.observed_elapsed_us
            ~hardware_sha256:hardware_sha256_value
            ~command_sha256:measurement.command_sha256
            ~environment_sha256:!environment_sha256_value
            ~executable_sha256:measurement.executable_sha256
            ~grant_sha256:context.kernel_grant_sha256
            ~generation_sha256:(sha256 context.kernel_generation)
            ~issue_decision_sha256:(sha256 issue_decision)
            ~consume_decision_sha256:(sha256 decision)
            ~result_sha256
        in
        let outcome_decision =
          try
            invoke_outcome_authority root outcome_policy outcome_frame
              execution_environment
          with Authority_denied (code, denied) ->
            failf "execution-outcome-authority-denied:rc=%d:%s" code denied
        in
        let receipt =
          execution_outcome_receipt observation result_sha256 outcome_policy
            outcome_frame outcome_decision
        in
        let receipt_sha256 = sha256 receipt in
        ignore
          (write_durable_outcome_record outcome_directory "outcome"
             receipt_sha256 receipt);
        outcome_test_pause ();
        kernel_record_outcome context receipt receipt_sha256;
        outcome_test_replay context receipt receipt_sha256;
        append_outcome_commit_log ~root ~context ~outcome ~result_sha256
          ~receipt_sha256 ~decision_sha256:(sha256 outcome_decision);
        if outcome.observed_kind = 1 then outcome.observed_exit_code
        else if outcome.observed_kind = 2 then 128 + outcome.observed_signal
        else 126)
  with
  | Error reason as error ->
      cleanup ();
      (try append ~decision:"DENY" ~reason with _ -> ());
      raise error
  | Sys_error reason as error ->
      cleanup ();
      (try append ~decision:"DENY" ~reason with _ -> ());
      raise error
  | Unix_error (error, function_name, argument) as unix_error ->
      cleanup ();
      let reason =
        Printf.sprintf "%s:%s(%s)" (Unix.error_message error) function_name argument
      in
      (try append ~decision:"DENY" ~reason with _ -> ());
      raise unix_error

let consume_capability_file root token =
  ignore (file_capability_fixture_requested true);
  validate_token token;
  let root = Unix.realpath root in
  let directory = capability_directory root in
  let source = Filename.concat directory (token ^ ".cap") in
  let consuming =
    Filename.concat directory
      (Printf.sprintf "%s.consuming.%d" token (Unix.getpid ()))
  in
  let cleanup () =
    if Sys.file_exists consuming then (try Unix.unlink consuming with _ -> ())
  in
  let entered_core = ref false in
  let append_route_refusal reason =
    let hardware = hardware_record () in
    append_decision_log ~root ~phase:"CONSUME" ~decision:"DENY" ~reason
      ~token_sha256:(sha256 token) ~manifest_sha256:"-" ~source_sha256:"-"
      ~semantics_sha256:"-" ~command_hex:"-" ~command_sha256:"-"
      ~executable_hex:"-" ~executable_sha256:"-"
      ~hardware_record_hex:(hex_encode hardware)
      ~hardware_sha256:(hardware_hash hardware) ~environment_sha256:"-"
      ~language:13 ~execution_class:5 ~closure_attested:0 ~frame_sha256:"-"
      ~authority_result:"unavailable"
  in
  try
    (try Unix.rename source consuming
     with Unix_error (ENOENT, _, _) -> failf "capability-missing-or-replayed");
    fsync_directory directory;
    let info = Unix.lstat consuming in
    if info.st_kind <> S_REG then failf "capability-record-not-regular";
    if info.st_uid <> Unix.geteuid () then failf "capability-record-owner-mismatch";
    if info.st_perm land 0o077 <> 0 then failf "capability-record-mode-insecure";
    let content = read_file ~limit:(1024 * 1024) consuming in
    let burn () =
      Unix.unlink consuming;
      fsync_directory directory
    in
    entered_core := true;
    execute_capability_content ~root ~token ~content ~burn ~cleanup
      ~outcome_context:None
  with error ->
    cleanup ();
    if not !entered_core then (
      let reason =
        match error with
        | Error message | Sys_error message -> message
        | Unix_error (unix_error, function_name, argument) ->
            Printf.sprintf "%s:%s(%s)" (Unix.error_message unix_error)
              function_name argument
        | _ -> Printexc.to_string error
      in
      try append_route_refusal reason with _ -> ());
    raise error

let consume_capability_kernel root instance generation handle =
  validate_token generation;
  validate_token handle;
  let expected_instance = required_environment "SOUNIO_LOOM_INSTANCE_ID" in
  if instance <> expected_instance then failf "execution-kernel-instance-drift";
  let content, payload_sha256 =
    match kernel_request "EXEC_CONSUME" [ instance; generation; handle ] with
    | [ "OK"; "EXEC_CONSUMED"; actual_instance; actual_generation;
        payload_sha256; payload_hex ]
      when actual_instance = instance && actual_generation = generation ->
        let payload = hex_decode "kernel_payload" payload_hex in
        if sha256 payload <> payload_sha256 then
          failf "execution-kernel-payload-digest-mismatch";
        (payload, payload_sha256)
    | _ -> failf "execution-kernel-invalid-consume-response"
  in
  let table = parse_capability content in
  let token = capability_required table "token" in
  let context =
    { kernel_instance = instance; kernel_generation = generation;
      kernel_handle = handle; kernel_grant_sha256 = payload_sha256 }
  in
  execute_capability_content ~root ~token ~content ~burn:(fun () -> ())
    ~cleanup:(fun () -> ()) ~outcome_context:(Some context)

let run arguments =
  try
    let root = find_repo_root (Unix.getcwd ()) in
    (match arguments with
    | [ "--test-file-capability-fixture"; "--token"; token ] ->
        consume_capability_file root token
    | [ "--instance"; instance; "--generation"; generation; "--handle"; handle ] ->
        consume_capability_kernel root instance generation handle
    | _ ->
        failf
          "usage: exec-capability --instance INSTANCE --generation GENERATION --handle HANDLE")
  with
  | Error message
  | Sys_error message ->
      Printf.eprintf "sounio execution capability refused: %s\n%!" message;
      126
  | Unix_error (error, function_name, argument) ->
      Printf.eprintf "sounio execution capability refused: %s:%s(%s)\n%!"
        (Unix.error_message error) function_name argument;
      126
