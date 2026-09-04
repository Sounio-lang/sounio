open Unix

exception Error of string

let freeze_sha256 =
  "0765d7e941a5def05e8ae7d08a90c7826491c86b4c1efc8679b40a6a728de29d"

let semantics_sha256 =
  "9a323d98a6c732e0a7f70a6d50cf684e5039eb2af211e5f891fd0c9761351549"

let sounio_source_sha256 =
  "2016d7f46e112c88d7b59b77beff4277fb5c4f023c7a2fc64b9c6282ab1d4a16"

let sounio_executable_sha256 =
  "68d3f8efd22454dc3a66242f2beafad804cfae8550fee94716c718614d1ead90"

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format
let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"
let process_timeout_seconds = 30.0
let lock_timeout_seconds = 2.0
let max_process_output_bytes = 8 * 1024 * 1024

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let sha256_file path =
  let stat = Unix.lstat path in
  if stat.st_kind <> S_REG then failf "file-not-regular:%s" path;
  let channel = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () ->
      Cryptokit.hash_channel (Cryptokit.Hash.sha256 ()) channel
      |> Cryptokit.transform_string (Cryptokit.Hexa.encode ()))

let read_file path =
  let channel = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () ->
      let output = Buffer.create 4096 and bytes = Bytes.create 16384 in
      let rec loop total =
        let count = input channel bytes 0 (Bytes.length bytes) in
        if count = 0 then Buffer.contents output
        else if total + count > 8 * 1024 * 1024 then failf "file-too-large:%s" path
        else (Buffer.add_subbytes output bytes 0 count; loop (total + count))
      in
      loop 0)

let ensure_governed_file path =
  let stat = Unix.lstat path in
  if stat.st_kind <> S_REG then failf "governed-file-not-regular:%s" path;
  if stat.st_uid <> Unix.geteuid () then failf "governed-file-owner-drift:%s" path;
  if stat.st_perm land 0o022 <> 0 then failf "governed-file-permissions-open:%s" path

let read_governed_file path =
  ensure_governed_file path;
  read_file path

let parse_fields label text =
  let fields = Hashtbl.create 48 in
  String.split_on_char '\n' text
  |> List.iter (fun line ->
         if line <> "" then
           match String.index_opt line '=' with
           | None -> failf "%s-malformed" label
           | Some split ->
               let key = String.sub line 0 split in
               let value = String.sub line (split + 1) (String.length line - split - 1) in
               if key = "" || value = "" then failf "%s-empty-field" label;
               if Hashtbl.mem fields key && key <> "capability" then
                 failf "%s-duplicate-field:%s" label key;
               if not (Hashtbl.mem fields key) then Hashtbl.add fields key value);
  fields

let required label fields key =
  match Hashtbl.find_opt fields key with
  | Some value when value <> "" -> value
  | _ -> failf "%s-missing:%s" label key

let safe_token value =
  if value = "" then failf "empty-token";
  String.iter
    (function
      | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '.' | '_' | '-' -> ()
      | _ -> failf "unsafe-token:%s" value)
    value;
  value

let slug value =
  String.map
    (function 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '.' | '_' | '-' as c -> c | _ -> '-')
    value

let rec mkdir_p path =
  if path = "" || path = "." || path = "/" || Sys.file_exists path then ()
  else (mkdir_p (Filename.dirname path); Unix.mkdir path 0o700)

let write_all descriptor value =
  let rec loop offset =
    if offset < String.length value then
      let count = Unix.write_substring descriptor value offset (String.length value - offset) in
      if count = 0 then failf "short-write" else loop (offset + count)
  in
  loop 0

let atomic_write path value =
  mkdir_p (Filename.dirname path);
  let temporary = Printf.sprintf "%s.%d.tmp" path (Unix.getpid ()) in
  let descriptor = Unix.openfile temporary [ O_WRONLY; O_CREAT; O_EXCL ] 0o600 in
  Fun.protect
    ~finally:(fun () -> try Unix.close descriptor with _ -> ())
    (fun () -> write_all descriptor value; Unix.fsync descriptor);
  Unix.rename temporary path

let encoded parts =
  parts |> List.map (fun value -> Printf.sprintf "%d:%s" (String.length value) value)
  |> String.concat ""

type identity = {
  agent : string; lane : string; session_id : string; generation : string;
  harness : string; worktree : string; host : string; boot_id : string;
  pid_namespace : string; pid : string; pid_start : string;
}

let identity_digest identity =
  sha256
    (encoded
       [ identity.agent; identity.lane; identity.session_id; identity.generation;
         identity.harness; identity.worktree; identity.host; identity.boot_id;
         identity.pid_namespace; identity.pid; identity.pid_start ])

let identity_of_presence fields =
  { agent = required "presence" fields "agent";
    lane = required "presence" fields "lane";
    session_id = required "presence" fields "session_id";
    generation = required "presence" fields "generation";
    harness = required "presence" fields "harness";
    worktree = required "presence" fields "worktree";
    host = required "presence" fields "host";
    boot_id = required "presence" fields "boot_id";
    pid_namespace = required "presence" fields "pid_namespace";
    pid = required "presence" fields "pid";
    pid_start = required "presence" fields "pid_start" }

let process_start pid =
  let stat = read_file (Printf.sprintf "/proc/%s/stat" pid) in
  let closing =
    match String.rindex_opt stat ')' with
    | Some value -> value
    | None -> failf "invalid-process-stat"
  in
  let tail = String.sub stat (closing + 2) (String.length stat - closing - 2)
             |> String.split_on_char ' ' |> List.filter (( <> ) "") in
  match List.nth_opt tail 19 with
  | Some value -> value
  | None -> failf "process-start-missing"

let kernel_identity_matches identity =
  identity.host = Unix.gethostname ()
  && identity.boot_id = String.trim (read_file "/proc/sys/kernel/random/boot_id")
  && identity.pid_namespace = Unix.readlink "/proc/self/ns/pid"
  && (try process_start identity.pid = identity.pid_start with _ -> false)

type runtime = {
  id : string; directory : string; manifest_sha256 : string;
  loom_sha256 : string; coord_sha256 : string; source_sha : string;
}

let runtime_root root =
  match Sys.getenv_opt "SOUNIO_COORD_RUNTIME_DIR" with
  | Some value when value <> "" -> Unix.realpath value
  | _ -> Filename.concat root "sounio-coord-runtime"

let validate_runtime runtime_root id =
  let id = safe_token id in
  let versions = Unix.realpath (Filename.concat runtime_root "versions") in
  let directory = Unix.realpath (Filename.concat versions id) in
  if Filename.dirname directory <> versions then failf "runtime-path-escape";
  let manifest_path = Filename.concat directory "manifest" in
  let manifest = parse_fields "runtime-manifest" (read_governed_file manifest_path) in
  if required "runtime-manifest" manifest "runtime_id" <> id then failf "runtime-id-drift";
  let loom = Filename.concat directory "bin/sounio-loom-runtime" in
  let coord = Filename.concat directory "bin/sounio-coord-runtime" in
  let loom_sha256 = required "runtime-manifest" manifest "loom_runtime_sha256" in
  let coord_sha256 = required "runtime-manifest" manifest "coord_runtime_sha256" in
  if sha256_file loom <> loom_sha256 || sha256_file coord <> coord_sha256 then
    failf "runtime-executable-drift:%s" id;
  { id; directory; manifest_sha256 = sha256_file manifest_path; loom_sha256;
    coord_sha256; source_sha = required "runtime-manifest" manifest "source_sha" }

let selector_runtime runtime_root selector =
  let path = Filename.concat runtime_root selector in
  let target = Unix.realpath path in
  let versions = Unix.realpath (Filename.concat runtime_root "versions") in
  if Filename.dirname target <> versions then failf "selector-path-escape:%s" selector;
  validate_runtime runtime_root (Filename.basename target)

let state_root git_common =
  match Sys.getenv_opt "SOUNIO_COORD_DIR" with
  | Some path when path <> "" -> path
  | _ -> Filename.concat git_common "sounio-coord-state"

let pin_directory state = Filename.concat state "generation-runtime-pins"
let pin_path state identity =
  Filename.concat (pin_directory state) (slug identity.agent ^ "--" ^ slug identity.lane ^ ".pin")

let policy_root source_root =
  match Sys.getenv_opt "SOUNIO_LOOM_GENERATION_PIN_POLICY_ROOT" with
  | Some value when value <> "" && test_mode () -> value
  | Some _ -> failf "policy-root-override-requires-test-mode"
  | _ ->
      let installed = Filename.concat (Filename.dirname (Filename.dirname (Unix.realpath Sys.executable_name)))
          "policy/generation-pinned-cutover" in
      if Sys.file_exists installed then installed else source_root

let load_authority source_root =
  let root = policy_root source_root in
  let manifest_path = Filename.concat root "tools/loom/generation_pinned_cutover.freeze.v1" in
  if sha256_file manifest_path <> freeze_sha256 then failf "action-9048-freeze-drift";
  let manifest = parse_fields "action-9048-freeze" (read_governed_file manifest_path) in
  if required "action-9048-freeze" manifest "stage" <> "SEMANTICS_FROZEN"
     || required "action-9048-freeze" manifest "action" <> "9048"
     || required "action-9048-freeze" manifest "semantic_authority" <> "Sounio"
     || required "action-9048-freeze" manifest "semantics_sha256" <> semantics_sha256
  then failf "action-9048-policy-invalid";
  let installed = Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-generation-pinned-cutover" in
  let fallback = Filename.concat source_root
      "tools/loom/_build/default/src/sounio-loom-generation-pinned-cutover" in
  let executable = if Sys.file_exists installed then installed else fallback in
  if sha256_file executable <> required "action-9048-freeze" manifest "executable_sha256" then
    failf "action-9048-executable-drift";
  executable

type process_result = { code : int; output : string }

let run_process ?(input = "") ?(environment = Unix.environment ()) command arguments =
  let input_read, input_write = Unix.pipe () in
  let output_read, output_write = Unix.pipe () in
  let pid = match Unix.fork () with
    | 0 ->
        Unix.dup2 input_read Unix.stdin; Unix.dup2 output_write Unix.stdout;
        Unix.dup2 output_write Unix.stderr; Unix.close input_write; Unix.close output_read;
        (try Unix.execvpe command (Array.of_list (command :: arguments)) environment
         with _ -> Unix._exit 127)
    | pid -> pid
  in
  Unix.close input_read; Unix.close output_write;
  let close_noerr descriptor = try Unix.close descriptor with _ -> () in
  let kill_noerr () =
    (try Unix.kill pid Sys.sigkill with _ -> ());
    (try ignore (Unix.waitpid [] pid) with _ -> ())
  in
  Fun.protect
    ~finally:(fun () -> close_noerr input_write; close_noerr output_read)
    (fun () ->
      (try write_all input_write input with error -> kill_noerr (); raise error);
      Unix.close input_write;
      let deadline = Unix.gettimeofday () +. process_timeout_seconds in
      let buffer = Buffer.create 1024 and bytes = Bytes.create 4096 in
      let rec drain () =
        let remaining = deadline -. Unix.gettimeofday () in
        if remaining <= 0.0 then (kill_noerr (); failf "process-timeout:%s" command);
        let ready, _, _ = Unix.select [ output_read ] [] [] remaining in
        if ready = [] then (kill_noerr (); failf "process-timeout:%s" command)
        else
          match Unix.read output_read bytes 0 (Bytes.length bytes) with
          | 0 -> ()
          | count ->
              if Buffer.length buffer + count > max_process_output_bytes then (
                kill_noerr (); failf "process-output-too-large:%s" command);
              Buffer.add_subbytes buffer bytes 0 count;
              drain ()
          | exception Unix_error (EINTR, _, _) -> drain ()
      in
      drain ();
      let _, status = Unix.waitpid [] pid in
      let code = match status with
        | WEXITED n -> n | WSIGNALED n | WSTOPPED n -> 128 + n
      in
      { code; output = Buffer.contents buffer })

let digest_u60 digest offset =
  Int64.of_string ("0x" ^ String.sub digest offset 15) |> Int64.to_string

let audit_value value =
  String.map (function '\n' | '\r' | '\t' -> ' ' | character -> character) value

let append_decision state ~verdict ~reason ~command ~identity_digest ~runtime_digest
    ~transaction_digest ~result =
  mkdir_p state;
  let path = Filename.concat state "generation-pin-decisions.log" in
  if Sys.file_exists path then ensure_governed_file path;
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  let line = Printf.sprintf
      "verdict=%s reason=%s source_sha256=%s semantics_sha256=%s executable_sha256=%s producing_language=Sounio language_role=SEMANTIC_AUTHORITY parity_language=OCaml parity_role=EFFECT_PARITY toolchain=ocaml-%s hardware_host=%s command=%s identity_sha256=%s runtime_digest=%s transaction_sha256=%s result=%s\n"
      verdict (audit_value reason) sounio_source_sha256 semantics_sha256
      sounio_executable_sha256 Sys.ocaml_version (Unix.gethostname ())
      (audit_value command) identity_digest runtime_digest transaction_digest
      (audit_value result)
  in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
      write_all descriptor line; Unix.fsync descriptor)

let authority_decide state executable ~command mode word identity_digest runtime_digest
    transaction_digest expected =
  let frame = Printf.sprintf "9048 %d 3 %d %s %s %s %s %s 14 14\n" mode word
      (digest_u60 identity_digest 0) (digest_u60 identity_digest 15)
      (digest_u60 runtime_digest 0) (digest_u60 runtime_digest 15)
      (digest_u60 transaction_digest 0) in
  let result = run_process ~input:frame executable [] in
  let output = String.trim result.output in
  let allowed = result.code = 0 && output = expected in
  append_decision state ~verdict:(if allowed then "ALLOW" else "DENY")
    ~reason:(if allowed then expected else "action-9048-denied") ~command
    ~identity_digest ~runtime_digest ~transaction_digest ~result:output;
  if not allowed then failf "action-9048-denied:%s" output

let pin_text identity presence_sha selection runtime =
  let digest = identity_digest identity in
  String.concat "\n"
    [ "schema=loom-generation-runtime-pin-v1"; "state=SEALED";
      "identity_sha256=" ^ digest; "agent=" ^ identity.agent; "lane=" ^ identity.lane;
      "session_id=" ^ identity.session_id; "generation=" ^ identity.generation;
      "harness=" ^ identity.harness; "worktree=" ^ identity.worktree;
      "host=" ^ identity.host; "boot_id=" ^ identity.boot_id;
      "pid_namespace=" ^ identity.pid_namespace; "pid=" ^ identity.pid;
      "pid_start=" ^ identity.pid_start; "presence_sha256=" ^ presence_sha;
      "selection=" ^ selection; "runtime_id=" ^ runtime.id;
      "runtime_manifest_sha256=" ^ runtime.manifest_sha256;
      "loom_runtime_sha256=" ^ runtime.loom_sha256;
      "coord_runtime_sha256=" ^ runtime.coord_sha256;
      "runtime_source_sha=" ^ runtime.source_sha;
      "semantic_authority=Sounio"; "language_role=SEMANTIC_AUTHORITY";
      "action=9048"; "semantics_sha256=" ^ semantics_sha256;
      "freeze_sha256=" ^ freeze_sha256; "" ]

let validate_pin state runtime_root identity =
  let path = pin_path state identity in
  let text = read_governed_file path in
  let fields = parse_fields "generation-pin" text in
  let exact key expected =
    if required "generation-pin" fields key <> expected then failf "pin-identity-drift:%s" key
  in
  exact "schema" "loom-generation-runtime-pin-v1"; exact "state" "SEALED";
  exact "agent" identity.agent; exact "lane" identity.lane;
  exact "session_id" identity.session_id; exact "generation" identity.generation;
  exact "harness" identity.harness; exact "worktree" identity.worktree;
  exact "host" identity.host; exact "boot_id" identity.boot_id;
  exact "pid_namespace" identity.pid_namespace; exact "pid" identity.pid;
  exact "pid_start" identity.pid_start; exact "identity_sha256" (identity_digest identity);
  exact "presence_sha256" (identity_digest identity);
  exact "action" "9048"; exact "semantics_sha256" semantics_sha256;
  exact "freeze_sha256" freeze_sha256;
  let runtime = validate_runtime runtime_root (required "generation-pin" fields "runtime_id") in
  exact "runtime_manifest_sha256" runtime.manifest_sha256;
  exact "loom_runtime_sha256" runtime.loom_sha256;
  exact "coord_runtime_sha256" runtime.coord_sha256;
  (text, runtime)

let capability_runtime state runtime_root identity =
  let path = Filename.concat (Filename.concat state "hook-capabilities")
      (slug identity.agent ^ "--" ^ slug identity.lane ^ ".capability") in
  if not (Sys.file_exists path) then None
  else
    try
      let fields = parse_fields "capability" (read_governed_file path) in
      if required "capability" fields "agent" <> identity.agent
         || required "capability" fields "lane" <> identity.lane
         || required "capability" fields "session_id" <> identity.session_id
         || required "capability" fields "presence_pid" <> identity.pid
         || required "capability" fields "presence_pid_start" <> identity.pid_start
         || required "capability" fields "presence_boot_id" <> identity.boot_id
         || required "capability" fields "presence_pid_namespace" <> identity.pid_namespace
      then None
      else Some (validate_runtime runtime_root (required "capability" fields "runtime_id"))
    with _ -> None

let regular_files directory suffix =
  if not (Sys.file_exists directory) then []
  else Sys.readdir directory |> Array.to_list |> List.sort String.compare
       |> List.filter (fun name -> Filename.check_suffix name suffix)

let with_lock state operation =
  mkdir_p (pin_directory state);
  let path = Filename.concat (pin_directory state) ".lock" in
  let descriptor = Unix.openfile path [ O_RDWR; O_CREAT ] 0o600 in
  let deadline = Unix.gettimeofday () +. lock_timeout_seconds in
  let rec acquire () =
    try Unix.lockf descriptor F_TLOCK 0
    with
    | Unix_error ((EACCES | EAGAIN), _, _) ->
        if Unix.gettimeofday () >= deadline then failf "generation-pin-lock-timeout";
        ignore (Unix.select [] [] [] 0.02);
        acquire ()
  in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
      acquire ();
      (match Sys.getenv_opt "SOUNIO_LOOM_GENERATION_PIN_TEST_HOLD_LOCK_SECONDS" with
       | Some value when test_mode () ->
           let seconds = try float_of_string value with _ -> failf "invalid-test-lock-hold" in
           if seconds < 0.0 || seconds > 10.0 then failf "invalid-test-lock-hold";
           ignore (Unix.select [] [] [] seconds)
       | Some _ -> failf "test-lock-hold-requires-test-mode"
       | None -> ());
      operation ())

let seal ~source_root ~git_common ~old_runtime_id ~candidate_runtime_id =
  let state = state_root git_common and runtimes = runtime_root git_common in
  let authority = load_authority source_root in
  let old_runtime = validate_runtime runtimes old_runtime_id in
  let candidate = validate_runtime runtimes candidate_runtime_id in
  if (selector_runtime runtimes "current").id <> old_runtime.id
     || (selector_runtime runtimes "native-next").id <> candidate.id
     || old_runtime.id = candidate.id then failf "pre-cutover-selector-drift";
  with_lock state (fun () ->
      let live = ref [] in
      let classified = ref [] in
      regular_files (Filename.concat state "process-presences") ".presence"
      |> List.iter (fun name ->
             let path = Filename.concat (Filename.concat state "process-presences") name in
             let text = read_governed_file path in
             let fields = parse_fields "presence" text in
             let identity = identity_of_presence fields in
             if name <> slug identity.agent ^ "--" ^ slug identity.lane ^ ".presence" then
               failf "presence-filename-drift";
             if test_mode () && not (kernel_identity_matches identity) then
               failf "fixture-kernel-identity-mismatch:host=%b:boot=%b:namespace=%b:start=%b"
                 (identity.host = Unix.gethostname ())
                 (identity.boot_id = String.trim (read_file "/proc/sys/kernel/random/boot_id"))
                 (identity.pid_namespace = Unix.readlink "/proc/self/ns/pid")
                 (try process_start identity.pid = identity.pid_start with _ -> false);
             let presence_sha = identity_digest identity in
             if kernel_identity_matches identity then (
               let selected, selection, mode =
                 match capability_runtime state runtimes identity with
                 | Some runtime -> (runtime, "capability", 1)
                 | None -> (old_runtime, "pre-cutover-current", 2)
               in
               let pin = pin_text identity presence_sha selection selected in
               let transaction = sha256 (presence_sha ^ sha256 pin ^ candidate.manifest_sha256) in
               authority_decide state authority ~command:"hook-generation-pin-seal" mode 33546239 (identity_digest identity)
                 selected.manifest_sha256 transaction
                 (Printf.sprintf "SOUNIO_GENERATION_PINNED_CUTOVER %s semantic_authority=Sounio action=9048"
                    (if mode = 1 then "SEALED_CAPABILITY" else "SEALED_LEGACY"));
               let destination = pin_path state identity in
               if Sys.file_exists destination then (
                 let existing, existing_runtime = validate_pin state runtimes identity in
                 let existing_fields = parse_fields "generation-pin" existing in
                 if existing_runtime.id <> selected.id
                    || required "generation-pin" existing_fields "selection" <> selection
                 then failf "pin-overwrite-refused")
               else atomic_write destination pin;
               live := (identity_digest identity ^ ":" ^ selected.id) :: !live;
               classified := (name ^ ":LIVE:" ^ presence_sha) :: !classified)
             else
               classified := (name ^ ":NOT_LIVE:" ^ presence_sha) :: !classified);
      let inventory = !classified |> List.sort String.compare |> String.concat "\n" |> sha256 in
      let receipt = String.concat "\n"
          [ "schema=loom-generation-pin-set-v1"; "state=SEALED";
            "old_runtime_id=" ^ old_runtime.id; "candidate_runtime_id=" ^ candidate.id;
            "inventory_sha256=" ^ inventory;
            "pin_count=" ^ string_of_int (List.length !live);
            "semantic_authority=Sounio"; "action=9048";
            "semantics_sha256=" ^ semantics_sha256; "freeze_sha256=" ^ freeze_sha256; "" ] in
      let transaction = sha256 receipt in
      authority_decide state authority ~command:"hook-generation-pin-cutover-ready" 6 33554431 inventory candidate.manifest_sha256
        transaction
        "SOUNIO_GENERATION_PINNED_CUTOVER CUTOVER_READY semantic_authority=Sounio action=9048";
      let activation = Filename.concat (pin_directory state) "activation.v1" in
      if Sys.file_exists activation then (
        if read_governed_file activation <> receipt then failf "activation-receipt-overwrite-refused")
      else atomic_write activation receipt;
      Printf.printf "LOOM_GENERATION_PIN_SEALED pins=%d inventory_sha256=%s old_runtime=%s candidate_runtime=%s\n%!"
        (List.length !live) inventory old_runtime.id candidate.id;
      Printf.printf "SOUNIO_GENERATION_PINNED_CUTOVER CUTOVER_READY semantic_authority=Sounio action=9048\n%!")

let identity_from_caller ~agent ~lane ~session_id ~generation ~harness ~worktree
    ~host ~boot_id ~pid_namespace ~pid ~pid_start =
  { agent; lane; session_id; generation; harness; worktree; host; boot_id;
    pid_namespace; pid = string_of_int pid; pid_start }

let dispatch ~source_root ~git_common ~agent ~lane ~session_id ~harness
    ~worktree ~host ~boot_id ~pid_namespace ~pid ~pid_start ~raw_event ~arguments =
  let state = state_root git_common in
  let activation = Filename.concat (pin_directory state) "activation.v1" in
  if not (Sys.file_exists activation) then None
  else
    let authority = load_authority source_root in
    let runtimes = runtime_root git_common in
    let activation_fields =
      parse_fields "generation-pin-activation" (read_governed_file activation)
    in
    let activation_exact key expected =
      if required "generation-pin-activation" activation_fields key <> expected then
        failf "generation-pin-activation-drift:%s" key
    in
    activation_exact "schema" "loom-generation-pin-set-v1";
    activation_exact "state" "SEALED";
    activation_exact "semantic_authority" "Sounio";
    activation_exact "action" "9048";
    activation_exact "semantics_sha256" semantics_sha256;
    activation_exact "freeze_sha256" freeze_sha256;
    let old_runtime_id = required "generation-pin-activation" activation_fields "old_runtime_id" in
    let candidate_runtime_id =
      required "generation-pin-activation" activation_fields "candidate_runtime_id"
    in
    ignore (validate_runtime runtimes old_runtime_id);
    ignore (validate_runtime runtimes candidate_runtime_id);
    let current_runtime_id = (selector_runtime runtimes "current").id in
    let provisional = identity_from_caller ~agent ~lane ~session_id ~generation:"1" ~harness
        ~worktree ~host ~boot_id ~pid_namespace ~pid ~pid_start in
    let path = pin_path state provisional in
    let generation =
      if Sys.file_exists path then
        required "generation-pin" (parse_fields "generation-pin" (read_file path)) "generation"
      else "1"
    in
    let identity = { provisional with generation } in
    if not (Sys.file_exists path) && current_runtime_id = old_runtime_id then None
    else (
    if not (Sys.file_exists path) then
      with_lock state (fun () ->
          if not (Sys.file_exists path) then (
            let target = selector_runtime runtimes "current" in
            let source_digest = sha256 raw_event in
            let pin = pin_text identity source_digest "post-cutover-birth" target in
            let transaction = sha256 (source_digest ^ sha256 pin ^ target.manifest_sha256) in
            authority_decide state authority ~command:"hook-generation-pin-birth" 5 33021951 (identity_digest identity)
              target.manifest_sha256 transaction
              "SOUNIO_GENERATION_PINNED_CUTOVER BIRTH_PINNED semantic_authority=Sounio action=9048";
            atomic_write path pin));
    let pin, target = validate_pin state runtimes identity in
    if not (kernel_identity_matches identity) then failf "generation-pin-kernel-drift";
    let executing = Unix.realpath Sys.executable_name in
    let target_executable = Filename.concat target.directory "bin/sounio-loom-runtime" in
    let same = executing = Unix.realpath target_executable in
    let forwarded_target =
      Sys.getenv_opt "SOUNIO_LOOM_GENERATION_PIN_FORWARD_TARGET"
    in
    (match forwarded_target with
     | Some runtime_id when runtime_id <> target.id ->
         failf "generation-pin-forward-target-drift"
     | _ -> ());
    let decision = if same then "CONTINUE" else "FORWARD" in
    authority_decide state authority ~command:"hook-generation-pin-resolve" (if same then 3 else 4) 33550335
      (identity_digest identity) target.manifest_sha256 (sha256 pin)
      (Printf.sprintf "SOUNIO_GENERATION_PINNED_CUTOVER %s semantic_authority=Sounio action=9048" decision);
    if same then None
    else if forwarded_target <> None then failf "generation-pin-recursive-forward"
    else
      let selector =
        Filename.concat (Filename.concat runtimes "generation-selectors") target.id
      in
      if not (Sys.file_exists selector)
         || Unix.realpath (Filename.concat selector "current") <> target.directory
      then failf "generation-pin-selector-missing:%s" target.id;
      let inherited =
        Unix.environment () |> Array.to_list
        |> List.filter (fun value ->
             let prefix = "SOUNIO_COORD_RUNTIME_DIR=" in
             String.length value < String.length prefix
             || String.sub value 0 (String.length prefix) <> prefix)
        |> Array.of_list
      in
      let environment = Array.append
          [| "SOUNIO_LOOM_GENERATION_PIN_FORWARD_TARGET=" ^ target.id;
             "SOUNIO_COORD_RUNTIME_DIR=" ^ selector |]
          inherited in
      Some (run_process ~input:raw_event ~environment target_executable ("agent-hook" :: arguments)))

let run_seal arguments =
  let rec parse source_root git_common old_runtime candidate = function
    | [] -> (source_root, git_common, old_runtime, candidate)
    | "--source-root" :: value :: rest -> parse value git_common old_runtime candidate rest
    | "--git-common" :: value :: rest -> parse source_root value old_runtime candidate rest
    | "--old-runtime" :: value :: rest -> parse source_root git_common value candidate rest
    | "--candidate-runtime" :: value :: rest -> parse source_root git_common old_runtime value rest
    | _ -> failf "usage: hook-generation-pin-seal --source-root ROOT --git-common DIR --old-runtime ID --candidate-runtime ID"
  in
  let source_root, git_common, old_runtime, candidate = parse "" "" "" "" arguments in
  if source_root = "" || git_common = "" || old_runtime = "" || candidate = "" then
    failf "hook-generation-pin-seal-missing-argument";
  seal ~source_root:(Unix.realpath source_root) ~git_common:(Unix.realpath git_common)
    ~old_runtime_id:old_runtime ~candidate_runtime_id:candidate;
  0
