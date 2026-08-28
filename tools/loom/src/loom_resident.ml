open Unix

exception Error of string

let pinned_runtime_manifest_sha256 =
  "458f9d0294dc3a5a484b04eb532d412e32d1cf0cebe99292e90bfb49670abcf1"

let max_file_bytes = 8 * 1024 * 1024
let max_frame_bytes = 65_535

external monotonic_us : unit -> int64 = "sounio_loom_monotonic_us"

type policy = {
  manifest_sha256 : string;
  runtime : string;
  runtime_sha256 : string;
  parent_9023_sha256 : string;
  parent_9024_sha256 : string;
}

type decision = {
  code : int;
  output : string;
  sequence : int;
  latency_us : int64;
  generation_sha256 : string;
  resident_pid : int;
}

type t = {
  root : string;
  policy : policy;
  environment : string array;
  pid : int;
  birth_identity : string;
  input : file_descr;
  output : file_descr;
  output_buffer : Buffer.t;
  generation_sha256 : string;
  mutable sequence : int;
  mutable outstanding : bool;
  mutable poisoned : bool;
  mutable closed : bool;
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
        else if total + count > limit then failf "resident-file-too-large:%s" path
        else (Buffer.add_subbytes output bytes 0 count; loop (total + count))
      in
      loop 0)

let sha256_file path = sha256 (read_file path)

let parse_manifest path =
  let table = Hashtbl.create 64 in
  read_file path |> String.split_on_char '\n'
  |> List.iter (fun line ->
         match String.index_opt line '=' with
         | None when line = "" -> ()
         | None -> failf "malformed-resident-runtime-manifest"
         | Some index ->
             let key = String.sub line 0 index in
             if Hashtbl.mem table key then failf "duplicate-resident-field:%s" key;
             Hashtbl.add table key
               (String.sub line (index + 1) (String.length line - index - 1)));
  table

let required table key =
  match Hashtbl.find_opt table key with
  | Some value when value <> "" -> value
  | _ -> failf "missing-resident-field:%s" key

let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"

let test_override name =
  match Sys.getenv_opt name with
  | Some value when value <> "" && test_mode () -> Some value
  | Some value when value <> "" -> failf "%s-override-requires-test-mode" name
  | _ -> None

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

let runtime_path root manifest =
  let sibling =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-resident-membrane-runtime"
  in
  let local = Filename.concat root (required manifest "runtime_path") in
  let selected =
    match test_override "SOUNIO_LOOM_RESIDENT_MEMBRANE_RUNTIME" with
    | Some path -> path
    | None when Sys.file_exists sibling -> sibling
    | None -> local
  in
  if not (Sys.file_exists selected) then failf "resident-runtime-missing:%s" selected;
  let expected = required manifest "runtime_sha256" in
  if sha256_file selected <> expected then failf "resident-runtime-hash-mismatch";
  (Unix.realpath selected, expected)

let load_policy root =
  let path =
    match test_override "SOUNIO_LOOM_RESIDENT_MEMBRANE_MANIFEST" with
    | Some path -> path
    | None -> Filename.concat root "tools/loom/resident_membrane.runtime.v1"
  in
  if not (Sys.file_exists path) then failf "resident-runtime-manifest-missing";
  let manifest_sha256 = sha256_file path in
  if manifest_sha256 <> pinned_runtime_manifest_sha256 then
    failf "resident-runtime-manifest-hash-mismatch";
  let manifest = parse_manifest path in
  if required manifest "schema" <> "loom-resident-membrane-runtime-v1"
     || required manifest "stage" <> "SOUNIO_RESIDENT_REALIZATION"
     || required manifest "producing_language" <> "Sounio"
     || required manifest "language_role" <> "SEMANTIC_AUTHORITY"
     || required manifest "actions" <> "9023,9024"
     || required manifest "runtime_frozen" <> "true"
     || required manifest "route_9024" <> "1"
     || required manifest "route_9023" <> "2"
     || required manifest "route_stop" <> "0"
     || required manifest "max_frame_bytes" <> "65535"
     || required manifest "framing" <> "sounio-read-byte-newline"
     || required manifest "ocaml_resident_started" <> "false"
     || required manifest "performance_gate" <> "false"
     || required manifest "membrane_integration" <> "false"
  then failf "resident-runtime-manifest-state-invalid";
  let parent_9023_sha256 = required manifest "parent_9023_manifest_sha256" in
  let parent_9024_sha256 = required manifest "parent_9024_manifest_sha256" in
  let parent_9023 = Filename.concat root (required manifest "parent_9023_manifest_path") in
  let parent_9024 = Filename.concat root (required manifest "parent_9024_manifest_path") in
  if sha256_file parent_9023 <> parent_9023_sha256 then
    failf "resident-parent-9023-hash-mismatch";
  if sha256_file parent_9024 <> parent_9024_sha256 then
    failf "resident-parent-9024-hash-mismatch";
  let dispatcher = Filename.concat root (required manifest "dispatcher_path") in
  let build_script = Filename.concat root (required manifest "build_script_path") in
  let gate_script = Filename.concat root (required manifest "gate_script_path") in
  if sha256_file dispatcher <> required manifest "dispatcher_sha256"
     || sha256_file build_script <> required manifest "build_script_sha256"
     || sha256_file gate_script <> required manifest "gate_script_sha256"
  then failf "resident-runtime-source-hash-mismatch";
  let runtime, runtime_sha256 = runtime_path root manifest in
  { manifest_sha256; runtime; runtime_sha256; parent_9023_sha256;
    parent_9024_sha256 }

let digest_u32_of_hex digest =
  if String.length digest <> 64 then failf "invalid-resident-sha256:%s" digest;
  List.init 8 (fun index ->
      let chunk = String.sub digest (index * 8) 8 in
      try Int64.to_string (Int64.of_string ("0x" ^ chunk))
      with _ -> failf "invalid-resident-sha256:%s" digest)
  |> String.concat " "

let zero_digest = "0 0 0 0 0 0 0 0"

let random_generation () =
  let descriptor = Unix.openfile "/dev/urandom" [ O_RDONLY ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      let bytes = Bytes.create 32 in
      let rec loop offset =
        if offset < Bytes.length bytes then
          match Unix.read descriptor bytes offset (Bytes.length bytes - offset) with
          | 0 -> failf "resident-random-eof"
          | count -> loop (offset + count)
          | exception Unix_error (EINTR, _, _) -> loop offset
      in
      loop 0;
      sha256 (Bytes.unsafe_to_string bytes))

let process_birth_identity pid runtime =
  let stat_path = Printf.sprintf "/proc/%d/stat" pid in
  let exe_path = Printf.sprintf "/proc/%d/exe" pid in
  let stat = read_file ~limit:65536 stat_path in
  let closing =
    match String.rindex_opt stat ')' with
    | Some index -> index
    | None -> failf "resident-process-stat-malformed"
  in
  let rest = String.sub stat (closing + 2) (String.length stat - closing - 2) in
  let fields = String.split_on_char ' ' rest |> List.filter (( <> ) "") in
  let start_time =
    try List.nth fields 19 with _ -> failf "resident-process-start-time-missing"
  in
  let executable = Unix.realpath exe_path in
  if executable <> runtime then failf "resident-process-executable-mismatch";
  Printf.sprintf "%d:%s:%s" pid start_time executable

let await_process_birth_identity pid runtime deadline_us =
  let rec loop () =
    (match Unix.waitpid [ WNOHANG ] pid with
    | 0, _ -> ()
    | _, status ->
        failf "resident-exited-before-admission:%d"
          (match status with
          | WEXITED value -> value
          | WSIGNALED value | WSTOPPED value -> 128 + value));
    try process_birth_identity pid runtime with
    | Error "resident-process-executable-mismatch"
    | Sys_error _
    | Unix_error _ ->
        if monotonic_us () >= deadline_us then
          failf "resident-exec-admission-timeout";
        ignore (Unix.select [] [] [] 0.001);
        loop ()
  in
  loop ()

let close_noerr descriptor = try Unix.close descriptor with _ -> ()

let kill_and_wait pid =
  (try Unix.kill pid Sys.sigkill with _ -> ());
  (try ignore (Unix.waitpid [] pid) with _ -> ())

let resident_alive resident =
  if resident.closed || resident.poisoned then false
  else
    match Unix.waitpid [ WNOHANG ] resident.pid with
    | 0, _ ->
        (try
           process_birth_identity resident.pid resident.policy.runtime
           = resident.birth_identity
         with _ -> false)
    | _ -> false

let remaining_seconds deadline_us =
  let remaining = Int64.sub deadline_us (monotonic_us ()) in
  if remaining <= 0L then 0.0 else Int64.to_float remaining /. 1_000_000.0

let wait_readable descriptor deadline_us =
  let timeout = remaining_seconds deadline_us in
  if timeout <= 0.0 then failf "resident-response-timeout";
  let readable, _, _ = Unix.select [ descriptor ] [] [] timeout in
  if readable = [] then failf "resident-response-timeout"

let wait_writable descriptor deadline_us =
  let timeout = remaining_seconds deadline_us in
  if timeout <= 0.0 then failf "resident-request-timeout";
  let _, writable, _ = Unix.select [] [ descriptor ] [] timeout in
  if writable = [] then failf "resident-request-timeout"

let write_all resident deadline_us value =
  let rec loop offset =
    if offset < String.length value then (
      wait_writable resident.input deadline_us;
      match Unix.write_substring resident.input value offset
              (String.length value - offset) with
      | 0 -> failf "resident-request-short-write"
      | count -> loop (offset + count)
      | exception Unix_error ((EAGAIN | EWOULDBLOCK | EINTR), _, _) -> loop offset)
  in
  loop 0

let extract_line buffer =
  let content = Buffer.contents buffer in
  match String.index_opt content '\n' with
  | None -> None
  | Some index ->
      let line = String.sub content 0 index in
      let rest =
        String.sub content (index + 1) (String.length content - index - 1)
      in
      Buffer.clear buffer;
      Buffer.add_string buffer rest;
      Some line

let read_line resident deadline_us =
  let bytes = Bytes.create 4096 in
  let rec loop () =
    match extract_line resident.output_buffer with
    | Some line -> line
    | None ->
        if Buffer.length resident.output_buffer > max_frame_bytes then
          failf "resident-response-too-large";
        wait_readable resident.output deadline_us;
        (match Unix.read resident.output bytes 0 (Bytes.length bytes) with
        | 0 -> failf "resident-response-eof"
        | count -> Buffer.add_subbytes resident.output_buffer bytes 0 count; loop ()
        | exception Unix_error ((EAGAIN | EWOULDBLOCK | EINTR), _, _) -> loop ())
  in
  loop ()

let decision_code output =
  let marker = " code=" in
  let rec find offset =
    if offset + String.length marker > String.length output then
      failf "resident-decision-code-missing"
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
  with _ -> failf "resident-decision-code-invalid"

let validate_output route output =
  let prefix =
    if route = 1 then "SOUNIO_RESIDENT_AUTHORITY_"
    else if route = 2 then "SOUNIO_SUBPROCESS_MEMBRANE_"
    else failf "resident-route-invalid"
  in
  if not (starts_with output prefix)
     || not (ends_with output "stage=SEMANTICS_FROZEN")
  then failf "resident-decision-malformed:%s" output;
  decision_code output

let utc_now () =
  let tm = Unix.gmtime (Unix.gettimeofday ()) in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02dZ"
    (tm.tm_year + 1900) (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min tm.tm_sec

let receipt_path resident =
  match test_override "SOUNIO_LOOM_RESIDENT_RECEIPT_LOG" with
  | Some path -> path
  | None -> Filename.concat (git_common_dir resident.root)
      "sounio-loom-resident-authority.tsv"

let append_receipt resident ~event ~sequence ~frame ~code ~output ~latency_us =
  let descriptor =
    Unix.openfile (receipt_path resident) [ O_WRONLY; O_CREAT; O_APPEND ] 0o600
  in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      let line =
        String.concat "\t"
          [ "schema=loom-resident-decision-v1"; "utc=" ^ utc_now ();
            "event=" ^ event; "generation_sha256=" ^ resident.generation_sha256;
            "sequence=" ^ string_of_int sequence;
            "decision=" ^ (if code = 0 then "ALLOW" else "DENY");
            "code=" ^ string_of_int code; "pid=" ^ string_of_int resident.pid;
            "birth_sha256=" ^ sha256 resident.birth_identity;
            "parent_9023_manifest_sha256=" ^ resident.policy.parent_9023_sha256;
            "parent_9024_manifest_sha256=" ^ resident.policy.parent_9024_sha256;
            "resident_manifest_sha256=" ^ resident.policy.manifest_sha256;
            "resident_runtime_sha256=" ^ resident.policy.runtime_sha256;
            "frame_sha256=" ^ sha256 frame; "result_sha256=" ^ sha256 output;
            "latency_us=" ^ Int64.to_string latency_us ] ^ "\n"
      in
      let rec write offset =
        if offset < String.length line then
          match Unix.write_substring descriptor line offset
                  (String.length line - offset) with
          | 0 -> failf "resident-receipt-short-write"
          | count -> write (offset + count)
          | exception Unix_error (EINTR, _, _) -> write offset
      in
      write 0;
      Unix.fsync descriptor;
      Unix.lockf descriptor F_ULOCK 0)

let poison resident reason =
  if not resident.poisoned then (
    resident.poisoned <- true;
    resident.outstanding <- false;
    kill_and_wait resident.pid;
    close_noerr resident.input;
    close_noerr resident.output;
    resident.closed <- true;
    let output = "poison:" ^ reason in
    (try
       append_receipt resident ~event:"POISON" ~sequence:resident.sequence
         ~frame:output ~code:445 ~output ~latency_us:0L
     with _ -> ()))

let ensure_usable resident =
  if resident.poisoned then failf "resident-generation-poisoned";
  if resident.closed then failf "resident-generation-closed";
  if not (resident_alive resident) then (
    poison resident "resident-not-alive";
    failf "resident-not-alive")

let invoke resident ~route ~event ~sequence ~deadline_us frame =
  ensure_usable resident;
  if String.length frame > max_frame_bytes then failf "resident-request-too-large";
  let started = monotonic_us () in
  try
    write_all resident deadline_us
      (string_of_int route ^ "\n" ^ frame ^ "\n");
    let output = read_line resident deadline_us in
    let code = validate_output route output in
    let latency_us = Int64.sub (monotonic_us ()) started in
    append_receipt resident ~event ~sequence ~frame ~code ~output ~latency_us;
    (code, output, latency_us)
  with
  | Error reason as error -> poison resident reason; raise error
  | Sys_error reason as error -> poison resident reason; raise error
  | Unix_error (unix_error, function_name, argument) as error ->
      poison resident
        (Printf.sprintf "%s:%s(%s)" (Unix.error_message unix_error)
           function_name argument);
      raise error

let deadline_after_ms deadline_ms =
  if deadline_ms < 1 || deadline_ms > 120_000 then
    failf "resident-deadline-out-of-range";
  let now = monotonic_us () in
  if now <= 0L then failf "resident-monotonic-clock-failed";
  Int64.add now (Int64.mul (Int64.of_int deadline_ms) 1000L)

let resident_frame resident ~event_kind ~sequence ~previous_sequence
    ~request_present ~response_present ~correlation_valid ~deadline_us
    ~request_hash ~result_hash =
  let deadline_hash = sha256 ("deadline_monotonic_us=" ^ Int64.to_string deadline_us ^ "\n") in
  String.concat " "
    [ "9024"; "3"; string_of_int event_kind; "1"; "1";
      string_of_int sequence; string_of_int previous_sequence;
      string_of_int request_present; string_of_int response_present;
      string_of_int correlation_valid; "1"; "1"; "0";
      digest_u32_of_hex resident.policy.parent_9023_sha256;
      digest_u32_of_hex resident.generation_sha256;
      (if request_present = 1 then digest_u32_of_hex request_hash else zero_digest);
      (if response_present = 1 then digest_u32_of_hex result_hash else zero_digest);
      digest_u32_of_hex deadline_hash ]

let spawn ~root ~environment ~deadline_ms =
  let root = Unix.realpath root in
  let policy = load_policy root in
  let startup_deadline_us = deadline_after_ms deadline_ms in
  let input_read, input_write = Unix.pipe () in
  let output_read, output_write = Unix.pipe () in
  Unix.set_close_on_exec input_write;
  Unix.set_close_on_exec output_read;
  let pid =
    match Unix.fork () with
    | 0 ->
        Unix.close input_write;
        Unix.close output_read;
        Unix.dup2 input_read Unix.stdin;
        Unix.dup2 output_write Unix.stdout;
        Unix.dup2 output_write Unix.stderr;
        if input_read <> Unix.stdin then Unix.close input_read;
        if output_write <> Unix.stdout && output_write <> Unix.stderr then
          Unix.close output_write;
        (try Unix.chdir root;
             Unix.execve policy.runtime [| policy.runtime |] environment
         with _ -> Unix._exit 127)
    | pid -> pid
  in
  Unix.close input_read;
  Unix.close output_write;
  Unix.set_nonblock input_write;
  Unix.set_nonblock output_read;
  let resident =
    try
      { root; policy; environment; pid;
        birth_identity =
          await_process_birth_identity pid policy.runtime startup_deadline_us;
        input = input_write; output = output_read; output_buffer = Buffer.create 4096;
        generation_sha256 = random_generation (); sequence = 0;
        outstanding = false; poisoned = false; closed = false }
    with error ->
      kill_and_wait pid;
      close_noerr input_write;
      close_noerr output_read;
      raise error
  in
  let deadline_us = deadline_after_ms deadline_ms in
  let start_frame =
    resident_frame resident ~event_kind:1 ~sequence:0 ~previous_sequence:0
      ~request_present:0 ~response_present:0 ~correlation_valid:1 ~deadline_us
      ~request_hash:(sha256 "start") ~result_hash:(sha256 "start")
  in
  let code, _, _ =
    invoke resident ~route:1 ~event:"START" ~sequence:0 ~deadline_us start_frame
  in
  if code <> 0 then (
    poison resident ("start-denied-" ^ string_of_int code);
    failf "resident-start-denied:%d" code);
  resident

let decide resident ~deadline_ms frame =
  ensure_usable resident;
  if resident.outstanding then (
    poison resident "concurrent-request";
    failf "resident-request-already-outstanding");
  let frame =
    if ends_with frame "\n" then String.sub frame 0 (String.length frame - 1)
    else frame
  in
  let deadline_us = deadline_after_ms deadline_ms in
  let previous_sequence = resident.sequence in
  let sequence = previous_sequence + 1 in
  let request_hash = sha256 frame in
  let request_frame =
    resident_frame resident ~event_kind:2 ~sequence ~previous_sequence
      ~request_present:1 ~response_present:0 ~correlation_valid:1 ~deadline_us
      ~request_hash ~result_hash:(sha256 "pending")
  in
  resident.outstanding <- true;
  try
    let request_code, _, request_latency =
      invoke resident ~route:1 ~event:"REQUEST" ~sequence ~deadline_us
        request_frame
    in
    if request_code <> 0 then failf "resident-request-denied:%d" request_code;
    let effect_code, output, effect_latency =
      invoke resident ~route:2 ~event:"EFFECT" ~sequence ~deadline_us frame
    in
    let result_hash = sha256 output in
    let response_frame =
      resident_frame resident ~event_kind:3 ~sequence ~previous_sequence
        ~request_present:1 ~response_present:1 ~correlation_valid:1 ~deadline_us
        ~request_hash ~result_hash
    in
    let response_code, _, response_latency =
      invoke resident ~route:1 ~event:"RESPONSE" ~sequence ~deadline_us
        response_frame
    in
    if response_code <> 0 then failf "resident-response-denied:%d" response_code;
    resident.sequence <- sequence;
    resident.outstanding <- false;
    { code = effect_code; output; sequence;
      latency_us = Int64.add request_latency (Int64.add effect_latency response_latency);
      generation_sha256 = resident.generation_sha256; resident_pid = resident.pid }
  with error ->
    resident.outstanding <- false;
    (match error with
    | Error reason -> poison resident reason
    | Sys_error reason -> poison resident reason
    | Unix_error (unix_error, function_name, argument) ->
        poison resident
          (Printf.sprintf "%s:%s(%s)" (Unix.error_message unix_error)
             function_name argument)
    | _ -> poison resident "unexpected-resident-error");
    raise error

let close resident ~deadline_ms =
  if resident.poisoned || resident.closed then ()
  else
    let deadline_us = deadline_after_ms deadline_ms in
    let stop_frame =
      resident_frame resident ~event_kind:4 ~sequence:resident.sequence
        ~previous_sequence:resident.sequence ~request_present:0
        ~response_present:0 ~correlation_valid:1 ~deadline_us
        ~request_hash:(sha256 "stop") ~result_hash:(sha256 "stop")
    in
    let code, _, _ =
      invoke resident ~route:1 ~event:"STOP" ~sequence:resident.sequence
        ~deadline_us stop_frame
    in
    if code <> 0 then (
      poison resident ("stop-denied-" ^ string_of_int code);
      failf "resident-stop-denied:%d" code);
    (try write_all resident deadline_us "0\n"
     with error -> poison resident "stop-route-failed"; raise error);
    close_noerr resident.input;
    let rec wait () =
      match Unix.waitpid [ WNOHANG ] resident.pid with
      | 0, _ ->
          if remaining_seconds deadline_us <= 0.0 then (
            poison resident "stop-timeout";
            failf "resident-stop-timeout")
          else (ignore (Unix.select [] [] [] 0.001); wait ())
      | _, WEXITED 0 -> ()
      | _, status ->
          poison resident "stop-exit-invalid";
          failf "resident-stop-exit-invalid:%d"
            (match status with
            | WEXITED value -> value
            | WSIGNALED value | WSTOPPED value -> 128 + value)
    in
    wait ();
    close_noerr resident.output;
    resident.closed <- true

let with_generation ~root ~environment ~deadline_ms callback =
  let resident = spawn ~root ~environment ~deadline_ms in
  match callback resident with
  | result ->
      if not resident.closed && not resident.poisoned then
        close resident ~deadline_ms;
      result
  | exception callback_error ->
      if not resident.closed && not resident.poisoned then
        (try close resident ~deadline_ms with _ -> ());
      raise callback_error

let test_replay resident ~deadline_ms frame =
  if not (test_mode ()) then failf "resident-test-replay-requires-test-mode";
  ensure_usable resident;
  let deadline_us = deadline_after_ms deadline_ms in
  let sequence = resident.sequence in
  let request_hash = sha256 frame in
  let replay_frame =
    resident_frame resident ~event_kind:2 ~sequence
      ~previous_sequence:resident.sequence ~request_present:1
      ~response_present:0 ~correlation_valid:1 ~deadline_us ~request_hash
      ~result_hash:(sha256 "pending")
  in
  let code, _, _ =
    invoke resident ~route:1 ~event:"REPLAY" ~sequence ~deadline_us replay_frame
  in
  if code <> 442 then (
    poison resident "replay-control-failed";
    failf "resident-replay-control-failed:%d" code);
  poison resident "replay-denied";
  code

let test_uncorrelated resident ~deadline_ms frame =
  if not (test_mode ()) then failf "resident-test-correlation-requires-test-mode";
  ensure_usable resident;
  let deadline_us = deadline_after_ms deadline_ms in
  let previous_sequence = resident.sequence in
  let sequence = previous_sequence + 1 in
  let request_hash = sha256 frame in
  let response_frame =
    resident_frame resident ~event_kind:3 ~sequence ~previous_sequence
      ~request_present:1 ~response_present:1 ~correlation_valid:0 ~deadline_us
      ~request_hash ~result_hash:(sha256 "mismatched")
  in
  let code, _, _ =
    invoke resident ~route:1 ~event:"UNCORRELATED" ~sequence ~deadline_us
      response_frame
  in
  if code <> 443 then (
    poison resident "correlation-control-failed";
    failf "resident-correlation-control-failed:%d" code);
  poison resident "correlation-denied";
  code

let test_timeout resident frame =
  if not (test_mode ()) then failf "resident-test-timeout-requires-test-mode";
  let deadline_us = Int64.sub (monotonic_us ()) 1L in
  try
    ignore (invoke resident ~route:2 ~event:"TIMEOUT" ~sequence:1
              ~deadline_us frame);
    failf "resident-timeout-control-admitted"
  with
  | Error "resident-request-timeout" -> true
  | Error "resident-response-timeout" -> true

let test_eof resident ~deadline_ms frame =
  if not (test_mode ()) then failf "resident-test-eof-requires-test-mode";
  Unix.kill resident.pid Sys.sigkill;
  let deadline_us = deadline_after_ms deadline_ms in
  try
    ignore (invoke resident ~route:2 ~event:"EOF" ~sequence:1
              ~deadline_us frame);
    failf "resident-eof-control-admitted"
  with
  | Error "resident-not-alive" -> true
  | Error "resident-response-eof" -> true

let is_poisoned resident = resident.poisoned
let generation resident = resident.generation_sha256
let pid resident = resident.pid
let birth resident = resident.birth_identity
let sequence resident = resident.sequence
let now_us () = monotonic_us ()
