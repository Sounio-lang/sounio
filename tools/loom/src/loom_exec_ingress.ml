open Unix

exception Error of string

let max_line_bytes = 4096
let descriptor_deadline_seconds = 2.0

external file_descr_of_int : int -> file_descr = "sounio_loom_file_descr_of_int"
external peer_credentials : file_descr -> int * int * int
  = "sounio_loom_peer_credentials"

type observation = {
  descriptor_present : bool;
  descriptor_bound : bool;
  peer_pid : int;
  peer_uid : int;
  peer_gid : int;
  peer_distinct_uid : bool;
  activation_code : int;
  activation_generation_sha256 : string;
}

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let starts_with value prefix =
  String.length value >= String.length prefix
  && String.sub value 0 (String.length prefix) = prefix

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let valid_sha256 value =
  String.length value = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"

let exact_test_flag name =
  match Sys.getenv_opt name with
  | None -> false
  | Some "1" when test_mode () -> true
  | Some _ when test_mode () -> failf "%s-invalid" name
  | Some _ -> failf "%s-requires-test-mode" name

let required_mode () =
  match Sys.getenv_opt "SOUNIO_LOOM_EXEC_INGRESS_REQUIRED" with
  | None | Some "0" -> false
  | Some "1" -> true
  | Some _ -> failf "product-exec-ingress-required-mode-invalid"

let probe_only () = exact_test_flag "SOUNIO_LOOM_EXEC_INGRESS_PROBE_ONLY"

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

let first_line value =
  match String.split_on_char '\n' value with
  | line :: _ -> String.trim line
  | [] -> ""

let read_file path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () -> really_input_string channel (in_channel_length channel))

let read_stream_file path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let output = Buffer.create 256 in
      let bytes = Bytes.create 1024 in
      let rec loop total =
        let count = input channel bytes 0 (Bytes.length bytes) in
        if count = 0 then Buffer.contents output
        else if total + count > max_line_bytes then
          failf "product-exec-ingress-probe-parent-argv-too-large"
        else (Buffer.add_subbytes output bytes 0 count; loop (total + count))
      in
      loop 0)

let same_uid_fixture_allowed () =
  if not (exact_test_flag "SOUNIO_LOOM_EXEC_INGRESS_ALLOW_SAME_UID_TEST") then
    false
  else if not (probe_only ()) then
    failf "product-exec-ingress-same-uid-fixture-requires-probe-only"
  else
    let parent = Unix.getppid () in
    let parent_executable, self_executable =
      try
        (Unix.stat (Printf.sprintf "/proc/%d/exe" parent),
         Unix.stat "/proc/self/exe")
      with _ -> failf "product-exec-ingress-probe-parent-unavailable"
    in
    let parent_arguments =
      try
        read_stream_file (Printf.sprintf "/proc/%d/cmdline" parent)
        |> String.split_on_char '\000'
      with _ -> failf "product-exec-ingress-probe-parent-argv-unavailable"
    in
    if parent_executable.st_dev <> self_executable.st_dev
       || parent_executable.st_ino <> self_executable.st_ino
    then failf "product-exec-ingress-same-uid-fixture-parent-executable-mismatch";
    if not (List.mem "exec-ingress-probe" parent_arguments) then
      failf "product-exec-ingress-same-uid-fixture-parent-argv-mismatch";
    true

let git_common_dir root =
  let marker = Filename.concat root ".git" in
  if Sys.is_directory marker then Unix.realpath marker
  else
    let line = first_line (read_file marker) in
    if not (starts_with line "gitdir: ") then failf "invalid-gitdir-marker";
    let raw = String.sub line 8 (String.length line - 8) in
    let git_dir = Unix.realpath (normalize_absolute root raw) in
    let common_marker = Filename.concat git_dir "commondir" in
    if Sys.file_exists common_marker then
      let common = first_line (read_file common_marker) in
      Unix.realpath (normalize_absolute git_dir common)
    else git_dir

let rec mkdir_p path =
  if path = "" || path = "/" || Sys.file_exists path then ()
  else (mkdir_p (Filename.dirname path); Unix.mkdir path 0o700)

let audit_path root =
  match Sys.getenv_opt "SOUNIO_LOOM_EXEC_INGRESS_DARK_LOG" with
  | Some value when value <> "" && test_mode () -> value
  | Some value when value <> "" ->
      failf "product-exec-ingress-log-override-requires-test-mode"
  | _ ->
      Filename.concat
        (Filename.concat (git_common_dir root) "sounio-loom-product-exec-ingress")
        "dark.tsv"

let write_all descriptor value =
  let rec loop offset =
    if offset < String.length value then
      match Unix.write_substring descriptor value offset
              (String.length value - offset) with
      | 0 -> failf "product-exec-ingress-short-write"
      | count -> loop (offset + count)
      | exception Unix_error (EINTR, _, _) -> loop offset
  in
  loop 0

let read_line descriptor deadline =
  let output = Buffer.create 256 in
  let byte = Bytes.create 1 in
  let rec loop () =
    let remaining = deadline -. Unix.gettimeofday () in
    if remaining <= 0.0 then failf "product-exec-ingress-response-timeout";
    let ready, _, _ = Unix.select [ descriptor ] [] [] remaining in
    if ready = [] then failf "product-exec-ingress-response-timeout";
    match Unix.read descriptor byte 0 1 with
    | 0 -> failf "product-exec-ingress-response-eof"
    | _ ->
        let character = Bytes.get byte 0 in
        if character = '\n' then Buffer.contents output
        else if character = '\r' || Buffer.length output >= max_line_bytes then
          failf "product-exec-ingress-response-malformed"
        else (Buffer.add_char output character; loop ())
    | exception Unix_error (EINTR, _, _) -> loop ()
  in
  loop ()

let require_eof descriptor deadline =
  let byte = Bytes.create 1 in
  let rec loop () =
    let remaining = deadline -. Unix.gettimeofday () in
    if remaining <= 0.0 then failf "product-exec-ingress-peer-did-not-close";
    let ready, _, _ = Unix.select [ descriptor ] [] [] remaining in
    if ready = [] then failf "product-exec-ingress-peer-did-not-close";
    match Unix.read descriptor byte 0 1 with
    | 0 -> ()
    | _ -> failf "product-exec-ingress-response-trailing-bytes"
    | exception Unix_error (EINTR, _, _) -> loop ()
  in
  loop ()

let peer_alive pid =
  if pid <= 0 then false
  else
    try Unix.kill pid 0; true with
    | Unix_error (EPERM, _, _) -> true
    | Unix_error _ -> false

let descriptor_from_environment () =
  match Sys.getenv_opt "SOUNIO_LOOM_EXEC_INGRESS_FD" with
  | None | Some "" -> None
  | Some raw ->
      let number =
        try int_of_string raw
        with _ -> failf "product-exec-ingress-descriptor-invalid"
      in
      if number < 3 || number > 65_535 then
        failf "product-exec-ingress-descriptor-out-of-range";
      (try Some (file_descr_of_int number)
       with Failure _ -> failf "product-exec-ingress-descriptor-unavailable")

let append_audit ~root ~agent ~lane ~session_id ~cwd ~event_sha256
    ~command_sha256 ~descriptor_present ~descriptor_bound ~peer_pid ~peer_uid
    ~peer_gid ~peer_distinct_uid ~decision ~reason evaluation =
  let path = audit_path root in
  mkdir_p (Filename.dirname path);
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      let activation_fields =
        match evaluation with
        | None ->
            [ "decision_authority=OCaml-structural-precondition";
              "sounio_evaluated=false"; "semantic_authority=Sounio";
              "producing_language=OCaml";
              "language_role=OPERATIONAL_ATTACHMENT";
              "activation_code=-1"; "action_manifest_sha256=-";
              "semantics_sha256=-"; "projection_sha256=-";
              "projection_label=-"; "capsule_state_after=-";
              "authority_result_sha256=-";
              "authority_generation_sha256=-"; "authority_pid=0";
              "authority_sequence=0" ]
        | Some (evaluation : Loom_membrane.activation_dark_evaluation) ->
            let policy = evaluation.activation_policy in
            let authority = evaluation.activation_decision in
            [ "decision_authority=Sounio"; "sounio_evaluated=true";
              "semantic_authority=Sounio"; "producing_language=Sounio";
              "language_role=SEMANTIC_AUTHORITY";
              "activation_code=" ^ string_of_int authority.code;
              "action_manifest_sha256=" ^ policy.action_manifest_sha256;
              "semantics_sha256=" ^ policy.semantics_sha256;
              "projection_sha256=" ^ policy.projection_sha256;
              "projection_label=" ^ policy.label;
              "capsule_state_after=" ^ evaluation.activation_capsule_state;
              "authority_result_sha256=" ^ sha256 authority.output;
              "authority_generation_sha256=" ^ authority.generation_sha256;
              "authority_pid=" ^ string_of_int authority.resident_pid;
              "authority_sequence=" ^ string_of_int authority.sequence ]
      in
      let line =
        String.concat "\t"
          ([ "schema=loom-product-exec-ingress-dark-decision-v1";
             "utc=" ^ utc_now (); "decision=" ^ decision; "reason=" ^ reason;
             "authorizing=false"; "production_activation=false";
             "exec_attached=false"; "descriptor_present=" ^
               string_of_bool descriptor_present;
             "descriptor_bound=" ^ string_of_bool descriptor_bound;
             "descriptor_transport=unix-stream-inherited";
             "descriptor_is_bearer=false";
             "process_pid=" ^ string_of_int (Unix.getpid ());
             "process_parent_pid=" ^ string_of_int (Unix.getppid ());
             "process_uid=" ^ string_of_int (Unix.getuid ());
             "process_euid=" ^ string_of_int (Unix.geteuid ());
             "process_gid=" ^ string_of_int (Unix.getgid ());
             "process_egid=" ^ string_of_int (Unix.getegid ());
             "peer_pid=" ^ string_of_int peer_pid;
             "peer_uid=" ^ string_of_int peer_uid;
             "peer_gid=" ^ string_of_int peer_gid;
             "peer_distinct_uid=" ^ string_of_bool peer_distinct_uid;
             "agent_sha256=" ^ sha256 agent; "lane_sha256=" ^ sha256 lane;
             "session_id_sha256=" ^ sha256 session_id;
             "cwd_sha256=" ^ sha256 cwd; "event_sha256=" ^ event_sha256;
             "command_sha256=" ^ command_sha256;
             "operational_language=OCaml";
             "operational_role=OPERATIONAL_ATTACHMENT" ] @ activation_fields)
        ^ "\n"
      in
      write_all descriptor line;
      Unix.fsync descriptor;
      Unix.lockf descriptor F_ULOCK 0)

let observe ~root ~agent ~lane ~session_id ~cwd ~event_sha256 ~command_sha256 =
  if not (valid_sha256 event_sha256 && valid_sha256 command_sha256) then
    failf "product-exec-ingress-digest-invalid";
  let root = Unix.realpath root in
  let cwd = Unix.realpath cwd in
  match descriptor_from_environment () with
  | None ->
      append_audit ~root ~agent ~lane ~session_id ~cwd ~event_sha256
        ~command_sha256 ~descriptor_present:false ~descriptor_bound:false
        ~peer_pid:0 ~peer_uid:(Unix.geteuid ()) ~peer_gid:(Unix.getegid ())
        ~peer_distinct_uid:false ~decision:"DENY" ~reason:"descriptor-absent"
        None;
      if required_mode () then failf "product-exec-ingress-descriptor-absent";
      None
  | Some descriptor ->
      Fun.protect
        ~finally:(fun () -> try Unix.close descriptor with _ -> ())
        (fun () ->
          Unix.set_close_on_exec descriptor;
          let info = Unix.fstat descriptor in
          if info.st_kind <> S_SOCK then
            failf "product-exec-ingress-descriptor-not-socket";
          (try ignore (Unix.getpeername descriptor)
           with Unix_error _ -> failf "product-exec-ingress-descriptor-not-connected");
          let peer_pid, peer_uid, peer_gid =
            try peer_credentials descriptor
            with Failure _ -> failf "product-exec-ingress-peer-credentials-unavailable"
          in
          if peer_pid = Unix.getpid () || not (peer_alive peer_pid) then
            failf "product-exec-ingress-peer-identity-invalid";
          let peer_distinct_uid = peer_uid <> Unix.geteuid () in
          let peer_admitted =
            if peer_distinct_uid then true
            else
              try same_uid_fixture_allowed () with
              | Error reason as error ->
                  append_audit ~root ~agent ~lane ~session_id ~cwd
                    ~event_sha256 ~command_sha256 ~descriptor_present:true
                    ~descriptor_bound:false ~peer_pid ~peer_uid ~peer_gid
                    ~peer_distinct_uid:false ~decision:"DENY" ~reason None;
                  raise error
          in
          if not peer_admitted
          then (
            append_audit ~root ~agent ~lane ~session_id ~cwd ~event_sha256
              ~command_sha256 ~descriptor_present:true ~descriptor_bound:false
              ~peer_pid ~peer_uid ~peer_gid ~peer_distinct_uid:false
              ~decision:"DENY" ~reason:"peer-not-distinct" None;
            failf "product-exec-ingress-peer-not-distinct");
          let deadline = Unix.gettimeofday () +. descriptor_deadline_seconds in
          write_all descriptor
            (String.concat "\t"
               [ "LOOM_EXEC_INGRESS/1"; event_sha256; command_sha256 ] ^ "\n");
          Unix.shutdown descriptor SHUTDOWN_SEND;
          let response = read_line descriptor deadline in
          let expected =
            String.concat "\t"
              [ "LOOM_EXEC_INGRESS_BOUND/1"; event_sha256; command_sha256 ]
          in
          if response <> expected then
            failf "product-exec-ingress-response-binding-mismatch";
          require_eof descriptor deadline;
          let evaluation =
            Loom_membrane.evaluate_product_activation_dark ~policy_root:root
              ~audit_root:root ~deadline_ms:15_000
          in
          let authority = evaluation.activation_decision in
          append_audit ~root ~agent ~lane ~session_id ~cwd ~event_sha256
            ~command_sha256 ~descriptor_present:true ~descriptor_bound:true
            ~peer_pid ~peer_uid ~peer_gid ~peer_distinct_uid
            ~decision:(if authority.code = 0 then "ALLOW" else "DENY")
            ~reason:"descriptor-bound-action-9031" (Some evaluation);
          if authority.code = 0 then
            failf "product-exec-ingress-dark-unexpected-allow";
          Some
            { descriptor_present = true; descriptor_bound = true; peer_pid;
              peer_uid; peer_gid; peer_distinct_uid;
              activation_code = authority.code;
              activation_generation_sha256 = authority.generation_sha256 })
