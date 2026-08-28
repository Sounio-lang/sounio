open Unix

exception Error of string

let pinned_fixture_manifest_sha256 =
  "10401ebe4d302647220433eadb0b1240ce2b3128801f421f2712ae757f5105b5"

let max_frame_bytes = 65_535

type phase = Await_issue | Await_consume | Await_terminal | Finished

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let starts_with value prefix =
  String.length value >= String.length prefix
  && String.sub value 0 (String.length prefix) = prefix

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let read_file path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let length = in_channel_length channel in
      if length > 8 * 1024 * 1024 then failf "controller-file-too-large";
      really_input_string channel length)

let sha256_file path = sha256 (read_file path)

let required_environment name =
  match Sys.getenv_opt name with
  | Some value when value <> "" -> value
  | _ -> failf "missing-controller-environment:%s" name

let parse_positive_integer name value =
  try
    let parsed = int_of_string value in
    if parsed <= 0 then failf "invalid-controller-environment:%s" name;
    parsed
  with Failure _ -> failf "invalid-controller-environment:%s" name

let valid_digest value =
  String.length value = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let read_line_bounded channel =
  let output = Buffer.create 256 in
  let rec loop length =
    match input_char channel with
    | '\n' -> Some (Buffer.contents output)
    | '\r' -> failf "controller-carriage-return-refused"
    | _ when length >= max_frame_bytes -> failf "controller-frame-too-large"
    | character -> Buffer.add_char output character; loop (length + 1)
    | exception End_of_file ->
        if length = 0 then None else failf "controller-truncated-frame"
  in
  loop 0

let split_command line =
  match String.index_opt line ' ' with
  | Some index when index > 0 && index + 1 < String.length line ->
      ( String.sub line 0 index,
        String.sub line (index + 1) (String.length line - index - 1) )
  | _ -> (line, "")

let filtered_environment () =
  let prohibited =
    [ "LD_PRELOAD="; "LD_LIBRARY_PATH="; "LD_AUDIT=";
      "SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_";
      "SOUNIO_LOOM_RESIDENT_MEMBRANE_" ]
  in
  Unix.environment () |> Array.to_list
  |> List.filter (fun binding ->
         not
           (List.exists
              (fun prefix -> starts_with binding prefix)
              prohibited))
  |> Array.of_list

let controller_root () =
  match Sys.getenv_opt "SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_ROOT" with
  | Some root
    when root <> "" && Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"
    -> Unix.realpath root
  | Some root when root <> "" ->
      failf "controller-root-override-requires-test-mode"
  | _ -> Unix.realpath "/usr/lib/sounio/loom/current"

let verify_fixture_manifest root =
  let path = Filename.concat root "tools/loom/host_exec_quorum_fixture.freeze.v1" in
  if not (Sys.file_exists path) then failf "controller-fixture-manifest-missing";
  if sha256_file path <> pinned_fixture_manifest_sha256 then
    failf "controller-fixture-manifest-hash-mismatch"

let state_name cell =
  Loom_exec_grant_cell.state cell |> Loom_exec_grant_cell.state_name

let emit cell ~operation ~controller_generation ~transaction_digest
    ~frame_hash ~code ~quorum_ready ~terminal =
  Printf.printf
    "LOOM_EXEC_GRANT_CONTROLLER_V1 operation=%s semantic_authority=Sounio operational_kernel=OCaml controller_generation_sha256=%s transaction_digest_sha256=%s frame_sha256=%s code=%d state=%s resident_pid=%d resident_generation_sha256=%s resident_sequence=%d resident_poisoned=%s quorum_ready=%s controller_terminal=%s single_resident_controller=true non_bearer_transport=pending material_grant=false material_execution=false barrier_release=false exec_attached=false parity_open=false claim_ready=false\n%!"
    operation controller_generation transaction_digest frame_hash code
    (state_name cell) (Loom_exec_grant_cell.resident_pid cell)
    (Loom_exec_grant_cell.generation cell)
    (Loom_exec_grant_cell.sequence cell)
    (if Loom_exec_grant_cell.is_poisoned cell then "true" else "false")
    (if quorum_ready then "true" else "false")
    (if terminal then "true" else "false")

let run () =
  if required_environment "SOUNIO_LOOM_EXEC_GRANT_CONTROLLER" <> "1" then
    failf "controller-internal-marker-invalid";
  let root = controller_root () in
  verify_fixture_manifest root;
  let parent_pid =
    required_environment "SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_PARENT_PID"
    |> parse_positive_integer "SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_PARENT_PID"
  in
  if Unix.getppid () <> parent_pid then failf "controller-parent-mismatch";
  let controller_generation =
    required_environment "SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_GENERATION"
  in
  if not (valid_digest controller_generation) then
    failf "controller-generation-invalid";
  let deadline_ms =
    required_environment "SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_DEADLINE_MS"
    |> parse_positive_integer "SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_DEADLINE_MS"
  in
  if deadline_ms > 60_000 then failf "controller-deadline-out-of-range";
  let environment = filtered_environment () in
  Loom_exec_grant_cell.with_cell ~root ~environment ~deadline_ms (fun cell ->
      let phase = ref Await_issue in
      let transaction_digest = ref (sha256 ("exec-quorum-v1\000" ^ controller_generation)) in
      while !phase <> Finished do
        if Unix.getppid () <> parent_pid then failf "controller-parent-drift";
        let line =
          match read_line_bounded Stdlib.stdin with
          | Some value -> value
          | None -> failf "controller-unexpected-eof"
        in
        let command, frame = split_command line in
        match (!phase, command, frame) with
        | Await_issue, "ISSUE", frame when frame <> "" ->
            let frame_hash = sha256 frame in
            transaction_digest :=
              sha256 (!transaction_digest ^ "\000ISSUE\000" ^ frame_hash);
            let decision = Loom_exec_grant_cell.issue cell frame in
            let allowed = decision.code = 0 in
            emit cell ~operation:"ISSUE" ~controller_generation
              ~transaction_digest:!transaction_digest ~frame_hash
              ~code:decision.code ~quorum_ready:false ~terminal:(not allowed);
            phase := if allowed then Await_consume else Finished
        | Await_consume, "CONSUME", frame when frame <> "" ->
            let frame_hash = sha256 frame in
            transaction_digest :=
              sha256 (!transaction_digest ^ "\000CONSUME\000" ^ frame_hash);
            let decision = Loom_exec_grant_cell.consume cell frame in
            let allowed = decision.code = 0 in
            emit cell ~operation:"CONSUME" ~controller_generation
              ~transaction_digest:!transaction_digest ~frame_hash
              ~code:decision.code ~quorum_ready:allowed ~terminal:(not allowed);
            phase := if allowed then Await_terminal else Finished
        | (Await_consume | Await_terminal), "REVOKE", frame when frame <> "" ->
            let frame_hash = sha256 frame in
            transaction_digest :=
              sha256 (!transaction_digest ^ "\000REVOKE\000" ^ frame_hash);
            let decision = Loom_exec_grant_cell.revoke cell frame in
            emit cell ~operation:"REVOKE" ~controller_generation
              ~transaction_digest:!transaction_digest ~frame_hash
              ~code:decision.code ~quorum_ready:false ~terminal:true;
            phase := Finished
        | Await_terminal, "CLOSE", frame when frame <> "" ->
            let frame_hash = sha256 frame in
            transaction_digest :=
              sha256 (!transaction_digest ^ "\000CLOSE\000" ^ frame_hash);
            let decision = Loom_exec_grant_cell.close_outcome cell frame in
            emit cell ~operation:"CLOSE" ~controller_generation
              ~transaction_digest:!transaction_digest ~frame_hash
              ~code:decision.code ~quorum_ready:false ~terminal:true;
            phase := Finished
        | _, "STOP", "" ->
            let digest = sha256 (!transaction_digest ^ "\000STOP") in
            emit cell ~operation:"STOP" ~controller_generation
              ~transaction_digest:digest ~frame_hash:(sha256 "") ~code:0
              ~quorum_ready:false ~terminal:true;
            phase := Finished
        | _ -> failf "controller-protocol-state-mismatch"
      done)

let () =
  try run ()
  with
  | Error reason ->
      Printf.eprintf "loom-exec-grant-controller: REFUSE reason=%s\n%!" reason;
      exit 70
  | Loom_exec_grant_cell.Error reason ->
      Printf.eprintf "loom-exec-grant-controller: REFUSE reason=%s\n%!" reason;
      exit 70
  | Loom_resident.Error reason ->
      Printf.eprintf "loom-exec-grant-controller: REFUSE reason=%s\n%!" reason;
      exit 70
  | Sys_error reason ->
      Printf.eprintf "loom-exec-grant-controller: REFUSE reason=%s\n%!" reason;
      exit 70
  | Unix.Unix_error (error, operation, argument) ->
      Printf.eprintf "loom-exec-grant-controller: REFUSE reason=%s:%s(%s)\n%!"
        (Unix.error_message error) operation argument;
      exit 70
