open Unix

exception Error of string

let semantic_manifest_sha256 =
  "966f022c98bc7df89ce40a90ede9ec8a9a726499baec0fd21e72f327f286a176"

let material_manifest_sha256 =
  "2045439f1a07d737a0cb8370ad080a80cd0715db2966863539f3c0794d14d7e3"

let max_payload_bytes = 1024 * 1024
let max_output_bytes = 8 * 1024 * 1024
let result_schema = "loom-sovereign-exec-result-v1"

external pidfd_open : int -> file_descr option = "sounio_loom_pidfd_open"
external arm_parent_death_kill : unit -> unit
  = "sounio_loom_arm_parent_death_kill"

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format
let sha256 = Loom_exec.sha256
let sha256_file = Loom_exec.sha256_file
let hex_encode = Loom_exec.hex_encode
let hex_decode = Loom_exec.hex_decode
let valid_sha256 value =
  String.length value = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value
let read_file = Loom_exec.read_file
let write_all = Loom_exec.write_all
let starts_with = Loom_exec.starts_with
let trim = Loom_exec.trim

let rec mkdir_p path =
  if path = "" || path = "." || path = "/" || Sys.file_exists path then ()
  else (
    mkdir_p (Filename.dirname path);
    try Unix.mkdir path 0o700 with Unix_error (EEXIST, _, _) -> ())

let required_mode () =
  match Sys.getenv_opt "SOUNIO_LOOM_SOVEREIGN_EXEC_REQUIRED" with
  | Some "1" -> true
  | None | Some "0" -> false
  | Some _ -> failf "sovereign-required-mode-invalid"

let exact table key expected =
  let actual = Loom_exec.required table key in
  if actual <> expected then failf "sovereign-%s-invalid:%s" key actual

let require_digest label value =
  if not (valid_sha256 value) then failf "sovereign-%s-invalid" label;
  value

let require_int label value =
  try int_of_string value with _ -> failf "sovereign-%s-invalid" label

let require_int64 label value =
  try Int64.of_string value with _ -> failf "sovereign-%s-invalid" label

let process_start pid =
  let line = read_file ~limit:65536 (Printf.sprintf "/proc/%d/stat" pid) in
  let close =
    match String.rindex_opt line ')' with
    | Some value -> value
    | None -> failf "sovereign-process-stat-malformed"
  in
  let suffix =
    String.sub line (close + 2) (String.length line - close - 2)
    |> String.split_on_char ' '
  in
  match List.nth_opt suffix 19 with
  | Some value when value <> "" -> value
  | _ -> failf "sovereign-process-start-missing"

let pidfd_alive descriptor =
  let ready, _, _ = Unix.select [ descriptor ] [] [] 0.0 in
  ready = []

type gate = {
  semantic_sha256 : string;
  material_sha256 : string;
  runtime_sha256 : string;
  production_decision : string;
  production_decision_sha256 : string;
}

let load_gate root environment =
  let semantic_path =
    Filename.concat root "tools/loom/sovereign_execution_kernel.freeze.v1"
  in
  let material_path =
    Filename.concat root
      "tools/loom/sovereign_execution_kernel_material.runtime.v1"
  in
  if sha256_file semantic_path <> semantic_manifest_sha256 then
    failf "sovereign-semantic-manifest-hash-mismatch";
  if sha256_file material_path <> material_manifest_sha256 then
    failf "sovereign-material-manifest-hash-mismatch";
  let semantic = Loom_exec.parse_manifest semantic_path in
  let material = Loom_exec.parse_manifest material_path in
  exact semantic "schema" "loom-sovereign-execution-kernel-freeze-v1";
  exact semantic "stage" "SEMANTICS_FROZEN";
  exact semantic "semantic_authority" "Sounio";
  exact semantic "action" "9042";
  exact semantic "grant_is_bearer" "false";
  exact semantic "exported_token" "false";
  exact semantic "exported_handle" "false";
  exact semantic "release_authority" "HostGuardian-only";
  exact material "schema" "loom-sovereign-execution-kernel-material-runtime-v1";
  exact material "stage" "MATERIAL_EXECUTION_FROZEN";
  exact material "action" "9042";
  exact material "grant_is_bearer" "false";
  exact material "same_uid_peer_isolation" "true";
  exact material "production_gate_ready" "true";
  exact material "causal_sabotage" "PASS";
  let sibling_runtime =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-sovereign-execution-kernel"
  in
  let runtime =
    if Sys.file_exists sibling_runtime then sibling_runtime
    else
      Filename.concat root
        "tools/loom/_build/default/src/sounio-loom-sovereign-execution-kernel"
  in
  let runtime_sha256 = Loom_exec.required semantic "executable_sha256" in
  if sha256_file runtime <> runtime_sha256 then
    failf "sovereign-authority-runtime-hash-mismatch";
  let frame =
    String.concat " "
      [ Loom_exec.required semantic "wire_schema";
        Loom_exec.required semantic "production_mode";
        Loom_exec.required semantic "production_stage";
        Loom_exec.required semantic "production_word";
        Loom_exec.required semantic "sabotage_count";
        Loom_exec.required semantic "sabotage_required" ]
    ^ "\n"
  in
  let expected = Loom_exec.required semantic "production_decision" in
  let result =
    Loom_exec.run_process ~input:frame ~environment ~cwd:root runtime []
  in
  let decision = Loom_exec.first_line result.output in
  if result.code <> 0 || decision <> expected then
    failf "sovereign-production-gate-refused:%d:%s" result.code decision;
  { semantic_sha256 = semantic_manifest_sha256;
    material_sha256 = material_manifest_sha256;
    runtime_sha256;
    production_decision = decision;
    production_decision_sha256 = sha256 decision }

type prepared = {
  payload : string;
  payload_sha256 : string;
  command_sha256 : string;
  classification_reason : string;
}

let record_with_digest fields =
  let body = String.concat "\n" fields ^ "\n" in
  body ^ "record_sha256=" ^ sha256 body ^ "\n"

let prepare ~root ~cwd ~event_sha256 ~command =
  let event_sha256 = require_digest "event-sha256" event_sha256 in
  let root = Unix.realpath root in
  let cwd = Loom_exec.canonical_directory cwd in
  if not (Loom_exec.within root cwd) then
    failf "sovereign-cwd-outside-worktree";
  let environment_bindings = Loom_exec.environment_bindings () in
  Loom_exec.ensure_safe_shell_bridge_environment environment_bindings;
  let environment_record =
    Loom_exec.environment_record_from environment_bindings
  in
  let execution_environment =
    Loom_exec.environment_array_from environment_bindings
  in
  let environment_sha256 = Loom_exec.environment_hash environment_record in
  let hardware_record = Loom_exec.hardware_record () in
  let hardware_sha256 = Loom_exec.hardware_hash hardware_record in
  let gate = load_gate root execution_environment in
  let policy = Loom_exec.load_policy root in
  let measurement = Loom_exec.measure_command cwd command in
  if measurement.argv = [] || measurement.executable = "" then
    failf "sovereign-empty-measurement:%s" measurement.classification_reason;
  let frame = Loom_exec.authority_frame policy measurement hardware_sha256 in
  let decision =
    try Loom_exec.invoke_authority root policy frame execution_environment with
    | Loom_exec.Authority_denied (code, output) ->
        failf "sovereign-preexec-denied:%d:%s" code output
  in
  let issued_us = Loom_exec.current_time_us () in
  let expires_us =
    Int64.add issued_us
      (Int64.mul (Int64.of_int (Loom_exec.capability_ttl_seconds ())) 1_000_000L)
  in
  let fixed =
    [ "schema=loom-sovereign-exec-request-v1";
      "event_sha256=" ^ event_sha256;
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
      "preexec_manifest_sha256=" ^ policy.manifest_sha256;
      "preexec_source_sha256=" ^ policy.source_sha256;
      "preexec_semantics_sha256=" ^ policy.semantics_sha256;
      "hardware_record_hex=" ^ hex_encode hardware_record;
      "hardware_sha256=" ^ hardware_sha256;
      "semantic_9042_sha256=" ^ gate.semantic_sha256;
      "material_9042_sha256=" ^ gate.material_sha256;
      "runtime_9042_sha256=" ^ gate.runtime_sha256;
      "production_gate_decision_sha256=" ^ gate.production_decision_sha256;
      "producing_language=" ^ Loom_exec.language_name measurement.language;
      "language_role=" ^ Loom_exec.language_role measurement.language;
      "language=" ^ string_of_int measurement.language;
      "purpose=" ^ string_of_int measurement.purpose;
      "surface=" ^ string_of_int measurement.surface;
      "execution_class=" ^ string_of_int measurement.execution_class;
      "closure_attested=" ^ string_of_int measurement.closure_attested;
      "argv_count=" ^ string_of_int (List.length measurement.argv) ]
  in
  let arguments =
    List.mapi
      (fun index argument ->
        Printf.sprintf "arg_%d_hex=%s" index (hex_encode argument))
      measurement.argv
  in
  let payload =
    record_with_digest
      (fixed @
       [ "frame_hex=" ^ hex_encode frame;
         "preexec_decision_hex=" ^ hex_encode decision;
         "production_gate_decision_hex=" ^ hex_encode gate.production_decision ]
       @ arguments)
  in
  if String.length payload > max_payload_bytes then
    failf "sovereign-payload-too-large";
  { payload; payload_sha256 = sha256 payload;
    command_sha256 = measurement.command_sha256;
    classification_reason = measurement.classification_reason }

let parse_record label content =
  if content = "" || content.[String.length content - 1] <> '\n' then
    failf "%s-missing-final-newline" label;
  let lines =
    String.split_on_char '\n' content |> List.filter (fun line -> line <> "")
  in
  let reversed = List.rev lines in
  let digest_line, body_lines =
    match reversed with
    | digest :: rest when starts_with digest "record_sha256=" ->
        (digest, List.rev rest)
    | _ -> failf "%s-record-digest-missing" label
  in
  let body = String.concat "\n" body_lines ^ "\n" in
  let expected =
    String.sub digest_line 14 (String.length digest_line - 14)
  in
  if sha256 body <> expected then failf "%s-record-digest-mismatch" label;
  let table = Hashtbl.create (List.length body_lines) in
  List.iter
    (fun line ->
      match String.index_opt line '=' with
      | None -> failf "%s-field-malformed" label
      | Some index ->
          let key = String.sub line 0 index in
          let value =
            String.sub line (index + 1) (String.length line - index - 1)
          in
          if Hashtbl.mem table key then failf "%s-field-duplicate:%s" label key;
          Hashtbl.add table key value)
    body_lines;
  (table, expected)

let record_field table key =
  match Hashtbl.find_opt table key with
  | Some value -> value
  | None -> failf "sovereign-result-field-missing:%s" key

let environment_from_record record =
  let lines = String.split_on_char '\n' record in
  let table = Hashtbl.create 64 in
  (match lines with
  | "schema=loom-execution-environment-v1" :: tail ->
      List.iter
        (fun line ->
          if line <> "" then
            match String.index_opt line '=' with
            | None -> failf "sovereign-environment-field-malformed"
            | Some index ->
                let name = String.sub line 0 index in
                let value =
                  String.sub line (index + 1) (String.length line - index - 1)
                in
                if not (Loom_exec.selected_environment_name name) then
                  failf "sovereign-environment-name-refused:%s" name;
                if value = "absent" then ()
                else if starts_with value "hex:" then
                  Hashtbl.add table name
                    (hex_decode "environment-value"
                       (String.sub value 4 (String.length value - 4)))
                else failf "sovereign-environment-value-malformed:%s" name)
        tail
  | _ -> failf "sovereign-environment-schema-mismatch");
  Loom_exec.ensure_safe_shell_bridge_environment table;
  if Loom_exec.environment_record_from table <> record then
    failf "sovereign-environment-canonicalization-mismatch";
  table

type validated = {
  event_sha256 : string;
  command_sha256 : string;
  payload_sha256 : string;
  root : string;
  cwd : string;
  environment : string array;
  environment_sha256 : string;
  hardware_sha256 : string;
  measurement : Loom_exec.measurement;
  preexec_policy : Loom_exec.policy;
  preexec_decision : string;
  production_gate_decision : string;
}

let validate_payload ~root ~event_sha256 ~command_sha256 payload =
  if String.length payload > max_payload_bytes then
    failf "sovereign-payload-too-large";
  let table, _ = parse_record "sovereign-request" payload in
  let payload_sha256 = sha256 payload in
  exact table "schema" "loom-sovereign-exec-request-v1";
  let event_sha256 = require_digest "event" event_sha256 in
  let command_sha256 = require_digest "command" command_sha256 in
  exact table "event_sha256" event_sha256;
  exact table "command_sha256" command_sha256;
  if require_int "uid" (Loom_exec.required table "uid") <> Unix.geteuid () then
    failf "sovereign-uid-mismatch";
  if Loom_exec.current_time_us () > require_int64 "expires-us" (Loom_exec.required table "expires_us") then
    failf "sovereign-request-expired";
  let root = Unix.realpath root in
  let recorded_root = hex_decode "root" (Loom_exec.required table "root_hex") in
  let cwd = hex_decode "cwd" (Loom_exec.required table "cwd_hex") |> Unix.realpath in
  if recorded_root <> root || not (Loom_exec.within root cwd) then
    failf "sovereign-root-or-cwd-mismatch";
  let environment_record =
    hex_decode "environment-record"
      (Loom_exec.required table "environment_record_hex")
  in
  let environment_table = environment_from_record environment_record in
  let environment = Loom_exec.environment_array_from environment_table in
  let environment_sha256 = Loom_exec.environment_hash environment_record in
  exact table "environment_sha256" environment_sha256;
  let hardware_record = Loom_exec.hardware_record () in
  let hardware_sha256 = Loom_exec.hardware_hash hardware_record in
  exact table "hardware_record_hex" (hex_encode hardware_record);
  exact table "hardware_sha256" hardware_sha256;
  let gate = load_gate root environment in
  exact table "semantic_9042_sha256" gate.semantic_sha256;
  exact table "material_9042_sha256" gate.material_sha256;
  exact table "runtime_9042_sha256" gate.runtime_sha256;
  exact table "production_gate_decision_sha256"
    gate.production_decision_sha256;
  exact table "production_gate_decision_hex"
    (hex_encode gate.production_decision);
  let preexec_policy = Loom_exec.load_policy root in
  exact table "preexec_manifest_sha256" preexec_policy.manifest_sha256;
  exact table "preexec_source_sha256" preexec_policy.source_sha256;
  exact table "preexec_semantics_sha256" preexec_policy.semantics_sha256;
  let command = hex_decode "command" (Loom_exec.required table "command_hex") in
  let measurement = Loom_exec.measure_command cwd command in
  if measurement.argv = [] || measurement.executable = "" then
    failf "sovereign-measurement-empty";
  exact table "executable_hex" (hex_encode measurement.executable);
  exact table "executable_sha256" measurement.executable_sha256;
  exact table "producing_language" (Loom_exec.language_name measurement.language);
  exact table "language_role" (Loom_exec.language_role measurement.language);
  exact table "language" (string_of_int measurement.language);
  exact table "purpose" (string_of_int measurement.purpose);
  exact table "surface" (string_of_int measurement.surface);
  exact table "execution_class" (string_of_int measurement.execution_class);
  exact table "closure_attested" (string_of_int measurement.closure_attested);
  exact table "broker_sha256" (sha256_file (Unix.realpath Sys.executable_name));
  let count = require_int "argv-count" (Loom_exec.required table "argv_count") in
  if count <> List.length measurement.argv then
    failf "sovereign-argv-count-mismatch";
  List.iteri
    (fun index argument ->
      exact table (Printf.sprintf "arg_%d_hex" index) (hex_encode argument))
    measurement.argv;
  let frame = Loom_exec.authority_frame preexec_policy measurement hardware_sha256 in
  exact table "frame_hex" (hex_encode frame);
  let preexec_decision =
    try Loom_exec.invoke_authority root preexec_policy frame environment with
    | Loom_exec.Authority_denied (code, output) ->
        failf "sovereign-preexec-denied:%d:%s" code output
  in
  exact table "preexec_decision_hex" (hex_encode preexec_decision);
  { event_sha256; command_sha256; payload_sha256; root; cwd; environment;
    environment_sha256; hardware_sha256; measurement; preexec_policy;
    preexec_decision; production_gate_decision = gate.production_decision }

let write_atomic path content =
  let directory = Filename.dirname path in
  mkdir_p directory;
  let temporary =
    Filename.concat directory
      (Printf.sprintf ".result.%d.%s" (Unix.getpid ())
         (Loom_exec.random_token ()))
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
      Unix.rename temporary path;
      Loom_exec.fsync_directory directory)

type material_result = {
  kind : int;
  exit_code : int;
  signal : int;
  elapsed_us : int64;
  stdout : string;
  stderr : string;
  guardian_revoked : bool;
  pdeathsig_armed : bool;
}

let run_material ~guardian_pid ~guardian_start validated =
  if process_start guardian_pid <> guardian_start then
    failf "sovereign-guardian-start-mismatch";
  let guardian_pidfd =
    match pidfd_open guardian_pid with
    | Some descriptor when pidfd_alive descriptor -> descriptor
    | Some descriptor ->
        Unix.close descriptor;
        failf "sovereign-guardian-not-alive"
    | None -> failf "sovereign-guardian-pidfd-unavailable"
  in
  let stdout_read, stdout_write = Unix.pipe ~cloexec:true () in
  let stderr_read, stderr_write = Unix.pipe ~cloexec:true () in
  let started_us = Loom_exec.current_time_us () in
  let child =
    match Unix.fork () with
    | 0 ->
        Unix.close guardian_pidfd;
        Unix.close stdout_read;
        Unix.close stderr_read;
        (try
           ignore (Unix.setsid ());
           arm_parent_death_kill ();
           Unix.chdir validated.cwd;
           Unix.dup2 stdout_write Unix.stdout;
           Unix.dup2 stderr_write Unix.stderr;
           if stdout_write <> Unix.stdout then Unix.close stdout_write;
           if stderr_write <> Unix.stderr then Unix.close stderr_write;
           Unix.execve validated.measurement.executable
             (Array.of_list validated.measurement.argv) validated.environment
         with _ -> Unix._exit 127)
    | pid -> pid
  in
  Unix.close stdout_write;
  Unix.close stderr_write;
  Unix.set_nonblock stdout_read;
  Unix.set_nonblock stderr_read;
  let stdout = Buffer.create 4096 in
  let stderr = Buffer.create 4096 in
  let stdout_open = ref true in
  let stderr_open = ref true in
  let status = ref None in
  let revoked = ref false in
  let append descriptor buffer =
    let bytes = Bytes.create 65536 in
    match Unix.read descriptor bytes 0 (Bytes.length bytes) with
    | 0 -> false
    | count ->
        if Buffer.length buffer + count > max_output_bytes then
          failf "sovereign-material-output-too-large";
        Buffer.add_subbytes buffer bytes 0 count;
        true
    | exception Unix_error ((EAGAIN | EWOULDBLOCK | EINTR), _, _) -> true
  in
  let kill_material () =
    (try Unix.kill (-child) Sys.sigkill with _ -> ());
    (try Unix.kill child Sys.sigkill with _ -> ())
  in
  let previous =
    List.map
      (fun signal ->
        let behavior =
          Sys.signal signal
            (Sys.Signal_handle (fun _ -> kill_material ()))
        in
        (signal, behavior))
      [ Sys.sigterm; Sys.sigint; Sys.sighup ]
  in
  let cleanup () =
    if !status = None then (
      kill_material ();
      (try
         let _, observed = Unix.waitpid [] child in
         status := Some observed
       with _ -> ()));
    List.iter (fun (signal, behavior) -> Sys.set_signal signal behavior) previous;
    (try Unix.close guardian_pidfd with _ -> ());
    (try Unix.close stdout_read with _ -> ());
    (try Unix.close stderr_read with _ -> ())
  in
  Fun.protect ~finally:cleanup (fun () ->
      while !status = None || !stdout_open || !stderr_open do
        if not (pidfd_alive guardian_pidfd) ||
           (try process_start guardian_pid <> guardian_start with _ -> true)
        then (
          revoked := true;
          kill_material ());
        (match Unix.waitpid [ WNOHANG ] child with
        | 0, _ -> ()
        | _, observed -> status := Some observed
        | exception Unix_error (ECHILD, _, _) ->
            if !status = None then status := Some (WEXITED 127));
        let reads =
          (if !stdout_open then [ stdout_read ] else []) @
          (if !stderr_open then [ stderr_read ] else []) @
          [ guardian_pidfd ]
        in
        let ready, _, _ = Unix.select reads [] [] 0.05 in
        if List.mem stdout_read ready then
          stdout_open := append stdout_read stdout;
        if List.mem stderr_read ready then
          stderr_open := append stderr_read stderr;
        if !status <> None && ready = [] then (
          if !stdout_open then stdout_open := append stdout_read stdout;
          if !stderr_open then stderr_open := append stderr_read stderr)
      done;
      let elapsed_us = Int64.sub (Loom_exec.current_time_us ()) started_us in
      let kind, exit_code, signal =
        match Option.get !status with
        | WEXITED code -> (1, code, 0)
        | WSIGNALED signal -> (2, 0, Loom_exec.linux_signal_number signal)
        | WSTOPPED signal -> (3, 0, Loom_exec.linux_signal_number signal)
      in
      { kind; exit_code; signal;
        elapsed_us = if elapsed_us < 0L then 0L else elapsed_us;
        stdout = Buffer.contents stdout; stderr = Buffer.contents stderr;
        guardian_revoked = !revoked; pdeathsig_armed = true })

let worker ~root ~event_sha256 ~command_sha256 ~payload ~job_id
    ~kernel_generation ~guardian_pid ~guardian_start ~result_path ~start_gate =
  let fail_record reason =
    let record =
      record_with_digest
        [ "schema=" ^ result_schema; "state=REFUSED"; "job_id=" ^ job_id;
          "event_sha256=" ^ event_sha256;
          "command_sha256=" ^ command_sha256;
          "payload_sha256=" ^ sha256 payload;
          "kernel_generation=" ^ kernel_generation;
          "guardian_pid=" ^ string_of_int guardian_pid;
          "guardian_start=" ^ guardian_start;
          "worker_pid=" ^ string_of_int (Unix.getpid ());
          "reason_hex=" ^ hex_encode reason;
          "material_started=false"; "material_completed=false";
          "guardian_revoked=false" ]
    in
    write_atomic result_path record
  in
  try
    let gate = Bytes.create 1 in
    let rec wait_gate () =
      match Unix.read start_gate gate 0 1 with
      | 1 when Bytes.get gate 0 = 'G' -> ()
      | 0 -> failf "sovereign-guardian-release-absent"
      | _ -> failf "sovereign-guardian-release-invalid"
      | exception Unix_error (EINTR, _, _) -> wait_gate ()
    in
    wait_gate ();
    Unix.close start_gate;
    let validated =
      validate_payload ~root ~event_sha256 ~command_sha256 payload
    in
    let result = run_material ~guardian_pid ~guardian_start validated in
    let material_body =
      String.concat "\n"
        [ "schema=" ^ result_schema;
          "state=" ^ (if result.guardian_revoked then "GUARDIAN_REVOKED" else "COMPLETED");
          "job_id=" ^ job_id;
          "event_sha256=" ^ validated.event_sha256;
          "command_sha256=" ^ validated.command_sha256;
          "payload_sha256=" ^ validated.payload_sha256;
          "kernel_generation=" ^ kernel_generation;
          "guardian_pid=" ^ string_of_int guardian_pid;
          "guardian_start=" ^ guardian_start;
          "worker_pid=" ^ string_of_int (Unix.getpid ());
          "executable_sha256=" ^ validated.measurement.executable_sha256;
          "environment_sha256=" ^ validated.environment_sha256;
          "hardware_sha256=" ^ validated.hardware_sha256;
          "semantic_9042_sha256=" ^ semantic_manifest_sha256;
          "material_9042_sha256=" ^ material_manifest_sha256;
          "preexec_decision_sha256=" ^ sha256 validated.preexec_decision;
          "production_gate_decision_sha256=" ^
            sha256 validated.production_gate_decision;
          "outcome_kind=" ^ string_of_int result.kind;
          "exit_code=" ^ string_of_int result.exit_code;
          "signal=" ^ string_of_int result.signal;
          "elapsed_us=" ^ Int64.to_string result.elapsed_us;
          "stdout_sha256=" ^ sha256 result.stdout;
          "stderr_sha256=" ^ sha256 result.stderr;
          "stdout_hex=" ^ hex_encode result.stdout;
          "stderr_hex=" ^ hex_encode result.stderr;
          "material_started=true";
          "material_completed=" ^ string_of_bool (not result.guardian_revoked);
          "guardian_revoked=" ^ string_of_bool result.guardian_revoked;
          "pdeathsig_armed=" ^ string_of_bool result.pdeathsig_armed ]
      ^ "\n"
    in
    let record =
      let fields =
        String.split_on_char '\n' material_body
        |> List.filter (fun line -> line <> "")
      in
      if result.guardian_revoked then
        record_with_digest (fields @ [ "outcome_authority_invoked=false" ])
      else
        let outcome_policy = Loom_exec.load_outcome_policy validated.root in
        let material_sha256 = sha256 material_body in
        let outcome_frame =
          Loom_exec.execution_outcome_frame outcome_policy
            ~outcome_kind:result.kind ~exit_code:result.exit_code
            ~signal:result.signal ~elapsed_us:result.elapsed_us
            ~hardware_sha256:validated.hardware_sha256
            ~command_sha256:validated.command_sha256
            ~environment_sha256:validated.environment_sha256
            ~executable_sha256:validated.measurement.executable_sha256
            ~grant_sha256:validated.payload_sha256
            ~generation_sha256:(sha256 kernel_generation)
            ~issue_decision_sha256:(sha256 validated.production_gate_decision)
            ~consume_decision_sha256:(sha256 validated.preexec_decision)
            ~result_sha256:material_sha256
        in
        let outcome_decision =
          try
            Loom_exec.invoke_outcome_authority validated.root outcome_policy
              outcome_frame validated.environment
          with Loom_exec.Authority_denied (code, output) ->
            failf "sovereign-outcome-denied:%d:%s" code output
        in
        record_with_digest
          (fields @
           [ "outcome_authority_invoked=true";
             "outcome_frame_sha256=" ^ sha256 outcome_frame;
             "outcome_decision_hex=" ^ hex_encode outcome_decision;
             "outcome_decision_sha256=" ^ sha256 outcome_decision ])
    in
    write_atomic result_path record;
    if result.guardian_revoked then 125 else 0
  with
  | Error reason | Loom_exec.Error reason ->
      (try fail_record reason with _ -> ());
      126
  | Loom_exec.Authority_denied (code, output) ->
      (try fail_record (Printf.sprintf "authority-denied:%d:%s" code output)
       with _ -> ());
      126
  | Sys_error reason ->
      (try fail_record reason with _ -> ());
      126
  | Unix_error (error, name, argument) ->
      let reason =
        Printf.sprintf "%s:%s(%s)" (Unix.error_message error) name argument
      in
      (try fail_record reason with _ -> ());
      126

let validate_result_file ~path ~job_id ~payload_sha256 =
  let info = Unix.lstat path in
  if info.st_kind <> S_REG || info.st_uid <> Unix.geteuid () ||
     info.st_perm land 0o077 <> 0
  then failf "sovereign-result-file-unsafe";
  let content =
    read_file ~limit:(max_output_bytes * 4 + 1024 * 1024) path
  in
  let table, record_sha256 = parse_record "sovereign-result" content in
  exact table "schema" result_schema;
  exact table "job_id" job_id;
  exact table "payload_sha256" payload_sha256;
  (table, record_sha256, content)

let connect_socket () =
  let path =
    match Sys.getenv_opt "SOUNIO_LOOM_SOCKET" with
    | Some value when value <> "" -> value
    | _ -> failf "sovereign-kernel-socket-missing"
  in
  let descriptor = Unix.socket PF_UNIX SOCK_STREAM 0 in
  try
    Unix.connect descriptor (ADDR_UNIX path);
    descriptor
  with error ->
    Unix.close descriptor;
    raise error

let read_line descriptor =
  let buffer = Buffer.create 512 in
  let byte = Bytes.create 1 in
  let rec loop () =
    match Unix.read descriptor byte 0 1 with
    | 0 -> failf "sovereign-kernel-response-eof"
    | 1 ->
        let character = Bytes.get byte 0 in
        if character = '\n' then Buffer.contents buffer
        else (
          if Buffer.length buffer >= 65536 then
            failf "sovereign-kernel-response-too-large";
          Buffer.add_char buffer character;
          loop ())
    | _ -> assert false
    | exception Unix_error (EINTR, _, _) -> loop ()
  in
  loop ()

let kernel_exchange fields =
  let descriptor = connect_socket () in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      write_all descriptor (String.concat "\t" ("LOOM_EXEC/1" :: fields) ^ "\n");
      read_line descriptor |> String.split_on_char '\t')

type started = {
  instance : string;
  generation : string;
  job_id : string;
  payload_sha256 : string;
}

let start ~event_sha256 (prepared : prepared) =
  let instance =
    match Sys.getenv_opt "SOUNIO_LOOM_INSTANCE_ID" with
    | Some value when value <> "" -> value
    | _ -> failf "sovereign-instance-missing"
  in
  match
    kernel_exchange
      [ "START"; instance; event_sha256; prepared.command_sha256;
        prepared.payload_sha256; hex_encode prepared.payload ]
  with
  | [ "OK"; "SOVEREIGN_STARTED"; actual_instance; generation; job_id;
      payload_sha256 ]
    when actual_instance = instance && payload_sha256 = prepared.payload_sha256
         && valid_sha256 generation && valid_sha256 job_id ->
      { instance; generation; job_id; payload_sha256 }
  | "ERR" :: reason :: _ -> failf "sovereign-kernel-refused:%s" reason
  | _ -> failf "sovereign-kernel-start-response-invalid"

let presentation_command started =
  String.concat " "
    [ Loom_exec.shell_quote (Unix.realpath Sys.executable_name);
      "sovereign-result"; "--instance"; started.instance;
      "--generation"; started.generation; "--job"; started.job_id;
      "--payload-sha256"; started.payload_sha256 ]

type wait_state = Pending | Complete of string * string | Failed of string

let wait_once ~instance ~generation ~job_id ~payload_sha256 =
  match
    kernel_exchange
      [ "WAIT"; instance; generation; job_id; payload_sha256 ]
  with
  | [ "OK"; "SOVEREIGN_PENDING"; actual_job ] when actual_job = job_id -> Pending
  | [ "OK"; "SOVEREIGN_COMPLETE"; actual_job; path_hex; record_sha256 ]
    when actual_job = job_id && valid_sha256 record_sha256 ->
      Complete (hex_decode "result-path" path_hex, record_sha256)
  | [ "ERR"; reason ] -> Failed reason
  | _ -> failf "sovereign-kernel-wait-response-invalid"

let present_result ~instance ~generation ~job_id ~payload_sha256 =
  List.iter (fun (label, value) -> ignore (require_digest label value))
    [ "generation", generation; "job", job_id; "payload", payload_sha256 ];
  let deadline = Unix.gettimeofday () +. 300.0 in
  let rec wait () =
    if Unix.gettimeofday () >= deadline then failf "sovereign-result-timeout";
    match wait_once ~instance ~generation ~job_id ~payload_sha256 with
    | Pending -> Unix.sleepf 0.05; wait ()
    | Failed reason -> failf "sovereign-result-refused:%s" reason
    | Complete (path, expected_sha256) ->
        let table, record_sha256, _ =
          validate_result_file ~path ~job_id ~payload_sha256
        in
        if record_sha256 <> expected_sha256 then
          failf "sovereign-result-hash-mismatch";
        let state = Loom_exec.required table "state" in
        if state <> "COMPLETED" then failf "sovereign-result-state:%s" state;
        let stdout_text =
          hex_decode "stdout" (record_field table "stdout_hex")
        in
        let stderr_text =
          hex_decode "stderr" (record_field table "stderr_hex")
        in
        exact table "stdout_sha256" (sha256 stdout_text);
        exact table "stderr_sha256" (sha256 stderr_text);
        output_string Stdlib.stdout stdout_text;
        output_string Stdlib.stderr stderr_text;
        flush Stdlib.stdout;
        flush Stdlib.stderr;
        let kind = require_int "outcome-kind" (Loom_exec.required table "outcome_kind") in
        let exit_code = require_int "exit-code" (Loom_exec.required table "exit_code") in
        let signal = require_int "signal" (Loom_exec.required table "signal") in
        if kind = 1 then exit_code else if kind = 2 then 128 + signal else 126
  in
  wait ()
