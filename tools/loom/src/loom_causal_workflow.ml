open Unix

exception Error of string

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let pinned_manifest_sha256 =
  "ef8dc0fadeb1cb33b1ba31d77551b8ca962bc97d1204998057b071b9b7921437"

let journal_domain = "loom-causal-workflow-journal-v1"
let zero_digest = String.make 64 '0'

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let sha256_file path =
  let channel = open_in_bin path in
  let hash = Cryptokit.Hash.sha256 () in
  let bytes = Bytes.create 65536 in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let rec loop () =
        match input channel bytes 0 (Bytes.length bytes) with
        | 0 -> ()
        | count ->
            hash#add_substring bytes 0 count;
            loop ()
      in
      loop ());
  hash#result |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let valid_digest value =
  String.length value = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let digest label value =
  let value = String.lowercase_ascii value in
  if not (valid_digest value) || value = zero_digest then
    failf "causal-workflow-%s-digest-invalid" label;
  value

let validate_atom label value =
  if value = "" || String.length value > 256 then
    failf "causal-workflow-%s-invalid" label;
  String.iter
    (function
      | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '.' | '_' | '-' -> ()
      | _ -> failf "causal-workflow-%s-invalid" label)
    value

let parse_manifest path =
  let table = Hashtbl.create 128 in
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let rec loop () =
        match input_line channel with
        | line ->
            (match String.index_opt line '=' with
            | None -> failf "causal-workflow-manifest-line-malformed"
            | Some index ->
                let key = String.sub line 0 index in
                let value =
                  String.sub line (index + 1) (String.length line - index - 1)
                in
                if Hashtbl.mem table key then
                  failf "causal-workflow-manifest-field-duplicate:%s" key;
                Hashtbl.add table key value;
                loop ())
        | exception End_of_file -> table
      in
      loop ())

let required table key =
  match Hashtbl.find_opt table key with
  | Some value -> value
  | None -> failf "causal-workflow-manifest-field-missing:%s" key

let exact table key expected =
  let actual = required table key in
  if actual <> expected then
    failf "causal-workflow-manifest-field-mismatch:%s" key

let require_regular_file path =
  let stat = Unix.lstat path in
  if stat.st_kind <> S_REG || stat.st_nlink <> 1 then
    failf "causal-workflow-file-not-private-regular:%s" path;
  stat

type policy = {
  manifest_sha256 : string;
  semantics_sha256 : string;
  runtime : string;
}

let load_policy ~repo_root =
  let repo_root = Unix.realpath repo_root in
  let manifest_path =
    match Sys.getenv_opt "SOUNIO_LOOM_CAUSAL_WORKFLOW_MANIFEST" with
    | Some path -> path
    | None -> Filename.concat repo_root "tools/loom/causal_workflow_kernel.freeze.v1"
  in
  ignore (require_regular_file manifest_path);
  if sha256_file manifest_path <> pinned_manifest_sha256 then
    failf "causal-workflow-manifest-hash-mismatch";
  let manifest = parse_manifest manifest_path in
  exact manifest "schema" "loom-causal-workflow-kernel-freeze-v1";
  exact manifest "stage" "SEMANTICS_FROZEN";
  exact manifest "producing_language" "Sounio";
  exact manifest "language_role" "SEMANTIC_AUTHORITY";
  exact manifest "action" "9037";
  exact manifest "concept_id" "SOUNIO-LOOM-CAUSAL-WORKFLOW-KERNEL";
  exact manifest "run_ticket_is_bearer" "false";
  exact manifest "run_ticket_is_execution_authority" "false";
  exact manifest "launch_authority" "action-9030";
  exact manifest "exactly_once_scope" "live-HostGuardian-generation";
  exact manifest "guardian_host_or_store_loss" "fail-closed";
  exact manifest "ocaml_journal_attached" "false";
  exact manifest "material_execution" "false";
  exact manifest "parity_open" "false";
  exact manifest "claim_ready" "false";
  List.iter
    (fun key ->
      let path = Filename.concat repo_root (required manifest (key ^ "_path")) in
      ignore (require_regular_file path);
      if sha256_file path <> required manifest (key ^ "_sha256") then
        failf "causal-workflow-%s-hash-mismatch" key)
    [ "garden"; "contract"; "source"; "entrypoint"; "build_script";
      "selftest"; "first_manifest"; "first_evidence"; "freeze_evidence";
      "parent_9030_manifest"; "parent_9031_manifest";
      "parent_9032_manifest"; "parent_9033_manifest";
      "parent_9034_manifest"; "parent_9035_manifest";
      "parent_9036_manifest"; "toolchain_wrapper"; "toolchain_compiler" ];
  let runtime =
    match Sys.getenv_opt "SOUNIO_LOOM_CAUSAL_WORKFLOW_RUNTIME" with
    | Some path -> path
    | None ->
        Filename.concat repo_root
          "tools/loom/_build/default/src/sounio-loom-causal-workflow-kernel"
  in
  let runtime = Unix.realpath runtime in
  let stat = require_regular_file runtime in
  if stat.st_perm land 0o111 = 0 then
    failf "causal-workflow-runtime-not-executable";
  if sha256_file runtime <> required manifest "executable_sha256" then
    failf "causal-workflow-runtime-hash-mismatch";
  { manifest_sha256 = pinned_manifest_sha256;
    semantics_sha256 = required manifest "semantics_sha256" |> digest "semantics";
    runtime }

let write_all descriptor value =
  let bytes = Bytes.of_string value in
  let rec loop offset =
    if offset < Bytes.length bytes then
      match Unix.write descriptor bytes offset (Bytes.length bytes - offset) with
      | 0 -> failf "causal-workflow-short-write"
      | count -> loop (offset + count)
      | exception Unix_error (EINTR, _, _) -> loop offset
  in
  loop 0

let process_exchange executable input =
  let input_read, input_write = Unix.pipe () in
  let output_read, output_write = Unix.pipe () in
  Unix.set_close_on_exec input_write;
  Unix.set_close_on_exec output_read;
  let pid =
    Unix.create_process executable [| executable |] input_read output_write
      output_write
  in
  Unix.close input_read;
  Unix.close output_write;
  let close descriptor = try Unix.close descriptor with _ -> () in
  Fun.protect
    ~finally:(fun () -> close input_write; close output_read)
    (fun () ->
      write_all input_write input;
      close input_write;
      let buffer = Buffer.create 2048 in
      let bytes = Bytes.create 4096 in
      let rec read () =
        match Unix.read output_read bytes 0 (Bytes.length bytes) with
        | 0 -> ()
        | count ->
            if Buffer.length buffer + count > 64 * 1024 then
              failf "causal-workflow-authority-output-too-large";
            Buffer.add_subbytes buffer bytes 0 count;
            read ()
        | exception Unix_error (EINTR, _, _) -> read ()
      in
      read ();
      let _, status = Unix.waitpid [] pid in
      let code =
        match status with
        | WEXITED value -> value
        | WSIGNALED signal | WSTOPPED signal -> 128 + signal
      in
      let output = Buffer.contents buffer in
      let output =
        if output <> "" && output.[String.length output - 1] = '\n' then
          String.sub output 0 (String.length output - 1)
        else output
      in
      (code, output))

let set_bit word bit = Int64.logor word (Int64.shift_left 1L bit)

let set_range word first last =
  let rec loop word bit =
    if bit > last then word else loop (set_bit word bit) (bit + 1)
  in
  loop word first

let level current next = max current next

let expected_exit_code = 0

let expected_empty_sha256 =
  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

let authority_field fields key =
  match Hashtbl.find_opt fields key with
  | Some value -> value
  | None -> failf "causal-workflow-payload-field-missing:%s" key

let authority_bool fields key =
  match authority_field fields key with
  | "true" -> true
  | "false" -> false
  | _ -> failf "causal-workflow-payload-boolean-invalid:%s" key

let authority_words ~transition ~current ~next ~fields =
  let reached = level current next in
  let word0 = set_range 0L 0 19 in
  let word0 =
    Int64.logor word0 (Int64.shift_left (Int64.of_int transition) 20)
  in
  let word0 = Int64.logor word0 (Int64.shift_left (Int64.of_int current) 24) in
  let word0 = Int64.logor word0 (Int64.shift_left (Int64.of_int next) 28) in
  let word0 = set_bit word0 32 |> fun value -> set_bit value 44 in
  let word0 = set_range word0 45 56 in
  let word0 = if reached >= 3 then set_range word0 33 37 else word0 in
  let word0 = if reached >= 6 then set_range word0 38 40 else word0 in
  let word0 = if reached >= 10 then set_range word0 41 43 else word0 in
  let word0 =
    if reached < 6 then word0
    else if transition <> 6 then set_range word0 57 59
    else
      let value =
        if authority_field fields "exit_code" = string_of_int expected_exit_code then
          set_bit word0 57
        else word0
      in
      let value =
        if authority_field fields "stdout_sha256" = expected_empty_sha256 then
          set_bit value 58
        else value
      in
      if authority_field fields "stderr_sha256" = expected_empty_sha256 then
        set_bit value 59
      else value
  in
  let word0 =
    if reached < 7 then word0
    else if transition <> 7 ||
            (authority_bool fields "pid_extinct" &&
             authority_bool fields "descendants_extinct" &&
             authority_bool fields "cgroup_unit_extinct" &&
             authority_bool fields "grant_extinct" &&
             authority_bool fields "capsule_extinct") then
      set_bit word0 60
    else word0
  in
  let word0 = if reached >= 10 then set_bit word0 61 else word0 in
  let word1 = set_range 0L 33 40 |> fun value -> set_range value 49 51 in
  let word1 = if reached >= 3 then set_range word1 3 5 |> fun value -> set_bit value 0 else word1 in
  let word1 =
    if reached >= 3 then set_bit word1 44 |> fun value -> set_bit value 46
    else word1
  in
  let word1 =
    if reached >= 4 then
      word1 |> fun value -> set_bit value 6 |> fun value -> set_bit value 7
      |> fun value -> set_range value 9 11 |> fun value -> set_bit value 41
      |> fun value -> set_bit value 42 |> fun value -> set_bit value 45
    else word1
  in
  let word1 =
    if reached >= 5 then
      word1 |> fun value -> set_bit value 13 |> fun value -> set_bit value 15
    else word1
  in
  let word1 = if transition = 5 then set_bit word1 12 else word1 in
  let word1 =
    if reached >= 6 then
      word1 |> fun value -> set_bit value 16 |> fun value -> set_bit value 43
      |> fun value -> set_bit value 47
    else word1
  in
  let word1 =
    if reached < 7 then word1
    else if transition <> 7 then set_range word1 17 21
    else
      let value = if authority_bool fields "pid_extinct" then set_bit word1 17 else word1 in
      let value =
        if authority_bool fields "descendants_extinct" then set_bit value 18 else value
      in
      let value =
        if authority_bool fields "cgroup_unit_extinct" then set_bit value 19 else value
      in
      let value = if authority_bool fields "grant_extinct" then set_bit value 20 else value in
      if authority_bool fields "capsule_extinct" then set_bit value 21 else value
  in
  let word1 = if reached >= 10 then set_bit word1 48 else word1 in
  let word1 =
    if transition = 11 then
      let value = set_range word1 22 28 in
      let value = if current >= 4 then set_bit value 29 else value in
      let value = if current >= 5 then set_bit value 30 else value in
      let value = if current >= 6 then set_bit value 31 else value in
      if current >= 8 then set_bit value 32 else value
    else word1
  in
  (word0, word1)

let starts_with value prefix =
  String.length value >= String.length prefix
  && String.sub value 0 (String.length prefix) = prefix

let authorize_transition (policy : policy) ~transition ~current ~next ~fields =
  let word0, word1 = authority_words ~transition ~current ~next ~fields in
  let frame = Printf.sprintf "9037 %Ld %Ld\n" word0 word1 in
  let code, output = process_exchange policy.runtime frame in
  let decision =
    if transition = 11 then
      "SOUNIO_CAUSAL_WORKFLOW RECOVER semantic_authority=Sounio action=9037"
    else "SOUNIO_CAUSAL_WORKFLOW ADVANCE semantic_authority=Sounio action=9037"
  in
  if code <> 0 || not (starts_with output (decision ^ "\n")) then
    failf "causal-workflow-sounio-refused:%d:%s" code output;
  sha256 output

let hex_of_string value =
  let alphabet = "0123456789abcdef" in
  let output = Bytes.create (String.length value * 2) in
  String.iteri
    (fun index character ->
      let code = Char.code character in
      Bytes.set output (index * 2) alphabet.[code lsr 4];
      Bytes.set output ((index * 2) + 1) alphabet.[code land 0x0f])
    value;
  Bytes.unsafe_to_string output

let hex_value = function
  | '0' .. '9' as value -> Char.code value - Char.code '0'
  | 'a' .. 'f' as value -> 10 + Char.code value - Char.code 'a'
  | _ -> failf "causal-workflow-invalid-hex"

let string_of_hex value =
  if String.length value mod 2 <> 0 then failf "causal-workflow-invalid-hex";
  let output = Bytes.create (String.length value / 2) in
  for index = 0 to Bytes.length output - 1 do
    let high = hex_value value.[index * 2] in
    let low = hex_value value.[(index * 2) + 1] in
    Bytes.set output index (Char.chr ((high lsl 4) lor low))
  done;
  Bytes.unsafe_to_string output

let encode_fields fields =
  fields
  |> List.map (fun (key, value) ->
         validate_atom "payload-key" key;
         if String.contains value '\000' then
           failf "causal-workflow-payload-value-invalid";
         key ^ "=" ^ value)
  |> String.concat "\000"

let decode_fields payload =
  let table = Hashtbl.create 32 in
  String.split_on_char '\000' payload
  |> List.iter (fun encoded ->
         match String.index_opt encoded '=' with
         | None -> failf "causal-workflow-payload-field-malformed"
         | Some index ->
             let key = String.sub encoded 0 index in
             let value =
               String.sub encoded (index + 1) (String.length encoded - index - 1)
             in
             if Hashtbl.mem table key then
               failf "causal-workflow-payload-field-duplicate:%s" key;
             Hashtbl.add table key value);
  table

let field fields key =
  match Hashtbl.find_opt fields key with
  | Some value -> value
  | None -> failf "causal-workflow-payload-field-missing:%s" key

let int_field fields key =
  try int_of_string (field fields key)
  with _ -> failf "causal-workflow-payload-integer-invalid:%s" key

let bool_field fields key =
  match field fields key with
  | "true" -> true
  | "false" -> false
  | _ -> failf "causal-workflow-payload-boolean-invalid:%s" key

let fsync_directory path =
  let descriptor = Unix.openfile path [ O_RDONLY ] 0 in
  Unix.set_close_on_exec descriptor;
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () -> Unix.fsync descriptor)

let rec mkdir_p path =
  if path = "" || path = "." || path = "/" || Sys.file_exists path then ()
  else (
    mkdir_p (Filename.dirname path);
    Unix.mkdir path 0o700;
    fsync_directory (Filename.dirname path))

let utc_now () =
  let value = Unix.gettimeofday () in
  let tm = Unix.gmtime value in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02dZ"
    (tm.tm_year + 1900) (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min
    tm.tm_sec

let store_dir root = Filename.concat root "causal-workflows"
let lock_path root = Filename.concat (store_dir root) "journal.lock"

let journal_path root workflow_key =
  Filename.concat (store_dir root) (workflow_key ^ ".journal")

let with_lock root action =
  mkdir_p (store_dir root);
  let descriptor = Unix.openfile (lock_path root) [ O_RDWR; O_CREAT ] 0o600 in
  Unix.set_close_on_exec descriptor;
  Fun.protect
    ~finally:(fun () ->
      (try Unix.lockf descriptor F_ULOCK 0 with _ -> ());
      Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      action ())

type event = {
  sequence : int;
  observed_at_utc : string;
  previous_sha256 : string;
  semantics_sha256 : string;
  workflow_key : string;
  kind : string;
  payload : string;
  event_sha256 : string;
}

let event_body sequence observed previous semantics workflow_key kind payload_hex =
  String.concat "\t"
    [ string_of_int sequence; observed; previous; semantics; workflow_key; kind;
      payload_hex ]

let event_digest body = sha256 (journal_domain ^ "\000" ^ body)

let make_event ~sequence ~previous ~semantics ~workflow_key ~kind ~payload =
  validate_atom "event-kind" kind;
  let observed = utc_now () in
  let body =
    event_body sequence observed previous semantics workflow_key kind
      (hex_of_string payload)
  in
  { sequence; observed_at_utc = observed; previous_sha256 = previous;
    semantics_sha256 = semantics; workflow_key; kind; payload;
    event_sha256 = event_digest body }

let encode_event event =
  let body =
    event_body event.sequence event.observed_at_utc event.previous_sha256
      event.semantics_sha256 event.workflow_key event.kind
      (hex_of_string event.payload)
  in
  body ^ "\t" ^ event.event_sha256 ^ "\n"

let parse_event (policy : policy) expected_sequence expected_previous expected_key line =
  match String.split_on_char '\t' line with
  | [ sequence; observed; previous; semantics; workflow_key; kind; payload_hex;
      event_sha256 ] ->
      let sequence =
        try int_of_string sequence
        with _ -> failf "causal-workflow-journal-sequence-invalid"
      in
      if sequence <> expected_sequence then
        failf "causal-workflow-journal-sequence-gap";
      if previous <> expected_previous then
        failf "causal-workflow-journal-previous-mismatch";
      if semantics <> policy.semantics_sha256 then
        failf "causal-workflow-journal-semantics-mismatch";
      if workflow_key <> expected_key then
        failf "causal-workflow-journal-workflow-mismatch";
      let body =
        event_body sequence observed previous semantics workflow_key kind payload_hex
      in
      if event_digest body <> event_sha256 then
        failf "causal-workflow-journal-event-digest-mismatch";
      { sequence; observed_at_utc = observed; previous_sha256 = previous;
        semantics_sha256 = semantics; workflow_key; kind;
        payload = string_of_hex payload_hex; event_sha256 }
  | _ -> failf "causal-workflow-journal-record-malformed"

type phase =
  | Waiting_compile | Compile_armed | Compile_running | Compiled_closed
  | Run_armed | Running | Run_measured | Run_closed | Attest_armed
  | Attest_running | Attested_closed | Refused_poisoned

let phase_code = function
  | Waiting_compile -> 0 | Compile_armed -> 1 | Compile_running -> 2
  | Compiled_closed -> 3 | Run_armed -> 4 | Running -> 5
  | Run_measured -> 6 | Run_closed -> 7 | Attest_armed -> 8
  | Attest_running -> 9 | Attested_closed -> 10 | Refused_poisoned -> 11

let phase_name = function
  | Waiting_compile -> "WAITING_COMPILE" | Compile_armed -> "COMPILE_ARMED"
  | Compile_running -> "COMPILE_RUNNING" | Compiled_closed -> "COMPILED_CLOSED"
  | Run_armed -> "RUN_ARMED" | Running -> "RUNNING"
  | Run_measured -> "RUN_MEASURED" | Run_closed -> "RUN_CLOSED"
  | Attest_armed -> "ATTEST_ARMED" | Attest_running -> "ATTEST_RUNNING"
  | Attested_closed -> "ATTESTED_CLOSED" | Refused_poisoned -> "REFUSED_POISONED"

let phase_of_code = function
  | 0 -> Waiting_compile | 1 -> Compile_armed | 2 -> Compile_running
  | 3 -> Compiled_closed | 4 -> Run_armed | 5 -> Running
  | 6 -> Run_measured | 7 -> Run_closed | 8 -> Attest_armed
  | 9 -> Attest_running | 10 -> Attested_closed | 11 -> Refused_poisoned
  | _ -> failf "causal-workflow-phase-invalid"

type snapshot = {
  workflow_id : string;
  workflow_key : string;
  workflow_generation : string;
  guardian_generation : string;
  journal_id : string;
  store_id : string;
  controller_generation : string;
  source_sha256 : string;
  phase : phase;
  compile_count : int;
  ticket_count : int;
  launch_count : int;
  compile_receipt : string option;
  artifact_record : string option;
  artifact_handle : string option;
  run_ticket : string option;
  run_grant : string option;
  run_grant_generation : string option;
  start_receipt : string option;
  run_pid_identity : string option;
  result_record : string option;
  result_handle : string option;
  exit_code : int option;
  stdout_sha256 : string option;
  stderr_sha256 : string option;
  attestation_record : string option;
  attestation_handle : string option;
  sequence : int;
  head_sha256 : string;
}

let initial_snapshot event =
  if event.kind <> "WORKFLOW_OPENED" || event.sequence <> 1 ||
     event.previous_sha256 <> zero_digest then
    failf "causal-workflow-open-event-invalid";
  let fields = decode_fields event.payload in
  let workflow_id = field fields "workflow_id" in
  validate_atom "workflow-id" workflow_id;
  if sha256 workflow_id <> event.workflow_key then
    failf "causal-workflow-key-mismatch";
  { workflow_id; workflow_key = event.workflow_key;
    workflow_generation = field fields "workflow_generation" |> digest "workflow-generation";
    guardian_generation = field fields "guardian_generation" |> digest "guardian-generation";
    journal_id = field fields "journal_id" |> digest "journal-id";
    store_id = field fields "store_id" |> digest "store-id";
    controller_generation = field fields "controller_generation" |> digest "controller-generation";
    source_sha256 = field fields "source_sha256" |> digest "source";
    phase = Waiting_compile; compile_count = 0; ticket_count = 0;
    launch_count = 0; compile_receipt = None; artifact_record = None;
    artifact_handle = None; run_ticket = None; run_grant = None;
    run_grant_generation = None; start_receipt = None;
    run_pid_identity = None; result_record = None; result_handle = None;
    exit_code = None; stdout_sha256 = None; stderr_sha256 = None;
    attestation_record = None; attestation_handle = None;
    sequence = event.sequence; head_sha256 = event.event_sha256 }

let require_option label = function
  | Some value -> value
  | None -> failf "causal-workflow-%s-missing" label

let apply_transition (policy : policy) (snapshot : snapshot) (event : event) =
  let fields = decode_fields event.payload in
  let transition = int_field fields "transition" in
  let current = int_field fields "current_state" in
  let next = int_field fields "next_state" in
  if current <> phase_code snapshot.phase then
    failf "causal-workflow-current-state-mismatch";
  if field fields "observed_predecessor_sha256" <> snapshot.head_sha256 then
    failf "causal-workflow-predecessor-receipt-mismatch";
  ignore (authorize_transition policy ~transition ~current ~next ~fields);
  let next_phase = phase_of_code next in
  let base = { snapshot with phase = next_phase; sequence = event.sequence;
    head_sha256 = event.event_sha256 } in
  match event.kind, transition with
  | "COMPILE_ARMED", 1 -> base
  | "COMPILE_STARTED", 2 -> base
  | "COMPILE_CLOSED", 3 ->
      if snapshot.compile_count <> 0 then failf "causal-workflow-recompile-refused";
      { base with compile_count = 1;
        compile_receipt = Some (field fields "compile_receipt" |> digest "compile-receipt");
        artifact_record = Some (field fields "artifact_record" |> digest "artifact-record");
        artifact_handle = Some (field fields "artifact_handle" |> digest "artifact-handle") }
  | "RUN_TICKET_COMMITTED", 4 ->
      if snapshot.ticket_count <> 0 || snapshot.run_ticket <> None then
        failf "causal-workflow-duplicate-ticket-refused";
      let artifact_record = field fields "artifact_record" |> digest "artifact-record" in
      if artifact_record <> require_option "artifact-record" snapshot.artifact_record then
        failf "causal-workflow-ticket-artifact-mismatch";
      { base with ticket_count = 1;
        run_ticket = Some (field fields "run_ticket" |> digest "run-ticket");
        run_grant = Some (field fields "run_grant" |> digest "run-grant");
        run_grant_generation =
          Some (field fields "run_grant_generation" |> digest "run-grant-generation") }
  | "RUN_LAUNCHED", 5 ->
      if snapshot.launch_count <> 0 then failf "causal-workflow-replay-launch-refused";
      let ticket = field fields "run_ticket" |> digest "run-ticket" in
      if ticket <> require_option "run-ticket" snapshot.run_ticket then
        failf "causal-workflow-run-ticket-mismatch";
      { base with launch_count = 1;
        start_receipt = Some (field fields "start_receipt" |> digest "start-receipt");
        run_pid_identity = Some (field fields "run_pid_identity" |> digest "run-pid") }
  | "RUN_RESULT_SEALED", 6 ->
      let exit_code = int_field fields "exit_code" in
      if exit_code < 0 || exit_code > 255 then
        failf "causal-workflow-exit-code-invalid";
      let stdout_sha256 = field fields "stdout_sha256" |> digest "stdout" in
      let stderr_sha256 = field fields "stderr_sha256" |> digest "stderr" in
      { base with
        result_record = Some (field fields "result_record" |> digest "result-record");
        result_handle = Some (field fields "result_handle" |> digest "result-handle");
        exit_code = Some exit_code; stdout_sha256 = Some stdout_sha256;
        stderr_sha256 = Some stderr_sha256 }
  | "RUN_CLOSED", 7 ->
      if not (bool_field fields "pid_extinct" &&
              bool_field fields "descendants_extinct" &&
              bool_field fields "cgroup_unit_extinct" &&
              bool_field fields "grant_extinct" &&
              bool_field fields "capsule_extinct") then
        failf "causal-workflow-run-extinction-incomplete";
      base
  | "ATTEST_ARMED", 8 -> base
  | "ATTEST_STARTED", 9 -> base
  | "ATTEST_CLOSED", 10 ->
      let artifact_record = field fields "artifact_record" |> digest "artifact-record" in
      let result_record = field fields "result_record" |> digest "result-record" in
      if artifact_record <> require_option "artifact-record" snapshot.artifact_record ||
         result_record <> require_option "result-record" snapshot.result_record then
        failf "causal-workflow-attestation-lineage-mismatch";
      { base with
        attestation_record =
          Some (field fields "attestation_record" |> digest "attestation-record");
        attestation_handle =
          Some (field fields "attestation_handle" |> digest "attestation-handle") }
  | "CONTROLLER_RECOVERED", 11 ->
      if not (bool_field fields "predecessor_controller_extinct") then
        failf "causal-workflow-predecessor-controller-live";
      let guardian = field fields "guardian_generation" |> digest "guardian-generation" in
      let journal = field fields "journal_id" |> digest "journal-id" in
      let store = field fields "store_id" |> digest "store-id" in
      if guardian <> snapshot.guardian_generation || journal <> snapshot.journal_id ||
         store <> snapshot.store_id then
        failf "causal-workflow-recovery-custody-mismatch";
      { base with controller_generation =
          field fields "successor_controller_generation"
          |> digest "successor-controller-generation" }
  | _ -> failf "causal-workflow-event-transition-mismatch:%s" event.kind

let read_lines path =
  let channel = open_in_bin path in
  let rec loop values =
    match input_line channel with
    | line -> loop (line :: values)
    | exception End_of_file -> List.rev values
  in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () -> loop [])

let load_snapshot_unlocked (policy : policy) ~state_root ~workflow_id =
  validate_atom "workflow-id" workflow_id;
  let workflow_key = sha256 workflow_id in
  let path = journal_path state_root workflow_key in
  let stat = require_regular_file path in
  if stat.st_perm land 0o077 <> 0 || stat.st_uid <> Unix.geteuid () then
    failf "causal-workflow-journal-permission-mismatch";
  let lines = read_lines path in
  match lines with
  | [] -> failf "causal-workflow-journal-empty"
  | first :: rest ->
      let first = parse_event policy 1 zero_digest workflow_key first in
      let initial = initial_snapshot first in
      let rec replay snapshot = function
        | [] -> snapshot
        | line :: remaining ->
            let event =
              parse_event policy (snapshot.sequence + 1) snapshot.head_sha256
                workflow_key line
            in
            replay (apply_transition policy snapshot event) remaining
      in
      replay initial rest

let load_snapshot ~repo_root ~state_root ~workflow_id =
  let policy = load_policy ~repo_root in
  with_lock state_root (fun () ->
      load_snapshot_unlocked policy ~state_root ~workflow_id)

let publish_first_event path event =
  let descriptor =
    Unix.openfile path [ O_WRONLY; O_CREAT; O_EXCL ] 0o600
  in
  Unix.set_close_on_exec descriptor;
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () -> write_all descriptor (encode_event event); Unix.fsync descriptor);
  fsync_directory (Filename.dirname path)

let append_event path event =
  let stat = require_regular_file path in
  if stat.st_perm land 0o077 <> 0 || stat.st_uid <> Unix.geteuid () then
    failf "causal-workflow-journal-permission-mismatch";
  let descriptor = Unix.openfile path [ O_WRONLY; O_APPEND ] 0 in
  Unix.set_close_on_exec descriptor;
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () -> write_all descriptor (encode_event event); Unix.fsync descriptor)

let open_workflow ~repo_root ~state_root ~workflow_id ~workflow_generation
    ~guardian_generation ~journal_id ~store_id ~controller_generation
    ~source_sha256 =
  validate_atom "workflow-id" workflow_id;
  let policy = load_policy ~repo_root in
  let workflow_key = sha256 workflow_id in
  let payload =
    encode_fields
      [ ("workflow_id", workflow_id);
        ("workflow_generation", digest "workflow-generation" workflow_generation);
        ("guardian_generation", digest "guardian-generation" guardian_generation);
        ("journal_id", digest "journal-id" journal_id);
        ("store_id", digest "store-id" store_id);
        ("controller_generation", digest "controller-generation" controller_generation);
        ("source_sha256", digest "source" source_sha256) ]
  in
  with_lock state_root (fun () ->
      let path = journal_path state_root workflow_key in
      if Sys.file_exists path then failf "causal-workflow-already-open";
      let event =
        make_event ~sequence:1 ~previous:zero_digest
          ~semantics:policy.semantics_sha256 ~workflow_key
          ~kind:"WORKFLOW_OPENED" ~payload
      in
      publish_first_event path event;
      initial_snapshot event)

let append_transition ~repo_root ~state_root ~workflow_id ~kind ~transition
    ~expected ~next ~extra_fields =
  let policy = load_policy ~repo_root in
  with_lock state_root (fun () ->
      let snapshot = load_snapshot_unlocked policy ~state_root ~workflow_id in
      if snapshot.phase <> expected then
        failf "causal-workflow-phase-refused:expected=%s:actual=%s"
          (phase_name expected) (phase_name snapshot.phase);
      let fields =
        [ ("transition", string_of_int transition);
          ("current_state", string_of_int (phase_code snapshot.phase));
          ("next_state", string_of_int (phase_code next));
          ("observed_predecessor_sha256", snapshot.head_sha256) ]
        @ extra_fields
      in
      let event =
        make_event ~sequence:(snapshot.sequence + 1)
          ~previous:snapshot.head_sha256 ~semantics:policy.semantics_sha256
          ~workflow_key:snapshot.workflow_key ~kind
          ~payload:(encode_fields fields)
      in
      let next_snapshot = apply_transition policy snapshot event in
      append_event (journal_path state_root snapshot.workflow_key) event;
      next_snapshot)

let arm_compile ~repo_root ~state_root ~workflow_id =
  append_transition ~repo_root ~state_root ~workflow_id ~kind:"COMPILE_ARMED"
    ~transition:1 ~expected:Waiting_compile ~next:Compile_armed ~extra_fields:[]

let start_compile ~repo_root ~state_root ~workflow_id =
  append_transition ~repo_root ~state_root ~workflow_id ~kind:"COMPILE_STARTED"
    ~transition:2 ~expected:Compile_armed ~next:Compile_running ~extra_fields:[]

let close_compile ~repo_root ~state_root ~workflow_id ~compile_receipt
    ~artifact_record ~artifact_handle =
  append_transition ~repo_root ~state_root ~workflow_id ~kind:"COMPILE_CLOSED"
    ~transition:3 ~expected:Compile_running ~next:Compiled_closed
    ~extra_fields:[ ("compile_receipt", compile_receipt);
      ("artifact_record", artifact_record); ("artifact_handle", artifact_handle) ]

let recover_controller ~repo_root ~state_root ~workflow_id
    ~successor_controller_generation ~guardian_generation ~journal_id ~store_id =
  let snapshot = load_snapshot ~repo_root ~state_root ~workflow_id in
  if phase_code snapshot.phase < 3 || phase_code snapshot.phase > 10 then
    failf "causal-workflow-recovery-state-refused";
  append_transition ~repo_root ~state_root ~workflow_id
    ~kind:"CONTROLLER_RECOVERED" ~transition:11 ~expected:snapshot.phase
    ~next:snapshot.phase
    ~extra_fields:[
      ("predecessor_controller_extinct", "true");
      ("successor_controller_generation", successor_controller_generation);
      ("guardian_generation", guardian_generation);
      ("journal_id", journal_id); ("store_id", store_id) ]

let commit_run_ticket ~repo_root ~state_root ~workflow_id ~run_ticket
    ~run_grant ~run_grant_generation =
  let snapshot = load_snapshot ~repo_root ~state_root ~workflow_id in
  let artifact_record = require_option "artifact-record" snapshot.artifact_record in
  append_transition ~repo_root ~state_root ~workflow_id
    ~kind:"RUN_TICKET_COMMITTED" ~transition:4 ~expected:Compiled_closed
    ~next:Run_armed ~extra_fields:[ ("run_ticket", run_ticket);
      ("run_grant", run_grant); ("run_grant_generation", run_grant_generation);
      ("artifact_record", artifact_record) ]

let mark_run_launched ~repo_root ~state_root ~workflow_id ~start_receipt
    ~run_pid_identity =
  let snapshot = load_snapshot ~repo_root ~state_root ~workflow_id in
  let run_ticket = require_option "run-ticket" snapshot.run_ticket in
  append_transition ~repo_root ~state_root ~workflow_id ~kind:"RUN_LAUNCHED"
    ~transition:5 ~expected:Run_armed ~next:Running
    ~extra_fields:[ ("run_ticket", run_ticket); ("start_receipt", start_receipt);
      ("run_pid_identity", run_pid_identity) ]

let seal_run_result ~repo_root ~state_root ~workflow_id ~exit_code
    ~stdout_sha256 ~stderr_sha256 ~result_record ~result_handle =
  append_transition ~repo_root ~state_root ~workflow_id
    ~kind:"RUN_RESULT_SEALED" ~transition:6 ~expected:Running ~next:Run_measured
    ~extra_fields:[ ("exit_code", string_of_int exit_code);
      ("stdout_sha256", stdout_sha256);
      ("stderr_sha256", stderr_sha256);
      ("result_record", result_record); ("result_handle", result_handle) ]

let string_of_bool value = if value then "true" else "false"

let close_run ~repo_root ~state_root ~workflow_id ~pid_extinct
    ~descendants_extinct ~cgroup_unit_extinct ~grant_extinct ~capsule_extinct =
  append_transition ~repo_root ~state_root ~workflow_id ~kind:"RUN_CLOSED"
    ~transition:7 ~expected:Run_measured ~next:Run_closed
    ~extra_fields:[ ("pid_extinct", string_of_bool pid_extinct);
      ("descendants_extinct", string_of_bool descendants_extinct);
      ("cgroup_unit_extinct", string_of_bool cgroup_unit_extinct);
      ("grant_extinct", string_of_bool grant_extinct);
      ("capsule_extinct", string_of_bool capsule_extinct) ]

let arm_attest ~repo_root ~state_root ~workflow_id =
  append_transition ~repo_root ~state_root ~workflow_id ~kind:"ATTEST_ARMED"
    ~transition:8 ~expected:Run_closed ~next:Attest_armed ~extra_fields:[]

let start_attest ~repo_root ~state_root ~workflow_id =
  append_transition ~repo_root ~state_root ~workflow_id ~kind:"ATTEST_STARTED"
    ~transition:9 ~expected:Attest_armed ~next:Attest_running ~extra_fields:[]

let close_attest ~repo_root ~state_root ~workflow_id ~attestation_record
    ~attestation_handle =
  let snapshot = load_snapshot ~repo_root ~state_root ~workflow_id in
  append_transition ~repo_root ~state_root ~workflow_id ~kind:"ATTEST_CLOSED"
    ~transition:10 ~expected:Attest_running ~next:Attested_closed
    ~extra_fields:[
      ("artifact_record", require_option "artifact-record" snapshot.artifact_record);
      ("result_record", require_option "result-record" snapshot.result_record);
      ("attestation_record", attestation_record);
      ("attestation_handle", attestation_handle) ]

let journal_file ~state_root ~workflow_id =
  journal_path state_root (sha256 workflow_id)
