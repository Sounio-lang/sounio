exception Error of string

let pinned_manifest_sha256 =
  "8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051"

let pinned_resident_v4_manifest_sha256 =
  "f61c93a3aefdbab792ed757faddf778017d34e0fa6bed97c565b56fe3147d473"

let max_file_bytes = 8 * 1024 * 1024

type policy = {
  manifest_sha256 : string;
  source_sha256 : string;
  semantics_sha256 : string;
  parent_9029_sha256 : string;
  parent_9021_sha256 : string;
  parent_9022_sha256 : string;
  resident_v4_sha256 : string;
}

type state = Vacant | Issued | Outcome_pending | Closed | Revoked | Poisoned

type t = {
  policy : policy;
  resident : Loom_resident.t;
  deadline_ms : int;
  mutable state : state;
  mutable authority_poisoned : bool;
}

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

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
        else if total + count > limit then
          failf "exec-grant-cell-file-too-large:%s" path
        else (Buffer.add_subbytes output bytes 0 count; loop (total + count))
      in
      loop 0)

let sha256_file path = sha256 (read_file path)

let parse_manifest path =
  let table = Hashtbl.create 96 in
  read_file path |> String.split_on_char '\n'
  |> List.iter (fun line ->
         match String.index_opt line '=' with
         | None when line = "" -> ()
         | None -> failf "malformed-exec-grant-cell-manifest"
         | Some index ->
             let key = String.sub line 0 index in
             if Hashtbl.mem table key then
               failf "duplicate-exec-grant-cell-field:%s" key;
             Hashtbl.add table key
               (String.sub line (index + 1) (String.length line - index - 1)));
  table

let required table key =
  match Hashtbl.find_opt table key with
  | Some value when value <> "" -> value
  | _ -> failf "missing-exec-grant-cell-field:%s" key

let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"

let manifest_path root =
  match Sys.getenv_opt "SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_MANIFEST" with
  | Some value when value <> "" && test_mode () -> value
  | Some value when value <> "" ->
      failf
        "SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_MANIFEST-override-requires-test-mode"
  | _ ->
      Filename.concat root "tools/loom/kernel_exec_grant_cell_authority.freeze.v1"

let verify_file root manifest path_key hash_key reason =
  let path = Filename.concat root (required manifest path_key) in
  if not (Sys.file_exists path)
     || sha256_file path <> required manifest hash_key
  then failf "%s" reason

let load root =
  let root = Unix.realpath root in
  let path = manifest_path root in
  if not (Sys.file_exists path) then failf "exec-grant-cell-manifest-missing";
  if sha256_file path <> pinned_manifest_sha256 then
    failf "exec-grant-cell-manifest-hash-mismatch";
  let manifest = parse_manifest path in
  if required manifest "schema"
       <> "loom-kernel-exec-grant-cell-authority-freeze-v1"
     || required manifest "stage" <> "SEMANTICS_FROZEN"
     || required manifest "producing_language" <> "Sounio"
     || required manifest "language_role" <> "SEMANTIC_AUTHORITY"
     || required manifest "action" <> "9030"
     || required manifest "parent_actions" <> "9029,9021,9022"
     || required manifest "write_shape_validation" <> "pre_mutation"
     || required manifest "absence_model"
          <> "affirmative_state_generation_authority_triple"
     || required manifest "handle_is_bearer" <> "false"
     || required manifest "python_oracle_authority" <> "false"
     || required manifest "rust_oracle_authority" <> "false"
     || required manifest "review_only_authority" <> "false"
     || required manifest "material_grant" <> "false"
     || required manifest "same_uid_peer_isolation" <> "false"
     || required manifest "parity_open" <> "false"
     || required manifest "exec_attached" <> "false"
     || required manifest "commit_attached" <> "false"
     || required manifest "ci_attached" <> "false"
     || required manifest "claim_ready" <> "false"
  then failf "exec-grant-cell-manifest-state-invalid";
  verify_file root manifest "garden_path" "garden_sha256"
    "exec-grant-cell-garden-hash-mismatch";
  verify_file root manifest "source_path" "source_sha256"
    "exec-grant-cell-source-hash-mismatch";
  verify_file root manifest "entrypoint_path" "entrypoint_sha256"
    "exec-grant-cell-entrypoint-hash-mismatch";
  verify_file root manifest "parent_9029_manifest_path"
    "parent_9029_manifest_sha256" "exec-grant-cell-parent-9029-hash-mismatch";
  verify_file root manifest "parent_9021_manifest_path"
    "parent_9021_manifest_sha256" "exec-grant-cell-parent-9021-hash-mismatch";
  verify_file root manifest "parent_9022_manifest_path"
    "parent_9022_manifest_sha256" "exec-grant-cell-parent-9022-hash-mismatch";
  let resident_v4_path =
    Filename.concat root "tools/loom/resident_membrane.runtime.v4"
  in
  if sha256_file resident_v4_path <> pinned_resident_v4_manifest_sha256 then
    failf "exec-grant-cell-resident-v4-manifest-hash-mismatch";
  let resident_v4 = parse_manifest resident_v4_path in
  if required resident_v4 "parent_9030_sha256" <> pinned_manifest_sha256
     || required resident_v4 "runtime_frozen" <> "true"
     || required resident_v4 "route_9030" <> "5"
  then failf "exec-grant-cell-resident-v4-binding-invalid";
  { manifest_sha256 = pinned_manifest_sha256;
    source_sha256 = required manifest "source_sha256";
    semantics_sha256 = required manifest "semantics_sha256";
    parent_9029_sha256 = required manifest "parent_9029_manifest_sha256";
    parent_9021_sha256 = required manifest "parent_9021_manifest_sha256";
    parent_9022_sha256 = required manifest "parent_9022_manifest_sha256";
    resident_v4_sha256 = pinned_resident_v4_manifest_sha256 }

let state_name = function
  | Vacant -> "VACANT"
  | Issued -> "ISSUED"
  | Outcome_pending -> "OUTCOME_PENDING"
  | Closed -> "CLOSED"
  | Revoked -> "REVOKED"
  | Poisoned -> "POISONED"

let state_code = function
  | Vacant -> 0
  | Issued -> 1
  | Outcome_pending -> 3
  | Closed -> 4
  | Revoked -> 5
  | Poisoned -> 6

let state_of_code = function
  | 0 -> Vacant
  | 1 -> Issued
  | 3 -> Outcome_pending
  | 4 -> Closed
  | 5 -> Revoked
  | 6 -> Poisoned
  | value -> failf "exec-grant-cell-state-unsupported:%d" value

let frame_transition frame =
  let fields =
    String.trim frame |> String.split_on_char ' ' |> List.filter (( <> ) "")
  in
  try
    let action = List.nth fields 0 |> int_of_string in
    let stage = List.nth fields 1 |> int_of_string in
    let operation = List.nth fields 3 |> int_of_string in
    let current_state = List.nth fields 4 |> int_of_string in
    let next_state = List.nth fields 5 |> int_of_string in
    if action <> 9030 || stage <> 3 then failf "exec-grant-cell-frame-domain-mismatch";
    (operation, current_state, next_state)
  with
  | Error _ as error -> raise error
  | _ -> failf "exec-grant-cell-frame-malformed"

let invalidate cell reason =
  if not cell.authority_poisoned then (
    cell.authority_poisoned <- true;
    cell.state <- Poisoned;
    Loom_resident.invalidate cell.resident reason)

let require_transition cell ~operation frame =
  if cell.authority_poisoned then failf "exec-grant-cell-authority-poisoned";
  let actual_operation, current_state, next_state =
    try frame_transition frame with Error reason ->
      invalidate cell "exec-grant-cell-malformed-frame";
      failf "exec-grant-cell-frame-refused:%s" reason
  in
  if actual_operation <> operation || current_state <> state_code cell.state then (
    invalidate cell "exec-grant-cell-transition-mismatch";
    failf
      "exec-grant-cell-transition-mismatch:expected-%d:%d:actual-%d:%d"
      operation (state_code cell.state) actual_operation current_state);
  state_of_code next_state

let invoke cell ~operation frame =
  let next_state = require_transition cell ~operation frame in
  try
    let decision =
      Loom_resident.decide_exec_grant_cell cell.resident
        ~deadline_ms:cell.deadline_ms frame
    in
    if decision.code = 0 then cell.state <- next_state;
    decision
  with
  | Loom_resident.Error reason ->
      invalidate cell ("exec-grant-cell-resident-refusal:" ^ reason);
      failf "exec-grant-cell-resident-refusal:%s" reason
  | Sys_error reason ->
      invalidate cell ("exec-grant-cell-resident-system-refusal:" ^ reason);
      failf "exec-grant-cell-resident-system-refusal:%s" reason
  | Unix.Unix_error (error, function_name, argument) ->
      let reason =
        Printf.sprintf "%s:%s(%s)" (Unix.error_message error) function_name
          argument
      in
      invalidate cell ("exec-grant-cell-resident-unix-refusal:" ^ reason);
      failf "exec-grant-cell-resident-unix-refusal:%s" reason

let issue cell frame = invoke cell ~operation:1 frame
let consume cell frame = invoke cell ~operation:2 frame

let close_outcome cell frame =
  let decision = invoke cell ~operation:3 frame in
  if decision.code = 0 then
    Loom_resident.close cell.resident ~deadline_ms:cell.deadline_ms;
  decision

let revoke cell frame =
  let decision = invoke cell ~operation:4 frame in
  if decision.code = 0 then
    Loom_resident.invalidate cell.resident "exec-grant-cell-revoked";
  decision

let create ~root ~environment ~deadline_ms =
  let root = Unix.realpath root in
  let policy = load root in
  let resident = Loom_resident.spawn_v4 ~root ~environment ~deadline_ms in
  { policy; resident; deadline_ms; state = Vacant; authority_poisoned = false }

let close cell =
  if cell.state <> Closed && not cell.authority_poisoned then
    Loom_resident.close cell.resident ~deadline_ms:cell.deadline_ms

let with_cell ~root ~environment ~deadline_ms callback =
  let cell = create ~root ~environment ~deadline_ms in
  match callback cell with
  | result -> close cell; result
  | exception error ->
      (try close cell with _ -> ());
      raise error

let test_timeout cell frame =
  ignore (require_transition cell ~operation:2 frame);
  try
    let refused = Loom_resident.test_exec_grant_timeout cell.resident frame in
    invalidate cell "exec-grant-cell-timeout";
    refused
  with Loom_resident.Error reason ->
    invalidate cell "exec-grant-cell-timeout-control-failed";
    failf "exec-grant-cell-timeout-control-failed:%s" reason

let test_eof cell frame =
  ignore (require_transition cell ~operation:2 frame);
  try
    let refused =
      Loom_resident.test_exec_grant_eof cell.resident
        ~deadline_ms:cell.deadline_ms frame
    in
    invalidate cell "exec-grant-cell-eof";
    refused
  with Loom_resident.Error reason ->
    invalidate cell "exec-grant-cell-eof-control-failed";
    failf "exec-grant-cell-eof-control-failed:%s" reason

let state cell = cell.state
let is_poisoned cell = cell.authority_poisoned
let generation cell = Loom_resident.generation cell.resident
let resident_pid cell = Loom_resident.pid cell.resident
let sequence cell = Loom_resident.sequence cell.resident
let manifest_sha256 cell = cell.policy.manifest_sha256
let semantics_sha256 cell = cell.policy.semantics_sha256
let resident_v4_sha256 cell = cell.policy.resident_v4_sha256
