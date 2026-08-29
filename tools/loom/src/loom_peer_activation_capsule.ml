exception Error of string

let pinned_manifest_sha256 =
  "f2da55138bcfe5a8a2c65ebd79c1e534f152b33af5c6cc3d1f2b4eb3b4af6e7e"

let pinned_resident_v5_manifest_sha256 =
  "b3cf8c1e0524be35fc67b2b5a779bad9a9291195d65dc82dbc87595396fb5353"

let max_file_bytes = 8 * 1024 * 1024

type policy = {
  manifest_sha256 : string;
  source_sha256 : string;
  semantics_sha256 : string;
  parent_9025_sha256 : string;
  parent_9030_sha256 : string;
  resident_v5_sha256 : string;
}

type state = Empty | Sealed | Consumed | Extinct | Poisoned

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
          failf "peer-activation-capsule-file-too-large:%s" path
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
         | None -> failf "malformed-peer-activation-capsule-manifest"
         | Some index ->
             let key = String.sub line 0 index in
             if Hashtbl.mem table key then
               failf "duplicate-peer-activation-capsule-field:%s" key;
             Hashtbl.add table key
               (String.sub line (index + 1) (String.length line - index - 1)));
  table

let required table key =
  match Hashtbl.find_opt table key with
  | Some value when value <> "" -> value
  | _ -> failf "missing-peer-activation-capsule-field:%s" key

let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"

let manifest_path root =
  match Sys.getenv_opt "SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_MANIFEST" with
  | Some value when value <> "" && test_mode () -> value
  | Some value when value <> "" ->
      failf
        "SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_MANIFEST-override-requires-test-mode"
  | _ ->
      Filename.concat root "tools/loom/kernel_peer_activation_capsule_authority.freeze.v1"

let verify_file root manifest path_key hash_key reason =
  let path = Filename.concat root (required manifest path_key) in
  if not (Sys.file_exists path)
     || sha256_file path <> required manifest hash_key
  then failf "%s" reason

let load root =
  let root = Unix.realpath root in
  let path = manifest_path root in
  if not (Sys.file_exists path) then failf "peer-activation-capsule-manifest-missing";
  if sha256_file path <> pinned_manifest_sha256 then
    failf "peer-activation-capsule-manifest-hash-mismatch";
  let manifest = parse_manifest path in
  if required manifest "schema"
       <> "loom-kernel-peer-activation-capsule-authority-freeze-v1"
     || required manifest "stage" <> "SEMANTICS_FROZEN"
     || required manifest "producing_language" <> "Sounio"
     || required manifest "language_role" <> "SEMANTIC_AUTHORITY"
     || required manifest "action" <> "9031"
     || required manifest "parent_actions" <> "9025,9030"
     || required manifest "write_shape_validation" <> "pre_mutation"
     || required manifest "absence_model"
          <> "registry_absent+kernel_extinct+replay_refused"
     || required manifest "capsule_is_bearer" <> "false"
     || required manifest "python_oracle_authority" <> "false"
     || required manifest "rust_oracle_authority" <> "false"
     || required manifest "review_only_authority" <> "false"
     || required manifest "operational_realization" <> "false"
     || required manifest "capsule_material" <> "false"
     || required manifest "production_activation" <> "false"
     || required manifest "hardware_same_uid_peer_isolation" <> "true"
     || required manifest "launch_open" <> "false"
     || required manifest "recycle_open" <> "false"
     || required manifest "parity_open" <> "false"
     || required manifest "exec_attached" <> "false"
     || required manifest "commit_attached" <> "false"
     || required manifest "ci_attached" <> "false"
     || required manifest "claim_ready" <> "false"
  then failf "peer-activation-capsule-manifest-state-invalid";
  verify_file root manifest "garden_path" "garden_sha256"
    "peer-activation-capsule-garden-hash-mismatch";
  verify_file root manifest "source_path" "source_sha256"
    "peer-activation-capsule-source-hash-mismatch";
  verify_file root manifest "entrypoint_path" "entrypoint_sha256"
    "peer-activation-capsule-entrypoint-hash-mismatch";
  verify_file root manifest "parent_9025_manifest_path"
    "parent_9025_manifest_sha256" "peer-activation-capsule-parent-9025-hash-mismatch";
  verify_file root manifest "parent_9030_manifest_path"
    "parent_9030_manifest_sha256" "peer-activation-capsule-parent-9030-hash-mismatch";
  let resident_v5_path =
    Filename.concat root "tools/loom/resident_membrane.runtime.v5"
  in
  if sha256_file resident_v5_path <> pinned_resident_v5_manifest_sha256 then
    failf "peer-activation-capsule-resident-v5-manifest-hash-mismatch";
  let resident_v5 = parse_manifest resident_v5_path in
  if required resident_v5 "parent_9031_sha256" <> pinned_manifest_sha256
     || required resident_v5 "runtime_frozen" <> "true"
     || required resident_v5 "route_9031" <> "6"
     || required resident_v5 "parent_9025_v13_sha256"
          <> required manifest "parent_9025_manifest_sha256"
     || required resident_v5 "parent_9030_sha256"
          <> required manifest "parent_9030_manifest_sha256"
  then failf "peer-activation-capsule-resident-v5-binding-invalid";
  { manifest_sha256 = pinned_manifest_sha256;
    source_sha256 = required manifest "source_sha256";
    semantics_sha256 = required manifest "semantics_sha256";
    parent_9025_sha256 = required manifest "parent_9025_manifest_sha256";
    parent_9030_sha256 = required manifest "parent_9030_manifest_sha256";
    resident_v5_sha256 = pinned_resident_v5_manifest_sha256 }

let state_name = function
  | Empty -> "EMPTY"
  | Sealed -> "SEALED"
  | Consumed -> "CONSUMED"
  | Extinct -> "EXTINCT"
  | Poisoned -> "POISONED"

let state_code = function
  | Empty -> 0
  | Sealed -> 1
  | Consumed -> 2
  | Extinct -> 3
  | Poisoned -> 4

let state_of_code = function
  | 0 -> Empty
  | 1 -> Sealed
  | 2 -> Consumed
  | 3 -> Extinct
  | 4 -> Poisoned
  | value -> failf "peer-activation-capsule-state-unsupported:%d" value

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
    if action <> 9031 || stage <> 3 then failf "peer-activation-capsule-frame-domain-mismatch";
    (operation, current_state, next_state)
  with
  | Error _ as error -> raise error
  | _ -> failf "peer-activation-capsule-frame-malformed"

let invalidate cell reason =
  if not cell.authority_poisoned then (
    cell.authority_poisoned <- true;
    cell.state <- Poisoned;
    Loom_resident.invalidate cell.resident reason)

let require_transition cell ~operation frame =
  if cell.authority_poisoned then failf "peer-activation-capsule-authority-poisoned";
  let actual_operation, current_state, next_state =
    try frame_transition frame with Error reason ->
      invalidate cell "peer-activation-capsule-malformed-frame";
      failf "peer-activation-capsule-frame-refused:%s" reason
  in
  if actual_operation <> operation || current_state <> state_code cell.state then (
    invalidate cell "peer-activation-capsule-transition-mismatch";
    failf
      "peer-activation-capsule-transition-mismatch:expected-%d:%d:actual-%d:%d"
      operation (state_code cell.state) actual_operation current_state);
  state_of_code next_state

let invoke cell ~operation frame =
  let next_state = require_transition cell ~operation frame in
  try
    let decision =
      Loom_resident.decide_peer_activation_capsule cell.resident
        ~deadline_ms:cell.deadline_ms frame
    in
    if decision.code = 0 then cell.state <- next_state;
    decision
  with
  | Loom_resident.Error reason ->
      invalidate cell ("peer-activation-capsule-resident-refusal:" ^ reason);
      failf "peer-activation-capsule-resident-refusal:%s" reason
  | Sys_error reason ->
      invalidate cell ("peer-activation-capsule-resident-system-refusal:" ^ reason);
      failf "peer-activation-capsule-resident-system-refusal:%s" reason
  | Unix.Unix_error (error, function_name, argument) ->
      let reason =
        Printf.sprintf "%s:%s(%s)" (Unix.error_message error) function_name
          argument
      in
      invalidate cell ("peer-activation-capsule-resident-unix-refusal:" ^ reason);
      failf "peer-activation-capsule-resident-unix-refusal:%s" reason

let seal cell frame = invoke cell ~operation:1 frame
let consume cell frame = invoke cell ~operation:2 frame

let extinguish cell frame =
  let decision = invoke cell ~operation:3 frame in
  if decision.code = 0 then
    Loom_resident.close cell.resident ~deadline_ms:cell.deadline_ms;
  decision

let poison cell frame =
  let decision = invoke cell ~operation:4 frame in
  if decision.code = 0 then
    invalidate cell "peer-activation-capsule-poisoned";
  decision

let create ~root ~environment ~deadline_ms =
  let root = Unix.realpath root in
  let policy = load root in
  let resident = Loom_resident.spawn_v5 ~root ~environment ~deadline_ms in
  { policy; resident; deadline_ms; state = Empty; authority_poisoned = false }

let close cell =
  if cell.state <> Extinct && not cell.authority_poisoned then
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
    let refused = Loom_resident.test_peer_activation_timeout cell.resident frame in
    invalidate cell "peer-activation-capsule-timeout";
    refused
  with Loom_resident.Error reason ->
    invalidate cell "peer-activation-capsule-timeout-control-failed";
    failf "peer-activation-capsule-timeout-control-failed:%s" reason

let test_eof cell frame =
  ignore (require_transition cell ~operation:2 frame);
  try
    let refused =
      Loom_resident.test_peer_activation_eof cell.resident
        ~deadline_ms:cell.deadline_ms frame
    in
    invalidate cell "peer-activation-capsule-eof";
    refused
  with Loom_resident.Error reason ->
    invalidate cell "peer-activation-capsule-eof-control-failed";
    failf "peer-activation-capsule-eof-control-failed:%s" reason

let state cell = cell.state
let is_poisoned cell = cell.authority_poisoned
let resident cell = cell.resident
let generation cell = Loom_resident.generation cell.resident
let resident_pid cell = Loom_resident.pid cell.resident
let sequence cell = Loom_resident.sequence cell.resident
let manifest_sha256 cell = cell.policy.manifest_sha256
let semantics_sha256 cell = cell.policy.semantics_sha256
let resident_v5_sha256 cell = cell.policy.resident_v5_sha256
