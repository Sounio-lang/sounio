exception Error of string

let pinned_manifest_sha256 =
  "61918604bf177753c6141f6cd0f05d342a1869ab8fc08d187306a481de33d70e"

let max_file_bytes = 8 * 1024 * 1024

type policy = {
  manifest_sha256 : string;
  source_sha256 : string;
  semantics_sha256 : string;
  parent_9028_sha256 : string;
  parent_9025_sha256 : string;
  parent_9023_sha256 : string;
}

type lifecycle =
  | Unprepared
  | Prepared
  | Effect_stopped
  | Closed
  | Poisoned

type t = {
  policy : policy;
  resident : Loom_resident.t;
  deadline_ms : int;
  mutable lifecycle : lifecycle;
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
          failf "invocation-cell-file-too-large:%s" path
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
         | None -> failf "malformed-invocation-cell-manifest"
         | Some index ->
             let key = String.sub line 0 index in
             if Hashtbl.mem table key then
               failf "duplicate-invocation-cell-field:%s" key;
             Hashtbl.add table key
               (String.sub line (index + 1) (String.length line - index - 1)));
  table

let required table key =
  match Hashtbl.find_opt table key with
  | Some value when value <> "" -> value
  | _ -> failf "missing-invocation-cell-field:%s" key

let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"

let manifest_path root =
  match Sys.getenv_opt "SOUNIO_LOOM_KERNEL_INVOCATION_CELL_MANIFEST" with
  | Some value when value <> "" && test_mode () -> value
  | Some value when value <> "" ->
      failf
        "SOUNIO_LOOM_KERNEL_INVOCATION_CELL_MANIFEST-override-requires-test-mode"
  | _ ->
      Filename.concat root "tools/loom/kernel_invocation_cell_authority.freeze.v1"

let verify_file root manifest path_key hash_key reason =
  let path = Filename.concat root (required manifest path_key) in
  if not (Sys.file_exists path)
     || sha256_file path <> required manifest hash_key
  then failf "%s" reason

let load root =
  let root = Unix.realpath root in
  let path = manifest_path root in
  if not (Sys.file_exists path) then failf "invocation-cell-manifest-missing";
  if sha256_file path <> pinned_manifest_sha256 then
    failf "invocation-cell-manifest-hash-mismatch";
  let manifest = parse_manifest path in
  if required manifest "schema"
       <> "loom-kernel-invocation-cell-authority-freeze-v1"
     || required manifest "stage" <> "SEMANTICS_FROZEN"
     || required manifest "producing_language" <> "Sounio"
     || required manifest "language_role" <> "SEMANTIC_AUTHORITY"
     || required manifest "action" <> "9029"
     || required manifest "parent_actions" <> "9028,9025,9023"
     || required manifest "python_oracle_authority" <> "false"
     || required manifest "rust_oracle_authority" <> "false"
     || required manifest "review_only_authority" <> "false"
     || required manifest "cell_digest_is_bearer" <> "false"
     || required manifest "material_invocation" <> "false"
     || required manifest "material_coverage" <> "false"
     || required manifest "same_uid_peer_isolation" <> "false"
     || required manifest "parity_open" <> "false"
     || required manifest "exec_attached" <> "false"
     || required manifest "commit_attached" <> "false"
     || required manifest "ci_attached" <> "false"
     || required manifest "claim_ready" <> "false"
  then failf "invocation-cell-manifest-state-invalid";
  verify_file root manifest "garden_path" "garden_sha256"
    "invocation-cell-garden-hash-mismatch";
  verify_file root manifest "source_path" "source_sha256"
    "invocation-cell-source-hash-mismatch";
  verify_file root manifest "entrypoint_path" "entrypoint_sha256"
    "invocation-cell-entrypoint-hash-mismatch";
  verify_file root manifest "parent_9028_manifest_path"
    "parent_9028_manifest_sha256" "invocation-cell-parent-9028-hash-mismatch";
  verify_file root manifest "parent_9025_manifest_path"
    "parent_9025_manifest_sha256" "invocation-cell-parent-9025-hash-mismatch";
  verify_file root manifest "parent_9023_manifest_path"
    "parent_9023_manifest_sha256" "invocation-cell-parent-9023-hash-mismatch";
  { manifest_sha256 = pinned_manifest_sha256;
    source_sha256 = required manifest "source_sha256";
    semantics_sha256 = required manifest "semantics_sha256";
    parent_9028_sha256 = required manifest "parent_9028_manifest_sha256";
    parent_9025_sha256 = required manifest "parent_9025_manifest_sha256";
    parent_9023_sha256 = required manifest "parent_9023_manifest_sha256" }

let state_name = function
  | Unprepared -> "UNPREPARED"
  | Prepared -> "PREPARED"
  | Effect_stopped -> "EFFECT_STOPPED"
  | Closed -> "CLOSED"
  | Poisoned -> "POISONED"

let frame_operation frame =
  let fields =
    String.trim frame |> String.split_on_char ' ' |> List.filter (( <> ) "")
  in
  try
    let action = List.nth fields 0 |> int_of_string in
    let stage = List.nth fields 1 |> int_of_string in
    let operation = List.nth fields 3 |> int_of_string in
    let state = List.nth fields 4 |> int_of_string in
    if action <> 9029 || stage <> 3 then failf "invocation-cell-frame-domain-mismatch";
    (operation, state)
  with
  | Error _ as error -> raise error
  | _ -> failf "invocation-cell-frame-malformed"

let invalidate cell reason =
  if cell.lifecycle <> Poisoned && cell.lifecycle <> Closed then (
    cell.lifecycle <- Poisoned;
    Loom_resident.invalidate cell.resident reason)

let require_transition cell ~operation ~semantic_state allowed =
  if not (List.mem cell.lifecycle allowed) then (
    invalidate cell "invocation-cell-lifecycle-replay";
    failf "invocation-cell-transition-refused:%s:operation-%d"
      (state_name cell.lifecycle) operation);
  let actual_operation, actual_state = frame_operation semantic_state in
  if actual_operation <> operation || actual_state <> operation then (
    invalidate cell "invocation-cell-operation-mismatch";
    failf "invocation-cell-operation-mismatch:expected-%d:actual-%d:%d"
      operation actual_operation actual_state)

let invoke cell ~operation ~frame ~allowed ~next =
  require_transition cell ~operation ~semantic_state:frame allowed;
  try
    let decision =
      Loom_resident.decide_invocation_cell cell.resident
        ~deadline_ms:cell.deadline_ms frame
    in
    if decision.code = 0 then cell.lifecycle <- next
    else if cell.lifecycle <> Unprepared then
      invalidate cell ("invocation-cell-semantic-deny-" ^ string_of_int decision.code);
    decision
  with
  | Loom_resident.Error reason ->
      cell.lifecycle <- Poisoned;
      failf "invocation-cell-resident-refusal:%s" reason

let prepare cell frame =
  invoke cell ~operation:1 ~frame ~allowed:[ Unprepared ] ~next:Prepared

let admit cell frame =
  invoke cell ~operation:2 ~frame ~allowed:[ Prepared; Effect_stopped ]
    ~next:Effect_stopped

let close_outcome cell frame =
  let decision =
    invoke cell ~operation:3 ~frame ~allowed:[ Prepared; Effect_stopped ]
      ~next:Closed
  in
  if decision.code = 0 then
    Loom_resident.close cell.resident ~deadline_ms:cell.deadline_ms;
  decision

let abort cell frame =
  let decision =
    invoke cell ~operation:4 ~frame ~allowed:[ Prepared; Effect_stopped ]
      ~next:Poisoned
  in
  if decision.code = 0 then
    Loom_resident.invalidate cell.resident "invocation-cell-aborted";
  decision

let create ~root ~environment ~deadline_ms =
  let root = Unix.realpath root in
  let policy = load root in
  let resident = Loom_resident.spawn_v3 ~root ~environment ~deadline_ms in
  { policy; resident; deadline_ms; lifecycle = Unprepared }

let close cell =
  if cell.lifecycle <> Closed && cell.lifecycle <> Poisoned then
    Loom_resident.close cell.resident ~deadline_ms:cell.deadline_ms

let with_cell ~root ~environment ~deadline_ms callback =
  let cell = create ~root ~environment ~deadline_ms in
  match callback cell with
  | result -> close cell; result
  | exception error ->
      (try close cell with _ -> ());
      raise error

let test_timeout cell frame =
  require_transition cell ~operation:2 ~semantic_state:frame
    [ Prepared; Effect_stopped ];
  try
    let refused = Loom_resident.test_invocation_timeout cell.resident frame in
    cell.lifecycle <- Poisoned;
    refused
  with Loom_resident.Error reason ->
    cell.lifecycle <- Poisoned;
    failf "invocation-cell-timeout-control-failed:%s" reason

let test_eof cell frame =
  require_transition cell ~operation:2 ~semantic_state:frame
    [ Prepared; Effect_stopped ];
  try
    let refused =
      Loom_resident.test_invocation_eof cell.resident
        ~deadline_ms:cell.deadline_ms frame
    in
    cell.lifecycle <- Poisoned;
    refused
  with Loom_resident.Error reason ->
    cell.lifecycle <- Poisoned;
    failf "invocation-cell-eof-control-failed:%s" reason

let lifecycle cell = cell.lifecycle
let is_poisoned cell = cell.lifecycle = Poisoned
let generation cell = Loom_resident.generation cell.resident
let resident_pid cell = Loom_resident.pid cell.resident
let sequence cell = Loom_resident.sequence cell.resident
let manifest_sha256 cell = cell.policy.manifest_sha256
let semantics_sha256 cell = cell.policy.semantics_sha256
