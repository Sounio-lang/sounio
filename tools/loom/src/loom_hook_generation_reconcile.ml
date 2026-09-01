open Unix

exception Error of string

let pinned_manifest_sha256 =
  "a38fcb98dbaeb68b1913aec07b1646d8e965249a1bb05a01427327a78aea7cd7"

let semantics_sha256 =
  "63733afa5f88bb5bc867ce59f5a7b481927b0126096d602c3bdf949b25935fff"

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

external flock_try_exclusive : Unix.file_descr -> bool =
  "sounio_loom_flock_try_exclusive"

external flock_unlock : Unix.file_descr -> unit = "sounio_loom_flock_unlock"
external host_identity : unit -> string = "sounio_loom_host_identity"

let starts_with = Loom_hook_generation_drain.starts_with
let sha256 = Loom_hook_generation_drain.sha256
let sha256_file = Loom_hook_generation_drain.sha256_file
let read_file = Loom_hook_generation_drain.read_file
let parse_fields = Loom_hook_generation_drain.parse_fields
let required = Loom_hook_generation_drain.required
let exact = Loom_hook_generation_drain.exact
let decimal = Loom_hook_generation_drain.decimal
let field = Loom_hook_generation_drain.field
let json_escape = Loom_hook_generation_drain.json_escape
let slug = Loom_hook_generation_drain.slug
let hash_u60 = Loom_hook_generation_drain.hash_u60
let split_hash = Loom_hook_generation_drain.split_hash
let utc_now = Loom_hook_generation_drain.utc_now

type policy = {
  runtime : string;
  manifest_sha256 : string;
  executable_sha256 : string;
  sabotage_count : int;
  sabotage_required : int;
}

type cause = No_absence | Boot_changed | Namespace_changed | Pid_absent | Pid_reused

let cause_name = function
  | No_absence -> "none"
  | Boot_changed -> "boot-changed"
  | Namespace_changed -> "pid-namespace-changed"
  | Pid_absent -> "process-missing"
  | Pid_reused -> "pid-reused"

let cause_code = function
  | No_absence -> 0
  | Boot_changed -> 1
  | Namespace_changed -> 2
  | Pid_absent -> 3
  | Pid_reused -> 4

let bit word shift enabled = if enabled then word lor (1 lsl shift) else word

let rec find_source_root path =
  let marker =
    Filename.concat path "tools/loom/native_hook_generation_reconcile.freeze.v1"
  in
  if Sys.file_exists marker then path
  else
    let parent = Filename.dirname path in
    if parent = path then failf "source-root-not-found:%s" path
    else find_source_root parent

let verify_manifest_file root manifest path_key hash_key reason =
  let path = Filename.concat root (required "freeze-manifest" manifest path_key) in
  let expected = required "freeze-manifest" manifest hash_key in
  if sha256_file reason path <> expected then failf "%s-hash-mismatch" reason

let choose_runtime root manifest =
  let repository_runtime =
    Filename.concat root
      "tools/loom/_build/default/src/sounio-loom-native-hook-generation-reconcile"
  in
  let installed_runtime =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-native-hook-generation-reconcile"
  in
  let selected =
    match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_GENERATION_RECONCILE_RUNTIME" with
    | Some value when value <> "" ->
        if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
          failf "authority-runtime-override-requires-test-mode";
        value
    | _ ->
        if Sys.file_exists repository_runtime then repository_runtime
        else installed_runtime
  in
  let stat = Loom_hook_generation_drain.require_regular_file "authority-runtime" selected in
  if stat.st_perm land 0o111 = 0 then failf "authority-runtime-not-executable";
  let expected = required "freeze-manifest" manifest "executable_sha256" in
  if sha256_file "authority-runtime" selected <> expected then
    failf "authority-runtime-hash-mismatch";
  Unix.realpath selected

let load_policy root =
  let path =
    match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_GENERATION_RECONCILE_MANIFEST" with
    | Some value when value <> "" ->
        if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
          failf "freeze-manifest-override-requires-test-mode";
        value
    | _ ->
        Filename.concat root "tools/loom/native_hook_generation_reconcile.freeze.v1"
  in
  if sha256_file "freeze-manifest" path <> pinned_manifest_sha256 then
    failf "freeze-manifest-hash-mismatch";
  let manifest = parse_fields "freeze-manifest" (read_file "freeze-manifest" path) in
  exact "freeze-manifest" manifest "schema"
    "loom-native-hook-generation-reconcile-freeze-v1";
  exact "freeze-manifest" manifest "stage" "SEMANTICS_FROZEN";
  exact "freeze-manifest" manifest "semantic_authority" "Sounio";
  exact "freeze-manifest" manifest "producing_language" "Sounio";
  exact "freeze-manifest" manifest "language_role" "SEMANTIC_AUTHORITY";
  exact "freeze-manifest" manifest "action" "9047";
  exact "freeze-manifest" manifest "parent_action" "9046-frozen";
  exact "freeze-manifest" manifest "semantics_sha256" semantics_sha256;
  exact "freeze-manifest" manifest "load_bearing_rule" "pid_absent";
  exact "freeze-manifest" manifest "causal_sabotage" "pid-absence-rule-removed";
  exact "freeze-manifest" manifest "python_executed" "false";
  exact "freeze-manifest" manifest "rust_executed" "false";
  exact "freeze-manifest" manifest "disposable_oracle_executed" "false";
  List.iter
    (fun (path_key, hash_key, reason) ->
      verify_manifest_file root manifest path_key hash_key reason)
    [ ("garden_path", "garden_sha256", "garden");
      ("source_path", "source_sha256", "authority-source");
      ("entrypoint_path", "entrypoint_sha256", "authority-entrypoint");
      ("build_script_path", "build_script_sha256", "authority-build-script");
      ("selftest_path", "selftest_sha256", "authority-selftest");
      ("freeze_selftest_path", "freeze_selftest_sha256", "authority-freeze-selftest");
      ("first_manifest_path", "first_manifest_sha256", "first-manifest");
      ("first_evidence_path", "first_evidence_sha256", "first-evidence");
      ("parent_9046_freeze_path", "parent_9046_freeze_sha256", "parent-9046");
      ("toolchain_wrapper_path", "toolchain_wrapper_sha256", "toolchain-wrapper");
      ("toolchain_compiler_path", "toolchain_compiler_sha256", "toolchain-compiler") ] ;
  { runtime = choose_runtime root manifest;
    manifest_sha256 = pinned_manifest_sha256;
    executable_sha256 = required "freeze-manifest" manifest "executable_sha256";
    sabotage_count = required "freeze-manifest" manifest "sabotage_count" |> decimal "sabotage-count";
    sabotage_required =
      required "freeze-manifest" manifest "sabotage_required"
      |> decimal "sabotage-required" }

let state_directory root =
  match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_RECONCILE_STATE_DIR" with
  | Some value when value <> "" ->
      if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
        failf "reconcile-state-override-requires-test-mode";
      Unix.realpath value
  | _ ->
      Filename.concat (Loom_hook_generation_drain.git_common_dir root)
        "sounio-coord-state"

let with_state_lock state operation =
  let path = Filename.concat state ".claims.lock" in
  let descriptor = Unix.openfile path [ O_RDWR; O_CREAT ] 0o600 in
  let deadline = Unix.gettimeofday () +. 2.0 in
  let rec acquire () =
    if flock_try_exclusive descriptor then ()
    else if Unix.gettimeofday () >= deadline then failf "state-lock-timeout"
    else (ignore (Unix.select [] [] [] 0.02); acquire ())
  in
  Fun.protect
    ~finally:(fun () ->
      (try flock_unlock descriptor with _ -> ());
      Unix.close descriptor)
    (fun () -> acquire (); operation ())

let ensure_identity label values ~key ~agent ~lane ~worktree =
  if key <> "" && required label values key <> slug agent ^ "--" ^ slug lane then
    failf "%s-id-drift" label;
  if required label values "agent" <> agent || required label values "lane" <> lane then
    failf "%s-principal-drift" label;
  match Hashtbl.find_opt values "worktree" with
  | Some value when value = worktree -> ()
  | Some _ -> failf "%s-worktree-drift" label
  | None -> ()

type artifact = { relative : string; source : string; digest : string; text : string }

let same_field label values key expected =
  if required label values key <> expected then failf "%s-generation-drift:%s" label key

let related_artifacts state key agent lane worktree presence_values =
  let candidates =
    [ ("claims", ".claim", "claim_id");
      ("delivery-endpoints", ".endpoint", "endpoint_id");
      ("hook-capabilities", ".capability", "");
      ("process-presences", ".presence", "presence_id") ]
  in
  let artifacts =
    List.filter_map
      (fun (directory, suffix, identity_key) ->
        let relative = directory ^ "/" ^ key ^ suffix in
        let source = Filename.concat state relative in
        if not (Sys.file_exists source) then None
        else
          let text = read_file ("related-" ^ directory) source in
          let values = parse_fields ("related-" ^ directory) text in
          ensure_identity ("related-" ^ directory) values ~key:identity_key ~agent
            ~lane ~worktree;
          if directory = "delivery-endpoints" then (
            same_field "related-delivery-endpoints" values "session_id"
              (required "presence-record" presence_values "session_id");
            same_field "related-delivery-endpoints" values "harness"
              (required "presence-record" presence_values "harness");
            same_field "related-delivery-endpoints" values "harness_pid"
              (required "presence-record" presence_values "pid");
            same_field "related-delivery-endpoints" values "harness_pid_start"
              (required "presence-record" presence_values "pid_start"));
          if directory = "hook-capabilities" then (
            same_field "related-hook-capabilities" values "session_id"
              (required "presence-record" presence_values "session_id");
            same_field "related-hook-capabilities" values "generation"
              (required "presence-record" presence_values "generation");
            same_field "related-hook-capabilities" values "harness"
              (required "presence-record" presence_values "harness");
            same_field "related-hook-capabilities" values "presence_pid"
              (required "presence-record" presence_values "pid");
            same_field "related-hook-capabilities" values "presence_pid_start"
              (required "presence-record" presence_values "pid_start");
            same_field "related-hook-capabilities" values "presence_boot_id"
              (required "presence-record" presence_values "boot_id");
            same_field "related-hook-capabilities" values "presence_pid_namespace"
              (required "presence-record" presence_values "pid_namespace"));
          Some { relative; source; digest = sha256 text; text })
      candidates
  in
  if not (List.exists (fun value -> Filename.check_suffix value.relative ".presence") artifacts)
  then failf "presence-record-missing";
  artifacts

let artifact_signature artifacts =
  artifacts
  |> List.map (fun artifact -> artifact.relative ^ "\000" ^ artifact.digest)
  |> String.concat "\000" |> fun value -> sha256 ("loom-related-v1\000" ^ value)

let presence_inventory state =
  let directory = Filename.concat state "process-presences" in
  let records = Loom_hook_generation_drain.capture_inventory directory in
  let signature = Loom_hook_generation_drain.inventory_signature records in
  (records, sha256 ("loom-presence-inventory-v1\000" ^ signature))

let process_start_if_present pid =
  let path = Printf.sprintf "/proc/%d/stat" pid in
  if not (Sys.file_exists path) then None
  else
    try Some (Loom_hook_generation_drain.process_start pid) with _ -> None

type observation = {
  agent : string;
  lane : string;
  key : string;
  worktree : string;
  session_id : string;
  generation : string;
  record_digest : string;
  inventory_digest : string;
  related_digest : string;
  kernel_digest : string;
  transaction_digest : string;
  cause : cause;
  heartbeat_expired : bool;
  related : artifact list;
  destination : string;
}

let observe_locked state agent lane =
  let key = slug agent ^ "--" ^ slug lane in
  let presence = Filename.concat state ("process-presences/" ^ key ^ ".presence") in
  let text = read_file "presence-record" presence in
  let values = parse_fields "presence-record" text in
  let worktree = required "presence-record" values "worktree" in
  ensure_identity "presence-record" values ~key:"presence_id" ~agent ~lane ~worktree;
  let expected_boot = required "presence-record" values "boot_id" in
  let expected_namespace = required "presence-record" values "pid_namespace" in
  let pid = required "presence-record" values "pid" |> decimal "presence-pid" in
  if pid <= 0 then failf "presence-pid-not-positive";
  let expected_start = required "presence-record" values "pid_start" in
  ignore (decimal "presence-pid-start" expected_start);
  let last_seen = required "presence-record" values "last_seen_epoch" |> decimal "last-seen" in
  let ttl = required "presence-record" values "ttl_seconds" |> decimal "ttl-seconds" in
  if ttl <= 0 then failf "ttl-not-positive";
  let boot =
    Loom_hook_generation_drain.read_stream "kernel-boot-id"
      "/proc/sys/kernel/random/boot_id" |> String.trim
  in
  let namespace =
    try Unix.readlink "/proc/self/ns/pid"
    with _ -> failf "pid-namespace-unavailable"
  in
  if boot = "" || namespace = "" then failf "kernel-identity-empty";
  let observed_start = process_start_if_present pid in
  let cause =
    if expected_boot <> boot then Boot_changed
    else if expected_namespace <> namespace then Namespace_changed
    else match observed_start with
      | None -> Pid_absent
      | Some value when value <> expected_start -> Pid_reused
      | Some _ -> No_absence
  in
  let heartbeat_expired =
    cause = No_absence && int_of_float (Unix.time ()) > last_seen + ttl
  in
  let first_inventory, inventory_digest = presence_inventory state in
  let related = related_artifacts state key agent lane worktree values in
  let related_digest = artifact_signature related in
  let second_inventory, second_digest = presence_inventory state in
  if inventory_digest <> second_digest
     || Loom_hook_generation_drain.inventory_signature first_inventory
        <> Loom_hook_generation_drain.inventory_signature second_inventory
  then failf "presence-inventory-changed-under-lock";
  let record_digest = sha256 text in
  if not (List.exists (fun artifact -> artifact.digest = record_digest) related) then
    failf "presence-record-not-related";
  let kernel_digest =
    sha256
      (String.concat "\000"
         [ "loom-kernel-observation-v1"; expected_boot; boot; expected_namespace;
           namespace; string_of_int pid; expected_start;
           Option.value ~default:"absent" observed_start; cause_name cause ])
  in
  let transaction_digest =
    sha256
      (String.concat "\000"
         [ "loom-generation-reconcile-v1"; key; record_digest; inventory_digest;
           related_digest; kernel_digest; cause_name cause ])
  in
  let destination =
    Filename.concat state ("generation-quarantine/" ^ transaction_digest)
  in
  { agent; lane; key; worktree;
    session_id = required "presence-record" values "session_id";
    generation = required "presence-record" values "generation";
    record_digest; inventory_digest; related_digest; kernel_digest;
    transaction_digest; cause; heartbeat_expired; related; destination }

let authority_word observation ~transaction_ready =
  0 |> fun word -> bit word 0 true
  |> fun word -> bit word 1 true
  |> fun word -> bit word 2 true
  |> fun word -> bit word 3 true
  |> fun word -> bit word 4 true
  |> fun word -> bit word 5 true
  |> fun word -> bit word 6 true
  |> fun word -> bit word 7 true
  |> fun word -> bit word 8 true
  |> fun word -> bit word 9 true
  |> fun word -> bit word 10 true
  |> fun word -> bit word 11 (observation.cause = Boot_changed)
  |> fun word -> bit word 12 (observation.cause = Namespace_changed)
  |> fun word -> bit word 13 (observation.cause = Pid_absent)
  |> fun word -> bit word 14 (observation.cause = Pid_reused)
  |> fun word -> bit word 15 observation.heartbeat_expired
  |> fun word -> bit word 16 true
  |> fun word -> bit word 17 (not (Sys.file_exists observation.destination))
  |> fun word -> bit word 18 transaction_ready
  |> fun word -> bit word 19 transaction_ready
  |> fun word -> bit word 20 true
  |> fun word -> bit word 21 true
  |> fun word -> bit word 22 true
  |> fun word -> bit word 23 true

let authority_frame policy observation mode ~transaction_ready =
  let record_hash0, record_hash1 = split_hash observation.record_digest in
  Printf.sprintf "%d %d %d %d %d %d %d %Ld %Ld %Ld %Ld %d %d\n"
    9047 mode 3 (authority_word observation ~transaction_ready)
    (List.length observation.related) (List.length observation.related)
    (cause_code observation.cause) record_hash0 record_hash1
    (hash_u60 observation.kernel_digest)
    (hash_u60 observation.transaction_digest) policy.sabotage_count
    policy.sabotage_required

let parse_decision output =
  let prefix = "SOUNIO_NATIVE_HOOK_GENERATION_RECONCILE " in
  let suffix = " semantic_authority=Sounio action=9047" in
  if not (starts_with output prefix) || not (Filename.check_suffix output suffix) then
    failf "authority-output-invalid";
  String.sub output (String.length prefix)
    (String.length output - String.length prefix - String.length suffix)

let evaluate policy root observation mode ~transaction_ready =
  let frame = authority_frame policy observation mode ~transaction_ready in
  let result =
    Loom_hook.run_process ~input:frame ~timeout_seconds:5.0 ~cwd:root
      policy.runtime []
  in
  let output = String.trim result.output in
  let decision = parse_decision output in
  let admitted =
    decision = "KEEP" || decision = "QUARANTINE_ELIGIBLE"
    || decision = "QUARANTINE_READY"
  in
  let expected = if admitted then 0 else 42 in
  if result.code <> expected then
    failf "authority-exit-mismatch:%s:%d:%d" decision expected result.code;
  (decision, output, frame)

let ensure_directory path = Loom_hook_generation_guardian.ensure_directory path

let append_decision state line =
  let path = Filename.concat state "generation-reconcile-decisions.log" in
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
      Loom_hook_generation_guardian.write_all descriptor line;
      Unix.fsync descriptor)

let audit_line ~verdict ~decision ~reason ~agent ~lane ~transaction =
  let clean =
    String.map (function '\n' | '\r' -> '_' | character -> character)
  in
  Printf.sprintf
    "%s action=9047 verdict=%s decision=%s reason=%s agent=%s lane=%s transaction=%s\n"
    (utc_now (Unix.time ())) verdict decision (clean reason) (clean agent)
    (clean lane) transaction

let receipt_text policy observation decision output frame state =
  let command =
    Printf.sprintf "loom hook-generation-reconcile --agent %s --lane %s --apply"
      observation.agent observation.lane
  in
  Printf.sprintf
    "schema=loom-native-hook-generation-reconcile-receipt-v1\nstage=PARITY_OPEN\nsemantic_authority=Sounio\nproducing_language=Sounio\noperational_realization=OCaml\nlanguage_role=OPERATIONAL_REALIZATION\naction=9047\ndecision=%s\nabsence_reason=%s\nagent=%s\nlane=%s\nworktree=%s\nsession_id=%s\ngeneration=%s\nrecord_sha256=%s\ninventory_sha256=%s\nrelated_artifacts_sha256=%s\nkernel_observation_sha256=%s\ntransaction_sha256=%s\nmanifest_sha256=%s\nsemantics_sha256=%s\nexecutable_sha256=%s\nocaml_runtime_sha256=%s\nauthority_frame_sha256=%s\nauthority_output_sha256=%s\ntoolchain=OCaml-dune-plus-Sounio-lean_single\nhardware=%s\ncommand=%s\nresult=%s\nstate_directory=%s\nrelated_total=%d\npython_executed=false\nrust_executed=false\ndisposable_oracle_executed=false\nsame_uid_peer_isolation=false\ncreated_utc=%s\n"
    decision (cause_name observation.cause) observation.agent observation.lane
    observation.worktree observation.session_id observation.generation
    observation.record_digest observation.inventory_digest observation.related_digest
    observation.kernel_digest observation.transaction_digest policy.manifest_sha256
    semantics_sha256 policy.executable_sha256
    (sha256_file "ocaml-runtime" Sys.executable_name) (sha256 frame) (sha256 output)
    (host_identity ()) command decision state
    (List.length observation.related) (utc_now (Unix.time ()))

let wal_text observation state_value =
  let artifacts =
    observation.related
    |> List.map (fun artifact ->
           Printf.sprintf "artifact=%s:%s" artifact.relative artifact.digest)
    |> String.concat "\n"
  in
  Printf.sprintf
    "schema=loom-native-hook-generation-reconcile-wal-v1\nstate=%s\ntransaction_sha256=%s\nagent=%s\nlane=%s\nrecord_sha256=%s\ninventory_sha256=%s\nrelated_artifacts_sha256=%s\nkernel_observation_sha256=%s\nabsence_reason=%s\n%s\n"
    state_value observation.transaction_digest observation.agent observation.lane
    observation.record_digest observation.inventory_digest observation.related_digest
    observation.kernel_digest (cause_name observation.cause) artifacts

let move_to_quarantine state observation =
  let quarantine_root = Filename.concat state "generation-quarantine" in
  let wal_directory = Filename.concat state "generation-quarantine-wal" in
  let receipt_directory = Filename.concat state "generation-quarantine-receipts" in
  ensure_directory quarantine_root;
  ensure_directory wal_directory;
  ensure_directory receipt_directory;
  if Sys.file_exists observation.destination then failf "quarantine-destination-exists";
  let wal_name = observation.transaction_digest ^ ".wal" in
  let wal_path =
    Loom_hook_generation_guardian.atomic_write wal_directory wal_name
      (wal_text observation "PREPARED")
  in
  (wal_path, receipt_directory)

let require_same_observation expected observed =
  if observed.transaction_digest <> expected.transaction_digest
     || observed.record_digest <> expected.record_digest
     || observed.inventory_digest <> expected.inventory_digest
     || observed.related_digest <> expected.related_digest
     || observed.kernel_digest <> expected.kernel_digest
     || observed.cause <> expected.cause
  then failf "causal-observation-changed-before-commit"

let commit_quarantine policy state observation output frame receipt_directory wal_path =
  Unix.mkdir observation.destination 0o700;
  Loom_hook_generation_guardian.fsync_directory (Filename.dirname observation.destination);
  let moved = ref [] in
  let rollback () =
    List.iter
      (fun (source, target) ->
        if Sys.file_exists target && not (Sys.file_exists source) then
          try Unix.rename target source with _ -> ())
      !moved
  in
  try
    List.iter
      (fun artifact ->
        let target =
          Filename.concat observation.destination
            (String.map (function '/' -> '_' | character -> character) artifact.relative)
        in
        Unix.rename artifact.source target;
        moved := (artifact.source, target) :: !moved)
      observation.related;
    List.iter
      (fun artifact ->
        Loom_hook_generation_guardian.fsync_directory
          (Filename.dirname artifact.source))
      observation.related;
    Loom_hook_generation_guardian.fsync_directory observation.destination;
    let receipt = receipt_text policy observation "QUARANTINE_READY" output frame state in
    let receipt_name = observation.transaction_digest ^ ".receipt" in
    let receipt_path =
      Loom_hook_generation_guardian.atomic_write receipt_directory receipt_name receipt
    in
    ignore
      (Loom_hook_generation_guardian.atomic_write (Filename.dirname wal_path)
         (Filename.basename wal_path) (wal_text observation "COMMITTED"));
    (receipt_path, List.length !moved)
  with error ->
    rollback ();
    raise error

let result_json policy observation decision output frame ~applied ~moved ~receipt =
  Printf.sprintf
    "{\"schema\":\"loom-native-hook-generation-reconcile-v1\",\"action\":9047,\"stage\":\"PARITY_OPEN\",\"semantic_authority\":\"Sounio\",\"operational_realization\":\"OCaml\",\"decision\":\"%s\",\"absence_reason\":\"%s\",\"agent\":\"%s\",\"lane\":\"%s\",\"record_sha256\":\"%s\",\"inventory_sha256\":\"%s\",\"related_artifacts_sha256\":\"%s\",\"kernel_observation_sha256\":\"%s\",\"transaction_sha256\":\"%s\",\"related_total\":%d,\"mutation_applied\":%s,\"moved_artifacts\":%d,\"receipt\":\"%s\",\"python_executed\":false,\"rust_executed\":false,\"disposable_oracle_executed\":false,\"same_uid_peer_isolation\":false,\"authority\":{\"manifest_sha256\":\"%s\",\"semantics_sha256\":\"%s\",\"executable_sha256\":\"%s\",\"frame_sha256\":\"%s\",\"output_sha256\":\"%s\",\"output\":\"%s\"}}"
    decision (cause_name observation.cause) (json_escape observation.agent)
    (json_escape observation.lane) observation.record_digest
    observation.inventory_digest observation.related_digest observation.kernel_digest
    observation.transaction_digest (List.length observation.related)
    (string_of_bool applied) moved (json_escape receipt) policy.manifest_sha256
    semantics_sha256 policy.executable_sha256 (sha256 frame) (sha256 output)
    (json_escape output)

let fail_closed_json reason =
  Printf.sprintf
    "{\"schema\":\"loom-native-hook-generation-reconcile-v1\",\"action\":9047,\"stage\":\"PARITY_OPEN\",\"semantic_authority\":\"Sounio\",\"operational_realization\":\"OCaml\",\"decision\":\"FAIL_CLOSED\",\"reason\":\"%s\",\"mutation_applied\":false,\"python_executed\":false,\"rust_executed\":false,\"disposable_oracle_executed\":false,\"same_uid_peer_isolation\":false}"
    (json_escape reason)

let plan_json ~cwd ~agent ~lane =
  try
    if agent = "" || lane = "" then failf "agent-and-lane-required";
    let root = find_source_root (Unix.realpath cwd) in
    let policy = load_policy root in
    let state = state_directory root in
    with_state_lock state (fun () ->
        let observation = observe_locked state agent lane in
        let decision, output, frame =
          evaluate policy root observation 1 ~transaction_ready:true
        in
        if starts_with decision "DENY" then
          failf "authority-refused-plan:%s" decision;
        result_json policy observation decision output frame ~applied:false ~moved:0
          ~receipt:"")
  with error -> fail_closed_json (Printexc.to_string error)

let parse_arguments arguments =
  let rec loop cwd agent lane apply = function
    | [] -> (cwd, agent, lane, apply)
    | "--cwd" :: value :: tail -> loop (Some value) agent lane apply tail
    | "--agent" :: value :: tail -> loop cwd (Some value) lane apply tail
    | "--lane" :: value :: tail -> loop cwd agent (Some value) apply tail
    | "--apply" :: tail -> loop cwd agent lane true tail
    | option :: _ -> failf "unknown-option:%s" option
  in
  loop None None None false arguments

let run arguments =
  try
    let cwd, agent, lane, apply = parse_arguments arguments in
    let cwd = Option.value ~default:(Unix.getcwd ()) cwd |> Unix.realpath in
    let agent = Option.value ~default:"" agent in
    let lane = Option.value ~default:"" lane in
    if agent = "" || lane = "" then failf "agent-and-lane-required";
    let root = find_source_root cwd in
    let policy = load_policy root in
    let state = state_directory root in
    let json =
      with_state_lock state (fun () ->
          let transaction = ref "unbound" in
          try
            let observation = observe_locked state agent lane in
            transaction := observation.transaction_digest;
            if apply && observation.cause <> No_absence then (
              let wal_path, receipt_directory = move_to_quarantine state observation in
              require_same_observation observation
                (observe_locked state agent lane);
              let decision, output, frame =
                evaluate policy root observation 2 ~transaction_ready:true
              in
              if decision <> "QUARANTINE_READY" then
                failf "authority-refused-quarantine:%s" decision;
              require_same_observation observation
                (observe_locked state agent lane);
              let receipt, moved =
                commit_quarantine policy state observation output frame receipt_directory
                  wal_path
              in
              append_decision state
                (audit_line ~verdict:"ALLOW" ~decision ~reason:(cause_name observation.cause)
                   ~agent ~lane ~transaction:observation.transaction_digest);
              result_json policy observation decision output frame ~applied:true ~moved
                ~receipt)
            else
              let decision, output, frame =
                evaluate policy root observation 1 ~transaction_ready:true
              in
              if starts_with decision "DENY" then
                failf "authority-refused-plan:%s" decision;
              append_decision state
                (audit_line ~verdict:"ALLOW" ~decision ~reason:(cause_name observation.cause)
                   ~agent ~lane ~transaction:observation.transaction_digest);
              result_json policy observation decision output frame ~applied:false ~moved:0
                ~receipt:""
          with error ->
            let reason = Printexc.to_string error in
            append_decision state
              (audit_line ~verdict:"DENY" ~decision:"FAIL_CLOSED" ~reason ~agent
                 ~lane ~transaction:!transaction);
            raise error)
    in
    print_endline json;
    0
  with error ->
    print_endline (fail_closed_json (Printexc.to_string error));
    42
