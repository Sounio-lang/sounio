open Unix

exception Error of string

let pinned_manifest_sha256 =
  "9a40674a135a4c4f43ae0ba8a2658eba32e311b6cedaad5c26124eb6de657ca1"

let semantics_sha256 =
  "00c5d07b77434b37844e3704dd935d04367646c4f8541a8cce77bc143deb46a3"

let max_file_bytes = 8 * 1024 * 1024

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let starts_with value prefix =
  String.length value >= String.length prefix
  && String.sub value 0 (String.length prefix) = prefix

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let json_escape value =
  let buffer = Buffer.create (String.length value + 8) in
  String.iter
    (fun character ->
      match character with
      | '"' -> Buffer.add_string buffer "\\\""
      | '\\' -> Buffer.add_string buffer "\\\\"
      | '\n' -> Buffer.add_string buffer "\\n"
      | '\r' -> Buffer.add_string buffer "\\r"
      | '\t' -> Buffer.add_string buffer "\\t"
      | character when Char.code character < 32 ->
          Buffer.add_string buffer (Printf.sprintf "\\u%04x" (Char.code character))
      | _ -> Buffer.add_char buffer character)
    value;
  Buffer.contents buffer

let require_regular_file label path =
  let stat =
    try Unix.lstat path
    with Unix_error (ENOENT, _, _) -> failf "%s-missing:%s" label path
  in
  if stat.st_kind <> S_REG then failf "%s-not-regular:%s" label path;
  if stat.st_size > max_file_bytes then failf "%s-too-large:%s" label path;
  stat

let read_file label path =
  ignore (require_regular_file label path);
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () -> really_input_string channel (in_channel_length channel))

let sha256_file label path = sha256 (read_file label path)

let parse_fields ?(allow_duplicate = false) label text =
  let table = Hashtbl.create 32 in
  String.split_on_char '\n' text
  |> List.iter (fun line ->
         if line <> "" then
           match String.index_opt line '=' with
           | None -> failf "%s-malformed-field" label
           | Some split ->
               let key = String.sub line 0 split in
               let value =
                 String.sub line (split + 1) (String.length line - split - 1)
               in
               if key = "" || value = "" then failf "%s-empty-field" label;
               if Hashtbl.mem table key && not allow_duplicate then
                 failf "%s-duplicate-field:%s" label key;
               if not (Hashtbl.mem table key) then Hashtbl.add table key value);
  table

let required label table key =
  match Hashtbl.find_opt table key with
  | Some value when value <> "" -> value
  | _ -> failf "%s-field-missing:%s" label key

let exact label table key expected =
  let actual = required label table key in
  if actual <> expected then failf "%s-field-invalid:%s:%s" label key actual

let decimal label value =
  if value = ""
     || not (String.for_all (function '0' .. '9' -> true | _ -> false) value)
  then failf "%s-not-decimal" label;
  try int_of_string value with _ -> failf "%s-out-of-range" label

let boolean label = function
  | "true" -> true
  | "false" -> false
  | value -> failf "%s-not-boolean:%s" label value

let rec find_operational_root path =
  let candidate = Filename.concat path ".git" in
  if Sys.file_exists candidate then path
  else
    let parent = Filename.dirname path in
    if parent = path then failf "operational-root-not-found:%s" path
    else find_operational_root parent

let find_source_root = find_operational_root

let resolve_relative base value =
  if Filename.is_relative value then Filename.concat base value else value

let git_common_dir root =
  let dot_git = Filename.concat root ".git" in
  let stat = Unix.lstat dot_git in
  if stat.st_kind = S_DIR then Unix.realpath dot_git
  else if stat.st_kind = S_REG then (
    let line = String.trim (read_file "gitdir" dot_git) in
    let prefix = "gitdir: " in
    if not (starts_with line prefix) then failf "gitdir-malformed";
    let gitdir =
      String.sub line (String.length prefix) (String.length line - String.length prefix)
      |> resolve_relative root |> Unix.realpath
    in
    let commondir = Filename.concat gitdir "commondir" in
    if Sys.file_exists commondir then
      read_file "commondir" commondir |> String.trim |> resolve_relative gitdir
      |> Unix.realpath
    else gitdir)
  else failf "gitdir-not-file-or-directory"

type policy = {
  runtime : string;
  manifest_sha256 : string;
  executable_sha256 : string;
  draining_word : int;
  cutover_ready_word : int;
  sabotage_count : int;
  sabotage_required : int;
}

let choose_authority_runtime root manifest =
  let repository_runtime =
    Filename.concat root
      "tools/loom/_build/default/src/sounio-loom-native-hook-generation-drain"
  in
  let installed_runtime =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-native-hook-generation-drain"
  in
  let selected =
    match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_GENERATION_DRAIN_RUNTIME" with
    | Some value when value <> "" ->
        if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
          failf "authority-runtime-override-requires-test-mode";
        value
    | _ -> if Sys.file_exists installed_runtime then installed_runtime else repository_runtime
  in
  let stat = require_regular_file "authority-runtime" selected in
  if stat.st_perm land 0o111 = 0 then failf "authority-runtime-not-executable";
  let expected = required "freeze-manifest" manifest "executable_sha256" in
  if sha256_file "authority-runtime" selected <> expected then
    failf "authority-runtime-hash-mismatch";
  Unix.realpath selected

let verify_manifest_file root manifest path_key hash_key reason =
  let path = Filename.concat root (required "freeze-manifest" manifest path_key) in
  let expected = required "freeze-manifest" manifest hash_key in
  if sha256_file reason path <> expected then failf "%s-hash-mismatch" reason

let load_policy root =
  let installed_policy_root =
    Filename.concat
      (Filename.dirname (Filename.dirname (Unix.realpath Sys.executable_name)))
      "policy/native-hook-generation-drain"
  in
  let installed_manifest =
    Filename.concat installed_policy_root
      "tools/loom/native_hook_generation_drain.freeze.v1"
  in
  let path, policy_root =
    match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_GENERATION_DRAIN_MANIFEST" with
    | Some value when value <> "" ->
        if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
          failf "freeze-manifest-override-requires-test-mode";
        (value, root)
    | _ when Sys.file_exists installed_manifest ->
        (installed_manifest, installed_policy_root)
    | _ ->
        (Filename.concat root "tools/loom/native_hook_generation_drain.freeze.v1", root)
  in
  if sha256_file "freeze-manifest" path <> pinned_manifest_sha256 then
    failf "freeze-manifest-hash-mismatch";
  let manifest = parse_fields "freeze-manifest" (read_file "freeze-manifest" path) in
  exact "freeze-manifest" manifest "schema"
    "loom-native-hook-generation-drain-freeze-v1";
  exact "freeze-manifest" manifest "stage" "SEMANTICS_FROZEN";
  exact "freeze-manifest" manifest "semantic_authority" "Sounio";
  exact "freeze-manifest" manifest "producing_language" "Sounio";
  exact "freeze-manifest" manifest "language_role" "SEMANTIC_AUTHORITY";
  exact "freeze-manifest" manifest "action" "9046";
  exact "freeze-manifest" manifest "parent_action" "9045-frozen";
  exact "freeze-manifest" manifest "semantics_sha256" semantics_sha256;
  exact "freeze-manifest" manifest "load_bearing_rule" "inventory_complete";
  exact "freeze-manifest" manifest "causal_sabotage"
    "inventory-completeness-rule-removed";
  List.iter
    (fun (path_key, hash_key, reason) ->
      verify_manifest_file policy_root manifest path_key hash_key reason)
    [ ("garden_path", "garden_sha256", "garden");
      ("source_path", "source_sha256", "authority-source");
      ("entrypoint_path", "entrypoint_sha256", "authority-entrypoint");
      ("build_script_path", "build_script_sha256", "authority-build-script");
      ("selftest_path", "selftest_sha256", "authority-selftest");
      ("freeze_selftest_path", "freeze_selftest_sha256", "authority-freeze-selftest");
      ("first_manifest_path", "first_manifest_sha256", "first-manifest");
      ("first_evidence_path", "first_evidence_sha256", "first-evidence");
      ("parent_9045_freeze_path", "parent_9045_freeze_sha256", "parent-9045");
      ("toolchain_wrapper_path", "toolchain_wrapper_sha256", "toolchain-wrapper");
      ("toolchain_compiler_path", "toolchain_compiler_sha256", "toolchain-compiler") ];
  { runtime = choose_authority_runtime root manifest;
    manifest_sha256 = pinned_manifest_sha256;
    executable_sha256 = required "freeze-manifest" manifest "executable_sha256";
    draining_word = required "freeze-manifest" manifest "draining_word" |> decimal "draining-word";
    cutover_ready_word =
      required "freeze-manifest" manifest "cutover_ready_word"
      |> decimal "cutover-ready-word";
    sabotage_count =
      required "freeze-manifest" manifest "sabotage_count" |> decimal "sabotage-count";
    sabotage_required =
      required "freeze-manifest" manifest "sabotage_required"
      |> decimal "sabotage-required" }

type classification = Native | Legacy | Unknown | Unresponsive

let classification_name = function
  | Native -> "native"
  | Legacy -> "legacy"
  | Unknown -> "unknown"
  | Unresponsive -> "unresponsive"

type member = {
  agent : string;
  lane : string;
  harness : string;
  session_id : string;
  generation : string;
  pid : string;
  pid_start : string;
  boot_id : string;
  pid_namespace : string;
  presence_state : string;
  presence_reason : string;
  worktree : string;
  classification : classification;
  capability_reason : string;
}

type observation = {
  snapshot_utc : string;
  inventory_fresh : bool;
  inventory_complete : bool;
  classification_complete : bool;
  process_generation_bound : bool;
  hook_capability_bound : bool;
  old_runtime_bound : bool;
  candidate_runtime_bound : bool;
  candidate_config_bound : bool;
  final_config_bound : bool;
  canary_mask : int;
  rollback_pair_tested : bool;
  native_entry_open : bool;
  bridge_free_candidate : bool;
  current_legacy_bridge : bool;
  activation_requested : bool;
  zero_legacy_claimed : bool;
  total : int;
  classified : int;
  native : int;
  legacy : int;
  unknown : int;
  unresponsive : int;
  inventory_sha256 : string;
  old_runtime_sha256 : string;
  candidate_runtime_sha256 : string;
  config_pair_sha256 : string;
  current_runtime_id : string;
  candidate_runtime_id : string;
  members : member list;
}

let field table key = Option.value ~default:"" (Hashtbl.find_opt table key)

let positive_decimal label value =
  let parsed = decimal label value in
  if parsed = 0 then failf "%s-not-positive" label;
  parsed

let read_stream label path =
  let channel =
    try open_in_bin path
    with Sys_error message -> failf "%s-open-refused:%s" label message
  in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let buffer = Buffer.create 256 in
      let chunk = Bytes.create 4096 in
      let rec consume total =
        let count = input channel chunk 0 (Bytes.length chunk) in
        if count = 0 then Buffer.contents buffer
        else if total + count > max_file_bytes then failf "%s-too-large:%s" label path
        else (
          Buffer.add_subbytes buffer chunk 0 count;
          consume (total + count))
      in
      consume 0)

let utc_now timestamp =
  let tm = Unix.gmtime timestamp in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02dZ" (tm.tm_year + 1900)
    (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min tm.tm_sec

let has_suffix value suffix =
  let value_length = String.length value and suffix_length = String.length suffix in
  value_length >= suffix_length
  && String.sub value (value_length - suffix_length) suffix_length = suffix

let inventory_paths directory =
  let stat =
    try Unix.lstat directory
    with Unix_error (ENOENT, _, _) -> failf "presence-directory-missing:%s" directory
  in
  if stat.st_kind <> S_DIR then failf "presence-directory-not-directory:%s" directory;
  Sys.readdir directory |> Array.to_list
  |> List.filter (fun name -> has_suffix name ".presence")
  |> List.sort String.compare
  |> List.map (fun name ->
         let path = Filename.concat directory name in
         ignore (require_regular_file "presence-record" path);
         (name, path))

let process_exists pid =
  try
    Unix.kill pid 0;
    true
  with Unix_error _ -> false

let process_start pid =
  let path = Printf.sprintf "/proc/%d/stat" pid in
  let stat = String.trim (read_stream "process-stat" path) in
  let closing =
    try String.rindex stat ')'
    with Not_found -> failf "process-stat-malformed:%d" pid
  in
  if closing + 2 > String.length stat then failf "process-stat-malformed:%d" pid;
  let tail = String.sub stat (closing + 2) (String.length stat - closing - 2) in
  let fields =
    String.split_on_char ' ' tail |> List.filter (fun value -> value <> "")
  in
  try List.nth fields 19 with _ -> failf "process-stat-missing-start:%d" pid

let presence_state ~now ~boot_id ~pid_namespace values =
  try
    let pid =
      required "presence-record" values "pid" |> positive_decimal "presence-pid"
    in
    let expected_start = required "presence-record" values "pid_start" in
    ignore (positive_decimal "presence-pid-start" expected_start);
    let last_seen =
      required "presence-record" values "last_seen_epoch"
      |> decimal "presence-last-seen"
    in
    let ttl =
      required "presence-record" values "ttl_seconds"
      |> positive_decimal "presence-ttl"
    in
    if required "presence-record" values "boot_id" <> boot_id then
      ("orphaned", "boot-changed")
    else if required "presence-record" values "pid_namespace" <> pid_namespace then
      ("orphaned", "pid-namespace-changed")
    else if
      not (Sys.file_exists (Printf.sprintf "/proc/%d/stat" pid))
      || not (process_exists pid)
    then ("orphaned", "process-missing")
    else if process_start pid <> expected_start then ("orphaned", "pid-reused")
    else if int_of_float now > last_seen + ttl then
      ("unresponsive", "heartbeat-expired")
    else ("live", "process-verified")
  with Error _ -> ("orphaned", "invalid-record")

let slug value =
  let buffer = Buffer.create (min 80 (String.length value)) in
  String.iter
    (fun character ->
      if Buffer.length buffer < 80 then
        match character with
        | 'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' | '.' | '_' | '-' ->
            Buffer.add_char buffer character
        | _ -> Buffer.add_char buffer '_')
    value;
  if Buffer.length buffer = 0 then "unnamed" else Buffer.contents buffer

let provider_harness = function
  | "codex" | "claude" | "cursor" | "grok" -> true
  | _ -> false

let executable path =
  try
    let stat = Unix.stat path in
    stat.st_kind = S_REG && stat.st_perm land 0o111 <> 0
  with _ -> false

let runtime_identity path =
  if path = "" then ("", "")
  else
    let manifest_path = Filename.concat path "manifest" in
    try
      let fields =
        parse_fields ~allow_duplicate:true "runtime-manifest"
          (read_file "runtime-manifest" manifest_path)
      in
      (field fields "runtime_id", sha256_file "runtime-manifest" manifest_path)
    with _ -> ("", "")

let sorted_directories path =
  if not (Sys.file_exists path) then []
  else
    Sys.readdir path |> Array.to_list |> List.sort (fun a b -> String.compare b a)
    |> List.filter_map (fun name ->
           let candidate = Filename.concat path name in
           try if (Unix.stat candidate).st_kind = S_DIR then Some candidate else None
           with _ -> None)

let bridge_free_runtime path =
  let manifest_path = Filename.concat path "manifest" in
  let python_bridge = Filename.concat path "hooks/sounio_coord_agent_hook_runtime.py" in
  let native = Filename.concat path "bin/sounio-loom-runtime" in
  try
    let fields =
      parse_fields ~allow_duplicate:true "runtime-manifest"
        (read_file "runtime-manifest" manifest_path)
    in
    field fields "loom_native_hook_cutover_python_bridge_absent" = "true"
    && not (Sys.file_exists python_bridge) && executable native
  with _ -> false

let find_candidate runtime_root current =
  match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_CANDIDATE" with
  | Some value when value <> "" && Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1" ->
      let resolved = Unix.realpath value in
      if resolved = current then failf "candidate-override-selects-current-runtime";
      if not (bridge_free_runtime resolved) then failf "candidate-override-not-bridge-free";
      resolved
  | Some value when value <> "" -> failf "candidate-override-requires-test-mode"
  | _ ->
      let links = [ "native-next"; "next" ] in
      let from_link =
        List.find_map
          (fun name ->
            let path = Filename.concat runtime_root name in
            try
              let resolved = Unix.realpath path in
              if resolved <> current && bridge_free_runtime resolved then Some resolved else None
            with _ -> None)
          links
      in
      (match from_link with
      | Some path -> path
      | None ->
          Filename.concat runtime_root "versions" |> sorted_directories
          |> List.find_opt (fun path -> path <> current && bridge_free_runtime path)
          |> Option.value ~default:"")

let capability_for common member current candidate =
  let path =
    Filename.concat common
      (Printf.sprintf "sounio-coord-state/hook-capabilities/%s--%s.capability"
         (slug member.agent) (slug member.lane))
  in
  if not (Sys.file_exists path) then (Unknown, "capability-absent")
  else
    try
      let values = parse_fields "hook-capability" (read_file "hook-capability" path) in
      let current_id, _ = runtime_identity current in
      let candidate_id, _ = runtime_identity candidate in
      let matches =
        field values "schema" = "loom-native-hook-capability-v1"
        && field values "state" = "NATIVE_HOOK_ATTESTED"
        && field values "agent" = member.agent && field values "lane" = member.lane
        && field values "session_id" = member.session_id
        && field values "generation" = member.generation
        && field values "worktree" = member.worktree
        && field values "harness" = member.harness
        && field values "presence_pid" = member.pid
        && field values "presence_pid_start" = member.pid_start
        && field values "presence_boot_id" = member.boot_id
        && field values "presence_pid_namespace" = member.pid_namespace
        && field values "caller_pid" = member.pid
        && field values "caller_pid_start" = member.pid_start
        && field values "caller_boot_id" = member.boot_id
        && field values "caller_pid_namespace" = member.pid_namespace
        && field values "wake_eligible" = "1"
      in
      if not matches then (Unknown, "capability-generation-drift")
      else
        let created = field values "created_epoch" |> decimal "capability-created" in
        let expires = field values "expires_epoch" |> decimal "capability-expiry" in
        let now = int_of_float (Unix.time ()) in
        if created > now || expires < created then (Unknown, "capability-time-invalid")
        else if expires < now then (Unknown, "capability-expired")
        else
          let runtime_id = field values "runtime_id" in
          let runtime, classification, reason =
            if candidate_id <> "" && runtime_id = candidate_id then
              (candidate, Native, "candidate-native-attested")
            else if current_id <> "" && runtime_id = current_id then
              (current, Legacy, "current-generation-attested")
            else ("", Unknown, "capability-runtime-unclassified")
          in
          if runtime = "" then (classification, reason)
          else
            let manifest =
              parse_fields ~allow_duplicate:true "runtime-manifest"
                (read_file "runtime-manifest" (Filename.concat runtime "manifest"))
            in
            let producer = Unix.realpath (required "hook-capability" values "producer_executable") in
            let coord = Unix.realpath (required "hook-capability" values "coord_executable") in
            let caller = Unix.realpath (required "hook-capability" values "caller_executable") in
            let expected_producer = Unix.realpath (Filename.concat runtime "bin/sounio-loom-runtime") in
            let expected_coord = Unix.realpath (Filename.concat runtime "bin/sounio-coord-runtime") in
            let live_caller = Unix.readlink (Printf.sprintf "/proc/%s/exe" member.pid) |> Unix.realpath in
            let bound =
              producer = expected_producer && coord = expected_coord && caller = live_caller
              && executable producer && executable coord && executable caller
              && sha256_file "capability-producer" producer
                 = required "hook-capability" values "producer_sha256"
              && sha256_file "capability-coord" coord
                 = required "hook-capability" values "coord_sha256"
              && sha256_file "capability-caller" caller
                 = required "hook-capability" values "caller_sha256"
              && field manifest "runtime_id" = runtime_id
              && field manifest "source_sha" = field values "source_sha"
              && field manifest "loom_runtime_sha256" = field values "producer_sha256"
              && field manifest "coord_runtime_sha256" = field values "coord_sha256"
            in
            if bound then (classification, reason)
            else (Unknown, "capability-binary-or-manifest-drift")
    with _ -> (Unknown, "capability-invalid")

let count classification members =
  List.fold_left
    (fun total member -> if member.classification = classification then total + 1 else total)
    0 members

let digest_regular path =
  try sha256_file "binding" path with _ -> String.make 64 '0'

let marker_directory common =
  match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR" with
  | Some value when value <> "" ->
      if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
        failf "drain-state-override-requires-test-mode";
      value
  | _ -> Filename.concat common "sounio-coord-runtime/native-hook-drain"

let read_marker common name =
  let path = Filename.concat (marker_directory common) name in
  if Sys.file_exists path then Some (read_file "drain-marker" path |> String.trim) else None

let config_bindings =
  [ (".codex/hooks.json", "loom_native_hook_cutover_codex_config_sha256");
    (".claude/settings.json", "loom_native_hook_cutover_claude_config_sha256");
    (".cursor/hooks.json", "loom_native_hook_cutover_cursor_config_sha256");
    (".grok/hooks/loom-native.json", "loom_native_hook_cutover_grok_config_sha256") ]

let config_bundle_sha256 root =
  config_bindings
  |> List.map (fun (relative, _) ->
         relative ^ "\000" ^ digest_regular (Filename.concat root relative))
  |> String.concat "\000" |> fun value -> sha256 ("loom-hook-config-bundle-v1\000" ^ value)

let candidate_config_binding root candidate =
  try
    let manifest =
      parse_fields ~allow_duplicate:true "runtime-manifest"
        (read_file "runtime-manifest" (Filename.concat candidate "manifest"))
    in
    let matched =
      List.for_all
        (fun (relative, manifest_key) ->
          sha256_file "hook-config" (Filename.concat root relative)
          = required "runtime-manifest" manifest manifest_key)
        config_bindings
    in
    (matched, config_bundle_sha256 root)
  with _ -> (false, String.make 64 '0')

let valid_final_config_marker marker candidate_id candidate_runtime_sha256 config_sha256
    guardian_public_key_sha256 =
  try
    let values = parse_fields "final-config-marker" marker in
    field values "schema" = "loom-native-hook-final-config-v1"
    && field values "state" = "FINAL_CONFIG_BOUND"
    && field values "runtime_id" = candidate_id
    && field values "runtime_manifest_sha256" = candidate_runtime_sha256
    && field values "config_bundle_sha256" = config_sha256
    && field values "guardian_public_key_sha256" = guardian_public_key_sha256
    && field values "semantic_authority" = "Sounio"
    && field values "action" = "9046"
  with _ -> false

let valid_rollback_marker marker current_id old_runtime_sha256 candidate_id
    candidate_runtime_sha256 config_sha256 =
  try
    let values = parse_fields "rollback-marker" marker in
    field values "schema" = "loom-native-hook-rollback-pair-v1"
    && field values "state" = "ROLLBACK_PAIR_TESTED"
    && field values "forward_result" = "PASS"
    && field values "rollback_result" = "PASS"
    && field values "old_runtime_id" = current_id
    && field values "old_runtime_manifest_sha256" = old_runtime_sha256
    && field values "candidate_runtime_id" = candidate_id
    && field values "candidate_runtime_manifest_sha256" = candidate_runtime_sha256
    && field values "config_bundle_sha256" = config_sha256
    && field values "semantic_authority" = "Sounio"
    && field values "action" = "9046"
  with _ -> false

type inventory_record = {
  inventory_name : string;
  inventory_text : string;
  inventory_digest : string;
  inventory_error : string option;
}

let capture_inventory directory =
  inventory_paths directory
  |> List.map (fun (name, path) ->
         let stat = Unix.lstat path in
         try
           let text = read_file "presence-record" path in
           { inventory_name = name; inventory_text = text;
             inventory_digest = sha256 text; inventory_error = None }
         with _ ->
           let metadata =
             Printf.sprintf "%d:%d:%.0f:%.0f:%d:%d" stat.st_ino stat.st_size
               stat.st_mtime stat.st_ctime stat.st_perm stat.st_uid
           in
           { inventory_name = name; inventory_text = "";
             inventory_digest = sha256 ("unreadable-presence-v1\000" ^ name ^ "\000" ^ metadata);
             inventory_error = Some "presence-record-unreadable" })

let inventory_signature records =
  records
  |> List.map (fun record -> record.inventory_name ^ "\000" ^ record.inventory_digest)
  |> String.concat "\000"

let inferred_identity name =
  let base = String.sub name 0 (String.length name - String.length ".presence") in
  let rec delimiter index =
    if index + 1 >= String.length base then None
    else if base.[index] = '-' && base.[index + 1] = '-' then Some index
    else delimiter (index + 1)
  in
  match delimiter 0 with
  | Some split ->
      (String.sub base 0 split,
       String.sub base (split + 2) (String.length base - split - 2))
  | _ -> ("unknown", base)

let invalid_member record reason =
  let agent, lane = inferred_identity record.inventory_name in
  ({ agent; lane; harness = agent; session_id = "unknown"; generation = "invalid";
     pid = "unknown"; pid_start = "unknown"; boot_id = "unknown";
     pid_namespace = "unknown"; presence_state = "orphaned"; presence_reason = reason;
     worktree = "unknown"; classification = Unresponsive;
     capability_reason = reason }, false)

let member_from_record ~now ~boot_id ~pid_namespace record =
  match record.inventory_error with
  | Some reason -> invalid_member record reason
  | None ->
      (try
         let values =
           parse_fields ("presence-record:" ^ record.inventory_name) record.inventory_text
         in
         let agent = required "presence-record" values "agent" in
         let lane = required "presence-record" values "lane" in
         let expected_id = slug agent ^ "--" ^ slug lane in
         exact "presence-record" values "presence_id" expected_id;
         if record.inventory_name <> expected_id ^ ".presence" then
           failf "presence-record-name-mismatch:%s" record.inventory_name;
         let state, reason = presence_state ~now ~boot_id ~pid_namespace values in
         ({ agent; lane; harness = required "presence-record" values "harness";
            session_id = required "presence-record" values "session_id";
            generation = required "presence-record" values "generation";
            pid = required "presence-record" values "pid";
            pid_start = required "presence-record" values "pid_start";
            boot_id = required "presence-record" values "boot_id";
            pid_namespace = required "presence-record" values "pid_namespace";
            presence_state = state;
            presence_reason = reason;
            worktree = required "presence-record" values "worktree";
            classification = Unknown; capability_reason = "unmeasured" }, true)
       with Error _ -> invalid_member record "presence-record-invalid")

let live_observation ?(verify_canaries = true) root =
  let common = git_common_dir root in
  let runtime_root = Filename.concat common "sounio-coord-runtime" in
  let current =
    try Unix.realpath (Filename.concat runtime_root "current") with _ -> ""
  in
  let candidate = find_candidate runtime_root current in
  let current_id, old_runtime_sha256 = runtime_identity current in
  let candidate_id, candidate_runtime_sha256 = runtime_identity candidate in
  let now = Unix.time () in
  let presence_directory = Filename.concat common "sounio-coord-state/process-presences" in
  let first_inventory = capture_inventory presence_directory in
  let second_inventory = capture_inventory presence_directory in
  let first_signature = inventory_signature first_inventory in
  let second_signature = inventory_signature second_inventory in
  if first_signature <> second_signature then failf "presence-inventory-changed-during-snapshot";
  let boot_id =
    read_stream "kernel-boot-id" "/proc/sys/kernel/random/boot_id" |> String.trim
  in
  if boot_id = "" then failf "kernel-boot-id-empty";
  let pid_namespace =
    try Unix.readlink "/proc/self/ns/pid"
    with Unix_error _ -> failf "pid-namespace-unavailable"
  in
  if pid_namespace = "" then failf "pid-namespace-empty";
  let parsed_members =
    List.map (member_from_record ~now ~boot_id ~pid_namespace) second_inventory
  in
  let inventory_complete =
    List.for_all (fun record -> Option.is_none record.inventory_error) second_inventory
    && List.for_all snd parsed_members
  in
  let raw_members =
    List.map fst parsed_members
    |> List.filter (fun member -> provider_harness member.harness)
  in
  let members =
    raw_members
    |> List.map (fun member ->
           if member.presence_state <> "live" then
             { member with classification = Unresponsive;
                           capability_reason = member.presence_reason }
           else
             let classification, capability_reason =
               capability_for common member current candidate
             in
             { member with classification; capability_reason })
  in
  let native = count Native members and legacy = count Legacy members in
  let unknown = count Unknown members and unresponsive = count Unresponsive members in
  let total = List.length members in
  let process_generation_bound =
    List.for_all
      (fun member ->
        member.generation <> ""
        && String.for_all (function '0' .. '9' -> true | _ -> false) member.generation)
      members
  in
  let hook_capability_bound =
    List.for_all
      (fun member ->
        member.classification = Native || member.classification = Legacy
        || member.classification = Unresponsive)
      members
  in
  let candidate_config_bound, repository_config_sha256 =
    candidate_config_binding root candidate
  in
  let candidate_loom_runtime_sha256 =
    try
      let manifest =
        parse_fields ~allow_duplicate:true "candidate-runtime-manifest"
          (read_file "candidate-runtime-manifest" (Filename.concat candidate "manifest"))
      in
      required "candidate-runtime-manifest" manifest "loom_runtime_sha256"
    with _ -> String.make 64 '0'
  in
  let canary_mask =
    if verify_canaries then
      Loom_hook_generation_canary.verified_mask
        ~state_directory:(marker_directory common) ~candidate_id
        ~candidate_manifest_sha256:candidate_runtime_sha256
        ~candidate_loom_runtime_sha256 ~config_bundle_sha256:repository_config_sha256
    else 0
  in
  let final_config = read_marker common "final-config.v1" in
  let rollback_marker = read_marker common "rollback-pair-tested.v1" in
  let guardian_public_key_sha256 =
    digest_regular
      (Filename.concat (marker_directory common) "guardian-ed25519-public.pem")
  in
  let final_config_bound =
    match final_config with
    | Some marker ->
        valid_final_config_marker marker candidate_id candidate_runtime_sha256
          repository_config_sha256 guardian_public_key_sha256
    | None -> false
  in
  let rollback_pair_tested =
    match rollback_marker with
    | Some marker ->
        valid_rollback_marker marker current_id old_runtime_sha256 candidate_id
          candidate_runtime_sha256 repository_config_sha256
    | None -> false
  in
  let config_pair_sha256 =
    sha256
      ("loom-hook-config-pair-v1\000"
      ^ repository_config_sha256 ^ "\000"
      ^ Option.value ~default:repository_config_sha256 final_config)
  in
  { snapshot_utc = utc_now now; inventory_fresh = true;
    inventory_complete; classification_complete = true;
    process_generation_bound; hook_capability_bound;
    old_runtime_bound = current_id <> "" && old_runtime_sha256 <> String.make 64 '0';
    candidate_runtime_bound =
      candidate_id <> "" && candidate_runtime_sha256 <> String.make 64 '0';
    candidate_config_bound; final_config_bound;
    canary_mask; rollback_pair_tested;
    native_entry_open = candidate <> "" && executable (Filename.concat candidate "bin/sounio-loom-runtime");
    bridge_free_candidate = candidate <> "" && bridge_free_runtime candidate;
    current_legacy_bridge =
      current <> ""
      && Sys.file_exists (Filename.concat current "hooks/sounio_coord_agent_hook_runtime.py");
    activation_requested = false; zero_legacy_claimed = false; total;
    classified = total; native; legacy; unknown; unresponsive;
    inventory_sha256 = sha256 ("loom-hook-presence-inventory-v1\000" ^ second_signature);
    old_runtime_sha256;
    candidate_runtime_sha256; config_pair_sha256; current_runtime_id = current_id;
    candidate_runtime_id = candidate_id; members }

let fixture_observation path =
  let values = parse_fields "generation-drain-fixture" (read_file "generation-drain-fixture" path) in
  exact "generation-drain-fixture" values "schema"
    "loom-native-hook-generation-drain-fixture-v1";
  let get_bool key = required "generation-drain-fixture" values key |> boolean key in
  let get_int key = required "generation-drain-fixture" values key |> decimal key in
  let get_hash key =
    let value = required "generation-drain-fixture" values key in
    if String.length value <> 64 then failf "%s-invalid-sha256" key;
    value
  in
  { snapshot_utc = required "generation-drain-fixture" values "snapshot_utc";
    inventory_fresh = get_bool "inventory_fresh";
    inventory_complete = get_bool "inventory_complete";
    classification_complete = get_bool "classification_complete";
    process_generation_bound = get_bool "process_generation_bound";
    hook_capability_bound = get_bool "hook_capability_bound";
    old_runtime_bound = get_bool "old_runtime_bound";
    candidate_runtime_bound = get_bool "candidate_runtime_bound";
    candidate_config_bound = get_bool "candidate_config_bound";
    final_config_bound = get_bool "final_config_bound";
    canary_mask = get_int "canary_mask";
    rollback_pair_tested = get_bool "rollback_pair_tested";
    native_entry_open = get_bool "native_entry_open";
    bridge_free_candidate = get_bool "bridge_free_candidate";
    current_legacy_bridge = get_bool "current_legacy_bridge";
    activation_requested = get_bool "activation_requested";
    zero_legacy_claimed = get_bool "zero_legacy_claimed";
    total = get_int "total"; classified = get_int "classified";
    native = get_int "native"; legacy = get_int "legacy";
    unknown = get_int "unknown"; unresponsive = get_int "unresponsive";
    inventory_sha256 = get_hash "inventory_sha256";
    old_runtime_sha256 = get_hash "old_runtime_sha256";
    candidate_runtime_sha256 = get_hash "candidate_runtime_sha256";
    config_pair_sha256 = get_hash "config_pair_sha256";
    current_runtime_id = required "generation-drain-fixture" values "current_runtime_id";
    candidate_runtime_id = required "generation-drain-fixture" values "candidate_runtime_id";
    members = [] }

let bit word shift enabled = if enabled then word lor (1 lsl shift) else word

let authority_word observation =
  0 |> fun word -> bit word 0 true
  |> fun word -> bit word 1 observation.old_runtime_bound
  |> fun word -> bit word 2 observation.candidate_runtime_bound
  |> fun word -> bit word 3 observation.candidate_config_bound
  |> fun word -> bit word 4 observation.final_config_bound
  |> fun word -> bit word 5 (observation.snapshot_utc <> "")
  |> fun word -> bit word 6 observation.inventory_fresh
  |> fun word -> bit word 7 observation.inventory_complete
  |> fun word -> bit word 8 observation.classification_complete
  |> fun word -> bit word 9 observation.process_generation_bound
  |> fun word -> bit word 10 observation.hook_capability_bound
  |> fun word -> bit word 11 (observation.canary_mask = 15)
  |> fun word -> bit word 12 observation.rollback_pair_tested
  |> fun word -> bit word 13 true
  |> fun word -> bit word 14 true
  |> fun word -> bit word 15 true
  |> fun word -> bit word 16 true
  |> fun word -> bit word 17 true
  |> fun word -> bit word 18 observation.activation_requested
  |> fun word -> bit word 19 observation.native_entry_open
  |> fun word -> bit word 20 observation.bridge_free_candidate
  |> fun word -> bit word 21 observation.current_legacy_bridge
  |> fun word -> bit word 22 observation.zero_legacy_claimed

let hash_u60 digest =
  if String.length digest < 15 then 0L
  else Int64.of_string ("0x" ^ String.sub digest 0 15)

let split_hash digest =
  (hash_u60 digest, hash_u60 (String.sub digest 15 (String.length digest - 15) |> sha256))

let mode observation = if observation.activation_requested then 2 else 1

let authority_frame policy observation =
  let inventory_hash0, inventory_hash1 = split_hash observation.inventory_sha256 in
  Printf.sprintf "%d %d %d %d %d %d %d %d %d %d %d %Ld %Ld %Ld %Ld %Ld %d %d\n"
    9046 (mode observation) 3 (authority_word observation) observation.total
    observation.classified observation.native observation.legacy observation.unknown
    observation.unresponsive observation.canary_mask inventory_hash0 inventory_hash1
    (hash_u60 observation.old_runtime_sha256)
    (hash_u60 observation.candidate_runtime_sha256)
    (hash_u60 observation.config_pair_sha256) policy.sabotage_count
    policy.sabotage_required

let decision_code = function
  | "DRAINING" -> 1
  | "CUTOVER_READY" -> 2
  | value when starts_with value "DENY" ->
      (try int_of_string (String.sub value 4 (String.length value - 4)) with _ -> 424)
  | _ -> 424

let block_reason = function
  | "DRAINING" -> "non-native-generations-remain"
  | "CUTOVER_READY" -> "none"
  | "DENY671" -> "parent-or-stage-not-frozen"
  | "DENY672" -> "runtime-or-config-generation-unbound"
  | "DENY673" -> "inventory-not-fresh-complete-and-classified"
  | "DENY674" -> "inventory-count-arithmetic-invalid"
  | "DENY675" -> "process-generation-or-hook-capability-unbound"
  | "DENY676" -> "forbidden-oracle-present"
  | "DENY677" -> "causal-sabotage-coverage-incomplete"
  | "DENY678" -> "canary-rollback-or-native-entry-incomplete"
  | "DENY679" -> "draining-mode-invariant-violated"
  | "DENY680" -> "affirmative-absence-not-proven"
  | "DENY681" -> "mode-invalid"
  | _ -> "authority-output-invalid"

let parse_decision output =
  let prefix = "SOUNIO_NATIVE_HOOK_GENERATION_DRAIN " in
  let suffix = " semantic_authority=Sounio action=9046" in
  if not (starts_with output prefix) then failf "authority-output-prefix-invalid";
  let length = String.length output - String.length prefix - String.length suffix in
  if length <= 0 || String.sub output (String.length output - String.length suffix) (String.length suffix) <> suffix
  then failf "authority-output-suffix-invalid";
  String.sub output (String.length prefix) length

let member_json member =
  Printf.sprintf
    "{\"agent\":\"%s\",\"lane\":\"%s\",\"harness\":\"%s\",\"session_id\":\"%s\",\"generation\":\"%s\",\"pid\":\"%s\",\"presence_state\":\"%s\",\"presence_reason\":\"%s\",\"worktree\":\"%s\",\"classification\":\"%s\",\"capability_reason\":\"%s\"}"
    (json_escape member.agent) (json_escape member.lane) (json_escape member.harness)
    (json_escape member.session_id) (json_escape member.generation) (json_escape member.pid)
    (json_escape member.presence_state) (json_escape member.presence_reason)
    (json_escape member.worktree) (classification_name member.classification)
    (json_escape member.capability_reason)

let snapshot_json policy observation frame output decision =
  let admitted = decision = "DRAINING" || decision = "CUTOVER_READY" in
  let ready = decision = "CUTOVER_READY" in
  Printf.sprintf
    "{\"schema\":\"loom-native-hook-generation-drain-snapshot-v1\",\"action\":9046,\"stage\":\"SEMANTICS_FROZEN\",\"semantic_authority\":\"Sounio\",\"operational_realization\":\"OCaml\",\"authority_observed\":true,\"decision\":\"%s\",\"decision_code\":%d,\"admitted\":%s,\"cutover_ready\":%s,\"cutover_command_exposed\":%s,\"block_reason\":\"%s\",\"snapshot_utc\":\"%s\",\"inventory\":{\"fresh\":%s,\"complete\":%s,\"classification_complete\":%s,\"process_generation_bound\":%s,\"hook_capability_bound\":%s,\"sha256\":\"%s\",\"total\":%d,\"classified\":%d,\"native\":%d,\"legacy\":%d,\"unknown\":%d,\"unresponsive\":%d},\"bindings\":{\"current_runtime_id\":\"%s\",\"candidate_runtime_id\":\"%s\",\"old_runtime_bound\":%s,\"candidate_runtime_bound\":%s,\"candidate_config_bound\":%s,\"final_config_bound\":%s,\"old_runtime_sha256\":\"%s\",\"candidate_runtime_sha256\":\"%s\",\"config_pair_sha256\":\"%s\"},\"canaries\":{\"mask\":%d,\"required_mask\":15,\"four_provider_complete\":%s},\"rollback_pair_tested\":%s,\"native_entry_open\":%s,\"bridge_free_candidate\":%s,\"current_legacy_bridge\":%s,\"authority\":{\"manifest_sha256\":\"%s\",\"semantics_sha256\":\"%s\",\"executable_sha256\":\"%s\",\"frame_sha256\":\"%s\",\"output_sha256\":\"%s\",\"output\":\"%s\"},\"members\":[%s]}"
    decision (decision_code decision) (string_of_bool admitted) (string_of_bool ready)
    (string_of_bool ready) (block_reason decision) (json_escape observation.snapshot_utc)
    (string_of_bool observation.inventory_fresh)
    (string_of_bool observation.inventory_complete)
    (string_of_bool observation.classification_complete)
    (string_of_bool observation.process_generation_bound)
    (string_of_bool observation.hook_capability_bound) observation.inventory_sha256
    observation.total observation.classified observation.native observation.legacy
    observation.unknown observation.unresponsive (json_escape observation.current_runtime_id)
    (json_escape observation.candidate_runtime_id)
    (string_of_bool observation.old_runtime_bound)
    (string_of_bool observation.candidate_runtime_bound)
    (string_of_bool observation.candidate_config_bound)
    (string_of_bool observation.final_config_bound) observation.old_runtime_sha256
    observation.candidate_runtime_sha256 observation.config_pair_sha256
    observation.canary_mask (string_of_bool (observation.canary_mask = 15))
    (string_of_bool observation.rollback_pair_tested)
    (string_of_bool observation.native_entry_open)
    (string_of_bool observation.bridge_free_candidate)
    (string_of_bool observation.current_legacy_bridge) policy.manifest_sha256 semantics_sha256
    policy.executable_sha256 (sha256 frame) (sha256 output) (json_escape output)
    (observation.members |> List.map member_json |> String.concat ",")

let evaluate_with_decision root observation =
  let policy = load_policy root in
  let frame = authority_frame policy observation in
  let result = Loom_hook.run_process ~input:frame ~timeout_seconds:5.0 ~cwd:root policy.runtime [] in
  let output = String.trim result.output in
  let decision = parse_decision output in
  let expected_code = if decision = "DRAINING" || decision = "CUTOVER_READY" then 0 else 42 in
  if result.code <> expected_code then
    failf "authority-exit-mismatch:decision=%s:expected=%d:observed=%d" decision expected_code result.code;
  (snapshot_json policy observation frame output decision, expected_code, decision)

let evaluate root observation =
  let snapshot, code, _decision = evaluate_with_decision root observation in
  (snapshot, code)

let attach_ui_attestation common snapshot =
  if String.length snapshot = 0 || snapshot.[String.length snapshot - 1] <> '}' then
    failf "ui-snapshot-json-invalid";
  let directory = marker_directory common in
  let private_key = Filename.concat directory "guardian-ed25519-private.pem" in
  let public_key = Filename.concat directory "guardian-ed25519-public.pem" in
  let key_manifest = Filename.concat directory "guardian-ed25519-key.v1" in
  let public_text = read_file "guardian-public-key" public_key in
  let public_sha256 = sha256 public_text in
  let private_sha256 = sha256_file "guardian-private-key" private_key in
  let key_id = Loom_epistemic.outcome_public_key_id public_text in
  let manifest =
    parse_fields "guardian-key-manifest" (read_file "guardian-key-manifest" key_manifest)
  in
  if field manifest "schema" <> "loom-native-hook-guardian-key-v1"
     || field manifest "algorithm" <> "ed25519"
     || field manifest "key_id" <> key_id
     || field manifest "private_key_sha256" <> private_sha256
     || field manifest "public_key_sha256" <> public_sha256
  then failf "guardian-key-manifest-drift";
  let payload_sha256 = sha256 snapshot in
  let signature = Loom_epistemic.outcome_ed25519_sign private_key snapshot in
  if not (Loom_epistemic.outcome_ed25519_verify public_text snapshot signature) then
    failf "ui-snapshot-signature-self-verification-refused";
  String.sub snapshot 0 (String.length snapshot - 1)
  ^ Printf.sprintf
      ",\"ui_attestation\":{\"schema\":\"loom-native-hook-ui-attestation-v1\",\"algorithm\":\"ed25519\",\"verified\":true,\"signed_payload_sha256\":\"%s\",\"key_id\":\"%s\",\"public_key_sha256\":\"%s\",\"signature_base64\":\"%s\",\"same_uid_peer_isolation\":false}}"
      payload_sha256 key_id public_sha256 (json_escape signature)

let evaluate_live root =
  let snapshot, code = evaluate root (live_observation root) in
  let common = git_common_dir root in
  (attach_ui_attestation common snapshot, code)

let cutover_observation observation =
  let affirmative_absence =
    observation.legacy = 0 && observation.unknown = 0
    && observation.unresponsive = 0
  in
  { observation with activation_requested = true;
    zero_legacy_claimed = affirmative_absence }

let evaluate_cutover_live root observation =
  let snapshot, _authority_code, decision =
    evaluate_with_decision root (cutover_observation observation)
  in
  let common = git_common_dir root in
  let signed_snapshot = attach_ui_attestation common snapshot in
  (signed_snapshot, if decision = "CUTOVER_READY" then 0 else 42)

let fail_closed_json reason =
  Printf.sprintf
    "{\"schema\":\"loom-native-hook-generation-drain-snapshot-v1\",\"action\":9046,\"stage\":\"SEMANTICS_FROZEN\",\"semantic_authority\":\"Sounio\",\"operational_realization\":\"OCaml\",\"authority_observed\":false,\"decision\":\"FAIL_CLOSED\",\"decision_code\":424,\"admitted\":false,\"cutover_ready\":false,\"cutover_command_exposed\":false,\"block_reason\":\"%s\",\"snapshot_utc\":\"\",\"inventory\":{\"fresh\":false,\"complete\":false,\"classification_complete\":false,\"process_generation_bound\":false,\"hook_capability_bound\":false,\"sha256\":\"\",\"total\":0,\"classified\":0,\"native\":0,\"legacy\":0,\"unknown\":0,\"unresponsive\":0},\"bindings\":{\"current_runtime_id\":\"\",\"candidate_runtime_id\":\"\",\"old_runtime_bound\":false,\"candidate_runtime_bound\":false,\"candidate_config_bound\":false,\"final_config_bound\":false,\"old_runtime_sha256\":\"\",\"candidate_runtime_sha256\":\"\",\"config_pair_sha256\":\"\"},\"canaries\":{\"mask\":0,\"required_mask\":15,\"four_provider_complete\":false},\"rollback_pair_tested\":false,\"native_entry_open\":false,\"bridge_free_candidate\":false,\"current_legacy_bridge\":false,\"authority\":{\"manifest_sha256\":\"%s\",\"semantics_sha256\":\"%s\",\"executable_sha256\":\"\",\"frame_sha256\":\"\",\"output_sha256\":\"\",\"output\":\"\"},\"members\":[]}"
    (json_escape reason) pinned_manifest_sha256 semantics_sha256

let live_json ~cwd =
  try
    let root = find_operational_root (Unix.realpath cwd) in
    fst (evaluate_live root)
  with error -> fail_closed_json (Printexc.to_string error)

let parse_arguments arguments =
  let rec loop cwd fixture = function
    | [] -> (cwd, fixture)
    | "--cwd" :: value :: tail -> loop (Some value) fixture tail
    | "--fixture" :: value :: tail -> loop cwd (Some value) tail
    | option :: _ -> failf "unknown-option:%s" option
  in
  loop None None arguments

let run arguments =
  try
    let cwd, fixture = parse_arguments arguments in
    let cwd = Option.value ~default:(Unix.getcwd ()) cwd |> Unix.realpath in
    let root = find_operational_root cwd in
    let json, code =
      match fixture with
      | Some path -> evaluate root (fixture_observation path)
      | None -> evaluate_live root
    in
    print_endline json;
    code
  with error ->
    print_endline (fail_closed_json (Printexc.to_string error));
    42

let run_cutover_admit arguments =
  try
    let cwd, fixture = parse_arguments arguments in
    let cwd = Option.value ~default:(Unix.getcwd ()) cwd |> Unix.realpath in
    let root = find_operational_root cwd in
    let json, code =
      match fixture with
      | Some path ->
          let snapshot, _authority_code, decision =
            evaluate_with_decision root
              (fixture_observation path |> cutover_observation)
          in
          (snapshot, if decision = "CUTOVER_READY" then 0 else 42)
      | None -> evaluate_cutover_live root (live_observation root)
    in
    print_endline json;
    code
  with error ->
    print_endline (fail_closed_json (Printexc.to_string error));
    42
