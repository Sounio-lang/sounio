open Unix

exception Error of string

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let semantics_9045_sha256 =
  "27c5fd758d161026c5c41d0cd0be0f1aa90bd4e3f4287da3c60fb748d1334882"

let provider_bit = function
  | "codex" -> 1
  | "claude" -> 2
  | "cursor" -> 4
  | "grok" -> 8
  | value -> failf "canary-provider-unsupported:%s" value

let provider_manifest_key provider =
  "loom_native_hook_cutover_" ^ provider ^ "_config_sha256"

let config_bindings =
  [ ".codex/hooks.json"; ".claude/settings.json"; ".cursor/hooks.json";
    ".grok/hooks/loom-native.json" ]

let parse_fields ?(allow_capability = false) label text =
  let fields = Hashtbl.create 32 in
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
               if Hashtbl.mem fields key then (
                 if not (allow_capability && key = "capability") then
                   failf "%s-duplicate-field:%s" label key)
               else Hashtbl.add fields key value);
  fields

let required label fields key =
  match Hashtbl.find_opt fields key with
  | Some value when value <> "" -> value
  | _ -> failf "%s-field-missing:%s" label key

let exact label fields key expected =
  let observed = required label fields key in
  if observed <> expected then
    failf "%s-field-invalid:%s:%s" label key observed

let decimal label value =
  if value = ""
     || not (String.for_all (function '0' .. '9' -> true | _ -> false) value)
  then failf "%s-not-decimal" label;
  try int_of_string value with _ -> failf "%s-out-of-range" label

let regular_file label path =
  let stat =
    try Unix.lstat path
    with Unix_error (ENOENT, _, _) -> failf "%s-missing:%s" label path
  in
  if stat.st_kind <> S_REG then failf "%s-not-regular:%s" label path;
  if stat.st_size > 8 * 1024 * 1024 then failf "%s-too-large:%s" label path

let read_file label path =
  regular_file label path;
  Loom_hook.read_file path

let sha256_file label path = Loom_hook.sha256 (read_file label path)

let config_bundle_sha256 root =
  config_bindings
  |> List.map (fun relative ->
         relative ^ "\000"
         ^ sha256_file "canary-provider-config" (Filename.concat root relative))
  |> String.concat "\000"
  |> fun value -> Loom_hook.sha256 ("loom-hook-config-bundle-v1\000" ^ value)

let utc_now timestamp =
  let tm = Unix.gmtime timestamp in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02dZ" (tm.tm_year + 1900)
    (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min tm.tm_sec

let candidate_directory common =
  let runtime_root = Filename.concat common "sounio-coord-runtime" in
  let link =
    match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_CANDIDATE" with
    | Some value when value <> "" && Loom_hook.test_mode () -> value
    | Some value when value <> "" ->
        failf "canary-candidate-override-requires-test-mode"
    | _ -> Filename.concat runtime_root "native-next"
  in
  let candidate =
    try Unix.realpath link
    with _ -> failf "canary-candidate-selector-missing:%s" link
  in
  if not (Loom_hook.test_mode ()) then (
    let versions = Unix.realpath (Filename.concat runtime_root "versions") in
    if not (Loom_hook.starts_with candidate (versions ^ "/")) then
      failf "canary-candidate-selector-outside-versions");
  candidate

type candidate = {
  path : string;
  runtime_id : string;
  manifest_sha256 : string;
  loom_runtime_sha256 : string;
  config_bundle_sha256 : string;
  provider_config_sha256 : string;
}

let candidate_binding root common provider =
  let path = candidate_directory common in
  let manifest_path = Filename.concat path "manifest" in
  let manifest_text = read_file "canary-candidate-manifest" manifest_path in
  let manifest = parse_fields ~allow_capability:true "canary-candidate-manifest" manifest_text in
  exact "canary-candidate-manifest" manifest
    "loom_native_hook_cutover_python_bridge_absent" "true";
  let loom_path = Filename.concat path "bin/sounio-loom-runtime" in
  let loom_runtime_sha256 = required "canary-candidate-manifest" manifest "loom_runtime_sha256" in
  if sha256_file "canary-candidate-loom" loom_path <> loom_runtime_sha256 then
    failf "canary-candidate-loom-hash-mismatch";
  let config_bundle_sha256 = config_bundle_sha256 root in
  let provider_config_sha256 =
    required "canary-candidate-manifest" manifest (provider_manifest_key provider)
  in
  { path; runtime_id = required "canary-candidate-manifest" manifest "runtime_id";
    manifest_sha256 = Loom_hook.sha256 manifest_text; loom_runtime_sha256;
    config_bundle_sha256; provider_config_sha256 }

let count_regular directory =
  if not (Sys.file_exists directory) then 0
  else
    Sys.readdir directory |> Array.fold_left (fun total name ->
        let path = Filename.concat directory name in
        try if (Unix.lstat path).st_kind = S_REG then total + 1 else total
        with _ -> total) 0

let active_state_count common =
  let state = Filename.concat common "sounio-coord-state" in
  [ "claims"; "process-presences"; "hook-capabilities"; "endpoints" ]
  |> List.fold_left
       (fun total name -> total + count_regular (Filename.concat state name)) 0

let watcher_count common =
  let directory =
    Filename.concat common "sounio-coord-state/hook-session-lifecycle"
  in
  if not (Sys.file_exists directory) then 0
  else
    Sys.readdir directory |> Array.fold_left (fun total name ->
        if Filename.check_suffix name ".watcher" then total + 1 else total) 0

let tab_fields label line =
  let fields = Hashtbl.create 32 in
  String.split_on_char '\t' line
  |> List.iter (fun item ->
         match String.index_opt item '=' with
         | None -> failf "%s-malformed-item" label
         | Some split ->
             let key = String.sub item 0 split in
             let value =
               String.sub item (split + 1) (String.length item - split - 1)
             in
             if key = "" || Hashtbl.mem fields key then
               failf "%s-duplicate-or-empty-item:%s" label key;
             Hashtbl.add fields key value);
  fields

let nonempty_lines text =
  String.split_on_char '\n' text |> List.filter (fun line -> line <> "")

let decision_counts provider candidate text =
  let rows = nonempty_lines text in
  if rows = [] then failf "canary-decision-log-empty";
  let starts = ref 0 and prompts = ref 0 and stops = ref 0 and ends = ref 0 in
  List.iteri
    (fun index line ->
      let label = Printf.sprintf "canary-decision-%d" index in
      let fields = tab_fields label line in
      exact label fields "schema" "loom-agent-hook-receipt-v1";
      exact label fields "decision" "ALLOW";
      exact label fields "provider" provider;
      exact label fields "semantics_sha256" semantics_9045_sha256;
      exact label fields "semantic_authority_language" "Sounio";
      exact label fields "semantic_authority_role" "SEMANTIC_AUTHORITY";
      exact label fields "toolchain_sha256" candidate.loom_runtime_sha256;
      exact label fields "provider_config_sha256" candidate.provider_config_sha256;
      exact label fields "result"
        "SOUNIO_NATIVE_HOOK_CUTOVER HOOK_EVENT_ADMIT semantic_authority=Sounio action=9045";
      match required label fields "event" with
      | "SessionStart" -> incr starts
      | "UserPromptSubmit" -> incr prompts
      | "Stop" -> incr stops
      | "SessionEnd" -> incr ends
      | _ -> ())
    rows;
  if !starts = 0 then failf "canary-session-start-absent";
  if provider = "codex" then (
    if !prompts = 0 then failf "canary-codex-prompt-absent";
    if !stops = 0 then failf "canary-codex-stop-absent")
  else if !ends = 0 then failf "canary-session-end-absent";
  (List.length rows, !starts, !prompts, !stops, !ends)

let lifecycle_counts provider text =
  let closed = ref 0 and process_exit = ref 0 and failed = ref 0 in
  nonempty_lines text
  |> List.iteri (fun index line ->
         let label = Printf.sprintf "canary-lifecycle-%d" index in
         let fields = tab_fields label line in
         exact label fields "schema" "loom-hook-session-lifecycle-v1";
         exact label fields "agent" provider;
         (match required label fields "action" with
         | "CLOSED" -> incr closed
         | "PROCESS_EXIT_CLOSED" -> incr process_exit
         | "CLOSE_FAILED" | "PROCESS_EXIT_CLOSE_FAILED" -> incr failed
         | _ -> ()));
  if !failed <> 0 then failf "canary-lifecycle-close-failed";
  if provider = "codex" then (
    if !process_exit = 0 then failf "canary-process-exit-close-absent")
  else if !closed = 0 then failf "canary-session-close-absent";
  (!closed, !process_exit)

let signing_paths state_directory =
  ( Filename.concat state_directory "guardian-ed25519-private.pem",
    Filename.concat state_directory "guardian-ed25519-public.pem",
    Filename.concat state_directory "guardian-ed25519-key.v1" )

let verified_key state_directory ~need_private =
  let private_key, public_key, key_manifest = signing_paths state_directory in
  let public_text = read_file "canary-guardian-public-key" public_key in
  let public_sha256 = Loom_hook.sha256 public_text in
  let key_id = Loom_epistemic.outcome_public_key_id public_text in
  let manifest =
    parse_fields "canary-guardian-key-manifest"
      (read_file "canary-guardian-key-manifest" key_manifest)
  in
  exact "canary-guardian-key-manifest" manifest "schema"
    "loom-native-hook-guardian-key-v1";
  exact "canary-guardian-key-manifest" manifest "algorithm" "ed25519";
  exact "canary-guardian-key-manifest" manifest "key_id" key_id;
  exact "canary-guardian-key-manifest" manifest "public_key_sha256" public_sha256;
  if need_private then
    exact "canary-guardian-key-manifest" manifest "private_key_sha256"
      (sha256_file "canary-guardian-private-key" private_key);
  (private_key, public_text, public_sha256, key_id)

let fsync_directory path =
  let descriptor = Unix.openfile path [ O_RDONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () -> Unix.fsync descriptor)

let ensure_directory path =
  if Sys.file_exists path then (
    if (Unix.lstat path).st_kind <> S_DIR then
      failf "canary-state-not-directory:%s" path)
  else Unix.mkdir path 0o700

let with_guardian_lock state_directory operation =
  ensure_directory state_directory;
  let path = Filename.concat state_directory "guardian.lock" in
  let descriptor = Unix.openfile path [ O_RDWR; O_CREAT ] 0o600 in
  Fun.protect
    ~finally:(fun () ->
      (try Unix.lockf descriptor F_ULOCK 0 with _ -> ());
      Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      operation ())

let atomic_write directory name text =
  ensure_directory directory;
  let target = Filename.concat directory name in
  let temporary =
    Filename.concat directory
      (Printf.sprintf ".%s.%d.%Ld.tmp" name (Unix.getpid ())
         (Int64.of_float (Unix.gettimeofday () *. 1_000_000.0)))
  in
  let descriptor = Unix.openfile temporary [ O_WRONLY; O_CREAT; O_EXCL ] 0o600 in
  Fun.protect
    ~finally:(fun () ->
      (try Unix.close descriptor with _ -> ());
      if Sys.file_exists temporary then (try Unix.unlink temporary with _ -> ()))
    (fun () ->
      Loom_hook.write_all descriptor text;
      Unix.fsync descriptor;
      Unix.close descriptor;
      Unix.rename temporary target;
      fsync_directory directory);
  target

let split_signed_receipt text =
  match List.rev (String.split_on_char '\n' text) with
  | "" :: signature_line :: reversed_payload
    when Loom_hook.starts_with signature_line "signature_base64=" ->
      let signature =
        String.sub signature_line 17 (String.length signature_line - 17)
      in
      if signature = "" then failf "canary-receipt-signature-empty";
      let payload = String.concat "\n" (List.rev reversed_payload) ^ "\n" in
      (payload, signature)
  | _ -> failf "canary-receipt-signature-missing-or-not-last"

let receipt_directory state_directory = Filename.concat state_directory "canaries"

let receipt_name provider = provider ^ ".canary.v1"

let validate_receipt ~state_directory ~candidate_id ~candidate_manifest_sha256
    ~candidate_loom_runtime_sha256 ~config_bundle_sha256 provider =
  let path = Filename.concat (receipt_directory state_directory) (receipt_name provider) in
  let payload, signature = split_signed_receipt (read_file "canary-receipt" path) in
  let fields = parse_fields "canary-receipt" payload in
  let _, public_text, public_sha256, key_id = verified_key state_directory ~need_private:false in
  exact "canary-receipt" fields "schema" "loom-native-hook-provider-canary-v1";
  exact "canary-receipt" fields "state" "PASS";
  exact "canary-receipt" fields "provider" provider;
  exact "canary-receipt" fields "provider_bit" (string_of_int (provider_bit provider));
  exact "canary-receipt" fields "semantic_authority" "Sounio";
  exact "canary-receipt" fields "action" "9046";
  exact "canary-receipt" fields "operational_language" "OCaml";
  exact "canary-receipt" fields "operational_role" "CANARY_EVIDENCE";
  exact "canary-receipt" fields "candidate_runtime_id" candidate_id;
  exact "canary-receipt" fields "candidate_runtime_manifest_sha256"
    candidate_manifest_sha256;
  exact "canary-receipt" fields "candidate_loom_runtime_sha256"
    candidate_loom_runtime_sha256;
  exact "canary-receipt" fields "config_bundle_sha256" config_bundle_sha256;
  exact "canary-receipt" fields "deny_count" "0";
  exact "canary-receipt" fields "residual_active_count" "0";
  exact "canary-receipt" fields "residual_watcher_count" "0";
  exact "canary-receipt" fields "closure_result" "PASS";
  exact "canary-receipt" fields "guardian_key_id" key_id;
  exact "canary-receipt" fields "guardian_public_key_sha256" public_sha256;
  if decimal "canary-allow-count" (required "canary-receipt" fields "allow_count") = 0 then
    failf "canary-receipt-allow-count-zero";
  if not (Loom_epistemic.outcome_ed25519_verify public_text payload signature) then
    failf "canary-receipt-signature-invalid:%s" provider;
  provider_bit provider

let verified_mask ~state_directory ~candidate_id ~candidate_manifest_sha256
    ~candidate_loom_runtime_sha256 ~config_bundle_sha256 =
  [ "codex"; "claude"; "cursor"; "grok" ]
  |> List.fold_left
       (fun mask provider ->
         let path = Filename.concat (receipt_directory state_directory) (receipt_name provider) in
         if not (Sys.file_exists path) then mask
         else
           mask
           lor validate_receipt ~state_directory ~candidate_id
                 ~candidate_manifest_sha256 ~candidate_loom_runtime_sha256
                 ~config_bundle_sha256 provider)
       0

let issue ~root ~state_directory ~provider ~canary_root ~output_path ~expected_output
    ~apply =
  let common = Loom_hook.git_common_dir root in
  let candidate = candidate_binding root common provider in
  let canary_root = Unix.realpath canary_root in
  let canary_common = Loom_hook.git_common_dir canary_root in
  let observed_candidate = candidate_directory canary_common in
  let observed_manifest = Filename.concat observed_candidate "manifest" in
  if sha256_file "canary-observed-runtime-manifest" observed_manifest
     <> candidate.manifest_sha256
  then failf "canary-observed-runtime-not-selected-candidate";
  let decisions_path =
    Filename.concat canary_common "sounio-loom-language-authority/agent-hook.tsv"
  in
  let lifecycle_path =
    Filename.concat canary_common
      "sounio-coord-state/hook-session-lifecycle/events.tsv"
  in
  let decisions = read_file "canary-decision-log" decisions_path in
  let lifecycle = read_file "canary-lifecycle-log" lifecycle_path in
  let output = read_file "canary-provider-output" output_path in
  if not (Loom_hook.contains output expected_output) then
    failf "canary-expected-output-absent";
  let allow_count, start_count, prompt_count, stop_count, end_count =
    decision_counts provider candidate decisions
  in
  let closed_count, process_exit_count = lifecycle_counts provider lifecycle in
  let residual_active_count = active_state_count canary_common in
  let residual_watcher_count = watcher_count canary_common in
  if residual_active_count <> 0 then failf "canary-active-state-residual:%d" residual_active_count;
  if residual_watcher_count <> 0 then
    failf "canary-watcher-residual:%d" residual_watcher_count;
  let private_key, public_text, public_sha256, key_id =
    verified_key state_directory ~need_private:true
  in
  let closure_result =
    if (provider = "codex" && process_exit_count > 0)
       || (provider <> "codex" && closed_count > 0)
    then "PASS"
    else "FAIL"
  in
  let payload =
    String.concat "\n"
      [ "schema=loom-native-hook-provider-canary-v1"; "state=PASS";
        "provider=" ^ provider; "provider_bit=" ^ string_of_int (provider_bit provider);
        "semantic_authority=Sounio"; "action=9046";
        "operational_language=OCaml"; "operational_role=CANARY_EVIDENCE";
        "candidate_runtime_id=" ^ candidate.runtime_id;
        "candidate_runtime_manifest_sha256=" ^ candidate.manifest_sha256;
        "candidate_loom_runtime_sha256=" ^ candidate.loom_runtime_sha256;
        "provider_config_sha256=" ^ candidate.provider_config_sha256;
        "config_bundle_sha256=" ^ candidate.config_bundle_sha256;
        "expected_output_sha256=" ^ Loom_hook.sha256 expected_output;
        "provider_output_sha256=" ^ Loom_hook.sha256 output;
        "decision_log_sha256=" ^ Loom_hook.sha256 decisions;
        "lifecycle_log_sha256=" ^ Loom_hook.sha256 lifecycle;
        "allow_count=" ^ string_of_int allow_count; "deny_count=0";
        "session_start_count=" ^ string_of_int start_count;
        "prompt_count=" ^ string_of_int prompt_count;
        "stop_count=" ^ string_of_int stop_count;
        "session_end_count=" ^ string_of_int end_count;
        "process_exit_closed_count=" ^ string_of_int process_exit_count;
        "residual_active_count=" ^ string_of_int residual_active_count;
        "residual_watcher_count=" ^ string_of_int residual_watcher_count;
        "closure_result=" ^ closure_result;
        "canary_root_sha256=" ^ Loom_hook.sha256 canary_root;
        "recorded_utc=" ^ utc_now (Unix.time ());
        "guardian_key_id=" ^ key_id;
        "guardian_public_key_sha256=" ^ public_sha256; "" ]
  in
  let signature = Loom_epistemic.outcome_ed25519_sign private_key payload in
  if not (Loom_epistemic.outcome_ed25519_verify public_text payload signature) then
    failf "canary-signature-self-verification-refused";
  if not apply then
    Printf.sprintf
      "{\"schema\":\"loom-native-hook-provider-canary-v1\",\"state\":\"PLAN_READY\",\"applied\":false,\"provider\":\"%s\",\"candidate_runtime_id\":\"%s\",\"payload_sha256\":\"%s\"}"
      provider candidate.runtime_id (Loom_hook.sha256 payload)
  else
    let directory = receipt_directory state_directory in
    let path = atomic_write directory (receipt_name provider)
        (payload ^ "signature_base64=" ^ signature ^ "\n")
    in
    let bit =
      validate_receipt ~state_directory ~candidate_id:candidate.runtime_id
        ~candidate_manifest_sha256:candidate.manifest_sha256
        ~candidate_loom_runtime_sha256:candidate.loom_runtime_sha256
        ~config_bundle_sha256:candidate.config_bundle_sha256 provider
    in
    Printf.sprintf
      "{\"schema\":\"loom-native-hook-provider-canary-v1\",\"state\":\"RECORDED\",\"applied\":true,\"provider\":\"%s\",\"provider_bit\":%d,\"candidate_runtime_id\":\"%s\",\"payload_sha256\":\"%s\",\"receipt_sha256\":\"%s\",\"receipt_path\":\"%s\",\"same_uid_peer_isolation\":false}"
      provider bit candidate.runtime_id (Loom_hook.sha256 payload)
      (sha256_file "canary-written-receipt" path) (Loom_hook.json_escape path)

let verify_current ~root ~state_directory =
  let common = Loom_hook.git_common_dir root in
  let candidate_path = candidate_directory common in
  let manifest_path = Filename.concat candidate_path "manifest" in
  let manifest_text = read_file "canary-candidate-manifest" manifest_path in
  let manifest =
    parse_fields ~allow_capability:true "canary-candidate-manifest" manifest_text
  in
  let candidate_id = required "canary-candidate-manifest" manifest "runtime_id" in
  let candidate_manifest_sha256 = Loom_hook.sha256 manifest_text in
  let candidate_loom_runtime_sha256 =
    required "canary-candidate-manifest" manifest "loom_runtime_sha256"
  in
  let config_bundle_sha256 = config_bundle_sha256 root in
  let mask =
    verified_mask ~state_directory ~candidate_id ~candidate_manifest_sha256
      ~candidate_loom_runtime_sha256 ~config_bundle_sha256
  in
  Printf.sprintf
    "{\"schema\":\"loom-native-hook-provider-canary-set-v1\",\"state\":\"VERIFIED\",\"candidate_runtime_id\":\"%s\",\"mask\":%d,\"required_mask\":15,\"four_provider_complete\":%s,\"same_uid_peer_isolation\":false}"
    (Loom_hook.json_escape candidate_id) mask (string_of_bool (mask = 15))

type arguments = {
  cwd : string;
  provider : string;
  canary_root : string;
  output_path : string;
  expected_output : string;
  apply : bool;
}

let parse_arguments values =
  let rec loop cwd provider canary_root output_path expected_output apply = function
    | [] ->
        { cwd = Option.value ~default:(Unix.getcwd ()) cwd;
          provider = Option.value ~default:"" provider;
          canary_root = Option.value ~default:"" canary_root;
          output_path = Option.value ~default:"" output_path;
          expected_output = Option.value ~default:"" expected_output; apply }
    | "--cwd" :: value :: tail -> loop (Some value) provider canary_root output_path expected_output apply tail
    | "--provider" :: value :: tail -> loop cwd (Some value) canary_root output_path expected_output apply tail
    | "--canary-root" :: value :: tail -> loop cwd provider (Some value) output_path expected_output apply tail
    | "--output" :: value :: tail -> loop cwd provider canary_root (Some value) expected_output apply tail
    | "--expect" :: value :: tail -> loop cwd provider canary_root output_path (Some value) apply tail
    | "--apply" :: tail -> loop cwd provider canary_root output_path expected_output true tail
    | option :: _ -> failf "canary-unknown-option:%s" option
  in
  let parsed = loop None None None None None false values in
  ignore (provider_bit parsed.provider);
  if parsed.canary_root = "" then failf "canary-root-required";
  if parsed.output_path = "" then failf "canary-output-required";
  if parsed.expected_output = "" then failf "canary-expected-output-required";
  parsed

let run values =
  try
    let verify, values =
      match values with "--verify" :: tail -> (true, tail) | _ -> (false, values)
    in
    let parsed =
      if verify then
        let cwd = function
          | [] -> Unix.getcwd ()
          | [ "--cwd"; value ] -> value
          | option :: _ -> failf "canary-verify-unknown-option:%s" option
        in
        { cwd = cwd values; provider = "codex"; canary_root = ".";
          output_path = "."; expected_output = "verify"; apply = false }
      else parse_arguments values
    in
    let root = Loom_hook.git_root parsed.cwd |> Unix.realpath in
    let common = Loom_hook.git_common_dir root in
    let state_directory =
      match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR" with
      | Some value when value <> "" && Loom_hook.test_mode () -> value
      | Some value when value <> "" -> failf "canary-state-override-requires-test-mode"
      | _ -> Filename.concat common "sounio-coord-runtime/native-hook-drain"
    in
    let operation () =
      if verify then verify_current ~root ~state_directory
      else
        issue ~root ~state_directory ~provider:parsed.provider
          ~canary_root:parsed.canary_root ~output_path:parsed.output_path
          ~expected_output:parsed.expected_output ~apply:parsed.apply
    in
    print_endline
      (if verify || parsed.apply then with_guardian_lock state_directory operation
       else operation ());
    0
  with error ->
    Printf.printf
      "{\"schema\":\"loom-native-hook-provider-canary-v1\",\"state\":\"FAIL_CLOSED\",\"applied\":false,\"reason\":\"%s\"}\n"
      (Loom_hook.json_escape (Printexc.to_string error));
    42
