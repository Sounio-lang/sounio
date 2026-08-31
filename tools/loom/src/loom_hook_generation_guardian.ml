open Unix

exception Error of string

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let ensure_directory path =
  if Sys.file_exists path then (
    let stat = Unix.lstat path in
    if stat.st_kind <> S_DIR then failf "guardian-path-not-directory:%s" path)
  else Unix.mkdir path 0o700

let fsync_directory path =
  let descriptor = Unix.openfile path [ O_RDONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () -> Unix.fsync descriptor)

let write_all descriptor text =
  let rec loop offset =
    if offset < String.length text then
      let written = Unix.write_substring descriptor text offset (String.length text - offset) in
      if written = 0 then failf "guardian-atomic-write-short" else loop (offset + written)
  in
  loop 0

let atomic_write directory name text =
  ensure_directory directory;
  let target = Filename.concat directory name in
  let temporary =
    Filename.concat directory
      (Printf.sprintf ".%s.%d.%Ld.tmp" name (Unix.getpid ())
         (Int64.of_float (Unix.gettimeofday () *. 1_000_000.0)))
  in
  let descriptor =
    Unix.openfile temporary [ O_WRONLY; O_CREAT; O_EXCL ] 0o600
  in
  Fun.protect
    ~finally:(fun () ->
      (try Unix.close descriptor with _ -> ());
      if Sys.file_exists temporary then (try Unix.unlink temporary with _ -> ()))
    (fun () ->
      write_all descriptor text;
      Unix.fsync descriptor;
      Unix.close descriptor;
      Unix.rename temporary target;
      fsync_directory directory);
  target

let atomic_symlink directory name target =
  let link = Filename.concat directory name in
  let temporary =
    Filename.concat directory
      (Printf.sprintf ".%s.%d.%Ld.link" name (Unix.getpid ())
         (Int64.of_float (Unix.gettimeofday () *. 1_000_000.0)))
  in
  Fun.protect
    ~finally:(fun () -> if Sys.file_exists temporary then (try Unix.unlink temporary with _ -> ()))
    (fun () ->
      Unix.symlink target temporary;
      Unix.rename temporary link;
      fsync_directory directory);
  link

let verify_link label link expected =
  let observed = Unix.realpath link in
  if observed <> Unix.realpath expected then
    failf "%s-link-drift:expected=%s:observed=%s" label expected observed

let remove_probe directory =
  Array.iter
    (fun name ->
      let path = Filename.concat directory name in
      try Unix.unlink path with _ -> ())
    (Sys.readdir directory);
  Unix.rmdir directory

let with_lock state_directory operation =
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

let state_directory common = Loom_hook_generation_drain.marker_directory common

let runtime_paths common =
  let root = Filename.concat common "sounio-coord-runtime" in
  let current = Unix.realpath (Filename.concat root "current") in
  let candidate = Loom_hook_generation_drain.find_candidate root current in
  if candidate = "" then failf "guardian-candidate-runtime-missing";
  (current, candidate)

let require_prepare_boundary observation =
  let open Loom_hook_generation_drain in
  if not observation.old_runtime_bound then failf "guardian-old-runtime-unbound";
  if not observation.candidate_runtime_bound then failf "guardian-candidate-runtime-unbound";
  if not observation.candidate_config_bound then failf "guardian-candidate-config-unbound";
  if not observation.native_entry_open then failf "guardian-native-entry-closed";
  if not observation.bridge_free_candidate then failf "guardian-candidate-not-bridge-free";
  if not observation.current_legacy_bridge then failf "guardian-current-runtime-not-legacy";
  if observation.current_runtime_id = observation.candidate_runtime_id then
    failf "guardian-generations-not-distinct"

let rollback_probe state_directory current candidate =
  let probes = Filename.concat state_directory "probes" in
  ensure_directory probes;
  let directory =
    Filename.concat probes
      (Printf.sprintf "probe-%d-%Ld" (Unix.getpid ())
         (Int64.of_float (Unix.gettimeofday () *. 1_000_000.0)))
  in
  Unix.mkdir directory 0o700;
  Fun.protect
    ~finally:(fun () -> try remove_probe directory with _ -> ())
    (fun () ->
      let link = atomic_symlink directory "current" current in
      verify_link "rollback-probe-old-initial" link current;
      ignore (atomic_symlink directory "current" candidate);
      verify_link "rollback-probe-candidate" link candidate;
      ignore (atomic_symlink directory "current" current);
      verify_link "rollback-probe-old-restored" link current;
      Loom_hook_generation_drain.sha256
        ("loom-native-hook-rollback-probe-v1\000" ^ Unix.realpath current ^ "\000"
       ^ Unix.realpath candidate ^ "\000" ^ Unix.realpath current))

let final_marker observation created_utc =
  let open Loom_hook_generation_drain in
  Printf.sprintf
    "schema=loom-native-hook-final-config-v1\nstate=FINAL_CONFIG_BOUND\nruntime_id=%s\nruntime_manifest_sha256=%s\nconfig_bundle_sha256=%s\nsemantic_authority=Sounio\naction=9046\ncreated_utc=%s\n"
    observation.candidate_runtime_id observation.candidate_runtime_sha256
    observation.config_pair_sha256 created_utc

let rollback_marker observation config_sha256 probe_sha256 created_utc =
  let open Loom_hook_generation_drain in
  Printf.sprintf
    "schema=loom-native-hook-rollback-pair-v1\nstate=ROLLBACK_PAIR_TESTED\nforward_result=PASS\nrollback_result=PASS\nold_runtime_id=%s\nold_runtime_manifest_sha256=%s\ncandidate_runtime_id=%s\ncandidate_runtime_manifest_sha256=%s\nconfig_bundle_sha256=%s\nprobe_sha256=%s\nsemantic_authority=Sounio\naction=9046\ncreated_utc=%s\n"
    observation.current_runtime_id observation.old_runtime_sha256
    observation.candidate_runtime_id observation.candidate_runtime_sha256 config_sha256
    probe_sha256 created_utc

let prepare ~cwd ~apply =
  let root =
    Loom_hook_generation_drain.find_source_root (Unix.realpath cwd)
  in
  let common = Loom_hook_generation_drain.git_common_dir root in
  let state = state_directory common in
  let observation = Loom_hook_generation_drain.live_observation root in
  require_prepare_boundary observation;
  let current, candidate = runtime_paths common in
  let config_sha256 = Loom_hook_generation_drain.config_bundle_sha256 root in
  if config_sha256 = String.make 64 '0' then failf "guardian-config-bundle-unbound";
  if not apply then
    Printf.sprintf
      "{\"schema\":\"loom-native-hook-generation-guardian-v1\",\"action\":9046,\"semantic_authority\":\"Sounio\",\"operational_realization\":\"OCaml\",\"state\":\"PLAN_READY\",\"applied\":false,\"current_runtime_id\":\"%s\",\"candidate_runtime_id\":\"%s\",\"config_bundle_sha256\":\"%s\"}"
      (Loom_hook_generation_drain.json_escape observation.current_runtime_id)
      (Loom_hook_generation_drain.json_escape observation.candidate_runtime_id)
      config_sha256
  else
    with_lock state (fun () ->
        let probe_sha256 = rollback_probe state current candidate in
        let created_utc =
          Loom_hook_generation_drain.utc_now (Unix.time ())
        in
        let rollback = rollback_marker observation config_sha256 probe_sha256 created_utc in
        let final = final_marker { observation with config_pair_sha256 = config_sha256 } created_utc in
        let rollback_path = atomic_write state "rollback-pair-tested.v1" rollback in
        let final_path = atomic_write state "final-config.v1" final in
        Printf.sprintf
          "{\"schema\":\"loom-native-hook-generation-guardian-v1\",\"action\":9046,\"semantic_authority\":\"Sounio\",\"operational_realization\":\"OCaml\",\"state\":\"PREPARED\",\"applied\":true,\"live_runtime_unchanged\":true,\"current_runtime_id\":\"%s\",\"candidate_runtime_id\":\"%s\",\"config_bundle_sha256\":\"%s\",\"rollback_probe_sha256\":\"%s\",\"rollback_marker_sha256\":\"%s\",\"final_marker_sha256\":\"%s\",\"rollback_marker_path\":\"%s\",\"final_marker_path\":\"%s\"}"
          (Loom_hook_generation_drain.json_escape observation.current_runtime_id)
          (Loom_hook_generation_drain.json_escape observation.candidate_runtime_id)
          config_sha256 probe_sha256
          (Loom_hook_generation_drain.sha256 rollback)
          (Loom_hook_generation_drain.sha256 final)
          (Loom_hook_generation_drain.json_escape rollback_path)
          (Loom_hook_generation_drain.json_escape final_path))

let parse_arguments arguments =
  let rec loop cwd apply = function
    | [] -> (Option.value ~default:(Unix.getcwd ()) cwd, apply)
    | "--cwd" :: value :: tail -> loop (Some value) apply tail
    | "--apply" :: tail -> loop cwd true tail
    | option :: _ -> failf "guardian-unknown-option:%s" option
  in
  loop None false arguments

let run arguments =
  try
    let cwd, apply = parse_arguments arguments in
    print_endline (prepare ~cwd ~apply);
    0
  with error ->
    Printf.printf
      "{\"schema\":\"loom-native-hook-generation-guardian-v1\",\"action\":9046,\"semantic_authority\":\"Sounio\",\"operational_realization\":\"OCaml\",\"state\":\"FAIL_CLOSED\",\"applied\":false,\"reason\":\"%s\"}\n"
      (Loom_hook_generation_drain.json_escape (Printexc.to_string error));
    42
