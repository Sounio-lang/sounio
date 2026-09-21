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

let signing_paths state_directory =
  ( Filename.concat state_directory "guardian-ed25519-private.pem",
    Filename.concat state_directory "guardian-ed25519-public.pem",
    Filename.concat state_directory "guardian-ed25519-key.v1" )

let validate_signing_pair private_key public_key =
  let probe = "loom-native-hook-generation-guardian-signing-probe-v1\n" in
  let public_text =
    Loom_hook_generation_drain.read_file "guardian-public-key" public_key
  in
  let signature = Loom_epistemic.outcome_ed25519_sign private_key probe in
  if not (Loom_epistemic.outcome_ed25519_verify public_text probe signature) then
    failf "guardian-signing-pair-invalid";
  ( Loom_hook_generation_drain.sha256_file "guardian-private-key" private_key,
    Loom_hook_generation_drain.sha256 public_text,
    Loom_epistemic.outcome_public_key_id public_text )

let generate_signing_pair state_directory =
  let private_key, public_key, key_manifest = signing_paths state_directory in
  let temporary_private =
    Filename.concat state_directory
      (Printf.sprintf ".guardian-private.%d.pem" (Unix.getpid ()))
  in
  let temporary_public =
    Filename.concat state_directory
      (Printf.sprintf ".guardian-public.%d.pem" (Unix.getpid ()))
  in
  let openssl = "/usr/bin/openssl" in
  Fun.protect
    ~finally:(fun () ->
      List.iter (fun path -> if Sys.file_exists path then (try Unix.unlink path with _ -> ()))
        [ temporary_private; temporary_public ])
    (fun () ->
      if not (Loom_hook_generation_drain.executable openssl) then
        failf "guardian-openssl-unavailable";
      let generated =
        Loom_hook.run_process ~timeout_seconds:10.0 ~cwd:state_directory openssl
          [ "genpkey"; "-algorithm"; "ED25519"; "-out"; temporary_private ]
      in
      if generated.code <> 0 then
        failf "guardian-private-key-generation-refused:%s" generated.output;
      Unix.chmod temporary_private 0o600;
      let derived =
        Loom_hook.run_process ~timeout_seconds:10.0 ~cwd:state_directory openssl
          [ "pkey"; "-in"; temporary_private; "-pubout"; "-out";
            temporary_public ]
      in
      if derived.code <> 0 then
        failf "guardian-public-key-derivation-refused:%s" derived.output;
      let private_text =
        Loom_hook_generation_drain.read_file "guardian-private-key" temporary_private
      in
      let public_text =
        Loom_hook_generation_drain.read_file "guardian-public-key" temporary_public
      in
      ignore (atomic_write state_directory (Filename.basename private_key) private_text);
      ignore (atomic_write state_directory (Filename.basename public_key) public_text);
      let private_sha256, public_sha256, key_id =
        validate_signing_pair private_key public_key
      in
      let created_utc = Loom_hook_generation_drain.utc_now (Unix.time ()) in
      let manifest =
        Printf.sprintf
          "schema=loom-native-hook-guardian-key-v1\nalgorithm=ed25519\nkey_id=%s\nprivate_key_sha256=%s\npublic_key_sha256=%s\ncreated_utc=%s\n"
          key_id private_sha256 public_sha256 created_utc
      in
      ignore (atomic_write state_directory (Filename.basename key_manifest) manifest);
      (public_sha256, key_id))

let ensure_signing_pair state_directory =
  let private_key, public_key, key_manifest = signing_paths state_directory in
  match
    (Sys.file_exists private_key, Sys.file_exists public_key,
     Sys.file_exists key_manifest)
  with
  | false, false, false -> generate_signing_pair state_directory
  | true, true, true ->
      let private_sha256, public_sha256, key_id =
        validate_signing_pair private_key public_key
      in
      let values =
        Loom_hook_generation_drain.parse_fields "guardian-key-manifest"
          (Loom_hook_generation_drain.read_file "guardian-key-manifest" key_manifest)
      in
      if Loom_hook_generation_drain.field values "schema"
           <> "loom-native-hook-guardian-key-v1"
         || Loom_hook_generation_drain.field values "algorithm" <> "ed25519"
         || Loom_hook_generation_drain.field values "key_id" <> key_id
         || Loom_hook_generation_drain.field values "private_key_sha256"
            <> private_sha256
         || Loom_hook_generation_drain.field values "public_key_sha256"
            <> public_sha256
      then failf "guardian-key-manifest-drift";
      (public_sha256, key_id)
  | _ -> failf "guardian-signing-pair-incomplete"

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

let final_marker observation guardian_public_key_sha256 created_utc =
  let open Loom_hook_generation_drain in
  Printf.sprintf
    "schema=loom-native-hook-final-config-v1\nstate=FINAL_CONFIG_BOUND\nruntime_id=%s\nruntime_manifest_sha256=%s\nconfig_bundle_sha256=%s\nguardian_public_key_sha256=%s\nsemantic_authority=Sounio\naction=9046\ncreated_utc=%s\n"
    observation.candidate_runtime_id observation.candidate_runtime_sha256
    observation.config_pair_sha256 guardian_public_key_sha256 created_utc

let rollback_marker observation config_sha256 probe_sha256 created_utc =
  let open Loom_hook_generation_drain in
  Printf.sprintf
    "schema=loom-native-hook-rollback-pair-v1\nstate=ROLLBACK_PAIR_TESTED\nforward_result=PASS\nrollback_result=PASS\nold_runtime_id=%s\nold_runtime_manifest_sha256=%s\ncandidate_runtime_id=%s\ncandidate_runtime_manifest_sha256=%s\nconfig_bundle_sha256=%s\nprobe_sha256=%s\nsemantic_authority=Sounio\naction=9046\ncreated_utc=%s\n"
    observation.current_runtime_id observation.old_runtime_sha256
    observation.candidate_runtime_id observation.candidate_runtime_sha256 config_sha256
    probe_sha256 created_utc

let provider_receipts state_directory =
  [ ("codex", 1); ("claude", 2); ("cursor", 4); ("grok", 8) ]
  |> List.filter_map (fun (provider, bit) ->
         let relative = "canaries/" ^ provider ^ ".canary.v1" in
         let path = Filename.concat state_directory relative in
         if Sys.file_exists path then Some (provider, bit, relative, path) else None)

let proof_residual state_directory =
  Sys.file_exists (Filename.concat state_directory "rollback-pair-tested.v1")
  || provider_receipts state_directory <> []

let require_field label fields key expected =
  let observed = Loom_hook_generation_drain.field fields key in
  if observed <> expected then
    failf "%s-field-invalid:%s:%s" label key observed

let archive_relative archive relative = Filename.concat archive relative

let remove_tree path =
  let rec remove path =
    if Sys.file_exists path then
      match (Unix.lstat path).st_kind with
      | S_DIR ->
          Sys.readdir path
          |> Array.iter (fun name -> remove (Filename.concat path name));
          Unix.rmdir path
      | _ -> Unix.unlink path
  in
  remove path

let archive_proof_files state_directory receipts =
  [ ("final_config_sha256", "final-config.v1",
     Filename.concat state_directory "final-config.v1");
    ("rollback_pair_sha256", "rollback-pair-tested.v1",
     Filename.concat state_directory "rollback-pair-tested.v1") ]
  @ List.map
      (fun (provider, _bit, relative, path) ->
        (provider ^ "_canary_sha256", relative, path))
      receipts

let verify_archive ~state_directory ~archive ~old_id ~old_manifest_sha256
    ~successor_id ~current_id ~current_manifest_sha256 =
  let receipt_path = Filename.concat archive "generation-archive.v1" in
  let payload, signature =
    Loom_hook_generation_canary.split_signed_receipt
      (Loom_hook_generation_drain.read_file "guardian-generation-archive" receipt_path)
  in
  let fields =
    Loom_hook_generation_drain.parse_fields "guardian-generation-archive" payload
  in
  let _, public_key, _ = signing_paths state_directory in
  let public_text =
    Loom_hook_generation_drain.read_file "guardian-public-key" public_key
  in
  let public_sha256 = Loom_hook_generation_drain.sha256 public_text in
  let key_id = Loom_epistemic.outcome_public_key_id public_text in
  List.iter
    (fun (key, expected) ->
      require_field "guardian-generation-archive" fields key expected)
    [ ("schema", "loom-native-hook-generation-archive-v1");
      ("state", "ARCHIVED");
      ("prior_candidate_runtime_id", old_id);
      ("prior_candidate_runtime_manifest_sha256", old_manifest_sha256);
      ("successor_candidate_runtime_id", successor_id);
      ("old_runtime_id", current_id);
      ("old_runtime_manifest_sha256", current_manifest_sha256);
      ("guardian_key_id", key_id);
      ("guardian_public_key_sha256", public_sha256);
      ("semantic_authority", "Sounio");
      ("action", "9046");
      ("operational_language", "OCaml");
      ("operational_role", "GENERATION_EVIDENCE_ARCHIVE") ];
  if not (Loom_epistemic.outcome_ed25519_verify public_text payload signature) then
    failf "guardian-generation-archive-signature-invalid";
  let mask =
    Loom_hook_generation_canary.decimal "guardian-generation-archive-mask"
      (Loom_hook_generation_drain.field fields "canary_mask")
  in
  let receipts =
    [ ("codex", 1); ("claude", 2); ("cursor", 4); ("grok", 8) ]
    |> List.filter_map (fun (provider, bit) ->
           if mask land bit = 0 then None
           else
             let relative = "canaries/" ^ provider ^ ".canary.v1" in
             Some (provider, bit, relative,
               Filename.concat state_directory relative))
  in
  let proof_files = archive_proof_files state_directory receipts in
  List.iter
    (fun (key, relative, active) ->
      let archived = archive_relative archive relative in
      let archived_sha256 =
        Loom_hook_generation_drain.sha256_file "guardian-archived-proof" archived
      in
      require_field "guardian-generation-archive" fields key archived_sha256;
      if Sys.file_exists active
         && Loom_hook_generation_drain.sha256_file "guardian-active-proof" active
            <> archived_sha256
      then failf "guardian-active-proof-drift:%s" relative)
    proof_files;
  let expected_mask =
    provider_receipts state_directory
    |> List.fold_left (fun total (_provider, bit, _relative, _path) -> total lor bit) 0
  in
  if expected_mask land lnot mask <> 0 then failf "guardian-active-canary-not-archived";
  (mask, receipts)

let create_archive ~state_directory ~archive ~old_id ~old_manifest_sha256
    ~old_loom_sha256 ~old_config_sha256 ~successor_id ~current_id
    ~current_manifest_sha256 ~mask ~receipts ~guardian_public_key_sha256
    ~guardian_key_id =
  let archive_root = Filename.dirname archive in
  ensure_directory archive_root;
  let temporary =
    Filename.concat archive_root
      (Printf.sprintf ".generation.%d.%Ld.tmp" (Unix.getpid ())
         (Int64.of_float (Unix.gettimeofday () *. 1_000_000.0)))
  in
  Unix.mkdir temporary 0o700;
  Fun.protect
    ~finally:(fun () -> if Sys.file_exists temporary then (try remove_tree temporary with _ -> ()))
    (fun () ->
      let temporary_canaries = Filename.concat temporary "canaries" in
      if receipts <> [] then Unix.mkdir temporary_canaries 0o700;
      let proof_files = archive_proof_files state_directory receipts in
      let hashes =
        List.map
          (fun (key, relative, source) ->
            let text =
              Loom_hook_generation_drain.read_file "guardian-active-proof" source
            in
            let target_directory =
              if Filename.dirname relative = "." then temporary else temporary_canaries
            in
            ignore (atomic_write target_directory (Filename.basename relative) text);
            (key, Loom_hook_generation_drain.sha256 text))
          proof_files
      in
      let created_utc = Loom_hook_generation_drain.utc_now (Unix.time ()) in
      let payload =
        String.concat "\n"
          ([ "schema=loom-native-hook-generation-archive-v1"; "state=ARCHIVED";
             "prior_candidate_runtime_id=" ^ old_id;
             "prior_candidate_runtime_manifest_sha256=" ^ old_manifest_sha256;
             "prior_candidate_loom_runtime_sha256=" ^ old_loom_sha256;
             "prior_config_bundle_sha256=" ^ old_config_sha256;
             "successor_candidate_runtime_id=" ^ successor_id;
             "old_runtime_id=" ^ current_id;
             "old_runtime_manifest_sha256=" ^ current_manifest_sha256;
             "canary_mask=" ^ string_of_int mask ]
          @ List.map (fun (key, value) -> key ^ "=" ^ value) hashes
          @ [ "guardian_key_id=" ^ guardian_key_id;
              "guardian_public_key_sha256=" ^ guardian_public_key_sha256;
              "semantic_authority=Sounio"; "action=9046";
              "operational_language=OCaml";
              "operational_role=GENERATION_EVIDENCE_ARCHIVE";
              "created_utc=" ^ created_utc; "" ])
      in
      let private_key, public_key, _ = signing_paths state_directory in
      let public_text =
        Loom_hook_generation_drain.read_file "guardian-public-key" public_key
      in
      let signature = Loom_epistemic.outcome_ed25519_sign private_key payload in
      if not (Loom_epistemic.outcome_ed25519_verify public_text payload signature) then
        failf "guardian-generation-archive-signature-self-check-refused";
      ignore
        (atomic_write temporary "generation-archive.v1"
           (payload ^ "signature_base64=" ^ signature ^ "\n"));
      if receipts <> [] then fsync_directory temporary_canaries;
      fsync_directory temporary;
      Unix.rename temporary archive;
      fsync_directory archive_root)

let clear_archived_generation state_directory receipts =
  List.iter
    (fun (_provider, _bit, _relative, path) ->
      if Sys.file_exists path then Unix.unlink path)
    receipts;
  let canaries = Filename.concat state_directory "canaries" in
  if Sys.file_exists canaries then fsync_directory canaries;
  let rollback = Filename.concat state_directory "rollback-pair-tested.v1" in
  if Sys.file_exists rollback then Unix.unlink rollback;
  fsync_directory state_directory;
  let final = Filename.concat state_directory "final-config.v1" in
  if Sys.file_exists final then Unix.unlink final;
  fsync_directory state_directory

let archive_prior_generation ~state_directory ~current_id ~current_manifest_sha256
    ~successor_id ~guardian_public_key_sha256 ~guardian_key_id =
  let final_path = Filename.concat state_directory "final-config.v1" in
  if not (Sys.file_exists final_path) then (
    if proof_residual state_directory then failf "guardian-prior-proof-set-incomplete";
    None)
  else
    let final_text =
      Loom_hook_generation_drain.read_file "guardian-prior-final-config" final_path
    in
    let final_fields =
      Loom_hook_generation_drain.parse_fields "guardian-prior-final-config" final_text
    in
    let old_id = Loom_hook_generation_drain.field final_fields "runtime_id" in
    if old_id = successor_id then None
    else (
      List.iter
        (fun (key, expected) ->
          require_field "guardian-prior-final-config" final_fields key expected)
        [ ("schema", "loom-native-hook-final-config-v1");
          ("state", "FINAL_CONFIG_BOUND");
          ("guardian_public_key_sha256", guardian_public_key_sha256);
          ("semantic_authority", "Sounio"); ("action", "9046") ];
      let old_manifest_sha256 =
        Loom_hook_generation_drain.field final_fields "runtime_manifest_sha256"
      in
      let old_config_sha256 =
        Loom_hook_generation_drain.field final_fields "config_bundle_sha256"
      in
      let final_sha256 = Loom_hook_generation_drain.sha256 final_text in
      let archive_root = Filename.concat state_directory "archives" in
      let archive =
        Filename.concat archive_root
          (Loom_hook_generation_drain.slug old_id ^ "-"
          ^ String.sub final_sha256 0 12)
      in
      let receipts =
        if Sys.file_exists archive then
          snd
            (verify_archive ~state_directory ~archive ~old_id ~old_manifest_sha256
               ~successor_id ~current_id ~current_manifest_sha256)
        else (
          let rollback_path =
            Filename.concat state_directory "rollback-pair-tested.v1"
          in
          let rollback_text =
            Loom_hook_generation_drain.read_file "guardian-prior-rollback" rollback_path
          in
          if not
               (Loom_hook_generation_drain.valid_rollback_marker rollback_text current_id
                  current_manifest_sha256 old_id old_manifest_sha256 old_config_sha256)
          then failf "guardian-prior-rollback-invalid";
          let active_receipts = provider_receipts state_directory in
          let old_loom_sha256 =
            match active_receipts with
            | [] -> String.make 64 '0'
            | (_provider, _bit, _relative, path) :: _ ->
                let payload, _ =
                  Loom_hook_generation_canary.split_signed_receipt
                    (Loom_hook_generation_drain.read_file "guardian-prior-canary" path)
                in
                let fields =
                  Loom_hook_generation_canary.parse_fields "guardian-prior-canary"
                    payload
                in
                Loom_hook_generation_canary.required "guardian-prior-canary" fields
                  "candidate_loom_runtime_sha256"
          in
          let mask =
            Loom_hook_generation_canary.verified_mask ~state_directory
              ~candidate_id:old_id ~candidate_manifest_sha256:old_manifest_sha256
              ~candidate_loom_runtime_sha256:old_loom_sha256
              ~config_bundle_sha256:old_config_sha256
          in
          create_archive ~state_directory ~archive ~old_id ~old_manifest_sha256
            ~old_loom_sha256 ~old_config_sha256 ~successor_id ~current_id
            ~current_manifest_sha256 ~mask ~receipts:active_receipts
            ~guardian_public_key_sha256 ~guardian_key_id;
          active_receipts)
      in
      clear_archived_generation state_directory receipts;
      Some archive)

let prepare ~cwd ~apply =
  let root =
    Loom_hook_generation_drain.find_source_root (Unix.realpath cwd)
  in
  let common = Loom_hook_generation_drain.git_common_dir root in
  let state = state_directory common in
  let current, candidate = runtime_paths common in
  let candidate_id, _ = Loom_hook_generation_drain.runtime_identity candidate in
  let current_id, current_manifest_sha256 =
    Loom_hook_generation_drain.runtime_identity current
  in
  if not apply then
    let observation =
      Loom_hook_generation_drain.live_observation ~verify_canaries:false root
    in
    require_prepare_boundary observation;
    let config_sha256 = Loom_hook_generation_drain.config_bundle_sha256 root in
    if config_sha256 = String.make 64 '0' then failf "guardian-config-bundle-unbound";
    let rotation_required =
      let final = Filename.concat state "final-config.v1" in
      Sys.file_exists final
      && Loom_hook_generation_drain.field
           (Loom_hook_generation_drain.parse_fields "guardian-prior-final-config"
              (Loom_hook_generation_drain.read_file "guardian-prior-final-config" final))
           "runtime_id"
         <> candidate_id
    in
    Printf.sprintf
      "{\"schema\":\"loom-native-hook-generation-guardian-v1\",\"action\":9046,\"semantic_authority\":\"Sounio\",\"operational_realization\":\"OCaml\",\"state\":\"PLAN_READY\",\"applied\":false,\"rotation_required\":%s,\"current_runtime_id\":\"%s\",\"candidate_runtime_id\":\"%s\",\"config_bundle_sha256\":\"%s\"}"
      (string_of_bool rotation_required)
      (Loom_hook_generation_drain.json_escape observation.current_runtime_id)
      (Loom_hook_generation_drain.json_escape observation.candidate_runtime_id) config_sha256
  else
    with_lock state (fun () ->
        let guardian_public_key_sha256, guardian_key_id =
          ensure_signing_pair state
        in
        let archive =
          archive_prior_generation ~state_directory:state ~current_id
            ~current_manifest_sha256 ~successor_id:candidate_id
            ~guardian_public_key_sha256 ~guardian_key_id
        in
        let observation =
          Loom_hook_generation_drain.live_observation ~verify_canaries:false root
        in
        require_prepare_boundary observation;
        let config_sha256 = Loom_hook_generation_drain.config_bundle_sha256 root in
        if config_sha256 = String.make 64 '0' then failf "guardian-config-bundle-unbound";
        let probe_sha256 = rollback_probe state current candidate in
        let created_utc =
          Loom_hook_generation_drain.utc_now (Unix.time ())
        in
        let rollback = rollback_marker observation config_sha256 probe_sha256 created_utc in
        let final =
          final_marker { observation with config_pair_sha256 = config_sha256 }
            guardian_public_key_sha256 created_utc
        in
        let rollback_path = atomic_write state "rollback-pair-tested.v1" rollback in
        let final_path = atomic_write state "final-config.v1" final in
        Printf.sprintf
          "{\"schema\":\"loom-native-hook-generation-guardian-v1\",\"action\":9046,\"semantic_authority\":\"Sounio\",\"operational_realization\":\"OCaml\",\"state\":\"PREPARED\",\"applied\":true,\"live_runtime_unchanged\":true,\"prior_generation_archived\":%s,\"archive_path\":\"%s\",\"current_runtime_id\":\"%s\",\"candidate_runtime_id\":\"%s\",\"config_bundle_sha256\":\"%s\",\"guardian_key_id\":\"%s\",\"guardian_public_key_sha256\":\"%s\",\"rollback_probe_sha256\":\"%s\",\"rollback_marker_sha256\":\"%s\",\"final_marker_sha256\":\"%s\",\"rollback_marker_path\":\"%s\",\"final_marker_path\":\"%s\"}"
          (string_of_bool (Option.is_some archive))
          (Loom_hook_generation_drain.json_escape (Option.value ~default:"" archive))
          (Loom_hook_generation_drain.json_escape observation.current_runtime_id)
          (Loom_hook_generation_drain.json_escape observation.candidate_runtime_id)
          config_sha256 guardian_key_id guardian_public_key_sha256 probe_sha256
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
