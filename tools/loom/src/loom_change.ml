open Unix

exception Error of string

let semantic_manifest_sha256 =
  "c84c5e7ff608f86ac51872de143516b0feb0981d0ee962583e2c62f66cbbacfb"

let material_manifest_sha256 =
  "662e01af4aed45ab22a0cfce283fd7aa9ec8775a65a2fb5a7a94a02c2c174c00"

let max_mutation_bytes = 1024 * 1024
let grant_ttl_us = 120_000_000L

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format
let sha256 = Loom_exec.sha256
let hex_encode = Loom_exec.hex_encode
let hex_decode = Loom_exec.hex_decode
let starts_with = Loom_exec.starts_with
let read_file = Loom_exec.read_file

let valid_sha256 value =
  String.length value = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let valid_git_oid value =
  (String.length value = 40 || String.length value = 64)
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let required_mode () =
  match Sys.getenv_opt "SOUNIO_LOOM_SOVEREIGN_CHANGE_REQUIRED" with
  | Some "1" -> true
  | None | Some "0" -> false
  | Some _ -> failf "change-required-mode-invalid"

type mutation =
  | Write of { path : string; content : string }
  | Edit of {
      path : string;
      old_string : string;
      new_string : string;
      replace_all : bool;
    }
  | Apply_patch of string

type file_state = Missing | Regular of string

type prepared = {
  descriptor : string;
  call_id : string;
  event_sha256 : string;
  patch_sha256 : string;
  session_id : string;
  paths : string list;
  head_sha256 : string;
  index_sha256 : string;
  ungranted_sha256 : string;
  original_states : (string * file_state) list;
  expected_states : (string * file_state) list;
  expected_post_sha256 : string;
  stage_root : string;
  material_prepare_frame_sha256 : string;
  material_prepare_decision : string;
  issued_us : int64;
  expires_us : int64;
}

type consumed = {
  consumed_descriptor : string;
  consumed_event_sha256 : string;
  consumed_patch_sha256 : string;
  consumed_session_id : string;
  consumed_paths : string list;
  consumed_head_sha256 : string;
  consumed_pre_index_sha256 : string;
  consumed_post_index_sha256 : string;
  consumed_post_sha256 : string;
  consumed_states : (string * file_state) list;
  consumed_material_frame_sha256 : string;
  consumed_material_decision : string;
  consumed_us : int64;
}

type commit_receipt = {
  commit_descriptor : string;
  commit_oid : string;
  commit_tree_oid : string;
  commit_parent_oid : string;
  commit_ref : string;
  commit_message_sha256 : string;
  commit_paths : string list;
  commit_changes_sha256 : string;
  commit_material_frame_sha256 : string;
  commit_material_decision : string;
  commit_ci_descriptor : string;
  commit_ci_frame_sha256 : string;
  commit_ci_decision : string;
  commit_source_sha256 : string;
  commit_semantics_sha256 : string;
  commit_authority_executable_sha256 : string;
  commit_toolchain : string;
  commit_hardware : string;
  commit_command_sha256 : string;
  commit_result_sha256 : string;
  commit_us : int64;
}

type remote_prepared = {
  remote_descriptor : string;
  remote_stage_root : string;
}

let encode_record fields =
  fields
  |> List.map (fun (key, value) -> key ^ "=" ^ hex_encode value)
  |> String.concat "\n"
  |> fun body -> body ^ "\n"

let decode_record label payload =
  if payload = "" || payload.[String.length payload - 1] <> '\n' then
    failf "%s-record-missing-final-newline" label;
  let fields = Hashtbl.create 8 in
  String.split_on_char '\n' payload
  |> List.iter (fun line ->
         if line <> "" then
           match String.index_opt line '=' with
           | None -> failf "%s-record-field-malformed" label
           | Some index ->
               let key = String.sub line 0 index in
               let value =
                 String.sub line (index + 1) (String.length line - index - 1)
                 |> hex_decode (label ^ "-" ^ key)
               in
               if key = "" || Hashtbl.mem fields key then
                 failf "%s-record-field-duplicate:%s" label key;
               Hashtbl.add fields key value);
  fields

let field label fields key =
  match Hashtbl.find_opt fields key with
  | Some value -> value
  | None -> failf "%s-record-field-missing:%s" label key

let encode_mutation = function
  | Write { path; content } ->
      encode_record [ ("kind", "Write"); ("path", path); ("content", content) ]
  | Edit { path; old_string; new_string; replace_all } ->
      encode_record
        [ ("kind", "Edit"); ("path", path); ("old", old_string);
          ("new", new_string);
          ("replace_all", if replace_all then "true" else "false") ]
  | Apply_patch patch ->
      encode_record [ ("kind", "apply_patch"); ("patch", patch) ]

let decode_mutation payload =
  if String.length payload > max_mutation_bytes then failf "change-payload-too-large";
  let fields = decode_record "change" payload in
  match field "change" fields "kind" with
  | "Write" ->
      Write
        { path = field "change" fields "path";
          content = field "change" fields "content" }
  | "Edit" ->
      let replace_all =
        match field "change" fields "replace_all" with
        | "true" -> true
        | "false" -> false
        | _ -> failf "change-edit-replace-all-invalid"
      in
      Edit
        { path = field "change" fields "path";
          old_string = field "change" fields "old";
          new_string = field "change" fields "new";
          replace_all }
  | "apply_patch" -> Apply_patch (field "change" fields "patch")
  | kind -> failf "change-kind-refused:%s" kind

let mutation_kind = function
  | Write _ -> "Write"
  | Edit _ -> "Edit"
  | Apply_patch _ -> "apply_patch"

let mutation_paths = function
  | Write { path; _ } | Edit { path; _ } -> [ path ]
  | Apply_patch patch ->
      String.split_on_char '\n' patch
      |> List.filter_map (fun line ->
             List.find_map
               (fun prefix ->
                 if starts_with line prefix then
                   Some
                     (String.sub line (String.length prefix)
                        (String.length line - String.length prefix))
                 else None)
               [ "*** Add File: "; "*** Update File: "; "*** Delete File: " ])
      |> List.sort_uniq String.compare

let valid_relative_path path =
  path <> "" && path <> "." && Filename.is_relative path
  && not
       (String.split_on_char '/' path
        |> List.exists (fun part -> part = "" || part = "." || part = ".."))
  && (match String.split_on_char '/' path with ".git" :: _ -> false | _ -> true)

let safe_path root path =
  if not (valid_relative_path path) then failf "change-path-invalid:%s" path;
  let root = Unix.realpath root in
  let candidate = Filename.concat root path in
  let rec existing candidate suffix =
    if Sys.file_exists candidate then
      List.fold_left Filename.concat (Unix.realpath candidate) suffix
    else
      let parent = Filename.dirname candidate in
      if parent = candidate then failf "change-path-parent-missing:%s" path;
      existing parent (Filename.basename candidate :: suffix)
  in
  let resolved = existing candidate [] in
  let prefix = if root = "/" then "/" else root ^ "/" in
  if resolved <> root && not (starts_with resolved prefix) then
    failf "change-path-outside-worktree:%s" path;
  candidate

let normalize_declared_path root path =
  if Filename.is_relative path then path
  else
    let root = Unix.realpath root in
    let prefix = if root = "/" then "/" else root ^ "/" in
    if starts_with path prefix then
      String.sub path (String.length prefix) (String.length path - String.length prefix)
    else failf "change-path-outside-worktree:%s" path

let file_state root path =
  let absolute = safe_path root path in
  try
    let stats = Unix.lstat absolute in
    match stats.st_kind with
    | S_REG ->
        if stats.st_nlink <> 1 then failf "change-hardlink-refused:%s" path;
        Regular (read_file ~limit:max_mutation_bytes absolute)
    | S_LNK -> failf "change-symlink-refused:%s" path
    | _ -> failf "change-nonregular-refused:%s" path
  with Unix_error (ENOENT, _, _) -> Missing

let file_state_sha256 = function
  | Missing -> sha256 "missing\000"
  | Regular content -> sha256 ("regular\000" ^ content)

let state_set_sha256 states =
  states
  |> List.sort (fun (left, _) (right, _) -> String.compare left right)
  |> List.map (fun (path, state) -> path ^ "\000" ^ file_state_sha256 state)
  |> String.concat "\000"
  |> fun value -> sha256 ("loom-change-state-v1\000" ^ value)

let rec mkdir_p path =
  if path = "" || path = "." || path = "/" || Sys.file_exists path then ()
  else (mkdir_p (Filename.dirname path); Unix.mkdir path 0o700)

let rec remove_tree path =
  match (Unix.lstat path).st_kind with
  | S_DIR ->
      Sys.readdir path
      |> Array.iter (fun name -> remove_tree (Filename.concat path name));
      Unix.rmdir path
  | _ -> Unix.unlink path
  | exception Unix_error (ENOENT, _, _) -> ()

let write_atomic ?(mode = 0o600) path content =
  let directory = Filename.dirname path in
  mkdir_p directory;
  let temporary =
    Filename.concat directory
      (Printf.sprintf ".loom-change.%d.%s" (Unix.getpid ())
         (Loom_exec.random_token ()))
  in
  let descriptor = Unix.openfile temporary [ O_WRONLY; O_CREAT; O_EXCL ] 0o600 in
  Fun.protect
    ~finally:(fun () ->
      (try Unix.close descriptor with _ -> ());
      if Sys.file_exists temporary then (try Unix.unlink temporary with _ -> ()))
    (fun () ->
      Unix.fchmod descriptor mode;
      Loom_exec.write_all descriptor content;
      Unix.fsync descriptor;
      Unix.close descriptor;
      Unix.rename temporary path;
      Loom_exec.fsync_directory directory)

let stage_path stage_root path = safe_path stage_root path

let populate_stage stage_root original_states =
  mkdir_p stage_root;
  List.iter
    (fun (path, state) ->
      let selected = stage_path stage_root path in
      match state with
      | Missing -> mkdir_p (Filename.dirname selected)
      | Regular content -> write_atomic selected content)
    original_states;
  Loom_exec.fsync_directory stage_root

let staged_states grant =
  List.map
    (fun (path, _) -> (path, file_state grant.stage_root path))
    grant.expected_states

let materialize_states root original_states expected_states =
  let observed = List.map (fun (path, _) -> (path, file_state root path)) original_states in
  if observed <> original_states then failf "change-origin-preimage-drift";
  List.iter
    (fun (path, expected) ->
      let selected = safe_path root path in
      match expected with
      | Regular content ->
          let mode =
            match List.assoc path original_states with
            | Missing -> 0o644
            | Regular _ -> (Unix.lstat selected).st_perm
          in
          write_atomic ~mode selected content
      | Missing ->
          (match file_state root path with
          | Missing -> ()
          | Regular _ ->
              Unix.unlink selected;
              Loom_exec.fsync_directory (Filename.dirname selected)))
    expected_states

let run_git root arguments =
  let executable = Loom_exec.resolve_executable root "git" in
  Loom_exec.run_process ~cwd:root executable ("-C" :: root :: arguments)

let git_output root arguments =
  let result = run_git root arguments in
  if result.code <> 0 then
    failf "change-git-failed:%s:%s" (String.concat ":" arguments)
      (String.trim result.output);
  String.trim result.output

let head_sha256 root =
  let value = git_output root [ "rev-parse"; "HEAD" ] in
  if not (valid_git_oid value) then failf "change-head-invalid";
  sha256 ("git-head-v1\000" ^ value)

let index_sha256 root =
  let path = git_output root [ "rev-parse"; "--git-path"; "index" ] in
  let path = if Filename.is_relative path then Filename.concat root path else path in
  if Sys.file_exists path then Loom_exec.sha256_file path else sha256 "missing-index"

let ensure_index_clean root =
  let result = run_git root [ "diff"; "--cached"; "--quiet"; "HEAD"; "--" ] in
  match result.code with
  | 0 -> ()
  | 1 -> failf "change-index-dirty-before-prepare"
  | code -> failf "change-index-probe-failed:%d:%s" code (String.trim result.output)

let nul_fields value =
  String.split_on_char '\000' value |> List.filter (fun item -> item <> "")

let dirty_paths root =
  let result =
    run_git root [ "ls-files"; "-m"; "-d"; "-o"; "--exclude-standard"; "-z" ]
  in
  if result.code <> 0 then failf "change-worktree-probe-failed:%d" result.code;
  nul_fields result.output |> List.sort_uniq String.compare

let ungranted_sha256 root granted =
  dirty_paths root
  |> List.filter (fun path -> not (List.mem path granted))
  |> List.map (fun path -> (path, file_state root path))
  |> state_set_sha256

let find_from text pattern start =
  let text_length = String.length text and pattern_length = String.length pattern in
  let rec loop index =
    if pattern_length = 0 then Some start
    else if index + pattern_length > text_length then None
    else if String.sub text index pattern_length = pattern then Some index
    else loop (index + 1)
  in
  loop start

let replace_one text old_string new_string =
  if old_string = "" then failf "change-edit-empty-old-string";
  match find_from text old_string 0 with
  | None -> failf "change-edit-old-string-missing"
  | Some first ->
      (match find_from text old_string (first + String.length old_string) with
      | Some _ -> failf "change-edit-old-string-ambiguous"
      | None ->
          String.sub text 0 first ^ new_string ^
          String.sub text (first + String.length old_string)
            (String.length text - first - String.length old_string))

let replace_all text old_string new_string =
  if old_string = "" then failf "change-edit-empty-old-string";
  let output = Buffer.create (String.length text) in
  let rec loop cursor count =
    match find_from text old_string cursor with
    | None ->
        Buffer.add_substring output text cursor (String.length text - cursor);
        if count = 0 then failf "change-edit-old-string-missing";
        Buffer.contents output
    | Some index ->
        Buffer.add_substring output text cursor (index - cursor);
        Buffer.add_string output new_string;
        loop (index + String.length old_string) (count + 1)
  in
  loop 0 0

type patch_action =
  | Add_file of string * string list
  | Delete_file of string
  | Update_file of string * string list

let parse_patch patch =
  let lines = String.split_on_char '\n' patch in
  let lines =
    match List.rev lines with "" :: tail -> List.rev tail | _ -> lines
  in
  match lines with
  | "*** Begin Patch" :: tail ->
      let rec collect_body body = function
        | [] -> failf "change-patch-end-missing"
        | "*** End Patch" :: rest -> (List.rev body, `End rest)
        | line :: rest when starts_with line "*** Add File: "
                         || starts_with line "*** Update File: "
                         || starts_with line "*** Delete File: " ->
            (List.rev body, `Header (line, rest))
        | line :: rest -> collect_body (line :: body) rest
      in
      let rec actions values = function
        | [] -> failf "change-patch-end-missing"
        | "*** End Patch" :: [] -> List.rev values
        | "*** End Patch" :: _ -> failf "change-patch-trailing-data"
        | line :: rest when starts_with line "*** Add File: " ->
            let path =
              String.sub line 14 (String.length line - 14)
            in
            let body, next = collect_body [] rest in
            let action = Add_file (path, body) in
            (match next with
            | `End remaining -> actions (action :: values) ("*** End Patch" :: remaining)
            | `Header (header, remaining) -> actions (action :: values) (header :: remaining))
        | line :: rest when starts_with line "*** Update File: " ->
            let path =
              String.sub line 17 (String.length line - 17)
            in
            let body, next = collect_body [] rest in
            let action = Update_file (path, body) in
            (match next with
            | `End remaining -> actions (action :: values) ("*** End Patch" :: remaining)
            | `Header (header, remaining) -> actions (action :: values) (header :: remaining))
        | line :: rest when starts_with line "*** Delete File: " ->
            let path =
              String.sub line 17 (String.length line - 17)
            in
            (match rest with
            | ("*** End Patch" :: _ | []) -> actions (Delete_file path :: values) rest
            | header :: _ when starts_with header "*** Add File: "
                            || starts_with header "*** Update File: "
                            || starts_with header "*** Delete File: " ->
                actions (Delete_file path :: values) rest
            | _ -> failf "change-delete-patch-body-refused")
        | line :: _ -> failf "change-patch-header-refused:%s" line
      in
      actions [] tail
  | _ -> failf "change-patch-begin-missing"

let hunk_chunks body =
  let flush current values =
    if current = [] then values else List.rev current :: values
  in
  let rec loop current values = function
    | [] -> List.rev (flush current values)
    | line :: rest when starts_with line "@@" ->
        loop [] (flush current values) rest
    | "*** End of File" :: rest -> loop current values rest
    | line :: rest ->
        if line = "" then failf "change-patch-empty-body-line"
        else
          match line.[0] with
          | ' ' | '+' | '-' -> loop (line :: current) values rest
          | _ -> failf "change-patch-line-refused:%s" line
  in
  loop [] [] body

let apply_hunk content body =
  let chunks prefix =
    body
    |> List.filter_map (fun line ->
           match line.[0], prefix with
           | '+', `Old | '-', `New -> None
           | (' ' | '-' | '+'), _ ->
               Some (String.sub line 1 (String.length line - 1) ^ "\n")
           | _ -> failf "change-patch-line-refused:%s" line)
    |> String.concat ""
  in
  let old_chunk = chunks `Old and new_chunk = chunks `New in
  if old_chunk = "" then failf "change-update-hunk-empty-old"
  else replace_one content old_chunk new_chunk

let expected_patch_states root patch =
  parse_patch patch
  |> List.map (function
       | Add_file (path, body) ->
           let path = normalize_declared_path root path in
           (match file_state root path with
           | Missing -> ()
           | Regular _ -> failf "change-add-target-exists:%s" path);
           let content =
             body
             |> List.map (fun line ->
                    if starts_with line "+" then
                      String.sub line 1 (String.length line - 1)
                    else failf "change-add-line-refused:%s" line)
             |> String.concat "\n"
             |> fun value -> if body = [] then value else value ^ "\n"
           in
           (path, Regular content)
       | Delete_file path ->
           let path = normalize_declared_path root path in
           (match file_state root path with
           | Missing -> failf "change-delete-target-missing:%s" path
           | Regular _ -> ());
           (path, Missing)
       | Update_file (path, body) ->
           let path = normalize_declared_path root path in
           let content =
             match file_state root path with
             | Missing -> failf "change-update-target-missing:%s" path
             | Regular value -> value
           in
           let updated =
             List.fold_left apply_hunk content (hunk_chunks body)
           in
           (path, Regular updated))

let expected_states root mutation =
  match mutation with
  | Write { path; content } ->
      ignore (file_state root path);
      [ (path, Regular content) ]
  | Edit { path; old_string; new_string; replace_all = all } ->
      let content =
        match file_state root path with
        | Missing -> failf "change-edit-target-missing:%s" path
        | Regular value -> value
      in
      let updated =
        if all then replace_all content old_string new_string
        else replace_one content old_string new_string
      in
      [ (path, Regular updated) ]
  | Apply_patch patch -> expected_patch_states root patch

let sorted_paths paths =
  let paths = List.sort_uniq String.compare paths in
  if paths = [] then failf "change-path-set-empty";
  List.iter (fun path -> ignore (safe_path "." path)) [];
  paths

let path_set_sha256 paths =
  sha256 ("loom-change-paths-v1\000" ^ String.concat "\000" paths)

let current_time_us () =
  Int64.of_float (Unix.gettimeofday () *. 1_000_000.0)

type gate = {
  manifest : (string, string) Hashtbl.t;
  runtime : string;
  environment : string array;
}

type material_gate = {
  material_manifest : (string, string) Hashtbl.t;
  material_runtime : string;
  material_environment : string array;
}

let exact manifest key expected =
  let actual = Loom_exec.required manifest key in
  if actual <> expected then failf "change-freeze-%s-invalid:%s" key actual

let policy_root root =
  match Sys.getenv_opt "SOUNIO_LOOM_SOVEREIGN_CHANGE_ROOT" with
  | Some value when value <> "" -> Unix.realpath value
  | _ ->
      let worktree = Unix.realpath root in
      let local =
        Filename.concat worktree "tools/loom/sovereign_material_change.freeze.v2"
      in
      if Sys.file_exists local then worktree
      else
        let binary_dir = Filename.dirname (Unix.realpath Sys.executable_name) in
        let capsule =
          Filename.concat (Filename.dirname binary_dir) "policy/sovereign-change"
        in
        if Sys.file_exists
             (Filename.concat capsule
                "tools/loom/sovereign_material_change.freeze.v2")
        then Unix.realpath capsule
        else failf "sovereign-change-policy-root-missing"

let load_gate root =
  let selected = policy_root root in
  let manifest_path =
    Filename.concat selected "tools/loom/sovereign_change_kernel.freeze.v1"
  in
  if Loom_exec.sha256_file manifest_path <> semantic_manifest_sha256 then
    failf "change-semantic-manifest-hash-mismatch";
  let manifest = Loom_exec.parse_manifest manifest_path in
  exact manifest "schema" "loom-sovereign-change-kernel-freeze-v1";
  exact manifest "stage" "SEMANTICS_FROZEN";
  exact manifest "semantic_authority" "Sounio";
  exact manifest "action" "9043";
  exact manifest "parent_action" "9042-frozen+production-active";
  exact manifest "grant_resident_memory" "true";
  exact manifest "grant_is_bearer" "false";
  exact manifest "grant_single_use" "true";
  exact manifest "consume_atomic" "true";
  exact manifest "ci_policy" "consume-not-reinterpret";
  let sibling =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-sovereign-change-kernel"
  in
  let runtime =
    if Sys.file_exists sibling then sibling
    else
      Filename.concat selected
        "tools/loom/_build/default/src/sounio-loom-sovereign-change-kernel"
  in
  if Loom_exec.sha256_file runtime <> Loom_exec.required manifest "executable_sha256"
  then failf "change-authority-runtime-hash-mismatch";
  { manifest; runtime; environment = Loom_exec.environment_array_from
      (Loom_exec.environment_bindings ()) }

let admit gate mode =
  let prefix =
    match mode with
    | `Prepare -> "prepare"
    | `Consume -> "consume"
    | `Commit -> "commit"
    | `Ci -> "ci"
    | `Production -> "production"
  in
  let frame =
    String.concat " "
      [ Loom_exec.required gate.manifest "wire_schema";
        Loom_exec.required gate.manifest (prefix ^ "_mode");
        Loom_exec.required gate.manifest "stage_word";
        Loom_exec.required gate.manifest (prefix ^ "_word");
        Loom_exec.required gate.manifest "sabotage_count";
        Loom_exec.required gate.manifest "sabotage_required" ]
    ^ "\n"
  in
  let expected = Loom_exec.required gate.manifest (prefix ^ "_decision") in
  let result =
    Loom_exec.run_process ~input:frame ~environment:gate.environment ~cwd:"/"
      gate.runtime []
  in
  let observed = Loom_exec.first_line result.output in
  if result.code <> 0 || observed <> expected then
    failf "change-authority-%s-refused:%d:%s" prefix result.code observed;
  observed

let load_material_gate root =
  let selected = policy_root root in
  let manifest_path =
    Filename.concat selected "tools/loom/sovereign_material_change.freeze.v2"
  in
  if Loom_exec.sha256_file manifest_path <> material_manifest_sha256 then
    failf "material-change-semantic-manifest-hash-mismatch";
  let manifest = Loom_exec.parse_manifest manifest_path in
  exact manifest "schema" "loom-sovereign-material-change-freeze-v2";
  exact manifest "stage" "SEMANTICS_FROZEN";
  exact manifest "semantic_authority" "Sounio";
  exact manifest "action" "9044";
  exact manifest "parent_action" "9043-frozen";
  exact manifest "provider_root_readonly" "true";
  exact manifest "staging_outside_root" "true";
  exact manifest "grant_resident_memory" "true";
  exact manifest "grant_is_bearer" "false";
  exact manifest "grant_single_use" "true";
  let sibling =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-sovereign-material-change"
  in
  let runtime =
    if Sys.file_exists sibling then sibling
    else
      Filename.concat selected
        "tools/loom/_build/default/src/sounio-loom-sovereign-material-change"
  in
  if Loom_exec.sha256_file runtime <> Loom_exec.required manifest "executable_sha256"
  then failf "material-change-authority-runtime-hash-mismatch";
  { material_manifest = manifest; material_runtime = runtime;
    material_environment = Loom_exec.environment_array_from
      (Loom_exec.environment_bindings ()) }

let descriptor_words descriptor =
  if not (valid_sha256 descriptor) then failf "material-descriptor-invalid";
  List.init 4 (fun index ->
      let chunk = String.sub descriptor (index * 16) 16 in
      let value = Int64.of_string ("0x" ^ chunk) in
      Int64.logand value Int64.max_int |> Int64.to_string)

let admit_material gate mode descriptor =
  let prefix =
    match mode with
    | `Prepare -> "prepare"
    | `Consume -> "consume"
    | `Commit -> "commit"
    | `Ci -> "ci"
    | `Claim -> "claim"
  in
  let frame =
    String.concat " "
      ([ Loom_exec.required gate.material_manifest "wire_schema";
         Loom_exec.required gate.material_manifest (prefix ^ "_mode");
         Loom_exec.required gate.material_manifest "stage_word";
         Loom_exec.required gate.material_manifest (prefix ^ "_word") ]
       @ descriptor_words descriptor
       @ [ Loom_exec.required gate.material_manifest "sabotage_count";
           Loom_exec.required gate.material_manifest "sabotage_required" ])
    ^ "\n"
  in
  let expected =
    Loom_exec.required gate.material_manifest (prefix ^ "_decision")
  in
  let result =
    Loom_exec.run_process ~input:frame ~environment:gate.material_environment
      ~cwd:"/" gate.material_runtime []
  in
  let observed = Loom_exec.first_line result.output in
  if result.code <> 0 || observed <> expected then
    failf "material-change-authority-%s-refused:%d:%s" prefix result.code
      observed;
  (sha256 frame, observed)

let validate_paths root mutation paths =
  let paths = List.sort_uniq String.compare paths in
  List.iter (fun path -> ignore (safe_path root path)) paths;
  let mutation_paths =
    mutation_paths mutation
    |> List.map (normalize_declared_path root)
    |> List.sort_uniq String.compare
  in
  if paths = [] || paths <> mutation_paths then
    failf "change-path-set-mismatch";
  paths

let prepare ~root ~stage_parent ~kernel_generation ~session_id ~call_id
    ~event_sha256 ~patch_sha256 ~mutation_payload ~paths
    ~provider_root_readonly =
  if not (valid_sha256 event_sha256 && valid_sha256 patch_sha256)
  then failf "change-prepare-digest-invalid";
  if call_id = "" || String.length call_id > 256 then
    failf "change-call-id-invalid";
  if sha256 mutation_payload <> patch_sha256 then
    failf "change-patch-digest-mismatch";
  let mutation = decode_mutation mutation_payload in
  let paths = validate_paths root mutation paths in
  let expected_descriptor =
    sha256
      (String.concat "\000"
         [ "loom-change-descriptor-v2"; kernel_generation; session_id; call_id;
           event_sha256; patch_sha256; path_set_sha256 paths ])
  in
  let descriptor = expected_descriptor in
  if not provider_root_readonly then failf "material-provider-root-not-readonly";
  ensure_index_clean root;
  let head_sha256 = head_sha256 root in
  let index_sha256 = index_sha256 root in
  let ungranted_sha256 = ungranted_sha256 root paths in
  let original_states = List.map (fun path -> (path, file_state root path)) paths in
  let expected_states = expected_states root mutation in
  let expected_post_sha256 = state_set_sha256 expected_states in
  let gate = load_gate root in
  ignore (admit gate `Prepare);
  let material_gate = load_material_gate root in
  let material_prepare_frame_sha256, material_prepare_decision =
    admit_material material_gate `Prepare descriptor
  in
  let stage_root = Filename.concat stage_parent descriptor in
  if Sys.file_exists stage_root then failf "change-stage-collision";
  populate_stage stage_root original_states;
  let issued_us = current_time_us () in
  { descriptor; call_id; event_sha256; patch_sha256; session_id; paths;
    head_sha256; index_sha256; ungranted_sha256; original_states;
    expected_states; expected_post_sha256; stage_root;
    material_prepare_frame_sha256; material_prepare_decision; issued_us;
    expires_us = Int64.add issued_us grant_ttl_us }

let consume ~root grant ~session_id ~call_id ~event_sha256 =
  if current_time_us () > grant.expires_us then failf "change-grant-expired";
  if session_id <> grant.session_id then failf "change-session-mismatch";
  if call_id <> grant.call_id then failf "change-call-id-mismatch";
  if not (valid_sha256 event_sha256) then failf "change-post-event-digest-invalid";
  if head_sha256 root <> grant.head_sha256 then failf "change-head-drift";
  let post_index_sha256 = index_sha256 root in
  if post_index_sha256 <> grant.index_sha256 then failf "change-index-drift";
  if ungranted_sha256 root grant.paths <> grant.ungranted_sha256 then
    failf "change-ungranted-path-drift";
  let original_states =
    List.map (fun path -> (path, file_state root path)) grant.paths
  in
  if original_states <> grant.original_states then
    failf "change-origin-mutated-before-consume";
  let observed_staged_states = staged_states grant in
  let observed_staged_sha256 = state_set_sha256 observed_staged_states in
  if observed_staged_sha256 <> grant.expected_post_sha256 ||
     observed_staged_states <> grant.expected_states
  then failf "change-staged-post-image-mismatch";
  let gate = load_gate root in
  ignore (admit gate `Consume);
  let material_gate = load_material_gate root in
  let consumed_material_frame_sha256, consumed_material_decision =
    admit_material material_gate `Consume grant.descriptor
  in
  materialize_states root grant.original_states grant.expected_states;
  let observed_states =
    List.map (fun path -> (path, file_state root path)) grant.paths
  in
  let observed_post_sha256 = state_set_sha256 observed_states in
  if observed_post_sha256 <> grant.expected_post_sha256 ||
     observed_states <> grant.expected_states
  then failf "change-kernel-materialization-mismatch";
  remove_tree grant.stage_root;
  { consumed_descriptor = grant.descriptor;
    consumed_event_sha256 = event_sha256;
    consumed_patch_sha256 = grant.patch_sha256;
    consumed_session_id = session_id;
    consumed_paths = grant.paths;
    consumed_head_sha256 = grant.head_sha256;
    consumed_pre_index_sha256 = grant.index_sha256;
    consumed_post_index_sha256 = post_index_sha256;
    consumed_post_sha256 = observed_post_sha256;
    consumed_states = observed_states;
    consumed_material_frame_sha256;
    consumed_material_decision;
    consumed_us = current_time_us () }

let consumed_digest value =
  sha256
    (String.concat "\000"
       [ "loom-change-consumed-v2"; value.consumed_descriptor;
         value.consumed_event_sha256; value.consumed_patch_sha256;
         value.consumed_session_id; String.concat "\000" value.consumed_paths;
         value.consumed_head_sha256; value.consumed_pre_index_sha256;
         value.consumed_post_index_sha256; value.consumed_post_sha256;
         value.consumed_material_frame_sha256;
         value.consumed_material_decision;
         Int64.to_string value.consumed_us ])

let replace_environment name value environment =
  let prefix = name ^ "=" in
  Array.to_list environment
  |> List.filter (fun item -> not (starts_with item prefix))
  |> fun retained -> Array.of_list ((prefix ^ value) :: retained)

let git_index_environment index =
  Unix.environment ()
  |> replace_environment "GIT_INDEX_FILE" index
  |> replace_environment "GIT_AUTHOR_NAME" "Loom Sovereign Change Kernel"
  |> replace_environment "GIT_AUTHOR_EMAIL" "loom@sounio.local"
  |> replace_environment "GIT_COMMITTER_NAME" "Loom Sovereign Change Kernel"
  |> replace_environment "GIT_COMMITTER_EMAIL" "loom@sounio.local"

let run_git_index root index ?(input = "") arguments =
  let executable = Loom_exec.resolve_executable root "git" in
  Loom_exec.run_process ~input ~environment:(git_index_environment index)
    ~cwd:root executable ("-C" :: root :: arguments)

let git_index_output root index ?input arguments =
  let result = run_git_index root index ?input arguments in
  if result.code <> 0 then
    failf "change-commit-git-failed:%s:%s" (String.concat ":" arguments)
      (String.trim result.output);
  String.trim result.output

let receipt_line key value = key ^ "=" ^ hex_encode value ^ "\n"

let commit_receipt_body receipt =
  [ "schema", "loom-sovereign-change-commit-receipt-v1";
    "semantic_authority", "Sounio";
    "producing_language", "Sounio";
    "language_role", "SEMANTIC_AUTHORITY";
    "action", "9044";
    "freeze_manifest_sha256", material_manifest_sha256;
    "source_sha256", receipt.commit_source_sha256;
    "semantics_sha256", receipt.commit_semantics_sha256;
    "authority_executable_sha256", receipt.commit_authority_executable_sha256;
    "change_descriptor", receipt.commit_descriptor;
    "changes_sha256", receipt.commit_changes_sha256;
    "commit_oid", receipt.commit_oid;
    "tree_oid", receipt.commit_tree_oid;
    "parent_oid", receipt.commit_parent_oid;
    "ref", receipt.commit_ref;
    "message_sha256", receipt.commit_message_sha256;
    "paths", String.concat "\000" receipt.commit_paths;
    "commit_frame_sha256", receipt.commit_material_frame_sha256;
    "commit_decision", receipt.commit_material_decision;
    "ci_descriptor", receipt.commit_ci_descriptor;
    "ci_frame_sha256", receipt.commit_ci_frame_sha256;
    "ci_decision", receipt.commit_ci_decision;
    "ci_policy", "consume-not-reinterpret";
    "toolchain", receipt.commit_toolchain;
    "hardware", receipt.commit_hardware;
    "command_sha256", receipt.commit_command_sha256;
    "result_sha256", receipt.commit_result_sha256;
    "commit_us", Int64.to_string receipt.commit_us ]
  |> List.map (fun (key, value) -> receipt_line key value)
  |> String.concat ""

let persist_commit_receipt ~root receipt =
  let common = git_output root [ "rev-parse"; "--path-format=absolute";
                                  "--git-common-dir" ] in
  let directory = Filename.concat common "sounio-loom-change-receipts" in
  mkdir_p directory;
  Unix.chmod directory 0o700;
  let body = commit_receipt_body receipt in
  let digest = sha256 body in
  let content = body ^ receipt_line "receipt_sha256" digest in
  let path = Filename.concat directory (digest ^ ".receipt") in
  if Sys.file_exists path then (
    if read_file path <> content then failf "change-commit-receipt-collision")
  else write_atomic ~mode:0o600 path content;
  (digest, path, content)

let commit_changes ~root ~message changes =
  if changes = [] then failf "change-commit-no-consumed-changes";
  if message = "" || String.length message > 16384 || String.contains message '\000'
  then failf "change-commit-message-invalid";
  let changes =
    List.sort
      (fun left right -> Int64.compare left.consumed_us right.consumed_us)
      changes
  in
  let parent_oid = git_output root [ "rev-parse"; "HEAD" ] in
  if not (valid_git_oid parent_oid) then failf "change-commit-parent-invalid";
  let parent_sha256 = sha256 ("git-head-v1\000" ^ parent_oid) in
  List.iter
    (fun change ->
      if change.consumed_head_sha256 <> parent_sha256 then
        failf "change-commit-parent-drift";
      if change.consumed_post_index_sha256 <> index_sha256 root then
        failf "change-commit-index-drift")
    changes;
  let latest = Hashtbl.create 16 in
  List.iter
    (fun change ->
      List.iter (fun (path, state) -> Hashtbl.replace latest path state)
        change.consumed_states)
    changes;
  let commit_states =
    Hashtbl.fold (fun path state values -> (path, state) :: values) latest []
    |> List.sort (fun (left, _) (right, _) -> String.compare left right)
  in
  List.iter
    (fun (path, expected) ->
      if file_state root path <> expected then
        failf "change-commit-postimage-drift:%s" path)
    commit_states;
  let paths = List.map fst commit_states in
  let changes_sha256 =
    changes |> List.map consumed_digest |> String.concat "\000"
    |> fun value -> sha256 ("loom-change-commit-members-v1\000" ^ value)
  in
  let descriptor =
    sha256
      (String.concat "\000"
         [ "loom-change-commit-v1"; parent_oid; sha256 message;
           path_set_sha256 paths; changes_sha256 ])
  in
  let common = git_output root [ "rev-parse"; "--path-format=absolute";
                                  "--git-common-dir" ] in
  let index = git_output root [ "rev-parse"; "--git-path"; "index" ] in
  let index = if Filename.is_relative index then Filename.concat root index else index in
  let temporary_index =
    Filename.concat common
      (Printf.sprintf ".loom-change-index.%d.%s" (Unix.getpid ())
         (Loom_exec.random_token ()))
  in
  Fun.protect
    ~finally:(fun () ->
      if Sys.file_exists temporary_index then
        (try Unix.unlink temporary_index with _ -> ()))
    (fun () ->
      ignore (git_index_output root temporary_index [ "read-tree"; parent_oid ]);
      ignore
        (git_index_output root temporary_index
           ([ "add"; "-A"; "--" ] @ paths));
      let tree_oid = git_index_output root temporary_index [ "write-tree" ] in
      if not (valid_git_oid tree_oid) then failf "change-commit-tree-invalid";
      let material_gate = load_material_gate root in
      let material_frame_sha256, material_decision =
        admit_material material_gate `Commit descriptor
      in
      List.iter
        (fun (path, expected) ->
          if file_state root path <> expected then
            failf "change-commit-postimage-raced:%s" path)
        commit_states;
      if git_output root [ "rev-parse"; "HEAD" ] <> parent_oid then
        failf "change-commit-parent-raced";
      let commit_oid =
        git_index_output root temporary_index ~input:(message ^ "\n")
          [ "commit-tree"; tree_oid; "-p"; parent_oid ]
      in
      if not (valid_git_oid commit_oid) then failf "change-commit-oid-invalid";
      let reference = git_output root [ "symbolic-ref"; "HEAD" ] in
      if not (starts_with reference "refs/heads/") then
        failf "change-commit-detached-head-refused";
      let ci_descriptor =
        sha256
          (String.concat "\000"
             [ "loom-change-ci-v1"; descriptor; commit_oid; tree_oid;
               parent_oid; material_manifest_sha256 ])
      in
      let ci_frame_sha256, ci_decision =
        admit_material material_gate `Ci ci_descriptor
      in
      let source_sha256 =
        Loom_exec.required material_gate.material_manifest "source_sha256"
      in
      let semantics_sha256 =
        Loom_exec.required material_gate.material_manifest "semantics_sha256"
      in
      let authority_executable_sha256 =
        Loom_exec.required material_gate.material_manifest "executable_sha256"
      in
      let toolchain =
        Printf.sprintf "OCaml-%s+Sounio-9044 executable_sha256=%s"
          Sys.ocaml_version authority_executable_sha256
      in
      let hardware =
        String.concat ";"
          [ "host=" ^ Unix.gethostname (); "os=" ^ Sys.os_type;
            "word_size=" ^ string_of_int Sys.word_size;
            "cpuinfo_sha256=" ^
              (if Sys.file_exists "/proc/cpuinfo" then
                 Loom_exec.sha256_file "/proc/cpuinfo"
               else sha256 "unavailable") ]
      in
      let command_sha256 = sha256 ("git-commit-v1\000" ^ message) in
      let result_sha256 =
        sha256 (String.concat "\000" [ commit_oid; tree_oid; parent_oid ])
      in
      let receipt =
        { commit_descriptor = descriptor; commit_oid; commit_tree_oid = tree_oid;
          commit_parent_oid = parent_oid; commit_ref = reference;
          commit_message_sha256 = sha256 message; commit_paths = paths;
          commit_changes_sha256 = changes_sha256;
          commit_material_frame_sha256 = material_frame_sha256;
          commit_material_decision = material_decision;
          commit_ci_descriptor = ci_descriptor;
          commit_ci_frame_sha256 = ci_frame_sha256;
          commit_ci_decision = ci_decision;
          commit_source_sha256 = source_sha256;
          commit_semantics_sha256 = semantics_sha256;
          commit_authority_executable_sha256 = authority_executable_sha256;
          commit_toolchain = toolchain; commit_hardware = hardware;
          commit_command_sha256 = command_sha256;
          commit_result_sha256 = result_sha256;
          commit_us = current_time_us () }
      in
      let receipt_sha256, receipt_path, receipt_content =
        persist_commit_receipt ~root receipt
      in
      let receipt_blob_oid =
        git_index_output root temporary_index ~input:receipt_content
          [ "hash-object"; "-w"; "--stdin" ]
      in
      if not (valid_git_oid receipt_blob_oid) then
        failf "change-commit-receipt-blob-invalid";
      List.iter
        (fun (path, expected) ->
          if file_state root path <> expected then
            failf "change-commit-postimage-final-race:%s" path)
        commit_states;
      if git_output root [ "rev-parse"; "HEAD" ] <> parent_oid then
        failf "change-commit-parent-final-race";
      let receipt_ref = "refs/loom/change-receipts/" ^ receipt_sha256 in
      let transaction =
        String.concat "\n"
          [ "start";
            Printf.sprintf "update %s %s %s" reference commit_oid parent_oid;
            Printf.sprintf "create %s %s" receipt_ref receipt_blob_oid;
            "prepare"; "commit"; "" ]
      in
      ignore
        (git_index_output root temporary_index ~input:transaction
           [ "update-ref"; "--stdin" ]);
      Unix.rename temporary_index index;
      Loom_exec.fsync_directory common;
      (receipt, receipt_sha256, receipt_path))

let commit_receipt_digest receipt =
  sha256
    (String.concat "\000"
       [ "loom-change-commit-receipt-v1"; receipt.commit_descriptor;
         receipt.commit_oid; receipt.commit_tree_oid; receipt.commit_parent_oid;
         receipt.commit_ref; receipt.commit_message_sha256;
         String.concat "\000" receipt.commit_paths;
         receipt.commit_changes_sha256; receipt.commit_material_frame_sha256;
         receipt.commit_material_decision; receipt.commit_ci_descriptor;
         receipt.commit_ci_frame_sha256; receipt.commit_ci_decision;
         receipt.commit_source_sha256; receipt.commit_semantics_sha256;
         receipt.commit_authority_executable_sha256; receipt.commit_toolchain;
         receipt.commit_hardware; receipt.commit_command_sha256;
         receipt.commit_result_sha256; Int64.to_string receipt.commit_us ])

let receipt_fields content = decode_record "commit-receipt" content

let verify_ci_receipt ~root ~path =
  let selected = Unix.realpath path in
  let common = git_output root [ "rev-parse"; "--path-format=absolute";
                                  "--git-common-dir" ] in
  let directory = Unix.realpath (Filename.concat common "sounio-loom-change-receipts") in
  if Filename.dirname selected <> directory then failf "change-ci-receipt-outside-store";
  let stats = Unix.lstat selected in
  if stats.st_kind <> S_REG || stats.st_nlink <> 1 || stats.st_perm land 0o077 <> 0
  then failf "change-ci-receipt-insecure";
  let content = read_file selected in
  let fields = receipt_fields content in
  let receipt_sha256 = field "commit-receipt" fields "receipt_sha256" in
  let lines = String.split_on_char '\n' content in
  let body =
    lines
    |> List.filter (fun line -> line <> "" && not (starts_with line "receipt_sha256="))
    |> String.concat "\n" |> fun value -> value ^ "\n"
  in
  if sha256 body <> receipt_sha256 then failf "change-ci-receipt-hash-mismatch";
  let require key expected =
    if field "commit-receipt" fields key <> expected then
      failf "change-ci-receipt-field-mismatch:%s" key
  in
  require "schema" "loom-sovereign-change-commit-receipt-v1";
  require "semantic_authority" "Sounio";
  require "producing_language" "Sounio";
  require "language_role" "SEMANTIC_AUTHORITY";
  require "action" "9044";
  require "freeze_manifest_sha256" material_manifest_sha256;
  require "ci_policy" "consume-not-reinterpret";
  let receipt_ref = "refs/loom/change-receipts/" ^ receipt_sha256 in
  let receipt_ref_oid = git_output root [ "rev-parse"; "--verify"; receipt_ref ] in
  let receipt_blob_oid =
    let executable = Loom_exec.resolve_executable root "git" in
    let result =
      Loom_exec.run_process ~input:content ~cwd:root executable
        [ "-C"; root; "hash-object"; "--stdin" ]
    in
    if result.code <> 0 then failf "change-ci-receipt-blob-hash-failed";
    String.trim result.output
  in
  if receipt_ref_oid <> receipt_blob_oid then
    failf "change-ci-receipt-ref-mismatch";
  let gate = load_material_gate root in
  require "source_sha256" (Loom_exec.required gate.material_manifest "source_sha256");
  require "semantics_sha256"
    (Loom_exec.required gate.material_manifest "semantics_sha256");
  require "authority_executable_sha256"
    (Loom_exec.required gate.material_manifest "executable_sha256");
  require "ci_decision" (Loom_exec.required gate.material_manifest "ci_decision");
  let oid = field "commit-receipt" fields "commit_oid" in
  let tree = field "commit-receipt" fields "tree_oid" in
  if not (valid_git_oid oid && valid_git_oid tree) then
    failf "change-ci-git-oid-invalid";
  let object_probe = run_git root [ "cat-file"; "-e"; oid ^ "^{commit}" ] in
  if object_probe.code <> 0 then failf "change-ci-commit-missing";
  if git_output root [ "show"; "-s"; "--format=%T"; oid ] <> tree then
    failf "change-ci-tree-mismatch";
  let reference = field "commit-receipt" fields "ref" in
  let ancestry = run_git root [ "merge-base"; "--is-ancestor"; oid; reference ] in
  if ancestry.code <> 0 then failf "change-ci-commit-not-reachable";
  let consumption_body =
    String.concat "\n"
      [ "schema=loom-sovereign-change-ci-consumption-v1";
        "source_receipt_sha256=" ^ receipt_sha256;
        "commit_oid=" ^ oid; "tree_oid=" ^ tree;
        "freeze_manifest_sha256=" ^ material_manifest_sha256;
        "ci_decision_sha256=" ^
          sha256 (field "commit-receipt" fields "ci_decision");
        "ci_policy=consume-not-reinterpret";
        "policy_executed_by_ci=false" ] ^ "\n"
  in
  let consumption_sha256 = sha256 consumption_body in
  let consumption =
    consumption_body ^ "consumption_sha256=" ^ consumption_sha256 ^ "\n"
  in
  let consumption_path = selected ^ ".ci" in
  if Sys.file_exists consumption_path then (
    if read_file consumption_path <> consumption then
      failf "change-ci-consumption-collision")
  else write_atomic ~mode:0o600 consumption_path consumption;
  (receipt_sha256, oid, tree, consumption_sha256, consumption_path)

let claim_ready ~root ~path =
  let receipt_sha256, oid, tree, consumption_sha256, consumption_path =
    verify_ci_receipt ~root ~path
  in
  let claim_descriptor =
    sha256
      (String.concat "\000"
         [ "loom-sovereign-change-claim-v1"; receipt_sha256; oid; tree;
           consumption_sha256; material_manifest_sha256 ])
  in
  let gate = load_material_gate root in
  let claim_frame_sha256, claim_decision =
    admit_material gate `Claim claim_descriptor
  in
  let claim_body =
    String.concat "\n"
      [ "schema=loom-sovereign-change-claim-v1";
        "semantic_authority=Sounio";
        "action=9044";
        "source_receipt_sha256=" ^ receipt_sha256;
        "commit_oid=" ^ oid;
        "tree_oid=" ^ tree;
        "ci_consumption_sha256=" ^ consumption_sha256;
        "ci_consumption_path_sha256=" ^ sha256 consumption_path;
        "claim_descriptor=" ^ claim_descriptor;
        "claim_frame_sha256=" ^ claim_frame_sha256;
        "claim_decision=" ^ claim_decision;
        "freeze_manifest_sha256=" ^ material_manifest_sha256;
        "ci_policy=consume-not-reinterpret";
        "policy_executed_by_ci=false";
        "claim_policy_executed_by=Sounio";
        "claim_ready=true" ] ^ "\n"
  in
  let claim_sha256 = sha256 claim_body in
  let claim = claim_body ^ "claim_sha256=" ^ claim_sha256 ^ "\n" in
  let claim_path = path ^ ".claim" in
  if Sys.file_exists claim_path then (
    if read_file claim_path <> claim then failf "change-claim-collision")
  else write_atomic ~mode:0o600 claim_path claim;
  (receipt_sha256, oid, consumption_sha256, claim_sha256, claim_path,
   claim_frame_sha256, claim_decision)

let mutation_request mutation paths =
  let mutation_payload = encode_mutation mutation in
  if String.length mutation_payload > max_mutation_bytes then
    failf "change-payload-too-large";
  let patch_sha256 = sha256 mutation_payload in
  let paths = List.sort_uniq String.compare paths in
  (mutation_payload, patch_sha256, paths)

let connect_socket () =
  let socket =
    match Sys.getenv_opt "SOUNIO_LOOM_SOCKET" with
    | Some value when value <> "" -> value
    | _ -> failf "change-kernel-socket-missing"
  in
  let descriptor = Unix.socket PF_UNIX SOCK_STREAM 0 in
  Unix.set_close_on_exec descriptor;
  try Unix.connect descriptor (ADDR_UNIX socket); descriptor
  with error -> Unix.close descriptor; raise error

let read_line descriptor =
  let buffer = Buffer.create 256 in
  let byte = Bytes.create 1 in
  let rec loop () =
    let count = Unix.read descriptor byte 0 1 in
    if count = 0 then failf "change-kernel-response-eof";
    let character = Bytes.get byte 0 in
    if character = '\n' then Buffer.contents buffer
    else if Buffer.length buffer >= 65536 then failf "change-kernel-response-too-large"
    else (Buffer.add_char buffer character; loop ())
  in
  loop ()

let exchange fields =
  let descriptor = connect_socket () in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      Loom_exec.write_all descriptor (String.concat "\t" fields ^ "\n");
      match String.split_on_char '\t' (read_line descriptor) with
      | "OK" :: operation :: rest -> (operation, rest)
      | "ERR" :: reason :: _ -> failf "change-kernel-refused:%s" reason
      | _ -> failf "change-kernel-response-invalid")

let instance () =
  match Sys.getenv_opt "SOUNIO_LOOM_INSTANCE_ID" with
  | Some value when value <> "" -> value
  | _ -> failf "change-instance-missing"

let path_fields paths = string_of_int (List.length paths) :: List.map hex_encode paths

let prepare_remote ~session_id ~call_id ~event_sha256 mutation paths =
  let payload, patch_sha256, paths = mutation_request mutation paths in
  let operation, response =
    exchange
      ([ "LOOM_CHANGE/2"; "PREPARE"; instance (); hex_encode session_id;
         hex_encode call_id; event_sha256; patch_sha256; hex_encode payload ]
       @ path_fields paths)
  in
  match operation, response with
  | "CHANGE_PREPARED", [ descriptor; stage_root ] when valid_sha256 descriptor ->
      { remote_descriptor = descriptor; remote_stage_root = hex_decode "stage-root" stage_root }
  | _ -> failf "change-prepare-response-invalid"

let consume_remote ~session_id ~call_id ~event_sha256 =
  let operation, response =
    exchange
      [ "LOOM_CHANGE/2"; "CONSUME"; instance (); hex_encode session_id;
        hex_encode call_id; event_sha256 ]
  in
  match operation, response with
  | "CHANGE_CONSUMED", [ digest ] when valid_sha256 digest -> digest
  | _ -> failf "change-consume-response-invalid"

let commit_remote ~session_id ~call_id ~event_sha256 ~message =
  if message = "" || String.length message > 16384 then
    failf "change-commit-message-invalid";
  let operation, response =
    exchange
      [ "LOOM_CHANGE/2"; "COMMIT"; instance (); hex_encode session_id;
        hex_encode call_id; event_sha256; hex_encode message ]
  in
  match operation, response with
  | "CHANGE_COMMITTED", [ receipt; oid; path_hex ]
    when valid_sha256 receipt && valid_git_oid oid ->
      (receipt, oid, hex_decode "commit-receipt-path" path_hex)
  | _ -> failf "change-commit-response-invalid"
