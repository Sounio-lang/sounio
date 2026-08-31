open Unix

exception Error of string

let pinned_manifest_sha256 =
  "5fe5e5c9cdcb83935770f58df52f2d614d11f8abde519c4a2505ca20998fae2e"

let pinned_native_hook_cutover_manifest_sha256 =
  "16a4f7e24e1fcdb71690b3031914b2fe6cd389ad866154b7bf73907f007cfc4a"

let max_event_bytes = 8 * 1024 * 1024
let process_timeout_seconds = 5.0
let coordination_process_timeout_seconds = 15.0

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let starts_with value prefix =
  String.length value >= String.length prefix
  && String.sub value 0 (String.length prefix) = prefix

let ends_with value suffix =
  String.length value >= String.length suffix
  && String.sub value (String.length value - String.length suffix)
       (String.length suffix) = suffix

let contains value needle =
  let value_length = String.length value and needle_length = String.length needle in
  let rec search index =
    if needle_length = 0 then true
    else if index + needle_length > value_length then false
    else if String.sub value index needle_length = needle then true
    else search (index + 1)
  in
  search 0

let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"

let trim = String.trim

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let read_file ?(limit = max_event_bytes) path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let output = Buffer.create 4096 in
      let bytes = Bytes.create 16384 in
      let rec loop total =
        let count = input channel bytes 0 (Bytes.length bytes) in
        if count = 0 then Buffer.contents output
        else if total + count > limit then failf "file-too-large:%s" path
        else (Buffer.add_subbytes output bytes 0 count; loop (total + count))
      in
      loop 0)

let read_stdin () =
  let buffer = Buffer.create 4096 in
  let bytes = Bytes.create 16384 in
  let rec loop total =
    let count = input Stdlib.stdin bytes 0 (Bytes.length bytes) in
    if count = 0 then Buffer.contents buffer
    else if total + count > max_event_bytes then failf "hook-event-too-large"
    else (Buffer.add_subbytes buffer bytes 0 count; loop (total + count))
  in
  loop 0

let sha256_file path =
  let stat = Unix.lstat path in
  if stat.st_kind <> S_REG then failf "file-not-regular:%s" path;
  let channel = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () ->
      Cryptokit.hash_channel (Cryptokit.Hash.sha256 ()) channel
      |> Cryptokit.transform_string (Cryptokit.Hexa.encode ()))

let rec mkdir_p path =
  if path = "" || path = "." || path = "/" || Sys.file_exists path then ()
  else (mkdir_p (Filename.dirname path); Unix.mkdir path 0o700)

let write_all descriptor value =
  let rec loop offset =
    if offset < String.length value then
      match Unix.write_substring descriptor value offset (String.length value - offset) with
      | 0 -> failf "short-process-input-write"
      | count -> loop (offset + count)
      | exception Unix_error (EINTR, _, _) -> loop offset
  in
  loop 0

type process_result = { code : int; output : string }

let exit_code = function
  | WEXITED code -> code
  | WSIGNALED signal | WSTOPPED signal -> 128 + signal

let replace_environment name value environment =
  let prefix = name ^ "=" in
  let retained =
    Array.to_list environment
    |> List.filter (fun item -> not (starts_with item prefix))
  in
  Array.of_list ((prefix ^ value) :: retained)

let drop_environment_prefix prefix environment =
  Array.to_list environment
  |> List.filter (fun item -> not (starts_with item prefix))
  |> Array.of_list

let run_process ?(input = "") ?(environment = Unix.environment ())
    ?(timeout_seconds = process_timeout_seconds) ~cwd command arguments =
  let stdin_read, stdin_write = Unix.pipe () in
  let output_read, output_write = Unix.pipe () in
  Unix.set_close_on_exec stdin_write;
  Unix.set_close_on_exec output_read;
  let argv = Array.of_list (command :: arguments) in
  let pid =
    match Unix.fork () with
    | 0 ->
        Unix.close stdin_write;
        Unix.close output_read;
        Unix.dup2 stdin_read Unix.stdin;
        Unix.dup2 output_write Unix.stdout;
        Unix.dup2 output_write Unix.stderr;
        if stdin_read <> Unix.stdin then Unix.close stdin_read;
        if output_write <> Unix.stdout && output_write <> Unix.stderr then
          Unix.close output_write;
        (try
           Unix.chdir cwd;
           Unix.execvpe command argv environment
         with _ -> Unix._exit 127)
    | pid -> pid
  in
  Unix.close stdin_read;
  Unix.close output_write;
  let close_noerr descriptor = try Unix.close descriptor with _ -> () in
  let kill_noerr () =
    (try Unix.kill pid Sys.sigkill with _ -> ());
    (try ignore (Unix.waitpid [] pid) with _ -> ())
  in
  Fun.protect
    ~finally:(fun () -> close_noerr stdin_write; close_noerr output_read)
    (fun () ->
      (try write_all stdin_write input
       with error -> kill_noerr (); raise error);
      Unix.close stdin_write;
      let deadline = Unix.gettimeofday () +. timeout_seconds in
      let output = Buffer.create 4096 in
      let bytes = Bytes.create 16384 in
      let rec drain () =
        let remaining = deadline -. Unix.gettimeofday () in
        if remaining <= 0.0 then (kill_noerr (); failf "process-timeout:%s" command);
        let ready, _, _ = Unix.select [ output_read ] [] [] remaining in
        if ready = [] then (kill_noerr (); failf "process-timeout:%s" command)
        else
          match Unix.read output_read bytes 0 (Bytes.length bytes) with
          | 0 -> ()
          | count ->
              if Buffer.length output + count > max_event_bytes then (
                kill_noerr ();
                failf "process-output-too-large:%s" command);
              Buffer.add_subbytes output bytes 0 count;
              drain ()
          | exception Unix_error (EINTR, _, _) -> drain ()
      in
      drain ();
      let _, status = Unix.waitpid [] pid in
      { code = exit_code status; output = Buffer.contents output })

let process_ok ?input ?environment ~cwd command arguments =
  let result = run_process ?input ?environment ~cwd command arguments in
  if result.code <> 0 then
    failf "process-failed:%s:rc=%d:%s" command result.code (trim result.output);
  trim result.output

type json_value =
  | Json_object of (string * json_value) list
  | Json_array of json_value list
  | Json_string of string
  | Json_number of string
  | Json_bool of bool
  | Json_null

let hex_value character =
  match character with
  | '0' .. '9' -> Char.code character - Char.code '0'
  | 'a' .. 'f' -> 10 + Char.code character - Char.code 'a'
  | 'A' .. 'F' -> 10 + Char.code character - Char.code 'A'
  | _ -> failf "invalid-json:bad-hex"

let parse_json value =
  let length = String.length value in
  let index = ref 0 in
  let invalid message = failf "invalid-json:%s:at=%d" message !index in
  let rec whitespace () =
    if !index < length then
      match value.[!index] with
      | ' ' | '\t' | '\n' | '\r' -> incr index; whitespace ()
      | _ -> ()
  and string_literal () =
    if !index >= length || value.[!index] <> '"' then invalid "expected-string";
    incr index;
    let output = Buffer.create 32 in
    let rec loop () =
      if !index >= length then invalid "unterminated-string";
      match value.[!index] with
      | '"' -> incr index; Buffer.contents output
      | '\\' ->
          incr index;
          if !index >= length then invalid "unterminated-escape";
          let escaped = value.[!index] in
          incr index;
          (match escaped with
          | '"' | '\\' | '/' -> Buffer.add_char output escaped
          | 'b' -> Buffer.add_char output '\b'
          | 'f' -> Buffer.add_char output '\012'
          | 'n' -> Buffer.add_char output '\n'
          | 'r' -> Buffer.add_char output '\r'
          | 't' -> Buffer.add_char output '\t'
          | 'u' ->
              if !index + 4 > length then invalid "short-unicode-escape";
              let code =
                (hex_value value.[!index] lsl 12)
                lor (hex_value value.[!index + 1] lsl 8)
                lor (hex_value value.[!index + 2] lsl 4)
                lor hex_value value.[!index + 3]
              in
              index := !index + 4;
              if code <= 0x7f then Buffer.add_char output (Char.chr code)
              else if code <= 0x7ff then (
                Buffer.add_char output (Char.chr (0xc0 lor (code lsr 6)));
                Buffer.add_char output (Char.chr (0x80 lor (code land 0x3f))))
              else (
                Buffer.add_char output (Char.chr (0xe0 lor (code lsr 12)));
                Buffer.add_char output
                  (Char.chr (0x80 lor ((code lsr 6) land 0x3f)));
                Buffer.add_char output (Char.chr (0x80 lor (code land 0x3f))))
          | _ -> invalid "unknown-escape");
          loop ()
      | character when Char.code character < 32 -> invalid "control-in-string"
      | character -> Buffer.add_char output character; incr index; loop ()
    in
    loop ()
  and number_literal () =
    let start = !index in
    if !index < length && value.[!index] = '-' then incr index;
    let digits () =
      let before = !index in
      while !index < length && value.[!index] >= '0' && value.[!index] <= '9' do
        incr index
      done;
      if before = !index then invalid "expected-number"
    in
    digits ();
    if !index < length && value.[!index] = '.' then (incr index; digits ());
    if !index < length && (value.[!index] = 'e' || value.[!index] = 'E') then (
      incr index;
      if !index < length && (value.[!index] = '+' || value.[!index] = '-') then
        incr index;
      digits ());
    String.sub value start (!index - start)
  and keyword literal parsed =
    let ending = !index + String.length literal in
    if ending > length || String.sub value !index (String.length literal) <> literal
    then invalid ("expected-" ^ literal);
    index := ending;
    parsed
  and item () =
    whitespace ();
    if !index >= length then invalid "expected-value";
    match value.[!index] with
    | '{' -> object_literal ()
    | '[' -> array_literal ()
    | '"' -> Json_string (string_literal ())
    | 't' -> keyword "true" (Json_bool true)
    | 'f' -> keyword "false" (Json_bool false)
    | 'n' -> keyword "null" Json_null
    | '-' | '0' .. '9' -> Json_number (number_literal ())
    | _ -> invalid "unexpected-token"
  and object_literal () =
    incr index;
    whitespace ();
    let rec members values =
      whitespace ();
      if !index < length && value.[!index] = '}' then (
        incr index;
        Json_object (List.rev values))
      else
        let key = string_literal () in
        if List.mem_assoc key values then invalid ("duplicate-key-" ^ key);
        whitespace ();
        if !index >= length || value.[!index] <> ':' then invalid "expected-colon";
        incr index;
        let member = item () in
        whitespace ();
        if !index < length && value.[!index] = ',' then (
          incr index;
          whitespace ();
          if !index < length && value.[!index] = '}' then
            invalid "trailing-object-comma";
          members ((key, member) :: values))
        else if !index < length && value.[!index] = '}' then (
          incr index;
          Json_object (List.rev ((key, member) :: values)))
        else invalid "expected-object-separator"
    in
    members []
  and array_literal () =
    incr index;
    whitespace ();
    let rec elements values =
      whitespace ();
      if !index < length && value.[!index] = ']' then (
        incr index;
        Json_array (List.rev values))
      else
        let element = item () in
        whitespace ();
        if !index < length && value.[!index] = ',' then (
          incr index;
          whitespace ();
          if !index < length && value.[!index] = ']' then
            invalid "trailing-array-comma";
          elements (element :: values))
        else if !index < length && value.[!index] = ']' then (
          incr index;
          Json_array (List.rev (element :: values)))
        else invalid "expected-array-separator"
    in
    elements []
  in
  let parsed = item () in
  whitespace ();
  if !index <> length then invalid "trailing-data";
  parsed

let json_escape value =
  let output = Buffer.create (String.length value + 8) in
  String.iter
    (fun character ->
      match character with
      | '"' -> Buffer.add_string output "\\\""
      | '\\' -> Buffer.add_string output "\\\\"
      | '\b' -> Buffer.add_string output "\\b"
      | '\012' -> Buffer.add_string output "\\f"
      | '\n' -> Buffer.add_string output "\\n"
      | '\r' -> Buffer.add_string output "\\r"
      | '\t' -> Buffer.add_string output "\\t"
      | value when Char.code value < 32 ->
          Buffer.add_string output (Printf.sprintf "\\u%04x" (Char.code value))
      | value -> Buffer.add_char output value)
    value;
  Buffer.contents output

let rec json_string = function
  | Json_object fields ->
      "{" ^
      String.concat ","
        (List.map
           (fun (name, value) ->
             "\"" ^ json_escape name ^ "\":" ^ json_string value)
           fields)
      ^ "}"
  | Json_array values -> "[" ^ String.concat "," (List.map json_string values) ^ "]"
  | Json_string value -> "\"" ^ json_escape value ^ "\""
  | Json_number value -> value
  | Json_bool true -> "true"
  | Json_bool false -> "false"
  | Json_null -> "null"

let object_field value name =
  match value with
  | Json_object fields -> List.assoc_opt name fields
  | _ -> failf "invalid-json:expected-object"

let string_field ?(default = "") value name =
  match object_field value name with
  | Some (Json_string found) -> found
  | Some Json_null | None -> default
  | Some _ -> failf "invalid-json:%s-must-be-string" name

let string_array_field value name =
  match object_field value name with
  | Some (Json_array values) ->
      Some
        (List.map
           (function
             | Json_string found -> found
             | _ -> failf "invalid-json:%s-items-must-be-strings" name)
           values)
  | Some Json_null | None -> None
  | Some _ -> failf "invalid-json:%s-must-be-array" name

let replace_object_field value name replacement =
  match value with
  | Json_object fields ->
      if not (List.mem_assoc name fields) then failf "invalid-json:missing-%s" name;
      Json_object
        (List.map
           (fun (field_name, field_value) ->
             if field_name = name then (field_name, replacement)
             else (field_name, field_value))
           fields)
  | _ -> failf "invalid-json:tool_input-must-be-object"

type hook_profile = {
  provider_id : string;
  provider_code : int;
  dialect_name : string;
  dialect_code : int;
  camel_case : bool;
}

let hook_profile agent =
  if starts_with agent "codex" then
    { provider_id = "codex"; provider_code = 1; dialect_name = "snake";
      dialect_code = 1; camel_case = false }
  else if starts_with agent "claude" then
    { provider_id = "claude"; provider_code = 2; dialect_name = "snake";
      dialect_code = 1; camel_case = false }
  else if starts_with agent "cursor" then
    { provider_id = "cursor"; provider_code = 3;
      dialect_name = "cursor-camel"; dialect_code = 2; camel_case = false }
  else if starts_with agent "grok" then
    { provider_id = "grok"; provider_code = 4;
      dialect_name = "grok-camel"; dialect_code = 3; camel_case = true }
  else failf "unsupported-hook-agent:%s" agent

let reject_alias event forbidden =
  match object_field event forbidden with
  | None -> ()
  | Some _ -> failf "provider-hook-dialect-mismatch:field=%s" forbidden

let aliased_field event profile snake camel =
  if profile.camel_case then (reject_alias event snake; object_field event camel)
  else (reject_alias event camel; object_field event snake)

let aliased_string ?(default = "") event profile snake camel =
  match aliased_field event profile snake camel with
  | Some (Json_string value) -> value
  | Some Json_null | None -> default
  | Some _ -> failf "invalid-json:%s-must-be-string" (if profile.camel_case then camel else snake)

let normalize_hook_event_name profile value =
  match profile.provider_id, value with
  | ("codex" | "claude"),
    ("SessionStart" | "SessionEnd" | "UserPromptSubmit" | "PreToolUse"
    | "PostToolUse" | "Stop") -> value
  | "cursor", ("SessionStart" | "sessionStart") -> "SessionStart"
  | "cursor", ("SessionEnd" | "sessionEnd") -> "SessionEnd"
  | "cursor", ("UserPromptSubmit" | "beforeSubmitPrompt") -> "UserPromptSubmit"
  | "cursor", ("PreToolUse" | "preToolUse" | "beforeShellExecution"
    | "beforeFileEdit") -> "PreToolUse"
  | "cursor", ("PostToolUse" | "postToolUse" | "afterShellExecution"
    | "afterFileEdit") -> "PostToolUse"
  | "cursor", ("Stop" | "stop") -> "Stop"
  | "grok", ("SessionStart" | "sessionStart" | "session_start") -> "SessionStart"
  | "grok", ("SessionEnd" | "sessionEnd" | "session_end") -> "SessionEnd"
  | "grok", ("UserPromptSubmit" | "beforeSubmitPrompt" | "user_prompt_submit") -> "UserPromptSubmit"
  | "grok", ("PreToolUse" | "preToolUse" | "pre_tool_use") -> "PreToolUse"
  | "grok", ("PostToolUse" | "postToolUse" | "post_tool_use") -> "PostToolUse"
  | "grok", ("Stop" | "stop") -> "Stop"
  | _ -> failf "provider-hook-event-unsupported:%s:%s" profile.provider_id value

let normalize_hook_tool profile value =
  match profile.provider_id, value with
  | "cursor", "run_terminal_command" -> "Bash"
  | "cursor", "search_replace" -> "Edit"
  | "cursor", "write_file" -> "Write"
  | _, value -> value

let cursor_workspace_root event =
  let direct = string_field event "cwd" in
  match string_array_field event "workspace_roots" with
  | None -> direct
  | Some [] -> failf "hook-workspace-roots-empty"
  | Some [ root ] when root = "" -> failf "hook-workspace-roots-empty"
  | Some [ root ] when direct = "" || direct = root -> root
  | Some [ _ ] -> failf "hook-workspace-root-conflict"
  | Some _ -> failf "hook-workspace-roots-ambiguous"

let normalize_hook_event profile event =
  let raw_name = aliased_string event profile "hook_event_name" "hookEventName" in
  if raw_name = "" then failf "hook-event-name-missing";
  let raw_session = aliased_string event profile "session_id" "sessionId" in
  if raw_session = "" then failf "hook-session-id-missing";
  let cwd =
    if profile.provider_id = "cursor" then cursor_workspace_root event
    else if profile.camel_case then
      let direct = string_field event "cwd" in
      if direct <> "" then direct else string_field event "workspaceRoot"
    else string_field event "cwd"
  in
  if cwd = "" then failf "hook-cwd-missing";
  let tool_name =
    aliased_string event profile "tool_name" "toolName"
    |> normalize_hook_tool profile
  in
  let tool_input = aliased_field event profile "tool_input" "toolInput" in
  let optional name value fields =
    match value with None -> fields | Some found -> (name, found) :: fields
  in
  let call_id =
    if profile.camel_case then
      match object_field event "toolUseId" with
      | Some value -> Some value
      | None -> object_field event "toolCallId"
    else
      match object_field event "tool_use_id" with
      | Some value -> Some value
      | None -> object_field event "tool_call_id"
  in
  let fields =
    [ ("hook_event_name", Json_string (normalize_hook_event_name profile raw_name));
      ("session_id", Json_string raw_session); ("cwd", Json_string cwd) ]
  in
  let fields =
    if tool_name = "" then fields else ("tool_name", Json_string tool_name) :: fields
  in
  Json_object
    (fields
     |> optional "tool_input" tool_input
     |> optional "tool_use_id" call_id)

let hook_event_code event_name =
  match event_name with
  | "SessionStart" -> 1
  | "UserPromptSubmit" -> 2
  | "PreToolUse" -> 3
  | "PostToolUse" -> 4
  | "Stop" -> 5
  | "SessionEnd" -> 6
  | _ -> failf "hook-event-unsupported:%s" event_name

let execution_tool name =
  List.mem name [ "Bash"; "Exec"; "exec_command"; "shell"; "Shell" ]

let execution_command input =
  match input with
  | Json_object fields ->
      let found =
        List.filter_map
          (fun name ->
            match List.assoc_opt name fields with
            | Some (Json_string command) -> Some (name, command)
            | Some Json_null -> None
            | Some _ -> failf "invalid-json:%s-must-be-string" name
            | None -> None)
          [ "command"; "cmd"; "script" ]
      in
      (match found with
      | [ value ] -> value
      | [] -> failf "execution-command-missing"
      | _ -> failf "execution-command-ambiguous")
  | _ -> failf "invalid-json:tool_input-must-be-object"

let execution_cwd event input root =
  let event_cwd = string_field ~default:root event "cwd" in
  let selected =
    match input with
    | Json_object fields ->
        (match List.assoc_opt "workdir" fields, List.assoc_opt "cwd" fields with
        | Some (Json_string value), _ when value <> "" -> value
        | Some Json_null, Some (Json_string value) when value <> "" -> value
        | None, Some (Json_string value) when value <> "" -> value
        | Some (Json_string _), _ | Some Json_null, _ | None, None -> event_cwd
        | Some _, _ -> failf "invalid-json:workdir-must-be-string"
        | None, Some _ -> failf "invalid-json:cwd-must-be-string")
    | _ -> failf "invalid-json:tool_input-must-be-object"
  in
  if Filename.is_relative selected then Filename.concat event_cwd selected else selected

let execution_hook_output ?(reason =
    "Sounio 9021 authorized one single-use execution capability") input field
    replacement =
  let updated_input = replace_object_field input field (Json_string replacement) in
  Json_object
    [ ("hookSpecificOutput",
       Json_object
         [ ("hookEventName", Json_string "PreToolUse");
           ("permissionDecision", Json_string "allow");
           ("permissionDecisionReason", Json_string reason);
           ("updatedInput", updated_input) ]) ]

let rec collect_named_strings names value =
  match value with
  | Json_object fields ->
      List.fold_left
        (fun collected (name, member) ->
          let direct =
            if List.mem name names then
              match member with Json_string text when text <> "" -> [ text ] | _ -> []
            else []
          in
          direct @ collect_named_strings names member @ collected)
        [] fields
  | Json_array values ->
      List.fold_left
        (fun collected member -> collect_named_strings names member @ collected)
        [] values
  | Json_string _ | Json_number _ | Json_bool _ | Json_null -> []

let unique values = List.sort_uniq String.compare values

let patch_paths patch =
  String.split_on_char '\n' patch
  |> List.filter_map (fun line ->
         let prefixes = [ "*** Add File: "; "*** Update File: "; "*** Delete File: " ] in
         List.find_map
           (fun prefix ->
             if starts_with line prefix then
               Some (String.sub line (String.length prefix)
                       (String.length line - String.length prefix))
             else None)
           prefixes)

let extract_paths event =
  let tool_name = string_field event "tool_name" in
  match object_field event "tool_input" with
  | None | Some Json_null -> []
  | Some input ->
      let direct = collect_named_strings [ "file_path"; "notebook_path" ] input in
      let patches =
        if List.mem tool_name [ "apply_patch"; "Edit"; "Write"; "MultiEdit" ] then
          match input with
          | Json_string patch -> patch_paths patch
          | _ ->
              collect_named_strings [ "patch"; "input" ] input
              |> List.concat_map patch_paths
        else []
      in
      unique (direct @ patches)

let change_tool name = List.mem name [ "Write"; "Edit"; "apply_patch" ]

let change_string_field input name =
  match object_field input name with
  | Some (Json_string value) -> value
  | Some _ -> failf "change-%s-must-be-string" name
  | None -> failf "change-%s-missing" name

let change_bool_field ~default input name =
  match object_field input name with
  | Some (Json_bool value) -> value
  | Some _ -> failf "change-%s-must-be-boolean" name
  | None -> default

let change_mutation event target_paths =
  let tool_name = string_field event "tool_name" in
  let input =
    match object_field event "tool_input" with
    | Some value -> value
    | None -> failf "change-tool-input-missing"
  in
  match tool_name, input, target_paths with
  | "Write", Json_object _, [ path ] ->
      Loom_change.Write { path; content = change_string_field input "content" }
  | "Edit", Json_object _, [ path ] ->
      Loom_change.Edit
        { path;
          old_string = change_string_field input "old_string";
          new_string = change_string_field input "new_string";
          replace_all = change_bool_field ~default:false input "replace_all" }
  | "apply_patch", Json_string patch, _ -> Loom_change.Apply_patch patch
  | "apply_patch", Json_object _, _ ->
      let patch =
        match object_field input "patch", object_field input "input" with
        | Some (Json_string value), _ -> value
        | None, Some (Json_string value) -> value
        | Some _, _ -> failf "change-patch-must-be-string"
        | None, Some _ -> failf "change-input-must-be-string"
        | None, None -> failf "change-patch-missing"
      in
      Loom_change.Apply_patch patch
  | ("Write" | "Edit"), _, _ -> failf "change-single-path-required"
  | _ -> failf "change-tool-refused:%s" tool_name

let change_call_id event =
  [ "tool_use_id"; "tool_call_id"; "toolUseId" ]
  |> List.find_map (fun name ->
         match object_field event name with
         | Some (Json_string value) when value <> "" -> Some value
         | _ -> None)
  |> function
  | Some value when String.length value <= 256 -> value
  | Some _ -> failf "change-tool-call-id-too-long"
  | None -> failf "change-tool-call-id-missing"

let rewrite_patch_for_stage root stage_root patch =
  String.split_on_char '\n' patch
  |> List.map (fun line ->
         let prefixes = [ "*** Add File: "; "*** Update File: "; "*** Delete File: " ] in
         match
           List.find_map
             (fun prefix ->
               if starts_with line prefix then Some prefix else None)
             prefixes
         with
         | None -> line
         | Some prefix ->
             let raw =
               String.sub line (String.length prefix)
                 (String.length line - String.length prefix)
             in
             let relative = Loom_change.normalize_declared_path root raw in
             prefix ^ Filename.concat stage_root relative)
  |> String.concat "\n"

let change_hook_output root input mutation stage_root =
  let updated_input =
    match mutation, input with
    | (Loom_change.Write { path; _ } | Loom_change.Edit { path; _ }),
      Json_object _ ->
        replace_object_field input "file_path"
          (Json_string (Filename.concat stage_root path))
    | Loom_change.Apply_patch patch, Json_string _ ->
        Json_string (rewrite_patch_for_stage root stage_root patch)
    | Loom_change.Apply_patch patch, Json_object _ ->
        let field =
          match object_field input "patch", object_field input "input" with
          | Some _, _ -> "patch"
          | None, Some _ -> "input"
          | None, None -> failf "change-patch-missing"
        in
        replace_object_field input field
          (Json_string (rewrite_patch_for_stage root stage_root patch))
    | _ -> failf "change-stage-input-invalid"
  in
  Json_object
    [ ("hookSpecificOutput",
       Json_object
         [ ("hookEventName", Json_string "PreToolUse");
           ("permissionDecision", Json_string "allow");
           ("permissionDecisionReason",
            Json_string "Sounio 9043 admitted a kernel-resident staged change");
           ("updatedInput", updated_input) ]) ]

let provider_hook_output profile output =
  match profile.provider_id with
  | "codex" | "claude" -> output
  | "cursor" | "grok" ->
      let specific =
        match object_field output "hookSpecificOutput" with
        | Some (Json_object _ as value) -> value
        | Some _ -> failf "native-hook-output-specific-must-be-object"
        | None -> failf "native-hook-output-specific-missing"
      in
      let decision = string_field specific "permissionDecision" in
      let reason = string_field specific "permissionDecisionReason" in
      let updated =
        match object_field specific "updatedInput" with
        | Some value -> value
        | None -> failf "native-hook-output-updated-input-missing"
      in
      if decision <> "allow" then
        failf "native-hook-output-decision-unsupported:%s" decision;
      if profile.provider_id = "cursor" then
        Json_object
          [ ("permission", Json_string "allow");
            ("agent_message", Json_string reason);
            ("updated_input", updated) ]
      else
        Json_object
          [ ("decision", Json_string "allow");
            ("reason", Json_string reason);
            ("hookSpecificOutput", specific) ]
  | provider -> failf "native-hook-output-provider-unsupported:%s" provider

let git_commit_message command =
  try
    match Loom_exec.lex_command command with
    | [ executable; "commit"; ("-m" | "--message"); message ]
      when Filename.basename executable = "git" && message <> "" -> Some message
    | words when
        (match words with executable :: "commit" :: _ ->
           Filename.basename executable = "git" | _ -> false) ->
        failf "change-git-commit-form-refused"
    | _ -> None
  with Loom_exec.Dynamic_command reason ->
    if starts_with (String.trim command) "git commit" then
      failf "change-git-commit-dynamic-refused:%s" reason
    else None

let commit_presentation receipt oid path =
  "/usr/bin/printf '%s\\n' " ^
  Loom_exec.shell_quote
    (Printf.sprintf
       "LOOM_CHANGE_COMMITTED receipt_sha256=%s commit=%s receipt_path=%s"
       receipt oid path)

let safe_token ?(limit = 24) value =
  let output = Buffer.create (min limit (String.length value)) in
  String.iter
    (fun character ->
      if Buffer.length output < limit then
        match character with
        | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '.' | '_' | '-' ->
            Buffer.add_char output character
        | _ -> Buffer.add_char output '_')
    value;
  let token = Buffer.contents output in
  if token = "" then "unknown" else token

let git_output cwd arguments = process_ok ~cwd "git" ("-C" :: cwd :: arguments)

let git_root cwd = git_output cwd [ "rev-parse"; "--show-toplevel" ]

let git_common_dir root =
  let value = git_output root [ "rev-parse"; "--path-format=absolute"; "--git-common-dir" ] in
  Unix.realpath value

let normalize_absolute cwd value =
  let raw = if Filename.is_relative value then Filename.concat cwd value else value in
  let parts = String.split_on_char '/' raw in
  let reduced =
    List.fold_left
      (fun stack part ->
        match part, stack with
        | ("" | "."), _ -> stack
        | "..", _ :: tail -> tail
        | "..", [] -> []
        | _, _ -> part :: stack)
      [] parts
    |> List.rev
  in
  "/" ^ String.concat "/" reduced

let rec canonical_missing path suffix =
  if Sys.file_exists path then
    List.fold_left Filename.concat (Unix.realpath path) suffix
  else
    let parent = Filename.dirname path in
    if parent = path then failf "path-has-no-existing-ancestor:%s" path;
    canonical_missing parent (Filename.basename path :: suffix)

let existing_directory path =
  let rec loop candidate =
    if Sys.file_exists candidate then
      if Sys.is_directory candidate then candidate else Filename.dirname candidate
    else
      let parent = Filename.dirname candidate in
      if parent = candidate then failf "path-has-no-existing-directory:%s" path;
      loop parent
  in
  loop path

let relative_to root path =
  let prefix = if root = "/" then "/" else root ^ "/" in
  if path = root then "."
  else if starts_with path prefix then
    String.sub path (String.length prefix) (String.length path - String.length prefix)
  else failf "path-outside-worktree:%s" path

let target_scope cwd session_root paths =
  let session_common = git_common_dir session_root in
  let grouped = Hashtbl.create 2 in
  List.iter
    (fun value ->
      let absolute = canonical_missing (normalize_absolute cwd value) [] in
      let target_root =
        try git_root (existing_directory absolute) |> Unix.realpath
        with Error _ | Unix_error _ ->
          failf "write-path-outside-current-repository:%s" value
      in
      if git_common_dir target_root <> session_common then
        failf "write-path-outside-current-repository:%s" value;
      if target_root <> Unix.realpath session_root then
        failf "write-path-outside-session-worktree:%s" value;
      let relative = relative_to target_root absolute in
      let previous = Option.value ~default:[] (Hashtbl.find_opt grouped target_root) in
      Hashtbl.replace grouped target_root (relative :: previous))
    paths;
  let roots = Hashtbl.to_seq_keys grouped |> List.of_seq in
  match roots with
  | [ root ] -> (root, unique (Hashtbl.find grouped root))
  | _ -> failf "write-paths-span-multiple-worktrees"

let parse_manifest path =
  let table = Hashtbl.create 48 in
  read_file path |> String.split_on_char '\n'
  |> List.iter (fun line ->
         match String.index_opt line '=' with
         | None when line = "" -> ()
         | None -> failf "malformed-freeze-manifest-line"
         | Some index ->
             let key = String.sub line 0 index in
             if Hashtbl.mem table key then failf "duplicate-freeze-field:%s" key;
             Hashtbl.add table key
               (String.sub line (index + 1) (String.length line - index - 1)));
  table

let required table key =
  match Hashtbl.find_opt table key with
  | Some value when value <> "" -> value
  | _ -> failf "missing-freeze-field:%s" key

let digest_u32_of_hex digest =
  if String.length digest <> 64 then failf "invalid-sha256:%s" digest;
  List.init 8 (fun index ->
      let chunk = String.sub digest (index * 8) 8 in
      try Int64.to_string (Int64.of_string ("0x" ^ chunk))
      with _ -> failf "invalid-sha256:%s" digest)
  |> String.concat " "

let digest_u32_field table key =
  let value = required table key in
  let parts = String.split_on_char ',' value in
  if List.length parts <> 8 then failf "invalid-freeze-digest-field:%s" key;
  List.iter
    (fun part ->
      try
        let value = Int64.of_string part in
        if value < 0L || value > 4294967295L then raise Exit
      with _ -> failf "invalid-freeze-digest-field:%s" key)
    parts;
  String.concat " " parts

let authority_runtime root manifest =
  let explicit = Sys.getenv_opt "SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME" in
  let sibling =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-language-authority-runtime"
  in
  let local =
    Filename.concat root
      "tools/loom/.runtime/sounio-loom-language-authority-runtime"
  in
  let selected =
    match explicit with
    | Some path when path <> "" -> path
    | _ when Sys.file_exists sibling -> sibling
    | _ -> local
  in
  if not (Sys.file_exists selected) then failf "Sounio-authority-runtime-missing:%s" selected;
  if sha256_file selected <> required manifest "executable_sha256" then
    failf "Sounio-authority-runtime-hash-mismatch";
  selected

type authority_receipt = {
  sounio_source_sha256 : string;
  semantics_sha256 : string;
  producing_language : string;
  language_role : string;
  semantic_authority_language : string;
  semantic_authority_role : string;
  semantic_authority_origin : string;
  toolchain : string;
  toolchain_sha256 : string;
  hardware : string;
  hardware_sha256 : string;
  command : string;
  command_sha256 : string;
  parent_authority_result : string;
  provider : string;
  dialect : string;
  provider_config_sha256 : string;
  result : string;
}

let operational_receipt raw_event command =
  let executable = Unix.realpath Sys.executable_name in
  let toolchain =
    Printf.sprintf "OCaml %s executable=%s" Sys.ocaml_version executable
  in
  let hardware =
    String.concat ";"
      [ "os_type=" ^ Sys.os_type; "word_size=" ^ string_of_int Sys.word_size;
        "hostname=" ^ Unix.gethostname ();
        "cpuinfo_sha256=" ^
          (if Sys.file_exists "/proc/cpuinfo" then sha256_file "/proc/cpuinfo"
           else sha256 "unavailable") ]
  in
  { sounio_source_sha256 = "unavailable";
    semantics_sha256 = "unavailable";
    producing_language = "OCaml";
    language_role = "OPERATIONAL_REALIZATION";
    semantic_authority_language = "unverified";
    semantic_authority_role = "unverified";
    semantic_authority_origin = "unverified";
    toolchain;
    toolchain_sha256 = sha256_file executable;
    hardware;
    hardware_sha256 = sha256 hardware;
    command;
    command_sha256 = sha256 raw_event;
    parent_authority_result = "unavailable";
    provider = "unverified";
    dialect = "unverified";
    provider_config_sha256 = "unavailable";
    result = "unavailable" }

let runtime_authority_root () =
  let binary_dir = Filename.dirname (Unix.realpath Sys.executable_name) in
  Filename.concat (Filename.dirname binary_dir) "policy/language-authority"

let authority_policy_root worktree_root =
  let local_manifest =
    Filename.concat worktree_root "tools/loom/language_authority.freeze.v1"
  in
  let selected =
    match Sys.getenv_opt "SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT" with
    | Some path when path <> "" -> path
    | _ when Sys.file_exists local_manifest -> worktree_root
    | _ -> runtime_authority_root ()
  in
  let selected = Unix.realpath selected in
  let runtime_root = runtime_authority_root () in
  let origin =
    if selected = Unix.realpath worktree_root then "worktree"
    else if Sys.file_exists runtime_root && selected = Unix.realpath runtime_root then
      "runtime-capsule"
    else "explicit-root"
  in
  (selected, origin)

let authorize_guard root _raw_event base_receipt =
  let policy_root, policy_origin = authority_policy_root root in
  let manifest_path =
    match Sys.getenv_opt "SOUNIO_LOOM_LANGUAGE_AUTHORITY_MANIFEST" with
    | Some path when path <> "" -> path
    | _ -> Filename.concat policy_root "tools/loom/language_authority.freeze.v1"
  in
  if not (Sys.file_exists manifest_path) then failf "Sounio-authority-policy-missing";
  if sha256_file manifest_path <> pinned_manifest_sha256 then
    failf "Sounio-authority-policy-hash-mismatch";
  let manifest = parse_manifest manifest_path in
  if required manifest "stage" <> "SEMANTICS_FROZEN"
     || required manifest "producing_language" <> "Sounio"
     || required manifest "language_role" <> "SEMANTIC_AUTHORITY"
     || required manifest "parity_open" <> "false"
     || required manifest "claim_ready" <> "false"
  then failf "Sounio-authority-policy-state-invalid";
  let source_path = Filename.concat policy_root (required manifest "source_path") in
  let entrypoint_path = Filename.concat policy_root (required manifest "entrypoint_path") in
  if sha256_file source_path <> required manifest "source_sha256" then
    failf "Sounio-authority-source-hash-mismatch";
  if sha256_file entrypoint_path <> required manifest "entrypoint_sha256" then
    failf "Sounio-authority-entrypoint-hash-mismatch";
  if sha256 (read_file source_path ^ read_file entrypoint_path)
     <> required manifest "semantics_sha256"
  then failf "Sounio-authority-semantics-hash-mismatch";
  let runtime = authority_runtime policy_root manifest in
  let source = digest_u32_field manifest "source_sha256_u32" in
  let semantics = digest_u32_field manifest "semantics_sha256_u32" in
  let toolchain = digest_u32_of_hex base_receipt.toolchain_sha256 in
  let hardware = digest_u32_of_hex base_receipt.hardware_sha256 in
  let command = digest_u32_of_hex base_receipt.command_sha256 in
  let zero = "0 0 0 0 0 0 0 0" in
  let frame =
    String.concat " "
      [ "9020 3 6 9 8 1 0 0 0 0 0 0 0 0 0 1 1 1";
        source; semantics; semantics; toolchain; hardware; command; zero; zero ]
    ^ "\n"
  in
  let result = run_process ~input:frame ~cwd:policy_root runtime [] in
  let decision = trim result.output in
  if result.code <> 0
     || not (starts_with decision "SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 ")
     || not (ends_with decision "next_stage=SEMANTICS_FROZEN")
  then failf "Sounio-authority-denied:rc=%d:%s" result.code decision;
  { base_receipt with
    sounio_source_sha256 = required manifest "source_sha256";
    semantics_sha256 = required manifest "semantics_sha256";
    semantic_authority_language = required manifest "producing_language";
    semantic_authority_role = required manifest "language_role";
    semantic_authority_origin = policy_origin;
    result = decision }

let runtime_native_hook_cutover_root () =
  let binary_dir = Filename.dirname (Unix.realpath Sys.executable_name) in
  Filename.concat (Filename.dirname binary_dir) "policy/native-hook-cutover"

let native_hook_cutover_policy_root worktree_root =
  let local_manifest =
    Filename.concat worktree_root "tools/loom/native_hook_cutover.freeze.v1"
  in
  let selected =
    match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_ROOT" with
    | Some path when path <> "" -> path
    | _ when Sys.file_exists local_manifest -> worktree_root
    | _ -> runtime_native_hook_cutover_root ()
  in
  let selected = Unix.realpath selected in
  let runtime_root = runtime_native_hook_cutover_root () in
  let origin =
    if selected = Unix.realpath worktree_root then "worktree"
    else if Sys.file_exists runtime_root && selected = Unix.realpath runtime_root then
      "runtime-capsule"
    else "explicit-root"
  in
  (selected, origin)

let native_hook_cutover_runtime root manifest =
  let explicit = Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_RUNTIME" in
  let sibling =
    Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
      "sounio-loom-native-hook-cutover"
  in
  let local =
    Filename.concat root "tools/loom/.runtime/sounio-loom-native-hook-cutover"
  in
  let selected =
    match explicit with
    | Some path when path <> "" -> path
    | _ when Sys.file_exists sibling -> sibling
    | _ -> local
  in
  if not (Sys.file_exists selected) then
    failf "Sounio-native-hook-cutover-runtime-missing:%s" selected;
  if sha256_file selected <> required manifest "executable_sha256" then
    failf "Sounio-native-hook-cutover-runtime-hash-mismatch";
  selected

let provider_config_relative profile =
  match profile.provider_id with
  | "codex" -> ".codex/hooks.json"
  | "claude" -> ".claude/settings.json"
  | "cursor" -> ".cursor/hooks.json"
  | "grok" -> ".grok/hooks/loom-native.json"
  | value -> failf "unsupported-hook-provider-config:%s" value

let native_hook_config_path root policy_root profile =
  match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_CONFIG" with
  | Some path when path <> "" && test_mode () -> path
  | Some _ when not (test_mode ()) -> failf "hook-config-override-requires-test-mode"
  | _ ->
      let worktree = Filename.concat root (provider_config_relative profile) in
      if Sys.file_exists worktree then worktree
      else Filename.concat (Filename.concat policy_root "configs")
          (profile.provider_id ^ ".json")

let validate_native_hook_config profile path =
  if not (Sys.file_exists path) then failf "native-hook-provider-config-missing:%s" path;
  let content = read_file path in
  let lowered = String.lowercase_ascii content in
  List.iter
    (fun prohibited ->
      if contains lowered prohibited then
        failf "native-hook-provider-config-prohibited-bridge:%s" prohibited)
    [ "python"; "pypy"; "rustc"; "cargo"; "node "; "ruby "; "awk "; "bc " ];
  if not (contains content "exec env SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT=")
     || not (contains content "bin/sounio-loom-runtime")
     || not (contains content ("agent-hook --agent " ^ profile.provider_id))
  then failf "native-hook-provider-config-not-direct:%s" profile.provider_id;
  sha256 content

let digest_u60 digest offset =
  if String.length digest <> 64 || offset < 0 || offset + 15 > 64 then
    failf "invalid-sha256:%s" digest;
  try Int64.of_string ("0x" ^ String.sub digest offset 15) |> Int64.to_string
  with _ -> failf "invalid-sha256:%s" digest

let authorize_native_hook_cutover root profile event _raw_event base_receipt =
  let policy_root, policy_origin = native_hook_cutover_policy_root root in
  let manifest_path =
    match Sys.getenv_opt "SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_MANIFEST" with
    | Some path when path <> "" -> path
    | _ -> Filename.concat policy_root "tools/loom/native_hook_cutover.freeze.v1"
  in
  if not (Sys.file_exists manifest_path) then
    failf "Sounio-native-hook-cutover-policy-missing";
  if sha256_file manifest_path <> pinned_native_hook_cutover_manifest_sha256 then
    failf "Sounio-native-hook-cutover-policy-hash-mismatch";
  let manifest = parse_manifest manifest_path in
  if required manifest "stage" <> "SEMANTICS_FROZEN"
     || required manifest "producing_language" <> "Sounio"
     || required manifest "language_role" <> "SEMANTIC_AUTHORITY"
     || required manifest "action" <> "9045"
     || required manifest "parity_open" <> "false"
     || required manifest "claim_ready" <> "false"
  then failf "Sounio-native-hook-cutover-policy-state-invalid";
  let source_path = Filename.concat policy_root (required manifest "source_path") in
  let entrypoint_path = Filename.concat policy_root (required manifest "entrypoint_path") in
  if sha256_file source_path <> required manifest "source_sha256" then
    failf "Sounio-native-hook-cutover-source-hash-mismatch";
  if sha256_file entrypoint_path <> required manifest "entrypoint_sha256" then
    failf "Sounio-native-hook-cutover-entrypoint-hash-mismatch";
  if sha256 (read_file source_path ^ read_file entrypoint_path)
     <> required manifest "semantics_sha256"
  then failf "Sounio-native-hook-cutover-semantics-hash-mismatch";
  let runtime = native_hook_cutover_runtime policy_root manifest in
  let config_path = native_hook_config_path root policy_root profile in
  let config_sha256 = validate_native_hook_config profile config_path in
  let event_name = string_field event "hook_event_name" in
  let event_code = hook_event_code event_name in
  let word = if event_code = 3 then 8388607 else 8359935 in
  let semantics = required manifest "semantics_sha256" in
  let frame =
    Printf.sprintf "9045 1 3 %d %d %d %d 0 %s %s %s %s 4 4\n"
      profile.provider_code profile.dialect_code event_code word
      (digest_u60 semantics 0) (digest_u60 semantics 15)
      (digest_u60 base_receipt.toolchain_sha256 0)
      (digest_u60 config_sha256 0)
  in
  let result = run_process ~input:frame ~cwd:policy_root runtime [] in
  let decision = trim result.output in
  if result.code <> 0
     || decision <>
        "SOUNIO_NATIVE_HOOK_CUTOVER HOOK_EVENT_ADMIT semantic_authority=Sounio action=9045"
  then failf "Sounio-native-hook-cutover-denied:rc=%d:%s" result.code decision;
  { base_receipt with
    sounio_source_sha256 = required manifest "source_sha256";
    semantics_sha256 = semantics;
    semantic_authority_language = required manifest "producing_language";
    semantic_authority_role = required manifest "language_role";
    semantic_authority_origin = policy_origin;
    parent_authority_result = base_receipt.result;
    provider = profile.provider_id;
    dialect = profile.dialect_name;
    provider_config_sha256 = config_sha256;
    result = decision }

let utc_now () =
  let tm = Unix.gmtime (Unix.gettimeofday ()) in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02dZ"
    (tm.tm_year + 1900) (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min tm.tm_sec

let log_escape value =
  String.map (function '\t' | '\n' | '\r' -> ' ' | character -> character) value

let append_decision_log root decision reason agent lane event receipt =
  let path =
    match Sys.getenv_opt "SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG" with
    | Some value when value <> "" && test_mode () -> value
    | Some value when value <> "" -> failf "decision-log-override-requires-test-mode"
    | _ ->
        Filename.concat
          (Filename.concat (git_common_dir root) "sounio-loom-language-authority")
          "agent-hook.tsv"
  in
  mkdir_p (Filename.dirname path);
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      let line =
        String.concat "\t"
          [ "schema=loom-agent-hook-receipt-v1";
            "utc=" ^ utc_now ();
            "decision=" ^ decision;
            "reason=" ^ log_escape reason;
            "agent=" ^ log_escape agent;
            "lane=" ^ log_escape lane;
            "event=" ^ log_escape event;
            "sounio_source_sha256=" ^ receipt.sounio_source_sha256;
            "semantics_sha256=" ^ receipt.semantics_sha256;
            "producing_language=" ^ receipt.producing_language;
            "language_role=" ^ receipt.language_role;
            "semantic_authority_language=" ^ receipt.semantic_authority_language;
            "semantic_authority_role=" ^ receipt.semantic_authority_role;
            "semantic_authority_origin=" ^ receipt.semantic_authority_origin;
            "toolchain=" ^ log_escape receipt.toolchain;
            "toolchain_sha256=" ^ receipt.toolchain_sha256;
            "hardware=" ^ log_escape receipt.hardware;
            "hardware_sha256=" ^ receipt.hardware_sha256;
            "command=" ^ log_escape receipt.command;
            "command_sha256=" ^ receipt.command_sha256;
            "parent_authority_result=" ^ log_escape receipt.parent_authority_result;
            "provider=" ^ receipt.provider;
            "dialect=" ^ receipt.dialect;
            "provider_config_sha256=" ^ receipt.provider_config_sha256;
            "result=" ^ log_escape receipt.result ] ^ "\n"
      in
      write_all descriptor line;
      Unix.fsync descriptor;
      Unix.lockf descriptor F_ULOCK 0)

let coordination_environment () =
  let ttl = Option.value ~default:"1800" (Sys.getenv_opt "SOUNIO_COORD_HOOK_TTL_SECONDS") in
  Unix.environment ()
  |> drop_environment_prefix "SOUNIO_AGENTD_"
  |> replace_environment "SOUNIO_COORD_TTL_SECONDS" ttl

let run_coord root worktree arguments =
  run_process ~environment:(coordination_environment ())
    ~timeout_seconds:coordination_process_timeout_seconds ~cwd:worktree
    (Filename.concat root "bin/sounio-coord") arguments

let coord_ok root worktree arguments =
  let result = run_coord root worktree arguments in
  if result.code <> 0 then
    failf "coordination-failed:rc=%d:%s" result.code (trim result.output);
  trim result.output

let scope_arguments agent lane intent =
  [ "--agent"; agent; "--lane"; lane; "--intent"; intent ]

let harness_of_agent agent =
  if starts_with agent "claude" then "claude"
  else if starts_with agent "codex" then "codex"
  else if starts_with agent "cursor" then "cursor"
  else if starts_with agent "grok" then "grok"
  else failf "unsupported-hook-agent:%s" agent

let process_identity () =
  let pid = Unix.getppid () in
  let stat = read_file (Printf.sprintf "/proc/%d/stat" pid) in
  let closing =
    match String.rindex_opt stat ')' with
    | Some index -> index
    | None -> failf "invalid-parent-process-stat"
  in
  let tail =
    String.sub stat (closing + 2) (String.length stat - closing - 2)
    |> String.split_on_char ' ' |> List.filter (( <> ) "")
  in
  let pid_start =
    match List.nth_opt tail 19 with Some value -> value | None -> failf "parent-start-missing"
  in
  (pid, pid_start, trim (read_file "/proc/sys/kernel/random/boot_id"),
   Unix.readlink "/proc/self/ns/pid", Unix.gethostname ())

let parent_process () =
  try
    let pid = Unix.getppid () in
    let executable = Unix.realpath (Unix.readlink (Printf.sprintf "/proc/%d/exe" pid)) in
    let arguments =
      read_file (Printf.sprintf "/proc/%d/cmdline" pid)
      |> String.split_on_char '\000'
      |> List.filter (( <> ) "")
    in
    Some (executable, arguments)
  with _ -> None

let exact_cursor_parent executable arguments =
  match arguments with
  | launcher :: rest ->
      Filename.basename executable = "node"
      && Filename.basename launcher = "cursor-agent"
      && List.exists
           (fun argument ->
             Filename.basename argument = "index.js"
             && contains argument "/cursor-agent/versions/")
           rest
  | [] -> false

let exact_grok_parent executable arguments =
  let command = Filename.basename executable in
  match arguments with
  | launcher :: _ ->
      Filename.basename launcher = "grok"
      && (command = "grok"
          || (starts_with command "grok-" && ends_with command "-linux-x86_64"))
  | [] -> false

let observed_compatibility_provider () =
  match parent_process () with
  | Some (executable, arguments) when exact_cursor_parent executable arguments ->
      Some "cursor"
  | Some (executable, arguments) when exact_grok_parent executable arguments ->
      Some "grok"
  | _ -> None

let route_compatibility_agent requested =
  match observed_compatibility_provider () with
  | Some observed when not (starts_with requested observed) ->
      (observed, "verified-" ^ observed ^ "-provider-compat")
  | _ -> (requested, "direct")

let exact_environment name expected =
  Sys.getenv_opt name = Some expected

let agentd_process_worktree root agent lane raw_session_id =
  if not
       (exact_environment "SOUNIO_AGENTD_AGENT" agent
        && exact_environment "SOUNIO_AGENTD_LANE" lane
        && exact_environment "SOUNIO_AGENTD_SESSION_ID" raw_session_id)
  then None
  else
    match Sys.getenv_opt "SOUNIO_AGENTD_WORKTREE" with
    | Some path when path <> "" ->
        (try
           let process_root = git_root path |> Unix.realpath in
           if git_common_dir process_root = git_common_dir root then
             Some process_root
           else None
         with _ -> None)
    | _ -> None

let agentd_identity_matches root agent lane raw_session_id =
  Option.is_some (agentd_process_worktree root agent lane raw_session_id)

let process_worktree root agent lane raw_session_id =
  Option.value ~default:root
    (agentd_process_worktree root agent lane raw_session_id)

let refresh_presence tool_root process_root claim_root agent lane raw_session_id =
  let harness = harness_of_agent agent in
  let pid, pid_start, boot_id, pid_namespace, host = process_identity () in
  let ttl = Option.value ~default:"1800" (Sys.getenv_opt "SOUNIO_COORD_HOOK_TTL_SECONDS") in
  let arguments =
    [ "presence-register"; "--agent"; agent; "--lane"; lane; "--harness"; harness;
      "--session-id"; raw_session_id; "--pid"; string_of_int pid; "--pid-start";
      pid_start; "--boot-id"; boot_id; "--pid-namespace"; pid_namespace;
      "--host"; host; "--ttl-seconds"; ttl ]
  in
  let result = run_coord tool_root process_root arguments in
  let result =
    if result.code <> 0 && String.contains result.output ':'
       && String.split_on_char '\n' result.output
          |> List.exists (fun line -> starts_with (trim line) "error: claim not found:")
    then (
      ignore
        (coord_ok tool_root claim_root
           ("scope" :: scope_arguments agent lane ("active " ^ agent ^ " session")));
      run_coord tool_root process_root arguments)
    else result
  in
  if result.code <> 0 then
    failf "process-presence-refused:%s" (trim result.output)

let refresh_hook_capability tool_root process_root agent lane raw_session_id =
  ignore
    (coord_ok tool_root process_root
       [ "hook-capability-register"; "--agent"; agent; "--lane"; lane;
         "--session-id"; raw_session_id ])

let tmux_endpoint root =
  match Sys.getenv_opt "TMUX", Sys.getenv_opt "TMUX_PANE" with
  | Some tmux, Some pane when tmux <> "" && pane <> "" ->
      let socket =
        match String.split_on_char ',' tmux with
        | value :: _ -> value
        | [] -> ""
      in
      if socket = "" then None
      else
        let result =
          run_process ~cwd:root "tmux"
            [ "-S"; socket; "display-message"; "-p"; "-t"; pane;
              "#{pane_id}|#{pane_current_path}" ]
        in
        if result.code <> 0 then None
        else
          (match String.split_on_char '|' (trim result.output) with
          | [ pane_id; pane_cwd ] when pane_id <> "" && pane_cwd <> "" ->
              (try
                 let pane_root = git_root pane_cwd |> Unix.realpath in
                 if pane_root = Unix.realpath root then Some (socket, pane_id)
                 else None
               with _ -> None)
          | _ -> None)
  | _ -> None

let refresh_endpoint tool_root root agent lane raw_session_id =
  let harness = harness_of_agent agent in
  let ttl = Option.value ~default:"1800" (Sys.getenv_opt "SOUNIO_COORD_HOOK_TTL_SECONDS") in
  match Sys.getenv_opt "SOUNIO_AGENTD_SOCKET", Sys.getenv_opt "SOUNIO_AGENTD_TOKEN_FILE" with
  | Some socket, Some token when socket <> "" && token <> ""
                                  && agentd_identity_matches root agent lane raw_session_id
                                  && Sys.file_exists socket && Sys.file_exists token ->
      ignore
        (coord_ok tool_root root
           [ "endpoint-register"; "--agent"; agent; "--lane"; lane;
             "--harness"; harness; "--transport"; "agentd"; "--address"; socket;
             "--socket"; socket; "--token-file"; token; "--ttl-seconds"; ttl ])
  | _ ->
      (match Sys.getenv_opt "SOUNIO_LOOM_SOCKET",
             Sys.getenv_opt "SOUNIO_LOOM_TOKEN_FILE" with
      | Some socket, Some token when socket <> "" && token <> ""
                                     && Sys.file_exists socket
                                     && Sys.file_exists token ->
          ignore
            (run_coord tool_root root
               [ "endpoint-register"; "--agent"; agent; "--lane"; lane;
                 "--harness"; harness; "--transport"; "loom";
                 "--address"; socket; "--socket"; socket; "--token-file";
                 token; "--ttl-seconds"; ttl ])
      | _ ->
          (match tmux_endpoint root with
          | Some (socket, pane) ->
              ignore
                (coord_ok tool_root root
                   [ "endpoint-register"; "--agent"; agent; "--lane"; lane;
                     "--harness"; harness; "--transport"; "tmux";
                     "--address"; pane; "--socket"; socket; "--ttl-seconds";
                     ttl ])
          | None -> ()))

let message_lines output =
  String.split_on_char '\n' output |> List.filter (fun line -> starts_with line "MESSAGE ")

let message_id line =
  match String.split_on_char ' ' line with
  | "MESSAGE" :: id :: _ when starts_with id "id=" ->
      Some (String.sub id 3 (String.length id - 3))
  | _ -> None

let inject_messages tool_root root agent lane =
  let inbox =
    run_coord tool_root root
      [ "inbox"; "--agent"; agent; "--lane"; lane; "--directed-only";
        "--newest-first"; "--limit"; "12" ]
  in
  if inbox.code <> 0 then failf "coordination-inbox-failed:%s" (trim inbox.output);
  let lines = message_lines inbox.output in
  if lines <> [] then (
    print_endline "Recent directed Sounio lane messages waiting for this agent:";
    List.iter print_endline lines;
    let ids = List.filter_map message_id lines in
    if ids <> [] then
      ignore
        (coord_ok tool_root root
           ([ "injected"; "--agent"; agent; "--lane"; lane; "--messages" ] @ ids));
    Printf.printf
      "After handling one, acknowledge it with bin/sounio-coord ack --agent %s --lane %s --message <id>.\n%!"
      agent lane)

let notify_conflict tool_root root agent lane paths output =
  let tokens = String.split_on_char ' ' output in
  let value prefix =
    List.find_map
      (fun token ->
        if starts_with token prefix then
          Some (String.sub token (String.length prefix)
                  (String.length token - String.length prefix))
        else None)
      tokens
  in
  match value "agent=", value "lane=" with
  | Some owner_agent, Some owner_lane ->
      ignore
        (run_coord tool_root root
           [ "send"; "--agent"; agent; "--lane"; lane; "--to-agent"; owner_agent;
             "--to-lane"; owner_lane; "--kind"; "request"; "--message";
             "Write conflict requested by " ^ agent ^ "/" ^ lane ^ ": " ^
             String.concat ", " paths ])
  | _ -> ()

let execute_event tool_root root event agent lane raw_session_id
    file_capability_fixture event_sha256 =
  let event_name = string_field event "hook_event_name" in
  if event_name = "" then failf "hook-event-name-missing";
  let intent = "active " ^ agent ^ " session" in
  let common = scope_arguments agent lane intent in
  let presence_root = process_worktree root agent lane raw_session_id in
  let coordination_enabled =
    Sys.getenv_opt "SOUNIO_LOOM_COORD_AUTO" <> Some "0" || not (test_mode ())
  in
  let obligation_supervisor_enabled =
    Sys.getenv_opt "SOUNIO_COORD_DURABLE_OBLIGATIONS" <> Some "0"
    || not (test_mode ())
  in
  if event_name = "SessionEnd" && not coordination_enabled then None
  else if event_name = "SessionEnd" then (
    refresh_presence tool_root presence_root root agent lane raw_session_id;
    ignore
      (coord_ok tool_root presence_root
         [ "hook-capability-unregister"; "--agent"; agent; "--lane"; lane;
           "--session-id"; raw_session_id ]);
    ignore
      (run_coord tool_root presence_root
         [ "endpoint-unregister"; "--agent"; agent; "--lane"; lane ]);
    ignore
      (run_coord tool_root presence_root
         [ "presence-unregister"; "--agent"; agent; "--lane"; lane ]);
    ignore
      (coord_ok tool_root root
         [ "release"; "--agent"; agent; "--lane"; lane; "--reason";
           "agent session ended" ]);
    None)
  else if event_name = "PreToolUse" then (
    let paths = extract_paths event in
    let tool_name = string_field event "tool_name" in
    let change_target = ref None in
    let staged_change_output = ref None in
    if List.mem tool_name [ "apply_patch"; "Edit"; "Write"; "MultiEdit";
                            "NotebookEdit" ] && paths = []
    then failf "write-path-missing";
    if paths <> [] then (
      let target_root, target_paths =
        target_scope (string_field ~default:root event "cwd") root paths
      in
      change_target := Some (target_root, target_paths);
      if coordination_enabled then (
        refresh_presence tool_root presence_root root agent lane raw_session_id;
        refresh_hook_capability tool_root presence_root agent lane raw_session_id;
        let authorization =
          run_coord tool_root target_root
            ([ "authorize"; "--agent"; agent; "--files" ] @ target_paths)
        in
        if authorization.code <> 0 then (
          let scoped =
            if target_root = root then
              run_coord tool_root root
                ([ "scope" ] @ common @ [ "--files" ] @ target_paths)
            else authorization
          in
          if scoped.code <> 0 then (
            notify_conflict tool_root root agent lane target_paths scoped.output;
            failf "coordination-write-refused:%s" (trim scoped.output)));
        refresh_endpoint tool_root presence_root agent lane raw_session_id));
    if Loom_change.required_mode () && change_tool tool_name then (
      match !change_target with
      | Some (_, target_paths) ->
          let mutation = change_mutation event target_paths in
          let prepared =
            Loom_change.prepare_remote ~session_id:raw_session_id
              ~call_id:(change_call_id event) ~event_sha256 mutation target_paths
          in
          let input =
            match object_field event "tool_input" with
            | Some value -> value
            | None -> failf "change-tool-input-missing"
          in
          staged_change_output :=
            Some
              (change_hook_output root input mutation
                 prepared.Loom_change.remote_stage_root)
      | None -> failf "change-target-missing");
    if execution_tool tool_name then (
      let input =
        match object_field event "tool_input" with
        | Some value -> value
        | None -> failf "execution-tool-input-missing"
      in
      let field, command = execution_command input in
      let cwd = execution_cwd event input root in
      match git_commit_message command with
      | Some message when Loom_change.required_mode () ->
          let receipt, oid, receipt_path =
            Loom_change.commit_remote ~session_id:raw_session_id
              ~call_id:(change_call_id event) ~event_sha256 ~message
          in
          Some
            (execution_hook_output
               ~reason:"Sounio 9044 admitted a byte-exact kernel Git commit"
               input field (commit_presentation receipt oid receipt_path))
      | _ ->
      let ingress =
        Loom_exec_ingress.observe ~root ~agent ~lane ~session_id:raw_session_id
          ~cwd ~event_sha256 ~command ~command_sha256:(sha256 command)
      in
      (match ingress with
      | Some
          { Loom_exec_ingress.result =
              Some (Loom_exec_ingress.Frozen_result result); _ } ->
          Some
            (execution_hook_output
               ~reason:"Sounio 9033 returned a read-only ExecCell result"
               input field (Loom_exec_result.presentation_command result))
      | Some
          { Loom_exec_ingress.result =
              Some (Loom_exec_ingress.Operation_record result); _ } ->
          Some
            (execution_hook_output
               ~reason:"Sounio 9036 returned a verified ExecCell operation record"
               input field
               (Loom_exec_result_record.presentation_command result))
      | _ when Loom_exec_ingress.probe_only () -> None
      | _ ->
          if coordination_enabled then (
            refresh_presence tool_root presence_root root agent lane raw_session_id;
            refresh_hook_capability tool_root presence_root agent lane raw_session_id;
            refresh_endpoint tool_root presence_root agent lane raw_session_id);
          let replacement =
            if Loom_sovereign_exec.required_mode () then (
              if file_capability_fixture then
                failf "file-capability-fixture-forbidden-in-sovereign-mode";
              let prepared =
                Loom_sovereign_exec.prepare ~root ~cwd ~event_sha256 ~command
              in
              Loom_sovereign_exec.start ~event_sha256 prepared
              |> Loom_sovereign_exec.presentation_command)
            else
              Loom_exec.authorize_and_issue ~file_capability_fixture ~root ~cwd
                ~command
          in
          Some (execution_hook_output input field replacement)))
    else !staged_change_output)
  else (
    if event_name = "PostToolUse" && Loom_change.required_mode () then (
      let tool_name = string_field event "tool_name" in
      if change_tool tool_name then (
        ignore
          (Loom_change.consume_remote ~session_id:raw_session_id
             ~call_id:(change_call_id event) ~event_sha256)));
    if not coordination_enabled then None
    else (
      let claim =
        if event_name = "SessionStart" then run_coord tool_root root ([ "scope" ] @ common)
        else
          let heartbeat =
            run_coord tool_root root [ "heartbeat"; "--agent"; agent; "--lane"; lane ]
          in
          if heartbeat.code = 0 then heartbeat
          else run_coord tool_root root ([ "scope" ] @ common)
      in
      if claim.code <> 0 && not (contains claim.output "claim belongs to worktree ")
      then failf "coordination-claim-refused:%s" (trim claim.output);
      refresh_presence tool_root presence_root root agent lane raw_session_id;
      refresh_hook_capability tool_root presence_root agent lane raw_session_id;
      refresh_endpoint tool_root presence_root agent lane raw_session_id;
      if event_name = "SessionStart" then (
        if obligation_supervisor_enabled then
          ignore
            (coord_ok tool_root root
               [ "obligation-supervisor-ensure"; "--interval-seconds"; "1" ]);
        Printf.printf
          "Sounio coordination joined: agent=%s lane=%s. Use this same agent/lane with `bin/sounio-coord scope` before write-bearing Bash commands.\n%!"
          agent lane);
      if event_name = "UserPromptSubmit" || event_name = "PostToolUse" then
        inject_messages tool_root root agent lane;
      None))

let parse_agent arguments =
  let loop = function
    | [ "--agent"; value ] when value <> "" -> (safe_token value, false)
    | [ "--agent"; value; "--test-file-capability-fixture" ] when value <> "" ->
        if not (test_mode ()) then failf "file-capability-fixture-requires-test-mode";
        (safe_token value, true)
    | _ ->
        failf
          "usage: agent-hook --agent codex|claude|cursor|grok [--test-file-capability-fixture]"
  in
  loop arguments

let run arguments =
  let root = ref None in
  let agent = ref "unknown" in
  let lane = ref "session-unknown" in
  let event_name = ref "unknown" in
  let receipt = ref None in
  try
    let parsed_agent, file_capability_fixture = parse_agent arguments in
    agent := parsed_agent;
    let raw_event = read_stdin () in
    if raw_event = "" then failf "hook-event-empty";
    (try root := Some (git_root (Unix.getcwd ()) |> Unix.realpath) with _ -> ());
    receipt := Some (operational_receipt raw_event "sounio-loom agent-hook event=unparsed");
    let parsed_event = parse_json raw_event in
    let effective_agent, provider_route =
      route_compatibility_agent parsed_agent
    in
    agent := effective_agent;
    let profile = hook_profile !agent in
    let event = normalize_hook_event profile parsed_event in
    let cwd = string_field ~default:(Unix.getcwd ()) event "cwd" in
    let current_root = git_root cwd |> Unix.realpath in
    let tool_root =
      match Sys.getenv_opt "SOUNIO_LOOM_TOOL_ROOT" with
      | Some value when value <> "" ->
          let selected = Unix.realpath value in
          if not (Sys.file_exists (Filename.concat selected "bin/sounio-coord")) then
            failf "configured-tool-root-missing-coordination-launcher";
          selected
      | _ -> current_root
    in
    root := Some current_root;
    let raw_session_id = string_field ~default:"unknown" event "session_id" in
    lane := "session-" ^ safe_token raw_session_id;
    event_name := string_field ~default:"unknown" event "hook_event_name";
    let tool_name = string_field ~default:"none" event "tool_name" in
    let command =
      let base =
        Printf.sprintf "sounio-loom agent-hook --agent %s event=%s tool=%s"
          !agent !event_name tool_name
      in
      if provider_route = "direct" then base
      else
        Printf.sprintf "%s requested_provider=%s provider_route=%s" base
          parsed_agent provider_route
    in
    let base_receipt = operational_receipt raw_event command in
    receipt := Some base_receipt;
    let parent_receipt = authorize_guard current_root raw_event base_receipt in
    receipt := Some parent_receipt;
    let authorized_receipt =
      authorize_native_hook_cutover current_root profile event raw_event parent_receipt
    in
    receipt := Some authorized_receipt;
    let hook_output =
      execute_event tool_root current_root event !agent !lane raw_session_id
        file_capability_fixture (sha256 raw_event)
      |> Option.map (provider_hook_output profile)
    in
    append_decision_log current_root "ALLOW" authorized_receipt.result !agent !lane
      !event_name authorized_receipt;
    (match hook_output with Some output -> print_endline (json_string output) | None -> ());
    0
  with
  | Error message
  | Loom_exec.Error message
  | Loom_sovereign_exec.Error message
  | Loom_change.Error message
  | Loom_exec_intent.Error message
  | Loom_exec_catalog.Error message
  | Loom_exec_result.Error message
  | Loom_exec_result_record.Error message
  | Loom_exec_ingress.Error message
  | Loom_membrane.Error message
  | Sys_error message ->
      (match !root, !receipt with
      | Some current_root, Some current_receipt ->
          (try
             append_decision_log current_root "DENY" message !agent !lane !event_name
               current_receipt
           with _ -> ())
      | _ -> ());
      Printf.eprintf "sounio native hook refused: %s\n%!" message;
      2
  | Unix_error (error, function_name, argument) ->
      let message =
        Printf.sprintf "%s:%s(%s)" (Unix.error_message error) function_name argument
      in
      (match !root, !receipt with
      | Some current_root, Some current_receipt ->
          (try
             append_decision_log current_root "DENY" message !agent !lane !event_name
               current_receipt
           with _ -> ())
      | _ -> ());
      Printf.eprintf "sounio native hook refused: %s\n%!" message;
      2
