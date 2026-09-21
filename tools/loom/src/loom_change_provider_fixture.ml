open Unix

exception Fixture_error of string

let failf format = Printf.ksprintf (fun value -> raise (Fixture_error value)) format

let contains value needle =
  let rec search index =
    if String.length needle = 0 then true
    else if index + String.length needle > String.length value then false
    else if String.sub value index (String.length needle) = needle then true
    else search (index + 1)
  in
  search 0

let json_quote value =
  let output = Buffer.create (String.length value + 8) in
  Buffer.add_char output '"';
  String.iter
    (function
      | '"' -> Buffer.add_string output "\\\""
      | '\\' -> Buffer.add_string output "\\\\"
      | '\n' -> Buffer.add_string output "\\n"
      | '\r' -> Buffer.add_string output "\\r"
      | '\t' -> Buffer.add_string output "\\t"
      | character when Char.code character < 32 ->
          Buffer.add_string output (Printf.sprintf "\\u%04x" (Char.code character))
      | character -> Buffer.add_char output character)
    value;
  Buffer.add_char output '"';
  Buffer.contents output

let write_all descriptor value =
  let rec loop offset =
    if offset < String.length value then
      let count =
        Unix.write_substring descriptor value offset (String.length value - offset)
      in
      if count = 0 then failf "short write" else loop (offset + count)
  in
  loop 0

let read_all descriptor =
  let output = Buffer.create 4096 in
  let bytes = Bytes.create 4096 in
  let rec loop () =
    match Unix.read descriptor bytes 0 (Bytes.length bytes) with
    | 0 -> Buffer.contents output
    | count -> Buffer.add_subbytes output bytes 0 count; loop ()
    | exception Unix_error (EINTR, _, _) -> loop ()
  in
  loop ()

let exit_code = function
  | WEXITED code -> code
  | WSIGNALED signal | WSTOPPED signal -> 128 + signal

let run_hook runtime event =
  let input_read, input_write = Unix.pipe ~cloexec:true () in
  let output_read, output_write = Unix.pipe ~cloexec:true () in
  let pid =
    match Unix.fork () with
    | 0 ->
        Unix.dup2 input_read Unix.stdin;
        Unix.dup2 output_write Unix.stdout;
        Unix.dup2 output_write Unix.stderr;
        Unix.close input_write;
        Unix.close output_read;
        Unix.execve runtime [| runtime; "agent-hook"; "--agent"; "codex" |]
          (Unix.environment ())
    | pid -> pid
  in
  Unix.close input_read;
  Unix.close output_write;
  write_all input_write (event ^ "\n");
  Unix.close input_write;
  let output = read_all output_read in
  Unix.close output_read;
  let _, status = Unix.waitpid [] pid in
  (exit_code status, output)

let run_program executable arguments =
  let output_read, output_write = Unix.pipe ~cloexec:true () in
  let pid =
    match Unix.fork () with
    | 0 ->
        Unix.dup2 output_write Unix.stdout;
        Unix.dup2 output_write Unix.stderr;
        Unix.close output_read;
        Unix.execvp executable (Array.of_list (executable :: arguments))
    | pid -> pid
  in
  Unix.close output_write;
  let output = read_all output_read in
  Unix.close output_read;
  let _, status = Unix.waitpid [] pid in
  (exit_code status, output)

let write_file path content =
  let output = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out_noerr output)
    (fun () -> output_string output content; flush output)

let remove path = try Unix.unlink path with Unix_error (ENOENT, _, _) -> ()

let required_environment name =
  match Sys.getenv_opt name with
  | Some value when value <> "" -> value
  | _ -> failf "required environment is missing: %s" name

let event ~name ~session_id ~root ~call_id ~tool_name tool_input =
  Printf.sprintf
    "{\"hook_event_name\":%s,\"session_id\":%s,\"cwd\":%s,\"tool_use_id\":%s,\"tool_name\":%s,\"tool_input\":%s}"
    (json_quote name) (json_quote session_id) (json_quote root)
    (json_quote call_id) (json_quote tool_name) tool_input

let expect_hook runtime label expected event =
  let code, output = run_hook runtime event in
  if code <> expected then
    failf "%s expected rc=%d actual=%d output=%s" label expected code output;
  output

let expect_contains label value expected =
  if not (contains value expected) then
    failf "%s omitted %s: %s" label expected value

let extract_between label value prefix suffix =
  let prefix_index =
    match
      let rec loop index =
        if index + String.length prefix > String.length value then None
        else if String.sub value index (String.length prefix) = prefix then Some index
        else loop (index + 1)
      in
      loop 0
    with
    | Some index -> index + String.length prefix
    | None -> failf "%s omitted stage prefix %s: %s" label prefix value
  in
  let suffix_index =
    match
      let rec loop index =
        if index + String.length suffix > String.length value then None
        else if String.sub value index (String.length suffix) = suffix then Some index
        else loop (index + 1)
      in
      loop prefix_index
    with
    | Some index -> index
    | None -> failf "%s omitted stage suffix %s: %s" label suffix value
  in
  String.sub value prefix_index (suffix_index - prefix_index)

let stage_file label output =
  extract_between label output "\"file_path\":\"" "\""

let stage_patch_file label output =
  extract_between label output "*** Add File: " "\\n"

let expect_root_readonly path =
  try
    write_file path "provider-bypass\n";
    failf "provider direct write unexpectedly succeeded: %s" path
  with
  | Sys_error _ -> ()
  | Unix_error ((EROFS | EACCES | EPERM), _, _) -> ()

let write_input path content =
  Printf.sprintf "{\"file_path\":%s,\"content\":%s}"
    (json_quote path) (json_quote content)

let edit_input path old_string new_string =
  Printf.sprintf
    "{\"file_path\":%s,\"old_string\":%s,\"new_string\":%s,\"replace_all\":false}"
    (json_quote path) (json_quote old_string) (json_quote new_string)

let patch_input patch = Printf.sprintf "{\"patch\":%s}" (json_quote patch)

let bash_input command = Printf.sprintf "{\"command\":%s}" (json_quote command)

let receipt_path output =
  let prefix = "receipt_path=" in
  let start =
    let rec loop index =
      if index + String.length prefix > String.length output then
        failf "commit output omitted receipt path: %s" output
      else if String.sub output index (String.length prefix) = prefix then
        index + String.length prefix
      else loop (index + 1)
    in
    loop 0
  in
  let suffix = ".receipt" in
  let ending =
    let rec loop index =
      if index + String.length suffix > String.length output then
        failf "commit output has malformed receipt path: %s" output
      else if String.sub output index (String.length suffix) = suffix then
        index + String.length suffix
      else loop (index + 1)
    in
    loop start
  in
  String.sub output start (ending - start)

let run runtime root report =
  let session_id = required_environment "SOUNIO_LOOM_SESSION_ID" in
  let base = Printf.sprintf "tools/loom/.change-fixture-%d" (Unix.getpid ()) in
  let relative suffix = base ^ "-" ^ suffix in
  let absolute relative = Filename.concat root relative in
  let targets =
    List.map relative [ "write.txt"; "edit.txt"; "patch.txt"; "wrong.txt";
                        "granted.txt"; "ungranted.txt" ]
  in
  write_file (report ^ ".targets")
    (targets |> List.map absolute |> String.concat "\n" |> fun value -> value ^ "\n");
  Fun.protect
    ~finally:(fun () ->
      if Sys.getenv_opt "SOUNIO_LOOM_MATERIAL_READONLY" <> Some "1" then
        List.iter (fun path -> remove (absolute path)) targets)
    (fun () ->
      let write_path = relative "write.txt" in
      let write_json = write_input write_path "alpha\n" in
      let write_pre =
        event ~name:"PreToolUse" ~session_id ~root ~call_id:"write-1"
          ~tool_name:"Write" write_json
      in
      let write_post =
        event ~name:"PostToolUse" ~session_id ~root ~call_id:"write-1"
          ~tool_name:"Write" write_json
      in
      let write_output = expect_hook runtime "write-prepare" 0 write_pre in
      expect_root_readonly (absolute write_path);
      write_file (stage_file "write-prepare" write_output) "alpha\n";
      ignore (expect_hook runtime "write-consume" 0 write_post);
      let write_check = Unix.openfile (absolute write_path) [ O_RDONLY ] 0 in
      let write_observed =
        Fun.protect ~finally:(fun () -> Unix.close write_check)
          (fun () -> read_all write_check)
      in
      if write_observed <> "alpha\n" then failf "kernel did not materialize Write";
      let replay = expect_hook runtime "write-replay" 2 write_post in
      expect_contains "write-replay" replay "change-grant-missing-or-replayed";

      let edit_path = write_path in
      let edit_json = edit_input edit_path "alpha" "after" in
      let edit_pre =
        event ~name:"PreToolUse" ~session_id ~root ~call_id:"edit-1"
          ~tool_name:"Edit" edit_json
      in
      let edit_post =
        event ~name:"PostToolUse" ~session_id ~root ~call_id:"edit-1"
          ~tool_name:"Edit" edit_json
      in
      let edit_output = expect_hook runtime "edit-prepare" 0 edit_pre in
      write_file (stage_file "edit-prepare" edit_output) "after\n";
      ignore (expect_hook runtime "edit-consume" 0 edit_post);

      let patch_path = relative "patch.txt" in
      let patch =
        Printf.sprintf "*** Begin Patch\n*** Add File: %s\n+probe\n*** End Patch"
          patch_path
      in
      let patch_json = patch_input patch in
      let patch_pre =
        event ~name:"PreToolUse" ~session_id ~root ~call_id:"patch-1"
          ~tool_name:"apply_patch"
          patch_json
      in
      let patch_post =
        event ~name:"PostToolUse" ~session_id ~root ~call_id:"patch-1"
          ~tool_name:"apply_patch"
          patch_json
      in
      let patch_output = expect_hook runtime "patch-prepare" 0 patch_pre in
      write_file (stage_patch_file "patch-prepare" patch_output) "probe\n";
      ignore (expect_hook runtime "patch-consume" 0 patch_post);

      let wrong_path = relative "wrong.txt" in
      let wrong_json = write_input wrong_path "expected\n" in
      let wrong_pre =
        event ~name:"PreToolUse" ~session_id ~root ~call_id:"wrong-1"
          ~tool_name:"Write" wrong_json
      in
      let wrong_post =
        event ~name:"PostToolUse" ~session_id ~root ~call_id:"wrong-1"
          ~tool_name:"Write" wrong_json
      in
      let wrong_output = expect_hook runtime "wrong-prepare" 0 wrong_pre in
      let wrong_stage = stage_file "wrong-prepare" wrong_output in
      write_file wrong_stage "wrong\n";
      let mismatch = expect_hook runtime "wrong-post" 2 wrong_post in
      expect_contains "wrong-post" mismatch "change-staged-post-image-mismatch";
      let burned = expect_hook runtime "wrong-burned" 2 wrong_post in
      expect_contains "wrong-burned" burned "change-grant-missing-or-replayed";

      let granted_path = relative "granted.txt" in
      let ungranted_path = relative "ungranted.txt" in
      let drift_json = write_input granted_path "bound\n" in
      let drift_pre =
        event ~name:"PreToolUse" ~session_id ~root ~call_id:"drift-1"
          ~tool_name:"Write" drift_json
      in
      let drift_post =
        event ~name:"PostToolUse" ~session_id ~root ~call_id:"drift-1"
          ~tool_name:"Write" drift_json
      in
      let drift_output = expect_hook runtime "drift-prepare" 0 drift_pre in
      write_file (stage_file "drift-prepare" drift_output) "bound\n";
      expect_root_readonly (absolute ungranted_path);
      ignore (expect_hook runtime "drift-consume" 0 drift_post);

      let direct_commit_code, _ =
        run_program "git"
          [ "-C"; root; "commit"; "--allow-empty"; "-m"; "provider bypass" ]
      in
      if direct_commit_code = 0 then
        failf "provider direct Git commit unexpectedly succeeded";
      let invalid_commit_event =
        event ~name:"PreToolUse" ~session_id ~root ~call_id:"commit-invalid"
          ~tool_name:"Bash"
          (bash_input "git commit -a -m 'provider widened commit'")
      in
      let invalid_commit =
        expect_hook runtime "commit-invalid-form" 2 invalid_commit_event
      in
      expect_contains "commit-invalid-form" invalid_commit
        "change-git-commit-form-refused";
      let commit_command = "git commit -m 'loom sovereign change fixture'" in
      let commit_event =
        event ~name:"PreToolUse" ~session_id ~root ~call_id:"commit-1"
          ~tool_name:"Bash" (bash_input commit_command)
      in
      let commit_output = expect_hook runtime "commit" 0 commit_event in
      expect_contains "commit" commit_output "LOOM_CHANGE_COMMITTED";
      let receipt = receipt_path commit_output in
      if not (Sys.file_exists receipt) then
        failf "kernel commit receipt is absent: %s" receipt;
      write_file (report ^ ".receipt") (receipt ^ "\n");
      let replay_commit_event =
        event ~name:"PreToolUse" ~session_id ~root ~call_id:"commit-replay"
          ~tool_name:"Bash" (bash_input "git commit -m 'replay'")
      in
      let replay_commit =
        expect_hook runtime "commit-replay" 2 replay_commit_event
      in
      expect_contains "commit-replay" replay_commit
        "change-commit-no-consumed-changes";

      let session_end =
        Printf.sprintf
          "{\"hook_event_name\":\"SessionEnd\",\"session_id\":%s,\"cwd\":%s}"
          (json_quote session_id) (json_quote root)
      in
      ignore (expect_hook runtime "session-end" 0 session_end);
      write_file report
        "loom-change-provider-fixture: PASS language=OCaml role=OPERATIONAL_ATTACHMENT semantic_authority=false provider_root=READ_ONLY write=KERNEL_MATERIALIZED edit=KERNEL_MATERIALIZED apply_patch=KERNEL_MATERIALIZED replay=REFUSED wrong_stage=REFUSED+GRANT_BURNED direct_root_write=EROFS direct_git_commit=REFUSED widened_commit=REFUSED commit_replay=REFUSED kernel_commit=ADMITTED receipt=ISSUED python_executed=false rust_executed=false\n")

let () =
  try
    if Array.length Sys.argv <> 4 then
      failf "usage: loom_change_provider_fixture RUNTIME ROOT REPORT";
    run (Unix.realpath Sys.argv.(1)) (Unix.realpath Sys.argv.(2)) Sys.argv.(3)
  with
  | Fixture_error message
  | Sys_error message ->
      Printf.eprintf "loom-change-provider-fixture: FAIL: %s\n%!" message;
      exit 1
  | Unix_error (error, name, argument) ->
      Printf.eprintf "loom-change-provider-fixture: FAIL: %s:%s(%s)\n%!"
        (Unix.error_message error) name argument;
      exit 1
