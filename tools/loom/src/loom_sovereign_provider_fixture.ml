open Unix

let failf format =
  Printf.ksprintf
    (fun message ->
      Printf.eprintf "loom-sovereign-provider-fixture: %s\n%!" message;
      exit 64)
    format

let write_all descriptor value =
  let bytes = Bytes.of_string value in
  let rec loop offset =
    if offset < Bytes.length bytes then
      let count = Unix.write descriptor bytes offset (Bytes.length bytes - offset) in
      if count = 0 then failf "short-write" else loop (offset + count)
  in
  loop 0

let read_all descriptor =
  let output = Buffer.create 4096 in
  let bytes = Bytes.create 16384 in
  let rec loop () =
    match Unix.read descriptor bytes 0 (Bytes.length bytes) with
    | 0 -> Buffer.contents output
    | count -> Buffer.add_subbytes output bytes 0 count; loop ()
    | exception Unix_error (EINTR, _, _) -> loop ()
  in
  loop ()

let status_code = function
  | WEXITED code -> code
  | WSIGNALED signal | WSTOPPED signal -> 128 + signal

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
      | character -> Buffer.add_char output character)
    value;
  Buffer.add_char output '"';
  Buffer.contents output

let event command =
  let cwd = Unix.getcwd () |> Unix.realpath in
  let session_id =
    match Sys.getenv_opt "SOUNIO_LOOM_SESSION_ID" with
    | Some value when value <> "" -> value
    | _ -> failf "session-id-missing"
  in
  Printf.sprintf
    "{\"hook_event_name\":\"PreToolUse\",\"session_id\":%s,\"cwd\":%s,\"tool_name\":\"Bash\",\"tool_input\":{\"command\":%s}}"
    (json_quote session_id) (json_quote cwd) (json_quote command)

let runtime () =
  match Sys.getenv_opt "SOUNIO_LOOM_PROVIDER_HOOK_RUNTIME" with
  | Some path when not (Filename.is_relative path) -> Unix.realpath path
  | Some _ -> failf "hook-runtime-must-be-absolute"
  | None -> failf "hook-runtime-is-required"

let run_hook_direct runtime event =
  let input_read, input_write = Unix.pipe ~cloexec:true () in
  let output_read, output_write = Unix.pipe ~cloexec:true () in
  let child =
    Unix.create_process runtime
      [| runtime; "agent-hook"; "--agent"; "codex" |]
      input_read output_write Unix.stderr
  in
  Unix.close input_read;
  Unix.close output_write;
  write_all input_write (event ^ "\n");
  Unix.close input_write;
  let output = Fun.protect ~finally:(fun () -> Unix.close output_read)
      (fun () -> read_all output_read)
  in
  let _, status = Unix.waitpid [] child in
  (status_code status, output)

let run_hook_relay runtime event =
  let output_read, output_write = Unix.pipe ~cloexec:true () in
  let relay =
    match Unix.fork () with
    | 0 ->
        Unix.close output_read;
        let code, output = run_hook_direct runtime event in
        write_all output_write output;
        Unix.close output_write;
        Unix._exit code
    | pid -> pid
  in
  Unix.close output_write;
  let output = Fun.protect ~finally:(fun () -> Unix.close output_read)
      (fun () -> read_all output_read)
  in
  let _, status = Unix.waitpid [] relay in
  (status_code status, output)

let find_from value pattern start =
  let pattern_length = String.length pattern in
  let rec loop index =
    if index + pattern_length > String.length value then None
    else if String.sub value index pattern_length = pattern then Some index
    else loop (index + 1)
  in
  loop start

let decode_json_string value start =
  if start >= String.length value || value.[start] <> '"' then
    failf "updated-command-is-not-json-string";
  let output = Buffer.create 256 in
  let rec loop index =
    if index >= String.length value then failf "unterminated-json-string";
    match value.[index] with
    | '"' -> Buffer.contents output
    | '\\' ->
        if index + 1 >= String.length value then failf "truncated-json-escape";
        let character =
          match value.[index + 1] with
          | '"' -> '"' | '\\' -> '\\' | '/' -> '/'
          | 'b' -> '\b' | 'f' -> '\012' | 'n' -> '\n'
          | 'r' -> '\r' | 't' -> '\t'
          | _ -> failf "unsupported-json-escape"
        in
        Buffer.add_char output character;
        loop (index + 2)
    | character -> Buffer.add_char output character; loop (index + 1)
  in
  loop (start + 1)

let replacement output =
  let updated = "\"updatedInput\":{" in
  let command = "\"command\":" in
  let updated_at =
    match find_from output updated 0 with
    | Some index -> index + String.length updated
    | None -> failf "updated-input-missing"
  in
  let command_at =
    match find_from output command updated_at with
    | Some index -> index + String.length command
    | None -> failf "updated-command-missing"
  in
  decode_json_string output command_at

type quote = Unquoted | Single | Double

let shell_words value =
  let words = ref [] in
  let word = Buffer.create 128 in
  let active = ref false in
  let quote = ref Unquoted in
  let finish () =
    if !active then (
      words := Buffer.contents word :: !words;
      Buffer.clear word;
      active := false)
  in
  let rec loop index =
    if index >= String.length value then (
      if !quote <> Unquoted then failf "unterminated-shell-quote";
      finish ();
      List.rev !words)
    else
      match !quote, value.[index] with
      | Unquoted, (' ' | '\t' | '\n') -> finish (); loop (index + 1)
      | Unquoted, '\'' -> active := true; quote := Single; loop (index + 1)
      | Unquoted, '"' -> active := true; quote := Double; loop (index + 1)
      | Single, '\'' -> quote := Unquoted; loop (index + 1)
      | Double, '"' -> quote := Unquoted; loop (index + 1)
      | (Unquoted | Double), '\\' ->
          if index + 1 >= String.length value then failf "truncated-shell-escape";
          active := true;
          Buffer.add_char word value.[index + 1];
          loop (index + 2)
      | _, character ->
          active := true;
          Buffer.add_char word character;
          loop (index + 1)
  in
  loop 0

let run_presenter command =
  let arguments = shell_words command in
  match arguments with
  | [] -> failf "presenter-command-empty"
  | executable :: _ ->
      let argv = Array.of_list arguments in
      let child = Unix.create_process executable argv Unix.stdin Unix.stdout Unix.stderr in
      let _, status = Unix.waitpid [] child in
      status_code status

let require_sovereign_environment () =
  if Sys.getenv_opt "SOUNIO_LOOM_SOVEREIGN_EXEC_REQUIRED" <> Some "1" then
    failf "sovereign-required-mode-absent";
  if Sys.getenv_opt "SOUNIO_LOOM_TOKEN_FILE" <> None then
    failf "legacy-token-leaked-to-harness"

let () =
  require_sovereign_environment ();
  if Array.length Sys.argv <> 3 then
    failf "usage: fixture execute|transport-exit|spoof-start COMMAND";
  let mode = Sys.argv.(1) in
  let command = Sys.argv.(2) in
  let runtime = runtime () in
  let event = event command in
  if mode = "spoof-start" then (
    let code, output = run_hook_relay runtime event in
    if code = 0 || output <> "" then failf "same-uid-spoof-was-not-refused";
    Printf.printf
      "LOOM_SOVEREIGN_FIXTURE mode=spoof-start control_refused=true before_execution=true hook_code=%d\n%!"
      code;
    exit 0);
  let code, output = run_hook_direct runtime event in
  if code <> 0 then failf "hook-refused:%d" code;
  let presenter = replacement output in
  if String.contains presenter '\t' ||
     find_from presenter "--handle" 0 <> None ||
     find_from presenter "capability" 0 <> None
  then failf "bearer-material-exported";
  if mode = "transport-exit" then (
    Printf.printf
      "LOOM_SOVEREIGN_FIXTURE mode=transport-exit start_accepted=true bearer_exported=false presenter_sha256_unreported=true\n%!";
    exit 0);
  if mode <> "execute" then failf "unknown-mode:%s" mode;
  let presenter_code = run_presenter presenter in
  Printf.eprintf
    "LOOM_SOVEREIGN_FIXTURE mode=execute start_accepted=true result_presented=true bearer_exported=false presenter_code=%d\n%!"
    presenter_code;
  exit presenter_code
