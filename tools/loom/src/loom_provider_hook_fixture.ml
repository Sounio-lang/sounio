open Unix

let fail message =
  prerr_endline ("loom-provider-hook-fixture: " ^ message);
  exit 64

let write_all descriptor value =
  let bytes = Bytes.of_string value in
  let rec loop offset =
    if offset < Bytes.length bytes then
      let count = Unix.write descriptor bytes offset (Bytes.length bytes - offset) in
      if count = 0 then fail "short-write" else loop (offset + count)
  in
  loop 0

let require_absolute_runtime () =
  match Sys.getenv_opt "SOUNIO_LOOM_PROVIDER_HOOK_RUNTIME" with
  | Some path when not (Filename.is_relative path) -> path
  | Some _ -> fail "hook-runtime-must-be-absolute"
  | None -> fail "hook-runtime-is-required"

let event_argument () =
  let count = Array.length Sys.argv in
  if count < 3 || Sys.argv.(1) <> "exec" then fail "codex-exec-abi-required";
  Sys.argv.(count - 1)

let await_provider_start_ready () =
  let path =
    match Sys.getenv_opt "SOUNIO_LOOM_PROVIDER_START_READY_PATH" with
    | Some value when not (Filename.is_relative value) -> value
    | Some _ -> fail "provider-start-ready-path-must-be-absolute"
    | None -> fail "provider-start-ready-path-is-required"
  in
  let deadline = Unix.gettimeofday () +. 8.0 in
  let rec wait () =
    if Sys.file_exists path then ()
    else if Unix.gettimeofday () >= deadline then fail "provider-start-ready-timeout"
    else (
      Unix.sleepf 0.01;
      wait ())
  in
  wait ()

let exit_with_status = function
  | WEXITED code -> exit code
  | WSIGNALED signal | WSTOPPED signal -> exit (128 + signal)

let () =
  let runtime = require_absolute_runtime () in
  let event = event_argument () in
  let read_descriptor, write_descriptor = Unix.pipe ~cloexec:true () in
  let argv = [| runtime; "agent-hook"; "--agent"; "codex" |] in
  let child =
    Unix.create_process runtime argv read_descriptor Unix.stdout Unix.stderr
  in
  Unix.close read_descriptor;
  (try write_all write_descriptor (event ^ "\n")
   with error ->
     Unix.close write_descriptor;
     raise error);
  Unix.close write_descriptor;
  let _, status = Unix.waitpid [] child in
  await_provider_start_ready ();
  exit_with_status status
