open Unix

exception Loom_error of string

let protocol_version = 1
let runtime_version = "2026.08.24.0"
let max_control_bytes = 16 * 1024
let max_snapshot_bytes = 1024 * 1024
let max_pending_bytes = 8 * 1024 * 1024

external forkpty : unit -> int * file_descr = "sounio_loom_forkpty"
external set_winsize : file_descr -> int -> int -> unit = "sounio_loom_set_winsize"

let failf format = Printf.ksprintf (fun value -> raise (Loom_error value)) format

let starts_with value prefix =
  String.length value >= String.length prefix
  && String.sub value 0 (String.length prefix) = prefix

let trim value = String.trim value

let split_on character value = String.split_on_char character value

let hex_of_string value =
  let alphabet = "0123456789abcdef" in
  let output = Bytes.create (String.length value * 2) in
  String.iteri
    (fun index character ->
      let code = Char.code character in
      Bytes.set output (index * 2) alphabet.[code lsr 4];
      Bytes.set output ((index * 2) + 1) alphabet.[code land 0x0f])
    value;
  Bytes.unsafe_to_string output

let hex_value character =
  match character with
  | '0' .. '9' -> Char.code character - Char.code '0'
  | 'a' .. 'f' -> 10 + Char.code character - Char.code 'a'
  | 'A' .. 'F' -> 10 + Char.code character - Char.code 'A'
  | _ -> failf "invalid hexadecimal character"

let string_of_hex value =
  if String.length value mod 2 <> 0 then failf "invalid hexadecimal payload";
  let output = Bytes.create (String.length value / 2) in
  for index = 0 to Bytes.length output - 1 do
    let high = hex_value value.[index * 2] in
    let low = hex_value value.[(index * 2) + 1] in
    Bytes.set output index (Char.chr ((high lsl 4) lor low))
  done;
  Bytes.unsafe_to_string output

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let random_hex byte_count =
  let descriptor = Unix.openfile "/dev/urandom" [ O_RDONLY ] 0 in
  let bytes = Bytes.create byte_count in
  let rec fill offset =
    if offset < byte_count then
      let count = Unix.read descriptor bytes offset (byte_count - offset) in
      if count = 0 then failf "unexpected EOF from /dev/urandom"
      else fill (offset + count)
  in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () -> fill 0; hex_of_string (Bytes.unsafe_to_string bytes))

let utc_now () =
  let value = Unix.gettimeofday () in
  let seconds = int_of_float value in
  let micros = int_of_float ((value -. float_of_int seconds) *. 1_000_000.) in
  let tm = Unix.gmtime value in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02d.%06dZ"
    (tm.tm_year + 1900) (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min
    tm.tm_sec micros

let rec mkdir_p path =
  if path = "" || path = "." || path = "/" || Sys.file_exists path then ()
  else (
    mkdir_p (Filename.dirname path);
    Unix.mkdir path 0o700)

let write_all descriptor value =
  let bytes = Bytes.unsafe_of_string value in
  let rec write offset =
    if offset < Bytes.length bytes then
      let count = Unix.write descriptor bytes offset (Bytes.length bytes - offset) in
      if count = 0 then failf "short write" else write (offset + count)
  in
  write 0

let read_file path =
  let channel = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () ->
      let output = Buffer.create 4096 in
      let bytes = Bytes.create 65536 in
      let rec loop () =
        let count = input channel bytes 0 (Bytes.length bytes) in
        if count > 0 then (Buffer.add_subbytes output bytes 0 count; loop ())
      in
      loop ();
      Buffer.contents output)

let file_size path =
  try (Unix.stat path).st_size with _ -> 0

let atomic_write ?(mode = 0o600) path value =
  mkdir_p (Filename.dirname path);
  let temporary =
    Printf.sprintf "%s/.%s.%d.%s" (Filename.dirname path)
      (Filename.basename path) (Unix.getpid ()) (random_hex 4)
  in
  let descriptor =
    Unix.openfile temporary [ O_WRONLY; O_CREAT; O_EXCL; O_TRUNC ] mode
  in
  try
    write_all descriptor value;
    Unix.fsync descriptor;
    Unix.close descriptor;
    Unix.rename temporary path;
    Unix.chmod path mode
  with error ->
    (try Unix.close descriptor with _ -> ());
    (try Unix.unlink temporary with _ -> ());
    raise error

let read_lines path =
  let input = open_in path in
  let rec loop values =
    match input_line input with
    | line -> loop (line :: values)
    | exception End_of_file -> List.rev values
  in
  Fun.protect ~finally:(fun () -> close_in_noerr input) (fun () -> loop [])

let parse_key_values path =
  let table = Hashtbl.create 24 in
  if Sys.file_exists path then
    List.iter
      (fun line ->
        match String.index_opt line '=' with
        | None -> ()
        | Some index ->
            Hashtbl.replace table (String.sub line 0 index)
              (String.sub line (index + 1) (String.length line - index - 1)))
      (read_lines path);
  table

let table_value ?(default = "") table key =
  match Hashtbl.find_opt table key with Some value -> value | None -> default

let slug value =
  let output = Bytes.of_string value in
  Bytes.iteri
    (fun index character ->
      match character with
      | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '.' | '_' | '-' -> ()
      | _ -> Bytes.set output index '-')
    output;
  let result = Bytes.unsafe_to_string output in
  if result = "" then "lane" else result

let process_output command arguments =
  let input = Unix.open_process_args_in command arguments in
  let output =
    try input_line input with End_of_file -> ""
  in
  match Unix.close_process_in input with
  | WEXITED 0 -> trim output
  | _ -> failf "command failed: %s" command

let process_start pid =
  let value = read_file (Printf.sprintf "/proc/%d/stat" pid) in
  let closing =
    try String.rindex value ')' with Not_found -> failf "invalid process stat for pid %d" pid
  in
  let tail = String.sub value (closing + 2) (String.length value - closing - 2) in
  match List.nth_opt (split_on ' ' tail) 19 with
  | Some start -> start
  | None -> failf "process stat omitted start time for pid %d" pid

let json_quote value =
  let buffer = Buffer.create (String.length value + 8) in
  Buffer.add_char buffer '"';
  String.iter
    (fun character ->
      match character with
      | '"' -> Buffer.add_string buffer "\\\""
      | '\\' -> Buffer.add_string buffer "\\\\"
      | '\b' -> Buffer.add_string buffer "\\b"
      | '\012' -> Buffer.add_string buffer "\\f"
      | '\n' -> Buffer.add_string buffer "\\n"
      | '\r' -> Buffer.add_string buffer "\\r"
      | '\t' -> Buffer.add_string buffer "\\t"
      | character when Char.code character < 32 ->
          Buffer.add_string buffer (Printf.sprintf "\\u%04x" (Char.code character))
      | _ -> Buffer.add_char buffer character)
    value;
  Buffer.add_char buffer '"';
  Buffer.contents buffer

let command_argv_digest command =
  command |> Array.to_list |> List.map json_quote |> String.concat ","
  |> Printf.sprintf "[%s]" |> sha256

let logical_command_name command =
  if Array.length command = 0 then ""
  else
    let first = Filename.basename command.(0) in
    if first <> "env" then first
    else
      let rec find index =
        if index >= Array.length command then first
        else if String.contains command.(index) '=' then find (index + 1)
        else Filename.basename command.(index)
      in
      find 1

let git_common_dir cwd =
  let output =
    process_output "git"
      [| "git"; "-C"; cwd; "rev-parse"; "--path-format=absolute"; "--git-common-dir" |]
  in
  if Filename.is_relative output then Filename.concat cwd output else output

let state_root ?override cwd =
  let root =
    match override with
    | Some value -> value
    | None -> (
        match Sys.getenv_opt "SOUNIO_LOOM_DIR" with
        | Some value -> value
        | None -> (
            try Filename.concat (git_common_dir cwd) "sounio-loom"
            with _ ->
              let home = Option.value ~default:"/tmp" (Sys.getenv_opt "HOME") in
              Filename.concat home ".local/state/sounio-loom"))
  in
  mkdir_p root;
  Unix.chmod root 0o700;
  Unix.realpath root

type paths = {
  session_dir : string;
  socket_path : string;
  token_path : string;
  descriptor_path : string;
  lock_path : string;
  daemon_log_path : string;
  cursor_path : string;
}

let session_paths root agent lane =
  let key = Printf.sprintf "%s--%s" (slug agent) (slug lane) in
  let session_dir = Filename.concat (Filename.concat root "sessions") key in
  let socket_root =
    match Sys.getenv_opt "XDG_RUNTIME_DIR" with
    | Some value -> Filename.concat value (Printf.sprintf "sounio-loom-%d" (Unix.getuid ()))
    | None -> Filename.concat "/tmp" (Printf.sprintf "sounio-loom-%d" (Unix.getuid ()))
  in
  mkdir_p socket_root;
  Unix.chmod socket_root 0o700;
  let socket_name = Digest.to_hex (Digest.string (root ^ "\000" ^ agent ^ "\000" ^ lane)) ^ ".sock" in
  let cursor_dir = Filename.concat root "cursors" in
  mkdir_p cursor_dir;
  {
    session_dir;
    socket_path = Filename.concat socket_root socket_name;
    token_path = Filename.concat session_dir "capability";
    descriptor_path = Filename.concat session_dir "session.state";
    lock_path = Filename.concat session_dir "daemon.lock";
    daemon_log_path = Filename.concat session_dir "daemon.log";
    cursor_path = Filename.concat cursor_dir (key ^ ".cursor");
  }

let descriptor_text fields =
  fields
  |> List.map (fun (key, value) -> key ^ "=" ^ value ^ "\n")
  |> String.concat ""

type journal_event = {
  seq : int;
  previous : string;
  hash : string;
  utc : string;
  kind : string;
  payload_hex : string;
}

type journal = {
  channel : out_channel;
  descriptor : file_descr;
  mutable seq : int;
  mutable previous : string;
}

let event_material seq previous utc kind payload_hex =
  Printf.sprintf "%d\t%s\t%s\t%s\t%s" seq previous utc kind payload_hex

let encode_event (event : journal_event) =
  Printf.sprintf "%d\t%s\t%s\t%s\t%s\t%s\n" event.seq event.previous
    event.hash event.utc event.kind event.payload_hex

let append_event journal kind payload =
  let seq = journal.seq + 1 in
  let utc = utc_now () in
  let payload_hex = hex_of_string payload in
  let hash = sha256 (event_material seq journal.previous utc kind payload_hex) in
  let event = { seq; previous = journal.previous; hash; utc; kind; payload_hex } in
  output_string journal.channel (encode_event event);
  flush journal.channel;
  Unix.fsync journal.descriptor;
  journal.seq <- seq;
  journal.previous <- hash;
  event

let parse_event line =
  match split_on '\t' line with
  | [ seq; previous; hash; utc; kind; payload_hex ] ->
      let seq = try int_of_string seq with _ -> failf "journal sequence is not an integer" in
      { seq; previous; hash; utc; kind; payload_hex }
  | _ -> failf "journal record does not have six fields"

type journal_phase = Initial | Active | Exited

let verify_events events =
  let expected_seq = ref 1 in
  let expected_previous = ref (String.make 64 '0') in
  let phase = ref Initial in
  let lease = ref None in
  List.iter
    (fun (event : journal_event) ->
      if event.seq <> !expected_seq then
        failf "hash:non-contiguous-sequence expected=%d actual=%d" !expected_seq event.seq;
      if event.previous <> !expected_previous then failf "hash:previous-digest-mismatch seq=%d" event.seq;
      let expected_hash =
        sha256
          (event_material event.seq event.previous event.utc event.kind event.payload_hex)
      in
      if event.hash <> expected_hash then failf "hash:event-digest-mismatch seq=%d" event.seq;
      let payload = string_of_hex event.payload_hex in
      (match (event.kind, !phase) with
      | "SESSION_STARTED", Initial -> phase := Active
      | "SESSION_STARTED", _ -> failf "semantic:duplicate-session-start seq=%d" event.seq
      | "SESSION_EXITED", Active ->
          if !lease <> None then failf "semantic:exit-with-active-input-lease seq=%d" event.seq;
          phase := Exited
      | "LEASE_ACQUIRED", Active -> (
          match !lease with
          | None -> lease := Some payload
          | Some _ -> failf "semantic:duplicate-input-lease seq=%d" event.seq)
      | "LEASE_RELEASED", Active -> (
          match !lease with
          | Some holder when holder = payload -> lease := None
          | None -> failf "semantic:release-without-input-lease seq=%d" event.seq
          | Some _ -> failf "semantic:wrong-input-lease-holder seq=%d" event.seq)
      | ( "OUTPUT" | "INPUT" | "WAKE" | "OBSERVER_ATTACHED" | "OBSERVER_DETACHED" ),
        Active -> ()
      | _, Initial -> failf "semantic:event-before-session-start seq=%d" event.seq
      | _, Exited -> failf "semantic:event-after-session-exit seq=%d" event.seq
      | _ -> failf "semantic:unknown-event kind=%s seq=%d" event.kind event.seq);
      expected_previous := event.hash;
      incr expected_seq)
    events;
  (!phase, !expected_previous)

let load_and_verify_journal path =
  let events = read_lines path |> List.filter (fun line -> trim line <> "") |> List.map parse_event in
  if events = [] then failf "journal is empty";
  let phase, digest = verify_events events in
  (events, phase, digest)

let open_journal path =
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_TRUNC ] 0o600 in
  Unix.set_close_on_exec descriptor;
  let channel = Unix.out_channel_of_descr descriptor in
  { channel; descriptor; seq = 0; previous = String.make 64 '0' }

type stream_mode = Awaiting | Observer | Interactive of string

type client = {
  fd : file_descr;
  id : string;
  input : Buffer.t;
  mutable mode : stream_mode;
  mutable pending : string;
  mutable pending_offset : int;
}

type kernel = {
  paths : paths;
  agent : string;
  lane : string;
  session_id : string;
  cwd : string;
  command : string array;
  instance_id : string;
  output_path : string;
  journal_path : string;
  token : string;
  listener : file_descr;
  master_fd : file_descr;
  daemon_pid_start : string;
  harness_pid : int;
  harness_pid_start : string;
  started_utc : string;
  output_channel : out_channel;
  output_descriptor : file_descr;
  journal : journal;
  clients : (file_descr, client) Hashtbl.t;
  mutable next_client : int;
  mutable input_holder : file_descr option;
  mutable output_cursor : int;
  mutable stopping : bool;
  mutable harness_exit : int option;
  mutable next_coord_refresh : float;
  mutable coord_pid : int option;
}

let queue client value =
  let remaining = String.length client.pending - client.pending_offset in
  let pending =
    if remaining = 0 then value
    else String.sub client.pending client.pending_offset remaining ^ value
  in
  if String.length pending > max_pending_bytes then failf "client output queue exceeded recovery window";
  client.pending <- pending;
  client.pending_offset <- 0

let flush_client client =
  let remaining = String.length client.pending - client.pending_offset in
  if remaining > 0 then
    try
      let count = Unix.write_substring client.fd client.pending client.pending_offset remaining in
      client.pending_offset <- client.pending_offset + count;
      if client.pending_offset = String.length client.pending then (
        client.pending <- "";
        client.pending_offset <- 0)
    with Unix_error ((EAGAIN | EWOULDBLOCK), _, _) -> ()

let descriptor_fields kernel state =
  [
    ("protocol", string_of_int protocol_version);
    ("runtime_version", runtime_version);
    ("state", state);
    ("agent", kernel.agent);
    ("lane", kernel.lane);
    ("session_id", kernel.session_id);
    ("worktree", kernel.cwd);
    ("instance_id", kernel.instance_id);
    ("daemon_pid", string_of_int (Unix.getpid ()));
    ("daemon_pid_start", kernel.daemon_pid_start);
    ("harness_pid", string_of_int kernel.harness_pid);
    ("harness_pid_start", kernel.harness_pid_start);
    ("socket", kernel.paths.socket_path);
    ("token_file", kernel.paths.token_path);
    ("output_file", kernel.output_path);
    ("journal_file", kernel.journal_path);
    ("output_cursor", string_of_int kernel.output_cursor);
    ("command", logical_command_name kernel.command);
    ("argv_digest", command_argv_digest kernel.command);
    ("started_utc", kernel.started_utc);
  ]

let write_descriptor kernel state =
  atomic_write kernel.paths.descriptor_path (descriptor_text (descriptor_fields kernel state))

let status_fields kernel =
  let observers = ref 0 in
  Hashtbl.iter
    (fun _ client -> if client.mode = Observer then incr observers)
    kernel.clients;
  [
    ("state", "active");
    ("agent", kernel.agent);
    ("lane", kernel.lane);
    ("session_id", kernel.session_id);
    ("instance_id", kernel.instance_id);
    ("daemon_pid", string_of_int (Unix.getpid ()));
    ("daemon_pid_start", kernel.daemon_pid_start);
    ("harness_pid", string_of_int kernel.harness_pid);
    ("harness_pid_start", kernel.harness_pid_start);
    ("output_cursor", string_of_int kernel.output_cursor);
    ("interactive_clients", if kernel.input_holder = None then "0" else "1");
    ("observer_clients", string_of_int !observers);
    ("journal", kernel.journal_path);
    ("output", kernel.output_path);
    ("worktree", kernel.cwd);
    ("command", logical_command_name kernel.command);
    ("argv_digest", command_argv_digest kernel.command);
  ]

let field_escape value =
  let buffer = Buffer.create (String.length value) in
  String.iter
    (fun character ->
      match character with
      | '\t' | '\n' | '\r' | '%' -> Buffer.add_string buffer (Printf.sprintf "%%%02X" (Char.code character))
      | _ -> Buffer.add_char buffer character)
    value;
  Buffer.contents buffer

let field_unescape value =
  let buffer = Buffer.create (String.length value) in
  let rec loop index =
    if index < String.length value then
      if value.[index] = '%' && index + 2 < String.length value then (
        Buffer.add_char buffer
          (Char.chr ((hex_value value.[index + 1] lsl 4) lor hex_value value.[index + 2]));
        loop (index + 3))
      else (Buffer.add_char buffer value.[index]; loop (index + 1))
  in
  loop 0;
  Buffer.contents buffer

let control_line fields = String.concat "\t" fields ^ "\n"

let close_client kernel descriptor =
  match Hashtbl.find_opt kernel.clients descriptor with
  | None -> ()
  | Some client ->
      (match client.mode with
      | Interactive holder ->
          if kernel.input_holder = Some descriptor then kernel.input_holder <- None;
          ignore (append_event kernel.journal "LEASE_RELEASED" holder)
      | Observer -> ignore (append_event kernel.journal "OBSERVER_DETACHED" client.id)
      | Awaiting -> ());
      Hashtbl.remove kernel.clients descriptor;
      (try Unix.close descriptor with _ -> ())

let read_output_range kernel cursor limit =
  if cursor < 0 || cursor > kernel.output_cursor then
    failf "cursor-ahead cursor=%d end=%d" cursor kernel.output_cursor;
  let length = min limit (kernel.output_cursor - cursor) in
  let descriptor = Unix.openfile kernel.output_path [ O_RDONLY ] 0 in
  let bytes = Bytes.create length in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      ignore (Unix.lseek descriptor cursor SEEK_SET);
      let rec fill offset =
        if offset < length then
          let count = Unix.read descriptor bytes offset (length - offset) in
          if count = 0 then failf "durable output ended before cursor" else fill (offset + count)
      in
      fill 0;
      Bytes.unsafe_to_string bytes)

let parse_nonnegative name value =
  let parsed = try int_of_string value with _ -> failf "invalid-%s" name in
  if parsed < 0 then failf "invalid-%s" name;
  parsed

let handle_request kernel client line =
  let refuse code =
    queue client (control_line [ "ERR"; code ]);
    client.mode <- Awaiting
  in
  match split_on '\t' line with
  | magic :: token :: operation :: arguments
    when magic = Printf.sprintf "LOOM/%d" protocol_version && token = kernel.token -> (
      match (operation, arguments) with
      | "STATUS", [] ->
          let fields =
            status_fields kernel
            |> List.map (fun (key, value) -> key ^ "=" ^ field_escape value)
          in
          queue client (control_line ("OK" :: "STATUS" :: fields))
      | "SNAPSHOT", [ cursor; limit ] -> (
          try
            let cursor = parse_nonnegative "cursor" cursor in
            let limit = min max_snapshot_bytes (parse_nonnegative "limit" limit) in
            let data = read_output_range kernel cursor limit in
            let ending = cursor + String.length data in
            queue client
              (control_line
                 [ "OK"; "SNAPSHOT"; kernel.instance_id; string_of_int cursor;
                   string_of_int ending; string_of_int (String.length data) ]);
            queue client data
          with Loom_error error -> refuse error)
      | "ATTACH", [ mode; cursor ] -> (
          try
            let cursor = parse_nonnegative "cursor" cursor in
            if cursor > kernel.output_cursor then
              failf "cursor-ahead cursor=%d end=%d" cursor kernel.output_cursor;
            let replay_length = kernel.output_cursor - cursor in
            if replay_length > max_pending_bytes then failf "replay-window-too-large";
            let replay = read_output_range kernel cursor replay_length in
            if mode = "interactive" then (
              if kernel.input_holder <> None then failf "interactive-client-active";
              let holder = client.id ^ ":" ^ random_hex 8 in
              kernel.input_holder <- Some client.fd;
              client.mode <- Interactive holder;
              ignore (append_event kernel.journal "LEASE_ACQUIRED" holder))
            else if mode = "observe" then (
              client.mode <- Observer;
              ignore (append_event kernel.journal "OBSERVER_ATTACHED" client.id))
            else failf "invalid-attach-mode";
            queue client
              (control_line
                 [ "OK"; "ATTACHED"; kernel.instance_id; string_of_int cursor;
                   string_of_int kernel.output_cursor; mode ]);
            queue client replay
          with Loom_error error -> refuse error)
      | "WAKE", [ session_id; message_id; prompt_hex ] -> (
          try
            if session_id <> kernel.session_id then failf "identity-mismatch";
            let prompt = string_of_hex prompt_hex in
            if prompt = "" || String.length prompt > 16 * 1024
               || String.contains prompt '\000'
            then failf "invalid-prompt";
            write_all kernel.master_fd prompt;
            Unix.sleepf 0.025;
            write_all kernel.master_fd "\r";
            ignore
              (append_event kernel.journal "WAKE"
                 (Printf.sprintf "%s:%s" message_id (sha256 prompt)));
            queue client
              (control_line [ "OK"; "WAKE"; kernel.instance_id; message_id ])
          with Loom_error error -> refuse error)
      | "STOP", [] ->
          queue client (control_line [ "OK"; "STOPPING" ]);
          flush_client client;
          kernel.stopping <- true
      | _ -> refuse "unknown-operation")
  | magic :: _ when magic <> Printf.sprintf "LOOM/%d" protocol_version -> refuse "protocol-refused"
  | _ -> refuse "authentication-refused"

let read_client kernel descriptor =
  match Hashtbl.find_opt kernel.clients descriptor with
  | None -> ()
  | Some client ->
      let bytes = Bytes.create 65536 in
      (try
         let count = Unix.read descriptor bytes 0 (Bytes.length bytes) in
         if count = 0 then close_client kernel descriptor
         else
           match client.mode with
           | Interactive holder ->
               let value = Bytes.sub_string bytes 0 count in
               ignore
                 (append_event kernel.journal "INPUT"
                    (Printf.sprintf "%s:%d:%s" holder count (sha256 value)));
               write_all kernel.master_fd value
           | Observer -> close_client kernel descriptor
           | Awaiting ->
               Buffer.add_subbytes client.input bytes 0 count;
               if Buffer.length client.input > max_control_bytes then close_client kernel descriptor
               else
                 let value = Buffer.contents client.input in
                 (match String.index_opt value '\n' with
                 | None -> ()
                 | Some index ->
                     if index <> String.length value - 1 then close_client kernel descriptor
                     else handle_request kernel client (String.sub value 0 index))
       with
      | Unix_error ((EAGAIN | EWOULDBLOCK), _, _) -> ()
      | Unix_error _ -> close_client kernel descriptor)

let accept_client kernel =
  try
    let descriptor, _ = Unix.accept kernel.listener in
    Unix.set_close_on_exec descriptor;
    Unix.set_nonblock descriptor;
    kernel.next_client <- kernel.next_client + 1;
    let client =
      {
        fd = descriptor;
        id = Printf.sprintf "client-%d-%d" (Unix.getpid ()) kernel.next_client;
        input = Buffer.create 256;
        mode = Awaiting;
        pending = "";
        pending_offset = 0;
      }
    in
    Hashtbl.add kernel.clients descriptor client
  with Unix_error ((EAGAIN | EWOULDBLOCK), _, _) -> ()

let read_pty kernel =
  let bytes = Bytes.create 65536 in
  try
    let count = Unix.read kernel.master_fd bytes 0 (Bytes.length bytes) in
    if count = 0 then ()
    else
      let value = Bytes.sub_string bytes 0 count in
      let start = kernel.output_cursor in
      output_string kernel.output_channel value;
      flush kernel.output_channel;
      Unix.fsync kernel.output_descriptor;
      kernel.output_cursor <- start + count;
      ignore
        (append_event kernel.journal "OUTPUT"
           (Printf.sprintf "%d:%d:%s" start kernel.output_cursor (sha256 value)));
      let slow = ref [] in
      Hashtbl.iter
        (fun descriptor client ->
          match client.mode with
          | Observer | Interactive _ ->
              (try queue client value with Loom_error _ -> slow := descriptor :: !slow)
          | Awaiting -> ())
        kernel.clients;
      List.iter (close_client kernel) !slow
  with Unix_error ((EAGAIN | EWOULDBLOCK | EIO), _, _) -> ()

let child_status kernel =
  match Unix.waitpid [ WNOHANG ] kernel.harness_pid with
  | 0, _ -> None
  | _, status -> Some status
  | exception Unix_error (ECHILD, _, _) -> Some (WEXITED 0)

let exit_code = function WEXITED code -> code | WSIGNALED signal -> 128 + signal | WSTOPPED signal -> 128 + signal

let stop_child kernel =
  if kernel.harness_exit = None then (
    (try Unix.kill kernel.harness_pid Sys.sigterm with _ -> ());
    let deadline = Unix.gettimeofday () +. 2.0 in
    let rec wait () =
      match child_status kernel with
      | Some status -> kernel.harness_exit <- Some (exit_code status)
      | None when Unix.gettimeofday () < deadline -> Unix.sleepf 0.05; wait ()
      | None ->
          (try Unix.kill kernel.harness_pid Sys.sigkill with _ -> ());
          let _, status = Unix.waitpid [] kernel.harness_pid in
          kernel.harness_exit <- Some (exit_code status)
    in
    wait ())

let harness_for_agent agent =
  let value = String.lowercase_ascii agent in
  if starts_with value "claude" then Some "claude"
  else if starts_with value "codex" then Some "codex"
  else if starts_with value "grok" then Some "grok"
  else if starts_with value "cursor" then Some "cursor"
  else if starts_with value "kimi" then Some "kimi"
  else None

let coordination_command kernel =
  match Sys.getenv_opt "SOUNIO_COORD_COMMAND" with
  | Some path when Sys.file_exists path -> Some path
  | _ ->
      let sibling = Filename.concat (Filename.dirname Sys.executable_name) "sounio-coord-runtime" in
      if Sys.file_exists sibling then Some sibling
      else
        let launcher = Filename.concat (Filename.concat kernel.cwd "bin") "sounio-coord" in
        if Sys.file_exists launcher then Some launcher else None

let environment_replacing assignments =
  let keys = List.map fst assignments in
  let inherited =
    Unix.environment () |> Array.to_list
    |> List.filter (fun item ->
           match String.index_opt item '=' with
           | None -> true
           | Some index -> not (List.mem (String.sub item 0 index) keys))
  in
  Array.of_list (inherited @ List.map (fun (key, value) -> key ^ "=" ^ value) assignments)

let run_quiet cwd command arguments =
  match Unix.fork () with
  | 0 ->
      (try
         Unix.chdir cwd;
         let null = Unix.openfile "/dev/null" [ O_RDWR ] 0 in
         Unix.dup2 null Unix.stdin;
         Unix.dup2 null Unix.stdout;
         Unix.dup2 null Unix.stderr;
         if null <> Unix.stdin && null <> Unix.stdout && null <> Unix.stderr then Unix.close null;
         let environment = environment_replacing [ ("SOUNIO_COORD_WORKTREE", cwd) ] in
         Unix.execve command (Array.of_list (command :: arguments)) environment
       with _ -> exit 127)
  | pid ->
      let _, status = Unix.waitpid [] pid in
      exit_code status

let coord_call kernel arguments =
  match coordination_command kernel with
  | None -> 127
  | Some command -> run_quiet kernel.cwd command arguments

let refresh_coordination kernel =
  match harness_for_agent kernel.agent with
  | None -> ()
  | Some harness ->
      let ttl = Option.value ~default:"1800" (Sys.getenv_opt "SOUNIO_LOOM_COORD_TTL_SECONDS") in
      let identity = [ "--agent"; kernel.agent; "--lane"; kernel.lane ] in
      if coord_call kernel ("heartbeat" :: identity) <> 0 then
        ignore
          (coord_call kernel
             ([ "scope"; "--agent"; kernel.agent; "--lane"; kernel.lane; "--intent";
                Printf.sprintf "loom-supervised %s session" harness ]));
      let presence =
        [ "presence-register"; "--agent"; kernel.agent; "--lane"; kernel.lane;
          "--harness"; harness; "--session-id"; kernel.session_id; "--pid";
          string_of_int kernel.harness_pid; "--pid-start"; process_start kernel.harness_pid;
          "--boot-id"; trim (read_file "/proc/sys/kernel/random/boot_id");
          "--pid-namespace"; Unix.readlink (Printf.sprintf "/proc/%d/ns/pid" kernel.harness_pid);
          "--host"; Unix.gethostname (); "--ttl-seconds"; ttl ]
      in
      if coord_call kernel presence <> 0 then
        Printf.eprintf "LOOM_COORDINATION_WARNING operation=presence-register\n%!"
      else
        let endpoint =
          [ "endpoint-register"; "--agent"; kernel.agent; "--lane"; kernel.lane;
            "--harness"; harness; "--transport"; "loom"; "--address";
            kernel.paths.socket_path; "--socket"; kernel.paths.socket_path;
            "--token-file"; kernel.paths.token_path; "--ttl-seconds"; ttl ]
        in
        if coord_call kernel endpoint <> 0 then
          Printf.eprintf "LOOM_COORDINATION_WARNING operation=endpoint-register\n%!";
      ()

let reap_coordination kernel =
  match kernel.coord_pid with
  | None -> ()
  | Some pid -> (
      match Unix.waitpid [ WNOHANG ] pid with
      | 0, _ -> ()
      | _ -> kernel.coord_pid <- None
      | exception Unix_error (ECHILD, _, _) -> kernel.coord_pid <- None)

let spawn_coordination_refresh kernel =
  reap_coordination kernel;
  if kernel.coord_pid = None then (
    match Unix.fork () with
    | 0 ->
        Sys.set_signal Sys.sigterm Sys.Signal_default;
        Sys.set_signal Sys.sigint Sys.Signal_default;
        (try refresh_coordination kernel; exit 0
         with _ -> exit 1)
    | pid ->
        kernel.coord_pid <- Some pid;
        kernel.next_coord_refresh <- Unix.gettimeofday () +. 300.0)

let unregister_coordination kernel =
  match harness_for_agent kernel.agent with
  | None -> ()
  | Some _ ->
      ignore
        (coord_call kernel
           [ "endpoint-unregister"; "--agent"; kernel.agent; "--lane"; kernel.lane ]);
      ignore
        (coord_call kernel
           [ "presence-unregister"; "--agent"; kernel.agent; "--lane"; kernel.lane ])

let run_kernel kernel =
  write_descriptor kernel "active";
  ignore
    (append_event kernel.journal "SESSION_STARTED"
       (Printf.sprintf "%s:%d" kernel.instance_id kernel.harness_pid));
  if Sys.getenv_opt "SOUNIO_LOOM_COORD_AUTO" <> Some "0" then spawn_coordination_refresh kernel;
  let signal_stop _ = kernel.stopping <- true in
  Sys.set_signal Sys.sigterm (Sys.Signal_handle signal_stop);
  Sys.set_signal Sys.sigint (Sys.Signal_handle signal_stop);
  while not kernel.stopping && kernel.harness_exit = None do
    reap_coordination kernel;
    if Sys.getenv_opt "SOUNIO_LOOM_COORD_AUTO" <> Some "0"
       && Unix.gettimeofday () >= kernel.next_coord_refresh
    then spawn_coordination_refresh kernel;
    (match child_status kernel with
    | Some status -> kernel.harness_exit <- Some (exit_code status)
    | None -> ());
    let client_fds = Hashtbl.fold (fun fd _ values -> fd :: values) kernel.clients [] in
    let write_fds =
      Hashtbl.fold
        (fun fd client values ->
          if client.pending_offset < String.length client.pending then fd :: values else values)
        kernel.clients []
    in
    let readable, writable, _ =
      Unix.select (kernel.listener :: kernel.master_fd :: client_fds) write_fds [] 0.2
    in
    List.iter
      (fun descriptor ->
        if descriptor = kernel.listener then accept_client kernel
        else if descriptor = kernel.master_fd then read_pty kernel
        else read_client kernel descriptor)
      readable;
    List.iter
      (fun descriptor ->
        match Hashtbl.find_opt kernel.clients descriptor with
        | Some client -> (try flush_client client with _ -> close_client kernel descriptor)
        | None -> ())
      writable
  done;
  if kernel.stopping then stop_child kernel;
  let clients = Hashtbl.fold (fun fd _ values -> fd :: values) kernel.clients [] in
  List.iter (close_client kernel) clients;
  (match kernel.coord_pid with
  | Some pid ->
      (try Unix.kill pid Sys.sigkill with _ -> ());
      (try ignore (Unix.waitpid [] pid) with _ -> ());
      kernel.coord_pid <- None
  | None -> ());
  if Sys.getenv_opt "SOUNIO_LOOM_COORD_AUTO" <> Some "0" then unregister_coordination kernel;
  let code = Option.value ~default:0 kernel.harness_exit in
  ignore (append_event kernel.journal "SESSION_EXITED" (string_of_int code));
  write_descriptor kernel "exited";
  code

let create_listener path =
  (try Unix.unlink path with Unix_error (ENOENT, _, _) -> ());
  let listener = Unix.socket PF_UNIX SOCK_STREAM 0 in
  Unix.set_close_on_exec listener;
  Unix.bind listener (ADDR_UNIX path);
  Unix.chmod path 0o600;
  Unix.listen listener 32;
  Unix.set_nonblock listener;
  listener

let redirect_daemon_log path =
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  let null = Unix.openfile "/dev/null" [ O_RDONLY ] 0 in
  Unix.dup2 null Unix.stdin;
  Unix.dup2 descriptor Unix.stdout;
  Unix.dup2 descriptor Unix.stderr;
  Unix.close null;
  if descriptor <> Unix.stdout && descriptor <> Unix.stderr then Unix.close descriptor

let serve_session paths agent lane session_id cwd command =
  mkdir_p paths.session_dir;
  let lock = Unix.openfile paths.lock_path [ O_WRONLY; O_CREAT ] 0o600 in
  Unix.set_close_on_exec lock;
  (try Unix.lockf lock F_TLOCK 0 with Unix_error _ -> failf "another Loom generation owns this lane");
  let instance_id = random_hex 16 in
  let generation_dir = Filename.concat (Filename.concat paths.session_dir "generations") instance_id in
  mkdir_p generation_dir;
  let output_path = Filename.concat generation_dir "output.bin" in
  let journal_path = Filename.concat generation_dir "journal.tsv" in
  let output_descriptor = Unix.openfile output_path [ O_WRONLY; O_CREAT; O_TRUNC ] 0o600 in
  Unix.set_close_on_exec output_descriptor;
  let output_channel = Unix.out_channel_of_descr output_descriptor in
  let journal = open_journal journal_path in
  let listener = create_listener paths.socket_path in
  let child_pid, master_fd = forkpty () in
  if child_pid = 0 then (
    Unix.chdir cwd;
    let environment =
      Array.append (Unix.environment ())
        [| Printf.sprintf "SOUNIO_LOOM_SOCKET=%s" paths.socket_path;
           Printf.sprintf "SOUNIO_LOOM_TOKEN_FILE=%s" paths.token_path;
           Printf.sprintf "SOUNIO_LOOM_AGENT=%s" agent;
           Printf.sprintf "SOUNIO_LOOM_LANE=%s" lane;
           Printf.sprintf "SOUNIO_LOOM_SESSION_ID=%s" session_id |]
    in
    Unix.execvpe command.(0) command environment);
  Unix.set_close_on_exec master_fd;
  Unix.set_nonblock master_fd;
  (try set_winsize master_fd 40 140 with _ -> ());
  let kernel =
    {
      paths;
      agent;
      lane;
      session_id;
      cwd;
      command;
      instance_id;
      output_path;
      journal_path;
      token = trim (read_file paths.token_path);
      listener;
      master_fd;
      daemon_pid_start = process_start (Unix.getpid ());
      harness_pid = child_pid;
      harness_pid_start = process_start child_pid;
      started_utc = utc_now ();
      output_channel;
      output_descriptor;
      journal;
      clients = Hashtbl.create 16;
      next_client = 0;
      input_holder = None;
      output_cursor = 0;
      stopping = false;
      harness_exit = None;
      next_coord_refresh = 0.0;
      coord_pid = None;
    }
  in
  let code =
    try run_kernel kernel
    with error ->
      kernel.stopping <- true;
      stop_child kernel;
      (try write_descriptor kernel "failed" with _ -> ());
      raise error
  in
  (try Unix.unlink paths.socket_path with _ -> ());
  close_out_noerr output_channel;
  close_out_noerr journal.channel;
  Unix.close listener;
  Unix.close master_fd;
  Unix.close lock;
  code

type cli = {
  options : (string, string) Hashtbl.t;
  flags : (string, bool) Hashtbl.t;
  rest : string list;
}

let parse_cli boolean_flags arguments =
  let options = Hashtbl.create 16 in
  let flags = Hashtbl.create 8 in
  let rec loop rest positional =
    match rest with
    | [] -> { options; flags; rest = List.rev positional }
    | "--" :: tail -> { options; flags; rest = List.rev_append positional tail }
    | key :: tail when starts_with key "--" && List.mem key boolean_flags ->
        Hashtbl.replace flags key true;
        loop tail positional
    | key :: value :: tail when starts_with key "--" ->
        Hashtbl.replace options key value;
        loop tail positional
    | key :: [] when starts_with key "--" -> failf "%s requires a value" key
    | value :: tail -> loop tail (value :: positional)
  in
  loop arguments []

let required cli name =
  match Hashtbl.find_opt cli.options name with Some value -> value | None -> failf "%s is required" name

let optional cli name = Hashtbl.find_opt cli.options name
let flag cli name = Hashtbl.mem cli.flags name

let cwd_option cli =
  match optional cli "--cwd" with Some value -> Unix.realpath value | None -> Unix.getcwd ()

let root_option cli cwd = state_root ?override:(optional cli "--state-dir") cwd

let connect paths =
  let descriptor = Unix.socket PF_UNIX SOCK_STREAM 0 in
  try
    Unix.connect descriptor (ADDR_UNIX paths.socket_path);
    descriptor
  with error -> Unix.close descriptor; raise error

let read_line_fd descriptor =
  let buffer = Buffer.create 128 in
  let byte = Bytes.create 1 in
  let rec loop () =
    let count = Unix.read descriptor byte 0 1 in
    if count = 0 then failf "connection closed before protocol header";
    let character = Bytes.get byte 0 in
    if character = '\n' then Buffer.contents buffer
    else if Buffer.length buffer >= max_control_bytes then failf "protocol header too large"
    else (Buffer.add_char buffer character; loop ())
  in
  loop ()

let request_line token operation arguments =
  control_line (Printf.sprintf "LOOM/%d" protocol_version :: token :: operation :: arguments)

let read_exact descriptor length =
  let bytes = Bytes.create length in
  let rec loop offset =
    if offset < length then
      let count = Unix.read descriptor bytes offset (length - offset) in
      if count = 0 then failf "connection closed during payload" else loop (offset + count)
  in
  loop 0;
  Bytes.unsafe_to_string bytes

let parse_ok_header line operation =
  match split_on '\t' line with
  | "ERR" :: error :: _ -> failf "%s" error
  | "OK" :: actual :: fields when actual = operation -> fields
  | _ -> failf "invalid %s response" operation

let session_locator cli =
  let cwd = cwd_option cli in
  let direct = optional cli "--socket" <> None && optional cli "--token-file" <> None in
  let root =
    if direct && optional cli "--state-dir" = None then
      state_root ~override:(Filename.concat "/tmp" (Printf.sprintf "sounio-loom-direct-%d" (Unix.getuid ()))) cwd
    else root_option cli cwd
  in
  let agent = required cli "--agent" in
  let lane = required cli "--lane" in
  let paths = session_paths root agent lane in
  let paths =
    {
      paths with
      socket_path = Option.value ~default:paths.socket_path (optional cli "--socket");
      token_path = Option.value ~default:paths.token_path (optional cli "--token-file");
    }
  in
  (cwd, paths)

let status_request paths =
  let token = trim (read_file paths.token_path) in
  let descriptor = connect paths in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
      write_all descriptor (request_line token "STATUS" []);
      parse_ok_header (read_line_fd descriptor) "STATUS")

let snapshot_request paths cursor limit =
  let token = trim (read_file paths.token_path) in
  let descriptor = connect paths in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
      write_all descriptor
        (request_line token "SNAPSHOT" [ string_of_int cursor; string_of_int limit ]);
      match parse_ok_header (read_line_fd descriptor) "SNAPSHOT" with
      | [ instance; start; ending; length ] ->
          let start = int_of_string start and ending = int_of_string ending and length = int_of_string length in
          (instance, start, ending, read_exact descriptor length)
      | _ -> failf "invalid SNAPSHOT response fields")

let start_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let agent = required cli "--agent" in
  let lane = required cli "--lane" in
  let session_id = required cli "--session-id" in
  let command = Array.of_list cli.rest in
  if Array.length command = 0 then failf "start requires a command after --";
  let paths = session_paths root agent lane in
  mkdir_p paths.session_dir;
  let already_active =
    try ignore (status_request paths); true with _ -> false
  in
  if already_active then failf "an active Loom generation already owns %s/%s" agent lane;
  atomic_write paths.token_path (random_hex 32 ^ "\n");
  (try Unix.unlink paths.descriptor_path with _ -> ());
  match Unix.fork () with
  | 0 ->
      ignore (Unix.setsid ());
      Sys.set_signal Sys.sighup Sys.Signal_ignore;
      redirect_daemon_log paths.daemon_log_path;
      let code = serve_session paths agent lane session_id cwd command in
      exit code
  | daemon_pid ->
      let deadline = Unix.gettimeofday () +. 8.0 in
      let rec wait () =
        if Unix.gettimeofday () >= deadline then failf "Loom daemon did not become ready";
        if Sys.file_exists paths.descriptor_path then
          let values = parse_key_values paths.descriptor_path in
          if table_value values "state" = "active"
             && table_value values "daemon_pid" = string_of_int daemon_pid
          then values
          else (Unix.sleepf 0.05; wait ())
        else (Unix.sleepf 0.05; wait ())
      in
      let values = wait () in
      Printf.printf "LOOM_STARTED agent=%s lane=%s instance=%s daemon_pid=%d harness_pid=%s\n%!"
        agent lane (table_value values "instance_id") daemon_pid (table_value values "harness_pid")

let status_command cli =
  let _, paths = session_locator cli in
  let fields = status_request paths in
  if flag cli "--machine" then
    List.iter
      (fun field ->
        match String.index_opt field '=' with
        | None -> print_endline field
        | Some index ->
            Printf.printf "%s=%s\n" (String.sub field 0 index)
              (field_unescape
                 (String.sub field (index + 1) (String.length field - index - 1))))
      fields
  else Printf.printf "LOOM_STATUS %s\n%!" (String.concat " " fields)

let wake_command cli =
  let _, paths = session_locator cli in
  let session_id = required cli "--session-id" in
  let message_id = required cli "--message-id" in
  let prompt = required cli "--prompt" in
  let token = trim (read_file paths.token_path) in
  let descriptor = connect paths in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
      write_all descriptor
        (request_line token "WAKE" [ session_id; message_id; hex_of_string prompt ]);
      match parse_ok_header (read_line_fd descriptor) "WAKE" with
      | [ instance; delivered_message ] when delivered_message = message_id ->
          Printf.printf "LOOM_WAKE state=delivered instance=%s message_id=%s\n%!" instance message_id
      | _ -> failf "invalid WAKE response fields")

let stop_command cli =
  let _, paths = session_locator cli in
  let token = trim (read_file paths.token_path) in
  let descriptor = connect paths in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
      write_all descriptor (request_line token "STOP" []);
      ignore (parse_ok_header (read_line_fd descriptor) "STOPPING"));
  Printf.printf "LOOM_STOP_REQUESTED agent=%s lane=%s\n%!" (required cli "--agent") (required cli "--lane")

let load_cursor paths instance =
  try
    match split_on '\t' (trim (read_file paths.cursor_path)) with
    | [ saved_instance; cursor ] when saved_instance = instance -> int_of_string cursor
    | _ -> 0
  with _ -> 0

let save_cursor paths instance cursor =
  atomic_write paths.cursor_path (Printf.sprintf "%s\t%d\n" instance cursor)

let set_terminal_raw descriptor =
  let original = Unix.tcgetattr descriptor in
  let raw = { original with c_icanon = false; c_echo = false; c_vmin = 1; c_vtime = 0 } in
  Unix.tcsetattr descriptor TCSANOW raw;
  original

let stream_command cli interactive =
  let _, paths = session_locator cli in
  let descriptor_values = parse_key_values paths.descriptor_path in
  let expected_instance = table_value descriptor_values "instance_id" in
  let cursor =
    match optional cli "--cursor" with
    | Some "auto" | None -> load_cursor paths expected_instance
    | Some "end" -> file_size (table_value descriptor_values "output_file")
    | Some value -> parse_nonnegative "cursor" value
  in
  let token = trim (read_file paths.token_path) in
  let socket = connect paths in
  let mode = if interactive then "interactive" else "observe" in
  write_all socket (request_line token "ATTACH" [ mode; string_of_int cursor ]);
  let instance, start_cursor =
    match parse_ok_header (read_line_fd socket) "ATTACHED" with
    | [ instance; start; _ending; actual_mode ] when actual_mode = mode -> (instance, int_of_string start)
    | _ -> failf "invalid ATTACHED response fields"
  in
  let cursor = ref start_cursor in
  let terminal =
    if interactive && Unix.isatty Unix.stdin && not (flag cli "--no-raw") then
      Some (set_terminal_raw Unix.stdin)
    else None
  in
  let running = ref true in
  Fun.protect
    ~finally:(fun () ->
      Option.iter (fun original -> Unix.tcsetattr Unix.stdin TCSANOW original) terminal;
      Unix.close socket)
    (fun () ->
      while !running do
        let read_fds = if interactive then [ socket; Unix.stdin ] else [ socket ] in
        let readable, _, _ = Unix.select read_fds [] [] (-1.0) in
        List.iter
          (fun descriptor ->
            if descriptor = socket then (
              let bytes = Bytes.create 65536 in
              let count = Unix.read socket bytes 0 (Bytes.length bytes) in
              if count = 0 then running := false
              else (
                write_all Unix.stdout (Bytes.sub_string bytes 0 count);
                cursor := !cursor + count;
                save_cursor paths instance !cursor))
            else
              let bytes = Bytes.create 65536 in
              let count = Unix.read Unix.stdin bytes 0 (Bytes.length bytes) in
              if count = 0 then running := false
              else
                let value = Bytes.sub_string bytes 0 count in
                match String.index_opt value '\029' with
                | None -> write_all socket value
                | Some index ->
                    if index > 0 then write_all socket (String.sub value 0 index);
                    running := false)
          readable
      done)

let snapshot_command cli =
  let _, paths = session_locator cli in
  let cursor = optional cli "--cursor" |> Option.value ~default:"0" |> parse_nonnegative "cursor" in
  let limit = optional cli "--limit" |> Option.value ~default:(string_of_int max_snapshot_bytes) |> parse_nonnegative "limit" in
  let instance, start, ending, data = snapshot_request paths cursor limit in
  if flag cli "--meta" then
    Printf.eprintf "LOOM_SNAPSHOT instance=%s start=%d end=%d bytes=%d\n%!" instance start ending (String.length data);
  output_string Stdlib.stdout data;
  flush Stdlib.stdout

let session_descriptors root =
  let sessions = Filename.concat root "sessions" in
  if not (Sys.file_exists sessions) then []
  else
    Sys.readdir sessions |> Array.to_list |> List.sort String.compare
    |> List.filter_map (fun name ->
           let path = Filename.concat (Filename.concat sessions name) "session.state" in
           if Sys.file_exists path then Some (path, parse_key_values path) else None)

let list_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let sessions = session_descriptors root in
  List.iter
    (fun (_, values) ->
      Printf.printf
        "LOOM_SESSION state=%s agent=%s lane=%s instance=%s daemon_pid=%s harness_pid=%s cursor=%s\n"
        (table_value values "state") (table_value values "agent") (table_value values "lane")
        (table_value values "instance_id") (table_value values "daemon_pid")
        (table_value values "harness_pid") (string_of_int (file_size (table_value values "output_file"))))
    sessions;
  Printf.printf "loom_sessions=%d\n%!" (List.length sessions)

let verify_command cli =
  let path = required cli "--journal" in
  let events, phase, digest = load_and_verify_journal path in
  let phase_name = match phase with Initial -> "initial" | Active -> "active" | Exited -> "exited" in
  Printf.printf "JOURNAL_OK events=%d phase=%s digest=%s\n%!" (List.length events) phase_name digest

let forge_duplicate_lease cli =
  let input_path = required cli "--journal" in
  let output_path = required cli "--output" in
  let events, _, digest = load_and_verify_journal input_path in
  atomic_write output_path (String.concat "" (List.map encode_event events));
  let descriptor = Unix.openfile output_path [ O_WRONLY; O_APPEND ] 0o600 in
  let channel = Unix.out_channel_of_descr descriptor in
  let journal =
    { channel; descriptor; seq = List.length events; previous = digest }
  in
  ignore (append_event journal "LEASE_ACQUIRED" "sabotage-holder-a");
  ignore (append_event journal "LEASE_ACQUIRED" "sabotage-holder-b");
  close_out channel;
  Printf.printf "FORGED semantic=duplicate-input-lease output=%s\n%!" output_path

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
      | character when Char.code character < 32 -> Buffer.add_string buffer (Printf.sprintf "\\u%04x" (Char.code character))
      | _ -> Buffer.add_char buffer character)
    value;
  Buffer.contents buffer

let sessions_json root =
  session_descriptors root
  |> List.map (fun (_, values) ->
         let output = table_value values "output_file" in
         Printf.sprintf
           "{\"agent\":\"%s\",\"lane\":\"%s\",\"session_id\":\"%s\",\"instance_id\":\"%s\",\"state\":\"%s\",\"daemon_pid\":\"%s\",\"harness_pid\":\"%s\",\"cursor\":%d}"
           (json_escape (table_value values "agent")) (json_escape (table_value values "lane"))
           (json_escape (table_value values "session_id")) (json_escape (table_value values "instance_id"))
           (json_escape (table_value values "state")) (json_escape (table_value values "daemon_pid"))
           (json_escape (table_value values "harness_pid")) (file_size output))
  |> String.concat "," |> Printf.sprintf "[%s]"

let html =
  {|
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sounio Loom</title>
<style>
:root{color-scheme:dark;--bg:#0c0e0f;--panel:#131719;--line:#2a3235;--text:#e8eeee;--muted:#8e9a9d;--cyan:#62c6d4;--green:#74cf88;--amber:#e1b55f;--red:#e57373}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:0;height:100vh;overflow:hidden}
header{height:48px;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 16px;gap:18px;background:#101315}header strong{font-size:15px;color:#fff}header span{color:var(--muted)}#connection{margin-left:auto;color:var(--green)}
main{display:grid;grid-template-columns:minmax(260px,340px) 1fr;height:calc(100vh - 48px)}aside{border-right:1px solid var(--line);overflow:auto;background:var(--panel)}
.lane{width:100%;display:grid;grid-template-columns:12px 1fr auto;gap:10px;text-align:left;padding:12px 14px;border:0;border-bottom:1px solid var(--line);background:transparent;color:var(--text);cursor:pointer}.lane:hover,.lane.active{background:#1b2123}.dot{width:8px;height:8px;background:var(--green);margin-top:4px}.lane.exited .dot{background:var(--red)}.lane small{display:block;color:var(--muted);margin-top:4px}.cursor{color:var(--cyan);font-size:11px}
section{min-width:0;display:grid;grid-template-rows:44px 1fr}.meta{border-bottom:1px solid var(--line);display:flex;align-items:center;gap:22px;padding:0 15px;color:var(--muted)}.meta b{color:var(--text);font-weight:500}.terminal{margin:0;padding:16px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.45;color:#dbe6e6;background:#0a0c0d}.empty{padding:22px;color:var(--muted)}
@media(max-width:700px){main{grid-template-columns:1fr;grid-template-rows:190px 1fr}aside{border-right:0;border-bottom:1px solid var(--line)}header span{display:none}.meta{gap:10px;overflow:auto}}
</style>
</head>
<body>
<header><strong>SOUNIO LOOM</strong><span>durable fleet multiplexer</span><span id="connection">LOCAL / READ ONLY</span></header>
<main><aside id="lanes"></aside><section><div class="meta" id="meta"><span>No lane selected</span></div><pre class="terminal" id="terminal"></pre></section></main>
<script>
const lanes=document.querySelector('#lanes'),term=document.querySelector('#terminal'),meta=document.querySelector('#meta');let selected=null,cursor=0,timer=null;
const clean=s=>s.replace(/\x1b\[[0-?]*[ -\/]*[@-~]/g,'').replace(/\x1b\][^\x07]*(?:\x07|\x1b\\)/g,'');
async function refresh(){const list=await fetch('/api/sessions',{cache:'no-store'}).then(r=>r.json());lanes.replaceChildren();for(const s of list){const b=document.createElement('button'),dot=document.createElement('i'),label=document.createElement('span'),detail=document.createElement('small'),size=document.createElement('em');b.className='lane '+s.state+(selected&&selected.instance_id===s.instance_id?' active':'');dot.className='dot';label.textContent=s.agent;detail.textContent=s.lane;size.className='cursor';size.textContent=s.cursor+' B';label.appendChild(detail);b.append(dot,label,size);b.onclick=()=>choose(s);lanes.appendChild(b)}if(!list.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='No Loom sessions';lanes.appendChild(empty)}if(selected){const now=list.find(s=>s.instance_id===selected.instance_id);if(now)selected=now;await poll()}setTimeout(refresh,1000)}
function metaField(label,value){const span=document.createElement('span'),name=document.createTextNode(label+' '),strong=document.createElement('b');strong.textContent=value;span.append(name,strong);return span}function choose(s){selected=s;cursor=0;term.textContent='';meta.replaceChildren(metaField('lane',s.lane),metaField('generation',s.instance_id.slice(0,12)),metaField('pid',s.harness_pid));}
async function poll(){if(!selected)return;const q=new URLSearchParams({agent:selected.agent,lane:selected.lane,cursor:String(cursor)});const r=await fetch('/api/snapshot?'+q,{cache:'no-store'});if(!r.ok)return;const data=new Uint8Array(await r.arrayBuffer());cursor=Number(r.headers.get('x-loom-cursor')||cursor);if(data.length){term.textContent+=clean(new TextDecoder().decode(data));term.scrollTop=term.scrollHeight}}
refresh();
</script>
</body>
</html>
|}

let http_response ?(headers = []) status content_type body =
  let header_lines =
    ("Content-Type", content_type) :: ("Content-Length", string_of_int (String.length body))
    :: ("Cache-Control", "no-store") :: ("Connection", "close") :: headers
  in
  Printf.sprintf "HTTP/1.1 %s\r\n%s\r\n\r\n%s" status
    (String.concat "\r\n" (List.map (fun (key, value) -> key ^ ": " ^ value) header_lines)) body

let percent_decode value =
  let buffer = Buffer.create (String.length value) in
  let rec loop index =
    if index < String.length value then
      if value.[index] = '%' && index + 2 < String.length value then (
        Buffer.add_char buffer (Char.chr ((hex_value value.[index + 1] lsl 4) lor hex_value value.[index + 2]));
        loop (index + 3))
      else (
        Buffer.add_char buffer (if value.[index] = '+' then ' ' else value.[index]);
        loop (index + 1))
  in
  loop 0;
  Buffer.contents buffer

let parse_query uri =
  match String.index_opt uri '?' with
  | None -> (uri, Hashtbl.create 0)
  | Some index ->
      let table = Hashtbl.create 8 in
      String.sub uri (index + 1) (String.length uri - index - 1)
      |> split_on '&'
      |> List.iter (fun item ->
             match String.index_opt item '=' with
             | Some split ->
                 Hashtbl.replace table (percent_decode (String.sub item 0 split))
                   (percent_decode (String.sub item (split + 1) (String.length item - split - 1)))
             | None -> ());
      (String.sub uri 0 index, table)

let snapshot_from_files root query =
  let agent = table_value query "agent" and lane = table_value query "lane" in
  let cursor = table_value ~default:"0" query "cursor" |> parse_nonnegative "cursor" in
  let matching =
    session_descriptors root
    |> List.find_opt (fun (_, values) -> table_value values "agent" = agent && table_value values "lane" = lane)
  in
  match matching with
  | None -> ("404 Not Found", [], "unknown lane")
  | Some (_, values) ->
      let output = table_value values "output_file" in
      let ending = file_size output in
      if cursor > ending then ("409 Conflict", [], "cursor ahead")
      else
        let length = min max_snapshot_bytes (ending - cursor) in
        let descriptor = Unix.openfile output [ O_RDONLY ] 0 in
        let bytes = Bytes.create length in
        Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
            ignore (Unix.lseek descriptor cursor SEEK_SET);
            let rec fill offset =
              if offset < length then
                let count = Unix.read descriptor bytes offset (length - offset) in
                if count > 0 then fill (offset + count)
            in
            fill 0;
            ( "200 OK",
              [ ("X-Loom-Cursor", string_of_int (cursor + length));
                ("X-Loom-Instance", table_value values "instance_id") ],
              Bytes.unsafe_to_string bytes ))

let serve_http cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let bind = optional cli "--bind" |> Option.value ~default:"127.0.0.1" in
  if bind <> "127.0.0.1" && bind <> "localhost" && not (flag cli "--allow-remote") then
    failf "remote GUI bind requires --allow-remote";
  let port = optional cli "--port" |> Option.value ~default:"8787" |> int_of_string in
  let address =
    try Unix.inet_addr_of_string bind
    with _ -> (Unix.gethostbyname bind).h_addr_list.(0)
  in
  let server = Unix.socket PF_INET SOCK_STREAM 0 in
  Unix.setsockopt server SO_REUSEADDR true;
  Unix.bind server (ADDR_INET (address, port));
  Unix.listen server 32;
  let actual_port = match Unix.getsockname server with ADDR_INET (_, value) -> value | _ -> port in
  let running = ref true in
  let stop _ = running := false in
  Sys.set_signal Sys.sigterm (Sys.Signal_handle stop);
  Sys.set_signal Sys.sigint (Sys.Signal_handle stop);
  Printf.printf "LOOM_GUI url=http://%s:%d read_only=true\n%!" bind actual_port;
  while !running do
    let readable, _, _ = Unix.select [ server ] [] [] 0.25 in
    if readable <> [] then
      let client, _ = Unix.accept server in
      (try
         let bytes = Bytes.create 16384 in
         let count = Unix.read client bytes 0 (Bytes.length bytes) in
         let request = Bytes.sub_string bytes 0 count in
         let first_line = match split_on '\n' request with line :: _ -> trim line | [] -> "" in
         let response =
           match split_on ' ' first_line with
           | [ "GET"; uri; _ ] ->
               let path, query = parse_query uri in
               if path = "/" then http_response "200 OK" "text/html; charset=utf-8" html
               else if path = "/api/sessions" then
                 http_response "200 OK" "application/json" (sessions_json root)
               else if path = "/api/snapshot" then
                 let status, headers, body = snapshot_from_files root query in
                 http_response ~headers status "application/octet-stream" body
               else http_response "404 Not Found" "text/plain" "not found\n"
           | _ -> http_response "400 Bad Request" "text/plain" "bad request\n"
         in
         write_all client response
       with _ -> ());
      Unix.close client
  done;
  Unix.close server

let tui_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  if not (Unix.isatty Unix.stdin) then list_command cli
  else
    let original = set_terminal_raw Unix.stdin in
    let selected = ref 0 and running = ref true in
    Fun.protect
      ~finally:(fun () -> Unix.tcsetattr Unix.stdin TCSANOW original; print_string "\027[?25h\027[0m\n"; flush Stdlib.stdout)
      (fun () ->
        print_string "\027[?25l";
        while !running do
          let sessions = session_descriptors root in
          if !selected >= List.length sessions then selected := max 0 (List.length sessions - 1);
          Printf.printf "\027[2J\027[H\027[1;37mSOUNIO LOOM\027[0m  durable fleet multiplexer\n";
          Printf.printf "\027[90m%-4s %-14s %-32s %-9s %-8s %s\027[0m\n" "" "AGENT" "LANE" "STATE" "PID" "CURSOR";
          List.iteri
            (fun index (_, values) ->
              let marker = if index = !selected then "\027[36m > " else "   " in
              Printf.printf "%s%-14s %-32s %-9s %-8s %s\027[0m\n" marker
                (table_value values "agent") (table_value values "lane")
                (table_value values "state") (table_value values "harness_pid")
                (string_of_int (file_size (table_value values "output_file"))))
            sessions;
          print_string "\n\027[90mj/k select   enter attach   o observe   r refresh   q quit   detach: Ctrl-]\027[0m\n";
          flush Stdlib.stdout;
          let readable, _, _ = Unix.select [ Unix.stdin ] [] [] 1.0 in
          if readable <> [] then
            let byte = Bytes.create 1 in
            if Unix.read Unix.stdin byte 0 1 = 1 then
              match Bytes.get byte 0 with
              | 'q' -> running := false
              | 'j' -> if !selected + 1 < List.length sessions then incr selected
              | 'k' -> if !selected > 0 then decr selected
              | '\r' | '\n' | 'o' as key -> (
                  match List.nth_opt sessions !selected with
                  | None -> ()
                  | Some (_, values) ->
                      Unix.tcsetattr Unix.stdin TCSANOW original;
                      print_string "\027[2J\027[H\027[?25h";
                      flush Stdlib.stdout;
                      let attach_cli =
                        { options = Hashtbl.copy cli.options; flags = Hashtbl.create 2; rest = [] }
                      in
                      Hashtbl.replace attach_cli.options "--agent" (table_value values "agent");
                      Hashtbl.replace attach_cli.options "--lane" (table_value values "lane");
                      Hashtbl.replace attach_cli.options "--cursor" "auto";
                      (try stream_command attach_cli (key <> 'o') with Loom_error error -> Printf.eprintf "\nLoom: %s\n%!" error);
                      ignore (set_terminal_raw Unix.stdin);
                      print_string "\027[?25l")
              | _ -> ()
        done)

let usage () =
  Printf.eprintf
    "Sounio Loom %s\n\nCommands:\n  start --agent A --lane L --session-id S --cwd DIR -- COMMAND...\n  status|stop|attach|observe|snapshot --agent A --lane L [options]\n  list|tui|serve [--state-dir DIR]\n  verify-journal --journal PATH\n"
    runtime_version

let arguments_after_command () =
  let values = Array.to_list Sys.argv in
  match values with _program :: _command :: tail -> tail | _ -> []

let main () =
  if Array.length Sys.argv < 2 then (usage (); 2)
  else
    let command = Sys.argv.(1) in
    let booleans = [ "--no-raw"; "--meta"; "--machine"; "--allow-remote" ] in
    let cli = parse_cli booleans (arguments_after_command ()) in
    match command with
    | "runtime-version" ->
        Printf.printf "protocol_version=%d\nruntime_version=%s\nlanguage=OCaml\n" protocol_version runtime_version;
        0
    | "start" -> start_command cli; 0
    | "status" -> status_command cli; 0
    | "wake" -> wake_command cli; 0
    | "stop" -> stop_command cli; 0
    | "attach" -> stream_command cli true; 0
    | "observe" -> stream_command cli false; 0
    | "snapshot" -> snapshot_command cli; 0
    | "list" -> list_command cli; 0
    | "tui" -> tui_command cli; 0
    | "serve" -> serve_http cli; 0
    | "verify-journal" -> verify_command cli; 0
    | "_forge-duplicate-lease" -> forge_duplicate_lease cli; 0
    | _ -> usage (); 2

let () =
  try exit (main ())
  with
  | Loom_error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Sys_error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Unix_error (error, function_name, argument) ->
      Printf.eprintf "error: %s: %s(%s)\n%!" (Unix.error_message error) function_name argument;
      exit 1
