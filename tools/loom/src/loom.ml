open Unix

exception Loom_error of string

let protocol_version = 1
let guardian_protocol_version = 1
let runtime_version = "2026.08.24.7"
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

let base64_encode value =
  Cryptokit.transform_string (Cryptokit.Base64.encode_compact_pad ()) value

let base64_decode value =
  try Cryptokit.transform_string (Cryptokit.Base64.decode ()) value
  with _ -> failf "continuity-signature-invalid-base64"

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

let process_exchange command arguments input =
  let stdin_read, stdin_write = Unix.pipe () in
  let stdout_read, stdout_write = Unix.pipe () in
  Unix.set_close_on_exec stdin_write;
  Unix.set_close_on_exec stdout_read;
  let pid =
    try
      Unix.create_process command arguments stdin_read stdout_write stdout_write
    with error ->
      Unix.close stdin_read;
      Unix.close stdin_write;
      Unix.close stdout_read;
      Unix.close stdout_write;
      raise error
  in
  Unix.close stdin_read;
  Unix.close stdout_write;
  let output = Buffer.create 512 in
  let bytes = Bytes.create 4096 in
  let close_noerr descriptor = try Unix.close descriptor with _ -> () in
  Fun.protect
    ~finally:(fun () ->
      close_noerr stdin_write;
      close_noerr stdout_read)
    (fun () ->
      write_all stdin_write input;
      Unix.close stdin_write;
      let rec drain () =
        match Unix.read stdout_read bytes 0 (Bytes.length bytes) with
        | 0 -> ()
        | count -> Buffer.add_subbytes output bytes 0 count; drain ()
      in
      drain ();
      let status =
        try
          let _, status = Unix.waitpid [] pid in
          Some status
        with Unix_error (ECHILD, _, _) -> None
      in
      let text = trim (Buffer.contents output) in
      match status with
      | None | Some (WEXITED 0) -> text
      | Some (WEXITED code) ->
          failf "command failed: %s rc=%d output=%s" command code text
      | Some (WSIGNALED signal) ->
          failf "command signaled: %s signal=%d output=%s" command signal text
      | Some (WSTOPPED signal) ->
          failf "command stopped: %s signal=%d output=%s" command signal text)

let process_quiet command arguments =
  let null = Unix.openfile "/dev/null" [ O_RDWR ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close null)
    (fun () ->
      let pid = Unix.create_process command arguments null null null in
      match snd (Unix.waitpid [] pid) with WEXITED 0 -> true | _ -> false)

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

type json_value =
  | Json_object of (string * json_value) list
  | Json_array of json_value list
  | Json_string of string
  | Json_number of string
  | Json_bool of bool
  | Json_null

let parse_json value =
  let length = String.length value in
  let index = ref 0 in
  let fail message = failf "invalid-json:%s at=%d" message !index in
  let rec whitespace () =
    if !index < length then
      match value.[!index] with
      | ' ' | '\t' | '\n' | '\r' -> incr index; whitespace ()
      | _ -> ()
  and string_literal () =
    if !index >= length || value.[!index] <> '"' then fail "expected-string";
    incr index;
    let output = Buffer.create 32 in
    let rec loop () =
      if !index >= length then fail "unterminated-string";
      match value.[!index] with
      | '"' -> incr index; Buffer.contents output
      | '\\' ->
          incr index;
          if !index >= length then fail "unterminated-escape";
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
              if !index + 4 > length then fail "short-unicode-escape";
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
          | _ -> fail "unknown-escape");
          loop ()
      | character when Char.code character < 32 -> fail "control-in-string"
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
      if before = !index then fail "expected-number"
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
    then fail ("expected-" ^ literal);
    index := ending;
    parsed
  and item () =
    whitespace ();
    if !index >= length then fail "expected-value";
    match value.[!index] with
    | '{' -> object_literal ()
    | '[' -> array_literal ()
    | '"' -> Json_string (string_literal ())
    | 't' -> keyword "true" (Json_bool true)
    | 'f' -> keyword "false" (Json_bool false)
    | 'n' -> keyword "null" Json_null
    | '-' | '0' .. '9' -> Json_number (number_literal ())
    | _ -> fail "unexpected-token"
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
        whitespace ();
        if !index >= length || value.[!index] <> ':' then fail "expected-colon";
        incr index;
        let member = item () in
        whitespace ();
        if !index < length && value.[!index] = ',' then (
          incr index;
          members ((key, member) :: values))
        else if !index < length && value.[!index] = '}' then (
          incr index;
          Json_object (List.rev ((key, member) :: values)))
        else fail "expected-object-separator"
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
          elements (element :: values))
        else if !index < length && value.[!index] = ']' then (
          incr index;
          Json_array (List.rev (element :: values)))
        else fail "expected-array-separator"
    in
    elements []
  in
  let parsed = item () in
  whitespace ();
  if !index <> length then fail "trailing-data";
  parsed

let json_object_field object_value name =
  match object_value with
  | Json_object fields -> List.assoc_opt name fields
  | _ -> failf "invalid-json:expected-object"

let json_string_field ?default object_value names =
  let rec find = function
    | [] -> Option.value ~default:"" default
    | name :: tail -> (
        match json_object_field object_value name with
        | Some (Json_string value) -> value
        | Some Json_null -> Option.value ~default:"" default
        | Some _ -> failf "invalid-json:%s-must-be-string" name
        | None -> find tail)
  in
  find names

let json_int_field ~default object_value names =
  let rec find = function
    | [] -> default
    | name :: tail -> (
        match json_object_field object_value name with
        | Some (Json_number value) -> (
            try int_of_string value
            with _ -> failf "invalid-json:%s-must-be-integer" name)
        | Some Json_null -> default
        | Some _ -> failf "invalid-json:%s-must-be-integer" name
        | None -> find tail)
  in
  find names

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
  guardian_socket_path : string;
  token_path : string;
  descriptor_path : string;
  guardian_descriptor_path : string;
  lock_path : string;
  guardian_lock_path : string;
  daemon_log_path : string;
  guardian_log_path : string;
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
    guardian_socket_path = Filename.concat socket_root ("guardian-" ^ socket_name);
    token_path = Filename.concat session_dir "capability";
    descriptor_path = Filename.concat session_dir "session.state";
    guardian_descriptor_path = Filename.concat session_dir "guardian.state";
    lock_path = Filename.concat session_dir "daemon.lock";
    guardian_lock_path = Filename.concat session_dir "guardian.lock";
    daemon_log_path = Filename.concat session_dir "daemon.log";
    guardian_log_path = Filename.concat session_dir "guardian.log";
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

let parse_output_span payload =
  match split_on ':' payload with
  | start :: ending :: _ ->
      let parse name value =
        try int_of_string value with _ -> failf "semantic:%s-is-not-an-integer" name
      in
      (parse "output-start" start, parse "output-end" ending)
  | _ -> failf "semantic:invalid-output-span"

let verify_events events =
  let expected_seq = ref 1 in
  let expected_previous = ref (String.make 64 '0') in
  let phase = ref Initial in
  let lease = ref None in
  let output_cursor = ref 0 in
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
      | ("OUTPUT" | "OUTPUT_RECONCILED"), Active ->
          let start, ending = parse_output_span payload in
          if start <> !output_cursor || ending < start then
            failf "semantic:non-contiguous-output expected=%d actual=%d:%d seq=%d"
              !output_cursor start ending event.seq;
          output_cursor := ending
      | "KERNEL_RECOVERED", Active -> lease := None
      | ( "INPUT" | "WAKE" | "RESIZE" | "SIGNAL" | "OBSERVER_ATTACHED"
        | "OBSERVER_DETACHED" ), Active -> ()
      | _, Initial -> failf "semantic:event-before-session-start seq=%d" event.seq
      | _, Exited -> failf "semantic:event-after-session-exit seq=%d" event.seq
      | _ -> failf "semantic:unknown-event kind=%s seq=%d" event.kind event.seq);
      expected_previous := event.hash;
      incr expected_seq)
    events;
  (!phase, !expected_previous)

let semantic_output_cursor events =
  List.fold_left
    (fun cursor (event : journal_event) ->
      match event.kind with
      | "OUTPUT" | "OUTPUT_RECONCILED" ->
          let start, ending = parse_output_span (string_of_hex event.payload_hex) in
          if start <> cursor then
            failf "semantic:non-contiguous-output expected=%d actual=%d" cursor start;
          ending
      | _ -> cursor)
    0 events

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

let resume_journal path =
  let events, phase, digest = load_and_verify_journal path in
  if phase <> Active then failf "cannot recover a non-active semantic journal";
  let descriptor = Unix.openfile path [ O_WRONLY; O_APPEND ] 0o600 in
  Unix.set_close_on_exec descriptor;
  let channel = Unix.out_channel_of_descr descriptor in
  ({ channel; descriptor; seq = List.length events; previous = digest }, events)

let field_escape value =
  let buffer = Buffer.create (String.length value) in
  String.iter
    (fun character ->
      match character with
      | '\t' | '\n' | '\r' | '%' ->
          Buffer.add_string buffer (Printf.sprintf "%%%02X" (Char.code character))
      | _ -> Buffer.add_char buffer character)
    value;
  Buffer.contents buffer

let field_unescape value =
  let buffer = Buffer.create (String.length value) in
  let rec loop index =
    if index < String.length value then
      if value.[index] = '%' && index + 2 < String.length value then (
        Buffer.add_char buffer
          (Char.chr
             ((hex_value value.[index + 1] lsl 4)
             lor hex_value value.[index + 2]));
        loop (index + 3))
      else (
        Buffer.add_char buffer value.[index];
        loop (index + 1))
  in
  loop 0;
  Buffer.contents buffer

let control_line fields = String.concat "\t" fields ^ "\n"

let parse_nonnegative name value =
  let parsed = try int_of_string value with _ -> failf "invalid-%s" name in
  if parsed < 0 then failf "invalid-%s" name;
  parsed

type guardian_client_mode = Guardian_awaiting | Guardian_bridge

type guardian_client = {
  guardian_fd : file_descr;
  guardian_input : Buffer.t;
  mutable guardian_mode : guardian_client_mode;
  mutable guardian_pending : string;
  mutable guardian_pending_offset : int;
}

type guardian = {
  guardian_paths : paths;
  guardian_agent : string;
  guardian_lane : string;
  guardian_session_id : string;
  guardian_cwd : string;
  guardian_command : string array;
  guardian_instance_id : string;
  guardian_output_path : string;
  guardian_journal_path : string;
  guardian_token : string;
  guardian_listener : file_descr;
  guardian_master_fd : file_descr;
  guardian_pid_start : string;
  guardian_harness_pid : int;
  guardian_harness_pid_start : string;
  guardian_started_utc : string;
  guardian_output_channel : out_channel;
  guardian_output_descriptor : file_descr;
  guardian_journal : journal;
  guardian_clients : (file_descr, guardian_client) Hashtbl.t;
  mutable guardian_bridge : file_descr option;
  mutable guardian_output_cursor : int;
  mutable guardian_stopping : bool;
  mutable guardian_harness_exit : int option;
}

let guardian_queue client value =
  let remaining = String.length client.guardian_pending - client.guardian_pending_offset in
  let pending =
    if remaining = 0 then value
    else
      String.sub client.guardian_pending client.guardian_pending_offset remaining ^ value
  in
  if String.length pending > max_pending_bytes then
    failf "guardian bridge exceeded its durable replay window";
  client.guardian_pending <- pending;
  client.guardian_pending_offset <- 0

let guardian_flush client =
  let remaining = String.length client.guardian_pending - client.guardian_pending_offset in
  if remaining > 0 then
    try
      let count =
        Unix.write_substring client.guardian_fd client.guardian_pending
          client.guardian_pending_offset remaining
      in
      client.guardian_pending_offset <- client.guardian_pending_offset + count;
      if client.guardian_pending_offset = String.length client.guardian_pending then (
        client.guardian_pending <- "";
        client.guardian_pending_offset <- 0)
    with Unix_error ((EAGAIN | EWOULDBLOCK), _, _) -> ()

let guardian_descriptor_fields guardian state =
  [
    ("protocol", string_of_int guardian_protocol_version);
    ("runtime_version", runtime_version);
    ("state", state);
    ("agent", guardian.guardian_agent);
    ("lane", guardian.guardian_lane);
    ("session_id", guardian.guardian_session_id);
    ("worktree", guardian.guardian_cwd);
    ("instance_id", guardian.guardian_instance_id);
    ("guardian_pid", string_of_int (Unix.getpid ()));
    ("guardian_pid_start", guardian.guardian_pid_start);
    ("harness_pid", string_of_int guardian.guardian_harness_pid);
    ("harness_pid_start", guardian.guardian_harness_pid_start);
    ("guardian_socket", guardian.guardian_paths.guardian_socket_path);
    ("output_file", guardian.guardian_output_path);
    ("guardian_journal_file", guardian.guardian_journal_path);
    ("output_cursor", string_of_int guardian.guardian_output_cursor);
    ("command", logical_command_name guardian.guardian_command);
    ("argv_digest", command_argv_digest guardian.guardian_command);
    ("started_utc", guardian.guardian_started_utc);
    ( "exit_code",
      match guardian.guardian_harness_exit with
      | Some value -> string_of_int value
      | None -> "" );
  ]

let write_guardian_descriptor guardian state =
  atomic_write guardian.guardian_paths.guardian_descriptor_path
    (descriptor_text (guardian_descriptor_fields guardian state))

let create_unix_listener path =
  (try Unix.unlink path with Unix_error (ENOENT, _, _) -> ());
  let listener = Unix.socket PF_UNIX SOCK_STREAM 0 in
  Unix.set_close_on_exec listener;
  Unix.bind listener (ADDR_UNIX path);
  Unix.chmod path 0o600;
  Unix.listen listener 32;
  Unix.set_nonblock listener;
  listener

let redirect_process_log path =
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  let null = Unix.openfile "/dev/null" [ O_RDONLY ] 0 in
  Unix.dup2 null Unix.stdin;
  Unix.dup2 descriptor Unix.stdout;
  Unix.dup2 descriptor Unix.stderr;
  Unix.close null;
  if descriptor <> Unix.stdout && descriptor <> Unix.stderr then Unix.close descriptor

let guardian_close_client guardian descriptor =
  match Hashtbl.find_opt guardian.guardian_clients descriptor with
  | None -> ()
  | Some client ->
      if client.guardian_mode = Guardian_bridge
         && guardian.guardian_bridge = Some descriptor
      then guardian.guardian_bridge <- None;
      Hashtbl.remove guardian.guardian_clients descriptor;
      (try Unix.close descriptor with _ -> ())

let guardian_output_range guardian cursor limit =
  if cursor < 0 || cursor > guardian.guardian_output_cursor then
    failf "cursor-ahead cursor=%d end=%d" cursor guardian.guardian_output_cursor;
  let length = min limit (guardian.guardian_output_cursor - cursor) in
  let descriptor = Unix.openfile guardian.guardian_output_path [ O_RDONLY ] 0 in
  let bytes = Bytes.create length in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      ignore (Unix.lseek descriptor cursor SEEK_SET);
      let rec fill offset =
        if offset < length then
          let count = Unix.read descriptor bytes offset (length - offset) in
          if count = 0 then failf "guardian output ended before its durable cursor"
          else fill (offset + count)
      in
      fill 0;
      Bytes.unsafe_to_string bytes)

let guardian_status_fields guardian =
  [
    ("state", "active");
    ("agent", guardian.guardian_agent);
    ("lane", guardian.guardian_lane);
    ("session_id", guardian.guardian_session_id);
    ("instance_id", guardian.guardian_instance_id);
    ("guardian_pid", string_of_int (Unix.getpid ()));
    ("guardian_pid_start", guardian.guardian_pid_start);
    ("harness_pid", string_of_int guardian.guardian_harness_pid);
    ("harness_pid_start", guardian.guardian_harness_pid_start);
    ("output_cursor", string_of_int guardian.guardian_output_cursor);
    ("bridge_clients", if guardian.guardian_bridge = None then "0" else "1");
    ("worktree", guardian.guardian_cwd);
    ("command", logical_command_name guardian.guardian_command);
    ("argv_digest", command_argv_digest guardian.guardian_command);
  ]

let guardian_handle_request guardian client line =
  let refuse code = guardian_queue client (control_line [ "ERR"; code ]) in
  match split_on '\t' line with
  | magic :: token :: operation :: arguments
    when magic = Printf.sprintf "GUARD/%d" guardian_protocol_version
         && token = guardian.guardian_token -> (
      match (operation, arguments) with
      | "STATUS", [] ->
          let fields =
            guardian_status_fields guardian
            |> List.map (fun (key, value) -> key ^ "=" ^ field_escape value)
          in
          guardian_queue client (control_line ("OK" :: "STATUS" :: fields))
      | "STREAM", [ cursor ] -> (
          try
            let cursor = parse_nonnegative "guardian-cursor" cursor in
            if guardian.guardian_bridge <> None then failf "kernel-bridge-active";
            let length = guardian.guardian_output_cursor - cursor in
            if length < 0 then
              failf "cursor-ahead cursor=%d end=%d" cursor guardian.guardian_output_cursor;
            if length > max_pending_bytes - max_control_bytes then
              failf "replay-window-too-large";
            let replay = guardian_output_range guardian cursor length in
            guardian.guardian_bridge <- Some client.guardian_fd;
            client.guardian_mode <- Guardian_bridge;
            guardian_queue client
              (control_line
                 [ "OK"; "STREAM"; guardian.guardian_instance_id;
                   string_of_int cursor; string_of_int guardian.guardian_output_cursor;
                   string_of_int length ]);
            guardian_queue client replay
          with Loom_error error -> refuse error)
      | "STOP", [] ->
          guardian_queue client (control_line [ "OK"; "STOPPING" ]);
          guardian_flush client;
          guardian.guardian_stopping <- true
      | "RESIZE", [ cols; rows ] -> (
          try
            let cols = parse_nonnegative "cols" cols in
            let rows = parse_nonnegative "rows" rows in
            if cols < 1 || cols > 1000 || rows < 1 || rows > 1000 then
              failf "invalid-terminal-size";
            set_winsize guardian.guardian_master_fd cols rows;
            ignore
              (append_event guardian.guardian_journal "RESIZE"
                 (Printf.sprintf "%d:%d" cols rows));
            guardian_queue client
              (control_line
                 [ "OK"; "RESIZED"; guardian.guardian_instance_id;
                   string_of_int cols; string_of_int rows ])
          with Loom_error error -> refuse error)
      | "SIGNAL", [ signal ] -> (
          try
            (match signal with
            | "SIGINT" -> write_all guardian.guardian_master_fd "\003"
            | "SIGTERM" ->
                Unix.kill guardian.guardian_harness_pid Sys.sigterm
            | "SIGHUP" -> Unix.kill guardian.guardian_harness_pid Sys.sighup
            | _ -> failf "unsupported-signal");
            ignore (append_event guardian.guardian_journal "SIGNAL" signal);
            guardian_queue client
              (control_line
                 [ "OK"; "SIGNALED"; guardian.guardian_instance_id; signal ])
          with
          | Loom_error error -> refuse error
          | Unix_error _ -> refuse "signal-failed")
      | _ -> refuse "unknown-operation")
  | magic :: _
    when magic <> Printf.sprintf "GUARD/%d" guardian_protocol_version ->
      refuse "protocol-refused"
  | _ -> refuse "authentication-refused"

let guardian_read_client guardian descriptor =
  match Hashtbl.find_opt guardian.guardian_clients descriptor with
  | None -> ()
  | Some client ->
      let bytes = Bytes.create 65536 in
      (try
         let count = Unix.read descriptor bytes 0 (Bytes.length bytes) in
         if count = 0 then guardian_close_client guardian descriptor
         else
           match client.guardian_mode with
           | Guardian_bridge ->
               let value = Bytes.sub_string bytes 0 count in
               ignore
                 (append_event guardian.guardian_journal "INPUT"
                    (Printf.sprintf "%d:%s" count (sha256 value)));
               write_all guardian.guardian_master_fd value
           | Guardian_awaiting ->
               Buffer.add_subbytes client.guardian_input bytes 0 count;
               if Buffer.length client.guardian_input > max_control_bytes then
                 guardian_close_client guardian descriptor
               else
                 let value = Buffer.contents client.guardian_input in
                 (match String.index_opt value '\n' with
                 | None -> ()
                 | Some index ->
                     if index <> String.length value - 1 then
                       guardian_close_client guardian descriptor
                     else
                       guardian_handle_request guardian client
                         (String.sub value 0 index))
       with
      | Unix_error ((EAGAIN | EWOULDBLOCK), _, _) -> ()
      | Unix_error _ -> guardian_close_client guardian descriptor)

let guardian_accept_client guardian =
  try
    let descriptor, _ = Unix.accept guardian.guardian_listener in
    Unix.set_close_on_exec descriptor;
    Unix.set_nonblock descriptor;
    Hashtbl.add guardian.guardian_clients descriptor
      {
        guardian_fd = descriptor;
        guardian_input = Buffer.create 256;
        guardian_mode = Guardian_awaiting;
        guardian_pending = "";
        guardian_pending_offset = 0;
      }
  with Unix_error ((EAGAIN | EWOULDBLOCK), _, _) -> ()

let guardian_read_pty guardian =
  let bytes = Bytes.create 65536 in
  try
    let count = Unix.read guardian.guardian_master_fd bytes 0 (Bytes.length bytes) in
    if count > 0 then (
      let value = Bytes.sub_string bytes 0 count in
      let start = guardian.guardian_output_cursor in
      output_string guardian.guardian_output_channel value;
      flush guardian.guardian_output_channel;
      Unix.fsync guardian.guardian_output_descriptor;
      guardian.guardian_output_cursor <- start + count;
      ignore
        (append_event guardian.guardian_journal "OUTPUT"
           (Printf.sprintf "%d:%d:%s" start guardian.guardian_output_cursor
              (sha256 value)));
      write_guardian_descriptor guardian "active";
      match guardian.guardian_bridge with
      | None -> ()
      | Some descriptor -> (
          match Hashtbl.find_opt guardian.guardian_clients descriptor with
          | None -> guardian.guardian_bridge <- None
          | Some client ->
              (try guardian_queue client value
               with Loom_error _ -> guardian_close_client guardian descriptor)))
  with Unix_error ((EAGAIN | EWOULDBLOCK | EIO), _, _) -> ()

let process_exit_code = function
  | WEXITED code -> code
  | WSIGNALED signal -> 128 + abs signal
  | WSTOPPED signal -> 128 + abs signal

let guardian_child_status guardian =
  match Unix.waitpid [ WNOHANG ] guardian.guardian_harness_pid with
  | 0, _ -> None
  | _, status -> Some status
  | exception Unix_error (ECHILD, _, _) -> Some (WEXITED 0)

let guardian_stop_child guardian =
  if guardian.guardian_harness_exit = None then (
    (try Unix.kill guardian.guardian_harness_pid Sys.sigterm with _ -> ());
    let deadline = Unix.gettimeofday () +. 2.0 in
    let rec wait () =
      match guardian_child_status guardian with
      | Some status -> guardian.guardian_harness_exit <- Some (process_exit_code status)
      | None when Unix.gettimeofday () < deadline -> Unix.sleepf 0.05; wait ()
      | None ->
          (try Unix.kill guardian.guardian_harness_pid Sys.sigkill with _ -> ());
          let _, status = Unix.waitpid [] guardian.guardian_harness_pid in
          guardian.guardian_harness_exit <- Some (process_exit_code status)
    in
    wait ())

let run_guardian paths agent lane session_id cwd command instance_id output_path
    guardian_journal_path =
  let lock = Unix.openfile paths.guardian_lock_path [ O_WRONLY; O_CREAT ] 0o600 in
  Unix.set_close_on_exec lock;
  (try Unix.lockf lock F_TLOCK 0
   with Unix_error _ -> failf "another guardian owns this Loom generation");
  let output_descriptor =
    Unix.openfile output_path [ O_WRONLY; O_CREAT; O_TRUNC ] 0o600
  in
  Unix.set_close_on_exec output_descriptor;
  let output_channel = Unix.out_channel_of_descr output_descriptor in
  let journal = open_journal guardian_journal_path in
  let listener = create_unix_listener paths.guardian_socket_path in
  let child_pid, master_fd = forkpty () in
  if child_pid = 0 then (
    Unix.chdir cwd;
    let environment =
      Array.append (Unix.environment ())
        [| Printf.sprintf "SOUNIO_LOOM_GUARDIAN_SOCKET=%s"
             paths.guardian_socket_path;
           Printf.sprintf "SOUNIO_LOOM_TOKEN_FILE=%s" paths.token_path;
           Printf.sprintf "SOUNIO_LOOM_AGENT=%s" agent;
           Printf.sprintf "SOUNIO_LOOM_LANE=%s" lane;
           Printf.sprintf "SOUNIO_LOOM_SESSION_ID=%s" session_id |]
    in
    Unix.execvpe command.(0) command environment);
  Unix.set_close_on_exec master_fd;
  Unix.set_nonblock master_fd;
  (try set_winsize master_fd 40 140 with _ -> ());
  let guardian =
    {
      guardian_paths = paths;
      guardian_agent = agent;
      guardian_lane = lane;
      guardian_session_id = session_id;
      guardian_cwd = cwd;
      guardian_command = command;
      guardian_instance_id = instance_id;
      guardian_output_path = output_path;
      guardian_journal_path;
      guardian_token = trim (read_file paths.token_path);
      guardian_listener = listener;
      guardian_master_fd = master_fd;
      guardian_pid_start = process_start (Unix.getpid ());
      guardian_harness_pid = child_pid;
      guardian_harness_pid_start = process_start child_pid;
      guardian_started_utc = utc_now ();
      guardian_output_channel = output_channel;
      guardian_output_descriptor = output_descriptor;
      guardian_journal = journal;
      guardian_clients = Hashtbl.create 8;
      guardian_bridge = None;
      guardian_output_cursor = 0;
      guardian_stopping = false;
      guardian_harness_exit = None;
    }
  in
  write_guardian_descriptor guardian "active";
  ignore
    (append_event journal "GUARDIAN_STARTED"
       (Printf.sprintf "%s:%d" instance_id child_pid));
  let signal_stop _ = guardian.guardian_stopping <- true in
  Sys.set_signal Sys.sigterm (Sys.Signal_handle signal_stop);
  Sys.set_signal Sys.sigint (Sys.Signal_handle signal_stop);
  while
    not guardian.guardian_stopping && guardian.guardian_harness_exit = None
  do
    let client_fds =
      Hashtbl.fold
        (fun fd _ values -> fd :: values)
        guardian.guardian_clients []
    in
    let write_fds =
      Hashtbl.fold
        (fun fd client values ->
          if
            client.guardian_pending_offset < String.length client.guardian_pending
          then fd :: values
          else values)
        guardian.guardian_clients []
    in
    let readable, writable, _ =
      Unix.select
        (guardian.guardian_listener :: guardian.guardian_master_fd :: client_fds)
        write_fds [] 0.2
    in
    List.iter
      (fun descriptor ->
        if descriptor = guardian.guardian_listener then
          guardian_accept_client guardian
        else if descriptor = guardian.guardian_master_fd then
          guardian_read_pty guardian
        else guardian_read_client guardian descriptor)
      readable;
    List.iter
      (fun descriptor ->
        match Hashtbl.find_opt guardian.guardian_clients descriptor with
        | Some client ->
            (try guardian_flush client
             with _ -> guardian_close_client guardian descriptor)
        | None -> ())
      writable;
    match guardian_child_status guardian with
    | Some status -> guardian.guardian_harness_exit <- Some (process_exit_code status)
    | None -> ()
  done;
  if guardian.guardian_stopping then guardian_stop_child guardian;
  let clients =
    Hashtbl.fold
      (fun fd _ values -> fd :: values)
      guardian.guardian_clients []
  in
  List.iter (guardian_close_client guardian) clients;
  let code = Option.value ~default:0 guardian.guardian_harness_exit in
  ignore (append_event journal "GUARDIAN_EXITED" (string_of_int code));
  write_guardian_descriptor guardian "exited";
  (try Unix.unlink paths.guardian_socket_path with _ -> ());
  close_out_noerr output_channel;
  close_out_noerr journal.channel;
  Unix.close listener;
  Unix.close master_fd;
  Unix.close lock;
  code

type guardian_phase = Guardian_initial | Guardian_active | Guardian_exited

let verify_guardian_events events =
  let expected_seq = ref 1 in
  let expected_previous = ref (String.make 64 '0') in
  let phase = ref Guardian_initial in
  let output_cursor = ref 0 in
  List.iter
    (fun (event : journal_event) ->
      if event.seq <> !expected_seq then
        failf "guardian-hash:non-contiguous-sequence expected=%d actual=%d"
          !expected_seq event.seq;
      if event.previous <> !expected_previous then
        failf "guardian-hash:previous-digest-mismatch seq=%d" event.seq;
      let expected_hash =
        sha256
          (event_material event.seq event.previous event.utc event.kind
             event.payload_hex)
      in
      if event.hash <> expected_hash then
        failf "guardian-hash:event-digest-mismatch seq=%d" event.seq;
      let payload = string_of_hex event.payload_hex in
      (match (event.kind, !phase) with
      | "GUARDIAN_STARTED", Guardian_initial -> phase := Guardian_active
      | "GUARDIAN_STARTED", _ ->
          failf "guardian-semantic:duplicate-start seq=%d" event.seq
      | "OUTPUT", Guardian_active ->
          let start, ending = parse_output_span payload in
          if start <> !output_cursor || ending < start then
            failf
              "guardian-semantic:non-contiguous-output expected=%d actual=%d:%d seq=%d"
              !output_cursor start ending event.seq;
          output_cursor := ending
      | ("INPUT" | "RESIZE" | "SIGNAL"), Guardian_active -> ()
      | "GUARDIAN_EXITED", Guardian_active -> phase := Guardian_exited
      | _, Guardian_initial ->
          failf "guardian-semantic:event-before-start seq=%d" event.seq
      | _, Guardian_exited ->
          failf "guardian-semantic:event-after-exit seq=%d" event.seq
      | _ ->
          failf "guardian-semantic:unknown-event kind=%s seq=%d" event.kind
            event.seq);
      expected_previous := event.hash;
      incr expected_seq)
    events;
  (!phase, !output_cursor, !expected_previous)

let load_and_verify_guardian_journal path =
  let events =
    read_lines path
    |> List.filter (fun line -> trim line <> "")
    |> List.map parse_event
  in
  if events = [] then failf "guardian journal is empty";
  let phase, cursor, digest = verify_guardian_events events in
  (events, phase, cursor, digest)

let connect_unix path =
  let descriptor = Unix.socket PF_UNIX SOCK_STREAM 0 in
  Unix.set_close_on_exec descriptor;
  try
    Unix.connect descriptor (ADDR_UNIX path);
    descriptor
  with error ->
    Unix.close descriptor;
    raise error

let read_protocol_line descriptor =
  let buffer = Buffer.create 256 in
  let byte = Bytes.create 1 in
  let rec loop () =
    let count = Unix.read descriptor byte 0 1 in
    if count = 0 then failf "connection closed before protocol header";
    let character = Bytes.get byte 0 in
    if character = '\n' then Buffer.contents buffer
    else if Buffer.length buffer >= max_control_bytes then
      failf "protocol header too large"
    else (
      Buffer.add_char buffer character;
      loop ())
  in
  loop ()

let read_protocol_exact descriptor length =
  let bytes = Bytes.create length in
  let rec loop offset =
    if offset < length then
      let count = Unix.read descriptor bytes offset (length - offset) in
      if count = 0 then failf "connection closed during protocol payload"
      else loop (offset + count)
  in
  loop 0;
  Bytes.unsafe_to_string bytes

let guardian_request_line token operation arguments =
  control_line
    (Printf.sprintf "GUARD/%d" guardian_protocol_version :: token :: operation
   :: arguments)

let guardian_parse_ok line operation =
  match split_on '\t' line with
  | "ERR" :: error :: _ -> failf "%s" error
  | "OK" :: actual :: fields when actual = operation -> fields
  | _ -> failf "invalid guardian %s response" operation

let guardian_status_request paths token =
  let descriptor = connect_unix paths.guardian_socket_path in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      write_all descriptor (guardian_request_line token "STATUS" []);
      let fields = guardian_parse_ok (read_protocol_line descriptor) "STATUS" in
      let values = Hashtbl.create 24 in
      List.iter
        (fun field ->
          match String.index_opt field '=' with
          | None -> ()
          | Some index ->
              Hashtbl.replace values (String.sub field 0 index)
                (field_unescape
                   (String.sub field (index + 1)
                      (String.length field - index - 1))))
        fields;
      values)

let guardian_open_stream paths token cursor =
  let descriptor = connect_unix paths.guardian_socket_path in
  write_all descriptor
    (guardian_request_line token "STREAM" [ string_of_int cursor ]);
  match guardian_parse_ok (read_protocol_line descriptor) "STREAM" with
  | [ instance; start; ending; length ] ->
      let start = int_of_string start in
      let ending = int_of_string ending in
      let length = int_of_string length in
      let replay = read_protocol_exact descriptor length in
      Unix.set_nonblock descriptor;
      (descriptor, instance, start, ending, replay)
  | _ ->
      Unix.close descriptor;
      failf "invalid guardian STREAM response fields"

let guardian_stop_request paths token =
  let descriptor = connect_unix paths.guardian_socket_path in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      write_all descriptor (guardian_request_line token "STOP" []);
      ignore (guardian_parse_ok (read_protocol_line descriptor) "STOPPING"))

let guardian_resize_request paths token cols rows =
  let descriptor = connect_unix paths.guardian_socket_path in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      write_all descriptor
        (guardian_request_line token "RESIZE"
           [ string_of_int cols; string_of_int rows ]);
      match guardian_parse_ok (read_protocol_line descriptor) "RESIZED" with
      | [ instance; actual_cols; actual_rows ] ->
          (instance, int_of_string actual_cols, int_of_string actual_rows)
      | _ -> failf "invalid guardian RESIZED response fields")

let guardian_signal_request paths token signal =
  let descriptor = connect_unix paths.guardian_socket_path in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      write_all descriptor (guardian_request_line token "SIGNAL" [ signal ]);
      match guardian_parse_ok (read_protocol_line descriptor) "SIGNALED" with
      | [ instance; actual ] when actual = signal -> instance
      | _ -> failf "invalid guardian SIGNALED response fields")

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
  command_name : string;
  command_digest : string;
  instance_id : string;
  output_path : string;
  journal_path : string;
  token : string;
  listener : file_descr;
  guardian_fd : file_descr;
  guardian_pid : int;
  guardian_pid_start : string;
  guardian_journal_path : string;
  daemon_pid_start : string;
  harness_pid : int;
  harness_pid_start : string;
  started_utc : string;
  journal : journal;
  clients : (file_descr, client) Hashtbl.t;
  mutable next_client : int;
  mutable input_holder : file_descr option;
  mutable output_cursor : int;
  mutable stopping : bool;
  mutable harness_exit : int option;
  mutable next_coord_refresh : float;
  mutable coord_pid : int option;
  mutable crash_at : string option;
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
    ("guardian_pid", string_of_int kernel.guardian_pid);
    ("guardian_pid_start", kernel.guardian_pid_start);
    ("guardian_socket", kernel.paths.guardian_socket_path);
    ("socket", kernel.paths.socket_path);
    ("token_file", kernel.paths.token_path);
    ("output_file", kernel.output_path);
    ("journal_file", kernel.journal_path);
    ("guardian_journal_file", kernel.guardian_journal_path);
    ("output_cursor", string_of_int kernel.output_cursor);
    ("command", kernel.command_name);
    ("argv_digest", kernel.command_digest);
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
    ("guardian_pid", string_of_int kernel.guardian_pid);
    ("guardian_pid_start", kernel.guardian_pid_start);
    ("output_cursor", string_of_int kernel.output_cursor);
    ("interactive_clients", if kernel.input_holder = None then "0" else "1");
    ("observer_clients", string_of_int !observers);
    ("journal", kernel.journal_path);
    ("output", kernel.output_path);
    ("worktree", kernel.cwd);
    ("command", kernel.command_name);
    ("argv_digest", kernel.command_digest);
  ]

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
            if replay_length > max_pending_bytes - max_control_bytes then
              failf "replay-window-too-large";
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
            write_all kernel.guardian_fd prompt;
            Unix.sleepf 0.025;
            write_all kernel.guardian_fd "\r";
            ignore
              (append_event kernel.journal "WAKE"
                 (Printf.sprintf "%s:%s" message_id (sha256 prompt)));
            queue client
              (control_line [ "OK"; "WAKE"; kernel.instance_id; message_id ])
          with Loom_error error -> refuse error)
      | "RESIZE", [ cols; rows ] -> (
          try
            let cols = parse_nonnegative "cols" cols in
            let rows = parse_nonnegative "rows" rows in
            if cols < 1 || cols > 1000 || rows < 1 || rows > 1000 then
              failf "invalid-terminal-size";
            let instance, actual_cols, actual_rows =
              guardian_resize_request kernel.paths kernel.token cols rows
            in
            if instance <> kernel.instance_id then failf "guardian-identity-mismatch";
            ignore
              (append_event kernel.journal "RESIZE"
                 (Printf.sprintf "%d:%d" actual_cols actual_rows));
            queue client
              (control_line
                 [ "OK"; "RESIZED"; instance; string_of_int actual_cols;
                   string_of_int actual_rows ])
          with Loom_error error -> refuse error)
      | "SIGNAL", [ signal ] -> (
          try
            if not (List.mem signal [ "SIGINT"; "SIGTERM"; "SIGHUP" ]) then
              failf "unsupported-signal";
            let instance = guardian_signal_request kernel.paths kernel.token signal in
            if instance <> kernel.instance_id then failf "guardian-identity-mismatch";
            ignore (append_event kernel.journal "SIGNAL" signal);
            queue client (control_line [ "OK"; "SIGNALED"; instance; signal ])
          with Loom_error error -> refuse error)
      | "STOP", [] ->
          queue client (control_line [ "OK"; "STOPPING" ]);
          flush_client client;
          kernel.stopping <- true
      | "CRASH", [ point ]
        when List.mem point
               [ "now"; "after_guardian_read"; "after_output_journal";
                 "after_broadcast" ] ->
          queue client (control_line [ "OK"; "CRASH_ARMED"; point ]);
          flush_client client;
          if point = "now" then Unix._exit 86 else kernel.crash_at <- Some point
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
               write_all kernel.guardian_fd value
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

let crash_if_armed kernel point =
  if kernel.crash_at = Some point then Unix._exit 86

let read_guardian_stream kernel =
  let bytes = Bytes.create 65536 in
  try
    let count = Unix.read kernel.guardian_fd bytes 0 (Bytes.length bytes) in
    if count = 0 then (
      let values = parse_key_values kernel.paths.guardian_descriptor_path in
      let code =
        try int_of_string (table_value ~default:"255" values "exit_code")
        with _ -> 255
      in
      kernel.harness_exit <- Some code)
    else
      let value = Bytes.sub_string bytes 0 count in
      let start = kernel.output_cursor in
      kernel.output_cursor <- start + count;
      crash_if_armed kernel "after_guardian_read";
      ignore
        (append_event kernel.journal "OUTPUT"
           (Printf.sprintf "%d:%d:%s" start kernel.output_cursor (sha256 value)));
      crash_if_armed kernel "after_output_journal";
      let slow = ref [] in
      Hashtbl.iter
        (fun descriptor client ->
          match client.mode with
          | Observer | Interactive _ ->
              (try queue client value with Loom_error _ -> slow := descriptor :: !slow)
          | Awaiting -> ())
        kernel.clients;
      List.iter (close_client kernel) !slow;
      crash_if_armed kernel "after_broadcast"
  with Unix_error ((EAGAIN | EWOULDBLOCK | EIO), _, _) -> ()

let stop_guardian kernel =
  if kernel.harness_exit = None then (
    (try guardian_stop_request kernel.paths kernel.token with _ -> ());
    let deadline = Unix.gettimeofday () +. 4.0 in
    let rec wait () =
      let values = parse_key_values kernel.paths.guardian_descriptor_path in
      if table_value values "state" = "exited" then
        kernel.harness_exit <-
          Some
            (try int_of_string (table_value ~default:"255" values "exit_code")
             with _ -> 255)
      else if Unix.gettimeofday () < deadline then (
        Unix.sleepf 0.05;
        wait ())
      else (
        (try Unix.kill kernel.guardian_pid Sys.sigkill with _ -> ());
        kernel.harness_exit <- Some 255)
    in
    wait ())

let harness_for_agent agent =
  let value = String.lowercase_ascii agent in
  if starts_with value "claude" then Some "claude"
  else if starts_with value "codex" then Some "codex"
  else if starts_with value "grok" then Some "grok"
  else if starts_with value "cursor" then Some "cursor"
  else if starts_with value "kimi" then Some "kimi"
  else if starts_with value "beagle" then Some "beagle"
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
      process_exit_code status

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
  if Sys.getenv_opt "SOUNIO_LOOM_COORD_AUTO" <> Some "0" then spawn_coordination_refresh kernel;
  let signal_stop _ = kernel.stopping <- true in
  Sys.set_signal Sys.sigterm (Sys.Signal_handle signal_stop);
  Sys.set_signal Sys.sigint (Sys.Signal_handle signal_stop);
  while not kernel.stopping && kernel.harness_exit = None do
    reap_coordination kernel;
    if Sys.getenv_opt "SOUNIO_LOOM_COORD_AUTO" <> Some "0"
       && Unix.gettimeofday () >= kernel.next_coord_refresh
    then spawn_coordination_refresh kernel;
    let client_fds = Hashtbl.fold (fun fd _ values -> fd :: values) kernel.clients [] in
    let write_fds =
      Hashtbl.fold
        (fun fd client values ->
          if client.pending_offset < String.length client.pending then fd :: values else values)
        kernel.clients []
    in
    let readable, writable, _ =
      Unix.select
        (kernel.listener :: kernel.guardian_fd :: client_fds)
        write_fds [] 0.2
    in
    List.iter
      (fun descriptor ->
        if descriptor = kernel.listener then accept_client kernel
        else if descriptor = kernel.guardian_fd then read_guardian_stream kernel
        else read_client kernel descriptor)
      readable;
    List.iter
      (fun descriptor ->
        match Hashtbl.find_opt kernel.clients descriptor with
        | Some client -> (try flush_client client with _ -> close_client kernel descriptor)
        | None -> ())
      writable
  done;
  if kernel.stopping then stop_guardian kernel;
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
  create_unix_listener path

let redirect_daemon_log path =
  redirect_process_log path

let acquire_kernel_lock paths =
  mkdir_p paths.session_dir;
  let lock = Unix.openfile paths.lock_path [ O_WRONLY; O_CREAT ] 0o600 in
  Unix.set_close_on_exec lock;
  (try Unix.lockf lock F_TLOCK 0
   with Unix_error _ -> failf "another Loom kernel owns this lane");
  lock

let launch_guardian paths agent lane session_id cwd command instance_id
    output_path guardian_journal_path kernel_lock =
  (try Unix.unlink paths.guardian_descriptor_path with _ -> ());
  match Unix.fork () with
  | 0 ->
      Unix.close kernel_lock;
      ignore (Unix.setsid ());
      Sys.set_signal Sys.sighup Sys.Signal_ignore;
      Sys.set_signal Sys.sigchld Sys.Signal_default;
      redirect_process_log paths.guardian_log_path;
      let code =
        try
          run_guardian paths agent lane session_id cwd command instance_id
            output_path guardian_journal_path
        with
        | Loom_error error ->
            Printf.eprintf "guardian error: %s\n%!" error;
            1
        | Unix_error (error, function_name, argument) ->
            Printf.eprintf "guardian error: %s: %s(%s)\n%!"
              (Unix.error_message error) function_name argument;
            1
      in
      exit code
  | guardian_pid ->
      let deadline = Unix.gettimeofday () +. 8.0 in
      let rec wait () =
        if Unix.gettimeofday () >= deadline then (
          (try Unix.kill guardian_pid Sys.sigkill with _ -> ());
          failf "Loom guardian did not become ready");
        if Sys.file_exists paths.guardian_descriptor_path then
          let values = parse_key_values paths.guardian_descriptor_path in
          if
            table_value values "state" = "active"
            && table_value values "guardian_pid" = string_of_int guardian_pid
            && table_value values "instance_id" = instance_id
          then values
          else (
            Unix.sleepf 0.05;
            wait ())
        else (
          Unix.sleepf 0.05;
          wait ())
      in
      wait ()

let build_kernel paths agent lane session_id cwd instance_id output_path
    journal_path guardian_journal_path journal semantic_cursor =
  let token = trim (read_file paths.token_path) in
  let guardian_values = guardian_status_request paths token in
  if table_value guardian_values "state" <> "active" then
    failf "guardian is not active";
  if table_value guardian_values "instance_id" <> instance_id then
    failf "guardian generation identity changed";
  let guardian_fd, stream_instance, start_cursor, ending, replay =
    guardian_open_stream paths token semantic_cursor
  in
  if stream_instance <> instance_id || start_cursor <> semantic_cursor then (
    Unix.close guardian_fd;
    failf "guardian stream identity or cursor changed");
  if ending < semantic_cursor || String.length replay <> ending - semantic_cursor then (
    Unix.close guardian_fd;
    failf "guardian replay length does not match its durable cursor");
  if ending > semantic_cursor then
    ignore
      (append_event journal "OUTPUT_RECONCILED"
         (Printf.sprintf "%d:%d:%s" semantic_cursor ending (sha256 replay)));
  let int_field name =
    try int_of_string (table_value guardian_values name)
    with _ -> failf "guardian status omitted %s" name
  in
  let listener = create_listener paths.socket_path in
  {
    paths;
    agent;
    lane;
    session_id;
    cwd;
    command_name = table_value guardian_values "command";
    command_digest = table_value guardian_values "argv_digest";
    instance_id;
    output_path;
    journal_path;
    token;
    listener;
    guardian_fd;
    guardian_pid = int_field "guardian_pid";
    guardian_pid_start = table_value guardian_values "guardian_pid_start";
    guardian_journal_path;
    daemon_pid_start = process_start (Unix.getpid ());
    harness_pid = int_field "harness_pid";
    harness_pid_start = table_value guardian_values "harness_pid_start";
    started_utc = utc_now ();
    journal;
    clients = Hashtbl.create 16;
    next_client = 0;
    input_holder = None;
    output_cursor = ending;
    stopping = false;
    harness_exit = None;
    next_coord_refresh = 0.0;
    coord_pid = None;
    crash_at = None;
  }

let close_kernel kernel lock =
  (try Unix.unlink kernel.paths.socket_path with _ -> ());
  close_out_noerr kernel.journal.channel;
  (try Unix.close kernel.listener with _ -> ());
  (try Unix.close kernel.guardian_fd with _ -> ());
  Unix.close lock

let serve_session paths agent lane session_id cwd command =
  let lock = acquire_kernel_lock paths in
  let instance_id = random_hex 16 in
  let generation_dir = Filename.concat (Filename.concat paths.session_dir "generations") instance_id in
  mkdir_p generation_dir;
  let output_path = Filename.concat generation_dir "output.bin" in
  let journal_path = Filename.concat generation_dir "journal.tsv" in
  let guardian_journal_path = Filename.concat generation_dir "guardian.tsv" in
  ignore
    (launch_guardian paths agent lane session_id cwd command instance_id output_path
       guardian_journal_path lock);
  let journal = open_journal journal_path in
  ignore
    (append_event journal "SESSION_STARTED"
       (Printf.sprintf "%s:%s" instance_id
          (table_value (parse_key_values paths.guardian_descriptor_path)
             "harness_pid")));
  let kernel =
    build_kernel paths agent lane session_id cwd instance_id output_path
      journal_path guardian_journal_path journal 0
  in
  let code =
    try run_kernel kernel
    with error ->
      (try write_descriptor kernel "recoverable" with _ -> ());
      close_kernel kernel lock;
      raise error
  in
  close_kernel kernel lock;
  code

let recover_session paths =
  let descriptor = parse_key_values paths.descriptor_path in
  let agent = table_value descriptor "agent" in
  let lane = table_value descriptor "lane" in
  let session_id = table_value descriptor "session_id" in
  let cwd = table_value descriptor "worktree" in
  let instance_id = table_value descriptor "instance_id" in
  let output_path = table_value descriptor "output_file" in
  let journal_path = table_value descriptor "journal_file" in
  let guardian_journal_path = table_value descriptor "guardian_journal_file" in
  if
    List.exists (( = ) "")
      [ agent; lane; session_id; cwd; instance_id; output_path; journal_path;
        guardian_journal_path ]
  then failf "session descriptor is not recoverable";
  let lock = acquire_kernel_lock paths in
  let journal, events = resume_journal journal_path in
  let semantic_cursor = semantic_output_cursor events in
  let token = trim (read_file paths.token_path) in
  let guardian_values = guardian_status_request paths token in
  if table_value guardian_values "instance_id" <> instance_id then
    failf "guardian identity does not match the semantic journal";
  ignore
    (append_event journal "KERNEL_RECOVERED"
       (Printf.sprintf "%s:%s:%d:%s" instance_id
          (table_value guardian_values "guardian_pid") (Unix.getpid ())
          (process_start (Unix.getpid ()))));
  let kernel =
    build_kernel paths agent lane session_id cwd instance_id output_path
      journal_path guardian_journal_path journal semantic_cursor
  in
  let code =
    try run_kernel kernel
    with error ->
      (try write_descriptor kernel "recoverable" with _ -> ());
      close_kernel kernel lock;
      raise error
  in
  close_kernel kernel lock;
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

let protocol_fields values =
  let table = Hashtbl.create 24 in
  List.iter
    (fun field ->
      match String.index_opt field '=' with
      | None -> ()
      | Some index ->
          Hashtbl.replace table (String.sub field 0 index)
            (field_unescape
               (String.sub field (index + 1)
                  (String.length field - index - 1))))
    values;
  table

let resize_request paths cols rows =
  let token = trim (read_file paths.token_path) in
  let descriptor = connect paths in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      write_all descriptor
        (request_line token "RESIZE" [ string_of_int cols; string_of_int rows ]);
      match parse_ok_header (read_line_fd descriptor) "RESIZED" with
      | [ instance; actual_cols; actual_rows ] ->
          (instance, int_of_string actual_cols, int_of_string actual_rows)
      | _ -> failf "invalid RESIZED response fields")

let signal_request paths signal =
  let token = trim (read_file paths.token_path) in
  let descriptor = connect paths in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      write_all descriptor (request_line token "SIGNAL" [ signal ]);
      match parse_ok_header (read_line_fd descriptor) "SIGNALED" with
      | [ instance; actual ] when actual = signal -> instance
      | _ -> failf "invalid SIGNALED response fields")

let input_request paths data =
  let status = status_request paths |> protocol_fields in
  let cursor = table_value status "output_cursor" |> parse_nonnegative "cursor" in
  let token = trim (read_file paths.token_path) in
  let descriptor = connect paths in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      write_all descriptor
        (request_line token "ATTACH" [ "interactive"; string_of_int cursor ]);
      match parse_ok_header (read_line_fd descriptor) "ATTACHED" with
      | [ instance; start; ending; "interactive" ] ->
          let replay_length = int_of_string ending - int_of_string start in
          if replay_length > 0 then ignore (read_exact descriptor replay_length);
          write_all descriptor data;
          instance
      | _ -> failf "invalid interactive ATTACHED response fields")

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
  let recoverable_guardian_active =
    if not (Sys.file_exists paths.token_path) then false
    else
      try
        let token = trim (read_file paths.token_path) in
        let values = guardian_status_request paths token in
        table_value values "state" = "active"
      with _ -> false
  in
  if recoverable_guardian_active then
    failf "a recoverable Guardian still owns %s/%s; use recover" agent lane;
  atomic_write paths.token_path (random_hex 32 ^ "\n");
  (try Unix.unlink paths.descriptor_path with _ -> ());
  match Unix.fork () with
  | 0 ->
      ignore (Unix.setsid ());
      Sys.set_signal Sys.sighup Sys.Signal_ignore;
      Sys.set_signal Sys.sigchld Sys.Signal_default;
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

let recover_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let agent = required cli "--agent" in
  let lane = required cli "--lane" in
  let paths = session_paths root agent lane in
  (try
     ignore (status_request paths);
     failf "an active Loom kernel already owns %s/%s" agent lane
   with
  | Loom_error message when starts_with message "an active Loom kernel" ->
      raise (Loom_error message)
  | _ -> ());
  let descriptor = parse_key_values paths.descriptor_path in
  if table_value descriptor "agent" <> agent || table_value descriptor "lane" <> lane then
    failf "recover target does not match the persisted lane identity";
  let token = trim (read_file paths.token_path) in
  let deadline = Unix.gettimeofday () +. 4.0 in
  let rec wait_bridge () =
    let values = guardian_status_request paths token in
    if table_value values "bridge_clients" = "0" then values
    else if Unix.gettimeofday () >= deadline then failf "previous kernel bridge is still active"
    else (
      Unix.sleepf 0.05;
      wait_bridge ())
  in
  let guardian_before = wait_bridge () in
  match Unix.fork () with
  | 0 ->
      ignore (Unix.setsid ());
      Sys.set_signal Sys.sighup Sys.Signal_ignore;
      Sys.set_signal Sys.sigchld Sys.Signal_default;
      redirect_daemon_log paths.daemon_log_path;
      let code = recover_session paths in
      exit code
  | daemon_pid ->
      let deadline = Unix.gettimeofday () +. 8.0 in
      let rec wait () =
        if Unix.gettimeofday () >= deadline then failf "recovered Loom kernel did not become ready";
        let values = parse_key_values paths.descriptor_path in
        if
          table_value values "state" = "active"
          && table_value values "daemon_pid" = string_of_int daemon_pid
        then values
        else (
          Unix.sleepf 0.05;
          wait ())
      in
      let values = wait () in
      Printf.printf
        "LOOM_RECOVERED agent=%s lane=%s instance=%s daemon_pid=%d guardian_pid=%s harness_pid=%s cursor=%s\n%!"
        agent lane (table_value values "instance_id") daemon_pid
        (table_value guardian_before "guardian_pid")
        (table_value values "harness_pid") (table_value values "output_cursor")

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

let guardian_status_command cli =
  let _, paths = session_locator cli in
  let token = trim (read_file paths.token_path) in
  let values = guardian_status_request paths token in
  let fields =
    [ "state"; "agent"; "lane"; "session_id"; "instance_id";
      "guardian_pid"; "guardian_pid_start"; "harness_pid";
      "harness_pid_start"; "output_cursor"; "bridge_clients"; "worktree";
      "command"; "argv_digest" ]
    |> List.map (fun key -> key ^ "=" ^ field_escape (table_value values key))
  in
  Printf.printf "LOOM_GUARDIAN_STATUS %s\n%!" (String.concat " " fields)

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

let crash_kernel_command cli =
  let _, paths = session_locator cli in
  let point = required cli "--at" in
  let token = trim (read_file paths.token_path) in
  let descriptor = connect paths in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      write_all descriptor (request_line token "CRASH" [ point ]);
      match parse_ok_header (read_line_fd descriptor) "CRASH_ARMED" with
      | [ actual ] when actual = point ->
          Printf.printf "LOOM_CRASH_ARMED point=%s\n%!" point
      | _ -> failf "invalid CRASH response fields")

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

let descriptor_process_alive values pid_field start_field =
  let expected_start = table_value values start_field in
  if expected_start = "" then false
  else
    try
      let pid = int_of_string (table_value values pid_field) in
      pid > 0 && process_start pid = expected_start
    with _ -> false

let effective_session_state values =
  match table_value values "state" with
  | "active" | "recoverable" ->
      if descriptor_process_alive values "daemon_pid" "daemon_pid_start" then
        "active"
      else if
        descriptor_process_alive values "guardian_pid" "guardian_pid_start"
      then "recoverable"
      else "lost"
  | state -> state

let session_descriptors root =
  let sessions = Filename.concat root "sessions" in
  if not (Sys.file_exists sessions) then []
  else
    Sys.readdir sessions |> Array.to_list |> List.sort String.compare
    |> List.filter_map (fun name ->
           let path = Filename.concat (Filename.concat sessions name) "session.state" in
           if Sys.file_exists path then
             let values = parse_key_values path in
             Hashtbl.replace values "state" (effective_session_state values);
             Some (path, values)
           else None)
    |> List.sort (fun (_, left) (_, right) ->
           let state_rank values =
             match table_value values "state" with
             | "active" -> 0
             | "recoverable" -> 1
             | "lost" -> 2
             | "exited" -> 3
             | _ -> 4
           in
           let rank = compare (state_rank left) (state_rank right) in
           if rank <> 0 then rank
           else
             compare
               (table_value left "agent" ^ "\000" ^ table_value left "lane")
               (table_value right "agent" ^ "\000" ^ table_value right "lane"))

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

let verify_guardian_command cli =
  let path = required cli "--journal" in
  let events, phase, cursor, digest = load_and_verify_guardian_journal path in
  let phase_name =
    match phase with
    | Guardian_initial -> "initial"
    | Guardian_active -> "active"
    | Guardian_exited -> "exited"
  in
  Printf.printf
    "GUARDIAN_JOURNAL_OK events=%d phase=%s cursor=%d digest=%s\n%!"
    (List.length events) phase_name cursor digest

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
.lane{width:100%;display:grid;grid-template-columns:12px 1fr auto;gap:10px;text-align:left;padding:12px 14px;border:0;border-bottom:1px solid var(--line);background:transparent;color:var(--text);cursor:pointer}.lane:hover,.lane.active{background:#1b2123}.dot{width:8px;height:8px;background:var(--green);margin-top:4px}.lane.recoverable .dot{background:var(--amber)}.lane.exited .dot,.lane.lost .dot{background:var(--red)}.lane small{display:block;color:var(--muted);margin-top:4px}.cursor{color:var(--cyan);font-size:11px}
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
async function refresh(){const list=await fetch('/api/sessions',{cache:'no-store'}).then(r=>r.json());lanes.replaceChildren();if(!selected&&list.length)choose(list.find(s=>s.state==='active')||list[0]);for(const s of list){const b=document.createElement('button'),dot=document.createElement('i'),label=document.createElement('span'),detail=document.createElement('small'),size=document.createElement('em');b.className='lane '+s.state+(selected&&selected.instance_id===s.instance_id?' active':'');dot.className='dot';label.textContent=s.agent;detail.textContent=s.lane;size.className='cursor';size.textContent=s.cursor+' B';label.appendChild(detail);b.append(dot,label,size);b.onclick=()=>choose(s);lanes.appendChild(b)}if(!list.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='No Loom sessions';lanes.appendChild(empty)}if(selected){const now=list.find(s=>s.instance_id===selected.instance_id);if(now)selected=now;await poll()}setTimeout(refresh,1000)}
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

type http_request = {
  http_method : string;
  http_target : string;
  http_headers : (string, string) Hashtbl.t;
  http_body : string;
}

let find_string value needle start =
  let value_length = String.length value and needle_length = String.length needle in
  let rec loop index =
    if index + needle_length > value_length then None
    else if String.sub value index needle_length = needle then Some index
    else loop (index + 1)
  in
  loop start

let read_http_request descriptor =
  let maximum = 1024 * 1024 in
  let buffer = Buffer.create 4096 in
  let bytes = Bytes.create 16384 in
  let rec read_headers () =
    match find_string (Buffer.contents buffer) "\r\n\r\n" 0 with
    | Some ending -> ending
    | None ->
        if Buffer.length buffer >= maximum then failf "http-request-too-large";
        let count = Unix.read descriptor bytes 0 (Bytes.length bytes) in
        if count = 0 then failf "http-request-ended-before-headers";
        Buffer.add_subbytes buffer bytes 0 count;
        read_headers ()
  in
  let header_ending = read_headers () in
  let raw = Buffer.contents buffer in
  let header_text = String.sub raw 0 header_ending in
  let lines = split_on '\n' header_text |> List.map trim in
  let request_line, header_lines =
    match lines with line :: rest -> (line, rest) | [] -> failf "empty-http-request"
  in
  let http_method, http_target =
    match split_on ' ' request_line with
    | method_name :: target :: _ -> (method_name, target)
    | _ -> failf "invalid-http-request-line"
  in
  let headers = Hashtbl.create 16 in
  List.iter
    (fun line ->
      match String.index_opt line ':' with
      | None -> ()
      | Some index ->
          let name =
            String.sub line 0 index |> trim |> String.lowercase_ascii
          in
          let value =
            String.sub line (index + 1) (String.length line - index - 1)
            |> trim
          in
          Hashtbl.replace headers name value)
    header_lines;
  let content_length =
    match Hashtbl.find_opt headers "content-length" with
    | None -> 0
    | Some value -> parse_nonnegative "content-length" value
  in
  if content_length > maximum then failf "http-body-too-large";
  let body_start = header_ending + 4 in
  let initial_body = String.length raw - body_start in
  let rec read_body () =
    if Buffer.length buffer - body_start >= content_length then ()
    else
      let count = Unix.read descriptor bytes 0 (Bytes.length bytes) in
      if count = 0 then failf "http-request-ended-before-body"
      else (Buffer.add_subbytes buffer bytes 0 count; read_body ())
  in
  if initial_body < content_length then read_body ();
  let raw = Buffer.contents buffer in
  {
    http_method;
    http_target;
    http_headers = headers;
    http_body = String.sub raw body_start content_length;
  }

let json_response ?(headers = []) status body =
  http_response ~headers status "application/json; charset=utf-8" body

let beagle_bridge_protocol = "beagle-pty-supervisor-v1"
let beagle_bridge_runtime = "sounio-loom-beagle-bridge-v1"
let beagle_agent = "beagle-workbench"

let beagle_lane_of_pane pane_id = "pane-" ^ hex_of_string pane_id

let beagle_pane_of_lane lane =
  if starts_with lane "pane-" then
    try Some (string_of_hex (String.sub lane 5 (String.length lane - 5)))
    with _ -> None
  else None

let beagle_meta_path paths = Filename.concat paths.session_dir "beagle.meta"

let read_beagle_meta paths = parse_key_values (beagle_meta_path paths)

let write_beagle_meta paths fields =
  atomic_write (beagle_meta_path paths)
    (descriptor_text
       (List.map (fun (key, value) -> (key, field_escape value)) fields))

let beagle_meta_value ?default metadata key =
  table_value ?default metadata key |> field_unescape

let read_range path cursor length =
  if length <= 0 then ""
  else
    let descriptor = Unix.openfile path [ O_RDONLY ] 0 in
    let bytes = Bytes.create length in
    Fun.protect
      ~finally:(fun () -> Unix.close descriptor)
      (fun () ->
        ignore (Unix.lseek descriptor cursor SEEK_SET);
        let rec fill offset =
          if offset < length then
            let count = Unix.read descriptor bytes offset (length - offset) in
            if count > 0 then fill (offset + count)
        in
        fill 0;
        Bytes.unsafe_to_string bytes)

let read_tail path limit =
  let ending = file_size path in
  let start = max 0 (ending - limit) in
  (start, ending, read_range path start (ending - start))

let beagle_paths root pane_id =
  session_paths root beagle_agent (beagle_lane_of_pane pane_id)

let beagle_lineage_path paths =
  Filename.concat paths.session_dir "beagle.lineage.tsv"

type beagle_lineage_record = {
  transition : string;
  predecessor : string;
  successor : string;
  predecessor_semantic_head : string;
  predecessor_guardian_head : string;
}

type beagle_lineage_status = {
  lineage_verified : bool;
  lineage_head : string;
  latest_transition : string;
  predecessor_instance : string;
  predecessor_semantic_head : string;
  predecessor_guardian_head : string;
  transition_count : int;
  pod_resurrection_count : int;
}

type sounio_continuity_status = {
  policy_verified : bool;
  receipt_digest : string;
  runtime_digest : string;
  signature_verified : bool;
  signer_key_id : string;
  signer_principal_id : string;
  predecessor_receipt_digest : string;
  independent_observation_verified : bool;
  observer_key_id : string;
  observer_principal_id : string;
  independent_observation_digest : string;
}

let beagle_generation_journals paths instance =
  let generation_dir =
    Filename.concat (Filename.concat paths.session_dir "generations") instance
  in
  ( Filename.concat generation_dir "journal.tsv",
    Filename.concat generation_dir "guardian.tsv" )

let beagle_generation_evidence paths instance =
  if instance = "" then failf "generation-lineage-empty-predecessor";
  let semantic_path, guardian_path = beagle_generation_journals paths instance in
  let _, semantic_phase, semantic_head =
    load_and_verify_journal semantic_path
  in
  let _, _, _, guardian_head =
    load_and_verify_guardian_journal guardian_path
  in
  (semantic_phase, semantic_head, guardian_head)

let parse_beagle_lineage_payload event =
  if event.kind <> "GENERATION_LINKED" then
    failf "generation-lineage-unknown-event:%s" event.kind;
  match split_on ':' (string_of_hex event.payload_hex) with
  | [ pane_hex; session_hex; transition; predecessor; successor;
      semantic_head; guardian_head ] ->
      (pane_hex, session_hex,
       { transition; predecessor; successor;
         predecessor_semantic_head = semantic_head;
         predecessor_guardian_head = guardian_head })
  | _ -> failf "generation-lineage-invalid-payload"

let load_and_verify_beagle_lineage paths pane_id session_id =
  let path = beagle_lineage_path paths in
  if not (Sys.file_exists path) then ([], String.make 64 '0')
  else
    let events =
      read_lines path
      |> List.filter (fun line -> trim line <> "")
      |> List.map parse_event
    in
    if events = [] then failf "generation-lineage-empty";
    let expected_seq = ref 1 in
    let expected_previous = ref (String.make 64 '0') in
    let expected_predecessor = ref None in
    let records =
      List.map
        (fun (event : journal_event) ->
          if event.seq <> !expected_seq then
            failf "generation-lineage-non-contiguous-sequence";
          if event.previous <> !expected_previous then
            failf "generation-lineage-previous-digest-mismatch";
          let expected_hash =
            sha256
              (event_material event.seq event.previous event.utc event.kind
                 event.payload_hex)
          in
          if event.hash <> expected_hash then
            failf "generation-lineage-event-digest-mismatch";
          let pane_hex, session_hex, record =
            parse_beagle_lineage_payload event
          in
          if pane_hex <> hex_of_string pane_id then
            failf "generation-lineage-pane-mismatch";
          if session_hex <> hex_of_string session_id then
            failf "generation-lineage-session-mismatch";
          if record.predecessor = record.successor then
            failf "generation-lineage-self-cycle";
          if
            record.transition <> "pod-resurrected"
            && record.transition <> "clean-respawn"
          then failf "generation-lineage-transition-invalid";
          (match !expected_predecessor with
          | Some expected when record.predecessor <> expected ->
              failf "generation-lineage-generation-gap"
          | _ -> ());
          let semantic_phase, semantic_head, guardian_head =
            beagle_generation_evidence paths record.predecessor
          in
          if semantic_head <> record.predecessor_semantic_head then
            failf "generation-lineage-semantic-head-mismatch";
          if guardian_head <> record.predecessor_guardian_head then
            failf "generation-lineage-guardian-head-mismatch";
          (match (record.transition, semantic_phase) with
          | "pod-resurrected", Active -> ()
          | "clean-respawn", Exited -> ()
          | _ -> failf "generation-lineage-transition-phase-mismatch");
          expected_predecessor := Some record.successor;
          expected_previous := event.hash;
          incr expected_seq;
          record)
        events
    in
    (records, !expected_previous)

let beagle_preflight_transition paths pane_id session_id predecessor =
  try
    let metadata = read_beagle_meta paths in
    let metadata_instance = beagle_meta_value metadata "instance_id" in
    if metadata_instance <> "" && metadata_instance <> predecessor then
      failf "generation-lineage-metadata-mismatch";
    let records, _ =
      load_and_verify_beagle_lineage paths pane_id session_id
    in
    (match List.rev records with
    | latest :: _ when latest.successor <> predecessor ->
        failf "generation-lineage-current-generation-mismatch"
    | _ -> ());
    ignore (beagle_generation_evidence paths predecessor)
  with Loom_error error ->
    failf "generation-lineage-proof-invalid:%s" error

let append_beagle_lineage paths pane_id session_id predecessor successor =
  let semantic_phase, semantic_head, guardian_head =
    beagle_generation_evidence paths predecessor
  in
  let transition =
    match semantic_phase with
    | Active -> "pod-resurrected"
    | Exited -> "clean-respawn"
    | Initial -> failf "generation-lineage-predecessor-not-started"
  in
  let records, previous =
    load_and_verify_beagle_lineage paths pane_id session_id
  in
  match List.rev records with
  | latest :: _
    when latest.predecessor = predecessor && latest.successor = successor ->
      if
        latest.transition <> transition
        || latest.predecessor_semantic_head <> semantic_head
        || latest.predecessor_guardian_head <> guardian_head
      then failf "generation-lineage-idempotence-mismatch";
      latest
  | latest :: _ when latest.successor <> predecessor ->
      failf "generation-lineage-generation-gap"
  | _ ->
      let path = beagle_lineage_path paths in
      let descriptor =
        Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600
      in
      Unix.set_close_on_exec descriptor;
      let channel = Unix.out_channel_of_descr descriptor in
      let journal =
        { channel; descriptor; seq = List.length records; previous }
      in
      Fun.protect
        ~finally:(fun () -> close_out_noerr channel)
        (fun () ->
          let payload =
            String.concat ":"
              [ hex_of_string pane_id; hex_of_string session_id; transition;
                predecessor; successor; semantic_head; guardian_head ]
          in
          ignore (append_event journal "GENERATION_LINKED" payload);
          { transition; predecessor; successor;
            predecessor_semantic_head = semantic_head;
            predecessor_guardian_head = guardian_head })

let beagle_lineage_status paths pane_id session_id current_instance =
  try
    let records, head =
      load_and_verify_beagle_lineage paths pane_id session_id
    in
    match List.rev records with
    | [] ->
        { lineage_verified = true; lineage_head = "";
          latest_transition = "initial"; predecessor_instance = "";
          predecessor_semantic_head = ""; predecessor_guardian_head = "";
          transition_count = 0; pod_resurrection_count = 0 }
    | latest :: _ ->
        if latest.successor <> current_instance then
          failf "generation-lineage-current-generation-mismatch";
        { lineage_verified = true; lineage_head = head;
          latest_transition = latest.transition;
          predecessor_instance = latest.predecessor;
          predecessor_semantic_head = latest.predecessor_semantic_head;
          predecessor_guardian_head = latest.predecessor_guardian_head;
          transition_count = List.length records;
          pod_resurrection_count =
            List.fold_left
              (fun count record ->
                if record.transition = "pod-resurrected" then count + 1
                else count)
              0 records }
  with _ ->
    { lineage_verified = false; lineage_head = "";
      latest_transition = "unverified"; predecessor_instance = "";
      predecessor_semantic_head = ""; predecessor_guardian_head = "";
      transition_count = 0; pod_resurrection_count = 0 }

let sounio_continuity_adapter () =
  match Sys.getenv_opt "SOUNIO_LOOM_CONTINUITY_ADAPTER" with
  | Some path when path <> "" -> path
  | _ ->
      let executable = Unix.realpath Sys.executable_name in
      Filename.concat (Filename.dirname executable)
        "sounio-loom-continuity-runtime"

let sounio_continuity_token domain value =
  let digest = sha256 (domain ^ "\000" ^ value) in
  let bounded = Int64.of_string ("0x" ^ String.sub digest 0 15) in
  Int64.to_string (Int64.add bounded 1L)

let sounio_optional_token domain value =
  if value = "" then "0" else sounio_continuity_token domain value

type continuity_signing =
  | Unsigned_continuity
  | Ed25519_continuity of {
      private_key : string;
      public_key : string;
      key_id : string;
      principal_id : string;
    }

type verified_signed_receipt = {
  signed_receipt_digest : string;
  signed_key_id : string;
  signed_principal_id : string;
  signed_facts : string;
  signed_facts_digest : string;
  signed_adapter_digest : string;
}

type verified_independent_observation = {
  observer_key_id : string;
  observer_principal_id : string;
  subject_signer_key_id : string;
  subject_principal_id : string;
  subject_receipt_digest : string;
  observation_digest : string;
}

let openssl_command () =
  match Sys.getenv_opt "SOUNIO_LOOM_OPENSSL" with
  | Some path when path <> "" -> path
  | _ -> "/usr/bin/openssl"

let continuity_key_path label path =
  if path = "" || not (Sys.file_exists path) then
    failf "sounio-continuity-%s-key-missing:%s" label path;
  let resolved = Unix.realpath path in
  if (Unix.stat resolved).st_kind <> S_REG then
    failf "sounio-continuity-%s-key-not-regular:%s" label resolved;
  resolved

let ed25519_principal_id public_key =
  let der_path = Filename.temp_file "loom-public-" ".der" in
  Fun.protect
    ~finally:(fun () -> try Sys.remove der_path with _ -> ())
    (fun () ->
      let openssl = openssl_command () in
      let arguments =
        [| openssl; "pkey"; "-pubin"; "-in"; public_key; "-outform";
           "DER"; "-out"; der_path |]
      in
      if not (process_quiet openssl arguments) then
        failf "sounio-continuity-public-key-canonicalization-failed";
      sha256 (read_file der_path))

let continuity_signing () =
  let required =
    match Sys.getenv_opt "SOUNIO_LOOM_REQUIRE_SIGNED_RECEIPTS" with
    | None | Some "" | Some "0" | Some "false" -> false
    | Some "1" | Some "true" -> true
    | Some value -> failf "sounio-continuity-invalid-signing-requirement:%s" value
  in
  match
    ( Sys.getenv_opt "SOUNIO_LOOM_SIGNING_KEY",
      Sys.getenv_opt "SOUNIO_LOOM_VERIFY_KEY" )
  with
  | None, None when not required -> Unsigned_continuity
  | Some private_key, Some public_key when private_key <> "" && public_key <> "" ->
      let private_key = continuity_key_path "private" private_key in
      let public_key = continuity_key_path "public" public_key in
      let openssl = openssl_command () in
      if not (Sys.file_exists openssl) then
        failf "sounio-continuity-openssl-missing:%s" openssl;
      Ed25519_continuity
        { private_key; public_key; key_id = sha256 (read_file public_key);
          principal_id = ed25519_principal_id public_key }
  | _ -> failf "sounio-continuity-signing-keypair-incomplete"

let independent_observer_required () =
  match Sys.getenv_opt "SOUNIO_LOOM_REQUIRE_INDEPENDENT_OBSERVER" with
  | None | Some "" | Some "0" | Some "false" -> false
  | Some "1" | Some "true" -> true
  | Some value ->
      failf "sounio-continuity-invalid-independent-observer-requirement:%s" value

let independent_observer_public_key () =
  if not (independent_observer_required ()) then None
  else
    match Sys.getenv_opt "SOUNIO_LOOM_OBSERVER_VERIFY_KEY" with
    | Some path when path <> "" ->
        Some (continuity_key_path "observer-public" path)
    | _ -> failf "sounio-continuity-independent-observer-key-missing"

let remove_noerr path = try Sys.remove path with _ -> ()

let with_continuity_temp_files directory operation =
  let payload_path = Filename.temp_file ~temp_dir:directory "loom-payload-" ".bin" in
  let signature_path = Filename.temp_file ~temp_dir:directory "loom-signature-" ".bin" in
  Fun.protect
    ~finally:(fun () -> remove_noerr payload_path; remove_noerr signature_path)
    (fun () -> operation payload_path signature_path)

let ed25519_sign signing directory payload =
  match signing with
  | Unsigned_continuity -> failf "sounio-continuity-signing-not-configured"
  | Ed25519_continuity keys ->
      with_continuity_temp_files directory (fun payload_path signature_path ->
          atomic_write payload_path payload;
          let openssl = openssl_command () in
          let arguments =
            [| openssl; "pkeyutl"; "-sign"; "-rawin"; "-inkey";
               keys.private_key; "-in"; payload_path; "-out"; signature_path |]
          in
          if not (process_quiet openssl arguments) then
            failf "sounio-continuity-ed25519-signature-failed";
          let signature = read_file signature_path in
          if String.length signature <> 64 then
            failf "sounio-continuity-ed25519-signature-size:%d"
              (String.length signature);
          base64_encode signature)

let ed25519_verify public_key directory payload signature_base64 =
  with_continuity_temp_files directory (fun payload_path signature_path ->
      atomic_write payload_path payload;
      atomic_write signature_path (base64_decode signature_base64);
      let openssl = openssl_command () in
      let arguments =
        [| openssl; "pkeyutl"; "-verify"; "-pubin"; "-rawin"; "-inkey";
           public_key; "-in"; payload_path; "-sigfile"; signature_path |]
      in
      process_quiet openssl arguments)

let signed_continuity_payload key_id runtime_digest facts_digest facts verdict =
  Printf.sprintf
    "schema=loom-native-continuity-signed-payload-v1\nalgorithm=ed25519\nkey_id=%s\nadapter_sha256=%s\nfacts_sha256=%s\nfacts=%s\nverdict=%s\n"
    key_id runtime_digest facts_digest facts verdict

let signed_continuity_receipt key_id runtime_digest facts_digest facts verdict
    payload_digest signature =
  Printf.sprintf
    "schema=loom-native-continuity-receipt-v2\nalgorithm=ed25519\nkey_id=%s\nadapter_sha256=%s\nfacts_sha256=%s\nfacts=%s\nverdict=%s\nsigned_payload_sha256=%s\nsignature_base64=%s\n"
    key_id runtime_digest facts_digest facts verdict payload_digest signature

let signed_continuity_expected_verdict facts =
  let values = split_on ' ' facts in
  if List.length values = 15 && List.nth values 14 = "1" then
    Some "SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v2 authenticity=ed25519"
  else if List.length values = 18 && List.nth values 14 = "2" then
    Some "SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v3 authenticity=ed25519+independent-observer"
  else None

let independent_observation_payload observer_key_id observer_principal_id
    subject_signer_key_id subject_principal_id subject_receipt_digest
    subject_facts_digest subject_adapter_digest =
  Printf.sprintf
    "schema=loom-independent-observation-payload-v1\nalgorithm=ed25519\nobserver_key_id=%s\nobserver_principal_id=%s\nsubject_signer_key_id=%s\nsubject_principal_id=%s\nsubject_receipt_sha256=%s\nsubject_facts_sha256=%s\nsubject_adapter_sha256=%s\nobservation=precommitted-receipt-digest\n"
    observer_key_id observer_principal_id subject_signer_key_id
    subject_principal_id subject_receipt_digest subject_facts_digest
    subject_adapter_digest

let independent_observation_receipt observer_key_id observer_principal_id
    subject_signer_key_id subject_principal_id subject_receipt_digest
    subject_facts_digest subject_adapter_digest payload_digest signature =
  Printf.sprintf
    "schema=loom-independent-observation-attestation-v1\nalgorithm=ed25519\nobserver_key_id=%s\nobserver_principal_id=%s\nsubject_signer_key_id=%s\nsubject_principal_id=%s\nsubject_receipt_sha256=%s\nsubject_facts_sha256=%s\nsubject_adapter_sha256=%s\nobservation=precommitted-receipt-digest\nsigned_payload_sha256=%s\nsignature_base64=%s\n"
    observer_key_id observer_principal_id subject_signer_key_id
    subject_principal_id subject_receipt_digest subject_facts_digest
    subject_adapter_digest payload_digest signature

let verify_signed_continuity_receipt ~adapter ~runtime_digest ~public_key path =
  if not (Sys.file_exists path) then
    failf "sounio-continuity-predecessor-receipt-missing:%s" path;
  let stored = read_file path in
  let fields = parse_key_values path in
  let schema = table_value fields "schema" in
  let algorithm = table_value fields "algorithm" in
  let key_id = table_value fields "key_id" in
  let stored_adapter = table_value fields "adapter_sha256" in
  let facts_digest = table_value fields "facts_sha256" in
  let facts = table_value fields "facts" in
  let verdict = table_value fields "verdict" in
  let payload_digest = table_value fields "signed_payload_sha256" in
  let signature = table_value fields "signature_base64" in
  let expected_key_id = sha256 (read_file public_key) in
  let payload =
    signed_continuity_payload key_id stored_adapter facts_digest facts verdict
  in
  let canonical =
    signed_continuity_receipt key_id stored_adapter facts_digest facts verdict
      payload_digest signature
  in
  let expected_verdict = signed_continuity_expected_verdict facts in
  if schema <> "loom-native-continuity-receipt-v2"
     || algorithm <> "ed25519" || key_id <> expected_key_id
     || stored_adapter <> runtime_digest || facts = ""
     || facts_digest <> sha256 (facts ^ "\n")
     || expected_verdict = None
     || verdict <> Option.value ~default:"" expected_verdict
     || payload_digest <> sha256 payload || signature = "" || stored <> canonical
  then failf "sounio-continuity-signed-receipt-mismatch";
  if not (ed25519_verify public_key (Filename.dirname path) payload signature) then
    failf "sounio-continuity-signature-invalid";
  let replayed =
    try process_exchange adapter [| adapter |] (facts ^ "\n")
    with Loom_error error -> failf "sounio-continuity-replay-refused:%s" error
  in
  if replayed <> verdict then
    failf "sounio-continuity-replay-mismatch:%s" replayed;
  { signed_receipt_digest = sha256 stored; signed_key_id = key_id;
    signed_principal_id = ed25519_principal_id public_key;
    signed_facts = facts; signed_facts_digest = facts_digest;
    signed_adapter_digest = stored_adapter }

let verify_independent_observation_attestation ~subject ~subject_public_key
    ~observer_public_key path =
  if not (Sys.file_exists path) then
    failf "sounio-continuity-independent-observation-missing:%s" path;
  let stored = read_file path in
  let fields = parse_key_values path in
  let schema = table_value fields "schema" in
  let algorithm = table_value fields "algorithm" in
  let observer_key_id = table_value fields "observer_key_id" in
  let observer_principal_id = table_value fields "observer_principal_id" in
  let subject_signer_key_id = table_value fields "subject_signer_key_id" in
  let subject_principal_id = table_value fields "subject_principal_id" in
  let subject_receipt_digest = table_value fields "subject_receipt_sha256" in
  let subject_facts_digest = table_value fields "subject_facts_sha256" in
  let subject_adapter_digest = table_value fields "subject_adapter_sha256" in
  let observation = table_value fields "observation" in
  let payload_digest = table_value fields "signed_payload_sha256" in
  let signature = table_value fields "signature_base64" in
  let expected_observer_key_id = sha256 (read_file observer_public_key) in
  let expected_observer_principal_id = ed25519_principal_id observer_public_key in
  let expected_subject_principal_id = ed25519_principal_id subject_public_key in
  let payload =
    independent_observation_payload observer_key_id observer_principal_id
      subject_signer_key_id subject_principal_id subject_receipt_digest
      subject_facts_digest subject_adapter_digest
  in
  let canonical =
    independent_observation_receipt observer_key_id observer_principal_id
      subject_signer_key_id subject_principal_id subject_receipt_digest
      subject_facts_digest subject_adapter_digest payload_digest signature
  in
  if schema <> "loom-independent-observation-attestation-v1"
     || algorithm <> "ed25519" || observation <> "precommitted-receipt-digest"
     || observer_key_id <> expected_observer_key_id
     || observer_principal_id <> expected_observer_principal_id
     || subject_signer_key_id <> subject.signed_key_id
     || subject_principal_id <> subject.signed_principal_id
     || subject_principal_id <> expected_subject_principal_id
     || subject_receipt_digest <> subject.signed_receipt_digest
     || subject_facts_digest <> subject.signed_facts_digest
     || subject_adapter_digest <> subject.signed_adapter_digest
     || payload_digest <> sha256 payload || signature = "" || stored <> canonical
  then failf "sounio-continuity-independent-observation-mismatch";
  if not
       (ed25519_verify observer_public_key (Filename.dirname path) payload signature)
  then failf "sounio-continuity-independent-observation-signature-invalid";
  { observer_key_id; observer_principal_id; subject_signer_key_id;
    subject_principal_id; subject_receipt_digest;
    observation_digest = sha256 stored }

let verify_independent_pre_spawn_admission paths predecessor =
  if not (independent_observer_required ()) then ()
  else
    let adapter = Unix.realpath (sounio_continuity_adapter ()) in
    let runtime_digest = sha256 (read_file adapter) in
    let signer_public_key =
      match continuity_signing () with
      | Unsigned_continuity ->
          failf "sounio-continuity-independent-observer-requires-signed-receipts"
      | Ed25519_continuity keys -> keys.public_key
    in
    let observer_public_key =
      match independent_observer_public_key () with
      | Some path -> path
      | None -> failf "sounio-continuity-independent-observer-key-missing"
    in
    let predecessor_dir =
      Filename.concat (Filename.concat paths.session_dir "generations") predecessor
    in
    let predecessor_path =
      Filename.concat predecessor_dir "sounio-continuity.receipt"
    in
    let subject =
      verify_signed_continuity_receipt ~adapter ~runtime_digest
        ~public_key:signer_public_key predecessor_path
    in
    let facts = split_on ' ' subject.signed_facts in
    if (List.length facts <> 15 && List.length facts <> 18)
       || List.nth facts 2 <> sounio_continuity_token "generation" predecessor
    then failf "sounio-continuity-pre-spawn-predecessor-splice";
    let observation =
      verify_independent_observation_attestation ~subject
        ~subject_public_key:signer_public_key ~observer_public_key
        (Filename.concat predecessor_dir
           "sounio-continuity.observer-attestation")
    in
    let frame =
      String.concat " "
        [ "9003";
          sounio_continuity_token "predecessor-receipt"
            subject.signed_receipt_digest;
          sounio_continuity_token "principal-authority"
            subject.signed_principal_id;
          sounio_continuity_token "principal-authority"
            observation.observer_principal_id;
          sounio_continuity_token "independent-observation"
            observation.observation_digest ]
      ^ "\n"
    in
    let verdict =
      try process_exchange adapter [| adapter |] frame
      with Loom_error error ->
        failf "sounio-continuity-pre-spawn-policy-refused:%s" error
    in
    if verdict
       <> "SOUNIO_CONTINUITY_PRESPAWN_ACCEPT schema=loom-native-pre-spawn-v1 authority=disjoint-principals"
    then failf "sounio-continuity-pre-spawn-verdict-mismatch:%s" verdict

let verify_predecessor_binding lineage receipt =
  let facts = split_on ' ' receipt.signed_facts in
  if List.length facts <> 15 && List.length facts <> 18 then
    failf "sounio-continuity-predecessor-fact-count";
  let expected_generation =
    sounio_continuity_token "generation" lineage.predecessor_instance
  in
  let authenticity_mode = List.nth facts 14 in
  if List.nth facts 2 <> expected_generation
     || (authenticity_mode <> "1" && authenticity_mode <> "2")
  then
    failf "sounio-continuity-predecessor-receipt-splice"

let verify_sounio_continuity paths pane_id session_id instance fingerprint
    semantic_head guardian_head lineage =
  let transition_kind =
    match lineage.latest_transition with
    | "initial" -> 1
    | "clean-respawn" -> 2
    | "pod-resurrected" -> 3
    | transition -> failf "sounio-continuity-transition-refused:%s" transition
  in
  let evidence_set_token =
    sounio_continuity_token "evidence-set" (pane_id ^ "\000" ^ session_id)
  in
  let adapter = sounio_continuity_adapter () in
  if not (Sys.file_exists adapter) then
    failf "sounio-continuity-adapter-missing:%s" adapter;
  let adapter = Unix.realpath adapter in
  let runtime_digest = sha256 (read_file adapter) in
  let signing = continuity_signing () in
  let independent_required = independent_observer_required () in
  let observer_public_key = independent_observer_public_key () in
  let predecessor_receipt_digest, signer_key_id, signer_principal_id,
      public_key, observer_key_id, observer_principal_id,
      independent_observation_digest =
    match (signing, lineage.predecessor_instance) with
    | Unsigned_continuity, _ when independent_required ->
        failf "sounio-continuity-independent-observer-requires-signed-receipts"
    | Unsigned_continuity, _ -> ("", "", "", "", "", "", "")
    | Ed25519_continuity keys, "" ->
        ("", keys.key_id, keys.principal_id, keys.public_key, "", "", "")
    | Ed25519_continuity keys, predecessor ->
        let predecessor_dir =
          Filename.concat
            (Filename.concat paths.session_dir "generations") predecessor
        in
        let predecessor_path =
          Filename.concat predecessor_dir "sounio-continuity.receipt"
        in
        let receipt =
          verify_signed_continuity_receipt ~adapter ~runtime_digest
            ~public_key:keys.public_key predecessor_path
        in
        verify_predecessor_binding lineage receipt;
        let observer_key_id, observer_principal_id, observation_digest =
          match observer_public_key with
          | None -> ("", "", "")
          | Some observer_key ->
              let observation_path =
                Filename.concat predecessor_dir
                  "sounio-continuity.observer-attestation"
              in
              let observation =
                verify_independent_observation_attestation ~subject:receipt
                  ~subject_public_key:keys.public_key
                  ~observer_public_key:observer_key observation_path
              in
              (observation.observer_key_id, observation.observer_principal_id,
               observation.observation_digest)
        in
        (receipt.signed_receipt_digest, receipt.signed_key_id,
         receipt.signed_principal_id, keys.public_key, observer_key_id,
         observer_principal_id, observation_digest)
  in
  let chain_material =
    String.concat "\000"
      [ pane_id; session_id; instance; fingerprint; semantic_head; guardian_head;
        lineage.lineage_head; lineage.predecessor_instance;
        lineage.predecessor_semantic_head; lineage.predecessor_guardian_head;
        lineage.latest_transition; string_of_int lineage.transition_count;
        string_of_int lineage.pod_resurrection_count;
        predecessor_receipt_digest; signer_principal_id; observer_key_id;
        observer_principal_id;
        independent_observation_digest ]
  in
  let legacy_facts =
    [ evidence_set_token;
      sounio_continuity_token "receipt-chain" chain_material;
      sounio_continuity_token "generation" instance;
      sounio_continuity_token "generation-fingerprint" fingerprint;
      sounio_continuity_token "semantic-head" semantic_head;
      sounio_continuity_token "guardian-head" guardian_head;
      sounio_optional_token "lineage-head" lineage.lineage_head;
      sounio_optional_token "predecessor-generation"
        lineage.predecessor_instance;
      sounio_optional_token "predecessor-semantic-head"
        lineage.predecessor_semantic_head;
      sounio_optional_token "predecessor-guardian-head"
        lineage.predecessor_guardian_head;
      string_of_int transition_kind;
      string_of_int lineage.transition_count;
      string_of_int lineage.pod_resurrection_count ]
  in
  let signed_mode =
    match signing with Unsigned_continuity -> false | Ed25519_continuity _ -> true
  in
  let independently_observed_mode =
    independent_required && lineage.predecessor_instance <> ""
  in
  let facts =
    if independently_observed_mode then
      legacy_facts
      @ [ sounio_continuity_token "predecessor-receipt"
            predecessor_receipt_digest;
          "2";
          sounio_continuity_token "principal-authority" signer_principal_id;
          sounio_continuity_token "principal-authority" observer_principal_id;
          sounio_continuity_token "independent-observation"
            independent_observation_digest ]
    else if signed_mode then
      legacy_facts
      @ [ sounio_optional_token "predecessor-receipt"
            predecessor_receipt_digest;
          "1" ]
    else legacy_facts
  in
  let fact_frame = String.concat " " facts ^ "\n" in
  let verdict =
    try process_exchange adapter [| adapter |] fact_frame
    with Loom_error error -> failf "sounio-continuity-policy-refused:%s" error
  in
  let expected =
    if independently_observed_mode then
      "SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v3 authenticity=ed25519+independent-observer"
    else if signed_mode then
      "SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v2 authenticity=ed25519"
    else "SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v1"
  in
  if verdict <> expected then
    failf "sounio-continuity-verdict-mismatch:%s" verdict;
  let generation_dir =
    Filename.concat (Filename.concat paths.session_dir "generations") instance
  in
  let receipt_path = Filename.concat generation_dir "sounio-continuity.receipt" in
  let fresh_receipt =
    match signing with
    | Unsigned_continuity ->
        Printf.sprintf
          "schema=loom-native-continuity-receipt-v1\nadapter_sha256=%s\nfacts_sha256=%s\nfacts=%s\nverdict=%s\n"
          runtime_digest (sha256 fact_frame) (trim fact_frame) verdict
    | Ed25519_continuity keys ->
        let facts = trim fact_frame in
        let facts_digest = sha256 fact_frame in
        let payload =
          signed_continuity_payload keys.key_id runtime_digest facts_digest facts
            verdict
        in
        let signature = ed25519_sign signing generation_dir payload in
        if not (ed25519_verify keys.public_key generation_dir payload signature) then
          failf "sounio-continuity-signing-keypair-mismatch";
        signed_continuity_receipt keys.key_id runtime_digest facts_digest facts
          verdict (sha256 payload) signature
  in
  let receipt =
    if not (Sys.file_exists receipt_path) then (
      atomic_write receipt_path fresh_receipt;
      fresh_receipt)
    else if signed_mode then (
      let verified =
        verify_signed_continuity_receipt ~adapter ~runtime_digest ~public_key
          receipt_path
      in
      if verified.signed_facts <> trim fact_frame
         || verified.signed_key_id <> signer_key_id
      then failf "sounio-continuity-signed-replay-facts-mismatch";
      read_file receipt_path)
    else
      let stored = read_file receipt_path in
      let fields = parse_key_values receipt_path in
      let schema = table_value fields "schema" in
      let stored_adapter = table_value fields "adapter_sha256" in
      let stored_facts = table_value fields "facts" in
      let stored_facts_digest = table_value fields "facts_sha256" in
      let stored_verdict = table_value fields "verdict" in
      let stored_frame = stored_facts ^ "\n" in
      let canonical =
        Printf.sprintf
          "schema=%s\nadapter_sha256=%s\nfacts_sha256=%s\nfacts=%s\nverdict=%s\n"
          schema stored_adapter stored_facts_digest stored_facts stored_verdict
      in
      if schema <> "loom-native-continuity-receipt-v1"
         || stored_adapter <> runtime_digest || stored_facts = ""
         || stored_facts_digest <> sha256 stored_frame
         || stored_verdict <> expected || stored <> canonical
      then failf "sounio-continuity-receipt-mismatch";
      let replayed =
        try process_exchange adapter [| adapter |] stored_frame
        with Loom_error error ->
          failf "sounio-continuity-replay-refused:%s" error
      in
      if replayed <> expected then
        failf "sounio-continuity-replay-mismatch:%s" replayed;
      stored
  in
  { policy_verified = true; receipt_digest = sha256 receipt; runtime_digest;
    signature_verified = signed_mode; signer_key_id; signer_principal_id;
    predecessor_receipt_digest;
    independent_observation_verified = independently_observed_mode;
    observer_key_id; observer_principal_id; independent_observation_digest }

let verify_continuity_receipt_command cli =
  let receipt_path = Unix.realpath (required cli "--receipt") in
  let public_key =
    continuity_key_path "public" (required cli "--public-key")
  in
  let adapter =
    match optional cli "--adapter" with
    | Some path -> Unix.realpath path
    | None -> Unix.realpath (sounio_continuity_adapter ())
  in
  let runtime_digest = sha256 (read_file adapter) in
  let verified =
    verify_signed_continuity_receipt ~adapter ~runtime_digest ~public_key
      receipt_path
  in
  Printf.printf
    "LOOM_CONTINUITY_RECEIPT_VERIFIED schema=loom-native-continuity-receipt-v2 algorithm=ed25519 key_id=%s receipt_sha256=%s facts_sha256=%s\n%!"
    verified.signed_key_id verified.signed_receipt_digest
    verified.signed_facts_digest

let attest_continuity_receipt_command cli =
  let receipt_path = Unix.realpath (required cli "--receipt") in
  let subject_public_key =
    continuity_key_path "subject-public" (required cli "--subject-public-key")
  in
  let observer_private_key =
    continuity_key_path "observer-private" (required cli "--observer-private-key")
  in
  let observer_public_key =
    continuity_key_path "observer-public" (required cli "--observer-public-key")
  in
  let output_path = required cli "--out" in
  let adapter =
    match optional cli "--adapter" with
    | Some path -> Unix.realpath path
    | None -> Unix.realpath (sounio_continuity_adapter ())
  in
  let runtime_digest = sha256 (read_file adapter) in
  let subject =
    verify_signed_continuity_receipt ~adapter ~runtime_digest
      ~public_key:subject_public_key receipt_path
  in
  let observer_key_id = sha256 (read_file observer_public_key) in
  let observer_principal_id = ed25519_principal_id observer_public_key in
  let signing =
    Ed25519_continuity
      { private_key = observer_private_key; public_key = observer_public_key;
        key_id = observer_key_id; principal_id = observer_principal_id }
  in
  let payload =
    independent_observation_payload observer_key_id observer_principal_id
      subject.signed_key_id subject.signed_principal_id
      subject.signed_receipt_digest subject.signed_facts_digest
      subject.signed_adapter_digest
  in
  let directory = Filename.dirname output_path in
  if not (Sys.file_exists directory) || (Unix.stat directory).st_kind <> S_DIR then
    failf "sounio-continuity-observation-output-directory-missing:%s" directory;
  let signature = ed25519_sign signing directory payload in
  if not (ed25519_verify observer_public_key directory payload signature) then
    failf "sounio-continuity-observer-keypair-mismatch";
  let attestation =
    independent_observation_receipt observer_key_id observer_principal_id
      subject.signed_key_id subject.signed_principal_id
      subject.signed_receipt_digest subject.signed_facts_digest
      subject.signed_adapter_digest (sha256 payload) signature
  in
  if Sys.file_exists output_path then (
    if read_file output_path <> attestation then
      failf "sounio-continuity-independent-observation-output-conflict")
  else atomic_write output_path attestation;
  let verified =
    verify_independent_observation_attestation ~subject ~subject_public_key
      ~observer_public_key output_path
  in
  Printf.printf
    "LOOM_CONTINUITY_INDEPENDENT_OBSERVATION_ATTESTED schema=loom-independent-observation-attestation-v1 observer_key_id=%s observer_principal_id=%s subject_signer_key_id=%s subject_principal_id=%s subject_receipt_sha256=%s observation_sha256=%s\n%!"
    verified.observer_key_id verified.observer_principal_id
    verified.subject_signer_key_id verified.subject_principal_id
    verified.subject_receipt_digest verified.observation_digest

let beagle_descriptor root pane_id =
  let paths = beagle_paths root pane_id in
  if not (Sys.file_exists paths.descriptor_path) then failf "pane-not-found";
  let descriptor = parse_key_values paths.descriptor_path in
  Hashtbl.replace descriptor "state" (effective_session_state descriptor);
  (paths, descriptor)

let beagle_status_json root pane_id =
  let paths, descriptor = beagle_descriptor root pane_id in
  let metadata = read_beagle_meta paths in
  let output_path = table_value descriptor "output_file" in
  let _, cursor, snapshot = read_tail output_path 12000 in
  let loom_state = table_value descriptor "state" in
  let status = if loom_state = "active" then "running" else loom_state in
  let instance = table_value descriptor "instance_id" in
  let session_id = table_value descriptor "session_id" in
  let generation_fingerprint =
    sha256
      (String.concat "\000"
         [ pane_id; session_id; instance; table_value descriptor "harness_pid_start";
           table_value descriptor "argv_digest" ])
  in
  let lineage =
    beagle_lineage_status paths pane_id session_id instance
  in
  let journal_verified, semantic_head, guardian_head, recovery_count =
    try
      let events, _, semantic_head =
        load_and_verify_journal (table_value descriptor "journal_file")
      in
      let _, _, _, guardian_head =
        load_and_verify_guardian_journal
          (table_value descriptor "guardian_journal_file")
      in
      let recovery_count =
        List.fold_left
          (fun count (event : journal_event) ->
            if event.kind = "KERNEL_RECOVERED" then count + 1 else count)
          0 events
      in
      (true, semantic_head, guardian_head, recovery_count)
    with _ -> (false, "", "", 0)
  in
  let continuity =
    if journal_verified && lineage.lineage_verified then
      verify_sounio_continuity paths pane_id session_id instance
        generation_fingerprint semantic_head guardian_head lineage
    else
      { policy_verified = false; receipt_digest = ""; runtime_digest = "";
        signature_verified = false; signer_key_id = ""; signer_principal_id = "";
        predecessor_receipt_digest = "";
        independent_observation_verified = false; observer_key_id = "";
        observer_principal_id = ""; independent_observation_digest = "" }
  in
  Printf.sprintf
    "{\"paneId\":%s,\"sessionId\":%s,\"pid\":%s,\"status\":%s,\"createdAt\":%s,\"updatedAt\":%s,\"cwd\":%s,\"cols\":%s,\"rows\":%s,\"snapshot\":%s,\"supervisorRuntime\":%s,\"supervisorProtocol\":%s,\"loomInstanceId\":%s,\"loomKernelPid\":%s,\"loomGuardianPid\":%s,\"loomState\":%s,\"loomCursor\":%d,\"generationFingerprint\":%s,\"authorityStatus\":{\"owner\":\"loom\",\"journalVerified\":%s,\"semanticJournalHead\":%s,\"guardianJournalHead\":%s,\"kernelRecoveryCount\":%d,\"lineageVerified\":%s,\"generationLineageHead\":%s,\"generationTransition\":%s,\"generationTransitionCount\":%d,\"podResurrectionCount\":%d,\"predecessorInstanceId\":%s,\"predecessorSemanticJournalHead\":%s,\"predecessorGuardianJournalHead\":%s,\"sounioPolicyVerified\":%s,\"sounioPolicyReceipt\":%s,\"sounioPolicyRuntimeDigest\":%s,\"sounioPolicySignatureVerified\":%s,\"sounioPolicySignerKeyId\":%s,\"sounioPolicySignerPrincipalId\":%s,\"sounioPolicyPredecessorReceipt\":%s,\"sounioPolicyIndependentObservationVerified\":%s,\"sounioPolicyObserverKeyId\":%s,\"sounioPolicyObserverPrincipalId\":%s,\"sounioPolicyIndependentObservation\":%s}}"
    (json_quote pane_id) (json_quote session_id)
    (table_value ~default:"0" descriptor "harness_pid") (json_quote status)
    (json_quote
       (beagle_meta_value ~default:(table_value descriptor "started_utc")
          metadata "created_at"))
    (json_quote
       (beagle_meta_value ~default:(table_value descriptor "started_utc")
          metadata "updated_at"))
    (json_quote (table_value descriptor "worktree"))
    (beagle_meta_value ~default:"120" metadata "cols")
    (beagle_meta_value ~default:"34" metadata "rows")
    (json_quote snapshot) (json_quote beagle_bridge_runtime)
    (json_quote beagle_bridge_protocol) (json_quote instance)
    (table_value ~default:"0" descriptor "daemon_pid")
    (table_value ~default:"0" descriptor "guardian_pid")
    (json_quote loom_state) cursor (json_quote generation_fingerprint)
    (if journal_verified then "true" else "false")
    (json_quote semantic_head) (json_quote guardian_head) recovery_count
    (if lineage.lineage_verified then "true" else "false")
    (json_quote lineage.lineage_head)
    (json_quote lineage.latest_transition) lineage.transition_count
    lineage.pod_resurrection_count
    (json_quote lineage.predecessor_instance)
    (json_quote lineage.predecessor_semantic_head)
    (json_quote lineage.predecessor_guardian_head)
    (if continuity.policy_verified then "true" else "false")
    (json_quote continuity.receipt_digest)
    (json_quote continuity.runtime_digest)
    (if continuity.signature_verified then "true" else "false")
    (json_quote continuity.signer_key_id)
    (json_quote continuity.signer_principal_id)
    (json_quote continuity.predecessor_receipt_digest)
    (if continuity.independent_observation_verified then "true" else "false")
    (json_quote continuity.observer_key_id)
    (json_quote continuity.observer_principal_id)
    (json_quote continuity.independent_observation_digest)

let beagle_metadata_for paths pane_id session_id instance cols rows =
  let previous = read_beagle_meta paths in
  let now = utc_now () in
  let previous_instance = beagle_meta_value previous "instance_id" in
  if previous_instance <> "" && previous_instance <> instance then
    ignore
      (append_beagle_lineage paths pane_id session_id previous_instance instance);
  let created_at =
    if previous_instance = instance then
      beagle_meta_value ~default:now previous "created_at"
    else now
  in
  write_beagle_meta paths
    [ ("pane_id", pane_id); ("session_id", session_id);
      ("instance_id", instance); ("created_at", created_at);
      ("updated_at", now); ("cols", string_of_int cols);
      ("rows", string_of_int rows) ]

let beagle_cli root cwd pane_id session_id rest =
  let options = Hashtbl.create 8 in
  Hashtbl.replace options "--agent" beagle_agent;
  Hashtbl.replace options "--lane" (beagle_lane_of_pane pane_id);
  Hashtbl.replace options "--session-id" session_id;
  Hashtbl.replace options "--cwd" cwd;
  Hashtbl.replace options "--state-dir" root;
  { options; flags = Hashtbl.create 0; rest }

let ensure_beagle_pane root body =
  let pane_id = json_string_field body [ "paneId"; "pane_id" ] in
  if pane_id = "" || String.length pane_id > 512 || String.contains pane_id '\000'
  then failf "invalid-pane-id";
  let session_id =
    json_string_field ~default:pane_id body [ "sessionId"; "session_id" ]
  in
  if session_id = "" || String.length session_id > 512
     || String.contains session_id '\000'
  then failf "invalid-session-id";
  let requested_cwd =
    json_string_field ~default:(Unix.getcwd ()) body [ "cwd" ]
  in
  let cwd = Unix.realpath requested_cwd in
  if (Unix.stat cwd).st_kind <> S_DIR then failf "cwd-is-not-a-directory";
  let shell =
    json_string_field
      ~default:(Option.value ~default:"/bin/bash" (Sys.getenv_opt "SHELL"))
      body [ "shell" ]
  in
  if not (Sys.file_exists shell) then failf "shell-not-found";
  let cols = json_int_field ~default:120 body [ "cols" ] in
  let rows = json_int_field ~default:34 body [ "rows" ] in
  if cols < 1 || cols > 1000 || rows < 1 || rows > 1000 then
    failf "invalid-terminal-size";
  let paths = beagle_paths root pane_id in
  let state =
    if Sys.file_exists paths.descriptor_path then
      let descriptor = parse_key_values paths.descriptor_path in
      let existing_session = table_value descriptor "session_id" in
      let existing_cwd = table_value descriptor "worktree" in
      if existing_session <> session_id then failf "pane-identity-conflict";
      if existing_cwd <> cwd then
        failf "pane-cwd-conflict";
      effective_session_state descriptor
    else "absent"
  in
  let cli = beagle_cli root cwd pane_id session_id [] in
  (match state with
  | "active" -> ()
  | "recoverable" -> recover_command cli
  | "absent" ->
      start_command
        { cli with
          rest =
            [ "env"; "TERM=xterm-256color"; "COLORTERM=truecolor";
              "BEAGLE_WORKBENCH=1"; "BEAGLE_PTY_SUPERVISOR=loom";
              "WORKSPACE_ROOT=" ^ cwd; shell; "-l" ] }
  | "lost" | "exited" ->
      let descriptor = parse_key_values paths.descriptor_path in
      let predecessor = table_value descriptor "instance_id" in
      beagle_preflight_transition paths pane_id session_id predecessor;
      verify_independent_pre_spawn_admission paths predecessor;
      start_command
        { cli with
          rest =
            [ "env"; "TERM=xterm-256color"; "COLORTERM=truecolor";
              "BEAGLE_WORKBENCH=1"; "BEAGLE_PTY_SUPERVISOR=loom";
              "WORKSPACE_ROOT=" ^ cwd; shell; "-l" ] }
  | actual -> failf "pane-state-refused:%s" actual);
  let _, descriptor = beagle_descriptor root pane_id in
  let instance, actual_cols, actual_rows = resize_request paths cols rows in
  if instance <> table_value descriptor "instance_id" then
    failf "resize-generation-mismatch";
  beagle_metadata_for paths pane_id session_id instance actual_cols actual_rows;
  beagle_status_json root pane_id

let update_beagle_metadata paths update =
  let metadata = read_beagle_meta paths in
  let value key fallback = beagle_meta_value ~default:fallback metadata key in
  let fields =
    [ ("pane_id", value "pane_id" "");
      ("session_id", value "session_id" "");
      ("instance_id", value "instance_id" "");
      ("created_at", value "created_at" (utc_now ()));
      ("updated_at", utc_now ());
      ("cols", value "cols" "120"); ("rows", value "rows" "34") ]
  in
  write_beagle_meta paths (update fields)

let replace_field name replacement fields =
  List.map (fun (key, value) -> if key = name then (key, replacement) else (key, value)) fields

let websocket_accept key =
  let digest =
    Cryptokit.hash_string (Cryptokit.Hash.sha1 ())
      (key ^ "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")
  in
  Cryptokit.transform_string (Cryptokit.Base64.encode_compact_pad ()) digest

let websocket_frame payload =
  let length = String.length payload in
  let header =
    if length < 126 then Bytes.init 2 (function 0 -> Char.chr 0x81 | _ -> Char.chr length)
    else if length <= 0xffff then
      Bytes.init 4 (function
        | 0 -> Char.chr 0x81
        | 1 -> Char.chr 126
        | 2 -> Char.chr ((length lsr 8) land 0xff)
        | _ -> Char.chr (length land 0xff))
    else
      Bytes.init 10 (function
        | 0 -> Char.chr 0x81
        | 1 -> Char.chr 127
        | index ->
            let shift = (9 - index) * 8 in
            Char.chr ((length lsr shift) land 0xff))
  in
  Bytes.unsafe_to_string header ^ payload

let websocket_send descriptor payload = write_all descriptor (websocket_frame payload)

let beagle_websocket_stream root pane_id descriptor request =
  let key =
    match Hashtbl.find_opt request.http_headers "sec-websocket-key" with
    | Some value -> value
    | None -> failf "websocket-key-required"
  in
  let paths, state = beagle_descriptor root pane_id in
  let output_path = table_value state "output_file" in
  write_all descriptor
    (Printf.sprintf
       "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n"
       (websocket_accept key));
  let _, ending, snapshot = read_tail output_path 12000 in
  let cursor = ref ending in
  websocket_send descriptor
    (Printf.sprintf
       "{\"type\":\"ready\",\"paneId\":%s,\"sessionId\":%s,\"at\":%s,\"supervisorRuntime\":%s,\"supervisorProtocol\":%s,\"loomInstanceId\":%s,\"loomCursor\":%d,\"snapshot\":%s}"
       (json_quote pane_id) (json_quote (table_value state "session_id"))
       (json_quote (utc_now ())) (json_quote beagle_bridge_runtime)
       (json_quote beagle_bridge_protocol)
       (json_quote (table_value state "instance_id")) ending
       (json_quote snapshot));
  let running = ref true and exit_sent = ref false in
  while !running do
    let readable, _, _ = Unix.select [ descriptor ] [] [] 0.05 in
    if readable <> [] then (
      let probe = Bytes.create 256 in
      let count = Unix.read descriptor probe 0 (Bytes.length probe) in
      if count = 0 || count > 0 then running := false)
    else
      let current = file_size output_path in
      if current > !cursor then (
        let length = min 65536 (current - !cursor) in
        let data = read_range output_path !cursor length in
        cursor := !cursor + String.length data;
        websocket_send descriptor
          (Printf.sprintf
             "{\"type\":\"raw_output\",\"paneId\":%s,\"sessionId\":%s,\"at\":%s,\"data\":%s,\"loomCursor\":%d,\"loomInstanceId\":%s}"
             (json_quote pane_id) (json_quote (table_value state "session_id"))
             (json_quote (utc_now ())) (json_quote data) !cursor
             (json_quote (table_value state "instance_id"))));
      let current_state =
        parse_key_values paths.descriptor_path |> effective_session_state
      in
      if current_state = "exited" && !cursor >= file_size output_path
         && not !exit_sent
      then (
        exit_sent := true;
        let guardian = parse_key_values paths.guardian_descriptor_path in
        let exit_code = table_value ~default:"0" guardian "exit_code" in
        websocket_send descriptor
          (Printf.sprintf
             "{\"type\":\"exit\",\"paneId\":%s,\"sessionId\":%s,\"at\":%s,\"exitCode\":%s,\"signal\":\"\",\"detail\":%s}"
             (json_quote pane_id) (json_quote (table_value state "session_id"))
             (json_quote (utc_now ())) exit_code
             (json_quote ("pty exited (" ^ exit_code ^ ")")));
        running := false)
  done

let beagle_pane_route target =
  let path, _ = parse_query target in
  match split_on '/' path with
  | [ ""; "v1"; "panes"; pane_id; action ] ->
      Some (percent_decode pane_id, action)
  | _ -> None

let beagle_handle_http root descriptor request =
  let path, _ = parse_query request.http_target in
  let respond status body = write_all descriptor (json_response status body) in
  if request.http_method = "GET"
     && (path = "/health" || path = "/v1/health")
  then
    let panes =
      session_descriptors root
      |> List.filter (fun (_, values) -> table_value values "agent" = beagle_agent)
      |> List.length
    in
    respond "200 OK"
      (Printf.sprintf
         "{\"status\":\"ok\",\"supervisor\":%s,\"supervisorRuntime\":%s,\"supervisorProtocol\":%s,\"panes\":%d,\"authority\":\"loom\"}"
         (json_quote beagle_bridge_runtime) (json_quote beagle_bridge_runtime)
         (json_quote beagle_bridge_protocol) panes)
  else if request.http_method = "GET" && path = "/v1/panes" then
    let panes =
      session_descriptors root
      |> List.filter_map (fun (_, values) ->
             if table_value values "agent" <> beagle_agent then None
             else
               beagle_pane_of_lane (table_value values "lane")
               |> Option.map (beagle_status_json root))
      |> String.concat ","
    in
    respond "200 OK" ("{\"panes\":[" ^ panes ^ "]}")
  else if request.http_method = "POST" && path = "/v1/spawn" then
    let pane = ensure_beagle_pane root (parse_json request.http_body) in
    respond "200 OK" ("{\"pane\":" ^ pane ^ "}")
  else
    match beagle_pane_route request.http_target with
    | Some (pane_id, "snapshot") when request.http_method = "GET" ->
        respond "200 OK"
          ("{\"pane\":" ^ beagle_status_json root pane_id ^ "}")
    | Some (pane_id, action) when request.http_method = "POST" ->
        let paths, _ = beagle_descriptor root pane_id in
        let body = parse_json request.http_body in
        (match action with
        | "input" ->
            let data = json_string_field ~default:"" body [ "data" ] in
            ignore (input_request paths data)
        | "resize" ->
            let metadata = read_beagle_meta paths in
            let cols =
              json_int_field
                ~default:(int_of_string
                            (beagle_meta_value ~default:"120" metadata "cols"))
                body [ "cols" ]
            in
            let rows =
              json_int_field
                ~default:(int_of_string
                            (beagle_meta_value ~default:"34" metadata "rows"))
                body [ "rows" ]
            in
            let _, actual_cols, actual_rows = resize_request paths cols rows in
            update_beagle_metadata paths (fun fields ->
                fields |> replace_field "cols" (string_of_int actual_cols)
                |> replace_field "rows" (string_of_int actual_rows))
        | "signal" ->
            let signal = json_string_field ~default:"SIGINT" body [ "signal" ] in
            ignore (signal_request paths signal)
        | "terminate" ->
            let token = trim (read_file paths.token_path) in
            guardian_stop_request paths token
        | _ -> failf "unknown-pane-action");
        respond "200 OK"
          ("{\"pane\":" ^ beagle_status_json root pane_id ^ "}")
    | _ -> respond "404 Not Found" "{\"error\":\"not_found\"}"

let beagle_handle_connection root descriptor =
  try
    let request = read_http_request descriptor in
    match beagle_pane_route request.http_target with
    | Some (pane_id, "stream")
      when String.lowercase_ascii
             (Option.value ~default:""
                (Hashtbl.find_opt request.http_headers "upgrade"))
           = "websocket" ->
        beagle_websocket_stream root pane_id descriptor request
    | _ -> beagle_handle_http root descriptor request
  with
  | Loom_error message ->
      let status =
        if message = "pane-not-found" then "404 Not Found"
        else if starts_with message "generation-lineage-proof-invalid:" then
          "409 Conflict"
        else if starts_with message "sounio-continuity-" then "409 Conflict"
        else if
          List.mem message
            [ "pane-identity-conflict"; "pane-cwd-conflict";
              "interactive-client-active" ]
        then "409 Conflict"
        else "400 Bad Request"
      in
      (try
         write_all descriptor
           (json_response status
              (Printf.sprintf "{\"error\":%s}" (json_quote message)))
       with _ -> ())
  | Unix_error _ -> ()
  | error ->
      (try
         write_all descriptor
           (json_response "500 Internal Server Error"
              (Printf.sprintf "{\"error\":%s}"
                 (json_quote (Printexc.to_string error))))
       with _ -> ())

let serve_beagle_bridge cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let bind = optional cli "--bind" |> Option.value ~default:"127.0.0.1" in
  if bind <> "127.0.0.1" && bind <> "localhost"
     && not (flag cli "--allow-remote")
  then failf "remote Beagle bridge bind requires --allow-remote";
  let port = optional cli "--port" |> Option.value ~default:"4372" |> int_of_string in
  let address =
    try Unix.inet_addr_of_string bind
    with _ -> (Unix.gethostbyname bind).h_addr_list.(0)
  in
  let server = Unix.socket PF_INET SOCK_STREAM 0 in
  Unix.setsockopt server SO_REUSEADDR true;
  Unix.bind server (ADDR_INET (address, port));
  Unix.listen server 64;
  let actual_port =
    match Unix.getsockname server with ADDR_INET (_, value) -> value | _ -> port
  in
  let running = ref true in
  let stop _ = running := false in
  Sys.set_signal Sys.sigterm (Sys.Signal_handle stop);
  Sys.set_signal Sys.sigint (Sys.Signal_handle stop);
  Sys.set_signal Sys.sigpipe Sys.Signal_ignore;
  Sys.set_signal Sys.sigchld Sys.Signal_ignore;
  Printf.printf
    "LOOM_BEAGLE_BRIDGE url=http://%s:%d protocol=%s runtime=%s authority=loom\n%!"
    bind actual_port beagle_bridge_protocol beagle_bridge_runtime;
  while !running do
    let readable, _, _ = Unix.select [ server ] [] [] 0.25 in
    if readable <> [] then
      let client, _ = Unix.accept server in
      match Unix.fork () with
      | 0 ->
          Unix.close server;
          Sys.set_signal Sys.sigchld Sys.Signal_default;
          beagle_handle_connection root client;
          Unix.close client;
          Unix._exit 0
      | _ -> Unix.close client
  done;
  Unix.close server

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

type fleet_spec = {
  fleet_slot : string;
  fleet_kind : string;
  fleet_home : string;
  fleet_cwd : string;
  fleet_enabled : bool;
}

let fleet_kinds = [ "claude"; "codex"; "kimi"; "grok"; "cursor"; "empryo" ]

let fleet_directory root = Filename.concat root "fleet"

let fleet_spec_path root slot =
  Filename.concat (fleet_directory root) (slug slot ^ ".state")

let fleet_spec_fields spec =
  [
    ("version", "1");
    ("enabled", if spec.fleet_enabled then "true" else "false");
    ("slot", spec.fleet_slot);
    ("kind", spec.fleet_kind);
    ("home", spec.fleet_home);
    ("cwd", spec.fleet_cwd);
  ]

let validate_fleet_atom name value =
  if value = "" || String.exists (fun character -> character = '\n' || character = '\r') value
  then failf "invalid fleet %s" name

let fleet_spec_of_values path values =
  if table_value values "version" <> "1" then
    failf "fleet catalog version is not supported: %s" path;
  let slot = table_value values "slot" in
  let kind = table_value values "kind" in
  let home = table_value values "home" in
  let cwd = table_value values "cwd" in
  List.iter (fun (name, value) -> validate_fleet_atom name value)
    [ ("slot", slot); ("kind", kind); ("home", home); ("cwd", cwd) ];
  if not (List.mem kind fleet_kinds) then
    failf "unsupported fleet kind %s in %s" kind path;
  if not ((not (Filename.is_relative home)) && Sys.file_exists home && Sys.is_directory home) then
    failf "fleet home is unavailable for slot %s: %s" slot home;
  if not ((not (Filename.is_relative cwd)) && Sys.file_exists cwd && Sys.is_directory cwd) then
    failf "fleet cwd is unavailable for slot %s: %s" slot cwd;
  let enabled =
    match table_value values "enabled" with
    | "true" -> true
    | "false" -> false
    | _ -> failf "invalid enabled state in %s" path
  in
  { fleet_slot = slot; fleet_kind = kind; fleet_home = home; fleet_cwd = cwd;
    fleet_enabled = enabled }

let load_fleet_specs root =
  let directory = fleet_directory root in
  if not (Sys.file_exists directory) then []
  else
    let seen = Hashtbl.create 32 in
    Sys.readdir directory |> Array.to_list |> List.sort String.compare
    |> List.filter (fun name -> Filename.check_suffix name ".state")
    |> List.map (fun name ->
           let path = Filename.concat directory name in
           if (Unix.lstat path).st_kind <> S_REG then
             failf "fleet catalog entry is not a regular file: %s" path;
           let spec = fleet_spec_of_values path (parse_key_values path) in
           if Hashtbl.mem seen spec.fleet_slot then
             failf "duplicate fleet slot in catalog: %s" spec.fleet_slot;
           Hashtbl.add seen spec.fleet_slot path;
           spec)

let fleet_enroll_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let slot = required cli "--slot" in
  let kind = required cli "--kind" in
  let home = Unix.realpath (required cli "--home") in
  validate_fleet_atom "slot" slot;
  if not (List.mem kind fleet_kinds) then failf "unsupported fleet kind: %s" kind;
  let spec =
    { fleet_slot = slot; fleet_kind = kind; fleet_home = home; fleet_cwd = cwd;
      fleet_enabled = true }
  in
  let directory = fleet_directory root in
  mkdir_p directory;
  let path = fleet_spec_path root slot in
  let desired = descriptor_text (fleet_spec_fields spec) in
  if Sys.file_exists path then (
    let existing = parse_key_values path in
    let existing_spec = fleet_spec_of_values path existing in
    if existing_spec <> spec && not (flag cli "--replace") then
      failf "fleet slot %s already has different desired state" slot);
  atomic_write path desired;
  Printf.printf "LOOM_FLEET_ENROLLED slot=%s kind=%s cwd=%s state=enabled\n%!"
    slot kind cwd

let fleet_disable_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let slot = required cli "--slot" in
  let path = fleet_spec_path root slot in
  if not (Sys.file_exists path) then failf "fleet slot is not enrolled: %s" slot;
  let spec = fleet_spec_of_values path (parse_key_values path) in
  if spec.fleet_slot <> slot then failf "fleet slot path identity mismatch: %s" slot;
  let disabled = { spec with fleet_enabled = false } in
  atomic_write path (descriptor_text (fleet_spec_fields disabled));
  Printf.printf "LOOM_FLEET_DISABLED slot=%s\n%!" slot

type captured_process = { captured_code : int; captured_output : string }

let run_captured executable arguments =
  let reader, writer = Unix.pipe () in
  Unix.set_close_on_exec reader;
  match Unix.fork () with
  | 0 ->
      Unix.close reader;
      Unix.dup2 writer Unix.stdout;
      Unix.dup2 writer Unix.stderr;
      if writer <> Unix.stdout && writer <> Unix.stderr then Unix.close writer;
      (try Unix.execv executable (Array.of_list (executable :: arguments))
       with _ -> Unix._exit 127)
  | pid ->
      Unix.close writer;
      let output = Buffer.create 1024 in
      let bytes = Bytes.create 4096 in
      let rec read () =
        match Unix.read reader bytes 0 (Bytes.length bytes) with
        | 0 -> ()
        | count -> Buffer.add_subbytes output bytes 0 count; read ()
        | exception Unix_error (EINTR, _, _) -> read ()
      in
      Fun.protect ~finally:(fun () -> Unix.close reader) read;
      let _, status = Unix.waitpid [] pid in
      { captured_code = process_exit_code status; captured_output = Buffer.contents output }

let fleet_agent_command () =
  match Sys.getenv_opt "SOUNIO_LOOM_FLEET_AGENT_COMMAND" with
  | Some path when Sys.file_exists path -> Unix.realpath path
  | Some path -> failf "configured fleet agent command is unavailable: %s" path
  | None ->
      let sibling =
        Filename.concat (Filename.dirname Sys.executable_name)
          "sounio-fleet-agent-runtime"
      in
      if Sys.file_exists sibling then sibling
      else failf "sounio-fleet-agent-runtime is not installed beside Loom"

let fleet_observed_state output slot =
  let status_prefix = "FLEET_SLOT_STATUS" in
  let rec field name = function
    | [] -> ""
    | token :: rest ->
        let prefix = name ^ "=" in
        if starts_with token prefix then
          String.sub token (String.length prefix) (String.length token - String.length prefix)
        else field name rest
  in
  split_on '\n' output
  |> List.find_map (fun line ->
         let tokens = split_on ' ' (trim line) in
         match tokens with
         | prefix :: fields when prefix = status_prefix && field "slot" fields = slot ->
             Some (field "state" fields)
         | _ -> None)

let fleet_probe helper spec =
  let result =
    run_captured helper
      [ "status"; "--cwd"; spec.fleet_cwd; "--slot"; spec.fleet_slot ]
  in
  let state =
    match fleet_observed_state result.captured_output spec.fleet_slot with
    | Some state -> state
    | None
      when result.captured_code = 0
           && List.exists
                (fun line -> starts_with (trim line) "fleet_slots=0")
                (split_on '\n' result.captured_output) ->
        "absent"
    | None ->
        failf "fleet status failed for %s: %s" spec.fleet_slot
          (trim result.captured_output)
  in
  if state = "drifted" then
    failf "fleet slot %s has identity drift" spec.fleet_slot;
  (state, result)

let fleet_reconcile_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let apply = flag cli "--apply" in
  let helper = fleet_agent_command () in
  let specs = load_fleet_specs root |> List.filter (fun spec -> spec.fleet_enabled) in
  let started = ref 0 and healthy = ref 0 in
  List.iter
    (fun spec ->
      let state, _ = fleet_probe helper spec in
      if state = "active" then (
        incr healthy;
        Printf.printf "LOOM_FLEET slot=%s state=active action=noop\n%!" spec.fleet_slot)
      else if not apply then
        Printf.printf "LOOM_FLEET slot=%s state=%s action=start mode=plan\n%!"
          spec.fleet_slot state
      else (
        let result =
          run_captured helper
            [ "launch-kind"; "--slot"; spec.fleet_slot; "--kind";
              spec.fleet_kind; "--home"; spec.fleet_home; "--cwd";
              spec.fleet_cwd; "--no-attach" ]
        in
        if result.captured_code <> 0 then
          failf "fleet launch failed for %s: %s" spec.fleet_slot
            (trim result.captured_output);
        let after, _ = fleet_probe helper spec in
        if after <> "active" then
          failf "fleet slot %s did not become active after launch" spec.fleet_slot;
        incr started;
        Printf.printf "LOOM_FLEET slot=%s state=active action=started\n%!"
          spec.fleet_slot))
    specs;
  Printf.printf "loom_fleet_slots=%d healthy=%d started=%d mode=%s\n%!"
    (List.length specs) !healthy !started (if apply then "apply" else "plan")

let usage () =
  Printf.eprintf
    "Sounio Loom %s\n\nCommands:\n  start --agent A --lane L --session-id S --cwd DIR -- COMMAND...\n  recover --agent A --lane L --cwd DIR\n  status|guardian-status|stop|attach|observe|snapshot --agent A --lane L [options]\n  crash-kernel --agent A --lane L --at POINT\n  fleet-enroll --slot S --kind K --home DIR --cwd DIR\n  fleet-disable --slot S --cwd DIR\n  fleet-reconcile [--apply] [--state-dir DIR]\n  list|tui|serve [--state-dir DIR]\n  beagle-serve [--bind 127.0.0.1] [--port 4372] [--state-dir DIR]\n  verify-journal|verify-guardian-journal --journal PATH\n  verify-continuity-receipt --receipt PATH --public-key PATH [--adapter PATH]\n  attest-continuity-receipt --receipt PATH --subject-public-key PATH --observer-private-key PATH --observer-public-key PATH --out PATH [--adapter PATH]\n"
    runtime_version

let arguments_after_command () =
  let values = Array.to_list Sys.argv in
  match values with _program :: _command :: tail -> tail | _ -> []

let main () =
  if Array.length Sys.argv < 2 then (usage (); 2)
  else
    let command = Sys.argv.(1) in
    let booleans =
      [ "--no-raw"; "--meta"; "--machine"; "--allow-remote"; "--apply";
        "--replace" ]
    in
    let cli = parse_cli booleans (arguments_after_command ()) in
    match command with
    | "runtime-version" ->
        Printf.printf "protocol_version=%d\nruntime_version=%s\nlanguage=OCaml\n" protocol_version runtime_version;
        0
    | "start" -> start_command cli; 0
    | "recover" -> recover_command cli; 0
    | "status" -> status_command cli; 0
    | "guardian-status" -> guardian_status_command cli; 0
    | "wake" -> wake_command cli; 0
    | "crash-kernel" -> crash_kernel_command cli; 0
    | "stop" -> stop_command cli; 0
    | "attach" -> stream_command cli true; 0
    | "observe" -> stream_command cli false; 0
    | "snapshot" -> snapshot_command cli; 0
    | "list" -> list_command cli; 0
    | "tui" -> tui_command cli; 0
    | "serve" -> serve_http cli; 0
    | "beagle-serve" -> serve_beagle_bridge cli; 0
    | "fleet-enroll" -> fleet_enroll_command cli; 0
    | "fleet-disable" -> fleet_disable_command cli; 0
    | "fleet-reconcile" -> fleet_reconcile_command cli; 0
    | "verify-journal" -> verify_command cli; 0
    | "verify-guardian-journal" -> verify_guardian_command cli; 0
    | "verify-continuity-receipt" -> verify_continuity_receipt_command cli; 0
    | "attest-continuity-receipt" -> attest_continuity_receipt_command cli; 0
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
