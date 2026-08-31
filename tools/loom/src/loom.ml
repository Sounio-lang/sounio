open Unix

exception Loom_error of string

let protocol_version = 1
let guardian_protocol_version = 1
let runtime_version = "2026.08.30.42"
let max_control_bytes = 16 * 1024
let max_kernel_control_bytes = 2 * 1024 * 1024
let max_snapshot_bytes = 1024 * 1024
let max_pending_bytes = 8 * 1024 * 1024
let max_outcome_measurement_bytes = 16 * 1024 * 1024
let max_outcome_receipt_bytes = 16 * 1024
let max_exec_capability_payload_bytes = 512 * 1024

external forkpty : unit -> int * file_descr = "sounio_loom_forkpty"
external set_winsize : file_descr -> int -> int -> unit = "sounio_loom_set_winsize"
external peer_credentials : file_descr -> int * int * int = "sounio_loom_peer_credentials"
external pidfd_open : int -> file_descr option = "sounio_loom_pidfd_open"
external int_of_file_descr : file_descr -> int = "sounio_loom_int_of_file_descr"

let failf format = Printf.ksprintf (fun value -> raise (Loom_error value)) format

let start_ready_timeout () =
  match Sys.getenv_opt "SOUNIO_LOOM_START_READY_TIMEOUT_SECONDS" with
  | None | Some "" -> 30.0
  | Some raw ->
      let seconds =
        try int_of_string raw
        with _ -> failf "invalid Loom start readiness timeout"
      in
      if seconds < 1 || seconds > 300 then
        failf "Loom start readiness timeout must be between 1 and 300 seconds";
      float_of_int seconds

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

let read_file_bounded label limit path =
  let channel = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () ->
      let output = Buffer.create (min limit 4096) in
      let bytes = Bytes.create 65536 in
      let rec loop length =
        let count = input channel bytes 0 (Bytes.length bytes) in
        if count > 0 then (
          let length = length + count in
          if length > limit then failf "%s exceeds %d bytes" label limit;
          Buffer.add_subbytes output bytes 0 count;
          loop length)
      in
      loop 0;
      Buffer.contents output)

let file_size path =
  try (Unix.stat path).st_size with _ -> 0

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

let process_output_all cwd command arguments =
  let reader, writer = Unix.pipe () in
  Unix.set_close_on_exec reader;
  match Unix.fork () with
  | 0 ->
      Unix.close reader;
      Unix.dup2 writer Unix.stdout;
      Unix.dup2 writer Unix.stderr;
      if writer <> Unix.stdout && writer <> Unix.stderr then Unix.close writer;
      (try
         Unix.chdir cwd;
         Unix.execve command arguments (Unix.environment ())
       with _ -> Unix._exit 127)
  | pid ->
      Unix.close writer;
      let output = Buffer.create 4096 in
      let bytes = Bytes.create 16384 in
      let rec read () =
        match Unix.read reader bytes 0 (Bytes.length bytes) with
        | 0 -> ()
        | count -> Buffer.add_subbytes output bytes 0 count; read ()
        | exception Unix_error (EINTR, _, _) -> read ()
      in
      Fun.protect ~finally:(fun () -> Unix.close reader) read;
      let _, status = Unix.waitpid [] pid in
      let code =
        match status with
        | WEXITED value -> value
        | WSIGNALED signal | WSTOPPED signal -> 128 + signal
      in
      (code, Buffer.contents output)

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

let environment_flag name =
  match Sys.getenv_opt name with
  | None | Some "" | Some "0" | Some "false" -> false
  | Some "1" | Some "true" -> true
  | Some value -> failf "invalid-boolean-environment:%s=%s" name value

let observation_authority_required () =
  environment_flag "SOUNIO_LOOM_REQUIRE_OBSERVATION_AUTHORITY"

let journal_authority_required () =
  observation_authority_required ()
  || environment_flag "SOUNIO_LOOM_REQUIRE_JOURNAL_AUTHORITY"

let journal_openssl_command () =
  match Sys.getenv_opt "SOUNIO_LOOM_OPENSSL" with
  | Some path when path <> "" -> path
  | _ -> "/usr/bin/openssl"

let regular_key_path label path =
  if path = "" || not (Sys.file_exists path) then
    failf "sounio-journal-authority-%s-key-missing:%s" label path;
  let resolved = Unix.realpath path in
  if (Unix.stat resolved).st_kind <> S_REG then
    failf "sounio-journal-authority-%s-key-not-regular:%s" label resolved;
  resolved

let journal_crypto_temp_files directory operation =
  let payload_path =
    Filename.temp_file ~temp_dir:directory "loom-authority-payload-" ".bin"
  in
  let signature_path =
    Filename.temp_file ~temp_dir:directory "loom-authority-signature-" ".bin"
  in
  Fun.protect
    ~finally:(fun () ->
      (try Sys.remove payload_path with _ -> ());
      (try Sys.remove signature_path with _ -> ()))
    (fun () -> operation payload_path signature_path)

let journal_principal_id public_key =
  let der_path = Filename.temp_file "loom-authority-public-" ".der" in
  Fun.protect
    ~finally:(fun () -> try Sys.remove der_path with _ -> ())
    (fun () ->
      let openssl = journal_openssl_command () in
      let arguments =
        [| openssl; "pkey"; "-pubin"; "-in"; public_key; "-outform";
           "DER"; "-out"; der_path |]
      in
      if not (process_quiet openssl arguments) then
        failf "sounio-journal-authority-public-key-canonicalization-failed";
      sha256 (read_file der_path))

let journal_sign_with_key private_key directory payload =
  journal_crypto_temp_files directory (fun payload_path signature_path ->
      atomic_write payload_path payload;
      let openssl = journal_openssl_command () in
      let arguments =
        [| openssl; "pkeyutl"; "-sign"; "-rawin"; "-inkey"; private_key;
           "-in"; payload_path; "-out"; signature_path |]
      in
      if not (process_quiet openssl arguments) then
        failf "sounio-journal-authority-signature-failed";
      let signature = read_file signature_path in
      if String.length signature <> 64 then
        failf "sounio-journal-authority-signature-size:%d"
          (String.length signature);
      base64_encode signature)

let journal_verify_with_key public_key directory payload signature_base64 =
  journal_crypto_temp_files directory (fun payload_path signature_path ->
      atomic_write payload_path payload;
      atomic_write signature_path (base64_decode signature_base64);
      let openssl = journal_openssl_command () in
      let arguments =
        [| openssl; "pkeyutl"; "-verify"; "-pubin"; "-rawin"; "-inkey";
           public_key; "-in"; payload_path; "-sigfile"; signature_path |]
      in
      process_quiet openssl arguments)

let valid_sha256 value =
  String.length value = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let positive_epoch label value =
  let parsed =
    try int_of_string value
    with _ -> failf "sounio-journal-authority-%s-not-integer:%s" label value
  in
  if parsed <= 0 then
    failf "sounio-journal-authority-%s-not-positive:%d" label parsed;
  parsed

let journal_authority_expected_epoch () =
  match Sys.getenv_opt "SOUNIO_LOOM_JOURNAL_AUTHORITY_EPOCH" with
  | Some value when value <> "" -> positive_epoch "epoch" value
  | _ -> failf "sounio-journal-authority-epoch-missing"

let journal_authority_epoch_is_revoked epoch =
  match Sys.getenv_opt "SOUNIO_LOOM_JOURNAL_AUTHORITY_REVOKED_EPOCHS" with
  | None | Some "" -> false
  | Some values ->
      split_on ',' values
      |> List.filter (fun value -> trim value <> "")
      |> List.exists (fun value ->
             positive_epoch "revoked-epoch" (trim value) = epoch)

let journal_authority_public_key () =
  match Sys.getenv_opt "SOUNIO_LOOM_JOURNAL_AUTHORITY_VERIFY_KEY" with
  | Some path when path <> "" -> regular_key_path "public" path
  | _ -> failf "sounio-journal-authority-public-key-missing"

let journal_authority_socket () =
  match Sys.getenv_opt "SOUNIO_LOOM_JOURNAL_AUTHORITY_SOCKET" with
  | Some path when path <> "" -> path
  | _ -> failf "sounio-journal-authority-socket-missing"

type journal_authority_member = {
  member_socket : string;
  member_public_key : string;
  member_principal_id : string;
}

type journal_authority_configuration =
  | Single_journal_authority of journal_authority_member
  | Journal_authority_quorum of {
      required : int;
      members : journal_authority_member list;
    }

let indexed_journal_authority_member index =
  let socket_name =
    Printf.sprintf "SOUNIO_LOOM_JOURNAL_AUTHORITY_%d_SOCKET" index
  in
  let key_name =
    Printf.sprintf "SOUNIO_LOOM_JOURNAL_AUTHORITY_%d_VERIFY_KEY" index
  in
  let socket =
    match Sys.getenv_opt socket_name with
    | Some path when path <> "" -> path
    | _ -> failf "sounio-journal-authority-quorum-member-%d-socket-missing" index
  in
  let public_key =
    match Sys.getenv_opt key_name with
    | Some path when path <> "" -> regular_key_path "public" path
    | _ -> failf "sounio-journal-authority-quorum-member-%d-key-missing" index
  in
  { member_socket = socket; member_public_key = public_key;
    member_principal_id = journal_principal_id public_key }

let journal_authority_configuration () =
  match Sys.getenv_opt "SOUNIO_LOOM_JOURNAL_AUTHORITY_QUORUM" with
  | None | Some "" ->
      let public_key = journal_authority_public_key () in
      Single_journal_authority
        { member_socket = journal_authority_socket (); member_public_key = public_key;
          member_principal_id = journal_principal_id public_key }
  | Some value ->
      let required = positive_epoch "quorum" value in
      if required <> 2 then failf "sounio-journal-authority-quorum-must-be-two";
      let members =
        [ indexed_journal_authority_member 1;
          indexed_journal_authority_member 2;
          indexed_journal_authority_member 3 ]
      in
      let principals = List.map (fun member -> member.member_principal_id) members in
      if List.length (List.sort_uniq String.compare principals) <> 3 then
        failf "sounio-journal-authority-quorum-principals-not-disjoint";
      Journal_authority_quorum { required; members }

let journal_authority_payload context_digest epoch principal_id seq previous
    event_hash =
  Printf.sprintf
    "schema=loom-journal-authority-event-v1\nepoch=%d\nauthority_principal_id=%s\njournal_context_sha256=%s\nsequence=%d\nprevious_sha256=%s\nevent_sha256=%s\n"
    epoch principal_id context_digest seq previous event_hash

let journal_authority_signature_is_valid public_key directory payload signature =
  journal_verify_with_key public_key directory payload signature

let journal_authority_exchange socket_path request =
  let descriptor = Unix.socket PF_UNIX SOCK_STREAM 0 in
  Unix.set_close_on_exec descriptor;
  let buffer = Buffer.create 512 in
  let byte = Bytes.create 1 in
  Fun.protect
    ~finally:(fun () -> try Unix.close descriptor with _ -> ())
    (fun () ->
      Unix.connect descriptor (ADDR_UNIX socket_path);
      write_all descriptor request;
      Unix.shutdown descriptor SHUTDOWN_SEND;
      let rec read () =
        let count = Unix.read descriptor byte 0 1 in
        if count = 0 then failf "sounio-journal-authority-empty-response";
        let character = Bytes.get byte 0 in
        if character = '\n' then Buffer.contents buffer
        else if Buffer.length buffer >= max_control_bytes then
          failf "sounio-journal-authority-response-too-large"
        else (Buffer.add_char buffer character; read ())
      in
      read ())

let process_start pid =
  let value = read_file (Printf.sprintf "/proc/%d/stat" pid) in
  let closing =
    try String.rindex value ')' with Not_found -> failf "invalid process stat for pid %d" pid
  in
  let tail = String.sub value (closing + 2) (String.length value - closing - 2) in
  match List.nth_opt (split_on ' ' tail) 19 with
  | Some start -> start
  | None -> failf "process stat omitted start time for pid %d" pid

let process_parent pid =
  let value = read_file_bounded "process stat" 65536 (Printf.sprintf "/proc/%d/stat" pid) in
  let closing =
    try String.rindex value ')' with Not_found -> failf "invalid process stat for pid %d" pid
  in
  let tail = String.sub value (closing + 2) (String.length value - closing - 2) in
  match List.nth_opt (split_on ' ' tail) 1 with
  | Some parent ->
      (try int_of_string parent with _ -> failf "invalid parent pid for pid %d" pid)
  | None -> failf "process stat omitted parent pid for pid %d" pid

let process_pid_namespace pid =
  Unix.readlink (Printf.sprintf "/proc/%d/ns/pid" pid)

let process_cwd pid = Unix.realpath (Printf.sprintf "/proc/%d/cwd" pid)

let process_executable pid = Unix.realpath (Printf.sprintf "/proc/%d/exe" pid)

let process_executable_sha256 pid =
  read_file_bounded "peer executable" (128 * 1024 * 1024)
    (Printf.sprintf "/proc/%d/exe" pid)
  |> sha256

let process_arguments pid =
  read_file_bounded "process command line" (256 * 1024)
    (Printf.sprintf "/proc/%d/cmdline" pid)
  |> split_on '\000' |> List.filter (( <> ) "")

let path_within root path =
  path = root
  || starts_with path (if root = "/" then "/" else root ^ "/")

let process_descends_from ~pid ~ancestor ~ancestor_start =
  let rec walk current depth =
    if depth > 64 || current <= 1 then false
    else if current = ancestor then process_start current = ancestor_start
    else
      let parent = process_parent current in
      parent <> current && walk parent (depth + 1)
  in
  walk pid 0

let current_time_us () = Int64.of_float (Unix.gettimeofday () *. 1_000_000.0)

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

let json_string_list values =
  "[" ^ String.concat "," (List.map json_quote values) ^ "]"

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
  else if
    Array.length command >= 3
    && List.mem command.(1) [ "_provider-exec"; "_provider-tui" ]
  then
    Filename.basename command.(2)
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

let product_activation_policy_root () =
  let manifest_relative =
    "tools/loom/kernel_peer_activation_capsule_authority.freeze.v1"
  in
  let rec source_root candidate =
    let manifest = Filename.concat candidate manifest_relative in
    if Sys.file_exists manifest then Some (Unix.realpath candidate)
    else
      let parent = Filename.dirname candidate in
      if parent = candidate then None else source_root parent
  in
  let runtime_root =
    let binary_dir = Filename.dirname (Unix.realpath Sys.executable_name) in
    Filename.concat (Filename.dirname binary_dir) "policy/product-activation"
  in
  let selected =
    match Loom_membrane.test_override "SOUNIO_LOOM_PRODUCT_ACTIVATION_ROOT" with
    | Some path -> path
    | _ -> (
        match source_root (Filename.dirname (Unix.realpath Sys.executable_name)) with
        | Some root -> root
        | None -> runtime_root)
  in
  let selected = Unix.realpath selected in
  if not (Sys.file_exists (Filename.concat selected manifest_relative)) then
    failf "product-activation-policy-root-missing:%s" selected;
  selected

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

type journal_authority_stamp = {
  context_digest : string;
  epoch : int;
  principal_id : string;
  signature : string;
}

type journal_authority_member_stamp = {
  quorum_principal_id : string;
  quorum_signature : string option;
}

type journal_authority_quorum_certificate = {
  quorum_context_digest : string;
  quorum_epoch : int;
  quorum_required : int;
  quorum_members : journal_authority_member_stamp list;
}

type journal_authority_proof =
  | Single_authority_stamp of journal_authority_stamp
  | Quorum_authority_certificate of journal_authority_quorum_certificate

type journal_event = {
  seq : int;
  previous : string;
  hash : string;
  utc : string;
  kind : string;
  payload_hex : string;
  authority : journal_authority_proof option;
}

type journal = {
  channel : out_channel;
  descriptor : file_descr;
  mutable seq : int;
  mutable previous : string;
  authority_context : string option;
  authority_directory : string;
}

let event_material seq previous utc kind payload_hex =
  Printf.sprintf "%d\t%s\t%s\t%s\t%s" seq previous utc kind payload_hex

let encode_event (event : journal_event) =
  match event.authority with
  | None ->
      Printf.sprintf "%d\t%s\t%s\t%s\t%s\t%s\n" event.seq event.previous
        event.hash event.utc event.kind event.payload_hex
  | Some (Single_authority_stamp stamp) ->
      Printf.sprintf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n"
        event.seq event.previous event.hash event.utc event.kind
        event.payload_hex stamp.context_digest stamp.epoch stamp.principal_id
        stamp.signature
  | Some (Quorum_authority_certificate certificate) ->
      let fields =
        List.fold_left
          (fun fields member ->
            fields
            @ [ member.quorum_principal_id;
                (match member.quorum_signature with Some value -> value | None -> "-") ])
          [ string_of_int event.seq; event.previous; event.hash; event.utc;
            event.kind; event.payload_hex; certificate.quorum_context_digest;
            string_of_int certificate.quorum_epoch; "quorum-v1";
            string_of_int certificate.quorum_required ]
          certificate.quorum_members
      in
      String.concat "\t" fields ^ "\n"

let journal_context_digest path =
  let generation_dir = Filename.dirname path in
  let session_dir = Filename.dirname (Filename.dirname generation_dir) in
  sha256
    (String.concat "\000"
       [ "loom-journal-context-v1"; Filename.basename session_dir;
         Filename.basename generation_dir; Filename.basename path ])

let journal_authority_context path =
  if not (journal_authority_required ()) then None
  else
    match Filename.basename path with
    | "journal.tsv" | "guardian.tsv" -> Some (journal_context_digest path)
    | _ -> None

let request_journal_authority_member_stamp directory context_digest epoch seq
    previous event_hash member =
  let request =
    String.concat "\t"
      [ "SOUNIO_JOURNAL_AUTHORITY_V1"; "SIGN"; context_digest;
        string_of_int seq; previous; event_hash ]
    ^ "\n"
  in
  let response = journal_authority_exchange member.member_socket request in
  match split_on '\t' response with
  | [ "OK"; "SIGNED"; stored_epoch; principal_id; signature ] ->
      let stored_epoch = positive_epoch "response-epoch" stored_epoch in
      if stored_epoch <> epoch then
        failf "sounio-journal-authority-epoch-mismatch";
      if principal_id <> member.member_principal_id then
        failf "sounio-journal-authority-principal-mismatch";
      let payload =
        journal_authority_payload context_digest epoch principal_id seq previous
          event_hash
      in
      if not
           (journal_authority_signature_is_valid member.member_public_key directory payload
              signature)
      then failf "sounio-journal-authority-signature-invalid";
      signature
  | [ "REFUSE"; reason ] -> failf "sounio-journal-authority-refused:%s" reason
  | _ -> failf "sounio-journal-authority-invalid-response"

let request_journal_authority_stamp directory context_digest seq previous
    event_hash =
  let epoch = journal_authority_expected_epoch () in
  if journal_authority_epoch_is_revoked epoch then
    failf "sounio-journal-authority-epoch-revoked:%d" epoch;
  match journal_authority_configuration () with
  | Single_journal_authority member ->
      let signature =
        request_journal_authority_member_stamp directory context_digest epoch seq
          previous event_hash member
      in
      Some
        (Single_authority_stamp
           { context_digest; epoch; principal_id = member.member_principal_id;
             signature })
  | Journal_authority_quorum { required; members } ->
      let quorum_members =
        List.map
          (fun member ->
            let signature =
              try
                Some
                  (request_journal_authority_member_stamp directory context_digest
                     epoch seq previous event_hash member)
              with
              | Unix_error _ | Sys_error _ | Loom_error _ -> None
            in
            { quorum_principal_id = member.member_principal_id;
              quorum_signature = signature })
          members
      in
      let valid =
        List.fold_left
          (fun count member ->
            match member.quorum_signature with Some _ -> count + 1 | None -> count)
          0 quorum_members
      in
      if valid < required then
        failf "sounio-journal-authority-quorum-unsatisfied:valid=%d:required=%d"
          valid required;
      Some
        (Quorum_authority_certificate
           { quorum_context_digest = context_digest; quorum_epoch = epoch;
             quorum_required = required; quorum_members })

let append_event journal kind payload =
  let seq = journal.seq + 1 in
  let utc = utc_now () in
  let payload_hex = hex_of_string payload in
  let hash = sha256 (event_material seq journal.previous utc kind payload_hex) in
  let authority =
    match journal.authority_context with
    | None -> None
    | Some context_digest ->
        request_journal_authority_stamp journal.authority_directory
          context_digest seq journal.previous hash
  in
  let event =
    { seq; previous = journal.previous; hash; utc; kind; payload_hex; authority }
  in
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
      { seq; previous; hash; utc; kind; payload_hex; authority = None }
  | [ seq; previous; hash; utc; kind; payload_hex; context_digest; epoch;
      principal_id; signature ] ->
      let seq =
        try int_of_string seq
        with _ -> failf "journal sequence is not an integer"
      in
      let epoch = positive_epoch "event-epoch" epoch in
      { seq; previous; hash; utc; kind; payload_hex;
        authority =
          Some
            (Single_authority_stamp
               { context_digest; epoch; principal_id; signature }) }
  | [ seq; previous; hash; utc; kind; payload_hex; context_digest; epoch;
      "quorum-v1"; required; principal_a; signature_a; principal_b;
      signature_b; principal_c; signature_c ] ->
      let seq =
        try int_of_string seq
        with _ -> failf "journal sequence is not an integer"
      in
      let epoch = positive_epoch "event-epoch" epoch in
      let required = positive_epoch "event-quorum" required in
      let member principal signature =
        { quorum_principal_id = principal;
          quorum_signature = if signature = "-" then None else Some signature }
      in
      { seq; previous; hash; utc; kind; payload_hex;
        authority =
          Some
            (Quorum_authority_certificate
               { quorum_context_digest = context_digest; quorum_epoch = epoch;
                 quorum_required = required;
                 quorum_members =
                   [ member principal_a signature_a;
                     member principal_b signature_b;
                     member principal_c signature_c ] }) }
  | _ -> failf "journal record does not have six, ten, or sixteen fields"

let verify_journal_event_authority path (event : journal_event) =
  match (journal_authority_required (), event.authority) with
  | false, None -> ()
  | true, None ->
      failf "sounio-journal-authority-event-unsigned seq=%d" event.seq
  | _, Some (Single_authority_stamp stamp) ->
      (match Sys.getenv_opt "SOUNIO_LOOM_JOURNAL_AUTHORITY_QUORUM" with
      | Some value when value <> "" ->
          failf "sounio-journal-authority-event-single-proof-in-quorum-mode"
      | _ -> ());
      let public_key = journal_authority_public_key () in
      let expected_principal = journal_principal_id public_key in
      let expected_epoch = journal_authority_expected_epoch () in
      if journal_authority_epoch_is_revoked stamp.epoch then
        failf "sounio-journal-authority-epoch-revoked:%d" stamp.epoch;
      if stamp.epoch <> expected_epoch then
        failf "sounio-journal-authority-event-epoch-mismatch seq=%d" event.seq;
      if stamp.principal_id <> expected_principal then
        failf "sounio-journal-authority-event-principal-mismatch seq=%d" event.seq;
      if stamp.context_digest <> journal_context_digest path then
        failf "sounio-journal-authority-event-context-mismatch seq=%d" event.seq;
      let payload =
        journal_authority_payload stamp.context_digest stamp.epoch
          stamp.principal_id event.seq event.previous event.hash
      in
      if not
           (journal_authority_signature_is_valid public_key
              (Filename.dirname path) payload stamp.signature)
      then failf "sounio-journal-authority-event-signature-invalid seq=%d" event.seq
  | _, Some (Quorum_authority_certificate certificate) ->
      let required, configured =
        match journal_authority_configuration () with
        | Single_journal_authority _ ->
            failf "sounio-journal-authority-event-quorum-proof-in-single-mode"
        | Journal_authority_quorum config -> (config.required, config.members)
      in
      let expected_epoch = journal_authority_expected_epoch () in
      if journal_authority_epoch_is_revoked certificate.quorum_epoch then
        failf "sounio-journal-authority-epoch-revoked:%d"
          certificate.quorum_epoch;
      if certificate.quorum_epoch <> expected_epoch then
        failf "sounio-journal-authority-event-epoch-mismatch seq=%d" event.seq;
      if certificate.quorum_required <> required then
        failf "sounio-journal-authority-event-quorum-mismatch seq=%d" event.seq;
      if certificate.quorum_context_digest <> journal_context_digest path then
        failf "sounio-journal-authority-event-context-mismatch seq=%d" event.seq;
      let rec verify_members valid configured stamped =
        match (configured, stamped) with
        | [], [] -> valid
        | member :: configured_tail, stamp :: stamped_tail ->
            if stamp.quorum_principal_id <> member.member_principal_id then
              failf
                "sounio-journal-authority-event-principal-mismatch seq=%d"
                event.seq;
            let valid =
              match stamp.quorum_signature with
              | None -> valid
              | Some signature ->
                  let payload =
                    journal_authority_payload certificate.quorum_context_digest
                      certificate.quorum_epoch stamp.quorum_principal_id event.seq
                      event.previous event.hash
                  in
                  if not
                       (journal_authority_signature_is_valid member.member_public_key
                          (Filename.dirname path) payload signature)
                  then
                    failf
                      "sounio-journal-authority-event-signature-invalid seq=%d"
                      event.seq;
                  valid + 1
            in
            verify_members valid configured_tail stamped_tail
        | _ ->
            failf "sounio-journal-authority-event-member-count-mismatch seq=%d"
              event.seq
      in
      let valid = verify_members 0 configured certificate.quorum_members in
      if valid < required then
        failf
          "sounio-journal-authority-event-quorum-unsatisfied seq=%d valid=%d required=%d"
          event.seq valid required

type journal_phase = Initial | Active | Exited

let parse_output_span payload =
  match split_on ':' payload with
  | start :: ending :: _ ->
      let parse name value =
        try int_of_string value with _ -> failf "semantic:%s-is-not-an-integer" name
      in
      (parse "output-start" start, parse "output-end" ending)
  | _ -> failf "semantic:invalid-output-span"

let verify_events path events =
  let expected_seq = ref 1 in
  let expected_previous = ref (String.make 64 '0') in
  let phase = ref Initial in
  let lease = ref None in
  let output_cursor = ref 0 in
  List.iter
    (fun (event : journal_event) ->
      verify_journal_event_authority path event;
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
        | "OBSERVER_DETACHED" | "KERNEL_GENERATION" | "PEER_REFUSED"
        | "EXEC_GRANT_ISSUED" | "EXEC_GRANT_EXPIRED"
        | "EXEC_GRANT_REFUSED" | "EXEC_GRANT_CONSUMED"
        | "EXEC_CONSUME_REFUSED" | "EXEC_OUTCOME_RECORDED"
        | "EXEC_OUTCOME_REFUSED" | "EXEC_OUTCOME_INCOMPLETE"
        | "SOVEREIGN_GRANT_CONSUMED" | "SOVEREIGN_EXEC_COMPLETED"
        | "SOVEREIGN_EXEC_REFUSED" ), Active -> ()
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
  let phase, digest = verify_events path events in
  (events, phase, digest)

let open_journal path =
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_TRUNC ] 0o600 in
  Unix.set_close_on_exec descriptor;
  let channel = Unix.out_channel_of_descr descriptor in
  { channel; descriptor; seq = 0; previous = String.make 64 '0';
    authority_context = journal_authority_context path;
    authority_directory = Filename.dirname path }

let resume_journal path =
  let events, phase, digest = load_and_verify_journal path in
  if phase <> Active then failf "cannot recover a non-active semantic journal";
  let descriptor = Unix.openfile path [ O_WRONLY; O_APPEND ] 0o600 in
  Unix.set_close_on_exec descriptor;
  let channel = Unix.out_channel_of_descr descriptor in
  ({ channel; descriptor; seq = List.length events; previous = digest;
     authority_context = journal_authority_context path;
     authority_directory = Filename.dirname path }, events)

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

let process_pidfd_alive descriptor =
  let readable, _, _ = Unix.select [ descriptor ] [] [] 0.0 in
  readable = []

type guardian_client_mode = Guardian_awaiting | Guardian_bridge

type guardian_client = {
  guardian_fd : file_descr;
  guardian_peer_pid : int;
  guardian_peer_uid : int;
  guardian_peer_gid : int;
  guardian_peer_start : string;
  guardian_peer_pidfd : file_descr;
  guardian_input : Buffer.t;
  mutable guardian_mode : guardian_client_mode;
  mutable guardian_pending : string;
  mutable guardian_pending_offset : int;
}

type guardian_material = {
  guardian_material_pid : int;
  guardian_material_start : string;
  guardian_material_job : string;
  guardian_material_pidfd : file_descr;
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
  guardian_materials : (file_descr, guardian_material) Hashtbl.t;
  guardian_kernel_pid : int;
  guardian_kernel_start : string;
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
    ("material_witnesses", string_of_int (Hashtbl.length guardian.guardian_materials));
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
      (try Unix.close client.guardian_peer_pidfd with _ -> ());
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
    ("material_witnesses", string_of_int (Hashtbl.length guardian.guardian_materials));
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
      | "MATERIAL_REGISTER", [ job_id; pid_raw; start ] -> (
          try
            if not (valid_sha256 job_id) then failf "material-job-invalid";
            let pid = parse_nonnegative "material-pid" pid_raw in
            if client.guardian_peer_pid <> guardian.guardian_kernel_pid ||
               client.guardian_peer_uid <> Unix.geteuid () ||
               client.guardian_peer_gid <> Unix.getegid () ||
               client.guardian_peer_start <> guardian.guardian_kernel_start ||
               not (process_pidfd_alive client.guardian_peer_pidfd)
            then failf "material-register-peer-refused";
            if process_start pid <> start then
              failf "material-register-start-mismatch";
            let parent = process_parent pid in
            if parent <> guardian.guardian_kernel_pid then
              failf "material-register-parent-mismatch";
            if process_executable pid <> process_executable guardian.guardian_kernel_pid
               || process_executable_sha256 pid <>
                  process_executable_sha256 guardian.guardian_kernel_pid
            then failf "material-register-executable-mismatch";
            let pidfd =
              match pidfd_open pid with
              | Some descriptor when process_pidfd_alive descriptor -> descriptor
              | Some descriptor ->
                  Unix.close descriptor;
                  failf "material-register-worker-dead"
              | None -> failf "material-register-pidfd-unavailable"
            in
            let duplicate =
              Hashtbl.fold
                (fun _ material found ->
                  found || material.guardian_material_pid = pid
                  || material.guardian_material_job = job_id)
                guardian.guardian_materials false
            in
            if duplicate then (
              Unix.close pidfd;
              failf "material-register-duplicate");
            Hashtbl.add guardian.guardian_materials pidfd
              { guardian_material_pid = pid;
                guardian_material_start = start;
                guardian_material_job = job_id;
                guardian_material_pidfd = pidfd };
            ignore
              (append_event guardian.guardian_journal "MATERIAL_REGISTERED"
                 (String.concat ":" [ job_id; string_of_int pid; start ]));
            guardian_queue client
              (control_line
                 [ "OK"; "MATERIAL_REGISTERED";
                   guardian.guardian_instance_id; job_id ])
          with
          | Loom_error error -> refuse error
          | Unix_error (error, name, argument) ->
              refuse
                (Printf.sprintf "%s:%s(%s)" (Unix.error_message error) name
                   argument))
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
    let peer_pidfd = ref None in
    (try
       Unix.set_close_on_exec descriptor;
       let peer_pid, peer_uid, peer_gid = peer_credentials descriptor in
       let peer_start = process_start peer_pid in
       let pidfd =
         match pidfd_open peer_pid with
         | Some value when process_pidfd_alive value -> value
         | Some value ->
             Unix.close value;
             failf "guardian-peer-dead"
         | None -> failf "guardian-peer-pidfd-unavailable"
       in
       peer_pidfd := Some pidfd;
       Unix.set_nonblock descriptor;
       Hashtbl.add guardian.guardian_clients descriptor
         {
           guardian_fd = descriptor;
           guardian_peer_pid = peer_pid;
           guardian_peer_uid = peer_uid;
           guardian_peer_gid = peer_gid;
           guardian_peer_start = peer_start;
           guardian_peer_pidfd = pidfd;
           guardian_input = Buffer.create 256;
           guardian_mode = Guardian_awaiting;
           guardian_pending = "";
           guardian_pending_offset = 0;
         }
     with error ->
       Option.iter (fun value -> try Unix.close value with _ -> ()) !peer_pidfd;
       (try Unix.close descriptor with _ -> ());
       raise error)
  with
  | Unix_error ((EAGAIN | EWOULDBLOCK), _, _) -> ()
  | _ -> ()

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

let guardian_reap_material guardian descriptor =
  match Hashtbl.find_opt guardian.guardian_materials descriptor with
  | None -> ()
  | Some material ->
      Hashtbl.remove guardian.guardian_materials descriptor;
      (try Unix.close material.guardian_material_pidfd with _ -> ());
      ignore
        (append_event guardian.guardian_journal "MATERIAL_EXTINCT"
           (String.concat ":"
              [ material.guardian_material_job;
                string_of_int material.guardian_material_pid;
                material.guardian_material_start ]))

let guardian_stop_materials guardian =
  let materials =
    Hashtbl.fold
      (fun descriptor material values -> (descriptor, material) :: values)
      guardian.guardian_materials []
  in
  List.iter
    (fun (descriptor, material) ->
      (try Unix.kill material.guardian_material_pid Sys.sigkill with _ -> ());
      Hashtbl.remove guardian.guardian_materials descriptor;
      (try Unix.close material.guardian_material_pidfd with _ -> ());
      ignore
        (append_event guardian.guardian_journal "MATERIAL_REVOKED"
           (String.concat ":"
              [ material.guardian_material_job;
                string_of_int material.guardian_material_pid;
                material.guardian_material_start ])))
    materials

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
           Printf.sprintf "SOUNIO_LOOM_SOCKET=%s" paths.socket_path;
           "SOUNIO_LOOM_SOVEREIGN_EXEC_REQUIRED=1";
           Printf.sprintf "SOUNIO_LOOM_AGENT=%s" agent;
           Printf.sprintf "SOUNIO_LOOM_LANE=%s" lane;
           Printf.sprintf "SOUNIO_LOOM_SESSION_ID=%s" session_id;
           Printf.sprintf "SOUNIO_LOOM_INSTANCE_ID=%s" instance_id |]
    in
    Unix.execvpe command.(0) command environment);
  Unix.set_close_on_exec master_fd;
  Unix.set_nonblock master_fd;
  (try set_winsize master_fd 40 140 with _ -> ());
  let kernel_pid = Unix.getppid () in
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
      guardian_materials = Hashtbl.create 8;
      guardian_kernel_pid = kernel_pid;
      guardian_kernel_start = process_start kernel_pid;
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
    not guardian.guardian_stopping &&
    (guardian.guardian_harness_exit = None ||
     Hashtbl.length guardian.guardian_materials > 0)
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
    let material_fds =
      Hashtbl.fold
        (fun descriptor _ values -> descriptor :: values)
        guardian.guardian_materials []
    in
    let guardian_inputs =
      guardian.guardian_listener :: client_fds @ material_fds @
      (if guardian.guardian_harness_exit = None
       then [ guardian.guardian_master_fd ] else [])
    in
    let readable, writable, _ =
      Unix.select
        guardian_inputs write_fds [] 0.2
    in
    List.iter
      (fun descriptor ->
        if descriptor = guardian.guardian_listener then
          guardian_accept_client guardian
        else if Hashtbl.mem guardian.guardian_materials descriptor then
          guardian_reap_material guardian descriptor
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
    if guardian.guardian_harness_exit = None then
      match guardian_child_status guardian with
      | Some status ->
          guardian.guardian_harness_exit <- Some (process_exit_code status);
          write_guardian_descriptor guardian
            (if Hashtbl.length guardian.guardian_materials = 0
             then "exited" else "material-active")
      | None -> ()
  done;
  if guardian.guardian_stopping then (
    guardian_stop_materials guardian;
    guardian_stop_child guardian);
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

let verify_guardian_events path events =
  let expected_seq = ref 1 in
  let expected_previous = ref (String.make 64 '0') in
  let phase = ref Guardian_initial in
  let output_cursor = ref 0 in
  List.iter
    (fun (event : journal_event) ->
      verify_journal_event_authority path event;
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
      | ("INPUT" | "RESIZE" | "SIGNAL" | "MATERIAL_REGISTERED"
        | "MATERIAL_EXTINCT" | "MATERIAL_REVOKED"), Guardian_active -> ()
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
  let phase, cursor, digest = verify_guardian_events path events in
  (events, phase, cursor, digest)

let guardian_output_payload event =
  let payload = string_of_hex event.payload_hex in
  match split_on ':' payload with
  | [ start_value; ending_value; digest ] ->
      let parse label value =
        try int_of_string value
        with _ ->
          failf "guardian-output:%s-is-not-an-integer seq=%d" label event.seq
      in
      let start = parse "start" start_value in
      let ending = parse "end" ending_value in
      if start < 0 || ending < start then
        failf "guardian-output:invalid-span seq=%d" event.seq;
      if not (valid_sha256 digest) then
        failf "guardian-output:invalid-digest seq=%d" event.seq;
      (start, ending, digest)
  | _ -> failf "guardian-output:invalid-payload seq=%d" event.seq

let read_guardian_output_chunk descriptor (event : journal_event) start ending =
  let length = ending - start in
  let bytes = Bytes.create length in
  ignore (Unix.lseek descriptor start SEEK_SET);
  let rec fill offset =
    if offset < length then
      let count = Unix.read descriptor bytes offset (length - offset) in
      if count = 0 then
        failf "guardian-output:unexpected-eof seq=%d offset=%d" event.seq
          (start + offset)
      else fill (offset + count)
  in
  fill 0;
  Bytes.unsafe_to_string bytes

let verified_guardian_output_range events output_path expected_size cursor length =
  let descriptor = Unix.openfile output_path [ O_RDONLY ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      let stats = Unix.fstat descriptor in
      if stats.st_kind <> S_REG then failf "guardian-output:not-regular-file";
      if stats.st_size <> expected_size then
        failf "guardian cursor does not match durable output";
      let requested_end = cursor + length in
      let output = Buffer.create length in
      List.iter
        (fun (event : journal_event) ->
          if event.kind = "OUTPUT" then (
            let start, ending, expected_digest = guardian_output_payload event in
            let chunk = read_guardian_output_chunk descriptor event start ending in
            let measured_digest = sha256 chunk in
            if measured_digest <> expected_digest then
              failf "guardian-output:digest-mismatch seq=%d" event.seq;
            let overlap_start = max cursor start in
            let overlap_end = min requested_end ending in
            if overlap_end > overlap_start then
              Buffer.add_substring output chunk (overlap_start - start)
                (overlap_end - overlap_start)))
        events;
      if (Unix.fstat descriptor).st_size <> expected_size then
        failf "guardian-output:size-changed-during-replay";
      let value = Buffer.contents output in
      if String.length value <> length then
        failf "guardian-output:range-incomplete expected=%d actual=%d" length
          (String.length value);
      value)

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
  peer_pid : int;
  peer_uid : int;
  peer_gid : int;
  peer_start : string;
  peer_pid_namespace : string;
  peer_pidfd : file_descr;
  input : Buffer.t;
  mutable mode : stream_mode;
  mutable pending : string;
  mutable pending_offset : int;
}

type exec_grant = {
  exec_payload : string;
  exec_payload_sha256 : string;
  exec_cwd : string;
  exec_expires_us : int64;
  exec_generation : string;
}

type exec_outcome_obligation = {
  outcome_payload_sha256 : string;
  outcome_cwd : string;
  outcome_generation : string;
  outcome_peer_pid : int;
  outcome_peer_start : string;
  outcome_consumed_us : int64;
}

type sovereign_job_state =
  | Sovereign_running
  | Sovereign_complete of string * string
  | Sovereign_failed of string

type sovereign_job = {
  sovereign_job_id : string;
  sovereign_payload_sha256 : string;
  sovereign_event_sha256 : string;
  sovereign_command_sha256 : string;
  sovereign_worker_pid : int;
  sovereign_worker_start : string;
  sovereign_worker_pidfd : file_descr;
  sovereign_result_path : string;
  mutable sovereign_state : sovereign_job_state;
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
  kernel_generation : string;
  boot_id : string;
  pid_namespace : string;
  executable_path : string;
  executable_sha256 : string;
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
  exec_grants : (string, exec_grant) Hashtbl.t;
  exec_outcomes : (string, exec_outcome_obligation) Hashtbl.t;
  sovereign_grants : (string, unit) Hashtbl.t;
  sovereign_jobs : (string, sovereign_job) Hashtbl.t;
  sovereign_exec_required : bool;
  mutable next_client : int;
  mutable input_holder : file_descr option;
  mutable output_cursor : int;
  mutable stopping : bool;
  mutable harness_exit : int option;
  mutable next_coord_refresh : float;
  mutable coord_pid : int option;
  mutable coord_failures : int;
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
    ("kernel_generation", kernel.kernel_generation);
    ("boot_id", kernel.boot_id);
    ("pid_namespace", kernel.pid_namespace);
    ("daemon_pid", string_of_int (Unix.getpid ()));
    ("daemon_pid_start", kernel.daemon_pid_start);
    ("harness_pid", string_of_int kernel.harness_pid);
    ("harness_pid_start", kernel.harness_pid_start);
    ("guardian_pid", string_of_int kernel.guardian_pid);
    ("guardian_pid_start", kernel.guardian_pid_start);
    ("guardian_socket", kernel.paths.guardian_socket_path);
    ("socket", kernel.paths.socket_path);
    ("token_file", kernel.paths.token_path);
    ("sovereign_exec_required", string_of_bool kernel.sovereign_exec_required);
    ("exec_release_protocol", if kernel.sovereign_exec_required then "LOOM_EXEC/1" else "LOOM/1-legacy");
    ("output_file", kernel.output_path);
    ("journal_file", kernel.journal_path);
    ("guardian_journal_file", kernel.guardian_journal_path);
    ("output_cursor", string_of_int kernel.output_cursor);
    ("command", kernel.command_name);
    ("argv_digest", kernel.command_digest);
    ("started_utc", kernel.started_utc);
  ]

let write_descriptor kernel state =
  let text = descriptor_text (descriptor_fields kernel state) in
  atomic_write kernel.paths.descriptor_path text;
  atomic_write (Filename.concat (Filename.dirname kernel.output_path) "session.state")
    text

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
    ("kernel_generation", kernel.kernel_generation);
    ("pending_exec_grants", string_of_int (Hashtbl.length kernel.exec_grants));
    ("pending_exec_outcomes", string_of_int (Hashtbl.length kernel.exec_outcomes));
    ("pending_sovereign_grants", string_of_int (Hashtbl.length kernel.sovereign_grants));
    ("sovereign_jobs", string_of_int (Hashtbl.length kernel.sovereign_jobs));
    ("sovereign_exec_required", string_of_bool kernel.sovereign_exec_required);
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
      (try Unix.close client.peer_pidfd with _ -> ());
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

let pidfd_alive descriptor =
  let readable, _, _ = Unix.select [ descriptor ] [] [] 0.0 in
  readable = []

let arguments_contain_pair arguments first second =
  let rec loop = function
    | left :: right :: _ when left = first && right = second -> true
    | _ :: tail -> loop tail
    | [] -> false
  in
  loop arguments

let authenticate_exec_peer kernel client operation handle =
  if client.peer_uid <> Unix.geteuid () || client.peer_gid <> Unix.getegid () then
    failf "exec-peer-credential-mismatch";
  if not (pidfd_alive client.peer_pidfd) then failf "exec-peer-exited";
  if process_start client.peer_pid <> client.peer_start then
    failf "exec-peer-start-changed";
  let namespace = process_pid_namespace client.peer_pid in
  if namespace <> client.peer_pid_namespace || namespace <> kernel.pid_namespace then
    failf "exec-peer-pid-namespace-mismatch";
  if trim (read_file "/proc/sys/kernel/random/boot_id") <> kernel.boot_id then
    failf "exec-peer-boot-identity-mismatch";
  if process_executable client.peer_pid <> kernel.executable_path
     || process_executable_sha256 client.peer_pid <> kernel.executable_sha256
  then failf "exec-peer-executable-mismatch";
  if
    not
      (process_descends_from ~pid:client.peer_pid ~ancestor:kernel.harness_pid
         ~ancestor_start:kernel.harness_pid_start)
  then failf "exec-peer-outside-harness-ancestry";
  let peer_cwd = process_cwd client.peer_pid in
  if not (path_within kernel.cwd peer_cwd) then failf "exec-peer-cwd-outside-worktree";
  let arguments = process_arguments client.peer_pid in
  if operation = "sovereign-start" then (
    if process_parent client.peer_pid <> kernel.harness_pid then
      failf "sovereign-issuer-not-direct-harness-child";
    if not (List.mem "agent-hook" arguments) then
      failf "sovereign-issuer-command-mismatch")
  else if operation = "sovereign-present" then (
    if process_parent client.peer_pid <> kernel.harness_pid then
      failf "sovereign-presenter-not-direct-harness-child";
    if not (List.mem "sovereign-result" arguments) then
      failf "sovereign-presenter-command-mismatch";
    match handle with
    | Some expected when arguments_contain_pair arguments "--job" expected -> ()
    | _ -> failf "sovereign-presenter-job-mismatch")
  else if operation = "issue" then (
    if not (List.mem "agent-hook" arguments) then
      failf "exec-issuer-command-mismatch")
  else if operation = "consume" || operation = "outcome" then (
    if not (List.mem "exec-capability" arguments) then
      failf "exec-consumer-command-mismatch";
    match handle with
    | Some expected when arguments_contain_pair arguments "--handle" expected -> ()
    | _ -> failf "exec-consumer-handle-mismatch")
  else failf "exec-peer-operation-invalid";
  peer_cwd

let valid_exec_handle value =
  String.length value = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let expire_exec_grants kernel =
  let now = current_time_us () in
  let expired =
    Hashtbl.fold
      (fun handle grant values ->
        if now > grant.exec_expires_us then handle :: values else values)
      kernel.exec_grants []
  in
  List.iter
    (fun handle ->
      Hashtbl.remove kernel.exec_grants handle;
      ignore
        (append_event kernel.journal "EXEC_GRANT_EXPIRED"
           (sha256 handle)))
    expired

let materialize_exec_outcome_incomplete kernel handle obligation reason =
  ignore
    (append_event kernel.journal "EXEC_OUTCOME_INCOMPLETE"
       (String.concat ":"
          [ sha256 handle; obligation.outcome_payload_sha256;
            obligation.outcome_generation;
            Int64.to_string obligation.outcome_consumed_us; reason ]));
  Hashtbl.remove kernel.exec_outcomes handle

let materialize_incomplete_exec_outcomes kernel reason =
  let pending =
    Hashtbl.fold
      (fun handle obligation values -> (handle, obligation) :: values)
      kernel.exec_outcomes []
  in
  List.iter
    (fun (handle, obligation) ->
      materialize_exec_outcome_incomplete kernel handle obligation reason)
    pending

let materialize_orphaned_exec_outcomes kernel =
  let orphaned =
    Hashtbl.fold
      (fun handle obligation values ->
        let alive =
          try process_start obligation.outcome_peer_pid = obligation.outcome_peer_start
          with _ -> false
        in
        if alive then values else (handle, obligation) :: values)
      kernel.exec_outcomes []
  in
  List.iter
    (fun (handle, obligation) ->
      materialize_exec_outcome_incomplete kernel handle obligation "broker-exited")
    orphaned

let guardian_register_material kernel job_id worker_pid worker_start =
  let descriptor = connect_unix kernel.paths.guardian_socket_path in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      write_all descriptor
        (guardian_request_line kernel.token "MATERIAL_REGISTER"
           [ job_id; string_of_int worker_pid; worker_start ]);
      match
        guardian_parse_ok (read_protocol_line descriptor) "MATERIAL_REGISTERED"
      with
      | [ instance; actual_job ]
        when instance = kernel.instance_id && actual_job = job_id -> ()
      | _ -> failf "guardian-material-register-response-invalid")

let sovereign_result_directory kernel =
  let path =
    Filename.concat (Filename.dirname kernel.output_path) "sovereign-results"
  in
  mkdir_p path;
  path

let close_sovereign_worker_inherited kernel start_gate_write =
  let close descriptor =
    if descriptor <> start_gate_write then try Unix.close descriptor with _ -> ()
  in
  close kernel.listener;
  close kernel.guardian_fd;
  close kernel.journal.descriptor;
  Hashtbl.iter
    (fun descriptor client ->
      close descriptor;
      close client.peer_pidfd)
    kernel.clients;
  Hashtbl.iter
    (fun _ job -> close job.sovereign_worker_pidfd)
    kernel.sovereign_jobs

let new_sovereign_job_id kernel =
  let rec choose () =
    let value = random_hex 32 in
    if Hashtbl.mem kernel.sovereign_jobs value then choose () else value
  in
  choose ()

let start_sovereign_job kernel client ~event_sha256 ~command_sha256
    ~payload_sha256 ~payload =
  if not kernel.sovereign_exec_required then
    failf "sovereign-exec-not-required";
  ignore
    (authenticate_exec_peer kernel client "sovereign-start" None);
  if not (valid_sha256 event_sha256 && valid_sha256 command_sha256 &&
          valid_sha256 payload_sha256)
  then failf "sovereign-exec-digest-invalid";
  if payload = "" || String.length payload > max_exec_capability_payload_bytes
  then failf "sovereign-exec-payload-size-refused";
  if sha256 payload <> payload_sha256 then
    failf "sovereign-exec-payload-digest-mismatch";
  ignore
    (Loom_sovereign_exec.validate_payload ~root:kernel.cwd ~event_sha256
       ~command_sha256 payload);
  let grant_id =
    sha256
      (String.concat ":"
         [ kernel.kernel_generation; string_of_int client.peer_pid;
           client.peer_start; event_sha256; command_sha256; payload_sha256 ])
  in
  if Hashtbl.mem kernel.sovereign_grants grant_id then
    failf "sovereign-grant-duplicate";
  Hashtbl.add kernel.sovereign_grants grant_id ();
  let job_id = new_sovereign_job_id kernel in
  let result_path =
    Filename.concat (sovereign_result_directory kernel) (job_id ^ ".record")
  in
  if Sys.file_exists result_path then (
    Hashtbl.remove kernel.sovereign_grants grant_id;
    failf "sovereign-result-collision");
  let start_gate_read, start_gate_write = Unix.pipe ~cloexec:true () in
  let worker_pid =
    match Unix.fork () with
    | 0 ->
        Unix.close start_gate_write;
        close_sovereign_worker_inherited kernel start_gate_read;
        let code =
          Loom_sovereign_exec.worker ~root:kernel.cwd ~event_sha256
            ~command_sha256 ~payload ~job_id
            ~kernel_generation:kernel.kernel_generation
            ~guardian_pid:kernel.guardian_pid
            ~guardian_start:kernel.guardian_pid_start ~result_path
            ~start_gate:start_gate_read
        in
        Unix._exit code
    | pid -> pid
  in
  Unix.close start_gate_read;
  let worker_start =
    try process_start worker_pid with error ->
      Unix.close start_gate_write;
      Hashtbl.remove kernel.sovereign_grants grant_id;
      (try Unix.kill worker_pid Sys.sigkill with _ -> ());
      (try ignore (Unix.waitpid [] worker_pid) with _ -> ());
      raise error
  in
  let worker_pidfd =
    match pidfd_open worker_pid with
    | Some descriptor when pidfd_alive descriptor -> descriptor
    | Some descriptor ->
        Unix.close descriptor;
        Unix.close start_gate_write;
        Hashtbl.remove kernel.sovereign_grants grant_id;
        (try Unix.kill worker_pid Sys.sigkill with _ -> ());
        (try ignore (Unix.waitpid [] worker_pid) with _ -> ());
        failf "sovereign-worker-not-alive"
    | None ->
        Unix.close start_gate_write;
        Hashtbl.remove kernel.sovereign_grants grant_id;
        (try Unix.kill worker_pid Sys.sigkill with _ -> ());
        (try ignore (Unix.waitpid [] worker_pid) with _ -> ());
        failf "sovereign-worker-pidfd-unavailable"
  in
  (try
     guardian_register_material kernel job_id worker_pid worker_start;
     if not (Hashtbl.mem kernel.sovereign_grants grant_id) then
       failf "sovereign-grant-state-lost";
     Hashtbl.remove kernel.sovereign_grants grant_id;
     write_all start_gate_write "G";
     Unix.close start_gate_write
   with error ->
     Hashtbl.remove kernel.sovereign_grants grant_id;
     (try Unix.close start_gate_write with _ -> ());
     (try Unix.kill worker_pid Sys.sigkill with _ -> ());
     (try ignore (Unix.waitpid [] worker_pid) with _ -> ());
     Unix.close worker_pidfd;
     raise error);
  let job =
    { sovereign_job_id = job_id;
      sovereign_payload_sha256 = payload_sha256;
      sovereign_event_sha256 = event_sha256;
      sovereign_command_sha256 = command_sha256;
      sovereign_worker_pid = worker_pid;
      sovereign_worker_start = worker_start;
      sovereign_worker_pidfd = worker_pidfd;
      sovereign_result_path = result_path;
      sovereign_state = Sovereign_running }
  in
  Hashtbl.add kernel.sovereign_jobs job_id job;
  ignore
    (append_event kernel.journal "SOVEREIGN_GRANT_CONSUMED"
       (String.concat ":"
          [ sha256 grant_id; job_id; payload_sha256; event_sha256;
            command_sha256; string_of_int client.peer_pid;
            string_of_int worker_pid ]));
  job

let reap_sovereign_jobs kernel =
  Hashtbl.iter
    (fun _ job ->
      match job.sovereign_state with
      | Sovereign_complete _ | Sovereign_failed _ -> ()
      | Sovereign_running -> (
          let identity_error =
            if pidfd_alive job.sovereign_worker_pidfd then
              let actual_start =
                try Some (process_start job.sovereign_worker_pid)
                with _ -> None
              in
              match actual_start with
              | Some value when value = job.sovereign_worker_start -> None
              | Some _ -> Some "sovereign-worker-identity-changed"
              | None -> Some "sovereign-worker-identity-unreadable"
            else None
          in
          match identity_error with
          | Some reason ->
              (try Unix.kill job.sovereign_worker_pid Sys.sigkill with _ -> ());
              (try ignore (Unix.waitpid [] job.sovereign_worker_pid) with _ -> ());
              (try Unix.close job.sovereign_worker_pidfd with _ -> ());
              job.sovereign_state <- Sovereign_failed reason;
              ignore
                (append_event kernel.journal "SOVEREIGN_EXEC_REFUSED"
                   (String.concat ":"
                      [ job.sovereign_job_id; sha256 reason; "255" ]))
          | None -> (match Unix.waitpid [ WNOHANG ] job.sovereign_worker_pid with
          | 0, _ -> ()
          | _, status ->
              (try Unix.close job.sovereign_worker_pidfd with _ -> ());
              let code = process_exit_code status in
              if code = 0 then
                (try
                   let table, record_sha256, _ =
                     Loom_sovereign_exec.validate_result_file
                       ~path:job.sovereign_result_path
                       ~job_id:job.sovereign_job_id
                       ~payload_sha256:job.sovereign_payload_sha256
                   in
                   if Loom_exec.required table "state" <> "COMPLETED" then
                     failf "sovereign-worker-result-not-complete";
                   if Loom_exec.required table "event_sha256" <>
                      job.sovereign_event_sha256
                   then failf "sovereign-worker-event-mismatch";
                   if Loom_exec.required table "command_sha256" <>
                      job.sovereign_command_sha256
                   then failf "sovereign-worker-command-mismatch";
                   job.sovereign_state <-
                     Sovereign_complete
                       (job.sovereign_result_path, record_sha256);
                   ignore
                     (append_event kernel.journal "SOVEREIGN_EXEC_COMPLETED"
                        (String.concat ":"
                           [ job.sovereign_job_id; record_sha256;
                             job.sovereign_payload_sha256;
                             string_of_int job.sovereign_worker_pid ]))
                 with
                 | Loom_sovereign_exec.Error reason
                 | Loom_error reason ->
                     job.sovereign_state <- Sovereign_failed reason;
                     ignore
                       (append_event kernel.journal "SOVEREIGN_EXEC_REFUSED"
                          (String.concat ":"
                             [ job.sovereign_job_id; sha256 reason;
                               string_of_int code ])))
              else (
                let reason = Printf.sprintf "worker-exit-%d" code in
                job.sovereign_state <- Sovereign_failed reason;
                ignore
                  (append_event kernel.journal "SOVEREIGN_EXEC_REFUSED"
                     (String.concat ":"
                        [ job.sovereign_job_id; sha256 reason;
                          string_of_int code ])))
          | exception Unix_error (ECHILD, _, _) ->
              let reason = "worker-reap-lost" in
              job.sovereign_state <- Sovereign_failed reason;
              (try Unix.close job.sovereign_worker_pidfd with _ -> ());
              ignore
                (append_event kernel.journal "SOVEREIGN_EXEC_REFUSED"
                   (String.concat ":"
                      [ job.sovereign_job_id; sha256 reason; "255" ])))))
    kernel.sovereign_jobs

let handle_request kernel client line =
  let refuse code =
    queue client (control_line [ "ERR"; code ]);
    client.mode <- Awaiting
  in
  match split_on '\t' line with
  | [ "LOOM_EXEC/1"; "START"; instance; event_sha256; command_sha256;
      payload_sha256; payload_hex ] -> (
      try
        if instance <> kernel.instance_id then
          failf "sovereign-instance-mismatch";
        let payload = string_of_hex payload_hex in
        let job =
          start_sovereign_job kernel client ~event_sha256 ~command_sha256
            ~payload_sha256 ~payload
        in
        queue client
          (control_line
             [ "OK"; "SOVEREIGN_STARTED"; kernel.instance_id;
               kernel.kernel_generation; job.sovereign_job_id;
               job.sovereign_payload_sha256 ])
      with
      | Loom_error error
      | Loom_sovereign_exec.Error error ->
          ignore
            (append_event kernel.journal "SOVEREIGN_EXEC_REFUSED"
               (String.concat ":"
                  [ sha256 error; string_of_int client.peer_pid; "pre-exec" ]));
          refuse error
      | Unix_error (error, name, argument) ->
          let reason =
            Printf.sprintf "%s:%s(%s)" (Unix.error_message error) name argument
          in
          ignore
            (append_event kernel.journal "SOVEREIGN_EXEC_REFUSED"
               (String.concat ":"
                  [ sha256 reason; string_of_int client.peer_pid; "pre-exec" ]));
          refuse reason)
  | [ "LOOM_EXEC/1"; "WAIT"; instance; generation; job_id;
      payload_sha256 ] -> (
      try
        if instance <> kernel.instance_id then
          failf "sovereign-instance-mismatch";
        if generation <> kernel.kernel_generation then
          failf "sovereign-generation-mismatch";
        if not (valid_sha256 job_id && valid_sha256 payload_sha256) then
          failf "sovereign-result-identity-invalid";
        ignore
          (authenticate_exec_peer kernel client "sovereign-present"
             (Some job_id));
        reap_sovereign_jobs kernel;
        let job =
          match Hashtbl.find_opt kernel.sovereign_jobs job_id with
          | Some value -> value
          | None -> failf "sovereign-result-missing"
        in
        if job.sovereign_payload_sha256 <> payload_sha256 then
          failf "sovereign-result-payload-mismatch";
        (match job.sovereign_state with
        | Sovereign_running ->
            queue client
              (control_line [ "OK"; "SOVEREIGN_PENDING"; job_id ])
        | Sovereign_complete (path, record_sha256) ->
            queue client
              (control_line
                 [ "OK"; "SOVEREIGN_COMPLETE"; job_id;
                   hex_of_string path; record_sha256 ])
        | Sovereign_failed reason -> refuse reason)
      with
      | Loom_error error
      | Loom_sovereign_exec.Error error -> refuse error
      | Unix_error (error, name, argument) ->
          refuse
            (Printf.sprintf "%s:%s(%s)" (Unix.error_message error) name
               argument))
  | "LOOM_EXEC/1" :: _ ->
      ignore
        (append_event kernel.journal "SOVEREIGN_EXEC_REFUSED"
           (String.concat ":"
              [ sha256 line; string_of_int client.peer_pid; "unknown-operation" ]));
      refuse "sovereign-operation-refused"
  | magic :: token :: operation :: arguments
    when magic = Printf.sprintf "LOOM/%d" protocol_version && token = kernel.token -> (
      match (operation, arguments) with
      | "STATUS", [] ->
          let fields =
            status_fields kernel
            |> List.map (fun (key, value) -> key ^ "=" ^ field_escape value)
          in
          queue client (control_line ("OK" :: "STATUS" :: fields))
      | "EXEC_ISSUE", [ instance; cwd_hex; ttl_raw; payload_sha256; payload_hex ] -> (
          try
            if kernel.sovereign_exec_required then
              failf "legacy-exec-route-disabled";
            if instance <> kernel.instance_id then failf "exec-instance-mismatch";
            ignore (authenticate_exec_peer kernel client "issue" None);
            let ttl = parse_nonnegative "exec-ttl" ttl_raw in
            if ttl < 1 || ttl > 120 then failf "exec-ttl-out-of-range";
            let requested_cwd = string_of_hex cwd_hex |> Unix.realpath in
            if not (path_within kernel.cwd requested_cwd) then
              failf "exec-cwd-outside-worktree";
            let payload = string_of_hex payload_hex in
            if payload = "" || String.length payload > max_exec_capability_payload_bytes then
              failf "exec-capability-payload-size-refused";
            if sha256 payload <> payload_sha256 then
              failf "exec-capability-payload-digest-mismatch";
            let handle = random_hex 32 in
            let expires_us =
              Int64.add (current_time_us ())
                (Int64.mul (Int64.of_int ttl) 1_000_000L)
            in
            Hashtbl.add kernel.exec_grants handle
              { exec_payload = payload;
                exec_payload_sha256 = payload_sha256;
                exec_cwd = requested_cwd;
                exec_expires_us = expires_us;
                exec_generation = kernel.kernel_generation };
            ignore
              (append_event kernel.journal "EXEC_GRANT_ISSUED"
                 (String.concat ":"
                    [ sha256 handle; payload_sha256; Int64.to_string expires_us;
                      string_of_int client.peer_pid ]));
            queue client
              (control_line
                 [ "OK"; "EXEC_ISSUED"; kernel.instance_id;
                   kernel.kernel_generation; handle; Int64.to_string expires_us;
                   payload_sha256 ])
          with
          | Loom_error error ->
              ignore
                (append_event kernel.journal "EXEC_GRANT_REFUSED"
                   (sha256 error));
              refuse error
          | Unix_error (error, function_name, argument) ->
              let reason =
                Printf.sprintf "%s:%s(%s)" (Unix.error_message error)
                  function_name argument
              in
              ignore
                (append_event kernel.journal "EXEC_GRANT_REFUSED"
                   (sha256 reason));
              refuse reason)
      | "EXEC_CONSUME", [ instance; generation; handle ] -> (
          try
            if kernel.sovereign_exec_required then
              failf "legacy-exec-route-disabled";
            if instance <> kernel.instance_id then failf "exec-instance-mismatch";
            if generation <> kernel.kernel_generation then
              failf "exec-kernel-generation-mismatch";
            if not (valid_exec_handle handle) then failf "exec-handle-invalid";
            let peer_cwd =
              authenticate_exec_peer kernel client "consume" (Some handle)
            in
            let grant =
              match Hashtbl.find_opt kernel.exec_grants handle with
              | Some grant -> grant
              | None -> failf "exec-handle-missing-or-replayed"
            in
            if grant.exec_generation <> kernel.kernel_generation then
              failf "exec-grant-generation-mismatch";
            if current_time_us () > grant.exec_expires_us then (
              Hashtbl.remove kernel.exec_grants handle;
              failf "exec-grant-expired");
            if peer_cwd <> grant.exec_cwd then failf "exec-grant-cwd-mismatch";
            Hashtbl.remove kernel.exec_grants handle;
            Hashtbl.replace kernel.exec_outcomes handle
              { outcome_payload_sha256 = grant.exec_payload_sha256;
                outcome_cwd = grant.exec_cwd;
                outcome_generation = kernel.kernel_generation;
                outcome_peer_pid = client.peer_pid;
                outcome_peer_start = client.peer_start;
                outcome_consumed_us = current_time_us () };
            ignore
              (append_event kernel.journal "EXEC_GRANT_CONSUMED"
                 (String.concat ":"
                    [ sha256 handle; grant.exec_payload_sha256;
                      string_of_int client.peer_pid ]));
            queue client
              (control_line
                 [ "OK"; "EXEC_CONSUMED"; kernel.instance_id;
                   kernel.kernel_generation; grant.exec_payload_sha256;
                   hex_of_string grant.exec_payload ])
          with
          | Loom_error error ->
              ignore
                (append_event kernel.journal "EXEC_CONSUME_REFUSED"
                   (sha256 error));
              refuse error
          | Unix_error (error, function_name, argument) ->
              let reason =
                Printf.sprintf "%s:%s(%s)" (Unix.error_message error)
                  function_name argument
              in
              ignore
                (append_event kernel.journal "EXEC_CONSUME_REFUSED"
                   (sha256 reason));
              refuse reason)
      | "EXEC_OUTCOME",
        [ instance; generation; handle; receipt_sha256; receipt_hex ] -> (
          try
            if kernel.sovereign_exec_required then
              failf "legacy-exec-route-disabled";
            if instance <> kernel.instance_id then failf "exec-instance-mismatch";
            if generation <> kernel.kernel_generation then
              failf "exec-kernel-generation-mismatch";
            if not (valid_exec_handle handle) then failf "exec-handle-invalid";
            if not (valid_sha256 receipt_sha256) then
              failf "exec-outcome-receipt-digest-invalid";
            let peer_cwd =
              authenticate_exec_peer kernel client "outcome" (Some handle)
            in
            let obligation =
              match Hashtbl.find_opt kernel.exec_outcomes handle with
              | Some obligation -> obligation
              | None -> failf "exec-outcome-missing-or-replayed"
            in
            if obligation.outcome_generation <> kernel.kernel_generation then
              failf "exec-outcome-generation-mismatch";
            if obligation.outcome_peer_pid <> client.peer_pid
               || obligation.outcome_peer_start <> client.peer_start
            then failf "exec-outcome-broker-mismatch";
            if peer_cwd <> obligation.outcome_cwd then
              failf "exec-outcome-cwd-mismatch";
            let receipt = string_of_hex receipt_hex in
            if receipt = "" || String.length receipt > max_outcome_receipt_bytes
            then failf "exec-outcome-receipt-size-refused";
            if sha256 receipt <> receipt_sha256 then
              failf "exec-outcome-receipt-digest-mismatch";
            ignore
              (append_event kernel.journal "EXEC_OUTCOME_RECORDED"
                 (String.concat ":"
                    [ sha256 handle; obligation.outcome_payload_sha256;
                      receipt_sha256; kernel.kernel_generation;
                      string_of_int client.peer_pid ]));
            Hashtbl.remove kernel.exec_outcomes handle;
            queue client
              (control_line
                 [ "OK"; "EXEC_OUTCOME_RECORDED"; kernel.instance_id;
                   kernel.kernel_generation; receipt_sha256 ])
          with
          | Loom_error error ->
              ignore
                (append_event kernel.journal "EXEC_OUTCOME_REFUSED"
                   (sha256 error));
              refuse error
          | Unix_error (error, function_name, argument) ->
              let reason =
                Printf.sprintf "%s:%s(%s)" (Unix.error_message error)
                  function_name argument
              in
              ignore
                (append_event kernel.journal "EXEC_OUTCOME_REFUSED"
                   (sha256 reason));
              refuse reason)
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
            let submit_delay =
              min 0.35 (0.075 +. (float_of_int (String.length prompt) /. 4000.0))
            in
            Unix.sleepf submit_delay;
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
               if Buffer.length client.input > max_kernel_control_bytes then
                 close_client kernel descriptor
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
    let pidfd = ref None in
    (try
       Unix.set_close_on_exec descriptor;
       let peer_pid, peer_uid, peer_gid = peer_credentials descriptor in
       let peer_pidfd =
         match pidfd_open peer_pid with
         | Some descriptor -> descriptor
         | None -> failf "pidfd-open-refused"
       in
       pidfd := Some peer_pidfd;
       let peer_start = process_start peer_pid in
       let peer_pid_namespace = process_pid_namespace peer_pid in
       Unix.set_nonblock descriptor;
       kernel.next_client <- kernel.next_client + 1;
       let client =
         {
           fd = descriptor;
           id = Printf.sprintf "client-%d-%d" (Unix.getpid ()) kernel.next_client;
           peer_pid;
           peer_uid;
           peer_gid;
           peer_start;
           peer_pid_namespace;
           peer_pidfd;
           input = Buffer.create 256;
           mode = Awaiting;
           pending = "";
           pending_offset = 0;
         }
       in
       Hashtbl.add kernel.clients descriptor client
     with error ->
       Option.iter (fun fd -> try Unix.close fd with _ -> ()) !pidfd;
       (try Unix.close descriptor with _ -> ());
       ignore
         (append_event kernel.journal "PEER_REFUSED"
            (sha256 (Printexc.to_string error))))
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
  | None -> true
  | Some harness ->
      let ttl = Option.value ~default:"1800" (Sys.getenv_opt "SOUNIO_LOOM_COORD_TTL_SECONDS") in
      let identity = [ "--agent"; kernel.agent; "--lane"; kernel.lane ] in
      let claim_ready =
        coord_call kernel ("heartbeat" :: identity) = 0
        || coord_call kernel
             [ "scope"; "--agent"; kernel.agent; "--lane"; kernel.lane;
               "--intent"; Printf.sprintf "loom-supervised %s session" harness ]
           = 0
      in
      if not claim_ready then (
        Printf.eprintf "LOOM_COORDINATION_WARNING operation=heartbeat-or-scope\n%!";
        false)
      else
        let presence =
          [ "presence-register"; "--agent"; kernel.agent; "--lane"; kernel.lane;
            "--harness"; harness; "--session-id"; kernel.session_id; "--pid";
            string_of_int kernel.harness_pid; "--pid-start"; process_start kernel.harness_pid;
            "--boot-id"; trim (read_file "/proc/sys/kernel/random/boot_id");
            "--pid-namespace"; Unix.readlink (Printf.sprintf "/proc/%d/ns/pid" kernel.harness_pid);
            "--host"; Unix.gethostname (); "--ttl-seconds"; ttl ]
        in
        if coord_call kernel presence <> 0 then (
          Printf.eprintf "LOOM_COORDINATION_WARNING operation=presence-register\n%!";
          false)
        else
          let endpoint =
            [ "endpoint-register"; "--agent"; kernel.agent; "--lane"; kernel.lane;
              "--harness"; harness; "--transport"; "loom"; "--address";
              kernel.paths.socket_path; "--socket"; kernel.paths.socket_path;
              "--token-file"; kernel.paths.token_path; "--ttl-seconds"; ttl ]
          in
          if coord_call kernel endpoint <> 0 then (
            Printf.eprintf "LOOM_COORDINATION_WARNING operation=endpoint-register\n%!";
            false)
          else true

let coordination_retry_delay kernel =
  let base =
    match min kernel.coord_failures 6 with
    | 0 | 1 -> 1.0
    | 2 -> 2.0
    | 3 -> 4.0
    | 4 -> 8.0
    | 5 -> 16.0
    | _ -> 30.0
  in
  let lane_hash = Hashtbl.hash (kernel.agent ^ "/" ^ kernel.lane) land max_int in
  let spread = float_of_int (lane_hash mod 1000) /. 2000.0 in
  min 30.0 (base +. spread)

let finish_coordination_refresh kernel code =
  kernel.coord_pid <- None;
  if code = 0 then (
    kernel.coord_failures <- 0;
    kernel.next_coord_refresh <- Unix.gettimeofday () +. 300.0)
  else (
    kernel.coord_failures <- kernel.coord_failures + 1;
    let delay = coordination_retry_delay kernel in
    kernel.next_coord_refresh <- Unix.gettimeofday () +. delay;
    Printf.eprintf
      "LOOM_COORDINATION_RETRY failures=%d delay_seconds=%.3f exit_code=%d\n%!"
      kernel.coord_failures delay code)

let reap_coordination kernel =
  match kernel.coord_pid with
  | None -> ()
  | Some pid -> (
      match Unix.waitpid [ WNOHANG ] pid with
      | 0, _ -> ()
      | _, status -> finish_coordination_refresh kernel (process_exit_code status)
      | exception Unix_error (ECHILD, _, _) -> finish_coordination_refresh kernel 255)

let spawn_coordination_refresh kernel =
  reap_coordination kernel;
  if kernel.coord_pid = None then (
    match Unix.fork () with
    | 0 ->
        Sys.set_signal Sys.sigterm Sys.Signal_default;
        Sys.set_signal Sys.sigint Sys.Signal_default;
        (try exit (if refresh_coordination kernel then 0 else 75)
         with exn ->
           Printf.eprintf "LOOM_COORDINATION_WARNING operation=refresh error=%s\n%!"
             (Printexc.to_string exn);
           exit 1)
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
           [ "presence-unregister"; "--agent"; kernel.agent; "--lane"; kernel.lane ]);
      ignore
        (coord_call kernel
           [ "release"; "--agent"; kernel.agent; "--lane"; kernel.lane;
             "--reason"; "Loom session exited" ])

let run_kernel kernel =
  ignore
    (append_event kernel.journal "KERNEL_GENERATION"
       (String.concat ":"
          [ kernel.kernel_generation; kernel.boot_id; kernel.pid_namespace ]));
  write_descriptor kernel "active";
  if Sys.getenv_opt "SOUNIO_LOOM_COORD_AUTO" <> Some "0" then spawn_coordination_refresh kernel;
  let signal_stop _ = kernel.stopping <- true in
  Sys.set_signal Sys.sigterm (Sys.Signal_handle signal_stop);
  Sys.set_signal Sys.sigint (Sys.Signal_handle signal_stop);
  while not kernel.stopping && kernel.harness_exit = None do
    expire_exec_grants kernel;
    materialize_orphaned_exec_outcomes kernel;
    reap_sovereign_jobs kernel;
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
  materialize_incomplete_exec_outcomes kernel
    (if kernel.stopping then "kernel-stopping" else "harness-exited");
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
      let deadline = Unix.gettimeofday () +. start_ready_timeout () in
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
  let self_pid = Unix.getpid () in
  {
    paths;
    agent;
    lane;
    session_id;
    cwd;
    command_name = table_value guardian_values "command";
    command_digest = table_value guardian_values "argv_digest";
    instance_id;
    kernel_generation = random_hex 32;
    boot_id = trim (read_file "/proc/sys/kernel/random/boot_id");
    pid_namespace =
      process_pid_namespace (int_field "harness_pid");
    executable_path = process_executable self_pid;
    executable_sha256 = process_executable_sha256 self_pid;
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
    exec_grants = Hashtbl.create 16;
    exec_outcomes = Hashtbl.create 16;
    sovereign_grants = Hashtbl.create 4;
    sovereign_jobs = Hashtbl.create 16;
    sovereign_exec_required = true;
    next_client = 0;
    input_holder = None;
    output_cursor = ending;
    stopping = false;
    harness_exit = None;
    next_coord_refresh = 0.0;
    coord_pid = None;
    coord_failures = 0;
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

let unclosed_exec_outcome_digests events =
  let pending = Hashtbl.create 16 in
  let handle_digest event =
    match split_on ':' (string_of_hex event.payload_hex) with
    | digest :: _ when valid_sha256 digest -> Some digest
    | _ -> None
  in
  List.iter
    (fun event ->
      match event.kind, handle_digest event with
      | "EXEC_GRANT_CONSUMED", Some digest -> Hashtbl.replace pending digest ()
      | ("EXEC_OUTCOME_RECORDED" | "EXEC_OUTCOME_INCOMPLETE"), Some digest ->
          Hashtbl.remove pending digest
      | _ -> ())
    events;
  Hashtbl.fold (fun digest () values -> digest :: values) pending []

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
  List.iter
    (fun handle_digest ->
      ignore
        (append_event journal "EXEC_OUTCOME_INCOMPLETE"
           (String.concat ":"
              [ handle_digest; sha256 "kernel-recovery"; "recovery" ])))
    (unclosed_exec_outcome_digests events);
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

let journal_authority_state_path state_dir context_digest =
  Filename.concat state_dir (context_digest ^ ".state")

let fsync_directory path =
  let descriptor = Unix.openfile path [ O_RDONLY ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () -> Unix.fsync descriptor)

let journal_authority_response epoch principal signature =
  control_line
    [ "OK"; "SIGNED"; string_of_int epoch; principal; signature ]

let journal_authority_sign_request state_dir private_key public_key epoch
    principal_id fields =
  match fields with
  | [ context_digest; sequence; previous; event_hash ] ->
      if not (valid_sha256 context_digest) then
        failf "invalid-context-digest";
      if not (valid_sha256 previous) then failf "invalid-previous-digest";
      if not (valid_sha256 event_hash) then failf "invalid-event-digest";
      let sequence = positive_epoch "sequence" sequence in
      let path = journal_authority_state_path state_dir context_digest in
      if Sys.file_exists path then (
        let state = parse_key_values path in
        let stored_schema = table_value state "schema" in
        let stored_context = table_value state "journal_context_sha256" in
        let stored_epoch =
          positive_epoch "stored-epoch" (table_value state "epoch")
        in
        let stored_principal = table_value state "authority_principal_id" in
        let stored_sequence =
          positive_epoch "stored-sequence" (table_value state "sequence")
        in
        let stored_previous = table_value state "previous_sha256" in
        let stored_head = table_value state "event_sha256" in
        let stored_signature = table_value state "signature_base64" in
        if stored_schema <> "loom-journal-authority-state-v1"
           || stored_context <> context_digest || stored_epoch <> epoch
           || stored_principal <> principal_id || not (valid_sha256 stored_head)
           || not (valid_sha256 stored_previous) || stored_signature = ""
        then failf "stored-state-invalid";
        let stored_payload =
          journal_authority_payload stored_context stored_epoch stored_principal
            stored_sequence stored_previous stored_head
        in
        if not
             (journal_authority_signature_is_valid public_key state_dir
                stored_payload stored_signature)
        then failf "stored-state-signature-invalid";
        if sequence = stored_sequence && event_hash = stored_head
           && previous = stored_previous
        then journal_authority_response epoch principal_id stored_signature
        else if sequence <> stored_sequence + 1 || previous <> stored_head then
          failf "non-monotonic-event"
        else
          let payload =
            journal_authority_payload context_digest epoch principal_id sequence
              previous event_hash
          in
          let signature =
            journal_sign_with_key private_key state_dir payload
          in
          atomic_write path
            (Printf.sprintf
               "schema=loom-journal-authority-state-v1\njournal_context_sha256=%s\nepoch=%d\nauthority_principal_id=%s\nsequence=%d\nprevious_sha256=%s\nevent_sha256=%s\nsignature_base64=%s\n"
               context_digest epoch principal_id sequence previous event_hash
               signature);
          fsync_directory state_dir;
          journal_authority_response epoch principal_id signature)
      else (
        if sequence <> 1 || previous <> String.make 64 '0' then
          failf "first-event-not-genesis";
        let payload =
          journal_authority_payload context_digest epoch principal_id sequence
            previous event_hash
        in
        let signature = journal_sign_with_key private_key state_dir payload in
        atomic_write path
          (Printf.sprintf
             "schema=loom-journal-authority-state-v1\njournal_context_sha256=%s\nepoch=%d\nauthority_principal_id=%s\nsequence=%d\nprevious_sha256=%s\nevent_sha256=%s\nsignature_base64=%s\n"
             context_digest epoch principal_id sequence previous event_hash
             signature);
        fsync_directory state_dir;
        journal_authority_response epoch principal_id signature)
  | _ -> failf "invalid-sign-request"

let handle_journal_authority_request state_dir private_key public_key epoch
    principal_id line =
  match split_on '\t' line with
  | [ "SOUNIO_JOURNAL_AUTHORITY_V1"; "STATUS" ] ->
      control_line
        [ "OK"; "STATUS"; string_of_int epoch; principal_id ]
  | "SOUNIO_JOURNAL_AUTHORITY_V1" :: "SIGN" :: fields ->
      journal_authority_sign_request state_dir private_key public_key epoch
        principal_id fields
  | _ -> failf "invalid-request"

let journal_authority_serve_command cli =
  let socket_path = required cli "--socket" in
  let state_dir = required cli "--state-dir" in
  let private_key = regular_key_path "private" (required cli "--private-key") in
  let public_key = regular_key_path "public" (required cli "--public-key") in
  let epoch = positive_epoch "epoch" (required cli "--epoch") in
  if (Unix.stat private_key).st_perm land 0o077 <> 0 then
    failf "sounio-journal-authority-private-key-permissions";
  mkdir_p state_dir;
  Unix.chmod state_dir 0o700;
  mkdir_p (Filename.dirname socket_path);
  let principal_id = journal_principal_id public_key in
  let probe = "loom-journal-authority-keypair-probe-v1" in
  let probe_signature = journal_sign_with_key private_key state_dir probe in
  if not
       (journal_authority_signature_is_valid public_key state_dir probe
          probe_signature)
  then failf "sounio-journal-authority-keypair-mismatch";
  let lock_path = Filename.concat state_dir "authority.lock" in
  let lock = Unix.openfile lock_path [ O_WRONLY; O_CREAT ] 0o600 in
  Unix.set_close_on_exec lock;
  (try Unix.lockf lock F_TLOCK 0
   with Unix_error _ -> failf "sounio-journal-authority-already-active");
  let listener = create_unix_listener socket_path in
  let stopping = ref false in
  let stop _ = stopping := true in
  Sys.set_signal Sys.sigterm (Sys.Signal_handle stop);
  Sys.set_signal Sys.sigint (Sys.Signal_handle stop);
  Printf.printf
    "LOOM_JOURNAL_AUTHORITY_READY schema=loom-journal-authority-v1 epoch=%d principal_id=%s socket=%s\n%!"
    epoch principal_id socket_path;
  Fun.protect
    ~finally:(fun () ->
      (try Unix.close listener with _ -> ());
      (try Unix.unlink socket_path with _ -> ());
      (try Unix.close lock with _ -> ()))
    (fun () ->
      while not !stopping do
        let ready =
          try
            let ready, _, _ = Unix.select [ listener ] [] [] 1.0 in
            ready
          with Unix_error (EINTR, _, _) -> []
        in
        if ready <> [] then
          let client, _ = Unix.accept listener in
          Fun.protect
            ~finally:(fun () -> try Unix.close client with _ -> ())
            (fun () ->
              let response =
                try
                  handle_journal_authority_request state_dir private_key
                    public_key epoch principal_id (read_line_fd client)
                with Loom_error reason -> control_line [ "REFUSE"; reason ]
              in
              write_all client response)
      done)

let journal_authority_status_command cli =
  let socket_path = required cli "--socket" in
  let response =
    journal_authority_exchange socket_path
      "SOUNIO_JOURNAL_AUTHORITY_V1\tSTATUS\n"
  in
  match split_on '\t' response with
  | [ "OK"; "STATUS"; epoch; principal_id ] ->
      Printf.printf
        "LOOM_JOURNAL_AUTHORITY_STATUS state=active epoch=%s principal_id=%s socket=%s\n%!"
        epoch principal_id socket_path
  | _ -> failf "sounio-journal-authority-invalid-status"

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

let start_command ?(launch_source = "start")
    ?(ready_timeout = start_ready_timeout ()) cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let agent = required cli "--agent" in
  let lane = required cli "--lane" in
  let session_id = required cli "--session-id" in
  let command = Array.of_list cli.rest in
  if Array.length command = 0 then failf "start requires a command after --";
  let command_sha256 = command_argv_digest command in
  let paths = session_paths root agent lane in
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
  let launch_observation =
    Loom_membrane.observe_product_launch
      ~policy_root:(product_activation_policy_root ()) ~audit_root:root
      ~operation:"start" ~launch_source ~agent ~lane ~session_id ~cwd
      ~command_sha256 ~deadline_ms:15_000
  in
  mkdir_p paths.session_dir;
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
      let deadline = Unix.gettimeofday () +. ready_timeout in
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
      Printf.printf
        "LOOM_STARTED agent=%s lane=%s instance=%s daemon_pid=%d harness_pid=%s launch_source=%s launch_dark_code=%d launch_dark_projection_sha256=%s launch_dark_generation_sha256=%s launch_dark_pid=%d launch_dark_sequence=%d authorizing=false production_activation=false\n%!"
        agent lane (table_value values "instance_id") daemon_pid
        (table_value values "harness_pid") launch_source
        launch_observation.launch_code
        launch_observation.launch_projection_sha256
        launch_observation.launch_authority_generation_sha256
        launch_observation.launch_authority_pid
        launch_observation.launch_authority_sequence

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
  let session_id = table_value descriptor "session_id" in
  let command_sha256 = table_value descriptor "argv_digest" in
  let launch_observation =
    Loom_membrane.observe_product_launch
      ~policy_root:(product_activation_policy_root ()) ~audit_root:root
      ~operation:"recover" ~launch_source:"recover" ~agent ~lane ~session_id ~cwd
      ~command_sha256 ~deadline_ms:15_000
  in
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
        "LOOM_RECOVERED agent=%s lane=%s instance=%s daemon_pid=%d guardian_pid=%s harness_pid=%s cursor=%s launch_source=recover launch_dark_code=%d launch_dark_projection_sha256=%s launch_dark_generation_sha256=%s launch_dark_pid=%d launch_dark_sequence=%d authorizing=false production_activation=false\n%!"
        agent lane (table_value values "instance_id") daemon_pid
        (table_value guardian_before "guardian_pid")
        (table_value values "harness_pid") (table_value values "output_cursor")
        launch_observation.launch_code
        launch_observation.launch_projection_sha256
        launch_observation.launch_authority_generation_sha256
        launch_observation.launch_authority_pid
        launch_observation.launch_authority_sequence

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

let offline_snapshot paths cursor limit =
  let values = parse_key_values paths.descriptor_path in
  if table_value values "state" <> "exited" then
    failf "offline snapshot requires an exited session";
  let instance = table_value values "instance_id" in
  let output_path = table_value values "output_file" in
  let journal_path = table_value values "journal_file" in
  let guardian_journal_path = table_value values "guardian_journal_file" in
  if
    List.exists (( = ) "")
      [ instance; output_path; journal_path; guardian_journal_path ]
  then failf "exited session descriptor is incomplete";
  let _, semantic_phase, _ = load_and_verify_journal journal_path in
  if semantic_phase <> Exited then failf "semantic journal is not terminal";
  let guardian_events, guardian_phase, guardian_cursor, _ =
    load_and_verify_guardian_journal guardian_journal_path
  in
  if guardian_phase <> Guardian_exited then failf "guardian journal is not terminal";
  let ending = guardian_cursor in
  if cursor > ending then failf "cursor ahead of durable output";
  let length = min limit (ending - cursor) in
  let data =
    verified_guardian_output_range guardian_events output_path ending cursor length
  in
  (instance, cursor, cursor + length, data)

let snapshot_command cli =
  let _, paths = session_locator cli in
  let cursor = optional cli "--cursor" |> Option.value ~default:"0" |> parse_nonnegative "cursor" in
  let limit =
    optional cli "--limit" |> Option.value ~default:(string_of_int max_snapshot_bytes)
    |> parse_nonnegative "limit" |> min max_snapshot_bytes
  in
  let descriptor = parse_key_values paths.descriptor_path in
  let source, (instance, start, ending, data) =
    try ("kernel", snapshot_request paths cursor limit)
    with error ->
      if table_value descriptor "state" = "exited" then
        ("offline", offline_snapshot paths cursor limit)
      else raise error
  in
  if flag cli "--meta" then
    Printf.eprintf
      "LOOM_SNAPSHOT instance=%s start=%d end=%d bytes=%d source=%s\n%!"
      instance start ending (String.length data) source;
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
    { channel; descriptor; seq = List.length events; previous = digest;
      authority_context = None;
      authority_directory = Filename.dirname output_path }
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
           "{\"agent\":\"%s\",\"lane\":\"%s\",\"session_id\":\"%s\",\"instance_id\":\"%s\",\"state\":\"%s\",\"daemon_pid\":\"%s\",\"guardian_pid\":\"%s\",\"harness_pid\":\"%s\",\"cursor\":%d,\"started_utc\":\"%s\",\"worktree\":\"%s\",\"command\":\"%s\"}"
           (json_escape (table_value values "agent")) (json_escape (table_value values "lane"))
           (json_escape (table_value values "session_id")) (json_escape (table_value values "instance_id"))
           (json_escape (table_value values "state")) (json_escape (table_value values "daemon_pid"))
           (json_escape (table_value values "guardian_pid"))
           (json_escape (table_value values "harness_pid")) (file_size output)
           (json_escape (table_value values "started_utc"))
           (json_escape (table_value values "worktree"))
           (json_escape (table_value values "command")))
  |> String.concat "," |> Printf.sprintf "[%s]"

let take_last limit values =
  let length = List.length values in
  if length <= limit then values
  else
    let rec drop count remaining =
      if count <= 0 then remaining
      else match remaining with [] -> [] | _ :: tail -> drop (count - 1) tail
    in
    drop (length - limit) values

let journal_event_json source (event : journal_event) =
  Printf.sprintf
    "{\"source\":\"%s\",\"seq\":%d,\"utc\":\"%s\",\"kind\":\"%s\",\"hash\":\"%s\"}"
    source event.seq (json_escape event.utc) (json_escape event.kind)
    (json_escape event.hash)

let spectral_event values source head (event : journal_event) =
  Loom_arrow.
    { agent = table_value values "agent";
      lane = table_value values "lane";
      instance_id = table_value values "instance_id";
      session_state = table_value values "state";
      journal = source;
      sequence = Int64.of_int event.seq;
      observed_at_utc = event.utc;
      kind = event.kind;
      payload = string_of_hex event.payload_hex;
      previous_sha256 = event.previous;
      event_sha256 = event.hash;
      journal_head_sha256 = head;
      verified = true }

type verified_session_journals = {
  semantic_events : journal_event list;
  semantic_head : string;
  guardian_journal : (journal_event list * string) option;
}

let guardianless_runtime_versions = [ "2026.08.24.0" ]
let guardian_release_utc = "2026-08-24T08:58:13Z"

let verify_guardianless_generation values semantic_path runtime semantic_events
    semantic_phase =
  let generation_dir = Filename.dirname semantic_path in
  let snapshot_path = Filename.concat generation_dir "session.state" in
  if Sys.file_exists snapshot_path then (
    let snapshot = parse_key_values snapshot_path in
    let snapshot_runtime = table_value snapshot "runtime_version" in
    if snapshot_runtime <> runtime then
      failf
        "guardianless-generation-runtime-mismatch:descriptor=%s:generation=%s"
        runtime (if snapshot_runtime = "" then "unknown" else snapshot_runtime);
    let descriptor_instance = table_value values "instance_id" in
    let snapshot_instance = table_value snapshot "instance_id" in
    if descriptor_instance = "" || snapshot_instance <> descriptor_instance then
      failf
        "guardianless-generation-instance-mismatch:descriptor=%s:generation=%s"
        (if descriptor_instance = "" then "unknown" else descriptor_instance)
        (if snapshot_instance = "" then "unknown" else snapshot_instance);
    let snapshot_journal = table_value snapshot "journal_file" in
    if snapshot_journal <> semantic_path then
      failf "guardianless-generation-journal-mismatch";
    let snapshot_guardian = table_value snapshot "guardian_journal_file" in
    if snapshot_guardian <> "" then
      failf "guardianless-generation-declares-guardian:path=%s"
        snapshot_guardian);
  let hidden_guardian = Filename.concat generation_dir "guardian.tsv" in
  if Sys.file_exists hidden_guardian then
    failf "guardianless-generation-hides-guardian:path=%s" hidden_guardian;
  if semantic_phase <> Exited then
    failf "guardianless-semantic-journal-not-terminal";
  match semantic_events with
  | (first : journal_event) :: _
    when first.seq = 1 && first.kind = "SESSION_STARTED"
         && first.utc < guardian_release_utc -> ()
  | (first : journal_event) :: _
    when first.seq = 1 && first.kind = "SESSION_STARTED" ->
      failf "guardianless-session-after-guardian-release:started=%s:release=%s"
        first.utc guardian_release_utc
  | _ -> failf "guardianless-session-start-receipt-missing"

let load_verified_session_journals values =
  let runtime = table_value values "runtime_version" in
  let semantic_path = table_value values "journal_file" in
  if semantic_path = "" then failf "semantic-journal-required";
  if not (Sys.file_exists semantic_path) then
    failf "semantic-journal-missing:path=%s" semantic_path;
  let semantic_events, semantic_phase, semantic_head =
    load_and_verify_journal semantic_path
  in
  let guardian_path = table_value values "guardian_journal_file" in
  let guardian_journal =
    if guardian_path = "" then (
      if not (List.mem runtime guardianless_runtime_versions) then
        failf "guardian-journal-required:runtime-version=%s"
          (if runtime = "" then "unknown" else runtime);
      verify_guardianless_generation values semantic_path runtime semantic_events
        semantic_phase;
      None)
    else (
      if not (Sys.file_exists guardian_path) then
        failf "guardian-journal-missing:path=%s" guardian_path;
      let events, _, _, head =
        load_and_verify_guardian_journal guardian_path
      in
      Some (events, head))
  in
  { semantic_events; semantic_head; guardian_journal }

let session_spectral_events (_, values) =
  let journals = load_verified_session_journals values in
  let guardian_events =
    match journals.guardian_journal with
    | None -> []
    | Some (events, head) ->
        List.map (spectral_event values "guardian" head) events
  in
  ( List.map
      (spectral_event values "semantic" journals.semantic_head)
      journals.semantic_events
    @ guardian_events,
    journals.guardian_journal = None )

let epistemic_spectral_events root =
  Loom_epistemic.spectral_events root
  |> List.map (fun event ->
         Loom_arrow.
           { agent = event.Loom_epistemic.spectral_agent;
             lane = event.spectral_lane;
             instance_id = event.spectral_world;
             session_state = event.spectral_state;
             journal = "epistemic-worldline";
             sequence = event.spectral_sequence;
             observed_at_utc = event.spectral_observed_at_utc;
             kind = event.spectral_kind;
             payload = event.spectral_payload;
             previous_sha256 = event.spectral_previous_sha256;
             event_sha256 = event.spectral_event_sha256;
             journal_head_sha256 = event.spectral_head_sha256;
             verified = true })

type spectral_projection = {
  spectral_rows : Loom_arrow.event list;
  guardian_sessions : int;
  legacy_semantic_only_sessions : int;
}

let spectral_projection root =
  let session_events, guardian_sessions, legacy_semantic_only_sessions =
    session_descriptors root
    |> List.fold_left
         (fun (events, guardian_count, legacy_count) descriptor ->
           let projected, legacy = session_spectral_events descriptor in
           ( List.rev_append projected events,
             guardian_count + (if legacy then 0 else 1),
             legacy_count + (if legacy then 1 else 0) ))
         ([], 0, 0)
  in
  let spectral_rows =
    session_events @ epistemic_spectral_events root
    |> List.sort (fun left right ->
           compare
             ( left.Loom_arrow.observed_at_utc,
               left.agent,
               left.lane,
               left.journal,
               left.sequence )
             ( right.Loom_arrow.observed_at_utc,
               right.agent,
               right.lane,
               right.journal,
               right.sequence ))
  in
  { spectral_rows; guardian_sessions; legacy_semantic_only_sessions }

let events_arrow root =
  let projection = spectral_projection root in
  let bytes =
    try Loom_arrow.encode projection.spectral_rows
    with Failure message -> failf "arrow-ipc-encode:%s" message
  in
  (bytes, projection)

let export_events_arrow_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let output = required cli "--out" in
  let bytes, projection = events_arrow root in
  atomic_write output bytes;
  Printf.printf
    "LOOM_ARROW_EXPORTED schema=loom-spectral-events-v1 authority=verified-derived rows=%d guardian_sessions=%d legacy_semantic_only_sessions=%d bytes=%d output=%s\n%!"
    (List.length projection.spectral_rows) projection.guardian_sessions
    projection.legacy_semantic_only_sessions (String.length bytes) output

let verify_events_arrow_command cli =
  let path = required cli "--file" in
  let bytes = read_file path in
  let summary =
    try Loom_arrow.inspect bytes
    with Failure message -> failf "arrow-ipc-verify:%s" message
  in
  Printf.printf "LOOM_ARROW_OK %s bytes=%d file=%s\n%!" summary
    (String.length bytes) path

let epistemic_root cli =
  let cwd = cwd_option cli in
  root_option cli cwd

let epistemic_print action =
  try Printf.printf "%s\n%!" (action ())
  with Loom_epistemic.Error message -> failf "%s" message

let world_create_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.create ~root ~world:(required cli "--world")
        ~agent:(required cli "--agent") ~lane:(required cli "--lane"))

let knowledge_observe_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.observe ~root ~world:(required cli "--world")
        ~knowledge:(required cli "--knowledge") ~value:(required cli "--value")
        ~arithmetic_error:(required cli "--error")
        ~uncertainty:(required cli "--uncertainty")
        ~confidence:(required cli "--confidence")
        ~provenance:(required cli "--provenance"))

let epistemic_claim_open_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.open_claim ~root ~world:(required cli "--world")
        ~claim:(required cli "--claim") ~knowledge:(required cli "--knowledge")
        ~evidence:(required cli "--evidence"))

let epistemic_claim_challenge_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.challenge ~root ~world:(required cli "--world")
        ~claim:(required cli "--claim")
        ~challenge:(required cli "--challenge")
        ~falsifier:(required cli "--falsifier"))

let epistemic_capability_acquire_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.acquire_capability ~root ~world:(required cli "--world")
        ~capability:(required cli "--capability")
        ~resource:(required cli "--resource") ~owner:(required cli "--owner")
        ~generation:(required cli "--generation"))

let epistemic_capability_release_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.release_capability ~root ~world:(required cli "--world")
        ~capability:(required cli "--capability")
        ~owner:(required cli "--owner")
        ~generation:(required cli "--generation"))

let world_fork_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.fork ~root ~parent:(required cli "--parent")
        ~child:(required cli "--child") ~agent:(required cli "--agent")
        ~lane:(required cli "--lane")
        ~hypothesis:(required cli "--hypothesis")
        ~expected_parent_head:
          (Option.value ~default:"" (optional cli "--parent-head")))

let world_status_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.status ~root ~world:(required cli "--world"))

let world_verify_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.verify ~root ~world:(required cli "--world"))

let world_list_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () -> Loom_epistemic.list ~root)

let attention_budget cli =
  let raw = required cli "--budget" in
  try int_of_string raw
  with _ -> failf "attention budget must be an integer: %s" raw

let attention_compile_command cli =
  let root = epistemic_root cli in
  let policy =
    Loom_epistemic.attention_policy_of_string (required cli "--policy")
  in
  epistemic_print (fun () ->
      Loom_epistemic.compile_attention ~root ~world:(required cli "--world")
        ~plan:(required cli "--plan")
        ~candidate_file:(required cli "--candidates")
        ~budget:(attention_budget cli) ~policy ~owner:(required cli "--owner")
        ~generation:(required cli "--generation"))

let attention_complete_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.complete_attention ~root ~world:(required cli "--world")
        ~plan:(required cli "--plan") ~owner:(required cli "--owner")
        ~generation:(required cli "--generation")
        ~outcome:(required cli "--outcome"))

let portfolio_integer cli option =
  let raw = required cli option in
  try int_of_string raw
  with _ -> failf "portfolio %s must be an integer: %s" option raw

let attention_portfolio_compile_command cli =
  let root = epistemic_root cli in
  let policy =
    Loom_epistemic.attention_policy_of_string (required cli "--policy")
  in
  epistemic_print (fun () ->
      Loom_epistemic.compile_attention_portfolio ~root
        ~world:(required cli "--world")
        ~portfolio:(required cli "--portfolio")
        ~candidate_file:(required cli "--candidates")
        ~token_budget:(portfolio_integer cli "--token-budget")
        ~wall_budget:(portfolio_integer cli "--wall-budget")
        ~gpu_budget:(portfolio_integer cli "--gpu-budget")
        ~quota_budget:(portfolio_integer cli "--quota-budget")
        ~policy ~owner:(required cli "--owner")
        ~generation:(required cli "--generation"))

let attention_portfolio_complete_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.complete_attention_portfolio ~root
        ~world:(required cli "--world")
        ~portfolio:(required cli "--portfolio")
        ~owner:(required cli "--owner")
        ~generation:(required cli "--generation")
        ~outcome:(required cli "--outcome"))

let contingent_integer cli option =
  let raw = required cli option in
  try int_of_string raw
  with _ -> failf "contingent policy %s must be an integer: %s" option raw

let contingent_policy_compile_command cli =
  let root = epistemic_root cli in
  let order =
    Loom_epistemic.attention_policy_of_string (required cli "--order")
  in
  epistemic_print (fun () ->
      Loom_epistemic.compile_contingent_policy ~root
        ~world:(required cli "--world")
        ~policy_id:(required cli "--contingent-policy")
        ~root_state:(required cli "--root-state")
        ~action_file:(required cli "--actions")
        ~outcome_file:(required cli "--outcomes")
        ~token_budget:(contingent_integer cli "--token-budget")
        ~wall_budget:(contingent_integer cli "--wall-budget")
        ~gpu_budget:(contingent_integer cli "--gpu-budget")
        ~quota_budget:(contingent_integer cli "--quota-budget")
        ~order ~owner:(required cli "--owner")
        ~generation:(required cli "--generation")
        ~measurement_principal:(optional cli "--measurement-principal")
        ~measurement_public_key_file:
          (optional cli "--measurement-public-key")
        ~classifier_principal:(optional cli "--classifier-principal")
        ~classifier_public_key_file:(optional cli "--classifier-public-key")
        ~classifier_spec_digest:(optional cli "--classifier-spec-digest"))

let contingent_measurement_attest_command cli =
  let root = epistemic_root cli in
  let receipt_path = required cli "--receipt" in
  let canonical, receipt_digest, measurement_digest =
    Loom_epistemic.attest_contingent_measurement ~root
      ~world:(required cli "--world")
      ~policy_id:(required cli "--contingent-policy")
      ~principal:(required cli "--measurement-principal")
      ~nonce:(required cli "--measurement-nonce")
      ~private_key:(required cli "--measurement-private-key")
      ~measurement_bytes:
        (read_file_bounded "measurement" max_outcome_measurement_bytes
           (required cli "--measurement"))
  in
  atomic_write receipt_path canonical;
  Printf.printf
    "LOOM_MEASUREMENT_ATTESTED schema=loom-outcome-evidence-authority-v0 measurement_sha256=%s receipt_sha256=%s receipt=%s\n%!"
    measurement_digest receipt_digest receipt_path

let contingent_classification_attest_command cli =
  let root = epistemic_root cli in
  let receipt_path = required cli "--receipt" in
  let canonical, receipt_digest =
    Loom_epistemic.attest_contingent_classification ~root
      ~world:(required cli "--world")
      ~policy_id:(required cli "--contingent-policy")
      ~outcome_id:(required cli "--outcome")
      ~principal:(required cli "--classifier-principal")
      ~private_key:(required cli "--classifier-private-key")
      ~measurement_canonical:
        (read_file_bounded "measurement receipt" max_outcome_receipt_bytes
           (required cli "--measurement-receipt"))
  in
  atomic_write receipt_path canonical;
  Printf.printf
    "LOOM_CLASSIFICATION_ATTESTED schema=loom-outcome-evidence-authority-v0 receipt_sha256=%s receipt=%s\n%!"
    receipt_digest receipt_path

let contingent_policy_observe_attested_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.observe_contingent_policy_attested ~root
        ~world:(required cli "--world")
        ~policy_id:(required cli "--contingent-policy")
        ~owner:(required cli "--owner")
        ~generation:(required cli "--generation")
        ~measurement_canonical:
          (read_file_bounded "measurement receipt" max_outcome_receipt_bytes
             (required cli "--measurement-receipt"))
        ~classification_canonical:
          (read_file_bounded "classification receipt"
             max_outcome_receipt_bytes
             (required cli "--classification-receipt")))

let contingent_policy_observe_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_epistemic.observe_contingent_policy ~root
        ~world:(required cli "--world")
        ~policy_id:(required cli "--contingent-policy")
        ~outcome_id:(required cli "--outcome")
        ~owner:(required cli "--owner")
        ~generation:(required cli "--generation")
        ~outcome_digest:(required cli "--outcome-digest"))

let witness_port cli =
  match optional cli "--port" with
  | None -> 0
  | Some raw ->
      let value =
        try int_of_string raw
        with _ -> failf "witness --port must be an integer: %s" raw
      in
      if value < 0 || value > 65535 then
        failf "witness --port must be between 0 and 65535: %d" value;
      value

let witness_serve_command cli =
  Loom_witness.serve
    ~state_dir:(required cli "--witness-state-dir")
    ~membership_file:(required cli "--membership")
    ~witness_id:(required cli "--witness")
    ~private_key:(required cli "--private-key")
    ~bind:(optional cli "--bind" |> Option.value ~default:"127.0.0.1")
    ~port:(witness_port cli)

let witness_mesh_anchor_command cli =
  let root = epistemic_root cli in
  epistemic_print (fun () ->
      Loom_witness.anchor ~root ~world:(required cli "--world")
        ~membership_file:(required cli "--membership")
        ~endpoints_file:(required cli "--endpoints")
        ~anchor_private_key:(required cli "--anchor-private-key"))

let witness_mesh_verify_command cli =
  let root = epistemic_root cli in
  let policy =
    optional cli "--policy" |> Option.value ~default:"byzantine-strict"
    |> Loom_witness.verification_policy_of_string
  in
  epistemic_print (fun () ->
      Loom_witness.verify ~root ~world:(required cli "--world")
        ~membership_file:(required cli "--membership")
        ~endpoints_file:(required cli "--endpoints") ~policy)

let witness_epoch_integer cli option =
  let raw = required cli option in
  let value =
    try int_of_string raw
    with _ -> failf "witness epoch %s must be an integer: %s" option raw
  in
  if value <= 0 then
    failf "witness epoch %s must be positive: %d" option value;
  value

let witness_epoch_handoff_command cli =
  epistemic_print (fun () ->
      Loom_witness_epoch.handoff
        ~epoch_state_dir:(required cli "--epoch-state-dir")
        ~world:(required cli "--world")
        ~from_epoch:(witness_epoch_integer cli "--from-epoch")
        ~to_epoch:(witness_epoch_integer cli "--to-epoch")
        ~old_root:(required cli "--old-state-dir")
        ~old_membership_file:(required cli "--old-membership")
        ~old_endpoints_file:(required cli "--old-endpoints")
        ~new_root:(required cli "--new-state-dir")
        ~new_membership_file:(required cli "--new-membership")
        ~new_endpoints_file:(required cli "--new-endpoints"))

let witness_epoch_verify_command cli =
  epistemic_print (fun () ->
      Loom_witness_epoch.verify_active
        ~epoch_state_dir:(required cli "--epoch-state-dir")
        ~world:(required cli "--world")
        ~active_root:(required cli "--active-state-dir")
        ~membership_file:(required cli "--membership")
        ~endpoints_file:(required cli "--endpoints"))

let witness_epoch_transparency_port cli =
  match optional cli "--log-port" with
  | None -> 0
  | Some raw ->
      let value =
        try int_of_string raw
        with _ -> failf "witness epoch transparency --log-port must be an integer: %s" raw
      in
      if value < 0 || value > 65535 then
        failf "witness epoch transparency --log-port must be between 0 and 65535: %d"
          value;
      value

let witness_epoch_log_serve_command cli =
  Loom_witness_transparency.serve
    ~state_dir:(required cli "--log-state-dir")
    ~operator:(required cli "--operator")
    ~operator_public_key_file:(required cli "--operator-public-key")
    ~operator_private_key:(required cli "--operator-private-key")
    ~publisher_public_key_file:(required cli "--publisher-public-key")
    ~bind:(optional cli "--bind" |> Option.value ~default:"127.0.0.1")
    ~port:(witness_epoch_transparency_port cli)

let witness_epoch_log_status_command cli =
  epistemic_print (fun () ->
      Loom_witness_transparency.status
        ~host:(required cli "--log-host")
        ~port:(witness_epoch_transparency_port cli)
        ~operator:(required cli "--operator")
        ~operator_public_key_file:(required cli "--operator-public-key")
        ~world:(required cli "--world"))

let witness_epoch_transparency_publish_command cli =
  epistemic_print (fun () ->
      Loom_witness_transparency.publish
        ~epoch_state_dir:(required cli "--epoch-state-dir")
        ~transparency_state_dir:(required cli "--transparency-state-dir")
        ~world:(required cli "--world")
        ~log_host:(required cli "--log-host")
        ~log_port:(witness_epoch_transparency_port cli)
        ~operator:(required cli "--operator")
        ~operator_public_key_file:(required cli "--operator-public-key")
        ~publisher_public_key_file:(required cli "--publisher-public-key")
        ~publisher_private_key:(required cli "--publisher-private-key")
        ~membership_file:(required cli "--transparency-membership")
        ~endpoints_file:(required cli "--transparency-endpoints")
        ~anchor_private_key:(required cli "--transparency-anchor-private-key"))

let witness_epoch_transparency_verify_command cli =
  epistemic_print (fun () ->
      Loom_witness_transparency.verify
        ~epoch_state_dir:(required cli "--epoch-state-dir")
        ~transparency_state_dir:(required cli "--transparency-state-dir")
        ~world:(required cli "--world")
        ~log_host:(required cli "--log-host")
        ~log_port:(witness_epoch_transparency_port cli)
        ~operator:(required cli "--operator")
        ~operator_public_key_file:(required cli "--operator-public-key")
        ~membership_file:(required cli "--transparency-membership")
        ~endpoints_file:(required cli "--transparency-endpoints"))

let session_events_json (_, values) =
  let agent = table_value values "agent" in
  let lane = table_value values "lane" in
  let instance = table_value values "instance_id" in
  let state = table_value values "state" in
  try
    let journals = load_verified_session_journals values in
    let guardian_events, guardian_head, journal_profile =
      match journals.guardian_journal with
      | Some (events, head) -> (events, head, "semantic+guardian")
      | None -> ([], "", "semantic-only-legacy")
    in
    let recoveries =
      List.fold_left
        (fun count (event : journal_event) ->
          if event.kind = "KERNEL_RECOVERED" then count + 1 else count)
        0 journals.semantic_events
    in
    let events =
      (List.map (fun event -> (event.utc, journal_event_json "semantic" event))
         (take_last 128 journals.semantic_events))
      @ (List.map (fun event -> (event.utc, journal_event_json "guardian" event))
           (take_last 128 guardian_events))
      |> List.sort (fun (left, _) (right, _) -> compare left right)
      |> List.map snd |> String.concat ","
    in
    Printf.sprintf
      "{\"agent\":\"%s\",\"lane\":\"%s\",\"instance_id\":\"%s\",\"state\":\"%s\",\"verified\":true,\"journal_profile\":\"%s\",\"recoveries\":%d,\"semantic_head\":\"%s\",\"guardian_head\":\"%s\",\"events\":[%s]}"
      (json_escape agent) (json_escape lane) (json_escape instance)
      (json_escape state) (json_escape journal_profile) recoveries
      (json_escape journals.semantic_head)
      (json_escape guardian_head) events
  with error ->
    Printf.sprintf
      "{\"agent\":\"%s\",\"lane\":\"%s\",\"instance_id\":\"%s\",\"state\":\"%s\",\"verified\":false,\"recoveries\":0,\"error\":\"%s\",\"events\":[]}"
      (json_escape agent) (json_escape lane) (json_escape instance)
      (json_escape state) (json_escape (Printexc.to_string error))

let events_json root =
  session_descriptors root |> List.map session_events_json |> String.concat ","
  |> Printf.sprintf "[%s]"

let lane_health_parity_command () =
  let sabotage_index =
    match Sys.getenv_opt "SOUNIO_LOOM_LANE_HEALTH_PARITY_SABOTAGE_INDEX" with
    | None -> None
    | Some value ->
        (try Some (int_of_string value)
         with _ -> failf "lane-health-parity-invalid-sabotage-index:%s" value)
  in
  Printf.printf "%s\n%!"
    (Loom_lane_health.parity_line ?sabotage_index
       ~prefix:"OCAML_LANE_HEALTH_PARITY" ())

type authority_lane = {
  authority_agent : string;
  authority_lane : string;
  mutable authority_claim : string;
  mutable authority_presence : string;
  mutable authority_presence_reason : string;
  mutable authority_endpoint : string;
  mutable authority_transport : string;
  mutable authority_harness : string;
  mutable authority_session_id : string;
  mutable authority_generation : string;
  mutable authority_pid : string;
  mutable authority_last_seen : string;
  mutable authority_worktree : string;
  mutable authority_loom_state : string;
  mutable authority_loom_instance : string;
  mutable authority_guardian_pid : string;
  mutable authority_harness_pid : string;
  mutable authority_harness_pid_start : string;
  mutable authority_started_utc : string;
  mutable authority_command : string;
  mutable authority_cursor : int;
  mutable authority_pending_obligations : int;
  mutable authority_active_obligations : int;
  mutable authority_blocker_active : bool;
  mutable authority_obligation_census_complete : bool;
  mutable authority_progress_observed : bool;
  mutable authority_progress_window_complete : bool;
  mutable authority_ready_observed : bool;
  mutable authority_observation_authorized : bool;
  mutable authority_sample_fresh : bool;
}

let empty_authority_lane agent lane =
  { authority_agent = agent; authority_lane = lane; authority_claim = "missing";
    authority_presence = "missing"; authority_presence_reason = "no-record";
    authority_endpoint = "missing"; authority_transport = "none";
    authority_harness = "unknown"; authority_session_id = "";
    authority_generation = ""; authority_pid = ""; authority_last_seen = "";
    authority_worktree = ""; authority_loom_state = "none";
    authority_loom_instance = ""; authority_guardian_pid = "";
    authority_harness_pid = ""; authority_harness_pid_start = "";
    authority_started_utc = ""; authority_command = ""; authority_cursor = 0;
    authority_pending_obligations = 0; authority_active_obligations = 0;
    authority_blocker_active = false;
    authority_obligation_census_complete = false;
    authority_progress_observed = false;
    authority_progress_window_complete = false;
    authority_ready_observed = false;
    authority_observation_authorized = false;
    authority_sample_fresh = false }

let authority_key agent lane = agent ^ "\000" ^ lane

let authority_entry lanes agent lane =
  let key = authority_key agent lane in
  match Hashtbl.find_opt lanes key with
  | Some value -> value
  | None ->
      let value = empty_authority_lane agent lane in
      Hashtbl.add lanes key value;
      value

type coordination_snapshot_command = {
  snapshot_command_path : string;
  snapshot_command_authorized : bool;
}

let coordination_snapshot_command cwd =
  match Sys.getenv_opt "SOUNIO_COORD_COMMAND" with
  | Some path when Sys.file_exists path ->
      Some { snapshot_command_path = path; snapshot_command_authorized = false }
  | _ ->
      let sibling =
        Filename.concat (Filename.dirname Sys.executable_name)
          "sounio-coord-runtime"
      in
      if Sys.file_exists sibling then
        Some
          { snapshot_command_path = sibling;
            snapshot_command_authorized = true }
      else
        let launcher = Filename.concat (Filename.concat cwd "bin") "sounio-coord" in
        if Sys.file_exists launcher then
          Some
            { snapshot_command_path = launcher;
              snapshot_command_authorized = true }
        else None

let snapshot_fields values =
  let fields = Hashtbl.create 16 in
  List.iter
    (fun field ->
      match String.index_opt field '=' with
      | None -> ()
      | Some index ->
          Hashtbl.replace fields (String.sub field 0 index)
            (String.sub field (index + 1) (String.length field - index - 1)))
    values;
  fields

let load_authority_snapshot cwd lanes =
  match coordination_snapshot_command cwd with
  | None -> (false, "", false)
  | Some command_spec ->
      let command = command_spec.snapshot_command_path in
      let code, output =
        process_output_all cwd command [| command; "cockpit-snapshot" |]
      in
      if code <> 0 then (false, "", false)
      else
        let valid = ref false and snapshot_utc = ref "" in
        output |> split_on '\n'
        |> List.iter (fun line ->
               match split_on '\t' line with
               | "COCKPIT" :: values ->
                   let fields = snapshot_fields values in
                   if table_value fields "protocol" = "1" then valid := true;
                   snapshot_utc := table_value fields "snapshot_utc"
               | ("CLAIM" as kind) :: values
               | ("ENDPOINT" as kind) :: values
               | ("PRESENCE" as kind) :: values ->
                   let fields = snapshot_fields values in
                   let agent = table_value fields "agent" in
                   let lane = table_value fields "lane" in
                   if agent <> "" && lane <> "" then (
                     let entry = authority_entry lanes agent lane in
                     let prefer key current =
                       let value = table_value fields key in
                       if value <> "" then value else current
                     in
                     entry.authority_last_seen <-
                       prefer "last_seen" entry.authority_last_seen;
                     entry.authority_worktree <-
                       prefer "worktree" entry.authority_worktree;
                     if kind = "CLAIM" then
                       entry.authority_claim <- table_value fields "state"
                     else if kind = "ENDPOINT" then (
                       entry.authority_endpoint <- table_value fields "state";
                       entry.authority_transport <-
                         prefer "transport" entry.authority_transport;
                       entry.authority_harness <-
                         prefer "harness" entry.authority_harness)
                     else (
                       entry.authority_presence <- table_value fields "state";
                       entry.authority_presence_reason <-
                         prefer "reason" entry.authority_presence_reason;
                       entry.authority_harness <-
                         prefer "harness" entry.authority_harness;
                       entry.authority_session_id <-
                         prefer "session_id" entry.authority_session_id;
                       entry.authority_generation <-
                         prefer "generation" entry.authority_generation;
                       entry.authority_pid <- prefer "pid" entry.authority_pid;
                       if table_value fields "ready" = "1" then
                         entry.authority_ready_observed <- true))
               | _ -> ());
        (!valid, !snapshot_utc,
         !valid && command_spec.snapshot_command_authorized)

type authority_process_observation =
  | Authority_process_verified
  | Authority_process_absent
  | Authority_process_unknown

let authority_harness_matches lane arguments =
  let expected = String.lowercase_ascii (trim lane.authority_harness) in
  if expected = "" || expected = "unknown" then false
  else
    List.exists
      (fun argument ->
        let name =
          argument |> Filename.basename |> String.lowercase_ascii
        in
        name = expected || starts_with name (expected ^ "-")
        || starts_with name (expected ^ "_"))
      arguments

let authority_process_observation lane =
  let observe pid expected_start verify_arguments =
    if pid <= 1 then Authority_process_unknown
    else if not (Sys.file_exists (Printf.sprintf "/proc/%d/stat" pid)) then
      Authority_process_absent
    else
      try
        if expected_start <> "" then
          if process_start pid = expected_start then Authority_process_verified
          else Authority_process_unknown
        else if verify_arguments (process_arguments pid) then
          Authority_process_verified
        else Authority_process_unknown
      with _ -> Authority_process_unknown
  in
  let positive_pid value =
    try
      let pid = int_of_string value in
      if pid > 1 then Some pid else None
    with _ -> None
  in
  match positive_pid lane.authority_harness_pid with
  | Some pid ->
      observe pid lane.authority_harness_pid_start (authority_harness_matches lane)
  | None ->
      (match positive_pid lane.authority_pid with
      | Some pid -> observe pid "" (authority_harness_matches lane)
      | None -> Authority_process_unknown)

type authority_progress_sample = {
  progress_generation : string;
  progress_cursor : int;
  progress_time : float;
}

let authority_progress_samples = Hashtbl.create 64
let authority_progress_window_seconds = 5.0

let authority_generation lane =
  if lane.authority_loom_instance <> "" then lane.authority_loom_instance
  else lane.authority_session_id ^ ":" ^ lane.authority_generation

let observe_authority_progress lane =
  let key = authority_key lane.authority_agent lane.authority_lane in
  let generation = authority_generation lane in
  let now = Unix.gettimeofday () in
  match Hashtbl.find_opt authority_progress_samples key with
  | None ->
      Hashtbl.replace authority_progress_samples key
        { progress_generation = generation; progress_cursor = lane.authority_cursor;
          progress_time = now }
  | Some previous when previous.progress_generation <> generation ->
      Hashtbl.replace authority_progress_samples key
        { progress_generation = generation; progress_cursor = lane.authority_cursor;
          progress_time = now }
  | Some previous ->
      let elapsed = now -. previous.progress_time in
      lane.authority_progress_observed <-
        lane.authority_cursor > previous.progress_cursor;
      lane.authority_progress_window_complete <-
        elapsed >= authority_progress_window_seconds;
      if lane.authority_progress_window_complete then
        Hashtbl.replace authority_progress_samples key
          { progress_generation = generation;
            progress_cursor = lane.authority_cursor; progress_time = now }

let authority_obligation_enricher =
  ref (fun (_root : string) (_lanes : (string, authority_lane) Hashtbl.t) -> false)

let authority_observation lane =
  let process = authority_process_observation lane in
  let liveness_window_complete =
    lane.authority_presence = "unresponsive"
    || lane.authority_presence = "orphaned"
    || lane.authority_loom_state = "lost"
    || lane.authority_loom_state = "exited"
  in
  let process_verified = process = Authority_process_verified in
  { Loom_lane_health.policy_state =
      (if lane.authority_observation_authorized
          && lane.authority_sample_fresh then 1 else 0);
    expected_lane = true;
    claim_active = lane.authority_claim = "active";
    record_residue =
      lane.authority_loom_instance <> ""
      || lane.authority_presence = "orphaned";
    pane_or_harness_exists =
      process_verified || lane.authority_loom_state = "active";
    process_verified;
    process_unresponsive =
      process_verified && lane.authority_presence = "unresponsive";
    process_absent =
      lane.authority_presence = "orphaned"
      || (liveness_window_complete && process = Authority_process_absent);
    endpoint_verified = lane.authority_endpoint = "active";
    endpoint_absent = lane.authority_endpoint = "unavailable";
    endpoint_stale =
      lane.authority_endpoint = "stale"
      || lane.authority_endpoint = "drifted";
    custody_active = lane.authority_loom_state = "active";
    custody_recoverable = lane.authority_loom_state = "recoverable";
    obligation_active = lane.authority_active_obligations > 0;
    blocker_active = lane.authority_blocker_active;
    obligation_census_complete =
      lane.authority_obligation_census_complete;
    progress_observed = lane.authority_progress_observed;
    progress_window_complete = lane.authority_progress_window_complete;
    liveness_window_complete;
    ready_observed = lane.authority_ready_observed;
    observation_authority_verified =
      lane.authority_observation_authorized;
    sample_fresh = lane.authority_sample_fresh }

let truthful_state lane =
  Loom_lane_health.classify (authority_observation lane)

let legacy_operational_state lane =
  if lane.authority_loom_state = "active" then "active"
  else if lane.authority_presence = "live" then "live"
  else if lane.authority_presence = "unresponsive" then "unresponsive"
  else if lane.authority_presence = "orphaned" then "orphaned"
  else if lane.authority_claim = "active" then "claimed"
  else if lane.authority_loom_state <> "none" then lane.authority_loom_state
  else "offline"

let operational_state lane =
  truthful_state lane |> Loom_lane_health.name |> String.lowercase_ascii

let authority_rank lane =
  match truthful_state lane with
  | Loom_lane_health.Working -> 0
  | Loom_lane_health.Blocked -> 1
  | Loom_lane_health.Disconnected -> 2
  | Loom_lane_health.Unresponsive -> 3
  | Loom_lane_health.Idle -> 4
  | Loom_lane_health.Orphaned -> 5
  | Loom_lane_health.Dead -> 6
  | Loom_lane_health.Conflicted -> 7
  | Loom_lane_health.Unknown -> 8

let sorted_authority_values lanes =
  Hashtbl.fold (fun _ value found -> value :: found) lanes []
  |> List.sort (fun left right ->
         let rank = compare (authority_rank left) (authority_rank right) in
         if rank <> 0 then rank
         else
           compare
             (left.authority_agent, left.authority_lane)
             (right.authority_agent, right.authority_lane))

let count_authority_health values state =
  List.fold_left
    (fun total lane -> if truthful_state lane = state then total + 1 else total)
    0 values

let authority_lane_json lane =
  let health = truthful_state lane in
  let boolean value = if value then "true" else "false" in
  Printf.sprintf
    "{\"agent\":\"%s\",\"lane\":\"%s\",\"state\":\"%s\",\"health_state\":\"%s\",\"health_code\":%d,\"health_authority\":\"Sounio\",\"health_realization\":\"OCaml\",\"health_semantics_sha256\":\"%s\",\"legacy_state\":\"%s\",\"claim_state\":\"%s\",\"presence_state\":\"%s\",\"presence_reason\":\"%s\",\"endpoint_state\":\"%s\",\"transport\":\"%s\",\"harness\":\"%s\",\"session_id\":\"%s\",\"generation\":\"%s\",\"pid\":\"%s\",\"last_seen_utc\":\"%s\",\"worktree\":\"%s\",\"loom_state\":\"%s\",\"loom_instance\":\"%s\",\"guardian_pid\":\"%s\",\"harness_pid\":\"%s\",\"harness_pid_start\":\"%s\",\"started_utc\":\"%s\",\"command\":\"%s\",\"cursor\":%d,\"pending_obligations\":%d,\"active_obligations\":%d,\"blocker_active\":%s,\"obligation_census_complete\":%s,\"progress_observed\":%s,\"progress_window_complete\":%s,\"ready_observed\":%s,\"observation_authorized\":%s,\"sample_fresh\":%s}"
    (json_escape lane.authority_agent) (json_escape lane.authority_lane)
    (operational_state lane) (Loom_lane_health.name health)
    (Loom_lane_health.code health)
    Loom_lane_health.parent_semantics_sha256
    (json_escape (legacy_operational_state lane))
    (json_escape lane.authority_claim)
    (json_escape lane.authority_presence)
    (json_escape lane.authority_presence_reason)
    (json_escape lane.authority_endpoint)
    (json_escape lane.authority_transport) (json_escape lane.authority_harness)
    (json_escape lane.authority_session_id)
    (json_escape lane.authority_generation) (json_escape lane.authority_pid)
    (json_escape lane.authority_last_seen)
    (json_escape lane.authority_worktree)
    (json_escape lane.authority_loom_state)
    (json_escape lane.authority_loom_instance)
    (json_escape lane.authority_guardian_pid)
    (json_escape lane.authority_harness_pid)
    (json_escape lane.authority_harness_pid_start)
    (json_escape lane.authority_started_utc)
    (json_escape lane.authority_command) lane.authority_cursor
    lane.authority_pending_obligations lane.authority_active_obligations
    (boolean lane.authority_blocker_active)
    (boolean lane.authority_obligation_census_complete)
    (boolean lane.authority_progress_observed)
    (boolean lane.authority_progress_window_complete)
    (boolean lane.authority_ready_observed)
    (boolean lane.authority_observation_authorized)
    (boolean lane.authority_sample_fresh)

let load_authority_lanes root cwd =
  let lanes = Hashtbl.create 64 in
  let coordination_available, snapshot_utc, snapshot_authorized =
    load_authority_snapshot cwd lanes
  in
  session_descriptors root
  |> List.iter (fun (_, values) ->
         let agent = table_value values "agent" in
         let lane = table_value values "lane" in
         if agent <> "" && lane <> "" then (
           let entry = authority_entry lanes agent lane in
           let output = table_value values "output_file" in
           entry.authority_loom_state <- table_value values "state";
           entry.authority_loom_instance <- table_value values "instance_id";
           entry.authority_guardian_pid <- table_value values "guardian_pid";
           entry.authority_harness_pid <- table_value values "harness_pid";
           entry.authority_harness_pid_start <-
             table_value values "harness_pid_start";
           entry.authority_started_utc <- table_value values "started_utc";
           entry.authority_command <- table_value values "command";
           entry.authority_cursor <- file_size output;
           if entry.authority_session_id = "" then
             entry.authority_session_id <- table_value values "session_id";
           if entry.authority_worktree = "" then
             entry.authority_worktree <- table_value values "worktree"));
  ignore ((!authority_obligation_enricher) root lanes);
  Hashtbl.iter
    (fun _ entry ->
      entry.authority_observation_authorized <- snapshot_authorized;
      entry.authority_sample_fresh <- coordination_available;
      observe_authority_progress entry)
    lanes;
  (coordination_available, snapshot_utc, snapshot_authorized, lanes)

let fleet_json root cwd =
  let coordination_available, snapshot_utc, snapshot_authorized, lanes =
    load_authority_lanes root cwd
  in
  let values = sorted_authority_values lanes in
  let count predicate =
    List.fold_left (fun total value -> if predicate value then total + 1 else total) 0 values
  in
  let count_health = count_authority_health values in
  Printf.sprintf
    "{\"schema\":\"loom-authority-overlay-v2\",\"compatibility_schema\":\"loom-authority-overlay-v1\",\"snapshot_utc\":\"%s\",\"coordination_available\":%s,\"observation_authorized\":%s,\"health_authority\":\"Sounio\",\"health_realization\":\"OCaml\",\"health_semantics_sha256\":\"%s\",\"summary\":{\"lanes\":%d,\"live\":%d,\"raw_unresponsive\":%d,\"raw_orphaned\":%d,\"loom_custody\":%d,\"active_endpoints\":%d,\"working\":%d,\"idle\":%d,\"blocked\":%d,\"disconnected\":%d,\"unresponsive\":%d,\"orphaned\":%d,\"dead\":%d,\"conflicted\":%d,\"unknown\":%d},\"lanes\":[%s]}"
    (json_escape snapshot_utc)
    (if coordination_available then "true" else "false")
    (if snapshot_authorized then "true" else "false")
    Loom_lane_health.parent_semantics_sha256
    (List.length values)
    (count (fun lane -> lane.authority_presence = "live"))
    (count (fun lane -> lane.authority_presence = "unresponsive"))
    (count (fun lane -> lane.authority_presence = "orphaned"))
    (count (fun lane -> lane.authority_loom_state = "active" || lane.authority_loom_state = "recoverable"))
    (count (fun lane -> lane.authority_endpoint = "active"))
    (count_health Loom_lane_health.Working)
    (count_health Loom_lane_health.Idle)
    (count_health Loom_lane_health.Blocked)
    (count_health Loom_lane_health.Disconnected)
    (count_health Loom_lane_health.Unresponsive)
    (count_health Loom_lane_health.Orphaned)
    (count_health Loom_lane_health.Dead)
    (count_health Loom_lane_health.Conflicted)
    (count_health Loom_lane_health.Unknown)
    (values |> List.map authority_lane_json |> String.concat ",")

let legacy_html =
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

let html = if Loom_ui.html = "" then legacy_html else Loom_ui.html

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
  independent_measurement_verified : bool;
  observer_key_id : string;
  observer_principal_id : string;
  independent_observation_digest : string;
  observation_authority_verified : bool;
  full_digest_agreement_verified : bool;
  journal_authority_principal_id : string;
  journal_authority_principal_ids : string list;
  journal_authority_required_quorum : int;
  journal_authority_min_valid_signatures : int;
  journal_authority_quorum_verified : bool;
  journal_authority_epoch : int;
  journal_authority_checkpoint_digest : string;
}

let beagle_generation_journals paths instance =
  let generation_dir =
    Filename.concat (Filename.concat paths.session_dir "generations") instance
  in
  ( Filename.concat generation_dir "journal.tsv",
    Filename.concat generation_dir "guardian.tsv" )

let beagle_generation_descriptor paths instance =
  Filename.concat
    (Filename.concat (Filename.concat paths.session_dir "generations") instance)
    "session.state"

let beagle_generation_fingerprint pane_id descriptor =
  sha256
    (String.concat "\000"
       [ pane_id; table_value descriptor "session_id";
         table_value descriptor "instance_id";
         table_value descriptor "harness_pid_start";
         table_value descriptor "argv_digest" ])

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
        { channel; descriptor; seq = List.length records; previous;
          authority_context = None;
          authority_directory = Filename.dirname path }
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

type continuity_fact_digests = {
  generation_digest : string;
  fingerprint_digest : string;
  semantic_head_digest : string;
  guardian_head_digest : string;
}

type continuity_decision_facts = {
  decision_generation : string;
  decision_generation_fingerprint : string;
  decision_semantic_head : string;
  decision_guardian_head : string;
}

let continuity_fact_digest domain value =
  sha256
    (String.concat "\000"
       [ "loom-continuity-fact-digest-v1"; domain; value ])

let continuity_fact_digests generation fingerprint semantic_head guardian_head =
  { generation_digest = continuity_fact_digest "generation" generation;
    fingerprint_digest =
      continuity_fact_digest "generation-fingerprint" fingerprint;
    semantic_head_digest = continuity_fact_digest "semantic-head" semantic_head;
    guardian_head_digest = continuity_fact_digest "guardian-head" guardian_head }

let continuity_decision_fact_digests facts =
  continuity_fact_digests facts.decision_generation
    facts.decision_generation_fingerprint facts.decision_semantic_head
    facts.decision_guardian_head

let digest256_limbs digest =
  if not (valid_sha256 digest) then failf "sounio-continuity-invalid-full-digest";
  List.init 8 (fun index ->
      Int64.to_string
        (Int64.of_string ("0x" ^ String.sub digest (index * 8) 8)))

let fact_digest_limbs facts =
  digest256_limbs facts.generation_digest
  @ digest256_limbs facts.fingerprint_digest
  @ digest256_limbs facts.semantic_head_digest
  @ digest256_limbs facts.guardian_head_digest

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
  signed_fact_digests : continuity_fact_digests option;
  signed_decision_facts : continuity_decision_facts option;
}

type verified_independent_observation = {
  observer_key_id : string;
  observer_principal_id : string;
  subject_signer_key_id : string;
  subject_principal_id : string;
  subject_receipt_digest : string;
  observation_digest : string;
}

type independently_measured_generation = {
  measured_generation : string;
  measured_generation_fingerprint : string;
  measured_semantic_head : string;
  measured_guardian_head : string;
  measured_semantic_journal_digest : string;
  measured_guardian_journal_digest : string;
  measured_descriptor_digest : string;
  measured_fact_digests : continuity_fact_digests;
  measured_journal_authority_principal_id : string;
  measured_journal_authority_principal_ids : string list;
  measured_journal_authority_required_quorum : int;
  measured_journal_authority_min_valid_signatures : int;
  measured_journal_authority_epoch : int;
  measured_journal_authority_checkpoint_digest : string;
}

type verified_independent_measurement = {
  measured_observation : verified_independent_observation;
  measured_generation_facts : independently_measured_generation;
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

let independent_measurement_required () =
  observation_authority_required ()
  || environment_flag "SOUNIO_LOOM_REQUIRE_INDEPENDENT_MEASUREMENT"

let independent_observer_explicitly_required () =
  match Sys.getenv_opt "SOUNIO_LOOM_REQUIRE_INDEPENDENT_OBSERVER" with
  | None | Some "" | Some "0" | Some "false" -> false
  | Some "1" | Some "true" -> true
  | Some value ->
      failf "sounio-continuity-invalid-independent-observer-requirement:%s" value

let independent_observer_required () =
  independent_observer_explicitly_required () || independent_measurement_required ()

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

let signed_continuity_payload_v3 key_id runtime_digest facts_digest facts
    decision_facts fact_digests verdict =
  Printf.sprintf
    "schema=loom-native-continuity-signed-payload-v2\nalgorithm=ed25519\nkey_id=%s\nadapter_sha256=%s\nfacts_sha256=%s\nfacts=%s\nfact_digest_schema=loom-continuity-fact-digest-v1\ndecision_generation=%s\ndecision_generation_fingerprint=%s\ndecision_semantic_head=%s\ndecision_guardian_head=%s\ndecision_generation_sha256=%s\ndecision_generation_fingerprint_sha256=%s\ndecision_semantic_head_sha256=%s\ndecision_guardian_head_sha256=%s\nverdict=%s\n"
    key_id runtime_digest facts_digest facts decision_facts.decision_generation
    decision_facts.decision_generation_fingerprint
    decision_facts.decision_semantic_head decision_facts.decision_guardian_head
    fact_digests.generation_digest
    fact_digests.fingerprint_digest fact_digests.semantic_head_digest
    fact_digests.guardian_head_digest verdict

let signed_continuity_receipt_v3 key_id runtime_digest facts_digest facts
    decision_facts fact_digests verdict payload_digest signature =
  Printf.sprintf
    "schema=loom-native-continuity-receipt-v3\nalgorithm=ed25519\nkey_id=%s\nadapter_sha256=%s\nfacts_sha256=%s\nfacts=%s\nfact_digest_schema=loom-continuity-fact-digest-v1\ndecision_generation=%s\ndecision_generation_fingerprint=%s\ndecision_semantic_head=%s\ndecision_guardian_head=%s\ndecision_generation_sha256=%s\ndecision_generation_fingerprint_sha256=%s\ndecision_semantic_head_sha256=%s\ndecision_guardian_head_sha256=%s\nverdict=%s\nsigned_payload_sha256=%s\nsignature_base64=%s\n"
    key_id runtime_digest facts_digest facts decision_facts.decision_generation
    decision_facts.decision_generation_fingerprint
    decision_facts.decision_semantic_head decision_facts.decision_guardian_head
    fact_digests.generation_digest fact_digests.fingerprint_digest
    fact_digests.semantic_head_digest
    fact_digests.guardian_head_digest verdict payload_digest signature

let signed_continuity_expected_verdict facts =
  let values = split_on ' ' facts in
  if List.length values = 15 && List.nth values 14 = "1" then
    Some "SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v2 authenticity=ed25519"
  else if List.length values = 18 && List.nth values 14 = "2" then
    Some "SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v3 authenticity=ed25519+independent-observer"
  else None

type journal_authority_stream_identity = {
  stream_principal_ids : string list;
  stream_epoch : int;
  stream_required_quorum : int;
  stream_min_valid_signatures : int;
}

let journal_authority_proof_identity = function
  | Single_authority_stamp stamp ->
      { stream_principal_ids = [ stamp.principal_id ]; stream_epoch = stamp.epoch;
        stream_required_quorum = 1; stream_min_valid_signatures = 1 }
  | Quorum_authority_certificate certificate ->
      let principals =
        List.map (fun member -> member.quorum_principal_id)
          certificate.quorum_members
      in
      let valid =
        List.fold_left
          (fun count member ->
            match member.quorum_signature with Some _ -> count + 1 | None -> count)
          0 certificate.quorum_members
      in
      { stream_principal_ids = principals;
        stream_epoch = certificate.quorum_epoch;
        stream_required_quorum = certificate.quorum_required;
        stream_min_valid_signatures = valid }

let journal_event_authority_identity events =
  let saw_unsigned, identity =
    List.fold_left
      (fun (saw_unsigned, identity) (event : journal_event) ->
        match (identity, event.authority) with
        | None, None -> (true, None)
        | None, Some proof ->
            if saw_unsigned then
              failf "sounio-journal-authority-partially-signed-journal";
            (false, Some (journal_authority_proof_identity proof))
        | Some _, None ->
            failf "sounio-journal-authority-partially-signed-journal"
        | Some current, Some proof ->
            let next = journal_authority_proof_identity proof in
            if next.stream_principal_ids <> current.stream_principal_ids
               || next.stream_epoch <> current.stream_epoch
               || next.stream_required_quorum <> current.stream_required_quorum
            then
              failf "sounio-journal-authority-mixed-identity";
            ( false,
              Some
                { current with
                  stream_min_valid_signatures =
                    min current.stream_min_valid_signatures
                      next.stream_min_valid_signatures } ))
      (false, None) events
  in
  ignore saw_unsigned;
  identity

type measured_journal_authority = {
  measured_authority_principal_id : string;
  measured_authority_principal_ids : string list;
  measured_authority_epoch : int;
  measured_authority_required_quorum : int;
  measured_authority_min_valid_signatures : int;
  measured_authority_checkpoint : string;
}

let measured_journal_authority semantic_events guardian_events
    semantic_journal_digest guardian_journal_digest descriptor_digest =
  match
    ( journal_event_authority_identity semantic_events,
      journal_event_authority_identity guardian_events )
  with
  | None, None when not (observation_authority_required ()) ->
      { measured_authority_principal_id = "";
        measured_authority_principal_ids = [];
        measured_authority_epoch = 0; measured_authority_required_quorum = 0;
        measured_authority_min_valid_signatures = 0;
        measured_authority_checkpoint = "" }
  | Some semantic, Some guardian ->
      if semantic.stream_principal_ids <> guardian.stream_principal_ids
         || semantic.stream_epoch <> guardian.stream_epoch
         || semantic.stream_required_quorum <> guardian.stream_required_quorum
      then failf "sounio-journal-authority-stream-identity-mismatch";
      let principal_id =
        match semantic.stream_principal_ids with
        | [ principal ] when semantic.stream_required_quorum = 1 -> principal
        | principals ->
            sha256
              (String.concat "\000"
                 ("loom-journal-authority-principal-set-v1"
                 :: string_of_int semantic.stream_required_quorum :: principals))
      in
      let checkpoint =
        let material =
          if semantic.stream_required_quorum = 1 then
            [ "loom-journal-authority-checkpoint-v1"; principal_id;
              string_of_int semantic.stream_epoch; semantic_journal_digest;
              guardian_journal_digest; descriptor_digest ]
          else
            [ "loom-journal-authority-quorum-checkpoint-v1"; principal_id;
              string_of_int semantic.stream_epoch;
              string_of_int semantic.stream_required_quorum;
              semantic_journal_digest; guardian_journal_digest;
              descriptor_digest ]
        in
        sha256 (String.concat "\000" material)
      in
      { measured_authority_principal_id = principal_id;
        measured_authority_principal_ids = semantic.stream_principal_ids;
        measured_authority_epoch = semantic.stream_epoch;
        measured_authority_required_quorum = semantic.stream_required_quorum;
        measured_authority_min_valid_signatures =
          min semantic.stream_min_valid_signatures
            guardian.stream_min_valid_signatures;
        measured_authority_checkpoint = checkpoint }
  | _ -> failf "sounio-journal-authority-required-stream-missing"

let independently_measure_generation ?decision_facts paths pane_id generation =
  if generation = "" then failf "sounio-continuity-measurement-empty-generation";
  let descriptor_path = beagle_generation_descriptor paths generation in
  if not (Sys.file_exists descriptor_path) then
    failf "sounio-continuity-measurement-descriptor-missing";
  let descriptor_text = read_file descriptor_path in
  let descriptor = parse_key_values descriptor_path in
  if table_value descriptor "instance_id" <> generation then
    failf "sounio-continuity-measurement-generation-mismatch";
  if table_value descriptor "session_id" = "" then
    failf "sounio-continuity-measurement-session-missing";
  let semantic_path, guardian_path =
    beagle_generation_journals paths generation
  in
  let semantic_text = read_file semantic_path in
  let guardian_text = read_file guardian_path in
  let semantic_events, _, semantic_head =
    load_and_verify_journal semantic_path
  in
  let guardian_events, _, _, guardian_head =
    load_and_verify_guardian_journal guardian_path
  in
  let semantic_journal_digest = sha256 semantic_text in
  let guardian_journal_digest = sha256 guardian_text in
  let descriptor_digest = sha256 descriptor_text in
  let fingerprint = beagle_generation_fingerprint pane_id descriptor in
  let measured_generation, measured_fingerprint, measured_semantic_head,
      measured_guardian_head =
    match decision_facts with
    | None -> (generation, fingerprint, semantic_head, guardian_head)
    | Some decision ->
        if decision.decision_generation <> generation
           || decision.decision_generation_fingerprint <> fingerprint
        then failf "sounio-continuity-measurement-decision-descriptor-mismatch";
        if not
             (List.exists
                (fun (event : journal_event) ->
                  event.hash = decision.decision_semantic_head)
                semantic_events)
        then failf "sounio-continuity-measurement-semantic-checkpoint-missing";
        if not
             (List.exists
                (fun (event : journal_event) ->
                  event.hash = decision.decision_guardian_head)
                guardian_events)
        then failf "sounio-continuity-measurement-guardian-checkpoint-missing";
        (decision.decision_generation,
         decision.decision_generation_fingerprint,
         decision.decision_semantic_head, decision.decision_guardian_head)
  in
  let fact_digests =
    continuity_fact_digests measured_generation measured_fingerprint
      measured_semantic_head measured_guardian_head
  in
  let journal_authority =
    measured_journal_authority semantic_events guardian_events
      semantic_journal_digest guardian_journal_digest descriptor_digest
  in
  { measured_generation;
    measured_generation_fingerprint = measured_fingerprint;
    measured_semantic_head;
    measured_guardian_head;
    measured_semantic_journal_digest = semantic_journal_digest;
    measured_guardian_journal_digest = guardian_journal_digest;
    measured_descriptor_digest = descriptor_digest;
    measured_fact_digests = fact_digests;
    measured_journal_authority_principal_id =
      journal_authority.measured_authority_principal_id;
    measured_journal_authority_principal_ids =
      journal_authority.measured_authority_principal_ids;
    measured_journal_authority_required_quorum =
      journal_authority.measured_authority_required_quorum;
    measured_journal_authority_min_valid_signatures =
      journal_authority.measured_authority_min_valid_signatures;
    measured_journal_authority_epoch =
      journal_authority.measured_authority_epoch;
    measured_journal_authority_checkpoint_digest =
      journal_authority.measured_authority_checkpoint }

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

let independent_measurement_payload observer_key_id observer_principal_id
    subject_signer_key_id subject_principal_id subject_receipt_digest
    subject_facts_digest subject_adapter_digest measurement =
  if measurement.measured_journal_authority_principal_id = "" then
    Printf.sprintf
      "schema=loom-independent-measurement-payload-v1\nalgorithm=ed25519\nobserver_key_id=%s\nobserver_principal_id=%s\nsubject_signer_key_id=%s\nsubject_principal_id=%s\nsubject_receipt_sha256=%s\nsubject_facts_sha256=%s\nsubject_adapter_sha256=%s\nmeasurement_source=verified-generation-artifacts-v1\nmeasured_generation=%s\nmeasured_generation_fingerprint=%s\nmeasured_semantic_head=%s\nmeasured_guardian_head=%s\nsemantic_journal_sha256=%s\nguardian_journal_sha256=%s\ndescriptor_sha256=%s\nobservation=independent-generation-measurement\n"
      observer_key_id observer_principal_id subject_signer_key_id
      subject_principal_id subject_receipt_digest subject_facts_digest
      subject_adapter_digest measurement.measured_generation
      measurement.measured_generation_fingerprint measurement.measured_semantic_head
      measurement.measured_guardian_head
      measurement.measured_semantic_journal_digest
      measurement.measured_guardian_journal_digest
      measurement.measured_descriptor_digest
  else if measurement.measured_journal_authority_required_quorum = 2 then
    let digests = measurement.measured_fact_digests in
    let principal_a, principal_b, principal_c =
      match measurement.measured_journal_authority_principal_ids with
      | [ a; b; c ] -> (a, b, c)
      | _ -> failf "sounio-journal-authority-quorum-member-count-mismatch"
    in
    Printf.sprintf
      "schema=loom-independent-measurement-payload-v3\nalgorithm=ed25519\nobserver_key_id=%s\nobserver_principal_id=%s\nsubject_signer_key_id=%s\nsubject_principal_id=%s\nsubject_receipt_sha256=%s\nsubject_facts_sha256=%s\nsubject_adapter_sha256=%s\nmeasurement_source=quorum-write-authorized-generation-artifacts-v3\nmeasured_generation=%s\nmeasured_generation_fingerprint=%s\nmeasured_semantic_head=%s\nmeasured_guardian_head=%s\nmeasured_generation_sha256=%s\nmeasured_generation_fingerprint_sha256=%s\nmeasured_semantic_head_sha256=%s\nmeasured_guardian_head_sha256=%s\nsemantic_journal_sha256=%s\nguardian_journal_sha256=%s\ndescriptor_sha256=%s\njournal_authority_principal_id=%s\njournal_authority_principal_id_1=%s\njournal_authority_principal_id_2=%s\njournal_authority_principal_id_3=%s\njournal_authority_required_quorum=%d\njournal_authority_min_valid_signatures=%d\njournal_authority_epoch=%d\njournal_authority_checkpoint_sha256=%s\nobservation=quorum-write-authorized-generation-measurement\n"
      observer_key_id observer_principal_id subject_signer_key_id
      subject_principal_id subject_receipt_digest subject_facts_digest
      subject_adapter_digest measurement.measured_generation
      measurement.measured_generation_fingerprint measurement.measured_semantic_head
      measurement.measured_guardian_head digests.generation_digest
      digests.fingerprint_digest digests.semantic_head_digest
      digests.guardian_head_digest measurement.measured_semantic_journal_digest
      measurement.measured_guardian_journal_digest
      measurement.measured_descriptor_digest
      measurement.measured_journal_authority_principal_id principal_a principal_b
      principal_c measurement.measured_journal_authority_required_quorum
      measurement.measured_journal_authority_min_valid_signatures
      measurement.measured_journal_authority_epoch
      measurement.measured_journal_authority_checkpoint_digest
  else
    let digests = measurement.measured_fact_digests in
    Printf.sprintf
      "schema=loom-independent-measurement-payload-v2\nalgorithm=ed25519\nobserver_key_id=%s\nobserver_principal_id=%s\nsubject_signer_key_id=%s\nsubject_principal_id=%s\nsubject_receipt_sha256=%s\nsubject_facts_sha256=%s\nsubject_adapter_sha256=%s\nmeasurement_source=write-authorized-generation-artifacts-v2\nmeasured_generation=%s\nmeasured_generation_fingerprint=%s\nmeasured_semantic_head=%s\nmeasured_guardian_head=%s\nmeasured_generation_sha256=%s\nmeasured_generation_fingerprint_sha256=%s\nmeasured_semantic_head_sha256=%s\nmeasured_guardian_head_sha256=%s\nsemantic_journal_sha256=%s\nguardian_journal_sha256=%s\ndescriptor_sha256=%s\njournal_authority_principal_id=%s\njournal_authority_epoch=%d\njournal_authority_checkpoint_sha256=%s\nobservation=write-authorized-generation-measurement\n"
      observer_key_id observer_principal_id subject_signer_key_id
      subject_principal_id subject_receipt_digest subject_facts_digest
      subject_adapter_digest measurement.measured_generation
      measurement.measured_generation_fingerprint measurement.measured_semantic_head
      measurement.measured_guardian_head digests.generation_digest
      digests.fingerprint_digest digests.semantic_head_digest
      digests.guardian_head_digest measurement.measured_semantic_journal_digest
      measurement.measured_guardian_journal_digest
      measurement.measured_descriptor_digest
      measurement.measured_journal_authority_principal_id
      measurement.measured_journal_authority_epoch
      measurement.measured_journal_authority_checkpoint_digest

let independent_measurement_receipt observer_key_id observer_principal_id
    subject_signer_key_id subject_principal_id subject_receipt_digest
    subject_facts_digest subject_adapter_digest measurement payload_digest signature =
  if measurement.measured_journal_authority_principal_id = "" then
    Printf.sprintf
      "schema=loom-independent-measurement-attestation-v1\nalgorithm=ed25519\nobserver_key_id=%s\nobserver_principal_id=%s\nsubject_signer_key_id=%s\nsubject_principal_id=%s\nsubject_receipt_sha256=%s\nsubject_facts_sha256=%s\nsubject_adapter_sha256=%s\nmeasurement_source=verified-generation-artifacts-v1\nmeasured_generation=%s\nmeasured_generation_fingerprint=%s\nmeasured_semantic_head=%s\nmeasured_guardian_head=%s\nsemantic_journal_sha256=%s\nguardian_journal_sha256=%s\ndescriptor_sha256=%s\nobservation=independent-generation-measurement\nsigned_payload_sha256=%s\nsignature_base64=%s\n"
      observer_key_id observer_principal_id subject_signer_key_id
      subject_principal_id subject_receipt_digest subject_facts_digest
      subject_adapter_digest measurement.measured_generation
      measurement.measured_generation_fingerprint measurement.measured_semantic_head
      measurement.measured_guardian_head
      measurement.measured_semantic_journal_digest
      measurement.measured_guardian_journal_digest
      measurement.measured_descriptor_digest payload_digest signature
  else if measurement.measured_journal_authority_required_quorum = 2 then
    let digests = measurement.measured_fact_digests in
    let principal_a, principal_b, principal_c =
      match measurement.measured_journal_authority_principal_ids with
      | [ a; b; c ] -> (a, b, c)
      | _ -> failf "sounio-journal-authority-quorum-member-count-mismatch"
    in
    Printf.sprintf
      "schema=loom-independent-measurement-attestation-v3\nalgorithm=ed25519\nobserver_key_id=%s\nobserver_principal_id=%s\nsubject_signer_key_id=%s\nsubject_principal_id=%s\nsubject_receipt_sha256=%s\nsubject_facts_sha256=%s\nsubject_adapter_sha256=%s\nmeasurement_source=quorum-write-authorized-generation-artifacts-v3\nmeasured_generation=%s\nmeasured_generation_fingerprint=%s\nmeasured_semantic_head=%s\nmeasured_guardian_head=%s\nmeasured_generation_sha256=%s\nmeasured_generation_fingerprint_sha256=%s\nmeasured_semantic_head_sha256=%s\nmeasured_guardian_head_sha256=%s\nsemantic_journal_sha256=%s\nguardian_journal_sha256=%s\ndescriptor_sha256=%s\njournal_authority_principal_id=%s\njournal_authority_principal_id_1=%s\njournal_authority_principal_id_2=%s\njournal_authority_principal_id_3=%s\njournal_authority_required_quorum=%d\njournal_authority_min_valid_signatures=%d\njournal_authority_epoch=%d\njournal_authority_checkpoint_sha256=%s\nobservation=quorum-write-authorized-generation-measurement\nsigned_payload_sha256=%s\nsignature_base64=%s\n"
      observer_key_id observer_principal_id subject_signer_key_id
      subject_principal_id subject_receipt_digest subject_facts_digest
      subject_adapter_digest measurement.measured_generation
      measurement.measured_generation_fingerprint measurement.measured_semantic_head
      measurement.measured_guardian_head digests.generation_digest
      digests.fingerprint_digest digests.semantic_head_digest
      digests.guardian_head_digest measurement.measured_semantic_journal_digest
      measurement.measured_guardian_journal_digest
      measurement.measured_descriptor_digest
      measurement.measured_journal_authority_principal_id principal_a principal_b
      principal_c measurement.measured_journal_authority_required_quorum
      measurement.measured_journal_authority_min_valid_signatures
      measurement.measured_journal_authority_epoch
      measurement.measured_journal_authority_checkpoint_digest payload_digest signature
  else
    let digests = measurement.measured_fact_digests in
    Printf.sprintf
      "schema=loom-independent-measurement-attestation-v2\nalgorithm=ed25519\nobserver_key_id=%s\nobserver_principal_id=%s\nsubject_signer_key_id=%s\nsubject_principal_id=%s\nsubject_receipt_sha256=%s\nsubject_facts_sha256=%s\nsubject_adapter_sha256=%s\nmeasurement_source=write-authorized-generation-artifacts-v2\nmeasured_generation=%s\nmeasured_generation_fingerprint=%s\nmeasured_semantic_head=%s\nmeasured_guardian_head=%s\nmeasured_generation_sha256=%s\nmeasured_generation_fingerprint_sha256=%s\nmeasured_semantic_head_sha256=%s\nmeasured_guardian_head_sha256=%s\nsemantic_journal_sha256=%s\nguardian_journal_sha256=%s\ndescriptor_sha256=%s\njournal_authority_principal_id=%s\njournal_authority_epoch=%d\njournal_authority_checkpoint_sha256=%s\nobservation=write-authorized-generation-measurement\nsigned_payload_sha256=%s\nsignature_base64=%s\n"
      observer_key_id observer_principal_id subject_signer_key_id
      subject_principal_id subject_receipt_digest subject_facts_digest
      subject_adapter_digest measurement.measured_generation
      measurement.measured_generation_fingerprint measurement.measured_semantic_head
      measurement.measured_guardian_head digests.generation_digest
      digests.fingerprint_digest digests.semantic_head_digest
      digests.guardian_head_digest measurement.measured_semantic_journal_digest
      measurement.measured_guardian_journal_digest
      measurement.measured_descriptor_digest
      measurement.measured_journal_authority_principal_id
      measurement.measured_journal_authority_epoch
      measurement.measured_journal_authority_checkpoint_digest payload_digest
      signature

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
  let fact_digests =
    { generation_digest = table_value fields "decision_generation_sha256";
      fingerprint_digest =
        table_value fields "decision_generation_fingerprint_sha256";
      semantic_head_digest =
        table_value fields "decision_semantic_head_sha256";
      guardian_head_digest =
        table_value fields "decision_guardian_head_sha256" }
  in
  let decision_facts =
    { decision_generation = table_value fields "decision_generation";
      decision_generation_fingerprint =
        table_value fields "decision_generation_fingerprint";
      decision_semantic_head = table_value fields "decision_semantic_head";
      decision_guardian_head = table_value fields "decision_guardian_head" }
  in
  let payload, canonical, verified_fact_digests, verified_decision_facts =
    if schema = "loom-native-continuity-receipt-v2" then
      ( signed_continuity_payload key_id stored_adapter facts_digest facts verdict,
        signed_continuity_receipt key_id stored_adapter facts_digest facts verdict
          payload_digest signature,
        None, None )
    else if schema = "loom-native-continuity-receipt-v3" then (
      let decision_digests = continuity_decision_fact_digests decision_facts in
      let fact_values = split_on ' ' facts in
      if table_value fields "fact_digest_schema"
           <> "loom-continuity-fact-digest-v1"
         || decision_facts.decision_generation = ""
         || decision_facts.decision_generation_fingerprint = ""
         || not (valid_sha256 decision_facts.decision_semantic_head)
         || not (valid_sha256 decision_facts.decision_guardian_head)
         || not (valid_sha256 fact_digests.generation_digest)
         || not (valid_sha256 fact_digests.fingerprint_digest)
         || not (valid_sha256 fact_digests.semantic_head_digest)
         || not (valid_sha256 fact_digests.guardian_head_digest)
         || fact_digests.generation_digest <> decision_digests.generation_digest
         || fact_digests.fingerprint_digest <> decision_digests.fingerprint_digest
         || fact_digests.semantic_head_digest
            <> decision_digests.semantic_head_digest
         || fact_digests.guardian_head_digest <> decision_digests.guardian_head_digest
         || (List.length fact_values <> 15 && List.length fact_values <> 18)
         || List.nth fact_values 2
            <> sounio_continuity_token "generation"
                 decision_facts.decision_generation
         || List.nth fact_values 3
            <> sounio_continuity_token "generation-fingerprint"
                 decision_facts.decision_generation_fingerprint
         || List.nth fact_values 4
            <> sounio_continuity_token "semantic-head"
                 decision_facts.decision_semantic_head
         || List.nth fact_values 5
            <> sounio_continuity_token "guardian-head"
                 decision_facts.decision_guardian_head
      then failf "sounio-continuity-signed-full-digest-mismatch";
      ( signed_continuity_payload_v3 key_id stored_adapter facts_digest facts
          decision_facts fact_digests verdict,
        signed_continuity_receipt_v3 key_id stored_adapter facts_digest facts
          decision_facts fact_digests verdict payload_digest signature,
        Some fact_digests, Some decision_facts ))
    else failf "sounio-continuity-signed-receipt-schema"
  in
  let expected_verdict = signed_continuity_expected_verdict facts in
  if algorithm <> "ed25519" || key_id <> expected_key_id
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
    signed_adapter_digest = stored_adapter;
    signed_fact_digests = verified_fact_digests;
    signed_decision_facts = verified_decision_facts }

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

let verify_independent_measurement_attestation ~subject ~subject_public_key
    ~observer_public_key ~paths ~pane_id ~predecessor path =
  if not (Sys.file_exists path) then
    failf "sounio-continuity-independent-measurement-missing:%s" path;
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
  let measurement_source = table_value fields "measurement_source" in
  let measured_generation = table_value fields "measured_generation" in
  let measured_generation_fingerprint =
    table_value fields "measured_generation_fingerprint"
  in
  let measured_semantic_head = table_value fields "measured_semantic_head" in
  let measured_guardian_head = table_value fields "measured_guardian_head" in
  let quorum_attestation =
    schema = "loom-independent-measurement-attestation-v3"
  in
  let authority_attestation =
    schema = "loom-independent-measurement-attestation-v2"
    || quorum_attestation
  in
  let measured_fact_digests =
    if authority_attestation then
      { generation_digest = table_value fields "measured_generation_sha256";
        fingerprint_digest =
          table_value fields "measured_generation_fingerprint_sha256";
        semantic_head_digest =
          table_value fields "measured_semantic_head_sha256";
        guardian_head_digest =
          table_value fields "measured_guardian_head_sha256" }
    else
      continuity_fact_digests measured_generation
        measured_generation_fingerprint measured_semantic_head
        measured_guardian_head
  in
  let measurement =
    { measured_generation;
      measured_generation_fingerprint;
      measured_semantic_head;
      measured_guardian_head;
      measured_semantic_journal_digest =
        table_value fields "semantic_journal_sha256";
      measured_guardian_journal_digest =
        table_value fields "guardian_journal_sha256";
      measured_descriptor_digest = table_value fields "descriptor_sha256";
      measured_fact_digests;
      measured_journal_authority_principal_id =
        table_value fields "journal_authority_principal_id";
      measured_journal_authority_principal_ids =
        if quorum_attestation then
          [ table_value fields "journal_authority_principal_id_1";
            table_value fields "journal_authority_principal_id_2";
            table_value fields "journal_authority_principal_id_3" ]
        else if authority_attestation then
          [ table_value fields "journal_authority_principal_id" ]
        else [];
      measured_journal_authority_required_quorum =
        if quorum_attestation then
          positive_epoch "attestation-quorum"
            (table_value fields "journal_authority_required_quorum")
        else if authority_attestation then 1 else 0;
      measured_journal_authority_min_valid_signatures =
        if quorum_attestation then
          positive_epoch "attestation-valid-signatures"
            (table_value fields "journal_authority_min_valid_signatures")
        else if authority_attestation then 1 else 0;
      measured_journal_authority_epoch =
        if authority_attestation then
          positive_epoch "attestation-epoch"
            (table_value fields "journal_authority_epoch")
        else 0;
      measured_journal_authority_checkpoint_digest =
        table_value fields "journal_authority_checkpoint_sha256" }
  in
  let observation = table_value fields "observation" in
  let payload_digest = table_value fields "signed_payload_sha256" in
  let signature = table_value fields "signature_base64" in
  let expected_observer_key_id = sha256 (read_file observer_public_key) in
  let expected_observer_principal_id = ed25519_principal_id observer_public_key in
  let expected_subject_principal_id = ed25519_principal_id subject_public_key in
  let decision_checkpoint =
    if authority_attestation then
      match subject.signed_decision_facts with
      | Some decision -> Some decision
      | None ->
          failf "sounio-continuity-observation-authority-requires-receipt-v3"
    else None
  in
  let expected_measurement =
    independently_measure_generation ?decision_facts:decision_checkpoint paths
      pane_id predecessor
  in
  let payload =
    independent_measurement_payload observer_key_id observer_principal_id
      subject_signer_key_id subject_principal_id subject_receipt_digest
      subject_facts_digest subject_adapter_digest measurement
  in
  let canonical =
    independent_measurement_receipt observer_key_id observer_principal_id
      subject_signer_key_id subject_principal_id subject_receipt_digest
      subject_facts_digest subject_adapter_digest measurement payload_digest
      signature
  in
  let schema_valid =
    (schema = "loom-independent-measurement-attestation-v1"
     && measurement_source = "verified-generation-artifacts-v1"
     && observation = "independent-generation-measurement")
    || (schema = "loom-independent-measurement-attestation-v2"
        && measurement_source = "write-authorized-generation-artifacts-v2"
        && observation = "write-authorized-generation-measurement")
    || (quorum_attestation
        && measurement_source = "quorum-write-authorized-generation-artifacts-v3"
        && observation = "quorum-write-authorized-generation-measurement")
  in
  if not schema_valid || algorithm <> "ed25519"
     || observer_key_id <> expected_observer_key_id
     || observer_principal_id <> expected_observer_principal_id
     || subject_signer_key_id <> subject.signed_key_id
     || subject_principal_id <> subject.signed_principal_id
     || subject_principal_id <> expected_subject_principal_id
     || subject_receipt_digest <> subject.signed_receipt_digest
     || subject_facts_digest <> subject.signed_facts_digest
     || subject_adapter_digest <> subject.signed_adapter_digest
     || measurement.measured_generation
        <> expected_measurement.measured_generation
     || measurement.measured_generation_fingerprint
        <> expected_measurement.measured_generation_fingerprint
     || measurement.measured_semantic_head
        <> expected_measurement.measured_semantic_head
     || measurement.measured_guardian_head
        <> expected_measurement.measured_guardian_head
     || measurement.measured_semantic_journal_digest
        <> expected_measurement.measured_semantic_journal_digest
     || measurement.measured_guardian_journal_digest
        <> expected_measurement.measured_guardian_journal_digest
     || measurement.measured_descriptor_digest
        <> expected_measurement.measured_descriptor_digest
     || measurement.measured_fact_digests.generation_digest
        <> expected_measurement.measured_fact_digests.generation_digest
     || measurement.measured_fact_digests.fingerprint_digest
        <> expected_measurement.measured_fact_digests.fingerprint_digest
     || measurement.measured_fact_digests.semantic_head_digest
        <> expected_measurement.measured_fact_digests.semantic_head_digest
     || measurement.measured_fact_digests.guardian_head_digest
        <> expected_measurement.measured_fact_digests.guardian_head_digest
     || measurement.measured_journal_authority_principal_id
        <> expected_measurement.measured_journal_authority_principal_id
     || measurement.measured_journal_authority_principal_ids
        <> expected_measurement.measured_journal_authority_principal_ids
     || measurement.measured_journal_authority_required_quorum
        <> expected_measurement.measured_journal_authority_required_quorum
     || measurement.measured_journal_authority_min_valid_signatures
        <> expected_measurement.measured_journal_authority_min_valid_signatures
     || measurement.measured_journal_authority_epoch
        <> expected_measurement.measured_journal_authority_epoch
     || measurement.measured_journal_authority_checkpoint_digest
        <> expected_measurement.measured_journal_authority_checkpoint_digest
     || payload_digest <> sha256 payload || signature = "" || stored <> canonical
  then failf "sounio-continuity-independent-measurement-mismatch";
  if not
       (ed25519_verify observer_public_key (Filename.dirname path) payload signature)
  then failf "sounio-continuity-independent-measurement-signature-invalid";
  { measured_observation =
      { observer_key_id; observer_principal_id; subject_signer_key_id;
        subject_principal_id; subject_receipt_digest;
        observation_digest = sha256 stored };
    measured_generation_facts = measurement }

let verify_independent_pre_spawn_admission paths pane_id predecessor =
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
    let attestation_path =
      Filename.concat predecessor_dir "sounio-continuity.observer-attestation"
    in
    let measurement_required = independent_measurement_required () in
    let observation, measurement =
      if measurement_required then
        let verified =
          verify_independent_measurement_attestation ~subject
            ~subject_public_key:signer_public_key ~observer_public_key ~paths
            ~pane_id ~predecessor attestation_path
        in
        (verified.measured_observation,
         Some verified.measured_generation_facts)
      else
        (verify_independent_observation_attestation ~subject
           ~subject_public_key:signer_public_key ~observer_public_key
           attestation_path,
         None)
    in
    let common_frame =
      [ sounio_continuity_token "predecessor-receipt"
          subject.signed_receipt_digest;
        sounio_continuity_token "principal-authority"
          subject.signed_principal_id;
        sounio_continuity_token "principal-authority"
          observation.observer_principal_id;
        sounio_continuity_token "independent-observation"
          observation.observation_digest ]
    in
    let authority_required = observation_authority_required () in
    let frame, expected =
      match measurement with
      | None ->
          (String.concat " " ("9003" :: common_frame) ^ "\n",
           "SOUNIO_CONTINUITY_PRESPAWN_ACCEPT schema=loom-native-pre-spawn-v1 authority=disjoint-principals")
      | Some measured ->
          let measured_tokens =
            [ sounio_continuity_token "generation"
                measured.measured_generation;
              sounio_continuity_token "generation-fingerprint"
                measured.measured_generation_fingerprint;
              sounio_continuity_token "semantic-head"
                measured.measured_semantic_head;
              sounio_continuity_token "guardian-head"
                measured.measured_guardian_head ]
          in
          let decision_tokens =
            [ List.nth facts 2; List.nth facts 3; List.nth facts 4;
              List.nth facts 5 ]
          in
          if authority_required then
            let decision_digests =
              match subject.signed_fact_digests with
              | Some digests -> digests
              | None ->
                  failf "sounio-continuity-observation-authority-requires-receipt-v3"
            in
            if measured.measured_journal_authority_principal_id = ""
               || measured.measured_journal_authority_epoch <= 0
               || not
                    (valid_sha256
                       measured.measured_journal_authority_checkpoint_digest)
            then failf "sounio-continuity-journal-authority-evidence-missing";
            if measured.measured_journal_authority_required_quorum = 2 then
              let principal_a, principal_b, principal_c =
                match measured.measured_journal_authority_principal_ids with
                | [ a; b; c ] -> (a, b, c)
                | _ ->
                    failf "sounio-continuity-journal-quorum-member-count"
              in
              let authority_frame =
                [ sounio_continuity_token "predecessor-receipt"
                    subject.signed_receipt_digest;
                  sounio_continuity_token "principal-authority"
                    subject.signed_principal_id;
                  sounio_continuity_token "principal-authority"
                    observation.observer_principal_id;
                  sounio_continuity_token "principal-authority" principal_a;
                  sounio_continuity_token "principal-authority" principal_b;
                  sounio_continuity_token "principal-authority" principal_c;
                  string_of_int measured.measured_journal_authority_required_quorum;
                  string_of_int
                    measured.measured_journal_authority_min_valid_signatures;
                  sounio_continuity_token "independent-observation"
                    observation.observation_digest;
                  sounio_continuity_token "journal-authority-checkpoint"
                    measured.measured_journal_authority_checkpoint_digest;
                  string_of_int measured.measured_journal_authority_epoch ]
              in
              (String.concat " "
                 ("9006" :: authority_frame @ decision_tokens @ measured_tokens
                  @ fact_digest_limbs decision_digests
                  @ fact_digest_limbs measured.measured_fact_digests)
               ^ "\n",
               "SOUNIO_CONTINUITY_PRESPAWN_ACCEPT schema=loom-native-pre-spawn-v4 authority=five-principals+2-of-3-journal-quorum+full-sha256-agreement")
            else
              let authority_frame =
                [ sounio_continuity_token "predecessor-receipt"
                    subject.signed_receipt_digest;
                  sounio_continuity_token "principal-authority"
                    subject.signed_principal_id;
                  sounio_continuity_token "principal-authority"
                    observation.observer_principal_id;
                  sounio_continuity_token "principal-authority"
                    measured.measured_journal_authority_principal_id;
                  sounio_continuity_token "independent-observation"
                    observation.observation_digest;
                  sounio_continuity_token "journal-authority-checkpoint"
                    measured.measured_journal_authority_checkpoint_digest;
                  string_of_int measured.measured_journal_authority_epoch ]
              in
              (String.concat " "
                 ("9005" :: authority_frame @ decision_tokens @ measured_tokens
                  @ fact_digest_limbs decision_digests
                  @ fact_digest_limbs measured.measured_fact_digests)
               ^ "\n",
               "SOUNIO_CONTINUITY_PRESPAWN_ACCEPT schema=loom-native-pre-spawn-v3 authority=three-principals+full-sha256-agreement")
          else
            (String.concat " "
               ("9004" :: common_frame @ decision_tokens @ measured_tokens)
             ^ "\n",
             "SOUNIO_CONTINUITY_PRESPAWN_ACCEPT schema=loom-native-pre-spawn-v2 authority=disjoint-principals+measured-fact-agreement")
    in
    let verdict =
      try process_exchange adapter [| adapter |] frame
      with Loom_error error ->
        if measurement_required then
          failf "sounio-continuity-pre-spawn-measurement-policy-refused:%s" error
        else failf "sounio-continuity-pre-spawn-policy-refused:%s" error
    in
    if verdict <> expected
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
  let measurement_required = independent_measurement_required () in
  let observer_public_key = independent_observer_public_key () in
  let predecessor_receipt_digest, signer_key_id, signer_principal_id,
      public_key, observer_key_id, observer_principal_id,
      independent_observation_digest, journal_authority_principal_id,
      journal_authority_principal_ids, journal_authority_required_quorum,
      journal_authority_min_valid_signatures, journal_authority_epoch,
      journal_authority_checkpoint_digest =
    match (signing, lineage.predecessor_instance) with
    | Unsigned_continuity, _ when independent_required ->
        failf "sounio-continuity-independent-observer-requires-signed-receipts"
    | Unsigned_continuity, _ ->
        ("", "", "", "", "", "", "", "", [], 0, 0, 0, "")
    | Ed25519_continuity keys, "" ->
        ("", keys.key_id, keys.principal_id, keys.public_key, "", "", "", "",
         [], 0, 0, 0, "")
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
        let observer_key_id, observer_principal_id, observation_digest,
            journal_principal, journal_principals, journal_required,
            journal_min_valid, journal_epoch, journal_checkpoint =
          match observer_public_key with
          | None -> ("", "", "", "", [], 0, 0, 0, "")
          | Some observer_key ->
              let observation_path =
                Filename.concat predecessor_dir
                  "sounio-continuity.observer-attestation"
              in
              let observation, journal_principal, journal_principals,
                  journal_required, journal_min_valid, journal_epoch,
                  journal_checkpoint =
                if measurement_required then
                  let verified =
                    verify_independent_measurement_attestation ~subject:receipt
                      ~subject_public_key:keys.public_key
                      ~observer_public_key:observer_key ~paths ~pane_id
                      ~predecessor observation_path
                  in
                  let measured = verified.measured_generation_facts in
                  (verified.measured_observation,
                   measured.measured_journal_authority_principal_id,
                   measured.measured_journal_authority_principal_ids,
                   measured.measured_journal_authority_required_quorum,
                   measured.measured_journal_authority_min_valid_signatures,
                   measured.measured_journal_authority_epoch,
                   measured.measured_journal_authority_checkpoint_digest)
                else
                  (verify_independent_observation_attestation ~subject:receipt
                     ~subject_public_key:keys.public_key
                     ~observer_public_key:observer_key observation_path,
                   "", [], 0, 0, 0, "")
              in
              (observation.observer_key_id, observation.observer_principal_id,
               observation.observation_digest, journal_principal,
               journal_principals, journal_required, journal_min_valid,
               journal_epoch, journal_checkpoint)
        in
        (receipt.signed_receipt_digest, receipt.signed_key_id,
         receipt.signed_principal_id, keys.public_key, observer_key_id,
         observer_principal_id, observation_digest, journal_principal,
         journal_principals, journal_required, journal_min_valid, journal_epoch,
         journal_checkpoint)
  in
  let chain_values =
    [ pane_id; session_id; instance; fingerprint; semantic_head; guardian_head;
      lineage.lineage_head; lineage.predecessor_instance;
      lineage.predecessor_semantic_head; lineage.predecessor_guardian_head;
      lineage.latest_transition; string_of_int lineage.transition_count;
      string_of_int lineage.pod_resurrection_count;
      predecessor_receipt_digest; signer_principal_id; observer_key_id;
      observer_principal_id; independent_observation_digest ]
  in
  let chain_values =
    if observation_authority_required () then
      chain_values
      @ journal_authority_principal_ids
      @ [ journal_authority_principal_id;
          string_of_int journal_authority_required_quorum;
          string_of_int journal_authority_min_valid_signatures;
          string_of_int journal_authority_epoch;
          journal_authority_checkpoint_digest ]
    else chain_values
  in
  let chain_material = String.concat "\000" chain_values
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
  let independently_measured_mode =
    measurement_required && lineage.predecessor_instance <> ""
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
        let fact_digests =
          continuity_fact_digests instance fingerprint semantic_head guardian_head
        in
        let decision_facts =
          { decision_generation = instance;
            decision_generation_fingerprint = fingerprint;
            decision_semantic_head = semantic_head;
            decision_guardian_head = guardian_head }
        in
        let authority_receipt = observation_authority_required () in
        let payload =
          if authority_receipt then
            signed_continuity_payload_v3 keys.key_id runtime_digest facts_digest
              facts decision_facts fact_digests verdict
          else
            signed_continuity_payload keys.key_id runtime_digest facts_digest
              facts verdict
        in
        let signature = ed25519_sign signing generation_dir payload in
        if not (ed25519_verify keys.public_key generation_dir payload signature) then
          failf "sounio-continuity-signing-keypair-mismatch";
        if authority_receipt then
          signed_continuity_receipt_v3 keys.key_id runtime_digest facts_digest
            facts decision_facts fact_digests verdict (sha256 payload) signature
        else
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
         || (observation_authority_required ()
             && verified.signed_fact_digests = None)
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
    independent_measurement_verified = independently_measured_mode;
    observer_key_id; observer_principal_id; independent_observation_digest;
    observation_authority_verified =
      observation_authority_required () && independently_measured_mode;
    full_digest_agreement_verified =
      observation_authority_required () && independently_measured_mode;
    journal_authority_principal_id; journal_authority_principal_ids;
    journal_authority_required_quorum; journal_authority_min_valid_signatures;
    journal_authority_quorum_verified =
      journal_authority_required_quorum = 2
      && journal_authority_min_valid_signatures >= 2;
    journal_authority_epoch;
    journal_authority_checkpoint_digest }

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

let measure_continuity_generation_command cli =
  let state_dir = Unix.realpath (required cli "--state-dir") in
  let pane_id = required cli "--pane-id" in
  let generation = required cli "--generation" in
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
  let paths = beagle_paths state_dir pane_id in
  let measurement =
    independently_measure_generation
      ?decision_facts:subject.signed_decision_facts paths pane_id generation
  in
  let observer_key_id = sha256 (read_file observer_public_key) in
  let observer_principal_id = ed25519_principal_id observer_public_key in
  let signing =
    Ed25519_continuity
      { private_key = observer_private_key; public_key = observer_public_key;
        key_id = observer_key_id; principal_id = observer_principal_id }
  in
  let payload =
    independent_measurement_payload observer_key_id observer_principal_id
      subject.signed_key_id subject.signed_principal_id
      subject.signed_receipt_digest subject.signed_facts_digest
      subject.signed_adapter_digest measurement
  in
  let directory = Filename.dirname output_path in
  if not (Sys.file_exists directory) || (Unix.stat directory).st_kind <> S_DIR then
    failf "sounio-continuity-measurement-output-directory-missing:%s" directory;
  let signature = ed25519_sign signing directory payload in
  if not (ed25519_verify observer_public_key directory payload signature) then
    failf "sounio-continuity-measurement-keypair-mismatch";
  let attestation =
    independent_measurement_receipt observer_key_id observer_principal_id
      subject.signed_key_id subject.signed_principal_id
      subject.signed_receipt_digest subject.signed_facts_digest
      subject.signed_adapter_digest measurement (sha256 payload) signature
  in
  if Sys.file_exists output_path then (
    if read_file output_path <> attestation then
      failf "sounio-continuity-independent-measurement-output-conflict")
  else atomic_write output_path attestation;
  let verified =
    verify_independent_measurement_attestation ~subject ~subject_public_key
      ~observer_public_key ~paths ~pane_id ~predecessor:generation output_path
  in
  let measured = verified.measured_generation_facts in
  if measured.measured_journal_authority_principal_id = "" then
    Printf.printf
      "LOOM_CONTINUITY_INDEPENDENT_MEASUREMENT_ATTESTED schema=loom-independent-measurement-attestation-v1 observer_principal_id=%s subject_principal_id=%s measured_generation=%s measured_generation_fingerprint=%s measured_semantic_head=%s measured_guardian_head=%s observation_sha256=%s\n%!"
      verified.measured_observation.observer_principal_id
      verified.measured_observation.subject_principal_id
      measured.measured_generation measured.measured_generation_fingerprint
      measured.measured_semantic_head measured.measured_guardian_head
      verified.measured_observation.observation_digest
  else if measured.measured_journal_authority_required_quorum = 2 then
    Printf.printf
      "LOOM_CONTINUITY_INDEPENDENT_MEASUREMENT_ATTESTED schema=loom-independent-measurement-attestation-v3 observer_principal_id=%s subject_principal_id=%s journal_authority_principal_set_id=%s journal_authority_required_quorum=%d journal_authority_min_valid_signatures=%d journal_authority_epoch=%d journal_authority_checkpoint_sha256=%s measured_generation=%s measured_generation_fingerprint=%s measured_semantic_head=%s measured_guardian_head=%s measured_generation_sha256=%s measured_generation_fingerprint_sha256=%s measured_semantic_head_sha256=%s measured_guardian_head_sha256=%s observation_sha256=%s\n%!"
      verified.measured_observation.observer_principal_id
      verified.measured_observation.subject_principal_id
      measured.measured_journal_authority_principal_id
      measured.measured_journal_authority_required_quorum
      measured.measured_journal_authority_min_valid_signatures
      measured.measured_journal_authority_epoch
      measured.measured_journal_authority_checkpoint_digest
      measured.measured_generation measured.measured_generation_fingerprint
      measured.measured_semantic_head measured.measured_guardian_head
      measured.measured_fact_digests.generation_digest
      measured.measured_fact_digests.fingerprint_digest
      measured.measured_fact_digests.semantic_head_digest
      measured.measured_fact_digests.guardian_head_digest
      verified.measured_observation.observation_digest
  else
    Printf.printf
      "LOOM_CONTINUITY_INDEPENDENT_MEASUREMENT_ATTESTED schema=loom-independent-measurement-attestation-v2 observer_principal_id=%s subject_principal_id=%s journal_authority_principal_id=%s journal_authority_epoch=%d journal_authority_checkpoint_sha256=%s measured_generation=%s measured_generation_fingerprint=%s measured_semantic_head=%s measured_guardian_head=%s measured_generation_sha256=%s measured_generation_fingerprint_sha256=%s measured_semantic_head_sha256=%s measured_guardian_head_sha256=%s observation_sha256=%s\n%!"
      verified.measured_observation.observer_principal_id
      verified.measured_observation.subject_principal_id
      measured.measured_journal_authority_principal_id
      measured.measured_journal_authority_epoch
      measured.measured_journal_authority_checkpoint_digest
      measured.measured_generation measured.measured_generation_fingerprint
      measured.measured_semantic_head measured.measured_guardian_head
      measured.measured_fact_digests.generation_digest
      measured.measured_fact_digests.fingerprint_digest
      measured.measured_fact_digests.semantic_head_digest
      measured.measured_fact_digests.guardian_head_digest
      verified.measured_observation.observation_digest

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
  let generation_fingerprint = beagle_generation_fingerprint pane_id descriptor in
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
        independent_observation_verified = false;
        independent_measurement_verified = false; observer_key_id = "";
        observer_principal_id = ""; independent_observation_digest = "";
        observation_authority_verified = false;
        full_digest_agreement_verified = false;
        journal_authority_principal_id = "";
        journal_authority_principal_ids = [];
        journal_authority_required_quorum = 0;
        journal_authority_min_valid_signatures = 0;
        journal_authority_quorum_verified = false;
        journal_authority_epoch = 0;
        journal_authority_checkpoint_digest = "" }
  in
  Printf.sprintf
    "{\"paneId\":%s,\"sessionId\":%s,\"pid\":%s,\"status\":%s,\"createdAt\":%s,\"updatedAt\":%s,\"cwd\":%s,\"cols\":%s,\"rows\":%s,\"snapshot\":%s,\"supervisorRuntime\":%s,\"supervisorProtocol\":%s,\"loomInstanceId\":%s,\"loomKernelPid\":%s,\"loomGuardianPid\":%s,\"loomState\":%s,\"loomCursor\":%d,\"generationFingerprint\":%s,\"authorityStatus\":{\"owner\":\"loom\",\"journalVerified\":%s,\"semanticJournalHead\":%s,\"guardianJournalHead\":%s,\"kernelRecoveryCount\":%d,\"lineageVerified\":%s,\"generationLineageHead\":%s,\"generationTransition\":%s,\"generationTransitionCount\":%d,\"podResurrectionCount\":%d,\"predecessorInstanceId\":%s,\"predecessorSemanticJournalHead\":%s,\"predecessorGuardianJournalHead\":%s,\"sounioPolicyVerified\":%s,\"sounioPolicyReceipt\":%s,\"sounioPolicyRuntimeDigest\":%s,\"sounioPolicySignatureVerified\":%s,\"sounioPolicySignerKeyId\":%s,\"sounioPolicySignerPrincipalId\":%s,\"sounioPolicyPredecessorReceipt\":%s,\"sounioPolicyIndependentObservationVerified\":%s,\"sounioPolicyIndependentMeasurementVerified\":%s,\"sounioPolicyObserverKeyId\":%s,\"sounioPolicyObserverPrincipalId\":%s,\"sounioPolicyIndependentObservation\":%s,\"sounioPolicyObservationAuthorityVerified\":%s,\"sounioPolicyFullDigestAgreementVerified\":%s,\"sounioPolicyJournalAuthorityPrincipalId\":%s,\"sounioPolicyJournalAuthorityPrincipalIds\":%s,\"sounioPolicyJournalAuthorityRequiredQuorum\":%d,\"sounioPolicyJournalAuthorityMinValidSignatures\":%d,\"sounioPolicyJournalAuthorityQuorumVerified\":%s,\"sounioPolicyJournalAuthorityEpoch\":%d,\"sounioPolicyJournalAuthorityCheckpoint\":%s}}"
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
    (if continuity.independent_measurement_verified then "true" else "false")
    (json_quote continuity.observer_key_id)
    (json_quote continuity.observer_principal_id)
    (json_quote continuity.independent_observation_digest)
    (if continuity.observation_authority_verified then "true" else "false")
    (if continuity.full_digest_agreement_verified then "true" else "false")
    (json_quote continuity.journal_authority_principal_id)
    (json_string_list continuity.journal_authority_principal_ids)
    continuity.journal_authority_required_quorum
    continuity.journal_authority_min_valid_signatures
    (if continuity.journal_authority_quorum_verified then "true" else "false")
    continuity.journal_authority_epoch
    (json_quote continuity.journal_authority_checkpoint_digest)

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
      verify_independent_pre_spawn_admission paths pane_id predecessor;
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
               else if path = "/api/fleet" then
                 http_response "200 OK" "application/json" (fleet_json root cwd)
               else if path = "/api/events" then
                 http_response "200 OK" "application/json" (events_json root)
               else if path = "/api/events.arrow" then
                 (try
                    let body, projection = events_arrow root in
                    http_response
                      ~headers:
                        [ ("X-Loom-Schema", "loom-spectral-events-v1");
                          ("X-Loom-Authority", "verified-derived");
                          ( "X-Loom-Rows",
                            string_of_int
                              (List.length projection.spectral_rows) );
                          ( "X-Loom-Guardian-Sessions",
                            string_of_int projection.guardian_sessions );
                          ( "X-Loom-Legacy-Semantic-Only-Sessions",
                            string_of_int
                              projection.legacy_semantic_only_sessions ) ]
                      "200 OK" "application/vnd.apache.arrow.stream" body
                  with error ->
                    http_response "409 Conflict" "application/json"
                      (Printf.sprintf
                         "{\"error\":\"spectral_projection_refused\",\"reason\":\"%s\"}"
                         (json_escape (Printexc.to_string error))))
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

let tui_health_color = function
  | Loom_lane_health.Working -> "\027[38;5;48m"
  | Loom_lane_health.Idle -> "\027[38;5;51m"
  | Loom_lane_health.Blocked -> "\027[38;5;220m"
  | Loom_lane_health.Disconnected -> "\027[38;5;208m"
  | Loom_lane_health.Unresponsive -> "\027[38;5;203m"
  | Loom_lane_health.Orphaned -> "\027[38;5;171m"
  | Loom_lane_health.Dead -> "\027[38;5;244m"
  | Loom_lane_health.Conflicted -> "\027[1;38;5;197m"
  | Loom_lane_health.Unknown -> "\027[38;5;250m"

let tui_clip width value =
  if String.length value <= width then value
  else if width <= 1 then String.sub value 0 width
  else String.sub value 0 (width - 1) ^ "~"

let tui_window_size () =
  match Sys.getenv_opt "LINES" with
  | Some value ->
      (try max 6 (min 30 (int_of_string value - 10)) with _ -> 14)
  | None -> 14

let rec tui_drop count values =
  if count <= 0 then values
  else match values with [] -> [] | _ :: tail -> tui_drop (count - 1) tail

let rec tui_take count values =
  if count <= 0 then []
  else match values with [] -> [] | head :: tail -> head :: tui_take (count - 1) tail

let tui_machine_snapshot root cwd =
  let coordination_available, snapshot_utc, snapshot_authorized, lanes =
    load_authority_lanes root cwd
  in
  let values = sorted_authority_values lanes in
  Printf.printf
    "LOOM_TUI schema=loom-truthful-fleet-tui-v1 authority=Sounio realization=OCaml semantics_sha256=%s snapshot_utc=%s coordination_available=%s observation_authorized=%s lanes=%d\n%!"
    Loom_lane_health.parent_semantics_sha256 snapshot_utc
    (if coordination_available then "true" else "false")
    (if snapshot_authorized then "true" else "false")
    (List.length values);
  List.iter
    (fun lane ->
      let health = truthful_state lane in
      Printf.printf
        "LOOM_TUI_LANE health=%s agent=%s lane=%s pid=%s claim=%s presence=%s reason=%s endpoint=%s custody=%s active_obligations=%d pending_obligations=%d progress=%s ready=%s\n%!"
        (Loom_lane_health.name health) lane.authority_agent lane.authority_lane
        (if lane.authority_harness_pid <> "" then lane.authority_harness_pid
         else lane.authority_pid)
        lane.authority_claim lane.authority_presence lane.authority_presence_reason
        lane.authority_endpoint lane.authority_loom_state
        lane.authority_active_obligations lane.authority_pending_obligations
        (if lane.authority_progress_observed then "yes" else "no")
        (if lane.authority_ready_observed then "yes" else "no"))
    values

let tui_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  if flag cli "--machine" then tui_machine_snapshot root cwd
  else if not (Unix.isatty Unix.stdin) then list_command cli
  else
    let original = set_terminal_raw Unix.stdin in
    let selected = ref 0 and running = ref true and notice = ref "" in
    Fun.protect
      ~finally:(fun () -> Unix.tcsetattr Unix.stdin TCSANOW original; print_string "\027[?25h\027[0m\n"; flush Stdlib.stdout)
      (fun () ->
        print_string "\027[?25l";
        while !running do
          let coordination_available, _, snapshot_authorized, lanes =
            load_authority_lanes root cwd
          in
          let values = sorted_authority_values lanes in
          let length = List.length values in
          if !selected >= length then selected := max 0 (length - 1);
          let window = tui_window_size () in
          let first =
            max 0 (min (max 0 (length - window)) (!selected - (window / 2)))
          in
          let visible = tui_take window (tui_drop first values) in
          let count state = count_authority_health values state in
          Printf.printf
            "\027[2J\027[H\027[1;37mSOUNIO LOOM\027[0m  Sounio-authoritative fleet health\n";
          Printf.printf
            "\027[90mauthority=Sounio  realization=OCaml  semantics=%s  observation=%s  lanes=%d\027[0m\n"
            (String.sub Loom_lane_health.parent_semantics_sha256 0 12)
            (if coordination_available && snapshot_authorized then
               "VERIFIED" else "UNKNOWN")
            length;
          Printf.printf
            "\027[38;5;48mWORK %d\027[0m  \027[38;5;51mIDLE %d\027[0m  \027[38;5;220mBLOCK %d\027[0m  \027[38;5;208mDISC %d\027[0m  \027[38;5;203mUNRESP %d\027[0m\n"
            (count Loom_lane_health.Working) (count Loom_lane_health.Idle)
            (count Loom_lane_health.Blocked)
            (count Loom_lane_health.Disconnected)
            (count Loom_lane_health.Unresponsive);
          Printf.printf
            "\027[38;5;171mORPH %d\027[0m  \027[38;5;244mDEAD %d\027[0m  \027[1;38;5;197mCONFLICT %d\027[0m  \027[38;5;250mUNKNOWN %d\027[0m\n"
            (count Loom_lane_health.Orphaned)
            (count Loom_lane_health.Dead)
            (count Loom_lane_health.Conflicted)
            (count Loom_lane_health.Unknown);
          Printf.printf
            "\027[90m   %-12s %-10s %-26s %-7s %-3s %-8s\027[0m\n"
            "HEALTH" "AGENT" "LANE" "PID" "OBL" "ENDPOINT";
          List.iteri
            (fun visible_index lane ->
              let index = first + visible_index in
              let health = truthful_state lane in
              let marker = if index = !selected then " > " else "   " in
              let pid =
                if lane.authority_harness_pid <> "" then
                  lane.authority_harness_pid else lane.authority_pid
              in
              Printf.printf "%s%s%s%-12s %-10s %-26s %-7s %-3d %-8s\027[0m\n"
                (if index = !selected then "\027[1m" else "")
                (tui_health_color health) marker (Loom_lane_health.name health)
                (tui_clip 10 lane.authority_agent)
                (tui_clip 26 lane.authority_lane) (tui_clip 7 pid)
                (lane.authority_active_obligations
                 + lane.authority_pending_obligations)
                (tui_clip 8 lane.authority_endpoint))
            visible;
          Printf.printf "\027[90mshowing %d-%d/%d\027[0m\n"
            (if length = 0 then 0 else first + 1)
            (min length (first + List.length visible)) length;
          (match List.nth_opt values !selected with
          | None -> print_string "\nno observed lanes\n"
          | Some lane ->
              Printf.printf
                "selected %s/%s  claim=%s presence=%s(%s) custody=%s\n"
                (tui_clip 14 lane.authority_agent)
                (tui_clip 42 lane.authority_lane) lane.authority_claim
                lane.authority_presence lane.authority_presence_reason
                lane.authority_loom_state;
              Printf.printf
                "endpoint=%s progress=%s ready=%s obligations=%d/%d%s\n"
                lane.authority_endpoint
                (if lane.authority_progress_observed then "yes" else "no")
                (if lane.authority_ready_observed then "yes" else "no")
                lane.authority_active_obligations
                lane.authority_pending_obligations
                (if !notice = "" then "" else "  " ^ !notice));
          print_string "\027[90mj/k select   enter attach   o observe   r refresh   q quit   detach: Ctrl-]\027[0m\n";
          flush Stdlib.stdout;
          let readable, _, _ = Unix.select [ Unix.stdin ] [] [] 1.0 in
          if readable <> [] then
            let byte = Bytes.create 1 in
            if Unix.read Unix.stdin byte 0 1 = 1 then
              match Bytes.get byte 0 with
              | 'q' -> running := false
              | 'j' -> if !selected + 1 < length then incr selected
              | 'k' -> if !selected > 0 then decr selected
              | 'r' -> notice := "snapshot refreshed"
              | '\r' | '\n' | 'o' as key -> (
                  match List.nth_opt values !selected with
                  | None -> ()
                  | Some lane when lane.authority_loom_state <> "active" ->
                      notice :=
                        (if lane.authority_loom_state = "recoverable" then
                           "attach refused: recover custody first"
                         else "attach refused: no Loom custody")
                  | Some lane ->
                      Unix.tcsetattr Unix.stdin TCSANOW original;
                      print_string "\027[2J\027[H\027[?25h";
                      flush Stdlib.stdout;
                      let attach_cli =
                        { options = Hashtbl.copy cli.options; flags = Hashtbl.create 2; rest = [] }
                      in
                      Hashtbl.replace attach_cli.options "--agent" lane.authority_agent;
                      Hashtbl.replace attach_cli.options "--lane" lane.authority_lane;
                      Hashtbl.replace attach_cli.options "--cursor" "auto";
                      (try stream_command attach_cli (key <> 'o') with Loom_error error -> Printf.eprintf "\nLoom: %s\n%!" error);
                      ignore (set_terminal_raw Unix.stdin);
                      print_string "\027[?25l";
                      notice := "")
              | _ -> ()
        done)

let provider_uuid value =
  let rec valid index =
    if index = String.length value then true
    else if List.mem index [ 8; 13; 18; 23 ] then
      value.[index] = '-' && valid (index + 1)
    else
      match value.[index] with
      | '0' .. '9' | 'a' .. 'f' | 'A' .. 'F' -> valid (index + 1)
      | _ -> false
  in
  String.length value = 36 && valid 0

type fleet_spec = {
  fleet_slot : string;
  fleet_kind : string;
  fleet_custody : string;
  fleet_agent : string;
  fleet_home : string;
  fleet_cwd : string;
  fleet_coord_dir : string;
  fleet_enabled : bool;
  fleet_session_id : string;
  fleet_provider_mode : string;
  fleet_provider_session : string;
  fleet_prompt_file : string;
  fleet_prompt_sha256 : string;
  fleet_model : string;
  fleet_unsafe_auto : bool;
}

let fleet_kinds = [ "claude"; "codex"; "kimi"; "grok"; "cursor"; "empryo" ]
let persistent_fleet_kinds = [ "claude"; "codex"; "kimi" ]

let fleet_directory root = Filename.concat root "fleet"

let fleet_spec_path root slot =
  Filename.concat (fleet_directory root) (slug slot ^ ".state")

let fleet_prompt_directory root = Filename.concat (fleet_directory root) "prompts"

let fleet_prompt_path root slot =
  Filename.concat (fleet_prompt_directory root) (slug slot ^ ".txt")

let fleet_spec_fields spec =
  [
    ("version", "3");
    ("enabled", if spec.fleet_enabled then "true" else "false");
    ("slot", spec.fleet_slot);
    ("kind", spec.fleet_kind);
    ("custody", spec.fleet_custody);
    ("agent", spec.fleet_agent);
    ("home", spec.fleet_home);
    ("cwd", spec.fleet_cwd);
    ("coord_dir", spec.fleet_coord_dir);
    ("session_id", spec.fleet_session_id);
    ("provider_mode", spec.fleet_provider_mode);
    ("provider_session", spec.fleet_provider_session);
    ("prompt_file", spec.fleet_prompt_file);
    ("prompt_sha256", spec.fleet_prompt_sha256);
    ("model", spec.fleet_model);
    ("unsafe_auto", if spec.fleet_unsafe_auto then "true" else "false");
  ]

let validate_fleet_atom name value =
  if value = "" || String.exists (fun character -> character = '\n' || character = '\r') value
  then failf "invalid fleet %s" name

let fleet_spec_of_values path values =
  let version = table_value values "version" in
  if version <> "1" && version <> "2" && version <> "3" then
    failf "fleet catalog version is not supported: %s" path;
  let slot = table_value values "slot" in
  let kind = table_value values "kind" in
  let custody = if version = "1" then "agentd" else table_value values "custody" in
  let agent = if version = "1" then kind else table_value values "agent" in
  let home = table_value values "home" in
  let cwd = table_value values "cwd" in
  let coord_dir = if version = "1" then "" else table_value values "coord_dir" in
  List.iter (fun (name, value) -> validate_fleet_atom name value)
    [ ("slot", slot); ("kind", kind); ("custody", custody); ("agent", agent);
      ("home", home); ("cwd", cwd) ];
  if not (List.mem kind fleet_kinds) then
    failf "unsupported fleet kind %s in %s" kind path;
  if custody <> "agentd" && custody <> "loom" then
    failf "unsupported fleet custody %s in %s" custody path;
  if custody = "loom" && not (List.mem kind persistent_fleet_kinds) then
    failf "persistent fleet provider unavailable for kind %s in %s" kind path;
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
  let session_id = if version = "1" then "" else table_value values "session_id" in
  let provider_mode =
    if version = "3" then table_value values "provider_mode"
    else if custody = "loom" then "new" else ""
  in
  let provider_session =
    if version = "3" then table_value values "provider_session" else ""
  in
  let prompt_file = if version = "1" then "" else table_value values "prompt_file" in
  let prompt_sha256 = if version = "1" then "" else table_value values "prompt_sha256" in
  let model = if version = "1" then "" else table_value values "model" in
  let unsafe_auto =
    if version = "1" then false
    else
      match table_value values "unsafe_auto" with
      | "true" -> true
      | "false" -> false
      | _ -> failf "invalid unsafe_auto state in %s" path
  in
  if String.exists (fun character -> character = '\n' || character = '\r') model then
    failf "invalid fleet model in %s" path;
  if custody = "loom" then (
    if not (provider_uuid session_id) then
      failf "invalid fleet session id for slot %s" slot;
    if Filename.is_relative prompt_file then
      failf "fleet prompt file must be absolute for slot %s" slot;
    let expected_directory = Filename.concat (Filename.dirname path) "prompts" in
    if Filename.dirname prompt_file <> expected_directory then
      failf "fleet prompt escaped sealed catalog storage for slot %s" slot;
    if not (Sys.file_exists prompt_file && (Unix.stat prompt_file).st_kind = S_REG) then
      failf "fleet prompt is unavailable for slot %s: %s" slot prompt_file;
    let actual_prompt_sha256 = sha256 (read_file prompt_file) in
    if prompt_sha256 = "" || actual_prompt_sha256 <> prompt_sha256 then
      failf "fleet prompt digest mismatch for slot %s" slot;
    if provider_mode <> "new" && provider_mode <> "resume" then
      failf "invalid fleet provider mode for slot %s: %s" slot provider_mode;
    if provider_mode = "new" && provider_session <> "" then
      failf "new fleet provider contains resume identity for slot %s" slot;
    if provider_mode = "resume" then (
      if provider_session = "" then
        failf "fleet provider resume identity is missing for slot %s" slot;
      if kind <> "claude" then
        failf "persistent provider resume unavailable for slot %s kind %s"
          slot kind;
      if kind = "claude" && not (provider_uuid provider_session)
      then failf "invalid fleet provider resume identity for slot %s" slot);
    if coord_dir <> ""
       && (Filename.is_relative coord_dir || not (Sys.file_exists coord_dir)
           || not (Sys.is_directory coord_dir))
    then failf "fleet coordination authority is unavailable for slot %s: %s"
      slot coord_dir)
  else if
    session_id <> "" || provider_mode <> "" || provider_session <> ""
    || prompt_file <> "" || prompt_sha256 <> ""
    || coord_dir <> "" || model <> "" || unsafe_auto
  then failf "agentd fleet slot %s contains Loom-only authority fields" slot;
  { fleet_slot = slot; fleet_kind = kind; fleet_custody = custody;
    fleet_agent = agent; fleet_home = home; fleet_cwd = cwd;
    fleet_coord_dir = coord_dir; fleet_enabled = enabled;
    fleet_session_id = session_id;
    fleet_provider_mode = provider_mode;
    fleet_provider_session = provider_session;
    fleet_prompt_file = prompt_file; fleet_prompt_sha256 = prompt_sha256;
    fleet_model = model; fleet_unsafe_auto = unsafe_auto }

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

let run_captured ?environment executable arguments =
  let environment = Option.value ~default:(Unix.environment ()) environment in
  let reader, writer = Unix.pipe () in
  Unix.set_close_on_exec reader;
  match Unix.fork () with
  | 0 ->
      Unix.close reader;
      Unix.dup2 writer Unix.stdout;
      Unix.dup2 writer Unix.stderr;
      if writer <> Unix.stdout && writer <> Unix.stderr then Unix.close writer;
      (try Unix.execve executable (Array.of_list (executable :: arguments)) environment
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

type provider_auth_probe =
  | Codex_auth
  | Claude_auth
  | Kimi_auth
  | Grok_auth
  | Opencode_auth

type provider_spec = {
  provider_id : string;
  provider_name : string;
  provider_executable : string;
  provider_version_args : string list;
  provider_stream : string;
  provider_session_binding : string;
  provider_capabilities : string list;
  provider_auth_probe : provider_auth_probe;
  provider_login_args : string list;
}

type provider_status = {
  status_spec : provider_spec;
  status_executable : string option;
  status_version : string;
  status_auth : string;
  status_auth_reason : string;
}

type provider_plan = {
  plan_spec : provider_spec;
  plan_lifecycle : string;
  plan_stdin_authority : string;
  plan_mode : string;
  plan_executable : string;
  plan_cwd : string;
  plan_session_id : string;
  plan_provider_session : string;
  plan_model : string;
  plan_unsafe_auto : bool;
  plan_context_isolation : bool;
  plan_prompt_transport : string;
  plan_prompt : string;
  plan_argv : string list;
}

let provider_abi_schema = "loom-provider-abi-v1"
let max_provider_prompt_bytes = 1024 * 1024

let provider_specs =
  [
    { provider_id = "codex"; provider_name = "OpenAI Codex CLI";
      provider_executable = "codex"; provider_version_args = [ "--version" ];
      provider_stream = "jsonl"; provider_session_binding = "stream-observed";
      provider_capabilities =
        [ "interactive"; "headless"; "event-stream"; "resume"; "model-select";
          "auth-login"; "auth-status"; "doctor"; "context-isolation" ];
      provider_auth_probe = Codex_auth; provider_login_args = [ "login" ] };
    { provider_id = "claude"; provider_name = "Anthropic Claude Code";
      provider_executable = "claude"; provider_version_args = [ "--version" ];
      provider_stream = "stream-json"; provider_session_binding = "caller";
      provider_capabilities =
        [ "interactive"; "headless"; "event-stream"; "resume"; "model-select";
          "external-session-id"; "auth-login"; "auth-status"; "doctor";
          "context-isolation" ];
      provider_auth_probe = Claude_auth;
      provider_login_args = [ "auth"; "login" ] };
    { provider_id = "kimi"; provider_name = "Moonshot Kimi Code CLI";
      provider_executable = "kimi"; provider_version_args = [ "--version" ];
      provider_stream = "stream-json"; provider_session_binding = "native-store";
      provider_capabilities =
        [ "interactive"; "headless"; "event-stream"; "resume"; "model-select";
          "auth-login"; "doctor"; "persistent-input" ];
      provider_auth_probe = Kimi_auth; provider_login_args = [ "login" ] };
    { provider_id = "grok"; provider_name = "xAI Grok CLI";
      provider_executable = "grok"; provider_version_args = [ "--version" ];
      provider_stream = "streaming-json"; provider_session_binding = "caller";
      provider_capabilities =
        [ "interactive"; "headless"; "event-stream"; "resume"; "model-select";
          "external-session-id"; "auth-login"; "doctor"; "context-isolation" ];
      provider_auth_probe = Grok_auth; provider_login_args = [ "login" ] };
    { provider_id = "opencode"; provider_name = "OpenCode";
      provider_executable = "opencode"; provider_version_args = [ "--version" ];
      provider_stream = "json-events"; provider_session_binding = "stream-observed";
      provider_capabilities =
        [ "interactive"; "headless"; "event-stream"; "resume"; "model-select";
          "auth-login"; "auth-status"; "multi-provider"; "context-isolation" ];
      provider_auth_probe = Opencode_auth;
      provider_login_args = [ "providers"; "login" ] };
  ]

let provider_spec provider_id =
  match List.find_opt (fun spec -> spec.provider_id = provider_id) provider_specs with
  | Some spec -> spec
  | None -> failf "unsupported-provider:%s" provider_id

let provider_override_name spec =
  "SOUNIO_LOOM_PROVIDER_" ^ String.uppercase_ascii spec.provider_id

let valid_provider_executable path =
  try
    let resolved = Unix.realpath path in
    if (Unix.stat resolved).st_kind <> S_REG then None
    else (
      Unix.access resolved [ X_OK ];
      Some resolved)
  with _ -> None

let provider_executable spec =
  match Sys.getenv_opt (provider_override_name spec) with
  | Some path when path <> "" ->
      if Filename.is_relative path then
        failf "provider-override-must-be-absolute:%s" spec.provider_id;
      (match valid_provider_executable path with
      | Some resolved -> Some resolved
      | None -> failf "provider-override-is-not-executable:%s" spec.provider_id)
  | _ ->
      let search = Option.value ~default:"" (Sys.getenv_opt "PATH") in
      split_on ':' search
      |> List.filter (fun directory -> directory <> "")
      |> List.find_map (fun directory ->
             valid_provider_executable
               (Filename.concat directory spec.provider_executable))

let string_contains value needle =
  let value_length = String.length value and needle_length = String.length needle in
  let rec search index =
    if needle_length = 0 then true
    else if index + needle_length > value_length then false
    else if String.sub value index needle_length = needle then true
    else search (index + 1)
  in
  search 0

let first_probe_line value =
  match
    split_on '\n' value |> List.map trim |> List.find_opt (fun line -> line <> "")
  with
  | None -> ""
  | Some line when String.length line > 256 -> String.sub line 0 256
  | Some line -> line

let provider_version spec executable =
  let result = run_captured executable spec.provider_version_args in
  if result.captured_code = 0 then first_probe_line result.captured_output else ""

let provider_auth_status spec executable =
  match spec.provider_auth_probe with
  | Codex_auth ->
      let result = run_captured executable [ "login"; "status" ] in
      if result.captured_code = 0
         && string_contains result.captured_output "Logged in"
      then ("authenticated", "native-login-status")
      else if string_contains result.captured_output "Not logged in" then
        ("unauthenticated", "native-login-status")
      else ("unknown", "native-login-status-inconclusive")
  | Claude_auth ->
      let result = run_captured executable [ "auth"; "status"; "--json" ] in
      (try
         match json_object_field (parse_json (trim result.captured_output)) "loggedIn" with
         | Some (Json_bool true) -> ("authenticated", "native-auth-status")
         | Some (Json_bool false) -> ("unauthenticated", "native-auth-status")
         | _ -> ("unknown", "native-auth-status-inconclusive")
       with Loom_error _ ->
         if result.captured_code = 0 then
           ("unknown", "native-auth-status-invalid-json")
         else ("unknown", "native-auth-status-failed"))
  | Kimi_auth -> ("unknown", "native-cli-has-no-offline-auth-status")
  | Grok_auth -> ("unknown", "native-cli-has-no-offline-auth-status")
  | Opencode_auth ->
      let result = run_captured executable [ "providers"; "list" ] in
      if result.captured_code = 0 then ("delegated", "native-multiprovider-store")
      else ("unknown", "native-provider-status-failed")

let provider_status spec =
  match provider_executable spec with
  | None ->
      { status_spec = spec; status_executable = None; status_version = "";
        status_auth = "unavailable"; status_auth_reason = "executable-not-found" }
  | Some executable ->
      let auth, reason = provider_auth_status spec executable in
      { status_spec = spec; status_executable = Some executable;
        status_version = provider_version spec executable; status_auth = auth;
        status_auth_reason = reason }

let provider_status_json status =
  let spec = status.status_spec in
  Printf.sprintf
    "{\"provider\":%s,\"name\":%s,\"installed\":%s,\"executable\":%s,\"version\":%s,\"auth\":%s,\"auth_reason\":%s,\"credential_authority\":\"native\",\"stream\":%s,\"session_binding\":%s,\"capabilities\":%s}"
    (json_quote spec.provider_id) (json_quote spec.provider_name)
    (if status.status_executable = None then "false" else "true")
    (match status.status_executable with None -> "null" | Some path -> json_quote path)
    (json_quote status.status_version) (json_quote status.status_auth)
    (json_quote status.status_auth_reason) (json_quote spec.provider_stream)
    (json_quote spec.provider_session_binding)
    (json_string_list spec.provider_capabilities)

let print_provider_status status =
  let spec = status.status_spec in
  Printf.printf
    "LOOM_PROVIDER_ABI schema=%s provider=%s installed=%s executable=%s version=%s auth=%s auth_reason=%s credential_authority=native stream=%s session_binding=%s capabilities=%s\n%!"
    provider_abi_schema spec.provider_id
    (if status.status_executable = None then "false" else "true")
    (match status.status_executable with None -> "-" | Some path -> field_escape path)
    (if status.status_version = "" then "-" else field_escape status.status_version)
    status.status_auth status.status_auth_reason spec.provider_stream
    spec.provider_session_binding (String.concat "," spec.provider_capabilities)

let provider_list_command cli =
  let statuses = List.map provider_status provider_specs in
  if flag cli "--json" then
    Printf.printf "{\"schema\":%s,\"providers\":[%s]}\n%!"
      (json_quote provider_abi_schema)
      (String.concat "," (List.map provider_status_json statuses))
  else List.iter print_provider_status statuses

let provider_status_command cli =
  let status = required cli "--provider" |> provider_spec |> provider_status in
  if flag cli "--json" then
    Printf.printf "{\"schema\":%s,\"status\":%s}\n%!"
      (json_quote provider_abi_schema) (provider_status_json status)
  else print_provider_status status

let provider_prompt cli =
  match (optional cli "--prompt", optional cli "--prompt-file") with
  | Some _, Some _ -> failf "provider-prompt-source-is-ambiguous"
  | None, None -> failf "provider-prompt-is-required"
  | Some prompt, None ->
      if String.length prompt > max_provider_prompt_bytes then
        failf "provider-prompt-too-large";
      if String.contains prompt '\000' then failf "provider-prompt-contains-nul";
      prompt
  | None, Some path ->
      let resolved = Unix.realpath path in
      let stats = Unix.stat resolved in
      if stats.st_kind <> S_REG then failf "provider-prompt-file-is-not-regular";
      if stats.st_size > max_provider_prompt_bytes then
        failf "provider-prompt-too-large";
      let prompt = read_file resolved in
      if String.contains prompt '\000' then failf "provider-prompt-contains-nul";
      prompt

let provider_model_args spec model =
  if model = "" then []
  else
    match spec.provider_id with
    | "codex" -> [ "-m"; model ]
    | "kimi" -> [ "-m"; model ]
    | "claude" | "grok" | "opencode" -> [ "--model"; model ]
    | _ -> failf "unsupported-provider:%s" spec.provider_id

let provider_unsafe_args spec enabled =
  if not enabled then []
  else
    match spec.provider_id with
    | "codex" -> [ "--dangerously-bypass-approvals-and-sandbox" ]
    | "claude" -> [ "--dangerously-skip-permissions" ]
    | "kimi" -> [ "--auto" ]
    | "grok" -> [ "--always-approve" ]
    | "opencode" -> [ "--auto" ]
    | _ -> failf "unsupported-provider:%s" spec.provider_id

let provider_context_isolation_args spec enabled =
  if not enabled then []
  else
    match spec.provider_id with
    | "codex" -> [ "--ephemeral"; "--ignore-rules" ]
    | "claude" -> [ "--safe-mode" ]
    | "kimi" -> failf "provider-context-isolation-unavailable:kimi"
    | "grok" ->
        [ "--no-memory"; "--no-subagents"; "--disable-web-search";
          "--max-turns"; "2" ]
    | "opencode" -> [ "--pure" ]
    | _ -> failf "unsupported-provider:%s" spec.provider_id

let provider_argv spec lifecycle executable mode cwd session_id provider_session
    model unsafe_auto context_isolation prompt =
  let model_args = provider_model_args spec model in
  let unsafe_args = provider_unsafe_args spec unsafe_auto in
  let context_args =
    if lifecycle = "persistent" && context_isolation then
      failf "persistent-context-isolation-unavailable:%s" spec.provider_id
    else provider_context_isolation_args spec context_isolation
  in
  match (spec.provider_id, lifecycle, mode) with
  | "codex", "turn", "new" ->
      [ executable; "exec"; "--json"; "--color"; "never";
        "--skip-git-repo-check"; "-C"; cwd ]
      @ context_args @ model_args @ unsafe_args @ [ prompt ]
  | "codex", "turn", "resume" ->
      [ executable; "exec"; "--json"; "--color"; "never";
        "--skip-git-repo-check"; "-C"; cwd ]
      @ context_args @ model_args @ unsafe_args
      @ [ "resume"; provider_session; prompt ]
  | "claude", "turn", "new" ->
      [ executable; "--print"; "--output-format"; "stream-json"; "--verbose";
        "--session-id"; session_id ]
      @ context_args @ model_args @ unsafe_args @ [ prompt ]
  | "claude", "turn", "resume" ->
      [ executable; "--print"; "--output-format"; "stream-json"; "--verbose";
        "--resume"; provider_session ]
      @ context_args @ model_args @ unsafe_args @ [ prompt ]
  | "kimi", "turn", "new" ->
      [ executable; "--output-format"; "stream-json" ]
      @ model_args @ unsafe_args @ [ "--prompt"; prompt ]
  | "kimi", "turn", "resume" ->
      [ executable; "--session"; provider_session; "--output-format";
        "stream-json" ]
      @ model_args @ unsafe_args @ [ "--prompt"; prompt ]
  | "grok", "turn", "new" ->
      [ executable; "--no-leader"; "--cwd"; cwd; "--output-format";
        "streaming-json"; "--session-id"; session_id ]
      @ context_args @ model_args @ unsafe_args @ [ "-p"; prompt ]
  | "grok", "turn", "resume" ->
      [ executable; "--no-leader"; "--cwd"; cwd; "--output-format";
        "streaming-json"; "--resume"; provider_session ]
      @ context_args @ model_args @ unsafe_args @ [ "-p"; prompt ]
  | "opencode", "turn", "new" ->
      [ executable; "run"; "--format"; "json"; "--dir"; cwd ]
      @ context_args @ model_args @ unsafe_args @ [ prompt ]
  | "opencode", "turn", "resume" ->
      [ executable; "run"; "--format"; "json"; "--dir"; cwd; "--session";
        provider_session ]
      @ context_args @ model_args @ unsafe_args @ [ prompt ]
  | "codex", "persistent", "new" ->
      [ executable; "--no-alt-screen"; "-C"; cwd ]
      @ model_args @ unsafe_args @ [ prompt ]
  | "claude", "persistent", "new" ->
      [ executable; "--session-id"; session_id;
        "--setting-sources"; "user,local" ]
      @ model_args @ unsafe_args
  | "claude", "persistent", "resume" ->
      [ executable; "--resume"; provider_session;
        "--setting-sources"; "user,local" ]
      @ model_args @ unsafe_args
  | "kimi", "persistent", "new" ->
      [ executable ] @ model_args @ unsafe_args
  | _, "persistent", _ ->
      failf "persistent-provider-unavailable:%s:%s" spec.provider_id mode
  | _, _, _ ->
      failf "unsupported-provider-mode:%s:%s:%s" spec.provider_id lifecycle mode

let provider_plan cli default_lifecycle =
  let spec = required cli "--provider" |> provider_spec in
  let executable =
    match provider_executable spec with
    | Some path -> path
    | None -> failf "provider-executable-not-found:%s" spec.provider_id
  in
  let mode = Option.value ~default:"new" (optional cli "--mode") in
  if mode <> "new" && mode <> "resume" then failf "invalid-provider-mode:%s" mode;
  let lifecycle =
    Option.value ~default:default_lifecycle (optional cli "--lifecycle")
  in
  if lifecycle <> "turn" && lifecycle <> "persistent" then
    failf "invalid-provider-lifecycle:%s" lifecycle;
  let cwd = cwd_option cli in
  let session_id = required cli "--session-id" in
  let provider_session = Option.value ~default:"" (optional cli "--provider-session") in
  if mode = "resume" && provider_session = "" then
    failf "provider-session-is-required-for-resume";
  if mode = "resume" && spec.provider_session_binding = "caller"
     && not (provider_uuid provider_session)
  then failf "provider-session-must-be-uuid:%s" spec.provider_id;
  if mode = "new" && spec.provider_session_binding = "caller"
     && not (provider_uuid session_id)
  then failf "provider-session-id-must-be-uuid:%s" spec.provider_id;
  let model = Option.value ~default:"" (optional cli "--model") in
  let unsafe_auto = flag cli "--unsafe-auto" in
  let context_isolation = flag cli "--isolate-context" in
  let prompt = provider_prompt cli in
  let argv =
    provider_argv spec lifecycle executable mode cwd session_id provider_session
      model unsafe_auto context_isolation prompt
  in
  let prompt_transport =
    if lifecycle = "persistent"
       && (spec.provider_id = "claude" || spec.provider_id = "kimi")
    then "loom-wake"
    else "argv"
  in
  { plan_spec = spec; plan_lifecycle = lifecycle;
    plan_stdin_authority =
      (if lifecycle = "persistent" then "loom-lease" else "closed");
    plan_mode = mode; plan_executable = executable;
    plan_cwd = cwd; plan_session_id = session_id;
    plan_provider_session = provider_session; plan_model = model;
    plan_unsafe_auto = unsafe_auto; plan_context_isolation = context_isolation;
    plan_prompt_transport = prompt_transport;
    plan_prompt = prompt; plan_argv = argv }

let redacted_provider_argv plan =
  if plan.plan_prompt_transport = "loom-wake" then plan.plan_argv
  else
    let rec redact = function
      | [] -> []
      | [ _prompt ] ->
          [ Printf.sprintf "<PROMPT sha256=%s bytes=%d>" (sha256 plan.plan_prompt)
              (String.length plan.plan_prompt) ]
      | head :: tail -> head :: redact tail
    in
    redact plan.plan_argv

let provider_plan_json plan =
  Printf.sprintf
    "{\"schema\":%s,\"provider\":%s,\"lifecycle\":%s,\"stdin_authority\":%s,\"prompt_transport\":%s,\"mode\":%s,\"executable\":%s,\"stream\":%s,\"credential_authority\":\"native\",\"session_binding\":%s,\"loom_session\":%s,\"provider_session\":%s,\"cwd\":%s,\"model\":%s,\"unsafe_auto\":%s,\"context_isolation\":%s,\"prompt_bytes\":%d,\"prompt_sha256\":%s,\"argv_sha256\":%s,\"argv\":%s}"
    (json_quote provider_abi_schema) (json_quote plan.plan_spec.provider_id)
    (json_quote plan.plan_lifecycle) (json_quote plan.plan_stdin_authority)
    (json_quote plan.plan_prompt_transport)
    (json_quote plan.plan_mode) (json_quote plan.plan_executable)
    (json_quote plan.plan_spec.provider_stream)
    (json_quote plan.plan_spec.provider_session_binding)
    (json_quote plan.plan_session_id) (json_quote plan.plan_provider_session)
    (json_quote plan.plan_cwd) (json_quote plan.plan_model)
    (if plan.plan_unsafe_auto then "true" else "false")
    (if plan.plan_context_isolation then "true" else "false")
    (String.length plan.plan_prompt) (json_quote (sha256 plan.plan_prompt))
    (json_quote (command_argv_digest (Array.of_list plan.plan_argv)))
    (json_string_list (redacted_provider_argv plan))

let provider_plan_command cli =
  let plan = provider_plan cli "turn" in
  if flag cli "--json" then Printf.printf "%s\n%!" (provider_plan_json plan)
  else
    Printf.printf
      "LOOM_PROVIDER_PLAN schema=%s provider=%s lifecycle=%s stdin_authority=%s prompt_transport=%s mode=%s stream=%s credential_authority=native session_binding=%s prompt_bytes=%d prompt_sha256=%s argv_sha256=%s unsafe_auto=%s context_isolation=%s\n%!"
      provider_abi_schema plan.plan_spec.provider_id plan.plan_lifecycle
      plan.plan_stdin_authority plan.plan_prompt_transport plan.plan_mode
      plan.plan_spec.provider_stream plan.plan_spec.provider_session_binding
      (String.length plan.plan_prompt) (sha256 plan.plan_prompt)
      (command_argv_digest (Array.of_list plan.plan_argv))
      (if plan.plan_unsafe_auto then "true" else "false")
      (if plan.plan_context_isolation then "true" else "false")

let provider_start_command cli =
  let plan = provider_plan cli "turn" in
  if plan.plan_lifecycle <> "turn" then
    failf "provider-start-requires-turn-lifecycle";
  let runtime = Unix.realpath Sys.executable_name in
  let start_cli =
    { options = Hashtbl.copy cli.options; flags = Hashtbl.create 2;
      rest = runtime :: "_provider-exec" :: plan.plan_argv }
  in
  let ready_key = "SOUNIO_LOOM_PROVIDER_START_READY_PATH" in
  let ready_paths =
    session_paths (root_option cli plan.plan_cwd) (required cli "--agent")
      (required cli "--lane")
  in
  let ready_path = Filename.concat ready_paths.session_dir "provider-start.ready" in
  mkdir_p ready_paths.session_dir;
  (try Unix.unlink ready_path with Unix_error (ENOENT, _, _) -> ());
  let previous_ready_path = Sys.getenv_opt ready_key in
  Unix.putenv ready_key ready_path;
  Fun.protect
    ~finally:(fun () ->
      Unix.putenv ready_key (Option.value ~default:"" previous_ready_path))
    (fun () ->
      start_command ~launch_source:"provider-start"
        ~ready_timeout:(start_ready_timeout ())
        start_cli);
  atomic_write ready_path "ready\n";
  Printf.printf
    "LOOM_PROVIDER_STARTED schema=%s provider=%s stream=%s session_binding=%s prompt_sha256=%s argv_sha256=%s unsafe_auto=%s context_isolation=%s\n%!"
    provider_abi_schema plan.plan_spec.provider_id plan.plan_spec.provider_stream
    plan.plan_spec.provider_session_binding (sha256 plan.plan_prompt)
    (command_argv_digest (Array.of_list plan.plan_argv))
    (if plan.plan_unsafe_auto then "true" else "false")
    (if plan.plan_context_isolation then "true" else "false");
  if flag cli "--wait" then (
    let _, paths = session_locator cli in
    let descriptor_state () =
      table_value (parse_key_values paths.descriptor_path) "state"
    in
    let replay_terminal () =
      Hashtbl.replace cli.options "--cursor" "0";
      snapshot_command cli
    in
    if descriptor_state () = "exited" then replay_terminal ()
    else
      (try stream_command cli false
       with error ->
         if descriptor_state () = "exited" then replay_terminal ()
         else raise error);
    let deadline = Unix.gettimeofday () +. 30.0 in
    let rec await_terminal_state () =
      let values = parse_key_values paths.descriptor_path in
      match table_value values "state" with
      | "exited" -> ()
      | "active" when Unix.gettimeofday () < deadline ->
          Unix.sleepf 0.01;
          await_terminal_state ()
      | "active" -> failf "provider-wait-terminal-state-timeout"
      | state -> failf "provider-wait-invalid-terminal-state:%s" state
    in
    await_terminal_state ())

let provider_open_command cli =
  let plan = provider_plan cli "persistent" in
  if plan.plan_lifecycle <> "persistent" then
    failf "provider-open-requires-persistent-lifecycle";
  if plan.plan_prompt_transport = "loom-wake" && plan.plan_prompt <> ""
     && List.exists
          (fun argument -> string_contains argument plan.plan_prompt)
          plan.plan_argv
  then failf "provider-loom-wake-argv-contains-prompt";
  let runtime = Unix.realpath Sys.executable_name in
  let start_cli =
    { options = Hashtbl.copy cli.options; flags = Hashtbl.create 2;
      rest = runtime :: "_provider-tui" :: plan.plan_argv }
  in
  start_command ~launch_source:"provider-open" start_cli;
  if plan.plan_prompt_transport = "loom-wake" then (
    let wake_cli =
      { options = Hashtbl.copy cli.options; flags = Hashtbl.create 2; rest = [] }
    in
    let bootstrap_digest =
      sha256
        (plan.plan_spec.provider_id ^ "\000" ^ plan.plan_session_id ^ "\000"
       ^ plan.plan_prompt)
    in
    Hashtbl.replace wake_cli.options "--message-id"
      ("provider-bootstrap-" ^ String.sub bootstrap_digest 0 16);
    Hashtbl.replace wake_cli.options "--prompt" plan.plan_prompt;
    (try wake_command wake_cli
     with error ->
       (try stop_command wake_cli with _ -> ());
       raise error));
  Printf.printf
    "LOOM_PROVIDER_OPENED schema=%s provider=%s lifecycle=persistent stdin_authority=loom-lease prompt_transport=%s session_binding=%s prompt_sha256=%s argv_sha256=%s unsafe_auto=%s context_isolation=false\n%!"
    provider_abi_schema plan.plan_spec.provider_id
    plan.plan_prompt_transport
    plan.plan_spec.provider_session_binding (sha256 plan.plan_prompt)
    (command_argv_digest (Array.of_list plan.plan_argv))
    (if plan.plan_unsafe_auto then "true" else "false")

let provider_auth_login_command cli =
  let spec = required cli "--provider" |> provider_spec in
  let executable =
    match provider_executable spec with
    | Some path -> path
    | None -> failf "provider-executable-not-found:%s" spec.provider_id
  in
  Printf.printf
    "LOOM_PROVIDER_AUTH_DELEGATE schema=%s provider=%s credential_authority=native\n%!"
    provider_abi_schema spec.provider_id;
  Unix.execv executable (Array.of_list (executable :: spec.provider_login_args))

let provider_clean_environment () =
  let harness_keys =
    [ "CODEX_SESSION_ID"; "CODEX_THREAD_ID"; "CODEX_CI"; "CLAUDECODE";
      "CLAUDE_CODE_ENTRYPOINT"; "CLAUDE_CODE_SESSION_ID";
      "KIMI_SESSION_ID"; "KIMI_CLI_SESSION_ID";
      "CURSOR_SESSION_ID"; "CURSOR_AGENT_SESSION_ID"; "GROK_SESSION_ID";
      "SOUNIO_AGENT_ID"; "SOUNIO_LANE_ID"; "TMUX"; "TMUX_PANE";
      "TMUX_TMPDIR" ]
  in
  let harness_prefixes = [ "SOUNIO_AGENTD_" ] in
  Unix.environment () |> Array.to_list
  |> List.filter (fun entry ->
         not
           (List.exists (fun key -> starts_with entry (key ^ "=")) harness_keys
           || List.exists (fun prefix -> starts_with entry prefix) harness_prefixes))
  |> Array.of_list

let provider_exec_command arguments =
  match arguments with
  | executable :: tail ->
      let null = Unix.openfile "/dev/null" [ O_RDONLY ] 0 in
      Unix.dup2 null Unix.stdin;
      if null <> Unix.stdin then Unix.close null;
      Unix.execve executable (Array.of_list (executable :: tail))
        (provider_clean_environment ())
  | [] -> failf "provider-exec-requires-an-executable"

let provider_tui_command arguments =
  match arguments with
  | executable :: tail ->
      Unix.execve executable (Array.of_list (executable :: tail))
        (provider_clean_environment ())
  | [] -> failf "provider-tui-requires-an-executable"

type obligation_paths = {
  obligation_dir : string;
  obligation_journal_path : string;
  obligation_lock_path : string;
}

type obligation_view = {
  obligation_message_id : string;
  obligation_message_digest : string;
  obligation_from_agent : string;
  obligation_from_lane : string;
  obligation_to_agent : string;
  obligation_to_lane : string;
  obligation_state : int;
  obligation_actor : string;
  obligation_lane : string;
  obligation_generation : string;
  obligation_claim : string;
  obligation_predecessor_claim : string;
  obligation_lease_deadline : int;
  obligation_outcome_digest : string;
  obligation_evidence_digest : string;
  obligation_outcome_path : string;
  obligation_evidence_path : string;
  obligation_last_epoch : int;
  obligation_sequence : int;
  obligation_head : string;
}

let obligation_state_name = function
  | 1 -> "durable"
  | 2 -> "consumed"
  | 3 -> "claimed"
  | 4 -> "interrupted"
  | 5 -> "recoverable"
  | 6 -> "completed"
  | value -> failf "obligation-invalid-state:%d" value

let obligation_root root = Filename.concat root "loom-obligations"

let obligation_paths root message_id =
  if trim message_id = "" then failf "obligation-message-id-empty";
  let key = slug message_id ^ "-" ^ String.sub (sha256 message_id) 0 16 in
  let obligation_dir = Filename.concat (obligation_root root) key in
  { obligation_dir;
    obligation_journal_path = Filename.concat obligation_dir "journal.tsv";
    obligation_lock_path = Filename.concat obligation_dir "obligation.lock" }

let with_obligation_lock paths operation =
  mkdir_p paths.obligation_dir;
  Unix.chmod paths.obligation_dir 0o700;
  let descriptor =
    Unix.openfile paths.obligation_lock_path [ O_WRONLY; O_CREAT ] 0o600
  in
  Unix.set_close_on_exec descriptor;
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      operation ())

let obligation_payload fields =
  fields
  |> List.map (fun (key, value) -> key ^ "=" ^ field_escape value ^ "\n")
  |> String.concat ""

let obligation_payload_table payload =
  let table = Hashtbl.create 24 in
  payload |> split_on '\n'
  |> List.iter (fun line ->
         if line <> "" then
           match String.index_opt line '=' with
           | None -> failf "obligation-payload-field-malformed"
           | Some index ->
               let key = String.sub line 0 index in
               let value =
                 String.sub line (index + 1) (String.length line - index - 1)
                 |> field_unescape
               in
               if key = "" || Hashtbl.mem table key then
                 failf "obligation-payload-field-duplicate:%s" key;
               Hashtbl.add table key value);
  table

let obligation_field table key =
  match Hashtbl.find_opt table key with
  | Some value -> value
  | None -> failf "obligation-payload-field-missing:%s" key

let obligation_int label value =
  try int_of_string value
  with _ -> failf "obligation-%s-not-integer:%s" label value

let obligation_positive_int label value =
  let parsed = obligation_int label value in
  if parsed <= 0 then failf "obligation-%s-not-positive:%d" label parsed;
  parsed

let obligation_event_epoch table =
  obligation_positive_int "event-epoch" (obligation_field table "event_epoch")

let obligation_require_nonempty label value =
  if trim value = "" then failf "obligation-%s-empty" label;
  value

let obligation_owner_matches view actor lane generation =
  actor = view.obligation_actor && lane = view.obligation_lane
  && generation = view.obligation_generation

let obligation_identity_matches view table =
  let message_id = obligation_field table "message_id" in
  let message_digest = obligation_field table "message_digest" in
  if message_id <> view.obligation_message_id then
    failf "obligation-message-id-drift";
  if message_digest <> view.obligation_message_digest then
    failf "obligation-message-digest-drift"

let verify_obligation_hash_chain path events =
  let expected_sequence = ref 1 in
  let expected_previous = ref (String.make 64 '0') in
  List.iter
    (fun (event : journal_event) ->
      verify_journal_event_authority path event;
      if event.seq <> !expected_sequence then
        failf "obligation-hash-non-contiguous-sequence:expected=%d:actual=%d"
          !expected_sequence event.seq;
      if event.previous <> !expected_previous then
        failf "obligation-hash-previous-mismatch:seq=%d" event.seq;
      let expected_hash =
        sha256
          (event_material event.seq event.previous event.utc event.kind
             event.payload_hex)
      in
      if event.hash <> expected_hash then
        failf "obligation-hash-event-mismatch:seq=%d" event.seq;
      expected_previous := event.hash;
      incr expected_sequence)
    events;
  !expected_previous

let reduce_obligation_event current (event : journal_event) =
  let table = obligation_payload_table (string_of_hex event.payload_hex) in
  let epoch = obligation_event_epoch table in
  match (current, event.kind) with
  | None, "OBLIGATION_OPENED" ->
      let message_id =
        obligation_require_nonempty "message-id" (obligation_field table "message_id")
      in
      let message_digest = obligation_field table "message_digest" in
      if not (valid_sha256 message_digest) then
        failf "obligation-message-digest-invalid";
      Some
        { obligation_message_id = message_id;
          obligation_message_digest = message_digest;
          obligation_from_agent = obligation_field table "from_agent";
          obligation_from_lane = obligation_field table "from_lane";
          obligation_to_agent = obligation_field table "to_agent";
          obligation_to_lane = obligation_field table "to_lane";
          obligation_state = 1; obligation_actor = ""; obligation_lane = "";
          obligation_generation = ""; obligation_claim = "";
          obligation_predecessor_claim = ""; obligation_lease_deadline = 0;
          obligation_outcome_digest = ""; obligation_evidence_digest = "";
          obligation_outcome_path = ""; obligation_evidence_path = "";
          obligation_last_epoch = epoch; obligation_sequence = event.seq;
          obligation_head = event.hash }
  | None, _ -> failf "obligation-first-event-must-open"
  | Some view, _ ->
      obligation_identity_matches view table;
      if epoch < view.obligation_last_epoch then
        failf "obligation-event-time-regressed:seq=%d" event.seq;
      let updated state actor lane generation claim predecessor deadline outcome
          evidence outcome_path evidence_path =
        Some
          { view with obligation_state = state; obligation_actor = actor;
            obligation_lane = lane; obligation_generation = generation;
            obligation_claim = claim;
            obligation_predecessor_claim = predecessor;
            obligation_lease_deadline = deadline;
            obligation_outcome_digest = outcome;
            obligation_evidence_digest = evidence;
            obligation_outcome_path = outcome_path;
            obligation_evidence_path = evidence_path;
            obligation_last_epoch = epoch; obligation_sequence = event.seq;
            obligation_head = event.hash }
      in
      (match event.kind with
      | "OBLIGATION_CONSUMED" ->
          if view.obligation_state <> 1 then
            failf "obligation-consume-invalid-state:%s"
              (obligation_state_name view.obligation_state);
          let actor =
            obligation_require_nonempty "actor" (obligation_field table "actor")
          in
          let lane =
            obligation_require_nonempty "lane" (obligation_field table "lane")
          in
          let generation =
            obligation_require_nonempty "generation"
              (obligation_field table "generation")
          in
          let deadline =
            obligation_positive_int "lease-deadline"
              (obligation_field table "lease_deadline")
          in
          if deadline <= epoch then failf "obligation-consume-already-expired";
          updated 2 actor lane generation "" "" deadline "" "" "" ""
      | "OBLIGATION_CLAIMED" ->
          if view.obligation_state <> 2 && view.obligation_state <> 5 then
            failf "obligation-claim-invalid-state:%s"
              (obligation_state_name view.obligation_state);
          let actor = obligation_field table "actor" in
          let lane = obligation_field table "lane" in
          let generation = obligation_field table "generation" in
          if not (obligation_owner_matches view actor lane generation) then
            failf "obligation-claim-owner-mismatch";
          if view.obligation_state = 2 && epoch > view.obligation_lease_deadline then
            failf "obligation-consumer-lease-expired";
          let claim =
            obligation_require_nonempty "claim" (obligation_field table "claim")
          in
          if claim = view.obligation_predecessor_claim then
            failf "obligation-claim-reuses-predecessor";
          let deadline =
            obligation_positive_int "lease-deadline"
              (obligation_field table "lease_deadline")
          in
          if deadline <= epoch then failf "obligation-claim-already-expired";
          updated 3 actor lane generation claim view.obligation_predecessor_claim
            deadline "" "" "" ""
      | "OBLIGATION_RENEWED" ->
          if view.obligation_state <> 3 then
            failf "obligation-renew-invalid-state:%s"
              (obligation_state_name view.obligation_state);
          let actor = obligation_field table "actor" in
          let lane = obligation_field table "lane" in
          let generation = obligation_field table "generation" in
          let claim = obligation_field table "claim" in
          if not (obligation_owner_matches view actor lane generation)
             || claim <> view.obligation_claim then
            failf "obligation-renew-claim-mismatch";
          if epoch > view.obligation_lease_deadline then
            failf "obligation-renew-after-expiry";
          let deadline =
            obligation_positive_int "lease-deadline"
              (obligation_field table "lease_deadline")
          in
          if deadline <= epoch || deadline <= view.obligation_lease_deadline then
            failf "obligation-renew-not-monotone";
          updated 3 actor lane generation claim view.obligation_predecessor_claim
            deadline "" "" "" ""
      | "OBLIGATION_INTERRUPTED" ->
          if view.obligation_state <> 2 && view.obligation_state <> 3 then
            failf "obligation-interrupt-invalid-state:%s"
              (obligation_state_name view.obligation_state);
          let actor = obligation_field table "actor" in
          let lane = obligation_field table "lane" in
          let generation = obligation_field table "generation" in
          let claim = obligation_field table "claim" in
          if not (obligation_owner_matches view actor lane generation)
             || claim <> view.obligation_claim then
            failf "obligation-interrupt-claim-mismatch";
          let interrupter_actor = obligation_field table "interrupter_actor" in
          let interrupter_lane = obligation_field table "interrupter_lane" in
          let interrupter_generation =
            obligation_field table "interrupter_generation"
          in
          let self_interrupt =
            obligation_owner_matches view interrupter_actor interrupter_lane
              interrupter_generation
          in
          if (not self_interrupt) && epoch <= view.obligation_lease_deadline then
            failf "obligation-live-claim-cannot-be-interrupted";
          let reason_digest = obligation_field table "reason_digest" in
          if not (valid_sha256 reason_digest) then
            failf "obligation-interrupt-reason-digest-invalid";
          updated 4 actor lane generation claim claim view.obligation_lease_deadline
            "" "" "" ""
      | "OBLIGATION_RECOVERED" ->
          if view.obligation_state <> 4 then
            failf "obligation-recover-invalid-state:%s"
              (obligation_state_name view.obligation_state);
          let actor =
            obligation_require_nonempty "actor" (obligation_field table "actor")
          in
          let lane =
            obligation_require_nonempty "lane" (obligation_field table "lane")
          in
          let generation =
            obligation_require_nonempty "generation"
              (obligation_field table "generation")
          in
          if obligation_owner_matches view actor lane generation then
            failf "obligation-recovery-must-change-owner-generation";
          let predecessor = obligation_field table "predecessor_claim" in
          if predecessor <> view.obligation_claim then
            failf "obligation-recovery-predecessor-mismatch";
          updated 5 actor lane generation "" predecessor 0 "" "" "" ""
      | "OBLIGATION_COMPLETED" ->
          if view.obligation_state <> 3 then
            failf "obligation-complete-invalid-state:%s"
              (obligation_state_name view.obligation_state);
          let actor = obligation_field table "actor" in
          let lane = obligation_field table "lane" in
          let generation = obligation_field table "generation" in
          let claim = obligation_field table "claim" in
          if not (obligation_owner_matches view actor lane generation)
             || claim <> view.obligation_claim then
            failf "obligation-complete-claim-mismatch";
          if epoch > view.obligation_lease_deadline then
            failf "obligation-complete-after-lease-expiry";
          let outcome = obligation_field table "outcome_digest" in
          let evidence = obligation_field table "evidence_digest" in
          if not (valid_sha256 outcome) || not (valid_sha256 evidence)
             || outcome = String.make 64 '0' || evidence = String.make 64 '0'
             || outcome = evidence then
            failf "obligation-completion-evidence-not-bound";
          let outcome_path = obligation_field table "outcome_path" in
          let evidence_path = obligation_field table "evidence_path" in
          if outcome_path = evidence_path then
            failf "obligation-completion-artifacts-not-distinct";
          updated 6 actor lane generation claim view.obligation_predecessor_claim
            view.obligation_lease_deadline outcome evidence outcome_path
            evidence_path
      | "OBLIGATION_OPENED" -> failf "obligation-open-duplicate"
      | kind -> failf "obligation-event-unknown:%s" kind)

let load_obligation_journal path =
  if not (Sys.file_exists path) then failf "obligation-journal-missing:%s" path;
  let events =
    read_lines path |> List.filter (fun line -> trim line <> "")
    |> List.map parse_event
  in
  if events = [] then failf "obligation-journal-empty:%s" path;
  let head = verify_obligation_hash_chain path events in
  let view = List.fold_left reduce_obligation_event None events in
  match view with
  | None -> failf "obligation-journal-has-no-state"
  | Some value ->
      if value.obligation_head <> head then failf "obligation-reducer-head-drift";
      (events, value)

let resume_obligation_journal path events view =
  let descriptor = Unix.openfile path [ O_WRONLY; O_APPEND ] 0o600 in
  Unix.set_close_on_exec descriptor;
  let channel = Unix.out_channel_of_descr descriptor in
  { channel; descriptor; seq = List.length events;
    previous = view.obligation_head;
    authority_context = journal_authority_context path;
    authority_directory = Filename.dirname path }

let obligation_event_fields view epoch fields =
  [ ("message_id", view.obligation_message_id);
    ("message_digest", view.obligation_message_digest);
    ("event_epoch", string_of_int epoch) ]
  @ fields

let obligation_adapter () =
  match Sys.getenv_opt "SOUNIO_LOOM_OBLIGATION_ADAPTER" with
  | Some path when path <> "" -> path
  | _ ->
      Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
        "sounio-loom-obligation-runtime"

let run_captured_input executable arguments input =
  let stdin_reader, stdin_writer = Unix.pipe () in
  let output_reader, output_writer = Unix.pipe () in
  Unix.set_close_on_exec stdin_writer;
  Unix.set_close_on_exec output_reader;
  let pid =
    Unix.create_process executable (Array.of_list (executable :: arguments))
      stdin_reader output_writer output_writer
  in
  Unix.close stdin_reader;
  Unix.close output_writer;
  write_all stdin_writer input;
  Unix.close stdin_writer;
  let output = Buffer.create 512 in
  let bytes = Bytes.create 4096 in
  let rec drain () =
    match Unix.read output_reader bytes 0 (Bytes.length bytes) with
    | 0 -> ()
    | count -> Buffer.add_subbytes output bytes 0 count; drain ()
    | exception Unix_error (EINTR, _, _) -> drain ()
  in
  Fun.protect ~finally:(fun () -> Unix.close output_reader) drain;
  let _, status = Unix.waitpid [] pid in
  { captured_code = process_exit_code status;
    captured_output = trim (Buffer.contents output) }

let run_captured_input_timeout ~timeout_seconds executable arguments input =
  let stdin_reader, stdin_writer = Unix.pipe () in
  let output_reader, output_writer = Unix.pipe () in
  Unix.set_close_on_exec stdin_writer;
  Unix.set_close_on_exec output_reader;
  let pid =
    Unix.create_process executable (Array.of_list (executable :: arguments))
      stdin_reader output_writer output_writer
  in
  Unix.close stdin_reader;
  Unix.close output_writer;
  let status = ref None in
  let close descriptor = try Unix.close descriptor with _ -> () in
  let reap () =
    match !status with
    | Some _ -> ()
    | None ->
        (try Unix.kill pid Sys.sigkill with _ -> ());
        (try
           let _, observed = Unix.waitpid [] pid in
           status := Some observed
         with _ -> ())
  in
  Fun.protect
    ~finally:(fun () -> close stdin_writer; close output_reader; reap ())
    (fun () ->
      write_all stdin_writer input;
      close stdin_writer;
      let output = Buffer.create 512 in
      let bytes = Bytes.create 4096 in
      let deadline = Unix.gettimeofday () +. timeout_seconds in
      let eof = ref false in
      while not (!eof && Option.is_some !status) do
        let remaining = deadline -. Unix.gettimeofday () in
        if remaining <= 0. then
          failf "captured-input-process-timeout:%s" executable;
        (match Unix.waitpid [ WNOHANG ] pid with
        | 0, _ -> ()
        | _, observed -> status := Some observed);
        if not !eof then (
          let ready, _, _ =
            Unix.select [ output_reader ] [] [] (min remaining 0.05)
          in
          if ready <> [] then
            match Unix.read output_reader bytes 0 (Bytes.length bytes) with
            | 0 -> eof := true
            | count -> Buffer.add_subbytes output bytes 0 count
            | exception Unix_error (EINTR, _, _) -> ())
        else if Option.is_none !status then
          ignore (Unix.select [] [] [] (min remaining 0.01))
      done;
      let observed = Option.get !status in
      { captured_code = process_exit_code observed;
        captured_output = trim (Buffer.contents output) })

let obligation_zero_digest = String.make 64 '0'

let verify_obligation_native_transition transition previous next view actor lane
    generation claim deadline outcome evidence =
  let adapter = obligation_adapter () in
  if not (Sys.file_exists adapter) then
    failf "sounio-obligation-adapter-missing:%s" adapter;
  let adapter = Unix.realpath adapter in
  let transition_code =
    match transition with
    | "open" -> 1 | "consume" -> 2 | "claim" -> 3 | "renew" -> 4
    | "interrupt" -> 5 | "recover" -> 6 | "complete" -> 7
    | _ -> failf "obligation-transition-unknown:%s" transition
  in
  let token domain value =
    if value = "" then "0" else sounio_continuity_token domain value
  in
  let frame =
    let actor_token =
      if actor = "" && lane = "" then "0"
      else token "loom-obligation-actor" (actor ^ "\000" ^ lane)
    in
    [ "9007"; string_of_int transition_code; string_of_int previous;
      string_of_int next;
      token "loom-obligation-id" view.obligation_message_id;
      actor_token;
      token "loom-obligation-generation" generation;
      token "loom-obligation-claim" claim; string_of_int deadline ]
    @ digest256_limbs view.obligation_message_digest
    @ digest256_limbs outcome @ digest256_limbs evidence
    |> String.concat " "
  in
  let result = run_captured_input adapter [] (frame ^ "\n") in
  let expected =
    Printf.sprintf
      "SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=%s state=%d"
      transition next
  in
  if result.captured_code <> 0 || result.captured_output <> expected then
    failf "sounio-obligation-transition-refused:%s:rc=%d:output=%s:frame=%s"
      transition result.captured_code result.captured_output frame

let obligation_now_epoch () = int_of_float (Unix.gettimeofday ())

let obligation_ttl cli =
  let ttl =
    match optional cli "--ttl-seconds" with
    | None -> 1800
    | Some value -> obligation_positive_int "ttl-seconds" value
  in
  if ttl > 86400 then failf "obligation-ttl-seconds-too-large:%d" ttl;
  ttl

let obligation_regular_file label path =
  if path = "" || not (Sys.file_exists path) then
    failf "obligation-%s-missing:%s" label path;
  let resolved = Unix.realpath path in
  let stat = Unix.stat resolved in
  if stat.st_kind <> S_REG then failf "obligation-%s-not-regular:%s" label resolved;
  if stat.st_size <= 0 then failf "obligation-%s-empty:%s" label resolved;
  resolved

let obligation_initial_view message_id message_digest from_agent from_lane
    to_agent to_lane epoch =
  { obligation_message_id = message_id;
    obligation_message_digest = message_digest;
    obligation_from_agent = from_agent; obligation_from_lane = from_lane;
    obligation_to_agent = to_agent; obligation_to_lane = to_lane;
    obligation_state = 0; obligation_actor = ""; obligation_lane = "";
    obligation_generation = ""; obligation_claim = "";
    obligation_predecessor_claim = ""; obligation_lease_deadline = 0;
    obligation_outcome_digest = ""; obligation_evidence_digest = "";
    obligation_outcome_path = ""; obligation_evidence_path = "";
    obligation_last_epoch = epoch; obligation_sequence = 0;
    obligation_head = String.make 64 '0' }

let append_obligation_transition paths events view kind fields =
  let journal = resume_obligation_journal paths.obligation_journal_path events view in
  Fun.protect
    ~finally:(fun () -> close_out_noerr journal.channel)
    (fun () -> ignore (append_event journal kind (obligation_payload fields)));
  snd (load_obligation_journal paths.obligation_journal_path)

let obligation_lease_state view now =
  if view.obligation_state <> 2 && view.obligation_state <> 3 then "none"
  else if now <= view.obligation_lease_deadline then "active"
  else "expired"

let print_obligation ?(prefix = "LOOM_OBLIGATION") view =
  let now = obligation_now_epoch () in
  Printf.printf
    "%s message=%s state=%s state_code=%d unclosed=%s actor=%s lane=%s generation=%s claim=%s predecessor_claim=%s lease_deadline=%d lease=%s outcome_digest=%s evidence_digest=%s outcome_path=%s evidence_path=%s seq=%d head=%s\n%!"
    prefix (field_escape view.obligation_message_id)
    (obligation_state_name view.obligation_state) view.obligation_state
    (if view.obligation_state = 6 then "no" else "yes")
    (field_escape view.obligation_actor) (field_escape view.obligation_lane)
    (field_escape view.obligation_generation) (field_escape view.obligation_claim)
    (field_escape view.obligation_predecessor_claim)
    view.obligation_lease_deadline (obligation_lease_state view now)
    view.obligation_outcome_digest view.obligation_evidence_digest
    (field_escape view.obligation_outcome_path)
    (field_escape view.obligation_evidence_path)
    view.obligation_sequence view.obligation_head

let obligation_json view =
  let now = obligation_now_epoch () in
  Printf.sprintf
    "{\"messageId\":%s,\"state\":%s,\"stateCode\":%d,\"unclosed\":%s,\"actor\":%s,\"lane\":%s,\"generation\":%s,\"claim\":%s,\"predecessorClaim\":%s,\"leaseDeadline\":%d,\"lease\":%s,\"outcomeDigest\":%s,\"evidenceDigest\":%s,\"outcomePath\":%s,\"evidencePath\":%s,\"sequence\":%d,\"head\":%s}"
    (json_quote view.obligation_message_id)
    (json_quote (obligation_state_name view.obligation_state))
    view.obligation_state
    (if view.obligation_state = 6 then "false" else "true")
    (json_quote view.obligation_actor) (json_quote view.obligation_lane)
    (json_quote view.obligation_generation) (json_quote view.obligation_claim)
    (json_quote view.obligation_predecessor_claim)
    view.obligation_lease_deadline
    (json_quote (obligation_lease_state view now))
    (json_quote view.obligation_outcome_digest)
    (json_quote view.obligation_evidence_digest)
    (json_quote view.obligation_outcome_path)
    (json_quote view.obligation_evidence_path)
    view.obligation_sequence (json_quote view.obligation_head)

let obligation_views root =
  let directory = obligation_root root in
  if not (Sys.file_exists directory) then []
  else
    Sys.readdir directory |> Array.to_list |> List.sort String.compare
    |> List.filter_map (fun name ->
           let path = Filename.concat (Filename.concat directory name) "journal.tsv" in
           if Sys.file_exists path then Some (snd (load_obligation_journal path))
           else None)
    |> List.sort (fun left right ->
           String.compare left.obligation_message_id right.obligation_message_id)

let () =
  authority_obligation_enricher :=
    (fun root lanes ->
      try
        Hashtbl.iter
          (fun _ lane -> lane.authority_obligation_census_complete <- true)
          lanes;
        obligation_views root
        |> List.iter (fun view ->
               let actor, lane =
                 if view.obligation_actor <> "" && view.obligation_lane <> "" then
                   (view.obligation_actor, view.obligation_lane)
                 else (view.obligation_to_agent, view.obligation_to_lane)
               in
               if actor <> "" && lane <> "" then (
                 let entry = authority_entry lanes actor lane in
                 entry.authority_obligation_census_complete <- true;
                 if view.obligation_state = 1 then
                   entry.authority_pending_obligations <-
                     entry.authority_pending_obligations + 1
                 else if
                   view.obligation_state = 2 || view.obligation_state = 3
                   || view.obligation_state = 4 || view.obligation_state = 5
                 then (
                   entry.authority_active_obligations <-
                     entry.authority_active_obligations + 1;
                   if view.obligation_state = 4 then
                     entry.authority_blocker_active <- true)));
        true
      with _ -> false)

let obligation_open_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let message_id = required cli "--message" in
  let message_digest = required cli "--message-digest" in
  if not (valid_sha256 message_digest) || message_digest = obligation_zero_digest then
    failf "obligation-message-digest-invalid";
  let from_agent = required cli "--from-agent" in
  let from_lane = required cli "--from-lane" in
  let to_agent = required cli "--to-agent" in
  let to_lane = required cli "--to-lane" in
  List.iter
    (fun (label, value) -> ignore (obligation_require_nonempty label value))
    [ ("from-agent", from_agent); ("from-lane", from_lane);
      ("to-agent", to_agent); ("to-lane", to_lane) ];
  let paths = obligation_paths root message_id in
  with_obligation_lock paths (fun () ->
      if Sys.file_exists paths.obligation_journal_path then (
        let _, view = load_obligation_journal paths.obligation_journal_path in
        if view.obligation_message_id <> message_id
           || view.obligation_message_digest <> message_digest
           || view.obligation_from_agent <> from_agent
           || view.obligation_from_lane <> from_lane
           || view.obligation_to_agent <> to_agent
           || view.obligation_to_lane <> to_lane then
          failf "obligation-open-identity-conflict";
        print_obligation ~prefix:"LOOM_OBLIGATION_OPEN idempotent=yes" view)
      else (
        let epoch = obligation_now_epoch () in
        let initial =
          obligation_initial_view message_id message_digest from_agent from_lane
            to_agent to_lane epoch
        in
        verify_obligation_native_transition "open" 0 1 initial "" "" "" "" 0
          obligation_zero_digest obligation_zero_digest;
        let journal = open_journal paths.obligation_journal_path in
        Fun.protect
          ~finally:(fun () -> close_out_noerr journal.channel)
          (fun () ->
            ignore
              (append_event journal "OBLIGATION_OPENED"
                 (obligation_payload
                    [ ("message_id", message_id);
                      ("message_digest", message_digest);
                      ("event_epoch", string_of_int epoch);
                      ("from_agent", from_agent); ("from_lane", from_lane);
                      ("to_agent", to_agent); ("to_lane", to_lane) ])));
        fsync_directory paths.obligation_dir;
        let _, view = load_obligation_journal paths.obligation_journal_path in
        print_obligation ~prefix:"LOOM_OBLIGATION_OPEN idempotent=no" view))

let with_existing_obligation cli operation =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let message_id = required cli "--message" in
  let paths = obligation_paths root message_id in
  with_obligation_lock paths (fun () ->
      let events, view = load_obligation_journal paths.obligation_journal_path in
      if view.obligation_message_id <> message_id then
        failf "obligation-message-path-collision";
      operation paths events view)

let obligation_owner_options cli =
  let actor = required cli "--actor" in
  let lane = required cli "--lane" in
  let generation = required cli "--generation" in
  ignore (obligation_require_nonempty "actor" actor);
  ignore (obligation_require_nonempty "lane" lane);
  ignore (obligation_require_nonempty "generation" generation);
  (actor, lane, generation)

let obligation_consume_command cli =
  let actor, lane, generation = obligation_owner_options cli in
  let ttl = obligation_ttl cli in
  with_existing_obligation cli (fun paths events view ->
      if view.obligation_state <> 1 then
        failf "obligation-consume-invalid-state:%s"
          (obligation_state_name view.obligation_state);
      let epoch = obligation_now_epoch () in
      let deadline = epoch + ttl in
      verify_obligation_native_transition "consume" 1 2 view actor lane generation
        "" deadline obligation_zero_digest obligation_zero_digest;
      let updated =
        append_obligation_transition paths events view "OBLIGATION_CONSUMED"
          (obligation_event_fields view epoch
             [ ("actor", actor); ("lane", lane); ("generation", generation);
               ("lease_deadline", string_of_int deadline) ])
      in
      print_obligation ~prefix:"LOOM_OBLIGATION_CONSUMED" updated)

let obligation_claim_command cli =
  let actor, lane, generation = obligation_owner_options cli in
  let claim =
    match optional cli "--claim" with
    | Some value -> obligation_require_nonempty "claim" value
    | None -> "claim-" ^ random_hex 16
  in
  let ttl = obligation_ttl cli in
  with_existing_obligation cli (fun paths events view ->
      if view.obligation_state <> 2 && view.obligation_state <> 5 then
        failf "obligation-claim-invalid-state:%s"
          (obligation_state_name view.obligation_state);
      if not (obligation_owner_matches view actor lane generation) then
        failf "obligation-claim-owner-mismatch";
      let epoch = obligation_now_epoch () in
      if view.obligation_state = 2 && epoch > view.obligation_lease_deadline then
        failf "obligation-consumer-lease-expired";
      if claim = view.obligation_predecessor_claim then
        failf "obligation-claim-reuses-predecessor";
      let deadline = epoch + ttl in
      verify_obligation_native_transition "claim" view.obligation_state 3 view
        actor lane generation claim deadline obligation_zero_digest
        obligation_zero_digest;
      let updated =
        append_obligation_transition paths events view "OBLIGATION_CLAIMED"
          (obligation_event_fields view epoch
             [ ("actor", actor); ("lane", lane); ("generation", generation);
               ("claim", claim); ("lease_deadline", string_of_int deadline) ])
      in
      print_obligation ~prefix:"LOOM_OBLIGATION_CLAIMED" updated)

let obligation_renew_command cli =
  let actor, lane, generation = obligation_owner_options cli in
  let claim = required cli "--claim" in
  let ttl = obligation_ttl cli in
  with_existing_obligation cli (fun paths events view ->
      if view.obligation_state <> 3
         || not (obligation_owner_matches view actor lane generation)
         || claim <> view.obligation_claim then
        failf "obligation-renew-current-claim-required";
      let epoch = obligation_now_epoch () in
      if epoch > view.obligation_lease_deadline then
        failf "obligation-renew-after-expiry";
      let deadline = max (epoch + ttl) (view.obligation_lease_deadline + 1) in
      verify_obligation_native_transition "renew" 3 3 view actor lane generation
        claim deadline obligation_zero_digest obligation_zero_digest;
      let updated =
        append_obligation_transition paths events view "OBLIGATION_RENEWED"
          (obligation_event_fields view epoch
             [ ("actor", actor); ("lane", lane); ("generation", generation);
               ("claim", claim); ("lease_deadline", string_of_int deadline) ])
      in
      print_obligation ~prefix:"LOOM_OBLIGATION_RENEWED" updated)

let obligation_interrupt_command cli =
  let interrupter_actor, interrupter_lane, interrupter_generation =
    obligation_owner_options cli
  in
  let claim = Option.value ~default:"" (optional cli "--claim") in
  let reason = Option.value ~default:"explicit-interruption" (optional cli "--reason") in
  with_existing_obligation cli (fun paths events view ->
      if (view.obligation_state <> 2 && view.obligation_state <> 3)
         || claim <> view.obligation_claim then
        failf "obligation-interrupt-current-claim-required";
      let epoch = obligation_now_epoch () in
      let self_interrupt =
        obligation_owner_matches view interrupter_actor interrupter_lane
          interrupter_generation
      in
      if (not self_interrupt) && epoch <= view.obligation_lease_deadline then
        failf "obligation-live-claim-owned-by-another-generation";
      verify_obligation_native_transition "interrupt" view.obligation_state 4 view
        view.obligation_actor view.obligation_lane view.obligation_generation
        view.obligation_claim view.obligation_lease_deadline
        obligation_zero_digest obligation_zero_digest;
      let updated =
        append_obligation_transition paths events view "OBLIGATION_INTERRUPTED"
          (obligation_event_fields view epoch
             [ ("actor", view.obligation_actor); ("lane", view.obligation_lane);
               ("generation", view.obligation_generation);
               ("claim", view.obligation_claim);
               ("interrupter_actor", interrupter_actor);
               ("interrupter_lane", interrupter_lane);
               ("interrupter_generation", interrupter_generation);
               ("reason_digest", sha256 reason) ])
      in
      print_obligation ~prefix:"LOOM_OBLIGATION_INTERRUPTED" updated)

let obligation_recover_command cli =
  let actor, lane, generation = obligation_owner_options cli in
  with_existing_obligation cli (fun paths events view ->
      if view.obligation_state <> 4 then
        failf "obligation-recover-invalid-state:%s"
          (obligation_state_name view.obligation_state);
      if obligation_owner_matches view actor lane generation then
        failf "obligation-recovery-must-change-owner-generation";
      let epoch = obligation_now_epoch () in
      verify_obligation_native_transition "recover" 4 5 view actor lane generation
        view.obligation_claim 0 obligation_zero_digest obligation_zero_digest;
      let updated =
        append_obligation_transition paths events view "OBLIGATION_RECOVERED"
          (obligation_event_fields view epoch
             [ ("actor", actor); ("lane", lane); ("generation", generation);
               ("predecessor_claim", view.obligation_claim) ])
      in
      print_obligation ~prefix:"LOOM_OBLIGATION_RECOVERED" updated)

let obligation_complete_command cli =
  let actor, lane, generation = obligation_owner_options cli in
  let claim = required cli "--claim" in
  let outcome_path = obligation_regular_file "outcome" (required cli "--outcome") in
  let evidence_path =
    obligation_regular_file "evidence" (required cli "--evidence")
  in
  if outcome_path = evidence_path then
    failf "obligation-completion-artifacts-not-distinct";
  let outcome_digest = sha256 (read_file outcome_path) in
  let evidence_digest = sha256 (read_file evidence_path) in
  if outcome_digest = evidence_digest then
    failf "obligation-completion-digests-not-distinct";
  with_existing_obligation cli (fun paths events view ->
      if view.obligation_state <> 3
         || not (obligation_owner_matches view actor lane generation)
         || claim <> view.obligation_claim then
        failf "obligation-complete-current-claim-required";
      let epoch = obligation_now_epoch () in
      if epoch > view.obligation_lease_deadline then
        failf "obligation-complete-after-lease-expiry";
      verify_obligation_native_transition "complete" 3 6 view actor lane generation
        claim view.obligation_lease_deadline outcome_digest evidence_digest;
      let updated =
        append_obligation_transition paths events view "OBLIGATION_COMPLETED"
          (obligation_event_fields view epoch
             [ ("actor", actor); ("lane", lane); ("generation", generation);
               ("claim", claim); ("outcome_digest", outcome_digest);
               ("evidence_digest", evidence_digest);
               ("outcome_path", outcome_path); ("evidence_path", evidence_path) ])
      in
      print_obligation ~prefix:"LOOM_OBLIGATION_COMPLETED" updated)

let obligation_status_command cli =
  with_existing_obligation cli (fun _paths _events view ->
      if flag cli "--json" then Printf.printf "%s\n%!" (obligation_json view)
      else print_obligation view)

let obligation_list_json root =
  let views = obligation_views root in
  let unclosed =
    List.fold_left
      (fun count view -> if view.obligation_state = 6 then count else count + 1)
      0 views
  in
  Printf.sprintf
    "{\"schema\":\"loom-obligation-list-v1\",\"count\":%d,\"unclosed\":%d,\"obligations\":[%s]}"
    (List.length views) unclosed (String.concat "," (List.map obligation_json views))

let obligation_list_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let views = obligation_views root in
  let unclosed =
    List.fold_left
      (fun count view -> if view.obligation_state = 6 then count else count + 1)
      0 views
  in
  if flag cli "--json" then
    Printf.printf "%s\n%!" (obligation_list_json root)
  else (
    List.iter print_obligation views;
    Printf.printf "LOOM_OBLIGATION_LIST count=%d unclosed=%d\n%!"
      (List.length views) unclosed)

let obligation_tui_command cli =
  if not (Unix.isatty Unix.stdin) then obligation_list_command cli
  else
    let cwd = cwd_option cli in
    let root = root_option cli cwd in
    let original = set_terminal_raw Unix.stdin in
    let running = ref true in
    Fun.protect
      ~finally:(fun () ->
        Unix.tcsetattr Unix.stdin TCSANOW original;
        print_string "\027[?25h\027[0m\n";
        flush Stdlib.stdout)
      (fun () ->
        print_string "\027[?25l";
        while !running do
          let views = obligation_views root in
          let unclosed =
            List.fold_left
              (fun count view ->
                if view.obligation_state = 6 then count else count + 1)
              0 views
          in
          Printf.printf
            "\027[2J\027[H\027[1;37mSOUNIO LOOM / OBLIGATIONS\027[0m  %d total  %d unclosed\n"
            (List.length views) unclosed;
          Printf.printf
            "\027[90m%-34s %-12s %-14s %-18s %-9s %s\027[0m\n"
            "MESSAGE" "STATE" "ACTOR" "GENERATION" "LEASE" "CLAIM";
          List.iter
            (fun view ->
              let message =
                if String.length view.obligation_message_id > 32 then
                  String.sub view.obligation_message_id 0 32
                else view.obligation_message_id
              in
              let generation =
                if String.length view.obligation_generation > 16 then
                  String.sub view.obligation_generation 0 16
                else view.obligation_generation
              in
              Printf.printf "%-34s %-12s %-14s %-18s %-9s %s\n" message
                (obligation_state_name view.obligation_state)
                view.obligation_actor generation
                (obligation_lease_state view (obligation_now_epoch ()))
                view.obligation_claim)
            views;
          print_string "\n\027[90mq quit   auto-refresh 1s\027[0m\n";
          flush Stdlib.stdout;
          let readable, _, _ = Unix.select [ Unix.stdin ] [] [] 1.0 in
          if readable <> [] then
            let byte = Bytes.create 1 in
            if Unix.read Unix.stdin byte 0 1 = 1 && Bytes.get byte 0 = 'q' then
              running := false
        done)

let obligation_html =
  {|
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sounio Loom Obligations</title>
<style>
:root{color-scheme:dark;--bg:#0b0d0e;--panel:#121617;--line:#293134;--text:#e8eeee;--muted:#8f9b9e;--cyan:#63c7d5;--green:#76d08a;--amber:#e1b65f;--red:#e27676}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:0;min-height:100vh}
header{height:48px;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 16px;gap:18px;background:#101314;position:sticky;top:0}header strong{font-size:15px;color:#fff}header span{color:var(--muted)}#summary{margin-left:auto;color:var(--cyan)}
main{overflow:auto}table{width:100%;border-collapse:collapse;table-layout:fixed}th{text-align:left;color:var(--muted);font-weight:500;background:var(--panel);position:sticky;top:48px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}tbody tr:hover{background:#171c1e}.state{color:var(--green)}.state.claimed,.state.recoverable{color:var(--amber)}.state.interrupted{color:var(--red)}.lease{color:var(--muted)}.empty{padding:28px;color:var(--muted)}
@media(max-width:760px){.generation,.claim{display:none}th,td{padding:9px 8px}header span{display:none}}
</style>
</head>
<body>
<header><strong>SOUNIO LOOM</strong><span>durable obligations</span><b id="summary">LOCAL / READ ONLY</b></header>
<main><table><thead><tr><th>MESSAGE</th><th>STATE</th><th>ACTOR</th><th class="generation">GENERATION</th><th>LEASE</th><th class="claim">CLAIM</th></tr></thead><tbody id="rows"></tbody></table><div class="empty" id="empty" hidden>No obligations</div></main>
<script>
const rows=document.querySelector('#rows'),empty=document.querySelector('#empty'),summary=document.querySelector('#summary');
function cell(text,kind=''){const td=document.createElement('td');td.textContent=text||'';if(kind)td.className=kind;td.title=text||'';return td}
async function refresh(){try{const data=await fetch('/api/obligations',{cache:'no-store'}).then(r=>r.json());rows.replaceChildren();summary.textContent=data.unclosed+' UNCLOSED / '+data.count+' TOTAL';empty.hidden=data.count!==0;for(const o of data.obligations){const tr=document.createElement('tr');tr.append(cell(o.messageId),cell(o.state,'state '+o.state),cell(o.actor),cell(o.generation,'generation'),cell(o.lease,'lease'),cell(o.claim,'claim'));rows.append(tr)}}finally{setTimeout(refresh,1000)}}refresh();
</script>
</body>
</html>
|}

let obligation_serve_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let bind = optional cli "--bind" |> Option.value ~default:"127.0.0.1" in
  if bind <> "127.0.0.1" && bind <> "localhost"
     && not (flag cli "--allow-remote") then
    failf "remote obligation GUI bind requires --allow-remote";
  let port = optional cli "--port" |> Option.value ~default:"8788" |> int_of_string in
  let address =
    try Unix.inet_addr_of_string bind
    with _ -> (Unix.gethostbyname bind).h_addr_list.(0)
  in
  let server = Unix.socket PF_INET SOCK_STREAM 0 in
  Unix.setsockopt server SO_REUSEADDR true;
  Unix.bind server (ADDR_INET (address, port));
  Unix.listen server 32;
  let actual_port =
    match Unix.getsockname server with ADDR_INET (_, value) -> value | _ -> port
  in
  let running = ref true in
  let stop _ = running := false in
  Sys.set_signal Sys.sigterm (Sys.Signal_handle stop);
  Sys.set_signal Sys.sigint (Sys.Signal_handle stop);
  Printf.printf
    "LOOM_OBLIGATION_GUI url=http://%s:%d read_only=true authority=replayed-journal\n%!"
    bind actual_port;
  while !running do
    let readable, _, _ = Unix.select [ server ] [] [] 0.25 in
    if readable <> [] then
      let client, _ = Unix.accept server in
      (try
         let bytes = Bytes.create 16384 in
         let count = Unix.read client bytes 0 (Bytes.length bytes) in
         let request = Bytes.sub_string bytes 0 count in
         let first_line =
           match split_on '\n' request with line :: _ -> trim line | [] -> ""
         in
         let response =
           match split_on ' ' first_line with
           | [ "GET"; uri; _ ] ->
               let path, _query = parse_query uri in
               if path = "/" then
                 http_response "200 OK" "text/html; charset=utf-8" obligation_html
               else if path = "/api/obligations" then
                 http_response "200 OK" "application/json"
                   (obligation_list_json root)
               else http_response "404 Not Found" "text/plain" "not found\n"
           | _ -> http_response "400 Bad Request" "text/plain" "bad request\n"
         in
         write_all client response
       with _ -> ());
      Unix.close client
  done;
  Unix.close server

let obligation_verify_command cli =
  with_existing_obligation cli (fun _paths events view ->
      Printf.printf
        "LOOM_OBLIGATION_VERIFY message=%s state=%s events=%d hash_chain=PASS semantics=PASS head=%s\n%!"
        (field_escape view.obligation_message_id)
        (obligation_state_name view.obligation_state) (List.length events)
        view.obligation_head)

let obligation_supervisor_state root =
  Filename.concat root "obligation-supervisor.state"

let obligation_supervise_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let interval =
    match optional cli "--interval-seconds" with
    | None -> 2
    | Some value -> obligation_positive_int "supervisor-interval" value
  in
  if interval > 60 then failf "obligation-supervisor-interval-too-large";
  let once = flag cli "--once" in
  let rec replay () =
    let views = obligation_views root in
    let unclosed =
      List.fold_left
        (fun count view -> if view.obligation_state = 6 then count else count + 1)
        0 views
    in
    atomic_write (obligation_supervisor_state root)
      (descriptor_text
         [ ("schema", "loom-obligation-supervisor-v1");
           ("pid", string_of_int (Unix.getpid ()));
           ("pid_start", process_start (Unix.getpid ()));
           ("replayed_utc", utc_now ());
           ("count", string_of_int (List.length views));
           ("unclosed", string_of_int unclosed) ]);
    Printf.printf "LOOM_OBLIGATION_SUPERVISOR replayed=%d unclosed=%d\n%!"
      (List.length views) unclosed;
    if not once then (Unix.sleep interval; replay ())
  in
  replay ()

let obligation_supervisor_status_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let path = obligation_supervisor_state root in
  if not (Sys.file_exists path) then failf "obligation-supervisor-state-missing";
  let values = parse_key_values path in
  if table_value values "schema" <> "loom-obligation-supervisor-v1" then
    failf "obligation-supervisor-state-invalid";
  let pid = obligation_positive_int "supervisor-pid" (table_value values "pid") in
  let start = table_value values "pid_start" in
  let live = try process_start pid = start with _ -> false in
  Printf.printf
    "LOOM_OBLIGATION_SUPERVISOR_STATUS state=%s pid=%d replayed_utc=%s count=%s unclosed=%s\n%!"
    (if live then "live" else "stopped") pid (table_value values "replayed_utc")
    (table_value values "count") (table_value values "unclosed")

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

let fleet_observed_fields output slot =
  let status_prefix = "FLEET_SLOT_STATUS" in
  split_on '\n' output
  |> List.find_map (fun line ->
         let tokens = split_on ' ' (trim line) in
         match tokens with
         | prefix :: values when prefix = status_prefix ->
             let fields = snapshot_fields values in
             if table_value fields "slot" = slot then Some fields else None
         | _ -> None)

let fleet_observed_state output slot =
  fleet_observed_fields output slot
  |> Option.map (fun fields -> table_value fields "state")

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

let random_provider_uuid () =
  let value = Bytes.of_string (random_hex 16) in
  Bytes.set value 12 '4';
  Bytes.set value 16 '8';
  let hex = Bytes.unsafe_to_string value in
  Printf.sprintf "%s-%s-%s-%s-%s" (String.sub hex 0 8) (String.sub hex 8 4)
    (String.sub hex 12 4) (String.sub hex 16 4) (String.sub hex 20 12)

let fleet_provider_environment spec =
  let inherited =
    Unix.environment () |> Array.to_list
    |> List.filter (fun entry ->
           not
             (starts_with entry "HOME=" || starts_with entry "CODEX_HOME="
              || starts_with entry "SOUNIO_COORD_DIR="))
  in
  Array.of_list
    (("HOME=" ^ spec.fleet_home)
     :: ("SOUNIO_COORD_DIR=" ^ spec.fleet_coord_dir) :: inherited)

let fleet_coordination_dir cli =
  let candidate =
    match optional cli "--coord-dir" with
    | Some path -> path
    | None -> (
        match Sys.getenv_opt "SOUNIO_COORD_DIR" with
        | Some path when path <> "" -> path
        | _ ->
            try Filename.concat (git_common_dir (Unix.getcwd ())) "sounio-coord-state"
            with _ -> failf "Loom fleet custody requires --coord-dir outside a Git worktree")
  in
  if Filename.is_relative candidate then
    failf "fleet coordination authority must be an absolute directory";
  mkdir_p candidate;
  Unix.realpath candidate

let fleet_assert_loom_identity spec values =
  let assert_field name expected =
    let observed = table_value values name in
    if observed <> expected then
      failf "fleet Loom identity drift slot=%s field=%s expected=%s observed=%s"
        spec.fleet_slot name expected observed
  in
  assert_field "agent" spec.fleet_agent;
  assert_field "lane" spec.fleet_slot;
  assert_field "worktree" spec.fleet_cwd;
  if spec.fleet_session_id <> "" then
    assert_field "session_id" spec.fleet_session_id;
  if spec.fleet_custody = "loom" then (
    let provider = provider_spec spec.fleet_kind in
    let executable =
      match provider_executable provider with
      | Some path -> path
      | None -> failf "provider-executable-not-found:%s" spec.fleet_kind
    in
    assert_field "command" (Filename.basename executable))

let fleet_loom_state root spec =
  let paths = session_paths root spec.fleet_agent spec.fleet_slot in
  let active = try Some (status_request paths |> protocol_fields) with _ -> None in
  match active with
  | Some values ->
      fleet_assert_loom_identity spec values;
      "active"
  | None ->
      if not (Sys.file_exists paths.token_path) then "absent"
      else
        let guardian =
          try
            let token = trim (read_file paths.token_path) in
            Some (guardian_status_request paths token)
          with _ -> None
        in
        (match guardian with
        | Some values ->
            fleet_assert_loom_identity spec values;
            if table_value values "state" = "active" then "recoverable"
            else "absent"
        | None -> "absent")

let fleet_enroll_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let slot = required cli "--slot" in
  let kind = required cli "--kind" in
  let custody = Option.value ~default:"agentd" (optional cli "--custody") in
  let agent = Option.value ~default:kind (optional cli "--agent") in
  let home = Unix.realpath (required cli "--home") in
  validate_fleet_atom "slot" slot;
  validate_fleet_atom "agent" agent;
  if not (List.mem kind fleet_kinds) then failf "unsupported fleet kind: %s" kind;
  if custody <> "agentd" && custody <> "loom" then
    failf "unsupported fleet custody: %s" custody;
  if custody = "agentd"
     && (optional cli "--prompt" <> None || optional cli "--prompt-file" <> None
         || optional cli "--session-id" <> None || optional cli "--model" <> None
         || optional cli "--mode" <> None
         || optional cli "--provider-session" <> None
         || optional cli "--coord-dir" <> None || flag cli "--unsafe-auto"
         || flag cli "--adopt-active")
  then failf "agentd fleet enrollment contains Loom-only authority options";
  if custody = "loom" && not (List.mem kind persistent_fleet_kinds) then
    failf "persistent fleet provider unavailable for kind %s" kind;
  let prompt, prompt_file, prompt_sha256, session_id, provider_mode,
      provider_session, coord_dir, model, unsafe_auto =
    if custody = "loom" then (
      let provider = provider_spec kind in
      let executable =
        match provider_executable provider with
        | Some path -> path
        | None -> failf "provider-executable-not-found:%s" kind
      in
      let prompt = provider_prompt cli in
      let session_id =
        Option.value ~default:(random_provider_uuid ()) (optional cli "--session-id")
      in
      if not (provider_uuid session_id) then
        failf "provider-session-id-must-be-uuid:%s" kind;
      let provider_mode = Option.value ~default:"new" (optional cli "--mode") in
      if provider_mode <> "new" && provider_mode <> "resume" then
        failf "invalid-provider-mode:%s" provider_mode;
      let provider_session =
        Option.value ~default:"" (optional cli "--provider-session")
      in
      if provider_mode = "new" && provider_session <> "" then
        failf "provider-session-is-resume-only";
      if provider_mode = "resume" && provider_session = "" then
        failf "provider-session-is-required-for-resume";
      if provider_mode = "resume"
         && provider.provider_session_binding = "caller"
         && not (provider_uuid provider_session)
      then failf "provider-session-must-be-uuid:%s" kind;
      let model = Option.value ~default:"" (optional cli "--model") in
      let unsafe_auto = flag cli "--unsafe-auto" in
      ignore
        (provider_argv provider "persistent" executable provider_mode cwd
           session_id provider_session model unsafe_auto false prompt);
      (prompt, fleet_prompt_path root slot, sha256 prompt, session_id,
       provider_mode, provider_session,
       fleet_coordination_dir cli, model, unsafe_auto))
    else ("", "", "", "", "", "", "", "", false)
  in
  let spec =
    { fleet_slot = slot; fleet_kind = kind; fleet_custody = custody;
      fleet_agent = agent; fleet_home = home; fleet_cwd = cwd;
      fleet_coord_dir = coord_dir; fleet_enabled = true;
      fleet_session_id = session_id;
      fleet_provider_mode = provider_mode;
      fleet_provider_session = provider_session;
      fleet_prompt_file = prompt_file; fleet_prompt_sha256 = prompt_sha256;
      fleet_model = model; fleet_unsafe_auto = unsafe_auto }
  in
  if custody = "loom" then (
    let provider = provider_spec kind in
    if provider.provider_session_binding = "native-store" then
      match
        load_fleet_specs root
        |> List.find_opt (fun existing ->
               existing.fleet_enabled && existing.fleet_slot <> slot
               && existing.fleet_custody = "loom"
               && existing.fleet_kind = kind && existing.fleet_home = home)
      with
      | Some existing ->
          failf
            "fleet-native-store-home-conflict provider=%s home=%s existing_slot=%s requested_slot=%s"
            kind home existing.fleet_slot slot
      | None -> ());
  let directory = fleet_directory root in
  mkdir_p directory;
  let path = fleet_spec_path root slot in
  let existing =
    if Sys.file_exists path then Some (fleet_spec_of_values path (parse_key_values path))
    else None
  in
  if existing <> None && existing <> Some spec && not (flag cli "--replace") then
    failf "fleet slot %s already has different desired state" slot;
  let agentd_state =
    if custody = "loom" then
      let state, _ = fleet_probe (fleet_agent_command ()) spec in
      state
    else "unobserved"
  in
  let loom_state = fleet_loom_state root spec in
  if custody = "loom" && agentd_state = "active" then
    failf "fleet-authority-conflict slot=%s desired=loom observed=agentd:active" slot;
  if custody = "agentd" && loom_state <> "absent" then
    failf "fleet-authority-conflict slot=%s desired=agentd observed=loom:%s" slot loom_state;
  let same = existing = Some spec in
  if custody = "loom" && loom_state = "active" && not same
     && not (flag cli "--adopt-active")
  then failf "active Loom lane requires --adopt-active for slot %s" slot;
  if flag cli "--adopt-active" && (custody <> "loom" || loom_state <> "active") then
    failf "--adopt-active requires a matching active Loom lane for slot %s" slot;
  if custody = "loom" then atomic_write prompt_file prompt;
  atomic_write path (descriptor_text (fleet_spec_fields spec));
  Printf.printf
    "LOOM_FLEET_ENROLLED slot=%s kind=%s custody=%s agent=%s session_id=%s provider_mode=%s provider_session=%s coord_dir=%s cwd=%s state=enabled adopted=%s\n%!"
    slot kind custody agent (if session_id = "" then "-" else session_id)
    (if provider_mode = "" then "-" else provider_mode)
    (if provider_session = "" then "-" else provider_session)
    (if coord_dir = "" then "-" else coord_dir)
    cwd
    (if flag cli "--adopt-active" then "active" else "no")

let fleet_run_loom root spec action =
  let runtime = Unix.realpath Sys.executable_name in
  let arguments =
    if action = "recover" then
      [ "recover"; "--agent"; spec.fleet_agent; "--lane"; spec.fleet_slot;
        "--session-id"; spec.fleet_session_id; "--cwd"; spec.fleet_cwd;
        "--state-dir"; root ]
    else
      [ "provider-open"; "--provider"; spec.fleet_kind;
        "--agent"; spec.fleet_agent; "--lane"; spec.fleet_slot;
        "--session-id"; spec.fleet_session_id; "--cwd"; spec.fleet_cwd;
        "--state-dir"; root; "--prompt-file"; spec.fleet_prompt_file;
        "--mode"; spec.fleet_provider_mode ]
      @ (if spec.fleet_provider_session = "" then []
         else [ "--provider-session"; spec.fleet_provider_session ])
      @ (if spec.fleet_model = "" then [] else [ "--model"; spec.fleet_model ])
      @ (if spec.fleet_unsafe_auto then [ "--unsafe-auto" ] else [])
  in
  run_captured ~environment:(fleet_provider_environment spec) runtime arguments

let custody_transfer_semantics_sha256 =
  "5f53d3edcb6731c5b0f4e58ff7b27d251e6c0b40eda8c68366e48b17e596f55c"

let custody_transfer_manifest_sha256 =
  "ee4e5d128bf5b0fd7166e74c9815a17506a5b9844730c1be2155ac68c370be66"

let custody_transfer_executable_sha256 =
  "958398e61763d6118c5bd8b86292533dd1b5cc73449df1ede5fb117e37b54ce4"

let custody_transfer_policy_command () =
  let candidate =
    match Sys.getenv_opt "SOUNIO_LOOM_CUSTODY_TRANSFER_COMMAND" with
    | Some path when path <> "" -> path
    | _ ->
        Filename.concat (Filename.dirname Sys.executable_name)
          "sounio-loom-custody-transfer-runtime"
  in
  if Filename.is_relative candidate then
    failf "custody-transfer-policy-command-must-be-absolute";
  let resolved =
    try Unix.realpath candidate
    with _ -> failf "custody-transfer-policy-command-is-unavailable:%s" candidate
  in
  (try Unix.access resolved [ X_OK ]
   with _ -> failf "custody-transfer-policy-command-is-not-executable:%s" resolved);
  let digest = sha256 (read_file resolved) in
  if digest <> custody_transfer_executable_sha256 then
    failf
      "custody-transfer-policy-digest-mismatch:expected=%s:observed=%s"
      custody_transfer_executable_sha256 digest;
  resolved

type custody_transfer_frame = {
  transfer_phase : int;
  transfer_policy_state : int;
  transfer_source_catalog_agentd : int;
  transfer_catalog_committed_loom : int;
  transfer_target_staged : int;
  transfer_target_descriptor_sealed : int;
  transfer_resume_identity_bound : int;
  transfer_source_active : int;
  transfer_source_identity_verified : int;
  transfer_source_quiesced : int;
  transfer_target_active : int;
  transfer_target_presence_verified : int;
  transfer_target_endpoint_verified : int;
  transfer_target_session_verified : int;
  transfer_rollback_available : int;
  transfer_deadline_expired : int;
  transfer_observation_authority_verified : int;
  transfer_sample_fresh : int;
}

let custody_transfer_frame_line frame =
  [ 9040; frame.transfer_phase; frame.transfer_policy_state;
    frame.transfer_source_catalog_agentd;
    frame.transfer_catalog_committed_loom; frame.transfer_target_staged;
    frame.transfer_target_descriptor_sealed;
    frame.transfer_resume_identity_bound; frame.transfer_source_active;
    frame.transfer_source_identity_verified; frame.transfer_source_quiesced;
    frame.transfer_target_active; frame.transfer_target_presence_verified;
    frame.transfer_target_endpoint_verified;
    frame.transfer_target_session_verified; frame.transfer_rollback_available;
    frame.transfer_deadline_expired;
    frame.transfer_observation_authority_verified;
    frame.transfer_sample_fresh ]
  |> List.map string_of_int |> String.concat " " |> fun line -> line ^ "\n"

let custody_transfer_decision frame =
  let command = custody_transfer_policy_command () in
  let result =
    run_captured_input_timeout ~timeout_seconds:2.0 command []
      (custody_transfer_frame_line frame)
  in
  let output = trim result.captured_output in
  let fields = split_on ' ' output |> snapshot_fields in
  if table_value fields "authority" <> "Sounio" then
    failf "custody-transfer-policy-authority-missing:%s" output;
  let code =
    try int_of_string (table_value fields "code")
    with _ -> failf "custody-transfer-policy-result-invalid:%s" output
  in
  if code < 101 && result.captured_code <> 0 then
    failf "custody-transfer-policy-exit-mismatch:code=%d:exit=%d"
      code result.captured_code;
  if code >= 101 && result.captured_code <> code then
    failf "custody-transfer-policy-exit-mismatch:code=%d:exit=%d"
      code result.captured_code;
  (code, table_value fields "decision", output)

let require_custody_transfer_decision expected frame =
  let code, decision, receipt = custody_transfer_decision frame in
  if code <> expected then
    failf "custody-transfer-policy-refused:expected=%d:observed=%d:decision=%s"
      expected code decision;
  receipt

type fleet_transfer_paths = {
  transfer_directory : string;
  transfer_candidate_path : string;
  transfer_prompt_path : string;
  transfer_journal_path : string;
  transfer_lock_path : string;
}

let fleet_transfer_paths root slot =
  let directory =
    Filename.concat (Filename.concat (fleet_directory root) "transfers")
      (slug slot)
  in
  { transfer_directory = directory;
    transfer_candidate_path = Filename.concat directory "candidate.state";
    transfer_prompt_path =
      Filename.concat (Filename.concat directory "prompts") (slug slot ^ ".txt");
    transfer_journal_path = Filename.concat directory "transfer.state";
    transfer_lock_path = Filename.concat directory "transfer.lock" }

let with_fleet_transfer_lock paths operation =
  mkdir_p paths.transfer_directory;
  let descriptor =
    Unix.openfile paths.transfer_lock_path [ O_WRONLY; O_CREAT ] 0o600
  in
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      (try Unix.lockf descriptor F_TLOCK 0
       with Unix_error ((EACCES | EAGAIN), _, _) ->
         failf "fleet-custody-transfer-is-already-running");
      operation ())

type fleet_transfer = {
  custody_paths : fleet_transfer_paths;
  custody_source : fleet_spec;
  custody_target : fleet_spec;
  custody_source_agent : string;
  custody_source_lane : string;
  custody_source_session : string;
  custody_phase : int;
  custody_source_quiesced : bool;
  custody_catalog_committed : bool;
  custody_quiescence_receipt_sha256 : string;
  custody_policy_receipt_sha256 : string;
  custody_created_utc : string;
}

let fleet_transfer_journal_fields transfer =
  [ ("schema", "loom-transactional-custody-transfer-v1");
    ("semantics_sha256", custody_transfer_semantics_sha256);
    ("manifest_sha256", custody_transfer_manifest_sha256);
    ("phase", string_of_int transfer.custody_phase);
    ("source_quiesced",
     if transfer.custody_source_quiesced then "true" else "false");
    ("catalog_committed",
     if transfer.custody_catalog_committed then "true" else "false");
    ("source_slot", transfer.custody_source.fleet_slot);
    ("source_kind", transfer.custody_source.fleet_kind);
    ("source_catalog_agent", transfer.custody_source.fleet_agent);
    ("source_home", transfer.custody_source.fleet_home);
    ("source_cwd", transfer.custody_source.fleet_cwd);
    ("source_agent", transfer.custody_source_agent);
    ("source_lane", transfer.custody_source_lane);
    ("source_session", transfer.custody_source_session);
    ("target_candidate", transfer.custody_paths.transfer_candidate_path);
    ("target_candidate_sha256",
     sha256 (read_file transfer.custody_paths.transfer_candidate_path));
    ("quiescence_receipt_sha256",
     transfer.custody_quiescence_receipt_sha256);
    ("policy_receipt_sha256", transfer.custody_policy_receipt_sha256);
    ("created_utc", transfer.custody_created_utc);
    ("updated_utc", utc_now ()) ]

let write_fleet_transfer_journal transfer =
  atomic_write transfer.custody_paths.transfer_journal_path
    (descriptor_text (fleet_transfer_journal_fields transfer))

let fleet_transfer_bool fields name =
  match table_value fields name with
  | "true" -> true
  | "false" -> false
  | value -> failf "fleet-custody-transfer-invalid-%s:%s" name value

let fleet_transfer_phase fields =
  try
    let phase = int_of_string (table_value fields "phase") in
    if phase < 1 || phase > 6 then failf "fleet-custody-transfer-invalid-phase";
    phase
  with Failure _ -> failf "fleet-custody-transfer-invalid-phase"

let load_fleet_transfer root slot =
  let paths = fleet_transfer_paths root slot in
  if not (Sys.file_exists paths.transfer_journal_path) then
    failf "fleet-custody-transfer-journal-missing:%s" slot;
  let fields = parse_key_values paths.transfer_journal_path in
  if table_value fields "schema" <> "loom-transactional-custody-transfer-v1"
  then failf "fleet-custody-transfer-journal-schema-invalid";
  if table_value fields "semantics_sha256" <> custody_transfer_semantics_sha256
  then failf "fleet-custody-transfer-semantics-drift";
  if table_value fields "manifest_sha256" <> custody_transfer_manifest_sha256
  then failf "fleet-custody-transfer-manifest-drift";
  if table_value fields "source_slot" <> slot then
    failf "fleet-custody-transfer-slot-drift";
  if table_value fields "target_candidate" <> paths.transfer_candidate_path then
    failf "fleet-custody-transfer-candidate-path-drift";
  if not (Sys.file_exists paths.transfer_candidate_path) then
    failf "fleet-custody-transfer-candidate-missing";
  let candidate_sha256 = sha256 (read_file paths.transfer_candidate_path) in
  if table_value fields "target_candidate_sha256" <> candidate_sha256 then
    failf "fleet-custody-transfer-candidate-digest-mismatch";
  let target =
    fleet_spec_of_values paths.transfer_candidate_path
      (parse_key_values paths.transfer_candidate_path)
  in
  if target.fleet_slot <> slot || target.fleet_custody <> "loom" then
    failf "fleet-custody-transfer-candidate-authority-invalid";
  let source =
    { fleet_slot = slot; fleet_kind = table_value fields "source_kind";
      fleet_custody = "agentd";
      fleet_agent = table_value fields "source_catalog_agent";
      fleet_home = table_value fields "source_home";
      fleet_cwd = table_value fields "source_cwd"; fleet_coord_dir = "";
      fleet_enabled = true; fleet_session_id = "";
      fleet_provider_mode = ""; fleet_provider_session = "";
      fleet_prompt_file = ""; fleet_prompt_sha256 = "";
      fleet_model = ""; fleet_unsafe_auto = false }
  in
  List.iter (fun (name, value) -> validate_fleet_atom name value)
    [ ("source_kind", source.fleet_kind);
      ("source_catalog_agent", source.fleet_agent);
      ("source_home", source.fleet_home); ("source_cwd", source.fleet_cwd);
      ("source_agent", table_value fields "source_agent");
      ("source_lane", table_value fields "source_lane");
      ("source_session", table_value fields "source_session") ];
  { custody_paths = paths; custody_source = source; custody_target = target;
    custody_source_agent = table_value fields "source_agent";
    custody_source_lane = table_value fields "source_lane";
    custody_source_session = table_value fields "source_session";
    custody_phase = fleet_transfer_phase fields;
    custody_source_quiesced = fleet_transfer_bool fields "source_quiesced";
    custody_catalog_committed = fleet_transfer_bool fields "catalog_committed";
    custody_quiescence_receipt_sha256 =
      table_value fields "quiescence_receipt_sha256";
    custody_policy_receipt_sha256 = table_value fields "policy_receipt_sha256";
    custody_created_utc = table_value fields "created_utc" }

let fleet_transfer_catalog_target root transfer =
  { transfer.custody_target with
    fleet_prompt_file =
      fleet_prompt_path root transfer.custody_target.fleet_slot }

let fleet_transfer_source_argv transfer =
  let source = transfer.custody_source in
  let provider = provider_spec source.fleet_kind in
  let executable =
    match provider_executable provider with
    | Some path -> path
    | None -> failf "provider-executable-not-found:%s" source.fleet_kind
  in
  provider_argv provider "persistent" executable "resume" source.fleet_cwd
    transfer.custody_source_session transfer.custody_source_session "" false
    false ""

let fleet_transfer_source_plan transfer =
  fleet_transfer_source_argv transfer |> Array.of_list |> command_argv_digest

let fleet_transfer_source_observation helper transfer =
  let source = transfer.custody_source in
  let result =
    run_captured helper
      [ "status"; "--cwd"; source.fleet_cwd; "--slot"; source.fleet_slot ]
  in
  match fleet_observed_fields result.captured_output source.fleet_slot with
  | Some fields ->
      let state = table_value fields "state" in
      let identity =
        state = "active"
        && table_value fields "agent" = transfer.custody_source_agent
        && table_value fields "lane" = transfer.custody_source_lane
        && table_value fields "session_id" = transfer.custody_source_session
        && table_value fields "identity" = "exact"
      in
      if state = "active" && not identity then
        failf "fleet-custody-transfer-source-identity-drift";
      (state, identity, result.captured_output)
  | None
    when List.exists
           (fun line -> trim line = "fleet_slots=0 unhealthy=0")
           (split_on '\n' result.captured_output) ->
      ("absent", false, result.captured_output)
  | None ->
      failf "fleet-custody-transfer-source-observation-failed:%s"
        (trim result.captured_output)

let fleet_transfer_quiesce_source helper transfer =
  let source = transfer.custody_source in
  let result =
    run_captured helper
      [ "stop"; "--cwd"; source.fleet_cwd; "--slot"; source.fleet_slot ]
  in
  let stop_states =
    split_on '\n' result.captured_output
    |> List.filter_map (fun line ->
           match split_on ' ' (trim line) with
           | "FLEET_SLOT_STOPPED" :: values ->
               let fields = snapshot_fields values in
               if table_value fields "slot" = source.fleet_slot then
                 Some (table_value fields "state")
               else None
           | _ -> None)
  in
  if result.captured_code <> 0
     || not (stop_states = [ "active" ] || stop_states = [ "absent" ])
  then
    failf "fleet-custody-transfer-source-stop-unproved:%s"
      (trim result.captured_output);
  let state, _, observation = fleet_transfer_source_observation helper transfer in
  if state <> "absent" then
    failf "fleet-custody-transfer-source-did-not-quiesce:state=%s" state;
  sha256 (result.captured_output ^ "\000" ^ observation)

let fleet_transfer_restore_source helper transfer =
  let source = transfer.custody_source in
  let command = fleet_transfer_source_argv transfer in
  let result =
    run_captured helper
      ([ "launch"; "--slot"; source.fleet_slot; "--agent";
         transfer.custody_source_agent; "--lane";
         transfer.custody_source_lane; "--session-id";
         transfer.custody_source_session; "--identity"; "exact"; "--home";
         source.fleet_home; "--cwd"; source.fleet_cwd; "--no-attach"; "--" ]
       @ command)
  in
  if result.captured_code <> 0 then
    failf "fleet-custody-transfer-source-rollback-failed:%s"
      (trim result.captured_output);
  let state, identity, _ = fleet_transfer_source_observation helper transfer in
  if state <> "active" || not identity then
    failf "fleet-custody-transfer-source-rollback-unproved";
  sha256 result.captured_output

type fleet_transfer_target_observation = {
  target_is_active : bool;
  target_presence_is_verified : bool;
  target_endpoint_is_verified : bool;
  target_session_is_verified : bool;
  target_observation_is_authorized : bool;
  target_sample_is_fresh : bool;
}

let fleet_transfer_target_observation root cwd target =
  let loom_state = fleet_loom_state root target in
  let _, _, authorized, lanes = load_authority_lanes root cwd in
  let lane = authority_entry lanes target.fleet_agent target.fleet_slot in
  { target_is_active = loom_state = "active";
    target_presence_is_verified =
      authorized && lane.authority_presence = "live";
    target_endpoint_is_verified =
      authorized && lane.authority_endpoint = "active";
    target_session_is_verified =
      loom_state = "active"
      && (lane.authority_session_id = ""
          || lane.authority_session_id = target.fleet_session_id);
    target_observation_is_authorized = true;
    target_sample_is_fresh = true }

let fleet_transfer_frame transfer source_active source_identity target
    deadline_expired observation_authorized sample_fresh =
  { transfer_phase = transfer.custody_phase; transfer_policy_state = 1;
    transfer_source_catalog_agentd =
      (if transfer.custody_catalog_committed then 0 else 1);
    transfer_catalog_committed_loom =
      (if transfer.custody_catalog_committed then 1 else 0);
    transfer_target_staged = 1; transfer_target_descriptor_sealed = 1;
    transfer_resume_identity_bound = 1;
    transfer_source_active = if source_active then 1 else 0;
    transfer_source_identity_verified = if source_identity then 1 else 0;
    transfer_source_quiesced =
      if transfer.custody_source_quiesced then 1 else 0;
    transfer_target_active = if target.target_is_active then 1 else 0;
    transfer_target_presence_verified =
      if target.target_presence_is_verified then 1 else 0;
    transfer_target_endpoint_verified =
      if target.target_endpoint_is_verified then 1 else 0;
    transfer_target_session_verified =
      if target.target_session_is_verified then 1 else 0;
    transfer_rollback_available = 1;
    transfer_deadline_expired = if deadline_expired then 1 else 0;
    transfer_observation_authority_verified =
      if observation_authorized then 1 else 0;
    transfer_sample_fresh = if sample_fresh then 1 else 0 }

let empty_fleet_transfer_target_observation =
  { target_is_active = false; target_presence_is_verified = false;
    target_endpoint_is_verified = false; target_session_is_verified = false;
    target_observation_is_authorized = true;
    target_sample_is_fresh = true }

let fleet_transfer_crash point =
  match Sys.getenv_opt "SOUNIO_LOOM_TRANSFER_CRASH_AT" with
  | Some requested when requested = point ->
      failf "fleet-custody-transfer-crash-injected:%s" point
  | _ -> ()

let fleet_transfer_target_identity_equal left right =
  left.fleet_slot = right.fleet_slot
  && left.fleet_kind = right.fleet_kind
  && left.fleet_custody = right.fleet_custody
  && left.fleet_agent = right.fleet_agent
  && left.fleet_home = right.fleet_home
  && left.fleet_cwd = right.fleet_cwd
  && left.fleet_coord_dir = right.fleet_coord_dir
  && left.fleet_enabled = right.fleet_enabled
  && left.fleet_session_id = right.fleet_session_id
  && left.fleet_provider_mode = right.fleet_provider_mode
  && left.fleet_provider_session = right.fleet_provider_session
  && left.fleet_prompt_sha256 = right.fleet_prompt_sha256
  && left.fleet_model = right.fleet_model
  && left.fleet_unsafe_auto = right.fleet_unsafe_auto

let fleet_transfer_catalog_state root transfer =
  let path = fleet_spec_path root transfer.custody_source.fleet_slot in
  if not (Sys.file_exists path) then failf "fleet-custody-transfer-catalog-missing";
  let observed = fleet_spec_of_values path (parse_key_values path) in
  if observed.fleet_custody = "agentd" then (
    if observed.fleet_slot <> transfer.custody_source.fleet_slot
       || observed.fleet_kind <> transfer.custody_source.fleet_kind
       || observed.fleet_home <> transfer.custody_source.fleet_home
       || observed.fleet_cwd <> transfer.custody_source.fleet_cwd
    then failf "fleet-custody-transfer-source-catalog-drift";
    `Agentd)
  else
    let expected = fleet_transfer_catalog_target root transfer in
    if not (fleet_transfer_target_identity_equal observed expected) then
      failf "fleet-custody-transfer-target-catalog-drift";
    `Loom

let fleet_transfer_stop_target root transfer =
  let target = transfer.custody_target in
  let options = Hashtbl.create 8 in
  Hashtbl.replace options "--state-dir" root;
  Hashtbl.replace options "--agent" target.fleet_agent;
  Hashtbl.replace options "--lane" target.fleet_slot;
  Hashtbl.replace options "--cwd" target.fleet_cwd;
  let cli = { options; flags = Hashtbl.create 2; rest = [] } in
  (try stop_command cli with _ -> ());
  match fleet_loom_state root target with
  | "absent" -> ()
  | state -> failf "fleet-custody-transfer-target-stop-unproved:state=%s" state

let fleet_transfer_policy_receipt transfer receipt =
  let updated =
    { transfer with custody_policy_receipt_sha256 = sha256 receipt }
  in
  write_fleet_transfer_journal updated;
  updated

let fleet_transfer_rollback root helper transfer reason =
  let target = fleet_transfer_target_observation root
      transfer.custody_target.fleet_cwd transfer.custody_target in
  let source_state, source_identity, _ =
    fleet_transfer_source_observation helper transfer
  in
  let source_active = source_state = "active" in
  let frame =
    fleet_transfer_frame transfer source_active source_identity target true
      target.target_observation_is_authorized target.target_sample_is_fresh
  in
  let code, _, receipt = custody_transfer_decision frame in
  let transfer = fleet_transfer_policy_receipt transfer receipt in
  let transfer =
    if code = 6 then (
      fleet_transfer_stop_target root transfer;
      { transfer with custody_phase = 6 })
    else if code = 5 || code = 8 || code = 9 then
      { transfer with custody_phase = 6 }
    else
      failf "fleet-custody-transfer-rollback-refused:code=%d:reason=%s" code reason
  in
  write_fleet_transfer_journal transfer;
  let target = fleet_transfer_target_observation root
      transfer.custody_target.fleet_cwd transfer.custody_target in
  if target.target_is_active then
    failf "fleet-custody-transfer-rollback-target-still-active";
  let source_state, source_identity, _ =
    fleet_transfer_source_observation helper transfer
  in
  let transfer =
    if source_state = "active" && source_identity then transfer
    else (
      let frame =
        fleet_transfer_frame transfer false false target false true true
      in
      let receipt = require_custody_transfer_decision 5 frame in
      let transfer = fleet_transfer_policy_receipt transfer receipt in
      ignore (fleet_transfer_restore_source helper transfer);
      transfer)
  in
  let source_state, source_identity, _ =
    fleet_transfer_source_observation helper transfer
  in
  let final_frame =
    fleet_transfer_frame transfer (source_state = "active") source_identity
      target false true true
  in
  let receipt = require_custody_transfer_decision 9 final_frame in
  let transfer = fleet_transfer_policy_receipt transfer receipt in
  write_fleet_transfer_journal transfer;
  Printf.printf
    "LOOM_FLEET_TRANSFER state=ROLLED_BACK slot=%s reason=%s authority=Sounio semantics_sha256=%s\n%!"
    transfer.custody_source.fleet_slot (field_escape reason)
    custody_transfer_semantics_sha256

let fleet_transfer_commit root transfer =
  let committed = fleet_transfer_catalog_target root transfer in
  mkdir_p (Filename.dirname committed.fleet_prompt_file);
  atomic_write committed.fleet_prompt_file
    (read_file transfer.custody_target.fleet_prompt_file);
  atomic_write (fleet_spec_path root committed.fleet_slot)
    (descriptor_text (fleet_spec_fields committed));
  let updated =
    { transfer with custody_phase = 5; custody_catalog_committed = true }
  in
  write_fleet_transfer_journal updated;
  updated

let rec fleet_transfer_finish_committed root helper transfer deadline =
  let target = fleet_transfer_catalog_target root transfer in
  let target_observation =
    fleet_transfer_target_observation root target.fleet_cwd target
  in
  let source_state, source_identity, _ =
    fleet_transfer_source_observation helper transfer
  in
  let frame =
    fleet_transfer_frame transfer (source_state = "active") source_identity
      target_observation (Unix.gettimeofday () >= deadline)
      target_observation.target_observation_is_authorized
      target_observation.target_sample_is_fresh
  in
  let code, _, receipt = custody_transfer_decision frame in
  let transfer = fleet_transfer_policy_receipt transfer receipt in
  match code with
  | 4 ->
      Printf.printf
        "LOOM_FLEET_TRANSFER state=COMPLETE slot=%s custody=loom provider=%s provider_session=%s authority=Sounio semantics_sha256=%s\n%!"
        target.fleet_slot target.fleet_kind target.fleet_provider_session
        custody_transfer_semantics_sha256
  | 10 ->
      ignore (fleet_transfer_quiesce_source helper transfer);
      fleet_transfer_finish_committed root helper transfer deadline
  | 11 ->
      let state = fleet_loom_state root target in
      let action = if state = "recoverable" then "recover" else "provider-open" in
      let result = fleet_run_loom root target action in
      if result.captured_code <> 0 then
        failf "fleet-custody-transfer-target-%s-failed:%s" action
          (trim result.captured_output);
      fleet_transfer_finish_committed root helper transfer deadline
  | 7 when Unix.gettimeofday () < deadline ->
      Unix.sleepf 0.05;
      fleet_transfer_finish_committed root helper transfer deadline
  | _ ->
      failf "fleet-custody-transfer-committed-recovery-refused:code=%d" code

let rec fleet_transfer_drive root helper transfer deadline =
  let catalog_state = fleet_transfer_catalog_state root transfer in
  let transfer =
    match (catalog_state, transfer.custody_catalog_committed) with
    | `Agentd, false -> transfer
    | `Loom, true -> transfer
    | `Loom, false
      when (transfer.custody_phase = 3 || transfer.custody_phase = 4)
           && transfer.custody_policy_receipt_sha256 <> "" ->
        let promoted =
          { transfer with custody_phase = 5; custody_catalog_committed = true }
        in
        write_fleet_transfer_journal promoted;
        promoted
    | `Loom, false ->
        failf "fleet-custody-transfer-catalog-commit-without-policy-receipt"
    | `Agentd, true ->
        failf "fleet-custody-transfer-catalog-rollback-after-commit"
  in
  match transfer.custody_phase with
  | 1 ->
      let source_state, source_identity, _ =
        fleet_transfer_source_observation helper transfer
      in
      if fleet_loom_state root transfer.custody_target <> "absent" then
        failf "fleet-custody-transfer-provisional-target-preexists";
      let transfer =
        if source_state = "active" && source_identity then (
          let frame =
            fleet_transfer_frame transfer true true
              empty_fleet_transfer_target_observation false true true
          in
          let receipt = require_custody_transfer_decision 1 frame in
          fleet_transfer_policy_receipt transfer receipt)
        else if source_state = "absent"
                && transfer.custody_policy_receipt_sha256 <> ""
        then transfer
        else failf "fleet-custody-transfer-source-is-not-active"
      in
      let quiescence = fleet_transfer_quiesce_source helper transfer in
      let next =
        { transfer with custody_phase = 2; custody_source_quiesced = true;
          custody_quiescence_receipt_sha256 = quiescence;
          custody_policy_receipt_sha256 = "" }
      in
      write_fleet_transfer_journal next;
      fleet_transfer_crash "after-quiesce";
      fleet_transfer_drive root helper next deadline
  | 2 ->
      let source_state, source_identity, _ =
        fleet_transfer_source_observation helper transfer
      in
      if source_state = "active" && source_identity then
        fleet_transfer_rollback root helper transfer
          "source-reappeared-before-target"
      else if source_state <> "absent" then
        failf "fleet-custody-transfer-source-reappeared-with-identity-drift"
      else (
        let observed =
          fleet_transfer_target_observation root
            transfer.custody_target.fleet_cwd transfer.custody_target
        in
        let next =
          if observed.target_is_active
             && transfer.custody_policy_receipt_sha256 <> ""
          then { transfer with custody_phase = 3 }
          else (
            let frame =
              fleet_transfer_frame transfer false true observed false true true
            in
            let receipt = require_custody_transfer_decision 2 frame in
            let authorized = fleet_transfer_policy_receipt transfer receipt in
            { authorized with custody_phase = 3 })
        in
        write_fleet_transfer_journal next;
        if not observed.target_is_active then (
          let result = fleet_run_loom root next.custody_target "provider-open" in
          if result.captured_code <> 0 then (
            fleet_transfer_rollback root helper next "target-start-failed";
            failf "fleet-custody-transfer-target-start-failed:%s"
              (trim result.captured_output)));
        fleet_transfer_crash "after-target";
        fleet_transfer_drive root helper next deadline)
  | 3 | 4 ->
      let target =
        fleet_transfer_target_observation root transfer.custody_target.fleet_cwd
          transfer.custody_target
      in
      let source_state, source_identity, _ =
        fleet_transfer_source_observation helper transfer
      in
      let expired = Unix.gettimeofday () >= deadline in
      let frame =
        fleet_transfer_frame transfer (source_state = "active") source_identity
          target expired target.target_observation_is_authorized
          target.target_sample_is_fresh
      in
      let code, _, receipt = custody_transfer_decision frame in
      let transfer = fleet_transfer_policy_receipt transfer receipt in
      (match code with
      | 3 ->
          fleet_transfer_crash "before-commit";
          let committed = fleet_transfer_commit root transfer in
          fleet_transfer_crash "after-commit";
          fleet_transfer_finish_committed root helper committed deadline
      | 7 when not expired ->
          Unix.sleepf 0.05;
          fleet_transfer_drive root helper transfer deadline
      | 5 | 6 | 8 ->
          fleet_transfer_rollback root helper transfer
            (Printf.sprintf "precommit-policy-%d" code)
      | _ ->
          fleet_transfer_rollback root helper transfer
            (Printf.sprintf "precommit-refusal-%d" code))
  | 5 -> fleet_transfer_finish_committed root helper transfer deadline
  | 6 -> fleet_transfer_rollback root helper transfer "resume-rollback"
  | _ -> failf "fleet-custody-transfer-phase-unreachable"

let fleet_transfer_deadline cli =
  let seconds =
    match optional cli "--deadline-seconds" with
    | None -> 10
    | Some value ->
        (try int_of_string value
         with _ -> failf "fleet-custody-transfer-deadline-invalid")
  in
  if seconds < 1 || seconds > 60 then
    failf "fleet-custody-transfer-deadline-out-of-range";
  Unix.gettimeofday () +. float_of_int seconds

let fleet_transfer_stage root _cwd cli =
  let slot = required cli "--slot" in
  let paths = fleet_transfer_paths root slot in
  if Sys.file_exists paths.transfer_journal_path then
    failf "fleet-custody-transfer-already-staged:%s" slot;
  let catalog_path = fleet_spec_path root slot in
  if not (Sys.file_exists catalog_path) then
    failf "fleet-custody-transfer-source-catalog-missing:%s" slot;
  let source = fleet_spec_of_values catalog_path (parse_key_values catalog_path) in
  if source.fleet_custody <> "agentd" || not source.fleet_enabled then
    failf "fleet-custody-transfer-source-catalog-is-not-active-agentd";
  if source.fleet_kind <> "claude" then
    failf "fleet-custody-transfer-provider-not-supported:%s" source.fleet_kind;
  let provider = provider_spec source.fleet_kind in
  let executable =
    match provider_executable provider with
    | Some path -> path
    | None -> failf "provider-executable-not-found:%s" source.fleet_kind
  in
  let loom_session = required cli "--session-id" in
  let provider_session = required cli "--provider-session" in
  if not (provider_uuid loom_session) then
    failf "provider-session-id-must-be-uuid:%s" source.fleet_kind;
  if not (provider_uuid provider_session) then
    failf "provider-session-must-be-uuid:%s" source.fleet_kind;
  let source_agent =
    Option.value ~default:source.fleet_kind (optional cli "--source-agent")
  in
  let source_lane = required cli "--source-lane" in
  let source_session =
    Option.value ~default:provider_session (optional cli "--source-session")
  in
  if source_session <> provider_session then
    failf "fleet-custody-transfer-provider-source-session-mismatch";
  let prompt = provider_prompt cli in
  let model = Option.value ~default:"" (optional cli "--model") in
  let unsafe_auto = flag cli "--unsafe-auto" in
  ignore
    (provider_argv provider "persistent" executable "resume" source.fleet_cwd
       loom_session
       provider_session model unsafe_auto false prompt);
  mkdir_p (Filename.dirname paths.transfer_prompt_path);
  atomic_write paths.transfer_prompt_path prompt;
  let target =
    { fleet_slot = slot; fleet_kind = source.fleet_kind; fleet_custody = "loom";
      fleet_agent =
        Option.value ~default:source.fleet_agent (optional cli "--agent");
      fleet_home = source.fleet_home; fleet_cwd = source.fleet_cwd;
      fleet_coord_dir = fleet_coordination_dir cli; fleet_enabled = true;
      fleet_session_id = loom_session; fleet_provider_mode = "resume";
      fleet_provider_session = provider_session;
      fleet_prompt_file = paths.transfer_prompt_path;
      fleet_prompt_sha256 = sha256 prompt; fleet_model = model;
      fleet_unsafe_auto = unsafe_auto }
  in
  atomic_write paths.transfer_candidate_path
    (descriptor_text (fleet_spec_fields target));
  let transfer =
    { custody_paths = paths; custody_source = source; custody_target = target;
      custody_source_agent = source_agent; custody_source_lane = source_lane;
      custody_source_session = source_session; custody_phase = 1;
      custody_source_quiesced = false; custody_catalog_committed = false;
      custody_quiescence_receipt_sha256 = "";
      custody_policy_receipt_sha256 = ""; custody_created_utc = utc_now () }
  in
  ignore (fleet_transfer_source_plan transfer);
  let state, identity, _ =
    fleet_transfer_source_observation (fleet_agent_command ()) transfer
  in
  if state <> "active" || not identity then
    failf "fleet-custody-transfer-source-preflight-failed";
  if fleet_loom_state root target <> "absent" then
    failf "fleet-custody-transfer-target-preflight-failed";
  write_fleet_transfer_journal transfer;
  fleet_transfer_crash "after-stage";
  transfer

let fleet_transfer_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let slot = required cli "--slot" in
  let paths = fleet_transfer_paths root slot in
  with_fleet_transfer_lock paths (fun () ->
      let transfer = fleet_transfer_stage root cwd cli in
      fleet_transfer_drive root (fleet_agent_command ()) transfer
        (fleet_transfer_deadline cli))

let fleet_transfer_recover_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let slot = required cli "--slot" in
  let paths = fleet_transfer_paths root slot in
  with_fleet_transfer_lock paths (fun () ->
      let transfer = load_fleet_transfer root slot in
      fleet_transfer_drive root (fleet_agent_command ()) transfer
        (fleet_transfer_deadline cli))

let fleet_transfer_reset_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let slot = required cli "--slot" in
  let paths = fleet_transfer_paths root slot in
  with_fleet_transfer_lock paths (fun () ->
      let transfer = load_fleet_transfer root slot in
      if transfer.custody_phase <> 6 || transfer.custody_catalog_committed then
        failf "fleet-custody-transfer-reset-requires-rolled-back-state";
      (match fleet_transfer_catalog_state root transfer with
      | `Agentd -> ()
      | `Loom -> failf "fleet-custody-transfer-reset-refuses-loom-catalog");
      (match fleet_loom_state root transfer.custody_target with
      | "absent" -> ()
      | state ->
          failf "fleet-custody-transfer-reset-target-not-absent:state=%s" state);
      let source_state, source_identity, _ =
        fleet_transfer_source_observation (fleet_agent_command ()) transfer
      in
      if source_state <> "active" || not source_identity then
        failf "fleet-custody-transfer-reset-source-not-restored";
      let receipt =
        require_custody_transfer_decision 9
          (fleet_transfer_frame transfer true true
             empty_fleet_transfer_target_observation false true true)
      in
      let transfer = fleet_transfer_policy_receipt transfer receipt in
      let archive_root =
        Filename.concat
          (Filename.concat (fleet_directory root) "transfer-archive")
          (slug slot)
      in
      mkdir_p archive_root;
      let attempt = sha256 (read_file paths.transfer_journal_path) in
      let archive_path =
        Filename.concat archive_root (String.sub attempt 0 24)
      in
      if Sys.file_exists archive_path then
        failf "fleet-custody-transfer-reset-archive-conflict";
      Unix.rename paths.transfer_directory archive_path;
      Printf.printf
        "LOOM_FLEET_TRANSFER_RESET slot=%s state=ARCHIVED archive=%s authority=Sounio semantics_sha256=%s policy_receipt_sha256=%s\n%!"
        slot archive_path custody_transfer_semantics_sha256
        transfer.custody_policy_receipt_sha256)

let fleet_truthful_state coordination_available snapshot_authorized lanes spec
    agentd_state loom_state =
  let lane = authority_entry lanes spec.fleet_agent spec.fleet_slot in
  let direct_active =
    agentd_state = "active" || loom_state = "active"
  in
  let direct_absent =
    agentd_state = "absent"
    && (loom_state = "absent" || loom_state = "recoverable")
  in
  let direct_complete =
    (agentd_state = "active" || agentd_state = "absent")
    && (loom_state = "active" || loom_state = "recoverable"
       || loom_state = "absent")
  in
  let endpoint_absent =
    lane.authority_endpoint = "unavailable"
    || (coordination_available && snapshot_authorized
       && lane.authority_endpoint = "missing")
  in
  Loom_lane_health.classify
    { Loom_lane_health.policy_state =
        (if coordination_available && snapshot_authorized then 1 else 0);
      expected_lane = true;
      claim_active = lane.authority_claim = "active";
      record_residue =
        lane.authority_loom_instance <> ""
        || lane.authority_presence = "orphaned"
        || loom_state = "recoverable";
      pane_or_harness_exists = direct_active;
      process_verified = direct_active;
      process_unresponsive =
        direct_active && lane.authority_presence = "unresponsive";
      process_absent = direct_absent;
      endpoint_verified = lane.authority_endpoint = "active";
      endpoint_absent;
      endpoint_stale =
        lane.authority_endpoint = "stale"
        || lane.authority_endpoint = "drifted";
      custody_active = loom_state = "active";
      custody_recoverable = loom_state = "recoverable";
      obligation_active = lane.authority_active_obligations > 0;
      blocker_active = lane.authority_blocker_active;
      obligation_census_complete =
        lane.authority_obligation_census_complete;
      progress_observed = lane.authority_progress_observed;
      progress_window_complete = lane.authority_progress_window_complete;
      liveness_window_complete = direct_complete;
      ready_observed = lane.authority_ready_observed;
      observation_authority_verified = snapshot_authorized;
      sample_fresh = coordination_available }

let fleet_reconcile_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let apply = flag cli "--apply" in
  let helper = fleet_agent_command () in
  let specs = load_fleet_specs root |> List.filter (fun spec -> spec.fleet_enabled) in
  let coordination_available, _, snapshot_authorized, lanes =
    load_authority_lanes root cwd
  in
  let started = ref 0 and recovered = ref 0 and healthy = ref 0
  and deferred = ref 0 in
  List.iter
    (fun spec ->
      let agentd_state, _ = fleet_probe helper spec in
      let loom_state = fleet_loom_state root spec in
      let health =
        fleet_truthful_state coordination_available snapshot_authorized lanes spec
          agentd_state loom_state
      in
      let health_name = Loom_lane_health.name health in
      if spec.fleet_custody = "agentd" then (
        if loom_state <> "absent" then
          failf "fleet-authority-conflict slot=%s desired=agentd observed=loom:%s"
            spec.fleet_slot loom_state;
        if health = Loom_lane_health.Working
           || health = Loom_lane_health.Idle
        then (
          incr healthy;
          Printf.printf
            "LOOM_FLEET slot=%s state=%s action=noop custody=agentd\n%!"
            spec.fleet_slot health_name)
        else if health = Loom_lane_health.Dead && not apply then
          Printf.printf
            "LOOM_FLEET slot=%s state=DEAD action=start mode=plan custody=agentd\n%!"
            spec.fleet_slot
        else if health = Loom_lane_health.Dead then (
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
          Printf.printf
            "LOOM_FLEET slot=%s state=DEAD action=started custody=agentd post_state=active\n%!"
            spec.fleet_slot)
        else (
          incr deferred;
          Printf.printf
            "LOOM_FLEET slot=%s state=%s action=operator-required custody=agentd agentd=%s loom=%s\n%!"
            spec.fleet_slot health_name agentd_state loom_state))
      else (
        if spec.fleet_coord_dir = "" then
          failf "fleet coordination authority is missing for Loom slot %s"
            spec.fleet_slot;
        if agentd_state = "active" then
          failf "fleet-authority-conflict slot=%s desired=loom observed=agentd:active"
            spec.fleet_slot;
        if health = Loom_lane_health.Working
           || health = Loom_lane_health.Idle
        then (
          incr healthy;
          Printf.printf
            "LOOM_FLEET slot=%s custody=loom state=%s action=noop\n%!"
            spec.fleet_slot health_name)
        else if health = Loom_lane_health.Dead && not apply then
          Printf.printf
            "LOOM_FLEET slot=%s custody=loom state=DEAD action=provider-open mode=plan\n%!"
            spec.fleet_slot
        else if
          health = Loom_lane_health.Orphaned && loom_state = "recoverable"
          && not apply
        then
          Printf.printf
            "LOOM_FLEET slot=%s custody=loom state=ORPHANED action=recover mode=plan\n%!"
            spec.fleet_slot
        else if
          health = Loom_lane_health.Dead
          || (health = Loom_lane_health.Orphaned
             && loom_state = "recoverable")
        then (
          let action =
            if health = Loom_lane_health.Orphaned then "recover"
            else "provider-open"
          in
          let result = fleet_run_loom root spec action in
          if result.captured_code <> 0 then
            failf "fleet Loom %s failed for %s: %s" action spec.fleet_slot
              (trim result.captured_output);
          let after = fleet_loom_state root spec in
          if after <> "active" then
            failf "fleet slot %s did not enter active Loom custody after %s"
              spec.fleet_slot action;
          if action = "recover" then incr recovered else incr started;
          Printf.printf
            "LOOM_FLEET slot=%s custody=loom state=%s action=%s post_state=active\n%!"
            spec.fleet_slot health_name
            (if action = "recover" then "recovered" else "opened"))
        else (
          incr deferred;
          Printf.printf
            "LOOM_FLEET slot=%s custody=loom state=%s action=operator-required agentd=%s loom=%s\n%!"
            spec.fleet_slot health_name agentd_state loom_state)))
    specs;
  Printf.printf
    "loom_fleet_slots=%d healthy=%d started=%d recovered=%d deferred=%d observation_authorized=%s mode=%s\n%!"
    (List.length specs) !healthy !started !recovered !deferred
    (if coordination_available && snapshot_authorized then "true" else "false")
    (if apply then "apply" else "plan")

type host_boot_decision =
  | Host_noop_active
  | Host_recover_same_physical
  | Host_hold_lineage_required
  | Host_hold_disabled
  | Host_hold_unenrolled
  | Host_denied of string

let host_boot_semantics_sha256 =
  "0d5174cd87b8c18b5f3bbfa7ed44d0258795a96f146730c879c46167abdddf7d"

let host_boot_runtime_sha256 =
  "99f5062729a171ac2d8c1b9b181497fbe1b8c9317859ee0fdc4d2cd4acaedb5b"

let host_boot_decision_name = function
  | Host_noop_active -> "NOOP_ACTIVE"
  | Host_recover_same_physical -> "RECOVER_SAME_PHYSICAL"
  | Host_hold_lineage_required -> "HOLD_LINEAGE_REQUIRED"
  | Host_hold_disabled -> "HOLD_DISABLED"
  | Host_hold_unenrolled -> "HOLD_UNENROLLED"
  | Host_denied value -> value

type host_boot_observation = {
  host_stage : int;
  host_preregistered : int;
  host_producer_sounio : int;
  host_expected_result_sounio : int;
  host_python_absent : int;
  host_rust_absent : int;
  host_policy_present : int;
  host_semantics_hash_bound : int;
  host_runtime_hash_bound : int;
  host_desired_catalog_bound : int;
  host_service_enabled : int;
  host_current_boot_observed : int;
  host_state_root_bound : int;
  host_lane_enrolled : int;
  host_kernel_live : int;
  host_guardian_live : int;
  host_guardian_pid_verified : int;
  host_guardian_start_verified : int;
  host_guardian_instance_verified : int;
  host_harness_live : int;
  host_harness_pid_verified : int;
  host_harness_start_verified : int;
  host_command_bound : int;
  host_boot_equal : int;
  host_journals_verified : int;
  host_output_prefix_preserved : int;
  host_no_same_pty_claim_after_loss : int;
  host_material_observation_joined : int;
  host_sabotage_count : int;
  host_sabotage_required : int;
}

let host_boot_frame observation =
  [ 9041; observation.host_stage; observation.host_preregistered;
    observation.host_producer_sounio;
    observation.host_expected_result_sounio; observation.host_python_absent;
    observation.host_rust_absent; observation.host_policy_present;
    observation.host_semantics_hash_bound;
    observation.host_runtime_hash_bound;
    observation.host_desired_catalog_bound;
    observation.host_service_enabled;
    observation.host_current_boot_observed;
    observation.host_state_root_bound; observation.host_lane_enrolled;
    observation.host_kernel_live; observation.host_guardian_live;
    observation.host_guardian_pid_verified;
    observation.host_guardian_start_verified;
    observation.host_guardian_instance_verified;
    observation.host_harness_live; observation.host_harness_pid_verified;
    observation.host_harness_start_verified; observation.host_command_bound;
    observation.host_boot_equal; observation.host_journals_verified;
    observation.host_output_prefix_preserved;
    observation.host_no_same_pty_claim_after_loss;
    observation.host_material_observation_joined;
    observation.host_sabotage_count; observation.host_sabotage_required ]
  |> List.map string_of_int |> String.concat " " |> fun line -> line ^ "\n"

let host_boot_authority_command () =
  let candidate =
    match Sys.getenv_opt "SOUNIO_LOOM_HOST_BOOT_AUTHORITY" with
    | Some path when path <> "" -> path
    | _ ->
        Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
          "sounio-loom-host-boot-reconciler"
  in
  if Filename.is_relative candidate then
    failf "host-boot-authority-command-must-be-absolute";
  let resolved =
    try Unix.realpath candidate
    with _ -> failf "host-boot-authority-command-is-unavailable:%s" candidate
  in
  (try Unix.access resolved [ X_OK ]
   with _ -> failf "host-boot-authority-command-is-not-executable:%s" resolved);
  let digest = sha256 (read_file resolved) in
  if digest <> host_boot_runtime_sha256 then
    failf
      "host-boot-authority-digest-mismatch:expected=%s:observed=%s"
      host_boot_runtime_sha256 digest;
  resolved

let host_boot_authority_decision observation =
  let command = host_boot_authority_command () in
  let result =
    run_captured_input_timeout ~timeout_seconds:2.0 command []
      (host_boot_frame observation)
  in
  let output = trim result.captured_output in
  let prefix = "SOUNIO_HOST_BOOT_RECONCILER " in
  let suffix = "semantic_authority=Sounio action=9041" in
  let has_suffix =
    String.length output >= String.length suffix
    && String.sub output (String.length output - String.length suffix)
         (String.length suffix)
       = suffix
  in
  if not (starts_with output prefix)
     || not (String.contains output ' ')
     || not has_suffix
  then failf "host-boot-authority-result-invalid:%s" output;
  let decision_name =
    match split_on ' ' output with
    | "SOUNIO_HOST_BOOT_RECONCILER" :: decision :: _ -> decision
    | _ -> failf "host-boot-authority-result-invalid:%s" output
  in
  let decision =
    match decision_name with
    | "NOOP_ACTIVE" -> Host_noop_active
    | "RECOVER_SAME_PHYSICAL" -> Host_recover_same_physical
    | "HOLD_LINEAGE_REQUIRED" -> Host_hold_lineage_required
    | "HOLD_DISABLED" -> Host_hold_disabled
    | "HOLD_UNENROLLED" -> Host_hold_unenrolled
    | value when starts_with value "DENY" -> Host_denied value
    | _ -> failf "host-boot-authority-decision-unknown:%s" decision_name
  in
  (match decision with
  | Host_denied _ when result.captured_code <> 42 ->
      failf "host-boot-authority-denial-exit-mismatch:%d" result.captured_code
  | Host_denied _ -> ()
  | _ when result.captured_code <> 0 ->
      failf "host-boot-authority-allow-exit-mismatch:%d" result.captured_code
  | _ -> ());
  (decision, output)

type hostd_desired_lane = {
  hostd_agent : string;
  hostd_lane : string;
  hostd_session_id : string;
  hostd_worktree : string;
  hostd_command : string;
  hostd_argv_digest : string;
  hostd_root_identity_sha256 : string;
  hostd_enabled : bool;
  hostd_catalog_sha256 : string;
}

let hostd_directory root = Filename.concat root "hostd"
let hostd_lanes_directory root = Filename.concat (hostd_directory root) "lanes"
let hostd_receipts_directory root = Filename.concat (hostd_directory root) "receipts"
let hostd_root_identity_path root = Filename.concat (hostd_directory root) "root.identity"
let hostd_lock_path root = Filename.concat (hostd_directory root) "hostd.lock"
let hostd_supervisor_lock_path root =
  Filename.concat (hostd_directory root) "supervisor.lock"
let hostd_supervisor_state_path root = Filename.concat (hostd_directory root) "supervisor.state"

let hostd_desired_path root agent lane =
  let identity = sha256 (agent ^ "\000" ^ lane) |> fun value -> String.sub value 0 16 in
  Filename.concat (hostd_lanes_directory root)
    (Printf.sprintf "%s--%s--%s.desired" (slug agent) (slug lane) identity)

let hostd_receipt_path root agent lane =
  let identity = sha256 (agent ^ "\000" ^ lane) |> fun value -> String.sub value 0 16 in
  Filename.concat (hostd_receipts_directory root)
    (Printf.sprintf "%s--%s--%s.tsv" (slug agent) (slug lane) identity)

let with_hostd_named_lock root path busy_reason wait_seconds callback =
  let directory = hostd_directory root in
  mkdir_p directory;
  Unix.chmod directory 0o700;
  let lock = Unix.openfile path [ O_WRONLY; O_CREAT ] 0o600 in
  Fun.protect
    ~finally:(fun () -> Unix.close lock)
    (fun () ->
      let deadline = Unix.gettimeofday () +. wait_seconds in
      let rec acquire () =
        try Unix.lockf lock F_TLOCK 0
        with
        | Unix_error ((EACCES | EAGAIN), _, _) ->
            if Unix.gettimeofday () >= deadline then failf "%s" busy_reason;
            Unix.sleepf 0.01;
            acquire ()
      in
      acquire ();
      callback ())

let with_hostd_lock root callback =
  with_hostd_named_lock root (hostd_lock_path root) "loom-hostd-lock-timeout"
    15.0 callback

let with_hostd_supervisor_lock root callback =
  with_hostd_named_lock root (hostd_supervisor_lock_path root)
    "loom-hostd-supervisor-already-active" 0.0 callback

let hostd_root_identity root create =
  let path = hostd_root_identity_path root in
  if not (Sys.file_exists path) then (
    if not create then failf "loom-hostd-root-identity-missing";
    atomic_write path (random_hex 32 ^ "\n");
    Unix.chmod path 0o600);
  let value = trim (read_file path) in
  if String.length value <> 64 then failf "loom-hostd-root-identity-invalid";
  sha256 value

let load_hostd_desired_lane path =
  if (Unix.lstat path).st_kind <> S_REG then failf "loom-hostd-catalog-entry-not-regular:%s" path;
  let values = parse_key_values path in
  if table_value values "schema" <> "loom-hostd-desired-lane-v1" then
    failf "loom-hostd-catalog-schema-invalid:%s" path;
  let required_field field =
    let value = table_value values field in
    if value = "" then failf "loom-hostd-catalog-field-missing:%s:%s" path field;
    value
  in
  let agent = required_field "agent" and lane = required_field "lane" in
  validate_fleet_atom "agent" agent;
  validate_fleet_atom "lane" lane;
  let worktree = Unix.realpath (required_field "worktree") in
  let enabled =
    match required_field "enabled" with
    | "true" -> true
    | "false" -> false
    | value -> failf "loom-hostd-catalog-enabled-invalid:%s" value
  in
  { hostd_agent = agent; hostd_lane = lane;
    hostd_session_id = required_field "session_id";
    hostd_worktree = worktree; hostd_command = required_field "command";
    hostd_argv_digest = required_field "argv_digest";
    hostd_root_identity_sha256 = required_field "state_root_identity_sha256";
    hostd_enabled = enabled; hostd_catalog_sha256 = sha256 (read_file path) }

let load_hostd_desired_lanes root =
  let directory = hostd_lanes_directory root in
  if not (Sys.file_exists directory) then []
  else
    Sys.readdir directory |> Array.to_list |> List.sort String.compare
    |> List.filter (fun name -> Filename.check_suffix name ".desired")
    |> List.map (fun name -> load_hostd_desired_lane (Filename.concat directory name))

let hostd_process_identity pid_text start =
  try
    let pid = int_of_string pid_text in
    pid > 1 && start <> "" && process_start pid = start
  with _ -> false

let hostd_kernel_status paths descriptor =
  let descriptor_pid = table_value descriptor "daemon_pid" in
  let descriptor_start = table_value descriptor "daemon_pid_start" in
  let descriptor_live = hostd_process_identity descriptor_pid descriptor_start in
  if Sys.file_exists paths.socket_path then
    try
      let values = status_request paths |> protocol_fields in
      if table_value values "agent" <> table_value descriptor "agent"
         || table_value values "lane" <> table_value descriptor "lane"
         || table_value values "instance_id" <> table_value descriptor "instance_id"
      then failf "loom-hostd-kernel-identity-drift";
      true
    with error ->
      if descriptor_live then raise error else false
  else if descriptor_live then failf "loom-hostd-live-kernel-socket-missing"
  else false

let hostd_guardian_status paths descriptor =
  let descriptor_pid = table_value descriptor "guardian_pid" in
  let descriptor_start = table_value descriptor "guardian_pid_start" in
  let descriptor_live = hostd_process_identity descriptor_pid descriptor_start in
  if Sys.file_exists paths.guardian_socket_path then
    try Some (guardian_status_request paths (trim (read_file paths.token_path)))
    with error -> if descriptor_live then raise error else None
  else if descriptor_live then failf "loom-hostd-live-guardian-socket-missing"
  else None

let hostd_continuity_observation descriptor guardian =
  try
    let journal_path = table_value descriptor "journal_file" in
    let guardian_path = table_value descriptor "guardian_journal_file" in
    let output_path = table_value descriptor "output_file" in
    let _, _, _ = load_and_verify_journal journal_path in
    let guardian_events, _, guardian_cursor, _ =
      load_and_verify_guardian_journal guardian_path
    in
    let observed_cursor =
      match guardian with
      | Some values -> int_of_string (table_value values "output_cursor")
      | None -> guardian_cursor
    in
    if observed_cursor <> guardian_cursor then
      failf "loom-hostd-guardian-output-cursor-drift";
    ignore
      (verified_guardian_output_range guardian_events output_path guardian_cursor
         0 guardian_cursor);
    (true, true)
  with _ -> (false, false)

let hostd_observation root service_enabled desired =
  let paths = session_paths root desired.hostd_agent desired.hostd_lane in
  if not (Sys.file_exists paths.descriptor_path) then
    failf "loom-hostd-session-descriptor-missing:%s/%s" desired.hostd_agent
      desired.hostd_lane;
  let descriptor = parse_key_values paths.descriptor_path in
  let current_boot = trim (read_file "/proc/sys/kernel/random/boot_id") in
  let root_bound =
    desired.hostd_root_identity_sha256 = hostd_root_identity root false
  in
  let kernel_live = hostd_kernel_status paths descriptor in
  let guardian = hostd_guardian_status paths descriptor in
  let guardian_live = Option.is_some guardian in
  let guardian_pid_verified, guardian_start_verified,
      guardian_instance_verified, harness_live, harness_pid_verified,
      harness_start_verified, guardian_command_bound =
    match guardian with
    | None -> (false, false, false, false, false, false, false)
    | Some values ->
        let guardian_pid_equal =
          table_value values "guardian_pid" = table_value descriptor "guardian_pid"
        in
        let guardian_start_equal =
          table_value values "guardian_pid_start"
          = table_value descriptor "guardian_pid_start"
          && hostd_process_identity (table_value descriptor "guardian_pid")
               (table_value descriptor "guardian_pid_start")
        in
        let harness_pid_equal =
          table_value values "harness_pid" = table_value descriptor "harness_pid"
        in
        let harness_start_equal =
          table_value values "harness_pid_start"
          = table_value descriptor "harness_pid_start"
          && hostd_process_identity (table_value descriptor "harness_pid")
               (table_value descriptor "harness_pid_start")
        in
        (guardian_pid_equal, guardian_start_equal,
         table_value values "instance_id" = table_value descriptor "instance_id",
         harness_start_equal, harness_pid_equal, harness_start_equal,
         table_value values "argv_digest" = table_value descriptor "argv_digest"
         && table_value values "command" = table_value descriptor "command")
  in
  let command_bound =
    guardian_command_bound
    && desired.hostd_session_id = table_value descriptor "session_id"
    && desired.hostd_worktree = table_value descriptor "worktree"
    && desired.hostd_command = table_value descriptor "command"
    && desired.hostd_argv_digest = table_value descriptor "argv_digest"
  in
  let journals_verified, output_prefix_preserved =
    hostd_continuity_observation descriptor guardian
  in
  let int value = if value then 1 else 0 in
  ({ host_stage = 3; host_preregistered = 1; host_producer_sounio = 1;
     host_expected_result_sounio = 1; host_python_absent = 1;
     host_rust_absent = 1; host_policy_present = 1;
     host_semantics_hash_bound = 1; host_runtime_hash_bound = 1;
     host_desired_catalog_bound = int (valid_sha256 desired.hostd_catalog_sha256);
     host_service_enabled = int (service_enabled && desired.hostd_enabled);
     host_current_boot_observed = int (String.length current_boot = 36);
     host_state_root_bound = int root_bound; host_lane_enrolled = 1;
     host_kernel_live = int kernel_live; host_guardian_live = int guardian_live;
     host_guardian_pid_verified = int guardian_pid_verified;
     host_guardian_start_verified = int guardian_start_verified;
     host_guardian_instance_verified = int guardian_instance_verified;
     host_harness_live = int harness_live;
     host_harness_pid_verified = int harness_pid_verified;
     host_harness_start_verified = int harness_start_verified;
     host_command_bound = int command_bound;
     host_boot_equal = int (table_value descriptor "boot_id" = current_boot);
     host_journals_verified = int journals_verified;
     host_output_prefix_preserved = int output_prefix_preserved;
     host_no_same_pty_claim_after_loss = 1;
     host_material_observation_joined = 1; host_sabotage_count = 1;
     host_sabotage_required = 1 }, descriptor)

let hostd_verify_receipts path =
  if not (Sys.file_exists path) then (0, String.make 64 '0')
  else
    let sequence = ref 0 and previous = ref (String.make 64 '0') in
    read_lines path
    |> List.filter (fun line -> trim line <> "")
    |> List.iter (fun line ->
           match split_on '\t' line with
           | [ seq; prior; utc; decision; observation; authority; applied; digest ] ->
               let observed_seq =
                 try int_of_string seq
                 with _ -> failf "loom-hostd-receipt-sequence-invalid"
               in
               if observed_seq <> !sequence + 1 then
                 failf "loom-hostd-receipt-sequence-gap";
               if prior <> !previous then failf "loom-hostd-receipt-chain-drift";
               let material =
                 String.concat "\t"
                   [ seq; prior; utc; decision; observation; authority; applied ]
               in
               if not (valid_sha256 observation) || not (valid_sha256 authority)
                  || digest <> sha256 material
               then failf "loom-hostd-receipt-digest-invalid";
               sequence := observed_seq;
               previous := digest
           | _ -> failf "loom-hostd-receipt-malformed");
    (!sequence, !previous)

let hostd_append_receipt root desired decision observation authority applied =
  let directory = hostd_receipts_directory root in
  mkdir_p directory;
  let path = hostd_receipt_path root desired.hostd_agent desired.hostd_lane in
  let sequence, previous = hostd_verify_receipts path in
  let fields =
    [ string_of_int (sequence + 1); previous; utc_now (); decision;
      observation; authority; if applied then "true" else "false" ]
  in
  let material = String.concat "\t" fields in
  let digest = sha256 material in
  let channel =
    open_out_gen [ Open_wronly; Open_creat; Open_append; Open_text ] 0o600 path
  in
  Fun.protect
    ~finally:(fun () -> close_out_noerr channel)
    (fun () ->
      output_string channel (material ^ "\t" ^ digest ^ "\n");
      flush channel;
      Unix.fsync (Unix.descr_of_out_channel channel));
  (sequence + 1, digest)

let host_enroll_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let agent = required cli "--agent" and lane = required cli "--lane" in
  validate_fleet_atom "agent" agent;
  validate_fleet_atom "lane" lane;
  with_hostd_lock root (fun () ->
      let paths = session_paths root agent lane in
      if not (Sys.file_exists paths.descriptor_path) then
        failf "loom-hostd-enroll-session-missing:%s/%s" agent lane;
      let descriptor = parse_key_values paths.descriptor_path in
      let token = trim (read_file paths.token_path) in
      let guardian = guardian_status_request paths token in
      if table_value guardian "instance_id" <> table_value descriptor "instance_id"
         || not
              (hostd_process_identity (table_value descriptor "guardian_pid")
                 (table_value descriptor "guardian_pid_start"))
         || not
              (hostd_process_identity (table_value descriptor "harness_pid")
                 (table_value descriptor "harness_pid_start"))
      then failf "loom-hostd-enroll-physical-identity-unverified";
      let root_identity = hostd_root_identity root true in
      let directory = hostd_lanes_directory root in
      mkdir_p directory;
      let path = hostd_desired_path root agent lane in
      let desired =
        descriptor_text
          [ ("schema", "loom-hostd-desired-lane-v1"); ("enabled", "true");
            ("agent", agent); ("lane", lane);
            ("session_id", table_value descriptor "session_id");
            ("worktree", table_value descriptor "worktree");
            ("command", table_value descriptor "command");
            ("argv_digest", table_value descriptor "argv_digest");
            ("state_root_identity_sha256", root_identity);
            ("enrolled_instance_id", table_value descriptor "instance_id");
            ("enrolled_boot_id", table_value descriptor "boot_id");
            ("semantic_authority", "Sounio"); ("semantic_action", "9041");
            ("semantics_sha256", host_boot_semantics_sha256) ]
      in
      if Sys.file_exists path && read_file path <> desired && not (flag cli "--replace")
      then failf "loom-hostd-enroll-conflict:%s/%s" agent lane;
      atomic_write path desired;
      Printf.printf
        "LOOM_HOSTD_ENROLLED agent=%s lane=%s catalog_sha256=%s root_identity_sha256=%s authority=Sounio action=9041 service_enabled=false production_activation=false\n%!"
        agent lane (sha256 desired) root_identity)

let hostd_reconcile root service_enabled apply cli =
  let agent_filter = optional cli "--agent" and lane_filter = optional cli "--lane" in
  if Option.is_some agent_filter <> Option.is_some lane_filter then
    failf "host-reconcile requires both --agent and --lane or neither";
  let desired =
    load_hostd_desired_lanes root
    |> List.filter (fun value ->
           match (agent_filter, lane_filter) with
           | Some agent, Some lane ->
               value.hostd_agent = agent && value.hostd_lane = lane
           | _ -> true)
  in
  if Option.is_some agent_filter && desired = [] then
    failf "loom-hostd-requested-lane-unenrolled";
  let noop = ref 0 and recovered = ref 0 and held = ref 0 in
  List.iter
    (fun lane ->
      let observation, _descriptor = hostd_observation root service_enabled lane in
      let observation_digest = sha256 (host_boot_frame observation) in
      let decision, authority_receipt = host_boot_authority_decision observation in
      let authority_digest = sha256 authority_receipt in
      let name = host_boot_decision_name decision in
      match decision with
      | Host_denied _ ->
          ignore
            (hostd_append_receipt root lane name observation_digest authority_digest
               false);
          failf "loom-hostd-authority-refused:%s/%s:%s" lane.hostd_agent
            lane.hostd_lane name
      | Host_recover_same_physical when apply ->
          let authorization_sequence, authorization_head =
            hostd_append_receipt root lane
              "RECOVER_SAME_PHYSICAL_AUTHORIZED" observation_digest
              authority_digest false
          in
          let options = Hashtbl.create 8 in
          Hashtbl.replace options "--agent" lane.hostd_agent;
          Hashtbl.replace options "--lane" lane.hostd_lane;
          Hashtbl.replace options "--session-id" lane.hostd_session_id;
          Hashtbl.replace options "--cwd" lane.hostd_worktree;
          Hashtbl.replace options "--state-dir" root;
          recover_command { options; flags = Hashtbl.create 0; rest = [] };
          let after, _ = hostd_observation root service_enabled lane in
          let after_decision, after_receipt = host_boot_authority_decision after in
          if after_decision <> Host_noop_active then
            failf "loom-hostd-post-recovery-authority-diverged:%s"
              (host_boot_decision_name after_decision);
          let sequence, head =
            hostd_append_receipt root lane "RECOVER_SAME_PHYSICAL_APPLIED"
              (sha256 (host_boot_frame after)) (sha256 after_receipt) true
          in
          incr recovered;
          Printf.printf
            "LOOM_HOSTD lane=%s/%s decision=RECOVER_SAME_PHYSICAL action=applied authorization_sequence=%d authorization_head=%s receipt_sequence=%d receipt_head=%s authority=Sounio semantics_sha256=%s runtime_sha256=%s production_activation=false\n%!"
            lane.hostd_agent lane.hostd_lane authorization_sequence
            authorization_head sequence head host_boot_semantics_sha256
            host_boot_runtime_sha256
      | Host_recover_same_physical ->
          let sequence, head =
            hostd_append_receipt root lane name observation_digest authority_digest
              false
          in
          Printf.printf
            "LOOM_HOSTD lane=%s/%s decision=%s action=plan receipt_sequence=%d receipt_head=%s authority=Sounio production_activation=false\n%!"
            lane.hostd_agent lane.hostd_lane name sequence head;
          incr held
      | Host_noop_active ->
          let sequence, head =
            hostd_append_receipt root lane name observation_digest authority_digest
              false
          in
          Printf.printf
            "LOOM_HOSTD lane=%s/%s decision=NOOP_ACTIVE action=noop receipt_sequence=%d receipt_head=%s authority=Sounio production_activation=false\n%!"
            lane.hostd_agent lane.hostd_lane sequence head;
          incr noop
      | (Host_hold_lineage_required | Host_hold_disabled | Host_hold_unenrolled) ->
          let sequence, head =
            hostd_append_receipt root lane name observation_digest authority_digest
              false
          in
          Printf.printf
            "LOOM_HOSTD lane=%s/%s decision=%s action=hold receipt_sequence=%d receipt_head=%s authority=Sounio same_pty_claim=false production_activation=false\n%!"
            lane.hostd_agent lane.hostd_lane name sequence head;
          incr held)
    desired;
  Printf.printf
    "loom_hostd_lanes=%d noop=%d recovered=%d held=%d mode=%s service_enabled=%s semantic_authority=Sounio action=9041 production_activation=false\n%!"
    (List.length desired) !noop !recovered !held
    (if apply then "apply" else "plan")
    (if service_enabled then "true" else "false")

let host_reconcile_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  with_hostd_lock root (fun () ->
      hostd_reconcile root (flag cli "--service-enabled") (flag cli "--apply") cli)

let host_supervise_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let interval =
    match optional cli "--interval-seconds" with
    | None -> 2
    | Some value -> obligation_positive_int "host-supervisor-interval" value
  in
  if interval > 60 then failf "host-supervisor-interval-too-large";
  let once = flag cli "--once" in
  with_hostd_supervisor_lock root (fun () ->
      let cycles = ref 0 in
      let rec loop () =
        incr cycles;
        (try
           with_hostd_lock root (fun () ->
               hostd_reconcile root (flag cli "--service-enabled")
                 (flag cli "--apply") cli;
               atomic_write (hostd_supervisor_state_path root)
                 (descriptor_text
                    [ ("schema", "loom-hostd-supervisor-v1");
                      ("state", "active");
                      ("pid", string_of_int (Unix.getpid ()));
                      ("pid_start", process_start (Unix.getpid ()));
                      ("boot_id",
                       trim (read_file "/proc/sys/kernel/random/boot_id"));
                      ("cycles", string_of_int !cycles);
                      ("reconciled_utc", utc_now ());
                      ("semantic_authority", "Sounio");
                      ("semantic_action", "9041");
                      ("semantics_sha256", host_boot_semantics_sha256);
                      ("runtime_sha256", host_boot_runtime_sha256) ]))
         with Loom_error reason ->
           atomic_write (hostd_supervisor_state_path root)
             (descriptor_text
                [ ("schema", "loom-hostd-supervisor-v1"); ("state", "refused");
                  ("pid", string_of_int (Unix.getpid ()));
                  ("pid_start", process_start (Unix.getpid ()));
                  ("boot_id", trim (read_file "/proc/sys/kernel/random/boot_id"));
                  ("cycles", string_of_int !cycles); ("refused_utc", utc_now ());
                  ("reason_sha256", sha256 reason); ("semantic_authority", "Sounio");
                  ("semantic_action", "9041") ]);
           raise (Loom_error reason));
        if not once then (Unix.sleep interval; loop ())
      in
      loop ())

let host_verify_command cli =
  let cwd = cwd_option cli in
  let root = root_option cli cwd in
  let agent = required cli "--agent" and lane = required cli "--lane" in
  let path = hostd_receipt_path root agent lane in
  let count, head = hostd_verify_receipts path in
  if count = 0 then failf "loom-hostd-receipts-missing:%s/%s" agent lane;
  Printf.printf
    "LOOM_HOSTD_VERIFY agent=%s lane=%s receipts=%d head=%s hash_chain=PASS semantic_authority=Sounio action=9041\n%!"
    agent lane count head

let subprocess_membrane_probe_command cli =
  if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
    failf "subprocess-membrane-probe-requires-test-mode";
  let root = required cli "--root" in
  let cwd = required cli "--cwd" in
  let scope = required cli "--scope" in
  let deadline_ms =
    try int_of_string (required cli "--deadline-ms")
    with _ -> failf "--deadline-ms must be an integer"
  in
  let argv = Array.of_list cli.rest in
  let outcome =
    Loom_membrane.run_probe ~root ~cwd ~scope ~deadline_ms ~argv
  in
  Printf.printf
    "LOOM_SUBPROCESS_MEMBRANE_PROBE kind=%d exit=%d signal=%d elapsed_us=%Ld events=%d decision_code=%d timed_out=%s policy_error=%s authority=resident-Sounio-v5 authority_pid=%d authority_generation_sha256=%s authority_sequence=%d activation_authority=Sounio activation_code=%d activation_result_sha256=%s activation_projection_sha256=%s activation_capsule_state=%s activation_mode=dark activation_authorizing=false production_activation=false closure_authority=Sounio closure_code=%d closure_result_sha256=%s closure_material=refused sandbox=bubblewrap sandbox_sha256=%s sandbox_ready=%s rootfs=readonly scope=readwrite tmp=ephemeral network=isolated pidns=isolated landlock_abi=%d inherited_fds=closed attachment=refused\n%!"
    outcome.kind outcome.exit_code outcome.signal outcome.elapsed_us
    outcome.event_count outcome.decision_code
    (if outcome.timed_out then "true" else "false")
    (if outcome.policy_error then "true" else "false")
    outcome.authority_pid outcome.authority_generation_sha256
    outcome.authority_sequence outcome.activation_dark_code
    outcome.activation_dark_result_sha256
    outcome.activation_dark_projection_sha256
    outcome.activation_dark_capsule_state outcome.closure_code
    outcome.closure_result_sha256 outcome.sandbox_sha256
    (if outcome.sandbox_ready then "true" else "false") outcome.landlock_abi;
  Loom_membrane.exit_status outcome

let resident_authority_probe_command cli =
  if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
    failf "resident-authority-probe-requires-test-mode";
  let root = required cli "--root" |> Unix.realpath in
  let mode = required cli "--mode" in
  let frame = required cli "--frame" |> read_file |> trim in
  let deadline_ms =
    try int_of_string (required cli "--deadline-ms")
    with _ -> failf "--deadline-ms must be an integer"
  in
  let prohibited_prefixes =
    [ "LD_PRELOAD="; "LD_LIBRARY_PATH="; "LD_AUDIT=";
      "SOUNIO_LOOM_RESIDENT_MEMBRANE_" ]
  in
  let environment =
    Unix.environment () |> Array.to_list
    |> List.filter (fun binding ->
           not (List.exists (fun prefix -> starts_with binding prefix)
                  prohibited_prefixes))
    |> Array.of_list
  in
  Loom_resident.with_generation ~root ~environment ~deadline_ms (fun resident ->
      let reuse_refused () =
        try
          ignore (Loom_resident.decide resident ~deadline_ms frame);
          false
        with Loom_resident.Error "resident-generation-poisoned" -> true
      in
      if mode = "happy" then (
        let decision = Loom_resident.decide resident ~deadline_ms frame in
        Printf.printf
          "LOOM_RESIDENT_OCAML_PROBE mode=happy semantic_authority=Sounio operational_realization=OCaml pid=%d process_identity=stable generation_sha256=%s sequence=%d decision_code=%d latency_us=%Ld poisoned=false\n%!"
          decision.resident_pid decision.generation_sha256 decision.sequence
          decision.code decision.latency_us)
      else if mode = "replay" then (
        ignore (Loom_resident.decide resident ~deadline_ms frame);
        let code = Loom_resident.test_replay resident ~deadline_ms frame in
        Printf.printf
          "LOOM_RESIDENT_OCAML_PROBE mode=replay semantic_authority=Sounio decision_code=%d poisoned=%s reuse_refused=%s\n%!"
          code (if Loom_resident.is_poisoned resident then "true" else "false")
          (if reuse_refused () then "true" else "false"))
      else if mode = "mismatch" then (
        let code = Loom_resident.test_uncorrelated resident ~deadline_ms frame in
        Printf.printf
          "LOOM_RESIDENT_OCAML_PROBE mode=mismatch semantic_authority=Sounio decision_code=%d poisoned=%s reuse_refused=%s\n%!"
          code (if Loom_resident.is_poisoned resident then "true" else "false")
          (if reuse_refused () then "true" else "false"))
      else if mode = "timeout" then (
        let refused = Loom_resident.test_timeout resident frame in
        Printf.printf
          "LOOM_RESIDENT_OCAML_PROBE mode=timeout semantic_authority=Sounio refused=%s poisoned=%s reuse_refused=%s\n%!"
          (if refused then "true" else "false")
          (if Loom_resident.is_poisoned resident then "true" else "false")
          (if reuse_refused () then "true" else "false"))
      else if mode = "eof" then (
        let refused = Loom_resident.test_eof resident ~deadline_ms frame in
        Printf.printf
          "LOOM_RESIDENT_OCAML_PROBE mode=eof semantic_authority=Sounio refused=%s poisoned=%s reuse_refused=%s\n%!"
          (if refused then "true" else "false")
          (if Loom_resident.is_poisoned resident then "true" else "false")
          (if reuse_refused () then "true" else "false"))
      else if mode = "finalize-eof" then (
        Unix.kill (Loom_resident.pid resident) Sys.sigkill;
        ignore (Unix.select [] [] [] 0.01);
        Printf.printf
          "LOOM_RESIDENT_OCAML_PROBE mode=finalize-eof semantic_authority=Sounio callback=returned\n%!")
      else if mode = "benchmark" then (
        let iterations =
          try
            optional cli "--iterations" |> Option.value ~default:"20"
            |> int_of_string
          with _ -> failf "--iterations must be an integer"
        in
        if iterations < 2 || iterations > 200 then
          failf "--iterations must be between 2 and 200";
        ignore (Loom_resident.decide resident ~deadline_ms frame);
        let resident_latencies = ref [] in
        let resident_started = Loom_resident.now_us () in
        for _index = 1 to iterations do
          let decision = Loom_resident.decide resident ~deadline_ms frame in
          if decision.code <> 0 then
            failf "resident benchmark decision changed: %d" decision.code;
          resident_latencies := decision.latency_us :: !resident_latencies
        done;
        let resident_audited_total =
          Int64.sub (Loom_resident.now_us ()) resident_started
        in
        let resident_transport_total =
          List.fold_left Int64.add 0L !resident_latencies
        in
        let single_policy = Loom_membrane.load_policy root in
        ignore
          (Loom_membrane.invoke_decision ~root ~policy:single_policy ~environment
             (frame ^ "\n"));
        let single_latencies = ref [] in
        let single_started = Loom_resident.now_us () in
        for _index = 1 to iterations do
          let started = Loom_resident.now_us () in
          let code, _ =
            Loom_membrane.invoke_decision ~root ~policy:single_policy
              ~environment (frame ^ "\n")
          in
          if code <> 0 then
            failf "single-shot benchmark decision changed: %d" code;
          single_latencies :=
            Int64.sub (Loom_resident.now_us ()) started :: !single_latencies
        done;
        let single_total = Int64.sub (Loom_resident.now_us ()) single_started in
        let percentile values numerator denominator =
          let sorted = List.sort Int64.compare values in
          let length = List.length sorted in
          let index = max 0 (min (length - 1) (((length * numerator) + denominator - 1) / denominator - 1)) in
          List.nth sorted index
        in
        let resident_p50 = percentile !resident_latencies 50 100 in
        let resident_p95 = percentile !resident_latencies 95 100 in
        let single_p50 = percentile !single_latencies 50 100 in
        let single_p95 = percentile !single_latencies 95 100 in
        let speedup_milli =
          if resident_transport_total <= 0L then 0L
          else
            Int64.div (Int64.mul single_total 1000L)
              resident_transport_total
        in
        let audit_overhead =
          Int64.sub resident_audited_total resident_transport_total
        in
        let passed = resident_transport_total < single_total in
        Printf.printf
          "LOOM_RESIDENT_OCAML_PROBE mode=benchmark semantic_authority=Sounio iterations=%d resident_transport_total_us=%Ld resident_audited_total_us=%Ld resident_audit_overhead_us=%Ld resident_p50_us=%Ld resident_p95_us=%Ld single_transport_total_us=%Ld single_p50_us=%Ld single_p95_us=%Ld speedup_milli=%Ld process_identity=stable decisions=parity receipt_policy=fsync-per-event performance_gate=%s\n%!"
          iterations resident_transport_total resident_audited_total audit_overhead
          resident_p50 resident_p95 single_total single_p50 single_p95
          speedup_milli (if passed then "PASS" else "FAIL");
        if not passed then failf "resident-performance-gate-failed")
      else failf "unknown resident-authority probe mode: %s" mode);
  0

let invocation_cell_probe_command cli =
  if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
    failf "invocation-cell-probe-requires-test-mode";
  let root = required cli "--root" |> Unix.realpath in
  let mode = required cli "--mode" in
  let deadline_ms =
    try int_of_string (required cli "--deadline-ms")
    with _ -> failf "--deadline-ms must be an integer"
  in
  let frame name = required cli name |> read_file |> trim in
  let prepare_frame = frame "--prepare" in
  let optional_frame name = optional cli name |> Option.map (fun path -> read_file path |> trim) in
  let require_frame name value =
    match value with Some frame -> frame | None -> failf "%s is required for mode %s" name mode
  in
  let admit_frame = optional_frame "--admit" in
  let close_frame = optional_frame "--close" in
  let abort_frame = optional_frame "--abort" in
  let prohibited_prefixes =
    [ "LD_PRELOAD="; "LD_LIBRARY_PATH="; "LD_AUDIT=";
      "SOUNIO_LOOM_RESIDENT_MEMBRANE_" ]
  in
  let environment =
    Unix.environment () |> Array.to_list
    |> List.filter (fun binding ->
           not (List.exists (fun prefix -> starts_with binding prefix)
                  prohibited_prefixes))
    |> Array.of_list
  in
  let refused callback =
    try ignore (callback ()); false
    with Loom_invocation_cell.Error _ -> true
  in
  Loom_invocation_cell.with_cell ~root ~environment ~deadline_ms (fun cell ->
      let print ~codes ~refused_control ~reuse_refused =
        Printf.printf
          "LOOM_INVOCATION_CELL_OCAML_PROBE mode=%s semantic_authority=Sounio operational_kernel=OCaml manifest_sha256=%s semantics_sha256=%s pid=%d generation_sha256=%s sequence=%d codes=%s state=%s poisoned=%s control_refused=%s reuse_refused=%s material_invocation=false material_coverage=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false\n%!"
          mode (Loom_invocation_cell.manifest_sha256 cell)
          (Loom_invocation_cell.semantics_sha256 cell)
          (Loom_invocation_cell.resident_pid cell)
          (Loom_invocation_cell.generation cell)
          (Loom_invocation_cell.sequence cell) codes
          (Loom_invocation_cell.lifecycle cell
           |> Loom_invocation_cell.state_name)
          (if Loom_invocation_cell.is_poisoned cell then "true" else "false")
          (if refused_control then "true" else "false")
          (if reuse_refused then "true" else "false")
      in
      if mode = "current" || mode = "python" then (
        let decision = Loom_invocation_cell.prepare cell prepare_frame in
        print ~codes:(string_of_int decision.code) ~refused_control:false
          ~reuse_refused:false)
      else if mode = "happy" then (
        let prepare = Loom_invocation_cell.prepare cell prepare_frame in
        let admit =
          Loom_invocation_cell.admit cell
            (require_frame "--admit" admit_frame)
        in
        let close =
          Loom_invocation_cell.close_outcome cell
            (require_frame "--close" close_frame)
        in
        print
          ~codes:(Printf.sprintf "%d,%d,%d" prepare.code admit.code close.code)
          ~refused_control:false ~reuse_refused:false)
      else if mode = "abort" then (
        let prepare = Loom_invocation_cell.prepare cell prepare_frame in
        let abort =
          Loom_invocation_cell.abort cell
            (require_frame "--abort" abort_frame)
        in
        print ~codes:(Printf.sprintf "%d,%d" prepare.code abort.code)
          ~refused_control:false ~reuse_refused:false)
      else if mode = "replay" then (
        let prepare = Loom_invocation_cell.prepare cell prepare_frame in
        let control_refused =
          refused (fun () -> Loom_invocation_cell.prepare cell prepare_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_invocation_cell.admit cell
                (require_frame "--admit" admit_frame))
        in
        print ~codes:(string_of_int prepare.code)
          ~refused_control:control_refused
          ~reuse_refused)
      else if mode = "mismatch" then (
        let prepare = Loom_invocation_cell.prepare cell prepare_frame in
        let control_refused =
          refused (fun () -> Loom_invocation_cell.admit cell prepare_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_invocation_cell.admit cell
                (require_frame "--admit" admit_frame))
        in
        print ~codes:(string_of_int prepare.code)
          ~refused_control:control_refused
          ~reuse_refused)
      else if mode = "timeout" then (
        let prepare = Loom_invocation_cell.prepare cell prepare_frame in
        let control_refused =
          Loom_invocation_cell.test_timeout cell
            (require_frame "--admit" admit_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_invocation_cell.admit cell
                (require_frame "--admit" admit_frame))
        in
        print ~codes:(string_of_int prepare.code)
          ~refused_control:control_refused
          ~reuse_refused)
      else if mode = "eof" then (
        let prepare = Loom_invocation_cell.prepare cell prepare_frame in
        let control_refused =
          Loom_invocation_cell.test_eof cell
            (require_frame "--admit" admit_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_invocation_cell.admit cell
                (require_frame "--admit" admit_frame))
        in
        print ~codes:(string_of_int prepare.code)
          ~refused_control:control_refused
          ~reuse_refused)
      else failf "unknown invocation-cell probe mode: %s" mode);
  0

let exec_grant_cell_probe_command cli =
  if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
    failf "exec-grant-cell-probe-requires-test-mode";
  let root = required cli "--root" |> Unix.realpath in
  let mode = required cli "--mode" in
  let deadline_ms =
    try int_of_string (required cli "--deadline-ms")
    with _ -> failf "--deadline-ms must be an integer"
  in
  let frame name = required cli name |> read_file |> trim in
  let issue_frame = frame "--issue" in
  let optional_frame name =
    optional cli name |> Option.map (fun path -> read_file path |> trim)
  in
  let require_frame name value =
    match value with
    | Some frame -> frame
    | None -> failf "%s is required for mode %s" name mode
  in
  let consume_frame = optional_frame "--consume" in
  let close_frame = optional_frame "--close" in
  let revoke_frame = optional_frame "--revoke" in
  let deny_frame = optional_frame "--deny" in
  let prohibited_prefixes =
    [ "LD_PRELOAD="; "LD_LIBRARY_PATH="; "LD_AUDIT=";
      "SOUNIO_LOOM_RESIDENT_MEMBRANE_" ]
  in
  let environment =
    Unix.environment () |> Array.to_list
    |> List.filter (fun binding ->
           not (List.exists (fun prefix -> starts_with binding prefix)
                  prohibited_prefixes))
    |> Array.of_list
  in
  let refused callback =
    try ignore (callback ()); false
    with Loom_exec_grant_cell.Error _ -> true
  in
  Loom_exec_grant_cell.with_cell ~root ~environment ~deadline_ms (fun cell ->
      let print ~codes ~control_refused ~reuse_refused ~deny_preserved =
        Printf.printf
          "LOOM_EXEC_GRANT_CELL_OCAML_PROBE mode=%s semantic_authority=Sounio operational_kernel=OCaml manifest_sha256=%s semantics_sha256=%s resident_v4_sha256=%s pid=%d generation_sha256=%s sequence=%d codes=%s state=%s poisoned=%s control_refused=%s reuse_refused=%s deny_preserved=%s material_grant=false material_coverage=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false\n%!"
          mode (Loom_exec_grant_cell.manifest_sha256 cell)
          (Loom_exec_grant_cell.semantics_sha256 cell)
          (Loom_exec_grant_cell.resident_v4_sha256 cell)
          (Loom_exec_grant_cell.resident_pid cell)
          (Loom_exec_grant_cell.generation cell)
          (Loom_exec_grant_cell.sequence cell) codes
          (Loom_exec_grant_cell.state cell
           |> Loom_exec_grant_cell.state_name)
          (if Loom_exec_grant_cell.is_poisoned cell then "true" else "false")
          (if control_refused then "true" else "false")
          (if reuse_refused then "true" else "false")
          (if deny_preserved then "true" else "false")
      in
      if mode = "current" || mode = "python" then (
        let decision = Loom_exec_grant_cell.issue cell issue_frame in
        print ~codes:(string_of_int decision.code) ~control_refused:false
          ~reuse_refused:false ~deny_preserved:false)
      else if mode = "happy" then (
        let issue = Loom_exec_grant_cell.issue cell issue_frame in
        let consume =
          Loom_exec_grant_cell.consume cell
            (require_frame "--consume" consume_frame)
        in
        let close =
          Loom_exec_grant_cell.close_outcome cell
            (require_frame "--close" close_frame)
        in
        print ~codes:(Printf.sprintf "%d,%d,%d" issue.code consume.code close.code)
          ~control_refused:false ~reuse_refused:false ~deny_preserved:false)
      else if mode = "deny-preserves" then (
        let issue = Loom_exec_grant_cell.issue cell issue_frame in
        let denied =
          Loom_exec_grant_cell.consume cell (require_frame "--deny" deny_frame)
        in
        let preserved = Loom_exec_grant_cell.state cell = Loom_exec_grant_cell.Issued in
        let consume =
          Loom_exec_grant_cell.consume cell
            (require_frame "--consume" consume_frame)
        in
        let close =
          Loom_exec_grant_cell.close_outcome cell
            (require_frame "--close" close_frame)
        in
        print
          ~codes:(Printf.sprintf "%d,%d,%d,%d" issue.code denied.code
                    consume.code close.code)
          ~control_refused:false ~reuse_refused:false ~deny_preserved:preserved)
      else if mode = "revoke" then (
        let issue = Loom_exec_grant_cell.issue cell issue_frame in
        let revoke =
          Loom_exec_grant_cell.revoke cell
            (require_frame "--revoke" revoke_frame)
        in
        print ~codes:(Printf.sprintf "%d,%d" issue.code revoke.code)
          ~control_refused:false ~reuse_refused:false ~deny_preserved:false)
      else if mode = "replay" then (
        let issue = Loom_exec_grant_cell.issue cell issue_frame in
        let control_refused =
          refused (fun () -> Loom_exec_grant_cell.issue cell issue_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_exec_grant_cell.consume cell
                (require_frame "--consume" consume_frame))
        in
        print ~codes:(string_of_int issue.code) ~control_refused
          ~reuse_refused ~deny_preserved:false)
      else if mode = "mismatch" then (
        let issue = Loom_exec_grant_cell.issue cell issue_frame in
        let control_refused =
          refused (fun () -> Loom_exec_grant_cell.consume cell issue_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_exec_grant_cell.consume cell
                (require_frame "--consume" consume_frame))
        in
        print ~codes:(string_of_int issue.code) ~control_refused
          ~reuse_refused ~deny_preserved:false)
      else if mode = "timeout" then (
        let issue = Loom_exec_grant_cell.issue cell issue_frame in
        let control_refused =
          Loom_exec_grant_cell.test_timeout cell
            (require_frame "--consume" consume_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_exec_grant_cell.consume cell
                (require_frame "--consume" consume_frame))
        in
        print ~codes:(string_of_int issue.code) ~control_refused
          ~reuse_refused ~deny_preserved:false)
      else if mode = "eof" then (
        let issue = Loom_exec_grant_cell.issue cell issue_frame in
        let control_refused =
          Loom_exec_grant_cell.test_eof cell
            (require_frame "--consume" consume_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_exec_grant_cell.consume cell
                (require_frame "--consume" consume_frame))
        in
        print ~codes:(string_of_int issue.code) ~control_refused
          ~reuse_refused ~deny_preserved:false)
      else failf "unknown exec-grant-cell probe mode: %s" mode);
  0

let peer_activation_capsule_probe_command cli =
  if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
    failf "peer-activation-capsule-probe-requires-test-mode";
  let root = required cli "--root" |> Unix.realpath in
  let mode = required cli "--mode" in
  let deadline_ms =
    try int_of_string (required cli "--deadline-ms")
    with _ -> failf "--deadline-ms must be an integer"
  in
  let frame name = required cli name |> read_file |> trim in
  let seal_frame = frame "--seal" in
  let optional_frame name =
    optional cli name |> Option.map (fun path -> read_file path |> trim)
  in
  let require_frame name value =
    match value with
    | Some frame -> frame
    | None -> failf "%s is required for mode %s" name mode
  in
  let consume_frame = optional_frame "--consume" in
  let extinguish_frame = optional_frame "--extinguish" in
  let poison_frame = optional_frame "--poison" in
  let deny_frame = optional_frame "--deny" in
  let prohibited_prefixes =
    [ "LD_PRELOAD="; "LD_LIBRARY_PATH="; "LD_AUDIT=";
      "SOUNIO_LOOM_RESIDENT_MEMBRANE_" ]
  in
  let environment =
    Unix.environment () |> Array.to_list
    |> List.filter (fun binding ->
           not (List.exists (fun prefix -> starts_with binding prefix)
                  prohibited_prefixes))
    |> Array.of_list
  in
  let refused callback =
    try ignore (callback ()); false
    with Loom_peer_activation_capsule.Error _ -> true
  in
  Loom_peer_activation_capsule.with_cell ~root ~environment ~deadline_ms
    (fun cell ->
      let print ~codes ~control_refused ~reuse_refused ~deny_preserved =
        Printf.printf
          "LOOM_PEER_ACTIVATION_CAPSULE_OCAML_PROBE mode=%s semantic_authority=Sounio operational_realization=OCaml resident_model=single-Sounio-pid manifest_sha256=%s semantics_sha256=%s resident_v5_sha256=%s pid=%d generation_sha256=%s sequence=%d codes=%s state=%s poisoned=%s control_refused=%s reuse_refused=%s deny_preserved=%s same_uid_peer_isolation=true capsule_material=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n%!"
          mode (Loom_peer_activation_capsule.manifest_sha256 cell)
          (Loom_peer_activation_capsule.semantics_sha256 cell)
          (Loom_peer_activation_capsule.resident_v5_sha256 cell)
          (Loom_peer_activation_capsule.resident_pid cell)
          (Loom_peer_activation_capsule.generation cell)
          (Loom_peer_activation_capsule.sequence cell) codes
          (Loom_peer_activation_capsule.state cell
           |> Loom_peer_activation_capsule.state_name)
          (if Loom_peer_activation_capsule.is_poisoned cell then "true"
           else "false")
          (if control_refused then "true" else "false")
          (if reuse_refused then "true" else "false")
          (if deny_preserved then "true" else "false")
      in
      if mode = "current" || mode = "python" then (
        let decision = Loom_peer_activation_capsule.seal cell seal_frame in
        let preserved =
          Loom_peer_activation_capsule.state cell
          = Loom_peer_activation_capsule.Empty
        in
        print ~codes:(string_of_int decision.code) ~control_refused:false
          ~reuse_refused:false ~deny_preserved:preserved)
      else if mode = "happy" then (
        let seal = Loom_peer_activation_capsule.seal cell seal_frame in
        let consume =
          Loom_peer_activation_capsule.consume cell
            (require_frame "--consume" consume_frame)
        in
        let extinguish =
          Loom_peer_activation_capsule.extinguish cell
            (require_frame "--extinguish" extinguish_frame)
        in
        print
          ~codes:(Printf.sprintf "%d,%d,%d" seal.code consume.code
                    extinguish.code)
          ~control_refused:false ~reuse_refused:false ~deny_preserved:false)
      else if mode = "deny-preserves" then (
        let denied =
          Loom_peer_activation_capsule.seal cell
            (require_frame "--deny" deny_frame)
        in
        let preserved =
          Loom_peer_activation_capsule.state cell
          = Loom_peer_activation_capsule.Empty
        in
        let seal = Loom_peer_activation_capsule.seal cell seal_frame in
        let consume =
          Loom_peer_activation_capsule.consume cell
            (require_frame "--consume" consume_frame)
        in
        let extinguish =
          Loom_peer_activation_capsule.extinguish cell
            (require_frame "--extinguish" extinguish_frame)
        in
        print
          ~codes:(Printf.sprintf "%d,%d,%d,%d" denied.code seal.code
                    consume.code extinguish.code)
          ~control_refused:false ~reuse_refused:false
          ~deny_preserved:preserved)
      else if mode = "poison" then (
        let seal = Loom_peer_activation_capsule.seal cell seal_frame in
        let consume =
          Loom_peer_activation_capsule.consume cell
            (require_frame "--consume" consume_frame)
        in
        let poison =
          Loom_peer_activation_capsule.poison cell
            (require_frame "--poison" poison_frame)
        in
        print ~codes:(Printf.sprintf "%d,%d,%d" seal.code consume.code poison.code)
          ~control_refused:false ~reuse_refused:false ~deny_preserved:false)
      else if mode = "replay" then (
        let seal = Loom_peer_activation_capsule.seal cell seal_frame in
        let control_refused =
          refused (fun () -> Loom_peer_activation_capsule.seal cell seal_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_peer_activation_capsule.consume cell
                (require_frame "--consume" consume_frame))
        in
        print ~codes:(string_of_int seal.code) ~control_refused ~reuse_refused
          ~deny_preserved:false)
      else if mode = "mismatch" then (
        let seal = Loom_peer_activation_capsule.seal cell seal_frame in
        let control_refused =
          refused (fun () ->
              Loom_peer_activation_capsule.consume cell seal_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_peer_activation_capsule.consume cell
                (require_frame "--consume" consume_frame))
        in
        print ~codes:(string_of_int seal.code) ~control_refused ~reuse_refused
          ~deny_preserved:false)
      else if mode = "timeout" then (
        let seal = Loom_peer_activation_capsule.seal cell seal_frame in
        let control_refused =
          Loom_peer_activation_capsule.test_timeout cell
            (require_frame "--consume" consume_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_peer_activation_capsule.consume cell
                (require_frame "--consume" consume_frame))
        in
        print ~codes:(string_of_int seal.code) ~control_refused ~reuse_refused
          ~deny_preserved:false)
      else if mode = "eof" then (
        let seal = Loom_peer_activation_capsule.seal cell seal_frame in
        let control_refused =
          Loom_peer_activation_capsule.test_eof cell
            (require_frame "--consume" consume_frame)
        in
        let reuse_refused =
          refused (fun () ->
              Loom_peer_activation_capsule.consume cell
                (require_frame "--consume" consume_frame))
        in
        print ~codes:(string_of_int seal.code) ~control_refused ~reuse_refused
          ~deny_preserved:false)
      else failf "unknown peer-activation-capsule probe mode: %s" mode);
  0

let exec_ingress_probe_command cli =
  if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
    failf "exec-ingress-probe requires SOUNIO_LOOM_HOOK_TEST_MODE=1";
  let root = required cli "--root" |> Unix.realpath in
  let mode = required cli "--mode" in
  let event_path = required cli "--event" in
  let result_mode =
    List.mem mode
      [ "result"; "result-binding"; "result-receipt"; "result-manifest" ]
  in
  let record_mode =
    List.mem mode
      [ "record"; "record-binding"; "record-digest"; "record-manifest" ]
  in
  if mode <> "inherited" && mode <> "forged" && mode <> "missing"
     && mode <> "fixture-escape" && not result_mode && not record_mode
  then
    failf
      "exec-ingress-probe mode must be inherited, forged, missing, fixture-escape, result variants, or record variants";
  let event = read_file event_path in
  if event = "" then failf "exec-ingress-probe event is empty";
  let result_policy =
    if result_mode then Some (Loom_exec_result.load ~root) else None
  in
  let record_policy =
    if record_mode then Some (Loom_exec_result_record.load ~root) else None
  in
  let result_receipt =
    if result_mode then read_file (required cli "--receipt") else ""
  in
  let result_record =
    if record_mode then read_file (required cli "--record") else ""
  in
  let result_record_sha256 = if record_mode then sha256 result_record else "" in
  let result_record_handle =
    if record_mode then required cli "--handle" else ""
  in
  let channel =
    if mode = "missing" then None else Some (Unix.socketpair PF_UNIX SOCK_STREAM 0)
  in
  let broker_pid =
    match channel with
    | None -> None
    | Some (server, client) ->
        Some
          (match Unix.fork () with
          | 0 ->
              Unix.close client;
              let code =
                try
                  let request = read_line_fd server in
                  match String.split_on_char '\t' request with
                  | [ "LOOM_EXEC_INGRESS/1"; event_sha256; command_sha256 ]
                    when String.length event_sha256 = 64
                         && String.length command_sha256 = 64 ->
                      (match result_policy, record_policy with
                      | None, None ->
                          write_all server
                            (String.concat "\t"
                               [ "LOOM_EXEC_INGRESS_BOUND/1"; event_sha256;
                                 command_sha256 ] ^ "\n")
                      | Some policy, None ->
                          let response_command =
                            if mode = "result-binding" then
                              String.sub command_sha256 0 63 ^
                              (if command_sha256.[63] = '0' then "1" else "0")
                            else command_sha256
                          in
                          let response_receipt =
                            if mode = "result-receipt" then
                              String.sub policy.result_receipt_sha256 0 63 ^
                              (if policy.result_receipt_sha256.[63] = '0'
                               then "1" else "0")
                            else policy.result_receipt_sha256
                          in
                          let response_manifest =
                            if mode = "result-manifest" then
                              String.sub policy.manifest_sha256 0 63 ^
                              (if policy.manifest_sha256.[63] = '0'
                               then "1" else "0")
                            else policy.manifest_sha256
                          in
                          write_all server
                            (String.concat "\t"
                               [ "LOOM_EXEC_RESULT/1"; event_sha256;
                                 response_command; policy.canonical_handle;
                                 response_receipt; hex_of_string result_receipt;
                                 response_manifest ] ^ "\n")
                      | None, Some policy ->
                          let response_command =
                            if mode = "record-binding" then
                              String.sub command_sha256 0 63 ^
                              (if command_sha256.[63] = '0' then "1" else "0")
                            else command_sha256
                          in
                          let response_record =
                            if mode = "record-digest" then
                              String.sub result_record_sha256 0 63 ^
                              (if result_record_sha256.[63] = '0'
                               then "1" else "0")
                            else result_record_sha256
                          in
                          let response_manifest =
                            if mode = "record-manifest" then
                              String.sub policy.manifest_sha256 0 63 ^
                              (if policy.manifest_sha256.[63] = '0'
                               then "1" else "0")
                            else policy.manifest_sha256
                          in
                          write_all server
                            (String.concat "\t"
                               [ "LOOM_EXEC_RESULT_RECORD/1"; event_sha256;
                                 response_command; result_record_handle;
                                 response_record; hex_of_string result_record;
                                 response_manifest ] ^ "\n")
                      | Some _, Some _ -> failf "exec-ingress-probe-result-mode-conflict");
                      Unix.shutdown server SHUTDOWN_ALL;
                      0
                  | _ -> 91
                with _ -> 90
              in
              Unix.close server;
              Unix._exit code
          | pid ->
              Unix.close server;
              pid)
  in
  let client = Option.map snd channel in
  let input_read, input_write = Unix.pipe () in
  let output_read, output_write = Unix.pipe () in
  Unix.set_close_on_exec input_write;
  Unix.set_close_on_exec output_read;
  let set_environment name value environment =
    let prefix = name ^ "=" in
    environment |> Array.to_list
    |> List.filter (fun binding -> not (starts_with binding prefix))
    |> fun bindings -> Array.of_list ((prefix ^ value) :: bindings)
  in
  let environment =
    Unix.environment ()
    |> set_environment "SOUNIO_LOOM_HOOK_TEST_MODE" "1"
    |> set_environment "SOUNIO_LOOM_EXEC_INGRESS_REQUIRED" "1"
    |> set_environment "SOUNIO_COORD_NATIVE_HOOK_SELFTEST" "1"
  in
  let environment =
    match result_policy, record_policy with
    | None, None -> environment
    | Some _, None ->
        set_environment "SOUNIO_LOOM_EXEC_INTENT_PROJECTION" "1" environment
    | None, Some _ ->
        set_environment "SOUNIO_LOOM_EXEC_OPERATION_PROJECTION" "1" environment
    | Some _, Some _ -> failf "exec-ingress-probe-result-mode-conflict"
  in
  let environment =
    if mode = "fixture-escape" then
      environment |> Array.to_list
      |> List.filter (fun binding ->
             not (starts_with binding "SOUNIO_LOOM_EXEC_INGRESS_PROBE_ONLY="))
      |> Array.of_list
    else
      set_environment "SOUNIO_LOOM_EXEC_INGRESS_PROBE_ONLY" "1" environment
  in
  let environment =
    match client with
    | None ->
        environment |> Array.to_list
        |> List.filter (fun binding ->
               not (starts_with binding "SOUNIO_LOOM_EXEC_INGRESS_FD=")
               && not
                    (starts_with binding
                       "SOUNIO_LOOM_EXEC_INGRESS_ALLOW_SAME_UID_TEST="))
        |> Array.of_list
    | Some descriptor ->
        let environment =
          set_environment "SOUNIO_LOOM_EXEC_INGRESS_FD"
            (string_of_int (int_of_file_descr descriptor)) environment
        in
        if mode = "inherited" || mode = "fixture-escape" || result_mode
           || record_mode then
          set_environment "SOUNIO_LOOM_EXEC_INGRESS_ALLOW_SAME_UID_TEST" "1"
            environment
        else
          environment |> Array.to_list
          |> List.filter (fun binding ->
                 not
                   (starts_with binding
                      "SOUNIO_LOOM_EXEC_INGRESS_ALLOW_SAME_UID_TEST="))
          |> Array.of_list
  in
  let hook_pid =
    match Unix.fork () with
    | 0 ->
        Unix.close input_write;
        Unix.close output_read;
        Unix.dup2 input_read Unix.stdin;
        Unix.dup2 output_write Unix.stdout;
        Unix.dup2 output_write Unix.stderr;
        if input_read <> Unix.stdin then Unix.close input_read;
        if output_write <> Unix.stdout && output_write <> Unix.stderr then
          Unix.close output_write;
        Option.iter Unix.clear_close_on_exec client;
        (try
           Unix.chdir root;
           let executable = Unix.realpath Sys.executable_name in
           Unix.execve executable
             [| executable; "agent-hook"; "--agent"; "codex" |]
             environment
         with _ -> Unix._exit 127)
    | pid -> pid
  in
  Unix.close input_read;
  Unix.close output_write;
  Option.iter Unix.close client;
  write_all input_write event;
  write_all input_write "\n";
  Unix.close input_write;
  let output = Buffer.create 4096 in
  let bytes = Bytes.create 16384 in
  let rec drain () =
    match Unix.read output_read bytes 0 (Bytes.length bytes) with
    | 0 -> ()
    | count -> Buffer.add_subbytes output bytes 0 count; drain ()
    | exception Unix_error (EINTR, _, _) -> drain ()
  in
  Fun.protect ~finally:(fun () -> Unix.close output_read) drain;
  let status_code = function
    | WEXITED code -> code
    | WSIGNALED signal | WSTOPPED signal -> 128 + signal
  in
  let _, hook_status = Unix.waitpid [] hook_pid in
  let broker_code =
    match broker_pid with
    | None -> -1
    | Some pid ->
        let _, status = Unix.waitpid [] pid in
        status_code status
  in
  let hook_output = Buffer.contents output in
  Printf.printf
    "LOOM_PRODUCT_EXEC_INGRESS_PROBE mode=%s hook_code=%d broker_code=%d output_sha256=%s result_returned=%s exact_fixture_hook_switched=%s production_activation=false exec_attached=false\n%s%!"
    mode (status_code hook_status) broker_code (sha256 hook_output)
    (if mode = "result" || mode = "record" then "true" else "false")
    (if mode = "result" then "true" else "false") hook_output;
  0

let exec_result_probe_command cli =
  if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
    failf "exec-result-probe requires SOUNIO_LOOM_HOOK_TEST_MODE=1";
  let root = required cli "--root" |> Unix.realpath in
  let store_root = required cli "--store" in
  let mode = required cli "--mode" in
  let print_result mode (result : Loom_exec_result.stored_result) =
    Printf.printf
      "LOOM_EXEC_RESULT_STORE_PROBE mode=%s semantic_authority=Sounio action=9033 operational_kernel=OCaml manifest_sha256=%s handle=%s record_sha256=%s receipt_sha256=%s authority_output_sha256=%s record_path=%s receipt_hex=%s material_result_store=true result_store_attached=false handle_is_bearer=false handle_is_execution_authority=false exec_attached=false provider_hook_switched=false production_activation=false\n%!"
      mode (Loom_exec_result.manifest_sha256 result) result.handle
      result.record_sha256 result.receipt_sha256
      (Loom_exec_result.authority_output_sha256 result) result.path
      (hex_of_string result.receipt)
  in
  if mode = "publish" then
    Loom_exec_result.publish ~root ~store_root
      ~receipt_path:(required cli "--receipt")
    |> print_result mode
  else if mode = "resolve" then
    Loom_exec_result.resolve ~root ~store_root
      ~handle:(required cli "--handle") ~purpose:Loom_exec_result.Result_read
    |> print_result mode
  else if mode = "command-mismatch" then (
    let policy, decision = Loom_exec_result.command_mismatch_control ~root in
    Printf.printf
      "LOOM_EXEC_RESULT_STORE_CONTROL mode=command-mismatch semantic_authority=Sounio action=9033 manifest_sha256=%s decision=%s control_refused=true material_mutation=false exec_attached=false provider_hook_switched=false production_activation=false\n%!"
      policy.manifest_sha256 decision)
  else if mode = "promote-authority" then
    ignore
      (Loom_exec_result.resolve ~root ~store_root
         ~handle:(required cli "--handle")
         ~purpose:Loom_exec_result.Authority_promotion)
  else failf
      "exec-result-probe mode must be publish, resolve, command-mismatch, or promote-authority";
  0

let exec_intent_probe_command cli =
  if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
    failf "exec-intent-probe requires SOUNIO_LOOM_HOOK_TEST_MODE=1";
  let root = required cli "--root" |> Unix.realpath in
  let mode = required cli "--mode" in
  if mode = "project" then (
    let projection =
      Loom_exec_intent.project ~root
        ~raw_event_sha256:(required cli "--raw-event")
        ~command_sha256:(required cli "--command")
    in
    Printf.printf
      "LOOM_EXEC_INTENT_PROJECTION mode=project semantic_authority=Sounio action=9034 operational_kernel=OCaml manifest_sha256=%s source_sha256=%s executable_sha256=%s raw_event_sha256=%s event_sha256=%s command_sha256=%s authority_output_sha256=%s raw_event_is_semantic_identity=false ocaml_projection_attached=true provider_lifecycle_attached=false arbitrary_command_projection=false exec_attached=false production_activation=false\n%!"
      projection.manifest_sha256 projection.source_sha256
      projection.executable_sha256 projection.raw_event_sha256
      projection.event_sha256 projection.command_sha256
      projection.authority_output_sha256)
  else if mode = "command-mismatch" then (
    let policy, decision = Loom_exec_intent.command_mismatch_control ~root in
    Printf.printf
      "LOOM_EXEC_INTENT_PROJECTION_CONTROL mode=command-mismatch semantic_authority=Sounio action=9034 operational_kernel=OCaml manifest_sha256=%s decision=%s control_refused=true material_mutation=false ocaml_projection_attached=true provider_lifecycle_attached=false arbitrary_command_projection=false exec_attached=false production_activation=false\n%!"
      policy.manifest_sha256 decision)
  else failf "exec-intent-probe mode must be project or command-mismatch";
  0

let exec_catalog_probe_command cli =
  if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
    failf "exec-catalog-probe requires SOUNIO_LOOM_HOOK_TEST_MODE=1";
  let projection =
    Loom_exec_catalog.project
      ~root:(required cli "--root" |> Unix.realpath)
      ~operation:(required cli "--operation")
      ~source:(optional cli "--source")
  in
  Printf.printf
    "LOOM_EXEC_OPERATION_CATALOG_PROJECTION semantic_authority=Sounio action=9035 operational_kernel=OCaml operation=%s source_path=%s source_sha256=%s catalog_sha256=%s manifest_sha256=%s authority_source_sha256=%s authority_executable_sha256=%s semantic_event_sha256=%s command_template_sha256=%s argument_schema_sha256=%s result_schema_sha256=%s sandbox_profile_sha256=%s authority_output_sha256=%s arbitrary_shell=false ocaml_catalog_projection_attached=true host_payload_selection_attached=false provider_lifecycle_attached=false general_exec_attached=false production_activation=false\n%!"
    projection.operation
    (Option.value ~default:"-" projection.source_path)
    (Option.value ~default:"-" projection.source_sha256)
    projection.catalog_sha256 projection.manifest_sha256
    projection.authority_source_sha256 projection.authority_executable_sha256
    projection.semantic_event_sha256 projection.command_template_sha256
    projection.argument_schema_sha256 projection.result_schema_sha256
    projection.sandbox_profile_sha256 projection.authority_output_sha256;
  0

let exec_catalog_material_probe_command cli =
  if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
    failf "exec-catalog-material-probe requires SOUNIO_LOOM_HOOK_TEST_MODE=1";
  let result =
    Loom_exec_catalog.execute_sounio_check
      ~retain_captures:false
      ~root:(required cli "--root" |> Unix.realpath)
      ~source:(required cli "--source") ~output:(required cli "--output")
  in
  let plan = result.plan in
  let projection = plan.projection in
  Printf.printf
    "LOOM_EXEC_OPERATION_MATERIAL_RESULT semantic_authority=Sounio action=9035 operational_kernel=OCaml material_selector=OCaml operation=%s source_path=%s source_sha256=%s catalog_sha256=%s manifest_sha256=%s command_template_sha256=%s result_schema_sha256=%s sandbox_profile_sha256=%s compiler_path=%s compiler_sha256=%s argv_sha256=%s output_path=%s artifact_sha256=%s artifact_bytes=%d stdout_sha256=%s stderr_sha256=%s diagnostics_sha256=%s direct_exec=true shell=false artifact_executed=false direct_exec_material_plan_attached=true host_payload_selection_attached=false provider_lifecycle_attached=false general_exec_attached=false production_activation=false\n%!"
    projection.operation (Option.get projection.source_path)
    (Option.get projection.source_sha256) projection.catalog_sha256
    projection.manifest_sha256 projection.command_template_sha256
    projection.result_schema_sha256 projection.sandbox_profile_sha256
    plan.executable plan.compiler_sha256 plan.argv_sha256 plan.output_path
    result.artifact_sha256 result.artifact_bytes result.stdout_sha256
    result.stderr_sha256 result.diagnostics_sha256;
  0

let exec_result_record_probe_command cli =
  if Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" <> Some "1" then
    failf "exec-result-record-probe requires SOUNIO_LOOM_HOOK_TEST_MODE=1";
  let root = required cli "--root" |> Unix.realpath in
  let mode = required cli "--mode" in
  if mode = "artifact-binding" then (
    let policy, decision = Loom_exec_result_record.artifact_binding_control ~root in
    Printf.printf
      "LOOM_EXEC_RESULT_RECORD_CONTROL semantic_authority=Sounio action=9036 mode=artifact-binding manifest_sha256=%s decision=%s control_refused=true material_mutation=false ocaml_record_projection_attached=true dynamic_user_host_attached=false provider_result_returned=false production_activation=false\n%!"
      policy.manifest_sha256 decision;
    0)
  else if mode = "issue" then (
    let material =
      Loom_exec_catalog.execute_sounio_check ~retain_captures:false ~root
        ~source:(required cli "--source") ~output:(required cli "--output")
    in
    let binding : Loom_exec_result_record.binding =
      { event_sha256 = required cli "--event";
        generation_sha256 = required cli "--generation";
        principal_sha256 = required cli "--principal";
        descriptor_binding_sha256 = required cli "--descriptor-binding";
        grant_receipt_sha256 = required cli "--grant-receipt" }
    in
    let result = Loom_exec_result_record.issue ~root ~material ~binding in
    Printf.printf
      "LOOM_EXEC_RESULT_RECORD_PROJECTION semantic_authority=Sounio action=9036 operational_kernel=OCaml operation=sounio-check event_sha256=%s generation_sha256=%s source_sha256=%s artifact_sha256=%s record_sha256=%s handle=%s manifest_sha256=%s authority_output_sha256=%s handle_is_bearer=false handle_is_execution_authority=false artifact_executed=false ocaml_record_projection_attached=true dynamic_user_host_attached=false provider_result_returned=false production_activation=false\n%s%!"
      binding.event_sha256 binding.generation_sha256
      (Option.get material.plan.projection.source_sha256)
      material.artifact_sha256 result.record_sha256 result.handle
      result.manifest_sha256 result.authority_output_sha256 result.record;
    0)
  else failf "exec-result-record-probe mode must be issue or artifact-binding"

let exec_operation_cell_command cli =
  Loom_exec_operation_cell.run
    ~root:(required cli "--root") ~source:(required cli "--source")
    ~output_dir:(required cli "--output-dir") ~unit:(required cli "--unit")
    ~mode:(required cli "--mode")

let exec_result_present_command cli =
  let result =
    Loom_exec_result.validate_transport
      ~root:(required cli "--root")
      ~event_sha256:(required cli "--event")
      ~command_sha256:(required cli "--command")
      ~handle:(required cli "--handle")
      ~receipt_sha256:(required cli "--receipt-sha256")
      ~receipt_hex:(required cli "--receipt-hex")
      ~manifest_sha256:(required cli "--manifest-sha256")
  in
  print_string result.receipt;
  flush Stdlib.stdout;
  0

let exec_result_record_present_command cli =
  let result =
    Loom_exec_result_record.validate_transport
      ~root:(required cli "--root")
      ~event_sha256:(required cli "--event")
      ~command_sha256:(required cli "--command")
      ~handle:(required cli "--handle")
      ~record_sha256:(required cli "--record-sha256")
      ~record_hex:(required cli "--record-hex")
      ~manifest_sha256:(required cli "--manifest-sha256")
  in
  print_string result.record;
  flush Stdlib.stdout;
  0

let sovereign_result_command cli =
  Loom_sovereign_exec.present_result
    ~instance:(required cli "--instance")
    ~generation:(required cli "--generation")
    ~job_id:(required cli "--job")
    ~payload_sha256:(required cli "--payload-sha256")

let usage () =
  Printf.eprintf
    "Sounio Loom %s\n\nCommands:\n  agent-hook --agent codex|claude\n  exec-capability --instance I --generation G --handle H\n  subprocess-membrane-probe --root DIR --cwd DIR --scope DIR --deadline-ms N -- COMMAND... (test mode only)\n  resident-authority-probe --root DIR --mode happy|replay|mismatch|timeout|eof|finalize-eof|benchmark --frame FILE --deadline-ms N (test mode only)\n  invocation-cell-probe --root DIR --mode current|python|happy|abort|replay|mismatch|timeout|eof --prepare FILE [--admit FILE] [--close FILE] [--abort FILE] --deadline-ms N (test mode only)\n  exec-grant-cell-probe --root DIR --mode current|python|happy|deny-preserves|revoke|replay|mismatch|timeout|eof --issue FILE [--consume FILE] [--close FILE] [--revoke FILE] [--deny FILE] --deadline-ms N (test mode only)\n  lane-health-parity\n  start --agent A --lane L --session-id S --cwd DIR -- COMMAND...\n  recover --agent A --lane L --cwd DIR\n  status|guardian-status|stop|attach|observe|snapshot --agent A --lane L [options]\n  crash-kernel --agent A --lane L --at POINT\n  host-enroll --agent A --lane L [--replace] [--state-dir DIR]\n  host-reconcile [--agent A --lane L] [--apply] [--service-enabled] [--state-dir DIR]\n  host-supervise [--once] [--interval-seconds N] [--apply] [--service-enabled] [--state-dir DIR]\n  host-verify --agent A --lane L [--state-dir DIR]\n  provider-list [--json]\n  provider-status --provider P [--json]\n  provider-plan --provider P --session-id S --cwd DIR (--prompt TEXT|--prompt-file PATH) [--lifecycle turn|persistent] [--mode new|resume] [--provider-session S] [--model M] [--isolate-context] [--unsafe-auto] [--json]\n  provider-start --provider P --agent A --lane L --session-id S --cwd DIR (--prompt TEXT|--prompt-file PATH) [provider-plan options]\n  provider-open --provider claude|codex|kimi --agent A --lane L --session-id S --cwd DIR (--prompt TEXT|--prompt-file PATH) [--mode new|resume] [--provider-session S] [--model M] [--unsafe-auto]\n  provider-auth-login --provider P\n  obligation-open --message ID --message-digest SHA --from-agent A --from-lane L --to-agent A --to-lane L\n  obligation-consume --message ID --actor A --lane L --generation G [--ttl-seconds N]\n  obligation-claim|obligation-renew --message ID --actor A --lane L --generation G [--claim ID] [--ttl-seconds N]\n  obligation-interrupt --message ID --actor A --lane L --generation G [--claim ID] [--reason TEXT]\n  obligation-recover --message ID --actor A --lane L --generation G\n  obligation-complete --message ID --actor A --lane L --generation G --claim ID --outcome PATH --evidence PATH\n  obligation-status --message ID [--json]\n  obligation-list|obligation-tui [--json] [--state-dir DIR]\n  obligation-serve [--bind 127.0.0.1] [--port 8788] [--state-dir DIR]\n  obligation-verify --message ID\n  obligation-supervise [--once] [--interval-seconds N] [--state-dir DIR]\n  obligation-supervisor-status [--state-dir DIR]\n  journal-authority-serve --socket PATH --state-dir PATH --private-key PATH --public-key PATH --epoch N\n  journal-authority-status --socket PATH\n  fleet-enroll --slot S --kind K --home DIR --cwd DIR\n  fleet-disable --slot S --cwd DIR\n  fleet-reconcile [--apply] [--state-dir DIR]\n  list|tui|serve [--state-dir DIR]\n  beagle-serve [--bind 127.0.0.1] [--port 4372] [--state-dir DIR]\n  verify-journal|verify-guardian-journal --journal PATH\n  verify-continuity-receipt --receipt PATH --public-key PATH [--adapter PATH]\n  attest-continuity-receipt --receipt PATH --subject-public-key PATH --observer-private-key PATH --observer-public-key PATH --out PATH [--adapter PATH]\n  measure-continuity-generation --state-dir PATH --pane-id ID --generation ID --receipt PATH --subject-public-key PATH --observer-private-key PATH --observer-public-key PATH --out PATH [--adapter PATH]\n"
    runtime_version;
  Printf.eprintf
    "  provider-start accepts --wait to observe the turn until terminal state\n";
  Printf.eprintf
    "  exec-ingress-probe --root DIR --mode inherited|forged|missing|fixture-escape|result|result-binding|result-receipt|result-manifest --event FILE [--receipt FILE] (test mode only)\n";
  Printf.eprintf
    "  exec-result-probe --root DIR --store DIR --mode publish|resolve|command-mismatch|promote-authority [--receipt FILE] [--handle HANDLE] (test mode only)\n";
  Printf.eprintf
    "  exec-intent-probe --root DIR --mode project|command-mismatch --raw-event SHA --command SHA (test mode only)\n";
  Printf.eprintf
    "  exec-catalog-probe --root DIR --operation calibration|sounio-check [--source RELATIVE.sio] (test mode only)\n";
  Printf.eprintf
    "  exec-catalog-material-probe --root DIR --source RELATIVE.sio --output ABSOLUTE.elf (test mode only)\n";
  Printf.eprintf
    "  exec-result-record-probe --root DIR --mode issue|artifact-binding [issue bindings] (test mode only)\n";
  Printf.eprintf
    "  exec-result-present --root DIR --event SHA --command SHA --handle HANDLE --receipt-sha256 SHA --receipt-hex HEX --manifest-sha256 SHA\n";
  Printf.eprintf
    "  exec-result-record-present --root DIR --event SHA --command SHA --handle HANDLE --record-sha256 SHA --record-hex HEX --manifest-sha256 SHA\n";
  Printf.eprintf
    "  sovereign-result --instance I --generation SHA --job SHA --payload-sha256 SHA\n";
  Printf.eprintf
    "  peer-activation-capsule-probe --root DIR --mode current|python|happy|deny-preserves|poison|replay|mismatch|timeout|eof --seal FILE [--consume FILE] [--extinguish FILE] [--poison FILE] [--deny FILE] --deadline-ms N (test mode only)\n";
  Printf.eprintf "  provider-open persistent providers: claude, codex, kimi\n";
  Printf.eprintf
    "\nSpectral data plane:\n  export-events-arrow --out PATH [--state-dir DIR]\n  verify-events-arrow --file PATH\n";
  Printf.eprintf
    "\nEpistemic machine v0:\n  world-create --world W --agent A --lane L\n  knowledge-observe --world W --knowledge K --value V --error E --uncertainty U --confidence P --provenance SHA\n  epistemic-claim-open --world W --claim C --knowledge K --evidence SHA\n  epistemic-claim-challenge --world W --claim C --challenge X --falsifier SHA\n  epistemic-capability-acquire --world W --capability C --resource R --owner A --generation G\n  epistemic-capability-release --world W --capability C --owner A --generation G\n  world-fork --parent W --child W --agent A --lane L --hypothesis H [--parent-head SHA]\n  world-status|world-verify --world W\n  world-list\n";
  Printf.eprintf
    "\nCounterfactual Attention Compiler v0:\n  attention-compile --world W --plan P --candidates FILE --budget N --policy information-first|falsification-first|counterfactual-first --owner A --generation G\n  attention-complete --world W --plan P --owner A --generation G --outcome SHA\n";
  Printf.eprintf
    "\nPareto Portfolio Attention Compiler v0:\n  attention-portfolio-compile --world W --portfolio P --candidates FILE --token-budget N --wall-budget N --gpu-budget N --quota-budget N --policy information-first|falsification-first|counterfactual-first --owner A --generation G\n  attention-portfolio-complete --world W --portfolio P --owner A --generation G --outcome SHA\n";
  Printf.eprintf
    "\nRobust Contingent Policy Compiler v0:\n  contingent-policy-compile --world W --contingent-policy P --root-state S --actions FILE --outcomes FILE --token-budget N --wall-budget N --gpu-budget N --quota-budget N --order information-first|falsification-first|counterfactual-first --owner A --generation G [--measurement-principal M --measurement-public-key PEM --classifier-principal C --classifier-public-key PEM --classifier-spec-digest SHA]\n  contingent-measurement-attest --world W --contingent-policy P --measurement FILE --measurement-principal M --measurement-private-key PEM --measurement-nonce N --receipt FILE\n  contingent-classification-attest --world W --contingent-policy P --measurement-receipt FILE --outcome O --classifier-principal C --classifier-private-key PEM --receipt FILE\n  contingent-policy-observe-attested --world W --contingent-policy P --measurement-receipt FILE --classification-receipt FILE --owner A --generation G\n  contingent-policy-observe --world W --contingent-policy P --outcome O --owner A --generation G --outcome-digest SHA (legacy opaque policies only)\n";
  Printf.eprintf
    "\nWitness Mesh v0/v1:\n  witness-serve --witness-state-dir DIR --membership FILE --witness ID --private-key PEM [--bind IP] [--port N]\n  witness-mesh-anchor --state-dir DIR --world W --membership FILE --endpoints FILE --anchor-private-key PEM\n  witness-mesh-verify --state-dir DIR --world W --membership FILE --endpoints FILE [--policy byzantine-strict|crash-quorum]\n  witness-epoch-handoff --epoch-state-dir DIR --world W --from-epoch N --to-epoch N --old-state-dir DIR --old-membership FILE --old-endpoints FILE --new-state-dir DIR --new-membership FILE --new-endpoints FILE\n  witness-epoch-verify --epoch-state-dir DIR --world W --active-state-dir DIR --membership FILE --endpoints FILE\n  witness-epoch-log-serve --log-state-dir DIR --operator ID --operator-public-key PEM --operator-private-key PEM --publisher-public-key PEM [--bind IP] [--log-port N]\n  witness-epoch-log-status --log-host HOST --log-port N --operator ID --operator-public-key PEM --world W\n  witness-epoch-transparency-publish --epoch-state-dir DIR --transparency-state-dir DIR --world W --log-host HOST --log-port N --operator ID --operator-public-key PEM --publisher-public-key PEM --publisher-private-key PEM --transparency-membership FILE --transparency-endpoints FILE --transparency-anchor-private-key PEM\n  witness-epoch-transparency-verify --epoch-state-dir DIR --transparency-state-dir DIR --world W --log-host HOST --log-port N --operator ID --operator-public-key PEM --transparency-membership FILE --transparency-endpoints FILE\n";
  Printf.eprintf
    "\nFleet catalog v3:\n  fleet-enroll --slot S --kind K --home DIR --cwd DIR --custody agentd|loom [--agent A] [--session-id S] [--mode new|resume] [--provider-session S] [--coord-dir DIR] [--prompt TEXT|--prompt-file PATH] [--model M] [--unsafe-auto] [--adopt-active]\n  fleet-transfer --slot S --session-id S --provider-session S --source-lane L [--source-agent A] [--source-session S] --coord-dir DIR (--prompt TEXT|--prompt-file PATH) [--deadline-seconds N]\n  fleet-transfer-recover --slot S [--deadline-seconds N]\n  fleet-transfer-reset --slot S\n"

let durable_lane_canary_child () =
  if Sys.getenv_opt "SOUNIO_LOOM_DURABLE_LANE_CANARY" <> Some "1" then
    failf "durable lane canary is test-only";
  let boot_id = trim (read_file "/proc/sys/kernel/random/boot_id") in
  Printf.printf
    "LOOM_DURABLE_LANE_CHILD READY pid=%d start_tick=%s boot_id=%s language=OCaml role=MATERIAL_WITNESS semantic_authority=false\n%!"
    (Unix.getpid ()) (process_start (Unix.getpid ())) boot_id;
  let sequence = ref 0 in
  let running = ref true in
  while !running do
    match input_line Stdlib.stdin with
    | line when line = "LOOM_DURABLE_LANE_EXIT" ->
        Printf.printf
          "LOOM_DURABLE_LANE_CHILD EXIT sequence=%d semantic_authority=false\n%!"
          !sequence;
        running := false
    | line ->
        incr sequence;
        Printf.printf
          "LOOM_DURABLE_LANE_CHILD ACK sequence=%d input_sha256=%s pid=%d semantic_authority=false\n%!"
          !sequence (sha256 line) (Unix.getpid ())
    | exception End_of_file -> running := false
  done;
  0

let arguments_after_command () =
  let values = Array.to_list Sys.argv in
  match values with _program :: _command :: tail -> tail | _ -> []

let main () =
  if Array.length Sys.argv < 2 then (usage (); 2)
  else
    let command = Sys.argv.(1) in
    if command = "_provider-exec" then
      provider_exec_command (arguments_after_command ())
    else if command = "_provider-tui" then
      provider_tui_command (arguments_after_command ())
    else if command = "_durable-lane-canary" then
      durable_lane_canary_child ()
    else if command = "agent-hook" then
      Loom_hook.run (arguments_after_command ())
    else if command = "exec-capability" then
      Loom_exec.run (arguments_after_command ())
    else
      let booleans =
        [ "--no-raw"; "--meta"; "--machine"; "--allow-remote"; "--apply";
          "--replace"; "--adopt-active"; "--json"; "--once"; "--unsafe-auto";
          "--isolate-context"; "--service-enabled"; "--wait" ]
      in
      let cli = parse_cli booleans (arguments_after_command ()) in
      match command with
    | "runtime-version" ->
        Printf.printf "protocol_version=%d\nruntime_version=%s\nlanguage=OCaml\n" protocol_version runtime_version;
        0
    | "lane-health-parity" -> lane_health_parity_command (); 0
    | "subprocess-membrane-probe" -> subprocess_membrane_probe_command cli
    | "resident-authority-probe" -> resident_authority_probe_command cli
    | "invocation-cell-probe" -> invocation_cell_probe_command cli
    | "exec-grant-cell-probe" -> exec_grant_cell_probe_command cli
    | "exec-ingress-probe" -> exec_ingress_probe_command cli
    | "exec-intent-probe" -> exec_intent_probe_command cli
    | "exec-catalog-probe" -> exec_catalog_probe_command cli
    | "exec-catalog-material-probe" ->
        exec_catalog_material_probe_command cli
    | "exec-result-record-probe" -> exec_result_record_probe_command cli
    | "_exec-operation-cell" -> exec_operation_cell_command cli
    | "exec-result-probe" -> exec_result_probe_command cli
    | "exec-result-present" -> exec_result_present_command cli
    | "exec-result-record-present" -> exec_result_record_present_command cli
    | "sovereign-result" -> sovereign_result_command cli
    | "peer-activation-capsule-probe" ->
        peer_activation_capsule_probe_command cli
    | "start" -> start_command cli; 0
    | "recover" -> recover_command cli; 0
    | "status" -> status_command cli; 0
    | "guardian-status" -> guardian_status_command cli; 0
    | "wake" -> wake_command cli; 0
    | "crash-kernel" -> crash_kernel_command cli; 0
    | "host-enroll" -> host_enroll_command cli; 0
    | "host-reconcile" -> host_reconcile_command cli; 0
    | "host-supervise" -> host_supervise_command cli; 0
    | "host-verify" -> host_verify_command cli; 0
    | "provider-list" -> provider_list_command cli; 0
    | "provider-status" -> provider_status_command cli; 0
    | "provider-plan" -> provider_plan_command cli; 0
    | "provider-start" -> provider_start_command cli; 0
    | "provider-open" -> provider_open_command cli; 0
    | "provider-auth-login" -> provider_auth_login_command cli
    | "obligation-open" -> obligation_open_command cli; 0
    | "obligation-consume" -> obligation_consume_command cli; 0
    | "obligation-claim" -> obligation_claim_command cli; 0
    | "obligation-renew" -> obligation_renew_command cli; 0
    | "obligation-interrupt" -> obligation_interrupt_command cli; 0
    | "obligation-recover" -> obligation_recover_command cli; 0
    | "obligation-complete" -> obligation_complete_command cli; 0
    | "obligation-status" -> obligation_status_command cli; 0
    | "obligation-list" -> obligation_list_command cli; 0
    | "obligation-tui" -> obligation_tui_command cli; 0
    | "obligation-serve" -> obligation_serve_command cli; 0
    | "obligation-verify" -> obligation_verify_command cli; 0
    | "obligation-supervise" -> obligation_supervise_command cli; 0
    | "obligation-supervisor-status" -> obligation_supervisor_status_command cli; 0
    | "journal-authority-serve" -> journal_authority_serve_command cli; 0
    | "journal-authority-status" -> journal_authority_status_command cli; 0
    | "stop" -> stop_command cli; 0
    | "attach" -> stream_command cli true; 0
    | "observe" -> stream_command cli false; 0
    | "snapshot" -> snapshot_command cli; 0
    | "list" -> list_command cli; 0
    | "tui" -> tui_command cli; 0
    | "serve" -> serve_http cli; 0
    | "world-create" -> world_create_command cli; 0
    | "knowledge-observe" -> knowledge_observe_command cli; 0
    | "epistemic-claim-open" -> epistemic_claim_open_command cli; 0
    | "epistemic-claim-challenge" -> epistemic_claim_challenge_command cli; 0
    | "epistemic-capability-acquire" ->
        epistemic_capability_acquire_command cli; 0
    | "epistemic-capability-release" ->
        epistemic_capability_release_command cli; 0
    | "world-fork" -> world_fork_command cli; 0
    | "world-status" -> world_status_command cli; 0
    | "world-verify" -> world_verify_command cli; 0
    | "world-list" -> world_list_command cli; 0
    | "attention-compile" -> attention_compile_command cli; 0
    | "attention-complete" -> attention_complete_command cli; 0
    | "attention-portfolio-compile" ->
        attention_portfolio_compile_command cli; 0
    | "attention-portfolio-complete" ->
        attention_portfolio_complete_command cli; 0
    | "contingent-policy-compile" ->
        contingent_policy_compile_command cli; 0
    | "contingent-measurement-attest" ->
        contingent_measurement_attest_command cli; 0
    | "contingent-classification-attest" ->
        contingent_classification_attest_command cli; 0
    | "contingent-policy-observe-attested" ->
        contingent_policy_observe_attested_command cli; 0
    | "contingent-policy-observe" ->
        contingent_policy_observe_command cli; 0
    | "witness-serve" -> witness_serve_command cli; 0
    | "witness-mesh-anchor" -> witness_mesh_anchor_command cli; 0
    | "witness-mesh-verify" -> witness_mesh_verify_command cli; 0
    | "witness-epoch-handoff" -> witness_epoch_handoff_command cli; 0
    | "witness-epoch-verify" -> witness_epoch_verify_command cli; 0
    | "witness-epoch-log-serve" -> witness_epoch_log_serve_command cli; 0
    | "witness-epoch-log-status" -> witness_epoch_log_status_command cli; 0
    | "witness-epoch-transparency-publish" ->
        witness_epoch_transparency_publish_command cli; 0
    | "witness-epoch-transparency-verify" ->
        witness_epoch_transparency_verify_command cli; 0
    | "export-events-arrow" -> export_events_arrow_command cli; 0
    | "verify-events-arrow" -> verify_events_arrow_command cli; 0
    | "beagle-serve" -> serve_beagle_bridge cli; 0
    | "fleet-enroll" -> fleet_enroll_command cli; 0
    | "fleet-disable" -> fleet_disable_command cli; 0
    | "fleet-reconcile" -> fleet_reconcile_command cli; 0
    | "fleet-transfer" -> fleet_transfer_command cli; 0
    | "fleet-transfer-recover" -> fleet_transfer_recover_command cli; 0
    | "fleet-transfer-reset" -> fleet_transfer_reset_command cli; 0
    | "verify-journal" -> verify_command cli; 0
    | "verify-guardian-journal" -> verify_guardian_command cli; 0
    | "verify-continuity-receipt" -> verify_continuity_receipt_command cli; 0
    | "attest-continuity-receipt" -> attest_continuity_receipt_command cli; 0
    | "measure-continuity-generation" ->
        measure_continuity_generation_command cli; 0
    | "_forge-duplicate-lease" -> forge_duplicate_lease cli; 0
    | _ -> usage (); 2

let () =
  try exit (main ())
  with
  | Loom_error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_membrane.Error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_resident.Error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_effect_closure.Error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_invocation_cell.Error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_exec_grant_cell.Error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_exec_intent.Error error
  | Loom_exec_catalog.Error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_exec_operation_cell.Error error ->
      Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_exec_result_record.Error error ->
      Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_exec_result.Error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_sovereign_exec.Error error ->
      Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_peer_activation_capsule.Error error ->
      Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_epistemic.Error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_witness.Error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_witness_epoch.Error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Loom_witness_transparency.Error error ->
      Printf.eprintf "error: %s\n%!" error; exit 1
  | Sys_error error -> Printf.eprintf "error: %s\n%!" error; exit 1
  | Unix_error (error, function_name, argument) ->
      Printf.eprintf "error: %s: %s(%s)\n%!" (Unix.error_message error) function_name argument;
      exit 1
