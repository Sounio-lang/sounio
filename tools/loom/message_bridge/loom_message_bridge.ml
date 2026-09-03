open Unix

exception Bridge_error of string

let max_message_bytes = 16 * 1024
let max_http_bytes = 64 * 1024

let failf format = Printf.ksprintf (fun value -> raise (Bridge_error value)) format

let starts_with value prefix =
  String.length value >= String.length prefix
  && String.sub value 0 (String.length prefix) = prefix

let split_on character value = String.split_on_char character value
let trim value = String.trim value

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let read_file path =
  let channel = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () ->
      let length = in_channel_length channel in
      really_input_string channel length)

let sha256_file path = read_file path |> sha256

let write_all descriptor value =
  let bytes = Bytes.unsafe_of_string value in
  let rec write offset =
    if offset < Bytes.length bytes then
      let count = Unix.write descriptor bytes offset (Bytes.length bytes - offset) in
      if count = 0 then failf "short-write" else write (offset + count)
  in
  write 0

let hex_value character =
  match character with
  | '0' .. '9' -> Char.code character - Char.code '0'
  | 'a' .. 'f' -> 10 + Char.code character - Char.code 'a'
  | 'A' .. 'F' -> 10 + Char.code character - Char.code 'A'
  | _ -> failf "invalid-json:invalid-unicode-escape"

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

let rec json_render = function
  | Json_object fields ->
      "{" ^ String.concat ","
        (List.map (fun (name, value) -> json_quote name ^ ":" ^ json_render value) fields)
      ^ "}"
  | Json_array values ->
      "[" ^ String.concat "," (List.map json_render values) ^ "]"
  | Json_string value -> json_quote value
  | Json_number value -> value
  | Json_bool value -> if value then "true" else "false"
  | Json_null -> "null"

let rec json_canonical = function
  | Json_object fields ->
      "{" ^ String.concat ","
        (fields
         |> List.sort (fun (left, _) (right, _) -> String.compare left right)
         |> List.map (fun (name, value) ->
                json_quote name ^ ":" ^ json_canonical value))
      ^ "}"
  | Json_array values ->
      "[" ^ String.concat "," (List.map json_canonical values) ^ "]"
  | Json_string value -> json_quote value
  | Json_number value -> value
  | Json_bool value -> if value then "true" else "false"
  | Json_null -> "null"

let json_object_replace name value = function
  | Json_object fields ->
      let found = ref false in
      let fields =
        List.map
          (fun (key, previous) ->
            if key = name then (found := true; (key, value)) else (key, previous))
          fields
      in
      Json_object (if !found then fields else fields @ [ (name, value) ])
  | _ -> failf "invalid-json:expected-object"

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

let json_string_array_field object_value name =
  match json_object_field object_value name with
  | Some (Json_array values) ->
      List.map
        (function
          | Json_string value -> value
          | _ -> failf "invalid-json:%s-must-be-string-array" name)
        values
  | Some _ -> failf "invalid-json:%s-must-be-string-array" name
  | None -> failf "invalid-json:%s-is-required" name

let json_int_field object_value name =
  match json_object_field object_value name with
  | Some (Json_number value) -> (
      match int_of_string_opt value with
      | Some parsed when parsed >= 0 -> parsed
      | _ -> failf "invalid-json:%s-must-be-nonnegative-int" name)
  | Some _ -> failf "invalid-json:%s-must-be-nonnegative-int" name
  | None -> failf "invalid-json:%s-is-required" name

let constant_time_equal left right =
  let left_length = String.length left and right_length = String.length right in
  let length = max left_length right_length in
  let difference = ref (left_length lxor right_length) in
  for index = 0 to length - 1 do
    let left_code = if index < left_length then Char.code left.[index] else 0 in
    let right_code = if index < right_length then Char.code right.[index] else 0 in
    difference := !difference lor (left_code lxor right_code)
  done;
  !difference = 0

let valid_field label maximum value =
  if value = "" then failf "message-bridge-%s-empty" label;
  if String.length value > maximum then failf "message-bridge-%s-too-long" label;
  if String.contains value '\000' || String.contains value '\n'
     || String.contains value '\r'
  then failf "message-bridge-%s-invalid" label;
  value

let token_from_file path =
  if not (Sys.file_exists path) then failf "message-bridge-token-missing";
  if (Unix.lstat path).st_kind <> S_REG then
    failf "message-bridge-token-not-regular";
  let metadata = Unix.stat path in
  if metadata.st_uid <> Unix.getuid () then failf "message-bridge-token-owner-mismatch";
  if metadata.st_perm land 0o077 <> 0 then failf "message-bridge-token-permissions";
  let raw = read_file path in
  let token = trim raw in
  let canonical_text =
    raw = token || raw = token ^ "\n" || raw = token ^ "\r\n"
  in
  let has_whitespace =
    String.exists
      (function ' ' | '\t' | '\n' | '\r' | '\011' | '\012' -> true | _ -> false)
      token
  in
  if String.length token < 32 || String.length token > 256
     || String.contains token '\000' || has_whitespace
     || not canonical_text
  then failf "message-bridge-token-invalid";
  token

let parse_nonnegative label value =
  match int_of_string_opt value with
  | Some parsed when parsed >= 0 -> parsed
  | _ -> failf "invalid-%s" label

let find_string value needle start =
  let value_length = String.length value and needle_length = String.length needle in
  let rec loop index =
    if index + needle_length > value_length then None
    else if String.sub value index needle_length = needle then Some index
    else loop (index + 1)
  in
  loop start

let find_last_string value needle start =
  let rec loop index found =
    match find_string value needle index with
    | None -> found
    | Some next -> loop (next + 1) (Some next)
  in
  loop start None

let percent_decode value =
  let output = Buffer.create (String.length value) in
  let rec loop index =
    if index < String.length value then
      match value.[index] with
      | '%' when index + 2 < String.length value ->
          let decoded = (hex_value value.[index + 1] lsl 4) lor hex_value value.[index + 2] in
          Buffer.add_char output (Char.chr decoded);
          loop (index + 3)
      | '+' -> Buffer.add_char output ' '; loop (index + 1)
      | character -> Buffer.add_char output character; loop (index + 1)
  in
  loop 0;
  Buffer.contents output

let split_http_target target =
  match String.index_opt target '?' with
  | None -> (target, [])
  | Some index ->
      let path = String.sub target 0 index in
      let query =
        String.sub target (index + 1) (String.length target - index - 1)
        |> split_on '&'
        |> List.filter_map (fun pair ->
               match String.index_opt pair '=' with
               | None when pair <> "" -> Some (percent_decode pair, "")
               | None -> None
               | Some separator ->
                   Some
                     ( percent_decode (String.sub pair 0 separator),
                       percent_decode
                         (String.sub pair (separator + 1)
                            (String.length pair - separator - 1)) ))
      in
      (path, query)

let query_value query name = List.assoc_opt name query

type http_request = {
  http_method : string;
  http_target : string;
  http_headers : (string, string) Hashtbl.t;
  http_body : string;
}

let read_http_request descriptor =
  let buffer = Buffer.create 4096 in
  let bytes = Bytes.create 16384 in
  let rec read_headers () =
    match find_string (Buffer.contents buffer) "\r\n\r\n" 0 with
    | Some ending -> ending
    | None ->
        if Buffer.length buffer >= max_http_bytes then failf "http-request-too-large";
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
            String.sub line (index + 1) (String.length line - index - 1) |> trim
          in
          Hashtbl.replace headers name value)
    header_lines;
  if Hashtbl.mem headers "transfer-encoding" then
    failf "http-transfer-encoding-refused";
  let content_length =
    match Hashtbl.find_opt headers "content-length" with
    | None -> 0
    | Some value -> parse_nonnegative "content-length" value
  in
  if content_length > max_message_bytes * 2 then failf "http-body-too-large";
  let body_start = header_ending + 4 in
  let rec read_body () =
    if Buffer.length buffer - body_start >= content_length then ()
    else
      let count = Unix.read descriptor bytes 0 (Bytes.length bytes) in
      if count = 0 then failf "http-request-ended-before-body"
      else (Buffer.add_subbytes buffer bytes 0 count; read_body ())
  in
  read_body ();
  let raw = Buffer.contents buffer in
  { http_method; http_target; http_headers = headers;
    http_body = String.sub raw body_start content_length }

let response ?(headers = []) status body =
  let headers =
    ("Content-Type", "application/json; charset=utf-8")
    :: ("Content-Length", string_of_int (String.length body))
    :: ("Cache-Control", "no-store") :: ("Connection", "close") :: headers
  in
  Printf.sprintf "HTTP/1.1 %s\r\n%s\r\n\r\n%s" status
    (String.concat "\r\n"
       (List.map (fun (key, value) -> key ^ ": " ^ value) headers))
    body

let process_output_timeout ?(environment = Unix.environment ()) ?input
    ~timeout_seconds cwd command arguments =
  let reader, writer = Unix.pipe () in
  let input_pipe = Option.map (fun _ -> Unix.pipe ()) input in
  Unix.set_close_on_exec reader;
  match Unix.fork () with
  | 0 ->
      Unix.close reader;
      Option.iter
        (fun (input_reader, input_writer) ->
          Unix.close input_writer;
          Unix.dup2 input_reader Unix.stdin;
          if input_reader <> Unix.stdin then Unix.close input_reader)
        input_pipe;
      Unix.dup2 writer Unix.stdout;
      Unix.dup2 writer Unix.stderr;
      if writer <> Unix.stdout && writer <> Unix.stderr then Unix.close writer;
      ignore (Unix.setsid ());
      (try Unix.chdir cwd; Unix.execve command arguments environment
       with _ -> Unix._exit 127)
  | pid ->
      Unix.close writer;
      Option.iter
        (fun (input_reader, input_writer) ->
          Unix.close input_reader;
          Fun.protect ~finally:(fun () -> Unix.close input_writer) (fun () ->
              write_all input_writer (Option.get input)))
        input_pipe;
      let output = Buffer.create 4096 in
      let bytes = Bytes.create 16384 in
      let deadline = Unix.gettimeofday () +. float_of_int timeout_seconds in
      let rec read_until_deadline () =
        let remaining = deadline -. Unix.gettimeofday () in
        if remaining <= 0.0 then true
        else
          let readable, _, _ =
            try Unix.select [ reader ] [] [] remaining
            with Unix_error (EINTR, _, _) -> ([], [], [])
          in
          if readable = [] then
            if Unix.gettimeofday () >= deadline then true else read_until_deadline ()
          else
            match Unix.read reader bytes 0 (Bytes.length bytes) with
            | 0 -> false
            | count -> Buffer.add_subbytes output bytes 0 count; read_until_deadline ()
            | exception Unix_error (EINTR, _, _) -> read_until_deadline ()
      in
      let timed_out =
        Fun.protect ~finally:(fun () -> Unix.close reader) read_until_deadline
      in
      if timed_out then (try Unix.kill (-pid) Sys.sigkill with Unix_error _ -> ());
      let _, status = Unix.waitpid [] pid in
      let code =
        if timed_out then 128 + Sys.sigalrm
        else match status with
        | WEXITED value -> value
        | WSIGNALED signal | WSTOPPED signal -> 128 + signal
      in
      (code, Buffer.contents output)

let table_of_fields fields =
  let table = Hashtbl.create 16 in
  List.iter
    (fun field ->
      match String.index_opt field '=' with
      | None -> ()
      | Some index ->
          Hashtbl.replace table (String.sub field 0 index)
            (String.sub field (index + 1) (String.length field - index - 1)))
    fields;
  table

let coordination_command cwd =
  match Sys.getenv_opt "SOUNIO_COORD_COMMAND" with
  | Some path when Sys.file_exists path -> path
  | _ ->
      let sibling =
        Filename.concat (Filename.dirname Sys.executable_name) "sounio-coord-runtime"
      in
      if Sys.file_exists sibling then sibling
      else
        let launcher = Filename.concat (Filename.concat cwd "bin") "sounio-coord" in
        if Sys.file_exists launcher then launcher
        else failf "message-bridge-runtime-missing"

let run_coord cwd arguments =
  let command = coordination_command cwd in
  let argv = Array.of_list (command :: arguments) in
  let code, output = process_output_timeout ~timeout_seconds:8 cwd command argv in
  if code = 128 + Sys.sigalrm then failf "message-bridge-runtime-timeout";
  if code <> 0 then failf "message-bridge-runtime-refused:%s" (sha256 output);
  output

type bus_message = {
  id : string;
  utc : string;
  created_epoch : int;
  from_agent : string;
  from_lane : string;
  to_agent : string;
  to_lane : string;
  kind : string;
  text : string;
  thread_id : string;
  reply_to : string;
}

type bus_status = {
  request_state : string;
  injected : int;
  acknowledged : int;
  responses : int;
  wakes : int;
  wake_pending : int;
  created_epoch : int;
  injection_utc : string;
  acknowledgement_utc : string;
  wake_utc : string;
}

let lines_with_prefix output prefix =
  output |> split_on '\n' |> List.filter (fun line -> starts_with line prefix)

let fields_after_prefix line prefix =
  if not (starts_with line prefix) then failf "message-bridge-runtime-invalid-record";
  String.sub line (String.length prefix) (String.length line - String.length prefix)
  |> split_on ' ' |> table_of_fields

let table_value ?(default = "") fields name =
  Hashtbl.find_opt fields name |> Option.value ~default

let bus_message_of_line line =
  if not (starts_with line "MESSAGE ") then
    failf "message-bridge-runtime-invalid-message";
  let text_marker = " text=" and thread_marker = " thread=" in
  let text_at =
    find_string line text_marker 0
    |> Option.value ~default:(-1)
  in
  if text_at < 0 then failf "message-bridge-runtime-invalid-message";
  let body_start = text_at + String.length text_marker in
  let thread_at =
    find_last_string line thread_marker body_start
    |> Option.value ~default:(-1)
  in
  if thread_at < body_start then failf "message-bridge-runtime-invalid-message";
  let header = String.sub line 8 (text_at - 8) |> split_on ' ' in
  let tail =
    String.sub line (thread_at + 1) (String.length line - thread_at - 1)
    |> split_on ' '
  in
  let fields = table_of_fields (header @ tail) in
  let required name =
    let value = table_value fields name in
    if value = "" then failf "message-bridge-runtime-invalid-message";
    value
  in
  let optional name =
    match table_value ~default:"-" fields name with "-" -> "" | value -> value
  in
  { id = required "id"; utc = required "utc";
    created_epoch = parse_nonnegative "created-epoch" (required "created_epoch");
    from_agent = required "from_agent"; from_lane = required "from_lane";
    to_agent = optional "to_agent"; to_lane = optional "to_lane";
    kind = required "kind";
    text = String.sub line body_start (thread_at - body_start);
    thread_id = required "thread"; reply_to = optional "reply_to" }

let bus_messages output =
  lines_with_prefix output "MESSAGE " |> List.map bus_message_of_line

let first_record_utc output prefix =
  match lines_with_prefix output prefix with
  | line :: _ -> table_value (fields_after_prefix line prefix) "utc"
  | [] -> ""

let bus_status_of_output output =
  let line =
    match lines_with_prefix output "MESSAGE_STATUS " with
    | line :: _ -> line
    | [] -> failf "message-bridge-runtime-invalid-status"
  in
  let fields = fields_after_prefix line "MESSAGE_STATUS " in
  let count name = parse_nonnegative name (table_value ~default:"0" fields name) in
  { request_state = table_value ~default:"unknown" fields "request_state";
    injected = count "injected"; acknowledged = count "acknowledged";
    responses = count "responses"; wakes = count "wakes";
    wake_pending = count "wake_pending";
    created_epoch = count "created_epoch";
    injection_utc = first_record_utc output "INJECTION ";
    acknowledgement_utc = first_record_utc output "ACKNOWLEDGEMENT ";
    wake_utc = first_record_utc output "WAKE_RECEIPT " }

let message_json message =
  Printf.sprintf
    "{\"id\":%s,\"utc\":%s,\"createdEpoch\":%d,\"fromAgent\":%s,\"fromLane\":%s,\"toAgent\":%s,\"toLane\":%s,\"kind\":%s,\"text\":%s,\"threadId\":%s,\"replyTo\":%s}"
    (json_quote message.id) (json_quote message.utc) message.created_epoch
    (json_quote message.from_agent) (json_quote message.from_lane)
    (json_quote message.to_agent) (json_quote message.to_lane)
    (json_quote message.kind) (json_quote message.text)
    (json_quote message.thread_id) (json_quote message.reply_to)

let event_json ~id ~utc ~kind ~state ~message_id ~actor ~body =
  Printf.sprintf
    "{\"id\":%s,\"utc\":%s,\"kind\":%s,\"state\":%s,\"messageId\":%s,\"actor\":%s,\"body\":%s}"
    (json_quote id) (json_quote utc) (json_quote kind) (json_quote state)
    (json_quote message_id) (json_quote actor) (json_quote body)

let status_for cwd sender_agent sender_lane message_id =
  run_coord cwd
    [ "message-status"; "--agent"; sender_agent; "--lane"; sender_lane;
      "--message"; message_id ]
  |> bus_status_of_output

let sent_receipt output =
  let line =
    output |> split_on '\n'
    |> List.find_opt (fun value -> starts_with value "SENT ")
    |> Option.value ~default:""
  in
  if line = "" then failf "message-bridge-runtime-omitted-receipt";
  let fields = split_on ' ' line |> List.tl |> table_of_fields in
  let value name = Hashtbl.find_opt fields name |> Option.value ~default:"" in
  let message_id = value "message_id" and thread_id = value "thread_id" in
  if message_id = "" || thread_id = "" then
    failf "message-bridge-runtime-invalid-receipt";
  (message_id, thread_id)

let send_message cwd sender_agent sender_lane body =
  let parsed = parse_json body in
  let target_agent =
    json_string_field parsed [ "toAgent"; "to_agent" ]
    |> valid_field "target-agent" 256
  in
  let target_lane =
    json_string_field parsed [ "toLane"; "to_lane" ]
    |> valid_field "target-lane" 512
  in
  let message =
    json_string_field parsed [ "message" ]
    |> valid_field "message" max_message_bytes
  in
  let kind = json_string_field ~default:"request" parsed [ "kind" ] in
  if not (List.mem kind [ "info"; "request" ]) then
    failf "message-bridge-kind-refused";
  let output =
    run_coord cwd
      [ "send"; "--agent"; sender_agent; "--lane"; sender_lane;
        "--to-agent"; target_agent; "--to-lane"; target_lane; "--kind"; kind;
        "--message"; message ]
  in
  let message_id, thread_id = sent_receipt output in
  let wake_status =
    if output |> split_on '\n'
       |> List.exists (fun line -> starts_with line "WAKE_UNAVAILABLE ")
    then "durable_only" else "delivery_attempted"
  in
  Printf.sprintf
    "{\"schema\":\"loom-message-receipt-v1\",\"messageId\":%s,\"threadId\":%s,\"toAgent\":%s,\"toLane\":%s,\"kind\":%s,\"status\":\"accepted\",\"wakeStatus\":%s}"
    (json_quote message_id) (json_quote thread_id) (json_quote target_agent)
    (json_quote target_lane) (json_quote kind) (json_quote wake_status)

let list_threads cwd sender_agent sender_lane query =
  let limit =
    query_value query "limit" |> Option.value ~default:"20"
    |> parse_nonnegative "limit"
  in
  if limit < 1 || limit > 100 then failf "message-bridge-limit-invalid";
  let target_agent =
    query_value query "toAgent" |> Option.value ~default:""
  in
  let target_lane =
    query_value query "toLane" |> Option.value ~default:""
  in
  if target_agent <> "" then ignore (valid_field "target-agent" 256 target_agent);
  if target_lane <> "" then ignore (valid_field "target-lane" 512 target_lane);
  let arguments =
    [ "outbox"; "--agent"; sender_agent; "--lane"; sender_lane;
      "--newest-first"; "--limit"; string_of_int limit; "--kind"; "request" ]
    @ (if target_agent = "" then [] else [ "--to-agent"; target_agent ])
    @ (if target_lane = "" then [] else [ "--to-lane"; target_lane ])
  in
  let messages = run_coord cwd arguments |> bus_messages in
  Printf.sprintf
    "{\"schema\":\"loom-message-thread-list-v1\",\"senderAgent\":%s,\"senderLane\":%s,\"threads\":[%s]}"
    (json_quote sender_agent) (json_quote sender_lane)
    (messages |> List.map message_json |> String.concat ",")

let thread_detail cwd sender_agent sender_lane message_id query =
  let message_id = valid_field "message-id" 256 message_id in
  if not (starts_with message_id "msg-") then failf "message-bridge-message-id-invalid";
  let timeout_seconds =
    query_value query "timeoutSeconds" |> Option.value ~default:"60"
    |> parse_nonnegative "timeout-seconds"
  in
  if timeout_seconds > 86400 then failf "message-bridge-timeout-seconds-invalid";
  let outbound =
    run_coord cwd
      [ "outbox"; "--agent"; sender_agent; "--lane"; sender_lane;
        "--thread"; message_id ]
    |> bus_messages
  in
  let request =
    match List.find_opt (fun message -> message.id = message_id) outbound with
    | Some message -> message
    | None -> failf "message-bridge-thread-not-visible"
  in
  if request.kind <> "request" then failf "message-bridge-thread-not-request";
  let request_status = status_for cwd sender_agent sender_lane message_id in
  let responses =
    run_coord cwd
      [ "inbox"; "--agent"; sender_agent; "--lane"; sender_lane; "--all";
        "--directed-only"; "--thread"; request.thread_id ]
    |> bus_messages
  in
  let response_statuses =
    List.map
      (fun message -> (message, status_for cwd sender_agent sender_lane message.id))
      responses
  in
  let timed_out =
    request_status.request_state = "open"
    && int_of_float (Unix.time ()) >= request_status.created_epoch + timeout_seconds
  in
  let state = if timed_out then "timed_out" else request_status.request_state in
  let delivery =
    if request_status.wakes > 0 then "wake_received"
    else if request_status.wake_pending > 0 then "wake_pending"
    else if request_status.injected > 0 then "injected"
    else "durable_only"
  in
  let events = ref [] in
  let add event = events := event :: !events in
  add
    (event_json ~id:("request:" ^ request.id) ~utc:request.utc ~kind:"request"
       ~state:"accepted" ~message_id:request.id ~actor:sender_agent ~body:request.text);
  if delivery = "durable_only" then
    add
      (event_json ~id:("durable:" ^ request.id) ~utc:request.utc
         ~kind:"durable_only" ~state:"stored" ~message_id:request.id
         ~actor:"loom-bus" ~body:"No immediate delivery receipt; request remains durable.")
  else if delivery = "wake_pending" then
    add
      (event_json ~id:("wake-pending:" ^ request.id) ~utc:request.utc
         ~kind:"wake" ~state:"pending" ~message_id:request.id
         ~actor:"loom-delivery" ~body:"Immediate delivery was submitted and remains pending.")
  else if delivery = "wake_received" then
    add
      (event_json ~id:("wake:" ^ request.id)
         ~utc:(if request_status.wake_utc = "" then request.utc else request_status.wake_utc)
         ~kind:"wake" ~state:"received" ~message_id:request.id
         ~actor:"loom-delivery" ~body:"The delivery endpoint recorded the wake.")
  else ();
  if request_status.injected > 0 then
    add
      (event_json ~id:("injection:" ^ request.id)
         ~utc:(if request_status.injection_utc = "" then request.utc else request_status.injection_utc)
         ~kind:"injection" ~state:"injected" ~message_id:request.id
         ~actor:(request.to_agent ^ "/" ^ request.to_lane)
         ~body:"The target harness surfaced the request.");
  List.iter
    (fun (message, status) ->
      add
        (event_json ~id:("response:" ^ message.id) ~utc:message.utc
           ~kind:"response" ~state:message.kind ~message_id:message.id
           ~actor:(message.from_agent ^ "/" ^ message.from_lane) ~body:message.text);
      if status.acknowledged > 0 then
        add
          (event_json ~id:("ack:" ^ message.id)
             ~utc:(if status.acknowledgement_utc = "" then message.utc else status.acknowledgement_utc)
             ~kind:"ack" ~state:"acknowledged" ~message_id:message.id
             ~actor:(sender_agent ^ "/" ^ sender_lane)
             ~body:"The Loom client acknowledged the response."))
    response_statuses;
  if timed_out then
    add
      (event_json ~id:("timeout:" ^ request.id) ~utc:request.utc ~kind:"timeout"
         ~state:"elapsed" ~message_id:request.id ~actor:"loom-clock"
         ~body:(Printf.sprintf "No correlated response within %d seconds." timeout_seconds));
  Printf.sprintf
    "{\"schema\":\"loom-message-thread-v1\",\"request\":%s,\"state\":%s,\"delivery\":%s,\"injected\":%d,\"acknowledged\":%d,\"responseCount\":%d,\"wakeCount\":%d,\"wakePending\":%d,\"timeoutSeconds\":%d,\"events\":[%s]}"
    (message_json request) (json_quote state) (json_quote delivery)
    request_status.injected request_status.acknowledged request_status.responses
    request_status.wakes request_status.wake_pending timeout_seconds
    (!events |> List.rev |> String.concat ",")

let acknowledge_message cwd sender_agent sender_lane message_id =
  let message_id = valid_field "message-id" 256 message_id in
  if not (starts_with message_id "msg-") then failf "message-bridge-message-id-invalid";
  let output =
    run_coord cwd
      [ "ack"; "--agent"; sender_agent; "--lane"; sender_lane;
        "--message"; message_id ]
  in
  if not (lines_with_prefix output "ACKED " <> []) then
    failf "message-bridge-runtime-invalid-ack";
  Printf.sprintf
    "{\"schema\":\"loom-message-ack-v1\",\"messageId\":%s,\"status\":\"acknowledged\"}"
    (json_quote message_id)

type routing_config = {
  revision : int;
  updated_epoch : int;
  policy : string;
  model : string;
  effort : string;
  pool_order : string list;
  adapter_order : string list;
}

let default_routing_config =
  { revision = 0; updated_epoch = 0; policy = "authority-first";
    model = "gpt-5.6-terra"; effort = "high";
    pool_order = [ "pool-openai-team" ]; adapter_order = [ "adapter-codex" ] }

let routing_config_json config =
  let array values =
    "[" ^ String.concat "," (List.map json_quote values) ^ "]"
  in
  Printf.sprintf
    "{\"schema\":\"loom-routing-config-v1\",\"revision\":%d,\"updatedEpoch\":%d,\"policy\":%s,\"model\":%s,\"effort\":%s,\"poolOrder\":%s,\"adapterOrder\":%s}"
    config.revision config.updated_epoch (json_quote config.policy)
    (json_quote config.model) (json_quote config.effort) (array config.pool_order)
    (array config.adapter_order)

let valid_routing_identifier label value =
  let value = valid_field label 256 value in
  if not
       (String.for_all
          (function
            | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '.' | '_' | '-' -> true
            | _ -> false)
          value)
  then failf "message-bridge-routing-%s-invalid" label;
  value

let routing_slug value =
  String.map
    (fun character ->
      match character with
      | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '-' | '_' -> character
      | _ -> '-')
    value

let valid_routing_selector label allowed value =
  let value = valid_routing_identifier label value in
  if not (List.mem value allowed) then
    failf "message-bridge-routing-%s-refused" label;
  value

let validate_routing_order label values =
  if values = [] then failf "message-bridge-routing-%s-empty" label;
  if List.length values > 32 then failf "message-bridge-routing-%s-too-long" label;
  let validated = List.map (valid_routing_identifier label) values in
  if List.length (List.sort_uniq String.compare validated) <> List.length validated
  then failf "message-bridge-routing-%s-duplicate" label;
  validated

let routing_config_of_json ~persisted parsed =
  let schema = json_string_field parsed [ "schema" ] in
  if schema <> "loom-routing-config-v1" then
    failf "message-bridge-routing-schema-refused";
  let policy =
    json_string_field parsed [ "policy" ]
    |> valid_routing_selector "policy"
         [ "authority-first"; "capacity-aware"; "latency-aware" ]
  in
  let model =
    json_string_field parsed [ "model" ]
    |> valid_routing_selector "model" [ "gpt-5.6-terra"; "gpt-5.6-sol" ]
  in
  let effort =
    json_string_field parsed [ "effort" ]
    |> valid_routing_selector "effort" [ "low"; "medium"; "high" ]
  in
  let pool_order =
    json_string_array_field parsed "poolOrder" |> validate_routing_order "pool-order"
  in
  let adapter_order =
    json_string_array_field parsed "adapterOrder" |> validate_routing_order "adapter-order"
  in
  let revision, updated_epoch =
    if persisted then
      (json_int_field parsed "revision", json_int_field parsed "updatedEpoch")
    else (0, 0)
  in
  { revision; updated_epoch; policy; model; effort; pool_order; adapter_order }

let process_git cwd arguments =
  let argv = Array.of_list ("git" :: "-C" :: cwd :: arguments) in
  let code, output = process_output_timeout ~timeout_seconds:4 cwd "git" argv in
  if code <> 0 then failf "message-bridge-routing-git-refused:%s" (sha256 output);
  trim output

let routing_state_dir cwd configured =
  if configured <> "" then (
    if Filename.is_relative configured then
      failf "message-bridge-routing-state-must-be-absolute";
    configured)
  else
    let common_dir = process_git cwd [ "rev-parse"; "--git-common-dir" ] in
    if common_dir = "" then failf "message-bridge-routing-git-empty";
    let common_dir =
      if Filename.is_relative common_dir then Filename.concat cwd common_dir else common_dir
    in
    Filename.concat common_dir "sounio-loom-routing-state"

let rec ensure_private_directory path =
  if Sys.file_exists path then (
    if (Unix.lstat path).st_kind <> S_DIR then
      failf "message-bridge-routing-state-not-directory";
    let metadata = Unix.stat path in
    if metadata.st_uid <> Unix.getuid () then
      failf "message-bridge-routing-state-owner-mismatch";
    if metadata.st_perm land 0o077 <> 0 then
      failf "message-bridge-routing-state-permissions")
  else
    try Unix.mkdir path 0o700
    with Unix_error (EEXIST, _, _) -> ensure_private_directory path

let routing_paths cwd configured_state_dir =
  let directory = routing_state_dir cwd configured_state_dir in
  ensure_private_directory directory;
  (Filename.concat directory "routing-config-v1.json",
   Filename.concat directory "routing-config-v1.lock")

let read_routing_config path =
  if not (Sys.file_exists path) then default_routing_config
  else (
    if (Unix.lstat path).st_kind <> S_REG then
      failf "message-bridge-routing-config-not-regular";
    let metadata = Unix.stat path in
    if metadata.st_uid <> Unix.getuid () then
      failf "message-bridge-routing-config-owner-mismatch";
    if metadata.st_perm land 0o077 <> 0 then
      failf "message-bridge-routing-config-permissions";
    read_file path |> parse_json |> routing_config_of_json ~persisted:true)

let write_routing_config path config =
  let temporary = path ^ ".tmp-" ^ string_of_int (Unix.getpid ()) in
  if Sys.file_exists temporary then Unix.unlink temporary;
  let descriptor =
    Unix.openfile temporary [ O_WRONLY; O_CREAT; O_EXCL ] 0o600
  in
  Fun.protect
    ~finally:(fun () -> try Unix.close descriptor with Unix_error _ -> ())
    (fun () -> write_all descriptor (routing_config_json config ^ "\n"); Unix.fsync descriptor);
  Unix.rename temporary path

let with_routing_lock lock_path action =
  let descriptor = Unix.openfile lock_path [ O_RDWR; O_CREAT ] 0o600 in
  Fun.protect
    ~finally:(fun () -> try Unix.close descriptor with Unix_error _ -> ())
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      Fun.protect ~finally:(fun () -> Unix.lockf descriptor F_ULOCK 0) action)

let routing_config cwd configured_state_dir =
  let config_path, _ = routing_paths cwd configured_state_dir in
  read_routing_config config_path |> routing_config_json

let update_routing_config cwd configured_state_dir body =
  let update = parse_json body |> routing_config_of_json ~persisted:false in
  let config_path, lock_path = routing_paths cwd configured_state_dir in
  with_routing_lock lock_path (fun () ->
      let previous = read_routing_config config_path in
      let next =
        { update with revision = previous.revision + 1;
          updated_epoch = int_of_float (Unix.time ()) }
      in
      let previous_digest = sha256 (routing_config_json previous) in
      let config_json = routing_config_json next in
      write_routing_config config_path next;
      Printf.sprintf
        "{\"schema\":\"loom-routing-config-receipt-v1\",\"revision\":%d,\"updatedEpoch\":%d,\"previousDigest\":%s,\"digest\":%s,\"status\":\"stored\",\"config\":%s}"
        next.revision next.updated_epoch (json_quote previous_digest)
        (json_quote (sha256 config_json)) config_json)

type quota_observation = {
  quota_state_code : int;
  quota_state_name : string;
  pool_health_code : int;
  pool_health_name : string;
  used_percent : float option;
  resets_at : int option;
  observed_utc : string;
  observed_epoch : int;
  source_path : string;
  source_sha256 : string;
}

type adapter_observation = {
  adapter_health_code : int;
  adapter_health_name : string;
  provider_status_sha256 : string;
}

let json_bool_field object_value name =
  match json_object_field object_value name with
  | Some (Json_bool value) -> value
  | Some _ -> failf "invalid-json:%s-must-be-bool" name
  | None -> failf "invalid-json:%s-is-required" name

let json_float_value = function
  | Json_number value -> float_of_string_opt value
  | _ -> None

let json_int_value = function
  | Json_number value -> int_of_string_opt value
  | _ -> None

let child_object object_value name =
  match json_object_field object_value name with
  | Some (Json_object _ as value) -> Some value
  | _ -> None

let rec regular_files_under path =
  if not (Sys.file_exists path) then []
  else
    match (Unix.lstat path).st_kind with
    | S_REG -> [ path ]
    | S_DIR ->
        Sys.readdir path |> Array.to_list
        |> List.filter (fun name -> name <> "." && name <> "..")
        |> List.concat_map (fun name -> regular_files_under (Filename.concat path name))
    | _ -> []

let codex_sessions_root () =
  match Sys.getenv_opt "SOUNIO_LOOM_CODEX_SESSIONS_DIR" with
  | Some path when path <> "" -> path
  | _ ->
      let codex_home =
        match Sys.getenv_opt "CODEX_HOME" with
        | Some path when path <> "" -> path
        | _ -> Filename.concat (Sys.getenv_opt "HOME" |> Option.value ~default:"/") ".codex"
      in
      Filename.concat codex_home "sessions"

let rate_limit_from_line path file_epoch line =
  try
    let root = parse_json line in
    let timestamp = json_string_field root [ "timestamp" ] in
    let payload = child_object root "payload" |> Option.get in
    let limits = child_object payload "rate_limits" |> Option.get in
    let primary = child_object limits "primary" |> Option.get in
    let used_percent =
      Option.bind (json_object_field primary "used_percent") json_float_value
    in
    let resets_at =
      Option.bind (json_object_field primary "resets_at") json_int_value
    in
    match used_percent with
    | None -> None
    | Some used ->
        let exhausted = used >= 100.0 in
        Some
          { quota_state_code = 2; quota_state_name = "estimated";
            pool_health_code = (if exhausted then 3 else 1);
            pool_health_name = (if exhausted then "exhausted" else "healthy");
            used_percent = Some used; resets_at; observed_utc = timestamp;
            observed_epoch = file_epoch; source_path = path;
            source_sha256 = sha256 line }
  with _ -> None

let quota_observation () =
  let root = codex_sessions_root () in
  let candidates =
    regular_files_under root
    |> List.filter (fun path -> Filename.check_suffix path ".jsonl")
    |> List.map (fun path -> (int_of_float (Unix.stat path).st_mtime, path))
    |> List.sort (fun (left, _) (right, _) -> compare right left)
  in
  let rec inspect = function
    | [] -> None
    | (epoch, path) :: tail ->
        let observations =
          read_file path |> split_on '\n'
          |> List.filter_map (rate_limit_from_line path epoch)
        in
        (match List.rev observations with [] -> inspect tail | value :: _ -> Some value)
  in
  let now = int_of_float (Unix.time ()) in
  match inspect candidates with
  | Some observation when now - observation.observed_epoch <= 900 -> observation
  | _ ->
      { quota_state_code = 3; quota_state_name = "unknown";
        pool_health_code = 1; pool_health_name = "healthy";
        used_percent = None; resets_at = None; observed_utc = "";
        observed_epoch = 0; source_path = root; source_sha256 = "" }

let environment_with name value =
  let prefix = name ^ "=" in
  let base =
    Unix.environment () |> Array.to_list
    |> List.filter (fun entry -> not (starts_with entry prefix))
  in
  Array.of_list ((prefix ^ value) :: base)

let first_existing paths = List.find_opt Sys.file_exists paths

let installed_policy_root policy =
  Filename.concat
    (Filename.concat (Filename.dirname (Filename.dirname Sys.executable_name)) "policy")
    policy

let loom_command cwd =
  match Sys.getenv_opt "SOUNIO_LOOM_COMMAND" with
  | Some path when Sys.file_exists path -> path
  | _ ->
      first_existing
        [ Filename.concat cwd "tools/loom/.runtime/sounio-loom";
          Filename.concat cwd "tools/loom/_build/default/src/loom.exe";
          Filename.concat (Filename.dirname Sys.executable_name) "sounio-loom" ]
      |> Option.value ~default:""

let routing_authority_command cwd =
  match Sys.getenv_opt "SOUNIO_LOOM_ROUTING_AUTHORITY_RUNTIME" with
  | Some path when Sys.file_exists path -> path
  | _ ->
      first_existing
        [ Filename.concat cwd "tools/loom/_build/default/src/sounio-loom-routing-authority-runtime";
          Filename.concat cwd "tools/loom/.runtime/sounio-loom-routing-authority-runtime";
          Filename.concat (Filename.dirname Sys.executable_name)
            "sounio-loom-routing-authority-runtime" ]
      |> Option.value ~default:""

let manifest_fields path =
  if not (Sys.file_exists path) then failf "message-bridge-routing-manifest-missing";
  read_file path |> split_on '\n' |> table_of_fields

let verify_routing_freeze cwd authority =
  let policy_root, freeze_path =
    let installed = installed_policy_root "routing-authority" in
    let installed_freeze =
      Filename.concat installed "tools/loom/routing_authority.freeze.v1"
    in
    if Sys.file_exists installed_freeze then (installed, installed_freeze)
    else (cwd, Filename.concat cwd "tools/loom/routing_authority.freeze.v1")
  in
  let fields = manifest_fields freeze_path in
  let source_path = Filename.concat policy_root (table_value fields "source_path") in
  let entrypoint_path = Filename.concat policy_root (table_value fields "entrypoint_path") in
  if table_value fields "stage" <> "SEMANTICS_FROZEN"
     || table_value fields "producing_language" <> "Sounio"
     || table_value fields "language_role" <> "SEMANTIC_AUTHORITY"
  then failf "message-bridge-routing-freeze-invalid";
  let semantics_sha = sha256 (read_file source_path ^ read_file entrypoint_path) in
  if semantics_sha <> table_value fields "semantics_sha256" then
    failf "message-bridge-routing-semantics-drift";
  if sha256_file authority <> table_value fields "executable_sha256" then
    failf "message-bridge-routing-executable-drift";
  (semantics_sha, sha256_file source_path, freeze_path)

let verify_execution_product cwd =
  let policy_root, product_path =
    let installed = installed_policy_root "sovereign-execution" in
    let installed_product =
      Filename.concat installed
        "tools/loom/sovereign_execution_kernel_product.runtime.v1"
    in
    if Sys.file_exists installed_product then (installed, installed_product)
    else
      (cwd,
       Filename.concat cwd
         "tools/loom/sovereign_execution_kernel_product.runtime.v1")
  in
  let fields = manifest_fields product_path in
  let semantic_manifest =
    Filename.concat policy_root (table_value fields "semantic_manifest_path")
  in
  table_value fields "stage" = "PRODUCT_EXECUTION_FROZEN"
  && table_value fields "semantic_action" = "9042"
  && table_value fields "production_activation" = "true"
  && table_value fields "exec_attached" = "true"
  && table_value fields "same_uid_peer_isolation" = "true"
  && Sys.file_exists semantic_manifest
  && sha256_file semantic_manifest = table_value fields "semantic_manifest_sha256"

let authority_decision cwd authority fields =
  let frame = String.concat " " (List.map string_of_int fields) ^ "\n" in
  let code, output =
    process_output_timeout ~input:frame ~timeout_seconds:5 cwd authority
      [| authority |]
  in
  if code = 128 + Sys.sigalrm then failf "message-bridge-routing-authority-timeout";
  if code <> 0 then failf "message-bridge-routing-authority-error:%s" (sha256 output);
  let output = trim output in
  if starts_with output "SOUNIO_ROUTING_AUTHORITY_ALLOW " then
    (`Allow, "allow", output)
  else if starts_with output "SOUNIO_ROUTING_AUTHORITY_DENY " then
    let fields = fields_after_prefix output "SOUNIO_ROUTING_AUTHORITY_DENY " in
    (`Deny, table_value ~default:"authority-deny" fields "reason", output)
  else failf "message-bridge-routing-authority-invalid-output"

let adapter_observation cwd loom =
  if loom = "" then
    { adapter_health_code = 3; adapter_health_name = "missing";
      provider_status_sha256 = "" }
  else
    let code, output =
      process_output_timeout ~timeout_seconds:8 cwd loom
        [| loom; "provider-status"; "--provider"; "codex"; "--json" |]
    in
    if code <> 0 then
      { adapter_health_code = 2; adapter_health_name = "broken";
        provider_status_sha256 = sha256 output }
    else
      try
        let parsed = parse_json (trim output) in
        let status = child_object parsed "status" |> Option.get in
        let installed = json_bool_field status "installed" in
        let auth = json_string_field status [ "auth" ] in
        if not installed then
          { adapter_health_code = 3; adapter_health_name = "missing";
            provider_status_sha256 = sha256 output }
        else if auth <> "authenticated" then
          { adapter_health_code = 4; adapter_health_name = "auth_required";
            provider_status_sha256 = sha256 output }
        else
          { adapter_health_code = 1; adapter_health_name = "healthy";
            provider_status_sha256 = sha256 output }
      with _ ->
        { adapter_health_code = 2; adapter_health_name = "broken";
          provider_status_sha256 = sha256 output }

let route_frame ~operation ~policy_state ~ownership_state quota adapter
    ~model_available ~plan_bound ~plan_fresh ~plan_match ~provider_plan_bound
    ~execution_grant ~receipt_bound =
  [ 9032; 3; operation; policy_state; 1; 1; 1; 1; 1; 1; 0; 1; 1;
    ownership_state; quota.quota_state_code; 1; quota.pool_health_code;
    adapter.adapter_health_code; model_available; 9; 8; 6; 6; 0;
    plan_bound; plan_fresh; plan_match; provider_plan_bound; execution_grant;
    receipt_bound ]

let route_receipt_json ~task_id config ~reason ~status ~fallback_chain
    ~semantics_sha ~source_sha ~authority_output ~config_sha quota adapter
    ~provider_plan_sha ~session_id ~command_sha ~result =
  let optional_float = function None -> "null" | Some value -> Printf.sprintf "%.3f" value in
  let optional_int = function None -> "null" | Some value -> string_of_int value in
  Printf.sprintf
    "{\"schema\":\"loom-route-operation-v1\",\"decision\":{\"schema\":\"loom-route-decision-v1\",\"id\":%s,\"taskId\":%s,\"policy\":%s,\"candidateAdapterIds\":[%s],\"selectedAdapterId\":%s,\"authority\":\"Sounio\",\"authorityOutputSha256\":%s},\"receipt\":{\"schema\":\"loom-route-receipt-v1\",\"taskId\":%s,\"policy\":%s,\"poolId\":%s,\"adapterId\":%s,\"model\":%s,\"effort\":%s,\"reason\":%s,\"fallbackChain\":%s,\"status\":%s,\"sourceHash\":%s,\"semanticsHash\":%s,\"producingLanguage\":\"Sounio\",\"languageRole\":\"SEMANTIC_AUTHORITY\",\"operationalLanguage\":\"OCaml\",\"operationalRole\":\"OPERATIONAL_REALIZATION\",\"providerRole\":\"REVIEW_ONLY\",\"toolchain\":\"frozen-sounio-action-9032\",\"hardware\":%s,\"commandSha256\":%s,\"result\":%s,\"configHash\":%s,\"authorityOutputHash\":%s,\"providerPlanHash\":%s,\"quotaState\":%s,\"poolHealth\":%s,\"adapterHealth\":%s,\"quotaUsedPercent\":%s,\"quotaResetsAt\":%s,\"quotaObservedUtc\":%s,\"quotaObservationSource\":%s,\"quotaObservationHash\":%s,\"adapterObservationHash\":%s,\"sessionId\":%s}}"
    (json_quote (task_id ^ "-decision")) (json_quote task_id) (json_quote config.policy)
    (String.concat "," (List.map json_quote config.adapter_order))
    (json_quote (List.hd config.adapter_order)) (json_quote (sha256 authority_output))
    (json_quote task_id) (json_quote config.policy)
    (json_quote (List.hd config.pool_order)) (json_quote (List.hd config.adapter_order))
    (json_quote config.model) (json_quote config.effort) (json_quote reason)
    fallback_chain (json_quote status) (json_quote source_sha)
    (json_quote semantics_sha) (json_quote (Unix.gethostname ()))
    (json_quote command_sha) (json_quote result) (json_quote config_sha)
    (json_quote (sha256 authority_output)) (json_quote provider_plan_sha)
    (json_quote quota.quota_state_name) (json_quote quota.pool_health_name)
    (json_quote adapter.adapter_health_name) (optional_float quota.used_percent)
    (optional_int quota.resets_at) (json_quote quota.observed_utc)
    (json_quote quota.source_path) (json_quote quota.source_sha256)
    (json_quote adapter.provider_status_sha256) (json_quote session_id)

let write_private_file path contents =
  let temporary = path ^ ".tmp-" ^ string_of_int (Unix.getpid ()) in
  let descriptor = Unix.openfile temporary [ O_WRONLY; O_CREAT; O_EXCL ] 0o600 in
  Fun.protect
    ~finally:(fun () -> try Unix.close descriptor with Unix_error _ -> ())
    (fun () -> write_all descriptor contents; Unix.fsync descriptor);
  Unix.rename temporary path

let latest_route_operation_path cwd configured_state_dir =
  Filename.concat (routing_state_dir cwd configured_state_dir)
    "latest-route-operation-v1.json"

let latest_route_operation cwd configured_state_dir =
  let path = latest_route_operation_path cwd configured_state_dir in
  if Sys.file_exists path then
    Printf.sprintf "{\"schema\":\"loom-latest-route-operation-v1\",\"operation\":%s}"
      (trim (read_file path))
  else
    "{\"schema\":\"loom-latest-route-operation-v1\",\"operation\":null}"

let persist_route_operation receipt_path latest_path receipt =
  let contents = receipt ^ "\n" in
  write_private_file receipt_path contents;
  write_private_file latest_path contents

type route_task_paths = {
  receipt_path : string;
  request_hash_path : string;
  lock_path : string;
}

let route_task_paths cwd configured_state_dir task_id =
  let receipts_dir =
    Filename.concat (routing_state_dir cwd configured_state_dir) "receipts"
  in
  ensure_private_directory receipts_dir;
  let stem = routing_slug task_id ^ "-" ^ String.sub (sha256 task_id) 0 16 in
  { receipt_path = Filename.concat receipts_dir (stem ^ ".json");
    request_hash_path = Filename.concat receipts_dir (stem ^ ".request.sha256");
    lock_path = Filename.concat receipts_dir (stem ^ ".lock") }

let route_operation_status operation =
  match child_object operation "receipt" with
  | Some receipt -> json_string_field receipt [ "status" ]
  | None -> failf "message-bridge-routing-receipt-invalid"

let route_operation_with_status operation ~status ~reason ~result =
  match child_object operation "receipt" with
  | None -> failf "message-bridge-routing-receipt-invalid"
  | Some receipt ->
      let receipt =
        receipt
        |> json_object_replace "status" (Json_string status)
        |> json_object_replace "reason" (Json_string reason)
        |> json_object_replace "result" (Json_string result)
      in
      operation |> json_object_replace "receipt" receipt |> json_render

let terminal_route_status = function
  | "committed" | "completed" | "cancelled" | "refused" | "failed" -> true
  | _ -> false

let audit_route_phase task_id phase =
  Printf.eprintf "LOOM_ROUTE_PHASE task_id=%s phase=%s pid=%d epoch_ms=%.0f\n%!"
    (routing_slug task_id) phase (Unix.getpid ())
    (Unix.gettimeofday () *. 1000.0)

let route_task cwd configured_state_dir body =
  let parsed = parse_json body in
  if json_string_field parsed [ "schema" ] <> "loom-route-task-v1" then
    failf "message-bridge-routing-task-schema-refused";
  let task_id = json_string_field parsed [ "taskId" ] |> valid_routing_identifier "task-id" in
  let title = json_string_field parsed [ "title" ] |> valid_field "routing-title" 512 in
  let prompt = json_string_field parsed [ "prompt" ] |> valid_field "routing-prompt" 12000 in
  let kind = json_string_field ~default:"review" parsed [ "kind" ] in
  let ownership_state = if kind = "review" then 1 else 2 in
  let config_path, config_lock_path = routing_paths cwd configured_state_dir in
  let paths = route_task_paths cwd configured_state_dir task_id in
  let request_sha = sha256 (json_canonical parsed) in
  with_routing_lock paths.lock_path (fun () ->
      audit_route_phase task_id "task-lock-acquired";
      if Sys.file_exists paths.request_hash_path then (
        if trim (read_file paths.request_hash_path) <> request_sha then
          failf "message-bridge-routing-task-id-conflict";
        if Sys.file_exists paths.receipt_path then trim (read_file paths.receipt_path)
        else failf "message-bridge-routing-task-incomplete")
      else (
      write_private_file paths.request_hash_path (request_sha ^ "\n");
      let config =
        with_routing_lock config_lock_path (fun () -> read_routing_config config_path)
      in
      audit_route_phase task_id "config-snapshotted";
      let config_json = routing_config_json config in
      let config_sha = sha256 config_json in
      let authority = routing_authority_command cwd in
      if authority = "" then failf "message-bridge-routing-authority-missing";
      let semantics_sha, source_sha, _ = verify_routing_freeze cwd authority in
      let loom = loom_command cwd in
      let quota =
        if List.hd config.pool_order = "pool-openai-team" then quota_observation ()
        else
          { quota_state_code = 3; quota_state_name = "unknown";
            pool_health_code = 2; pool_health_name = "degraded";
            used_percent = None; resets_at = None; observed_utc = "";
            observed_epoch = 0; source_path = "unmapped-pool"; source_sha256 = "" }
      in
      let adapter =
        if List.hd config.adapter_order = "adapter-codex" then
          adapter_observation cwd loom
        else
          { adapter_health_code = 3; adapter_health_name = "missing";
            provider_status_sha256 = "" }
      in
      let pool_health_code =
        if adapter.adapter_health_code = 4 then 4 else quota.pool_health_code
      in
      let pool_health_name =
        if adapter.adapter_health_code = 4 then "auth_required" else quota.pool_health_name
      in
      let quota = { quota with pool_health_code; pool_health_name } in
      let policy_state = if config.revision > 0 then 1 else 0 in
      let plan_model_available, provider_plan, provider_plan_sha, session_id =
        if loom = "" || adapter.adapter_health_code <> 1 then (0, "", "", "")
        else
          let seed = sha256 (task_id ^ config_sha ^ string_of_float (Unix.time ())) in
          let session =
            Printf.sprintf "%s-%s-%s-%s-%s"
              (String.sub seed 0 8) (String.sub seed 8 4) (String.sub seed 12 4)
              (String.sub seed 16 4) (String.sub seed 20 12)
          in
          let review_prompt =
            "REVIEW_ONLY. Do not edit files, run commands, confirm semantic results, or claim authority.\n\n"
            ^ title ^ "\n\n" ^ prompt
          in
          let argv =
            [| loom; "provider-plan"; "--provider"; "codex"; "--session-id";
               session; "--cwd"; cwd; "--prompt"; review_prompt; "--model";
               config.model; "--effort"; config.effort; "--json" |]
          in
          let code, output = process_output_timeout ~timeout_seconds:10 cwd loom argv in
          audit_route_phase task_id "provider-plan-returned";
          if code = 0 then (1, output, sha256 output, session)
          else (0, output, sha256 output, session)
      in
      let plan_frame =
        route_frame ~operation:1 ~policy_state ~ownership_state quota adapter
          ~model_available:plan_model_available ~plan_bound:0 ~plan_fresh:0
          ~plan_match:0 ~provider_plan_bound:0 ~execution_grant:0 ~receipt_bound:0
      in
      let plan_decision, plan_reason, plan_output =
        authority_decision cwd authority plan_frame
      in
      audit_route_phase task_id "plan-authority-returned";
      let fallback_chain =
        "[" ^ String.concat "," (List.map json_quote config.adapter_order) ^ "]"
      in
      let latest_path = latest_route_operation_path cwd configured_state_dir in
      let receipt_path = paths.receipt_path in
      let command_sha = sha256 provider_plan in
      match plan_decision with
      | `Deny ->
          let receipt =
            route_receipt_json ~task_id config ~reason:plan_reason ~status:"refused"
              ~fallback_chain ~semantics_sha ~source_sha ~authority_output:plan_output
              ~config_sha quota adapter ~provider_plan_sha ~session_id ~command_sha
              ~result:"not-launched"
          in
          persist_route_operation receipt_path latest_path receipt;
          receipt
      | `Allow ->
          write_private_file receipt_path "{\"schema\":\"loom-route-receipt-pending-v1\"}\n";
          let config_unchanged =
            sha256 (routing_config_json (read_routing_config config_path)) = config_sha
          in
          let execution_grant = if verify_execution_product cwd then 1 else 0 in
          let dispatch_frame =
            route_frame ~operation:2 ~policy_state ~ownership_state quota adapter
              ~model_available:plan_model_available ~plan_bound:1
              ~plan_fresh:(if config_unchanged then 1 else 0) ~plan_match:1
              ~provider_plan_bound:(if provider_plan_sha <> "" then 1 else 0)
              ~execution_grant ~receipt_bound:1
          in
          let dispatch_decision, dispatch_reason, dispatch_output =
            authority_decision cwd authority dispatch_frame
          in
          audit_route_phase task_id "dispatch-authority-returned";
          (match dispatch_decision with
          | `Deny ->
              let receipt =
                route_receipt_json ~task_id config ~reason:dispatch_reason
                  ~status:"refused" ~fallback_chain ~semantics_sha ~source_sha
                  ~authority_output:dispatch_output ~config_sha quota adapter
                  ~provider_plan_sha ~session_id ~command_sha ~result:"not-launched"
              in
              persist_route_operation receipt_path latest_path receipt;
              receipt
          | `Allow ->
              let review_prompt =
                "REVIEW_ONLY. Do not edit files, run commands, confirm semantic results, or claim authority.\n\n"
                ^ title ^ "\n\n" ^ prompt
              in
              let argv =
                [| loom; "provider-start"; "--provider"; "codex"; "--agent";
                   "loom-route"; "--lane"; routing_slug task_id; "--session-id";
                   session_id; "--cwd"; cwd; "--prompt"; review_prompt; "--model";
                   config.model; "--effort"; config.effort |]
              in
              let code, output =
                audit_route_phase task_id "provider-start-entered";
                process_output_timeout
                  ~environment:(environment_with "SOUNIO_LOOM_SOVEREIGN_EXEC_REQUIRED" "1")
                  ~timeout_seconds:60 cwd loom argv
              in
              audit_route_phase task_id "provider-start-returned";
              let launched = code = 0 in
              let receipt =
                route_receipt_json ~task_id config
                  ~reason:(if launched then "authorized-adapter-launched" else "adapter-launch-failed")
                  ~status:(if launched then "running" else "failed") ~fallback_chain
                  ~semantics_sha ~source_sha ~authority_output:dispatch_output
                  ~config_sha quota adapter ~provider_plan_sha ~session_id
                  ~command_sha:(sha256 (String.concat "\000" (Array.to_list argv)))
                  ~result:(if launched then "provider-custody-started" else "provider-start-refused:" ^ sha256 output)
              in
              persist_route_operation receipt_path latest_path receipt;
              receipt)))

let read_route_operation paths =
  if not (Sys.file_exists paths.receipt_path) then
    failf "message-bridge-routing-task-not-found";
  let operation = trim (read_file paths.receipt_path) |> parse_json in
  if json_string_field operation [ "schema" ] <> "loom-route-operation-v1" then
    failf "message-bridge-routing-receipt-invalid";
  operation

let persist_route_status cwd configured_state_dir paths operation =
  let receipt = json_render operation in
  persist_route_operation paths.receipt_path
    (latest_route_operation_path cwd configured_state_dir) receipt;
  receipt

let provider_session_state cwd task_id =
  let loom = loom_command cwd in
  if loom = "" then failf "message-bridge-routing-provider-runtime-missing";
  let argv =
    [| loom; "status"; "--agent"; "loom-route"; "--lane";
       routing_slug task_id; "--machine" |]
  in
  let code, output = process_output_timeout ~timeout_seconds:4 cwd loom argv in
  if code = 128 + Sys.sigalrm then
    failf "message-bridge-routing-status-timeout";
  if code = 0 then
    table_of_fields (split_on '\n' output) |> fun fields ->
    table_value ~default:"unknown" fields "state"
  else
    let list_argv = [| loom; "list"; "--cwd"; cwd |] in
    let list_code, list_output =
      process_output_timeout ~timeout_seconds:4 cwd loom list_argv
    in
    if list_code <> 0 then "unknown"
    else
      lines_with_prefix list_output "LOOM_SESSION "
      |> List.find_map (fun line ->
             let fields = fields_after_prefix line "LOOM_SESSION " in
             if table_value fields "agent" = "loom-route"
                && table_value fields "lane" = routing_slug task_id
             then Some (table_value ~default:"unknown" fields "state")
             else None)
      |> Option.value ~default:"unknown"

let route_task_status cwd configured_state_dir task_id =
  let task_id = valid_routing_identifier "task-id" task_id in
  let paths = route_task_paths cwd configured_state_dir task_id in
  with_routing_lock paths.lock_path (fun () ->
      let operation = read_route_operation paths in
      if route_operation_status operation <> "running" then json_render operation
      else
        match provider_session_state cwd task_id with
        | "exited" ->
            route_operation_with_status operation ~status:"completed"
              ~reason:"provider-turn-completed"
              ~result:"provider-custody-terminal"
            |> parse_json
            |> persist_route_status cwd configured_state_dir paths
        | _ -> json_render operation)

let cancel_route_task cwd configured_state_dir task_id =
  let task_id = valid_routing_identifier "task-id" task_id in
  let paths = route_task_paths cwd configured_state_dir task_id in
  with_routing_lock paths.lock_path (fun () ->
      let operation = read_route_operation paths in
      let status = route_operation_status operation in
      if terminal_route_status status then json_render operation
      else if status <> "running" then
        failf "message-bridge-routing-task-not-cancellable"
      else
        let loom = loom_command cwd in
        if loom = "" then failf "message-bridge-routing-provider-runtime-missing";
        let argv =
          [| loom; "stop"; "--agent"; "loom-route"; "--lane";
             routing_slug task_id |]
        in
        let code, output = process_output_timeout ~timeout_seconds:8 cwd loom argv in
        if code = 128 + Sys.sigalrm then
          failf "message-bridge-routing-cancel-timeout";
        if code <> 0 then
          failf "message-bridge-routing-cancel-refused:%s" (sha256 output);
        route_operation_with_status operation ~status:"cancelled"
          ~reason:"operator-cancelled" ~result:"provider-stop-requested"
        |> parse_json
        |> persist_route_status cwd configured_state_dir paths)

let authorized request token =
  match Hashtbl.find_opt request.http_headers "authorization" with
  | Some value when starts_with value "Bearer " ->
      constant_time_equal
        (String.sub value 7 (String.length value - 7)) token
  | _ -> false

let reason_slug value =
  String.map
    (fun character ->
      match character with
      | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '-' | '_' | ':' -> character
      | _ -> '-')
    value

let routing_task_target path =
  let prefix = "/v1/routing/tasks/" in
  if not (starts_with path prefix) then None
  else
    let suffix = String.sub path (String.length prefix)
        (String.length path - String.length prefix) in
    if suffix = "" then None
    else if String.length suffix > 7
            && String.sub suffix (String.length suffix - 7) 7 = "/cancel"
    then
      Some (`Cancel,
        String.sub suffix 0 (String.length suffix - 7) |> percent_decode)
    else Some (`Status, percent_decode suffix)

type operation_class = Priority | Background

let operation_class method_name path =
  if starts_with path "/v1/threads" then Background
  else if method_name = "GET" && path = "/health" then Priority
  else Priority

let operation_class_name = function Priority -> "priority" | Background -> "background"

let environment_positive name default =
  match Sys.getenv_opt name with
  | None -> default
  | Some value ->
      let parsed = parse_nonnegative name value in
      if parsed < 1 || parsed > 64 then failf "%s-out-of-range" name;
      parsed

let with_operation_slot cwd configured_state_dir operation_class action =
  let slots =
    match operation_class with
    | Priority -> environment_positive "SOUNIO_LOOM_PRIORITY_WORKERS" 4
    | Background -> environment_positive "SOUNIO_LOOM_BACKGROUND_WORKERS" 2
  in
  let wait_seconds = match operation_class with Priority -> 5.0 | Background -> 1.0 in
  let state_directory =
    if configured_state_dir <> "" then configured_state_dir
    else
      Filename.concat (Filename.get_temp_dir_name ())
        (Printf.sprintf "sounio-loom-message-bridge-%d-%s"
           (Unix.getuid ()) (String.sub (sha256 cwd) 0 16))
  in
  ensure_private_directory state_directory;
  let directory = Filename.concat state_directory "admission" in
  ensure_private_directory directory;
  let deadline = Unix.gettimeofday () +. wait_seconds in
  let rec acquire index =
    if Unix.gettimeofday () >= deadline then
      failf "message-bridge-%s-queue-timeout" (operation_class_name operation_class);
    let slot = index mod slots in
    let path =
      Filename.concat directory
        (Printf.sprintf "%s-%d.lock" (operation_class_name operation_class) slot)
    in
    let descriptor = Unix.openfile path [ O_RDWR; O_CREAT ] 0o600 in
    try
      Unix.lockf descriptor F_TLOCK 0;
      descriptor
    with
    | Unix_error ((EAGAIN | EACCES), _, _) ->
        Unix.close descriptor;
        if slot = slots - 1 then Unix.sleepf 0.01;
        acquire (index + 1)
    | error -> Unix.close descriptor; raise error
  in
  let descriptor = acquire 0 in
  Printf.eprintf "LOOM_MESSAGE_QUEUE class=%s event=acquired\n%!"
    (operation_class_name operation_class);
  Fun.protect
    ~finally:(fun () ->
      (try Unix.lockf descriptor F_ULOCK 0 with Unix_error _ -> ());
      (try Unix.close descriptor with Unix_error _ -> ()))
    action

let handle cwd routing_state_dir token sender_agent sender_lane descriptor =
  let respond status body =
    write_all descriptor
      (response ~headers:[ ("X-Loom-Authority", "durable-message-bus") ]
         status body)
  in
  try
    let request = read_http_request descriptor in
    let path, query = split_http_target request.http_target in
    let operation_class = operation_class request.http_method path in
    let with_admission action =
      if path = "/health" || authorized request token then (
        Printf.eprintf
          "LOOM_MESSAGE_QUEUE class=%s event=enqueue sender_agent=%s sender_lane=%s\n%!"
          (operation_class_name operation_class) sender_agent sender_lane;
        with_operation_slot cwd routing_state_dir operation_class action
      )
      else action ()
    in
    with_admission (fun () ->
    if request.http_method = "GET" && path = "/health" then
      respond "200 OK"
        "{\"schema\":\"loom-message-bridge-v1\",\"status\":\"ready\",\"authentication\":\"bearer-capability\"}"
    else if not (authorized request token) then (
      Printf.eprintf
        "LOOM_MESSAGE_DECISION decision=DENY reason=invalid-capability sender_agent=%s sender_lane=%s\n%!"
        sender_agent sender_lane;
      respond "401 Unauthorized" "{\"error\":\"unauthorized\"}")
    else if request.http_method = "POST" && path = "/v1/messages" then
      let receipt = send_message cwd sender_agent sender_lane request.http_body in
      Printf.eprintf
        "LOOM_MESSAGE_DECISION decision=ALLOW reason=durable-bus-accepted sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
        sender_agent sender_lane (sha256 receipt);
      respond "202 Accepted" receipt
    else if request.http_method = "GET" && path = "/v1/threads" then
      let projection = list_threads cwd sender_agent sender_lane query in
      Printf.eprintf
        "LOOM_MESSAGE_DECISION decision=ALLOW reason=thread-list-read sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
        sender_agent sender_lane (sha256 projection);
      respond "200 OK" projection
    else if request.http_method = "GET" && path = "/v1/routing/config" then
      let projection = routing_config cwd routing_state_dir in
      Printf.eprintf
        "LOOM_MESSAGE_DECISION decision=ALLOW reason=routing-config-read sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
        sender_agent sender_lane (sha256 projection);
      respond "200 OK" projection
    else if request.http_method = "GET" && path = "/v1/routing/receipts/latest" then
      let projection = latest_route_operation cwd routing_state_dir in
      Printf.eprintf
        "LOOM_MESSAGE_DECISION decision=ALLOW reason=routing-latest-read sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
        sender_agent sender_lane (sha256 projection);
      respond "200 OK" projection
    else if request.http_method = "PUT" && path = "/v1/routing/config" then
      let receipt = update_routing_config cwd routing_state_dir request.http_body in
      Printf.eprintf
        "LOOM_MESSAGE_DECISION decision=ALLOW reason=routing-config-stored sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
        sender_agent sender_lane (sha256 receipt);
      respond "200 OK" receipt
    else if request.http_method = "POST" && path = "/v1/routing/tasks" then
      let receipt = route_task cwd routing_state_dir request.http_body in
      let receipt_status =
        try
          let parsed_receipt = child_object (parse_json receipt) "receipt" |> Option.get in
          json_string_field parsed_receipt [ "status" ]
        with _ -> "invalid"
      in
      let decision =
        if receipt_status = "refused" || receipt_status = "failed" then "DENY"
        else "ALLOW"
      in
      Printf.eprintf
        "LOOM_MESSAGE_DECISION decision=%s reason=routing-task-%s sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
        decision (reason_slug receipt_status) sender_agent sender_lane (sha256 receipt);
      respond "200 OK" receipt
    else if request.http_method = "GET" then
      (match routing_task_target path with
      | Some (`Status, task_id) ->
          let receipt = route_task_status cwd routing_state_dir task_id in
          Printf.eprintf
            "LOOM_MESSAGE_DECISION decision=ALLOW reason=routing-task-status sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
            sender_agent sender_lane (sha256 receipt);
          respond "200 OK" receipt
      | _ ->
          if starts_with path "/v1/threads/" then
            let message_id =
              String.sub path 12 (String.length path - 12) |> percent_decode
            in
            let projection = thread_detail cwd sender_agent sender_lane message_id query in
            Printf.eprintf
              "LOOM_MESSAGE_DECISION decision=ALLOW reason=thread-detail-read sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
              sender_agent sender_lane (sha256 projection);
            respond "200 OK" projection
          else respond "404 Not Found" "{\"error\":\"not_found\"}")
    else if request.http_method = "POST" then
      (match routing_task_target path with
      | Some (`Cancel, task_id) ->
          let receipt = cancel_route_task cwd routing_state_dir task_id in
          Printf.eprintf
            "LOOM_MESSAGE_DECISION decision=ALLOW reason=routing-task-cancelled sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
            sender_agent sender_lane (sha256 receipt);
          respond "200 OK" receipt
      | _ when starts_with path "/v1/messages/"
               && String.length path > 17
               && String.sub path (String.length path - 4) 4 = "/ack" ->
          let message_id =
            String.sub path 13 (String.length path - 17) |> percent_decode
          in
          let receipt = acknowledge_message cwd sender_agent sender_lane message_id in
          Printf.eprintf
            "LOOM_MESSAGE_DECISION decision=ALLOW reason=message-acknowledged sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
            sender_agent sender_lane (sha256 receipt);
          respond "200 OK" receipt
      | _ -> respond "404 Not Found" "{\"error\":\"not_found\"}")
    else respond "404 Not Found" "{\"error\":\"not_found\"}")
  with
  | Bridge_error message ->
      let status =
        if message = "message-bridge-routing-task-not-found" then
          "404 Not Found"
        else if message = "message-bridge-routing-task-incomplete"
             || starts_with message "message-bridge-routing-authority-"
             || message = "message-bridge-routing-provider-runtime-missing"
             || message = "message-bridge-routing-status-timeout"
             || message = "message-bridge-routing-cancel-timeout"
        then "503 Service Unavailable"
        else if message = "message-bridge-routing-task-id-conflict"
             || message = "message-bridge-routing-task-not-cancellable"
        then "409 Conflict"
        else if starts_with message "message-bridge-priority-queue-timeout"
             || starts_with message "message-bridge-background-queue-timeout"
        then "503 Service Unavailable"
        else if starts_with message "invalid-json:"
           || starts_with message "message-bridge-target-"
           || starts_with message "message-bridge-message-"
           || starts_with message "message-bridge-routing-"
           || message = "message-bridge-kind-refused"
        then "400 Bad Request"
        else if starts_with message "message-bridge-runtime-" then
          "503 Service Unavailable"
        else "500 Internal Server Error"
      in
      Printf.eprintf
        "LOOM_MESSAGE_DECISION decision=DENY reason=%s sender_agent=%s sender_lane=%s\n%!"
        (reason_slug message) sender_agent sender_lane;
      (try respond status (Printf.sprintf "{\"error\":%s}" (json_quote message))
       with _ -> ())
  | error ->
      Printf.eprintf
        "LOOM_MESSAGE_DECISION decision=DENY reason=internal-error sender_agent=%s sender_lane=%s detail_sha256=%s\n%!"
        sender_agent sender_lane (sha256 (Printexc.to_string error));
      (try respond "500 Internal Server Error" "{\"error\":\"internal_error\"}"
       with _ -> ())

type cli = {
  options : (string, string) Hashtbl.t;
  flags : (string, bool) Hashtbl.t;
}

let parse_cli arguments =
  let options = Hashtbl.create 16 and flags = Hashtbl.create 4 in
  let rec loop = function
    | [] -> { options; flags }
    | "--allow-remote" :: tail ->
        Hashtbl.replace flags "--allow-remote" true;
        loop tail
    | key :: value :: tail when starts_with key "--" ->
        Hashtbl.replace options key value;
        loop tail
    | key :: [] when starts_with key "--" -> failf "%s-requires-value" key
    | value :: _ -> failf "unexpected-argument:%s" value
  in
  loop arguments

let option cli name default =
  Hashtbl.find_opt cli.options name |> Option.value ~default

let required cli name =
  match Hashtbl.find_opt cli.options name with
  | Some value -> value
  | None -> failf "%s-is-required" name

let serve cli =
  let cwd = option cli "--cwd" (Unix.getcwd ()) |> Unix.realpath in
  let bind = option cli "--bind" "127.0.0.1" in
  if bind <> "127.0.0.1" && bind <> "localhost"
     && not (Hashtbl.mem cli.flags "--allow-remote")
  then failf "remote message bridge bind requires --allow-remote";
  let port = option cli "--port" "8789" |> parse_nonnegative "port" in
  if port > 65535 then failf "invalid-port";
  let token_path = required cli "--token-file" in
  let token = token_from_file token_path in
  let routing_state_dir = option cli "--routing-state-dir" "" in
  let sender_agent =
    option cli "--agent" "loom-ui" |> valid_field "sender-agent" 256
  in
  let sender_lane =
    option cli "--lane" "apple-client" |> valid_field "sender-lane" 512
  in
  ignore (coordination_command cwd);
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
  Sys.set_signal Sys.sigpipe Sys.Signal_ignore;
  Sys.set_signal Sys.sigchld Sys.Signal_default;
  let max_children = environment_positive "SOUNIO_LOOM_MAX_HTTP_WORKERS" 32 in
  let children = ref 0 in
  let rec reap_children () =
    match Unix.waitpid [ WNOHANG ] (-1) with
    | 0, _ -> ()
    | _, _ -> decr children; reap_children ()
    | exception Unix_error (ECHILD, _, _) -> children := 0
    | exception Unix_error (EINTR, _, _) -> reap_children ()
  in
  Printf.printf
    "LOOM_MESSAGE_BRIDGE url=http://%s:%d schema=loom-message-bridge-v1 auth=bearer-capability sender_agent=%s sender_lane=%s\n%!"
    bind actual_port sender_agent sender_lane;
  while !running do
    reap_children ();
    let readable, _, _ =
      try
        if !children >= max_children then (Unix.sleepf 0.01; ([], [], []))
        else Unix.select [ server ] [] [] 0.25
      with Unix_error (EINTR, _, _) -> ([], [], [])
    in
    if readable <> [] then
      let client, _ = Unix.accept server in
      match Unix.fork () with
      | 0 ->
          Unix.close server;
          Sys.set_signal Sys.sigchld Sys.Signal_default;
          Unix.setsockopt_float client SO_RCVTIMEO 5.0;
          handle cwd routing_state_dir token sender_agent sender_lane client;
          Unix.close client;
          Unix._exit 0
      | _ -> incr children; Unix.close client
  done;
  Unix.close server

let usage () =
  Printf.eprintf
    "usage: sounio-loom message-serve --token-file PATH [--agent A] [--lane L] [--cwd PATH] [--routing-state-dir PATH] [--bind 127.0.0.1] [--port 8789] [--allow-remote]\n"

let () =
  try
    let arguments = Array.to_list Sys.argv |> List.tl in
    (match arguments with
    | "runtime-version" :: [] ->
        print_endline "protocol_version=1";
        print_endline "runtime_version=2026.08.29.40";
        print_endline "language=OCaml";
        print_endline "role=OPERATIONAL_MESSAGE_BRIDGE"
    | [] -> usage (); exit 2
    | arguments -> arguments |> parse_cli |> serve)
  with
  | Bridge_error message -> Printf.eprintf "error: %s\n%!" message; exit 1
  | error ->
      Printf.eprintf "error: internal-error detail_sha256=%s\n%!"
        (sha256 (Printexc.to_string error));
      exit 1
