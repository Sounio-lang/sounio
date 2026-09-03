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

let process_output_timeout ~timeout_seconds cwd command arguments =
  let reader, writer = Unix.pipe () in
  Unix.set_close_on_exec reader;
  match Unix.fork () with
  | 0 ->
      Unix.close reader;
      Unix.dup2 writer Unix.stdout;
      Unix.dup2 writer Unix.stderr;
      if writer <> Unix.stdout && writer <> Unix.stderr then Unix.close writer;
      ignore (Unix.setsid ());
      (try Unix.chdir cwd; Unix.execve command arguments (Unix.environment ())
       with _ -> Unix._exit 127)
  | pid ->
      Unix.close writer;
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

let handle cwd token sender_agent sender_lane descriptor =
  let respond status body =
    write_all descriptor
      (response ~headers:[ ("X-Loom-Authority", "durable-message-bus") ]
         status body)
  in
  try
    let request = read_http_request descriptor in
    let path, query = split_http_target request.http_target in
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
    else if request.http_method = "GET" && starts_with path "/v1/threads/" then
      let message_id =
        String.sub path 12 (String.length path - 12) |> percent_decode
      in
      let projection = thread_detail cwd sender_agent sender_lane message_id query in
      Printf.eprintf
        "LOOM_MESSAGE_DECISION decision=ALLOW reason=thread-detail-read sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
        sender_agent sender_lane (sha256 projection);
      respond "200 OK" projection
    else if request.http_method = "POST" && starts_with path "/v1/messages/"
            && String.length path > 17 && String.sub path (String.length path - 4) 4 = "/ack"
    then
      let message_id =
        String.sub path 13 (String.length path - 17) |> percent_decode
      in
      let receipt = acknowledge_message cwd sender_agent sender_lane message_id in
      Printf.eprintf
        "LOOM_MESSAGE_DECISION decision=ALLOW reason=message-acknowledged sender_agent=%s sender_lane=%s receipt_sha256=%s\n%!"
        sender_agent sender_lane (sha256 receipt);
      respond "200 OK" receipt
    else respond "404 Not Found" "{\"error\":\"not_found\"}"
  with
  | Bridge_error message ->
      let status =
        if starts_with message "invalid-json:"
           || starts_with message "message-bridge-target-"
           || starts_with message "message-bridge-message-"
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
  Sys.set_signal Sys.sigchld Sys.Signal_ignore;
  Printf.printf
    "LOOM_MESSAGE_BRIDGE url=http://%s:%d schema=loom-message-bridge-v1 auth=bearer-capability sender_agent=%s sender_lane=%s\n%!"
    bind actual_port sender_agent sender_lane;
  while !running do
    let readable, _, _ =
      try Unix.select [ server ] [] [] 0.25
      with Unix_error (EINTR, _, _) -> ([], [], [])
    in
    if readable <> [] then
      let client, _ = Unix.accept server in
      match Unix.fork () with
      | 0 ->
          Unix.close server;
          Sys.set_signal Sys.sigchld Sys.Signal_default;
          Unix.setsockopt_float client SO_RCVTIMEO 5.0;
          handle cwd token sender_agent sender_lane client;
          Unix.close client;
          Unix._exit 0
      | _ -> Unix.close client
  done;
  Unix.close server

let usage () =
  Printf.eprintf
    "usage: sounio-loom message-serve --token-file PATH [--agent A] [--lane L] [--cwd PATH] [--bind 127.0.0.1] [--port 8789] [--allow-remote]\n"

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
