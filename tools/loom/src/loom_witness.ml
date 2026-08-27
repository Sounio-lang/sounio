open Unix

exception Error of string

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let zero_digest = String.make 64 '0'
let max_wire_bytes = 4 * 1024 * 1024
let max_segment_bytes = 1024 * 1024
let wire_timeout_seconds = 5.0

let witness_failpoint point =
  match Sys.getenv_opt "SOUNIO_LOOM_WITNESS_FAILPOINT" with
  | Some configured when configured = point ->
      Printf.eprintf "LOOM_WITNESS_FAILPOINT name=%s exit=197\n%!" point;
      exit 197
  | _ -> ()

let sha256 = Loom_epistemic.sha256
let valid_digest = Loom_epistemic.valid_digest
let validate_atom = Loom_epistemic.validate_atom
let hex_of_string = Loom_epistemic.hex_of_string
let string_of_hex = Loom_epistemic.string_of_hex

let starts_with value prefix =
  String.length value >= String.length prefix
  && String.sub value 0 (String.length prefix) = prefix

let read_file_bounded label limit path =
  let channel = open_in_bin path in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () ->
      let length = in_channel_length channel in
      if length > limit then failf "%s-too-large:%d" label length;
      really_input_string channel length)

let read_lines path =
  let channel = open_in path in
  let rec loop values =
    match input_line channel with
    | value -> loop (value :: values)
    | exception End_of_file -> List.rev values
  in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () -> loop [])

let atomic_write path value =
  let directory = Filename.dirname path in
  Loom_epistemic.mkdir_p directory;
  let temporary = Filename.temp_file ~temp_dir:directory ".witness-" ".tmp" in
  let descriptor = Unix.openfile temporary [ O_WRONLY; O_TRUNC ] 0o600 in
  try
    Loom_epistemic.write_all descriptor value;
    Unix.fsync descriptor;
    Unix.close descriptor;
    Unix.rename temporary path;
    Unix.chmod path 0o600;
    Loom_epistemic.fsync_directory directory
  with error ->
    (try Unix.close descriptor with _ -> ());
    (try Unix.unlink temporary with _ -> ());
    raise error

let integer label raw =
  let value =
    try int_of_string raw with _ -> failf "%s-not-integer:%s" label raw
  in
  if value < 0 then failf "%s-negative:%d" label value;
  value

let positive_integer label raw =
  let value = integer label raw in
  if value <= 0 then failf "%s-not-positive:%d" label value;
  value

let digest label value =
  let value = String.lowercase_ascii value in
  if not (valid_digest value) then failf "%s-invalid-digest" label;
  value

let nonzero_digest label value =
  let value = digest label value in
  if value = zero_digest then failf "%s-zero-digest" label;
  value

let fields_of_text label text =
  let table = Hashtbl.create 24 in
  let lines = String.split_on_char '\n' text in
  let lines =
    match List.rev lines with "" :: rest -> List.rev rest | _ -> lines
  in
  List.iter
    (fun line ->
      match String.index_opt line '=' with
      | None -> failf "%s-malformed" label
      | Some index ->
          let key = String.sub line 0 index in
          let value =
            String.sub line (index + 1) (String.length line - index - 1)
          in
          if key = "" || Hashtbl.mem table key then failf "%s-malformed" label;
          Hashtbl.add table key value)
    lines;
  table

let field label fields key =
  match Hashtbl.find_opt fields key with
  | Some value -> value
  | None -> failf "%s-field-missing:%s" label key

type member = {
  member_id : string;
  public_key_path : string;
  public_key_pem : string;
  key_id : string;
}

type endpoint = {
  endpoint_member : member;
  endpoint_host : string;
  endpoint_port : int;
}

type topology = Mesh_v0 | Mesh_v1

let topology_member_count = function Mesh_v0 -> 3 | Mesh_v1 -> 4
let topology_quorum = function Mesh_v0 -> 2 | Mesh_v1 -> 3
let topology_frame = function Mesh_v0 -> 9013 | Mesh_v1 -> 9014
let topology_suffix = function Mesh_v0 -> "v0" | Mesh_v1 -> "v1"
let topology_schema = function
  | Mesh_v0 -> "loom-witness-mesh-v0"
  | Mesh_v1 -> "loom-witness-mesh-v1"

type membership = {
  topology : topology;
  members : member array;
  anchor_public_key_path : string;
  anchor_public_key_pem : string;
  anchor_key_id : string;
  canonical_membership : string;
  membership_digest : string;
}

let resolve_relative base path =
  let path = if Filename.is_relative path then Filename.concat base path else path in
  Unix.realpath path

let load_membership path =
  let directory = Filename.dirname (Unix.realpath path) in
  let lines = read_lines path in
  let topology, anchor_public_key_path, rows =
    match lines with
    | anchor_row :: "witness_id\tpublic_key" :: rest ->
        (match String.split_on_char '\t' anchor_row with
        | [ "anchor_public_key"; anchor_public_key_path ] ->
            (Mesh_v0, resolve_relative directory anchor_public_key_path, rest)
        | _ -> failf "witness-anchor-key-row-invalid")
    | "schema\tloom-witness-membership-v1" :: anchor_row
      :: "witness_id\tpublic_key" :: rest ->
        (match String.split_on_char '\t' anchor_row with
        | [ "anchor_public_key"; anchor_public_key_path ] ->
            (Mesh_v1, resolve_relative directory anchor_public_key_path, rest)
        | _ -> failf "witness-anchor-key-row-invalid")
    | _ -> failf "witness-membership-header-invalid"
  in
  let anchor_public_key_pem =
    read_file_bounded "witness-anchor-public-key" (8 * 1024)
      anchor_public_key_path
  in
  let anchor_key_id =
    Loom_epistemic.outcome_public_key_id anchor_public_key_pem
  in
  let member_count = topology_member_count topology in
  if List.length rows <> member_count then
    failf "witness-membership-cardinality-mismatch:topology=%s:expected=%d:actual=%d"
      (topology_suffix topology) member_count (List.length rows);
  let members =
    List.map
      (fun row ->
        match String.split_on_char '\t' row with
        | [ member_id; public_key_path ] ->
            validate_atom "witness-member" member_id;
            let public_key_path = resolve_relative directory public_key_path in
            let public_key_pem =
              read_file_bounded "witness-public-key" (8 * 1024) public_key_path
            in
            let key_id = Loom_epistemic.outcome_public_key_id public_key_pem in
            { member_id; public_key_path; public_key_pem; key_id }
        | _ -> failf "witness-membership-row-malformed")
      rows
    |> List.sort (fun left right -> String.compare left.member_id right.member_id)
    |> Array.of_list
  in
  let ensure_distinct label projection =
    let seen = Hashtbl.create member_count in
    Array.iter
      (fun member ->
        let value = projection member in
        if Hashtbl.mem seen value then failf "%s" label;
        Hashtbl.add seen value ())
      members
  in
  ensure_distinct "witness-membership-id-collapse" (fun member -> member.member_id);
  ensure_distinct "witness-membership-key-collapse" (fun member -> member.key_id);
  if Array.exists (fun member -> member.key_id = anchor_key_id) members then
    failf "witness-anchor-key-collapses-with-member";
  let canonical_membership =
    (match topology with
    | Mesh_v0 -> ""
    | Mesh_v1 -> "schema\tloom-witness-membership-v1\n")
    ^ "anchor\t" ^ anchor_key_id ^ "\n"
    ^ (Array.to_list members
      |> List.map (fun member -> member.member_id ^ "\t" ^ member.key_id)
      |> String.concat "\n")
    ^ "\n"
  in
  let membership_digest =
    sha256
      ("loom-witness-membership-" ^ topology_suffix topology ^ "\000"
       ^ canonical_membership)
  in
  { topology; members; anchor_public_key_path; anchor_public_key_pem; anchor_key_id;
    canonical_membership; membership_digest }

let member_by_id membership member_id =
  match Array.to_list membership.members
        |> List.find_opt (fun member -> member.member_id = member_id) with
  | Some member -> member
  | None -> failf "witness-member-not-configured:%s" member_id

let load_endpoints membership path =
  let lines = read_lines path in
  let rows =
    match lines with
    | "witness_id\thost\tport" :: rest -> rest
    | _ -> failf "witness-endpoints-header-invalid"
  in
  let member_count = topology_member_count membership.topology in
  if List.length rows <> member_count then
    failf "witness-endpoints-cardinality-mismatch:topology=%s:expected=%d:actual=%d"
      (topology_suffix membership.topology) member_count (List.length rows);
  let endpoints =
    List.map
      (fun row ->
        match String.split_on_char '\t' row with
        | [ member_id; host; raw_port ] ->
            validate_atom "witness-endpoint-member" member_id;
            validate_atom "witness-endpoint-host" host;
            let port = positive_integer "witness-endpoint-port" raw_port in
            if port > 65535 then failf "witness-endpoint-port-too-large:%d" port;
            { endpoint_member = member_by_id membership member_id;
              endpoint_host = host; endpoint_port = port }
        | _ -> failf "witness-endpoint-row-malformed")
      rows
    |> List.sort (fun left right ->
           String.compare left.endpoint_member.member_id
             right.endpoint_member.member_id)
    |> Array.of_list
  in
  Array.iteri
    (fun index endpoint ->
      if endpoint.endpoint_member.member_id <> membership.members.(index).member_id
      then failf "witness-endpoint-membership-mismatch")
    endpoints;
  endpoints

type request = {
  request_schema : string;
  request_witness : string;
  request_membership_digest : string;
  request_domain : string;
  request_anchor_sequence : int;
  request_start_count : int;
  request_start_head : string;
  request_end_count : int;
  request_end_head : string;
  request_segment_digest : string;
  request_segment : string;
  request_anchor_key_id : string;
  request_payload_digest : string;
  request_anchor_signature : string;
}

let request_payload request =
  Printf.sprintf
    "schema=%s\nop=anchor\nwitness=%s\nmembership_sha256=%s\ndomain=%s\nanchor_sequence=%d\nstart_event_count=%d\nstart_head_sha256=%s\nend_event_count=%d\nend_head_sha256=%s\nsegment_sha256=%s\nsegment_hex=%s\n"
    request.request_schema request.request_witness request.request_membership_digest
    request.request_domain request.request_anchor_sequence
    request.request_start_count request.request_start_head
    request.request_end_count request.request_end_head
    request.request_segment_digest (hex_of_string request.request_segment)

let anchor_authorization_message schema payload_digest =
  "loom-witness-anchor-authorization-"
  ^ (if Filename.check_suffix schema "-v1" then "v1" else "v0")
  ^ "\000" ^ payload_digest

let canonical_request request =
  request_payload request
  ^ Printf.sprintf
      "anchor_key_id=%s\nrequest_payload_sha256=%s\nanchor_signature_base64=%s\n"
      request.request_anchor_key_id request.request_payload_digest
      request.request_anchor_signature

let request_digest request canonical =
  sha256 (request.request_schema ^ "\000" ^ canonical)

let authorize_request membership private_key request =
  let payload_digest = sha256 (request_payload request) in
  let signed =
    Loom_epistemic.outcome_ed25519_sign private_key
      (anchor_authorization_message request.request_schema payload_digest)
  in
  let signature =
    match Sys.getenv_opt "SOUNIO_LOOM_WITNESS_TAMPER_ANCHOR_SIGNATURE" with
    | Some "1" when signed <> "" ->
        let bytes = Bytes.of_string signed in
        Bytes.set bytes 0 (if Bytes.get bytes 0 = 'A' then 'B' else 'A');
        Bytes.unsafe_to_string bytes
    | _ -> signed
  in
  { request with request_anchor_key_id = membership.anchor_key_id;
    request_payload_digest = payload_digest;
    request_anchor_signature = signature }

let parse_request membership canonical =
  if String.length canonical > max_wire_bytes then failf "witness-request-too-large";
  let fields = fields_of_text "witness-request" canonical in
  let expected_schema =
    "loom-witness-segment-request-" ^ topology_suffix membership.topology
  in
  if field "witness-request" fields "schema" <> expected_schema
     || field "witness-request" fields "op" <> "anchor"
  then failf "witness-request-schema-invalid";
  let segment =
    try string_of_hex (field "witness-request" fields "segment_hex")
    with _ -> failf "witness-request-segment-invalid-hex"
  in
  if String.length segment > max_segment_bytes then
    failf "witness-request-segment-too-large:%d" (String.length segment);
  let request =
    { request_schema = expected_schema;
      request_witness = field "witness-request" fields "witness";
      request_membership_digest =
        nonzero_digest "witness-request-membership"
          (field "witness-request" fields "membership_sha256");
      request_domain = field "witness-request" fields "domain";
      request_anchor_sequence =
        positive_integer "witness-request-anchor-sequence"
          (field "witness-request" fields "anchor_sequence");
      request_start_count =
        integer "witness-request-start-count"
          (field "witness-request" fields "start_event_count");
      request_start_head =
        digest "witness-request-start-head"
          (field "witness-request" fields "start_head_sha256");
      request_end_count =
        positive_integer "witness-request-end-count"
          (field "witness-request" fields "end_event_count");
      request_end_head =
        nonzero_digest "witness-request-end-head"
          (field "witness-request" fields "end_head_sha256");
      request_segment_digest =
        nonzero_digest "witness-request-segment"
          (field "witness-request" fields "segment_sha256");
      request_segment = segment;
      request_anchor_key_id =
        nonzero_digest "witness-request-anchor-key"
          (field "witness-request" fields "anchor_key_id");
      request_payload_digest =
        nonzero_digest "witness-request-payload"
          (field "witness-request" fields "request_payload_sha256");
      request_anchor_signature =
        field "witness-request" fields "anchor_signature_base64" }
  in
  validate_atom "witness-request-member" request.request_witness;
  validate_atom "witness-request-domain" request.request_domain;
  if canonical_request request <> canonical then failf "witness-request-noncanonical";
  if request.request_segment_digest
     <> sha256 ("loom-witness-journal-segment-v0\000" ^ segment)
  then failf "witness-request-segment-digest-mismatch";
  if request.request_anchor_key_id <> membership.anchor_key_id
     || request.request_payload_digest <> sha256 (request_payload request)
     || request.request_anchor_signature = ""
  then failf "witness-request-anchor-authorization-mismatch";
  if not
       (Loom_epistemic.outcome_ed25519_verify membership.anchor_public_key_pem
          (anchor_authorization_message request.request_schema
             request.request_payload_digest)
          request.request_anchor_signature)
  then failf "witness-request-anchor-signature-invalid";
  request

type receipt = {
  receipt_schema : string;
  receipt_witness : string;
  receipt_key_id : string;
  receipt_membership_digest : string;
  receipt_domain : string;
  receipt_anchor_sequence : int;
  receipt_start_count : int;
  receipt_start_head : string;
  receipt_end_count : int;
  receipt_end_head : string;
  receipt_segment_digest : string;
  receipt_request_digest : string;
  receipt_anchor_key_id : string;
  receipt_request_payload_digest : string;
  receipt_anchor_signature : string;
  receipt_payload_digest : string;
  receipt_signature : string;
}

let receipt_payload receipt =
  Printf.sprintf
    "schema=%s\nalgorithm=ed25519\nwitness=%s\nkey_id=%s\nmembership_sha256=%s\ndomain=%s\nanchor_sequence=%d\nstart_event_count=%d\nstart_head_sha256=%s\nend_event_count=%d\nend_head_sha256=%s\nsegment_sha256=%s\nrequest_sha256=%s\nanchor_key_id=%s\nrequest_payload_sha256=%s\nanchor_signature_base64=%s\n"
    receipt.receipt_schema receipt.receipt_witness receipt.receipt_key_id
    receipt.receipt_membership_digest receipt.receipt_domain
    receipt.receipt_anchor_sequence receipt.receipt_start_count
    receipt.receipt_start_head receipt.receipt_end_count
    receipt.receipt_end_head receipt.receipt_segment_digest
    receipt.receipt_request_digest receipt.receipt_anchor_key_id
    receipt.receipt_request_payload_digest receipt.receipt_anchor_signature

let canonical_receipt receipt =
  receipt_payload receipt
  ^ Printf.sprintf "signed_payload_sha256=%s\nsignature_base64=%s\n"
      receipt.receipt_payload_digest receipt.receipt_signature

let receipt_digest receipt canonical =
  sha256 (receipt.receipt_schema ^ "\000" ^ canonical)

let parse_receipt membership canonical =
  if String.length canonical > 16 * 1024 then failf "witness-receipt-too-large";
  let fields = fields_of_text "witness-receipt" canonical in
  let expected_schema =
    "loom-witness-share-payload-" ^ topology_suffix membership.topology
  in
  if field "witness-receipt" fields "schema" <> expected_schema
     || field "witness-receipt" fields "algorithm" <> "ed25519"
  then failf "witness-receipt-schema-invalid";
  let receipt =
    { receipt_schema = expected_schema;
      receipt_witness = field "witness-receipt" fields "witness";
      receipt_key_id =
        nonzero_digest "witness-receipt-key"
          (field "witness-receipt" fields "key_id");
      receipt_membership_digest =
        nonzero_digest "witness-receipt-membership"
          (field "witness-receipt" fields "membership_sha256");
      receipt_domain = field "witness-receipt" fields "domain";
      receipt_anchor_sequence =
        positive_integer "witness-receipt-anchor-sequence"
          (field "witness-receipt" fields "anchor_sequence");
      receipt_start_count =
        integer "witness-receipt-start-count"
          (field "witness-receipt" fields "start_event_count");
      receipt_start_head =
        digest "witness-receipt-start-head"
          (field "witness-receipt" fields "start_head_sha256");
      receipt_end_count =
        positive_integer "witness-receipt-end-count"
          (field "witness-receipt" fields "end_event_count");
      receipt_end_head =
        nonzero_digest "witness-receipt-end-head"
          (field "witness-receipt" fields "end_head_sha256");
      receipt_segment_digest =
        nonzero_digest "witness-receipt-segment"
          (field "witness-receipt" fields "segment_sha256");
      receipt_request_digest =
        nonzero_digest "witness-receipt-request"
          (field "witness-receipt" fields "request_sha256");
      receipt_anchor_key_id =
        nonzero_digest "witness-receipt-anchor-key"
          (field "witness-receipt" fields "anchor_key_id");
      receipt_request_payload_digest =
        nonzero_digest "witness-receipt-request-payload"
          (field "witness-receipt" fields "request_payload_sha256");
      receipt_anchor_signature =
        field "witness-receipt" fields "anchor_signature_base64";
      receipt_payload_digest =
        nonzero_digest "witness-receipt-payload"
          (field "witness-receipt" fields "signed_payload_sha256");
      receipt_signature = field "witness-receipt" fields "signature_base64" }
  in
  validate_atom "witness-receipt-member" receipt.receipt_witness;
  validate_atom "witness-receipt-domain" receipt.receipt_domain;
  let payload = receipt_payload receipt in
  if receipt.receipt_payload_digest <> sha256 payload
     || receipt.receipt_signature = ""
     || receipt.receipt_anchor_signature = ""
     || canonical_receipt receipt <> canonical
  then failf "witness-receipt-noncanonical";
  receipt

let verify_receipt membership member canonical =
  let receipt = parse_receipt membership canonical in
  if receipt.receipt_witness <> member.member_id
     || receipt.receipt_key_id <> member.key_id
     || receipt.receipt_membership_digest <> membership.membership_digest
     || receipt.receipt_anchor_key_id <> membership.anchor_key_id
  then failf "witness-receipt-authority-mismatch:%s" member.member_id;
  if not
       (Loom_epistemic.outcome_ed25519_verify membership.anchor_public_key_pem
          (anchor_authorization_message
             ("loom-witness-segment-request-"
              ^ topology_suffix membership.topology)
             receipt.receipt_request_payload_digest)
          receipt.receipt_anchor_signature)
  then failf "witness-receipt-anchor-signature-invalid:%s" member.member_id;
  if not
       (Loom_epistemic.outcome_ed25519_verify member.public_key_pem
          (receipt_payload receipt) receipt.receipt_signature)
  then failf "witness-receipt-signature-invalid:%s" member.member_id;
  receipt

let verify_segment request =
  if request.request_end_count <= request.request_start_count then
    failf "witness-segment-does-not-advance";
  let lines = String.split_on_char '\n' request.request_segment in
  let lines =
    match List.rev lines with "" :: rest -> List.rev rest | _ -> lines
  in
  let expected = request.request_end_count - request.request_start_count in
  if List.length lines <> expected then
    failf "witness-segment-count-mismatch:expected=%d:actual=%d" expected
      (List.length lines);
  let _, head =
    List.fold_left
      (fun (sequence, previous) line ->
        let event = Loom_epistemic.parse_event sequence previous line in
        (sequence + 1, event.Loom_epistemic.event_sha256))
      (request.request_start_count + 1, request.request_start_head)
      lines
  in
  if head <> request.request_end_head then failf "witness-segment-head-mismatch"

let status_request membership member domain =
  Printf.sprintf
    "schema=loom-witness-status-request-%s\nop=status\nwitness=%s\nmembership_sha256=%s\ndomain=%s\n"
    (topology_suffix membership.topology) member.member_id
    membership.membership_digest domain

let genesis_status membership member domain =
  Printf.sprintf
    "schema=loom-witness-status-%s\nstatus=genesis\nwitness=%s\nkey_id=%s\nmembership_sha256=%s\ndomain=%s\n"
    (topology_suffix membership.topology) member.member_id member.key_id
    membership.membership_digest domain

let parse_status_request membership member canonical =
  let expected = status_request membership member
      (field "witness-status-request"
         (fields_of_text "witness-status-request" canonical) "domain")
  in
  let fields = fields_of_text "witness-status-request" canonical in
  if field "witness-status-request" fields "schema"
       <> "loom-witness-status-request-" ^ topology_suffix membership.topology
     || field "witness-status-request" fields "op" <> "status"
     || field "witness-status-request" fields "witness" <> member.member_id
     || field "witness-status-request" fields "membership_sha256"
        <> membership.membership_digest
     || expected <> canonical
  then failf "witness-status-request-invalid";
  let domain = field "witness-status-request" fields "domain" in
  validate_atom "witness-status-domain" domain;
  domain

let parse_genesis_status membership member domain canonical =
  if canonical <> genesis_status membership member domain then
    failf "witness-genesis-status-invalid:%s" member.member_id

let state_path state_dir domain =
  Filename.concat state_dir
    ("domain-" ^ String.sub (sha256 ("loom-witness-domain-v0\000" ^ domain)) 0 32
     ^ ".receipt")

let load_state membership member state_dir domain =
  let path = state_path state_dir domain in
  if not (Sys.file_exists path) then None
  else
    let canonical = read_file_bounded "witness-state" (16 * 1024) path in
    let receipt = verify_receipt membership member canonical in
    if receipt.receipt_domain <> domain then failf "witness-state-domain-mismatch";
    Some (canonical, receipt)

let handle_anchor membership member private_key state_dir canonical =
  let request = parse_request membership canonical in
  if request.request_witness <> member.member_id
     || request.request_membership_digest <> membership.membership_digest
  then failf "witness-request-authority-mismatch";
  let state = load_state membership member state_dir request.request_domain in
  let canonical_request_digest = request_digest request canonical in
  (match state with
  | Some (previous_canonical, previous)
    when previous.receipt_request_digest = canonical_request_digest ->
      previous_canonical
  | _ ->
      let previous_sequence, previous_count, previous_head =
        match state with
        | None -> (0, 0, zero_digest)
        | Some (_, receipt) ->
            ( receipt.receipt_anchor_sequence, receipt.receipt_end_count,
              receipt.receipt_end_head )
      in
      if request.request_anchor_sequence <= previous_sequence then
        failf "witness-anchor-sequence-not-monotonic:previous=%d:requested=%d"
          previous_sequence request.request_anchor_sequence;
      if request.request_start_count <> previous_count
         || request.request_start_head <> previous_head
      then failf "witness-anchor-predecessor-mismatch";
      verify_segment request;
      let provisional =
        { receipt_schema =
            "loom-witness-share-payload-" ^ topology_suffix membership.topology;
          receipt_witness = member.member_id; receipt_key_id = member.key_id;
          receipt_membership_digest = membership.membership_digest;
          receipt_domain = request.request_domain;
          receipt_anchor_sequence = request.request_anchor_sequence;
          receipt_start_count = request.request_start_count;
          receipt_start_head = request.request_start_head;
          receipt_end_count = request.request_end_count;
          receipt_end_head = request.request_end_head;
          receipt_segment_digest = request.request_segment_digest;
          receipt_request_digest = canonical_request_digest;
          receipt_anchor_key_id = request.request_anchor_key_id;
          receipt_request_payload_digest = request.request_payload_digest;
          receipt_anchor_signature = request.request_anchor_signature;
          receipt_payload_digest = zero_digest; receipt_signature = "" }
      in
      let payload = receipt_payload provisional in
      let signature =
        Loom_epistemic.outcome_ed25519_sign private_key payload
      in
      if not
           (Loom_epistemic.outcome_ed25519_verify member.public_key_pem payload
              signature)
      then failf "witness-private-key-mismatch:%s" member.member_id;
      let receipt =
        { provisional with receipt_payload_digest = sha256 payload;
          receipt_signature = signature }
      in
      let receipt_canonical = canonical_receipt receipt in
      atomic_write (state_path state_dir request.request_domain)
        receipt_canonical;
      receipt_canonical)

let wait_for descriptor read deadline =
  let remaining = deadline -. Unix.gettimeofday () in
  if remaining <= 0.0 then failf "witness-wire-timeout";
  let readable, writable, _ =
    if read then Unix.select [ descriptor ] [] [] remaining
    else Unix.select [] [ descriptor ] [] remaining
  in
  if (read && readable = []) || ((not read) && writable = []) then
    failf "witness-wire-timeout"

let write_timeout descriptor value =
  let bytes = Bytes.unsafe_of_string value in
  let deadline = Unix.gettimeofday () +. wire_timeout_seconds in
  let rec loop offset =
    if offset < Bytes.length bytes then (
      wait_for descriptor false deadline;
      let count = Unix.write descriptor bytes offset (Bytes.length bytes - offset) in
      if count <= 0 then failf "witness-wire-short-write";
      loop (offset + count))
  in
  loop 0

let read_exact_timeout descriptor length =
  let bytes = Bytes.create length in
  let deadline = Unix.gettimeofday () +. wire_timeout_seconds in
  let rec loop offset =
    if offset < length then (
      wait_for descriptor true deadline;
      let count = Unix.read descriptor bytes offset (length - offset) in
      if count <= 0 then failf "witness-wire-short-read";
      loop (offset + count))
  in
  loop 0;
  Bytes.unsafe_to_string bytes

let send_frame descriptor payload =
  if String.length payload > max_wire_bytes then failf "witness-wire-frame-too-large";
  write_timeout descriptor (Printf.sprintf "%08x\n%s" (String.length payload) payload)

let receive_frame descriptor =
  let header = read_exact_timeout descriptor 9 in
  if header.[8] <> '\n' then failf "witness-wire-header-invalid";
  let length =
    try int_of_string ("0x" ^ String.sub header 0 8)
    with _ -> failf "witness-wire-header-invalid"
  in
  if length < 0 || length > max_wire_bytes then failf "witness-wire-frame-too-large";
  read_exact_timeout descriptor length

let inet_address host =
  try Unix.inet_addr_of_string host
  with _ ->
    let entry = Unix.gethostbyname host in
    if Array.length entry.h_addr_list = 0 then failf "witness-host-unresolved:%s" host;
    entry.h_addr_list.(0)

let connect_endpoint endpoint =
  let descriptor = Unix.socket PF_INET SOCK_STREAM 0 in
  Unix.set_close_on_exec descriptor;
  Unix.set_nonblock descriptor;
  try
    (try
       Unix.connect descriptor
         (ADDR_INET (inet_address endpoint.endpoint_host, endpoint.endpoint_port))
     with Unix_error ((EINPROGRESS | EWOULDBLOCK), _, _) ->
       let _, writable, _ = Unix.select [] [ descriptor ] [] wire_timeout_seconds in
       if writable = [] then failf "witness-connect-timeout:%s"
           endpoint.endpoint_member.member_id;
       (match Unix.getsockopt_error descriptor with
       | None -> ()
       | Some error -> raise (Unix_error (error, "connect", endpoint.endpoint_host))));
    Unix.clear_nonblock descriptor;
    descriptor
  with error ->
    Unix.close descriptor;
    raise error

let exchange endpoint payload =
  let descriptor = connect_endpoint endpoint in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
      send_frame descriptor payload;
      receive_frame descriptor)

type remote_status = Genesis | Latest of string * receipt | Unavailable of string

let query_status membership endpoint domain =
  try
    let canonical =
      exchange endpoint (status_request membership endpoint.endpoint_member domain)
    in
    if starts_with canonical
         ("schema=loom-witness-status-" ^ topology_suffix membership.topology
          ^ "\n")
    then (
      parse_genesis_status membership endpoint.endpoint_member domain canonical;
      Genesis)
    else
      let receipt =
        verify_receipt membership endpoint.endpoint_member canonical
      in
      if receipt.receipt_domain <> domain then failf "witness-status-domain-mismatch";
      Latest (canonical, receipt)
  with
  | Error error -> Unavailable error
  | Unix_error (error, _, _) -> Unavailable (Unix.error_message error)
  | Sys_error error -> Unavailable error

let send_anchor membership endpoint request =
  let canonical_request = canonical_request request in
  let response = exchange endpoint canonical_request in
  if starts_with response "REFUSED " then
    failf "witness-remote-refused:%s:%s" endpoint.endpoint_member.member_id
      (String.sub response 8 (String.length response - 8));
  let receipt = verify_receipt membership endpoint.endpoint_member response in
  if receipt.receipt_domain <> request.request_domain
     || receipt.receipt_anchor_sequence <> request.request_anchor_sequence
     || receipt.receipt_start_count <> request.request_start_count
     || receipt.receipt_start_head <> request.request_start_head
     || receipt.receipt_end_count <> request.request_end_count
     || receipt.receipt_end_head <> request.request_end_head
     || receipt.receipt_segment_digest <> request.request_segment_digest
     || receipt.receipt_request_digest <> request_digest request canonical_request
     || receipt.receipt_anchor_key_id <> request.request_anchor_key_id
     || receipt.receipt_request_payload_digest <> request.request_payload_digest
     || receipt.receipt_anchor_signature <> request.request_anchor_signature
  then failf "witness-remote-receipt-mismatch:%s"
      endpoint.endpoint_member.member_id;
  (response, receipt)

let serve ~state_dir ~membership_file ~witness_id ~private_key ~bind ~port =
  let membership = load_membership membership_file in
  let member = member_by_id membership witness_id in
  validate_atom "witness-domain-member" witness_id;
  if not (Sys.file_exists private_key) then
    failf "witness-private-key-missing:%s" private_key;
  Loom_epistemic.mkdir_p state_dir;
  let probe = "loom-witness-key-probe-v0\000" ^ membership.membership_digest in
  let probe_signature = Loom_epistemic.outcome_ed25519_sign private_key probe in
  if not
       (Loom_epistemic.outcome_ed25519_verify member.public_key_pem probe
          probe_signature)
  then failf "witness-private-key-mismatch:%s" witness_id;
  let server = Unix.socket PF_INET SOCK_STREAM 0 in
  Unix.set_close_on_exec server;
  Unix.setsockopt server SO_REUSEADDR true;
  Unix.bind server (ADDR_INET (inet_address bind, port));
  Unix.listen server 64;
  let actual_port =
    match Unix.getsockname server with ADDR_INET (_, value) -> value | _ -> port
  in
  Printf.printf
    "LOOM_WITNESS_READY schema=loom-witness-service-%s witness=%s key_id=%s membership_sha256=%s members=%d quorum=%d bind=%s port=%d authority=external-monotonic-state\n%!"
    (topology_suffix membership.topology) witness_id member.key_id
    membership.membership_digest (topology_member_count membership.topology)
    (topology_quorum membership.topology) bind actual_port;
  while true do
    let client, _ = Unix.accept server in
    Fun.protect ~finally:(fun () -> Unix.close client) (fun () ->
        try
          let canonical = receive_frame client in
          let fields = fields_of_text "witness-wire-request" canonical in
          let response =
            match Hashtbl.find_opt fields "op" with
            | Some "status" ->
                let domain = parse_status_request membership member canonical in
                (match load_state membership member state_dir domain with
                | None -> genesis_status membership member domain
                | Some (receipt, _) -> receipt)
            | Some "anchor" ->
                handle_anchor membership member private_key state_dir canonical
            | _ -> failf "witness-wire-operation-invalid"
          in
          send_frame client response
        with
        | Error error -> send_frame client ("REFUSED " ^ error)
        | Sys_error error -> send_frame client ("REFUSED " ^ error)
        | Unix_error (error, _, _) ->
            send_frame client ("REFUSED " ^ Unix.error_message error))
  done

type certificate = {
  certificate_topology : topology;
  certificate_world : string;
  certificate_membership_digest : string;
  certificate_previous_digest : string;
  certificate_previous_sequence : int;
  certificate_sequence : int;
  certificate_previous_event_count : int;
  certificate_event_count : int;
  certificate_journal_head : string;
  certificate_shares : string option array;
}

let share_field = function None -> "-" | Some canonical -> hex_of_string canonical

let canonical_certificate certificate =
  let prefix =
    Printf.sprintf
      "schema=loom-witness-mesh-certificate-%s\nworld=%s\nmembership_sha256=%s\nquorum=%d\nprevious_certificate_sha256=%s\nprevious_anchor_sequence=%d\nanchor_sequence=%d\nprevious_event_count=%d\nevent_count=%d\njournal_head_sha256=%s\n"
      (topology_suffix certificate.certificate_topology)
      certificate.certificate_world certificate.certificate_membership_digest
      (topology_quorum certificate.certificate_topology)
      certificate.certificate_previous_digest
      certificate.certificate_previous_sequence certificate.certificate_sequence
      certificate.certificate_previous_event_count certificate.certificate_event_count
      certificate.certificate_journal_head
  in
  let shares =
    Array.to_list certificate.certificate_shares
    |> List.mapi (fun index share ->
           Printf.sprintf "share%d=%s\n" (index + 1) (share_field share))
    |> String.concat ""
  in
  prefix ^ shares
  ^ Printf.sprintf "native_frame=%d\n"
      (topology_frame certificate.certificate_topology)

let certificate_digest topology canonical =
  sha256
    ("loom-witness-mesh-certificate-" ^ topology_suffix topology ^ "\000"
     ^ canonical)

let parse_share label value =
  if value = "-" then None
  else
    try Some (string_of_hex value)
    with _ -> failf "%s-invalid-hex" label

let parse_certificate membership canonical =
  if String.length canonical > 64 * 1024 then failf "witness-certificate-too-large";
  let fields = fields_of_text "witness-certificate" canonical in
  let topology = membership.topology in
  if field "witness-certificate" fields "schema"
       <> "loom-witness-mesh-certificate-" ^ topology_suffix topology
     || field "witness-certificate" fields "quorum"
        <> string_of_int (topology_quorum topology)
     || field "witness-certificate" fields "native_frame"
        <> string_of_int (topology_frame topology)
  then failf "witness-certificate-schema-invalid";
  let certificate =
    { certificate_topology = topology;
      certificate_world = field "witness-certificate" fields "world";
      certificate_membership_digest =
        nonzero_digest "witness-certificate-membership"
          (field "witness-certificate" fields "membership_sha256");
      certificate_previous_digest =
        digest "witness-certificate-previous"
          (field "witness-certificate" fields "previous_certificate_sha256");
      certificate_previous_sequence =
        integer "witness-certificate-previous-sequence"
          (field "witness-certificate" fields "previous_anchor_sequence");
      certificate_sequence =
        positive_integer "witness-certificate-sequence"
          (field "witness-certificate" fields "anchor_sequence");
      certificate_previous_event_count =
        integer "witness-certificate-previous-count"
          (field "witness-certificate" fields "previous_event_count");
      certificate_event_count =
        positive_integer "witness-certificate-count"
          (field "witness-certificate" fields "event_count");
      certificate_journal_head =
        nonzero_digest "witness-certificate-head"
          (field "witness-certificate" fields "journal_head_sha256");
      certificate_shares =
        Array.init (topology_member_count topology) (fun index ->
            let key = Printf.sprintf "share%d" (index + 1) in
            parse_share ("witness-certificate-" ^ key)
              (field "witness-certificate" fields key)) }
  in
  validate_atom "witness-certificate-world" certificate.certificate_world;
  if canonical_certificate certificate <> canonical then
    failf "witness-certificate-noncanonical";
  certificate

let adapter_path topology =
  let variable, binary =
    match topology with
    | Mesh_v0 ->
        ("SOUNIO_LOOM_WITNESS_MESH_ADAPTER",
         "sounio-loom-witness-mesh-runtime")
    | Mesh_v1 ->
        ("SOUNIO_LOOM_WITNESS_MESH_V1_ADAPTER",
         "sounio-loom-witness-mesh-v1-runtime")
  in
  match Sys.getenv_opt variable with
  | Some path when path <> "" -> path
  | _ ->
      Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
        binary

let zero_limbs = List.init 8 (fun _ -> "0")

let verify_native_frame membership certificate receipts =
  let signature_flags =
    Array.to_list receipts
    |> List.map (function None -> "0" | Some _ -> "1")
  in
  let members =
    Array.to_list membership.members
    |> List.map (fun member ->
           Loom_epistemic.token "loom-witness-member" member.key_id)
  in
  let slot_values projection token_domain =
    Array.to_list receipts
    |> List.map (function
         | None -> "0"
         | Some receipt -> token_domain (projection receipt))
  in
  let receipt_members =
    slot_values (fun receipt -> receipt.receipt_key_id)
      (Loom_epistemic.token "loom-witness-member")
  in
  let expected_domain =
    Loom_epistemic.token "loom-witness-domain" certificate.certificate_world
  in
  let receipt_domains =
    slot_values (fun receipt -> receipt.receipt_domain)
      (Loom_epistemic.token "loom-witness-domain")
  in
  let sequences =
    Array.to_list receipts
    |> List.map (function None -> "0" | Some value -> string_of_int value.receipt_anchor_sequence)
  in
  let counts =
    Array.to_list receipts
    |> List.map (function None -> "0" | Some value -> string_of_int value.receipt_end_count)
  in
  let digest_slots projection =
    Array.to_list receipts
    |> List.concat_map (function
         | None -> zero_limbs
         | Some receipt -> Loom_epistemic.digest_limbs (projection receipt))
  in
  let frame =
    [ string_of_int (topology_frame membership.topology);
      string_of_int (topology_quorum membership.topology) ]
    @ signature_flags @ members @ receipt_members
    @ [ expected_domain ] @ receipt_domains
    @ [ string_of_int certificate.certificate_previous_sequence;
        string_of_int certificate.certificate_sequence ]
    @ sequences
    @ [ string_of_int certificate.certificate_previous_event_count;
        string_of_int certificate.certificate_event_count ]
    @ counts
    @ Loom_epistemic.digest_limbs membership.membership_digest
    @ digest_slots (fun receipt -> receipt.receipt_membership_digest)
    @ Loom_epistemic.digest_limbs certificate.certificate_journal_head
    @ digest_slots (fun receipt -> receipt.receipt_end_head)
  in
  let adapter = adapter_path membership.topology in
  if not (Sys.file_exists adapter) then failf "witness-native-adapter-missing:%s" adapter;
  let code, output =
    Loom_epistemic.process_exchange (Unix.realpath adapter)
      (String.concat " " frame ^ "\n")
  in
  let expected =
    match membership.topology with
    | Mesh_v0 ->
        "SOUNIO_WITNESS_MESH_ACCEPT schema=loom-native-witness-mesh-v0 transition=anchor state=quorum-verified"
    | Mesh_v1 ->
        "SOUNIO_WITNESS_MESH_V1_ACCEPT schema=loom-native-witness-mesh-v1 transition=anchor state=quorum-verified"
  in
  if code <> 0 || output <> expected then
    failf "witness-native-refused:rc=%d:output=%s" code output

let events_and_heads root world =
  let events, head =
    Loom_epistemic.load_events (Loom_epistemic.journal_path root world)
  in
  let array = Array.of_list events in
  let head_at count =
    if count = 0 then zero_digest
    else if count > Array.length array then failf "witness-event-count-beyond-journal:%d" count
    else array.(count - 1).Loom_epistemic.event_sha256
  in
  (array, head, head_at)

let segment_between events start_count end_count =
  if start_count < 0 || end_count > Array.length events || end_count <= start_count
  then failf "witness-segment-range-invalid:%d:%d" start_count end_count;
  let buffer = Buffer.create 4096 in
  for index = start_count to end_count - 1 do
    Buffer.add_string buffer (Loom_epistemic.encode_event events.(index))
  done;
  let segment = Buffer.contents buffer in
  if String.length segment > max_segment_bytes then
    failf "witness-segment-too-large:%d" (String.length segment);
  segment

let mesh_dir root world =
  Filename.concat (Filename.concat root "loom-witness-mesh") world

let certificate_files root world =
  let directory = mesh_dir root world in
  if not (Sys.file_exists directory) then []
  else
    Sys.readdir directory |> Array.to_list
    |> List.filter (fun name -> starts_with name "checkpoint-" && Filename.check_suffix name ".receipt")
    |> List.sort String.compare
    |> List.map (Filename.concat directory)

let receipts_for_certificate membership certificate =
  Array.mapi
    (fun index share ->
      match share with
      | None -> None
      | Some canonical ->
          let receipt = verify_receipt membership membership.members.(index) canonical in
          if receipt.receipt_domain <> certificate.certificate_world
             || receipt.receipt_anchor_sequence <> certificate.certificate_sequence
             || receipt.receipt_end_count <> certificate.certificate_event_count
             || receipt.receipt_end_head <> certificate.certificate_journal_head
          then failf "witness-certificate-share-mismatch:%s"
              membership.members.(index).member_id;
          Some receipt)
    certificate.certificate_shares

let load_certificate_chain root world membership events head_at =
  let previous_digest = ref zero_digest in
  let previous_sequence = ref 0 in
  let previous_count = ref 0 in
  let latest = ref None in
  List.iter
    (fun path ->
      let canonical = read_file_bounded "witness-certificate" (64 * 1024) path in
      let certificate = parse_certificate membership canonical in
      if certificate.certificate_world <> world
         || certificate.certificate_membership_digest <> membership.membership_digest
         || certificate.certificate_previous_digest <> !previous_digest
         || certificate.certificate_previous_sequence <> !previous_sequence
         || certificate.certificate_sequence <> !previous_sequence + 1
         || certificate.certificate_previous_event_count <> !previous_count
         || certificate.certificate_event_count <= !previous_count
         || certificate.certificate_event_count > Array.length events
         || head_at certificate.certificate_event_count
            <> certificate.certificate_journal_head
      then failf "witness-certificate-chain-mismatch:%s" path;
      let receipts = receipts_for_certificate membership certificate in
      if Array.fold_left (fun count -> function None -> count | Some _ -> count + 1) 0 receipts
         < topology_quorum membership.topology
      then failf "witness-certificate-quorum-missing:%s" path;
      verify_native_frame membership certificate receipts;
      previous_digest := certificate_digest membership.topology canonical;
      previous_sequence := certificate.certificate_sequence;
      previous_count := certificate.certificate_event_count;
      latest := Some (canonical, certificate))
    (certificate_files root world);
  !latest

let validate_remote_journal_prefix member status events head_at =
  match status with
  | Unavailable _ | Genesis -> ()
  | Latest (_, receipt) ->
      if receipt.receipt_end_count > Array.length events then
        failf "witness-rollback-detected:%s:remote-count=%d:local=%d"
          member.member_id receipt.receipt_end_count (Array.length events);
      if head_at receipt.receipt_end_count <> receipt.receipt_end_head then
        failf "witness-fork-detected:%s:count=%d" member.member_id
          receipt.receipt_end_count

let validate_remote_prefix member status local_sequence events head_at =
  validate_remote_journal_prefix member status events head_at;
  match status with
  | Unavailable _ | Genesis -> ()
  | Latest (_, receipt) ->
      if receipt.receipt_anchor_sequence > local_sequence then
        failf "witness-rollback-detected:%s:remote-sequence=%d:local=%d"
          member.member_id receipt.receipt_anchor_sequence local_sequence

let recovery_target previous_sequence previous_count current_count current_head statuses =
  let next_sequence = previous_sequence + 1 in
  let ahead = ref [] in
  Array.iter
    (function
      | Genesis | Unavailable _ -> ()
      | Latest (canonical, receipt) ->
          if receipt.receipt_anchor_sequence > next_sequence then
            failf "witness-rollback-detected:remote-sequence=%d:local=%d"
              receipt.receipt_anchor_sequence previous_sequence;
          if receipt.receipt_anchor_sequence = next_sequence then
            ahead := (canonical, receipt) :: !ahead)
    statuses;
  match List.rev !ahead with
  | [] -> (current_count, current_head, false)
  | (_, first) :: rest ->
      if first.receipt_end_count <= previous_count then
        failf "witness-recovery-does-not-advance:previous=%d:remote=%d"
          previous_count first.receipt_end_count;
      List.iter
        (fun (_, receipt) ->
          if receipt.receipt_end_count <> first.receipt_end_count
             || receipt.receipt_end_head <> first.receipt_end_head
          then failf
              "witness-equivocation-detected:sequence=%d:first-count=%d:other-count=%d"
              next_sequence first.receipt_end_count receipt.receipt_end_count)
        rest;
      (first.receipt_end_count, first.receipt_end_head, true)

let with_mesh_lock root world operation =
  let directory = mesh_dir root world in
  Loom_epistemic.mkdir_p directory;
  let path = Filename.concat directory "mesh.lock" in
  let descriptor = Unix.openfile path [ O_RDWR; O_CREAT ] 0o600 in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      Fun.protect ~finally:(fun () -> Unix.lockf descriptor F_ULOCK 0) operation)

let validate_anchor_private_key membership private_key =
  if not (Sys.file_exists private_key) then
    failf "witness-anchor-private-key-missing:%s" private_key;
  let probe =
    "loom-witness-anchor-key-probe-v0\000" ^ membership.membership_digest
  in
  let signature = Loom_epistemic.outcome_ed25519_sign private_key probe in
  if not
       (Loom_epistemic.outcome_ed25519_verify membership.anchor_public_key_pem
          probe signature)
  then failf "witness-anchor-private-key-mismatch"

let anchor ~root ~world ~membership_file ~endpoints_file ~anchor_private_key =
  validate_atom "witness-world" world;
  let membership = load_membership membership_file in
  validate_anchor_private_key membership anchor_private_key;
  let endpoints = load_endpoints membership endpoints_file in
  with_mesh_lock root world (fun () ->
      let events, current_head, head_at = events_and_heads root world in
      if Array.length events = 0 then failf "witness-journal-empty:%s" world;
      let recovered_checkpoints = ref 0 in
      let rec advance () =
        let latest = load_certificate_chain root world membership events head_at in
        let previous_digest, previous_sequence, previous_count =
          match latest with
          | None -> (zero_digest, 0, 0)
          | Some (canonical, certificate) ->
              ( certificate_digest membership.topology canonical,
                certificate.certificate_sequence,
                certificate.certificate_event_count )
        in
        if Array.length events <= previous_count then
          failf "witness-no-unanchored-events:%s" world;
        let statuses =
          Array.map (fun endpoint -> query_status membership endpoint world) endpoints
        in
        Array.iteri
          (fun index status ->
            validate_remote_journal_prefix membership.members.(index) status events
              head_at)
          statuses;
        let target_count, target_head, recovering =
          recovery_target previous_sequence previous_count (Array.length events)
            current_head statuses
        in
        let anchor_sequence = previous_sequence + 1 in
        let member_count = topology_member_count membership.topology in
        let required_quorum = topology_quorum membership.topology in
        let receipts = Array.make member_count None in
        let failures = ref [] in
        Array.iteri
          (fun index endpoint ->
            match statuses.(index) with
            | Latest (canonical, receipt)
              when receipt.receipt_anchor_sequence = anchor_sequence
                   && receipt.receipt_end_count = target_count
                   && receipt.receipt_end_head = target_head ->
                receipts.(index) <- Some (canonical, receipt)
            | status ->
                let start_count, start_head =
                  match status with
                  | Genesis | Unavailable _ -> (0, zero_digest)
                  | Latest (_, receipt) ->
                      (receipt.receipt_end_count, receipt.receipt_end_head)
                in
                if start_count >= target_count then
                  failures :=
                    ( endpoint.endpoint_member.member_id,
                      Printf.sprintf
                        "witness-remote-state-not-behind-target:start=%d:target=%d"
                        start_count target_count )
                    :: !failures
                else
                  let segment = segment_between events start_count target_count in
                  let unsigned_request =
                    { request_schema =
                        "loom-witness-segment-request-"
                        ^ topology_suffix membership.topology;
                      request_witness = endpoint.endpoint_member.member_id;
                      request_membership_digest = membership.membership_digest;
                      request_domain = world;
                      request_anchor_sequence = anchor_sequence;
                      request_start_count = start_count; request_start_head = start_head;
                      request_end_count = target_count; request_end_head = target_head;
                      request_segment_digest =
                        sha256 ("loom-witness-journal-segment-v0\000" ^ segment);
                      request_segment = segment;
                      request_anchor_key_id = membership.anchor_key_id;
                      request_payload_digest = zero_digest;
                      request_anchor_signature = "" }
                  in
                  let request =
                    authorize_request membership anchor_private_key unsigned_request
                  in
                  (try
                     let canonical, receipt = send_anchor membership endpoint request in
                     receipts.(index) <- Some (canonical, receipt)
                   with
                  | Error error ->
                      failures :=
                        (endpoint.endpoint_member.member_id, error) :: !failures
                  | Unix_error (error, _, _) ->
                      failures :=
                        ( endpoint.endpoint_member.member_id,
                          Unix.error_message error )
                        :: !failures
                  | Sys_error error ->
                      failures :=
                        (endpoint.endpoint_member.member_id, error) :: !failures))
          endpoints;
        let receipt_count =
          Array.fold_left
            (fun count -> function None -> count | Some _ -> count + 1)
            0 receipts
        in
        if receipt_count < required_quorum then (
          let detail =
            List.rev !failures
            |> List.map (fun (member, error) -> member ^ "=" ^ error)
            |> String.concat ","
          in
          failf "witness-quorum-unavailable:valid=%d:failures=%s" receipt_count
            detail);
        let share_canonicals =
          Array.map
            (function None -> None | Some (canonical, _) -> Some canonical)
            receipts
        in
        let receipt_values =
          Array.map
            (function None -> None | Some (_, receipt) -> Some receipt)
            receipts
        in
        let certificate =
          { certificate_topology = membership.topology;
            certificate_world = world;
            certificate_membership_digest = membership.membership_digest;
            certificate_previous_digest = previous_digest;
            certificate_previous_sequence = previous_sequence;
            certificate_sequence = anchor_sequence;
            certificate_previous_event_count = previous_count;
            certificate_event_count = target_count;
            certificate_journal_head = target_head;
            certificate_shares = share_canonicals }
        in
        verify_native_frame membership certificate receipt_values;
        witness_failpoint "after-quorum-before-certificate";
        let canonical = canonical_certificate certificate in
        let output =
          Filename.concat (mesh_dir root world)
            (Printf.sprintf "checkpoint-%020d-%s.receipt" anchor_sequence
               (String.sub target_head 0 16))
        in
        atomic_write output canonical;
        if recovering then incr recovered_checkpoints;
        if target_count < Array.length events then advance ()
        else
          Printf.sprintf
            "LOOM_WITNESS_MESH_ANCHORED schema=%s world=%s sequence=%d event_count=%d journal_head=%s quorum=%d/%d recovered_checkpoints=%d membership_sha256=%s certificate_sha256=%s certificate=%s"
            (topology_schema membership.topology) world anchor_sequence
            target_count target_head receipt_count member_count
            !recovered_checkpoints membership.membership_digest
            (certificate_digest membership.topology canonical) output
      in
      advance ())

type verification_policy = Crash_quorum | Byzantine_strict

let verification_policy_of_string = function
  | "crash-quorum" -> Crash_quorum
  | "byzantine-strict" -> Byzantine_strict
  | value -> failf "witness-verification-policy-invalid:%s" value

let verification_policy_name = function
  | Crash_quorum -> "crash-quorum"
  | Byzantine_strict -> "byzantine-strict"

let verify ~root ~world ~membership_file ~endpoints_file ~policy =
  validate_atom "witness-world" world;
  let membership = load_membership membership_file in
  let endpoints = load_endpoints membership endpoints_file in
  with_mesh_lock root world (fun () ->
      let events, current_head, head_at = events_and_heads root world in
      let latest = load_certificate_chain root world membership events head_at in
      let canonical, certificate =
        match latest with
        | None -> failf "witness-certificate-missing:%s" world
        | Some value -> value
      in
      if certificate.certificate_event_count <> Array.length events
         || certificate.certificate_journal_head <> current_head
      then failf "witness-unanchored-journal-suffix:%s:anchored=%d:current=%d"
          world certificate.certificate_event_count (Array.length events);
      let statuses =
        Array.map (fun endpoint -> query_status membership endpoint world) endpoints
      in
      Array.iteri
        (fun index status ->
          validate_remote_prefix membership.members.(index) status
            certificate.certificate_sequence events head_at)
        statuses;
      let current =
        Array.fold_left
          (fun count -> function
            | Latest (_, receipt)
              when receipt.receipt_anchor_sequence = certificate.certificate_sequence
                   && receipt.receipt_end_count = certificate.certificate_event_count
                   && receipt.receipt_end_head = certificate.certificate_journal_head ->
                count + 1
            | _ -> count)
          0 statuses
      in
      let required, rollback_resistance =
        match (membership.topology, policy) with
        | _, Crash_quorum ->
            (topology_quorum membership.topology,
             "CONDITIONAL_ON_NON_EQUIVOCATION")
        | Mesh_v0, Byzantine_strict ->
            (topology_member_count membership.topology,
             "ONE_DISHONEST_WITNESS_ALL_MEMBERS")
        | Mesh_v1, Byzantine_strict ->
            (topology_quorum membership.topology,
             "ONE_DISHONEST_WITNESS_HONEST_INTERSECTION")
      in
      if current < required then
        failf "witness-current-quorum-unavailable:policy=%s:valid=%d:required=%d"
          (verification_policy_name policy) current required;
      Printf.sprintf
        "LOOM_WITNESS_MESH_OK schema=%s world=%s sequence=%d event_count=%d journal_head=%s verification_policy=%s remote_quorum=%d/%d required=%d/%d membership_sha256=%s certificate_sha256=%s rollback_resistance=%s scope=THROUGH_LATEST_CHECKPOINT"
        (topology_schema membership.topology) world
        certificate.certificate_sequence certificate.certificate_event_count
        certificate.certificate_journal_head (verification_policy_name policy)
        current (topology_member_count membership.topology) required
        (topology_member_count membership.topology) membership.membership_digest
        (certificate_digest membership.topology canonical)
        rollback_resistance)
