open Unix

exception Error of string

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let schema = "loom-epistemic-machine-v0"
let journal_domain = "loom-epistemic-journal-v0"
let zero_digest = String.make 64 '0'

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let trim = String.trim

let valid_digest value =
  String.length value = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
       value

let require_digest label value =
  let normalized = String.lowercase_ascii value in
  if not (valid_digest normalized) || normalized = zero_digest then
    failf "epistemic-%s-digest-invalid" label;
  normalized

let validate_atom label value =
  if value = "" || String.length value > 256 then
    failf "epistemic-%s-invalid" label;
  String.iter
    (function
      | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '.' | '_' | '-' -> ()
      | _ -> failf "epistemic-%s-invalid" label)
    value

let validate_text label value =
  if value = "" || String.length value > 4096 || String.contains value '\000'
  then failf "epistemic-%s-invalid" label

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

let hex_value = function
  | '0' .. '9' as value -> Char.code value - Char.code '0'
  | 'a' .. 'f' as value -> 10 + Char.code value - Char.code 'a'
  | 'A' .. 'F' as value -> 10 + Char.code value - Char.code 'A'
  | _ -> failf "epistemic-payload-invalid-hex"

let string_of_hex value =
  if String.length value mod 2 <> 0 then failf "epistemic-payload-invalid-hex";
  let output = Bytes.create (String.length value / 2) in
  for index = 0 to Bytes.length output - 1 do
    let high = hex_value value.[index * 2] in
    let low = hex_value value.[(index * 2) + 1] in
    Bytes.set output index (Char.chr ((high lsl 4) lor low))
  done;
  Bytes.unsafe_to_string output

let fsync_directory path =
  let descriptor = Unix.openfile path [ O_RDONLY ] 0 in
  Unix.set_close_on_exec descriptor;
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () -> Unix.fsync descriptor)

let rec mkdir_p path =
  if path = "" || path = "." || path = "/" || Sys.file_exists path then ()
  else (
    mkdir_p (Filename.dirname path);
    Unix.mkdir path 0o700;
    fsync_directory (Filename.dirname path))

let write_all descriptor value =
  let bytes = Bytes.unsafe_of_string value in
  let rec loop offset =
    if offset < Bytes.length bytes then
      let count = Unix.write descriptor bytes offset (Bytes.length bytes - offset) in
      if count = 0 then failf "epistemic-short-write" else loop (offset + count)
  in
  loop 0

let read_lines path =
  let channel = open_in_bin path in
  let rec loop values =
    match input_line channel with
    | value -> loop (value :: values)
    | exception End_of_file -> List.rev values
  in
  Fun.protect ~finally:(fun () -> close_in_noerr channel) (fun () -> loop [])

let utc_now () =
  let value = Unix.gettimeofday () in
  let seconds = int_of_float value in
  let micros = int_of_float ((value -. float_of_int seconds) *. 1_000_000.) in
  let tm = Unix.gmtime value in
  Printf.sprintf "%04d-%02d-%02dT%02d:%02d:%02d.%06dZ"
    (tm.tm_year + 1900) (tm.tm_mon + 1) tm.tm_mday tm.tm_hour tm.tm_min
    tm.tm_sec micros

let machine_dir root = Filename.concat root "loom-epistemic"
let worlds_dir root = Filename.concat (machine_dir root) "worlds"
let world_dir root world = Filename.concat (worlds_dir root) world
let journal_path root world = Filename.concat (world_dir root world) "journal.tsv"
let lock_path root = Filename.concat (machine_dir root) "machine.lock"

let with_machine_lock root action =
  mkdir_p (machine_dir root);
  let descriptor =
    Unix.openfile (lock_path root) [ O_RDWR; O_CREAT ] 0o600
  in
  Unix.set_close_on_exec descriptor;
  Fun.protect
    ~finally:(fun () ->
      (try Unix.lockf descriptor F_ULOCK 0 with _ -> ());
      Unix.close descriptor)
    (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      action ())

let encode_fields fields =
  fields
  |> List.map (fun (key, value) ->
         validate_atom "payload-key" key;
         if String.contains value '\000' then failf "epistemic-payload-value-invalid";
         key ^ "=" ^ value)
  |> String.concat "\000"

let decode_fields payload =
  let table = Hashtbl.create 24 in
  if payload <> "" then
    String.split_on_char '\000' payload
    |> List.iter (fun field ->
           match String.index_opt field '=' with
           | None -> failf "epistemic-payload-field-malformed"
           | Some index ->
               let key = String.sub field 0 index in
               let value =
                 String.sub field (index + 1) (String.length field - index - 1)
               in
               if Hashtbl.mem table key then
                 failf "epistemic-payload-field-duplicate:%s" key;
               Hashtbl.add table key value);
  table

let field fields key =
  match Hashtbl.find_opt fields key with
  | Some value -> value
  | None -> failf "epistemic-payload-field-missing:%s" key

let optional_field fields key = Option.value ~default:"" (Hashtbl.find_opt fields key)

type event = {
  sequence : int;
  observed_at_utc : string;
  previous_sha256 : string;
  kind : string;
  payload : string;
  event_sha256 : string;
}

let event_body sequence observed previous kind payload_hex =
  String.concat "\t"
    [ string_of_int sequence; observed; previous; kind; payload_hex ]

let event_digest body = sha256 (journal_domain ^ "\000" ^ body)

let encode_event event =
  let body =
    event_body event.sequence event.observed_at_utc event.previous_sha256
      event.kind (hex_of_string event.payload)
  in
  body ^ "\t" ^ event.event_sha256 ^ "\n"

let parse_event expected_sequence expected_previous line =
  match String.split_on_char '\t' line with
  | [ sequence; observed; previous; kind; payload_hex; digest ] ->
      let sequence =
        try int_of_string sequence
        with _ -> failf "epistemic-journal-sequence-invalid"
      in
      if sequence <> expected_sequence then
        failf "epistemic-journal-non-contiguous-sequence:expected=%d:actual=%d"
          expected_sequence sequence;
      if previous <> expected_previous then
        failf "epistemic-journal-previous-mismatch:seq=%d" sequence;
      if not (valid_digest previous) || not (valid_digest digest) then
        failf "epistemic-journal-digest-invalid:seq=%d" sequence;
      validate_atom "event-kind" kind;
      let body = event_body sequence observed previous kind payload_hex in
      if event_digest body <> digest then
        failf "epistemic-journal-event-digest-mismatch:seq=%d" sequence;
      { sequence; observed_at_utc = observed; previous_sha256 = previous; kind;
        payload = string_of_hex payload_hex; event_sha256 = digest }
  | _ -> failf "epistemic-journal-record-malformed:seq=%d" expected_sequence

let load_events path =
  if not (Sys.file_exists path) then failf "epistemic-journal-missing:%s" path;
  let rec loop sequence previous events = function
    | [] -> (List.rev events, previous)
    | line :: rest ->
        let event = parse_event sequence previous line in
        loop (sequence + 1) event.event_sha256 (event :: events) rest
  in
  loop 1 zero_digest [] (read_lines path)

let digest_limbs digest =
  if not (valid_digest digest) then failf "epistemic-frame-digest-invalid";
  List.init 8 (fun index ->
      Int64.to_string
        (Int64.of_string ("0x" ^ String.sub digest (index * 8) 8)))

let token domain value =
  if value = "" then "0"
  else
    let digest = sha256 (domain ^ "\000" ^ value) in
    let bounded = Int64.of_string ("0x" ^ String.sub digest 0 15) in
    Int64.to_string (Int64.add bounded 1L)

let adapter_path () =
  match Sys.getenv_opt "SOUNIO_LOOM_EPISTEMIC_ADAPTER" with
  | Some path when path <> "" -> path
  | _ ->
      Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
        "sounio-loom-epistemic-runtime"

let process_exchange executable input =
  let stdin_reader, stdin_writer = Unix.pipe () in
  let output_reader, output_writer = Unix.pipe () in
  let pid =
    try
      Unix.create_process executable [| executable |] stdin_reader output_writer
        output_writer
    with error ->
      Unix.close stdin_reader;
      Unix.close stdin_writer;
      Unix.close output_reader;
      Unix.close output_writer;
      raise error
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
  let code =
    match status with
    | WEXITED value -> value
    | WSIGNALED signal | WSTOPPED signal -> 128 + signal
  in
  (code, trim (Buffer.contents output))

type transition = {
  code : int;
  name : string;
  previous_state : int;
  next_state : int;
  world : string;
  parent : string;
  subject : string;
  related : string;
  owner : string;
  generation : string;
  value_digest : string;
  error_digest : string;
  uncertainty_digest : string;
  confidence_digest : string;
  provenance_digest : string;
  evidence_digest : string;
  falsifier_digest : string;
}

let verify_native transition =
  let adapter = adapter_path () in
  if not (Sys.file_exists adapter) then
    failf "epistemic-native-adapter-missing:%s" adapter;
  let adapter = Unix.realpath adapter in
  let frame =
    [ "9008"; string_of_int transition.code;
      string_of_int transition.previous_state;
      string_of_int transition.next_state;
      token "loom-epistemic-world" transition.world;
      token "loom-epistemic-world" transition.parent;
      token "loom-epistemic-subject" transition.subject;
      token "loom-epistemic-related" transition.related;
      token "loom-epistemic-owner" transition.owner;
      token "loom-epistemic-generation" transition.generation ]
    @ digest_limbs transition.value_digest
    @ digest_limbs transition.error_digest
    @ digest_limbs transition.uncertainty_digest
    @ digest_limbs transition.confidence_digest
    @ digest_limbs transition.provenance_digest
    @ digest_limbs transition.evidence_digest
    @ digest_limbs transition.falsifier_digest
    |> String.concat " "
  in
  let code, output = process_exchange adapter (frame ^ "\n") in
  let expected =
    Printf.sprintf
      "SOUNIO_EPISTEMIC_ACCEPT schema=loom-native-epistemic-v0 transition=%s state=active"
      transition.name
  in
  if code <> 0 || output <> expected then
    failf "epistemic-native-transition-refused:%s:rc=%d:output=%s"
      transition.name code output

let axis_digest axis value =
  sha256 (String.concat "\000" [ "loom-epistemic-axis-v0"; axis; value ])

let zero_transition code name previous_state next_state world =
  { code; name; previous_state; next_state; world; parent = ""; subject = "";
    related = ""; owner = ""; generation = "";
    value_digest = zero_digest; error_digest = zero_digest;
    uncertainty_digest = zero_digest; confidence_digest = zero_digest;
    provenance_digest = zero_digest; evidence_digest = zero_digest;
    falsifier_digest = zero_digest }

let transition_of_event event =
  let fields = decode_fields event.payload in
  let world = field fields "world" in
  match event.kind with
  | "WORLD_CREATED" -> zero_transition 1 "create" 0 1 world
  | "KNOWLEDGE_OBSERVED" ->
      { (zero_transition 2 "observe" 1 1 world) with
        subject = field fields "knowledge";
        value_digest = field fields "value_digest";
        error_digest = field fields "error_digest";
        uncertainty_digest = field fields "uncertainty_digest";
        confidence_digest = field fields "confidence_digest";
        provenance_digest = field fields "provenance_digest" }
  | "CLAIM_OPENED" ->
      { (zero_transition 3 "claim" 1 1 world) with
        subject = field fields "claim"; related = field fields "knowledge";
        evidence_digest = field fields "evidence_digest" }
  | "CLAIM_CHALLENGED" ->
      { (zero_transition 4 "challenge" 1 1 world) with
        subject = field fields "challenge"; related = field fields "claim";
        falsifier_digest = field fields "falsifier_digest" }
  | "CAPABILITY_ACQUIRED" ->
      { (zero_transition 5 "capability-acquire" 1 1 world) with
        subject = field fields "capability"; related = field fields "resource";
        owner = field fields "owner"; generation = field fields "generation" }
  | "CAPABILITY_RELEASED" ->
      { (zero_transition 6 "capability-release" 1 1 world) with
        subject = field fields "capability"; related = field fields "resource";
        owner = field fields "owner"; generation = field fields "generation" }
  | "WORLD_FORKED" ->
      { (zero_transition 7 "fork" 0 1 world) with
        parent = field fields "parent"; subject = field fields "hypothesis";
        provenance_digest = field fields "parent_head";
        evidence_digest = field fields "hypothesis_digest" }
  | kind -> failf "epistemic-event-kind-unknown:%s" kind

type knowledge = {
  knowledge_id : string;
  value : string;
  arithmetic_error : string;
  uncertainty : string;
  confidence : string;
  provenance_digest : string;
}

type claim = {
  claim_id : string;
  knowledge_id : string;
  evidence_digest : string;
  mutable challenge_id : string;
  mutable falsifier_digest : string;
}

type capability = {
  capability_id : string;
  resource : string;
  owner : string;
  generation : string;
  mutable released : bool;
}

type state = {
  world_id : string;
  agent : string;
  lane : string;
  parent_world : string;
  parent_head : string;
  hypothesis : string;
  events : event list;
  journal_head : string;
  knowledges : (string, knowledge) Hashtbl.t;
  claims : (string, claim) Hashtbl.t;
  capabilities : (string, capability) Hashtbl.t;
}

let confidence_value value =
  let parsed =
    try float_of_string value with _ -> failf "epistemic-confidence-invalid"
  in
  if classify_float parsed = FP_nan || classify_float parsed = FP_infinite
     || parsed < 0. || parsed > 1.
  then failf "epistemic-confidence-invalid";
  value

let reduce world events head =
  let first =
    match events with
    | event :: _ -> event
    | [] -> failf "epistemic-journal-empty:%s" world
  in
  let first_fields = decode_fields first.payload in
  let recorded_world = field first_fields "world" in
  if recorded_world <> world then failf "epistemic-world-identity-drift:%s" world;
  let agent = field first_fields "agent" in
  let lane = field first_fields "lane" in
  validate_atom "world" world;
  validate_atom "agent" agent;
  validate_atom "lane" lane;
  let parent_world, parent_head, hypothesis =
    match first.kind with
    | "WORLD_CREATED" -> ("", "", "")
    | "WORLD_FORKED" ->
        let parent = field first_fields "parent" in
        let parent_head = require_digest "parent-head" (field first_fields "parent_head") in
        let hypothesis = field first_fields "hypothesis" in
        validate_atom "parent" parent;
        validate_text "hypothesis" hypothesis;
        if parent = world then failf "epistemic-fork-self-parent:%s" world;
        (parent, parent_head, hypothesis)
    | _ -> failf "epistemic-first-event-invalid:%s" first.kind
  in
  let state =
    { world_id = world; agent; lane; parent_world; parent_head; hypothesis;
      events; journal_head = head; knowledges = Hashtbl.create 32;
      claims = Hashtbl.create 32; capabilities = Hashtbl.create 16 }
  in
  List.iteri
    (fun index event ->
      let transition = transition_of_event event in
      if transition.world <> world then failf "epistemic-event-world-drift:%d" event.sequence;
      verify_native transition;
      let fields = decode_fields event.payload in
      match event.kind with
      | "WORLD_CREATED" | "WORLD_FORKED" ->
          if index <> 0 then failf "epistemic-world-origin-duplicate:%d" event.sequence
      | "KNOWLEDGE_OBSERVED" ->
          let id = field fields "knowledge" in
          validate_atom "knowledge" id;
          if Hashtbl.mem state.knowledges id then
            failf "epistemic-knowledge-duplicate:%s" id;
          let item =
            { knowledge_id = id; value = field fields "value";
              arithmetic_error = field fields "arithmetic_error";
              uncertainty = field fields "uncertainty";
              confidence = field fields "confidence";
              provenance_digest =
                require_digest "provenance" (field fields "provenance_digest") }
          in
          validate_text "value" item.value;
          validate_text "arithmetic-error" item.arithmetic_error;
          validate_text "uncertainty" item.uncertainty;
          ignore (confidence_value item.confidence);
          Hashtbl.add state.knowledges id item
      | "CLAIM_OPENED" ->
          let id = field fields "claim" in
          let knowledge_id = field fields "knowledge" in
          validate_atom "claim" id;
          if Hashtbl.mem state.claims id then failf "epistemic-claim-duplicate:%s" id;
          if not (Hashtbl.mem state.knowledges knowledge_id) then
            failf "epistemic-claim-knowledge-missing:%s" knowledge_id;
          Hashtbl.add state.claims id
            { claim_id = id; knowledge_id;
              evidence_digest = require_digest "evidence" (field fields "evidence_digest");
              challenge_id = ""; falsifier_digest = "" }
      | "CLAIM_CHALLENGED" ->
          let challenge_id = field fields "challenge" in
          let claim_id = field fields "claim" in
          validate_atom "challenge" challenge_id;
          let item =
            match Hashtbl.find_opt state.claims claim_id with
            | Some value -> value
            | None -> failf "epistemic-challenge-claim-missing:%s" claim_id
          in
          if item.challenge_id <> "" then failf "epistemic-claim-already-challenged:%s" claim_id;
          item.challenge_id <- challenge_id;
          item.falsifier_digest <-
            require_digest "falsifier" (field fields "falsifier_digest")
      | "CAPABILITY_ACQUIRED" ->
          let id = field fields "capability" in
          validate_atom "capability" id;
          if Hashtbl.mem state.capabilities id then
            failf "epistemic-capability-duplicate:%s" id;
          let resource = field fields "resource" in
          let owner = field fields "owner" in
          let generation = field fields "generation" in
          validate_text "resource" resource;
          validate_atom "owner" owner;
          validate_atom "generation" generation;
          Hashtbl.iter
            (fun _ existing ->
              if not existing.released && existing.resource = resource then
                failf "epistemic-resource-already-owned:%s" resource)
            state.capabilities;
          Hashtbl.add state.capabilities id
            { capability_id = id; resource; owner; generation; released = false }
      | "CAPABILITY_RELEASED" ->
          let id = field fields "capability" in
          let item =
            match Hashtbl.find_opt state.capabilities id with
            | Some value -> value
            | None -> failf "epistemic-capability-missing:%s" id
          in
          if item.released then failf "epistemic-capability-already-released:%s" id;
          if item.resource <> field fields "resource"
             || item.owner <> field fields "owner"
             || item.generation <> field fields "generation"
          then failf "epistemic-capability-release-identity-drift:%s" id;
          item.released <- true
      | kind -> failf "epistemic-event-kind-unknown:%s" kind)
    events;
  state

let load_world_local root world =
  validate_atom "world" world;
  let events, head = load_events (journal_path root world) in
  reduce world events head

let world_ids root =
  let directory = worlds_dir root in
  if not (Sys.file_exists directory) then []
  else
    Sys.readdir directory |> Array.to_list |> List.sort String.compare
    |> List.filter (fun world ->
           try (Unix.stat (world_dir root world)).st_kind = S_DIR with _ -> false)

let validate_parent_binding root states =
  List.iter
    (fun state ->
      if state.parent_world <> "" then (
        if not (List.exists (fun candidate -> candidate.world_id = state.parent_world) states)
        then failf "epistemic-parent-world-missing:%s" state.parent_world;
        let parent_events, _ = load_events (journal_path root state.parent_world) in
        if not (List.exists (fun event -> event.event_sha256 = state.parent_head) parent_events)
        then failf "epistemic-parent-head-not-observed:%s" state.parent_head))
    states

let validate_global_capabilities states =
  let resources = Hashtbl.create 32 in
  List.iter
    (fun state ->
      Hashtbl.iter
        (fun _ capability ->
          if not capability.released then
            match Hashtbl.find_opt resources capability.resource with
            | None -> Hashtbl.add resources capability.resource (state.world_id, capability.capability_id)
            | Some (world, id) ->
                failf "epistemic-global-resource-conflict:%s:first=%s/%s:second=%s/%s"
                  capability.resource world id state.world_id capability.capability_id)
        state.capabilities)
    states

let load_all root =
  let states = List.map (load_world_local root) (world_ids root) in
  validate_parent_binding root states;
  validate_global_capabilities states;
  states

let find_world states world =
  match List.find_opt (fun state -> state.world_id = world) states with
  | Some state -> state
  | None -> failf "epistemic-world-missing:%s" world

let append root world kind fields =
  let path = journal_path root world in
  mkdir_p (Filename.dirname path);
  let journal_exists = Sys.file_exists path in
  let sequence, previous =
    if journal_exists then
      let events, head = load_events path in
      (List.length events + 1, head)
    else (1, zero_digest)
  in
  let payload = encode_fields fields in
  let observed_at_utc = utc_now () in
  let body = event_body sequence observed_at_utc previous kind (hex_of_string payload) in
  let event =
    { sequence; observed_at_utc; previous_sha256 = previous; kind; payload;
      event_sha256 = event_digest body }
  in
  verify_native (transition_of_event event);
  let descriptor = Unix.openfile path [ O_WRONLY; O_CREAT; O_APPEND ] 0o600 in
  Unix.set_close_on_exec descriptor;
  Fun.protect
    ~finally:(fun () -> Unix.close descriptor)
    (fun () ->
      write_all descriptor (encode_event event);
      Unix.fsync descriptor);
  if not journal_exists then fsync_directory (Filename.dirname path);
  event

let create ~root ~world ~agent ~lane =
  validate_atom "world" world;
  validate_atom "agent" agent;
  validate_atom "lane" lane;
  with_machine_lock root (fun () ->
      ignore (load_all root);
      if Sys.file_exists (journal_path root world) then
        failf "epistemic-world-exists:%s" world;
      let event =
        append root world "WORLD_CREATED"
          [ ("world", world); ("agent", agent); ("lane", lane);
            ("schema", schema) ]
      in
      ignore (load_all root);
      Printf.sprintf "LOOM_WORLD_CREATED schema=%s world=%s head=%s" schema world
        event.event_sha256)

let observe ~root ~world ~knowledge ~value ~arithmetic_error ~uncertainty
    ~confidence ~provenance =
  validate_atom "knowledge" knowledge;
  validate_text "value" value;
  validate_text "arithmetic-error" arithmetic_error;
  validate_text "uncertainty" uncertainty;
  let confidence = confidence_value confidence in
  let provenance = require_digest "provenance" provenance in
  with_machine_lock root (fun () ->
      let states = load_all root in
      let state = find_world states world in
      if Hashtbl.mem state.knowledges knowledge then
        failf "epistemic-knowledge-duplicate:%s" knowledge;
      let event =
        append root world "KNOWLEDGE_OBSERVED"
          [ ("world", world); ("knowledge", knowledge); ("value", value);
            ("arithmetic_error", arithmetic_error);
            ("uncertainty", uncertainty); ("confidence", confidence);
            ("value_digest", axis_digest "value" value);
            ("error_digest", axis_digest "arithmetic-error" arithmetic_error);
            ("uncertainty_digest", axis_digest "uncertainty" uncertainty);
            ("confidence_digest", axis_digest "confidence" confidence);
            ("provenance_digest", provenance) ]
      in
      ignore (load_all root);
      Printf.sprintf
        "LOOM_KNOWLEDGE_OBSERVED world=%s knowledge=%s axes=5 head=%s"
        world knowledge event.event_sha256)

let open_claim ~root ~world ~claim ~knowledge ~evidence =
  validate_atom "claim" claim;
  validate_atom "knowledge" knowledge;
  let evidence = require_digest "evidence" evidence in
  with_machine_lock root (fun () ->
      let state = find_world (load_all root) world in
      if not (Hashtbl.mem state.knowledges knowledge) then
        failf "epistemic-claim-knowledge-missing:%s" knowledge;
      if Hashtbl.mem state.claims claim then failf "epistemic-claim-duplicate:%s" claim;
      let event =
        append root world "CLAIM_OPENED"
          [ ("world", world); ("claim", claim); ("knowledge", knowledge);
            ("evidence_digest", evidence) ]
      in
      ignore (load_all root);
      Printf.sprintf "LOOM_CLAIM_OPENED world=%s claim=%s knowledge=%s head=%s"
        world claim knowledge event.event_sha256)

let challenge ~root ~world ~claim ~challenge ~falsifier =
  validate_atom "claim" claim;
  validate_atom "challenge" challenge;
  let falsifier = require_digest "falsifier" falsifier in
  with_machine_lock root (fun () ->
      let state = find_world (load_all root) world in
      let item =
        match Hashtbl.find_opt state.claims claim with
        | Some value -> value
        | None -> failf "epistemic-challenge-claim-missing:%s" claim
      in
      if item.challenge_id <> "" then failf "epistemic-claim-already-challenged:%s" claim;
      let event =
        append root world "CLAIM_CHALLENGED"
          [ ("world", world); ("claim", claim); ("challenge", challenge);
            ("falsifier_digest", falsifier) ]
      in
      ignore (load_all root);
      Printf.sprintf "LOOM_CLAIM_CHALLENGED world=%s claim=%s challenge=%s head=%s"
        world claim challenge event.event_sha256)

let acquire_capability ~root ~world ~capability ~resource ~owner ~generation =
  validate_atom "capability" capability;
  validate_text "resource" resource;
  validate_atom "owner" owner;
  validate_atom "generation" generation;
  with_machine_lock root (fun () ->
      let states = load_all root in
      ignore (find_world states world);
      List.iter
        (fun state ->
          Hashtbl.iter
            (fun _ existing ->
              if not existing.released && existing.resource = resource then
                failf "epistemic-global-resource-conflict:%s" resource)
            state.capabilities)
        states;
      let event =
        append root world "CAPABILITY_ACQUIRED"
          [ ("world", world); ("capability", capability);
            ("resource", resource); ("owner", owner);
            ("generation", generation) ]
      in
      ignore (load_all root);
      Printf.sprintf
        "LOOM_CAPABILITY_ACQUIRED world=%s capability=%s resource_sha256=%s head=%s"
        world capability (sha256 resource) event.event_sha256)

let release_capability ~root ~world ~capability ~owner ~generation =
  validate_atom "capability" capability;
  validate_atom "owner" owner;
  validate_atom "generation" generation;
  with_machine_lock root (fun () ->
      let state = find_world (load_all root) world in
      let item =
        match Hashtbl.find_opt state.capabilities capability with
        | Some value -> value
        | None -> failf "epistemic-capability-missing:%s" capability
      in
      if item.released then failf "epistemic-capability-already-released:%s" capability;
      if item.owner <> owner || item.generation <> generation then
        failf "epistemic-capability-release-identity-drift:%s" capability;
      let event =
        append root world "CAPABILITY_RELEASED"
          [ ("world", world); ("capability", capability);
            ("resource", item.resource); ("owner", owner);
            ("generation", generation) ]
      in
      ignore (load_all root);
      Printf.sprintf "LOOM_CAPABILITY_RELEASED world=%s capability=%s head=%s"
        world capability event.event_sha256)

let fork ~root ~parent ~child ~agent ~lane ~hypothesis ~expected_parent_head =
  validate_atom "parent" parent;
  validate_atom "child" child;
  validate_atom "agent" agent;
  validate_atom "lane" lane;
  validate_text "hypothesis" hypothesis;
  with_machine_lock root (fun () ->
      let states = load_all root in
      let parent_state = find_world states parent in
      if List.exists (fun state -> state.world_id = child) states then
        failf "epistemic-world-exists:%s" child;
      if expected_parent_head <> ""
         && String.lowercase_ascii expected_parent_head <> parent_state.journal_head
      then failf "epistemic-parent-head-mismatch:expected=%s:actual=%s"
          expected_parent_head parent_state.journal_head;
      let event =
        append root child "WORLD_FORKED"
          [ ("world", child); ("agent", agent); ("lane", lane);
            ("schema", schema); ("parent", parent);
            ("parent_head", parent_state.journal_head);
            ("hypothesis", hypothesis);
            ("hypothesis_digest", axis_digest "hypothesis" hypothesis) ]
      in
      ignore (load_all root);
      Printf.sprintf
        "LOOM_WORLD_FORKED schema=%s parent=%s child=%s parent_head=%s head=%s"
        schema parent child parent_state.journal_head event.event_sha256)

let status ~root ~world =
  with_machine_lock root (fun () ->
      let state = find_world (load_all root) world in
      let challenged =
        Hashtbl.fold
          (fun _ claim count -> if claim.challenge_id = "" then count else count + 1)
          state.claims 0
      in
      let live_capabilities =
        Hashtbl.fold
          (fun _ capability count -> if capability.released then count else count + 1)
          state.capabilities 0
      in
      Printf.sprintf
        "LOOM_WORLD_OK schema=%s world=%s events=%d knowledge=%d claims=%d challenged=%d live_capabilities=%d parent=%s head=%s"
        schema world (List.length state.events) (Hashtbl.length state.knowledges)
        (Hashtbl.length state.claims) challenged live_capabilities
        (if state.parent_world = "" then "-" else state.parent_world)
        state.journal_head)

let verify ~root ~world = status ~root ~world

let list ~root =
  with_machine_lock root (fun () ->
      load_all root
      |> List.map (fun state ->
             Printf.sprintf "LOOM_WORLD world=%s events=%d parent=%s head=%s"
               state.world_id (List.length state.events)
               (if state.parent_world = "" then "-" else state.parent_world)
               state.journal_head)
      |> fun rows ->
      String.concat "\n" (rows @ [ Printf.sprintf "loom_worlds=%d" (List.length rows) ]))

type spectral_event = {
  spectral_world : string;
  spectral_agent : string;
  spectral_lane : string;
  spectral_state : string;
  spectral_sequence : int64;
  spectral_observed_at_utc : string;
  spectral_kind : string;
  spectral_payload : string;
  spectral_previous_sha256 : string;
  spectral_event_sha256 : string;
  spectral_head_sha256 : string;
}

let spectral_events root =
  with_machine_lock root (fun () ->
      load_all root
      |> List.fold_left
           (fun rows state ->
             List.rev_append
               (List.map
                  (fun event ->
                    { spectral_world = state.world_id;
                      spectral_agent = state.agent;
                      spectral_lane = state.lane;
                      spectral_state = "active";
                      spectral_sequence = Int64.of_int event.sequence;
                      spectral_observed_at_utc = event.observed_at_utc;
                      spectral_kind = event.kind;
                      spectral_payload = event.payload;
                      spectral_previous_sha256 = event.previous_sha256;
                      spectral_event_sha256 = event.event_sha256;
                      spectral_head_sha256 = state.journal_head })
                  state.events)
               rows)
           [])
