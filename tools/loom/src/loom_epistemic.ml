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

type attention_policy =
  | Information_first
  | Falsification_first
  | Counterfactual_first

let attention_policy_name = function
  | Information_first -> "information-first"
  | Falsification_first -> "falsification-first"
  | Counterfactual_first -> "counterfactual-first"

let attention_policy_code = function
  | Information_first -> 1
  | Falsification_first -> 2
  | Counterfactual_first -> 3

let attention_policy_of_string = function
  | "information-first" -> Information_first
  | "falsification-first" -> Falsification_first
  | "counterfactual-first" -> Counterfactual_first
  | value -> failf "epistemic-attention-policy-invalid:%s" value

type attention_candidate = {
  candidate_id : string;
  target_world : string;
  target_claim : string;
  provider : string;
  resource : string;
  information_gain : int;
  falsification_power : int;
  counterfactual_divergence : int;
  cost : int;
  risk : int;
  candidate_evidence_digest : string;
  candidate_falsifier_digest : string;
}

let attention_candidate_header =
  String.concat "\t"
    [ "candidate_id"; "target_world"; "claim"; "provider"; "resource";
      "information"; "falsification"; "divergence"; "cost"; "risk";
      "evidence_sha256"; "falsifier_sha256" ]

let attention_int label ~minimum ~maximum value =
  let parsed =
    try int_of_string value
    with _ -> failf "epistemic-attention-%s-invalid" label
  in
  if parsed < minimum || parsed > maximum then
    failf "epistemic-attention-%s-invalid" label;
  parsed

let canonical_attention_candidate candidate =
  String.concat "\t"
    [ candidate.candidate_id; candidate.target_world; candidate.target_claim;
      candidate.provider; candidate.resource;
      string_of_int candidate.information_gain;
      string_of_int candidate.falsification_power;
      string_of_int candidate.counterfactual_divergence;
      string_of_int candidate.cost; string_of_int candidate.risk;
      candidate.candidate_evidence_digest;
      candidate.candidate_falsifier_digest ]

let attention_candidate_of_line line =
  match String.split_on_char '\t' line with
  | [ candidate_id; target_world; target_claim; provider; resource;
      information; falsification; divergence; cost; risk; evidence;
      falsifier ] ->
      validate_atom "attention-candidate" candidate_id;
      validate_atom "attention-target-world" target_world;
      validate_atom "attention-target-claim" target_claim;
      validate_atom "attention-provider" provider;
      validate_text "attention-resource" resource;
      if String.contains resource '\t' || String.contains resource '\n'
         || String.contains resource '\r'
      then failf "epistemic-attention-resource-invalid";
      { candidate_id; target_world; target_claim; provider; resource;
        information_gain =
          attention_int "information" ~minimum:1 ~maximum:1000 information;
        falsification_power =
          attention_int "falsification" ~minimum:1 ~maximum:1000 falsification;
        counterfactual_divergence =
          attention_int "divergence" ~minimum:1 ~maximum:1000 divergence;
        cost = attention_int "cost" ~minimum:1 ~maximum:1_000_000 cost;
        risk = attention_int "risk" ~minimum:0 ~maximum:1000 risk;
        candidate_evidence_digest = require_digest "attention-evidence" evidence;
        candidate_falsifier_digest =
          require_digest "attention-falsifier" falsifier }
  | _ -> failf "epistemic-attention-candidate-row-malformed"

let canonical_attention_candidates candidates =
  String.concat "\n"
    (attention_candidate_header
     :: List.map canonical_attention_candidate candidates)
  ^ "\n"

let parse_attention_candidate_lines lines =
  match lines with
  | header :: rows when header = attention_candidate_header ->
      if rows = [] || List.length rows > 64 then
        failf "epistemic-attention-candidate-count-invalid";
      if List.exists (fun row -> row = "") rows then
        failf "epistemic-attention-candidate-row-malformed";
      let candidates = List.map attention_candidate_of_line rows in
      let seen = Hashtbl.create (List.length candidates) in
      List.iter
        (fun candidate ->
          if Hashtbl.mem seen candidate.candidate_id then
            failf "epistemic-attention-candidate-duplicate:%s"
              candidate.candidate_id;
          Hashtbl.add seen candidate.candidate_id ())
        candidates;
      let canonical = canonical_attention_candidates candidates in
      if String.length canonical > 65_536 then
        failf "epistemic-attention-candidate-set-too-large";
      (candidates, canonical)
  | _ -> failf "epistemic-attention-candidate-header-invalid"

let parse_attention_candidate_text value =
  let lines = String.split_on_char '\n' value in
  let lines =
    match List.rev lines with
    | "" :: rest -> List.rev rest
    | _ -> lines
  in
  parse_attention_candidate_lines lines

let compare_high left right = Int.compare right left
let compare_low left right = Int.compare left right

let first_comparison comparisons =
  let rec loop = function
    | [] -> 0
    | value :: rest -> if value = 0 then loop rest else value
  in
  loop comparisons

let compare_attention_candidates policy left right =
  let information = compare_high left.information_gain right.information_gain in
  let falsification =
    compare_high left.falsification_power right.falsification_power
  in
  let divergence =
    compare_high left.counterfactual_divergence
      right.counterfactual_divergence
  in
  let risk = compare_low left.risk right.risk in
  let cost = compare_low left.cost right.cost in
  let policy_axes =
    match policy with
    | Information_first -> [ information; falsification; divergence ]
    | Falsification_first -> [ falsification; information; divergence ]
    | Counterfactual_first -> [ divergence; falsification; information ]
  in
  first_comparison
    (policy_axes @ [ risk; cost; String.compare left.candidate_id right.candidate_id ])

let select_attention_candidate policy budget candidates =
  if budget <= 0 || budget > 1_000_000 then
    failf "epistemic-attention-budget-invalid";
  match
    candidates
    |> List.filter (fun candidate -> candidate.cost <= budget)
    |> List.sort (compare_attention_candidates policy)
  with
  | selected :: _ -> selected
  | [] -> failf "epistemic-attention-no-feasible-candidate"

let attention_candidate_set_digest canonical =
  sha256 ("loom-attention-candidates-v0\000" ^ canonical)

let attention_adapter_path () =
  match Sys.getenv_opt "SOUNIO_LOOM_ATTENTION_ADAPTER" with
  | Some path when path <> "" -> path
  | _ ->
      Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
        "sounio-loom-attention-runtime"

let verify_attention_frame frame expected =
  let adapter = attention_adapter_path () in
  if not (Sys.file_exists adapter) then
    failf "epistemic-attention-native-adapter-missing:%s" adapter;
  let code, output =
    process_exchange (Unix.realpath adapter) (String.concat " " frame ^ "\n")
  in
  if code <> 0 || output <> expected then
    failf "epistemic-attention-native-refused:rc=%d:output=%s" code output

let verify_attention_pair ~plan ~policy ~budget ~owner ~generation
    ~candidate_set_digest ~selected ~rival =
  let zeros = List.init 8 (fun _ -> "0") in
  let frame =
    [ "9009"; "1"; string_of_int (attention_policy_code policy);
      string_of_int budget; token "loom-attention-plan" plan;
      token "loom-attention-candidate" selected.candidate_id;
      token "loom-attention-candidate" rival.candidate_id;
      token "loom-attention-owner" owner;
      token "loom-attention-generation" generation;
      string_of_int selected.information_gain;
      string_of_int selected.falsification_power;
      string_of_int selected.counterfactual_divergence;
      string_of_int selected.cost; string_of_int selected.risk;
      string_of_int rival.information_gain;
      string_of_int rival.falsification_power;
      string_of_int rival.counterfactual_divergence;
      string_of_int rival.cost; string_of_int rival.risk ]
    @ digest_limbs selected.candidate_evidence_digest
    @ digest_limbs selected.candidate_falsifier_digest
    @ digest_limbs candidate_set_digest @ zeros
  in
  verify_attention_frame frame
    (Printf.sprintf
       "SOUNIO_ATTENTION_ACCEPT schema=loom-native-attention-v0 transition=compile policy=%s"
       (attention_policy_name policy))

let verify_attention_completion ~plan ~candidate ~owner ~generation ~outcome =
  let zeros = List.init 8 (fun _ -> "0") in
  let frame =
    [ "9009"; "2"; "0"; "0"; token "loom-attention-plan" plan;
      token "loom-attention-candidate" candidate; "0";
      token "loom-attention-owner" owner;
      token "loom-attention-generation" generation ]
    @ List.init 10 (fun _ -> "0")
    @ zeros @ zeros @ zeros @ digest_limbs outcome
  in
  verify_attention_frame frame
    "SOUNIO_ATTENTION_ACCEPT schema=loom-native-attention-v0 transition=complete state=completed"

type attention_decision = {
  plan_id : string;
  policy : attention_policy;
  budget : int;
  candidates : attention_candidate list;
  canonical_candidates : string;
  candidate_set_digest : string;
  selected : attention_candidate;
  attention_owner : string;
  attention_generation : string;
  mutable completed : bool;
  mutable outcome_digest : string;
}

let attention_decision_of_event event =
  let fields = decode_fields event.payload in
  let plan_id = field fields "plan" in
  let policy = attention_policy_of_string (field fields "policy") in
  let budget =
    attention_int "budget" ~minimum:1 ~maximum:1_000_000
      (field fields "budget")
  in
  let stored_candidates = field fields "candidate_set" in
  let candidates, canonical_candidates =
    parse_attention_candidate_text stored_candidates
  in
  if stored_candidates <> canonical_candidates then
    failf "epistemic-attention-candidate-set-noncanonical:%s" plan_id;
  let candidate_set_digest =
    require_digest "attention-candidate-set"
      (field fields "candidate_set_digest")
  in
  if attention_candidate_set_digest canonical_candidates <> candidate_set_digest
  then failf "epistemic-attention-candidate-set-digest-mismatch:%s" plan_id;
  let selected = select_attention_candidate policy budget candidates in
  let selected_id = field fields "selected" in
  if selected.candidate_id <> selected_id then
    failf "epistemic-attention-selection-mismatch:expected=%s:actual=%s"
      selected.candidate_id selected_id;
  let attention_owner = field fields "owner" in
  let attention_generation = field fields "generation" in
  validate_atom "attention-plan" plan_id;
  validate_atom "attention-owner" attention_owner;
  validate_atom "attention-generation" attention_generation;
  let require_selected key actual =
    if field fields key <> actual then
      failf "epistemic-attention-selected-field-mismatch:%s" key
  in
  require_selected "target_world" selected.target_world;
  require_selected "claim" selected.target_claim;
  require_selected "provider" selected.provider;
  require_selected "resource" selected.resource;
  require_selected "information" (string_of_int selected.information_gain);
  require_selected "falsification" (string_of_int selected.falsification_power);
  require_selected "divergence"
    (string_of_int selected.counterfactual_divergence);
  require_selected "cost" (string_of_int selected.cost);
  require_selected "risk" (string_of_int selected.risk);
  require_selected "evidence_digest" selected.candidate_evidence_digest;
  require_selected "falsifier_digest" selected.candidate_falsifier_digest;
  List.iter
    (fun rival ->
      verify_attention_pair ~plan:plan_id ~policy ~budget
        ~owner:attention_owner ~generation:attention_generation
        ~candidate_set_digest ~selected ~rival)
    candidates;
  { plan_id; policy; budget; candidates; canonical_candidates;
    candidate_set_digest; selected; attention_owner; attention_generation;
    completed = false; outcome_digest = "" }

type portfolio_budget = {
  portfolio_token_budget : int;
  portfolio_wall_budget : int;
  portfolio_gpu_budget : int;
  portfolio_quota_budget : int;
}

type portfolio_candidate = {
  portfolio_candidate_id : string;
  portfolio_target_world : string;
  portfolio_target_claim : string;
  portfolio_provider : string;
  portfolio_resources : string list;
  portfolio_information : int;
  portfolio_falsification : int;
  portfolio_divergence : int;
  portfolio_token_cost : int;
  portfolio_wall_cost : int;
  portfolio_gpu_cost : int;
  portfolio_quota_cost : int;
  portfolio_risk : int;
  portfolio_evidence_digest : string;
  portfolio_falsifier_digest : string;
}

type portfolio_aggregate = {
  portfolio_candidates : portfolio_candidate list;
  portfolio_selected_resources : string list;
  aggregate_information : int;
  aggregate_falsification : int;
  aggregate_divergence : int;
  aggregate_token_cost : int;
  aggregate_wall_cost : int;
  aggregate_gpu_cost : int;
  aggregate_quota_cost : int;
  aggregate_risk : int;
}

let portfolio_candidate_header =
  String.concat "\t"
    [ "candidate_id"; "target_world"; "claim"; "provider"; "resources";
      "information"; "falsification"; "divergence"; "token_cost";
      "wall_cost"; "gpu_cost"; "quota_cost"; "risk";
      "evidence_sha256"; "falsifier_sha256" ]

let portfolio_int label ~minimum ~maximum value =
  let parsed =
    try int_of_string value
    with _ -> failf "epistemic-portfolio-%s-invalid" label
  in
  if parsed < minimum || parsed > maximum then
    failf "epistemic-portfolio-%s-invalid" label;
  parsed

let portfolio_resources_of_string value =
  validate_text "portfolio-resources" value;
  if String.contains value '\t' || String.contains value '\n'
     || String.contains value '\r'
  then failf "epistemic-portfolio-resources-invalid";
  let resources = String.split_on_char ',' value in
  List.iter
    (fun resource ->
      validate_text "portfolio-resource" resource;
      if String.contains resource ',' then
        failf "epistemic-portfolio-resource-invalid")
    resources;
  let canonical = List.sort_uniq String.compare resources in
  if resources <> canonical then
    failf "epistemic-portfolio-resources-noncanonical";
  resources

let canonical_portfolio_candidate candidate =
  String.concat "\t"
    [ candidate.portfolio_candidate_id; candidate.portfolio_target_world;
      candidate.portfolio_target_claim; candidate.portfolio_provider;
      String.concat "," candidate.portfolio_resources;
      string_of_int candidate.portfolio_information;
      string_of_int candidate.portfolio_falsification;
      string_of_int candidate.portfolio_divergence;
      string_of_int candidate.portfolio_token_cost;
      string_of_int candidate.portfolio_wall_cost;
      string_of_int candidate.portfolio_gpu_cost;
      string_of_int candidate.portfolio_quota_cost;
      string_of_int candidate.portfolio_risk;
      candidate.portfolio_evidence_digest;
      candidate.portfolio_falsifier_digest ]

let portfolio_candidate_of_line line =
  match String.split_on_char '\t' line with
  | [ candidate_id; target_world; target_claim; provider; resources;
      information; falsification; divergence; token_cost; wall_cost;
      gpu_cost; quota_cost; risk; evidence; falsifier ] ->
      validate_atom "portfolio-candidate" candidate_id;
      validate_atom "portfolio-target-world" target_world;
      validate_atom "portfolio-target-claim" target_claim;
      validate_atom "portfolio-provider" provider;
      { portfolio_candidate_id = candidate_id;
        portfolio_target_world = target_world;
        portfolio_target_claim = target_claim;
        portfolio_provider = provider;
        portfolio_resources = portfolio_resources_of_string resources;
        portfolio_information =
          portfolio_int "information" ~minimum:1 ~maximum:1000 information;
        portfolio_falsification =
          portfolio_int "falsification" ~minimum:1 ~maximum:1000 falsification;
        portfolio_divergence =
          portfolio_int "divergence" ~minimum:1 ~maximum:1000 divergence;
        portfolio_token_cost =
          portfolio_int "token-cost" ~minimum:1 ~maximum:1_000_000 token_cost;
        portfolio_wall_cost =
          portfolio_int "wall-cost" ~minimum:1 ~maximum:1_000_000 wall_cost;
        portfolio_gpu_cost =
          portfolio_int "gpu-cost" ~minimum:0 ~maximum:1_000_000 gpu_cost;
        portfolio_quota_cost =
          portfolio_int "quota-cost" ~minimum:0 ~maximum:1_000_000 quota_cost;
        portfolio_risk =
          portfolio_int "risk" ~minimum:0 ~maximum:1000 risk;
        portfolio_evidence_digest =
          require_digest "portfolio-evidence" evidence;
        portfolio_falsifier_digest =
          require_digest "portfolio-falsifier" falsifier }
  | _ -> failf "epistemic-portfolio-candidate-row-malformed"

let canonical_portfolio_candidates candidates =
  String.concat "\n"
    (portfolio_candidate_header
     :: List.map canonical_portfolio_candidate candidates)
  ^ "\n"

let parse_portfolio_candidate_lines lines =
  match lines with
  | header :: rows when header = portfolio_candidate_header ->
      if rows = [] || List.length rows > 18 then
        failf "epistemic-portfolio-candidate-count-invalid";
      if List.exists (fun row -> row = "") rows then
        failf "epistemic-portfolio-candidate-row-malformed";
      let candidates =
        List.map portfolio_candidate_of_line rows
        |> List.sort (fun left right ->
               String.compare left.portfolio_candidate_id
                 right.portfolio_candidate_id)
      in
      let rec reject_duplicate_ids = function
        | left :: (right :: _ as rest) ->
            if left.portfolio_candidate_id = right.portfolio_candidate_id then
              failf "epistemic-portfolio-candidate-duplicate:%s"
                left.portfolio_candidate_id;
            reject_duplicate_ids rest
        | _ -> ()
      in
      reject_duplicate_ids candidates;
      let canonical = canonical_portfolio_candidates candidates in
      if String.length canonical > 65_536 then
        failf "epistemic-portfolio-candidate-set-too-large";
      (candidates, canonical)
  | _ -> failf "epistemic-portfolio-candidate-header-invalid"

let parse_portfolio_candidate_text value =
  let lines = String.split_on_char '\n' value in
  let lines =
    match List.rev lines with
    | "" :: rest -> List.rev rest
    | _ -> lines
  in
  parse_portfolio_candidate_lines lines

let portfolio_budget_of_strings token_budget wall_budget gpu_budget quota_budget =
  { portfolio_token_budget =
      portfolio_int "token-budget" ~minimum:1 ~maximum:18_000_000 token_budget;
    portfolio_wall_budget =
      portfolio_int "wall-budget" ~minimum:1 ~maximum:18_000_000 wall_budget;
    portfolio_gpu_budget =
      portfolio_int "gpu-budget" ~minimum:0 ~maximum:18_000_000 gpu_budget;
    portfolio_quota_budget =
      portfolio_int "quota-budget" ~minimum:0 ~maximum:18_000_000 quota_budget }

let portfolio_budget ~token_budget ~wall_budget ~gpu_budget ~quota_budget =
  portfolio_budget_of_strings (string_of_int token_budget)
    (string_of_int wall_budget) (string_of_int gpu_budget)
    (string_of_int quota_budget)

let enumerate_feasible_portfolios budget candidates =
  let items = Array.of_list candidates in
  let count = Array.length items in
  if count = 0 || count > 18 then
    failf "epistemic-portfolio-candidate-count-invalid";
  let portfolios = ref [] in
  for mask = 1 to (1 lsl count) - 1 do
    let selected = ref [] in
    let resources = Hashtbl.create 16 in
    let feasible = ref true in
    let information = ref 0 in
    let falsification = ref 0 in
    let divergence = ref 0 in
    let token_cost = ref 0 in
    let wall_cost = ref 0 in
    let gpu_cost = ref 0 in
    let quota_cost = ref 0 in
    let risk = ref 0 in
    for index = 0 to count - 1 do
      if mask land (1 lsl index) <> 0 then (
        let candidate = items.(index) in
        List.iter
          (fun resource ->
            if Hashtbl.mem resources resource then feasible := false
            else Hashtbl.add resources resource ())
          candidate.portfolio_resources;
        selected := candidate :: !selected;
        information := !information + candidate.portfolio_information;
        falsification := !falsification + candidate.portfolio_falsification;
        divergence := !divergence + candidate.portfolio_divergence;
        token_cost := !token_cost + candidate.portfolio_token_cost;
        wall_cost := !wall_cost + candidate.portfolio_wall_cost;
        gpu_cost := !gpu_cost + candidate.portfolio_gpu_cost;
        quota_cost := !quota_cost + candidate.portfolio_quota_cost;
        risk := !risk + candidate.portfolio_risk)
    done;
    if !token_cost > budget.portfolio_token_budget
       || !wall_cost > budget.portfolio_wall_budget
       || !gpu_cost > budget.portfolio_gpu_budget
       || !quota_cost > budget.portfolio_quota_budget
    then feasible := false;
    if !feasible then
      let selected_resources =
        Hashtbl.fold (fun resource () values -> resource :: values)
          resources []
        |> List.sort String.compare
      in
      portfolios :=
        { portfolio_candidates = List.rev !selected;
          portfolio_selected_resources = selected_resources;
          aggregate_information = !information;
          aggregate_falsification = !falsification;
          aggregate_divergence = !divergence;
          aggregate_token_cost = !token_cost;
          aggregate_wall_cost = !wall_cost;
          aggregate_gpu_cost = !gpu_cost;
          aggregate_quota_cost = !quota_cost;
          aggregate_risk = !risk }
        :: !portfolios
  done;
  match List.rev !portfolios with
  | [] -> failf "epistemic-portfolio-no-feasible-subset"
  | values -> values

let portfolio_selected_set portfolio =
  portfolio.portfolio_candidates
  |> List.map (fun candidate -> candidate.portfolio_candidate_id)
  |> String.concat ","

let portfolio_candidate_set_digest canonical =
  sha256 ("loom-portfolio-candidates-v0\000" ^ canonical)

let portfolio_selected_set_digest selected_set =
  sha256 ("loom-portfolio-selected-set-v0\000" ^ selected_set)

let portfolio_evidence_digest portfolio =
  portfolio.portfolio_candidates
  |> List.map (fun candidate ->
         candidate.portfolio_candidate_id ^ "="
         ^ candidate.portfolio_evidence_digest)
  |> String.concat "\n"
  |> fun value -> sha256 ("loom-portfolio-evidence-v0\000" ^ value)

let portfolio_falsifier_digest portfolio =
  portfolio.portfolio_candidates
  |> List.map (fun candidate ->
         candidate.portfolio_candidate_id ^ "="
         ^ candidate.portfolio_falsifier_digest)
  |> String.concat "\n"
  |> fun value -> sha256 ("loom-portfolio-falsifier-v0\000" ^ value)

let portfolio_dominates left right =
  let no_worse =
    left.aggregate_information >= right.aggregate_information
    && left.aggregate_falsification >= right.aggregate_falsification
    && left.aggregate_divergence >= right.aggregate_divergence
    && left.aggregate_risk <= right.aggregate_risk
    && left.aggregate_token_cost <= right.aggregate_token_cost
    && left.aggregate_wall_cost <= right.aggregate_wall_cost
    && left.aggregate_gpu_cost <= right.aggregate_gpu_cost
    && left.aggregate_quota_cost <= right.aggregate_quota_cost
  in
  let strictly_better =
    left.aggregate_information > right.aggregate_information
    || left.aggregate_falsification > right.aggregate_falsification
    || left.aggregate_divergence > right.aggregate_divergence
    || left.aggregate_risk < right.aggregate_risk
    || left.aggregate_token_cost < right.aggregate_token_cost
    || left.aggregate_wall_cost < right.aggregate_wall_cost
    || left.aggregate_gpu_cost < right.aggregate_gpu_cost
    || left.aggregate_quota_cost < right.aggregate_quota_cost
  in
  no_worse && strictly_better

let portfolio_frontier_limit = 256
let portfolio_frontier_bytes_limit = 1_048_576

let pareto_frontier portfolios =
  let add frontier candidate =
    if List.exists (fun incumbent -> portfolio_dominates incumbent candidate)
         frontier
    then frontier
    else (
      let updated =
        candidate
        :: List.filter
             (fun incumbent -> not (portfolio_dominates candidate incumbent))
             frontier
      in
      let size = List.length updated in
      if size > portfolio_frontier_limit then
        failf "epistemic-portfolio-frontier-limit-exceeded:%d" size;
      updated)
  in
  List.fold_left add [] portfolios
  |> List.sort (fun left right ->
         String.compare (portfolio_selected_set left)
           (portfolio_selected_set right))

let compare_portfolios policy left right =
  let information =
    compare_high left.aggregate_information right.aggregate_information
  in
  let falsification =
    compare_high left.aggregate_falsification right.aggregate_falsification
  in
  let divergence =
    compare_high left.aggregate_divergence right.aggregate_divergence
  in
  let policy_axes =
    match policy with
    | Information_first -> [ information; falsification; divergence ]
    | Falsification_first -> [ falsification; information; divergence ]
    | Counterfactual_first -> [ divergence; falsification; information ]
  in
  first_comparison
    (policy_axes
     @ [ compare_low left.aggregate_risk right.aggregate_risk;
         compare_low left.aggregate_token_cost right.aggregate_token_cost;
         compare_low left.aggregate_wall_cost right.aggregate_wall_cost;
         compare_low left.aggregate_gpu_cost right.aggregate_gpu_cost;
         compare_low left.aggregate_quota_cost right.aggregate_quota_cost;
         String.compare (portfolio_selected_set left)
           (portfolio_selected_set right) ])

let select_portfolio policy frontier =
  match List.sort (compare_portfolios policy) frontier with
  | selected :: _ -> selected
  | [] -> failf "epistemic-portfolio-frontier-empty"

let portfolio_frontier_header =
  String.concat "\t"
    [ "selected_ids"; "resources"; "information"; "falsification";
      "divergence"; "risk"; "token_cost"; "wall_cost"; "gpu_cost";
      "quota_cost"; "evidence_sha256"; "falsifier_sha256" ]

let canonical_portfolio_frontier frontier =
  let row portfolio =
    String.concat "\t"
      [ portfolio_selected_set portfolio;
        String.concat "," portfolio.portfolio_selected_resources;
        string_of_int portfolio.aggregate_information;
        string_of_int portfolio.aggregate_falsification;
        string_of_int portfolio.aggregate_divergence;
        string_of_int portfolio.aggregate_risk;
        string_of_int portfolio.aggregate_token_cost;
        string_of_int portfolio.aggregate_wall_cost;
        string_of_int portfolio.aggregate_gpu_cost;
        string_of_int portfolio.aggregate_quota_cost;
        portfolio_evidence_digest portfolio;
        portfolio_falsifier_digest portfolio ]
  in
  let canonical =
    String.concat "\n" (portfolio_frontier_header :: List.map row frontier) ^ "\n"
  in
  if String.length canonical > portfolio_frontier_bytes_limit then
    failf "epistemic-portfolio-frontier-bytes-exceeded:%d"
      (String.length canonical);
  canonical

let portfolio_frontier_digest canonical =
  sha256 ("loom-portfolio-frontier-v0\000" ^ canonical)

let portfolio_adapter_path () =
  match Sys.getenv_opt "SOUNIO_LOOM_PORTFOLIO_ADAPTER" with
  | Some path when path <> "" -> path
  | _ ->
      Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
        "sounio-loom-portfolio-runtime"

let verify_portfolio_frame frame expected =
  let adapter = portfolio_adapter_path () in
  if not (Sys.file_exists adapter) then
    failf "epistemic-portfolio-native-adapter-missing:%s" adapter;
  let code, output =
    process_exchange (Unix.realpath adapter) (String.concat " " frame ^ "\n")
  in
  if code <> 0 || output <> expected then
    failf "epistemic-portfolio-native-refused:rc=%d:output=%s" code output

let verify_portfolio_pair ~portfolio_id ~policy ~budget ~owner ~generation
    ~candidate_set_digest ~frontier_digest ~selected ~rival =
  let selected_set = portfolio_selected_set selected in
  let rival_set = portfolio_selected_set rival in
  let zeros = List.init 8 (fun _ -> "0") in
  let frame =
    [ "9010"; "1"; string_of_int (attention_policy_code policy);
      string_of_int budget.portfolio_token_budget;
      string_of_int budget.portfolio_wall_budget;
      string_of_int budget.portfolio_gpu_budget;
      string_of_int budget.portfolio_quota_budget;
      token "loom-attention-portfolio" portfolio_id;
      token "loom-portfolio-selected-set" selected_set;
      token "loom-portfolio-rival-set" rival_set;
      token "loom-portfolio-owner" owner;
      token "loom-portfolio-generation" generation;
      string_of_int selected.aggregate_information;
      string_of_int selected.aggregate_falsification;
      string_of_int selected.aggregate_divergence;
      string_of_int selected.aggregate_risk;
      string_of_int selected.aggregate_token_cost;
      string_of_int selected.aggregate_wall_cost;
      string_of_int selected.aggregate_gpu_cost;
      string_of_int selected.aggregate_quota_cost;
      string_of_int rival.aggregate_information;
      string_of_int rival.aggregate_falsification;
      string_of_int rival.aggregate_divergence;
      string_of_int rival.aggregate_risk;
      string_of_int rival.aggregate_token_cost;
      string_of_int rival.aggregate_wall_cost;
      string_of_int rival.aggregate_gpu_cost;
      string_of_int rival.aggregate_quota_cost ]
    @ digest_limbs candidate_set_digest @ digest_limbs frontier_digest
    @ digest_limbs (portfolio_selected_set_digest selected_set)
    @ digest_limbs (portfolio_evidence_digest selected)
    @ digest_limbs (portfolio_falsifier_digest selected) @ zeros
  in
  verify_portfolio_frame frame
    (Printf.sprintf
       "SOUNIO_PORTFOLIO_ACCEPT schema=loom-native-portfolio-v0 transition=compile policy=%s"
       (attention_policy_name policy))

let verify_portfolio_completion ~portfolio_id ~selected_set ~owner ~generation
    ~outcome =
  let zeros = List.init 8 (fun _ -> "0") in
  let frame =
    [ "9010"; "2"; "0"; "0"; "0"; "0"; "0";
      token "loom-attention-portfolio" portfolio_id;
      token "loom-portfolio-selected-set" selected_set; "0";
      token "loom-portfolio-owner" owner;
      token "loom-portfolio-generation" generation ]
    @ List.init 16 (fun _ -> "0")
    @ zeros @ zeros @ zeros @ zeros @ zeros @ digest_limbs outcome
  in
  verify_portfolio_frame frame
    "SOUNIO_PORTFOLIO_ACCEPT schema=loom-native-portfolio-v0 transition=complete state=completed"

type portfolio_attention_decision = {
  portfolio_id : string;
  portfolio_policy : attention_policy;
  portfolio_budget : portfolio_budget;
  all_portfolio_candidates : portfolio_candidate list;
  canonical_portfolio_candidate_set : string;
  portfolio_candidate_set_digest : string;
  portfolio_frontier : portfolio_aggregate list;
  canonical_frontier : string;
  portfolio_frontier_digest : string;
  selected_portfolio : portfolio_aggregate;
  portfolio_owner : string;
  portfolio_generation : string;
  mutable portfolio_completed : bool;
  mutable portfolio_outcome_digest : string;
}

let portfolio_attention_decision_of_event event =
  let fields = decode_fields event.payload in
  let portfolio_id = field fields "portfolio" in
  let policy = attention_policy_of_string (field fields "policy") in
  let budget =
    portfolio_budget_of_strings (field fields "token_budget")
      (field fields "wall_budget") (field fields "gpu_budget")
      (field fields "quota_budget")
  in
  let stored_candidates = field fields "candidate_set" in
  let candidates, canonical_candidates =
    parse_portfolio_candidate_text stored_candidates
  in
  if stored_candidates <> canonical_candidates then
    failf "epistemic-portfolio-candidate-set-noncanonical:%s" portfolio_id;
  let candidate_set_digest =
    require_digest "portfolio-candidate-set"
      (field fields "candidate_set_digest")
  in
  if portfolio_candidate_set_digest canonical_candidates
     <> candidate_set_digest
  then failf "epistemic-portfolio-candidate-set-digest-mismatch:%s" portfolio_id;
  let feasible = enumerate_feasible_portfolios budget candidates in
  let frontier = pareto_frontier feasible in
  let canonical_frontier = canonical_portfolio_frontier frontier in
  if field fields "frontier" <> canonical_frontier then
    failf "epistemic-portfolio-frontier-mismatch:%s" portfolio_id;
  let frontier_digest =
    require_digest "portfolio-frontier" (field fields "frontier_digest")
  in
  if portfolio_frontier_digest canonical_frontier <> frontier_digest then
    failf "epistemic-portfolio-frontier-digest-mismatch:%s" portfolio_id;
  let selected = select_portfolio policy frontier in
  let selected_set = portfolio_selected_set selected in
  if field fields "selected_set" <> selected_set then
    failf "epistemic-portfolio-selection-mismatch:expected=%s:actual=%s"
      selected_set (field fields "selected_set");
  let selected_set_digest =
    require_digest "portfolio-selected-set"
      (field fields "selected_set_digest")
  in
  if portfolio_selected_set_digest selected_set <> selected_set_digest then
    failf "epistemic-portfolio-selected-set-digest-mismatch:%s" portfolio_id;
  let owner = field fields "owner" in
  let generation = field fields "generation" in
  validate_atom "attention-portfolio" portfolio_id;
  validate_atom "portfolio-owner" owner;
  validate_atom "portfolio-generation" generation;
  let require_selected key actual =
    if field fields key <> actual then
      failf "epistemic-portfolio-selected-field-mismatch:%s" key
  in
  require_selected "resources"
    (String.concat "," selected.portfolio_selected_resources);
  require_selected "information" (string_of_int selected.aggregate_information);
  require_selected "falsification"
    (string_of_int selected.aggregate_falsification);
  require_selected "divergence" (string_of_int selected.aggregate_divergence);
  require_selected "risk" (string_of_int selected.aggregate_risk);
  require_selected "token_cost" (string_of_int selected.aggregate_token_cost);
  require_selected "wall_cost" (string_of_int selected.aggregate_wall_cost);
  require_selected "gpu_cost" (string_of_int selected.aggregate_gpu_cost);
  require_selected "quota_cost" (string_of_int selected.aggregate_quota_cost);
  require_selected "evidence_digest" (portfolio_evidence_digest selected);
  require_selected "falsifier_digest" (portfolio_falsifier_digest selected);
  List.iter
    (fun rival ->
      verify_portfolio_pair ~portfolio_id ~policy ~budget ~owner ~generation
        ~candidate_set_digest ~frontier_digest ~selected ~rival)
    frontier;
  { portfolio_id; portfolio_policy = policy; portfolio_budget = budget;
    all_portfolio_candidates = candidates;
    canonical_portfolio_candidate_set = canonical_candidates;
    portfolio_candidate_set_digest = candidate_set_digest;
    portfolio_frontier = frontier; canonical_frontier;
    portfolio_frontier_digest = frontier_digest;
    selected_portfolio = selected; portfolio_owner = owner;
    portfolio_generation = generation; portfolio_completed = false;
    portfolio_outcome_digest = "" }

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
  attentions : (string, attention_decision) Hashtbl.t;
  attention_portfolios : (string, portfolio_attention_decision) Hashtbl.t;
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
      claims = Hashtbl.create 32; capabilities = Hashtbl.create 16;
      attentions = Hashtbl.create 16; attention_portfolios = Hashtbl.create 16 }
  in
  List.iteri
    (fun index event ->
      let fields = decode_fields event.payload in
      if field fields "world" <> world then
        failf "epistemic-event-world-drift:%d" event.sequence;
      (match event.kind with
      | "ATTENTION_COMPILED" | "ATTENTION_COMPLETED"
      | "ATTENTION_PORTFOLIO_COMPILED" | "ATTENTION_PORTFOLIO_COMPLETED" -> ()
      | _ -> verify_native (transition_of_event event));
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
          Hashtbl.iter
            (fun _ decision ->
              if not decision.completed && decision.selected.resource = resource
              then failf "epistemic-resource-already-owned:%s" resource)
            state.attentions;
          Hashtbl.iter
            (fun _ decision ->
              if not decision.portfolio_completed
                 && List.mem resource
                      decision.selected_portfolio.portfolio_selected_resources
              then failf "epistemic-resource-already-owned:%s" resource)
            state.attention_portfolios;
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
      | "ATTENTION_COMPILED" ->
          let decision = attention_decision_of_event event in
          if Hashtbl.mem state.attentions decision.plan_id then
            failf "epistemic-attention-plan-duplicate:%s" decision.plan_id;
          Hashtbl.iter
            (fun _ capability ->
              if not capability.released
                 && capability.resource = decision.selected.resource
              then
                failf "epistemic-resource-already-owned:%s"
                  decision.selected.resource)
            state.capabilities;
          Hashtbl.iter
            (fun _ existing ->
              if not existing.completed
                 && existing.selected.resource = decision.selected.resource
              then
                failf "epistemic-resource-already-owned:%s"
                  decision.selected.resource)
            state.attentions;
          Hashtbl.iter
            (fun _ existing ->
              if not existing.portfolio_completed
                 && List.mem decision.selected.resource
                      existing.selected_portfolio.portfolio_selected_resources
              then
                failf "epistemic-resource-already-owned:%s"
                  decision.selected.resource)
            state.attention_portfolios;
          Hashtbl.add state.attentions decision.plan_id decision
      | "ATTENTION_COMPLETED" ->
          let plan_id = field fields "plan" in
          let decision =
            match Hashtbl.find_opt state.attentions plan_id with
            | Some value -> value
            | None -> failf "epistemic-attention-plan-missing:%s" plan_id
          in
          if decision.completed then
            failf "epistemic-attention-plan-already-completed:%s" plan_id;
          let selected = field fields "selected" in
          let owner = field fields "owner" in
          let generation = field fields "generation" in
          if selected <> decision.selected.candidate_id
             || owner <> decision.attention_owner
             || generation <> decision.attention_generation
          then
            failf "epistemic-attention-completion-identity-drift:%s" plan_id;
          let outcome =
            require_digest "attention-outcome" (field fields "outcome_digest")
          in
          verify_attention_completion ~plan:plan_id ~candidate:selected ~owner
            ~generation ~outcome;
          decision.completed <- true;
          decision.outcome_digest <- outcome
      | "ATTENTION_PORTFOLIO_COMPILED" ->
          let decision = portfolio_attention_decision_of_event event in
          if Hashtbl.mem state.attention_portfolios decision.portfolio_id then
            failf "epistemic-attention-portfolio-duplicate:%s"
              decision.portfolio_id;
          List.iter
            (fun resource ->
              Hashtbl.iter
                (fun _ capability ->
                  if not capability.released
                     && capability.resource = resource
                  then failf "epistemic-resource-already-owned:%s" resource)
                state.capabilities;
              Hashtbl.iter
                (fun _ attention ->
                  if not attention.completed
                     && attention.selected.resource = resource
                  then failf "epistemic-resource-already-owned:%s" resource)
                state.attentions;
              Hashtbl.iter
                (fun _ existing ->
                  if not existing.portfolio_completed
                     && List.mem resource
                          existing.selected_portfolio.portfolio_selected_resources
                  then failf "epistemic-resource-already-owned:%s" resource)
                state.attention_portfolios)
            decision.selected_portfolio.portfolio_selected_resources;
          Hashtbl.add state.attention_portfolios decision.portfolio_id decision
      | "ATTENTION_PORTFOLIO_COMPLETED" ->
          let portfolio_id = field fields "portfolio" in
          let decision =
            match Hashtbl.find_opt state.attention_portfolios portfolio_id with
            | Some value -> value
            | None ->
                failf "epistemic-attention-portfolio-missing:%s" portfolio_id
          in
          if decision.portfolio_completed then
            failf "epistemic-attention-portfolio-already-completed:%s"
              portfolio_id;
          let selected_set = field fields "selected_set" in
          let owner = field fields "owner" in
          let generation = field fields "generation" in
          if selected_set <> portfolio_selected_set decision.selected_portfolio
             || owner <> decision.portfolio_owner
             || generation <> decision.portfolio_generation
          then
            failf "epistemic-portfolio-completion-identity-drift:%s"
              portfolio_id;
          let outcome =
            require_digest "portfolio-outcome" (field fields "outcome_digest")
          in
          verify_portfolio_completion ~portfolio_id ~selected_set ~owner
            ~generation ~outcome;
          decision.portfolio_completed <- true;
          decision.portfolio_outcome_digest <- outcome
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
  let reserve resource world id =
    match Hashtbl.find_opt resources resource with
    | None -> Hashtbl.add resources resource (world, id)
    | Some (first_world, first_id) ->
        failf "epistemic-global-resource-conflict:%s:first=%s/%s:second=%s/%s"
          resource first_world first_id world id
  in
  List.iter
    (fun state ->
      Hashtbl.iter
        (fun _ capability ->
          if not capability.released then
            reserve capability.resource state.world_id capability.capability_id)
        state.capabilities;
      Hashtbl.iter
        (fun _ decision ->
          if not decision.completed then
            reserve decision.selected.resource state.world_id decision.plan_id)
        state.attentions;
      Hashtbl.iter
        (fun _ decision ->
          if not decision.portfolio_completed then
            List.iter
              (fun resource ->
                reserve resource state.world_id decision.portfolio_id)
              decision.selected_portfolio.portfolio_selected_resources)
        state.attention_portfolios)
    states

let validate_attention_references states =
  let require_target target_world target_claim =
    let target =
      match
        List.find_opt (fun state -> state.world_id = target_world) states
      with
      | Some value -> value
      | None ->
          failf "epistemic-attention-target-world-missing:%s" target_world
    in
    if not (Hashtbl.mem target.claims target_claim) then
      failf "epistemic-attention-target-claim-missing:%s/%s"
        target_world target_claim
  in
  List.iter
    (fun scheduling_world ->
      Hashtbl.iter
        (fun _ decision ->
          List.iter
            (fun candidate ->
              require_target candidate.target_world candidate.target_claim)
            decision.candidates)
        scheduling_world.attentions;
      Hashtbl.iter
        (fun _ decision ->
          List.iter
            (fun candidate ->
              require_target candidate.portfolio_target_world
                candidate.portfolio_target_claim)
            decision.all_portfolio_candidates)
        scheduling_world.attention_portfolios)
    states

let load_all root =
  let states = List.map (load_world_local root) (world_ids root) in
  validate_parent_binding root states;
  validate_attention_references states;
  validate_global_capabilities states;
  states

let find_world states world =
  match List.find_opt (fun state -> state.world_id = world) states with
  | Some state -> state
  | None -> failf "epistemic-world-missing:%s" world

let append ?(verify = fun event -> verify_native (transition_of_event event))
    root world kind fields =
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
  verify event;
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
            state.capabilities;
          Hashtbl.iter
            (fun _ decision ->
              if not decision.completed && decision.selected.resource = resource
              then failf "epistemic-global-resource-conflict:%s" resource)
            state.attentions;
          Hashtbl.iter
            (fun _ decision ->
              if not decision.portfolio_completed
                 && List.mem resource
                      decision.selected_portfolio.portfolio_selected_resources
              then failf "epistemic-global-resource-conflict:%s" resource)
            state.attention_portfolios)
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

let compile_attention ~root ~world ~plan ~candidate_file ~budget ~policy
    ~owner ~generation =
  validate_atom "world" world;
  validate_atom "attention-plan" plan;
  validate_atom "attention-owner" owner;
  validate_atom "attention-generation" generation;
  if not (Sys.file_exists candidate_file) then
    failf "epistemic-attention-candidate-file-missing:%s" candidate_file;
  let candidates, canonical_candidates =
    parse_attention_candidate_lines (read_lines candidate_file)
  in
  let selected = select_attention_candidate policy budget candidates in
  let candidate_set_digest =
    attention_candidate_set_digest canonical_candidates
  in
  with_machine_lock root (fun () ->
      let states = load_all root in
      let scheduling_world = find_world states world in
      if Hashtbl.mem scheduling_world.attentions plan then
        failf "epistemic-attention-plan-duplicate:%s" plan;
      List.iter
        (fun candidate ->
          let target = find_world states candidate.target_world in
          if not (Hashtbl.mem target.claims candidate.target_claim) then
            failf "epistemic-attention-target-claim-missing:%s/%s"
              candidate.target_world candidate.target_claim)
        candidates;
      List.iter
        (fun state ->
          Hashtbl.iter
            (fun _ capability ->
              if not capability.released
                 && capability.resource = selected.resource
              then
                failf "epistemic-global-resource-conflict:%s"
                  selected.resource)
            state.capabilities;
          Hashtbl.iter
            (fun _ decision ->
              if not decision.completed
                 && decision.selected.resource = selected.resource
              then
                failf "epistemic-global-resource-conflict:%s"
                  selected.resource)
            state.attentions;
          Hashtbl.iter
            (fun _ decision ->
              if not decision.portfolio_completed
                 && List.mem selected.resource
                      decision.selected_portfolio.portfolio_selected_resources
              then
                failf "epistemic-global-resource-conflict:%s"
                  selected.resource)
            state.attention_portfolios)
        states;
      let fields =
        [ ("world", world); ("plan", plan);
          ("policy", attention_policy_name policy);
          ("budget", string_of_int budget);
          ("candidate_set", canonical_candidates);
          ("candidate_set_digest", candidate_set_digest);
          ("selected", selected.candidate_id);
          ("target_world", selected.target_world);
          ("claim", selected.target_claim); ("provider", selected.provider);
          ("resource", selected.resource);
          ("information", string_of_int selected.information_gain);
          ("falsification", string_of_int selected.falsification_power);
          ("divergence", string_of_int selected.counterfactual_divergence);
          ("cost", string_of_int selected.cost);
          ("risk", string_of_int selected.risk);
          ("evidence_digest", selected.candidate_evidence_digest);
          ("falsifier_digest", selected.candidate_falsifier_digest);
          ("owner", owner); ("generation", generation) ]
      in
      let event =
        append
          ~verify:(fun candidate_event ->
            ignore (attention_decision_of_event candidate_event))
          root world "ATTENTION_COMPILED" fields
      in
      ignore (load_all root);
      Printf.sprintf
        "LOOM_ATTENTION_COMPILED schema=loom-attention-compiler-v0 world=%s plan=%s policy=%s selected=%s target=%s/%s provider=%s cost=%d budget=%d resource_sha256=%s candidates=%d head=%s"
        world plan (attention_policy_name policy) selected.candidate_id
        selected.target_world selected.target_claim selected.provider selected.cost
        budget (sha256 selected.resource) (List.length candidates)
        event.event_sha256)

let complete_attention ~root ~world ~plan ~owner ~generation ~outcome =
  validate_atom "world" world;
  validate_atom "attention-plan" plan;
  validate_atom "attention-owner" owner;
  validate_atom "attention-generation" generation;
  let outcome = require_digest "attention-outcome" outcome in
  with_machine_lock root (fun () ->
      let state = find_world (load_all root) world in
      let decision =
        match Hashtbl.find_opt state.attentions plan with
        | Some value -> value
        | None -> failf "epistemic-attention-plan-missing:%s" plan
      in
      if decision.completed then
        failf "epistemic-attention-plan-already-completed:%s" plan;
      if owner <> decision.attention_owner
         || generation <> decision.attention_generation
      then failf "epistemic-attention-completion-identity-drift:%s" plan;
      let selected = decision.selected.candidate_id in
      let verify_completion event =
        let fields = decode_fields event.payload in
        verify_attention_completion ~plan:(field fields "plan")
          ~candidate:(field fields "selected")
          ~owner:(field fields "owner")
          ~generation:(field fields "generation")
          ~outcome:
            (require_digest "attention-outcome" (field fields "outcome_digest"))
      in
      let event =
        append ~verify:verify_completion root world "ATTENTION_COMPLETED"
          [ ("world", world); ("plan", plan); ("selected", selected);
            ("owner", owner); ("generation", generation);
            ("outcome_digest", outcome) ]
      in
      ignore (load_all root);
      Printf.sprintf
        "LOOM_ATTENTION_COMPLETED schema=loom-attention-compiler-v0 world=%s plan=%s selected=%s outcome=%s head=%s"
        world plan selected outcome event.event_sha256)

let compile_attention_portfolio ~root ~world ~portfolio ~candidate_file
    ~token_budget ~wall_budget ~gpu_budget ~quota_budget ~policy ~owner
    ~generation =
  validate_atom "world" world;
  validate_atom "attention-portfolio" portfolio;
  validate_atom "portfolio-owner" owner;
  validate_atom "portfolio-generation" generation;
  if not (Sys.file_exists candidate_file) then
    failf "epistemic-portfolio-candidate-file-missing:%s" candidate_file;
  let budget =
    portfolio_budget ~token_budget ~wall_budget ~gpu_budget ~quota_budget
  in
  let candidates, canonical_candidates =
    parse_portfolio_candidate_lines (read_lines candidate_file)
  in
  let feasible = enumerate_feasible_portfolios budget candidates in
  let frontier = pareto_frontier feasible in
  let canonical_frontier = canonical_portfolio_frontier frontier in
  let selected = select_portfolio policy frontier in
  let candidate_set_digest =
    portfolio_candidate_set_digest canonical_candidates
  in
  let frontier_digest = portfolio_frontier_digest canonical_frontier in
  let selected_set = portfolio_selected_set selected in
  let selected_set_digest = portfolio_selected_set_digest selected_set in
  with_machine_lock root (fun () ->
      let states = load_all root in
      let scheduling_world = find_world states world in
      if Hashtbl.mem scheduling_world.attention_portfolios portfolio then
        failf "epistemic-attention-portfolio-duplicate:%s" portfolio;
      List.iter
        (fun candidate ->
          let target = find_world states candidate.portfolio_target_world in
          if not (Hashtbl.mem target.claims candidate.portfolio_target_claim)
          then
            failf "epistemic-attention-target-claim-missing:%s/%s"
              candidate.portfolio_target_world candidate.portfolio_target_claim)
        candidates;
      List.iter
        (fun resource ->
          List.iter
            (fun state ->
              Hashtbl.iter
                (fun _ capability ->
                  if not capability.released
                     && capability.resource = resource
                  then failf "epistemic-global-resource-conflict:%s" resource)
                state.capabilities;
              Hashtbl.iter
                (fun _ attention ->
                  if not attention.completed
                     && attention.selected.resource = resource
                  then failf "epistemic-global-resource-conflict:%s" resource)
                state.attentions;
              Hashtbl.iter
                (fun _ existing ->
                  if not existing.portfolio_completed
                     && List.mem resource
                          existing.selected_portfolio.portfolio_selected_resources
                  then failf "epistemic-global-resource-conflict:%s" resource)
                state.attention_portfolios)
            states)
        selected.portfolio_selected_resources;
      let fields =
        [ ("world", world); ("portfolio", portfolio);
          ("policy", attention_policy_name policy);
          ("token_budget", string_of_int budget.portfolio_token_budget);
          ("wall_budget", string_of_int budget.portfolio_wall_budget);
          ("gpu_budget", string_of_int budget.portfolio_gpu_budget);
          ("quota_budget", string_of_int budget.portfolio_quota_budget);
          ("candidate_set", canonical_candidates);
          ("candidate_set_digest", candidate_set_digest);
          ("frontier", canonical_frontier);
          ("frontier_digest", frontier_digest);
          ("selected_set", selected_set);
          ("selected_set_digest", selected_set_digest);
          ("resources", String.concat "," selected.portfolio_selected_resources);
          ("information", string_of_int selected.aggregate_information);
          ("falsification", string_of_int selected.aggregate_falsification);
          ("divergence", string_of_int selected.aggregate_divergence);
          ("risk", string_of_int selected.aggregate_risk);
          ("token_cost", string_of_int selected.aggregate_token_cost);
          ("wall_cost", string_of_int selected.aggregate_wall_cost);
          ("gpu_cost", string_of_int selected.aggregate_gpu_cost);
          ("quota_cost", string_of_int selected.aggregate_quota_cost);
          ("evidence_digest", portfolio_evidence_digest selected);
          ("falsifier_digest", portfolio_falsifier_digest selected);
          ("owner", owner); ("generation", generation) ]
      in
      let event =
        append
          ~verify:(fun candidate_event ->
            ignore (portfolio_attention_decision_of_event candidate_event))
          root world "ATTENTION_PORTFOLIO_COMPILED" fields
      in
      ignore (load_all root);
      Printf.sprintf
        "LOOM_PORTFOLIO_COMPILED schema=loom-pareto-portfolio-v0 world=%s portfolio=%s policy=%s selected=%s selected_count=%d resources=%d candidates=%d enumerated=%d feasible=%d frontier=%d token=%d/%d wall=%d/%d gpu=%d/%d quota=%d/%d selected_set_sha256=%s frontier_sha256=%s head=%s"
        world portfolio (attention_policy_name policy) selected_set
        (List.length selected.portfolio_candidates)
        (List.length selected.portfolio_selected_resources)
        (List.length candidates) ((1 lsl List.length candidates) - 1)
        (List.length feasible) (List.length frontier)
        selected.aggregate_token_cost budget.portfolio_token_budget
        selected.aggregate_wall_cost budget.portfolio_wall_budget
        selected.aggregate_gpu_cost budget.portfolio_gpu_budget
        selected.aggregate_quota_cost budget.portfolio_quota_budget
        selected_set_digest frontier_digest event.event_sha256)

let complete_attention_portfolio ~root ~world ~portfolio ~owner ~generation
    ~outcome =
  validate_atom "world" world;
  validate_atom "attention-portfolio" portfolio;
  validate_atom "portfolio-owner" owner;
  validate_atom "portfolio-generation" generation;
  let outcome = require_digest "portfolio-outcome" outcome in
  with_machine_lock root (fun () ->
      let state = find_world (load_all root) world in
      let decision =
        match Hashtbl.find_opt state.attention_portfolios portfolio with
        | Some value -> value
        | None -> failf "epistemic-attention-portfolio-missing:%s" portfolio
      in
      if decision.portfolio_completed then
        failf "epistemic-attention-portfolio-already-completed:%s" portfolio;
      if owner <> decision.portfolio_owner
         || generation <> decision.portfolio_generation
      then failf "epistemic-portfolio-completion-identity-drift:%s" portfolio;
      let selected_set = portfolio_selected_set decision.selected_portfolio in
      let verify_completion event =
        let fields = decode_fields event.payload in
        verify_portfolio_completion ~portfolio_id:(field fields "portfolio")
          ~selected_set:(field fields "selected_set")
          ~owner:(field fields "owner")
          ~generation:(field fields "generation")
          ~outcome:
            (require_digest "portfolio-outcome" (field fields "outcome_digest"))
      in
      let event =
        append ~verify:verify_completion root world
          "ATTENTION_PORTFOLIO_COMPLETED"
          [ ("world", world); ("portfolio", portfolio);
            ("selected_set", selected_set); ("owner", owner);
            ("generation", generation); ("outcome_digest", outcome) ]
      in
      ignore (load_all root);
      Printf.sprintf
        "LOOM_PORTFOLIO_COMPLETED schema=loom-pareto-portfolio-v0 world=%s portfolio=%s selected=%s released_resources=%d outcome=%s head=%s"
        world portfolio selected_set
        (List.length decision.selected_portfolio.portfolio_selected_resources)
        outcome event.event_sha256)

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
      let live_attention =
        Hashtbl.fold
          (fun _ decision count -> if decision.completed then count else count + 1)
          state.attentions 0
      in
      let live_portfolios =
        Hashtbl.fold
          (fun _ decision count ->
            if decision.portfolio_completed then count else count + 1)
          state.attention_portfolios 0
      in
      Printf.sprintf
        "LOOM_WORLD_OK schema=%s world=%s events=%d knowledge=%d claims=%d challenged=%d live_capabilities=%d attention_plans=%d live_attention=%d attention_portfolios=%d live_portfolios=%d parent=%s head=%s"
        schema world (List.length state.events) (Hashtbl.length state.knowledges)
        (Hashtbl.length state.claims) challenged live_capabilities
        (Hashtbl.length state.attentions) live_attention
        (Hashtbl.length state.attention_portfolios) live_portfolios
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
