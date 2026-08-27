open Unix

exception Error of string

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let zero_digest = Loom_witness.zero_digest
let sha256 = Loom_epistemic.sha256
let handoff_schema = "loom-witness-epoch-handoff-v0"
let active_schema = "loom-witness-active-epoch-v0"
let max_handoff_bytes = 512 * 1024
let max_chain_depth = 64
let max_transition_from_epoch = 64

let failpoint point =
  match Sys.getenv_opt "SOUNIO_LOOM_WITNESS_EPOCH_FAILPOINT" with
  | Some configured when configured = point ->
      Printf.eprintf
        "LOOM_WITNESS_EPOCH_FAILPOINT name=%s exit=198\n%!" point;
      exit 198
  | _ -> ()

let path_value label value =
  if value = "" || String.contains value '\n' || String.contains value '\r' then
    failf "%s-invalid" label;
  value

let canonical_path label path =
  try Unix.realpath path
  with Unix.Unix_error (error, _, _) ->
    failf "%s-unavailable:%s" label (Unix.error_message error)

let encoded_path path = Loom_witness.hex_of_string path

let decoded_path label value =
  try Loom_witness.string_of_hex value |> path_value label
  with _ -> failf "%s-invalid-hex" label

let root_digest path =
  sha256 ("loom-witness-epoch-state-root-v0\000" ^ path)

let membership_file_digest path =
  let contents = Loom_witness.read_file_bounded "epoch-membership-file" (64 * 1024) path in
  sha256 ("loom-witness-epoch-membership-file-v0\000" ^ contents)

let handoff_digest canonical =
  sha256 (handoff_schema ^ "\000" ^ canonical)

let epoch_directory epoch_state_dir world =
  Filename.concat (Filename.concat epoch_state_dir "loom-witness-epochs") world

let handoff_directory epoch_state_dir world =
  Filename.concat (epoch_directory epoch_state_dir world) "handoffs"

let active_path epoch_state_dir world =
  Filename.concat (epoch_directory epoch_state_dir world) "active-epoch.receipt"

let handoff_path epoch_state_dir world from_epoch to_epoch =
  Filename.concat (handoff_directory epoch_state_dir world)
    (Printf.sprintf "handoff-%020d-%020d.receipt" from_epoch to_epoch)

let with_epoch_lock epoch_state_dir world operation =
  let directory = epoch_directory epoch_state_dir world in
  Loom_epistemic.mkdir_p directory;
  let path = Filename.concat directory "epoch.lock" in
  let descriptor = Unix.openfile path [ O_RDWR; O_CREAT ] 0o600 in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
      Unix.lockf descriptor F_LOCK 0;
      Fun.protect ~finally:(fun () -> Unix.lockf descriptor F_ULOCK 0) operation)

type side = {
  state_root : string;
  state_root_digest : string;
  membership_file : string;
  membership_file_digest : string;
  membership : Loom_witness.membership;
  events : Loom_epistemic.event array;
  head_at : int -> string;
  certificate_canonical : string;
  certificate : Loom_witness.certificate;
  certificate_digest : string;
  status_shares : string option array;
  status_receipts : Loom_witness.receipt option array;
  current_quorum : int;
}

let load_current_side ~root ~world ~membership_file ~endpoints_file =
  let state_root = canonical_path "epoch-state-root" root in
  let membership_file = canonical_path "epoch-membership-file" membership_file in
  let membership = Loom_witness.load_membership membership_file in
  (match membership.topology with
  | Loom_witness.Mesh_v1 -> ()
  | Loom_witness.Mesh_v0 -> failf "witness-epoch-handoff-requires-membership-v1");
  let endpoints = Loom_witness.load_endpoints membership endpoints_file in
  let events, current_head, head_at = Loom_witness.events_and_heads state_root world in
  let certificate_canonical, certificate =
    match
      Loom_witness.load_certificate_chain state_root world membership events head_at
    with
    | None -> failf "witness-epoch-certificate-missing:%s" world
    | Some value -> value
  in
  if certificate.certificate_event_count <> Array.length events
     || certificate.certificate_journal_head <> current_head
  then failf "witness-epoch-unanchored-journal:%s" world;
  let statuses =
    Array.map
      (fun endpoint -> Loom_witness.query_status membership endpoint world)
      endpoints
  in
  Array.iteri
    (fun index status ->
      Loom_witness.validate_remote_prefix membership.members.(index) status
        certificate.certificate_sequence events head_at)
    statuses;
  let status_shares =
    Array.map
      (function
        | Loom_witness.Latest (canonical, receipt)
          when receipt.receipt_anchor_sequence = certificate.certificate_sequence
               && receipt.receipt_end_count = certificate.certificate_event_count
               && receipt.receipt_end_head = certificate.certificate_journal_head ->
            Some canonical
        | _ -> None)
      statuses
  in
  let status_receipts =
    Array.mapi
      (fun index -> function
        | None -> None
        | Some canonical ->
            Some
              (Loom_witness.verify_receipt membership membership.members.(index)
                 canonical))
      status_shares
  in
  let current_quorum =
    Array.fold_left
      (fun count -> function None -> count | Some _ -> count + 1)
      0 status_receipts
  in
  if current_quorum < 3 then
    failf "witness-epoch-current-quorum-unavailable:valid=%d:required=3"
      current_quorum;
  Loom_witness.verify_native_frame membership certificate status_receipts;
  { state_root;
    state_root_digest = root_digest state_root;
    membership_file;
    membership_file_digest = membership_file_digest membership_file;
    membership;
    events;
    head_at;
    certificate_canonical;
    certificate;
    certificate_digest =
      Loom_witness.certificate_digest membership.topology certificate_canonical;
    status_shares;
    status_receipts;
    current_quorum }

type handoff = {
  world : string;
  from_epoch : int;
  to_epoch : int;
  old_state_root : string;
  new_state_root : string;
  old_state_root_digest : string;
  new_state_root_digest : string;
  old_membership_file : string;
  new_membership_file : string;
  old_membership_file_digest : string;
  new_membership_file_digest : string;
  old_membership_digest : string;
  new_membership_digest : string;
  old_anchor_sequence : int;
  new_anchor_sequence : int;
  event_count : int;
  journal_head : string;
  old_certificate_digest : string;
  new_certificate_digest : string;
  old_certificate : string;
  new_certificate : string;
  old_status_shares : string option array;
  new_status_shares : string option array;
  previous_handoff_digest : string;
  previous_handoff_path : string option;
}

let share_field = function
  | None -> "-"
  | Some canonical -> Loom_witness.hex_of_string canonical

let canonical_handoff handoff =
  let buffer = Buffer.create 32768 in
  let add format = Printf.ksprintf (Buffer.add_string buffer) format in
  add "schema=%s\n" handoff_schema;
  add "world=%s\n" handoff.world;
  add "from_epoch=%d\n" handoff.from_epoch;
  add "to_epoch=%d\n" handoff.to_epoch;
  add "old_state_root_hex=%s\n" (encoded_path handoff.old_state_root);
  add "new_state_root_hex=%s\n" (encoded_path handoff.new_state_root);
  add "old_state_root_sha256=%s\n" handoff.old_state_root_digest;
  add "new_state_root_sha256=%s\n" handoff.new_state_root_digest;
  add "old_membership_file_hex=%s\n" (encoded_path handoff.old_membership_file);
  add "new_membership_file_hex=%s\n" (encoded_path handoff.new_membership_file);
  add "old_membership_file_sha256=%s\n" handoff.old_membership_file_digest;
  add "new_membership_file_sha256=%s\n" handoff.new_membership_file_digest;
  add "old_membership_sha256=%s\n" handoff.old_membership_digest;
  add "new_membership_sha256=%s\n" handoff.new_membership_digest;
  add "old_quorum=3\nnew_quorum=3\nold_member_count=4\nnew_member_count=4\n";
  add "old_anchor_sequence=%d\n" handoff.old_anchor_sequence;
  add "new_anchor_sequence=%d\n" handoff.new_anchor_sequence;
  add "event_count=%d\n" handoff.event_count;
  add "journal_head_sha256=%s\n" handoff.journal_head;
  add "old_certificate_sha256=%s\n" handoff.old_certificate_digest;
  add "new_certificate_sha256=%s\n" handoff.new_certificate_digest;
  add "old_certificate_hex=%s\n" (Loom_witness.hex_of_string handoff.old_certificate);
  add "new_certificate_hex=%s\n" (Loom_witness.hex_of_string handoff.new_certificate);
  Array.iteri
    (fun index share -> add "old_status%d=%s\n" (index + 1) (share_field share))
    handoff.old_status_shares;
  Array.iteri
    (fun index share -> add "new_status%d=%s\n" (index + 1) (share_field share))
    handoff.new_status_shares;
  add "previous_handoff_sha256=%s\n" handoff.previous_handoff_digest;
  add "previous_handoff_path_hex=%s\n"
    (match handoff.previous_handoff_path with
    | None -> "-"
    | Some path -> encoded_path path);
  add "native_frame=9015\n";
  Buffer.contents buffer

let parse_optional_path label value =
  if value = "-" then None else Some (decoded_path label value)

let parse_handoff canonical =
  if String.length canonical > max_handoff_bytes then
    failf "witness-epoch-handoff-too-large:%d" (String.length canonical);
  let fields = Loom_witness.fields_of_text "witness-epoch-handoff" canonical in
  if Loom_witness.field "witness-epoch-handoff" fields "schema" <> handoff_schema
     || Loom_witness.field "witness-epoch-handoff" fields "old_quorum" <> "3"
     || Loom_witness.field "witness-epoch-handoff" fields "new_quorum" <> "3"
     || Loom_witness.field "witness-epoch-handoff" fields "old_member_count" <> "4"
     || Loom_witness.field "witness-epoch-handoff" fields "new_member_count" <> "4"
     || Loom_witness.field "witness-epoch-handoff" fields "native_frame" <> "9015"
  then failf "witness-epoch-handoff-schema-invalid";
  let status prefix =
    Array.init 4 (fun index ->
        Loom_witness.parse_share
          (Printf.sprintf "witness-epoch-%s-status%d" prefix (index + 1))
          (Loom_witness.field "witness-epoch-handoff" fields
             (Printf.sprintf "%s_status%d" prefix (index + 1))))
  in
  let handoff =
    { world = Loom_witness.field "witness-epoch-handoff" fields "world";
      from_epoch =
        Loom_witness.positive_integer "witness-epoch-from"
          (Loom_witness.field "witness-epoch-handoff" fields "from_epoch");
      to_epoch =
        Loom_witness.positive_integer "witness-epoch-to"
          (Loom_witness.field "witness-epoch-handoff" fields "to_epoch");
      old_state_root =
        decoded_path "witness-epoch-old-root"
          (Loom_witness.field "witness-epoch-handoff" fields "old_state_root_hex");
      new_state_root =
        decoded_path "witness-epoch-new-root"
          (Loom_witness.field "witness-epoch-handoff" fields "new_state_root_hex");
      old_state_root_digest =
        Loom_witness.nonzero_digest "witness-epoch-old-root"
          (Loom_witness.field "witness-epoch-handoff" fields "old_state_root_sha256");
      new_state_root_digest =
        Loom_witness.nonzero_digest "witness-epoch-new-root"
          (Loom_witness.field "witness-epoch-handoff" fields "new_state_root_sha256");
      old_membership_file =
        decoded_path "witness-epoch-old-membership-file"
          (Loom_witness.field "witness-epoch-handoff" fields "old_membership_file_hex");
      new_membership_file =
        decoded_path "witness-epoch-new-membership-file"
          (Loom_witness.field "witness-epoch-handoff" fields "new_membership_file_hex");
      old_membership_file_digest =
        Loom_witness.nonzero_digest "witness-epoch-old-membership-file"
          (Loom_witness.field "witness-epoch-handoff" fields "old_membership_file_sha256");
      new_membership_file_digest =
        Loom_witness.nonzero_digest "witness-epoch-new-membership-file"
          (Loom_witness.field "witness-epoch-handoff" fields "new_membership_file_sha256");
      old_membership_digest =
        Loom_witness.nonzero_digest "witness-epoch-old-membership"
          (Loom_witness.field "witness-epoch-handoff" fields "old_membership_sha256");
      new_membership_digest =
        Loom_witness.nonzero_digest "witness-epoch-new-membership"
          (Loom_witness.field "witness-epoch-handoff" fields "new_membership_sha256");
      old_anchor_sequence =
        Loom_witness.positive_integer "witness-epoch-old-sequence"
          (Loom_witness.field "witness-epoch-handoff" fields "old_anchor_sequence");
      new_anchor_sequence =
        Loom_witness.positive_integer "witness-epoch-new-sequence"
          (Loom_witness.field "witness-epoch-handoff" fields "new_anchor_sequence");
      event_count =
        Loom_witness.positive_integer "witness-epoch-count"
          (Loom_witness.field "witness-epoch-handoff" fields "event_count");
      journal_head =
        Loom_witness.nonzero_digest "witness-epoch-head"
          (Loom_witness.field "witness-epoch-handoff" fields "journal_head_sha256");
      old_certificate_digest =
        Loom_witness.nonzero_digest "witness-epoch-old-certificate"
          (Loom_witness.field "witness-epoch-handoff" fields "old_certificate_sha256");
      new_certificate_digest =
        Loom_witness.nonzero_digest "witness-epoch-new-certificate"
          (Loom_witness.field "witness-epoch-handoff" fields "new_certificate_sha256");
      old_certificate =
        (try
           Loom_witness.string_of_hex
             (Loom_witness.field "witness-epoch-handoff" fields "old_certificate_hex")
         with _ -> failf "witness-epoch-old-certificate-invalid-hex");
      new_certificate =
        (try
           Loom_witness.string_of_hex
             (Loom_witness.field "witness-epoch-handoff" fields "new_certificate_hex")
         with _ -> failf "witness-epoch-new-certificate-invalid-hex");
      old_status_shares = status "old";
      new_status_shares = status "new";
      previous_handoff_digest =
        Loom_witness.digest "witness-epoch-previous"
          (Loom_witness.field "witness-epoch-handoff" fields "previous_handoff_sha256");
      previous_handoff_path =
        parse_optional_path "witness-epoch-previous-path"
          (Loom_witness.field "witness-epoch-handoff" fields
             "previous_handoff_path_hex") }
  in
  Loom_witness.validate_atom "witness-epoch-world" handoff.world;
  if canonical_handoff handoff <> canonical then
    failf "witness-epoch-handoff-noncanonical";
  handoff

let adapter_path () =
  match Sys.getenv_opt "SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_ADAPTER" with
  | Some path when path <> "" -> path
  | _ ->
      Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
        "sounio-loom-witness-epoch-handoff-runtime"

let verify_native_handoff handoff =
  let domain = Loom_epistemic.token "loom-witness-domain" handoff.world in
  let frame =
    [ "9015"; "1"; "1";
      string_of_int handoff.from_epoch; string_of_int handoff.to_epoch;
      "3"; "3"; "4"; "4"; domain; domain;
      string_of_int handoff.old_anchor_sequence;
      string_of_int handoff.new_anchor_sequence;
      string_of_int handoff.event_count; string_of_int handoff.event_count ]
    @ Loom_epistemic.digest_limbs handoff.old_membership_digest
    @ Loom_epistemic.digest_limbs handoff.new_membership_digest
    @ Loom_epistemic.digest_limbs handoff.old_state_root_digest
    @ Loom_epistemic.digest_limbs handoff.new_state_root_digest
    @ Loom_epistemic.digest_limbs handoff.journal_head
    @ Loom_epistemic.digest_limbs handoff.journal_head
    @ Loom_epistemic.digest_limbs handoff.old_certificate_digest
    @ Loom_epistemic.digest_limbs handoff.new_certificate_digest
    @ Loom_epistemic.digest_limbs handoff.previous_handoff_digest
  in
  let adapter = adapter_path () in
  if not (Sys.file_exists adapter) then
    failf "witness-epoch-native-adapter-missing:%s" adapter;
  let code, output =
    Loom_epistemic.process_exchange (Unix.realpath adapter)
      (String.concat " " frame ^ "\n")
  in
  let expected =
    "SOUNIO_WITNESS_EPOCH_HANDOFF_ACCEPT schema=loom-native-witness-epoch-handoff-v0 transition=joint-quorum state=prepared"
  in
  if code <> 0 || output <> expected then
    failf "witness-epoch-native-refused:rc=%d:output=%s" code output

let find_certificate root world expected =
  List.exists
    (fun path ->
      Loom_witness.read_file_bounded "witness-epoch-certificate" (64 * 1024) path
      = expected)
    (Loom_witness.certificate_files root world)

let validate_retained_side handoff ~old =
  let state_root = if old then handoff.old_state_root else handoff.new_state_root in
  let expected_root_digest =
    if old then handoff.old_state_root_digest else handoff.new_state_root_digest
  in
  let membership_file =
    if old then handoff.old_membership_file else handoff.new_membership_file
  in
  let expected_membership_file_digest =
    if old then handoff.old_membership_file_digest
    else handoff.new_membership_file_digest
  in
  let expected_membership_digest =
    if old then handoff.old_membership_digest else handoff.new_membership_digest
  in
  let certificate_canonical =
    if old then handoff.old_certificate else handoff.new_certificate
  in
  let expected_certificate_digest =
    if old then handoff.old_certificate_digest else handoff.new_certificate_digest
  in
  let status_shares =
    if old then handoff.old_status_shares else handoff.new_status_shares
  in
  let expected_sequence =
    if old then handoff.old_anchor_sequence else handoff.new_anchor_sequence
  in
  let state_root = canonical_path "witness-epoch-retained-root" state_root in
  let membership_file =
    canonical_path "witness-epoch-retained-membership" membership_file
  in
  if root_digest state_root <> expected_root_digest then
    failf "witness-epoch-state-root-digest-mismatch";
  if membership_file_digest membership_file <> expected_membership_file_digest then
    failf "witness-epoch-membership-file-digest-mismatch";
  let membership = Loom_witness.load_membership membership_file in
  (match membership.topology with
  | Loom_witness.Mesh_v1 -> ()
  | Loom_witness.Mesh_v0 -> failf "witness-epoch-retained-membership-not-v1");
  if membership.membership_digest <> expected_membership_digest then
    failf "witness-epoch-membership-digest-mismatch";
  let events, _, head_at = Loom_witness.events_and_heads state_root handoff.world in
  ignore
    (Loom_witness.load_certificate_chain state_root handoff.world membership
       events head_at);
  if not (find_certificate state_root handoff.world certificate_canonical) then
    failf "witness-epoch-boundary-certificate-missing";
  let certificate =
    Loom_witness.parse_certificate membership certificate_canonical
  in
  if certificate.certificate_world <> handoff.world
     || certificate.certificate_membership_digest <> expected_membership_digest
     || certificate.certificate_sequence <> expected_sequence
     || certificate.certificate_event_count <> handoff.event_count
     || certificate.certificate_journal_head <> handoff.journal_head
     || Loom_witness.certificate_digest membership.topology certificate_canonical
        <> expected_certificate_digest
     || handoff.event_count > Array.length events
     || head_at handoff.event_count <> handoff.journal_head
  then failf "witness-epoch-boundary-certificate-mismatch";
  let certificate_receipts =
    Loom_witness.receipts_for_certificate membership certificate
  in
  Loom_witness.verify_native_frame membership certificate certificate_receipts;
  let status_receipts =
    Array.mapi
      (fun index -> function
        | None -> None
        | Some canonical ->
            let receipt =
              Loom_witness.verify_receipt membership membership.members.(index)
                canonical
            in
            if receipt.receipt_domain <> handoff.world
               || receipt.receipt_anchor_sequence <> expected_sequence
               || receipt.receipt_end_count <> handoff.event_count
               || receipt.receipt_end_head <> handoff.journal_head
            then failf "witness-epoch-status-share-mismatch:%d" (index + 1);
            Some receipt)
      status_shares
  in
  let count =
    Array.fold_left
      (fun total -> function None -> total | Some _ -> total + 1)
      0 status_receipts
  in
  if count < 3 then failf "witness-epoch-retained-quorum-missing:%d" count;
  Loom_witness.verify_native_frame membership certificate status_receipts;
  (membership, events, head_at, certificate)

let rec verify_handoff_chain path depth =
  if depth > max_chain_depth then failf "witness-epoch-chain-too-deep";
  let canonical_path_value = canonical_path "witness-epoch-handoff" path in
  let canonical =
    Loom_witness.read_file_bounded "witness-epoch-handoff" max_handoff_bytes
      canonical_path_value
  in
  let handoff = parse_handoff canonical in
  let _, _, old_head_at, _ = validate_retained_side handoff ~old:true in
  let _ = validate_retained_side handoff ~old:false in
  if handoff.from_epoch < 1
     || handoff.from_epoch > max_transition_from_epoch
     || handoff.to_epoch <> handoff.from_epoch + 1
     || handoff.old_membership_digest = handoff.new_membership_digest
     || handoff.old_state_root_digest = handoff.new_state_root_digest
     || handoff.old_certificate_digest = handoff.new_certificate_digest
  then failf "witness-epoch-handoff-boundary-invalid";
  verify_native_handoff handoff;
  (match (handoff.from_epoch, handoff.previous_handoff_path) with
  | 1, None ->
      if handoff.previous_handoff_digest <> zero_digest then
        failf "witness-epoch-genesis-predecessor-nonzero"
  | 1, Some _ -> failf "witness-epoch-genesis-predecessor-present"
  | _, None -> failf "witness-epoch-predecessor-missing"
  | _, Some previous_path ->
      let previous, previous_digest, _ =
        verify_handoff_chain previous_path (depth + 1)
      in
      if previous_digest <> handoff.previous_handoff_digest
         || previous.to_epoch <> handoff.from_epoch
         || previous.new_membership_digest <> handoff.old_membership_digest
         || previous.new_state_root_digest <> handoff.old_state_root_digest
         || previous.event_count > handoff.event_count
         || old_head_at previous.event_count <> previous.journal_head
      then failf "witness-epoch-predecessor-chain-mismatch");
  (handoff, handoff_digest canonical, canonical_path_value)

type active_epoch = {
  active_world : string;
  active_epoch : int;
  active_membership_digest : string;
  active_state_root : string;
  active_state_root_digest : string;
  active_event_count : int;
  active_journal_head : string;
  active_handoff_digest : string;
  active_handoff_path : string;
}

let canonical_active active =
  Printf.sprintf
    "schema=%s\nworld=%s\nepoch=%d\nmembership_sha256=%s\nstate_root_hex=%s\nstate_root_sha256=%s\nevent_count=%d\njournal_head_sha256=%s\nhandoff_sha256=%s\nhandoff_path_hex=%s\n"
    active_schema active.active_world active.active_epoch
    active.active_membership_digest (encoded_path active.active_state_root)
    active.active_state_root_digest active.active_event_count
    active.active_journal_head active.active_handoff_digest
    (encoded_path active.active_handoff_path)

let parse_active canonical =
  let fields = Loom_witness.fields_of_text "witness-active-epoch" canonical in
  if Loom_witness.field "witness-active-epoch" fields "schema" <> active_schema then
    failf "witness-active-epoch-schema-invalid";
  let active =
    { active_world = Loom_witness.field "witness-active-epoch" fields "world";
      active_epoch =
        Loom_witness.positive_integer "witness-active-epoch"
          (Loom_witness.field "witness-active-epoch" fields "epoch");
      active_membership_digest =
        Loom_witness.nonzero_digest "witness-active-membership"
          (Loom_witness.field "witness-active-epoch" fields "membership_sha256");
      active_state_root =
        decoded_path "witness-active-root"
          (Loom_witness.field "witness-active-epoch" fields "state_root_hex");
      active_state_root_digest =
        Loom_witness.nonzero_digest "witness-active-root"
          (Loom_witness.field "witness-active-epoch" fields "state_root_sha256");
      active_event_count =
        Loom_witness.positive_integer "witness-active-count"
          (Loom_witness.field "witness-active-epoch" fields "event_count");
      active_journal_head =
        Loom_witness.nonzero_digest "witness-active-head"
          (Loom_witness.field "witness-active-epoch" fields "journal_head_sha256");
      active_handoff_digest =
        Loom_witness.nonzero_digest "witness-active-handoff"
          (Loom_witness.field "witness-active-epoch" fields "handoff_sha256");
      active_handoff_path =
        decoded_path "witness-active-handoff-path"
          (Loom_witness.field "witness-active-epoch" fields "handoff_path_hex") }
  in
  Loom_witness.validate_atom "witness-active-world" active.active_world;
  if canonical_active active <> canonical then failf "witness-active-epoch-noncanonical";
  active

let load_active epoch_state_dir world =
  let path = active_path epoch_state_dir world in
  if not (Sys.file_exists path) then None
  else
    let canonical = Loom_witness.read_file_bounded "witness-active-epoch" (16 * 1024) path in
    let active = parse_active canonical in
    if active.active_world <> world then failf "witness-active-world-mismatch";
    let handoff, digest, handoff_path =
      verify_handoff_chain active.active_handoff_path 0
    in
    if digest <> active.active_handoff_digest
       || handoff_path <> canonical_path "witness-active-handoff" active.active_handoff_path
       || handoff.to_epoch <> active.active_epoch
       || handoff.new_membership_digest <> active.active_membership_digest
       || handoff.new_state_root_digest <> active.active_state_root_digest
       || handoff.event_count <> active.active_event_count
       || handoff.journal_head <> active.active_journal_head
    then failf "witness-active-epoch-handoff-mismatch";
    Some (active, handoff)

let handoff_matches_request handoff old_side new_side from_epoch to_epoch =
  handoff.from_epoch = from_epoch && handoff.to_epoch = to_epoch
  && handoff.old_membership_digest = old_side.membership.membership_digest
  && handoff.new_membership_digest = new_side.membership.membership_digest
  && handoff.old_state_root_digest = old_side.state_root_digest
  && handoff.new_state_root_digest = new_side.state_root_digest

let success_output handoff digest path ~prepared ~idempotent =
  Printf.sprintf
    "LOOM_WITNESS_EPOCH_HANDOFF_OK schema=%s world=%s from_epoch=%d to_epoch=%d joint_quorum=3/4+3/4 event_count=%d journal_head=%s old_membership_sha256=%s new_membership_sha256=%s previous_handoff_sha256=%s handoff_sha256=%s handoff=%s prepared=%s activated=yes idempotent=%s native_frame=9015"
    handoff_schema handoff.world handoff.from_epoch handoff.to_epoch
    handoff.event_count handoff.journal_head handoff.old_membership_digest
    handoff.new_membership_digest handoff.previous_handoff_digest digest path
    (if prepared then "yes" else "reused") (if idempotent then "yes" else "no")

let activate epoch_state_dir handoff digest path =
  let active =
    { active_world = handoff.world;
      active_epoch = handoff.to_epoch;
      active_membership_digest = handoff.new_membership_digest;
      active_state_root = handoff.new_state_root;
      active_state_root_digest = handoff.new_state_root_digest;
      active_event_count = handoff.event_count;
      active_journal_head = handoff.journal_head;
      active_handoff_digest = digest;
      active_handoff_path = path }
  in
  Loom_witness.atomic_write (active_path epoch_state_dir handoff.world)
    (canonical_active active)

let handoff ~epoch_state_dir ~world ~from_epoch ~to_epoch
    ~old_root ~old_membership_file ~old_endpoints_file
    ~new_root ~new_membership_file ~new_endpoints_file =
  Loom_witness.validate_atom "witness-epoch-world" world;
  if from_epoch < 1 || from_epoch > max_transition_from_epoch then
    failf "witness-epoch-out-of-range:%d:max=%d" from_epoch
      max_transition_from_epoch;
  if to_epoch <> from_epoch + 1 then
    failf "witness-epoch-not-adjacent:%d:%d" from_epoch to_epoch;
  let old_root_identity = canonical_path "epoch-old-state-root" old_root in
  let new_root_identity = canonical_path "epoch-new-state-root" new_root in
  if root_digest old_root_identity = root_digest new_root_identity then
    failf "witness-epoch-state-root-reuse";
  let old_membership_identity =
    Loom_witness.load_membership
      (canonical_path "epoch-old-membership-file" old_membership_file)
  in
  let new_membership_identity =
    Loom_witness.load_membership
      (canonical_path "epoch-new-membership-file" new_membership_file)
  in
  if old_membership_identity.membership_digest
       = new_membership_identity.membership_digest
  then failf "witness-epoch-membership-reuse";
  with_epoch_lock epoch_state_dir world (fun () ->
      let old_side =
        load_current_side ~root:old_root ~world ~membership_file:old_membership_file
          ~endpoints_file:old_endpoints_file
      in
      let new_side =
        load_current_side ~root:new_root ~world ~membership_file:new_membership_file
          ~endpoints_file:new_endpoints_file
      in
      if old_side.state_root_digest = new_side.state_root_digest then
        failf "witness-epoch-state-root-reuse";
      if old_side.membership.membership_digest = new_side.membership.membership_digest then
        failf "witness-epoch-membership-reuse";
      if old_side.certificate.certificate_event_count
           <> new_side.certificate.certificate_event_count
         || old_side.certificate.certificate_journal_head
            <> new_side.certificate.certificate_journal_head
      then failf "witness-epoch-checkpoint-drift";
      match load_active epoch_state_dir world with
      | Some (active, existing)
        when active.active_epoch = to_epoch ->
          if not
               (handoff_matches_request existing old_side new_side from_epoch to_epoch)
          then failf "witness-epoch-active-transition-conflict";
          success_output existing active.active_handoff_digest
            active.active_handoff_path ~prepared:false ~idempotent:true
      | active ->
          let previous_handoff_digest, previous_handoff_path =
            match active with
            | None when from_epoch = 1 -> (zero_digest, None)
            | None -> failf "witness-epoch-active-predecessor-missing:%d" from_epoch
            | Some (current, previous) ->
                if current.active_epoch <> from_epoch
                   || current.active_membership_digest
                      <> old_side.membership.membership_digest
                   || current.active_state_root_digest <> old_side.state_root_digest
                   || current.active_event_count
                      > old_side.certificate.certificate_event_count
                   || old_side.head_at current.active_event_count
                      <> current.active_journal_head
                   || previous.to_epoch <> from_epoch
                then failf "witness-epoch-active-predecessor-mismatch";
                (current.active_handoff_digest, Some current.active_handoff_path)
          in
          let path = handoff_path epoch_state_dir world from_epoch to_epoch in
          let prepared, handoff, digest, path =
            if Sys.file_exists path then (
              let existing, digest, path = verify_handoff_chain path 0 in
              if not
                   (handoff_matches_request existing old_side new_side from_epoch to_epoch)
                 || existing.previous_handoff_digest <> previous_handoff_digest
              then failf "witness-epoch-prepared-transition-conflict";
              (false, existing, digest, path))
            else
              let handoff =
                { world; from_epoch; to_epoch;
                  old_state_root = old_side.state_root;
                  new_state_root = new_side.state_root;
                  old_state_root_digest = old_side.state_root_digest;
                  new_state_root_digest = new_side.state_root_digest;
                  old_membership_file = old_side.membership_file;
                  new_membership_file = new_side.membership_file;
                  old_membership_file_digest = old_side.membership_file_digest;
                  new_membership_file_digest = new_side.membership_file_digest;
                  old_membership_digest = old_side.membership.membership_digest;
                  new_membership_digest = new_side.membership.membership_digest;
                  old_anchor_sequence = old_side.certificate.certificate_sequence;
                  new_anchor_sequence = new_side.certificate.certificate_sequence;
                  event_count = old_side.certificate.certificate_event_count;
                  journal_head = old_side.certificate.certificate_journal_head;
                  old_certificate_digest = old_side.certificate_digest;
                  new_certificate_digest = new_side.certificate_digest;
                  old_certificate = old_side.certificate_canonical;
                  new_certificate = new_side.certificate_canonical;
                  old_status_shares = old_side.status_shares;
                  new_status_shares = new_side.status_shares;
                  previous_handoff_digest;
                  previous_handoff_path }
              in
              verify_native_handoff handoff;
              let canonical = canonical_handoff handoff in
              let digest = handoff_digest canonical in
              Loom_witness.atomic_write path canonical;
              let verified, verified_digest, verified_path =
                verify_handoff_chain path 0
              in
              if verified_digest <> digest then
                failf "witness-epoch-written-handoff-digest-mismatch";
              (true, verified, verified_digest, verified_path)
          in
          failpoint "after-handoff-before-activation";
          activate epoch_state_dir handoff digest path;
          failpoint "after-activation-before-return";
          success_output handoff digest path ~prepared ~idempotent:false)

let verify_active ~epoch_state_dir ~world ~active_root ~membership_file
    ~endpoints_file =
  Loom_witness.validate_atom "witness-epoch-world" world;
  with_epoch_lock epoch_state_dir world (fun () ->
      let active, handoff =
        match load_active epoch_state_dir world with
        | None -> failf "witness-active-epoch-missing:%s" world
        | Some value -> value
      in
      let side =
        load_current_side ~root:active_root ~world ~membership_file
          ~endpoints_file
      in
      if side.state_root_digest <> active.active_state_root_digest
         || side.membership.membership_digest <> active.active_membership_digest
         || active.active_event_count > Array.length side.events
         || side.head_at active.active_event_count <> active.active_journal_head
         || handoff.to_epoch <> active.active_epoch
      then failf "witness-active-epoch-current-state-mismatch";
      Printf.sprintf
        "LOOM_WITNESS_EPOCH_ACTIVE_OK schema=%s world=%s epoch=%d current_event_count=%d activation_event_count=%d current_journal_head=%s membership_sha256=%s state_root_sha256=%s handoff_sha256=%s remote_quorum=%d/4 chain=VERIFIED native_frames=9014+9015"
        active_schema world active.active_epoch
        side.certificate.certificate_event_count active.active_event_count
        side.certificate.certificate_journal_head active.active_membership_digest
        active.active_state_root_digest active.active_handoff_digest
        side.current_quorum)
