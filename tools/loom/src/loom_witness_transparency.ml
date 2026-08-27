open Unix

exception Error of string

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let schema = "loom-witness-epoch-transparency-v0"
let leaf_schema = "loom-witness-epoch-transparency-leaf-v0"
let snapshot_schema = "loom-witness-epoch-transparency-snapshot-v0"
let append_schema = "loom-witness-epoch-transparency-append-v0"
let status_schema = "loom-witness-epoch-transparency-status-v0"
let active_schema = "loom-witness-epoch-transparent-active-v0"
let event_kind = "EPOCH_TRANSPARENCY_APPEND"
let max_epochs = 64
let max_log_bytes = 4 * 1024 * 1024
let max_receipt_bytes = 1024 * 1024
let zero_digest = Loom_witness.zero_digest
let sha256 = Loom_epistemic.sha256

let failpoint point =
  match Sys.getenv_opt "SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_FAILPOINT" with
  | Some configured when configured = point ->
      Printf.eprintf "LOOM_WITNESS_EPOCH_TRANSPARENCY_FAILPOINT name=%s exit=196\n%!"
        point;
      exit 196
  | _ -> ()

let key_digest domain public_key = sha256 (domain ^ "\000" ^ public_key)
let empty_tree_root = sha256 ""
let leaf_hash canonical = sha256 ("\000" ^ canonical)

let node_hash left right =
  sha256
    ("\001" ^ Loom_witness.string_of_hex left
     ^ Loom_witness.string_of_hex right)

let largest_power_of_two_less_than value =
  let rec loop current =
    if current lsl 1 < value then loop (current lsl 1) else current
  in
  if value <= 1 then 0 else loop 1

let rec merkle_range leaves offset count =
  if count = 0 then empty_tree_root
  else if count = 1 then leaf_hash leaves.(offset)
  else
    let split = largest_power_of_two_less_than count in
    node_hash
      (merkle_range leaves offset split)
      (merkle_range leaves (offset + split) (count - split))

let merkle_root leaves = merkle_range leaves 0 (Array.length leaves)

let require_digest label value = Loom_witness.nonzero_digest label value
let digest label value = Loom_witness.digest label value

let canonical_file label path =
  try Unix.realpath path
  with Unix.Unix_error (error, _, _) ->
    failf "%s-unavailable:%s" label (Unix.error_message error)

let public_key path =
  Loom_witness.read_file_bounded "epoch-transparency-public-key" (16 * 1024)
    (canonical_file "epoch-transparency-public-key" path)

type leaf = {
  world : string;
  index : int;
  from_epoch : int;
  to_epoch : int;
  handoff_digest : string;
  previous_handoff_digest : string;
  old_membership_digest : string;
  new_membership_digest : string;
  old_state_root_digest : string;
  new_state_root_digest : string;
  previous_tree_size : int;
  previous_tree_root : string;
}

let canonical_leaf leaf =
  Printf.sprintf
    "schema=%s\nworld=%s\nindex=%d\nfrom_epoch=%d\nto_epoch=%d\nhandoff_sha256=%s\nprevious_handoff_sha256=%s\nold_membership_sha256=%s\nnew_membership_sha256=%s\nold_state_root_sha256=%s\nnew_state_root_sha256=%s\nprevious_tree_size=%d\nprevious_tree_root_sha256=%s\nnative_frame=9016\n"
    leaf_schema leaf.world leaf.index leaf.from_epoch leaf.to_epoch
    leaf.handoff_digest leaf.previous_handoff_digest
    leaf.old_membership_digest leaf.new_membership_digest
    leaf.old_state_root_digest leaf.new_state_root_digest
    leaf.previous_tree_size leaf.previous_tree_root

let parse_leaf canonical =
  let fields = Loom_witness.fields_of_text "epoch-transparency-leaf" canonical in
  if Loom_witness.field "epoch-transparency-leaf" fields "schema" <> leaf_schema
     || Loom_witness.field "epoch-transparency-leaf" fields "native_frame" <> "9016"
  then failf "epoch-transparency-leaf-schema-invalid";
  let leaf =
    { world = Loom_witness.field "epoch-transparency-leaf" fields "world";
      index =
        Loom_witness.positive_integer "epoch-transparency-leaf-index"
          (Loom_witness.field "epoch-transparency-leaf" fields "index");
      from_epoch =
        Loom_witness.positive_integer "epoch-transparency-from-epoch"
          (Loom_witness.field "epoch-transparency-leaf" fields "from_epoch");
      to_epoch =
        Loom_witness.positive_integer "epoch-transparency-to-epoch"
          (Loom_witness.field "epoch-transparency-leaf" fields "to_epoch");
      handoff_digest =
        require_digest "epoch-transparency-handoff"
          (Loom_witness.field "epoch-transparency-leaf" fields "handoff_sha256");
      previous_handoff_digest =
        digest "epoch-transparency-previous-handoff"
          (Loom_witness.field "epoch-transparency-leaf" fields
             "previous_handoff_sha256");
      old_membership_digest =
        require_digest "epoch-transparency-old-membership"
          (Loom_witness.field "epoch-transparency-leaf" fields
             "old_membership_sha256");
      new_membership_digest =
        require_digest "epoch-transparency-new-membership"
          (Loom_witness.field "epoch-transparency-leaf" fields
             "new_membership_sha256");
      old_state_root_digest =
        require_digest "epoch-transparency-old-root"
          (Loom_witness.field "epoch-transparency-leaf" fields
             "old_state_root_sha256");
      new_state_root_digest =
        require_digest "epoch-transparency-new-root"
          (Loom_witness.field "epoch-transparency-leaf" fields
             "new_state_root_sha256");
      previous_tree_size =
        Loom_witness.integer "epoch-transparency-previous-tree-size"
          (Loom_witness.field "epoch-transparency-leaf" fields
             "previous_tree_size");
      previous_tree_root =
        require_digest "epoch-transparency-previous-tree-root"
          (Loom_witness.field "epoch-transparency-leaf" fields
             "previous_tree_root_sha256") }
  in
  Loom_witness.validate_atom "epoch-transparency-world" leaf.world;
  if leaf.index > max_epochs || leaf.from_epoch > max_epochs
     || leaf.to_epoch <> leaf.from_epoch + 1
     || leaf.index <> leaf.from_epoch
     || leaf.previous_tree_size <> leaf.index - 1
     || leaf.old_membership_digest = leaf.new_membership_digest
     || leaf.old_state_root_digest = leaf.new_state_root_digest
     || (leaf.from_epoch = 1 && leaf.previous_handoff_digest <> zero_digest)
     || (leaf.from_epoch > 1 && leaf.previous_handoff_digest = zero_digest)
     || canonical_leaf leaf <> canonical
  then failf "epoch-transparency-leaf-noncanonical";
  leaf

let leaf_payload leaf tree_root =
  Loom_epistemic.encode_fields
    [ ("schema", leaf_schema); ("world", leaf.world);
      ("leaf_hex", Loom_witness.hex_of_string (canonical_leaf leaf));
      ("leaf_sha256", leaf_hash (canonical_leaf leaf));
      ("tree_size", string_of_int leaf.index);
      ("previous_tree_root_sha256", leaf.previous_tree_root);
      ("tree_root_sha256", tree_root); ("native_frame", "9016") ]

let leaves_of_events events =
  let leaves = Array.make (List.length events) "" in
  let parsed = Array.make (List.length events) None in
  List.iteri
    (fun offset (event : Loom_epistemic.event) ->
      if event.kind <> event_kind || event.sequence <> offset + 1 then
        failf "epoch-transparency-journal-event-invalid:%d" (offset + 1);
      let fields = Loom_epistemic.decode_fields event.payload in
      if Loom_epistemic.field fields "schema" <> leaf_schema
         || Loom_epistemic.field fields "native_frame" <> "9016"
      then failf "epoch-transparency-journal-payload-invalid:%d" (offset + 1);
      let canonical =
        try
          Loom_witness.string_of_hex (Loom_epistemic.field fields "leaf_hex")
        with _ -> failf "epoch-transparency-leaf-invalid-hex:%d" (offset + 1)
      in
      let leaf = parse_leaf canonical in
      leaves.(offset) <- canonical;
      parsed.(offset) <- Some leaf;
      let prefix = Array.sub leaves 0 (offset + 1) in
      let previous = merkle_range leaves 0 offset in
      let root = merkle_root prefix in
      if leaf.index <> offset + 1
         || leaf.previous_tree_root <> previous
         || Loom_epistemic.field fields "leaf_sha256" <> leaf_hash canonical
         || Loom_epistemic.field fields "tree_size" <> string_of_int (offset + 1)
         || Loom_epistemic.field fields "previous_tree_root_sha256" <> previous
         || Loom_epistemic.field fields "tree_root_sha256" <> root
         || event.payload <> leaf_payload leaf root
      then failf "epoch-transparency-merkle-recomputation-mismatch:%d" (offset + 1);
      if offset > 0 then
        let previous_leaf = Option.get parsed.(offset - 1) in
        if leaf.from_epoch <> previous_leaf.to_epoch
           || leaf.previous_handoff_digest <> previous_leaf.handoff_digest
        then failf "epoch-transparency-handoff-order-mismatch:%d" (offset + 1))
    events;
  (leaves, Array.map Option.get parsed)

type snapshot = {
  operator : string;
  operator_host : string;
  operator_key_digest : string;
  world : string;
  tree_size : int;
  tree_root : string;
  journal_event_count : int;
  journal_head : string;
  journal : string;
  payload_digest : string;
  signature : string;
  leaves : string array;
  parsed_leaves : leaf array;
}

let snapshot_payload ~operator ~operator_host ~operator_key_digest ~world
    ~tree_size ~tree_root ~journal_event_count ~journal_head ~journal =
  Printf.sprintf
    "schema=%s\noperator=%s\noperator_host=%s\noperator_key_sha256=%s\nworld=%s\ntree_size=%d\ntree_root_sha256=%s\njournal_event_count=%d\njournal_head_sha256=%s\njournal_hex=%s\n"
    snapshot_schema operator operator_host operator_key_digest world tree_size
    tree_root journal_event_count journal_head
    (Loom_witness.hex_of_string journal)

let canonical_snapshot payload payload_digest signature =
  payload ^ Printf.sprintf "payload_sha256=%s\nsignature=%s\n" payload_digest signature

let state_root state_dir = Filename.concat state_dir "epoch-transparency-log"

let load_log root world =
  let path = Loom_epistemic.journal_path root world in
  if not (Sys.file_exists path) then ([||], [||], "", zero_digest)
  else
    let events, head = Loom_epistemic.load_events path in
    let leaves, parsed = leaves_of_events events in
    let journal =
      Loom_witness.read_file_bounded "epoch-transparency-journal" max_log_bytes path
    in
    (leaves, parsed, journal, head)

let make_snapshot ~state_dir ~operator ~operator_host ~operator_public_key
    ~operator_private_key ~world =
  let root = state_root state_dir in
  let leaves, _, journal, journal_head = load_log root world in
  let tree_size = Array.length leaves in
  let tree_root = merkle_root leaves in
  let operator_key_digest =
    key_digest "loom-witness-epoch-transparency-operator-key-v0"
      operator_public_key
  in
  let payload =
    snapshot_payload ~operator ~operator_host ~operator_key_digest ~world
      ~tree_size ~tree_root ~journal_event_count:tree_size ~journal_head ~journal
  in
  let payload_digest = sha256 payload in
  let signature =
    Loom_epistemic.outcome_ed25519_sign operator_private_key payload
  in
  if not
       (Loom_epistemic.outcome_ed25519_verify operator_public_key payload signature)
  then failf "epoch-transparency-operator-private-key-mismatch";
  canonical_snapshot payload payload_digest signature

let parse_snapshot ~expected_operator ~operator_public_key canonical =
  let fields = Loom_witness.fields_of_text "epoch-transparency-snapshot" canonical in
  if Loom_witness.field "epoch-transparency-snapshot" fields "schema"
       <> snapshot_schema
  then failf "epoch-transparency-snapshot-schema-invalid";
  let operator =
    Loom_witness.field "epoch-transparency-snapshot" fields "operator"
  in
  let operator_host =
    Loom_witness.field "epoch-transparency-snapshot" fields "operator_host"
  in
  let world = Loom_witness.field "epoch-transparency-snapshot" fields "world" in
  Loom_witness.validate_atom "epoch-transparency-operator" operator;
  Loom_witness.validate_atom "epoch-transparency-operator-host" operator_host;
  Loom_witness.validate_atom "epoch-transparency-world" world;
  if operator <> expected_operator then failf "epoch-transparency-operator-mismatch";
  let operator_key_digest =
    require_digest "epoch-transparency-operator-key"
      (Loom_witness.field "epoch-transparency-snapshot" fields
         "operator_key_sha256")
  in
  let expected_key_digest =
    key_digest "loom-witness-epoch-transparency-operator-key-v0"
      operator_public_key
  in
  if operator_key_digest <> expected_key_digest then
    failf "epoch-transparency-operator-key-mismatch";
  let tree_size =
    Loom_witness.integer "epoch-transparency-tree-size"
      (Loom_witness.field "epoch-transparency-snapshot" fields "tree_size")
  in
  let tree_root =
    require_digest "epoch-transparency-tree-root"
      (Loom_witness.field "epoch-transparency-snapshot" fields
         "tree_root_sha256")
  in
  let journal_event_count =
    Loom_witness.integer "epoch-transparency-journal-count"
      (Loom_witness.field "epoch-transparency-snapshot" fields
         "journal_event_count")
  in
  let journal_head =
    digest "epoch-transparency-journal-head"
      (Loom_witness.field "epoch-transparency-snapshot" fields
         "journal_head_sha256")
  in
  let journal =
    try
      Loom_witness.string_of_hex
        (Loom_witness.field "epoch-transparency-snapshot" fields "journal_hex")
    with _ -> failf "epoch-transparency-journal-invalid-hex"
  in
  if String.length journal > max_log_bytes then
    failf "epoch-transparency-journal-too-large:%d" (String.length journal);
  let payload =
    snapshot_payload ~operator ~operator_host ~operator_key_digest ~world
      ~tree_size ~tree_root ~journal_event_count ~journal_head ~journal
  in
  let payload_digest =
    require_digest "epoch-transparency-snapshot-payload"
      (Loom_witness.field "epoch-transparency-snapshot" fields "payload_sha256")
  in
  let signature =
    Loom_witness.field "epoch-transparency-snapshot" fields "signature"
  in
  if sha256 payload <> payload_digest
     || not
          (Loom_epistemic.outcome_ed25519_verify operator_public_key payload
             signature)
     || canonical_snapshot payload payload_digest signature <> canonical
  then failf "epoch-transparency-snapshot-signature-invalid";
  let validation_root =
    Filename.temp_file "loom-epoch-transparency-validate-" ".tmp"
  in
  Unix.unlink validation_root;
  Loom_epistemic.mkdir_p validation_root;
  let journal_path = Loom_epistemic.journal_path validation_root world in
  Loom_epistemic.mkdir_p (Filename.dirname journal_path);
  if journal <> "" then Loom_witness.atomic_write journal_path journal;
  let leaves, parsed_leaves, _, recomputed_head =
    Fun.protect
      ~finally:(fun () ->
        let rec remove path =
          if Sys.file_exists path then
            if Sys.is_directory path then (
              Sys.readdir path
              |> Array.iter (fun child -> remove (Filename.concat path child));
              Unix.rmdir path)
            else Unix.unlink path
        in
        try remove validation_root with _ -> ())
      (fun () -> load_log validation_root world)
  in
  if tree_size <> Array.length leaves || tree_size <> journal_event_count
     || tree_size > max_epochs || tree_root <> merkle_root leaves
     || journal_head <> recomputed_head
     || (tree_size = 0 && journal_head <> zero_digest)
  then failf "epoch-transparency-snapshot-recomputation-mismatch";
  { operator; operator_host; operator_key_digest; world; tree_size; tree_root;
    journal_event_count; journal_head; journal; payload_digest; signature;
    leaves; parsed_leaves }

let status_request ~operator ~world =
  Printf.sprintf "schema=%s\nop=status\noperator=%s\nworld=%s\n"
    status_schema operator world

let append_payload ~operator ~publisher_key_digest ~(leaf : leaf) =
  Printf.sprintf
    "schema=%s\nop=append\noperator=%s\npublisher_key_sha256=%s\nworld=%s\nleaf_hex=%s\n"
    append_schema operator publisher_key_digest leaf.world
    (Loom_witness.hex_of_string (canonical_leaf leaf))

let append_request ~operator ~publisher_public_key ~publisher_private_key
    ~(leaf : leaf) =
  let publisher_key_digest =
    key_digest "loom-witness-epoch-transparency-publisher-key-v0"
      publisher_public_key
  in
  let payload = append_payload ~operator ~publisher_key_digest ~leaf in
  let signature =
    Loom_epistemic.outcome_ed25519_sign publisher_private_key payload
  in
  if not
       (Loom_epistemic.outcome_ed25519_verify publisher_public_key payload
          signature)
  then failf "epoch-transparency-publisher-private-key-mismatch";
  payload ^ Printf.sprintf "payload_sha256=%s\nsignature=%s\n" (sha256 payload)
    signature

let parse_append_request ~expected_operator ~publisher_public_key canonical =
  let fields = Loom_witness.fields_of_text "epoch-transparency-append" canonical in
  let operator = Loom_witness.field "epoch-transparency-append" fields "operator" in
  let world = Loom_witness.field "epoch-transparency-append" fields "world" in
  if Loom_witness.field "epoch-transparency-append" fields "schema" <> append_schema
     || Loom_witness.field "epoch-transparency-append" fields "op" <> "append"
     || operator <> expected_operator
  then failf "epoch-transparency-append-schema-invalid";
  let leaf =
    try
      Loom_witness.string_of_hex
        (Loom_witness.field "epoch-transparency-append" fields "leaf_hex")
      |> parse_leaf
    with Error _ as error -> raise error
       | _ -> failf "epoch-transparency-append-leaf-invalid"
  in
  if leaf.world <> world then failf "epoch-transparency-append-world-mismatch";
  let publisher_key_digest =
    require_digest "epoch-transparency-publisher-key"
      (Loom_witness.field "epoch-transparency-append" fields
         "publisher_key_sha256")
  in
  let expected_key_digest =
    key_digest "loom-witness-epoch-transparency-publisher-key-v0"
      publisher_public_key
  in
  if publisher_key_digest <> expected_key_digest then
    failf "epoch-transparency-publisher-key-mismatch";
  let payload = append_payload ~operator ~publisher_key_digest ~leaf in
  let payload_digest =
    require_digest "epoch-transparency-append-payload"
      (Loom_witness.field "epoch-transparency-append" fields "payload_sha256")
  in
  let signature = Loom_witness.field "epoch-transparency-append" fields "signature" in
  if sha256 payload <> payload_digest
     || not
          (Loom_epistemic.outcome_ed25519_verify publisher_public_key payload
             signature)
     || payload
        ^ Printf.sprintf "payload_sha256=%s\nsignature=%s\n" payload_digest
            signature
        <> canonical
  then failf "epoch-transparency-append-signature-invalid";
  leaf

let connect host port =
  let descriptor = Unix.socket PF_INET SOCK_STREAM 0 in
  Unix.set_close_on_exec descriptor;
  try
    Unix.connect descriptor (ADDR_INET (Loom_witness.inet_address host, port));
    descriptor
  with error ->
    Unix.close descriptor;
    raise error

let exchange host port request =
  let descriptor = connect host port in
  Fun.protect ~finally:(fun () -> Unix.close descriptor) (fun () ->
      Loom_witness.send_frame descriptor request;
      let response = Loom_witness.receive_frame descriptor in
      if Loom_witness.starts_with response "REFUSED " then
        failf "epoch-transparency-log-refused:%s"
          (String.sub response 8 (String.length response - 8));
      response)

let append_at_server ~state_dir ~operator ~operator_host ~operator_public_key
    ~operator_private_key ~publisher_public_key canonical =
  let leaf =
    parse_append_request ~expected_operator:operator ~publisher_public_key canonical
  in
  let root = state_root state_dir in
  Loom_epistemic.with_machine_lock root (fun () ->
      let leaves, parsed, _, _ = load_log root leaf.world in
      let count = Array.length leaves in
      if count > 0 && parsed.(count - 1).handoff_digest = leaf.handoff_digest
      then begin
        if canonical_leaf parsed.(count - 1) <> canonical_leaf leaf then
          failf "epoch-transparency-idempotent-leaf-conflict"
      end else begin
        if leaf.index <> count + 1 || leaf.previous_tree_size <> count ||
           leaf.previous_tree_root <> merkle_root leaves then
          failf "epoch-transparency-append-predecessor-mismatch";
        if count > 0 then begin
          let previous = parsed.(count - 1) in
          if leaf.from_epoch <> previous.to_epoch
             || leaf.previous_handoff_digest <> previous.handoff_digest then
            failf "epoch-transparency-append-order-mismatch"
        end;
        let next = Array.append leaves [| canonical_leaf leaf |] in
        let root_digest = merkle_root next in
        ignore
          (Loom_epistemic.append ~verify:(fun _ -> ()) root leaf.world event_kind
             [ ("schema", leaf_schema); ("world", leaf.world);
               ("leaf_hex", Loom_witness.hex_of_string (canonical_leaf leaf));
               ("leaf_sha256", leaf_hash (canonical_leaf leaf));
               ("tree_size", string_of_int leaf.index);
               ("previous_tree_root_sha256", leaf.previous_tree_root);
               ("tree_root_sha256", root_digest); ("native_frame", "9016") ]);
        ignore (load_log root leaf.world)
      end;
      make_snapshot ~state_dir ~operator ~operator_host ~operator_public_key
        ~operator_private_key ~world:leaf.world)

let serve ~state_dir ~operator ~operator_public_key_file ~operator_private_key
    ~publisher_public_key_file ~bind ~port =
  Loom_witness.validate_atom "epoch-transparency-operator" operator;
  let operator_public_key = public_key operator_public_key_file in
  let publisher_public_key = public_key publisher_public_key_file in
  let operator_host = Unix.gethostname () in
  Loom_witness.validate_atom "epoch-transparency-operator-host" operator_host;
  let probe = "loom-witness-epoch-transparency-operator-probe-v0\000" ^ operator in
  let signature =
    Loom_epistemic.outcome_ed25519_sign operator_private_key probe
  in
  if not
       (Loom_epistemic.outcome_ed25519_verify operator_public_key probe signature)
  then failf "epoch-transparency-operator-private-key-mismatch";
  Loom_epistemic.mkdir_p state_dir;
  let server = Unix.socket PF_INET SOCK_STREAM 0 in
  Unix.set_close_on_exec server;
  Unix.setsockopt server SO_REUSEADDR true;
  Unix.bind server (ADDR_INET (Loom_witness.inet_address bind, port));
  Unix.listen server 64;
  let actual_port =
    match Unix.getsockname server with ADDR_INET (_, value) -> value | _ -> port
  in
  Printf.printf
    "LOOM_EPOCH_TRANSPARENCY_LOG_READY schema=%s operator=%s operator_host=%s operator_key_sha256=%s bind=%s port=%d authority=storage-not-truth\n%!"
    schema operator operator_host
    (key_digest "loom-witness-epoch-transparency-operator-key-v0"
       operator_public_key)
    bind actual_port;
  while true do
    let client, _ = Unix.accept server in
    Fun.protect ~finally:(fun () -> Unix.close client) (fun () ->
        try
          let canonical = Loom_witness.receive_frame client in
          let fields =
            Loom_witness.fields_of_text "epoch-transparency-wire" canonical
          in
          let response =
            match Hashtbl.find_opt fields "op" with
            | Some "status" ->
                if Loom_witness.field "epoch-transparency-status" fields "schema"
                     <> status_schema
                   || Loom_witness.field "epoch-transparency-status" fields
                        "operator"
                      <> operator
                then failf "epoch-transparency-status-invalid";
                let world =
                  Loom_witness.field "epoch-transparency-status" fields "world"
                in
                if status_request ~operator ~world <> canonical then
                  failf "epoch-transparency-status-noncanonical";
                make_snapshot ~state_dir ~operator ~operator_host
                  ~operator_public_key ~operator_private_key ~world
            | Some "append" ->
                append_at_server ~state_dir ~operator ~operator_host
                  ~operator_public_key ~operator_private_key
                  ~publisher_public_key canonical
            | _ -> failf "epoch-transparency-wire-operation-invalid"
          in
          Loom_witness.send_frame client response
        with
        | Error error -> Loom_witness.send_frame client ("REFUSED " ^ error)
        | Loom_epistemic.Error error ->
            Loom_witness.send_frame client ("REFUSED " ^ error)
        | Loom_witness.Error error ->
            Loom_witness.send_frame client ("REFUSED " ^ error)
        | Sys_error error -> Loom_witness.send_frame client ("REFUSED " ^ error)
        | Unix_error (error, _, _) ->
            Loom_witness.send_frame client
              ("REFUSED " ^ Unix.error_message error))
  done

let query_snapshot ~host ~port ~operator ~operator_public_key ~world =
  exchange host port (status_request ~operator ~world)
  |> parse_snapshot ~expected_operator:operator ~operator_public_key

let status ~host ~port ~operator ~operator_public_key_file ~world =
  let operator_public_key = public_key operator_public_key_file in
  let snapshot =
    query_snapshot ~host ~port ~operator ~operator_public_key ~world
  in
  Printf.sprintf
    "LOOM_EPOCH_TRANSPARENCY_LOG_STATUS schema=%s operator=%s operator_host=%s world=%s tree_size=%d tree_root=%s journal_event_count=%d journal_head=%s signature=VERIFIED authority=storage-not-truth"
    schema operator snapshot.operator_host world snapshot.tree_size
    snapshot.tree_root snapshot.journal_event_count snapshot.journal_head

let append_snapshot ~host ~port ~operator ~operator_public_key
    ~publisher_public_key ~publisher_private_key ~leaf =
  exchange host port
    (append_request ~operator ~publisher_public_key ~publisher_private_key ~leaf)
  |> parse_snapshot ~expected_operator:operator ~operator_public_key

let adapter_path () =
  match Sys.getenv_opt "SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_ADAPTER" with
  | Some path when path <> "" -> path
  | _ ->
      Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name))
        "sounio-loom-witness-epoch-transparency-runtime"

let verify_operator_independence ~operator ~operator_host membership =
  let local_host = Unix.gethostname () in
  if operator = local_host
     || Array.exists (fun member -> member.Loom_witness.member_id = operator)
          membership.Loom_witness.members
  then failf "epoch-transparency-operator-principal-collapse";
  let same_host_allowed =
    Sys.getenv_opt "SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_TEST_ALLOW_SAME_HOST"
    = Some "1"
  in
  if operator_host = local_host && not same_host_allowed then
    failf "epoch-transparency-operator-host-collapse:%s" local_host;
  same_host_allowed && operator_host = local_host

let materialize_snapshot root snapshot =
  let path = Loom_epistemic.journal_path root snapshot.world in
  Loom_epistemic.mkdir_p (Filename.dirname path);
  if snapshot.journal = "" then failf "epoch-transparency-empty-log";
  Loom_witness.atomic_write path snapshot.journal;
  let events, head = Loom_epistemic.load_events path in
  let leaves, parsed = leaves_of_events events in
  if Array.length leaves <> snapshot.tree_size || head <> snapshot.journal_head
     || merkle_root leaves <> snapshot.tree_root
  then failf "epoch-transparency-materialized-snapshot-mismatch";
  (events, leaves, parsed)

let verify_current_quorum membership endpoints snapshot certificate_canonical =
  let certificate =
    Loom_witness.parse_certificate membership certificate_canonical
  in
  if certificate.certificate_world <> snapshot.world
     || certificate.certificate_membership_digest
        <> membership.Loom_witness.membership_digest
     || certificate.certificate_event_count <> snapshot.journal_event_count
     || certificate.certificate_journal_head <> snapshot.journal_head
  then failf "epoch-transparency-certificate-coordinate-mismatch";
  let statuses =
    Array.map
      (fun endpoint -> Loom_witness.query_status membership endpoint snapshot.world)
      endpoints
  in
  let receipts =
    Array.mapi
      (fun index -> function
        | Loom_witness.Latest (canonical, receipt)
          when receipt.Loom_witness.receipt_anchor_sequence
                 = certificate.certificate_sequence
               && receipt.receipt_end_count = snapshot.journal_event_count
               && receipt.receipt_end_head = snapshot.journal_head ->
            Some
              (Loom_witness.verify_receipt membership
                 membership.Loom_witness.members.(index) canonical)
        | _ -> None)
      statuses
  in
  let count =
    Array.fold_left
      (fun total -> function None -> total | Some _ -> total + 1)
      0 receipts
  in
  if count < 3 then
    failf "epoch-transparency-current-quorum-unavailable:valid=%d:required=3"
      count;
  Loom_witness.verify_native_frame membership certificate receipts;
  (certificate, count)

let native_verify ~leaf ~snapshot ~membership ~certificate ~operator
    ~operator_host:_ ~same_host_test =
  let previous_root = leaf.previous_tree_root in
  let recomputed_previous =
    merkle_range snapshot.leaves 0 leaf.previous_tree_size
  in
  let recomputed_current = merkle_root snapshot.leaves in
  let members = membership.Loom_witness.members in
  let frame =
    [ "9016"; "1"; "1"; "1"; "1"; "1"; "1"; "1"; "1"; "1";
      "3"; "4"; string_of_int leaf.from_epoch; string_of_int leaf.to_epoch;
      string_of_int leaf.previous_tree_size; string_of_int snapshot.tree_size;
      string_of_int leaf.index; string_of_int certificate.Loom_witness.certificate_sequence;
      string_of_int snapshot.journal_event_count;
      Loom_epistemic.token "loom-epoch-transparency-principal" operator;
      Loom_epistemic.token "loom-epoch-transparency-principal"
        (Unix.gethostname ());
      Loom_epistemic.token "loom-epoch-transparency-principal" members.(0).member_id;
      Loom_epistemic.token "loom-epoch-transparency-principal" members.(1).member_id;
      Loom_epistemic.token "loom-epoch-transparency-principal" members.(2).member_id;
      Loom_epistemic.token "loom-epoch-transparency-principal" members.(3).member_id ]
    @ Loom_epistemic.digest_limbs leaf.handoff_digest
    @ Loom_epistemic.digest_limbs leaf.handoff_digest
    @ Loom_epistemic.digest_limbs leaf.previous_handoff_digest
    @ Loom_epistemic.digest_limbs leaf.previous_handoff_digest
    @ Loom_epistemic.digest_limbs previous_root
    @ Loom_epistemic.digest_limbs recomputed_previous
    @ Loom_epistemic.digest_limbs snapshot.tree_root
    @ Loom_epistemic.digest_limbs recomputed_current
    @ Loom_epistemic.digest_limbs snapshot.journal_head
    @ Loom_epistemic.digest_limbs
        (Loom_witness.certificate_digest membership.topology
           (Loom_witness.canonical_certificate certificate))
    @ Loom_epistemic.digest_limbs membership.membership_digest
    @ Loom_epistemic.digest_limbs snapshot.operator_key_digest
  in
  let adapter = adapter_path () in
  if not (Sys.file_exists adapter) then
    failf "epoch-transparency-native-adapter-missing:%s" adapter;
  let code, output =
    Loom_epistemic.process_exchange (Unix.realpath adapter)
      (String.concat " " frame ^ "\n")
  in
  let expected =
    "SOUNIO_WITNESS_EPOCH_TRANSPARENCY_ACCEPT schema=loom-native-witness-epoch-transparency-v0 rollback_bound=latest-quorum-witnessed-epoch state=verified"
  in
  if code <> 0 || output <> expected then
    failf "epoch-transparency-native-refused:rc=%d:output=%s" code output;
  if same_host_test then "SIMULATED_NOT_CLAIMED" else "EXTERNAL_HOST"

let transparency_directory epoch_state_dir world =
  Filename.concat (Loom_witness_epoch.epoch_directory epoch_state_dir world)
    "transparency"

let active_path epoch_state_dir world =
  Filename.concat (transparency_directory epoch_state_dir world)
    "transparent-active.receipt"

type active = {
  world : string;
  from_epoch : int;
  to_epoch : int;
  handoff_digest : string;
  handoff_path : string;
  leaf_canonical : string;
  leaf_digest : string;
  previous_tree_size : int;
  previous_tree_root : string;
  tree_size : int;
  tree_root : string;
  operator : string;
  operator_host : string;
  operator_key_digest : string;
  membership_file : string;
  membership_file_digest : string;
  membership_digest : string;
  journal_event_count : int;
  journal_head : string;
  certificate_canonical : string;
  certificate_digest : string;
  same_host_test : bool;
}

let canonical_active active =
  Printf.sprintf
    "schema=%s\nworld=%s\nfrom_epoch=%d\nto_epoch=%d\nhandoff_sha256=%s\nhandoff_path_hex=%s\nleaf_hex=%s\nleaf_sha256=%s\nprevious_tree_size=%d\nprevious_tree_root_sha256=%s\ntree_size=%d\ntree_root_sha256=%s\noperator=%s\noperator_host=%s\noperator_key_sha256=%s\nmembership_file_hex=%s\nmembership_file_sha256=%s\nmembership_sha256=%s\njournal_event_count=%d\njournal_head_sha256=%s\ncertificate_hex=%s\ncertificate_sha256=%s\nsame_host_test=%s\nnative_frame=9016\n"
    active_schema active.world active.from_epoch active.to_epoch
    active.handoff_digest (Loom_witness.hex_of_string active.handoff_path)
    (Loom_witness.hex_of_string active.leaf_canonical) active.leaf_digest
    active.previous_tree_size active.previous_tree_root active.tree_size
    active.tree_root active.operator active.operator_host
    active.operator_key_digest
    (Loom_witness.hex_of_string active.membership_file)
    active.membership_file_digest active.membership_digest
    active.journal_event_count active.journal_head
    (Loom_witness.hex_of_string active.certificate_canonical)
    active.certificate_digest
    (if active.same_host_test then "yes" else "no")

let parse_active canonical =
  let fields = Loom_witness.fields_of_text "epoch-transparent-active" canonical in
  if Loom_witness.field "epoch-transparent-active" fields "schema" <> active_schema
     || Loom_witness.field "epoch-transparent-active" fields "native_frame" <> "9016"
  then failf "epoch-transparent-active-schema-invalid";
  let decode_path key =
    try
      Loom_witness.string_of_hex
        (Loom_witness.field "epoch-transparent-active" fields key)
    with _ -> failf "epoch-transparent-active-path-invalid:%s" key
  in
  let active =
    { world = Loom_witness.field "epoch-transparent-active" fields "world";
      from_epoch =
        Loom_witness.positive_integer "epoch-transparent-from"
          (Loom_witness.field "epoch-transparent-active" fields "from_epoch");
      to_epoch =
        Loom_witness.positive_integer "epoch-transparent-to"
          (Loom_witness.field "epoch-transparent-active" fields "to_epoch");
      handoff_digest =
        require_digest "epoch-transparent-handoff"
          (Loom_witness.field "epoch-transparent-active" fields "handoff_sha256");
      handoff_path = decode_path "handoff_path_hex";
      leaf_canonical = decode_path "leaf_hex";
      leaf_digest =
        require_digest "epoch-transparent-leaf"
          (Loom_witness.field "epoch-transparent-active" fields "leaf_sha256");
      previous_tree_size =
        Loom_witness.integer "epoch-transparent-previous-tree-size"
          (Loom_witness.field "epoch-transparent-active" fields
             "previous_tree_size");
      previous_tree_root =
        require_digest "epoch-transparent-previous-tree-root"
          (Loom_witness.field "epoch-transparent-active" fields
             "previous_tree_root_sha256");
      tree_size =
        Loom_witness.positive_integer "epoch-transparent-tree-size"
          (Loom_witness.field "epoch-transparent-active" fields "tree_size");
      tree_root =
        require_digest "epoch-transparent-tree-root"
          (Loom_witness.field "epoch-transparent-active" fields
             "tree_root_sha256");
      operator = Loom_witness.field "epoch-transparent-active" fields "operator";
      operator_host =
        Loom_witness.field "epoch-transparent-active" fields "operator_host";
      operator_key_digest =
        require_digest "epoch-transparent-operator-key"
          (Loom_witness.field "epoch-transparent-active" fields
             "operator_key_sha256");
      membership_file = decode_path "membership_file_hex";
      membership_file_digest =
        require_digest "epoch-transparent-membership-file"
          (Loom_witness.field "epoch-transparent-active" fields
             "membership_file_sha256");
      membership_digest =
        require_digest "epoch-transparent-membership"
          (Loom_witness.field "epoch-transparent-active" fields
             "membership_sha256");
      journal_event_count =
        Loom_witness.positive_integer "epoch-transparent-journal-count"
          (Loom_witness.field "epoch-transparent-active" fields
             "journal_event_count");
      journal_head =
        require_digest "epoch-transparent-journal-head"
          (Loom_witness.field "epoch-transparent-active" fields
             "journal_head_sha256");
      certificate_canonical = decode_path "certificate_hex";
      certificate_digest =
        require_digest "epoch-transparent-certificate"
          (Loom_witness.field "epoch-transparent-active" fields
             "certificate_sha256");
      same_host_test =
        (match
           Loom_witness.field "epoch-transparent-active" fields "same_host_test"
         with
        | "yes" -> true
        | "no" -> false
        | _ -> failf "epoch-transparent-same-host-test-invalid") }
  in
  if canonical_active active <> canonical then
    failf "epoch-transparent-active-noncanonical";
  active

let membership_file_digest path =
  let contents =
    Loom_witness.read_file_bounded "epoch-transparency-membership-file"
      (64 * 1024) path
  in
  sha256 ("loom-witness-epoch-transparency-membership-file-v0\000" ^ contents)

let publish ~epoch_state_dir ~transparency_state_dir ~world ~log_host ~log_port
    ~operator ~operator_public_key_file ~publisher_public_key_file
    ~publisher_private_key ~membership_file ~endpoints_file
    ~anchor_private_key =
  let active_epoch, handoff =
    match Loom_witness_epoch.load_active epoch_state_dir world with
    | None -> failf "epoch-transparency-frame-9015-active-missing:%s" world
    | Some value -> value
  in
  let handoff_canonical =
    Loom_witness.read_file_bounded "epoch-transparency-handoff"
      Loom_witness_epoch.max_handoff_bytes active_epoch.active_handoff_path
  in
  let handoff_digest = Loom_witness_epoch.handoff_digest handoff_canonical in
  if handoff_digest <> active_epoch.active_handoff_digest then
    failf "epoch-transparency-active-handoff-digest-mismatch";
  let operator_public_key = public_key operator_public_key_file in
  let publisher_public_key = public_key publisher_public_key_file in
  let membership_file = canonical_file "epoch-transparency-membership" membership_file in
  let membership = Loom_witness.load_membership membership_file in
  let endpoints = Loom_witness.load_endpoints membership endpoints_file in
  let before =
    query_snapshot ~host:log_host ~port:log_port ~operator ~operator_public_key
      ~world
  in
  let same_host_test =
    verify_operator_independence ~operator ~operator_host:before.operator_host
      membership
  in
  let proposed_leaf =
    { world; index = handoff.to_epoch - 1; from_epoch = handoff.from_epoch;
      to_epoch = handoff.to_epoch; handoff_digest;
      previous_handoff_digest = handoff.previous_handoff_digest;
      old_membership_digest = handoff.old_membership_digest;
      new_membership_digest = handoff.new_membership_digest;
      old_state_root_digest = handoff.old_state_root_digest;
      new_state_root_digest = handoff.new_state_root_digest;
      previous_tree_size = before.tree_size;
      previous_tree_root = before.tree_root }
  in
  let snapshot, leaf =
    if before.tree_size > 0
       && before.parsed_leaves.(before.tree_size - 1).handoff_digest
          = handoff_digest
    then (before, before.parsed_leaves.(before.tree_size - 1))
    else
      ( append_snapshot ~host:log_host ~port:log_port ~operator
          ~operator_public_key ~publisher_public_key ~publisher_private_key
          ~leaf:proposed_leaf,
        proposed_leaf )
  in
  failpoint "after-log-append-before-anchor";
  if snapshot.tree_size <> leaf.index
     || snapshot.journal_event_count <> leaf.index
     || snapshot.parsed_leaves.(leaf.index - 1).handoff_digest <> handoff_digest
     || snapshot.parsed_leaves.(leaf.index - 1).previous_handoff_digest
        <> handoff.previous_handoff_digest
     || snapshot.tree_root <> merkle_root snapshot.leaves
     || merkle_range snapshot.leaves 0 leaf.previous_tree_size
        <> leaf.previous_tree_root
  then failf "epoch-transparency-published-snapshot-mismatch";
  let cache_root = Filename.concat transparency_state_dir "witness-cache" in
  ignore (materialize_snapshot cache_root snapshot);
  ignore
    (Loom_witness.anchor ~root:cache_root ~world ~membership_file
       ~endpoints_file ~anchor_private_key);
  let side =
    Loom_witness_epoch.load_current_side ~root:cache_root ~world ~membership_file
      ~endpoints_file
  in
  failpoint "after-anchor-before-transparent-active";
  let certificate, _ =
    verify_current_quorum membership endpoints snapshot side.certificate_canonical
  in
  let custody =
    native_verify ~leaf ~snapshot ~membership ~certificate ~operator
      ~operator_host:snapshot.operator_host ~same_host_test
  in
  let receipt =
    { world; from_epoch = leaf.from_epoch; to_epoch = leaf.to_epoch;
      handoff_digest; handoff_path = active_epoch.active_handoff_path;
      leaf_canonical = canonical_leaf leaf;
      leaf_digest = leaf_hash (canonical_leaf leaf);
      previous_tree_size = leaf.previous_tree_size;
      previous_tree_root = leaf.previous_tree_root;
      tree_size = snapshot.tree_size; tree_root = snapshot.tree_root;
      operator; operator_host = snapshot.operator_host;
      operator_key_digest = snapshot.operator_key_digest; membership_file;
      membership_file_digest = membership_file_digest membership_file;
      membership_digest = membership.membership_digest;
      journal_event_count = snapshot.journal_event_count;
      journal_head = snapshot.journal_head;
      certificate_canonical = side.certificate_canonical;
      certificate_digest =
        Loom_witness.certificate_digest membership.topology
          side.certificate_canonical;
      same_host_test }
  in
  Loom_witness.atomic_write (active_path epoch_state_dir world)
    (canonical_active receipt);
  failpoint "after-transparent-active-before-return";
  Printf.sprintf
    "LOOM_WITNESS_EPOCH_TRANSPARENCY_OK schema=%s world=%s epoch=%d tree_size=%d tree_root=%s handoff_sha256=%s log_operator=%s operator_host=%s quorum=%d/4 proof=MATERIALIZED_PREFIX_RECOMPUTED active=LATEST rollback_bound=LATEST_QUORUM_WITNESSED_EPOCH custody=%s native_frame=9016"
    schema world leaf.to_epoch snapshot.tree_size snapshot.tree_root
    handoff_digest operator snapshot.operator_host side.current_quorum custody

let verify ~epoch_state_dir ~transparency_state_dir ~world ~log_host ~log_port
    ~operator ~operator_public_key_file ~membership_file ~endpoints_file =
  let active_canonical =
    Loom_witness.read_file_bounded "epoch-transparent-active" max_receipt_bytes
      (active_path epoch_state_dir world)
  in
  let receipt = parse_active active_canonical in
  let active_epoch, handoff =
    match Loom_witness_epoch.load_active epoch_state_dir world with
    | None -> failf "epoch-transparency-frame-9015-active-missing:%s" world
    | Some value -> value
  in
  if receipt.world <> world || receipt.to_epoch <> active_epoch.active_epoch
     || receipt.handoff_digest <> active_epoch.active_handoff_digest
     || receipt.handoff_digest
        <> Loom_witness_epoch.handoff_digest
             (Loom_witness.read_file_bounded "epoch-transparency-handoff"
                Loom_witness_epoch.max_handoff_bytes receipt.handoff_path)
     || handoff.to_epoch <> receipt.to_epoch
  then failf "epoch-transparent-active-handoff-mismatch";
  let leaf = parse_leaf receipt.leaf_canonical in
  if leaf_hash receipt.leaf_canonical <> receipt.leaf_digest
     || leaf.handoff_digest <> receipt.handoff_digest
  then failf "epoch-transparent-active-leaf-mismatch";
  let operator_public_key = public_key operator_public_key_file in
  let membership_file = canonical_file "epoch-transparency-membership" membership_file in
  let membership = Loom_witness.load_membership membership_file in
  let endpoints = Loom_witness.load_endpoints membership endpoints_file in
  let fresh =
    query_snapshot ~host:log_host ~port:log_port ~operator ~operator_public_key
      ~world
  in
  let same_host_test =
    verify_operator_independence ~operator ~operator_host:fresh.operator_host
      membership
  in
  if fresh.operator_host <> receipt.operator_host
     || fresh.operator_key_digest <> receipt.operator_key_digest
     || fresh.tree_size <> receipt.tree_size
     || fresh.tree_root <> receipt.tree_root
     || fresh.journal_event_count <> receipt.journal_event_count
     || fresh.journal_head <> receipt.journal_head
     || fresh.tree_size <> leaf.index
     || fresh.parsed_leaves.(fresh.tree_size - 1).handoff_digest
        <> receipt.handoff_digest
     || merkle_range fresh.leaves 0 receipt.previous_tree_size
        <> receipt.previous_tree_root
     || merkle_root fresh.leaves <> receipt.tree_root
     || same_host_test <> receipt.same_host_test
  then failf "epoch-transparency-rollback-or-split-view-detected";
  let cache_root = Filename.concat transparency_state_dir "verify-cache" in
  ignore (materialize_snapshot cache_root fresh);
  if membership.membership_digest <> receipt.membership_digest
     || membership_file_digest membership_file <> receipt.membership_file_digest
     || Loom_witness.certificate_digest membership.topology
          receipt.certificate_canonical
        <> receipt.certificate_digest
  then failf "epoch-transparency-authority-binding-mismatch";
  let certificate, quorum =
    verify_current_quorum membership endpoints fresh receipt.certificate_canonical
  in
  let custody =
    native_verify ~leaf ~snapshot:fresh ~membership ~certificate ~operator
      ~operator_host:fresh.operator_host ~same_host_test
  in
  Printf.sprintf
    "LOOM_WITNESS_EPOCH_TRANSPARENCY_VERIFIED schema=%s world=%s epoch=%d tree_size=%d tree_root=%s handoff_sha256=%s log_operator=%s operator_host=%s quorum=%d/4 inclusion=VERIFIED consistency=MATERIALIZED_PREFIX_RECOMPUTED active=LATEST rollback=NOT_BELOW_LATEST_QUORUM_WITNESSED freeze_claim=NONE availability_claim=NONE recovery_claim=NONE custody=%s native_frame=9016"
    schema world receipt.to_epoch fresh.tree_size fresh.tree_root
    receipt.handoff_digest operator fresh.operator_host quorum custody
