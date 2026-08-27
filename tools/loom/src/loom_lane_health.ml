type state =
  | Working
  | Idle
  | Blocked
  | Disconnected
  | Unresponsive
  | Orphaned
  | Dead
  | Conflicted
  | Unknown

type observation = {
  policy_state : int;
  expected_lane : bool;
  claim_active : bool;
  record_residue : bool;
  pane_or_harness_exists : bool;
  process_verified : bool;
  process_unresponsive : bool;
  process_absent : bool;
  endpoint_verified : bool;
  endpoint_absent : bool;
  endpoint_stale : bool;
  custody_active : bool;
  custody_recoverable : bool;
  obligation_active : bool;
  blocker_active : bool;
  obligation_census_complete : bool;
  progress_observed : bool;
  progress_window_complete : bool;
  liveness_window_complete : bool;
  ready_observed : bool;
  observation_authority_verified : bool;
  sample_fresh : bool;
}

let parent_semantics_sha256 =
  "5eb48f9cb214f6018569fb24e1e419b3e800dccde2e6e8d775246f4c05e4c93f"

let code = function
  | Working -> 1
  | Idle -> 2
  | Blocked -> 3
  | Disconnected -> 4
  | Unresponsive -> 5
  | Orphaned -> 6
  | Dead -> 7
  | Conflicted -> 8
  | Unknown -> 9

let name = function
  | Working -> "WORKING"
  | Idle -> "IDLE"
  | Blocked -> "BLOCKED"
  | Disconnected -> "DISCONNECTED"
  | Unresponsive -> "UNRESPONSIVE"
  | Orphaned -> "ORPHANED"
  | Dead -> "DEAD"
  | Conflicted -> "CONFLICTED"
  | Unknown -> "UNKNOWN"

(* Exact operational realization of the frozen Sounio decision order. This
   function has no authority to introduce new states or expected outcomes. *)
let classify observation =
  if observation.policy_state <> 1 then Unknown
  else if
    (not observation.observation_authority_verified)
    || not observation.sample_fresh
  then Unknown
  else if
    observation.process_absent
    && (observation.process_verified || observation.process_unresponsive)
  then Conflicted
  else if
    observation.endpoint_verified
    && (observation.endpoint_absent || observation.endpoint_stale)
  then Conflicted
  else if observation.endpoint_absent && observation.endpoint_stale then
    Conflicted
  else if observation.custody_active && observation.custody_recoverable then
    Conflicted
  else if observation.blocker_active && not observation.obligation_active then
    Conflicted
  else if observation.ready_observed && observation.obligation_active then
    Conflicted
  else if not observation.expected_lane then Unknown
  else if
    observation.process_absent
    && observation.liveness_window_complete
    && (observation.record_residue || observation.claim_active
       || observation.custody_active || observation.custody_recoverable)
  then Orphaned
  else if
    observation.process_absent
    && observation.endpoint_absent
    && observation.liveness_window_complete
    && not observation.record_residue
    && not observation.claim_active
    && not observation.custody_active
    && not observation.custody_recoverable
  then Dead
  else if
    observation.process_verified
    && observation.process_unresponsive
    && observation.liveness_window_complete
  then Unresponsive
  else if
    observation.pane_or_harness_exists
    && (observation.endpoint_absent || observation.endpoint_stale)
  then Disconnected
  else if
    observation.process_verified
    && observation.endpoint_verified
    && observation.obligation_active
    && observation.blocker_active
    && not observation.progress_observed
    && observation.progress_window_complete
  then Blocked
  else if
    observation.process_verified
    && observation.endpoint_verified
    && not observation.process_unresponsive
    && not observation.blocker_active
    && (observation.obligation_active || observation.progress_observed)
  then Working
  else if
    observation.process_verified
    && observation.endpoint_verified
    && not observation.process_unresponsive
    && not observation.obligation_active
    && not observation.blocker_active
    && not observation.progress_observed
    && observation.obligation_census_complete
    && observation.progress_window_complete
    && observation.ready_observed
  then Idle
  else Unknown

let parity_flag mask index = mask land (1 lsl index) <> 0

let parity_observation mask =
  { policy_state = 1;
    expected_lane = parity_flag mask 0;
    claim_active = parity_flag mask 1;
    record_residue = parity_flag mask 2;
    pane_or_harness_exists = parity_flag mask 3;
    process_verified = parity_flag mask 4;
    process_unresponsive = parity_flag mask 5;
    process_absent = parity_flag mask 6;
    endpoint_verified = parity_flag mask 7;
    endpoint_absent = parity_flag mask 8;
    endpoint_stale = parity_flag mask 9;
    custody_active = parity_flag mask 10;
    custody_recoverable = parity_flag mask 11;
    obligation_active = parity_flag mask 12;
    blocker_active = parity_flag mask 13;
    obligation_census_complete = parity_flag mask 14;
    progress_observed = parity_flag mask 15;
    progress_window_complete = parity_flag mask 16;
    liveness_window_complete = parity_flag mask 17;
    ready_observed = parity_flag mask 18;
    observation_authority_verified = parity_flag mask 19;
    sample_fresh = parity_flag mask 20 }

let parity_domain = 4 * (1 lsl 21)

let parity_line ?sabotage_index ~prefix () =
  let decisions = Bytes.create parity_domain in
  let counts = Array.make 10 0 in
  let flags_domain = 1 lsl 21 in
  for policy = 0 to 3 do
    for mask = 0 to flags_domain - 1 do
      let index = (policy * flags_domain) + mask in
      let decision =
        if policy = 1 then classify (parity_observation mask) else Unknown
      in
      let decision =
        match sabotage_index with
        | Some sabotage when sabotage = index ->
            if decision = Working then Unknown else Working
        | _ -> decision
      in
      let decision_code = code decision in
      Bytes.set decisions index (Char.chr decision_code);
      counts.(decision_code) <- counts.(decision_code) + 1
    done
  done;
  let digest =
    Cryptokit.hash_string (Cryptokit.Hash.sha256 ())
      (Bytes.unsafe_to_string decisions)
    |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())
  in
  Printf.sprintf
    "%s domain=%d digest_sha256=%s counts=working:%d,idle:%d,blocked:%d,disconnected:%d,unresponsive:%d,orphaned:%d,dead:%d,conflicted:%d,unknown:%d parent_semantics_sha256=%s"
    prefix parity_domain digest counts.(1) counts.(2) counts.(3) counts.(4)
    counts.(5) counts.(6) counts.(7) counts.(8) counts.(9)
    parent_semantics_sha256
