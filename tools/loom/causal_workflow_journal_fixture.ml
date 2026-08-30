let workflow_id = "canonical-call-b-v1"
let source_sha256 =
  "899d05ffe60528a6b71871e24fa0d1bc105cd033b7ae2c5a0a6d2bb808cdcad9"

let identity label = Loom_causal_workflow.sha256 label

let print_snapshot label snapshot =
  Printf.printf
    "%s phase=%s sequence=%d compile_count=%d ticket_count=%d launch_count=%d head_sha256=%s controller_generation=%s\n%!"
    label (Loom_causal_workflow.phase_name snapshot.Loom_causal_workflow.phase)
    snapshot.sequence snapshot.compile_count snapshot.ticket_count
    snapshot.launch_count snapshot.head_sha256 snapshot.controller_generation

let phase_a ~repo_root ~state_root ~wait =
  let open Loom_causal_workflow in
  ignore
    (open_workflow ~repo_root ~state_root ~workflow_id
      ~workflow_generation:(identity "workflow-generation-v1")
      ~guardian_generation:(identity "guardian-generation-v1")
      ~journal_id:(identity "journal-id-v1") ~store_id:(identity "store-id-v1")
      ~controller_generation:(identity "controller-generation-a")
      ~source_sha256);
  ignore (arm_compile ~repo_root ~state_root ~workflow_id);
  ignore (start_compile ~repo_root ~state_root ~workflow_id);
  let snapshot =
    close_compile ~repo_root ~state_root ~workflow_id
      ~compile_receipt:(identity "compile-receipt-v1")
      ~artifact_record:(identity "artifact-record-v1")
      ~artifact_handle:(identity "artifact-handle-v1")
  in
  if snapshot.phase <> Compiled_closed || snapshot.compile_count <> 1 then
    failf "fixture-phase-a-state-diverged";
  print_snapshot "PHASE_A_READY" snapshot;
  if wait then ignore (Unix.sleep 300)

let expect_refused label action =
  match action () with
  | _ -> Loom_causal_workflow.failf "fixture-%s-unexpectedly-allowed" label
  | exception Loom_causal_workflow.Error _ -> ()

let phase_b ~repo_root ~state_root =
  let open Loom_causal_workflow in
  let recovered =
    recover_controller ~repo_root ~state_root ~workflow_id
      ~successor_controller_generation:(identity "controller-generation-b")
      ~guardian_generation:(identity "guardian-generation-v1")
      ~journal_id:(identity "journal-id-v1") ~store_id:(identity "store-id-v1")
  in
  if recovered.phase <> Compiled_closed || recovered.compile_count <> 1 then
    failf "fixture-recovery-state-diverged";
  expect_refused "recompile" (fun () ->
      arm_compile ~repo_root ~state_root ~workflow_id);
  let armed =
    commit_run_ticket ~repo_root ~state_root ~workflow_id
      ~run_ticket:(identity "run-ticket-v1")
      ~run_grant:(identity "action-9030-run-grant-v1")
      ~run_grant_generation:(identity "action-9030-run-grant-generation-v1")
  in
  expect_refused "duplicate-ticket" (fun () ->
      commit_run_ticket ~repo_root ~state_root ~workflow_id
        ~run_ticket:(identity "run-ticket-v2")
        ~run_grant:(identity "action-9030-run-grant-v2")
        ~run_grant_generation:(identity "action-9030-run-grant-generation-v2"));
  if armed.ticket_count <> 1 || armed.launch_count <> 0 then
    failf "fixture-run-arm-state-diverged";
  let running =
    mark_run_launched ~repo_root ~state_root ~workflow_id
      ~start_receipt:(identity "run-start-receipt-v1")
      ~run_pid_identity:(identity "run-pid-identity-v1")
  in
  expect_refused "duplicate-launch" (fun () ->
      mark_run_launched ~repo_root ~state_root ~workflow_id
        ~start_receipt:(identity "run-start-receipt-v2")
        ~run_pid_identity:(identity "run-pid-identity-v2"));
  if running.launch_count <> 1 then failf "fixture-launch-count-diverged";
  let measured =
    seal_run_result ~repo_root ~state_root ~workflow_id
      ~result_record:(identity "run-result-record-v1")
      ~result_handle:(identity "run-result-handle-v1")
  in
  let closed = close_run ~repo_root ~state_root ~workflow_id in
  let attest_armed = arm_attest ~repo_root ~state_root ~workflow_id in
  let attest_running = start_attest ~repo_root ~state_root ~workflow_id in
  let final =
    close_attest ~repo_root ~state_root ~workflow_id
      ~attestation_record:(identity "attestation-record-v1")
      ~attestation_handle:(identity "attestation-handle-v1")
  in
  ignore measured; ignore closed; ignore attest_armed; ignore attest_running;
  if final.phase <> Attested_closed || final.compile_count <> 1 ||
     final.ticket_count <> 1 || final.launch_count <> 1 then
    failf "fixture-final-state-diverged";
  print_snapshot "PHASE_B_COMPLETE recompile=REFUSED duplicate_ticket=REFUSED duplicate_launch=REFUSED" final

let wrong_recovery ~repo_root ~state_root =
  ignore
    (Loom_causal_workflow.recover_controller ~repo_root ~state_root ~workflow_id
       ~successor_controller_generation:(identity "controller-generation-wrong")
       ~guardian_generation:(identity "guardian-generation-substituted")
       ~journal_id:(identity "journal-id-v1") ~store_id:(identity "store-id-v1"));
  Loom_causal_workflow.failf "fixture-wrong-recovery-unexpectedly-allowed"

let status ~repo_root ~state_root =
  let snapshot =
    Loom_causal_workflow.load_snapshot ~repo_root ~state_root ~workflow_id
  in
  print_snapshot "STATUS" snapshot

let () =
  try
    if Array.length Sys.argv <> 4 then
      Loom_causal_workflow.failf "fixture-usage";
    let command = Sys.argv.(1) in
    let repo_root = Sys.argv.(2) in
    let state_root = Sys.argv.(3) in
    match command with
    | "phase-a" -> phase_a ~repo_root ~state_root ~wait:false
    | "phase-a-wait" -> phase_a ~repo_root ~state_root ~wait:true
    | "phase-b" -> phase_b ~repo_root ~state_root
    | "wrong-recovery" -> wrong_recovery ~repo_root ~state_root
    | "status" -> status ~repo_root ~state_root
    | _ -> Loom_causal_workflow.failf "fixture-command-unknown"
  with
  | Loom_causal_workflow.Error reason ->
      prerr_endline ("CAUSAL_WORKFLOW_FIXTURE_ERROR " ^ reason);
      exit 70
  | Unix.Unix_error (error, call, path) ->
      Printf.eprintf "CAUSAL_WORKFLOW_FIXTURE_ERROR unix:%s:%s:%s\n"
        (Unix.error_message error) call path;
      exit 70
