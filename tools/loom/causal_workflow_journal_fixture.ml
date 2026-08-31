let workflow_id = "canonical-call-b-v1"
let source_sha256 =
  "899d05ffe60528a6b71871e24fa0d1bc105cd033b7ae2c5a0a6d2bb808cdcad9"

let empty_sha256 =
  "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

let identity label = Loom_causal_workflow.sha256 label
let invocation_id label = String.sub (identity label) 0 32

let option_value = function Some value -> value | None -> "absent"
let option_int = function Some value -> string_of_int value | None -> "absent"

let print_snapshot label snapshot =
  Printf.printf
    "%s phase=%s sequence=%d compile_count=%d ticket_count=%d launch_count=%d result_count=%d attestation_count=%d head_sha256=%s controller_generation=%s start_receipt=%s unit_invocation_id=%s material_pid=%s material_start_tick=%s material_cgroup=%s barrier_nonce=%s run_pid_identity=%s exit_code=%s stdout_sha256=%s stderr_sha256=%s result_record=%s attestation_record=%s\n%!"
    label (Loom_causal_workflow.phase_name snapshot.Loom_causal_workflow.phase)
    snapshot.sequence snapshot.compile_count snapshot.ticket_count
    snapshot.launch_count snapshot.result_count snapshot.attestation_count
    snapshot.head_sha256 snapshot.controller_generation
    (option_value snapshot.start_receipt) (option_value snapshot.unit_invocation_id)
    (option_value snapshot.material_pid) (option_value snapshot.material_start_tick)
    (option_value snapshot.material_cgroup) (option_value snapshot.barrier_nonce)
    (option_value snapshot.run_pid_identity)
    (option_int snapshot.exit_code) (option_value snapshot.stdout_sha256)
    (option_value snapshot.stderr_sha256) (option_value snapshot.result_record)
    (option_value snapshot.attestation_record)

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
      ~unit_invocation_id:(invocation_id "unit-invocation-id-v1")
      ~material_pid:"4242"
      ~material_start_tick:"987654321"
      ~material_cgroup:(identity "material-cgroup-v1")
      ~barrier_nonce:(identity "barrier-nonce-v1")
      ~run_pid_identity:(identity "run-pid-identity-v1")
  in
  expect_refused "duplicate-launch" (fun () ->
      mark_run_launched ~repo_root ~state_root ~workflow_id
        ~start_receipt:(identity "run-start-receipt-v2")
        ~unit_invocation_id:(invocation_id "unit-invocation-id-v2")
        ~material_pid:"4243"
        ~material_start_tick:"987654322"
        ~material_cgroup:(identity "material-cgroup-v2")
        ~barrier_nonce:(identity "barrier-nonce-v2")
        ~run_pid_identity:(identity "run-pid-identity-v2"));
  if running.launch_count <> 1 then failf "fixture-launch-count-diverged";
  let measured =
    seal_run_result ~repo_root ~state_root ~workflow_id ~exit_code:0
      ~stdout_sha256:empty_sha256 ~stderr_sha256:empty_sha256
      ~result_record:(identity "run-result-record-v1")
      ~result_handle:(identity "run-result-handle-v1")
  in
  let closed =
    close_run ~repo_root ~state_root ~workflow_id ~pid_extinct:true
      ~descendants_extinct:true ~cgroup_unit_extinct:true ~grant_extinct:true
      ~capsule_extinct:true
  in
  let attest_armed = arm_attest ~repo_root ~state_root ~workflow_id in
  let attest_running = start_attest ~repo_root ~state_root ~workflow_id in
  let final =
    close_attest ~repo_root ~state_root ~workflow_id
      ~attestation_record:(identity "attestation-record-v1")
      ~attestation_handle:(identity "attestation-handle-v1")
  in
  ignore measured; ignore closed; ignore attest_armed; ignore attest_running;
  if final.phase <> Attested_closed || final.compile_count <> 1 ||
     final.ticket_count <> 1 || final.launch_count <> 1 ||
     final.result_count <> 1 || final.attestation_count <> 1 then
    failf "fixture-final-state-diverged";
  print_snapshot
    "PHASE_B_COMPLETE recompile=REFUSED duplicate_ticket=REFUSED duplicate_launch=REFUSED"
    final

let wrong_recovery ~repo_root ~state_root =
  ignore
    (Loom_causal_workflow.recover_controller ~repo_root ~state_root ~workflow_id
       ~successor_controller_generation:(identity "controller-generation-wrong")
       ~guardian_generation:(identity "guardian-generation-substituted")
       ~journal_id:(identity "journal-id-v1") ~store_id:(identity "store-id-v1"));
  Loom_causal_workflow.failf "fixture-wrong-recovery-unexpectedly-allowed"

let status ~repo_root ~state_root ~workflow_id label =
  Loom_causal_workflow.load_snapshot ~repo_root ~state_root ~workflow_id
  |> print_snapshot label

let require_arity expected =
  if Array.length Sys.argv <> expected then
    Loom_causal_workflow.failf "fixture-usage"

let bool_argument index =
  match Sys.argv.(index) with
  | "true" -> true
  | "false" -> false
  | _ -> Loom_causal_workflow.failf "fixture-boolean-invalid"

let int_argument index =
  try int_of_string Sys.argv.(index)
  with _ -> Loom_causal_workflow.failf "fixture-integer-invalid"

let wait_if requested = if requested then ignore (Unix.sleep 300)

let material_command command =
  let open Loom_causal_workflow in
  match command with
  | "material-open" ->
      require_arity 11;
      let repo_root, state_root, workflow_id =
        (Sys.argv.(2), Sys.argv.(3), Sys.argv.(4))
      in
      open_workflow ~repo_root ~state_root ~workflow_id
        ~workflow_generation:Sys.argv.(5) ~guardian_generation:Sys.argv.(6)
        ~journal_id:Sys.argv.(7) ~store_id:Sys.argv.(8)
        ~controller_generation:Sys.argv.(9) ~source_sha256:Sys.argv.(10)
      |> print_snapshot "MATERIAL_OPENED"
  | "material-arm-compile" ->
      require_arity 5;
      arm_compile ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
        ~workflow_id:Sys.argv.(4)
      |> print_snapshot "MATERIAL_COMPILE_ARMED"
  | "material-start-compile" ->
      require_arity 5;
      start_compile ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
        ~workflow_id:Sys.argv.(4)
      |> print_snapshot "MATERIAL_COMPILE_RUNNING"
  | "material-close-compile" ->
      require_arity 8;
      close_compile ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
        ~workflow_id:Sys.argv.(4) ~compile_receipt:Sys.argv.(5)
        ~artifact_record:Sys.argv.(6) ~artifact_handle:Sys.argv.(7)
      |> print_snapshot "MATERIAL_COMPILED"
  | "material-recover" ->
      require_arity 9;
      recover_controller ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
        ~workflow_id:Sys.argv.(4)
        ~successor_controller_generation:Sys.argv.(5)
        ~guardian_generation:Sys.argv.(6) ~journal_id:Sys.argv.(7)
        ~store_id:Sys.argv.(8)
      |> print_snapshot "MATERIAL_RECOVERED"
  | "material-recompile" ->
      require_arity 5;
      arm_compile ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
        ~workflow_id:Sys.argv.(4)
      |> print_snapshot "MATERIAL_RECOMPILE"
  | "material-arm-run" | "material-arm-run-wait" ->
      require_arity 8;
      let snapshot =
        commit_run_ticket ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
          ~workflow_id:Sys.argv.(4) ~run_ticket:Sys.argv.(5)
          ~run_grant:Sys.argv.(6) ~run_grant_generation:Sys.argv.(7)
      in
      print_snapshot "MATERIAL_RUN_ARMED" snapshot;
      wait_if (command = "material-arm-run-wait")
  | "material-mark-running" | "material-mark-running-wait" ->
      require_arity 12;
      let snapshot =
        mark_run_launched ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
          ~workflow_id:Sys.argv.(4) ~start_receipt:Sys.argv.(5)
          ~unit_invocation_id:Sys.argv.(6) ~material_pid:Sys.argv.(7)
          ~material_start_tick:Sys.argv.(8) ~material_cgroup:Sys.argv.(9)
          ~barrier_nonce:Sys.argv.(10) ~run_pid_identity:Sys.argv.(11)
      in
      print_snapshot "MATERIAL_RUNNING_IN_EXEC" snapshot;
      wait_if (command = "material-mark-running-wait")
  | "material-release-replay" ->
      require_arity 12;
      print_endline
        (admit_exec_release ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
           ~workflow_id:Sys.argv.(4) ~guardian_generation:Sys.argv.(5)
           ~unit_invocation_id:Sys.argv.(6) ~material_pid:Sys.argv.(7)
           ~material_start_tick:Sys.argv.(8) ~material_cgroup:Sys.argv.(9)
           ~run_grant_generation:Sys.argv.(10) ~barrier_nonce:Sys.argv.(11))
  | "material-claim-final" ->
      require_arity 5;
      print_endline
        (claim_after_attestation ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
           ~workflow_id:Sys.argv.(4))
  | "material-record-result" ->
      require_arity 10;
      seal_run_result ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
        ~workflow_id:Sys.argv.(4) ~exit_code:(int_argument 5)
        ~stdout_sha256:Sys.argv.(6) ~stderr_sha256:Sys.argv.(7)
        ~result_record:Sys.argv.(8) ~result_handle:Sys.argv.(9)
      |> print_snapshot "MATERIAL_RUN_MEASURED"
  | "material-close-run" ->
      require_arity 10;
      close_run ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
        ~workflow_id:Sys.argv.(4) ~pid_extinct:(bool_argument 5)
        ~descendants_extinct:(bool_argument 6)
        ~cgroup_unit_extinct:(bool_argument 7)
        ~grant_extinct:(bool_argument 8) ~capsule_extinct:(bool_argument 9)
      |> print_snapshot "MATERIAL_RUN_CLOSED"
  | "material-arm-attest" ->
      require_arity 5;
      arm_attest ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
        ~workflow_id:Sys.argv.(4)
      |> print_snapshot "MATERIAL_ATTEST_ARMED"
  | "material-start-attest" ->
      require_arity 5;
      start_attest ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
        ~workflow_id:Sys.argv.(4)
      |> print_snapshot "MATERIAL_ATTEST_RUNNING"
  | "material-close-attest" ->
      require_arity 7;
      close_attest ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
        ~workflow_id:Sys.argv.(4) ~attestation_record:Sys.argv.(5)
        ~attestation_handle:Sys.argv.(6)
      |> print_snapshot "MATERIAL_ATTESTED_CLOSED"
  | "material-status" ->
      require_arity 5;
      status ~repo_root:Sys.argv.(2) ~state_root:Sys.argv.(3)
        ~workflow_id:Sys.argv.(4) "MATERIAL_STATUS"
  | _ -> failf "fixture-command-unknown"

let () =
  try
    if Array.length Sys.argv < 2 then Loom_causal_workflow.failf "fixture-usage";
    let command = Sys.argv.(1) in
    match command with
    | "phase-a" | "phase-a-wait" | "phase-b" | "wrong-recovery" | "status" ->
        require_arity 4;
        let repo_root = Sys.argv.(2) in
        let state_root = Sys.argv.(3) in
        (match command with
        | "phase-a" -> phase_a ~repo_root ~state_root ~wait:false
        | "phase-a-wait" -> phase_a ~repo_root ~state_root ~wait:true
        | "phase-b" -> phase_b ~repo_root ~state_root
        | "wrong-recovery" -> wrong_recovery ~repo_root ~state_root
        | "status" -> status ~repo_root ~state_root ~workflow_id "STATUS"
        | _ -> assert false)
    | _ -> material_command command
  with
  | Loom_causal_workflow.Error reason ->
      prerr_endline ("CAUSAL_WORKFLOW_FIXTURE_ERROR " ^ reason);
      exit 70
  | Unix.Unix_error (error, call, path) ->
      Printf.eprintf "CAUSAL_WORKFLOW_FIXTURE_ERROR unix:%s:%s:%s\n"
        (Unix.error_message error) call path;
      exit 70
