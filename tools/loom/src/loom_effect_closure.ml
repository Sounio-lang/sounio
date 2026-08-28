exception Error of string

let pinned_manifest_sha256 =
  "c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91"

let max_file_bytes = 8 * 1024 * 1024
let zero_digest = "0 0 0 0 0 0 0 0"

type policy = {
  parent_9023_sha256 : string;
  parent_9024_sha256 : string;
  resident_runtime_sha256 : string;
  source_sha256 : string;
  semantics_sha256 : string;
  toolchain_sha256 : string;
  hardware_sha256 : string;
  command_sha256 : string;
  result_sha256 : string;
}

let failf format = Printf.ksprintf (fun value -> raise (Error value)) format

let sha256 value =
  Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) value
  |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())

let read_file ?(limit = max_file_bytes) path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () ->
      let output = Buffer.create 4096 in
      let bytes = Bytes.create 16384 in
      let rec loop total =
        let count = input channel bytes 0 (Bytes.length bytes) in
        if count = 0 then Buffer.contents output
        else if total + count > limit then failf "effect-closure-file-too-large:%s" path
        else (Buffer.add_subbytes output bytes 0 count; loop (total + count))
      in
      loop 0)

let sha256_file path = sha256 (read_file path)

let parse_manifest path =
  let table = Hashtbl.create 96 in
  read_file path |> String.split_on_char '\n'
  |> List.iter (fun line ->
         match String.index_opt line '=' with
         | None when line = "" -> ()
         | None -> failf "malformed-effect-closure-manifest"
         | Some index ->
             let key = String.sub line 0 index in
             if Hashtbl.mem table key then
               failf "duplicate-effect-closure-field:%s" key;
             Hashtbl.add table key
               (String.sub line (index + 1) (String.length line - index - 1)));
  table

let required table key =
  match Hashtbl.find_opt table key with
  | Some value when value <> "" -> value
  | _ -> failf "missing-effect-closure-field:%s" key

let test_mode () = Sys.getenv_opt "SOUNIO_LOOM_HOOK_TEST_MODE" = Some "1"

let manifest_path root =
  match Sys.getenv_opt "SOUNIO_LOOM_EFFECT_CLOSURE_MANIFEST" with
  | Some value when value <> "" && test_mode () -> value
  | Some value when value <> "" ->
      failf "SOUNIO_LOOM_EFFECT_CLOSURE_MANIFEST-override-requires-test-mode"
  | _ -> Filename.concat root "tools/loom/effect_closure_authority.freeze.v1"

let verify_file root manifest path_key hash_key reason =
  let path = Filename.concat root (required manifest path_key) in
  if not (Sys.file_exists path) || sha256_file path <> required manifest hash_key then
    failf "%s" reason

let load root =
  let root = Unix.realpath root in
  let path = manifest_path root in
  if not (Sys.file_exists path) then failf "effect-closure-manifest-missing";
  if sha256_file path <> pinned_manifest_sha256 then
    failf "effect-closure-manifest-hash-mismatch";
  let manifest = parse_manifest path in
  if required manifest "schema" <> "loom-effect-closure-authority-freeze-v1"
     || required manifest "stage" <> "SEMANTICS_FROZEN"
     || required manifest "producing_language" <> "Sounio"
     || required manifest "language_role" <> "SEMANTIC_AUTHORITY"
     || required manifest "action" <> "9025"
     || required manifest "parent_actions" <> "9023,9024"
     || required manifest "material_coverage" <> "false"
     || required manifest "resident_integration" <> "false"
     || required manifest "exec_attached" <> "false"
     || required manifest "commit_attached" <> "false"
     || required manifest "ci_attached" <> "false"
  then failf "effect-closure-manifest-state-invalid";
  verify_file root manifest "source_path" "source_sha256"
    "effect-closure-source-hash-mismatch";
  verify_file root manifest "entrypoint_path" "entrypoint_sha256"
    "effect-closure-entrypoint-hash-mismatch";
  verify_file root manifest "parent_9023_manifest_path"
    "parent_9023_manifest_sha256" "effect-closure-parent-9023-hash-mismatch";
  verify_file root manifest "parent_9024_manifest_path"
    "parent_9024_manifest_sha256" "effect-closure-parent-9024-hash-mismatch";
  verify_file root manifest "resident_runtime_manifest_path"
    "resident_runtime_manifest_sha256" "effect-closure-resident-v1-hash-mismatch";
  { parent_9023_sha256 = required manifest "parent_9023_manifest_sha256";
    parent_9024_sha256 = required manifest "parent_9024_manifest_sha256";
    resident_runtime_sha256 = required manifest "resident_runtime_sha256";
    source_sha256 = required manifest "source_sha256";
    semantics_sha256 = required manifest "semantics_sha256";
    toolchain_sha256 = required manifest "toolchain_record_sha256";
    hardware_sha256 = required manifest "hardware_record_sha256";
    command_sha256 = required manifest "command_sha256";
    result_sha256 = required manifest "result_sha256" }

let digest digest = Loom_resident.digest_u32_of_hex digest

let current_coverage_sha256 root =
  let path =
    Filename.concat root
      "tools/loom/evidence/loom-resident-membrane-integration-v1-20260828.txt"
  in
  if Sys.file_exists path then sha256_file path else String.make 64 '0'

let current_material_frame root =
  let root = Unix.realpath root in
  let policy = load root in
  let coverage_sha256 = current_coverage_sha256 root in
  String.concat " "
    [ "9025"; "3";
      (* Frozen parents, resident binding, and x86_64 diagnostic architecture. *)
      "1"; "1"; "1"; "1";
      (* Fail-closed and supervisor revocation exist; hostile same-UID and path
         race closure do not. *)
      "1"; "1"; "0"; "0";
      (* Twelve enumerated families, no per-family material sabotage receipts. *)
      "12"; "0"; "0";
      (* exec/process are mediated+backstopped; path/descriptor are observation
         only; the remaining known families are unclosed; unknown is denied. *)
      "3"; "3"; "1"; "1"; "0"; "0"; "0"; "0"; "0"; "0"; "0"; "2";
      digest policy.parent_9023_sha256;
      digest policy.parent_9024_sha256;
      digest policy.resident_runtime_sha256;
      digest policy.source_sha256;
      digest policy.semantics_sha256;
      digest coverage_sha256;
      zero_digest;
      digest policy.toolchain_sha256;
      digest policy.hardware_sha256;
      digest policy.command_sha256;
      digest policy.result_sha256 ]
