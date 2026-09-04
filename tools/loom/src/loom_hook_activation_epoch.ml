open Unix

exception Error of string

let freeze_sha256 = "b485768773da28e0b42b2a8f9d629167c9513ceacdd6df9d2df30f9f4ac82000"
let semantics_sha256 = "c2b117b11f58e90410b5222bf6fddfac7280d0d3c6af0b71f94e75efc001a017"
let executable_sha256 = "687d5ad15f0ec27a66db5da0f3df2790a4bbbfffb8a09cfaf11433e91adff46a"
let failf fmt = Printf.ksprintf (fun s -> raise (Error s)) fmt

let sha256 s = Cryptokit.hash_string (Cryptokit.Hash.sha256 ()) s |> Cryptokit.transform_string (Cryptokit.Hexa.encode ())
let read_file p =
  let ic = open_in_bin p in
  Fun.protect ~finally:(fun () -> close_in_noerr ic) (fun () -> really_input_string ic (in_channel_length ic))
let sha256_file p = sha256 (read_file p)
let fields label text =
  let t = Hashtbl.create 48 in
  String.split_on_char '\n' text |> List.iter (fun line ->
    if line <> "" then match String.index_opt line '=' with
    | None -> failf "%s-malformed" label
    | Some i ->
      let k = String.sub line 0 i and v = String.sub line (i+1) (String.length line-i-1) in
      if k = "" || v = "" || (Hashtbl.mem t k && k <> "capability") then failf "%s-invalid-field:%s" label k;
      if not (Hashtbl.mem t k) then Hashtbl.add t k v);
  t
let required label t k = match Hashtbl.find_opt t k with Some v when v <> "" -> v | _ -> failf "%s-missing:%s" label k
let regular p = let s = lstat p in if s.st_kind <> S_REG || s.st_uid <> geteuid () || s.st_perm land 0o022 <> 0 then failf "unsafe-file:%s" p
let governed p = regular p; read_file p
let rec mkdir_p p = if p="" || p="." || p="/" || Sys.file_exists p then () else (mkdir_p (Filename.dirname p); mkdir p 0o700)
let write_all fd s = let rec go o = if o < String.length s then let n=write_substring fd s o (String.length s-o) in if n=0 then failf "short-write" else go (o+n) in go 0
let atomic_write ?(exclusive=false) p s =
  mkdir_p (Filename.dirname p); let tmp=Printf.sprintf "%s.%d.tmp" p (getpid ()) in
  let fd=openfile tmp [O_WRONLY;O_CREAT;O_EXCL] 0o600 in
  Fun.protect ~finally:(fun () -> try close fd with _ -> ()) (fun () -> write_all fd s; fsync fd);
  if exclusive && Sys.file_exists p then (unlink tmp; failf "immutable-file-exists:%s" p) else rename tmp p

let runtime_root git_common = match Sys.getenv_opt "SOUNIO_COORD_RUNTIME_DIR" with Some p when p<>"" -> Unix.realpath p | _ -> Filename.concat git_common "sounio-coord-runtime"
let state_root git_common = match Sys.getenv_opt "SOUNIO_COORD_DIR" with Some p when p<>"" -> p | _ -> Filename.concat git_common "sounio-coord-state"
let safe_id id = String.iter (function 'a'..'z'|'A'..'Z'|'0'..'9'|'.'|'_'|'-' -> () | _ -> failf "unsafe-runtime-id") id; id
let validate_runtime ?(require_generation=true) root id =
  let versions=Unix.realpath (Filename.concat root "versions") in
  let dir=Unix.realpath (Filename.concat versions (safe_id id)) in
  if Filename.dirname dir <> versions then failf "runtime-path-escape";
  let mp=Filename.concat dir "manifest" in let m=fields "runtime" (governed mp) in
  if required "runtime" m "runtime_id" <> id then failf "runtime-id-drift";
  let verify key rel = let p=Filename.concat dir rel in regular p; if sha256_file p <> required "runtime" m key then failf "runtime-hash-drift:%s" key in
  verify "loom_runtime_sha256" "bin/sounio-loom-runtime"; verify "coord_runtime_sha256" "bin/sounio-coord-runtime";
  if Sys.file_exists (Filename.concat dir "hooks/sounio_coord_agent_hook_runtime.py") || Sys.file_exists (Filename.concat dir "hooks/sounio_coord_agent_hook.py") then failf "next-runtime-python-bridge";
  let manifest=governed mp in
  if require_generation &&
     (not (String.contains manifest '\n') ||
      not (List.mem "capability=loom-generation-pinned-cutover-v1" (String.split_on_char '\n' manifest)))
  then failf "next-runtime-capability-missing";
  (dir, sha256_file mp)

let load_authority source_root =
  let installed=Filename.concat (Filename.dirname (Unix.realpath Sys.executable_name)) "sounio-loom-activation-epoch" in
  let runtime_fallback=Filename.concat source_root "tools/loom/.runtime/sounio-loom-activation-epoch" in
  let build_fallback=Filename.concat source_root "tools/loom/_build/default/src/sounio-loom-activation-epoch" in
  let installed_policy=Filename.concat (Filename.dirname (Filename.dirname (Unix.realpath Sys.executable_name))) "policy/activation-epoch/tools/loom/activation_epoch.freeze.v1" in
  let use_installed=Sys.file_exists installed && Sys.file_exists installed_policy in
  let exe=if use_installed then installed else if Sys.file_exists runtime_fallback then runtime_fallback else build_fallback in
  if sha256_file exe <> executable_sha256 then failf "action-9049-executable-drift";
  let policy=if use_installed then installed_policy else Filename.concat source_root "tools/loom/activation_epoch.freeze.v1" in
  if sha256_file policy <> freeze_sha256 then failf "action-9049-freeze-drift";
  exe

let run_authority exe frame =
  let iread,iwrite=pipe () and oread,owrite=pipe () in
  match fork () with
  | 0 -> dup2 iread stdin; dup2 owrite stdout; dup2 owrite stderr; execv exe [|exe|]
  | pid -> close iread; close owrite; write_all iwrite frame; close iwrite;
    let ic=in_channel_of_descr oread in
    let out=Fun.protect ~finally:(fun()->close_in_noerr ic) (fun()->
      let b=Buffer.create 256 and bytes=Bytes.create 256 in
      let rec loop ()=let n=input ic bytes 0 256 in if n>0 then (Buffer.add_subbytes b bytes 0 n; loop ()) in
      loop (); Buffer.contents b) in
    let _,status=waitpid [] pid in
    let code=match status with WEXITED n->n|_->128 in
    if code<>0 || String.trim out <> "SOUNIO_ACTIVATION_EPOCH ADVANCE semantic_authority=Sounio action=9049" then failf "action-9049-denied:%s" (String.trim out)

let replace_candidate text next =
  String.split_on_char '\n' text |> List.map (fun l -> if String.length l >= 21 && String.sub l 0 21 = "candidate_runtime_id=" then "candidate_runtime_id="^next else l) |> String.concat "\n"

let pin_inventory pin_dir =
  Sys.readdir pin_dir |> Array.to_list |> List.sort String.compare
  |> List.filter (fun n -> Filename.check_suffix n ".pin")
  |> List.map (fun n -> let p=Filename.concat pin_dir n in regular p; n^"="^sha256_file p)
  |> String.concat "\n" |> sha256

let pinned_runtime_ids pin_dir =
  Sys.readdir pin_dir |> Array.to_list |> List.sort_uniq String.compare
  |> List.filter (fun n -> Filename.check_suffix n ".pin")
  |> List.filter_map (fun n ->
       let p=Filename.concat pin_dir n in
       let f=fields "generation-pin" (governed p) in
       if required "generation-pin" f "selection" = "capability" then
         Some (safe_id (required "generation-pin" f "runtime_id"))
       else None)
  |> List.sort_uniq String.compare

let ensure_symlink path target expected =
  if Sys.file_exists path || (try ignore (Unix.lstat path); true with _ -> false) then (
    if (Unix.lstat path).st_kind <> S_LNK then failf "activation-selector-not-symlink:%s" path;
    if Unix.realpath path <> Unix.realpath expected then failf "activation-selector-drift:%s" path)
  else Unix.symlink target path

let materialize_pin_selectors runtime_root pin_dir =
  let selector_root=Filename.concat runtime_root "generation-selectors" in
  mkdir_p selector_root;
  pinned_runtime_ids pin_dir |> List.iter (fun runtime_id ->
    let runtime_dir,_=validate_runtime ~require_generation:false runtime_root runtime_id in
    let selector=Filename.concat selector_root runtime_id in
    mkdir_p selector;
    ensure_symlink (Filename.concat selector "versions") "../../versions"
      (Filename.concat runtime_root "versions");
    ensure_symlink (Filename.concat selector "current") ("versions/"^runtime_id)
      runtime_dir;
    ensure_symlink (Filename.concat selector "native-next") ("versions/"^runtime_id)
      runtime_dir)

let digest_u60 d offset = Int64.to_string (Int64.logand (Int64.of_string ("0x"^String.sub d offset 15)) 0x0fffffffffffffffL)

let advance ~source_root ~git_common ~next_runtime =
  let rr=runtime_root git_common and state=state_root git_common in
  let pin_dir=Filename.concat state "generation-runtime-pins" in
  let head=Filename.concat pin_dir "activation.v1" in
  let current=Filename.basename (Unix.realpath (Filename.concat rr "current")) in
  let _,next_manifest=validate_runtime rr next_runtime in
  ignore(validate_runtime rr current);
  let lockp=Filename.concat pin_dir ".lock" in mkdir_p pin_dir;
  let fd=openfile lockp [O_RDWR;O_CREAT] 0o600 in
  Fun.protect ~finally:(fun()->close fd) (fun()->
    lockf fd F_LOCK 0;
    let old=governed head and hf=fields "activation-head" (governed head) in
    if required "activation-head" hf "schema" <> "loom-generation-pin-set-v1" || required "activation-head" hf "action" <> "9048" then failf "activation-head-invalid";
    let named=required "activation-head" hf "candidate_runtime_id" in
    let recovery = named <> current in
    if recovery && next_runtime <> current then failf "activation-epoch-skipped-recovery";
    if not recovery && next_runtime=current then failf "activation-epoch-noop";
    let before=pin_inventory pin_dir and old_sha=sha256 old in
    let base=Filename.concat pin_dir "activation-epochs" in let heads=Filename.concat base "heads" and epochs=Filename.concat base "epochs" in mkdir_p heads; mkdir_p epochs;
    let archive=Filename.concat heads (old_sha^".activation.v1") in if Sys.file_exists archive then (if governed archive<>old then failf "predecessor-archive-drift") else atomic_write ~exclusive:true archive old;
    let existing=Sys.readdir epochs |> Array.to_list |> List.filter (fun n->Filename.check_suffix n ".epoch.v1") in
    let previous_epoch=List.length existing in
    let epoch=previous_epoch+1 in
    let new_head=replace_candidate old next_runtime in
    let new_sha=sha256 new_head in
    materialize_pin_selectors rr pin_dir;
    let tx=sha256 (old_sha^next_manifest^before^new_sha^string_of_int epoch) in
    let exe=load_authority source_root in
    let frame=Printf.sprintf "9049 1 3 262143 %s %s %s %s %d %d\n" (digest_u60 old_sha 0) (digest_u60 next_manifest 0) (digest_u60 before 0) (digest_u60 tx 0) epoch previous_epoch in
    run_authority exe frame;
    if pin_inventory pin_dir <> before then failf "pin-inventory-drift";
    let receipt=String.concat "\n" ["schema=loom-activation-epoch-v1";"epoch="^string_of_int epoch;"previous_epoch="^string_of_int previous_epoch;"previous_head_sha256="^old_sha;"next_head_sha256="^new_sha;"previous_runtime_id="^named;"observed_current_runtime_id="^current;"recovery_epoch="^string_of_bool recovery;"next_runtime_id="^next_runtime;"next_runtime_manifest_sha256="^next_manifest;"pin_inventory_sha256="^before;"semantic_authority=Sounio";"producing_language=Sounio";"language_role=SEMANTIC_AUTHORITY";"action=9049";"semantics_sha256="^semantics_sha256;"freeze_sha256="^freeze_sha256;"projection_language=OCaml";"projection_role=OPERATIONAL_PARITY";"command=hook-activation-epoch-advance";"result=ADVANCE";""] in
    let ep=Filename.concat epochs (Printf.sprintf "%020d.epoch.v1" epoch) in atomic_write ~exclusive:true ep receipt;
    atomic_write head new_head;
    Printf.printf "SOUNIO_ACTIVATION_EPOCH ADVANCE semantic_authority=Sounio action=9049 epoch=%d previous_runtime=%s observed_current=%s next_runtime=%s predecessor_sha256=%s pin_inventory_sha256=%s\n%!" epoch named current next_runtime old_sha before)

let run arguments =
  let rec parse sr gc nr = function []->sr,gc,nr | "--source-root"::v::xs->parse v gc nr xs | "--git-common"::v::xs->parse sr v nr xs | "--next-runtime"::v::xs->parse sr gc v xs | _->failf "usage: hook-activation-epoch-advance --source-root ROOT --git-common DIR --next-runtime ID" in
  let sr,gc,nr=parse "" "" "" arguments in if sr=""||gc=""||nr="" then failf "activation-epoch-missing-argument";
  advance ~source_root:(Unix.realpath sr) ~git_common:(Unix.realpath gc) ~next_runtime:nr; 0
