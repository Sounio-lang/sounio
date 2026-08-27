type event = {
  agent : string;
  lane : string;
  instance_id : string;
  session_state : string;
  journal : string;
  sequence : int64;
  observed_at_utc : string;
  kind : string;
  payload : string;
  previous_sha256 : string;
  event_sha256 : string;
  journal_head_sha256 : string;
  verified : bool;
}

external encode_native : event array -> string = "sounio_loom_arrow_encode"
external inspect_native : string -> string = "sounio_loom_arrow_inspect"

let encode events = encode_native (Array.of_list events)
let inspect = inspect_native
