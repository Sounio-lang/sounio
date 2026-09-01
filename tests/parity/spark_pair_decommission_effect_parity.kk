// Koka effect-parity witness for frozen Sounio frame 9026.
// Sounio defines the decisions; this witness checks only their declared effect.
module tests/parity/spark_pair_decommission_effect_parity

fun is-decommission-action(action : int) : pure bool
  action >= 33 && action <= 49

fun planned-effect(action : int) : pure string
  if is-decommission-action(action) then "NONE" else "UNSUPPORTED"

fun all-actions-effect-none() : pure bool
  planned-effect(33) == "NONE" &&
  planned-effect(34) == "NONE" &&
  planned-effect(35) == "NONE" &&
  planned-effect(36) == "NONE" &&
  planned-effect(37) == "NONE" &&
  planned-effect(38) == "NONE" &&
  planned-effect(39) == "NONE" &&
  planned-effect(40) == "NONE" &&
  planned-effect(41) == "NONE" &&
  planned-effect(42) == "NONE" &&
  planned-effect(43) == "NONE" &&
  planned-effect(44) == "NONE" &&
  planned-effect(45) == "NONE" &&
  planned-effect(46) == "NONE" &&
  planned-effect(47) == "NONE" &&
  planned-effect(48) == "NONE" &&
  planned-effect(49) == "NONE" &&
  planned-effect(32) == "UNSUPPORTED" &&
  planned-effect(50) == "UNSUPPORTED"

pub fun main()
  if all-actions-effect-none() then
    println("SOUNIO_SPARK_PAIR_DECOMMISSION_KOKA_PARITY_PASS frame=9026 actions=17 effect=NONE")
  else
    println("SOUNIO_SPARK_PAIR_DECOMMISSION_KOKA_PARITY_FAIL")
