effect io
  fun io-op() : ()

fun f() : io int { 42 }
fun main-a() : io int { f() }
