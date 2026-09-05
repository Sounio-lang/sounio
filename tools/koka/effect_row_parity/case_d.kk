effect io
  fun io-op() : ()
effect mut
  fun mut-op() : ()
effect divv
  fun div-op() : ()

fun f() : io int { 42 }
fun main-d() : <io,mut,divv> int { f() }
