effect io
  fun io-op() : ()
effect mut
  fun mut-op() : ()

fun f() : <io,mut> int { 42 }
fun main-b() : io int { f() }
