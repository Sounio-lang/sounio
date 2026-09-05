effect io
  fun io-op() : ()
effect mut
  fun mut-op() : ()
effect divv
  fun div-op() : ()
effect pan
  fun pan-op() : ()

fun f() : pan int { 42 }
fun main-e() : <io,mut,divv> int { f() }
