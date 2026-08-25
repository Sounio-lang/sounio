#include <errno.h>
#include <pty.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <caml/version.h>

#if OCAML_VERSION >= 50000
#define CAML_INTERNALS
#endif

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#if OCAML_VERSION >= 50000
#include <caml/domain.h>
#include <caml/runtime_events.h>
#endif

CAMLprim value sounio_loom_forkpty(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);
  int master_fd = -1;

#if OCAML_VERSION >= 50000
  if (caml_domain_is_multicore()) {
    caml_failwith("forkpty may not be called after any domain has been spawned");
  }
#endif

  pid_t pid = forkpty(&master_fd, NULL, NULL, NULL);

  if (pid < 0) {
    caml_failwith("forkpty failed");
  }

#if OCAML_VERSION >= 50000
  if (pid == 0) {
    /* Match Unix.fork before returning from C into the OCaml 5 runtime. */
    caml_runtime_events_post_fork();
    caml_atfork_hook();
  }
#endif

  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(pid));
  Store_field(result, 1, Val_int(master_fd));
  CAMLreturn(result);
}

CAMLprim value sounio_loom_set_winsize(value fd_value, value rows_value,
                                        value columns_value) {
  CAMLparam3(fd_value, rows_value, columns_value);
  struct winsize size;
  size.ws_row = Int_val(rows_value);
  size.ws_col = Int_val(columns_value);
  size.ws_xpixel = 0;
  size.ws_ypixel = 0;

  if (ioctl(Int_val(fd_value), TIOCSWINSZ, &size) < 0) {
    caml_failwith("TIOCSWINSZ failed");
  }
  CAMLreturn(Val_unit);
}
