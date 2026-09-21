#include <errno.h>
#include <pty.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

CAMLprim value sounio_loom_forkpty(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);
  int master_fd = -1;
  pid_t pid = forkpty(&master_fd, NULL, NULL, NULL);

  if (pid < 0) {
    caml_failwith("forkpty failed");
  }

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

CAMLprim value sounio_loom_get_winsize(value fd_value) {
  CAMLparam1(fd_value);
  CAMLlocal1(result);
  struct winsize size;

  if (ioctl(Int_val(fd_value), TIOCGWINSZ, &size) < 0) {
    caml_failwith("TIOCGWINSZ failed");
  }

  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(size.ws_row));
  Store_field(result, 1, Val_int(size.ws_col));
  CAMLreturn(result);
}
