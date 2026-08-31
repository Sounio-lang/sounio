#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <pty.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#ifdef __linux__
#include <signal.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#endif
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

CAMLprim value sounio_loom_peer_credentials(value fd_value) {
  CAMLparam1(fd_value);
  CAMLlocal1(result);
#ifdef __linux__
  struct ucred credentials;
  socklen_t length = sizeof(credentials);

  if (getsockopt(Int_val(fd_value), SOL_SOCKET, SO_PEERCRED, &credentials,
                 &length) < 0 || length != sizeof(credentials)) {
    caml_failwith("SO_PEERCRED failed");
  }

  result = caml_alloc_tuple(3);
  Store_field(result, 0, Val_int(credentials.pid));
  Store_field(result, 1, Val_int(credentials.uid));
  Store_field(result, 2, Val_int(credentials.gid));
  CAMLreturn(result);
#else
  caml_failwith("SO_PEERCRED unavailable on this platform");
#endif
}

CAMLprim value sounio_loom_file_descr_of_int(value fd_value) {
  CAMLparam1(fd_value);
  const int descriptor = Int_val(fd_value);
  if (descriptor < 0 || fcntl(descriptor, F_GETFD) < 0) {
    caml_failwith("invalid inherited descriptor");
  }
  CAMLreturn(Val_int(descriptor));
}

CAMLprim value sounio_loom_int_of_file_descr(value fd_value) {
  CAMLparam1(fd_value);
  const int descriptor = Int_val(fd_value);
  if (descriptor < 0 || fcntl(descriptor, F_GETFD) < 0) {
    caml_failwith("invalid inherited descriptor");
  }
  CAMLreturn(Val_int(descriptor));
}

CAMLprim value sounio_loom_pidfd_open(value pid_value) {
  CAMLparam1(pid_value);
  CAMLlocal1(result);
#if defined(__linux__) && defined(SYS_pidfd_open)
  int descriptor = (int)syscall(SYS_pidfd_open, Int_val(pid_value), 0);
  if (descriptor < 0) {
    CAMLreturn(Val_int(0));
  }
  if (fcntl(descriptor, F_SETFD, FD_CLOEXEC) < 0) {
    close(descriptor);
    CAMLreturn(Val_int(0));
  }
  result = caml_alloc(1, 0);
  Store_field(result, 0, Val_int(descriptor));
  CAMLreturn(result);
#else
  CAMLreturn(Val_int(0));
#endif
}

CAMLprim value sounio_loom_arm_parent_death_kill(value unit) {
  CAMLparam1(unit);
#ifdef __linux__
  const pid_t parent = getppid();
  if (parent <= 1 || prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 ||
      getppid() != parent) {
    caml_failwith("PR_SET_PDEATHSIG failed");
  }
  CAMLreturn(Val_unit);
#else
  caml_failwith("PR_SET_PDEATHSIG unavailable on this platform");
#endif
}
