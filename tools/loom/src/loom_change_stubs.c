#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/file.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include <caml/mlvalues.h>

static void fail_errno(const char *operation, const char *path) {
  char message[768];
  snprintf(message, sizeof(message), "%s:%s:%s", operation,
           path == NULL ? "" : path, strerror(errno));
  caml_failwith(message);
}

static void write_mapping(const char *path, unsigned int inside,
                          unsigned int outside) {
  char mapping[96];
  int length = snprintf(mapping, sizeof(mapping), "%u %u 1\n", inside, outside);
  int descriptor = open(path, O_WRONLY | O_CLOEXEC);
  if (descriptor < 0) fail_errno("open-id-map", path);
  ssize_t written = write(descriptor, mapping, (size_t)length);
  int saved = errno;
  close(descriptor);
  errno = saved;
  if (written != length) fail_errno("write-id-map", path);
}

static void deny_setgroups(void) {
  int descriptor = open("/proc/self/setgroups", O_WRONLY | O_CLOEXEC);
  if (descriptor < 0) {
    if (errno == ENOENT) return;
    fail_errno("open-setgroups", "/proc/self/setgroups");
  }
  const char value[] = "deny\n";
  ssize_t written = write(descriptor, value, sizeof(value) - 1);
  int saved = errno;
  close(descriptor);
  errno = saved;
  if (written != (ssize_t)(sizeof(value) - 1))
    fail_errno("write-setgroups", "/proc/self/setgroups");
}

CAMLprim value sounio_loom_enter_readonly_namespace(value roots_value) {
  CAMLparam1(roots_value);
#ifdef __linux__
  const uid_t uid = getuid();
  const gid_t gid = getgid();
  if (unshare(CLONE_NEWUSER) != 0) fail_errno("unshare-user", NULL);
  deny_setgroups();
  write_mapping("/proc/self/uid_map", 0, (unsigned int)uid);
  write_mapping("/proc/self/gid_map", 0, (unsigned int)gid);
  if (unshare(CLONE_NEWNS) != 0) fail_errno("unshare-mount", NULL);
  if (mount(NULL, "/", NULL, MS_REC | MS_PRIVATE, NULL) != 0)
    fail_errno("mount-private", "/");

  const mlsize_t count = Wosize_val(roots_value);
  for (mlsize_t index = 0; index < count; ++index) {
    const char *path = String_val(Field(roots_value, index));
    if (mount(path, path, NULL, MS_BIND | MS_REC, NULL) != 0)
      fail_errno("mount-bind", path);
  }
  for (mlsize_t index = 0; index < count; ++index) {
    const char *path = String_val(Field(roots_value, index));
    if (mount(NULL, path, NULL,
              MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV,
              NULL) != 0)
      fail_errno("mount-readonly", path);
  }
  if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0)
    fail_errno("prctl-dumpable", NULL);
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0)
    fail_errno("prctl-no-new-privs", NULL);
  CAMLreturn(Val_unit);
#else
  caml_failwith("sovereign change namespaces require Linux");
#endif
}

CAMLprim value sounio_loom_flock_try_exclusive(value descriptor_value) {
  CAMLparam1(descriptor_value);
#ifdef __linux__
  int result;
  int saved;
  caml_enter_blocking_section();
  result = flock(Int_val(descriptor_value), LOCK_EX | LOCK_NB);
  saved = errno;
  caml_leave_blocking_section();
  if (result == 0) CAMLreturn(Val_true);
  if (saved == EWOULDBLOCK || saved == EAGAIN) CAMLreturn(Val_false);
  errno = saved;
  fail_errno("flock-exclusive", NULL);
#else
  caml_failwith("coordination flock requires Linux");
#endif
}

CAMLprim value sounio_loom_flock_unlock(value descriptor_value) {
  CAMLparam1(descriptor_value);
#ifdef __linux__
  if (flock(Int_val(descriptor_value), LOCK_UN) != 0)
    fail_errno("flock-unlock", NULL);
  CAMLreturn(Val_unit);
#else
  caml_failwith("coordination flock requires Linux");
#endif
}

CAMLprim value sounio_loom_host_identity(value unit_value) {
  CAMLparam1(unit_value);
  CAMLlocal1(result_value);
#ifdef __linux__
  struct utsname identity;
  char result[1024];
  if (uname(&identity) != 0) fail_errno("uname", NULL);
  snprintf(result, sizeof(result), "%s|%s|%s|%s", identity.sysname,
           identity.release, identity.machine, identity.nodename);
  result_value = caml_copy_string(result);
  CAMLreturn(result_value);
#else
  caml_failwith("host identity requires Linux");
#endif
}
