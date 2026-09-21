#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/landlock.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <caml/alloc.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#ifndef __WALL
#define __WALL 0x40000000
#endif

#ifndef O_TMPFILE
#define O_TMPFILE 020000000
#endif

enum membrane_result_kind {
  MEMBRANE_EXITED = 1,
  MEMBRANE_SIGNALED = 2,
  MEMBRANE_STOPPED = 3,
  MEMBRANE_TIMED_OUT = 4,
  MEMBRANE_POLICY_DENIED = 5,
  MEMBRANE_NATIVE_ERROR = 6
};

enum membrane_native_flag {
  MEMBRANE_TEST_DISABLE_FS_OBSERVER = 1
};

struct tracee_state {
  pid_t pid;
  int entering_syscall;
};

struct tracee_table {
  struct tracee_state *items;
  size_t length;
  size_t capacity;
};

static int64_t monotonic_us(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) < 0) {
    return 0;
  }
  return ((int64_t)value.tv_sec * 1000000LL) + (value.tv_nsec / 1000LL);
}

CAMLprim value sounio_loom_monotonic_us(value unit_value) {
  CAMLparam1(unit_value);
  int64_t now_us = monotonic_us();
  if (now_us <= 0) {
    caml_failwith("CLOCK_MONOTONIC unavailable");
  }
  CAMLreturn(caml_copy_int64(now_us));
}

static int landlock_abi_version(void) {
#if defined(SYS_landlock_create_ruleset)
  return (int)syscall(SYS_landlock_create_ruleset, NULL, 0,
                      LANDLOCK_CREATE_RULESET_VERSION);
#else
  errno = ENOSYS;
  return -1;
#endif
}

static int close_inherited_descriptors(void) {
#if defined(SYS_close_range)
  return (int)syscall(SYS_close_range, 3U, ~0U, 0U);
#else
  errno = ENOSYS;
  return -1;
#endif
}

static void tracee_table_free(struct tracee_table *table) {
  free(table->items);
  table->items = NULL;
  table->length = 0;
  table->capacity = 0;
}

static struct tracee_state *tracee_find(struct tracee_table *table, pid_t pid) {
  size_t index;
  for (index = 0; index < table->length; index++) {
    if (table->items[index].pid == pid) {
      return &table->items[index];
    }
  }
  return NULL;
}

static int tracee_add(struct tracee_table *table, pid_t pid) {
  struct tracee_state *existing = tracee_find(table, pid);
  struct tracee_state *grown;
  size_t capacity;
  if (existing != NULL) {
    return 0;
  }
  if (table->length == table->capacity) {
    capacity = table->capacity == 0 ? 8 : table->capacity * 2;
    grown = realloc(table->items, capacity * sizeof(*grown));
    if (grown == NULL) {
      return -1;
    }
    table->items = grown;
    table->capacity = capacity;
  }
  table->items[table->length].pid = pid;
  table->items[table->length].entering_syscall = 1;
  table->length++;
  return 0;
}

static void tracee_remove(struct tracee_table *table, pid_t pid) {
  size_t index;
  for (index = 0; index < table->length; index++) {
    if (table->items[index].pid == pid) {
      table->items[index] = table->items[table->length - 1];
      table->length--;
      return;
    }
  }
}

static char *copy_ocaml_string(value source) {
  mlsize_t length = caml_string_length(source);
  char *copy = malloc((size_t)length + 1);
  if (copy == NULL) {
    return NULL;
  }
  memcpy(copy, String_val(source), length);
  copy[length] = '\0';
  return copy;
}

static char **copy_ocaml_array(value source) {
  mlsize_t length = Wosize_val(source);
  char **copy = calloc((size_t)length + 1, sizeof(*copy));
  mlsize_t index;
  if (copy == NULL) {
    return NULL;
  }
  for (index = 0; index < length; index++) {
    copy[index] = copy_ocaml_string(Field(source, index));
    if (copy[index] == NULL) {
      while (index > 0) {
        index--;
        free(copy[index]);
      }
      free(copy);
      return NULL;
    }
  }
  return copy;
}

static void free_string_array(char **items) {
  size_t index;
  if (items == NULL) {
    return;
  }
  for (index = 0; items[index] != NULL; index++) {
    free(items[index]);
  }
  free(items);
}

static int read_tracee_bytes(pid_t pid, unsigned long address, void *target,
                              size_t length) {
  size_t offset = 0;
  unsigned char *output = target;
  while (offset < length) {
    long word;
    size_t count = sizeof(word);
    errno = 0;
    word = ptrace(PTRACE_PEEKDATA, pid, (void *)(address + offset), NULL);
    if (word == -1 && errno != 0) {
      return -1;
    }
    if (count > length - offset) {
      count = length - offset;
    }
    memcpy(output + offset, &word, count);
    offset += count;
  }
  return 0;
}

static int read_tracee_string(pid_t pid, unsigned long address, char *target,
                               size_t capacity) {
  size_t offset = 0;
  if (address == 0 || capacity < 2) {
    return -1;
  }
  while (offset + 1 < capacity) {
    long word;
    size_t index;
    errno = 0;
    word = ptrace(PTRACE_PEEKDATA, pid, (void *)(address + offset), NULL);
    if (word == -1 && errno != 0) {
      return -1;
    }
    for (index = 0; index < sizeof(word) && offset + 1 < capacity; index++) {
      char byte = ((char *)&word)[index];
      target[offset++] = byte;
      if (byte == '\0') {
        return 0;
      }
    }
  }
  target[capacity - 1] = '\0';
  return -1;
}

static int read_proc_link(pid_t pid, const char *suffix, char *target,
                          size_t capacity) {
  char source[128];
  ssize_t length;
  if (snprintf(source, sizeof(source), "/proc/%d/%s", pid, suffix) >=
      (int)sizeof(source)) {
    return -1;
  }
  length = readlink(source, target, capacity - 1);
  if (length < 0 || (size_t)length >= capacity - 1) {
    return -1;
  }
  target[length] = '\0';
  return 0;
}

static void lexical_normalize(const char *source, char *target,
                              size_t capacity) {
  char copy[PATH_MAX * 2];
  char *parts[PATH_MAX / 2];
  size_t count = 0;
  char *save = NULL;
  char *part;
  size_t used = 0;
  if (strlen(source) >= sizeof(copy)) {
    snprintf(target, capacity, "<path-too-long>");
    return;
  }
  strcpy(copy, source);
  for (part = strtok_r(copy, "/", &save); part != NULL;
       part = strtok_r(NULL, "/", &save)) {
    if (strcmp(part, ".") == 0 || strcmp(part, "") == 0) {
      continue;
    }
    if (strcmp(part, "..") == 0) {
      if (count > 0) {
        count--;
      }
      continue;
    }
    parts[count++] = part;
  }
  if (capacity == 0) {
    return;
  }
  target[0] = '\0';
  if (used + 1 < capacity) {
    target[used++] = '/';
    target[used] = '\0';
  }
  for (size_t index = 0; index < count; index++) {
    size_t length = strlen(parts[index]);
    if (used > 1 && used + 1 < capacity) {
      target[used++] = '/';
    }
    if (used + length >= capacity) {
      snprintf(target, capacity, "<path-too-long>");
      return;
    }
    memcpy(target + used, parts[index], length);
    used += length;
    target[used] = '\0';
  }
}

static int resolve_tracee_path(pid_t pid, int dirfd, const char *path,
                               char *target, size_t capacity) {
  char base[PATH_MAX];
  char combined[PATH_MAX * 2];
  char suffix[64];
  if (path == NULL || path[0] == '\0') {
    if (dirfd < 0) {
      return -1;
    }
    if (snprintf(suffix, sizeof(suffix), "fd/%d", dirfd) >=
        (int)sizeof(suffix)) {
      return -1;
    }
    return read_proc_link(pid, suffix, target, capacity);
  }
  if (path[0] == '/') {
    lexical_normalize(path, target, capacity);
    return target[0] == '<' ? -1 : 0;
  }
  if (dirfd == AT_FDCWD) {
    if (read_proc_link(pid, "cwd", base, sizeof(base)) < 0) {
      return -1;
    }
  } else {
    if (snprintf(suffix, sizeof(suffix), "fd/%d", dirfd) >=
        (int)sizeof(suffix) ||
        read_proc_link(pid, suffix, base, sizeof(base)) < 0) {
      return -1;
    }
  }
  if (snprintf(combined, sizeof(combined), "%s/%s", base, path) >=
      (int)sizeof(combined)) {
    return -1;
  }
  lexical_normalize(combined, target, capacity);
  return target[0] == '<' ? -1 : 0;
}

static int write_open_flags(unsigned long flags) {
  return (flags & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC | O_APPEND |
                   O_TMPFILE)) != 0;
}

static int callback_decision(value decision_closure, int kind, pid_t pid,
                             long syscall_number, const char *target,
                             size_t active_count, value *event,
                             value *callback_result) {
  *event = caml_alloc_tuple(5);
  Store_field(*event, 0, Val_int(kind));
  Store_field(*event, 1, Val_int(pid));
  Store_field(*event, 2, Val_long(syscall_number));
  Store_field(*event, 3, caml_copy_string(target));
  Store_field(*event, 4, Val_long(active_count));
  *callback_result = caml_callback_exn(decision_closure, *event);
  if (Is_exception_result(*callback_result)) {
    return -403;
  }
  return Int_val(*callback_result);
}

static int emit_path_callback(value decision_closure, int kind, pid_t pid,
                              long syscall_number, int dirfd,
                              unsigned long address, size_t active_count,
                              value *event, value *callback_result) {
  char raw[PATH_MAX];
  char resolved[PATH_MAX * 2];
  if (read_tracee_string(pid, address, raw, sizeof(raw)) < 0 ||
      resolve_tracee_path(pid, dirfd, raw, resolved, sizeof(resolved)) < 0) {
    return callback_decision(decision_closure, kind, pid, syscall_number,
                             "<unreadable-target>", active_count, event,
                             callback_result);
  }
  return callback_decision(decision_closure, kind, pid, syscall_number, resolved,
                           active_count, event, callback_result);
}

static int emit_exec_callback(value decision_closure, pid_t pid,
                              long syscall_number, int dirfd,
                              unsigned long address, size_t active_count,
                              const char *sandbox_executable,
                              const char *user_executable,
                              int *sandbox_exec_seen, int *sandbox_ready,
                              value *event, value *callback_result) {
  char raw[PATH_MAX];
  char resolved[PATH_MAX * 2];
  const char *target = "<unreadable-target>";
  if (read_tracee_string(pid, address, raw, sizeof(raw)) == 0 &&
      resolve_tracee_path(pid, dirfd, raw, resolved, sizeof(resolved)) == 0) {
    target = resolved;
    if (!*sandbox_exec_seen && strcmp(target, sandbox_executable) == 0) {
      *sandbox_exec_seen = 1;
    } else if (*sandbox_exec_seen && strcmp(target, user_executable) == 0) {
      *sandbox_ready = 1;
    }
  }
  return callback_decision(decision_closure, 3, pid, syscall_number, target,
                           active_count, event, callback_result);
}

static void kill_tracees(pid_t root, struct tracee_table *table) {
  size_t index;
  if (root > 0) {
    kill(-root, SIGKILL);
    kill(root, SIGKILL);
  }
  for (index = 0; index < table->length; index++) {
    ptrace(PTRACE_CONT, table->items[index].pid, NULL,
           (void *)(intptr_t)SIGKILL);
    kill(table->items[index].pid, SIGKILL);
  }
}

static int ptrace_resume(pid_t pid, int signal_number) {
  return ptrace(PTRACE_SYSCALL, pid, NULL, (void *)(intptr_t)signal_number) < 0
             ? -1
             : 0;
}

CAMLprim value sounio_loom_membrane_supervise(value config,
                                               value decision_closure) {
  CAMLparam2(config, decision_closure);
  CAMLlocal3(result, event, callback_result);
  char *executable = NULL;
  char **arguments = NULL;
  char **environment = NULL;
  char *cwd = NULL;
  char *user_executable = NULL;
  int64_t deadline_us;
  int native_flags;
  int landlock_abi = 0;
  int64_t started_us = monotonic_us();
  pid_t root = -1;
  struct tracee_table tracees = {0};
  int result_kind = MEMBRANE_NATIVE_ERROR;
  int result_exit = 0;
  int result_signal = 0;
  int denial_code = 0;
  int timed_out = 0;
  int policy_error = 0;
  int event_count = 0;
  int root_status_seen = 0;
  int native_error = 0;
  int sandbox_exec_seen = 0;
  int sandbox_ready = 0;

#if !defined(__linux__) || !defined(__x86_64__)
  (void)config;
  (void)decision_closure;
  result = caml_alloc_tuple(10);
  Store_field(result, 0, Val_int(MEMBRANE_NATIVE_ERROR));
  Store_field(result, 1, Val_int(0));
  Store_field(result, 2, Val_int(0));
  Store_field(result, 3, caml_copy_int64(0));
  Store_field(result, 4, Val_int(0));
  Store_field(result, 5, Val_int(415));
  Store_field(result, 6, Val_int(0));
  Store_field(result, 7, Val_int(1));
  Store_field(result, 8, Val_int(0));
  Store_field(result, 9, Val_int(0));
  CAMLreturn(result);
#else
  if (Wosize_val(config) != 7) {
    caml_invalid_argument("invalid membrane supervisor config");
  }
  executable = copy_ocaml_string(Field(config, 0));
  arguments = copy_ocaml_array(Field(config, 1));
  environment = copy_ocaml_array(Field(config, 2));
  cwd = copy_ocaml_string(Field(config, 3));
  user_executable = copy_ocaml_string(Field(config, 4));
  deadline_us = Int64_val(Field(config, 5));
  native_flags = Int_val(Field(config, 6));
  landlock_abi = landlock_abi_version();
  if (landlock_abi < 0) {
    landlock_abi = -errno;
  }
  if (started_us <= 0 || deadline_us <= 0) {
    native_error = 1;
    goto finish;
  }
  if (executable == NULL || arguments == NULL || environment == NULL ||
      cwd == NULL || user_executable == NULL) {
    native_error = 1;
    goto finish;
  }

  root = fork();
  if (root < 0) {
    native_error = 1;
    goto finish;
  }
  if (root == 0) {
    if (setpgid(0, 0) < 0 || close_inherited_descriptors() < 0 ||
        chdir(cwd) < 0) {
      _exit(125);
    }
    if (ptrace(PTRACE_TRACEME, 0, NULL, NULL) < 0) {
      _exit(126);
    }
    raise(SIGSTOP);
    execve(executable, arguments, environment);
    _exit(127);
  }
  setpgid(root, root);
  if (tracee_add(&tracees, root) < 0) {
    native_error = 1;
    kill_tracees(root, &tracees);
    goto drain;
  }

  {
    int status;
    pid_t waited;
    do {
      waited = waitpid(root, &status, WUNTRACED);
    } while (waited < 0 && errno == EINTR);
    if (waited == root && (WIFEXITED(status) || WIFSIGNALED(status))) {
      root_status_seen = 1;
      tracee_remove(&tracees, root);
      native_error = 1;
      goto finish;
    }
    if (waited != root || !WIFSTOPPED(status) ||
        ptrace(PTRACE_SETOPTIONS, root, NULL,
               (void *)(intptr_t)(PTRACE_O_TRACESYSGOOD |
                                  PTRACE_O_TRACEFORK |
                                  PTRACE_O_TRACEVFORK |
                                  PTRACE_O_TRACECLONE |
                                  PTRACE_O_TRACEEXEC |
                                  PTRACE_O_TRACEEXIT |
                                  PTRACE_O_EXITKILL)) < 0) {
      native_error = 1;
      kill_tracees(root, &tracees);
      goto drain;
    }
    if (ptrace_resume(root, 0) < 0) {
      native_error = 1;
      kill_tracees(root, &tracees);
      goto drain;
    }
  }

  while (tracees.length > 0) {
    int status = 0;
    pid_t pid;
    int64_t now = monotonic_us();
    if (now <= 0) {
      native_error = 1;
      kill_tracees(root, &tracees);
      break;
    }
    if (now - started_us >= deadline_us) {
      timed_out = 1;
      result_kind = MEMBRANE_TIMED_OUT;
      result_signal = SIGKILL;
      kill_tracees(root, &tracees);
      break;
    }
    pid = waitpid(-1, &status, __WALL | WNOHANG);
    if (pid == 0) {
      struct timespec pause = {0, 1000000};
      nanosleep(&pause, NULL);
      continue;
    }
    if (pid < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == ECHILD) {
        tracees.length = 0;
        break;
      }
      native_error = 1;
      kill_tracees(root, &tracees);
      break;
    }
    if (WIFEXITED(status) || WIFSIGNALED(status)) {
      if (pid == root) {
        root_status_seen = 1;
        if (WIFEXITED(status)) {
          result_kind = MEMBRANE_EXITED;
          result_exit = WEXITSTATUS(status);
        } else {
          result_kind = MEMBRANE_SIGNALED;
          result_signal = WTERMSIG(status);
        }
      }
      tracee_remove(&tracees, pid);
      continue;
    }
    if (!WIFSTOPPED(status)) {
      continue;
    }

    {
      unsigned int ptrace_event = (unsigned int)status >> 16;
      int stop_signal = WSTOPSIG(status);
      struct tracee_state *tracee = tracee_find(&tracees, pid);
      if (tracee == NULL) {
        if (tracee_add(&tracees, pid) < 0) {
          native_error = 1;
          kill_tracees(root, &tracees);
          break;
        }
        tracee = tracee_find(&tracees, pid);
      }

      if (ptrace_event == PTRACE_EVENT_FORK ||
          ptrace_event == PTRACE_EVENT_VFORK ||
          ptrace_event == PTRACE_EVENT_CLONE) {
        unsigned long child = 0;
        if (ptrace(PTRACE_GETEVENTMSG, pid, NULL, &child) < 0 ||
            child == 0 || tracee_add(&tracees, (pid_t)child) < 0) {
          native_error = 1;
          kill_tracees(root, &tracees);
          break;
        }
      } else if (ptrace_event == PTRACE_EVENT_EXEC) {
        /* The next syscall-stop completes the execve that raised this event. */
        tracee->entering_syscall = 0;
      }

      if (stop_signal == (SIGTRAP | 0x80)) {
        struct user_regs_struct registers;
        if (ptrace(PTRACE_GETREGS, pid, NULL, &registers) < 0) {
          native_error = 1;
          kill_tracees(root, &tracees);
          break;
        }
        if (tracee->entering_syscall) {
          long number = (long)registers.orig_rax;
          int decision = 0;
          if (number == SYS_execve) {
            decision = emit_exec_callback(
                decision_closure, pid, number, AT_FDCWD, registers.rdi,
                tracees.length, executable, user_executable, &sandbox_exec_seen,
                &sandbox_ready, &event, &callback_result);
#ifdef SYS_execveat
          } else if (number == SYS_execveat) {
            decision = emit_exec_callback(
                decision_closure, pid, number, (int)registers.rdi, registers.rsi,
                tracees.length, executable, user_executable, &sandbox_exec_seen,
                &sandbox_ready, &event, &callback_result);
#endif
          } else if (number == SYS_clone || number == SYS_fork ||
                     number == SYS_vfork
#ifdef SYS_clone3
                     || number == SYS_clone3
#endif
          ) {
            decision = callback_decision(decision_closure, 2, pid, number,
                                         "process-create", tracees.length,
                                         &event, &callback_result);
          } else if (sandbox_ready &&
                     !(native_flags & MEMBRANE_TEST_DISABLE_FS_OBSERVER) &&
                     number == SYS_open &&
                     write_open_flags(registers.rsi)) {
            decision = emit_path_callback(decision_closure, 4, pid, number, AT_FDCWD,
                                          registers.rdi, tracees.length, &event,
                                          &callback_result);
          } else if (sandbox_ready &&
                     !(native_flags & MEMBRANE_TEST_DISABLE_FS_OBSERVER) &&
                     number == SYS_openat &&
                     write_open_flags(registers.rdx)) {
            decision = emit_path_callback(decision_closure, 4, pid, number,
                                          (int)registers.rdi, registers.rsi,
                                          tracees.length, &event,
                                          &callback_result);
#ifdef SYS_openat2
          } else if (sandbox_ready &&
                     !(native_flags & MEMBRANE_TEST_DISABLE_FS_OBSERVER) &&
                     number == SYS_openat2) {
            uint64_t flags = 0;
            if (read_tracee_bytes(pid, registers.rdx, &flags, sizeof(flags)) < 0) {
              decision = callback_decision(decision_closure, 4, pid, number,
                                           "<unreadable-open-how>", tracees.length,
                                           &event, &callback_result);
            } else if (write_open_flags((unsigned long)flags)) {
              decision = emit_path_callback(decision_closure, 4, pid, number,
                                            (int)registers.rdi, registers.rsi,
                                            tracees.length, &event,
                                            &callback_result);
            }
#endif
          } else if (sandbox_ready &&
                     !(native_flags & MEMBRANE_TEST_DISABLE_FS_OBSERVER) &&
                     number == SYS_creat) {
            decision = emit_path_callback(decision_closure, 4, pid, number, AT_FDCWD,
                                          registers.rdi, tracees.length, &event,
                                          &callback_result);
          } else if (sandbox_ready &&
                     !(native_flags & MEMBRANE_TEST_DISABLE_FS_OBSERVER) &&
                     (number == SYS_unlink || number == SYS_rmdir ||
                      number == SYS_mkdir || number == SYS_truncate ||
                      number == SYS_chmod || number == SYS_chown ||
                      number == SYS_lchown || number == SYS_mknod)) {
            decision = emit_path_callback(decision_closure, 5, pid, number, AT_FDCWD,
                                          registers.rdi, tracees.length, &event,
                                          &callback_result);
          } else if (sandbox_ready &&
                     !(native_flags & MEMBRANE_TEST_DISABLE_FS_OBSERVER) &&
                     (number == SYS_unlinkat || number == SYS_mkdirat ||
                      number == SYS_mknodat || number == SYS_fchmodat ||
                      number == SYS_fchownat)) {
            decision = emit_path_callback(decision_closure, 5, pid, number,
                                          (int)registers.rdi, registers.rsi,
                                          tracees.length, &event,
                                          &callback_result);
          } else if (sandbox_ready &&
                     !(native_flags & MEMBRANE_TEST_DISABLE_FS_OBSERVER) &&
                     (number == SYS_ftruncate || number == SYS_fchmod ||
                      number == SYS_fchown)) {
            decision = callback_decision(decision_closure, 5, pid, number,
                                         "<unsupported-fd-target>",
                                         tracees.length, &event,
                                         &callback_result);
          } else if (sandbox_ready &&
                     !(native_flags & MEMBRANE_TEST_DISABLE_FS_OBSERVER) &&
                     (number == SYS_rename || number == SYS_link ||
                      number == SYS_symlink)) {
            decision = emit_path_callback(decision_closure, 5, pid, number, AT_FDCWD,
                                          registers.rdi, tracees.length, &event,
                                          &callback_result);
            if (decision == 0) {
              decision = emit_path_callback(decision_closure, 5, pid, number, AT_FDCWD,
                                            registers.rsi, tracees.length, &event,
                                            &callback_result);
            }
          } else if (sandbox_ready &&
                     !(native_flags & MEMBRANE_TEST_DISABLE_FS_OBSERVER) &&
                     (number == SYS_renameat || number == SYS_linkat)) {
            decision = emit_path_callback(decision_closure, 5, pid, number,
                                          (int)registers.rdi, registers.rsi,
                                          tracees.length, &event,
                                          &callback_result);
            if (decision == 0) {
              decision = emit_path_callback(decision_closure, 5, pid, number,
                                            (int)registers.rdx, registers.r10,
                                            tracees.length, &event,
                                            &callback_result);
            }
#ifdef SYS_renameat2
          } else if (sandbox_ready &&
                     !(native_flags & MEMBRANE_TEST_DISABLE_FS_OBSERVER) &&
                     number == SYS_renameat2) {
            decision = emit_path_callback(decision_closure, 5, pid, number,
                                          (int)registers.rdi, registers.rsi,
                                          tracees.length, &event,
                                          &callback_result);
            if (decision == 0) {
              decision = emit_path_callback(decision_closure, 5, pid, number,
                                            (int)registers.rdx, registers.r10,
                                            tracees.length, &event,
                                            &callback_result);
            }
#endif
#ifdef SYS_symlinkat
          } else if (sandbox_ready &&
                     !(native_flags & MEMBRANE_TEST_DISABLE_FS_OBSERVER) &&
                     number == SYS_symlinkat) {
            decision = emit_path_callback(decision_closure, 5, pid, number,
                                          (int)registers.rsi, registers.rdx,
                                          tracees.length, &event,
                                          &callback_result);
#endif
#ifdef SYS_io_uring_setup
          } else if (number == SYS_io_uring_setup) {
            decision = callback_decision(decision_closure, 4, pid, number,
                                         "<unsupported-io-uring>", tracees.length,
                                         &event, &callback_result);
#endif
          }
          event_count++;
          if (decision != 0) {
            if (decision < 0) {
              policy_error = 1;
              denial_code = -decision;
            } else {
              denial_code = decision;
            }
            result_kind = MEMBRANE_POLICY_DENIED;
            kill_tracees(root, &tracees);
            break;
          }
        }
        tracee->entering_syscall = !tracee->entering_syscall;
        if (ptrace_resume(pid, 0) < 0 && errno != ESRCH) {
          native_error = 1;
          kill_tracees(root, &tracees);
          break;
        }
      } else {
        int deliver = 0;
        if (stop_signal != SIGSTOP && stop_signal != SIGTRAP) {
          deliver = stop_signal;
        }
        if (ptrace_resume(pid, deliver) < 0 && errno != ESRCH) {
          native_error = 1;
          kill_tracees(root, &tracees);
          break;
        }
      }
    }
  }

drain:
  while (tracees.length > 0) {
    int status;
    pid_t pid = waitpid(-1, &status, __WALL);
    if (pid < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    if (WIFEXITED(status) || WIFSIGNALED(status)) {
      if (pid == root && !root_status_seen && result_kind < MEMBRANE_TIMED_OUT) {
        root_status_seen = 1;
        if (WIFEXITED(status)) {
          result_kind = MEMBRANE_EXITED;
          result_exit = WEXITSTATUS(status);
        } else {
          result_kind = MEMBRANE_SIGNALED;
          result_signal = WTERMSIG(status);
        }
      }
      tracee_remove(&tracees, pid);
    } else if (WIFSTOPPED(status)) {
      ptrace(PTRACE_CONT, pid, NULL, (void *)(intptr_t)SIGKILL);
      kill(pid, SIGKILL);
    }
  }

  if (native_error) {
    result_kind = MEMBRANE_NATIVE_ERROR;
    if (denial_code == 0) {
      denial_code = 415;
    }
    policy_error = 1;
  } else if (!sandbox_ready &&
             (result_kind == MEMBRANE_EXITED ||
              result_kind == MEMBRANE_SIGNALED)) {
    result_kind = MEMBRANE_NATIVE_ERROR;
    denial_code = 415;
    policy_error = 1;
  } else if (!root_status_seen && result_kind < MEMBRANE_TIMED_OUT) {
    result_kind = MEMBRANE_NATIVE_ERROR;
    denial_code = 415;
    policy_error = 1;
  }

finish:
  {
    int64_t elapsed_us = monotonic_us() - started_us;
    if (elapsed_us < 0) {
      elapsed_us = 0;
    }
    tracee_table_free(&tracees);
    free(executable);
    free(cwd);
    free(user_executable);
    free_string_array(arguments);
    free_string_array(environment);
    result = caml_alloc_tuple(10);
    Store_field(result, 0, Val_int(result_kind));
    Store_field(result, 1, Val_int(result_exit));
    Store_field(result, 2, Val_int(result_signal));
    Store_field(result, 3, caml_copy_int64(elapsed_us));
    Store_field(result, 4, Val_int(event_count));
    Store_field(result, 5, Val_int(denial_code));
    Store_field(result, 6, Val_int(timed_out));
    Store_field(result, 7, Val_int(policy_error));
    Store_field(result, 8, Val_int(landlock_abi));
    Store_field(result, 9, Val_int(sandbox_ready));
    CAMLreturn(result);
  }
#endif
}
