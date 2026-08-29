/* Transitory material parity for the frozen Sounio V12 authority plan. */

typedef unsigned int __u32;
typedef unsigned long long __u64;

struct kuid_t {
  __u32 val;
};

struct cred {
  struct kuid_t uid;
  struct kuid_t gid;
  struct kuid_t suid;
  struct kuid_t sgid;
  struct kuid_t euid;
  struct kuid_t egid;
  struct kuid_t fsuid;
  struct kuid_t fsgid;
} __attribute__((preserve_access_index));

struct task_struct {
  const struct cred *real_cred;
  const struct cred *cred;
} __attribute__((preserve_access_index));

#define SEC(name) __attribute__((section(name), used))
#define EACCES 13

static long (*bpf_probe_read_kernel)(void *destination, __u32 size,
                                     const void *source) = (void *)113;
static void *(*bpf_get_current_task_btf)(void) = (void *)158;

struct uid_vector {
  __u32 uid;
  __u32 euid;
  __u32 suid;
  __u32 fsuid;
};

#define CORE_READ(destination, source)                                         \
  bpf_probe_read_kernel(&(destination), sizeof(destination),                   \
                        __builtin_preserve_access_index(&(source)))

static __attribute__((always_inline)) int read_uid_vector(
    const struct cred *source, struct uid_vector *destination) {
  if (!source || !destination) return -EACCES;
  if (CORE_READ(destination->uid, source->uid.val) != 0) return -EACCES;
  if (CORE_READ(destination->euid, source->euid.val) != 0) return -EACCES;
  if (CORE_READ(destination->suid, source->suid.val) != 0) return -EACCES;
  if (CORE_READ(destination->fsuid, source->fsuid.val) != 0) return -EACCES;
  return 0;
}

static __attribute__((always_inline)) int same_uid_vector(
    const struct cred *left, const struct cred *right) {
  struct uid_vector left_ids = {};
  struct uid_vector right_ids = {};
  if (read_uid_vector(left, &left_ids) != 0 ||
      read_uid_vector(right, &right_ids) != 0) {
    return -EACCES;
  }
  return left_ids.uid == right_ids.uid && left_ids.euid == right_ids.euid &&
         left_ids.suid == right_ids.suid && left_ids.fsuid == right_ids.fsuid;
}

static __attribute__((always_inline)) int task_cred(
    const struct task_struct *task, const struct cred **result) {
  if (!task || !result) return -EACCES;
  if (CORE_READ(*result, task->cred) != 0 || !*result) return -EACCES;
  return 0;
}

SEC("lsm/task_kill")
int loom_v12_task_kill(__u64 *context) {
  const struct task_struct *target = (const struct task_struct *)context[0];
  const struct cred *caller = (const struct cred *)context[3];
  const int prior = (int)context[4];
  const struct cred *target_cred = (void *)0;
  if (prior != 0) return prior;
  if (!target || !caller) return -EACCES;
  const struct task_struct *current =
      (const struct task_struct *)bpf_get_current_task_btf();
  if (!current) return -EACCES;
  if (current == target) return 0;
  if (task_cred(target, &target_cred) != 0) return -EACCES;
  const int same = same_uid_vector(caller, target_cred);
  if (same < 0) return same;
  return same ? -EACCES : 0;
}

SEC("lsm/ptrace_access_check")
int loom_v12_ptrace_access_check(__u64 *context) {
  const struct task_struct *target = (const struct task_struct *)context[0];
  const int prior = (int)context[2];
  const struct task_struct *current =
      (const struct task_struct *)bpf_get_current_task_btf();
  const struct cred *caller_cred = (void *)0;
  const struct cred *target_cred = (void *)0;
  if (prior != 0) return prior;
  if (!current || !target) return -EACCES;
  if (current == target) return 0;
  if (task_cred(current, &caller_cred) != 0 ||
      task_cred(target, &target_cred) != 0) {
    return -EACCES;
  }
  const int same = same_uid_vector(caller_cred, target_cred);
  if (same < 0) return same;
  return same ? -EACCES : 0;
}

SEC("lsm/task_prlimit")
int loom_v12_task_prlimit(__u64 *context) {
  const struct cred *caller = (const struct cred *)context[0];
  const struct cred *target = (const struct cred *)context[1];
  const int prior = (int)context[3];
  if (prior != 0) return prior;
  if (!caller || !target) return -EACCES;
  if (caller == target) return 0;
  const int same = same_uid_vector(caller, target);
  if (same < 0) return same;
  return same ? -EACCES : 0;
}

char LICENSE[] SEC("license") = "GPL";
