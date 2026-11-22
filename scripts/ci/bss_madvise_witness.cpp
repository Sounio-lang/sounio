#include <cerrno>
#include <csignal>
#include <iostream>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>
// Require actual denied syscalls, not merely an executable that never calls one.
int main(int argc, char** argv) {
    if (argc < 3) return 2;
    const auto child = fork();
    if (child < 0) return 125;
    if (!child) {
        if (ptrace(PTRACE_TRACEME, 0, nullptr, nullptr)) _exit(125);
        raise(SIGSTOP);
        execv(argv[1], argv + 1); _exit(127);
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFSTOPPED(status)) return 125;
    if (ptrace(PTRACE_SETOPTIONS, child, nullptr, PTRACE_O_TRACESYSGOOD)) return 125;
    bool entry = true, pending = false;
    unsigned calls = 0, denied = 0;
    int deliver = 0;
    for (;;) {
        if (ptrace(PTRACE_SYSCALL, child, nullptr, deliver)) return 125;
        if (waitpid(child, &status, 0) != child) return 125;
        if (WIFEXITED(status)) {
            std::cout << "MADVISE_WITNESS calls=" << calls << " enosys=" << denied
                      << " child_exit=" << WEXITSTATUS(status) << '\n';
            return WEXITSTATUS(status) ? WEXITSTATUS(status) :
                   (calls && calls == denied ? 0 : 1);
        }
        if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
        if (!WIFSTOPPED(status)) return 125;
        const auto signal = WSTOPSIG(status);
        deliver = 0;
        if (signal == (SIGTRAP | 0x80)) {
            user_regs_struct regs{};
            if (ptrace(PTRACE_GETREGS, child, nullptr, &regs)) return 125;
            if (entry) {
                pending = regs.orig_rax == SYS_madvise;
                if (pending) ++calls;
            } else if (pending && static_cast<long long>(regs.rax) == -ENOSYS) {
                ++denied;
            }
            entry = !entry;
        } else if (signal != SIGTRAP && signal != SIGSTOP) {
            deliver = signal;
        }
    }
}
