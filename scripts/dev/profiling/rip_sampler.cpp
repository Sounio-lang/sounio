// rip_sampler -- sampling profiler for symbol-less static ELFs (the Madaros
// compiler as emitted by lean_single has no section headers, so perf/gdb have
// nothing to resolve against, and CI pods cannot open perf_event anyway).
//
// Launches the target as its own child (YAMA ptrace_scope=1 allows tracing a
// descendant), PTRACE_SEIZEs it, and every --interval-ms interrupts it, records
// RIP plus the return addresses on the RBP chain, then resumes it.
// Output: one line per sample, "<t_ms> <rip> <ret1> <ret2> ..." (hex).
//
// build: g++ -std=c++23 -O2 -o rip_sampler rip_sampler.cpp
// usage: rip_sampler --out F [--interval-ms 20] [--depth 16] -- prog args...
#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    std::string out = "samples.txt";
    int interval_ms = 20, depth = 16, i = 1;
    for (; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--") { ++i; break; }
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--interval-ms" && i + 1 < argc) interval_ms = std::atoi(argv[++i]);
        else if (a == "--depth" && i + 1 < argc) depth = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: %s --out F [--interval-ms N] [--depth N] -- prog args\n", argv[0]); return 2; }
    }
    if (i >= argc) { std::fprintf(stderr, "no program given\n"); return 2; }

    int sync[2];
    if (pipe(sync) != 0) { perror("pipe"); return 1; }
    pid_t child = fork();
    if (child == 0) {
        close(sync[1]);
        char c; if (read(sync[0], &c, 1) != 1) _exit(127);   // wait until seized
        execvp(argv[i], &argv[i]);
        perror("execvp"); _exit(127);
    }
    close(sync[0]);
    if (ptrace(PTRACE_SEIZE, child, nullptr, (void*)(long)PTRACE_O_EXITKILL) != 0) { perror("PTRACE_SEIZE"); return 1; }
    if (write(sync[1], "g", 1) != 1) { perror("write"); return 1; }
    close(sync[1]);

    FILE* f = std::fopen(out.c_str(), "w");
    if (!f) { perror("fopen"); return 1; }
    auto t0 = std::chrono::steady_clock::now();
    long samples = 0;
    int status = 0;
    bool profiler_error = false;
    // PTRACE_INTERRUPT fails with ESRCH once the child has exited but is not yet
    // reaped. Reap it so `status` is its real termination status rather than the
    // last ptrace stop; any other failure is the profiler's, not the child's.
    auto on_interrupt_failure = [&]() {
        if (errno == ESRCH) {
            while (waitpid(child, &status, __WALL) >= 0)
                if (WIFEXITED(status) || WIFSIGNALED(status)) return;
        }
        perror("[rip_sampler] PTRACE_INTERRUPT/waitpid");
        profiler_error = true;
    };
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        if (ptrace(PTRACE_INTERRUPT, child, nullptr, nullptr) != 0) { on_interrupt_failure(); goto done; }
        for (;;) {                                   // wait for our interrupt-stop
            pid_t w = waitpid(child, &status, __WALL);
            if (w < 0) { perror("[rip_sampler] waitpid"); profiler_error = true; goto done; }
            if (WIFEXITED(status) || WIFSIGNALED(status)) goto done;
            if (WIFSTOPPED(status) && (status >> 16) == PTRACE_EVENT_STOP) break;
            int sig = WIFSTOPPED(status) ? WSTOPSIG(status) : 0;   // real signal: pass through
            ptrace(PTRACE_CONT, child, nullptr, (void*)(long)sig);
            if (ptrace(PTRACE_INTERRUPT, child, nullptr, nullptr) != 0) { on_interrupt_failure(); goto done; }
        }
        {
            user_regs_struct r{};
            if (ptrace(PTRACE_GETREGS, child, nullptr, &r) == 0) {
                long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(f, "%ld %llx", ms, (unsigned long long)r.rip);
                unsigned long long fp = r.rbp;
                for (int d = 0; d < depth && fp; ++d) {
                    errno = 0;
                    long ret = ptrace(PTRACE_PEEKDATA, child, (void*)(fp + 8), nullptr);
                    long nxt = ptrace(PTRACE_PEEKDATA, child, (void*)fp, nullptr);
                    if (errno) break;
                    std::fprintf(f, " %llx", (unsigned long long)ret);
                    if ((unsigned long long)nxt <= fp) break;             // frames must grow upward
                    fp = (unsigned long long)nxt;
                }
                std::fputc('\n', f);
                ++samples;
            } else {
                // A dropped sample would bias the profile toward whatever is easy
                // to stop; refuse the run instead of publishing a partial one.
                perror("[rip_sampler] PTRACE_GETREGS");
                profiler_error = true;
                goto done;
            }
        }
        ptrace(PTRACE_CONT, child, nullptr, nullptr);
    }
done:
    // A short write (full disk, I/O error) would leave a truncated profile that
    // still looks complete; treat it as a profiler error.
    if (std::ferror(f) | std::fclose(f)) {
        perror("[rip_sampler] writing samples");
        profiler_error = true;
    }
    if (profiler_error) {
        std::fprintf(stderr, "[rip_sampler] %ld samples, PROFILER ERROR (child status unknown)\n", samples);
        return 125;
    }
    int code = WIFEXITED(status) ? WEXITSTATUS(status) : (WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 125);
    std::fprintf(stderr, "[rip_sampler] %ld samples, child exit %d\n", samples, code);
    return code;
}
