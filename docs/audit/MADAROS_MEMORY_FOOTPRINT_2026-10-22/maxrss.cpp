// maxrss: run a command, print its exit status, wall time and peak RSS (ru_maxrss, KiB) to stderr.
#include <chrono>
#include <cstdio>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: maxrss cmd args...\n"); return 2; }
    auto t0 = std::chrono::steady_clock::now();
    pid_t pid = fork();
    if (pid == 0) { execvp(argv[1], argv + 1); _exit(127); }
    int status = 0; rusage ru{};
    wait4(pid, &status, 0, &ru);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    int rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    std::fprintf(stderr, "maxrss_kib=%ld wall_s=%.2f rc=%d\n", ru.ru_maxrss, s, rc);
    return rc;
}
