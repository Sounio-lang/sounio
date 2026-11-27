// POSIX helper for the native Linux entry-trampoline exit regression.
#include <cerrno>
#include <pthread.h>
#include <unistd.h>

namespace {
int ready_pipe[2];
int block_pipe[2];

ssize_t read_byte(int fd, char* byte) {
    ssize_t result;
    do { result = read(fd, byte, 1); } while (result < 0 && errno == EINTR);
    return result;
}

void* blocked(void*) {
    const char ready = 'R';
    ssize_t result;
    do { result = write(ready_pipe[1], &ready, 1); } while (result < 0 && errno == EINTR);
    if (result != 1) return nullptr;
    char unused;
    // The write end is deliberately kept open and never written to.
    (void)read_byte(block_pipe[0], &unused);
    return nullptr;
}
}

extern "C" int start_blocked_thread() {
    if (pipe(ready_pipe) != 0 || pipe(block_pipe) != 0) return 1;
    pthread_t worker;
    if (pthread_create(&worker, nullptr, blocked, nullptr) != 0) return 2;
    char ready;
    if (read_byte(ready_pipe[0], &ready) != 1 || ready != 'R') return 3;
    return 0;
}
