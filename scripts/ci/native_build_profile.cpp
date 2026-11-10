// Observe the actual compiler; preserve its arguments, output bytes and exit.
// Timing arithmetic uses C++23. Darwin sample is diagnostic only.
#include <cerrno>
#include <csignal>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
using Clock = std::chrono::steady_clock;
int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: native-build-profile RECEIPTS COMMAND [ARGS...]\n"; return 2; }
    std::filesystem::create_directories(argv[1]);
    std::ofstream events(std::filesystem::path(argv[1]) / "events.tsv");
    if (!events) return 2;
    events << "event\telapsed_us\n";
    const auto start = Clock::now();
    auto event = [&](const std::string& name) {
        events << name << '\t' << std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count() << '\n'; events.flush();
    };
    int fd[2]; if (pipe(fd) != 0) return 2;
    const pid_t child = fork();
    if (child == -1) return 2;
    if (child == 0) {
        close(fd[0]);
        if (dup2(fd[1], STDOUT_FILENO) < 0 || dup2(fd[1], STDERR_FILENO) < 0) _exit(126);
        close(fd[1]); execvp(argv[2], argv+2); _exit(127);
    }
    close(fd[1]); event("start");
#ifdef __APPLE__
    std::mutex mutex; std::condition_variable cv; bool finished = false;
    std::thread sampler([&] {
        std::unique_lock lock(mutex);
        unsigned index = 0;
        while (!cv.wait_for(lock, std::chrono::seconds(30), [&]{return finished;})) {
            const auto filename = (std::filesystem::path(argv[1]) / ("sample-"+std::to_string(index++)+".txt")).string();
            const auto pid = std::to_string(child);
            lock.unlock();
            const pid_t sample = fork();
            if (sample == 0) {
                int null = open("/dev/null",O_WRONLY);
                if (null < 0 || dup2(null,STDOUT_FILENO) < 0 || dup2(null,STDERR_FILENO) < 0) _exit(126);
                close(null);
                execl("/usr/bin/sample","sample",pid.c_str(),"1","10","-file",filename.c_str(),static_cast<char*>(nullptr)); _exit(127);
            }
            int status = -1;
            if (sample > 0) while (waitpid(sample,&status,0) < 0 && errno == EINTR) {}
            std::ofstream(filename+".status") << status << '\n';
            lock.lock();
        }
    });
#endif
    std::string line; char buffer[8192]; ssize_t count; bool output_failed = false;
    while ((count = read(fd[0],buffer,sizeof buffer)) > 0 || (count < 0 && errno == EINTR)) {
        if (count < 0) continue;
        if (std::fwrite(buffer,1,static_cast<size_t>(count),stdout) != static_cast<size_t>(count)) { output_failed=true;kill(child,SIGTERM);break; }
        std::fflush(stdout);
        for (ssize_t i=0;i<count;++i) {
            if (buffer[i] != '\n') { line += buffer[i]; continue; }
            if (line.starts_with("source: ")) event("source_read");
            else if (line.starts_with("total source: ")) event("imports_and_prelude_done");
            else if (line.starts_with("tokens: ")) event("lex_and_resolution_done");
            else if (line.starts_with("arm64_compile: ")) event("arm64_compile_done");
            else if (line.starts_with("macho_arm64: ")) event("macho_writer");
            line.clear();
        }
    }
    const bool read_failed = count < 0;
    close(fd[0]); int status = 0;
    bool wait_failed = false;
    while (waitpid(child,&status,0) < 0) if (errno != EINTR) { wait_failed=true;break; }
#ifdef __APPLE__
    { std::lock_guard lock(mutex); finished=true; } cv.notify_all(); sampler.join();
#endif
    event("exit");
    if (read_failed || output_failed || wait_failed || !events) return 2;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return WIFSIGNALED(status) ? 128+WTERMSIG(status) : 2;
}
