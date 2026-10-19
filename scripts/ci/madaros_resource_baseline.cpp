// Linux executable measurements. Build/run only in an allocated Slurm job.
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
struct Case { const char* name; const char* source; const char* marker; };
constexpr std::array<Case, 4> cases{{
    {"minimal", "tests/run-pass/hello.sio", "Hello, Sounio!"},
    {"imports", "tests/run-pass/madaros_flat_arg_dep_reset.sio", "FLAT_ARG_DEP_RESET_OK"},
    {"pbpk", "tests/run-pass/rapamycin_epistemic_pbpk.sio", "ALL 6 TESTS PASSED"},
    {"self", "self-hosted/compiler/main.sio", "Madaros"}
}};

long long micros(timeval t) { return static_cast<long long>(t.tv_sec) * 1000000 + t.tv_usec; }
std::string contents(const fs::path& p) {
    std::ifstream in(p);
    return {std::istreambuf_iterator<char>(in), {}};
}
void sample(pid_t pid, std::ofstream& out, long long elapsed_ms, const fs::path& log,
            std::streamoff& position, std::string& phase) {
    std::ifstream status("/proc/" + std::to_string(pid) + "/status");
    std::array<long long, 5> values{};
    constexpr std::array<const char*, 5> keys{{"VmRSS:", "VmHWM:", "VmSize:", "RssAnon:", "RssFile:"}};
    for (std::string line; std::getline(status, line);) {
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (line.starts_with(keys[i])) {
                std::istringstream fields(line.substr(std::string(keys[i]).size()));
                fields >> values[i];
            }
        }
    }
    // Markers are emitted by the real compiler; this observer does not inject any.
    std::ifstream compiler_log(log);
    compiler_log.seekg(position);
    for (std::string line; std::getline(compiler_log, line);) {
        if (compiler_log.eof()) break; // Keep incomplete lines for the next sample.
        position = compiler_log.tellg();
        const std::string prefix = "compile_stage: ";
        const auto at = line.find(prefix);
        if (at != std::string::npos) phase = line.substr(at + prefix.size());
    }
    out << elapsed_ms;
    for (const auto v : values) out << '\t' << v;
    out << '\t' << phase << '\n';
    out.flush();
}
int measure(const std::vector<std::string>& args, const fs::path& log,
            const fs::path& samples, const std::string& label, int trial,
            std::ofstream& result) {
    const auto start = Clock::now();
    const auto pid = fork();
    if (pid < 0) throw std::runtime_error("fork failed");
    if (pid == 0) {
        setpgid(0, 0);
        const auto fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd < 0 || dup2(fd, STDOUT_FILENO) < 0 || dup2(fd, STDERR_FILENO) < 0) _exit(126);
        close(fd);
        std::vector<char*> argv;
        for (const auto& arg : args) argv.push_back(const_cast<char*>(arg.c_str()));
        argv.push_back(nullptr);
        execv(argv.front(), argv.data());
        _exit(127);
    }
    setpgid(pid, pid);
    std::ofstream trace(samples);
    trace << "elapsed_ms\trss_kib\thwm_kib\tvsize_kib\tanon_kib\tfile_kib\tcompiler_phase\n";
    int status = 0;
    rusage usage{};
    std::atomic<bool> done{false}, timed_out{false};
    std::mutex mutex;
    std::condition_variable wake;
    std::thread observer([&] {
        std::streamoff position = 0;
        std::string phase = "unmarked";
        while (!done.load()) {
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
            sample(pid, trace, ms, log, position, phase);
            if (ms >= 5400000 && !timed_out.exchange(true)) kill(-pid, SIGKILL);
            std::unique_lock lock(mutex);
            wake.wait_for(lock, std::chrono::milliseconds(500), [&] { return done.load(); });
        }
    });
    pid_t waited;
    do { waited = wait4(pid, &status, 0, &usage); } while (waited < 0 && errno == EINTR);
    const auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
    done = true;
    wake.notify_all();
    observer.join();
    if (waited != pid) throw std::runtime_error("wait4 failed");
    const auto rc = timed_out.load() ? 124 : WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    result << label << '\t' << trial << '\t' << rc << '\t' << wall_us << '\t'
           << micros(usage.ru_utime) << '\t' << micros(usage.ru_stime) << '\t'
           << usage.ru_maxrss << '\t' << usage.ru_majflt << '\t' << usage.ru_minflt << '\n';
    result.flush();
    return rc;
}
int main(int argc, char** argv) try {
    if (argc != 6) throw std::runtime_error("HOST ROOT OUT small|self REPEATS");
    const auto host = fs::absolute(argv[1]);
    const auto root = fs::absolute(argv[2]);
    const auto out = fs::absolute(argv[3]);
    const std::string mode = argv[4];
    const auto repeats = std::stoi(argv[5]);
    if ((mode != "small" && mode != "self") || repeats < 1 || repeats > 5)
        throw std::runtime_error("invalid mode/repeats");
    if (fs::exists(out)) throw std::runtime_error("output already exists; preserve evidence");
    fs::create_directories(out);
    fs::current_path(root);
    std::ofstream result(out / "measurements.tsv");
    result << "case\ttrial\texit\twall_us\tuser_us\tsystem_us\tmaxrss_kib\tmajor_faults\tminor_faults\n";
    std::ofstream identity(out / "environment.txt");
    identity << "host=" << host << "\nroot=" << root << "\nmode=" << mode
             << "\npeak_rss_unit=KiB (Linux wait4)\n" << contents("/proc/meminfo")
             << "\nswaps\n" << contents("/proc/swaps");
    int failures = 0;
    for (const auto& c : cases) {
        if ((mode == "self") != (std::string(c.name) == "self")) continue;
        if (!fs::is_regular_file(root / c.source)) throw std::runtime_error("missing source");
        for (int trial = 1; trial <= repeats; ++trial) {
            const auto stem = std::string(c.name) + "-" + std::to_string(trial);
            const auto executable = out / (stem + ".elf");
            const auto rc = measure({host.string(), "build", (root / c.source).string(),
                "--target", "x86_64-linux", "-o", executable.string()}, out / (stem + ".build.log"),
                out / (stem + ".build.memory.tsv"), std::string(c.name) + ".compile", trial, result);
            if (rc != 0 || !fs::is_regular_file(executable)) { ++failures; continue; }
            fs::permissions(executable, fs::perms::owner_exec, fs::perm_options::add);
            std::vector<std::string> run{executable.string()};
            if (std::string(c.name) == "self") run.push_back("--version");
            const auto runlog = out / (stem + ".run.log");
            const auto run_rc = measure(run, runlog, out / (stem + ".run.memory.tsv"),
                std::string(c.name) + ".execute", trial, result);
            const auto output = contents(runlog);
            if (run_rc != 0 || output.find(c.marker) == std::string::npos) ++failures;
        }
    }
    std::cout << "resource_baseline failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
