// SysV callee-saved preservation probe for the Madaros print_f64 builtin.
//
// Usage: print_f64_callee_saved_probe ELF EXPECTED_CALLS
//
// The ELF is a Madaros-compiled program that calls println(f64). It has no
// symbols, so print_f64 is located by its emitted body: the NaN/inf classify
// sequence that opens every print_f64, preceded by the function's
// `push rbp; mov rbp, rsp`. The probe runs the ELF under ptrace, and at each
// entry into print_f64 writes distinct sentinels into rbx, r12, r13, r14 and
// r15, then checks them at the return address. Any change is an ABI
// violation: those five registers belong to the caller.
//
// Prints one line per call and PRINT_F64_CALLEE_SAVED_PASS calls=N when every
// call preserved all five; exits 1 on any clobber, 2 on a harness error.
#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <format>
#include <string>
#include <utility>
#include <vector>

namespace {

template <class... A> void say(std::format_string<A...> fmt, A&&... a) {
    std::fputs((std::format(fmt, std::forward<A>(a)...) + "\n").c_str(), stderr);
}

[[noreturn]] void die(const std::string& why) {
    say("PRINT_F64_CALLEE_SAVED_ERROR: {}", why);
    std::exit(2);
}

struct Load { std::uint64_t off, vaddr, filesz; bool exec; };

std::vector<unsigned char> read_file(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die(std::string("cannot open ") + path);
    return {std::istreambuf_iterator<char>(in), {}};
}

template <class T> T at(const std::vector<unsigned char>& b, std::size_t o) {
    if (o + sizeof(T) > b.size()) die("truncated ELF");
    T v; std::memcpy(&v, b.data() + o, sizeof(T)); return v;
}

std::vector<Load> loads(const std::vector<unsigned char>& b) {
    if (b.size() < 64 || std::memcmp(b.data(), "\x7f" "ELF\x02\x01", 6) != 0) die("not an ELF64 LSB file");
    if (at<std::uint16_t>(b, 18) != 0x3e) die("not x86_64");
    auto phoff = at<std::uint64_t>(b, 32);
    auto phentsize = at<std::uint16_t>(b, 54), phnum = at<std::uint16_t>(b, 56);
    std::vector<Load> out;
    for (unsigned i = 0; i < phnum; ++i) {
        std::size_t p = phoff + std::size_t(i) * phentsize;
        if (at<std::uint32_t>(b, p) != 1) continue;  // PT_LOAD
        out.push_back({at<std::uint64_t>(b, p + 8), at<std::uint64_t>(b, p + 16),
                       at<std::uint64_t>(b, p + 32), (at<std::uint32_t>(b, p + 4) & 1) != 0});
    }
    return out;
}

// mov rax,rdi; movq xmm0,rax; mov rcx,52; shr rax,cl; mov rbx,0x7ff
constexpr std::array<unsigned char, 25> kClassify{
    0x48, 0x89, 0xf8, 0x66, 0x48, 0x0f, 0x6e, 0xc0, 0x48, 0xc7, 0xc1, 0x34, 0x00,
    0x00, 0x00, 0x48, 0xd3, 0xe8, 0x48, 0xc7, 0xc3, 0xff, 0x07, 0x00, 0x00};
constexpr std::array<unsigned char, 4> kPrologue{0x55, 0x48, 0x89, 0xe5};

std::uint64_t find_entry(const std::vector<unsigned char>& b) {
    std::optional<std::uint64_t> found;
    for (const auto& l : loads(b)) {
        if (!l.exec) continue;
        for (std::size_t i = l.off; i + kClassify.size() <= l.off + l.filesz && i + kClassify.size() <= b.size(); ++i) {
            if (std::memcmp(b.data() + i, kClassify.data(), kClassify.size()) != 0) continue;
            std::size_t limit = i > 96 ? i - 96 : l.off;
            std::optional<std::size_t> pro;
            for (std::size_t j = i; j-- > limit && j >= l.off;)
                if (std::memcmp(b.data() + j, kPrologue.data(), kPrologue.size()) == 0) { pro = j; break; }
            if (!pro) continue;
            std::uint64_t va = l.vaddr + (*pro - l.off);
            if (found && *found != va) die("more than one print_f64 body found");
            found = va;
        }
    }
    if (!found) die("print_f64 body not found");
    return *found;
}

constexpr std::array<std::uint64_t, 5> kSentinel{
    0xb0b0b0b0b0b0b0b0ULL, 0x1212121212121212ULL, 0x1313131313131313ULL,
    0x1414141414141414ULL, 0x1515151515151515ULL};
constexpr std::array<const char*, 5> kName{"rbx", "r12", "r13", "r14", "r15"};

long peek(pid_t pid, std::uint64_t a) {
    errno = 0;
    long w = ptrace(PTRACE_PEEKTEXT, pid, a, nullptr);
    if (errno) die("PEEKTEXT failed");
    return w;
}
void poke(pid_t pid, std::uint64_t a, long w) {
    if (ptrace(PTRACE_POKETEXT, pid, a, w) != 0) die("POKETEXT failed");
}
long arm(pid_t pid, std::uint64_t a) {
    long orig = peek(pid, a);
    poke(pid, a, (orig & ~0xffL) | 0xcc);
    return orig;
}
user_regs_struct regs(pid_t pid) {
    user_regs_struct r{};
    if (ptrace(PTRACE_GETREGS, pid, nullptr, &r) != 0) die("GETREGS failed");
    return r;
}
void set_regs(pid_t pid, const user_regs_struct& r) {
    if (ptrace(PTRACE_SETREGS, pid, nullptr, &r) != 0) die("SETREGS failed");
}
int wait_stop(pid_t pid) {
    int st = 0;
    if (waitpid(pid, &st, 0) != pid) die("waitpid failed");
    return st;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) die("usage: print_f64_callee_saved_probe ELF EXPECTED_CALLS");
    const long expected = std::strtol(argv[2], nullptr, 10);
    const std::uint64_t entry = find_entry(read_file(argv[1]));

    pid_t pid = fork();
    if (pid < 0) die("fork failed");
    if (pid == 0) {
        ptrace(PTRACE_TRACEME, 0, nullptr, nullptr);
        execl(argv[1], argv[1], static_cast<char*>(nullptr));
        _exit(127);
    }
    int st = wait_stop(pid);
    if (!WIFSTOPPED(st)) die("child did not stop at exec");

    long entry_orig = arm(pid, entry);
    long calls = 0, bad = 0;
    for (;;) {
        if (ptrace(PTRACE_CONT, pid, nullptr, nullptr) != 0) die("CONT failed");
        st = wait_stop(pid);
        if (WIFEXITED(st) || WIFSIGNALED(st)) break;
        auto r = regs(pid);
        if (WSTOPSIG(st) != SIGTRAP || r.rip - 1 != entry) die("unexpected stop");
        // At entry: put the sentinels in place, remember the return address.
        poke(pid, entry, entry_orig);
        r.rip = entry;
        r.rbx = kSentinel[0]; r.r12 = kSentinel[1]; r.r13 = kSentinel[2];
        r.r14 = kSentinel[3]; r.r15 = kSentinel[4];
        set_regs(pid, r);
        const std::uint64_t ret = static_cast<std::uint64_t>(peek(pid, r.rsp));
        const std::uint64_t entry_rsp = r.rsp;
        long ret_orig = arm(pid, ret);
        if (ptrace(PTRACE_CONT, pid, nullptr, nullptr) != 0) die("CONT failed");
        st = wait_stop(pid);
        if (!WIFSTOPPED(st) || WSTOPSIG(st) != SIGTRAP) die("child did not return from print_f64");
        r = regs(pid);
        if (r.rip - 1 != ret) die("stopped away from the return address");
        poke(pid, ret, ret_orig);
        r.rip = ret;
        set_regs(pid, r);
        ++calls;
        const std::array<std::uint64_t, 5> got{r.rbx, r.r12, r.r13, r.r14, r.r15};
        std::string line = std::format("call {}:", calls);
        bool ok = r.rsp == entry_rsp + 8;
        if (!ok) line += " rsp=WRONG";
        for (std::size_t i = 0; i < got.size(); ++i) {
            bool same = got[i] == kSentinel[i];
            ok = ok && same;
            line += std::format(" {}={}", kName[i], same ? "kept" : std::format("CLOBBERED(0x{:x})", got[i]));
        }
        say("{}", line);
        if (!ok) ++bad;
        entry_orig = arm(pid, entry);
    }
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) die("probed program did not exit 0");
    if (calls != expected) die(std::format("expected {} print_f64 calls, saw {}", expected, calls));
    if (bad) {
        say("PRINT_F64_CALLEE_SAVED_FAIL clobbered_calls={} calls={}", bad, calls);
        return 1;
    }
    say("PRINT_F64_CALLEE_SAVED_PASS calls={}", calls);
    return 0;
}
