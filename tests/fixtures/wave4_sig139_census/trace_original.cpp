#include <cstdint>
#include <cerrno>
#include <csignal>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>
int main(int argc, char** argv) {
  if (argc < 2) return 2;
  pid_t child = fork();
  if (child < 0) return 3;
  if (child == 0) {
    if (ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) < 0) _exit(120);
    raise(SIGSTOP);
    execvp(argv[1], argv + 1);
    _exit(127);
  }
  int status = 0;
  if (waitpid(child, &status, 0) < 0) return 4;
  ptrace(PTRACE_SETOPTIONS, child, nullptr, reinterpret_cast<void*>(PTRACE_O_EXITKILL));
  ptrace(PTRACE_CONT, child, nullptr, nullptr);
  while (waitpid(child, &status, 0) > 0) {
    if (WIFEXITED(status)) {
      std::cout << "exit=" << WEXITSTATUS(status) << "\n";
      return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
      std::cout << "signal_exit=" << WTERMSIG(status) << "\n";
      return 128 + WTERMSIG(status);
    }
    if (!WIFSTOPPED(status)) continue;
    int sig = WSTOPSIG(status);
    if (sig == SIGILL || sig == SIGSEGV || sig == SIGBUS) {
      siginfo_t info{};
      user_regs_struct regs{};
      ptrace(PTRACE_GETSIGINFO, child, nullptr, &info);
      ptrace(PTRACE_GETREGS, child, nullptr, &regs);
      std::cout << "rbp=0x" << std::hex << regs.rbp << " rsi=0x" << regs.rsi << " rdi=0x" << regs.rdi << " r10=0x" << regs.r10 << " r11=0x" << regs.r11 << std::dec << "\n";
      std::cout << "rdx=0x" << std::hex << regs.rdx << std::dec << "\n";
      if(regs.rip==0x5884ade){
        auto peek=[&](std::uint64_t address,const char* label){errno=0;auto word=ptrace(PTRACE_PEEKDATA,child,reinterpret_cast<void*>(address),nullptr);std::cout<<label<<"="<<static_cast<std::int64_t>(word)<<" errno="<<errno<<"\n";};
        peek(regs.rbp-0x20,"saved_report_site");
        peek(regs.rbp-0x938+0x10,"lowerer_current_fn");
        peek(regs.rbp-0x938+0x18,"lowerer_current_func");
        peek(regs.rbp-0x938+0x20,"lowerer_current_func_loaded");
        peek(regs.rbp-0x938+0x58,"lowerer_error_count");
      }
      std::cout << "fault_signal=" << sig << " code=" << info.si_code
                << " address=0x" << std::hex << reinterpret_cast<std::uintptr_t>(info.si_addr)
                << " rip=0x" << regs.rip << " rsp=0x" << regs.rsp
                << " rax=0x" << regs.rax << " rbx=0x" << regs.rbx << std::dec
                << "\nMAPS_BEGIN\n";
      std::ifstream maps("/proc/" + std::to_string(child) + "/maps");
      std::cout << maps.rdbuf() << "MAPS_END\n" << std::flush;
      auto frame = regs.rbp;
      for (int depth=0; depth<32 && frame>0 && (frame%8)==0; ++depth) {
        errno=0;
        auto next=static_cast<std::uint64_t>(ptrace(PTRACE_PEEKDATA,child,reinterpret_cast<void*>(frame),nullptr));
        if(errno!=0)break;
        errno=0;
        auto ret=static_cast<std::uint64_t>(ptrace(PTRACE_PEEKDATA,child,reinterpret_cast<void*>(frame+8),nullptr));
        if(errno!=0)break;
        std::cout<<"FRAME "<<depth<<" rbp=0x"<<std::hex<<frame<<" return=0x"<<ret;
        if(ret>=5){errno=0;auto call=static_cast<std::uint64_t>(ptrace(PTRACE_PEEKDATA,child,reinterpret_cast<void*>(ret-5),nullptr));
          if(errno==0 && (call&255)==232){auto displacement=static_cast<std::int32_t>(call>>8);std::cout<<" callee=0x"<<ret+static_cast<std::int64_t>(displacement);}}
        std::cout<<std::dec<<"\n";
        if(next<=frame || next-frame>67108864)break;
        frame=next;
      }
      std::cout<<std::flush;
      ptrace(PTRACE_KILL, child, nullptr, nullptr);
      waitpid(child, &status, 0);
      return 128 + sig;
    }
    ptrace(PTRACE_CONT, child, nullptr,
           reinterpret_cast<void*>(static_cast<intptr_t>(sig == SIGTRAP ? 0 : sig)));
  }
  return 5;
}
