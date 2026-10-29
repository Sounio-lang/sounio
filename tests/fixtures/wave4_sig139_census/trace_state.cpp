#include <cstdint>
#include <array>
#include <deque>
#include <sstream>
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

namespace {
bool read_word(pid_t child,std::uint64_t at,std::uint64_t& value){
 errno=0;const auto got=ptrace(PTRACE_PEEKDATA,child,reinterpret_cast<void*>(at),nullptr);
 if(errno!=0){return false;} value=static_cast<std::uint64_t>(got);return true;
}
struct Breakpoint{std::uint64_t pc,original;unsigned char expected;const char* tag;};
std::array<Breakpoint,3> points{{{0x5a656c7,0,0x55,"ARMS_ENTER"},{0x5a65f18,0,0x55,"ARM_ENTER"},{0x5a65871,0,0x48,"ARM_RETURN"}}};
std::deque<std::string> history;
std::string first_invalid;
std::uint64_t events=0;
void context(pid_t child,const char* tag,std::uint64_t base,const user_regs_struct& regs){
 std::array<std::uint64_t,16> words{};bool complete=true;
 for(std::size_t i=0;i<words.size();++i)if(!read_word(child,base+i*8,words[i]))complete=false;
 std::ostringstream s;s<<"STATE "<<events<<" "<<tag<<" base=0x"<<std::hex<<base<<" rsp=0x"<<regs.rsp<<" rdi=0x"<<regs.rdi<<" rsi=0x"<<regs.rsi<<" rax=0x"<<regs.rax<<std::dec<<" complete="<<complete;
 for(std::size_t i=0;i<words.size();++i)s<<" w"<<i<<"=0x"<<std::hex<<words[i]<<std::dec;
 s<<" current_fn_signed="<<static_cast<std::int64_t>(words[2]);
 const bool suspect=!complete||static_cast<std::int64_t>(words[0])<=0||static_cast<std::int64_t>(words[2])<0||static_cast<std::int64_t>(words[3])<=0||static_cast<std::int64_t>(words[13])<=0;
 s<<" suspect="<<suspect;const auto row=s.str();
 if(first_invalid.empty()&&suspect)first_invalid=row;
 history.push_back(row);if(history.size()>48)history.pop_front();
}
void dump_history(){
 std::cout<<"STATE_HISTORY_BEGIN events="<<events<<" first_invalid="<<(first_invalid.empty()?"none":first_invalid)<<"\n";
 for(const auto& row:history)std::cout<<row<<"\n";
 std::cout<<"STATE_HISTORY_END\n";
}
bool patch(pid_t child,const Breakpoint& b,bool enabled){
 auto word=enabled?(b.original&~UINT64_C(255))|UINT64_C(0xcc):b.original;
 return ptrace(PTRACE_POKETEXT,child,reinterpret_cast<void*>(b.pc),reinterpret_cast<void*>(word))==0;
}
}

int main(int argc, char** argv) {
  const bool state=argc>1&&std::string(argv[1])=="--raw03-state";
  if(state){++argv;--argc;}
  bool installed=false;
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
      if(state)dump_history();
      std::cout << "exit=" << WEXITSTATUS(status) << "\n";
      return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
      std::cout << "signal_exit=" << WTERMSIG(status) << "\n";
      return 128 + WTERMSIG(status);
    }
    if (!WIFSTOPPED(status)) continue;
    int sig = WSTOPSIG(status);
    if(state&&sig==SIGTRAP){
      user_regs_struct stop{};ptrace(PTRACE_GETREGS,child,nullptr,&stop);
      if(!installed){
        for(auto& p:points){
          if(!read_word(child,p.pc,p.original)||(p.original&255)!=p.expected||!patch(child,p,true)){
            std::cerr<<"BREAKPOINT_SCHEMA_REFUSED pc=0x"<<std::hex<<p.pc<<"\n";ptrace(PTRACE_KILL,child,nullptr,nullptr);waitpid(child,&status,0);return 98;
          }
        }
        installed=true;ptrace(PTRACE_CONT,child,nullptr,nullptr);continue;
      }
      bool hit=false;
      for(const auto& p:points)if(stop.rip==p.pc+1){
        hit=true;++events;if(events>1000000){std::cerr<<"STATE_EVENT_LIMIT\n";ptrace(PTRACE_KILL,child,nullptr,nullptr);waitpid(child,&status,0);return 99;}
        if(p.pc==0x5a65871){
          context(child,"RETURN_RAX",stop.rax,stop);
          context(child,"RETURN_SRET",stop.rbp-0x1b98,stop);
          context(child,"CALLER_LO_BEFORE_ASSIGN",stop.rbp-0x960,stop);
        }else context(child,p.tag,stop.rsi,stop);
        if(!patch(child,p,false))return 97;
        stop.rip=p.pc;ptrace(PTRACE_SETREGS,child,nullptr,&stop);
        ptrace(PTRACE_SINGLESTEP,child,nullptr,nullptr);
        if(waitpid(child,&status,0)<0||!WIFSTOPPED(status)||WSTOPSIG(status)!=SIGTRAP)return 96;
        if(!patch(child,p,true))return 97;
        ptrace(PTRACE_CONT,child,nullptr,nullptr);break;
      }
      if(hit)continue;
    }
    if (sig == SIGILL || sig == SIGSEGV || sig == SIGBUS) {
      siginfo_t info{};
      user_regs_struct regs{};
      ptrace(PTRACE_GETSIGINFO, child, nullptr, &info);
      ptrace(PTRACE_GETREGS, child, nullptr, &regs);
      std::cout << "rbp=0x" << std::hex << regs.rbp << " rsi=0x" << regs.rsi << " rdi=0x" << regs.rdi << " r10=0x" << regs.r10 << " r11=0x" << regs.r11 << std::dec << "\n";
      std::cout << "rdx=0x" << std::hex << regs.rdx << std::dec << "\n";
      if(state){
        dump_history();
        if(regs.rip==0x59e51d2){std::uint64_t self=0;if(read_word(child,regs.rbp-8,self))context(child,"FAULT_ENUM_SELF",self,regs);}
      }
      if(regs.rip==0x5884ade || (state&&regs.rip==0x588d598)){
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
