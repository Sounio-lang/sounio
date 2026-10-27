// Linux x86-64 test harness: inject one ENOMEM into an armed child's mmap.
// No attachment to existing processes, resource exhaustion, or global allocator changes.
// Interface reference: https://man7.org/linux/man-pages/man2/ptrace.2.html
#include <sys/ptrace.h>
#include <linux/ptrace.h>
#include <linux/audit.h>
#include <sys/user.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#if !defined(__linux__) || !defined(__x86_64__)
#error This test harness requires the native Linux x86-64 execution lane.
#endif
static constexpr std::string_view arm_marker="HEAP_VEC_INJECT_ARM\n";
static constexpr std::string_view done_marker="HEAP_VEC_INJECT_DONE\n";
static void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
static long trace(unsigned request,pid_t pid,std::uintptr_t address=0,std::uintptr_t data=0){
 errno=0;const auto result=ptrace(static_cast<__ptrace_request>(request),pid,reinterpret_cast<void*>(address),reinterpret_cast<void*>(data));
 if(result==-1&&errno!=0)throw std::runtime_error("ptrace request "+std::to_string(request)+": "+std::strerror(errno));
 return result;
}
static void wait_for(pid_t pid,int& status){
 for(;;){const auto got=waitpid(pid,&status,0);if(got==pid)return;if(got<0&&errno==EINTR)continue;throw std::runtime_error("waitpid");}
}
struct Child {
 pid_t pid=-1;
 ~Child(){if(pid>0){kill(pid,SIGKILL);int status;while(waitpid(pid,&status,0)<0&&errno==EINTR){}}}
};
static std::string bytes(pid_t pid,std::uintptr_t address,std::size_t count){
 need(count<=128,"marker size");std::string out;out.reserve(count);
 std::uintptr_t last=std::numeric_limits<std::uintptr_t>::max();unsigned long word=0;
 for(std::size_t i=0;i<count;++i){
  const auto at=address+i,aligned=at&~std::uintptr_t(sizeof(long)-1);
  if(aligned!=last){word=static_cast<unsigned long>(trace(PTRACE_PEEKDATA,pid,aligned));last=aligned;}
  out.push_back(static_cast<char>((word>>(8*(at-aligned)))&255));
 }
 return out;
}
static int execute(const std::string& program,const std::vector<std::string>& arguments){
 alarm(30);Child child;const auto parent=getpid();child.pid=fork();need(child.pid>=0,"fork");
 if(child.pid==0){
  if(prctl(PR_SET_PDEATHSIG,SIGKILL)!=0||getppid()!=parent)_exit(125);
  if(ptrace(static_cast<__ptrace_request>(PTRACE_TRACEME),0,nullptr,nullptr)<0)_exit(126);
  raise(SIGSTOP);
  std::vector<char*> argv;argv.push_back(const_cast<char*>(program.c_str()));
  for(const auto& arg:arguments){argv.push_back(const_cast<char*>(arg.c_str()));}
  argv.push_back(nullptr);
  execv(program.c_str(),argv.data());_exit(127);
 }
 int status=0;wait_for(child.pid,status);
 if(WIFEXITED(status)||WIFSIGNALED(status)){child.pid=-1;}
 need(WIFSTOPPED(status)&&WSTOPSIG(status)==SIGSTOP,"initial trace stop");
 trace(PTRACE_SETOPTIONS,child.pid,0,PTRACE_O_TRACESYSGOOD|PTRACE_O_EXITKILL|PTRACE_O_TRACEEXEC);
 unsigned armed=0,done=0,injected=0,executed=0;bool pending=false;
 trace(PTRACE_SYSCALL,child.pid);
 for(;;){
  wait_for(child.pid,status);
  if(WIFEXITED(status)){
   const auto code=WEXITSTATUS(status);child.pid=-1;
   need(code==0,"child exit "+std::to_string(code));
   need(armed==1&&done==1&&injected==1&&executed==1&&!pending,"missing exact arm/injection/done protocol");
   alarm(0);std::cout<<"HEAP_VEC_ALLOC_FAULT_INJECTED_ONCE errno=ENOMEM bytes=4096\n";return 0;
  }
  if(WIFSIGNALED(status)){child.pid=-1;throw std::runtime_error("child signal "+std::to_string(WTERMSIG(status)));}
  need(WIFSTOPPED(status),"unexpected child wait state");
  const int signal=WSTOPSIG(status);
  if(signal==(SIGTRAP|0x80)){
   ptrace_syscall_info info{};const auto length=trace(PTRACE_GET_SYSCALL_INFO,child.pid,sizeof(info),reinterpret_cast<std::uintptr_t>(&info));
   need(length>=static_cast<long>(offsetof(ptrace_syscall_info,entry))&&info.arch==AUDIT_ARCH_X86_64,"syscall info ABI");
   if(info.op==PTRACE_SYSCALL_INFO_ENTRY){
    need(length>=static_cast<long>(offsetof(ptrace_syscall_info,entry.args)+sizeof(info.entry.args)),"truncated syscall entry");
    if(info.entry.nr==SYS_write&&info.entry.args[0]==STDOUT_FILENO){
     const auto n=info.entry.args[2];
     if(n==arm_marker.size()||n==done_marker.size()){
      const auto text=bytes(child.pid,info.entry.args[1],static_cast<std::size_t>(n));
      if(text==arm_marker){need(armed==0&&done==0,"duplicate/late arm marker");++armed;}
      if(text==done_marker){need(armed==1&&injected==1&&done==0&&!pending,"done without completed injection");++done;}
     }
    }
    const auto flags=info.entry.args[3];
    const bool target=info.entry.nr==SYS_mmap&&info.entry.args[0]==0&&info.entry.args[1]==4096&&
     info.entry.args[2]==(PROT_READ|PROT_WRITE)&&
     (flags&(MAP_PRIVATE|MAP_ANONYMOUS))==(MAP_PRIVATE|MAP_ANONYMOUS)&&!(flags&MAP_FIXED);
    if(target&&armed==1&&done==0&&injected==0){
     user_regs_struct regs{};trace(PTRACE_GETREGS,child.pid,0,reinterpret_cast<std::uintptr_t>(&regs));
     regs.orig_rax=std::numeric_limits<unsigned long long>::max();
     trace(PTRACE_SETREGS,child.pid,0,reinterpret_cast<std::uintptr_t>(&regs));pending=true;++injected;
    }
   }else if(info.op==PTRACE_SYSCALL_INFO_EXIT&&pending){
    need(info.exit.is_error&&info.exit.rval==-ENOSYS,"suppressed mmap did not return ENOSYS");
    user_regs_struct regs{};trace(PTRACE_GETREGS,child.pid,0,reinterpret_cast<std::uintptr_t>(&regs));
    regs.rax=static_cast<unsigned long long>(-ENOMEM);
    trace(PTRACE_SETREGS,child.pid,0,reinterpret_cast<std::uintptr_t>(&regs));pending=false;
   }else need(info.op==PTRACE_SYSCALL_INFO_EXIT,"unexpected syscall stop kind");
  }else {
   need(signal==SIGTRAP&&static_cast<unsigned>(status>>16)==PTRACE_EVENT_EXEC&&executed==0,"unexpected child signal "+std::to_string(signal));++executed;
  }
  trace(PTRACE_SYSCALL,child.pid);
 }
}
static void marker(std::string_view text){
 const auto n=write(STDOUT_FILENO,text.data(),text.size());
 if(n!=static_cast<ssize_t>(text.size()))_exit(90);
}
static int fixture(const std::string& mode){
 const auto old=static_cast<unsigned char*>(mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
 if(old==MAP_FAILED)return 10;
 for(std::size_t i=0;i<4096;++i)old[i]=static_cast<unsigned char>(i%251);
 if(mode!="no-arm")marker(arm_marker);
 const std::size_t count=mode=="wrong-size"?8192:4096;
 errno=0;const auto next=mmap(nullptr,count,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 const bool failed=next==MAP_FAILED&&errno==ENOMEM;
 if(mode=="checked"||mode=="bad-exit"){
  if(!failed){if(next!=MAP_FAILED)munmap(next,count);munmap(old,4096);return 11;}
  for(std::size_t i=0;i<4096;++i)if(old[i]!=static_cast<unsigned char>(i%251)){munmap(old,4096);return 12;}
  const auto retry=mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if(retry==MAP_FAILED){munmap(old,4096);return 13;}
  for(std::size_t i=0;i<4096;++i)if(old[i]!=static_cast<unsigned char>(i%251)){munmap(retry,4096);munmap(old,4096);return 15;}
  if(munmap(retry,4096)!=0||munmap(old,4096)!=0)return 14;
  marker(done_marker);return mode=="bad-exit"?7:0;
 }
 if(next!=MAP_FAILED){munmap(next,count);}
 munmap(old,4096);marker(done_marker);return 0;
}
int main(int argc,char**argv){
 try{
  need(argc>=2,"mode");const std::string mode=argv[1];
  if(mode=="--child"){need(argc==3,"child mode");return fixture(argv[2]);}
  if(mode=="--run"){need(argc>=3,"--run executable [args]");std::vector<std::string> args;for(int i=3;i<argc;++i)args.emplace_back(argv[i]);return execute(argv[2],args);}
  const std::string self="/proc/self/exe";
  if(mode=="--selftest")return execute(self,{"--child","checked"});
  if(mode=="--selftest-no-arm")return execute(self,{"--child","no-arm"});
  if(mode=="--selftest-wrong-size")return execute(self,{"--child","wrong-size"});
  if(mode=="--selftest-bad-exit")return execute(self,{"--child","bad-exit"});
  throw std::runtime_error("unknown mode");
 }catch(const std::exception&e){std::cerr<<"HEAP_VEC_FAULT_HARNESS_REFUSED "<<e.what()<<"\n";return 1;}
}
