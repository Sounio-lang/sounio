#include <sys/ptrace.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
long peek(pid_t p,std::uint64_t a){errno=0;long x=ptrace(PTRACE_PEEKDATA,p,reinterpret_cast<void*>(a),nullptr);if(errno)throw errno;return x;}
int main(int argc,char**argv){if(argc!=2)return 64;pid_t p=fork();if(p<0)return 65;if(!p){if(ptrace(PTRACE_TRACEME,0,nullptr,nullptr)<0)_exit(125);raise(SIGSTOP);execl(argv[1],argv[1],static_cast<char*>(nullptr));_exit(126);}int status=0;if(waitpid(p,&status,0)!=p)return 66;if(!WIFSTOPPED(status)){std::cout<<"TRACE_INIT_EXIT "<<status<<'\n';return 1;}ptrace(PTRACE_SETOPTIONS,p,nullptr,PTRACE_O_EXITKILL);ptrace(PTRACE_CONT,p,nullptr,nullptr);bool observed=false;while(waitpid(p,&status,0)==p){if(WIFEXITED(status)){std::cout<<"TRACE_CHILD_EXIT "<<WEXITSTATUS(status)<<'\n';return observed?0:1;}if(WIFSIGNALED(status)){std::cout<<"TRACE_CHILD_SIGNAL "<<WTERMSIG(status)<<'\n';return observed?0:1;}if(!WIFSTOPPED(status))continue;int sig=WSTOPSIG(status);if(sig==SIGILL||sig==SIGSEGV){user_regs_struct r{};if(ptrace(PTRACE_GETREGS,p,nullptr,&r)<0)return 67;std::cout<<std::hex<<"TRACE_STOP signal="<<std::dec<<sig<<std::hex<<" rip=0x"<<r.rip<<" rsp=0x"<<r.rsp<<" rbp=0x"<<r.rbp<<" rax=0x"<<r.rax<<" rdi=0x"<<r.rdi<<" rsi=0x"<<r.rsi<<" rdx=0x"<<r.rdx<<'\n';try{std::cout<<"TRACE_OPCODE 0x"<<static_cast<std::uint64_t>(peek(p,r.rip))<<'\n';std::uint64_t bp=r.rbp;for(int i=0;i<10&&bp;++i){auto prev=static_cast<std::uint64_t>(peek(p,bp)),ret=static_cast<std::uint64_t>(peek(p,bp+8));std::cout<<"TRACE_FRAME "<<std::dec<<i<<std::hex<<" rbp=0x"<<bp<<" return=0x"<<ret<<'\n';if(prev<=bp||prev-bp>0x10000000)break;bp=prev;}for(int i=0;i<16;++i)std::cout<<"TRACE_STACK "<<std::dec<<i<<std::hex<<" value=0x"<<static_cast<std::uint64_t>(peek(p,r.rsp+8*i))<<'\n';}catch(int e){std::cout<<"TRACE_PEEK_ERROR "<<std::dec<<e<<' '<<std::strerror(e)<<'\n';}observed=true;}int deliver=sig==SIGTRAP?0:sig;ptrace(PTRACE_CONT,p,nullptr,deliver);}return 68;}
