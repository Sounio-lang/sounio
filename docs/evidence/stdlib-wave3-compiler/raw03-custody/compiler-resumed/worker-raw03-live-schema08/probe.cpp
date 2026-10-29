#include <elf.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cerrno>
using U=std::uint64_t;using I=std::int64_t;
struct Instr {I op,dst,s1,s2,imm,fp,label,fn,field,flags,bin,un,name,tag,payload,argc;};
static_assert(sizeof(Instr)==128);
struct Region {I slot,generation;auto operator<=>(const Region&)const=default;};
struct Pending {Region region;I index;U out;};
struct Image {
 std::vector<unsigned char> bytes;std::vector<Elf64_Phdr> loads;
 U va(U offset)const{for(auto p:loads)if(offset>=p.p_offset&&offset-p.p_offset<p.p_filesz)return p.p_vaddr+offset-p.p_offset;throw std::runtime_error("offset outside load");}
 bool executable(U a)const{for(auto p:loads)if((p.p_flags&PF_X)&&a>=p.p_vaddr&&a-p.p_vaddr<p.p_memsz)return true;return false;}
};
Image image(const char* path){Image r;std::ifstream f(path,std::ios::binary);r.bytes.assign(std::istreambuf_iterator<char>(f),{});if(r.bytes.size()<sizeof(Elf64_Ehdr))throw std::runtime_error("short image");Elf64_Ehdr h{};std::memcpy(&h,r.bytes.data(),sizeof h);if(std::memcmp(h.e_ident,ELFMAG,SELFMAG)||h.e_type!=ET_EXEC||h.e_machine!=EM_X86_64||h.e_phentsize!=sizeof(Elf64_Phdr)||h.e_phoff>r.bytes.size()||h.e_phnum>(r.bytes.size()-h.e_phoff)/sizeof(Elf64_Phdr))throw std::runtime_error("image ABI");for(unsigned i=0;i<h.e_phnum;++i){Elf64_Phdr p{};std::memcpy(&p,r.bytes.data()+h.e_phoff+i*sizeof p,sizeof p);if(p.p_type==PT_LOAD){if(p.p_offset>r.bytes.size()||p.p_filesz>r.bytes.size()-p.p_offset)throw std::runtime_error("bad segment");r.loads.push_back(p);}}return r;}
bool immediate(const std::vector<unsigned char>& b,U a,U z,I value){
 for(U i=a;i+7<=z;++i){
  if(b[i]==0x48&&b[i+1]==0xc7&&b[i+2]==0xc0){std::int32_t n{};std::memcpy(&n,b.data()+i+3,4);if(n==value)return true;}
  if(i+10<=z&&b[i]==0x48&&b[i+1]==0xb8){I n{};std::memcpy(&n,b.data()+i+2,8);if(n==value)return true;}
 }
 return false;
}
std::pair<U,U> callback(const Image& im){std::vector<U> starts;for(auto p:im.loads)if(p.p_flags&PF_X)for(U i=p.p_offset;i+7<=p.p_offset+p.p_filesz;++i){const unsigned char signature[]{0x55,0x48,0x89,0xe5,0x48,0x81,0xec};if(!std::memcmp(im.bytes.data()+i,signature,sizeof signature))starts.push_back(i);}std::vector<std::pair<U,U>> found;for(std::size_t i=0;i+1<starts.size();++i){bool valid=true;for(I n:{133,134,135,136,137,138,139,140,141,64066,64067,64076,64077,64078,64079})if(!immediate(im.bytes,starts[i],starts[i+1],n)){valid=false;break;}if(valid)found.emplace_back(starts[i],starts[i+1]);}if(found.size()!=1)throw std::runtime_error("callback machine fingerprint not unique: "+std::to_string(found.size()));return found[0];}
void command(const std::vector<std::string>& words,const std::string& output){const pid_t p=fork();if(p<0)throw std::runtime_error("fork objdump");if(p==0){const int fd=open(output.c_str(),O_CREAT|O_TRUNC|O_WRONLY,0644);if(fd<0)_exit(120);dup2(fd,1);close(fd);std::vector<char*> args;for(auto& w:words)args.push_back(const_cast<char*>(w.c_str()));args.push_back(nullptr);execvp(args[0],args.data());_exit(121);}int s=0;if(waitpid(p,&s,0)!=p||!WIFEXITED(s)||WEXITSTATUS(s))throw std::runtime_error("objdump failed");}
bool readmem(pid_t p,U a,void* dst,std::size_t n){if(a<4096||a>~U{0}-n)return false;iovec local{dst,n},remote{reinterpret_cast<void*>(a),n};return process_vm_readv(p,&local,1,&remote,1,0)==static_cast<ssize_t>(n);}
I word(pid_t p,U a){I n{};if(!readmem(p,a,&n,8))throw std::runtime_error("unreadable word");return n;}
bool named(pid_t p,U a,const char* name){const std::size_t n=std::strlen(name);std::array<char,384> text{};I len{};return readmem(p,a,text.data(),text.size())&&readmem(p,a+384,&len,8)&&len==static_cast<I>(n)&&!std::memcmp(text.data(),name,n);}
void byte(pid_t p,U a,unsigned char v){const U aligned=a&~U{7};errno=0;long raw=ptrace(PTRACE_PEEKTEXT,p,reinterpret_cast<void*>(aligned),nullptr);if(raw==-1&&errno)throw std::runtime_error("peek breakpoint");U n=static_cast<U>(raw);const U shift=(a-aligned)*8;n=(n&~(U{255}<<shift))|(static_cast<U>(v)<<shift);if(ptrace(PTRACE_POKETEXT,p,reinterpret_cast<void*>(aligned),reinterpret_cast<void*>(n))<0)throw std::runtime_error("poke breakpoint");}
user_regs_struct regs(pid_t p){user_regs_struct r{};if(ptrace(PTRACE_GETREGS,p,nullptr,&r)<0)throw std::runtime_error("getregs");return r;}
void setregs(pid_t p,user_regs_struct& r){if(ptrace(PTRACE_SETREGS,p,nullptr,&r)<0)throw std::runtime_error("setregs");}
void resume(pid_t p,int request){if(ptrace(static_cast<__ptrace_request>(request),p,nullptr,nullptr)<0)throw std::runtime_error("resume");}
void step(pid_t p){resume(p,PTRACE_SINGLESTEP);int s{};if(waitpid(p,&s,0)!=p||!WIFSTOPPED(s)||WSTOPSIG(s)!=SIGTRAP)throw std::runtime_error("single step");}
void metadata(pid_t p,U module,U function,Region& region,I& count){
 if(!named(p,function,"collision")){throw std::runtime_error("guard function name");}
 region={word(p,function+392),word(p,function+400)};count=word(p,function+408);const I pc=word(p,function+432);if(count!=16||pc!=1||region.slot<0||region.generation<=0)throw std::runtime_error("guard descriptor layout");
 const U table=static_cast<U>(word(p,module));const I total=word(p,module+8);if(total!=35)throw std::runtime_error("module count");const U chunk=static_cast<U>(word(p,table));
const U first=static_cast<U>(word(p,chunk));
const U second=static_cast<U>(word(p,chunk+8));
if(!named(p,first,"product_into")||!named(p,second,"collision")||!named(p,static_cast<U>(word(p,chunk+23*8)),"__rgn_exit_scalar")||!named(p,static_cast<U>(word(p,chunk+16*8)),"__rgn_enter"))throw std::runtime_error("indirect function-entry identity");
const U stride=second-first;
if(stride!=17824)throw std::runtime_error("record layout spacing");
 std::cout<<"LIVE_MODULE fn_count="<<total<<" stride="<<stride<<" region="<<region.slot<<":"<<region.generation<<"\n";
 for(I id:{I{1},I{16},I{23}}){const U f=static_cast<U>(word(p,chunk+static_cast<U>(id)*8));std::cout<<"LIVE_DESCRIPTOR fn="<<id<<" ic="<<word(p,f+408)<<" regs="<<word(p,f+416)<<" labels="<<word(p,f+424)<<" params="<<word(p,f+432)<<" strategy="<<word(p,f+952)<<" float="<<word(p,f+960)<<" sret="<<word(p,f+1392)<<" bss="<<word(p,f+1408)<<" first_ref="<<word(p,f+1416)<<" f128="<<word(p,f+1424)<<" ref_struct="<<word(p,f+1432)<<" profiler="<<word(p,f+1376)<<"\n";}
}
int main(int argc,char** argv){pid_t child=-1;try{
 if(argc!=5){return 2;}
 const auto im=image(argv[1]);const auto [begin,end]=callback(im);const U image_base=im.va(begin)-begin;const std::string dir=argv[4];std::cout<<"CALLBACK_TEXT file_begin="<<begin<<" file_end="<<end<<"\n";
 command({"objdump","-D","-b","binary","-m","i386:x86-64","--start-address="+std::to_string(begin),"--stop-address="+std::to_string(end),argv[1]},dir+"/callback.disasm");
 std::ifstream dis(dir+"/callback.disasm");std::string line;std::regex re(R"(^\s*([0-9a-f]+):\s+e8\s+[0-9a-f ]+\s+call\s+0x([0-9a-f]+))");std::map<U,unsigned char> original;while(std::getline(dis,line)){std::smatch m;if(std::regex_search(line,m,re)){U f=std::stoull(m[1].str(),nullptr,16);if(f<begin||f+5>end||im.bytes[f]!=0xe8)throw std::runtime_error("bad callsite");original[im.va(f)]=0xe8;}}if(original.size()<10||original.size()>1024)throw std::runtime_error("callsite census");std::cout<<"CALLSITES "<<original.size()<<"\n";
 child=fork();if(child<0)throw std::runtime_error("fork trace");if(child==0){const int out=open((dir+"/compile.stdout").c_str(),O_CREAT|O_TRUNC|O_WRONLY,0644);const int err=open((dir+"/compile.stderr").c_str(),O_CREAT|O_TRUNC|O_WRONLY,0644);if(out<0||err<0)_exit(122);dup2(out,1);dup2(err,2);close(out);close(err);setenv("SOUNIO_STDLIB_PATH",argv[3],1);setenv("SOUNIO_LOOP_REGIONS","0",1);setenv("SOUNIO_MADAROS_NOCACHE","1",1);if(ptrace(PTRACE_TRACEME,0,nullptr,nullptr)<0)_exit(123);execl(argv[1],argv[1],"compile",argv[2],"-o",(dir+"/refused.elf").c_str(),nullptr);_exit(124);}
 int status{};if(waitpid(child,&status,0)!=child||!WIFSTOPPED(status))throw std::runtime_error("exec stop");if(ptrace(PTRACE_SETOPTIONS,child,nullptr,reinterpret_cast<void*>(PTRACE_O_EXITKILL))<0)throw std::runtime_error("ptrace options");for(auto [a,b]:original){static_cast<void>(b);byte(child,a,0xcc);}resume(child,PTRACE_CONT);
 std::map<U,Pending> pending;std::map<Region,std::map<I,Instr>> captured;bool live=false;Region selected{};I instruction_count=0;std::size_t events=0;int result=-1;
 while(true){if(waitpid(child,&status,0)!=child)throw std::runtime_error("wait trace");if(WIFEXITED(status)){result=WEXITSTATUS(status);break;}if(WIFSIGNALED(status))throw std::runtime_error("trace child signal "+std::to_string(WTERMSIG(status)));if(!WIFSTOPPED(status)||WSTOPSIG(status)!=SIGTRAP)throw std::runtime_error("unexpected stop");if(++events>100000)throw std::runtime_error("trace budget");auto r=regs(child);const U a=r.rip-1;
  if(auto it=pending.find(a);it!=pending.end()){
   const auto q=it->second;const U f=a-image_base;if(f>=im.bytes.size())throw std::runtime_error("return byte offset");byte(child,a,im.bytes[f]);r.rip=a;setregs(child,r);
   Instr value{};if(r.rax==0&&readmem(child,q.out,&value,sizeof value)&&value.op>=0&&value.op<256&&value.dst>=-1&&value.dst<65536&&value.s1>=-1&&value.s1<65536&&value.s2>=-1&&value.s2<65536&&value.argc>=0&&value.argc<=64){auto& entries=captured[q.region];auto old=entries.find(q.index);if(old!=entries.end()&&std::memcmp(&old->second,&value,sizeof value))throw std::runtime_error("instruction changed during proof");entries[q.index]=value;}
   pending.erase(it);step(child);resume(child,PTRACE_CONT);continue;
  }
  auto it=original.find(a);if(it==original.end())throw std::runtime_error("unknown trap");byte(child,a,it->second);r.rip=a;setregs(child,r);
  bool fn1=false;I id{};if(readmem(child,r.rbp-16,&id,8)&&id==1)fn1=true;
  if(fn1&&named(child,r.rsi,"collision")){
   metadata(child,r.rdi,r.rsi,selected,instruction_count);live=true;std::cout<<"GUARD_CALL at="<<r.rdx<<" depth="<<r.rcx<<" site="<<a<<"\n";
  }
  Region region{};bool candidate=fn1&&readmem(child,r.rdi,&region,sizeof region)&&region.slot>=0&&region.slot<1048576&&region.generation>0&&region.generation<2147483647&&r.rsi<65536&&r.rdx>4096;Instr scratch{};candidate=candidate&&readmem(child,r.rdx,&scratch,sizeof scratch);
  step(child);byte(child,a,it->second==0xe8?0xcc:it->second);
  if(candidate){const U back=a+5;const U f=back-image_base;if(f>=im.bytes.size()||pending.contains(back))throw std::runtime_error("return breakpoint collision");pending.emplace(back,Pending{region,static_cast<I>(r.rsi),r.rdx});byte(child,back,0xcc);}
  resume(child,PTRACE_CONT);
 }
 child=-1;if(result!=1||!live||instruction_count!=16)throw std::runtime_error("baseline or capture missing");const auto found=captured.find(selected);if(found==captured.end()||found->second.size()!=16)throw std::runtime_error("whole function capture incomplete");for(I i=0;i<16;++i){const auto& x=found->second.at(i);std::cout<<"LIVE_IR i="<<i<<" op="<<x.op<<" dst="<<x.dst<<" src="<<x.s1<<","<<x.s2<<" imm="<<x.imm<<" label="<<x.label<<" fn="<<x.fn<<" field="<<x.field<<" flags="<<x.flags<<" argc="<<x.argc<<" bin="<<x.bin<<" unary="<<x.un<<" name="<<x.name<<"\n";}
 std::ifstream log(dir+"/compile.stdout");const std::string text((std::istreambuf_iterator<char>(log)),{});if(text.find("identity=0 capability=0 closure=141 fn=1 at=12")==std::string::npos)throw std::runtime_error("baseline diagnostic changed");std::cout<<"LIVE_SCHEMA_CAPTURE_OK baseline=141 whole_fn_instrs=16 no_compiler_file_patch=1\n";return 0;
 }catch(const std::exception& e){if(child>0){kill(child,SIGKILL);int s{};waitpid(child,&s,0);}std::cerr<<"LIVE_SCHEMA_REFUSED "<<e.what()<<"\n";return 1;}}
