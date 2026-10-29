#include <elf.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using Word=std::int64_t;
using Callback=Word(*)(Word);
extern "C" Word sounio_native_callback_thunk(Callback,Word);
struct Image {std::uint64_t zero{},plus{},region{},exit{};};
std::uint64_t down(std::uint64_t n){return n&~std::uint64_t{4095};}
std::uint64_t up(std::uint64_t n){if(n>std::numeric_limits<std::uint64_t>::max()-4095)throw std::runtime_error("overflow");return down(n+4095);}
Image load(const char* elf,const char* addresses){
 std::ifstream stream(elf,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(stream)),{});
 if(bytes.size()<sizeof(Elf64_Ehdr))throw std::runtime_error("short ELF");
 Elf64_Ehdr header{};std::memcpy(&header,bytes.data(),sizeof header);
 if(std::memcmp(header.e_ident,ELFMAG,SELFMAG)!=0||header.e_ident[EI_CLASS]!=ELFCLASS64||header.e_machine!=EM_X86_64||header.e_type!=ET_EXEC||header.e_phentsize!=sizeof(Elf64_Phdr))throw std::runtime_error("unsupported ELF");
 if(header.e_phoff>bytes.size()||header.e_phnum>(bytes.size()-header.e_phoff)/sizeof(Elf64_Phdr))throw std::runtime_error("bad program table");
 std::vector<Elf64_Phdr> segments;std::uint64_t low=std::numeric_limits<std::uint64_t>::max(),high=0;
 for(unsigned i=0;i<header.e_phnum;++i){Elf64_Phdr p{};std::memcpy(&p,bytes.data()+header.e_phoff+i*sizeof p,sizeof p);if(p.p_type!=PT_LOAD)continue;
  if(p.p_filesz>p.p_memsz||p.p_offset>bytes.size()||p.p_filesz>bytes.size()-p.p_offset||p.p_vaddr>std::numeric_limits<std::uint64_t>::max()-p.p_memsz)throw std::runtime_error("bad load");
  segments.push_back(p);low=std::min(low,down(p.p_vaddr));high=std::max(high,up(p.p_vaddr+p.p_memsz));
 }
 if(segments.empty()||high<=low||high-low>(std::uint64_t{8}<<30))throw std::runtime_error("bad mapping extent");
 void* mapped=mmap(reinterpret_cast<void*>(low),high-low,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0);
 if(mapped==MAP_FAILED||reinterpret_cast<std::uintptr_t>(mapped)!=low)throw std::runtime_error("fixed mapping unavailable");
 for(const auto& p:segments)if(p.p_filesz)std::memcpy(reinterpret_cast<void*>(p.p_vaddr),bytes.data()+p.p_offset,p.p_filesz);
 for(const auto& p:segments){const int prot=((p.p_flags&PF_R)?PROT_READ:0)|((p.p_flags&PF_W)?PROT_WRITE:0)|((p.p_flags&PF_X)?PROT_EXEC:0);if(mprotect(reinterpret_cast<void*>(down(p.p_vaddr)),up(p.p_vaddr+p.p_memsz)-down(p.p_vaddr),prot)!=0)throw std::runtime_error("segment protection");}
 std::size_t denied=0;
 for(const auto& p:segments)if(p.p_flags&PF_W){if(mprotect(reinterpret_cast<void*>(down(p.p_vaddr)),up(p.p_vaddr+p.p_memsz)-down(p.p_vaddr),PROT_NONE)!=0)throw std::runtime_error("deny writable image");++denied;}
 if(!denied)throw std::runtime_error("no writable control");
 Image result;std::ifstream input(addresses);std::string key;std::uint64_t value{};
 while(input>>key>>value){if(key=="ZERO_ADDR")result.zero=value;else if(key=="PLUS_ADDR")result.plus=value;else if(key=="REGION_ADDR")result.region=value;else throw std::runtime_error("unknown address");}
 for(const auto address:{result.zero,result.plus,result.region}){bool executable=false;for(const auto& p:segments)if((p.p_flags&PF_X)&&address>=p.p_vaddr&&address<p.p_vaddr+p.p_memsz)executable=true;if(!executable)throw std::runtime_error("address outside executable image");}
 const auto* code=reinterpret_cast<const unsigned char*>(result.zero);
 std::size_t exits=0;
 for(std::size_t i=0;i+5<0x91;++i)if(code[i]==0xe8){
  std::int32_t rel{};std::memcpy(&rel,code+i+1,sizeof rel);
  const auto target=static_cast<std::uint64_t>(static_cast<std::int64_t>(result.zero+i+5)+rel);
  bool executable=false;for(const auto& p:segments)if((p.p_flags&PF_X)&&p.p_memsz>=10&&target>=p.p_vaddr&&target-p.p_vaddr<=p.p_memsz-10)executable=true;
  if(!executable)continue;
  const unsigned char prefix[]{0x48,0x85,0xff,0x0f,0x84};
  if(std::memcmp(reinterpret_cast<const void*>(target),prefix,sizeof prefix)!=0)continue;
  result.exit=target;++exits;
 }
 if(exits!=1)throw std::runtime_error("cleanup call target is not unique");
 return result;
}
int child(const char* elf,const char* addresses,int region){
 const Image image=load(elf,addresses);
 if(region==2){static_cast<void>(sounio_native_callback_thunk(reinterpret_cast<Callback>(image.exit),1));return 3;}
 if(region==1){const auto callback=reinterpret_cast<Callback>(image.region);return sounio_native_callback_thunk(callback,20)==1?0:3;}
 static_cast<void>(sounio_native_callback_thunk(reinterpret_cast<Callback>(image.exit),0));
 const auto zero=reinterpret_cast<Callback>(image.zero);const auto plus=reinterpret_cast<Callback>(image.plus);
 for(const Word x:{Word{-10},Word{0},Word{20},Word{100}})if(sounio_native_callback_thunk(zero,x)!=0||sounio_native_callback_thunk(plus,x)!=x+7)return 4;
 return 0;
}
int run(const char* elf,const char* addresses,int region){const pid_t pid=fork();if(pid<0)throw std::runtime_error("fork");if(pid==0){try{_exit(child(elf,addresses,region));}catch(const std::exception& e){std::cerr<<e.what()<<'\n';_exit(5);}}int status=0;if(waitpid(pid,&status,0)!=pid)throw std::runtime_error("wait");std::cout<<"RUNTIME_CHILD region="<<region<<" status="<<status<<"\n";if(region)return WIFSIGNALED(status)&&WTERMSIG(status)==SIGSEGV?0:6;return WIFEXITED(status)&&WEXITSTATUS(status)==0?0:7;}
}
int main(int argc,char** argv){try{if(argc!=3)return 2;if(run(argv[1],argv[2],false)!=0||run(argv[1],argv[2],true)!=0||run(argv[1],argv[2],2)!=0)return 1;std::cout<<"ZERO_REGION_RUNTIME_GUARD_OK scalar_calls=8 direct_exit_token0_calls=1 nonzero_direct_exit=SIGSEGV zero_envelope_bypassed=1 real_region_control=SIGSEGV writable_segments=PROT_NONE\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
