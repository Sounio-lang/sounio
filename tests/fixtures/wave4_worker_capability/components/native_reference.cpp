#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <elf.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <csignal>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
using Word=std::uintptr_t;
#include "runtime_thunk.inc"
extern "C" std::int64_t worker_stack_probe(std::int64_t);
extern "C" std::int64_t worker_clobber_probe(std::int64_t);
asm(".text\n.globl worker_stack_probe\nworker_stack_probe:\nmovq %rsp,%rax\nandq $15,%rax\nret\n"
    ".globl worker_clobber_probe\nworker_clobber_probe:\nmovq $1,%rbx\nmovq $2,%r12\nmovq $3,%r13\nmovq $4,%r14\nmovq $5,%r15\nmovq $6,%rbp\nmovq %rdi,%rax\nret\n");
struct Elf {
 std::vector<unsigned char> bytes;
 std::map<std::string,Word> symbols;
 Elf(const char*path,const char*offset_path){
  std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("missing ELF");bytes={std::istreambuf_iterator<char>(input),{}};
  if(bytes.size()<sizeof(Elf64_Ehdr))throw std::runtime_error("ELF header");
  const auto h=read<Elf64_Ehdr>(0);
  if(std::memcmp(h.e_ident,ELFMAG,SELFMAG)||h.e_ident[EI_CLASS]!=ELFCLASS64||h.e_ident[EI_DATA]!=ELFDATA2LSB||h.e_machine!=EM_X86_64||h.e_type!=ET_EXEC)throw std::runtime_error("ELF ABI");
  const Word page=static_cast<Word>(sysconf(_SC_PAGESIZE));
  Word code_base=0,code_end=0;
  for(unsigned n=0;n<h.e_phnum;++n){
   const auto p=read<Elf64_Phdr>(h.e_phoff+n*h.e_phentsize);if(p.p_type!=PT_LOAD||p.p_memsz==0)continue;
   if(p.p_memsz>64*1024*1024||p.p_vaddr>~Word{}-p.p_memsz-page)throw std::runtime_error("ELF segment extent");
   const Word low=static_cast<Word>(p.p_vaddr)&~(page-1),high=(static_cast<Word>(p.p_vaddr+p.p_memsz)+page-1)&~(page-1);
   if(mmap(reinterpret_cast<void*>(low),high-low,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0)==MAP_FAILED)throw std::runtime_error("fixed isolated mapping unavailable");
   if(p.p_flags&PF_X){if(code_base!=0)throw std::runtime_error("multiple code segments");code_base=p.p_vaddr;code_end=p.p_vaddr+p.p_filesz;}
  }
  for(unsigned n=0;n<h.e_phnum;++n){
   const auto p=read<Elf64_Phdr>(h.e_phoff+n*h.e_phentsize);if(p.p_type!=PT_LOAD||p.p_flags&PF_W)continue;
   const Word low=static_cast<Word>(p.p_vaddr)&~(page-1),high=(static_cast<Word>(p.p_vaddr+p.p_memsz)+page-1)&~(page-1);
   if(p.p_offset>bytes.size()||p.p_filesz>bytes.size()-p.p_offset)throw std::runtime_error("segment extent");
   if(mprotect(reinterpret_cast<void*>(low),high-low,PROT_READ|PROT_WRITE))throw std::runtime_error("segment copy permission");
   std::memcpy(reinterpret_cast<void*>(p.p_vaddr),bytes.data()+p.p_offset,p.p_filesz);
   const int protection=(p.p_flags&PF_R?PROT_READ:0)|(p.p_flags&PF_X?PROT_EXEC:0);
   if(mprotect(reinterpret_cast<void*>(low),high-low,protection))throw std::runtime_error("segment final permission");
  }
  for(unsigned n=0;n<h.e_shnum;++n){
   const auto section=read<Elf64_Shdr>(h.e_shoff+n*h.e_shentsize);if(section.sh_type!=SHT_SYMTAB)continue;
   const auto strings=read<Elf64_Shdr>(h.e_shoff+section.sh_link*h.e_shentsize);
   if(strings.sh_offset>bytes.size()||strings.sh_size>bytes.size()-strings.sh_offset||section.sh_entsize!=sizeof(Elf64_Sym))throw std::runtime_error("symbol extent");
   for(Word i=0;i<section.sh_size/section.sh_entsize;++i){
    const auto sym=read<Elf64_Sym>(section.sh_offset+i*section.sh_entsize);if(sym.st_name>=strings.sh_size||sym.st_value==0)continue;
    const auto*name=reinterpret_cast<const char*>(bytes.data()+strings.sh_offset+sym.st_name);
    if(std::memchr(name,0,strings.sh_size-sym.st_name)==nullptr)throw std::runtime_error("symbol terminator");
    if(!symbols.emplace(name,static_cast<Word>(sym.st_value)).second)throw std::runtime_error("duplicate symbol");
   }
  }
  // The streaming writer emits no SHT_SYMTAB. Read offsets captured from the
  // actual emitter's NC_FN_OFFSETS by the sealed component driver instead.
  std::ifstream offsets(offset_path);if(!offsets)throw std::runtime_error("missing emitter offsets");
  std::string line;
  while(std::getline(offsets,line)){
   if(!line.starts_with("OFFSET "))continue;
   std::istringstream row(line.substr(7));std::string name,number,extra;
   if(!(row>>name>>number)||(row>>extra))throw std::runtime_error("offset row");
   Word offset=0;const auto parsed=std::from_chars(number.data(),number.data()+number.size(),offset);
   if(parsed.ec!=std::errc{}||parsed.ptr!=number.data()+number.size()||code_base==0||offset>=code_end-code_base)throw std::runtime_error("offset extent");
   if(!symbols.emplace(name,code_base+offset).second)throw std::runtime_error("duplicate emitter offset");
  }
 }
 template<class T>T read(Word offset)const{if(offset>bytes.size()||sizeof(T)>bytes.size()-offset)throw std::runtime_error("ELF read");T value;std::memcpy(&value,bytes.data()+offset,sizeof value);return value;}
 Word symbol(const std::string&name)const{const auto at=symbols.find(name);if(at==symbols.end())throw std::runtime_error("missing symbol "+name);return at->second;}
};
bool run(Word function,Word argument,int signal,Word expected){
 const auto pid=fork();if(pid<0)throw std::runtime_error("fork");
 if(pid==0){
  register Word b asm("rbx")=0x11112222,r12 asm("r12")=0x33334444,r13 asm("r13")=0x55556666,r14 asm("r14")=0x77778888,r15 asm("r15")=0x9999aaaa;
  asm volatile("" : "+r"(b),"+r"(r12),"+r"(r13),"+r"(r14),"+r"(r15));
  using Fn=std::int64_t(*)(std::int64_t);
  const auto result=sounio_native_callback_thunk(reinterpret_cast<Fn>(function),std::bit_cast<std::int64_t>(argument));
  const Word value=std::bit_cast<Word>(result);
  asm volatile("" : "+r"(b),"+r"(r12),"+r"(r13),"+r"(r14),"+r"(r15));
  if(b!=0x11112222||r12!=0x33334444||r13!=0x55556666||r14!=0x77778888||r15!=0x9999aaaa)_exit(74);
  _exit(value==expected?0:73);
 }
 int status=0;if(waitpid(pid,&status,0)!=pid)throw std::runtime_error("waitpid");
 if(signal)return WIFSIGNALED(status)&&WTERMSIG(status)==signal;
 return WIFEXITED(status)&&WEXITSTATUS(status)==0;
}
int main(int argc,char**argv){
 try{
  if(argc!=3 && argc!=4)throw std::runtime_error("usage: reference ELF EMITTER_LOG [--empty]");
  Elf elf(argv[1],argv[2]);
  const Word gi=elf.symbol("@worker:raw_i64"),gf=elf.symbol("@worker:raw_f64");
  if(argc==4){
   if(std::string(argv[3])!="--empty")throw std::runtime_error("unknown mode");
   for(const auto&[name,address]:elf.symbols){(void)address;if(name.starts_with("@worker:clone:"))throw std::runtime_error("empty list contains clone");}
   for(const auto guard:{gi,gf})for(const Word invalid:{Word{0},elf.symbol("worker_i64_original")})if(!run(guard,invalid,SIGILL,0))throw std::runtime_error("empty list accepted word");
   std::cout<<"WORKER_EMPTY_GUARD_OK refused=4 writable_segments=PROT_NONE\n";return 0;
  }
  const Word ci=elf.symbol("@worker:clone:8"),cf=elf.symbol("@worker:clone:9"),half=elf.symbol("@worker:clone:10");
  unsigned positives=0,negatives=0;
  auto expect=[&](bool ok,const char*what){if(!ok)throw std::runtime_error(what);};
  expect(run(reinterpret_cast<Word>(worker_stack_probe),0,0,8),"native thunk stack alignment");++positives;
  expect(run(reinterpret_cast<Word>(worker_clobber_probe),0x13579,0,0x13579),"native thunk callee-saved preservation");++positives;
  for(const auto address:{ci}){expect(run(gi,address,0,address),"i64 guard");++positives;}
  for(const auto address:{cf,half}){expect(run(gf,address,0,address),"f64 guard");++positives;}
  expect(run(ci,5,0,10),"i64 clone arithmetic");++positives;
  for(const Word bits:{Word{0},Word{0x8000000000000000ULL},Word{0x7ff8000000000042ULL},std::bit_cast<Word>(0.5)}){expect(run(cf,bits,0,bits),"f64 exact word ABI");++positives;}
  expect(run(half,std::bit_cast<Word>(1.0),0,std::bit_cast<Word>(0.5)),"f64 half arithmetic");++positives;
  for(const auto guard:{gi,gf})for(const Word invalid:{Word{0},Word{1},ci+1}){expect(run(guard,invalid,SIGILL,0),"corrupted word guard");++negatives;}
  for(const Word wrong:{cf,half}){expect(run(gi,wrong,SIGILL,0),"f64 word in i64 guard");++negatives;}
  expect(run(gf,ci,SIGILL,0),"i64 word in f64 guard");++negatives;
  expect(run(gi,elf.symbol("worker_i64_original"),SIGILL,0),"original i64 address guard");++negatives;
  expect(run(gf,elf.symbol("worker_f64_original"),SIGILL,0),"original f64 address guard");++negatives;
  unsigned controls=0;
  if(!run(gi,ci,0,ci+1))++controls;
  if(!run(half,std::bit_cast<Word>(1.0),0,std::bit_cast<Word>(1.0)))++controls;
  if(!run(gi,0,SIGSEGV,0))++controls;
  if(positives!=11||negatives!=11||controls!=3)throw std::runtime_error("nonempty oracle counts");
  std::cout<<"WORKER_NATIVE_GUARD_OK positive=11 invalid_words=11 wrong_controls=3 writable_segments=PROT_NONE native_thunk=canonical\n";
 }catch(const std::exception&e){std::cerr<<"WORKER_NATIVE_GUARD_REFUSED "<<e.what()<<'\n';return 1;}
}
