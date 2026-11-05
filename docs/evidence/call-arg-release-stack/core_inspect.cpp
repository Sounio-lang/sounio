#include <elf.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 std::ifstream f(argv[1],std::ios::binary); Elf64_Ehdr e{};
 f.read(reinterpret_cast<char*>(&e),sizeof e);
 if(std::memcmp(e.e_ident,ELFMAG,SELFMAG)||e.e_type!=ET_CORE)return 3;
 std::vector<Elf64_Phdr> p(e.e_phnum);
 f.seekg(e.e_phoff);f.read(reinterpret_cast<char*>(p.data()),p.size()*sizeof(Elf64_Phdr));
 uint64_t bp=0,sp=0;
 for(auto h:p)if(h.p_type==PT_NOTE){
  std::vector<char>b(h.p_filesz);f.seekg(h.p_offset);f.read(b.data(),b.size());
  size_t o=0;while(o+sizeof(Elf64_Nhdr)<=b.size()){
   Elf64_Nhdr n{};std::memcpy(&n,b.data()+o,sizeof n);o+=sizeof n;
   auto name=o;o+=(n.n_namesz+3)&~size_t(3);auto d=o;o+=(n.n_descsz+3)&~size_t(3);
   if(o>b.size())throw std::runtime_error("note bounds");
   if(std::string(b.data()+name)=="CORE"&&n.n_type==NT_PRSTATUS&&n.n_descsz>=328){
    uint64_t r[27];std::memcpy(r,b.data()+d+112,sizeof r);bp=r[4];sp=r[19];
    std::cout<<std::hex<<"rip=0x"<<r[16]<<" rax=0x"<<r[10]<<" rbp=0x"<<bp<<" rsp=0x"<<sp<<"\n";
   }
   if(std::string(b.data()+name)=="CORE"&&n.n_type==NT_SIGINFO&&n.n_descsz>=24){
    int32_t sig,code;uint64_t addr;std::memcpy(&sig,b.data()+d,4);std::memcpy(&code,b.data()+d+8,4);std::memcpy(&addr,b.data()+d+16,8);
    std::cout<<std::dec<<"signal="<<sig<<" code="<<code<<std::hex<<" fault=0x"<<addr<<"\n";
   }
  }
 }
 for(auto h:p)if(h.p_type==PT_LOAD&&bp>=h.p_vaddr&&bp<h.p_vaddr+h.p_memsz)
  std::cout<<std::hex<<"stack_map=0x"<<h.p_vaddr<<"..0x"<<h.p_vaddr+h.p_memsz<<"\n";
 for(int i=0;i<10&&bp;i++){
  bool found=false;
  for(auto h:p)if(h.p_type==PT_LOAD&&bp>=h.p_vaddr&&bp+16<=h.p_vaddr+h.p_filesz){
   uint64_t v[2];f.seekg(h.p_offset+bp-h.p_vaddr);f.read(reinterpret_cast<char*>(v),16);
   std::cout<<std::dec<<"frame="<<i<<std::hex<<" bp=0x"<<bp<<" prev=0x"<<v[0]<<" return=0x"<<v[1]<<"\n";
   if(v[0]<=bp)bp=0;else bp=v[0];found=true;break;
  }
  if(!found)break;
 }
}
