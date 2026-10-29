// C++23 ELF byte-window extractor only. No disassembly or semantic rewrite.
#include <cstdint>
#include <cstring>
#include <elf.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc,char**argv)try{
 if(argc!=4&&argc!=5)return 64;
 std::ifstream f(argv[1],std::ios::binary);if(!f)throw std::runtime_error("input");
 std::vector<unsigned char>b((std::istreambuf_iterator<char>(f)),{});
 if(b.size()<sizeof(Elf64_Ehdr)||b.size()>67108864)throw std::runtime_error("size");
 Elf64_Ehdr e;std::memcpy(&e,b.data(),sizeof e);
 if(std::memcmp(e.e_ident,ELFMAG,4)||e.e_machine!=EM_X86_64||e.e_ident[EI_CLASS]!=ELFCLASS64||e.e_ident[EI_DATA]!=ELFDATA2LSB||e.e_phentsize!=sizeof(Elf64_Phdr)||e.e_phnum>128||e.e_phoff>b.size()||std::uint64_t(e.e_phnum)*sizeof(Elf64_Phdr)>b.size()-e.e_phoff)throw std::runtime_error("header");
 std::size_t pos=0;std::uint64_t start=std::stoull(argv[2],&pos,0);if(pos!=std::string(argv[2]).size())throw std::runtime_error("address");
 pos=0;const std::uint64_t count=std::stoull(argv[3],&pos,0);if(pos!=std::string(argv[3]).size()||count==0||count>512)throw std::runtime_error("count");
 if(argc==5){if(std::string(argv[4])!="around"||start<32)throw std::runtime_error("around");start-=32;}
 for(unsigned k=0;k<e.e_phnum;++k){
  Elf64_Phdr p;std::memcpy(&p,b.data()+e.e_phoff+k*sizeof p,sizeof p);
  if(p.p_type!=PT_LOAD||!(p.p_flags&PF_X)||start<p.p_vaddr)continue;
  const auto delta=start-p.p_vaddr;
  if(delta>p.p_filesz||count>p.p_filesz-delta||p.p_offset>b.size()||p.p_filesz>b.size()-p.p_offset)continue;
  const auto off=std::size_t(p.p_offset+delta);
  std::cout<<"ELF_WINDOW va=0x"<<std::hex<<start<<" file_offset=0x"<<off<<std::dec<<" bytes="<<count<<"\n";
  for(std::uint64_t i=0;i<count;++i){
   if(i%16==0)std::cout<<"0x"<<std::hex<<(start+i)<<": ";
   std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(b[off+std::size_t(i)])<<' ';
   if(i%16==15||i+1==count)std::cout<<'\n';
  }
  return 0;
 }
 throw std::runtime_error("requested window outside executable file extent");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
