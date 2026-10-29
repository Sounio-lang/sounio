// ADR009 C++23 exact oracle. C++ stack fill is in THIS process only.
// Machine inspection recognizes listed x86 byte patterns; it is not a full disassembler.
#include <array>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <elf.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
using I=std::int64_t;
using Row=std::array<I,5>;
using Rows=std::vector<Row>;
[[noreturn]]static void bad(const std::string&s){throw std::runtime_error(s);}
[[gnu::noinline]]static I reference_clobber(I seed){
 volatile I cells[512];
 for(I i=0;i<512;++i)cells[i]=seed+i;
 I total=0;for(I i=0;i<512;++i)total+=cells[i];
 return total;
}
static Rows expected(){return {{0,42,1,7,0},{1,42,1,7,reference_clobber(100)},
 {2,-9,0,8,0},{3,-9,0,8,reference_clobber(101)}};}
static Rows read_rows(const char*p){
 std::ifstream f(p);if(!f)bad("missing native stdout");Rows r;
 for(std::string line;std::getline(f,line);){
  std::istringstream in(line);Row row{};for(I&v:row)if(!(in>>v))bad("invalid integer row");
  std::string extra;if(in>>extra)bad("extra row field");r.push_back(row);
 }
 return r;
}
static void verify(const Rows&r){
 const auto want=expected();if(r.size()!=want.size())bad("row count mismatch");
 for(std::size_t i=0;i<r.size();++i)if(r[i]!=want[i]){
  std::ostringstream s;s<<"row "<<i<<" mismatch got=";for(I v:r[i])s<<v<<',';s<<" expected=";for(I v:want[i])s<<v<<',';bad(s.str());
 }
}
static void emit(const Rows&r,std::ostream&f){for(const auto&a:r){for(std::size_t i=0;i<a.size();++i)f<<(i?" ":"")<<a[i];f<<'\n';}}
static void controls(){
 const auto e=expected();verify(e);I n=1;
 auto refuse=[&](Rows r){try{verify(r);}catch(const std::exception&){++n;return;}bad("oracle mutant accepted");};
 auto wrong=e;wrong[1][1]=43;refuse(wrong);
 auto drop=e;drop.pop_back();refuse(drop);
 auto extra=e;extra.push_back(e[0]);refuse(extra);
 auto order=e;std::swap(order[0],order[1]);refuse(order);
 auto sum=e;sum[1][4]+=1;refuse(sum);
 std::cout<<"TUPLE_CPP23_ORACLE_CONTROLS_OK count="<<n<<'\n';
}
using Bytes=std::vector<unsigned char>;
static std::uint32_t u32(const Bytes&b,std::size_t i){std::uint32_t n;std::memcpy(&n,b.data()+i,4);return n;}
static bool seq(const Bytes&b,std::size_t i,std::initializer_list<unsigned char>s){
 return i<=b.size()&&s.size()<=b.size()-i&&std::equal(s.begin(),s.end(),b.begin()+std::ptrdiff_t(i));
}
struct Frame{std::size_t off;std::uint64_t va;std::uint32_t bytes;bool sret;};
static void scan(const char*p){
 static_assert(std::endian::native==std::endian::little);
 std::ifstream f(p,std::ios::binary);if(!f)bad("missing ELF");
 Bytes b((std::istreambuf_iterator<char>(f)),{});
 if(b.size()<sizeof(Elf64_Ehdr)||b.size()>67108864)bad("ELF file size");
 Elf64_Ehdr e;std::memcpy(&e,b.data(),sizeof e);
 if(std::memcmp(e.e_ident,ELFMAG,SELFMAG)||e.e_ident[EI_CLASS]!=ELFCLASS64||
 e.e_ident[EI_DATA]!=ELFDATA2LSB||e.e_machine!=EM_X86_64||e.e_phentsize!=sizeof(Elf64_Phdr)||
 e.e_phnum==0||e.e_phnum>128||e.e_phoff>b.size()||std::uint64_t(e.e_phnum)*e.e_phentsize>b.size()-e.e_phoff)bad("ELF64 x86 header");
 std::size_t fills=0,frames=0,copies=0;
 for(unsigned k=0;k<e.e_phnum;++k){
  Elf64_Phdr h;std::memcpy(&h,b.data()+e.e_phoff+k*sizeof h,sizeof h);
  if(h.p_type!=PT_LOAD||!(h.p_flags&PF_X))continue;
  if(h.p_offset>b.size()||h.p_filesz>b.size()-h.p_offset||h.p_vaddr>UINT64_C(0x00007fffffffffff)||h.p_filesz>UINT64_C(0x00007fffffffffff)-h.p_vaddr)bad("ELF executable segment bounds");
  const auto begin=std::size_t(h.p_offset),end=begin+std::size_t(h.p_filesz);
  std::vector<Frame> ff;
  for(std::size_t i=begin;i+13<=end;++i)if(seq(b,i,{0x55,0x48,0x89,0xe5})){
   std::size_t q=i+4;const bool sret=seq(b,q,{0x41,0x54});if(sret)q+=2;
   if(q+7<=end&&seq(b,q,{0x48,0x81,0xec})){
    Frame x{i,h.p_vaddr+i-begin,u32(b,q+3),sret};ff.push_back(x);++frames;
    std::cout<<"X86_PATTERN_FRAME va=0x"<<std::hex<<x.va<<std::dec<<" frame_bytes="<<x.bytes<<" saved_r12="<<x.sret<<'\n';
   }
  }
  auto frame=[&](std::size_t i)->const Frame*{const Frame*r=nullptr;for(const auto&x:ff)if(x.off<=i)r=&x;return r;};
  for(std::size_t i=begin;i<end;++i){
   if(i+15<=end&&seq(b,i,{0x48,0x8d,0xbd})&&b[i+7]==0xb9&&u32(b,i+8)==512&&seq(b,i+12,{0xf3,0x48,0xab})){
    const auto disp=std::bit_cast<std::int32_t>(u32(b,i+3));const auto*x=frame(i);++fills;
    std::cout<<"X86_PATTERN_RBP_FILL512 va=0x"<<std::hex<<(h.p_vaddr+i-begin)<<std::dec<<" rbp_disp="<<disp<<" words=512";
    if(x)std::cout<<" frame_va=0x"<<std::hex<<x->va<<std::dec<<" frame_bytes="<<x->bytes;
    std::cout<<" NOTE=byte_pattern_not_general_disassembly\n";
   }
   if(i+7<=end&&seq(b,i,{0x48,0x8d,0x85})){
    std::cout<<"X86_PATTERN_RAX_RBP_ADDRESS va=0x"<<std::hex<<(h.p_vaddr+i-begin)<<std::dec<<" rbp_disp="<<std::bit_cast<std::int32_t>(u32(b,i+3));
    if(const auto*x=frame(i))std::cout<<" frame_va=0x"<<std::hex<<x->va<<std::dec;
    std::cout<<'\n';
   }
   if(i+3<=end&&seq(b,i,{0x4c,0x89,0xe1})){
    std::size_t q=i+3,n=0;
    while(q+14<=end&&seq(b,q,{0x48,0x8b,0x82})&&seq(b,q+7,{0x48,0x89,0x81})&&u32(b,q+3)==n*8&&u32(b,q+10)==n*8){++n;q+=14;}
    if(n&&q+3<=end&&seq(b,q,{0x4c,0x89,0xe0})){
     ++copies;std::cout<<"X86_PATTERN_SRET_QWORD_COPY va=0x"<<std::hex<<(h.p_vaddr+i-begin)<<std::dec<<" words="<<n;
     if(const auto*x=frame(i))std::cout<<" frame_va=0x"<<std::hex<<x->va<<std::dec;
     std::cout<<'\n';
    }
   }
   if(i+5<=end&&b[i]==0xe8){
    const auto delta=std::bit_cast<std::int32_t>(u32(b,i+1));
    const std::int64_t target=std::int64_t(h.p_vaddr+i-begin+5)+delta;
    for(const auto&x:ff)if(target>=0&&std::uint64_t(target)==x.va)
     std::cout<<"X86_PATTERN_CALL_TO_FRAME va=0x"<<std::hex<<(h.p_vaddr+i-begin)<<" target=0x"<<x.va<<std::dec<<'\n';
   }
  }
 }
 std::cout<<"X86_PATTERN_SUMMARY frames="<<frames<<" rbp_fill512="<<fills<<" sret_qword_copy="<<copies<<" scope=patterns_only\n";
}
int main(int argc,char**argv)try{
 if(argc==2&&std::string_view(argv[1])=="--expected"){emit(expected(),std::cout);return 0;}
 if(argc==2&&std::string_view(argv[1])=="--controls"){controls();return 0;}
 if(argc==3&&std::string_view(argv[1])=="--check"){verify(read_rows(argv[2]));std::cout<<"TUPLE_CPP23_EXACT_ROWS_OK rows=4\n";return 0;}
 if(argc==3&&std::string_view(argv[1])=="--scan"){scan(argv[2]);return 0;}
 return 64;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
