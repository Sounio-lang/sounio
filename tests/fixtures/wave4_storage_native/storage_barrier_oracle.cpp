#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
using word=std::uint64_t;
using Fn=word(*)(word,word,word,word);
std::vector<std::array<word,2>> trace;
extern "C" void storage_record(word value,word owner){trace.push_back({value,owner});}
struct Code {
 void*p{};std::size_t size=8192;
 explicit Code(const std::vector<unsigned char>&bytes,const std::vector<int>&relocs){
  if(bytes.empty()||bytes.size()>4096)throw std::runtime_error("invalid code size");
  p=mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if(p==MAP_FAILED)throw std::runtime_error("mmap");
  std::memcpy(p,bytes.data(),bytes.size());
  // Verify SysV call alignment at the barrier entry, then jump to C++ recorder.
  const unsigned char stub[]={0x48,0x89,0xe0,0x48,0x83,0xe0,15,0x48,0x83,0xf8,8,0x74,2,0x0f,0x0b,0x48,0xb8};
  auto*mem=static_cast<unsigned char*>(p);
  std::memcpy(mem+4096,stub,sizeof stub);
  const auto callback=reinterpret_cast<std::uint64_t>(&storage_record);
  std::memcpy(mem+4096+sizeof stub,&callback,8);
  mem[4096+sizeof stub+8]=0xff;mem[4096+sizeof stub+9]=0xe0;
  for(int pos:relocs){
   if(pos<=0||std::size_t(pos+4)>bytes.size()||mem[pos-1]!=0xe8)throw std::runtime_error("bad call relocation");
   std::int32_t displacement=4096-pos-4;std::memcpy(mem+pos,&displacement,4);
  }
  if(mprotect(p,size,PROT_READ|PROT_EXEC))throw std::runtime_error("mprotect");
 }
 ~Code(){munmap(p,size);}
 word call(word a=0,word b=0,word c=0,word d=0)const{
  return reinterpret_cast<Fn>(p)(a,b,c,d);
 }
};
void require(bool p){if(!p)throw std::runtime_error("storage oracle mismatch");}
void rejects(const Code&c,word b,word s,word n){
 pid_t pid=fork();if(pid<0)throw std::runtime_error("fork");
 if(pid==0){alarm(2);c.call(0,b,s,n);_exit(77);}
 int status=0;require(waitpid(pid,&status,0)==pid);
 require(WIFEXITED(status)&&WEXITSTATUS(status)==1);
}
int main(int argc,char**argv)try{
 if(argc!=2)return 64;
 std::ifstream in(argv[1]);require(bool(in));
 std::map<int,std::vector<unsigned char>>codes;
 std::map<int,std::vector<int>>relocs;
 for(int op,len,nrel;in>>op>>len>>nrel;){
  require(len>0&&len<=4096&&codes.count(op)==0);
  require(nrel>=0&&nrel<=4);
  auto&v=codes[op];
  for(int i=0,x;i<len;++i){require(bool(in>>x)&&x>=0&&x<=255);v.push_back(static_cast<unsigned char>(x));}
  for(int i=0,x;i<nrel;++i){require(bool(in>>x));relocs[op].push_back(x);}
 }
 require(in.eof()&&codes.size()==10);
 std::array<word,6>a{11,22,33,44,55,66};
 const auto ptr=[](auto*p){return reinterpret_cast<word>(p);};
 Code into(codes.at(1),relocs[1]),out(codes.at(2),relocs[2]),range(codes.at(3),relocs[3]);
 for(word bits:std::array<word,5>{0,1,~word{0},0x8000000000000000ULL,0x7ff8000000001234ULL}){
  require(into.call(0,ptr(&a[0]),bits)==0&&a[0]==bits);
  require(out.call(0,ptr(&a[0]))==bits&&a[0]==0);
 }
 for(int op:{4,5,20,21}){
  Code borrow(codes.at(op),relocs[op]);
  require(borrow.call(0,0x12345)==0x12345); // inaccessible address: no T read
 }
 for(int op:{17,18}){
  Code unit(codes.at(op),relocs[op]);require(unit.call(0,0,~word{0})==0);
 }
 Code unit_range(codes.at(19),relocs[19]);
 require(unit_range.call(0,0,0,0)==0);
 require(unit_range.call(0,0,0,1ULL<<62)==0);
 rejects(unit_range,0,0,~word{0});
 a={11,22,33,44,55,66};
 range.call(0,ptr(&a[1]),ptr(&a[0]),4);
 require(a==std::array<word,6>{0,11,22,33,44,66});
 a={11,22,33,44,55,66};
 range.call(0,ptr(&a[0]),ptr(&a[1]),4);
 require(a==std::array<word,6>{22,33,44,55,0,66});
 std::array<word,3>b{0,0,0};
 a={11,22,33,44,55,66};range.call(0,ptr(b.data()),ptr(a.data()),3);
 require(b==std::array<word,3>{11,22,33});
 require(a==std::array<word,6>{0,0,0,44,55,66});
 range.call(0,ptr(b.data()),ptr(b.data()),3);
 require(b==std::array<word,3>{11,22,33});
 require(range.call(0,0,0,0)==0);
 rejects(range,0,ptr(a.data()),1);
 rejects(range,ptr(a.data()),0,1);
 rejects(range,ptr(a.data())+1,ptr(b.data()),1);
 rejects(range,ptr(a.data()),ptr(b.data())+1,1);
 rejects(range,ptr(a.data()),ptr(b.data()),~word{0});
 rejects(range,ptr(a.data()),ptr(b.data()),1ULL<<61);
 rejects(range,~word{7},ptr(b.data()),1);
 rejects(range,ptr(a.data()),~word{7},1);
 rejects(into,0,9,0);rejects(out,0,0,0);
 rejects(into,ptr(a.data())+1,9,0);rejects(out,ptr(a.data())+1,0,0);
 trace.clear();a[0]=77;
 into.call(0xabc,ptr(&a[0]),0x1234);
 require(trace==std::vector<std::array<word,2>>{{0x1234,0xabc}});
 trace.clear();require(out.call(0xdef,ptr(&a[0]))==0x1234);
 require(trace==std::vector<std::array<word,2>>{{0x1234,0}});
 trace.clear();Code borrow(codes.at(4),relocs[4]);
 require(borrow.call(0x555,0x12345)==0x12345);
 require(trace==std::vector<std::array<word,2>>{{0x555,0}});
 trace.clear();a={11,22,33,44,55,66};
 range.call(0xabc,ptr(&a[1]),ptr(&a[0]),4);
 require(trace==std::vector<std::array<word,2>>{{44,0xabc},{33,0xabc},{22,0xabc},{11,0xabc}});
 for(int op:{17,18,19}){trace.clear();Code unit(codes.at(op),relocs[op]);unit.call(9,0,0,2);require(trace.empty());}
 std::cout<<"storage transport and barrier ABI: 10 opcodes; word bits, units, overlap, invalid ranges and aligned value/owner calls: OK\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
