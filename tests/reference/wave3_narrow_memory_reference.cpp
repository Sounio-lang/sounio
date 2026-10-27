// Independent C++23 byte-width oracle; writes exact Sounio regressions.
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
namespace fs=std::filesystem;
void fixture(const fs::path& dir,const std::string& name,const std::string& body){
 std::ofstream o(dir/(name+".sio"));
 o<<"// run-pass\n// requires: madaros\n// stdout: NARROW_WIDTH_OK\n// Expected byte patterns generated in C++23; allocation size is unpadded.\nfn main() with Alloc,Mut,Panic,IO {\n let base=heap_alloc(4088)\n assert((base as i64)>0)\n let words=base as *mut i64\n"<<body<<" heap_free(base)\n print(\"NARROW_WIDTH_OK\\n\")\n}\n";
 if(!o)throw std::runtime_error("write fixture");
}
std::string expr(std::int64_t x){
 if(x==std::numeric_limits<std::int64_t>::min())return "(0-9223372036854775807-1)";
 if(x<0)return "(0-"+std::to_string(-x)+")";return std::to_string(x);
}
int main(int argc,char**argv){
 if(argc!=2)return 64;fs::path d=argv[1];fs::create_directories(d);
 fixture(d,"narrow_i8_load"," write_i64(words,510,"+expr(std::numeric_limits<std::int64_t>::min())+")\n let last=((base as i64)+4087) as *mut i8\n assert((*last as i64)=="+expr(std::numeric_limits<std::int8_t>::min())+")\n");
 fixture(d,"narrow_u8_load"," write_i64(words,510,(0-1))\n let last=((base as i64)+4087) as *mut u8\n assert((*last as i64)=="+std::to_string(std::numeric_limits<std::uint8_t>::max())+")\n");
 for(const auto& t:std::array<std::string,2>{"i32","u32"}){
  bool sign=t=="i32";std::int64_t initial=sign?std::numeric_limits<std::int64_t>::min():-1;
  std::string value=sign?expr(std::numeric_limits<std::int32_t>::min()):std::to_string(std::numeric_limits<std::uint32_t>::max());
  fixture(d,"narrow_"+t+"_load"," write_i64(words,510,"+expr(initial)+")\n let last=((base as i64)+4084) as *mut "+t+"\n assert((*last as i64)=="+value+")\n");
  fixture(d,"narrow_"+t+"_store"," let last=((base as i64)+4084) as *mut "+t+"\n *last="+value+" as "+t+"\n assert((*last as i64)=="+value+")\n");
 }
 constexpr std::uint64_t original=0x1122334455667788ULL;
 constexpr std::uint64_t sentinel=0x1234567812345678ULL;
 const auto expected8=(original&~(std::uint64_t{255}<<24))|(std::uint64_t{128}<<24);
 const auto expected32=(original&0xFFFFFFFFULL)|(std::uint64_t{0x80000000}<<32);
 fixture(d,"narrow_i8_neighbors"," write_i64(words,0,"+std::to_string(original)+")\n write_i64(words,1,"+std::to_string(sentinel)+")\n let at=((base as i64)+3) as *mut i8\n *at=(0-128) as i8\n assert(read_i64(words,0)=="+std::to_string(expected8)+")\n assert(read_i64(words,1)=="+std::to_string(sentinel)+")\n assert((*at as i64)==(0-128))\n");
 fixture(d,"narrow_i32_neighbors"," write_i64(words,0,"+std::to_string(original)+")\n write_i64(words,1,"+std::to_string(sentinel)+")\n let at=((base as i64)+4) as *mut i32\n *at=(0-2147483648) as i32\n assert(read_i64(words,0)=="+expr(std::bit_cast<std::int64_t>(expected32))+")\n assert(read_i64(words,1)=="+std::to_string(sentinel)+")\n assert((*at as i64)==(0-2147483648))\n");
 std::cout<<"NARROW_WIDTH_ORACLE_OK 8\n";
}
