#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace fs=std::filesystem;
constexpr std::array<unsigned char,5> payload{0,1,127,128,255};
void require(bool p,std::string_view why){if(!p)throw std::runtime_error(std::string(why));}
void binary(const fs::path&p,std::uint64_t n,bool patterned){
 std::ofstream f(p,std::ios::binary);require(bool(f),"open fixture");
 for(std::uint64_t i=0;i<n;++i)f.put(static_cast<char>(patterned?(i*37+11)%256:payload[i%payload.size()]));
 require(bool(f),"write fixture");
}
void source(const fs::path&p,std::string_view asset,std::uint64_t n){
 std::ofstream f(p);
 f<<"fn byte_at(data:*const u8,index:i64)->i64 with Mut { let p=((data as i64)+index) as *const u8; (*p) as i64 }\n";
 f<<"fn main()->i32 with IO,Mut {\n let (data,size)=static_blob_bytes(\""<<asset<<"\")\n if size!="<<n<<" { return 1 }\n";
 if(n==5){for(std::size_t i=0;i<payload.size();++i)f<<" if byte_at(data,"<<i<<")!="<<unsigned(payload[i])<<" { return 2 }\n";}
 else {for(std::uint64_t i:std::array<std::uint64_t,5>{0,1,4095,1048576,n-1})f<<" if byte_at(data,"<<i<<")!="<<(i*37+11)%256<<" { return 3 }\n";}
 f<<" print(\"BLOB_ADDRESS=\") println(data as i64)\n println(\"STATIC_BLOB_READONLY_OK\")\n 0\n}\n";
 require(bool(f),"write source");
}
std::vector<unsigned char> bytes(const fs::path&p){
 std::ifstream f(p,std::ios::binary);require(bool(f),"read ELF");
 return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
std::uint64_t read(const std::vector<unsigned char>&b,std::uint64_t p,std::uint64_t n){
 require(n<=8&&p<=b.size()&&n<=b.size()-p,"ELF field bound");
 std::uint64_t v=0;for(std::uint64_t i=0;i<n;++i)v|=std::uint64_t(b[p+i])<<(8*i);return v;
}
std::uint64_t address(const fs::path&p){
 std::ifstream f(p);std::string s;std::uint64_t value=0;bool found=false,marker=false;
 while(std::getline(f,s)){
  if(s=="STATIC_BLOB_READONLY_OK")marker=true;
  if(s.starts_with("BLOB_ADDRESS=")){
   require(!found,"duplicate address");auto token=std::string_view(s).substr(13);
   const auto r=std::from_chars(token.data(),token.data()+token.size(),value);
   require(r.ec==std::errc{}&&r.ptr==token.data()+token.size(),"address parse");found=true;
  }
 }
 require(found&&marker,"native address and original success marker required");return value;
}
struct Location{std::uint64_t offset,flags_offset;};
Location locate(const std::vector<unsigned char>&b,std::uint64_t addr,std::uint64_t extent){
 require(b.size()>=64&&b[0]==127&&b[1]=='E'&&b[2]=='L'&&b[3]=='F'&&b[4]==2&&b[5]==1,"ELF64 little-endian");
 const auto ph=read(b,32,8),entry=read(b,54,2),count=read(b,56,2);
 require(entry>=56&&count>0&&count<=64,"ELF program header bounds");
 for(std::uint64_t i=0;i<count;++i){
  const auto p=ph+i*entry;if(read(b,p,4)!=1)continue;
  const auto va=read(b,p+16,8),size=read(b,p+32,8);
  if(addr>=va&&addr-va<=size&&extent<=size-(addr-va)){
   const auto flags=read(b,p+4,4);require((flags&4)!=0&&(flags&2)==0,"returned pointer is not in readonly readable PT_LOAD");
   const auto off=read(b,p+8,8)+(addr-va);require(off<=b.size()&&extent<=b.size()-off,"payload file extent");
   return {off,p+4};
  }
 }
 throw std::runtime_error("runtime pointer not backed by an ELF PT_LOAD");
}
int main(int argc,char**argv){try{
 if(argc==2){
  fs::path root(argv[1]);fs::create_directories(root/"nested");
  std::ofstream f(root/"nested"/"payload.bin",std::ios::binary);f.write(reinterpret_cast<const char*>(payload.data()),payload.size());f.close();
  std::ofstream(root/"empty.bin",std::ios::binary).close();require(bool(f),"small fixture");
  std::cout<<"STATIC_BLOB_REFERENCE_CREATED\n";return 0;
 }
 require(argc>=3,"usage: reference <fixtures> | --probes <dir> | --verify <elf> <stdout> <small|large>");
 const std::string_view mode=argv[1];
 if(mode=="--probes"){
  fs::path d(argv[2]);fs::create_directories(d);binary(d/"small.bin",5,false);source(d/"small.sio","small.bin",5);
  binary(d/"large.bin",3*1024*1024,true);source(d/"large.sio","large.bin",3*1024*1024);
  std::ofstream big(d/"oversized.bin",std::ios::binary);big.seekp(32*1024*1024-1);big.put(0);big.close();
  std::ofstream(d/"oversized.sio")<<"fn main()->i32 { let (p,n)=static_blob_bytes(\"oversized.bin\"); 0 }\n";
  binary(d/"stale.bin",5,false);
  const auto stale_path=(d/"stale.bin").string();
  require(stale_path.find('"')==std::string::npos&&stale_path.find('\\')==std::string::npos,"fixture path quoting");
  std::ofstream stale(d/"stale.sio");
  stale<<"use parser::ast::{make_name};\nuse compiler::static_assets::{static_asset_identity,static_asset_read_verified,static_asset_digest};\n";
  stale<<"fn main()->i32 with IO,Mut,Panic,Div {\n let path=\""<<stale_path<<"\"\n let identity=static_asset_identity(make_name(path),5)\n if identity.len<=0 {return 1}\n";
  stale<<" let hash=static_asset_digest(read_file(path),5)\n print(\"SEALED_SHA256=\") println(str_from_bytes(hash.buf,64))\n";
  stale<<" match static_asset_read_verified(identity,5) {Some(data)=>{},None=>{return 2}}\n";
  stale<<" match static_asset_read_verified(identity,4) {Some(data)=>{return 3},None=>{}}\n";
  stale<<" if write_file(path,\"abcde\",5)!=5 {return 4}\n";
  stale<<" match static_asset_read_verified(identity,5) {Some(data)=>{return 5},None=>{}}\n";
  stale<<" println(\"STALE_ASSET_SAME_LENGTH_REJECTED\")\n 0\n}\n";
  require(bool(stale),"write stale source");
  std::cout<<"STATIC_BLOB_CPP_PROBES_CREATED\n";return 0;
 }
 require(mode=="--verify"&&argc==5,"verify arguments");
 const bool small=std::string_view(argv[4])=="small";require(small||std::string_view(argv[4])=="large","verify mode");
 const auto b=bytes(argv[2]);const std::uint64_t n=small?5:3*1024*1024;
 const auto location=locate(b,address(argv[3]),n);
 for(std::uint64_t i=0;i<n;++i)require(b[location.offset+i]==(small?payload[i]:(i*37+11)%256),"embedded payload differs from C++ oracle");
 std::cout<<"STATIC_BLOB_CPP_READONLY_AND_ALL_BYTES_OK bytes="<<n<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"STATIC_BLOB_REFERENCE_REFUSAL "<<e.what()<<"\n";return 1;}}
