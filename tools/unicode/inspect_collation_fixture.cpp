// C++23 byte-safe fixture inspector, used to preserve exact failing collation rows.
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc,char**argv){try{if(argc!=3)return 64;std::uint64_t target=0;std::string arg=argv[2];auto[p,e]=std::from_chars(arg.data(),arg.data()+arg.size(),target);if(e!=std::errc{}||p!=arg.data()+arg.size())throw std::runtime_error("row number");std::ifstream in(argv[1],std::ios::binary);std::vector<unsigned char>b{std::istreambuf_iterator<char>(in),{}};auto read=[&](std::uint64_t at){if(at>b.size()||b.size()-at<4)throw std::runtime_error("u32 bounds");return std::uint32_t(b[at])|(std::uint32_t(b[at+1])<<8)|(std::uint32_t(b[at+2])<<16)|(std::uint32_t(b[at+3])<<24);};if(b.size()<24||std::string(b.begin(),b.begin()+8)!="SOUNUTF1"||read(8)!=4||target>=read(12)||read(16)!=b.size()-24||read(20)!=0x120000)throw std::runtime_error("header");std::uint64_t at=24;for(std::uint64_t row=0;row<=target;++row){auto root=read(at),strength=read(at+4),shifted=read(at+8),backwards=read(at+12),encoded=read(at+16);at+=20;if(row==target)std::cout<<"row="<<row<<" root="<<root<<" strength="<<strength<<" shifted="<<shifted<<" backwards="<<backwards<<" expected="<<(static_cast<std::int64_t>(encoded)-1)<<"\n";for(const char*name:{"left","right"}){auto n=read(at);at+=4;if(n>(b.size()-at)/4)throw std::runtime_error("sequence bounds");if(row==target)std::cout<<name<<": ";for(std::uint32_t i=0;i<n;++i){auto cp=read(at);at+=4;if(row==target)std::cout<<std::hex<<cp<<' ';}if(row==target)std::cout<<std::dec<<'\n';}}return 0;}catch(const std::exception&e){std::cerr<<"FIXTURE_INSPECT_ERROR "<<e.what()<<'\n';return 1;}}
