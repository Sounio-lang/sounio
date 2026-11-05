#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>
struct Entry { std::uint64_t off; std::string name; };
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 std::ifstream f(argv[1]);
 std::string s((std::istreambuf_iterator<char>(f)),{});
 std::regex r(R"(fnmap fn=\s*[0-9]+\s*off=\s*([0-9]+)\s*name=([^\s]+))");
 std::vector<Entry> entries;
 for(auto i=std::sregex_iterator(s.begin(),s.end(),r);i!=std::sregex_iterator();++i)
  entries.push_back({std::stoull((*i)[1].str()),(*i)[2].str()});
 for(std::size_t i=0;i+1<entries.size();++i)
  if(entries[i].name=="checker_release_call_arg_borrows_inplace"){
   std::cout<<std::hex<<"0x"<<0x401000+entries[i].off<<" 0x"<<0x401000+entries[i+1].off<<"\n";
   std::cerr<<"function="<<entries[i].name<<" next="<<entries[i+1].name<<"\n";
   return 0;
  }
 std::cerr<<"function not found; parsed entries="<<entries.size()<<"\n";return 1;
}
