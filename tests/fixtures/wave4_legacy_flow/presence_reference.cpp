#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
int main(int argc,char**argv) {
 if(argc!=2) return 2;
 std::ifstream f(argv[1]);
 if(!f) return 3;
 std::string line;std::vector<std::string> expected;
 expected.push_back("BITS 0 1 "+std::to_string(std::uint64_t{1}<<31)+" 0");
 for(auto tag:{"ABSENT","BOUNDARY","ALL"}) {
  std::array<std::uint32_t,3> masks{};
  std::array<bool,69> present{};
  for(unsigned i=0;i<69;++i) {
   present[i]=std::string(tag)=="ALL"||(std::string(tag)=="BOUNDARY"&&(i==0||i==31||i==32||i==63||i==64||i==68));
   if(present[i])masks[i/32]|=std::uint32_t{1}<<(i%32);
  }
  expected.push_back(std::string("MASK ")+tag+" "+std::to_string(masks[0])+" "+std::to_string(masks[1])+" "+std::to_string(masks[2]));
  for(unsigned i=0;i<69;++i) expected.push_back(std::string("FIELD ")+tag+" "+std::to_string(i)+" "+std::to_string(present[i]?100+static_cast<int>(i):-1));
 }
 std::size_t at=0,bad=0;
 while(std::getline(f,line)){if(at>=expected.size()||line!=expected[at]){std::cout<<"MISMATCH row="<<at<<" got="<<line<<"\n";++bad;}++at;}
 if(at!=expected.size()){std::cout<<"CENSUS got="<<at<<" expected="<<expected.size()<<"\n";++bad;}
 std::cout<<"PRESENCE_ORACLE rows="<<at<<" failed="<<bad<<"\n";return bad?4:0;
}
