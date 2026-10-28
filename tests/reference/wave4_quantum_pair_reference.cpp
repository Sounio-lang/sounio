#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
static_assert(17+25==42);
static std::string expected(std::string_view mode) {
 if(mode!="match"&&mode!="let")return {};
 return "PAIR_FIRST=17\nPAIR_SECOND=25\nPAIR_SUM=42\nLINEAR_PAIR_"+std::string(mode=="match"?"MATCH":"LET")+"_OK\n";
}
static bool verify(std::string_view mode,const std::string& got) {return !expected(mode).empty()&&got==expected(mode);}
int main(int argc,char**argv) {
 if(argc==2&&std::string_view(argv[1])=="--self-test"){
  const auto good=expected("match");
  if(!verify("match",good)||verify("let",good)||verify("match","")||verify("match",good+"extra\n"))return 1;
  auto swapped=good;swapped.replace(swapped.find("FIRST=17"),8,"FIRST=25");swapped.replace(swapped.find("SECOND=25"),9,"SECOND=17");
  auto wrongsum=good;wrongsum.replace(wrongsum.find("SUM=42"),6,"SUM=41");
  if(verify("match",swapped)||verify("match",wrongsum))return 1;
  std::cout<<"PAIR_ORACLE_ADVERSARIAL_OK\n";return 0;
 }
 if(argc!=3)return 64;
 std::ifstream f(argv[2],std::ios::binary);if(!f)return 2;
 std::ostringstream s;s<<f.rdbuf();
 if(!verify(argv[1],s.str())){std::cerr<<"PAIR_ROLE_ORACLE_REFUSED\n";return 1;}
 std::cout<<"PAIR_ROLE_ORACLE_OK\n";return 0;
}
