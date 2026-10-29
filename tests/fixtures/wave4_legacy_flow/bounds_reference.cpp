#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
struct C { const char* name; std::int64_t en,st,tc,nt,et,vt,ei,vc; };
static std::vector<C> cases() { return {
 {"valid",1,0,6,1,1,3,0,1},
 {"en_zero",0,0,6,1,1,3,0,1},
 {"en_negative",-1,0,6,1,1,3,0,1},
 {"en_last",256,0,6,1,1,3,0,1},
 {"en_first_oob",257,0,6,1,1,3,0,1},
 {"en_max",std::numeric_limits<std::int64_t>::max(),0,6,1,1,3,0,1},
 {"en_min",std::numeric_limits<std::int64_t>::min(),0,6,1,1,3,0,1},
 {"st_negative",1,-1,6,1,1,3,0,1},
 {"st_last",1,1024,6,1,1,3,0,1},
 {"st_first_oob",1,1025,6,1,1,3,0,1},
 {"st_max",1,std::numeric_limits<std::int64_t>::max(),6,1,1,3,0,1},
 {"st_min",1,std::numeric_limits<std::int64_t>::min(),6,1,1,3,0,1},
 {"tc_zero",1,0,0,1,1,3,0,1},
 {"tc_one",1,0,1,1,1,3,0,1},
 {"tc_two",1,0,2,1,1,3,0,1},
 {"tc_negative",1,0,-1,1,1,3,0,1},
 {"tc_last",1,0,2097152,1,1,3,0,1},
 {"tc_first_oob",1,0,2097153,1,1,3,0,1},
 {"tc_max",1,0,std::numeric_limits<std::int64_t>::max(),1,1,3,0,1},
 {"tc_min",1,0,std::numeric_limits<std::int64_t>::min(),1,1,3,0,1},
 {"nt_zero",1,0,6,0,1,3,0,1},
 {"nt_negative",1,0,6,-1,1,3,0,1},
 {"nt_last_token",1,0,6,5,1,3,0,1},
 {"nt_first_oob",1,0,6,6,1,3,0,1},
 {"nt_max",1,0,6,std::numeric_limits<std::int64_t>::max(),1,3,0,1},
 {"nt_min",1,0,6,std::numeric_limits<std::int64_t>::min(),1,3,0,1},
 {"ei_negative",1,0,6,1,1,3,-1,1},
 {"ei_first_oob",1,0,6,1,1,3,256,1},
 {"ei_max",1,0,6,1,1,3,std::numeric_limits<std::int64_t>::max(),1},
 {"et_max",1,0,6,1,std::numeric_limits<std::int64_t>::max(),3,0,1},
 {"vt_max",1,0,6,1,1,std::numeric_limits<std::int64_t>::max(),0,1},
 {"variant_count_zero",1,0,6,1,1,5,0,0},
 {"variant_count_255",1,0,6,1,1,5,0,255},
 {"variant_count_max",1,0,6,1,1,5,0,std::numeric_limits<std::int64_t>::max()},
};}
static bool check(const C&c,std::size_t at,const std::string&line) {
 std::istringstream in(line);std::string tag,extra;std::int64_t id,v,l,variant,reads,oob,kind;
 if(!(in>>tag>>id>>v>>l>>variant>>reads>>oob>>kind)||(in>>extra)||tag!="BOUNDS"||id!=static_cast<std::int64_t>(at)) return false;
 const bool counts=c.en>=0&&c.en<=256&&c.st>=0&&c.st<=1024&&c.tc>=2&&c.tc<=2097152;
 const bool nt=counts&&c.nt>0&&c.nt<c.tc-1;
 const bool nominal=nt&&c.en>0&&c.nt==1&&c.vc>=1&&c.vc<=254;
 const bool validated=nominal&&c.ei==0&&c.tc>=5&&c.vc==1;
 const auto expected_lookup=nominal?0:-1;
 const auto expected_variant=nominal&&c.ei==0&&c.et==1&&c.vt==3?0:-1;
 bool ok=v==(validated?1:0)&&l==expected_lookup&&variant==expected_variant&&oob==0&&kind==0&&reads>=0;
 if(!counts&&reads!=0) ok=false;
 std::cout<<"CASE "<<c.name<<" ok="<<ok<<" reads="<<reads<<" oob="<<oob<<" kind="<<kind<<"\n";
 return ok;
}
int main(int argc,char**argv) {
 auto cs=cases();std::size_t at=0,bad=0,timeouts=0;std::string line,extra;
 if(argc==3&&std::string(argv[1])=="--individual") {
  for(std::size_t i=0;i<cs.size();++i) {
   std::ostringstream n;n<<argv[2]<<"/before-";
   n<<std::setw(3)<<std::setfill('0')<<i;
   std::ifstream status(n.str()+".native.status");int rc;
   if(!(status>>rc))return 3;
   std::ifstream f(n.str()+".native.stdout");
   if(rc!=0||!std::getline(f,line)||std::getline(f,extra)) {
    std::cout<<"CASE "<<cs[i].name<<" ok=0 native="<<rc<<" missing_or_extra=1\n";
    ++bad;if(rc==124)++timeouts;
   } else if(!check(cs[i],i,line)) ++bad;
   ++at;
  }
 } else if(argc==2) {
  std::ifstream f(argv[1]);if(!f)return 3;
  while(std::getline(f,line)) {
   if(at>=cs.size())return 3;
   if(!check(cs[at],at,line))++bad;
   ++at;
  }
 } else return 2;
 if(at!=cs.size())return 3;
 std::cout<<"METADATA_BOUNDS cases="<<at<<" failed="<<bad<<" timeouts="<<timeouts<<"\n";
 return bad?4:0;
}
