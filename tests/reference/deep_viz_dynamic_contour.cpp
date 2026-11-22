#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using I=long long;
using Segment=std::tuple<I,I,I,I,I,I,I>;
std::vector<Segment> expected(){std::vector<Segment> out;
 auto add=[&](I panel,I cell,I branch,I x0,I y0,I x1,I y1){I origin=20+panel*80;out.emplace_back(cell,0,branch,origin+x0,y0,origin+x1,y1);};
 add(0,0,0,30,80,30,20);add(0,0,1,60,50,0,50);
 add(1,0,0,45,80,60,65);add(1,0,1,15,20,0,35);
 add(2,0,0,15,80,0,65);add(2,0,1,60,35,45,20);
 add(3,0,0,0,80,60,80);add(3,0,1,30,20,30,80);
 add(4,0,0,0,80,60,80);add(4,0,1,0,20,0,80);
 add(6,0,0,30,80,30,20);add(7,1,0,30,20,30,80);add(8,1,0,30,20,30,80);
 add(10,0,0,30,80,30,20);
 return out;}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage: checker actual-native-stdout");
 std::ifstream file(argv[1]);if(!file)throw std::runtime_error("input");std::string all((std::istreambuf_iterator<char>(file)),{});
 if(all.find("STDLIB_DEEP_VIZ_DYNAMIC_CONTOUR_PASS")==std::string::npos)throw std::runtime_error("completion");
 const std::regex re(R"re(<line data-contour-cell="([0-9]+)" data-contour-level-index="([0-9]+)" data-contour-branch="([0-9]+)" data-contour-level="[^"]+" x1="([0-9-]+)" y1="([0-9-]+)" x2="([0-9-]+)" y2="([0-9-]+)" stroke="rgb\(70,180,235\)" stroke-width="1" stroke-linecap="round"/>)re");
 std::vector<Segment> actual;for(std::sregex_iterator it(all.begin(),all.end(),re),end;it!=end;++it){const auto&m=*it;actual.emplace_back(std::stoll(m[1]),std::stoll(m[2]),std::stoll(m[3]),std::stoll(m[4]),std::stoll(m[5]),std::stoll(m[6]),std::stoll(m[7]));}
 const auto wanted=expected();if(actual!=wanted)throw std::runtime_error("exact saddle/flat-edge/T/L/ownership geometry");
 std::map<std::pair<I,I>,I> pixels,found;const I rgb=(70LL<<16)|(180LL<<8)|235;
 for(const auto&[cell,level,branch,ax,ay,bx,by]:wanted){(void)cell;(void)level;(void)branch;I dx=bx-ax,dy=by-ay;I steps=std::max(std::llabs(dx),std::llabs(dy));if(dx!=0&&dy!=0&&std::llabs(dx)!=std::llabs(dy))throw std::runtime_error("nonaxis/nondyadic fixture line");for(I i=0;i<=steps;++i)pixels[{ax+dx*i/steps,ay+dy*i/steps}]=rgb;}
 const I counts[]={2,2,2,2,2,0,1,1,1,0,1};
 std::istringstream rows(all);std::string row;I cases=0;
 while(std::getline(rows,row)){std::istringstream in(row);std::string tag;in>>tag;
  if(tag=="CONTOUR_PIXEL"){I x,y,color;in>>x>>y>>color;if(!in||!found.emplace(std::pair{x,y},color).second)throw std::runtime_error("pixel record");}
  if(tag=="CONTOUR_CASE"){I id,required,painted,flat,degenerate,masked;in>>id>>required>>painted>>flat>>degenerate>>masked;if(!in||id!=cases||id<0||id>=11)throw std::runtime_error("case record");if(required!=counts[id]||painted!=counts[id]||flat!=((id==5||id==8)?1:0)||degenerate!=(id==9?1:0)||masked!=(id==7?1:0))throw std::runtime_error("contour counters");++cases;}
 }
 if(found!=pixels||cases!=11)throw std::runtime_error("exact Canvas pixels/scissor/count");
 if(all.find("data-classification=\"high-ge-level\" data-saddle=\"exact-dyadic\"")==std::string::npos)throw std::runtime_error("predicate metadata");
 std::cout<<"DEEP_VIZ_DYNAMIC_CONTOUR_REFERENCE_PASS panels=11 segments="<<wanted.size()<<" pixels="<<found.size()<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
