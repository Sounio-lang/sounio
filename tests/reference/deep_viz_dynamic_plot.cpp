#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using I=long long;
using Line=std::tuple<I,I,I,I>;
using Rect=std::tuple<I,I,I,I,I>;
I ordinate(I i){return i==256?1:i==768?-1:0;}
bool active(I i){return i<512 || i>=515;}
std::vector<Line> lines(){
 std::set<I> indices{512};
 for(auto segment:{std::pair<I,I>{0,511},{515,1024}}){
  indices.insert(segment.first);indices.insert(segment.second);
  std::map<I,std::vector<I>> groups;
  for(I i=segment.first;i<=segment.second;++i)groups[std::min<I>(3,i/256)].push_back(i);
  for(const auto& [key,v]:groups){(void)key;indices.insert(v.front());indices.insert(v.back());
   auto cmp=[](I a,I b){return ordinate(a)<ordinate(b);};
   indices.insert(*std::min_element(v.begin(),v.end(),cmp));indices.insert(*std::max_element(v.begin(),v.end(),cmp));
  }
 }
 std::vector<Line> result;I previous=-1;
 for(I index:indices){if(previous>=0 && active(previous)&&active(index))result.emplace_back(20+previous*100/1024,60-40*ordinate(previous),20+index*100/1024,60-40*ordinate(index));previous=index;}
 return result;
}
std::vector<Rect> rectangles(){
 std::vector<Rect> result;
 auto add=[&](I part,I x,I y,I w,I h){I right=std::min<I>(261,x+w),bottom=std::min<I>(101,y+h);x=std::max<I>(160,x);y=std::max<I>(20,y);if(right>x&&bottom>y)result.emplace_back(part,x,y,right-x,bottom-y);};
 for(I i=0;i<3;++i){I x=160+50*i,center=100-40*i,lo=center+20,hi=center-20;
  add(0,x,std::max<I>(20,hi),1,std::min<I>(100,lo)-std::max<I>(20,hi)+1);
  if(lo<=100)add(1,x-2,lo,5,1);
  if(hi>=20)add(2,x-2,hi,5,1);
  add(3,x-1,center-1,3,3);
 }
 return result;
}
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("usage: checker actual-native-stdout");
 std::ifstream file(argv[1]);if(!file)throw std::runtime_error("input");
 std::string all((std::istreambuf_iterator<char>(file)),{});
 if(all.find("STDLIB_DEEP_VIZ_DYNAMIC_PLOT_PASS")==std::string::npos)throw std::runtime_error("completion");
 std::regex lr(R"re(<line data-dynamic-part="line" x1="([0-9-]+)" y1="([0-9-]+)" x2="([0-9-]+)" y2="([0-9-]+)" stroke="rgb\(70,180,235\)" stroke-width="1" stroke-linecap="round"/>)re");
 std::vector<Line> got;
 for(std::sregex_iterator i(all.begin(),all.end(),lr),end;i!=end;++i)got.emplace_back(std::stoll((*i)[1]),std::stoll((*i)[2]),std::stoll((*i)[3]),std::stoll((*i)[4]));
 if(got!=lines())throw std::runtime_error("line provenance/geometry");
 std::regex rr(R"re(<rect data-dynamic-interval-part="([0-3])" x="([0-9.-]+)" y="([0-9.-]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(230,120,40\)"/>)re");
 std::vector<Rect> actual;for(std::sregex_iterator i(all.begin(),all.end(),rr),end;i!=end;++i){const auto& m=*i;double x=std::stod(m[2]),y=std::stod(m[3]);I ix=static_cast<I>(x+.5),iy=static_cast<I>(y+.5);if(x!=static_cast<double>(ix)-.5||y!=static_cast<double>(iy)-.5)throw std::runtime_error("half-cell");actual.emplace_back(std::stoll(m[1]),ix,iy,std::stoll(m[4]),std::stoll(m[5]));}
 const auto expected=rectangles();if(actual!=expected)throw std::runtime_error("interval footprint");
 const std::string clip1="<rect x=\"19.500000\" y=\"19.500000\" width=\"101\" height=\"81\"/>";
 const std::string clip2="<rect x=\"159.500000\" y=\"19.500000\" width=\"101\" height=\"81\"/>";
 if(all.find(clip1)==std::string::npos||all.find(clip2)==std::string::npos)throw std::runtime_error("clip cells");
 if(all.find("data-interval-kind=\"4\" data-interval-level=\"0.000000\"")==std::string::npos)throw std::runtime_error("uncertainty metadata");
 std::set<std::pair<I,I>> expected_pixels,actual_pixels;
 for(const auto& [part,x,y,w,h]:expected){(void)part;for(I py=y;py<y+h;++py)for(I px=x;px<x+w;++px)expected_pixels.emplace(px,py);}
 I blue_count=0;std::istringstream stream(all);std::string row;
 while(std::getline(stream,row)){std::istringstream in(row);std::string tag;in>>tag;if(tag!="DYNAMIC_PIXEL")continue;I x,y,color;in>>x>>y>>color;if(!in)throw std::runtime_error("pixel parse");
  if(color==((230LL<<16)|(120LL<<8)|40)){if(!actual_pixels.emplace(x,y).second)throw std::runtime_error("duplicate pixel");}
  else if(x>=20&&x<=120&&y>=20&&y<=100){if(color<=0 || ((color>>16)&255)>70 || ((color>>8)&255)>180 || (color&255)>235)throw std::runtime_error("line AA color bounds");if(color==((70LL<<16)|(180LL<<8)|235))++blue_count;}
  else throw std::runtime_error("unexpected pixel color");
 }
 if(actual_pixels!=expected_pixels||blue_count==0)throw std::runtime_error("Canvas footprint");
 std::cout<<"DEEP_VIZ_DYNAMIC_PLOT_REFERENCE_PASS line_segments="<<got.size()<<" interval_rectangles="<<actual.size()<<" interval_pixels="<<actual_pixels.size()<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
