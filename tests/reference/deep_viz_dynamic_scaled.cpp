#include <algorithm>
#include <cmath>
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
using Rect=std::tuple<I,I,I,I,I>;
const long double xv[]={1,2,5,-1,20,50,101,1000};
const long double yv[]={-9,-3,-1,0,1,3,9,0};
long double sy(long double y){return y<0?-std::log1p(-y):std::log1p(y);}
I px(long double x,I origin){if(x==1)return origin;if(x==1000)return origin+180;return origin+static_cast<I>(180*std::log(x)/std::log(1000.L));}
I py(long double y){if(y==-9)return 120;if(y==9)return 20;if(y==0)return 70;return 120-static_cast<I>(100*(sy(y)+std::log(10.L))/(2*std::log(10.L)));}
std::vector<I> selection(){std::set<I> out{3};
 for(auto range:{std::pair<I,I>{0,2},{4,7}}){std::map<I,std::vector<I>> groups;
  for(I i=range.first;i<=range.second;++i){I bin=xv[i]==1000?2:static_cast<I>(3*std::log(xv[i])/std::log(1000.L));groups[bin].push_back(i);}
  out.insert(range.first);out.insert(range.second);
  for(auto& [key,v]:groups){(void)key;out.insert(v.front());out.insert(v.back());auto cmp=[](I a,I b){return yv[a]<yv[b];};out.insert(*std::min_element(v.begin(),v.end(),cmp));out.insert(*std::max_element(v.begin(),v.end(),cmp));}
 }return {out.begin(),out.end()};}
std::vector<Rect> rectangles(){std::vector<Rect> out;
 auto add=[&](I part,I x,I y,I w,I h){I right=std::min<I>(681,x+w),bottom=std::min<I>(121,y+h);x=std::max<I>(500,x);y=std::max<I>(20,y);if(right>x&&bottom>y)out.emplace_back(part,x,y,right-x,bottom-y);};
 const long double x[]={1,20,1000},m[]={-9,0,9},lo[]={-19,-1,3},hi[]={-3,1,19};
 for(I i=0;i<3;++i){I center=px(x[i],500);I top=hi[i]>9?20:py(hi[i]),bottom=lo[i]<-9?120:py(lo[i]);add(0,center,top,1,bottom-top+1);if(lo[i]>=-9)add(1,center-2,bottom,5,1);if(hi[i]<=9)add(2,center-2,top,5,1);add(3,center-1,py(m[i])-1,3,3);}
 return out;}
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("usage: checker actual-native-stdout");std::ifstream f(argv[1]);if(!f)throw std::runtime_error("input");std::string all((std::istreambuf_iterator<char>(f)),{});
 if(all.find("STDLIB_DEEP_VIZ_DYNAMIC_SCALED_PASS")==std::string::npos)throw std::runtime_error("completion");
 auto chosen=selection();std::vector<I> actual_indices;std::istringstream rows(all);std::string row;
 while(std::getline(rows,row)){std::istringstream r(row);std::string tag;r>>tag;if(tag=="SCALED_INDEX"){I ordinal,index;r>>ordinal>>index;if(!r||ordinal!=static_cast<I>(actual_indices.size()))throw std::runtime_error("ordinal");actual_indices.push_back(index);}}
 if(actual_indices!=chosen)throw std::runtime_error("transformed-X bucket provenance");
 std::vector<std::tuple<I,I,I,I>> expected_lines;
 for(std::size_t i=1;i<chosen.size();++i){I a=chosen[i-1],b=chosen[i];if(a==3||b==3)continue;expected_lines.emplace_back(px(xv[a],100),py(yv[a]),px(xv[b],100),py(yv[b]));}
 const std::regex lr(R"re(<line data-dynamic-part="line" x1="([0-9-]+)" y1="([0-9-]+)" x2="([0-9-]+)" y2="([0-9-]+)" stroke="rgb\(70,180,235\)" stroke-width="1" stroke-linecap="round"/>)re");
 std::vector<std::tuple<I,I,I,I>> got_lines;for(std::sregex_iterator it(all.begin(),all.end(),lr),end;it!=end;++it)got_lines.emplace_back(std::stoll((*it)[1]),std::stoll((*it)[2]),std::stoll((*it)[3]),std::stoll((*it)[4]));
 if(got_lines!=expected_lines)throw std::runtime_error("log/symlog line endpoint geometry");
 const std::regex rr(R"re(<rect data-dynamic-interval-part="([0-3])" x="([0-9.-]+)" y="([0-9.-]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(230,120,40\)"/>)re");
 std::vector<Rect> got;for(std::sregex_iterator it(all.begin(),all.end(),rr),end;it!=end;++it){const auto&m=*it;double x=std::stod(m[2]),y=std::stod(m[3]);I ix=static_cast<I>(x+.5),iy=static_cast<I>(y+.5);if(x!=static_cast<double>(ix)-.5||y!=static_cast<double>(iy)-.5)throw std::runtime_error("half cell");got.emplace_back(std::stoll(m[1]),ix,iy,std::stoll(m[4]),std::stoll(m[5]));}
 const auto wanted=rectangles();if(got!=wanted)throw std::runtime_error("transformed interval geometry");
 std::set<std::pair<I,I>> expected_pixels,orange;
 for(const auto&[part,x,y,w,h]:wanted){(void)part;for(I yy=y;yy<y+h;++yy)for(I xx=x;xx<x+w;++xx)expected_pixels.emplace(xx,yy);}
 std::istringstream pixels(all);I blue=0;
 while(std::getline(pixels,row)){std::istringstream r(row);std::string tag;r>>tag;if(tag!="SCALED_PIXEL")continue;I x,y,color;r>>x>>y>>color;if(!r)throw std::runtime_error("pixel record");if(color==((230LL<<16)|(120LL<<8)|40))orange.emplace(x,y);else if(color==((180LL<<16)|(180LL<<8)|180)){}else{if(x<100||x>280||y<20||y>120||color<=0||((color>>16)&255)>70||((color>>8)&255)>180||(color&255)>235)throw std::runtime_error("line Canvas scissor/color");++blue;}}
 if(orange!=expected_pixels||blue==0)throw std::runtime_error("Canvas transformed footprints");
 if(all.find("data-x-scale=\"1\" data-y-scale=\"2\"")==std::string::npos||all.find("data-axis-layout=\"fitted\"")==std::string::npos)throw std::runtime_error("shared scale metadata/axis");
 std::cout<<"DEEP_VIZ_DYNAMIC_SCALED_REFERENCE_PASS indices="<<chosen.size()<<" line_segments="<<got_lines.size()<<" interval_rects="<<got.size()<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
