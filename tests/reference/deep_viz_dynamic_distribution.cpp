#include <algorithm>
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
using Rect=std::tuple<I,I,I,I,I,I,I,I,I>;
std::vector<Rect> expected(){
 std::vector<Rect> out{{2,0,40,40,40,60,70,180,235},{4,0,100,80,20,20,70,180,235}};
 auto add=[&](I index,I part,I x,I y,I w,I h){I right=std::min<I>(261,x+w),bottom=std::min<I>(101,y+h);x=std::max<I>(160,x);y=std::max<I>(20,y);if(right>x&&bottom>y)out.emplace_back(index,part,x,y,right-x,bottom-y,230,120,40);};
 const I q1[]={10,30,70},m[]={20,40,75},q3[]={30,50,80},lo[]={-20,10,60},hi[]={40,70,100};
 // Coordinates are exact tenths: y=100-value for this fixture.
 for(I i=0;i<3;++i){I center=160+50*i;
  add(i,0,center,std::max<I>(20,100-hi[i]),1,std::min<I>(100,100-lo[i])-std::max<I>(20,100-hi[i])+1);
  if(lo[i]>=0&&lo[i]<=80)add(i,1,center-4,100-lo[i],9,1);
  if(hi[i]>=0&&hi[i]<=80)add(i,2,center-4,100-hi[i],9,1);
  if(q1[i]>=0&&q1[i]<=80)add(i,3,center-8,100-q1[i],17,1);
  if(q3[i]>=0&&q3[i]<=80)add(i,4,center-8,100-q3[i],17,1);
  add(i,5,center-8,100-q3[i],1,q3[i]-q1[i]+1);
  add(i,6,center+8,100-q3[i],1,q3[i]-q1[i]+1);
  if(m[i]>=0&&m[i]<=80)add(i,7,center-8,100-m[i],17,1);
 }
 return out;
}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage: checker actual-native-stdout");
 std::ifstream file(argv[1]);if(!file)throw std::runtime_error("input");std::string all((std::istreambuf_iterator<char>(file)),{});
 if(all.find("STDLIB_DEEP_VIZ_DYNAMIC_DISTRIBUTION_PASS")==std::string::npos)throw std::runtime_error("completion");
 const std::regex re(R"re(<rect data-distribution-index="([0-9]+)" data-distribution-part="([0-7])" x="([0-9.-]+)" y="([0-9.-]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)"/>)re");
 std::vector<Rect> actual;for(std::sregex_iterator it(all.begin(),all.end(),re),end;it!=end;++it){const auto&m=*it;double x=std::stod(m[3]),y=std::stod(m[4]);I ix=static_cast<I>(x+.5),iy=static_cast<I>(y+.5);if(x!=static_cast<double>(ix)-.5||y!=static_cast<double>(iy)-.5)throw std::runtime_error("pixel cells");actual.emplace_back(std::stoll(m[1]),std::stoll(m[2]),ix,iy,std::stoll(m[5]),std::stoll(m[6]),std::stoll(m[7]),std::stoll(m[8]),std::stoll(m[9]));}
 const auto wanted=expected();if(actual!=wanted)throw std::runtime_error("histogram/box rectangle geometry/provenance/color/count");
 std::map<std::pair<I,I>,I> pixels,found;
 for(const auto&[index,part,x,y,w,h,r,g,b]:wanted){(void)index;(void)part;for(I yy=y;yy<y+h;++yy)for(I xx=x;xx<x+w;++xx)pixels[{xx,yy}]=(r<<16)|(g<<8)|b;}
 std::istringstream rows(all);std::string row;while(std::getline(rows,row)){std::istringstream in(row);std::string tag;in>>tag;if(tag!="DISTRIBUTION_PIXEL")continue;I x,y,color;in>>x>>y>>color;if(!in||!found.emplace(std::pair{x,y},color).second)throw std::runtime_error("pixel record");}
 if(found!=pixels)throw std::runtime_error("exact Canvas geometry/color/transactionality");
 if(all.find("data-height-kind=\"count\" data-normalization=\"supplied\"")==std::string::npos||all.find("data-quantile-method=\"7\" data-whisker-policy=\"2\" data-interval-kind=\"none\"")==std::string::npos)throw std::runtime_error("scientific metadata");
 std::cout<<"DEEP_VIZ_DYNAMIC_DISTRIBUTION_REFERENCE_PASS hist_rects=2 box_rects="<<wanted.size()-2<<" pixels="<<found.size()<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
