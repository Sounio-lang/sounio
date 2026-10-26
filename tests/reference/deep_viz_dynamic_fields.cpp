// C++23 exact integer geometry/pixel oracle for actual native stdout.
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
using Rect=std::tuple<I,I,I,I,I,I,I,I,I,I>;
I packed(I r,I g,I b){return (r<<16)|(g<<8)|b;}
std::vector<Rect> expected(){
 std::vector<Rect> out{
  {0,-1,-1,20,83,2,2,70,180,235},
  {1024,-1,-1,20,51,1,3,70,180,235},
  {2048,-1,-1,51,20,3,2,70,180,235},
  {4095,-1,-1,84,51,1,3,70,180,235}};
 for(I row=0;row<9;++row)for(I col=0;col<11;++col){I index=row*11+col;if(index%13==0)continue;
  // BIDS/colormap bc549477db0c12b54a5928087552ad2cf274980f:
  // high viridis B=0.143936; floor(255*B)=36, independently checked
  // by canonical_palette_reference.cpp exact decimal and binary64 paths.
  I r=index%2?253:68,g=index%2?231:1,b=index%2?36:84;
  out.emplace_back(index,row,col,120+col*8,20+row*8,8,8,r,g,b);
 }
 return out;
}
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("usage: deep_viz_dynamic_fields actual-native-stdout");
 std::ifstream file(argv[1]);if(!file)throw std::runtime_error("input");
 std::string text((std::istreambuf_iterator<char>(file)),{});
 if(text.find("STDLIB_DEEP_VIZ_DYNAMIC_FIELDS_PASS")==std::string::npos)throw std::runtime_error("completion");
 const std::regex re(R"re(<rect data-source-index="([0-9-]+)" data-source-row="([0-9-]+)" data-source-col="([0-9-]+)" x="([0-9.-]+)" y="([0-9.-]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)"/>)re");
 std::vector<Rect> actual;
 for(std::sregex_iterator it(text.begin(),text.end(),re),end;it!=end;++it){const auto& m=*it;
  const double x=std::stod(m[4]),y=std::stod(m[5]);const I ix=static_cast<I>(x+.5),iy=static_cast<I>(y+.5);
  if(x!=static_cast<double>(ix)-.5||y!=static_cast<double>(iy)-.5)throw std::runtime_error("pixel cells");
  actual.emplace_back(std::stoll(m[1]),std::stoll(m[2]),std::stoll(m[3]),ix,iy,std::stoll(m[6]),std::stoll(m[7]),std::stoll(m[8]),std::stoll(m[9]),std::stoll(m[10]));
 }
 const auto wanted=expected();if(actual!=wanted)throw std::runtime_error("SVG geometry/provenance/color/count");
 std::map<std::pair<I,I>,I> pixels;
 for(const auto& [index,row,col,x,y,w,h,r,g,b]:wanted){(void)index;(void)row;(void)col;
  for(I py=y;py<y+h;++py)for(I px=x;px<x+w;++px)pixels[{px,py}]=packed(r,g,b);
 }
 std::map<std::pair<I,I>,I> found;std::istringstream lines(text);std::string line;
 while(std::getline(lines,line)){std::istringstream row(line);std::string tag;row>>tag;if(tag!="FIELD_PIXEL")continue;
  I x=-1,y=-1,color=-1;row>>x>>y>>color;if(!row||!found.emplace(std::pair{x,y},color).second)throw std::runtime_error("native pixel record");
 }
 if(found!=pixels)throw std::runtime_error("exact Canvas pixels/color/scissor");
 if(text.find("data-palette=\"viridis\" data-row-order=\"top-to-bottom\"")==std::string::npos)throw std::runtime_error("heatmap policy metadata");
 std::cout<<"DEEP_VIZ_DYNAMIC_FIELDS_REFERENCE_PASS scatter_rects=4 heat_cells=91 pixels="<<found.size()<<"\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
