// Independent numeric reference for the actual correlation-permutation figure.
// Reuses the independently implemented C++ oracle, never Sounio/native output.
// The renamed CLI entry point is not called.
#include <fstream>
#include <regex>
#include <iterator>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#define main correlation_permutation_reference_unused_cli
#include "correlation_permutation_reference.cpp"
#undef main
#pragma GCC diagnostic pop
int main(int argc,char**argv) {
 V x={0,0,1,2,3},y={0,1,1,2,3};
 std::vector<Case> rows={
  {"exact",x,y,5,2,0,0,true},
  {"mc64seed7",x,y,5,2,64,7,false},
  {"mc256seed11",x,y,5,2,256,11,false},
  {"mc1024seed19",x,y,5,2,1024,19,false}
 };
 std::cout<<std::scientific<<std::setprecision(17);
 std::cout<<"row\thits\tdraws\tseed\tcoefficient\tcorrected_p\tmc_se\twilson_lo\twilson_hi\n";
 for(const auto& c:rows) {
  auto r=eval(c);
  if(r.status!=0||r.p<=0||r.p>1)return 1;
  if(c.exact&&(r.draws!=120||r.mc||r.seed))return 2;
  if(!c.exact&&(!r.mc||r.lo<0||r.hi>1||r.lo>(Q)r.hits/r.draws||r.hi<(Q)r.hits/r.draws))return 3;
  std::cout<<c.name<<"\t"<<r.hits<<"\t"<<r.draws<<"\t"<<r.seed<<"\t"<<(double)r.coefficient<<"\t"<<(double)r.p<<"\t"<<(double)r.se<<"\t"<<(double)r.lo<<"\t"<<(double)r.hi<<"\n";
 }
 if(argc>1) {
  std::ifstream file(argv[1]);if(!file){std::cerr<<"GEOMETRY_FAIL input\n";return 10;}
  std::string svg((std::istreambuf_iterator<char>(file)),{});
  int checks=0;
  auto require=[&](bool ok,const std::string& what){if(!ok){std::cerr<<"GEOMETRY_FAIL "<<what<<"\n";return false;}++checks;return true;};
  auto has=[&](const std::string&type,const std::vector<std::pair<std::string,std::string>>& attrs) {
   std::regex tag("<"+type+"\\b[^>]*>");
   for(std::sregex_iterator it(svg.begin(),svg.end(),tag),end;it!=end;++it) {
    std::string s=it->str();bool ok=true;
    for(const auto&[key,value]:attrs)ok&=s.find(key+"=\""+value+"\"")!=std::string::npos;
    if(ok)return true;
   }return false;
  };
  if(!require(has("rect",{{"x","180"},{"y","120"},{"width","560"},{"height","88"},{"stroke","rgb(90,95,120)"}}),"plot bounds"))return 11;
  if(!require(has("line",{{"x1","180"},{"x2","740"},{"y1","226"},{"y2","226"},{"stroke","rgb(190,200,215)"}}),"probability axis span"))return 12;
  for(int j=0;j<=4;++j){auto x=std::to_string(180+140*j);if(!require(has("line",{{"x1",x},{"x2",x},{"y1","226"},{"y2","230"},{"stroke","rgb(190,200,215)"}}),"axis tick "+std::to_string(j)))return 13;}
  if(!require(svg.find(">0</text>")!=std::string::npos&&svg.find(">1.00000e+0</text>")!=std::string::npos,"domain labels 0 and 1"))return 14;
  if(!require(svg.find("data-interval-kind=\"confidence\"")!=std::string::npos,"interval metadata"))return 15;
  auto numeric_label=[&](const std::string& prefix,Q expected,int y) {
   std::regex text("<text\\b[^>]*y=\""+std::to_string(y)+"\"[^>]*>"+prefix+"([^<]+)</text>");
   std::smatch m;if(!std::regex_search(svg,m,text))return false;
   Q value=(Q)std::stod(m[1].str());return fabsq(value-expected)<=fabsq(expected)*(Q)5.1e-6;
  };
  for(int i=1;i<4;++i) {
   auto r=eval(rows[i]);int y=131+22*(i-1);
   int left=(int)floorq(180+560*r.lo),right=(int)floorq(180+560*r.hi);
   int center=(int)floorq(180+560*(Q)r.hits/r.draws);
   if(!require(has("line",{{"x1",std::to_string(left)},{"x2",std::to_string(right)},{"y1",std::to_string(y)},{"y2",std::to_string(y)},{"stroke","rgb(180,180,220)"}}),"Wilson interval "+std::to_string(i)))return 16;
   if(!require(has("rect",{{"x",std::to_string(center-3)},{"y",std::to_string(y-3)},{"width","7"},{"height","7"},{"fill","rgb(80,180,240)"}}),"raw hit center "+std::to_string(i)))return 17;
   if(!require(numeric_label("corrected p=",r.p,y),"corrected p label "+std::to_string(i)))return 18;
   std::cout<<"GEOMETRY_ROW\t"<<i<<"\t"<<left<<"\t"<<center<<"\t"<<right<<"\n";
  }
  auto exact=eval(rows[0]);Q theoretical=180+560*exact.p;int projection=(int)floorq(theoretical);
  if(!require(has("path",{{"stroke","rgb(220,135,225)"},{"d","M"+std::to_string(projection)+" 208 L"+std::to_string(projection)+" 120 "}}),"conditional exact line"))return 19;
  if(!require(numeric_label("Exact conditional tail p=",exact.p,405),"exact p label"))return 20;
  if(!require(svg.find("Wilson 95% intervals for q=hits/B; not correlation CIs")!=std::string::npos,"Wilson semantics"))return 21;
  if(!require(svg.find("Exchangeability unassessed; exact line is conditional")!=std::string::npos,"exchangeability semantics"))return 22;
  std::cout<<"EXACT_GEOMETRY\t"<<(double)theoretical<<"\t"<<projection<<"\n";
  std::cout<<"SVG_GEOMETRY_PASS\t"<<checks<<"\n";
 }
 return 0;
}
