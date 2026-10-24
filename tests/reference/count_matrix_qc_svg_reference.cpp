// Independent C++23 count TSV/QC and SVG geometry reference.
// Run only through the coordinator's approved remote execution path.
#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
struct Mark {int x,y,r,g,b;};
[[noreturn]] void fail(const std::string&s){throw std::runtime_error(s);}
int main(int argc,char**argv){
 try{
  if(argc!=2)fail("usage: count_matrix_qc_svg_reference SVG_OR_NATIVE_LOG");
  const std::string raw="0\t2\t4\r\n3\t0\t0\n0\t6\t0\r\n5\t7\t8";
  std::array<std::int64_t,3> totals{},zeros{};
  std::istringstream stream(raw);std::string line;std::int64_t rows=0;
  while(std::getline(stream,line)){
   if(!line.empty()&&line.back()=='\r')line.pop_back();
   std::size_t begin=0;
   for(int j=0;j<3;j++){
    auto end=line.find('\t',begin);
    if((j<2&&end==line.npos)||(j==2&&end!=line.npos))fail("reference columns");
    if(end==line.npos)end=line.size();
    std::int64_t count=0;auto [p,e]=std::from_chars(line.data()+begin,line.data()+end,count);
    if(e!=std::errc{}||p!=line.data()+end||begin==end||count<0)fail("reference count");
    totals[j]+=count;zeros[j]+=(count==0);begin=end+1;
   }
   ++rows;
  }
  if(rows!=4||totals!=std::array<std::int64_t,3>{8,15,12}||
     zeros!=std::array<std::int64_t,3>{2,1,2})fail("independent QC");
  std::ifstream in(argv[1]);if(!in)fail("cannot open SVG/log");
  std::string svg((std::istreambuf_iterator<char>(in)),{});
  auto start=svg.find("<svg ");auto end=svg.find("</svg>",start);
  if(start==svg.npos||end==svg.npos)fail("SVG bounds");
  svg=svg.substr(start,end+6-start);
  for(const std::string text:{
   "Count matrix: sample totals and zero fractions",
   "Counts parsed incrementally from synthetic TSV bytes",
   "Sample totals (counts)",
   "Zero fraction = zero counts / 4 feature rows",
   "No intervals or biological quality assessment",
   "Integer parsing exact; plotted i64 &gt; 2^53 may round",
   "sample index (declared order)","total (counts)","zero fraction (unitless)"})
    if(svg.find(text)==svg.npos)fail("missing label: "+text);
  if(svg.find("data-interval-kind=")!=svg.npos)fail("unexpected interval metadata");
  std::regex circle(R"rx(<circle cx="(-?[0-9]+)" cy="(-?[0-9]+)" r="([0-9]+)" fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)"/>)rx");
  std::vector<Mark> expected;
  // All physical coordinates are exact integer/rational projections here.
  // First panel: [0,4]x[0,20]; second: [0,4]x[0,1].
  for(int j=0;j<3;j++)expected.push_back({140+480*(j+1)/4,390-int(240*totals[j]/20),70,180,235});
  for(int j=0;j<3;j++)expected.push_back({800+480*(j+1)/4,390-int(240*zeros[j]/rows),245,170,65});
  std::size_t count=0;
  for(auto i=std::sregex_iterator(svg.begin(),svg.end(),circle);i!=std::sregex_iterator();++i){
   if(count>=expected.size())fail("excess circle");
   auto m=*i;const auto&e=expected[count];
   if(std::stoi(m[1])!=e.x||std::stoi(m[2])!=e.y||std::stoi(m[3])!=3||
      std::stoi(m[4])!=e.r||std::stoi(m[5])!=e.g||std::stoi(m[6])!=e.b)
      fail("circle geometry/color/radius "+std::to_string(count));
   ++count;
  }
  if(count!=6)fail("missing circle");
  // Catch circles with unexpected attribute/radius syntax rather than ignoring them.
  std::size_t p=0,all=0;while((p=svg.find("<circle",p))!=svg.npos){++all;p+=7;}
  if(all!=6)fail("unexpected circle markup/count");
  p=0;all=0;while((p=svg.find("<!-- scatter n=3 -->",p))!=svg.npos){++all;p+=20;}
  if(all!=2)fail("scatter metadata");
  std::cout<<"COUNT_MATRIX_QC_SVG_REFERENCE_PASS rows="<<rows<<" circles="<<count<<"\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<"REJECTED: "<<e.what()<<"\n";return 2;}
}
