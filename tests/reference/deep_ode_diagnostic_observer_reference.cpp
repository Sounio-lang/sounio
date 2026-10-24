// C++23 independent analytic A->B model, dyadic observations and actual SVG grammar.
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
struct Row{long double t,q,g,absolute,relative,flags;};
void need(bool ok,const char*why){if(!ok){throw std::runtime_error(why);}}
bool finite(const Row&r){return std::isfinite(r.t)&&std::isfinite(r.q)&&std::isfinite(r.g)&&std::isfinite(r.absolute)&&std::isfinite(r.relative)&&std::isfinite(r.flags);}
std::map<std::string,std::string> attributes(const std::string&s){
 std::map<std::string,std::string> out;const std::regex re(R"ATTR(([a-zA-Z0-9_-]+)="([^"]*)")ATTR");
 for(auto i=std::sregex_iterator(s.begin(),s.end(),re);i!=std::sregex_iterator();++i){need(out.emplace((*i)[1],(*i)[2]).second,"duplicate SVG attribute");}return out;
}
double number(const std::string&s){std::size_t n=0;const auto x=std::stod(s,&n);need(n==s.size()&&std::isfinite(x),"invalid SVG number");return x;}
void svg_check(const std::string&s,std::size_t n){
 need(s.find("<svg xmlns=")!=std::string::npos&&s.find("</svg>")!=std::string::npos,"missing SVG root");
 need(s.find("data-interval-kind=\"none\"")!=std::string::npos,"unexpected interval claim");
 for(int panel=0;panel<2;++panel){
  const auto start=s.find("<g id=\"scientific-panel-"+std::to_string(panel)+"\">");need(start!=std::string::npos,"missing panel");
  const auto end=panel==0?s.find("<g id=\"scientific-panel-1\">",start):s.find("<metadata data-scientific-document-complete=",start);
  need(end!=std::string::npos&&end>start,"missing panel delimiter");const auto body=s.substr(start,end-start);
  need(body.find("data-dynamic-series=\"scatter-3x3\"")!=std::string::npos,"wrong series kind");
  std::set<long long> seen;const std::regex re(R"(<rect\b[^>]*>)");
  for(auto i=std::sregex_iterator(body.begin(),body.end(),re);i!=std::sregex_iterator();++i){
   const auto a=attributes(i->str());need(a.contains("data-source-index"),"nonglyph rect in panel");const auto index=std::stoll(a.at("data-source-index"));
   need(index>=0&&index<static_cast<long long>(n)&&seen.insert(index).second,"wrong/duplicate glyph index");
   need(a.at("data-source-row")=="-1"&&a.at("data-source-col")=="-1","wrong source provenance");
   const auto w=number(a.at("width")),h=number(a.at("height"));need(w>=1&&w<=3&&h>=1&&h<=3&&std::floor(w)==w&&std::floor(h)==h,"bad clipped pixel glyph");
   number(a.at("x"));number(a.at("y"));need(a.at("fill")== (panel==0?"rgb(70,180,235)":"rgb(235,130,70)"),"wrong panel color");
  }
  need(seen.size()==n,"omitted glyph");
 }
 const std::regex meta(R"(<metadata data-scientific-document-complete="true"[^>]*>)");std::smatch match;need(std::regex_search(s,match,meta),"document completion absent");const auto a=attributes(match.str());
 for(const std::string side:{"left","right"}){need(a.at("data-"+side+"-input")==std::to_string(n)&&a.at("data-"+side+"-masked")=="0"&&a.at("data-"+side+"-outside")=="0","availability metadata differs");}
 // Right zero reference overlaps its y=0 baseline. Both complete horizontal
 // pixel-cell rectangles must be present, not merely one baseline rectangle.
 std::size_t zero=0;const std::regex rect(R"(<rect\b[^>]*>)");
 for(auto i=std::sregex_iterator(s.begin(),s.end(),rect);i!=std::sregex_iterator();++i){const auto r=attributes(i->str());if(!r.contains("data-document-axis")){continue;}if(number(r.at("x"))>690&&number(r.at("y"))==479.5&&number(r.at("height"))==1&&number(r.at("width"))>100){++zero;}}
 need(zero>=2,"right zero reference omitted");
}
int main(int argc,char**argv){try{
 const bool figure=argc==3&&std::string(argv[1])=="--figure";if(argc!=(figure?3:2)){return 2;}
 std::ifstream input(argv[figure?2:1]);need(bool(input),"missing native stdout");
 std::string line,all;std::vector<Row> rows;bool complete=false,bracket=false;long double left=0,right=0;std::int64_t kind=0,declared=0;
 while(std::getline(input,line)){
  all+=line+'\n';if(!figure&&line=="ODE_DIAGNOSTIC_OBSERVER_PASS"){complete=true;}if(figure&&line=="<!-- ODE_DIAGNOSTICS_COMPLETE -->"){complete=true;}
  const std::string prefix=figure?"<!-- ODE_RECORD ":"ODEOBS ";
  if(line.starts_with(prefix)){std::istringstream stream(line.substr(prefix.size()));std::int64_t i;Row r{};need(bool(stream>>i>>r.t>>r.q>>r.g>>r.absolute>>r.relative>>r.flags)&&finite(r)&&i==std::int64_t(rows.size()),"malformed observation");rows.push_back(r);}
  if(line.starts_with("<!-- ODE_BRACKET ")){need(!bracket,"duplicate bracket");bracket=true;std::istringstream stream(line.substr(17));need(bool(stream>>left>>right>>kind>>declared),"malformed bracket");}
 }
 need(complete,"missing completion");
 if(!figure){
  const std::array<Row,3> expected{{{0,2,1,0,0,3},{1,2.5L,-.5L,.5L,.25L,11},{2,3,0,1,.5L,7}}};need(rows.size()==expected.size(),"wrong dyadic record count");
  for(std::size_t i=0;i<rows.size();++i){const auto a=rows[i],e=expected[i];need(a.t==e.t&&a.q==e.q&&a.g==e.g&&a.absolute==e.absolute&&a.relative==e.relative&&a.flags==e.flags,"dyadic observation mismatch");}
  std::cout<<"ODE_OBSERVER_CPP_CONCORDANCE_PASS records=3 exact_dyadic=1\n";return 0;
 }
 need(bracket&&kind==2&&declared==std::int64_t(rows.size())&&rows.size()>=3,"bracket/count mismatch");
 need(rows.front().t==0&&std::abs(rows.back().t-1)<=1e-14L,"endpoint mismatch");bool found=false;
 for(std::size_t i=0;i<rows.size();++i){const auto&r=rows[i];
  if(i>0){need(r.t>rows[i-1].t,"unordered time");}
  need(std::abs(r.q-1)<=1e-10L&&std::abs(r.g-(std::exp(-2*r.t)-.5L))<=1e-6L,"analytic transfer mismatch");
  need(std::abs(r.absolute-std::abs(r.q-1))<=2e-15L&&std::abs(r.relative-r.absolute)<=2e-15L,"drift mismatch");
  need(r.flags>=0&&r.flags<=15&&std::floor(r.flags)==r.flags,"invalid flags");const auto flags=std::int64_t(r.flags);need((flags&3)==3,"missing drift availability");
  if(i>0&&rows[i-1].g>0&&r.g<0){need(!found,"multiple transfer crossings");found=true;need(std::abs(left-rows[i-1].t)<=1e-14L&&std::abs(right-r.t)<=1e-14L,"bracket is not accepted sample pair");}
 }
 const auto analytic=std::log(2.L)/2;need(found&&analytic>=left-2e-6L&&analytic<=right+2e-6L,"analytic event not bracketed");svg_check(all,rows.size());
 std::cout<<"ODE_DIAGNOSTIC_FIGURE_CPP_PASS records="<<rows.size()<<" glyphs="<<2*rows.size()<<" sampled_bracket=1 analytic_transfer=1\n";return 0;
 }catch(const std::exception&e){std::cerr<<"ODE_DIAGNOSTIC_CPP_FAIL "<<e.what()<<'\n';return 1;}}
