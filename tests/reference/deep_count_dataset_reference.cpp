// C++23 exact keyed join (std::map, independently of index heapsort).
// --workflow validates actual SVG grammar/status/provenance, not profile calibration.
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
void need(bool ok,const char*why){if(!ok){throw std::runtime_error(why);}}
std::map<std::string,std::string> attributes(const std::string&tag){
 std::map<std::string,std::string> a;
 const std::regex pattern(R"ATTR(([a-zA-Z0-9_-]+)="([^"]*)")ATTR");
 for(auto i=std::sregex_iterator(tag.begin(),tag.end(),pattern);i!=std::sregex_iterator();++i){need(a.emplace((*i)[1],(*i)[2]).second,"duplicate attribute");}
 return a;
}
double numeric(const std::string&s){std::size_t n=0;const auto v=std::stod(s,&n);need(n==s.size()&&std::isfinite(v),"invalid numeric attribute");return v;}
void panel(const std::string&svg,int which,const std::set<long long>&expected){
 const std::string marker="<g id=\"scientific-panel-"+std::to_string(which)+"\">";
 const auto start=svg.find(marker);need(start!=std::string::npos,"missing panel");
 const auto stop=which==0?svg.find("<g id=\"scientific-panel-1\">",start):svg.find("<metadata data-scientific-document-complete=",start);
 need(stop!=std::string::npos&&stop>start,"missing panel delimiter");
 const auto body=svg.substr(start,stop-start);need(body.find("data-dynamic-series=\"scatter-3x3\"")!=std::string::npos,"wrong series grammar");
 std::set<long long> seen;const std::regex tags(R"(<rect\b[^>]*>)");
 for(auto i=std::sregex_iterator(body.begin(),body.end(),tags);i!=std::sregex_iterator();++i){
  const auto a=attributes(i->str());need(a.contains("data-source-index"),"non-glyph rect in panel");
  const auto index=std::stoll(a.at("data-source-index"));need(expected.contains(index)&&seen.insert(index).second,"missing/duplicate/wrong source index");
  need(a.at("data-source-row")=="-1"&&a.at("data-source-col")=="-1","wrong source provenance");
  const auto w=numeric(a.at("width")),h=numeric(a.at("height"));
  need(w>=1&&w<=3&&h>=1&&h<=3&&std::floor(w)==w&&std::floor(h)==h,"invalid clipped pixel glyph");
  numeric(a.at("x"));numeric(a.at("y"));
  need(a.at("fill")== (which==0?"rgb(70,180,235)":"rgb(235,130,70)"),"panel color changed");
 }
 need(seen==expected,"glyph omitted");
}
void document(const std::string&s){
 need(s.find("<svg xmlns=")!=std::string::npos&&s.find("</svg>")!=std::string::npos,"missing SVG root");
 need(s.find("data-interval-kind=\"none\"")!=std::string::npos,"unexpected interval claim");
 const std::regex complete(R"(<metadata data-scientific-document-complete="true"[^>]*>)");std::smatch m;
 need(std::regex_search(s,m,complete),"missing document completion");const auto a=attributes(m.str());
 need(a.at("data-left-input")=="6"&&a.at("data-left-masked")=="2"&&a.at("data-left-outside")=="0","left availability metadata");
 need(a.at("data-right-input")=="16"&&a.at("data-right-masked")=="0"&&a.at("data-right-outside")=="0","right availability metadata");
 // Factory: y100,h380; symmetric right residual domain maps zero to y290.
 // Full horizontal pixel-cell reference is distinct from the y480 baseline.
 bool zero=false;const std::regex rects(R"(<rect\b[^>]*>)");
 for(auto i=std::sregex_iterator(s.begin(),s.end(),rects);i!=std::sregex_iterator();++i){const auto r=attributes(i->str());if(!r.contains("data-document-axis")){continue;}if(numeric(r.at("x"))>690&&numeric(r.at("y"))==289.5&&numeric(r.at("height"))==1&&numeric(r.at("width"))>100){zero=true;}}
 need(zero,"missing right zero reference");
}
int workflow(const char*path){
 std::ifstream f(path);need(bool(f),"missing native workflow stdout");const std::string s((std::istreambuf_iterator<char>(f)),{});
 const auto end=s.find("</svg>"),done=s.find("<!-- COUNT_WORKFLOW_COMPLETE -->");need(done!=std::string::npos&&end!=std::string::npos&&done>end,"missing terminal workflow completion");
 document(s);panel(s,0,{0,1,2,4});std::set<long long> right;for(long long i=0;i<16;++i){right.insert(i);}panel(s,1,right);
 const std::regex record(R"(<!-- gene ([0-9]+) status ([0-9]+) profile (-?[0-9]+) alpha ([-+0-9.eE]+) -->)");std::set<int> seen;
 for(auto i=std::sregex_iterator(s.begin(),s.end(),record);i!=std::sregex_iterator();++i){
  const int g=std::stoi((*i)[1]),status=std::stoi((*i)[2]),profile=std::stoi((*i)[3]);const auto alpha=numeric((*i)[4]);
  need(g>=0&&g<6&&seen.insert(g).second,"duplicate/wrong gene metadata");
  if(g==3){need(status!=0&&profile==-1,"zero-count gene not unavailable");}
  else if(g==5){need(status==0&&profile>=2&&profile<=4&&alpha>0,"positive constrained profile not censored");}
  else if(g<3){need(status==0&&profile==1&&alpha==0,"Poisson profile misclassified");}
  else {need(status==0&&profile==0&&alpha>0,"interior profile misclassified");}
 }
 need(seen.size()==6,"incomplete gene report");
 need(s.find("Poisson(alpha=0): 3; positive edges masked: 1; failed masked: 1.")!=std::string::npos,"visible status footer missing");
 need(s.find("no CI/FDR")!=std::string::npos,"missing inference limitation");
 std::cout<<"COUNT_WORKFLOW_CPP_STRUCTURE_PASS genes=6 glyphs=20 unavailable=1 positive_edges=1 poisson=3 (not profile calibration)\n";return 0;
}
int main(int argc,char**argv){try{
 if(argc==3&&std::string(argv[1])=="--workflow"){return workflow(argv[2]);}
 if(argc!=2){std::cerr<<"usage: deep_count_dataset_reference [--workflow] NATIVE_STDOUT\n";return 2;}
 std::ifstream in(argv[1]);need(bool(in),"native reference stdout missing");constexpr std::int64_t n=512;
 std::map<std::int64_t,std::int64_t> columns;for(std::int64_t j=0;j<n;++j){need(columns.emplace((n-1-j)*3-999,j).second,"oracle duplicate key");}
 std::set<std::int64_t> seen;bool done=false;std::string line;
 while(std::getline(in,line)){
  if(line=="COUNT_DATASET_REFERENCE_PASS"){need(!done,"duplicate completion");done=true;continue;}
  if(!line.starts_with("DATASET ")){continue;}
  std::istringstream row(line);std::string tag;std::int64_t i,col,count;double offset;need(bool(row>>tag>>i>>col>>count>>offset),"malformed DATASET row");
  std::string extra;need(!(row>>extra),"extra DATASET payload");need(i>=0&&i<n&&seen.insert(i).second,"duplicate/outside DATASET index");
  const auto found=columns.find(i*3-999);need(found!=columns.end(),"oracle missing key");const auto expected=found->second;
  need(col==expected&&count==(expected+1)*7&&offset==double(expected)*.25+.5,"exact keyed join mismatch");
 }
 need(done&&seen.size()==std::size_t(n),"incomplete DATASET output");std::cout<<"COUNT_DATASET_CPP_CONCORDANCE_PASS rows=512 exact_integer_join=1\n";return 0;
 }catch(const std::exception&e){std::cerr<<"COUNT_DATASET_CPP_FAIL "<<e.what()<<'\n';return 1;}}
