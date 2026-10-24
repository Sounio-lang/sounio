// C++23 canonical geometry comparison and byte measurement for native SVG logs.
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cctype>
using std::string;
string read(const std::filesystem::path& p){
 std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read "+p.string());
 return {std::istreambuf_iterator<char>(f),{}};
}
string attr(const string& tag,const string& key){
 string prefix=" "+key+"=\"";auto a=tag.find(prefix);
 if(a==string::npos)return {};
 a+=prefix.size();auto b=tag.find('"',a);
 if(b==string::npos)throw std::runtime_error("attribute quote");
 return tag.substr(a,b-a);
}
void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
struct Canon {string text;std::size_t segments=0,paths=0,lines=0,groups=0;};
string number(const string& s){
 std::size_t end=0;auto n=std::stoll(s,&end);require(end==s.size(),"integer coordinate");
 return std::to_string(n);
}
Canon canonical(const string& svg,bool compact){
 Canon out;std::size_t cursor=0;
 const string marker="<g clip-path=\"url(#plot-clip-";
 while(true){
  auto a=svg.find(marker,cursor);
  if(a==string::npos){out.text+=svg.substr(cursor);break;}
  auto start=svg.find('>',a);auto end=svg.find("</g>",start);
  require(start!=string::npos&&end!=string::npos,"clip group");
  out.text+=svg.substr(cursor,start+1-cursor);
  string body=svg.substr(start+1,end-start-1);std::size_t pos=0,count=0,elements=0;
  while(pos<body.size()){
   if(std::isspace((unsigned char)body[pos])){pos++;continue;}
   auto close=body.find("/>",pos);require(close!=string::npos,"shape close");
   string tag=body.substr(pos,close+2-pos);
   bool path=tag.starts_with("<path ");
   bool line=tag.starts_with("<line ");
   require(path||line,"unexpected clip shape");
   require(compact?path:line,"serialization kind");
   require(attr(tag,"stroke-width")=="1"&&attr(tag,"stroke-linecap")=="round","stroke contract");
   string style=attr(tag,"stroke");require(!style.empty(),"missing color");
   auto segment=[&](long long x0,long long y0,long long x1,long long y1){
    out.text+="[segment:"+style+":"+std::to_string(x0)+","+std::to_string(y0)+","+
     std::to_string(x1)+","+std::to_string(y1)+"]";
    count++;out.segments++;
   };
   if(path){
    require(attr(tag,"fill")=="none","path fill");
    string data=attr(tag,"d");std::istringstream in(data);std::size_t n=0;
    while(true){
     in>>std::ws;if(in.eof())break;
     char m=0,l=0;long long x0,y0,x1,y1;
     require(bool(in>>m>>x0>>y0>>l>>x1>>y1),"path grammar");
     require(m=='M'&&l=='L',"independent M/L subpath required");
     segment(x0,y0,x1,y1);n++;
    }
    require(n>0,"empty path");out.paths++;
   }else{
    segment(std::stoll(number(attr(tag,"x1"))),std::stoll(number(attr(tag,"y1"))),
            std::stoll(number(attr(tag,"x2"))),std::stoll(number(attr(tag,"y2"))));
    out.lines++;
   }
   pos=close+2;elements++;
  }
  require(!compact||elements<=1,"more than one path per group");
  auto end_audit=svg.find("-->",end);require(end_audit!=string::npos,"audit close");
  string audit=svg.substr(end+4,end_audit-(end+4)+3);
  auto v=audit.find("visible=");require(v!=string::npos,"missing audit");
  require(std::stoull(audit.substr(v+8))==count,"audit count mismatch");
  out.text+="</g>";cursor=end+4;out.groups++;
 }
 return out;
}
std::vector<string> documents(const string& log){
 std::vector<string> v;std::size_t p=0;
 while(true){auto a=log.find("<svg ",p);if(a==string::npos)break;
  auto b=log.find("</svg>",a);require(b!=string::npos,"SVG close");
  v.push_back(log.substr(a,b+6-a));p=b+6;
 }
 require(!v.empty(),"zero SVG documents");return v;
}
void sizes(std::size_t a,std::size_t b){
 require(a>0,"empty baseline size");
 std::cout<<"before_bytes="<<a<<" after_bytes="<<b<<" reduction_percent="
 <<std::fixed<<std::setprecision(4)<<(100.0L*(1.0L-(long double)b/a))<<"\n";
}
int main(int argc,char**argv)try{
 if(argc==4&&string(argv[1])=="--sizes"){
  sizes(read(argv[2]).size(),read(argv[3]).size());return 0;
 }
 if(argc!=4)return 2;
 auto before=documents(read(argv[1])),after=documents(read(argv[2]));
 require(before.size()==after.size(),"document count mismatch");
 std::filesystem::path dir=argv[3];std::filesystem::create_directories(dir);
 std::size_t n=0,paths=0,lines=0,groups=0,total_before=0,total_after=0;
 for(std::size_t i=0;i<before.size();i++){
  auto a=canonical(before[i],false),b=canonical(after[i],true);
  require(a.text==b.text,"geometry/style/mask/audit or other SVG content changed");
  std::ofstream(dir/("before-"+std::to_string(i)+".svg"))<<before[i];
  std::ofstream(dir/("after-"+std::to_string(i)+".svg"))<<after[i];
  n+=b.segments;paths+=b.paths;lines+=a.lines;groups+=b.groups;
  total_before+=before[i].size();total_after+=after[i].size();
  std::cout<<"document="<<i<<" segments="<<b.segments<<" paths="<<b.paths<<" ";
  sizes(before[i].size(),after[i].size());
 }
 std::cout<<"EQUIVALENT documents="<<before.size()<<" clip_groups="<<groups
 <<" segments="<<n<<" before_line_elements="<<lines<<" after_path_elements="<<paths<<" ";
 sizes(total_before,total_after);return 0;
}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<"\n";return 1;}
