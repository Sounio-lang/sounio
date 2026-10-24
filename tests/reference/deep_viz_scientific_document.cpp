// Independent C++23 oracle for actual complete SVG and scalar/native-pixel reports.
// No production projection/statistical code is included. Count means are the
// closed-form score roots of the two disjoint indicator groups (equal exposure).
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
struct Rect {long long x,y,w,h,index;};
static void require(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
static std::string read(const char* p){std::ifstream f(p);require(bool(f),"open stdout");return {std::istreambuf_iterator<char>(f),{}};}
static std::string decode(std::string s){
 for(const auto& [a,b]:std::vector<std::pair<std::string,std::string>>{{"&lt;","<"},{"&gt;",">"},{"&quot;","\""},{"&apos;","'"},{"&amp;","&"}}){
  std::size_t i=0;while((i=s.find(a,i))!=std::string::npos){s.replace(i,a.size(),b);i+=b.size();}
 }
 return s;
}
static void close(long double actual,long double expected,const std::string& label){
 require(std::isfinite(actual)&&std::fabs(actual-expected)<=2e-8L*(1+std::fabs(expected)),label);
}
static long double nb_residual(int y,long double mu){
 const long double r=4;
 const long double first=y==0?0:static_cast<long double>(y)*std::log(static_cast<long double>(y)/mu);
 const long double d=2*(first-(y+r)*std::log((y+r)/(mu+r)));
 return y==mu?0:(y<mu?-1:1)*std::sqrt(d);
}
struct Range {long long lo,hi;};
static Range project_x(long double v,long double error,long double domain,long long x,long long w){
 const long double a=(v-error)/domain*w,b=(v+error)/domain*w;
 return {x+static_cast<long long>(std::trunc(a)),x+static_cast<long long>(std::trunc(b))};
}
static Range project_y(long double v,long double error,long double bound,long double bound_error,long long y,long long h){
 long double low=1e100L,high=-1e100L;
 for(int i:{-1,1})for(int j:{-1,1}){
  const auto b=bound+j*bound_error;
  const auto scaled=std::clamp(((v+i*error+b)/(2*b))*h,0.L,static_cast<long double>(h));
  low=std::min(low,scaled);high=std::max(high,scaled);
 }
 // Roundoff allowance is arithmetic scale dependent, before integer truncation.
 const auto eps=64*std::numeric_limits<long double>::epsilon()*(1+h);
 return {y+h-static_cast<long long>(std::trunc(std::min(static_cast<long double>(h),high+eps))),
         y+h-static_cast<long long>(std::trunc(std::max(0.L,low-eps)))};
}
static bool rect_matches(const Rect& a,Range cx,Range cy,long long x,long long y,long long w,long long h){
 require(cx.hi-cx.lo<=1&&cy.hi-cy.lo<=1,"projection interval unexpectedly broad");
 for(auto sx=cx.lo;sx<=cx.hi;++sx)for(auto sy=cy.lo;sy<=cy.hi;++sy){
  const auto left=std::max(x,sx-1),top=std::max(y,sy-1);
  const auto right=std::min(x+w+1,sx+2),bottom=std::min(y+h+1,sy+2);
  if(a.x==left&&a.y==top&&a.w==right-left&&a.h==bottom-top)return true;
 }
 return false;
}
static std::vector<Rect> rectangles(const std::string& s,const std::string& color){
 const std::regex pattern(R"re(<rect data-source-index="([0-9]+)" data-source-row="-1" data-source-col="-1" x="([-0-9.]+)" y="([-0-9.]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(([0-9]+,[0-9]+,[0-9]+)\)"/>)re");
 std::vector<Rect> out;
 for(std::sregex_iterator i(s.begin(),s.end(),pattern),end;i!=end;++i){
  const auto& m=*i;require(m[6]==color,"wrong series RGB");
  const auto x=std::stold(m[2])+0.5L,y=std::stold(m[3])+0.5L;
  require(x==std::trunc(x)&&y==std::trunc(y),"not pixelcell half origin");
  out.push_back({static_cast<long long>(x),static_cast<long long>(y),std::stoll(m[4]),std::stoll(m[5]),std::stoll(m[1])});
 }
 return out;
}
int main(int argc,char** argv){
 try {
  const bool count=argc==3&&std::string(argv[1])=="--count";
  require(argc==(count?3:2),"usage: ORACLE [--count] actual.stdout");
  const auto raw=read(argv[count?2:1]);
  const auto z0=raw.find("ZERO_SVG_BEGIN\n"),z1=raw.find("ZERO_SVG_END",z0);
  require(z0!=std::string::npos&&z1!=std::string::npos,"missing rejection witness");
  const auto silence=raw.substr(z0+15,z1-(z0+15));
  require(silence.find_first_not_of("\r\n \t")==std::string::npos,"invalid late panel emitted bytes");
  const auto start=raw.find("<svg "),stop=raw.find("</svg>",start);
  require(start!=std::string::npos&&stop!=std::string::npos&&raw.find("<svg ",start+1)==std::string::npos,"one complete SVG required");
  const auto svg=raw.substr(start,stop+6-start);
  const long long width=count?1380:1280,height=count?620:480;
  require(svg.find("width=\""+std::to_string(width)+"\" height=\""+std::to_string(height)+"\"")!=std::string::npos,"root dimensions");
  require(svg.find("data-interval-kind=\"none\"")!=std::string::npos,"no fabricated intervals");
  require(svg.find("data-scientific-document-complete=\"true\"")!=std::string::npos,"missing completion metadata");
  require(svg.find("Counts &amp; fit &lt;declared&gt; &quot;x&quot; &apos;y&apos;")!=std::string::npos,"XML escaping");
  const std::regex text(R"re(<text [^>]*x="(-?[0-9]+)" y="(-?[0-9]+)"[^>]*textLength="([0-9]+)"[^>]*>([^<]*)</text>)re");
  int text_count=0;
  for(std::sregex_iterator i(svg.begin(),svg.end(),text),end;i!=end;++i){
   const auto x=std::stoll((*i)[1]),y=std::stoll((*i)[2]),w=std::stoll((*i)[3]);
   require(x>=0&&x+w<=width&&y-7>=0&&y+1<=height,"clipped document label");
   require(w==static_cast<long long>(decode((*i)[4]).size())*8,"canonical label width");++text_count;
  }
  require(text_count>=12,"missing labels/axes");
  const auto panel0=svg.find("<g id=\"scientific-panel-0\">"),panel1=svg.find("<g id=\"scientific-panel-1\">");
  require(panel0!=std::string::npos&&panel1>panel0,"unique panels");
  const auto left=rectangles(svg.substr(panel0,panel1-panel0),"70,180,235");
  const auto right=rectangles(svg.substr(panel1),"235,130,70");
  const long long n=count?1028:1025;
  require(left.size()==static_cast<std::size_t>(n)&&right.size()==left.size(),"lost or duplicate marks");
  // Scientific labels: max Y width88 (negative six-significant-digit label),
  // max X width80. Shared margins24+88 and24+80/2; root halves are exact.
  const long long x=112,y=100,w=width/2-112-64,h=height-240,rx=width/2+x;
  std::array<long double,4> expected{};
  long double bound=2;
  if(count){
   bound=0;
   for(int i=0;i<4;++i){expected[i]=nb_residual(i*2,i%2?4:2);bound=std::max(bound,std::fabs(expected[i]));}
   std::istringstream in(raw);std::string token;int rows=0;bool saw_bound=false;
   while(in>>token){
    if(token=="COUNT_BOUND"){long double value;require(bool(in>>value),"bound payload");close(value,bound,"independent residual envelope");saw_bound=true;}
    if(token=="COUNT_ROW"){int index;long double mean,residual;require(bool(in>>index>>mean>>residual),"row payload");require(index==rows&&index<4,"row order");close(mean,index%2?4:2,"independent group mean");close(residual,expected[index],"independent signed deviance");++rows;}
   }
   require(rows==4&&saw_bound,"missing computed diagnostics");
   require(svg.find("data-assumptions-assessed=\"false\" data-inference-available=\"false\"")!=std::string::npos,"false inference flags");
   require(svg.find("data-document-reference=\"identity\"")!=std::string::npos,"missing identity reference");
  }
  for(long long i=0;i<n;++i){
   require(left[i].index==i&&right[i].index==i,"source-index provenance");
   const long double mu=count?(i%2?4:2):1+i%3;
   const long double observation=count?(i%4)*2:mu;
   const long double residual=count?expected[i%4]:static_cast<long double>(i%3)-1;
   const long double maximum=count?6:4;
   const long double em=count?2e-8L*(1+std::fabs(mu)):0;
   const long double er=count?2e-8L*(1+std::fabs(residual)):0;
   const long double eb=count?2e-8L*(1+bound):0;
   const auto cx=project_x(mu,em,maximum,x,w),crx=project_x(mu,em,maximum,rx,w);
   const auto sy=y+h-static_cast<long long>(std::trunc(observation/maximum*h));
   require(rect_matches(left[i],cx,{sy,sy},x,y,w,h),"left mark geometry index"+std::to_string(i));
   const auto cy=count?project_y(residual,er,bound,eb,y,h):Range{y+h-static_cast<long long>((residual+2)/4*h),y+h-static_cast<long long>((residual+2)/4*h)};
   require(rect_matches(right[i],crx,cy,rx,y,w,h),"right mark geometry index"+std::to_string(i));
  }
  const std::regex pixel(count?R"(COUNT_PIXEL ([0-9]+) ([0-9]+) ([0-9]+))":R"(DOC_PIXEL ([0-9]+) ([0-9]+) ([0-9]+))");
  std::smatch m;require(std::regex_search(raw,m,pixel),"missing actual Canvas pixel");
  const auto px=std::stoll(m[1]),py=std::stoll(m[2]),rgb=std::stoll(m[3]);
  const auto center=project_x(count?2:1,count?6e-8L:0,count?6:4,x,w);
  require(px>=center.lo&&px<=center.hi&&py==(count?y+h:y+h-h/4),"pixel probe geometry");
  require(rgb==((70LL<<16)|(180LL<<8)|235),"actual Canvas series RGB");
  require(raw.find(count?"STDLIB_DEEP_VIZ_COUNT_DIAGNOSTICS_PASS":"STDLIB_DEEP_VIZ_SCIENTIFIC_DOCUMENT_PASS")!=std::string::npos,"missing native completion");
  std::cout<<"SCIENTIFIC_DOCUMENT_REFERENCE_PASS mode="<<(count?"count":"document")<<" marks="<<2*n<<" text_bounds="<<text_count<<" zero_bytes_verified=1\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
