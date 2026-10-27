// C++23 independent scientific/geometry reader. No production solver is copied.
// Reuse only the reviewed primary-data/font lexical readers (not its figure check).
#define main canonical_palette_cli_unused
#include "canonical_palette_reference.cpp"
#undef main
#include <limits>
#include <set>
#include <iomanip>
#include <filesystem>
namespace science {
constexpr I width=1480,height=800;
struct S {I x0,y0,x1,y1,cell,level,branch;};
struct Input {std::string all,svg;std::vector<std::uint32_t> pixels;std::vector<double> nodes;std::vector<S> segments;I cycles{},work{},declared_segments{};double residual{},maximum{};};
void require(bool v,const std::string& why){if(!v)throw std::runtime_error(why);}
double number(I bits){return std::bit_cast<double>(static_cast<std::uint64_t>(bits));}
std::string science_text_decode(const std::string&text){std::string out;for(std::size_t i=0;i<text.size();){if(text[i]!='&'){out+=text[i++];continue;}const auto end=text.find(';',i);require(end!=std::string::npos,"XML text entity terminator");const auto entity=text.substr(i,end-i+1);if(entity=="&amp;")out+='&';else if(entity=="&lt;")out+='<';else if(entity=="&gt;")out+='>';else if(entity=="&quot;")out+='"';else if(entity=="&apos;")out+='\'';else throw std::runtime_error("unsupported XML text entity");i=end+1;}return out;}
Input read(const std::string& p,const std::string&completion="SCIENTIFIC_POISSON_COMPLETE"){
 Input r;r.all=contents(p);require(r.all.size()<32*1024*1024,"bounded native output");
 const auto start=r.all.find("<svg "),end=r.all.find("</svg>");require(start==0&&end!=std::string::npos,"complete SVG prefix/root");r.svg=r.all.substr(start,end+6);
 require(r.all.find(completion)!=std::string::npos,"science completion marker");
 r.pixels.assign(width*height,0);r.nodes.reserve(4225);I next_pixel=0;bool begin=false,done=false,meta=false;
 std::istringstream rows(r.all);std::string row;
 while(std::getline(rows,row)){
  std::istringstream in(row);std::string a,tag;in>>a>>tag;if(a!="<!--")continue;
  if(tag=="CANVAS_RLE"){I w,h;in>>w>>h;require(in&&w==width&&h==height&&!begin,"raster header");begin=true;}
  else if(tag=="PIX"){
   I y,x,n,rgb;in>>y>>x>>n>>rgb;require(in&&begin&&!done&&y>=0&&y<height&&x>=0&&x<width&&n>0&&n<=width-x&&rgb>=0&&rgb<=0xffffff,"raster run bounds");
   require(y*width+x==next_pixel,"raster exact coverage/order");std::fill_n(r.pixels.begin()+next_pixel,n,static_cast<std::uint32_t>(rgb));next_pixel+=n;
  }else if(tag=="CANVAS_RLE_END"){require(begin&&!done&&next_pixel==width*height,"raster complete");done=true;}
  else if(tag=="NODE"){I i,b;in>>i>>b;require(in&&i==static_cast<I>(r.nodes.size())&&i<4225,"nodal row completeness");const auto u=number(b);require(std::isfinite(u),"nonfinite solved node");r.nodes.push_back(u);}
  else if(tag=="POISSON_META"){I rb,eb;in>>r.cycles>>r.work>>rb>>eb>>r.declared_segments;require(in&&!meta,"solver metadata once");meta=true;r.residual=number(rb);r.maximum=number(eb);}
  else if(tag=="SEGMENT"){I i,status;S s{};in>>i>>status>>s.x0>>s.y0>>s.x1>>s.y1>>s.cell>>s.level>>s.branch;require(in&&status==0&&i==static_cast<I>(r.segments.size()),"prepared provenance row");r.segments.push_back(s);}
 }
 require(done,"complete raster payload");if(completion=="SCIENTIFIC_POISSON_COMPLETE")require(meta&&r.nodes.size()==4225&&r.declared_segments==static_cast<I>(r.segments.size()),"complete science payload");return r;
}
void check_science(const Input&r){
 require(r.cycles>0&&r.cycles<=80&&r.work>0&&r.work<=100000000,"declared solver budgets");
 require(std::isfinite(r.residual)&&r.residual>=0&&r.residual<=1e-10,"residual stopping contract");
 double maximum=0;long double error=0;
 for(I y=0;y<=64;++y)for(I x=0;x<=64;++x){
  const long double a=static_cast<long double>(x)/64,b=static_cast<long double>(y)/64,target=a*(1-a)*b*(1-b);
  const double value=r.nodes[y*65+x];error=std::max(error,std::fabs(static_cast<long double>(value)-target));
  maximum=std::max(maximum,std::fabs(value-static_cast<double>(target)));
  if(x==0||x==64||y==0||y==64)require(value==0,"Dirichlet zero boundary");
 }
 require(error<1e-9L,"manufactured PDE analytic solution");require(std::isfinite(r.maximum)&&r.maximum==maximum,"actual maximum nodal error metadata");
 long double residual=0;
 for(I y=1;y<64;++y)for(I x=1;x<64;++x){const I i=y*65+x;const long double a=static_cast<long double>(x)/64,b=static_cast<long double>(y)/64;
  const long double f=2*(a*(1-a)+b*(1-b));const long double lhs=(4*static_cast<long double>(r.nodes[i])-r.nodes[i-1]-r.nodes[i+1]-r.nodes[i-65]-r.nodes[i+65])*4096;
  residual=std::max(residual,std::fabs(lhs-f));}
 require(residual<1.2e-10L,"independent finite-difference residual");
 std::cout<<std::setprecision(18)<<"SCIENCE_POISSON_PASS nodes=4225 independent_max_error="<<error<<" independent_residual="<<residual<<'\n';
}
std::pair<I,I> projection(long double raw,I origin,bool reverse){
 const long double q=raw*512;
 // Independent long-double interpolation versus binary64 arithmetic. Only a
 // genuine integer-boundary straddle permits adjacent pixels; never blanket +/-1.
 const long double e=64*std::numeric_limits<double>::epsilon()*(1+std::fabs(q));
 I a=static_cast<I>(std::floor(q-e)),b=static_cast<I>(std::floor(q+e));a=std::max<I>(0,a);b=std::min<I>(512,b);
 return reverse?std::pair<I,I>{origin+512-b,origin+512-a}:std::pair<I,I>{origin+a,origin+b};
}
void check_segments(const Input&r){
 std::size_t selected=0;
 for(I level=0;level<5;++level)for(I y=0;y<64;++y)for(I x=0;x<64;++x){
  const long double l=static_cast<double>(static_cast<double>(level+1)*0.01);
  const I ix[]{y*65+x,y*65+x+1,(y+1)*65+x+1,(y+1)*65+x};
  const long double xx[]{static_cast<long double>(x)/64,static_cast<long double>(x+1)/64,static_cast<long double>(x+1)/64,static_cast<long double>(x)/64};
  const long double yy[]{static_cast<long double>(y)/64,static_cast<long double>(y)/64,static_cast<long double>(y+1)/64,static_cast<long double>(y+1)/64};
  std::vector<std::pair<long double,long double>> crossing;
  for(I e=0;e<4;++e){const I next=(e+1)%4;const long double a=r.nodes[ix[e]],b=r.nodes[ix[next]];
   require(a!=l||b!=l,"fixture has no flat contour edge");if((a>=l)!=(b>=l)){const long double t=(l-a)/(b-a);crossing.emplace_back(xx[e]+t*(xx[next]-xx[e]),yy[e]+t*(yy[next]-yy[e]));}}
  if(crossing.empty())continue;
  require(crossing.size()==2,"manufactured fixture has no saddle cell");require(selected<r.segments.size(),"missing prepared segment");const auto&s=r.segments[selected++];
  require(s.cell==y*64+x&&s.level==level&&s.branch==0,"cell/level/branch provenance");
  const auto ax=projection(crossing[0].first,112,false),ay=projection(crossing[0].second,112,true),bx=projection(crossing[1].first,112,false),by=projection(crossing[1].second,112,true);
  require(s.x0>=ax.first&&s.x0<=ax.second&&s.y0>=ay.first&&s.y0<=ay.second&&s.x1>=bx.first&&s.x1<=bx.second&&s.y1>=by.first&&s.y1<=by.second,"independent contour projection");
 }
 require(selected==r.segments.size(),"extra prepared segment");
 const std::regex line(R"re(<line data-contour-cell="([0-9]+)" data-contour-level-index="([0-9]+)" data-contour-branch="([0-9]+)" data-contour-level="[^"]+" x1="([0-9-]+)" y1="([0-9-]+)" x2="([0-9-]+)" y2="([0-9-]+)" stroke="rgb\(240,240,240\)" stroke-width="1" stroke-linecap="round"/>)re");
 selected=0;for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),line),end;it!=end;++it){require(selected<r.segments.size(),"extra SVG segment");const auto&s=r.segments[selected++];const auto&m=*it;
  require(std::stoll(m[1])==s.cell&&std::stoll(m[2])==s.level&&std::stoll(m[3])==s.branch&&std::stoll(m[4])==s.x0&&std::stoll(m[5])==s.y0&&std::stoll(m[6])==s.x1&&std::stoll(m[7])==s.y1,"SVG consumes same prepared snapshot");}
 require(selected==r.segments.size(),"SVG segment completeness");
 std::cout<<"SCIENCE_PREPARED_GEOMETRY_PASS segments="<<selected<<" independent_cell_walk_and_long_double_projection\n";
}
I color(const Maps&maps,I palette,double value,double maximum){const I bin=maximum==0?128:value>=maximum?255:std::clamp<I>(static_cast<I>(std::floor(256*(value/maximum))),0,255);return packed(maps[palette][bin]);}
std::pair<I,I> check_text_and_export(const Input&r,const std::string&ppm,I minimum_texts){
 // Text is checked from its true SVG position against the pinned public-domain
 // 8x8 glyph data; full textLength rectangles must fit the document.
 const auto glyphs=font();const std::regex tr(R"(<text\s[^>]*>[^<]*</text>)");I texts=0,glyph_checks=0;
 for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),tr),end;it!=end;++it){const std::string tag=it->str();const I x=std::stoll(attr(tag,"x")),baseline=std::stoll(attr(tag,"y")),length=std::stoll(attr(tag,"textLength"));const auto begin=tag.find('>'),close=tag.rfind('<');const auto label=science_text_decode(tag.substr(begin+1,close-begin-1));
  require(x>=0&&x+length<=width&&baseline>=7&&baseline<height&&length==static_cast<I>(label.size())*8,"actual text footprint/bounds");
  for(std::size_t j=0;j<label.size();++j){const auto ch=static_cast<unsigned char>(label[j]);require(ch>=32&&ch<=126,"ASCII scientific label");for(I y=0;y<8;++y)for(I k=0;k<8;++k)if((glyphs[ch]>>(8*y+k))&1){require(r.pixels[(baseline-7+y)*width+x+static_cast<I>(j)*8+k]==0xebf0f5,"actual Canvas primary-font glyph");++glyph_checks;}}
  ++texts;
 }
 const std::regex text_open(R"(<text(?:\s|>))");I opening_texts=0;for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),text_open),end;it!=end;++it)++opening_texts;
 require(texts==opening_texts,"every SVG text element is included in bounds/glyph validation");
 require(texts>=minimum_texts,"scientific label completeness");
 std::set<std::string> ids;const std::regex id(R"re(\sid="([^"]+)")re");for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),id),end;it!=end;++it)require(ids.insert((*it)[1]).second,"unique SVG IDs");

 // Rules are true half-open pixel cells, also checked against actual Canvas.
 const std::regex rule(R"(<rect data-science-rule="1"[^>]+/>)");
 I rules=0;
 for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),rule),end;it!=end;++it){const auto tag=it->str();const I x=cell_coordinate(attr(tag,"x")),y=cell_coordinate(attr(tag,"y")),w=std::stoll(attr(tag,"width")),h=std::stoll(attr(tag,"height"));
  require(x>=0&&y>=0&&w>0&&h>0&&w<=width-x&&h<=height-y,"axis rule footprint");
  for(I yy=y;yy<y+h;++yy)for(I xx=x;xx<x+w;++xx){require(r.pixels[yy*width+xx]==0xb4bec8,"actual Canvas/SVG axis rule");}
  ++rules;
 }
 require(rules>=4,"axis rule completeness");
 if(!ppm.empty()){std::ofstream image(ppm,std::ios::binary);require(bool(image),"actual raster output");image<<"P6\n"<<width<<' '<<height<<"\n255\n";for(const auto rgb:r.pixels){const char bytes[]{static_cast<char>(rgb>>16),static_cast<char>(rgb>>8),static_cast<char>(rgb)};image.write(bytes,3);}require(bool(image),"actual raster write");}
 if(!ppm.empty()){
  std::ofstream out(ppm+".rle");require(bool(out),"actual raster RLE output");out<<"CANVAS_SIZE "<<width<<' '<<height<<'\n';
  std::size_t first=0;
  while(first<r.pixels.size()){std::size_t last=first+1;while(last<r.pixels.size()&&r.pixels[last]==r.pixels[first])++last;out<<"CANVAS_RUN "<<first<<' '<<last-first<<' '<<r.pixels[first]<<'\n';first=last;}
  out<<"CANVAS_END\n";require(bool(out),"actual raster RLE write");
 }
 return {texts,glyph_checks};
}
I fill_color(const std::string&tag);
void check_palette_svg(const Input&r,const Maps&maps,I first,I second,double max0,double max1){
 const std::regex groups(R"re(<g data-dynamic-series="(heatmap|canonical-colorbar)"[^>]*>)re");I count=0;
 for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),groups),end;it!=end;++it){const auto tag=it->str();require(count<4,"extra field/colorbar policy metadata");const I panel=count%2,palette=panel?second:first;const double maximum=panel?max1:max0;
  require(attr(tag,"data-dynamic-series")==std::string(count<2?"heatmap":"canonical-colorbar")&&attr(tag,"data-palette")==names[palette],"field/colorbar canonical palette identity");
  require(attr(tag,"data-palette-source")=="BIDS/colormap@bc549477db0c12b54a5928087552ad2cf274980f"&&attr(tag,"data-palette-rgb8")=="floor-255-channel"&&attr(tag,"data-palette-bins")=="floor-256-t-endpoint-255","actual canonical palette source and quantization");
  const double low=std::stod(attr(tag,"data-color-min")),high=std::stod(attr(tag,"data-color-max"));require(low==0&&std::isfinite(high)&&(maximum==0?high==0:std::fabs(high-maximum)<=4e-15*std::fabs(maximum)),"true color-domain metadata including tiny nonzero errors");
  if(count<2){require(attr(tag,"data-row-order")=="top-to-bottom","explicit matrix row order");}else{require(attr(tag,"data-orientation")=="increasing-left-to-right"&&attr(tag,"data-constant-domain")==std::string(maximum==0?"true":"false"),"honest colorbar orientation/constant policy");}
  ++count;
 }
 require(count==4,"complete field/colorbar policy metadata");
 const I first_steps=max0==0?1:256,second_steps=max1==0?1:256;const std::regex rect(R"(<rect data-colorbar-step="[0-9]+"[^>]+/>)");count=0;
 for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),rect),end;it!=end;++it){require(count<first_steps+second_steps,"extra colorbar step");const auto tag=it->str();const I panel=count>=first_steps,step=panel?count-first_steps:count,steps=panel?second_steps:first_steps,origin=panel?816:112,palette=panel?second:first,bin=steps==1?128:step;
  require(std::stoll(attr(tag,"data-colorbar-step"))==step&&std::stoll(attr(tag,"data-palette-index"))==bin&&cell_coordinate(attr(tag,"x"))==origin+step*512/steps&&cell_coordinate(attr(tag,"y"))==688&&std::stoll(attr(tag,"width"))==512/steps&&std::stoll(attr(tag,"height"))==16&&fill_color(tag)==packed(maps[palette][bin]),"independent actual SVG colorbar cells");++count;
 }
 require(count==first_steps+second_steps,"SVG colorbar step completeness");
}
void check_pixels(const Input&r,const Maps&maps,const std::string&ppm){
 std::vector<bool> fringe(width*height,false);
 for(const auto&s:r.segments){for(I y=std::max<I>(0,std::min(s.y0,s.y1)-2);y<=std::min<I>(height-1,std::max(s.y0,s.y1)+2);++y)for(I x=std::max<I>(0,std::min(s.x0,s.x1)-2);x<=std::min<I>(width-1,std::max(s.x0,s.x1)+2);++x){
  const double dx=s.x1-s.x0,dy=s.y1-s.y0,denom=dx*dx+dy*dy;const double t=denom==0?0:std::clamp(((x-s.x0)*dx+(y-s.y0)*dy)/denom,0.,1.);const double ex=x-(s.x0+t*dx),ey=y-(s.y0+t*dy);if(ex*ex+ey*ey<=4)fringe[y*width+x]=true;}}
 std::size_t checks=0;std::array<I,8192> expected_colors{};
 for(I row=0;row<64;++row)for(I col=0;col<64;++col){const I y=63-row;double value=0,error=0;
  for(I corner=0;corner<4;++corner){const I ix=col+corner%2,iy=y+corner/2;const double a=static_cast<double>(ix)/64,b=static_cast<double>(iy)/64,node=r.nodes[iy*65+ix];value+=node*0.25;error+=std::fabs(node-a*(1-a)*b*(1-b))*0.25;}
  for(I panel=0;panel<2;++panel){const I origin=panel?816:112;const I wanted=color(maps,panel?2:0,panel?error:value,panel?r.maximum:0.0625);expected_colors[panel*4096+row*64+col]=wanted;
   const I left=origin+col*513/64,right=origin+(col+1)*513/64,top=112+row*513/64,bottom=112+(row+1)*513/64;
   for(I py=top;py<bottom;++py)for(I px=left;px<right;++px){if(px==origin||py==624)continue;if(panel==0&&fringe[py*width+px])continue;require(r.pixels[py*width+px]==wanted,"actual Canvas heatmap RGB/cell geometry");++checks;}
  }
 }
 const std::regex cells(R"(<rect data-source-index="[^"]+"[^>]+/>)");I cell_count=0;
 for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),cells),end;it!=end;++it){require(cell_count<8192,"extra Poisson SVG cell");const auto tag=it->str();const I panel=cell_count/4096,index=cell_count%4096,row=index/64,col=index%64,origin=panel?816:112;
  require(std::stoll(attr(tag,"data-source-index"))==index&&std::stoll(attr(tag,"data-source-row"))==row&&std::stoll(attr(tag,"data-source-col"))==col,"Poisson SVG cell provenance");
  const I left=origin+col*513/64,right=origin+(col+1)*513/64,top=112+row*513/64,bottom=112+(row+1)*513/64;
  require(cell_coordinate(attr(tag,"x"))==left&&cell_coordinate(attr(tag,"y"))==top&&std::stoll(attr(tag,"width"))==right-left&&std::stoll(attr(tag,"height"))==bottom-top,"Poisson SVG cell geometry");require(fill_color(tag)==expected_colors[cell_count],"Poisson SVG heatmap color");++cell_count;
 }
 require(cell_count==8192,"all Poisson SVG cells checked");check_palette_svg(r,maps,0,2,0.0625,r.maximum);
 for(I panel=0;panel<2;++panel)for(I bin=0;bin<256;++bin)for(I y=688;y<=703;++y)for(I x=0;x<2;++x){const I px=(panel?816:112)+2*bin+x;require(r.pixels[y*width+px]==packed(maps[panel?2:0][panel&&r.maximum==0?128:bin]),"actual canonical colorbar pixel");++checks;}
 const auto [texts,glyph_checks]=check_text_and_export(r,ppm,30);
 std::cout<<"SCIENCE_CANVAS_PASS complete_pixels="<<r.pixels.size()<<" exact_SVG_field_cells="<<cell_count<<" exact_field_and_colorbar_pixels="<<checks<<" texts="<<texts<<" primary_glyph_pixels="<<glyph_checks<<" contour_AA_fringe_excluded_from_color_assertions_not_redrawn\n";
}
I integer(const std::string&s){I out=0;const auto v=std::from_chars(s.data(),s.data()+s.size(),out);require(v.ec==std::errc{}&&v.ptr==s.data()+s.size(),"integer payload token");return out;}
std::vector<std::vector<std::string>> records(const Input&r,const std::string&tag){std::vector<std::vector<std::string>> result;std::istringstream lines(r.all);std::string row;const auto prefix="<!-- "+tag+" ";while(std::getline(lines,row))if(row.starts_with(prefix)){require(row.ends_with(" -->"),"comment record terminator");std::istringstream fields(row.substr(prefix.size(),row.size()-prefix.size()-4));std::vector<std::string> values;std::string value;while(fields>>value)values.push_back(value);result.push_back(std::move(values));}return result;}
I fill_color(const std::string&tag){const auto text=attr(tag,"fill");const std::regex rgb(R"(rgb\(([0-9]+),([0-9]+),([0-9]+)\))");std::smatch m;require(std::regex_match(text,m,rgb),"RGB attribute");const I a=std::stoll(m[1]),b=std::stoll(m[2]),c=std::stoll(m[3]);require(a<=255&&b<=255&&c<=255,"RGB8 bounds");return (a<<16)|(b<<8)|c;}
void check_normalization(const Input&r,const Maps&maps,const std::string&ppm){
 const auto meta=records(r,"NORMALIZATION_META"),factor=records(r,"FACTOR"),rows=records(r,"COUNT");
 require(meta.size()==1&&meta[0].size()==2&&integer(meta[0][0])==96&&integer(meta[0][1])==0,"normalization status/eligibility");
 require(factor.size()==16&&rows.size()==1536,"normalization record completeness");
 std::array<double,16> sf{};std::array<double,1536> normalized{};const long double center=std::sqrt(8.0L);
 long double max_error=0;
 for(I j=0;j<16;++j){require(factor[j].size()==3&&integer(factor[j][0])==j,"size-factor identity");sf[j]=number(integer(factor[j][1]));const double log=number(integer(factor[j][2]));const long double expected=static_cast<long double>(1LL<<(j%4))/center;
  require(std::isfinite(sf[j])&&sf[j]>0&&std::fabs(sf[j]-expected)<=2e-12L*expected,"independent median-ratio size factor");require(std::isfinite(log)&&std::fabs(log-std::log(expected))<=2e-12L,"independent log-size factor");
 }
 for(I i=0;i<1536;++i){require(rows[i].size()==3&&integer(rows[i][0])==i,"count identity");const I raw=integer(rows[i][1]),expected=(i/16+1)*(1LL<<(i%4));require(raw==expected,"exact original CSR count");normalized[i]=number(integer(rows[i][2]));const long double target=(i/16+1)*center;
  require(std::isfinite(normalized[i])&&normalized[i]>0,"normalized finite payload");max_error=std::max(max_error,std::fabs(normalized[i]-target));require(std::fabs(normalized[i]-target)<=3e-12L*target,"independent normalized count");require(normalized[i]==static_cast<double>(raw)/sf[i%16],"figure consumes actual returned size factors");
 }
 const std::regex rect(R"(<rect data-source-index="[^"]+"[^>]+/>)");I cell_count=0;std::size_t pixels=0;
 for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),rect),end;it!=end;++it){const auto tag=it->str();const I panel=cell_count/1536,index=cell_count%1536,row=index/16,col=index%16,origin=panel?816:112;
  require(panel<2&&integer(attr(tag,"data-source-index"))==index&&integer(attr(tag,"data-source-row"))==row&&integer(attr(tag,"data-source-col"))==col,"SVG count-matrix provenance");
  const I x=cell_coordinate(attr(tag,"x")),y=cell_coordinate(attr(tag,"y")),w=integer(attr(tag,"width")),h=integer(attr(tag,"height"));
  const I left=origin+col*513/16,right=origin+(col+1)*513/16,top=112+row*513/96,bottom=112+(row+1)*513/96;
  require(x==left&&y==top&&w==right-left&&h==bottom-top,"independent matrix cell geometry");const double value=panel?normalized[index]:static_cast<double>((row+1)*(1LL<<(col%4)));const I wanted=color(maps,0,value,768.0);require(fill_color(tag)==wanted,"canonical normalized SVG color");
  for(I py=top;py<bottom;++py){for(I px=left;px<right;++px){if(px==origin||py==624)continue;require(r.pixels[py*width+px]==wanted,"actual normalized Canvas cell color");++pixels;}}
  ++cell_count;
 }
 require(cell_count==3072,"complete two-panel matrix SVG");
 for(I panel=0;panel<2;++panel)for(I bin=0;bin<256;++bin)for(I yy=688;yy<=703;++yy)for(I xx=0;xx<2;++xx){require(r.pixels[yy*width+(panel?816:112)+2*bin+xx]==packed(maps[0][bin]),"normalization canonical shared colorbar");++pixels;}
 // Captions belong to their panel column, not merely the whole document.
 const std::regex caption(R"(<text\s[^>]*>[^<]*</text>)");I caption_count=0;
 for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),caption),end;it!=end;++it){const auto tag=it->str();const I x=integer(attr(tag,"x")),y=integer(attr(tag,"y")),length=integer(attr(tag,"textLength"));
  if(y==31||y==55||y==79||y==751||y==775){const I origin=x<700?112:816;require(x>=origin&&x+length<=origin+512,"normalization caption exceeds its panel column");++caption_count;}
 }
 require(caption_count==9,"normalization split-caption completeness");
 require(r.svg.find("No biological quality or DE inference")!=std::string::npos&&r.svg.find("No uncertainty or coverage claim")!=std::string::npos&&r.svg.find("Shared color domain: 0 to 768 counts")!=std::string::npos&&r.svg.find("observed / checked size factor")!=std::string::npos,"normalization scientific scope captions");
 check_palette_svg(r,maps,0,0,768.0,768.0);
 const auto[t,g]=check_text_and_export(r,ppm,30);
 std::cout<<std::setprecision(18)<<"SCIENCE_NORMALIZATION_PASS CSR_values=1536 features=96 samples=16 exact_SVG_cells="<<cell_count<<" actual_Canvas_pixels="<<pixels<<" max_analytic_error="<<max_error<<" texts="<<t<<" primary_glyph_pixels="<<g<<'\n';
}
struct Rect {I x,y,w,h,part,color;};
struct Line {I x0,y0,x1,y1,color;};
void exact_line(const Input&r,const Line&line,I origin,std::size_t&pixels){
 require(line.x0==line.x1||line.y0==line.y1,"axis-aligned KM step");const I n=std::max(std::llabs(line.x1-line.x0),std::llabs(line.y1-line.y0));
 for(I j=0;j<=n;++j){const I x=n?line.x0+(line.x1-line.x0)*j/n:line.x0,y=n?line.y0+(line.y1-line.y0)*j/n:line.y0;
  if(x==origin||y==624)continue;
  require(r.pixels[y*width+x]==line.color,"actual Canvas KM step pixel");++pixels;}
}
void check_survival(const Input&r,const std::string&ppm){
 const auto meta=records(r,"SURVIVAL_META"),samples=records(r,"SAMPLE"),risks=records(r,"RISK"),curves=records(r,"CURVE");
 require(meta.size()==1&&meta[0].size()==5&&integer(meta[0][0])==12&&integer(meta[0][1])==17&&integer(meta[0][2])==7&&integer(meta[0][3])==26&&integer(meta[0][4])==26,"KM metadata/selected-gap count");
 require(samples.size()==24&&risks.size()==36&&curves.size()==288,"complete survival raw/curve records");
 for(I i=0;i<24;++i){require(samples[i].size()==4&&integer(samples[i][0])==i&&number(integer(samples[i][1]))==i/2+1&&integer(samples[i][3])==i%2,"synthetic follow-up/group identity");const I observed=i%2==0?(i%6==0?0:1):(i%8==1?0:1);require(integer(samples[i][2])==observed,"synthetic event/censor identity");}
 std::array<std::array<double,8>,36> values{};std::array<I,36> available{};
 for(I i=0;i<288;++i){require(curves[i].size()==3&&integer(curves[i][0])==i/8&&integer(curves[i][1])==i%8,"curve record order");values[i/8][i%8]=number(integer(curves[i][2]));}
 long double lo_z=0,hi_z=8;const long double target=1-static_cast<long double>(0.05)/2;
 for(I i=0;i<100;++i){const long double mid=(lo_z+hi_z)/2,cdf=(1+std::erf(mid/std::sqrt(2.L)))/2;if(cdf<target)lo_z=mid;else hi_z=mid;}
 const long double z=(lo_z+hi_z)/2;std::array<long double,3> survival{1,1,1},greenwood{},hazard{};std::array<I,3> risk{24,12,12};
 for(I time=1;time<=12;++time){for(I cohort=0;cohort<3;++cohort){const I index=3*(time-1)+cohort;I d=0,c=0;
  for(I g=0;g<2;++g){if(cohort&&g!=cohort-1)continue;const I sample=2*(time-1)+g;const I event=integer(samples[sample][2]);d+=event;c+=1-event;}
  require(risks[index].size()==5&&integer(risks[index][0])==index&&integer(risks[index][1])==risk[cohort]&&integer(risks[index][2])==d&&integer(risks[index][3])==c,"tie risk set uses deaths before censor removals");
  const I before=risk[cohort];hazard[cohort]+=static_cast<long double>(d)/before;
  const bool terminal=d==before;
  if(terminal)survival[cohort]=0;else{survival[cohort]*=static_cast<long double>(before-d)/before;greenwood[cohort]+=static_cast<long double>(d)/(before*(before-d));}
  risk[cohort]-=d+c;const I flags=terminal?35:survival[cohort]==1?47:63;available[index]=integer(risks[index][4]);require(available[index]==flags,"independent estimate/CI availability flags");
  const auto& v=values[index];require(v[0]==time&&std::isfinite(v[1])&&std::fabs(v[1]-survival[cohort])<2e-12L,"independent Kaplan-Meier product");
  if(terminal)require(std::isinf(v[2])&&std::signbit(v[2]),"genuine negative-infinite terminal log-survival");else require(std::isfinite(v[2])&&std::fabs(v[2]-std::log(survival[cohort]))<2e-12L,"independent log survival");
  if(terminal){require(v[3]==0&&v[4]==0&&v[5]==0&&v[6]==0,"unavailable Greenwood/SE/CI placeholders");}
  else{require(std::isfinite(v[3])&&std::fabs(v[3]-greenwood[cohort])<2e-12L,"independent Greenwood sum");const long double se=survival[cohort]*std::sqrt(greenwood[cohort]);require(std::isfinite(v[4])&&std::fabs(v[4]-se)<2e-12L,"independent survival standard error");
   if(survival[cohort]==1)require(v[5]==0&&v[6]==0,"singular S1 confidence limit unavailable");
   else{const long double ls=std::log(survival[cohort]),center=std::log(-ls),half=z*std::sqrt(greenwood[cohort])/std::fabs(ls),lo=std::exp(-std::exp(center+half)),hi=std::exp(-std::exp(center-half));require(std::isfinite(v[5])&&std::isfinite(v[6])&&std::fabs(v[5]-lo)<2e-11L&&std::fabs(v[6]-hi)<2e-11L&&v[5]<=v[1]&&v[1]<=v[6],"independent pointwise log-log confidence limits");}
  }
  require(std::isfinite(v[7])&&std::fabs(v[7]-hazard[cohort])<2e-12L,"independent Nelson-Aalen hazard");
 }}
 std::vector<Line> expected_lines;std::vector<Rect> expected_ci;std::size_t pixels=0;
 for(I group=0;group<2;++group){const I origin=group?816:112,rgb=group?0xeb8246:0x46b4eb;I oldx=origin,oldy=112;
  for(I row=0;row<12;++row){const I index=3*row+group+1;const auto&v=values[index];const I x=origin+(row+1)*512/12,y=624-static_cast<I>(std::floor(v[1]*512));expected_lines.push_back({oldx,oldy,x,oldy,rgb});expected_lines.push_back({x,oldy,x,y,rgb});oldx=x;oldy=y;
   if(available[index]&16){const I low=624-static_cast<I>(std::floor(v[5]*512)),high=624-static_cast<I>(std::floor(v[6]*512));expected_ci.push_back({x,high,1,low-high+1,0,rgb});expected_ci.push_back({x-2,low,5,1,1,rgb});expected_ci.push_back({x-2,high,5,1,2,rgb});expected_ci.push_back({x-1,y-1,3,3,3,rgb});}
  }
 }
 const std::regex line(R"(<line data-dynamic-part="line"[^>]+/>)");std::size_t count=0;
 for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),line),end;it!=end;++it){require(count<expected_lines.size(),"extra KM SVG step");const auto&want=expected_lines[count];const auto tag=it->str();require(integer(attr(tag,"x1"))==want.x0&&integer(attr(tag,"y1"))==want.y0&&integer(attr(tag,"x2"))==want.x1&&integer(attr(tag,"y2"))==want.y1,"independent KM step SVG geometry");require(fill_color(std::regex_replace(tag,std::regex("stroke="),"fill="))==want.color,"KM group color identity");exact_line(r,want,count<24?112:816,pixels);++count;}
 require(count==48&&count==expected_lines.size(),"complete KM step paths without extrapolation");
 const std::regex rect(R"(<rect data-dynamic-interval-part="[0-9]+"[^>]+/>)");count=0;
 for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),rect),end;it!=end;++it){require(count<expected_ci.size(),"extra CI primitive");const auto&v=expected_ci[count++];const auto tag=it->str();require(integer(attr(tag,"data-dynamic-interval-part"))==v.part&&cell_coordinate(attr(tag,"x"))==v.x&&cell_coordinate(attr(tag,"y"))==v.y&&integer(attr(tag,"width"))==v.w&&integer(attr(tag,"height"))==v.h&&fill_color(tag)==v.color,"independent pointwise CI geometry/provenance");for(I yy=v.y;yy<v.y+v.h;++yy)for(I xx=v.x;xx<v.x+v.w;++xx){require(r.pixels[yy*width+xx]==v.color,"actual Canvas confidence-limit pixel");++pixels;}}
 require(count==80&&count==expected_ci.size(),"only available pointwise CIs are drawn");
 const std::regex interval(R"(<metadata data-dynamic-series="supplied-intervals"[^>]+/>)");I metadata=0;for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),interval),end;it!=end;++it){const auto tag=it->str();require(attr(tag,"data-interval-kind")=="1"&&std::stod(attr(tag,"data-interval-level"))==0.95,"confidence kind and level, not prediction");++metadata;}require(metadata==2,"two pointwise interval metadata records");
 require(r.svg.find("POINTWISE asymptotic Greenwood / log-log")!=std::string::npos&&r.svg.find("Independent censoring assumptions are unassessed")!=std::string::npos,"survival scientific captions");
 const auto[t,g]=check_text_and_export(r,ppm,30);std::cout<<"SCIENCE_SURVIVAL_PASS records=36 exact_risk_sets=36 independently_checked_curve_fields=288 KM_step_segments=48 CI_rectangles=80 actual_pixels="<<pixels<<" texts="<<t<<" glyphs="<<g<<" pointwise_asymptotic_not_coverage_certificate\n";
}
std::pair<I,I> scaled_projection(double v,double lo,double hi,I origin,bool reverse){const long double unit=(static_cast<long double>(v)-lo)/(static_cast<long double>(hi)-lo);return projection(unit,origin,reverse);}
void check_ode(const Input&r,const std::string&ppm){
 const auto meta=records(r,"DP54_META"),bracket=records(r,"DP54_BRACKET"),rows=records(r,"DENSE");require(meta.size()==1&&meta[0].size()==11&&bracket.size()==1&&bracket[0].size()==4&&rows.size()==65,"complete actual DP54/event records");
 const auto&m=meta[0];require(integer(m[0])==5,"successful terminal event, not ordinary endpoint");const double begin=number(integer(m[1])),end=number(integer(m[2])),event=number(integer(m[3])),lo=number(integer(m[4])),hi=number(integer(m[5]));
 require(std::isfinite(begin)&&std::isfinite(end)&&std::isfinite(event)&&std::isfinite(lo)&&std::isfinite(hi)&&begin<end&&end==event&&lo<=event&&event<=hi&&begin<=lo&&hi<=4.0,"computed-interpolant event bracket/domain");
 require(hi-lo<=1.1e-11,"declared absolute event localization criterion");
 const long double analytic_time=-2*std::log(static_cast<long double>(0.4));require(std::fabs(event-analytic_time)<2e-9L,"independent analytic decay event time");
 // The bracket certifies localization of the computed interpolation root,
 // never a guaranteed enclosure of the true ODE event time.
 const I rhs=integer(m[6]),ev=integer(m[7]),root=integer(m[8]);require(rhs==integer(m[9])&&ev==integer(m[10])&&rhs>0&&rhs<=10000&&ev>0&&ev<=100000&&root>0&&root<=10000,"actual callback counts and work budgets; no extra dense-query callbacks");
 const double glo=number(integer(bracket[0][0])),ghi=number(integer(bracket[0][1])),g=number(integer(bracket[0][2])),final=number(integer(bracket[0][3]));require(std::isfinite(glo)&&std::isfinite(ghi)&&std::isfinite(g)&&std::isfinite(final)&&glo>=0&&ghi<=0,"decreasing physical-time event direction");require(std::fabs(final-0.4)<1e-10&&std::fabs(g)<1e-10,"computed event state/functional");
 std::array<double,65> t{},value{},derivative{};long double maximum=0,slope_max=0;
 for(I i=0;i<65;++i){require(rows[i].size()==4&&integer(rows[i][0])==i,"dense sample identity");t[i]=number(integer(rows[i][1]));value[i]=number(integer(rows[i][2]));derivative[i]=number(integer(rows[i][3]));
  require(std::isfinite(t[i])&&std::isfinite(value[i])&&std::isfinite(derivative[i])&&t[i]>=begin&&t[i]<=end,"finite available dense payload");if(i)require(t[i]>t[i-1],"represented dense query times strictly increasing");
  const long double target=std::exp(-0.5L*t[i]);maximum=std::max(maximum,std::fabs(value[i]-target));slope_max=std::max(slope_max,std::fabs(derivative[i]+0.5L*target));
  require(std::fabs(value[i]-target)<2e-9L&&std::fabs(derivative[i]+0.5L*target)<1e-7L,"independent exponential dense value/derivative");
  const long double query=i==64?end:static_cast<long double>(begin)+static_cast<long double>(i)/64*(static_cast<long double>(end)-begin);require(std::fabs(t[i]-query)<=8*std::numeric_limits<double>::epsilon()*(1+std::fabs(query)),"caller query grid is confined to retained segment");
 }
 require(t[0]==begin&&t[64]==end&&std::fabs(value[64]-final)<1e-12,"queried final event endpoint");
 const std::regex re(R"(<line data-dynamic-part="line"[^>]+/>)");I count=0;std::size_t pixels=0;
 for(std::sregex_iterator it(r.svg.begin(),r.svg.end(),re),last;it!=last;++it){const auto tag=it->str();require(count<128,"extra dense SVG segment");const I panel=count/64,index=count%64,origin=panel?816:112,rgb=panel?0xeb8246:0x46b4eb;const double ymin=panel?-0.22:0.39,ymax=panel?-0.195:0.44;
  const auto xa=scaled_projection(t[index],begin,end,origin,false),xb=scaled_projection(t[index+1],begin,end,origin,false),ya=scaled_projection(panel?derivative[index]:value[index],ymin,ymax,112,true),yb=scaled_projection(panel?derivative[index+1]:value[index+1],ymin,ymax,112,true);
  const I x0=integer(attr(tag,"x1")),y0=integer(attr(tag,"y1")),x1=integer(attr(tag,"x2")),y1=integer(attr(tag,"y2"));require(x0>=xa.first&&x0<=xa.second&&y0>=ya.first&&y0<=ya.second&&x1>=xb.first&&x1<=xb.second&&y1>=yb.first&&y1<=yb.second,"independent actual dense sample projection");
  require(fill_color(std::regex_replace(tag,std::regex("stroke="),"fill="))==rgb,"dense series identity colors");
  // Integer endpoints have full AA coverage; the interior AA blend is not
  // reconstructed by this oracle and no external SVG raster is substituted.
  for(const auto&[x,y]:{std::pair{x0,y0},std::pair{x1,y1}}){if(x==origin||y==624)continue;require(r.pixels[y*width+x]==rgb,"actual Canvas dense segment endpoint");++pixels;}
  ++count;
 }
 require(count==128,"complete retained-segment polylines");require(r.svg.find("Only the final retained segment is shown")!=std::string::npos&&r.svg.find("not statistical coverage guarantees")!=std::string::npos,"dense-output scientific captions");
 const auto[texts,glyphs]=check_text_and_export(r,ppm,24);std::cout<<std::setprecision(18)<<"SCIENCE_DP54_PASS dense_queries=65 real_SVG_segments=128 actual_endpoint_pixels="<<pixels<<" max_state_error="<<maximum<<" max_derivative_error="<<slope_max<<" RHS_calls="<<rhs<<" event_calls="<<ev<<" texts="<<texts<<" glyphs="<<glyphs<<" event_bracket_is_computed_interpolant_only\n";
}
void mutate(const std::string&mode,const std::string&input,const std::string&output){
 require(std::filesystem::absolute(input).lexically_normal()!=std::filesystem::absolute(output).lexically_normal(),"mutation input/output distinct");
 auto data=contents(input);const auto original=data;bool changed=false;
 if(mode=="--mutate-node") {const std::regex re(R"(<!-- NODE 2112 [-0-9]+ -->)");std::smatch m;require(std::regex_search(data,m,re),"node mutation target");data.replace(m.position(),m.length(),"<!-- NODE 2112 0 -->");changed=true;}
 else if(mode=="--mutate-factor") {const std::regex re(R"(<!-- FACTOR 0 ([-0-9]+) ([-0-9]+) -->)");std::smatch m;require(std::regex_search(data,m,re),"factor mutation target");data.replace(m.position(),m.length(),"<!-- FACTOR 0 0 "+m[2].str()+" -->");changed=true;}
 else if(mode=="--mutate-ci-kind") {const std::string before="data-interval-kind=\"1\"";const auto at=data.find(before);require(at!=std::string::npos,"confidence mutation target");data.replace(at,before.size(),"data-interval-kind=\"2\"");changed=true;}
 else if(mode=="--mutate-dense") {const std::regex re(R"(<!-- DENSE 32 ([-0-9]+) ([-0-9]+) ([-0-9]+) -->)");std::smatch m;require(std::regex_search(data,m,re),"dense mutation target");data.replace(m.position(),m.length(),"<!-- DENSE 32 "+m[1].str()+" "+m[2].str()+" 0 -->");changed=true;}
 else if(mode=="--mutate-caption-column") {const std::string before="<text x=\"112\" y=\"55\"";const auto at=data.find(before);require(at!=std::string::npos,"caption column mutation target");data.replace(at,before.size(),"<text x=\"612\" y=\"55\"");changed=true;}
 else if(mode=="--mutate-label-bounds") {const std::regex re(R"re(<text x="([0-9]+)" y="719")re");std::smatch m;require(std::regex_search(data,m,re),"global text bounds mutation target");data.replace(m.position(1),m.length(1),"1479");changed=true;}
 else if(mode=="--mutate-svg-color") {const std::regex re(R"re(<rect data-source-index="0"[^\n]+fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)")re");std::smatch m;require(std::regex_search(data,m,re),"SVG color mutation target");data.replace(m.position(3),m.length(3),std::to_string(std::stoll(m[3])^1));changed=true;}
 else if(mode=="--mutate-color-domain") {const std::regex re(R"re(<g data-dynamic-series="heatmap"[^\n]+data-palette="inferno"[^\n]+data-color-max="([^"]+)")re");std::smatch m;require(std::regex_search(data,m,re),"tiny domain mutation target");data.replace(m.position(1),m.length(1),"0");changed=true;}
 else if(mode=="--mutate-segment") {const std::regex re(R"re(<line data-contour-cell="[^\n]+? x1="([0-9]+)")re");std::smatch m;require(std::regex_search(data,m,re),"geometry mutation target");const auto start=m.position(1);data.replace(start,m.length(1),std::to_string(std::stoll(m[1])+2));changed=true;}
 else if(mode=="--mutate-pixel") {std::istringstream rows(data);std::ostringstream out;std::string row;while(std::getline(rows,row)){std::istringstream fields(row);std::string a,tag;I y,x,n,rgb;fields>>a>>tag;if(a=="<!--"&&tag=="PIX"){fields>>y>>x>>n>>rgb;if(y==200&&x<=900&&x+n>900){require(!changed,"single pixel mutation target");row="<!-- PIX "+std::to_string(y)+" "+std::to_string(x)+" "+std::to_string(n)+" "+std::to_string(rgb^1)+" -->";changed=true;}}out<<row<<'\n';}data=out.str();}
 require(changed&&data!=original&&!data.empty(),"mutation actually changes nonempty input");std::ofstream f(output,std::ios::binary);require(bool(f),"mutation output");f<<data;require(bool(f),"mutation write");
}
}
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1]).starts_with("--mutate-")){science::mutate(argv[1],argv[2],argv[3]);return 0;}
 if(argc!=4&&argc!=5)throw std::runtime_error("usage: --poisson|--normalization|--survival|--dp54 PRIMARY_LUT_TEXT NATIVE_STDOUT [ACTUAL_CANVAS_PPM]");
 const std::string mode=argv[1],ppm=argc==5?argv[4]:"";const auto maps=read_maps(argv[2]);
 if(mode=="--poisson"){const auto data=science::read(argv[3]);science::check_science(data);science::check_segments(data);science::check_pixels(data,maps,ppm);}
 else if(mode=="--normalization"){const auto data=science::read(argv[3],"SCIENTIFIC_NORMALIZATION_COMPLETE");science::check_normalization(data,maps,ppm);}
 else if(mode=="--survival"){const auto data=science::read(argv[3],"SCIENTIFIC_SURVIVAL_COMPLETE");science::check_survival(data,ppm);}
 else if(mode=="--dp54"){const auto data=science::read(argv[3],"SCIENTIFIC_DP54_COMPLETE");science::check_ode(data,ppm);}
 else throw std::runtime_error("unknown mode");
 std::cout<<"SCIENTIFIC_NATIVE_WORKFLOW_REFERENCE_PASS\n";return 0;
 }catch(const std::exception&e){std::cerr<<"scientific-native: "<<e.what()<<'\n';return 1;}}
