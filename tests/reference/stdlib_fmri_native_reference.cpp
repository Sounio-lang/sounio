#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
struct Input:std::runtime_error {using std::runtime_error::runtime_error;};
static void need(bool ok,const char* text){if(!ok){throw Input(text);}}
using Bytes=std::vector<unsigned char>;
static Bytes read(const fs::path& p){std::ifstream f(p,std::ios::binary);need(bool(f),"input open");Bytes b((std::istreambuf_iterator<char>(f)),{});need(!b.empty()&&b.size()<=65536,"input size");return b;}
static void write(const fs::path&p,const Bytes&b){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(b.data()),static_cast<std::streamsize>(b.size()));need(bool(f),"output write");}
static std::uint64_t word(const Bytes&b,std::size_t p,int n,bool big){
 need(p+static_cast<std::size_t>(n)<=b.size(),"word extent");std::uint64_t x=0;
 for(int i=0;i<n;++i){const int shift=big?n-1-i:i;x|=std::uint64_t(b[p+static_cast<std::size_t>(i)])<<(8*shift);}
 return x;
}
static void put(Bytes&b,std::size_t p,int n,std::uint64_t x,bool big=false){need(p+static_cast<std::size_t>(n)<=b.size(),"put extent");for(int i=0;i<n;++i){b[p+static_cast<std::size_t>(i)]=static_cast<unsigned char>(x>>(8*(big?n-1-i:i)));}}
struct Image {Bytes bytes;bool big=false;int version=0;std::uint64_t nx=0,ny=0,nz=0,nt=0,offset=0,nvox=0;};
static Image image(const fs::path&p,bool atlas=false){
 Image a;a.bytes=read(p);auto h=word(a.bytes,0,4,false);
 if(h!=348&&h!=540){a.big=true;h=word(a.bytes,0,4,true);}
 need(h==348||h==540,"header size");a.version=h==348?1:2;
 need(word(a.bytes,h==348?70:12,2,a.big)==2&&word(a.bytes,h==348?72:14,2,a.big)==8,"datatype");
 const std::size_t magic=h==348?344:4;
 need(a.bytes.size()>magic+3&&a.bytes[magic]=='n'&&a.bytes[magic+1]=='+'&&a.bytes[magic+2]==(h==348?'1':'2')&&a.bytes[magic+3]==0,"magic");
 const auto rank=word(a.bytes,h==348?40:16,h==348?2:8,a.big);
 need(atlas?(rank==3||rank==4):rank==4,"input dimensionality");
 const auto base=h==348?42:24;const int width=h==348?2:8;
 a.nx=word(a.bytes,base,width,a.big);a.ny=word(a.bytes,base+width,width,a.big);a.nz=word(a.bytes,base+2*width,width,a.big);a.nt=word(a.bytes,base+3*width,width,a.big);
 if(rank==3)a.nt=1;
 need(a.nx>0&&a.ny>0&&a.nz>0&&a.nx<=4096&&a.ny<=4096&&a.nz<=4096&&(atlas?a.nt==1:(a.nt>=2&&a.nt<=500)),"dimensions");
 a.nvox=a.nx*a.ny*a.nz;need(a.nvox<=4096/a.nt,"sample budget");
 if(h==348){
  const float offset=std::bit_cast<float>(static_cast<std::uint32_t>(word(a.bytes,108,4,a.big)));
  const float slope=std::bit_cast<float>(static_cast<std::uint32_t>(word(a.bytes,112,4,a.big)));
  const float intercept=std::bit_cast<float>(static_cast<std::uint32_t>(word(a.bytes,116,4,a.big)));
  need(std::isfinite(offset)&&offset>=352&&offset<=65536&&std::floor(offset)==offset,"offset");
  need(slope==0||(slope==1&&intercept==0),"scaling");a.offset=static_cast<std::uint64_t>(offset);
 }else{
  a.offset=word(a.bytes,168,8,a.big);
  const double slope=std::bit_cast<double>(word(a.bytes,176,8,a.big)),intercept=std::bit_cast<double>(word(a.bytes,184,8,a.big));
  need(a.offset>=544&&a.offset<=65536,"offset");need(slope==0||(slope==1&&intercept==0),"scaling");
 }
 need(a.offset<=a.bytes.size()&&a.nvox*a.nt<=a.bytes.size()-a.offset,"payload extent");
 return a;
}
static std::vector<std::vector<long double>> motion(const fs::path&p){
 const auto bytes=read(p);std::string text(bytes.begin(),bytes.end());std::istringstream in(text);std::string line;
 need(bool(std::getline(in,line)),"motion header");if(!line.empty()&&line.back()=='\r'){line.pop_back();}
 need(line=="tx\tty\ttz\trx\try\trz","motion columns");
 std::vector<std::vector<long double>> rows;
 while(std::getline(in,line)){
  if(!line.empty()&&line.back()=='\r'){line.pop_back();}need(!line.empty(),"blank motion row");
  std::vector<long double> values;std::size_t start=0;
  for(;;){
   auto end=line.find('\t',start);if(end==std::string::npos){end=line.size();}
   const auto token=line.substr(start,end-start);need(!token.empty(),"empty motion field");
   std::size_t pos=0;if(token[pos]=='+'||token[pos]=='-'){++pos;}bool digit=false,point=false;int fraction=0;
   for(;pos<token.size();++pos){char c=token[pos];if(c=='.'&&!point){point=true;continue;}need(c>='0'&&c<='9',"motion number");digit=true;if(point){++fraction;}}
   need(digit&&fraction<=9,"motion precision");char* stop=nullptr;errno=0;long double v=std::strtold(token.c_str(),&stop);
   need(errno==0&&stop==token.c_str()+token.size()&&std::isfinite(v)&&std::fabs(v)<=9223372036.854775807L,"motion range");values.push_back(v);
   if(end==line.size()){break;}start=end+1;
  }
  need(values.size()==6,"motion width");rows.push_back(values);
 }
 need(!rows.empty()&&rows.size()<=500,"motion rows");return rows;
}

struct IndexConfig {
 std::array<double,5> weights{},scales{};
 std::int64_t formula=0,coverage_mode=0,expected_regions=0,alignment_policy=0;
};
static const std::array<std::string,5> feature_names{"fd_mean","dvars","inverse_tsnr","fc_ci_width","uncovered_fraction"};
static std::vector<std::string> fields(const std::string& line){
 std::vector<std::string> out;std::size_t at=0;
 for(;;){const auto end=line.find('\t',at);out.push_back(line.substr(at,end==std::string::npos?end:end-at));if(end==std::string::npos)break;at=end+1;}
 return out;
}
static std::int64_t config_integer(const std::string& token){
 need(!token.empty(),"empty config integer");auto first=token.data();const auto end=first+token.size();if(*first=='+')++first;
 std::int64_t value=0;const auto result=std::from_chars(first,end,value);
 need(first!=end&&result.ec==std::errc{}&&result.ptr==end,"config integer");return value;
}
static double config_number(const std::string& token){
 need(!token.empty(),"empty config number");std::string lower=token;
 std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
 if(lower=="nan"||lower=="+nan"||lower=="-nan")return std::bit_cast<double>(std::uint64_t{0x7ff8000000000000ULL});
 if(lower=="inf"||lower=="+inf"||lower=="infinity"||lower=="+infinity")return std::numeric_limits<double>::infinity();
 if(lower=="-inf"||lower=="-infinity")return -std::numeric_limits<double>::infinity();
 auto first=token.data();const auto end=first+token.size();if(*first=='+')++first;
 double value=0;const auto result=std::from_chars(first,end,value,std::chars_format::general);
 need(first!=end&&result.ec==std::errc{}&&result.ptr==end,"config number");return value;
}
static IndexConfig config(const fs::path& path){
 const auto bytes=read(path);std::istringstream in(std::string(bytes.begin(),bytes.end()));std::string line;std::vector<std::vector<std::string>> rows;
 while(std::getline(in,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();need(!line.empty(),"blank config row");rows.push_back(fields(line));}
 need(rows.size()==9,"config row count");IndexConfig c;
 const std::array<std::string,4> metadata{"formula","coverage_mode","expected_regions","alignment_policy"};
 std::array<std::int64_t,4> values{};
 for(std::size_t i=0;i<metadata.size();++i){need(rows[i].size()==2&&rows[i][0]==metadata[i],"config metadata");values[i]=config_integer(rows[i][1]);}
 c.formula=values[0];c.coverage_mode=values[1];c.expected_regions=values[2];c.alignment_policy=values[3];
 for(std::size_t i=0;i<feature_names.size();++i){need(rows[i+4].size()==3&&rows[i+4][0]==feature_names[i],"config feature rows");c.weights[i]=config_number(rows[i+4][1]);c.scales[i]=config_number(rows[i+4][2]);}
 return c;
}
static void valid_config(const IndexConfig& c){
 need(c.formula==1&&c.alignment_policy==1,"invalid config formula/alignment");
 need((c.coverage_mode==1||c.coverage_mode==2)&&c.expected_regions>=1&&c.expected_regions<=255,"invalid config coverage/regions");
 double maximum=0;
 for(std::size_t i=0;i<5;++i){need(std::isfinite(c.weights[i])&&c.weights[i]>=0&&std::isfinite(c.scales[i])&&c.scales[i]>0,"invalid config weights/scales");maximum=std::max(maximum,c.weights[i]);}
 need(maximum>0,"zero config weights");
}
static std::string number_token(double value){
 if(std::isnan(value))return "NaN";
 if(std::isinf(value))return std::signbit(value)?"-Inf":"Inf";
 std::ostringstream out;out<<std::setprecision(std::numeric_limits<double>::max_digits10)<<value;return out.str();
}
static Bytes config_bytes(const IndexConfig& c){
 std::ostringstream out;out<<"formula\t"<<c.formula<<"\ncoverage_mode\t"<<c.coverage_mode<<"\nexpected_regions\t"<<c.expected_regions<<"\nalignment_policy\t"<<c.alignment_policy<<"\n";
 for(std::size_t i=0;i<5;++i)out<<feature_names[i]<<'\t'<<number_token(c.weights[i])<<'\t'<<number_token(c.scales[i])<<'\n';
 const auto text=out.str();return Bytes(text.begin(),text.end());
}
static std::string sounio_number(double value){
 if(!std::isfinite(value)||std::fpclassify(value)==FP_SUBNORMAL){
  return "bits_to_f64("+std::to_string(std::bit_cast<std::int64_t>(value))+")";
 }
 auto text=number_token(value);if(text.find_first_of(".eE")==std::string::npos)text+=".0";return text;
}
static std::string sounio_integer(std::int64_t value){
 if(value==std::numeric_limits<std::int64_t>::min())return "(0-9223372036854775807-1)";
 return std::to_string(value);
}
static void emit_config(const IndexConfig& c){
 // Semantic-invalid configs intentionally reach the Sounio constructor; its
 // status17 is tested. Malformed TSV syntax is still refused before emission.
 std::cout<<"fmri_index_config_new([";
 for(std::size_t i=0;i<5;++i){if(i)std::cout<<',';std::cout<<sounio_number(c.weights[i]);}
 std::cout<<"],[";
 for(std::size_t i=0;i<5;++i){if(i)std::cout<<',';std::cout<<sounio_number(c.scales[i]);}
 std::cout<<"],"<<sounio_integer(c.coverage_mode)<<','<<sounio_integer(c.expected_regions)<<','<<sounio_integer(c.alignment_policy)<<','<<sounio_integer(c.formula)<<")\n";
}

static long double median(std::vector<long double> a){need(!a.empty(),"empty median");std::sort(a.begin(),a.end());const auto n=a.size();return n%2?a[n/2]:(a[n/2-1]+a[n/2])/2;}
struct Expected {std::map<std::string,long double> values;std::map<std::string,std::string> unavailable;};
static Expected calculate(const Image&a,const Image&b,const std::vector<std::vector<long double>>&m,const Image&atlas,const IndexConfig&c){
 valid_config(c);
 need(a.nx==b.nx&&a.ny==b.ny&&a.nz==b.nz&&a.nt==b.nt&&m.size()==a.nt,"paired dimensions");
 Expected e;e.values={{"nifti1_status",1},{"nifti2_status",1}};
 long double fd_sum=0,fd_max=0;std::size_t scrubbed=0;
 for(std::size_t t=1;t<m.size();++t){long double fd=0;for(int c=0;c<6;++c){fd+=std::fabs(m[t][c]-m[t-1][c])*(c<3?1:50);}fd_sum+=fd;fd_max=std::max(fd_max,fd);if(fd>0.2L){++scrubbed;}}
 e.values["qc_fd_mean"]=fd_sum/(m.size()-1);e.values["qc_fd_max"]=fd_max;e.values["qc_scrub_fraction"]=static_cast<long double>(scrubbed)/m.size();
 auto sample=[](const Image&i,std::uint64_t t,std::uint64_t v){return static_cast<long double>(i.bytes[i.offset+t*i.nvox+v]);};
 std::vector<long double> tsnr,dvars;bool tsnr_ok=true;
 for(std::uint64_t v=0;v<a.nvox;++v){
  long double sum=0;for(std::uint64_t t=0;t<a.nt;++t){sum+=sample(a,t,v);}const auto mean=sum/a.nt;
  long double square=0;for(std::uint64_t t=0;t<a.nt;++t){const auto d=sample(a,t,v)-mean;square+=d*d;}
  if(square<=0){tsnr_ok=false;}else{tsnr.push_back(mean/std::sqrt(square/(a.nt-1)));}
 }
 if(tsnr_ok){e.values["qc_tsnr_median"]=median(tsnr);}else{e.unavailable["qc_tsnr_median"]="zero_temporal_variance";}
 for(std::uint64_t t=1;t<a.nt;++t){long double square=0;for(std::uint64_t v=0;v<a.nvox;++v){const auto d=sample(a,t,v)-sample(a,t-1,v);square+=d*d;}dvars.push_back(std::sqrt(square/a.nvox));}
 e.values["qc_dvars_median"]=median(dvars);
 std::vector<long double>x(a.nt),y(a.nt);long double mx=0,my=0;
 for(std::uint64_t t=0;t<a.nt;++t){for(std::uint64_t v=0;v<a.nvox;++v){x[t]+=sample(a,t,v);y[t]+=sample(b,t,v);}x[t]/=a.nvox;y[t]/=a.nvox;mx+=x[t];my+=y[t];}
 mx/=a.nt;my/=a.nt;long double xx=0,yy=0,xy=0;
 for(std::uint64_t t=0;t<a.nt;++t){const auto dx=x[t]-mx,dy=y[t]-my;xx+=dx*dx;yy+=dy*dy;xy+=dx*dy;}
 if(xx<=0||yy<=0){e.unavailable["fc_r"]="zero_global_signal_variance";e.unavailable["fc_ci_width"]="insufficient_or_constant_timepoints";}
 else{
  const long double r=std::clamp(xy/std::sqrt(xx*yy),-1.0L,1.0L);e.values["fc_r"]=r;
  if(a.nt<=3){e.unavailable["fc_ci_width"]="insufficient_or_constant_timepoints";}
  else if(r==1||r==-1){e.values["fc_ci_width"]=0;}
  else{const auto z=std::atanh(r),radius=1.96L/std::sqrt(static_cast<long double>(a.nt-3));e.values["fc_ci_width"]=std::tanh(z+radius)-std::tanh(z-radius);}
 }

 need(atlas.nx==a.nx&&atlas.ny==a.ny&&atlas.nz==a.nz&&atlas.nt==1,"atlas dimensions");
 std::array<bool,256> seen{};std::uint64_t labeled=0,regions=0;
 for(std::uint64_t v=0;v<a.nvox;++v){
  const auto label=atlas.bytes[atlas.offset+v];need(label<=c.expected_regions,"atlas label bound");
  if(label!=0){++labeled;if(!seen[label]){seen[label]=true;++regions;}}
 }
 const long double coverage=c.coverage_mode==1?static_cast<long double>(labeled)/a.nvox:static_cast<long double>(regions)/c.expected_regions;
 e.values["atlas_labeled_voxels"]=labeled;e.values["atlas_present_regions"]=regions;e.values["atlas_coverage_ratio"]=coverage;
 const bool tsnr_required=c.weights[2]>0,ci_required=c.weights[3]>0;
 if((tsnr_required&&(!e.values.contains("qc_tsnr_median")||e.values.at("qc_tsnr_median")<=0))||(ci_required&&!e.values.contains("fc_ci_width"))){
  e.unavailable["uncertainty_score"]="required_metric_unavailable";
  return e;
 }
 const std::array<long double,5> features{e.values.at("qc_fd_mean"),e.values.at("qc_dvars_median"),e.values.contains("qc_tsnr_median")&&e.values.at("qc_tsnr_median")>0?1/e.values.at("qc_tsnr_median"):0,e.values.contains("fc_ci_width")?e.values.at("fc_ci_width"):0,1-coverage};
 const double maximum=*std::max_element(c.weights.begin(),c.weights.end());
 long double numerator=0,denominator=0;double binary_numerator=0,binary_denominator=0;
 for(std::size_t i=0;i<5;++i)if(c.weights[i]>0){
  const double fraction=c.weights[i]/maximum,feature=static_cast<double>(features[i]),scaled=feature/c.scales[i],term=fraction*scaled;
  need(std::isfinite(fraction)&&fraction!=0&&std::isfinite(scaled)&&!(feature!=0&&scaled==0)&&std::isfinite(term)&&!(scaled!=0&&term==0),"config index arithmetic");
  binary_numerator+=term;binary_denominator+=fraction;need(std::isfinite(binary_numerator),"config index sum");
  const long double normalized=static_cast<long double>(c.weights[i])/maximum;
  numerator+=normalized*(features[i]/c.scales[i]);denominator+=normalized;
 }
 const double binary_result=binary_numerator/binary_denominator;
 need(std::isfinite(binary_result)&&!(binary_numerator!=0&&binary_result==0),"config index result");
 e.values["uncertainty_score"]=numerator/denominator;
 return e;

}
static void check(const Expected&e,const fs::path&p){
 std::ifstream in(p);need(bool(in),"output open");std::string line;std::map<std::string,long double> got;std::map<std::string,std::string> missing;bool marker=false;
 while(std::getline(in,line)){
  if(line=="SCIENCE_FMRI_OK"){if(marker){throw std::runtime_error("duplicate marker");}marker=true;}
  else if(line.rfind("FMRI_BITS ",0)==0){std::istringstream row(line);std::string tag,name,extra;std::int64_t bits=0;need(bool(row>>tag>>name>>bits)&&!(row>>extra),"bit row");if(got.contains(name)){throw std::runtime_error("duplicate metric");}got[name]=std::bit_cast<double>(bits);}
  else if(line.rfind("SCIENCE_UNAVAILABLE fmri ",0)==0){std::istringstream row(line);std::string tag,lane,name,reason,extra;need(bool(row>>tag>>lane>>name>>reason)&&!(row>>extra),"availability row");if(missing.contains(name)){throw std::runtime_error("duplicate availability");}missing[name]=reason;}
  else if(line.rfind("SCIENCE_METRIC fmri ",0)!=0){throw std::runtime_error("unexpected output");}
 }
 if(!marker||got.size()!=e.values.size()||missing!=e.unavailable){throw std::runtime_error("missing metrics or availability");}
 for(const auto&[name,want]:e.values){const auto it=got.find(name);if(it==got.end()||!std::isfinite(it->second)||std::fabs(it->second-want)>1e-11L*std::max(1.0L,std::fabs(want))){std::cerr<<std::setprecision(21)<<name<<" got="<<(it==got.end()?0:it->second)<<" expected="<<want<<"\n";throw std::runtime_error("numerical mismatch");}}
}
static Bytes minimal_big(const Image&a){
 Bytes b(a.version==1?352:544,0);const bool big=true;put(b,0,4,a.version==1?348:540,big);
 put(b,a.version==1?40:16,a.version==1?2:8,4,big);
 const int base=a.version==1?42:24,w=a.version==1?2:8;const std::uint64_t dims[]{a.nx,a.ny,a.nz,a.nt};
 for(int i=0;i<4;++i){put(b,base+i*w,w,dims[i],big);}
 put(b,a.version==1?70:12,2,2,big);put(b,a.version==1?72:14,2,8,big);
 if(a.version==1){for(int i=0;i<8;++i){put(b,76+4*i,4,std::bit_cast<std::uint32_t>(1.0f),big);}put(b,108,4,std::bit_cast<std::uint32_t>(352.0f),big);b[344]='n';b[345]='+';b[346]='1';}
 else{for(int i=0;i<8;++i){put(b,104+8*i,8,std::bit_cast<std::uint64_t>(1.0),big);}put(b,168,8,544,big);const unsigned char magic[]{'n','+','2',0,13,10,26,10};std::copy(magic,magic+8,b.begin()+4);}
 b.insert(b.end(),a.bytes.begin()+static_cast<std::ptrdiff_t>(a.offset),a.bytes.begin()+static_cast<std::ptrdiff_t>(a.offset+a.nvox*a.nt));return b;
}
static void mutants(const Image&a,const Image&b,const Bytes&motion_bytes,const Image&atlas,const IndexConfig&c,const Bytes&config_raw,const fs::path&out){
 need(!fs::exists(out),"existing mutants");fs::create_directories(out);std::ofstream plan(out/"PLAN.tsv");
 auto add=[&](std::string name,Bytes x,Bytes y,Bytes m,int exit){
  const auto d=out/name;fs::create_directory(d);write(d/"a.nii",x);write(d/"b.nii",y);write(d/"motion.tsv",m);write(d/"atlas.nii",atlas.bytes);write(d/"config.tsv",config_raw);plan<<name<<'\t'<<exit<<'\n';
 };
 add("baseline",a.bytes,b.bytes,motion_bytes,0);
 auto x=a.bytes;x[a.offset]=static_cast<unsigned char>((x[a.offset]+37)%256);add("voxel-change",x,b.bytes,motion_bytes,0);
 std::string text(motion_bytes.begin(),motion_bytes.end());auto p=text.find('\n');need(p!=std::string::npos,"motion mutation");p=text.find('\n',p+1);need(p!=std::string::npos,"motion mutation row");const auto stop=text.find('\t',p+1);need(stop!=std::string::npos,"motion mutation field");text.replace(p+1,stop-p-1,"0.18");add("motion-change",a.bytes,b.bytes,Bytes(text.begin(),text.end()),0);
 x=a.bytes;std::fill(x.begin()+static_cast<std::ptrdiff_t>(a.offset),x.begin()+static_cast<std::ptrdiff_t>(a.offset+a.nvox*a.nt),1);add("constant-signal",x,b.bytes,motion_bytes,0);
 x=a.bytes;x[a.version==1?344:4]='x';add("bad-magic",x,b.bytes,motion_bytes,13);
 x=a.bytes;x.pop_back();add("truncated-payload",x,b.bytes,motion_bytes,15);
 x=a.bytes;x.resize(65537,0);add("oversize",x,b.bytes,motion_bytes,10);
 if(a.version==1){x=a.bytes;put(x,108,4,std::bit_cast<std::uint32_t>(std::numeric_limits<float>::quiet_NaN()),a.big);add("invalid-offset",x,b.bytes,motion_bytes,13);
  x=a.bytes;x.insert(x.begin()+static_cast<std::ptrdiff_t>(a.offset),16,0);put(x,108,4,std::bit_cast<std::uint32_t>(static_cast<float>(a.offset+16)),a.big);add("extended-offset",x,b.bytes,motion_bytes,0);}
 text.assign(motion_bytes.begin(),motion_bytes.end());p=text.find('\n');const auto end=text.find('\n',p+1);const auto tab=text.rfind('\t',end);need(tab>p,"arity mutation");text.erase(tab,end-tab);add("motion-arity",a.bytes,b.bytes,Bytes(text.begin(),text.end()),15);
 text.assign(motion_bytes.begin(),motion_bytes.end());text.replace(0,2,"zz");add("motion-header",a.bytes,b.bytes,Bytes(text.begin(),text.end()),15);
 text.assign(motion_bytes.begin(),motion_bytes.end());p=text.find('\n');const auto t=text.find('\t',p+1);text.replace(p+1,t-p-1,"0.1234567891");add("motion-precision",a.bytes,b.bytes,Bytes(text.begin(),text.end()),15);
 x=a.bytes;put(x,a.version==1?40:16,a.version==1?2:8,5,a.big);add("unsupported-dimensions",x,b.bytes,motion_bytes,13);
 if(a.version==1){x=a.bytes;put(x,112,4,std::bit_cast<std::uint32_t>(2.0f),a.big);add("unsupported-scaling-a",x,b.bytes,motion_bytes,13);}
 if(b.version==2){auto y=b.bytes;put(y,168,8,540,b.big);add("short-header-offset-b",a.bytes,y,motion_bytes,14);y=b.bytes;put(y,176,8,std::bit_cast<std::uint64_t>(2.0),b.big);add("unsupported-scaling-b",a.bytes,y,motion_bytes,14);}
 add("big-endian",minimal_big(a),minimal_big(b),motion_bytes,0);

 // Every legacy case above keeps its original inputs/exit contract and now
 // carries the explicit atlas/config. Additional cases exercise their meaning.
 auto add_full=[&](const std::string& name,const Bytes& aa,const Bytes& bb,const Bytes& mm,const Bytes& labels,const IndexConfig& cfg,int expected_exit){
  const auto d=out/name;need(fs::create_directory(d),"duplicate mutant name");write(d/"a.nii",aa);write(d/"b.nii",bb);write(d/"motion.tsv",mm);write(d/"atlas.nii",labels);write(d/"config.tsv",config_bytes(cfg));plan<<name<<'\t'<<expected_exit<<'\n';
 };
 auto labels=atlas.bytes;labels[atlas.offset]=labels[atlas.offset]==0?1:0;add_full("atlas-change",a.bytes,b.bytes,motion_bytes,labels,c,0);
 add_full("atlas-big-endian",a.bytes,b.bytes,motion_bytes,minimal_big(atlas),c,0);
 labels=atlas.bytes;put(labels,atlas.version==1?42:24,atlas.version==1?2:8,atlas.nx==4096?4095:atlas.nx+1,atlas.big);add_full("atlas-dimensions",a.bytes,b.bytes,motion_bytes,labels,c,16);
 auto cfg=c;labels=atlas.bytes;if(cfg.expected_regions==255)cfg.expected_regions=254;labels[atlas.offset]=static_cast<unsigned char>(cfg.expected_regions+1);add_full("atlas-label-out-of-range",a.bytes,b.bytes,motion_bytes,labels,cfg,16);
 labels=atlas.bytes;std::fill(labels.begin()+static_cast<std::ptrdiff_t>(atlas.offset),labels.begin()+static_cast<std::ptrdiff_t>(atlas.offset+atlas.nvox),0);add_full("atlas-background",a.bytes,b.bytes,motion_bytes,labels,c,0);
 labels=atlas.bytes;labels[atlas.version==1?344:4]='x';add_full("atlas-bad-magic",a.bytes,b.bytes,motion_bytes,labels,c,16);
 labels=atlas.bytes;labels.resize(static_cast<std::size_t>(atlas.offset+atlas.nvox-1));add_full("atlas-truncated",a.bytes,b.bytes,motion_bytes,labels,c,16);
 labels=atlas.bytes;put(labels,atlas.version==1?40:16,atlas.version==1?2:8,5,atlas.big);add_full("atlas-unsupported-dimensions",a.bytes,b.bytes,motion_bytes,labels,c,16);
 labels=atlas.bytes;put(labels,atlas.version==1?40:16,atlas.version==1?2:8,4,atlas.big);put(labels,48,atlas.version==1?2:8,2,atlas.big);add_full("atlas-timepoints",a.bytes,b.bytes,motion_bytes,labels,c,16);
 labels=atlas.bytes;
 if(atlas.version==1)put(labels,112,4,std::bit_cast<std::uint32_t>(2.0f),atlas.big);else put(labels,176,8,std::bit_cast<std::uint64_t>(2.0),atlas.big);
 add_full("atlas-unsupported-scaling",a.bytes,b.bytes,motion_bytes,labels,c,16);
 cfg=c;cfg.coverage_mode=2;add_full("atlas-region-coverage",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,0);
 cfg=c;cfg.weights[0]=3;add_full("config-weight-change",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,0);
 cfg=c;cfg.scales[0]=2;add_full("config-scale-change",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,0);
 cfg=c;cfg.weights.fill(1e300);add_full("config-max-normalized",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,0);
 cfg=c;cfg.weights[0]=-1;add_full("config-negative-weight",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.weights.fill(0);add_full("config-zero-weights",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.scales[0]=0;add_full("config-zero-scale",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.scales[0]=-1;add_full("config-negative-scale",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.weights[0]=std::bit_cast<double>(std::uint64_t{0x7ff8000000000000ULL});add_full("config-nan-weight",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.scales[0]=std::bit_cast<double>(std::uint64_t{0x7ff8000000000000ULL});add_full("config-nan-scale",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.weights[0]=std::numeric_limits<double>::infinity();add_full("config-infinite-weight",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.scales[0]=std::numeric_limits<double>::infinity();add_full("config-infinite-scale",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.formula=2;add_full("config-formula",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.alignment_policy=2;add_full("config-alignment",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.coverage_mode=3;add_full("config-coverage-mode",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.expected_regions=0;add_full("config-regions-zero",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 cfg=c;cfg.expected_regions=256;add_full("config-regions-too-many",a.bytes,b.bytes,motion_bytes,atlas.bytes,cfg,17);
 auto constant=a.bytes;std::fill(constant.begin()+static_cast<std::ptrdiff_t>(a.offset),constant.begin()+static_cast<std::ptrdiff_t>(a.offset+a.nvox*a.nt),1);
 cfg=c;cfg.weights[0]=1;cfg.weights[2]=0;cfg.weights[3]=0;add_full("config-ignore-unavailable",constant,b.bytes,motion_bytes,atlas.bytes,cfg,0);
 cfg=c;cfg.weights.fill(0);cfg.weights[3]=1;add_full("config-require-ci",constant,b.bytes,motion_bytes,atlas.bytes,cfg,0);
 cfg=c;cfg.weights.fill(0);cfg.weights[2]=1;add_full("config-require-tsnr",constant,b.bytes,motion_bytes,atlas.bytes,cfg,0);

 need(bool(plan),"plan write");
}
int main(int argc,char**argv){
 try{
  need(argc>=2,"mode");const std::string mode=argv[1];
  if(mode=="--wrong-output"){
   need(argc==4,"wrong-output arguments");std::ifstream in(argv[2]);std::ofstream out(argv[3]);std::string line;bool changed=false;
   while(std::getline(in,line)){if(line.rfind("FMRI_BITS qc_fd_mean ",0)==0){line="FMRI_BITS qc_fd_mean 0";changed=true;}out<<line<<'\n';}
   need(changed&&bool(out),"wrong-output mutation");return 0;
  }
  if(mode=="--sounio-config"){need(argc==3,"sounio-config arguments");emit_config(config(argv[2]));return 0;}
  if(mode=="--check"||mode=="--mutants")need(argc==8,"A B Motion Atlas Config Output required");
  else if(mode=="--validate")need(argc==7,"A B Motion Atlas Config required");
  else throw Input("unknown mode");
  const auto c=config(argv[6]);valid_config(c);
  const auto a=image(argv[2]),b=image(argv[3]),atlas=image(argv[5],true);
  const auto m=motion(argv[4]);const auto expected=calculate(a,b,m,atlas,c);
  if(mode=="--check"){check(expected,argv[7]);std::cout<<"FMRI_NATIVE_CPP_OK\n";}
  else if(mode=="--validate"){for(const auto&[name,value]:expected.values)std::cout<<std::setprecision(21)<<name<<" "<<value<<"\n";for(const auto&[name,reason]:expected.unavailable)std::cout<<"UNAVAILABLE "<<name<<" "<<reason<<"\n";}
  else{mutants(a,b,read(argv[4]),atlas,c,read(argv[6]),argv[7]);std::cout<<"FMRI_NATIVE_MUTANTS_WRITTEN\n";}
  return 0;
 }catch(const Input&e){std::cerr<<"INPUT_REFUSED "<<e.what()<<"\n";return 2;}
 catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
