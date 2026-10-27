#include <algorithm>
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
static Image image(const fs::path&p){
 Image a;a.bytes=read(p);auto h=word(a.bytes,0,4,false);
 if(h!=348&&h!=540){a.big=true;h=word(a.bytes,0,4,true);}
 need(h==348||h==540,"header size");a.version=h==348?1:2;
 need(word(a.bytes,h==348?70:12,2,a.big)==2&&word(a.bytes,h==348?72:14,2,a.big)==8,"datatype");
 const std::size_t magic=h==348?344:4;
 need(a.bytes.size()>magic+3&&a.bytes[magic]=='n'&&a.bytes[magic+1]=='+'&&a.bytes[magic+2]==(h==348?'1':'2')&&a.bytes[magic+3]==0,"magic");
 need(word(a.bytes,h==348?40:16,h==348?2:8,a.big)==4,"four-dimensional input required");
 const auto base=h==348?42:24;const int width=h==348?2:8;
 a.nx=word(a.bytes,base,width,a.big);a.ny=word(a.bytes,base+width,width,a.big);a.nz=word(a.bytes,base+2*width,width,a.big);a.nt=word(a.bytes,base+3*width,width,a.big);
 need(a.nx>0&&a.ny>0&&a.nz>0&&a.nx<=4096&&a.ny<=4096&&a.nz<=4096&&a.nt>=2&&a.nt<=500,"dimensions");
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
static long double median(std::vector<long double> a){need(!a.empty(),"empty median");std::sort(a.begin(),a.end());const auto n=a.size();return n%2?a[n/2]:(a[n/2-1]+a[n/2])/2;}
struct Expected {std::map<std::string,long double> values;std::map<std::string,std::string> unavailable;};
static Expected calculate(const Image&a,const Image&b,const std::vector<std::vector<long double>>&m){
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
 e.unavailable["atlas_coverage_ratio"]="atlas_input_required";e.unavailable["uncertainty_score"]="index_definition_required";return e;
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
static void mutants(const Image&a,const Image&b,const Bytes&motion_bytes,const fs::path&out){
 need(!fs::exists(out),"existing mutants");fs::create_directories(out);std::ofstream plan(out/"PLAN.tsv");
 auto add=[&](std::string name,Bytes x,Bytes y,Bytes m,int exit){
  const auto d=out/name;fs::create_directory(d);write(d/"a.nii",x);write(d/"b.nii",y);write(d/"motion.tsv",m);plan<<name<<'\t'<<exit<<'\n';
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
  need(argc>=5,"input arguments");const auto a=image(argv[2]),b=image(argv[3]);const auto m=motion(argv[4]);const auto expected=calculate(a,b,m);
  if(mode=="--check"){need(argc==6,"check arguments");check(expected,argv[5]);std::cout<<"FMRI_NATIVE_CPP_OK\n";}
  else if(mode=="--validate"){need(argc==5,"validate arguments");for(const auto&[name,value]:expected.values){std::cout<<std::setprecision(21)<<name<<" "<<value<<"\n";}}
  else if(mode=="--mutants"){need(argc==6,"mutant arguments");mutants(a,b,read(argv[4]),argv[5]);std::cout<<"FMRI_NATIVE_MUTANTS_WRITTEN\n";}
  else{throw Input("unknown mode");}
  return 0;
 }catch(const Input&e){std::cerr<<"INPUT_REFUSED "<<e.what()<<"\n";return 2;}
 catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
