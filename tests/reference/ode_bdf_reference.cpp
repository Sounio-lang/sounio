// C++23 binary128 independent analytic and RK4 control for actual native rows.
#include <quadmath.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
using Q=__float128;
using V=std::array<Q,3>;
V rhs(const V&y) { Q x=Q(0.04)*y[0],z=Q(10000)*y[1]*y[2],w=Q(30000000)*y[1]*y[1];return {-x+z,x-z-w,w}; }
V control(int count) {
 V y={1,0,0};Q h=strtoflt128("0.01",nullptr)/count;
 for(int i=0;i<count;++i) {
  V a=rhs(y),v;for(int k=0;k<3;++k)v[k]=y[k]+h*a[k]/2;
  V b=rhs(v);for(int k=0;k<3;++k)v[k]=y[k]+h*b[k]/2;
  V c=rhs(v);for(int k=0;k<3;++k)v[k]=y[k]+h*c[k];
  V d=rhs(v);for(int k=0;k<3;++k)y[k]+=h*(a[k]+2*b[k]+2*c[k]+d[k])/6;
 }
 return y;
}
double decode(std::int64_t b) {return std::bit_cast<double>(b);}
bool close(double actual,Q expected,Q tolerance) {return std::isfinite(actual)&&fabsq(Q(actual)-expected)<=tolerance;}
int main(int argc,char**argv) {
 if(argc!=2 && argc!=3) return 2;
 std::ifstream in(argv[argc-1]);
 if(!in) return 3;
 if(argc==3 && std::string(argv[1])=="--scale") {
  for(int i=0;i<128;++i) {
   std::string line,tag,extra;int index;std::int64_t bits;
   if(!std::getline(in,line))return 31;
   std::istringstream row(line);
   if(!(row>>tag>>index>>bits)||tag!="SCALE"||index!=i||row>>extra)return 32;
   Q initial=1+Q(i)/128;
   if(!close(decode(bits),initial*expq(-Q(0.01)),strtoflt128("1e-5",nullptr)*initial))return 33;
  }
  std::string line,tag,extra;std::int64_t a,attempts,calls,factors,bytes;
  if(!std::getline(in,line))return 34;
  std::istringstream row(line);
  if(!(row>>tag>>a>>attempts>>calls>>factors>>bytes)||tag!="SCALE_WORK"||row>>extra)return 35;
  if(a<=0||attempts<a||calls!=6*attempts||factors!=3*attempts||bytes!=8*(9*128+3*128*128))return 36;
  if(!std::getline(in,line)||line!="BDF_SCALE_PASS"||std::getline(in,line))return 37;
  std::cout<<"BDF_SCALE_REFERENCE_PASS n=128 payload_bytes="<<bytes<<'\n';return 0;
 }
 if(argc==3 && std::string(argv[1])=="--legacy") {
  std::string line,tag,extra;
  if(!std::getline(in,line)||!line.starts_with("LEGACY 1 1 1 1 "))return 41;
  if(!std::getline(in,line))return 42;
  std::istringstream row(line);int count;std::int64_t time,value;
  if(!(row>>tag>>count>>time>>value)||tag!="LEGACY_TRACE"||count!=64||row>>extra)return 43;
  if(decode(time)!=0.001||!close(decode(value),Q(0.001),strtoflt128("1e-12",nullptr)))return 44;
  if(!std::getline(in,line)||line!="GUM_COUPLED_REFUSED")return 45;
  if(!std::getline(in,line))return 47;
  {std::istringstream norm(line);std::string tag,extra;std::int64_t tiny,large;
   if(!(norm>>tag>>tiny>>large)||tag!="LEGACY_NORM"||norm>>extra)return 48;
   if(std::bit_cast<double>(tiny)!=1.0||std::bit_cast<double>(large)!=1.0e200)return 49;}
  if(!std::getline(in,line)||line!="BDF_LEGACY_PASS"||std::getline(in,line))return 46;
  std::cout<<"BDF_LEGACY_REFERENCE_PASS endpoint_trace_and_covariance_refusal\n";return 0;
 }
 if(argc!=2)return 2;
 const std::array<int,17> dims={1,1,1,1,1,2,1,1,1,1,2,3,1,1,1,1,1};
 const std::array<int,17> expected={0,0,0,2,6,4,4,3,7,8,0,0,1,1,4,1,0};
 V reference{},refined{};
 int rows=0;std::string line;bool marker=false;
 while(std::getline(in,line)) {
  if(line=="BDF_CHECKED_PASS") {if(marker)return 5;marker=true;continue;}
  std::istringstream s(line);std::string tag;int id,n,status;std::int64_t tb,xb,yb,zb;
  if(!(s>>tag>>id>>n>>status>>tb>>xb>>yb>>zb)||tag!="BDF"||id!=rows||id>=17||n!=dims[id]||status!=expected[id])return 6;
  std::string extra;if(s>>extra)return 7;
  double t=decode(tb),x=decode(xb),y=decode(yb),z=decode(zb);
  if(!std::isfinite(t)||(!std::isfinite(x)&&id!=14)||!std::isfinite(y)||!std::isfinite(z))return 8;
  if(!std::getline(in,line))return 9;
  std::istringstream w(line);int wid;std::int64_t a,r,attempts,nfe,nje,nfactor,nlinear,updates,cb,kind,linear,res,err;
  if(!(w>>tag>>wid>>a>>r>>attempts>>nfe>>nje>>nfactor>>nlinear>>updates>>cb>>kind>>linear>>res>>err)||tag!="WORK"||wid!=id)return 10;
  if(w>>extra)return 11;
  if(a<0||r<0||attempts!=a+r||updates<0||nlinear!=updates||nfactor<nlinear||nje<nfactor||nfe<nje)return 12;
  if(!std::isfinite(decode(res))||!std::isfinite(decode(err)))return 13;
  if(id==0) {if(t!=0.1||!close(x,expq(-Q(1)),strtoflt128("1e-4",nullptr))||nfe!=6*attempts||nje!=3*attempts||nfactor!=nje)return 14;}
  if(id==1) {if(t!=0||x!=1||attempts||nfe)return 15;}
  if(id==2) {if(t!=1e-16||x!=1||r||nfe!=3*attempts||nje)return 16;}
  if(id==3) {if(t!=1||x!=1||attempts||nfe)return 17;}
  if(id==4) {if(t||x!=1||nfe!=1||nje||cb!=17||kind!=1)return 18;}
  if(id==5) {if(t||x!=1||y||nfe!=1||nje)return 19;}
  if(id==6) {if(t||x!=1||nfe!=1||nje!=1||nfactor)return 20;}
  if(id==7) {if(t||x!=1||attempts!=1||a||r!=1)return 21;}
  if(id==8) {if(t||x!=1||nfe!=2||nje!=1||nlinear!=1||a)return 22;}
  if(id==9) {if(t||x!=1||nfe!=1||nje!=1||nfactor!=1||nlinear||linear==0)return 23;}
  if(id==10) {Q t0=Q(0.01);if(t!=0.01||!close(x,expq(-t0)+expq(-1000*t0),strtoflt128("2e-4",nullptr))||!close(y,expq(-t0),strtoflt128("1e-5",nullptr)))return 24;}
  if(id==11) {
   reference=control(100000);refined=control(200000);
   for(int k=0;k<3;++k) if(fabsq(reference[k]-refined[k])>strtoflt128("1e-18",nullptr)) return 4;
   if(t!=0.01||!close(x,refined[0],strtoflt128("2e-7",nullptr))||!close(y,refined[1],strtoflt128("2e-7",nullptr))||!close(z,refined[2],strtoflt128("2e-7",nullptr))||x<0||y<0||z<0||std::abs(x+y+z-1)>1e-12)return 25;}
  if(id==12) {if(t||x!=1||attempts||nfe)return 26;}
  if(id==13 || id==15) {if(t||x!=1||attempts||nfe)return 28;}
  if(id==14) {if(t||!std::isnan(x)||attempts||nfe)return 29;}
  if(id==16) {
   Q ulp=Q(std::numeric_limits<double>::epsilon()),lambda=100000000000000LL;
   Q coarse=1/(1+3*ulp*lambda),fine=1/((1+2*ulp*lambda)*(1+ulp*lambda));
   Q error=fabsq(fine-coarse)*Q(1.25)/(Q(0.001)+Q(0.001));
   if(t!=std::bit_cast<double>(std::int64_t(4607182418800017411LL))||!close(x,fine,strtoflt128("2e-15",nullptr))||!close(decode(err),error,strtoflt128("2e-10",nullptr))||a!=1||attempts!=1)return 30;
  }
  ++rows;
 }
 if(rows!=17||!marker)return 27;
 std::cout<<"BDF_REFERENCE_PASS rows="<<rows<<" control=RK4-binary128-100000-200000\n";
}
