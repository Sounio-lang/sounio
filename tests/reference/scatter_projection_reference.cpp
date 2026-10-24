// Independent C++23 quad scatter projection reference + Sounio fixture emitter.
// Compile on t560: g++ -std=c++23 -O2 -Wall -Wextra -pedantic scatter_projection_reference.cpp -lquadmath -o scatter_projection_reference
// Binary64 subtraction/division are used ONLY to decide the documented
// representability guards. Coordinate expectations use independent quad
// arithmetic and truncq. Test ratios stay away from rounding bin boundaries,
// except exactly representable endpoints/halves/guard endpoints.
// No production projection code is included.
#include <quadmath.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
using I=std::int64_t;using Q=__float128;
struct Case {
 std::string name;
 double sx=1,sy=1,xlo=0,xhi=4,ylo=0,yhi=4;
 I x=11,y=23,w=7,h=9;bool clipped=false;
};
struct Point {I x=0,y=0,status=2;};
Point evaluate(const Case&c) {
 Point out;
 if(c.x < -1000000||c.x>1000000||c.y < -1000000||c.y>1000000||
    c.w<1||c.w>1000000||c.h<1||c.h>1000000)return out;
 if(c.x+c.w>1000000||c.y+c.h>1000000)return out;
 for(double v:{c.sx,c.sy,c.xlo,c.xhi,c.ylo,c.yhi})
  if(!std::isfinite(v))return out;
 if(c.xlo>c.xhi||c.ylo>c.yhi)return out;
 const bool cx=c.xlo==c.xhi,cy=c.ylo==c.yhi;
 const double dx=c.xhi-c.xlo,dy=c.yhi-c.ylo;
 if((!cx&&(!std::isfinite(dx)||dx<=0))||
    (!cy&&(!std::isfinite(dy)||dy<=0)))return out;
 if((cx&&c.sx!=c.xlo)||(cy&&c.sy!=c.ylo))return out;
 if(c.clipped&&(c.sx<c.xlo||c.sx>c.xhi||c.sy<c.ylo||c.sy>c.yhi)){
  out.status=1;return out;
 }
 // Guard availability follows the public binary64 numerical boundary; no
 // unstable native float result is accepted merely because quad can represent it.
 if(!cx){double t=(c.sx-c.xlo)/dx;if(!std::isfinite(t)||std::abs(t)>1000000)return out;}
 if(!cy){double t=(c.sy-c.ylo)/dy;if(!std::isfinite(t)||std::abs(t)>1000000)return out;}
 I ox=c.w/2,oy=c.h/2;
 if(!cx)ox=static_cast<I>(truncq(((Q)c.sx-(Q)c.xlo)/((Q)c.xhi-(Q)c.xlo)*(Q)c.w));
 if(!cy)oy=static_cast<I>(truncq(((Q)c.sy-(Q)c.ylo)/((Q)c.yhi-(Q)c.ylo)*(Q)c.h));
 out.x=c.x+ox;out.y=c.y+c.h-oy;out.status=0;
 return out;
}
std::vector<Case> cases(){
 std::vector<Case> all;
 auto add=[&](Case c,std::string name){c.name=std::move(name);all.push_back(c);};
 Case a;
 add(a,"dyadic_noninteger_offsets");
 a.sx=0;a.sy=0;add(a,"lower_closed_endpoint");
 a.sx=4;a.sy=4;add(a,"upper_closed_endpoint");
 a.sx=2;a.sy=2;add(a,"half_odd_extent");
 a.clipped=true;add(a,"half_clipped");
 a={};a.xlo=0;a.xhi=0;a.ylo=0;a.yhi=0;a.sx=0;a.sy=0;
 add(a,"constant_zero_odd");a.clipped=true;add(a,"constant_zero_clipped");
 a.sx=1e300;a.xlo=1e300;a.xhi=1e300;
 a.sy=-1e300;a.ylo=-1e300;a.yhi=-1e300;
 add(a,"constant_large_signed");
 a.sx=1e-300;a.xlo=1e-300;a.xhi=1e-300;
 a.sy=-1e-300;a.ylo=-1e-300;a.yhi=-1e-300;
 add(a,"constant_tiny_signed");
 a={};a.sx=3;a.xlo=3;a.xhi=3;add(a,"constant_x_only");
 a={};a.sy=-3;a.ylo=-3;a.yhi=-3;add(a,"constant_y_only");
 a={};a.sx=1;a.xlo=0;a.xhi=0;add(a,"constant_mismatch_unclipped");
 a.clipped=true;add(a,"constant_mismatch_before_cull");
 a={};a.sy=1;a.ylo=0;a.yhi=0;a.clipped=true;add(a,"constant_y_mismatch");
 for(bool clip:{false,true}){
  a={};a.clipped=clip;a.sx=-1;add(a,clip?"outside_left_culled":"outside_left_unclipped_trunc");
  a.sx=5;add(a,clip?"outside_right_culled":"outside_right_unclipped");
  a={};a.clipped=clip;a.sy=-1;add(a,clip?"outside_bottom_culled":"outside_bottom_unclipped");
  a.sy=5;add(a,clip?"outside_top_culled":"outside_top_unclipped");
 }
 a={};a.xlo=-1e300;a.xhi=1e300;a.ylo=-1e-300;a.yhi=1e-300;a.sx=0;a.sy=0;
 add(a,"opposite_unit_scales_half");
 a={};a.x=-1000000;a.y=0;a.w=1000000;a.h=1000000;a.sx=2;a.sy=2;
 add(a,"maximum_valid_geometry");
 a={};a.x=999999;a.y=999999;a.w=1;a.h=1;add(a,"closed_geometry_maximum");
 a={};a.x=-1000000;a.y=-1000000;a.w=1;a.h=1;a.sx=0;a.sy=0;a.xhi=0;a.yhi=0;
 add(a,"constant_minimum_geometry");
 a={};a.w=0;add(a,"zero_width");a.w=-1;add(a,"negative_width");a.w=1000001;add(a,"oversized_width");
 a={};a.h=0;add(a,"zero_height");a.h=1000001;add(a,"oversized_height");
 a={};a.x=1000000;a.w=1;add(a,"x_endpoint_outside");
 a={};a.y=1000000;a.h=1;add(a,"y_endpoint_outside");
 a={};a.x=-1000001;add(a,"x_origin_outside");
 a={};a.y=-1000001;add(a,"y_origin_outside");
 a={};a.x=std::numeric_limits<I>::max();add(a,"i64_max_origin");
 a={};a.y=std::numeric_limits<I>::min();add(a,"i64_min_origin");
 a={};a.xlo=4;a.xhi=0;add(a,"reversed_x");
 a={};a.ylo=4;a.yhi=0;add(a,"reversed_y");
 const double big=std::numeric_limits<double>::max();
 a={};a.xlo=-big;a.xhi=big;a.sx=0;add(a,"overflow_x_span");
 a.clipped=true;a.sx=big;add(a,"overflow_span_before_cull");
 a={};a.ylo=-big;a.yhi=big;a.sy=0;add(a,"overflow_y_span");
 a={};a.xlo=-big;a.xhi=-big*.5;a.sx=big;
 add(a,"overflow_numerator_unclipped");a.clipped=true;add(a,"outside_before_numerator");
 const double nan=std::numeric_limits<double>::quiet_NaN();
 const double inf=std::numeric_limits<double>::infinity();
 for(int slot=0;slot<6;++slot){
  a={};double* values[]={&a.sx,&a.sy,&a.xlo,&a.xhi,&a.ylo,&a.yhi};
  *values[slot]=nan;add(a,"nan_"+std::to_string(slot));
  a={};double* infinite[]={&a.sx,&a.sy,&a.xlo,&a.xhi,&a.ylo,&a.yhi};
  *infinite[slot]=inf;add(a,"infinity_"+std::to_string(slot));
 }
 a={};a.xlo=0;a.xhi=1;a.sx=1000000;add(a,"positive_normalized_guard_endpoint");
 a.sx=-1000000;add(a,"negative_normalized_guard_endpoint");
 a.sx=std::nextafter(1000000.,inf);add(a,"positive_normalized_guard_exceeded");
 a.sx=std::nextafter(-1000000.,-inf);add(a,"negative_normalized_guard_exceeded");
 a.clipped=true;add(a,"clipped_before_normalized_guard");
 const double tiny=std::numeric_limits<double>::denorm_min();
 a={};a.xlo=tiny;a.xhi=2*tiny;a.sx=tiny;add(a,"subnormal_lower_endpoint");
 a.sx=2*tiny;add(a,"subnormal_upper_endpoint");
 a.sx=1;add(a,"subnormal_span_huge_normalized");
 return all;
}
std::string literal(double value){
 if(std::isnan(value))return "(0.0/0.0)";
 if(std::isinf(value))return value<0?"(-1.0e308*10.0)":"(1.0e308*10.0)";
 std::ostringstream out;out<<std::scientific<<std::setprecision(17);
 if(std::fpclassify(value)==FP_SUBNORMAL)out<<"("<<std::ldexp(value,52)<<" / 4503599627370496.0)";
 else out<<value;
 return out.str();
}
std::string integer(I value){
 if(value==std::numeric_limits<I>::min())return "(0-9223372036854775807-1)";
 return std::to_string(value);
}
int main(int argc,char**argv){
 const bool fixture=argc==2&&std::string(argv[1])=="--fixture";
 if(argc>1&&!fixture)return 2;
 const auto all=cases();
 if(fixture)std::cout<<"//@ run-pass\n//@ expect-stdout: VIZ_SCATTER_PROJECTION_REFERENCE_PASS\n// Independent C++23 quad coordinates with binary64 availability guards.\nuse viz::scatter::{scatter_project}\nfn check(sx:f64,sy:f64,xlo:f64,xhi:f64,ylo:f64,yhi:f64,x:i64,y:i64,w:i64,h:i64,clipped:bool,status:i64,px:i64,py:i64) with Div,Panic {\n let p=scatter_project(sx,sy,xlo,xhi,ylo,yhi,x,y,w,h,clipped)\n assert(p.status==status && p.x==px && p.y==py)\n}\nfn main()->i32 with IO,Div,Panic {\n";
 for(std::size_t i=0;i<all.size();++i){
  const auto&c=all[i];auto point=evaluate(c);
  if(!fixture){std::cout<<i<<"\t"<<c.name<<"\t"<<point.status<<"\t"<<point.x<<"\t"<<point.y<<"\n";continue;}
  std::cout<<" // "<<c.name<<"\n check(";
  bool first=true;for(double v:{c.sx,c.sy,c.xlo,c.xhi,c.ylo,c.yhi}){std::cout<<(first?"":",")<<literal(v);first=false;}
  for(I v:{c.x,c.y,c.w,c.h})std::cout<<","<<integer(v);
  std::cout<<","<<(c.clipped?"true":"false")<<","<<point.status<<","<<point.x<<","<<point.y<<")\n";
 }
 if(fixture)std::cout<<" println(\"VIZ_SCATTER_PROJECTION_REFERENCE_PASS\")\n 0\n}\n";
}
