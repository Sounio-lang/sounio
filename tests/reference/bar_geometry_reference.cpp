// Independent C++23/quad bar geometry reference and compact fixture emitter.
// g++ -std=c++23 -O2 -Wall -Wextra -pedantic bar_geometry_reference.cpp -lquadmath -o bar_geometry_reference
// Binary64 calculations only classify the specified representability guards.
// Expected offsets use quad differences/ratios and truncq; selected coordinates
// are dyadic or safely away from truncation boundaries. No production include.
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
#include <algorithm>
using I=std::int64_t;using Q=__float128;
struct RectCase {
 std::string name;double sx=2,sy=4,xlo=0,xhi=4,ylo=-8,yhi=8;
 I x=20,y=30,w=40,h=32,width=8;bool clipped=false;
};
struct Rect {I x=0,y=0,w=0,h=0,status=2;};
struct WidthCase {std::string name;double x0=0,x1=1;I count=2;double lo=0,hi=4;I x=20,w=40;};
struct Width {I value=0,status=2;};
bool geometry_x(I x,I w) {
 return x>=-1000000&&x<=1000000&&w>=1&&w<=1000000&&x+w<=1000000;
}
bool finite_domain(double lo,double hi) {
 if(!std::isfinite(lo)||!std::isfinite(hi)||lo>hi)return false;
 const double difference=hi-lo;
 return std::isfinite(difference)&&(lo==hi||difference>0);
}
bool offset(double value,double lo,double hi,I extent,I&out) {
 if(!std::isfinite(value)||!finite_domain(lo,hi))return false;
 if(lo==hi){if(value!=lo)return false;out=extent/2;return true;}
 const double ratio=(value-lo)/(hi-lo);
 if(!std::isfinite(ratio)||std::abs(ratio)>1000000)return false;
 out=static_cast<I>(truncq(((Q)value-(Q)lo)/((Q)hi-(Q)lo)*(Q)extent));
 return true;
}
Rect rectangle(const RectCase&c) {
 Rect out;
 if(!geometry_x(c.x,c.w)||!geometry_x(c.y,c.h)||c.width<2||c.width>40)return out;
 if(!std::isfinite(c.sx)||!std::isfinite(c.sy)||
    !finite_domain(c.xlo,c.xhi)||!finite_domain(c.ylo,c.yhi))return out;
 if(c.xlo==c.xhi&&c.sx!=c.xlo)return out;
 if(c.ylo==c.yhi){
  if(c.ylo!=0||c.sy!=0)return out;
  out.status=1;return out;
 }
 double tip=c.sy,base=0;
 if(c.clipped){tip=std::clamp(tip,c.ylo,c.yhi);base=std::clamp(base,c.ylo,c.yhi);}
 I ox=0,tip_offset=0,base_offset=0;
 if(!offset(c.sx,c.xlo,c.xhi,c.w,ox)||
    !offset(tip,c.ylo,c.yhi,c.h,tip_offset)||
    !offset(base,c.ylo,c.yhi,c.h,base_offset))return out;
 const I center=c.x+ox,ty=c.y+c.h-tip_offset,by=c.y+c.h-base_offset;
 I left=center-c.width/2,right=left+c.width;
 I top=std::min(ty,by),bottom=std::max(ty,by);
 if(c.clipped){
  // Rectangle intersection, not center culling. Viewport's closed endpoint
  // cells correspond to this explicit half-open integer interval.
  left=std::max(left,c.x);right=std::min(right,c.x+c.w+1);
  top=std::max(top,c.y);bottom=std::min(bottom,c.y+c.h+1);
 }
 if(right<=left||bottom<=top){out.status=1;return out;}
 out.x=left;out.y=top;out.w=right-left;out.h=bottom-top;out.status=0;return out;
}
Width pair_width(const WidthCase&c) {
 Width out;
 if(c.count<0||c.count>2||!geometry_x(c.x,c.w)||!finite_domain(c.lo,c.hi))return out;
 I first=0,second=0;
 if(c.count>=1&&!offset(c.x0,c.lo,c.hi,c.w,first))return out;
 if(c.count==2&&!offset(c.x1,c.lo,c.hi,c.w,second))return out;
 out.value=8;
 if(c.count==2){const I gap=first>second?first-second:second-first;out.value=std::clamp(gap*8/10,I(2),I(40));}
 out.status=0;return out;
}
std::vector<RectCase> rectangles() {
 std::vector<RectCase> out;auto add=[&](RectCase c,std::string name){c.name=std::move(name);out.push_back(c);};
 RectCase a;add(a,"positive_actual_x");a.sx=0;add(a,"left_actual_x");a.sx=4;add(a,"right_actual_x");
 a={};a.sy=-4;add(a,"negative_from_zero");a.sy=0;add(a,"zero_height");
 a={};a.width=5;add(a,"odd_bar_width");a.width=2;add(a,"minimum_bar_width");a.width=40;add(a,"maximum_bar_width");
 a={};a.ylo=4;a.yhi=12;a.sy=8;add(a,"positive_window_legacy_zero_baseline");
 a.clipped=true;add(a,"positive_window_clipped_baseline_outside");
 a.sy=100;add(a,"tip_above_positive_window");a.sy=1;add(a,"both_below_positive_window_empty");
 a={};a.ylo=-12;a.yhi=-4;a.sy=-8;a.clipped=true;add(a,"negative_window_baseline_outside");
 a.sy=-100;add(a,"tip_below_negative_window");a.sy=-1;add(a,"both_above_negative_window_empty");
 a={};a.clipped=true;a.sx=-.125;add(a,"center_outside_left_rect_overlaps");
 a.sx=4.125;add(a,"center_outside_right_rect_overlaps");
 a.sx=-1;add(a,"horizontal_no_overlap_left");a.sx=5;add(a,"horizontal_no_overlap_right");
 a.sx=0;add(a,"left_endpoint_partial_bar");a.sx=4;add(a,"right_endpoint_partial_bar");
 a={};a.xlo=3;a.xhi=3;a.sx=3;a.w=7;add(a,"constant_x_odd_center");
 a.sx=1e300;a.xlo=1e300;a.xhi=1e300;add(a,"constant_x_huge");
 a.sx=-1e300;a.xlo=-1e300;a.xhi=-1e300;add(a,"constant_x_negative_huge");
 a.sx=1;add(a,"constant_x_mismatch");
 a={};a.ylo=0;a.yhi=0;a.sy=0;add(a,"constant_zero_y_empty");a.clipped=true;add(a,"constant_zero_y_empty_clipped");
 a.sy=1;add(a,"constant_zero_y_nonmatching");
 a.ylo=2;a.yhi=2;a.sy=2;add(a,"nonzero_constant_y_invalid");
 a={};a.ylo=0;a.yhi=8;a.h=4;a.sy=.5;add(a,"nonzero_value_subpixel_empty");
 a={};a.ylo=-1e300;a.yhi=1e300;a.sy=5e299;add(a,"large_y_units_dyadic");
 a.ylo=-1e-300;a.yhi=1e-300;a.sy=5e-301;add(a,"tiny_y_units_dyadic");
 a={};a.x=-1000000;a.y=-1000000;a.w=1000000;a.h=1000000;add(a,"maximum_geometry");
 a={};a.width=1;add(a,"invalid_width_one");a.width=0;add(a,"invalid_width_zero");a.width=41;add(a,"invalid_width_41");
 a={};a.w=0;add(a,"invalid_extent");a={};a.x=1000000;add(a,"invalid_endpoint");
 a={};a.x=std::numeric_limits<I>::max();add(a,"i64_origin_max");a={};a.y=std::numeric_limits<I>::min();add(a,"i64_origin_min");
 a={};a.xlo=4;a.xhi=0;add(a,"reversed_x");a={};a.ylo=8;a.yhi=-8;add(a,"reversed_y");
 const double max=std::numeric_limits<double>::max();
 a={};a.xlo=-max;a.xhi=max;add(a,"overflow_x_span");
 a={};a.ylo=-max;a.yhi=max;a.clipped=true;add(a,"overflow_y_span_before_clamp");
 a={};a.xhi=1;a.sx=1000000;add(a,"normalized_x_endpoint");
 a.sx=std::nextafter(1000000.,std::numeric_limits<double>::infinity());add(a,"normalized_x_exceeded");
 a={};a.ylo=1e300;a.yhi=std::nextafter(1e300,max);a.sy=a.yhi;add(a,"unclipped_baseline_normalized_overflow");
 a.clipped=true;add(a,"clipped_baseline_extreme_units");
 a={};a.sy=max;a.clipped=true;add(a,"finite_huge_tip_clamped_before_projection");
 const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
 for(int slot=0;slot<6;++slot){
  a={};double* vars[]={&a.sx,&a.sy,&a.xlo,&a.xhi,&a.ylo,&a.yhi};
  *vars[slot]=nan;add(a,"nan_"+std::to_string(slot));
  a={};double* vars2[]={&a.sx,&a.sy,&a.xlo,&a.xhi,&a.ylo,&a.yhi};
  *vars2[slot]=inf;add(a,"infinity_"+std::to_string(slot));
 }
 return out;
}
std::vector<WidthCase> widths(){
 std::vector<WidthCase> out;auto add=[&](WidthCase c,std::string name){c.name=std::move(name);out.push_back(c);};
 WidthCase a;add(a,"actual_first_pair");a.x0=4;a.x1=0;add(a,"input_order_changes_width");
 a.x0=1;a.x1=0;add(a,"reversed_first_pair_same_gap");
 a={};a.x1=0;add(a,"duplicate_first_pair_min2");
 a.x1=.125;add(a,"truncated_small_gap_min2");
 a.x1=4;a.w=100;add(a,"large_gap_max40");
 a={};a.count=1;a.x1=std::numeric_limits<double>::quiet_NaN();add(a,"singleton8_inactive_x1");
 a={};a.count=0;a.x0=std::numeric_limits<double>::quiet_NaN();a.x1=std::numeric_limits<double>::infinity();add(a,"empty8_inactive_both");
 a={};a.count=1;a.lo=1e300;a.hi=1e300;a.x0=1e300;add(a,"singleton_constant_huge");
 a.x0=0;add(a,"singleton_constant_mismatch");
 a={};a.lo=3;a.hi=3;a.x0=3;a.x1=3;add(a,"constant_pair_width2");
 a.count=0;a.x0=0;a.x1=1;add(a,"empty_constant_no_active_values");
 a={};a.count=-1;add(a,"negative_count");a.count=3;add(a,"count_exceeds_two");
 a={};a.x0=std::numeric_limits<double>::quiet_NaN();add(a,"active_x0_nan");
 a={};a.x1=std::numeric_limits<double>::infinity();add(a,"active_x1_inf");
 a={};a.count=0;a.w=0;add(a,"inactive_does_not_skip_geometry");
 a={};a.count=0;a.lo=4;a.hi=0;add(a,"inactive_does_not_skip_domain");
 a={};a.count=0;a.lo=-std::numeric_limits<double>::max();a.hi=std::numeric_limits<double>::max();add(a,"inactive_does_not_skip_span");
 a={};a.x0=1000000;a.hi=1;add(a,"normalized_endpoint_gap_clamp");
 a.x0=1000001;add(a,"normalized_exceeded");
 a={};a.x=-1000000;a.w=1000000;add(a,"maximum_width_geometry");
 return out;
}
std::string literal(double value){
 if(std::isnan(value))return "(0.0/0.0)";
 if(std::isinf(value))return value<0?"(-1.0e308*10.0)":"(1.0e308*10.0)";
 std::ostringstream s;s<<std::scientific<<std::setprecision(17);
 if(std::fpclassify(value)==FP_SUBNORMAL)s<<"("<<std::ldexp(value,52)<<" / 4503599627370496.0)";
 else s<<value;
 return s.str();
}
std::string integer(I value){return value==std::numeric_limits<I>::min()?"(0-9223372036854775807-1)":std::to_string(value);}
int main(int argc,char**argv){
 const bool fixture=argc==2&&std::string(argv[1])=="--fixture";if(argc>1&&!fixture)return 2;
 if(fixture)std::cout<<R"SIO(//@ run-pass
//@ expect-stdout: VIZ_BAR_GEOMETRY_REFERENCE_PASS
// Independent C++23 quad rectangles; actual-x and zero-baseline semantics.
use viz::bar::{bar_project,bar_pair_width}
fn check_rect(sx:f64,sy:f64,xlo:f64,xhi:f64,ylo:f64,yhi:f64,x:i64,y:i64,w:i64,h:i64,width:i64,clipped:bool,status:i64,rx:i64,ry:i64,rw:i64,rh:i64) with Div,Panic {
 let r=bar_project(sx,sy,xlo,xhi,ylo,yhi,x,y,w,h,width,clipped)
 assert(r.status==status && r.x==rx && r.y==ry && r.w==rw && r.h==rh)
}
fn check_width(x0:f64,x1:f64,count:i64,lo:f64,hi:f64,x:i64,w:i64,status:i64,value:i64) with Div,Panic {
 let r=bar_pair_width(x0,x1,count,lo,hi,x,w)
 assert(r.status==status && r.value==value)
}
fn main()->i32 with IO,Div,Panic {
)SIO";
 for(const auto&c:rectangles()){
  const auto r=rectangle(c);
  if(!fixture){std::cout<<"rect\t"<<c.name<<"\t"<<r.status<<"\t"<<r.x<<"\t"<<r.y<<"\t"<<r.w<<"\t"<<r.h<<"\n";continue;}
  std::cout<<" // "<<c.name<<"\n check_rect(";
  bool first=true;for(double v:{c.sx,c.sy,c.xlo,c.xhi,c.ylo,c.yhi}){std::cout<<(first?"":",")<<literal(v);first=false;}
  for(I v:{c.x,c.y,c.w,c.h,c.width})std::cout<<","<<integer(v);
  std::cout<<","<<(c.clipped?"true":"false")<<","<<r.status<<","<<r.x<<","<<r.y<<","<<r.w<<","<<r.h<<")\n";
 }
 for(const auto&c:widths()){
  const auto r=pair_width(c);
  if(!fixture){std::cout<<"width\t"<<c.name<<"\t"<<r.status<<"\t"<<r.value<<"\n";continue;}
  std::cout<<" // "<<c.name<<"\n check_width("<<literal(c.x0)<<","<<literal(c.x1)<<","<<c.count<<","<<literal(c.lo)<<","<<literal(c.hi)<<","<<integer(c.x)<<","<<integer(c.w)<<","<<r.status<<","<<r.value<<")\n";
 }
 if(fixture)std::cout<<" println(\"VIZ_BAR_GEOMETRY_REFERENCE_PASS\")\n 0\n}\n";
}

