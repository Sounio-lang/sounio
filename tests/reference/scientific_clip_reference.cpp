#include <array>
#include <iostream>
#include <iomanip>
#include <cmath>
struct Point { long double x,y; };
int code(Point p){return (p.x<0?1:0)|(p.x>1?2:0)|(p.y<0?4:0)|(p.y>1?8:0);}
bool clip(Point& a,Point& b){
 for(int i=0;i<16;++i){
  int ca=code(a),cb=code(b); if(!(ca|cb))return true;if(ca&cb)return false;
  int c=ca?ca:cb;Point p{};
  if(c&8){p.y=1;p.x=a.x+(b.x-a.x)*(1-a.y)/(b.y-a.y);}
  else if(c&4){p.y=0;p.x=a.x+(b.x-a.x)*(0-a.y)/(b.y-a.y);}
  else if(c&2){p.x=1;p.y=a.y+(b.y-a.y)*(1-a.x)/(b.x-a.x);}
  else {p.x=0;p.y=a.y+(b.y-a.y)*(0-a.x)/(b.x-a.x);}
  if(ca)a=p;else b=p;
 }
 std::abort();
}
int main(){
 const std::array<std::array<double,4>,20> cases={{
 {0.1,0.2,0.9,0.8},{-1,0.5,2,0.5},{2,0.5,-1,0.5},
 {0.5,-2,0.5,3},{0.5,3,0.5,-2},{-2,0.1,-1,0.8},
 {0.1,2,0.8,3},{-1,0,2,0},{0,2,0,-1},{0.5,0.5,0.5,0.5},
 {-1,-1,-1,-1},{-1,-1,0,0},{-1,1,1,-1},{-1,2,2,-1},
 {-1,1.01,0.01,2},{-0.25,0.75,1.25,0.25},{1,1,1,1},
 {0,0,1,1},{1,0,0,1},{-1000000,0.5,1000000,0.5}}};
 std::cout<<std::scientific<<std::setprecision(17);
 std::cout<<"fn reference_clip() with IO, Mut, Div, Panic {\n";
 int index=0;
 for(auto c:cases){
  Point a{c[0],c[1]},b{c[2],c[3]};bool visible=clip(a,b);
  std::cout<<" { println(\"CLIP_CASE_"<<index++<<"\")\n let result=clip_unit_segment("<<c[0]<<","<<c[1]<<","<<c[2]<<","<<c[3]<<")\n assert(result.status == "<<(visible?0:1)<<")\n";
  if(visible)std::cout<<" assert(close(result.x0,"<<(double)a.x<<"))\n assert(close(result.y0,"<<(double)a.y<<"))\n assert(close(result.x1,"<<(double)b.x<<"))\n assert(close(result.y1,"<<(double)b.y<<"))\n";
  if(visible){
   const std::array<const char*,4> names={"x0","y0","x1","y1"};
   const std::array<long double,4> coordinates={a.x,a.y,b.x,b.y};
   for(int j=0;j<4;++j)if(coordinates[j]==0 || coordinates[j]==1)
    std::cout<<" assert(result."<<names[j]<<" == "<<(coordinates[j]==0?"0.0":"1.0")<<")\n";
  }
  std::cout<<" }\n";
 }
 std::cout<<"}\n";
 // Integer rational y-major AA reference (40,30)->(80,110).
 long long coverage_y=40LL*255/80;
 std::cout<<"fn reference_aa_y(canvas: &Canvas) with Panic {\n";
 std::cout<<" assert(read_i64(canvas.pixels,31*320+40) == "<<((255-coverage_y)<<16)<<")\n";
 std::cout<<" assert(read_i64(canvas.pixels,31*320+41) == "<<(coverage_y<<16)<<")\n}\n";
 // Negative-slope rational AA reference (40,110)->(240,30).
 long long cov_negative=120LL*255/200;
 std::cout<<"fn reference_aa_negative(canvas: &Canvas) with Panic {\n";
 std::cout<<" assert(read_i64(canvas.pixels,109*320+41) == "<<((255-cov_negative)<<16)<<")\n";
 std::cout<<" assert(read_i64(canvas.pixels,110*320+41) == "<<(cov_negative<<16)<<")\n}\n";
 // Integer rational reference for a red AA segment (40,30)->(240,110).
 long long span=200,remainder=80,coverage=remainder*255/span;
 std::cout<<"fn reference_aa(canvas: &Canvas) with Panic {\n";
 std::cout<<" assert(read_i64(canvas.pixels,30*320+41) == "<<((255-coverage)<<16)<<")\n";
 std::cout<<" assert(read_i64(canvas.pixels,31*320+41) == "<<(coverage<<16)<<")\n}\n";
}
