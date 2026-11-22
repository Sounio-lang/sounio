// Independent C++23 QR and explicit case-deletion oracle. No numerical Python.
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using A=std::array<long double,100>;
struct Fit { long double xa,ya,xs,ys,b0,b1,sigma,r00,r01,r11; A h{},e{}; };
Fit qr(const A& x,const A& y,int n,int omit=-1) {
    Fit f{}; int a=omit==0?1:0,m=n-(omit>=0);
    f.xa=x[a];f.ya=y[a];
    for(int i=0;i<n;i++) if(i!=omit) {
        f.xs=std::max(f.xs,std::abs(x[i]-f.xa));
        f.ys=std::max(f.ys,std::abs(y[i]-f.ya));
    }
    if(f.xs==0)throw std::runtime_error("rank");
    if(f.ys==0)f.ys=1;
    A q0{},v{},q1{},z{};
    long double r00=std::sqrt((long double)m),r01=0;
    for(int i=0;i<n;i++)if(i!=omit) {
        q0[i]=1/r00;v[i]=(x[i]-f.xa)/f.xs;z[i]=(y[i]-f.ya)/f.ys;
        r01+=q0[i]*v[i];
    }
    for(int i=0;i<n;i++)if(i!=omit)v[i]-=r01*q0[i];
    long double correction=0;
    for(int i=0;i<n;i++)if(i!=omit)correction+=q0[i]*v[i];
    r01+=correction;
    long double r11=0;
    for(int i=0;i<n;i++)if(i!=omit){v[i]-=correction*q0[i];r11=std::hypot(r11,v[i]);}
    long double c0=0,c1=0;
    for(int i=0;i<n;i++)if(i!=omit){q1[i]=v[i]/r11;c0+=q0[i]*z[i];c1+=q1[i]*z[i];}
    f.r00=r00;f.r01=r01;f.r11=r11;
    f.b1=c1/r11;f.b0=(c0-r01*f.b1)/r00;
    long double norm=0;
    for(int i=0;i<n;i++)if(i!=omit){
        f.e[i]=z[i]-(q0[i]*c0+q1[i]*c1);
        norm=std::hypot(norm,f.e[i]);f.h[i]=q0[i]*q0[i]+q1[i]*q1[i];
    }
    f.sigma=norm/std::sqrt((long double)(m-2));
    return f;
}
long double pred(const Fit& f,long double x){return f.b0+f.b1*((x-f.xa)/f.xs);}
long double difference(const Fit& f,const Fit& d,long double x){
    return (f.ya-d.ya)/f.ys+pred(f,x)-(d.ys/f.ys)*pred(d,x);
}
void check(std::ostream& o,std::string actual,long double expected,long double unit=1) {
    if(!std::isfinite(expected)||unit<=0||!std::isfinite(unit))throw std::runtime_error("oracle range");
    o<<"    assert(near("<<actual<<","<<(double)expected<<","<<(double)unit<<"))\n";
}
int main(int argc,char**argv){
    if(argc!=2)return 2;
    std::ofstream o(argv[1]);o<<std::scientific<<std::setprecision(17);
    o<<"//@ run-pass\n//@ expect-stdout: OLS_QR_REFERENCE_PASS\n"
       "use stats::regression::ols::{ols_fit_simple,ols_influence_simple}\n"
       "fn near(a:f64,b:f64,unit:f64) -> bool with Div { let d=if a>b {a-b}else{b-a} d/unit < 2.0e-8 }\n";
    int cases=0;
    const double ybase[8]={2,5.5,8,11,13.75,17,20,25};
    for(int mode=0;mode<11;mode++) {
        A x{},y{}; double sx=1,sy=1,tx=0,ty=0;
        if(mode==1)sx=1e-200;
        if(mode==2)sx=1e200;
        if(mode==3)sy=1e-200;
        if(mode==4)sy=1e200;
        if(mode==5){sx=1e-150;sy=1e150;}
        if(mode==6){sx=1e150;sy=1e-150;}
        if(mode==7){tx=1099511627776.0;ty=-2199023255552.0;}
        for(int i=0;i<8;i++){
            double xx=tx+sx*i, yy=ty+sy*ybase[i];
            if(mode==8 && i==7)xx=100;
            if(mode==9 && i==7)yy=200;
            if(mode==10 && i==6)yy=-30;
            x[i]=xx;y[i]=yy;
        }
        Fit f=qr(x,y,8);long double slope=f.b1*f.ys/f.xs;
        long double intercept=(f.ya-slope*f.xa)+f.ys*f.b0;
        o<<"fn case_"<<mode<<"() with IO, Mut, Div, Panic {\n"
         <<"    println(\"QR_CASE_"<<mode<<"\")\n"
         <<"    var x:[f64;100]=[0.0;100]\n    var y:[f64;100]=[0.0;100]\n";
        for(int i=0;i<8;i++)o<<"    x["<<i<<"]="<<(double)x[i]<<";y["<<i<<"]="<<(double)y[i]<<"\n";
        o<<"    let fit=ols_fit_simple(&x,&y,8)\n    assert(fit.status == 0 && fit.inference_status == 0)\n";
        check(o,"fit.slope",slope,std::abs(slope)+f.ys/f.xs);
        check(o,"fit.intercept",intercept,std::abs(intercept)+f.ys);
        check(o,"fit.residual_se",f.sigma*f.ys,f.ys);
        long double slope_se=(f.sigma/f.r11)*(f.ys/f.xs);
        long double xmean=f.xa+f.xs*f.r01/f.r00;
        long double intercept_se=std::hypot(f.sigma*f.ys/f.r00,slope_se*xmean);
        check(o,"fit.slope_se",slope_se,slope_se);
        check(o,"fit.intercept_se",intercept_se,intercept_se);
        o<<"    let d=ols_influence_simple(&x,&y,8)\n    assert(d.status == 0 && d.valid_count == 8)\n";
        for(int i=0;i<8;i++){
            Fit del=qr(x,y,8,i);
            const long double ds=del.sigma*del.ys/f.ys;
            long double dff=difference(f,del,x[i])/(ds*std::sqrt(f.h[i]));
            long double cook=0;
            for(int j=0;j<8;j++){long double z=difference(f,del,x[j])/f.sigma;cook+=z*z/2;}
            check(o,"fit.leverage["+std::to_string(i)+"]",f.h[i]);
            check(o,"fit.residuals["+std::to_string(i)+"]",f.e[i]*f.ys,f.ys);
            check(o,"d.dffits["+std::to_string(i)+"]",dff,1+std::abs(dff));
            check(o,"d.cooks_distance["+std::to_string(i)+"]",cook,1+std::abs(cook));
        }
        o<<"}\n";cases++;
    }
    o<<"fn main() with IO, Mut, Div, Panic {\n";
    for(int i=0;i<cases;i++)o<<"    case_"<<i<<"()\n";
    o<<"    println(\"OLS_QR_REFERENCE_PASS\")\n    0\n}\n";
    if(!o)return 3;
    std::cout<<"C++23 long-double QR: "<<cases<<" datasets, 88 actual deletion refits; generated numerical assertions.\n";
}
