// Independent official Boost stepper controls with the same RMS controller.
// Default remains ORIGINAL positive-d4 Boost; corrected modes change ONLY d4.
// No production Sounio coefficients or kernels are called. Slurm execution only.
#define main ros4_fixture_emitter_main
#include "deep_ros4_reference.cpp"
#undef main
#include <algorithm>
#include <cstdint>
struct PureTimeRhs {void operator()(const State& y,State& d,double t)const {(void)y;d[0]=2*t;}};
struct PureTimeJac {void operator()(const State& y,Matrix& j,double t,State& dfdt)const {(void)y;(void)t;j(0,0)=0;dfdt[0]=2;}};
struct KppRodasCoefficients: on::default_rosenbrock_coefficients<double> {
 using Base=on::default_rosenbrock_coefficients<double>;
 const double d4;
 KppRodasCoefficients():Base(),d4(-0.3620000000000023e-01){}
};
using KppStep=on::rosenbrock4<double,KppRodasCoefficients>;
template<class SelectedStep> int run_control(int argc,char**argv){
 const bool augmented=argc>1&&std::string(argv[1]).find("augmented")!=std::string::npos;
 if(argc>1&&std::string(argv[1]).find("pure-time")!=std::string::npos){
  SelectedStep single;State initial(1),trial(1),error(1);initial[0]=0;double h=.1;bool okay=true;
  for(int k=0;k<8;++k){single.do_step(std::make_pair(PureTimeRhs{},PureTimeJac{}),initial,0.0,trial,h,error);double defect=trial[0]-h*h;std::cout<<std::setprecision(17)<<"PURE_TIME h "<<h<<" state "<<trial[0]<<" target "<<h*h<<" defect "<<defect<<" estimate "<<error[0]<<"\n";if(std::abs(defect)>1e-13*h*h)okay=false;h*=.5;}
  return okay?0:1;
 }
 const int n=augmented?3:2;
 State y(n),trial(n),error(n);for(int i=0;i<n;++i)y[i]=0;
 SelectedStep step;double t=0,dt=.001;
 std::int64_t attempts=0,accepted=0,rejected=0;
 std::cout<<std::setprecision(17);
 while(t!=1.0&&attempts<1000000){
  const double h=std::min(dt,1.0-t),next=t+h;
  if(next<=t)return 2;
  step.do_step(std::make_pair(Rhs{augmented?3:1},Jac{augmented?3:1}),y,t,trial,h,error);
  ++attempts;double ssq=0;
  for(int i=0;i<n;++i){double scale=1e-14+1e-11*std::max(std::abs(y[i]),std::abs(trial[i]));double v=error[i]/scale;ssq+=v*v;}
  const double measured=std::sqrt(ssq/n);
  double multiplier=measured>0?.9*std::exp(-.25*std::log(measured)):10;
  multiplier=std::clamp(multiplier,.2,10.0);
  if(measured<=1){
   y=trial;t=next;++accepted;dt=std::clamp(h*multiplier,1e-12,.01);
   if(accepted<=5||accepted%1024==0)std::cout<<"BOOST_PROGRESS "<<accepted<<" t "<<t<<" h "<<h<<" y0 "<<y[0]<<" y1 "<<y[1]<<" err "<<measured<<"\n";
  }else{
   ++rejected;multiplier=std::min(.9,multiplier);if(h<=1e-12)return 2;
   dt=std::clamp(h*multiplier,1e-12,.01);
  }
 }
 std::cout<<"BOOST_RESULT augmented "<<augmented<<" attempts "<<attempts<<" accepted "<<accepted<<" rejected "<<rejected<<" t "<<t<<" y0 "<<y[0]<<" y1 "<<y[1]<<"\n";
 if(t!=1)return 1;
 for(int i=0;i<n;++i)if(std::abs(y[i]-1)>1e-8)return 1;
 return 0;
}

int main(int argc,char**argv){
 if(argc>1&&std::string(argv[1]).find("corrected")!=std::string::npos)return run_control<KppStep>(argc,argv);
 on::default_rosenbrock_coefficients<double> original;
 if(BOOST_VERSION!=108300||original.d4<=0){std::cerr<<"Original-control unavailable: expected attested Boost1.83 positive d4\n";return 77;}
 return run_control<OfficialBoostStep>(argc,argv);
}
