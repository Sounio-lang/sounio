// Independent C++23 analytic and exact-rational witnesses for production DP54.
// Only native imported-function records are accepted; no Sounio equations linked.
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Q=boost::multiprecision::cpp_dec_float_100;
using R=boost::multiprecision::cpp_rational;
Q exact(double x){auto b=std::bit_cast<std::uint64_t>(x);int e=int((b>>52)&2047);auto f=b&((1ull<<52)-1);if(e==2047)throw std::runtime_error("nonfinite native value");Q q=e?Q(f|(1ull<<52)):Q(f);q=ldexp(q,e?e-1075:-1074);return b>>63?-q:q;}
double decode(const std::string&s){std::size_t p=0;auto b=std::stoll(s,&p);if(p!=s.size())throw std::runtime_error("invalid bits");double x=std::bit_cast<double>(std::uint64_t(b));if(!std::isfinite(x))throw std::runtime_error("nonfinite record");return x;}
R rat(std::int64_t a,std::int64_t b=1){return R(a)/b;}
void coefficient_witness(){
 // Exact rational data for the published Shampine quartic, not a theorem claim.
 const R d1=rat(-12715105075ll,11282082432ll),d3=rat(87487479700ll,32700410799ll),d4=rat(-10690763975ll,1880347072ll),d5=rat(701980252875ll,199316789632ll),d6=rat(-1453857185ll,822651844ll),d7=rat(69997945ll,29380423ll);
 const R k1=0,k3=4*rat(3,10)*rat(3,10)*rat(3,10),k4=4*rat(4,5)*rat(4,5)*rat(4,5),k5=4*rat(8,9)*rat(8,9)*rat(8,9),k6=4,k7=4;
 const R y1=rat(35,384)*k1+rat(500,1113)*k3+rat(125,192)*k4-rat(2187,6784)*k5+rat(11,84)*k6;
 const R r5=d1*k1+d3*k3+d4*k4+d5*k5+d6*k6+d7*k7;
 if(y1!=1||r5!=1)throw std::runtime_error("quartic monomial rational witness");
 const R defect=rat(71,57600)*k1-rat(71,16695)*k3+rat(71,1920)*k4-rat(17253,339200)*k5+rat(22,525)*k6-rat(1,40)*k7;
 if(defect!=0)throw std::runtime_error("embedded polynomial exactness");
 const R a=3,b=-7,h=rat(1,8),f0=5,f1=-2,z=17;
 const R r2=b-a,r3=h*f0-r2,r4=r2-h*f1-r3;
 const std::array<R,5> p={a,r2+r3,-r3+r4+z,-r4-2*z,z};
 R sum=0,slope=0;for(int i=0;i<5;++i){sum+=p[i];slope+=i*p[i];}
 if(p[0]!=a||sum!=b||p[1]/h!=f0||slope/h!=f1)throw std::runtime_error("quartic endpoint/slope rational witness");
}
void near(const std::string&key,double actual,Q expected,Q rel=Q("5e-11"),Q abs_tol=Q("1e-14")){Q err=abs(exact(actual)-expected);if(err>abs(expected)*rel+abs_tol){std::ostringstream s;s<<key<<" actual="<<actual<<" expected="<<expected<<" abs_error="<<err;throw std::runtime_error(s.str());}}
struct Meta{long status,steps,attempts,nfeval,nreject,actual;};
struct Event{long status,count,evals,iters,actual,found,bracket;};
struct Point{double t,y,slope;};
struct Case{long status,receipt_status;bool found,bracket;long dense_status;double time;};
const std::map<std::string,Case> cases={
 {"terminal",{5,0,true,true,0,.5}},{"backward",{5,0,true,true,0,.5}},
 {"wrong_direction",{0,0,false,false,0,1}},{"nonterminal",{0,0,true,true,0,1}},
 {"initial",{5,0,true,true,3,0}},{"initial_ignored",{0,0,false,false,0,1}},
 {"initial_directed",{5,0,true,true,3,0}},{"endpoint",{5,0,true,true,0,1}},
 {"root_budget0",{6,6,false,true,6,0}},{"root_budget1",{6,6,false,true,6,0}},
 {"event_budget0",{1,1,false,false,3,0}},{"event_budget1",{1,1,false,false,3,0}},
 {"nan_initial",{4,4,false,false,3,0}},{"nan_endpoint",{4,4,false,false,3,0}},
 {"nan_interior",{4,4,false,true,4,0}},{"exponential",{5,0,true,true,0,0}},
 {"grazing",{0,0,false,false,0,1}},{"reuse_first",{5,0,true,true,0,.5}},
 {"reuse_second",{0,0,false,false,0,.625}},
 {"multi_any",{0,0,true,true,0,1}},{"multi_rising",{0,0,true,true,0,1}},{"multi_falling",{0,0,true,true,0,1}}
};
int baseline(const std::string&mode,const std::string&file){
 struct Row{long status,steps,attempts,nfeval,actual;double x,t;};std::map<std::string,Row>rows;
 std::ifstream f(file);if(!f)throw std::runtime_error("missing baseline stdout");int markers=0;
 for(std::string line;std::getline(f,line);){if(line=="DP54_BASELINE_DONE"){++markers;continue;}std::istringstream in(line);std::string tag,name,xb,tb,extra;Row r;if(!(in>>tag>>name>>r.status>>r.steps>>r.attempts>>r.nfeval>>r.actual>>xb>>tb)||(in>>extra)||tag!="DP54_BASELINE")throw std::runtime_error("bad baseline record");r.x=decode(xb);r.t=decode(tb);if(!rows.emplace(name,r).second)throw std::runtime_error("duplicate baseline");}
 if(markers!=1||rows.size()!=3)throw std::runtime_error("incomplete baseline");
 const auto&fixed=rows.at("fixed");const auto&actual=rows.at("actual");const auto&cross=rows.at("cross");
 for(const auto&[name,r]:rows){if(r.status!=0||r.nfeval!=r.actual||r.steps!=r.attempts)throw std::runtime_error(name+" baseline counters");}
 near("baseline fixed",fixed.x,exact(.5));if(fixed.t!=.5||fixed.steps!=4)throw std::runtime_error("fixed baseline mesh");
 double target=1.0000000000000089;Q true_span=exact(target)-1,true_value=true_span*exact(1e15);
 if(actual.t!=target||actual.steps!=2||cross.t!= -1)throw std::runtime_error("baseline endpoints");
 if(mode=="before"){
  if(fixed.nfeval!=28||actual.nfeval!=14||cross.nfeval!=7||cross.steps!=1)throw std::runtime_error("old baseline call cost");
  Q error=abs(exact(actual.x)-true_value);double rounded_span=-1.0-1e20,reached=1e20+rounded_span;
  if(error<Q("0.01")||reached==cross.t)throw std::runtime_error("before lost discrimination");
  std::cout<<"DP54_BEFORE_WITNESS_PASS fixed_rhs=28 actual_span_error="<<error<<" crossed_endpoint="<<cross.t<<" represented_addition_endpoint="<<reached<<" old_cross_steps=1\n";
 }else{
  if(fixed.nfeval!=25||actual.nfeval!=13||cross.nfeval!=13||cross.steps!=2)throw std::runtime_error("FSAL baseline call cost");
  near("corrected actual span",actual.x,true_value,Q("5e-14"));near("cross state",cross.x,-exact(1e20)-1,Q("5e-14"));
  std::cout<<"DP54_AFTER_CONTROL_PASS fixed_rhs=25 actual_rhs=13 cross_steps=2 cross_rhs=13\n";
 }
 return 0;
}
int invalid_control(const std::string&file){
 std::ifstream f(file);if(!f)throw std::runtime_error("missing invalid-request stdout");int n=0,markers=0;
 const std::array<long,9>expected={3,0,0,3,0,0,0,0,0};
 for(std::string line;std::getline(f,line);){if(line=="DP54_INVALID_PASS"){++markers;continue;}std::istringstream s(line);std::string tag,extra;int kind;if(!(s>>tag>>kind)||tag!="DP54_INVALID"||kind!=n)throw std::runtime_error("invalid-request record");for(long e:expected){long actual;if(!(s>>actual)||actual!=e)throw std::runtime_error("invalid-request status/callback contract");}if(s>>extra)throw std::runtime_error("invalid-request extra fields");++n;}
 if(n!=12||markers!=1)throw std::runtime_error("missing invalid-request records");
 std::cout<<"DP54_INVALID_CPP23_PASS cases="<<n<<" no_rhs_or_event_callbacks=verified stale_receipt_reset=verified\n";return 0;
}
int main(int argc,char**argv){try{
 coefficient_witness();
 if(argc==3&&std::string(argv[1])=="invalid")return invalid_control(argv[2]);
 if(argc==3&&(std::string(argv[1])=="before"||std::string(argv[1])=="after"))return baseline(argv[1],argv[2]);
 if(argc==2&&std::string(argv[1])=="selftest"){std::cout<<"DP54_CPP23_RATIONAL_WITNESS_PASS monomial=quartic endpoints=values_and_slopes\n";return 0;}
 if(argc!=2)throw std::runtime_error("usage: reference selftest|native.stdout");
 std::ifstream in(argv[1]);if(!in)throw std::runtime_error("missing stdout");
 std::map<std::string,double> values;std::map<std::string,Meta> metas;std::map<std::string,Event> events;std::map<int,Point> points;int markers=0;
 for(std::string line;std::getline(in,line);){std::istringstream s(line);std::string tag;s>>tag;if(tag.empty())continue;
  if(tag=="STDLIB_DP54_DENSE_EVENTS_PASS"){std::string rest;if(s>>rest)throw std::runtime_error("malformed marker");++markers;continue;}
  if(tag=="DP54_VALUE"){std::string key,bits,rest;if(!(s>>key>>bits)||(s>>rest)||!values.emplace(key,decode(bits)).second)throw std::runtime_error("malformed/duplicate value");continue;}
  if(tag=="DP54_META"){std::string key,rest;Meta m;if(!(s>>key>>m.status>>m.steps>>m.attempts>>m.nfeval>>m.nreject>>m.actual)||(s>>rest)||!metas.emplace(key,m).second)throw std::runtime_error("malformed/duplicate metadata");continue;}
  if(tag=="DP54_EVENT"){std::string key,rest;Event e;if(!(s>>key>>e.status>>e.count>>e.evals>>e.iters>>e.actual>>e.found>>e.bracket)||(s>>rest)||!events.emplace(key,e).second)throw std::runtime_error("malformed/duplicate event");continue;}
  if(tag=="DP54_POINT"){int id;std::string t,y,d,rest;if(!(s>>id>>t>>y>>d)||(s>>rest)||!points.emplace(id,Point{decode(t),decode(y),decode(d)}).second)throw std::runtime_error("malformed/duplicate point");continue;}
  if(tag.rfind("DP54_",0)==0)throw std::runtime_error("unexpected DP54 record");
 }
 if(markers!=1)throw std::runtime_error("missing completion marker");
 std::set<std::string> consumed;
 auto take=[&](const std::string&key){consumed.insert(key);return values.at(key);};
 auto check=[&](const std::string&key,Q expected,Q rel=Q("5e-11"),Q a=Q("1e-14")){near(key,take(key),expected,rel,a);};
 check("constant.end",exact(.5));check("constant.t0",exact(.375),0,0);check("constant.t1",exact(.5),0,0);check("constant.interior",exact(.4));check("constant.slope",1);
 check("decay.end",exp(Q(-1)),Q("5e-10"));double mid=take("decay.midtime");if(!(mid>0&&mid<1))throw std::runtime_error("decay dense domain");check("decay.mid",exp(-exact(mid)),Q("5e-9"));check("decay.slope",-exp(-exact(mid)),Q("5e-8"));check("restart.end",exp(Q(-1))+exact(.25),Q("5e-10"));
 double target=1.0000000000000089;check("time.actual.mean",exact(1e15)*exact(target-1),Q("5e-14"));check("time.actual.time",exact(target),0,0);
 check("time.stall.mean",0,0,0);check("time.stall.time",1,0,0);check("time.tiny.mean",exact(1e-20),Q("5e-13"),Q("1e-36"));check("time.tiny.time",exact(1e-20),0,0);check("time.cross.mean",-exact(1e20)-1,Q("5e-14"));check("time.cross.time",-1,0,0);
 const std::array<double,7> nodes={0,.2,.3,.5,.8,8.0/9.0,1};
 if(points.size()!=nodes.size())throw std::runtime_error("missing dense nodes");
 for(std::size_t i=0;i<nodes.size();++i){const auto&p=points.at(int(i));if(p.t!=nodes[i])throw std::runtime_error("node input parsing");Q t=exact(p.t);near("quartic value",p.y,t*t*t*t,Q("5e-12"),Q("1e-14"));near("quartic slope",p.slope,4*t*t*t,Q("5e-12"),Q("1e-14"));}
 auto meta=[&](const std::string&key,long status,long steps,long attempts,long nfeval,long reject){const auto&m=metas.at(key);if(m.status!=status||m.steps!=steps||m.attempts!=attempts||m.nfeval!=nfeval||m.nreject!=reject||m.actual!=nfeval)throw std::runtime_error(key+" counters");};
 meta("constant",0,4,4,25,0);meta("polynomial",0,1,1,7,0);meta("restart",0,1,1,7,0);
 meta("budget0",1,0,1,0,0);meta("budget1",1,0,1,1,0);meta("partial",4,0,1,1,0);meta("staged_partial",4,0,1,2,0);meta("stage_stall",2,0,0,0,0);meta("cross_zero",0,2,2,13,0);
 const auto&rejection=metas.at("rejection");if(rejection.status!=0||rejection.nreject<=0||rejection.steps+rejection.nreject!=rejection.attempts||rejection.nfeval!=1+6*rejection.attempts||rejection.actual!=rejection.nfeval)throw std::runtime_error("rejection FSAL counter");
 for(const auto&key:{"actual_span","tiny_span"}){const auto&m=metas.at(key);if(m.status!=0||m.steps<=0||m.steps!=m.attempts||m.nreject!=0||m.nfeval!=1+6*m.attempts||m.actual!=m.nfeval)throw std::runtime_error(std::string(key)+" represented grid counter");}
 for(const auto&[name,c]:cases){
  const auto&m=metas.at(name);const auto&e=events.at(name);const std::string k="event."+name+".";
  if(m.status!=c.status||m.actual!=m.nfeval||e.status!=c.receipt_status||e.count!=(name=="multi_any"?2:c.found?1:0)||e.found!=long(c.found)||e.bracket!=long(c.bracket)||e.evals!=e.actual||e.iters<0)throw std::runtime_error(name+" event metadata");
  if(m.attempts>0&&m.nfeval!=1+6*m.attempts)throw std::runtime_error(name+" full-trial FSAL count");
  if(m.attempts==0&&m.nfeval!=0)throw std::runtime_error(name+" callback before trial");
  long expected_eval=0;
  if(name=="initial"||name=="nan_initial"||name=="event_budget1")expected_eval=1;
  else if(name=="event_budget0")expected_eval=0;
  else if(name=="nonterminal")expected_eval=5;
  else if(name.rfind("multi_",0)==0)expected_eval=9;
  else if(name=="root_budget1"||name=="nan_interior")expected_eval=3;
  else if(name=="terminal"||name=="backward"||name=="exponential"||name=="reuse_first")expected_eval=1+m.steps+e.iters;
  else expected_eval=2;
  if(e.evals!=expected_eval)throw std::runtime_error(name+" actual event callback count");
  if((name=="nonterminal"||name.rfind("multi_",0)==0||name=="endpoint"||name=="initial"||name=="initial_ignored"||name=="initial_directed"||name=="wrong_direction"||name=="grazing"||name=="reuse_second")&&e.iters!=0)throw std::runtime_error(name+" unperformed root iterations");
  if((name=="root_budget0"&&e.iters!=0)||(name=="root_budget1"&&e.iters!=1)||(name=="nan_interior"&&e.iters!=1))throw std::runtime_error(name+" root budget counter");
  if((name=="initial"||name=="event_budget0"||name=="nan_initial")&&(m.attempts!=0||m.steps!=0))throw std::runtime_error(name+" initial work");
  if((c.status==1||c.status==4||c.status==6||name=="initial_directed")&&m.steps!=0)throw std::runtime_error(name+" failure state atomicity");
  Q time=name=="exponential"?log(Q(2)):exact(c.time);double observed_time=take(k+"time");near(k+"time",observed_time,time,Q("5e-10"),Q("2e-10"));
  Q mean=name=="exponential"?Q(2):name=="reuse_second"?exact(.75):time;check(k+"mean",mean,Q("5e-10"),Q("2e-10"));
  Q root_time=0,root_state=0;
  if(c.found){root_time=name=="initial"||name=="initial_directed"?Q(0):name=="endpoint"?Q(1):name=="exponential"?log(Q(2)):name=="multi_any"||name=="multi_rising"?exact(.75):name=="multi_falling"?exact(.25):exact(.5);root_state=name=="exponential"?Q(2):root_time;}
  check(k+"root_time",root_time,Q("5e-10"),Q("2e-10"));check(k+"root_state",root_state,Q("5e-10"),Q("2e-10"));check(k+"root_value",0,0,Q("1e-9"));
  check(k+"state_status",c.found?0:c.receipt_status?c.receipt_status:3,0,0);
  check(k+"dense_status",c.dense_status,0,0);
  double ds=take(k+"dense_start"),de=take(k+"dense_end");
  if(c.dense_status!=0){if(ds!=0||de!=0)throw std::runtime_error(name+" unavailable dense payload");}
  else {if(de!=observed_time)throw std::runtime_error(name+" terminal dense domain");if(name=="backward"){if(ds!=1||de>=ds)throw std::runtime_error("backward dense domain");}else if(name=="exponential"){if(!(ds>=0&&ds<de))throw std::runtime_error("exponential dense domain");}else {double expected_start=name=="nonterminal"?.75:name=="reuse_second"?.5:name.rfind("multi_",0)==0?.875:0;if(ds!=expected_start)throw std::runtime_error(name+" dense start");}}
  double lo=take(k+"lo"),hi=take(k+"hi"),gl=take(k+"glo"),gh=take(k+"ghi");
  if(!c.bracket){if(lo!=0||hi!=0||gl!=0||gh!=0)throw std::runtime_error(name+" absent bracket payload");}
  else {if(!(lo<=hi&&gl<=0&&gh>=0))throw std::runtime_error(name+" invalid actual root bracket");if(c.found){double rt=values.at(k+"root_time");if(!(lo<=rt&&rt<=hi)||hi-lo>1.000001e-10)throw std::runtime_error(name+" root bracket width");}if(name=="root_budget0"||name=="root_budget1"||name=="nan_interior"){near(name+" g_lo",gl,exact(lo)-exact(.3),0,Q("1e-15"));near(name+" g_hi",gh,exact(hi)-exact(.3),0,Q("1e-15"));}}
 }
 if(consumed.size()!=values.size()||metas.size()!=12+cases.size()||events.size()!=cases.size())throw std::runtime_error("unknown/missing scientific records");
 std::cout<<"DP54_CPP23_REFERENCE_PASS values="<<values.size()<<" dense_nodes="<<points.size()<<" metadata="<<metas.size()<<" events="<<events.size()<<" precision=decimal100 exact_rational_coefficients=qualified\n";
 return 0;
}catch(const std::exception&e){std::cerr<<"DP54_CPP23_REFERENCE_FAIL "<<e.what()<<'\n';return 1;}}
