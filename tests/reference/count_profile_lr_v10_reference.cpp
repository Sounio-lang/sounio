// Fixed-point diagnostic variant10. Independent FULL PMFs; no refit or clamps.
// Compile/run only in Slurm, uses count_dispersion_increment_reference.cpp.
#define main saved_alpha_main
#include "count_dispersion_increment_reference.cpp"
#undef main
#include <fstream>
#include <sstream>
static double raw(long long b){return std::bit_cast<double>(std::bit_cast<std::uint64_t>(b));}
struct VRow{int y;double mu0,mu1,p0,p1;};
int main(int argc,char** argv){try{need(argc==2,"diagnostic path");std::ifstream in(argv[1]);need(bool(in),"open original");std::vector<VRow> rows;B a0=0,a1=0;double native=0;bool complete=false,has0=false,has1=false,haslr=false;std::string line;
 while(std::getline(in,line)){std::istringstream f(line);std::string tag;f>>tag;if(tag=="LAST_ALPHA0"||tag=="LAST_ALPHA1"||tag=="FAIL_LR"){long long b;f>>b;need(bool(f),"scalar rawbits");if(tag=="LAST_ALPHA0"){a0=exact(raw(b));has0=true;}else if(tag=="LAST_ALPHA1"){a1=exact(raw(b));has1=true;}else{native=raw(b);haslr=true;}}else if(tag=="DIAG_ROW"){int index,s0,s1;long long y,observed,m0,m1,e0,e1,p0,p1;f>>index>>observed>>y>>m0>>m1>>e0>>e1>>s0>>p0>>s1>>p1;need(bool(f)&&index==int(rows.size())&&observed==6&&y==6&&s0==0&&s1==0,"original v10 row");rows.push_back({int(y),raw(m0),raw(m1),raw(p0),raw(p1)});}else if(tag=="LR_DIAGNOSTIC_COMPLETE")complete=true;}
 need(complete&&has0&&has1&&haslr&&rows.size()==16&&a0==0&&a1==0,"complete v10 original");B fixed=0,rounded=0;
 for(const auto& r:rows){fixed+=2*(pmf(r.y,exact(r.mu1),a1)-pmf(r.y,exact(r.mu0),a0));rounded+=2*(exact(r.p1)-exact(r.p0));}
 std::cout<<std::setprecision(70)<<"fixed_native_point_LR\t"<<fixed<<"\nnative_paired_LR\t"<<exact(native)<<"\nrounded_row_PMF_LR\t"<<rounded<<"\n";
 need(fixed<0&&native<0,"signed negative truth preserved");need(rounded==0,"old rounded full-PMF tie reproduced");need(abs(exact(native)-fixed)<=abs(fixed)*B("1e-11"),"native local paired vs independent fixed-point LR");
 std::cout<<"V10_FIXED_POINT_REFERENCE_COMPLETE\nLIMITS\tpoints captured from original variant10; no refit; rounded PMF tie is not exact equality\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
