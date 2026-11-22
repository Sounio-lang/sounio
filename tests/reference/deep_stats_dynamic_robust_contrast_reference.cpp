// C++23/binary128 direct raw-influence contrast reference. No c'Cov*c
// cancellation and no production QR/contrast implementation reused.
// Slurm: g++ -std=c++23 -O2 tests/reference/deep_stats_dynamic_robust_contrast_reference.cpp -lquadmath -o robust_contrast_reference
#define main saved_dynamic_robust_reference_main
#include "deep_stats_dynamic_robust_reference.cpp"
#undef main
#include <limits>
struct DRCReference{int status=0,variance_status=0;Q estimate=0,se=0,variance=0;};
static DRCReference drcreference(int kind){
 DRCReference out;
 if(kind>=4){out.status=kind==4?2:kind==7?4:1;out.variance_status=2;return out;}
 auto fit=drreference(kind==1?1:0);std::array<Q,DRP> c{};
 if(kind!=2){c[0]=kind==3?Q(double(1e200)):Q(1);if(kind!=3)c[1]=-Q(0.5);}
 for(int j=0;j<DRP;j++)out.estimate+=c[j]*fit.beta[j];
 std::map<int,Q> scores;std::array<Q,DRP> means{};
 for(int i=0;i<DRN;i++){int g=i/DRM,t=i%DRM;means[g]+=Q((g+1)*10+t%5-2)+Q(t%3)/4;}
 for(auto& mean:means)mean/=DRM;
 for(int i=0;i<DRN;i++){int g=i/DRM,t=i%DRM;Q residual=Q((g+1)*10+t%5-2)+Q(t%3)/4-means[g];
  Q influence=c[g]*residual/DRM;if(kind!=1)influence/=1-Q(1)/DRM;
  scores[kind==1?t%7:i]+=influence;
 }
 for(const auto& [key,v]:scores){(void)key;out.variance+=v*v;}
 out.se=sqrtq(out.variance);
 if(kind==3){drneed(out.se>0&&out.variance>Q(double(std::numeric_limits<double>::max())),"variance-only overflow discriminator");out.variance_status=1;out.variance=0;}
 return out;
}
int main(int argc,char** argv){try{
 std::array<DRCReference,8> refs{};for(int kind=0;kind<8;kind++){refs[kind]=drcreference(kind);std::cout<<"reference contrast case: "<<kind<<'\n';drrow("estimate",refs[kind].estimate);drrow("SE",refs[kind].se);}
 if(argc==2){std::ifstream in(argv[1]);drneed(bool(in),"native contrast stdout open");std::string line;int kind=-1,cases=0,seen=0;bool done=false;std::string prefix="dynamic robust contrast ";
  while(std::getline(in,line)){
   if(line=="DEEP DYNAMIC ROBUST CONTRAST PASS")done=true;
   if(!line.starts_with(prefix))continue;auto sep=line.find(": ");drneed(sep!=std::string::npos,"contrast syntax");auto key=line.substr(prefix.size(),sep-prefix.size()),value=line.substr(sep+2);
   if(key=="case"){if(kind>=0)drneed(seen==31,"previous contrast fields");kind=std::stoi(value);drneed(kind==cases&&kind<8,"contrast order");cases++;seen=0;continue;}
   drneed(kind>=0,"contrast context");const auto& r=refs[kind];Q x=strtoflt128(value.c_str(),nullptr);
   if(key=="status"){drneed(std::stoi(value)==r.status,"contrast availability status");seen|=1;}
   else if(key=="estimate"){drclose(x,r.estimate,0,Q(2e-10),"native contrast estimate");seen|=2;}
   else if(key=="se"){drclose(x,r.se,0,Q(2e-9),"native influence contrast SE");seen|=4;}
   else if(key=="variance"){drclose(x,r.variance,0,Q(4e-9),"native contrast variance");seen|=8;}
   else if(key=="variance status"){drneed(std::stoi(value)==r.variance_status,"variance availability independent from SE");seen|=16;}
  }
  drneed(cases==8&&seen==31&&done,"full native contrast completion");std::cout<<"ACTUAL ROBUST CONTRAST CONCORDANCE PASS\n";
 }
 std::cout<<"DEEP ROBUST CONTRAST CPP PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
