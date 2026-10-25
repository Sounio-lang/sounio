// Independent C++23 binary128 raw-log estimator and actual native report verifier.
// Numerical execution belongs to an allocated Slurm job, never the workspace.
#include <quadmath.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Q=__float128;
static void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
static long long count(int g,int s,bool zeros){if(zeros&&(g==5||(g==0&&s==0)))return 0;int m=g<3?(s%2==0?1:2):(s==0?1:(s==1?8:(s==2?4:16)));return (g+1)*4*m;}
struct Reference{std::array<Q,4> logs{},factors{};int used=0,excluded=0;};
static Reference reference(bool zeros,int mode){
 Reference out;std::array<Q,6> means{};std::array<bool,6> eligible{};
 for(int g=0;g<6;++g){int positive=0;Q total=0;for(int s=0;s<4;++s){auto y=count(g,s,zeros);if(y>0){++positive;total+=logq(Q(y));}}eligible[g]=positive>0&&(mode==1||positive==4);if(eligible[g]){++out.used;means[g]=total/Q(positive);}}
 out.excluded=6-out.used;Q center=0;
 for(int s=0;s<4;++s){std::vector<Q> ratios;for(int g=0;g<6;++g)if(eligible[g]&&count(g,s,zeros)>0)ratios.push_back(logq(Q(count(g,s,zeros)))-means[g]);need(!ratios.empty(),"reference eligible ratios");std::sort(ratios.begin(),ratios.end());size_t n=ratios.size();out.logs[s]=n%2?ratios[n/2]:(ratios[n/2-1]+ratios[n/2])/2;center+=out.logs[s];}
 center/=4;for(int s=0;s<4;++s){out.logs[s]-=center;out.factors[s]=expq(out.logs[s]);}return out;
}
static Q number(const std::string& s){char* end=nullptr;Q x=strtoflt128(s.c_str(),&end);need(end!=s.c_str()&&*end=='\0'&&!isnanq(x)&&!isinfq(x),"complete finite number");return x;}
static void close(Q actual,Q expected,const char* name){Q tolerance=strtoflt128("2e-11",nullptr)*(1+fabsq(expected));need(fabsq(actual-expected)<=tolerance,std::string("global normalization mismatch: ")+name);}
struct Reader{std::ifstream in;explicit Reader(const char* p):in(p){need(bool(in),"open actual native report");}std::string line(){std::string s;need(bool(std::getline(in,s)),"complete native report");return s;}
 std::vector<long long> ints(const char* key,size_t n){std::istringstream row(line());std::string k;need(bool(row>>k)&&k==key,"record schema/order");std::vector<long long> out(n);for(auto& v:out)need(bool(row>>v),"integer field");need(!(row>>k),"no extra integer field");return out;}
 Q scalar(const char* key){std::string s=line(),prefix=std::string(key)+" ";need(s.starts_with(prefix),"scalar key/order");return number(s.substr(prefix.size()));}
};
int main(int argc,char**argv){try{
 if(argc!=2)return 64;
 Reader reader(argv[1]);auto payload=reader.ints("PAYLOAD",5);
 need(payload[0]-payload[1]==8*(3*3+2*4),"exact omitted worker normalization scratch");need(payload[2]==8*2*4&&payload[3]==8*(3*6+2*4)&&payload[4]==payload[2]+payload[3],"global owner retained/scratch/peak query");
 int samples=0,means=0;
 for(int zeros=0;zeros<2;++zeros)for(int mode=0;mode<2;++mode){auto expected=reference(zeros!=0,mode);for(int part=0;part<4;++part){auto row=reader.ints("CASE",5);need(row[0]==zeros&&row[1]==mode&&row[2]==part&&row[3]==expected.used&&row[4]==expected.excluded,"global reference population metadata");int rotation=(part+3)%4;for(int j=0;j<4;++j){int sample=(j+rotation)%4;auto id=reader.ints("SAMPLE",1);need(id[0]==sample,"byte cohort permutation binding");close(reader.scalar("FACTOR"),expected.factors[sample],"factor");close(reader.scalar("OFFSET"),expected.logs[sample]+Q(sample)/4,"exposure added exactly once");++samples;}
 if(part<2){int gene=part*3;Q mean=0;for(int sample=0;sample<4;++sample)mean+=Q(count(gene,sample,zeros!=0))/expq(expected.logs[sample]+Q(sample)/4)/4;close(reader.scalar("MEAN"),mean,"actual point-fit normalized mean");++means;}
 }}
 need(reader.line()=="GENOMICS_GLOBAL_NORMALIZATION_PASS","native completion marker");std::string extra;need(!std::getline(reader.in,extra),"no trailing native records");
 std::cout<<"GENOMICS_GLOBAL_NORMALIZATION_REFERENCE_PASS scenarios=16 sample_bindings="<<samples<<" actual_point_means="<<means<<" modes=2 zero_policies=2 partitions=4\n";return 0;
}catch(const std::exception& e){std::cerr<<"GENOMICS_GLOBAL_NORMALIZATION_REFERENCE_REJECT "<<e.what()<<'\n';return 2;}}
