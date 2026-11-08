#include <array>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using I=long long;
using S=std::array<I,7>;
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("usage: bench-reference side native-stdout");
 const I n=std::stoll(argv[1]);if(n!=256&&n!=1024)throw std::runtime_error("side");
 std::ifstream f(argv[2]);if(!f)throw std::runtime_error("input");std::string all((std::istreambuf_iterator<char>(f)),{});
 if(all.find("PREPARED_CONTOUR_BENCH_PASS")==std::string::npos)throw std::runtime_error("incomplete benchmark");
 const std::regex re(R"re(<line data-contour-cell="([0-9]+)" data-contour-level-index="([0-9]+)" data-contour-branch="([0-9]+)" data-contour-level="[^"]+" x1="([0-9-]+)" y1="([0-9-]+)" x2="([0-9-]+)" y2="([0-9-]+)" stroke="rgb\(70,180,235\)" stroke-width="1" stroke-linecap="round"/>)re");
 std::vector<S> got,want;
 for(std::sregex_iterator it(all.begin(),all.end(),re),end;it!=end;++it){S s{};for(int j=0;j<7;++j)s[j]=std::stoll((*it)[j+1]);got.push_back(s);}
 auto x=[&](I q){return q*256/(4*n);};auto y=[&](I q){return 256-q*256/(4*n);};
 for(I r=0;r<n;++r)for(I c=0;c<n;++c){
  if(r+c==n-1)want.push_back({r*n+c,0,0,x(4*c+4),y(4*r+1),x(4*c+1),y(4*r+4)});
  if(r+c==n)want.push_back({r*n+c,0,0,x(4*c+1),y(4*r),x(4*c),y(4*r+1)});
 }
 if(got!=want)throw std::runtime_error("independent quarter-grid geometry/provenance mismatch");
 std::cout<<"PREPARED_BENCH_REFERENCE_PASS side="<<n<<" cells="<<n*n<<" segments="<<want.size()<<" exact_rational_projection\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
