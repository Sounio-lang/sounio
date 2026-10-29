#include <bit>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <stdexcept>
using I=std::int64_t;
std::string read(const char*p){std::ifstream f(p);if(!f)throw std::runtime_error(p);return {std::istreambuf_iterator<char>(f),{}};}
I ordinal(std::string s,std::string en,std::string value){s=std::regex_replace(s,std::regex("//[^\\n]*"),"");std::smatch m;if(!std::regex_search(s,m,std::regex("pub enum "+en+R"(\s*\{([^}]+)\})")))throw std::runtime_error("enum");std::string b=m[1];std::regex r(R"(([A-Za-z_][A-Za-z0-9_]*)\s*,)");I i=0;for(auto q=std::sregex_iterator(b.begin(),b.end(),r);q!=std::sregex_iterator();++q,++i)if((*q)[1]==value)return i;throw std::runtime_error("variant");}
I sum(I seed){I n=0;for(I i=0;i<512;++i)n+=seed+i;return n;}
template<class...T>std::string line(T...x){std::ostringstream s;bool first=true;auto add=[&](auto a){if(!first)s<<' ';first=false;s<<a;};(add(x),...);return s.str();}
int main(int ac,char**av){try{if(ac!=4)throw std::runtime_error("argc");I lit=ordinal(read(av[1]),"ExprKind","ExprIntLit"),path=ordinal(read(av[1]),"ExprKind","ExprPath"),i64=ordinal(read(av[2]),"TypeKind","TyI64");std::vector<std::string>e;for(I id=0;id<3;++id){I n=id==2?3:id;e.push_back(line("PARAMS",id,n,n,sum(700+n)));for(I j=0;j<n;++j)e.push_back(line("TYPE",id,j,1,64+n-j,i64,0,0,j+1,-1,std::bit_cast<I>(double(j+1)+0.5),0,-1,-1,0,-1,-1,-1,101+j,-1,-1,-1,-1,-1,-1));}for(I id=0;id<5;++id)e.push_back(line("LOWER",id,id==2?3:0,sum(710+id),id==0?0:1,id==0?-1:lit,id==0?-1:7));for(I id=0;id<4;++id){I op=id==2?1:-1;e.push_back(line("REWRITE",id,op,sum(720+id)));if(id>0)e.push_back(line("EXPR",id,0,lit,8));if(id>=2){e.push_back(line("EXPR",id,1,id==2?lit:path,id==2?1:0));e.push_back(line("EXPR",id,2,lit,9));}e.push_back(line("LENGTH",id,id==0?0:id==1?1:3));}std::ifstream f(av[3]);std::string l;std::size_t n=0;while(std::getline(f,l)){if(n>=e.size()||l!=e[n]){std::cerr<<"expected "<<(n<e.size()?e[n]:"EOF")<<"\nactual   "<<l<<"\n";throw std::runtime_error("row "+std::to_string(n));}++n;}if(n!=e.size())throw std::runtime_error("missing rows");std::cout<<"PASS exact_consumers3 recursive_FnParamList actual_AST_and_TypeEntry typed_seams rows"<<n<<" clobber512\n";return 0;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

